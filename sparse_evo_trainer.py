"""
Sparse Evolutionary Training for E88

Key insight: Don't evolve all parameters. Track which heads/layers contribute,
and evolve sparse subsets based on "energy" (contribution to loss reduction).

- High energy heads: working well → small mutations (exploitation)
- Low energy heads: not contributing → large mutations or death (exploration)

This is like attention over parameters: compute a "fitness gradient" that
allocates mutation budget to where it matters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import copy


@dataclass
class HeadStats:
    """Track statistics for a single head."""
    layer: int
    head: int
    # Contribution metrics
    activation_variance: float = 0.0  # How much does output vary?
    gradient_magnitude: float = 0.0   # How strong are gradients?
    ablation_delta: float = 0.0       # How much does removing hurt?
    # Derived energy
    energy: float = 1.0
    # Evolution tracking
    mutations: int = 0
    last_mutated: int = 0


class SparseEvoTracker(nn.Module):
    """Tracks per-head energy and decides which parameters to evolve."""

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        energy_momentum: float = 0.9,
        energy_method: str = 'gradient',  # 'gradient', 'variance', 'ablation'
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.energy_momentum = energy_momentum
        self.energy_method = energy_method

        # Per-head energy tracking (high = good, low = needs exploration)
        # Shape: [n_layers, n_heads]
        self.register_buffer(
            'head_energy',
            torch.ones(n_layers, n_heads)
        )

        # Track mutation history
        self.register_buffer(
            'mutation_count',
            torch.zeros(n_layers, n_heads, dtype=torch.long)
        )

        # Track recent contribution (for energy calculation)
        self.register_buffer(
            'recent_contribution',
            torch.zeros(n_layers, n_heads)
        )

        # Step counter
        self.step = 0

    def update_energy_from_gradients(
        self,
        model: nn.Module,
        layer_indices: Optional[List[int]] = None
    ):
        """Update head energy based on gradient magnitudes.

        Heads with high gradients are "active" - contributing to learning.
        Heads with low gradients are "dormant" - candidates for mutation.
        """
        if layer_indices is None:
            layer_indices = range(self.n_layers)

        for layer_idx in layer_indices:
            # Get the E88 layer
            layer = self._get_e88_layer(model, layer_idx)
            if layer is None:
                continue

            # Measure gradient magnitude for head-specific parameters
            # A_log has shape [n_heads] - one value per head
            if hasattr(layer, 'A_log') and layer.A_log.grad is not None:
                grad_mag = layer.A_log.grad.abs()  # [n_heads]
            elif hasattr(layer, 'dt_bias') and layer.dt_bias.grad is not None:
                grad_mag = layer.dt_bias.grad.abs()  # [n_heads]
            else:
                # Fall back to projection gradients if available
                continue

            # Normalize to [0, 1] range
            if grad_mag.max() > 0:
                grad_mag = grad_mag / (grad_mag.max() + 1e-8)

            # Move to same device as buffers
            grad_mag = grad_mag.to(self.head_energy.device)

            # Update energy with momentum
            self.recent_contribution[layer_idx] = grad_mag
            self.head_energy[layer_idx] = (
                self.energy_momentum * self.head_energy[layer_idx] +
                (1 - self.energy_momentum) * grad_mag
            )

    def update_energy_from_activations(
        self,
        layer_outputs: Dict[int, torch.Tensor]
    ):
        """Update head energy based on output variance.

        Heads with high variance are "expressive" - capturing diverse patterns.
        Heads with low variance are "collapsed" - candidates for mutation.

        Args:
            layer_outputs: Dict mapping layer_idx to output tensor [B, T, H, head_v_dim]
        """
        for layer_idx, output in layer_outputs.items():
            if layer_idx >= self.n_layers:
                continue

            # Compute variance per head: [H]
            # output: [B, T, H, head_v_dim]
            head_var = output.var(dim=(0, 1, 3))  # Variance over batch, time, features

            # Normalize
            if head_var.max() > 0:
                head_var = head_var / (head_var.max() + 1e-8)

            # Update with momentum
            self.recent_contribution[layer_idx, :head_var.size(0)] = head_var
            self.head_energy[layer_idx, :head_var.size(0)] = (
                self.energy_momentum * self.head_energy[layer_idx, :head_var.size(0)] +
                (1 - self.energy_momentum) * head_var
            )

    def get_mutation_probabilities(
        self,
        base_prob: float = 0.1,
        energy_scale: float = 2.0,
    ) -> torch.Tensor:
        """Get per-head mutation probabilities.

        Low energy → high mutation probability (exploration)
        High energy → low mutation probability (exploitation)

        Returns:
            probs: [n_layers, n_heads] mutation probabilities
        """
        # Inverse energy scaling: low energy = high mutation
        # Add small epsilon to avoid division issues
        inv_energy = 1.0 / (self.head_energy + 0.1)

        # Normalize to [0, 1]
        inv_energy = inv_energy / inv_energy.max()

        # Scale by base probability
        probs = base_prob * (1 + energy_scale * inv_energy)

        # Clip to valid range
        return probs.clamp(0, 1)

    def select_heads_to_mutate(
        self,
        n_mutations: int,
        temperature: float = 1.0,
    ) -> List[Tuple[int, int]]:
        """Select which heads to mutate this step.

        Uses energy-weighted sampling: low energy heads more likely.

        Args:
            n_mutations: Number of heads to mutate
            temperature: Higher = more random, lower = more energy-focused

        Returns:
            List of (layer_idx, head_idx) tuples
        """
        # Get probabilities
        probs = self.get_mutation_probabilities()

        # Apply temperature
        if temperature != 1.0:
            probs = probs.pow(1.0 / temperature)
            probs = probs / probs.sum()

        # Flatten and sample
        flat_probs = probs.view(-1)
        flat_probs = flat_probs / flat_probs.sum()  # Normalize

        # Sample without replacement
        indices = torch.multinomial(
            flat_probs,
            min(n_mutations, flat_probs.numel()),
            replacement=False
        )

        # Convert to (layer, head) tuples
        selected = []
        for idx in indices:
            layer_idx = idx.item() // self.n_heads
            head_idx = idx.item() % self.n_heads
            selected.append((layer_idx, head_idx))
            self.mutation_count[layer_idx, head_idx] += 1

        self.step += 1
        return selected

    def _get_e88_layer(self, model: nn.Module, layer_idx: int):
        """Get E88 layer from model by index."""
        # Handle different model structures
        if hasattr(model, 'layers'):
            if layer_idx < len(model.layers):
                layer = model.layers[layer_idx]
                if hasattr(layer, 'mixer'):
                    return layer.mixer
                return layer
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            if layer_idx < len(model.model.layers):
                layer = model.model.layers[layer_idx]
                if hasattr(layer, 'mixer'):
                    return layer.mixer
                return layer
        return None

    def get_energy_summary(self) -> dict:
        """Get summary statistics about head energy distribution."""
        return {
            'mean_energy': self.head_energy.mean().item(),
            'std_energy': self.head_energy.std().item(),
            'min_energy': self.head_energy.min().item(),
            'max_energy': self.head_energy.max().item(),
            'dormant_heads': (self.head_energy < 0.1).sum().item(),
            'active_heads': (self.head_energy > 0.5).sum().item(),
            'total_mutations': self.mutation_count.sum().item(),
        }


class SparseEvolver:
    """Performs sparse evolutionary mutations on E88 models."""

    def __init__(
        self,
        model: nn.Module,
        tracker: SparseEvoTracker,
        mutation_scale: float = 0.1,
        mutation_type: str = 'gaussian',  # 'gaussian', 'uniform', 'cauchy'
        elite_fraction: float = 0.1,  # Top 10% heads protected from mutation
    ):
        self.model = model
        self.tracker = tracker
        self.mutation_scale = mutation_scale
        self.mutation_type = mutation_type
        self.elite_fraction = elite_fraction

        # Store original parameter values for rollback
        self.checkpoint = None

    def save_checkpoint(self):
        """Save current parameters for potential rollback."""
        self.checkpoint = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }

    def restore_checkpoint(self):
        """Restore parameters from checkpoint."""
        if self.checkpoint is None:
            return
        for name, param in self.model.named_parameters():
            if name in self.checkpoint:
                param.data.copy_(self.checkpoint[name])

    def mutate_heads(
        self,
        heads_to_mutate: List[Tuple[int, int]],
        scale_by_energy: bool = True,
    ):
        """Apply mutations to selected heads.

        Args:
            heads_to_mutate: List of (layer_idx, head_idx) tuples
            scale_by_energy: If True, low-energy heads get larger mutations
        """
        for layer_idx, head_idx in heads_to_mutate:
            layer = self.tracker._get_e88_layer(self.model, layer_idx)
            if layer is None:
                continue

            # Get mutation scale (larger for low-energy heads)
            if scale_by_energy:
                energy = self.tracker.head_energy[layer_idx, head_idx].item()
                # Low energy → scale up to 2x, high energy → scale down to 0.5x
                adaptive_scale = self.mutation_scale * (2.0 - energy)
            else:
                adaptive_scale = self.mutation_scale

            # Mutate head-specific parameters
            self._mutate_head_params(layer, head_idx, adaptive_scale)

    def _mutate_head_params(
        self,
        layer: nn.Module,
        head_idx: int,
        scale: float,
    ):
        """Mutate parameters for a single head."""
        # Mutate A_log (decay eigenvalue) - this controls memory retention
        if hasattr(layer, 'A_log') and layer.A_log is not None:
            with torch.no_grad():
                noise = self._sample_noise(1, layer.A_log.device, layer.A_log.dtype)
                layer.A_log.data[head_idx] += scale * noise.item()

        # Mutate dt_bias (time-step bias) - this controls update rate
        if hasattr(layer, 'dt_bias') and layer.dt_bias is not None:
            with torch.no_grad():
                noise = self._sample_noise(1, layer.dt_bias.device, layer.dt_bias.dtype)
                layer.dt_bias.data[head_idx] += scale * noise.item()

        # Optionally mutate a_proj weights for this head
        # a_proj maps to [n_heads], so we can perturb the corresponding row
        if hasattr(layer, 'a_proj') and layer.a_proj is not None:
            with torch.no_grad():
                noise = self._sample_noise(
                    layer.a_proj.weight[head_idx].shape,
                    layer.a_proj.weight.device,
                    layer.a_proj.weight.dtype
                )
                layer.a_proj.weight.data[head_idx] += scale * 0.01 * noise  # Smaller scale for weights

    def _sample_noise(
        self,
        shape,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Sample noise for mutation."""
        if self.mutation_type == 'gaussian':
            return torch.randn(shape, device=device, dtype=dtype)
        elif self.mutation_type == 'uniform':
            return torch.rand(shape, device=device, dtype=dtype) * 2 - 1
        elif self.mutation_type == 'cauchy':
            # Cauchy has heavier tails - allows larger jumps
            return torch.distributions.Cauchy(0, 1).sample(shape).to(device=device, dtype=dtype)
        else:
            raise ValueError(f"Unknown mutation type: {self.mutation_type}")

    def evolve_step(
        self,
        loss_before: float,
        loss_after: float,
        heads_mutated: List[Tuple[int, int]],
        accept_threshold: float = 0.0,
    ) -> bool:
        """Accept or reject mutations based on loss change.

        Args:
            loss_before: Loss before mutations
            loss_after: Loss after mutations and a few gradient steps
            heads_mutated: Which heads were mutated
            accept_threshold: Accept if loss_after < loss_before + threshold

        Returns:
            True if mutations accepted, False if rolled back
        """
        improvement = loss_before - loss_after

        if improvement > -accept_threshold:
            # Accept - update energy for mutated heads
            # Successful mutations → boost energy
            for layer_idx, head_idx in heads_mutated:
                bonus = 0.1 if improvement > 0 else 0.0
                self.tracker.head_energy[layer_idx, head_idx] += bonus
            return True
        else:
            # Reject - rollback
            self.restore_checkpoint()
            # Failed mutations → reduce energy (more exploration needed)
            for layer_idx, head_idx in heads_mutated:
                self.tracker.head_energy[layer_idx, head_idx] *= 0.9
            return False


class SparseEvoTrainer:
    """Combined gradient + sparse evolution training loop."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        n_layers: int,
        n_heads: int,
        # Evolution params
        evo_every: int = 100,        # Evolve every N gradient steps
        n_mutations: int = 10,        # How many heads to mutate per evolution
        mutation_scale: float = 0.1,  # Base mutation magnitude
        evo_warmup: int = 500,        # Gradient-only warmup steps
        # Gradient steps between evolution eval
        evo_eval_steps: int = 20,
    ):
        self.model = model
        self.optimizer = optimizer

        self.tracker = SparseEvoTracker(n_layers, n_heads)
        self.evolver = SparseEvolver(model, self.tracker, mutation_scale)

        self.evo_every = evo_every
        self.n_mutations = n_mutations
        self.evo_warmup = evo_warmup
        self.evo_eval_steps = evo_eval_steps

        self.step = 0
        self.evo_accepts = 0
        self.evo_rejects = 0

    def train_step(
        self,
        batch,
        compute_loss_fn,
    ) -> dict:
        """Single training step with potential evolution.

        Args:
            batch: Input batch
            compute_loss_fn: Function that takes (model, batch) -> loss

        Returns:
            Dict with loss and evolution stats
        """
        stats = {}

        # Standard gradient step
        self.optimizer.zero_grad()
        loss = compute_loss_fn(self.model, batch)
        loss.backward()

        # Update energy from gradients
        self.tracker.update_energy_from_gradients(self.model)

        self.optimizer.step()
        stats['loss'] = loss.item()

        # Check if time for evolution
        if (self.step >= self.evo_warmup and
            self.step % self.evo_every == 0):

            evo_stats = self._evolution_step(batch, compute_loss_fn)
            stats.update(evo_stats)

        self.step += 1
        return stats

    def _evolution_step(
        self,
        batch,
        compute_loss_fn,
    ) -> dict:
        """Perform one evolution step."""
        stats = {'evo_step': True}

        # Record loss before mutations
        with torch.no_grad():
            loss_before = compute_loss_fn(self.model, batch).item()
        stats['loss_before_evo'] = loss_before

        # Save checkpoint for rollback
        self.evolver.save_checkpoint()

        # Select and mutate heads
        heads = self.tracker.select_heads_to_mutate(self.n_mutations)
        self.evolver.mutate_heads(heads)
        stats['heads_mutated'] = len(heads)

        # Do a few gradient steps to let mutations settle
        for _ in range(self.evo_eval_steps):
            self.optimizer.zero_grad()
            loss = compute_loss_fn(self.model, batch)
            loss.backward()
            self.optimizer.step()

        # Evaluate after mutations
        with torch.no_grad():
            loss_after = compute_loss_fn(self.model, batch).item()
        stats['loss_after_evo'] = loss_after

        # Accept or reject
        accepted = self.evolver.evolve_step(loss_before, loss_after, heads)
        stats['evo_accepted'] = accepted

        if accepted:
            self.evo_accepts += 1
        else:
            self.evo_rejects += 1

        stats['evo_accept_rate'] = self.evo_accepts / max(1, self.evo_accepts + self.evo_rejects)
        stats.update(self.tracker.get_energy_summary())

        return stats


def test_sparse_evo():
    """Test the sparse evolutionary trainer."""
    print("Testing Sparse Evolutionary Trainer...")

    # Create a simple test model
    from elman.models.e88_fla_hybrid import E88FLAHybrid

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32

    dim = 256
    n_heads = 8
    n_state = 16
    n_layers = 4

    # Build a simple multi-layer model
    class SimpleE88Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(1000, dim)
            self.layers = nn.ModuleList([
                E88FLAHybrid(dim, n_heads=n_heads, n_state=n_state)
                for _ in range(n_layers)
            ])
            self.head = nn.Linear(dim, 1000)

        def forward(self, x):
            x = self.embed(x)
            for layer in self.layers:
                out, _ = layer(x)
                x = x + out
            return self.head(x)

    model = SimpleE88Model().to(device).to(dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Create trainer
    trainer = SparseEvoTrainer(
        model=model,
        optimizer=optimizer,
        n_layers=n_layers,
        n_heads=n_heads,
        evo_every=10,       # Evolve frequently for testing
        n_mutations=4,      # Mutate 4 heads at a time
        mutation_scale=0.05,
        evo_warmup=5,       # Short warmup for testing
        evo_eval_steps=3,
    )
    # Move tracker to same device
    trainer.tracker = trainer.tracker.to(device)

    # Define loss function
    def compute_loss(model, batch):
        x, y = batch
        logits = model(x)
        return F.cross_entropy(logits.view(-1, 1000), y.view(-1))

    # Training loop
    print(f"\nTraining for 50 steps...")
    for i in range(50):
        # Random batch
        x = torch.randint(0, 1000, (4, 32), device=device)
        y = torch.randint(0, 1000, (4, 32), device=device)
        batch = (x, y)

        stats = trainer.train_step(batch, compute_loss)

        if i % 10 == 0 or stats.get('evo_step', False):
            print(f"Step {i}: loss={stats['loss']:.4f}", end="")
            if stats.get('evo_step'):
                print(f" | EVO: mutated={stats['heads_mutated']}, "
                      f"accepted={stats['evo_accepted']}, "
                      f"rate={stats['evo_accept_rate']:.2f}", end="")
            print()

    # Print final energy distribution
    print("\nFinal energy distribution:")
    summary = trainer.tracker.get_energy_summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print("\nPer-layer energy:")
    for layer_idx in range(n_layers):
        energies = trainer.tracker.head_energy[layer_idx].tolist()
        print(f"  Layer {layer_idx}: {[f'{e:.2f}' for e in energies]}")


if __name__ == "__main__":
    test_sparse_evo()
