"""
End-to-end training tests on tiny models with clean synthetic data.
These tests verify that models can actually learn, not just produce outputs.
Runs on CPU with small dimensions for speed.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


# Tiny model parameters for fast CPU testing
DIM = 16
N_LAYERS = 1
T = 8
B = 4
VOCAB_SIZE = 8
TRAIN_STEPS = 50
LR = 1e-3


def get_trainable_levels():
    """Get levels that can train on CPU."""
    from elman.models.stock_elman import StockElman
    from elman.models.gated_elman import GatedElman
    from elman.models.selective_elman import SelectiveElman
    from elman.models.diagonal_selective import DiagonalSelective
    from elman.models.logspace_polynomial import LogSpacePolynomial
    from elman.models.logspace_selective import LogSpaceSelective
    from elman.models.logspace_diagonal_selective import LogSpaceDiagonalSelective

    return [
        ('level_0', StockElman),
        ('level_1', GatedElman),
        ('level_2', SelectiveElman),
        ('level_3', DiagonalSelective),
        ('log_0', LogSpacePolynomial),
        ('log_1', LogSpaceSelective),
        ('log_2', LogSpaceDiagonalSelective),
    ]


class TinyLM(nn.Module):
    """Minimal language model for testing."""

    def __init__(self, rnn_class, dim, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.rnn = rnn_class(dim=dim, expansion=1.0)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        # x: [B, T] token ids
        h = self.embed(x)  # [B, T, dim]
        h, _ = self.rnn(h)  # [B, T, dim]
        return self.head(h)  # [B, T, vocab_size]


def generate_copy_data(batch_size, seq_len, vocab_size):
    """
    Generate copy task data.
    Input: [a, b, c, 0, 0, 0, ...]
    Target: [0, 0, 0, a, b, c, ...]

    The model must learn to copy the first half to the second half.
    """
    half = seq_len // 2
    # Generate random tokens for first half (excluding 0 which is padding)
    source = torch.randint(1, vocab_size, (batch_size, half))
    padding = torch.zeros(batch_size, half, dtype=torch.long)

    input_seq = torch.cat([source, padding], dim=1)
    target_seq = torch.cat([padding, source], dim=1)

    return input_seq, target_seq


def generate_repeat_data(batch_size, seq_len, vocab_size):
    """
    Generate repeating pattern data.
    Pattern: [a, b, a, b, a, b, ...]
    Target: predict next token (shifted by 1)

    Simple pattern that RNNs should easily learn.
    """
    pattern_len = 2
    pattern = torch.randint(0, vocab_size, (batch_size, pattern_len))
    repeats = (seq_len + pattern_len - 1) // pattern_len
    full = pattern.repeat(1, repeats)[:, :seq_len]

    input_seq = full[:, :-1]
    target_seq = full[:, 1:]

    return input_seq, target_seq


def train_loop(model, data_fn, steps, lr, vocab_size):
    """Run training loop and return final loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    losses = []
    for step in range(steps):
        optimizer.zero_grad()
        inputs, targets = data_fn()

        logits = model(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())

    return losses


@pytest.mark.parametrize("name,LayerClass", get_trainable_levels())
def test_loss_decreases_copy_task(name, LayerClass):
    """Test that loss decreases on copy task."""
    torch.manual_seed(42)

    model = TinyLM(LayerClass, DIM, VOCAB_SIZE)

    def data_fn():
        return generate_copy_data(B, T, VOCAB_SIZE)

    losses = train_loop(model, data_fn, TRAIN_STEPS, LR, VOCAB_SIZE)

    # Loss should decrease from start
    initial_loss = sum(losses[:5]) / 5
    final_loss = sum(losses[-5:]) / 5

    assert final_loss < initial_loss, \
        f"{name}: loss did not decrease on copy task ({initial_loss:.3f} -> {final_loss:.3f})"


@pytest.mark.parametrize("name,LayerClass", get_trainable_levels())
def test_loss_decreases_repeat_task(name, LayerClass):
    """Test that loss decreases on repeat pattern task."""
    torch.manual_seed(42)

    model = TinyLM(LayerClass, DIM, VOCAB_SIZE)

    def data_fn():
        return generate_repeat_data(B, T, VOCAB_SIZE)

    losses = train_loop(model, data_fn, TRAIN_STEPS, LR, VOCAB_SIZE)

    initial_loss = sum(losses[:5]) / 5
    final_loss = sum(losses[-5:]) / 5

    assert final_loss < initial_loss, \
        f"{name}: loss did not decrease on repeat task ({initial_loss:.3f} -> {final_loss:.3f})"


@pytest.mark.parametrize("name,LayerClass", get_trainable_levels())
def test_can_overfit_small_batch(name, LayerClass):
    """Test that model can overfit a single small batch."""
    torch.manual_seed(42)

    model = TinyLM(LayerClass, DIM, VOCAB_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # Single fixed batch
    inputs = torch.randint(0, VOCAB_SIZE, (2, 4))
    targets = torch.randint(0, VOCAB_SIZE, (2, 4))

    model.train()
    for _ in range(100):
        optimizer.zero_grad()
        logits = model(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            targets.reshape(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    final_loss = loss.item()

    # Should be able to drive loss very low on tiny batch
    assert final_loss < 1.0, \
        f"{name}: cannot overfit small batch (loss={final_loss:.3f})"


@pytest.mark.parametrize("name,LayerClass", get_trainable_levels())
def test_gradients_finite_during_training(name, LayerClass):
    """Test that gradients remain finite throughout training."""
    torch.manual_seed(42)

    model = TinyLM(LayerClass, DIM, VOCAB_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for step in range(20):
        optimizer.zero_grad()
        inputs, targets = generate_repeat_data(B, T, VOCAB_SIZE)

        logits = model(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            targets.reshape(-1)
        )

        loss.backward()

        # Check all gradients are finite
        for pname, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), \
                    f"{name}: NaN/Inf gradient in {pname} at step {step}"

        optimizer.step()


@pytest.mark.parametrize("name,LayerClass", get_trainable_levels())
def test_output_changes_with_training(name, LayerClass):
    """Test that model outputs actually change during training."""
    torch.manual_seed(42)

    model = TinyLM(LayerClass, DIM, VOCAB_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    inputs = torch.randint(0, VOCAB_SIZE, (2, 4))

    # Get initial output
    model.eval()
    with torch.no_grad():
        initial_output = model(inputs).clone()

    # Train for a few steps
    model.train()
    for _ in range(10):
        optimizer.zero_grad()
        targets = torch.randint(0, VOCAB_SIZE, (2, 4))
        loss = F.cross_entropy(
            model(inputs).reshape(-1, VOCAB_SIZE),
            targets.reshape(-1)
        )
        loss.backward()
        optimizer.step()

    # Get final output
    model.eval()
    with torch.no_grad():
        final_output = model(inputs)

    # Outputs should be different
    diff = (final_output - initial_output).abs().mean().item()
    assert diff > 0.001, \
        f"{name}: outputs did not change during training (diff={diff:.6f})"


class TestMathematicalProperties:
    """Test mathematical properties that all levels should satisfy."""

    @pytest.mark.parametrize("name,LayerClass", get_trainable_levels())
    def test_output_bounded(self, name, LayerClass):
        """Test that outputs have reasonable magnitude."""
        layer = LayerClass(dim=DIM, expansion=1.0)
        x = torch.randn(B, T, DIM)

        output, _ = layer(x)

        # Outputs should be reasonably bounded (not exploding)
        max_val = output.abs().max().item()
        assert max_val < 1000, f"{name}: output too large ({max_val:.1f})"

    @pytest.mark.parametrize("name,LayerClass", get_trainable_levels())
    def test_gradient_magnitude_reasonable(self, name, LayerClass):
        """Test that gradients have reasonable magnitude."""
        layer = LayerClass(dim=DIM, expansion=1.0)
        x = torch.randn(B, T, DIM, requires_grad=True)

        output, _ = layer(x)
        loss = output.sum()
        loss.backward()

        # Input gradient should be reasonably bounded
        grad_norm = x.grad.norm().item()
        assert grad_norm < 1e6, f"{name}: input gradient too large ({grad_norm:.1f})"

        # Parameter gradients should be reasonably bounded
        for pname, p in layer.named_parameters():
            if p.grad is not None:
                pgrad_norm = p.grad.norm().item()
                assert pgrad_norm < 1e9, \
                    f"{name}: gradient for {pname} too large ({pgrad_norm:.1f})"
