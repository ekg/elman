#!/usr/bin/env python3
"""
Self-Gate Activation Distribution Analysis

Analyzes what h values the trained E42 model produces,
and how self_gate(h) = h² * sigmoid(h) transforms them.

The hypothesis is that self-gate compensates for the small W matrices
by amplifying important (high-magnitude) activations.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

# Add elman to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def silu(x):
    return x * torch.sigmoid(x)


def self_gate(h):
    """h * silu(h) = h² * sigmoid(h)"""
    return h * silu(h)


def analyze_activations(checkpoint_path, data_path='data/pile.txt', num_batches=10):
    """Run model on data and collect activation statistics."""
    from elman.models import LadderLM

    print(f"\nLoading model from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cuda', weights_only=False)

    # Figure out model config from checkpoint
    state = ckpt['model_state_dict']
    dim = state['embedding.weight'].shape[1]
    vocab_size = state['embedding.weight'].shape[0]
    depth = sum(1 for k in state if 'cell.W' in k)

    print(f"Model: dim={dim}, depth={depth}, vocab={vocab_size}")

    # Create model
    model = LadderLM(
        vocab_size=vocab_size,
        dim=dim,
        depth=depth,
        level=42,  # E42
    ).cuda()
    model.load_state_dict(state)
    model.eval()

    # Load some data
    import mmap
    with open(data_path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    batch_size = 4
    seq_len = 512

    all_h_values = []
    all_gate_values = []

    print(f"\nCollecting activations from {num_batches} batches...")

    with torch.no_grad():
        for batch_idx in range(num_batches):
            # Random positions
            positions = np.random.randint(0, len(mm) - seq_len - 1, size=batch_size)
            x = torch.zeros(batch_size, seq_len, dtype=torch.long)
            for i, pos in enumerate(positions):
                x[i] = torch.tensor(list(mm[pos:pos+seq_len]), dtype=torch.long)
            x = x.cuda()

            # Run through model and collect intermediate activations
            # We need to hook into the layers to get h values

            h_samples = []
            gate_samples = []

            def hook_fn(module, input, output):
                # output is (gated_output, hidden_states)
                if isinstance(output, tuple) and len(output) == 2:
                    gated, hidden = output
                    # hidden is [T+1, B, D], take last hidden state
                    h = hidden[-1]  # [B, D]
                    h_samples.append(h.cpu())
                    gate_samples.append(self_gate(h).cpu())

            # Register hooks on E42 cells
            hooks = []
            for i, layer in enumerate(model.layers):
                if hasattr(layer, 'cell'):
                    hooks.append(layer.cell.register_forward_hook(hook_fn))

            # Forward pass
            _ = model(x)

            # Remove hooks
            for h in hooks:
                h.remove()

            if h_samples:
                all_h_values.extend([s.numpy().flatten() for s in h_samples])
                all_gate_values.extend([s.numpy().flatten() for s in gate_samples])

            if (batch_idx + 1) % 5 == 0:
                print(f"  Batch {batch_idx + 1}/{num_batches}")

    mm.close()

    if not all_h_values:
        print("ERROR: No activations collected!")
        return

    # Concatenate all samples
    h_all = np.concatenate(all_h_values)
    gate_all = np.concatenate(all_gate_values)

    print(f"\nCollected {len(h_all):,} activation samples")

    # Statistics
    print("\n" + "="*60)
    print("Hidden State (h) Statistics")
    print("="*60)
    print(f"Mean:   {h_all.mean():.4f}")
    print(f"Std:    {h_all.std():.4f}")
    print(f"Min:    {h_all.min():.4f}")
    print(f"Max:    {h_all.max():.4f}")
    print(f"Median: {np.median(h_all):.4f}")

    # Distribution
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    print(f"\nPercentiles:")
    for p in percentiles:
        val = np.percentile(h_all, p)
        print(f"  {p}%:  {val:.4f}")

    # Histogram bins
    bins = [-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5]
    hist, _ = np.histogram(h_all, bins=bins)
    print(f"\nHistogram of h values:")
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        pct = hist[i] / len(h_all) * 100
        bar = '#' * int(pct / 2)
        print(f"  [{lo:5.1f}, {hi:5.1f}): {pct:5.1f}% {bar}")

    print("\n" + "="*60)
    print("Self-Gate Output Statistics")
    print("="*60)
    print(f"Mean:   {gate_all.mean():.4f}")
    print(f"Std:    {gate_all.std():.4f}")
    print(f"Min:    {gate_all.min():.4f}")
    print(f"Max:    {gate_all.max():.4f}")

    # Self-gate amplification analysis
    print("\n" + "="*60)
    print("Self-Gate Amplification Analysis")
    print("="*60)

    # For different h magnitudes, compute average amplification
    bins = [0, 0.5, 1, 2, 3, 5]
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (np.abs(h_all) >= lo) & (np.abs(h_all) < hi)
        if mask.sum() > 0:
            h_sub = h_all[mask]
            gate_sub = gate_all[mask]
            # Amplification = |gate| / |h|
            amp = np.abs(gate_sub) / (np.abs(h_sub) + 1e-8)
            print(f"  |h| in [{lo}, {hi}): mean amp = {amp.mean():.2f}x, samples = {mask.sum():,}")

    # How many activations get amplified vs suppressed?
    print("\nAmplification vs Suppression:")
    amp_ratio = np.abs(gate_all) / (np.abs(h_all) + 1e-8)
    print(f"  Amplified (>1x):    {(amp_ratio > 1).sum() / len(amp_ratio) * 100:.1f}%")
    print(f"  Suppressed (<1x):   {(amp_ratio < 1).sum() / len(amp_ratio) * 100:.1f}%")
    print(f"  Strong amp (>2x):   {(amp_ratio > 2).sum() / len(amp_ratio) * 100:.1f}%")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data', type=str, default='data/pile.txt')
    parser.add_argument('--batches', type=int, default=10)
    args = parser.parse_args()

    analyze_activations(args.checkpoint, args.data, args.batches)
