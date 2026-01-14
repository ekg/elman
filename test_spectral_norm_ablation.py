#!/usr/bin/env python3
"""
Quick A/B test: E1 with vs without spectral normalization.

Spectral norm constrains W_h to have spectral radius < 0.99.
For nonlinear models with tanh, this may not be necessary since tanh bounds outputs.

Usage:
    CUDA_VISIBLE_DEVICES=5 python test_spectral_norm_ablation.py
"""

import os
import sys
import time

# Add CUDA path BEFORE importing torch/elman models
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'elman/cuda'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import mmap
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from elman.models.ladder_lm import LadderLM


def load_batch_mmap(mm, batch_size, seq_len, device):
    """Load a random batch from memory-mapped file."""
    max_pos = len(mm) - seq_len - 2
    positions = np.random.randint(0, max_pos, size=batch_size)

    buf = np.zeros((batch_size, seq_len + 1), dtype=np.uint8)
    for i, pos in enumerate(positions):
        buf[i] = np.frombuffer(mm[pos:pos + seq_len + 1], dtype=np.uint8)

    return torch.from_numpy(buf.astype(np.int64)).to(device)


def train_model(model, mm, batch_size, seq_len, train_minutes, device, label):
    """Train a model for a fixed time and return final loss."""
    model = model.to(device).bfloat16()
    model.train()

    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95))

    start_time = time.time()
    end_time = start_time + train_minutes * 60

    step = 0
    total_tokens = 0
    losses = []

    print(f"\n{'='*60}")
    print(f"Training: {label}")
    print(f"Parameters: {model.get_num_params():,}")
    print(f"{'='*60}")

    while time.time() < end_time:
        # Load batch
        batch = load_batch_mmap(mm, batch_size, seq_len, device)

        # Forward pass
        loss = model(batch, return_loss=True)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Track
        step += 1
        total_tokens += batch_size * seq_len
        losses.append(loss.item())

        # Log every 50 steps
        if step % 50 == 0:
            elapsed = time.time() - start_time
            tok_per_sec = total_tokens / elapsed
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            print(f"Step {step:4d} | Loss: {avg_loss:.4f} | {tok_per_sec/1000:.1f}K tok/s")

    # Final stats
    elapsed = time.time() - start_time
    final_throughput = total_tokens / elapsed
    final_loss = sum(losses[-100:]) / len(losses[-100:])  # Last 100 avg

    return final_loss, final_throughput, step


def main():
    # Config
    data_path = "data/pile.txt"
    batch_size = 128
    seq_len = 512
    train_minutes = 5.0
    seed = 42

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Memory map data
    with open(data_path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    print(f"Data: {len(mm) / 1e9:.2f} GB")

    results = {}

    # Test 1: E1 WITH spectral norm (default)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model_with = LadderLM(
        vocab_size=256,
        dim=1024,
        depth=6,
        level=1,
        r_h_mode='spectral_norm',  # Default
    )
    loss_with, throughput_with, steps_with = train_model(
        model_with, mm, batch_size, seq_len, train_minutes, device,
        "E1 WITH spectral_norm (r_h_mode='spectral_norm')"
    )
    results['with_spectral'] = {
        'loss': loss_with,
        'throughput': throughput_with,
        'steps': steps_with,
    }
    del model_with
    torch.cuda.empty_cache()

    # Test 2: E1 WITHOUT spectral norm
    torch.manual_seed(seed)
    np.random.seed(seed)
    model_without = LadderLM(
        vocab_size=256,
        dim=1024,
        depth=6,
        level=1,
        r_h_mode='none',  # Disable spectral norm
    )
    loss_without, throughput_without, steps_without = train_model(
        model_without, mm, batch_size, seq_len, train_minutes, device,
        "E1 WITHOUT spectral_norm (r_h_mode='none')"
    )
    results['without_spectral'] = {
        'loss': loss_without,
        'throughput': throughput_without,
        'steps': steps_without,
    }
    del model_without
    torch.cuda.empty_cache()

    # Test 3: E33 WITH spectral norm (if we have time)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model_e33_with = LadderLM(
        vocab_size=256,
        dim=1024,
        depth=6,
        level=33,
        r_h_mode='spectral_norm',
    )
    loss_e33_with, throughput_e33_with, steps_e33_with = train_model(
        model_e33_with, mm, batch_size, seq_len, train_minutes, device,
        "E33 WITH spectral_norm"
    )
    results['e33_with_spectral'] = {
        'loss': loss_e33_with,
        'throughput': throughput_e33_with,
        'steps': steps_e33_with,
    }
    del model_e33_with
    torch.cuda.empty_cache()

    # Test 4: E33 WITHOUT spectral norm
    torch.manual_seed(seed)
    np.random.seed(seed)
    model_e33_without = LadderLM(
        vocab_size=256,
        dim=1024,
        depth=6,
        level=33,
        r_h_mode='none',
    )
    loss_e33_without, throughput_e33_without, steps_e33_without = train_model(
        model_e33_without, mm, batch_size, seq_len, train_minutes, device,
        "E33 WITHOUT spectral_norm"
    )
    results['e33_without_spectral'] = {
        'loss': loss_e33_without,
        'throughput': throughput_e33_without,
        'steps': steps_e33_without,
    }

    mm.close()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Spectral Normalization Ablation (5 min training each)")
    print("="*70)
    print(f"{'Model':<30} {'Loss':>10} {'Throughput':>15} {'Steps':>10}")
    print("-"*70)
    print(f"{'E1 + spectral_norm':<30} {results['with_spectral']['loss']:>10.4f} {results['with_spectral']['throughput']/1000:>12.1f}K/s {results['with_spectral']['steps']:>10}")
    print(f"{'E1 - spectral_norm':<30} {results['without_spectral']['loss']:>10.4f} {results['without_spectral']['throughput']/1000:>12.1f}K/s {results['without_spectral']['steps']:>10}")
    print(f"{'E33 + spectral_norm':<30} {results['e33_with_spectral']['loss']:>10.4f} {results['e33_with_spectral']['throughput']/1000:>12.1f}K/s {results['e33_with_spectral']['steps']:>10}")
    print(f"{'E33 - spectral_norm':<30} {results['e33_without_spectral']['loss']:>10.4f} {results['e33_without_spectral']['throughput']/1000:>12.1f}K/s {results['e33_without_spectral']['steps']:>10}")
    print("-"*70)

    # Analysis
    e1_loss_diff = results['without_spectral']['loss'] - results['with_spectral']['loss']
    e1_speed_diff = (results['without_spectral']['throughput'] - results['with_spectral']['throughput']) / results['with_spectral']['throughput'] * 100
    e33_loss_diff = results['e33_without_spectral']['loss'] - results['e33_with_spectral']['loss']
    e33_speed_diff = (results['e33_without_spectral']['throughput'] - results['e33_with_spectral']['throughput']) / results['e33_with_spectral']['throughput'] * 100

    print(f"\nE1 impact of removing spectral norm:")
    print(f"  Loss change: {e1_loss_diff:+.4f} ({'worse' if e1_loss_diff > 0 else 'better'})")
    print(f"  Speed change: {e1_speed_diff:+.1f}%")

    print(f"\nE33 impact of removing spectral norm:")
    print(f"  Loss change: {e33_loss_diff:+.4f} ({'worse' if e33_loss_diff > 0 else 'better'})")
    print(f"  Speed change: {e33_speed_diff:+.1f}%")


if __name__ == "__main__":
    main()
