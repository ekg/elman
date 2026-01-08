#!/usr/bin/env python3
"""
Quick comparison of E1 (tanh) vs E15 (softsign) activation functions.

Usage:
    python compare_activations.py --gpu 0 --minutes 5
    python compare_activations.py --gpu 0 --minutes 10
"""

import argparse
import os
import sys
import time
import torch
import torch.nn.functional as F
from pathlib import Path
import mmap
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from elman.models import LadderLM


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    parser.add_argument("--minutes", type=float, default=5.0, help="Training time in minutes")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    parser.add_argument("--dim", type=int, default=1280, help="Model dimension")
    parser.add_argument("--depth", type=int, default=6, help="Number of layers")
    parser.add_argument("--data", type=str, default="/home/erikg/elman/data/pile.txt", help="Data file")
    return parser.parse_args()


def load_mmap(path):
    """Load data file as mmap."""
    with open(path, 'rb') as f:
        return mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)


def get_batch(mm, batch_size, seq_len):
    """Get a random batch from mmap."""
    max_start = len(mm) - seq_len - 1
    starts = np.random.randint(0, max_start, size=batch_size)

    buf = np.zeros((batch_size, seq_len + 1), dtype=np.uint8)
    for i, start in enumerate(starts):
        buf[i] = np.frombuffer(mm[start:start + seq_len + 1], dtype=np.uint8)

    x = torch.from_numpy(buf[:, :-1].astype(np.int64))
    y = torch.from_numpy(buf[:, 1:].astype(np.int64))
    return x, y


def train_model(model, mm, args, max_time_seconds):
    """Train model and return final loss."""
    device = f'cuda:{args.gpu}'
    model = model.to(device).bfloat16()

    # Count params
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params/1e6:.1f}M")

    # Optimizer
    from schedulefree import AdamWScheduleFree
    optimizer = AdamWScheduleFree(model.parameters(), lr=3e-4, weight_decay=0.1)
    optimizer.train()

    # Training loop
    start_time = time.time()
    step = 0
    total_tokens = 0
    loss_sum = 0.0
    loss_count = 0
    recent_losses = []

    while time.time() - start_time < max_time_seconds:
        x, y = get_batch(mm, args.batch_size, args.seq_len)
        x, y = x.to(device), y.to(device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, 256), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step += 1
        total_tokens += args.batch_size * args.seq_len
        loss_val = loss.item()
        loss_sum += loss_val
        loss_count += 1
        recent_losses.append(loss_val)
        if len(recent_losses) > 100:
            recent_losses.pop(0)

        if step % 50 == 0:
            elapsed = time.time() - start_time
            avg_loss = sum(recent_losses) / len(recent_losses)
            tok_per_sec = total_tokens / elapsed
            print(f"  Step {step}: loss={avg_loss:.4f}, tok/s={tok_per_sec/1000:.1f}K")

    # Final stats
    elapsed = time.time() - start_time
    final_loss = sum(recent_losses) / len(recent_losses)
    tok_per_sec = total_tokens / elapsed

    return final_loss, tok_per_sec, params


def main():
    args = get_args()

    print("=" * 60)
    print("Activation Function Comparison: E1 (tanh) vs E15 (softsign)")
    print("=" * 60)
    print(f"Config: dim={args.dim}, depth={args.depth}, batch={args.batch_size}, seq={args.seq_len}")
    print(f"Training time: {args.minutes} minutes per model")
    print()

    # Load data
    print("Loading data...")
    mm = load_mmap(args.data)
    print(f"Data size: {len(mm)/1e9:.2f}GB")
    print()

    max_time = args.minutes * 60
    results = {}

    # Test E1 (tanh)
    print("-" * 40)
    print("Testing E1 (tanh)...")
    print("-" * 40)
    model_e1 = LadderLM(
        vocab_size=256,
        dim=args.dim,
        depth=args.depth,
        level=1,
        expansion=1.0,
        mamba2_init=True
    )
    loss_e1, tps_e1, params_e1 = train_model(model_e1, mm, args, max_time)
    results['E1 (tanh)'] = {'loss': loss_e1, 'tok/s': tps_e1, 'params': params_e1}
    del model_e1
    torch.cuda.empty_cache()
    print()

    # Test E15 (softsign)
    print("-" * 40)
    print("Testing E15 (softsign)...")
    print("-" * 40)
    model_e15 = LadderLM(
        vocab_size=256,
        dim=args.dim,
        depth=args.depth,
        level=15,
        expansion=1.0,
        mamba2_init=True
    )
    loss_e15, tps_e15, params_e15 = train_model(model_e15, mm, args, max_time)
    results['E15 (softsign)'] = {'loss': loss_e15, 'tok/s': tps_e15, 'params': params_e15}
    del model_e15
    torch.cuda.empty_cache()
    print()

    # Summary
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Model':<20} {'Loss':>10} {'Tok/s':>12} {'Params':>12}")
    print("-" * 54)
    for name, r in results.items():
        print(f"{name:<20} {r['loss']:>10.4f} {r['tok/s']/1000:>10.1f}K {r['params']/1e6:>10.1f}M")
    print()

    # Comparison
    diff = results['E15 (softsign)']['loss'] - results['E1 (tanh)']['loss']
    speedup = results['E15 (softsign)']['tok/s'] / results['E1 (tanh)']['tok/s']
    print(f"Loss difference: {diff:+.4f} (negative = softsign better)")
    print(f"Speedup: {speedup:.2f}x (>1 = softsign faster)")


if __name__ == "__main__":
    main()
