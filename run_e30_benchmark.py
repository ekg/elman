#!/usr/bin/env python3
"""
Benchmark E30 (E1 + diagonal gating) vs E1.

E30 adds SSM-style diagonal gating with minimal overhead:
  gate = silu(z * g_z + h * g_h + b_gate)
  output = h * gate

Extra params: 3*d_inner per layer (negligible)
"""

import os
import sys
import time
import argparse
import mmap
import numpy as np
import torch
import torch.nn.functional as F

from elman.models import LadderLM


def get_batch(mm, batch_size, seq_len, device):
    """Sample random batches from memory-mapped file."""
    max_pos = len(mm) - seq_len - 1
    positions = np.random.randint(0, max_pos, size=batch_size)

    buf = np.zeros((batch_size, seq_len + 1), dtype=np.uint8)
    for i, pos in enumerate(positions):
        buf[i] = np.frombuffer(mm[pos:pos + seq_len + 1], dtype=np.uint8)

    x = torch.from_numpy(buf[:, :-1]).long().to(device)
    y = torch.from_numpy(buf[:, 1:]).long().to(device)
    return x, y


def train_model(model, mm, batch_size, seq_len, steps, device, warmup=50):
    """Train for fixed steps, return final loss and throughput."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    model.train()
    losses = []
    start_time = None
    total_tokens = 0

    for step in range(steps):
        x, y = get_batch(mm, batch_size, seq_len, device)

        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step >= warmup:
            if start_time is None:
                start_time = time.time()
                total_tokens = 0
            total_tokens += batch_size * seq_len
            losses.append(loss.item())

        if step % 50 == 0:
            print(f"  Step {step}: loss = {loss.item():.4f}")

    elapsed = time.time() - start_time
    throughput = total_tokens / elapsed
    avg_loss = np.mean(losses[-100:])  # Last 100 steps

    return avg_loss, throughput


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/pile.txt', help='Training data file')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--seq-len', type=int, default=512, help='Sequence length')
    parser.add_argument('--dim', type=int, default=1280, help='Model dimension')
    parser.add_argument('--depth', type=int, default=6, help='Model depth')
    parser.add_argument('--steps', type=int, default=500, help='Training steps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    device = 'cuda'
    dtype = torch.bfloat16

    # Open data file
    with open(args.data, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    print(f"Config: dim={args.dim}, depth={args.depth}, batch={args.batch_size}, seq={args.seq_len}")
    print(f"Data: {args.data} ({len(mm) / 1e9:.2f} GB)")
    print()

    results = {}

    # Test E1
    print("=" * 60)
    print("E1 (Gated Elman)")
    print("=" * 60)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_e1 = LadderLM(
        vocab_size=256, dim=args.dim, depth=args.depth,
        level=1, expansion=1.0
    ).to(device).to(dtype)

    n_params = sum(p.numel() for p in model_e1.parameters())
    print(f"Parameters: {n_params / 1e6:.1f}M")

    loss_e1, throughput_e1 = train_model(
        model_e1, mm, args.batch_size, args.seq_len, args.steps, device
    )
    print(f"E1: loss={loss_e1:.4f}, throughput={throughput_e1 / 1000:.1f}K tok/s")
    results['E1'] = {'loss': loss_e1, 'throughput': throughput_e1}

    del model_e1
    torch.cuda.empty_cache()

    # Test E30
    print()
    print("=" * 60)
    print("E30 (E1 + Diagonal Gating)")
    print("=" * 60)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_e30 = LadderLM(
        vocab_size=256, dim=args.dim, depth=args.depth,
        level=30, expansion=1.0
    ).to(device).to(dtype)

    n_params = sum(p.numel() for p in model_e30.parameters())
    print(f"Parameters: {n_params / 1e6:.1f}M")

    loss_e30, throughput_e30 = train_model(
        model_e30, mm, args.batch_size, args.seq_len, args.steps, device
    )
    print(f"E30: loss={loss_e30:.4f}, throughput={throughput_e30 / 1000:.1f}K tok/s")
    results['E30'] = {'loss': loss_e30, 'throughput': throughput_e30}

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"E1:  loss={loss_e1:.4f}, throughput={throughput_e1 / 1000:.1f}K tok/s")
    print(f"E30: loss={loss_e30:.4f}, throughput={throughput_e30 / 1000:.1f}K tok/s")
    print(f"Loss improvement: {loss_e1 - loss_e30:+.4f} nats")
    print(f"Throughput ratio: {throughput_e30 / throughput_e1:.2f}x")

    mm.close()


if __name__ == '__main__':
    main()
