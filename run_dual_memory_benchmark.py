#!/usr/bin/env python3
"""
Benchmark dual-memory models: E23, E24, E25, E26, E28
Runs across 8 GPUs in parallel with 10 min timeout.
"""

import os
import sys
import time
import math
import mmap
import numpy as np
import torch
import torch.nn.functional as F
from schedulefree import AdamWScheduleFree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elman.models import create_ladder_model


def get_batch(mm, batch_size, seq_len, device):
    """Sample random positions from mmap."""
    pos = np.random.randint(0, len(mm) - seq_len - 2, size=batch_size)
    buf = np.zeros((batch_size, seq_len + 1), dtype=np.int64)
    for i, p in enumerate(pos):
        buf[i] = np.frombuffer(mm[p:p + seq_len + 1], dtype=np.uint8).astype(np.int64)
    return torch.from_numpy(buf).to(device)


def train_model(level, output_file, timeout=600, batch_size=4, seq_len=512):
    """Train a model for timeout seconds and log results."""
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Create model
    model = create_ladder_model(
        target_params="50m",
        level=level,
        vocab_size=256,
    ).to(device=device, dtype=dtype)

    num_params = model.get_num_params()

    # Check if fused kernel available
    try:
        layer = model.layers[0]
        fused = getattr(layer, 'use_cuda', False) or hasattr(layer, 'cuda_forward')
    except:
        fused = False

    print(f"Created Level {level} model: dim=512, depth={model.depth}, params={num_params:,}")
    print(f"E{level}: {num_params:,} params, fused={fused}")

    # Open data
    with open('data/pile.txt', 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    # Optimizer
    optimizer = AdamWScheduleFree(model.parameters(), lr=3e-4, weight_decay=0.1)

    model.train()
    optimizer.train()

    start_time = time.time()
    step = 0
    tokens_seen = 0
    losses = []

    while True:
        step += 1
        elapsed = time.time() - start_time

        if elapsed >= timeout:
            break

        # Get batch
        batch = get_batch(mm, batch_size, seq_len, device)

        # Forward
        optimizer.zero_grad()
        loss = model(batch, return_loss=True)
        loss.backward()

        # Clip and step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)
        tokens_seen += batch_size * seq_len

        # Log every 100 steps
        if step % 100 == 0:
            avg_loss = sum(losses[-100:]) / min(100, len(losses))
            tps = tokens_seen / elapsed / 1000
            print(f"Step {step} | {elapsed:.0f}s | Loss {loss_val:.4f} | Avg100 {avg_loss:.4f} | {tps:.1f}K tok/s")

    # Final stats
    final_avg = sum(losses[-100:]) / min(100, len(losses))
    final_tps = tokens_seen / elapsed / 1000

    print(f"\n=== FINAL: E{level} ===")
    print(f"Params: {num_params:,}")
    print(f"Steps: {step}")
    print(f"Final Avg100 Loss: {final_avg:.4f}")
    print(f"Throughput: {final_tps:.1f}K tok/s")

    mm.close()
    return {"level": level, "params": num_params, "steps": step, "loss": final_avg, "tps": final_tps}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=str, required=True, help="Model level (23, 24, 25, 26, 28, etc)")
    parser.add_argument("--output", type=str, required=True, help="Output log file")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds")
    args = parser.parse_args()

    # Convert level to int or keep as string for variants
    try:
        level = int(args.level)
    except ValueError:
        level = args.level  # e.g., "23n32"

    # Redirect stdout to file
    import sys
    sys.stdout = open(args.output, 'w', buffering=1)

    train_model(level, args.output, timeout=args.timeout)
