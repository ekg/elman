#!/usr/bin/env python3
"""
E12 vs E1 vs Mamba2 Benchmark
Compare selective gating (E12) against E1 and Mamba2 at 400M scale.

Uses same methodology as run_same_steps_sf.py:
- Same batch size (16), same steps (1500) for all models
- Schedule-free AdamW, no gradient clipping
- Byte-level training on pile.txt
"""

import os
import sys
import time
import mmap
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Set up environment
os.environ.setdefault('LD_LIBRARY_PATH', '/home/erikg/.local/lib/python3.12/site-packages/torch/lib')

from schedulefree import AdamWScheduleFree

# Get batch function
def make_get_batch(filepath, device):
    with open(filepath, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    file_len = len(mm)

    def get_batch(batch_size, seq_len):
        pos = np.random.randint(0, file_len - seq_len - 1, size=batch_size)
        buf = np.zeros((batch_size, seq_len + 1), dtype=np.uint8)
        for i, p in enumerate(pos):
            buf[i] = np.frombuffer(mm[p:p+seq_len+1], dtype=np.uint8)
        x = torch.from_numpy(buf[:, :-1]).long().to(device)
        y = torch.from_numpy(buf[:, 1:]).long().to(device)
        return x, y

    return get_batch


class ElmanLM(nn.Module):
    """Language model wrapper for E1/E12 cells."""

    def __init__(self, vocab_size, dim, depth, cell_class):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([cell_class(dim) for _ in range(depth)])
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.embed.weight  # Tie weights

    def forward(self, x, return_loss=False):
        if return_loss:
            targets = x[:, 1:].contiguous()
            x = x[:, :-1]

        B, T = x.shape
        h = self.embed(x)  # [B, T, dim]

        for layer in self.layers:
            out, _ = layer(h)
            h = h + out  # Residual

        logits = self.head(h)

        if return_loss:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return loss
        return logits


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train_model(name, model, get_batch, device, batch_size=16, seq_len=512, num_steps=1500, gpu_id=0):
    """Train model and return results."""

    model = model.to(device).bfloat16()
    torch.cuda.set_device(device)

    # Schedule-free AdamW
    opt = AdamWScheduleFree(model.parameters(), lr=3e-4, weight_decay=0.1)
    model.train()
    opt.train()

    # Warmup
    for _ in range(3):
        x, y = get_batch(batch_size, seq_len)
        loss = model(x, return_loss=True)
        loss.backward()
        opt.step()
        opt.zero_grad()

    torch.cuda.synchronize()

    # Track memory
    torch.cuda.reset_peak_memory_stats(device)

    initial_loss = None
    final_loss = None
    total_tokens = 0

    start = time.time()

    for step in range(1, num_steps + 1):
        x, y = get_batch(batch_size, seq_len)
        loss = model(x, return_loss=True)

        if step == 1:
            initial_loss = loss.item()

        loss.backward()
        opt.step()
        opt.zero_grad()

        total_tokens += batch_size * seq_len

        if step % 300 == 0 or step == num_steps:
            final_loss = loss.item()
            elapsed = time.time() - start
            tps = total_tokens / elapsed / 1000
            print(f"[{name}] Step {step}: loss={final_loss:.4f}, {tps:.1f}K tok/s")

    elapsed = time.time() - start
    memory = torch.cuda.max_memory_allocated(device) / 1e9
    tps = total_tokens / elapsed / 1000

    return {
        'name': name,
        'params': count_params(model),
        'loss': final_loss,
        'initial_loss': initial_loss,
        'tok_s': tps,
        'memory': memory,
        'time': elapsed
    }


def main():
    # Configuration - match 400M scale
    vocab_size = 256
    batch_size = 16
    seq_len = 512
    num_steps = 1500

    # E1/E12 config for ~400M params: dim=1760, depth=26
    e1_dim = 1760
    e1_depth = 26

    device = 'cuda:0'
    data_path = '/home/erikg/elman/data/pile.txt'

    # Seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    get_batch = make_get_batch(data_path, device)

    print("=" * 70)
    print("E12 vs E1 vs Mamba2 Benchmark (400M scale)")
    print("=" * 70)
    print(f"Config: batch={batch_size}, seq_len={seq_len}, steps={num_steps}")
    print(f"E1/E12: dim={e1_dim}, depth={e1_depth}")
    print()

    results = []

    # Test E1 (Mamba-Gated Elman)
    print("Testing E1 (Mamba-Gated Elman)...")
    from elman.models.mamba_gated_elman import MambaGatedElman

    torch.manual_seed(42)
    np.random.seed(42)

    e1_model = ElmanLM(vocab_size, e1_dim, e1_depth,
                       lambda dim: MambaGatedElman(dim, expansion=1.0))
    result = train_model('E1', e1_model, get_batch, device, batch_size, seq_len, num_steps)
    results.append(result)
    print(f"E1: {result['params']/1e6:.1f}M params, loss={result['loss']:.4f}, {result['tok_s']:.1f}K tok/s, {result['memory']:.1f} GB")
    del e1_model
    torch.cuda.empty_cache()
    print()

    # Test E12 (Selective Gated Elman)
    print("Testing E12 (Selective Gated Elman)...")
    from elman.models.selective_gated_elman import SelectiveGatedElman

    torch.manual_seed(42)
    np.random.seed(42)

    e12_model = ElmanLM(vocab_size, e1_dim, e1_depth,
                        lambda dim: SelectiveGatedElman(dim, expansion=1.0))
    result = train_model('E12', e12_model, get_batch, device, batch_size, seq_len, num_steps)
    results.append(result)
    print(f"E12: {result['params']/1e6:.1f}M params, loss={result['loss']:.4f}, {result['tok_s']:.1f}K tok/s, {result['memory']:.1f} GB")
    del e12_model
    torch.cuda.empty_cache()
    print()

    # Summary
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'Model':<10} {'Params':<10} {'Loss':<8} {'Tok/s':<10} {'Memory':<10} {'Time':<8}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<10} {r['params']/1e6:.1f}M{'':<4} {r['loss']:.4f}   {r['tok_s']:.1f}K{'':<4} {r['memory']:.1f} GB{'':<4} {r['time']:.0f}s")

    # Compare E12 to E1
    e1_result = [r for r in results if r['name'] == 'E1'][0]
    e12_result = [r for r in results if r['name'] == 'E12'][0]

    print()
    print("Comparison:")
    print(f"  E12 loss: {e12_result['loss']:.4f} vs E1 loss: {e1_result['loss']:.4f}")
    print(f"  Loss difference: {e12_result['loss'] - e1_result['loss']:.4f} nats")
    print(f"  E12 throughput: {e12_result['tok_s']:.1f}K vs E1: {e1_result['tok_s']:.1f}K ({100*e12_result['tok_s']/e1_result['tok_s']:.0f}%)")
    print(f"  E12 memory: {e12_result['memory']:.1f}GB vs E1: {e1_result['memory']:.1f}GB")


if __name__ == '__main__':
    main()
