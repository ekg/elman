#!/usr/bin/env python
"""
Quick throughput test for MoM E88 vs E88.
"""

import time
import torch
import sys
sys.path.insert(0, '/home/erikg/elman')

from elman.models.mom_e88 import MoME88
from elman.models.e88_fla_hybrid import E88FLAHybrid

device = 'cuda:0'
dtype = torch.bfloat16

batch_size = 8
seq_len = 512
warmup_steps = 10
measure_steps = 50

def measure_throughput(model, name):
    """Measure throughput for a model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Warmup
    for _ in range(warmup_steps):
        x = torch.randn(batch_size, seq_len, model.dim, device=device, dtype=dtype)
        out, _ = model(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()

    # Measure
    start_time = time.perf_counter()
    total_tokens = 0

    for step in range(measure_steps):
        x = torch.randn(batch_size, seq_len, model.dim, device=device, dtype=dtype)
        out, _ = model(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_tokens += batch_size * seq_len

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    tok_per_sec = total_tokens / elapsed
    mem_used = torch.cuda.max_memory_allocated() / 1e9

    print(f"{name}:")
    print(f"  Throughput: {tok_per_sec/1000:.1f}K tok/s")
    print(f"  Peak memory: {mem_used:.2f} GB")
    print(f"  Time: {elapsed:.2f}s for {measure_steps} steps")

    del optimizer
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


print("=" * 60)
print("MoM E88 vs E88 Throughput Comparison")
print("=" * 60)

# E88 baseline - 64 heads
print("\n--- E88 (64 heads) ---")
e88 = E88FLAHybrid(
    dim=512,
    n_heads=64,
    n_state=16,
    expansion=1.0,
    use_gate=True,
).to(device=device, dtype=dtype)
n_params = sum(p.numel() for p in e88.parameters())
print(f"Params: {n_params/1e6:.1f}M")
measure_throughput(e88, "E88 h=64")
del e88

# MoM E88 - 64 heads, top_k=16 (4x sparse)
print("\n--- MoM E88 (64 heads, top_k=16) ---")
mom64k16 = MoME88(
    dim=512,
    n_heads=64,
    top_k=16,
    n_state=16,
    expansion=1.0,
    use_gate=True,
).to(device=device, dtype=dtype)
n_params = sum(p.numel() for p in mom64k16.parameters())
print(f"Params: {n_params/1e6:.1f}M")
measure_throughput(mom64k16, "MoM h=64 k=16")
del mom64k16

# MoM E88 - 128 heads, top_k=16 (8x sparse, 2x E88 heads)
print("\n--- MoM E88 (128 heads, top_k=16) ---")
mom128k16 = MoME88(
    dim=512,
    n_heads=128,
    top_k=16,
    n_state=16,
    expansion=1.0,
    use_gate=True,
).to(device=device, dtype=dtype)
n_params = sum(p.numel() for p in mom128k16.parameters())
print(f"Params: {n_params/1e6:.1f}M")
measure_throughput(mom128k16, "MoM h=128 k=16")
del mom128k16

# MoM E88 - 128 heads, top_k=32 (4x sparse)
print("\n--- MoM E88 (128 heads, top_k=32) ---")
mom128k32 = MoME88(
    dim=512,
    n_heads=128,
    top_k=32,
    n_state=16,
    expansion=1.0,
    use_gate=True,
).to(device=device, dtype=dtype)
n_params = sum(p.numel() for p in mom128k32.parameters())
print(f"Params: {n_params/1e6:.1f}M")
measure_throughput(mom128k32, "MoM h=128 k=32")
del mom128k32

print("\n" + "=" * 60)
print("Done")
