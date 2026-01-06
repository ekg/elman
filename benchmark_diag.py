#!/usr/bin/env python3
"""Benchmark diagonal slot elman."""

import torch
import time

device = 'cuda'
B, T, D = 32, 512, 768
x = torch.randn(B, T, D, device=device, dtype=torch.bfloat16)

from elman.models.lowrank_slot_elman import LowRankSlotElmanCell

print("Benchmarking diagonal vs low-rank (CUDA kernel)...")

# Low-rank with CUDA kernel
cell_lr = LowRankSlotElmanCell(dim=D, n_slots=8, rank=64, diag=False).to(device).bfloat16()
x_t = x.permute(1, 0, 2).contiguous()
z_t = torch.randn_like(x_t)

for _ in range(3):
    out, h = cell_lr(x_t, z_t)
    out.sum().backward()
    cell_lr.zero_grad()

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    out, h = cell_lr(x_t, z_t)
    out.sum().backward()
    cell_lr.zero_grad()
torch.cuda.synchronize()
lr_time = (time.perf_counter() - t0) / 10 * 1000
print(f"Low-rank (CUDA): {lr_time:.1f}ms")

# Diagonal (Python loop)
cell_diag = LowRankSlotElmanCell(dim=D, n_slots=8, diag=True).to(device).bfloat16()

for _ in range(3):
    out, h = cell_diag(x_t, z_t)
    out.sum().backward()
    cell_diag.zero_grad()

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    out, h = cell_diag(x_t, z_t)
    out.sum().backward()
    cell_diag.zero_grad()
torch.cuda.synchronize()
diag_time = (time.perf_counter() - t0) / 10 * 1000
print(f"Diagonal (Python loop): {diag_time:.1f}ms")

# Try torch.compile
print("\nTrying torch.compile on diagonal...")
cell_diag_compiled = torch.compile(cell_diag, mode="reduce-overhead")

for _ in range(5):  # More warmup for compile
    out, h = cell_diag_compiled(x_t, z_t)
    out.sum().backward()
    cell_diag.zero_grad()

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    out, h = cell_diag_compiled(x_t, z_t)
    out.sum().backward()
    cell_diag.zero_grad()
torch.cuda.synchronize()
compiled_time = (time.perf_counter() - t0) / 10 * 1000
print(f"Diagonal (compiled): {compiled_time:.1f}ms")

print(f"\nSpeedup from compile: {diag_time/compiled_time:.1f}x")
