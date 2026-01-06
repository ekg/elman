#!/usr/bin/env python3
"""Benchmark E3 with diagonal Triton kernel."""

import torch
import time

device = 'cuda'
B, T, D = 32, 512, 768

from elman.models.lowrank_slot_elman import LowRankSlotElman

print("Benchmarking E3 layer with diag=True (Triton) vs diag=False (CUDA)...")

x = torch.randn(B, T, D, device=device, dtype=torch.bfloat16)

# Diagonal (Triton)
layer_diag = LowRankSlotElman(dim=D, expansion=1.0, n_slots=8, diag=True).to(device).bfloat16()
print(f"Diag layer: {layer_diag.extra_repr()}")

for _ in range(5):
    out, h = layer_diag(x)
    out.sum().backward()
    layer_diag.zero_grad()

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(20):
    out, h = layer_diag(x)
    out.sum().backward()
    layer_diag.zero_grad()
torch.cuda.synchronize()
diag_time = (time.perf_counter() - t0) / 20 * 1000
print(f"Diag (Triton): {diag_time:.1f}ms, {B * T / (diag_time / 1000) / 1000:.1f}k tok/s")

# Low-rank (CUDA)
layer_lr = LowRankSlotElman(dim=D, expansion=1.0, n_slots=8, rank=64, diag=False).to(device).bfloat16()
print(f"Low-rank layer: {layer_lr.extra_repr()}")

for _ in range(5):
    out, h = layer_lr(x)
    out.sum().backward()
    layer_lr.zero_grad()

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(20):
    out, h = layer_lr(x)
    out.sum().backward()
    layer_lr.zero_grad()
torch.cuda.synchronize()
lr_time = (time.perf_counter() - t0) / 20 * 1000
print(f"Low-rank (CUDA): {lr_time:.1f}ms, {B * T / (lr_time / 1000) / 1000:.1f}k tok/s")

print(f"\nSpeedup: {lr_time / diag_time:.1f}x")
