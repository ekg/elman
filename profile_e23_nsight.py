#!/usr/bin/env python3
"""
Simple profiling script for nsight-compute to analyze E23 kernels.
"""
import torch
import sys
sys.path.insert(0, 'elman/cuda')

from elman.models.dual_memory_elman import DualMemoryElman

batch_size = 64
seq_len = 512
dim = 512
n_slots = 16  # Test n=16 specifically

print(f"E23 n={n_slots} profiling for nsight-compute")
print(f"batch={batch_size}, seq={seq_len}, dim={dim}")

layer = DualMemoryElman(dim=dim, n_slots=n_slots).cuda().bfloat16()
x = torch.randn(batch_size, seq_len, dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)

# Warmup
for _ in range(3):
    x_in = x.detach().clone().requires_grad_(True)
    out = layer(x_in)
    out[0].sum().backward()

torch.cuda.synchronize()

# Profile a single forward+backward
x_in = x.detach().clone().requires_grad_(True)
out = layer(x_in)
out[0].sum().backward()

torch.cuda.synchronize()
print("Done")
