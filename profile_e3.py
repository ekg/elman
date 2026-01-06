#!/usr/bin/env python3
"""Profile E3 to find bottlenecks."""

import torch
import torch.nn.functional as F
import time

device = 'cuda'
B, T, D = 32, 512, 768
x = torch.randn(B, T, D, device=device, dtype=torch.bfloat16)

from elman.models.lowrank_slot_elman import LowRankSlotElman

print("Profiling E3 (LowRankSlotElman) FULL LAYER with rank=64...")
layer = LowRankSlotElman(dim=D, expansion=1.0, n_slots=8, rank=64).to(device).bfloat16()

# Warmup
for _ in range(3):
    out, h = layer(x)
    out.sum().backward()
    layer.zero_grad()

torch.cuda.synchronize()

# Profile with torch
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
) as prof:
    for _ in range(5):
        out, h = layer(x)
        out.sum().backward()
        layer.zero_grad()
        torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
