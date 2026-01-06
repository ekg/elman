#!/usr/bin/env python3
import torch
import time

device = 'cuda'
B, T, D, S = 32, 512, 768, 8

from elman.kernels.diag_slot_triton import diag_slot_forward, diag_slot_recurrence

Wx = torch.randn(T, B, D, device=device, dtype=torch.bfloat16)
z = torch.randn(T, B, D, device=device, dtype=torch.bfloat16)
h0 = torch.zeros(B, S, D, device=device, dtype=torch.bfloat16)
A = torch.sigmoid(torch.randn(S, D, device=device, dtype=torch.bfloat16)) * 0.99
b = torch.zeros(D, device=device, dtype=torch.bfloat16)
C = torch.ones(S, device=device, dtype=torch.bfloat16) / S

# Warmup
for _ in range(5):
    output, h = diag_slot_forward(Wx, z, h0, A, b, C)
torch.cuda.synchronize()

# Time forward only
t0 = time.perf_counter()
for _ in range(20):
    output, h = diag_slot_forward(Wx, z, h0, A, b, C)
torch.cuda.synchronize()
fwd_time = (time.perf_counter() - t0) / 20 * 1000
print(f"Forward only: {fwd_time:.1f}ms")

# Time forward + backward
Wx.requires_grad_(True)
A.requires_grad_(True)

for _ in range(5):
    output, h = diag_slot_recurrence(Wx, z, h0, A, b, C)
    output.sum().backward()
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(20):
    Wx_g = Wx.detach().requires_grad_(True)
    A_g = A.detach().requires_grad_(True)
    output, h = diag_slot_recurrence(Wx_g, z, h0, A_g, b, C)
    output.sum().backward()
torch.cuda.synchronize()
total_time = (time.perf_counter() - t0) / 20 * 1000
print(f"Forward + Backward: {total_time:.1f}ms")
print(f"Backward only: {total_time - fwd_time:.1f}ms")
print(f"Throughput: {B * T / (total_time / 1000) / 1000:.1f}k tok/s")
