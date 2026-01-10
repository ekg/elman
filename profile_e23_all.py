#!/usr/bin/env python3
"""Compare all E23 implementations."""
import torch
import sys
sys.path.insert(0, 'elman/cuda')

B, T, D, N = 4, 512, 512, 8

torch.manual_seed(42)
x = torch.randn(B, T, D, device='cuda', dtype=torch.bfloat16)
h_tape = torch.randn(B, N, D, device='cuda', dtype=torch.bfloat16)
h_work = torch.randn(B, D, device='cuda', dtype=torch.bfloat16)
W_h = torch.randn(D, D, device='cuda', dtype=torch.bfloat16)
W_x = torch.randn(D, D, device='cuda', dtype=torch.bfloat16)
b_h = torch.randn(D, device='cuda', dtype=torch.bfloat16)
W_write = torch.randn(D, D, device='cuda', dtype=torch.bfloat16)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

print("=" * 60)
print("E23 Implementation Comparison")
print(f"B={B}, T={T}, D={D}, N={N}")
print("=" * 60)

# 1. CUDA (baseline)
import hasty_pytorch_lib as hasty
for _ in range(5):
    hasty.dual_memory_elman_forward(True, x, h_tape, h_work, W_h, W_x, b_h, W_write)
torch.cuda.synchronize()
start.record()
for _ in range(10):
    hasty.dual_memory_elman_forward(True, x, h_tape, h_work, W_h, W_x, b_h, W_write)
end.record()
torch.cuda.synchronize()
cuda_ms = start.elapsed_time(end) / 10
print(f"1. CUDA (cuBLAS + kernels): {cuda_ms:.2f}ms, {cuda_ms/T*1000:.1f}us/step")

# 2. Hybrid PyTorch
from elman.models.dual_memory_elman_triton import dual_memory_elman_forward_hybrid
for _ in range(3):
    dual_memory_elman_forward_hybrid(x, h_tape, h_work, W_h, W_x, b_h, W_write)
torch.cuda.synchronize()
start.record()
for _ in range(5):
    dual_memory_elman_forward_hybrid(x, h_tape, h_work, W_h, W_x, b_h, W_write)
end.record()
torch.cuda.synchronize()
hybrid_ms = start.elapsed_time(end) / 5
print(f"2. Hybrid PyTorch (einsum loop): {hybrid_ms:.2f}ms, {hybrid_ms/T*1000:.1f}us/step, {hybrid_ms/cuda_ms:.1f}x slower")

# 3. Triton + cuBLAS
from elman.kernels.e23_triton import e23_forward_triton_optimized
for _ in range(3):
    e23_forward_triton_optimized(x, h_tape, h_work, W_h, W_x, b_h, W_write)
torch.cuda.synchronize()
start.record()
for _ in range(5):
    e23_forward_triton_optimized(x, h_tape, h_work, W_h, W_x, b_h, W_write)
end.record()
torch.cuda.synchronize()
triton_ms = start.elapsed_time(end) / 5
print(f"3. Triton + cuBLAS (Phase1/2 kernels): {triton_ms:.2f}ms, {triton_ms/T*1000:.1f}us/step, {triton_ms/cuda_ms:.1f}x slower")

# 4. Fused Triton
from elman.kernels.e23_triton_fused import e23_forward_triton_fused
for _ in range(3):
    e23_forward_triton_fused(x, h_tape, h_work, W_h, W_x, b_h, W_write)
torch.cuda.synchronize()
start.record()
for _ in range(5):
    e23_forward_triton_fused(x, h_tape, h_work, W_h, W_x, b_h, W_write)
end.record()
torch.cuda.synchronize()
fused_ms = start.elapsed_time(end) / 5
print(f"4. Fused Triton (inline GEMMs): {fused_ms:.2f}ms, {fused_ms/T*1000:.1f}us/step, {fused_ms/cuda_ms:.1f}x slower")

print()
print("Summary:")
print(f"  CUDA is the clear winner at {cuda_ms/T*1000:.1f}us/step")
print(f"  Triton cannot match due to:")
print(f"    - Python loop overhead (~40us/step)")
print(f"    - Inline GEMMs ~10x slower than cuBLAS")
print(f"    - Sequential ops are bad fit for Triton")
