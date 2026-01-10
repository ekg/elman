#!/usr/bin/env python3
"""Profile E23 CUDA vs Triton implementations."""
import torch
import sys
sys.path.insert(0, 'elman/cuda')

batch_size = 4
seq_len = 512
dim = 512
n_slots = 8

# Create inputs
x = torch.randn(batch_size, seq_len, dim, device='cuda', dtype=torch.bfloat16)
W_h = torch.randn(dim, dim, device='cuda', dtype=torch.bfloat16)
b_h = torch.randn(dim, device='cuda', dtype=torch.bfloat16)
W_x = torch.randn(dim, dim, device='cuda', dtype=torch.bfloat16)
W_write = torch.randn(dim, dim, device='cuda', dtype=torch.bfloat16)
h_tape_init = torch.randn(batch_size, n_slots, dim, device='cuda', dtype=torch.bfloat16)
h_work_init = torch.randn(batch_size, dim, device='cuda', dtype=torch.bfloat16)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

print("=" * 60)
print(f"E23 Implementation Comparison")
print(f"batch={batch_size}, seq={seq_len}, dim={dim}, n_slots={n_slots}")
print("=" * 60)

# 1. CUDA kernel
print("\n1. CUDA kernel (hasty)...")
import hasty_pytorch_lib as hasty

for _ in range(5):
    result = hasty.dual_memory_elman_forward(True, x, h_tape_init, h_work_init, W_h, W_x, b_h, W_write)

torch.cuda.synchronize()
start.record()
for _ in range(10):
    result = hasty.dual_memory_elman_forward(True, x, h_tape_init, h_work_init, W_h, W_x, b_h, W_write)
end.record()
torch.cuda.synchronize()
cuda_ms = start.elapsed_time(end) / 10
print(f"   Time: {cuda_ms:.2f}ms ({cuda_ms/seq_len*1000:.1f}us/step)")

# 2. Hybrid implementation (PyTorch + einsum)
print("\n2. Hybrid (PyTorch + einsum)...")
from elman.models.dual_memory_elman_triton import dual_memory_elman_forward_hybrid

for _ in range(5):
    result = dual_memory_elman_forward_hybrid(x, h_tape_init, h_work_init, W_h, W_x, b_h, W_write)

torch.cuda.synchronize()
start.record()
for _ in range(10):
    result = dual_memory_elman_forward_hybrid(x, h_tape_init, h_work_init, W_h, W_x, b_h, W_write)
end.record()
torch.cuda.synchronize()
hybrid_ms = start.elapsed_time(end) / 10
print(f"   Time: {hybrid_ms:.2f}ms ({hybrid_ms/seq_len*1000:.1f}us/step)")
print(f"   vs CUDA: {hybrid_ms/cuda_ms:.2f}x slower")

# 3. Profile individual operations in hybrid
print("\n3. Breaking down hybrid implementation...")

# Pre-compute x_proj
x_proj = x @ W_x.T

# Single timestep operations
h_tape = h_tape_init.clone()
h_work = h_work_init.clone()
scale = 1.0 / (dim ** 0.5)

# Read attention
torch.cuda.synchronize()
start.record()
for _ in range(1000):
    read_scores = torch.einsum('bnd,bd->bn', h_tape, h_work) * scale
    read_attn = torch.softmax(read_scores, dim=-1)
    read = torch.einsum('bn,bnd->bd', read_attn, h_tape)
end.record()
torch.cuda.synchronize()
read_us = start.elapsed_time(end)
print(f"   Read attention: {read_us:.1f}us/step")

# Update h_work (GEMM + element-wise)
torch.cuda.synchronize()
start.record()
for _ in range(1000):
    pre_act = h_work @ W_h.T + x_proj[:, 0] + read + b_h
    h_work_new = torch.tanh(pre_act)
end.record()
torch.cuda.synchronize()
update_us = start.elapsed_time(end)
print(f"   Update h_work: {update_us:.1f}us/step")

# Write attention + tape update
torch.cuda.synchronize()
start.record()
for _ in range(1000):
    write_value = h_work_new @ W_write.T
    write_scores = torch.einsum('bnd,bd->bn', h_tape, h_work_new) * scale
    write_attn = torch.softmax(write_scores, dim=-1)
    h_tape_new = (1 - write_attn[:, :, None]) * h_tape + write_attn[:, :, None] * write_value[:, None, :]
end.record()
torch.cuda.synchronize()
write_us = start.elapsed_time(end)
print(f"   Write + tape update: {write_us:.1f}us/step")

total_breakdown = read_us + update_us + write_us
print(f"   Sum of ops: {total_breakdown:.1f}us/step")
print(f"   Actual hybrid: {hybrid_ms/seq_len*1000:.1f}us/step")
print(f"   Python loop overhead: {hybrid_ms/seq_len*1000 - total_breakdown/1000:.1f}us/step")

# 4. Test larger batch to see scaling
print("\n4. Batch size scaling...")
for bs in [4, 16, 64]:
    x_bs = torch.randn(bs, seq_len, dim, device='cuda', dtype=torch.bfloat16)
    h_tape_bs = torch.randn(bs, n_slots, dim, device='cuda', dtype=torch.bfloat16)
    h_work_bs = torch.randn(bs, dim, device='cuda', dtype=torch.bfloat16)

    # CUDA
    for _ in range(3):
        hasty.dual_memory_elman_forward(True, x_bs, h_tape_bs, h_work_bs, W_h, W_x, b_h, W_write)
    torch.cuda.synchronize()
    start.record()
    for _ in range(5):
        hasty.dual_memory_elman_forward(True, x_bs, h_tape_bs, h_work_bs, W_h, W_x, b_h, W_write)
    end.record()
    torch.cuda.synchronize()
    cuda_bs = start.elapsed_time(end) / 5

    # Hybrid
    for _ in range(3):
        dual_memory_elman_forward_hybrid(x_bs, h_tape_bs, h_work_bs, W_h, W_x, b_h, W_write)
    torch.cuda.synchronize()
    start.record()
    for _ in range(5):
        dual_memory_elman_forward_hybrid(x_bs, h_tape_bs, h_work_bs, W_h, W_x, b_h, W_write)
    end.record()
    torch.cuda.synchronize()
    hybrid_bs = start.elapsed_time(end) / 5

    print(f"   batch={bs:2d}: CUDA={cuda_bs:.1f}ms, Hybrid={hybrid_bs:.1f}ms, ratio={hybrid_bs/cuda_bs:.2f}x")
