#!/usr/bin/env python3
"""
Debug NaN issue in E23 backward pass.
"""
import torch
import sys
sys.path.insert(0, 'elman/cuda')

import hasty_pytorch_lib as hasty

batch_size = 4
seq_len = 32
dim = 512
n_slots = 8

print("=" * 70)
print("E23 NaN Debug")
print(f"batch={batch_size}, seq={seq_len}, dim={dim}, n_slots={n_slots}")
print("=" * 70)

# Use smaller random values to avoid numerical issues
torch.manual_seed(42)
scale = 0.1

W_h = (torch.randn(dim, dim, device='cuda', dtype=torch.bfloat16) * scale)
W_x = (torch.randn(dim, dim, device='cuda', dtype=torch.bfloat16) * scale)
b_h = torch.zeros(dim, device='cuda', dtype=torch.bfloat16)
W_write = (torch.randn(dim, dim, device='cuda', dtype=torch.bfloat16) * scale)
h_tape_init = (torch.randn(batch_size, n_slots, dim, device='cuda', dtype=torch.bfloat16) * scale)
h_work_init = (torch.randn(batch_size, dim, device='cuda', dtype=torch.bfloat16) * scale)
x_seq = (torch.randn(batch_size, seq_len, dim, device='cuda', dtype=torch.bfloat16) * scale)

print("\n1. Check forward pass")
print("-" * 50)

result = hasty.dual_memory_elman_forward(True, x_seq, h_tape_init, h_work_init, W_h, W_x, b_h, W_write)
h_work_out, h_tape_final, h_tape_all, read_attn, write_attn = result

print(f"h_work_out: shape={h_work_out.shape}, nan={h_work_out.isnan().any()}, max={h_work_out.abs().max():.4f}")
print(f"h_tape_final: shape={h_tape_final.shape}, nan={h_tape_final.isnan().any()}, max={h_tape_final.abs().max():.4f}")
print(f"h_tape_all: shape={h_tape_all.shape}, nan={h_tape_all.isnan().any()}, max={h_tape_all.abs().max():.4f}")
print(f"read_attn: shape={read_attn.shape}, nan={read_attn.isnan().any()}, max={read_attn.abs().max():.4f}")
print(f"write_attn: shape={write_attn.shape}, nan={write_attn.isnan().any()}, max={write_attn.abs().max():.4f}")

print("\n2. Check backward pass")
print("-" * 50)

# Use small gradient inputs
d_h_work_out = torch.ones_like(h_work_out) * 0.01
d_h_tape_final = torch.zeros_like(h_tape_final)

x_proj = x_seq @ W_x.T
print(f"x_proj: shape={x_proj.shape}, nan={x_proj.isnan().any()}, max={x_proj.abs().max():.4f}")

dx_proj, dW_h, db_h, dW_write = hasty.dual_memory_elman_backward(
    x_proj, h_work_out, h_tape_all, read_attn, write_attn,
    W_h, W_write, d_h_work_out, d_h_tape_final)

print(f"dx_proj: shape={dx_proj.shape}, nan={dx_proj.isnan().any()}, max={dx_proj.abs().max():.4f}")
print(f"dW_h: shape={dW_h.shape}, nan={dW_h.isnan().any()}, max={dW_h.abs().max():.4f}")
print(f"db_h: shape={db_h.shape}, nan={db_h.isnan().any()}, max={db_h.abs().max():.4f}")
print(f"dW_write: shape={dW_write.shape}, nan={dW_write.isnan().any()}, max={dW_write.abs().max():.4f}")

print("\n3. Check where NaN appears")
print("-" * 50)

if dx_proj.isnan().any():
    nan_idx = dx_proj.isnan().nonzero()
    print(f"dx_proj has NaN at {nan_idx.shape[0]} positions")
    print(f"First few NaN positions: {nan_idx[:5]}")

if dW_h.isnan().any():
    nan_idx = dW_h.isnan().nonzero()
    print(f"dW_h has NaN at {nan_idx.shape[0]} positions")

if dW_write.isnan().any():
    nan_idx = dW_write.isnan().nonzero()
    print(f"dW_write has NaN at {nan_idx.shape[0]} positions")

print("\n4. Compare with Python reference")
print("-" * 50)

# Run Python backward
from elman.models.dual_memory_elman import DualMemoryElmanCell, e23_backward_python

cell = DualMemoryElmanCell(dim=dim, n_slots=n_slots).cuda().bfloat16()
# Copy weights
cell.W_h.data = W_h.clone()
cell.W_x.data = W_x.clone()
cell.b_h.data = b_h.clone()
cell.W_write.data = W_write.clone()

# Run forward with Python
x_in = x_seq.detach().clone().requires_grad_(True)
h_work_py, h_tape_py, _ = cell(x_in, h_tape_init.clone(), h_work_init.clone(), use_cuda=False, use_triton=False)

print(f"Python h_work_out: nan={h_work_py.isnan().any()}, max={h_work_py.abs().max():.4f}")
print(f"Match forward: {torch.allclose(h_work_out.float(), h_work_py.float(), atol=0.1)}")

# Run backward with Python
loss = h_work_py.sum()
loss.backward()

print(f"Python dx: nan={x_in.grad.isnan().any()}, max={x_in.grad.abs().max():.4f}")
print(f"Python dW_h: nan={cell.W_h.grad.isnan().any()}, max={cell.W_h.grad.abs().max():.4f}")

# Compare
print(f"\nCUDA dx_proj max: {dx_proj.abs().max():.4f}")
print(f"Python dx max: {x_in.grad.abs().max():.4f}")
