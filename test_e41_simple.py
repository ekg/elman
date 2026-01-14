#!/usr/bin/env python3
"""
Simple test of E41: single timestep forward/backward.
"""

import torch
import torch.nn.functional as F

device = 'cuda:0'
dtype = torch.bfloat16

# Simple dimensions
B, dim = 2, 8
T = 1

print("=" * 60)
print("E41 Simple Single-Step Test")
print("=" * 60)

torch.manual_seed(42)

# Create simple test data
x = torch.randn(T, B, dim, device=device, dtype=dtype, requires_grad=True)
h0 = torch.zeros(B, dim, device=device, dtype=dtype)
d_x = torch.ones(dim, device=device, dtype=dtype, requires_grad=True)  # diagonal = 1 (identity)
W_h = torch.zeros(dim, dim, device=device, dtype=dtype, requires_grad=True)  # W_h = 0 (no recurrence)
b = torch.zeros(dim, device=device, dtype=dtype, requires_grad=True)

# Clone for CUDA
x_cuda = x.clone().detach().requires_grad_(True)
d_x_cuda = d_x.clone().detach().requires_grad_(True)
W_h_cuda = W_h.clone().detach().requires_grad_(True)
b_cuda = b.clone().detach().requires_grad_(True)

print("\n--- Python Forward ---")
# Python: raw = d_x * x + h_prev @ W_h.T + b
# With h_prev=0, W_h=0: raw = d_x * x = x (since d_x=1)
h_prev = h0
x_t = x[0]
raw_py = d_x * x_t + h_prev @ W_h.T + b
h_py = torch.tanh(raw_py)
out_py = h_py * F.silu(h_py)
print(f"  raw_py (should equal x): {raw_py[:1, :4]}")
print(f"  h_py (tanh of x): {h_py[:1, :4]}")
print(f"  out_py (h * silu(h)): {out_py[:1, :4]}")

print("\n--- CUDA Forward ---")
import hasty_pytorch_lib
h_cuda, output_cuda, v_cuda = hasty_pytorch_lib.e41_diagonal_wx_forward(
    True,  # training
    x_cuda,
    h0,
    d_x_cuda,
    W_h_cuda,
    b_cuda
)
print(f"  v_cuda (pre-activation): {v_cuda[0, :1, :4]}")
print(f"  h_cuda: {h_cuda[1, :1, :4]}")
print(f"  out_cuda: {output_cuda[0, :1, :4]}")

# Compare
v_diff = (raw_py.unsqueeze(0) - v_cuda).abs().max().item()
h_diff = (h_py.unsqueeze(0) - h_cuda[1]).abs().max().item()
out_diff = (out_py.unsqueeze(0) - output_cuda[0]).abs().max().item()
print(f"\n  v diff: {v_diff:.6e}")
print(f"  h diff: {h_diff:.6e}")
print(f"  out diff: {out_diff:.6e}")

print("\n--- Backward Test ---")
grad_out = torch.ones_like(out_py)

# Python backward
loss_py = out_py.sum()
loss_py.backward()
print(f"  Python dx grad: {x.grad[:1, :4]}")
print(f"  Python d_x grad: {d_x.grad[:4]}")

# CUDA backward
dx_cuda, dd_x_cuda, dW_h_cuda, db_cuda = hasty_pytorch_lib.e41_diagonal_wx_backward(
    d_x_cuda,
    W_h_cuda,
    x_cuda,
    h_cuda,
    v_cuda,
    grad_out.unsqueeze(0)
)
print(f"  CUDA dx grad: {dx_cuda[0, :1, :4]}")
print(f"  CUDA d_x grad: {dd_x_cuda[:4]}")

dx_diff = (x.grad - dx_cuda[0]).abs().max().item()
dd_x_diff = (d_x.grad - dd_x_cuda).abs().max().item()
print(f"\n  dx diff: {dx_diff:.6e}")
print(f"  dd_x diff: {dd_x_diff:.6e}")

print("\n" + "=" * 60)
if out_diff < 1e-3 and dx_diff < 1e-2:
    print("Test PASSED")
else:
    print("Test FAILED")
