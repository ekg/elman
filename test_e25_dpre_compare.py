"""Compare d_pre_act between Python and CUDA."""

import torch
import torch.nn as nn
import math

torch.manual_seed(42)

B = 1
T = 1
D = 256
N = 8

device = 'cuda'
dtype = torch.bfloat16

print("="*60)
print("E25 d_pre_act Comparison Test")
print("="*60)

from elman.models.e25_entmax import E25DualMemoryElmanCell

# Create cell
torch.manual_seed(42)
cell = E25DualMemoryElmanCell(dim=D, n_slots=N).to(device).to(dtype)

# Input
torch.manual_seed(123)
x = torch.randn(B, T, D, device=device, dtype=dtype)
h_tape_init = torch.zeros(B, N, D, device=device, dtype=dtype)
h_work_init = torch.zeros(B, D, device=device, dtype=dtype)

scale = 1.0 / math.sqrt(D)

# === Python forward and backward ===
print("\n--- Python Path ---")
from elman.models.e25_entmax import e25_forward_step_python

h_work_py, h_tape_py, read_attn_py, write_attn_py, read_py = e25_forward_step_python(
    x[0, 0, None, :].float(),  # [1, D] - add batch dim
    h_tape_init[0, None, :, :].float(),  # [1, N, D] - keep batch dim
    h_work_init[0, None, :].float(),  # [1, D] - add batch dim
    cell.W_h.float(), cell.W_x.float(), cell.b_h.float(), cell.W_write.float(),
    scale
)
h_work_py = h_work_py[0]  # Remove batch dim

print(f"h_work_py sample: {h_work_py[:4].tolist()}")

# Python backward
d_h_work = torch.ones(D, device=device, dtype=torch.float32)  # Upstream gradient
d_pre_act_py = d_h_work * (1 - h_work_py ** 2)
print(f"d_pre_act_py norm: {d_pre_act_py.norm().item():.4f}")
print(f"d_pre_act_py sample: {d_pre_act_py[:4].tolist()}")

# dW_x = d_pre_act^T @ x
dW_x_py = d_pre_act_py[:, None] @ x[0, 0, None, :].float()  # [D, D]
print(f"dW_x_py[0, :4]: {dW_x_py[0, :4].tolist()}")

# === CUDA forward and backward with inspection ===
print("\n--- CUDA Path (via hook) ---")

# Run CUDA forward
import hasty_pytorch_lib

h_work_all_cuda, h_tape_final_cuda, h_tape_all_cuda, read_attn_all_cuda, write_attn_all_cuda = \
    hasty_pytorch_lib.e25_entmax_forward(
        True,  # training
        x.contiguous(),
        h_tape_init.contiguous(),
        h_work_init.contiguous(),
        cell.W_h.contiguous(),
        cell.W_x.contiguous(),
        cell.b_h.contiguous(),
        cell.W_write.contiguous()
    )

print(f"h_work_cuda sample: {h_work_all_cuda[0, 0, :4].float().tolist()}")

# Check forward match
print(f"Forward h_work diff: {(h_work_py.to(dtype) - h_work_all_cuda[0, 0]).abs().max().item():.6f}")

# Now run CUDA backward to get d_pre_act_all
# Need to call the backward function directly
d_h_work_out = torch.ones(B, T, D, device=device, dtype=dtype)
d_h_tape_final = torch.zeros(B, N, D, device=device, dtype=dtype)

dx_cuda, dW_h_cuda, dW_x_cuda, db_h_cuda, dW_write_cuda = \
    hasty_pytorch_lib.e25_entmax_backward(
        x.contiguous(),
        h_work_all_cuda.contiguous(),
        h_work_init.contiguous(),
        h_tape_all_cuda.contiguous(),
        read_attn_all_cuda.contiguous(),
        write_attn_all_cuda.contiguous(),
        cell.W_h.contiguous(),
        cell.W_x.contiguous(),
        cell.W_write.contiguous(),
        d_h_work_out.contiguous(),
        d_h_tape_final.contiguous()
    )

print(f"dW_x_cuda[0, :4]: {dW_x_cuda[0, :4].float().tolist()}")
print(f"dW_x_cuda norm: {dW_x_cuda.float().norm().item():.4f}")

# Compare
print("\n--- Comparison ---")
print(f"dW_x diff: {(dW_x_py.to(dtype) - dW_x_cuda).float().abs().max().item():.4f}")

# Let's also check dx
dx_py = d_pre_act_py @ cell.W_x.float()  # [D]
print(f"\ndx_py norm: {dx_py.norm().item():.4f}")
print(f"dx_cuda norm: {dx_cuda.float().norm().item():.4f}")
print(f"dx diff: {(dx_py.to(dtype) - dx_cuda[0, 0]).float().abs().max().item():.4f}")

# The issue is in d_pre_act computation. Let's verify by computing dW_x from scratch
# using the CUDA h_work and Python's formula
h_work_cuda = h_work_all_cuda[0, 0].float()  # [D]
d_pre_act_from_cuda_fwd = d_h_work * (1 - h_work_cuda ** 2)
print(f"\nd_pre_act from CUDA forward h_work:")
print(f"  norm: {d_pre_act_from_cuda_fwd.norm().item():.4f}")
print(f"  sample: {d_pre_act_from_cuda_fwd[:4].tolist()}")

# This should match d_pre_act_py (since h_work should be the same)
print(f"  diff from Python d_pre_act: {(d_pre_act_py - d_pre_act_from_cuda_fwd).abs().max().item():.6f}")

# Now compute dW_x using this d_pre_act
dW_x_check = d_pre_act_from_cuda_fwd[:, None] @ x[0, 0, None, :].float()  # [D, D]
print(f"\ndW_x computed from CUDA forward h_work:")
print(f"  sample: {dW_x_check[0, :4].tolist()}")
print(f"  diff from Python dW_x: {(dW_x_py - dW_x_check).abs().max().item():.6f}")
print(f"  diff from CUDA dW_x: {(dW_x_check.to(dtype) - dW_x_cuda).float().abs().max().item():.4f}")

print("\n" + "="*60)
