"""Debug E25 CUDA backward by checking intermediate values."""

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
print("E25 CUDA Backward Debug")
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

# Run CUDA forward
import hasty_pytorch_lib

h_work_all_cuda, h_tape_final_cuda, h_tape_all_cuda, read_attn_all_cuda, write_attn_all_cuda = \
    hasty_pytorch_lib.e25_entmax_forward(
        True,
        x.contiguous(),
        h_tape_init.contiguous(),
        h_work_init.contiguous(),
        cell.W_h.contiguous(),
        cell.W_x.contiguous(),
        cell.b_h.contiguous(),
        cell.W_write.contiguous()
    )

print("\nForward output:")
print(f"  h_work_all sample: {h_work_all_cuda[0, 0, :4].float().tolist()}")

# Manual computation of d_pre_act
d_h_work_out = torch.ones(B, T, D, device=device, dtype=dtype)
h_work_t = h_work_all_cuda[0, 0].float()  # [D]
d_pre_act_expected = d_h_work_out[0, 0].float() * (1 - h_work_t ** 2)
print(f"\nExpected d_pre_act (manual):")
print(f"  norm: {d_pre_act_expected.norm().item():.4f}")
print(f"  sample: {d_pre_act_expected[:4].tolist()}")

# Expected dW_x = d_pre_act^T @ x
dW_x_expected = d_pre_act_expected[:, None] @ x[0, 0, None, :].float()
print(f"\nExpected dW_x (manual):")
print(f"  norm: {dW_x_expected.norm().item():.4f}")
print(f"  [0, :4]: {dW_x_expected[0, :4].tolist()}")

# Call CUDA backward
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

print(f"\nCUDA dW_x:")
print(f"  norm: {dW_x_cuda.float().norm().item():.4f}")
print(f"  [0, :4]: {dW_x_cuda[0, :4].float().tolist()}")

# Let's check dx as a proxy for d_pre_act
# dx = d_pre_act @ W_x
# If d_pre_act is correct, then dx should be correct

dx_expected = d_pre_act_expected @ cell.W_x.float()  # [D]
print(f"\nExpected dx (manual):")
print(f"  norm: {dx_expected.norm().item():.4f}")
print(f"  [:4]: {dx_expected[:4].tolist()}")

print(f"\nCUDA dx:")
print(f"  norm: {dx_cuda.float().norm().item():.4f}")
print(f"  [0, 0, :4]: {dx_cuda[0, 0, :4].float().tolist()}")

print(f"\ndx diff: {(dx_expected.to(dtype) - dx_cuda[0, 0]).float().abs().max().item():.4f}")

# Let me also check if the issue is with how dW_x is being accumulated
# If d_pre_act is wrong but x is correct, then the product will be wrong
# Let me compute d_pre_act from CUDA's dx by solving dx = d_pre_act @ W_x

# d_pre_act = dx @ W_x^{-1}... but W_x might not be invertible
# Instead, let me check the relationship between dx and d_pre_act

# Actually, let's just check if CUDA backward is returning the right tensors
print("\n--- Tensor Analysis ---")
print(f"x shape: {x.shape}")
print(f"x sample: {x[0, 0, :4].float().tolist()}")

# Check if there's an indexing issue
# The x_t in CUDA pybind is x.permute({1, 0, 2}) = [T, B, D] = [1, 1, D]
x_t = x.permute(1, 0, 2).contiguous()
print(f"x_t shape: {x_t.shape}")
print(f"x_t sample: {x_t[0, 0, :4].float().tolist()}")  # Should match x[0, 0, :4]

# If dW_x = d_pre_act^T @ x, and x is correct, let's back-calculate d_pre_act
# dW_x[i, j] = d_pre_act[i] * x[j] (for batch=1)
# So d_pre_act[0] = dW_x[0, j] / x[j] for any non-zero x[j]

# Find non-zero x values
x_vals = x[0, 0].float()
nonzero_idx = torch.where(x_vals.abs() > 0.1)[0][:4]
print(f"\nNon-zero x indices: {nonzero_idx.tolist()}")
print(f"x values at those indices: {x_vals[nonzero_idx].tolist()}")

dW_x_vals = dW_x_cuda.float()
for i in nonzero_idx.tolist()[:2]:
    inferred_d_pre_act_0 = dW_x_vals[0, i] / x_vals[i]
    print(f"  Inferred d_pre_act[0] from dW_x[0, {i}] / x[{i}] = {inferred_d_pre_act_0.item():.4f}")

print(f"Expected d_pre_act[0] = {d_pre_act_expected[0].item():.4f}")

print("\n" + "="*60)
