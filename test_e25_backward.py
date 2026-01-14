"""Focused backward pass comparison for E25."""

import torch
import math

torch.manual_seed(42)

# Test parameters
B = 2
T = 4
D = 256  # Use 256 since CUDA doesn't support dim=64
N = 8

device = 'cuda'
dtype = torch.bfloat16

print("="*60)
print("E25 Backward Pass Debug")
print("="*60)

from elman.models.e25_entmax import (
    e25_sequence_python,
    e25_backward_python,
    E25DualMemoryElmanCell,
    entmax_1_5_backward
)

# Create test inputs
x_seq = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
h_tape_init = torch.zeros(B, N, D, device=device, dtype=dtype)
h_work_init = torch.zeros(B, D, device=device, dtype=dtype)

# Create weights
W_h = torch.randn(D, D, device=device, dtype=dtype) * 0.1
W_x = torch.randn(D, D, device=device, dtype=dtype) * 0.1
b_h = torch.zeros(D, device=device, dtype=dtype)
W_write = torch.randn(D, D, device=device, dtype=dtype) * 0.1

scale = 1.0 / math.sqrt(D)

print("\n1. Running Python forward...")
h_work_all_py, h_tape_final_py, _, read_attn_py, write_attn_py = e25_sequence_python(
    x_seq.float().detach(),
    h_tape_init.float(),
    h_work_init.float(),
    W_h.float(), W_x.float(), b_h.float(), W_write.float()
)

# Stack attention for backward
read_attn_stack = torch.stack(read_attn_py, dim=1)  # [B, T, N]
write_attn_stack = torch.stack(write_attn_py, dim=1)  # [B, T, N]

# Create fake gradients
d_h_work_all = torch.randn(B, T, D, device=device, dtype=torch.float32)
d_h_tape_final = torch.randn(B, N, D, device=device, dtype=torch.float32)

# Build h_tape_all - need all timesteps
h_tape_list = [h_tape_init.float()]
h_tape = h_tape_init.float().clone()
h_work = h_work_init.float().clone()
for t in range(T):
    from elman.models.e25_entmax import e25_forward_step_python
    h_work, h_tape, _, _, _ = e25_forward_step_python(
        x_seq[:, t].float(),
        h_tape, h_work,
        W_h.float(), W_x.float(), b_h.float(), W_write.float(),
        scale
    )
    h_tape_list.append(h_tape)
h_tape_all = torch.stack(h_tape_list, dim=1)  # [B, T+1, N, D]

print(f"   h_work_all shape: {h_work_all_py.shape}")
print(f"   h_tape_all shape: {h_tape_all.shape}")
print(f"   read_attn shape: {read_attn_stack.shape}")

print("\n2. Running Python backward...")
dx_py, dW_h_py, dW_x_py, db_h_py, dW_write_py = e25_backward_python(
    x_seq.float().detach(),
    h_work_all_py,
    h_tape_all,
    read_attn_stack,
    write_attn_stack,
    W_h.float(), W_x.float(), b_h.float(), W_write.float(),
    h_work_init.float(),
    d_h_work_all,
    d_h_tape_final,
    scale
)

print(f"   dx norm: {dx_py.norm().item():.4f}")
print(f"   dW_h norm: {dW_h_py.norm().item():.4f}")
print(f"   dW_x norm: {dW_x_py.norm().item():.4f}")
print(f"   db_h norm: {db_h_py.norm().item():.4f}")
print(f"   dW_write norm: {dW_write_py.norm().item():.4f}")

print("\n3. Running CUDA backward...")
import hasty_pytorch_lib

# CUDA expects inputs in specific format
# Note: h_tape_all is [T+1, B, N, D] for CUDA kernel!
h_work_all_for_cuda = h_work_all_py.to(dtype).contiguous()
h_tape_all_for_cuda = h_tape_all.permute(1, 0, 2, 3).to(dtype).contiguous()  # [T+1, B, N, D]
read_attn_for_cuda = read_attn_stack.to(dtype).contiguous()
write_attn_for_cuda = write_attn_stack.to(dtype).contiguous()

# IMPORTANT: The Python code passes x_proj = x @ W_x.T to CUDA backward
# But then CUDA uses this to compute dW_x as if it were the original x!
x_proj = (x_seq @ W_x.T).contiguous()

dx_cuda, dW_h_cuda, dW_x_cuda, db_h_cuda, dW_write_cuda = hasty_pytorch_lib.e25_entmax_backward(
    x_proj,  # This is x_proj, not x_seq!
    h_work_all_for_cuda,
    h_work_init.contiguous(),
    h_tape_all_for_cuda,
    read_attn_for_cuda,
    write_attn_for_cuda,
    W_h.contiguous(),
    W_x.contiguous(),
    W_write.contiguous(),
    d_h_work_all.to(dtype).contiguous(),
    d_h_tape_final.to(dtype).contiguous()
)

print(f"   dx norm: {dx_cuda.float().norm().item():.4f}")
print(f"   dW_h norm: {dW_h_cuda.float().norm().item():.4f}")
print(f"   dW_x norm: {dW_x_cuda.float().norm().item():.4f}")
print(f"   db_h norm: {db_h_cuda.float().norm().item():.4f}")
print(f"   dW_write norm: {dW_write_cuda.float().norm().item():.4f}")

print("\n4. Comparing gradients...")
def compare(name, py, cuda):
    diff = (py.float() - cuda.float()).abs()
    rel_diff = diff / (py.float().abs() + 1e-8)
    print(f"   {name}: max_diff={diff.max().item():.6f}, rel_max={rel_diff.max().item():.4f}")

compare("dx", dx_py, dx_cuda)
compare("dW_h", dW_h_py, dW_h_cuda)
compare("dW_x", dW_x_py, dW_x_cuda)
compare("db_h", db_h_py, db_h_cuda)
compare("dW_write", dW_write_py, dW_write_cuda)

# Check if dW_x is particularly bad
print("\n5. Investigating dW_x discrepancy...")
print(f"   Python dW_x sample: {dW_x_py[:2, :2].tolist()}")
print(f"   CUDA dW_x sample: {dW_x_cuda.float()[:2, :2].tolist()}")

# What CUDA is computing: dW_x = sum_t (d_pre_act_t^T @ x_proj_t)
# What should be computed: dW_x = sum_t (d_pre_act_t^T @ x_t)

print("\n6. Testing if the issue is x vs x_proj...")
# Try calling CUDA backward with x_seq instead of x_proj
try:
    dx_test, dW_h_test, dW_x_test, db_h_test, dW_write_test = hasty_pytorch_lib.e25_entmax_backward(
        x_seq.contiguous(),  # Use x_seq instead of x_proj
        h_work_all_for_cuda,
        h_work_init.contiguous(),
        h_tape_all_for_cuda,
        read_attn_for_cuda,
        write_attn_for_cuda,
        W_h.contiguous(),
        W_x.contiguous(),
        W_write.contiguous(),
        d_h_work_all.to(dtype).contiguous(),
        d_h_tape_final.to(dtype).contiguous()
    )
    compare("dW_x (with x_seq)", dW_x_py, dW_x_test)
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "="*60)
print("Done!")
print("="*60)
