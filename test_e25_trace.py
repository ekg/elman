"""Trace backward pass comparing Python and CUDA."""

import torch
import math

torch.manual_seed(42)

B = 1
T = 1  # Single timestep for simplicity
D = 256
N = 8

device = 'cuda'

from elman.models.e25_entmax import (
    e25_forward_step_python,
    e25_backward_python,
    entmax_1_5_backward,
)

print("="*60)
print("Tracing E25 backward pass")
print("="*60)

# Create simple test case
x = torch.randn(B, D, device=device, dtype=torch.float32)
h_tape = torch.randn(B, N, D, device=device, dtype=torch.float32) * 0.1
h_work = torch.randn(B, D, device=device, dtype=torch.float32) * 0.1

W_h = torch.randn(D, D, device=device, dtype=torch.float32) * 0.1
W_x = torch.randn(D, D, device=device, dtype=torch.float32) * 0.1
b_h = torch.zeros(D, device=device, dtype=torch.float32)
W_write = torch.randn(D, D, device=device, dtype=torch.float32) * 0.1

scale = 1.0 / math.sqrt(D)

print("\n1. Forward pass (Python)...")
h_work_new, h_tape_new, read_attn, write_attn, read = e25_forward_step_python(
    x, h_tape, h_work,
    W_h, W_x, b_h, W_write, scale
)

print(f"   h_work_new sample: {h_work_new[0, :4].tolist()}")
print(f"   read_attn: {read_attn[0].tolist()}")
print(f"   write_attn: {write_attn[0].tolist()}")

print("\n2. Backward pass (Python)...")
# Upstream gradients
d_h_work = torch.ones_like(h_work_new)  # Gradient from loss
d_h_tape = torch.zeros_like(h_tape_new)  # No gradient on final tape

# Build tensors for backward
h_work_all = h_work_new.unsqueeze(1)  # [B, 1, D]
h_tape_all = torch.stack([h_tape, h_tape_new], dim=1)  # [B, 2, N, D]
x_seq = x.unsqueeze(1)  # [B, 1, D]
read_attn_all = read_attn.unsqueeze(1)  # [B, 1, N]
write_attn_all = write_attn.unsqueeze(1)  # [B, 1, N]
d_h_work_all = d_h_work.unsqueeze(1)  # [B, 1, D]

# Detailed backward computation
print("\n   === BACKWARD THROUGH TAPE WRITE ===")
d_write_value = (d_h_tape * write_attn[:, :, None]).sum(dim=1)
print(f"   d_write_value norm: {d_write_value.norm().item():.6f}")

write_value = h_work_new @ W_write.T
d_write_attn = (d_h_tape * (write_value[:, None, :] - h_tape)).sum(dim=-1)
print(f"   d_write_attn: {d_write_attn[0].tolist()}")

d_h_tape_pre_write = d_h_tape * (1 - write_attn[:, :, None])

# Entmax backward for write attention
d_write_scores = entmax_1_5_backward(write_attn, d_write_attn, dim=-1)
d_write_scores = d_write_scores * scale
print(f"   d_write_scores: {d_write_scores[0].tolist()}")

d_h_work_from_write = d_write_value @ W_write
d_h_work_from_write_attn = (d_write_scores[:, :, None] * h_tape).sum(dim=1)
d_h_tape_from_write_attn = d_write_scores[:, :, None] * h_work_new[:, None, :]

print(f"   d_h_work_from_write norm: {d_h_work_from_write.norm().item():.6f}")
print(f"   d_h_work_from_write_attn norm: {d_h_work_from_write_attn.norm().item():.6f}")

d_h_work_t_total = d_h_work + d_h_work_from_write + d_h_work_from_write_attn
print(f"   d_h_work_t_total norm: {d_h_work_t_total.norm().item():.6f}")

print("\n   === BACKWARD THROUGH WORKING MEMORY UPDATE ===")
d_pre_act = d_h_work_t_total * (1 - h_work_new ** 2)
print(f"   d_pre_act norm: {d_pre_act.norm().item():.6f}")
print(f"   d_pre_act sample: {d_pre_act[0, :4].tolist()}")

# dW_x for this timestep
dW_x = d_pre_act.T @ x
print(f"   dW_x (this timestep) norm: {dW_x.norm().item():.6f}")
print(f"   dW_x sample: {dW_x[0, :4].tolist()}")

# dx for this timestep
dx = d_pre_act @ W_x
print(f"   dx norm: {dx.norm().item():.6f}")

print("\n   === BACKWARD THROUGH READ ===")
d_read_attn = (d_pre_act[:, None, :] * h_tape).sum(dim=-1)
print(f"   d_read_attn: {d_read_attn[0].tolist()}")

d_read_scores = entmax_1_5_backward(read_attn, d_read_attn, dim=-1)
d_read_scores = d_read_scores * scale
print(f"   d_read_scores: {d_read_scores[0].tolist()}")

print("\n" + "="*60)
print("Done!")
