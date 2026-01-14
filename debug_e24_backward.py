#!/usr/bin/env python3
"""
Debug E24 backward pass - trace through one iteration
"""
import torch
import math
import sys
sys.path.insert(0, 'elman/cuda')

from elman.models.e24_single_gemm import e24_forward_step_python

device = 'cuda'
dtype = torch.bfloat16

B = 2
T = 2  # Short sequence for debugging
D = 256
N = 8

torch.manual_seed(42)

# Create weights
W_all = torch.randn(2*D, 2*D, device=device, dtype=dtype) * 0.1
b_h = torch.zeros(D, device=device, dtype=dtype)
scale = 1.0 / math.sqrt(D)

# Create input
x_seq = torch.randn(B, T, D, device=device, dtype=dtype)
h_tape_init = torch.zeros(B, N, D, device=device, dtype=dtype)
h_work_init = torch.zeros(B, D, device=device, dtype=dtype)

# Forward pass - collect intermediates
h_tape = h_tape_init.clone()
h_work = h_work_init.clone()

h_work_list = []
h_tape_list = [h_tape_init.clone()]
read_attn_list = []
write_attn_list = []
write_val_list = []

for t in range(T):
    h_work, h_tape, read_attn, write_attn, _, write_val = e24_forward_step_python(
        x_seq[:, t], h_tape, h_work, W_all, b_h, scale
    )
    h_work_list.append(h_work)
    h_tape_list.append(h_tape.clone())
    read_attn_list.append(read_attn)
    write_attn_list.append(write_attn)
    write_val_list.append(write_val)

h_work_all = torch.stack(h_work_list, dim=1)  # [B, T, D]
h_tape_all = torch.stack(h_tape_list, dim=1)  # [B, T+1, N, D]
read_attn_all = torch.stack(read_attn_list, dim=1)  # [B, T, N]
write_attn_all = torch.stack(write_attn_list, dim=1)  # [B, T, N]
write_val_all = torch.stack(write_val_list, dim=1)  # [B, T, D]

print("Forward pass complete")
print(f"h_work_all shape: {h_work_all.shape}")

# Manual backward pass - trace step by step
d_h_work_out = torch.ones_like(h_work_all)  # [B, T, D]
d_h_tape_final = torch.zeros(B, N, D, device=device, dtype=dtype)

# Initialize gradients
dx = torch.zeros_like(x_seq)
d_h_tape = d_h_tape_final.clone()  # [B, N, D]
d_h_work = torch.zeros(B, D, device=device, dtype=dtype)  # Accumulated gradient

print("\n" + "=" * 60)
print("Manual backward pass (Python reference)")
print("=" * 60)

for t in range(T - 1, -1, -1):
    print(f"\n--- Timestep t={t} ---")

    # Get saved tensors for this timestep
    h_work_t = h_work_all[:, t]  # [B, D] - h_work after update at t
    h_tape_t = h_tape_all[:, t]  # [B, N, D] - h_tape BEFORE update at t
    read_attn = read_attn_all[:, t]  # [B, N]
    write_attn = write_attn_all[:, t]  # [B, N]
    write_val = write_val_all[:, t]  # [B, D]

    # h_work_prev
    if t > 0:
        h_work_prev = h_work_all[:, t - 1]
    else:
        h_work_prev = h_work_init

    # Incoming gradients for this timestep
    d_h_work_in = d_h_work_out[:, t].clone()
    print(f"d_h_work_out[t] norm: {d_h_work_in.float().norm().item():.4f}")
    print(f"d_h_work (accumulated) norm: {d_h_work.float().norm().item():.4f}")

    d_h_work_t = d_h_work_in + d_h_work  # [B, D]
    print(f"d_h_work_t (combined) norm: {d_h_work_t.float().norm().item():.4f}")

    # === BACKWARD THROUGH STEP 3: TAPE WRITE ===
    d_write_val = (d_h_tape * write_attn[:, :, None]).sum(dim=1)  # [B, D]
    d_write_attn = (d_h_tape * (write_val[:, None, :] - h_tape_t)).sum(dim=-1)  # [B, N]
    d_h_tape_pre_write = d_h_tape * (1 - write_attn[:, :, None])  # [B, N, D]

    # Softmax backward
    d_write_scores = write_attn * (d_write_attn - (d_write_attn * write_attn).sum(dim=-1, keepdim=True))
    d_write_scores = d_write_scores * scale

    # Gradient w.r.t. h_work_new from write attention
    d_h_work_from_write_attn = (d_write_scores[:, :, None] * h_tape_t).sum(dim=1)  # [B, D]
    d_h_tape_from_write_attn = d_write_scores[:, :, None] * h_work_t[:, None, :]  # [B, N, D]

    # Total gradient to h_work_t (before tanh backward)
    d_h_work_t_total = d_h_work_t + d_h_work_from_write_attn
    print(f"d_write_val norm: {d_write_val.float().norm().item():.4f}")
    print(f"d_h_work_from_write_attn norm: {d_h_work_from_write_attn.float().norm().item():.4f}")
    print(f"d_h_work_t_total norm: {d_h_work_t_total.float().norm().item():.4f}")

    # === BACKWARD THROUGH STEP 2: WORKING MEMORY UPDATE ===
    # tanh backward
    d_pre_act = d_h_work_t_total * (1 - h_work_t.float() ** 2).to(dtype)  # [B, D]
    d_h_update = d_pre_act  # [B, D]
    d_read = d_pre_act  # [B, D]
    print(f"d_pre_act (d_h_update) norm: {d_pre_act.float().norm().item():.4f}")

    # === BACKWARD THROUGH STEP 1: READ FROM TAPE ===
    d_read_attn = (d_read[:, None, :] * h_tape_t).sum(dim=-1)  # [B, N]
    d_h_tape_from_read = d_read[:, None, :] * read_attn[:, :, None]  # [B, N, D]

    # Softmax backward
    d_read_scores = read_attn * (d_read_attn - (d_read_attn * read_attn).sum(dim=-1, keepdim=True))
    d_read_scores = d_read_scores * scale

    # Gradient w.r.t. h_work_prev from read attention
    d_h_work_from_read_attn = (d_read_scores[:, :, None] * h_tape_t).sum(dim=1)  # [B, D]
    d_h_tape_from_read_attn = d_read_scores[:, :, None] * h_work_prev[:, None, :]  # [B, N, D]
    print(f"d_h_work_from_read_attn norm: {d_h_work_from_read_attn.float().norm().item():.4f}")

    # === BACKWARD THROUGH STEP 0: THE SINGLE GEMM ===
    d_output = torch.cat([d_h_update, d_write_val], dim=-1)  # [B, 2D]
    input_concat = torch.cat([h_work_prev, x_seq[:, t]], dim=-1)  # [B, 2D]

    # Gradient w.r.t. input_concat
    d_input_concat = d_output @ W_all  # [B, 2D]
    d_h_work_from_gemm = d_input_concat[:, :D]  # [B, D]
    dx[:, t] = d_input_concat[:, D:]  # [B, D]
    print(f"d_output norm: {d_output.float().norm().item():.4f}")
    print(f"d_input_concat norm: {d_input_concat.float().norm().item():.4f}")
    print(f"d_h_work_from_gemm norm: {d_h_work_from_gemm.float().norm().item():.4f}")
    print(f"dx[t] norm: {dx[:, t].float().norm().item():.4f}")

    # Total gradient to h_work_prev - THIS IS THE KEY
    # This becomes d_h_work for the next iteration (t-1)
    d_h_work = d_h_work_from_gemm + d_h_work_from_read_attn
    print(f"d_h_work (for next iteration) norm: {d_h_work.float().norm().item():.4f}")

    # Total gradient to h_tape at t
    d_h_tape = d_h_tape_pre_write + d_h_tape_from_write_attn + d_h_tape_from_read + d_h_tape_from_read_attn

print("\n" + "=" * 60)
print("Final dx gradient norms")
print("=" * 60)
for t in range(T):
    print(f"dx[{t}] norm: {dx[:, t].float().norm().item():.4f}")
