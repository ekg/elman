"""Test E25 backward phase by phase to find the bug."""

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
print("E25 Phase-by-Phase Backward Test")
print("="*60)

from elman.models.e25_entmax import E25DualMemoryElmanCell, entmax_1_5, entmax_1_5_backward

# Create cell
torch.manual_seed(42)
cell = E25DualMemoryElmanCell(dim=D, n_slots=N).to(device).to(dtype)

# Input
torch.manual_seed(123)
x = torch.randn(B, T, D, device=device, dtype=dtype)
h_tape_init = torch.zeros(B, N, D, device=device, dtype=dtype)
h_work_init = torch.zeros(B, D, device=device, dtype=dtype)

scale = 1.0 / math.sqrt(D)

# Run CUDA forward to get saved tensors
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

print("\nForward output shapes:")
print(f"  h_work_all: {h_work_all_cuda.shape}")
print(f"  h_tape_all: {h_tape_all_cuda.shape}")
print(f"  read_attn_all: {read_attn_all_cuda.shape}")
print(f"  write_attn_all: {write_attn_all_cuda.shape}")

# Get values for manual backward
h_work_t = h_work_all_cuda[0, 0].float()  # [D] - h_work after tanh at t=0
h_tape_t = h_tape_all_cuda[0, 0].float()  # [N, D] - tape before t=0
read_attn_t = read_attn_all_cuda[0, 0].float()  # [N]
write_attn_t = write_attn_all_cuda[0, 0].float()  # [N]

W_h = cell.W_h.float()
W_x = cell.W_x.float()
W_write = cell.W_write.float()

print("\nForward values:")
print(f"  h_work_t sample: {h_work_t[:4].tolist()}")
print(f"  read_attn: {read_attn_t.tolist()}")
print(f"  write_attn: {write_attn_t.tolist()}")

# Upstream gradient: ones
d_h_work_out = torch.ones(B, T, D, device=device, dtype=dtype)
d_h_tape_final = torch.zeros(B, N, D, device=device, dtype=dtype)

# === MANUAL PYTHON BACKWARD ===
print("\n--- Manual Python Backward ---")

# Initialize
d_h_tape = d_h_tape_final[0].float()  # [N, D]
d_h_work = torch.zeros(D, device=device, dtype=torch.float32)  # [D]

# Phase 1: Write backward
print("\nPhase 1: Write backward")

# Step 1: d_h_work += d_h_work_out[t]
d_h_work = d_h_work + d_h_work_out[0, 0].float()  # [D]
print(f"  After step 1, d_h_work norm: {d_h_work.norm().item():.4f}")

# Compute write_value
write_value = h_work_t @ W_write.T  # [D]
print(f"  write_value sample: {write_value[:4].tolist()}")

# d_write_value = sum_n(write_attn[n] * d_h_tape[n])
d_write_value = (write_attn_t[:, None] * d_h_tape).sum(dim=0)  # [D]
print(f"  d_write_value norm: {d_write_value.norm().item():.4f}")

# d_write_attn = sum_d(d_h_tape * (write_value - h_tape_t))
d_write_attn = (d_h_tape * (write_value[None, :] - h_tape_t)).sum(dim=-1)  # [N]
print(f"  d_write_attn: {d_write_attn.tolist()}")

# d_h_tape_pre_write = d_h_tape * (1 - write_attn)
d_h_tape_pre_write = d_h_tape * (1 - write_attn_t[:, None])  # [N, D]

# Entmax backward for write attention
g = torch.sqrt(torch.clamp(write_attn_t, min=0))  # [N]
g_dp_sum = (g * d_write_attn).sum()
g_sum = g.sum().clamp(min=1e-9)
d_write_scores = g * (d_write_attn - g_dp_sum / g_sum) * scale  # [N]
print(f"  d_write_scores: {d_write_scores.tolist()}")

# d_h_work_from_write_attn = sum_n(d_write_scores[n] * h_tape_t[n])
d_h_work_from_write_attn = (d_write_scores[:, None] * h_tape_t).sum(dim=0)  # [D]
print(f"  d_h_work_from_write_attn norm: {d_h_work_from_write_attn.norm().item():.4f}")

# d_h_tape_from_write_attn = d_write_scores * h_work_t
d_h_tape_from_write_attn = d_write_scores[:, None] * h_work_t[None, :]  # [N, D]

# After Phase 1: d_h_work += W_write.T @ d_write_value
d_h_work_from_write = d_write_value @ W_write  # [D]
print(f"  d_h_work_from_write norm: {d_h_work_from_write.norm().item():.4f}")

d_h_work = d_h_work + d_h_work_from_write_attn + d_h_work_from_write
print(f"  After Phase 1, d_h_work norm: {d_h_work.norm().item():.4f}")

# Update d_h_tape after Phase 1 (for comparison with CUDA which does this in kernel)
d_h_tape_after_phase1 = d_h_tape_pre_write + d_h_tape_from_write_attn

# Phase 2: Tanh backward
print("\nPhase 2: Tanh backward")
d_pre_act = d_h_work * (1 - h_work_t ** 2)  # [D]
print(f"  d_pre_act norm: {d_pre_act.norm().item():.4f}")
print(f"  d_pre_act sample: {d_pre_act[:4].tolist()}")

# After Phase 2: d_h_work = W_h.T @ d_pre_act (for next timestep)
d_h_work_next = d_pre_act @ W_h  # [D]

# Phase 3: Read backward
print("\nPhase 3: Read backward")

# d_read = d_pre_act (read flows into tanh)
d_read = d_pre_act  # [D]

# d_read_attn = (d_read[:, None] @ h_tape_t^T) -> sum over D for each slot
d_read_attn = (d_read[None, :] * h_tape_t).sum(dim=-1)  # [N]
print(f"  d_read_attn: {d_read_attn.tolist()}")

# d_h_tape_from_read = d_read * read_attn
d_h_tape_from_read = read_attn_t[:, None] * d_read[None, :]  # [N, D]

# Entmax backward for read attention
g_read = torch.sqrt(torch.clamp(read_attn_t, min=0))  # [N]
g_dp_sum_read = (g_read * d_read_attn).sum()
g_sum_read = g_read.sum().clamp(min=1e-9)
d_read_scores = g_read * (d_read_attn - g_dp_sum_read / g_sum_read) * scale  # [N]
print(f"  d_read_scores: {d_read_scores.tolist()}")

# d_h_work_from_read_attn = sum_n(d_read_scores[n] * h_tape_t[n])
d_h_work_from_read_attn = (d_read_scores[:, None] * h_tape_t).sum(dim=0)  # [D]
print(f"  d_h_work_from_read_attn norm: {d_h_work_from_read_attn.norm().item():.4f}")

# d_h_tape_from_read_attn = d_read_scores * h_work_prev (h_work_init for t=0)
d_h_tape_from_read_attn = d_read_scores[:, None] * h_work_init[0, None, :].float()  # [N, D]

# Final d_h_tape
d_h_tape_final_py = d_h_tape_after_phase1 + d_h_tape_from_read + d_h_tape_from_read_attn
d_h_work_final_py = d_h_work_next + d_h_work_from_read_attn

print(f"\nFinal d_h_work norm: {d_h_work_final_py.norm().item():.4f}")
print(f"Final d_h_tape norm: {d_h_tape_final_py.norm().item():.4f}")

# Compute dW_x = d_pre_act.T @ x
dW_x_py = d_pre_act[:, None] @ x[0, 0, None, :].float()  # [D, D]
print(f"dW_x norm (Python): {dW_x_py.norm().item():.4f}")
print(f"dW_x sample (Python): {dW_x_py[0, :4].tolist()}")

# === CUDA BACKWARD ===
print("\n--- CUDA Backward ---")

cell.zero_grad()
x_cuda = x.clone().requires_grad_(True)
h_work_all_cuda2, h_tape_cuda2, h_work_final_cuda2 = cell(x_cuda, use_cuda=True)
loss = h_work_all_cuda2.sum()
loss.backward()

print(f"dW_x norm (CUDA): {cell.W_x.grad.float().norm().item():.4f}")
print(f"dW_x sample (CUDA): {cell.W_x.grad[0, :4].float().tolist()}")
print(f"dx norm (CUDA): {x_cuda.grad.float().norm().item():.4f}")

# === Compare ===
print("\n--- Comparison ---")
print(f"dW_x diff: {(dW_x_py.to(dtype) - cell.W_x.grad).float().abs().max().item():.4f}")

print("\n" + "="*60)
