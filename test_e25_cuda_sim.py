"""Simulate CUDA backward in Python to understand the discrepancy."""

import torch
import math

torch.manual_seed(42)

B = 1
T = 1
D = 256
N = 8

device = 'cuda'
dtype = torch.bfloat16

from elman.models.e25_entmax import E25DualMemoryElmanCell, entmax_1_5, entmax_1_5_backward

cell = E25DualMemoryElmanCell(dim=D, n_slots=N).to(device).to(dtype)
x = torch.randn(B, T, D, device=device, dtype=dtype)

h_tape_init = torch.zeros(B, N, D, device=device, dtype=dtype)
h_work_init = torch.zeros(B, D, device=device, dtype=dtype)

scale = 1.0 / math.sqrt(D)

print("="*60)
print("Simulating CUDA backward in Python")
print("="*60)

# Run CUDA forward to get the saved tensors
import hasty_pytorch_lib

h_work_all, h_tape_final, h_tape_all, read_attn_all, write_attn_all = \
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

print(f"h_work_all shape: {h_work_all.shape}")  # [B, T, D]
print(f"h_tape_all shape: {h_tape_all.shape}")  # [T+1, B, N, D]
print(f"read_attn_all shape: {read_attn_all.shape}")  # [B, T, N]
print(f"write_attn_all shape: {write_attn_all.shape}")  # [B, T, N]

# Upstream gradients
d_h_work_out = torch.ones_like(h_work_all)  # [B, T, D]
d_h_tape_final = torch.zeros(B, N, D, device=device, dtype=dtype)

print("\n--- Simulating CUDA backward ---")
print("(Following the same flow as the CUDA kernel)")

# Convert to float for simulation
h_work_t = h_work_all[:, 0].float()  # [B, D]
h_tape_t = h_tape_all[0].float()  # [B, N, D] - tape before write
write_attn = write_attn_all[:, 0].float()  # [B, N]
read_attn = read_attn_all[:, 0].float()  # [B, N]
d_h_work_out_t = d_h_work_out[:, 0].float()  # [B, D]

W_h = cell.W_h.float()
W_x = cell.W_x.float()
W_write = cell.W_write.float()

# Initialize
d_h_tape = d_h_tape_final.float()  # [B, N, D]
d_h_work = torch.zeros(B, D, device=device, dtype=torch.float32)  # [B, D]

# Phase 1: Write attention backward
print("\nPhase 1: Write attention backward")

# Step 1: d_h_work += d_h_work_out_t
d_h_work = d_h_work + d_h_work_out_t
print(f"  After Step 1, d_h_work norm: {d_h_work.norm().item():.4f}")

# Step 2: d_write_val = sum_n(attn[n] * d_h_tape[n])
d_write_val = (d_h_tape * write_attn[:, :, None]).sum(dim=1)
print(f"  d_write_val norm: {d_write_val.norm().item():.4f}")

# Step 3: d_write_attn = sum_d(d_h_tape * (write_val - h_tape))
write_val = h_work_t @ W_write.T
d_write_attn = (d_h_tape * (write_val[:, None, :] - h_tape_t)).sum(dim=-1)
print(f"  d_write_attn: {d_write_attn[0].tolist()}")

# Step 4: Entmax backward for write attention
g = torch.sqrt(torch.clamp(write_attn, min=0))
g_dp_sum = (g * d_write_attn).sum(dim=-1, keepdim=True)
g_sum = g.sum(dim=-1, keepdim=True).clamp(min=1e-9)
d_write_scores = g * (d_write_attn - g_dp_sum / g_sum) * scale
print(f"  d_write_scores: {d_write_scores[0].tolist()}")

# Step 5: d_h_work += sum_n(d_write_scores[n] * h_tape[n])
d_h_work_from_write_attn = (d_write_scores[:, :, None] * h_tape_t).sum(dim=1)
d_h_work = d_h_work + d_h_work_from_write_attn
print(f"  After Step 5, d_h_work norm: {d_h_work.norm().item():.4f}")

# Step 6: d_h_tape = (1-attn) * d_h_tape + d_write_scores * h_work
d_h_tape_from_write_attn = d_write_scores[:, :, None] * h_work_t[:, None, :]
d_h_tape = (1 - write_attn[:, :, None]) * d_h_tape + d_h_tape_from_write_attn

# cuBLAS: d_h_work += W_write.T @ d_write_val
d_h_work_from_write = d_write_val @ W_write
d_h_work = d_h_work + d_h_work_from_write
print(f"  After cuBLAS, d_h_work norm: {d_h_work.norm().item():.4f}")

# Phase 2: Tanh backward
print("\nPhase 2: Tanh backward")
d_pre_act = d_h_work * (1 - h_work_t ** 2)
print(f"  d_pre_act norm: {d_pre_act.norm().item():.4f}")
print(f"  d_pre_act sample: {d_pre_act[0, :4].tolist()}")

# cuBLAS: d_h_work = W_h.T @ d_pre_act (for next timestep)
d_h_work = d_pre_act @ W_h

# Phase 3: Read attention backward
print("\nPhase 3: Read attention backward")
# ... (similar to write attention)

# Compute dW_x = d_pre_act.T @ x
dW_x = d_pre_act.T @ x[:, 0].float()
print(f"\nSimulated dW_x norm: {dW_x.norm().item():.4f}")
print(f"Simulated dW_x sample: {dW_x[0, :4].tolist()}")

# Compare with actual CUDA backward
print("\n--- Actual CUDA backward ---")
cell.zero_grad()
x_grad = x.clone().requires_grad_(True)
out, _, _ = cell(x_grad, use_cuda=True)
loss = out.sum()
loss.backward()

print(f"CUDA dW_x norm: {cell.W_x.grad.float().norm().item():.4f}")
print(f"CUDA dW_x sample: {cell.W_x.grad[0, :4].float().tolist()}")

print(f"\nDifference: {(dW_x.to(dtype) - cell.W_x.grad).abs().max().item():.4f}")
