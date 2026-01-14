"""Test E25 Python backward against autograd."""

import torch
import torch.nn as nn
import math

torch.manual_seed(42)

B = 2
T = 4
D = 32
N = 8

device = 'cuda'
dtype = torch.float64  # Use float64 for numerical accuracy

from elman.models.e25_entmax import (
    entmax_1_5_forward, entmax_1_5_backward, entmax_1_5,
    e25_forward_step_python, e25_backward_python
)

print("="*60)
print("E25 Python Backward vs Autograd Test")
print("="*60)

# Create parameters as leaf tensors
W_h = nn.Parameter(torch.randn(D, D, device=device, dtype=dtype) * 0.1)
W_x = nn.Parameter(torch.randn(D, D, device=device, dtype=dtype) * 0.1)
b_h = nn.Parameter(torch.zeros(D, device=device, dtype=dtype))
W_write = nn.Parameter(torch.randn(D, D, device=device, dtype=dtype) * 0.1)
x_seq = nn.Parameter(torch.randn(B, T, D, device=device, dtype=dtype))

h_tape_init = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
h_work_init = torch.randn(B, D, device=device, dtype=dtype) * 0.1

scale = 1.0 / math.sqrt(D)

# Forward pass using autograd-trackable entmax
def forward_autograd(x_seq, h_tape_init, h_work_init, W_h, W_x, b_h, W_write):
    B, T, D = x_seq.shape
    N = h_tape_init.shape[1]
    scale = 1.0 / math.sqrt(D)

    h_tape = h_tape_init
    h_work = h_work_init

    h_work_list = []

    for t in range(T):
        # Read
        read_scores = (h_tape * h_work[:, None, :]).sum(dim=-1) * scale
        read_attn = entmax_1_5(read_scores, dim=-1)  # Use autograd version
        read = (read_attn[:, :, None] * h_tape).sum(dim=1)

        # Update h_work
        pre_act = h_work @ W_h.T + x_seq[:, t] @ W_x.T + read + b_h
        h_work_new = torch.tanh(pre_act)

        # Write
        write_value = h_work_new @ W_write.T
        write_scores = (h_tape * h_work_new[:, None, :]).sum(dim=-1) * scale
        write_attn = entmax_1_5(write_scores, dim=-1)  # Use autograd version
        h_tape = (1 - write_attn[:, :, None]) * h_tape + write_attn[:, :, None] * write_value[:, None, :]

        h_work = h_work_new
        h_work_list.append(h_work)

    h_work_all = torch.stack(h_work_list, dim=1)
    return h_work_all, h_tape

# Run forward with autograd
h_work_all_auto, h_tape_final_auto = forward_autograd(
    x_seq, h_tape_init, h_work_init, W_h, W_x, b_h, W_write
)

# Compute loss and backward with autograd
loss = h_work_all_auto.sum()
loss.backward()

print("\nAutograd gradients:")
print(f"  dW_h norm: {W_h.grad.norm().item():.6f}")
print(f"  dW_x norm: {W_x.grad.norm().item():.6f}")
print(f"  db_h norm: {b_h.grad.norm().item():.6f}")
print(f"  dW_write norm: {W_write.grad.norm().item():.6f}")
print(f"  dx norm: {x_seq.grad.norm().item():.6f}")

# Save autograd gradients
dW_h_auto = W_h.grad.clone()
dW_x_auto = W_x.grad.clone()
db_h_auto = b_h.grad.clone()
dW_write_auto = W_write.grad.clone()
dx_auto = x_seq.grad.clone()

# Now run Python backward
# First need to rerun forward to get the intermediate values
W_h_data = W_h.detach()
W_x_data = W_x.detach()
b_h_data = b_h.detach()
W_write_data = W_write.detach()
x_seq_data = x_seq.detach()

h_tape = h_tape_init.clone()
h_work = h_work_init.clone()

h_work_list = []
h_tape_list = [h_tape_init]
read_attn_list = []
write_attn_list = []

for t in range(T):
    h_work, h_tape, read_attn, write_attn, _ = e25_forward_step_python(
        x_seq_data[:, t], h_tape, h_work,
        W_h_data, W_x_data, b_h_data, W_write_data, scale
    )
    h_work_list.append(h_work)
    h_tape_list.append(h_tape)
    read_attn_list.append(read_attn)
    write_attn_list.append(write_attn)

h_work_all = torch.stack(h_work_list, dim=1)
h_tape_all = torch.stack(h_tape_list, dim=1)
read_attn_all = torch.stack(read_attn_list, dim=1)
write_attn_all = torch.stack(write_attn_list, dim=1)

# Upstream gradients (matching loss = h_work_all.sum())
d_h_work_all = torch.ones_like(h_work_all)
d_h_tape_final = torch.zeros(B, N, D, device=device, dtype=dtype)

# Run Python backward
dx_py, dW_h_py, dW_x_py, db_h_py, dW_write_py = e25_backward_python(
    x_seq_data, h_work_all, h_tape_all, read_attn_all, write_attn_all,
    W_h_data, W_x_data, b_h_data, W_write_data, h_work_init, d_h_work_all, d_h_tape_final, scale
)

print("\nPython backward gradients:")
print(f"  dW_h norm: {dW_h_py.norm().item():.6f}")
print(f"  dW_x norm: {dW_x_py.norm().item():.6f}")
print(f"  db_h norm: {db_h_py.norm().item():.6f}")
print(f"  dW_write norm: {dW_write_py.norm().item():.6f}")
print(f"  dx norm: {dx_py.norm().item():.6f}")

print("\nDifferences (max abs):")
print(f"  dW_h diff: {(dW_h_py - dW_h_auto).abs().max().item():.8f}")
print(f"  dW_x diff: {(dW_x_py - dW_x_auto).abs().max().item():.8f}")
print(f"  db_h diff: {(db_h_py - db_h_auto).abs().max().item():.8f}")
print(f"  dW_write diff: {(dW_write_py - dW_write_auto).abs().max().item():.8f}")
print(f"  dx diff: {(dx_py - dx_auto).abs().max().item():.8f}")

# Check if differences are significant
threshold = 1e-5
all_close = True
for name, diff in [
    ("dW_h", (dW_h_py - dW_h_auto).abs().max().item()),
    ("dW_x", (dW_x_py - dW_x_auto).abs().max().item()),
    ("db_h", (db_h_py - db_h_auto).abs().max().item()),
    ("dW_write", (dW_write_py - dW_write_auto).abs().max().item()),
    ("dx", (dx_py - dx_auto).abs().max().item()),
]:
    if diff > threshold:
        print(f"\nWARNING: {name} gradient has significant difference: {diff:.8f}")
        all_close = False

if all_close:
    print("\nSUCCESS: All Python gradients match autograd!")
else:
    print("\nFAILURE: Some Python gradients don't match autograd!")

print("\n" + "="*60)
