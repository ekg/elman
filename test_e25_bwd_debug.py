"""Debug the backward pass step by step."""

import torch
import math

torch.manual_seed(42)

B = 1
T = 2
D = 256
N = 8

device = 'cuda'
dtype = torch.bfloat16

from elman.models.e25_entmax import (
    e25_forward_step_python,
    e25_backward_python,
    entmax_1_5_backward,
    E25DualMemoryElmanCell
)

# Create test inputs with same weights
cell = E25DualMemoryElmanCell(dim=D, n_slots=N).to(device).to(dtype)
x_seq = torch.randn(B, T, D, device=device, dtype=dtype)

print("Running Python forward to get intermediate states...")
scale = 1.0 / math.sqrt(D)

# Run Python forward step by step
h_tape = torch.zeros(B, N, D, device=device, dtype=torch.float32)
h_work = torch.zeros(B, D, device=device, dtype=torch.float32)

h_work_list = []
h_tape_list = [h_tape.clone()]
read_attn_list = []
write_attn_list = []

W_h = cell.W_h.float()
W_x = cell.W_x.float()
b_h = cell.b_h.float()
W_write = cell.W_write.float()

for t in range(T):
    h_work, h_tape, read_attn, write_attn, _ = e25_forward_step_python(
        x_seq[:, t].float(),
        h_tape, h_work,
        W_h, W_x, b_h, W_write, scale
    )
    h_work_list.append(h_work)
    h_tape_list.append(h_tape)
    read_attn_list.append(read_attn)
    write_attn_list.append(write_attn)

h_work_all_py = torch.stack(h_work_list, dim=1)  # [B, T, D]
h_tape_all_py = torch.stack(h_tape_list, dim=1)  # [B, T+1, N, D]
read_attn_py = torch.stack(read_attn_list, dim=1)  # [B, T, N]
write_attn_py = torch.stack(write_attn_list, dim=1)  # [B, T, N]

print(f"h_work_all shape: {h_work_all_py.shape}")
print(f"h_tape_all shape: {h_tape_all_py.shape}")

# Create upstream gradients
d_h_work_all = torch.ones_like(h_work_all_py)
d_h_tape_final = torch.zeros(B, N, D, device=device, dtype=torch.float32)

print("\nRunning Python backward...")
dx_py, dW_h_py, dW_x_py, db_h_py, dW_write_py = e25_backward_python(
    x_seq.float(),
    h_work_all_py,
    h_tape_all_py,
    read_attn_py,
    write_attn_py,
    W_h, W_x, b_h, W_write,
    torch.zeros(B, D, device=device, dtype=torch.float32),  # h_work_init
    d_h_work_all,
    d_h_tape_final,
    scale
)

print(f"Python dW_x norm: {dW_x_py.norm().item():.4f}")
print(f"Python dW_x[0,:4]: {dW_x_py[0, :4].tolist()}")

print("\nRunning CUDA backward via cell...")
cell.zero_grad()
x_grad = x_seq.clone().requires_grad_(True)
out_cuda, _, _ = cell(x_grad, use_cuda=True)
loss = out_cuda.sum()
loss.backward()

print(f"CUDA dW_x norm: {cell.W_x.grad.float().norm().item():.4f}")
print(f"CUDA dW_x[0,:4]: {cell.W_x.grad[0, :4].float().tolist()}")

print("\nComparing...")
diff = (dW_x_py.to(dtype).float() - cell.W_x.grad.float()).abs()
print(f"Max diff: {diff.max().item():.4f}")
print(f"Mean diff: {diff.mean().item():.4f}")
