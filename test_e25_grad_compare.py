"""Compare gradients from E25 custom backward vs autograd."""

import torch
import torch.nn as nn
import math

torch.manual_seed(42)

B = 2
T = 4
D = 256
N = 8

device = 'cuda'
dtype = torch.float32  # Use float32 for numerical accuracy

print("="*60)
print("E25 Gradient Comparison: Custom vs Autograd")
print("="*60)

from elman.models.e25_entmax import E25DualMemoryElmanCell, entmax_1_5

class E25CellAutograd(nn.Module):
    """E25 cell using pure autograd (no custom backward)."""

    def __init__(self, dim, n_slots=64, w_h_init_scale=0.9):
        super().__init__()
        self.dim = dim
        self.n_slots = n_slots
        self.scale = 1.0 / math.sqrt(dim)

        self.W_h = nn.Parameter(torch.empty(dim, dim))
        self.W_x = nn.Parameter(torch.empty(dim, dim))
        self.b_h = nn.Parameter(torch.zeros(dim))
        self.W_write = nn.Parameter(torch.empty(dim, dim))

    def forward(self, x_seq, h_tape=None, h_work=None):
        B, T, D = x_seq.shape

        if h_tape is None:
            h_tape = torch.zeros(B, self.n_slots, D, device=x_seq.device, dtype=x_seq.dtype)
        if h_work is None:
            h_work = torch.zeros(B, D, device=x_seq.device, dtype=x_seq.dtype)

        h_work_list = []

        for t in range(T):
            # Read
            read_scores = (h_tape * h_work[:, None, :]).sum(dim=-1) * self.scale
            read_attn = entmax_1_5(read_scores, dim=-1)
            read = (read_attn[:, :, None] * h_tape).sum(dim=1)

            # Update h_work
            pre_act = h_work @ self.W_h.T + x_seq[:, t] @ self.W_x.T + read + self.b_h
            h_work_new = torch.tanh(pre_act)

            # Write
            write_value = h_work_new @ self.W_write.T
            write_scores = (h_tape * h_work_new[:, None, :]).sum(dim=-1) * self.scale
            write_attn = entmax_1_5(write_scores, dim=-1)
            h_tape = (1 - write_attn[:, :, None]) * h_tape + write_attn[:, :, None] * write_value[:, None, :]

            h_work = h_work_new
            h_work_list.append(h_work)

        h_work_all = torch.stack(h_work_list, dim=1)
        return h_work_all, h_tape, h_work

# Create shared initial weights
torch.manual_seed(42)
W_h_init = torch.randn(D, D, device=device, dtype=dtype) * 0.1
W_x_init = torch.randn(D, D, device=device, dtype=dtype) * 0.1
b_h_init = torch.zeros(D, device=device, dtype=dtype)
W_write_init = torch.randn(D, D, device=device, dtype=dtype) * 0.1

# Autograd cell
cell_auto = E25CellAutograd(dim=D, n_slots=N).to(device).to(dtype)
with torch.no_grad():
    cell_auto.W_h.copy_(W_h_init)
    cell_auto.W_x.copy_(W_x_init)
    cell_auto.b_h.copy_(b_h_init)
    cell_auto.W_write.copy_(W_write_init)

# Custom backward cell
cell_custom = E25DualMemoryElmanCell(dim=D, n_slots=N).to(device).to(dtype)
with torch.no_grad():
    cell_custom.W_h.copy_(W_h_init)
    cell_custom.W_x.copy_(W_x_init)
    cell_custom.b_h.copy_(b_h_init)
    cell_custom.W_write.copy_(W_write_init)

# Same input
torch.manual_seed(123)
x = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
x_custom = x.clone().detach().requires_grad_(True)

# Forward autograd
h_work_all_auto, h_tape_auto, h_work_final_auto = cell_auto(x)
loss_auto = h_work_all_auto.sum()
loss_auto.backward()

# Forward custom (Python backward)
h_work_all_custom, h_tape_custom, h_work_final_custom = cell_custom(x_custom, use_cuda=False)
loss_custom = h_work_all_custom.sum()
loss_custom.backward()

print("\nForward comparison:")
print(f"  h_work_all diff: {(h_work_all_auto - h_work_all_custom).abs().max().item():.8f}")
print(f"  h_tape diff: {(h_tape_auto - h_tape_custom).abs().max().item():.8f}")

print("\nAutograd gradients:")
print(f"  W_h: {cell_auto.W_h.grad.norm().item():.6f}")
print(f"  W_x: {cell_auto.W_x.grad.norm().item():.6f}")
print(f"  b_h: {cell_auto.b_h.grad.norm().item():.6f}")
print(f"  W_write: {cell_auto.W_write.grad.norm().item():.6f}")
print(f"  x: {x.grad.norm().item():.6f}")

print("\nCustom backward gradients:")
print(f"  W_h: {cell_custom.W_h.grad.norm().item():.6f}")
print(f"  W_x: {cell_custom.W_x.grad.norm().item():.6f}")
print(f"  b_h: {cell_custom.b_h.grad.norm().item():.6f}")
print(f"  W_write: {cell_custom.W_write.grad.norm().item():.6f}")
print(f"  x: {x_custom.grad.norm().item():.6f}")

print("\nGradient differences (max abs):")
print(f"  W_h: {(cell_auto.W_h.grad - cell_custom.W_h.grad).abs().max().item():.8f}")
print(f"  W_x: {(cell_auto.W_x.grad - cell_custom.W_x.grad).abs().max().item():.8f}")
print(f"  b_h: {(cell_auto.b_h.grad - cell_custom.b_h.grad).abs().max().item():.8f}")
print(f"  W_write: {(cell_auto.W_write.grad - cell_custom.W_write.grad).abs().max().item():.8f}")
print(f"  x: {(x.grad - x_custom.grad).abs().max().item():.8f}")

# Check relative error
print("\nRelative errors:")
for name, g_auto, g_custom in [
    ("W_h", cell_auto.W_h.grad, cell_custom.W_h.grad),
    ("W_x", cell_auto.W_x.grad, cell_custom.W_x.grad),
    ("b_h", cell_auto.b_h.grad, cell_custom.b_h.grad),
    ("W_write", cell_auto.W_write.grad, cell_custom.W_write.grad),
    ("x", x.grad, x_custom.grad),
]:
    rel_err = (g_auto - g_custom).abs().max() / (g_auto.abs().max() + 1e-8)
    print(f"  {name}: {rel_err.item():.6f}")

print("\n" + "="*60)
