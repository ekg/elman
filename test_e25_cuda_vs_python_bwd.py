"""Compare gradients from E25 CUDA backward vs Python backward."""

import torch
import torch.nn as nn
import math

torch.manual_seed(42)

B = 2
T = 4
D = 256
N = 8

device = 'cuda'
dtype = torch.bfloat16  # CUDA kernel requires bfloat16

print("="*60)
print("E25 Gradient Comparison: CUDA vs Python Backward")
print("="*60)

from elman.models.e25_entmax import E25DualMemoryElmanCell

# Create two cells with same weights
torch.manual_seed(42)
cell_python = E25DualMemoryElmanCell(dim=D, n_slots=N).to(device).to(dtype)
cell_cuda = E25DualMemoryElmanCell(dim=D, n_slots=N).to(device).to(dtype)

# Copy weights from Python cell to CUDA cell
with torch.no_grad():
    cell_cuda.W_h.copy_(cell_python.W_h)
    cell_cuda.W_x.copy_(cell_python.W_x)
    cell_cuda.b_h.copy_(cell_python.b_h)
    cell_cuda.W_write.copy_(cell_python.W_write)

# Same input
torch.manual_seed(123)
x = torch.randn(B, T, D, device=device, dtype=dtype)
x_python = x.clone().requires_grad_(True)
x_cuda = x.clone().requires_grad_(True)

# Forward with Python backward
print("\n--- Python backward ---")
h_work_all_py, h_tape_py, h_work_final_py = cell_python(x_python, use_cuda=False)
loss_py = h_work_all_py.sum()
loss_py.backward()

print(f"Loss: {loss_py.item():.4f}")
print(f"W_h grad norm: {cell_python.W_h.grad.float().norm().item():.4f}")
print(f"W_x grad norm: {cell_python.W_x.grad.float().norm().item():.4f}")
print(f"W_write grad norm: {cell_python.W_write.grad.float().norm().item():.4f}")
print(f"x grad norm: {x_python.grad.float().norm().item():.4f}")

# Forward with CUDA backward
print("\n--- CUDA backward ---")
h_work_all_cuda, h_tape_cuda, h_work_final_cuda = cell_cuda(x_cuda, use_cuda=True)
loss_cuda = h_work_all_cuda.sum()
loss_cuda.backward()

print(f"Loss: {loss_cuda.item():.4f}")
print(f"W_h grad norm: {cell_cuda.W_h.grad.float().norm().item():.4f}")
print(f"W_x grad norm: {cell_cuda.W_x.grad.float().norm().item():.4f}")
print(f"W_write grad norm: {cell_cuda.W_write.grad.float().norm().item():.4f}")
print(f"x grad norm: {x_cuda.grad.float().norm().item():.4f}")

# Compare
print("\n--- Comparison ---")
print(f"Forward h_work_all diff: {(h_work_all_py.float() - h_work_all_cuda.float()).abs().max().item():.6f}")
print(f"Forward h_tape diff: {(h_tape_py.float() - h_tape_cuda.float()).abs().max().item():.6f}")

print(f"\nW_h grad diff: {(cell_python.W_h.grad.float() - cell_cuda.W_h.grad.float()).abs().max().item():.6f}")
print(f"W_x grad diff: {(cell_python.W_x.grad.float() - cell_cuda.W_x.grad.float()).abs().max().item():.6f}")
print(f"W_write grad diff: {(cell_python.W_write.grad.float() - cell_cuda.W_write.grad.float()).abs().max().item():.6f}")
print(f"x grad diff: {(x_python.grad.float() - x_cuda.grad.float()).abs().max().item():.6f}")

# Relative errors
print("\nRelative errors:")
for name, g_py, g_cuda in [
    ("W_h", cell_python.W_h.grad.float(), cell_cuda.W_h.grad.float()),
    ("W_x", cell_python.W_x.grad.float(), cell_cuda.W_x.grad.float()),
    ("W_write", cell_python.W_write.grad.float(), cell_cuda.W_write.grad.float()),
    ("x", x_python.grad.float(), x_cuda.grad.float()),
]:
    rel_err = (g_py - g_cuda).abs().max() / (g_py.abs().max() + 1e-8)
    print(f"  {name}: {rel_err.item():.6f}")

print("\n" + "="*60)
