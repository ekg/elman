"""Minimal test for E25 to isolate the bug."""

import torch
import math

torch.manual_seed(42)

B = 1  # Single batch
T = 1  # Single timestep
D = 256
N = 8

device = 'cuda'
dtype = torch.float32  # Use float32 for debugging precision

print("="*60)
print("E25 Minimal Test (B=1, T=1)")
print("="*60)

from elman.models.e25_entmax import (
    e25_forward_step_python,
    entmax_1_5_backward,
    E25DualMemoryElmanCell
)

# Create cell
cell = E25DualMemoryElmanCell(dim=D, n_slots=N).to(device).to(dtype)

# Simple input
x_seq = torch.randn(B, T, D, device=device, dtype=dtype)

print("\n1. Python forward/backward...")
x_py = x_seq.clone().requires_grad_(True)
cell.zero_grad()
h_work_py, _, _ = cell(x_py, use_cuda=False)
loss_py = h_work_py.sum()
loss_py.backward()

print(f"   Forward h_work: {h_work_py[0, 0, :4].tolist()}")
print(f"   dx norm: {x_py.grad.norm().item():.6f}")
print(f"   dW_h norm: {cell.W_h.grad.norm().item():.6f}")
print(f"   dW_x norm: {cell.W_x.grad.norm().item():.6f}")

# Save Python grads
grad_x_py = x_py.grad.clone()
grad_W_h_py = cell.W_h.grad.clone()
grad_W_x_py = cell.W_x.grad.clone()
grad_W_write_py = cell.W_write.grad.clone()
grad_b_h_py = cell.b_h.grad.clone()

print("\n2. CUDA forward/backward...")
cell_cuda = E25DualMemoryElmanCell(dim=D, n_slots=N).to(device).to(torch.bfloat16)
# Copy weights
with torch.no_grad():
    cell_cuda.W_h.copy_(cell.W_h.to(torch.bfloat16))
    cell_cuda.W_x.copy_(cell.W_x.to(torch.bfloat16))
    cell_cuda.W_write.copy_(cell.W_write.to(torch.bfloat16))
    cell_cuda.b_h.copy_(cell.b_h.to(torch.bfloat16))

x_cuda = x_seq.clone().to(torch.bfloat16).requires_grad_(True)
cell_cuda.zero_grad()
h_work_cuda, _, _ = cell_cuda(x_cuda, use_cuda=True)
loss_cuda = h_work_cuda.sum()
loss_cuda.backward()

print(f"   Forward h_work: {h_work_cuda[0, 0, :4].float().tolist()}")
print(f"   dx norm: {x_cuda.grad.float().norm().item():.6f}")
print(f"   dW_h norm: {cell_cuda.W_h.grad.float().norm().item():.6f}")
print(f"   dW_x norm: {cell_cuda.W_x.grad.float().norm().item():.6f}")

print("\n3. Comparing forward...")
fwd_diff = (h_work_py.to(torch.bfloat16).float() - h_work_cuda.float()).abs()
print(f"   Max forward diff: {fwd_diff.max().item():.6f}")

print("\n4. Comparing gradients...")
def compare(name, py, cuda):
    diff = (py.float() - cuda.float()).abs()
    print(f"   {name}: max_diff={diff.max().item():.6f}, mean_diff={diff.mean().item():.6f}")
    return diff.max().item()

diffs = []
diffs.append(compare("dx", grad_x_py.to(torch.bfloat16), x_cuda.grad))
diffs.append(compare("dW_h", grad_W_h_py.to(torch.bfloat16), cell_cuda.W_h.grad))
diffs.append(compare("dW_x", grad_W_x_py.to(torch.bfloat16), cell_cuda.W_x.grad))
diffs.append(compare("dW_write", grad_W_write_py.to(torch.bfloat16), cell_cuda.W_write.grad))
diffs.append(compare("db_h", grad_b_h_py.to(torch.bfloat16), cell_cuda.b_h.grad))

if max(diffs) > 0.1:
    print("\n   WARNING: Large differences detected!")
else:
    print("\n   All gradients look good!")

# Sample values
print("\n5. Sample gradient values...")
print(f"   Python dW_x[0,:4]: {grad_W_x_py[0, :4].tolist()}")
print(f"   CUDA dW_x[0,:4]: {cell_cuda.W_x.grad[0, :4].float().tolist()}")

print("\n" + "="*60)
print("Done!")
print("="*60)
