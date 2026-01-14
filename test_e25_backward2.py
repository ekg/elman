"""Focused backward pass comparison for E25 through the Cell."""

import torch
import math

torch.manual_seed(42)

# Test parameters
B = 2
T = 4
D = 256
N = 8

device = 'cuda'
dtype = torch.bfloat16

print("="*60)
print("E25 Backward Pass Debug (via Cell)")
print("="*60)

from elman.models.e25_entmax import E25DualMemoryElmanCell

# Create cell
cell = E25DualMemoryElmanCell(dim=D, n_slots=N).to(device).to(dtype)

# Create test inputs
x_seq = torch.randn(B, T, D, device=device, dtype=dtype)

print("\n1. Testing Python backward...")
x_py = x_seq.clone().requires_grad_(True)
cell.zero_grad()
h_work_py, _, _ = cell(x_py, use_cuda=False)
loss_py = h_work_py.sum()
loss_py.backward()

grad_x_py = x_py.grad.clone()
grad_W_h_py = cell.W_h.grad.clone()
grad_W_x_py = cell.W_x.grad.clone()
grad_W_write_py = cell.W_write.grad.clone()
grad_b_h_py = cell.b_h.grad.clone()

print(f"   dx norm: {grad_x_py.norm().item():.4f}")
print(f"   dW_h norm: {grad_W_h_py.norm().item():.4f}")
print(f"   dW_x norm: {grad_W_x_py.norm().item():.4f}")
print(f"   dW_write norm: {grad_W_write_py.norm().item():.4f}")
print(f"   db_h norm: {grad_b_h_py.norm().item():.4f}")

print("\n2. Testing CUDA backward...")
x_cuda = x_seq.clone().requires_grad_(True)
cell.zero_grad()
h_work_cuda, _, _ = cell(x_cuda, use_cuda=True)
loss_cuda = h_work_cuda.sum()
loss_cuda.backward()

grad_x_cuda = x_cuda.grad.clone()
grad_W_h_cuda = cell.W_h.grad.clone()
grad_W_x_cuda = cell.W_x.grad.clone()
grad_W_write_cuda = cell.W_write.grad.clone()
grad_b_h_cuda = cell.b_h.grad.clone()

print(f"   dx norm: {grad_x_cuda.norm().item():.4f}")
print(f"   dW_h norm: {grad_W_h_cuda.norm().item():.4f}")
print(f"   dW_x norm: {grad_W_x_cuda.norm().item():.4f}")
print(f"   dW_write norm: {grad_W_write_cuda.norm().item():.4f}")
print(f"   db_h norm: {grad_b_h_cuda.norm().item():.4f}")

print("\n3. Comparing gradients...")
def compare(name, py, cuda):
    diff = (py.float() - cuda.float()).abs()
    rel_diff = diff / (py.float().abs().clamp(min=1e-6))
    print(f"   {name}: max_diff={diff.max().item():.6f}, mean_diff={diff.mean().item():.6f}, rel_max={rel_diff.max().item():.4f}")

compare("dx", grad_x_py, grad_x_cuda)
compare("dW_h", grad_W_h_py, grad_W_h_cuda)
compare("dW_x", grad_W_x_py, grad_W_x_cuda)
compare("dW_write", grad_W_write_py, grad_W_write_cuda)
compare("db_h", grad_b_h_py, grad_b_h_cuda)

# Check for issues
max_diffs = [
    (grad_x_py.float() - grad_x_cuda.float()).abs().max().item(),
    (grad_W_h_py.float() - grad_W_h_cuda.float()).abs().max().item(),
    (grad_W_x_py.float() - grad_W_x_cuda.float()).abs().max().item(),
    (grad_W_write_py.float() - grad_W_write_cuda.float()).abs().max().item(),
    (grad_b_h_py.float() - grad_b_h_cuda.float()).abs().max().item(),
]

if max(max_diffs) > 0.5:
    print("\n   WARNING: Large gradient differences detected!")
else:
    print("\n   All gradients look reasonable!")

print("\n" + "="*60)
print("Done!")
print("="*60)
