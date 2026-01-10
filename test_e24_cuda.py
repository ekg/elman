#!/usr/bin/env python3
"""
Test E24 CUDA implementation against Python reference.
"""
import torch
import sys
sys.path.insert(0, 'elman/cuda')

from elman.models.e24_single_gemm import E24Cell, e24_forward_step_python
import math

print("=" * 70)
print("E24 CUDA vs Python Numerical Verification")
print("=" * 70)

device = 'cuda'
dtype = torch.bfloat16

# Test parameters
B = 2  # batch size
T = 8  # sequence length
D = 256  # dimension
N = 16  # number of slots

torch.manual_seed(42)

# Create cell
cell = E24Cell(dim=D, n_slots=N).to(device).to(dtype)

# Create input
x = torch.randn(B, T, D, device=device, dtype=dtype)
h_tape = torch.zeros(B, N, D, device=device, dtype=dtype)
h_work = torch.zeros(B, D, device=device, dtype=dtype)

print(f"\nTest config: B={B}, T={T}, D={D}, N={N}")
print("-" * 50)

# Test forward pass
print("\n1. Testing forward pass...")

# CUDA
x_cuda = x.detach().clone()
h_tape_cuda = h_tape.clone()
h_work_cuda = h_work.clone()
h_work_all_cuda, h_tape_final_cuda, _ = cell(x_cuda, h_tape_cuda, h_work_cuda, use_cuda=True)

# Python
x_py = x.detach().clone()
h_tape_py = h_tape.clone()
h_work_py = h_work.clone()
h_work_all_py, h_tape_final_py, _ = cell(x_py, h_tape_py, h_work_py, use_cuda=False)

# Compare
h_work_diff = (h_work_all_cuda.float() - h_work_all_py.float()).abs().max().item()
h_tape_diff = (h_tape_final_cuda.float() - h_tape_final_py.float()).abs().max().item()

print(f"   h_work_all max diff: {h_work_diff:.6f}")
print(f"   h_tape_final max diff: {h_tape_diff:.6f}")
print(f"   Forward pass: {'PASS' if h_work_diff < 0.1 and h_tape_diff < 0.1 else 'FAIL'}")

# Test backward pass
print("\n2. Testing backward pass...")

# CUDA backward
x_cuda = x.detach().clone().requires_grad_(True)
cell.zero_grad()
h_work_all_cuda, h_tape_final_cuda, _ = cell(x_cuda, h_tape.clone(), h_work.clone(), use_cuda=True)
loss_cuda = h_work_all_cuda.sum()
loss_cuda.backward()
dx_cuda = x_cuda.grad.clone()
dW_all_cuda = cell.W_all.grad.clone() if cell.W_all.grad is not None else None

# Python backward
x_py = x.detach().clone().requires_grad_(True)
cell.zero_grad()
h_work_all_py, h_tape_final_py, _ = cell(x_py, h_tape.clone(), h_work.clone(), use_cuda=False)
loss_py = h_work_all_py.sum()
loss_py.backward()
dx_py = x_py.grad.clone()
dW_all_py = cell.W_all.grad.clone() if cell.W_all.grad is not None else None

# Compare gradients
dx_diff = (dx_cuda.float() - dx_py.float()).abs().max().item()
dx_rel = dx_diff / (dx_py.float().abs().max().item() + 1e-8)

print(f"   dx max diff: {dx_diff:.6f} (rel: {dx_rel:.4f})")

if dW_all_cuda is not None and dW_all_py is not None:
    dW_diff = (dW_all_cuda.float() - dW_all_py.float()).abs().max().item()
    dW_rel = dW_diff / (dW_all_py.float().abs().max().item() + 1e-8)
    print(f"   dW_all max diff: {dW_diff:.6f} (rel: {dW_rel:.4f})")
    print(f"   dW_all norms: CUDA={dW_all_cuda.float().norm().item():.2f}, Python={dW_all_py.float().norm().item():.2f}")

print(f"   Backward pass: {'PASS' if dx_rel < 0.1 else 'FAIL'}")

# Test gradient magnitude comparison
print("\n3. Testing gradient magnitudes per timestep...")
for t in range(min(4, T)):
    cuda_norm = dx_cuda[:, t].float().norm().item()
    py_norm = dx_py[:, t].float().norm().item()
    ratio = cuda_norm / (py_norm + 1e-8)
    diff = (dx_cuda[:, t].float() - dx_py[:, t].float()).norm().item()
    print(f"   t={t}: cuda_norm={cuda_norm:.2f}, py_norm={py_norm:.2f}, diff={diff:.4f}, ratio={ratio:.4f}")

# Test training
print("\n4. Testing training (loss should decrease)...")
cell2 = E24Cell(dim=D, n_slots=N).to(device).to(dtype)
optimizer = torch.optim.Adam(cell2.parameters(), lr=0.01)

for i in range(10):
    x_train = torch.randn(B, T, D, device=device, dtype=dtype)
    h_tape_train = torch.zeros(B, N, D, device=device, dtype=dtype)
    h_work_train = torch.zeros(B, D, device=device, dtype=dtype)

    optimizer.zero_grad()
    h_work_all, h_tape_final, _ = cell2(x_train, h_tape_train, h_work_train, use_cuda=True)
    loss = h_work_all.sum()
    loss.backward()
    optimizer.step()

    if i == 0 or i == 9:
        print(f"   Step {i+1}: loss = {loss.item():.2f}")

print("\n" + "=" * 70)
print("E24 verification complete")
print("=" * 70)
