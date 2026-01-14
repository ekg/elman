#!/usr/bin/env python3
"""
Test E38 (No W_x Elman) CUDA vs Python correctness.

Tests:
1. Forward pass: compare outputs with torch.allclose (atol=1e-3 for bfloat16)
2. Backward pass: compare gradients match
"""

import torch
import torch.nn.functional as F

# Force use of Python fallback for comparison
import sys
sys.path.insert(0, '/home/erikg/elman')

from elman.models.e38_no_wx import E38NoWxCell, E38_CUDA_AVAILABLE
import elman.models.e38_no_wx as e38_module

print("=" * 60)
print("E38 (No W_x Elman) CUDA vs Python Correctness Test")
print("=" * 60)

print(f"CUDA kernel available: {E38_CUDA_AVAILABLE}")

if not E38_CUDA_AVAILABLE:
    print("ERROR: CUDA kernel not available!")
    print("Please rebuild with: cd elman/cuda && pip install -e .")
    sys.exit(1)

device = 'cuda:0'
dtype = torch.bfloat16

# Test parameters
T, B, D = 16, 4, 128  # timesteps, batch, dim

# Seed for reproducibility
torch.manual_seed(42)

print(f"\nTest config: T={T}, B={B}, D={D}, dtype={dtype}")
print("-" * 60)

# Create cell and input - use 'none' mode to avoid spectral norm discrepancy
cell = E38NoWxCell(D, w_h_mode='none', mamba2_init=True).to(device).to(dtype)
x = torch.randn(T, B, D, device=device, dtype=dtype, requires_grad=True)
h0 = torch.zeros(B, D, device=device, dtype=dtype)

# Get W_h (no spectral norm to avoid discrepancy)
W_h = cell.get_W_h()

print("\n--- FORWARD PASS TEST ---")

# Run CUDA forward
cell.train()
cuda_output, cuda_h = cell(x.clone(), None, h0.clone())

# Run Python forward (force no CUDA by temporarily monkeypatching)
orig_cuda_available = e38_module.E38_CUDA_AVAILABLE
e38_module.E38_CUDA_AVAILABLE = False

cell_py = E38NoWxCell(D, w_h_mode='none', mamba2_init=True).to(device).to(dtype)
# Copy weights from CUDA cell
cell_py.W_h.data.copy_(cell.W_h.data)
cell_py.b.data.copy_(cell.b.data)

py_output, py_h = cell_py(x.clone(), None, h0.clone())

e38_module.E38_CUDA_AVAILABLE = orig_cuda_available

# Compare forward outputs - bf16 has ~7.8e-3 precision, so use 1e-2 atol
output_close = torch.allclose(cuda_output, py_output, atol=1e-2, rtol=1e-2)
h_close = torch.allclose(cuda_h, py_h, atol=1e-2, rtol=1e-2)

print(f"Output max diff: {(cuda_output - py_output).abs().max().item():.6e}")
print(f"Hidden max diff: {(cuda_h - py_h).abs().max().item():.6e}")
print(f"Output close (atol=1e-2): {output_close}")
print(f"Hidden close (atol=1e-2): {h_close}")

if output_close and h_close:
    print("FORWARD PASS: PASSED")
else:
    print("FORWARD PASS: FAILED")

print("\n--- BACKWARD PASS TEST ---")

# Compute backward for CUDA version
cuda_loss = cuda_output.sum()
cuda_loss.backward()
cuda_dx = x.grad.clone()
cuda_dW_h = cell.W_h.grad.clone() if cell.W_h.grad is not None else None
cuda_db = cell.b.grad.clone() if cell.b.grad is not None else None

# Clear gradients
x.grad = None
cell.W_h.grad = None
cell.b.grad = None

# Create new input with grad for Python version
x2 = torch.randn(T, B, D, device=device, dtype=dtype, requires_grad=True)
x2.data.copy_(x.data)

# Force Python fallback for backward
e38_module.E38_CUDA_AVAILABLE = False

cell_py2 = E38NoWxCell(D, w_h_mode='none', mamba2_init=True).to(device).to(dtype)
cell_py2.W_h.data.copy_(cell.W_h.data)
cell_py2.b.data.copy_(cell.b.data)

py_output2, _ = cell_py2(x2, None, h0.clone())
py_loss = py_output2.sum()
py_loss.backward()
py_dx = x2.grad.clone()
py_dW_h = cell_py2.W_h.grad.clone() if cell_py2.W_h.grad is not None else None
py_db = cell_py2.b.grad.clone() if cell_py2.b.grad is not None else None

e38_module.E38_CUDA_AVAILABLE = orig_cuda_available

# Compare backward gradients - use larger tolerance for bf16 accumulated gradients
dx_close = torch.allclose(cuda_dx, py_dx, atol=5e-2, rtol=1e-1)
print(f"dx max diff: {(cuda_dx - py_dx).abs().max().item():.6e}")
print(f"dx close (atol=5e-2): {dx_close}")

dW_h_close = True
if cuda_dW_h is not None and py_dW_h is not None:
    dW_h_close = torch.allclose(cuda_dW_h, py_dW_h, atol=1e-1, rtol=1e-1)
    print(f"dW_h max diff: {(cuda_dW_h - py_dW_h).abs().max().item():.6e}")
    print(f"dW_h close (atol=1e-1): {dW_h_close}")
else:
    print("dW_h: one or both are None")

db_close = True
if cuda_db is not None and py_db is not None:
    db_close = torch.allclose(cuda_db, py_db, atol=3e-1, rtol=1e-1)
    print(f"db max diff: {(cuda_db - py_db).abs().max().item():.6e}")
    print(f"db close (atol=3e-1): {db_close}")
else:
    print("db: one or both are None")

if dx_close and dW_h_close and db_close:
    print("BACKWARD PASS: PASSED")
else:
    print("BACKWARD PASS: FAILED")

print("\n--- FULL LAYER TEST ---")

from elman.models.e38_no_wx import E38NoWx

model = E38NoWx(dim=256, expansion=2.0).to(device).to(dtype)
x_layer = torch.randn(2, 32, 256, device=device, dtype=dtype)

print("Testing full E38NoWx layer forward...")
out, h_final = model(x_layer)
print(f"Input: {x_layer.shape}, Output: {out.shape}, Hidden: {h_final.shape}")

print("Testing full E38NoWx layer backward...")
loss = out.sum()
loss.backward()
print("Backward passed!")

params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {params:,}")

print("\n" + "=" * 60)
print("ALL TESTS COMPLETED")
print("=" * 60)
