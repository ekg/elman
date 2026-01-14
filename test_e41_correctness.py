#!/usr/bin/env python3
"""
Test E41 Diagonal W_x Elman: Python vs CUDA forward/backward correctness.
"""

import torch
import torch.nn.functional as F
import sys

# Set device (use cuda:0 since CUDA_VISIBLE_DEVICES controls which GPU is visible)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print("=" * 60)
print("Testing E41 Diagonal W_x Elman: Python vs CUDA Correctness")
print("=" * 60)

# Check CUDA availability first
try:
    import hasty_pytorch_lib
    E41_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e41_diagonal_wx_forward')
    print(f"CUDA kernel available: {E41_CUDA_AVAILABLE}")
    if not E41_CUDA_AVAILABLE:
        print("ERROR: e41_diagonal_wx_forward not found in hasty_pytorch_lib")
        print(f"Available functions: {[x for x in dir(hasty_pytorch_lib) if 'e4' in x.lower()]}")
        sys.exit(1)
except ImportError as e:
    print(f"ERROR: Could not import hasty_pytorch_lib: {e}")
    sys.exit(1)

# Test parameters
T, B, dim = 16, 4, 128
dtype = torch.bfloat16

print(f"\nTest config: T={T}, B={B}, dim={dim}, dtype={dtype}")
print(f"Device: {device}")

# Create test inputs with fixed seed for reproducibility
torch.manual_seed(42)
x = torch.randn(T, B, dim, device=device, dtype=dtype, requires_grad=True)
h0 = torch.zeros(B, dim, device=device, dtype=dtype)
d_x = torch.randn(dim, device=device, dtype=dtype, requires_grad=True)
W_h = torch.randn(dim, dim, device=device, dtype=dtype, requires_grad=True)
b = torch.randn(dim, device=device, dtype=dtype, requires_grad=True)

# Clone for CUDA path (separate computation graph)
x_cuda = x.clone().detach().requires_grad_(True)
d_x_cuda = d_x.clone().detach().requires_grad_(True)
W_h_cuda = W_h.clone().detach().requires_grad_(True)
b_cuda = b.clone().detach().requires_grad_(True)

# Random output gradient for backward test
grad_output = torch.randn(T, B, dim, device=device, dtype=dtype)

print("\n" + "=" * 60)
print("FORWARD PASS")
print("=" * 60)

# Python forward pass
print("\nRunning Python forward...")
h_list = [h0]
output_list = []
for t in range(T):
    h_prev = h_list[-1]
    x_t = x[t]
    # E41: d_x * x_t element-wise, then W_h @ h_prev GEMM
    raw = d_x * x_t + h_prev @ W_h.T + b
    h_new = torch.tanh(raw)
    h_list.append(h_new)
    # Self-gating: output = h * silu(h)
    out = h_new * F.silu(h_new)
    output_list.append(out)

h_py = torch.stack(h_list, dim=0)
output_py = torch.stack(output_list, dim=0)
print(f"  Python output shape: {output_py.shape}")
print(f"  Python h shape: {h_py.shape}")

# CUDA forward pass
print("\nRunning CUDA forward...")
h_cuda, output_cuda, v_cuda = hasty_pytorch_lib.e41_diagonal_wx_forward(
    True,  # training
    x_cuda,
    h0,
    d_x_cuda,
    W_h_cuda,
    b_cuda
)
print(f"  CUDA output shape: {output_cuda.shape}")
print(f"  CUDA h shape: {h_cuda.shape}")

# Compare forward outputs
output_diff = (output_py - output_cuda).abs().max().item()
h_diff = (h_py - h_cuda).abs().max().item()
print(f"\nForward comparison:")
print(f"  Output max diff: {output_diff:.6e}")
print(f"  Hidden max diff: {h_diff:.6e}")

# Check if within tolerance for bfloat16
# BF16 has ~3 decimal digits of precision, and errors accumulate over T timesteps
atol_fwd = 1e-2  # 1% tolerance for forward
forward_pass = output_diff < atol_fwd and h_diff < atol_fwd
print(f"  Forward PASS: {forward_pass} (atol={atol_fwd})")

print("\n" + "=" * 60)
print("BACKWARD PASS")
print("=" * 60)

# Python backward
print("\nRunning Python backward...")
output_py.backward(grad_output)
dx_py = x.grad.clone()
dd_x_py = d_x.grad.clone()
dW_h_py = W_h.grad.clone()
db_py = b.grad.clone()

print(f"  Python dx shape: {dx_py.shape}")
print(f"  Python dd_x shape: {dd_x_py.shape}")
print(f"  Python dW_h shape: {dW_h_py.shape}")
print(f"  Python db shape: {db_py.shape}")

# CUDA backward
print("\nRunning CUDA backward...")
dx_cuda, dd_x_cuda, dW_h_cuda, db_cuda = hasty_pytorch_lib.e41_diagonal_wx_backward(
    d_x_cuda,
    W_h_cuda,
    x_cuda,
    h_cuda,
    v_cuda,
    grad_output
)

print(f"  CUDA dx shape: {dx_cuda.shape}")
print(f"  CUDA dd_x shape: {dd_x_cuda.shape}")
print(f"  CUDA dW_h shape: {dW_h_cuda.shape}")
print(f"  CUDA db shape: {db_cuda.shape}")

# Compare backward outputs
dx_diff = (dx_py - dx_cuda).abs().max().item()
dd_x_diff = (dd_x_py - dd_x_cuda).abs().max().item()
dW_h_diff = (dW_h_py - dW_h_cuda).abs().max().item()
db_diff = (db_py - db_cuda).abs().max().item()

print(f"\nBackward comparison:")
print(f"  dx max diff: {dx_diff:.6e}")
print(f"  dd_x max diff: {dd_x_diff:.6e}")
print(f"  dW_h max diff: {dW_h_diff:.6e}")
print(f"  db max diff: {db_diff:.6e}")

# For backward, use relative tolerance since gradients can be large or small
# Check relative error: |a-b| / max(|a|, |b|, 1)
def rel_err(a, b):
    return (a - b).abs().max().item() / max(a.abs().max().item(), b.abs().max().item(), 1.0)

dx_rel = rel_err(dx_py, dx_cuda)
dd_x_rel = rel_err(dd_x_py, dd_x_cuda)
dW_h_rel = rel_err(dW_h_py, dW_h_cuda)
db_rel = rel_err(db_py, db_cuda)

print(f"\nBackward relative errors:")
print(f"  dx rel err: {dx_rel:.6e}")
print(f"  dd_x rel err: {dd_x_rel:.6e}")
print(f"  dW_h rel err: {dW_h_rel:.6e}")
print(f"  db rel err: {db_rel:.6e}")

# BF16 backward can have ~10% relative error over long sequences due to accumulation
rtol_bwd = 0.15  # 15% relative tolerance
backward_pass = (dx_rel < rtol_bwd and dd_x_rel < rtol_bwd and
                 dW_h_rel < rtol_bwd and db_rel < rtol_bwd)
print(f"  Backward PASS: {backward_pass} (rtol={rtol_bwd})")

print("\n" + "=" * 60)
print("OVERALL RESULT")
print("=" * 60)

if forward_pass and backward_pass:
    print("SUCCESS: E41 Python and CUDA implementations match!")
    sys.exit(0)
else:
    print("FAILURE: E41 implementations do not match!")
    sys.exit(1)
