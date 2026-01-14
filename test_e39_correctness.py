"""
Test E39 No-Bias Elman: Python vs CUDA implementation correctness.
Verifies forward and backward pass match between PyTorch fallback and CUDA kernel.
"""

import torch
import torch.nn.functional as F

# Check if CUDA kernel is available
try:
    import hasty_pytorch_lib
    E39_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e39_no_bias_forward')
    print(f"E39 CUDA kernel available: {E39_CUDA_AVAILABLE}")
except ImportError:
    E39_CUDA_AVAILABLE = False
    print("hasty_pytorch_lib not available")

# Test parameters
torch.manual_seed(42)
device = 'cuda:0'  # Use GPU (CUDA_VISIBLE_DEVICES=1 makes GPU 1 appear as cuda:0)
dtype = torch.float32  # Use float32 for stricter testing
T = 16  # sequence length
B = 4   # batch size
D = 128 # dimension
atol = 1e-4  # tolerance for float32

print(f"\nTest configuration:")
print(f"  Device: {device}")
print(f"  dtype: {dtype}")
print(f"  T={T}, B={B}, D={D}")
print(f"  atol={atol}")

# Create test inputs
x = torch.randn(T, B, D, device=device, dtype=dtype, requires_grad=True)
h0 = torch.zeros(B, D, device=device, dtype=dtype)
W_h = torch.randn(D, D, device=device, dtype=dtype, requires_grad=True)

# Scale W_h to be stable (spectral radius < 1)
with torch.no_grad():
    u, s, v = torch.linalg.svd(W_h.float())
    W_h.copy_((u @ torch.diag(torch.clamp(s, max=0.95)) @ v).to(dtype))

print("\n" + "="*60)
print("Testing E39 Forward Pass")
print("="*60)

# Python forward
h_list_py = [h0]
output_list_py = []
v_list_py = []  # pre-activation cache

for t in range(T):
    h_prev = h_list_py[-1]
    x_t = x[t]

    # E39: x + W_h @ h_prev (no W_x, no bias)
    raw = x_t + h_prev @ W_h.T
    v_list_py.append(raw)
    h_new = torch.tanh(raw)
    h_list_py.append(h_new)

    # Self-gating: output = h * silu(h)
    output = h_new * F.silu(h_new)
    output_list_py.append(output)

h_py = torch.stack(h_list_py, dim=0)
output_py = torch.stack(output_list_py, dim=0)
v_py = torch.stack(v_list_py, dim=0)

print(f"Python forward:")
print(f"  h shape: {h_py.shape}")
print(f"  output shape: {output_py.shape}")

# CUDA forward
if E39_CUDA_AVAILABLE:
    h_cuda, output_cuda, v_cuda = hasty_pytorch_lib.e39_no_bias_forward(
        True,  # training
        x.contiguous(),
        h0.contiguous(),
        W_h.contiguous()
    )

    print(f"\nCUDA forward:")
    print(f"  h shape: {h_cuda.shape}")
    print(f"  output shape: {output_cuda.shape}")

    # Compare
    h_close = torch.allclose(h_py, h_cuda, atol=atol, rtol=1e-2)
    output_close = torch.allclose(output_py, output_cuda, atol=atol, rtol=1e-2)

    print(f"\nForward comparison:")
    print(f"  h matches: {h_close}")
    print(f"  output matches: {output_close}")

    if not h_close:
        h_diff = (h_py - h_cuda).abs().max().item()
        print(f"  h max diff: {h_diff}")

    if not output_close:
        output_diff = (output_py - output_cuda).abs().max().item()
        print(f"  output max diff: {output_diff}")
else:
    print("Skipping CUDA forward test - kernel not available")

print("\n" + "="*60)
print("Testing E39 Backward Pass")
print("="*60)

# Need fresh tensors for backward test
x2 = torch.randn(T, B, D, device=device, dtype=dtype, requires_grad=True)
W_h2 = W_h.clone().detach().requires_grad_(True)

# Python forward/backward
h_list_py2 = [h0]
output_list_py2 = []

for t in range(T):
    h_prev = h_list_py2[-1]
    x_t = x2[t]
    raw = x_t + h_prev @ W_h2.T
    h_new = torch.tanh(raw)
    h_list_py2.append(h_new)
    output = h_new * F.silu(h_new)
    output_list_py2.append(output)

output_py2 = torch.stack(output_list_py2, dim=0)

# Backward
d_output = torch.randn_like(output_py2)
output_py2.backward(d_output)

dx_py = x2.grad.clone()
dW_h_py = W_h2.grad.clone()

print(f"Python backward:")
print(f"  dx shape: {dx_py.shape}")
print(f"  dW_h shape: {dW_h_py.shape}")

# CUDA backward
if E39_CUDA_AVAILABLE:
    x3 = x2.detach().clone().requires_grad_(True)
    W_h3 = W_h.clone().detach().requires_grad_(True)

    # Forward to get cached values
    h_cuda2, output_cuda2, v_cuda2 = hasty_pytorch_lib.e39_no_bias_forward(
        True,
        x3.contiguous(),
        h0.contiguous(),
        W_h3.contiguous()
    )

    # Backward
    dx_cuda, dW_h_cuda = hasty_pytorch_lib.e39_no_bias_backward(
        W_h3.contiguous(),
        x3.contiguous(),
        h_cuda2.contiguous(),
        v_cuda2.contiguous(),
        d_output.contiguous()
    )

    print(f"\nCUDA backward:")
    print(f"  dx shape: {dx_cuda.shape}")
    print(f"  dW_h shape: {dW_h_cuda.shape}")

    # Compare
    dx_close = torch.allclose(dx_py, dx_cuda, atol=atol, rtol=1e-2)
    dW_h_close = torch.allclose(dW_h_py, dW_h_cuda, atol=atol, rtol=1e-2)

    print(f"\nBackward comparison:")
    print(f"  dx matches: {dx_close}")
    print(f"  dW_h matches: {dW_h_close}")

    if not dx_close:
        dx_diff = (dx_py - dx_cuda).abs().max().item()
        print(f"  dx max diff: {dx_diff}")

    if not dW_h_close:
        dW_h_diff = (dW_h_py - dW_h_cuda).abs().max().item()
        print(f"  dW_h max diff: {dW_h_diff}")
else:
    print("Skipping CUDA backward test - kernel not available")

print("\n" + "="*60)
print("Testing E39 Full Model (E39NoBiasCell)")
print("="*60)

from elman.models.e39_no_bias import E39NoBiasCell

# Test with cell class
cell = E39NoBiasCell(D, w_h_mode='none').to(device).to(dtype)

# Put it in train mode
cell.train()

# Test input
x_cell = torch.randn(T, B, D, device=device, dtype=dtype)

# Python mode (disable CUDA by setting flag)
import elman.models.e39_no_bias as e39_module
original_flag = e39_module.E39_CUDA_AVAILABLE
e39_module.E39_CUDA_AVAILABLE = False

output_cell_py, h_cell_py = cell(x_cell)
print(f"Cell Python forward: output={output_cell_py.shape}, h={h_cell_py.shape}")

# CUDA mode
e39_module.E39_CUDA_AVAILABLE = E39_CUDA_AVAILABLE

if E39_CUDA_AVAILABLE:
    output_cell_cuda, h_cell_cuda = cell(x_cell)
    print(f"Cell CUDA forward: output={output_cell_cuda.shape}, h={h_cell_cuda.shape}")

    out_close = torch.allclose(output_cell_py, output_cell_cuda, atol=atol, rtol=1e-2)
    h_close = torch.allclose(h_cell_py, h_cell_cuda, atol=atol, rtol=1e-2)

    print(f"\nCell comparison:")
    print(f"  output matches: {out_close}")
    print(f"  h matches: {h_close}")

# Restore
e39_module.E39_CUDA_AVAILABLE = original_flag

print("\n" + "="*60)
print("Summary")
print("="*60)

all_passed = True
if E39_CUDA_AVAILABLE:
    if not (h_close and output_close):
        print("FAILED: Forward pass mismatch")
        all_passed = False
    if not (dx_close and dW_h_close):
        print("FAILED: Backward pass mismatch")
        all_passed = False

    if all_passed:
        print("ALL TESTS PASSED!")
else:
    print("CUDA kernel not available - could only test Python implementation")
