#!/usr/bin/env python3
"""Test E45 and E46 CUDA kernels against Python implementations."""

import sys
import os

# Add cuda directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'elman', 'cuda'))

import torch
import torch.nn.functional as F
import numpy as np
import hasty_pytorch_lib as hp

torch.manual_seed(42)

def silu(x):
    return x * torch.sigmoid(x)


def test_e45_forward():
    """Test E45: Pure accumulation h_t = h_{t-1} + x_t."""
    print("\n=== E45 Pure Accumulation Forward Test ===")

    T, B, D = 32, 4, 128
    x = torch.randn(T, B, D, device='cuda', dtype=torch.bfloat16)
    h0 = torch.zeros(B, D, device='cuda', dtype=torch.bfloat16)

    # Python reference
    h_py = [h0]
    out_py = []
    for t in range(T):
        h_t = h_py[-1] + x[t]
        h_py.append(h_t)
        # Self-gate: output = h * silu(h)
        out_py.append(h_t * silu(h_t))
    h_py = torch.stack(h_py)
    out_py = torch.stack(out_py)

    # CUDA kernel
    h_cuda, out_cuda = hp.e45_pure_accumulation_forward(True, x, h0)

    # Compare - use relative error for outputs (h can grow large with accumulation)
    h_err = (h_py - h_cuda).abs().max().item()
    out_abs_err = (out_py - out_cuda).abs().max().item()
    out_rel_err = ((out_py - out_cuda).abs() / (out_py.abs() + 1e-6)).max().item()

    print(f"  Hidden state max error: {h_err:.6e}")
    print(f"  Output max abs error: {out_abs_err:.6e}")
    print(f"  Output max rel error: {out_rel_err:.6e}")

    assert h_err < 1e-2, f"E45 forward hidden state error too large: {h_err}"
    assert out_rel_err < 0.02, f"E45 forward output relative error too large: {out_rel_err}"
    print("  PASSED")


def test_e45_backward():
    """Test E45 backward gradient correctness using PyTorch autograd."""
    print("\n=== E45 Pure Accumulation Backward Test ===")

    T, B, D = 16, 2, 64

    # PyTorch autograd reference (float32 for accuracy)
    x_ref = torch.randn(T, B, D, device='cuda', dtype=torch.float32, requires_grad=True)
    h0_ref = torch.zeros(B, D, device='cuda', dtype=torch.float32)

    # E45: h_t = h_{t-1} + x_t, output = h * silu(h) = h^2 * sigmoid(h)
    h_list = [h0_ref]
    for t in range(T):
        h_t = x_ref[t] + h_list[-1]
        h_list.append(h_t)

    h_ref = torch.stack(h_list)  # [T+1, B, D]
    out_ref = h_ref[1:] * (h_ref[1:] * torch.sigmoid(h_ref[1:]))  # [T, B, D]

    # Random upstream gradient
    d_output = torch.randn_like(out_ref)

    # PyTorch autograd backward
    out_ref.backward(d_output)
    dx_ref = x_ref.grad.clone()

    # CUDA kernel
    x_cuda = x_ref.detach().clone().bfloat16()
    h0_cuda = h0_ref.clone().bfloat16()

    h_cuda, out_cuda = hp.e45_pure_accumulation_forward(True, x_cuda, h0_cuda)
    dx_cuda, = hp.e45_pure_accumulation_backward(h_cuda, d_output.bfloat16())

    # Compare
    dx_abs_err = (dx_ref.bfloat16() - dx_cuda).abs().max().item()
    dx_rel_err = ((dx_ref.bfloat16() - dx_cuda).abs() / (dx_ref.abs().bfloat16() + 1e-6)).mean().item()

    print(f"  dx max abs error: {dx_abs_err:.6e}")
    print(f"  dx mean rel error: {dx_rel_err:.6e}")

    # BF16 has limited precision, use reasonable tolerance
    assert dx_abs_err < 1.0, f"E45 backward dx abs error too large: {dx_abs_err}"
    assert dx_rel_err < 0.1, f"E45 backward dx rel error too large: {dx_rel_err}"
    print("  PASSED")


def test_e45b_forward():
    """Test E45b: h_t = x_t + alpha * h_{t-1}."""
    print("\n=== E45b With Decay Forward Test ===")

    T, B, D = 32, 4, 128
    x = torch.randn(T, B, D, device='cuda', dtype=torch.bfloat16)
    h0 = torch.zeros(B, D, device='cuda', dtype=torch.bfloat16)
    alpha = 0.9

    # Python reference
    h_py = [h0]
    out_py = []
    for t in range(T):
        h_t = x[t] + alpha * h_py[-1]
        h_py.append(h_t)
        out_py.append(h_t * silu(h_t))
    h_py = torch.stack(h_py)
    out_py = torch.stack(out_py)

    # CUDA kernel
    h_cuda, out_cuda = hp.e45b_with_decay_forward(True, x, h0, alpha)

    # Compare using atol+rtol like numpy.allclose
    h_abs_err = (h_py - h_cuda).abs().max().item()
    out_abs_err = (out_py - out_cuda).abs().max().item()

    # Check with atol=0.1, rtol=0.05 (similar to allclose defaults for BF16)
    h_close = torch.allclose(h_py, h_cuda, atol=0.1, rtol=0.05)
    out_close = torch.allclose(out_py, out_cuda, atol=0.1, rtol=0.05)

    print(f"  Hidden state max abs error: {h_abs_err:.6e}")
    print(f"  Output max abs error: {out_abs_err:.6e}")
    print(f"  Hidden states allclose: {h_close}")
    print(f"  Outputs allclose: {out_close}")

    assert h_close, f"E45b forward hidden state not close"
    assert out_close, f"E45b forward output not close"
    print("  PASSED")


def test_e46_forward():
    """Test E46: h_t = W @ (x_t + h_{t-1}) + b."""
    print("\n=== E46 No In-Proj Forward Test ===")

    T, B, D = 32, 4, 128
    x = torch.randn(T, B, D, device='cuda', dtype=torch.bfloat16)
    h0 = torch.zeros(B, D, device='cuda', dtype=torch.bfloat16)
    W = torch.randn(D, D, device='cuda', dtype=torch.bfloat16) * 0.01
    b = torch.randn(D, device='cuda', dtype=torch.bfloat16) * 0.01

    # Python reference
    h_py = [h0]
    out_py = []
    for t in range(T):
        sum_xh = x[t] + h_py[-1]
        # W @ (x + h) + b - LINEAR recurrence
        h_t = F.linear(sum_xh, W, b)
        h_py.append(h_t)
        # Self-gate
        out_py.append(h_t * silu(h_t))
    h_py = torch.stack(h_py)
    out_py = torch.stack(out_py)

    # CUDA kernel
    h_cuda, out_cuda, v_cuda = hp.e46_no_in_proj_forward(True, x, h0, W, b)

    # Compare
    h_err = (h_py - h_cuda).abs().max().item()
    out_err = (out_py - out_cuda).abs().max().item()

    print(f"  Hidden state max error: {h_err:.6e}")
    print(f"  Output max error: {out_err:.6e}")

    assert h_err < 0.1, f"E46 forward hidden state error too large: {h_err}"
    assert out_err < 0.1, f"E46 forward output error too large: {out_err}"
    print("  PASSED")


def test_e46_backward():
    """Test E46 backward gradient correctness using PyTorch autograd."""
    print("\n=== E46 No In-Proj Backward Test ===")

    T, B, D = 8, 2, 32

    # PyTorch autograd reference
    x_ref = torch.randn(T, B, D, device='cuda', dtype=torch.float32, requires_grad=True)
    h0_ref = torch.zeros(B, D, device='cuda', dtype=torch.float32)
    W_ref = torch.nn.Parameter(torch.randn(D, D, device='cuda', dtype=torch.float32) * 0.01)
    b_ref = torch.nn.Parameter(torch.randn(D, device='cuda', dtype=torch.float32) * 0.01)

    # Python forward
    h_list = [h0_ref]
    out_list = []
    for t in range(T):
        sum_xh = x_ref[t] + h_list[-1]
        h_t = F.linear(sum_xh, W_ref, b_ref)
        h_list.append(h_t)
        out_list.append(h_t * silu(h_t))
    out_ref = torch.stack(out_list)

    # Random upstream gradient
    d_output = torch.randn_like(out_ref)

    # PyTorch autograd backward
    out_ref.backward(d_output)
    dx_ref = x_ref.grad.clone()
    dW_ref = W_ref.grad.clone()
    db_ref = b_ref.grad.clone()

    # CUDA kernel (using float32 for comparison)
    x_cuda = x_ref.detach().clone().bfloat16()
    W_cuda = W_ref.detach().clone().bfloat16()
    b_cuda = b_ref.detach().clone().bfloat16()
    h0_cuda = h0_ref.clone().bfloat16()

    h_cuda, out_cuda, v_cuda = hp.e46_no_in_proj_forward(True, x_cuda, h0_cuda, W_cuda, b_cuda)
    dx_cuda, dW_cuda, db_cuda = hp.e46_no_in_proj_backward(
        W_cuda, x_cuda, h_cuda, v_cuda, d_output.bfloat16())

    # Compare (with tolerance for bf16)
    dx_err = (dx_ref - dx_cuda.float()).abs().max().item()
    dW_err = (dW_ref - dW_cuda.float()).abs().max().item()
    db_err = (db_ref - db_cuda.float()).abs().max().item()

    print(f"  dx max error: {dx_err:.6e}")
    print(f"  dW max error: {dW_err:.6e}")
    print(f"  db max error: {db_err:.6e}")

    # BF16 has lower precision
    assert dx_err < 0.5, f"E46 backward dx error too large: {dx_err}"
    assert dW_err < 1.0, f"E46 backward dW error too large: {dW_err}"
    assert db_err < 0.5, f"E46 backward db error too large: {db_err}"
    print("  PASSED")


def test_e45_gradcheck():
    """Numerical gradient check for E45."""
    print("\n=== E45 Gradient Check (numerical) ===")

    T, B, D = 4, 2, 8
    eps = 1e-3

    x = torch.randn(T, B, D, device='cuda', dtype=torch.float32)
    h0 = torch.zeros(B, D, device='cuda', dtype=torch.float32)

    def forward_fn(x_in):
        h = [h0]
        for t in range(T):
            h_t = x_in[t] + h[-1]
            h.append(h_t)
        h = torch.stack(h[1:])
        return (h * silu(h)).sum()

    # Analytical gradient
    x.requires_grad_(True)
    loss = forward_fn(x)
    loss.backward()
    dx_analytical = x.grad.clone()

    # Numerical gradient
    dx_numerical = torch.zeros_like(x)
    with torch.no_grad():
        for t in range(T):
            for b in range(B):
                for d in range(D):
                    x_plus = x.clone()
                    x_plus[t, b, d] += eps
                    loss_plus = forward_fn(x_plus)

                    x_minus = x.clone()
                    x_minus[t, b, d] -= eps
                    loss_minus = forward_fn(x_minus)

                    dx_numerical[t, b, d] = (loss_plus - loss_minus) / (2 * eps)

    err = (dx_analytical - dx_numerical).abs().max().item()
    print(f"  Gradient check max error: {err:.6e}")
    assert err < 1e-2, f"E45 gradient check failed: {err}"
    print("  PASSED")


if __name__ == '__main__':
    print("Testing E45/E45b/E46 CUDA kernels...")

    test_e45_forward()
    test_e45_backward()
    test_e45b_forward()
    test_e45_gradcheck()
    test_e46_forward()
    test_e46_backward()

    print("\n" + "="*50)
    print("All tests PASSED!")
    print("="*50)
