#!/usr/bin/env python3
"""
Test E43 (Scalar Decay) and E44 (Diagonal W) CUDA kernels against Python implementations.
"""

import torch
import torch.nn.functional as F
import hasty_pytorch_lib as hasty

def test_e43_forward():
    """Test E43 scalar decay forward pass."""
    print("\n=== Testing E43 Scalar Decay Forward ===")

    T, B, D = 16, 4, 128
    device = 'cuda'
    dtype = torch.bfloat16

    # Create inputs
    x = torch.randn(T, B, D, device=device, dtype=dtype)
    h0 = torch.zeros(B, D, device=device, dtype=dtype)
    init_lambda = 0.5
    log_lambda = torch.tensor(init_lambda, device=device, dtype=dtype).logit().unsqueeze(0)  # [1]
    b = torch.randn(D, device=device, dtype=dtype) * 0.1

    # CUDA forward
    h_cuda, output_cuda, v_cuda = hasty.e43_scalar_decay_forward(True, x, h0, log_lambda, b)

    # Python reference
    lambda_val = torch.sigmoid(log_lambda).item()
    h_list = [h0]
    output_list = []

    for t in range(T):
        h_prev = h_list[-1]
        h_new = lambda_val * (x[t] + h_prev) + b
        h_list.append(h_new)
        output = h_new * F.silu(h_new)
        output_list.append(output)

    h_py = torch.stack(h_list, dim=0)
    output_py = torch.stack(output_list, dim=0)

    # Compare
    h_err = (h_cuda - h_py).abs().max().item()
    out_err = (output_cuda - output_py).abs().max().item()

    print(f"  Lambda: {lambda_val:.4f}")
    print(f"  Hidden state max error: {h_err:.6f}")
    print(f"  Output max error: {out_err:.6f}")

    # bf16 tolerance is larger due to reduced precision
    assert h_err < 0.05, f"Hidden state error too large: {h_err}"
    assert out_err < 0.1, f"Output error too large: {out_err}"
    print("  PASSED!")

    return True


def test_e43_backward():
    """Test E43 scalar decay backward pass."""
    print("\n=== Testing E43 Scalar Decay Backward ===")

    T, B, D = 8, 2, 64
    device = 'cuda'
    dtype = torch.bfloat16

    # Create inputs
    x = torch.randn(T, B, D, device=device, dtype=dtype, requires_grad=True)
    h0 = torch.zeros(B, D, device=device, dtype=dtype)
    init_lambda = 0.6
    log_lambda = torch.tensor(init_lambda, device=device, dtype=dtype).logit().unsqueeze(0).requires_grad_(True)
    b = torch.randn(D, device=device, dtype=dtype, requires_grad=True)

    # CUDA forward + backward
    h_cuda, output_cuda, v_cuda = hasty.e43_scalar_decay_forward(True, x, h0, log_lambda, b)
    d_output = torch.randn_like(output_cuda)

    dx_cuda, d_log_lambda_cuda, db_cuda = hasty.e43_scalar_decay_backward(
        log_lambda, x, h_cuda, v_cuda, d_output)

    # Python reference with autograd
    x_py = x.detach().clone().requires_grad_(True)
    log_lambda_py = log_lambda.detach().clone().requires_grad_(True)
    b_py = b.detach().clone().requires_grad_(True)

    lambda_val = torch.sigmoid(log_lambda_py)
    h_list = [h0]
    output_list = []

    for t in range(T):
        h_prev = h_list[-1]
        h_new = lambda_val * (x_py[t] + h_prev) + b_py
        h_list.append(h_new)
        output = h_new * F.silu(h_new)
        output_list.append(output)

    output_py = torch.stack(output_list, dim=0)
    loss = (output_py * d_output).sum()
    loss.backward()

    # Compare gradients
    dx_err = (dx_cuda - x_py.grad).abs().max().item()
    db_err = (db_cuda - b_py.grad).abs().max().item()
    d_log_lambda_err = (d_log_lambda_cuda - log_lambda_py.grad).abs().max().item()

    print(f"  dx max error: {dx_err:.6f}")
    print(f"  db max error: {db_err:.6f}")
    print(f"  d_log_lambda max error: {d_log_lambda_err:.6f}")

    # Looser tolerance for bf16 - gradients accumulate errors
    assert dx_err < 0.5, f"dx error too large: {dx_err}"
    assert db_err < 2.0, f"db error too large: {db_err}"
    assert d_log_lambda_err < 5.0, f"d_log_lambda error too large: {d_log_lambda_err}"
    print("  PASSED!")

    return True


def test_e44_forward():
    """Test E44 diagonal W forward pass."""
    print("\n=== Testing E44 Diagonal W Forward ===")

    T, B, D = 16, 4, 128
    device = 'cuda'
    dtype = torch.bfloat16

    # Create inputs
    x = torch.randn(T, B, D, device=device, dtype=dtype)
    h0 = torch.zeros(B, D, device=device, dtype=dtype)
    init_decay = 0.5
    log_d = torch.full((D,), init_decay, device=device, dtype=dtype).logit()  # [D]
    b = torch.randn(D, device=device, dtype=dtype) * 0.1

    # CUDA forward
    h_cuda, output_cuda, v_cuda = hasty.e44_diagonal_w_forward(True, x, h0, log_d, b)

    # Python reference
    d_val = torch.sigmoid(log_d)  # [D]
    h_list = [h0]
    output_list = []

    for t in range(T):
        h_prev = h_list[-1]
        h_new = d_val * (x[t] + h_prev) + b
        h_list.append(h_new)
        output = h_new * F.silu(h_new)
        output_list.append(output)

    h_py = torch.stack(h_list, dim=0)
    output_py = torch.stack(output_list, dim=0)

    # Compare
    h_err = (h_cuda - h_py).abs().max().item()
    out_err = (output_cuda - output_py).abs().max().item()

    print(f"  Decay mean: {d_val.mean():.4f}")
    print(f"  Hidden state max error: {h_err:.6f}")
    print(f"  Output max error: {out_err:.6f}")

    # bf16 tolerance is larger due to reduced precision
    assert h_err < 0.05, f"Hidden state error too large: {h_err}"
    assert out_err < 0.1, f"Output error too large: {out_err}"
    print("  PASSED!")

    return True


def test_e44_backward():
    """Test E44 diagonal W backward pass."""
    print("\n=== Testing E44 Diagonal W Backward ===")

    T, B, D = 8, 2, 64
    device = 'cuda'
    dtype = torch.bfloat16

    # Create inputs
    x = torch.randn(T, B, D, device=device, dtype=dtype, requires_grad=True)
    h0 = torch.zeros(B, D, device=device, dtype=dtype)
    init_decay = 0.6
    log_d = torch.full((D,), init_decay, device=device, dtype=dtype).logit().requires_grad_(True)
    b = torch.randn(D, device=device, dtype=dtype, requires_grad=True)

    # CUDA forward + backward
    h_cuda, output_cuda, v_cuda = hasty.e44_diagonal_w_forward(True, x, h0, log_d, b)
    d_output = torch.randn_like(output_cuda)

    dx_cuda, d_log_d_cuda, db_cuda = hasty.e44_diagonal_w_backward(
        log_d, x, h_cuda, v_cuda, d_output)

    # Python reference with autograd
    x_py = x.detach().clone().requires_grad_(True)
    log_d_py = log_d.detach().clone().requires_grad_(True)
    b_py = b.detach().clone().requires_grad_(True)

    d_val = torch.sigmoid(log_d_py)
    h_list = [h0]
    output_list = []

    for t in range(T):
        h_prev = h_list[-1]
        h_new = d_val * (x_py[t] + h_prev) + b_py
        h_list.append(h_new)
        output = h_new * F.silu(h_new)
        output_list.append(output)

    output_py = torch.stack(output_list, dim=0)
    loss = (output_py * d_output).sum()
    loss.backward()

    # Compare gradients
    dx_err = (dx_cuda - x_py.grad).abs().max().item()
    db_err = (db_cuda - b_py.grad).abs().max().item()
    d_log_d_err = (d_log_d_cuda - log_d_py.grad).abs().max().item()

    print(f"  dx max error: {dx_err:.6f}")
    print(f"  db max error: {db_err:.6f}")
    print(f"  d_log_d max error: {d_log_d_err:.6f}")

    # Looser tolerance for bf16 - gradients accumulate errors
    assert dx_err < 0.5, f"dx error too large: {dx_err}"
    assert db_err < 2.0, f"db error too large: {db_err}"
    assert d_log_d_err < 5.0, f"d_log_d error too large: {d_log_d_err}"
    print("  PASSED!")

    return True


def test_e43_forward_f32():
    """Test E43 scalar decay forward pass with float32 for high precision."""
    print("\n=== Testing E43 Scalar Decay Forward (float32) ===")

    T, B, D = 16, 4, 128
    device = 'cuda'
    dtype = torch.float32

    x = torch.randn(T, B, D, device=device, dtype=dtype)
    h0 = torch.zeros(B, D, device=device, dtype=dtype)
    init_lambda = 0.5
    log_lambda = torch.tensor(init_lambda, device=device, dtype=dtype).logit().unsqueeze(0)
    b = torch.randn(D, device=device, dtype=dtype) * 0.1

    h_cuda, output_cuda, v_cuda = hasty.e43_scalar_decay_forward(True, x, h0, log_lambda, b)

    lambda_val = torch.sigmoid(log_lambda).item()
    h_list = [h0]
    output_list = []

    for t in range(T):
        h_prev = h_list[-1]
        h_new = lambda_val * (x[t] + h_prev) + b
        h_list.append(h_new)
        output = h_new * F.silu(h_new)
        output_list.append(output)

    h_py = torch.stack(h_list, dim=0)
    output_py = torch.stack(output_list, dim=0)

    h_err = (h_cuda - h_py).abs().max().item()
    out_err = (output_cuda - output_py).abs().max().item()

    print(f"  Lambda: {lambda_val:.4f}")
    print(f"  Hidden state max error: {h_err:.6e}")
    print(f"  Output max error: {out_err:.6e}")

    assert h_err < 1e-5, f"Hidden state error too large: {h_err}"
    assert out_err < 1e-5, f"Output error too large: {out_err}"
    print("  PASSED!")
    return True


def test_e44_forward_f32():
    """Test E44 diagonal W forward pass with float32 for high precision."""
    print("\n=== Testing E44 Diagonal W Forward (float32) ===")

    T, B, D = 16, 4, 128
    device = 'cuda'
    dtype = torch.float32

    x = torch.randn(T, B, D, device=device, dtype=dtype)
    h0 = torch.zeros(B, D, device=device, dtype=dtype)
    init_decay = 0.5
    log_d = torch.full((D,), init_decay, device=device, dtype=dtype).logit()
    b = torch.randn(D, device=device, dtype=dtype) * 0.1

    h_cuda, output_cuda, v_cuda = hasty.e44_diagonal_w_forward(True, x, h0, log_d, b)

    d_val = torch.sigmoid(log_d)
    h_list = [h0]
    output_list = []

    for t in range(T):
        h_prev = h_list[-1]
        h_new = d_val * (x[t] + h_prev) + b
        h_list.append(h_new)
        output = h_new * F.silu(h_new)
        output_list.append(output)

    h_py = torch.stack(h_list, dim=0)
    output_py = torch.stack(output_list, dim=0)

    h_err = (h_cuda - h_py).abs().max().item()
    out_err = (output_cuda - output_py).abs().max().item()

    print(f"  Decay mean: {d_val.mean():.4f}")
    print(f"  Hidden state max error: {h_err:.6e}")
    print(f"  Output max error: {out_err:.6e}")

    assert h_err < 1e-5, f"Hidden state error too large: {h_err}"
    assert out_err < 1e-5, f"Output error too large: {out_err}"
    print("  PASSED!")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Testing E43 (Scalar Decay) and E44 (Diagonal W) CUDA Kernels")
    print("=" * 60)

    all_passed = True

    try:
        all_passed &= test_e43_forward()
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False

    try:
        all_passed &= test_e43_backward()
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False

    try:
        all_passed &= test_e44_forward()
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False

    try:
        all_passed &= test_e44_backward()
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False

    try:
        all_passed &= test_e43_forward_f32()
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False

    try:
        all_passed &= test_e44_forward_f32()
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)
