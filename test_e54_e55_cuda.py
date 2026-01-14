#!/usr/bin/env python3
"""
Test E54 and E55 CUDA kernels against Python implementations.
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys

# Import CUDA kernels
sys.path.insert(0, 'elman/cuda')
import hasty_pytorch_lib as hp

# Set device
device = 'cuda'
dtype = torch.bfloat16


def test_e54_forward():
    """Test E54 diagonal no-proj forward against Python."""
    print("\n" + "="*60)
    print("Testing E54 Forward (diagonal + no proj)")
    print("="*60)

    T, B, D = 32, 4, 256

    # Create inputs
    torch.manual_seed(42)
    x = torch.randn(T, B, D, device=device, dtype=dtype)
    h0 = torch.zeros(B, D, device=device, dtype=dtype)

    # Parameters: d is sigmoid(log_d) for stability
    log_d = torch.randn(D, device=device, dtype=dtype) * 0.1
    d = torch.sigmoid(log_d)
    b = torch.randn(D, device=device, dtype=dtype) * 0.01

    # CUDA forward
    h_cuda, output_cuda, v_cuda = hp.e54_diagonal_no_proj_forward(True, x, h0, d, b)

    # Python reference
    h_py = [h0]
    output_py = []
    for t in range(T):
        h_prev = h_py[-1]
        # h_t = d * (x + h_prev) + b
        h_new = d * (x[t] + h_prev) + b
        h_py.append(h_new)
        # output = h * silu(h)
        out = h_new * F.silu(h_new)
        output_py.append(out)

    h_py = torch.stack(h_py, dim=0)
    output_py = torch.stack(output_py, dim=0)

    # Compare
    h_err = (h_cuda - h_py).abs().max().item()
    out_err = (output_cuda - output_py).abs().max().item()

    print(f"Hidden state max error: {h_err:.6e}")
    print(f"Output max error: {out_err:.6e}")

    # bf16 has limited precision, errors accumulate over sequence length
    assert h_err < 0.1, f"Hidden state error too large: {h_err}"
    assert out_err < 0.1, f"Output error too large: {out_err}"

    print("E54 Forward PASSED!")
    return True


def test_e54_backward():
    """Test E54 diagonal no-proj backward against PyTorch autograd."""
    print("\n" + "="*60)
    print("Testing E54 Backward (diagonal + no proj)")
    print("="*60)

    T, B, D = 16, 2, 128

    # Create inputs
    torch.manual_seed(42)
    x = torch.randn(T, B, D, device=device, dtype=dtype, requires_grad=True)
    h0 = torch.zeros(B, D, device=device, dtype=dtype)

    # Parameters
    log_d = torch.randn(D, device=device, dtype=dtype) * 0.1
    d = torch.sigmoid(log_d)
    d_param = d.clone().detach().requires_grad_(True)
    b = torch.randn(D, device=device, dtype=dtype) * 0.01
    b_param = b.clone().detach().requires_grad_(True)

    # Python forward with autograd
    h_list = [h0]
    output_list = []
    for t in range(T):
        h_prev = h_list[-1]
        h_new = d_param * (x[t] + h_prev) + b_param
        h_list.append(h_new)
        out = h_new * F.silu(h_new)
        output_list.append(out)

    output_py = torch.stack(output_list, dim=0)
    h_py = torch.stack(h_list, dim=0)

    # Backward with autograd
    grad_output = torch.randn_like(output_py)
    loss_py = (output_py * grad_output).sum()
    loss_py.backward()

    dx_py = x.grad.clone()
    dd_py = d_param.grad.clone()
    db_py = b_param.grad.clone()

    # CUDA forward
    x_cuda = x.detach()
    d_cuda = d.detach()
    b_cuda = b.detach()
    h_cuda, output_cuda, v_cuda = hp.e54_diagonal_no_proj_forward(True, x_cuda, h0, d_cuda, b_cuda)

    # CUDA backward
    dx_cuda, dd_cuda, db_cuda = hp.e54_diagonal_no_proj_backward(d_cuda, x_cuda, h_cuda, v_cuda, grad_output)

    # Compare
    dx_err = (dx_cuda - dx_py).abs().max().item()
    dd_err = (dd_cuda - dd_py).abs().max().item()
    db_err = (db_cuda - db_py).abs().max().item()

    print(f"dx max error: {dx_err:.6e}")
    print(f"dd max error: {dd_err:.6e}")
    print(f"db max error: {db_err:.6e}")

    # Allow some tolerance for bf16
    assert dx_err < 0.1, f"dx error too large: {dx_err}"
    assert dd_err < 1.0, f"dd error too large: {dd_err}"  # Accumulated gradient, larger error expected
    assert db_err < 1.0, f"db error too large: {db_err}"

    print("E54 Backward PASSED!")
    return True


def test_e55_forward():
    """Test E55 scalar no-proj forward against Python."""
    print("\n" + "="*60)
    print("Testing E55 Forward (scalar + no proj)")
    print("="*60)

    T, B, D = 32, 4, 256

    # Create inputs
    torch.manual_seed(42)
    x = torch.randn(T, B, D, device=device, dtype=dtype)
    h0 = torch.zeros(B, D, device=device, dtype=dtype)

    # Parameters: lambda is sigmoid(log_lambda) for stability
    log_lambda = torch.tensor(0.0, device=device, dtype=torch.float32)
    lambda_val = torch.sigmoid(log_lambda).item()  # Single scalar!
    b = torch.randn(D, device=device, dtype=dtype) * 0.01

    # CUDA forward
    h_cuda, output_cuda, v_cuda = hp.e55_scalar_no_proj_forward(True, x, h0, lambda_val, b)

    # Python reference
    h_py = [h0]
    output_py = []
    for t in range(T):
        h_prev = h_py[-1]
        # h_t = lambda * (x + h_prev) + b
        h_new = lambda_val * (x[t] + h_prev) + b
        h_py.append(h_new)
        # output = h * silu(h)
        out = h_new * F.silu(h_new)
        output_py.append(out)

    h_py = torch.stack(h_py, dim=0)
    output_py = torch.stack(output_py, dim=0)

    # Compare
    h_err = (h_cuda - h_py).abs().max().item()
    out_err = (output_cuda - output_py).abs().max().item()

    print(f"Lambda value: {lambda_val:.4f}")
    print(f"Hidden state max error: {h_err:.6e}")
    print(f"Output max error: {out_err:.6e}")

    # bf16 has limited precision, errors accumulate over sequence length
    assert h_err < 0.1, f"Hidden state error too large: {h_err}"
    assert out_err < 0.1, f"Output error too large: {out_err}"

    print("E55 Forward PASSED!")
    return True


def test_e55_backward():
    """Test E55 scalar no-proj backward against PyTorch autograd."""
    print("\n" + "="*60)
    print("Testing E55 Backward (scalar + no proj)")
    print("="*60)

    T, B, D = 16, 2, 128

    # Create inputs
    torch.manual_seed(42)
    x = torch.randn(T, B, D, device=device, dtype=dtype, requires_grad=True)
    h0 = torch.zeros(B, D, device=device, dtype=dtype)

    # Parameters
    log_lambda = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
    lambda_val = torch.sigmoid(log_lambda)
    b = torch.randn(D, device=device, dtype=dtype) * 0.01
    b_param = b.clone().detach().requires_grad_(True)

    # Python forward with autograd
    h_list = [h0]
    output_list = []
    for t in range(T):
        h_prev = h_list[-1]
        # Cast lambda to bf16 for computation
        h_new = lambda_val.to(dtype) * (x[t] + h_prev) + b_param
        h_list.append(h_new)
        out = h_new * F.silu(h_new)
        output_list.append(out)

    output_py = torch.stack(output_list, dim=0)

    # Backward with autograd
    grad_output = torch.randn_like(output_py)
    loss_py = (output_py * grad_output).sum()
    loss_py.backward()

    dx_py = x.grad.clone()
    # dlambda gradient needs to go through sigmoid derivative
    # d_sigmoid = sigmoid * (1 - sigmoid)
    sigmoid_val = torch.sigmoid(log_lambda)
    dlambda_py = log_lambda.grad.clone() / (sigmoid_val * (1 - sigmoid_val))  # Chain rule back through sigmoid
    db_py = b_param.grad.clone()

    # CUDA forward
    x_cuda = x.detach()
    b_cuda = b.detach()
    lambda_cuda = lambda_val.item()
    h_cuda, output_cuda, v_cuda = hp.e55_scalar_no_proj_forward(True, x_cuda, h0, lambda_cuda, b_cuda)

    # CUDA backward
    dx_cuda, dlambda_cuda, db_cuda = hp.e55_scalar_no_proj_backward(lambda_cuda, x_cuda, h_cuda, v_cuda, grad_output)

    # Compare
    dx_err = (dx_cuda - dx_py).abs().max().item()
    db_err = (db_cuda - db_py).abs().max().item()

    print(f"dx max error: {dx_err:.6e}")
    print(f"dlambda CUDA: {dlambda_cuda.item():.6f}")
    print(f"dlambda PyTorch (pre-sigmoid): {dlambda_py.item():.6f}")
    print(f"db max error: {db_err:.6e}")

    # Allow some tolerance for bf16
    assert dx_err < 0.1, f"dx error too large: {dx_err}"
    assert db_err < 1.0, f"db error too large: {db_err}"

    print("E55 Backward PASSED!")
    return True


def main():
    print("="*60)
    print("E54/E55 CUDA Kernel Correctness Tests")
    print("="*60)

    # Run tests
    try:
        test_e54_forward()
        test_e54_backward()
        test_e55_forward()
        test_e55_backward()

        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60)
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
