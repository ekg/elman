#!/usr/bin/env python3
"""
Test E48 (No Projections) and E51 (No Self-Gate) CUDA kernels
against their Python reference implementations.
"""

import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/home/erikg/elman/elman/cuda')
import hasty_pytorch_lib as hasty

# Device and precision settings
device = 'cuda'
dtype = torch.bfloat16
# BF16 has lower precision - use relative tolerance for large values
# For values ~100-200, 1% relative error = 1-2 absolute error
# Backward pass with accumulated gradients can have larger numerical errors
atol = 2.0  # Absolute tolerance
rtol = 0.02  # 2% relative tolerance

def test_e48_forward():
    """Test E48 No Projections forward pass."""
    print("\n" + "="*60)
    print("Testing E48 (No Projections) Forward")
    print("="*60)

    torch.manual_seed(42)
    B, T, D = 4, 32, 128

    # Create inputs
    x = torch.randn(T, B, D, device=device, dtype=dtype)
    h0 = torch.zeros(B, D, device=device, dtype=dtype)
    W = torch.randn(D, D, device=device, dtype=dtype) * 0.1
    b = torch.zeros(D, device=device, dtype=dtype)

    # CUDA forward
    h_cuda, output_cuda, v_cuda = hasty.e48_no_projections_forward(True, x, h0, W, b)

    # Python reference forward (from E48NoProjectionsCell)
    h_list = [h0]
    output_list = []

    # Batched GEMM for W @ x
    x_flat = x.reshape(T * B, D)
    Wx_all = (x_flat @ W.T).reshape(T, B, D)

    for t in range(T):
        h_prev = h_list[-1]
        Wx_t = Wx_all[t]
        Wh = h_prev @ W.T
        h_new = Wx_t + Wh + b
        h_list.append(h_new)

        # Self-gating
        output = h_new * F.silu(h_new)
        output_list.append(output)

    h_ref = torch.stack(h_list, dim=0)
    output_ref = torch.stack(output_list, dim=0)

    # Compare using allclose (handles relative tolerance)
    h_match = torch.allclose(h_cuda, h_ref, atol=atol, rtol=rtol)
    out_match = torch.allclose(output_cuda, output_ref, atol=atol, rtol=rtol)

    h_err = (h_cuda - h_ref).abs().max().item()
    out_err = (output_cuda - output_ref).abs().max().item()
    out_rel_err = ((output_cuda - output_ref).abs() / (output_ref.abs() + 1e-6)).max().item()

    print(f"h max error: {h_err:.6f}")
    print(f"output max error: {out_err:.6f} (relative: {out_rel_err:.6f})")

    if h_match and out_match:
        print("[PASS] E48 forward matches Python reference")
        return True
    else:
        print("[FAIL] E48 forward mismatch!")
        return False

def test_e48_backward():
    """Test E48 No Projections backward pass."""
    print("\n" + "="*60)
    print("Testing E48 (No Projections) Backward")
    print("="*60)

    torch.manual_seed(42)
    B, T, D = 4, 16, 64  # Smaller for gradient checking

    # Create inputs
    x = torch.randn(T, B, D, device=device, dtype=dtype, requires_grad=True)
    h0 = torch.zeros(B, D, device=device, dtype=dtype)
    W = torch.randn(D, D, device=device, dtype=dtype, requires_grad=True) * 0.1
    b = torch.zeros(D, device=device, dtype=dtype, requires_grad=True)

    # Python reference (for gradient comparison)
    x_ref = x.clone().detach().requires_grad_(True)
    W_ref = W.clone().detach().requires_grad_(True)
    b_ref = b.clone().detach().requires_grad_(True)

    # Python forward
    h_list = [h0]
    output_list = []
    x_flat = x_ref.reshape(T * B, D)
    Wx_all = (x_flat @ W_ref.T).reshape(T, B, D)

    for t in range(T):
        h_prev = h_list[-1]
        Wx_t = Wx_all[t]
        Wh = h_prev @ W_ref.T
        h_new = Wx_t + Wh + b_ref
        h_list.append(h_new)
        output = h_new * F.silu(h_new)
        output_list.append(output)

    output_ref = torch.stack(output_list, dim=0)

    # Create random d_output
    d_output = torch.randn_like(output_ref)

    # Python backward
    output_ref.backward(d_output)
    dx_ref = x_ref.grad
    dW_ref = W_ref.grad
    db_ref = b_ref.grad

    # CUDA forward
    h_cuda, output_cuda, v_cuda = hasty.e48_no_projections_forward(True, x.detach(), h0, W.detach(), b.detach())

    # CUDA backward
    dx_cuda, dW_cuda, db_cuda = hasty.e48_no_projections_backward(
        W.detach(), x.detach(), h_cuda, v_cuda, d_output)

    # Compare gradients using allclose
    dx_match = torch.allclose(dx_cuda, dx_ref, atol=atol, rtol=rtol)
    dW_match = torch.allclose(dW_cuda, dW_ref, atol=atol, rtol=rtol)
    db_match = torch.allclose(db_cuda, db_ref, atol=atol, rtol=rtol)

    dx_err = (dx_cuda - dx_ref).abs().max().item()
    dW_err = (dW_cuda - dW_ref).abs().max().item()
    db_err = (db_cuda - db_ref).abs().max().item()

    print(f"dx max error: {dx_err:.6f}")
    print(f"dW max error: {dW_err:.6f}")
    print(f"db max error: {db_err:.6f}")

    if dx_match and dW_match and db_match:
        print("[PASS] E48 backward matches Python reference")
        return True
    else:
        print("[FAIL] E48 backward mismatch!")
        return False

def test_e51_forward():
    """Test E51 No Self-Gate forward pass."""
    print("\n" + "="*60)
    print("Testing E51 (No Self-Gate) Forward")
    print("="*60)

    torch.manual_seed(42)
    B, T, D = 4, 32, 128

    # Create inputs
    x = torch.randn(T, B, D, device=device, dtype=dtype)
    h0 = torch.zeros(B, D, device=device, dtype=dtype)
    W = torch.randn(D, D, device=device, dtype=dtype) * 0.1
    b = torch.zeros(D, device=device, dtype=dtype)

    # CUDA forward
    h_cuda, output_cuda, v_cuda = hasty.e51_no_self_gate_forward(True, x, h0, W, b)

    # Python reference forward (from E51NoSelfGateCell)
    h_list = [h0]
    output_list = []

    # Batched GEMM for W @ x
    x_flat = x.reshape(T * B, D)
    Wx_all = (x_flat @ W.T).reshape(T, B, D)

    for t in range(T):
        h_prev = h_list[-1]
        Wx_t = Wx_all[t]
        Wh = h_prev @ W.T
        h_new = Wx_t + Wh + b
        h_list.append(h_new)

        # NO self-gate! Linear output
        output = h_new
        output_list.append(output)

    h_ref = torch.stack(h_list, dim=0)
    output_ref = torch.stack(output_list, dim=0)

    # Compare using allclose
    h_match = torch.allclose(h_cuda, h_ref, atol=atol, rtol=rtol)
    out_match = torch.allclose(output_cuda, output_ref, atol=atol, rtol=rtol)

    h_err = (h_cuda - h_ref).abs().max().item()
    out_err = (output_cuda - output_ref).abs().max().item()

    print(f"h max error: {h_err:.6f}")
    print(f"output max error: {out_err:.6f}")

    if h_match and out_match:
        print("[PASS] E51 forward matches Python reference")
        return True
    else:
        print("[FAIL] E51 forward mismatch!")
        return False

def test_e51_backward():
    """Test E51 No Self-Gate backward pass."""
    print("\n" + "="*60)
    print("Testing E51 (No Self-Gate) Backward")
    print("="*60)

    torch.manual_seed(42)
    B, T, D = 4, 16, 64  # Smaller for gradient checking

    # Create inputs
    x = torch.randn(T, B, D, device=device, dtype=dtype, requires_grad=True)
    h0 = torch.zeros(B, D, device=device, dtype=dtype)
    W = torch.randn(D, D, device=device, dtype=dtype, requires_grad=True) * 0.1
    b = torch.zeros(D, device=device, dtype=dtype, requires_grad=True)

    # Python reference (for gradient comparison)
    x_ref = x.clone().detach().requires_grad_(True)
    W_ref = W.clone().detach().requires_grad_(True)
    b_ref = b.clone().detach().requires_grad_(True)

    # Python forward
    h_list = [h0]
    output_list = []
    x_flat = x_ref.reshape(T * B, D)
    Wx_all = (x_flat @ W_ref.T).reshape(T, B, D)

    for t in range(T):
        h_prev = h_list[-1]
        Wx_t = Wx_all[t]
        Wh = h_prev @ W_ref.T
        h_new = Wx_t + Wh + b_ref
        h_list.append(h_new)
        # NO self-gate - linear output
        output = h_new
        output_list.append(output)

    output_ref = torch.stack(output_list, dim=0)

    # Create random d_output
    d_output = torch.randn_like(output_ref)

    # Python backward
    output_ref.backward(d_output)
    dx_ref = x_ref.grad
    dW_ref = W_ref.grad
    db_ref = b_ref.grad

    # CUDA forward
    h_cuda, output_cuda, v_cuda = hasty.e51_no_self_gate_forward(True, x.detach(), h0, W.detach(), b.detach())

    # CUDA backward
    dx_cuda, dW_cuda, db_cuda = hasty.e51_no_self_gate_backward(
        W.detach(), x.detach(), h_cuda, v_cuda, d_output)

    # Compare gradients using allclose
    dx_match = torch.allclose(dx_cuda, dx_ref, atol=atol, rtol=rtol)
    dW_match = torch.allclose(dW_cuda, dW_ref, atol=atol, rtol=rtol)
    db_match = torch.allclose(db_cuda, db_ref, atol=atol, rtol=rtol)

    dx_err = (dx_cuda - dx_ref).abs().max().item()
    dW_err = (dW_cuda - dW_ref).abs().max().item()
    db_err = (db_cuda - db_ref).abs().max().item()

    print(f"dx max error: {dx_err:.6f}")
    print(f"dW max error: {dW_err:.6f}")
    print(f"db max error: {db_err:.6f}")

    if dx_match and dW_match and db_match:
        print("[PASS] E51 backward matches Python reference")
        return True
    else:
        print("[FAIL] E51 backward mismatch!")
        return False

def test_training_loop():
    """Quick training loop test to verify gradient flow."""
    print("\n" + "="*60)
    print("Testing Training Loop (Gradient Flow)")
    print("="*60)

    B, T, D = 4, 32, 128

    # E48 training test
    print("\nE48 training step:")
    x = torch.randn(T, B, D, device=device, dtype=dtype)
    h0 = torch.zeros(B, D, device=device, dtype=dtype)
    W = torch.randn(D, D, device=device, dtype=dtype, requires_grad=True) * 0.1
    b = torch.zeros(D, device=device, dtype=dtype, requires_grad=True)

    # Forward
    h, output, v = hasty.e48_no_projections_forward(True, x, h0, W, b)

    # Backward
    loss = output.sum()
    d_output = torch.ones_like(output)
    dx, dW, db = hasty.e48_no_projections_backward(W, x, h, v, d_output)

    print(f"  Forward output norm: {output.norm().item():.4f}")
    print(f"  dW norm: {dW.norm().item():.4f}")
    print(f"  db norm: {db.norm().item():.4f}")

    e48_pass = dW.norm().item() > 0 and db.norm().item() > 0
    if e48_pass:
        print("  [PASS] Gradients flow correctly")
    else:
        print("  [FAIL] Zero gradients!")

    # E51 training test
    print("\nE51 training step:")
    W = torch.randn(D, D, device=device, dtype=dtype, requires_grad=True) * 0.1
    b = torch.zeros(D, device=device, dtype=dtype, requires_grad=True)

    # Forward
    h, output, v = hasty.e51_no_self_gate_forward(True, x, h0, W, b)

    # Backward
    loss = output.sum()
    d_output = torch.ones_like(output)
    dx, dW, db = hasty.e51_no_self_gate_backward(W, x, h, v, d_output)

    print(f"  Forward output norm: {output.norm().item():.4f}")
    print(f"  dW norm: {dW.norm().item():.4f}")
    print(f"  db norm: {db.norm().item():.4f}")

    e51_pass = dW.norm().item() > 0 and db.norm().item() > 0
    if e51_pass:
        print("  [PASS] Gradients flow correctly")
    else:
        print("  [FAIL] Zero gradients!")

    return e48_pass and e51_pass

if __name__ == "__main__":
    print("="*60)
    print("E48/E51 CUDA Kernel Correctness Tests")
    print("="*60)

    results = []

    # Run all tests
    results.append(("E48 Forward", test_e48_forward()))
    results.append(("E48 Backward", test_e48_backward()))
    results.append(("E51 Forward", test_e51_forward()))
    results.append(("E51 Backward", test_e51_backward()))
    results.append(("Training Loop", test_training_loop()))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{name}: [{status}]")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nAll tests passed!")
        exit(0)
    else:
        print("\nSome tests failed!")
        exit(1)
