#!/usr/bin/env python3
"""
Test script for CUDA kernels E43-E55.
Verifies that all kernels compile, link, and execute without errors.
Tests both forward and backward passes with random data.
"""

import torch
import sys

# Import CUDA library
import hasty_pytorch_lib as hasty_elman

# Test parameters
BATCH_SIZE = 4
SEQ_LEN = 8
DIM = 64
DTYPE = torch.bfloat16
DEVICE = 'cuda'


def test_e43_scalar_decay():
    """E43: Scalar Decay Elman - h = sigmoid(log_lambda) * (x + h_prev) + b"""
    # hasty_elman imported at top level

    x = torch.randn(SEQ_LEN, BATCH_SIZE, DIM, device=DEVICE, dtype=DTYPE, requires_grad=True)
    h0 = torch.zeros(BATCH_SIZE, DIM, device=DEVICE, dtype=DTYPE)
    log_lambda = torch.zeros(1, device=DEVICE, dtype=DTYPE, requires_grad=True)  # sigmoid(0) = 0.5
    b = torch.zeros(DIM, device=DEVICE, dtype=DTYPE, requires_grad=True)

    # Forward
    h, output, v = hasty_elman.e43_scalar_decay_forward(True, x, h0, log_lambda, b)

    # Check shapes
    assert h.shape == (SEQ_LEN + 1, BATCH_SIZE, DIM), f"h shape mismatch: {h.shape}"
    assert output.shape == (SEQ_LEN, BATCH_SIZE, DIM), f"output shape mismatch: {output.shape}"
    assert v.shape == (SEQ_LEN, BATCH_SIZE, DIM), f"v shape mismatch: {v.shape}"

    # Backward
    d_output = torch.randn_like(output)
    dx, d_log_lambda, db = hasty_elman.e43_scalar_decay_backward(log_lambda, x, h, v, d_output)

    assert dx.shape == x.shape, f"dx shape mismatch: {dx.shape}"
    assert d_log_lambda.shape == (1,), f"d_log_lambda shape mismatch: {d_log_lambda.shape}"
    assert db.shape == (DIM,), f"db shape mismatch: {db.shape}"

    # Check for NaN/Inf
    assert not torch.isnan(output).any(), "NaN in output"
    assert not torch.isinf(output).any(), "Inf in output"
    assert not torch.isnan(dx).any(), "NaN in dx"

    return True


def test_e44_diagonal_w():
    """E44: Diagonal W Elman - h = sigmoid(log_d) * (x + h_prev) + b (per-dim decay)"""
    # hasty_elman imported at top level

    x = torch.randn(SEQ_LEN, BATCH_SIZE, DIM, device=DEVICE, dtype=DTYPE, requires_grad=True)
    h0 = torch.zeros(BATCH_SIZE, DIM, device=DEVICE, dtype=DTYPE)
    log_d = torch.zeros(DIM, device=DEVICE, dtype=DTYPE, requires_grad=True)  # sigmoid(0) = 0.5 per dim
    b = torch.zeros(DIM, device=DEVICE, dtype=DTYPE, requires_grad=True)

    # Forward
    h, output, v = hasty_elman.e44_diagonal_w_forward(True, x, h0, log_d, b)

    # Check shapes
    assert h.shape == (SEQ_LEN + 1, BATCH_SIZE, DIM), f"h shape mismatch: {h.shape}"
    assert output.shape == (SEQ_LEN, BATCH_SIZE, DIM), f"output shape mismatch: {output.shape}"
    assert v.shape == (SEQ_LEN, BATCH_SIZE, DIM), f"v shape mismatch: {v.shape}"

    # Backward
    d_output = torch.randn_like(output)
    dx, d_log_d, db = hasty_elman.e44_diagonal_w_backward(log_d, x, h, v, d_output)

    assert dx.shape == x.shape, f"dx shape mismatch: {dx.shape}"
    assert d_log_d.shape == (DIM,), f"d_log_d shape mismatch: {d_log_d.shape}"
    assert db.shape == (DIM,), f"db shape mismatch: {db.shape}"

    # Check for NaN/Inf
    assert not torch.isnan(output).any(), "NaN in output"
    assert not torch.isinf(output).any(), "Inf in output"
    assert not torch.isnan(dx).any(), "NaN in dx"

    return True


def test_e45_pure_accumulation():
    """E45: Pure Accumulation - h = x + h_prev (NO GEMM, simplest possible)"""
    # hasty_elman imported at top level

    x = torch.randn(SEQ_LEN, BATCH_SIZE, DIM, device=DEVICE, dtype=DTYPE, requires_grad=True)
    h0 = torch.zeros(BATCH_SIZE, DIM, device=DEVICE, dtype=DTYPE)

    # Forward
    h, output = hasty_elman.e45_pure_accumulation_forward(True, x, h0)

    # Check shapes
    assert h.shape == (SEQ_LEN + 1, BATCH_SIZE, DIM), f"h shape mismatch: {h.shape}"
    assert output.shape == (SEQ_LEN, BATCH_SIZE, DIM), f"output shape mismatch: {output.shape}"

    # Backward
    d_output = torch.randn_like(output)
    dx, = hasty_elman.e45_pure_accumulation_backward(h, d_output)

    assert dx.shape == x.shape, f"dx shape mismatch: {dx.shape}"

    # Check for NaN/Inf
    assert not torch.isnan(output).any(), "NaN in output"
    assert not torch.isinf(output).any(), "Inf in output"
    assert not torch.isnan(dx).any(), "NaN in dx"

    return True


def test_e46_no_in_proj():
    """E46: No In-Projection - h = W @ (x + h_prev) + b (operates directly on embeddings)"""
    # hasty_elman imported at top level

    x = torch.randn(SEQ_LEN, BATCH_SIZE, DIM, device=DEVICE, dtype=DTYPE, requires_grad=True)
    h0 = torch.zeros(BATCH_SIZE, DIM, device=DEVICE, dtype=DTYPE)
    W = torch.randn(DIM, DIM, device=DEVICE, dtype=DTYPE, requires_grad=True) * 0.1
    b = torch.zeros(DIM, device=DEVICE, dtype=DTYPE, requires_grad=True)

    # Forward
    h, output, v = hasty_elman.e46_no_in_proj_forward(True, x, h0, W, b)

    # Check shapes
    assert h.shape == (SEQ_LEN + 1, BATCH_SIZE, DIM), f"h shape mismatch: {h.shape}"
    assert output.shape == (SEQ_LEN, BATCH_SIZE, DIM), f"output shape mismatch: {output.shape}"
    assert v.shape == (SEQ_LEN, BATCH_SIZE, DIM), f"v shape mismatch: {v.shape}"

    # Backward
    d_output = torch.randn_like(output)
    dx, dW, db = hasty_elman.e46_no_in_proj_backward(W, x, h, v, d_output)

    assert dx.shape == x.shape, f"dx shape mismatch: {dx.shape}"
    assert dW.shape == (DIM, DIM), f"dW shape mismatch: {dW.shape}"
    assert db.shape == (DIM,), f"db shape mismatch: {db.shape}"

    # Check for NaN/Inf
    assert not torch.isnan(output).any(), "NaN in output"
    assert not torch.isinf(output).any(), "Inf in output"
    assert not torch.isnan(dx).any(), "NaN in dx"

    return True


def test_e48_no_projections():
    """E48: No Projections - h = W @ (x + h_prev) + b, output = h * silu(h)"""
    # hasty_elman imported at top level

    x = torch.randn(SEQ_LEN, BATCH_SIZE, DIM, device=DEVICE, dtype=DTYPE, requires_grad=True)
    h0 = torch.zeros(BATCH_SIZE, DIM, device=DEVICE, dtype=DTYPE)
    W = torch.randn(DIM, DIM, device=DEVICE, dtype=DTYPE, requires_grad=True) * 0.1
    b = torch.zeros(DIM, device=DEVICE, dtype=DTYPE, requires_grad=True)

    # Forward
    h, output, v = hasty_elman.e48_no_projections_forward(True, x, h0, W, b)

    # Check shapes
    assert h.shape == (SEQ_LEN + 1, BATCH_SIZE, DIM), f"h shape mismatch: {h.shape}"
    assert output.shape == (SEQ_LEN, BATCH_SIZE, DIM), f"output shape mismatch: {output.shape}"
    assert v.shape == (SEQ_LEN, BATCH_SIZE, DIM), f"v shape mismatch: {v.shape}"

    # Backward
    d_output = torch.randn_like(output)
    dx, dW, db = hasty_elman.e48_no_projections_backward(W, x, h, v, d_output)

    assert dx.shape == x.shape, f"dx shape mismatch: {dx.shape}"
    assert dW.shape == (DIM, DIM), f"dW shape mismatch: {dW.shape}"
    assert db.shape == (DIM,), f"db shape mismatch: {db.shape}"

    # Check for NaN/Inf
    assert not torch.isnan(output).any(), "NaN in output"
    assert not torch.isinf(output).any(), "Inf in output"
    assert not torch.isnan(dx).any(), "NaN in dx"

    return True


def test_e51_no_self_gate():
    """E51: No Self-Gate - h = W @ (x + h_prev) + b, output = h (linear!)"""
    # hasty_elman imported at top level

    x = torch.randn(SEQ_LEN, BATCH_SIZE, DIM, device=DEVICE, dtype=DTYPE, requires_grad=True)
    h0 = torch.zeros(BATCH_SIZE, DIM, device=DEVICE, dtype=DTYPE)
    W = torch.randn(DIM, DIM, device=DEVICE, dtype=DTYPE, requires_grad=True) * 0.1
    b = torch.zeros(DIM, device=DEVICE, dtype=DTYPE, requires_grad=True)

    # Forward
    h, output, v = hasty_elman.e51_no_self_gate_forward(True, x, h0, W, b)

    # Check shapes
    assert h.shape == (SEQ_LEN + 1, BATCH_SIZE, DIM), f"h shape mismatch: {h.shape}"
    assert output.shape == (SEQ_LEN, BATCH_SIZE, DIM), f"output shape mismatch: {output.shape}"
    assert v.shape == (SEQ_LEN, BATCH_SIZE, DIM), f"v shape mismatch: {v.shape}"

    # Backward
    d_output = torch.randn_like(output)
    dx, dW, db = hasty_elman.e51_no_self_gate_backward(W, x, h, v, d_output)

    assert dx.shape == x.shape, f"dx shape mismatch: {dx.shape}"
    assert dW.shape == (DIM, DIM), f"dW shape mismatch: {dW.shape}"
    assert db.shape == (DIM,), f"db shape mismatch: {db.shape}"

    # Check for NaN/Inf
    assert not torch.isnan(output).any(), "NaN in output"
    assert not torch.isinf(output).any(), "Inf in output"
    assert not torch.isnan(dx).any(), "NaN in dx"

    return True


def test_e52_quadratic_gate():
    """E52: Quadratic Gate - h = W @ (x + h_prev) + b, output = h^2"""
    # hasty_elman imported at top level

    x = torch.randn(SEQ_LEN, BATCH_SIZE, DIM, device=DEVICE, dtype=DTYPE, requires_grad=True)
    h0 = torch.zeros(BATCH_SIZE, DIM, device=DEVICE, dtype=DTYPE)
    W = torch.randn(DIM, DIM, device=DEVICE, dtype=DTYPE, requires_grad=True) * 0.1
    b = torch.zeros(DIM, device=DEVICE, dtype=DTYPE, requires_grad=True)

    # Forward (unsigned quadratic: h^2)
    h, output, v = hasty_elman.e52_quadratic_gate_forward(True, False, x, h0, W, b)

    # Check shapes
    assert h.shape == (SEQ_LEN + 1, BATCH_SIZE, DIM), f"h shape mismatch: {h.shape}"
    assert output.shape == (SEQ_LEN, BATCH_SIZE, DIM), f"output shape mismatch: {output.shape}"
    assert v.shape == (SEQ_LEN, BATCH_SIZE, DIM), f"v shape mismatch: {v.shape}"

    # Backward
    d_output = torch.randn_like(output)
    dx, dW, db = hasty_elman.e52_quadratic_gate_backward(False, W, x, h, v, d_output)

    assert dx.shape == x.shape, f"dx shape mismatch: {dx.shape}"
    assert dW.shape == (DIM, DIM), f"dW shape mismatch: {dW.shape}"
    assert db.shape == (DIM,), f"db shape mismatch: {db.shape}"

    # Test signed quadratic variant (h * |h|)
    h2, output2, v2 = hasty_elman.e52_quadratic_gate_forward(True, True, x, h0, W, b)
    dx2, dW2, db2 = hasty_elman.e52_quadratic_gate_backward(True, W, x, h2, v2, d_output)

    # Check for NaN/Inf
    assert not torch.isnan(output).any(), "NaN in output"
    assert not torch.isinf(output).any(), "Inf in output"
    assert not torch.isnan(dx).any(), "NaN in dx"

    return True


def test_e53_sigmoid_gate():
    """E53: Sigmoid Gate Only - h = W @ (x + h_prev) + b, output = silu(h)"""
    # hasty_elman imported at top level

    x = torch.randn(SEQ_LEN, BATCH_SIZE, DIM, device=DEVICE, dtype=DTYPE, requires_grad=True)
    h0 = torch.zeros(BATCH_SIZE, DIM, device=DEVICE, dtype=DTYPE)
    W = torch.randn(DIM, DIM, device=DEVICE, dtype=DTYPE, requires_grad=True) * 0.1
    b = torch.zeros(DIM, device=DEVICE, dtype=DTYPE, requires_grad=True)

    # Forward
    h, output, v = hasty_elman.e53_sigmoid_gate_forward(True, x, h0, W, b)

    # Check shapes
    assert h.shape == (SEQ_LEN + 1, BATCH_SIZE, DIM), f"h shape mismatch: {h.shape}"
    assert output.shape == (SEQ_LEN, BATCH_SIZE, DIM), f"output shape mismatch: {output.shape}"
    assert v.shape == (SEQ_LEN, BATCH_SIZE, DIM), f"v shape mismatch: {v.shape}"

    # Backward
    d_output = torch.randn_like(output)
    dx, dW, db = hasty_elman.e53_sigmoid_gate_backward(W, x, h, v, d_output)

    assert dx.shape == x.shape, f"dx shape mismatch: {dx.shape}"
    assert dW.shape == (DIM, DIM), f"dW shape mismatch: {dW.shape}"
    assert db.shape == (DIM,), f"db shape mismatch: {db.shape}"

    # Check for NaN/Inf
    assert not torch.isnan(output).any(), "NaN in output"
    assert not torch.isinf(output).any(), "Inf in output"
    assert not torch.isnan(dx).any(), "NaN in dx"

    return True


def test_e54_diagonal_no_proj():
    """E54: Diagonal No-Proj - h = d * (x + h_prev) + b (per-dim decay, NO GEMM)"""
    # hasty_elman imported at top level

    x = torch.randn(SEQ_LEN, BATCH_SIZE, DIM, device=DEVICE, dtype=DTYPE, requires_grad=True)
    h0 = torch.zeros(BATCH_SIZE, DIM, device=DEVICE, dtype=DTYPE)
    d = torch.zeros(DIM, device=DEVICE, dtype=DTYPE, requires_grad=True)  # sigmoid(0) = 0.5
    b = torch.zeros(DIM, device=DEVICE, dtype=DTYPE, requires_grad=True)

    # Forward
    h, output, v = hasty_elman.e54_diagonal_no_proj_forward(True, x, h0, d, b)

    # Check shapes
    assert h.shape == (SEQ_LEN + 1, BATCH_SIZE, DIM), f"h shape mismatch: {h.shape}"
    assert output.shape == (SEQ_LEN, BATCH_SIZE, DIM), f"output shape mismatch: {output.shape}"
    assert v.shape == (SEQ_LEN, BATCH_SIZE, DIM), f"v shape mismatch: {v.shape}"

    # Backward
    d_output = torch.randn_like(output)
    dx, dd, db = hasty_elman.e54_diagonal_no_proj_backward(d, x, h, v, d_output)

    assert dx.shape == x.shape, f"dx shape mismatch: {dx.shape}"
    assert dd.shape == (DIM,), f"dd shape mismatch: {dd.shape}"
    assert db.shape == (DIM,), f"db shape mismatch: {db.shape}"

    # Check for NaN/Inf
    assert not torch.isnan(output).any(), "NaN in output"
    assert not torch.isinf(output).any(), "Inf in output"
    assert not torch.isnan(dx).any(), "NaN in dx"

    return True


def test_e55_scalar_no_proj():
    """E55: Scalar No-Proj - h = lambda * (x + h_prev) + b (single scalar, NO GEMM)"""
    # hasty_elman imported at top level

    x = torch.randn(SEQ_LEN, BATCH_SIZE, DIM, device=DEVICE, dtype=DTYPE, requires_grad=True)
    h0 = torch.zeros(BATCH_SIZE, DIM, device=DEVICE, dtype=DTYPE)
    lambda_val = 0.5  # decay factor
    b = torch.zeros(DIM, device=DEVICE, dtype=DTYPE, requires_grad=True)

    # Forward
    h, output, v = hasty_elman.e55_scalar_no_proj_forward(True, x, h0, lambda_val, b)

    # Check shapes
    assert h.shape == (SEQ_LEN + 1, BATCH_SIZE, DIM), f"h shape mismatch: {h.shape}"
    assert output.shape == (SEQ_LEN, BATCH_SIZE, DIM), f"output shape mismatch: {output.shape}"
    assert v.shape == (SEQ_LEN, BATCH_SIZE, DIM), f"v shape mismatch: {v.shape}"

    # Backward
    d_output = torch.randn_like(output)
    dx, dlambda, db = hasty_elman.e55_scalar_no_proj_backward(lambda_val, x, h, v, d_output)

    assert dx.shape == x.shape, f"dx shape mismatch: {dx.shape}"
    assert dlambda.shape == (1,), f"dlambda shape mismatch: {dlambda.shape}"
    assert db.shape == (DIM,), f"db shape mismatch: {db.shape}"

    # Check for NaN/Inf
    assert not torch.isnan(output).any(), "NaN in output"
    assert not torch.isinf(output).any(), "Inf in output"
    assert not torch.isnan(dx).any(), "NaN in dx"

    return True


def main():
    print("=" * 60)
    print("Testing CUDA kernels E43-E55")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Dtype: {DTYPE}")
    print(f"Batch size: {BATCH_SIZE}, Seq len: {SEQ_LEN}, Dim: {DIM}")
    print()

    tests = [
        ("E43: Scalar Decay", test_e43_scalar_decay),
        ("E44: Diagonal W", test_e44_diagonal_w),
        ("E45: Pure Accumulation", test_e45_pure_accumulation),
        ("E46: No In-Proj", test_e46_no_in_proj),
        ("E48: No Projections", test_e48_no_projections),
        ("E51: No Self-Gate", test_e51_no_self_gate),
        ("E52: Quadratic Gate", test_e52_quadratic_gate),
        ("E53: Sigmoid Gate", test_e53_sigmoid_gate),
        ("E54: Diagonal No-Proj", test_e54_diagonal_no_proj),
        ("E55: Scalar No-Proj", test_e55_scalar_no_proj),
    ]

    results = []
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"[PASS] {name}")
            results.append((name, True, None))
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            results.append((name, False, str(e)))

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(1 for _, success, _ in results if success)
    failed = len(results) - passed

    print(f"Passed: {passed}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")

    if failed > 0:
        print("\nFailed tests:")
        for name, success, error in results:
            if not success:
                print(f"  - {name}: {error}")
        sys.exit(1)
    else:
        print("\nAll tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
