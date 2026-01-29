#!/usr/bin/env python3
"""
E90 CUDA Kernel Verification

Compares CUDA forward and backward passes against Python reference implementation.
This is the standard verification to ensure CUDA matches Python exactly.
"""

import torch
import torch.nn.functional as F
import sys

# Import E90DualRate
from elman.models.e90_dual_rate import (
    E90DualRate, E90_CUDA_AVAILABLE, USE_E90_CUDA,
    E90DualRateCUDAFunction
)
import elman.models.e90_dual_rate as e90_module


def test_e90_forward_cuda_vs_python():
    """Compare CUDA forward pass with Python reference."""
    print("=" * 60)
    print("E90 CUDA vs Python Forward Pass Verification")
    print("=" * 60)

    if not E90_CUDA_AVAILABLE:
        print("SKIP: E90 CUDA kernel not available")
        return False

    # Set up test parameters
    B, T, dim = 2, 16, 256
    H = 8
    k_fast, v_fast = 16, 16
    k_slow, v_slow = 48, 48

    torch.manual_seed(42)
    device = torch.device('cuda')
    dtype = torch.bfloat16

    # Create model
    model = E90DualRate(
        dim=dim,
        n_heads=H,
        k_fast=k_fast,
        k_slow=k_slow,
        expansion=1.0,
        use_gate=False,  # Disable gate to simplify comparison
        linear_state=False,
    ).to(device=device, dtype=dtype)

    # Create input
    x = torch.randn(B, T, dim, device=device, dtype=dtype)

    # Run Python path (disable CUDA)
    e90_module.USE_E90_CUDA = False
    with torch.no_grad():
        output_py, (S_fast_py, S_slow_py) = model(x)

    # Run CUDA path
    e90_module.USE_E90_CUDA = True
    with torch.no_grad():
        output_cuda, (S_fast_cuda, S_slow_cuda) = model(x)

    # Compare outputs
    output_diff = (output_py - output_cuda).abs().max().item()
    S_fast_diff = (S_fast_py - S_fast_cuda).abs().max().item()
    S_slow_diff = (S_slow_py - S_slow_cuda).abs().max().item()

    print(f"Output max diff:       {output_diff:.6e}")
    print(f"S_fast max diff:       {S_fast_diff:.6e}")
    print(f"S_slow max diff:       {S_slow_diff:.6e}")

    # BF16 tolerance (larger than FP32)
    tolerance = 1e-2
    passed = output_diff < tolerance and S_fast_diff < tolerance and S_slow_diff < tolerance

    print(f"\nForward pass: {'PASSED' if passed else 'FAILED'} (tolerance: {tolerance})")
    return passed


def test_e90_backward_cuda_vs_python():
    """Compare CUDA backward pass with Python reference."""
    print("\n" + "=" * 60)
    print("E90 CUDA vs Python Backward Pass Verification")
    print("=" * 60)

    if not E90_CUDA_AVAILABLE:
        print("SKIP: E90 CUDA kernel not available")
        return False

    # Set up test parameters
    B, T, dim = 2, 8, 128  # Smaller for gradient computation
    H = 4
    k_fast, v_fast = 16, 16
    k_slow, v_slow = 32, 32

    torch.manual_seed(42)
    device = torch.device('cuda')
    dtype = torch.bfloat16

    # Create model
    model = E90DualRate(
        dim=dim,
        n_heads=H,
        k_fast=k_fast,
        k_slow=k_slow,
        expansion=1.0,
        use_gate=False,
        linear_state=False,
    ).to(device=device, dtype=dtype)

    # Create input
    x = torch.randn(B, T, dim, device=device, dtype=dtype, requires_grad=True)

    # Run Python path
    e90_module.USE_E90_CUDA = False
    output_py, _ = model(x)
    loss_py = output_py.sum()
    loss_py.backward()
    grad_py = x.grad.clone()
    x.grad.zero_()

    # Run CUDA path
    e90_module.USE_E90_CUDA = True
    output_cuda, _ = model(x)
    loss_cuda = output_cuda.sum()
    loss_cuda.backward()
    grad_cuda = x.grad.clone()

    # Compare gradients
    grad_diff = (grad_py - grad_cuda).abs().max().item()
    grad_rel_diff = grad_diff / (grad_py.abs().max().item() + 1e-8)

    print(f"x.grad max diff:       {grad_diff:.6e}")
    print(f"x.grad rel diff:       {grad_rel_diff:.6e}")

    # BF16 gradient tolerance
    tolerance = 5e-2
    passed = grad_rel_diff < tolerance

    print(f"\nBackward pass: {'PASSED' if passed else 'FAILED'} (rel tolerance: {tolerance})")
    return passed


def test_e90_standalone_kernel():
    """Test the E90 CUDA kernel directly (without the full model)."""
    print("\n" + "=" * 60)
    print("E90 Standalone CUDA Kernel Test")
    print("=" * 60)

    if not E90_CUDA_AVAILABLE:
        print("SKIP: E90 CUDA kernel not available")
        return False

    import hasty_pytorch_lib

    # Set up test parameters
    T, B, H = 8, 2, 4
    k_fast, v_fast = 16, 16
    k_slow, v_slow = 32, 32
    out_v_dim = max(v_fast, v_slow)

    torch.manual_seed(42)
    device = torch.device('cuda')
    dtype = torch.bfloat16

    # Create inputs (already in [T, B, H, dim] format for CUDA)
    k_fast_t = torch.randn(T, B, H, k_fast, device=device, dtype=dtype)
    v_fast_t = torch.randn(T, B, H, v_fast, device=device, dtype=dtype)
    q_fast_t = torch.randn(T, B, H, k_fast, device=device, dtype=dtype)
    decay_fast_t = torch.sigmoid(torch.randn(T, B, H, device=device, dtype=dtype))

    k_slow_t = torch.randn(T, B, H, k_slow, device=device, dtype=dtype)
    v_slow_t = torch.randn(T, B, H, v_slow, device=device, dtype=dtype)
    q_slow_t = torch.randn(T, B, H, k_slow, device=device, dtype=dtype)
    decay_slow_t = torch.sigmoid(torch.randn(T, B, H, device=device, dtype=dtype))

    slow_gate_t = torch.sigmoid(torch.randn(T, B, H, device=device, dtype=dtype))
    mix_weights = torch.softmax(torch.randn(T, B, H, 2, device=device, dtype=dtype), dim=-1)
    mix_fast_t = mix_weights[..., 0].contiguous()
    mix_slow_t = mix_weights[..., 1].contiguous()

    S_fast0 = torch.zeros(B, H, k_fast, v_fast, device=device, dtype=dtype)
    S_slow0 = torch.zeros(B, H, k_slow, v_slow, device=device, dtype=dtype)

    # Call CUDA kernel directly
    try:
        results = hasty_pytorch_lib.e90_dual_rate_forward(
            k_fast_t, v_fast_t, q_fast_t, decay_fast_t,
            k_slow_t, v_slow_t, q_slow_t, decay_slow_t, slow_gate_t,
            mix_fast_t, mix_slow_t,
            S_fast0, S_slow0, H
        )
        S_fast_final, S_slow_final, output = results[0], results[1], results[2]
        print(f"Output shape: {output.shape} (expected: [{T}, {B}, {H}, {out_v_dim}])")
        print(f"S_fast shape: {S_fast_final.shape} (expected: [{B}, {H}, {k_fast}, {v_fast}])")
        print(f"S_slow shape: {S_slow_final.shape} (expected: [{B}, {H}, {k_slow}, {v_slow}])")

        # Check for NaN
        has_nan = torch.isnan(output).any() or torch.isnan(S_fast_final).any() or torch.isnan(S_slow_final).any()
        print(f"Has NaN: {has_nan}")

        passed = not has_nan and output.shape == (T, B, H, out_v_dim)
        print(f"\nStandalone kernel: {'PASSED' if passed else 'FAILED'}")
        return passed
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_e90_gradient_flow():
    """Test that gradients flow properly through the E90 model."""
    print("\n" + "=" * 60)
    print("E90 Gradient Flow Test")
    print("=" * 60)

    B, T, dim = 2, 16, 256
    H = 8
    k_fast, k_slow = 16, 48

    torch.manual_seed(42)
    device = torch.device('cuda')
    dtype = torch.bfloat16

    model = E90DualRate(
        dim=dim,
        n_heads=H,
        k_fast=k_fast,
        k_slow=k_slow,
        expansion=1.0,
        use_gate=True,
        gate_activation='silu',
    ).to(device=device, dtype=dtype)

    x = torch.randn(B, T, dim, device=device, dtype=dtype, requires_grad=True)

    # Forward
    e90_module.USE_E90_CUDA = E90_CUDA_AVAILABLE
    output, _ = model(x)
    loss = output.sum()

    # Backward
    loss.backward()

    # Check gradients
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()

    print("Gradient norms for model parameters:")
    for name, norm in sorted(grad_norms.items()):
        print(f"  {name}: {norm:.6f}")

    # Check x gradient
    x_grad_norm = x.grad.norm().item() if x.grad is not None else 0
    print(f"\nx.grad norm: {x_grad_norm:.6f}")

    # Pass if all gradients are finite and non-zero
    all_finite = all(not (torch.isnan(torch.tensor(n)) or torch.isinf(torch.tensor(n))) for n in grad_norms.values())
    has_flow = sum(1 for n in grad_norms.values() if n > 0) > len(grad_norms) * 0.5
    x_has_grad = x_grad_norm > 0 and not torch.isnan(torch.tensor(x_grad_norm))

    passed = all_finite and has_flow and x_has_grad
    print(f"\nGradient flow: {'PASSED' if passed else 'FAILED'}")
    return passed


def main():
    print("E90 Dual-Rate CUDA Kernel Verification")
    print("=" * 60)
    print(f"E90_CUDA_AVAILABLE: {E90_CUDA_AVAILABLE}")
    print(f"USE_E90_CUDA: {USE_E90_CUDA}")
    print()

    results = []

    # Test standalone kernel first
    results.append(("Standalone Kernel", test_e90_standalone_kernel()))

    # Test forward pass comparison
    results.append(("Forward Pass", test_e90_forward_cuda_vs_python()))

    # Test backward pass comparison
    results.append(("Backward Pass", test_e90_backward_cuda_vs_python()))

    # Test gradient flow
    results.append(("Gradient Flow", test_e90_gradient_flow()))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    all_passed = all(passed for _, passed in results)
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
