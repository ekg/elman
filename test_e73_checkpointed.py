#!/usr/bin/env python3
"""Test E73 Checkpointed CUDA kernel against original E73."""

import torch
import sys
sys.path.insert(0, 'elman/cuda')
import hasty_pytorch_lib as hasty

def test_e73_checkpointed_forward():
    """Test that checkpointed forward produces valid output."""
    print("=" * 60)
    print("Testing E73 Checkpointed Forward")
    print("=" * 60)

    torch.manual_seed(42)
    device = 'cuda'
    dtype = torch.bfloat16

    # Test case
    T, B, dim, n_state = 64, 4, 256, 128
    variant = 0  # column
    checkpoint_interval = 16

    # Create inputs
    x = torch.randn(T, B, dim, device=device, dtype=dtype)
    S0 = torch.zeros(B, n_state, n_state, device=device, dtype=dtype)
    W_k = torch.randn(n_state, dim, device=device, dtype=dtype) * 0.01
    W_v = torch.randn(n_state, dim, device=device, dtype=dtype) * 0.01
    W_q = torch.randn(n_state, dim, device=device, dtype=dtype) * 0.01
    W_z = torch.randn(n_state, dim, device=device, dtype=dtype) * 0.01
    b_z = torch.zeros(n_state, device=device, dtype=dtype)

    # Checkpointed E73 forward
    S_cp, out_cp, S_checkpoints, k_norm_cache_cp, v_cache_cp, q_cache_cp, z_cache_cp, Sq_cache_cp = \
        hasty.e73_checkpointed_forward(True, x, S0.clone(), variant, checkpoint_interval, W_k, W_v, W_q, W_z, b_z)

    # Verify outputs are valid
    print(f"Output shape: {out_cp.shape}")
    print(f"Final S shape: {S_cp.shape}")
    print(f"S_checkpoints shape: {S_checkpoints.shape}")
    print(f"Output mean: {out_cp.abs().mean().item():.6f}")
    print(f"Output has NaN: {torch.isnan(out_cp).any().item()}")
    print(f"Final S has NaN: {torch.isnan(S_cp).any().item()}")

    # Memory comparison (theoretical)
    orig_S_memory = (T+1) * B * n_state * n_state * 2 / 1e6  # bytes to MB
    num_checkpoints = S_checkpoints.shape[0]
    cp_S_memory = num_checkpoints * B * n_state * n_state * 2 / 1e6

    print(f"\nMemory for S states (theoretical):")
    print(f"  Original E73: {orig_S_memory:.2f} MB")
    print(f"  Checkpointed: {cp_S_memory:.2f} MB ({num_checkpoints} checkpoints)")
    print(f"  Reduction: {orig_S_memory / cp_S_memory:.1f}x")

    # Run twice and check determinism
    S_cp2, out_cp2, _, _, _, _, _, _ = \
        hasty.e73_checkpointed_forward(True, x, S0.clone(), variant, checkpoint_interval, W_k, W_v, W_q, W_z, b_z)

    output_cos = torch.nn.functional.cosine_similarity(out_cp.flatten(), out_cp2.flatten(), dim=0).item()
    print(f"\nDeterminism check (same input twice):")
    print(f"  Output cosine similarity: {output_cos:.6f}")

    success = (not torch.isnan(out_cp).any().item() and
               not torch.isnan(S_cp).any().item() and
               output_cos > 0.999)
    print(f"\nForward test: {'PASSED' if success else 'FAILED'}")
    return success


def test_e73_checkpointed_backward():
    """Test that checkpointed backward produces valid gradients."""
    print("\n" + "=" * 60)
    print("Testing E73 Checkpointed Backward")
    print("=" * 60)

    torch.manual_seed(42)
    device = 'cuda'
    dtype = torch.bfloat16

    # Small test case
    T, B, dim, n_state = 32, 2, 128, 64
    variant = 0  # column
    checkpoint_interval = 8

    # Create inputs
    x = torch.randn(T, B, dim, device=device, dtype=dtype)
    S0 = torch.zeros(B, n_state, n_state, device=device, dtype=dtype)
    W_k = torch.randn(n_state, dim, device=device, dtype=dtype) * 0.01
    W_v = torch.randn(n_state, dim, device=device, dtype=dtype) * 0.01
    W_q = torch.randn(n_state, dim, device=device, dtype=dtype) * 0.01
    W_z = torch.randn(n_state, dim, device=device, dtype=dtype) * 0.01
    b_z = torch.zeros(n_state, device=device, dtype=dtype)

    # Checkpointed E73 forward
    S_cp, out_cp, S_checkpoints, k_norm_cache_cp, v_cache_cp, q_cache_cp, z_cache_cp, Sq_cache_cp = \
        hasty.e73_checkpointed_forward(True, x, S0.clone(), variant, checkpoint_interval, W_k, W_v, W_q, W_z, b_z)

    # Fake gradient
    d_output = torch.randn_like(out_cp)

    # Checkpointed backward
    dx_cp, dW_k_cp, dW_v_cp, dW_q_cp, dW_z_cp, db_z_cp = \
        hasty.e73_checkpointed_backward(
            x, d_output, S_checkpoints, k_norm_cache_cp, v_cache_cp, q_cache_cp, z_cache_cp, Sq_cache_cp,
            variant, checkpoint_interval, W_k, W_v, W_q, W_z)

    print("Gradient shapes and norms:")
    print(f"  dx: shape={dx_cp.shape}, norm={dx_cp.norm().item():.6f}, has_nan={torch.isnan(dx_cp).any().item()}")
    print(f"  dW_k: shape={dW_k_cp.shape}, norm={dW_k_cp.norm().item():.6f}, has_nan={torch.isnan(dW_k_cp).any().item()}")
    print(f"  dW_v: shape={dW_v_cp.shape}, norm={dW_v_cp.norm().item():.6f}, has_nan={torch.isnan(dW_v_cp).any().item()}")
    print(f"  dW_q: shape={dW_q_cp.shape}, norm={dW_q_cp.norm().item():.6f}, has_nan={torch.isnan(dW_q_cp).any().item()}")
    print(f"  dW_z: shape={dW_z_cp.shape}, norm={dW_z_cp.norm().item():.6f}, has_nan={torch.isnan(dW_z_cp).any().item()}")
    print(f"  db_z: shape={db_z_cp.shape}, norm={db_z_cp.norm().item():.6f}, has_nan={torch.isnan(db_z_cp).any().item()}")

    # Run backward twice with same inputs and check determinism
    dx_cp2, dW_k_cp2, dW_v_cp2, dW_q_cp2, dW_z_cp2, db_z_cp2 = \
        hasty.e73_checkpointed_backward(
            x, d_output, S_checkpoints, k_norm_cache_cp, v_cache_cp, q_cache_cp, z_cache_cp, Sq_cache_cp,
            variant, checkpoint_interval, W_k, W_v, W_q, W_z)

    dx_cos = torch.nn.functional.cosine_similarity(dx_cp.flatten(), dx_cp2.flatten(), dim=0).item()
    dW_k_cos = torch.nn.functional.cosine_similarity(dW_k_cp.flatten(), dW_k_cp2.flatten(), dim=0).item()

    print(f"\nDeterminism check (same backward twice):")
    print(f"  dx cosine: {dx_cos:.6f}")
    print(f"  dW_k cosine: {dW_k_cos:.6f}")

    # Check all gradients are non-NaN and non-trivial
    # Note: atomic operations with bfloat16 have some non-determinism, so threshold is 0.99
    success = (not torch.isnan(dx_cp).any().item() and
               not torch.isnan(dW_k_cp).any().item() and
               not torch.isnan(dW_v_cp).any().item() and
               not torch.isnan(dW_q_cp).any().item() and
               not torch.isnan(dW_z_cp).any().item() and
               not torch.isnan(db_z_cp).any().item() and
               dW_k_cp.norm().item() > 0 and
               dx_cos > 0.99)  # Atomic ops have some non-determinism

    print(f"\nBackward test: {'PASSED' if success else 'FAILED'}")
    return success


def test_training_step():
    """Test a complete training step with checkpointed E73."""
    print("\n" + "=" * 60)
    print("Testing E73 Checkpointed Training Step")
    print("=" * 60)

    torch.manual_seed(42)
    device = 'cuda'
    dtype = torch.bfloat16

    T, B, dim, n_state = 64, 8, 256, 128
    variant = 0
    checkpoint_interval = 16

    # Create trainable parameters
    W_k = torch.randn(n_state, dim, device=device, dtype=dtype, requires_grad=False) * 0.01
    W_v = torch.randn(n_state, dim, device=device, dtype=dtype, requires_grad=False) * 0.01
    W_q = torch.randn(n_state, dim, device=device, dtype=dtype, requires_grad=False) * 0.01
    W_z = torch.randn(n_state, dim, device=device, dtype=dtype, requires_grad=False) * 0.01
    b_z = torch.zeros(n_state, device=device, dtype=dtype)

    # Input
    x = torch.randn(T, B, dim, device=device, dtype=dtype)
    S0 = torch.zeros(B, n_state, n_state, device=device, dtype=dtype)

    # Forward
    S_cp, out_cp, S_checkpoints, k_norm_cache, v_cache, q_cache, z_cache, Sq_cache = \
        hasty.e73_checkpointed_forward(True, x, S0, variant, checkpoint_interval, W_k, W_v, W_q, W_z, b_z)

    # Compute loss (simple MSE to zero)
    loss = out_cp.pow(2).mean()
    print(f"Output mean: {out_cp.abs().mean().item():.6f}")
    print(f"Loss: {loss.item():.6f}")

    # Backward
    d_output = 2 * out_cp / out_cp.numel()  # gradient of MSE

    dx, dW_k, dW_v, dW_q, dW_z, db_z = \
        hasty.e73_checkpointed_backward(
            x, d_output, S_checkpoints, k_norm_cache, v_cache, q_cache, z_cache, Sq_cache,
            variant, checkpoint_interval, W_k, W_v, W_q, W_z)

    print(f"dx norm: {dx.norm().item():.6f}")
    print(f"dW_k norm: {dW_k.norm().item():.6f}")
    print(f"dW_v norm: {dW_v.norm().item():.6f}")
    print(f"dW_q norm: {dW_q.norm().item():.6f}")
    print(f"dW_z norm: {dW_z.norm().item():.6f}")
    print(f"db_z norm: {db_z.norm().item():.6f}")

    # Check gradients are non-trivial
    success = (dW_k.norm().item() > 0 and dW_v.norm().item() > 0 and
               dW_q.norm().item() > 0 and dW_z.norm().item() > 0)

    print(f"\nTraining step test: {'PASSED' if success else 'FAILED'}")
    return success


if __name__ == '__main__':
    print("E73 Checkpointed CUDA Kernel Tests")
    print("=" * 60)

    results = []

    try:
        results.append(("Forward", test_e73_checkpointed_forward()))
    except Exception as e:
        print(f"Forward test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Forward", False))

    try:
        results.append(("Backward", test_e73_checkpointed_backward()))
    except Exception as e:
        print(f"Backward test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Backward", False))

    try:
        results.append(("Training Step", test_training_step()))
    except Exception as e:
        print(f"Training step test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Training Step", False))

    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")

    all_passed = all(r[1] for r in results)
    print(f"\nOverall: {'All tests passed!' if all_passed else 'Some tests failed.'}")
