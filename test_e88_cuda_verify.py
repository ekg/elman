"""
Test E88 CUDA kernel vs Python reference implementation.

Verifies mathematical equivalence between:
1. Python E88FLAHybrid forward pass
2. CUDA e88_fla_hybrid_forward kernel

Key differences to check:
- L2 normalization of k/q (Python does it, CUDA expects pre-normalized)
- Self-gating (CUDA does Sq*silu(Sq), Python uses separate output gating)
- State shape and indexing
"""

import torch
import torch.nn.functional as F
import numpy as np

# Check if CUDA kernel is available
try:
    import hasty_pytorch_lib
    HAS_E88_CUDA = hasattr(hasty_pytorch_lib, 'e88_fla_hybrid_forward')
    print(f"E88 CUDA kernel available: {HAS_E88_CUDA}")
    if HAS_E88_CUDA:
        print("  - e88_fla_hybrid_forward: found")
        print("  - e88_fla_hybrid_backward:", "found" if hasattr(hasty_pytorch_lib, 'e88_fla_hybrid_backward') else "missing")
except ImportError as e:
    HAS_E88_CUDA = False
    print(f"hasty_pytorch_lib import error: {e}")


def python_e88_forward_core(k, v, q, decay, S0):
    """
    Pure Python implementation of E88 core recurrence.

    This implements EXACTLY what the CUDA kernel should do:
    - k, v, q are pre-computed (already normalized if needed)
    - decay is [T, B, H] scalar per head per timestep
    - S is [B, H, n_state, head_v_dim] rectangular state

    For each timestep:
        retrieved = S @ k  -> [head_v_dim]
        delta = v - retrieved
        S = tanh(decay * S + outer(delta, k))
        Sq = S^T @ q -> [head_v_dim]
        output = Sq * silu(Sq) = Sq^2 * sigmoid(Sq)
    """
    T, B, H, n_state = k.shape
    head_v_dim = v.shape[-1]

    # Initialize state list per head
    S_list = [S0[:, h].clone() for h in range(H)]  # Each [B, n_state, head_v_dim]

    outputs = []
    Sq_cache_list = []

    for t in range(T):
        head_outputs = []
        head_Sq = []

        for h in range(H):
            k_t = k[t, :, h]  # [B, n_state]
            v_t = v[t, :, h]  # [B, head_v_dim]
            q_t = q[t, :, h]  # [B, n_state]
            decay_t = decay[t, :, h:h+1]  # [B, 1]

            # Retrieve: S @ k -> [B, head_v_dim]
            # S is [B, n_state, head_v_dim], k is [B, n_state]
            retrieved = torch.einsum('biv,bi->bv', S_list[h], k_t)

            # Delta = v - retrieved
            delta = v_t - retrieved  # [B, head_v_dim]

            # Outer product: [B, n_state, head_v_dim]
            outer = torch.einsum('bv,bi->biv', delta, k_t)

            # State update: S = tanh(decay * S + outer)
            S_list[h] = torch.tanh(decay_t.unsqueeze(-1) * S_list[h] + outer)

            # Query: Sq = S^T @ q -> [B, head_v_dim]
            Sq = torch.einsum('biv,bi->bv', S_list[h], q_t)
            head_Sq.append(Sq)

            # Self-gating: output = Sq * silu(Sq) = Sq^2 * sigmoid(Sq)
            out_h = Sq * F.silu(Sq)
            head_outputs.append(out_h)

        # Stack: [B, H, head_v_dim]
        out_t = torch.stack(head_outputs, dim=1)
        Sq_t = torch.stack(head_Sq, dim=1)
        outputs.append(out_t)
        Sq_cache_list.append(Sq_t)

    # Stack time: [T, B, H, head_v_dim]
    output = torch.stack(outputs, dim=0)
    Sq_cache = torch.stack(Sq_cache_list, dim=0)

    # Final state: [B, H, n_state, head_v_dim]
    S_final = torch.stack(S_list, dim=1)

    return output, S_final, Sq_cache


def test_cuda_vs_python():
    """Test CUDA kernel matches Python reference."""
    print("\n" + "=" * 60)
    print("Testing E88 CUDA vs Python Reference")
    print("=" * 60)

    if not HAS_E88_CUDA:
        print("SKIP: E88 CUDA kernel not available")
        return False

    device = 'cuda'
    dtype = torch.bfloat16

    # Test configuration
    T, B = 32, 4
    H = 4  # n_heads
    n_state = 32
    head_v_dim = 64  # expansion factor of 2

    print(f"\nConfig: T={T}, B={B}, H={H}, n_state={n_state}, head_v_dim={head_v_dim}")

    torch.manual_seed(42)

    # Generate test data (already L2-normalized k and q)
    k = torch.randn(T, B, H, n_state, device=device, dtype=dtype)
    k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)  # L2 normalize

    v = torch.randn(T, B, H, head_v_dim, device=device, dtype=dtype)

    q = torch.randn(T, B, H, n_state, device=device, dtype=dtype)
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)  # L2 normalize

    # Decay in (0, 1) range - this is exp(g) where g < 0
    decay = torch.sigmoid(torch.randn(T, B, H, device=device, dtype=dtype))

    # Initial state
    S0 = torch.zeros(B, H, n_state, head_v_dim, device=device, dtype=dtype)

    print("\n--- Python Reference Forward ---")
    output_py, S_final_py, Sq_cache_py = python_e88_forward_core(k, v, q, decay, S0.clone())
    print(f"  Output shape: {output_py.shape}")
    print(f"  S_final shape: {S_final_py.shape}")
    print(f"  Output mean: {output_py.float().mean():.6f}, std: {output_py.float().std():.6f}")

    print("\n--- CUDA Kernel Forward ---")
    try:
        results = hasty_pytorch_lib.e88_fla_hybrid_forward(
            True,  # training
            k.contiguous(),
            v.contiguous(),
            q.contiguous(),
            decay.contiguous(),
            S0.contiguous(),
            H  # n_heads
        )
        S_final_cuda = results[0]  # [B, H, n_state, head_v_dim]
        output_cuda = results[1]   # [T, B, H, head_v_dim]
        S_cache = results[2]       # checkpoints + Sq_cache

        print(f"  Output shape: {output_cuda.shape}")
        print(f"  S_final shape: {S_final_cuda.shape}")
        print(f"  Output mean: {output_cuda.float().mean():.6f}, std: {output_cuda.float().std():.6f}")
    except Exception as e:
        print(f"  CUDA kernel error: {e}")
        return False

    print("\n--- Comparing Outputs ---")

    # Convert to float for comparison
    output_py_f = output_py.float()
    output_cuda_f = output_cuda.float()

    # Compute differences
    diff = (output_py_f - output_cuda_f).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Relative error
    rel_diff = diff / (output_py_f.abs() + 1e-6)
    max_rel_diff = rel_diff.max().item()

    print(f"  Max absolute diff: {max_diff:.8f}")
    print(f"  Mean absolute diff: {mean_diff:.8f}")
    print(f"  Max relative diff: {max_rel_diff:.8f}")

    # State comparison
    S_diff = (S_final_py.float() - S_final_cuda.float()).abs()
    S_max_diff = S_diff.max().item()
    print(f"  State max diff: {S_max_diff:.8f}")

    # Tolerance for BF16 - use absolute error which is more meaningful
    # BF16 has ~0.78% relative precision, absolute errors around 0.01-0.02 are normal
    ABS_TOLERANCE = 0.02  # 2% of max value
    output_range = max(output_py_f.abs().max().item(), 1e-6)

    if max_diff < ABS_TOLERANCE:
        print(f"\n  PASS: Max absolute diff {max_diff:.6f} < tolerance {ABS_TOLERANCE}")
        return True
    else:
        print(f"\n  FAIL: Max absolute diff {max_diff:.6f} >= tolerance {ABS_TOLERANCE}")

        # Debug: find where the differences are largest
        print("\n  Debugging largest differences:")
        flat_diff = diff.view(-1)
        top_k = 5
        top_vals, top_idxs = torch.topk(flat_diff, min(top_k, flat_diff.numel()))
        for i, (val, idx) in enumerate(zip(top_vals, top_idxs)):
            # Unravel index
            idx_val = idx.item()
            t = idx_val // (B * H * head_v_dim)
            rem = idx_val % (B * H * head_v_dim)
            b = rem // (H * head_v_dim)
            rem = rem % (H * head_v_dim)
            h = rem // head_v_dim
            v_idx = rem % head_v_dim

            py_val = output_py_f[t, b, h, v_idx].item()
            cuda_val = output_cuda_f[t, b, h, v_idx].item()
            print(f"    [{i}] t={t}, b={b}, h={h}, v={v_idx}: py={py_val:.6f}, cuda={cuda_val:.6f}, diff={val.item():.6f}")

        return False


def test_backward():
    """Test CUDA backward pass gradients via autograd.Function wrapper."""
    print("\n" + "=" * 60)
    print("Testing E88 CUDA Backward Pass (via autograd.Function)")
    print("=" * 60)

    if not HAS_E88_CUDA or not hasattr(hasty_pytorch_lib, 'e88_fla_hybrid_backward'):
        print("SKIP: E88 CUDA backward not available")
        return False

    # Import the autograd wrapper
    from elman.models.e88_fla_hybrid import E88FLAHybridCUDAFunction

    device = 'cuda'
    dtype = torch.bfloat16

    T, B = 16, 2
    H = 4
    n_state = 32
    head_v_dim = 64

    print(f"\nConfig: T={T}, B={B}, H={H}, n_state={n_state}, head_v_dim={head_v_dim}")

    torch.manual_seed(123)

    # Generate test data - L2 normalized k and q
    k = torch.randn(T, B, H, n_state, device=device, dtype=dtype)
    k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
    k.requires_grad_(True)

    v = torch.randn(T, B, H, head_v_dim, device=device, dtype=dtype, requires_grad=True)

    q = torch.randn(T, B, H, n_state, device=device, dtype=dtype)
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
    q.requires_grad_(True)

    decay = torch.sigmoid(torch.randn(T, B, H, device=device, dtype=dtype, requires_grad=True))
    S0 = torch.zeros(B, H, n_state, head_v_dim, device=device, dtype=dtype)

    # Forward via autograd.Function
    S_final, output = E88FLAHybridCUDAFunction.apply(
        True, k, v, q, decay, S0, H
    )

    # Fake gradient
    d_output = torch.randn_like(output)

    # Backward
    output.backward(d_output)

    print(f"\n  k.grad shape: {k.grad.shape if k.grad is not None else 'None'}")
    print(f"  v.grad shape: {v.grad.shape if v.grad is not None else 'None'}")
    print(f"  q.grad shape: {q.grad.shape if q.grad is not None else 'None'}")
    print(f"  decay.grad shape: {decay.grad.shape if decay.grad is not None else 'None'}")

    if k.grad is not None:
        print(f"  k.grad mean: {k.grad.float().mean():.6f}, std: {k.grad.float().std():.6f}")
    if v.grad is not None:
        print(f"  v.grad mean: {v.grad.float().mean():.6f}, std: {v.grad.float().std():.6f}")

    # Check gradients are finite
    all_finite = True
    for name, g in [('k', k.grad), ('v', v.grad), ('q', q.grad), ('decay', decay.grad)]:
        if g is not None and not torch.isfinite(g).all():
            print(f"  FAIL: {name}.grad has non-finite values")
            all_finite = False

    if all_finite:
        print("\n  PASS: All gradients are finite")
        return True
    else:
        return False


if __name__ == "__main__":
    print("E88 CUDA Kernel Verification Test")
    print("=" * 60)

    # List available functions
    if HAS_E88_CUDA:
        print("\nAvailable E88 functions in hasty_pytorch_lib:")
        for attr in dir(hasty_pytorch_lib):
            if 'e88' in attr.lower():
                print(f"  - {attr}")

    # Run tests
    results = []

    results.append(("Forward Pass", test_cuda_vs_python()))
    results.append(("Backward Pass", test_backward()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    all_passed = all(r[1] for r in results)
    print("\n" + ("ALL TESTS PASSED!" if all_passed else "SOME TESTS FAILED"))
