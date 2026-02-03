#!/usr/bin/env python3
"""
Test E74 Full Matrix no_z CUDA kernel gradient correctness.

Compares CUDA gradients against PyTorch autograd reference for the no_z
projection type (separate k, v, q) at n_state=64 (which uses the new
global memory backward kernel).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import CUDA kernel
try:
    import hasty_pytorch_lib as elman_ladder_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Error: CUDA kernels not available!")
    exit(1)


def pytorch_e74_full_matrix_forward_noz(x, S0, W_k, W_v, W_q, use_tanh=True):
    """
    PyTorch reference implementation of E74 full matrix forward (no_z).

    Args:
        x: [T, B, dim] input
        S0: [B, n, n] initial state
        W_k, W_v, W_q: [n, dim] projection weights
        use_tanh: whether to apply tanh nonlinearity

    Returns:
        output: [T, B, n]
        S_final: [B, n, n]
        Sq_cache: [T, B, n] for backward
    """
    T, B, dim = x.shape
    n = W_k.shape[0]

    # Compute projections: [T*B, n]
    x_flat = x.reshape(T * B, dim)
    k_all = x_flat @ W_k.T  # [T*B, n]
    v_all = x_flat @ W_v.T  # [T*B, n]
    q_all = x_flat @ W_q.T  # [T*B, n]

    k_all = k_all.reshape(T, B, n)
    v_all = v_all.reshape(T, B, n)
    q_all = q_all.reshape(T, B, n)

    S = S0.clone()
    outputs = []
    Sq_cache = []

    for t in range(T):
        k_t = k_all[t]  # [B, n]
        v_t = v_all[t]
        q_t = q_all[t]

        # Normalize k
        k_norm = k_t / (k_t.norm(dim=-1, keepdim=True) + 1e-6)

        # Retrieve: S @ k_norm
        retrieved = torch.einsum('bij,bj->bi', S, k_norm)

        # Delta: v - retrieved
        delta = v_t - retrieved

        # Update: S = f(S + outer(delta, k_norm))
        outer = torch.einsum('bi,bj->bij', delta, k_norm)
        S_raw = S + outer

        if use_tanh:
            S = torch.tanh(S_raw)
        else:
            S = S_raw

        # Output: Sq = S @ q, out = Sq * silu(Sq)
        Sq = torch.einsum('bij,bj->bi', S, q_t)
        Sq_cache.append(Sq)
        out = Sq * F.silu(Sq)
        outputs.append(out)

    output = torch.stack(outputs, dim=0)
    Sq_cache_tensor = torch.stack(Sq_cache, dim=0)

    return output, S, Sq_cache_tensor


def test_gradient_simple(device='cuda', seed=42):
    """
    Simple test of gradient computation without the kernel.
    """
    torch.manual_seed(seed)

    T = 2
    B = 2
    dim = 32
    n = 32
    dtype = torch.float32  # Use float32 for PyTorch testing

    print(f"\n{'='*60}")
    print(f"Simple gradient test (PyTorch only)")
    print(f"{'='*60}")

    # Create leaf tensors
    x = torch.randn(T, B, dim, device=device, dtype=dtype, requires_grad=True)
    S0 = torch.zeros(B, n, n, device=device, dtype=dtype)

    W_k = nn.Parameter(torch.randn(n, dim, device=device, dtype=dtype) * 0.1)
    W_v = nn.Parameter(torch.randn(n, dim, device=device, dtype=dtype) * 0.1)
    W_q = nn.Parameter(torch.randn(n, dim, device=device, dtype=dtype) * 0.1)

    # Forward
    output, S_final, _ = pytorch_e74_full_matrix_forward_noz(
        x, S0, W_k, W_v, W_q, use_tanh=True
    )

    print(f"Output shape: {output.shape}")
    print(f"Output stats: min={output.min():.4f}, max={output.max():.4f}")

    # Check for NaN in output
    if torch.isnan(output).any():
        print("WARNING: Output contains NaN!")
        return False

    # Backward
    loss = output.mean()
    loss.backward()

    print(f"dx stats: min={x.grad.min():.4f}, max={x.grad.max():.4f}")
    print(f"dW_k stats: min={W_k.grad.min():.4f}, max={W_k.grad.max():.4f}")
    print(f"dW_v stats: min={W_v.grad.min():.4f}, max={W_v.grad.max():.4f}")
    print(f"dW_q stats: min={W_q.grad.min():.4f}, max={W_q.grad.max():.4f}")

    # Check for NaN in gradients
    has_nan = False
    if torch.isnan(x.grad).any():
        print("WARNING: dx contains NaN!")
        has_nan = True
    if torch.isnan(W_k.grad).any():
        print("WARNING: dW_k contains NaN!")
        has_nan = True
    if torch.isnan(W_v.grad).any():
        print("WARNING: dW_v contains NaN!")
        has_nan = True
    if torch.isnan(W_q.grad).any():
        print("WARNING: dW_q contains NaN!")
        has_nan = True

    if not has_nan:
        print("All gradients computed without NaN!")
        return True
    return False


def test_e74_noz_gradient_correctness(n_state=64, use_tanh=True, device='cuda', seed=42):
    """
    Test gradient correctness for E74 full matrix no_z kernel.

    Compares CUDA kernel gradients against PyTorch autograd.
    """
    torch.manual_seed(seed)

    T = 4
    B = 2
    dim = 64
    n = n_state
    dtype = torch.bfloat16

    print(f"\n{'='*60}")
    print(f"Testing E74 Full Matrix no_z gradient correctness")
    print(f"n_state={n}, T={T}, B={B}, dim={dim}, use_tanh={use_tanh}")
    print(f"{'='*60}")

    # Create inputs with moderate scale (avoid very large values)
    x = torch.randn(T, B, dim, device=device, dtype=dtype) * 0.1
    S0 = torch.zeros(B, n, n, device=device, dtype=dtype)

    # Create weights with small initialization to avoid large Sq values
    W_k = torch.randn(n, dim, device=device, dtype=dtype) * 0.1
    W_v = torch.randn(n, dim, device=device, dtype=dtype) * 0.1
    W_q = torch.randn(n, dim, device=device, dtype=dtype) * 0.1
    W_kvq = torch.empty(0, device=device, dtype=dtype)

    proj_type_int = 2  # no_z

    # ========== CUDA Forward ==========
    print("\n--- CUDA Forward ---")
    results = elman_ladder_cuda.e74_full_matrix_forward(
        True,  # training
        x.detach(),
        S0,
        proj_type_int,
        use_tanh,
        W_kvq,
        W_k.detach(),
        W_v.detach(),
        W_q.detach(),
    )

    cuda_S_final = results[0]
    cuda_output = results[1]
    cuda_k_cache = results[2]
    cuda_v_cache = results[3]
    cuda_q_cache = results[4]
    cuda_S_checkpoints = results[5]
    cuda_Sq_cache = results[6]

    print(f"CUDA output shape: {cuda_output.shape}")
    print(f"CUDA output stats: min={cuda_output.min():.6f}, max={cuda_output.max():.6f}")

    # ========== PyTorch Forward ==========
    print("\n--- PyTorch Forward ---")
    # Use float32 for PyTorch reference to get stable gradients
    x_pt = x.detach().float().clone().requires_grad_(True)
    S0_pt = S0.detach().float()
    W_k_pt = W_k.detach().float().clone().requires_grad_(True)
    W_v_pt = W_v.detach().float().clone().requires_grad_(True)
    W_q_pt = W_q.detach().float().clone().requires_grad_(True)

    pt_output, pt_S_final, pt_Sq_cache = pytorch_e74_full_matrix_forward_noz(
        x_pt, S0_pt, W_k_pt, W_v_pt, W_q_pt, use_tanh
    )

    print(f"PyTorch output shape: {pt_output.shape}")
    print(f"PyTorch output stats: min={pt_output.min():.6f}, max={pt_output.max():.6f}")

    # ========== Compare Forward ==========
    print("\n--- Forward Comparison ---")
    fwd_diff = (cuda_output.float() - pt_output.float()).abs().max().item()
    print(f"Max output difference: {fwd_diff:.6e}")

    state_diff = (cuda_S_final.float() - pt_S_final.float()).abs().max().item()
    print(f"Max state difference: {state_diff:.6e}")

    if fwd_diff > 0.01:
        print("WARNING: Large forward difference, gradient comparison may be unreliable")

    # ========== Create gradient for backward ==========
    d_output = torch.randn_like(cuda_output)

    # ========== CUDA Backward ==========
    print("\n--- CUDA Backward ---")
    backward_results = elman_ladder_cuda.e74_full_matrix_backward(
        proj_type_int,
        use_tanh,
        W_kvq,
        W_k.detach(),
        W_v.detach(),
        W_q.detach(),
        x.detach(),
        cuda_S_checkpoints,
        cuda_Sq_cache,
        cuda_k_cache,
        cuda_v_cache,
        cuda_q_cache,
        d_output.contiguous(),
    )

    cuda_dx = backward_results[0]
    cuda_dW_kvq = backward_results[1]
    cuda_dW_k = backward_results[2]
    cuda_dW_v = backward_results[3]
    cuda_dW_q = backward_results[4]

    print(f"CUDA dx shape: {cuda_dx.shape}")
    print(f"CUDA dx stats: min={cuda_dx.min():.6f}, max={cuda_dx.max():.6f}")
    print(f"CUDA dW_k stats: min={cuda_dW_k.min():.6f}, max={cuda_dW_k.max():.6f}")
    print(f"CUDA dW_v stats: min={cuda_dW_v.min():.6f}, max={cuda_dW_v.max():.6f}")
    print(f"CUDA dW_q stats: min={cuda_dW_q.min():.6f}, max={cuda_dW_q.max():.6f}")

    # Check for NaN
    cuda_nan = False
    if torch.isnan(cuda_dx).any():
        print("WARNING: CUDA dx contains NaN!")
        cuda_nan = True
    if torch.isnan(cuda_dW_k).any():
        print("WARNING: CUDA dW_k contains NaN!")
        cuda_nan = True
    if torch.isnan(cuda_dW_v).any():
        print("WARNING: CUDA dW_v contains NaN!")
        cuda_nan = True
    if torch.isnan(cuda_dW_q).any():
        print("WARNING: CUDA dW_q contains NaN!")
        cuda_nan = True

    # ========== PyTorch Backward ==========
    print("\n--- PyTorch Backward ---")
    pt_output.backward(d_output.float())

    pt_dx = x_pt.grad
    pt_dW_k = W_k_pt.grad
    pt_dW_v = W_v_pt.grad
    pt_dW_q = W_q_pt.grad

    print(f"PyTorch dx shape: {pt_dx.shape}")
    print(f"PyTorch dx stats: min={pt_dx.min():.6f}, max={pt_dx.max():.6f}")
    print(f"PyTorch dW_k stats: min={pt_dW_k.min():.6f}, max={pt_dW_k.max():.6f}")
    print(f"PyTorch dW_v stats: min={pt_dW_v.min():.6f}, max={pt_dW_v.max():.6f}")
    print(f"PyTorch dW_q stats: min={pt_dW_q.min():.6f}, max={pt_dW_q.max():.6f}")

    # Check for NaN in PyTorch gradients
    pt_nan = False
    if torch.isnan(pt_dx).any():
        print("WARNING: PyTorch dx contains NaN!")
        pt_nan = True
    if torch.isnan(pt_dW_k).any():
        print("WARNING: PyTorch dW_k contains NaN!")
        pt_nan = True
    if torch.isnan(pt_dW_v).any():
        print("WARNING: PyTorch dW_v contains NaN!")
        pt_nan = True
    if torch.isnan(pt_dW_q).any():
        print("WARNING: PyTorch dW_q contains NaN!")
        pt_nan = True

    if pt_nan:
        print("\nPyTorch reference has NaN gradients, cannot compare!")
        return False

    # ========== Compare Backward ==========
    print("\n--- Backward Comparison ---")

    def compare_grads(name, cuda_grad, pt_grad, atol=0.01, rtol=0.1):
        if cuda_grad is None or pt_grad is None:
            print(f"{name}: SKIPPED (None)")
            return True
        if cuda_grad.numel() == 0:
            print(f"{name}: SKIPPED (empty)")
            return True

        cuda_f = cuda_grad.float()
        pt_f = pt_grad.float()

        if torch.isnan(cuda_f).any():
            print(f"{name}: FAILED (CUDA NaN)")
            return False
        if torch.isnan(pt_f).any():
            print(f"{name}: FAILED (PyTorch NaN)")
            return False

        abs_diff = (cuda_f - pt_f).abs()
        max_abs_diff = abs_diff.max().item()

        # Relative diff only where pt_grad is non-tiny
        mask = pt_f.abs() > 1e-6
        if mask.any():
            rel_diff = abs_diff[mask] / pt_f[mask].abs()
            max_rel_diff = rel_diff.max().item()
        else:
            max_rel_diff = 0.0

        passed = (max_abs_diff < atol) or (max_rel_diff < rtol)
        status = "PASSED" if passed else "FAILED"
        print(f"{name}: max_abs={max_abs_diff:.6e}, max_rel={max_rel_diff:.6e} [{status}]")
        return passed

    results = []
    results.append(compare_grads("dx", cuda_dx, pt_dx, atol=0.01, rtol=0.1))
    results.append(compare_grads("dW_k", cuda_dW_k, pt_dW_k, atol=0.01, rtol=0.1))
    results.append(compare_grads("dW_v", cuda_dW_v, pt_dW_v, atol=0.01, rtol=0.1))
    results.append(compare_grads("dW_q", cuda_dW_q, pt_dW_q, atol=0.01, rtol=0.1))

    all_passed = all(results)

    if all_passed:
        print(f"\nAll gradient checks PASSED for n_state={n}")
    else:
        print(f"\nSome gradient checks FAILED for n_state={n}")

    return all_passed


def main():
    if not torch.cuda.is_available():
        print("Error: CUDA not available!")
        return

    device = 'cuda'

    # First, verify PyTorch reference works without NaN
    test_gradient_simple(device=device)

    # Test gradient for n=64 with tanh (new global memory kernel)
    print("\n" + "="*70)
    print("Testing n_state=64 (global memory kernel)")
    print("="*70)
    test_e74_noz_gradient_correctness(n_state=64, use_tanh=True, device=device)

    # Also test n=48 for comparison
    print("\n" + "="*70)
    print("Testing n_state=48 (global memory kernel)")
    print("="*70)
    test_e74_noz_gradient_correctness(n_state=48, use_tanh=True, device=device)


if __name__ == "__main__":
    main()
