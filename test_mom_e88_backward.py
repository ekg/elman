#!/usr/bin/env python
"""
Test MoM E88 backward pass using finite differences.
"""

import torch
import sys
sys.path.insert(0, '/home/erikg/elman')

import hasty_pytorch_lib

device = 'cuda'
dtype = torch.bfloat16


def compute_forward(k, v, q, decay, head_indices, router_weights, S0, H, K):
    """Run CUDA forward pass."""
    k_t = k.transpose(0, 1).contiguous()
    v_t = v.transpose(0, 1).contiguous()
    q_t = q.transpose(0, 1).contiguous()
    decay_t = decay.transpose(0, 1).contiguous()
    head_indices_t = head_indices.transpose(0, 1).contiguous()
    router_weights_t = router_weights.transpose(0, 1).contiguous()

    output_t, S_out, S_cache = hasty_pytorch_lib.mom_e88_forward(
        True, k_t, v_t, q_t, decay_t,
        head_indices_t, router_weights_t,
        S0, H, K
    )
    return output_t.transpose(0, 1), S_out, S_cache


def compute_backward(k, v, q, decay, head_indices, router_weights, S0, H, K, d_output, S_cache):
    """Run CUDA backward pass."""
    k_t = k.transpose(0, 1).contiguous()
    v_t = v.transpose(0, 1).contiguous()
    q_t = q.transpose(0, 1).contiguous()
    decay_t = decay.transpose(0, 1).contiguous()
    head_indices_t = head_indices.transpose(0, 1).contiguous()
    router_weights_t = router_weights.transpose(0, 1).contiguous()
    d_output_t = d_output.transpose(0, 1).contiguous()

    d_k, d_v, d_q, d_decay, d_router_weights = hasty_pytorch_lib.mom_e88_backward(
        k_t, v_t, q_t, decay_t,
        head_indices_t, router_weights_t,
        S_cache, d_output_t, H, K
    )

    return d_k.transpose(0, 1), d_v.transpose(0, 1), d_q.transpose(0, 1), \
           d_decay.transpose(0, 1), d_router_weights.transpose(0, 1)


def test_backward_finite_diff():
    """Test backward pass using finite differences."""
    print("Testing MoM E88 backward pass with finite differences")
    print("=" * 60)

    B, T, H, K = 1, 2, 4, 2
    n_state = 16
    head_v_dim = 16

    torch.manual_seed(42)

    # Use float32 for finite differences (more precision needed)
    k = torch.randn(B, T, H, n_state, device=device, dtype=torch.float32)
    v = torch.randn(B, T, H, head_v_dim, device=device, dtype=torch.float32)
    q = torch.randn(B, T, H, n_state, device=device, dtype=torch.float32)
    decay = torch.sigmoid(torch.randn(B, T, H, device=device, dtype=torch.float32))
    router_weights = torch.softmax(torch.randn(B, T, K, device=device, dtype=torch.float32), dim=-1)

    # Normalize k and q
    k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)

    # Fixed head indices
    head_indices = torch.tensor([[[0, 1], [2, 3]]], device=device, dtype=torch.int32)

    # Initial state (per-slot)
    S0 = torch.zeros(B, K, n_state, head_v_dim, device=device, dtype=dtype)

    # Random gradient
    d_output = torch.randn(B, T, head_v_dim, device=device, dtype=dtype)

    # Convert to bf16 for forward/backward
    k_bf16 = k.to(dtype)
    v_bf16 = v.to(dtype)
    q_bf16 = q.to(dtype)
    decay_bf16 = decay.to(dtype)
    router_weights_bf16 = router_weights.to(dtype)

    # Compute forward
    output, S_out, S_cache = compute_forward(
        k_bf16, v_bf16, q_bf16, decay_bf16,
        head_indices, router_weights_bf16, S0, H, K
    )

    # Compute backward
    d_k, d_v, d_q, d_decay, d_rw = compute_backward(
        k_bf16, v_bf16, q_bf16, decay_bf16,
        head_indices, router_weights_bf16, S0, H, K,
        d_output, S_cache
    )

    print(f"Backward pass returned:")
    print(f"  d_k shape: {d_k.shape}, max: {d_k.abs().max().item():.6f}")
    print(f"  d_v shape: {d_v.shape}, max: {d_v.abs().max().item():.6f}")
    print(f"  d_q shape: {d_q.shape}, max: {d_q.abs().max().item():.6f}")
    print(f"  d_decay shape: {d_decay.shape}, max: {d_decay.abs().max().item():.6f}")
    print(f"  d_router_weights shape: {d_rw.shape}, max: {d_rw.abs().max().item():.6f}")

    # Finite difference check for a few elements
    eps = 1e-3
    errors = []

    # Check d_v gradient for one element
    print("\nFinite difference check for d_v[0, 0, 0, 0]:")
    v_plus = v_bf16.clone()
    v_plus[0, 0, 0, 0] += eps
    output_plus, _, _ = compute_forward(k_bf16, v_plus, q_bf16, decay_bf16, head_indices, router_weights_bf16, S0, H, K)

    v_minus = v_bf16.clone()
    v_minus[0, 0, 0, 0] -= eps
    output_minus, _, _ = compute_forward(k_bf16, v_minus, q_bf16, decay_bf16, head_indices, router_weights_bf16, S0, H, K)

    # Compute loss as sum(output * d_output)
    loss_plus = (output_plus.float() * d_output.float()).sum()
    loss_minus = (output_minus.float() * d_output.float()).sum()
    fd_grad = (loss_plus - loss_minus) / (2 * eps)
    cuda_grad = d_v[0, 0, 0, 0].float().item()

    print(f"  CUDA gradient: {cuda_grad:.6f}")
    print(f"  Finite diff:   {fd_grad.item():.6f}")
    rel_error = abs(cuda_grad - fd_grad.item()) / (abs(fd_grad.item()) + 1e-8)
    print(f"  Relative error: {rel_error:.4f}")
    errors.append(rel_error)

    # Check d_k gradient
    print("\nFinite difference check for d_k[0, 0, 0, 0]:")
    k_plus = k_bf16.clone()
    k_plus[0, 0, 0, 0] += eps
    output_plus, _, _ = compute_forward(k_plus, v_bf16, q_bf16, decay_bf16, head_indices, router_weights_bf16, S0, H, K)

    k_minus = k_bf16.clone()
    k_minus[0, 0, 0, 0] -= eps
    output_minus, _, _ = compute_forward(k_minus, v_bf16, q_bf16, decay_bf16, head_indices, router_weights_bf16, S0, H, K)

    loss_plus = (output_plus.float() * d_output.float()).sum()
    loss_minus = (output_minus.float() * d_output.float()).sum()
    fd_grad = (loss_plus - loss_minus) / (2 * eps)
    cuda_grad = d_k[0, 0, 0, 0].float().item()

    print(f"  CUDA gradient: {cuda_grad:.6f}")
    print(f"  Finite diff:   {fd_grad.item():.6f}")
    rel_error = abs(cuda_grad - fd_grad.item()) / (abs(fd_grad.item()) + 1e-8)
    print(f"  Relative error: {rel_error:.4f}")
    errors.append(rel_error)

    # Check d_q gradient
    print("\nFinite difference check for d_q[0, 0, 0, 0]:")
    q_plus = q_bf16.clone()
    q_plus[0, 0, 0, 0] += eps
    output_plus, _, _ = compute_forward(k_bf16, v_bf16, q_plus, decay_bf16, head_indices, router_weights_bf16, S0, H, K)

    q_minus = q_bf16.clone()
    q_minus[0, 0, 0, 0] -= eps
    output_minus, _, _ = compute_forward(k_bf16, v_bf16, q_minus, decay_bf16, head_indices, router_weights_bf16, S0, H, K)

    loss_plus = (output_plus.float() * d_output.float()).sum()
    loss_minus = (output_minus.float() * d_output.float()).sum()
    fd_grad = (loss_plus - loss_minus) / (2 * eps)
    cuda_grad = d_q[0, 0, 0, 0].float().item()

    print(f"  CUDA gradient: {cuda_grad:.6f}")
    print(f"  Finite diff:   {fd_grad.item():.6f}")
    rel_error = abs(cuda_grad - fd_grad.item()) / (abs(fd_grad.item()) + 1e-8)
    print(f"  Relative error: {rel_error:.4f}")
    errors.append(rel_error)

    print("\n" + "=" * 60)
    avg_error = sum(errors) / len(errors)
    print(f"Average relative error: {avg_error:.4f}")

    # Threshold is high because we're comparing BF16 CUDA with finite diff
    threshold = 0.5
    if avg_error < threshold:
        print(f"PASS: Errors within acceptable range (< {threshold})")
        return True
    else:
        print(f"FAIL: Errors exceed threshold ({threshold})")
        return False


if __name__ == '__main__':
    success = test_backward_finite_diff()
    sys.exit(0 if success else 1)
