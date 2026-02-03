#!/usr/bin/env python3
"""
Test backward with T=1 to simplify debugging.
"""

import torch
import torch.nn.functional as F
import hasty_pytorch_lib as elman_ladder_cuda

def test_backward_t1():
    torch.manual_seed(42)
    device = torch.device('cuda')
    dtype = torch.bfloat16

    T, B, dim, n = 1, 1, 32, 32  # Single timestep

    scale = 1.0  # Larger scale for better precision
    W = torch.randn(n, dim, device=device, dtype=dtype) * scale
    x = torch.randn(T, B, dim, device=device, dtype=dtype) * scale
    S0 = torch.zeros(B, n, n, device=device, dtype=dtype)

    # ===== PYTORCH Reference =====
    print("=" * 60)
    print("PyTorch (T=1)")
    print("=" * 60)

    x_py = x.clone().detach().requires_grad_(True)
    W_py = W.clone().detach().requires_grad_(True)

    # Forward
    x_flat = x_py.reshape(T * B, dim)
    k_all = (x_flat.float() @ W_py.float().T).reshape(T, B, n)
    k_all.retain_grad()

    S = S0.float()
    k_raw = k_all[0]
    k_norm_val = (k_raw * k_raw).sum(dim=-1, keepdim=True).sqrt() + 1e-6
    k_norm = k_raw / k_norm_val
    retrieved = torch.einsum('bij,bj->bi', S, k_norm)  # 0 since S=0
    delta = k_raw - retrieved  # = k_raw
    outer = torch.einsum('bi,bj->bij', delta, k_norm)
    S = torch.tanh(S + outer)  # = tanh(outer)
    Sq = torch.einsum('bij,bj->bi', S, k_raw)
    sig = 1.0 / (1.0 + torch.exp(-Sq))
    out = (Sq * Sq * sig).to(dtype)

    # Backward
    d_output = torch.ones_like(out)
    out.backward(d_output)

    d_k_py = k_all.grad
    d_W_py = W_py.grad
    d_x_py = x_py.grad

    print(f"k_raw[:8]: {k_raw[0,:8].detach().float()}")
    print(f"d_k[:8]: {d_k_py[0,0,:8].float()}")
    print(f"d_W[0,:8]: {d_W_py[0,:8].float()}")
    print(f"d_x[0,0,:8]: {d_x_py[0,0,:8].float()}")

    # ===== CUDA =====
    print("\n" + "=" * 60)
    print("CUDA (T=1)")
    print("=" * 60)

    W_cuda = W.clone().detach()
    x_cuda = x.clone().detach()
    empty = torch.empty(0, device=device, dtype=dtype)

    # Forward
    results = elman_ladder_cuda.e74_full_matrix_forward(
        True, x_cuda, S0, 0, True, W_cuda, empty, empty, empty)

    output_cuda = results[1]
    k_cache = results[2]
    S_checkpoints = results[5]
    Sq_cache = results[6]

    print(f"k_cache[:8]: {k_cache[0,0,:8].float()}")

    # Backward
    d_output_cuda = torch.ones_like(output_cuda)
    results_bwd = elman_ladder_cuda.e74_full_matrix_backward(
        0, True, W_cuda, empty, empty, empty,
        x_cuda, S_checkpoints, Sq_cache, k_cache, empty, empty,
        d_output_cuda)

    d_x_cuda = results_bwd[0]
    d_W_cuda = results_bwd[1]

    print(f"d_W[0,:8]: {d_W_cuda[0,:8].float()}")
    print(f"d_x[0,0,:8]: {d_x_cuda[0,0,:8].float()}")

    # ===== COMPARE =====
    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)

    k_diff = (k_all.detach().float() - k_cache.float()).abs()
    d_W_diff = (d_W_py.float() - d_W_cuda.float()).abs()
    d_x_diff = (d_x_py.float() - d_x_cuda.float()).abs()

    print(f"k projection diff: max={k_diff.max():.8f}")
    print(f"d_W diff: max={d_W_diff.max():.6f}, mean={d_W_diff.mean():.6f}")
    print(f"d_x diff: max={d_x_diff.max():.6f}, mean={d_x_diff.mean():.6f}")

    # Relative errors
    d_W_rel = d_W_diff / (d_W_py.float().abs() + 1e-10)
    d_x_rel = d_x_diff / (d_x_py.float().abs() + 1e-10)
    print(f"\nd_W relative: max={d_W_rel.max():.4f}, mean={d_W_rel.mean():.4f}")
    print(f"d_x relative: max={d_x_rel.max():.4f}, mean={d_x_rel.mean():.4f}")


if __name__ == '__main__':
    test_backward_t1()
