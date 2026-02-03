#!/usr/bin/env python3
"""
Check if CUDA d_W is transposed compared to PyTorch d_W.
"""

import torch
import hasty_pytorch_lib as elman_ladder_cuda

def test_dw_transpose():
    torch.manual_seed(42)
    device = torch.device('cuda')
    dtype = torch.bfloat16

    T, B, dim, n = 1, 1, 32, 32

    W = torch.randn(n, dim, device=device, dtype=dtype)
    x = torch.randn(T, B, dim, device=device, dtype=dtype)
    S0 = torch.zeros(B, n, n, device=device, dtype=dtype)

    # PyTorch reference
    x_py = x.clone().detach().requires_grad_(True)
    W_py = W.clone().detach().requires_grad_(True)

    x_flat = x_py.reshape(T * B, dim)
    k_all = (x_flat.float() @ W_py.float().T).reshape(T, B, n)
    k_all.retain_grad()

    S = S0.float()
    k_raw = k_all[0]
    k_norm_val = (k_raw * k_raw).sum(dim=-1, keepdim=True).sqrt() + 1e-6
    k_norm = k_raw / k_norm_val
    retrieved = torch.einsum('bij,bj->bi', S, k_norm)
    delta = k_raw - retrieved
    outer = torch.einsum('bi,bj->bij', delta, k_norm)
    S = torch.tanh(S + outer)
    Sq = torch.einsum('bij,bj->bi', S, k_raw)
    sig = 1.0 / (1.0 + torch.exp(-Sq))
    out = (Sq * Sq * sig).to(dtype)

    d_output = torch.ones_like(out)
    out.backward(d_output)
    d_W_py = W_py.grad

    # CUDA
    empty = torch.empty(0, device=device, dtype=dtype)
    results = elman_ladder_cuda.e74_full_matrix_forward(
        True, x, S0, 0, True, W, empty, empty, empty)
    k_cache = results[2]
    S_checkpoints = results[5]
    Sq_cache = results[6]

    results_bwd = elman_ladder_cuda.e74_full_matrix_backward(
        0, True, W, empty, empty, empty,
        x, S_checkpoints, Sq_cache, k_cache, empty, empty,
        d_output)
    d_W_cuda = results_bwd[1]

    print("d_W_py shape:", d_W_py.shape)
    print("d_W_cuda shape:", d_W_cuda.shape)

    # Check if CUDA d_W is transposed
    d_W_cuda_T = d_W_cuda.T  # Try transpose

    diff_original = (d_W_py.float() - d_W_cuda.float()).abs()
    diff_transposed = (d_W_py.float() - d_W_cuda_T.float()).abs()

    print(f"\nOriginal diff: max={diff_original.max():.4f}, mean={diff_original.mean():.4f}")
    print(f"Transposed diff: max={diff_transposed.max():.4f}, mean={diff_transposed.mean():.4f}")

    # Check a specific pattern
    print(f"\nPyTorch d_W[0,:4]: {d_W_py[0,:4].float()}")
    print(f"CUDA d_W[0,:4]:    {d_W_cuda[0,:4].float()}")
    print(f"CUDA d_W[:,0][:4]: {d_W_cuda[:,0][:4].float()}")

    if diff_transposed.max() < diff_original.max() * 0.1:
        print("\n*** CUDA d_W IS TRANSPOSED! ***")
    else:
        print("\n*** NOT a simple transpose issue ***")


if __name__ == '__main__':
    test_dw_transpose()
