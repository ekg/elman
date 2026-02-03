#!/usr/bin/env python3
"""
Debug d_k values directly to find the gradient bug.
"""

import torch
import torch.nn.functional as F
import hasty_pytorch_lib as elman_ladder_cuda

def test_dk_debug():
    torch.manual_seed(42)
    device = torch.device('cuda')
    dtype = torch.bfloat16

    T, B, dim, n = 2, 1, 32, 32  # n_state must be 32, 48, 64, or 96

    scale = 0.1
    W = torch.randn(n, dim, device=device, dtype=dtype) * scale
    x = torch.randn(T, B, dim, device=device, dtype=dtype) * scale
    S0 = torch.zeros(B, n, n, device=device, dtype=dtype)

    # ===== PYTORCH: Compute d_k directly =====
    print("=" * 60)
    print("PyTorch: Compute d_k")
    print("=" * 60)

    x_req_grad = x.clone().detach().requires_grad_(True)
    W_req_grad = W.clone().detach().requires_grad_(True)

    # Forward
    x_flat = x_req_grad.reshape(T * B, dim)
    k_all = (x_flat.float() @ W_req_grad.float().T).reshape(T, B, n)
    k_all.retain_grad()  # Keep gradient for k

    S = S0.float()
    outputs = []

    for t in range(T):
        k_raw = k_all[t]
        k_norm_val = (k_raw * k_raw).sum(dim=-1, keepdim=True).sqrt() + 1e-6
        k_norm = k_raw / k_norm_val
        retrieved = torch.einsum('bij,bj->bi', S, k_norm)
        delta = k_raw - retrieved
        outer = torch.einsum('bi,bj->bij', delta, k_norm)
        S = torch.tanh(S + outer)
        Sq = torch.einsum('bij,bj->bi', S, k_raw)
        sig = 1.0 / (1.0 + torch.exp(-Sq))
        out = Sq * Sq * sig
        outputs.append(out)

    output = torch.stack(outputs, dim=0).to(dtype)

    # Backward
    torch.manual_seed(123)
    d_output = torch.randn_like(output) * 0.1
    output.backward(d_output)

    d_k_py = k_all.grad  # This is the gradient wrt k_all
    d_W_py = W_req_grad.grad
    d_x_py = x_req_grad.grad

    print(f"d_k[0,0,:8]: {d_k_py[0,0,:8].float()}")
    print(f"d_k[1,0,:8]: {d_k_py[1,0,:8].float()}")
    print(f"d_W[0,:4]: {d_W_py[0,:4].float()}")
    print(f"d_x[0,0,:4]: {d_x_py[0,0,:4].float()}")

    # ===== CUDA: Get d_k from backward =====
    print("\n" + "=" * 60)
    print("CUDA: Compute d_k")
    print("=" * 60)

    W_cuda = W.clone().detach()
    x_cuda = x.clone().detach()
    empty = torch.empty(0, device=device, dtype=dtype)

    # Forward
    results = elman_ladder_cuda.e74_full_matrix_forward(
        True, x_cuda, S0, 0, True, W_cuda, empty, empty, empty)

    S_cuda = results[0]
    output_cuda = results[1]
    k_cache = results[2]  # This is the projected k
    S_checkpoints = results[5]
    Sq_cache = results[6]

    # Backward
    results_bwd = elman_ladder_cuda.e74_full_matrix_backward(
        0, True, W_cuda, empty, empty, empty,
        x_cuda, S_checkpoints, Sq_cache, k_cache, empty, empty,
        d_output)

    d_x_cuda = results_bwd[0]
    d_W_cuda = results_bwd[1]

    # To get d_k from CUDA, we need to look at the workspace or add it to return values
    # For now, let's compute d_k from d_x and d_W relationship:
    # k = x @ W^T, so d_k @ x^T = d_W, d_x = d_k @ W
    # We can verify d_x = d_k @ W by computing d_k_cuda = d_x @ W_cuda^T / ||some factor||
    # But this is circular...

    # Instead, let's just compare the final gradients
    print(f"k_cache[0,0,:8]: {k_cache[0,0,:8].float()}")
    print(f"k_cache[1,0,:8]: {k_cache[1,0,:8].float()}")
    print(f"d_W[0,:4]: {d_W_cuda[0,:4].float()}")
    print(f"d_x[0,0,:4]: {d_x_cuda[0,0,:4].float()}")

    # ===== COMPARE =====
    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)

    # Compare k projections
    k_diff = (k_all.detach().float() - k_cache.float()).abs()
    print(f"k projection diff: max={k_diff.max():.8f}")

    # Compare d_W
    d_W_diff = (d_W_py.float() - d_W_cuda.float()).abs()
    print(f"d_W diff: max={d_W_diff.max():.6f}, mean={d_W_diff.mean():.6f}")

    # Element-wise comparison
    print(f"\nPyTorch d_W[0,:8]: {d_W_py[0,:8].float()}")
    print(f"CUDA d_W[0,:8]:    {d_W_cuda[0,:8].float()}")

    # Compare d_x
    d_x_diff = (d_x_py.float() - d_x_cuda.float()).abs()
    print(f"\nd_x diff: max={d_x_diff.max():.6f}, mean={d_x_diff.mean():.6f}")


if __name__ == '__main__':
    test_dk_debug()
