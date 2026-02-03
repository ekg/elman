#!/usr/bin/env python3
"""
Compare CUDA backward against PyTorch with scaled inputs to avoid NaN.
"""

import torch
import torch.nn.functional as F
import hasty_pytorch_lib as elman_ladder_cuda

def pytorch_forward_backward(x, W, S0, d_output):
    """PyTorch reference using fp32 internally (like CUDA)."""
    T, B, dim = x.shape
    n = W.shape[0]

    x_flat = x.reshape(T * B, dim)
    k_all = (x_flat.float() @ W.float().T).reshape(T, B, n)

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

    output = torch.stack(outputs, dim=0).to(x.dtype)
    output.backward(d_output)

    return x.grad.clone(), W.grad.clone(), output


def test_backward_compare():
    torch.manual_seed(42)
    device = torch.device('cuda')
    dtype = torch.bfloat16

    T, B, dim, n = 4, 2, 32, 32

    # Scale inputs to avoid numerical issues
    scale = 0.1

    W = torch.randn(n, dim, device=device, dtype=dtype) * scale
    x = torch.randn(T, B, dim, device=device, dtype=dtype) * scale
    S0 = torch.zeros(B, n, n, device=device, dtype=dtype)

    # Gradient
    torch.manual_seed(123)
    d_output = torch.randn(T, B, n, device=device, dtype=dtype) * 0.1

    # ===== PYTORCH REFERENCE =====
    print("=" * 60)
    print("PyTorch Reference")
    print("=" * 60)

    W_py = W.clone().detach().requires_grad_(True)
    x_py = x.clone().detach().requires_grad_(True)

    d_x_py, d_W_py, output_py = pytorch_forward_backward(x_py, W_py, S0, d_output)

    print(f"output[-1,0,:4]: {output_py[-1,0,:4].float()}")
    print(f"d_x[0,0,:4]: {d_x_py[0,0,:4].float()}")
    print(f"d_W[0,:4]: {d_W_py[0,:4].float()}")
    print(f"d_x has NaN: {torch.isnan(d_x_py).any()}")

    # ===== CUDA =====
    print("\n" + "=" * 60)
    print("CUDA Kernel")
    print("=" * 60)

    W_cuda = W.clone().detach()
    x_cuda = x.clone().detach()
    empty = torch.empty(0, device=device, dtype=dtype)

    # Forward
    results = elman_ladder_cuda.e74_full_matrix_forward(
        True, x_cuda, S0, 0, True, W_cuda, empty, empty, empty)

    S_cuda = results[0]
    output_cuda = results[1]
    k_cache = results[2]
    v_cache = results[3]
    q_cache = results[4]
    S_checkpoints = results[5]
    Sq_cache = results[6]

    print(f"output[-1,0,:4]: {output_cuda[-1,0,:4].float()}")

    # Compare forward
    out_diff = (output_py.float() - output_cuda.float()).abs()
    print(f"Forward diff: max={out_diff.max():.8f}")

    # Backward
    results_bwd = elman_ladder_cuda.e74_full_matrix_backward(
        0,  # proj_type
        True,  # use_tanh
        W_cuda,
        empty, empty, empty,
        x_cuda,
        S_checkpoints,
        Sq_cache,
        k_cache,
        v_cache,
        q_cache,
        d_output)

    d_x_cuda = results_bwd[0]
    d_W_cuda = results_bwd[1]

    print(f"d_x[0,0,:4]: {d_x_cuda[0,0,:4].float()}")
    print(f"d_W[0,:4]: {d_W_cuda[0,:4].float()}")
    print(f"d_x has NaN: {torch.isnan(d_x_cuda).any()}")

    # ===== COMPARE =====
    print("\n" + "=" * 60)
    print("Gradient Comparison")
    print("=" * 60)

    d_x_diff = (d_x_py.float() - d_x_cuda.float()).abs()
    d_W_diff = (d_W_py.float() - d_W_cuda.float()).abs()

    print(f"d_x diff: max={d_x_diff.max():.6f}, mean={d_x_diff.mean():.6f}")
    print(f"d_W diff: max={d_W_diff.max():.6f}, mean={d_W_diff.mean():.6f}")

    # Relative errors
    d_x_rel = d_x_diff / (d_x_py.float().abs() + 1e-8)
    d_W_rel = d_W_diff / (d_W_py.float().abs() + 1e-8)
    print(f"\nd_x relative: max={d_x_rel.max():.4f}, mean={d_x_rel.mean():.4f}")
    print(f"d_W relative: max={d_W_rel.max():.4f}, mean={d_W_rel.mean():.4f}")

    # Find worst mismatches
    max_idx = d_x_diff.argmax()
    t, b, d_idx = max_idx // (B * dim), (max_idx % (B * dim)) // dim, max_idx % dim
    print(f"\nWorst d_x mismatch at [{t},{b},{d_idx}]:")
    print(f"  PyTorch: {d_x_py[t,b,d_idx].float():.8f}")
    print(f"  CUDA:    {d_x_cuda[t,b,d_idx].float():.8f}")

    if d_x_diff.max() < 0.01 and d_W_diff.max() < 0.01:
        print("\n*** BACKWARD MATCHES! ***")
    else:
        print("\n*** BACKWARD MISMATCH ***")


if __name__ == '__main__':
    test_backward_compare()
