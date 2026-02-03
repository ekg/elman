#!/usr/bin/env python3
"""
Compare CUDA backward against PyTorch autograd reference.
"""

import torch
import torch.nn.functional as F
import hasty_pytorch_lib as elman_ladder_cuda

def pytorch_forward(x, W, S0):
    """
    PyTorch reference forward that matches CUDA exactly.
    Uses fp32 intermediates with bf16 storage.
    """
    T, B, dim = x.shape
    n = W.shape[0]

    # Projection (like cuBLAS)
    x_flat = x.reshape(T * B, dim)
    k_all = (x_flat @ W.T).reshape(T, B, n)

    # Forward loop
    S = S0.float()  # fp32 for computation
    outputs = []

    for t in range(T):
        k_raw = k_all[t].float()

        # Normalize k
        k_norm_val = (k_raw * k_raw).sum(dim=-1, keepdim=True).sqrt() + 1e-6
        k_norm = k_raw / k_norm_val

        # retrieved = S @ k_norm
        retrieved = torch.einsum('bij,bj->bi', S, k_norm)

        # delta = v - retrieved (v = original k)
        delta = k_raw - retrieved

        # State update
        outer = torch.einsum('bi,bj->bij', delta, k_norm)
        S = torch.tanh(S + outer)

        # Output: Sq = S @ q (q = original k)
        Sq = torch.einsum('bij,bj->bi', S, k_raw)
        sig = 1.0 / (1.0 + torch.exp(-Sq))
        out = (Sq * Sq * sig).to(x.dtype)
        outputs.append(out)

    output = torch.stack(outputs, dim=0)
    S_final = S.to(x.dtype)

    return output, S_final


def test_backward():
    torch.manual_seed(42)
    device = torch.device('cuda')
    dtype = torch.bfloat16

    T, B, dim, n = 8, 2, 32, 32

    # Inputs with gradients
    W = torch.randn(n, dim, device=device, dtype=dtype, requires_grad=True)
    x = torch.randn(T, B, dim, device=device, dtype=dtype, requires_grad=True)
    S0 = torch.zeros(B, n, n, device=device, dtype=dtype)

    # ===== PYTORCH REFERENCE =====
    print("=" * 60)
    print("PyTorch Reference")
    print("=" * 60)

    W_py = W.clone().detach().requires_grad_(True)
    x_py = x.clone().detach().requires_grad_(True)

    output_py, S_py = pytorch_forward(x_py, W_py, S0)

    print(f"output shape: {output_py.shape}")
    print(f"output[-1,0,:4]: {output_py[-1,0,:4].float()}")

    # Backward
    torch.manual_seed(123)
    d_output = torch.randn_like(output_py)

    output_py.backward(d_output)
    d_x_py = x_py.grad.clone()
    d_W_py = W_py.grad.clone()

    print(f"\nd_x shape: {d_x_py.shape}")
    print(f"d_x[0,0,:4]: {d_x_py[0,0,:4].float()}")
    print(f"d_W[0,:4]: {d_W_py[0,:4].float()}")

    # ===== CUDA =====
    print("\n" + "=" * 60)
    print("CUDA Kernel")
    print("=" * 60)

    W_cuda = W.clone().detach().requires_grad_(True)
    x_cuda = x.clone().detach().requires_grad_(True)
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

    print(f"output shape: {output_cuda.shape}")
    print(f"output[-1,0,:4]: {output_cuda[-1,0,:4].float()}")

    # Compare forward
    out_diff = (output_py.float() - output_cuda.float()).abs()
    print(f"\nForward output diff: max={out_diff.max():.8f}")

    # Backward
    results_bwd = elman_ladder_cuda.e74_full_matrix_backward(
        0,  # proj_type
        True,  # use_tanh
        W_cuda,
        empty, empty, empty,  # W_k, W_v, W_q
        x_cuda,
        S_checkpoints,
        Sq_cache,
        k_cache,
        v_cache,
        q_cache,
        d_output)

    d_x_cuda = results_bwd[0]
    d_W_cuda = results_bwd[1]

    print(f"\nd_x shape: {d_x_cuda.shape}")
    print(f"d_x[0,0,:4]: {d_x_cuda[0,0,:4].float()}")
    print(f"d_W shape: {d_W_cuda.shape}")
    print(f"d_W[0,:4]: {d_W_cuda[0,:4].float()}")

    # ===== COMPARE =====
    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)

    d_x_diff = (d_x_py.float() - d_x_cuda.float()).abs()
    d_W_diff = (d_W_py.float() - d_W_cuda.float()).abs()

    print(f"d_x diff: max={d_x_diff.max():.6f}, mean={d_x_diff.mean():.6f}")
    print(f"d_W diff: max={d_W_diff.max():.6f}, mean={d_W_diff.mean():.6f}")

    # Find where max diff is
    max_idx = d_x_diff.argmax()
    t = max_idx // (B * dim)
    b = (max_idx % (B * dim)) // dim
    d_idx = max_idx % dim

    print(f"\nMax d_x diff at [{t},{b},{d_idx}]:")
    print(f"  PyTorch: {d_x_py[t,b,d_idx].float():.8f}")
    print(f"  CUDA:    {d_x_cuda[t,b,d_idx].float():.8f}")

    # Check relative error
    d_x_rel = d_x_diff / (d_x_py.float().abs() + 1e-8)
    print(f"\nd_x relative error: max={d_x_rel.max():.4f}")

    if d_x_diff.max() < 0.1 and d_W_diff.max() < 0.1:
        print("\n*** BACKWARD MATCHES! ***")
    else:
        print("\n*** BACKWARD MISMATCH - CUDA kernel has bugs ***")


if __name__ == '__main__':
    test_backward()
