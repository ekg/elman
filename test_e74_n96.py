#!/usr/bin/env python3
"""
Test E74 Full Matrix CUDA kernel with n=96 (global memory kernel).
"""

import torch
import hasty_pytorch_lib as elman_ladder_cuda

def pytorch_reference(x, W, S0, T, B, n):
    """PyTorch reference implementation."""
    x_flat = x.reshape(T * B, -1)
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

    return torch.stack(outputs, dim=0)

def test_n96():
    torch.manual_seed(42)
    device = torch.device('cuda')
    dtype = torch.bfloat16

    T, B, dim, n = 4, 2, 128, 96  # n=96 uses global memory kernel
    scale = 0.1

    W = torch.randn(n, dim, device=device, dtype=dtype) * scale
    x = torch.randn(T, B, dim, device=device, dtype=dtype) * scale
    S0 = torch.zeros(B, n, n, device=device, dtype=dtype)

    print("=" * 80)
    print(f"Testing E74 Full Matrix with n={n} (Global Memory Kernel)")
    print(f"Config: T={T}, B={B}, dim={dim}, n={n}")
    print("=" * 80)

    # PyTorch reference with gradients
    x_py = x.clone().detach().requires_grad_(True)
    W_py = W.clone().detach().requires_grad_(True)
    output_py = pytorch_reference(x_py, W_py, S0, T, B, n).to(dtype)

    d_output = torch.ones_like(output_py)
    output_py.backward(d_output)
    d_W_py = W_py.grad
    d_x_py = x_py.grad

    if d_W_py.isnan().any() or d_x_py.isnan().any():
        print("PyTorch produced NaN gradients!")
        return

    print("PyTorch reference computed successfully")
    print(f"  output range: [{output_py.min().item():.4f}, {output_py.max().item():.4f}]")

    # CUDA implementation
    try:
        empty = torch.empty(0, device=device, dtype=dtype)
        results = elman_ladder_cuda.e74_full_matrix_forward(
            True, x, S0, 0, True, W, empty, empty, empty)

        output_cuda = results[1]
        k_cache = results[2]
        S_checkpoints = results[5]
        Sq_cache = results[6]

        print("CUDA forward computed successfully")
        print(f"  output range: [{output_cuda.min().item():.4f}, {output_cuda.max().item():.4f}]")

        results_bwd = elman_ladder_cuda.e74_full_matrix_backward(
            0, True, W, empty, empty, empty,
            x, S_checkpoints, Sq_cache, k_cache, empty, empty,
            d_output)

        d_x_cuda = results_bwd[0]
        d_W_cuda = results_bwd[1]

        print("CUDA backward computed successfully")

    except Exception as e:
        print(f"CUDA ERROR: {e}")
        return

    # Compute metrics
    out_cos = torch.nn.functional.cosine_similarity(
        output_py.float().flatten().unsqueeze(0),
        output_cuda.float().flatten().unsqueeze(0)).item()

    d_W_cos = torch.nn.functional.cosine_similarity(
        d_W_py.float().flatten().unsqueeze(0),
        d_W_cuda.float().flatten().unsqueeze(0)).item()

    d_x_cos = torch.nn.functional.cosine_similarity(
        d_x_py.float().flatten().unsqueeze(0),
        d_x_cuda.float().flatten().unsqueeze(0)).item()

    out_diff = (output_py.float() - output_cuda.float()).abs()
    d_W_diff = (d_W_py.float() - d_W_cuda.float()).abs()
    d_x_diff = (d_x_py.float() - d_x_cuda.float()).abs()

    print()
    print("Results:")
    print(f"  Forward output:  cosine={out_cos:.6f}, max_diff={out_diff.max().item():.6f}")
    print(f"  d_W gradient:    cosine={d_W_cos:.6f}, max_diff={d_W_diff.max().item():.6f}")
    print(f"  d_x gradient:    cosine={d_x_cos:.6f}, max_diff={d_x_diff.max().item():.6f}")

    # Pass/fail
    ok = out_cos > 0.999 and d_W_cos > 0.999 and d_x_cos > 0.999
    print()
    if ok:
        print("PASS: n=96 global memory kernel works correctly!")
    else:
        print("FAIL: Cosine similarity below threshold")
    print("=" * 80)

if __name__ == '__main__':
    test_n96()
