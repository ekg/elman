#!/usr/bin/env python3
"""
Robust backward test with proper relative error handling.
"""

import torch
import hasty_pytorch_lib as elman_ladder_cuda

def test_backward(T, B, dim, n, scale=0.1, seed=42):
    torch.manual_seed(seed)
    device = torch.device('cuda')
    dtype = torch.bfloat16

    W = torch.randn(n, dim, device=device, dtype=dtype) * scale
    x = torch.randn(T, B, dim, device=device, dtype=dtype) * scale
    S0 = torch.zeros(B, n, n, device=device, dtype=dtype)

    # PyTorch reference
    x_py = x.clone().detach().requires_grad_(True)
    W_py = W.clone().detach().requires_grad_(True)

    x_flat = x_py.reshape(T * B, dim)
    k_all = (x_flat.float() @ W_py.float().T).reshape(T, B, n)

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

    d_output = torch.ones_like(output)
    output.backward(d_output)
    d_W_py = W_py.grad
    d_x_py = x_py.grad
    
    if d_W_py.isnan().any() or d_x_py.isnan().any():
        return {'status': 'pytorch_nan'}

    # CUDA
    empty = torch.empty(0, device=device, dtype=dtype)
    results = elman_ladder_cuda.e74_full_matrix_forward(
        True, x, S0, 0, True, W, empty, empty, empty)

    k_cache = results[2]
    S_checkpoints = results[5]
    Sq_cache = results[6]
    output_cuda = results[1]

    results_bwd = elman_ladder_cuda.e74_full_matrix_backward(
        0, True, W, empty, empty, empty,
        x, S_checkpoints, Sq_cache, k_cache, empty, empty,
        d_output)

    d_x_cuda = results_bwd[0]
    d_W_cuda = results_bwd[1]
    
    if d_W_cuda.isnan().any() or d_x_cuda.isnan().any():
        return {'status': 'cuda_nan'}

    # Compare using normalized RMSE instead of max relative error
    out_diff = (output.float() - output_cuda.float()).abs()
    d_W_diff = (d_W_py.float() - d_W_cuda.float()).abs()
    d_x_diff = (d_x_py.float() - d_x_cuda.float()).abs()

    # Cosine similarity (more robust than relative error)
    d_W_cos = torch.nn.functional.cosine_similarity(
        d_W_py.float().flatten().unsqueeze(0),
        d_W_cuda.float().flatten().unsqueeze(0)
    ).item()
    
    d_x_cos = torch.nn.functional.cosine_similarity(
        d_x_py.float().flatten().unsqueeze(0),
        d_x_cuda.float().flatten().unsqueeze(0)
    ).item()

    # Normalized RMSE
    d_W_nrmse = (d_W_diff ** 2).mean().sqrt() / (d_W_py.float().abs().mean() + 1e-10)
    d_x_nrmse = (d_x_diff ** 2).mean().sqrt() / (d_x_py.float().abs().mean() + 1e-10)

    return {
        'status': 'ok',
        'out_max': out_diff.max().item(),
        'd_W_diff_max': d_W_diff.max().item(),
        'd_W_cos': d_W_cos,
        'd_W_nrmse': d_W_nrmse.item(),
        'd_x_diff_max': d_x_diff.max().item(),
        'd_x_cos': d_x_cos,
        'd_x_nrmse': d_x_nrmse.item(),
    }

if __name__ == '__main__':
    print("=" * 80)
    print("E74 Full Matrix Backward Pass Verification")
    print("=" * 80)
    
    configs = [
        (1, 1, 32, 32),
        (2, 1, 32, 32),
        (4, 1, 32, 32),
        (8, 1, 32, 32),
        (16, 1, 32, 32),
        (32, 1, 32, 32),
        (4, 4, 64, 48),
        (8, 2, 128, 64),
        (4, 2, 256, 96),
    ]
    
    all_pass = True
    for T, B, dim, n in configs:
        r = test_backward(T, B, dim, n, scale=0.1)
        if r['status'] != 'ok':
            status = f"[{r['status']}]"
            all_pass = False
            print(f"T={T:2d} B={B} dim={dim:3d} n={n:2d}: {status}")
        else:
            # Pass if cosine similarity > 0.99 and nrmse < 0.05
            ok = r['d_W_cos'] > 0.99 and r['d_x_cos'] > 0.99 and r['d_W_nrmse'] < 0.05 and r['d_x_nrmse'] < 0.05
            status = "PASS" if ok else "FAIL"
            if not ok:
                all_pass = False
            print(f"T={T:2d} B={B} dim={dim:3d} n={n:2d}: "
                  f"d_W cos={r['d_W_cos']:.4f} nrmse={r['d_W_nrmse']:.4f} | "
                  f"d_x cos={r['d_x_cos']:.4f} nrmse={r['d_x_nrmse']:.4f} [{status}]")
    
    print("=" * 80)
    if all_pass:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED")
