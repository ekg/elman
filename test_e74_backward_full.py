#!/usr/bin/env python3
"""
Full backward test with smaller scale inputs (to avoid PyTorch NaN).
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

    # Compare
    out_diff = (output.float() - output_cuda.float()).abs()
    d_W_diff = (d_W_py.float() - d_W_cuda.float()).abs()
    d_x_diff = (d_x_py.float() - d_x_cuda.float()).abs()

    d_W_rel = d_W_diff / (d_W_py.float().abs() + 1e-6)
    d_x_rel = d_x_diff / (d_x_py.float().abs() + 1e-6)

    return {
        'status': 'ok',
        'out_max': out_diff.max().item(),
        'out_mean': out_diff.mean().item(),
        'd_W_max': d_W_diff.max().item(),
        'd_W_rel_max': d_W_rel.max().item(),
        'd_W_rel_mean': d_W_rel.mean().item(),
        'd_x_max': d_x_diff.max().item(),
        'd_x_rel_max': d_x_rel.max().item(),
        'd_x_rel_mean': d_x_rel.mean().item(),
    }

if __name__ == '__main__':
    print("=" * 70)
    print("E74 Full Matrix Backward Pass Verification (scale=0.1)")
    print("=" * 70)
    
    configs = [
        (1, 1, 32, 32),    # T=1, small
        (2, 1, 32, 32),    # T=2, small
        (4, 1, 32, 32),    # T=4, small
        (8, 1, 32, 32),    # T=8, small
        (16, 1, 32, 32),   # T=16
        (32, 1, 32, 32),   # T=32 (covers 2 checkpoints)
        (4, 4, 64, 48),    # Larger batch
        (8, 2, 128, 64),   # Larger dim
        (4, 2, 256, 96),   # Maximum n_state
    ]
    
    all_pass = True
    for T, B, dim, n in configs:
        r = test_backward(T, B, dim, n, scale=0.1)
        if r['status'] != 'ok':
            status = f"[{r['status']}]"
            all_pass = False
            print(f"T={T:2d} B={B} dim={dim:3d} n={n:2d}: {status}")
        else:
            ok = r['d_W_rel_max'] < 0.05 and r['d_x_rel_max'] < 0.05
            status = "PASS" if ok else "FAIL"
            if not ok:
                all_pass = False
            print(f"T={T:2d} B={B} dim={dim:3d} n={n:2d}: out={r['out_max']:.4f} "
                  f"d_W_rel={r['d_W_rel_max']:.4f}/{r['d_W_rel_mean']:.4f} "
                  f"d_x_rel={r['d_x_rel_max']:.4f}/{r['d_x_rel_mean']:.4f} [{status}]")
    
    print("=" * 70)
    if all_pass:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED")
