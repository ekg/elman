"""Phase 7 — warm-start benchmark: Pararnn vs CUDA at realistic conditions.

In a real training loop, each training step's Newton solve starts from
the previous step's converged trajectory (plus small perturbation from
changed inputs). We simulate this with a 2% perturbation.

Fair comparison with CUDA (bf16 sequential) on single-GPU at various T.
"""

import sys
import os
import time

import torch

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, '/home/erikg/elman/elman/cuda')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase7_fused_iter import fused_newton_iter
from phase4_newton_driver import sequential_e88_forward

from elman.models.e88_fla_hybrid import E88FLAHybridCUDAFunction


def bench_pararnn_warmstart(B, H, T, n, N_WARM_ITERS=3, n_repeat=5):
    """Benchmark Pararnn-fused-Newton with warm-start (fixed 3 iters)."""
    g = torch.Generator(device='cuda').manual_seed(0)
    dt = torch.float32
    S0 = 0.1 * torch.randn(B, H, n, n, generator=g, dtype=dt, device='cuda')
    K = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    V = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    decay = 0.9 + 0.1 * torch.rand(B, H, T, generator=g, dtype=dt, device='cuda')

    # Pre-compute the "previous step's" trajectory — our warm start
    S_prev = sequential_e88_forward(S0, K, V, decay)
    S_init = S_prev[:, :, 1:] + 0.02 * torch.randn_like(S_prev[:, :, 1:])

    def run():
        S_var = S_init.clone()
        for _ in range(N_WARM_ITERS):
            delta = fused_newton_iter(S0, S_var, K, V, decay)
            S_var = S_var + delta
        return S_var

    # Warmup
    for _ in range(3):
        run()
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(n_repeat):
        _ = run()
    torch.cuda.synchronize()
    return (time.time() - t0) / n_repeat * 1000


def bench_cuda_forward_only(B, H, T, n, n_repeat=5):
    """Existing E88 CUDA kernel, bf16, forward ONLY (training=True because
    training=False triggers an illegal memory access in the kernel)."""
    g = torch.Generator(device='cuda').manual_seed(0)
    dt = torch.bfloat16
    k = (0.3 * torch.randn(T, B, H, n, generator=g, dtype=dt, device='cuda')).requires_grad_(True)
    v = (0.3 * torch.randn(T, B, H, n, generator=g, dtype=dt, device='cuda')).requires_grad_(True)
    q = (0.3 * torch.randn(T, B, H, n, generator=g, dtype=dt, device='cuda')).requires_grad_(True)
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, generator=g, dtype=dt, device='cuda')
                          ).detach().requires_grad_(True)
    S0 = 0.1 * torch.randn(B, H, n, n, generator=g, dtype=dt, device='cuda')

    def run():
        return E88FLAHybridCUDAFunction.apply(True, k, v, q, decay, S0, H)

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat):
        _ = run()
    torch.cuda.synchronize()
    return (time.time() - t0) / n_repeat * 1000


def bench_cuda(B, H, T, n, n_repeat=5):
    """Compatibility alias — forward only."""
    return bench_cuda_forward_only(B, H, T, n, n_repeat)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--T', type=int, default=0, help='if >0, run only this T')
    parser.add_argument('--path', type=str, default='both', choices=['cuda', 'pararnn', 'both'])
    parser.add_argument('--H', type=int, default=32)
    parser.add_argument('--B', type=int, default=1)
    parser.add_argument('--n', type=int, default=32)
    args = parser.parse_args()

    if args.T > 0:
        shapes = [(args.B, args.H, args.T, args.n)]
    else:
        shapes = [(1, 32, T, 32) for T in (512, 1024, 2048, 4096, 8192, 16384)]

    print(f"{'B':>3s} {'H':>3s} {'T':>6s} {'n':>3s}  "
          f"{'CUDA (bf16)':>14s}  {'Pararnn warm (fp32, 3 iter)':>30s}  {'speedup':>10s}")
    print("-" * 80)
    for B, H, T, n in shapes:
        cuda_ms = float('nan')
        par_ms = float('nan')
        if args.path in ('cuda', 'both'):
            cuda_ms = bench_cuda(B, H, T, n)
        if args.path in ('pararnn', 'both'):
            par_ms = bench_pararnn_warmstart(B, H, T, n)
        print(f"{B:>3d} {H:>3d} {T:>6d} {n:>3d}  "
              f"{cuda_ms:>11.2f} ms  {par_ms:>27.2f} ms  "
              f"{(cuda_ms/par_ms if par_ms == par_ms and cuda_ms == cuda_ms else float('nan')):>8.2f}×")
