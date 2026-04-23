"""Benchmark: Pararnn warm-start (3 Newton iters, in-place) vs CUDA forward.

At T=128K — the context length where parallel methods are supposed to win.
"""

import sys
import os
import time
import torch

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase7_fused_iter import fused_newton_iter_inplace
from phase4_newton_driver import sequential_e88_forward
from elman.models.e88_fla_hybrid import E88FLAHybridCUDAFunction


def bench_pararnn_inplace(B, H, T, n, n_warm_iters=3, n_repeat=3):
    """Pararnn warm-start: 3 Newton iters in-place on S_var, no δ buffer."""
    g = torch.Generator(device='cuda').manual_seed(0)
    dt = torch.float32
    S0 = 0.1 * torch.randn(B, H, n, n, generator=g, dtype=dt, device='cuda')
    K = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    V = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    decay = 0.9 + 0.1 * torch.rand(B, H, T, generator=g, dtype=dt, device='cuda')

    # Simulate warm-start trajectory (in real training this is the previous
    # step's converged S_var — no sequential forward needed).
    S_prev = sequential_e88_forward(S0, K, V, decay)
    S_init = S_prev[:, :, 1:].clone()
    del S_prev
    torch.cuda.empty_cache()
    CHUNK = 8192
    for t_chunk in range(0, T, CHUNK):
        t_end = min(t_chunk + CHUNK, T)
        S_init[:, :, t_chunk:t_end].add_(
            0.02 * torch.randn_like(S_init[:, :, t_chunk:t_end])
        )

    def run():
        # Don't mutate S_init across repeats — use a working copy.
        # For a realistic benchmark, the working copy IS S_init itself,
        # but for repeatable timing we need to reset. Since S_init is
        # already close to the fixed point, just re-use it.
        for _ in range(n_warm_iters):
            _ = fused_newton_iter_inplace(S0, S_init, K, V, decay)

    # Warmup
    for _ in range(2):
        run()
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(n_repeat):
        run()
    torch.cuda.synchronize()
    ms = (time.time() - t0) / n_repeat * 1000

    peak = torch.cuda.max_memory_allocated() / 1024**3
    return ms, peak


def bench_cuda_forward(B, H, T, n, n_repeat=3):
    """Existing E88 CUDA sequential kernel forward (bf16), training=True."""
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

    for _ in range(2):
        run()
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(n_repeat):
        _ = run()
    torch.cuda.synchronize()
    ms = (time.time() - t0) / n_repeat * 1000

    peak = torch.cuda.max_memory_allocated() / 1024**3
    return ms, peak


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--T', type=int, default=0)
    p.add_argument('--path', type=str, default='both', choices=['cuda', 'pararnn', 'both'])
    args = p.parse_args()

    shapes = [(1, 32, args.T, 32)] if args.T > 0 else [
        (1, 32, T, 32) for T in (8192, 16384, 32768, 65536, 131072)
    ]

    print(f"\n{'B':>3s} {'H':>3s} {'T':>6s} {'n':>3s}  "
          f"{'CUDA (bf16) fwd':>16s}  {'Pararnn warm (3iter, fp32)':>28s}  "
          f"{'speedup':>10s}")
    print("-" * 82)
    for B, H, T, n in shapes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        cuda_ms = float('nan')
        par_ms = float('nan')
        cuda_gb = par_gb = float('nan')
        if args.path in ('cuda', 'both'):
            try:
                cuda_ms, cuda_gb = bench_cuda_forward(B, H, T, n)
            except Exception as e:
                print(f"  CUDA failed at T={T}: {str(e)[:120]}")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        if args.path in ('pararnn', 'both'):
            try:
                par_ms, par_gb = bench_pararnn_inplace(B, H, T, n)
            except Exception as e:
                print(f"  Pararnn failed at T={T}: {str(e)[:120]}")
        speedup = cuda_ms / par_ms if par_ms == par_ms and cuda_ms == cuda_ms else float('nan')
        print(f"{B:>3d} {H:>3d} {T:>6d} {n:>3d}  "
              f"{cuda_ms:>10.1f} ms {cuda_gb:>3.0f}GB  "
              f"{par_ms:>14.1f} ms {par_gb:>3.0f}GB         "
              f"{speedup:>7.2f}×")
