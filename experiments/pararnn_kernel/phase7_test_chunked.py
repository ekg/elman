"""Correctness + benchmark for chunked parallel Newton.

Chunked Newton splits T into C chunks that run in parallel. Critical
sequential depth = T/C + O(n_iters_needed_to_propagate), vs T for
the single-chunk approach. At T=128K, C=8 gives T/C=16K depth.
"""

import sys
import os
import time
import torch

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase7_fused_iter import (
    fused_newton_iter_inplace, newton_chunked,
)
from phase4_newton_driver import sequential_e88_forward


def setup(B, H, T, n, seed=0):
    g = torch.Generator(device='cuda').manual_seed(seed)
    dt = torch.float32
    S0 = 0.1 * torch.randn(B, H, n, n, generator=g, dtype=dt, device='cuda')
    K = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    V = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    decay = 0.9 + 0.1 * torch.rand(B, H, T, generator=g, dtype=dt, device='cuda')
    return S0, K, V, decay


def make_warmstart(S0, K, V, decay, perturb=0.02):
    """Run sequential forward, perturb — simulates previous step's converged S_var."""
    S_prev = sequential_e88_forward(S0, K, V, decay)
    S_init = S_prev[:, :, 1:].clone()
    del S_prev
    torch.cuda.empty_cache()
    T = S_init.shape[2]
    CHUNK = 8192
    for t_chunk in range(0, T, CHUNK):
        t_end = min(t_chunk + CHUNK, T)
        S_init[:, :, t_chunk:t_end].add_(
            perturb * torch.randn_like(S_init[:, :, t_chunk:t_end])
        )
    return S_init


def test_chunked_correctness(B, H, T, n, C_list):
    S0, K, V, decay = setup(B, H, T, n)
    S_seq = sequential_e88_forward(S0, K, V, decay)[:, :, 1:]

    print(f"  Shape: B={B} H={H} T={T} n={n}")
    S_init_base = make_warmstart(S0, K, V, decay)

    # Single-chunk (C=1) reference
    S_ref = S_init_base.clone()
    for _ in range(10):
        d = fused_newton_iter_inplace(S0, S_ref, K, V, decay)
        if d < 1e-5:
            break
    print(f"    single-chunk (C=1): converged to max|δ|={d:.2e}  "
          f"max|S_ref − S_seq|={(S_ref - S_seq).abs().max().item():.2e}")

    for C in C_list:
        if T % C != 0:
            continue
        S_var = S_init_base.clone()
        _, iters, dmax = newton_chunked(S0, K, V, decay, S_var, C=C, max_iters=30, tol=1e-5)
        diff_vs_seq = (S_var - S_seq).abs().max().item()
        diff_vs_ref = (S_var - S_ref).abs().max().item()
        status = "PASS" if diff_vs_ref < 1e-3 else "FAIL"
        print(f"    C={C:2d} T_chunk={T//C:6d}  iters={iters:2d}  dmax={dmax:.2e}  "
              f"|C-vs-C=1|={diff_vs_ref:.2e}  [{status}]")


def bench_chunked(B, H, T, n, C_list):
    S0, K, V, decay = setup(B, H, T, n)
    S_init = make_warmstart(S0, K, V, decay)

    # Single-chunk baseline — 3 iters
    def run_inplace(n_iter=3):
        for _ in range(n_iter):
            fused_newton_iter_inplace(S0, S_init, K, V, decay)

    for _ in range(2):
        run_inplace()
    torch.cuda.synchronize()
    t0 = time.time()
    run_inplace()
    torch.cuda.synchronize()
    ms_single = (time.time() - t0) * 1000

    print(f"  T={T} n={n}  single-chunk 3-iter: {ms_single:.1f} ms")

    for C in C_list:
        if T % C != 0:
            continue
        T_CHUNK = T // C
        boundary = torch.empty(B, H, C, n, n, dtype=S_init.dtype, device='cuda')

        # Warmup
        for _ in range(2):
            from phase7_fused_iter import _refresh_chunk_boundaries, fused_newton_iter_chunked
            for _ in range(3):
                _refresh_chunk_boundaries(S0, S_init, boundary, C, T_CHUNK)
                fused_newton_iter_chunked(S0, S_init, K, V, decay, boundary, C)
        torch.cuda.synchronize()

        from phase7_fused_iter import _refresh_chunk_boundaries, fused_newton_iter_chunked
        # Time 3 iters with boundary refresh
        t0 = time.time()
        for _ in range(3):
            _refresh_chunk_boundaries(S0, S_init, boundary, C, T_CHUNK)
            fused_newton_iter_chunked(S0, S_init, K, V, decay, boundary, C)
        torch.cuda.synchronize()
        ms_c = (time.time() - t0) * 1000
        print(f"    C={C:2d} T_chunk={T_CHUNK:6d}  3-iter: {ms_c:.1f} ms  "
              f"speedup vs single-chunk: {ms_single/ms_c:.2f}×")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--skip_test', action='store_true')
    args = p.parse_args()

    if not args.skip_test:
        print("Correctness tests:\n")
        for shape in [(1, 8, 1024, 32), (1, 32, 2048, 32), (1, 32, 8192, 32)]:
            test_chunked_correctness(*shape, C_list=[1, 2, 4, 8])
            torch.cuda.empty_cache()

    print("\n\nBenchmark chunked vs single-chunk (3 iters):\n")
    for B, H, T, n in [(1, 32, 8192, 32),
                       (1, 32, 32768, 32),
                       (1, 32, 131072, 32)]:
        bench_chunked(B, H, T, n, C_list=[1, 2, 4, 8, 16, 32])
        torch.cuda.empty_cache()
