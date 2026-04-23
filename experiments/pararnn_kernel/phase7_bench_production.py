"""Benchmark at production E88 configs.

Production shapes (from CLAUDE.md, 480M models):
- E88 n32: H=83, n=32, depth=17
- E88 n16: H=141, n=16

Normal training T=512. Long-context training at 32K. Exploratory at 128K.
Test: at what scales and configs does Pararnn actually help?
"""

import sys, os, time
import torch
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase7_fused_iter import fused_newton_iter_inplace
from phase4_newton_driver import sequential_e88_forward
from elman.models.e88_fla_hybrid import E88FLAHybridCUDAFunction


def bench_cuda(B, H, T, n, n_repeat=5):
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


def bench_pararnn(B, H, T, n, n_iter, n_repeat=5):
    g = torch.Generator(device='cuda').manual_seed(0)
    dt = torch.float32
    S0 = 0.1 * torch.randn(B, H, n, n, generator=g, dtype=dt, device='cuda')
    K = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    V = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    decay = 0.9 + 0.1 * torch.rand(B, H, T, generator=g, dtype=dt, device='cuda')

    # Realistic warm-start: perturbed converged trajectory
    S_prev = sequential_e88_forward(S0, K, V, decay)
    S_init = S_prev[:, :, 1:].clone()
    del S_prev
    torch.cuda.empty_cache()
    CHUNK = 8192
    for t_chunk in range(0, T, CHUNK):
        t_end = min(t_chunk + CHUNK, T)
        S_init[:, :, t_chunk:t_end].add_(
            0.01 * torch.randn_like(S_init[:, :, t_chunk:t_end])
        )

    def run():
        for _ in range(n_iter):
            fused_newton_iter_inplace(S0, S_init, K, V, decay)

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat):
        run()
    torch.cuda.synchronize()
    return (time.time() - t0) / n_repeat * 1000


if __name__ == '__main__':
    # Production configs × realistic T values
    configs = [
        ("E88-n32 480M", 83, 32),
        ("E88-n16 480M", 141, 16),
        ("E88-n32 100M", 32, 32),   # for reference
    ]
    T_values = [512, 2048, 8192, 32768, 131072]

    for name, H, n in configs:
        print(f"\n=== {name}: H={H}, n={n} ===")
        print(f"{'T':>7s}  {'CUDA':>10s}  {'Par-1it':>10s}  {'spd-1':>6s}  "
              f"{'Par-2it':>10s}  {'spd-2':>6s}  "
              f"{'Par-3it':>10s}  {'spd-3':>6s}")
        for T in T_values:
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                cuda_ms = bench_cuda(1, H, T, n)
                torch.cuda.empty_cache()
                p1 = bench_pararnn(1, H, T, n, 1)
                torch.cuda.empty_cache()
                p2 = bench_pararnn(1, H, T, n, 2)
                torch.cuda.empty_cache()
                p3 = bench_pararnn(1, H, T, n, 3)
                print(f"{T:>7d}  {cuda_ms:>7.2f} ms  "
                      f"{p1:>7.2f} ms  {cuda_ms/p1:>5.2f}×  "
                      f"{p2:>7.2f} ms  {cuda_ms/p2:>5.2f}×  "
                      f"{p3:>7.2f} ms  {cuda_ms/p3:>5.2f}×")
            except Exception as e:
                print(f"{T:>7d}  FAIL: {str(e)[:80]}")
                torch.cuda.empty_cache()
