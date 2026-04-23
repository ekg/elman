"""Direct benchmark: 1-iter Pararnn vs CUDA across T, at H=32.

Warm-start represents previous-step's converged S_var + small perturbation,
which is the realistic training-time scenario. At perturb=0.001 (typical
training step effect) → 1 iter reaches fp32 precision.
"""

import sys, os, time
import torch
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase7_fused_iter import fused_newton_iter_inplace
from phase4_newton_driver import sequential_e88_forward
from elman.models.e88_fla_hybrid import E88FLAHybridCUDAFunction


def bench_pararnn(B, H, T, n, n_iter=1, perturb=0.001, n_repeat=5):
    """1-iter Pararnn with realistic warm-start (perturbed converged trajectory)."""
    g = torch.Generator(device='cuda').manual_seed(0)
    dt = torch.float32
    S0 = 0.1 * torch.randn(B, H, n, n, generator=g, dtype=dt, device='cuda')
    K = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    V = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    decay = 0.9 + 0.1 * torch.rand(B, H, T, generator=g, dtype=dt, device='cuda')

    S_prev = sequential_e88_forward(S0, K, V, decay)
    S_init = S_prev[:, :, 1:].clone()
    del S_prev
    torch.cuda.empty_cache()
    CHUNK = 8192
    for t_chunk in range(0, T, CHUNK):
        t_end = min(t_chunk + CHUNK, T)
        S_init[:, :, t_chunk:t_end].add_(
            perturb * torch.randn_like(S_init[:, :, t_chunk:t_end])
        )

    def run():
        d = 0.0
        for _ in range(n_iter):
            d = fused_newton_iter_inplace(S0, S_init, K, V, decay)
        return d

    # warmup
    for _ in range(3):
        run()
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(n_repeat):
        run()
    torch.cuda.synchronize()
    ms = (time.time() - t0) / n_repeat * 1000

    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    return ms, peak_gb


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
    ms = (time.time() - t0) / n_repeat * 1000
    return ms, torch.cuda.max_memory_allocated() / 1024**3


if __name__ == '__main__':
    print(f"\n{'shape':>15s}  {'CUDA (bf16)':>12s}  "
          f"{'Par 1-iter':>12s}  {'Par 2-iter':>12s}  {'Par 3-iter':>12s}  "
          f"{'speedup(1it)':>12s}")
    print("-" * 90)
    for B, H, T, n in [(1, 32, 8192, 32),
                       (1, 32, 16384, 32),
                       (1, 32, 32768, 32),
                       (1, 32, 65536, 32),
                       (1, 32, 131072, 32)]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        cuda_ms, _ = bench_cuda(B, H, T, n)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        p1, _ = bench_pararnn(B, H, T, n, n_iter=1)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        p2, _ = bench_pararnn(B, H, T, n, n_iter=2)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        p3, _ = bench_pararnn(B, H, T, n, n_iter=3)
        shape = f"(1,32,{T},32)"
        print(f"{shape:>15s}  {cuda_ms:>9.1f} ms  "
              f"{p1:>9.1f} ms  {p2:>9.1f} ms  {p3:>9.1f} ms  "
              f"{cuda_ms/p1:>10.2f}×")
