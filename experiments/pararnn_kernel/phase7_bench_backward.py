"""Benchmark: Pararnn backward vs CUDA backward.

Pararnn backward is a LINEAR reverse scan — it's exact in 1 pass (no
Newton needed because the adjoint is linear given the converged
forward). CUDA backward is ALSO a sequential reverse scan, but over
32 blocks vs Pararnn's 1024 programs.
"""

import sys, os, time
import torch
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase5_backward import backward_e88_triton
from phase4_newton_driver import sequential_e88_forward
from elman.models.e88_fla_hybrid import E88FLAHybridCUDAFunction


def bench_pararnn_backward(B, H, T, n, n_repeat=3):
    g = torch.Generator(device='cuda').manual_seed(0)
    dt = torch.float32
    S0 = 0.1 * torch.randn(B, H, n, n, generator=g, dtype=dt, device='cuda')
    K = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    V = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    decay = 0.9 + 0.1 * torch.rand(B, H, T, generator=g, dtype=dt, device='cuda')

    # Forward to get S_traj
    S_traj = sequential_e88_forward(S0, K, V, decay)
    # Fake loss gradient
    dL_dS_traj = torch.randn_like(S_traj) * 0.01

    def run():
        return backward_e88_triton(S_traj, K, V, decay, dL_dS_traj)

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


def bench_cuda_fwd_bwd(B, H, T, n, n_repeat=3):
    """Measure CUDA forward+backward combined (standard training step)."""
    g = torch.Generator(device='cuda').manual_seed(0)
    dt = torch.bfloat16
    k = (0.3 * torch.randn(T, B, H, n, generator=g, dtype=dt, device='cuda')).requires_grad_(True)
    v = (0.3 * torch.randn(T, B, H, n, generator=g, dtype=dt, device='cuda')).requires_grad_(True)
    q = (0.3 * torch.randn(T, B, H, n, generator=g, dtype=dt, device='cuda')).requires_grad_(True)
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, generator=g, dtype=dt, device='cuda')
                          ).detach().requires_grad_(True)
    S0 = 0.1 * torch.randn(B, H, n, n, generator=g, dtype=dt, device='cuda')

    def run():
        S_final, output = E88FLAHybridCUDAFunction.apply(True, k, v, q, decay, S0, H)
        loss = output.sum() + S_final.pow(2).sum() * 1e-4
        loss.backward()
        k.grad = None
        v.grad = None
        q.grad = None
        decay.grad = None

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


def bench_cuda_fwd_only(B, H, T, n, n_repeat=3):
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
    return ms


if __name__ == '__main__':
    print(f"\n{'shape':>16s}  {'CUDA fwd':>10s}  {'CUDA f+b':>10s}  "
          f"{'CUDA bwd*':>10s}  {'Par bwd':>10s}  "
          f"{'Par-bwd speedup':>16s}")
    print("-" * 90)
    for B, H, T, n in [(1, 32, 8192, 32),
                       (1, 32, 16384, 32),
                       (1, 32, 32768, 32)]:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            cuda_fwd = bench_cuda_fwd_only(B, H, T, n)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            cuda_fb, _ = bench_cuda_fwd_bwd(B, H, T, n)
            cuda_bwd = cuda_fb - cuda_fwd  # estimate (assumes fwd+bwd split)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            par_bwd, par_peak = bench_pararnn_backward(B, H, T, n)
            shape = f"(1,32,{T},32)"
            print(f"{shape:>16s}  {cuda_fwd:>7.1f} ms  {cuda_fb:>7.1f} ms  "
                  f"{cuda_bwd:>7.1f} ms  {par_bwd:>7.1f} ms ({par_peak:.0f}GB)  "
                  f"{cuda_bwd/par_bwd:>14.2f}×")
        except Exception as e:
            print(f"  FAIL T={T}: {str(e)[:120]}")
            torch.cuda.empty_cache()
