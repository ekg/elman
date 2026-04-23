"""Benchmark fused Pararnn backward at production configs vs CUDA fwd+bwd.

Real training step = forward + backward. CUDA does both. Hybrid
approach: CUDA forward (fast sequential) + Pararnn backward (parallel,
linear → 1 exact pass).
"""

import sys, os, time
import torch
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase7_fused_backward import backward_e88_fused
from phase4_newton_driver import sequential_e88_forward
from elman.models.e88_fla_hybrid import E88FLAHybridCUDAFunction


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
    for _ in range(3): run()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat): _ = run()
    torch.cuda.synchronize()
    return (time.time() - t0) / n_repeat * 1000


def bench_cuda_fwd_bwd(B, H, T, n, n_repeat=3):
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
        k.grad = None; v.grad = None; q.grad = None; decay.grad = None
    for _ in range(3): run()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat): run()
    torch.cuda.synchronize()
    return (time.time() - t0) / n_repeat * 1000


def bench_pararnn_bwd(B, H, T, n, n_repeat=3, num_warps=4, num_stages=1):
    g = torch.Generator(device='cuda').manual_seed(0)
    dt = torch.float32
    S0 = 0.1 * torch.randn(B, H, n, n, generator=g, dtype=dt, device='cuda')
    K = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    V = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    decay = 0.9 + 0.1 * torch.rand(B, H, T, generator=g, dtype=dt, device='cuda')
    S_traj = sequential_e88_forward(S0, K, V, decay)
    dL_dS_traj = 0.01 * torch.randn_like(S_traj)

    def run():
        return backward_e88_fused(S_traj, K, V, decay, dL_dS_traj,
                                    num_warps=num_warps, num_stages=num_stages)
    for _ in range(3): run()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat): _ = run()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1024**3
    return (time.time() - t0) / n_repeat * 1000, peak


if __name__ == '__main__':
    configs = [("E88-n32 480M", 83, 32), ("E88-n16 480M", 141, 16), ("Small H=32", 32, 32)]
    T_vals = [512, 2048, 8192, 32768]  # T=128K will OOM with S_traj+dL_dS_traj on some configs

    for name, H, n in configs:
        print(f"\n=== {name}: H={H}, n={n} ===")
        print(f"{'T':>7s}  {'CUDA fwd':>10s}  {'CUDA f+b':>10s}  {'CUDA bwd':>10s}  "
              f"{'Par bwd':>10s} {'(GB)':>6s}  {'bwd spd':>8s}")
        for T in T_vals:
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                cuda_fwd = bench_cuda_fwd_only(1, H, T, n)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                cuda_fb = bench_cuda_fwd_bwd(1, H, T, n)
                cuda_bwd = cuda_fb - cuda_fwd
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                par_bwd, peak = bench_pararnn_bwd(1, H, T, n)
                spd = cuda_bwd / par_bwd
                print(f"{T:>7d}  {cuda_fwd:>7.2f} ms  {cuda_fb:>7.2f} ms  "
                      f"{cuda_bwd:>7.2f} ms  "
                      f"{par_bwd:>7.2f} ms ({peak:>4.1f})  {spd:>6.2f}×")
            except Exception as e:
                print(f"{T:>7d}  FAIL: {str(e)[:80]}")
                torch.cuda.empty_cache()
