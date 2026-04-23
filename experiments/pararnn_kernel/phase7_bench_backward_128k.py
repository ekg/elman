"""Push backward to T=128K at production configs, with per-config tuning."""

import sys, os, time
import torch
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase7_fused_backward import backward_e88_fused
from phase4_newton_driver import sequential_e88_forward
from elman.models.e88_fla_hybrid import E88FLAHybridCUDAFunction


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


def bench_par_bwd_tuned(B, H, T, n, n_repeat=3):
    """Search num_warps / num_stages, return best."""
    g = torch.Generator(device='cuda').manual_seed(0)
    dt = torch.float32
    S0 = 0.1 * torch.randn(B, H, n, n, generator=g, dtype=dt, device='cuda')
    K = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    V = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    decay = 0.9 + 0.1 * torch.rand(B, H, T, generator=g, dtype=dt, device='cuda')
    S_traj = sequential_e88_forward(S0, K, V, decay)
    dL_dS_traj = 0.01 * torch.randn_like(S_traj)

    best = (float('inf'), None, None)
    peak_gb = 0.0
    for nw in [1, 2, 4, 8]:
        for ns in [1, 2, 3]:
            try:
                def run():
                    return backward_e88_fused(S_traj, K, V, decay, dL_dS_traj,
                                                num_warps=nw, num_stages=ns)
                for _ in range(3): run()
                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(n_repeat): _ = run()
                torch.cuda.synchronize()
                ms = (time.time() - t0) / n_repeat * 1000
                peak_gb = max(peak_gb, torch.cuda.max_memory_allocated() / 1024**3)
                if ms < best[0]:
                    best = (ms, nw, ns)
            except Exception:
                continue
    return best[0], best[1], best[2], peak_gb


if __name__ == '__main__':
    configs = [
        ("E88-n32 480M", 83, 32),
        ("E88-n16 480M", 141, 16),
        ("Small H=32",    32, 32),
    ]

    for name, H, n in configs:
        print(f"\n=== {name}: H={H}, n={n} ===")
        print(f"{'T':>7s}  {'CUDA fwd':>9s}  {'CUDA f+b':>9s}  "
              f"{'Par bwd (nw,ns)':>20s}  {'Hybrid tot':>11s}  {'e2e spd':>8s}")
        for T in [8192, 32768, 65536, 131072]:
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                cuda_fwd = bench_cuda_fwd_only(1, H, T, n)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                cuda_fb = bench_cuda_fwd_bwd(1, H, T, n)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                par_bwd, nw, ns, peak = bench_par_bwd_tuned(1, H, T, n)
                hybrid = cuda_fwd + par_bwd
                e2e_spd = cuda_fb / hybrid
                print(f"{T:>7d}  {cuda_fwd:>6.1f} ms  {cuda_fb:>6.1f} ms  "
                      f"{par_bwd:>6.1f} ms (nw={nw},ns={ns})  "
                      f"{hybrid:>8.1f} ms  {e2e_spd:>6.2f}×  [peak {peak:.1f}GB]")
            except torch.cuda.OutOfMemoryError as e:
                print(f"{T:>7d}  OOM: {str(e)[:80]}")
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"{T:>7d}  FAIL: {str(e)[:80]}")
                torch.cuda.empty_cache()
