"""Where does it shine? Extreme T at n=16 (fits best in memory).

Test T = 32K, 128K, 256K, 512K, 1M if memory allows. See if the
backward-hybrid speedup grows or stays flat.
"""

import sys, os, time
import torch
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase7_fused_backward import backward_e88_fused_rank1
from phase4_newton_driver import sequential_e88_forward
from elman.models.e88_fla_hybrid import E88FLAHybridCUDAFunction


def bench_cuda_fwd(B, H, T, n, n_repeat=3):
    g = torch.Generator(device='cuda').manual_seed(0)
    dt = torch.bfloat16
    k = (0.3 * torch.randn(T, B, H, n, generator=g, dtype=dt, device='cuda')).requires_grad_(True)
    v = (0.3 * torch.randn(T, B, H, n, generator=g, dtype=dt, device='cuda')).requires_grad_(True)
    q = (0.3 * torch.randn(T, B, H, n, generator=g, dtype=dt, device='cuda')).requires_grad_(True)
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, generator=g, dtype=dt, device='cuda')
                          ).detach().requires_grad_(True)
    S0 = 0.1 * torch.randn(B, H, n, n, generator=g, dtype=dt, device='cuda')
    def run(): return E88FLAHybridCUDAFunction.apply(True, k, v, q, decay, S0, H)
    for _ in range(2): run()
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
    for _ in range(2): run()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat): run()
    torch.cuda.synchronize()
    return (time.time() - t0) / n_repeat * 1000


def bench_pararnn_bwd(B, H, T, n, n_repeat=3):
    g = torch.Generator(device='cuda').manual_seed(0)
    dt = torch.bfloat16
    S0 = 0.1 * torch.randn(B, H, n, n, generator=g, dtype=torch.float32, device='cuda')
    K = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=torch.float32, device='cuda')
    V = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=torch.float32, device='cuda')
    decay = 0.9 + 0.1 * torch.rand(B, H, T, generator=g, dtype=torch.float32, device='cuda')
    S_traj = sequential_e88_forward(S0, K, V, decay).to(dt)
    del S0, K, V, decay
    torch.cuda.empty_cache()

    K_d = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    V_d = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    decay_d = (0.9 + 0.1 * torch.rand(B, H, T, generator=g, dtype=dt, device='cuda'))
    g_T = 0.01 * torch.randn(B, H, n, n, dtype=dt, device='cuda')
    dL_dout = 0.01 * torch.randn(B, H, T, n, dtype=dt, device='cuda')
    q = 0.3 * torch.randn(B, H, T, n, dtype=dt, device='cuda')

    def run():
        d = backward_e88_fused_rank1(S_traj, K_d, V_d, decay_d, g_T, dL_dout, q,
                                      num_warps=1 if n == 16 else 4, num_stages=1)
        dQ = torch.einsum('bhti,bhtij->bhtj', dL_dout, S_traj[:, :, 1:])
        return d, dQ
    for _ in range(2): run()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat): _ = run()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1024**3
    return (time.time() - t0) / n_repeat * 1000, peak


if __name__ == '__main__':
    # Long T sweep at n=16 (fits best in memory)
    for H in [141, 32]:
        print(f"\n=== H={H}, n=16 ===")
        print(f"{'T':>8s}  {'CUDA f+b':>10s}  {'CUDA fwd':>9s}  {'CUDA bwd':>9s}  "
              f"{'Par bwd+dQ':>11s}  {'Hybrid':>8s}  {'e2e spd':>8s}  {'ceiling':>8s}")
        for T in [32768, 131072, 262144, 524288, 1048576]:
            c_fwd = c_fb = p_bwd = float('nan'); peak = 0
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                c_fwd = bench_cuda_fwd(1, H, T, 16)
            except Exception as e:
                print(f"  CUDA fwd T={T} FAIL: {str(e)[:80]}")
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                c_fb = bench_cuda_fwd_bwd(1, H, T, 16)
            except Exception as e:
                print(f"  CUDA f+b T={T} FAIL: {str(e)[:80]}")
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                p_bwd, peak = bench_pararnn_bwd(1, H, T, 16)
            except Exception as e:
                print(f"  Par bwd T={T} FAIL: {str(e)[:80]}")

            c_bwd = c_fb - c_fwd if (c_fb == c_fb and c_fwd == c_fwd) else float('nan')
            hybrid = c_fwd + p_bwd if (c_fwd == c_fwd and p_bwd == p_bwd) else float('nan')
            spd = c_fb / hybrid if (c_fb == c_fb and hybrid == hybrid) else float('nan')
            ceiling = c_fb / c_fwd if (c_fb == c_fb and c_fwd == c_fwd) else float('nan')
            print(f"{T:>8d}  {c_fb:>7.1f} ms  {c_fwd:>6.1f} ms  "
                  f"{c_bwd:>6.1f} ms  {p_bwd:>8.1f} ms  {hybrid:>5.1f} ms  "
                  f"{spd:>6.2f}×  {ceiling:>6.2f}×  [peak {peak:.1f}GB]")
