"""Benchmark rank-1 backward vs rank-1-materialized backward.

In real training, dL/dS[t] = outer(dL/doutput[t], q[t]). Passing rank-1
factors instead of the N×N tensor saves 16 GB of HBM at T=128K and cuts
memory traffic per step.
"""

import sys, os, time
import torch
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase7_fused_backward import backward_e88_fused, backward_e88_fused_rank1
from phase4_newton_driver import sequential_e88_forward


def test_rank1_matches_materialized(B, H, T, n, seed=0):
    """Verify rank-1 kernel gives same result as materialized."""
    torch.manual_seed(seed)
    dt = torch.float32
    S0 = 0.1 * torch.randn(B, H, n, n, dtype=dt, device='cuda')
    K = 0.3 * torch.randn(B, H, T, n, dtype=dt, device='cuda')
    V = 0.3 * torch.randn(B, H, T, n, dtype=dt, device='cuda')
    decay = 0.9 + 0.1 * torch.rand(B, H, T, dtype=dt, device='cuda')

    g_T = 0.1 * torch.randn(B, H, n, n, dtype=dt, device='cuda')
    dL_dout = 0.1 * torch.randn(B, H, T, n, dtype=dt, device='cuda')
    q = 0.3 * torch.randn(B, H, T, n, dtype=dt, device='cuda')

    # Materialized dL_dS_traj: [B, H, T+1, N, N]
    # At t=T, we have g_T. At t=0..T-1, outer(dL_dout, q).
    dL_dS_traj = torch.zeros(B, H, T + 1, n, n, dtype=dt, device='cuda')
    dL_dS_traj[:, :, :T] = torch.einsum('bhti,bhtj->bhtij', dL_dout, q)
    dL_dS_traj[:, :, T] = g_T

    S_traj = sequential_e88_forward(S0, K, V, decay)

    dS0_m, dK_m, dV_m, ddec_m = backward_e88_fused(S_traj, K, V, decay, dL_dS_traj)
    dS0_r, dK_r, dV_r, ddec_r = backward_e88_fused_rank1(S_traj, K, V, decay, g_T, dL_dout, q)

    def rel_err(a, b):
        return (a - b).abs().max().item() / max(b.abs().max().item(), 1e-30)

    eS0 = rel_err(dS0_r, dS0_m)
    eK  = rel_err(dK_r,  dK_m)
    eV  = rel_err(dV_r,  dV_m)
    ed  = rel_err(ddec_r, ddec_m)
    worst = max(eS0, eK, eV, ed)
    status = "PASS" if worst < 1e-5 else "FAIL"
    print(f"  B={B} H={H:3d} T={T:4d} n={n}  "
          f"rel vs materialized: S0={eS0:.1e} K={eK:.1e} V={eV:.1e} dec={ed:.1e}  [{status}]")


def bench(B, H, T, n, use_rank1, num_warps=4, num_stages=1, n_repeat=3):
    g = torch.Generator(device='cuda').manual_seed(0)
    dt = torch.float32
    S0 = 0.1 * torch.randn(B, H, n, n, generator=g, dtype=dt, device='cuda')
    K = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    V = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    decay = 0.9 + 0.1 * torch.rand(B, H, T, generator=g, dtype=dt, device='cuda')
    S_traj = sequential_e88_forward(S0, K, V, decay)

    if use_rank1:
        g_T = 0.01 * torch.randn(B, H, n, n, dtype=dt, device='cuda')
        dL_dout = 0.01 * torch.randn(B, H, T, n, dtype=dt, device='cuda')
        q = 0.3 * torch.randn(B, H, T, n, dtype=dt, device='cuda')
        def run():
            return backward_e88_fused_rank1(S_traj, K, V, decay, g_T, dL_dout, q,
                                             num_warps=num_warps, num_stages=num_stages)
    else:
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
    print("Correctness (rank-1 == materialized):\n")
    for shape in [(1, 4, 128, 16), (1, 16, 512, 32), (1, 32, 2048, 32), (2, 8, 512, 32)]:
        test_rank1_matches_materialized(*shape)

    print("\nBenchmark rank-1 vs materialized backward:\n")
    print(f"{'shape':>20s}  {'Mat bwd':>10s}  {'Rank-1 bwd':>12s}  {'spd':>6s}  {'mem mat':>8s}  {'mem r1':>8s}")
    for B, H, T, n in [(1, 83, 2048, 32),
                       (1, 83, 8192, 32),
                       (1, 83, 32768, 32),
                       (1, 83, 65536, 32),
                       (1, 141, 8192, 16),
                       (1, 141, 32768, 16),
                       (1, 141, 65536, 16),
                       (1, 141, 131072, 16),
                       (1, 83, 131072, 32)]:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            mat_ms, mat_peak = bench(B, H, T, n, use_rank1=False)
        except Exception as e:
            mat_ms, mat_peak = float('nan'), float('nan')
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            r1_ms, r1_peak = bench(B, H, T, n, use_rank1=True)
        except Exception as e:
            r1_ms, r1_peak = float('nan'), float('nan')
        shape = f"(B={B},H={H},T={T},n={n})"
        spd = mat_ms / r1_ms if mat_ms == mat_ms and r1_ms == r1_ms else float('nan')
        print(f"{shape:>20s}  {mat_ms:>7.1f} ms  {r1_ms:>9.1f} ms  {spd:>5.2f}×  "
              f"{mat_peak:>5.1f}GB  {r1_peak:>5.1f}GB")
