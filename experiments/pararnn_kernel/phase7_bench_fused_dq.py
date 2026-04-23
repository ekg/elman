"""Benchmark fused dQ backward vs separate einsum."""

import sys, os, time
import torch
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase7_fused_backward import backward_e88_fused_rank1, backward_e88_fused_rank1_dQ
from phase4_newton_driver import sequential_e88_forward
from elman.models.e88_fla_hybrid import E88FLAHybridCUDAFunction


def test_correctness(B, H, T, n, seed=0):
    """Verify fused-dQ matches separate-einsum path."""
    torch.manual_seed(seed)
    dt = torch.float32
    S0 = 0.1 * torch.randn(B, H, n, n, dtype=dt, device='cuda')
    K = 0.3 * torch.randn(B, H, T, n, dtype=dt, device='cuda')
    V = 0.3 * torch.randn(B, H, T, n, dtype=dt, device='cuda')
    decay = 0.9 + 0.1 * torch.rand(B, H, T, dtype=dt, device='cuda')
    g_T = 0.1 * torch.randn(B, H, n, n, dtype=dt, device='cuda')
    dL_dout = 0.1 * torch.randn(B, H, T, n, dtype=dt, device='cuda')
    q = 0.3 * torch.randn(B, H, T, n, dtype=dt, device='cuda')

    S_traj = sequential_e88_forward(S0, K, V, decay)

    # Path 1: Separate einsum
    dS0_a, dK_a, dV_a, ddec_a = backward_e88_fused_rank1(
        S_traj, K, V, decay, g_T, dL_dout, q)
    dQ_a = torch.einsum('bhti,bhtij->bhtj', dL_dout, S_traj[:, :, 1:])

    # Path 2: Fused-dQ kernel
    dS0_b, dK_b, dV_b, ddec_b, dQ_b = backward_e88_fused_rank1_dQ(
        S_traj, K, V, decay, g_T, dL_dout, q)

    def rel(a, b): return (a - b).abs().max().item() / max(b.abs().max().item(), 1e-30)
    errs = [rel(dS0_b, dS0_a), rel(dK_b, dK_a), rel(dV_b, dV_a),
            rel(ddec_b, ddec_a), rel(dQ_b, dQ_a)]
    worst = max(errs)
    status = "PASS" if worst < 1e-5 else "FAIL"
    print(f"  B={B} H={H:3d} T={T:5d} n={n}  "
          f"err S0,K,V,dec,Q = {errs[0]:.1e}, {errs[1]:.1e}, {errs[2]:.1e}, "
          f"{errs[3]:.1e}, {errs[4]:.1e}  [{status}]")


def bench_separate(B, H, T, n, dt=torch.bfloat16, n_repeat=5):
    g = torch.Generator(device='cuda').manual_seed(0)
    S0 = 0.1 * torch.randn(B, H, n, n, generator=g, dtype=torch.float32, device='cuda')
    K = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=torch.float32, device='cuda')
    V = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=torch.float32, device='cuda')
    decay = 0.9 + 0.1 * torch.rand(B, H, T, generator=g, dtype=torch.float32, device='cuda')
    S_traj = sequential_e88_forward(S0, K, V, decay).to(dt)
    K_d = K.to(dt); V_d = V.to(dt); decay_d = decay.to(dt)
    g_T = 0.01 * torch.randn(B, H, n, n, dtype=dt, device='cuda')
    dL_dout = 0.01 * torch.randn(B, H, T, n, dtype=dt, device='cuda')
    q = 0.3 * torch.randn(B, H, T, n, dtype=dt, device='cuda')

    def run():
        d = backward_e88_fused_rank1(S_traj, K_d, V_d, decay_d, g_T, dL_dout, q)
        dQ = torch.einsum('bhti,bhtij->bhtj', dL_dout, S_traj[:, :, 1:])
        return d, dQ

    for _ in range(3): run()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat): _ = run()
    torch.cuda.synchronize()
    return (time.time() - t0) / n_repeat * 1000


def bench_fused(B, H, T, n, dt=torch.bfloat16, n_repeat=5):
    g = torch.Generator(device='cuda').manual_seed(0)
    S0 = 0.1 * torch.randn(B, H, n, n, generator=g, dtype=torch.float32, device='cuda')
    K = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=torch.float32, device='cuda')
    V = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=torch.float32, device='cuda')
    decay = 0.9 + 0.1 * torch.rand(B, H, T, generator=g, dtype=torch.float32, device='cuda')
    S_traj = sequential_e88_forward(S0, K, V, decay).to(dt)
    K_d = K.to(dt); V_d = V.to(dt); decay_d = decay.to(dt)
    g_T = 0.01 * torch.randn(B, H, n, n, dtype=dt, device='cuda')
    dL_dout = 0.01 * torch.randn(B, H, T, n, dtype=dt, device='cuda')
    q = 0.3 * torch.randn(B, H, T, n, dtype=dt, device='cuda')

    def run():
        return backward_e88_fused_rank1_dQ(S_traj, K_d, V_d, decay_d, g_T, dL_dout, q)

    for _ in range(3): run()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat): _ = run()
    torch.cuda.synchronize()
    return (time.time() - t0) / n_repeat * 1000


if __name__ == '__main__':
    print("Correctness (fused-dQ vs separate-einsum):\n")
    for shape in [(1, 4, 128, 16), (1, 16, 512, 32), (1, 32, 2048, 32)]:
        test_correctness(*shape)

    print("\n\nBench separate vs fused-dQ:\n")
    print(f"{'shape':>25s}  {'Sep (bwd+dQ)':>14s}  {'Fused dQ':>10s}  {'speedup':>8s}")
    for name, H, n in [("E88-n32 480M", 83, 32),
                       ("E88-n16 480M", 141, 16),
                       ("Small H=32",    32, 32)]:
        for T in [8192, 32768]:
            try:
                torch.cuda.empty_cache()
                sep = bench_separate(1, H, T, n)
                torch.cuda.empty_cache()
                fus = bench_fused(1, H, T, n)
                config = f"{name} T={T}"
                print(f"{config:>25s}  {sep:>11.2f} ms  {fus:>7.2f} ms  {sep/fus:>6.2f}×")
            except Exception as e:
                print(f"{name} T={T}  FAIL: {str(e)[:80]}")
                torch.cuda.empty_cache()
