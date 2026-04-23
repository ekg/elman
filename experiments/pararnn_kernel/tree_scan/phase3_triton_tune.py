"""Phase 3 quick win: exhaustive tuning of num_warps/num_stages for the
current fused backward kernel. Before committing to CUDA rewrite, rule
out that the Triton kernel just needs better tuning.
"""

import sys, os, time
import torch

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase4_newton_driver import sequential_e88_forward
from phase7_fused_backward import backward_e88_fused_rank1


def setup(B, H, T, N):
    dt = torch.bfloat16
    g = torch.Generator(device='cuda').manual_seed(0)
    S0_f = 0.1 * torch.randn(B, H, N, N, generator=g, dtype=torch.float32, device='cuda')
    K_f = 0.3 * torch.randn(B, H, T, N, generator=g, dtype=torch.float32, device='cuda')
    V_f = 0.3 * torch.randn(B, H, T, N, generator=g, dtype=torch.float32, device='cuda')
    decay_f = torch.sigmoid(0.5 + 0.1 * torch.randn(B, H, T, generator=g, dtype=torch.float32, device='cuda'))
    S_traj = sequential_e88_forward(S0_f, K_f, V_f, decay_f).to(dt)
    del S0_f, K_f, V_f, decay_f; torch.cuda.empty_cache()
    K = 0.3 * torch.randn(B, H, T, N, generator=g, dtype=dt, device='cuda')
    V = 0.3 * torch.randn(B, H, T, N, generator=g, dtype=dt, device='cuda')
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(B, H, T, generator=g, dtype=dt, device='cuda'))
    g_T = 0.01 * torch.randn(B, H, N, N, dtype=dt, device='cuda')
    dL_dout = 0.01 * torch.randn(B, H, T, N, dtype=dt, device='cuda')
    q = 0.3 * torch.randn(B, H, T, N, dtype=dt, device='cuda')
    return S_traj, K, V, decay, g_T, dL_dout, q


def bench_config(B, H, T, N, num_warps, num_stages, n_repeat=5):
    S_traj, K, V, decay, g_T, dL_dout, q = setup(B, H, T, N)
    def run():
        return backward_e88_fused_rank1(S_traj, K, V, decay, g_T, dL_dout, q,
                                         num_warps=num_warps, num_stages=num_stages)
    try:
        for _ in range(3): run()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_repeat): run()
        torch.cuda.synchronize()
        return (time.time() - t0) / n_repeat * 1000
    except Exception as e:
        return float('nan')


if __name__ == '__main__':
    print("Phase 3 — Triton backward tuning sweep\n")
    for H, N in [(141, 16), (83, 32)]:
        for T in [32768]:
            print(f"\nH={H} T={T} N={N}")
            print(f"  {'nw\\ns':>5s}  {'1':>7s}  {'2':>7s}  {'3':>7s}  {'4':>7s}  {'5':>7s}")
            best = (float('inf'), None, None)
            for nw in [1, 2, 4, 8, 16]:
                line = f"  {nw:>5d}  "
                for ns in [1, 2, 3, 4, 5]:
                    ms = bench_config(1, H, T, N, num_warps=nw, num_stages=ns)
                    if ms == ms and ms < best[0]:
                        best = (ms, nw, ns)
                    line += f"{ms:>5.2f}  " if ms == ms else "  N/A  "
                print(line)
            print(f"  BEST: num_warps={best[1]}, num_stages={best[2]}: {best[0]:.2f}ms")
