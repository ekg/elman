"""Phase 3 — profile the current Pararnn backward.

Is it HBM-bound (no kernel work helps) or latency-bound (can hide via prefetch)?

Same analysis as phase0 HBM analysis but for the fused backward kernel.
"""

import sys, os, time
import torch

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase4_newton_driver import sequential_e88_forward
from phase7_fused_backward import backward_e88_fused_rank1


def measure(B, H, T, N, n_repeat=10):
    """Time the Pararnn bf16 fused backward + compute HBM."""
    dt = torch.bfloat16
    g = torch.Generator(device='cuda').manual_seed(0)

    # Build S_traj (bf16) via fp32 sequential + cast
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

    def run():
        return backward_e88_fused_rank1(S_traj, K, V, decay, g_T, dL_dout, q,
                                         num_warps=4 if N == 32 else 1, num_stages=1)
    for _ in range(5): run()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat): run()
    torch.cuda.synchronize()
    ms = (time.time() - t0) / n_repeat * 1000

    # HBM traffic accounting (bf16 = 2 bytes)
    reads = {
        'S_traj':  B * H * (T + 1) * N * N * 2,  # full trajectory
        'K':       B * H * T * N * 2,
        'V':       B * H * T * N * 2,
        'decay':   B * H * T * 2,
        'g_T':     B * H * N * N * 2,
        'dL_dout': B * H * T * N * 2,
        'q':       B * H * T * N * 2,
    }
    writes = {
        'dS0':     B * H * N * N * 2,
        'dK':      B * H * T * N * 2,
        'dV':      B * H * T * N * 2,
        'ddecay':  B * H * T * 2,
    }
    total = sum(reads.values()) + sum(writes.values())
    peak = 1.555e12
    bw = total / (ms / 1000)
    frac = bw / peak * 100
    lb_ms = total / peak * 1000

    print(f"  H={H:3d} T={T:6d} N={N:2d}  time={ms:>6.2f}ms  HBM={total/1e9:.2f}GB  "
          f"bw={bw/1e9:.1f}GB/s = {frac:.1f}% peak  (lb={lb_ms:.2f}ms, ratio={ms/lb_ms:.1f}×)")


if __name__ == '__main__':
    print("Phase 3 — Pararnn backward HBM analysis\n")
    for H, N in [(141, 16), (83, 32), (32, 16)]:
        for T in [8192, 16384, 32768, 65536]:
            try:
                measure(1, H, T, N)
            except Exception as e:
                print(f"  FAIL: {str(e)[:100]}")
