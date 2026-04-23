"""Measure Pararnn sequential scan vs HBM bandwidth peak.

Is sequential at the memory wall? If yes, no kernel optimization helps.
If not, there's headroom — find what's limiting.
"""

import sys, os, time
import torch

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase0_pytorch_ref import _random_case
from phase4_newton_driver import sequential_e88_forward
from phase7_fused_iter import fused_newton_iter


def measure(B, H, T, N, n_repeat=10):
    """Measure one Newton iter; compute HBM traffic and effective bandwidth."""
    S0, K, V, decay = _random_case(B, H, T, N, seed=0, dtype=torch.float32)
    S_traj = sequential_e88_forward(S0, K, V, decay)
    S_var = S_traj[:, :, 1:] * 0.9

    # Warmup
    for _ in range(5): _ = fused_newton_iter(S0, S_var, K, V, decay)
    torch.cuda.synchronize()

    # Time
    t0 = time.time()
    for _ in range(n_repeat):
        _ = fused_newton_iter(S0, S_var, K, V, decay)
    torch.cuda.synchronize()
    ms = (time.time() - t0) / n_repeat * 1000

    # Count HBM traffic per kernel call.
    # fused_newton_iter writes delta = same shape as S_var [B, H, T, N, N] fp32.
    # Reads: S0, S_var, K, V, decay
    # Writes: delta
    reads = {
        'S0':    B * H * N * N * 4,
        'S_var': B * H * T * N * N * 4,
        'K':     B * H * T * N * 4,
        'V':     B * H * T * N * 4,
        'decay': B * H * T * 4,
    }
    writes = {
        'delta': B * H * T * N * N * 4,
    }
    total_bytes = sum(reads.values()) + sum(writes.values())

    # A100 peak HBM: ~1.5 TB/s (1555 GB/s measured, ~80% of theoretical 2 TB/s)
    peak_hbm = 1.555e12
    bw_measured = (total_bytes / (ms / 1000))
    bw_frac = bw_measured / peak_hbm * 100

    # Theoretical lower bound: total_bytes / peak_hbm
    lb_ms = total_bytes / peak_hbm * 1000

    # Per-chain work: B*H*N_row chains, each does T sequential steps
    num_chains = B * H * N  # H * N_row per batch
    peak_compute_per_sm = 19.5e12  # A100 fp32 peak
    flops_per_step = N * N * 10  # rough: matvec + tanh + combine
    total_flops = T * flops_per_step * num_chains
    compute_lb_ms = total_flops / peak_compute_per_sm * 1000

    print(f"  B={B} H={H:3d} T={T:6d} N={N:2d}  "
          f"time={ms:>7.2f} ms  "
          f"HBM={total_bytes/1e9:>5.2f} GB  ({bw_measured/1e9:>5.1f} GB/s = {bw_frac:.1f}% peak)  "
          f"LB: HBM={lb_ms:.2f}ms, compute={compute_lb_ms:.2f}ms")
    return ms, total_bytes, bw_measured


if __name__ == '__main__':
    print("Pararnn r=1 sequential: HBM bandwidth analysis\n")
    print("Ratio measured/LB_HBM tells us how close to memory wall we are.\n")
    print("  100% peak HBM = memory wall (no kernel optimization can help)\n")
    for B, H, T, N in [
        (1, 32,  1024, 16),
        (1, 32,  8192, 16),
        (1, 32, 32768, 16),
        (1, 141,  1024, 16),
        (1, 141,  8192, 16),
        (1, 141, 32768, 16),
        (1, 32,  8192, 32),
        (1, 83,  8192, 32),
        (1, 83,  32768, 32),
    ]:
        try:
            measure(B, H, T, N)
        except Exception as e:
            print(f"  FAIL: {str(e)[:100]}")
