"""Test fused in-place Newton iter at T=128K. Must fit in 48 GB."""

import sys
import os
import time
import torch

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase7_fused_iter import fused_newton_iter_inplace
from phase4_newton_driver import sequential_e88_forward


def test_at_T(B, H, T, n, n_warm_iters=3):
    print(f"\n=== T={T} (B={B} H={H} n={n}) ===")
    bytes_S = B * H * T * n * n * 4
    print(f"  S_var size: {bytes_S / 1024**3:.2f} GB")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    g = torch.Generator(device='cuda').manual_seed(0)
    dt = torch.float32
    S0 = 0.1 * torch.randn(B, H, n, n, generator=g, dtype=dt, device='cuda')
    K = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    V = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    decay = 0.9 + 0.1 * torch.rand(B, H, T, generator=g, dtype=dt, device='cuda')

    t0 = time.time()
    S_prev = sequential_e88_forward(S0, K, V, decay)
    torch.cuda.synchronize()
    print(f"  Sequential forward (warm-start): {time.time() - t0:.2f}s")

    # Perturb to simulate warm start from previous step — MUST free S_prev first
    # so we don't hold both S_prev, S_init, AND a 16 GB randn at once.
    S_init = S_prev[:, :, 1:].clone()
    del S_prev
    torch.cuda.empty_cache()
    # Perturb in chunks so the tmp randn is small (1 GB per chunk, not 16 GB)
    CHUNK = 8192
    for t_chunk in range(0, T, CHUNK):
        t_end = min(t_chunk + CHUNK, T)
        S_init[:, :, t_chunk:t_end].add_(
            0.02 * torch.randn_like(S_init[:, :, t_chunk:t_end])
        )

    print(f"  After setup: {torch.cuda.memory_allocated()/1024**3:.2f} GB "
          f"(peak so far: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB)")

    # Newton iters
    total_ms = 0.0
    d_max = None
    for it in range(n_warm_iters):
        torch.cuda.synchronize()
        t0 = time.time()
        d_max = fused_newton_iter_inplace(S0, S_init, K, V, decay)
        torch.cuda.synchronize()
        ms = (time.time() - t0) * 1000
        total_ms += ms
        print(f"  iter {it}: {ms:.1f} ms  max|δ|={d_max:.3e}")

    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  Total {n_warm_iters} iters: {total_ms:.1f} ms  peak={peak:.2f} GB")
    return total_ms


if __name__ == '__main__':
    for T in [32768, 65536, 131072]:
        try:
            test_at_T(1, 32, T, 32)
        except Exception as e:
            print(f"  FAIL at T={T}: {e}")
            torch.cuda.empty_cache()
