"""Test fused Newton iter at T=128K — verify int64 fix resolves overflow."""

import sys
import os
import time
import torch

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase7_fused_iter import fused_newton_iter
from phase4_newton_driver import sequential_e88_forward


def test_at_T(B, H, T, n):
    print(f"\n=== T={T} (B={B} H={H} n={n}) ===")
    bytes_S = B * H * T * n * n * 4
    print(f"  S_var size: {bytes_S / 1024**3:.2f} GB  (int32 offset range: "
          f"bh*T*N*N max = {(B*H - 1) * T * n * n} vs 2^31 = {2**31})")

    g = torch.Generator(device='cuda').manual_seed(0)
    dt = torch.float32
    S0 = 0.1 * torch.randn(B, H, n, n, generator=g, dtype=dt, device='cuda')
    K = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    V = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    decay = 0.9 + 0.1 * torch.rand(B, H, T, generator=g, dtype=dt, device='cuda')

    torch.cuda.synchronize()
    alloc_before = torch.cuda.memory_allocated() / 1024**3
    print(f"  Alloc before forward: {alloc_before:.2f} GB")

    # Warm-start: sequential forward
    t0 = time.time()
    S_prev = sequential_e88_forward(S0, K, V, decay)
    torch.cuda.synchronize()
    print(f"  Sequential forward: {time.time() - t0:.2f}s, "
          f"alloc={torch.cuda.memory_allocated()/1024**3:.2f} GB")

    S_init = S_prev[:, :, 1:].clone()
    del S_prev
    torch.cuda.empty_cache()

    print(f"  After clone+free: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # One Newton iter
    t0 = time.time()
    delta = fused_newton_iter(S0, S_init, K, V, decay)
    torch.cuda.synchronize()
    ms = (time.time() - t0) * 1000
    max_peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  fused_newton_iter: {ms:.1f} ms  "
          f"peak={max_peak:.2f} GB  max|delta|={delta.abs().max().item():.3e}")
    return ms


if __name__ == '__main__':
    torch.cuda.reset_peak_memory_stats()
    # Check int64 fix works at increasing T.
    for T in [32768, 65536, 131072]:
        try:
            test_at_T(1, 32, T, 32)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        except Exception as e:
            print(f"  FAIL at T={T}: {e}")
            torch.cuda.empty_cache()
