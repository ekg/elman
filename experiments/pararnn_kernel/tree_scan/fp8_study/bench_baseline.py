"""Step 1: baseline measurement of bf16 Pararnn forward+backward.

Measures forward kernel time, backward kernel time, peak memory at
production shapes on GPU 2.
"""
import os, sys, time
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pararnn_seq_fwd_v2 import pararnn_seq_fwd_v2, backward_v2


def bench_one(B, H, T, N, n_warmup=3, n_rep=5):
    dt = torch.bfloat16
    dev = 'cuda'
    g = torch.Generator(device=dev).manual_seed(0)
    K = 0.3 * torch.randn(B, H, T, N, generator=g, dtype=dt, device=dev)
    V = 0.3 * torch.randn(B, H, T, N, generator=g, dtype=dt, device=dev)
    q = 0.3 * torch.randn(B, H, T, N, generator=g, dtype=dt, device=dev)
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(B, H, T, generator=g, dtype=dt, device=dev))
    S0 = 0.1 * torch.randn(B, H, N, N, generator=g, dtype=dt, device=dev)
    dL_dout = 0.01 * torch.randn(B, H, T, N, dtype=dt, device=dev)
    g_T = torch.zeros(B, H, N, N, dtype=dt, device=dev)
    nw_fwd = 1 if N == 16 else 4
    nw_bwd = 1 if N == 16 else 2

    # Forward-only timing
    for _ in range(n_warmup):
        S_traj = pararnn_seq_fwd_v2(S0, K, V, decay, num_warps=nw_fwd)
        del S_traj
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(n_rep):
        S_traj = pararnn_seq_fwd_v2(S0, K, V, decay, num_warps=nw_fwd)
    torch.cuda.synchronize()
    fwd_ms = (time.time() - t0) / n_rep * 1000
    fwd_peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # Full roundtrip: fwd + backward
    def run_full():
        S_traj = pararnn_seq_fwd_v2(S0, K, V, decay, num_warps=nw_fwd)
        bwd = backward_v2(S0, S_traj, K, V, decay, g_T, dL_dout, q,
                          num_warps=nw_bwd, num_stages=1)
        return bwd

    for _ in range(n_warmup):
        run_full()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(n_rep):
        run_full()
    torch.cuda.synchronize()
    full_ms = (time.time() - t0) / n_rep * 1000
    full_peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # Backward-only (subtract)
    bwd_ms = full_ms - fwd_ms

    # S_traj HBM size
    traj_bytes = B * H * T * N * N * 2  # bf16 = 2 bytes
    return fwd_ms, bwd_ms, full_ms, fwd_peak_gb, full_peak_gb, traj_bytes


if __name__ == '__main__':
    print("BF16 baseline on GPU 2 (RTX 6000 Ada)\n")
    shapes_N16 = [(1, 141, 4096, 16), (1, 141, 16384, 16), (1, 141, 32768, 16), (1, 141, 65536, 16)]
    shapes_N32 = [(1, 83, 16384, 32), (1, 83, 32768, 32)]
    print(f"{'B':>3} {'H':>4} {'T':>6} {'N':>3}  {'fwd_ms':>8}  {'bwd_ms':>8}  {'full_ms':>8}  {'fwd_peak_GB':>11}  {'full_peak_GB':>12}  {'traj_MB':>8}")
    for shp in shapes_N16 + shapes_N32:
        B, H, T, N = shp
        try:
            fwd_ms, bwd_ms, full_ms, fwd_peak, full_peak, traj_bytes = bench_one(B, H, T, N)
            print(f"{B:3d} {H:4d} {T:6d} {N:3d}  {fwd_ms:8.2f}  {bwd_ms:8.2f}  {full_ms:8.2f}  "
                  f"{fwd_peak:11.2f}  {full_peak:12.2f}  {traj_bytes/1e6:8.1f}")
        except Exception as e:
            print(f"{B:3d} {H:4d} {T:6d} {N:3d}  FAIL: {str(e)[:80]}")
        torch.cuda.empty_cache()
