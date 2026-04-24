"""Measure peak memory of bf16 vs fp8 S_traj at production shapes."""
import os, sys
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pararnn_seq_fwd_v2 import pararnn_seq_fwd_v2, backward_v2
from pararnn_seq_fwd_v2_fp8 import pararnn_seq_fwd_v2_fp8, backward_v2_fp8


def measure_mem(B, H, T, N):
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

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # BF16 fwd
    st_bf = pararnn_seq_fwd_v2(S0, K, V, decay, num_warps=nw_fwd)
    torch.cuda.synchronize()
    bf_fwd_peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    # add bwd
    bwd_bf = backward_v2(S0, st_bf, K, V, decay, g_T, dL_dout, q,
                         num_warps=nw_bwd, num_stages=1)
    torch.cuda.synchronize()
    bf_full_peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    bf_traj_bytes = st_bf.numel() * st_bf.element_size()
    del st_bf, bwd_bf
    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()
    # FP8 fwd
    st_fp8 = pararnn_seq_fwd_v2_fp8(S0, K, V, decay, num_warps=nw_fwd)
    torch.cuda.synchronize()
    fp8_fwd_peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    bwd_fp8 = backward_v2_fp8(S0, st_fp8, K, V, decay, g_T, dL_dout, q,
                              num_warps=nw_bwd, num_stages=1)
    torch.cuda.synchronize()
    fp8_full_peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    fp8_traj_bytes = st_fp8.numel() * st_fp8.element_size()

    return {
        'bf_fwd': bf_fwd_peak, 'bf_full': bf_full_peak, 'bf_traj_MB': bf_traj_bytes / 1e6,
        'fp8_fwd': fp8_fwd_peak, 'fp8_full': fp8_full_peak, 'fp8_traj_MB': fp8_traj_bytes / 1e6,
    }


if __name__ == '__main__':
    shapes = [(1, 141, 16384, 16), (1, 141, 32768, 16), (1, 141, 65536, 16),
              (1, 83, 16384, 32), (1, 83, 32768, 32)]
    print(f"{'B':>2} {'H':>4} {'T':>6} {'N':>3}  {'bf_traj_MB':>11}  {'fp8_traj_MB':>12}  {'bf_fwd_GB':>10}  {'fp8_fwd_GB':>11}  {'bf_full_GB':>11}  {'fp8_full_GB':>12}")
    for (B, H, T, N) in shapes:
        try:
            r = measure_mem(B, H, T, N)
            print(f"{B:2d} {H:4d} {T:6d} {N:3d}  {r['bf_traj_MB']:11.1f}  {r['fp8_traj_MB']:12.1f}  "
                  f"{r['bf_fwd']:10.2f}  {r['fp8_fwd']:11.2f}  {r['bf_full']:11.2f}  {r['fp8_full']:12.2f}")
        except Exception as e:
            print(f"{B:2d} {H:4d} {T:6d} {N:3d}  FAIL: {str(e)[:80]}")
        torch.cuda.empty_cache()
