"""Benchmark quasi-DEER Triton vs sequential forward at production shapes.

Compares:
  - pararnn_seq_fwd_v2 (sequential Triton kernel; current best at production)
  - quasi_deer Triton scan (log-depth, but requires multiple Newton iters)

At which T does quasi-DEER become faster than sequential?
"""

import sys
import os
import time
import torch
import triton

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pararnn_seq_fwd_v2 import pararnn_seq_fwd_v2
from quasi_deer_ref import _random_case, build_diag_ingredients
from quasi_deer_triton import qd_diagonal_scan_triton, quasi_deer_newton_triton


def bench_fn(fn, warmup=3, repeat=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / repeat * 1000  # ms


def run(H, T, N, dtype=torch.float32, tol=1e-4, max_iters=30, block_T=512):
    print(f"\n=== H={H:3d} T={T:6d} N={N:2d} dtype={str(dtype).split('.')[-1]} ===")
    S0, K, V, decay = _random_case(1, H, T, N, seed=0, dtype=dtype,
                                    l2_normalize_k=True, v_scale=0.3)

    # 1) Sequential Triton forward (current best)
    num_warps = 1 if N == 16 else 4
    def run_seq():
        S_traj = pararnn_seq_fwd_v2(S0, K, V, decay, num_warps=num_warps)
    seq_ms = bench_fn(run_seq)

    # 2) Quasi-DEER full Newton solve
    def run_qd():
        S_tri, it, res, _ = quasi_deer_newton_triton(
            S0, K, V, decay, max_iters=max_iters, tol=tol, block_T=block_T,
        )
    # First call warms up & tells us iter count
    S_tri, iters, res, history = quasi_deer_newton_triton(
        S0, K, V, decay, max_iters=max_iters, tol=tol, block_T=block_T,
    )
    qd_ms = bench_fn(run_qd)

    # 3) Per-iter scan time (isolated scan kernel, not including residual)
    D, b_vec, _ = build_diag_ingredients(S0, torch.zeros_like(S0.unsqueeze(2).expand(-1, -1, T, -1, -1)).contiguous(), K, V, decay)
    def run_scan():
        _ = qd_diagonal_scan_triton(D, b_vec, block_T=block_T)
    scan_ms = bench_fn(run_scan)

    # 4) Correctness vs sequential
    S_seq_traj = pararnn_seq_fwd_v2(S0, K, V, decay, num_warps=num_warps)
    # S_tri has shape [1, H, T+1, N, N] where [:,:,1:] = converged. Compare to S_seq_traj [1,H,T,N,N].
    diff = (S_tri[:, :, 1:] - S_seq_traj).abs().max().item()
    rel = diff / max(S_seq_traj.abs().max().item(), 1e-10)

    speedup = seq_ms / qd_ms
    print(f"  sequential    : {seq_ms:7.2f} ms")
    print(f"  quasi-DEER    : {qd_ms:7.2f} ms  ({iters} iters, final_res={res:.2e})")
    print(f"  single scan   : {scan_ms:7.2f} ms  ×{iters} = {scan_ms*iters:.1f} ms of scan")
    print(f"  speedup (qd/seq): {speedup:.2f}×  {'WIN' if speedup > 1 else 'LOSE'}")
    print(f"  correctness   : max|qd-seq|={diff:.2e}  rel={rel:.2e}")
    return {
        'H': H, 'T': T, 'N': N,
        'seq_ms': seq_ms, 'qd_ms': qd_ms,
        'scan_ms': scan_ms, 'iters': iters,
        'rel_err': rel, 'speedup': speedup,
    }


if __name__ == '__main__':
    results = []

    # Production E88 shapes
    print("Production E88 shapes — H=141 N=16:")
    for T in [1024, 4096, 16384, 32768]:
        try:
            results.append(run(141, T, 16))
        except Exception as e:
            print(f"  FAILED T={T}: {e}")

    print("\n\nProduction E88 shapes — H=83 N=32:")
    for T in [1024, 4096, 16384]:
        try:
            results.append(run(83, T, 32))
        except Exception as e:
            print(f"  FAILED T={T}: {e}")

    print("\n\nScaling to 65K with smaller H (memory constrained):")
    for T in [65536]:
        try:
            results.append(run(32, T, 16))
        except Exception as e:
            print(f"  FAILED T={T}: {e}")

    # Summary
    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'H':>4} {'T':>6} {'N':>3} | {'seq(ms)':>8} {'qd(ms)':>8} {'iters':>5} | "
          f"{'speedup':>8} {'rel_err':>10}")
    for r in results:
        print(f"{r['H']:>4} {r['T']:>6} {r['N']:>3} | "
              f"{r['seq_ms']:>8.2f} {r['qd_ms']:>8.2f} {r['iters']:>5} | "
              f"{r['speedup']:>7.2f}× {r['rel_err']:>10.2e}")
