"""Phase 3b — Triton kernel: sequential scan in T, parallel over (batch, head, row).

Key realization from exploration:

  - r=1 combine is LOSSLESS for LEFT-TO-RIGHT sequential scan (Phase 2).
  - r=1 combine is NOT ASSOCIATIVE → tree-parallel scan (Hillis-Steele) fails.
  - To time-parallelize we'd need dense n×n matrix state (DEER-style, O(n³) combines).
  - BUT: for E88 at B × H × n = O(1000s) rows per batch, we already have
    enough (batch, head, row) independent scans to saturate a GPU via
    DATA parallelism. Time parallelism isn't needed.

So this kernel runs ONE program per (batch, head, row) triple. Within
each program, the scan is sequential in T using the r=1 combine — exactly
what Phase 2 proved is lossless.

Step 1 (this file): single-row kernel, T up to 1024, correctness test
against Phase 3a PyTorch scan.
"""

import sys
import os
import math

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase3_triton import (
    _make_step, _combine_r1, scan_r1_pytorch,
    _rank2_to_rank1_triton, _combine_r1_triton,
)
from phase1_reference import _random_inputs, e88_row_step


# -----------------------------------------------------------------------------
# Triton kernel — single row, sequential scan over T.
# -----------------------------------------------------------------------------

@triton.jit
def _seq_scan_r1_kernel(
    D_ptr, u_ptr, v_ptr, b_ptr,        # inputs,  [T, N]
    out_ptr,                           # output δ = b-prefix, [T, N]
    T: tl.constexpr,
    N: tl.constexpr,
):
    """Sequential r=1 structured prefix scan.  One program per row."""
    n_idx = tl.arange(0, N)                    # [N]

    # Initialize prefix = step 0
    D_p = tl.load(D_ptr + n_idx)               # [N]
    u_p = tl.load(u_ptr + n_idx)
    v_p = tl.load(v_ptr + n_idx)
    b_p = tl.load(b_ptr + n_idx)

    # Write first prefix (= step 0) to output
    tl.store(out_ptr + n_idx, b_p)

    # Sequential scan for t = 1 .. T-1 (dynamic loop, not unrolled)
    for t in range(1, T):
        D_t = tl.load(D_ptr + t * N + n_idx)
        u_t = tl.load(u_ptr + t * N + n_idx)
        v_t = tl.load(v_ptr + t * N + n_idx)
        b_t = tl.load(b_ptr + t * N + n_idx)

        # Combine prefix with step t:  new = combine(prefix, step_t)
        D_p, u_p, v_p, b_p = _combine_r1_triton(D_p, u_p, v_p, b_p,
                                                 D_t, u_t, v_t, b_t)

        tl.store(out_ptr + t * N + n_idx, b_p)


def scan_r1_triton_seq(D, u, v, b):
    """Triton r=1 sequential scan, one program per row. Returns δ [T, N]."""
    T, N = D.shape
    assert T <= 1024, "step-1 kernel: static_range requires T constexpr"
    out = torch.empty_like(b)
    _seq_scan_r1_kernel[(1,)](
        D.contiguous(), u.contiguous(), v.contiguous(), b.contiguous(),
        out, T=T, N=N,
    )
    return out


# -----------------------------------------------------------------------------
# Correctness tests — against Phase 3a PyTorch scan on the same inputs.
# -----------------------------------------------------------------------------

def test_one(T, n, seed=0, dtype=torch.float32, device='cuda'):
    S0_row, K, V_i, decay = _random_inputs(T, n, seed=seed, dtype=dtype, device=device)
    S_var = torch.zeros(T, n, dtype=dtype, device=device)
    r = torch.empty(T, n, dtype=dtype, device=device)
    r[0] = -e88_row_step(S0_row, K[0], V_i[0], decay[0])
    for t in range(1, T):
        r[t] = -e88_row_step(S_var[t - 1], K[t], V_i[t], decay[t])

    steps = []
    for t in range(T):
        s_prev = S0_row if t == 0 else S_var[t - 1]
        steps.append(_make_step(s_prev, K[t], V_i[t], decay[t], r[t]))

    prefix = scan_r1_pytorch(steps)
    delta_pt = torch.stack([p[3] for p in prefix])

    D = torch.stack([s[0] for s in steps])
    u = torch.stack([s[1] for s in steps])
    v = torch.stack([s[2] for s in steps])
    b = torch.stack([s[3] for s in steps])

    delta_tr = scan_r1_triton_seq(D, u, v, b)

    diff = (delta_pt - delta_tr).abs().max().item()
    max_val = delta_pt.abs().max().item()
    status = "PASS" if diff < max(1e-4, 1e-4 * max_val) else "FAIL"
    print(f"  T={T:4d} n={n:3d}  max|δ|={max_val:.3e}  "
          f"max|δ_pt − δ_tr|={diff:.3e}  [{status}]")
    return diff


if __name__ == '__main__':
    print("Phase 3b step 1: single-row sequential Triton scan (float32).")
    for T, n in [(4, 4), (16, 8), (64, 16), (128, 32), (256, 32), (512, 32)]:
        test_one(T, n)
    print()
    print("If all PASS, Triton sequential scan with r=1 combine matches")
    print("the PyTorch reference. Next step: one program per (B, H, row).")
