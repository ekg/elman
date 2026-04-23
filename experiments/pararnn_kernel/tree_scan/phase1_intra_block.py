"""Phase 1 — Triton intra-block tree scan.

Scan all T positions of affine ops (A[t], b[t]) to produce δ[t] = b_cum[t],
using a Kogge-Stone tree via tl.associative_scan inside a single block.

Encoding: pack (A, b) into augmented matrix M = [[A, b], [0^T, 1]].
  Composition (A2, b2) ∘ (A1, b1) is then just M2 @ M1.
  Scan combine is a single matmul.

This phase handles T ≤ T_BLOCK positions. Phase 2 adds hierarchy for full T.

Program grid: (B*H*N_row,).  Each program scans one independent chain.
Each position's state: (M_DIM × M_DIM) matrix where M_DIM = N+1.

For the test we use N=16, M_DIM=17. Triton handles non-power-of-2 tile
dims but prefers powers of 2; we'll pad to M_PAD = 32 for tensor cores.
"""

import sys
import os
import time

import torch
import triton
import triton.language as tl

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase0_pytorch_ref import build_AB, affine_scan_sequential, affine_scan_tree, _random_case
from phase4_newton_driver import sequential_e88_forward


# -----------------------------------------------------------------------------
# Triton kernel
# -----------------------------------------------------------------------------

@triton.jit
def _matmul_combine(x, y):
    """x is earlier, y is later. Result: y @ x."""
    return tl.dot(y, x)


@triton.jit
def _intra_block_scan_kernel(
    M_ptr,              # [B, H, T, N_row, M_PAD, M_PAD]  augmented matrix
    out_ptr,            # [B, H, T, N_row, N]             = δ
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, N_ROW: tl.constexpr,
    N: tl.constexpr, M_PAD: tl.constexpr,
):
    """One program per (b, h, row). Scan all T positions using a single
    augmented-matrix scan with tl.associative_scan.

    Augmented M[t] = [[A[t], b[t], 0…], [0…, 1, 0…], [0…, 0, 1]…] padded
    to M_PAD × M_PAD (identity on the [N+1:, N+1:] block).
    After scan, δ[t] is in column N of M_cum[t], rows 0..N-1.
    """
    pid = tl.program_id(0).to(tl.int64)
    b_idx = pid // (H * N_ROW)
    h_idx = (pid // N_ROW) % H
    row = pid % N_ROW

    t_idx = tl.arange(0, T)
    r_idx = tl.arange(0, M_PAD)
    c_idx = tl.arange(0, M_PAD)

    bh_stride = T * N_ROW * M_PAD * M_PAD
    t_stride = N_ROW * M_PAD * M_PAD
    row_stride = M_PAD * M_PAD
    base = (b_idx * H + h_idx) * bh_stride + row * row_stride

    offsets = (t_idx[:, None, None] * t_stride +
               r_idx[None, :, None] * M_PAD +
               c_idx[None, None, :])
    M_tile = tl.load(M_ptr + base + offsets)

    M_cum = tl.associative_scan(M_tile, axis=0, combine_fn=_matmul_combine)

    # Extract δ[t, i] = M_cum[t, i, N] for i < N.
    col_mask = (c_idx == N)[None, None, :]
    row_mask = (r_idx < N)[None, :, None]
    picked = M_cum * (col_mask & row_mask).to(M_cum.dtype)
    delta_padded = tl.sum(picked, axis=2)  # [T, M_PAD]

    out_bh_stride = T * N_ROW * N
    out_t_stride = N_ROW * N
    out_row_stride = N
    out_base = (b_idx * H + h_idx) * out_bh_stride + row * out_row_stride
    out_offsets = t_idx[:, None] * out_t_stride + r_idx[None, :]
    out_mask = (r_idx < N)[None, :]
    tl.store(out_ptr + out_base + out_offsets, delta_padded, mask=out_mask)


# -----------------------------------------------------------------------------
# Host wrapper
# -----------------------------------------------------------------------------

def _pad_to(x, last_dim_size):
    """Pad last dim of x to `last_dim_size`. For A [..., N, N] pad both last dims.
    For b [..., N] pad last dim only."""
    return x


def _pack_augmented(A, b, M_PAD):
    """Build augmented M[t] = [[A, b, 0], [0, 1, 0], [0, 0, I]] padded to M_PAD."""
    *batch, N, _ = A.shape
    M = torch.zeros(*batch, M_PAD, M_PAD, dtype=A.dtype, device=A.device)
    M[..., :N, :N] = A
    M[..., :N, N] = b
    # Identity on padded diagonal indices N, N+1, ... (so they don't affect scan)
    for i in range(N, M_PAD):
        M[..., i, i] = 1.0
    return M


def intra_block_scan_triton(A, b):
    """Run Triton intra-block scan on (A, b), return δ.

    A: [B, H, T, N_row, N, N]
    b: [B, H, T, N_row, N]

    Returns δ: [B, H, T, N_row, N]
    """
    B, H, T, N_row, N, _ = A.shape
    M_PAD = 1
    while M_PAD < N + 1 or M_PAD < 16:
        M_PAD *= 2
    M = _pack_augmented(A, b, M_PAD).contiguous()
    out = torch.zeros(B, H, T, N_row, N, dtype=A.dtype, device=A.device)

    grid = (B * H * N_row,)
    _intra_block_scan_kernel[grid](
        M, out,
        B=B, H=H, T=T, N_ROW=N_row, N=N, M_PAD=M_PAD,
    )

    return out


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def test_correctness(B, H, T, N, seed=0, dtype=torch.float32):
    """Compare Triton intra-block scan to PyTorch reference."""
    S0, K, V, decay = _random_case(B, H, T, N, seed=seed, dtype=dtype)
    S_traj = sequential_e88_forward(S0, K, V, decay)
    S_var = S_traj[:, :, 1:] * 0.9

    A, b = build_AB(S0, S_var, K, V, decay)

    delta_ref = affine_scan_sequential(A, b)
    delta_triton = intra_block_scan_triton(A, b)

    diff = (delta_ref - delta_triton).abs().max().item()
    tol = max(1e-4, 1e-6 * T)
    status = "PASS" if diff < tol else "FAIL"
    print(f"  B={B} H={H:2d} T={T:3d} N={N:2d}  max|ref-tri|={diff:.2e}  "
          f"(tol {tol:.1e})  [{status}]")
    return diff


if __name__ == '__main__':
    print("Phase 1 — Triton intra-block tree scan correctness:\n")
    for shape in [(1, 2, 8, 16),
                  (1, 4, 16, 16),
                  (1, 4, 32, 16),
                  (1, 8, 64, 16),
                  (1, 4, 16, 32),
                  (1, 4, 32, 32)]:
        test_correctness(*shape)
