"""Phase 1c — Manual Kogge-Stone scan with tl.dot.

Since tl.associative_scan can't use tl.dot (scalar combine semantics),
implement Kogge-Stone explicitly: each level is a shifted-read + matmul.

For a single (b, h, row) chain of length T_BLOCK:
  Level d (for d in 1, 2, 4, ..., T_BLOCK/2):
    for t >= d: (A[t], b[t]) = combine((A[t-d], b[t-d]), (A[t], b[t]))

We do this by loading the full block into shared memory (the [T_BLOCK, M, M]
tile lives in registers for small T_BLOCK), and at each level doing explicit
combines.

Augmented matrix form: M = [[A, b, 0], [0, 1, 0], [0, 0, I]] padded.
"""

import sys, os, time
import torch
import triton
import triton.language as tl

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase0_pytorch_ref import build_AB, _random_case
from phase4_newton_driver import sequential_e88_forward
from phase7_fused_iter import fused_newton_iter


@triton.jit
def _manual_kogge_kernel(
    M_ptr,           # [B, H, N_row, T, M_PAD, M_PAD] packed augmented matrices
    out_ptr,         # [B, H, N_row, T, N]
    B: tl.constexpr, H: tl.constexpr, N_ROW: tl.constexpr,
    T: tl.constexpr, N: tl.constexpr, M_PAD: tl.constexpr,
):
    """One program per (b, h, row). Manual Kogge-Stone over T positions.

    For this kernel, T must be small enough that a [T, M_PAD, M_PAD] tile
    fits in registers/shared memory.
    """
    pid = tl.program_id(0).to(tl.int64)
    b_idx = pid // (H * N_ROW)
    h_idx = (pid // N_ROW) % H
    row = pid % N_ROW

    # Per-program base into M
    # Layout: [B, H, N_row, T, M_PAD, M_PAD]
    bh_stride = H * N_ROW * T * M_PAD * M_PAD
    h_stride = N_ROW * T * M_PAD * M_PAD
    row_stride = T * M_PAD * M_PAD
    t_stride = M_PAD * M_PAD
    base = b_idx * bh_stride + h_idx * h_stride + row * row_stride

    t_idx = tl.arange(0, T)
    r_idx = tl.arange(0, M_PAD)
    c_idx = tl.arange(0, M_PAD)

    offsets = (t_idx[:, None, None] * t_stride
               + r_idx[None, :, None] * M_PAD
               + c_idx[None, None, :])
    M = tl.load(M_ptr + base + offsets)  # [T, M_PAD, M_PAD]

    # Manual Kogge-Stone (unrolled). We express shifting via per-row broadcast.
    # At level d: for t in [d, T-1]: M_new[t] = M[t] @ M[t-d]
    # We simulate by constructing a "shifted" tile M_left where
    # M_left[t] = M[t-d] if t >= d else identity.
    # Then M_new = M @ M_left (batched matmul).

    # For each level, do the combine:
    # tl.dot works on 2D tiles. For 3D scan tile [T, M_PAD, M_PAD], we'd
    # need a batched dot. Triton's tl.dot doesn't directly support batched
    # input of this form.
    #
    # Workaround: loop over t for each level. That's T iterations per level
    # times log(T) levels = T*log(T) iterations total.
    #
    # Alternative: transpose to interpret T as a batch dim packed into M.
    # We can combine many matmuls into one big tl.dot by block-diagonalizing.

    # For phase 1c simplicity: loop over t using Triton's static_range.
    # This isn't log depth per program but single program can still be fast.

    # Sequential scan within the block (depth T, but uses tl.dot with tensor cores)
    # Output b-part is [t, :N, N] of cumulative at each position.

    # Actually let me do proper Kogge-Stone but with level loops over t.
    # Level d:
    #   for t in [d, T-1] (iterated in static order):
    #       M[t] = tl.dot(M[t], M_saved[t-d])
    #
    # But we can't dynamically index into 3D Triton tiles at runtime.

    # Simplest Triton-feasible: output b_cum using sequential scan with tl.dot.
    # Each step: A_cum_new = tl.dot(A_t, A_cum); b_cum_new = tl.dot(A_t, b_cum_col) + b_t_col
    # But this is exactly sequential scan — no log depth.

    # Given Triton constraints, this kernel demonstrates sequential scan with
    # tl.dot. No parallelism-in-time.

    # Initialize cumulative state = identity
    A_cum = tl.zeros((M_PAD, M_PAD), dtype=tl.float32)
    diag_idx = tl.arange(0, M_PAD)
    A_cum = tl.where(diag_idx[:, None] == diag_idx[None, :], 1.0, A_cum)
    b_cum = tl.zeros((M_PAD, 1), dtype=tl.float32)

    # This will produce a kernel that sequential-scans. We'd need a separate
    # mechanism for actual tree parallelism.
    # For now, just sequential. This is our "reference Triton kernel using tensor cores".

    for t_local in range(T):
        # Load A[t], b[t] from M_tile. Since M_tile is a static 3D Triton
        # tile, we need to extract slice [t_local, :, :]. Use masks.
        t_mask = (t_idx == t_local)[:, None, None]  # [T, 1, 1]
        # A_t = reduce over T axis of M * mask
        slice_t = tl.sum(M * t_mask.to(M.dtype), axis=0)  # [M_PAD, M_PAD]

        A_t = slice_t  # [M_PAD, M_PAD]
        # b_t is column N of the augmented matrix (rows 0..N-1)
        b_t_full = A_t[:, N]  # Nope — can't slice like this in Triton easily

        # ugh, Triton 3D indexing limits...
        # Simplest: skip this kernel variant. It won't do what we want.
        pass


# This kernel turns out not to be implementable cleanly in Triton given the
# tile-indexing constraints. Let me note the limitations and move on.

print("Phase 1c: pure-Triton Kogge-Stone with tl.dot is blocked by:")
print("  1. tl.associative_scan passes scalars to combine, can't use tl.dot.")
print("  2. Manual Kogge-Stone needs dynamic tile indexing which Triton lacks.")
print("  3. Batched matmul of [T, N, N] @ [T, N, N] not supported by tl.dot.")
print()
print("Path forward: write custom CUDA C++ kernel with raw tensor core intrinsics.")
print("Phase 2+ should be implemented in CUDA, not Triton.")
