"""Phase 1d — Sequential scan with tl.dot per step (tensor cores).

Realization: Triton's tl.associative_scan doesn't support matrix-valued
combine. Manual tree scan with dynamic indexing is blocked by tile limits.
The remaining Triton-friendly path: sequential scan inside each program,
but use tl.dot for the per-step matmul (tensor cores).

Depth is still O(T), same as our current Pararnn. Speedup comes from
tensor cores accelerating the N×N matmul per step vs elementwise rank-1.

For this to win: matmul via tensor cores must be faster than the current
per-step ops in phase7_fused_iter.py. That's a question of memory-bound
vs compute-bound trade-off.
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
def _seq_scan_tldot_kernel(
    A_ptr,          # [B, H, N_row, T, N_PAD, N_PAD]  full-matrix A per position
    b_ptr,          # [B, H, N_row, T, N_PAD]         b vector per position
    out_ptr,        # [B, H, N_row, T, N]             output δ
    B: tl.constexpr, H: tl.constexpr, N_ROW: tl.constexpr,
    T: tl.constexpr, N: tl.constexpr, N_PAD: tl.constexpr,
):
    """Sequential scan over T positions for one (b, h, row).

    At each step: b_cum = tl.dot(A[t], b_cum_col) + b[t]
    Depth = T. Uses tensor cores for per-step matmul.
    """
    pid = tl.program_id(0).to(tl.int64)
    b_idx = pid // (H * N_ROW)
    h_idx = (pid // N_ROW) % H
    row = pid % N_ROW

    # Strides for A: [B, H, N_row, T, N_PAD, N_PAD]
    bh_stride_A = H * N_ROW * T * N_PAD * N_PAD
    h_stride_A = N_ROW * T * N_PAD * N_PAD
    row_stride_A = T * N_PAD * N_PAD
    t_stride_A = N_PAD * N_PAD
    base_A = b_idx * bh_stride_A + h_idx * h_stride_A + row * row_stride_A

    # Strides for b: [B, H, N_row, T, N_PAD]
    bh_stride_b = H * N_ROW * T * N_PAD
    h_stride_b = N_ROW * T * N_PAD
    row_stride_b = T * N_PAD
    t_stride_b = N_PAD
    base_b = b_idx * bh_stride_b + h_idx * h_stride_b + row * row_stride_b

    r_idx = tl.arange(0, N_PAD)
    c_idx = tl.arange(0, N_PAD)

    # Initialize b_cum = 0 as [N_PAD, N_PAD] matrix (column-padded)
    b_cum_mat = tl.zeros((N_PAD, N_PAD), dtype=tl.float32)

    # Store output; we'll write first-column slice each step
    for t in range(T):
        # Load A[t] as [N_PAD, N_PAD]
        A_offsets = r_idx[:, None] * N_PAD + c_idx[None, :]
        A_t = tl.load(A_ptr + base_A + t * t_stride_A + A_offsets).to(tl.float32)

        # Load b[t] as [N_PAD] vector, broadcast to [N_PAD, N_PAD] (only col 0 nonzero)
        b_t = tl.load(b_ptr + base_b + t * t_stride_b + r_idx).to(tl.float32)
        # b_t_col: [N_PAD, N_PAD] with b_t in column 0
        col0_mask = (c_idx == 0)[None, :]
        b_t_col = b_t[:, None] * col0_mask.to(tl.float32)

        # b_cum = A_t @ b_cum + b_t_col (in matrix form)
        # tl.dot requires both 2D: [N_PAD, N_PAD] @ [N_PAD, N_PAD]
        b_cum_mat = tl.dot(A_t, b_cum_mat) + b_t_col

        # Extract delta[t] = first column of b_cum_mat, rows 0..N-1
        # Write to out[t, 0..N-1]
        out_offsets = t * N + r_idx
        out_mask = r_idx < N
        # First column of b_cum_mat: select col==0
        delta_col = tl.sum(b_cum_mat * col0_mask.to(tl.float32), axis=1)  # [N_PAD]

        # Out strides
        bh_stride_out = H * N_ROW * T * N
        h_stride_out = N_ROW * T * N
        row_stride_out = T * N
        base_out = b_idx * bh_stride_out + h_idx * h_stride_out + row * row_stride_out
        tl.store(out_ptr + base_out + out_offsets, delta_col, mask=out_mask)


def seq_scan_tldot(A, b):
    """Triton sequential scan with tl.dot per step."""
    B, H, T, N_row, N, _ = A.shape
    N_PAD = 1
    while N_PAD < max(N, 16):
        N_PAD *= 2

    # Pad A to [..., N_PAD, N_PAD]; identity on padded rows
    A_padded = torch.zeros(B, H, N_row, T, N_PAD, N_PAD, dtype=A.dtype, device=A.device)
    # Reorder A from [B, H, T, N_row, N, N] to [B, H, N_row, T, N, N]
    A_re = A.permute(0, 1, 3, 2, 4, 5)
    A_padded[..., :N, :N] = A_re
    # Identity on padding diagonal
    for i in range(N, N_PAD):
        A_padded[..., i, i] = 1.0

    # Pad b similarly
    b_padded = torch.zeros(B, H, N_row, T, N_PAD, dtype=b.dtype, device=b.device)
    b_re = b.permute(0, 1, 3, 2, 4)
    b_padded[..., :N] = b_re

    out = torch.zeros(B, H, N_row, T, N, dtype=A.dtype, device=A.device)

    grid = (B * H * N_row,)
    _seq_scan_tldot_kernel[grid](
        A_padded.contiguous(), b_padded.contiguous(), out,
        B=B, H=H, N_ROW=N_row, T=T, N=N, N_PAD=N_PAD,
    )

    # Reorder output back to [B, H, T, N_row, N]
    return out.permute(0, 1, 3, 2, 4).contiguous()


def test(B, H, T, N, seed=0, dtype=torch.float32):
    S0, K, V, decay = _random_case(B, H, T, N, seed=seed, dtype=dtype)
    S_traj = sequential_e88_forward(S0, K, V, decay)
    S_var = S_traj[:, :, 1:] * 0.9
    A, b = build_AB(S0, S_var, K, V, decay)

    delta_triton = seq_scan_tldot(A, b)
    delta_ref = fused_newton_iter(S0, S_var, K, V, decay)
    diff = (delta_triton - delta_ref).abs().max().item()
    tol = max(1e-4, 1e-5 * T)
    status = "PASS" if diff < tol else "FAIL"
    print(f"  B={B} H={H:3d} T={T:5d} N={N:2d}  max|tri - ref|={diff:.2e}  "
          f"(tol {tol:.1e})  [{status}]")
    return diff


def bench(B, H, T, N, n_repeat=3, dtype=torch.float32):
    S0, K, V, decay = _random_case(B, H, T, N, seed=0, dtype=dtype)
    S_traj = sequential_e88_forward(S0, K, V, decay)
    S_var = S_traj[:, :, 1:] * 0.9

    for _ in range(3): _ = fused_newton_iter(S0, S_var, K, V, decay)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat): _ = fused_newton_iter(S0, S_var, K, V, decay)
    torch.cuda.synchronize()
    par_ms = (time.time() - t0) / n_repeat * 1000

    A, b = build_AB(S0, S_var, K, V, decay)
    for _ in range(3): _ = seq_scan_tldot(A, b)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat): _ = seq_scan_tldot(A, b)
    torch.cuda.synchronize()
    tri_ms = (time.time() - t0) / n_repeat * 1000

    print(f"  B={B} H={H:3d} T={T:5d} N={N:2d}   "
          f"Pararnn r=1={par_ms:>6.1f} ms   seq+tl.dot={tri_ms:>6.1f} ms   "
          f"ratio={par_ms/tri_ms:.2f}×")


if __name__ == '__main__':
    print("Phase 1d — correctness:\n")
    for shape in [(1, 4, 32, 16), (1, 8, 128, 16), (1, 8, 256, 32)]:
        test(*shape)

    print("\nPhase 1d — perf (incl. build_AB + kernel):\n")
    for shape in [(1, 32, 1024, 16), (1, 141, 1024, 16), (1, 32, 8192, 16)]:
        try:
            bench(*shape)
        except Exception as e:
            print(f"  FAIL {shape}: {str(e)[:80]}")
