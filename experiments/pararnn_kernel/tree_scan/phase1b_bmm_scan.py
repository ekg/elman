"""Phase 1b — tree scan using torch.bmm (cuBLAS batched matmul).

Given Triton's tl.associative_scan doesn't support matrix-valued combines
(scalar semantics), and a manual Triton Kogge-Stone is costly,  we use
torch.bmm which calls cuBLAS batched matmul — tensor-core accelerated.

Each tree level is one big batched matmul. 17 launches at T=128K.

This lets us actually measure whether the tree-scan algorithmic approach
can hit the 5× target. If it works, Phase 2 can rewrite in pure Triton
using the same math.
"""

import sys, os, time
import torch

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase0_pytorch_ref import build_AB, _random_case
from phase4_newton_driver import sequential_e88_forward
from phase7_fused_iter import fused_newton_iter


def tree_scan_bmm(A, b):
    """Hillis-Steele inclusive scan using torch.bmm for the combine.

    A: [B, H, T, N_row, N, N]
    b: [B, H, T, N_row, N]

    At each level d: for t >= d, combine (A[t-d], b[t-d]) with (A[t], b[t]).
    Result: b_cum[t] = δ[t].
    """
    B, H, T, N_row, N, _ = A.shape

    # Flatten into batched matrices for bmm.
    # All (B, H, N_row) chains are independent.
    # Total chains = B * H * N_row. Each chain has T positions of (N×N, N).

    batch_size = B * H * N_row
    A_flat = A.permute(0, 1, 3, 2, 4, 5).reshape(batch_size, T, N, N).contiguous()
    b_flat = b.permute(0, 1, 3, 2, 4).reshape(batch_size, T, N).contiguous()

    # Work with A_cum, b_cum as running partial-prefix state
    A_cur = A_flat.clone()   # [batch_size, T, N, N]
    b_cur = b_flat.clone()   # [batch_size, T, N]

    d = 1
    while d < T:
        # For t >= d: new[t] = combine(cur[t-d], cur[t]) = (cur[t].A @ cur[t-d].A,
        #                                                    cur[t].A @ cur[t-d].b + cur[t].b)
        # Gather A_left, b_left at positions [0 .. T-d-1]  (these are "cur[t-d]" for t in [d, T-1])
        # Gather A_right, b_right at positions [d .. T-1]  (these are "cur[t]" for t in [d, T-1])
        A_left = A_cur[:, :T - d]                 # [batch, T-d, N, N]
        b_left = b_cur[:, :T - d]                 # [batch, T-d, N]
        A_right = A_cur[:, d:]                    # [batch, T-d, N, N]
        b_right = b_cur[:, d:]                    # [batch, T-d, N]

        # Reshape for bmm: collapse batch + sequence dims
        bs = batch_size * (T - d)
        A_l = A_left.reshape(bs, N, N)
        A_r = A_right.reshape(bs, N, N)
        b_l = b_left.reshape(bs, N, 1)
        b_r = b_right.reshape(bs, N, 1)

        # A_new = A_right @ A_left
        A_new = torch.bmm(A_r, A_l).reshape(batch_size, T - d, N, N)
        # b_new = A_right @ b_left + b_right
        b_new_ = torch.bmm(A_r, b_l).reshape(batch_size, T - d, N) + b_right

        # Write back to positions [d .. T-1]
        A_cur = A_cur.clone()
        b_cur = b_cur.clone()
        A_cur[:, d:] = A_new
        b_cur[:, d:] = b_new_

        d *= 2

    # b_cur is now the inclusive prefix. Reshape back to [B, H, T, N_row, N].
    delta = b_cur.reshape(B, H, N_row, T, N).permute(0, 1, 3, 2, 4).contiguous()
    return delta


def tree_scan_bmm_inplace(A, b):
    """Same as tree_scan_bmm but avoid `.clone()` inside the loop.

    Uses ping-pong buffers (or just updates in place by doing the combine
    first into a temp then copying).
    """
    B, H, T, N_row, N, _ = A.shape
    batch_size = B * H * N_row
    A_cur = A.permute(0, 1, 3, 2, 4, 5).reshape(batch_size, T, N, N).contiguous()
    b_cur = b.permute(0, 1, 3, 2, 4).reshape(batch_size, T, N).contiguous()

    d = 1
    while d < T:
        A_left = A_cur[:, :T - d]
        b_left = b_cur[:, :T - d]
        A_right = A_cur[:, d:]
        b_right = b_cur[:, d:]

        bs = batch_size * (T - d)
        A_new = torch.bmm(A_right.reshape(bs, N, N), A_left.reshape(bs, N, N)).reshape(batch_size, T - d, N, N)
        b_new = torch.bmm(A_right.reshape(bs, N, N), b_left.reshape(bs, N, 1)).reshape(batch_size, T - d, N) + b_right

        A_cur[:, d:] = A_new
        b_cur[:, d:] = b_new

        d *= 2

    delta = b_cur.reshape(B, H, N_row, T, N).permute(0, 1, 3, 2, 4).contiguous()
    return delta


def newton_step_bmm(S0, S_var, K, V, decay):
    """One Newton step using BMM tree scan. Returns δ."""
    A, b = build_AB(S0, S_var, K, V, decay)
    return tree_scan_bmm_inplace(A, b)


def test_correctness(B, H, T, N, seed=0, dtype=torch.float32):
    S0, K, V, decay = _random_case(B, H, T, N, seed=seed, dtype=dtype)
    S_traj = sequential_e88_forward(S0, K, V, decay)
    S_var = S_traj[:, :, 1:] * 0.9

    delta_bmm = newton_step_bmm(S0, S_var, K, V, decay)
    delta_ref = fused_newton_iter(S0, S_var, K, V, decay)

    diff = (delta_bmm - delta_ref).abs().max().item()
    tol = max(1e-4, 1e-5 * T)
    status = "PASS" if diff < tol else "FAIL"
    print(f"  B={B} H={H:2d} T={T:5d} N={N:2d}  max|bmm - r=1-ref|={diff:.2e}  "
          f"(tol {tol:.1e})  [{status}]")


def bench(B, H, T, N, n_repeat=3, dtype=torch.float32):
    S0, K, V, decay = _random_case(B, H, T, N, seed=0, dtype=dtype)
    S_traj = sequential_e88_forward(S0, K, V, decay)
    S_var = S_traj[:, :, 1:] * 0.9
    A, b = build_AB(S0, S_var, K, V, decay)

    # Warmup
    for _ in range(3): _ = tree_scan_bmm_inplace(A, b)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(n_repeat): _ = tree_scan_bmm_inplace(A, b)
    torch.cuda.synchronize()
    bmm_ms = (time.time() - t0) / n_repeat * 1000

    # Baseline: Pararnn fused forward (our current single-iter speed)
    for _ in range(3): _ = fused_newton_iter(S0, S_var, K, V, decay)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat): _ = fused_newton_iter(S0, S_var, K, V, decay)
    torch.cuda.synchronize()
    par_ms = (time.time() - t0) / n_repeat * 1000

    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  B={B} H={H:3d} T={T:5d} N={N:2d}  "
          f"Pararnn 1-iter={par_ms:>6.1f} ms   BMM tree scan={bmm_ms:>6.1f} ms   "
          f"ratio={par_ms/bmm_ms:.2f}×   peak={peak:.1f} GB")


if __name__ == '__main__':
    print("Phase 1b — BMM tree scan correctness:\n")
    for shape in [(1, 4, 64, 16), (1, 8, 128, 16), (1, 8, 256, 32), (1, 4, 512, 32)]:
        test_correctness(*shape)

    print("\nPhase 1b — performance vs Pararnn r=1 fused sequential scan:\n")
    for shape in [(1, 141, 1024, 16),
                  (1, 141, 8192, 16),
                  (1, 32, 8192, 16),
                  (1, 32, 32768, 16),
                  (1, 83, 8192, 32)]:
        try:
            bench(*shape)
        except Exception as e:
            print(f"  FAIL {shape}: {str(e)[:100]}")
