"""Phase 0b — rank-K truncation feasibility for tree-scan Newton.

Core question: if we maintain scan state as rank-K affine op
A = diag(D) - sum_{j=1..K} u_j v_j^T, and truncate back to rank-K after
each combine (which produces rank ≤ 2K intermediate), does the scan still
produce an accurate enough δ to support Newton convergence?

If yes: per combine is O(K²N) FLOPs instead of O(N³). At K=4, N=32:
K²N = 512 vs N³ = 32K → **64× less work**. Makes 5× speedup reachable.

If no: tree scan path needs full N×N matrices, confirming 5× infeasible.

Plan:
  1. Build true full-matrix reference Newton step.
  2. Build rank-K truncated scan (sequential — just to test the math).
  3. Measure |δ_rank_k - δ_full| across K, T.
  4. Iterate Newton with rank-K. Does it still converge to sequential
     forward trajectory?
"""

import sys
import os
import torch

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase0_pytorch_ref import build_AB, affine_scan_sequential, _random_case
from phase4_newton_driver import sequential_e88_forward


def truncate_to_rank_k(A_full, N, K):
    """Project a full N×N matrix to its best rank-K approximation of
    the (diag + low-rank) form.

    A_full = diag(D) + R where R is the non-diagonal remainder.
    Extract D = diag(A_full); compute R = A_full - diag(D).
    Best rank-K approximation of R: truncated SVD, keeping top K.

    Returns:
      D: [..., N]         — diagonal
      U: [..., N, K]      — left factors
      V: [..., N, K]      — right factors
      Reconstruction:  A = diag(D) + U @ V^T
    """
    *batch, _, _ = A_full.shape
    D = torch.diagonal(A_full, dim1=-2, dim2=-1)  # [..., N]
    eye = torch.eye(N, device=A_full.device, dtype=A_full.dtype)
    R = A_full - D.unsqueeze(-1) * eye  # [..., N, N]

    # Truncated SVD of R: keep top K singular values.
    U_full, S, Vh = torch.linalg.svd(R, full_matrices=False)
    # Take top K
    U_k = U_full[..., :K]       # [..., N, K]
    S_k = S[..., :K]            # [..., K]
    V_k = Vh[..., :K, :].transpose(-1, -2)  # [..., N, K]

    # Fold sqrt(S) into U and V
    S_sqrt = torch.sqrt(S_k).unsqueeze(-2)  # [..., 1, K]
    U_out = U_k * S_sqrt
    V_out = V_k * S_sqrt

    return D, U_out, V_out


def reconstruct_from_rank_k(D, U, V):
    """Build A = diag(D) + U V^T."""
    N = D.shape[-1]
    eye = torch.eye(N, device=D.device, dtype=D.dtype)
    A = D.unsqueeze(-1) * eye + torch.einsum('...ik,...jk->...ij', U, V)
    return A


def rank_k_combine(D1, U1, V1, b1, D2, U2, V2, b2, K):
    """Combine two rank-K affine ops: (later ∘ earlier) applied to state.

    op_i(x) = (diag(D_i) + U_i V_i^T) @ x + b_i
    (op_2 ∘ op_1)(x) = op_2(op_1(x))
                     = (A_2)(A_1 x + b_1) + b_2
                     = A_2 A_1 x + A_2 b_1 + b_2

    The composed A_2 A_1 is rank ≤ 2K non-diagonal + diagonal(D_2 D_1).
    We truncate back to rank K using SVD on the non-diagonal part.
    """
    # Reconstruct full A_1, A_2 (slow but clear for reference)
    A_1 = reconstruct_from_rank_k(D1, U1, V1)
    A_2 = reconstruct_from_rank_k(D2, U2, V2)

    A_new = torch.einsum('...ij,...jk->...ik', A_2, A_1)
    b_new = torch.einsum('...ij,...j->...i', A_2, b1) + b2

    # Truncate A_new back to rank K (diag + rank K)
    N = D1.shape[-1]
    D_new, U_new, V_new = truncate_to_rank_k(A_new, N, K)

    return D_new, U_new, V_new, b_new


def full_rank_sequential_reference(A, b):
    """Sequential scan of full matrices — ground truth."""
    return affine_scan_sequential(A, b)


def rank_k_tree_scan(A, b, K):
    """Hillis-Steele inclusive tree scan with rank-K truncation at every combine.

    Maintains state as rank-K (diag + K rank-1 updates) throughout.
    Each combine produces rank-2K, then truncates back to rank-K via SVD.

    A: [B, H, T, N_row, N, N]
    b: [B, H, T, N_row, N]
    K: rank to keep

    Returns δ: [B, H, T, N_row, N]
    """
    B, H, T, N_row, N, _ = A.shape

    # Current state: keep as full N×N matrices for simplicity, truncate after each combine.
    A_cur = A.clone()
    b_cur = b.clone()

    d = 1
    while d < T:
        # For t >= d: combine(cur[t-d], cur[t]) = (A[t] @ A[t-d], A[t] @ b[t-d] + b[t])
        A_left = A_cur[:, :, :T - d]   # [B, H, T-d, N_row, N, N]
        b_left = b_cur[:, :, :T - d]
        A_right = A_cur[:, :, d:]
        b_right = b_cur[:, :, d:]

        # Combine
        A_new = torch.einsum('...ij,...jk->...ik', A_right, A_left)
        b_new = torch.einsum('...ij,...j->...i', A_right, b_left) + b_right

        # Truncate A_new back to rank-K (diag + rank K)
        if K < N:
            D_new, U_new, V_new = truncate_to_rank_k(A_new, N, K)
            A_new = reconstruct_from_rank_k(D_new, U_new, V_new)

        A_cur[:, :, d:] = A_new
        b_cur[:, :, d:] = b_new

        d *= 2

    return b_cur


def rank_k_sequential_scan(A, b, K):
    """Sequential scan with rank-K truncated state.

    A: [B, H, T, N_row, N, N]
    b: [B, H, T, N_row, N]
    K: rank to keep

    Returns:
      delta: [B, H, T, N_row, N]  (= b_cum[t])
      max_trunc_err: max relative truncation error across the scan
    """
    B, H, T, N_row, N, _ = A.shape

    # Initial state: identity (D=1, U=0, V=0, b=0)
    D_cum = torch.ones(B, H, N_row, N, dtype=A.dtype, device=A.device)
    U_cum = torch.zeros(B, H, N_row, N, K, dtype=A.dtype, device=A.device)
    V_cum = torch.zeros(B, H, N_row, N, K, dtype=A.dtype, device=A.device)
    b_cum = torch.zeros(B, H, N_row, N, dtype=b.dtype, device=b.device)

    delta_out = torch.empty_like(b)
    max_err = 0.0
    for t in range(T):
        A_t = A[:, :, t]   # [B, H, N_row, N, N]
        b_t = b[:, :, t]

        # Extract rank-K form of A_t (A_t is already exact rank-1 structured,
        # so rank-1 extraction should be lossless for K >= 1).
        D_t, U_t, V_t = truncate_to_rank_k(A_t, N, K)

        # Combine: new_cum = (A_t, b_t) ∘ (A_cum, b_cum) = later ∘ earlier
        # But we want prefix scan with earliest first — so:
        # scan[0] = (A_0, b_0)
        # scan[t] = combine(scan[t-1], (A_t, b_t))
        # combine(earlier, later) = (A_later @ A_earlier, A_later @ b_earlier + b_later)
        D_cum, U_cum, V_cum, b_cum = rank_k_combine(
            D_cum, U_cum, V_cum, b_cum,     # earlier (cumulative so far)
            D_t, U_t, V_t, b_t,              # later (this step)
            K
        )
        delta_out[:, :, t] = b_cum

    return delta_out


def newton_step_rank_k(S0, S_var, K, V, decay, rank_K, use_tree=False):
    """One Newton step using rank-K truncated scan (tree or sequential)."""
    A, b = build_AB(S0, S_var, K, V, decay)
    if use_tree:
        return rank_k_tree_scan(A, b, rank_K)
    else:
        return rank_k_sequential_scan(A, b, rank_K)


def test_tree_vs_sequential_rank_k(B, H, T, N, K, seed=0, dtype=torch.float64):
    """Tree-scan rank-K produces the same δ as sequential rank-K?"""
    S0, K_in, V, decay = _random_case(B, H, T, N, seed=seed, dtype=dtype)
    S_traj = sequential_e88_forward(S0, K_in, V, decay)
    S_var = S_traj[:, :, 1:] * 0.9
    A, b = build_AB(S0, S_var, K_in, V, decay)

    delta_seq = rank_k_sequential_scan(A, b, K)
    delta_tree = rank_k_tree_scan(A, b, K)

    diff = (delta_seq - delta_tree).abs().max().item()
    rel = diff / max(delta_seq.abs().max().item(), 1e-30)
    print(f"  T={T:4d} N={N:2d} K={K}  seq vs tree: abs={diff:.2e} rel={rel:.2e}")


def test_newton_tree_rank_k(B, H, T, N, K, seed=0, dtype=torch.float64,
                             max_iters=20, tol=1e-5):
    """Does TREE-scan rank-K Newton converge?"""
    S0, K_in, V, decay = _random_case(B, H, T, N, seed=seed, dtype=dtype)
    S_seq = sequential_e88_forward(S0, K_in, V, decay)[:, :, 1:]
    S_var = S_seq * 0.9 + 0.02 * torch.randn_like(S_seq)

    d_max = None
    for it in range(max_iters):
        delta = newton_step_rank_k(S0, S_var, K_in, V, decay, K, use_tree=True)
        S_var = S_var + delta
        d_max = delta.abs().max().item()
        if d_max < tol:
            break

    diff = (S_var - S_seq).abs().max().item()
    expected_tol = max(1e-3, 1e-5 * T)
    status = "PASS" if diff < expected_tol else "FAIL"
    print(f"  (tree) T={T:4d} N={N:2d} K={K}  iters={it+1} dmax={d_max:.1e}  "
          f"max|S_tree-S_seq|={diff:.2e}  [{status}]")


def test_truncation_accuracy(B, H, T, N, K, seed=0, dtype=torch.float64):
    """How close is rank-K truncated δ to full-matrix δ?"""
    S0, K_in, V, decay = _random_case(B, H, T, N, seed=seed, dtype=dtype)
    S_traj = sequential_e88_forward(S0, K_in, V, decay)
    S_var = S_traj[:, :, 1:] * 0.9

    A, b = build_AB(S0, S_var, K_in, V, decay)

    delta_full = full_rank_sequential_reference(A, b)
    delta_k = rank_k_sequential_scan(A, b, K)

    abs_err = (delta_full - delta_k).abs().max().item()
    rel_err = abs_err / max(delta_full.abs().max().item(), 1e-30)
    print(f"  T={T:4d} N={N:2d} K={K}  abs={abs_err:.2e}  rel={rel_err:.2e}")
    return abs_err, rel_err


def test_newton_convergence_with_rank_k(B, H, T, N, K, seed=0, dtype=torch.float64,
                                         max_iters=10, tol=1e-5):
    """Does rank-K Newton still converge to sequential forward?"""
    S0, K_in, V, decay = _random_case(B, H, T, N, seed=seed, dtype=dtype)
    S_seq = sequential_e88_forward(S0, K_in, V, decay)[:, :, 1:]
    S_var = S_seq * 0.9 + 0.02 * torch.randn_like(S_seq)

    d_max = None
    for it in range(max_iters):
        delta = newton_step_rank_k(S0, S_var, K_in, V, decay, K)
        S_var = S_var + delta
        d_max = delta.abs().max().item()
        if d_max < tol:
            break

    diff = (S_var - S_seq).abs().max().item()
    expected_tol = max(1e-4, 1e-5 * T)
    status = "PASS" if diff < expected_tol else "FAIL"
    print(f"  T={T:4d} N={N:2d} K={K}  iters={it+1} final_dmax={d_max:.1e}  "
          f"max|S_k-S_seq|={diff:.2e}  [{status}]")


if __name__ == '__main__':
    print("Phase 0b — rank-K TREE SCAN with truncation at each level:\n")
    print("Tree scan produces same δ as sequential (at same rank)?\n")
    for N in [16, 32]:
        for K in [2, 4, N // 2, N]:
            for T in [64, 256, 1024]:
                test_tree_vs_sequential_rank_k(1, 2, T, N, K)
            print()

    print("\nTree-scan Newton convergence (the real test for 5× path):\n")
    for N in [16, 32]:
        for K in [2, 4, N // 2, N]:
            for T in [256, 1024, 4096]:
                test_newton_tree_rank_k(1, 2, T, N, K)
            print()
