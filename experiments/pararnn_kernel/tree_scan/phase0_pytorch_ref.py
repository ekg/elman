"""Phase 0 — PyTorch reference for tree-scan Newton.

Goal: build full N×N affine ops (A, b), tree-scan them, extract δ,
and verify it matches our existing sequential Pararnn Newton iteration
to fp32 machine precision.

This locks down the math before kernel work.

Mathematical setup:

  Newton step solves J·δ = -r for the block-bidiagonal system:
    J[t, t]   = I             (N-row-size identity per row)
    J[t, t-1] = -∂f/∂S[t-1]

  Forward-substitute: δ[t] = A[t] · δ[t-1] + b[t]  where
    A[t] = ∂f/∂S[t-1] = diag(D[t]) - u[t] v[t]^T
    b[t] = -r[t]

  For E88, per row i:
    D[t, i, :] = decay[t] * tanh'(pre[t, i, :])
    u[t, i, :] = tanh'(pre[t, i, :]) * K[t, :]
    v[t, :]    = K[t, :]    (shared across rows)
    pre[t, i, :] = decay[t]*S_prev[t, i, :] + (V[t, i] - K[t]·S_prev[t, i, :]) · K[t]

  Each row has its own independent affine-op chain. Tree-scan per row.
"""

import sys
import os
import torch

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase4_newton_driver import sequential_e88_forward
from phase7_fused_iter import fused_newton_iter


# -----------------------------------------------------------------------------
# Step 1: Build the (A, b) tensors from Newton's current iterate.
# -----------------------------------------------------------------------------

def build_AB(S0, S_var, K, V, decay):
    """Build full-matrix affine operators per (B, H, t, row).

    Inputs:
      S0:    [B, H, N, N]
      S_var: [B, H, T, N, N]  current Newton iterate
      K:     [B, H, T, N]
      V:     [B, H, T, N]
      decay: [B, H, T]

    Returns:
      A: [B, H, T, N_row, N, N]  per (t, row) Jacobian matrix
      b: [B, H, T, N_row, N]     per (t, row) residual vector = -r
    """
    B, H, T, N, _ = S_var.shape

    # S_prev[t] = S[t-1] (= S0 when t=0)
    S_prev_all = torch.cat([S0.unsqueeze(2), S_var[:, :, :-1]], dim=2)  # [B,H,T,N,N]

    # Per-step forward quantities
    retrieved = torch.einsum('bhtij,bhtj->bhti', S_prev_all, K)  # [B, H, T, N]
    delta_inner = V - retrieved  # [B, H, T, N]
    outer = torch.einsum('bhti,bhtj->bhtij', delta_inner, K)  # [B, H, T, N, N]
    pre = decay[..., None, None] * S_prev_all + outer  # [B, H, T, N, N]
    f_val = torch.tanh(pre)
    tanh_deriv = 1.0 - f_val * f_val  # [B, H, T, N, N]

    # Residual r[t, i, :] = S_var[t, i, :] - f[t, i, :]
    r = S_var - f_val  # [B, H, T, N, N]

    # Per-row Jacobian components
    D = decay[..., None, None] * tanh_deriv  # [B, H, T, N_row, N]
    u = tanh_deriv * K[..., None, :]  # [B, H, T, N_row, N]  (broadcast K across rows)
    v = K                              # [B, H, T, N]

    # Build full A[t, i, :, :] = diag(D[t, i, :]) - u[t, i, :] · v[t, :]^T
    diag_D = torch.diag_embed(D)  # [B, H, T, N_row, N, N]
    uvT = u.unsqueeze(-1) * v[..., None, None, :]  # [B, H, T, N_row, N, N]
    A = diag_D - uvT  # [B, H, T, N_row, N, N]

    b = -r  # [B, H, T, N_row, N]

    return A, b


# -----------------------------------------------------------------------------
# Step 2: Sequential scan (sanity reference).
# -----------------------------------------------------------------------------

def affine_scan_sequential(A, b):
    """Sequential prefix scan of affine operators.

    Compose left-to-right: at time t, apply A[t] to running state then add b[t].
    Combine: (A_cum, b_cum) -> (A[t] @ A_cum, A[t] @ b_cum + b[t])

    Returns b_cum[t] which equals δ[t] starting from δ_start = 0.
    """
    B, H, T, N_row, N, _ = A.shape

    # Initialize: A_cum = I, b_cum = 0
    A_cum = torch.eye(N, dtype=A.dtype, device=A.device).expand(B, H, N_row, N, N).clone()
    b_cum = torch.zeros(B, H, N_row, N, dtype=b.dtype, device=b.device)

    delta_out = torch.empty_like(b)
    for t in range(T):
        A_t = A[:, :, t]  # [B, H, N_row, N, N]
        b_t = b[:, :, t]  # [B, H, N_row, N]
        # b_cum_new = A_t @ b_cum + b_t
        b_cum = torch.einsum('bhrij,bhrj->bhri', A_t, b_cum) + b_t
        # A_cum_new = A_t @ A_cum
        A_cum = torch.einsum('bhrij,bhrjk->bhrik', A_t, A_cum)
        delta_out[:, :, t] = b_cum

    return delta_out


# -----------------------------------------------------------------------------
# Step 3: Real tree scan (parallel prefix, O(log T) depth).
# -----------------------------------------------------------------------------

def affine_combine(A_left, b_left, A_right, b_right):
    """Compose (A_left, b_left) FIRST then (A_right, b_right).

    Result represents: x -> A_right @ (A_left @ x + b_left) + b_right
                        = (A_right @ A_left) @ x + (A_right @ b_left + b_right)
    """
    A_out = torch.einsum('...ij,...jk->...ik', A_right, A_left)
    b_out = torch.einsum('...ij,...j->...i', A_right, b_left) + b_right
    return A_out, b_out


def affine_scan_tree(A, b):
    """Hillis-Steele inclusive prefix scan (log T depth).

    For N positions, the Hillis-Steele scan has depth log2(N). At each
    level d, combine each position with the one 2^d to its left:
      x[t] <- x[t-2^d] ∘ x[t]  (if t >= 2^d)

    Non-in-place for clarity. This is the reference algorithm.
    """
    B, H, T, N_row, N, _ = A.shape

    A_s = A.clone()
    b_s = b.clone()

    d = 1
    while d < T:
        # For t >= d: combine(result[t-d], step[t])
        A_new = A_s.clone()
        b_new = b_s.clone()

        # indices for the "left" operand: t-d for t >= d
        # Combine at positions [d .. T-1]
        A_left = A_s[:, :, :T - d]   # [B, H, T-d, ...]
        b_left = b_s[:, :, :T - d]
        A_right = A_s[:, :, d:T]
        b_right = b_s[:, :, d:T]

        A_comb, b_comb = affine_combine(A_left, b_left, A_right, b_right)

        A_new[:, :, d:T] = A_comb
        b_new[:, :, d:T] = b_comb

        A_s = A_new
        b_s = b_new
        d *= 2

    return b_s  # inclusive prefix's b_cum


# -----------------------------------------------------------------------------
# Step 4: End-to-end Newton step via tree scan — compare to fused Pararnn.
# -----------------------------------------------------------------------------

def newton_step_treescan(S0, S_var, K, V, decay, use_tree=True):
    """One Newton iter via full-matrix tree scan. Returns δ."""
    A, b = build_AB(S0, S_var, K, V, decay)
    if use_tree:
        delta = affine_scan_tree(A, b)
    else:
        delta = affine_scan_sequential(A, b)
    return delta


def _random_case(B, H, T, n, seed=0, dtype=torch.float64, device='cuda'):
    g = torch.Generator(device=device).manual_seed(seed)
    S0 = 0.1 * torch.randn(B, H, n, n, generator=g, dtype=dtype, device=device)
    K = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dtype, device=device)
    V = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dtype, device=device)
    decay = 0.9 + 0.1 * torch.rand(B, H, T, generator=g, dtype=dtype, device=device)
    return S0, K, V, decay


def test_tree_vs_sequential(B, H, T, n, seed=0):
    """Tree scan and sequential scan must give bit-identical answers at fp64."""
    S0, K, V, decay = _random_case(B, H, T, n, seed=seed, dtype=torch.float64)
    S_traj = sequential_e88_forward(S0, K, V, decay)
    S_var = S_traj[:, :, 1:] * 0.9  # perturbed guess

    A, b = build_AB(S0, S_var, K, V, decay)

    delta_seq = affine_scan_sequential(A, b)
    delta_tree = affine_scan_tree(A, b)

    diff = (delta_seq - delta_tree).abs().max().item()
    status = "PASS" if diff < 1e-10 else "FAIL"
    print(f"  B={B} H={H:2d} T={T:4d} n={n:2d}  seq vs tree: max|Δ|={diff:.2e}  [{status}]")
    return diff


def test_full_matrix_vs_rank1(B, H, T, n, seed=0, dtype=torch.float32):
    """Full-matrix tree scan must match our rank-1 fused sequential scan
    (they compute the same Newton δ, proving the r=1 scan is lossless)."""
    S0, K, V, decay = _random_case(B, H, T, n, seed=seed, dtype=dtype)
    S_traj = sequential_e88_forward(S0, K, V, decay)
    S_var = S_traj[:, :, 1:] * 0.9

    # Full-matrix tree scan
    delta_tree = newton_step_treescan(S0, S_var, K, V, decay, use_tree=True)

    # Existing rank-1 fused Triton scan
    delta_r1 = fused_newton_iter(S0, S_var, K, V, decay)

    # Tolerance appropriate for fp32 with T steps of compounding:
    tol = max(1e-4, 3e-6 * T)
    diff = (delta_tree - delta_r1).abs().max().item()
    # The FULL-matrix answer is our "ground truth" — r=1 should match it.
    status = "PASS" if diff < tol else "FAIL"
    print(f"  B={B} H={H:2d} T={T:5d} n={n:2d}  full-tree vs r=1-fused: max|Δ|={diff:.2e}  "
          f"(tol {tol:.1e})  [{status}]")
    return diff


def test_newton_convergence_to_sequential(B, H, T, n, seed=0, dtype=torch.float32,
                                           max_iters=10, tol_conv=1e-5):
    """Iterate tree-scan Newton until converged, check against sequential forward."""
    S0, K, V, decay = _random_case(B, H, T, n, seed=seed, dtype=dtype)
    S_seq = sequential_e88_forward(S0, K, V, decay)[:, :, 1:]
    S_var = S_seq * 0.9 + 0.02 * torch.randn_like(S_seq)

    d_max = None
    for it in range(max_iters):
        delta = newton_step_treescan(S0, S_var, K, V, decay, use_tree=True)
        S_var = S_var + delta
        d_max = delta.abs().max().item()
        if d_max < tol_conv:
            break

    diff = (S_var - S_seq).abs().max().item()
    expected = max(1e-4, 3e-6 * T)
    status = "PASS" if diff < expected else "FAIL"
    print(f"  B={B} H={H:2d} T={T:5d} n={n:2d}  iters={it+1} dmax={d_max:.2e}  "
          f"max|newton − seq|={diff:.2e}  [{status}]")


if __name__ == '__main__':
    print("Test 1: Tree scan == sequential scan (same math, fp64):\n")
    for shape in [(1, 2, 8, 4), (1, 4, 32, 8), (1, 4, 64, 16), (1, 2, 128, 16)]:
        test_tree_vs_sequential(*shape)

    print("\nTest 2: Full-matrix tree scan == rank-1 fused Pararnn (fp32):\n")
    for shape in [(1, 4, 64, 16), (1, 8, 256, 32), (1, 16, 512, 32), (1, 32, 1024, 32)]:
        test_full_matrix_vs_rank1(*shape)

    print("\nTest 3: Newton with tree scan converges to sequential forward:\n")
    for shape in [(1, 4, 64, 16), (1, 16, 256, 32), (1, 32, 1024, 32)]:
        test_newton_convergence_to_sequential(*shape)
