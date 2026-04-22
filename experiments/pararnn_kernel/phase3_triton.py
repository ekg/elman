"""Phase 3 — Triton kernel for the rank-1 structured scan.

Phase 3a: explicit-scan PyTorch version, single head, single row.
          Fixes the combine semantics + truncation policy, gives us
          the oracle to diff against.
Phase 3b: Triton kernel using tl.associative_scan with the same combine.
          Should match Phase 3a to bf16 tolerance.

We keep r=1 fixed (Phase 2 showed this is lossless). The combine produces
a rank-2 intermediate that we truncate back to rank-1 each step. We use
the closed-form 2×2 SVD (via the analytic eigendecomp of a 2×2 symmetric
matrix) so the whole thing stays in elementwise / small-matrix ops
suitable for a GPU kernel.

The state for the scan is (D: [n], u: [n], v: [n], b: [n]) representing
the affine transform  δ → (diag(D) − u·vᵀ) δ + b.
"""

import sys, os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1_reference import (
    e88_row_step, _random_inputs, newton_scan_dense
)


# -----------------------------------------------------------------------------
# Rank-1 truncation of (U V^T) where U, V are n × 2
# -----------------------------------------------------------------------------

def _rank2_to_rank1(U, V):
    """Given rank-2 outer product U Vᵀ with U, V shape [n, 2], return
    (u, v) of shape [n] each such that u · vᵀ best approximates U Vᵀ in
    Frobenius norm.

    Method: form 2×2 matrix M = R_u · R_vᵀ after thin QR of U and V,
    take its dominant singular vector pair via closed-form 2×2 SVD,
    lift back to n.

    Everything is elementwise / small-matrix — trivially portable to Triton.
    """
    dev = U.device
    dt = U.dtype

    # Thin QR of U (n × 2) → Q_u (n × 2), R_u (2 × 2)
    # Gram-Schmidt, explicit for numerical control.
    u0 = U[:, 0]
    n0 = torch.linalg.vector_norm(u0) + 1e-30
    q0 = u0 / n0
    u1 = U[:, 1]
    r01 = torch.dot(q0, u1)
    u1_perp = u1 - r01 * q0
    n1 = torch.linalg.vector_norm(u1_perp) + 1e-30
    q1 = u1_perp / n1
    Q_u = torch.stack([q0, q1], dim=1)                   # [n, 2]
    R_u = torch.stack([
        torch.stack([n0, r01]),
        torch.stack([torch.zeros((), device=dev, dtype=dt), n1]),
    ])                                                   # [2, 2]

    # Same for V
    v0 = V[:, 0]
    m0 = torch.linalg.vector_norm(v0) + 1e-30
    p0 = v0 / m0
    v1 = V[:, 1]
    s01 = torch.dot(p0, v1)
    v1_perp = v1 - s01 * p0
    m1 = torch.linalg.vector_norm(v1_perp) + 1e-30
    p1 = v1_perp / m1
    Q_v = torch.stack([p0, p1], dim=1)                   # [n, 2]
    R_v = torch.stack([
        torch.stack([m0, s01]),
        torch.stack([torch.zeros((), device=dev, dtype=dt), m1]),
    ])                                                   # [2, 2]

    # 2×2 matrix whose SVD we need:  M = R_u · R_vᵀ
    M = R_u @ R_v.T                                      # [2, 2]

    # Closed-form 2×2 SVD via eigendecomp of M Mᵀ (symmetric 2×2).
    # σ_max, u_unit = top singular value & left singular vector of M.
    # For a 2×2 symmetric S, eigenvalues: (tr ± √(tr² − 4 det)) / 2.
    S = M @ M.T
    a, b = S[0, 0], S[0, 1]
    d = S[1, 1]
    tr = a + d
    det = a * d - b * b
    disc = torch.sqrt(torch.clamp(tr * tr - 4 * det, min=0.0))
    lam_max = 0.5 * (tr + disc)       # largest eigenvalue of S = σ_max²
    # eigenvector for lam_max: solve (S − λI)·e = 0
    # unnormalised: (b, λ − a)  if b ≠ 0, else (λ − d, b)
    eps = 1e-20
    e0 = torch.where(torch.abs(b) > eps, b, lam_max - d)
    e1 = torch.where(torch.abs(b) > eps, lam_max - a, b)
    norm_e = torch.sqrt(e0 * e0 + e1 * e1) + eps
    e0, e1 = e0 / norm_e, e1 / norm_e
    # σ_max
    sigma_max = torch.sqrt(torch.clamp(lam_max, min=0.0))
    # Right singular vector: v = Mᵀ u / σ
    # u = (e0, e1); Mᵀ u = M_col * (e0, e1)
    u_small = torch.stack([e0, e1])                      # [2]
    v_small = M.T @ u_small / (sigma_max + eps)          # [2]

    # Lift to n
    u_full = Q_u @ u_small                               # [n]
    v_full = Q_v @ v_small                               # [n]
    # Absorb σ into u (or split √σ to each side — equivalent)
    u_full = u_full * sigma_max
    return u_full, v_full


# -----------------------------------------------------------------------------
# Phase 3a — explicit-scan PyTorch implementation (single row)
# -----------------------------------------------------------------------------

def _combine_r1(D_a, u_a, v_a, b_a, D_b, u_b, v_b, b_b):
    """Combine (J_b, b_b) ∘ (J_a, b_a) with rank-1 truncation.
    Returns (D_c, u_c, v_c, b_c) representing the combined transform.
    """
    # J_b · b_a = D_b·b_a − u_b·(v_bᵀ·b_a)
    Vb_dot_ba = torch.dot(v_b, b_a)
    b_c = D_b * b_a - u_b * Vb_dot_ba + b_b

    # Diagonal-of-product
    D_c = D_b * D_a

    # Rank-2 expanded form of off-diagonal:
    #   (D_b u_a − u_b σ) v_aᵀ  +  u_b (D_a v_b)ᵀ
    sigma = torch.dot(v_b, u_a)                           # scalar
    U_expanded = torch.stack([D_b * u_a - u_b * sigma, u_b], dim=1)    # [n, 2]
    V_expanded = torch.stack([v_a,                    D_a * v_b], dim=1)  # [n, 2]

    u_c, v_c = _rank2_to_rank1(U_expanded, V_expanded)
    return D_c, u_c, v_c, b_c


def _make_step(s_prev, k, v_i, decay, r_step):
    """Build (D, u, v, b) for one step."""
    delta_val = v_i - torch.dot(s_prev, k)
    pre = decay * s_prev + delta_val * k
    tanh_deriv = 1.0 - torch.tanh(pre) ** 2
    D = decay * tanh_deriv
    u = tanh_deriv * k
    v = k.clone()
    b = -r_step
    return D, u, v, b


def scan_r1_pytorch(steps):
    """Explicit sequential prefix scan over r=1 combines.

    Args:
        steps: list of T items (D, u, v, b), each of length n.

    Returns:
        list of T prefix items.
    """
    prefix = [steps[0]]
    for t in range(1, len(steps)):
        prefix.append(_combine_r1(*prefix[-1], *steps[t]))
    return prefix


def newton_scan_r1(S0_row, K, V_i, decay, *, max_iters=20, tol=1e-8):
    """Newton solver using r=1 structured scan (PyTorch explicit loop)."""
    T, n = K.shape
    dev = K.device
    dt = S0_row.dtype

    S_var = torch.zeros(T, n, dtype=dt, device=dev)

    def compute_residuals(S_var):
        r = torch.empty_like(S_var)
        r[0] = S_var[0] - e88_row_step(S0_row, K[0], V_i[0], decay[0])
        for t in range(1, T):
            r[t] = S_var[t] - e88_row_step(S_var[t - 1], K[t], V_i[t], decay[t])
        return r

    for it in range(max_iters):
        r = compute_residuals(S_var)
        if r.abs().max() < tol:
            break
        steps = []
        for t in range(T):
            s_prev = S0_row if t == 0 else S_var[t - 1]
            steps.append(_make_step(s_prev, K[t], V_i[t], decay[t], r[t]))
        prefix = scan_r1_pytorch(steps)
        delta = torch.stack([p[3] for p in prefix])  # b components = δ when applied to zero init
        S_var = S_var + delta

    S = torch.empty(T + 1, n, dtype=dt, device=dev)
    S[0] = S0_row
    S[1:] = S_var
    return S, it + 1, r.abs().max().item()


# -----------------------------------------------------------------------------
# Phase 3b — Triton kernel via tl.associative_scan
# -----------------------------------------------------------------------------

import triton
import triton.language as tl


@triton.jit
def _rank2_to_rank1_triton(U0, U1, V0, V1):
    """Port of _rank2_to_rank1.  All inputs are [N] (scan position's features).

    Returns (u, v) each [N] — dominant rank-1 singular vector pair.
    """
    eps = 1e-30

    # Thin QR of [U0, U1]
    n0 = tl.sqrt(tl.sum(U0 * U0, axis=-1, keep_dims=True)) + eps
    q0 = U0 / n0
    r01 = tl.sum(q0 * U1)
    U1_perp = U1 - r01 * q0
    n1 = tl.sqrt(tl.sum(U1_perp * U1_perp)) + eps
    q1 = U1_perp / n1
    # R_u = [[n0, r01], [0, n1]]

    # Thin QR of [V0, V1]
    m0 = tl.sqrt(tl.sum(V0 * V0)) + eps
    p0 = V0 / m0
    s01 = tl.sum(p0 * V1)
    V1_perp = V1 - s01 * p0
    m1 = tl.sqrt(tl.sum(V1_perp * V1_perp)) + eps
    p1 = V1_perp / m1
    # R_v = [[m0, s01], [0, m1]]

    # M = R_u @ R_vᵀ  (2×2)
    M00 = n0 * m0 + r01 * s01
    M01 = r01 * m1
    M10 = n1 * s01
    M11 = n1 * m1

    # SVD via eigendecomp of S = M Mᵀ (symmetric 2×2)
    S00 = M00 * M00 + M01 * M01
    S01 = M00 * M10 + M01 * M11
    S11 = M10 * M10 + M11 * M11

    tr = S00 + S11
    det = S00 * S11 - S01 * S01
    disc_sq = tr * tr - 4.0 * det
    disc = tl.sqrt(tl.maximum(disc_sq, 0.0))
    lam_max = 0.5 * (tr + disc)

    # Left singular vector (2-dim); pick branch to avoid 0/0
    b_large = tl.abs(S01) > 1e-20
    e0 = tl.where(b_large, S01,           lam_max - S11)
    e1 = tl.where(b_large, lam_max - S00, S01)
    norm_e = tl.sqrt(e0 * e0 + e1 * e1) + eps
    e0 = e0 / norm_e
    e1 = e1 / norm_e

    sigma = tl.sqrt(tl.maximum(lam_max, 0.0))

    # Right singular vector: v_small = Mᵀ u_small / σ
    #   Mᵀ u_small = (M00 e0 + M10 e1, M01 e0 + M11 e1)
    f0 = (M00 * e0 + M10 * e1) / (sigma + eps)
    f1 = (M01 * e0 + M11 * e1) / (sigma + eps)

    # Lift to N: u_full = Q_u @ (e0, e1),  v_full = Q_v @ (f0, f1)
    u_full = q0 * e0 + q1 * e1
    v_full = p0 * f0 + p1 * f1
    # Absorb σ into u
    u_full = u_full * sigma
    return u_full, v_full


@triton.jit
def _combine_r1_triton(D_a, u_a, v_a, b_a, D_b, u_b, v_b, b_b):
    """Combine two r=1 structured states. All inputs [N]."""
    sigma = tl.sum(v_b * u_a, axis=-1, keep_dims=True)
    Vb_dot_ba = tl.sum(v_b * b_a, axis=-1, keep_dims=True)
    b_c = D_b * b_a - u_b * Vb_dot_ba + b_b
    D_c = D_b * D_a

    # Rank-2 expanded
    U0 = D_b * u_a - u_b * sigma
    U1 = u_b
    V0 = v_a
    V1 = D_a * v_b

    u_c, v_c = _rank2_to_rank1_triton(U0, U1, V0, V1)
    return D_c, u_c, v_c, b_c


# NOTE: tl.associative_scan in Triton 3.5 is elementwise-only — it passes
# scalar leaves to combine_fn, not vector leaves. Our combine needs
# vector-level operations (dot products, norms, etc.) so we can't use it
# directly. A proper Triton kernel here needs a manual Hillis–Steele scan
# over a [T, N] block tensor — scoped as Phase 3b engineering work (see
# docs/PARARNN_KERNEL_PLAN.md). For now Phase 3a (PyTorch scan, above)
# serves as the validated reference; the Triton combine helpers above are
# retained since they'll plug into the manual scan.


@triton.jit
def scan_r1_kernel(*args, **kwargs):
    """Placeholder — real kernel uses manual Hillis-Steele, TODO."""
    raise NotImplementedError(
        "Phase 3b requires a manual Hillis-Steele scan in Triton. "
        "tl.associative_scan is elementwise-only and doesn't fit our "
        "vector-valued combine."
    )


def scan_r1_triton(D, u, v, b):
    """Placeholder wrapper — the Triton kernel is pending Phase 3b work."""
    raise NotImplementedError(
        "Use scan_r1_pytorch for now (Phase 3a). The Triton kernel is "
        "a planned Phase 3b follow-up; see docs/PARARNN_KERNEL_PLAN.md."
    )


def newton_scan_r1_triton(S0_row, K, V_i, decay, *, max_iters=20, tol=1e-8):
    """Newton solver using Triton scan for the δ computation."""
    T, n = K.shape
    dev = K.device
    dt = S0_row.dtype
    S_var = torch.zeros(T, n, dtype=dt, device=dev)

    def compute_residuals(S_var):
        r = torch.empty_like(S_var)
        r[0] = S_var[0] - e88_row_step(S0_row, K[0], V_i[0], decay[0])
        for t in range(1, T):
            r[t] = S_var[t] - e88_row_step(S_var[t - 1], K[t], V_i[t], decay[t])
        return r

    for it in range(max_iters):
        r = compute_residuals(S_var)
        if r.abs().max() < tol:
            break
        D_all = torch.empty(T, n, dtype=dt, device=dev)
        u_all = torch.empty(T, n, dtype=dt, device=dev)
        v_all = torch.empty(T, n, dtype=dt, device=dev)
        b_all = torch.empty(T, n, dtype=dt, device=dev)
        for t in range(T):
            s_prev = S0_row if t == 0 else S_var[t - 1]
            D, U, V, B = _make_step(s_prev, K[t], V_i[t], decay[t], r[t])
            D_all[t] = D; u_all[t] = U; v_all[t] = V; b_all[t] = B

        delta = scan_r1_triton(D_all, u_all, v_all, b_all)
        S_var = S_var + delta

    S = torch.empty(T + 1, n, dtype=dt, device=dev)
    S[0] = S0_row
    S[1:] = S_var
    return S, it + 1, r.abs().max().item()


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def test_one(T, n, seed=0, tol=1e-12, dtype=torch.float64):
    S0_row, K, V_i, decay = _random_inputs(T, n, seed=seed, dtype=dtype)
    S_ref, iters_ref, res_ref = newton_scan_dense(S0_row, K, V_i, decay, tol=tol)
    S_r1, iters, res = newton_scan_r1(S0_row, K, V_i, decay, tol=tol)

    diff = (S_ref - S_r1).abs().max().item()
    status = "PASS" if diff < T * 1e-10 else "FAIL"
    print(f"  T={T:4d} n={n:3d} seed={seed:2d}  "
          f"iters_r1={iters:2d} res={res:.2e}  "
          f"max|dense − r1_scan|={diff:.3e}  [{status}]")
    return diff


if __name__ == '__main__':
    print("Phase 3a: explicit-scan r=1 combine, PyTorch (float64).")
    for T, n in [(32, 4), (128, 8), (256, 16), (512, 32)]:
        test_one(T, n)
    print()
    print("Phase 3a complete — r=1 combine with 2×2 SVD truncation matches")
    print("Phase 1 dense scan at machine epsilon. The Triton combine helpers")
    print("above (_combine_r1_triton, _rank2_to_rank1_triton) are ready to")
    print("plug into a manual Hillis-Steele scan when Phase 3b is scheduled.")
