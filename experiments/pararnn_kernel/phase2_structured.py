"""Phase 2 — Structured propagation with rank-r truncation.

We represent each step's affine transform (J, b) where J = D − U·Vᵀ is a
diagonal-plus-low-rank matrix. Composition of two steps stays diagonal plus
low-rank, but rank grows additively. We cap total rank at r via SVD
truncation after each combine.

Key question answered here: how small can r be for the overall scan output
to match Phase 1 within float precision?

We scan the affine transforms (not the Jacobians alone), so output is
{δ_t} = solution of (I − J)·δ = −r, at every t, from a single prefix
scan over time. Then the Newton update S ← S + δ is done as in Phase 1.
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1_reference import (
    e88_row_step, sequential_trajectory, jacobian_dense_full, _random_inputs
)


# -----------------------------------------------------------------------------
# Structured affine state  (D, U, V, b)
#   J = diag(D) − U Vᵀ    (D: [n];  U, V: [n, r])
#   b: [n]
# -----------------------------------------------------------------------------

def _make_structured_step(s_prev, k, v_i, decay, r_step):
    """Build (D, U, V, b) for one step's affine transform.

    Recurrence:  δ_t = J_t · δ_{t-1} − r_step
    where J_t has diag+rank-1 structure derived from the current guess
    s_prev at timestep t.

    Returns:
        D: [n],  U: [n, 1],  V: [n, 1],  b: [n]
    """
    n = s_prev.shape[0]
    delta_val = v_i - torch.dot(s_prev, k)
    pre = decay * s_prev + delta_val * k
    tanh_deriv = 1.0 - torch.tanh(pre) ** 2           # [n]

    D = decay * tanh_deriv                             # [n]
    U = (tanh_deriv * k).unsqueeze(1)                  # [n, 1]
    V = k.unsqueeze(1).clone()                         # [n, 1]
    b = -r_step                                        # [n]
    return D, U, V, b


def _combine(D_a, U_a, V_a, b_a, D_b, U_b, V_b, b_b, r_max):
    """Compose (J_b, b_b) ∘ (J_a, b_a)  — apply A then B.

      J_new = J_b J_a
      b_new = J_b b_a + b_b

    Truncate the low-rank part of J_new to rank ≤ r_max.
    """
    n = D_a.shape[0]

    # Σ = V_bᵀ U_a        shape [r_b, r_a]
    Sigma = V_b.T @ U_a

    # diagonal stays diagonal
    D_new = D_b * D_a

    # expanded U, V (rank r_a + r_b before truncation)
    U_new = torch.cat([D_b.unsqueeze(1) * U_a - U_b @ Sigma, U_b], dim=1)
    V_new = torch.cat([V_a, D_a.unsqueeze(1) * V_b], dim=1)

    # bias update:  b_new = D_b · b_a − U_b (V_bᵀ b_a) + b_b
    b_new = D_b * b_a - U_b @ (V_b.T @ b_a) + b_b

    # SVD-truncate U_new Vᵀ_new to rank r_max (if needed)
    if U_new.shape[1] > r_max:
        # Reduced QR to avoid full n×n materialization
        Q_u, R_u = torch.linalg.qr(U_new, mode='reduced')   # [n, r], [r, r]
        Q_v, R_v = torch.linalg.qr(V_new, mode='reduced')
        M = R_u @ R_v.T                                      # [r, r]
        Us, S, Vh = torch.linalg.svd(M, full_matrices=False)
        # keep top r_max singular values
        S_sqrt = torch.sqrt(S[:r_max].clamp_min(0))
        U_trunc = Q_u @ (Us[:, :r_max] * S_sqrt)
        V_trunc = Q_v @ (Vh[:r_max, :].T * S_sqrt)
        U_new, V_new = U_trunc, V_trunc

    return D_new, U_new, V_new, b_new


def _scan_structured(steps, r_max):
    """Prefix-scan the list of structured affine transforms.

    Args:
        steps: list of T items, each (D, U, V, b).
        r_max: truncate U·Vᵀ to rank ≤ r_max after each combine.

    Returns:
        List of T prefix-products, same format. Item t is A_t ∘ A_{t-1} ∘ … ∘ A_1.
    """
    prefix = [steps[0]]
    for t in range(1, len(steps)):
        prev = prefix[-1]
        prefix.append(_combine(*prev, *steps[t], r_max=r_max))
    return prefix


def _apply_affine_to_zero(D, U, V, b):
    """Apply (J, b) to [δ=0; 1] — result is just b."""
    return b  # [n]


# -----------------------------------------------------------------------------
# Newton solver using structured scan
# -----------------------------------------------------------------------------

def newton_scan_structured(S0_row, K, V_i, decay, *, r_max, max_iters=20,
                           tol=1e-8, initial_guess=None, verbose=False):
    """Same as Phase 1's newton_scan_dense, but computes δ via a structured scan.

    With r_max ≥ n·T, this reduces to the dense case (modulo SVD noise).
    With r_max small, we measure how much truncation error accumulates.
    """
    T, n = K.shape
    device = K.device
    dtype = S0_row.dtype

    if initial_guess is not None:
        S_var = initial_guess.clone()
    else:
        S_var = torch.zeros(T, n, dtype=dtype, device=device)

    def compute_residuals(S_var):
        r = torch.empty_like(S_var)
        r[0] = S_var[0] - e88_row_step(S0_row, K[0], V_i[0], decay[0])
        for t in range(1, T):
            r[t] = S_var[t] - e88_row_step(S_var[t - 1], K[t], V_i[t], decay[t])
        return r

    for it in range(max_iters):
        r = compute_residuals(S_var)
        res_norm = r.abs().max().item()
        if verbose:
            print(f"  Newton iter {it}: max|r| = {res_norm:.3e}")
        if res_norm < tol:
            break

        # Build structured per-step affine transforms.
        # Step t: δ_t = J_t δ_{t-1} − r_t, where J_t depends on S_{t-1}.
        # For t=1 we use the fixed S_0.
        steps = []
        for t in range(T):
            s_prev = S0_row if t == 0 else S_var[t - 1]
            D, U, V, b = _make_structured_step(s_prev, K[t], V_i[t], decay[t],
                                               r_step=r[t])
            steps.append((D, U, V, b))

        # Prefix-scan: prefix[t] represents A_t ∘ ... ∘ A_1.
        prefix = _scan_structured(steps, r_max=r_max)

        # Apply each prefix to [0; 1] to get δ_t — which is just the bias.
        delta = torch.stack([_apply_affine_to_zero(*p) for p in prefix])

        S_var = S_var + delta

    S = torch.empty(T + 1, n, dtype=dtype, device=device)
    S[0] = S0_row
    S[1:] = S_var
    return S, it + 1, res_norm


# -----------------------------------------------------------------------------
# Test harness — measure error vs Phase 1 as a function of r
# -----------------------------------------------------------------------------

def sweep_rank(T, n, seed=0, dtype=torch.float64, ranks=(1, 2, 4, 8, 16)):
    """For given shape, run structured scan at several r values; report error vs Phase 1 dense."""
    from phase1_reference import newton_scan_dense

    S0_row, K, V_i, decay = _random_inputs(T, n, seed=seed, dtype=dtype)
    S_dense, iters_d, res_d = newton_scan_dense(S0_row, K, V_i, decay, tol=1e-12)

    print(f"\nShape T={T}, n={n}, dtype={str(dtype).replace('torch.','')}, seed={seed}")
    print(f"  Phase 1 (dense): iters={iters_d}, residual={res_d:.2e}")
    print(f"  {'r_max':>6s}   {'iters':>5s}   {'residual':>12s}   {'max|dense-struct|':>18s}")

    results = {}
    for r_max in ranks:
        S_struct, iters_s, res_s = newton_scan_structured(
            S0_row, K, V_i, decay, r_max=r_max, tol=1e-12)
        diff = (S_dense - S_struct).abs().max().item()
        print(f"  {r_max:>6d}   {iters_s:>5d}   {res_s:>12.2e}   {diff:>18.3e}")
        results[r_max] = {'iters': iters_s, 'res': res_s, 'diff': diff}
    return results


def sweep_rank_amplitude(T, n, amplitude=0.3, seed=0, dtype=torch.float64,
                         ranks=(1, 2, 4, 8, 16)):
    """Stress test with bigger-amplitude inputs to probe whether low rank still suffices."""
    from phase1_reference import newton_scan_dense

    g = torch.Generator(device='cpu').manual_seed(seed)
    S0_row = 0.1 * torch.randn(n, generator=g, dtype=dtype)
    K = amplitude * torch.randn(T, n, generator=g, dtype=dtype)
    V_i = amplitude * torch.randn(T, generator=g, dtype=dtype)
    decay = 0.9 + 0.1 * torch.rand(T, generator=g, dtype=dtype)

    S_dense, iters_d, res_d = newton_scan_dense(S0_row, K, V_i, decay, tol=1e-12)

    print(f"\nShape T={T}, n={n}, amplitude={amplitude}, seed={seed}")
    print(f"  Phase 1 (dense): iters={iters_d}, residual={res_d:.2e}")
    print(f"  {'r_max':>6s}   {'iters':>5s}   {'residual':>12s}   {'max|dense-struct|':>18s}")

    for r_max in ranks:
        S_struct, iters_s, res_s = newton_scan_structured(
            S0_row, K, V_i, decay, r_max=r_max, tol=1e-12)
        diff = (S_dense - S_struct).abs().max().item()
        print(f"  {r_max:>6d}   {iters_s:>5d}   {res_s:>12.2e}   {diff:>18.3e}")


if __name__ == '__main__':
    # Small shapes: expect tiny errors even at r=1
    sweep_rank(T=32,  n=4,  seed=0, ranks=(1, 2, 4, 8))
    sweep_rank(T=128, n=8,  seed=0, ranks=(1, 2, 4, 8, 16))

    # Target E88 shape: n=32, moderate T
    sweep_rank(T=256, n=32, seed=0, ranks=(1, 2, 4, 8, 16, 32))
    sweep_rank(T=512, n=32, seed=0, ranks=(1, 2, 4, 8, 16, 32))

    # Stress: larger input amplitude (drives Jacobians further from identity)
    sweep_rank_amplitude(T=512, n=32, amplitude=1.0,  ranks=(1, 2, 4, 8, 16, 32))
    sweep_rank_amplitude(T=512, n=32, amplitude=3.0,  ranks=(1, 2, 4, 8, 16, 32))

    # Multiple seeds — is the low rank robust?
    for s in [1, 2, 3]:
        sweep_rank(T=512, n=32, seed=s, ranks=(1, 2, 4, 8))

    print("\nPhase 2: rank-truncation error curve measured above.")
    print("Look for the smallest r where error is within an order of magnitude")
    print("of Phase 1's dense-scan residual tolerance (1e-10 at fp64).")
