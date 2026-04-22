"""Phase 1 — Pure-Python reference for E88's parallel Newton scan.

Status: unambiguous, dense, slow. Ground truth for all subsequent phases.

The key ideas:

1. E88's sequential recurrence on a single row:
     S_t[:] = tanh(decay_t · S_{t-1}[:] + (v_t − S_{t-1}@k_t) · k_t)
   Call this  S_t = f(S_{t-1}, x_t).

2. Given T timesteps, we solve the fixed-point system
     S_1 = f(S_0, x_1)
     S_2 = f(S_1, x_2)
     ...
     S_T = f(S_{T-1}, x_T)
   simultaneously via Newton's method. At each iteration we linearize
   around the current guess, producing a block-bidiagonal linear system
   that is solved by a prefix-scan over the per-step Jacobians.

3. The final Newton fixed point must equal the sequential trajectory.

This file does the whole Newton loop in dense PyTorch. No structure
exploitation, no truncation. The sole point is to be obviously correct.

Tests at the bottom compare the Newton-scan output to the reference
sequential trajectory on a synthetic E88 recurrence.
"""

import torch


# -----------------------------------------------------------------------------
# E88 recurrence — single row of the matrix state.
# -----------------------------------------------------------------------------

def e88_row_step(s_prev, k, v_i, decay):
    """One timestep of E88 for a single row of S.

    Args:
        s_prev: [n]    — previous row state.
        k:      [n]    — key at this timestep.
        v_i:    scalar — value component for this row (v[i]).
        decay:  scalar — decay at this timestep.

    Returns:
        s_new: [n] — new row state.
    """
    delta = v_i - torch.dot(s_prev, k)   # scalar
    pre = decay * s_prev + delta * k     # [n]
    return torch.tanh(pre)


def sequential_trajectory(S0_row, K, V_i, decay):
    """Reference sequential roll-out for one row.

    Args:
        S0_row: [n]       — initial row state.
        K:      [T, n]    — keys per timestep.
        V_i:    [T]       — row's value per timestep.
        decay:  [T]       — decay per timestep.

    Returns:
        S: [T+1, n] — states at t=0..T (t=0 is S0_row).
    """
    T, n = K.shape
    S = torch.empty(T + 1, n, dtype=S0_row.dtype, device=S0_row.device)
    S[0] = S0_row
    for t in range(T):
        S[t + 1] = e88_row_step(S[t], K[t], V_i[t], decay[t])
    return S


# -----------------------------------------------------------------------------
# Newton-based parallel-scan solver (dense, unstructured).
# -----------------------------------------------------------------------------

def jacobian_dense(s_prev, k, decay):
    """Dense Jacobian  J = ∂e88_row_step(S) / ∂S  evaluated at s_prev.

    Derived analytically:
        J = diag(decay · tanh'(pre)) − (tanh'(pre) ⊙ k) · k^T

    Args:
        s_prev: [n]
        k:      [n]
        decay:  scalar

    Returns:
        J: [n, n]
    """
    n = s_prev.shape[0]
    # Recompute pre to get tanh' without storing it from forward
    delta = s_prev.new_zeros(())        # v_i cancels in the derivative path
    # pre depends on v_i — but tanh'(pre) does depend on v_i!
    # Caller must pass full input. Let's redo this properly:
    raise NotImplementedError("pass v_i too; use jacobian_dense_full instead")


def jacobian_dense_full(s_prev, k, v_i, decay):
    """Full Jacobian given all inputs at this step."""
    n = s_prev.shape[0]
    delta = v_i - torch.dot(s_prev, k)
    pre = decay * s_prev + delta * k
    tanh_deriv = 1.0 - torch.tanh(pre) ** 2   # [n]

    # J = diag(decay · tanh')  −  (tanh' ⊙ k) k^T
    J = torch.diag(decay * tanh_deriv) - torch.outer(tanh_deriv * k, k)
    return J


def newton_scan_dense(S0_row, K, V_i, decay, *, max_iters=20, tol=1e-8,
                     initial_guess=None, verbose=False):
    """Solve the whole T-step trajectory in one Newton scan, dense.

    Args:
        S0_row: [n]       — fixed initial state (not a variable).
        K:      [T, n]
        V_i:    [T]
        decay:  [T]
        max_iters, tol: Newton convergence controls.
        initial_guess: [T, n] or None  — starting guess for S_1..S_T.
                                         None = start from zeros.

    Returns:
        S: [T+1, n]       — converged trajectory including S_0 at index 0.
        iters_used: int
        final_max_residual: float
    """
    T, n = K.shape
    device = K.device
    dtype = S0_row.dtype

    # Variables: S_1, ..., S_T  (T rows of length n)
    if initial_guess is not None:
        S_var = initial_guess.clone()
    else:
        S_var = torch.zeros(T, n, dtype=dtype, device=device)

    def compute_residuals(S_var):
        """r_t = S_t − f(S_{t-1}, x_t)  for t=1..T"""
        r = torch.empty_like(S_var)
        # t=1 uses fixed S_0
        r[0] = S_var[0] - e88_row_step(S0_row, K[0], V_i[0], decay[0])
        for t in range(1, T):
            r[t] = S_var[t] - e88_row_step(S_var[t - 1], K[t], V_i[t], decay[t])
        return r

    def compute_jacobians(S_var):
        """J_t = ∂f(S, x_t) / ∂S evaluated at S_{t-1}  for t=1..T"""
        Js = torch.empty(T, n, n, dtype=dtype, device=device)
        Js[0] = jacobian_dense_full(S0_row, K[0], V_i[0], decay[0])
        for t in range(1, T):
            Js[t] = jacobian_dense_full(S_var[t - 1], K[t], V_i[t], decay[t])
        return Js

    def solve_bidiagonal(Js, r):
        """Solve the block-bidiagonal system (I − J)·δ = −r.

        For the system
            δ_1 = −r_1
            δ_2 − J_2 δ_1 = −r_2
            δ_3 − J_3 δ_2 = −r_3
            ...
        forward-substitute:
            δ_t = J_t δ_{t−1} − r_t

        Equivalent to the prefix-product-weighted-sum formula, but this
        direct forward substitution is exact in dense arithmetic.
        """
        delta = torch.empty_like(r)
        delta[0] = -r[0]
        for t in range(1, T):
            delta[t] = Js[t] @ delta[t - 1] - r[t]
        return delta

    for it in range(max_iters):
        r = compute_residuals(S_var)
        res_norm = r.abs().max().item()
        if verbose:
            print(f"  Newton iter {it}: max|r| = {res_norm:.3e}")
        if res_norm < tol:
            break
        Js = compute_jacobians(S_var)
        delta = solve_bidiagonal(Js, r)
        S_var = S_var + delta

    S = torch.empty(T + 1, n, dtype=dtype, device=device)
    S[0] = S0_row
    S[1:] = S_var
    return S, it + 1, res_norm


# -----------------------------------------------------------------------------
# Test harness — lock the parallel scan output to the sequential reference.
# -----------------------------------------------------------------------------

def _random_inputs(T, n, seed=0, dtype=torch.float64, device='cpu'):
    g = torch.Generator(device=device).manual_seed(seed)
    S0_row = 0.1 * torch.randn(n, generator=g, dtype=dtype, device=device)
    K = 0.3 * torch.randn(T, n, generator=g, dtype=dtype, device=device)
    V_i = 0.3 * torch.randn(T, generator=g, dtype=dtype, device=device)
    decay = 0.9 + 0.1 * torch.rand(T, generator=g, dtype=dtype, device=device)
    return S0_row, K, V_i, decay


def test_one(T, n, seed=0, tol=1e-10, dtype=torch.float64, verbose=False):
    """Run sequential and Newton-scan on the same random inputs, compare."""
    S0_row, K, V_i, decay = _random_inputs(T, n, seed=seed, dtype=dtype)

    S_seq = sequential_trajectory(S0_row, K, V_i, decay)
    S_scan, iters, res = newton_scan_dense(S0_row, K, V_i, decay, tol=tol,
                                           max_iters=50, verbose=verbose)

    diff = (S_seq - S_scan).abs().max().item()
    print(f"T={T:4d} n={n:3d} seed={seed:2d} "
          f"dtype={str(dtype).replace('torch.',''):10s} "
          f"newton_iters={iters:3d} final_res={res:.2e} "
          f"max|seq - scan|={diff:.3e}")
    assert diff < T * 1e-10, (
        f"Phase 1 parallel-scan disagrees with sequential at T={T}, n={n}: "
        f"diff={diff:.3e} > {T*1e-10:.3e}"
    )
    return diff


if __name__ == '__main__':
    # Sweep across the shapes we care about. All in float64 for tight tol.
    # At float64, seq_len*machine_precision ≈ T * 2.2e-16.
    # Our assert tolerance T*1e-10 is conservative for Newton convergence.
    for T, n in [(32, 4), (128, 4), (128, 8), (256, 16), (512, 32)]:
        test_one(T, n, seed=0)
    print("\nPhase 1: all tests pass.")
