"""Can warm-start reduce quasi-DEER iters?

If we initialize S_var with a cheap LINEAR approximation (apply the recurrence
without the nonlinearity), Newton should need fewer iters to refine.
"""

import sys, os
import torch

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quasi_deer_ref import _random_case, build_diag_ingredients, diag_scan_sequential
from phase4_newton_driver import sequential_e88_forward


def linear_forward_warmstart(S0, K, V, decay):
    """Compute E88 forward WITHOUT the final tanh nonlinearity.

    S_lin[t+1] = decay[t] * S_lin[t] + (V - S_lin[t] @ K) outer K
               (same update, but no tanh wrap)

    This is still sequential, so only useful if much faster than tanh-forward,
    OR if we can parallelize it via linear recurrence (Mamba2-style).
    """
    B, H, T, N = K.shape
    S = torch.empty(B, H, T + 1, N, N, dtype=S0.dtype, device=S0.device)
    S[:, :, 0] = S0
    for t in range(T):
        S_prev = S[:, :, t]
        retrieved = torch.einsum('bhij,bhj->bhi', S_prev, K[:, :, t])
        delta = V[:, :, t] - retrieved
        outer = torch.einsum('bhi,bhj->bhij', delta, K[:, :, t])
        pre = decay[:, :, t, None, None] * S_prev + outer
        S[:, :, t + 1] = pre  # NO TANH
    return S


def quasi_deer_from_warmstart(S0, K, V, decay, S_var_init, max_iters=30, tol=1e-4):
    from quasi_deer_triton import qd_diagonal_scan_triton
    B, H, T, N = K.shape
    S_var = S_var_init.clone()

    history = []
    for it in range(max_iters):
        D, b_vec, r_norm = build_diag_ingredients(S0, S_var, K, V, decay)
        history.append(r_norm)
        if r_norm < tol:
            break
        delta = qd_diagonal_scan_triton(D, b_vec, block_T=512)
        S_var = S_var + delta
    return S_var, it + 1, r_norm, history


if __name__ == '__main__':
    print("Can warm-start with LINEAR recurrence reduce quasi-DEER iters?\n")

    # N=16 case
    for H, T, N in [(16, 1024, 16), (16, 4096, 16)]:
        S0, K, V, decay = _random_case(1, H, T, N, seed=0, dtype=torch.float32,
                                        l2_normalize_k=True, v_scale=0.3)
        # Linear warm start
        S_lin = linear_forward_warmstart(S0, K, V, decay)[:, :, 1:]
        # Wrap with tanh to match the fixed point domain (tanh of the linear state)
        S_lin_tanh = torch.tanh(S_lin)

        # Cold start (zeros)
        _, it0, r0, hist0 = quasi_deer_from_warmstart(
            S0, K, V, decay, torch.zeros_like(S_lin), max_iters=40, tol=1e-4)
        # Warm start (linear + tanh)
        _, it1, r1, hist1 = quasi_deer_from_warmstart(
            S0, K, V, decay, S_lin_tanh, max_iters=40, tol=1e-4)
        # Warm start (linear, no tanh)
        _, it2, r2, hist2 = quasi_deer_from_warmstart(
            S0, K, V, decay, S_lin, max_iters=40, tol=1e-4)

        # Also try: tanh of the linear at each step, but using ACTUAL sequential
        # forward as "oracle" warm start to see theoretical minimum
        S_true = sequential_e88_forward(S0, K, V, decay)[:, :, 1:]
        _, it3, r3, hist3 = quasi_deer_from_warmstart(
            S0, K, V, decay, S_true, max_iters=40, tol=1e-4)

        print(f"H={H:3d} T={T:5d} N={N:2d}")
        print(f"  cold start  (zeros)          : {it0:3d} iters  final_res={r0:.2e}  init_res={hist0[0]:.2e}")
        print(f"  linear+tanh warm start       : {it1:3d} iters  final_res={r1:.2e}  init_res={hist1[0]:.2e}")
        print(f"  linear (no tanh) warm start  : {it2:3d} iters  final_res={r2:.2e}  init_res={hist2[0]:.2e}")
        print(f"  oracle warm start (true S)   : {it3:3d} iters  final_res={r3:.2e}  init_res={hist3[0]:.2e}")
