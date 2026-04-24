"""Quasi-DEER Python reference for E88 recurrence.

Quasi-DEER (Gonzalez et al. 2024, arXiv:2407.19115) = DEER with DIAGONAL-only
Jacobian approximation. The rank-1 off-diagonal correction (-u v^T) in E88's
Jacobian is dropped. The Jacobian then factors as a pure diagonal and
composes trivially: log(T)-depth parallel scan of scalar multiplications.

For E88 per row i, the per-step Jacobian is
    A[t,i,:,:] = diag(D[t,i,:]) - u[t,i,:] v[t,:]^T
where
    D[t,i,c] = decay[t] * tanh'(pre[t,i,c])
    u[t,i,c] = tanh'(pre[t,i,c]) * K[t,c]
    v[t,c]   = K[t,c]

Quasi-DEER sets A_approx[t,i,:,:] = diag(D[t,i,:]). Then the Newton solve of
J (block-bidiagonal with I on diagonal, -A[t] subdiag) becomes

    delta[t] = A_approx[t] @ delta[t-1] + b[t]

which is a purely elementwise scan:
    d_cum[p, c] = D[t, p, c] * d_cum_prev[p, c] + b[t, p, c]

and each (p, c) is an INDEPENDENT length-T scalar prefix-scan.

  - combine: (D2, b2) . (D1, b1) = (D2*D1, D2*b1 + b2)
  - cumprod/cumsum style, log(T)-depth with associative scan.

This file: Python reference, correctness + iteration count study.
"""

import sys
import os
import time

import torch

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase4_newton_driver import sequential_e88_forward, compute_residuals_batched


# ---------------------------------------------------------------------------
# Quasi-DEER Newton iteration: diagonal-only Jacobian, parallel scan per (p,c).
# ---------------------------------------------------------------------------

def build_diag_ingredients(S0, S_var, K, V, decay):
    """Build per-step DIAGONAL Jacobian (D) and residual (-r) for Newton step.

    Inputs:
      S0:    [B, H, N, N]
      S_var: [B, H, T, N, N]  current Newton iterate (solve state)
      K, V:  [B, H, T, N]
      decay: [B, H, T]

    Returns:
      D: [B, H, T, N, N]  — diagonal Jacobian coefficient per (t, p, c)
         D[b,h,t,p,c] = decay[b,h,t] * (1 - tanh(pre[b,h,t,p,c])^2)
      b: [B, H, T, N, N]  — residual = -(S_var - f(S_prev))
      r_norm: scalar max|residual|
    """
    B, H, T, N, _ = S_var.shape

    # S_prev_all[t] = S_var[t-1] for t>=1 else S0
    S_prev_all = torch.empty_like(S_var)
    S_prev_all[:, :, 0] = S0
    S_prev_all[:, :, 1:] = S_var[:, :, :-1]

    retrieved = torch.einsum('bhtpq,bhtq->bhtp', S_prev_all, K)   # [B,H,T,N]
    delta_inner = V - retrieved                                   # [B,H,T,N]
    outer = torch.einsum('bhtp,bhtq->bhtpq', delta_inner, K)      # [B,H,T,N,N]
    pre = decay[..., None, None] * S_prev_all + outer              # [B,H,T,N,N]
    f_val = torch.tanh(pre)
    tanh_deriv = 1.0 - f_val * f_val

    D = decay[..., None, None] * tanh_deriv                        # [B,H,T,N,N]
    r = S_var - f_val                                              # [B,H,T,N,N]
    return D, -r, r.abs().max().item()


def diag_scan_sequential(D, b):
    """Sequential reference for diagonal affine-op prefix scan.

    delta[t] = D[t] * delta[t-1] + b[t], starting delta_{-1} = 0.

    Shapes: D, b both [B, H, T, N, N]. Scan axis is T; each (p, c) is
    independent.

    Returns delta: [B, H, T, N, N].
    """
    B, H, T, N, _ = D.shape
    delta = torch.empty_like(D)
    cur = torch.zeros(B, H, N, N, dtype=D.dtype, device=D.device)
    for t in range(T):
        cur = D[:, :, t] * cur + b[:, :, t]
        delta[:, :, t] = cur
    return delta


def diag_scan_parallel(D, b):
    """Hillis-Steele style parallel prefix for diagonal affine ops.

    Each (p, c) is independent: combine over the T axis is scalar.
    combine((D1,b1),(D2,b2)) = (D2*D1, D2*b1 + b2)
    """
    B, H, T, N, _ = D.shape
    # Work on (D_s, b_s) arrays, updating in Hillis-Steele fashion.
    D_s = D.clone()
    b_s = b.clone()
    d = 1
    while d < T:
        # Combine positions [d..T-1] with positions [0..T-1-d]
        D_left = D_s[:, :, :T - d]
        b_left = b_s[:, :, :T - d]
        D_right = D_s[:, :, d:T]
        b_right = b_s[:, :, d:T]
        D_new = D_right * D_left
        b_new = D_right * b_left + b_right

        D_s = D_s.clone()
        b_s = b_s.clone()
        D_s[:, :, d:T] = D_new
        b_s[:, :, d:T] = b_new
        d *= 2
    return b_s


def quasi_deer_newton(
    S0, K, V, decay,
    *,
    max_iters=20,
    tol=1e-5,
    use_parallel_scan=False,
    verbose=False,
):
    """Quasi-DEER Newton solve: diagonal Jacobian, parallel prefix-scan.

    Returns (S_traj [B,H,T+1,N,N], iters_used, final_res_norm, convergence_history).
    """
    B, H, T, N = K.shape
    dtype = S0.dtype
    device = S0.device
    S_var = torch.zeros(B, H, T, N, N, dtype=dtype, device=device)

    history = []
    for it in range(max_iters):
        D, b_vec, r_norm = build_diag_ingredients(S0, S_var, K, V, decay)
        history.append(r_norm)
        if verbose:
            print(f"    iter {it}: max|r| = {r_norm:.3e}")
        if r_norm < tol:
            break
        if use_parallel_scan:
            delta = diag_scan_parallel(D, b_vec)
        else:
            delta = diag_scan_sequential(D, b_vec)
        S_var = S_var + delta

    S = torch.empty(B, H, T + 1, N, N, dtype=dtype, device=device)
    S[:, :, 0] = S0
    S[:, :, 1:] = S_var
    return S, it + 1, r_norm, history


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

def _random_case(B, H, T, N, seed=0, dtype=torch.float32, device='cuda',
                 decay_sigmoid_bias=0.5, k_scale=0.3, l2_normalize_k=False,
                 v_scale=0.3):
    g = torch.Generator(device=device).manual_seed(seed)
    S0 = 0.1 * torch.randn(B, H, N, N, generator=g, dtype=dtype, device=device)
    K = k_scale * torch.randn(B, H, T, N, generator=g, dtype=dtype, device=device)
    if l2_normalize_k:
        K = K / (K.norm(dim=-1, keepdim=True) + 1e-6)
    V = v_scale * torch.randn(B, H, T, N, generator=g, dtype=dtype, device=device)
    # Sigmoid bias 0.5 gives decay ~ 0.62 on average — a realistic E88 pretrained range.
    decay = torch.sigmoid(
        decay_sigmoid_bias + 0.1 * torch.randn(B, H, T, generator=g, dtype=dtype, device=device)
    )
    return S0, K, V, decay


def test_scan_correctness():
    """Sequential and parallel diagonal scans must produce the same result."""
    print("Test: Sequential vs parallel diagonal scan (fp64, random D/b)")
    for B, H, T, N in [(1, 2, 16, 4), (1, 4, 64, 8), (1, 2, 128, 16), (1, 4, 512, 16)]:
        g = torch.Generator(device='cuda').manual_seed(0)
        D = 0.5 * torch.randn(B, H, T, N, N, generator=g, dtype=torch.float64, device='cuda')
        bv = 0.3 * torch.randn(B, H, T, N, N, generator=g, dtype=torch.float64, device='cuda')
        s_seq = diag_scan_sequential(D, bv)
        s_par = diag_scan_parallel(D, bv)
        diff = (s_seq - s_par).abs().max().item()
        status = "PASS" if diff < 1e-10 else "FAIL"
        print(f"  B={B} H={H:2d} T={T:4d} N={N}: max|seq-par|={diff:.2e}  [{status}]")


def test_convergence(B, H, T, N, seed=0, max_iters=30, tol=1e-4,
                     decay_sigmoid_bias=0.5, k_scale=0.3, dtype=torch.float32,
                     use_parallel_scan=True, quiet=False, l2_normalize_k=False,
                     v_scale=0.3):
    """Run quasi-DEER Newton, compare converged S to golden sequential forward."""
    S0, K, V, decay = _random_case(
        B, H, T, N, seed=seed, dtype=dtype,
        decay_sigmoid_bias=decay_sigmoid_bias, k_scale=k_scale,
        l2_normalize_k=l2_normalize_k, v_scale=v_scale,
    )
    # Golden: sequential E88 in fp32 (or whatever dtype)
    S_seq = sequential_e88_forward(S0, K, V, decay)

    S_qd, iters, res, history = quasi_deer_newton(
        S0, K, V, decay, max_iters=max_iters, tol=tol,
        use_parallel_scan=use_parallel_scan, verbose=False,
    )

    # Compare converged trajectory to golden
    diff = (S_qd - S_seq).abs().max().item()
    diff_rel = diff / max(S_seq.abs().max().item(), 1e-10)
    # Final residual already stored in `res`
    final_residual = res

    if not quiet:
        print(f"  B={B} H={H:3d} T={T:6d} N={N:3d}  "
              f"decay_bias={decay_sigmoid_bias:+.2f} k={k_scale:.2f}  "
              f"iters={iters:3d}  final_res={final_residual:.2e}  "
              f"max|qd-seq|={diff:.2e} (rel {diff_rel:.2e})")
        # First few residuals to see shape of convergence
        rr = '  '.join(f'{h:.1e}' for h in history[:6])
        print(f"    first residuals: {rr}")
    return iters, final_residual, diff, diff_rel, history


if __name__ == '__main__':
    print("=" * 72)
    print("Quasi-DEER Python reference study")
    print("=" * 72)

    test_scan_correctness()

    print("\nTest: Quasi-DEER convergence at small T (warmup, tighter tol)")
    for T in [64, 256, 1024]:
        test_convergence(1, 4, T, 16, tol=1e-5, max_iters=40)

    print("\nTest: Convergence at production shapes (H=141 N=16)")
    for T in [512, 4096, 16384]:
        test_convergence(1, 141, T, 16, tol=1e-4, max_iters=40,
                         decay_sigmoid_bias=0.5, k_scale=0.3)
    # N=32 case — use smaller H to avoid OOM at large T
    print("\nTest: Convergence at N=32 (scaled-down H to fit memory)")
    for T in [512, 4096, 16384]:
        test_convergence(1, 16, T, 32, tol=1e-4, max_iters=40,
                         decay_sigmoid_bias=0.5, k_scale=0.3)

    print("\nTest: sensitivity to decay (higher decay bias ⇒ stiffer recurrence)")
    for bias in [-1.0, 0.0, 1.0, 2.0]:
        test_convergence(1, 16, 1024, 16, tol=1e-4, max_iters=40,
                         decay_sigmoid_bias=bias, k_scale=0.3)

    print("\nTest: sensitivity to K scale (larger K ⇒ larger rank-1 term we're dropping)")
    for ks in [0.1, 0.3, 0.5, 0.8]:
        test_convergence(1, 16, 1024, 16, tol=1e-4, max_iters=40,
                         decay_sigmoid_bias=0.5, k_scale=ks)

    print("\nTest: tighter tol (what iters needed for 1e-6 residual?)")
    for tol in [1e-3, 1e-4, 1e-5, 1e-6]:
        test_convergence(1, 16, 1024, 16, tol=tol, max_iters=60,
                         decay_sigmoid_bias=0.5, k_scale=0.3)

    print("\nTest: L2-normalized K (realistic E88 production — ||K||=1)")
    # E88 uses L2-normalized K, so |K[c]| ~ 1/sqrt(N).
    print("  -- N=16 (|K[c]| ~ 0.25) --")
    for T in [512, 4096, 16384]:
        test_convergence(1, 16, T, 16, tol=1e-4, max_iters=50,
                         decay_sigmoid_bias=0.5, l2_normalize_k=True, v_scale=0.3)
    print("  -- N=32 (|K[c]| ~ 0.18) --")
    for T in [512, 4096]:
        test_convergence(1, 16, T, 32, tol=1e-4, max_iters=50,
                         decay_sigmoid_bias=0.5, l2_normalize_k=True, v_scale=0.3)
    print("  -- sensitivity to V scale with normalized K --")
    for v_scale in [0.1, 0.3, 0.5, 1.0]:
        test_convergence(1, 16, 1024, 16, tol=1e-4, max_iters=60,
                         decay_sigmoid_bias=0.5, l2_normalize_k=True, v_scale=v_scale)
