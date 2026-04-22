"""Phase 4 — Newton driver wrapping the Phase 3b Triton scan.

Given E88 per-row inputs (K, V, decay, S_0) for a full batched workload,
run Newton iterations using Triton scan inside. Output: the full state
trajectory [B, H, T+1, n, n].

Validation: match a per-(B, H, row) independent Phase 1 solution exactly
(up to fp32 rounding).

Step 1: batched Newton driver, tests against Phase 1 baseline.
Step 2: integrate with real E88 layer (separate file).
"""

import sys
import os
import math

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1_reference import newton_scan_dense
from phase3b_sequential_kernel import scan_r1_triton_batched


# -----------------------------------------------------------------------------
# Vectorized (B, H, row, T) residual and per-step ingredient builders.
# -----------------------------------------------------------------------------

def e88_step_batched(S_prev, K, V, decay):
    """Apply one E88 step across all (B, H, row=n) in parallel.

    Args:
      S_prev: [B, H, n, n]   — previous state (row × col layout).
      K:      [B, H, n]      — key (shared across rows).
      V:      [B, H, n]      — value (one scalar per row).
      decay:  [B, H]         — decay (shared across rows).

    Returns:
      S_new:  [B, H, n, n]
    """
    # retrieved[b,h,i] = S[b,h,i,:] · K[b,h,:]
    retrieved = torch.einsum('bhij,bhj->bhi', S_prev, K)
    delta = V - retrieved                                 # [B, H, n]
    outer = torch.einsum('bhi,bhj->bhij', delta, K)      # [B, H, n, n]
    pre = decay[..., None, None] * S_prev + outer
    return torch.tanh(pre)


def compute_residuals_batched(S0, S_var, K, V, decay):
    """Residuals r[t] = S_var[t] − f(S_prev, x_t) for all (b, h, t, row).

    All t computations are independent given current S_var, so we do them
    in one vectorized pass across T.

    Args:
      S0:     [B, H, n, n]
      S_var:  [B, H, T, n, n]
      K, V:   [B, H, T, n]
      decay:  [B, H, T]

    Returns:
      r: [B, H, T, n, n]
    """
    B, H, T, n, _ = S_var.shape

    # Build S_prev_all[t] = S_var[t-1] for t≥1, else S0
    S_prev_all = torch.empty_like(S_var)
    S_prev_all[:, :, 0] = S0
    S_prev_all[:, :, 1:] = S_var[:, :, :-1]

    # Vectorized forward step over the T axis
    retrieved = torch.einsum('bhtij,bhtj->bhti', S_prev_all, K)     # [B,H,T,n]
    delta = V - retrieved
    outer = torch.einsum('bhti,bhtj->bhtij', delta, K)              # [B,H,T,n,n]
    pre = decay[..., None, None] * S_prev_all + outer
    f_of_prev = torch.tanh(pre)

    return S_var - f_of_prev


def build_step_ingredients(S0, S_var, K, V, decay, r):
    """Build (D, u, v, b) per (b, h, t, row) for the structured scan.

    S_prev at step t: for t=0 use S0, else S_var[t-1].

    Returns each tensor of shape [B, H, T, n, n].
    """
    B, H, T, n, _ = S_var.shape

    # Build S_prev: shift-right along T, with S0 prepended
    S_prev_t = torch.empty_like(S_var)
    S_prev_t[:, :, 0] = S0
    S_prev_t[:, :, 1:] = S_var[:, :, :-1]

    # delta = V − S_prev @ K  →  [B, H, T, n]
    retrieved = torch.einsum('bhtij,bhtj->bhti', S_prev_t, K)
    delta = V - retrieved

    # pre = decay·S_prev + delta⊗K  →  [B, H, T, n, n]
    outer = torch.einsum('bhti,bhtj->bhtij', delta, K)
    pre = decay[..., None, None] * S_prev_t + outer
    tanh_deriv = 1.0 - torch.tanh(pre) ** 2               # [B, H, T, n, n]

    # D[b,h,t,row,col] = decay[b,h,t] · tanh_deriv[b,h,t,row,col]
    D = decay[..., None, None] * tanh_deriv

    # u[b,h,t,row,col] = tanh_deriv[b,h,t,row,col] · K[b,h,t,col]
    #   (the same for every row — but we still materialize full [..,n,n] for scan input shape)
    u = tanh_deriv * K[..., None, :]

    # v[b,h,t,row,col] = K[b,h,t,col]  (shared across rows)
    v = K[..., None, :].expand_as(u).contiguous()

    # b = −r
    b = -r

    return D, u, v, b


# -----------------------------------------------------------------------------
# Wrap the Triton scan for 5D inputs.
# -----------------------------------------------------------------------------

def scan_r1_triton_5d(D, u, v, b):
    """Run Phase 3b Triton scan on [B, H, T, n, n] inputs.

    Each (B, H, row) triple is an independent scan of length T with state
    dim n. We flatten (B, H, row) → single batch dim of size B*H*n.
    """
    B, H, T, n, n2 = D.shape
    assert n == n2
    # Move T to axis -2: [B, H, n, T, n]   (row before T)
    def flat(x):
        return x.permute(0, 1, 3, 2, 4).reshape(B * H * n, T, n).contiguous()
    Df = flat(D); uf = flat(u); vf = flat(v); bf = flat(b)
    # Phase 3b kernel expects shape [B_ext, H_ext, T, N]. Feed as [B*H*n, 1, T, n].
    Df = Df.view(B * H * n, 1, T, n)
    uf = uf.view(B * H * n, 1, T, n)
    vf = vf.view(B * H * n, 1, T, n)
    bf = bf.view(B * H * n, 1, T, n)
    delta = scan_r1_triton_batched(Df, uf, vf, bf)          # [B*H*n, 1, T, n]
    # Unflatten: [B*H*n, T, n] → [B, H, n, T, n] → [B, H, T, n, n]
    delta = delta.view(B, H, n, T, n).permute(0, 1, 3, 2, 4).contiguous()
    return delta


# -----------------------------------------------------------------------------
# Newton solver (batched, Triton scan under the hood).
# -----------------------------------------------------------------------------

def newton_e88_triton(S0, K, V, decay, *, max_iters=20, tol=1e-5, verbose=False):
    """Newton iteration for E88 matrix-state recurrence, batched.

    Args:
      S0:    [B, H, n, n]
      K:     [B, H, T, n]
      V:     [B, H, T, n]
      decay: [B, H, T]

    Returns:
      S: [B, H, T+1, n, n]  (S[:, :, 0] = S0, S[:, :, 1:] = converged)
      iters_used, final_residual
    """
    B, H, T, n = K.shape
    device = K.device
    dtype = S0.dtype

    S_var = torch.zeros(B, H, T, n, n, dtype=dtype, device=device)

    for it in range(max_iters):
        r = compute_residuals_batched(S0, S_var, K, V, decay)
        res_norm = r.abs().max().item()
        if verbose:
            print(f"  iter {it}: max|r| = {res_norm:.3e}")
        if res_norm < tol:
            break
        D, u, v, b = build_step_ingredients(S0, S_var, K, V, decay, r)
        delta = scan_r1_triton_5d(D, u, v, b)
        S_var = S_var + delta

    S = torch.empty(B, H, T + 1, n, n, dtype=dtype, device=device)
    S[:, :, 0] = S0
    S[:, :, 1:] = S_var
    return S, it + 1, res_norm


# -----------------------------------------------------------------------------
# Test: compare to per-(B, H, row) independent Phase 1 solutions.
# -----------------------------------------------------------------------------

def sequential_e88_forward(S0, K, V, decay):
    """Reference: straight sequential E88 forward (no Newton)."""
    B, H, T, n = K.shape
    dev = K.device
    dt = S0.dtype
    S = torch.empty(B, H, T + 1, n, n, dtype=dt, device=dev)
    S[:, :, 0] = S0
    for t in range(T):
        S[:, :, t + 1] = e88_step_batched(
            S[:, :, t], K[:, :, t], V[:, :, t], decay[:, :, t]
        )
    return S


def test_newton(B, H, T, n, seed=0, dtype=torch.float32, device='cuda'):
    torch.manual_seed(seed)
    S0 = 0.1 * torch.randn(B, H, n, n, dtype=dtype, device=device)
    K  = 0.3 * torch.randn(B, H, T, n, dtype=dtype, device=device)
    V  = 0.3 * torch.randn(B, H, T, n, dtype=dtype, device=device)
    decay = 0.9 + 0.1 * torch.rand(B, H, T, dtype=dtype, device=device)

    # Reference: direct sequential forward
    S_seq = sequential_e88_forward(S0, K, V, decay)

    # Triton-Newton solution
    S_tri, iters, res = newton_e88_triton(S0, K, V, decay, tol=1e-4, max_iters=50)

    diff = (S_seq - S_tri).abs().max().item()
    status = "PASS" if diff < max(1e-3, T * 1e-5) else "FAIL"
    print(f"  B={B} H={H:3d} T={T:4d} n={n:3d}  "
          f"newton_iters={iters:2d} res={res:.2e}  "
          f"max|seq − triton|={diff:.3e}  [{status}]")
    return diff


def benchmark():
    """Time Newton-Triton vs direct sequential at full E88 scale."""
    import time
    device = 'cuda'
    dtype = torch.float32
    B, H, T, n = 4, 112, 512, 32
    torch.manual_seed(0)
    S0 = 0.1 * torch.randn(B, H, n, n, dtype=dtype, device=device)
    K  = 0.3 * torch.randn(B, H, T, n, dtype=dtype, device=device)
    V  = 0.3 * torch.randn(B, H, T, n, dtype=dtype, device=device)
    decay = 0.9 + 0.1 * torch.rand(B, H, T, dtype=dtype, device=device)

    # Warmup
    _ = newton_e88_triton(S0, K, V, decay)
    _ = sequential_e88_forward(S0, K, V, decay)
    torch.cuda.synchronize()

    # Newton-Triton
    t0 = time.time()
    for _ in range(5):
        S_tri, iters, _ = newton_e88_triton(S0, K, V, decay, tol=1e-4)
    torch.cuda.synchronize()
    newton_ms = (time.time() - t0) / 5 * 1000

    # Direct sequential (vectorized across B,H — this IS the E88 inference
    # path without the CUDA kernel, so it's a fair PyTorch baseline)
    t0 = time.time()
    for _ in range(5):
        S_seq = sequential_e88_forward(S0, K, V, decay)
    torch.cuda.synchronize()
    seq_ms = (time.time() - t0) / 5 * 1000

    print(f"\nTiming at B={B} H={H} T={T} n={n}:")
    print(f"  Newton-Triton: {newton_ms:.2f} ms  ({iters} iters)")
    print(f"  Sequential PyTorch (vectorized): {seq_ms:.2f} ms")
    print(f"  Ratio: {newton_ms / seq_ms:.2f}×  (Newton time / seq time)")


if __name__ == '__main__':
    print("Phase 4 step 1: batched Newton solver using Triton scan.")
    print("Compare Newton's final trajectory to direct sequential E88 forward.\n")

    for B, H, T, n in [(1,  1, 16,  4),
                       (1,  4, 32,  8),
                       (1, 16, 64, 16),
                       (2, 32, 128, 32),
                       (4, 112, 512, 32)]:   # full E88 scale
        test_newton(B, H, T, n)

    benchmark()
