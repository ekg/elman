"""Phase 7 — fused single-Newton-iter Triton kernel.

Per-iteration path fuses:
  1. Residual:   r[t] = S_var[t] − f(S_prev[t], x_t)
  2. Jacobian:   (D, u, v, b) from s_prev, k, v_i, decay, r
  3. Scan:       combine r=1 structured state left-to-right
  4. Output:     δ[t] = b-component of prefix at t

Instead of materializing [B, H, T, n, n] intermediates (D, u, v, b each!)
between these phases, we do everything per-row in a single Triton program.
Only inputs (S_var, K, V, decay, S0) are in gmem; δ out.

Shape: [B, H, T, n, n] for S_var, [B, H, T, n] for K/V, [B, H, T] for decay.
One program per (B, H, row=i). Grid = (B*H*n,).

Expected win:
  - Memory: no intermediate D, u, v, b, r materialization. Previously
    ~5× the S_var tensor; now just S_var + delta.
  - Speed: fewer kernel launches; less memory traffic.
"""

import sys
import os
import math
import time

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase3b_sequential_kernel import _combine_r1_triton, _rank2_to_rank1_triton
from phase4_newton_driver import (
    sequential_e88_forward, compute_residuals_batched,
    build_step_ingredients, scan_r1_triton_5d,
)
from phase1_reference import _random_inputs


@triton.jit
def _fused_newton_iter_kernel(
    S0_ptr,          # [B, H, n, n]
    S_var_ptr,       # [B, H, T, n, n]
    K_ptr,           # [B, H, T, n]
    V_ptr,           # [B, H, T, n]  — note V[:, :, :, i] is row i's value scalar
    decay_ptr,       # [B, H, T]
    delta_ptr,       # [B, H, T, n, n] (output)
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    N: tl.constexpr,
):
    """One program per (b, h, row). Fused residual + Jacobian + scan.

    For each t = 0..T-1:
      s_prev = S0[b,h,row,:] if t==0 else S_var[b,h,t-1,row,:]
      k_t = K[b,h,t,:]
      v_i = V[b,h,t,row]   (scalar)
      dec = decay[b,h,t]
      pre = dec * s_prev + (v_i - s_prev · k_t) * k_t
      f_val = tanh(pre)
      r = S_var[b,h,t,row,:] - f_val
      D = dec * tanh'(pre)
      u = tanh'(pre) * k_t
      v = k_t
      b_step = -r
      (combine into running prefix)
      store prefix's b  → delta[b,h,t,row,:]
    """
    pid = tl.program_id(0)
    n_idx = tl.arange(0, N)

    # Decode pid → (b, h, row)
    b_idx = pid // (H * N)
    h_idx = (pid // N) % H
    row = pid % N

    # Base offsets into the 5D tensors [B, H, T, n, n]
    #   flat = (((b*H + h)*T + t)*n + row)*n + col
    bh = b_idx * H + h_idx
    bh_T_nn = bh * T * N * N
    # Initial S0 row: [B, H, n, n] → (bh * n + row) * n
    s0_row_off = (bh * N + row) * N

    # K/V/decay offsets
    bh_T_n = bh * T * N
    bh_T = bh * T

    # Initialize prefix state = None (identity for combine)
    # For the first step t=0: build (D_0, u_0, v_0, b_0) and use directly as prefix.
    # Implementation: special-case t=0 outside the loop.

    # t = 0
    s_prev = tl.load(S0_ptr + s0_row_off + n_idx)                   # [N]
    K_t = tl.load(K_ptr + bh_T_n + 0 * N + n_idx)                   # [N]
    V_i = tl.load(V_ptr + bh_T_n + 0 * N + row)                     # scalar
    dec = tl.load(decay_ptr + bh_T + 0)                             # scalar

    retrieved = tl.sum(s_prev * K_t, axis=-1, keep_dims=True)        # scalar as [1]
    delta_scalar = V_i - retrieved
    pre = dec * s_prev + delta_scalar * K_t                          # [N]
    # tanh via exp — Triton 3.5 doesn't expose tanh directly, compute it.
    e2x = tl.exp(2.0 * pre)
    f_val = (e2x - 1.0) / (e2x + 1.0)
    # Read current S_var row
    S_var_row = tl.load(S_var_ptr + bh_T_nn + 0 * N * N + row * N + n_idx)
    r = S_var_row - f_val
    tanh_deriv = 1.0 - f_val * f_val
    D_p = dec * tanh_deriv
    u_p = tanh_deriv * K_t
    v_p = K_t
    b_p = -r

    # Store prefix's b for t=0 as delta[t=0]
    tl.store(delta_ptr + bh_T_nn + 0 * N * N + row * N + n_idx, b_p)

    # Sequential scan t = 1..T-1
    for t in range(1, T):
        # Previous state for this row is S_var[b, h, t-1, row, :]
        s_prev_off = bh_T_nn + (t - 1) * N * N + row * N
        s_prev = tl.load(S_var_ptr + s_prev_off + n_idx)
        K_t = tl.load(K_ptr + bh_T_n + t * N + n_idx)
        V_i = tl.load(V_ptr + bh_T_n + t * N + row)
        dec = tl.load(decay_ptr + bh_T + t)

        retrieved = tl.sum(s_prev * K_t, axis=-1, keep_dims=True)
        delta_scalar = V_i - retrieved
        pre = dec * s_prev + delta_scalar * K_t
        # tanh via exp — Triton 3.5 doesn't expose tanh directly.
        e2x = tl.exp(2.0 * pre)
        f_val = (e2x - 1.0) / (e2x + 1.0)
        S_var_row = tl.load(S_var_ptr + bh_T_nn + t * N * N + row * N + n_idx)
        r = S_var_row - f_val
        tanh_deriv = 1.0 - f_val * f_val

        D_step = dec * tanh_deriv
        u_step = tanh_deriv * K_t
        v_step = K_t
        b_step = -r

        # Combine: new_prefix = step ∘ prefix  (apply prefix first, then step)
        D_p, u_p, v_p, b_p = _combine_r1_triton(
            D_p, u_p, v_p, b_p,
            D_step, u_step, v_step, b_step
        )

        tl.store(delta_ptr + bh_T_nn + t * N * N + row * N + n_idx, b_p)


def fused_newton_iter(S0, S_var, K, V, decay):
    """Run ONE Newton iteration via fused Triton kernel, returns δ."""
    B, H, T, n, _ = S_var.shape
    delta = torch.empty_like(S_var)
    grid = (B * H * n,)
    _fused_newton_iter_kernel[grid](
        S0.contiguous(), S_var.contiguous(),
        K.contiguous(), V.contiguous(), decay.contiguous(),
        delta, B=B, H=H, T=T, N=n,
    )
    return delta


# -----------------------------------------------------------------------------
# Correctness: match separate-phase path.
# -----------------------------------------------------------------------------

def test_fused(B, H, T, n, seed=0, dtype=torch.float32, device='cuda'):
    g = torch.Generator(device=device).manual_seed(seed)
    S0 = 0.1 * torch.randn(B, H, n, n, generator=g, dtype=dtype, device=device)
    K = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dtype, device=device)
    V = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dtype, device=device)
    decay = 0.9 + 0.1 * torch.rand(B, H, T, generator=g, dtype=dtype, device=device)

    # Starting point: a partial sequential forward run's trajectory
    S_traj = sequential_e88_forward(S0, K, V, decay)
    S_var = S_traj[:, :, 1:] * 0.9 + 0.01 * torch.randn_like(S_traj[:, :, 1:])  # perturb

    # Separate-phase reference path
    r_ref = compute_residuals_batched(S0, S_var, K, V, decay)
    D, u, v, b = build_step_ingredients(S0, S_var, K, V, decay, r_ref)
    delta_ref = scan_r1_triton_5d(D, u, v, b)

    # Fused Triton kernel
    delta_fused = fused_newton_iter(S0, S_var, K, V, decay)

    diff = (delta_ref - delta_fused).abs().max().item()
    status = "PASS" if diff < 1e-4 else "FAIL"
    print(f"  B={B} H={H:3d} T={T:4d} n={n:3d}  "
          f"max|δ_ref − δ_fused|={diff:.3e}  [{status}]")
    return diff


def bench_fused_vs_separate(B, H, T, n):
    g = torch.Generator(device='cuda').manual_seed(0)
    dt = torch.float32
    S0 = 0.1 * torch.randn(B, H, n, n, generator=g, dtype=dt, device='cuda')
    K = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    V = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    decay = 0.9 + 0.1 * torch.rand(B, H, T, generator=g, dtype=dt, device='cuda')
    S_traj = sequential_e88_forward(S0, K, V, decay)
    S_var = S_traj[:, :, 1:] * 0.9

    # Warmup
    for _ in range(3):
        fused_newton_iter(S0, S_var, K, V, decay)
        r = compute_residuals_batched(S0, S_var, K, V, decay)
        D, u, v, b = build_step_ingredients(S0, S_var, K, V, decay, r)
        _ = scan_r1_triton_5d(D, u, v, b)
    torch.cuda.synchronize()

    # Fused
    N = 10
    t0 = time.time()
    for _ in range(N):
        _ = fused_newton_iter(S0, S_var, K, V, decay)
    torch.cuda.synchronize()
    fused_ms = (time.time() - t0) / N * 1000

    # Separate-phase path
    t0 = time.time()
    for _ in range(N):
        r = compute_residuals_batched(S0, S_var, K, V, decay)
        D, u, v, b = build_step_ingredients(S0, S_var, K, V, decay, r)
        _ = scan_r1_triton_5d(D, u, v, b)
    torch.cuda.synchronize()
    sep_ms = (time.time() - t0) / N * 1000

    speedup = sep_ms / fused_ms
    print(f"  B={B} H={H:3d} T={T:5d} n={n:3d}  "
          f"fused={fused_ms:.2f} ms  separate={sep_ms:.2f} ms  "
          f"speedup={speedup:.2f}×")


def newton_e88_fused(S0, K, V, decay, *, max_iters=20, tol=1e-4):
    """Newton solver using the fused kernel for each iteration."""
    B, H, T, n = K.shape
    S_var = torch.zeros(B, H, T, n, n, dtype=K.dtype, device=K.device)
    for it in range(max_iters):
        delta = fused_newton_iter(S0, S_var, K, V, decay)
        # δ_norm is a cheap proxy for residual norm (won't be zero at convergence
        # but monotonically decreasing)
        d_max = delta.abs().max().item()
        S_var = S_var + delta
        if d_max < tol:
            break
    S = torch.empty(B, H, T + 1, n, n, dtype=K.dtype, device=K.device)
    S[:, :, 0] = S0
    S[:, :, 1:] = S_var
    return S, it + 1, d_max


def test_newton_fused(B, H, T, n, seed=0, dtype=torch.float32, device='cuda'):
    g = torch.Generator(device=device).manual_seed(seed)
    S0 = 0.1 * torch.randn(B, H, n, n, generator=g, dtype=dtype, device=device)
    K = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dtype, device=device)
    V = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dtype, device=device)
    decay = 0.9 + 0.1 * torch.rand(B, H, T, generator=g, dtype=dtype, device=device)

    S_seq = sequential_e88_forward(S0, K, V, decay)
    S_f, iters, dmax = newton_e88_fused(S0, K, V, decay, tol=1e-4)
    diff = (S_seq - S_f).abs().max().item()
    status = "PASS" if diff < max(1e-3, T * 1e-5) else "FAIL"
    print(f"  B={B} H={H:3d} T={T:5d} n={n:3d}  iters={iters:2d} "
          f"dmax={dmax:.2e}  max|seq − fused|={diff:.3e}  [{status}]")
    return diff


def bench_e2e(B, H, T, n):
    """Time the FULL Newton solve (with fused kernel) vs CUDA kernel."""
    g = torch.Generator(device='cuda').manual_seed(0)
    dt = torch.float32
    S0_f = 0.1 * torch.randn(B, H, n, n, generator=g, dtype=dt, device='cuda')
    K_f = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    V_f = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    decay_f = 0.9 + 0.1 * torch.rand(B, H, T, generator=g, dtype=dt, device='cuda')

    # CUDA inputs (bf16, [T, B, H, n] layout)
    g2 = torch.Generator(device='cuda').manual_seed(0)
    dt_c = torch.bfloat16
    k_c = (0.3 * torch.randn(T, B, H, n, generator=g2, dtype=dt_c, device='cuda')).requires_grad_(False)
    v_c = (0.3 * torch.randn(T, B, H, n, generator=g2, dtype=dt_c, device='cuda')).requires_grad_(False)
    q_c = (0.3 * torch.randn(T, B, H, n, generator=g2, dtype=dt_c, device='cuda')).requires_grad_(False)
    dec_c = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, generator=g2, dtype=dt_c, device='cuda'))
    S0_c = 0.1 * torch.randn(B, H, n, n, generator=g2, dtype=dt_c, device='cuda')

    from elman.models.e88_fla_hybrid import E88FLAHybridCUDAFunction

    # Warmup
    for _ in range(3):
        _ = newton_e88_fused(S0_f, K_f, V_f, decay_f, tol=1e-4)
        _ = E88FLAHybridCUDAFunction.apply(False, k_c, v_c, q_c, dec_c, S0_c, H)
    torch.cuda.synchronize()

    N = 5
    t0 = time.time()
    for _ in range(N):
        _, iters, _ = newton_e88_fused(S0_f, K_f, V_f, decay_f, tol=1e-4)
    torch.cuda.synchronize()
    fused_ms = (time.time() - t0) / N * 1000

    t0 = time.time()
    for _ in range(N):
        _ = E88FLAHybridCUDAFunction.apply(False, k_c, v_c, q_c, dec_c, S0_c, H)
    torch.cuda.synchronize()
    cuda_ms = (time.time() - t0) / N * 1000

    print(f"  B={B} H={H:3d} T={T:5d} n={n}  CUDA (bf16)={cuda_ms:.2f} ms  "
          f"Fused-Newton (fp32, {iters} iter)={fused_ms:.2f} ms  "
          f"ratio (Newton/CUDA)={fused_ms/cuda_ms:.2f}×")


if __name__ == '__main__':
    print("Fused single-iter kernel: correctness vs separate-phase path.\n")
    for B, H, T, n in [(1, 4, 32, 8),
                       (1, 8, 128, 16),
                       (1, 16, 512, 32),
                       (2, 32, 1024, 32)]:
        test_fused(B, H, T, n)

    print("\nFull Newton loop (fused kernel): correctness vs sequential.\n")
    for B, H, T, n in [(1, 4, 64, 16),
                       (1, 16, 256, 32),
                       (1, 32, 1024, 32),
                       (2, 32, 2048, 32)]:
        test_newton_fused(B, H, T, n)

    print("\nEnd-to-end forward (Newton w/fused kernel) vs production CUDA.\n")
    for B, H, T, n in [(1, 32, 512, 32),
                       (1, 32, 2048, 32),
                       (1, 32, 8192, 32)]:
        bench_e2e(B, H, T, n)
