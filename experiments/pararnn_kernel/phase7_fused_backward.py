"""Phase 7 — fused backward kernel.

Fuses into a single Triton kernel:
  1. Recompute per-step tensors (retrieved, delta, pre, tanh', D, u)
  2. Reverse scan over T
  3. Emit per-step dV, dK, ddecay, and final dS0

Compared to phase5_backward:
  - No materialized D, u, pre, tanh_deriv tensors (saves 4 × 16 GB at T=128K).
  - No dK_partial buffer (saves 16 GB): we use one program per (B, H), with
    2D tile [N_rows, N_cols] per program, and tl.sum reduces rows in-register.
  - int64 offsets for T ≥ 128K.
  - num_warps tuned per kernel.

Grid: (B * H,).  Each program processes all rows of one head, sequential in T.
"""

import sys
import os
import time

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase4_newton_driver import sequential_e88_forward


@triton.jit
def _fused_backward_kernel(
    S_traj_ptr,               # [B, H, T+1, N, N]  fwd trajectory incl. S0
    K_ptr, V_ptr, decay_ptr,  # inputs
    g_T_ptr,                  # [B, H, N, N]       external grad at t=T
    dL_dS_int_ptr,            # [B, H, T, N, N]    external grads at t=0..T-1
    # outputs:
    g_out_ptr,                # [B, H, N, N]       dL/dS_0
    dK_ptr,                   # [B, H, T, N]
    dV_ptr,                   # [B, H, T, N]
    ddecay_ptr,               # [B, H, T]
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, N: tl.constexpr,
):
    """One program per (b, h). 2D tile [N_rows, N_cols] per program."""
    pid = tl.program_id(0).to(tl.int64)
    b = pid // H
    h = pid % H
    bh = b * H + h

    row_idx = tl.arange(0, N).to(tl.int64)                 # [N]
    col_idx = tl.arange(0, N).to(tl.int64)                 # [N]

    # offsets
    gT_base = bh * N * N                                   # into [B, H, N, N]
    S_traj_head_stride = (T + 1) * N * N                   # per-head stride in S_traj
    dL_internal_head_stride = T * N * N                    # per-head stride in dL_internal
    K_head_stride = T * N                                  # per-head stride in K, V
    dec_head_stride = T                                    # per-head stride in decay

    # 2D offset for [N, N] tile within a head's timestep slice:
    tile_2d = row_idx[:, None] * N + col_idx[None, :]      # [N, N]

    # Load initial g = dL/dS[T] as [N, N]
    g = tl.load(g_T_ptr + gT_base + tile_2d).to(tl.float32)

    # Reverse scan: t = T-1 down to 0
    for t_inv in range(T):
        t = T - 1 - t_inv

        # Load broadcast-shared tensors for this step
        K_t = tl.load(K_ptr + bh * K_head_stride + t * N + col_idx).to(tl.float32)  # [N]
        V_t = tl.load(V_ptr + bh * K_head_stride + t * N + row_idx).to(tl.float32)  # [N]
        dec = tl.load(decay_ptr + bh * dec_head_stride + t).to(tl.float32)           # scalar

        # S[t-1] from forward trajectory (index t because S_traj[0] = S_0)
        S_prev_base = bh * S_traj_head_stride + t * N * N
        S_prev = tl.load(S_traj_ptr + S_prev_base + tile_2d).to(tl.float32)          # [N, N]

        # retrieved[row] = sum_col S_prev[row, col] * K_t[col]
        retrieved = tl.sum(S_prev * K_t[None, :], axis=1)                            # [N]
        delta_row = V_t - retrieved                                                   # [N]

        # pre[row, col] = dec * S_prev[row, col] + delta_row[row] * K_t[col]
        pre = dec * S_prev + delta_row[:, None] * K_t[None, :]                        # [N, N]
        e2x = tl.exp(2.0 * pre)
        tanh_val = (e2x - 1.0) / (e2x + 1.0)
        tanh_deriv = 1.0 - tanh_val * tanh_val                                        # [N, N]

        u_mat = tanh_deriv * K_t[None, :]                                             # [N, N]

        # Per-row scalar g · u
        gu_row = tl.sum(g * u_mat, axis=1)                                            # [N]

        # dV[t, row] = g · u
        tl.store(dV_ptr + bh * K_head_stride + t * N + row_idx, gu_row)

        # dK[t, col] = sum_row [ delta[row] * (g ⊙ tanh')[row, col] - S_prev[row, col] * gu[row] ]
        g_times_tanhd = g * tanh_deriv                                                # [N, N]
        dK_contrib = delta_row[:, None] * g_times_tanhd - S_prev * gu_row[:, None]   # [N, N]
        dK_t = tl.sum(dK_contrib, axis=0)                                             # [N]
        tl.store(dK_ptr + bh * K_head_stride + t * N + col_idx, dK_t)

        # ddecay[t] = sum over (row, col) of g ⊙ tanh' ⊙ S_prev
        ddec_t = tl.sum(g_times_tanhd * S_prev)                                       # scalar
        tl.store(ddecay_ptr + bh * dec_head_stride + t, ddec_t)

        # g_new = D ⊙ g - K_t[None, :] * gu_row[:, None]
        D_mat = dec * tanh_deriv                                                      # [N, N]
        g_new = D_mat * g - K_t[None, :] * gu_row[:, None]                            # [N, N]

        # Add external grad at t
        ext = tl.load(dL_dS_int_ptr + bh * dL_internal_head_stride + t * N * N + tile_2d
                      ).to(tl.float32)
        g = g_new + ext

    # Write dL/dS_0
    tl.store(g_out_ptr + gT_base + tile_2d, g.to(g_out_ptr.dtype.element_ty))


@triton.jit
def _fused_backward_rank1_kernel(
    S_traj_ptr,               # [B, H, T+1, N, N]  fwd trajectory
    K_ptr, V_ptr, decay_ptr,
    g_T_ptr,                  # [B, H, N, N]       grad at t=T
    dL_dout_ptr,              # [B, H, T, N]       dL/doutput  (rank-1 factor A)
    q_ptr,                    # [B, H, T, N]       q           (rank-1 factor B)
    g_out_ptr,                # [B, H, N, N]       dL/dS_0
    dK_ptr,                   # [B, H, T, N]
    dV_ptr,                   # [B, H, T, N]
    ddecay_ptr,               # [B, H, T]
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, N: tl.constexpr,
):
    """Same as _fused_backward_kernel but dL/dS[t] is reconstructed from
    rank-1 factors on the fly:  dL/dS[t, row, col] = dL_dout[t, row] · q[t, col].
    Saves 16 GB of HBM at T=128K, avoids materializing the full [B, H, T, N, N]
    gradient tensor."""
    pid = tl.program_id(0).to(tl.int64)
    b = pid // H
    h = pid % H
    bh = b * H + h

    row_idx = tl.arange(0, N).to(tl.int64)
    col_idx = tl.arange(0, N).to(tl.int64)

    gT_base = bh * N * N
    S_traj_head_stride = (T + 1) * N * N
    K_head_stride = T * N
    dec_head_stride = T
    tile_2d = row_idx[:, None] * N + col_idx[None, :]

    g = tl.load(g_T_ptr + gT_base + tile_2d).to(tl.float32)

    for t_inv in range(T):
        t = T - 1 - t_inv

        K_t = tl.load(K_ptr + bh * K_head_stride + t * N + col_idx).to(tl.float32)
        V_t = tl.load(V_ptr + bh * K_head_stride + t * N + row_idx).to(tl.float32)
        dec = tl.load(decay_ptr + bh * dec_head_stride + t).to(tl.float32)

        S_prev_base = bh * S_traj_head_stride + t * N * N
        S_prev = tl.load(S_traj_ptr + S_prev_base + tile_2d).to(tl.float32)

        retrieved = tl.sum(S_prev * K_t[None, :], axis=1)
        delta_row = V_t - retrieved

        pre = dec * S_prev + delta_row[:, None] * K_t[None, :]
        e2x = tl.exp(2.0 * pre)
        tanh_val = (e2x - 1.0) / (e2x + 1.0)
        tanh_deriv = 1.0 - tanh_val * tanh_val

        u_mat = tanh_deriv * K_t[None, :]
        gu_row = tl.sum(g * u_mat, axis=1)
        tl.store(dV_ptr + bh * K_head_stride + t * N + row_idx, gu_row)

        g_times_tanhd = g * tanh_deriv
        dK_contrib = delta_row[:, None] * g_times_tanhd - S_prev * gu_row[:, None]
        dK_t = tl.sum(dK_contrib, axis=0)
        tl.store(dK_ptr + bh * K_head_stride + t * N + col_idx, dK_t)

        ddec_t = tl.sum(g_times_tanhd * S_prev)
        tl.store(ddecay_ptr + bh * dec_head_stride + t, ddec_t)

        D_mat = dec * tanh_deriv
        g_new = D_mat * g - K_t[None, :] * gu_row[:, None]

        # Rank-1 external grad: dL/dS[t, row, col] = dL_dout[t, row] * q[t, col]
        dL_dout_t = tl.load(dL_dout_ptr + bh * K_head_stride + t * N + row_idx).to(tl.float32)
        q_t = tl.load(q_ptr + bh * K_head_stride + t * N + col_idx).to(tl.float32)
        ext = dL_dout_t[:, None] * q_t[None, :]
        g = g_new + ext

    tl.store(g_out_ptr + gT_base + tile_2d, g.to(g_out_ptr.dtype.element_ty))


@triton.jit
def _fused_backward_rank1_dQ_kernel(
    S_traj_ptr,
    K_ptr, V_ptr, decay_ptr,
    g_T_ptr, dL_dout_ptr, q_ptr,
    g_out_ptr, dK_ptr, dV_ptr, ddecay_ptr, dQ_ptr,
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, N: tl.constexpr,
):
    """Rank-1 backward fused with dQ computation.

    dQ[t, col] = sum_row dL_dout[t, row] * S[t+1, row, col]

    At kernel iter t (going T-1 → 0), we have S[t+1] cached from the
    previous iteration (or loaded fresh at t=T-1). We compute dQ[t]
    using S[t+1], then load S[t] for the backward step.
    """
    pid = tl.program_id(0).to(tl.int64)
    b = pid // H
    h = pid % H
    bh = b * H + h

    row_idx = tl.arange(0, N).to(tl.int64)
    col_idx = tl.arange(0, N).to(tl.int64)

    gT_base = bh * N * N
    S_traj_head_stride = (T + 1) * N * N
    K_head_stride = T * N
    dec_head_stride = T
    tile_2d = row_idx[:, None] * N + col_idx[None, :]

    g = tl.load(g_T_ptr + gT_base + tile_2d).to(tl.float32)

    # Load S[T] = S_traj[T] as initial S_next for dQ[T-1] computation
    S_next = tl.load(S_traj_ptr + bh * S_traj_head_stride + T * N * N + tile_2d).to(tl.float32)

    for t_inv in range(T):
        t = T - 1 - t_inv

        # dQ[t, col] = sum_row dL_dout[t, row] * S_next[row, col]
        dL_dout_t = tl.load(dL_dout_ptr + bh * K_head_stride + t * N + row_idx).to(tl.float32)
        dQ_t = tl.sum(dL_dout_t[:, None] * S_next, axis=0)
        tl.store(dQ_ptr + bh * K_head_stride + t * N + col_idx, dQ_t)

        # Load for the backward step (uses S_prev = S[t])
        K_t = tl.load(K_ptr + bh * K_head_stride + t * N + col_idx).to(tl.float32)
        V_t = tl.load(V_ptr + bh * K_head_stride + t * N + row_idx).to(tl.float32)
        dec = tl.load(decay_ptr + bh * dec_head_stride + t).to(tl.float32)

        S_prev_base = bh * S_traj_head_stride + t * N * N
        S_prev = tl.load(S_traj_ptr + S_prev_base + tile_2d).to(tl.float32)

        retrieved = tl.sum(S_prev * K_t[None, :], axis=1)
        delta_row = V_t - retrieved

        pre = dec * S_prev + delta_row[:, None] * K_t[None, :]
        e2x = tl.exp(2.0 * pre)
        tanh_val = (e2x - 1.0) / (e2x + 1.0)
        tanh_deriv = 1.0 - tanh_val * tanh_val

        u_mat = tanh_deriv * K_t[None, :]
        gu_row = tl.sum(g * u_mat, axis=1)
        tl.store(dV_ptr + bh * K_head_stride + t * N + row_idx, gu_row)

        g_times_tanhd = g * tanh_deriv
        dK_contrib = delta_row[:, None] * g_times_tanhd - S_prev * gu_row[:, None]
        dK_t = tl.sum(dK_contrib, axis=0)
        tl.store(dK_ptr + bh * K_head_stride + t * N + col_idx, dK_t)

        ddec_t = tl.sum(g_times_tanhd * S_prev)
        tl.store(ddecay_ptr + bh * dec_head_stride + t, ddec_t)

        D_mat = dec * tanh_deriv
        g_new = D_mat * g - K_t[None, :] * gu_row[:, None]

        q_t = tl.load(q_ptr + bh * K_head_stride + t * N + col_idx).to(tl.float32)
        ext = dL_dout_t[:, None] * q_t[None, :]
        g = g_new + ext

        # S_prev becomes next iteration's S_next (S[t] = next iter's S_next)
        S_next = S_prev

    tl.store(g_out_ptr + gT_base + tile_2d, g.to(g_out_ptr.dtype.element_ty))


def backward_e88_fused_rank1_dQ(S_traj, K, V, decay, g_T, dL_dout, q,
                                  num_warps=4, num_stages=2):
    """Fused backward + dQ. One pass, all gradients."""
    B, H, T1, n, _ = S_traj.shape
    T = T1 - 1
    device = K.device
    dtype = K.dtype

    dV = torch.zeros_like(V)
    dK = torch.zeros_like(K)
    ddec = torch.zeros_like(decay)
    dS0 = torch.zeros(B, H, n, n, dtype=dtype, device=device)
    dQ = torch.zeros_like(q)

    grid = (B * H,)
    _fused_backward_rank1_dQ_kernel[grid](
        S_traj.contiguous(),
        K.contiguous(), V.contiguous(), decay.contiguous(),
        g_T.contiguous(), dL_dout.contiguous(), q.contiguous(),
        dS0, dK, dV, ddec, dQ,
        B=B, H=H, T=T, N=n,
        num_warps=num_warps, num_stages=num_stages,
    )
    return dS0, dK, dV, ddec, dQ


def backward_e88_fused_rank1(S_traj, K, V, decay, g_T, dL_dout, q,
                              num_warps=4, num_stages=2):
    """Rank-1 backward: dL/dS[t] = outer(dL_dout[t], q[t]) reconstructed in kernel.

    Args:
      S_traj:  [B, H, T+1, N, N]
      K, V:    [B, H, T, N]
      decay:   [B, H, T]
      g_T:     [B, H, N, N] gradient at final state
      dL_dout: [B, H, T, N] dL/doutput[t, row]
      q:       [B, H, T, N] the q value from forward pass
    """
    B, H, T1, n, _ = S_traj.shape
    T = T1 - 1
    device = K.device
    dtype = K.dtype

    dV = torch.zeros_like(V)
    dK = torch.zeros_like(K)
    ddec = torch.zeros_like(decay)
    dS0 = torch.zeros(B, H, n, n, dtype=dtype, device=device)

    grid = (B * H,)
    _fused_backward_rank1_kernel[grid](
        S_traj.contiguous(),
        K.contiguous(), V.contiguous(), decay.contiguous(),
        g_T.contiguous(), dL_dout.contiguous(), q.contiguous(),
        dS0, dK, dV, ddec,
        B=B, H=H, T=T, N=n,
        num_warps=num_warps, num_stages=num_stages,
    )
    return dS0, dK, dV, ddec


def backward_e88_fused(S_traj, K, V, decay, dL_dS_traj,
                        num_warps=4, num_stages=2):
    """Fused Triton backward — matches backward_e88_triton's API.

    Returns dS0, dK, dV, ddecay.
    """
    B, H, T1, n, _ = S_traj.shape
    T = T1 - 1
    device = K.device
    dtype = K.dtype

    dV = torch.zeros_like(V)
    dK = torch.zeros_like(K)
    ddec = torch.zeros_like(decay)
    dS0 = torch.zeros(B, H, n, n, dtype=dtype, device=device)

    g_T = dL_dS_traj[:, :, T].contiguous()
    dL_internal = dL_dS_traj[:, :, :T].contiguous()

    grid = (B * H,)
    _fused_backward_kernel[grid](
        S_traj.contiguous(),
        K.contiguous(), V.contiguous(), decay.contiguous(),
        g_T, dL_internal,
        dS0, dK, dV, ddec,
        B=B, H=H, T=T, N=n,
        num_warps=num_warps, num_stages=num_stages,
    )

    return dS0, dK, dV, ddec


def test_matches_reference(B, H, T, n, seed=0, dtype=torch.float32, device='cuda'):
    """Verify fused backward matches PyTorch autograd."""
    from phase5_backward import autograd_reference, _random_case

    torch.manual_seed(seed)
    S0 = 0.1 * torch.randn(B, H, n, n, dtype=dtype, device=device)
    K = 0.3 * torch.randn(B, H, T, n, dtype=dtype, device=device)
    V = 0.3 * torch.randn(B, H, T, n, dtype=dtype, device=device)
    decay = 0.9 + 0.1 * torch.rand(B, H, T, dtype=dtype, device=device)
    dL_dS_traj = 0.1 * torch.randn(B, H, T + 1, n, n, dtype=dtype, device=device)

    gS0_ag, gK_ag, gV_ag, gdec_ag = autograd_reference(S0, K, V, decay, dL_dS_traj)
    S_traj = sequential_e88_forward(S0, K, V, decay)

    dS0, dK, dV, ddec = backward_e88_fused(S_traj, K, V, decay, dL_dS_traj)

    def rel_err(a, b):
        return (a - b).abs().max().item() / max(b.abs().max().item(), 1e-30)

    eS0 = rel_err(dS0, gS0_ag)
    eK = rel_err(dK, gK_ag)
    eV = rel_err(dV, gV_ag)
    ed = rel_err(ddec, gdec_ag)
    worst = max(eS0, eK, eV, ed)
    status = "PASS" if worst < 1e-4 else "FAIL"
    print(f"  B={B} H={H:3d} T={T:4d} n={n:3d}  "
          f"relErr dS0={eS0:.2e} dK={eK:.2e} dV={eV:.2e} ddec={ed:.2e}  [{status}]")
    return worst


def bench_fused_backward(B, H, T, n, num_warps=4, num_stages=2, n_repeat=5):
    g = torch.Generator(device='cuda').manual_seed(0)
    dt = torch.float32
    S0 = 0.1 * torch.randn(B, H, n, n, generator=g, dtype=dt, device='cuda')
    K = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    V = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    decay = 0.9 + 0.1 * torch.rand(B, H, T, generator=g, dtype=dt, device='cuda')
    S_traj = sequential_e88_forward(S0, K, V, decay)
    dL_dS_traj = 0.01 * torch.randn_like(S_traj)

    def run():
        return backward_e88_fused(S_traj, K, V, decay, dL_dS_traj,
                                    num_warps=num_warps, num_stages=num_stages)

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat):
        _ = run()
    torch.cuda.synchronize()
    ms = (time.time() - t0) / n_repeat * 1000
    peak = torch.cuda.max_memory_allocated() / 1024**3
    return ms, peak


if __name__ == '__main__':
    print("Correctness:\n")
    for shape in [(1, 4, 128, 16), (1, 16, 512, 32), (2, 8, 256, 32)]:
        test_matches_reference(*shape)

    print("\nTuning num_warps/num_stages at T=32K, H=83 n=32 (production):")
    for nw in [1, 2, 4, 8, 16]:
        for ns in [1, 2, 3, 4, 5, 6]:
            try:
                ms, peak = bench_fused_backward(1, 83, 32768, 32, nw, ns)
                print(f"  nw={nw:>2d} ns={ns}: {ms:.2f} ms  peak={peak:.1f} GB")
            except Exception as e:
                print(f"  nw={nw} ns={ns}: FAIL {str(e)[:60]}")
