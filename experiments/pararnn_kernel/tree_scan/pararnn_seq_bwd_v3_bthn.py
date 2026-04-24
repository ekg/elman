"""V3 backward — [B, T, H, N]-native stride-parametrized Triton kernel.

Accepts dL_dout in [B, T, H, N] (production layout), Q/K/V in [B, T, H, N],
decay in [B, T, H], and S_traj in [B, H, T, N, N] (contig, no S_0 slot).

Emits dK, dV, dQ in [B, T, H, N], ddecay in [B, T, H], dS0 in [B, H, N, N].

No permutes! This eliminates the four dK/dV/dQ/ddecay output permutes v2 does.

Convention and math are identical to _bwd_v2_kernel — only indexing differs.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _seq_bwd_v3_bthn_kernel(
    # Inputs:
    S0_ptr,      # [B, H, N, N]  — S_0 (branch-free final step)
    S_traj_ptr,  # [B, H, T, N, N]  — S_1..S_T
    K_ptr,       # [B, T, H, N]
    V_ptr,       # [B, T, H, N]
    Q_ptr,       # [B, T, H, N]
    decay_ptr,   # [B, T, H]
    g_T_ptr,     # [B, H, N, N]   — dL/dS_T (seed)
    dL_dout_ptr, # [B, T, H, N]   — dL/doutput[t, row]
    # Outputs:
    g_out_ptr,   # [B, H, N, N]    — dL/dS_0
    dK_ptr,      # [B, T, H, N]
    dV_ptr,      # [B, T, H, N]
    dQ_ptr,      # [B, T, H, N]
    ddecay_ptr,  # [B, T, H]
    # Runtime strides (K/V/Q/dL_dout share layout; dK/dV/dQ share layout):
    k_b_stride, k_t_stride, k_h_stride, k_n_stride,
    dec_b_stride, dec_t_stride, dec_h_stride,
    dL_b_stride, dL_t_stride, dL_h_stride, dL_n_stride,
    dK_b_stride, dK_t_stride, dK_h_stride, dK_n_stride,
    ddec_b_stride, ddec_t_stride, ddec_h_stride,
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, N: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    b = pid // H
    h = pid % H
    bh = b * H + h
    row_idx = tl.arange(0, N).to(tl.int64)
    col_idx = tl.arange(0, N).to(tl.int64)
    tile_2d = row_idx[:, None] * N + col_idx[None, :]
    S_head_stride = N * N
    traj_head_stride = T * N * N

    # Base pointers for this (b, h)
    k_base = b * k_b_stride + h * k_h_stride
    dec_base = b * dec_b_stride + h * dec_h_stride
    dL_base = b * dL_b_stride + h * dL_h_stride
    dK_base = b * dK_b_stride + h * dK_h_stride
    ddec_base = b * ddec_b_stride + h * ddec_h_stride

    # Load g = dL/dS_T
    g = tl.load(g_T_ptr + bh * S_head_stride + tile_2d).to(tl.float32)

    # Main loop: t = T-1, T-2, ..., 1  (T-1 iterations)
    for t_inv in range(T - 1):
        t = T - 1 - t_inv

        # Load dL_dout[t], Q[t]
        dL_off_row = dL_base + t * dL_t_stride + row_idx * dL_n_stride
        q_off = k_base + t * k_t_stride + col_idx * k_n_stride
        dL_dout_t = tl.load(dL_dout_ptr + dL_off_row).to(tl.float32)
        q_t = tl.load(Q_ptr + q_off).to(tl.float32)
        g = g + dL_dout_t[:, None] * q_t[None, :]

        # Load K[t], V[t], decay[t]
        k_off = k_base + t * k_t_stride + col_idx * k_n_stride
        v_off = k_base + t * k_t_stride + row_idx * k_n_stride
        d_off = dec_base + t * dec_t_stride
        K_t = tl.load(K_ptr + k_off).to(tl.float32)
        V_t = tl.load(V_ptr + v_off).to(tl.float32)
        dec = tl.load(decay_ptr + d_off).to(tl.float32)

        # S_prev = S_t = S_traj[t-1]
        S_prev = tl.load(
            S_traj_ptr + bh * traj_head_stride + (t - 1) * N * N + tile_2d
        ).to(tl.float32)

        retrieved = tl.sum(S_prev * K_t[None, :], axis=1)
        delta_row = V_t - retrieved
        pre = dec * S_prev + delta_row[:, None] * K_t[None, :]
        e2x = tl.exp(2.0 * pre)
        tanh_val = (e2x - 1.0) / (e2x + 1.0)
        tanh_deriv = 1.0 - tanh_val * tanh_val

        # dQ[t, col] = sum_row dL_dout[t, row] * tanh_val[row, col]
        # (tanh_val = S_{t+1}, and output_t = S_{t+1} @ q_t; dQ[t] = dL_dout[t] @ S_{t+1})
        dQ_t = tl.sum(dL_dout_t[:, None] * tanh_val, axis=0)
        dQ_out_off = dK_base + t * dK_t_stride + col_idx * dK_n_stride
        tl.store(dQ_ptr + dQ_out_off, dQ_t.to(dQ_ptr.dtype.element_ty))

        u_mat = tanh_deriv * K_t[None, :]
        gu_row = tl.sum(g * u_mat, axis=1)
        dV_out_off = dK_base + t * dK_t_stride + row_idx * dK_n_stride
        tl.store(dV_ptr + dV_out_off, gu_row.to(dV_ptr.dtype.element_ty))

        g_times_tanhd = g * tanh_deriv
        dK_contrib = delta_row[:, None] * g_times_tanhd - S_prev * gu_row[:, None]
        dK_t = tl.sum(dK_contrib, axis=0)
        dK_out_off = dK_base + t * dK_t_stride + col_idx * dK_n_stride
        tl.store(dK_ptr + dK_out_off, dK_t.to(dK_ptr.dtype.element_ty))

        ddec_t = tl.sum(g_times_tanhd * S_prev)
        ddec_out_off = ddec_base + t * ddec_t_stride
        tl.store(ddecay_ptr + ddec_out_off, ddec_t.to(ddecay_ptr.dtype.element_ty))

        D_mat = dec * tanh_deriv
        g = D_mat * g - K_t[None, :] * gu_row[:, None]

    # Final step: t = 0, with S_prev = S_0
    dL_off_row = dL_base + 0 * dL_t_stride + row_idx * dL_n_stride
    q_off = k_base + 0 * k_t_stride + col_idx * k_n_stride
    dL_dout_0 = tl.load(dL_dout_ptr + dL_off_row).to(tl.float32)
    q_0 = tl.load(Q_ptr + q_off).to(tl.float32)
    g = g + dL_dout_0[:, None] * q_0[None, :]

    k_off = k_base + 0 * k_t_stride + col_idx * k_n_stride
    v_off = k_base + 0 * k_t_stride + row_idx * k_n_stride
    d_off = dec_base + 0 * dec_t_stride
    K_0 = tl.load(K_ptr + k_off).to(tl.float32)
    V_0 = tl.load(V_ptr + v_off).to(tl.float32)
    dec_0 = tl.load(decay_ptr + d_off).to(tl.float32)

    S_prev_0 = tl.load(S0_ptr + bh * S_head_stride + tile_2d).to(tl.float32)

    retrieved = tl.sum(S_prev_0 * K_0[None, :], axis=1)
    delta_row = V_0 - retrieved
    pre = dec_0 * S_prev_0 + delta_row[:, None] * K_0[None, :]
    e2x = tl.exp(2.0 * pre)
    tanh_val = (e2x - 1.0) / (e2x + 1.0)
    tanh_deriv = 1.0 - tanh_val * tanh_val

    dQ_0 = tl.sum(dL_dout_0[:, None] * tanh_val, axis=0)
    dQ_out_off = dK_base + 0 * dK_t_stride + col_idx * dK_n_stride
    tl.store(dQ_ptr + dQ_out_off, dQ_0.to(dQ_ptr.dtype.element_ty))

    u_mat = tanh_deriv * K_0[None, :]
    gu_row = tl.sum(g * u_mat, axis=1)
    dV_out_off = dK_base + 0 * dK_t_stride + row_idx * dK_n_stride
    tl.store(dV_ptr + dV_out_off, gu_row.to(dV_ptr.dtype.element_ty))

    g_times_tanhd = g * tanh_deriv
    dK_contrib = delta_row[:, None] * g_times_tanhd - S_prev_0 * gu_row[:, None]
    dK_0 = tl.sum(dK_contrib, axis=0)
    dK_out_off = dK_base + 0 * dK_t_stride + col_idx * dK_n_stride
    tl.store(dK_ptr + dK_out_off, dK_0.to(dK_ptr.dtype.element_ty))

    ddec_0 = tl.sum(g_times_tanhd * S_prev_0)
    ddec_out_off = ddec_base + 0 * ddec_t_stride
    tl.store(ddecay_ptr + ddec_out_off, ddec_0.to(ddecay_ptr.dtype.element_ty))

    D_mat = dec_0 * tanh_deriv
    g = D_mat * g - K_0[None, :] * gu_row[:, None]

    tl.store(g_out_ptr + bh * S_head_stride + tile_2d,
             g.to(g_out_ptr.dtype.element_ty))


def backward_v3_bthn(S0, S_traj, K, V, Q, decay, g_T, dL_dout,
                      num_warps=1, num_stages=1):
    """[B, T, H, N]-native backward.

    Args:
      S0:      [B, H, N, N]
      S_traj:  [B, H, T, N, N] contig  (from v3 forward)
      K:       [B, T, H, N]
      V:       [B, T, H, N]
      Q:       [B, T, H, N]
      decay:   [B, T, H]
      g_T:     [B, H, N, N]  (dL/dS_T)
      dL_dout: [B, T, H, N]

    Returns:
      dS0:   [B, H, N, N]
      dK:    [B, T, H, N]
      dV:    [B, T, H, N]
      dQ:    [B, T, H, N]
      ddec:  [B, T, H]
    """
    B, T, H, N = K.shape
    assert V.shape == Q.shape == dL_dout.shape == (B, T, H, N)
    assert decay.shape == (B, T, H)
    assert S0.shape == (B, H, N, N)
    assert S_traj.shape == (B, H, T, N, N)

    device = K.device
    dtype = K.dtype

    dK = torch.empty_like(K)
    dV = torch.empty_like(V)
    dQ = torch.empty_like(Q)
    ddec = torch.empty_like(decay)
    dS0 = torch.empty(B, H, N, N, dtype=dtype, device=device)

    k_bs, k_ts, k_hs, k_ns = K.stride()
    dec_bs, dec_ts, dec_hs = decay.stride()
    dL_bs, dL_ts, dL_hs, dL_ns = dL_dout.stride()
    dK_bs, dK_ts, dK_hs, dK_ns = dK.stride()
    ddec_bs, ddec_ts, ddec_hs = ddec.stride()

    grid = (B * H,)
    _seq_bwd_v3_bthn_kernel[grid](
        S0.contiguous(), S_traj.contiguous(),
        K, V, Q, decay,
        g_T.contiguous(), dL_dout,
        dS0, dK, dV, dQ, ddec,
        k_bs, k_ts, k_hs, k_ns,
        dec_bs, dec_ts, dec_hs,
        dL_bs, dL_ts, dL_hs, dL_ns,
        dK_bs, dK_ts, dK_hs, dK_ns,
        ddec_bs, ddec_ts, ddec_hs,
        B=B, H=H, T=T, N=N,
        num_warps=num_warps, num_stages=num_stages,
    )
    return dS0, dK, dV, dQ, ddec


if __name__ == '__main__':
    import os
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '3')
    import sys
    THIS = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, THIS)
    from pararnn_seq_fwd_v2 import pararnn_seq_fwd_v2, backward_v2
    from pararnn_seq_fwd_v3_bthn import pararnn_seq_fwd_v3_bthn

    print("Correctness v3 backward vs v2 backward:\n")
    for B, H, T, N in [(1, 4, 512, 16), (1, 4, 1024, 32), (2, 8, 256, 16), (16, 4, 512, 32)]:
        dt = torch.float32
        torch.manual_seed(0)
        # Generate [B, T, H, N] inputs
        K_bt = 0.3 * torch.randn(B, T, H, N, dtype=dt, device='cuda')
        V_bt = 0.3 * torch.randn(B, T, H, N, dtype=dt, device='cuda')
        Q_bt = 0.3 * torch.randn(B, T, H, N, dtype=dt, device='cuda')
        decay_bt = torch.sigmoid(0.5 + 0.1 * torch.randn(B, T, H, dtype=dt, device='cuda'))
        S0 = 0.1 * torch.randn(B, H, N, N, dtype=dt, device='cuda')
        dL_dout_bt = 0.01 * torch.randn(B, T, H, N, dtype=dt, device='cuda')
        g_T = torch.zeros(B, H, N, N, dtype=dt, device='cuda')

        # v3 path
        S_traj_v3, Sq_v3 = pararnn_seq_fwd_v3_bthn(
            S0, K_bt, V_bt, Q_bt, decay_bt, num_warps=1 if N == 16 else 4)
        dS0_v3, dK_v3, dV_v3, dQ_v3, ddec_v3 = backward_v3_bthn(
            S0, S_traj_v3, K_bt, V_bt, Q_bt, decay_bt, g_T, dL_dout_bt,
            num_warps=1 if N == 16 else 2)

        # v2 path: permute inputs
        K_p = K_bt.permute(0, 2, 1, 3).contiguous()
        V_p = V_bt.permute(0, 2, 1, 3).contiguous()
        Q_p = Q_bt.permute(0, 2, 1, 3).contiguous()
        decay_p = decay_bt.permute(0, 2, 1).contiguous()
        dL_dout_p = dL_dout_bt.permute(0, 2, 1, 3).contiguous()
        S_traj_v2 = pararnn_seq_fwd_v2(S0, K_p, V_p, decay_p, num_warps=1 if N == 16 else 4)
        dS0_v2, dK_v2, dV_v2, dQ_v2, ddec_v2 = backward_v2(
            S0, S_traj_v2, K_p, V_p, decay_p, g_T, dL_dout_p, Q_p,
            num_warps=1 if N == 16 else 2)
        # permute v2 outputs back to [B, T, H, N]
        dK_v2_bt = dK_v2.permute(0, 2, 1, 3).contiguous()
        dV_v2_bt = dV_v2.permute(0, 2, 1, 3).contiguous()
        dQ_v2_bt = dQ_v2.permute(0, 2, 1, 3).contiguous()
        ddec_v2_bt = ddec_v2.permute(0, 2, 1).contiguous()

        def rel(a, b): return (a - b).abs().max().item() / max(b.abs().max().item(), 1e-10)
        r = {
            'dS0': rel(dS0_v3, dS0_v2),
            'dK': rel(dK_v3, dK_v2_bt),
            'dV': rel(dV_v3, dV_v2_bt),
            'dQ': rel(dQ_v3, dQ_v2_bt),
            'ddec': rel(ddec_v3, ddec_v2_bt),
        }
        w = max(r.values())
        ok = "PASS" if w < 1e-4 else "FAIL"
        details = "  ".join(f"{k}={v:.1e}" for k, v in r.items())
        print(f"  B={B} H={H} T={T} N={N}:  {details}  [{ok}]")
