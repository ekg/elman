"""Pararnn sequential fwd/bwd with FP8-E4M3 S_traj storage.

Identical math to pararnn_seq_fwd_v2.py, but S_traj is allocated in
torch.float8_e4m3fn and Triton stores/loads via tl.float8e4nv.  This
roughly halves the HBM bandwidth for the trajectory tensor (which is the
dominant HBM cost of the fwd).

Pass storage order (contig [B, H, T, N, N]) is unchanged.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _seq_fwd_v2_fp8_kernel(
    S0_ptr,      # [B, H, N, N]        bf16/fp16/fp32
    K_ptr,       # [B, H, T, N]
    V_ptr,       # [B, H, T, N]
    decay_ptr,   # [B, H, T]
    S_traj_ptr,  # [B, H, T, N, N]     fp8_e4m3 — stores S_1..S_T
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
    K_head_stride = T * N
    dec_head_stride = T

    S = tl.load(S0_ptr + bh * S_head_stride + tile_2d).to(tl.float32)
    for t in range(T):
        K_t = tl.load(K_ptr + bh * K_head_stride + t * N + col_idx).to(tl.float32)
        V_t = tl.load(V_ptr + bh * K_head_stride + t * N + row_idx).to(tl.float32)
        dec = tl.load(decay_ptr + bh * dec_head_stride + t).to(tl.float32)
        retrieved = tl.sum(S * K_t[None, :], axis=1)
        delta_row = V_t - retrieved
        pre = dec * S + delta_row[:, None] * K_t[None, :]
        e2x = tl.exp(2.0 * pre)
        S = (e2x - 1.0) / (e2x + 1.0)
        # FP8 store: cast the fp32 S to E4M3 and write.
        # Triton will do in-register conversion (no separate kernel).
        tl.store(
            S_traj_ptr + bh * traj_head_stride + t * N * N + tile_2d,
            S.to(tl.float8e4nv),
        )


def pararnn_seq_fwd_v2_fp8(S0, K, V, decay, num_warps=1):
    """Forward that stores S_1..S_T as fp8-E4M3 in a [B, H, T, N, N] buffer.

    Other arg dtypes unchanged (typically bf16).  Returns an fp8 tensor.
    """
    B, H, T, N = K.shape
    S_traj = torch.empty(B, H, T, N, N, dtype=torch.float8_e4m3fn, device=S0.device)
    grid = (B * H,)
    _seq_fwd_v2_fp8_kernel[grid](
        S0.contiguous(), K.contiguous(), V.contiguous(), decay.contiguous(),
        S_traj,
        B=B, H=H, T=T, N=N,
        num_warps=num_warps, num_stages=1,
    )
    return S_traj


@triton.jit
def _bwd_v2_fp8_kernel(
    S0_ptr,      # [B, H, N, N]       bf16/fp32
    S_traj_ptr,  # [B, H, T, N, N]    fp8_e4m3 — stores S_1..S_T
    K_ptr, V_ptr, decay_ptr,
    g_T_ptr, dL_dout_ptr, q_ptr,
    g_out_ptr, dK_ptr, dV_ptr, ddecay_ptr, dQ_ptr,
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, N: tl.constexpr,
):
    """Backward that loads S_prev as fp8-E4M3 and hydrates to fp32 in-reg."""
    pid = tl.program_id(0).to(tl.int64)
    b = pid // H
    h = pid % H
    bh = b * H + h
    row_idx = tl.arange(0, N).to(tl.int64)
    col_idx = tl.arange(0, N).to(tl.int64)
    tile_2d = row_idx[:, None] * N + col_idx[None, :]
    S_head_stride = N * N
    traj_head_stride = T * N * N
    K_head_stride = T * N
    dec_head_stride = T

    g = tl.load(g_T_ptr + bh * S_head_stride + tile_2d).to(tl.float32)

    # Main reverse loop: t = T-1 ... 1 (reads S_traj[t-1] as fp8)
    for t_inv in range(T - 1):
        t = T - 1 - t_inv

        dL_dout_t = tl.load(dL_dout_ptr + bh * K_head_stride + t * N + row_idx).to(tl.float32)
        q_t = tl.load(q_ptr + bh * K_head_stride + t * N + col_idx).to(tl.float32)
        g = g + dL_dout_t[:, None] * q_t[None, :]

        K_t = tl.load(K_ptr + bh * K_head_stride + t * N + col_idx).to(tl.float32)
        V_t = tl.load(V_ptr + bh * K_head_stride + t * N + row_idx).to(tl.float32)
        dec = tl.load(decay_ptr + bh * dec_head_stride + t).to(tl.float32)

        # fp8 load: element_ty is fp8e4nv, convert to fp32 on load
        S_prev = tl.load(
            S_traj_ptr + bh * traj_head_stride + (t - 1) * N * N + tile_2d
        ).to(tl.float32)

        retrieved = tl.sum(S_prev * K_t[None, :], axis=1)
        delta_row = V_t - retrieved
        pre = dec * S_prev + delta_row[:, None] * K_t[None, :]
        e2x = tl.exp(2.0 * pre)
        tanh_val = (e2x - 1.0) / (e2x + 1.0)
        tanh_deriv = 1.0 - tanh_val * tanh_val

        dQ_t = tl.sum(dL_dout_t[:, None] * tanh_val, axis=0)
        tl.store(dQ_ptr + bh * K_head_stride + t * N + col_idx, dQ_t)

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
        g = D_mat * g - K_t[None, :] * gu_row[:, None]

    # Final step: t=0 with S_prev = S0 (still fp32/bf16)
    dL_dout_0 = tl.load(dL_dout_ptr + bh * K_head_stride + 0 * N + row_idx).to(tl.float32)
    q_0 = tl.load(q_ptr + bh * K_head_stride + 0 * N + col_idx).to(tl.float32)
    g = g + dL_dout_0[:, None] * q_0[None, :]

    K_0 = tl.load(K_ptr + bh * K_head_stride + 0 * N + col_idx).to(tl.float32)
    V_0 = tl.load(V_ptr + bh * K_head_stride + 0 * N + row_idx).to(tl.float32)
    dec_0 = tl.load(decay_ptr + bh * dec_head_stride + 0).to(tl.float32)

    S_prev_0 = tl.load(S0_ptr + bh * S_head_stride + tile_2d).to(tl.float32)

    retrieved = tl.sum(S_prev_0 * K_0[None, :], axis=1)
    delta_row = V_0 - retrieved
    pre = dec_0 * S_prev_0 + delta_row[:, None] * K_0[None, :]
    e2x = tl.exp(2.0 * pre)
    tanh_val = (e2x - 1.0) / (e2x + 1.0)
    tanh_deriv = 1.0 - tanh_val * tanh_val

    dQ_0 = tl.sum(dL_dout_0[:, None] * tanh_val, axis=0)
    tl.store(dQ_ptr + bh * K_head_stride + 0 * N + col_idx, dQ_0)

    u_mat = tanh_deriv * K_0[None, :]
    gu_row = tl.sum(g * u_mat, axis=1)
    tl.store(dV_ptr + bh * K_head_stride + 0 * N + row_idx, gu_row)

    g_times_tanhd = g * tanh_deriv
    dK_contrib = delta_row[:, None] * g_times_tanhd - S_prev_0 * gu_row[:, None]
    dK_0 = tl.sum(dK_contrib, axis=0)
    tl.store(dK_ptr + bh * K_head_stride + 0 * N + col_idx, dK_0)

    ddec_0 = tl.sum(g_times_tanhd * S_prev_0)
    tl.store(ddecay_ptr + bh * dec_head_stride + 0, ddec_0)

    D_mat = dec_0 * tanh_deriv
    g = D_mat * g - K_0[None, :] * gu_row[:, None]

    tl.store(g_out_ptr + bh * S_head_stride + tile_2d, g.to(g_out_ptr.dtype.element_ty))


def backward_v2_fp8(S0, S_traj_fp8, K, V, decay, g_T, dL_dout, q,
                    num_warps=1, num_stages=1):
    """Backward consuming fp8-E4M3 S_traj. Same interface as backward_v2."""
    B, H, T, N = K.shape
    device = S_traj_fp8.device
    # Gradient dtypes match the other inputs (not fp8)
    out_dtype = K.dtype
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)
    dQ = torch.zeros_like(q)
    ddec = torch.zeros_like(decay)
    dS0 = torch.zeros(B, H, N, N, dtype=out_dtype, device=device)
    grid = (B * H,)
    _bwd_v2_fp8_kernel[grid](
        S0.contiguous(), S_traj_fp8.contiguous(),
        K.contiguous(), V.contiguous(), decay.contiguous(),
        g_T.contiguous(), dL_dout.contiguous(), q.contiguous(),
        dS0, dK, dV, ddec, dQ,
        B=B, H=H, T=T, N=N,
        num_warps=num_warps, num_stages=num_stages,
    )
    return dS0, dK, dV, dQ, ddec


if __name__ == '__main__':
    # Smoke test + compare against bf16 variant.
    import sys, os, time
    sys.path.insert(0, '/home/erikg/elman')
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pararnn_seq_fwd_v2 import pararnn_seq_fwd_v2, backward_v2

    print("FP8 vs BF16 correctness (each compared to bf16 reference):\n")
    for B, H, T, N in [(1, 4, 512, 16), (1, 4, 1024, 16),
                        (1, 4, 4096, 16), (1, 4, 16384, 16),
                        (1, 4, 1024, 32), (1, 4, 4096, 32)]:
        dt = torch.bfloat16
        torch.manual_seed(0)
        K = 0.3 * torch.randn(B, H, T, N, dtype=dt, device='cuda')
        V = 0.3 * torch.randn(B, H, T, N, dtype=dt, device='cuda')
        q = 0.3 * torch.randn(B, H, T, N, dtype=dt, device='cuda')
        decay = torch.sigmoid(0.5 + 0.1 * torch.randn(B, H, T, dtype=dt, device='cuda'))
        S0 = 0.1 * torch.randn(B, H, N, N, dtype=dt, device='cuda')
        dL_dout = 0.01 * torch.randn(B, H, T, N, dtype=dt, device='cuda')
        g_T = torch.zeros(B, H, N, N, dtype=dt, device='cuda')
        nw_fwd = 1 if N == 16 else 4
        nw_bwd = 1 if N == 16 else 2

        # BF16 reference
        S_traj_bf = pararnn_seq_fwd_v2(S0, K, V, decay, num_warps=nw_fwd)
        dS0_bf, dK_bf, dV_bf, dQ_bf, ddec_bf = backward_v2(
            S0, S_traj_bf, K, V, decay, g_T, dL_dout, q,
            num_warps=nw_bwd, num_stages=1)

        # FP8 variant
        S_traj_fp8 = pararnn_seq_fwd_v2_fp8(S0, K, V, decay, num_warps=nw_fwd)
        # Measure S_traj error
        s_err = (S_traj_fp8.to(torch.float32) - S_traj_bf.to(torch.float32)).abs().max().item()
        s_bf_max = S_traj_bf.to(torch.float32).abs().max().item()
        s_rel = s_err / max(s_bf_max, 1e-10)

        dS0_f8, dK_f8, dV_f8, dQ_f8, ddec_f8 = backward_v2_fp8(
            S0, S_traj_fp8, K, V, decay, g_T, dL_dout, q,
            num_warps=nw_bwd, num_stages=1)

        def rel(a, b):
            num = (a.float() - b.float()).abs().max().item()
            denom = max(b.float().abs().max().item(), 1e-10)
            return num / denom

        print(f"  B={B} H={H} T={T:5d} N={N}:  "
              f"S_rel={s_rel:.2e}  dK={rel(dK_f8, dK_bf):.2e}  dV={rel(dV_f8, dV_bf):.2e}  "
              f"dQ={rel(dQ_f8, dQ_bf):.2e}  ddec={rel(ddec_f8, ddec_bf):.2e}  "
              f"dS0={rel(dS0_f8, dS0_bf):.2e}")
