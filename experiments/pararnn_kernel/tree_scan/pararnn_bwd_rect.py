"""Rectangular-state Triton backward for Pararnn E88.

Computes dL/d(S0, K, V, decay) given:
  S_traj: [B, H, T+1, M, N]     Pararnn-convention (Pararnn_S = CUDA_S^T)
  K, Q:   [B, H, T, N]           Pararnn k,q — contract along N dim
  V:      [B, H, T, M]           Pararnn v  — indexed by M dim
  decay:  [B, H, T]
  g_T:    [B, H, M, N]           dL/dS at t=T (Pararnn-convention)
  dL_dout:[B, H, T, M]           dL/doutput (output is indexed by M)

dQ is computed separately as einsum('bhti,bhtij->bhtj', dL_dout, S_traj[:,:,1:]).

This mirrors backward_e88_fused_rank1 from phase7_fused_backward.py but
with independent M (rows of Pararnn S) and N (cols).  Also applies the
timestep-off-by-one fix (inject output grad at start of iter, not end).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _pararnn_bwd_rect_kernel(
    S_traj_ptr,    # [B, H, T+1, M, N]
    K_ptr,         # [B, H, T, N]
    V_ptr,         # [B, H, T, M]
    decay_ptr,     # [B, H, T]
    g_T_ptr,       # [B, H, M, N]
    dL_dout_ptr,   # [B, H, T, M]
    q_ptr,         # [B, H, T, N]
    g_out_ptr,     # [B, H, M, N]  dS0
    dK_ptr,        # [B, H, T, N]
    dV_ptr,        # [B, H, T, M]
    ddecay_ptr,    # [B, H, T]
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr,
    M: tl.constexpr, N: tl.constexpr,
    M_P2: tl.constexpr, N_P2: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    b = pid // H
    h = pid % H
    bh = b * H + h

    row_idx = tl.arange(0, M_P2).to(tl.int64)
    col_idx = tl.arange(0, N_P2).to(tl.int64)
    row_mask = row_idx < M
    col_mask = col_idx < N
    mask_2d = row_mask[:, None] & col_mask[None, :]
    r_safe = tl.where(row_mask, row_idx, 0)
    c_safe = tl.where(col_mask, col_idx, 0)
    tile_2d = r_safe[:, None] * N + c_safe[None, :]

    gT_base = bh * M * N
    S_traj_head_stride = (T + 1) * M * N
    K_head_stride = T * N          # K, q
    V_head_stride = T * M          # V, dL_dout
    dec_head_stride = T

    g = tl.load(g_T_ptr + gT_base + tile_2d, mask=mask_2d, other=0.0).to(tl.float32)

    for t_inv in range(T):
        t = T - 1 - t_inv

        # Inject output_t contribution: dL_dout[t, p] * q[t, q] into dL/dS_{t+1}.
        dL_dout_t = tl.load(dL_dout_ptr + bh * V_head_stride + t * M + r_safe,
                            mask=row_mask, other=0.0).to(tl.float32)
        q_t = tl.load(q_ptr + bh * K_head_stride + t * N + c_safe,
                      mask=col_mask, other=0.0).to(tl.float32)
        g = g + dL_dout_t[:, None] * q_t[None, :]
        g = tl.where(mask_2d, g, 0.0)

        K_t = tl.load(K_ptr + bh * K_head_stride + t * N + c_safe,
                      mask=col_mask, other=0.0).to(tl.float32)
        V_t = tl.load(V_ptr + bh * V_head_stride + t * M + r_safe,
                      mask=row_mask, other=0.0).to(tl.float32)
        dec = tl.load(decay_ptr + bh * dec_head_stride + t).to(tl.float32)

        S_prev = tl.load(S_traj_ptr + bh * S_traj_head_stride + t * M * N + tile_2d,
                         mask=mask_2d, other=0.0).to(tl.float32)

        retrieved = tl.sum(S_prev * K_t[None, :], axis=1)       # [M]
        delta_row = V_t - retrieved

        pre = dec * S_prev + delta_row[:, None] * K_t[None, :]
        e2x = tl.exp(2.0 * pre)
        tanh_val = (e2x - 1.0) / (e2x + 1.0)
        tanh_deriv = 1.0 - tanh_val * tanh_val
        tanh_deriv = tl.where(mask_2d, tanh_deriv, 0.0)

        u_mat = tanh_deriv * K_t[None, :]
        gu_row = tl.sum(g * u_mat, axis=1)                      # [M]
        tl.store(dV_ptr + bh * V_head_stride + t * M + r_safe,
                 gu_row, mask=row_mask)

        g_times_tanhd = g * tanh_deriv
        dK_contrib = delta_row[:, None] * g_times_tanhd - S_prev * gu_row[:, None]
        dK_contrib = tl.where(mask_2d, dK_contrib, 0.0)
        dK_t = tl.sum(dK_contrib, axis=0)                       # [N]
        tl.store(dK_ptr + bh * K_head_stride + t * N + c_safe,
                 dK_t, mask=col_mask)

        ddec_t = tl.sum(tl.where(mask_2d, g_times_tanhd * S_prev, 0.0))
        tl.store(ddecay_ptr + bh * dec_head_stride + t, ddec_t)

        D_mat = dec * tanh_deriv
        g = D_mat * g - K_t[None, :] * gu_row[:, None]
        g = tl.where(mask_2d, g, 0.0)

    tl.store(g_out_ptr + gT_base + tile_2d,
             g.to(g_out_ptr.dtype.element_ty), mask=mask_2d)


def _next_pow2(x):
    p = 1
    while p < x:
        p <<= 1
    return p


def pararnn_bwd_rect(S_traj, K, V, decay, g_T, dL_dout, q, num_warps=1, num_stages=1):
    """
    S_traj: [B, H, T+1, M, N]
    K, Q:   [B, H, T, N]
    V:      [B, H, T, M]
    decay:  [B, H, T]
    g_T:    [B, H, M, N]
    dL_dout:[B, H, T, M]
    Returns (dS0 [B,H,M,N], dK [B,H,T,N], dV [B,H,T,M], ddec [B,H,T]).
    """
    B, H, T1, M, N = S_traj.shape
    T = T1 - 1
    device = S_traj.device
    dtype = S_traj.dtype
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)
    ddec = torch.zeros_like(decay)
    dS0 = torch.zeros(B, H, M, N, dtype=dtype, device=device)

    M_P2 = _next_pow2(M)
    N_P2 = _next_pow2(N)

    grid = (B * H,)
    _pararnn_bwd_rect_kernel[grid](
        S_traj.contiguous(), K.contiguous(), V.contiguous(), decay.contiguous(),
        g_T.contiguous(), dL_dout.contiguous(), q.contiguous(),
        dS0, dK, dV, ddec,
        B=B, H=H, T=T, M=M, N=N, M_P2=M_P2, N_P2=N_P2,
        num_warps=num_warps, num_stages=num_stages,
    )
    return dS0, dK, dV, ddec


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, '/home/erikg/elman')
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pararnn_seq_fwd_rect import pararnn_seq_fwd_output_triton, cuda_forward_py

    # End-to-end correctness at rectangular fp32 vs autograd.
    for B, H, T, Ns, Hv in [
        (1, 4, 256, 16, 16),
        (1, 4, 256, 32, 32),
        (1, 4, 256, 32, 24),
        (1, 4, 256, 32, 23),
        (1, 4, 256, 16, 14),
    ]:
        dt = torch.float32
        torch.manual_seed(0)
        k = (0.3 * torch.randn(T, B, H, Ns, dtype=dt, device='cuda')).requires_grad_(True)
        v = (0.3 * torch.randn(T, B, H, Hv, dtype=dt, device='cuda')).requires_grad_(True)
        q = (0.3 * torch.randn(T, B, H, Ns, dtype=dt, device='cuda')).requires_grad_(True)
        decay = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, dtype=dt, device='cuda')).detach().requires_grad_(True)
        S0 = (0.1 * torch.randn(B, H, Ns, Hv, dtype=dt, device='cuda')).requires_grad_(True)

        S_final_ref, output_ref = cuda_forward_py(S0, k, v, q, decay)
        torch.manual_seed(1)
        dL_dout = 0.01 * torch.randn_like(output_ref)
        # Match CUDA convention: no dS_final term — use output only.
        loss = (output_ref * dL_dout).sum()
        loss.backward()

        dK_ref, dV_ref, dQ_ref = k.grad.clone(), v.grad.clone(), q.grad.clone()
        ddec_ref = decay.grad.clone()
        dS0_ref = S0.grad.clone()

        # Hybrid: Triton fwd + Triton bwd (rect) with transpose convention
        with torch.no_grad():
            K_p = k.detach().permute(1, 2, 0, 3).contiguous()  # [B, H, T, Ns=N]
            V_p = v.detach().permute(1, 2, 0, 3).contiguous()  # [B, H, T, Hv=M]
            Q_p = q.detach().permute(1, 2, 0, 3).contiguous()
            decay_p = decay.detach().permute(1, 2, 0).contiguous()
            S0_p = S0.detach().transpose(-1, -2).contiguous()  # [B, H, M, N]

            S_traj, Sq = pararnn_seq_fwd_output_triton(
                S0_p, K_p, V_p, Q_p, decay_p,
                num_warps=4 if Ns * Hv >= 32 * 32 else 1,
            )

            dL_dout_p = dL_dout.permute(1, 2, 0, 3).contiguous()  # [B, H, T, Hv=M]
            g_T = torch.zeros(B, H, Hv, Ns, dtype=dt, device='cuda')  # dS_final = 0

            dS0_p, dK_p, dV_p, ddec_p = pararnn_bwd_rect(
                S_traj, K_p, V_p, decay_p, g_T, dL_dout_p, Q_p,
                num_warps=4 if Ns * Hv >= 32 * 32 else 1,
            )
            dQ_p = torch.einsum('bhti,bhtij->bhtj', dL_dout_p, S_traj[:, :, 1:])

            dK_hyb = dK_p.permute(2, 0, 1, 3).contiguous()
            dV_hyb = dV_p.permute(2, 0, 1, 3).contiguous()
            dQ_hyb = dQ_p.permute(2, 0, 1, 3).contiguous()
            ddec_hyb = ddec_p.permute(2, 0, 1).contiguous()
            dS0_hyb = dS0_p.transpose(-1, -2).contiguous()

        def rel(a, b):
            num = (a - b).abs().max().item()
            denom = max(b.abs().max().item(), 1e-10)
            return num / denom

        # dS0_ref receives a contribution from "S_final" (but we set dS_final via output only)
        # So dS0_ref here is the result of propagation through the whole RNN — hybrid should match.
        rs = {
            'dK': rel(dK_hyb, dK_ref),
            'dV': rel(dV_hyb, dV_ref),
            'dQ': rel(dQ_hyb, dQ_ref),
            'ddec': rel(ddec_hyb, ddec_ref),
            'dS0': rel(dS0_hyb, dS0_ref),
        }
        worst = max(rs.values())
        ok = "PASS" if worst < 1e-3 else "FAIL"
        shape_str = "sq" if Ns == Hv else f"rect"
        details = "  ".join(f"{k}={v:.1e}" for k, v in rs.items())
        print(f"  T={T:4d} Ns={Ns} Hv={Hv} ({shape_str}):  {details}  [{ok}]")
