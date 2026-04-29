"""E93 backward Triton kernel — gradients including dW_h.

Parametrized with ablation flags so each variant gets a specialized kernel.

Per program (b, m_tile), backward in time:
  d_pre = dS_full * dNL/dpre   (tanh: 1-S^2; linear: 1)
  Through outer(k, delta):
    ddelta_tile = sum_n d_pre[n,m] * k[n]   (always)
    dk_from_outer = sum_m d_pre[:,m] * delta[m]
  Through delta = v - retrieved (only if USE_DELTA):
    dv_tile = ddelta_tile
    dretrieved = -ddelta_tile
    dS_prev_from_retr = k[:, None] * dretrieved[None, :]
    dk_from_retr = sum_m S_prev[:,m] * dretrieved[m]
  Else: dv_tile = ddelta_tile; dS_prev_from_retr = 0
  Through dec*Wh_S (only if USE_DECAY):
    d_alpha = sum(d_pre * Wh_S)  (atomic_add)
    d_Wh_S = dec * d_pre
  Else: d_Wh_S = d_pre
  Through W_h @ S_prev (only if USE_W_H):
    dW_h += d_Wh_S @ S_prev^T  (atomic_add at end, local accumulate)
    dS_prev_from_Wh = W_h^T @ d_Wh_S
  Else: dS_prev_from_Wh = d_Wh_S
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _e93_bwd_kernel(
    S0_ptr, S_traj_ptr, K_ptr, V_ptr, decay_ptr, W_h_ptr,
    d_Sflat_ptr, d_S_T_ptr,
    dK_ptr, dV_ptr, ddec_ptr, dS0_ptr, dW_h_ptr,
    k_b, k_t, k_n,
    v_b, v_t, v_m,
    dec_b, dec_t,
    wh_n1, wh_n2,
    s0_b, s0_n, s0_m,
    st_b, st_t, st_n, st_m,
    sf_b, sf_t, sf_nm,
    B: tl.constexpr, T: tl.constexpr, N: tl.constexpr,
    M: tl.constexpr, M_TILE: tl.constexpr,
    USE_W_H: tl.constexpr, USE_DECAY: tl.constexpr,
    USE_DELTA: tl.constexpr, NL_KIND: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    M_TILES = M // M_TILE
    b = pid // M_TILES
    m_tile = pid % M_TILES
    m_offset = m_tile * M_TILE

    n_idx = tl.arange(0, N).to(tl.int64)
    m_idx = tl.arange(0, M_TILE).to(tl.int64)
    m_global_idx = m_offset + m_idx

    if USE_W_H:
        wh_tile = n_idx[:, None] * wh_n1 + n_idx[None, :] * wh_n2
        W_h = tl.load(W_h_ptr + wh_tile).to(tl.float32)
        W_h_T = tl.trans(W_h)
        dWh_local = tl.zeros([N, N], dtype=tl.float32)

    dst_off = b * s0_b + n_idx[:, None] * s0_n + m_global_idx[None, :] * s0_m
    dS = tl.load(d_S_T_ptr + dst_off).to(tl.float32)

    for t_rev in range(T):
        t = T - 1 - t_rev

        st_off = b * st_b + t * st_t + n_idx[:, None] * st_n + m_global_idx[None, :] * st_m
        S_t = tl.load(S_traj_ptr + st_off).to(tl.float32)
        if t > 0:
            stp_off = b * st_b + (t - 1) * st_t + n_idx[:, None] * st_n + m_global_idx[None, :] * st_m
            S_prev = tl.load(S_traj_ptr + stp_off).to(tl.float32)
        else:
            s0_off = b * s0_b + n_idx[:, None] * s0_n + m_global_idx[None, :] * s0_m
            S_prev = tl.load(S0_ptr + s0_off).to(tl.float32)

        k_off = b * k_b + t * k_t + n_idx * k_n
        k_t_v = tl.load(K_ptr + k_off).to(tl.float32)
        v_off = b * v_b + t * v_t + m_global_idx * v_m
        v_tile = tl.load(V_ptr + v_off).to(tl.float32)
        if USE_DECAY:
            dec = tl.load(decay_ptr + b * dec_b + t * dec_t).to(tl.float32)
        else:
            dec = 1.0

        flat_idx = n_idx[:, None] * M + m_global_idx[None, :]
        df_off = b * sf_b + t * sf_t + flat_idx * sf_nm
        d_Sflat = tl.load(d_Sflat_ptr + df_off).to(tl.float32)
        dS_full = dS + d_Sflat

        # Through nonlinearity
        if NL_KIND == 0:  # tanh
            dpre = dS_full * (1.0 - S_t * S_t)
        else:  # linear
            dpre = dS_full

        # Recompute retrieved/delta only if needed
        if USE_DELTA:
            retrieved_tile = tl.sum(S_prev * k_t_v[:, None], axis=0)
            delta_tile = v_tile - retrieved_tile
        else:
            delta_tile = v_tile

        # === Through dec * Wh_S ===
        if USE_W_H:
            Wh_S = tl.dot(W_h, S_prev, allow_tf32=False)
        else:
            Wh_S = S_prev

        if USE_DECAY:
            d_alpha_part = tl.sum(dpre * Wh_S)
            tl.atomic_add(ddec_ptr + b * dec_b + t * dec_t, d_alpha_part)
            d_Wh_S = dec * dpre
        else:
            d_Wh_S = dpre

        # === Through W_h @ S_prev ===
        if USE_W_H:
            S_prev_T = tl.trans(S_prev)
            dWh_step = tl.dot(d_Wh_S, S_prev_T, allow_tf32=False)
            dWh_local += dWh_step
            dS_prev = tl.dot(W_h_T, d_Wh_S, allow_tf32=False)
        else:
            dS_prev = d_Wh_S

        # === Through outer(k, delta) ===
        ddelta_tile = tl.sum(dpre * k_t_v[:, None], axis=0)
        dk_from_outer = tl.sum(dpre * delta_tile[None, :], axis=1)

        tl.store(dV_ptr + b * v_b + t * v_t + m_global_idx * v_m,
                 ddelta_tile.to(dV_ptr.dtype.element_ty))

        # === Through retrieved (only if USE_DELTA) ===
        if USE_DELTA:
            dretrieved = -ddelta_tile
            dS_prev_from_retr = k_t_v[:, None] * dretrieved[None, :]
            dS_prev += dS_prev_from_retr
            dk_from_retr = tl.sum(S_prev * dretrieved[None, :], axis=1)
            dk_partial = dk_from_outer + dk_from_retr
        else:
            dk_partial = dk_from_outer

        tl.atomic_add(dK_ptr + b * k_b + t * k_t + n_idx * k_n, dk_partial)

        dS = dS_prev

    if USE_W_H:
        dwh_off = n_idx[:, None] * wh_n1 + n_idx[None, :] * wh_n2
        tl.atomic_add(dW_h_ptr + dwh_off, dWh_local)

    s0_out_off = b * s0_b + n_idx[:, None] * s0_n + m_global_idx[None, :] * s0_m
    tl.store(dS0_ptr + s0_out_off, dS.to(dS0_ptr.dtype.element_ty))


def e93_seq_bwd(S0, S_traj, W_h, K, V, decay, d_Sflat, d_S_T, M_TILE=64, num_warps=2,
                use_w_h=True, use_decay=True, use_delta=True, nl_kind=0):
    """E93 backward."""
    B, T, N = K.shape
    M = V.shape[-1]
    assert M % M_TILE == 0

    dK = torch.zeros_like(K)
    dV = torch.empty_like(V)
    ddec = torch.zeros_like(decay)
    dS0 = torch.empty_like(S0)
    dW_h = torch.zeros_like(W_h, dtype=torch.float32)

    k_b, k_t, k_n = K.stride()
    v_b, v_t, v_m = V.stride()
    dec_b, dec_t = decay.stride()
    wh_n1, wh_n2 = W_h.stride()
    s0_b, s0_n, s0_m = S0.stride()
    st_b, st_t, st_n, st_m = S_traj.stride()
    sf_b, sf_t, sf_nm = d_Sflat.stride()

    M_TILES = M // M_TILE
    grid = (B * M_TILES,)
    _e93_bwd_kernel[grid](
        S0.contiguous(), S_traj.contiguous(),
        K, V, decay, W_h.contiguous(),
        d_Sflat.contiguous(), d_S_T.contiguous(),
        dK, dV, ddec, dS0, dW_h,
        k_b, k_t, k_n,
        v_b, v_t, v_m,
        dec_b, dec_t,
        wh_n1, wh_n2,
        s0_b, s0_n, s0_m,
        st_b, st_t, st_n, st_m,
        sf_b, sf_t, sf_nm,
        B=B, T=T, N=N, M=M, M_TILE=M_TILE,
        USE_W_H=use_w_h, USE_DECAY=use_decay,
        USE_DELTA=use_delta, NL_KIND=nl_kind,
        num_warps=num_warps,
    )
    return dS0, dW_h.to(W_h.dtype), dK, dV, ddec


if __name__ == '__main__':
    import os, sys
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '5')
    THIS = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, THIS)
    from e93_seq_fwd import e93_seq_fwd, e93_seq_fwd_torch_ref

    print("E93 backward correctness across ablation flags:\n")
    configs = [
        ('vanilla',  dict(use_w_h=True,  use_decay=True,  use_delta=True,  nl_kind=0)),
        ('no_wh',    dict(use_w_h=False, use_decay=True,  use_delta=True,  nl_kind=0)),
        ('no_delta', dict(use_w_h=True,  use_decay=True,  use_delta=False, nl_kind=0)),
        ('no_decay', dict(use_w_h=True,  use_decay=False, use_delta=True,  nl_kind=0)),
        ('linear',   dict(use_w_h=True,  use_decay=True,  use_delta=True,  nl_kind=1)),
        ('min_tanh', dict(use_w_h=False, use_decay=False, use_delta=False, nl_kind=0)),
        ('min_lin',  dict(use_w_h=False, use_decay=False, use_delta=False, nl_kind=1)),
    ]
    for name, flags in configs:
        worst_all = 0.0
        for B, T, N, M in [(1, 16, 16, 64), (2, 32, 16, 128)]:
            torch.manual_seed(0)
            dt = torch.float32
            S0 = (0.1 * torch.randn(B, N, M, dtype=dt, device='cuda').contiguous()).requires_grad_(True)
            W_h = ((torch.eye(N) + 0.05 * torch.randn(N, N)).to('cuda').contiguous()).requires_grad_(True)
            K_raw = (0.3 * torch.randn(B, T, N, dtype=dt, device='cuda')).requires_grad_(True)
            V = (0.3 * torch.randn(B, T, M, dtype=dt, device='cuda')).requires_grad_(True)
            decay = torch.sigmoid(0.5 + 0.1 * torch.randn(B, T, dtype=dt, device='cuda')).requires_grad_(True)
            K = torch.nn.functional.normalize(K_raw, dim=-1)

            S_traj_ref, Sflat_ref = e93_seq_fwd_torch_ref(S0, W_h, K, V, decay, **flags)
            d_Sflat = torch.randn_like(Sflat_ref)
            d_S_T = torch.randn_like(S_traj_ref[:, -1])
            loss_ref = (Sflat_ref * d_Sflat).sum() + (S_traj_ref[:, -1] * d_S_T).sum()
            grads_ref = torch.autograd.grad(
                loss_ref, [S0, W_h, K, V, decay], retain_graph=False,
                allow_unused=True,
            )
            dS0_ref, dWh_ref, dK_ref, dV_ref, ddec_ref = grads_ref
            if dWh_ref is None:
                dWh_ref = torch.zeros_like(W_h)
            if ddec_ref is None:
                ddec_ref = torch.zeros_like(decay)

            with torch.no_grad():
                S_traj, Sflat = e93_seq_fwd(S0.detach(), W_h.detach(), K.detach(), V.detach(),
                                            decay.detach(), M_TILE=min(M, 64), **flags)
                dS0_t, dWh_t, dK_t, dV_t, ddec_t = e93_seq_bwd(
                    S0.detach(), S_traj, W_h.detach(),
                    K.detach(), V.detach(), decay.detach(),
                    d_Sflat, d_S_T, M_TILE=min(M, 64), **flags,
                )

            def rel(a, b): return (a - b).abs().max().item() / max(b.abs().max().item(), 1e-10)
            rs = rel(dS0_t, dS0_ref)
            rwh = rel(dWh_t, dWh_ref) if flags['use_w_h'] else 0.0
            rk = rel(dK_t, dK_ref)
            rv = rel(dV_t, dV_ref)
            rd = rel(ddec_t, ddec_ref) if flags['use_decay'] else 0.0
            worst = max(rs, rwh, rk, rv, rd)
            worst_all = max(worst_all, worst)
            ok = "PASS" if worst < 1e-3 else "FAIL"
            print(f"  {name:10s} B={B} T={T} N={N} M={M}: dS0={rs:.1e} dWh={rwh:.1e} dK={rk:.1e} dV={rv:.1e} ddec={rd:.1e}  [{ok}]")
