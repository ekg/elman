"""E93 backward Triton kernel — gradients including dW_h.

Per program (b, m_tile), backward in time from t=T-1 to t=0:
  Recompute retrieved, delta from S_prev (saved in S_traj)
  Compute d_pre = dS_t * (1 - S_t^2)
  Through "outer(k, delta)":
    ddelta_tile = d_pre^T @ k  [M_TILE]
    dk_from_outer (partial for this tile) = sum_m d_pre[:, m] * delta[m]   [N]
  Through "delta = v - retrieved":
    dv_tile = ddelta_tile
    dretrieved = -ddelta_tile
  Through "retrieved[m] = S_prev[:, m]·k":
    dS_prev_from_retr[:, m] = dretrieved[m] * k    (rank-1 outer per column)
    dk_from_retr (partial) = sum_m S_prev[:, m] * dretrieved[m]   [N]
  Through "alpha * Wh_S":
    d_alpha_partial = sum(d_pre * Wh_S)   scalar
    d_Wh_S = alpha * d_pre
    dW_h_partial += d_Wh_S @ S_prev^T   [N, N]
    dS_prev_from_Wh = W_h^T @ d_Wh_S   [N, M_TILE]
  dS_prev = dS_prev_from_retr + dS_prev_from_Wh
  Atomic_add: dk_partial → dK[b, t], d_alpha_partial → ddec[b, t]
  Store dv_tile → dV[b, t, m_tile_range]

After T loop:
  Atomic_add dW_h_partial → dW_h
  Store dS_prev → dS0[b, :, m_tile_range]
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _e93_bwd_kernel(
    S0_ptr, S_traj_ptr, K_ptr, V_ptr, decay_ptr, W_h_ptr,
    d_Sflat_ptr,      # [B, T, N*M]  — gradient of Sflat output
    d_S_T_ptr,        # [B, N, M]    — gradient of final state
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
):
    pid = tl.program_id(0).to(tl.int64)
    M_TILES = M // M_TILE
    b = pid // M_TILES
    m_tile = pid % M_TILES
    m_offset = m_tile * M_TILE

    n_idx = tl.arange(0, N).to(tl.int64)
    m_idx = tl.arange(0, M_TILE).to(tl.int64)
    m_global_idx = m_offset + m_idx
    wh_tile = n_idx[:, None] * wh_n1 + n_idx[None, :] * wh_n2

    W_h = tl.load(W_h_ptr + wh_tile).to(tl.float32)
    W_h_T = tl.trans(W_h)

    # Local accumulator for dW_h
    dWh_local = tl.zeros([N, N], dtype=tl.float32)

    # Initialize dS = d_S_T (this tile)
    dst_off = b * s0_b + n_idx[:, None] * s0_n + m_global_idx[None, :] * s0_m
    dS = tl.load(d_S_T_ptr + dst_off).to(tl.float32)

    for t_rev in range(T):
        t = T - 1 - t_rev

        # Load S_t and S_prev tiles
        st_off = b * st_b + t * st_t + n_idx[:, None] * st_n + m_global_idx[None, :] * st_m
        S_t = tl.load(S_traj_ptr + st_off).to(tl.float32)
        if t > 0:
            stp_off = b * st_b + (t - 1) * st_t + n_idx[:, None] * st_n + m_global_idx[None, :] * st_m
            S_prev = tl.load(S_traj_ptr + stp_off).to(tl.float32)
        else:
            s0_off = b * s0_b + n_idx[:, None] * s0_n + m_global_idx[None, :] * s0_m
            S_prev = tl.load(S0_ptr + s0_off).to(tl.float32)

        # Load K, V, decay, dSflat at t
        k_off = b * k_b + t * k_t + n_idx * k_n
        k_t_v = tl.load(K_ptr + k_off).to(tl.float32)
        v_off = b * v_b + t * v_t + m_global_idx * v_m
        v_tile = tl.load(V_ptr + v_off).to(tl.float32)
        dec = tl.load(decay_ptr + b * dec_b + t * dec_t).to(tl.float32)

        # Add d_Sflat at this t (incoming gradient to S_t via flatten path)
        flat_idx = n_idx[:, None] * M + m_global_idx[None, :]
        df_off = b * sf_b + t * sf_t + flat_idx * sf_nm
        d_Sflat = tl.load(d_Sflat_ptr + df_off).to(tl.float32)
        dS_full = dS + d_Sflat

        # Through tanh
        dpre = dS_full * (1.0 - S_t * S_t)

        # Recompute retrieved and delta
        retrieved_tile = tl.sum(S_prev * k_t_v[:, None], axis=0)
        delta_tile = v_tile - retrieved_tile

        # Recompute Wh_S for d_alpha
        Wh_S = tl.dot(W_h, S_prev, allow_tf32=False)

        # === d_alpha (scalar per tile, accumulate via atomic) ===
        d_alpha_part = tl.sum(dpre * Wh_S)
        tl.atomic_add(ddec_ptr + b * dec_b + t * dec_t, d_alpha_part)

        # === Through alpha * Wh_S ===
        d_Wh_S = dec * dpre
        # dW_h_partial = d_Wh_S @ S_prev^T (small per tile, accumulate locally)
        S_prev_T = tl.trans(S_prev)
        dWh_step = tl.dot(d_Wh_S, S_prev_T, allow_tf32=False)
        dWh_local += dWh_step
        # dS_prev (from Wh_S) = W_h^T @ d_Wh_S
        dS_prev = tl.dot(W_h_T, d_Wh_S, allow_tf32=False)

        # === Through outer(k, delta) ===
        # ddelta_tile = d_pre^T @ k  per column
        # = sum_n d_pre[n, m] * k[n]
        ddelta_tile = tl.sum(dpre * k_t_v[:, None], axis=0)
        # dk_from_outer (partial for this tile) = sum_m d_pre[:, m] * delta[m]
        dk_from_outer = tl.sum(dpre * delta_tile[None, :], axis=1)

        # === dV[t, tile] = ddelta_tile ===
        tl.store(dV_ptr + b * v_b + t * v_t + m_global_idx * v_m,
                 ddelta_tile.to(dV_ptr.dtype.element_ty))

        # === Through retrieved = S_prev^T @ k (per col) ===
        dretrieved = -ddelta_tile  # [M_TILE]
        # dS_prev_from_retr[:, m] = dretrieved[m] * k
        dS_prev_from_retr = k_t_v[:, None] * dretrieved[None, :]   # [N, M_TILE]
        dS_prev += dS_prev_from_retr
        # dk_from_retr (partial) = sum_m S_prev[:, m] * dretrieved[m]
        dk_from_retr = tl.sum(S_prev * dretrieved[None, :], axis=1)

        # === atomic_add partial dk to dK[b, t, :] ===
        dk_partial = dk_from_outer + dk_from_retr
        tl.atomic_add(dK_ptr + b * k_b + t * k_t + n_idx * k_n, dk_partial)

        dS = dS_prev

    # Atomic_add local dW_h to global
    dwh_off = n_idx[:, None] * wh_n1 + n_idx[None, :] * wh_n2
    tl.atomic_add(dW_h_ptr + dwh_off, dWh_local)

    # Store dS0
    s0_out_off = b * s0_b + n_idx[:, None] * s0_n + m_global_idx[None, :] * s0_m
    tl.store(dS0_ptr + s0_out_off, dS.to(dS0_ptr.dtype.element_ty))


def e93_seq_bwd(S0, S_traj, W_h, K, V, decay, d_Sflat, d_S_T, M_TILE=64, num_warps=2):
    """E93 backward.

    Returns: dS0 [B,N,M], dW_h [N,N], dK [B,T,N], dV [B,T,M], ddec [B,T]
    """
    B, T, N = K.shape
    M = V.shape[-1]
    assert M % M_TILE == 0

    dtype = S0.dtype

    dK = torch.zeros_like(K)            # zero — atomic_add accumulates
    dV = torch.empty_like(V)             # written exactly once per element
    ddec = torch.zeros_like(decay)       # zero — atomic_add
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
        num_warps=num_warps,
    )
    return dS0, dW_h.to(W_h.dtype), dK, dV, ddec


if __name__ == '__main__':
    import os, sys
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '5')
    THIS = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, THIS)
    from e93_seq_fwd import e93_seq_fwd, e93_seq_fwd_torch_ref

    print("E93 backward correctness: Triton bwd vs torch.autograd through ref\n")
    for B, T, N, M in [(1, 16, 16, 64), (2, 32, 16, 128)]:
        torch.manual_seed(0)
        dt = torch.float32
        S0 = (0.1 * torch.randn(B, N, M, dtype=dt, device='cuda').contiguous()).requires_grad_(True)
        W_h = ((torch.eye(N) + 0.05 * torch.randn(N, N)).to('cuda').contiguous()).requires_grad_(True)
        K_raw = (0.3 * torch.randn(B, T, N, dtype=dt, device='cuda')).requires_grad_(True)
        V = (0.3 * torch.randn(B, T, M, dtype=dt, device='cuda')).requires_grad_(True)
        decay = torch.sigmoid(0.5 + 0.1 * torch.randn(B, T, dtype=dt, device='cuda')).requires_grad_(True)
        K = torch.nn.functional.normalize(K_raw, dim=-1)

        # Reference forward + autograd
        S_traj_ref, Sflat_ref = e93_seq_fwd_torch_ref(S0, W_h, K, V, decay)
        d_Sflat = torch.randn_like(Sflat_ref)
        d_S_T = torch.randn_like(S_traj_ref[:, -1])
        loss_ref = (Sflat_ref * d_Sflat).sum() + (S_traj_ref[:, -1] * d_S_T).sum()
        dS0_ref, dWh_ref, dK_ref, dV_ref, ddec_ref = torch.autograd.grad(
            loss_ref, [S0, W_h, K, V, decay], retain_graph=False
        )

        # Triton fwd + bwd (use detached, normalized K)
        with torch.no_grad():
            S_traj, Sflat = e93_seq_fwd(S0.detach(), W_h.detach(), K.detach(), V.detach(), decay.detach(), M_TILE=min(M, 64))
            dS0_t, dWh_t, dK_t, dV_t, ddec_t = e93_seq_bwd(
                S0.detach(), S_traj, W_h.detach(),
                K.detach(), V.detach(), decay.detach(),
                d_Sflat, d_S_T, M_TILE=min(M, 64),
            )

        def rel(a, b): return (a - b).abs().max().item() / max(b.abs().max().item(), 1e-10)
        rs = rel(dS0_t, dS0_ref)
        rwh = rel(dWh_t, dWh_ref)
        rk = rel(dK_t, dK_ref)
        rv = rel(dV_t, dV_ref)
        rd = rel(ddec_t, ddec_ref)
        worst = max(rs, rwh, rk, rv, rd)
        ok = "PASS" if worst < 1e-3 else "FAIL"
        print(f"  B={B} T={T} N={N} M={M}: dS0={rs:.1e} dWh={rwh:.1e} dK={rk:.1e} dV={rv:.1e} ddec={rd:.1e}  [{ok}]")
