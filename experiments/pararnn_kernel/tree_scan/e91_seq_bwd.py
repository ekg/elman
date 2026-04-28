"""E91 backward Triton kernel — gradients through rank-r delta rule.

Inputs (from forward):
  S0       [B, H, N, N]
  K, V     [B, T, H, N, R]
  Q        [B, T, H, N]
  decay    [B, T, H]
  S_traj   [B, H, T, N, N]  (state after each step)

Upstream gradients:
  d_Sq     [B, T, H, N]      (gradient of Sq output)
  d_S_T    [B, H, N, N]       (gradient of final state)

Outputs:
  dK, dV   [B, T, H, N, R]
  dQ       [B, T, H, N]
  ddec     [B, T, H]
  dS0      [B, H, N, N]

Algorithm (per (b, h), backward in time):
  dS = d_S_T
  for t = T-1 .. 0:
    S_t = S_traj[t], S_prev = S_traj[t-1] if t>0 else S0
    # Add Sq gradient at this step
    dS += outer(d_Sq[t], Q[t])
    dQ[t] = S_t.T @ d_Sq[t]
    # Through tanh
    dpre = dS * (1 - S_t**2)
    # decay * S_prev contribution
    ddec[t] = sum(dpre * S_prev)
    dS_prev = decay[t] * dpre
    # update = delta @ K^T  (where delta = V - S_prev @ K)
    K_t = K[t]; V_t = V[t]
    retrieved = S_prev @ K_t
    delta = V_t - retrieved
    # dupdate = dpre
    ddelta = dpre @ K_t
    dK_from_update = dpre.T @ delta
    # delta = V - retrieved
    dV[t] = ddelta
    dretrieved = -ddelta
    # retrieved = S_prev @ K
    dS_prev += dretrieved @ K_t.T
    dK_from_retrieved = S_prev.T @ dretrieved
    dK[t] = dK_from_update + dK_from_retrieved
    dS = dS_prev
  dS0 = dS
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _e91_bwd_kernel(
    # Saved tensors:
    S0_ptr, S_traj_ptr,
    K_ptr, V_ptr, Q_ptr, decay_ptr,
    # Upstream grads:
    d_Sq_ptr, d_ST_ptr,
    # Output grads:
    dK_ptr, dV_ptr, dQ_ptr, ddec_ptr, dS0_ptr,
    # Strides for [B, T, H, N, R] tensors (K, V, dK, dV):
    kv_b, kv_t, kv_h, kv_n, kv_r,
    # Strides for [B, T, H, N] tensors (Q, d_Sq, dQ):
    q_b, q_t, q_h, q_n,
    # Strides for [B, T, H] (decay, ddec):
    dec_b, dec_t, dec_h,
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr,
    N: tl.constexpr, R: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    b = pid // H
    h = pid % H
    bh = b * H + h

    row_idx = tl.arange(0, N).to(tl.int64)
    col_idx = tl.arange(0, N).to(tl.int64)
    rank_idx = tl.arange(0, R).to(tl.int64)
    s_tile = row_idx[:, None] * N + col_idx[None, :]
    nr_offset_template = row_idx[:, None] * kv_n + rank_idx[None, :] * kv_r

    S_head_stride = N * N
    traj_head_stride = T * N * N

    kv_base = b * kv_b + h * kv_h
    q_base = b * q_b + h * q_h
    dec_base = b * dec_b + h * dec_h

    # Initialize dS = d_S_T
    dS = tl.load(d_ST_ptr + bh * S_head_stride + s_tile).to(tl.float32)

    for t_rev in range(T):
        t = T - 1 - t_rev

        # Load S_t, S_prev
        S_t = tl.load(S_traj_ptr + bh * traj_head_stride + t * N * N + s_tile).to(tl.float32)
        if t > 0:
            S_prev = tl.load(S_traj_ptr + bh * traj_head_stride + (t - 1) * N * N + s_tile).to(tl.float32)
        else:
            S_prev = tl.load(S0_ptr + bh * S_head_stride + s_tile).to(tl.float32)

        # Load K, V, Q at t
        kv_t_off = t * kv_t
        K_t = tl.load(K_ptr + kv_base + kv_t_off + nr_offset_template).to(tl.float32)
        V_t = tl.load(V_ptr + kv_base + kv_t_off + nr_offset_template).to(tl.float32)
        Q_t = tl.load(Q_ptr + q_base + t * q_t + col_idx * q_n).to(tl.float32)

        dec = tl.load(decay_ptr + dec_base + t * dec_t).to(tl.float32)

        # Load d_Sq at t
        d_Sq_t = tl.load(d_Sq_ptr + q_base + t * q_t + row_idx * q_n).to(tl.float32)  # [N]

        # === Add Sq gradient ===
        # Sq_t = S_t @ Q_t
        # dS += outer(d_Sq_t, Q_t)
        dS_with_sq = dS + d_Sq_t[:, None] * Q_t[None, :]
        # dQ_t = S_t.T @ d_Sq_t  [N]
        # dQ[i] = sum_j S_t[j, i] * d_Sq_t[j]
        dQ_t = tl.sum(S_t * d_Sq_t[:, None], axis=0)
        # Store dQ
        tl.store(dQ_ptr + q_base + t * q_t + col_idx * q_n,
                 dQ_t.to(dQ_ptr.dtype.element_ty))

        # === Through tanh: dpre = dS * (1 - S_t**2) ===
        dpre = dS_with_sq * (1.0 - S_t * S_t)

        # === ddec[t] = sum(dpre * S_prev) ===
        ddec_t = tl.sum(dpre * S_prev)
        tl.store(ddec_ptr + dec_base + t * dec_t, ddec_t.to(ddec_ptr.dtype.element_ty))

        # === dS_prev contribution from decay term ===
        dS_prev = dec * dpre

        # === update = delta @ K^T;  delta = V - S_prev @ K ===
        # Recompute retrieved and delta
        if R >= 16:
            retrieved = tl.dot(S_prev, K_t, allow_tf32=False)  # [N, R]
        else:
            retrieved = tl.sum(S_prev[:, :, None] * K_t[None, :, :], axis=1)
        delta = V_t - retrieved

        # ddelta = dpre @ K  [N,N] @ [N,R] = [N,R]
        if R >= 16:
            ddelta = tl.dot(dpre, K_t, allow_tf32=False)
        else:
            ddelta = tl.sum(dpre[:, :, None] * K_t[None, :, :], axis=1)

        # dK_from_update = dpre.T @ delta  [N,N]^T @ [N,R] = [N,R]
        # = dpre^T @ delta where dpre is [N (i), N (j)], delta [N (i), R]
        # dK_from_update[j, r] = sum_i dpre[i, j] * delta[i, r]
        if R >= 16:
            dpre_T = tl.trans(dpre)
            dK_from_update = tl.dot(dpre_T, delta, allow_tf32=False)
        else:
            # Manual: sum_i over i axis
            dK_from_update = tl.sum(dpre[:, :, None] * delta[:, None, :], axis=0)  # [N, R]

        # === delta = V - retrieved:  dV = ddelta, dretrieved = -ddelta ===
        dV_t = ddelta
        # Store dV
        tl.store(dV_ptr + kv_base + kv_t_off + nr_offset_template,
                 dV_t.to(dV_ptr.dtype.element_ty))
        dretrieved = -ddelta

        # === retrieved = S_prev @ K:
        # dS_prev (from retrieved) = dretrieved @ K^T  [N,R] @ [R,N] = [N,N]
        # dK_from_retrieved = S_prev.T @ dretrieved  [N,N]^T @ [N,R] = [N,R]
        if R >= 16:
            K_T = tl.trans(K_t)
            dS_prev_from_retrieved = tl.dot(dretrieved, K_T, allow_tf32=False)
            S_prev_T = tl.trans(S_prev)
            dK_from_retrieved = tl.dot(S_prev_T, dretrieved, allow_tf32=False)
        else:
            dS_prev_from_retrieved = tl.sum(dretrieved[:, None, :] * K_t[None, :, :], axis=2)  # [N, N]
            dK_from_retrieved = tl.sum(S_prev[:, :, None] * dretrieved[:, None, :], axis=0)  # [N, R]

        dS_prev += dS_prev_from_retrieved
        dK_t = dK_from_update + dK_from_retrieved

        # Store dK
        tl.store(dK_ptr + kv_base + kv_t_off + nr_offset_template,
                 dK_t.to(dK_ptr.dtype.element_ty))

        dS = dS_prev

    # After loop, dS = dS0
    tl.store(dS0_ptr + bh * S_head_stride + s_tile,
             dS.to(dS0_ptr.dtype.element_ty))


def e91_seq_bwd(S0, S_traj, K, V, Q, decay, d_Sq, d_S_T, num_warps=2, num_stages=1):
    """Backward pass for E91 rank-r delta rule.

    Returns: dS0, dK, dV, dQ, ddec
    """
    B, T, H, N, R = K.shape
    assert V.shape == K.shape
    assert Q.shape == (B, T, H, N)
    assert d_Sq.shape == (B, T, H, N)
    assert d_S_T.shape == (B, H, N, N)
    assert S_traj.shape == (B, H, T, N, N)
    assert S0.shape == (B, H, N, N)
    assert decay.shape == (B, T, H)

    dtype = S0.dtype
    device = S0.device

    dK = torch.empty_like(K)
    dV = torch.empty_like(V)
    dQ = torch.empty_like(Q)
    ddec = torch.empty_like(decay)
    dS0 = torch.empty_like(S0)

    kv_b, kv_t, kv_h, kv_n, kv_r = K.stride()
    q_b, q_t, q_h, q_n = Q.stride()
    dec_b, dec_t, dec_h = decay.stride()

    grid = (B * H,)
    _e91_bwd_kernel[grid](
        S0.contiguous(), S_traj.contiguous(),
        K, V, Q, decay,
        d_Sq, d_S_T.contiguous(),
        dK, dV, dQ, ddec, dS0,
        kv_b, kv_t, kv_h, kv_n, kv_r,
        q_b, q_t, q_h, q_n,
        dec_b, dec_t, dec_h,
        B=B, H=H, T=T, N=N, R=R,
        num_warps=num_warps, num_stages=num_stages,
    )
    return dS0, dK, dV, dQ, ddec


# ============================================================================
# Self-test: verify backward via torch.autograd
# ============================================================================
if __name__ == '__main__':
    import os
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '7')

    import sys
    THIS = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, THIS)
    from e91_seq_fwd import e91_seq_fwd, e91_seq_fwd_torch_ref

    print("E91 backward correctness: Triton bwd vs PyTorch autograd through ref\n")
    for B, T, H, N, R in [(1, 16, 4, 16, 16), (2, 32, 2, 16, 8), (1, 8, 4, 16, 1)]:
        dt = torch.float32
        torch.manual_seed(0)
        S0 = (0.1 * torch.randn(B, H, N, N, dtype=dt, device='cuda')).requires_grad_(True)
        K = (0.3 * torch.randn(B, T, H, N, R, dtype=dt, device='cuda')).requires_grad_(True)
        V = (0.3 * torch.randn(B, T, H, N, R, dtype=dt, device='cuda')).requires_grad_(True)
        Q = (0.3 * torch.randn(B, T, H, N, dtype=dt, device='cuda')).requires_grad_(True)
        decay = torch.sigmoid(0.5 + 0.1 * torch.randn(B, T, H, dtype=dt, device='cuda')).requires_grad_(True)

        # PyTorch autograd: forward + backward
        S_traj_ref, Sq_ref = e91_seq_fwd_torch_ref(S0, K, V, Q, decay)
        # Construct loss = sum(Sq) + sum(S_T) so all gradients are nonzero
        d_Sq = torch.randn_like(Sq_ref)
        d_S_T = torch.randn_like(S_traj_ref[:, :, -1])
        loss_ref = (Sq_ref * d_Sq).sum() + (S_traj_ref[:, :, -1] * d_S_T).sum()
        dS0_ref, dK_ref, dV_ref, dQ_ref, ddec_ref = torch.autograd.grad(
            loss_ref, [S0, K, V, Q, decay], retain_graph=False
        )

        # Triton backward (forward must be done first to get S_traj)
        with torch.no_grad():
            S_traj_tri, Sq_tri = e91_seq_fwd(S0.detach(), K.detach(), V.detach(), Q.detach(), decay.detach())
            dS0_tri, dK_tri, dV_tri, dQ_tri, ddec_tri = e91_seq_bwd(
                S0.detach(), S_traj_tri,
                K.detach(), V.detach(), Q.detach(), decay.detach(),
                d_Sq, d_S_T,
            )

        def rel(a, b): return (a - b).abs().max().item() / max(b.abs().max().item(), 1e-10)
        rs0 = rel(dS0_tri, dS0_ref)
        rk = rel(dK_tri, dK_ref)
        rv = rel(dV_tri, dV_ref)
        rq = rel(dQ_tri, dQ_ref)
        rd = rel(ddec_tri, ddec_ref)
        worst = max(rs0, rk, rv, rq, rd)
        ok = "PASS" if worst < 1e-3 else "FAIL"
        print(f"  B={B} T={T} H={H} N={N} R={R}: dS0={rs0:.1e} dK={rk:.1e} dV={rv:.1e} dQ={rq:.1e} ddec={rd:.1e}  [{ok}]")
