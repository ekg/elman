"""E92 backward Triton kernel — gradients including dW_h.

Forward at step t:
  retrieved = S_prev @ k_t
  delta = v_t - retrieved
  Wh_S = W_h @ S_prev
  pre = alpha * Wh_S + delta · k_t^T (rank-1 outer)
  S_t = tanh(pre)
  Sq_t = S_t @ q_t

Backward at step t:
  dS_t = (running) + outer(d_Sq[t], Q[t])
  dQ[t] = S_t^T @ d_Sq[t]
  dpre = dS_t * (1 - S_t^2)
  ddec[t] = sum(dpre * Wh_S)
  d_Wh_S = alpha * dpre
  dW_h[h] += d_Wh_S @ S_prev^T   (accumulate via atomic_add across batches)
  dS_prev (from Wh_S) = W_h^T @ d_Wh_S
  ddelta = dpre @ k_t
  dk_outer = dpre^T @ delta   (rank-1 outer product backward)
  dV[t] = ddelta
  dretrieved = -ddelta
  dS_prev_retr = dretrieved · k_t^T  (rank-1 outer)
  dS_prev += dS_prev_retr
  dk_retr = S_prev^T @ dretrieved
  dK[t] = dk_outer + dk_retr
  dS = dS_prev
After loop: dS0 = dS
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _e92_bwd_kernel(
    S0_ptr, S_traj_ptr, K_ptr, V_ptr, Q_ptr, decay_ptr, W_h_ptr,
    d_Sq_ptr, d_ST_ptr,
    dK_ptr, dV_ptr, dQ_ptr, ddec_ptr, dS0_ptr, dW_h_ptr,
    k_b, k_t, k_h, k_n,
    q_b, q_t, q_h, q_n,
    dec_b, dec_t, dec_h,
    wh_h, wh_n1, wh_n2,
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, N: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    b = pid // H
    h = pid % H
    bh = b * H + h

    row_idx = tl.arange(0, N).to(tl.int64)
    col_idx = tl.arange(0, N).to(tl.int64)
    s_tile = row_idx[:, None] * N + col_idx[None, :]
    wh_tile = row_idx[:, None] * wh_n1 + col_idx[None, :] * wh_n2

    S_head_stride = N * N
    traj_head_stride = T * N * N

    k_base = b * k_b + h * k_h
    q_base = b * q_b + h * q_h
    dec_base = b * dec_b + h * dec_h

    W_h = tl.load(W_h_ptr + h * wh_h + wh_tile).to(tl.float32)
    W_h_T = tl.trans(W_h)

    # Local accumulator for dW_h (per program, per head)
    dWh_local = tl.zeros([N, N], dtype=tl.float32)

    # Initialize dS = d_S_T
    dS = tl.load(d_ST_ptr + bh * S_head_stride + s_tile).to(tl.float32)

    for t_rev in range(T):
        t = T - 1 - t_rev

        S_t = tl.load(S_traj_ptr + bh * traj_head_stride + t * N * N + s_tile).to(tl.float32)
        if t > 0:
            S_prev = tl.load(S_traj_ptr + bh * traj_head_stride + (t - 1) * N * N + s_tile).to(tl.float32)
        else:
            S_prev = tl.load(S0_ptr + bh * S_head_stride + s_tile).to(tl.float32)

        k_t_v = tl.load(K_ptr + k_base + t * k_t + col_idx * k_n).to(tl.float32)
        v_t_v = tl.load(V_ptr + k_base + t * k_t + row_idx * k_n).to(tl.float32)
        q_t_v = tl.load(Q_ptr + q_base + t * q_t + col_idx * q_n).to(tl.float32)
        dec = tl.load(decay_ptr + dec_base + t * dec_t).to(tl.float32)

        d_Sq_t = tl.load(d_Sq_ptr + q_base + t * q_t + row_idx * q_n).to(tl.float32)

        # === Sq gradient ===
        dS_with_sq = dS + d_Sq_t[:, None] * q_t_v[None, :]
        dQ_t = tl.sum(S_t * d_Sq_t[:, None], axis=0)
        tl.store(dQ_ptr + q_base + t * q_t + col_idx * q_n,
                 dQ_t.to(dQ_ptr.dtype.element_ty))

        # === Through tanh ===
        dpre = dS_with_sq * (1.0 - S_t * S_t)

        # Recompute Wh_S for ddec
        Wh_S = tl.dot(W_h, S_prev, allow_tf32=False)

        # ddec[t] = sum(dpre * Wh_S)
        ddec_t = tl.sum(dpre * Wh_S)
        tl.store(ddec_ptr + dec_base + t * dec_t, ddec_t.to(ddec_ptr.dtype.element_ty))

        # === Through "alpha * Wh_S" ===
        d_Wh_S = dec * dpre  # [N, N]
        # dW_h += d_Wh_S @ S_prev^T   (accumulate locally; atomic add to global at end)
        S_prev_T = tl.trans(S_prev)
        dWh_step = tl.dot(d_Wh_S, S_prev_T, allow_tf32=False)
        dWh_local += dWh_step
        # dS_prev (from Wh_S) = W_h^T @ d_Wh_S
        dS_prev = tl.dot(W_h_T, d_Wh_S, allow_tf32=False)

        # === Through rank-1 outer "delta · k^T" ===
        # ddelta = dpre @ k_t  [N]
        ddelta = tl.sum(dpre * k_t_v[None, :], axis=1)
        # dk_outer = dpre^T @ delta  [N]
        # Recompute delta:
        retrieved = tl.sum(S_prev * k_t_v[None, :], axis=1)
        delta = v_t_v - retrieved
        dk_outer = tl.sum(dpre * delta[:, None], axis=0)

        # dV[t] = ddelta
        tl.store(dV_ptr + k_base + t * k_t + row_idx * k_n,
                 ddelta.to(dV_ptr.dtype.element_ty))

        # === retrieved = S_prev @ k:  dS_prev += dretrieved · k^T (rank-1 outer);
        #     dk_retr = S_prev^T @ dretrieved
        dretrieved = -ddelta
        dS_prev += dretrieved[:, None] * k_t_v[None, :]
        # dk_retr[r] = sum_i S_prev[i, r] * dretrieved[i] = sum over rows of (S_prev * dret[:, None])
        dk_retr = tl.sum(S_prev * dretrieved[:, None], axis=0)

        dK_t = dk_outer + dk_retr
        tl.store(dK_ptr + k_base + t * k_t + col_idx * k_n,
                 dK_t.to(dK_ptr.dtype.element_ty))

        dS = dS_prev

    # Atomic add local dW_h to global dW_h[h]
    # dW_h is [H, N, N]; offset for head h is h * N * N (assume contiguous)
    tl.atomic_add(dW_h_ptr + h * wh_h + wh_tile, dWh_local)

    tl.store(dS0_ptr + bh * S_head_stride + s_tile,
             dS.to(dS0_ptr.dtype.element_ty))


def e92_seq_bwd(S0, S_traj, W_h, K, V, Q, decay, d_Sq, d_S_T, num_warps=2, num_stages=1):
    B, T, H, N = K.shape
    dtype = S0.dtype
    device = S0.device

    dK = torch.empty_like(K)
    dV = torch.empty_like(V)
    dQ = torch.empty_like(Q)
    ddec = torch.empty_like(decay)
    dS0 = torch.empty_like(S0)
    # dW_h must be ZERO-INITIALIZED (atomic add from many (b,h) programs).
    # Use float32 for accumulation accuracy.
    dW_h = torch.zeros_like(W_h, dtype=torch.float32)

    k_bs, k_ts, k_hs, k_ns = K.stride()
    q_bs, q_ts, q_hs, q_ns = Q.stride()
    dec_bs, dec_ts, dec_hs = decay.stride()
    wh_hs, wh_n1s, wh_n2s = W_h.stride()

    grid = (B * H,)
    _e92_bwd_kernel[grid](
        S0.contiguous(), S_traj.contiguous(),
        K, V, Q, decay, W_h.contiguous(),
        d_Sq, d_S_T.contiguous(),
        dK, dV, dQ, ddec, dS0, dW_h,
        k_bs, k_ts, k_hs, k_ns,
        q_bs, q_ts, q_hs, q_ns,
        dec_bs, dec_ts, dec_hs,
        wh_hs, wh_n1s, wh_n2s,
        B=B, H=H, T=T, N=N,
        num_warps=num_warps, num_stages=num_stages,
    )
    return dS0, dW_h.to(W_h.dtype), dK, dV, dQ, ddec


if __name__ == '__main__':
    import os, sys
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '5')
    THIS = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, THIS)
    from e92_seq_fwd import e92_seq_fwd, e92_seq_fwd_torch_ref

    print("E92 backward correctness: Triton bwd vs torch.autograd through ref\n")
    for B, T, H, N in [(1, 16, 4, 16), (2, 32, 2, 16)]:
        dt = torch.float32
        torch.manual_seed(0)
        S0 = (0.1 * torch.randn(B, H, N, N, dtype=dt, device='cuda')).requires_grad_(True)
        W_h = ((torch.eye(N).unsqueeze(0).repeat(H, 1, 1).to('cuda') + 0.05 * torch.randn(H, N, N, device='cuda')).contiguous()).requires_grad_(True)
        K = (0.3 * torch.randn(B, T, H, N, dtype=dt, device='cuda')).requires_grad_(True)
        V = (0.3 * torch.randn(B, T, H, N, dtype=dt, device='cuda')).requires_grad_(True)
        Q = (0.3 * torch.randn(B, T, H, N, dtype=dt, device='cuda')).requires_grad_(True)
        decay = torch.sigmoid(0.5 + 0.1 * torch.randn(B, T, H, dtype=dt, device='cuda')).requires_grad_(True)

        # Reference
        S_traj_ref, Sq_ref = e92_seq_fwd_torch_ref(S0, W_h, K, V, Q, decay)
        d_Sq = torch.randn_like(Sq_ref)
        d_S_T = torch.randn_like(S_traj_ref[:, :, -1])
        loss_ref = (Sq_ref * d_Sq).sum() + (S_traj_ref[:, :, -1] * d_S_T).sum()
        dS0_ref, dWh_ref, dK_ref, dV_ref, dQ_ref, ddec_ref = torch.autograd.grad(
            loss_ref, [S0, W_h, K, V, Q, decay], retain_graph=False
        )

        # Triton
        with torch.no_grad():
            S_traj, Sq = e92_seq_fwd(S0.detach(), W_h.detach(), K.detach(), V.detach(), Q.detach(), decay.detach())
            dS0_t, dWh_t, dK_t, dV_t, dQ_t, ddec_t = e92_seq_bwd(
                S0.detach(), S_traj, W_h.detach(),
                K.detach(), V.detach(), Q.detach(), decay.detach(),
                d_Sq, d_S_T,
            )

        def rel(a, b): return (a - b).abs().max().item() / max(b.abs().max().item(), 1e-10)
        rs0 = rel(dS0_t, dS0_ref)
        rwh = rel(dWh_t, dWh_ref)
        rk = rel(dK_t, dK_ref)
        rv = rel(dV_t, dV_ref)
        rq = rel(dQ_t, dQ_ref)
        rd = rel(ddec_t, ddec_ref)
        worst = max(rs0, rwh, rk, rv, rq, rd)
        ok = "PASS" if worst < 1e-3 else "FAIL"
        print(f"  B={B} T={T} H={H} N={N}: dS0={rs0:.1e} dWh={rwh:.1e} dK={rk:.1e} dV={rv:.1e} dQ={rq:.1e} ddec={rd:.1e}  [{ok}]")
