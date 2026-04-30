"""E94 time-recurrence backward Triton kernel.

Mirror of e94_time_fwd. Backward through:
    S^t = tanh( W_h · S^{t-1} + write^t )

Backward (sequential from t=T-1 to t=0):
    Through tanh: d_pre = dS_full * (1 - S^t * S^t)
    Through W_h · S^{t-1} + write:
        d_write = d_pre   (passed back to the source: K, V or precomputed Write)
        d_W_h += d_pre @ S^{t-1}^T  (atomic-accumulated across t for this head)
        dS^{t-1} (from this term) = W_h^T @ d_pre

For USE_DELTA path (write = k ⊗ (v - retrieved), retrieved = S^T · k):
    d_v = sum over rows of d_write * k = d_write^T @ k (along N)
    d_delta = sum over rows of d_write * k = same (per col)
    d_k_from_outer = sum over cols of d_write * delta
    d_v_from_delta = d_delta
    d_retrieved = -d_delta
    d_S_prev_from_retr[n, c] = -d_delta[c] * k[n]   (= -d_delta per col, weighted by k row)
    d_k_from_retr = sum_c S_prev[:, c] * (-d_delta[c]) (negated; = -S_prev @ d_delta_T summed)

For pre-computed Write path (l>0):
    d_Write[t] = d_pre  (just the upstream gradient)
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _e94_time_bwd_kernel(
    S0_ptr,            # [B, H, N, HD]
    Wh_time_ptr,       # [H, N, N]
    K_ptr, V_ptr,      # [B, T, H, N], [B, T, H, HD] (USE_DELTA only)
    Write_ptr,         # [B, T, H, N, HD] (use_delta=False only)
    S_traj_ptr,        # [B, T, H, N, HD]
    d_S_traj_ptr,      # [B, T, H, N, HD] - upstream grad
    d_S_T_ptr,         # [B, H, N, HD] - grad w.r.t. final state
    # outputs
    dS0_ptr,           # [B, H, N, HD]
    dWh_time_ptr,      # [H, N, N]
    dK_ptr, dV_ptr,    # USE_DELTA only
    dWrite_ptr,        # use_delta=False only
    # strides
    s0_b, s0_h, s0_n, s0_c,
    wh_h, wh_n1, wh_n2,
    k_b, k_t, k_h, k_n,
    v_b, v_t, v_h, v_c,
    w_b, w_t, w_h_, w_n, w_c,
    st_b, st_t, st_h, st_n, st_c,
    B: tl.constexpr, T: tl.constexpr, H: tl.constexpr,
    N: tl.constexpr, HD: tl.constexpr,
    USE_DELTA: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    b = pid // H
    h = pid % H

    n_idx = tl.arange(0, N).to(tl.int64)
    c_idx = tl.arange(0, HD).to(tl.int64)

    wh_tile = h * wh_h + n_idx[:, None] * wh_n1 + n_idx[None, :] * wh_n2
    W_h = tl.load(Wh_time_ptr + wh_tile).to(tl.float32)
    W_h_T = tl.trans(W_h)

    # Local accumulator for dW_h (this head, this batch — atomic_add at end)
    dWh_local = tl.zeros([N, N], dtype=tl.float32)

    # Initialize dS = d_S_T
    dst_off = b * s0_b + h * s0_h + n_idx[:, None] * s0_n + c_idx[None, :] * s0_c
    dS = tl.load(d_S_T_ptr + dst_off).to(tl.float32)

    for t_rev in range(T):
        t = T - 1 - t_rev

        # Load S^t (from forward)
        st_off = b * st_b + t * st_t + h * st_h + n_idx[:, None] * st_n + c_idx[None, :] * st_c
        S_t = tl.load(S_traj_ptr + st_off).to(tl.float32)

        # Load S^{t-1}: from S_traj if t>0 else from S0
        if t > 0:
            stp_off = b * st_b + (t - 1) * st_t + h * st_h + n_idx[:, None] * st_n + c_idx[None, :] * st_c
            S_prev = tl.load(S_traj_ptr + stp_off).to(tl.float32)
        else:
            S_prev = tl.load(S0_ptr + dst_off).to(tl.float32)

        # Add upstream grad d_S_traj[t]
        d_st_off = b * st_b + t * st_t + h * st_h + n_idx[:, None] * st_n + c_idx[None, :] * st_c
        d_S_traj_t = tl.load(d_S_traj_ptr + d_st_off).to(tl.float32)
        dS_full = dS + d_S_traj_t

        # Through tanh
        d_pre = dS_full * (1.0 - S_t * S_t)

        # Accumulate dW_h: d_pre @ S_prev^T  (small, [N, N])
        S_prev_T = tl.trans(S_prev)
        dWh_step = tl.dot(d_pre, S_prev_T, allow_tf32=False)
        dWh_local += dWh_step

        # dS^{t-1} from W_h · S_prev term: W_h^T @ d_pre
        dS_prev = tl.dot(W_h_T, d_pre, allow_tf32=False)

        # Through write
        if USE_DELTA:
            # write = k ⊗ delta where delta = v - retrieved, retrieved = S_prev^T · k
            k_off = b * k_b + t * k_t + h * k_h + n_idx * k_n
            k_vec = tl.load(K_ptr + k_off).to(tl.float32)
            v_off = b * v_b + t * v_t + h * v_h + c_idx * v_c
            v_vec = tl.load(V_ptr + v_off).to(tl.float32)

            # Recompute retrieved + delta
            retrieved = tl.sum(S_prev * k_vec[:, None], axis=0)
            delta = v_vec - retrieved

            # d_pre is gradient w.r.t. write (= k ⊗ delta)
            # Through outer(k, delta):
            #   d_delta[c] = sum_n d_pre[n, c] * k[n]
            #   d_k_from_outer[n] = sum_c d_pre[n, c] * delta[c]
            d_delta = tl.sum(d_pre * k_vec[:, None], axis=0)         # [HD]
            d_k_from_outer = tl.sum(d_pre * delta[None, :], axis=1)  # [N]

            # dV = d_delta
            tl.store(dV_ptr + v_off, d_delta.to(dV_ptr.dtype.element_ty))

            # Through retrieved = S_prev^T · k:
            #   d_retrieved = -d_delta
            #   dS_prev_from_retr[n, c] = -d_delta[c] * k[n]
            #   d_k_from_retr[n] = sum_c (-d_delta[c]) * S_prev[n, c]
            d_retrieved = -d_delta
            dS_prev_from_retr = k_vec[:, None] * d_retrieved[None, :]
            dS_prev += dS_prev_from_retr

            d_k_from_retr = tl.sum(S_prev * d_retrieved[None, :], axis=1)
            dk_total = d_k_from_outer + d_k_from_retr
            tl.store(dK_ptr + k_off, dk_total.to(dK_ptr.dtype.element_ty))
        else:
            # write was passed in as Write[t]; gradient wrt Write is just d_pre
            w_off = b * w_b + t * w_t + h * w_h_ + n_idx[:, None] * w_n + c_idx[None, :] * w_c
            tl.store(dWrite_ptr + w_off, d_pre.to(dWrite_ptr.dtype.element_ty))

        dS = dS_prev

    # Atomic_add local dW_h to global dW_h
    dwh_off = h * wh_h + n_idx[:, None] * wh_n1 + n_idx[None, :] * wh_n2
    tl.atomic_add(dWh_time_ptr + dwh_off, dWh_local)

    # Store dS0
    tl.store(dS0_ptr + dst_off, dS.to(dS0_ptr.dtype.element_ty))


def e94_time_bwd(S0, S_traj, W_h_time, d_S_traj, d_S_T,
                  K=None, V=None, Write=None, use_delta=True, num_warps=2):
    """Backward for e94_time_fwd.

    Returns:
        dS0: [B, H, N, HD]
        dW_h_time: [H, N, N]
        dK: [B, T, H, N]  (if use_delta else None)
        dV: [B, T, H, HD] (if use_delta else None)
        dWrite: [B, T, H, N, HD] (if not use_delta else None)
    """
    B, H, N, HD = S0.shape
    T = S_traj.shape[1]

    dS0 = torch.empty_like(S0)
    dW_h = torch.zeros_like(W_h_time, dtype=torch.float32)

    if use_delta:
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)
        dWrite = K  # placeholder
    else:
        dK = None
        dV = None
        dWrite = torch.empty_like(Write)

    s0_b, s0_h, s0_n, s0_c = S0.stride()
    wh_h_, wh_n1, wh_n2 = W_h_time.stride()
    if use_delta:
        k_b, k_t, k_h, k_n = K.stride()
        v_b, v_t, v_h, v_c = V.stride()
        w_b = w_t = w_h__ = w_n = w_c = 0
    else:
        k_b = k_t = k_h = k_n = 0
        v_b = v_t = v_h = v_c = 0
        w_b, w_t, w_h__, w_n, w_c = Write.stride()
    st_b, st_t, st_h, st_n, st_c = S_traj.stride()

    K_arg = K if use_delta else Write
    V_arg = V if use_delta else Write
    Write_arg = Write if not use_delta else (K if K is not None else S0)
    dK_arg = dK if use_delta else (Write if Write is not None else S0)
    dV_arg = dV if use_delta else Write
    dWrite_arg = dWrite if not use_delta else (K if K is not None else S0)

    grid = (B * H,)
    _e94_time_bwd_kernel[grid](
        S0.contiguous(), W_h_time.contiguous(),
        K_arg, V_arg, Write_arg,
        S_traj.contiguous(), d_S_traj.contiguous(), d_S_T.contiguous(),
        dS0, dW_h, dK_arg, dV_arg, dWrite_arg,
        s0_b, s0_h, s0_n, s0_c,
        wh_h_, wh_n1, wh_n2,
        k_b, k_t, k_h, k_n,
        v_b, v_t, v_h, v_c,
        w_b, w_t, w_h__, w_n, w_c,
        st_b, st_t, st_h, st_n, st_c,
        B=B, T=T, H=H, N=N, HD=HD,
        USE_DELTA=use_delta,
        num_warps=num_warps,
    )
    if use_delta:
        return dS0, dW_h.to(W_h_time.dtype), dK, dV, None
    else:
        return dS0, dW_h.to(W_h_time.dtype), None, None, dWrite


if __name__ == '__main__':
    import os, sys
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '6')
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from e94_time_fwd import e94_time_fwd, e94_time_fwd_torch_ref

    print("E94 time-recurrence backward parity tests\n")

    for B, T, H, N, HD in [(2, 32, 16, 16, 16), (1, 64, 8, 16, 16)]:
        torch.manual_seed(0)
        dt = torch.float32

        # Build inputs with grad tracking
        # Note: K is treated as already-normalized leaf (kernel doesn't normalize internally).
        S0 = (0.05 * torch.randn(B, H, N, HD, dtype=dt, device='cuda')).contiguous().requires_grad_(True)
        W_h = ((torch.eye(N, device='cuda').unsqueeze(0).expand(H, -1, -1).contiguous() +
               0.05 * torch.randn(H, N, N, dtype=dt, device='cuda'))).contiguous().requires_grad_(True)
        K_raw = 0.3 * torch.randn(B, T, H, N, dtype=dt, device='cuda')
        K = torch.nn.functional.normalize(K_raw, dim=-1).detach().requires_grad_(True).contiguous()
        V = (0.3 * torch.randn(B, T, H, HD, dtype=dt, device='cuda')).requires_grad_(True)

        # Reference forward + autograd backward
        S_traj_ref = e94_time_fwd_torch_ref(S0, W_h, K=K, V=V, use_delta=True)
        d_S_traj = torch.randn_like(S_traj_ref)
        d_S_T = torch.randn_like(S_traj_ref[:, -1])
        loss_ref = (S_traj_ref * d_S_traj).sum() + (S_traj_ref[:, -1] * d_S_T).sum()
        grads_ref = torch.autograd.grad(loss_ref, [S0, W_h, K, V])
        dS0_ref, dWh_ref, dK_ref, dV_ref = grads_ref

        # Triton fwd+bwd
        with torch.no_grad():
            S_traj = e94_time_fwd(S0.detach(), W_h.detach(), K=K.detach(), V=V.detach(), use_delta=True)
            dS0_t, dWh_t, dK_t, dV_t, _ = e94_time_bwd(
                S0.detach(), S_traj, W_h.detach(),
                d_S_traj, d_S_T,
                K=K.detach(), V=V.detach(), use_delta=True,
            )

        def rel(a, b):
            denom = max(b.abs().max().item(), 1e-10)
            return (a - b).abs().max().item() / denom

        rs = rel(dS0_t, dS0_ref)
        rwh = rel(dWh_t, dWh_ref)
        rk = rel(dK_t, dK_ref)
        rv = rel(dV_t, dV_ref)
        worst = max(rs, rwh, rk, rv)
        ok = "PASS" if worst < 1e-3 else "FAIL"
        print(f"  delta B={B} T={T} H={H}: dS0={rs:.1e} dWh={rwh:.1e} dK={rk:.1e} dV={rv:.1e}  [{ok}]")
