"""E91 forward Triton kernel — rank-r matrix-matrix nonlinear RNN.

Per (b, h, t):
  retrieved = S @ K_t            [N, R]
  delta = V_t - retrieved        [N, R]
  update = delta @ K_t.T         [N, N]
  S = tanh(α_t · S + update)
  Sq_t = S @ Q_t                 [N]

Layout:
  K, V: [B, T, H, N, R]  — rank-r per timestep
  Q:    [B, T, H, N]     — rank-1 query (output is single vector per step)
  decay:[B, T, H]
  S0:   [B, H, N, N]
  S_traj: [B, H, T, N, N]
  Sq:   [B, T, H, N]
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _e91_fwd_kernel(
    S0_ptr, K_ptr, V_ptr, Q_ptr, decay_ptr,
    S_traj_ptr, Sq_ptr,
    # K, V strides: [B, T, H, N, R] → 5 strides each
    k_b, k_t, k_h, k_n, k_r,
    # Q strides: [B, T, H, N] → 4 strides
    q_b, q_t, q_h, q_n,
    dec_b, dec_t, dec_h,
    sq_b, sq_t, sq_h, sq_n,
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr,
    N: tl.constexpr, R: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    b = pid // H
    h = pid % H
    bh = b * H + h

    row_idx = tl.arange(0, N).to(tl.int64)        # [N]
    col_idx = tl.arange(0, N).to(tl.int64)        # [N]
    rank_idx = tl.arange(0, R).to(tl.int64)        # [R]

    # 2-D tile indices for state [N, N]
    s_tile = row_idx[:, None] * N + col_idx[None, :]
    # 2-D tile for [N, R] (K, V) — stride pattern: [n_stride, r_stride]
    nr_offset = row_idx[:, None] * k_n + rank_idx[None, :] * k_r

    S_head_stride = N * N
    traj_head_stride = T * N * N

    k_base = b * k_b + h * k_h
    q_base = b * q_b + h * q_h
    dec_base = b * dec_b + h * dec_h
    sq_base = b * sq_b + h * sq_h

    # Load initial state S0
    S = tl.load(S0_ptr + bh * S_head_stride + s_tile).to(tl.float32)

    for t in range(T):
        # === Load K_t [N, R], V_t [N, R] ===
        kv_t_off = t * k_t
        K_t = tl.load(K_ptr + k_base + kv_t_off + nr_offset).to(tl.float32)  # [N, R]
        V_t = tl.load(V_ptr + k_base + kv_t_off + nr_offset).to(tl.float32)  # [N, R]
        # Q is [N] (rank-1)
        Q_t = tl.load(Q_ptr + q_base + t * q_t + col_idx * q_n).to(tl.float32)  # [N]

        dec = tl.load(decay_ptr + dec_base + t * dec_t).to(tl.float32)

        # === Retrieved = S @ K_t  →  [N, R] ===
        # S [N, N] @ K_t [N, R] = retrieved [N, R]
        # Use 3D broadcast for sum; works correctly when N >= 16 (Triton's quirk).
        if R >= 16:
            retrieved = tl.dot(S, K_t, allow_tf32=False)  # TC: 16×16×16 mma
        else:
            retrieved = tl.sum(S[:, :, None] * K_t[None, :, :], axis=1)  # [N, R]

        # === Delta = V - retrieved ===
        delta = V_t - retrieved   # [N, R]

        # === Update = delta @ K^T  →  [N, N] ===
        if R >= 16:
            K_t_T = tl.trans(K_t)  # [R, N]
            update = tl.dot(delta, K_t_T, allow_tf32=False)  # [N, R] @ [R, N] = [N, N]
        else:
            # Manually accumulate rank-r outer products (broadcast trick fails at N=R=16)
            update = tl.zeros([N, N], dtype=tl.float32)
            for r in tl.static_range(R):
                # Extract column r of delta and K via masking
                mask = (tl.arange(0, R) == r)
                delta_r = tl.sum(delta * mask[None, :].to(tl.float32), axis=1)  # [N]
                K_r = tl.sum(K_t * mask[None, :].to(tl.float32), axis=1)        # [N]
                update += delta_r[:, None] * K_r[None, :]

        pre = dec * S + update
        # tanh(pre) — use direct formula (E88 already showed exp baseline is fastest)
        e2x = tl.exp(2.0 * pre)
        S = (e2x - 1.0) / (e2x + 1.0)

        # Store S_t in trajectory
        tl.store(
            S_traj_ptr + bh * traj_head_stride + t * N * N + s_tile,
            S.to(S_traj_ptr.dtype.element_ty),
        )

        # Compute Sq_t = S @ Q_t   [N]
        Sq_t = tl.sum(S * Q_t[None, :], axis=1)  # [N]
        sq_off = sq_base + t * sq_t + row_idx * sq_n
        tl.store(Sq_ptr + sq_off, Sq_t.to(Sq_ptr.dtype.element_ty))


def e91_seq_fwd(S0, K, V, Q, decay, num_warps=2, num_stages=1):
    """Forward pass for E91 rank-r delta rule.

    Args:
      S0:    [B, H, N, N]
      K:     [B, T, H, N, R]
      V:     [B, T, H, N, R]
      Q:     [B, T, H, N]
      decay: [B, T, H]

    Returns:
      S_traj: [B, H, T, N, N]
      Sq:     [B, T, H, N]
    """
    B, T, H, N, R = K.shape
    assert V.shape == K.shape
    assert Q.shape == (B, T, H, N)
    assert decay.shape == (B, T, H)
    assert S0.shape == (B, H, N, N)

    dtype = S0.dtype
    device = S0.device
    S_traj = torch.empty(B, H, T, N, N, dtype=dtype, device=device)
    Sq = torch.empty(B, T, H, N, dtype=dtype, device=device)

    k_bs, k_ts, k_hs, k_ns, k_rs = K.stride()
    q_bs, q_ts, q_hs, q_ns = Q.stride()
    dec_bs, dec_ts, dec_hs = decay.stride()
    sq_bs, sq_ts, sq_hs, sq_ns = Sq.stride()

    grid = (B * H,)
    _e91_fwd_kernel[grid](
        S0.contiguous(), K, V, Q, decay,
        S_traj, Sq,
        k_bs, k_ts, k_hs, k_ns, k_rs,
        q_bs, q_ts, q_hs, q_ns,
        dec_bs, dec_ts, dec_hs,
        sq_bs, sq_ts, sq_hs, sq_ns,
        B=B, H=H, T=T, N=N, R=R,
        num_warps=num_warps, num_stages=num_stages,
    )
    return S_traj, Sq


# ============================================================================
# Reference: pure PyTorch implementation matching the kernel exactly
# ============================================================================
def e91_seq_fwd_torch_ref(S0, K, V, Q, decay):
    """Pure PyTorch reference matching e91_seq_fwd. Uses fp32 internally."""
    B, T, H, N, R = K.shape
    S = S0.float().clone()  # [B, H, N, N]
    K = K.float()
    V = V.float()
    Q = Q.float()
    decay = decay.float()

    S_traj = torch.empty(B, H, T, N, N, dtype=S0.dtype, device=S0.device)
    Sq = torch.empty(B, T, H, N, dtype=S0.dtype, device=S0.device)

    for t in range(T):
        K_t = K[:, t]                    # [B, H, N, R]
        V_t = V[:, t]                    # [B, H, N, R]
        Q_t = Q[:, t]                    # [B, H, N]
        dec = decay[:, t].view(B, H, 1, 1)  # [B, H, 1, 1]

        retrieved = torch.matmul(S, K_t)               # [B, H, N, R]
        delta = V_t - retrieved                          # [B, H, N, R]
        update = torch.matmul(delta, K_t.transpose(-1, -2))  # [B, H, N, N]
        S = torch.tanh(dec * S + update)

        S_traj[:, :, t] = S.to(S0.dtype)
        Sq[:, t] = torch.matmul(S, Q_t.unsqueeze(-1)).squeeze(-1).to(S0.dtype)

    return S_traj, Sq


# ============================================================================
# Self-test
# ============================================================================
if __name__ == '__main__':
    import os
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '7')

    print("E91 forward correctness: Triton vs PyTorch reference (fp32)\n")
    for B, T, H, N, R in [(1, 32, 4, 16, 16), (2, 64, 8, 16, 8), (1, 128, 4, 16, 1), (4, 32, 2, 16, 16)]:
        dt = torch.float32
        torch.manual_seed(0)
        S0 = 0.1 * torch.randn(B, H, N, N, dtype=dt, device='cuda')
        K = 0.3 * torch.randn(B, T, H, N, R, dtype=dt, device='cuda')
        V = 0.3 * torch.randn(B, T, H, N, R, dtype=dt, device='cuda')
        Q = 0.3 * torch.randn(B, T, H, N, dtype=dt, device='cuda')
        decay = torch.sigmoid(0.5 + 0.1 * torch.randn(B, T, H, dtype=dt, device='cuda'))

        S_tri, Sq_tri = e91_seq_fwd(S0, K, V, Q, decay, num_warps=2 if N == 16 else 4)
        S_ref, Sq_ref = e91_seq_fwd_torch_ref(S0, K, V, Q, decay)

        rel_traj = (S_tri - S_ref).abs().max().item() / max(S_ref.abs().max().item(), 1e-10)
        rel_sq = (Sq_tri - Sq_ref).abs().max().item() / max(Sq_ref.abs().max().item(), 1e-10)
        ok = "PASS" if max(rel_traj, rel_sq) < 1e-4 else "FAIL"
        print(f"  B={B} T={T} H={H} N={N} R={R}:  S_traj={rel_traj:.2e}  Sq={rel_sq:.2e}  [{ok}]")
