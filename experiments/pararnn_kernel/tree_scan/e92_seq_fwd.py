"""E92 forward Triton kernel — matrix-matrix nonlinear RNN with learned W_h.

Per (b, h, t):
  retrieved = S @ k_t              [N]
  delta = v_t - retrieved          [N]
  Wh_S = W_h @ S                   [N, N]   ← TC-friendly matmul
  S = tanh(alpha · Wh_S + delta · k^T)
  Sq_t = S @ q_t                   [N]

Layout:
  K, V, Q: [B, T, H, N]    (rank-1 vectors)
  W_h:     [H, N, N]       (learned per-layer)
  decay:   [B, T, H]
  S0:      [B, H, N, N]
  S_traj:  [B, H, T, N, N]
  Sq:      [B, T, H, N]
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _e92_fwd_kernel(
    S0_ptr, K_ptr, V_ptr, Q_ptr, decay_ptr, W_h_ptr,
    S_traj_ptr, Sq_ptr,
    # K, V, Q strides: [B, T, H, N]
    k_b, k_t, k_h, k_n,
    q_b, q_t, q_h, q_n,
    dec_b, dec_t, dec_h,
    sq_b, sq_t, sq_h, sq_n,
    # W_h strides: [H, N, N]
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
    sq_base = b * sq_b + h * sq_h

    # Load W_h for this head once (stays in registers across the T loop)
    W_h = tl.load(W_h_ptr + h * wh_h + wh_tile).to(tl.float32)

    # Load S0
    S = tl.load(S0_ptr + bh * S_head_stride + s_tile).to(tl.float32)

    for t in range(T):
        k_t_v = tl.load(K_ptr + k_base + t * k_t + col_idx * k_n).to(tl.float32)
        v_t_v = tl.load(V_ptr + k_base + t * k_t + row_idx * k_n).to(tl.float32)
        q_t_v = tl.load(Q_ptr + q_base + t * q_t + col_idx * q_n).to(tl.float32)
        dec = tl.load(decay_ptr + dec_base + t * dec_t).to(tl.float32)

        # Retrieve: S @ k_t
        retrieved = tl.sum(S * k_t_v[None, :], axis=1)
        delta = v_t_v - retrieved

        # W_h @ S — TC-friendly matrix-matrix
        Wh_S = tl.dot(W_h, S, allow_tf32=False)

        # pre = alpha * Wh_S + delta · k^T (rank-1 outer product)
        outer_dk = delta[:, None] * k_t_v[None, :]
        pre = dec * Wh_S + outer_dk
        e2x = tl.exp(2.0 * pre)
        S = (e2x - 1.0) / (e2x + 1.0)

        tl.store(S_traj_ptr + bh * traj_head_stride + t * N * N + s_tile,
                 S.to(S_traj_ptr.dtype.element_ty))

        Sq_t = tl.sum(S * q_t_v[None, :], axis=1)
        tl.store(Sq_ptr + sq_base + t * sq_t + row_idx * sq_n,
                 Sq_t.to(Sq_ptr.dtype.element_ty))


def e92_seq_fwd(S0, W_h, K, V, Q, decay, num_warps=2, num_stages=1):
    B, T, H, N = K.shape
    assert V.shape == K.shape and Q.shape == K.shape
    assert decay.shape == (B, T, H)
    assert S0.shape == (B, H, N, N)
    assert W_h.shape == (H, N, N)

    dtype = S0.dtype
    device = S0.device
    S_traj = torch.empty(B, H, T, N, N, dtype=dtype, device=device)
    Sq = torch.empty(B, T, H, N, dtype=dtype, device=device)

    k_bs, k_ts, k_hs, k_ns = K.stride()
    q_bs, q_ts, q_hs, q_ns = Q.stride()
    dec_bs, dec_ts, dec_hs = decay.stride()
    sq_bs, sq_ts, sq_hs, sq_ns = Sq.stride()
    wh_hs, wh_n1s, wh_n2s = W_h.stride()

    grid = (B * H,)
    _e92_fwd_kernel[grid](
        S0.contiguous(), K, V, Q, decay, W_h,
        S_traj, Sq,
        k_bs, k_ts, k_hs, k_ns,
        q_bs, q_ts, q_hs, q_ns,
        dec_bs, dec_ts, dec_hs,
        sq_bs, sq_ts, sq_hs, sq_ns,
        wh_hs, wh_n1s, wh_n2s,
        B=B, H=H, T=T, N=N,
        num_warps=num_warps, num_stages=num_stages,
    )
    return S_traj, Sq


def e92_seq_fwd_torch_ref(S0, W_h, K, V, Q, decay):
    """Pure PyTorch reference."""
    B, T, H, N = K.shape
    S = S0.float().clone()
    K = K.float(); V = V.float(); Q = Q.float()
    decay = decay.float(); W_h = W_h.float()
    S_traj = torch.empty(B, H, T, N, N, dtype=S0.dtype, device=S0.device)
    Sq = torch.empty(B, T, H, N, dtype=S0.dtype, device=S0.device)
    Wh_b = W_h.unsqueeze(0)

    for t in range(T):
        k_t = K[:, t]; v_t = V[:, t]; q_t = Q[:, t]
        dec = decay[:, t].view(B, H, 1, 1)
        retrieved = torch.matmul(S, k_t.unsqueeze(-1)).squeeze(-1)
        delta = v_t - retrieved
        Wh_S = torch.matmul(Wh_b, S)
        outer = delta.unsqueeze(-1) * k_t.unsqueeze(-2)
        S = torch.tanh(dec * Wh_S + outer)
        S_traj[:, :, t] = S.to(S0.dtype)
        Sq[:, t] = torch.matmul(S, q_t.unsqueeze(-1)).squeeze(-1).to(S0.dtype)
    return S_traj, Sq


if __name__ == '__main__':
    import os
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '5')
    print("E92 forward correctness: Triton vs PyTorch reference (fp32)\n")
    for B, T, H, N in [(1, 32, 4, 16), (2, 64, 8, 16), (4, 32, 2, 16)]:
        dt = torch.float32
        torch.manual_seed(0)
        S0 = 0.1 * torch.randn(B, H, N, N, dtype=dt, device='cuda')
        W_h = torch.eye(N).unsqueeze(0).repeat(H, 1, 1).to('cuda').contiguous()
        W_h = (W_h + 0.05 * torch.randn(H, N, N, device='cuda')).contiguous()
        K = 0.3 * torch.randn(B, T, H, N, dtype=dt, device='cuda')
        V = 0.3 * torch.randn(B, T, H, N, dtype=dt, device='cuda')
        Q = 0.3 * torch.randn(B, T, H, N, dtype=dt, device='cuda')
        decay = torch.sigmoid(0.5 + 0.1 * torch.randn(B, T, H, dtype=dt, device='cuda'))

        S_tri, Sq_tri = e92_seq_fwd(S0, W_h, K, V, Q, decay)
        S_ref, Sq_ref = e92_seq_fwd_torch_ref(S0, W_h, K, V, Q, decay)

        rt = (S_tri - S_ref).abs().max().item() / max(S_ref.abs().max().item(), 1e-10)
        rs = (Sq_tri - Sq_ref).abs().max().item() / max(Sq_ref.abs().max().item(), 1e-10)
        ok = "PASS" if max(rt, rs) < 1e-4 else "FAIL"
        print(f"  B={B} T={T} H={H} N={N}:  S={rt:.2e}  Sq={rs:.2e}  [{ok}]")
