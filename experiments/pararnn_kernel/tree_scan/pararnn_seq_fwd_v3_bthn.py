"""V3 forward — [B, T, H, N]-native stride-parametrized Triton kernel.

Eliminates the permute([1,2,0,3]).contiguous() that v2 does internally.
Accepts K, V, Q, decay in production layout [B, T, H, N] / [B, T, H],
emits Sq directly in [B, T, H, N] AND S_traj in [B, H, T, N, N].

The S_traj layout stays [B, H, T, N, N] — that's optimal for backward
because consecutive t values are N*N apart per head (cache-friendly for
sequential backward scan).

Convention: we store S_1..S_T contiguously at indices 0..T-1 (no S_0 slot).
S0 stays as a separate tensor.

Strides are passed as runtime args, so the kernel can accept any input layout.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _seq_fwd_v3_bthn_kernel(
    # Inputs:
    S0_ptr,      # [B, H, N, N] (CUDA: [B, H, Ns=N, Hv=M]; Pararnn: [B, H, M, N])
    K_ptr,       # [B, T, H, N] layout (stride params give it)
    V_ptr,       # [B, T, H, N]
    Q_ptr,       # [B, T, H, N]
    decay_ptr,   # [B, T, H]
    # Outputs:
    S_traj_ptr,  # [B, H, T, N, N] contig
    Sq_ptr,      # [B, T, H, N] contig
    # Runtime strides (K/V/Q share layout):
    k_b_stride, k_t_stride, k_h_stride, k_n_stride,
    dec_b_stride, dec_t_stride, dec_h_stride,
    sq_b_stride, sq_t_stride, sq_h_stride, sq_n_stride,
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

    # Base pointers for this (b, h):
    k_base = b * k_b_stride + h * k_h_stride
    dec_base = b * dec_b_stride + h * dec_h_stride
    sq_base = b * sq_b_stride + h * sq_h_stride

    S = tl.load(S0_ptr + bh * S_head_stride + tile_2d).to(tl.float32)
    for t in range(T):
        # Load per-step: K, V, Q are [N] vectors at (b, t, h, :).
        k_off = k_base + t * k_t_stride + col_idx * k_n_stride
        v_off = k_base + t * k_t_stride + row_idx * k_n_stride
        q_off = k_base + t * k_t_stride + col_idx * k_n_stride
        d_off = dec_base + t * dec_t_stride

        K_t = tl.load(K_ptr + k_off).to(tl.float32)
        V_t = tl.load(V_ptr + v_off).to(tl.float32)
        Q_t = tl.load(Q_ptr + q_off).to(tl.float32)
        dec = tl.load(decay_ptr + d_off).to(tl.float32)

        retrieved = tl.sum(S * K_t[None, :], axis=1)
        delta_row = V_t - retrieved
        pre = dec * S + delta_row[:, None] * K_t[None, :]
        e2x = tl.exp(2.0 * pre)
        S = (e2x - 1.0) / (e2x + 1.0)

        # Store S_{t+1} at index t of contiguous S_traj.
        tl.store(
            S_traj_ptr + bh * traj_head_stride + t * N * N + tile_2d,
            S.to(S_traj_ptr.dtype.element_ty),
        )

        # Compute Sq[t] = S @ Q[t]; write to [B, T, H, N] at (b, t, h, :).
        # Sq[row] = sum_col S[row, col] * Q[col]
        Sq_t = tl.sum(S * Q_t[None, :], axis=1)
        sq_off = sq_base + t * sq_t_stride + row_idx * sq_n_stride
        tl.store(Sq_ptr + sq_off, Sq_t.to(Sq_ptr.dtype.element_ty))


def pararnn_seq_fwd_v3_bthn(S0, K, V, Q, decay, num_warps=1, num_stages=1):
    """[B, T, H, N]-native forward.

    Args:
      S0:    [B, H, N, N]    (Pararnn convention, transposed from CUDA S0)
      K:     [B, T, H, N]
      V:     [B, T, H, N]
      Q:     [B, T, H, N]
      decay: [B, T, H]

    Returns:
      S_traj: [B, H, T, N, N] contig — stores S_1..S_T
      Sq:     [B, T, H, N]    contig — per-step output
    """
    B, T, H, N = K.shape
    assert V.shape == (B, T, H, N) and Q.shape == (B, T, H, N)
    assert decay.shape == (B, T, H)
    assert S0.shape == (B, H, N, N)
    dtype = S0.dtype
    device = S0.device

    S_traj = torch.empty(B, H, T, N, N, dtype=dtype, device=device)
    Sq = torch.empty(B, T, H, N, dtype=dtype, device=device)

    # Extract strides (so kernel is layout-agnostic)
    k_bs, k_ts, k_hs, k_ns = K.stride()
    dec_bs, dec_ts, dec_hs = decay.stride()
    sq_bs, sq_ts, sq_hs, sq_ns = Sq.stride()

    grid = (B * H,)
    _seq_fwd_v3_bthn_kernel[grid](
        S0.contiguous(), K, V, Q, decay,
        S_traj, Sq,
        k_bs, k_ts, k_hs, k_ns,
        dec_bs, dec_ts, dec_hs,
        sq_bs, sq_ts, sq_hs, sq_ns,
        B=B, H=H, T=T, N=N,
        num_warps=num_warps, num_stages=num_stages,
    )
    return S_traj, Sq


if __name__ == '__main__':
    import os
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '3')
    import sys
    THIS = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, THIS)
    from pararnn_seq_fwd_v2 import pararnn_seq_fwd_v2

    print("Correctness v3_bthn vs v2 (same inputs, both layouts):\n")
    for B, H, T, N in [(1, 4, 512, 16), (1, 4, 1024, 32), (2, 8, 256, 16), (16, 4, 512, 32)]:
        dt = torch.float32
        torch.manual_seed(0)
        # Generate in [B, T, H, N] layout
        K_bt = 0.3 * torch.randn(B, T, H, N, dtype=dt, device='cuda')
        V_bt = 0.3 * torch.randn(B, T, H, N, dtype=dt, device='cuda')
        Q_bt = 0.3 * torch.randn(B, T, H, N, dtype=dt, device='cuda')
        decay_bt = torch.sigmoid(0.5 + 0.1 * torch.randn(B, T, H, dtype=dt, device='cuda'))
        S0 = 0.1 * torch.randn(B, H, N, N, dtype=dt, device='cuda')

        # v3 path: direct [B, T, H, N]
        S_traj_v3, Sq_v3 = pararnn_seq_fwd_v3_bthn(S0, K_bt, V_bt, Q_bt, decay_bt,
                                                     num_warps=1 if N == 16 else 4)

        # v2 reference: permute inputs to [B, H, T, N]
        K_p = K_bt.permute(0, 2, 1, 3).contiguous()
        V_p = V_bt.permute(0, 2, 1, 3).contiguous()
        Q_p = Q_bt.permute(0, 2, 1, 3).contiguous()
        decay_p = decay_bt.permute(0, 2, 1).contiguous()
        S_traj_v2 = pararnn_seq_fwd_v2(S0, K_p, V_p, decay_p,
                                         num_warps=1 if N == 16 else 4)
        Sq_v2 = torch.einsum('bhtpq,bhtq->bhtp', S_traj_v2, Q_p)
        # permute back to [B, T, H, N]
        Sq_v2_bt = Sq_v2.permute(0, 2, 1, 3).contiguous()

        def rel(a, b): return (a - b).abs().max().item() / max(b.abs().max().item(), 1e-10)
        r_traj = rel(S_traj_v3, S_traj_v2)
        r_sq = rel(Sq_v3, Sq_v2_bt)
        w = max(r_traj, r_sq)
        ok = "PASS" if w < 1e-4 else "FAIL"
        print(f"  B={B} H={H} T={T} N={N}:  S_traj={r_traj:.1e}  Sq={r_sq:.1e}  [{ok}]")
