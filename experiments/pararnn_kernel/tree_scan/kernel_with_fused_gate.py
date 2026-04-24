"""Modified Triton kernel that applies silu(g) gating within the forward kernel.

Saves one elementwise kernel (F.silu + multiply) per layer.
Expected savings: 0.9 ms/call * 50 calls = 45 ms/step = 1.5% speedup.
"""

import os, sys
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, THIS)
sys.path.insert(0, os.path.dirname(THIS))


@triton.jit
def _pararnn_seq_fwd_gated_kernel(
    S0_ptr, K_ptr, V_ptr, Q_ptr, G_ptr, decay_ptr,
    S_traj_ptr, Output_ptr,   # Output = Sq * silu(g)
    Sq_raw_ptr,               # Sq (ungated) for backward
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

    S_head_stride = M * N
    traj_head_stride = (T + 1) * M * N
    K_head_stride = T * N
    V_head_stride = T * M
    dec_head_stride = T

    S0_base = bh * S_head_stride
    S = tl.load(S0_ptr + S0_base + tile_2d, mask=mask_2d, other=0.0).to(tl.float32)
    traj_base = bh * traj_head_stride
    tl.store(S_traj_ptr + traj_base + tile_2d, S.to(S_traj_ptr.dtype.element_ty), mask=mask_2d)

    for t in range(T):
        K_t = tl.load(K_ptr + bh * K_head_stride + t * N + c_safe, mask=col_mask, other=0.0).to(tl.float32)
        V_t = tl.load(V_ptr + bh * V_head_stride + t * M + r_safe, mask=row_mask, other=0.0).to(tl.float32)
        Q_t = tl.load(Q_ptr + bh * K_head_stride + t * N + c_safe, mask=col_mask, other=0.0).to(tl.float32)
        G_t = tl.load(G_ptr + bh * V_head_stride + t * M + r_safe, mask=row_mask, other=0.0).to(tl.float32)
        dec = tl.load(decay_ptr + bh * dec_head_stride + t).to(tl.float32)

        retrieved = tl.sum(S * K_t[None, :], axis=1)
        delta_row = V_t - retrieved
        pre = dec * S + delta_row[:, None] * K_t[None, :]
        e2x = tl.exp(2.0 * pre)
        S = (e2x - 1.0) / (e2x + 1.0)
        S = tl.where(mask_2d, S, 0.0)

        traj_t_base = traj_base + (t + 1) * M * N
        tl.store(S_traj_ptr + traj_t_base + tile_2d, S.to(S_traj_ptr.dtype.element_ty), mask=mask_2d)

        # Sq = S @ Q_t  -> shape [M]
        Sq = tl.sum(S * Q_t[None, :], axis=1)
        # Store raw Sq for backward
        Sq_base_t = bh * T * M + t * M
        tl.store(Sq_raw_ptr + Sq_base_t + r_safe, Sq.to(Sq_raw_ptr.dtype.element_ty), mask=row_mask)
        # Apply silu(G_t) gate: out = Sq * G_t * sigmoid(G_t)
        sig_g = 1.0 / (1.0 + tl.exp(-G_t))
        silu_g = G_t * sig_g
        gated = Sq * silu_g
        tl.store(Output_ptr + Sq_base_t + r_safe, gated.to(Output_ptr.dtype.element_ty), mask=row_mask)


def pararnn_seq_fwd_gated_triton(S0, K, V, Q, G, decay, num_warps=1):
    """S0:[B,H,M,N], K:[B,H,T,N], V:[B,H,T,M], Q:[B,H,T,N], G:[B,H,T,M], decay:[B,H,T]
    Returns: S_traj [B,H,T+1,M,N], output [B,H,T,M] gated, Sq_raw [B,H,T,M] ungated."""
    B, H, M, N = S0.shape
    T = K.shape[2]
    S_traj = torch.empty(B, H, T + 1, M, N, dtype=S0.dtype, device=S0.device)
    output = torch.empty(B, H, T, M, dtype=S0.dtype, device=S0.device)
    Sq_raw = torch.empty(B, H, T, M, dtype=S0.dtype, device=S0.device)
    grid = (B * H,)
    def _next_pow2(x):
        p = 1
        while p < x:
            p <<= 1
        return p
    M_P2 = _next_pow2(M)
    N_P2 = _next_pow2(N)
    _pararnn_seq_fwd_gated_kernel[grid](
        S0, K, V, Q, G, decay, S_traj, output, Sq_raw,
        B, H, T, M, N, M_P2, N_P2,
        num_warps=num_warps, num_stages=1,
    )
    return S_traj, output, Sq_raw


if __name__ == '__main__':
    # Correctness vs stock hybrid + external gate
    from pararnn_seq_fwd_rect import pararnn_seq_fwd_output_triton

    torch.manual_seed(0)
    B, H, T, N = 1, 8, 512, 16
    M = N
    dt = torch.bfloat16
    K = torch.randn(B, H, T, N, dtype=dt, device='cuda')
    V = torch.randn(B, H, T, M, dtype=dt, device='cuda')
    Q = torch.randn(B, H, T, N, dtype=dt, device='cuda')
    G = torch.randn(B, H, T, M, dtype=dt, device='cuda')
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(B, H, T, dtype=dt, device='cuda'))
    S0 = 0.1 * torch.randn(B, H, M, N, dtype=dt, device='cuda')

    S_traj_ref, Sq_ref = pararnn_seq_fwd_output_triton(S0, K, V, Q, decay)
    out_ref = Sq_ref * F.silu(G)

    S_traj_new, out_new, Sq_raw_new = pararnn_seq_fwd_gated_triton(S0, K, V, Q, G, decay)

    def rel(a, b):
        return (a.float() - b.float()).abs().max().item() / max(b.float().abs().max().item(), 1e-10)

    print(f"S_traj rel={rel(S_traj_new, S_traj_ref):.2e}")
    print(f"Sq_raw rel={rel(Sq_raw_new, Sq_ref):.2e}")
    print(f"output rel={rel(out_new, out_ref):.2e}")

    # Timing: at production shape B=1 T=32K H=141 N=16
    import time
    print("\n=== Timing kernel comparison at production shape ===")
    B, H, T, N = 1, 141, 32768, 16
    M = N
    K = torch.randn(B, H, T, N, dtype=dt, device='cuda')
    V = torch.randn(B, H, T, M, dtype=dt, device='cuda')
    Q = torch.randn(B, H, T, N, dtype=dt, device='cuda')
    G = torch.randn(B, H, T, M, dtype=dt, device='cuda')
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(B, H, T, dtype=dt, device='cuda'))
    S0 = 0.1 * torch.randn(B, H, M, N, dtype=dt, device='cuda')

    def old():
        S_traj_ref, Sq_ref = pararnn_seq_fwd_output_triton(S0, K, V, Q, decay, num_warps=1)
        out_ref = Sq_ref * F.silu(G)
        return out_ref

    def new():
        S_traj_new, out_new, Sq_raw_new = pararnn_seq_fwd_gated_triton(S0, K, V, Q, G, decay, num_warps=1)
        return out_new

    for _ in range(3): old()
    torch.cuda.synchronize()
    evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(20)]
    for s, e in evs:
        s.record(); old(); e.record()
    torch.cuda.synchronize()
    t_old = sorted([s.elapsed_time(e) for s, e in evs])[10]

    for _ in range(3): new()
    torch.cuda.synchronize()
    evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(20)]
    for s, e in evs:
        s.record(); new(); e.record()
    torch.cuda.synchronize()
    t_new = sorted([s.elapsed_time(e) for s, e in evs])[10]
    print(f"  old (fwd kernel + silu*out): {t_old:>7.3f} ms")
    print(f"  new (fused gate kernel)    : {t_new:>7.3f} ms")
    print(f"  savings: {t_old - t_new:.3f} ms  ({(t_old-t_new)/t_old*100:.1f}%)")
