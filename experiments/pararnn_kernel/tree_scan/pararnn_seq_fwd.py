"""Triton sequential forward for Pararnn-convention E88.

Replaces the slow Python `sequential_e88_forward` loop with a single
Triton kernel that runs T steps per (B, H) in registers and dumps dense
S_traj to HBM.

Pararnn convention:
  S_new[p,q] = tanh(dec * S[p,q] + (v[p] - sum_r S[p,r]*k[r]) * k[q])
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _pararnn_seq_fwd_kernel(
    S0_ptr,     # [B, H, N, N]
    K_ptr,      # [B, H, T, N]
    V_ptr,      # [B, H, T, N]
    decay_ptr,  # [B, H, T]
    S_traj_ptr, # [B, H, T+1, N, N]  (output)
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
    traj_head_stride = (T + 1) * N * N
    K_head_stride = T * N
    dec_head_stride = T

    # Load S0
    S0_base = bh * S_head_stride
    S = tl.load(S0_ptr + S0_base + tile_2d).to(tl.float32)
    # Write S_traj[0] = S0
    traj_base = bh * traj_head_stride
    tl.store(S_traj_ptr + traj_base + tile_2d, S.to(S_traj_ptr.dtype.element_ty))

    for t in range(T):
        K_t = tl.load(K_ptr + bh * K_head_stride + t * N + col_idx).to(tl.float32)
        V_t = tl.load(V_ptr + bh * K_head_stride + t * N + row_idx).to(tl.float32)
        dec = tl.load(decay_ptr + bh * dec_head_stride + t).to(tl.float32)

        retrieved = tl.sum(S * K_t[None, :], axis=1)  # [N]
        delta_row = V_t - retrieved
        pre = dec * S + delta_row[:, None] * K_t[None, :]
        e2x = tl.exp(2.0 * pre)
        S = (e2x - 1.0) / (e2x + 1.0)

        # Write S_traj[t+1]
        tl.store(S_traj_ptr + traj_base + (t + 1) * N * N + tile_2d,
                 S.to(S_traj_ptr.dtype.element_ty))


def pararnn_seq_fwd_triton(S0, K, V, decay, num_warps=1):
    """
    S0: [B, H, N, N]
    K:  [B, H, T, N]
    V:  [B, H, T, N]
    decay: [B, H, T]
    Returns S_traj: [B, H, T+1, N, N]  (same dtype as S0)
    """
    B, H, T, N = K.shape
    dtype = S0.dtype
    S_traj = torch.empty(B, H, T + 1, N, N, dtype=dtype, device=S0.device)
    grid = (B * H,)
    _pararnn_seq_fwd_kernel[grid](
        S0.contiguous(), K.contiguous(), V.contiguous(), decay.contiguous(),
        S_traj,
        B=B, H=H, T=T, N=N,
        num_warps=num_warps, num_stages=1,
    )
    return S_traj


if __name__ == '__main__':
    import time
    import sys, os
    sys.path.insert(0, '/home/erikg/elman')
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from phase4_newton_driver import sequential_e88_forward

    for B, H, T, N in [(1, 4, 1024, 16), (1, 4, 4096, 16), (1, 4, 4096, 32)]:
        dt = torch.float32
        torch.manual_seed(0)
        S0 = 0.1 * torch.randn(B, H, N, N, dtype=dt, device='cuda')
        K = 0.3 * torch.randn(B, H, T, N, dtype=dt, device='cuda')
        V = 0.3 * torch.randn(B, H, T, N, dtype=dt, device='cuda')
        decay = torch.sigmoid(0.5 + 0.1 * torch.randn(B, H, T, dtype=dt, device='cuda'))

        # Reference
        S_ref = sequential_e88_forward(S0, K, V, decay)

        # Triton
        S_tri = pararnn_seq_fwd_triton(S0, K, V, decay, num_warps=4 if N == 32 else 1)

        err = (S_ref - S_tri).abs().max().item()
        print(f"B={B} H={H} T={T} N={N}  max_err={err:.2e}")

        # Speed
        for _ in range(3):
            _ = pararnn_seq_fwd_triton(S0, K, V, decay, num_warps=4 if N == 32 else 1)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(5):
            _ = pararnn_seq_fwd_triton(S0, K, V, decay, num_warps=4 if N == 32 else 1)
        torch.cuda.synchronize()
        tri_ms = (time.time() - t0) / 5 * 1000

        for _ in range(2):
            _ = sequential_e88_forward(S0, K, V, decay)
        torch.cuda.synchronize()
        t0 = time.time()
        py_ms = 0
        if T <= 4096:
            for _ in range(2):
                _ = sequential_e88_forward(S0, K, V, decay)
            torch.cuda.synchronize()
            py_ms = (time.time() - t0) / 2 * 1000
        print(f"  triton={tri_ms:.2f}ms  python={py_ms:.2f}ms  speedup={py_ms/max(tri_ms,1e-6):.1f}×")
