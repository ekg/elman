"""Rectangular-state Triton sequential forward for Pararnn-convention E88,
with fused per-step output (Sq) emission.

Supports CUDA_S with shape [N_STATE, HEAD_V_DIM] where these dims differ.
Under the Pararnn_S = CUDA_S^T convention, Pararnn_S has shape [M, N]
where M = HEAD_V_DIM (= Pararnn p dim = CUDA j) and N = N_STATE
(= Pararnn q dim = CUDA i).

Pararnn update (rectangular):
  retrieved[p] = sum_q S[p,q] * k[q]         # k has N values
  delta[p] = v[p] - retrieved[p]              # v has M values
  S_new[p,q] = tanh(dec*S[p,q] + delta[p]*k[q])
  output[p] = sum_q S_new[p,q] * q_vec[q]    # q_vec has N values

Output is indexed by M = HEAD_V_DIM (matches CUDA output layout).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _pararnn_seq_fwd_output_kernel(
    S0_ptr,        # [B, H, M, N]
    K_ptr,         # [B, H, T, N]
    V_ptr,         # [B, H, T, M]
    Q_ptr,         # [B, H, T, N]
    decay_ptr,     # [B, H, T]
    S_traj_ptr,    # [B, H, T+1, M, N]  output
    Sq_ptr,        # [B, H, T, M]       output (per-step outputs)
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr,
    M: tl.constexpr, N: tl.constexpr,
    M_P2: tl.constexpr, N_P2: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    b = pid // H
    h = pid % H
    bh = b * H + h

    # Use power-of-2 aranges and mask to valid [M, N].
    row_idx = tl.arange(0, M_P2).to(tl.int64)
    col_idx = tl.arange(0, N_P2).to(tl.int64)
    row_mask = row_idx < M
    col_mask = col_idx < N
    mask_2d = row_mask[:, None] & col_mask[None, :]
    # Use clamped indices for safe stores within the logical [M, N] region.
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
    tl.store(S_traj_ptr + traj_base + tile_2d,
             S.to(S_traj_ptr.dtype.element_ty), mask=mask_2d)

    for t in range(T):
        K_t = tl.load(K_ptr + bh * K_head_stride + t * N + c_safe,
                      mask=col_mask, other=0.0).to(tl.float32)
        V_t = tl.load(V_ptr + bh * V_head_stride + t * M + r_safe,
                      mask=row_mask, other=0.0).to(tl.float32)
        Q_t = tl.load(Q_ptr + bh * K_head_stride + t * N + c_safe,
                      mask=col_mask, other=0.0).to(tl.float32)
        dec = tl.load(decay_ptr + bh * dec_head_stride + t).to(tl.float32)

        # retrieved[p] = sum_q S[p,q] * K_t[q]   (masked: out-of-range terms are 0)
        retrieved = tl.sum(S * K_t[None, :], axis=1)
        delta_row = V_t - retrieved
        pre = dec * S + delta_row[:, None] * K_t[None, :]
        e2x = tl.exp(2.0 * pre)
        S = (e2x - 1.0) / (e2x + 1.0)
        # Zero out-of-range entries to avoid polluting next iter's sums.
        S = tl.where(mask_2d, S, 0.0)

        tl.store(S_traj_ptr + traj_base + (t + 1) * M * N + tile_2d,
                 S.to(S_traj_ptr.dtype.element_ty), mask=mask_2d)

        Sq_t = tl.sum(S * Q_t[None, :], axis=1)
        tl.store(Sq_ptr + bh * V_head_stride + t * M + r_safe,
                 Sq_t.to(Sq_ptr.dtype.element_ty), mask=row_mask)


def _next_pow2(x):
    p = 1
    while p < x:
        p <<= 1
    return p


def pararnn_seq_fwd_output_triton(S0, K, V, Q, decay, num_warps=1):
    """
    S0:    [B, H, M, N]   Pararnn state (= CUDA_S^T)
    K:     [B, H, T, N]
    V:     [B, H, T, M]
    Q:     [B, H, T, N]
    decay: [B, H, T]
    Returns:
      S_traj: [B, H, T+1, M, N]
      Sq:     [B, H, T, M]       per-step outputs
    """
    B, H = K.shape[0], K.shape[1]
    T = K.shape[2]
    M = S0.shape[-2]
    N = S0.shape[-1]
    assert K.shape[-1] == N and V.shape[-1] == M and Q.shape[-1] == N
    M_P2 = _next_pow2(M)
    N_P2 = _next_pow2(N)
    dtype = S0.dtype
    S_traj = torch.empty(B, H, T + 1, M, N, dtype=dtype, device=S0.device)
    Sq = torch.empty(B, H, T, M, dtype=dtype, device=S0.device)
    grid = (B * H,)
    _pararnn_seq_fwd_output_kernel[grid](
        S0.contiguous(), K.contiguous(), V.contiguous(), Q.contiguous(),
        decay.contiguous(),
        S_traj, Sq,
        B=B, H=H, T=T, M=M, N=N, M_P2=M_P2, N_P2=N_P2,
        num_warps=num_warps, num_stages=1,
    )
    return S_traj, Sq


def cuda_step_py(S_prev, k, v, decay):
    """CUDA-convention step, Python reference for rectangular states.
    S_prev: [B, H, Ns, Hv], k: [B, H, Ns], v: [B, H, Hv], decay: [B, H]
    """
    retrieved = torch.einsum('bhij,bhi->bhj', S_prev, k)  # [B, H, Hv]
    delta = v - retrieved
    outer = torch.einsum('bhj,bhi->bhij', delta, k)       # [B, H, Ns, Hv]
    return torch.tanh(decay[..., None, None] * S_prev + outer)


def cuda_forward_py(S0, K, V, Q, decay):
    T = K.shape[0]
    S = S0
    outputs = []
    for t in range(T):
        S = cuda_step_py(S, K[t], V[t], decay[t])
        outputs.append(torch.einsum('bhij,bhi->bhj', S, Q[t]))
    return S, torch.stack(outputs, 0)


if __name__ == '__main__':
    import time
    import sys, os
    sys.path.insert(0, '/home/erikg/elman')
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    for B, H, T, Ns, Hv in [
        (1, 4, 1024, 16, 16),
        (1, 4, 1024, 32, 32),
        (1, 4, 512, 32, 24),    # rectangular (CUDA kernel has no such template)
        (1, 4, 512, 32, 23),    # E88-n32 480M ratio
        (1, 4, 512, 16, 14),    # E88-n16 480M ratio  (dim=1536/141≈10.9; try 14)
    ]:
        dt = torch.float32  # fp32 — precise compare vs Python autograd ref
        torch.manual_seed(0)
        k = (0.3 * torch.randn(T, B, H, Ns, dtype=dt, device='cuda'))
        v = (0.3 * torch.randn(T, B, H, Hv, dtype=dt, device='cuda'))
        q = (0.3 * torch.randn(T, B, H, Ns, dtype=dt, device='cuda'))
        decay = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, dtype=dt, device='cuda'))
        S0 = 0.1 * torch.randn(B, H, Ns, Hv, dtype=dt, device='cuda')

        # Python CUDA-convention reference
        S_final_ref, output_ref = cuda_forward_py(S0, k, v, q, decay)

        # Triton Pararnn with transpose
        S0_p = S0.transpose(-1, -2).contiguous()     # [B, H, M=Hv, N=Ns]
        K_p = k.permute(1, 2, 0, 3).contiguous()
        V_p = v.permute(1, 2, 0, 3).contiguous()
        Q_p = q.permute(1, 2, 0, 3).contiguous()
        decay_p = decay.permute(1, 2, 0).contiguous()
        S_traj_p, Sq_p = pararnn_seq_fwd_output_triton(
            S0_p, K_p, V_p, Q_p, decay_p,
            num_warps=4 if Ns * Hv >= 32 * 32 else 1,
        )
        S_final_t = S_traj_p[:, :, -1].transpose(-1, -2).contiguous()
        output_t = Sq_p.permute(2, 0, 1, 3).contiguous()

        err_s = (S_final_ref - S_final_t).abs().max().item()
        err_o = (output_ref - output_t).abs().max().item()
        rel_s = err_s / max(S_final_ref.abs().max().item(), 1e-10)
        rel_o = err_o / max(output_ref.abs().max().item(), 1e-10)
        ok = "PASS" if rel_s < 1e-4 and rel_o < 1e-4 else "FAIL"
        shape_str = "square" if Ns == Hv else f"rect {Ns}x{Hv}"
        print(f"  B={B} H={H} T={T:4d} Ns={Ns} Hv={Hv} ({shape_str}):  "
              f"S_final rel={rel_s:.2e}  output rel={rel_o:.2e}  [{ok}]")
