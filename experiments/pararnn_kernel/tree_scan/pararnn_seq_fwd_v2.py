"""Pararnn sequential forward with CONTIGUOUS output layout.

Stores S_traj as [B, H, T, N, N] where S_traj[t] = S_{t+1} (the state
AFTER step t).  S_0 stays as the separate S0 tensor.

This eliminates the non-contiguous S_traj[:, :, 1:] gap that makes the
Sq/dQ einsums ~2-3× slower than they should be.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _seq_fwd_v2_kernel(
    S0_ptr,      # [B, H, N, N]
    K_ptr,       # [B, H, T, N]
    V_ptr,       # [B, H, T, N]
    decay_ptr,   # [B, H, T]
    S_traj_ptr,  # [B, H, T, N, N]  — contig, stores S_1..S_T
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
    K_head_stride = T * N
    dec_head_stride = T

    S = tl.load(S0_ptr + bh * S_head_stride + tile_2d).to(tl.float32)
    for t in range(T):
        K_t = tl.load(K_ptr + bh * K_head_stride + t * N + col_idx).to(tl.float32)
        V_t = tl.load(V_ptr + bh * K_head_stride + t * N + row_idx).to(tl.float32)
        dec = tl.load(decay_ptr + bh * dec_head_stride + t).to(tl.float32)
        retrieved = tl.sum(S * K_t[None, :], axis=1)
        delta_row = V_t - retrieved
        pre = dec * S + delta_row[:, None] * K_t[None, :]
        e2x = tl.exp(2.0 * pre)
        S = (e2x - 1.0) / (e2x + 1.0)
        # Store S_{t+1} at index t of contiguous S_traj
        tl.store(S_traj_ptr + bh * traj_head_stride + t * N * N + tile_2d,
                 S.to(S_traj_ptr.dtype.element_ty))


def pararnn_seq_fwd_v2(S0, K, V, decay, num_warps=1):
    """Forward that stores S_1..S_T contiguously as [B, H, T, N, N].
    Returns S_traj of that shape (no leading S_0 slot).
    """
    B, H, T, N = K.shape
    dtype = S0.dtype
    S_traj = torch.empty(B, H, T, N, N, dtype=dtype, device=S0.device)
    grid = (B * H,)
    _seq_fwd_v2_kernel[grid](
        S0.contiguous(), K.contiguous(), V.contiguous(), decay.contiguous(),
        S_traj,
        B=B, H=H, T=T, N=N,
        num_warps=num_warps, num_stages=1,
    )
    return S_traj


# Variant backward that takes S0 separately and S_traj [B, H, T, N, N]
@triton.jit
def _bwd_v2_kernel(
    S0_ptr,      # [B, H, N, N]      — S_0 (needed for last step t=0)
    S_traj_ptr,  # [B, H, T, N, N]   — S_1..S_T (contig, no S_0)
    K_ptr, V_ptr, decay_ptr,
    g_T_ptr, dL_dout_ptr, q_ptr,
    g_out_ptr, dK_ptr, dV_ptr, ddecay_ptr, dQ_ptr,
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, N: tl.constexpr,
):
    """Reverse scan; for each step we need S_prev = S_{t-1} (under 1-indexed)
    or equivalently the state BEFORE the forward step t.  In our convention
    S_traj[i] = S_{i+1}, so:
      - for iterations t = T-1 down to 1, S_prev comes from S_traj[t-1]
      - for iteration t = 0, S_prev comes from S0
    To avoid an in-loop branch, we split: the first (T-1) reverse iterations
    read from S_traj contiguously, then one final iteration handles t=0 from
    S0.
    """
    pid = tl.program_id(0).to(tl.int64)
    b = pid // H
    h = pid % H
    bh = b * H + h
    row_idx = tl.arange(0, N).to(tl.int64)
    col_idx = tl.arange(0, N).to(tl.int64)
    tile_2d = row_idx[:, None] * N + col_idx[None, :]
    S_head_stride = N * N
    traj_head_stride = T * N * N
    K_head_stride = T * N
    dec_head_stride = T

    g = tl.load(g_T_ptr + bh * S_head_stride + tile_2d).to(tl.float32)

    # Main loop: t = T-1, T-2, ..., 1  (T-1 iterations)
    # Equivalent: t = T - 1 - t_inv  for t_inv in 0..T-2
    for t_inv in range(T - 1):
        t = T - 1 - t_inv

        dL_dout_t = tl.load(dL_dout_ptr + bh * K_head_stride + t * N + row_idx).to(tl.float32)
        q_t = tl.load(q_ptr + bh * K_head_stride + t * N + col_idx).to(tl.float32)
        g = g + dL_dout_t[:, None] * q_t[None, :]

        K_t = tl.load(K_ptr + bh * K_head_stride + t * N + col_idx).to(tl.float32)
        V_t = tl.load(V_ptr + bh * K_head_stride + t * N + row_idx).to(tl.float32)
        dec = tl.load(decay_ptr + bh * dec_head_stride + t).to(tl.float32)

        # S_prev = S_t = S_traj[t-1]  (since S_traj[i] = S_{i+1})
        S_prev = tl.load(
            S_traj_ptr + bh * traj_head_stride + (t - 1) * N * N + tile_2d
        ).to(tl.float32)

        retrieved = tl.sum(S_prev * K_t[None, :], axis=1)
        delta_row = V_t - retrieved
        pre = dec * S_prev + delta_row[:, None] * K_t[None, :]
        e2x = tl.exp(2.0 * pre)
        tanh_val = (e2x - 1.0) / (e2x + 1.0)
        tanh_deriv = 1.0 - tanh_val * tanh_val

        dQ_t = tl.sum(dL_dout_t[:, None] * tanh_val, axis=0)
        tl.store(dQ_ptr + bh * K_head_stride + t * N + col_idx, dQ_t)

        u_mat = tanh_deriv * K_t[None, :]
        gu_row = tl.sum(g * u_mat, axis=1)
        tl.store(dV_ptr + bh * K_head_stride + t * N + row_idx, gu_row)

        g_times_tanhd = g * tanh_deriv
        dK_contrib = delta_row[:, None] * g_times_tanhd - S_prev * gu_row[:, None]
        dK_t = tl.sum(dK_contrib, axis=0)
        tl.store(dK_ptr + bh * K_head_stride + t * N + col_idx, dK_t)

        ddec_t = tl.sum(g_times_tanhd * S_prev)
        tl.store(ddecay_ptr + bh * dec_head_stride + t, ddec_t)

        D_mat = dec * tanh_deriv
        g = D_mat * g - K_t[None, :] * gu_row[:, None]

    # Final step: t = 0, with S_prev = S_0 (loaded from S0 tensor, branch-free)
    dL_dout_0 = tl.load(dL_dout_ptr + bh * K_head_stride + 0 * N + row_idx).to(tl.float32)
    q_0 = tl.load(q_ptr + bh * K_head_stride + 0 * N + col_idx).to(tl.float32)
    g = g + dL_dout_0[:, None] * q_0[None, :]

    K_0 = tl.load(K_ptr + bh * K_head_stride + 0 * N + col_idx).to(tl.float32)
    V_0 = tl.load(V_ptr + bh * K_head_stride + 0 * N + row_idx).to(tl.float32)
    dec_0 = tl.load(decay_ptr + bh * dec_head_stride + 0).to(tl.float32)

    S_prev_0 = tl.load(S0_ptr + bh * S_head_stride + tile_2d).to(tl.float32)

    retrieved = tl.sum(S_prev_0 * K_0[None, :], axis=1)
    delta_row = V_0 - retrieved
    pre = dec_0 * S_prev_0 + delta_row[:, None] * K_0[None, :]
    e2x = tl.exp(2.0 * pre)
    tanh_val = (e2x - 1.0) / (e2x + 1.0)
    tanh_deriv = 1.0 - tanh_val * tanh_val

    dQ_0 = tl.sum(dL_dout_0[:, None] * tanh_val, axis=0)
    tl.store(dQ_ptr + bh * K_head_stride + 0 * N + col_idx, dQ_0)

    u_mat = tanh_deriv * K_0[None, :]
    gu_row = tl.sum(g * u_mat, axis=1)
    tl.store(dV_ptr + bh * K_head_stride + 0 * N + row_idx, gu_row)

    g_times_tanhd = g * tanh_deriv
    dK_contrib = delta_row[:, None] * g_times_tanhd - S_prev_0 * gu_row[:, None]
    dK_0 = tl.sum(dK_contrib, axis=0)
    tl.store(dK_ptr + bh * K_head_stride + 0 * N + col_idx, dK_0)

    ddec_0 = tl.sum(g_times_tanhd * S_prev_0)
    tl.store(ddecay_ptr + bh * dec_head_stride + 0, ddec_0)

    D_mat = dec_0 * tanh_deriv
    g = D_mat * g - K_0[None, :] * gu_row[:, None]

    tl.store(g_out_ptr + bh * S_head_stride + tile_2d, g.to(g_out_ptr.dtype.element_ty))


def backward_v2(S0, S_traj, K, V, decay, g_T, dL_dout, q, num_warps=1, num_stages=1):
    """Backward with contiguous [B, H, T, N, N] S_traj (no S_0 slot) +
    separate S0 tensor.  Fuses dQ inline.  Returns (dS0, dK, dV, dQ, ddec)."""
    B, H, T, N = K.shape
    device = S_traj.device
    dtype = S_traj.dtype
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)
    dQ = torch.zeros_like(q)
    ddec = torch.zeros_like(decay)
    dS0 = torch.zeros(B, H, N, N, dtype=dtype, device=device)
    grid = (B * H,)
    _bwd_v2_kernel[grid](
        S0.contiguous(), S_traj.contiguous(),
        K.contiguous(), V.contiguous(), decay.contiguous(),
        g_T.contiguous(), dL_dout.contiguous(), q.contiguous(),
        dS0, dK, dV, ddec, dQ,
        B=B, H=H, T=T, N=N,
        num_warps=num_warps, num_stages=num_stages,
    )
    return dS0, dK, dV, dQ, ddec


if __name__ == '__main__':
    import sys, os, time
    sys.path.insert(0, '/home/erikg/elman')
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from phase7_fused_backward import backward_e88_fused_rank1
    from pararnn_seq_fwd import pararnn_seq_fwd_triton

    # Correctness: compare v2 to baseline (v1 + separate einsum)
    print("Correctness:")
    for H, T, N in [(4, 512, 16), (4, 1024, 32)]:
        dt = torch.float32
        torch.manual_seed(0)
        K = 0.3 * torch.randn(1, H, T, N, dtype=dt, device='cuda')
        V = 0.3 * torch.randn(1, H, T, N, dtype=dt, device='cuda')
        q = 0.3 * torch.randn(1, H, T, N, dtype=dt, device='cuda')
        decay = torch.sigmoid(0.5 + 0.1 * torch.randn(1, H, T, dtype=dt, device='cuda'))
        S0 = 0.1 * torch.randn(1, H, N, N, dtype=dt, device='cuda')
        # Baseline
        S_traj_v1 = pararnn_seq_fwd_triton(S0, K, V, decay, num_warps=1 if N == 16 else 4)
        dL_dout = 0.01 * torch.randn(1, H, T, N, dtype=dt, device='cuda')
        g_T = torch.zeros(1, H, N, N, dtype=dt, device='cuda')
        dS0_b, dK_b, dV_b, ddec_b = backward_e88_fused_rank1(
            S_traj_v1, K, V, decay, g_T, dL_dout, q,
            num_warps=1 if N == 16 else 2, num_stages=1)
        dQ_b = torch.einsum('bhti,bhtij->bhtj', dL_dout, S_traj_v1[:, :, 1:])

        # v2
        S_traj_v2 = pararnn_seq_fwd_v2(S0, K, V, decay, num_warps=1 if N == 16 else 4)
        dS0_v, dK_v, dV_v, dQ_v, ddec_v = backward_v2(
            S0, S_traj_v2, K, V, decay, g_T, dL_dout, q,
            num_warps=1 if N == 16 else 2, num_stages=1)

        def rel(a, b): return (a - b).abs().max().item() / max(b.abs().max().item(), 1e-10)
        r = {
            'S_traj': rel(S_traj_v2, S_traj_v1[:, :, 1:]),
            'dS0': rel(dS0_v, dS0_b), 'dK': rel(dK_v, dK_b),
            'dV': rel(dV_v, dV_b), 'dQ': rel(dQ_v, dQ_b),
            'ddec': rel(ddec_v, ddec_b),
        }
        w = max(r.values())
        ok = "PASS" if w < 1e-3 else "FAIL"
        print(f"  H={H} T={T} N={N}:  " + "  ".join(f"{k}={v:.1e}" for k, v in r.items()) + f"  [{ok}]")

    # Benchmark: v2 vs v1 paths
    print("\nBenchmark (compute forward + backward, includes Sq/dQ):\n")
    for H, N in [(141, 16), (83, 32)]:
        for T in [16384, 32768]:
            dt = torch.bfloat16
            g = torch.Generator(device='cuda').manual_seed(0)
            K = 0.3 * torch.randn(1, H, T, N, generator=g, dtype=dt, device='cuda')
            V = 0.3 * torch.randn(1, H, T, N, generator=g, dtype=dt, device='cuda')
            q = 0.3 * torch.randn(1, H, T, N, generator=g, dtype=dt, device='cuda')
            decay = torch.sigmoid(0.5 + 0.1 * torch.randn(1, H, T, generator=g, dtype=dt, device='cuda'))
            S0 = 0.1 * torch.randn(1, H, N, N, generator=g, dtype=dt, device='cuda')
            dL_dout = 0.01 * torch.randn(1, H, T, N, dtype=dt, device='cuda')
            g_T = torch.zeros(1, H, N, N, dtype=dt, device='cuda')

            # v1: forward + Sq einsum + backward + dQ einsum
            def run_v1():
                S_traj = pararnn_seq_fwd_triton(S0, K, V, decay, num_warps=1 if N == 16 else 4)
                Sq = torch.einsum('bhtpq,bhtq->bhtp', S_traj[:, :, 1:], q)
                bwd = backward_e88_fused_rank1(S_traj, K, V, decay, g_T, dL_dout, q,
                                                 num_warps=1 if N == 16 else 2, num_stages=1)
                dQ = torch.einsum('bhti,bhtij->bhtj', dL_dout, S_traj[:, :, 1:])

            # v2: forward (contig [B,H,T,N,N]) + Sq einsum (contig!) + backward_v2
            def run_v2():
                S_traj = pararnn_seq_fwd_v2(S0, K, V, decay, num_warps=1 if N == 16 else 4)
                # Sq einsum on CONTIG S_traj
                Sq = torch.einsum('bhtpq,bhtq->bhtp', S_traj, q)
                # backward_v2 fuses dQ
                bwd = backward_v2(S0, S_traj, K, V, decay, g_T, dL_dout, q,
                                    num_warps=1 if N == 16 else 2, num_stages=1)

            for _ in range(5): run_v1()
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(10): run_v1()
            torch.cuda.synchronize()
            v1_ms = (time.time() - t0) / 10 * 1000

            for _ in range(5): run_v2()
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(10): run_v2()
            torch.cuda.synchronize()
            v2_ms = (time.time() - t0) / 10 * 1000
            print(f"  H={H:3d} T={T:6d} N={N}:  v1={v1_ms:>6.2f}ms  v2={v2_ms:>6.2f}ms  spd={v1_ms/v2_ms:.2f}×")
