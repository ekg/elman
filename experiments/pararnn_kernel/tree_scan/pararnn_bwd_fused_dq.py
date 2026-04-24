"""Backward with dQ fused INTO the kernel.

At each iteration t, the kernel already recomputes S_{t+1} = tanh(pre).
We just emit dQ[t, c] = sum_p dL_dout[t, p] * S_{t+1}[p, c] right there,
avoiding a separate einsum pass over S_traj[:, :, 1:] (which is
non-contiguous and ~2× slower than a bmm on contiguous data).

Keeps the off-by-one fix from phase7_fused_backward.py.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_backward_with_dq_kernel(
    S_traj_ptr, K_ptr, V_ptr, decay_ptr,
    g_T_ptr, dL_dout_ptr, q_ptr,
    g_out_ptr, dK_ptr, dV_ptr, ddecay_ptr, dQ_ptr,
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, N: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    b = pid // H
    h = pid % H
    bh = b * H + h
    row_idx = tl.arange(0, N).to(tl.int64)
    col_idx = tl.arange(0, N).to(tl.int64)
    tile_2d = row_idx[:, None] * N + col_idx[None, :]
    gT_base = bh * N * N
    S_traj_head_stride = (T + 1) * N * N
    K_head_stride = T * N
    dec_head_stride = T

    g = tl.load(g_T_ptr + gT_base + tile_2d).to(tl.float32)

    for t_inv in range(T):
        t = T - 1 - t_inv

        dL_dout_t = tl.load(dL_dout_ptr + bh * K_head_stride + t * N + row_idx).to(tl.float32)
        q_t = tl.load(q_ptr + bh * K_head_stride + t * N + col_idx).to(tl.float32)
        # output-grad injection: g (= dL/dS_{t+1}) += dL_dout[t] ⊗ q[t]
        g = g + dL_dout_t[:, None] * q_t[None, :]

        K_t = tl.load(K_ptr + bh * K_head_stride + t * N + col_idx).to(tl.float32)
        V_t = tl.load(V_ptr + bh * K_head_stride + t * N + row_idx).to(tl.float32)
        dec = tl.load(decay_ptr + bh * dec_head_stride + t).to(tl.float32)
        S_prev = tl.load(
            S_traj_ptr + bh * S_traj_head_stride + t * N * N + tile_2d
        ).to(tl.float32)

        retrieved = tl.sum(S_prev * K_t[None, :], axis=1)
        delta_row = V_t - retrieved

        pre = dec * S_prev + delta_row[:, None] * K_t[None, :]
        e2x = tl.exp(2.0 * pre)
        tanh_val = (e2x - 1.0) / (e2x + 1.0)   # = S_{t+1}
        tanh_deriv = 1.0 - tanh_val * tanh_val

        # Fused dQ: dQ[t, c] = sum_p dL_dout[t, p] * S_{t+1}[p, c]
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

    tl.store(g_out_ptr + gT_base + tile_2d, g.to(g_out_ptr.dtype.element_ty))


def backward_with_dq(S_traj, K, V, decay, g_T, dL_dout, q, num_warps=1, num_stages=1):
    B, H, T1, N, _ = S_traj.shape
    T = T1 - 1
    device = S_traj.device
    dtype = S_traj.dtype
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)
    dQ = torch.zeros_like(q)
    ddec = torch.zeros_like(decay)
    dS0 = torch.zeros(B, H, N, N, dtype=dtype, device=device)
    grid = (B * H,)
    _fused_backward_with_dq_kernel[grid](
        S_traj.contiguous(), K.contiguous(), V.contiguous(), decay.contiguous(),
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

    # Correctness
    print("Correctness (fused dq vs baseline + separate einsum):")
    for H, T, N in [(4, 512, 16), (4, 1024, 32)]:
        dt = torch.float32
        torch.manual_seed(0)
        K = 0.3 * torch.randn(1, H, T, N, dtype=dt, device='cuda')
        V = 0.3 * torch.randn(1, H, T, N, dtype=dt, device='cuda')
        q = 0.3 * torch.randn(1, H, T, N, dtype=dt, device='cuda')
        decay = torch.sigmoid(0.5 + 0.1 * torch.randn(1, H, T, dtype=dt, device='cuda'))
        S0 = 0.1 * torch.randn(1, H, N, N, dtype=dt, device='cuda')
        S_traj = pararnn_seq_fwd_triton(S0, K, V, decay, num_warps=1 if N == 16 else 4)
        dL_dout = 0.01 * torch.randn(1, H, T, N, dtype=dt, device='cuda')
        g_T = torch.zeros(1, H, N, N, dtype=dt, device='cuda')

        # Baseline: separate einsum
        dS0_b, dK_b, dV_b, ddec_b = backward_e88_fused_rank1(
            S_traj, K, V, decay, g_T, dL_dout, q,
            num_warps=1 if N == 16 else 2, num_stages=1)
        dQ_b = torch.einsum('bhti,bhtij->bhtj', dL_dout, S_traj[:, :, 1:])

        # Fused
        dS0_f, dK_f, dV_f, dQ_f, ddec_f = backward_with_dq(
            S_traj, K, V, decay, g_T, dL_dout, q,
            num_warps=1 if N == 16 else 2, num_stages=1)

        def rel(a, b): return (a - b).abs().max().item() / max(b.abs().max().item(), 1e-10)
        r = {
            'dS0': rel(dS0_f, dS0_b), 'dK': rel(dK_f, dK_b),
            'dV': rel(dV_f, dV_b), 'dQ': rel(dQ_f, dQ_b),
            'ddec': rel(ddec_f, ddec_b),
        }
        w = max(r.values())
        ok = "PASS" if w < 1e-3 else "FAIL"
        print(f"  H={H} T={T} N={N}:  " + "  ".join(f"{k}={v:.1e}" for k, v in r.items()) + f"  [{ok}]")

    # Benchmark
    print("\nBenchmark: fused-dQ backward vs baseline (backward+einsum)\n")
    for H, N in [(141, 16), (83, 32)]:
        for T in [16384, 32768]:
            dt = torch.bfloat16
            g = torch.Generator(device='cuda').manual_seed(0)
            K = 0.3 * torch.randn(1, H, T, N, generator=g, dtype=dt, device='cuda')
            V = 0.3 * torch.randn(1, H, T, N, generator=g, dtype=dt, device='cuda')
            q = 0.3 * torch.randn(1, H, T, N, generator=g, dtype=dt, device='cuda')
            decay = torch.sigmoid(0.5 + 0.1 * torch.randn(1, H, T, generator=g, dtype=dt, device='cuda'))
            S0 = 0.1 * torch.randn(1, H, N, N, generator=g, dtype=dt, device='cuda')
            S_traj = pararnn_seq_fwd_triton(S0, K, V, decay, num_warps=1 if N == 16 else 4)
            dL_dout = 0.01 * torch.randn(1, H, T, N, dtype=dt, device='cuda')
            g_T = torch.zeros(1, H, N, N, dtype=dt, device='cuda')

            def run_base():
                r = backward_e88_fused_rank1(S_traj, K, V, decay, g_T, dL_dout, q,
                                               num_warps=1 if N == 16 else 2, num_stages=1)
                dQ = torch.einsum('bhti,bhtij->bhtj', dL_dout, S_traj[:, :, 1:])
                return r, dQ
            def run_fused():
                return backward_with_dq(S_traj, K, V, decay, g_T, dL_dout, q,
                                          num_warps=1 if N == 16 else 2, num_stages=1)
            for _ in range(5): run_base()
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(10): run_base()
            torch.cuda.synchronize()
            base_ms = (time.time() - t0) / 10 * 1000

            for _ in range(5): run_fused()
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(10): run_fused()
            torch.cuda.synchronize()
            fus_ms = (time.time() - t0) / 10 * 1000
            print(f"  H={H:3d} T={T:6d} N={N}:  base={base_ms:>6.2f}ms  fused_dq={fus_ms:>6.2f}ms  "
                  f"spd={base_ms/fus_ms:.2f}×")
