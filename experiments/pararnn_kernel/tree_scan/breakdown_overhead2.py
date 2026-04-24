"""Measure the exact path E88OptimizedCUDAFunction patched_optimized takes.

This replicates the monkey-patched _opt_apply wrapper used by time_fwd_and_bwd.py
to confirm the per-call timing matches the 21.8 ms/call figure.
"""

import os, sys, time, statistics
import torch
import torch.nn.functional as F

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, THIS)
sys.path.insert(0, os.path.dirname(THIS))

# Install patched version like in training
from install_hybrid import install
install()

import elman.models.e88_fla_hybrid as e88m
from phase6_hybrid import PararnnHybridE88V2


def time_it(fn, n=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(n)]
    for s, e in evs:
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in evs]
    return sorted(times)[n // 2], min(times), max(times)


def main():
    torch.manual_seed(0)
    B, T, H, N = 1, 32768, 141, 16
    dt = torch.bfloat16

    # Use [B, T, H, N] layout (what the optimized function sees)
    k_bt = torch.randn(B, T, H, N, dtype=dt, device='cuda')
    v_bt = torch.randn(B, T, H, N, dtype=dt, device='cuda')
    q_bt = torch.randn(B, T, H, N, dtype=dt, device='cuda')
    decay_bt = torch.sigmoid(0.5 + 0.1 * torch.randn(B, T, H, dtype=dt, device='cuda'))
    g_bt = torch.randn(B, T, H, N, dtype=dt, device='cuda')
    S0 = 0.1 * torch.randn(B, H, N, N, dtype=dt, device='cuda')

    # Path 1: full patched_optimized (apply_gate=True, normalize_kq=False)
    # This is what training actually invokes.
    def path_optimized_no_grad():
        # Clone grads-off versions; they dont need requires_grad because we measure forward only
        with torch.no_grad():
            S_final, out = e88m.E88OptimizedCUDAFunction.apply(
                True, k_bt, v_bt, q_bt, decay_bt, g_bt, S0, H, True, False, 16
            )
        return out

    def path_optimized_with_grad():
        k_ = k_bt.detach().clone().requires_grad_(True)
        v_ = v_bt.detach().clone().requires_grad_(True)
        q_ = q_bt.detach().clone().requires_grad_(True)
        d_ = decay_bt.detach().clone().requires_grad_(True)
        g_ = g_bt.detach().clone().requires_grad_(True)
        S_ = S0.detach().clone().requires_grad_(True)
        return e88m.E88OptimizedCUDAFunction.apply(True, k_, v_, q_, d_, g_, S_, H, True, False, 16)

    # Path 2: direct hybrid_with_fused_gate (skip the outer BT<->TB)
    from phase7_fused_gate_hybrid import hybrid_with_fused_gate

    k_tb = k_bt.transpose(0, 1).contiguous()
    v_tb = v_bt.transpose(0, 1).contiguous()
    q_tb = q_bt.transpose(0, 1).contiguous()
    decay_tb = decay_bt.transpose(0, 1).contiguous()
    g_tb = g_bt.transpose(0, 1).contiguous()

    def path_hybrid_fused_direct():
        with torch.no_grad():
            S_final, out_tb = hybrid_with_fused_gate(True, k_tb, v_tb, q_tb, decay_tb, g_tb, S0, H)
        return out_tb

    # Path 3: PararnnHybridE88V2.apply directly (no gate)
    def path_hybrid_v2_direct():
        with torch.no_grad():
            S_final, Sq = PararnnHybridE88V2.apply(True, k_tb, v_tb, q_tb, decay_tb, S0, H)
        return Sq

    # Path 4: Only the Triton kernel + einsum (no permutes, no gate)
    K_p = k_tb.permute(1, 2, 0, 3).contiguous()
    V_p = v_tb.permute(1, 2, 0, 3).contiguous()
    Q_p = q_tb.permute(1, 2, 0, 3).contiguous()
    decay_p = decay_tb.permute(1, 2, 0).contiguous()
    S0_p = S0.transpose(-1, -2).contiguous()

    from pararnn_seq_fwd_rect import pararnn_seq_fwd_output_triton

    def path_kernel_only():
        return pararnn_seq_fwd_output_triton(S0_p, K_p, V_p, Q_p, decay_p, num_warps=1)

    print(f"Config: B={B} T={T} H={H} N={N}")
    print("=" * 72)
    print("Per-call overhead breakdown (median / min / max ms, 20 runs)")
    print("=" * 72)

    for name, fn in [
        ("Triton kernel (no permutes, no gate, no grad)", path_kernel_only),
        ("PararnnHybridE88V2.apply [TB layout] no-grad", path_hybrid_v2_direct),
        ("hybrid_with_fused_gate [TB layout] no-grad", path_hybrid_fused_direct),
        ("E88OptimizedCUDAFunction.apply [BT layout] no-grad", path_optimized_no_grad),
        ("E88OptimizedCUDAFunction.apply [BT layout] grad-tracking", path_optimized_with_grad),
    ]:
        med, mn, mx = time_it(fn)
        print(f"  {name:<58s}: {med:>7.3f}  ({mn:.3f} - {mx:.3f})")


if __name__ == '__main__':
    main()
