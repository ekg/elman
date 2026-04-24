"""Break down the per-call overhead of hybrid_with_fused_gate in the E88 training path.

Measures each subpiece separately using cudaEvent timing:
  1. permute+contiguous on k/v/q/decay/g (input layout transform)
  2. S0 transpose+contiguous
  3. Triton forward kernel (pararnn_seq_fwd_v2 or pararnn_seq_fwd_output_triton)
  4. einsum for Sq (v2 path) or fused with kernel (non-v2 path)
  5. S_final transpose+contiguous
  6. output permute+contiguous
  7. F.silu(g) + multiply
  8. autograd.Function.apply + ctx.save_for_backward overhead

Runs 20 iterations with cuda events, reports median and mean.
"""

import os, sys, time, statistics
import torch
import torch.nn.functional as F

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, THIS)
sys.path.insert(0, os.path.dirname(THIS))

from pararnn_seq_fwd_v2 import pararnn_seq_fwd_v2
from pararnn_seq_fwd_rect import pararnn_seq_fwd_output_triton


def time_it(fn, n=20, warmup=5):
    """CUDA-event-timed runtime in ms, list of times."""
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
    return sorted(times)[n // 2], times  # median


def main():
    torch.manual_seed(0)
    # E88-n16 production config at one training layer
    B, T, H, N = 1, 32768, 141, 16
    dt = torch.bfloat16

    # Allocate with the layout the E88OptimizedCUDAFunction sees: [B, T, H, N]
    k_bt = (0.3 * torch.randn(B, T, H, N, dtype=dt, device='cuda'))
    v_bt = (0.3 * torch.randn(B, T, H, N, dtype=dt, device='cuda'))
    q_bt = (0.3 * torch.randn(B, T, H, N, dtype=dt, device='cuda'))
    decay_bt = torch.sigmoid(0.5 + 0.1 * torch.randn(B, T, H, dtype=dt, device='cuda'))
    g_bt = torch.randn(B, T, H, N, dtype=dt, device='cuda')
    S0 = 0.1 * torch.randn(B, H, N, N, dtype=dt, device='cuda')

    print(f"Config: B={B} T={T} H={H} N={N} dtype={dt}")
    print(f"Tensors: k,v,q: {k_bt.shape}  decay: {decay_bt.shape}  g: {g_bt.shape}  S0: {S0.shape}")
    print()

    # ========================================================================
    # Stage 1: [B,T,H,N] -> [T,B,H,N] transpose (install_hybrid does this first)
    # ========================================================================
    def stage_1_tb_transpose():
        k_tb = k_bt.transpose(0, 1).contiguous()
        v_tb = v_bt.transpose(0, 1).contiguous()
        q_tb = q_bt.transpose(0, 1).contiguous()
        decay_tb = decay_bt.transpose(0, 1).contiguous()
        g_tb = g_bt.transpose(0, 1).contiguous()
        return k_tb, v_tb, q_tb, decay_tb, g_tb

    # Precompute to use in later stages
    k_tb, v_tb, q_tb, decay_tb, g_tb = stage_1_tb_transpose()

    # ========================================================================
    # Stage 2: [T,B,H,N] -> [B,H,T,N] permute (phase6_hybrid does this inside)
    # ========================================================================
    def stage_2_pararnn_permute():
        K_p = k_tb.permute(1, 2, 0, 3).contiguous()
        V_p = v_tb.permute(1, 2, 0, 3).contiguous()
        Q_p = q_tb.permute(1, 2, 0, 3).contiguous()
        decay_p = decay_tb.permute(1, 2, 0).contiguous()
        S0_p = S0.transpose(-1, -2).contiguous()
        return K_p, V_p, Q_p, decay_p, S0_p

    K_p, V_p, Q_p, decay_p, S0_p = stage_2_pararnn_permute()

    # ========================================================================
    # Stage 3: Triton forward kernel (no wrapper, no einsum, no permutes)
    # ========================================================================
    use_v2 = (N >= 32)  # For N=16, uses non-v2 path (fused output kernel)
    print(f"Using v2 path: {use_v2}")

    if use_v2:
        # v2 path: separate kernel + einsum
        def stage_3_fwd_kernel():
            return pararnn_seq_fwd_v2(S0_p, K_p, V_p, decay_p, num_warps=4)

        S_traj = stage_3_fwd_kernel()

        def stage_4_einsum_sq():
            return torch.einsum('bhtpq,bhtq->bhtp', S_traj, Q_p)

        Sq = stage_4_einsum_sq()
    else:
        # N=16: fused fwd+output kernel (no separate einsum needed)
        def stage_3_fwd_kernel():
            fwd_nw = 4 if max(N, N) >= 24 else 1
            return pararnn_seq_fwd_output_triton(S0_p, K_p, V_p, Q_p, decay_p, num_warps=fwd_nw)

        S_traj, Sq = stage_3_fwd_kernel()

    # ========================================================================
    # Stage 5: S_final transpose + contiguous back to CUDA layout
    # ========================================================================
    S_final_p = S_traj[:, :, -1]

    def stage_5_sfinal_transpose():
        return S_final_p.transpose(-1, -2).contiguous()

    # ========================================================================
    # Stage 6: output permute [B,H,T,M] -> [T,B,H,M]
    # ========================================================================
    def stage_6_output_permute():
        return Sq.permute(2, 0, 1, 3).contiguous()

    output_tb = stage_6_output_permute()

    # ========================================================================
    # Stage 7: F.silu(g) * output
    # ========================================================================
    def stage_7_silu_gate():
        return output_tb * F.silu(g_tb)

    # ========================================================================
    # Stage 8: output [T,B,H,N] -> [B,T,H,N] (to go back to optimized caller layout)
    # ========================================================================
    output_silu_tb = stage_7_silu_gate()

    def stage_8_out_bt():
        return output_silu_tb.transpose(0, 1).contiguous()

    # ========================================================================
    # Full pipeline: end-to-end hybrid_with_fused_gate (from install_hybrid)
    # ========================================================================
    from phase6_hybrid import PararnnHybridE88V2

    def full_pipeline_grad_off():
        """Replicates install_hybrid's patched_optimized call without autograd tracking."""
        k_tb_local = k_bt.transpose(0, 1).contiguous()
        v_tb_local = v_bt.transpose(0, 1).contiguous()
        q_tb_local = q_bt.transpose(0, 1).contiguous()
        decay_tb_local = decay_bt.transpose(0, 1).contiguous()
        g_tb_local = g_bt.transpose(0, 1).contiguous()

        with torch.no_grad():
            S_final, Sq_out = PararnnHybridE88V2.apply(True, k_tb_local, v_tb_local, q_tb_local, decay_tb_local, S0, H)
            out = Sq_out * F.silu(g_tb_local)
            return out.transpose(0, 1).contiguous()

    def full_pipeline_grad_on():
        """Includes autograd.Function.apply + ctx.save_for_backward overhead."""
        k_tb_local = k_bt.transpose(0, 1).contiguous().requires_grad_(True)
        v_tb_local = v_bt.transpose(0, 1).contiguous().requires_grad_(True)
        q_tb_local = q_bt.transpose(0, 1).contiguous().requires_grad_(True)
        decay_tb_local = decay_bt.transpose(0, 1).contiguous().requires_grad_(True)
        g_tb_local = g_bt.transpose(0, 1).contiguous().requires_grad_(True)
        S0_local = S0.clone().requires_grad_(True)

        S_final, Sq_out = PararnnHybridE88V2.apply(True, k_tb_local, v_tb_local, q_tb_local, decay_tb_local, S0_local, H)
        out = Sq_out * F.silu(g_tb_local)
        return out.transpose(0, 1).contiguous()

    # ========================================================================
    # Run benchmarks
    # ========================================================================
    print("=" * 72)
    print("STAGE BREAKDOWN (median of 20 runs, ms)")
    print("=" * 72)

    stages = [
        ("Stage 1: BT->TB transpose (k,v,q,decay,g)", stage_1_tb_transpose),
        ("Stage 2: TB->Pararnn permute (5 tensors)", stage_2_pararnn_permute),
        ("Stage 3: Triton forward kernel", stage_3_fwd_kernel),
    ]
    if use_v2:
        stages.append(("Stage 4: einsum for Sq", stage_4_einsum_sq))
    stages += [
        ("Stage 5: S_final transpose", stage_5_sfinal_transpose),
        ("Stage 6: output permute B H T M -> T B H M", stage_6_output_permute),
        ("Stage 7: F.silu(g) * output", stage_7_silu_gate),
        ("Stage 8: output TB->BT", stage_8_out_bt),
    ]

    total_stage_sum = 0.0
    for name, fn in stages:
        med, times = time_it(fn)
        total_stage_sum += med
        print(f"  {name:<50s}: {med:>7.3f} ms  (min={min(times):.3f} max={max(times):.3f})")

    print(f"  {'SUM OF STAGES':<50s}: {total_stage_sum:>7.3f} ms")
    print()

    # End-to-end without autograd tracking
    med_nograd, _ = time_it(full_pipeline_grad_off)
    print(f"  {'Full pipeline (no grad)':<50s}: {med_nograd:>7.3f} ms")

    # End-to-end with autograd tracking
    med_grad, _ = time_it(full_pipeline_grad_on)
    print(f"  {'Full pipeline (grad on)':<50s}: {med_grad:>7.3f} ms")

    print(f"  {'  autograd overhead':<50s}: {med_grad - med_nograd:>7.3f} ms")
    print(f"  {'  Stage-sum minus kernel-only':<50s}: {total_stage_sum - stages[2][1] and (total_stage_sum - time_it(stages[2][1])[0]):>7.3f} ms")
    print()

    # Kernel-only: just the Triton kernel
    kernel_med, _ = time_it(stage_3_fwd_kernel)
    print(f"  Triton kernel alone         : {kernel_med:>7.3f} ms")
    print(f"  End-to-end full             : {med_grad:>7.3f} ms")
    print(f"  OVERHEAD above kernel       : {med_grad - kernel_med:>7.3f} ms")


if __name__ == '__main__':
    main()
