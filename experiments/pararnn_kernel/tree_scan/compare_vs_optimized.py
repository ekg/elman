"""Compare Pararnn hybrid vs THE REAL production kernel (E88OptimizedCUDAFunction).

E88OptimizedCUDAFunction (warp/coalesced variant) is what the training
actually uses — not E88FLAHybridCUDAFunction which we've been benchmarking
against.
"""

import sys, os, time
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elman.models.e88_fla_hybrid import (
    E88OptimizedCUDAFunction, E88FLAHybridCUDAFunction, E88FusedGateCUDAFunction,
)
from phase6_hybrid import PararnnHybridE88V2
from phase7_fused_gate_hybrid import hybrid_with_fused_gate


def bench_fn(fn, n_repeat=10):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat): fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n_repeat * 1000


def compare(B, T, H, N):
    print(f"\n=== B={B} T={T} H={H} N={N} ===")
    dt = torch.bfloat16
    torch.manual_seed(0)
    # Optimized kernel uses [B, T, H, N] layout
    k_bt = (0.3 * torch.randn(B, T, H, N, dtype=dt, device='cuda')).requires_grad_(True)
    v_bt = (0.3 * torch.randn(B, T, H, N, dtype=dt, device='cuda')).requires_grad_(True)
    q_bt = (0.3 * torch.randn(B, T, H, N, dtype=dt, device='cuda')).requires_grad_(True)
    decay_bt = torch.sigmoid(0.5 + 0.1 * torch.randn(B, T, H, dtype=dt, device='cuda')).detach().requires_grad_(True)
    g_bt = torch.randn(B, T, H, N, dtype=dt, device='cuda').requires_grad_(True)
    S0 = (0.1 * torch.randn(B, H, N, N, dtype=dt, device='cuda')).requires_grad_(True)

    # Hybrid + FusedGate used [T, B, H, N] layout
    k_tb = k_bt.detach().transpose(0, 1).contiguous().requires_grad_(True)
    v_tb = v_bt.detach().transpose(0, 1).contiguous().requires_grad_(True)
    q_tb = q_bt.detach().transpose(0, 1).contiguous().requires_grad_(True)
    decay_tb = decay_bt.detach().transpose(0, 1).contiguous().requires_grad_(True)
    g_tb = g_bt.detach().transpose(0, 1).contiguous().requires_grad_(True)

    # === E88OptimizedCUDAFunction (the production default) ===
    def run_opt():
        S_f, out = E88OptimizedCUDAFunction.apply(
            True, k_bt, v_bt, q_bt, decay_bt, g_bt, S0, H, True, False, 16
        )
        out.sum().backward()
        k_bt.grad = None; v_bt.grad = None; q_bt.grad = None
        decay_bt.grad = None; g_bt.grad = None; S0.grad = None

    # === E88FusedGateCUDAFunction (the non-Optimized "fused gate" baseline) ===
    def run_fused_gate():
        S_f, out = E88FusedGateCUDAFunction.apply(
            True, k_tb, v_tb, q_tb, decay_tb, g_tb, S0, H
        )
        out.sum().backward()
        for t in [k_tb, v_tb, q_tb, decay_tb, g_tb, S0]:
            if t.grad is not None: t.grad = None

    # === Pararnn hybrid with fused gate ===
    def run_hybrid():
        S_f, out = hybrid_with_fused_gate(True, k_tb, v_tb, q_tb, decay_tb, g_tb, S0, H)
        out.sum().backward()
        for t in [k_tb, v_tb, q_tb, decay_tb, g_tb, S0]:
            if t.grad is not None: t.grad = None

    try:
        opt_ms = bench_fn(run_opt)
    except Exception as e:
        opt_ms = float('nan'); print(f"  Opt failed: {e}")
    try:
        fused_ms = bench_fn(run_fused_gate)
    except Exception as e:
        fused_ms = float('nan'); print(f"  Fused failed: {e}")
    try:
        hyb_ms = bench_fn(run_hybrid)
    except Exception as e:
        hyb_ms = float('nan'); print(f"  Hyb failed: {e}")

    print(f"  E88Optimized (PRODUCTION)  : {opt_ms:>7.3f} ms")
    print(f"  E88FusedGate (baseline)    : {fused_ms:>7.3f} ms")
    print(f"  PararnnHybrid (ours)       : {hyb_ms:>7.3f} ms")
    print(f"  Hybrid vs Optimized        : {opt_ms/hyb_ms:.2f}×  {'WIN' if hyb_ms < opt_ms else 'lose'}")
    print(f"  Hybrid vs FusedGate        : {fused_ms/hyb_ms:.2f}×  {'WIN' if hyb_ms < fused_ms else 'lose'}")


if __name__ == '__main__':
    # Training shape
    compare(16, 512, 141, 16)
    compare(16, 512, 83, 32)
    # Long shape
    compare(1, 16384, 141, 16)
    compare(1, 16384, 83, 32)
    compare(1, 32768, 141, 16)
