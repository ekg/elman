"""Break down backward overhead of hybrid_with_fused_gate."""

import os, sys, time
import torch
import torch.nn.functional as F

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, THIS)
sys.path.insert(0, os.path.dirname(THIS))

from install_hybrid import install
install()

import elman.models.e88_fla_hybrid as e88m
from phase6_hybrid import PararnnHybridE88V2


def time_it(fn, n=15, warmup=3):
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

    def new_grad_inputs():
        k = torch.randn(B, T, H, N, dtype=dt, device='cuda').requires_grad_(True)
        v = torch.randn(B, T, H, N, dtype=dt, device='cuda').requires_grad_(True)
        q = torch.randn(B, T, H, N, dtype=dt, device='cuda').requires_grad_(True)
        d = torch.sigmoid(0.5 + 0.1 * torch.randn(B, T, H, dtype=dt, device='cuda')).detach().requires_grad_(True)
        g = torch.randn(B, T, H, N, dtype=dt, device='cuda').requires_grad_(True)
        S0 = (0.1 * torch.randn(B, H, N, N, dtype=dt, device='cuda')).requires_grad_(True)
        return k, v, q, d, g, S0

    def fwd_bwd_optimized():
        k, v, q, d, g, S0 = new_grad_inputs()
        S_final, out = e88m.E88OptimizedCUDAFunction.apply(True, k, v, q, d, g, S0, H, True, False, 16)
        out.sum().backward()

    def fwd_bwd_direct_hybrid():
        k = torch.randn(T, B, H, N, dtype=dt, device='cuda').requires_grad_(True)
        v = torch.randn(T, B, H, N, dtype=dt, device='cuda').requires_grad_(True)
        q = torch.randn(T, B, H, N, dtype=dt, device='cuda').requires_grad_(True)
        d = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, dtype=dt, device='cuda')).detach().requires_grad_(True)
        g = torch.randn(T, B, H, N, dtype=dt, device='cuda').requires_grad_(True)
        S0 = (0.1 * torch.randn(B, H, N, N, dtype=dt, device='cuda')).requires_grad_(True)
        from phase7_fused_gate_hybrid import hybrid_with_fused_gate
        S_final, out = hybrid_with_fused_gate(True, k, v, q, d, g, S0, H)
        out.sum().backward()

    print(f"Config: B={B} T={T} H={H} N={N}")
    print("=" * 72)
    print("Forward + Backward timing (median / min / max ms, 15 runs)")
    print("=" * 72)

    for name, fn in [
        ("E88OptimizedCUDAFunction fwd+bwd (BT layout)", fwd_bwd_optimized),
        ("hybrid_with_fused_gate fwd+bwd (TB layout)", fwd_bwd_direct_hybrid),
    ]:
        med, mn, mx = time_it(fn)
        print(f"  {name:<58s}: {med:>7.3f}  ({mn:.3f} - {mx:.3f})")


if __name__ == '__main__':
    main()
