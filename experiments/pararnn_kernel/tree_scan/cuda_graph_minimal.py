"""Minimal CUDA graph test: can we even capture the E88 forward through Triton?

Test at small scale first to see if the Triton kernels work in CUDA graphs at all.
Then scale up progressively until OOM.
"""

import os, sys, time
import torch
import torch.nn.functional as F

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, THIS)
sys.path.insert(0, os.path.dirname(THIS))

from install_hybrid import install
install()

from phase6_hybrid import PararnnHybridE88V2
from phase7_fused_gate_hybrid import hybrid_with_fused_gate


def time_it(fn, n=10, warmup=3):
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
    # Test at a single-layer scale first
    B, H, N = 1, 141, 16
    dt = torch.bfloat16

    for T in [4096, 8192, 16384, 32768]:
        print(f"\n{'=' * 60}")
        print(f"T={T}  B={B} H={H} N={N}")
        print('=' * 60)

        # Allocate static buffers (required for CUDA graph replay)
        k_tb = torch.randn(T, B, H, N, dtype=dt, device='cuda')
        v_tb = torch.randn(T, B, H, N, dtype=dt, device='cuda')
        q_tb = torch.randn(T, B, H, N, dtype=dt, device='cuda')
        decay_tb = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, dtype=dt, device='cuda'))
        g_tb = torch.randn(T, B, H, N, dtype=dt, device='cuda')
        S0 = 0.1 * torch.randn(B, H, N, N, dtype=dt, device='cuda')

        # Baseline
        def baseline():
            with torch.no_grad():
                _, out = hybrid_with_fused_gate(True, k_tb, v_tb, q_tb, decay_tb, g_tb, S0, H)
            return out

        med, mn, mx = time_it(baseline)
        print(f"  baseline no-grad:      {med:>7.3f} ms  ({mn:.3f}-{mx:.3f})")

        # CUDA graph
        try:
            # Warmup on side stream
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    with torch.no_grad():
                        _, out_warm = hybrid_with_fused_gate(True, k_tb, v_tb, q_tb, decay_tb, g_tb, S0, H)
            torch.cuda.current_stream().wait_stream(s)

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                with torch.no_grad():
                    _, out_captured = hybrid_with_fused_gate(True, k_tb, v_tb, q_tb, decay_tb, g_tb, S0, H)

            def replay():
                g.replay()

            med, mn, mx = time_it(replay)
            print(f"  CUDA graph replay:     {med:>7.3f} ms  ({mn:.3f}-{mx:.3f})")
        except Exception as e:
            print(f"  CUDA graph FAILED: {type(e).__name__}: {str(e)[:150]}")


if __name__ == '__main__':
    main()
