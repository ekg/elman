"""Test CUDA graphs at higher batch (where kernel launches matter more)."""

import os, sys
import torch
import torch.nn.functional as F

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, THIS)
sys.path.insert(0, os.path.dirname(THIS))

from install_hybrid import install
install()

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
    H, N = 141, 16
    dt = torch.bfloat16

    # Short T = large T-count test where kernel launches amortize less
    for B, T in [(16, 2048), (32, 1024), (64, 512)]:
        print(f"\n{'=' * 60}")
        print(f"B={B} T={T} H={H} N={N}")
        print('=' * 60)

        k_tb = torch.randn(T, B, H, N, dtype=dt, device='cuda')
        v_tb = torch.randn(T, B, H, N, dtype=dt, device='cuda')
        q_tb = torch.randn(T, B, H, N, dtype=dt, device='cuda')
        decay_tb = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, dtype=dt, device='cuda'))
        g_tb = torch.randn(T, B, H, N, dtype=dt, device='cuda')
        S0 = 0.1 * torch.randn(B, H, N, N, dtype=dt, device='cuda')

        def baseline():
            with torch.no_grad():
                _, out = hybrid_with_fused_gate(True, k_tb, v_tb, q_tb, decay_tb, g_tb, S0, H)
            return out

        med, mn, mx = time_it(baseline)
        print(f"  baseline no-grad:    {med:>7.3f} ms")

        # CUDA graph
        try:
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

            med_g, mn_g, mx_g = time_it(replay)
            print(f"  CUDA graph replay:   {med_g:>7.3f} ms   speedup={med/med_g:.2f}x")
        except Exception as e:
            print(f"  CUDA graph FAILED: {type(e).__name__}: {str(e)[:150]}")


if __name__ == '__main__':
    main()
