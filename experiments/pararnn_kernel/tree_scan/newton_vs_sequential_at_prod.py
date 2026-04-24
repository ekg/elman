"""Serious benchmark: Newton vs sequential forward at PRODUCTION shapes.

The memory notes claim Newton loses to sequential at H>=32.  Let's see
exactly how much, and whether iter count is the issue or raw work.
"""

import sys, os, time
import torch

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase4_newton_driver import newton_e88_triton, sequential_e88_forward
from pararnn_seq_fwd_v2 import pararnn_seq_fwd_v2


def bench_fn(fn, n_repeat=5):
    for _ in range(2): fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat): fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n_repeat * 1000


def measure(B, H, T, N, label=""):
    dt = torch.float32  # Newton driver is fp32
    torch.manual_seed(0)
    S0 = (0.1 * torch.randn(B, H, N, N, dtype=dt, device='cuda'))
    K = 0.3 * torch.randn(B, H, T, N, dtype=dt, device='cuda')
    V = 0.3 * torch.randn(B, H, T, N, dtype=dt, device='cuda')
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(B, H, T, dtype=dt, device='cuda'))

    # Sequential reference
    def run_seq():
        return sequential_e88_forward(S0, K, V, decay)
    seq_ms = bench_fn(run_seq)

    # Triton sequential (our fast implementation, bf16)
    S0_bf = S0.to(torch.bfloat16)
    K_bf = K.to(torch.bfloat16)
    V_bf = V.to(torch.bfloat16)
    decay_bf = decay.to(torch.bfloat16)
    # Pararnn_S = CUDA_S^T — transpose for consistency
    S0_bf_t = S0_bf.transpose(-1, -2).contiguous()
    def run_triton_seq():
        return pararnn_seq_fwd_v2(S0_bf_t, K_bf, V_bf, decay_bf,
                                   num_warps=1 if N == 16 else 4)
    tri_ms = bench_fn(run_triton_seq)

    # Newton — warmup, then time.  Track iteration count.
    _, iters, res = newton_e88_triton(S0, K, V, decay, tol=1e-4, max_iters=30)
    def run_newton():
        return newton_e88_triton(S0, K, V, decay, tol=1e-4, max_iters=30)
    newton_ms = bench_fn(run_newton, n_repeat=3)

    print(f"  {label}  B={B} H={H} T={T} N={N}:")
    print(f"    sequential Python (fp32):      {seq_ms:>8.2f} ms")
    print(f"    Triton seq_fwd_v2 (bf16):      {tri_ms:>8.2f} ms")
    print(f"    Newton Triton (fp32, {iters} iters, res={res:.1e}): {newton_ms:>8.2f} ms")
    print(f"    Newton vs Triton seq:          {tri_ms/newton_ms:>5.2f}×  {'WIN' if newton_ms < tri_ms else 'lose'}")


if __name__ == '__main__':
    # Production shapes
    print("=== Production E88-n16 (H=141 N=16) ===")
    for T in [1024, 4096, 16384, 32768]:
        try:
            measure(1, 141, T, 16, label=f"T={T}")
        except Exception as e:
            print(f"  T={T}: FAIL {str(e)[:100]}")
            torch.cuda.empty_cache()

    print("\n=== Production E88-n32 (H=83 N=32) ===")
    for T in [1024, 4096, 16384]:
        try:
            measure(1, 83, T, 32, label=f"T={T}")
        except Exception as e:
            print(f"  T={T}: FAIL {str(e)[:100]}")
            torch.cuda.empty_cache()

    print("\n=== Low H regime (H=8 N=16) — where Newton is supposed to win ===")
    for T in [4096, 16384, 65536]:
        try:
            measure(1, 8, T, 16, label=f"T={T}")
        except Exception as e:
            print(f"  T={T}: FAIL {str(e)[:100]}")
            torch.cuda.empty_cache()
