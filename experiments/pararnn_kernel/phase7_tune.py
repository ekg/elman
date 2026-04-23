"""Tune num_stages and num_warps for the fused inplace kernel."""

import sys, os, time
import torch
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase7_fused_iter import fused_newton_iter_inplace
from phase4_newton_driver import sequential_e88_forward


def bench(B, H, T, n, num_stages, num_warps, n_repeat=5):
    g = torch.Generator(device='cuda').manual_seed(0)
    dt = torch.float32
    S0 = 0.1 * torch.randn(B, H, n, n, generator=g, dtype=dt, device='cuda')
    K = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    V = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    decay = 0.9 + 0.1 * torch.rand(B, H, T, generator=g, dtype=dt, device='cuda')

    S_prev = sequential_e88_forward(S0, K, V, decay)
    S_init = S_prev[:, :, 1:].clone()
    del S_prev
    torch.cuda.empty_cache()

    # warmup (compile)
    for _ in range(3):
        fused_newton_iter_inplace(S0, S_init, K, V, decay,
                                   num_stages=num_stages, num_warps=num_warps)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(n_repeat):
        fused_newton_iter_inplace(S0, S_init, K, V, decay,
                                   num_stages=num_stages, num_warps=num_warps)
    torch.cuda.synchronize()
    return (time.time() - t0) / n_repeat * 1000


if __name__ == '__main__':
    for T in [32768, 131072]:
        print(f"\nT={T}:")
        print(f"  {'stages':>7s} {'warps':>6s}  {'ms/iter':>8s}")
        for ns in [1, 2, 3, 4]:
            for nw in [1, 2, 4]:
                try:
                    torch.cuda.empty_cache()
                    ms = bench(1, 32, T, 32, ns, nw)
                    print(f"  {ns:>7d} {nw:>6d}  {ms:>6.1f} ms")
                except Exception as e:
                    print(f"  ns={ns} nw={nw}: FAIL — {str(e)[:80]}")
