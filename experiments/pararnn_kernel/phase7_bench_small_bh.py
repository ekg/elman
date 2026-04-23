"""Scenario: low B*H where CUDA under-saturates the GPU.

CUDA kernel has B*H blocks (32 at B=1 H=32, but only 1 at B=1 H=1).
Pararnn has B*H*N programs (32x more — saturates GPU even at low B*H).

Hypothesis: Pararnn wins at small B*H because it uses more SMs.
"""

import sys
import os
import time
import torch

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase7_fused_iter import fused_newton_iter_inplace
from phase4_newton_driver import sequential_e88_forward
from elman.models.e88_fla_hybrid import E88FLAHybridCUDAFunction


def setup_both(B, H, T, n):
    g = torch.Generator(device='cuda').manual_seed(0)
    dt = torch.float32
    S0 = 0.1 * torch.randn(B, H, n, n, generator=g, dtype=dt, device='cuda')
    K = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    V = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    decay = 0.9 + 0.1 * torch.rand(B, H, T, generator=g, dtype=dt, device='cuda')

    # Warmstart: sequential forward + perturb
    S_prev = sequential_e88_forward(S0, K, V, decay)
    S_init = S_prev[:, :, 1:].clone()
    del S_prev
    torch.cuda.empty_cache()
    CHUNK = 8192
    for t_chunk in range(0, T, CHUNK):
        t_end = min(t_chunk + CHUNK, T)
        S_init[:, :, t_chunk:t_end].add_(
            0.02 * torch.randn_like(S_init[:, :, t_chunk:t_end])
        )

    # CUDA inputs
    g2 = torch.Generator(device='cuda').manual_seed(0)
    dt_c = torch.bfloat16
    k_c = (0.3 * torch.randn(T, B, H, n, generator=g2, dtype=dt_c, device='cuda')).requires_grad_(False)
    v_c = (0.3 * torch.randn(T, B, H, n, generator=g2, dtype=dt_c, device='cuda')).requires_grad_(False)
    q_c = (0.3 * torch.randn(T, B, H, n, generator=g2, dtype=dt_c, device='cuda')).requires_grad_(False)
    dec_c = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, generator=g2, dtype=dt_c, device='cuda'))
    S0_c = 0.1 * torch.randn(B, H, n, n, generator=g2, dtype=dt_c, device='cuda')

    return (S0, K, V, decay, S_init), (k_c, v_c, q_c, dec_c, S0_c, H)


def bench(B, H, T, n, n_iter=3, n_repeat=3):
    par_in, cuda_in = setup_both(B, H, T, n)
    S0, K, V, decay, S_init = par_in
    k_c, v_c, q_c, dec_c, S0_c, Hh = cuda_in

    def run_par():
        for _ in range(n_iter):
            fused_newton_iter_inplace(S0, S_init, K, V, decay)

    def run_cuda():
        _ = E88FLAHybridCUDAFunction.apply(True, k_c, v_c, q_c, dec_c, S0_c, Hh)

    for _ in range(2):
        run_par()
        run_cuda()
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(n_repeat):
        run_par()
    torch.cuda.synchronize()
    ms_par = (time.time() - t0) / n_repeat * 1000

    t0 = time.time()
    for _ in range(n_repeat):
        run_cuda()
    torch.cuda.synchronize()
    ms_cuda = (time.time() - t0) / n_repeat * 1000

    n_cuda_blocks = B * H
    n_par_programs = B * H * n
    print(f"  B={B:>2d} H={H:>3d} T={T:>6d} n={n}  "
          f"CUDA={ms_cuda:>7.1f}ms ({n_cuda_blocks:>5d} blocks)  "
          f"Par-3iter={ms_par:>7.1f}ms ({n_par_programs:>6d} progs)  "
          f"speedup={ms_cuda/ms_par:>5.2f}×")


if __name__ == '__main__':
    print("\nSmall B*H where CUDA under-saturates:\n")
    for B, H, T, n in [(1, 1, 16384, 32),
                       (1, 2, 16384, 32),
                       (1, 4, 16384, 32),
                       (1, 8, 16384, 32),
                       (1, 16, 16384, 32),
                       (1, 32, 16384, 32),
                       (1, 1, 65536, 32),
                       (1, 4, 65536, 32),
                       (1, 16, 65536, 32),
                       (1, 32, 65536, 32),
                       (1, 1, 131072, 32),
                       (1, 4, 131072, 32),
                       (1, 16, 131072, 32),
                       (1, 32, 131072, 32),
                       ]:
        try:
            torch.cuda.empty_cache()
            bench(B, H, T, n)
        except Exception as e:
            print(f"  FAIL B={B} H={H} T={T}: {str(e)[:100]}")
