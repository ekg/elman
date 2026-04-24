"""Profile: where does time go in quasi-DEER Newton iter?"""

import sys
import os
import time
import torch

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quasi_deer_ref import _random_case, build_diag_ingredients
from quasi_deer_triton import qd_diagonal_scan_triton


def bench(fn, warmup=3, repeat=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / repeat * 1000


def profile_iter(H, T, N):
    S0, K, V, decay = _random_case(1, H, T, N, seed=0, dtype=torch.float32,
                                    l2_normalize_k=True, v_scale=0.3)
    S_var = torch.zeros(1, H, T, N, N, dtype=torch.float32, device='cuda')

    def run_ingredients():
        _ = build_diag_ingredients(S0, S_var, K, V, decay)
    t_ing = bench(run_ingredients)

    D, bv, _ = build_diag_ingredients(S0, S_var, K, V, decay)
    def run_scan():
        _ = qd_diagonal_scan_triton(D, bv, block_T=512)
    t_scan = bench(run_scan)

    def run_update():
        _ = S_var + D  # stand-in for addition
    t_add = bench(run_update)

    total = t_ing + t_scan + t_add
    print(f"H={H:3d} T={T:6d} N={N:2d}:  ingredients={t_ing:6.2f} ms  "
          f"scan={t_scan:5.2f} ms  add={t_add:5.2f} ms  "
          f"total/iter={total:6.2f} ms   (scan={100*t_scan/total:.0f}% of iter)")
    return t_ing, t_scan, t_add


if __name__ == '__main__':
    print("Breakdown of quasi-DEER Newton iteration cost:\n")
    for H, T, N in [(141, 1024, 16), (141, 4096, 16), (141, 16384, 16),
                    (83, 1024, 32), (83, 4096, 32),
                    (32, 65536, 16)]:
        profile_iter(H, T, N)
