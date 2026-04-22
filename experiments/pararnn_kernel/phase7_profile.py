"""Phase 7 profiling — where does Newton's time actually go?

Before optimizing, measure. Times each sub-operation of one Newton
iteration at varying T, so we know what to optimize.
"""

import sys
import os
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase4_newton_driver import (
    compute_residuals_batched, build_step_ingredients, scan_r1_triton_5d
)
from phase1_reference import _random_inputs


def profile_newton_iter(B, H, T, n, dtype=torch.float32, device='cuda'):
    torch.manual_seed(0)
    g = torch.Generator(device=device).manual_seed(0)
    S0 = 0.1 * torch.randn(B, H, n, n, generator=g, dtype=dtype, device=device)
    K = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dtype, device=device)
    V = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dtype, device=device)
    decay = 0.9 + 0.1 * torch.rand(B, H, T, generator=g, dtype=dtype, device=device)

    # After first Newton iter, S_var is nonzero — use a reasonable guess
    from phase4_newton_driver import sequential_e88_forward
    S_traj = sequential_e88_forward(S0, K, V, decay)
    S_var = S_traj[:, :, 1:].clone()

    def bench(fn, n_warm=2, n_iter=5):
        for _ in range(n_warm):
            fn()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iter):
            fn()
        torch.cuda.synchronize()
        return (time.time() - t0) / n_iter * 1000

    # --- residual compute ---
    def do_res():
        _ = compute_residuals_batched(S0, S_var, K, V, decay)
    res_ms = bench(do_res)

    r = compute_residuals_batched(S0, S_var, K, V, decay)

    # --- step ingredient build ---
    def do_build():
        _ = build_step_ingredients(S0, S_var, K, V, decay, r)
    build_ms = bench(do_build)

    D, u, v, b = build_step_ingredients(S0, S_var, K, V, decay, r)

    # --- scan ---
    def do_scan():
        _ = scan_r1_triton_5d(D, u, v, b)
    scan_ms = bench(do_scan)

    delta = scan_r1_triton_5d(D, u, v, b)

    # --- update ---
    def do_update():
        _ = S_var + delta
    update_ms = bench(do_update)

    total = res_ms + build_ms + scan_ms + update_ms
    print(f"B={B} H={H} T={T:5d} n={n}: "
          f"residual={res_ms:.2f} build={build_ms:.2f} "
          f"scan={scan_ms:.2f} update={update_ms:.2f}  total/iter={total:.2f} ms")
    # Memory: peak allocated during Newton iter
    print(f"    peak mem: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB "
          f"(reset after)")
    torch.cuda.reset_peak_memory_stats()

    return total


if __name__ == '__main__':
    print("Per-Newton-iter cost breakdown:\n")
    for B, H, T, n in [(1, 32, 512, 32),
                       (1, 32, 2048, 32),
                       (1, 32, 8192, 32)]:
        profile_newton_iter(B, H, T, n)
