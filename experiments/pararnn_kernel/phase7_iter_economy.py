"""How much precision does each Newton iter actually buy?

If warm-start is already at fp32 precision, subsequent iters are
wasted. We might only need 1 iter, maybe 2.
"""

import sys, os, time
import torch
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase7_fused_iter import fused_newton_iter_inplace
from phase4_newton_driver import sequential_e88_forward


def measure(B, H, T, n, perturb=0.02, max_iters=5):
    g = torch.Generator(device='cuda').manual_seed(0)
    dt = torch.float32
    S0 = 0.1 * torch.randn(B, H, n, n, generator=g, dtype=dt, device='cuda')
    K = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    V = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    decay = 0.9 + 0.1 * torch.rand(B, H, T, generator=g, dtype=dt, device='cuda')

    S_seq = sequential_e88_forward(S0, K, V, decay)[:, :, 1:]

    # Warm-start with perturbation
    S_init = S_seq.clone()
    CHUNK = 8192
    for t_chunk in range(0, T, CHUNK):
        t_end = min(t_chunk + CHUNK, T)
        S_init[:, :, t_chunk:t_end].add_(
            perturb * torch.randn_like(S_init[:, :, t_chunk:t_end])
        )

    # Time each iter individually, measure residual vs ground truth.
    # Reuse S_test memory: apply perturbation in place starting from previous
    # S_test to avoid additional 16 GB clones at T=128K.
    S_test = S_seq.clone()
    last_perturb = 0.0
    for perturb_scale in [0.001, 0.01, 0.02, 0.05, 0.1]:
        # Re-initialize S_test from S_seq by overwriting (still one clone-worth of HBM traffic but no alloc)
        for t_chunk in range(0, T, CHUNK):
            t_end = min(t_chunk + CHUNK, T)
            S_test[:, :, t_chunk:t_end].copy_(S_seq[:, :, t_chunk:t_end])
        for t_chunk in range(0, T, CHUNK):
            t_end = min(t_chunk + CHUNK, T)
            S_test[:, :, t_chunk:t_end].add_(
                perturb_scale * torch.randn_like(S_test[:, :, t_chunk:t_end])
            )
        # warmup
        torch.cuda.synchronize()
        line = f"  perturb={perturb_scale:5.3f}: "
        diff0 = (S_test - S_seq).abs().max().item()
        line += f"init|err|={diff0:.2e}  "
        for it in range(max_iters):
            t0 = time.time()
            d = fused_newton_iter_inplace(S0, S_test, K, V, decay)
            torch.cuda.synchronize()
            ms = (time.time() - t0) * 1000
            diff = (S_test - S_seq).abs().max().item()
            line += f"iter{it+1}|err|={diff:.2e}({ms:.0f}ms) "
        print(line)


if __name__ == '__main__':
    print("T=32K convergence by iter at varying warm-start quality:")
    measure(1, 32, 32768, 32)
    print("\nT=128K convergence:")
    measure(1, 32, 131072, 32)
