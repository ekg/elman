"""Cold-start convergence: how many iters from S_var = broadcast(S0) or zeros?

Cold-start is the realistic scenario for standard SGD training (fresh batches).
If cold-start needs 3 iters to converge at T=128K, Pararnn ties CUDA.
If cold-start needs 2 iters, Pararnn wins.
"""

import sys, os, time
import torch
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase7_fused_iter import fused_newton_iter_inplace


def measure(B, H, T, n, max_iters=6):
    g = torch.Generator(device='cuda').manual_seed(0)
    dt = torch.float32
    S0 = 0.1 * torch.randn(B, H, n, n, generator=g, dtype=dt, device='cuda')
    K = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    V = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    decay = 0.9 + 0.1 * torch.rand(B, H, T, generator=g, dtype=dt, device='cuda')

    print(f"\nT={T}  B={B} H={H} n={n}")

    # Cold-start strategies
    for strategy in ['zeros', 'broadcast_S0']:
        torch.cuda.empty_cache()
        if strategy == 'zeros':
            S_init = torch.zeros(B, H, T, n, n, dtype=dt, device='cuda')
        elif strategy == 'broadcast_S0':
            S_init = S0.unsqueeze(2).expand(B, H, T, n, n).contiguous()

        # warmup
        for _ in range(2):
            _ = fused_newton_iter_inplace(S0, S_init.clone(), K, V, decay)
        torch.cuda.synchronize()

        S_test = S_init
        line = f"  {strategy:>15s}: "
        total_ms = 0
        for it in range(max_iters):
            t0 = time.time()
            d_max = fused_newton_iter_inplace(S0, S_test, K, V, decay)
            torch.cuda.synchronize()
            ms = (time.time() - t0) * 1000
            total_ms += ms
            line += f"iter{it+1}|δ|={d_max:.2e}({ms:.0f}ms) "
            if d_max < 1e-5:
                line += f" → converged at iter {it+1}, total={total_ms:.0f}ms"
                break
        print(line)
        del S_test


if __name__ == '__main__':
    for T in [8192, 32768, 131072]:
        try:
            measure(1, 32, T, 32)
        except Exception as e:
            print(f"FAIL T={T}: {str(e)[:100]}")
            torch.cuda.empty_cache()
