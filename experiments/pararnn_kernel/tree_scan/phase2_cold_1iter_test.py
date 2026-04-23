"""Phase 2 pre-check: is cold-start 1-iter ADMM already bf16-safe?

Since chunks are long (T_chunk=4096 at P=16, T=65K) and E88 decay<1
forgets initial state in O(1/(1-decay)) steps ≈ 20-100 steps, maybe the
BROADCAST(S0) cold-start with just 1 ADMM iter already produces
bf16-safe output — no coarse solver needed.
"""

import sys, os, time
import torch

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase1_warmstart_bench import admm_forward_fixed_iters


def compare(B, H, T, N, P):
    """Cold-start 1 iter vs 2 iter — measure output error."""
    dt = torch.bfloat16
    g = torch.Generator(device='cuda').manual_seed(0)
    k = 0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')
    v = 0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')
    q = 0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, generator=g, dtype=dt, device='cuda'))
    S0 = 0.1 * torch.randn(B, H, N, N, generator=g, dtype=dt, device='cuda')

    # 1 iter cold-start
    _, _, out_1iter = admm_forward_fixed_iters(S0, k, v, q, decay, H, P,
                                                  num_iters=1, init_boundaries=None)
    # 2 iter cold-start (reference)
    _, _, out_2iter = admm_forward_fixed_iters(S0, k, v, q, decay, H, P,
                                                  num_iters=2, init_boundaries=None)
    # 3 iter cold-start (overkill reference)
    _, _, out_3iter = admm_forward_fixed_iters(S0, k, v, q, decay, H, P,
                                                  num_iters=3, init_boundaries=None)

    # Errors at each position
    err_1v2 = (out_1iter.float() - out_2iter.float()).abs()
    err_2v3 = (out_2iter.float() - out_3iter.float()).abs()

    max_1v2 = err_1v2.max().item()
    max_2v3 = err_2v3.max().item()

    # Look at where error is concentrated
    err_per_t_1v2 = err_1v2.view(T, -1).max(dim=1)[0]
    T_chunk = T // P
    chunk_starts = [p * T_chunk for p in range(P)]
    err_at_starts = max(err_per_t_1v2[s].item() for s in chunk_starts)

    # Also at mid, end of each chunk
    chunk_offsets = [0, 1, 5, 20, 50, 100, T_chunk // 4, T_chunk // 2, T_chunk - 1]
    print(f"  H={H} T={T} N={N} P={P} T_chunk={T_chunk}")
    print(f"    1iter vs 2iter max err: {max_1v2:.2e}  (bf16 ~ 8e-3)")
    print(f"    2iter vs 3iter max err: {max_2v3:.2e}  (convergence verification)")
    print(f"    Error at each chunk offset from start (1iter vs 2iter):")
    for off in chunk_offsets:
        max_at_off = max(err_per_t_1v2[p * T_chunk + off].item() for p in range(P)
                          if p * T_chunk + off < T)
        print(f"      offset+{off:5d}: {max_at_off:.2e}")


if __name__ == '__main__':
    print("Is cold-start 1-iter already bf16-safe at our chunk sizes?\n")
    for H, N in [(141, 16), (83, 32)]:
        for T in [32768, 65536]:
            for P in [8, 16, 32]:
                try:
                    compare(1, H, T, N, P)
                    print()
                except Exception as e:
                    print(f"  FAIL: {str(e)[:100]}")
                    torch.cuda.empty_cache()
