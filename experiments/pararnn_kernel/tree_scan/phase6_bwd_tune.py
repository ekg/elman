"""Sweep num_warps × num_stages for backward_e88_fused_rank1 at production shapes."""

import sys, os, time
import torch

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase7_fused_backward import backward_e88_fused_rank1
from pararnn_seq_fwd_rect import pararnn_seq_fwd_output_triton


def bench(B, H, T, N, nw, ns, n_repeat=3):
    dt = torch.bfloat16
    g = torch.Generator(device='cuda').manual_seed(0)
    K = 0.3 * torch.randn(B, H, T, N, generator=g, dtype=dt, device='cuda')
    V = 0.3 * torch.randn(B, H, T, N, generator=g, dtype=dt, device='cuda')
    q = 0.3 * torch.randn(B, H, T, N, generator=g, dtype=dt, device='cuda')
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(B, H, T, generator=g, dtype=dt, device='cuda'))
    S0 = 0.1 * torch.randn(B, H, N, N, generator=g, dtype=dt, device='cuda')

    # Build S_traj via Triton fwd (realistic)
    S_traj, _ = pararnn_seq_fwd_output_triton(S0, K, V, q, decay, num_warps=1 if N < 32 else 4)

    dL_dout = 0.01 * torch.randn(B, H, T, N, dtype=dt, device='cuda')
    g_T = torch.zeros(B, H, N, N, dtype=dt, device='cuda')

    def run():
        return backward_e88_fused_rank1(S_traj, K, V, decay, g_T, dL_dout, q,
                                          num_warps=nw, num_stages=ns)
    try:
        for _ in range(3): run()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_repeat): run()
        torch.cuda.synchronize()
        return (time.time() - t0) / n_repeat * 1000
    except Exception as e:
        return float('inf')


if __name__ == '__main__':
    for H, N in [(141, 16), (83, 32)]:
        for T in [16384, 32768]:
            print(f"\n  H={H} T={T} N={N}  num_warps × num_stages:")
            for nw in [1, 2, 4, 8]:
                for ns in [1, 2, 3]:
                    try:
                        ms = bench(1, H, T, N, nw, ns)
                        ok = f"{ms:7.2f} ms" if ms != float('inf') else "FAIL"
                    except Exception as e:
                        ok = f"ERR"
                    print(f"    nw={nw} ns={ns}:  {ok}")
