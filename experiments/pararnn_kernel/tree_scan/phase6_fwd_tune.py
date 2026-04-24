"""Tune num_warps for the Triton sequential forward."""

import sys, os, time
import torch

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pararnn_seq_fwd_rect import pararnn_seq_fwd_output_triton


def bench(B, H, T, Ns, Hv, nw, n_repeat=3):
    dt = torch.bfloat16
    g = torch.Generator(device='cuda').manual_seed(0)
    K = 0.3 * torch.randn(B, H, T, Ns, generator=g, dtype=dt, device='cuda')
    V = 0.3 * torch.randn(B, H, T, Hv, generator=g, dtype=dt, device='cuda')
    Q = 0.3 * torch.randn(B, H, T, Ns, generator=g, dtype=dt, device='cuda')
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(B, H, T, generator=g, dtype=dt, device='cuda'))
    S0 = 0.1 * torch.randn(B, H, Hv, Ns, dtype=dt, device='cuda')  # Pararnn layout [M, N]

    def run():
        return pararnn_seq_fwd_output_triton(S0, K, V, Q, decay, num_warps=nw)
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
    for H, Ns, Hv in [(141, 16, 16), (83, 32, 32), (83, 32, 23), (141, 16, 14)]:
        for T in [16384, 32768]:
            print(f"  H={H} T={T} Ns={Ns} Hv={Hv}:")
            for nw in [1, 2, 4, 8]:
                ms = bench(1, H, T, Ns, Hv, nw)
                ok = f"{ms:7.2f} ms" if ms != float('inf') else "FAIL"
                print(f"    nw={nw}: {ok}")
