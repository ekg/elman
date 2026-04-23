"""Verify fused_newton_iter_inplace gives the same trajectory as the
separate-phase Newton (which we already validated against sequential)."""

import sys
import os
import torch

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase7_fused_iter import fused_newton_iter, fused_newton_iter_inplace
from phase4_newton_driver import sequential_e88_forward


def test_inplace_matches_outofplace(B, H, T, n, seed=0):
    g = torch.Generator(device='cuda').manual_seed(seed)
    dt = torch.float32
    S0 = 0.1 * torch.randn(B, H, n, n, generator=g, dtype=dt, device='cuda')
    K = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    V = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    decay = 0.9 + 0.1 * torch.rand(B, H, T, generator=g, dtype=dt, device='cuda')

    S_traj = sequential_e88_forward(S0, K, V, decay)
    S_start = S_traj[:, :, 1:] * 0.8 + 0.01 * torch.randn_like(S_traj[:, :, 1:])

    # Out-of-place path
    S_oop = S_start.clone()
    for _ in range(3):
        d = fused_newton_iter(S0, S_oop, K, V, decay)
        S_oop = S_oop + d

    # In-place path
    S_ip = S_start.clone()
    for _ in range(3):
        _ = fused_newton_iter_inplace(S0, S_ip, K, V, decay)

    diff = (S_oop - S_ip).abs().max().item()
    # Final: both should also match sequential_e88_forward trajectory
    diff_seq_oop = (S_oop - S_traj[:, :, 1:]).abs().max().item()
    diff_seq_ip = (S_ip - S_traj[:, :, 1:]).abs().max().item()
    status = "PASS" if diff < 1e-5 else "FAIL"
    print(f"  B={B} H={H:3d} T={T:5d} n={n}  "
          f"|oop-ip|={diff:.2e}  |oop-seq|={diff_seq_oop:.2e}  "
          f"|ip-seq|={diff_seq_ip:.2e}  [{status}]")


if __name__ == '__main__':
    print("Verify in-place == out-of-place Newton trajectory:\n")
    for shape in [(1, 4, 128, 16), (1, 16, 512, 32),
                  (1, 32, 1024, 32), (1, 32, 4096, 32)]:
        test_inplace_matches_outofplace(*shape)
