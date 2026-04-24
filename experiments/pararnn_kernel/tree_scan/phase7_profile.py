"""Break hybrid backward into phases and measure each separately."""

import sys, os, time
import torch

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase7_fused_backward import backward_e88_fused_rank1
from pararnn_seq_fwd import pararnn_seq_fwd_triton


def bench(fn, n_repeat=10):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat): fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n_repeat * 1000


def profile(H, N, T):
    dt = torch.bfloat16
    g = torch.Generator(device='cuda').manual_seed(0)
    K = 0.3 * torch.randn(1, H, T, N, generator=g, dtype=dt, device='cuda')
    V = 0.3 * torch.randn(1, H, T, N, generator=g, dtype=dt, device='cuda')
    q = 0.3 * torch.randn(1, H, T, N, generator=g, dtype=dt, device='cuda')
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(1, H, T, generator=g, dtype=dt, device='cuda'))
    S0 = 0.1 * torch.randn(1, H, N, N, generator=g, dtype=dt, device='cuda')
    dL_dout = 0.01 * torch.randn(1, H, T, N, dtype=dt, device='cuda')
    g_T = torch.zeros(1, H, N, N, dtype=dt, device='cuda')

    # Phase: forward (seq_fwd)
    def fn_fwd():
        return pararnn_seq_fwd_triton(S0, K, V, decay, num_warps=1 if N == 16 else 4)
    fwd_ms = bench(fn_fwd)

    S_traj = fn_fwd()

    # Phase: dQ einsum
    def fn_dq():
        return torch.einsum('bhti,bhtij->bhtj', dL_dout, S_traj[:, :, 1:])
    dq_ms = bench(fn_dq)

    # Phase: Sq output einsum (for split-fwd path at N>=32)
    def fn_sq():
        return torch.einsum('bhtpq,bhtq->bhtp', S_traj[:, :, 1:], q)
    sq_ms = bench(fn_sq)

    # Phase: backward kernel
    def fn_bwd():
        return backward_e88_fused_rank1(S_traj, K, V, decay, g_T, dL_dout, q,
                                          num_warps=1 if N == 16 else 2, num_stages=1)
    bwd_ms = bench(fn_bwd)

    # HBM traffic estimates
    S_traj_bytes = (T + 1) * H * N * N * 2  # bf16
    # Backward reads: S_traj once, K/V/q/dL_dout once, decay once
    per_step = N * 2 * 4 + 2   # K, V, q, dL_dout each N*bf16=32B, decay 2B
    per_step_writes = N * 2 * 2 + 2  # dK, dV (N*bf16) + ddec (2B)
    bwd_hbm_GB = (S_traj_bytes + T * H * (per_step + per_step_writes)) / 1e9

    # RTX 6000 Ada: ~960 GB/s peak GDDR6 BW
    bwd_peak_ms = bwd_hbm_GB / 960 * 1000

    print(f"\n  H={H} N={N} T={T}")
    print(f"    forward seq_fwd    : {fwd_ms:>6.2f} ms")
    print(f"    dQ einsum (bwd)    : {dq_ms:>6.2f} ms")
    print(f"    Sq einsum (fwd)    : {sq_ms:>6.2f} ms")
    print(f"    backward kernel    : {bwd_ms:>6.2f} ms")
    print(f"    total fwd+bwd      : {fwd_ms + sq_ms + bwd_ms + dq_ms:>6.2f} ms")
    print(f"    bwd HBM traffic    : {bwd_hbm_GB:.2f} GB  ({bwd_peak_ms:.2f} ms @ peak BW)")
    print(f"    bwd BW util        : {bwd_peak_ms / bwd_ms * 100:.0f}% of peak")


if __name__ == '__main__':
    for H, N, T in [(141, 16, 16384), (141, 16, 32768), (83, 32, 16384), (83, 32, 32768)]:
        profile(H, N, T)
