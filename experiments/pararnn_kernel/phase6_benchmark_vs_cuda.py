"""Phase 6 benchmark: Pararnn-E88 Triton kernel vs existing E88 CUDA kernel.

The E88 CUDA forward/backward (E88FLAHybridCUDAFunction) is the production
path. Our Pararnn Newton approach is the candidate replacement for long-T.

We compare:
  forward + backward wall clock at (B, H, T, n) shape sweep.

Dtype note:
  - E88 CUDA kernel REQUIRES bfloat16.
  - Pararnn uses fp32 by default (Newton has tighter precision needs).
  - For the benchmark, each path uses its native dtype. That's the
    realistic deployed comparison.
"""

import sys
import os
import time
import torch

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, '/home/erikg/elman/elman/cuda')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase6_autograd import pararnn_e88

# Import the existing E88 CUDA path
from elman.models.e88_fla_hybrid import E88FLAHybridCUDAFunction


def bench_cuda(B, H, T, n, n_warmup=3, n_iter=5, device='cuda'):
    """Time forward+backward of the existing E88 CUDA kernel."""
    torch.manual_seed(0)
    dt = torch.bfloat16

    def fresh():
        g = torch.Generator(device=device).manual_seed(0)
        k = (0.3 * torch.randn(T, B, H, n, generator=g, dtype=dt, device=device)).requires_grad_(True)
        v = (0.3 * torch.randn(T, B, H, n, generator=g, dtype=dt, device=device)).requires_grad_(True)
        q = (0.3 * torch.randn(T, B, H, n, generator=g, dtype=dt, device=device)).requires_grad_(True)
        decay = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, generator=g, dtype=dt, device=device)
                              ).detach().requires_grad_(True)
        S0 = (0.1 * torch.randn(B, H, n, n, generator=g, dtype=dt, device=device))
        return k, v, q, decay, S0

    def run_once():
        k, v, q, decay, S0 = fresh()
        S_final, output = E88FLAHybridCUDAFunction.apply(
            True, k, v, q, decay, S0, H
        )
        # Random-direction scalar loss hitting all outputs
        target = torch.randn_like(output) * 0.01
        loss = (target * output).sum() + S_final.pow(2).sum() * 1e-4
        loss.backward()

    for _ in range(n_warmup):
        run_once()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iter):
        run_once()
    torch.cuda.synchronize()
    return (time.time() - t0) / n_iter * 1000


def bench_pararnn(B, H, T, n, n_warmup=3, n_iter=5, device='cuda'):
    """Time forward+backward of Pararnn-E88, computing the same output
    shape (trajectory * q for the readout)."""
    torch.manual_seed(0)
    dt = torch.float32  # Newton needs tighter precision

    def fresh():
        g = torch.Generator(device=device).manual_seed(0)
        # Pararnn uses [B, H, T, n] layout
        K = (0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device=device)).requires_grad_(True)
        V = (0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device=device)).requires_grad_(True)
        Q = (0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device=device)).requires_grad_(True)
        decay = torch.sigmoid(0.5 + 0.1 * torch.randn(B, H, T, generator=g, dtype=dt, device=device)
                              ).detach().requires_grad_(True)
        S0 = (0.1 * torch.randn(B, H, n, n, generator=g, dtype=dt, device=device)).requires_grad_(True)
        return K, V, Q, decay, S0

    def run_once():
        K, V, Q, decay, S0 = fresh()
        S_traj = pararnn_e88(S0, K, V, decay, tol=1e-4)
        # Readout: Sq[t] = S_traj[t+1] @ Q[t]  (matching CUDA kernel's output semantics)
        Sq = torch.einsum('bhtij,bhtj->bhti', S_traj[:, :, 1:], Q)
        S_final = S_traj[:, :, -1]
        target = torch.randn_like(Sq) * 0.01
        loss = (target * Sq).sum() + S_final.pow(2).sum() * 1e-4
        loss.backward()

    for _ in range(n_warmup):
        run_once()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iter):
        run_once()
    torch.cuda.synchronize()
    return (time.time() - t0) / n_iter * 1000


def sweep(shapes):
    print(f"{'B':>3s}  {'H':>3s}  {'T':>6s}  {'n':>3s}  "
          f"{'CUDA (bf16)':>14s}  {'Pararnn (fp32)':>16s}  {'Ratio (C/P)':>14s}")
    print("-" * 72)
    for B, H, T, n in shapes:
        try:
            cuda_ms = bench_cuda(B, H, T, n)
        except Exception as e:
            cuda_ms = float('nan')
            print(f"  CUDA failed at T={T}: {str(e)[:80]}")
        try:
            par_ms = bench_pararnn(B, H, T, n)
        except Exception as e:
            par_ms = float('nan')
            print(f"  Pararnn failed at T={T}: {str(e)[:80]}")

        if cuda_ms == cuda_ms and par_ms == par_ms:  # both not NaN
            ratio = cuda_ms / par_ms
            print(f"{B:>3d}  {H:>3d}  {T:>6d}  {n:>3d}  {cuda_ms:>11.1f} ms  "
                  f"{par_ms:>13.1f} ms  {ratio:>11.2f}×")


if __name__ == '__main__':
    # Start conservative; grow T. Use n=32 which is E88's optimal.
    # B=1 H=32 matches mid-scale E88 inference/training shape.
    shapes = [
        (1, 32, 512,    32),
        (1, 32, 2048,   32),
        (1, 32, 8192,   32),
        (1, 32, 32768,  32),
        (1, 32, 65536,  32),
    ]
    print("\nSweep at B=1, H=32, n=32 (small batch, increasing T):\n")
    sweep(shapes)
