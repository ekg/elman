"""Phase 2 — warm-up scan coarse solver.

Insight: for contractive E88 (decay < 1), state at time t depends almost
entirely on inputs in the last W steps (decay^W is small).
  At decay=0.95 W=100: 0.95^100 ≈ 0.006 → initial state contributes 0.6%.

So: to estimate boundary at time p*T_chunk, run E88 forward from ANY
initial state over the last W positions [p*T_chunk - W, p*T_chunk].
Decay wipes out initial-state error; we get a good boundary.

All P-1 boundary estimates can be computed IN PARALLEL via one CUDA call
with batched chunks of size W each.

Expected cost: ~P × W × per-step-cost = 16 × 100 × ~10μs/step ≈ 16 ms.
But in parallel via batched CUDA: much less.
"""

import sys, os, time
import torch

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elman.models.e88_fla_hybrid import E88FLAHybridCUDAFunction
from phase1_warmstart_bench import admm_forward_fixed_iters


def warmup_scan_boundaries(S0, k, v, q, decay, H, P, W=128):
    """Compute chunk-boundary estimates via warm-up scan.

    For each chunk p in [1, P-1], run E88 forward on positions
    [p*T_chunk - W, p*T_chunk) starting from broadcast(S0). Use output as
    boundary estimate.

    All P-1 warm-ups batched into one CUDA call.

    Inputs (CUDA layout [T, B, H, N]):
      S0: [B, H, N, N]
      k, v, q: [T, B, H, N]
      decay: [T, B, H]
      P: number of chunks
      W: warm-up window size

    Returns:
      boundaries: [B, P, H, N, N]
        boundaries[:, 0] = S0
        boundaries[:, p] for p>=1 = warm-up-scan estimate
    """
    T, B, H, N = k.shape
    assert T % P == 0
    T_chunk = T // P
    dt = k.dtype

    # Boundary 0 is exact (= S0).
    # For p in [1, P-1], we need state at t = p*T_chunk.
    # Warm-up window: [p*T_chunk - W, p*T_chunk), length W.
    # If p*T_chunk < W (chunks too small), use available positions.
    # For our configs W=128 T_chunk>=1024, so always have enough.

    # Build batched input: for each p in [1, P-1], stack the warm-up window.
    # Shape: [W, B * (P-1), H, N]
    # Each "chain" in the batch is a warm-up for one boundary.
    num_warmups = P - 1
    warm_k = torch.empty(W, B * num_warmups, H, N, dtype=dt, device=k.device)
    warm_v = torch.empty(W, B * num_warmups, H, N, dtype=dt, device=k.device)
    warm_q = torch.empty(W, B * num_warmups, H, N, dtype=dt, device=k.device)
    warm_decay = torch.empty(W, B * num_warmups, H, dtype=dt, device=k.device)

    for p in range(1, P):
        start = p * T_chunk - W
        end = p * T_chunk
        # Layout: warm_*[., b*num_warmups + (p-1), ., .] = k[start:end, b, ., .]
        for b in range(B):
            slot = b * num_warmups + (p - 1)
            warm_k[:, slot] = k[start:end, b]
            warm_v[:, slot] = v[start:end, b]
            warm_q[:, slot] = q[start:end, b]
            warm_decay[:, slot] = decay[start:end, b]

    # Initial state: broadcast S0 for each warm-up
    # S0_warm[b * num_warmups + (p-1)] = S0[b]
    S0_warm = S0.unsqueeze(1).expand(B, num_warmups, H, N, N).reshape(B * num_warmups, H, N, N).contiguous()

    # Run CUDA forward on warm-ups
    S_end_warm, _ = E88FLAHybridCUDAFunction.apply(
        True, warm_k.contiguous(), warm_v.contiguous(), warm_q.contiguous(),
        warm_decay.contiguous(), S0_warm, H
    )
    # S_end_warm shape: [B * num_warmups, H, N, N]
    S_end_warm = S_end_warm.view(B, num_warmups, H, N, N)  # [B, P-1, H, N, N]

    # Assemble full boundaries tensor
    boundaries = torch.empty(B, P, H, N, N, dtype=dt, device=k.device)
    boundaries[:, 0] = S0
    boundaries[:, 1:] = S_end_warm

    return boundaries


def measure_warmup_quality(B, H, T, N, P, W):
    """Check warm-up scan boundary accuracy vs true boundaries."""
    dt = torch.bfloat16
    g = torch.Generator(device='cuda').manual_seed(0)
    k = 0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')
    v = 0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')
    q = 0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, generator=g, dtype=dt, device='cuda'))
    S0 = 0.1 * torch.randn(B, H, N, N, generator=g, dtype=dt, device='cuda')

    # Time warm-up scan
    torch.cuda.synchronize()
    t0 = time.time()
    warm_bd = warmup_scan_boundaries(S0, k, v, q, decay, H, P, W=W)
    torch.cuda.synchronize()
    warm_ms = (time.time() - t0) * 1000

    # True boundaries via full ADMM
    _, true_bd, _ = admm_forward_fixed_iters(S0, k, v, q, decay, H, P,
                                               num_iters=5, init_boundaries=None)

    err = (warm_bd.float() - true_bd.float()).abs()
    max_err = err[:, 1:].max().item()  # skip boundary[0]=S0 exact
    print(f"  H={H} T={T} N={N} P={P} W={W}  warmup={warm_ms:.2f}ms  "
          f"max|warm-true|={max_err:.2e}  (bf16 ~ 8e-3)")
    return warm_ms, max_err


def bench_1iter_warmup(B, H, T, N, P, W, n_repeat=3):
    """Full pipeline: warm-up coarse solver + 1 ADMM iter."""
    dt = torch.bfloat16
    g = torch.Generator(device='cuda').manual_seed(0)
    k = 0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')
    v = 0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')
    q = 0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, generator=g, dtype=dt, device='cuda'))
    S0 = 0.1 * torch.randn(B, H, N, N, generator=g, dtype=dt, device='cuda')

    def run():
        bd = warmup_scan_boundaries(S0, k, v, q, decay, H, P, W=W)
        _, _, out = admm_forward_fixed_iters(S0, k, v, q, decay, H, P,
                                                num_iters=1, init_boundaries=bd)
        return out

    for _ in range(3): run()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat): _ = run()
    torch.cuda.synchronize()
    return (time.time() - t0) / n_repeat * 1000


if __name__ == '__main__':
    print("Phase 2 — warm-up scan boundary quality\n")
    for H, N in [(141, 16), (83, 32)]:
        for T in [32768, 65536]:
            for W in [32, 64, 128, 256]:
                try:
                    measure_warmup_quality(1, H, T, N, P=16, W=W)
                except Exception as e:
                    print(f"  FAIL: {str(e)[:100]}")

    print("\n\nPhase 2 — full pipeline: warm-up + 1-iter ADMM vs 2-iter cold-start\n")
    print(f"{'Config':>22s}  {'T':>6s}  {'W':>4s}  {'coarse+1iter':>13s}  {'2-iter cold':>12s}  {'Speedup':>8s}")
    for H, N in [(141, 16), (83, 32)]:
        for T in [32768, 65536]:
            W = 128
            from phase1_warmstart_bench import bench_2iter_cold
            t2cold = bench_2iter_cold(1, H, T, N, P=16)
            t_full = bench_1iter_warmup(1, H, T, N, P=16, W=W)
            name = f"H={H} N={N}"
            print(f"{name:>22s}  {T:>6d}  {W:>4d}  {t_full:>10.2f} ms  {t2cold:>9.2f} ms  {t2cold/t_full:>6.2f}×")
