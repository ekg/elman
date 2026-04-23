"""Phase 1 benchmark: warm-start ADMM with 1 iter, measure trajectory error
and e2e speedup.
"""

import sys, os, time
import torch

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase4_newton_driver import sequential_e88_forward
from phase7_fused_backward import backward_e88_fused_rank1
from elman.models.e88_fla_hybrid import E88FLAHybridCUDAFunction


def cuda_forward(k, v, q, decay, S0, H):
    S_final, output = E88FLAHybridCUDAFunction.apply(True, k, v, q, decay, S0, H)
    return S_final, output


def admm_forward_fixed_iters(S0, k, v, q, decay, H, P, num_iters, init_boundaries=None):
    """ADMM forward with explicit iter count (no convergence check).

    With init_boundaries and num_iters=1: the 'warm-start single-iter' mode.
    With init_boundaries=None and num_iters=2: the current cold-start mode.

    Returns: (S_end per chunk, output per position)
      S_end: [B, P, H, N, N]
    """
    T = k.shape[0]
    B = k.shape[1]
    N = k.shape[3]
    T_chunk = T // P

    k_chunks = k.view(P, T_chunk, B, H, N).permute(1, 2, 0, 3, 4).reshape(T_chunk, B * P, H, N).contiguous()
    v_chunks = v.view(P, T_chunk, B, H, N).permute(1, 2, 0, 3, 4).reshape(T_chunk, B * P, H, N).contiguous()
    q_chunks = q.view(P, T_chunk, B, H, N).permute(1, 2, 0, 3, 4).reshape(T_chunk, B * P, H, N).contiguous()
    decay_chunks = decay.view(P, T_chunk, B, H).permute(1, 2, 0, 3).reshape(T_chunk, B * P, H).contiguous()

    if init_boundaries is None:
        S_boundary = S0.unsqueeze(1).expand(B, P, H, N, N).contiguous()
    else:
        S_boundary = init_boundaries.contiguous()
    S_boundary_flat = S_boundary.reshape(B * P, H, N, N).contiguous()

    for _ in range(num_iters):
        S_end_flat, output_chunks = cuda_forward(
            k_chunks, v_chunks, q_chunks, decay_chunks, S_boundary_flat, H)
        S_end = S_end_flat.view(B, P, H, N, N)
        S_boundary_new = S_boundary.clone()
        S_boundary_new[:, 1:, :, :, :] = S_end[:, :P - 1, :, :, :]
        S_boundary = S_boundary_new
        S_boundary_flat = S_boundary.reshape(B * P, H, N, N).contiguous()

    # After num_iters, S_end is the per-chunk end; output_chunks has the per-position outputs
    # output_chunks shape: [T_chunk, B*P, H, N]  → reassemble to [T, B, H, N]
    output_chunks_reshaped = output_chunks.view(T_chunk, B, P, H, N).permute(
        2, 0, 1, 3, 4).reshape(T, B, H, N).contiguous()

    return S_end, S_boundary, output_chunks_reshaped


def measure_trajectory_error(B, H, T, N, P, perturb=0.01):
    """Compare 1-warm-iter ADMM output to fully-converged output per position."""
    dt = torch.bfloat16
    g0 = torch.Generator(device='cuda').manual_seed(0)
    k0 = 0.3 * torch.randn(T, B, H, N, generator=g0, dtype=dt, device='cuda')
    v0 = 0.3 * torch.randn(T, B, H, N, generator=g0, dtype=dt, device='cuda')
    q0 = 0.3 * torch.randn(T, B, H, N, generator=g0, dtype=dt, device='cuda')
    decay0 = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, generator=g0, dtype=dt, device='cuda'))
    S0 = 0.1 * torch.randn(B, H, N, N, generator=g0, dtype=dt, device='cuda')

    # Step 0: converge fully to get warm boundaries
    _, warm_bd, _ = admm_forward_fixed_iters(S0, k0, v0, q0, decay0, H, P,
                                              num_iters=5, init_boundaries=None)

    # Step 1: perturbed inputs
    k1 = k0 + perturb * torch.randn_like(k0)
    v1 = v0 + perturb * torch.randn_like(v0)
    q1 = q0 + perturb * torch.randn_like(q0)
    decay1 = torch.sigmoid(torch.logit(decay0.clamp(1e-6, 1 - 1e-6))
                           + perturb * torch.randn_like(decay0))

    # Fully-converged reference with cold-start (more iters for exactness)
    _, _, out_true = admm_forward_fixed_iters(S0, k1, v1, q1, decay1, H, P,
                                                num_iters=5, init_boundaries=None)

    # 1 warm iter
    _, _, out_warm1 = admm_forward_fixed_iters(S0, k1, v1, q1, decay1, H, P,
                                                 num_iters=1, init_boundaries=warm_bd)

    # 2 warm iter (sanity check)
    _, _, out_warm2 = admm_forward_fixed_iters(S0, k1, v1, q1, decay1, H, P,
                                                 num_iters=2, init_boundaries=warm_bd)

    # Errors at each chunk position (max over H, N)
    # out has shape [T, B, H, N]. We'd expect error near chunk starts.
    err_warm1 = (out_warm1.float() - out_true.float()).abs().max(dim=1)[0].max(dim=1)[0].max(dim=1)[0]  # [T]
    err_warm2 = (out_warm2.float() - out_true.float()).abs().max(dim=1)[0].max(dim=1)[0].max(dim=1)[0]  # [T]

    # Summarize: error at chunk boundaries vs chunk centers
    T_chunk = T // P
    chunk_starts = torch.arange(P, device='cuda') * T_chunk
    chunk_mid = chunk_starts + T_chunk // 2
    chunk_end = (torch.arange(P, device='cuda') + 1) * T_chunk - 1

    max_err_warm1 = err_warm1.max().item()
    max_err_warm2 = err_warm2.max().item()

    # Where is the error concentrated?
    err_at_starts = err_warm1[chunk_starts].max().item()
    err_at_mids = err_warm1[chunk_mid].max().item()
    err_at_ends = err_warm1[chunk_end].max().item()

    print(f"  perturb={perturb:.3f}  "
          f"1-warm max err={max_err_warm1:.2e} (starts={err_at_starts:.2e}, mids={err_at_mids:.2e}, ends={err_at_ends:.2e})   "
          f"2-warm={max_err_warm2:.2e}")
    return max_err_warm1, max_err_warm2


def bench_1iter_warm(B, H, T, N, P, n_repeat=3):
    """Simulate real training: ADMM with warm-start from prev converged iter, 1 iter only."""
    dt = torch.bfloat16
    g = torch.Generator(device='cuda').manual_seed(0)
    k = 0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')
    v = 0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')
    q = 0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, generator=g, dtype=dt, device='cuda'))
    S0 = 0.1 * torch.randn(B, H, N, N, generator=g, dtype=dt, device='cuda')

    # Pre-warm boundaries
    _, warm_bd, _ = admm_forward_fixed_iters(S0, k, v, q, decay, H, P,
                                              num_iters=5, init_boundaries=None)

    for _ in range(3):
        _, _, _ = admm_forward_fixed_iters(S0, k, v, q, decay, H, P,
                                             num_iters=1, init_boundaries=warm_bd)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat):
        _, _, _ = admm_forward_fixed_iters(S0, k, v, q, decay, H, P,
                                             num_iters=1, init_boundaries=warm_bd)
    torch.cuda.synchronize()
    return (time.time() - t0) / n_repeat * 1000


def bench_2iter_cold(B, H, T, N, P, n_repeat=3):
    """Current state: ADMM cold-start 2 iters."""
    dt = torch.bfloat16
    g = torch.Generator(device='cuda').manual_seed(0)
    k = 0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')
    v = 0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')
    q = 0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, generator=g, dtype=dt, device='cuda'))
    S0 = 0.1 * torch.randn(B, H, N, N, generator=g, dtype=dt, device='cuda')

    for _ in range(3):
        _, _, _ = admm_forward_fixed_iters(S0, k, v, q, decay, H, P, num_iters=2)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat):
        _, _, _ = admm_forward_fixed_iters(S0, k, v, q, decay, H, P, num_iters=2)
    torch.cuda.synchronize()
    return (time.time() - t0) / n_repeat * 1000


if __name__ == '__main__':
    print("Phase 1 — trajectory error: does 1 warm iter produce bf16-OK output?\n")
    for name, H, N in [("E88-n16 480M", 141, 16), ("E88-n32 480M", 83, 32)]:
        print(f"\n{name} T=32768 P=16  (bf16 ~ 8e-3 precision)")
        for perturb in [0.001, 0.01, 0.1, 1.0]:  # cover full range
            measure_trajectory_error(1, H, 32768, N, P=16, perturb=perturb)

    print("\n\nPhase 1 — 1-warm-iter speedup vs 2-iter cold-start:\n")
    print(f"{'Config':>22s}  {'T':>6s}  {'2-iter cold':>12s}  {'1-iter warm':>12s}  {'Speedup':>8s}")
    for name, H, N in [("E88-n16 480M", 141, 16), ("E88-n32 480M", 83, 32)]:
        for T in [16384, 32768, 65536]:
            t2 = bench_2iter_cold(1, H, T, N, P=16)
            t1 = bench_1iter_warm(1, H, T, N, P=16)
            print(f"{name:>22s}  {T:>6d}  {t2:>9.2f} ms  {t1:>9.2f} ms  {t2/t1:>6.2f}×")
