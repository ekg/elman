"""Phase 1 — warm-start ADMM.

Question: when can we skip the first outer iter by reusing boundaries
from a previous training step?

Realistic training scenarios:
  (a) Gradient accumulation: same batch, multiple micro-batches. Params
      constant within an accumulation; K,V,decay IDENTICAL → warm-start
      should converge in 0 outer iters (boundaries already exact).
  (b) Truncated BPTT: same sequence, sliding window. K,V,decay partially
      overlap. Warm-start correct for overlap region, wrong for new part.
  (c) Standard SGD: fresh batch each step. K,V,decay unrelated. Warm-start
      boundaries are STALE — probably out of basin.

This phase measures convergence iters for each scenario.

If scenario (a) gives 1 iter and scenario (c) gives 2-3 iters, the
warm-start optimization is conditional: apply when available, fall back
to 2 iters when not.
"""

import sys, os, time
import torch

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elman.models.e88_fla_hybrid import E88FLAHybridCUDAFunction


def cuda_forward(k, v, q, decay, S0, H):
    S_final, output = E88FLAHybridCUDAFunction.apply(True, k, v, q, decay, S0, H)
    return S_final, output


def admm_cuda_forward(S0, k, v, q, decay, H, P, init_boundaries=None,
                      max_iters=10, tol=1e-4):
    """Parallel-in-time ADMM with optional warm-start boundaries.

    init_boundaries: [B, P, H, N, N] or None (defaults to broadcast(S0))
    """
    T = k.shape[0]
    B = k.shape[1]
    N = k.shape[3]
    assert T % P == 0
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

    for it in range(max_iters):
        S_end_flat, _ = cuda_forward(k_chunks, v_chunks, q_chunks, decay_chunks, S_boundary_flat, H)
        S_end = S_end_flat.view(B, P, H, N, N)

        S_boundary_new = S_boundary.clone()
        S_boundary_new[:, 1:, :, :, :] = S_end[:, :P - 1, :, :, :]

        d_max = (S_boundary_new - S_boundary).float().abs().max().item()
        S_boundary = S_boundary_new
        S_boundary_flat = S_boundary.reshape(B * P, H, N, N).contiguous()

        if d_max < tol:
            break

    return S_end, S_boundary, it + 1


def generate_inputs(B, H, T, N, seed=0, dtype=torch.bfloat16):
    g = torch.Generator(device='cuda').manual_seed(seed)
    k = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dtype, device='cuda'))
    v = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dtype, device='cuda'))
    q = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dtype, device='cuda'))
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, generator=g, dtype=dtype, device='cuda'))
    S0 = 0.1 * torch.randn(B, H, N, N, generator=g, dtype=dtype, device='cuda')
    return k, v, q, decay, S0


def test_tolerance_scan(B, H, T, N, P, perturb=0.001):
    """Sweep tolerance and see when warm-start converges in 1 iter."""
    k0, v0, q0, decay0, S0 = generate_inputs(B, H, T, N, seed=0)
    _, converged_boundaries, _ = admm_cuda_forward(
        S0, k0, v0, q0, decay0, H, P, init_boundaries=None, max_iters=10, tol=1e-4)

    k1 = k0 + perturb * torch.randn_like(k0)
    v1 = v0 + perturb * torch.randn_like(v0)
    q1 = q0 + perturb * torch.randn_like(q0)
    decay1 = torch.sigmoid(torch.logit(decay0.clamp(1e-6, 1 - 1e-6))
                           + perturb * torch.randn_like(decay0))

    # Also compute "true" converged result for output-error comparison
    _, true_boundaries, _ = admm_cuda_forward(
        S0, k1, v1, q1, decay1, H, P, init_boundaries=None, max_iters=20, tol=1e-8)

    # Measure boundary error after 1 warm iter
    S_boundary = converged_boundaries.clone()
    T_chunk = T // P
    k_chunks = k1.view(P, T_chunk, 1, H, N).permute(1, 2, 0, 3, 4).reshape(T_chunk, P, H, N).contiguous()
    v_chunks = v1.view(P, T_chunk, 1, H, N).permute(1, 2, 0, 3, 4).reshape(T_chunk, P, H, N).contiguous()
    q_chunks = q1.view(P, T_chunk, 1, H, N).permute(1, 2, 0, 3, 4).reshape(T_chunk, P, H, N).contiguous()
    dec_chunks = decay1.view(P, T_chunk, 1, H).permute(1, 2, 0, 3).reshape(T_chunk, P, H).contiguous()
    S_boundary_flat = S_boundary.reshape(P, H, N, N).contiguous()
    S_end_flat, _ = cuda_forward(k_chunks, v_chunks, q_chunks, dec_chunks, S_boundary_flat, H)
    S_end_after_1 = S_end_flat.view(1, P, H, N, N)

    # After 1 warm iter, the "result" we'd ship is S_end_after_1[:, P-1]
    # (= final state of last chunk). Compare to true converged S_end.
    # True final S: last chunk's end after full convergence.
    _, true_bd, iters_true = admm_cuda_forward(
        S0, k1, v1, q1, decay1, H, P, init_boundaries=None, max_iters=20, tol=1e-8)

    # Recompute true end states via one CUDA call with true boundaries
    S_true_flat, _ = cuda_forward(k_chunks, v_chunks, q_chunks, dec_chunks,
                                   true_bd.reshape(P, H, N, N).contiguous(), H)
    S_true_end = S_true_flat.view(1, P, H, N, N)

    # Compare output at FINAL chunk (what we'd return as forward result)
    final_warm = S_end_after_1[:, P - 1].float()
    final_true = S_true_end[:, P - 1].float()
    rel_diff = (final_warm - final_true).abs().max().item() / max(final_true.abs().max().item(), 1e-10)

    # Also boundary change during warm iter
    S_boundary_new = S_boundary.clone()
    S_boundary_new[:, 1:, :, :, :] = S_end_after_1[:, :P - 1, :, :, :]
    d_bd = (S_boundary_new - S_boundary).float().abs().max().item()

    print(f"  T={T:>6d} perturb={perturb:.4f}  "
          f"after 1 warm iter: |S_final - S_true|_rel={rel_diff:.2e}  "
          f"boundary Δ={d_bd:.2e}")
    return rel_diff, d_bd


def test_scenario(B, H, T, N, P, scenario, perturb=0.0):
    """Run two consecutive "training steps" with varying perturbation.

    scenario = 'identical': same inputs both steps (gradient accumulation)
               'new_batch': completely different inputs (standard SGD)
               'perturbed': small perturbation of inputs (BPTT-ish)
    """
    # Step 0: run to convergence with cold-start (broadcast_S0 init)
    k0, v0, q0, decay0, S0 = generate_inputs(B, H, T, N, seed=0)
    _, converged_boundaries, iters0 = admm_cuda_forward(
        S0, k0, v0, q0, decay0, H, P, init_boundaries=None, max_iters=10)

    # Step 1: prepare inputs according to scenario
    if scenario == 'identical':
        k1, v1, q1, decay1 = k0, v0, q0, decay0
    elif scenario == 'new_batch':
        k1, v1, q1, decay1, _ = generate_inputs(B, H, T, N, seed=1)  # fresh
    elif scenario == 'perturbed':
        k1 = k0 + perturb * torch.randn_like(k0)
        v1 = v0 + perturb * torch.randn_like(v0)
        q1 = q0 + perturb * torch.randn_like(q0)
        decay1 = torch.sigmoid(torch.logit(decay0.clamp(1e-6, 1 - 1e-6))
                               + perturb * torch.randn_like(decay0))
    else:
        raise ValueError(scenario)

    # Run with COLD start (reference)
    _, _, iters_cold = admm_cuda_forward(S0, k1, v1, q1, decay1, H, P, max_iters=10)

    # Run with WARM start (using step 0's converged boundaries)
    _, _, iters_warm = admm_cuda_forward(
        S0, k1, v1, q1, decay1, H, P,
        init_boundaries=converged_boundaries,
        max_iters=10)

    print(f"  scenario={scenario:>9s}  perturb={perturb:.3f}  "
          f"cold={iters_cold} iters,  warm={iters_warm} iters")
    return iters_cold, iters_warm


if __name__ == '__main__':
    print("Phase 1 — warm-start ADMM convergence at tight tolerance\n")
    for name, H, N in [("E88-n16 480M", 141, 16), ("E88-n32 480M", 83, 32)]:
        for T in [16384, 32768, 65536]:
            P = 16
            print(f"\n{name} T={T} P={P}:")
            test_scenario(1, H, T, N, P, 'identical')
            test_scenario(1, H, T, N, P, 'perturbed', perturb=0.001)
            test_scenario(1, H, T, N, P, 'perturbed', perturb=0.01)
            test_scenario(1, H, T, N, P, 'perturbed', perturb=0.1)
            test_scenario(1, H, T, N, P, 'new_batch')

    print("\n\n=== HONEST error after 1 warm iter (what the final output looks like) ===")
    print("What matters: does S_final(1_warm_iter) match S_final(fully_converged)?\n")
    for name, H, N in [("E88-n16 480M", 141, 16), ("E88-n32 480M", 83, 32)]:
        print(f"\n{name}  (bf16 ~ 8e-3 precision)")
        for T in [32768]:
            for perturb in [0.0001, 0.001, 0.01, 0.05]:
                test_tolerance_scan(1, H, T, N, P=16, perturb=perturb)
