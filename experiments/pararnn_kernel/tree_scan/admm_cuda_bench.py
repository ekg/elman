"""ADMM benchmark using CUDA E88 kernel for each chunk.

Compare:
  - Pure sequential: one CUDA forward over full T
  - ADMM: P chunks in parallel, K outer iters
"""

import sys, os, time
import torch

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase0_pytorch_ref import _random_case
from phase4_newton_driver import sequential_e88_forward
from elman.models.e88_fla_hybrid import E88FLAHybridCUDAFunction


def cuda_forward(k, v, q, decay, S0, H):
    """Wrapper around CUDA E88 forward. k,v,q: [T, B, H, N], decay: [T, B, H], S0: [B, H, N, N]"""
    S_final, output = E88FLAHybridCUDAFunction.apply(True, k, v, q, decay, S0, H)
    return S_final, output


def admm_cuda_forward(S0, k, v, q, decay, H, P, max_iters=5, tol=1e-4):
    """Parallel-in-time ADMM using CUDA E88 forward on each chunk.

    Inputs in CUDA layout:
      S0: [B, H, N, N]
      k, v, q: [T, B, H, N]
      decay: [T, B, H]
    """
    T = k.shape[0]
    B = k.shape[1]
    N = k.shape[3]
    assert T % P == 0
    T_chunk = T // P

    # Reshape inputs into chunks. Put P as a batch dim.
    # k: [T, B, H, N] → [P, T_chunk, B, H, N] → [T_chunk, B*P, H, N] via permute+reshape
    k_chunks = k.view(P, T_chunk, B, H, N).permute(1, 2, 0, 3, 4).reshape(T_chunk, B * P, H, N).contiguous()
    v_chunks = v.view(P, T_chunk, B, H, N).permute(1, 2, 0, 3, 4).reshape(T_chunk, B * P, H, N).contiguous()
    q_chunks = q.view(P, T_chunk, B, H, N).permute(1, 2, 0, 3, 4).reshape(T_chunk, B * P, H, N).contiguous()
    decay_chunks = decay.view(P, T_chunk, B, H).permute(1, 2, 0, 3).reshape(T_chunk, B * P, H).contiguous()

    # Boundaries: S_boundary has shape [B, P, H, N, N]. Flatten to [B*P, H, N, N] for CUDA.
    # S_boundary[b, 0, h] = S0[b, h] (exact). Others: copies of S0 (initial guess).
    S_boundary = S0.unsqueeze(1).expand(B, P, H, N, N).contiguous()  # [B, P, H, N, N]
    S_boundary_flat = S_boundary.reshape(B * P, H, N, N).contiguous()

    for it in range(max_iters):
        # Run CUDA forward on all P*B chains in parallel
        S_end_flat, _ = cuda_forward(k_chunks, v_chunks, q_chunks, decay_chunks, S_boundary_flat, H)
        S_end = S_end_flat.view(B, P, H, N, N)  # [B, P, H, N, N]

        # Update boundaries: S_boundary[b, p+1] = S_end[b, p] for p < P-1
        S_boundary_new = S_boundary.clone()
        S_boundary_new[:, 1:, :, :, :] = S_end[:, :P - 1, :, :, :]

        d_max = (S_boundary_new - S_boundary).abs().max().item()
        S_boundary = S_boundary_new
        S_boundary_flat = S_boundary.reshape(B * P, H, N, N).contiguous()

        if d_max < tol:
            break

    # Return final states (and could return full trajectory too, but just time it)
    return S_end, it + 1


def bench_sequential(B, H, T, N, n_repeat=3):
    g = torch.Generator(device='cuda').manual_seed(0)
    dt = torch.bfloat16
    k = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')).requires_grad_(False)
    v = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')).requires_grad_(False)
    q = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')).requires_grad_(False)
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, generator=g, dtype=dt, device='cuda'))
    S0 = 0.1 * torch.randn(B, H, N, N, generator=g, dtype=dt, device='cuda')

    for _ in range(3): _ = cuda_forward(k, v, q, decay, S0, H)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat): _ = cuda_forward(k, v, q, decay, S0, H)
    torch.cuda.synchronize()
    return (time.time() - t0) / n_repeat * 1000


def bench_admm(B, H, T, N, P, max_iters=5, n_repeat=3):
    g = torch.Generator(device='cuda').manual_seed(0)
    dt = torch.bfloat16
    k = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')).requires_grad_(False)
    v = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')).requires_grad_(False)
    q = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')).requires_grad_(False)
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, generator=g, dtype=dt, device='cuda'))
    S0 = 0.1 * torch.randn(B, H, N, N, generator=g, dtype=dt, device='cuda')

    # Warmup
    for _ in range(3):
        _, _ = admm_cuda_forward(S0, k, v, q, decay, H, P, max_iters=max_iters)
    torch.cuda.synchronize()
    t0 = time.time()
    iters_used = 0
    for _ in range(n_repeat):
        _, iters_used = admm_cuda_forward(S0, k, v, q, decay, H, P, max_iters=max_iters)
    torch.cuda.synchronize()
    return (time.time() - t0) / n_repeat * 1000, iters_used


def test_correctness(B, H, T, N, P):
    """Compare ADMM-CUDA output to pure sequential CUDA forward."""
    g = torch.Generator(device='cuda').manual_seed(0)
    dt = torch.bfloat16
    k = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')).requires_grad_(False)
    v = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')).requires_grad_(False)
    q = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')).requires_grad_(False)
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, generator=g, dtype=dt, device='cuda'))
    S0 = 0.1 * torch.randn(B, H, N, N, generator=g, dtype=dt, device='cuda')

    # Ground truth: sequential CUDA forward
    S_final_seq, _ = cuda_forward(k, v, q, decay, S0, H)
    # S_final_seq shape: [B, H, N, N]

    # ADMM: returns S at end of LAST chunk = global final state
    S_end_admm, iters = admm_cuda_forward(S0, k, v, q, decay, H, P, max_iters=10, tol=1e-4)
    # S_end_admm shape: [B, P, H, N, N] — last chunk's end = S_final
    S_final_admm = S_end_admm[:, P - 1]  # [B, H, N, N]

    diff = (S_final_admm.float() - S_final_seq.float()).abs().max().item()
    rel_diff = diff / max(S_final_seq.float().abs().max().item(), 1e-10)
    tol = 1e-2  # bf16 tolerance
    status = "PASS" if rel_diff < tol else "FAIL"
    print(f"  B={B} H={H} T={T} N={N} P={P}  iters={iters}  "
          f"max|admm-seq|_rel={rel_diff:.2e}  [{status}]")


if __name__ == '__main__':
    print("ADMM CUDA correctness check (bf16)\n")
    for B, H, T, N, P in [(1, 32, 1024, 16, 8),
                           (1, 32, 8192, 16, 16),
                           (1, 141, 8192, 16, 16),
                           (1, 83, 16384, 32, 16)]:
        try:
            test_correctness(B, H, T, N, P)
        except Exception as e:
            print(f"  FAIL: {str(e)[:100]}")

    print()
    # Production E88-n16 config sweep
    print("ADMM vs sequential CUDA forward (bf16)\n")
    print(f"{'Shape':>20s}  {'Seq':>8s}  {'ADMM(P=4)':>10s}  {'ADMM(P=8)':>10s}  "
          f"{'ADMM(P=16)':>11s}  {'ADMM(P=32)':>11s}")
    for H, N in [(32, 16), (141, 16), (32, 32), (83, 32)]:
        for T in [4096, 16384, 32768, 65536]:
            seq_ms = bench_sequential(1, H, T, N)
            results = []
            for P in [4, 8, 16, 32]:
                if T % P != 0:
                    results.append("-")
                    continue
                try:
                    admm_ms, iters = bench_admm(1, H, T, N, P)
                    speedup = seq_ms / admm_ms
                    results.append(f"{admm_ms:.1f}ms/{iters}it/{speedup:.2f}×")
                except Exception as e:
                    results.append(f"FAIL:{str(e)[:20]}")
            shape = f"H={H} T={T} N={N}"
            print(f"{shape:>20s}  {seq_ms:>6.1f} ms   " +
                  "  ".join(f"{r:>10s}" for r in results))
