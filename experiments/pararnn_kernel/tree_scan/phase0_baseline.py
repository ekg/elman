"""Phase 0 — lock in baseline measurements at full production T range.

Produces the single table of numbers we're trying to beat in subsequent
phases.

Measures:
  (Seq f+b)   All-CUDA forward + backward — the pure baseline
  (Hybrid)    ADMM CUDA forward + Pararnn rank-1 fused backward — current best
  (Fwd only)  ADMM forward alone — for isolating where the gains come from
  (Bwd only)  Pararnn backward alone — same isolation
"""

import sys, os, time
import torch

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase4_newton_driver import sequential_e88_forward
from phase7_fused_backward import backward_e88_fused_rank1
from elman.models.e88_fla_hybrid import E88FLAHybridCUDAFunction
from admm_cuda_bench import admm_cuda_forward, cuda_forward


def bench_cuda_fwd_only(B, H, T, N, n_repeat=3):
    g = torch.Generator(device='cuda').manual_seed(0)
    dt = torch.bfloat16
    k = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')).requires_grad_(False)
    v = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')).requires_grad_(False)
    q = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')).requires_grad_(False)
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, generator=g, dtype=dt, device='cuda'))
    S0 = 0.1 * torch.randn(B, H, N, N, generator=g, dtype=dt, device='cuda')
    def run(): return cuda_forward(k, v, q, decay, S0, H)
    for _ in range(3): run()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat): _ = run()
    torch.cuda.synchronize()
    return (time.time() - t0) / n_repeat * 1000


def bench_cuda_fwd_bwd(B, H, T, N, n_repeat=3):
    g = torch.Generator(device='cuda').manual_seed(0)
    dt = torch.bfloat16
    k = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')).requires_grad_(True)
    v = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')).requires_grad_(True)
    q = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')).requires_grad_(True)
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, generator=g, dtype=dt, device='cuda')
                          ).detach().requires_grad_(True)
    S0 = 0.1 * torch.randn(B, H, N, N, generator=g, dtype=dt, device='cuda')
    def run():
        S_final, output = E88FLAHybridCUDAFunction.apply(True, k, v, q, decay, S0, H)
        loss = output.sum() + S_final.pow(2).sum() * 1e-4
        loss.backward()
        k.grad = None; v.grad = None; q.grad = None; decay.grad = None
    for _ in range(3): run()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat): run()
    torch.cuda.synchronize()
    return (time.time() - t0) / n_repeat * 1000


def bench_admm_fwd(B, H, T, N, P, n_repeat=3):
    g = torch.Generator(device='cuda').manual_seed(0)
    dt = torch.bfloat16
    k = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')).requires_grad_(False)
    v = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')).requires_grad_(False)
    q = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')).requires_grad_(False)
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, generator=g, dtype=dt, device='cuda'))
    S0 = 0.1 * torch.randn(B, H, N, N, generator=g, dtype=dt, device='cuda')

    for _ in range(3): _ = admm_cuda_forward(S0, k, v, q, decay, H, P, max_iters=5, tol=1e-4)
    torch.cuda.synchronize()
    t0 = time.time()
    iters_used = 0
    for _ in range(n_repeat):
        _, iters_used = admm_cuda_forward(S0, k, v, q, decay, H, P, max_iters=5, tol=1e-4)
    torch.cuda.synchronize()
    return (time.time() - t0) / n_repeat * 1000, iters_used


def bench_pararnn_bwd(B, H, T, N, n_repeat=3):
    """Pararnn rank-1 fused backward. S_traj precomputed."""
    g = torch.Generator(device='cuda').manual_seed(0)
    dt = torch.bfloat16

    S0_f = 0.1 * torch.randn(B, H, N, N, generator=g, dtype=torch.float32, device='cuda')
    K_f = 0.3 * torch.randn(B, H, T, N, generator=g, dtype=torch.float32, device='cuda')
    V_f = 0.3 * torch.randn(B, H, T, N, generator=g, dtype=torch.float32, device='cuda')
    decay_f = torch.sigmoid(0.5 + 0.1 * torch.randn(B, H, T, generator=g, dtype=torch.float32, device='cuda'))
    S_traj = sequential_e88_forward(S0_f, K_f, V_f, decay_f).to(dt)
    del S0_f, K_f, V_f, decay_f
    torch.cuda.empty_cache()

    K_d = 0.3 * torch.randn(B, H, T, N, generator=g, dtype=dt, device='cuda')
    V_d = 0.3 * torch.randn(B, H, T, N, generator=g, dtype=dt, device='cuda')
    decay_d = torch.sigmoid(0.5 + 0.1 * torch.randn(B, H, T, generator=g, dtype=dt, device='cuda'))
    g_T = 0.01 * torch.randn(B, H, N, N, dtype=dt, device='cuda')
    dL_dout = 0.01 * torch.randn(B, H, T, N, dtype=dt, device='cuda')
    q_d = 0.3 * torch.randn(B, H, T, N, dtype=dt, device='cuda')

    def run():
        d = backward_e88_fused_rank1(S_traj, K_d, V_d, decay_d, g_T, dL_dout, q_d,
                                       num_warps=4 if N == 32 else 1, num_stages=1)
        dQ = torch.einsum('bhti,bhtij->bhtj', dL_dout, S_traj[:, :, 1:])
        return d, dQ

    for _ in range(3): run()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat): _ = run()
    torch.cuda.synchronize()
    return (time.time() - t0) / n_repeat * 1000


if __name__ == '__main__':
    configs = [
        ("E88-n16 480M", 141, 16),
        ("E88-n32 480M", 83, 32),
    ]
    Ts = [4096, 16384, 32768, 65536]
    # Note: T=128K often OOMs the sequential_e88_forward; we still time the
    # forward-only component where possible.

    print("Phase 0 — Production baseline\n")
    print(f"{'Config':>18s}  {'T':>6s}  {'C.fwd':>8s}  {'C.bwd*':>8s}  "
          f"{'C.f+b':>8s}  {'P':>3s}  {'ADMM.fwd':>9s}  "
          f"{'Par.bwd':>8s}  {'Hybrid':>8s}  {'Speedup':>8s}")
    print("-" * 110)

    results = []
    for name, H, N in configs:
        for T in Ts:
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                # CUDA fwd only
                cuda_fwd = bench_cuda_fwd_only(1, H, T, N)
                torch.cuda.empty_cache()
                # CUDA f+b
                cuda_fb = bench_cuda_fwd_bwd(1, H, T, N)
                cuda_bwd_est = cuda_fb - cuda_fwd
                torch.cuda.empty_cache()

                # Pick P per our earlier measurements
                P = 16 if T >= 2048 else 8
                # ADMM forward
                admm_fwd, iters = bench_admm_fwd(1, H, T, N, P)
                torch.cuda.empty_cache()
                # Pararnn backward
                par_bwd = bench_pararnn_bwd(1, H, T, N)
                torch.cuda.empty_cache()

                hybrid = admm_fwd + par_bwd
                speedup = cuda_fb / hybrid

                results.append((name, T, cuda_fwd, cuda_bwd_est, cuda_fb, P,
                                admm_fwd, iters, par_bwd, hybrid, speedup))
                print(f"{name:>18s}  {T:>6d}  {cuda_fwd:>6.1f}ms  {cuda_bwd_est:>6.1f}ms  "
                      f"{cuda_fb:>6.1f}ms  {P:>3d}  {admm_fwd:>6.1f}ms  "
                      f"{par_bwd:>6.1f}ms  {hybrid:>6.1f}ms  {speedup:>6.2f}×")
            except Exception as e:
                print(f"  FAIL {name} T={T}: {str(e)[:100]}")
                torch.cuda.empty_cache()

    # Write markdown table
    lines = ["# Phase 0 Baseline", "",
             "Honest e2e measurements with dQ included. All at B=1, bf16.",
             "",
             "| Config | T | CUDA fwd | CUDA bwd* | CUDA f+b | P | ADMM fwd | Iters | Par bwd | Hybrid | Speedup |",
             "|--------|---|----------|-----------|----------|---|----------|-------|---------|--------|---------|"]
    for r in results:
        name, T, cfw, cbw, cfb, P, afw, iters, pbw, hy, spd = r
        lines.append(f"| {name} | {T} | {cfw:.1f} | {cbw:.1f} | {cfb:.1f} | {P} | "
                     f"{afw:.1f} | {iters} | {pbw:.1f} | {hy:.1f} | **{spd:.2f}×** |")
    lines.extend(["",
                  "* CUDA bwd estimated as (CUDA f+b) - (CUDA fwd).",
                  "",
                  "## Targets for subsequent phases",
                  "",
                  "Phase 1 (warm-start ADMM): cut ADMM iters 2 → 1. Expected:",
                  "  - E88-n16 480M T=65K: ~ hybrid drops by ~half of ADMM fwd time",
                  "  - E88-n32 480M T=65K: same proportional reduction",
                  "",
                  "Phase 3 (faster backward): target ~1.5× speedup on Par bwd column."])

    doc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', '..', 'docs', 'FIVEX_BASELINE.md')
    with open(doc_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\nBaseline written to {doc_path}")
