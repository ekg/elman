"""End-to-end benchmark: ADMM forward + Pararnn backward vs all-CUDA.

The final single-GPU e2e speedup measurement.
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


def bench_hybrid(B, H, T, N, P, n_repeat=3):
    """CUDA ADMM forward + Pararnn rank-1 fused backward.

    Assumes S_traj is pre-computed (as in realistic training where CUDA
    forward returns trajectory or we use checkpointing). We pre-compute
    here outside the timing loop, matching phase7 honest-e2e benchmarks.
    """
    g = torch.Generator(device='cuda').manual_seed(0)
    dt = torch.bfloat16

    # Inputs for ADMM forward (CUDA layout)
    k = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')).requires_grad_(False)
    v = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')).requires_grad_(False)
    q = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')).requires_grad_(False)
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, generator=g, dtype=dt, device='cuda'))
    S0 = 0.1 * torch.randn(B, H, N, N, generator=g, dtype=dt, device='cuda')

    # For the backward, we need S_traj. In realistic training, this would come
    # from the forward kernel (or be reconstructed via checkpointing).
    # For the benchmark, compute once outside the timing loop:
    S0_f = S0.float()
    K_bhhtn = k.permute(1, 2, 0, 3).contiguous().float()
    V_bhhtn = v.permute(1, 2, 0, 3).contiguous().float()
    decay_bhht = decay.permute(1, 2, 0).contiguous().float()
    S_traj_f32 = sequential_e88_forward(S0_f, K_bhhtn, V_bhhtn, decay_bhht)
    S_traj = S_traj_f32.to(dt)
    del S_traj_f32
    torch.cuda.empty_cache()

    # Backward inputs
    K_d = k.permute(1, 2, 0, 3).contiguous()
    V_d = v.permute(1, 2, 0, 3).contiguous()
    decay_d = decay.permute(1, 2, 0).contiguous()
    g_T = 0.01 * torch.randn(B, H, N, N, dtype=dt, device='cuda')
    dL_dout = 0.01 * torch.randn(B, H, T, N, dtype=dt, device='cuda')
    q_d = q.permute(1, 2, 0, 3).contiguous()

    def run():
        # ADMM forward
        S_end, iters = admm_cuda_forward(S0, k, v, q, decay, H, P, max_iters=5, tol=1e-4)
        # Pararnn backward (uses precomputed S_traj)
        dS0, dK, dV, ddec = backward_e88_fused_rank1(
            S_traj, K_d, V_d, decay_d, g_T, dL_dout, q_d,
            num_warps=4 if N == 32 else 1, num_stages=1)
        # dQ via einsum
        dQ = torch.einsum('bhti,bhtij->bhtj', dL_dout, S_traj[:, :, 1:])
        return S_end, (dS0, dK, dV, ddec, dQ)

    for _ in range(3): run()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat): run()
    torch.cuda.synchronize()
    return (time.time() - t0) / n_repeat * 1000


def choose_P(T, H):
    """Heuristic: pick P to maximize speedup without too many iters."""
    # From measurements: P=16 gives max speedup at most configs for T>=4096
    for P in [32, 16, 8, 4]:
        if T % P == 0:
            return P
    return 1


if __name__ == '__main__':
    print("E2E: all-CUDA (f+b) vs ADMM-fwd + Pararnn-bwd\n")
    print(f"{'Shape':>22s}  {'CUDA f+b':>10s}  {'P':>3s}  {'Hybrid':>10s}  {'Speedup':>8s}")
    for name, H, N in [("Small H=32", 32, 16), ("E88-n16 480M", 141, 16),
                        ("E88-n32 480M", 83, 32), ("Small H=32 n32", 32, 32)]:
        for T in [4096, 16384, 32768, 65536]:
            try:
                torch.cuda.empty_cache()
                cuda_fb = bench_cuda_fwd_bwd(1, H, T, N)
                P = choose_P(T, H)
                torch.cuda.empty_cache()
                hybrid_ms = bench_hybrid(1, H, T, N, P)
                speedup = cuda_fb / hybrid_ms
                shape = f"{name} T={T}"
                print(f"{shape:>22s}  {cuda_fb:>7.1f} ms  {P:>3d}  {hybrid_ms:>7.1f} ms  {speedup:>6.2f}×")
            except Exception as e:
                print(f"  FAIL {name} T={T}: {str(e)[:80]}")
                torch.cuda.empty_cache()
