"""Final honest e2e benchmark with all optimizations.

Assumes CUDA forward kernel returns both S_final AND S_traj (needs
kernel modification — costs 16 GB HBM at T=128K bf16). For the
benchmark we pre-compute S_traj via sequential_e88_forward and
time only the post-forward work.

Final numbers for the write-up.
"""

import sys, os, time
import torch
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase7_fused_backward import backward_e88_fused_rank1
from phase4_newton_driver import sequential_e88_forward
from elman.models.e88_fla_hybrid import E88FLAHybridCUDAFunction


def bench_cuda_fwd_only(B, H, T, n, n_repeat=5):
    g = torch.Generator(device='cuda').manual_seed(0)
    dt = torch.bfloat16
    k = (0.3 * torch.randn(T, B, H, n, generator=g, dtype=dt, device='cuda')).requires_grad_(True)
    v = (0.3 * torch.randn(T, B, H, n, generator=g, dtype=dt, device='cuda')).requires_grad_(True)
    q = (0.3 * torch.randn(T, B, H, n, generator=g, dtype=dt, device='cuda')).requires_grad_(True)
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, generator=g, dtype=dt, device='cuda')
                          ).detach().requires_grad_(True)
    S0 = 0.1 * torch.randn(B, H, n, n, generator=g, dtype=dt, device='cuda')
    def run(): return E88FLAHybridCUDAFunction.apply(True, k, v, q, decay, S0, H)
    for _ in range(3): run()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat): _ = run()
    torch.cuda.synchronize()
    return (time.time() - t0) / n_repeat * 1000


def bench_cuda_fwd_bwd(B, H, T, n, n_repeat=5):
    g = torch.Generator(device='cuda').manual_seed(0)
    dt = torch.bfloat16
    k = (0.3 * torch.randn(T, B, H, n, generator=g, dtype=dt, device='cuda')).requires_grad_(True)
    v = (0.3 * torch.randn(T, B, H, n, generator=g, dtype=dt, device='cuda')).requires_grad_(True)
    q = (0.3 * torch.randn(T, B, H, n, generator=g, dtype=dt, device='cuda')).requires_grad_(True)
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, generator=g, dtype=dt, device='cuda')
                          ).detach().requires_grad_(True)
    S0 = 0.1 * torch.randn(B, H, n, n, generator=g, dtype=dt, device='cuda')
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


def bench_hybrid_bwd_plus_dq(B, H, T, n, n_repeat=5):
    """Pararnn backward (bf16 S_traj) + dQ einsum. S_traj pre-computed."""
    g = torch.Generator(device='cuda').manual_seed(0)
    dt = torch.bfloat16
    S0 = 0.1 * torch.randn(B, H, n, n, generator=g, dtype=torch.float32, device='cuda')
    K = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=torch.float32, device='cuda')
    V = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=torch.float32, device='cuda')
    decay = 0.9 + 0.1 * torch.rand(B, H, T, generator=g, dtype=torch.float32, device='cuda')
    S_traj = sequential_e88_forward(S0, K, V, decay).to(dt)
    del S0, K, V, decay  # free fp32 originals
    K_d = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    V_d = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    decay_d = (0.9 + 0.1 * torch.rand(B, H, T, generator=g, dtype=dt, device='cuda'))
    g_T = 0.01 * torch.randn(B, H, n, n, dtype=dt, device='cuda')
    dL_dout = 0.01 * torch.randn(B, H, T, n, dtype=dt, device='cuda')
    q = 0.3 * torch.randn(B, H, T, n, dtype=dt, device='cuda')

    def run():
        dS0_, dK_, dV_, ddec_ = backward_e88_fused_rank1(
            S_traj, K_d, V_d, decay_d, g_T, dL_dout, q,
            num_warps=4 if n == 32 else 1, num_stages=1)
        dQ_ = torch.einsum('bhti,bhtij->bhtj', dL_dout, S_traj[:, :, 1:])
        return dS0_, dK_, dV_, ddec_, dQ_

    for _ in range(3): run()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat): _ = run()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1024**3
    return (time.time() - t0) / n_repeat * 1000, peak


if __name__ == '__main__':
    print(f"{'config':>22s}  {'CUDA f+b':>9s}  {'CUDA fwd':>9s}  "
          f"{'Par bwd+dQ':>11s}  {'Hybrid tot':>11s}  {'e2e spd':>8s}")
    for name, H, n in [("E88-n32 480M", 83, 32),
                       ("E88-n16 480M", 141, 16),
                       ("Small H=32",    32, 32)]:
        for T in [8192, 32768, 65536, 131072]:
            config = f"{name} T={T}"
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                cuda_fb = bench_cuda_fwd_bwd(1, H, T, n)
                torch.cuda.empty_cache()
                cuda_fwd = bench_cuda_fwd_only(1, H, T, n)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                par_ms, peak = bench_hybrid_bwd_plus_dq(1, H, T, n)
                hybrid = cuda_fwd + par_ms
                spd = cuda_fb / hybrid
                print(f"{config:>22s}  {cuda_fb:>6.1f} ms  {cuda_fwd:>6.1f} ms  "
                      f"{par_ms:>8.1f} ms  {hybrid:>8.1f} ms  {spd:>6.2f}×  "
                      f"[peak {peak:.1f}GB]")
            except Exception as e:
                print(f"{config:>22s}  FAIL: {str(e)[:80]}")
                torch.cuda.empty_cache()
