"""Phase 2 full e2e benchmark — coarse-warmup + 1-iter ADMM + Pararnn backward."""

import sys, os, time
import torch

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase4_newton_driver import sequential_e88_forward
from phase7_fused_backward import backward_e88_fused_rank1
from elman.models.e88_fla_hybrid import E88FLAHybridCUDAFunction
from phase1_warmstart_bench import admm_forward_fixed_iters
from phase2_warmup_scan import warmup_scan_boundaries


def bench_cuda_fb(B, H, T, N, n_repeat=3):
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


def bench_hybrid(B, H, T, N, P, W, mode, n_repeat=3):
    """mode = 'cold_2iter' or 'warmup_1iter'"""
    dt = torch.bfloat16
    g = torch.Generator(device='cuda').manual_seed(0)
    k = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')).requires_grad_(False)
    v = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')).requires_grad_(False)
    q = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')).requires_grad_(False)
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, generator=g, dtype=dt, device='cuda'))
    S0 = 0.1 * torch.randn(B, H, N, N, generator=g, dtype=dt, device='cuda')

    # Pre-compute S_traj for Pararnn bwd (per honest e2e benchmark convention)
    K_bhhtn = k.permute(1, 2, 0, 3).contiguous().float()
    V_bhhtn = v.permute(1, 2, 0, 3).contiguous().float()
    decay_bhht = decay.permute(1, 2, 0).contiguous().float()
    S_traj = sequential_e88_forward(S0.float(), K_bhhtn, V_bhhtn, decay_bhht).to(dt)
    torch.cuda.empty_cache()

    K_d = k.permute(1, 2, 0, 3).contiguous()
    V_d = v.permute(1, 2, 0, 3).contiguous()
    decay_d = decay.permute(1, 2, 0).contiguous()
    g_T = 0.01 * torch.randn(B, H, N, N, dtype=dt, device='cuda')
    dL_dout = 0.01 * torch.randn(B, H, T, N, dtype=dt, device='cuda')
    q_d = q.permute(1, 2, 0, 3).contiguous()

    def run():
        if mode == 'cold_2iter':
            _, _, _ = admm_forward_fixed_iters(S0, k, v, q, decay, H, P,
                                                  num_iters=2, init_boundaries=None)
        elif mode == 'warmup_1iter':
            bd = warmup_scan_boundaries(S0, k, v, q, decay, H, P, W=W)
            _, _, _ = admm_forward_fixed_iters(S0, k, v, q, decay, H, P,
                                                  num_iters=1, init_boundaries=bd)
        dS0, dK, dV, ddec = backward_e88_fused_rank1(
            S_traj, K_d, V_d, decay_d, g_T, dL_dout, q_d,
            num_warps=4 if N == 32 else 1, num_stages=1)
        dQ = torch.einsum('bhti,bhtij->bhtj', dL_dout, S_traj[:, :, 1:])

    for _ in range(3): run()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat): run()
    torch.cuda.synchronize()
    return (time.time() - t0) / n_repeat * 1000


if __name__ == '__main__':
    print("Phase 2 e2e: coarse-warmup + 1-iter ADMM + Pararnn backward\n")
    print(f"{'Config':>22s}  {'T':>6s}  {'CUDA f+b':>10s}  "
          f"{'P2 cold-2iter':>14s}  {'P2 warmup-1iter':>16s}  "
          f"{'spd cold':>9s}  {'spd warmup':>11s}")
    for name, H, N in [("E88-n16 480M", 141, 16), ("E88-n32 480M", 83, 32)]:
        for T in [16384, 32768, 65536]:
            try:
                torch.cuda.empty_cache()
                cuda_fb = bench_cuda_fb(1, H, T, N)
                torch.cuda.empty_cache()
                cold = bench_hybrid(1, H, T, N, P=16, W=128, mode='cold_2iter')
                torch.cuda.empty_cache()
                warm = bench_hybrid(1, H, T, N, P=16, W=128, mode='warmup_1iter')
                s_cold = cuda_fb / cold
                s_warm = cuda_fb / warm
                shape = f"{name} T={T}"
                print(f"{shape:>22s}  {T:>6d}  {cuda_fb:>7.1f} ms  "
                      f"{cold:>11.1f} ms  {warm:>13.1f} ms  "
                      f"{s_cold:>6.2f}×  {s_warm:>8.2f}×")
            except Exception as e:
                print(f"  FAIL {name} T={T}: {str(e)[:80]}")
