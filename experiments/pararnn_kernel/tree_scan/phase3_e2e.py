"""Phase 3 e2e: does the CUDA backward translate to a meaningful
end-to-end speedup when plugged into the Phase 2 pipeline?
"""

import sys, os, time
import torch
from torch.utils.cpp_extension import load

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase4_newton_driver import sequential_e88_forward
from phase7_fused_backward import backward_e88_fused_rank1
from elman.models.e88_fla_hybrid import E88FLAHybridCUDAFunction
from phase1_warmstart_bench import admm_forward_fixed_iters
from phase2_warmup_scan import warmup_scan_boundaries


_CUDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cuda_fused_backward.cu')
ext = load(name='cuda_bwd_proto', sources=[_CUDA],
           extra_cuda_cflags=['-O3', '-std=c++17', '--use_fast_math',
                               '-gencode=arch=compute_80,code=sm_80'],
           verbose=False)


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


def bench_hybrid(B, H, T, N, P, W, bwd_kernel, n_repeat=3):
    """bwd_kernel = 'triton' or 'cuda'"""
    dt = torch.bfloat16
    g = torch.Generator(device='cuda').manual_seed(0)
    k = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')).requires_grad_(False)
    v = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')).requires_grad_(False)
    q = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')).requires_grad_(False)
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, generator=g, dtype=dt, device='cuda'))
    S0 = 0.1 * torch.randn(B, H, N, N, generator=g, dtype=dt, device='cuda')

    # Pre-compute S_traj (shared by both bwd kernels for the benchmark)
    K_bhhtn = k.permute(1, 2, 0, 3).contiguous().float()
    V_bhhtn = v.permute(1, 2, 0, 3).contiguous().float()
    decay_bhht = decay.permute(1, 2, 0).contiguous().float()
    S_traj = sequential_e88_forward(S0.float(), K_bhhtn, V_bhhtn, decay_bhht).to(dt)
    del K_bhhtn, V_bhhtn, decay_bhht
    torch.cuda.empty_cache()

    K_d = k.permute(1, 2, 0, 3).contiguous()
    V_d = v.permute(1, 2, 0, 3).contiguous()
    decay_d = decay.permute(1, 2, 0).contiguous()
    g_T = 0.01 * torch.randn(B, H, N, N, dtype=dt, device='cuda')
    dL_dout = 0.01 * torch.randn(B, H, T, N, dtype=dt, device='cuda')
    q_d = q.permute(1, 2, 0, 3).contiguous()

    def run():
        # Warmup coarse solver + 1-iter ADMM
        bd = warmup_scan_boundaries(S0, k, v, q, decay, H, P, W=W)
        _, _, _ = admm_forward_fixed_iters(S0, k, v, q, decay, H, P,
                                              num_iters=1, init_boundaries=bd)
        # Backward
        if bwd_kernel == 'triton':
            dS0, dK, dV, ddec = backward_e88_fused_rank1(
                S_traj, K_d, V_d, decay_d, g_T, dL_dout, q_d,
                num_warps=4 if N == 32 else 1, num_stages=1)
        elif bwd_kernel == 'cuda':
            dS0, dK, dV, ddec = ext.cuda_fused_backward_cpasync(
                S_traj, K_d, V_d, decay_d, g_T, dL_dout, q_d)
        # dQ
        dQ = torch.einsum('bhti,bhtij->bhtj', dL_dout, S_traj[:, :, 1:])

    for _ in range(3): run()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat): run()
    torch.cuda.synchronize()
    return (time.time() - t0) / n_repeat * 1000


if __name__ == '__main__':
    print("Phase 3 e2e: CUDA backward integrated\n")
    print(f"{'Config':>22s}  {'T':>6s}  {'CUDA f+b':>10s}  "
          f"{'Hyb(triton)':>13s}  {'Hyb(cuda)':>11s}  "
          f"{'spd-tri':>8s}  {'spd-cu':>8s}")
    for name, H, N in [("E88-n16 480M", 141, 16), ("E88-n32 480M", 83, 32)]:
        for T in [16384, 32768, 65536]:
            try:
                torch.cuda.empty_cache()
                cuda_fb = bench_cuda_fb(1, H, T, N)
                torch.cuda.empty_cache()
                tri = bench_hybrid(1, H, T, N, P=16, W=128, bwd_kernel='triton')
                torch.cuda.empty_cache()
                cu = bench_hybrid(1, H, T, N, P=16, W=128, bwd_kernel='cuda')
                s_t = cuda_fb / tri
                s_c = cuda_fb / cu
                print(f"{name:>22s}  {T:>6d}  {cuda_fb:>7.1f} ms  "
                      f"{tri:>10.1f} ms  {cu:>8.1f} ms  "
                      f"{s_t:>6.2f}×  {s_c:>6.2f}×")
            except Exception as e:
                print(f"  FAIL {name} T={T}: {str(e)[:80]}")
                torch.cuda.empty_cache()
