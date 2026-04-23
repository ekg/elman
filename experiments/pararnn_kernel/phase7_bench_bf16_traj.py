"""Check bf16 S_traj quality + test at T=128K with production H.

The fused backward kernel casts bf16 loads to fp32 for compute. If we
pass bf16 inputs, we halve S_traj memory (42 GB → 21 GB at T=128K H=83).

Precision loss: ~2^-8 per forward step in bf16 storage. Over T=128K
steps the forward trajectory accumulates error, but since it's stored
(not iterated), accumulation happens once. Gradients derived from bf16
forward are ~bf16-accurate — typical for mixed-precision training.
"""

import sys, os, time
import torch
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase7_fused_backward import backward_e88_fused_rank1
from phase4_newton_driver import sequential_e88_forward
from elman.models.e88_fla_hybrid import E88FLAHybridCUDAFunction


def test_bf16_gradient_quality(B, H, T, n):
    """Compare gradients from bf16 vs fp32 S_traj."""
    g = torch.Generator(device='cuda').manual_seed(0)

    S0_f32 = 0.1 * torch.randn(B, H, n, n, generator=g, dtype=torch.float32, device='cuda')
    K_f32 = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=torch.float32, device='cuda')
    V_f32 = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=torch.float32, device='cuda')
    decay_f32 = 0.9 + 0.1 * torch.rand(B, H, T, generator=g, dtype=torch.float32, device='cuda')

    g_T_f32 = 0.1 * torch.randn(B, H, n, n, dtype=torch.float32, device='cuda')
    dL_dout_f32 = 0.1 * torch.randn(B, H, T, n, dtype=torch.float32, device='cuda')
    q_f32 = 0.3 * torch.randn(B, H, T, n, dtype=torch.float32, device='cuda')

    # fp32 reference
    S_traj_f32 = sequential_e88_forward(S0_f32, K_f32, V_f32, decay_f32)
    dS0_f, dK_f, dV_f, ddec_f = backward_e88_fused_rank1(
        S_traj_f32, K_f32, V_f32, decay_f32, g_T_f32, dL_dout_f32, q_f32)

    # bf16 S_traj, fp32 rest
    S_traj_bf16 = S_traj_f32.to(torch.bfloat16)
    K_bf16 = K_f32.to(torch.bfloat16)
    V_bf16 = V_f32.to(torch.bfloat16)
    decay_bf16 = decay_f32.to(torch.bfloat16)
    g_T_bf16 = g_T_f32.to(torch.bfloat16)
    dL_dout_bf16 = dL_dout_f32.to(torch.bfloat16)
    q_bf16 = q_f32.to(torch.bfloat16)

    dS0_b, dK_b, dV_b, ddec_b = backward_e88_fused_rank1(
        S_traj_bf16, K_bf16, V_bf16, decay_bf16, g_T_bf16, dL_dout_bf16, q_bf16)

    def rel_err(a, b):
        return (a.float() - b.float()).abs().max().item() / max(b.float().abs().max().item(), 1e-30)

    eS0 = rel_err(dS0_b, dS0_f)
    eK = rel_err(dK_b, dK_f)
    eV = rel_err(dV_b, dV_f)
    ed = rel_err(ddec_b, ddec_f)
    worst = max(eS0, eK, eV, ed)
    # Typical bf16 precision: 2^-7 ≈ 8e-3
    status = "OK" if worst < 2e-2 else "CONCERN"
    print(f"  B={B} H={H:3d} T={T:5d} n={n}  "
          f"bf16-vs-fp32 grad rel err: S0={eS0:.1e} K={eK:.1e} V={eV:.1e} dec={ed:.1e}  [{status}]")


def bench_bf16_bwd(B, H, T, n, n_repeat=3, num_warps=4, num_stages=1):
    g = torch.Generator(device='cuda').manual_seed(0)
    dt = torch.bfloat16
    S0 = 0.1 * torch.randn(B, H, n, n, generator=g, dtype=dt, device='cuda')
    K = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    V = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    decay = 0.9 + 0.1 * torch.rand(B, H, T, generator=g, dtype=dt, device='cuda')

    # Generate bf16 forward — but sequential_e88_forward in bf16 may have issues.
    # Safer: forward in fp32, cast down.
    S0_f = S0.float(); K_f = K.float(); V_f = V.float(); decay_f = decay.float()
    S_traj_f32 = sequential_e88_forward(S0_f, K_f, V_f, decay_f)
    S_traj = S_traj_f32.to(dt)
    del S_traj_f32; torch.cuda.empty_cache()

    g_T = 0.01 * torch.randn(B, H, n, n, dtype=dt, device='cuda')
    dL_dout = 0.01 * torch.randn(B, H, T, n, dtype=dt, device='cuda')
    q = 0.3 * torch.randn(B, H, T, n, dtype=dt, device='cuda')

    def run():
        return backward_e88_fused_rank1(S_traj, K, V, decay, g_T, dL_dout, q,
                                         num_warps=num_warps, num_stages=num_stages)
    for _ in range(3): run()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat): _ = run()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1024**3
    return (time.time() - t0) / n_repeat * 1000, peak


def bench_cuda_fwd_bwd(B, H, T, n, n_repeat=3):
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
    peak = torch.cuda.max_memory_allocated() / 1024**3
    return (time.time() - t0) / n_repeat * 1000, peak


def bench_cuda_fwd(B, H, T, n, n_repeat=3):
    g = torch.Generator(device='cuda').manual_seed(0)
    dt = torch.bfloat16
    k = (0.3 * torch.randn(T, B, H, n, generator=g, dtype=dt, device='cuda')).requires_grad_(True)
    v = (0.3 * torch.randn(T, B, H, n, generator=g, dtype=dt, device='cuda')).requires_grad_(True)
    q = (0.3 * torch.randn(T, B, H, n, generator=g, dtype=dt, device='cuda')).requires_grad_(True)
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, generator=g, dtype=dt, device='cuda')
                          ).detach().requires_grad_(True)
    S0 = 0.1 * torch.randn(B, H, n, n, generator=g, dtype=dt, device='cuda')
    def run():
        return E88FLAHybridCUDAFunction.apply(True, k, v, q, decay, S0, H)
    for _ in range(3): run()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat): _ = run()
    torch.cuda.synchronize()
    return (time.time() - t0) / n_repeat * 1000


if __name__ == '__main__':
    print("Quality of bf16 S_traj gradients vs fp32:\n")
    for shape in [(1, 16, 1024, 32), (1, 32, 4096, 32), (1, 83, 2048, 32)]:
        test_bf16_gradient_quality(*shape)

    print("\n\nE2E training step at T=128K with bf16 S_traj:\n")
    print(f"{'config':>22s}  {'CUDA f+b':>10s}  {'CUDA fwd':>9s}  {'Par bwd (bf16)':>15s}  "
          f"{'Hybrid tot':>11s}  {'e2e spd':>8s}")
    for name, H, n in [("E88-n32 480M", 83, 32),
                       ("E88-n16 480M", 141, 16),
                       ("Small H=32",    32, 32)]:
        for T in [32768, 65536, 131072]:
            config = f"{name} T={T}"
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                cuda_fwd = bench_cuda_fwd(1, H, T, n)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                cuda_fb, _ = bench_cuda_fwd_bwd(1, H, T, n)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                par_bwd, peak = bench_bf16_bwd(1, H, T, n)
                hybrid = cuda_fwd + par_bwd
                spd = cuda_fb / hybrid
                print(f"{config:>22s}  {cuda_fb:>7.1f} ms  {cuda_fwd:>6.1f} ms  "
                      f"{par_bwd:>12.1f} ms  {hybrid:>8.1f} ms  {spd:>6.2f}×  "
                      f"[peak {peak:.1f}GB]")
            except Exception as e:
                print(f"{config:>22s}  FAIL: {str(e)[:100]}")
                torch.cuda.empty_cache()
