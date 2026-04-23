"""Test the CUDA fused backward baseline for correctness vs Triton version."""

import sys, os, time
import torch
from torch.utils.cpp_extension import load

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase4_newton_driver import sequential_e88_forward
from phase7_fused_backward import backward_e88_fused_rank1


_CUDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cuda_fused_backward.cu')
print(f"Compiling {_CUDA}...")
ext = load(name='cuda_bwd_proto', sources=[_CUDA],
           extra_cuda_cflags=['-O3', '-std=c++17', '--use_fast_math',
                               '-gencode=arch=compute_80,code=sm_80'],
           verbose=True)
print("Compiled.")


def setup(B, H, T, N, seed=0):
    dt = torch.bfloat16
    g = torch.Generator(device='cuda').manual_seed(seed)
    # Build fp32 S_traj via sequential forward, cast to bf16
    S0_f = 0.1 * torch.randn(B, H, N, N, generator=g, dtype=torch.float32, device='cuda')
    K_f = 0.3 * torch.randn(B, H, T, N, generator=g, dtype=torch.float32, device='cuda')
    V_f = 0.3 * torch.randn(B, H, T, N, generator=g, dtype=torch.float32, device='cuda')
    decay_f = torch.sigmoid(0.5 + 0.1 * torch.randn(B, H, T, generator=g, dtype=torch.float32, device='cuda'))
    S_traj = sequential_e88_forward(S0_f, K_f, V_f, decay_f).to(dt)
    del S0_f, K_f, V_f, decay_f
    K = 0.3 * torch.randn(B, H, T, N, generator=g, dtype=dt, device='cuda')
    V = 0.3 * torch.randn(B, H, T, N, generator=g, dtype=dt, device='cuda')
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(B, H, T, generator=g, dtype=dt, device='cuda'))
    g_T = 0.01 * torch.randn(B, H, N, N, dtype=dt, device='cuda')
    dL_dout = 0.01 * torch.randn(B, H, T, N, dtype=dt, device='cuda')
    q = 0.3 * torch.randn(B, H, T, N, dtype=dt, device='cuda')
    return S_traj, K, V, decay, g_T, dL_dout, q


def test_correctness(B, H, T, N):
    S_traj, K, V, decay, g_T, dL_dout, q = setup(B, H, T, N)

    # Triton reference
    dS0_t, dK_t, dV_t, ddec_t = backward_e88_fused_rank1(
        S_traj, K, V, decay, g_T, dL_dout, q,
        num_warps=4 if N == 32 else 1, num_stages=1)

    # CUDA warp-parallel
    dS0_c, dK_c, dV_c, ddec_c = ext.cuda_fused_backward(
        S_traj, K, V, decay, g_T, dL_dout, q)

    def rel(a, b):
        return (a.float() - b.float()).abs().max().item() / max(b.float().abs().max().item(), 1e-10)

    e_S0 = rel(dS0_c, dS0_t)
    e_K = rel(dK_c, dK_t)
    e_V = rel(dV_c, dV_t)
    e_d = rel(ddec_c, ddec_t)
    # bf16 tolerance ~ 8e-3
    tol = 5e-2  # allow some bf16 accumulation slack
    worst = max(e_S0, e_K, e_V, e_d)
    status = "PASS" if worst < tol else "FAIL"
    print(f"  B={B} H={H:3d} T={T:5d} N={N}  rel err: "
          f"S0={e_S0:.1e} K={e_K:.1e} V={e_V:.1e} dec={e_d:.1e}  [{status}]")


def bench(B, H, T, N, n_repeat=3):
    S_traj, K, V, decay, g_T, dL_dout, q = setup(B, H, T, N)

    def run_triton():
        return backward_e88_fused_rank1(S_traj, K, V, decay, g_T, dL_dout, q,
                                         num_warps=4 if N == 32 else 1, num_stages=1)

    def run_cuda():
        return ext.cuda_fused_backward(S_traj, K, V, decay, g_T, dL_dout, q)

    for _ in range(3): run_triton()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat): run_triton()
    torch.cuda.synchronize()
    tri_ms = (time.time() - t0) / n_repeat * 1000

    for _ in range(3): run_cuda()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat): run_cuda()
    torch.cuda.synchronize()
    cuda_ms = (time.time() - t0) / n_repeat * 1000

    print(f"  B={B} H={H:3d} T={T:5d} N={N}  Triton={tri_ms:>7.2f} ms   "
          f"CUDA-base={cuda_ms:>7.2f} ms   ratio={tri_ms/cuda_ms:.2f}×")


if __name__ == '__main__':
    print("CUDA backward baseline correctness:\n")
    for shape in [(1, 2, 64, 16), (1, 4, 256, 16), (1, 2, 128, 32)]:
        test_correctness(*shape)

    print("\nCUDA backward baseline timing (target: match Triton, then exceed):\n")
    for shape in [(1, 141, 8192, 16), (1, 141, 32768, 16), (1, 83, 8192, 32), (1, 83, 32768, 32)]:
        try:
            bench(*shape)
        except Exception as e:
            print(f"  FAIL: {str(e)[:100]}")
