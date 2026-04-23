"""Run a single (H, T) benchmark — for use with subprocess-per-T."""

import sys, os, time
import torch
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase7_fused_backward import backward_e88_fused_rank1
from phase4_newton_driver import sequential_e88_forward
from elman.models.e88_fla_hybrid import E88FLAHybridCUDAFunction


def main(H, T, n):
    # CUDA f+b
    g = torch.Generator(device='cuda').manual_seed(0)
    dt = torch.bfloat16
    k = (0.3 * torch.randn(T, 1, H, n, generator=g, dtype=dt, device='cuda')).requires_grad_(True)
    v = (0.3 * torch.randn(T, 1, H, n, generator=g, dtype=dt, device='cuda')).requires_grad_(True)
    q = (0.3 * torch.randn(T, 1, H, n, generator=g, dtype=dt, device='cuda')).requires_grad_(True)
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(T, 1, H, generator=g, dtype=dt, device='cuda')
                          ).detach().requires_grad_(True)
    S0 = 0.1 * torch.randn(1, H, n, n, generator=g, dtype=dt, device='cuda')

    def cuda_fwd():
        return E88FLAHybridCUDAFunction.apply(True, k, v, q, decay, S0, H)

    def cuda_fb():
        S_final, output = E88FLAHybridCUDAFunction.apply(True, k, v, q, decay, S0, H)
        loss = output.sum() + S_final.pow(2).sum() * 1e-4
        loss.backward()
        k.grad = None; v.grad = None; q.grad = None; decay.grad = None

    cuda_fwd(); cuda_fwd()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(3): cuda_fwd()
    torch.cuda.synchronize()
    c_fwd = (time.time() - t0) / 3 * 1000

    cuda_fb(); cuda_fb()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(3): cuda_fb()
    torch.cuda.synchronize()
    c_fb = (time.time() - t0) / 3 * 1000

    # Pararnn bwd (separate process, fresh memory)
    del k, v, q, decay, S0
    torch.cuda.empty_cache()

    g2 = torch.Generator(device='cuda').manual_seed(0)
    S0_f = 0.1 * torch.randn(1, H, n, n, generator=g2, dtype=torch.float32, device='cuda')
    K_f = 0.3 * torch.randn(1, H, T, n, generator=g2, dtype=torch.float32, device='cuda')
    V_f = 0.3 * torch.randn(1, H, T, n, generator=g2, dtype=torch.float32, device='cuda')
    decay_f = 0.9 + 0.1 * torch.rand(1, H, T, generator=g2, dtype=torch.float32, device='cuda')
    S_traj = sequential_e88_forward(S0_f, K_f, V_f, decay_f).to(dt)
    del S0_f, K_f, V_f, decay_f
    torch.cuda.empty_cache()

    K_d = 0.3 * torch.randn(1, H, T, n, generator=g2, dtype=dt, device='cuda')
    V_d = 0.3 * torch.randn(1, H, T, n, generator=g2, dtype=dt, device='cuda')
    decay_d = (0.9 + 0.1 * torch.rand(1, H, T, generator=g2, dtype=dt, device='cuda'))
    g_T = 0.01 * torch.randn(1, H, n, n, dtype=dt, device='cuda')
    dL_dout = 0.01 * torch.randn(1, H, T, n, dtype=dt, device='cuda')
    q_d = 0.3 * torch.randn(1, H, T, n, dtype=dt, device='cuda')

    def par_bwd_plus_dq():
        d = backward_e88_fused_rank1(S_traj, K_d, V_d, decay_d, g_T, dL_dout, q_d,
                                      num_warps=1 if n == 16 else 4, num_stages=1)
        dQ = torch.einsum('bhti,bhtij->bhtj', dL_dout, S_traj[:, :, 1:])
        return d, dQ

    par_bwd_plus_dq(); par_bwd_plus_dq()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(3): par_bwd_plus_dq()
    torch.cuda.synchronize()
    p_bwd = (time.time() - t0) / 3 * 1000
    peak = torch.cuda.max_memory_allocated() / 1024**3

    hybrid = c_fwd + p_bwd
    spd = c_fb / hybrid
    ceiling = c_fb / c_fwd
    print(f"H={H} n={n} T={T:>8d}  "
          f"CUDA f+b={c_fb:>8.1f}  fwd={c_fwd:>7.1f}  bwd={c_fb - c_fwd:>7.1f}  "
          f"Par bwd={p_bwd:>7.1f}  hybrid={hybrid:>7.1f}  "
          f"spd={spd:.2f}x  ceiling={ceiling:.2f}x  peak={peak:.1f}GB")


if __name__ == '__main__':
    H = int(sys.argv[1])
    T = int(sys.argv[2])
    n = int(sys.argv[3])
    main(H, T, n)
