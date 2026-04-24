"""Profile: where does time go in a single E88 layer forward+backward at
training-relevant shapes?

Training typical: B=16, T=512, H=141, N=16.
Our benchmark regime: B=1, T=16K-65K.

The kernel microbenchmark showed 2.65× speedup, but training is slower.
Figure out why.
"""

import sys, os, time
import torch

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elman.models.e88_fla_hybrid import E88FLAHybridCUDAFunction
from phase6_hybrid import PararnnHybridE88V2


def bench(fn, n_repeat=20):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat): fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n_repeat * 1000


def profile(B, H, T, N, name="", with_autograd=True):
    """Compare CUDA vs HYBRID at given shape."""
    dt = torch.bfloat16
    g = torch.Generator(device='cuda').manual_seed(0)
    k = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda'))
    v = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda'))
    q = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda'))
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, generator=g, dtype=dt, device='cuda'))
    S0 = (0.1 * torch.randn(B, H, N, N, generator=g, dtype=dt, device='cuda'))
    if with_autograd:
        k.requires_grad_(True); v.requires_grad_(True); q.requires_grad_(True)
        decay = decay.detach().requires_grad_(True)
        S0 = S0.clone().requires_grad_(True)

    # --- CUDA fwd only ---
    def cuda_fwd():
        _, out = E88FLAHybridCUDAFunction.apply(True, k, v, q, decay, S0, H)
        return out
    cuda_fwd_ms = bench(cuda_fwd)

    # --- CUDA f+b ---
    def cuda_fb():
        _, out = E88FLAHybridCUDAFunction.apply(True, k, v, q, decay, S0, H)
        loss = out.sum()
        loss.backward()
        k.grad = None; v.grad = None; q.grad = None; decay.grad = None
        if S0.grad is not None: S0.grad = None
    cuda_fb_ms = bench(cuda_fb) if with_autograd else float('nan')

    # --- HYBRID fwd only ---
    def hyb_fwd():
        _, out = PararnnHybridE88V2.apply(True, k, v, q, decay, S0, H)
        return out
    hyb_fwd_ms = bench(hyb_fwd)

    # --- HYBRID f+b ---
    def hyb_fb():
        _, out = PararnnHybridE88V2.apply(True, k, v, q, decay, S0, H)
        loss = out.sum()
        loss.backward()
        k.grad = None; v.grad = None; q.grad = None; decay.grad = None
        if S0.grad is not None: S0.grad = None
    hyb_fb_ms = bench(hyb_fb) if with_autograd else float('nan')

    print(f"  {name or f'B={B} H={H} T={T} N={N}'}:")
    print(f"    CUDA   fwd={cuda_fwd_ms:>6.3f} ms   f+b={cuda_fb_ms:>6.3f} ms")
    print(f"    HYBRID fwd={hyb_fwd_ms:>6.3f} ms   f+b={hyb_fb_ms:>6.3f} ms")
    print(f"    ratio: fwd={cuda_fwd_ms/hyb_fwd_ms:.2f}×  f+b={cuda_fb_ms/hyb_fb_ms:.2f}×")


if __name__ == '__main__':
    print("=== Training-relevant shapes (B=16, T=512, per layer) ===\n")
    profile(16, 141, 512, 16, "E88-n16 training (H=141 B=16 T=512)")
    profile(16, 83, 512, 32, "E88-n32 training (H=83  B=16 T=512)")

    print("\n=== Kernel microbench regime (B=1, long T) ===\n")
    profile(1, 141, 16384, 16, "E88-n16 micro (H=141 B=1 T=16K)")
    profile(1, 83, 16384, 32, "E88-n32 micro (H=83  B=1 T=16K)")

    print("\n=== Intermediate (B=1, T=32K with grad_ckpt-like pattern) ===\n")
    profile(1, 141, 32768, 16, "E88-n16 (H=141 B=1 T=32K)")
