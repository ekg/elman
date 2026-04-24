"""Final benchmark: install_hybrid (V2) vs install_hybrid_v3 via E88OptimizedCUDAFunction.

Both install-paths are monkey patches; we can only have one active at a time,
so we save/restore `E88OptimizedCUDAFunction.apply` between trials.
"""

import os
import sys
import time

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '3')

import torch

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
sys.path.insert(0, os.path.dirname(THIS))
sys.path.insert(0, '/home/erikg/elman')


def time_fn(fn, n_warmup=5, n_iter=20):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n_iter * 1000


def make_inputs(B, T, H, N, dtype=torch.bfloat16):
    torch.manual_seed(0)
    k = (0.3 * torch.randn(B, T, H, N, dtype=dtype, device='cuda')).requires_grad_(True)
    v = (0.3 * torch.randn(B, T, H, N, dtype=dtype, device='cuda')).requires_grad_(True)
    q = (0.3 * torch.randn(B, T, H, N, dtype=dtype, device='cuda')).requires_grad_(True)
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(B, T, H, dtype=dtype, device='cuda')).detach().requires_grad_(True)
    g = torch.randn(B, T, H, N, dtype=dtype, device='cuda').requires_grad_(True)
    S0 = (0.1 * torch.randn(B, H, N, N, dtype=dtype, device='cuda')).requires_grad_(True)
    return k, v, q, decay, g, S0


def main():
    import elman.models.e88_fla_hybrid as e88m
    ORIG_APPLY = e88m.E88OptimizedCUDAFunction.apply

    shapes = [
        (16, 512, 141, 16),
        (16, 512, 83, 32),
        (1, 32768, 141, 16),
        (1, 32768, 83, 32),
        (8, 1024, 141, 16),
        (4, 2048, 141, 16),
    ]

    def run_path(path_apply, k, v, q, decay, g, S0, H):
        def _run():
            k.grad = None; v.grad = None; q.grad = None
            decay.grad = None; g.grad = None; S0.grad = None
            S_f, out = path_apply(True, k, v, q, decay, g, S0, H)
            out.sum().backward()
        return _run

    # --- Install V2 and measure ---
    from install_hybrid import install as install_v2
    install_v2()
    V2_APPLY = e88m.E88OptimizedCUDAFunction.apply

    v2_times = {}
    for B, T, H, N in shapes:
        torch.cuda.empty_cache()
        k, v, q, decay, g, S0 = make_inputs(B, T, H, N)
        ms = time_fn(run_path(V2_APPLY, k, v, q, decay, g, S0, H))
        v2_times[(B, T, H, N)] = ms

    # Restore original and install V3
    e88m.E88OptimizedCUDAFunction.apply = ORIG_APPLY
    from install_hybrid_v3 import install as install_v3
    install_v3()
    V3_APPLY = e88m.E88OptimizedCUDAFunction.apply

    v3_times = {}
    for B, T, H, N in shapes:
        torch.cuda.empty_cache()
        k, v, q, decay, g, S0 = make_inputs(B, T, H, N)
        ms = time_fn(run_path(V3_APPLY, k, v, q, decay, g, S0, H))
        v3_times[(B, T, H, N)] = ms

    # Also time raw CUDA (no patching)
    cuda_times = {}
    for B, T, H, N in shapes:
        torch.cuda.empty_cache()
        k, v, q, decay, g, S0 = make_inputs(B, T, H, N)
        ms = time_fn(run_path(ORIG_APPLY, k, v, q, decay, g, S0, H))
        cuda_times[(B, T, H, N)] = ms

    print()
    print("=" * 90)
    print("Install V3 vs Install V2 (via E88OptimizedCUDAFunction.apply)")
    print("=" * 90)
    print(f"{'Shape':<32} {'CUDA ms':>9} {'V2 ms':>9} {'V3 ms':>9} {'V3/V2':>8} {'V3/CUDA':>9}")
    print("-" * 90)
    for shape in shapes:
        B, T, H, N = shape
        label = f"B={B} T={T:>5d} H={H:>3d} N={N}"
        c = cuda_times[shape]
        v2_ = v2_times[shape]
        v3_ = v3_times[shape]
        print(f"{label:<32} {c:>9.3f} {v2_:>9.3f} {v3_:>9.3f} {v2_/v3_:>7.2f}x {c/v3_:>8.2f}x")


if __name__ == '__main__':
    main()
