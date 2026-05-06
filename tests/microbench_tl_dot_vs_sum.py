"""Microbench: is tl.dot faster than tl.sum-of-products for the
matrix-vector reductions used in E88 backward at our shape?

E88 backward at BLOCK_H=1, N=V=32 has these key reductions per step:
  - Type A: result[v] = sum_n M[n,v] * vec[n]   (M [N,V], vec [N], result [V])
  - Type B: result[n] = sum_v M[n,v] * vec[v]   (M [N,V], vec [V], result [N])

Currently implemented as `tl.sum(M * vec[:, :, None], axis=...)`.
tl.dot would use tensor cores. For M=1 (since BLOCK_H=1) this may not
hit minimum tile sizes; we test both directions.
"""
from __future__ import absolute_import

import os, sys, time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import triton
import triton.language as tl


# -- Type A: result[v] = sum_n M[n, v] * vec[n] -----------------

@triton.jit
def kernel_type_a_sum(M_ptr, V_ptr, R_ptr, N: tl.constexpr, V: tl.constexpr,
                       num_iters: tl.constexpr):
    """Reduce via tl.sum: result = sum_n M[n, v] * vec[n]."""
    # Each program does num_iters iterations of the same reduction.
    # This simulates the K=16 steps within a backward segment.
    n_idx = tl.arange(0, N)
    v_idx = tl.arange(0, V)
    M_off = n_idx[:, None] * V + v_idx[None, :]
    M = tl.load(M_ptr + M_off)
    vec = tl.load(V_ptr + n_idx)

    acc = tl.zeros([V], dtype=tl.float32)
    for _ in range(num_iters):
        # The actual reduction.
        result = tl.sum(M * vec[:, None], axis=0)  # [V]
        acc += result

    tl.store(R_ptr + v_idx, acc)


@triton.jit
def kernel_type_a_dot(M_ptr, V_ptr, R_ptr, N: tl.constexpr, V: tl.constexpr,
                      num_iters: tl.constexpr):
    """Reduce via tl.dot: result = vec_row @ M, broadcasting to satisfy MMA tile sizes."""
    n_idx = tl.arange(0, N)
    v_idx = tl.arange(0, V)
    M_off = n_idx[:, None] * V + v_idx[None, :]
    M = tl.load(M_ptr + M_off)
    vec = tl.load(V_ptr + n_idx)

    # Make only row 0 nonzero (mask) so tl.dot produces the answer in
    # row 0 and zeros elsewhere. This satisfies the M>=16 tile size for
    # tensor cores. Then sum along axis=0 picks out row 0.
    i_idx = tl.arange(0, 16)
    mask_row0 = (i_idx == 0)[:, None]  # [16, 1]
    vec_zeros = tl.zeros([16, N], dtype=vec.dtype)
    vec_padded = tl.where(mask_row0, vec[None, :], vec_zeros)  # [16, N]

    acc = tl.zeros([V], dtype=tl.float32)
    for _ in range(num_iters):
        # tl.dot([16, N], [N, V]) -> [16, V]; row 0 is the answer, rest are 0.
        result_padded = tl.dot(vec_padded, M)  # [16, V]
        # Sum across the padding rows — only row 0 is nonzero, so sum picks it.
        result = tl.sum(result_padded, axis=0)  # [V]
        acc += result

    tl.store(R_ptr + v_idx, acc)


def bench_kernel(kernel_fn, args, kwargs, num_progs, warmup=10, iters=50):
    grid = (num_progs,)
    for _ in range(warmup):
        kernel_fn[grid](*args, **kwargs)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        kernel_fn[grid](*args, **kwargs)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def main():
    if not torch.cuda.is_available():
        sys.exit(1)
    device = "cuda"
    N = V = 32
    NUM_PROGS = 3088   # matches H=386 * B=8 production
    NUM_ITERS = 16     # K=16 steps per segment
    DTYPE = torch.bfloat16

    M = torch.randn((N, V), dtype=DTYPE, device=device).contiguous()
    vec = torch.randn((N,), dtype=DTYPE, device=device).contiguous()
    R_sum = torch.empty((V,), dtype=torch.float32, device=device)
    R_dot = torch.empty((V,), dtype=torch.float32, device=device)

    # First, verify correctness at one program.
    kernel_type_a_sum[(1,)](M, vec, R_sum, N=N, V=V, num_iters=1)
    kernel_type_a_dot[(1,)](M, vec, R_dot, N=N, V=V, num_iters=1)
    diff = (R_sum.float() - R_dot.float()).abs().max().item()
    print(f"Parity check sum vs dot: max_abs={diff:.3e}")
    if diff > 0.5:
        print("  WARN: large diff. Algebra may be wrong.")

    # Bench.
    print(f"\nBench: {NUM_PROGS} programs x {NUM_ITERS} iters of [N={N}, V={V}] reduction (Type A)")
    print(f"  num_warps=2 (matches production):")
    t_sum = bench_kernel(kernel_type_a_sum, (M, vec, R_sum), {"N": N, "V": V, "num_iters": NUM_ITERS}, NUM_PROGS)
    print(f"    tl.sum: {t_sum:.3f} ms / call")
    t_dot = bench_kernel(kernel_type_a_dot, (M, vec, R_dot), {"N": N, "V": V, "num_iters": NUM_ITERS}, NUM_PROGS)
    print(f"    tl.dot: {t_dot:.3f} ms / call")
    print(f"    speedup: {t_sum/t_dot:.2f}x")


if __name__ == "__main__":
    main()
