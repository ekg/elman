"""Sweep Triton backward BLOCK_H at production H=386 to find the best
config. The current default (BLOCK_H=4 for H>=256) gives ~97 programs/
batch, which appears to under-saturate the SMs at this scale.

Tests BLOCK_H in {1, 2, 4, 8, 16} with num_warps in {2, 4, 8} where
SRAM permits. Runs at the production E88 1.27B-shape: H=386, N=V=32,
B=1 (matches the 1.27B Pile bench above).
"""
from __future__ import absolute_import

import os
import sys
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import triton

from elman.triton.e88_triton_forward import e88_triton_forward, DEFAULT_CKPT_INTERVAL
from elman.triton.e88_triton_backward import (
    _e88_backward_kernel,
    _next_pow2,
)


def make_inputs(B, T, H, N, V, dtype, device):
    g = torch.Generator(device=device).manual_seed(7)
    k = torch.randn((T, B, H, N), generator=g, device=device, dtype=torch.float32)
    q = torch.randn((T, B, H, N), generator=g, device=device, dtype=torch.float32)
    v = 0.3 * torch.randn((T, B, H, V), generator=g, device=device, dtype=torch.float32)
    k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
    decay = torch.exp(-torch.nn.functional.softplus(
        torch.randn((T, B, H), generator=g, device=device, dtype=torch.float32)))
    S0 = torch.zeros((B, H, N, V), dtype=torch.float32, device=device)
    return (
        S0.to(dtype).contiguous(),
        k.to(dtype).contiguous(),
        v.to(dtype).contiguous(),
        q.to(dtype).contiguous(),
        decay.to(dtype).contiguous(),
    )


def call_backward_with_block_h(k, v, q, decay, S_ckpt, d_out, block_h, num_warps,
                                ckpt_interval=DEFAULT_CKPT_INTERVAL):
    """Manually launch the backward kernel with explicit (block_h, num_warps)."""
    T, B, H, N = k.shape
    Vsz = v.shape[-1]
    BLOCK_N = _next_pow2(N)
    BLOCK_V = _next_pow2(Vsz)

    k_c = k.contiguous(); v_c = v.contiguous(); q_c = q.contiguous()
    d_c = decay.contiguous(); sc_c = S_ckpt.contiguous()
    do_c = d_out.contiguous()
    dsf_c = torch.zeros((B, H, N, Vsz), dtype=k_c.dtype, device=k.device)

    out_dtype = k_c.dtype
    d_k = torch.empty_like(k_c)
    d_v = torch.empty_like(v_c)
    d_q = torch.empty_like(q_c)
    d_decay = torch.empty_like(d_c)
    d_S0 = torch.empty((B, H, N, Vsz), dtype=out_dtype, device=k.device)

    num_progs_h = (H + block_h - 1) // block_h
    grid = (B, num_progs_h)

    # Per-program scratch (matches the wrapper allocation).
    scratch_numel = (
        B * num_progs_h
        * (ckpt_interval + 1)
        * block_h * BLOCK_N * BLOCK_V
    )
    seg_scratch = torch.empty(scratch_numel, dtype=torch.float32, device=k.device)

    _e88_backward_kernel[grid](
        k_c, v_c, q_c, d_c, sc_c,
        seg_scratch,
        do_c, dsf_c,
        d_k, d_v, d_q, d_decay, d_S0,
        k_c.stride(0), k_c.stride(1), k_c.stride(2), k_c.stride(3),
        v_c.stride(0), v_c.stride(1), v_c.stride(2), v_c.stride(3),
        q_c.stride(0), q_c.stride(1), q_c.stride(2), q_c.stride(3),
        d_c.stride(0), d_c.stride(1), d_c.stride(2),
        sc_c.stride(0), sc_c.stride(1), sc_c.stride(2),
        sc_c.stride(3), sc_c.stride(4),
        do_c.stride(0), do_c.stride(1), do_c.stride(2), do_c.stride(3),
        dsf_c.stride(0), dsf_c.stride(1), dsf_c.stride(2), dsf_c.stride(3),
        d_k.stride(0), d_k.stride(1), d_k.stride(2), d_k.stride(3),
        d_v.stride(0), d_v.stride(1), d_v.stride(2), d_v.stride(3),
        d_q.stride(0), d_q.stride(1), d_q.stride(2), d_q.stride(3),
        d_decay.stride(0), d_decay.stride(1), d_decay.stride(2),
        d_S0.stride(0), d_S0.stride(1), d_S0.stride(2), d_S0.stride(3),
        T=T, B=B, H=H, N=N, V=Vsz,
        BLOCK_N=BLOCK_N, BLOCK_V=BLOCK_V,
        BLOCK_H=block_h,
        CKPT_INTERVAL=ckpt_interval,
        NUM_PROGS_H=num_progs_h,
        num_warps=num_warps,
    )


def bench(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    if not torch.cuda.is_available():
        print("CUDA required"); sys.exit(1)
    device = torch.device("cuda")

    # Production-shape config — H=386 from the 1.27B E88.
    H = 386
    N = V = 32

    print(f"Sweep BLOCK_H for backward at H={H}, N=V={N}")
    print(f"{'B':>3} {'T':>6} {'BH':>3} {'nw':>3} {'time ms':>10}")
    print("-" * 32)

    shapes = [
        (1, 512), (1, 4096), (1, 16384),
    ]

    # Candidate (block_h, num_warps) pairs. Skip BLOCK_H>16 (SRAM).
    # SRAM budget for backward: state tile [BLOCK_H, N, V] x4 (S_t, S_tm1, dS_t, d_pre)
    # at fp32 = BLOCK_H * 32 * 32 * 4 * 4 bytes = BLOCK_H * 16 KB
    #   BLOCK_H=8 -> 128 KB (over default 48 KB SRAM, will spill)
    #   BLOCK_H=4 -> 64 KB (over)
    #   BLOCK_H=2 -> 32 KB (fits!)
    #   BLOCK_H=1 -> 16 KB (fits well)
    # But Triton also packs into registers if not in shared. So worth benching all.
    candidates = []
    for bh in [1, 2, 4, 8, 16]:
        if bh > H:
            continue
        for nw in [2, 4, 8]:
            candidates.append((bh, nw))

    best_per_shape = {}
    for B, T in shapes:
        S0, k, v, q, decay = make_inputs(B, T, H, N, V, torch.bfloat16, device)
        out, _, S_ckpt = e88_triton_forward(S0, k, v, q, decay)
        d_out = torch.randn_like(out)

        best_t = float("inf")
        best_cfg = None
        for bh, nw in candidates:
            try:
                t = bench(
                    lambda: call_backward_with_block_h(k, v, q, decay, S_ckpt, d_out, bh, nw),
                    iters=10, warmup=3,
                )
                marker = ""
                if t < best_t:
                    best_t = t
                    best_cfg = (bh, nw)
                    marker = " *"
                print(f"{B:>3} {T:>6} {bh:>3} {nw:>3} {t*1e3:>10.3f}{marker}")
            except Exception as e:
                print(f"{B:>3} {T:>6} {bh:>3} {nw:>3}  FAIL: {type(e).__name__}")
        print(f"  BEST: BLOCK_H={best_cfg[0]} num_warps={best_cfg[1]}  ({best_t*1e3:.3f} ms)")
        best_per_shape[(B, T)] = (best_cfg, best_t)
        print()


if __name__ == "__main__":
    main()
