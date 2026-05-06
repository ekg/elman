"""Sweep num_stages and num_warps for the sparse-ckpt backward at H=386.

num_stages controls Triton's software pipelining of memory loads with
compute. Higher num_stages = more loads in flight before the compute
catches up (better latency hiding) but uses more registers/SRAM. Worth
trying since the backward kernel does many sequential loads per
timestep.
"""
from __future__ import absolute_import

import os, sys, time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from elman.triton.e88_triton_forward import e88_triton_forward, DEFAULT_CKPT_INTERVAL
from elman.triton.e88_triton_backward import _e88_backward_kernel, _next_pow2


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
    return (S0.to(dtype).contiguous(),
            k.to(dtype).contiguous(),
            v.to(dtype).contiguous(),
            q.to(dtype).contiguous(),
            decay.to(dtype).contiguous())


def call_bwd_with_stages(k, v, q, decay, S_ckpt, d_out, num_stages, num_warps,
                         ckpt_interval=DEFAULT_CKPT_INTERVAL, block_h=1):
    T, B, H, N = k.shape
    Vsz = v.shape[-1]
    BLOCK_N = _next_pow2(N); BLOCK_V = _next_pow2(Vsz)
    k_c, v_c, q_c, d_c = k.contiguous(), v.contiguous(), q.contiguous(), decay.contiguous()
    sc_c, do_c = S_ckpt.contiguous(), d_out.contiguous()
    dsf_c = torch.zeros((B, H, N, Vsz), dtype=k_c.dtype, device=k.device)
    d_k = torch.empty_like(k_c); d_v = torch.empty_like(v_c); d_q = torch.empty_like(q_c)
    d_decay = torch.empty_like(d_c)
    d_S0 = torch.empty((B, H, N, Vsz), dtype=k_c.dtype, device=k.device)
    num_progs_h = (H + block_h - 1) // block_h
    grid = (B, num_progs_h)
    scratch_numel = B * num_progs_h * (ckpt_interval + 1) * block_h * BLOCK_N * BLOCK_V
    seg_scratch = torch.empty(scratch_numel, dtype=k_c.dtype, device=k.device)

    _e88_backward_kernel[grid](
        k_c, v_c, q_c, d_c, sc_c, seg_scratch, do_c, dsf_c,
        d_k, d_v, d_q, d_decay, d_S0,
        k_c.stride(0), k_c.stride(1), k_c.stride(2), k_c.stride(3),
        v_c.stride(0), v_c.stride(1), v_c.stride(2), v_c.stride(3),
        q_c.stride(0), q_c.stride(1), q_c.stride(2), q_c.stride(3),
        d_c.stride(0), d_c.stride(1), d_c.stride(2),
        sc_c.stride(0), sc_c.stride(1), sc_c.stride(2), sc_c.stride(3), sc_c.stride(4),
        do_c.stride(0), do_c.stride(1), do_c.stride(2), do_c.stride(3),
        dsf_c.stride(0), dsf_c.stride(1), dsf_c.stride(2), dsf_c.stride(3),
        d_k.stride(0), d_k.stride(1), d_k.stride(2), d_k.stride(3),
        d_v.stride(0), d_v.stride(1), d_v.stride(2), d_v.stride(3),
        d_q.stride(0), d_q.stride(1), d_q.stride(2), d_q.stride(3),
        d_decay.stride(0), d_decay.stride(1), d_decay.stride(2),
        d_S0.stride(0), d_S0.stride(1), d_S0.stride(2), d_S0.stride(3),
        T=T, B=B, H=H, N=N, V=Vsz,
        BLOCK_N=BLOCK_N, BLOCK_V=BLOCK_V, BLOCK_H=block_h,
        CKPT_INTERVAL=ckpt_interval, NUM_PROGS_H=num_progs_h,
        num_warps=num_warps, num_stages=num_stages,
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
    if not torch.cuda.is_available(): sys.exit(1)
    device = torch.device("cuda")
    H, N, V = 386, 32, 32

    print(f"Backward num_stages sweep at H={H}, N=V={N}")
    print(f"{'B':>3} {'T':>6} {'NS':>3} {'NW':>3} {'time ms':>10}")
    print("-" * 32)

    for B, T in [(1, 512), (1, 4096), (1, 16384)]:
        S0, k, v, q, decay = make_inputs(B, T, H, N, V, torch.bfloat16, device)
        out, _, S_ckpt = e88_triton_forward(S0, k, v, q, decay)
        d_out = torch.randn_like(out)
        best_t = float("inf"); best_cfg = None
        for ns in [1, 2, 3, 4, 5]:
            for nw in [2, 4]:
                try:
                    t = bench(lambda: call_bwd_with_stages(k, v, q, decay, S_ckpt, d_out, ns, nw),
                              iters=10, warmup=3)
                    marker = ""
                    if t < best_t:
                        best_t = t; best_cfg = (ns, nw); marker = " *"
                    print(f"{B:>3} {T:>6} {ns:>3} {nw:>3} {t*1e3:>10.3f}{marker}")
                except Exception as e:
                    print(f"{B:>3} {T:>6} {ns:>3} {nw:>3}  FAIL: {type(e).__name__}: {str(e)[:50]}")
        print(f"  BEST: num_stages={best_cfg[0]} num_warps={best_cfg[1]}  ({best_t*1e3:.3f} ms)")
        print()


if __name__ == "__main__":
    main()
