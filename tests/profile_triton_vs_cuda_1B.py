"""Profile Triton vs CUDA E88 at 1.27B production shape to find where
the remaining ~17% throughput gap lives.

We don't run a full Pile training step — just the inner recurrence
forward + backward pass, in isolation. This lets us see whether the
gap is in the kernel itself (compute), allocation, autograd, or
something else.
"""
from __future__ import absolute_import

import os
import sys
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from elman.triton.e88_triton_forward import e88_triton_forward
from elman.triton.e88_triton_backward import e88_triton_backward, e88_triton

try:
    import hasty_pytorch_lib
except Exception as e:
    print(f"[warn] CUDA E88 lib not loaded: {e}")
    hasty_pytorch_lib = None


# Production E88 1.27B inner shape (per layer):
B, T, H, N, V = 8, 512, 386, 32, 32
DEPTH = 14  # production 1.27B depth
DTYPE = torch.bfloat16


def make_inputs(B, T, H, N, V, dtype, device, requires_grad=True, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    k = torch.randn((T, B, H, N), generator=g, device=device, dtype=torch.float32)
    q = torch.randn((T, B, H, N), generator=g, device=device, dtype=torch.float32)
    v = 0.3 * torch.randn((T, B, H, V), generator=g, device=device, dtype=torch.float32)
    k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
    decay = torch.exp(-torch.nn.functional.softplus(
        torch.randn((T, B, H), generator=g, device=device, dtype=torch.float32)))
    S0 = torch.zeros((B, H, N, V), dtype=torch.float32, device=device)
    return (
        S0.to(dtype).contiguous().requires_grad_(requires_grad),
        k.to(dtype).contiguous().requires_grad_(requires_grad),
        v.to(dtype).contiguous().requires_grad_(requires_grad),
        q.to(dtype).contiguous().requires_grad_(requires_grad),
        decay.to(dtype).contiguous().requires_grad_(requires_grad),
    )


def time_event(fn, iters=20, warmup=5):
    """Time fn() with cuda events."""
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starter.record()
    for _ in range(iters):
        fn()
    ender.record()
    torch.cuda.synchronize()
    return starter.elapsed_time(ender) / iters  # ms


def main():
    device = torch.device("cuda")
    print(f"Profile config: B={B} T={T} H={H} N=V={N} dtype={DTYPE}")
    print(f"Production layer count: {DEPTH}\n")

    # ---- 1) Triton forward (kernel-only, fresh inputs each call) ----
    S0, k, v, q, decay = make_inputs(B, T, H, N, V, DTYPE, device, requires_grad=False)

    def triton_fwd_only():
        out, S_final, S_ckpt = e88_triton_forward(S0, k, v, q, decay)
        return out, S_final, S_ckpt

    t_tri_fwd = time_event(triton_fwd_only, iters=20, warmup=5)
    print(f"  Triton forward only:        {t_tri_fwd:7.3f} ms/call")

    # ---- 2) Triton backward (kernel-only) ----
    out, S_final, S_ckpt = e88_triton_forward(S0, k, v, q, decay)
    d_out = torch.randn_like(out)

    def triton_bwd_only():
        e88_triton_backward(k, v, q, decay, S_ckpt, d_out, d_S_final=None)

    t_tri_bwd = time_event(triton_bwd_only, iters=20, warmup=5)
    print(f"  Triton backward only:       {t_tri_bwd:7.3f} ms/call")

    # ---- 3) Triton autograd fwd+bwd ----
    S0_g, k_g, v_g, q_g, decay_g = make_inputs(B, T, H, N, V, DTYPE, device, requires_grad=True)
    g_out_static = torch.randn((T, B, H, V), device=device, dtype=DTYPE)

    def triton_fwd_bwd_autograd():
        out, S_final = e88_triton(S0_g, k_g, v_g, q_g, decay_g)
        # Sum is irrelevant; we want fwd+bwd timing.
        out.backward(g_out_static, retain_graph=False)
        # zero grads to allow next call
        for p in (S0_g, k_g, v_g, q_g, decay_g):
            p.grad = None

    t_tri_full = time_event(triton_fwd_bwd_autograd, iters=10, warmup=3)
    print(f"  Triton autograd fwd+bwd:    {t_tri_full:7.3f} ms/call")
    print(f"    overhead vs raw kernels:  {t_tri_full - t_tri_fwd - t_tri_bwd:7.3f} ms")

    # ---- 4) CUDA reg-own forward + backward (kernel-only) ----
    if hasty_pytorch_lib is None:
        print("\n  CUDA bench skipped (lib not loaded)")
        return

    # CUDA optimized path uses [B, T, H, *] layout.
    k_bt = k.transpose(0, 1).contiguous()
    v_bt = v.transpose(0, 1).contiguous()
    q_bt = q.transpose(0, 1).contiguous()
    decay_bt = decay.transpose(0, 1).contiguous()
    S0_bt = S0.contiguous()
    g_empty = torch.empty(0, device=device, dtype=DTYPE)

    checkpoint_interval = 16
    num_checkpoints = (T + checkpoint_interval - 1) // checkpoint_interval + 1
    cache_size = num_checkpoints * B * H * N * V + B * T * H * V

    output_bt = torch.empty(B, T, H, V, device=device, dtype=DTYPE)
    S_cache_bt = torch.empty(cache_size, device=device, dtype=DTYPE)

    def cuda_fwd_only():
        hasty_pytorch_lib.e88_warp_optimized_forward(
            True, k_bt, v_bt, q_bt, decay_bt, g_empty,
            S0_bt, output_bt, S_cache_bt, H, False, False,
            checkpoint_interval,
        )

    t_cuda_fwd = time_event(cuda_fwd_only, iters=20, warmup=5)
    print(f"\n  CUDA fwd only (warp_opt):   {t_cuda_fwd:7.3f} ms/call")

    # CUDA backward (register-owned)
    cuda_fwd_only()  # populate S_cache
    d_out_bt = torch.randn(B, T, H, V, device=device, dtype=DTYPE)
    d_k_cu = torch.empty_like(k_bt)
    d_v_cu = torch.empty_like(v_bt)
    d_q_cu = torch.empty_like(q_bt)
    d_decay_cu = torch.empty_like(decay_bt)
    d_g_cu = torch.empty(0, device=device, dtype=DTYPE)
    cache_entry_size = N * V + N + V + 1
    segment_cache = torch.empty(
        B * H * checkpoint_interval * cache_entry_size,
        dtype=DTYPE, device=device,
    )

    def cuda_bwd_only():
        hasty_pytorch_lib.e88_register_owned_backward(
            k_bt, v_bt, q_bt, decay_bt, g_empty,
            S_cache_bt, d_out_bt,
            d_k_cu, d_v_cu, d_q_cu, d_decay_cu, d_g_cu,
            segment_cache, H, False, False,
            checkpoint_interval,
        )

    t_cuda_bwd = time_event(cuda_bwd_only, iters=20, warmup=5)
    print(f"  CUDA bwd only (reg-own):    {t_cuda_bwd:7.3f} ms/call")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"Per-call summary (B={B} T={T} H={H} N={N} V={V}):")
    print(f"  Triton fwd:     {t_tri_fwd:7.3f} ms")
    print(f"  Triton bwd:     {t_tri_bwd:7.3f} ms")
    print(f"  Triton fwd+bwd: {t_tri_fwd + t_tri_bwd:7.3f} ms (sum)")
    print(f"  Triton autograd:{t_tri_full:7.3f} ms (e88_triton.apply with backward)")
    print()
    print(f"  CUDA fwd:       {t_cuda_fwd:7.3f} ms")
    print(f"  CUDA bwd:       {t_cuda_bwd:7.3f} ms")
    print(f"  CUDA fwd+bwd:   {t_cuda_fwd + t_cuda_bwd:7.3f} ms (sum)")
    print()
    print(f"  Triton/CUDA ratio fwd:     {t_tri_fwd/t_cuda_fwd:.2f}x")
    print(f"  Triton/CUDA ratio bwd:     {t_tri_bwd/t_cuda_bwd:.2f}x")
    print(f"  Triton/CUDA ratio fwd+bwd: {(t_tri_fwd+t_tri_bwd)/(t_cuda_fwd+t_cuda_bwd):.2f}x")

    # Estimate per-step contribution at depth=14 with grad_ckpt
    # (1 fwd + 1 fwd-replay-during-bwd + 1 bwd per layer)
    triton_per_step = DEPTH * (2 * t_tri_fwd + t_tri_bwd)
    cuda_per_step = DEPTH * (2 * t_cuda_fwd + t_cuda_bwd)
    print()
    print(f"Estimated recurrence time per training step (depth={DEPTH}, grad_ckpt):")
    print(f"  Triton: {triton_per_step:.1f} ms ({DEPTH} layers x (2 fwd + bwd))")
    print(f"  CUDA:   {cuda_per_step:.1f} ms")
    print(f"  Ratio:  {triton_per_step/cuda_per_step:.2f}x")
    print()
    print(f"At observed Triton ~3728 tok/s, full step time = {B*T/3728*1000:.1f} ms")
    print(f"At observed CUDA   ~4501 tok/s, full step time = {B*T/4501*1000:.1f} ms")
    print(f"Recurrence as fraction of step:")
    print(f"  Triton: {triton_per_step / (B*T/3728*1000) * 100:.0f}%")
    print(f"  CUDA:   {cuda_per_step / (B*T/4501*1000) * 100:.0f}%")


if __name__ == "__main__":
    main()
