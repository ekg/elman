"""Backward perf comparison: Triton vs CUDA E88.

Calls the CUDA forward (e88_fla_hybrid_forward, no-gate no-norm path) to
produce a valid S_cache, then times the CUDA backward
(e88_fla_hybrid_backward) against our Triton backward across a few
shapes that span the regimes we actually care about:

  - short, batched:   B=16, T=512     (production pretraining chunk)
  - mid, single:      B=1,  T=4K
  - long, single:     B=1,  T=16K     (long-context training regime)

H, N, V default to the production E88 config (H=83, N=32, V=32) but the
CLI lets you override.
"""
from __future__ import absolute_import

import argparse
import os
import sys
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from elman.triton.e88_triton_forward import e88_triton_forward
from elman.triton.e88_triton_backward import e88_triton_backward

# Try to import the CUDA library (top-level "hasty_pytorch_lib", as in
# elman.models.e88_fla_hybrid).
try:
    import hasty_pytorch_lib
    CUDA_OK = True
except Exception as e:
    print(f"[warn] CUDA E88 lib not loaded: {e}")
    hasty_pytorch_lib = None
    CUDA_OK = False


def _make_inputs(B, T, H, N, V, dtype, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    k = torch.randn((T, B, H, N), generator=g, device=device, dtype=torch.float32)
    q = torch.randn((T, B, H, N), generator=g, device=device, dtype=torch.float32)
    v = 0.3 * torch.randn((T, B, H, V), generator=g, device=device, dtype=torch.float32)
    k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
    decay_logits = torch.randn((T, B, H), generator=g, device=device, dtype=torch.float32)
    decay = torch.exp(-torch.nn.functional.softplus(decay_logits))
    S0 = torch.zeros((B, H, N, V), dtype=torch.float32, device=device)
    d_out = torch.randn((T, B, H, V), generator=g, device=device, dtype=torch.float32)
    return (
        S0.to(dtype).contiguous(),
        k.to(dtype).contiguous(),
        v.to(dtype).contiguous(),
        q.to(dtype).contiguous(),
        decay.to(dtype).contiguous(),
        d_out.to(dtype).contiguous(),
    )


def _bench(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def bench_one(B, T, H, N, V, dtype, device):
    print(f"\n=== B={B} T={T} H={H} N={N} V={V} dtype={dtype} ===")
    S0, k, v, q, decay, d_out = _make_inputs(B, T, H, N, V, dtype, device, seed=0)

    # ---- Triton path ----------------------------------------------------
    out_tri, S_final_tri, S_ckpt = e88_triton_forward(S0, k, v, q, decay)

    # Warm + bench Triton backward.
    def run_triton_bwd():
        e88_triton_backward(k, v, q, decay, S_ckpt, d_out, d_S_final=None)
    t_tri = _bench(run_triton_bwd, iters=10, warmup=3)
    print(f"  Triton bwd    : {t_tri*1e3:8.3f} ms / iter")

    # ---- CUDA paths -----------------------------------------------------
    # Two CUDA backward paths to compare:
    #   (1) Legacy fused: e88_fla_hybrid_forward + e88_fla_hybrid_backward
    #       in [T, B, H, *] layout.
    #   (2) Register-owned (production path for N,V<=32): pairs with
    #       e88_warp_optimized_forward in [B, T, H, *] layout.
    t_cuda_legacy = float("nan")
    t_cuda_reg = float("nan")

    # ---- Legacy [T, B, H, *] path --------------------------------------
    if CUDA_OK and hasattr(hasty_pytorch_lib, "e88_fla_hybrid_forward") \
            and hasattr(hasty_pytorch_lib, "e88_fla_hybrid_backward"):
        results = hasty_pytorch_lib.e88_fla_hybrid_forward(
            True, k, v, q, decay, S0, H,
        )
        S_cache = results[2]

        checkpoint_interval = 16
        num_checkpoints = (T + checkpoint_interval - 1) // checkpoint_interval + 1
        ck_size = num_checkpoints * B * H * N * V
        sq_size = T * B * H * V
        S_checkpoints = S_cache[:ck_size].view(num_checkpoints, B, H, N, V)
        Sq_cache = S_cache[ck_size:ck_size + sq_size].view(T, B, H, V)

        def run_cuda_bwd_legacy():
            hasty_pytorch_lib.e88_fla_hybrid_backward(
                k, v, q, decay, S_checkpoints, Sq_cache,
                d_out.contiguous(), H,
            )
        for _ in range(3):
            run_cuda_bwd_legacy()
        torch.cuda.synchronize()
        t_cuda_legacy = _bench(run_cuda_bwd_legacy, iters=10, warmup=3)
        print(f"  CUDA legacy   : {t_cuda_legacy*1e3:8.3f} ms / iter")
    else:
        print("  CUDA legacy   : (unavailable)")

    # ---- Register-owned [B, T, H, *] path ------------------------------
    if CUDA_OK and hasattr(hasty_pytorch_lib, "e88_register_owned_backward") \
            and hasattr(hasty_pytorch_lib, "e88_warp_optimized_forward") \
            and N <= 32 and V <= 32:
        # Re-shape inputs to [B, T, H, *] for the optimized forward.
        k_bt = k.transpose(0, 1).contiguous()        # [B, T, H, N]
        v_bt = v.transpose(0, 1).contiguous()
        q_bt = q.transpose(0, 1).contiguous()
        decay_bt = decay.transpose(0, 1).contiguous()  # [B, T, H]
        d_out_bt = d_out.transpose(0, 1).contiguous()  # [B, T, H, V]
        S0_bt = S0.contiguous()                       # already [B, H, N, V]

        checkpoint_interval = 16
        num_checkpoints = (T + checkpoint_interval - 1) // checkpoint_interval + 1
        cache_size = num_checkpoints * B * H * N * V + B * T * H * V
        output_bt = torch.empty(B, T, H, V, device=device, dtype=dtype)
        S_cache_bt = torch.empty(cache_size, device=device, dtype=dtype)
        g_empty = torch.empty(0, device=device, dtype=dtype)

        # Apply the warp-optimized forward (apply_gate=False, normalize_kq=False).
        hasty_pytorch_lib.e88_warp_optimized_forward(
            True, k_bt, v_bt, q_bt, decay_bt, g_empty,
            S0_bt, output_bt, S_cache_bt, H, False, False,
            checkpoint_interval,
        )

        d_k = torch.empty_like(k_bt)
        d_v = torch.empty_like(v_bt)
        d_q = torch.empty_like(q_bt)
        d_decay = torch.empty_like(decay_bt)
        d_g = torch.empty(0, device=device, dtype=dtype)
        cache_entry_size = N * V + N + V + 1
        segment_cache = torch.empty(
            B * H * checkpoint_interval * cache_entry_size,
            dtype=dtype, device=device,
        )

        def run_cuda_bwd_reg():
            hasty_pytorch_lib.e88_register_owned_backward(
                k_bt, v_bt, q_bt, decay_bt, g_empty,
                S_cache_bt, d_out_bt,
                d_k, d_v, d_q, d_decay, d_g,
                segment_cache, H, False, False,
                checkpoint_interval,
            )
        for _ in range(3):
            run_cuda_bwd_reg()
        torch.cuda.synchronize()
        t_cuda_reg = _bench(run_cuda_bwd_reg, iters=10, warmup=3)
        print(f"  CUDA reg-own  : {t_cuda_reg*1e3:8.3f} ms / iter")
    else:
        print("  CUDA reg-own  : (unsupported size or symbol missing)")

    # ---- Ratios ---------------------------------------------------------
    def _ratio(t_cuda):
        return t_cuda / t_tri if t_cuda == t_cuda else float("nan")

    r_legacy = _ratio(t_cuda_legacy)
    r_reg = _ratio(t_cuda_reg)
    if r_legacy == r_legacy:
        print(f"  legacy/Triton : {r_legacy:6.2f}x")
    if r_reg == r_reg:
        print(f"  reg-own/Triton: {r_reg:6.2f}x")

    return t_tri, t_cuda_legacy, t_cuda_reg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--H", type=int, default=83)
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--V", type=int, default=32)
    ap.add_argument("--shapes", nargs="+", default=None,
                    help="space-separated B,T pairs; default: 16,512 1,4096 1,16384")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required")
        sys.exit(1)
    device = torch.device("cuda")
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    if args.shapes is None:
        shapes = [(16, 512), (1, 4096), (1, 16384)]
    else:
        shapes = [tuple(int(x) for x in s.split(",")) for s in args.shapes]

    print(f"E88 backward perf — Triton vs CUDA")
    print(f"  fixed: H={args.H} N={args.N} V={args.V} dtype={args.dtype}")
    print(f"  shapes (B, T): {shapes}")

    rows = []
    for B, T in shapes:
        t_tri, t_cuda_legacy, t_cuda_reg = bench_one(
            B, T, args.H, args.N, args.V, dtype, device,
        )
        rows.append((B, T, t_tri, t_cuda_legacy, t_cuda_reg))

    def _fmt_ms(x):
        return "n/a" if x != x else f"{x*1e3:.3f}"

    def _fmt_ratio(t_c, t_t):
        return "n/a" if t_c != t_c else f"{t_c/t_t:.2f}x"

    print()
    print("Summary")
    print(f"  {'B':>3}  {'T':>6}  {'Triton ms':>10}  {'legacy ms':>10}  {'reg-own ms':>11}  {'leg/tri':>8}  {'reg/tri':>8}")
    for B, T, t_tri, t_l, t_r in rows:
        print(f"  {B:>3}  {T:>6}  {t_tri*1e3:>10.3f}  {_fmt_ms(t_l):>10}  {_fmt_ms(t_r):>11}  "
              f"{_fmt_ratio(t_l, t_tri):>8}  {_fmt_ratio(t_r, t_tri):>8}")


if __name__ == "__main__":
    main()
