"""Parity tests for the Triton E88 forward kernel.

Compares the Triton implementation in
``elman.triton.e88_triton_forward`` against a pure-PyTorch reference
that mirrors the slow fallback in
``elman.models.e88_fla_hybrid.E88FLAHybrid``.
"""
from __future__ import absolute_import

import os
import sys
import time

# Make sure we can import the elman package when running this file as a
# script (pytest already handles this when invoked from the repo root).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest
import torch

from elman.triton.e88_triton_forward import (
    e88_triton_forward,
    e88_torch_reference,
    DEFAULT_CKPT_INTERVAL,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_inputs(B, T, H, N, V, dtype, device, seed=0):
    """Build a realistic E88 input batch.

    k and q are L2-normalized to match the production usage. decay is
    sampled in (0, 1) to stay in the stable regime expected by E88.
    """
    g = torch.Generator(device=device).manual_seed(seed)

    # Use fp32 for sampling, then cast — gives identical inputs across dtypes.
    k = torch.randn((T, B, H, N), generator=g, device=device, dtype=torch.float32)
    q = torch.randn((T, B, H, N), generator=g, device=device, dtype=torch.float32)
    v = 0.3 * torch.randn((T, B, H, V), generator=g, device=device, dtype=torch.float32)

    # L2 normalize k, q (matches production E88 with use_l2_norm=True).
    k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)

    # decay in (0, 1) — using exp(-softplus(.)) shape: in [0, 1].
    decay_logits = torch.randn((T, B, H), generator=g, device=device, dtype=torch.float32)
    decay = torch.exp(-torch.nn.functional.softplus(decay_logits))

    S0 = 0.05 * torch.randn((B, H, N, V), generator=g, device=device, dtype=torch.float32)

    return (
        S0.to(dtype).contiguous(),
        k.to(dtype).contiguous(),
        v.to(dtype).contiguous(),
        q.to(dtype).contiguous(),
        decay.to(dtype).contiguous(),
    )


def _check_close(name, a, b, atol, rtol):
    diff = (a.float() - b.float()).abs()
    max_abs = diff.max().item()
    denom = b.float().abs().max().item()
    max_rel = max_abs / denom if denom > 0 else max_abs
    ok = torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol)
    status = "PASS" if ok else "FAIL"
    print(f"  {name}: max_abs={max_abs:.3e} max_rel={max_rel:.3e} [{status}]")
    return ok, max_abs, max_rel


def _subsample_dense_ckpt(ckpt_dense, ckpt_interval):
    """Pick out the slots in a dense [T+1, ...] checkpoint that correspond
    to the sparse kernel layout: slot 0 = S0, slot k = S after step
    (k*ckpt_interval - 1) for k >= 1.

    Returns a tensor shaped [num_ckpts, ...] matching the sparse layout.
    """
    T = ckpt_dense.shape[0] - 1
    num_ckpts = T // ckpt_interval + 1
    # Indices into the dense ckpt (which has T+1 slots, slot 0 = S0,
    # slot t+1 = S after step t).
    idxs = [0] + [k * ckpt_interval for k in range(1, num_ckpts)]
    return ckpt_dense[idxs]


# ---------------------------------------------------------------------------
# Parametrized tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda")


SHAPE_CONFIGS = [
    # (N, V) — kernel currently supports both <= 64.
    (16, 16),
    (32, 32),
    (32, 64),
]


@pytest.mark.parametrize("N,V", SHAPE_CONFIGS)
def test_e88_triton_forward_parity_bf16(device, N, V):
    B, T, H = 2, 64, 4
    dtype = torch.bfloat16
    S0, k, v, q, decay = _make_inputs(B, T, H, N, V, dtype, device, seed=0)

    out_ref, S_final_ref, ckpt_ref = e88_torch_reference(S0, k, v, q, decay)
    out_tri, S_final_tri, ckpt_tri = e88_triton_forward(S0, k, v, q, decay)

    # Tolerances: bf16 has ~1e-3 unit roundoff, and we accumulate over T
    # steps with chained tanh, so 1e-2 is the right ballpark.
    atol, rtol = 1e-2, 1e-2
    ok_o, _, _ = _check_close(f"out  bf16 N={N} V={V}", out_tri, out_ref, atol, rtol)
    ok_s, _, _ = _check_close(f"Sfin bf16 N={N} V={V}", S_final_tri, S_final_ref, atol, rtol)
    # Sparse-checkpoint: kernel only stores S every CKPT_INTERVAL steps.
    # Compare against the equivalent slots from the dense reference.
    ckpt_ref_sparse = _subsample_dense_ckpt(ckpt_ref, DEFAULT_CKPT_INTERVAL)
    assert ckpt_tri.shape == ckpt_ref_sparse.shape, (
        f"sparse ckpt shape mismatch: {ckpt_tri.shape} vs {ckpt_ref_sparse.shape}"
    )
    ok_c, _, _ = _check_close(
        f"ckpt(sparse) bf16 N={N} V={V}", ckpt_tri, ckpt_ref_sparse, atol, rtol,
    )
    assert ok_o and ok_s and ok_c, "bf16 parity failed"


@pytest.mark.parametrize("N,V", SHAPE_CONFIGS)
def test_e88_triton_forward_parity_fp32(device, N, V):
    B, T, H = 2, 64, 4
    dtype = torch.float32
    S0, k, v, q, decay = _make_inputs(B, T, H, N, V, dtype, device, seed=1)

    out_ref, S_final_ref, ckpt_ref = e88_torch_reference(S0, k, v, q, decay)
    out_tri, S_final_tri, ckpt_tri = e88_triton_forward(S0, k, v, q, decay)

    # fp32 — both impls use fp32 accumulators, so the only sources of
    # difference are reduction order in tl.sum vs einsum and tanh impls.
    # 1e-5 absolute is realistic; we relax rtol slightly because some
    # outputs have tiny magnitudes.
    atol, rtol = 1e-5, 1e-4
    ok_o, _, _ = _check_close(f"out  fp32 N={N} V={V}", out_tri, out_ref, atol, rtol)
    ok_s, _, _ = _check_close(f"Sfin fp32 N={N} V={V}", S_final_tri, S_final_ref, atol, rtol)
    ckpt_ref_sparse = _subsample_dense_ckpt(ckpt_ref, DEFAULT_CKPT_INTERVAL)
    assert ckpt_tri.shape == ckpt_ref_sparse.shape, (
        f"sparse ckpt shape mismatch: {ckpt_tri.shape} vs {ckpt_ref_sparse.shape}"
    )
    ok_c, _, _ = _check_close(
        f"ckpt(sparse) fp32 N={N} V={V}", ckpt_tri, ckpt_ref_sparse, atol, rtol,
    )
    assert ok_o and ok_s and ok_c, "fp32 parity failed"


# ---------------------------------------------------------------------------
# Timing comparison (also runs as part of `python tests/...`).
# ---------------------------------------------------------------------------

def _bench(fn, iters=3, warmup=1):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def test_e88_triton_forward_timing_t512(device):
    B, T, H, N, V = 2, 512, 4, 32, 32
    dtype = torch.bfloat16
    S0, k, v, q, decay = _make_inputs(B, T, H, N, V, dtype, device, seed=42)

    # Verify parity first at this scale (use slightly looser tolerance —
    # T=512 is a long chain of tanh/decay accumulations in bf16).
    out_ref, _, _ = e88_torch_reference(S0, k, v, q, decay)
    out_tri, _, _ = e88_triton_forward(S0, k, v, q, decay)
    diff = (out_ref.float() - out_tri.float()).abs().max().item()
    assert diff < 5e-2, f"T=512 bf16 parity drift too large: {diff:.3e}"

    t_ref = _bench(lambda: e88_torch_reference(S0, k, v, q, decay), iters=3, warmup=1)
    t_tri = _bench(lambda: e88_triton_forward(S0, k, v, q, decay), iters=10, warmup=3)

    speedup = t_ref / t_tri
    print()
    print(f"  T=512 timing  (B={B} H={H} N={N} V={V}, bf16):")
    print(f"    PyTorch ref : {t_ref*1e3:8.2f} ms / iter")
    print(f"    Triton      : {t_tri*1e3:8.2f} ms / iter")
    print(f"    speedup     : {speedup:8.2f}x")


# ---------------------------------------------------------------------------
# Script entrypoint
# ---------------------------------------------------------------------------

def _run_all_as_script():
    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(1)
    dev = torch.device("cuda")

    print("E88 Triton forward parity tests\n")
    print("--- bf16 ---")
    all_ok = True
    for (N, V) in SHAPE_CONFIGS:
        try:
            test_e88_triton_forward_parity_bf16.__wrapped__(dev, N, V)
        except AttributeError:
            # pytest.mark wraps it in a way that may or may not expose __wrapped__;
            # fall back to inlining the body manually.
            B, T, H = 2, 64, 4
            S0, k, v, q, decay = _make_inputs(B, T, H, N, V, torch.bfloat16, dev, seed=0)
            out_ref, S_final_ref, ckpt_ref = e88_torch_reference(S0, k, v, q, decay)
            out_tri, S_final_tri, ckpt_tri = e88_triton_forward(S0, k, v, q, decay)
            ok_o, _, _ = _check_close(f"out  bf16 N={N} V={V}", out_tri, out_ref, 1e-2, 1e-2)
            ok_s, _, _ = _check_close(f"Sfin bf16 N={N} V={V}", S_final_tri, S_final_ref, 1e-2, 1e-2)
            ckpt_ref_sparse = _subsample_dense_ckpt(ckpt_ref, DEFAULT_CKPT_INTERVAL)
            ok_c, _, _ = _check_close(
                f"ckpt(sparse) bf16 N={N} V={V}", ckpt_tri, ckpt_ref_sparse, 1e-2, 1e-2
            )
            all_ok = all_ok and ok_o and ok_s and ok_c
        except AssertionError as e:
            print(f"  FAIL: {e}")
            all_ok = False

    print("\n--- fp32 ---")
    for (N, V) in SHAPE_CONFIGS:
        B, T, H = 2, 64, 4
        S0, k, v, q, decay = _make_inputs(B, T, H, N, V, torch.float32, dev, seed=1)
        out_ref, S_final_ref, ckpt_ref = e88_torch_reference(S0, k, v, q, decay)
        out_tri, S_final_tri, ckpt_tri = e88_triton_forward(S0, k, v, q, decay)
        ok_o, _, _ = _check_close(f"out  fp32 N={N} V={V}", out_tri, out_ref, 1e-5, 1e-4)
        ok_s, _, _ = _check_close(f"Sfin fp32 N={N} V={V}", S_final_tri, S_final_ref, 1e-5, 1e-4)
        ckpt_ref_sparse = _subsample_dense_ckpt(ckpt_ref, DEFAULT_CKPT_INTERVAL)
        ok_c, _, _ = _check_close(
            f"ckpt(sparse) fp32 N={N} V={V}", ckpt_tri, ckpt_ref_sparse, 1e-5, 1e-4
        )
        all_ok = all_ok and ok_o and ok_s and ok_c

    print("\n--- timing ---")
    test_e88_triton_forward_timing_t512(dev)

    print()
    print("ALL PASS" if all_ok else "SOME FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    _run_all_as_script()
