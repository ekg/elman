"""Parity tests for the Triton E88 backward kernel.

Validates that ``e88_triton_backward`` matches the gradients produced by
torch.autograd through the slow PyTorch reference recurrence.

Tested across multiple (N, V) shapes in both bf16 and fp32, plus an
end-to-end ``E88TritonFunction.apply`` (autograd) parity check.
"""
from __future__ import absolute_import

import os
import sys
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest
import torch

from elman.triton.e88_triton_forward import (
    e88_triton_forward,
    e88_torch_reference,
)
from elman.triton.e88_triton_backward import (
    e88_triton_backward,
    e88_triton,
    E88TritonFunction,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_inputs(B, T, H, N, V, dtype, device, seed=0, requires_grad=True):
    """Build inputs that exercise the backward chain rule.

    For numerical stability the test uses fp32 reference inputs — bf16
    parity is then checked by casting the gradient outputs back to fp32.
    """
    g = torch.Generator(device=device).manual_seed(seed)

    k = torch.randn((T, B, H, N), generator=g, device=device, dtype=torch.float32)
    q = torch.randn((T, B, H, N), generator=g, device=device, dtype=torch.float32)
    v = 0.3 * torch.randn((T, B, H, V), generator=g, device=device, dtype=torch.float32)

    k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)

    decay_logits = torch.randn((T, B, H), generator=g, device=device, dtype=torch.float32)
    decay = torch.exp(-torch.nn.functional.softplus(decay_logits))

    S0 = 0.05 * torch.randn((B, H, N, V), generator=g, device=device, dtype=torch.float32)

    # Cast to target dtype, then require_grad on the cast tensors.
    S0 = S0.to(dtype).contiguous().requires_grad_(requires_grad)
    k = k.to(dtype).contiguous().requires_grad_(requires_grad)
    v = v.to(dtype).contiguous().requires_grad_(requires_grad)
    q = q.to(dtype).contiguous().requires_grad_(requires_grad)
    decay = decay.to(dtype).contiguous().requires_grad_(requires_grad)
    return S0, k, v, q, decay


def _grad_pt_reference(S0, k, v, q, decay, d_out, d_S_final):
    """Run the PyTorch reference forward + autograd backward.

    Returns (d_k, d_v, d_q, d_decay, d_S0).
    """
    out_ref, S_final_ref, _ = e88_torch_reference(S0, k, v, q, decay)

    grads = torch.autograd.grad(
        outputs=[out_ref, S_final_ref],
        inputs=[k, v, q, decay, S0],
        grad_outputs=[d_out, d_S_final],
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )
    d_k, d_v, d_q, d_decay, d_S0 = grads
    return d_k, d_v, d_q, d_decay, d_S0


def _check(name, a, b, atol, rtol):
    diff = (a.float() - b.float()).abs()
    max_abs = diff.max().item()
    denom = b.float().abs().max().item()
    max_rel = max_abs / denom if denom > 0 else max_abs
    ok = torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol)
    status = "PASS" if ok else "FAIL"
    print(f"  {name}: max_abs={max_abs:.3e} max_rel={max_rel:.3e} [{status}]")
    return ok


# ---------------------------------------------------------------------------
# Parametrized tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda")


SHAPE_CONFIGS = [
    (16, 16),
    (32, 32),
    (32, 64),
]


def _grad_triton_direct(S0, k, v, q, decay, d_out, d_S_final):
    """Run forward (Triton) + backward (Triton kernel directly).

    No autograd — calls the kernel functions explicitly. Detached inputs.
    """
    out, S_final, S_ckpt = e88_triton_forward(
        S0.detach(), k.detach(), v.detach(), q.detach(), decay.detach()
    )
    d_k, d_v, d_q, d_decay, d_S0 = e88_triton_backward(
        k.detach(), v.detach(), q.detach(), decay.detach(),
        S_ckpt, d_out, d_S_final,
    )
    return d_k, d_v, d_q, d_decay, d_S0


@pytest.mark.parametrize("N,V", SHAPE_CONFIGS)
def test_e88_triton_backward_parity_fp32(device, N, V):
    B, T, H = 2, 32, 4
    dtype = torch.float32
    S0, k, v, q, decay = _make_inputs(B, T, H, N, V, dtype, device, seed=0)

    g = torch.Generator(device=device).manual_seed(11)
    d_out = torch.randn((T, B, H, V), generator=g, device=device, dtype=dtype)
    d_S_final = 0.1 * torch.randn((B, H, N, V), generator=g, device=device, dtype=dtype)

    # Reference via autograd.
    d_k_ref, d_v_ref, d_q_ref, d_decay_ref, d_S0_ref = _grad_pt_reference(
        S0, k, v, q, decay, d_out, d_S_final,
    )

    # Triton kernel directly.
    d_k, d_v, d_q, d_decay, d_S0 = _grad_triton_direct(
        S0, k, v, q, decay, d_out, d_S_final,
    )

    atol, rtol = 5e-5, 1e-4
    ok = True
    ok &= _check(f"d_k     fp32 N={N} V={V}", d_k,     d_k_ref,     atol, rtol)
    ok &= _check(f"d_v     fp32 N={N} V={V}", d_v,     d_v_ref,     atol, rtol)
    ok &= _check(f"d_q     fp32 N={N} V={V}", d_q,     d_q_ref,     atol, rtol)
    ok &= _check(f"d_decay fp32 N={N} V={V}", d_decay, d_decay_ref, atol, rtol)
    ok &= _check(f"d_S0    fp32 N={N} V={V}", d_S0,    d_S0_ref,    atol, rtol)
    assert ok


@pytest.mark.parametrize("N,V", SHAPE_CONFIGS)
def test_e88_triton_backward_parity_bf16(device, N, V):
    B, T, H = 2, 32, 4
    dtype = torch.bfloat16
    S0, k, v, q, decay = _make_inputs(B, T, H, N, V, dtype, device, seed=0)

    g = torch.Generator(device=device).manual_seed(11)
    d_out = torch.randn((T, B, H, V), generator=g, device=device, dtype=dtype)
    d_S_final = 0.1 * torch.randn((B, H, N, V), generator=g, device=device, dtype=dtype)

    d_k_ref, d_v_ref, d_q_ref, d_decay_ref, d_S0_ref = _grad_pt_reference(
        S0, k, v, q, decay, d_out, d_S_final,
    )
    d_k, d_v, d_q, d_decay, d_S0 = _grad_triton_direct(
        S0, k, v, q, decay, d_out, d_S_final,
    )

    # bf16: ~1e-2 absolute tolerance for grads — chained tanh' multiplies,
    # accumulates over T steps. The reference also accumulates in bf16,
    # so this measures kernel-vs-reference disagreement, not absolute drift.
    atol, rtol = 5e-2, 5e-2
    ok = True
    ok &= _check(f"d_k     bf16 N={N} V={V}", d_k,     d_k_ref,     atol, rtol)
    ok &= _check(f"d_v     bf16 N={N} V={V}", d_v,     d_v_ref,     atol, rtol)
    ok &= _check(f"d_q     bf16 N={N} V={V}", d_q,     d_q_ref,     atol, rtol)
    ok &= _check(f"d_decay bf16 N={N} V={V}", d_decay, d_decay_ref, atol, rtol)
    ok &= _check(f"d_S0    bf16 N={N} V={V}", d_S0,    d_S0_ref,    atol, rtol)
    assert ok


def test_e88_triton_autograd_parity_fp32(device):
    """End-to-end: E88TritonFunction.apply through autograd."""
    B, T, H, N, V = 2, 32, 4, 32, 32
    dtype = torch.float32
    S0, k, v, q, decay = _make_inputs(B, T, H, N, V, dtype, device, seed=42)

    g = torch.Generator(device=device).manual_seed(17)
    d_out = torch.randn((T, B, H, V), generator=g, device=device, dtype=dtype)
    d_S_final = 0.1 * torch.randn((B, H, N, V), generator=g, device=device, dtype=dtype)

    # Reference (PyTorch).
    out_ref, Sfin_ref, _ = e88_torch_reference(S0, k, v, q, decay)
    grads_ref = torch.autograd.grad(
        outputs=[out_ref, Sfin_ref],
        inputs=[k, v, q, decay, S0],
        grad_outputs=[d_out, d_S_final],
    )

    # Detach + re-create requires_grad inputs for the autograd-wrapped Triton path.
    S0_t = S0.detach().clone().requires_grad_(True)
    k_t  = k.detach().clone().requires_grad_(True)
    v_t  = v.detach().clone().requires_grad_(True)
    q_t  = q.detach().clone().requires_grad_(True)
    d_t  = decay.detach().clone().requires_grad_(True)

    out_tri, Sfin_tri = e88_triton(S0_t, k_t, v_t, q_t, d_t)
    grads_tri = torch.autograd.grad(
        outputs=[out_tri, Sfin_tri],
        inputs=[k_t, v_t, q_t, d_t, S0_t],
        grad_outputs=[d_out, d_S_final],
    )

    names = ["d_k", "d_v", "d_q", "d_decay", "d_S0"]
    ok = True
    for name, gt, gr in zip(names, grads_tri, grads_ref):
        ok &= _check(f"autograd {name} fp32", gt, gr, 5e-5, 1e-4)
    assert ok


# ---------------------------------------------------------------------------
# Script entrypoint
# ---------------------------------------------------------------------------

def _run_all_as_script():
    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(1)
    dev = torch.device("cuda")

    print("E88 Triton backward parity tests\n")

    print("--- fp32 ---")
    all_ok = True
    for (N, V) in SHAPE_CONFIGS:
        try:
            test_e88_triton_backward_parity_fp32.__wrapped__(dev, N, V)
        except (AttributeError, TypeError):
            B, T, H = 2, 32, 4
            S0, k, v, q, decay = _make_inputs(B, T, H, N, V, torch.float32, dev, seed=0)
            g = torch.Generator(device=dev).manual_seed(11)
            d_out = torch.randn((T, B, H, V), generator=g, device=dev, dtype=torch.float32)
            d_S_final = 0.1 * torch.randn((B, H, N, V), generator=g, device=dev, dtype=torch.float32)
            d_k_ref, d_v_ref, d_q_ref, d_decay_ref, d_S0_ref = _grad_pt_reference(
                S0, k, v, q, decay, d_out, d_S_final,
            )
            d_k, d_v, d_q, d_decay, d_S0 = _grad_triton_direct(
                S0, k, v, q, decay, d_out, d_S_final,
            )
            ok = True
            ok &= _check(f"d_k     fp32 N={N} V={V}", d_k,     d_k_ref,     5e-5, 1e-4)
            ok &= _check(f"d_v     fp32 N={N} V={V}", d_v,     d_v_ref,     5e-5, 1e-4)
            ok &= _check(f"d_q     fp32 N={N} V={V}", d_q,     d_q_ref,     5e-5, 1e-4)
            ok &= _check(f"d_decay fp32 N={N} V={V}", d_decay, d_decay_ref, 5e-5, 1e-4)
            ok &= _check(f"d_S0    fp32 N={N} V={V}", d_S0,    d_S0_ref,    5e-5, 1e-4)
            all_ok = all_ok and ok
        except AssertionError as e:
            print(f"  FAIL: {e}")
            all_ok = False

    print("\n--- bf16 ---")
    for (N, V) in SHAPE_CONFIGS:
        B, T, H = 2, 32, 4
        S0, k, v, q, decay = _make_inputs(B, T, H, N, V, torch.bfloat16, dev, seed=0)
        g = torch.Generator(device=dev).manual_seed(11)
        d_out = torch.randn((T, B, H, V), generator=g, device=dev, dtype=torch.bfloat16)
        d_S_final = 0.1 * torch.randn((B, H, N, V), generator=g, device=dev, dtype=torch.bfloat16)
        d_k_ref, d_v_ref, d_q_ref, d_decay_ref, d_S0_ref = _grad_pt_reference(
            S0, k, v, q, decay, d_out, d_S_final,
        )
        d_k, d_v, d_q, d_decay, d_S0 = _grad_triton_direct(
            S0, k, v, q, decay, d_out, d_S_final,
        )
        ok = True
        ok &= _check(f"d_k     bf16 N={N} V={V}", d_k,     d_k_ref,     5e-2, 5e-2)
        ok &= _check(f"d_v     bf16 N={N} V={V}", d_v,     d_v_ref,     5e-2, 5e-2)
        ok &= _check(f"d_q     bf16 N={N} V={V}", d_q,     d_q_ref,     5e-2, 5e-2)
        ok &= _check(f"d_decay bf16 N={N} V={V}", d_decay, d_decay_ref, 5e-2, 5e-2)
        ok &= _check(f"d_S0    bf16 N={N} V={V}", d_S0,    d_S0_ref,    5e-2, 5e-2)
        all_ok = all_ok and ok

    print("\n--- autograd end-to-end (fp32) ---")
    B, T, H, N, V = 2, 32, 4, 32, 32
    S0, k, v, q, decay = _make_inputs(B, T, H, N, V, torch.float32, dev, seed=42)
    g = torch.Generator(device=dev).manual_seed(17)
    d_out = torch.randn((T, B, H, V), generator=g, device=dev, dtype=torch.float32)
    d_S_final = 0.1 * torch.randn((B, H, N, V), generator=g, device=dev, dtype=torch.float32)
    out_ref, Sfin_ref, _ = e88_torch_reference(S0, k, v, q, decay)
    grads_ref = torch.autograd.grad(
        outputs=[out_ref, Sfin_ref],
        inputs=[k, v, q, decay, S0],
        grad_outputs=[d_out, d_S_final],
    )
    S0_t = S0.detach().clone().requires_grad_(True)
    k_t  = k.detach().clone().requires_grad_(True)
    v_t  = v.detach().clone().requires_grad_(True)
    q_t  = q.detach().clone().requires_grad_(True)
    d_t  = decay.detach().clone().requires_grad_(True)
    out_tri, Sfin_tri = e88_triton(S0_t, k_t, v_t, q_t, d_t)
    grads_tri = torch.autograd.grad(
        outputs=[out_tri, Sfin_tri],
        inputs=[k_t, v_t, q_t, d_t, S0_t],
        grad_outputs=[d_out, d_S_final],
    )
    names = ["d_k", "d_v", "d_q", "d_decay", "d_S0"]
    for name, gt, gr in zip(names, grads_tri, grads_ref):
        all_ok = all_ok and _check(f"autograd {name} fp32", gt, gr, 5e-5, 1e-4)

    print()
    print("ALL PASS" if all_ok else "SOME FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    _run_all_as_script()
