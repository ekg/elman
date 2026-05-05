"""End-to-end smoke test: E88FLAHybrid with use_triton=True vs CUDA path.

Builds two identical E88FLAHybrid layers (same weights, same input), one
with use_triton=False (CUDA reg-own) and one with use_triton=True
(Triton fwd+bwd). Compares forward output and gradients of all
parameters after a single backward pass.

This is the gating test for swapping backends in real training.
"""
from __future__ import absolute_import

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from elman.models.e88_fla_hybrid import E88FLAHybrid


def _check(name, a, b, atol, rtol):
    diff = (a.float() - b.float()).abs()
    max_abs = diff.max().item()
    denom = b.float().abs().max().item()
    max_rel = max_abs / denom if denom > 0 else max_abs
    ok = torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol)
    status = "PASS" if ok else "FAIL"
    print(f"  {name}: max_abs={max_abs:.3e} max_rel={max_rel:.3e} [{status}]")
    return ok


def main():
    if not torch.cuda.is_available():
        print("CUDA required")
        sys.exit(1)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    B, T = 2, 64
    dim = 256
    n_state = 32
    n_heads = 8

    # Build two layers with identical config + identical weights.
    torch.manual_seed(11)
    layer_cuda = E88FLAHybrid(
        dim=dim, n_state=n_state, n_heads=n_heads,
        use_l2_norm=True, use_gate=True, gate_activation="silu",
        use_triton=False,
    ).to(device).to(dtype)

    torch.manual_seed(11)
    layer_tri = E88FLAHybrid(
        dim=dim, n_state=n_state, n_heads=n_heads,
        use_l2_norm=True, use_gate=True, gate_activation="silu",
        use_triton=True,
    ).to(device).to(dtype)

    # Sanity-check weight parity.
    n_params_diff = 0
    for (name_c, p_c), (name_t, p_t) in zip(
        layer_cuda.named_parameters(), layer_tri.named_parameters()
    ):
        if not torch.equal(p_c.detach(), p_t.detach()):
            n_params_diff += 1
    if n_params_diff > 0:
        print(f"[warn] {n_params_diff} weights differ between layers")

    # Same input.
    g = torch.Generator(device=device).manual_seed(99)
    x = torch.randn((B, T, dim), generator=g, device=device, dtype=dtype)
    x_c = x.clone().detach().requires_grad_(True)
    x_t = x.clone().detach().requires_grad_(True)

    # Forward.
    layer_cuda.train()
    layer_tri.train()
    out_c, _ = layer_cuda(x_c)
    out_t, _ = layer_tri(x_t)

    print("Forward parity (E88 layer use_triton vs CUDA):")
    ok = True
    ok &= _check("output", out_t, out_c, 5e-2, 5e-2)

    # Backward.
    g_out = torch.randn_like(out_c)
    out_c.backward(g_out, retain_graph=False)
    out_t.backward(g_out, retain_graph=False)

    print("\nBackward parity (input + parameter grads):")
    ok &= _check("d_x", x_t.grad, x_c.grad, 1e-1, 1e-1)

    n_param_grad_diffs = 0
    for (name_c, p_c), (name_t, p_t) in zip(
        layer_cuda.named_parameters(), layer_tri.named_parameters()
    ):
        if p_c.grad is None and p_t.grad is None:
            continue
        if p_c.grad is None or p_t.grad is None:
            print(f"  {name_c}: grad missing (cuda={p_c.grad is not None}, tri={p_t.grad is not None})")
            n_param_grad_diffs += 1
            continue
        diff = (p_c.grad.float() - p_t.grad.float()).abs().max().item()
        denom = p_c.grad.float().abs().max().item()
        rel = diff / denom if denom > 0 else diff
        if rel > 5e-2:
            print(f"  {name_c}: max_abs={diff:.3e} max_rel={rel:.3e} [FAIL]")
            n_param_grad_diffs += 1
        else:
            print(f"  {name_c}: max_abs={diff:.3e} max_rel={rel:.3e} [PASS]")
    if n_param_grad_diffs > 0:
        print(f"\n{n_param_grad_diffs} parameter grad mismatches")
        ok = False

    print()
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
