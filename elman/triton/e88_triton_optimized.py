"""Drop-in Triton replacement for E88OptimizedCUDAFunction.

Mirrors the call signature of
``elman.models.e88_fla_hybrid.E88OptimizedCUDAFunction.apply`` but uses
the Triton forward + backward kernels under the hood. Gate (silu) and L2
normalization are applied as differentiable PyTorch ops outside the
kernel — the recurrence math is unchanged.

Layout: production E88 uses ``[B, T, H, *]``; the Triton kernels use
``[T, B, H, *]``, so we transpose at the boundary.

This wrapper is meant for parity / portability work — it should produce
the same loss as the CUDA path (within numerical tolerance) so you can
swap backends on the fly with ``use_triton=True``.
"""
from __future__ import absolute_import

from typing import Tuple

import torch

from .e88_triton_backward import e88_triton


def e88_triton_optimized_apply(
    training: bool,
    k: torch.Tensor,        # [B, T, H, n_state]
    v: torch.Tensor,        # [B, T, H, head_v_dim]
    q: torch.Tensor,        # [B, T, H, n_state]
    decay: torch.Tensor,    # [B, T, H]
    g: torch.Tensor = None, # [B, T, H, head_v_dim] gate (None or empty if no gate)
    S0: torch.Tensor = None,# [B, H, n_state, head_v_dim]
    n_heads: int = None,
    apply_gate: bool = True,
    normalize_kq: bool = False,
    checkpoint_interval: int = 16,  # ignored — Triton stores all checkpoints
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton-backed E88 recurrence with optional pre-norm and post-gate.

    Returns:
        S_final: [B, H, n_state, head_v_dim]
        output:  [B, T, H, head_v_dim]   (after gate if apply_gate=True)
    """
    assert k.dim() == 4 and v.dim() == 4 and q.dim() == 4, \
        "k/v/q must be [B, T, H, *]"
    B, T, H, N = k.shape
    Vsz = v.shape[-1]
    assert q.shape == (B, T, H, N)
    assert v.shape == (B, T, H, Vsz)
    assert decay.shape == (B, T, H)

    # Optional L2 normalization (matches the CUDA `normalize_kq` flag).
    if normalize_kq:
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)

    # Triton kernels expect [T, B, H, *].
    k_t = k.transpose(0, 1).contiguous()
    v_t = v.transpose(0, 1).contiguous()
    q_t = q.transpose(0, 1).contiguous()
    decay_t = decay.transpose(0, 1).contiguous()
    if S0 is None:
        S0 = torch.zeros(
            (B, H, N, Vsz), dtype=k.dtype, device=k.device,
        )

    # Triton forward + backward (autograd-wrapped).
    out_t, S_final = e88_triton(S0, k_t, v_t, q_t, decay_t)

    # Back to [B, T, H, V].
    output = out_t.transpose(0, 1).contiguous()

    # Post-gate: output = output * silu(g)
    if apply_gate and g is not None and getattr(g, "numel", lambda: 0)() > 0:
        output = output * torch.nn.functional.silu(g)

    return S_final, output
