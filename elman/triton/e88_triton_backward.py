"""Triton backward kernel for E88 FLA-Hybrid recurrence.

This is Phase 2 of the Triton port. Mirrors
``elman/triton/e88_triton_forward.py`` (which produces output, S_final and
the full S_checkpoints tensor) and walks the reverse-time chain rule to
produce gradients with respect to k, v, q, decay, and S0.

Forward recurrence (per (b, h), see e88_triton_forward.py for context):
    r_t       = S_{t-1}.T @ k_t                     # [V]
    delta_t   = v_t - r_t
    pre_t     = decay_t * S_{t-1} + outer(k_t, delta_t)
    S_t       = tanh(pre_t)                          # [N, V]
    out_t     = S_t.T @ q_t                          # [V]

Backward (let upstream gradient be d_out, d_S_final):
    dS_t          = (carry from t+1) + outer(q_t, d_out_t)
    d_q_t         = S_t @ d_out_t
    d_pre_t       = dS_t * (1 - S_t**2)
    d_decay_t     = sum_{n,v} d_pre_t * S_{t-1}
    d_outer_t     = d_pre_t                          # [N, V]
    d_k_t (from outer)     = sum_v delta_t * d_outer_t
    d_delta_t              = sum_n k_t  * d_outer_t  # [V]
    d_v_t                  = d_delta_t
    d_retrieve_t           = -d_delta_t
    d_k_t (from retrieve)  = sum_v S_{t-1} * d_retrieve_t
    dS_{t-1}_from_decay    = decay_t * d_pre_t
    dS_{t-1}_from_retrieve = outer(k_t, d_retrieve_t)
    carry        = dS_{t-1}_from_decay + dS_{t-1}_from_retrieve
After the time loop, dS_carry is the gradient w.r.t. S0.

Shapes (matching the forward kernel layout):
    k, q:         [T, B, H, N]  bf16 or fp32
    v:            [T, B, H, V]
    decay:        [T, B, H]
    S_ckpt:       [T+1, B, H, N, V] from forward (reused as S_{t-1} reads).
    d_out:        [T, B, H, V]
    d_S_final:    [B, H, N, V]  (often zero; included for completeness).
Outputs:
    d_k:          [T, B, H, N]
    d_v:          [T, B, H, V]
    d_q:          [T, B, H, N]
    d_decay:      [T, B, H]
    d_S0:         [B, H, N, V]
"""
from __future__ import absolute_import

from typing import Tuple

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

@triton.jit
def _e88_backward_kernel(
    # Forward-side inputs (need re-reads).
    K_ptr,              # [T, B, H, N]
    V_ptr,              # [T, B, H, V]
    Q_ptr,              # [T, B, H, N]
    D_ptr,              # [T, B, H]
    Sckpt_ptr,          # [T+1, B, H, N, V]
    # Upstream grads.
    DOut_ptr,           # [T, B, H, V]
    DSfinal_ptr,        # [B, H, N, V]
    # Output grads.
    DK_ptr,             # [T, B, H, N]
    DV_ptr,             # [T, B, H, V]
    DQ_ptr,             # [T, B, H, N]
    DD_ptr,             # [T, B, H]
    DS0_ptr,            # [B, H, N, V]
    # Strides for every tensor (in elements).
    sk_t, sk_b, sk_h, sk_n,
    sv_t, sv_b, sv_h, sv_v,
    sq_t, sq_b, sq_h, sq_n,
    sd_t, sd_b, sd_h,
    sc_t, sc_b, sc_h, sc_n, sc_v,
    sdo_t, sdo_b, sdo_h, sdo_v,
    sdsf_b, sdsf_h, sdsf_n, sdsf_v,
    sdk_t, sdk_b, sdk_h, sdk_n,
    sdv_t, sdv_b, sdv_h, sdv_v,
    sdq_t, sdq_b, sdq_h, sdq_n,
    sdd_t, sdd_b, sdd_h,
    sds0_b, sds0_h, sds0_n, sds0_v,
    # Sizes.
    T: tl.constexpr, B: tl.constexpr, H: tl.constexpr,
    N: tl.constexpr, V: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """One program per (batch, head_block). Reverse-time loop."""
    b = tl.program_id(0).to(tl.int64)
    hg = tl.program_id(1).to(tl.int64)

    h_start = hg * BLOCK_H
    h_idx = h_start + tl.arange(0, BLOCK_H)
    h_mask = h_idx < H

    n_idx = tl.arange(0, BLOCK_N)
    v_idx = tl.arange(0, BLOCK_V)
    n_mask = n_idx < N
    v_mask = v_idx < V

    mask_hnv = (h_mask[:, None, None] & n_mask[None, :, None] & v_mask[None, None, :])
    mask_hn = h_mask[:, None] & n_mask[None, :]
    mask_hv = h_mask[:, None] & v_mask[None, :]

    # Initialize dS_carry from upstream d_S_final.
    dsf_off = (
        b * sdsf_b
        + h_idx[:, None, None] * sdsf_h
        + n_idx[None, :, None] * sdsf_n
        + v_idx[None, None, :] * sdsf_v
    )
    dS_carry = tl.load(DSfinal_ptr + dsf_off, mask=mask_hnv, other=0.0).to(tl.float32)

    # Reverse-time loop: t = T-1 .. 0.
    # Cast t / (t+1) to int64 below for offset arithmetic on large tensors
    # (S_ckpt at production T*B*H*N*V can exceed int32 range).
    for ti in range(T):
        t = T - 1 - ti
        t_i64 = tl.full([1], t, dtype=tl.int64)
        tp1_i64 = tl.full([1], t + 1, dtype=tl.int64)

        # Load forward inputs.
        k_off = (
            t_i64 * sk_t + b * sk_b
            + h_idx[:, None] * sk_h
            + n_idx[None, :] * sk_n
        )
        q_off = (
            t_i64 * sq_t + b * sq_b
            + h_idx[:, None] * sq_h
            + n_idx[None, :] * sq_n
        )
        v_off = (
            t_i64 * sv_t + b * sv_b
            + h_idx[:, None] * sv_h
            + v_idx[None, :] * sv_v
        )
        d_off = t_i64 * sd_t + b * sd_b + h_idx * sd_h

        k_vec = tl.load(K_ptr + k_off, mask=mask_hn, other=0.0).to(tl.float32)   # [BH, BN]
        q_vec = tl.load(Q_ptr + q_off, mask=mask_hn, other=0.0).to(tl.float32)   # [BH, BN]
        v_vec = tl.load(V_ptr + v_off, mask=mask_hv, other=0.0).to(tl.float32)   # [BH, BV]
        decay_val = tl.load(D_ptr + d_off, mask=h_mask, other=0.0).to(tl.float32)  # [BH]

        # Load S_t (= ckpt[t+1]) and S_{t-1} (= ckpt[t]).
        s_t_off = (
            tp1_i64 * sc_t + b * sc_b
            + h_idx[:, None, None] * sc_h
            + n_idx[None, :, None] * sc_n
            + v_idx[None, None, :] * sc_v
        )
        s_tm1_off = (
            t_i64 * sc_t + b * sc_b
            + h_idx[:, None, None] * sc_h
            + n_idx[None, :, None] * sc_n
            + v_idx[None, None, :] * sc_v
        )
        S_t = tl.load(Sckpt_ptr + s_t_off, mask=mask_hnv, other=0.0).to(tl.float32)
        S_tm1 = tl.load(Sckpt_ptr + s_tm1_off, mask=mask_hnv, other=0.0).to(tl.float32)

        # Upstream d_out_t.
        do_off = (
            t_i64 * sdo_t + b * sdo_b
            + h_idx[:, None] * sdo_h
            + v_idx[None, :] * sdo_v
        )
        d_out = tl.load(DOut_ptr + do_off, mask=mask_hv, other=0.0).to(tl.float32)  # [BH, BV]

        # dS_t = carry + outer(q_t, d_out_t)
        dS_t = dS_carry + q_vec[:, :, None] * d_out[:, None, :]                     # [BH, BN, BV]

        # d_q_t = sum_v S_t * d_out  -> reduce over V axis
        d_q = tl.sum(S_t * d_out[:, None, :], axis=2)                                # [BH, BN]

        # d_pre = dS_t * (1 - S_t^2)
        d_pre = dS_t * (1.0 - S_t * S_t)                                             # [BH, BN, BV]

        # d_decay_t = sum_{n,v} d_pre * S_{t-1}  (reduce over N then V)
        d_decay = tl.sum(tl.sum(d_pre * S_tm1, axis=2), axis=1)                      # [BH]

        # Recompute retrieve_t and delta_t from S_{t-1} and k_t.
        retrieved = tl.sum(S_tm1 * k_vec[:, :, None], axis=1)                        # [BH, BV]
        delta = v_vec - retrieved                                                    # [BH, BV]

        # d_outer = d_pre. Recover d_k (outer term) and d_delta.
        # outer[n,v] = k[n] * delta[v]
        # d_k_outer[n] = sum_v delta[v] * d_pre[n, v]   -> reduce over V (axis=2)
        d_k_outer = tl.sum(d_pre * delta[:, None, :], axis=2)                        # [BH, BN]
        # d_delta[v] = sum_n k[n] * d_pre[n, v]         -> reduce over N (axis=1)
        d_delta = tl.sum(d_pre * k_vec[:, :, None], axis=1)                          # [BH, BV]

        # delta = v - retrieve  =>  d_v = d_delta;  d_retrieve = -d_delta
        d_v = d_delta
        # d_k_retrieve[n] = sum_v S_{t-1}[n, v] * d_retrieve[v]
        #                 = -sum_v S_{t-1}[n, v] * d_delta[v]
        d_k_retrieve = -tl.sum(S_tm1 * d_delta[:, None, :], axis=2)                  # [BH, BN]

        d_k = d_k_outer + d_k_retrieve

        # dS_{t-1}_from_decay     = decay_t * d_pre
        # dS_{t-1}_from_retrieve  = outer(k_t, d_retrieve) = -k * d_delta
        dS_carry = (
            decay_val[:, None, None] * d_pre
            - k_vec[:, :, None] * d_delta[:, None, :]
        )

        # Store grads at this timestep.
        dk_off = (
            t_i64 * sdk_t + b * sdk_b
            + h_idx[:, None] * sdk_h
            + n_idx[None, :] * sdk_n
        )
        dv_off = (
            t_i64 * sdv_t + b * sdv_b
            + h_idx[:, None] * sdv_h
            + v_idx[None, :] * sdv_v
        )
        dq_off = (
            t_i64 * sdq_t + b * sdq_b
            + h_idx[:, None] * sdq_h
            + n_idx[None, :] * sdq_n
        )
        dd_off = t_i64 * sdd_t + b * sdd_b + h_idx * sdd_h

        tl.store(DK_ptr + dk_off, d_k.to(DK_ptr.dtype.element_ty), mask=mask_hn)
        tl.store(DV_ptr + dv_off, d_v.to(DV_ptr.dtype.element_ty), mask=mask_hv)
        tl.store(DQ_ptr + dq_off, d_q.to(DQ_ptr.dtype.element_ty), mask=mask_hn)
        tl.store(DD_ptr + dd_off, d_decay.to(DD_ptr.dtype.element_ty), mask=h_mask)

    # Write d_S0 = remaining carry.
    ds0_off = (
        b * sds0_b
        + h_idx[:, None, None] * sds0_h
        + n_idx[None, :, None] * sds0_n
        + v_idx[None, None, :] * sds0_v
    )
    tl.store(DS0_ptr + ds0_off, dS_carry.to(DS0_ptr.dtype.element_ty), mask=mask_hnv)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return max(p, 16)


def e88_triton_backward(
    k: torch.Tensor,
    v: torch.Tensor,
    q: torch.Tensor,
    decay: torch.Tensor,
    S_ckpt: torch.Tensor,
    d_out: torch.Tensor,
    d_S_final: torch.Tensor = None,
    block_h: int = None,
    num_warps: int = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the E88 backward recurrence in Triton.

    Args:
        k, q:      [T, B, H, N]  -- forward inputs.
        v:         [T, B, H, V]
        decay:     [T, B, H]
        S_ckpt:    [T+1, B, H, N, V]   from forward (S_ckpt[0]=S0).
        d_out:     [T, B, H, V]        upstream gradient w.r.t. output.
        d_S_final: [B, H, N, V] or None  upstream gradient w.r.t. S_final.
                                          Defaults to zero.
        block_h, num_warps: optional kernel tuning overrides.

    Returns:
        d_k:    [T, B, H, N]
        d_v:    [T, B, H, V]
        d_q:    [T, B, H, N]
        d_decay:[T, B, H]
        d_S0:   [B, H, N, V]
    """
    assert k.is_cuda
    T, B, H, N = k.shape
    Vsz = v.shape[-1]
    assert q.shape == (T, B, H, N)
    assert v.shape == (T, B, H, Vsz)
    assert decay.shape == (T, B, H)
    assert S_ckpt.shape == (T + 1, B, H, N, Vsz)
    assert d_out.shape == (T, B, H, Vsz)

    BLOCK_N = _next_pow2(N)
    BLOCK_V = _next_pow2(Vsz)
    if BLOCK_N > 64 or BLOCK_V > 64:
        raise NotImplementedError(
            f"e88_triton_backward currently supports N, V <= 64 (got {N},{Vsz})"
        )

    k_c = k.contiguous()
    v_c = v.contiguous()
    q_c = q.contiguous()
    d_c = decay.contiguous()
    sc_c = S_ckpt.contiguous()
    do_c = d_out.contiguous()

    if d_S_final is None:
        dsf_c = torch.zeros((B, H, N, Vsz), dtype=k_c.dtype, device=k.device)
    else:
        dsf_c = d_S_final.contiguous()
    assert dsf_c.shape == (B, H, N, Vsz)

    out_dtype = k_c.dtype
    d_k = torch.empty_like(k_c)
    d_v = torch.empty_like(v_c)
    d_q = torch.empty_like(q_c)
    d_decay = torch.empty_like(d_c)
    d_S0 = torch.empty((B, H, N, Vsz), dtype=out_dtype, device=k.device)

    # Default heads-per-program. Empirically tuned at H=386 N=V=32:
    # BLOCK_H=1 num_warps=2 is ~2x faster than BLOCK_H=4 num_warps=4 at
    # production scale. Despite "more parallelism" intuition, BLOCK_H>1
    # at high H gets worse SM utilization here — likely register spills
    # of the [BLOCK_H, N, V] state tile (16 KB per BLOCK_H step at fp32).
    # See tests/sweep_triton_block_h_at_386.py for the sweep.
    if block_h is None:
        if H >= 64:
            # Sweet spot at scale: 1 head per program, 2 warps per program.
            block_h = 1
            if num_warps is None:
                num_warps = 2
        else:
            block_h = 1
            if num_warps is None:
                num_warps = 4
    if num_warps is None:
        num_warps = 2 if block_h == 1 else (4 if block_h <= 4 else 8)

    grid = (B, (H + block_h - 1) // block_h)

    _e88_backward_kernel[grid](
        k_c, v_c, q_c, d_c, sc_c,
        do_c, dsf_c,
        d_k, d_v, d_q, d_decay, d_S0,
        # k, v, q, d, ckpt, d_out, d_S_final, d_k, d_v, d_q, d_decay, d_S0 strides
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
        num_warps=num_warps,
    )

    return d_k, d_v, d_q, d_decay, d_S0


# ---------------------------------------------------------------------------
# autograd.Function wrapper combining forward + backward.
# ---------------------------------------------------------------------------

class E88TritonFunction(torch.autograd.Function):
    """torch.autograd.Function gluing forward + backward Triton kernels.

    Ignores `linear_state=True` (not currently supported by the Triton path
    — the forward kernel always applies tanh).
    """

    @staticmethod
    def forward(ctx, S0, k, v, q, decay):
        from elman.triton.e88_triton_forward import e88_triton_forward
        out, S_final, S_ckpt = e88_triton_forward(S0, k, v, q, decay)
        # Save for backward. Note: S0 isn't strictly required (it equals
        # S_ckpt[0]), but saving it is cheap and explicit.
        ctx.save_for_backward(k, v, q, decay, S_ckpt)
        return out, S_final

    @staticmethod
    def backward(ctx, d_out, d_S_final):
        k, v, q, decay, S_ckpt = ctx.saved_tensors
        d_k, d_v, d_q, d_decay, d_S0 = e88_triton_backward(
            k, v, q, decay, S_ckpt,
            d_out=d_out.contiguous(),
            d_S_final=d_S_final.contiguous() if d_S_final is not None else None,
        )
        # Match forward signature order (S0, k, v, q, decay).
        return d_S0, d_k, d_v, d_q, d_decay


def e88_triton(S0, k, v, q, decay):
    """Differentiable Triton E88 — returns (out, S_final)."""
    return E88TritonFunction.apply(S0, k, v, q, decay)
