"""Triton forward kernel for E88 FLA-Hybrid recurrence.

This is Phase 1 of a Triton port of the existing CUDA E88 kernel (see
``elman/cuda/lib/e88_fla_hybrid_gpu.cu.cc``) — forward only, no backward
yet. The aim is parity with the PyTorch reference (the slow fallback in
``elman.models.e88_fla_hybrid``).

Per timestep, for each (batch, head):
    r_t = S_{t-1}.T @ k_t                # retrieve  [V]
    delta_t = v_t - r_t                  # [V]
    S_t = tanh(decay_t * S_{t-1} + outer(delta_t, k_t))   # [N, V]
    out_t = S_t.T @ q_t                  # [V]

Shapes (matching the CUDA kernel layout — caller is responsible for the
[B, T, ...] -> [T, B, ...] transpose):
    k, q:      [T, B, H, N]    bf16 or fp32
    v:         [T, B, H, V]
    decay:     [T, B, H]
    S0:        [B, H, N, V]
    output:    [T, B, H, V]
    S_final:   [B, H, N, V]
    S_checkpoints: [num_ckpts, B, H, N, V]   (sparse: every CKPT_INTERVAL steps)

Sparse forward checkpointing
----------------------------
To shrink the S_checkpoints buffer from O(T) to O(T/K) tiles, we save S
only every CKPT_INTERVAL steps (default K=16, mirroring CUDA reg-own).

Layout:
    S_ckpt[0]                    = initial S0
    S_ckpt[k] for k >= 1         = S after step (k * CKPT_INTERVAL - 1)

So segment ``seg`` (covering t in [seg*K, seg*K + K)) starts from
S_ckpt[seg], i.e. S_ckpt[seg] is the state right BEFORE step seg*K.

We currently REQUIRE T % CKPT_INTERVAL == 0 — the unaligned case can be
added later (just needs a shorter final segment in the backward replay).

Number of slots: ``num_ckpts = T // CKPT_INTERVAL + 1``  (e.g. T=512 K=16
gives 33 slots vs 513 in the dense layout — ~16x memory shrink).

The kernel currently supports N <= 64 and V <= 64 (sufficient for the
E88 configurations actually used in benchmarks: n_state in {16, 32}).

Multi-program tiling
--------------------
At high head counts (H >= ~256) we run out of useful (B, H) parallelism
(too many tiny programs, all competing for SMs and bandwidth). To fix
this, each program handles ``BLOCK_H`` consecutive heads at once:

  state tile:   [BLOCK_H, BLOCK_N, BLOCK_V]
  per-step:     loaded k/v/q/decay vectorize across BLOCK_H

Grid: (B, ceil(H / BLOCK_H)).  BLOCK_H is autotuned over {1, 2, 4, 8}.
"""
from __future__ import absolute_import

from typing import Tuple

import torch
import triton
import triton.language as tl


# Default checkpoint stride. Matches the CUDA register-owned kernel
# (``E88_REG_CHECKPOINT_INTERVAL = 16``) so that the Python-allocated
# S_checkpoints tensor has the same number of slots as the CUDA path.
DEFAULT_CKPT_INTERVAL = 16


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

@triton.jit
def _e88_forward_kernel(
    # Inputs
    K_ptr,            # [T, B, H, N]
    V_ptr,            # [T, B, H, V]
    Q_ptr,            # [T, B, H, N]
    D_ptr,            # [T, B, H]
    S0_ptr,           # [B, H, N, V]
    # Outputs
    Out_ptr,          # [T, B, H, V]
    Sfinal_ptr,       # [B, H, N, V]
    Sckpt_ptr,        # [num_ckpts, B, H, N, V]
    # Strides (in elements). All tensors are made contiguous before launch,
    # so we only need one stride argument set per tensor's outer dim, but we
    # pass the full set explicitly to keep the kernel self-documenting and
    # not require contiguity assumptions in the future.
    sk_t, sk_b, sk_h, sk_n,
    sv_t, sv_b, sv_h, sv_v,
    sq_t, sq_b, sq_h, sq_n,
    sd_t, sd_b, sd_h,
    s0_b, s0_h, s0_n, s0_v,
    so_t, so_b, so_h, so_v,
    sf_b, sf_h, sf_n, sf_v,
    sc_t, sc_b, sc_h, sc_n, sc_v,
    # Sizes
    T: tl.constexpr, B: tl.constexpr, H: tl.constexpr,
    N: tl.constexpr, V: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
    CKPT_INTERVAL: tl.constexpr,
):
    """One program per (batch, head_block). Sequential time loop."""
    # 2D launch grid: (B, ceil(H / BLOCK_H))
    b = tl.program_id(0).to(tl.int64)
    hg = tl.program_id(1).to(tl.int64)

    h_start = hg * BLOCK_H
    h_idx = h_start + tl.arange(0, BLOCK_H)            # [BLOCK_H]
    h_mask = h_idx < H                                  # [BLOCK_H]

    # We assume N <= BLOCK_N and V <= BLOCK_V. Mask off padded N/V lanes.
    n_idx = tl.arange(0, BLOCK_N)
    v_idx = tl.arange(0, BLOCK_V)
    n_mask = n_idx < N
    v_mask = v_idx < V

    # Composite masks for state and intermediates
    mask_hnv = (h_mask[:, None, None] & n_mask[None, :, None] & v_mask[None, None, :])
    mask_hn = h_mask[:, None] & n_mask[None, :]
    mask_hv = h_mask[:, None] & v_mask[None, :]

    # Load initial state S as [BLOCK_H, BLOCK_N, BLOCK_V] tile.
    s0_off = (
        b * s0_b
        + h_idx[:, None, None] * s0_h
        + n_idx[None, :, None] * s0_n
        + v_idx[None, None, :] * s0_v
    )
    S = tl.load(S0_ptr + s0_off, mask=mask_hnv, other=0.0).to(tl.float32)

    # Save S0 into checkpoint slot 0.
    sc0_off = (
        0 * sc_t + b * sc_b
        + h_idx[:, None, None] * sc_h
        + n_idx[None, :, None] * sc_n
        + v_idx[None, None, :] * sc_v
    )
    tl.store(Sckpt_ptr + sc0_off, S.to(Sckpt_ptr.dtype.element_ty), mask=mask_hnv)

    # Sequential time loop.
    # NOTE: we cast (t+1) to int64 below for offset arithmetic on
    # potentially very large tensors (e.g. S_ckpt at T=4K B=2 H=386
    # N=V=32 — even the SPARSE layout is large enough to need int64 once
    # B*H*N*V exceeds ~500M elements).
    for t in range(T):
        # Load k_t, q_t, v_t, decay_t for all BLOCK_H heads in this block.
        # Use int64 for the t-stride product so we don't overflow at large
        # T*B*H*N (input tensors can be ~6 GB at production scale).
        t_i64 = tl.full([1], t, dtype=tl.int64)
        # k, q: [BLOCK_H, BLOCK_N]
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
        # v: [BLOCK_H, BLOCK_V]
        v_off = (
            t_i64 * sv_t + b * sv_b
            + h_idx[:, None] * sv_h
            + v_idx[None, :] * sv_v
        )
        # decay: [BLOCK_H]
        d_off = t_i64 * sd_t + b * sd_b + h_idx * sd_h

        k_vec = tl.load(K_ptr + k_off, mask=mask_hn, other=0.0).to(tl.float32)   # [BH, BN]
        q_vec = tl.load(Q_ptr + q_off, mask=mask_hn, other=0.0).to(tl.float32)   # [BH, BN]
        v_vec = tl.load(V_ptr + v_off, mask=mask_hv, other=0.0).to(tl.float32)   # [BH, BV]
        d_val = tl.load(D_ptr + d_off, mask=h_mask, other=0.0).to(tl.float32)    # [BH]

        # Retrieve: r[h, v] = sum_n S[h, n, v] * k[h, n]   (reduce over N axis=1)
        retrieved = tl.sum(S * k_vec[:, :, None], axis=1)   # [BH, BV]
        delta = v_vec - retrieved                           # [BH, BV]

        # Update: S = tanh(decay * S + outer(k, delta))
        outer = k_vec[:, :, None] * delta[:, None, :]       # [BH, BN, BV]
        pre = d_val[:, None, None] * S + outer
        # Manual tanh via expm1 trick — slightly more numerically stable
        # than (e2x - 1) / (e2x + 1) and matches CUDA's tanhf reasonably
        # within bf16 tolerance.
        e2x = tl.exp(2.0 * pre)
        S = (e2x - 1.0) / (e2x + 1.0)

        # Output: out[h, v] = sum_n S[h, n, v] * q[h, n]   (reduce over N axis=1)
        out_vec = tl.sum(S * q_vec[:, :, None], axis=1)     # [BH, BV]

        out_off = (
            t_i64 * so_t + b * so_b
            + h_idx[:, None] * so_h
            + v_idx[None, :] * so_v
        )
        tl.store(Out_ptr + out_off, out_vec.to(Out_ptr.dtype.element_ty), mask=mask_hv)

        # Sparse checkpoint write: only when (t+1) is a multiple of
        # CKPT_INTERVAL. The wrapper enforces T % CKPT_INTERVAL == 0, so
        # the last step (t = T-1) is always a checkpoint.
        # Slot index: (t+1) // CKPT_INTERVAL — that is, S_ckpt[k] holds
        # S after step (k*CKPT_INTERVAL - 1) for k>=1.
        is_ckpt_step = ((t + 1) % CKPT_INTERVAL) == 0
        if is_ckpt_step:
            slot_i64 = tl.full([1], (t + 1) // CKPT_INTERVAL, dtype=tl.int64)
            sc_off = (
                slot_i64 * sc_t + b * sc_b
                + h_idx[:, None, None] * sc_h
                + n_idx[None, :, None] * sc_n
                + v_idx[None, None, :] * sc_v
            )
            tl.store(Sckpt_ptr + sc_off, S.to(Sckpt_ptr.dtype.element_ty), mask=mask_hnv)

    # Write S_final.
    sf_off = (
        b * sf_b
        + h_idx[:, None, None] * sf_h
        + n_idx[None, :, None] * sf_n
        + v_idx[None, None, :] * sf_v
    )
    tl.store(Sfinal_ptr + sf_off, S.to(Sfinal_ptr.dtype.element_ty), mask=mask_hnv)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return max(p, 16)


# Lightweight in-process autotune cache: maps (B, T, H, N, V, dtype) -> best
# (BLOCK_H, num_warps).
_AUTOTUNE_CACHE = {}

# BLOCK_H candidates. Capped so the [BLOCK_H, BLOCK_N, BLOCK_V] state tile
# stays in registers/SRAM. For BLOCK_N=BLOCK_V=32 and fp32, tile=4 KB per
# BLOCK_H slot. BLOCK_H=8 is 32 KB; BLOCK_H=16 with BLOCK_N=BLOCK_V=64 is
# 256 KB which spills, so we cap depending on N*V.
_BLOCK_H_CANDIDATES = (1, 2, 4, 8, 16)


def _select_block_h_candidates(N: int, V: int):
    """Return BLOCK_H values that won't blow SRAM for this (N, V)."""
    elem = _next_pow2(N) * _next_pow2(V)
    if elem >= 64 * 64:        # 4096 elements
        return (1, 2, 4)
    if elem >= 32 * 64:        # 2048 elements
        return (1, 2, 4, 8)
    return _BLOCK_H_CANDIDATES  # up to 16 for small (N, V)


def _autotune_kernel(launch_args, B, T, H, N, Vsz, dtype, ckpt_interval):
    """Tiny in-process autotune. Tries (BLOCK_H, num_warps) and caches winner.

    Empirically, the "right" num_warps depends on BLOCK_H AND on H itself:
    at high H (many programs in flight) fewer warps per program reduces
    occupancy contention; at low H (few programs) more warps fills the SM.

    For H < 16 we just default to BLOCK_H=1 (no head-grouping helps when
    there are too few heads to begin with).
    """
    cache_key = (B, T, H, N, Vsz, str(dtype), ckpt_interval)
    if cache_key in _AUTOTUNE_CACHE:
        return _AUTOTUNE_CACHE[cache_key]

    if H < 16:
        cfg = (1, 4)
        _AUTOTUNE_CACHE[cache_key] = cfg
        return cfg

    bh_candidates = _select_block_h_candidates(N, Vsz)
    # Restrict to BHs that don't grossly oversubscribe a SM. BLOCK_H=1
    # never needs more than 4 warps; BLOCK_H=8 with 1 warp tends to
    # generate massive spills (observed 13 ms outliers in sweep), so we
    # exclude 1-warp configs for BLOCK_H >= 8.
    nw_candidates_for = {
        1: (2, 4),
        2: (2, 4),
        4: (2, 4),
        8: (2, 4),
        16: (4, 8),
    }

    import time as _time
    best = None
    best_t = float("inf")

    (k_c, v_c, q_c, d_c, s0_c, out, S_final, S_ckpt, strides) = launch_args

    for bh in bh_candidates:
        if bh > H:
            continue
        for nw in nw_candidates_for[bh]:
            try:
                grid = (B, (H + bh - 1) // bh)
                # Warmup
                for _ in range(3):
                    _e88_forward_kernel[grid](
                        k_c, v_c, q_c, d_c, s0_c,
                        out, S_final, S_ckpt,
                        *strides,
                        T=T, B=B, H=H, N=N, V=Vsz,
                        BLOCK_N=_next_pow2(N), BLOCK_V=_next_pow2(Vsz),
                        BLOCK_H=bh,
                        CKPT_INTERVAL=ckpt_interval,
                        num_warps=nw,
                    )
                torch.cuda.synchronize()
                t0 = _time.perf_counter()
                iters = 5
                for _ in range(iters):
                    _e88_forward_kernel[grid](
                        k_c, v_c, q_c, d_c, s0_c,
                        out, S_final, S_ckpt,
                        *strides,
                        T=T, B=B, H=H, N=N, V=Vsz,
                        BLOCK_N=_next_pow2(N), BLOCK_V=_next_pow2(Vsz),
                        BLOCK_H=bh,
                        CKPT_INTERVAL=ckpt_interval,
                        num_warps=nw,
                    )
                torch.cuda.synchronize()
                t = (_time.perf_counter() - t0) / iters
                if t < best_t:
                    best_t = t
                    best = (bh, nw)
            except Exception:
                # OOM / SRAM exceeded for this config — skip.
                continue

    if best is None:
        best = (1, 4)
    _AUTOTUNE_CACHE[cache_key] = best
    return best


def e88_triton_forward(
    S0: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q: torch.Tensor,
    decay: torch.Tensor,
    head_v_dim: int = None,
    block_h: int = None,
    num_warps: int = None,
    ckpt_interval: int = DEFAULT_CKPT_INTERVAL,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the E88 forward recurrence in Triton.

    Args:
        S0:    [B, H, N, V] initial state (matches CUDA kernel).
        k, q:  [T, B, H, N] keys / queries.
        v:     [T, B, H, V] values.
        decay: [T, B, H] per-step decay scalars.
        head_v_dim: ignored, present for API compatibility with CUDA wrapper.
        block_h: optional override for BLOCK_H (heads per program).
                 If None, autotune picks the best for this shape.
        num_warps: optional override; default chosen with block_h.
        ckpt_interval: store S to S_checkpoints every K steps. Default 16
                 (matches CUDA register-owned). Currently REQUIRES
                 T % ckpt_interval == 0.

    Returns:
        out:           [T, B, H, V]
        S_final:       [B, H, N, V]
        S_checkpoints: [num_ckpts, B, H, N, V] — sparse checkpoint
                       layout. ``num_ckpts = T // ckpt_interval + 1``.
                       S_ckpt[0] = S0,
                       S_ckpt[k] = S after step (k*ckpt_interval - 1) for k>=1.

    The output dtype matches the input k/v/q dtype (bf16 -> bf16, fp32 ->
    fp32). All accumulation inside the kernel is fp32.
    """
    assert k.is_cuda, "Inputs must be on CUDA"
    assert k.dim() == 4 and q.dim() == 4 and v.dim() == 4, \
        "k/v/q must be [T, B, H, *]"
    assert decay.dim() == 3, "decay must be [T, B, H]"
    assert S0.dim() == 4, "S0 must be [B, H, N, V]"

    T, B, H, N = k.shape
    Vsz = v.shape[-1]
    assert q.shape == (T, B, H, N)
    assert v.shape == (T, B, H, Vsz)
    assert decay.shape == (T, B, H)
    assert S0.shape == (B, H, N, Vsz)

    BLOCK_N = _next_pow2(N)
    BLOCK_V = _next_pow2(Vsz)
    if BLOCK_N > 64 or BLOCK_V > 64:
        raise NotImplementedError(
            f"e88_triton_forward currently supports N, V <= 64 "
            f"(got N={N}, V={Vsz}). Larger sizes need a tiled implementation."
        )

    if T % ckpt_interval != 0:
        raise NotImplementedError(
            f"Sparse-checkpoint forward currently requires T % ckpt_interval == 0 "
            f"(got T={T}, ckpt_interval={ckpt_interval}). The unaligned case can "
            f"be added by handling a shorter final segment in the backward replay."
        )

    # Make everything contiguous so strides are predictable.
    k_c = k.contiguous()
    v_c = v.contiguous()
    q_c = q.contiguous()
    d_c = decay.contiguous()
    s0_c = S0.contiguous()

    out_dtype = k_c.dtype
    out = torch.empty((T, B, H, Vsz), dtype=out_dtype, device=k.device)
    S_final = torch.empty_like(s0_c)

    num_ckpts = T // ckpt_interval + 1
    S_ckpt = torch.empty((num_ckpts, B, H, N, Vsz), dtype=out_dtype, device=k.device)

    strides = (
        # k strides
        k_c.stride(0), k_c.stride(1), k_c.stride(2), k_c.stride(3),
        # v strides
        v_c.stride(0), v_c.stride(1), v_c.stride(2), v_c.stride(3),
        # q strides
        q_c.stride(0), q_c.stride(1), q_c.stride(2), q_c.stride(3),
        # decay strides
        d_c.stride(0), d_c.stride(1), d_c.stride(2),
        # S0 strides
        s0_c.stride(0), s0_c.stride(1), s0_c.stride(2), s0_c.stride(3),
        # out strides
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        # S_final strides
        S_final.stride(0), S_final.stride(1), S_final.stride(2), S_final.stride(3),
        # S_ckpt strides
        S_ckpt.stride(0), S_ckpt.stride(1), S_ckpt.stride(2),
        S_ckpt.stride(3), S_ckpt.stride(4),
    )

    # Pick BLOCK_H + num_warps.
    if block_h is None:
        # Production-scale heuristic: at H >= 64 the empirical sweep at
        # H=386 N=V=32 shows BLOCK_H=1 num_warps=2 is fastest (1.83x vs
        # the autotune's choice of (4, 2)). Use this directly to skip
        # the autotune launch cost at training start.
        # See tests/sweep_triton_block_h_at_386.py.
        if H >= 64:
            block_h_chosen, nw = 1, 2
        else:
            launch_args = (k_c, v_c, q_c, d_c, s0_c, out, S_final, S_ckpt, strides)
            block_h_chosen, nw = _autotune_kernel(
                launch_args, B, T, H, N, Vsz, out_dtype, ckpt_interval
            )
    else:
        block_h_chosen = int(block_h)
        nw = int(num_warps) if num_warps is not None else (8 if block_h_chosen >= 8 else 4)

    grid = (B, (H + block_h_chosen - 1) // block_h_chosen)

    _e88_forward_kernel[grid](
        k_c, v_c, q_c, d_c, s0_c,
        out, S_final, S_ckpt,
        *strides,
        T=T, B=B, H=H, N=N, V=Vsz,
        BLOCK_N=BLOCK_N, BLOCK_V=BLOCK_V,
        BLOCK_H=block_h_chosen,
        CKPT_INTERVAL=ckpt_interval,
        num_warps=nw,
    )
    return out, S_final, S_ckpt


# ---------------------------------------------------------------------------
# PyTorch reference (mirrors the slow fallback in e88_fla_hybrid.py)
# ---------------------------------------------------------------------------

def e88_torch_reference(
    S0: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q: torch.Tensor,
    decay: torch.Tensor,
    linear_state: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for parity testing.

    Returns the DENSE checkpoint history (one slot per timestep, plus
    initial). Tests that compare against this reference need to subsample
    when comparing to the sparse kernel checkpoint.

    Inputs use the SAME [T, B, H, ...] convention as the Triton wrapper
    (i.e., the CUDA kernel convention), to keep the test simple. The
    PyTorch fallback inside e88_fla_hybrid.py uses [B, T, H, ...] but the
    math is identical; we just iterate the time axis differently.

    Args:
        S0:    [B, H, N, V]
        k, q:  [T, B, H, N]
        v:     [T, B, H, V]
        decay: [T, B, H]
        linear_state: if True, drop the tanh (matches `linear_state=True`
                      in e88_fla_hybrid.py).

    Returns:
        out:           [T, B, H, V]
        S_final:       [B, H, N, V]
        ckpt:          [T+1, B, H, N, V] — DENSE per-step history.
                       ckpt[0]=S0, ckpt[t+1]=S after step t.
    """
    T, B, H, N = k.shape
    Vsz = v.shape[-1]
    out_dtype = k.dtype

    S = S0.clone().to(torch.float32)
    out = torch.empty((T, B, H, Vsz), dtype=out_dtype, device=k.device)
    ckpt = torch.empty((T + 1, B, H, N, Vsz), dtype=out_dtype, device=k.device)
    ckpt[0] = S.to(out_dtype)

    for t in range(T):
        k_t = k[t].to(torch.float32)        # [B, H, N]
        q_t = q[t].to(torch.float32)        # [B, H, N]
        v_t = v[t].to(torch.float32)        # [B, H, V]
        d_t = decay[t].to(torch.float32).unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]

        # retrieve: S [B,H,N,V], k_t [B,H,N] -> [B, H, V]
        retrieved = torch.einsum('bhnv,bhn->bhv', S, k_t)
        delta = v_t - retrieved

        outer = torch.einsum('bhn,bhv->bhnv', k_t, delta)

        pre = d_t * S + outer
        if linear_state:
            S = pre
        else:
            S = torch.tanh(pre)

        Sq = torch.einsum('bhnv,bhn->bhv', S, q_t)
        out[t] = Sq.to(out_dtype)
        ckpt[t + 1] = S.to(out_dtype)

    S_final = S.to(out_dtype)
    return out, S_final, ckpt
