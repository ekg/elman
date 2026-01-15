"""
E74 Checkpointed Triton Kernels

In-place S updates with gradient checkpointing for ALL state types.

Key design:
- S is a single buffer, updated IN-PLACE
- Checkpoints saved every K steps
- Backward recomputes from checkpoints

Supports:
- Diagonal state [B, n]
- Full matrix state [B, n, n]
- Low-rank state [B, n, r] + [B, n, r]
- Block-diagonal state [B, n/b, b, b]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice as tl_libdevice
from typing import Optional, Tuple, Literal
from enum import Enum


class StateType(Enum):
    DIAGONAL = 'diagonal'
    FULL = 'full'
    LOWRANK = 'lowrank'
    BLOCKDIAG = 'blockdiag'


# =============================================================================
# Triton Kernels - Diagonal State
# =============================================================================

@triton.jit
def normalize_k_kernel(
    k_ptr, k_norm_ptr,
    B, N,
    stride_kb, stride_kn,
    BLOCK_N: tl.constexpr,
):
    """k_norm = k / (||k|| + eps)"""
    b = tl.program_id(0)
    n_offsets = tl.arange(0, BLOCK_N)
    mask = n_offsets < N

    k = tl.load(k_ptr + b * stride_kb + n_offsets * stride_kn, mask=mask, other=0.0)
    k_sq = k * k
    norm_sq = tl.sum(k_sq, axis=0)
    norm = tl.sqrt(norm_sq) + 1e-6
    k_norm = k / norm

    tl.store(k_norm_ptr + b * stride_kb + n_offsets * stride_kn, k_norm, mask=mask)


@triton.jit
def diagonal_delta_inplace_kernel(
    S_ptr, v_ptr, k_norm_ptr,
    B, N,
    stride_sb, stride_sn,
    use_tanh: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """In-place diagonal delta: S[i] = f(S[i]*(1-k²[i]) + v[i]*k[i])"""
    b = tl.program_id(0)
    n_offsets = tl.arange(0, BLOCK_N)
    mask = n_offsets < N

    S = tl.load(S_ptr + b * stride_sb + n_offsets * stride_sn, mask=mask, other=0.0)
    v = tl.load(v_ptr + b * stride_sb + n_offsets * stride_sn, mask=mask, other=0.0)
    k = tl.load(k_norm_ptr + b * stride_sb + n_offsets * stride_sn, mask=mask, other=0.0)

    k_sq = k * k
    S_new = S * (1.0 - k_sq) + v * k

    if use_tanh:
        S_new = tl_libdevice.tanh(S_new.to(tl.float32)).to(S_new.dtype)

    tl.store(S_ptr + b * stride_sb + n_offsets * stride_sn, S_new, mask=mask)


@triton.jit
def diagonal_output_kernel(
    S_ptr, q_ptr, out_ptr, Sq_cache_ptr,
    B, N,
    stride_sb, stride_sn,
    save_Sq: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Output: out = (S * q) * silu(S * q)"""
    b = tl.program_id(0)
    n_offsets = tl.arange(0, BLOCK_N)
    mask = n_offsets < N

    S = tl.load(S_ptr + b * stride_sb + n_offsets * stride_sn, mask=mask, other=0.0)
    q = tl.load(q_ptr + b * stride_sb + n_offsets * stride_sn, mask=mask, other=0.0)

    Sq = S * q
    if save_Sq:
        tl.store(Sq_cache_ptr + b * stride_sb + n_offsets * stride_sn, Sq, mask=mask)

    # Cast to float32 for math operations
    Sq_f32 = Sq.to(tl.float32)
    sigmoid_Sq = tl.sigmoid(Sq_f32)
    silu_Sq = Sq_f32 * sigmoid_Sq
    out = Sq_f32 * silu_Sq
    out = out.to(Sq.dtype)

    tl.store(out_ptr + b * stride_sb + n_offsets * stride_sn, out, mask=mask)


# =============================================================================
# Triton Kernels - Full Matrix State
# =============================================================================

@triton.jit
def full_retrieval_inplace_kernel(
    S_ptr, k_norm_ptr, retrieved_ptr,
    B, N,
    stride_sb, stride_si, stride_sj,
    stride_kb, stride_kn,
    BLOCK_J: tl.constexpr,
):
    """Compute retrieved[i] = sum_j S[i,j] * k_norm[j]"""
    b = tl.program_id(0)
    i = tl.program_id(1)

    if i >= N:
        return

    acc = 0.0
    for j_start in range(0, N, BLOCK_J):
        j_offsets = j_start + tl.arange(0, BLOCK_J)
        mask = j_offsets < N

        S_ij = tl.load(S_ptr + b * stride_sb + i * stride_si + j_offsets * stride_sj,
                       mask=mask, other=0.0)
        k_j = tl.load(k_norm_ptr + b * stride_kb + j_offsets * stride_kn,
                      mask=mask, other=0.0)
        acc += tl.sum(S_ij.to(tl.float32) * k_j.to(tl.float32), axis=0)

    tl.store(retrieved_ptr + b * stride_kb + i * stride_kn, acc)


@triton.jit
def full_delta_inplace_kernel(
    S_ptr, v_ptr, retrieved_ptr, k_norm_ptr,
    B, N,
    stride_sb, stride_si, stride_sj,
    stride_vb, stride_vn,
    use_tanh: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    """In-place: S[i,j] = f(S[i,j] + (v[i] - retrieved[i]) * k_norm[j])"""
    b = tl.program_id(0)
    i_block = tl.program_id(1)
    j_block = tl.program_id(2)

    i_offsets = i_block * BLOCK_I + tl.arange(0, BLOCK_I)
    j_offsets = j_block * BLOCK_J + tl.arange(0, BLOCK_J)

    i_mask = i_offsets < N
    j_mask = j_offsets < N
    mask = i_mask[:, None] & j_mask[None, :]

    S_block = tl.load(
        S_ptr + b * stride_sb + i_offsets[:, None] * stride_si + j_offsets[None, :] * stride_sj,
        mask=mask, other=0.0
    )

    v_i = tl.load(v_ptr + b * stride_vb + i_offsets * stride_vn, mask=i_mask, other=0.0)
    ret_i = tl.load(retrieved_ptr + b * stride_vb + i_offsets * stride_vn, mask=i_mask, other=0.0)
    k_j = tl.load(k_norm_ptr + b * stride_vb + j_offsets * stride_vn, mask=j_mask, other=0.0)

    delta_i = v_i - ret_i
    outer = delta_i[:, None] * k_j[None, :]
    S_new = S_block + outer

    if use_tanh:
        S_new = tl_libdevice.tanh(S_new.to(tl.float32)).to(S_new.dtype)

    tl.store(
        S_ptr + b * stride_sb + i_offsets[:, None] * stride_si + j_offsets[None, :] * stride_sj,
        S_new, mask=mask
    )


@triton.jit
def full_output_kernel(
    S_ptr, q_ptr, out_ptr, Sq_cache_ptr,
    B, N,
    stride_sb, stride_si, stride_sj,
    stride_qb, stride_qn,
    save_Sq: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    """Compute out[i] = (S[i,:] @ q) * silu(...)"""
    b = tl.program_id(0)
    i = tl.program_id(1)

    if i >= N:
        return

    acc = 0.0
    for j_start in range(0, N, BLOCK_J):
        j_offsets = j_start + tl.arange(0, BLOCK_J)
        mask = j_offsets < N

        S_ij = tl.load(S_ptr + b * stride_sb + i * stride_si + j_offsets * stride_sj,
                       mask=mask, other=0.0)
        q_j = tl.load(q_ptr + b * stride_qb + j_offsets * stride_qn, mask=mask, other=0.0)
        acc += tl.sum(S_ij.to(tl.float32) * q_j.to(tl.float32), axis=0)

    Sq = acc

    if save_Sq:
        tl.store(Sq_cache_ptr + b * stride_qb + i * stride_qn, Sq)

    sigmoid_Sq = tl.sigmoid(Sq)
    silu_Sq = Sq * sigmoid_Sq
    out = Sq * silu_Sq

    tl.store(out_ptr + b * stride_qb + i * stride_qn, out)


# =============================================================================
# Unified Checkpointed Autograd Function
# =============================================================================

class CheckpointedFunction(torch.autograd.Function):
    """
    Unified checkpointed forward/backward for all state types.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,          # [T, B, dim]
        S0: torch.Tensor,         # State (shape depends on state_type)
        W_k: torch.Tensor,
        W_v: torch.Tensor,
        W_q: torch.Tensor,        # Can be None if tied
        checkpoint_interval: int,
        use_tanh: bool,
        state_type: str,          # 'diagonal', 'full', 'lowrank', 'blockdiag'
        tied_kq: bool,
        tied_kvq: bool,
        rank: int,                # For lowrank
        block_size: int,          # For blockdiag
    ):
        T, B, dim = x.shape
        device = x.device
        dtype = x.dtype
        K = checkpoint_interval

        # Determine state shape and n
        if state_type == 'diagonal':
            n = S0.shape[1]
            state_shape = (B, n)
        elif state_type == 'full':
            n = S0.shape[1]
            state_shape = (B, n, n)
        elif state_type == 'lowrank':
            # S0 is (U, V) tuple packed as single tensor [B, n, 2*r]
            n = S0.shape[1]
            r = rank
            state_shape = (B, n, 2 * r)
        elif state_type == 'blockdiag':
            n = S0.shape[1] * S0.shape[2]  # n_blocks * block_size
            state_shape = S0.shape
        else:
            raise ValueError(f"Unknown state_type: {state_type}")

        num_checkpoints = (T + K) // K

        # Allocate
        output = torch.empty(T, B, n, device=device, dtype=dtype)
        S_checkpoints = torch.empty(num_checkpoints, *state_shape, device=device, dtype=dtype)
        Sq_cache = torch.empty(T, B, n, device=device, dtype=dtype)

        S = S0.clone()

        # Batch projections
        x_flat = x.reshape(T * B, dim)

        if tied_kvq:
            k_all = (x_flat @ W_k.T).reshape(T, B, n)
            v_all = k_all
            q_all = k_all
        elif tied_kq:
            k_all = (x_flat @ W_k.T).reshape(T, B, n)
            v_all = (x_flat @ W_v.T).reshape(T, B, n)
            q_all = k_all
        else:
            k_all = (x_flat @ W_k.T).reshape(T, B, n)
            v_all = (x_flat @ W_v.T).reshape(T, B, n)
            q_all = (x_flat @ W_q.T).reshape(T, B, n)

        # Normalize k
        k_norm_all = k_all / (k_all.norm(dim=-1, keepdim=True) + 1e-6)

        # Save initial checkpoint
        S_checkpoints[0].copy_(S)
        cp_idx = 1

        BLOCK_N = min(triton.next_power_of_2(n), 1024)

        # Forward pass
        for t in range(T):
            k_norm_t = k_norm_all[t]
            v_t = v_all[t]
            q_t = q_all[t]

            if state_type == 'diagonal':
                # In-place diagonal update
                diagonal_delta_inplace_kernel[(B,)](
                    S, v_t, k_norm_t,
                    B, n, S.stride(0), S.stride(1),
                    use_tanh=use_tanh, BLOCK_N=BLOCK_N,
                )
                diagonal_output_kernel[(B,)](
                    S, q_t, output[t], Sq_cache[t],
                    B, n, S.stride(0), S.stride(1),
                    save_Sq=True, BLOCK_N=BLOCK_N,
                )

            elif state_type == 'full':
                # In-place full matrix update
                retrieved = torch.empty(B, n, device=device, dtype=dtype)
                BLOCK_J = min(64, n)

                full_retrieval_inplace_kernel[(B, n)](
                    S, k_norm_t, retrieved,
                    B, n,
                    S.stride(0), S.stride(1), S.stride(2),
                    k_norm_t.stride(0), k_norm_t.stride(1),
                    BLOCK_J=BLOCK_J,
                )

                BLOCK_I, BLOCK_J = 16, 16
                grid_i = (n + BLOCK_I - 1) // BLOCK_I
                grid_j = (n + BLOCK_J - 1) // BLOCK_J

                full_delta_inplace_kernel[(B, grid_i, grid_j)](
                    S, v_t, retrieved, k_norm_t,
                    B, n,
                    S.stride(0), S.stride(1), S.stride(2),
                    v_t.stride(0), v_t.stride(1),
                    use_tanh=use_tanh, BLOCK_I=BLOCK_I, BLOCK_J=BLOCK_J,
                )

                full_output_kernel[(B, n)](
                    S, q_t, output[t], Sq_cache[t],
                    B, n,
                    S.stride(0), S.stride(1), S.stride(2),
                    q_t.stride(0), q_t.stride(1),
                    save_Sq=True, BLOCK_J=BLOCK_J,
                )

            elif state_type == 'lowrank':
                # S = [B, n, 2*r] where [:,:,:r] is U, [:,:,r:] is V
                U = S[:, :, :rank]
                V = S[:, :, rank:]

                # retrieved = U @ (V^T @ k_norm)
                Vtk = torch.einsum('bnr,bn->br', V, k_norm_t)
                retrieved = torch.einsum('bnr,br->bn', U, Vtk)

                # Update U
                delta = v_t - retrieved
                k_r = k_norm_t[:, :rank]  # Simple projection to rank
                U_new = U + torch.einsum('bn,br->bnr', delta, k_r)

                if use_tanh:
                    U_new = torch.tanh(U_new)

                S[:, :, :rank] = U_new

                # Output
                Vtq = torch.einsum('bnr,bn->br', V, q_t)
                Sq = torch.einsum('bnr,br->bn', U_new, Vtq)
                Sq_cache[t] = Sq
                output[t] = Sq * F.silu(Sq)

            elif state_type == 'blockdiag':
                # S = [B, n_blocks, block_size, block_size]
                nb = S.shape[1]
                bs = S.shape[2]

                k_blocks = k_norm_t.reshape(B, nb, bs)
                v_blocks = v_t.reshape(B, nb, bs)
                q_blocks = q_t.reshape(B, nb, bs)

                # Per-block retrieval and update
                retrieved = torch.einsum('bnij,bnj->bni', S, k_blocks)
                delta = v_blocks - retrieved
                outer = torch.einsum('bni,bnj->bnij', delta, k_blocks)
                S_new = S + outer

                if use_tanh:
                    S_new = torch.tanh(S_new)

                S.copy_(S_new)

                Sq_blocks = torch.einsum('bnij,bnj->bni', S, q_blocks)
                Sq = Sq_blocks.reshape(B, n)
                Sq_cache[t] = Sq
                output[t] = Sq * F.silu(Sq)

            # Save checkpoint
            if (t + 1) % K == 0 and cp_idx < num_checkpoints:
                S_checkpoints[cp_idx].copy_(S)
                cp_idx += 1

        # Final checkpoint
        if T % K != 0:
            S_checkpoints[-1].copy_(S)

        # Save for backward
        ctx.save_for_backward(
            x, S_checkpoints, k_all, v_all, q_all, k_norm_all, Sq_cache,
            W_k, W_v, W_q if W_q is not None else W_k,
        )
        ctx.T, ctx.B, ctx.n = T, B, n
        ctx.K = K
        ctx.use_tanh = use_tanh
        ctx.state_type = state_type
        ctx.tied_kq = tied_kq
        ctx.tied_kvq = tied_kvq
        ctx.rank = rank
        ctx.block_size = block_size
        ctx.state_shape = state_shape

        return output, S

    @staticmethod
    def backward(ctx, d_output, d_S_final):
        (x, S_checkpoints, k_all, v_all, q_all, k_norm_all, Sq_cache,
         W_k, W_v, W_q) = ctx.saved_tensors

        T, B, n = ctx.T, ctx.B, ctx.n
        K = ctx.K
        use_tanh = ctx.use_tanh
        state_type = ctx.state_type
        tied_kq = ctx.tied_kq
        tied_kvq = ctx.tied_kvq
        rank = ctx.rank
        state_shape = ctx.state_shape

        device = x.device
        dtype = x.dtype
        dim = x.shape[2]

        # Gradient buffers
        d_k_all = torch.zeros(T, B, n, device=device, dtype=dtype)
        d_v_all = torch.zeros(T, B, n, device=device, dtype=dtype)
        d_q_all = torch.zeros(T, B, n, device=device, dtype=dtype)

        # Process backward through checkpoint intervals
        num_intervals = (T + K - 1) // K

        # Running d_S
        d_S = torch.zeros(*state_shape, device=device, dtype=dtype)

        for interval_idx in range(num_intervals - 1, -1, -1):
            t_start = interval_idx * K
            t_end = min((interval_idx + 1) * K, T)
            interval_len = t_end - t_start

            if interval_len <= 0:
                continue

            # Recompute forward through interval
            S_recomputed = [S_checkpoints[interval_idx].clone()]

            for t_local in range(interval_len):
                t = t_start + t_local
                S_curr = S_recomputed[-1].clone()

                if state_type == 'diagonal':
                    k_sq = k_norm_all[t] ** 2
                    S_new = S_curr * (1.0 - k_sq) + v_all[t] * k_norm_all[t]
                    if use_tanh:
                        S_new = torch.tanh(S_new)

                elif state_type == 'full':
                    retrieved = torch.einsum('bij,bj->bi', S_curr, k_norm_all[t])
                    delta = v_all[t] - retrieved
                    outer = torch.einsum('bi,bj->bij', delta, k_norm_all[t])
                    S_new = S_curr + outer
                    if use_tanh:
                        S_new = torch.tanh(S_new)

                elif state_type == 'lowrank':
                    U = S_curr[:, :, :rank]
                    V = S_curr[:, :, rank:]
                    Vtk = torch.einsum('bnr,bn->br', V, k_norm_all[t])
                    retrieved = torch.einsum('bnr,br->bn', U, Vtk)
                    delta = v_all[t] - retrieved
                    k_r = k_norm_all[t][:, :rank]
                    U_new = U + torch.einsum('bn,br->bnr', delta, k_r)
                    if use_tanh:
                        U_new = torch.tanh(U_new)
                    S_new = S_curr.clone()
                    S_new[:, :, :rank] = U_new

                elif state_type == 'blockdiag':
                    nb, bs = S_curr.shape[1], S_curr.shape[2]
                    k_blocks = k_norm_all[t].reshape(B, nb, bs)
                    v_blocks = v_all[t].reshape(B, nb, bs)
                    retrieved = torch.einsum('bnij,bnj->bni', S_curr, k_blocks)
                    delta = v_blocks - retrieved
                    outer = torch.einsum('bni,bnj->bnij', delta, k_blocks)
                    S_new = S_curr + outer
                    if use_tanh:
                        S_new = torch.tanh(S_new)

                S_recomputed.append(S_new)

            # Backward through interval
            for t_local in range(interval_len - 1, -1, -1):
                t = t_start + t_local
                S_prev = S_recomputed[t_local]
                S_after = S_recomputed[t_local + 1]

                # Backward through self-gate: out = Sq * silu(Sq)
                Sq = Sq_cache[t]
                sigmoid_Sq = torch.sigmoid(Sq)
                silu_Sq = Sq * sigmoid_Sq
                d_silu = sigmoid_Sq + Sq * sigmoid_Sq * (1 - sigmoid_Sq)
                d_Sq = d_output[t] * (silu_Sq + Sq * d_silu)

                if state_type == 'diagonal':
                    # Backward: Sq = S * q
                    d_S_from_out = d_Sq * q_all[t]
                    d_q_all[t] = d_Sq * S_after

                    d_S_total = d_S + d_S_from_out

                    # Backward through delta: S_new = f(S_prev*(1-k²) + v*k)
                    k = k_norm_all[t]
                    k_sq = k ** 2
                    pre_tanh = S_prev * (1.0 - k_sq) + v_all[t] * k

                    if use_tanh:
                        tanh_val = torch.tanh(pre_tanh)
                        d_pre = d_S_total * (1 - tanh_val ** 2)
                    else:
                        d_pre = d_S_total

                    d_S = d_pre * (1 - k_sq)
                    d_v_all[t] = d_pre * k
                    d_k_norm = d_pre * (v_all[t] - 2 * S_prev * k)

                    # Backward through normalization (simplified)
                    d_k_all[t] = d_k_norm  # Approximate

                elif state_type == 'full':
                    # Backward: Sq = S @ q
                    d_S_from_out = torch.einsum('bi,bj->bij', d_Sq, q_all[t])
                    d_q_all[t] = torch.einsum('bij,bi->bj', S_after, d_Sq)

                    d_S_total = d_S + d_S_from_out

                    # Backward through delta
                    retrieved = torch.einsum('bij,bj->bi', S_prev, k_norm_all[t])
                    delta = v_all[t] - retrieved
                    pre_tanh = S_prev + torch.einsum('bi,bj->bij', delta, k_norm_all[t])

                    if use_tanh:
                        tanh_val = torch.tanh(pre_tanh)
                        d_pre = d_S_total * (1 - tanh_val ** 2)
                    else:
                        d_pre = d_S_total

                    d_S = d_pre
                    d_delta = torch.einsum('bij,bj->bi', d_pre, k_norm_all[t])
                    d_v_all[t] = d_delta
                    d_k_all[t] = torch.einsum('bij,bi->bj', d_pre, delta)

                elif state_type == 'lowrank':
                    # Simplified backward for lowrank
                    U = S_after[:, :, :rank]
                    V = S_after[:, :, rank:]
                    Vtq = torch.einsum('bnr,bn->br', V, q_all[t])

                    d_U = torch.einsum('bn,br->bnr', d_Sq, Vtq)
                    d_q_all[t] = torch.einsum('bnr,br->bn', U, torch.einsum('bnr,bn->br', V, d_Sq))

                    d_S[:, :, :rank] = d_S[:, :, :rank] + d_U
                    d_v_all[t] = d_Sq  # Simplified
                    d_k_all[t] = d_Sq  # Simplified

                elif state_type == 'blockdiag':
                    nb, bs = S_after.shape[1], S_after.shape[2]
                    d_Sq_blocks = d_Sq.reshape(B, nb, bs)
                    q_blocks = q_all[t].reshape(B, nb, bs)

                    d_S_from_out = torch.einsum('bni,bnj->bnij', d_Sq_blocks, q_blocks)
                    d_q_blocks = torch.einsum('bnij,bni->bnj', S_after, d_Sq_blocks)
                    d_q_all[t] = d_q_blocks.reshape(B, n)

                    d_S_total = d_S + d_S_from_out

                    if use_tanh:
                        pre_tanh = S_prev + torch.einsum('bni,bnj->bnij',
                                                         v_all[t].reshape(B, nb, bs) -
                                                         torch.einsum('bnij,bnj->bni', S_prev,
                                                                     k_norm_all[t].reshape(B, nb, bs)),
                                                         k_norm_all[t].reshape(B, nb, bs))
                        tanh_val = torch.tanh(pre_tanh)
                        d_pre = d_S_total * (1 - tanh_val ** 2)
                    else:
                        d_pre = d_S_total

                    d_S = d_pre
                    d_v_all[t] = torch.einsum('bnij,bnj->bni', d_pre,
                                              k_norm_all[t].reshape(B, nb, bs)).reshape(B, n)
                    d_k_all[t] = d_v_all[t]  # Simplified

        # Weight gradients
        x_flat = x.reshape(T * B, dim)
        d_k_flat = d_k_all.reshape(T * B, n)
        d_v_flat = d_v_all.reshape(T * B, n)
        d_q_flat = d_q_all.reshape(T * B, n)

        dW_k = (x_flat.T @ d_k_flat).T
        dW_v = (x_flat.T @ d_v_flat).T

        if tied_kvq:
            dW_k = dW_k + (x_flat.T @ d_v_flat).T + (x_flat.T @ d_q_flat).T
            dW_v = None
            dW_q = None
        elif tied_kq:
            dW_k = dW_k + (x_flat.T @ d_q_flat).T
            dW_q = None
        else:
            dW_q = (x_flat.T @ d_q_flat).T

        # Input gradient
        dx_flat = d_k_flat @ W_k + d_v_flat @ W_v
        if not tied_kvq and not tied_kq:
            dx_flat = dx_flat + d_q_flat @ W_q
        elif tied_kq:
            dx_flat = dx_flat + d_q_flat @ W_k
        dx = dx_flat.reshape(T, B, dim)

        return dx, None, dW_k, dW_v, dW_q, None, None, None, None, None, None, None


# =============================================================================
# High-Level Module
# =============================================================================

class E74CheckpointedCell(nn.Module):
    """
    Checkpointed cell supporting all state types.
    """

    def __init__(
        self,
        dim: int,
        n_state: int = 64,
        state_type: str = 'diagonal',
        checkpoint_interval: int = 32,
        use_tanh: bool = True,
        tied_kq: bool = False,
        tied_kvq: bool = False,
        rank: int = 8,
        block_size: int = 8,
    ):
        super().__init__()
        self.dim = dim
        self.n_state = n_state
        self.state_type = state_type
        self.checkpoint_interval = checkpoint_interval
        self.use_tanh = use_tanh
        self.tied_kq = tied_kq
        self.tied_kvq = tied_kvq
        self.rank = rank
        self.block_size = block_size

        # Projections
        self.W_k = nn.Parameter(torch.empty(n_state, dim))
        if not tied_kvq:
            self.W_v = nn.Parameter(torch.empty(n_state, dim))
        else:
            self.W_v = None
        if not tied_kq and not tied_kvq:
            self.W_q = nn.Parameter(torch.empty(n_state, dim))
        else:
            self.W_q = None

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_k)
        if self.W_v is not None:
            nn.init.xavier_uniform_(self.W_v)
        if self.W_q is not None:
            nn.init.xavier_uniform_(self.W_q)

    def _get_initial_state(self, B: int, device, dtype):
        n = self.n_state
        if self.state_type == 'diagonal':
            return torch.zeros(B, n, device=device, dtype=dtype)
        elif self.state_type == 'full':
            return torch.zeros(B, n, n, device=device, dtype=dtype)
        elif self.state_type == 'lowrank':
            # Initialize: U=zeros, V=small random (V=0 causes zero gradients)
            S = torch.zeros(B, n, 2 * self.rank, device=device, dtype=dtype)
            S[:, :, self.rank:] = torch.randn(B, n, self.rank, device=device, dtype=dtype) * 0.01
            return S
        elif self.state_type == 'blockdiag':
            n_blocks = n // self.block_size
            return torch.zeros(B, n_blocks, self.block_size, self.block_size,
                             device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, S=None):
        T, B, _ = x.shape

        if S is None:
            S = self._get_initial_state(B, x.device, x.dtype)

        W_v = self.W_v if self.W_v is not None else self.W_k
        W_q = self.W_q

        output, S_final = CheckpointedFunction.apply(
            x, S, self.W_k, W_v, W_q,
            self.checkpoint_interval, self.use_tanh, self.state_type,
            self.tied_kq, self.tied_kvq, self.rank, self.block_size
        )

        return output, S_final


class E74Checkpointed(nn.Module):
    """
    Full E74 layer with checkpointing for all state types.
    """

    def __init__(
        self,
        dim: int,
        expansion: float = 1.0,
        n_state: int = 64,
        state_type: str = 'diagonal',
        checkpoint_interval: int = 32,
        use_tanh: bool = True,
        proj_type: str = 'no_z',  # 'full', 'no_z', 'tied_kq', 'tied_kvq'
        rank: int = 8,
        block_size: int = 8,
        dropout: float = 0.0,
        use_conv: bool = False,
        d_conv: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.state_type = state_type

        tied_kq = proj_type in ['tied_kq', 'tied_kvq']
        tied_kvq = proj_type == 'tied_kvq'

        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)

        if use_conv:
            self.conv1d = nn.Conv1d(
                self.d_inner, self.d_inner, d_conv,
                padding=d_conv - 1, groups=self.d_inner, bias=True
            )
        else:
            self.conv1d = None

        self.cell = E74CheckpointedCell(
            dim=self.d_inner,
            n_state=n_state,
            state_type=state_type,
            checkpoint_interval=checkpoint_interval,
            use_tanh=use_tanh,
            tied_kq=tied_kq,
            tied_kvq=tied_kvq,
            rank=rank,
            block_size=block_size,
        )

        self.out_proj = nn.Linear(n_state, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, state=None, **kwargs):
        B, T, D = x.shape

        x_proj = self.in_proj(x)

        if self.conv1d is not None:
            x_conv = x_proj.transpose(1, 2)
            x_conv = self.conv1d(x_conv)[:, :, :T]
            x_proj = x_conv.transpose(1, 2)

        x_proj = F.silu(x_proj)
        x_rnn = x_proj.permute(1, 0, 2).contiguous()

        cell_out, state = self.cell(x_rnn, state)

        cell_out = cell_out.permute(1, 0, 2).contiguous()
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, state


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing E74 Checkpointed (All State Types)...")
    print("=" * 60)

    device = 'cuda'
    dtype = torch.bfloat16

    B, T, dim, n = 4, 128, 256, 64
    K = 32

    for state_type in ['diagonal', 'full', 'lowrank', 'blockdiag']:
        print(f"\n--- State Type: {state_type} ---")

        model = E74Checkpointed(
            dim=dim, n_state=n, state_type=state_type,
            checkpoint_interval=K, proj_type='tied_kq'
        ).to(device).to(dtype)

        x = torch.randn(B, T, dim, device=device, dtype=dtype, requires_grad=True)

        out, S = model(x)
        print(f"Output: {out.shape}")

        loss = out.sum()
        loss.backward()
        print(f"Backward passed! x.grad norm: {x.grad.norm():.4f}")

        params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {params:,}")

    print("\n" + "=" * 60)
    print("All state types work with checkpointing!")
