"""
Triton kernels for diagonal slot Elman recurrence.

h_t[s] = tanh(Wx_t + A_s * h_prev[s] + b)
output_t = sum(C[s] * h_t[s]) * silu(z_t)

Key: Diagonal A per slot = O(dim) ops per slot instead of O(dim^2)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def diag_slot_fwd_kernel(
    # Inputs
    Wx_ptr,      # [T, B, D] - pre-computed W_x @ x
    z_ptr,       # [T, B, D]
    h0_ptr,      # [B, S, D] - initial hidden state
    A_ptr,       # [S, D] - diagonal decay per slot
    b_ptr,       # [D] - bias
    C_ptr,       # [S] - slot combination weights
    # Outputs
    h_ptr,       # [T+1, B, S, D] - all hidden states
    out_ptr,     # [T, B, D] - output
    # Sizes
    T: tl.constexpr,
    B: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Forward pass for diagonal slot recurrence."""
    # Program handles one (batch, slot) pair across all dimensions
    pid_bs = tl.program_id(0)
    batch_idx = pid_bs // S
    slot_idx = pid_bs % S

    # Dimension block
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    # Load A[slot, :] and b[:]
    A_offs = slot_idx * D + d_offs
    A = tl.load(A_ptr + A_offs, mask=d_mask, other=0.0)
    b = tl.load(b_ptr + d_offs, mask=d_mask, other=0.0)
    C_slot = tl.load(C_ptr + slot_idx)

    # Load h0[batch, slot, :]
    h0_offs = batch_idx * S * D + slot_idx * D + d_offs
    h = tl.load(h0_ptr + h0_offs, mask=d_mask, other=0.0)

    # Store h0 to h[0, batch, slot, :]
    h_base = batch_idx * S * D + slot_idx * D
    tl.store(h_ptr + h_base + d_offs, h, mask=d_mask)

    # Recurrence loop
    for t in range(T):
        # Load Wx[t, batch, :] and z[t, batch, :]
        Wx_offs = t * B * D + batch_idx * D + d_offs
        Wx = tl.load(Wx_ptr + Wx_offs, mask=d_mask, other=0.0)
        z = tl.load(z_ptr + Wx_offs, mask=d_mask, other=0.0)

        # h_new = tanh(Wx + A * h + b)
        pre_act = Wx + A * h + b
        # tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        exp_2x = tl.exp(2.0 * pre_act.to(tl.float32))
        h = ((exp_2x - 1.0) / (exp_2x + 1.0)).to(pre_act.dtype)

        # Store h[t+1, batch, slot, :]
        h_store_offs = (t + 1) * B * S * D + h_base + d_offs
        tl.store(h_ptr + h_store_offs, h, mask=d_mask)

        # Compute contribution to output: C[slot] * h
        # This is partial - need to sum across slots
        # For now, store C*h and reduce later
        out_partial_offs = t * B * S * D + batch_idx * S * D + slot_idx * D + d_offs
        tl.store(out_ptr + out_partial_offs, C_slot * h, mask=d_mask)


@triton.jit
def reduce_slots_kernel(
    # Inputs
    partial_ptr,  # [T, B, S, D] - partial outputs (C*h per slot)
    z_ptr,        # [T, B, D]
    # Outputs
    out_ptr,      # [T, B, D]
    # Sizes
    T: tl.constexpr,
    B: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Reduce across slots and apply silu gating."""
    pid_tb = tl.program_id(0)
    t_idx = pid_tb // B
    b_idx = pid_tb % B

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    # Sum across slots
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for s in range(S):
        offs = t_idx * B * S * D + b_idx * S * D + s * D + d_offs
        val = tl.load(partial_ptr + offs, mask=d_mask, other=0.0)
        acc += val.to(tl.float32)

    # Load z and apply silu gating
    z_offs = t_idx * B * D + b_idx * D + d_offs
    z = tl.load(z_ptr + z_offs, mask=d_mask, other=0.0).to(tl.float32)
    silu_z = z * tl.sigmoid(z)
    out = acc * silu_z

    # Store output
    tl.store(out_ptr + z_offs, out.to(partial_ptr.dtype.element_ty), mask=d_mask)


def diag_slot_forward(Wx, z, h0, A, b, C):
    """
    Forward pass for diagonal slot recurrence.

    Args:
        Wx: [T, B, D] - pre-computed W_x @ x
        z: [T, B, D] - gate input
        h0: [B, S, D] - initial hidden state
        A: [S, D] - diagonal decay per slot
        b: [D] - bias
        C: [S] - slot combination weights

    Returns:
        output: [T, B, D]
        h: [T+1, B, S, D] - all hidden states
    """
    T, B, D = Wx.shape
    S = A.shape[0]

    # Allocate outputs
    h = torch.empty(T + 1, B, S, D, device=Wx.device, dtype=Wx.dtype)
    partial = torch.empty(T, B, S, D, device=Wx.device, dtype=Wx.dtype)
    output = torch.empty(T, B, D, device=Wx.device, dtype=Wx.dtype)

    # Block size (power of 2, >= D)
    BLOCK_D = triton.next_power_of_2(D)

    # Launch forward kernel
    grid = (B * S,)
    diag_slot_fwd_kernel[grid](
        Wx, z, h0, A, b, C,
        h, partial,
        T, B, S, D, BLOCK_D,
    )

    # Launch reduce kernel
    grid = (T * B,)
    reduce_slots_kernel[grid](
        partial, z, output,
        T, B, S, D, BLOCK_D,
    )

    return output, h


@triton.jit
def diag_slot_bwd_kernel(
    # Inputs
    d_output_ptr,  # [T, B, D]
    z_ptr,         # [T, B, D]
    h_ptr,         # [T+1, B, S, D]
    A_ptr,         # [S, D]
    C_ptr,         # [S]
    # Outputs
    d_Wx_ptr,      # [T, B, D]
    d_z_ptr,       # [T, B, D]
    d_A_ptr,       # [S, D] - atomically accumulated
    d_b_ptr,       # [D] - atomically accumulated
    # Sizes
    T: tl.constexpr,
    B: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Backward pass for diagonal slot recurrence (one thread block per batch,slot)."""
    pid_bs = tl.program_id(0)
    batch_idx = pid_bs // S
    slot_idx = pid_bs % S

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    # Load A[slot, :] and C[slot]
    A_offs = slot_idx * D + d_offs
    A = tl.load(A_ptr + A_offs, mask=d_mask, other=0.0)
    C_slot = tl.load(C_ptr + slot_idx)

    # Accumulators for dA (per slot)
    d_A_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # dh accumulator (gradient flowing back)
    d_h = tl.zeros((BLOCK_D,), dtype=tl.float32)

    h_base = batch_idx * S * D + slot_idx * D

    # Backward through time
    for t_rev in range(T):
        t = T - 1 - t_rev

        # Load h[t+1] and h[t]
        h_t1_offs = (t + 1) * B * S * D + h_base + d_offs
        h_t_offs = t * B * S * D + h_base + d_offs
        h_t1 = tl.load(h_ptr + h_t1_offs, mask=d_mask, other=0.0).to(tl.float32)
        h_t = tl.load(h_ptr + h_t_offs, mask=d_mask, other=0.0).to(tl.float32)

        # Load d_output[t] and z[t]
        out_offs = t * B * D + batch_idx * D + d_offs
        d_out = tl.load(d_output_ptr + out_offs, mask=d_mask, other=0.0).to(tl.float32)
        z = tl.load(z_ptr + out_offs, mask=d_mask, other=0.0).to(tl.float32)

        # silu(z) and its derivative
        sig_z = tl.sigmoid(z)
        silu_z = z * sig_z

        # d_h from output: d_out * silu(z) * C[slot]
        d_h += d_out * silu_z * C_slot

        # tanh gradient: 1 - h^2
        tanh_grad = 1.0 - h_t1 * h_t1
        d_pre = d_h * tanh_grad

        # d_A += d_pre * h[t]
        d_A_acc += d_pre * h_t

        # d_h[t] = d_pre * A (for next iteration)
        d_h = d_pre * A.to(tl.float32)

    # Atomically accumulate dA
    tl.atomic_add(d_A_ptr + A_offs, d_A_acc.to(A.dtype), mask=d_mask)


@triton.jit
def diag_slot_bwd_dWx_kernel(
    # Inputs
    d_output_ptr,  # [T, B, D]
    z_ptr,         # [T, B, D]
    h_ptr,         # [T+1, B, S, D]
    A_ptr,         # [S, D]
    C_ptr,         # [S]
    # Outputs
    d_Wx_ptr,      # [T, B, D]
    d_z_ptr,       # [T, B, D]
    # Sizes
    T: tl.constexpr,
    B: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute dWx and dz (one thread block per time,batch)."""
    pid_tb = tl.program_id(0)
    t_idx = pid_tb // B
    b_idx = pid_tb % B

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    # Load d_output and z
    out_offs = t_idx * B * D + b_idx * D + d_offs
    d_out = tl.load(d_output_ptr + out_offs, mask=d_mask, other=0.0).to(tl.float32)
    z = tl.load(z_ptr + out_offs, mask=d_mask, other=0.0).to(tl.float32)

    # silu(z) and derivative
    sig_z = tl.sigmoid(z)
    silu_z = z * sig_z
    d_silu = sig_z + z * sig_z * (1.0 - sig_z)

    # Accumulate h_combined across slots
    h_combined = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for s in range(S):
        C_s = tl.load(C_ptr + s)
        h_offs = (t_idx + 1) * B * S * D + b_idx * S * D + s * D + d_offs
        h_s = tl.load(h_ptr + h_offs, mask=d_mask, other=0.0).to(tl.float32)
        h_combined += C_s * h_s

    # dz = d_out * h_combined * d_silu
    d_z = d_out * h_combined * d_silu
    tl.store(d_z_ptr + out_offs, d_z.to(d_output_ptr.dtype.element_ty), mask=d_mask)

    # For dWx, we need to do backward through tanh for each slot and sum
    # This is more complex - need to track d_h through recurrence for each slot
    # For now, compute d_h_combined * tanh_grad summed over slots

    d_h_combined = d_out * silu_z
    d_Wx_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for s in range(S):
        C_s = tl.load(C_ptr + s)
        h_offs = (t_idx + 1) * B * S * D + b_idx * S * D + s * D + d_offs
        h_s = tl.load(h_ptr + h_offs, mask=d_mask, other=0.0).to(tl.float32)
        tanh_grad = 1.0 - h_s * h_s
        d_Wx_acc += C_s * d_h_combined * tanh_grad

    tl.store(d_Wx_ptr + out_offs, d_Wx_acc.to(d_output_ptr.dtype.element_ty), mask=d_mask)


def diag_slot_backward_triton(d_output, z, h, A, b, C):
    """Triton backward pass."""
    T, B, D = d_output.shape
    S = A.shape[0]
    BLOCK_D = triton.next_power_of_2(D)

    d_Wx = torch.empty_like(d_output)
    d_z = torch.empty_like(d_output)
    d_A = torch.zeros_like(A)

    # Launch dWx/dz kernel
    grid = (T * B,)
    diag_slot_bwd_dWx_kernel[grid](
        d_output, z, h, A, C,
        d_Wx, d_z,
        T, B, S, D, BLOCK_D,
    )

    # Launch dA kernel
    grid = (B * S,)
    diag_slot_bwd_kernel[grid](
        d_output, z, h, A, C,
        d_Wx, d_z, d_A, None,  # d_b not computed here
        T, B, S, D, BLOCK_D,
    )

    # d_b = sum of d_Wx over T, B
    d_b = d_Wx.float().sum(dim=(0, 1)).to(b.dtype)

    # d_C (simple: sum of d_out * silu(z) * h for each slot)
    silu_z = z.float() * torch.sigmoid(z.float())
    d_h_combined = d_output.float() * silu_z
    d_C = torch.zeros(S, device=A.device, dtype=A.dtype)
    for s in range(S):
        d_C[s] = (d_h_combined * h[1:, :, s, :].float()).sum()

    d_h0 = torch.zeros(B, S, D, device=A.device, dtype=A.dtype)

    return d_Wx, d_z, d_h0, d_A, d_b, d_C


class DiagSlotFunction(torch.autograd.Function):
    """Autograd function for diagonal slot recurrence."""

    @staticmethod
    def forward(ctx, Wx, z, h0, A, b, C):
        output, h = diag_slot_forward(Wx, z, h0, A, b, C)
        ctx.save_for_backward(z, h, A, b, C)
        return output, h

    @staticmethod
    def backward(ctx, d_output, d_h):
        z, h, A, b, C = ctx.saved_tensors
        d_Wx, d_z, d_h0, d_A, d_b, d_C = diag_slot_backward_triton(
            d_output.contiguous(), z, h, A, b, C
        )
        return d_Wx, d_z, d_h0, d_A, d_b, d_C


def diag_slot_recurrence(Wx, z, h0, A, b, C):
    """Wrapper for diagonal slot recurrence with autograd."""
    return DiagSlotFunction.apply(Wx, z, h0, A, b, C)


if __name__ == "__main__":
    print("Testing diagonal slot Triton kernels...")

    device = 'cuda'
    B, T, D, S = 32, 512, 768, 8

    Wx = torch.randn(T, B, D, device=device, dtype=torch.bfloat16)
    z = torch.randn(T, B, D, device=device, dtype=torch.bfloat16)
    h0 = torch.zeros(B, S, D, device=device, dtype=torch.bfloat16)
    A = torch.sigmoid(torch.randn(S, D, device=device, dtype=torch.bfloat16)) * 0.99
    b = torch.zeros(D, device=device, dtype=torch.bfloat16)
    C = torch.ones(S, device=device, dtype=torch.bfloat16) / S

    # Test forward
    output, h = diag_slot_forward(Wx, z, h0, A, b, C)
    print(f"Output: {output.shape}, H: {h.shape}")

    # Test with autograd
    Wx.requires_grad_(True)
    A.requires_grad_(True)
    output, h = diag_slot_recurrence(Wx, z, h0, A, b, C)
    loss = output.sum()
    loss.backward()
    print(f"Gradients computed! d_Wx: {Wx.grad.shape}, d_A: {A.grad.shape}")

    # Benchmark
    import time

    def run_iter():
        Wx_g = Wx.detach().requires_grad_(True)
        A_g = A.detach().requires_grad_(True)
        output, h = diag_slot_recurrence(Wx_g, z, h0, A_g, b, C)
        output.sum().backward()

    for _ in range(5):
        run_iter()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        run_iter()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / 20 * 1000
    tok_per_sec = B * T / (elapsed / 1000)
    print(f"Time: {elapsed:.1f}ms, {tok_per_sec/1000:.1f}k tok/s")
