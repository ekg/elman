"""
E23 Dual-Memory Elman - Fully fused Triton kernel.

Fuses all per-timestep operations into a single Triton kernel launch per timestep:
- Read attention (dot product + softmax)
- Update h_work (W_h @ h + x_proj + read + bias, then tanh)
- Write value (W_write @ h_work_new)
- Write attention + tape update

This eliminates Python overhead between operations.
"""

import math
import torch
import triton
import triton.language as tl


@triton.jit
def e23_fused_step_kernel(
    # Inputs (for this timestep)
    h_tape_ptr,        # [B, N, D] - tape memory (modified in-place)
    h_work_ptr,        # [B, D] - working memory (modified in-place)
    x_proj_ptr,        # [B, D] - x @ W_x.T for this timestep
    # Weights
    W_h_ptr,           # [D, D]
    b_h_ptr,           # [D]
    W_write_ptr,       # [D, D]
    # Outputs
    h_work_out_ptr,    # [B, D] - output h_work for this step
    read_attn_ptr,     # [B, N] - read attention
    write_attn_ptr,    # [B, N] - write attention
    # Dimensions
    B, N: tl.constexpr, D: tl.constexpr,
    scale,
    # Block sizes
    BLOCK_D: tl.constexpr,
):
    """
    Fully fused E23 forward step for one timestep.
    One program per batch element.
    """
    b = tl.program_id(0)
    if b >= B:
        return

    d_idx = tl.arange(0, BLOCK_D)
    n_idx = tl.arange(0, 64)  # Max N
    n_mask = n_idx < N

    # Load h_work [D]
    h_work = tl.load(h_work_ptr + b * D + d_idx, mask=d_idx < D, other=0.0)

    # ============================================
    # STEP 1: READ FROM TAPE
    # ============================================
    read_scores = tl.zeros((64,), dtype=tl.float32)
    for n in range(N):
        tape_row = tl.load(h_tape_ptr + b * N * D + n * D + d_idx, mask=d_idx < D, other=0.0)
        score = tl.sum(tape_row * h_work, axis=0)
        read_scores = tl.where(n_idx == n, score * scale, read_scores)

    # Softmax
    max_score = tl.max(tl.where(n_mask, read_scores, float('-inf')), axis=0)
    exp_scores = tl.math.exp(read_scores - max_score)
    exp_scores = tl.where(n_mask, exp_scores, 0.0)
    read_attn = exp_scores / tl.sum(exp_scores, axis=0)
    tl.store(read_attn_ptr + b * N + n_idx, read_attn, mask=n_mask)

    # Weighted read
    read_val = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for n in range(N):
        tape_row = tl.load(h_tape_ptr + b * N * D + n * D + d_idx, mask=d_idx < D, other=0.0)
        attn_n = tl.sum(tl.where(n_idx == n, read_attn, 0.0), axis=0)
        read_val = read_val + attn_n * tape_row

    # ============================================
    # STEP 2: UPDATE WORKING MEMORY (includes W_h @ h GEMM)
    # ============================================
    # Load x_proj, bias
    x_proj = tl.load(x_proj_ptr + b * D + d_idx, mask=d_idx < D, other=0.0)
    bias = tl.load(b_h_ptr + d_idx, mask=d_idx < D, other=0.0)

    # Compute W_h @ h_work (naive GEMM: one row at a time)
    Rh = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for i in range(D):
        w_h_row = tl.load(W_h_ptr + i * D + d_idx, mask=d_idx < D, other=0.0)
        dot = tl.sum(w_h_row * h_work, axis=0)
        Rh = tl.where(d_idx == i, dot, Rh)

    # pre_act = Rh + x_proj + read_val + bias
    pre_act = Rh + x_proj + read_val + bias

    # tanh
    exp_neg_2x = tl.math.exp(-2.0 * pre_act)
    h_work_new = 2.0 / (1.0 + exp_neg_2x) - 1.0

    # Store h_work output
    tl.store(h_work_out_ptr + b * D + d_idx, h_work_new, mask=d_idx < D)

    # ============================================
    # STEP 3: WRITE TO TAPE (includes W_write @ h GEMM)
    # ============================================
    # Compute write_val = W_write @ h_work_new
    write_val = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for i in range(D):
        w_write_row = tl.load(W_write_ptr + i * D + d_idx, mask=d_idx < D, other=0.0)
        dot = tl.sum(w_write_row * h_work_new, axis=0)
        write_val = tl.where(d_idx == i, dot, write_val)

    # Write attention scores
    write_scores = tl.zeros((64,), dtype=tl.float32)
    for n in range(N):
        tape_row = tl.load(h_tape_ptr + b * N * D + n * D + d_idx, mask=d_idx < D, other=0.0)
        score = tl.sum(tape_row * h_work_new, axis=0)
        write_scores = tl.where(n_idx == n, score * scale, write_scores)

    # Softmax
    max_score = tl.max(tl.where(n_mask, write_scores, float('-inf')), axis=0)
    exp_scores = tl.math.exp(write_scores - max_score)
    exp_scores = tl.where(n_mask, exp_scores, 0.0)
    write_attn = exp_scores / tl.sum(exp_scores, axis=0)
    tl.store(write_attn_ptr + b * N + n_idx, write_attn, mask=n_mask)

    # Update tape
    for n in range(N):
        tape_row = tl.load(h_tape_ptr + b * N * D + n * D + d_idx, mask=d_idx < D, other=0.0)
        attn_n = tl.sum(tl.where(n_idx == n, write_attn, 0.0), axis=0)
        new_tape_row = (1.0 - attn_n) * tape_row + attn_n * write_val
        tl.store(h_tape_ptr + b * N * D + n * D + d_idx, new_tape_row, mask=d_idx < D)

    # Update h_work for next step
    tl.store(h_work_ptr + b * D + d_idx, h_work_new, mask=d_idx < D)


def e23_forward_triton_fused(
    x_seq: torch.Tensor,       # [B, T, D]
    h_tape_init: torch.Tensor, # [B, N, D]
    h_work_init: torch.Tensor, # [B, D]
    W_h: torch.Tensor,         # [D, D]
    W_x: torch.Tensor,         # [D, D]
    b_h: torch.Tensor,         # [D]
    W_write: torch.Tensor,     # [D, D]
    training: bool = True,
):
    """Fused Triton E23 forward - one kernel launch per timestep."""
    B, T, D = x_seq.shape
    N = h_tape_init.shape[1]
    scale = 1.0 / math.sqrt(D)

    # Convert to float32
    dtype_orig = x_seq.dtype
    x_seq = x_seq.float()
    h_tape = h_tape_init.float().clone()
    h_work = h_work_init.float().clone()
    W_h = W_h.float().contiguous()
    W_x = W_x.float()
    b_h = b_h.float().contiguous()
    W_write = W_write.float().contiguous()

    # Pre-compute x @ W_x.T
    x_proj_all = x_seq @ W_x.T  # [B, T, D]

    # Allocate outputs
    h_work_all = torch.empty(B, T, D, device=x_seq.device, dtype=torch.float32)
    read_attn_all = torch.empty(B, T, N, device=x_seq.device, dtype=torch.float32)
    write_attn_all = torch.empty(B, T, N, device=x_seq.device, dtype=torch.float32)

    BLOCK_D = triton.next_power_of_2(D)
    grid = (B,)

    for t in range(T):
        e23_fused_step_kernel[grid](
            h_tape, h_work, x_proj_all[:, t].contiguous(),
            W_h, b_h, W_write,
            h_work_all[:, t], read_attn_all[:, t], write_attn_all[:, t],
            B, N, D, scale, BLOCK_D=BLOCK_D,
        )

    return (
        h_work_all.to(dtype_orig),
        h_tape.unsqueeze(1).to(dtype_orig),  # Final tape only
        read_attn_all.to(dtype_orig),
        write_attn_all.to(dtype_orig),
    )


if __name__ == "__main__":
    print("Testing E23 fused Triton kernel...")

    device = 'cuda'
    B, T, D, N = 4, 512, 512, 8

    torch.manual_seed(42)
    x = torch.randn(B, T, D, device=device, dtype=torch.bfloat16)
    h_tape = torch.randn(B, N, D, device=device, dtype=torch.bfloat16)
    h_work = torch.randn(B, D, device=device, dtype=torch.bfloat16)
    W_h = torch.randn(D, D, device=device, dtype=torch.bfloat16)
    W_x = torch.randn(D, D, device=device, dtype=torch.bfloat16)
    b_h = torch.randn(D, device=device, dtype=torch.bfloat16)
    W_write = torch.randn(D, D, device=device, dtype=torch.bfloat16)

    # Warmup
    print("Warming up...")
    for _ in range(3):
        result = e23_forward_triton_fused(x, h_tape, h_work, W_h, W_x, b_h, W_write)

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start.record()
    for _ in range(10):
        result = e23_forward_triton_fused(x, h_tape, h_work, W_h, W_x, b_h, W_write)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / 10
    print(f"Fused Triton: {ms:.2f}ms ({ms/T*1000:.1f}us/step)")

    # Compare with CUDA
    import sys
    sys.path.insert(0, 'elman/cuda')
    import hasty_pytorch_lib as hasty

    for _ in range(3):
        hasty.dual_memory_elman_forward(True, x, h_tape, h_work, W_h, W_x, b_h, W_write)

    torch.cuda.synchronize()
    start.record()
    for _ in range(10):
        hasty.dual_memory_elman_forward(True, x, h_tape, h_work, W_h, W_x, b_h, W_write)
    end.record()
    torch.cuda.synchronize()

    cuda_ms = start.elapsed_time(end) / 10
    print(f"CUDA kernel: {cuda_ms:.2f}ms ({cuda_ms/T*1000:.1f}us/step)")
    print(f"Triton/CUDA ratio: {ms/cuda_ms:.2f}x")
