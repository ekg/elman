"""
E23 Dual-Memory Elman - Optimized Triton kernels.

Strategy (matching CUDA):
1. Pre-compute x @ W_x.T for all T (batch GEMM via PyTorch/cuBLAS)
2. Per timestep:
   - Triton Phase1: read attention + accumulate pre_act
   - cuBLAS: W_h @ h_work
   - Triton Phase2: write attention + tape update
   - cuBLAS: W_write @ h_work_new

This minimizes Python overhead by keeping the loop in Triton where possible.
"""

import math
import torch
import triton
import triton.language as tl


@triton.jit
def e23_phase1_kernel(
    # Inputs
    h_tape_ptr,        # [B, N, D] - tape memory
    h_work_ptr,        # [B, D] - working memory
    Rh_ptr,            # [B, D] - W_h @ h_work (pre-computed)
    x_proj_t_ptr,      # [B, D] - W_x @ x[t] (pre-computed)
    b_h_ptr,           # [D] - bias
    # Outputs
    h_work_out_ptr,    # [B, D] - new working memory
    read_attn_ptr,     # [B, N] - read attention weights
    # Dimensions
    B, N: tl.constexpr, D: tl.constexpr,
    # Scale
    scale,
    # Block size
    BLOCK_D: tl.constexpr,
):
    """
    Phase 1: Read from tape + Update working memory

    Operations:
    1. Read attention: attn = softmax(h_tape @ h_work * scale)
    2. Read value: read = sum(attn * h_tape)
    3. Update: h_work_new = tanh(Rh + x_proj_t + read + b_h)
    """
    b = tl.program_id(0)
    if b >= B:
        return

    d_idx = tl.arange(0, BLOCK_D)
    n_idx = tl.arange(0, 64)  # Max N=64

    # Load h_work [D]
    h_work = tl.load(h_work_ptr + b * D + d_idx, mask=d_idx < D, other=0.0)

    # Compute read attention scores and weighted sum simultaneously
    # For each slot n: score[n] = sum_d(h_tape[n,d] * h_work[d])
    read_scores = tl.zeros((64,), dtype=tl.float32)  # Max N=64
    read_val = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # Pass 1: Compute scores
    for n in range(N):
        tape_row = tl.load(h_tape_ptr + b * N * D + n * D + d_idx, mask=d_idx < D, other=0.0)
        score = tl.sum(tape_row * h_work, axis=0)
        # Store score at position n
        read_scores = tl.where(tl.arange(0, 64) == n, score * scale, read_scores)

    # Softmax over N slots (mask positions >= N)
    n_mask = n_idx < N
    max_score = tl.max(tl.where(n_mask, read_scores, float('-inf')), axis=0)
    exp_scores = tl.exp(read_scores - max_score)
    exp_scores = tl.where(n_mask, exp_scores, 0.0)
    sum_exp = tl.sum(exp_scores, axis=0)
    read_attn = exp_scores / sum_exp

    # Store read attention [N]
    tl.store(read_attn_ptr + b * N + n_idx, read_attn, mask=n_idx < N)

    # Pass 2: Compute weighted read
    for n in range(N):
        tape_row = tl.load(h_tape_ptr + b * N * D + n * D + d_idx, mask=d_idx < D, other=0.0)
        attn_n = tl.sum(tl.where(n_idx == n, read_attn, 0.0), axis=0)
        read_val = read_val + attn_n * tape_row

    # Load Rh, x_proj_t, b_h
    Rh = tl.load(Rh_ptr + b * D + d_idx, mask=d_idx < D, other=0.0)
    x_proj = tl.load(x_proj_t_ptr + b * D + d_idx, mask=d_idx < D, other=0.0)
    bias = tl.load(b_h_ptr + d_idx, mask=d_idx < D, other=0.0)

    # Compute h_work_new = tanh(Rh + x_proj + read + bias)
    pre_act = Rh + x_proj + read_val + bias
    # tanh(x) = 2*sigmoid(2x) - 1 = 2/(1+exp(-2x)) - 1
    exp_neg_2x = tl.math.exp(-2.0 * pre_act)
    h_work_new = 2.0 / (1.0 + exp_neg_2x) - 1.0

    # Store output
    tl.store(h_work_out_ptr + b * D + d_idx, h_work_new, mask=d_idx < D)


@triton.jit
def e23_phase2_kernel(
    # Inputs
    h_tape_ptr,        # [B, N, D] - tape memory (will be modified)
    h_work_new_ptr,    # [B, D] - new working memory
    write_val_ptr,     # [B, D] - W_write @ h_work_new (pre-computed)
    # Outputs
    write_attn_ptr,    # [B, N] - write attention weights
    # Dimensions
    B, N: tl.constexpr, D: tl.constexpr,
    # Scale
    scale,
    # Block size
    BLOCK_D: tl.constexpr,
):
    """
    Phase 2: Write to tape

    Operations:
    1. Write attention: attn = softmax(h_tape @ h_work_new * scale)
    2. Update tape: h_tape = (1-attn) * h_tape + attn * write_val
    """
    b = tl.program_id(0)
    if b >= B:
        return

    d_idx = tl.arange(0, BLOCK_D)
    n_idx = tl.arange(0, 64)

    # Load h_work_new and write_val
    h_work_new = tl.load(h_work_new_ptr + b * D + d_idx, mask=d_idx < D, other=0.0)
    write_val = tl.load(write_val_ptr + b * D + d_idx, mask=d_idx < D, other=0.0)

    # Compute write attention scores
    write_scores = tl.zeros((64,), dtype=tl.float32)
    for n in range(N):
        tape_row = tl.load(h_tape_ptr + b * N * D + n * D + d_idx, mask=d_idx < D, other=0.0)
        score = tl.sum(tape_row * h_work_new, axis=0)
        write_scores = tl.where(n_idx == n, score * scale, write_scores)

    # Softmax (mask positions >= N)
    n_mask = n_idx < N
    max_score = tl.max(tl.where(n_mask, write_scores, float('-inf')), axis=0)
    exp_scores = tl.exp(write_scores - max_score)
    exp_scores = tl.where(n_mask, exp_scores, 0.0)
    sum_exp = tl.sum(exp_scores, axis=0)
    write_attn = exp_scores / sum_exp

    # Store write attention
    tl.store(write_attn_ptr + b * N + n_idx, write_attn, mask=n_idx < N)

    # Update tape: h_tape = (1 - attn) * h_tape + attn * write_val
    for n in range(N):
        tape_row = tl.load(h_tape_ptr + b * N * D + n * D + d_idx, mask=d_idx < D, other=0.0)
        attn_n = tl.sum(tl.where(n_idx == n, write_attn, 0.0), axis=0)
        new_tape_row = (1.0 - attn_n) * tape_row + attn_n * write_val
        tl.store(h_tape_ptr + b * N * D + n * D + d_idx, new_tape_row, mask=d_idx < D)


def e23_forward_triton_optimized(
    x_seq: torch.Tensor,       # [B, T, D]
    h_tape_init: torch.Tensor, # [B, N, D]
    h_work_init: torch.Tensor, # [B, D]
    W_h: torch.Tensor,         # [D, D]
    W_x: torch.Tensor,         # [D, D]
    b_h: torch.Tensor,         # [D]
    W_write: torch.Tensor,     # [D, D]
    training: bool = True,
):
    """
    Optimized Triton E23 forward pass.

    Uses cuBLAS for GEMMs + Triton for attention/update ops.
    """
    B, T, D = x_seq.shape
    N = h_tape_init.shape[1]
    scale = 1.0 / math.sqrt(D)

    # Convert to float32 for Triton
    dtype_orig = x_seq.dtype
    x_seq = x_seq.float()
    h_tape = h_tape_init.float().clone()
    h_work = h_work_init.float().clone()
    W_h = W_h.float()
    W_x = W_x.float()
    b_h = b_h.float()
    W_write = W_write.float()

    # Pre-compute x @ W_x.T for all timesteps (batch GEMM)
    x_proj = x_seq @ W_x.T  # [B, T, D]

    # Allocate outputs
    h_work_all = torch.empty(B, T, D, device=x_seq.device, dtype=torch.float32)
    read_attn_all = torch.empty(B, T, N, device=x_seq.device, dtype=torch.float32)
    write_attn_all = torch.empty(B, T, N, device=x_seq.device, dtype=torch.float32)

    # Workspace
    Rh = torch.empty(B, D, device=x_seq.device, dtype=torch.float32)
    write_val = torch.empty(B, D, device=x_seq.device, dtype=torch.float32)
    h_work_new = torch.empty(B, D, device=x_seq.device, dtype=torch.float32)

    # Tape history for backward
    if training:
        h_tape_all = torch.empty(B, T + 1, N, D, device=x_seq.device, dtype=torch.float32)
        h_tape_all[:, 0] = h_tape
    else:
        h_tape_all = None

    # Block size
    BLOCK_D = triton.next_power_of_2(D)
    grid = (B,)

    for t in range(T):
        # cuBLAS: Rh = h_work @ W_h.T
        torch.mm(h_work, W_h.T, out=Rh)

        # Triton Phase 1: read attention + update h_work
        e23_phase1_kernel[grid](
            h_tape, h_work, Rh, x_proj[:, t].contiguous(), b_h,
            h_work_new, read_attn_all[:, t],
            B, N, D, scale, BLOCK_D=BLOCK_D,
        )

        # cuBLAS: write_val = h_work_new @ W_write.T
        torch.mm(h_work_new, W_write.T, out=write_val)

        # Triton Phase 2: write attention + tape update
        e23_phase2_kernel[grid](
            h_tape, h_work_new, write_val,
            write_attn_all[:, t],
            B, N, D, scale, BLOCK_D=BLOCK_D,
        )

        # Store h_work output
        h_work_all[:, t] = h_work_new

        # Store tape history
        if training:
            h_tape_all[:, t + 1] = h_tape

        # Update h_work for next iteration
        h_work.copy_(h_work_new)

    return (
        h_work_all.to(dtype_orig),
        h_tape_all.to(dtype_orig) if h_tape_all is not None else h_tape.unsqueeze(1).to(dtype_orig),
        read_attn_all.to(dtype_orig),
        write_attn_all.to(dtype_orig),
    )


if __name__ == "__main__":
    print("Testing E23 Triton kernels...")

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
    for _ in range(3):
        result = e23_forward_triton_optimized(x, h_tape, h_work, W_h, W_x, b_h, W_write)

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start.record()
    for _ in range(10):
        result = e23_forward_triton_optimized(x, h_tape, h_work, W_h, W_x, b_h, W_write)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / 10
    print(f"Optimized Triton: {ms:.2f}ms ({ms/T*1000:.1f}us/step)")
