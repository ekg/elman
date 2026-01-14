#!/usr/bin/env python3
"""
Isolated backward pass profiling for E23.

Measure backward time more precisely by:
1. Pre-computing forward pass outputs
2. Timing only the backward pass
3. Multiple runs to ensure stable timing
"""
import torch
import sys
sys.path.insert(0, 'elman/cuda')

import hasty_pytorch_lib as hasty

batch_size = 64
seq_len = 512
dim = 512

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

print("=" * 70)
print("E23 Isolated Backward Profiling")
print(f"batch={batch_size}, seq={seq_len}, dim={dim}")
print("=" * 70)

for n_slots in [8, 16, 32]:
    print(f"\nn_slots={n_slots}")
    print("-" * 50)

    # Create inputs
    W_h = torch.randn(dim, dim, device='cuda', dtype=torch.bfloat16)
    W_x = torch.randn(dim, dim, device='cuda', dtype=torch.bfloat16)
    b_h = torch.randn(dim, device='cuda', dtype=torch.bfloat16)
    W_write = torch.randn(dim, dim, device='cuda', dtype=torch.bfloat16)
    h_tape_init = torch.randn(batch_size, n_slots, dim, device='cuda', dtype=torch.bfloat16)
    h_work_init = torch.randn(batch_size, dim, device='cuda', dtype=torch.bfloat16)
    x_seq = torch.randn(batch_size, seq_len, dim, device='cuda', dtype=torch.bfloat16)

    # Pre-compute forward pass
    result = hasty.dual_memory_elman_forward(True, x_seq, h_tape_init, h_work_init, W_h, W_x, b_h, W_write)
    h_work_out, h_tape_final, h_tape_all, read_attn, write_attn = result

    # Create gradient output (upstream gradient)
    d_h_work_out = torch.randn_like(h_work_out)
    d_h_tape_final = torch.randn_like(h_tape_final)

    # Precompute x_proj (needed for backward call)
    x_proj = x_seq @ W_x.T

    # Warmup backward
    for _ in range(5):
        dx_proj, dW_h, db_h, dW_write = hasty.dual_memory_elman_backward(
            x_proj, h_work_out, h_tape_all, read_attn, write_attn,
            W_h, W_write, d_h_work_out, d_h_tape_final)

    torch.cuda.synchronize()

    # Time backward - multiple runs
    times = []
    for run in range(10):
        torch.cuda.synchronize()
        start.record()
        dx_proj, dW_h, db_h, dW_write = hasty.dual_memory_elman_backward(
            x_proj, h_work_out, h_tape_all, read_attn, write_attn,
            W_h, W_write, d_h_work_out, d_h_tape_final)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    mean_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    print(f"  Backward: mean={mean_time:.2f}ms, min={min_time:.2f}ms, max={max_time:.2f}ms")

    # Also check output shapes
    print(f"  dx_proj shape: {dx_proj.shape}")
    print(f"  dW_h shape: {dW_h.shape}")
    print(f"  dW_write shape: {dW_write.shape}")

    # Check for NaN
    has_nan = any([dx_proj.isnan().any(), dW_h.isnan().any(), db_h.isnan().any(), dW_write.isnan().any()])
    print(f"  Has NaN: {has_nan}")

print("\n" + "=" * 70)
print("Testing with smaller batch to check scaling")
print("=" * 70)

for batch in [4, 16, 32, 64]:
    for n_slots in [8, 16]:
        # Create inputs
        W_h = torch.randn(dim, dim, device='cuda', dtype=torch.bfloat16)
        W_x = torch.randn(dim, dim, device='cuda', dtype=torch.bfloat16)
        b_h = torch.randn(dim, device='cuda', dtype=torch.bfloat16)
        W_write = torch.randn(dim, dim, device='cuda', dtype=torch.bfloat16)
        h_tape_init = torch.randn(batch, n_slots, dim, device='cuda', dtype=torch.bfloat16)
        h_work_init = torch.randn(batch, dim, device='cuda', dtype=torch.bfloat16)
        x_seq = torch.randn(batch, seq_len, dim, device='cuda', dtype=torch.bfloat16)

        # Pre-compute forward pass
        result = hasty.dual_memory_elman_forward(True, x_seq, h_tape_init, h_work_init, W_h, W_x, b_h, W_write)
        h_work_out, h_tape_final, h_tape_all, read_attn, write_attn = result

        # Create gradient output
        d_h_work_out = torch.randn_like(h_work_out)
        d_h_tape_final = torch.randn_like(h_tape_final)
        x_proj = x_seq @ W_x.T

        # Warmup
        for _ in range(3):
            hasty.dual_memory_elman_backward(
                x_proj, h_work_out, h_tape_all, read_attn, write_attn,
                W_h, W_write, d_h_work_out, d_h_tape_final)

        # Time
        torch.cuda.synchronize()
        start.record()
        for _ in range(10):
            hasty.dual_memory_elman_backward(
                x_proj, h_work_out, h_tape_all, read_attn, write_attn,
                W_h, W_write, d_h_work_out, d_h_tape_final)
        end.record()
        torch.cuda.synchronize()

        bwd_ms = start.elapsed_time(end) / 10
        print(f"batch={batch:2d}, n={n_slots:2d}: backward={bwd_ms:.2f}ms")
