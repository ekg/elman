#!/usr/bin/env python3
"""Detailed E23 profiling to identify optimization opportunities."""
import torch
import sys
sys.path.insert(0, 'elman/cuda')
import hasty_pytorch_lib as hasty

batch_size = 4
seq_len = 512
dim = 512
n_slots = 8

# Create inputs
x = torch.randn(batch_size, seq_len, dim, device='cuda', dtype=torch.bfloat16)
W_h = torch.randn(dim, dim, device='cuda', dtype=torch.bfloat16)
b_h = torch.randn(dim, device='cuda', dtype=torch.bfloat16)
W_x = torch.randn(dim, dim, device='cuda', dtype=torch.bfloat16)
W_write = torch.randn(dim, dim, device='cuda', dtype=torch.bfloat16)
h_tape_init = torch.randn(batch_size, n_slots, dim, device='cuda', dtype=torch.bfloat16)
h_work_init = torch.randn(batch_size, dim, device='cuda', dtype=torch.bfloat16)

print("=" * 60)
print("E23 Detailed Profiling")
print("=" * 60)
print(f"Config: batch={batch_size}, seq_len={seq_len}, dim={dim}, n_slots={n_slots}")
print()

# Use nsight compute profiling style: measure individual operations
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# 1. Profile the full E23 forward
torch.cuda.synchronize()
for _ in range(5):
    result = hasty.dual_memory_elman_forward(True, x, h_tape_init, h_work_init, W_h, W_x, b_h, W_write)

torch.cuda.synchronize()
start.record()
for _ in range(10):
    result = hasty.dual_memory_elman_forward(True, x, h_tape_init, h_work_init, W_h, W_x, b_h, W_write)
end.record()
torch.cuda.synchronize()
e23_ms = start.elapsed_time(end) / 10
print(f"E23 total forward: {e23_ms:.2f}ms ({e23_ms/seq_len*1000:.1f}us/step)")

# 2. Profile just the pre-compute x_proj GEMM (happens once)
x_t = x.permute(1, 0, 2).contiguous().view(-1, dim)  # [T*B, D]
x_proj = torch.empty(seq_len * batch_size, dim, device='cuda', dtype=torch.bfloat16)

torch.cuda.synchronize()
start.record()
for _ in range(100):
    torch.mm(x_t, W_x.T, out=x_proj)
end.record()
torch.cuda.synchronize()
x_proj_ms = start.elapsed_time(end) / 100
print(f"x_proj GEMM (one-time): {x_proj_ms:.3f}ms")

# 3. Profile a single W_h @ h GEMM
h = torch.randn(batch_size, dim, device='cuda', dtype=torch.bfloat16)
tmp_Rh = torch.empty(batch_size, dim, device='cuda', dtype=torch.bfloat16)

torch.cuda.synchronize()
start.record()
for _ in range(1000):
    torch.mm(h, W_h.T, out=tmp_Rh)
end.record()
torch.cuda.synchronize()
wh_gemm_us = start.elapsed_time(end)
print(f"W_h @ h GEMM: {wh_gemm_us:.1f}us (per call)")

# 4. Profile theoretical minimum: 2 GEMMs per step
# Per step: W_h @ h + W_write @ h_new (they have data dependency so can't parallel)
# Simulating by running 2 sequential GEMMs
torch.cuda.synchronize()
start.record()
for _ in range(1000):
    torch.mm(h, W_h.T, out=tmp_Rh)
    torch.mm(h, W_write.T, out=tmp_Rh)  # Uses same output buffer
end.record()
torch.cuda.synchronize()
two_gemm_us = start.elapsed_time(end)
print(f"Two sequential GEMMs: {two_gemm_us:.1f}us (per pair)")

print()
print("=" * 60)
print("Analysis")
print("=" * 60)
print(f"E23 per-step time: {e23_ms/seq_len*1000:.1f}us")
print(f"Theoretical 2-GEMM time: {two_gemm_us/1000*1000:.1f}us")
print(f"Overhead (attention + memcpy): {e23_ms/seq_len*1000 - two_gemm_us/1000:.1f}us")
print()

# 5. Profile with different batch sizes to understand scaling
print("=" * 60)
print("Batch size scaling")
print("=" * 60)

for bs in [1, 2, 4, 8, 16, 32]:
    x_bs = torch.randn(bs, seq_len, dim, device='cuda', dtype=torch.bfloat16)
    h_tape_bs = torch.randn(bs, n_slots, dim, device='cuda', dtype=torch.bfloat16)
    h_work_bs = torch.randn(bs, dim, device='cuda', dtype=torch.bfloat16)

    # Warmup
    for _ in range(3):
        r = hasty.dual_memory_elman_forward(True, x_bs, h_tape_bs, h_work_bs, W_h, W_x, b_h, W_write)

    torch.cuda.synchronize()
    start.record()
    for _ in range(10):
        r = hasty.dual_memory_elman_forward(True, x_bs, h_tape_bs, h_work_bs, W_h, W_x, b_h, W_write)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / 10
    toks_per_sec = bs * seq_len / ms * 1000
    print(f"batch={bs:3d}: {ms:7.2f}ms, {ms/seq_len*1000:.1f}us/step, {toks_per_sec/1000:.1f}K tok/s")

# 6. Profile with larger batch to see if we're compute or memory bound
print()
print("=" * 60)
print("Compute vs Memory Bound Analysis")
print("=" * 60)

# Small batch: likely kernel launch bound
# Large batch: should approach compute bound
for bs in [4, 64]:
    h_bs = torch.randn(bs, dim, device='cuda', dtype=torch.bfloat16)
    tmp = torch.empty(bs, dim, device='cuda', dtype=torch.bfloat16)

    torch.cuda.synchronize()
    start.record()
    for _ in range(1000):
        torch.mm(h_bs, W_h.T, out=tmp)
    end.record()
    torch.cuda.synchronize()

    gemm_us = start.elapsed_time(end)
    flops = 2 * bs * dim * dim * 1000  # 2*M*N*K for each of 1000 GEMMs
    tflops = flops / (gemm_us / 1000) / 1e12
    print(f"batch={bs:3d}: W_h GEMM = {gemm_us:.1f}us, {tflops:.1f} TFLOP/s")

# Peak A100: ~312 TFLOP/s for TF32, ~624 for BF16
# We're probably getting ~50-100 TFLOP/s on small batches due to memory bandwidth
print()

# 7. Kernel launch overhead measurement
print("=" * 60)
print("Kernel Launch Overhead")
print("=" * 60)

# Measure empty kernel launches
# Use a simple add as a minimal kernel
a = torch.randn(100, device='cuda', dtype=torch.bfloat16)
b = torch.randn(100, device='cuda', dtype=torch.bfloat16)

torch.cuda.synchronize()
start.record()
for _ in range(10000):
    c = a + b  # Simple elementwise triggers a small kernel
end.record()
torch.cuda.synchronize()
launch_us = start.elapsed_time(end) / 10000 * 1000
print(f"Minimal kernel launch: {launch_us:.2f}us")

# E23 does per timestep: 1 GEMM + 1 Phase1 + 1 GEMM + 1 Phase2 + 1 memcpy = 5 operations
print(f"E23 has ~5 operations per step")
print(f"Estimated launch overhead per step: {launch_us * 5:.1f}us")
print(f"Actual overhead: {e23_ms/seq_len*1000 - two_gemm_us/1000:.1f}us")
