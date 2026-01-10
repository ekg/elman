#!/usr/bin/env python3
import torch
import sys
sys.path.insert(0, 'elman/cuda')
import hasty_pytorch_lib as hasty

batch_size = 4
seq_len = 512
dim = 512
n_slots = 8

x = torch.randn(batch_size, seq_len, dim, device='cuda', dtype=torch.bfloat16)
W_h = torch.randn(dim, dim, device='cuda', dtype=torch.bfloat16)
b_h = torch.randn(dim, device='cuda', dtype=torch.bfloat16)
W_x = torch.randn(dim, dim, device='cuda', dtype=torch.bfloat16)
W_write = torch.randn(dim, dim, device='cuda', dtype=torch.bfloat16)
h_tape_init = torch.randn(batch_size, n_slots, dim, device='cuda', dtype=torch.bfloat16)
h_work_init = torch.randn(batch_size, dim, device='cuda', dtype=torch.bfloat16)

for _ in range(5):
    result = hasty.dual_memory_elman_forward(True, x, h_tape_init, h_work_init, W_h, W_x, b_h, W_write)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

torch.cuda.synchronize()
start.record()
for _ in range(10):
    result = hasty.dual_memory_elman_forward(True, x, h_tape_init, h_work_init, W_h, W_x, b_h, W_write)
end.record()
torch.cuda.synchronize()

total_ms = start.elapsed_time(end)
print(f'E23 forward: {total_ms/10:.2f}ms/iter, {total_ms/10/seq_len*1000:.1f}us/timestep')

from elman.models.mamba_gated_elman import MambaGatedElman
e1 = MambaGatedElman(dim=512).cuda().bfloat16()
x_e1 = torch.randn(batch_size, seq_len, dim, device='cuda', dtype=torch.bfloat16)

for _ in range(5):
    with torch.no_grad():
        y = e1(x_e1)

torch.cuda.synchronize()
start.record()
for _ in range(10):
    with torch.no_grad():
        y = e1(x_e1)
end.record()
torch.cuda.synchronize()

e1_ms = start.elapsed_time(end)
print(f'E1 forward: {e1_ms/10:.2f}ms/iter, {e1_ms/10/seq_len*1000:.1f}us/timestep')
print(f'E23/E1: {total_ms/e1_ms:.2f}x')
print()

# Now profile individual cuBLAS operations
print('=== Individual operation timing ===')

# Just the x_proj GEMM
x_t = x.permute(1, 0, 2).contiguous()  # [T, B, D]
x_proj = torch.empty(seq_len, batch_size, dim, device='cuda', dtype=torch.bfloat16)

torch.cuda.synchronize()
start.record()
for _ in range(100):
    torch.mm(x_t.view(-1, dim), W_x.T, out=x_proj.view(-1, dim))
end.record()
torch.cuda.synchronize()
gemm_ms = start.elapsed_time(end)
print(f'x_proj GEMM ({seq_len*batch_size}x{dim} @ {dim}x{dim}): {gemm_ms/100:.3f}ms')

# Single small GEMM (W_h @ h)
h = torch.randn(batch_size, dim, device='cuda', dtype=torch.bfloat16)
out = torch.empty(batch_size, dim, device='cuda', dtype=torch.bfloat16)

torch.cuda.synchronize()
start.record()
for _ in range(1000):
    torch.mm(h, W_h.T, out=out)
end.record()
torch.cuda.synchronize()
small_gemm_ms = start.elapsed_time(end)
print(f'W_h @ h GEMM ({batch_size}x{dim} @ {dim}x{dim}): {small_gemm_ms/1000*1000:.1f}us')

# Expected per-timestep time from GEMMs alone
gemm_per_timestep = small_gemm_ms/1000 * 2  # W_h and W_write
print(f'Expected GEMM time per timestep (W_h + W_write): {gemm_per_timestep*1000:.1f}us')
print(f'Actual E23 time per timestep: {total_ms/10/seq_len*1000:.1f}us')
print(f'Overhead (kernels + memcpy): {total_ms/10/seq_len*1000 - gemm_per_timestep*1000:.1f}us/timestep')
