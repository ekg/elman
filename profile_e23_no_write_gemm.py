#!/usr/bin/env python3
"""Profile E23 variant without W_write GEMM - write h_work_new directly to tape."""
import torch
import sys
sys.path.insert(0, 'elman/cuda')

batch_size = 4
seq_len = 512
dim = 512
n_slots = 8

# Simulate E23 without W_write GEMM:
# Per timestep: 1 GEMM (W_h @ h_work) + attention ops

x = torch.randn(batch_size, seq_len, dim, device='cuda', dtype=torch.bfloat16)
W_h = torch.randn(dim, dim, device='cuda', dtype=torch.bfloat16)
W_x = torch.randn(dim, dim, device='cuda', dtype=torch.bfloat16)
b_h = torch.randn(dim, device='cuda', dtype=torch.bfloat16)
h_tape = torch.randn(batch_size, n_slots, dim, device='cuda', dtype=torch.bfloat16)
h_work = torch.randn(batch_size, dim, device='cuda', dtype=torch.bfloat16)

# Pre-compute x_proj
x_t = x.permute(1, 0, 2).contiguous()  # [T, B, D]
x_proj = x_t.view(-1, dim) @ W_x.T  # [T*B, D]
x_proj = x_proj.view(seq_len, batch_size, dim)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

def e23_no_write_gemm_forward(x_proj, h_tape, h_work, W_h, b_h):
    """E23 forward without W_write GEMM - write h_work_new directly."""
    T, B, D = x_proj.shape
    N = h_tape.shape[1]
    scale = 1.0 / (D ** 0.5)

    h_work_out = torch.empty(T, B, D, device=x_proj.device, dtype=x_proj.dtype)
    h_tape_current = h_tape.clone()

    for t in range(T):
        # 1. Read attention: scores = h_tape @ h_work
        scores = torch.einsum('bnd,bd->bn', h_tape_current, h_work) * scale
        read_attn = torch.softmax(scores, dim=1)  # [B, N]

        # 2. Read value
        read_val = torch.einsum('bn,bnd->bd', read_attn, h_tape_current)  # [B, D]

        # 3. Update h_work (1 GEMM: W_h @ h_work)
        pre_act = h_work @ W_h.T + x_proj[t] + read_val + b_h
        h_work_new = torch.tanh(pre_act)
        h_work_out[t] = h_work_new

        # 4. Write attention
        write_scores = torch.einsum('bnd,bd->bn', h_tape_current, h_work_new) * scale
        write_attn = torch.softmax(write_scores, dim=1)  # [B, N]

        # 5. Write to tape - NO W_write GEMM, use h_work_new directly!
        write_val = h_work_new  # Instead of: write_val = h_work_new @ W_write.T
        h_tape_current = (1 - write_attn.unsqueeze(-1)) * h_tape_current + write_attn.unsqueeze(-1) * write_val.unsqueeze(1)

        h_work = h_work_new

    return h_work_out, h_tape_current

# Warmup
for _ in range(5):
    r = e23_no_write_gemm_forward(x_proj, h_tape, h_work, W_h, b_h)

torch.cuda.synchronize()
start.record()
for _ in range(10):
    r = e23_no_write_gemm_forward(x_proj, h_tape, h_work, W_h, b_h)
end.record()
torch.cuda.synchronize()

no_write_ms = start.elapsed_time(end) / 10
print(f"E23 (no W_write GEMM): {no_write_ms:.2f}ms, {no_write_ms/seq_len*1000:.1f}us/step")

# Compare with E1
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

e1_ms = start.elapsed_time(end) / 10
print(f"E1: {e1_ms:.2f}ms, {e1_ms/seq_len*1000:.1f}us/step")
print(f"E23 (no write) / E1: {no_write_ms/e1_ms:.2f}x")

# Original E23
import hasty_pytorch_lib as hasty
W_write = torch.randn(dim, dim, device='cuda', dtype=torch.bfloat16)

for _ in range(5):
    result = hasty.dual_memory_elman_forward(True, x, h_tape, h_work, W_h, W_x, b_h, W_write)

torch.cuda.synchronize()
start.record()
for _ in range(10):
    result = hasty.dual_memory_elman_forward(True, x, h_tape, h_work, W_h, W_x, b_h, W_write)
end.record()
torch.cuda.synchronize()

e23_ms = start.elapsed_time(end) / 10
print(f"E23 (CUDA): {e23_ms:.2f}ms, {e23_ms/seq_len*1000:.1f}us/step")
print()
print(f"Speedup if we remove W_write GEMM from CUDA kernel: {e23_ms/no_write_ms:.2f}x")
