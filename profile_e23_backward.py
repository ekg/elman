#!/usr/bin/env python3
"""
Profile E23 backward pass in detail.

Backward is 7.32x slower than E1 (vs 5.27x for forward).
Where is the extra slowdown?
"""
import torch
import sys
sys.path.insert(0, 'elman/cuda')

from elman.models import LadderLM, create_ladder_model
from elman.models.dual_memory_elman import DualMemoryElman
from elman.models.mamba_gated_elman import MambaGatedElman

batch_size = 64
seq_len = 512
dim = 512

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

print("=" * 70)
print("E23 Backward Pass Profiling")
print(f"batch={batch_size}, seq={seq_len}, dim={dim}")
print("=" * 70)

# Create single layers
print("\n1. Single Layer Forward+Backward")
print("-" * 70)

e1_layer = MambaGatedElman(dim=dim).cuda().bfloat16()
e23_layer = DualMemoryElman(dim=dim, n_slots=8).cuda().bfloat16()

x = torch.randn(batch_size, seq_len, dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)

def get_output(out):
    """Extract main output from layer (handles tuple returns)."""
    if isinstance(out, tuple):
        return out[0]
    return out

# E1 layer
for _ in range(3):
    x_e1 = x.detach().clone().requires_grad_(True)
    out = e1_layer(x_e1)
    get_output(out).sum().backward()

torch.cuda.synchronize()
start.record()
for _ in range(10):
    x_e1 = x.detach().clone().requires_grad_(True)
    out = e1_layer(x_e1)
    get_output(out).sum().backward()
end.record()
torch.cuda.synchronize()
e1_layer_ms = start.elapsed_time(end) / 10
print(f"E1 layer (fwd+bwd): {e1_layer_ms:.2f}ms")

# E23 layer
for _ in range(3):
    x_e23 = x.detach().clone().requires_grad_(True)
    out = e23_layer(x_e23)
    get_output(out).sum().backward()

torch.cuda.synchronize()
start.record()
for _ in range(10):
    x_e23 = x.detach().clone().requires_grad_(True)
    out = e23_layer(x_e23)
    get_output(out).sum().backward()
end.record()
torch.cuda.synchronize()
e23_layer_ms = start.elapsed_time(end) / 10
print(f"E23 layer (fwd+bwd): {e23_layer_ms:.2f}ms")
print(f"Ratio: {e23_layer_ms/e1_layer_ms:.2f}x")

# Separate forward and backward
print("\n2. Separate Forward and Backward")
print("-" * 70)

# E1 forward only
for _ in range(3):
    x_e1 = x.detach().clone()
    with torch.no_grad():
        out = e1_layer(x_e1)

torch.cuda.synchronize()
start.record()
for _ in range(10):
    x_e1 = x.detach().clone()
    with torch.no_grad():
        out = e1_layer(x_e1)
end.record()
torch.cuda.synchronize()
e1_fwd_ms = start.elapsed_time(end) / 10
print(f"E1 forward only: {e1_fwd_ms:.2f}ms")

# E23 forward only
for _ in range(3):
    x_e23 = x.detach().clone()
    with torch.no_grad():
        out = e23_layer(x_e23)

torch.cuda.synchronize()
start.record()
for _ in range(10):
    x_e23 = x.detach().clone()
    with torch.no_grad():
        out = e23_layer(x_e23)
end.record()
torch.cuda.synchronize()
e23_fwd_ms = start.elapsed_time(end) / 10
print(f"E23 forward only: {e23_fwd_ms:.2f}ms")
print(f"Forward ratio: {e23_fwd_ms/e1_fwd_ms:.2f}x")

e1_bwd_ms = e1_layer_ms - e1_fwd_ms
e23_bwd_ms = e23_layer_ms - e23_fwd_ms
print(f"\nE1 backward (derived): {e1_bwd_ms:.2f}ms")
print(f"E23 backward (derived): {e23_bwd_ms:.2f}ms")
print(f"Backward ratio: {e23_bwd_ms/e1_bwd_ms:.2f}x")

print("\n3. Memory Usage Per Layer")
print("-" * 70)

torch.cuda.reset_peak_memory_stats()
x_e1 = x.detach().clone().requires_grad_(True)
out = e1_layer(x_e1)
get_output(out).sum().backward()
e1_mem = torch.cuda.max_memory_allocated() / 1e9
print(f"E1 layer peak: {e1_mem:.2f} GB")

torch.cuda.reset_peak_memory_stats()
x_e23 = x.detach().clone().requires_grad_(True)
out = e23_layer(x_e23)
get_output(out).sum().backward()
e23_mem = torch.cuda.max_memory_allocated() / 1e9
print(f"E23 layer peak: {e23_mem:.2f} GB")
print(f"Memory ratio: {e23_mem/e1_mem:.2f}x")

print("\n4. Profile E23 Layer Components")
print("-" * 70)

# Let's profile the internal operations of E23
import hasty_pytorch_lib as hasty

# Create test inputs
W_h = torch.randn(dim, dim, device='cuda', dtype=torch.bfloat16)
W_x = torch.randn(dim, dim, device='cuda', dtype=torch.bfloat16)
b_h = torch.randn(dim, device='cuda', dtype=torch.bfloat16)
W_write = torch.randn(dim, dim, device='cuda', dtype=torch.bfloat16)
h_tape = torch.randn(batch_size, 8, dim, device='cuda', dtype=torch.bfloat16)
h_work = torch.randn(batch_size, dim, device='cuda', dtype=torch.bfloat16)
x_seq = torch.randn(batch_size, seq_len, dim, device='cuda', dtype=torch.bfloat16)

# Forward only
for _ in range(3):
    result = hasty.dual_memory_elman_forward(True, x_seq, h_tape, h_work, W_h, W_x, b_h, W_write)

torch.cuda.synchronize()
start.record()
for _ in range(10):
    result = hasty.dual_memory_elman_forward(True, x_seq, h_tape, h_work, W_h, W_x, b_h, W_write)
end.record()
torch.cuda.synchronize()
cuda_fwd_ms = start.elapsed_time(end) / 10
print(f"CUDA forward: {cuda_fwd_ms:.2f}ms")

# Forward + Backward
x_seq_grad = x_seq.detach().clone().requires_grad_(True)
h_tape_grad = h_tape.detach().clone()
h_work_grad = h_work.detach().clone()
W_h_grad = W_h.detach().clone().requires_grad_(True)
W_x_grad = W_x.detach().clone().requires_grad_(True)
b_h_grad = b_h.detach().clone().requires_grad_(True)
W_write_grad = W_write.detach().clone().requires_grad_(True)

for _ in range(3):
    x_seq_grad = x_seq.detach().clone().requires_grad_(True)
    W_h_grad = W_h.detach().clone().requires_grad_(True)
    W_x_grad = W_x.detach().clone().requires_grad_(True)
    b_h_grad = b_h.detach().clone().requires_grad_(True)
    W_write_grad = W_write.detach().clone().requires_grad_(True)
    result = hasty.dual_memory_elman_forward(True, x_seq_grad, h_tape_grad, h_work_grad,
                                              W_h_grad, W_x_grad, b_h_grad, W_write_grad)
    result[0].sum().backward()

torch.cuda.synchronize()
start.record()
for _ in range(10):
    x_seq_grad = x_seq.detach().clone().requires_grad_(True)
    W_h_grad = W_h.detach().clone().requires_grad_(True)
    W_x_grad = W_x.detach().clone().requires_grad_(True)
    b_h_grad = b_h.detach().clone().requires_grad_(True)
    W_write_grad = W_write.detach().clone().requires_grad_(True)
    result = hasty.dual_memory_elman_forward(True, x_seq_grad, h_tape_grad, h_work_grad,
                                              W_h_grad, W_x_grad, b_h_grad, W_write_grad)
    result[0].sum().backward()
end.record()
torch.cuda.synchronize()
cuda_fb_ms = start.elapsed_time(end) / 10
cuda_bwd_ms = cuda_fb_ms - cuda_fwd_ms
print(f"CUDA forward+backward: {cuda_fb_ms:.2f}ms")
print(f"CUDA backward (derived): {cuda_bwd_ms:.2f}ms")
print(f"Backward/Forward ratio: {cuda_bwd_ms/cuda_fwd_ms:.2f}x")

print("\n5. Summary")
print("-" * 70)
print(f"E23 backward is {e23_bwd_ms/e1_bwd_ms:.2f}x slower than E1 backward")
print(f"E23 backward is {cuda_bwd_ms/cuda_fwd_ms:.2f}x the forward time")
print(f"E1 backward is {e1_bwd_ms/e1_fwd_ms:.2f}x the forward time")
