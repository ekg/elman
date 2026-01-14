"""Compare forward pass details between Python and CUDA."""

import torch
import math

torch.manual_seed(42)

B = 1
T = 1
D = 256
N = 8

device = 'cuda'
dtype = torch.bfloat16

from elman.models.e25_entmax import E25DualMemoryElmanCell, e25_forward_step_python

cell = E25DualMemoryElmanCell(dim=D, n_slots=N).to(device).to(dtype)
x = torch.randn(B, T, D, device=device, dtype=dtype)

h_tape_init = torch.zeros(B, N, D, device=device, dtype=dtype)
h_work_init = torch.zeros(B, D, device=device, dtype=dtype)

scale = 1.0 / math.sqrt(D)

print("="*60)
print("Forward pass comparison")
print("="*60)

# Python forward
h_work_py, h_tape_py, read_attn_py, write_attn_py, read_py = e25_forward_step_python(
    x[:, 0].float(),
    h_tape_init.float(),
    h_work_init.float(),
    cell.W_h.float(), cell.W_x.float(), cell.b_h.float(), cell.W_write.float(),
    scale
)

print("\nPython forward:")
print(f"  h_work sample: {h_work_py[0, :4].tolist()}")
print(f"  read_attn: {read_attn_py[0, :4].tolist()}")
print(f"  write_attn: {write_attn_py[0, :4].tolist()}")
print(f"  read sample: {read_py[0, :4].tolist()}")

# CUDA forward
import hasty_pytorch_lib

h_work_cuda, h_tape_final_cuda, h_tape_all_cuda, read_attn_cuda, write_attn_cuda = \
    hasty_pytorch_lib.e25_entmax_forward(
        True,  # training
        x.contiguous(),
        h_tape_init.contiguous(),
        h_work_init.contiguous(),
        cell.W_h.contiguous(),
        cell.W_x.contiguous(),
        cell.b_h.contiguous(),
        cell.W_write.contiguous()
    )

print("\nCUDA forward:")
print(f"  h_work sample: {h_work_cuda[0, 0, :4].float().tolist()}")
print(f"  read_attn: {read_attn_cuda[0, 0, :4].float().tolist()}")
print(f"  write_attn: {write_attn_cuda[0, 0, :4].float().tolist()}")

# The CUDA kernel saves h_tape_all in shape [T+1, B, N, D]
# h_tape_all[0] is the initial tape, h_tape_all[1] is after first write
print(f"  h_tape_all shape: {h_tape_all_cuda.shape}")

print("\nDifferences:")
print(f"  h_work diff: {(h_work_py.to(dtype) - h_work_cuda[0, 0]).abs().max().item():.6f}")
print(f"  read_attn diff: {(read_attn_py.to(dtype) - read_attn_cuda[0, 0]).abs().max().item():.6f}")
print(f"  write_attn diff: {(write_attn_py.to(dtype) - write_attn_cuda[0, 0]).abs().max().item():.6f}")

# Check h_tape difference
h_tape_py_bf16 = h_tape_py.to(dtype)
h_tape_cuda = h_tape_all_cuda[1, 0]  # After first write
print(f"  h_tape diff: {(h_tape_py_bf16 - h_tape_cuda).abs().max().item():.6f}")
