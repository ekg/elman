#!/usr/bin/env python3
"""
Verify E23 Phase3 backward is working correctly by comparing
the gradient accumulation behavior between n=8 and n=16.
"""
import torch
import sys
sys.path.insert(0, 'elman/cuda')

from elman.models.dual_memory_elman import DualMemoryElmanCell

batch_size = 2
seq_len = 4  # Very short sequence to isolate
dim = 256

print("=" * 70)
print("E23 Phase3 Verification")
print(f"batch={batch_size}, seq={seq_len}, dim={dim}")
print("=" * 70)

torch.manual_seed(42)

for n_slots in [8, 16]:
    print(f"\nn_slots={n_slots}")
    print("-" * 50)

    # Create cell
    cell = DualMemoryElmanCell(dim=dim, n_slots=n_slots).cuda().bfloat16()

    # Create input
    x = torch.randn(batch_size, seq_len, dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    h_tape = torch.zeros(batch_size, n_slots, dim, device='cuda', dtype=torch.bfloat16)
    h_work = torch.zeros(batch_size, dim, device='cuda', dtype=torch.bfloat16)

    # Run with CUDA
    x_cuda = x.detach().clone().requires_grad_(True)
    cell.zero_grad()
    h_work_cuda, h_tape_cuda, _ = cell(x_cuda, h_tape.clone(), h_work.clone(), use_cuda=True, use_triton=False)

    # Use uniform gradient to make analysis easier
    d_out = torch.ones_like(h_work_cuda)
    h_work_cuda.backward(d_out)

    dx_cuda = x_cuda.grad.clone()

    # Run with Python
    cell.zero_grad()
    x_py = x.detach().clone().requires_grad_(True)
    h_work_py, h_tape_py, _ = cell(x_py, h_tape.clone(), h_work.clone(), use_cuda=False, use_triton=False)
    h_work_py.backward(d_out)

    dx_py = x_py.grad.clone()

    # Compare per-timestep gradients
    print("Per-timestep gradient comparison (dx):")
    for t in range(seq_len):
        cuda_norm = dx_cuda[:, t].float().norm().item()
        py_norm = dx_py[:, t].float().norm().item()
        diff = (dx_cuda[:, t].float() - dx_py[:, t].float()).norm().item()
        ratio = cuda_norm / (py_norm + 1e-8)
        print(f"  t={t}: cuda_norm={cuda_norm:.2f}, py_norm={py_norm:.2f}, diff={diff:.2f}, ratio={ratio:.2f}")

    # Compare W_h gradients
    dW_h_cuda = cell.W_h.grad.clone() if cell.W_h.grad is not None else torch.zeros_like(cell.W_h)

    cell.zero_grad()
    x_py2 = x.detach().clone().requires_grad_(True)
    h_work_py2, h_tape_py2, _ = cell(x_py2, h_tape.clone(), h_work.clone(), use_cuda=False, use_triton=False)
    h_work_py2.backward(d_out)
    dW_h_py = cell.W_h.grad.clone()

    print(f"\ndW_h comparison:")
    print(f"  CUDA norm: {dW_h_cuda.float().norm().item():.2f}")
    print(f"  Python norm: {dW_h_py.float().norm().item():.2f}")
    print(f"  Max diff: {(dW_h_cuda.float() - dW_h_py.float()).abs().max().item():.4f}")

print("\n" + "=" * 70)
print("Done")
print("=" * 70)
