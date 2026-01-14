#!/usr/bin/env python3
"""
Verify E23 CUDA gradients match Python reference for both n=8 and n=16.
"""
import torch
import sys
sys.path.insert(0, 'elman/cuda')

from elman.models.dual_memory_elman import DualMemoryElmanCell

batch_size = 2
seq_len = 16
dim = 256

print("=" * 70)
print("E23 Gradient Verification: CUDA vs Python")
print(f"batch={batch_size}, seq={seq_len}, dim={dim}")
print("=" * 70)

torch.manual_seed(42)

for n_slots in [8, 16]:
    print(f"\nn_slots={n_slots}")
    print("-" * 50)

    # Create cell - Python reference needs float32, CUDA needs bfloat16
    cell = DualMemoryElmanCell(dim=dim, n_slots=n_slots).cuda().bfloat16()

    # Create input - use bfloat16 for CUDA compatibility
    x = torch.randn(batch_size, seq_len, dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    h_tape = torch.randn(batch_size, n_slots, dim, device='cuda', dtype=torch.bfloat16)
    h_work = torch.randn(batch_size, dim, device='cuda', dtype=torch.bfloat16)

    # Run with CUDA
    x_cuda = x.detach().clone().requires_grad_(True)
    cell.zero_grad()
    h_work_cuda, h_tape_cuda, _ = cell(x_cuda, h_tape.clone(), h_work.clone(), use_cuda=True, use_triton=False)
    loss_cuda = h_work_cuda.sum()
    loss_cuda.backward()

    dx_cuda = x_cuda.grad.clone()
    dW_h_cuda = cell.W_h.grad.clone()
    dW_x_cuda = cell.W_x.grad.clone()
    db_h_cuda = cell.b_h.grad.clone()
    dW_write_cuda = cell.W_write.grad.clone()

    # Run with Python
    cell.zero_grad()
    x_py = x.detach().clone().requires_grad_(True)
    h_work_py, h_tape_py, _ = cell(x_py, h_tape.clone(), h_work.clone(), use_cuda=False, use_triton=False)
    loss_py = h_work_py.sum()
    loss_py.backward()

    dx_py = x_py.grad.clone()
    dW_h_py = cell.W_h.grad.clone()
    dW_x_py = cell.W_x.grad.clone()
    db_h_py = cell.b_h.grad.clone()
    dW_write_py = cell.W_write.grad.clone()

    # Compare outputs (use looser tolerance for bfloat16)
    atol, rtol = 1e-2, 5e-2  # bfloat16 has ~3 decimal digits of precision
    print(f"  Forward output match: {torch.allclose(h_work_cuda.float(), h_work_py.float(), atol=atol, rtol=rtol)}")
    print(f"    Max diff: {(h_work_cuda.float() - h_work_py.float()).abs().max().item():.6f}")

    # Compare gradients
    print(f"  dx match: {torch.allclose(dx_cuda.float(), dx_py.float(), atol=atol, rtol=rtol)}")
    print(f"    Max diff: {(dx_cuda.float() - dx_py.float()).abs().max().item():.6f}")
    print(f"    dx_cuda norm: {dx_cuda.float().norm().item():.4f}, dx_py norm: {dx_py.float().norm().item():.4f}")

    print(f"  dW_h match: {torch.allclose(dW_h_cuda.float(), dW_h_py.float(), atol=atol, rtol=rtol)}")
    print(f"    Max diff: {(dW_h_cuda.float() - dW_h_py.float()).abs().max().item():.6f}")

    print(f"  dW_x match: {torch.allclose(dW_x_cuda.float(), dW_x_py.float(), atol=atol, rtol=rtol)}")
    print(f"    Max diff: {(dW_x_cuda.float() - dW_x_py.float()).abs().max().item():.6f}")

    print(f"  db_h match: {torch.allclose(db_h_cuda.float(), db_h_py.float(), atol=atol, rtol=rtol)}")
    print(f"    Max diff: {(db_h_cuda.float() - db_h_py.float()).abs().max().item():.6f}")

    print(f"  dW_write match: {torch.allclose(dW_write_cuda.float(), dW_write_py.float(), atol=atol, rtol=rtol)}")
    print(f"    Max diff: {(dW_write_cuda.float() - dW_write_py.float()).abs().max().item():.6f}")

print("\n" + "=" * 70)
print("Gradient verification complete")
print("=" * 70)
