#!/usr/bin/env python3
"""
Debug why n=16 backward is faster than n=8.

Hypothesis: Some tensor is not being saved correctly for n=16 backward.
"""
import torch
import sys
sys.path.insert(0, 'elman/cuda')

import hasty_pytorch_lib as hasty
from elman.models.dual_memory_elman import DualMemoryElman

batch_size = 4
seq_len = 32
dim = 512

print("=" * 70)
print("E23 n=16 Debug")
print(f"batch={batch_size}, seq={seq_len}, dim={dim}")
print("=" * 70)

# Test with the raw CUDA function
print("\n1. Raw CUDA function test")
print("-" * 70)

for n_slots in [8, 16]:
    print(f"\nn_slots={n_slots}:")

    # Create inputs
    W_h = torch.randn(dim, dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    W_x = torch.randn(dim, dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    b_h = torch.randn(dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    W_write = torch.randn(dim, dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    h_tape = torch.randn(batch_size, n_slots, dim, device='cuda', dtype=torch.bfloat16)
    h_work = torch.randn(batch_size, dim, device='cuda', dtype=torch.bfloat16)
    x_seq = torch.randn(batch_size, seq_len, dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)

    # Forward
    result = hasty.dual_memory_elman_forward(True, x_seq, h_tape, h_work, W_h, W_x, b_h, W_write)
    h_work_out, h_tape_final, h_tape_all, read_attn, write_attn = result

    print(f"  h_work_out shape: {h_work_out.shape}")
    print(f"  h_tape_final shape: {h_tape_final.shape}")
    print(f"  h_tape_all shape: {h_tape_all.shape}")
    print(f"  read_attn shape: {read_attn.shape}")
    print(f"  write_attn shape: {write_attn.shape}")

    # Check h_tape_all content - is it all zeros or properly filled?
    print(f"  h_tape_all[0] norm: {h_tape_all[0].float().norm().item():.4f}")
    print(f"  h_tape_all[T//2] norm: {h_tape_all[seq_len//2].float().norm().item():.4f}")
    print(f"  h_tape_all[T] norm: {h_tape_all[seq_len].float().norm().item():.4f}")

    # Backward
    loss = h_work_out.sum()
    loss.backward()

    print(f"  dx norm: {x_seq.grad.float().norm().item():.4f}")
    print(f"  dW_h norm: {W_h.grad.float().norm().item():.4f}")
    print(f"  dW_write norm: {W_write.grad.float().norm().item():.4f}")

print("\n2. Layer test with detailed backward timing")
print("-" * 70)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

for n_slots in [8, 16, 32]:
    layer = DualMemoryElman(dim=dim, n_slots=n_slots).cuda().bfloat16()
    x = torch.randn(batch_size, seq_len, dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)

    # Warmup
    for _ in range(3):
        x_in = x.detach().clone().requires_grad_(True)
        out = layer(x_in)
        out[0].sum().backward()

    # Profile forward
    torch.cuda.synchronize()
    start.record()
    for _ in range(20):
        x_in = x.detach().clone()
        with torch.no_grad():
            out = layer(x_in)
    end.record()
    torch.cuda.synchronize()
    fwd_ms = start.elapsed_time(end) / 20

    # Profile backward only (using pre-computed forward)
    # First do forward with grad enabled
    x_in = x.detach().clone().requires_grad_(True)
    out = layer(x_in)

    # Time backward
    torch.cuda.synchronize()
    start.record()
    for _ in range(20):
        if x_in.grad is not None:
            x_in.grad.zero_()
        for p in layer.parameters():
            if p.grad is not None:
                p.grad.zero_()
        out[0].sum().backward(retain_graph=True)
    end.record()
    torch.cuda.synchronize()
    bwd_ms = start.elapsed_time(end) / 20

    print(f"n={n_slots:2d}: fwd={fwd_ms:.2f}ms, bwd_only={bwd_ms:.2f}ms")

    del layer, x, x_in, out
    torch.cuda.empty_cache()

print("\n3. Check autograd graph structure")
print("-" * 70)

for n_slots in [8, 16]:
    layer = DualMemoryElman(dim=dim, n_slots=n_slots).cuda().bfloat16()
    x = torch.randn(2, 8, dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)

    out = layer(x)
    h_work_out = out[0]

    # Walk the grad_fn chain
    print(f"\nn_slots={n_slots}:")
    print(f"  output requires_grad: {h_work_out.requires_grad}")
    print(f"  output grad_fn: {h_work_out.grad_fn}")

    # Count saved tensors
    def count_saved_tensors(grad_fn, depth=0, visited=None):
        if visited is None:
            visited = set()
        if grad_fn is None or id(grad_fn) in visited:
            return 0
        visited.add(id(grad_fn))

        count = 0
        if hasattr(grad_fn, 'saved_tensors'):
            count += len(grad_fn.saved_tensors)

        if hasattr(grad_fn, 'next_functions'):
            for fn, _ in grad_fn.next_functions:
                count += count_saved_tensors(fn, depth+1, visited)

        return count

    saved_count = count_saved_tensors(h_work_out.grad_fn)
    print(f"  saved tensors in graph: {saved_count}")

    del layer, x, out
    torch.cuda.empty_cache()

print("\n4. Verify gradient correctness with finite differences")
print("-" * 70)

for n_slots in [8, 16]:
    layer = DualMemoryElman(dim=dim, n_slots=n_slots).cuda().bfloat16()
    x = torch.randn(2, 8, dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)

    # Compute analytical gradient
    out = layer(x)
    loss = out[0].sum()
    loss.backward()
    analytical_grad = x.grad.clone()

    # Compute numerical gradient with larger epsilon for bfloat16
    eps = 1e-2  # Larger epsilon for bfloat16
    x_flat = x.detach().view(-1)
    numerical_grad = torch.zeros_like(x)

    errors = []
    for i in range(min(50, x_flat.numel())):
        x_plus = x_flat.clone()
        x_plus[i] += eps
        x_minus = x_flat.clone()
        x_minus[i] -= eps

        with torch.no_grad():
            out_plus = layer(x_plus.view_as(x))
            out_minus = layer(x_minus.view_as(x))
            loss_plus = out_plus[0].sum()
            loss_minus = out_minus[0].sum()

        numerical = (loss_plus - loss_minus).float() / (2 * eps)
        analytical = analytical_grad.view(-1)[i].float()

        rel_error = abs(analytical - numerical) / (abs(numerical) + 1e-6)
        errors.append(rel_error)

    mean_error = sum(errors) / len(errors)
    max_error = max(errors)
    print(f"n={n_slots}: mean_rel_error={mean_error:.4f}, max_rel_error={max_error:.4f}")

    del layer, x
    torch.cuda.empty_cache()
