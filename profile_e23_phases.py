#!/usr/bin/env python3
"""
Profile E23 backward phases separately to find n=16 anomaly.

n=8 backward: 21.7ms (2.01x forward)
n=16 backward: 16.4ms (1.52x forward) - WHY IS THIS FASTER?

Expected: n=16 should be slower than n=8 (more tape elements to process)
"""
import torch
import sys
sys.path.insert(0, 'elman/cuda')

from elman.models.dual_memory_elman import DualMemoryElman
from elman.models.mamba_gated_elman import MambaGatedElman

batch_size = 64
seq_len = 512
dim = 512

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

print("=" * 70)
print("E23 Backward Phase Profiling")
print(f"batch={batch_size}, seq={seq_len}, dim={dim}")
print("=" * 70)

def profile_fwd_bwd(layer, name, x, n_iters=10):
    """Profile forward+backward separately."""
    # Forward only
    for _ in range(3):
        x_in = x.detach().clone()
        with torch.no_grad():
            out = layer(x_in)

    torch.cuda.synchronize()
    start.record()
    for _ in range(n_iters):
        x_in = x.detach().clone()
        with torch.no_grad():
            out = layer(x_in)
    end.record()
    torch.cuda.synchronize()
    fwd_ms = start.elapsed_time(end) / n_iters

    # Forward + backward
    for _ in range(3):
        x_in = x.detach().clone().requires_grad_(True)
        out = layer(x_in)
        if isinstance(out, tuple):
            out[0].sum().backward()
        else:
            out.sum().backward()

    torch.cuda.synchronize()
    start.record()
    for _ in range(n_iters):
        x_in = x.detach().clone().requires_grad_(True)
        out = layer(x_in)
        if isinstance(out, tuple):
            out[0].sum().backward()
        else:
            out.sum().backward()
    end.record()
    torch.cuda.synchronize()
    total_ms = start.elapsed_time(end) / n_iters

    bwd_ms = total_ms - fwd_ms
    return fwd_ms, bwd_ms, total_ms

# Test different n_slots values
x = torch.randn(batch_size, seq_len, dim, device='cuda', dtype=torch.bfloat16)

print("\n1. E1 Baseline")
print("-" * 70)
e1_layer = MambaGatedElman(dim=dim).cuda().bfloat16()
e1_fwd, e1_bwd, e1_total = profile_fwd_bwd(e1_layer, "E1", x)
print(f"E1:  fwd={e1_fwd:.2f}ms, bwd={e1_bwd:.2f}ms, total={e1_total:.2f}ms")
del e1_layer
torch.cuda.empty_cache()

print("\n2. E23 with different n_slots")
print("-" * 70)

results = {}
for n_slots in [8, 16, 32, 64]:
    try:
        layer = DualMemoryElman(dim=dim, n_slots=n_slots).cuda().bfloat16()
        fwd, bwd, total = profile_fwd_bwd(layer, f"E23 n={n_slots}", x)
        results[n_slots] = (fwd, bwd, total)
        print(f"E23 n={n_slots:2d}: fwd={fwd:.2f}ms, bwd={bwd:.2f}ms, total={total:.2f}ms | fwd_ratio={fwd/e1_fwd:.2f}x, bwd_ratio={bwd/e1_bwd:.2f}x")
        del layer
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"E23 n={n_slots}: ERROR - {e}")

print("\n3. Analysis: Backward time vs n_slots")
print("-" * 70)
print("Expected: backward should scale with n_slots (more tape elements)")
print("Actual:")
for n, (fwd, bwd, total) in sorted(results.items()):
    expected_scale = n / 8
    print(f"  n={n:2d}: bwd={bwd:.2f}ms, expected_scale={expected_scale:.1f}x (vs n=8)")

print("\n4. Memory Analysis")
print("-" * 70)
for n_slots in [8, 16, 32, 64]:
    try:
        torch.cuda.reset_peak_memory_stats()
        layer = DualMemoryElman(dim=dim, n_slots=n_slots).cuda().bfloat16()
        x_in = x.detach().clone().requires_grad_(True)
        out = layer(x_in)
        if isinstance(out, tuple):
            out[0].sum().backward()
        else:
            out.sum().backward()
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"E23 n={n_slots:2d}: peak memory = {peak_mem:.2f} GB")
        del layer, x_in, out
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"E23 n={n_slots}: ERROR - {e}")

print("\n5. Check if backward is actually computing gradients correctly")
print("-" * 70)
for n_slots in [8, 16]:
    layer = DualMemoryElman(dim=dim, n_slots=n_slots).cuda().bfloat16()
    x_in = torch.randn(4, 32, dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)

    # Run forward + backward
    out = layer(x_in)
    loss = out[0].sum() if isinstance(out, tuple) else out.sum()
    loss.backward()

    # Check gradient norms
    dx_norm = x_in.grad.float().norm().item()
    param_grad_norms = {}
    for name, p in layer.named_parameters():
        if p.grad is not None:
            param_grad_norms[name] = p.grad.float().norm().item()

    print(f"\nE23 n={n_slots}:")
    print(f"  dx_norm = {dx_norm:.4f}")
    for name, norm in param_grad_norms.items():
        print(f"  d{name}_norm = {norm:.4f}")

    del layer, x_in
    torch.cuda.empty_cache()

print("\n6. Numerical Gradient Check")
print("-" * 70)
for n_slots in [8, 16]:
    layer = DualMemoryElman(dim=dim, n_slots=n_slots).cuda().float()
    x_in = torch.randn(2, 16, dim, device='cuda', dtype=torch.float32, requires_grad=True)

    def fn(x):
        out = layer(x)
        return out[0].sum() if isinstance(out, tuple) else out.sum()

    # Compute analytical gradient
    x_in.grad = None
    loss = fn(x_in)
    loss.backward()
    analytical_grad = x_in.grad.clone()

    # Compute numerical gradient
    eps = 1e-4
    numerical_grad = torch.zeros_like(x_in)
    x_flat = x_in.detach().view(-1)
    for i in range(min(100, x_flat.numel())):  # Check first 100 elements
        x_plus = x_flat.clone()
        x_plus[i] += eps
        x_minus = x_flat.clone()
        x_minus[i] -= eps

        with torch.no_grad():
            out_plus = layer(x_plus.view_as(x_in))
            out_minus = layer(x_minus.view_as(x_in))
            loss_plus = out_plus[0].sum() if isinstance(out_plus, tuple) else out_plus.sum()
            loss_minus = out_minus[0].sum() if isinstance(out_minus, tuple) else out_minus.sum()

        numerical_grad.view(-1)[i] = (loss_plus - loss_minus) / (2 * eps)

    # Compare
    analytical_flat = analytical_grad.view(-1)[:100]
    numerical_flat = numerical_grad.view(-1)[:100]

    rel_error = ((analytical_flat - numerical_flat).abs() / (numerical_flat.abs() + 1e-8)).mean()
    max_error = ((analytical_flat - numerical_flat).abs() / (numerical_flat.abs() + 1e-8)).max()

    print(f"E23 n={n_slots}: mean_rel_error={rel_error:.6f}, max_rel_error={max_error:.6f}")

    del layer, x_in
    torch.cuda.empty_cache()
