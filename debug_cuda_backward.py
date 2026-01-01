#!/usr/bin/env python3
"""Debug CUDA backward by comparing gradients with PyTorch fallback."""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the model components
import sys
sys.path.insert(0, '/home/erikg/elman')

from elman.models.log_space_triple_r import (
    LogSpaceTripleRCell,
    HASTE_LOG_TRIPLE_R_AVAILABLE,
    LogSpaceTripleRFunction,
)
from elman.models.logspace_polynomial import LOG_ZERO, to_log_space, from_log_space

print(f"HASTE available: {HASTE_LOG_TRIPLE_R_AVAILABLE}")

def test_gradient_comparison():
    """Compare gradients between CUDA kernel and PyTorch fallback."""

    torch.manual_seed(42)

    # Small sizes for debugging
    T, B, D = 4, 2, 64  # 4 timesteps, batch 2, dim 64
    n_groups = 8

    device = 'cuda'
    dtype = torch.float32  # Use float32 for better numerical comparison

    # Create input
    x = torch.randn(T, B, D, device=device, dtype=dtype, requires_grad=True)

    # Create two identical cells
    cell_cuda = LogSpaceTripleRCell(
        D, n_groups=n_groups, delta_init=-2.0,
        r_h_mode='spectral_norm', spectral_radius=0.99,
        diagonal_r_delta=False
    ).to(device).to(dtype)

    cell_pytorch = LogSpaceTripleRCell(
        D, n_groups=n_groups, delta_init=-2.0,
        r_h_mode='spectral_norm', spectral_radius=0.99,
        diagonal_r_delta=False
    ).to(device).to(dtype)

    # Copy weights from cuda to pytorch cell
    cell_pytorch.load_state_dict(cell_cuda.state_dict())

    # Initial hidden state
    log_h0 = torch.full((B, D), LOG_ZERO, device=device, dtype=dtype)
    sign_h0 = torch.ones(B, D, device=device, dtype=dtype)
    h0 = (log_h0, sign_h0)

    # Get constrained matrices (same for both)
    R_h = cell_cuda.get_R_h()
    R_x = cell_cuda.get_R_x()
    R_delta = cell_cuda.get_R_delta()

    print("\n=== Forward Pass Comparison ===")

    # CUDA forward
    x_cuda = x.detach().clone().requires_grad_(True)
    output_cuda, log_h_cuda, sign_h_cuda, _ = LogSpaceTripleRFunction.apply(
        True, x_cuda, log_h0.clone(), sign_h0.clone(),
        R_h.clone(), R_x.clone(), R_delta.clone(),
        cell_cuda.W_delta.clone(), cell_cuda.W_out.clone(),
        cell_cuda.b.clone(), cell_cuda.b_delta.clone(), n_groups
    )

    # PyTorch forward
    x_pytorch = x.detach().clone().requires_grad_(True)
    output_pytorch, log_h_pytorch, sign_h_pytorch, h_linear_pytorch = cell_pytorch._forward_pytorch(
        x_pytorch, log_h0.clone(), sign_h0.clone(),
        R_h.clone(), R_x.clone(), R_delta.clone()
    )

    # Compare forward outputs
    output_diff = (output_cuda - output_pytorch).abs().max().item()
    log_h_diff = (log_h_cuda - log_h_pytorch).abs().max().item()
    sign_h_diff = (sign_h_cuda - sign_h_pytorch).abs().max().item()

    print(f"Output max diff: {output_diff:.6e}")
    print(f"log_h max diff: {log_h_diff:.6e}")
    print(f"sign_h max diff: {sign_h_diff:.6e}")

    if output_diff > 1e-4:
        print("WARNING: Forward outputs differ significantly!")
        print(f"CUDA output sample: {output_cuda[0, 0, :5]}")
        print(f"PyTorch output sample: {output_pytorch[0, 0, :5]}")

    print("\n=== Backward Pass Comparison ===")

    # Create identical upstream gradients
    grad_output = torch.randn_like(output_cuda)

    # CUDA backward
    output_cuda.backward(grad_output.clone(), retain_graph=True)
    dx_cuda = x_cuda.grad.clone()

    # PyTorch backward
    output_pytorch.backward(grad_output.clone(), retain_graph=True)
    dx_pytorch = x_pytorch.grad.clone()

    # Compare dx
    dx_diff = (dx_cuda - dx_pytorch).abs()
    dx_max_diff = dx_diff.max().item()
    dx_mean_diff = dx_diff.mean().item()

    print(f"\ndx max diff: {dx_max_diff:.6e}")
    print(f"dx mean diff: {dx_mean_diff:.6e}")
    print(f"dx CUDA norm: {dx_cuda.norm().item():.6e}")
    print(f"dx PyTorch norm: {dx_pytorch.norm().item():.6e}")

    if dx_max_diff > 1e-3:
        print("\nWARNING: dx gradients differ significantly!")
        # Find where the biggest difference is
        max_idx = dx_diff.argmax()
        t, b, d = max_idx // (B * D), (max_idx % (B * D)) // D, max_idx % D
        print(f"Max diff at t={t}, b={b}, d={d}")
        print(f"CUDA: {dx_cuda[t, b, d].item():.6e}")
        print(f"PyTorch: {dx_pytorch[t, b, d].item():.6e}")

    # Now let's compare weight gradients by running fresh forward/backward
    print("\n=== Weight Gradient Comparison ===")

    # Fresh cells with gradient tracking
    cell_cuda2 = LogSpaceTripleRCell(
        D, n_groups=n_groups, delta_init=-2.0,
        r_h_mode='free',  # No spectral norm to simplify comparison
        diagonal_r_delta=False
    ).to(device).to(dtype)

    cell_pytorch2 = LogSpaceTripleRCell(
        D, n_groups=n_groups, delta_init=-2.0,
        r_h_mode='free',
        diagonal_r_delta=False
    ).to(device).to(dtype)

    # Copy weights
    cell_pytorch2.load_state_dict(cell_cuda2.state_dict())

    # Fresh input
    x2 = torch.randn(T, B, D, device=device, dtype=dtype)

    # CUDA forward/backward
    x_cuda2 = x2.clone().requires_grad_(True)
    out_cuda2, _, _, _ = LogSpaceTripleRFunction.apply(
        True, x_cuda2, log_h0.clone(), sign_h0.clone(),
        cell_cuda2.R_h, cell_cuda2.R_x, cell_cuda2.R_delta,
        cell_cuda2.W_delta, cell_cuda2.W_out,
        cell_cuda2.b, cell_cuda2.b_delta, n_groups
    )
    loss_cuda = out_cuda2.sum()
    loss_cuda.backward()

    # PyTorch forward/backward
    x_pytorch2 = x2.clone().requires_grad_(True)
    out_pytorch2, _, _, _ = cell_pytorch2._forward_pytorch(
        x_pytorch2, log_h0.clone(), sign_h0.clone(),
        cell_pytorch2.R_h, cell_pytorch2.R_x, cell_pytorch2.R_delta
    )
    loss_pytorch = out_pytorch2.sum()
    loss_pytorch.backward()

    # Compare weight gradients
    params_to_check = [
        ('R_h', cell_cuda2.R_h, cell_pytorch2.R_h),
        ('R_x', cell_cuda2.R_x, cell_pytorch2.R_x),
        ('R_delta', cell_cuda2.R_delta, cell_pytorch2.R_delta),
        ('W_delta', cell_cuda2.W_delta, cell_pytorch2.W_delta),
        ('W_out', cell_cuda2.W_out, cell_pytorch2.W_out),
        ('b', cell_cuda2.b, cell_pytorch2.b),
        ('b_delta', cell_cuda2.b_delta, cell_pytorch2.b_delta),
    ]

    for name, p_cuda, p_pytorch in params_to_check:
        if p_cuda.grad is None:
            print(f"{name}: CUDA grad is None!")
            continue
        if p_pytorch.grad is None:
            print(f"{name}: PyTorch grad is None!")
            continue

        diff = (p_cuda.grad - p_pytorch.grad).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        cuda_norm = p_cuda.grad.norm().item()
        pytorch_norm = p_pytorch.grad.norm().item()
        rel_diff = max_diff / (pytorch_norm + 1e-8)

        status = "✓" if rel_diff < 0.01 else "✗"
        print(f"{status} {name:10s}: max_diff={max_diff:.4e}, rel_diff={rel_diff:.4e}, "
              f"cuda_norm={cuda_norm:.4e}, pytorch_norm={pytorch_norm:.4e}")

        if rel_diff >= 0.01:
            # Show more details
            max_idx = diff.argmax()
            if p_cuda.grad.dim() == 2:
                i, j = max_idx // p_cuda.grad.shape[1], max_idx % p_cuda.grad.shape[1]
                print(f"    Max diff at [{i}, {j}]: CUDA={p_cuda.grad[i,j].item():.6e}, "
                      f"PyTorch={p_pytorch.grad[i,j].item():.6e}")
            else:
                print(f"    Max diff at [{max_idx}]: CUDA={p_cuda.grad[max_idx].item():.6e}, "
                      f"PyTorch={p_pytorch.grad[max_idx].item():.6e}")


def test_single_timestep():
    """Test a single timestep to isolate the issue."""
    print("\n\n=== Single Timestep Test ===")

    torch.manual_seed(42)

    T, B, D = 1, 2, 64  # Single timestep
    n_groups = 8
    device = 'cuda'
    dtype = torch.float32

    x = torch.randn(T, B, D, device=device, dtype=dtype)
    log_h0 = torch.full((B, D), LOG_ZERO, device=device, dtype=dtype)
    sign_h0 = torch.ones(B, D, device=device, dtype=dtype)

    # Create cells without spectral norm for simpler debugging
    cell_cuda = LogSpaceTripleRCell(
        D, n_groups=n_groups, r_h_mode='free', diagonal_r_delta=False
    ).to(device).to(dtype)

    cell_pytorch = LogSpaceTripleRCell(
        D, n_groups=n_groups, r_h_mode='free', diagonal_r_delta=False
    ).to(device).to(dtype)
    cell_pytorch.load_state_dict(cell_cuda.state_dict())

    # CUDA forward/backward
    x_cuda = x.clone().requires_grad_(True)
    out_cuda, log_h_cuda, sign_h_cuda, _ = LogSpaceTripleRFunction.apply(
        True, x_cuda, log_h0.clone(), sign_h0.clone(),
        cell_cuda.R_h, cell_cuda.R_x, cell_cuda.R_delta,
        cell_cuda.W_delta, cell_cuda.W_out,
        cell_cuda.b, cell_cuda.b_delta, n_groups
    )
    loss_cuda = out_cuda.sum()
    loss_cuda.backward()

    # PyTorch forward/backward
    x_pytorch = x.clone().requires_grad_(True)
    out_pytorch, log_h_pytorch, sign_h_pytorch, _ = cell_pytorch._forward_pytorch(
        x_pytorch, log_h0.clone(), sign_h0.clone(),
        cell_pytorch.R_h, cell_pytorch.R_x, cell_pytorch.R_delta
    )
    loss_pytorch = out_pytorch.sum()
    loss_pytorch.backward()

    print(f"Output diff: {(out_cuda - out_pytorch).abs().max().item():.6e}")
    print(f"dx diff: {(x_cuda.grad - x_pytorch.grad).abs().max().item():.6e}")

    for name in ['R_h', 'R_x', 'R_delta', 'W_delta', 'W_out', 'b', 'b_delta']:
        g_cuda = getattr(cell_cuda, name).grad
        g_pytorch = getattr(cell_pytorch, name).grad
        if g_cuda is not None and g_pytorch is not None:
            diff = (g_cuda - g_pytorch).abs().max().item()
            rel = diff / (g_pytorch.abs().max().item() + 1e-8)
            status = "✓" if rel < 0.01 else "✗"
            print(f"{status} {name:10s}: max_diff={diff:.4e}, rel_diff={rel:.4e}")


if __name__ == '__main__':
    test_gradient_comparison()
    test_single_timestep()
