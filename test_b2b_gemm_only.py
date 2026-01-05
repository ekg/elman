#!/usr/bin/env python3
"""Test just the B2B GEMM operation against reference."""

import torch
import sys
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, '/home/erikg/elman/elman/cuda')

import hasty_pytorch_lib

def test_b2b_gemm():
    """Test the B2B GEMM: output = h_prev @ V_h^T @ U_h^T"""
    torch.manual_seed(42)

    batch = 256
    dim = 1536
    rank = 256
    device = 'cuda'
    dtype = torch.bfloat16

    # Create data
    h_prev = torch.randn(batch, dim, device=device, dtype=dtype)
    U_h = torch.randn(dim, rank, device=device, dtype=dtype) * 0.1
    V_h = torch.randn(rank, dim, device=device, dtype=dtype) * 0.1

    # Reference: compute using PyTorch
    # We want: h_prev @ V_h^T @ U_h^T
    # Which is (U_h @ V_h @ h_prev^T)^T
    intermediate_ref = torch.mm(h_prev.float(), V_h.T.float())  # [batch, rank]
    output_ref = torch.mm(intermediate_ref, U_h.T.float())  # [batch, dim]
    output_ref = output_ref.to(dtype)

    print(f"Testing B2B GEMM: batch={batch}, dim={dim}, rank={rank}")

    # Run our kernel via the forward pass with h0=zeros and single step x
    # This won't give us direct access. Let me check what we can test...
    # Actually, let's just run both forward passes with steps=1 and h0=h_prev
    # Then the h contribution at step 1 should show us the difference

    # Create inputs that isolate h_prev contribution
    x = torch.zeros(1, batch, dim, device=device, dtype=dtype)
    h0 = h_prev.clone()
    b = torch.zeros(dim, device=device, dtype=dtype)

    # Set U_z, V_z to zero so gate is sigmoid(0) = 0.5
    U_z = torch.zeros(dim, rank, device=device, dtype=dtype)
    V_z = torch.zeros(rank, dim, device=device, dtype=dtype)

    # Set U_x, V_x to zero so x contribution is zero
    U_x = torch.zeros(dim, rank, device=device, dtype=dtype)
    V_x = torch.zeros(rank, dim, device=device, dtype=dtype)

    # Run fused kernel
    h_fused, out_fused, _ = hasty_pytorch_lib.pure_lowrank_elman_forward_fused(
        False, x, h0, U_h, V_h, U_x, V_x, U_z, V_z, b)
    torch.cuda.synchronize()

    # Run B2B kernel
    h_b2b, out_b2b, _ = hasty_pytorch_lib.b2b_lowrank_elman_forward(
        False, x, h0, U_h, V_h, U_x, V_x, U_z, V_z, b)
    torch.cuda.synchronize()

    # The hidden state at t=1 is:
    # h[1] = sigmoid(U_h @ V_h @ h0 + U_x @ V_x @ x[0] + U_z @ V_z @ x[0] + b) * h0 + tanh(...)
    # With x=0, U_z=0: h[1] = sigmoid(U_h @ V_h @ h0) * h0 + tanh(U_h @ V_h @ h0)
    # The key term U_h @ V_h @ h0 should be computed correctly

    # Compare hidden states
    diff_h = (h_fused[1] - h_b2b[1]).abs()
    print(f"\nh[1] (after 1 step from h0=h_prev):")
    print(f"  max_diff={diff_h.max():.6f}, mean_diff={diff_h.mean():.6f}")

    # Let me also compute what the expected result would be manually
    # From the kernel: h_t = tanh(U_h @ V_h @ h_{t-1} + U_x @ V_x @ x_t + b)
    # With x=0, b=0, U_x=V_x=0: h[1] = tanh(U_h @ V_h @ h0)
    # Computing: h0 @ V_h^T @ U_h^T (row-major convention)
    h_proj = torch.mm(h0.float(), V_h.T.float())  # [batch, rank]
    h_proj = torch.mm(h_proj, U_h.T.float())  # [batch, dim]
    h_proj = h_proj.to(dtype)

    h_expected = torch.tanh(h_proj.float()).to(dtype)

    diff_fused_expected = (h_fused[1] - h_expected).abs()
    diff_b2b_expected = (h_b2b[1] - h_expected).abs()

    print(f"\nFUSED vs reference (h0 @ V_h^T @ U_h^T):")
    print(f"  max_diff={diff_fused_expected.max():.6f}, mean_diff={diff_fused_expected.mean():.6f}")

    print(f"\nB2B vs reference (h0 @ V_h^T @ U_h^T):")
    print(f"  max_diff={diff_b2b_expected.max():.6f}, mean_diff={diff_b2b_expected.mean():.6f}")

    # Find where max error is
    max_idx = diff_h.argmax().item()
    batch_idx = max_idx // dim
    dim_idx = max_idx % dim

    print(f"\nMax diff location: batch={batch_idx}, dim={dim_idx}")
    print(f"  h_fused[1]: {h_fused[1, batch_idx, dim_idx]:.6f}")
    print(f"  h_b2b[1]:   {h_b2b[1, batch_idx, dim_idx]:.6f}")
    print(f"  h_expected: {h_expected[batch_idx, dim_idx]:.6f}")

if __name__ == '__main__':
    test_b2b_gemm()
