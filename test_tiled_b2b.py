#!/usr/bin/env python3
"""Debug test for tiled B2B GEMM kernel."""

import torch
import sys
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, '/home/erikg/elman/elman/cuda')

import hasty_pytorch_lib

def test_benchmark_setup():
    """Test with benchmark-like setup."""
    torch.manual_seed(42)

    batch = 256
    dim = 1536
    rank = 256
    steps = 2
    device = 'cuda'
    dtype = torch.bfloat16

    # Create data matching benchmark
    x = torch.randn(steps, batch, dim, device=device, dtype=dtype)
    h0 = torch.zeros(batch, dim, device=device, dtype=dtype)

    U_h = torch.randn(dim, rank, device=device, dtype=dtype) * 0.1
    V_h = torch.randn(rank, dim, device=device, dtype=dtype) * 0.1
    U_x = torch.randn(dim, rank, device=device, dtype=dtype) * 0.1
    V_x = torch.randn(rank, dim, device=device, dtype=dtype) * 0.1
    U_z = torch.randn(dim, rank, device=device, dtype=dtype) * 0.1
    V_z = torch.randn(rank, dim, device=device, dtype=dtype) * 0.1
    b = torch.zeros(dim, device=device, dtype=dtype)

    print(f"Test: batch={batch}, dim={dim}, rank={rank}, steps={steps}")

    # Run both kernels
    h_fused, out_fused, _ = hasty_pytorch_lib.pure_lowrank_elman_forward_fused(
        False, x, h0, U_h, V_h, U_x, V_x, U_z, V_z, b)
    torch.cuda.synchronize()

    h_b2b, out_b2b, _ = hasty_pytorch_lib.b2b_lowrank_elman_forward(
        False, x, h0, U_h, V_h, U_x, V_x, U_z, V_z, b)
    torch.cuda.synchronize()

    # Compare hidden states at each step
    print("\nHidden states comparison:")
    for t in range(steps + 1):
        diff = (h_fused[t] - h_b2b[t]).abs()
        print(f"  h[{t}]: max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}")

    # Compare outputs at each step
    print("\nOutput comparison:")
    for t in range(steps):
        diff = (out_fused[t] - out_b2b[t]).abs()
        print(f"  out[{t}]: max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}")

    # Overall comparison
    out_diff = (out_fused - out_b2b).abs()
    print(f"\nOverall output max diff: {out_diff.max():.6f}")

    # Debug: find where the error is largest
    h2_diff = (h_fused[2] - h_b2b[2]).abs()
    max_idx = h2_diff.argmax().item()
    batch_idx = max_idx // dim
    dim_idx = max_idx % dim

    print(f"\nh[2] max error at batch={batch_idx}, dim={dim_idx}")
    print(f"  FUSED value: {h_fused[2, batch_idx, dim_idx]:.6f}")
    print(f"  B2B value:   {h_b2b[2, batch_idx, dim_idx]:.6f}")
    print(f"  Error:       {h2_diff[batch_idx, dim_idx]:.6f}")

    # Check nearby values
    print(f"\nNearby h[2][{batch_idx}] values at dims {max(0,dim_idx-2)}:{dim_idx+3}:")
    print(f"  FUSED: {h_fused[2, batch_idx, max(0,dim_idx-2):dim_idx+3].tolist()}")
    print(f"  B2B:   {h_b2b[2, batch_idx, max(0,dim_idx-2):dim_idx+3].tolist()}")

    # Check h[1] at same location (the input to the problematic step)
    print(f"\nh[1][{batch_idx}] at same dim range:")
    print(f"  FUSED: {h_fused[1, batch_idx, max(0,dim_idx-2):dim_idx+3].tolist()}")
    print(f"  B2B:   {h_b2b[1, batch_idx, max(0,dim_idx-2):dim_idx+3].tolist()}")

if __name__ == '__main__':
    test_benchmark_setup()
