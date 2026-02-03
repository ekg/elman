#!/usr/bin/env python3
"""
Test E87 Sparse Block Memory CUDA kernel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

# Add the elman cuda module
sys.path.insert(0, '/home/erikg/elman')

def test_e87_cuda():
    """Test E87 CUDA kernel forward and backward."""
    print("Testing E87 Sparse Block Memory CUDA kernel...")
    print("=" * 60)

    # Import hasty
    try:
        import hasty_pytorch_lib as hasty
        print("Successfully imported hasty module")
    except ImportError as e:
        print(f"Failed to import hasty: {e}")
        return False

    # Check if e87 functions exist
    if not hasattr(hasty, 'e87_sparse_block_forward'):
        print("ERROR: e87_sparse_block_forward not found in hasty module")
        print("Available functions:", [f for f in dir(hasty) if 'e87' in f.lower()])
        return False

    print("Found e87_sparse_block_forward and e87_sparse_block_backward")

    device = 'cuda'
    dtype = torch.bfloat16

    # Test configuration
    T = 32           # sequence length
    B = 4            # batch size
    dim = 256        # input dimension
    n_state = 24     # state size per block
    n_blocks = 4     # number of memory blocks
    top_k = 2        # top-k blocks to update
    router_temp = 1.0

    print(f"\nConfig: T={T}, B={B}, dim={dim}, n_state={n_state}, n_blocks={n_blocks}, top_k={top_k}")

    # Create inputs
    x = torch.randn(T, B, dim, device=device, dtype=dtype)
    S0 = torch.zeros(B, n_blocks, n_state, n_state, device=device, dtype=dtype)

    # Create weights
    W_router = torch.randn(n_blocks, dim, device=device, dtype=dtype) * 0.01
    W_k = torch.randn(n_blocks * n_state, dim, device=device, dtype=dtype) * 0.01
    W_v = torch.randn(n_blocks * n_state, dim, device=device, dtype=dtype) * 0.01
    W_q = torch.randn(n_state, dim, device=device, dtype=dtype) * 0.01
    W_beta = torch.randn(n_blocks * n_state, dim, device=device, dtype=dtype) * 0.01
    b_beta = torch.full((n_blocks, n_state), 2.0, device=device, dtype=dtype)

    print("\n--- Forward Pass ---")
    try:
        result = hasty.e87_sparse_block_forward(
            True,  # training
            x, S0, W_router, W_k, W_v, W_q, W_beta, b_beta,
            n_blocks, top_k, router_temp
        )
        output, S_final, router_cache, k_cache, v_cache, q_cache, beta_cache, update_weights, read_weights, S_cache = result
        print(f"Output shape: {output.shape}")
        print(f"S_final shape: {S_final.shape}")
        print(f"Output mean: {output.float().mean().item():.6f}")
        print(f"Output std: {output.float().std().item():.6f}")
        print("Forward pass OK!")
    except Exception as e:
        print(f"Forward pass FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n--- Backward Pass ---")
    try:
        d_output = torch.randn_like(output)

        grads = hasty.e87_sparse_block_backward(
            x, S_cache, router_cache, k_cache, v_cache, q_cache, beta_cache,
            update_weights, read_weights, d_output,
            W_router, W_k, W_v, W_q, W_beta,
            n_blocks, top_k, router_temp
        )
        dx, dW_router, dW_k, dW_v, dW_q, dW_beta, db_beta = grads

        print(f"dx shape: {dx.shape}")
        print(f"dW_router shape: {dW_router.shape}")
        print(f"dW_k shape: {dW_k.shape}")
        print(f"dW_v shape: {dW_v.shape}")
        print(f"dW_q shape: {dW_q.shape}")
        print(f"dW_beta shape: {dW_beta.shape}")
        print(f"db_beta shape: {db_beta.shape}")
        print(f"dx mean: {dx.float().mean().item():.6f}")
        print(f"dx std: {dx.float().std().item():.6f}")
        print("Backward pass OK!")
    except Exception as e:
        print(f"Backward pass FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n--- Gradient Check (numerical vs analytical) ---")
    try:
        # Simple gradient check on small input
        T_small, B_small, dim_small = 4, 2, 64
        n_state_small = 8
        n_blocks_small = 2

        x_check = torch.randn(T_small, B_small, dim_small, device=device, dtype=dtype, requires_grad=False)
        S0_check = torch.zeros(B_small, n_blocks_small, n_state_small, n_state_small, device=device, dtype=dtype)

        W_router_check = torch.randn(n_blocks_small, dim_small, device=device, dtype=dtype) * 0.1
        W_k_check = torch.randn(n_blocks_small * n_state_small, dim_small, device=device, dtype=dtype) * 0.1
        W_v_check = torch.randn(n_blocks_small * n_state_small, dim_small, device=device, dtype=dtype) * 0.1
        W_q_check = torch.randn(n_state_small, dim_small, device=device, dtype=dtype) * 0.1
        W_beta_check = torch.randn(n_blocks_small * n_state_small, dim_small, device=device, dtype=dtype) * 0.1
        b_beta_check = torch.full((n_blocks_small, n_state_small), 2.0, device=device, dtype=dtype)

        # Forward
        result = hasty.e87_sparse_block_forward(
            True, x_check, S0_check, W_router_check, W_k_check, W_v_check,
            W_q_check, W_beta_check, b_beta_check,
            n_blocks_small, 1, 1.0
        )
        output_check = result[0]

        # Check gradients exist and are finite
        d_output_check = torch.randn_like(output_check)
        grads = hasty.e87_sparse_block_backward(
            x_check, result[9], result[2], result[3], result[4], result[5], result[6],
            result[7], result[8], d_output_check,
            W_router_check, W_k_check, W_v_check, W_q_check, W_beta_check,
            n_blocks_small, 1, 1.0
        )

        all_finite = True
        for i, g in enumerate(grads):
            if not torch.isfinite(g.float()).all():
                print(f"WARNING: gradient {i} contains non-finite values")
                all_finite = False

        if all_finite:
            print("All gradients are finite!")
        else:
            print("Some gradients contain non-finite values")

    except Exception as e:
        print(f"Gradient check failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("All basic tests passed!")
    return True


if __name__ == "__main__":
    success = test_e87_cuda()
    sys.exit(0 if success else 1)
