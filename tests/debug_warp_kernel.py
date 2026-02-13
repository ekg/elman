"""
Debug E88 Warp-Optimized kernel to find the bug.
"""

import torch
import hasty_pytorch_lib

def debug_single_step():
    """Test single timestep to isolate the bug."""

    device = torch.device('cuda')
    dtype = torch.bfloat16

    B, T, H, n_state, head_v_dim = 1, 1, 1, 32, 32

    # Simple deterministic inputs
    k = torch.ones(B, T, H, n_state, device=device, dtype=dtype)
    v = torch.ones(B, T, H, head_v_dim, device=device, dtype=dtype) * 2.0
    q = torch.ones(B, T, H, n_state, device=device, dtype=dtype)
    decay = torch.ones(B, T, H, device=device, dtype=dtype) * 0.9
    g = torch.ones(B, T, H, head_v_dim, device=device, dtype=dtype)
    S0 = torch.zeros(B, H, n_state, head_v_dim, device=device, dtype=dtype)

    # Cache
    checkpoint_interval = 16
    num_checkpoints = (T + checkpoint_interval - 1) // checkpoint_interval + 1
    cache_size = num_checkpoints * B * H * n_state * head_v_dim + B * T * H * head_v_dim

    output_fused = torch.empty(B, T, H, head_v_dim, device=device, dtype=dtype)
    output_chunked = torch.empty(B, T, H, head_v_dim, device=device, dtype=dtype)
    output_warp = torch.empty(B, T, H, head_v_dim, device=device, dtype=dtype)

    S_cache = torch.empty(cache_size, device=device, dtype=dtype)

    # Run all kernels
    hasty_pytorch_lib.e88_fused_forward(True, k, v, q, decay, g, S0, output_fused, S_cache.clone(), H, True)
    hasty_pytorch_lib.e88_chunked_forward(True, k, v, q, decay, g, S0, output_chunked, S_cache.clone(), H, True)
    hasty_pytorch_lib.e88_warp_optimized_forward(True, k, v, q, decay, g, S0, output_warp, S_cache.clone(), H, True)

    print("Single timestep T=1:")
    print(f"  Fused output:   {output_fused[0, 0, 0, :4].tolist()}")
    print(f"  Chunked output: {output_chunked[0, 0, 0, :4].tolist()}")
    print(f"  Warp output:    {output_warp[0, 0, 0, :4].tolist()}")
    print(f"  Max diff (fused vs warp): {(output_fused - output_warp).abs().max().item():.6f}")

    return (output_fused - output_warp).abs().max().item()


def debug_progressive():
    """Test increasing T to see where error starts."""

    device = torch.device('cuda')
    dtype = torch.bfloat16

    B, H, n_state, head_v_dim = 1, 1, 32, 32

    print("\nProgressive T test:")
    for T in [1, 2, 4, 8, 16, 32, 64]:
        k = torch.randn(B, T, H, n_state, device=device, dtype=dtype)
        v = torch.randn(B, T, H, head_v_dim, device=device, dtype=dtype)
        q = torch.randn(B, T, H, n_state, device=device, dtype=dtype)
        decay = torch.rand(B, T, H, device=device, dtype=dtype) * 0.5 + 0.5
        g = torch.randn(B, T, H, head_v_dim, device=device, dtype=dtype)
        S0 = torch.zeros(B, H, n_state, head_v_dim, device=device, dtype=dtype)

        checkpoint_interval = 16
        num_checkpoints = (T + checkpoint_interval - 1) // checkpoint_interval + 1
        cache_size = num_checkpoints * B * H * n_state * head_v_dim + B * T * H * head_v_dim

        output_fused = torch.empty(B, T, H, head_v_dim, device=device, dtype=dtype)
        output_warp = torch.empty(B, T, H, head_v_dim, device=device, dtype=dtype)
        S_cache = torch.empty(cache_size, device=device, dtype=dtype)

        hasty_pytorch_lib.e88_fused_forward(True, k, v, q, decay, g, S0, output_fused, S_cache.clone(), H, True)
        hasty_pytorch_lib.e88_warp_optimized_forward(True, k, v, q, decay, g, S0, output_warp, S_cache.clone(), H, True)

        max_diff = (output_fused - output_warp).abs().max().item()
        print(f"  T={T:3d}: max diff = {max_diff:.6f}")


def debug_with_multiple_heads():
    """Test with multiple heads to isolate head-related bugs."""

    device = torch.device('cuda')
    dtype = torch.bfloat16

    B, T, n_state, head_v_dim = 1, 4, 32, 32

    print("\nMultiple heads test (T=4):")
    for H in [1, 2, 4, 8]:
        k = torch.randn(B, T, H, n_state, device=device, dtype=dtype)
        v = torch.randn(B, T, H, head_v_dim, device=device, dtype=dtype)
        q = torch.randn(B, T, H, n_state, device=device, dtype=dtype)
        decay = torch.rand(B, T, H, device=device, dtype=dtype) * 0.5 + 0.5
        g = torch.randn(B, T, H, head_v_dim, device=device, dtype=dtype)
        S0 = torch.zeros(B, H, n_state, head_v_dim, device=device, dtype=dtype)

        checkpoint_interval = 16
        num_checkpoints = (T + checkpoint_interval - 1) // checkpoint_interval + 1
        cache_size = num_checkpoints * B * H * n_state * head_v_dim + B * T * H * head_v_dim

        output_fused = torch.empty(B, T, H, head_v_dim, device=device, dtype=dtype)
        output_warp = torch.empty(B, T, H, head_v_dim, device=device, dtype=dtype)
        S_cache = torch.empty(cache_size, device=device, dtype=dtype)

        hasty_pytorch_lib.e88_fused_forward(True, k, v, q, decay, g, S0, output_fused, S_cache.clone(), H, True)
        hasty_pytorch_lib.e88_warp_optimized_forward(True, k, v, q, decay, g, S0, output_warp, S_cache.clone(), H, True)

        max_diff = (output_fused - output_warp).abs().max().item()
        print(f"  H={H:3d}: max diff = {max_diff:.6f}")


if __name__ == '__main__':
    debug_single_step()
    debug_progressive()
    debug_with_multiple_heads()
