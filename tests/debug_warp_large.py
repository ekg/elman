"""
Debug warp kernel with larger configs.
"""

import torch
import hasty_pytorch_lib

device = torch.device('cuda')
dtype = torch.bfloat16

print("Testing larger configs:")
for config in [
    {'B': 1, 'T': 512, 'H': 1},
    {'B': 1, 'T': 512, 'H': 8},
    {'B': 1, 'T': 512, 'H': 32},
    {'B': 1, 'T': 512, 'H': 98},
    {'B': 8, 'T': 512, 'H': 98},
    {'B': 16, 'T': 512, 'H': 98},
    {'B': 16, 'T': 1024, 'H': 98},
]:
    B = config['B']
    T = config['T']
    H = config['H']
    n_state, head_v_dim = 32, 32

    torch.manual_seed(42)
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
    mean_diff = (output_fused - output_warp).abs().mean().item()

    status = "OK" if max_diff < 0.1 else "ERROR"
    print(f"  B={B:2d} T={T:4d} H={H:3d}: max={max_diff:.4f} mean={mean_diff:.6f}  [{status}]")
