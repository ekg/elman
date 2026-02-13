"""
Debug warp kernel around checkpoint boundaries (every 16 steps).
"""

import torch
import hasty_pytorch_lib

device = torch.device('cuda')
dtype = torch.bfloat16

B, H, n_state, head_v_dim = 1, 1, 32, 32

print("Testing around checkpoint boundaries (interval=16):")
for T in [15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65, 128, 256, 512]:
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
    # Find which timestep has max error
    diff_per_t = (output_fused - output_warp).abs().view(T, -1).max(dim=1)[0]
    worst_t = diff_per_t.argmax().item()
    worst_val = diff_per_t[worst_t].item()

    status = "OK" if max_diff < 0.01 else "ERROR"
    print(f"  T={T:4d}: max diff = {max_diff:.6f} at t={worst_t:3d}  [{status}]")
