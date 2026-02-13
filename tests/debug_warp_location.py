"""
Find where the error is concentrated in the warp kernel output.
"""

import torch
import hasty_pytorch_lib

device = torch.device('cuda')
dtype = torch.bfloat16

B, T, H, n_state, head_v_dim = 1, 64, 8, 32, 32

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

diff = (output_fused - output_warp).abs()

print(f"Shape: B={B}, T={T}, H={H}, head_v_dim={head_v_dim}")
print(f"Overall max diff: {diff.max().item():.6f}")
print(f"Overall mean diff: {diff.mean().item():.6f}")

print("\nPer-head max diff:")
for h in range(H):
    head_diff = diff[0, :, h, :].max().item()
    print(f"  Head {h}: {head_diff:.6f}")

print("\nPer-timestep max diff (first 10 and last 10):")
for t in list(range(10)) + list(range(T-10, T)):
    t_diff = diff[0, t, :, :].max().item()
    print(f"  t={t:3d}: {t_diff:.6f}")

# Find location of max error
max_idx = diff.argmax()
b_idx = max_idx // (T * H * head_v_dim)
t_idx = (max_idx % (T * H * head_v_dim)) // (H * head_v_dim)
h_idx = (max_idx % (H * head_v_dim)) // head_v_dim
v_idx = max_idx % head_v_dim
print(f"\nMax error location: batch={b_idx}, t={t_idx}, head={h_idx}, v_idx={v_idx}")
print(f"  Fused value: {output_fused[b_idx, t_idx, h_idx, v_idx].item():.6f}")
print(f"  Warp value:  {output_warp[b_idx, t_idx, h_idx, v_idx].item():.6f}")
