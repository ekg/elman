"""
Benchmark E88 Chunked Prefetch vs E88 Fused Forward

Compares:
1. E88 Fused: Current best forward implementation
2. E88 Chunked: Prefetch chunks into shared memory before processing
"""

import torch
import time
import hasty_pytorch_lib

def benchmark_kernels(B=16, T=512, H=98, n_state=32, head_v_dim=32, warmup=10, iterations=100):
    """Benchmark E88 fused vs chunked forward kernels."""

    device = torch.device('cuda')
    dtype = torch.bfloat16

    # Create inputs
    k = torch.randn(B, T, H, n_state, device=device, dtype=dtype)
    v = torch.randn(B, T, H, head_v_dim, device=device, dtype=dtype)
    q = torch.randn(B, T, H, n_state, device=device, dtype=dtype)
    decay = torch.rand(B, T, H, device=device, dtype=dtype) * 0.5 + 0.5  # 0.5-1.0
    g = torch.randn(B, T, H, head_v_dim, device=device, dtype=dtype)
    S0 = torch.zeros(B, H, n_state, head_v_dim, device=device, dtype=dtype)

    # Allocate outputs and cache
    output_fused = torch.empty(B, T, H, head_v_dim, device=device, dtype=dtype)
    output_chunked = torch.empty(B, T, H, head_v_dim, device=device, dtype=dtype)

    checkpoint_interval = 16
    num_checkpoints = (T + checkpoint_interval - 1) // checkpoint_interval + 1
    cache_size = num_checkpoints * B * H * n_state * head_v_dim + B * T * H * head_v_dim
    S_cache = torch.empty(cache_size, device=device, dtype=dtype)

    apply_gate = True
    training = True

    print(f"\nConfig: B={B}, T={T}, H={H}, n_state={n_state}, head_v_dim={head_v_dim}")
    print(f"State size per head: {n_state}x{head_v_dim} = {n_state * head_v_dim}")
    print(f"Total state: {B * H * n_state * head_v_dim * 2 / 1024 / 1024:.2f} MB")
    print("-" * 60)

    # Warmup fused
    for _ in range(warmup):
        hasty_pytorch_lib.e88_fused_forward(
            training, k, v, q, decay, g, S0, output_fused, S_cache.clone(), H, apply_gate
        )
        torch.cuda.synchronize()

    # Benchmark fused
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iterations):
        hasty_pytorch_lib.e88_fused_forward(
            training, k, v, q, decay, g, S0, output_fused, S_cache.clone(), H, apply_gate
        )
    torch.cuda.synchronize()
    fused_time = (time.perf_counter() - t0) / iterations * 1000

    # Warmup chunked
    for _ in range(warmup):
        hasty_pytorch_lib.e88_chunked_forward(
            training, k, v, q, decay, g, S0, output_chunked, S_cache.clone(), H, apply_gate
        )
        torch.cuda.synchronize()

    # Benchmark chunked
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iterations):
        hasty_pytorch_lib.e88_chunked_forward(
            training, k, v, q, decay, g, S0, output_chunked, S_cache.clone(), H, apply_gate
        )
    torch.cuda.synchronize()
    chunked_time = (time.perf_counter() - t0) / iterations * 1000

    # Verify correctness
    hasty_pytorch_lib.e88_fused_forward(
        training, k, v, q, decay, g, S0, output_fused, S_cache.clone(), H, apply_gate
    )
    hasty_pytorch_lib.e88_chunked_forward(
        training, k, v, q, decay, g, S0, output_chunked, S_cache.clone(), H, apply_gate
    )

    max_diff = (output_fused - output_chunked).abs().max().item()
    mean_diff = (output_fused - output_chunked).abs().mean().item()

    # Calculate throughput
    tokens = B * T
    fused_toks = tokens / (fused_time / 1000)
    chunked_toks = tokens / (chunked_time / 1000)

    speedup = fused_time / chunked_time

    print(f"E88 Fused:   {fused_time:.3f} ms  ({fused_toks/1000:.1f}K tok/s)")
    print(f"E88 Chunked: {chunked_time:.3f} ms  ({chunked_toks/1000:.1f}K tok/s)")
    print(f"Speedup: {speedup:.3f}x {'FASTER' if speedup > 1 else 'SLOWER'}")
    print(f"Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")

    return {
        'fused_ms': fused_time,
        'chunked_ms': chunked_time,
        'speedup': speedup,
        'max_diff': max_diff
    }


if __name__ == '__main__':
    print("=" * 60)
    print("E88 Chunked Prefetch Benchmark")
    print("=" * 60)

    # Test different configurations
    configs = [
        # Standard E88 480M config
        {'B': 16, 'T': 512, 'H': 98, 'n_state': 32, 'head_v_dim': 32},
        # Smaller batch
        {'B': 8, 'T': 512, 'H': 98, 'n_state': 32, 'head_v_dim': 32},
        # Longer sequence
        {'B': 16, 'T': 1024, 'H': 98, 'n_state': 32, 'head_v_dim': 32},
        # Different state size
        {'B': 16, 'T': 512, 'H': 64, 'n_state': 64, 'head_v_dim': 64},
    ]

    for cfg in configs:
        try:
            benchmark_kernels(**cfg)
        except Exception as e:
            print(f"Config {cfg} failed: {e}")
        print()
