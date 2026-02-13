"""
Benchmark ALL E88 Forward Kernels

Compares:
1. E88 Fused: Baseline with [B, T, H, dim] layout
2. E88 Chunked: Prefetch chunks into shared memory
3. E88 Warp-Optimized: Full thread utilization with parallel state update
4. E88 Coalesced: Transposed state matrix for coalesced memory access
"""

import torch
import time
import hasty_pytorch_lib

def benchmark_kernels(B=16, T=512, H=98, n_state=32, head_v_dim=32, warmup=10, iterations=100):
    """Benchmark all E88 forward kernels."""

    device = torch.device('cuda')
    dtype = torch.bfloat16

    # Create inputs
    k = torch.randn(B, T, H, n_state, device=device, dtype=dtype)
    v = torch.randn(B, T, H, head_v_dim, device=device, dtype=dtype)
    q = torch.randn(B, T, H, n_state, device=device, dtype=dtype)
    decay = torch.rand(B, T, H, device=device, dtype=dtype) * 0.5 + 0.5
    g = torch.randn(B, T, H, head_v_dim, device=device, dtype=dtype)
    S0 = torch.zeros(B, H, n_state, head_v_dim, device=device, dtype=dtype)

    # Allocate outputs
    output_fused = torch.empty(B, T, H, head_v_dim, device=device, dtype=dtype)
    output_chunked = torch.empty(B, T, H, head_v_dim, device=device, dtype=dtype)
    output_warp = torch.empty(B, T, H, head_v_dim, device=device, dtype=dtype)
    output_coalesced = torch.empty(B, T, H, head_v_dim, device=device, dtype=dtype)

    # Cache
    checkpoint_interval = 16
    num_checkpoints = (T + checkpoint_interval - 1) // checkpoint_interval + 1
    cache_size = num_checkpoints * B * H * n_state * head_v_dim + B * T * H * head_v_dim
    S_cache = torch.empty(cache_size, device=device, dtype=dtype)

    apply_gate = True
    training = True

    print(f"\nConfig: B={B}, T={T}, H={H}, n_state={n_state}, head_v_dim={head_v_dim}")
    print(f"Tokens: {B * T:,}")
    print("-" * 70)

    results = {}

    # Benchmark E88 Fused
    for _ in range(warmup):
        hasty_pytorch_lib.e88_fused_forward(
            training, k, v, q, decay, g, S0, output_fused, S_cache.clone(), H, apply_gate
        )
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iterations):
        hasty_pytorch_lib.e88_fused_forward(
            training, k, v, q, decay, g, S0, output_fused, S_cache.clone(), H, apply_gate
        )
    torch.cuda.synchronize()
    fused_time = (time.perf_counter() - t0) / iterations * 1000
    results['fused'] = fused_time

    # Benchmark E88 Chunked
    for _ in range(warmup):
        hasty_pytorch_lib.e88_chunked_forward(
            training, k, v, q, decay, g, S0, output_chunked, S_cache.clone(), H, apply_gate
        )
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iterations):
        hasty_pytorch_lib.e88_chunked_forward(
            training, k, v, q, decay, g, S0, output_chunked, S_cache.clone(), H, apply_gate
        )
    torch.cuda.synchronize()
    chunked_time = (time.perf_counter() - t0) / iterations * 1000
    results['chunked'] = chunked_time

    # Benchmark E88 Warp-Optimized
    for _ in range(warmup):
        hasty_pytorch_lib.e88_warp_optimized_forward(
            training, k, v, q, decay, g, S0, output_warp, S_cache.clone(), H, apply_gate
        )
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iterations):
        hasty_pytorch_lib.e88_warp_optimized_forward(
            training, k, v, q, decay, g, S0, output_warp, S_cache.clone(), H, apply_gate
        )
    torch.cuda.synchronize()
    warp_time = (time.perf_counter() - t0) / iterations * 1000
    results['warp'] = warp_time

    # Benchmark E88 Coalesced
    coalesced_time = None
    try:
        for _ in range(warmup):
            hasty_pytorch_lib.e88_coalesced_forward(
                training, k, v, q, decay, g, S0, output_coalesced, S_cache.clone(), H, apply_gate
            )
            torch.cuda.synchronize()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iterations):
            hasty_pytorch_lib.e88_coalesced_forward(
                training, k, v, q, decay, g, S0, output_coalesced, S_cache.clone(), H, apply_gate
            )
        torch.cuda.synchronize()
        coalesced_time = (time.perf_counter() - t0) / iterations * 1000
        results['coalesced'] = coalesced_time
    except Exception as e:
        print(f"Coalesced kernel not available: {e}")

    # Verify correctness
    hasty_pytorch_lib.e88_fused_forward(
        training, k, v, q, decay, g, S0, output_fused, S_cache.clone(), H, apply_gate
    )
    hasty_pytorch_lib.e88_chunked_forward(
        training, k, v, q, decay, g, S0, output_chunked, S_cache.clone(), H, apply_gate
    )
    hasty_pytorch_lib.e88_warp_optimized_forward(
        training, k, v, q, decay, g, S0, output_warp, S_cache.clone(), H, apply_gate
    )
    if coalesced_time is not None:
        hasty_pytorch_lib.e88_coalesced_forward(
            training, k, v, q, decay, g, S0, output_coalesced, S_cache.clone(), H, apply_gate
        )

    fused_chunked_diff = (output_fused - output_chunked).abs().max().item()
    fused_warp_diff = (output_fused - output_warp).abs().max().item()
    fused_coalesced_diff = (output_fused - output_coalesced).abs().max().item() if coalesced_time else None

    # Calculate throughput
    tokens = B * T
    fused_toks = tokens / (fused_time / 1000)
    chunked_toks = tokens / (chunked_time / 1000)
    warp_toks = tokens / (warp_time / 1000)
    coalesced_toks = tokens / (coalesced_time / 1000) if coalesced_time else 0

    # Find best
    times = {'fused': fused_time, 'chunked': chunked_time, 'warp': warp_time}
    if coalesced_time:
        times['coalesced'] = coalesced_time
    best_name = min(times, key=times.get)
    best_time = times[best_name]

    print(f"E88 Fused:         {fused_time:.3f} ms  ({fused_toks/1000:.1f}K tok/s)")
    print(f"E88 Chunked:       {chunked_time:.3f} ms  ({chunked_toks/1000:.1f}K tok/s)  {fused_time/chunked_time:.2f}x vs fused")
    print(f"E88 Warp-Optimized:{warp_time:.3f} ms  ({warp_toks/1000:.1f}K tok/s)  {fused_time/warp_time:.2f}x vs fused")
    if coalesced_time:
        print(f"E88 Coalesced:     {coalesced_time:.3f} ms  ({coalesced_toks/1000:.1f}K tok/s)  {fused_time/coalesced_time:.2f}x vs fused")
    print(f"\nBest: {best_name} ({best_time:.3f} ms)")
    print(f"Speedup over fused: {fused_time/best_time:.2f}x")
    print(f"\nCorrectness:")
    print(f"  Fused vs Chunked: {fused_chunked_diff:.6f}")
    print(f"  Fused vs Warp:    {fused_warp_diff:.6f}")
    if fused_coalesced_diff is not None:
        print(f"  Fused vs Coalesced: {fused_coalesced_diff:.6f}")

    return results


if __name__ == '__main__':
    print("=" * 70)
    print("E88 All Kernels Benchmark")
    print("=" * 70)

    # Standard config
    print("\n[Standard E88 480M Config]")
    benchmark_kernels(B=16, T=512, H=98, n_state=32, head_v_dim=32)

    # Longer sequence
    print("\n[Longer Sequence T=1024]")
    benchmark_kernels(B=16, T=1024, H=98, n_state=32, head_v_dim=32)

    # Smaller batch
    print("\n[Smaller Batch B=8]")
    benchmark_kernels(B=8, T=512, H=98, n_state=32, head_v_dim=32)

    # Different state size
    print("\n[n_state=16]")
    try:
        benchmark_kernels(B=16, T=512, H=64, n_state=16, head_v_dim=16)
    except Exception as e:
        print(f"Failed: {e}")

    # Larger state size
    print("\n[n_state=48]")
    try:
        benchmark_kernels(B=16, T=512, H=64, n_state=48, head_v_dim=48)
    except Exception as e:
        print(f"Failed: {e}")

    print("\n[n_state=64]")
    try:
        benchmark_kernels(B=16, T=512, H=32, n_state=64, head_v_dim=64)
    except Exception as e:
        print(f"Failed: {e}")

    # Very long sequence
    print("\n[T=2048]")
    try:
        benchmark_kernels(B=8, T=2048, H=98, n_state=32, head_v_dim=32)
    except Exception as e:
        print(f"Failed: {e}")
