"""
E88 cuBLAS Backward Kernel Benchmark

Compares:
1. Original per-head CUDA backward kernel
2. New cuBLAS tensor core backward kernel (batched across heads)

Run with: python test_e88_cublas_benchmark.py
"""

import torch
import time
import sys

def benchmark_kernel(name, fn, warmup=10, iterations=100):
    """Benchmark a function, return average time in ms."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000 / iterations

    return elapsed


def run_benchmark():
    print("E88 cuBLAS Backward Kernel Benchmark")
    print("=" * 60)

    try:
        import hasty_pytorch_lib
        from elman.models import e88_fla_hybrid

        if not hasattr(hasty_pytorch_lib, 'e88_fla_hybrid_forward'):
            print("E88 CUDA kernel not available")
            return

        has_cublas = hasattr(hasty_pytorch_lib, 'e88_fla_hybrid_backward_cublas')
        print(f"cuBLAS backward available: {has_cublas}")

    except ImportError as e:
        print(f"Import error: {e}")
        return

    torch.manual_seed(42)

    # Test configurations
    configs = [
        # (T, B, H, n_state, head_v_dim)
        ("Small", 128, 8, 8, 32, 64),
        ("Medium", 256, 16, 16, 32, 64),
        ("Large", 512, 32, 32, 32, 128),
        ("100M-like", 512, 32, 16, 32, 128),  # Typical 100M model config
    ]

    print("\nBenchmark Results:")
    print("-" * 60)
    print(f"{'Config':<15} {'Original (ms)':<15} {'cuBLAS (ms)':<15} {'Speedup':<10}")
    print("-" * 60)

    for config_name, T, B, H, n_state, head_v_dim in configs:
        # Check if this config is supported
        if n_state not in [32, 48, 64, 72] or head_v_dim not in [64, 72, 96, 128]:
            print(f"{config_name:<15} Unsupported n_state/head_v_dim combo")
            continue

        # Generate test data
        k = torch.randn(T, B, H, n_state, device='cuda', dtype=torch.bfloat16)
        v = torch.randn(T, B, H, head_v_dim, device='cuda', dtype=torch.bfloat16)
        q = torch.randn(T, B, H, n_state, device='cuda', dtype=torch.bfloat16)
        decay = torch.sigmoid(torch.randn(T, B, H, device='cuda', dtype=torch.bfloat16))
        S0 = torch.zeros(B, H, n_state, head_v_dim, device='cuda', dtype=torch.bfloat16)

        # L2 normalize k and q
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)

        # Forward to get outputs and cache
        k.requires_grad_(True)
        v.requires_grad_(True)
        q.requires_grad_(True)
        decay.requires_grad_(True)

        try:
            results = hasty_pytorch_lib.e88_fla_hybrid_forward(
                True, k, v, q, decay, S0, H
            )
            output = results[1]
            S_cache = results[2]

            d_output = torch.randn_like(output)

            # Compute checkpoint parameters
            checkpoint_interval = 32
            num_checkpoints = (T + checkpoint_interval - 1) // checkpoint_interval + 1
            checkpoints_size = num_checkpoints * B * H * n_state * head_v_dim

            S_checkpoints = S_cache[:checkpoints_size].view(num_checkpoints, B, H, n_state, head_v_dim)
            Sq_cache = S_cache[checkpoints_size:].view(T, B, H, head_v_dim)

        except Exception as e:
            print(f"{config_name:<15} Forward failed: {e}")
            continue

        # Benchmark original backward
        def original_backward():
            hasty_pytorch_lib.e88_fla_hybrid_backward(
                k, v, q, decay,
                S_checkpoints, Sq_cache,
                d_output, H
            )

        try:
            original_time = benchmark_kernel("original", original_backward, warmup=5, iterations=50)
        except Exception as e:
            original_time = float('inf')
            print(f"{config_name:<15} Original failed: {e}")
            continue

        # Benchmark cuBLAS backward
        if has_cublas:
            def cublas_backward():
                hasty_pytorch_lib.e88_fla_hybrid_backward_cublas(
                    k, v, q, decay,
                    S_checkpoints.view(-1),
                    d_output,
                    H,
                    checkpoint_interval
                )

            try:
                cublas_time = benchmark_kernel("cublas", cublas_backward, warmup=5, iterations=50)
                speedup = original_time / cublas_time
            except Exception as e:
                cublas_time = float('inf')
                speedup = 0.0
                print(f"{config_name:<15} cuBLAS failed: {e}")
                continue
        else:
            cublas_time = float('inf')
            speedup = 0.0

        print(f"{config_name:<15} {original_time:<15.2f} {cublas_time:<15.2f} {speedup:<10.2f}x")

        # Clear GPU memory
        del k, v, q, decay, S0, output, S_cache, d_output, S_checkpoints, Sq_cache
        torch.cuda.empty_cache()

    print("-" * 60)


def test_correctness():
    """Quick correctness check."""
    print("\nCorrectness Check:")
    print("-" * 60)

    try:
        import hasty_pytorch_lib
    except ImportError:
        print("hasty_pytorch_lib not available")
        return False

    if not hasattr(hasty_pytorch_lib, 'e88_fla_hybrid_backward_cublas'):
        print("cuBLAS backward not available")
        return False

    torch.manual_seed(42)

    T, B, H = 64, 4, 8
    n_state, head_v_dim = 32, 64
    checkpoint_interval = 32

    k = torch.randn(T, B, H, n_state, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(T, B, H, head_v_dim, device='cuda', dtype=torch.bfloat16)
    q = torch.randn(T, B, H, n_state, device='cuda', dtype=torch.bfloat16)
    decay = torch.sigmoid(torch.randn(T, B, H, device='cuda', dtype=torch.bfloat16))
    S0 = torch.zeros(B, H, n_state, head_v_dim, device='cuda', dtype=torch.bfloat16)

    k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)

    # Forward
    results = hasty_pytorch_lib.e88_fla_hybrid_forward(True, k, v, q, decay, S0, H)
    output = results[1]
    S_cache = results[2]

    d_output = torch.randn_like(output)

    num_checkpoints = (T + checkpoint_interval - 1) // checkpoint_interval + 1
    checkpoints_size = num_checkpoints * B * H * n_state * head_v_dim

    S_checkpoints = S_cache[:checkpoints_size].view(num_checkpoints, B, H, n_state, head_v_dim)
    Sq_cache = S_cache[checkpoints_size:].view(T, B, H, head_v_dim)

    # Original backward
    grads_orig = hasty_pytorch_lib.e88_fla_hybrid_backward(
        k, v, q, decay, S_checkpoints, Sq_cache, d_output, H
    )

    # cuBLAS backward
    grads_cublas = hasty_pytorch_lib.e88_fla_hybrid_backward_cublas(
        k, v, q, decay, S_checkpoints.view(-1), d_output, H, checkpoint_interval
    )

    # Compare
    all_close = True
    for i, name in enumerate(['d_k', 'd_v', 'd_q', 'd_decay']):
        diff = (grads_orig[i].float() - grads_cublas[i].float()).abs()
        max_diff = diff.max().item()
        rel_err = (diff / (grads_orig[i].float().abs() + 1e-6)).mean().item()

        status = "PASS" if rel_err < 0.1 else "FAIL"
        print(f"  {name}: max_diff={max_diff:.4f}, rel_err={rel_err:.4f} [{status}]")

        if rel_err >= 0.1:
            all_close = False

    return all_close


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--correctness":
        test_correctness()
    else:
        test_correctness()
        print()
        run_benchmark()
