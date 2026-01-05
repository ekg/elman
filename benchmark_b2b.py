#!/usr/bin/env python3
"""
Benchmark E5 B2B GEMM kernel vs fused kernel.

Compares:
- E5 Fused (PureLowRankElman with fused=True) - baseline
- E5 B2B (B2bLowRankElman) - CUTLASS B2B GEMM fusion candidate

Note: The B2B version requires rank = 64, 128, or 256.
"""

import argparse
import time
import torch
import sys
sys.path.insert(0, '/home/erikg/elman')

import hasty_pytorch_lib


def benchmark_kernel(forward_fn, backward_fn, x, h0, params, warmup=10, iters=50):
    """Benchmark forward + backward pass."""
    U_h, V_h, U_x, V_x, U_z, V_z, b = params

    # Warmup
    for _ in range(warmup):
        h, output, v = forward_fn(True, x, h0, U_h, V_h, U_x, V_x, U_z, V_z, b)
        d_output = torch.randn_like(output)
        grads = backward_fn(U_h, V_h, U_x, V_x, U_z, V_z, x, h, v, d_output)

    torch.cuda.synchronize()

    # Timed runs
    t0 = time.perf_counter()
    for _ in range(iters):
        h, output, v = forward_fn(True, x, h0, U_h, V_h, U_x, V_x, U_z, V_z, b)
        d_output = torch.randn_like(output)
        grads = backward_fn(U_h, V_h, U_x, V_x, U_z, V_z, x, h, v, d_output)
    torch.cuda.synchronize()

    elapsed = (time.perf_counter() - t0) / iters * 1000  # ms per iteration
    tokens = x.shape[0] * x.shape[1]  # seq_len * batch
    tok_per_sec = tokens / (elapsed / 1000)

    return elapsed, tok_per_sec


def benchmark_inference(forward_fn, x, h0, params, warmup=10, iters=50):
    """Benchmark inference (forward only)."""
    U_h, V_h, U_x, V_x, U_z, V_z, b = params

    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            h, output, v = forward_fn(False, x, h0, U_h, V_h, U_x, V_x, U_z, V_z, b)

        torch.cuda.synchronize()

        # Timed runs
        t0 = time.perf_counter()
        for _ in range(iters):
            h, output, v = forward_fn(False, x, h0, U_h, V_h, U_x, V_x, U_z, V_z, b)
        torch.cuda.synchronize()

    elapsed = (time.perf_counter() - t0) / iters * 1000
    tokens = x.shape[0] * x.shape[1]
    tok_per_sec = tokens / (elapsed / 1000)

    return elapsed, tok_per_sec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--dim', type=int, default=1536)
    parser.add_argument('--rank', type=int, default=256,
                        help='Must be 64, 128, or 256 for B2B kernel')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iters', type=int, default=50)
    args = parser.parse_args()

    # Validate rank
    if args.rank not in [64, 128, 256]:
        print(f"Warning: rank={args.rank} is not compatible with B2B kernel (requires 64, 128, or 256)")
        print("Adjusting rank to 256 for comparison")
        args.rank = 256

    device = 'cuda'
    dtype = torch.bfloat16

    print(f"E5 B2B GEMM Benchmark")
    print("=" * 70)
    print(f"Batch: {args.batch}, Seq: {args.seq_len}, Dim: {args.dim}, Rank: {args.rank}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iters}")
    print()

    # Create inputs
    x = torch.randn(args.seq_len, args.batch, args.dim, device=device, dtype=dtype)
    h0 = torch.zeros(args.batch, args.dim, device=device, dtype=dtype)

    # Create parameters
    U_h = torch.randn(args.dim, args.rank, device=device, dtype=dtype) * 0.1
    V_h = torch.randn(args.rank, args.dim, device=device, dtype=dtype) * 0.1
    U_x = torch.randn(args.dim, args.rank, device=device, dtype=dtype) * 0.1
    V_x = torch.randn(args.rank, args.dim, device=device, dtype=dtype) * 0.1
    U_z = torch.randn(args.dim, args.rank, device=device, dtype=dtype) * 0.1
    V_z = torch.randn(args.rank, args.dim, device=device, dtype=dtype) * 0.1
    b = torch.zeros(args.dim, device=device, dtype=dtype)
    params = (U_h, V_h, U_x, V_x, U_z, V_z, b)

    # Count params
    param_count = (
        args.dim * args.rank * 2 +  # U_h, V_h
        args.dim * args.rank * 2 +  # U_x, V_x
        args.dim * args.rank * 2 +  # U_z, V_z
        args.dim                     # b
    )
    print(f"Parameters per layer: {param_count:,}")
    print()

    # Verify correctness
    print("Verifying correctness...")
    with torch.no_grad():
        h_fused, out_fused, _ = hasty_pytorch_lib.pure_lowrank_elman_forward_fused(
            False, x, h0, U_h, V_h, U_x, V_x, U_z, V_z, b)
        h_b2b, out_b2b, _ = hasty_pytorch_lib.b2b_lowrank_elman_forward(
            False, x, h0, U_h, V_h, U_x, V_x, U_z, V_z, b)

    diff = (out_fused - out_b2b).abs().max().item()
    print(f"  Max diff: {diff:.2e}")
    if diff > 1e-2:
        print("  WARNING: Outputs differ significantly!")
    else:
        print("  Outputs match!")
    print()

    # ==================== INFERENCE BENCHMARKS ====================
    print("=" * 70)
    print("INFERENCE BENCHMARKS (forward only)")
    print("=" * 70)
    print()

    # Benchmark fused
    print("Benchmarking E5 Fused...")
    time_fused_inf, tok_fused_inf = benchmark_inference(
        hasty_pytorch_lib.pure_lowrank_elman_forward_fused, x, h0, params,
        args.warmup, args.iters)
    print(f"  Time: {time_fused_inf:.2f} ms/batch")
    print(f"  Throughput: {tok_fused_inf/1e3:.1f}k tok/s")
    print()

    # Benchmark B2B
    print("Benchmarking E5 B2B...")
    time_b2b_inf, tok_b2b_inf = benchmark_inference(
        hasty_pytorch_lib.b2b_lowrank_elman_forward, x, h0, params,
        args.warmup, args.iters)
    print(f"  Time: {time_b2b_inf:.2f} ms/batch")
    print(f"  Throughput: {tok_b2b_inf/1e3:.1f}k tok/s")
    print()

    speedup_inf = time_fused_inf / time_b2b_inf
    print(f"Inference Speedup: {speedup_inf:.2f}x")
    print()

    # ==================== TRAINING BENCHMARKS ====================
    print("=" * 70)
    print("TRAINING BENCHMARKS (forward + backward)")
    print("=" * 70)
    print()

    # Benchmark fused training
    print("Benchmarking E5 Fused...")
    time_fused, tok_fused = benchmark_kernel(
        hasty_pytorch_lib.pure_lowrank_elman_forward_fused,
        hasty_pytorch_lib.pure_lowrank_elman_backward_fused,
        x, h0, params, args.warmup, args.iters)
    print(f"  Time: {time_fused:.2f} ms/batch")
    print(f"  Throughput: {tok_fused/1e3:.1f}k tok/s")
    print()

    # Benchmark B2B training
    print("Benchmarking E5 B2B...")
    time_b2b, tok_b2b = benchmark_kernel(
        hasty_pytorch_lib.b2b_lowrank_elman_forward,
        hasty_pytorch_lib.b2b_lowrank_elman_backward,
        x, h0, params, args.warmup, args.iters)
    print(f"  Time: {time_b2b:.2f} ms/batch")
    print(f"  Throughput: {tok_b2b/1e3:.1f}k tok/s")
    print()

    speedup = time_fused / time_b2b
    print(f"Training Speedup: {speedup:.2f}x")
    print()

    # ==================== SUMMARY ====================
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"E5 Fused:     {tok_fused/1e3:.1f}k tok/s (training)")
    print(f"E5 B2B:       {tok_b2b/1e3:.1f}k tok/s (training) [{speedup:.2f}x]")
    print()
    print("Note: The B2B kernel currently uses cuBLAS sequential GEMMs as a baseline.")
    print("Full CUTLASS B2B fusion would fuse V_h@h and U_h@result into a single kernel,")
    print("keeping the intermediate result in shared memory instead of global memory.")
    print("Expected additional speedup from CUTLASS fusion: ~20-40%")


if __name__ == '__main__':
    main()
