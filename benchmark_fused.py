#!/usr/bin/env python3
"""
Benchmark E5 fused vs non-fused kernel performance + CUDA Graphs.

Compares:
- E5 (PureLowRankElman) with fused=False (original: 4 kernel launches per timestep)
- E5 (PureLowRankElman) with fused=True (optimized: 3 kernel launches per timestep)
- E5 (PureLowRankElman) with fused=True + CUDA Graphs (inference only)

Expected speedup: ~25% reduction in kernel launch overhead for the sequential loop.
"""

import argparse
import time
import torch
import sys
sys.path.insert(0, '/home/erikg/elman')

from elman.models.pure_lowrank_elman import PureLowRankElman


def benchmark_training(model, x, warmup=10, iters=50):
    """Benchmark forward + backward pass."""
    model.train()
    # Warmup
    for _ in range(warmup):
        out, h = model(x)
        loss = out.mean()
        loss.backward()
        model.zero_grad()

    torch.cuda.synchronize()

    # Timed runs
    t0 = time.perf_counter()
    for _ in range(iters):
        out, h = model(x)
        loss = out.mean()
        loss.backward()
        model.zero_grad()
    torch.cuda.synchronize()

    elapsed = (time.perf_counter() - t0) / iters * 1000  # ms per iteration
    tokens = x.shape[0] * x.shape[1]  # batch * seq_len
    tok_per_sec = tokens / (elapsed / 1000)

    return elapsed, tok_per_sec


def benchmark_inference(model, x, warmup=10, iters=50):
    """Benchmark inference (forward only)."""
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            out, h = model(x)

        torch.cuda.synchronize()

        # Timed runs
        t0 = time.perf_counter()
        for _ in range(iters):
            out, h = model(x)
        torch.cuda.synchronize()

    elapsed = (time.perf_counter() - t0) / iters * 1000  # ms per iteration
    tokens = x.shape[0] * x.shape[1]  # batch * seq_len
    tok_per_sec = tokens / (elapsed / 1000)

    return elapsed, tok_per_sec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--dim', type=int, default=1536)
    parser.add_argument('--rank', type=int, default=270)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--inference-only', action='store_true', help='Only run inference benchmarks')
    args = parser.parse_args()

    device = 'cuda'
    dtype = torch.bfloat16

    print(f"E5 Kernel Optimization Benchmark")
    print("=" * 70)
    print(f"Batch: {args.batch}, Seq: {args.seq_len}, Dim: {args.dim}, Rank: {args.rank}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iters}")
    print()

    # Create models with different optimizations
    model_original = PureLowRankElman(dim=args.dim, rank=args.rank, use_fused=False, use_cuda_graph=False).to(device).to(dtype)
    model_fused = PureLowRankElman(dim=args.dim, rank=args.rank, use_fused=True, use_cuda_graph=False).to(device).to(dtype)
    model_graph = PureLowRankElman(dim=args.dim, rank=args.rank, use_fused=True, use_cuda_graph=True).to(device).to(dtype)

    # Copy weights to ensure identical computation
    model_fused.load_state_dict(model_original.state_dict())
    model_graph.load_state_dict(model_original.state_dict())

    # Count params
    params = sum(p.numel() for p in model_original.parameters())
    print(f"Parameters per layer: {params:,}")
    print()

    # Create input
    x = torch.randn(args.batch, args.seq_len, args.dim, device=device, dtype=dtype)

    # Verify correctness
    print("Verifying correctness...")
    model_original.eval()
    model_fused.eval()
    model_graph.eval()
    with torch.no_grad():
        out_orig, h_orig = model_original(x)
        out_fused, h_fused = model_fused(x)
        out_graph, h_graph = model_graph(x)

    diff_fused = (out_orig - out_fused).abs().max().item()
    diff_graph = (out_orig - out_graph).abs().max().item()
    print(f"  Fused vs Original:     {diff_fused:.2e}")
    print(f"  CUDA Graph vs Original: {diff_graph:.2e}")
    if diff_fused > 1e-3 or diff_graph > 1e-3:
        print("  WARNING: Outputs differ significantly!")
    else:
        print("  All outputs match!")
    print()

    # ==================== INFERENCE BENCHMARKS ====================
    print("=" * 70)
    print("INFERENCE BENCHMARKS (forward only)")
    print("=" * 70)
    print()

    # Benchmark original inference
    print("Benchmarking E5 Original (4 kernels/step)...")
    time_orig_inf, tok_orig_inf = benchmark_inference(model_original, x, args.warmup, args.iters)
    print(f"  Time: {time_orig_inf:.2f} ms/batch")
    print(f"  Throughput: {tok_orig_inf/1e3:.1f}k tok/s")
    print()

    # Benchmark fused inference
    print("Benchmarking E5 Fused (3 kernels/step)...")
    time_fused_inf, tok_fused_inf = benchmark_inference(model_fused, x, args.warmup, args.iters)
    print(f"  Time: {time_fused_inf:.2f} ms/batch")
    print(f"  Throughput: {tok_fused_inf/1e3:.1f}k tok/s")
    print()

    # Benchmark CUDA Graph inference
    print("Benchmarking E5 Fused + CUDA Graph...")
    time_graph_inf, tok_graph_inf = benchmark_inference(model_graph, x, args.warmup, args.iters)
    print(f"  Time: {time_graph_inf:.2f} ms/batch")
    print(f"  Throughput: {tok_graph_inf/1e3:.1f}k tok/s")
    print()

    # Inference summary
    print("-" * 70)
    print("INFERENCE RESULTS:")
    speedup_fused_inf = time_orig_inf / time_fused_inf
    speedup_graph_inf = time_orig_inf / time_graph_inf
    print(f"  Original:   {tok_orig_inf/1e3:.1f}k tok/s (baseline)")
    print(f"  Fused:      {tok_fused_inf/1e3:.1f}k tok/s ({speedup_fused_inf:.2f}x)")
    print(f"  CUDA Graph: {tok_graph_inf/1e3:.1f}k tok/s ({speedup_graph_inf:.2f}x)")
    print()

    if args.inference_only:
        return

    # ==================== TRAINING BENCHMARKS ====================
    print("=" * 70)
    print("TRAINING BENCHMARKS (forward + backward)")
    print("=" * 70)
    print()

    # Benchmark original training
    print("Benchmarking E5 Original (4 kernels/step)...")
    time_orig, tok_orig = benchmark_training(model_original, x, args.warmup, args.iters)
    print(f"  Time: {time_orig:.2f} ms/batch")
    print(f"  Throughput: {tok_orig/1e3:.1f}k tok/s")
    print()

    # Benchmark fused training
    print("Benchmarking E5 Fused (3 kernels/step)...")
    time_fused, tok_fused = benchmark_training(model_fused, x, args.warmup, args.iters)
    print(f"  Time: {time_fused:.2f} ms/batch")
    print(f"  Throughput: {tok_fused/1e3:.1f}k tok/s")
    print()

    # Training summary
    print("-" * 70)
    print("TRAINING RESULTS:")
    speedup_fused = time_orig / time_fused
    print(f"  Original: {tok_orig/1e3:.1f}k tok/s (baseline)")
    print(f"  Fused:    {tok_fused/1e3:.1f}k tok/s ({speedup_fused:.2f}x)")
    print()
    print("Note: CUDA Graphs not supported for training (autograd needs dynamic graph)")
    print()

    # ==================== OVERALL SUMMARY ====================
    print("=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print()
    print("Optimizations Applied:")
    print("  1. Fused Kernel: Merges tanh + gate into single kernel (4->3 ops/step)")
    print("  2. CUDA Graph: Captures entire forward pass, replays with single launch")
    print()
    print(f"Inference Speedup:  {speedup_graph_inf:.2f}x (CUDA Graph)")
    print(f"Training Speedup:   {speedup_fused:.2f}x (Fused Kernel)")
    print()
    print("Analysis:")
    print("  - Fused kernel provides modest improvement (~3%) - GEMMs dominate")
    print("  - CUDA Graph can significantly reduce launch overhead for inference")
    print("  - Main bottleneck: Sequential GEMMs in recurrence loop")


if __name__ == '__main__':
    main()
