"""
Benchmark E37 optimization: Compare original E37 vs E37v2 (optimized) vs E33

E37 (original): W @ (x + h_prev) - sequential GEMM, can't batch
E37v2 (optimized): W @ x + W @ h_prev - batched GEMM for W @ x (like E33)
E33: W_x @ x + W_h @ h_prev - batched GEMM for W_x @ x

Expected results:
- E37 (original): ~92K tok/s (slow due to sequential GEMM)
- E37v2 (optimized): ~140K tok/s (matching E33)
- E33: ~140K tok/s (baseline)
"""

import torch
import torch.nn as nn
import time
import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

# Import models
from elman.models.e37_tied_weights import E37TiedWeights
from elman.models.e37_tied_weights_v2 import E37TiedWeightsV2
from elman.models.e33_self_gate import E33SelfGate


def benchmark_model(model, batch_size, seq_len, dim, num_iters=100, warmup=10):
    """Benchmark a model and return throughput in tokens/second."""
    device = 'cuda'
    dtype = torch.bfloat16

    model = model.to(device).to(dtype)
    model.train()

    # Create dummy data
    x = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype)
    target = torch.randint(0, 256, (batch_size, seq_len), device=device)

    # Warmup
    for _ in range(warmup):
        out, _ = model(x)
        loss = out.sum()
        loss.backward()
        model.zero_grad()

    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(num_iters):
        out, _ = model(x)
        loss = out.sum()
        loss.backward()
        model.zero_grad()
    torch.cuda.synchronize()
    elapsed = time.time() - start

    total_tokens = batch_size * seq_len * num_iters
    tokens_per_sec = total_tokens / elapsed

    return tokens_per_sec


def main():
    parser = argparse.ArgumentParser(description='Benchmark E37 optimization')
    parser.add_argument('--dim', type=int, default=1024, help='Model dimension')
    parser.add_argument('--expansion', type=float, default=2.0, help='Expansion factor')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--seq-len', type=int, default=512, help='Sequence length')
    parser.add_argument('--num-iters', type=int, default=100, help='Number of iterations')
    args = parser.parse_args()

    print(f"Benchmarking E37 optimization")
    print(f"dim={args.dim}, expansion={args.expansion}, batch={args.batch_size}, seq_len={args.seq_len}")
    print("=" * 70)

    # Calculate d_inner
    d_inner = int(args.dim * args.expansion)

    # E37 (original) - slow due to W @ (x + h_prev)
    print("\n[1/3] Benchmarking E37 (original - W @ (x + h_prev))...")
    model_e37 = E37TiedWeights(
        dim=args.dim,
        expansion=args.expansion,
        use_conv=False,
    )
    params_e37 = sum(p.numel() for p in model_e37.parameters())
    tps_e37 = benchmark_model(model_e37, args.batch_size, args.seq_len, args.dim, args.num_iters)
    print(f"  E37 (original): {tps_e37 / 1000:.1f}K tok/s, {params_e37:,} params")

    # E37v2 (optimized) - fast due to batched W @ x
    print("\n[2/3] Benchmarking E37v2 (optimized - W @ x + W @ h_prev)...")
    model_e37v2 = E37TiedWeightsV2(
        dim=args.dim,
        expansion=args.expansion,
        use_conv=False,
    )
    params_e37v2 = sum(p.numel() for p in model_e37v2.parameters())
    tps_e37v2 = benchmark_model(model_e37v2, args.batch_size, args.seq_len, args.dim, args.num_iters)
    print(f"  E37v2 (optimized): {tps_e37v2 / 1000:.1f}K tok/s, {params_e37v2:,} params")

    # E33 (baseline) - fast due to batched W_x @ x
    print("\n[3/3] Benchmarking E33 (baseline - W_x @ x + W_h @ h_prev)...")
    model_e33 = E33SelfGate(
        dim=args.dim,
        expansion=args.expansion,
        use_conv=False,
    )
    params_e33 = sum(p.numel() for p in model_e33.parameters())
    tps_e33 = benchmark_model(model_e33, args.batch_size, args.seq_len, args.dim, args.num_iters)
    print(f"  E33 (baseline): {tps_e33 / 1000:.1f}K tok/s, {params_e33:,} params")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<25} {'Throughput':>15} {'Params':>15} {'Speedup vs E37':>15}")
    print("-" * 70)
    print(f"{'E37 (original)':<25} {tps_e37/1000:>12.1f}K {params_e37:>15,} {'1.00x':>15}")
    print(f"{'E37v2 (optimized)':<25} {tps_e37v2/1000:>12.1f}K {params_e37v2:>15,} {tps_e37v2/tps_e37:>14.2f}x")
    print(f"{'E33 (baseline)':<25} {tps_e33/1000:>12.1f}K {params_e33:>15,} {tps_e33/tps_e37:>14.2f}x")
    print("-" * 70)

    # Check if optimization worked
    speedup = tps_e37v2 / tps_e37
    if speedup > 1.3:
        print(f"\nSUCCESS! E37v2 is {speedup:.2f}x faster than E37 (original)")
        print(f"E37v2 achieves {tps_e37v2/tps_e33*100:.1f}% of E33's throughput")
    else:
        print(f"\nWARNING: E37v2 only achieved {speedup:.2f}x speedup (expected >1.3x)")


if __name__ == "__main__":
    main()
