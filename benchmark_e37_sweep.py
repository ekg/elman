"""
Sweep benchmark: Compare E37 original vs E37v2 across different configurations.
"""

import torch
import torch.nn as nn
import time
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

from elman.models.e37_tied_weights import E37TiedWeights
from elman.models.e37_tied_weights_v2 import E37TiedWeightsV2
from elman.models.e33_self_gate import E33SelfGate


def benchmark_model(model, batch_size, seq_len, dim, num_iters=50, warmup=5):
    """Benchmark a model and return throughput in tokens/second."""
    device = 'cuda'
    dtype = torch.bfloat16

    model = model.to(device).to(dtype)
    model.train()

    x = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype)

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
    print("E37 Optimization Sweep: Finding when batched GEMM helps")
    print("=" * 90)
    print(f"{'dim':>6} {'batch':>6} {'seq':>6} {'E37 (K/s)':>12} {'E37v2 (K/s)':>12} {'E33 (K/s)':>12} {'v2/orig':>10} {'Winner':>12}")
    print("-" * 90)

    configs = [
        # (dim, batch, seq_len)
        (512, 16, 256),
        (512, 32, 256),
        (512, 32, 512),
        (512, 64, 512),
        (1024, 16, 256),
        (1024, 32, 256),
        (1024, 32, 512),
        (1024, 64, 512),
        (1024, 64, 1024),
        (1280, 32, 512),
        (1280, 64, 512),
        (1280, 64, 1024),
    ]

    results = []

    for dim, batch, seq_len in configs:
        try:
            # E37 original
            model_e37 = E37TiedWeights(dim=dim, expansion=2.0, use_conv=False)
            tps_e37 = benchmark_model(model_e37, batch, seq_len, dim, num_iters=30)
            del model_e37
            torch.cuda.empty_cache()

            # E37v2 optimized
            model_e37v2 = E37TiedWeightsV2(dim=dim, expansion=2.0, use_conv=False)
            tps_e37v2 = benchmark_model(model_e37v2, batch, seq_len, dim, num_iters=30)
            del model_e37v2
            torch.cuda.empty_cache()

            # E33 baseline
            model_e33 = E33SelfGate(dim=dim, expansion=2.0, use_conv=False)
            tps_e33 = benchmark_model(model_e33, batch, seq_len, dim, num_iters=30)
            del model_e33
            torch.cuda.empty_cache()

            ratio = tps_e37v2 / tps_e37
            winner = "E37v2" if ratio > 1.05 else ("E37" if ratio < 0.95 else "tie")

            print(f"{dim:>6} {batch:>6} {seq_len:>6} {tps_e37/1000:>12.1f} {tps_e37v2/1000:>12.1f} {tps_e33/1000:>12.1f} {ratio:>10.2f}x {winner:>12}")

            results.append({
                'dim': dim, 'batch': batch, 'seq': seq_len,
                'e37': tps_e37, 'e37v2': tps_e37v2, 'e33': tps_e33,
                'ratio': ratio, 'winner': winner
            })

        except Exception as e:
            print(f"{dim:>6} {batch:>6} {seq_len:>6} ERROR: {e}")
            torch.cuda.empty_cache()

    print("-" * 90)

    # Summary
    e37v2_wins = sum(1 for r in results if r['winner'] == 'E37v2')
    e37_wins = sum(1 for r in results if r['winner'] == 'E37')
    ties = sum(1 for r in results if r['winner'] == 'tie')

    print(f"\nSUMMARY: E37v2 wins {e37v2_wins}, E37 wins {e37_wins}, ties {ties}")

    # Best speedup
    if results:
        best = max(results, key=lambda r: r['ratio'])
        print(f"Best E37v2 speedup: {best['ratio']:.2f}x at dim={best['dim']}, batch={best['batch']}, seq={best['seq']}")

        worst = min(results, key=lambda r: r['ratio'])
        print(f"Worst E37v2 ratio: {worst['ratio']:.2f}x at dim={worst['dim']}, batch={worst['batch']}, seq={worst['seq']}")


if __name__ == "__main__":
    main()
