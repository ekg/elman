#!/usr/bin/env python3
"""
E8 Parameter Scan: Throughput test across rank and hidden dimension space.

Targets 50M parameter models with different (d_inner, rank, depth) configurations.
Runs 100 steps to measure throughput, parallelized across all available GPUs.

Usage:
    python scan_e8_throughput.py              # Run all configs across 8 GPUs
    python scan_e8_throughput.py --config 0   # Run specific config on GPU 0
"""

import argparse
import os
import sys
import time
import math
import json
import subprocess
import multiprocessing as mp
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Must be set before importing torch
os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')


@dataclass
class E8Config:
    """Configuration for E8 model."""
    dim: int
    d_inner: int
    rank: int
    depth: int
    name: str

    @property
    def params_per_layer(self) -> int:
        """Calculate parameters per E8 layer."""
        in_proj = self.dim * 2 * self.d_inner
        out_proj = self.d_inner * self.dim
        U_h = self.d_inner * self.rank
        V_h = self.rank * self.d_inner
        s_h = self.rank
        U_x = self.d_inner * self.rank
        V_x = self.rank * self.d_inner
        s_x = self.rank
        b = self.d_inner
        ln = 2 * self.dim
        return in_proj + out_proj + U_h + V_h + s_h + U_x + V_x + s_x + b + ln

    def total_params(self, vocab_size: int = 256) -> int:
        embed = vocab_size * self.dim
        final_ln = 2 * self.dim
        layers = self.depth * self.params_per_layer
        return embed + final_ln + layers


def find_e8_configs(target_params: int = 50_000_000, vocab_size: int = 256) -> List[E8Config]:
    """Find E8 configurations targeting the given parameter count."""
    configs = []
    dim = 512

    for d_inner in [768, 1024, 1280, 1536, 2048]:
        for rank in [64, 128, 256, 384, 512]:
            if rank > d_inner:
                continue

            config = E8Config(dim=dim, d_inner=d_inner, rank=rank, depth=1, name="")
            params_per_layer = config.params_per_layer

            embed = vocab_size * dim
            final_ln = 2 * dim
            fixed = embed + final_ln

            remaining = target_params - fixed
            if remaining <= 0:
                continue

            depth = max(1, round(remaining / params_per_layer))

            config = E8Config(
                dim=dim,
                d_inner=d_inner,
                rank=rank,
                depth=depth,
                name=f"d{d_inner}_r{rank}_L{depth}"
            )

            total = config.total_params(vocab_size)
            if 0.9 * target_params <= total <= 1.1 * target_params:
                configs.append(config)

    configs.sort(key=lambda c: (c.d_inner, c.rank))
    return configs


def run_single_config(args_tuple):
    """Run a single config on a specific GPU. Called by multiprocessing."""
    config_dict, gpu_id, num_steps, batch_size, seq_len, vocab_size = args_tuple

    # Set GPU for this process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    import torch
    import torch.nn.functional as F

    # Reconstruct config
    config = E8Config(**config_dict)

    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Import here to ensure correct CUDA device
        from elman.models.ladder_lm import LadderLM

        # Create model
        model = LadderLM(
            vocab_size=vocab_size,
            dim=config.dim,
            depth=config.depth,
            level=8,
            expansion=config.d_inner / config.dim,
            rank=config.rank,
        )
        model = model.to(device='cuda', dtype=torch.bfloat16)

        actual_params = sum(p.numel() for p in model.parameters())

        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

        # Warmup
        x = torch.randint(0, vocab_size, (batch_size, seq_len + 1), device='cuda')
        for _ in range(5):
            inp, target = x[:, :-1], x[:, 1:]
            logits, _ = model(inp, return_prev_hiddens=True)
            loss = F.cross_entropy(logits.view(-1, vocab_size), target.reshape(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.synchronize()

        # Benchmark
        tokens_per_step = batch_size * seq_len
        total_tokens = 0
        total_loss = 0.0

        start_time = time.time()
        for step in range(num_steps):
            x = torch.randint(0, vocab_size, (batch_size, seq_len + 1), device='cuda')
            inp, target = x[:, :-1], x[:, 1:]

            logits, _ = model(inp, return_prev_hiddens=True)
            loss = F.cross_entropy(logits.view(-1, vocab_size), target.reshape(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_tokens += tokens_per_step
            total_loss += loss.item()

        torch.cuda.synchronize()
        elapsed = time.time() - start_time

        throughput = total_tokens / elapsed
        avg_loss = total_loss / num_steps
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

        return {
            'config': config.name,
            'gpu': gpu_id,
            'dim': config.dim,
            'd_inner': config.d_inner,
            'rank': config.rank,
            'depth': config.depth,
            'params': actual_params,
            'throughput': throughput,
            'throughput_k': throughput / 1000,
            'avg_loss': avg_loss,
            'elapsed_s': elapsed,
            'memory_mb': memory_mb,
            'ms_per_step': elapsed / num_steps * 1000,
            'status': 'success',
        }

    except Exception as e:
        return {
            'config': config.name,
            'gpu': gpu_id,
            'error': str(e),
            'status': 'error',
        }


def main():
    parser = argparse.ArgumentParser(description="E8 Parameter Scan (Parallel)")
    parser.add_argument("--target", type=str, default="50m", help="Target params")
    parser.add_argument("--steps", type=int, default=100, help="Steps per config")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    parser.add_argument("--vocab_size", type=int, default=256, help="Vocab size")
    parser.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs")
    parser.add_argument("--config", type=int, default=None, help="Run specific config")
    parser.add_argument("--list", action="store_true", help="List configs and exit")
    parser.add_argument("--output", type=str, default="e8_scan_results.json", help="Output file")
    args = parser.parse_args()

    # Parse target params
    target_str = args.target.lower()
    if target_str.endswith('m'):
        target_params = int(float(target_str[:-1]) * 1_000_000)
    elif target_str.endswith('b'):
        target_params = int(float(target_str[:-1]) * 1_000_000_000)
    else:
        target_params = int(target_str)

    # Find configurations
    configs = find_e8_configs(target_params, args.vocab_size)

    print(f"Found {len(configs)} configurations targeting {target_params/1e6:.1f}M params:")
    print("=" * 80)
    print(f"{'Config':<20} {'d_inner':>8} {'rank':>6} {'depth':>6} {'params':>12}")
    print("-" * 80)
    for i, c in enumerate(configs):
        params = c.total_params(args.vocab_size)
        print(f"[{i:2d}] {c.name:<16} {c.d_inner:>8} {c.rank:>6} {c.depth:>6} {params:>12,}")
    print("=" * 80)

    if args.list:
        return

    # Run specific config or all
    if args.config is not None:
        configs = [configs[args.config]]

    print(f"\nRunning {len(configs)} configs across {args.num_gpus} GPUs...")
    print(f"Steps: {args.steps}, Batch: {args.batch_size}, SeqLen: {args.seq_len}")
    print()

    # Prepare tasks: (config_dict, gpu_id, ...)
    tasks = []
    for i, config in enumerate(configs):
        gpu_id = i % args.num_gpus
        config_dict = {
            'dim': config.dim,
            'd_inner': config.d_inner,
            'rank': config.rank,
            'depth': config.depth,
            'name': config.name,
        }
        tasks.append((config_dict, gpu_id, args.steps, args.batch_size, args.seq_len, args.vocab_size))

    # Run in parallel using multiprocessing
    start_time = time.time()
    with mp.Pool(processes=args.num_gpus) as pool:
        results = pool.map(run_single_config, tasks)
    total_time = time.time() - start_time

    # Sort by throughput
    valid_results = [r for r in results if r.get('status') == 'success']
    valid_results.sort(key=lambda r: r['throughput'], reverse=True)

    # Print summary
    print("\n" + "=" * 90)
    print("RESULTS (sorted by throughput)")
    print("=" * 90)
    print(f"{'Config':<20} {'Params':>10} {'Throughput':>12} {'Loss':>8} {'Memory':>10} {'GPU':>5}")
    print("-" * 90)
    for r in valid_results:
        print(f"{r['config']:<20} {r['params']/1e6:>9.1f}M {r['throughput_k']:>11.1f}K "
              f"{r['avg_loss']:>8.3f} {r['memory_mb']:>9.0f}MB {r['gpu']:>5}")
    print("=" * 90)

    # Print errors
    error_results = [r for r in results if r.get('status') == 'error']
    if error_results:
        print(f"\nErrors ({len(error_results)}):")
        for r in error_results:
            print(f"  {r['config']}: {r['error']}")

    # Comparison with E1
    print(f"\nComparison: E1 d1280×6 = 254K tok/s at 50M params")
    if valid_results:
        best = valid_results[0]
        print(f"Best E8: {best['config']} = {best['throughput_k']:.1f}K tok/s "
              f"({best['throughput_k']/254:.2f}× E1)")

    print(f"\nTotal scan time: {total_time:.1f}s")

    # Save results
    with open(args.output, 'w') as f:
        json.dump({
            'target_params': target_params,
            'configs_tested': len(configs),
            'total_time_s': total_time,
            'results': results,
            'best': valid_results[0] if valid_results else None,
        }, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
