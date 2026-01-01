#!/usr/bin/env python3
"""
Benchmark all Elman Ladder levels across 8 GPUs.

Runs 1k training steps for each level and collects metrics:
- Final loss
- Training throughput (tokens/sec)
- Max gradient norm
- Peak memory usage

Usage:
    python benchmark_all_levels.py --data data/tinystories_50mb.txt --steps 1000
"""

import os
import sys
import subprocess
import time
import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# All levels to benchmark
ALL_LEVELS = [
    # Linear-space levels (0-6)
    0, 1, 2, 3, 4, 5, 6,
    # Log-space levels
    'log_0', 'log_1', 'log_2', 'log_3', 'log_4', 'log_5',
]


def run_training(gpu_id, level, data_path, steps, output_base, params='100m'):
    """Run training for a single level on a specific GPU."""
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    log_name = f"level_{level}"
    log_file = output_base / f"{log_name}.log"

    cmd = [
        sys.executable, 'train.py',
        '--data', str(data_path),
        '--level', str(level),
        '--params', params,
        '--steps', str(steps),
        '--batch_size', '16',
        '--chunk_size', '512',
        '--lr', '3e-4',
        '--warmup_steps', '100',
        '--log_every', '10',
        '--save_every', '999999',  # Don't save checkpoints
        '--bf16',
        '--output', str(output_base / log_name),
    ]

    print(f"[GPU {gpu_id}] Starting level {level}")
    start_time = time.time()

    with open(log_file, 'w') as f:
        result = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            cwd='/home/erikg/elman',
        )

    elapsed = time.time() - start_time

    # Parse log file for metrics
    metrics = parse_log(log_file, level, elapsed)

    status = "✓" if result.returncode == 0 else "✗"
    print(f"[GPU {gpu_id}] {status} Level {level}: {elapsed:.1f}s, loss={metrics.get('final_loss', 'N/A')}, "
          f"tok/s={metrics.get('tokens_per_sec', 'N/A')}")

    return level, metrics


def parse_log(log_file, level, elapsed):
    """Parse training log file for metrics."""
    metrics = {
        'level': level,
        'elapsed_seconds': elapsed,
        'final_loss': None,
        'tokens_per_sec': None,
        'max_grad_norm': 0,
        'error': None,
    }

    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()

        # Look for the last loss value and throughput
        for line in reversed(lines):
            if 'loss=' in line.lower() and metrics['final_loss'] is None:
                # Parse: "Step 1000 | loss=5.3421 | ..."
                parts = line.split('|')
                for part in parts:
                    if 'loss=' in part.lower():
                        try:
                            metrics['final_loss'] = float(part.split('=')[1].strip().split()[0])
                        except:
                            pass
                    if 'tok/s=' in part.lower() or 'tokens/s' in part.lower():
                        try:
                            val = part.split('=')[1].strip().split()[0].replace('k', '000').replace('K', '000')
                            metrics['tokens_per_sec'] = float(val)
                        except:
                            pass
                    if 'grad=' in part.lower() or 'grad_norm=' in part.lower():
                        try:
                            val = float(part.split('=')[1].strip().split()[0])
                            metrics['max_grad_norm'] = max(metrics['max_grad_norm'], val)
                        except:
                            pass

            # Check for errors
            if 'error' in line.lower() or 'exception' in line.lower() or 'traceback' in line.lower():
                if metrics['error'] is None:
                    metrics['error'] = line.strip()

        # If we didn't find tok/s in the log, calculate from elapsed time
        if metrics['tokens_per_sec'] is None and metrics['final_loss'] is not None:
            # Rough estimate: steps * batch_size * chunk_size / elapsed
            estimated_tokens = 1000 * 16 * 512
            metrics['tokens_per_sec'] = estimated_tokens / elapsed

    except Exception as e:
        metrics['error'] = str(e)

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Benchmark all Elman Ladder levels')
    parser.add_argument('--data', type=str, default='data/pile_1mb.txt',
                        help='Path to training data')
    parser.add_argument('--steps', type=int, default=1000,
                        help='Training steps per level')
    parser.add_argument('--params', type=str, default='100m',
                        help='Model size (e.g., 100m, 500m)')
    parser.add_argument('--gpus', type=int, default=8,
                        help='Number of GPUs to use')
    parser.add_argument('--levels', type=str, default=None,
                        help='Comma-separated list of levels to run (default: all)')
    parser.add_argument('--output', type=str, default='./outputs/benchmark',
                        help='Output directory')
    args = parser.parse_args()

    # Parse levels
    if args.levels:
        levels = []
        for l in args.levels.split(','):
            l = l.strip()
            if l.startswith('log_'):
                levels.append(l)
            else:
                levels.append(int(l))
    else:
        levels = ALL_LEVELS

    # Setup output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_base = Path(args.output) / f"benchmark_{timestamp}"
    output_base.mkdir(parents=True, exist_ok=True)

    print(f"=" * 60)
    print(f"Elman Ladder Benchmark")
    print(f"=" * 60)
    print(f"Levels: {levels}")
    print(f"Steps: {args.steps}")
    print(f"Params: {args.params}")
    print(f"GPUs: {args.gpus}")
    print(f"Output: {output_base}")
    print(f"=" * 60)

    # Run benchmarks with GPU pool
    all_metrics = {}
    data_path = Path(args.data)

    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=args.gpus) as executor:
        futures = {}
        for i, level in enumerate(levels):
            gpu_id = i % args.gpus
            future = executor.submit(
                run_training, gpu_id, level, data_path, args.steps, output_base, args.params
            )
            futures[future] = level

        for future in as_completed(futures):
            level, metrics = future.result()
            all_metrics[str(level)] = metrics

    total_time = time.time() - start_time

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"BENCHMARK RESULTS ({total_time:.1f}s total)")
    print(f"{'=' * 60}")
    print(f"{'Level':<12} {'Loss':>10} {'Tok/s':>12} {'Max Grad':>10} {'Status':>8}")
    print(f"{'-' * 60}")

    for level in levels:
        m = all_metrics.get(str(level), {})
        loss = f"{m.get('final_loss', 0):.4f}" if m.get('final_loss') else "N/A"
        tok_s = f"{m.get('tokens_per_sec', 0):.0f}" if m.get('tokens_per_sec') else "N/A"
        grad = f"{m.get('max_grad_norm', 0):.2f}" if m.get('max_grad_norm') else "N/A"
        status = "✗ ERR" if m.get('error') else "✓ OK"
        print(f"{str(level):<12} {loss:>10} {tok_s:>12} {grad:>10} {status:>8}")

    # Save results
    results_file = output_base / 'results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'args': vars(args),
            'total_time_seconds': total_time,
            'metrics': all_metrics,
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
