#!/usr/bin/env python3
"""
Rigorous E75 100M Benchmark - Step-based, Sequential, Reproducible

Key differences from time-based benchmarks:
1. STEP-BASED: Train for exactly N steps (not time-limited)
2. SEQUENTIAL: One model at a time per GPU (no contention)
3. WARMUP EXCLUDED: 100 warmup steps before timing starts
4. REPRODUCIBLE: Same seed, same data order, deterministic
5. VERIFIED: Print actual param count and config at start

Usage:
    python run_rigorous_benchmark.py           # Run all models sequentially on GPU 0
    python run_rigorous_benchmark.py --gpu 0   # Run on specific GPU
    python run_rigorous_benchmark.py --parallel # Run in parallel (8 GPUs, one model each)
"""

import os
import subprocess
import sys
import time
from datetime import datetime

# Target: 3000 training steps (after 100 warmup steps)
TRAIN_STEPS = 3000
WARMUP_STEPS = 100  # Excluded from loss measurement

# E75 100M configurations - all ~100M params, depth=20, expansion=1.0
MODELS = [
    # (name, dim, n_state, n_heads, description)
    ('mamba2', 896, None, None, 'Mamba2 SSM baseline (102M)'),
    ('fla-gdn', 768, None, None, 'FLA GatedDeltaNet baseline'),
    ('E75h4n16', 2048, 16, 4, '4 heads, n=16 (98M)'),
    ('E75h4n24', 2048, 24, 4, '4 heads, n=24 (104M)'),
    ('E75h4n32', 1920, 32, 4, '4 heads, n=32 (99M)'),
    ('E75h8n16', 1920, 16, 8, '8 heads, n=16 (99M)'),
    ('E75h8n24', 1792, 24, 8, '8 heads, n=24 (99M)'),
]

# Fixed hyperparameters for reproducibility
COMMON_ARGS = [
    '--data', 'data/pile.txt',
    '--depth', '20',
    '--batch_size', '32',
    '--chunk_size', '512',
    '--lr', '3e-4',
    '--warmup_steps', str(WARMUP_STEPS),
    '--seed', '42',
    '--expansion', '1.0',
    '--steps', str(TRAIN_STEPS),  # Fixed step count (not time-based)
    '--bf16',
    '--val_every', '500',  # Eval every 500 steps
    '--log_every', '10',
]


def run_single_model(name, dim, n_state, n_heads, desc, output_dir, gpu):
    """Run a single model and return results."""
    print(f"\n{'='*70}")
    print(f"Running: {name} ({desc})")
    print(f"GPU: {gpu}, dim={dim}, n_state={n_state}, n_heads={n_heads}")
    print(f"{'='*70}")

    cmd = [
        'python', 'train.py',
        '--level', name,
        '--dim', str(dim),
        '--output', f'{output_dir}/{name}',
    ] + COMMON_ARGS

    if n_state is not None:
        cmd.extend(['--n_state', str(n_state)])

    log_file = f'{output_dir}/{name}.log'

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)

    start_time = time.time()

    with open(log_file, 'w') as log:
        result = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )

    elapsed = time.time() - start_time

    # Extract results from log
    try:
        with open(log_file, 'r') as f:
            content = f.read()

        # Get params
        import re
        params_match = re.search(r'([\d,]+)\s*parameters', content)
        params = int(params_match.group(1).replace(',', '')) if params_match else 0

        # Get final loss (from last step line)
        loss_matches = re.findall(r'step\s+\d+\s+\|\s+loss\s+([\d.]+)', content)
        final_loss = float(loss_matches[-1]) if loss_matches else None

        # Get avg throughput (exclude first 10 lines for compilation)
        toks_matches = re.findall(r'tok/s\s+(\d+)', content)
        if len(toks_matches) > 10:
            avg_toks = sum(int(t) for t in toks_matches[10:]) / len(toks_matches[10:])
        else:
            avg_toks = sum(int(t) for t in toks_matches) / len(toks_matches) if toks_matches else 0

        return {
            'name': name,
            'params': params,
            'loss': final_loss,
            'throughput': avg_toks,
            'elapsed': elapsed,
            'status': 'OK' if result.returncode == 0 else 'FAILED',
        }
    except Exception as e:
        return {
            'name': name,
            'params': 0,
            'loss': None,
            'throughput': 0,
            'elapsed': elapsed,
            'status': f'ERROR: {e}',
        }


def run_sequential(output_dir, gpu=0):
    """Run all models sequentially on one GPU."""
    results = []
    for name, dim, n_state, n_heads, desc in MODELS:
        result = run_single_model(name, dim, n_state, n_heads, desc, output_dir, gpu)
        results.append(result)
        print(f"  {name}: loss={result['loss']}, throughput={result['throughput']:.0f} tok/s")
    return results


def run_parallel(output_dir):
    """Run models in parallel, one per GPU."""
    import multiprocessing as mp

    def worker(args):
        idx, (name, dim, n_state, n_heads, desc) = args
        gpu = idx % 8  # Assume 8 GPUs
        return run_single_model(name, dim, n_state, n_heads, desc, output_dir, gpu)

    with mp.Pool(min(len(MODELS), 8)) as pool:
        results = pool.map(worker, enumerate(MODELS))

    return results


def print_results(results, output_dir):
    """Print formatted results table."""
    print(f"\n{'='*70}")
    print("BENCHMARK RESULTS")
    print(f"{'='*70}")
    print(f"Steps: {TRAIN_STEPS} (after {WARMUP_STEPS} warmup)")
    print(f"Output: {output_dir}")
    print()

    # Sort by loss
    results_sorted = sorted(results, key=lambda x: x['loss'] if x['loss'] else 999)

    print(f"{'Model':<12} {'Params':>10} {'Loss':>8} {'Throughput':>12} {'Status'}")
    print("-" * 60)
    for r in results_sorted:
        params_str = f"{r['params']/1e6:.1f}M" if r['params'] else "N/A"
        loss_str = f"{r['loss']:.4f}" if r['loss'] else "N/A"
        toks_str = f"{r['throughput']:.0f} tok/s" if r['throughput'] else "N/A"
        print(f"{r['name']:<12} {params_str:>10} {loss_str:>8} {toks_str:>12} {r['status']}")

    # Save to JSON
    import json
    with open(f'{output_dir}/results.json', 'w') as f:
        json.dump(results_sorted, f, indent=2)
    print(f"\nResults saved to {output_dir}/results.json")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use for sequential runs')
    parser.add_argument('--parallel', action='store_true', help='Run in parallel (one model per GPU)')
    parser.add_argument('--models', nargs='+', help='Specific models to run (default: all)')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'benchmark_results/e75_rigorous_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    print(f"Rigorous E75 100M Benchmark")
    print(f"{'='*70}")
    print(f"Training steps: {TRAIN_STEPS} (+ {WARMUP_STEPS} warmup)")
    print(f"Output: {output_dir}")
    print(f"Mode: {'parallel' if args.parallel else f'sequential (GPU {args.gpu})'}")

    # Filter models if specified
    global MODELS
    if args.models:
        MODELS = [m for m in MODELS if m[0] in args.models]
        print(f"Models: {[m[0] for m in MODELS]}")

    if args.parallel:
        results = run_parallel(output_dir)
    else:
        results = run_sequential(output_dir, args.gpu)

    print_results(results, output_dir)


if __name__ == '__main__':
    main()
