#!/usr/bin/env python3
"""
E79 All Variants 10-minute 100M param benchmark.

Tests all E79 bias configurations:
- e79: default (fixed learned bias)
- e79nb: no bias
- e79ib: input-dependent bias

Plus n_state variations for the default config.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# E79 variant configurations - all ~100M params
CONFIGS = {
    # Bias ablation variants (n_state=64 baseline)
    'e79': {'level': 79, 'dim': 2048, 'n_state': 64, 'expansion': 1.0},       # Default: fixed bias
    'e79nb': {'level': '79nb', 'dim': 2048, 'n_state': 64, 'expansion': 1.0}, # No bias
    'e79ib': {'level': '79ib', 'dim': 2048, 'n_state': 64, 'expansion': 1.0}, # Input-dependent bias

    # n_state variations with default bias
    'e79n32': {'level': '79n32', 'dim': 2176, 'n_state': 32, 'expansion': 1.0},
    'e79n48': {'level': '79n48', 'dim': 2048, 'n_state': 48, 'expansion': 1.0},
    'e79n96': {'level': '79n96', 'dim': 2048, 'n_state': 96, 'expansion': 1.0},

    # Input-bias with different n_state
    'e79n32ib': {'level': '79n32ib', 'dim': 2176, 'n_state': 32, 'expansion': 1.0},
}

# Standard benchmark settings
BENCHMARK_SETTINGS = {
    'data_path': 'data/pile.txt',
    'vocab_size': 256,
    'depth': 20,
    'batch_size': 32,
    'chunk_size': 512,
    'lr': 3e-4,
    'warmup_steps': 1000,
    'training_time': 600,  # 10 minutes
    'seed': 42,
    'log_every': 50,
}

def get_available_gpus():
    """Get list of available GPU indices."""
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
        capture_output=True, text=True
    )
    return [int(x.strip()) for x in result.stdout.strip().split('\n') if x.strip()]

def run_benchmark():
    """Run E79 variant benchmarks in parallel on different GPUs."""

    output_dir = Path('benchmark_results/e79_variants_100m')
    output_dir.mkdir(parents=True, exist_ok=True)

    gpus = get_available_gpus()
    print(f"Available GPUs: {gpus}")
    print(f"Running {len(CONFIGS)} E79 variant benchmarks")

    processes = []

    for i, (name, config) in enumerate(CONFIGS.items()):
        gpu = gpus[i % len(gpus)]
        log_file = output_dir / f'{name}.log'

        # Build training command
        cmd = [
            'python', '-u', 'train.py',
            '--data', BENCHMARK_SETTINGS['data_path'],
            '--dim', str(config['dim']),
            '--depth', str(BENCHMARK_SETTINGS['depth']),
            '--level', str(config['level']),
            '--expansion', str(config['expansion']),
            '--n_state', str(config['n_state']),
            '--batch_size', str(BENCHMARK_SETTINGS['batch_size']),
            '--chunk_size', str(BENCHMARK_SETTINGS['chunk_size']),
            '--lr', str(BENCHMARK_SETTINGS['lr']),
            '--warmup_steps', str(BENCHMARK_SETTINGS['warmup_steps']),
            '--train_minutes', str(BENCHMARK_SETTINGS['training_time'] // 60),
            '--seed', str(BENCHMARK_SETTINGS['seed']),
            '--log_every', str(BENCHMARK_SETTINGS['log_every']),
            '--output', str(output_dir / name),
            '--bf16',
        ]

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)

        print(f"\nStarting {name} on GPU {gpu}:")
        print(f"  level={config['level']}, dim={config['dim']}, n_state={config['n_state']}")
        print(f"  Log: {log_file}")

        with open(log_file, 'w') as f:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=os.getcwd()
            )
            processes.append((name, proc, log_file))

    print(f"\n{'='*60}")
    print(f"All {len(processes)} benchmarks started. Waiting for completion...")
    print(f"{'='*60}")

    # Wait for all processes and collect results
    results = {}
    for name, proc, log_file in processes:
        proc.wait()

        # Parse log for final loss
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()

            # Find last loss value
            last_loss = None
            last_step = None
            for line in reversed(lines):
                if 'loss=' in line or 'Loss:' in line:
                    # Try to parse loss
                    import re
                    match = re.search(r'loss[=:\s]+([0-9.]+)', line, re.IGNORECASE)
                    if match:
                        last_loss = float(match.group(1))
                    match = re.search(r'step[=:\s]+([0-9]+)', line, re.IGNORECASE)
                    if match:
                        last_step = int(match.group(1))
                    if last_loss:
                        break

            results[name] = {'loss': last_loss, 'steps': last_step, 'returncode': proc.returncode}
            print(f"{name}: loss={last_loss:.4f if last_loss else 'N/A'}, steps={last_step}, rc={proc.returncode}")
        except Exception as e:
            results[name] = {'error': str(e), 'returncode': proc.returncode}
            print(f"{name}: Error parsing results - {e}")

    # Print summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")

    sorted_results = sorted(
        [(k, v) for k, v in results.items() if v.get('loss')],
        key=lambda x: x[1]['loss']
    )

    for name, data in sorted_results:
        print(f"{name:15s}: loss={data['loss']:.4f}, steps={data['steps']}")

    print(f"\nResults saved to: {output_dir}")
    return results

if __name__ == '__main__':
    run_benchmark()
