#!/usr/bin/env python3
"""
E79 Coupled Matrix 10-minute benchmark with multiple n_state sizes.

Runs 4 parallel training jobs on different GPUs, one per n_state value.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# E79 benchmark configurations - all ~100M params with dim=2048
# n_state controls the matrix state size (n_state × n_state)
CONFIGS = {
    'e79n32': {'level': 79, 'dim': 2176, 'n_state': 32, 'expansion': 1.0},  # 102.3M
    'e79n48': {'level': 79, 'dim': 2048, 'n_state': 48, 'expansion': 1.0},  # 94.3M
    'e79n64': {'level': 79, 'dim': 2048, 'n_state': 64, 'expansion': 1.0},  # 97.6M
    'e79n96': {'level': 79, 'dim': 2048, 'n_state': 96, 'expansion': 1.0},  # 104.1M
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
    """Run E79 benchmarks in parallel on different GPUs."""

    output_dir = Path('benchmark_results/e79_100m_10min')
    output_dir.mkdir(parents=True, exist_ok=True)

    gpus = get_available_gpus()
    print(f"Available GPUs: {gpus}")

    if len(gpus) < len(CONFIGS):
        print(f"Warning: Only {len(gpus)} GPUs available for {len(CONFIGS)} configs")

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
        print(f"  dim={config['dim']}, n_state={config['n_state']}")
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
    print(f"Started {len(processes)} benchmark processes")
    print(f"Training for {BENCHMARK_SETTINGS['training_time']} seconds each")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")

    # Wait for all processes
    for name, proc, log_file in processes:
        proc.wait()
        print(f"\n{name} completed with exit code {proc.returncode}")

    # Extract results
    print(f"\n{'='*60}")
    print("FINAL RESULTS:")
    print(f"{'='*60}")

    for name, proc, log_file in processes:
        try:
            # Get last loss from log
            with open(log_file, 'r') as f:
                lines = f.readlines()

            # Find last line with loss
            last_loss = None
            last_step = None
            for line in reversed(lines):
                if 'loss=' in line or 'Loss:' in line:
                    parts = line.split()
                    for p in parts:
                        if p.startswith('loss='):
                            last_loss = float(p.split('=')[1].rstrip(','))
                        elif p.startswith('step='):
                            last_step = int(p.split('=')[1].rstrip(','))
                    if last_loss:
                        break

            config = CONFIGS[name]
            print(f"{name}: dim={config['dim']}, n_state={config['n_state']}, "
                  f"step={last_step}, loss={last_loss:.4f}" if last_loss else f"{name}: No results found")
        except Exception as e:
            print(f"{name}: Error reading results - {e}")

if __name__ == '__main__':
    run_benchmark()
