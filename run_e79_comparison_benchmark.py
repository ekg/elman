#!/usr/bin/env python3
"""
E79 vs E42/E61/E68/E75 comparison benchmark.

8-way comparison at 100M params, 10 minutes, depth=20, expansion=2.
Matches E83 benchmark setup exactly.

Models:
- E42: Linear tied self-gating
- E61: Decay gated
- E68: Self-gating multiplicative h
- E75: Gated delta rule
- E79n48: Coupled matrix (fixed bias, n=48)
- E79n48nb: Coupled matrix (no bias, n=48)
- E79n48ib: Coupled matrix (input bias, n=48)
- mamba2: Baseline SSM
"""

import os
import sys
import subprocess
from pathlib import Path

# All configs use expansion=2.0 to match E83 benchmark
# Dims calculated to hit ~100M params at depth=20

CONFIGS = {
    # E42: LinearTiedElman - from 100m benchmark: dim=768 → ~95M params
    'e42': {'level': 42, 'dim': 768, 'n_state': 64, 'expansion': 2.0},

    # E61: DecayGated - from 100m benchmark: dim=640 → ~98M params
    'e61': {'level': 61, 'dim': 640, 'n_state': 64, 'expansion': 2.0},

    # E68: Self-gating - from 100m benchmark: dim=640 → ~98M params
    'e68': {'level': 68, 'dim': 640, 'n_state': 64, 'expansion': 2.0},

    # E75: Gated delta rule - dim=1408 gives ~100M with expansion=2, n_state=48
    'e75n48': {'level': '75n48', 'dim': 1408, 'n_state': 48, 'expansion': 2.0},

    # E79 variants - expansion=1.0, dim=2048 gives ~94M params with n_state=48
    'e79n48': {'level': '79n48', 'dim': 2048, 'n_state': 48, 'expansion': 1.0},     # Fixed bias
    'e79n48nb': {'level': '79n48nb', 'dim': 2048, 'n_state': 48, 'expansion': 1.0}, # No bias
    'e79n48ib': {'level': '79n48ib', 'dim': 2048, 'n_state': 48, 'expansion': 1.0}, # Input bias

    # Baseline
    'mamba2': {'level': 'mamba2', 'dim': 896, 'n_state': 64, 'expansion': 2.0},
}

# Standard benchmark settings - matches E83 exactly
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

def run_benchmark(variants=None):
    """Run benchmarks in parallel on different GPUs."""

    output_dir = Path('benchmark_results/e79_comparison_100m')
    output_dir.mkdir(parents=True, exist_ok=True)

    gpus = get_available_gpus()
    print(f"Available GPUs: {gpus}")

    # Select which configs to run
    if variants:
        configs = {k: v for k, v in CONFIGS.items() if k in variants}
    else:
        configs = CONFIGS

    print(f"Running {len(configs)} benchmarks")

    processes = []

    for i, (name, config) in enumerate(configs.items()):
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
        print(f"  level={config['level']}, dim={config['dim']}, n_state={config['n_state']}, expansion={config['expansion']}")
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
                if 'loss' in line.lower():
                    import re
                    match = re.search(r'loss[=:\s]+([0-9.]+)', line, re.IGNORECASE)
                    if match:
                        val = match.group(1)
                        if val != 'nan':
                            last_loss = float(val)
                    match = re.search(r'step[=:\s]+([0-9]+)', line, re.IGNORECASE)
                    if match:
                        last_step = int(match.group(1))
                    if last_loss:
                        break

            results[name] = {'loss': last_loss, 'steps': last_step, 'returncode': proc.returncode}
            loss_str = f"{last_loss:.4f}" if last_loss else "N/A"
            print(f"{name}: loss={loss_str}, steps={last_step}, rc={proc.returncode}")
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--variants', type=str, nargs='+', default=None,
                        help='Specific variants to run (default: all)')
    parser.add_argument('--list', action='store_true', help='List available variants')
    args = parser.parse_args()

    if args.list:
        print("Available variants:")
        for name in CONFIGS:
            c = CONFIGS[name]
            print(f"  {name}: level={c['level']}, dim={c['dim']}, n_state={c['n_state']}, expansion={c['expansion']}")
        sys.exit(0)

    run_benchmark(args.variants)
