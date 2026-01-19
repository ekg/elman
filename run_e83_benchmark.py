#!/usr/bin/env python3
"""
E83 Circular Tower 10-minute 100M param benchmark.

Tests E83 variants with different K (number of matrices) and bias modes:
- K=2 (2 circular matrices)
- K=4 (4 circular matrices)
- K=8 (8 circular matrices - many small heads hypothesis)

Bias modes:
- default: fixed learned bias
- nb: no bias
- ib: input-dependent bias

Plus baselines: llama, mamba2
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# E83 variant configurations - all ~100M params with depth=20, expansion=2
# Dim calculation: need to account for K and n_state in parameter count
#
# E83 per-layer params (expansion=2):
#   in_proj: 2*dim²
#   W_kv: K * 2 * n_state * 2 * dim
#   W_q: n_state * 2 * dim
#   B_gates (fixed): K * n_state
#   W_b (input_bias): K * n_state * 2 * dim
#   out_proj: n_state * dim
#
# Total per layer (fixed bias): 2*dim² + (4*K + 2 + 0.5)*n_state*dim
# Total 20 layers + embed(256*dim): 40*dim² + 20*(4*K + 2.5)*n_state*dim + 256*dim

CONFIGS = {
    # K=2 variants (n_state=32)
    '83k2': {'level': '83k2', 'dim': 1536, 'n_state': 32},           # Fixed bias
    '83k2nb': {'level': '83k2nb', 'dim': 1536, 'n_state': 32},       # No bias
    '83k2ib': {'level': '83k2ib', 'dim': 1408, 'n_state': 32},       # Input bias (more params from W_b)

    # K=4 variants (n_state=32)
    '83k4n32': {'level': '83k4n32', 'dim': 1408, 'n_state': 32},     # Fixed bias
    '83k4n32nb': {'level': '83k4n32nb', 'dim': 1408, 'n_state': 32}, # No bias
    '83k4n32ib': {'level': '83k4n32ib', 'dim': 1280, 'n_state': 32}, # Input bias

    # K=4 variants (n_state=24 - smaller state)
    '83k4n24': {'level': '83k4n24', 'dim': 1536, 'n_state': 24},     # Fixed bias
    '83k4n24nb': {'level': '83k4n24nb', 'dim': 1536, 'n_state': 24}, # No bias
    '83k4n24ib': {'level': '83k4n24ib', 'dim': 1408, 'n_state': 24}, # Input bias

    # K=8 variants (n_state=24 - many small heads)
    '83k8n24': {'level': '83k8n24', 'dim': 1408, 'n_state': 24},     # Fixed bias
    '83k8n24nb': {'level': '83k8n24nb', 'dim': 1408, 'n_state': 24}, # No bias
    '83k8n24ib': {'level': '83k8n24ib', 'dim': 1280, 'n_state': 24}, # Input bias

    # K=8 with n_state=16 (even smaller state)
    '83k8n16': {'level': '83k8n16', 'dim': 1536, 'n_state': 16},     # Fixed bias
    '83k8n16ib': {'level': '83k8n16ib', 'dim': 1408, 'n_state': 16}, # Input bias

    # Baselines
    'llama': {'level': 'llama', 'dim': 640, 'n_state': 64},
    'mamba2': {'level': 'mamba2', 'dim': 896, 'n_state': 64},
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
    'expansion': 2.0,
}

def get_available_gpus():
    """Get list of available GPU indices."""
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
        capture_output=True, text=True
    )
    return [int(x.strip()) for x in result.stdout.strip().split('\n') if x.strip()]

def run_benchmark(variants=None):
    """Run E83 variant benchmarks in parallel on different GPUs."""

    output_dir = Path('benchmark_results/e83_100m')
    output_dir.mkdir(parents=True, exist_ok=True)

    gpus = get_available_gpus()
    print(f"Available GPUs: {gpus}")

    # Select which configs to run
    if variants:
        configs = {k: v for k, v in CONFIGS.items() if k in variants}
    else:
        configs = CONFIGS

    print(f"Running {len(configs)} E83 benchmarks")

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
            '--expansion', str(BENCHMARK_SETTINGS['expansion']),
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
                if 'loss' in line.lower():
                    # Try to parse loss
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
            print(f"  {name}: level={c['level']}, dim={c['dim']}, n_state={c['n_state']}")
        sys.exit(0)

    run_benchmark(args.variants)
