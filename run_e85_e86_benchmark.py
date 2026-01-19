#!/usr/bin/env python3
"""
E85 & E86 benchmark: 10-minute 100M param comparison.

E85: Input-as-Matrix - input IS the state update matrix
E86: Input-Matrix Delta Rule - E85 + E75's delta rule with multi-head support

8-way comparison:
- E85 variants: n32, n48
- E86 single-head: n32, n48
- E86 multi-head: h2, h4, h4n24, h8n16
"""

import os
import sys
import subprocess
from pathlib import Path

# E86 configurations - all ~100M params with depth=20
# E86 params per layer: dim * (n_heads * n_state^2) + (n_heads * n_state) * dim
# For n_heads=1, n_state=32: cell_dim=1024, cell_out=32
#   per_layer = dim * 1024 + 32 * dim = dim * 1056
#   20 layers + embed = 20 * dim * 1056 + 256 * dim ≈ dim * 21376
#   For 100M: dim ≈ 4680 → too big, need higher cell_dim

# Recalculation for different configs to hit ~100M:
CONFIGS = {
    # E86 single-head (cell_dim = n_state^2)
    # n_state=32: cell_dim=1024, per_layer ≈ dim*1056, need dim~2176 for 100M
    '86n32': {'level': '86n32', 'dim': 2176, 'n_state': 32, 'expansion': 1.0},

    # n_state=48: cell_dim=2304, per_layer ≈ dim*2352, need dim~2048 for 100M
    '86n48': {'level': '86n48', 'dim': 2048, 'n_state': 48, 'expansion': 1.0},

    # E86 multi-head (cell_dim = n_heads * n_state^2)
    # h2n32: cell_dim=2048, cell_out=64, per_layer ≈ dim*2112
    '86h2': {'level': '86h2', 'dim': 2176, 'n_state': 32, 'expansion': 1.0},

    # h4n32: cell_dim=4096, cell_out=128, per_layer ≈ dim*4224
    '86h4': {'level': '86h4', 'dim': 1152, 'n_state': 32, 'expansion': 1.0},

    # h4n24: cell_dim=2304, cell_out=96, per_layer ≈ dim*2400
    '86h4n24': {'level': '86h4n24', 'dim': 2048, 'n_state': 24, 'expansion': 1.0},

    # h8n16: cell_dim=2048, cell_out=128, per_layer ≈ dim*2176
    '86h8n16': {'level': '86h8n16', 'dim': 2176, 'n_state': 16, 'expansion': 1.0},

    # Baselines for comparison
    'e75n48': {'level': '75n48', 'dim': 1408, 'n_state': 48, 'expansion': 2.0},
    'mamba2': {'level': 'mamba2', 'dim': 896, 'n_state': 64, 'expansion': 2.0},
}

# Standard benchmark settings - matches other 100M benchmarks
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

    output_dir = Path('benchmark_results/e85_e86_100m')
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
