#!/usr/bin/env python3
"""
E88 vs FLA-GDN Benchmark
Compare E88 FLA Hybrid against FLA-GatedDeltaNet at matched parameters (~100M).

Based on 100M benchmark protocol from CLAUDE.md.
"""

import argparse
import subprocess
import sys
import os
from datetime import datetime

# Model configurations for ~100M params at depth=20
# E88: dim=768, expansion=2.0, n_state=32, n_heads=8
# FLA-GDN: dim=768, depth=20, expand=2

CONFIGS = {
    # E88 correct config: expansion=1.0, no conv, no gate, no norm, many heads of size 32
    # Matched to FLA-GDN at ~60M params
    'e88': {
        'level': 'E88_h20n32',  # 20 heads × 32×32 state, no conv/gate/norm
        'dim': 1152,            # ~60M params (matches FLA-GDN dim=768)
        'depth': 20,
    },
    # Larger E88 config for ~100M params
    'e88_100m': {
        'level': 'E88_h40n32',  # 40 heads × 32×32 state
        'dim': 1152,            # ~100M params
        'depth': 20,
    },
    # Mamba2 state-matched: h128n32 = 128 × 32² = 131,072 state/layer
    'e88_h128': {
        'level': 'E88_h128n32',
        'dim': 768,
        'depth': 20,
    },
    'fla-gdn': {
        'level': 'fla-gdn',
        'dim': 768,
        'depth': 20,
    },
    # FLA-GDN at 100M params
    'fla-gdn-100m': {
        'level': 'fla-gdn',
        'dim': 1024,            # ~106M params
        'depth': 20,
    },
}

def run_benchmark(model_name, config, output_dir, duration_minutes=10, gpu_id=0):
    """Run a single benchmark."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(output_dir, f'{model_name}.log')

    cmd = [
        'python3', 'train.py',
        '--data', 'data/pile.txt',
        '--level', str(config['level']),
        '--dim', str(config['dim']),
        '--depth', str(config.get('depth', 20)),
        '--batch_size', '32',
        '--chunk_size', '512',
        '--lr', '3e-4',
        '--warmup_steps', '1000',
        '--train_minutes', str(duration_minutes),
        '--output', os.path.join(output_dir, model_name),
        '--seed', '42',
        '--bf16',
    ]

    # Add model-specific args (only if explicitly specified)
    if 'expansion' in config:
        cmd.extend(['--expansion', str(config['expansion'])])
    if 'n_state' in config:
        cmd.extend(['--n_state', str(config['n_state'])])
    if 'n_heads' in config:
        cmd.extend(['--n_heads', str(config['n_heads'])])

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    print(f"Running {model_name} on GPU {gpu_id}...")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Log: {log_file}")

    with open(log_file, 'w') as f:
        proc = subprocess.Popen(
            cmd, stdout=f, stderr=subprocess.STDOUT, env=env
        )

    return proc, log_file

def main():
    parser = argparse.ArgumentParser(description='E88 vs FLA-GDN Benchmark')
    parser.add_argument('--output_dir', default='benchmark_results/e88_vs_fla_gdn',
                       help='Output directory for results')
    parser.add_argument('--duration', type=int, default=10,
                       help='Training duration in minutes per model')
    parser.add_argument('--models', nargs='+', default=['e88', 'fla-gdn'],
                       help='Models to benchmark')
    parser.add_argument('--parallel', action='store_true',
                       help='Run models in parallel on different GPUs')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Check available GPUs
    result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
    num_gpus = len([l for l in result.stdout.strip().split('\n') if l.startswith('GPU')])
    print(f"Found {num_gpus} GPUs")

    if args.parallel and num_gpus >= len(args.models):
        # Run models in parallel
        print(f"\nRunning {len(args.models)} models in parallel...")
        procs = []
        for i, model_name in enumerate(args.models):
            if model_name not in CONFIGS:
                print(f"Unknown model: {model_name}, skipping")
                continue
            proc, log = run_benchmark(
                model_name, CONFIGS[model_name],
                args.output_dir, args.duration, gpu_id=i
            )
            procs.append((model_name, proc, log))

        print("\nWaiting for all models to complete...")
        for model_name, proc, log in procs:
            proc.wait()
            print(f"  {model_name}: completed (exit code {proc.returncode})")
    else:
        # Run models sequentially
        print(f"\nRunning {len(args.models)} models sequentially...")
        for model_name in args.models:
            if model_name not in CONFIGS:
                print(f"Unknown model: {model_name}, skipping")
                continue
            proc, log = run_benchmark(
                model_name, CONFIGS[model_name],
                args.output_dir, args.duration, gpu_id=0
            )
            proc.wait()
            print(f"  {model_name}: completed (exit code {proc.returncode})")

    # Print results summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    for model_name in args.models:
        log_file = os.path.join(args.output_dir, f'{model_name}.log')
        if not os.path.exists(log_file):
            continue

        # Extract final loss from log
        with open(log_file, 'r') as f:
            lines = f.readlines()

        final_loss = None
        throughput = None
        for line in reversed(lines):
            if 'loss=' in line and final_loss is None:
                try:
                    loss_str = line.split('loss=')[1].split()[0].strip(',')
                    final_loss = float(loss_str)
                except:
                    pass
            if 'tok/s' in line and throughput is None:
                try:
                    toks_str = line.split('tok/s')[0].split()[-1].strip()
                    throughput = toks_str
                except:
                    pass

        print(f"{model_name:15s}: loss={final_loss or 'N/A':8s}  throughput={throughput or 'N/A'}")

    print("=" * 60)

if __name__ == '__main__':
    main()
