#!/usr/bin/env python3
"""
E87 High Block Count Benchmark

Exploring E87 with 16 and 32 blocks to understand scaling behavior.
"""

import os
import subprocess
import time
from datetime import datetime

# E87 high-block variants: (name, dim, n_state, n_blocks, top_k, description)
# Target ~100M params with depth=20
MODELS = [
    # 16 blocks with n_state=16 (256 params per block state)
    ('87b16k2', 1408, 16, 16, 2, '16 blocks, top-2 (sparse)'),
    ('87b16k4', 1408, 16, 16, 4, '16 blocks, top-4'),
    ('87b16k6', 1408, 16, 16, 6, '16 blocks, top-6'),
    ('87b16k8', 1408, 16, 16, 8, '16 blocks, top-8 (half)'),
    # 32 blocks with n_state=16 (256 params per block state)
    ('87b32k4', 1280, 16, 32, 4, '32 blocks, top-4 (sparse)'),
    ('87b32k8', 1280, 16, 32, 8, '32 blocks, top-8'),
    ('87b32k12', 1280, 16, 32, 12, '32 blocks, top-12'),
    ('87b32k16', 1280, 16, 32, 16, '32 blocks, top-16 (half)'),
]

COMMON_ARGS = [
    '--data', 'data/pile.txt',
    '--depth', '20',
    '--batch_size', '32',
    '--chunk_size', '512',
    '--lr', '3e-4',
    '--warmup_steps', '1',  # Minimal warmup (train.py requires > 0)
    '--seed', '42',
    '--expansion', '1.0',
    '--train_minutes', '10',
    '--bf16',
]

def get_available_gpus():
    """Get list of available CUDA devices."""
    result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
    return [i for i, _ in enumerate(result.stdout.strip().split('\n')) if 'GPU' in _]

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'benchmark_results/e87_highblock_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    gpus = get_available_gpus()
    print(f"Found {len(gpus)} GPUs: {gpus}")
    print(f"Output directory: {output_dir}")
    print(f"Running {len(MODELS)} E87 high-block variants")
    print()

    # Launch all models in parallel on available GPUs
    processes = []

    for i, (name, dim, n_state, n_blocks, top_k, desc) in enumerate(MODELS):
        gpu = gpus[i % len(gpus)]

        # Build command
        cmd = [
            'python', 'train.py',
            '--level', name,
            '--dim', str(dim),
            '--n_state', str(n_state),
            '--output', f'{output_dir}/{name}',
        ] + COMMON_ARGS

        log_file = f'{output_dir}/{name}.log'

        print(f"[GPU {gpu}] Starting {name} (dim={dim}, n_state={n_state}, n_blocks={n_blocks}, top_k={top_k})")
        print(f"  {desc}")

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)

        with open(log_file, 'w') as log:
            proc = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
            )
            processes.append((name, proc, log_file))

    print(f"\nAll {len(processes)} processes started. Waiting for completion...")
    print()

    # Wait for all to complete
    for name, proc, log_file in processes:
        proc.wait()
        ret = proc.returncode
        status = "OK" if ret == 0 else f"FAILED ({ret})"
        print(f"  {name}: {status}")

    print()
    print("=" * 60)
    print("E87 High-Block Benchmark Complete!")
    print(f"Results in: {output_dir}")
    print()

    # Print summary
    print("Summary:")
    print("-" * 60)
    for name, proc, log_file in processes:
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            # Find last step line
            for line in reversed(lines):
                if line.strip().startswith('step'):
                    parts = line.split('|')
                    step = parts[0].strip().split()[1]
                    loss = None
                    for p in parts:
                        if 'loss' in p:
                            loss = p.split()[1]
                            break
                    print(f"  {name:12s}: step={step}, loss={loss}")
                    break
        except Exception as e:
            print(f"  {name:12s}: Error reading log - {e}")

if __name__ == '__main__':
    main()
