#!/usr/bin/env python3
"""
E87 Sparse Block Memory Benchmark

Wave 2 of the 16-way benchmark comparing E87 variants with different
block counts and top-k values.
"""

import os
import subprocess
import time
from datetime import datetime

# E87 model variants: (name, dim, n_state, n_blocks, top_k, description)
# Target ~100M params with depth=20
MODELS = [
    ('87b4k2', 1408, 32, 4, 2, '4 blocks, top-2'),
    ('87b4k1', 1408, 32, 4, 1, '4 blocks, top-1'),
    ('87b8k2', 1280, 24, 8, 2, '8 blocks, top-2'),
    ('87b8k4', 1280, 24, 8, 4, '8 blocks, top-4'),
    ('87b4k2n48', 1280, 48, 4, 2, '4 blocks, n=48, top-2'),
    ('87b6k2', 1408, 32, 6, 2, '6 blocks, top-2'),
    ('87b6k3', 1408, 32, 6, 3, '6 blocks, top-3'),
    ('87b8k3', 1280, 24, 8, 3, '8 blocks, top-3'),
]

COMMON_ARGS = [
    '--data', 'data/pile.txt',
    '--depth', '20',
    '--batch_size', '32',
    '--chunk_size', '512',
    '--lr', '3e-4',
    '--warmup_steps', '1000',
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
    output_dir = f'benchmark_results/e87_100m_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    gpus = get_available_gpus()
    print(f"Found {len(gpus)} GPUs: {gpus}")
    print(f"Output directory: {output_dir}")
    print(f"Running {len(MODELS)} E87 variants")
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
    print("E87 Benchmark Complete!")
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
