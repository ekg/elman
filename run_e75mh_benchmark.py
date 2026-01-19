#!/usr/bin/env python
"""
Benchmark E75 Multi-Head variants (100M params, 10 min training)

Wave 1: E75 Multi-Head configurations (8 models, 8 GPUs)
- E75h2, E75h4, E75h8: varying number of heads
- E75h2n48, E75h4n24, E75h8n16: constant total state size ~4K
- E75h4n32, E75h8n24: additional configurations
"""

import subprocess
import os
import sys
from datetime import datetime

# Target ~100M params at depth=20
# E75 Multi-Head param formula:
#   in_proj: dim * d_inner
#   out_proj: H * n_state * dim
#   cell projections: 4 * H * n_state * d_inner
#   b_beta: H * n_state
#
# For E75 variants, we need to compute dims that hit ~100M params

MODELS = [
    # (level, dim, n_state, notes)
    ('E75h2', 1408, 64, '2 heads, n_state=64'),       # default n_state
    ('E75h4', 1408, 32, '4 heads, n_state=32'),       # default n_state
    ('E75h8', 1280, 24, '8 heads, n_state=24'),       # default n_state
    ('E75h2n48', 1536, 48, '2 heads, n_state=48'),
    ('E75h4n24', 1536, 24, '4 heads, n_state=24'),
    ('E75h8n16', 1536, 16, '8 heads, n_state=16'),
    ('E75h4n32', 1408, 32, '4 heads, n_state=32'),
    ('E75h8n24', 1280, 24, '8 heads, n_state=24'),
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
    '--train_minutes', '10',  # 10 minutes
    '--bf16',  # Use bfloat16 for CUDA
]

def run_benchmark():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'benchmark_results/e75mh_100m_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    # Check available GPUs
    result = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
                          capture_output=True, text=True)
    gpus = [int(x.strip()) for x in result.stdout.strip().split('\n') if x.strip()]
    num_gpus = len(gpus)
    print(f"Found {num_gpus} GPUs: {gpus}")

    if num_gpus < len(MODELS):
        print(f"Warning: Only {num_gpus} GPUs available for {len(MODELS)} models")

    # Launch all experiments in parallel
    processes = []
    for i, (level, dim, n_state, notes) in enumerate(MODELS):
        gpu_id = gpus[i % num_gpus]
        log_file = f'{output_dir}/{level}.log'

        cmd = [
            sys.executable, 'train.py',
            '--level', level,
            '--dim', str(dim),
            '--n_state', str(n_state),
            '--output', f'{output_dir}/{level}',
            *COMMON_ARGS,
        ]

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        print(f"GPU {gpu_id}: {level} (dim={dim}, n_state={n_state}) -> {log_file}")

        with open(log_file, 'w') as f:
            p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
            processes.append((level, p, log_file))

    # Wait for all to complete
    print(f"\nWaiting for {len(processes)} experiments...")
    for level, p, log_file in processes:
        p.wait()
        rc = p.returncode
        status = "OK" if rc == 0 else f"FAILED (rc={rc})"
        print(f"  {level}: {status}")

    # Summary
    print(f"\nResults in: {output_dir}/")
    print("Extract results with:")
    print(f"  grep -h 'loss:' {output_dir}/*.log | tail -n 100")

if __name__ == '__main__':
    run_benchmark()
