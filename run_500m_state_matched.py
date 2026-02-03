#!/usr/bin/env python
"""
500M Parameter Benchmark: State-Matched E88 vs Baselines

Using many-head configs with smaller n_state for better throughput.

State sizes per layer:
- Mamba2 (500M): ~163,840 state/layer
- E88 h40n64: 40 × 64² = 163,840 state/layer (exact Mamba2 match!)
- E88 h64n64: 64 × 64² = 262,144 state/layer
- E88 h72n48: 72 × 48² = 165,888 state/layer (close to Mamba2)
"""

import os
import subprocess
import time
import re

CONFIGS = {
    'e88_h40n64': {
        'level': 'E88_h40n64',
        'dim': 2432,
        'depth': 20,
        'state_per_layer': '163,840',  # Exact Mamba2 match
    },
    'e88_h64n64': {
        'level': 'E88_h64n64',
        'dim': 1536,
        'depth': 20,
        'state_per_layer': '262,144',
    },
    'e88_h72n48': {
        'level': 'E88_h72n48',
        'dim': 1792,
        'depth': 20,
        'state_per_layer': '165,888',
    },
    'e88_h96n32': {
        'level': 'E88_h96n32',
        'dim': 2048,
        'depth': 20,
        'state_per_layer': '98,304',
    },
    'mamba2': {
        'level': 'mamba2',
        'target_params': '500m',
        'state_per_layer': '~163,840'
    },
}

TRAIN_TIME = 30
BATCH_SIZE = 16
GRAD_ACCUM = 1
CHUNK_SIZE = 512
LR = 1e-4
WARMUP = 1000
DATA_PATH = 'data/pile.txt'
SEED = 42
OUTPUT_DIR = 'benchmark_results/500m_state_matched'

def get_available_gpus():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
        capture_output=True, text=True
    )
    return [int(x.strip()) for x in result.stdout.strip().split('\n') if x.strip()]

def run_benchmark(name, config, gpu_id, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f'{name}.log')

    cmd = [
        'python', '-u', 'train.py',
        '--level', config['level'],
        '--batch_size', str(BATCH_SIZE),
        '--grad_accum', str(GRAD_ACCUM),
        '--chunk_size', str(CHUNK_SIZE),
        '--lr', str(LR),
        '--warmup_steps', str(WARMUP),
        '--train_minutes', str(TRAIN_TIME),
        '--data', DATA_PATH,
        '--seed', str(SEED),
        '--output', os.path.join(output_dir, name),
        '--bf16',
    ]

    if 'dim' in config:
        cmd.extend(['--dim', str(config['dim'])])
    if 'depth' in config:
        cmd.extend(['--depth', str(config['depth'])])
    if 'target_params' in config:
        cmd.extend(['--params', config['target_params']])

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    print(f'Starting {name} on GPU {gpu_id} (state/layer: {config["state_per_layer"]})')

    with open(log_file, 'w') as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)

    return proc, log_file

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    gpus = get_available_gpus()
    print(f'Available GPUs: {gpus}')
    print(f'Batch size: {BATCH_SIZE}, Training time: {TRAIN_TIME} min')

    processes = []
    for i, (name, config) in enumerate(CONFIGS.items()):
        gpu_id = gpus[i % len(gpus)]
        proc, log_file = run_benchmark(name, config, gpu_id, OUTPUT_DIR)
        processes.append((name, proc, log_file, config))

    print(f'Started {len(processes)} experiments. Waiting...')

    for name, proc, log_file, config in processes:
        proc.wait()
        print(f'{name} completed (exit: {proc.returncode})')

    print('\n' + '='*60)
    print('RESULTS')
    print('='*60)
    for name, proc, log_file, config in processes:
        if os.path.exists(log_file):
            with open(log_file) as f:
                content = f.read()
            loss_match = re.search(r'loss[=:]\s*([\d.]+)', content[-2000:] if len(content) > 2000 else content)
            loss = loss_match.group(1) if loss_match else 'N/A'
            print(f'{name}: loss={loss}, state={config["state_per_layer"]}')

if __name__ == '__main__':
    main()
