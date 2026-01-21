#!/usr/bin/env python3
"""
500M Parameter Benchmark: E88 vs Baselines
Comparing at scale: deeper (48 layers) and wider models

Target: ~500M parameters per model
Training: 30 minutes per model (longer for better convergence signal at scale)
"""

import subprocess
import sys
import os
from datetime import datetime

# Benchmark configuration
TARGET_PARAMS = 500_000_000
DEPTH = 48  # Deeper than 100M benchmark (which used 20)
TRAIN_MINUTES = 30
BATCH_SIZE = 16  # Smaller batch for larger models
CHUNK_SIZE = 512
LR = 1e-4  # Lower LR for larger models
WARMUP_STEPS = 500
SEED = 42

# Output directory
OUTPUT_DIR = "benchmark_results/500m_30min"

# Model configurations for ~500M params at depth=48
# Calculated from param scaling analysis
CONFIGS = {
    # E88 variants (expansion=1.0, no conv/gate/norm - best from ablation)
    # With n_state=32 (optimal from ablation) - very param efficient
    'e88_h8n32': {
        'level': 'E88_h8n32_500m',
        'dim': 9728,  # 485M params
        'expansion': 1.0,
        'n_state': 32,
        'notes': 'E88 best config (h8n32) scaled to 500M',
    },
    # With n_state=64 - test if larger state helps at scale
    'e88_h8n64': {
        'level': 'E88_h8n64_500m',
        'dim': 4864,  # 482M params
        'expansion': 1.0,
        'n_state': 64,
        'notes': 'E88 with larger state (h8n64)',
    },

    # FLA-GDN (state-of-the-art linear attention, ICLR 2025)
    'fla_gdn': {
        'level': 'fla-gdn',
        'dim': 1152,  # 512M params
        'expansion': 2.0,
        'n_state': 64,
        'notes': 'FLA GatedDeltaNet (SOTA linear attention)',
    },

    # Mamba2 (state-of-the-art SSM)
    'mamba2': {
        'level': 'mamba2',
        'dim': 1280,  # 484M params
        'expansion': 2.0,
        'n_state': 64,
        'notes': 'Mamba2 SSM (SOTA SSM)',
    },

    # E42 - Linear tied self-gated (previous best simple Elman)
    'e42': {
        'level': 42,
        'dim': 1152,  # 510M params
        'expansion': 2.0,
        'n_state': 32,
        'notes': 'E42 linear tied self-gate',
    },

    # E1 - Original gated Elman baseline
    'e1': {
        'level': 1,
        'dim': 832,  # 466M params (closest to 500M)
        'expansion': 2.0,
        'n_state': 32,
        'notes': 'E1 gated Elman baseline',
    },

    # Llama-style transformer (attention baseline)
    'llama': {
        'level': 'llama',
        'dim': 896,  # 485M params
        'expansion': 2.0,
        'n_state': 32,
        'notes': 'Llama-style transformer baseline',
    },
}


def get_gpu_count():
    """Get number of available GPUs."""
    try:
        result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
        return len([l for l in result.stdout.strip().split('\n') if l.startswith('GPU')])
    except:
        return 1


def run_benchmark(config_name, config, gpu_id, output_dir):
    """Run a single benchmark."""
    level = config['level']
    dim = config['dim']
    expansion = config.get('expansion', 2.0)
    n_state = config.get('n_state', 32)

    log_file = os.path.join(output_dir, f"{config_name}.log")
    model_dir = os.path.join(output_dir, config_name)

    cmd = [
        'python', '-u', 'train.py',
        '--level', str(level),
        '--dim', str(dim),
        '--expansion', str(expansion),
        '--n_state', str(n_state),
        '--data', 'data/pile.txt',
        '--depth', str(DEPTH),
        '--batch_size', str(BATCH_SIZE),
        '--chunk_size', str(CHUNK_SIZE),
        '--lr', str(LR),
        '--warmup_steps', str(WARMUP_STEPS),
        '--seed', str(SEED),
        '--bf16',
        '--train_minutes', str(TRAIN_MINUTES),
        '--output', model_dir,
    ]

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    print(f"[GPU {gpu_id}] Starting {config_name}: {config.get('notes', '')}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Log: {log_file}")

    with open(log_file, 'w') as f:
        proc = subprocess.Popen(
            cmd, env=env, stdout=f, stderr=subprocess.STDOUT
        )

    return proc, config_name, log_file


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n_gpus = get_gpu_count()
    print(f"500M Parameter Benchmark")
    print(f"========================")
    print(f"Available GPUs: {n_gpus}")
    print(f"Target params: {TARGET_PARAMS:,}")
    print(f"Depth: {DEPTH}, Train time: {TRAIN_MINUTES} min")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # Print configs
    print("Model configurations:")
    for name, cfg in CONFIGS.items():
        print(f"  {name}: dim={cfg['dim']}, level={cfg['level']}, {cfg.get('notes', '')}")
    print()

    # Launch all models
    processes = []
    for i, (name, config) in enumerate(CONFIGS.items()):
        gpu_id = i % n_gpus
        proc, name, log = run_benchmark(name, config, gpu_id, OUTPUT_DIR)
        processes.append((proc, name, log))

    print(f"\nLaunched {len(processes)} jobs. Waiting for completion...")
    print(f"Monitor with: tail -f {OUTPUT_DIR}/*.log")

    # Wait for all
    for proc, name, log in processes:
        proc.wait()
        print(f"  {name} finished (exit code: {proc.returncode})")

    print("\nBenchmark complete!")
    print(f"Results in: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
