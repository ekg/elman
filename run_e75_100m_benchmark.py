#!/usr/bin/env python3
"""
E75 Multi-Head 100M Parameter Benchmark

All models target ~100M parameters with depth=20, expansion=1.0.
Dimensions are 128-aligned. n_state values are multiples of 8 (required for stability).

Includes mamba2 and fla-gdn baselines.
"""

import os
import subprocess
import time
from datetime import datetime

# E75 100M configurations calculated to hit ~100M params
# Format: (name, dim, n_state, n_heads, description, expected_params)
E75_MODELS = [
    # 4 heads - varying state size
    ('E75h4n16', 2048, 16, 4, '4 heads, n=16 (small state)', '97.6M'),
    ('E75h4n24', 2048, 24, 4, '4 heads, n=24 (medium state)', '104.1M'),
    ('E75h4n32', 1920, 32, 4, '4 heads, n=32 (large state)', '98.8M'),
    ('E75h4n48', 1792, 48, 4, '4 heads, n=48 (very large state)', '99.1M'),
    # 5-6 heads
    ('E75h5n24', 1920, 24, 5, '5 heads, n=24', '97.3M'),
    ('E75h6n24', 1920, 24, 6, '6 heads, n=24', '101.9M'),
    # 8 heads - more parallelism
    ('E75h8n16', 1920, 16, 8, '8 heads, n=16', '98.8M'),
    ('E75h8n24', 1792, 24, 8, '8 heads, n=24', '99.1M'),
]

# Baselines
BASELINE_MODELS = [
    ('mamba2', 896, None, None, 'Mamba2 SSM (102M)', '102M'),
    ('fla-gdn', 768, None, None, 'FLA GatedDeltaNet (95M)', '95M'),
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
    return [i for i, line in enumerate(result.stdout.strip().split('\n')) if 'GPU' in line]

def run_models(models, output_dir, gpus, wave_name):
    """Run a wave of models in parallel."""
    print(f"\n{'='*70}")
    print(f"{wave_name}")
    print(f"{'='*70}")

    processes = []

    for i, (name, dim, n_state, n_heads, desc, expected) in enumerate(models):
        gpu = gpus[i % len(gpus)]

        # Build command
        cmd = [
            'python', 'train.py',
            '--level', name,
            '--dim', str(dim),
            '--output', f'{output_dir}/{name}',
        ] + COMMON_ARGS

        # Add n_state for E75 variants
        if n_state is not None:
            cmd.extend(['--n_state', str(n_state)])

        log_file = f'{output_dir}/{name}.log'

        print(f"[GPU {gpu}] {name}: {desc} (~{expected})")

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

    # Wait for all to complete
    for name, proc, log_file in processes:
        proc.wait()
        ret = proc.returncode
        status = "OK" if ret == 0 else f"FAILED ({ret})"
        print(f"  {name}: {status}")

    return processes

def extract_results(output_dir, models):
    """Extract results from logs and checkpoints."""
    print(f"\n{'='*70}")
    print("Results Summary")
    print(f"{'='*70}")

    results = []
    for name, dim, n_state, n_heads, desc, expected in models:
        log_file = f'{output_dir}/{name}.log'
        try:
            # Get actual params from log
            with open(log_file, 'r') as f:
                first_lines = f.read(2000)
            params_match = None
            for line in first_lines.split('\n'):
                if 'parameters' in line:
                    import re
                    match = re.search(r'([\d,]+)\s*parameters', line)
                    if match:
                        params_match = int(match.group(1).replace(',', ''))
                        break

            # Get final checkpoint for loss
            import glob
            chk_pattern = f'{output_dir}/{name}/level*/checkpoint_*.pt'
            chks = glob.glob(chk_pattern)
            if chks:
                latest = max(chks, key=os.path.getmtime)
                loss = os.path.basename(latest).split('loss_')[1].split('.pt')[0]
                steps = os.path.basename(latest).split('step_')[1].split('_')[0].lstrip('0') or '0'
                params_str = f"{params_match/1e6:.1f}M" if params_match else expected
                results.append((name, dim, steps, loss, params_str, desc))
            else:
                results.append((name, dim, '-', 'FAILED', expected, desc))
        except Exception as e:
            results.append((name, dim, '-', f'ERROR', expected, desc))

    # Sort by loss
    def sort_key(x):
        try:
            return float(x[3])
        except:
            return 999
    results.sort(key=sort_key)

    print(f"\n{'Model':<12} {'Dim':<6} {'Steps':>6} {'Loss':>8} {'Params':>10} {'Description'}")
    print("-" * 80)
    for name, dim, steps, loss, params, desc in results:
        print(f"{name:<12} {dim:<6} {steps:>6} {loss:>8} {params:>10} {desc}")

    return results

def main():
    import sys

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'benchmark_results/e75_100m_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    gpus = get_available_gpus()
    print(f"Found {len(gpus)} GPUs: {gpus}")
    print(f"Output directory: {output_dir}")

    all_models = E75_MODELS + BASELINE_MODELS
    print(f"Total models to benchmark: {len(all_models)}")
    print()

    # Print configurations
    print("Configurations:")
    print("-" * 70)
    for name, dim, n_state, n_heads, desc, expected in all_models:
        state_info = f"H={n_heads}, n={n_state}" if n_state else "N/A"
        print(f"  {name:<12} dim={dim:<5} {state_info:<15} ~{expected}")
    print()

    # Check for wave argument
    if len(sys.argv) > 1:
        wave = sys.argv[1]
        if wave == 'e75':
            run_models(E75_MODELS, output_dir, gpus, "E75 Models Only")
            extract_results(output_dir, E75_MODELS)
        elif wave == 'baselines':
            run_models(BASELINE_MODELS, output_dir, gpus, "Baselines Only")
            extract_results(output_dir, BASELINE_MODELS)
        else:
            print(f"Unknown wave: {wave}. Use 'e75' or 'baselines' or no arg for all.")
            return
    else:
        # Run all models
        if len(gpus) >= len(all_models):
            # All at once
            run_models(all_models, output_dir, gpus, "All Models (single wave)")
            extract_results(output_dir, all_models)
        else:
            # Two waves
            run_models(E75_MODELS, output_dir, gpus, "Wave 1: E75 Models")
            run_models(BASELINE_MODELS, output_dir, gpus, "Wave 2: Baselines")
            extract_results(output_dir, all_models)

    print(f"\n{'='*70}")
    print("Benchmark Complete!")
    print(f"Results in: {output_dir}")


if __name__ == '__main__':
    main()
