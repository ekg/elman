#!/usr/bin/env python3
"""
E75 Multi-Head Parameter Scan Benchmark

Deep exploration around the best E75h4n24 configuration.
Includes mamba2 and fla-gdn baselines for comparison.

Two waves of 8 jobs each (16 total).
"""

import os
import subprocess
import time
from datetime import datetime

# Wave 1: Focus on 3-4 heads and baselines
WAVE1_MODELS = [
    # (name, dim, n_state, description)
    ('E75h4n24', 1536, 24, '4 heads, n=24 (baseline best)'),
    ('E75h3n32', 1536, 32, '3 heads, n=32'),
    ('E75h4n20', 1536, 20, '4 heads, n=20'),
    ('E75h4n28', 1408, 28, '4 heads, n=28'),
    ('E75h5n20', 1408, 20, '5 heads, n=20'),
    ('E75h6n16', 1536, 16, '6 heads, n=16'),
    ('mamba2', 896, None, 'Mamba2 SSM baseline'),
    ('fla-gdn', 768, None, 'FLA GatedDeltaNet baseline'),
]

# Wave 2: More exploration of 3-6 heads
WAVE2_MODELS = [
    ('E75h3n24', 1536, 24, '3 heads, n=24'),
    ('E75h3n28', 1536, 28, '3 heads, n=28'),
    ('E75h4n16', 1664, 16, '4 heads, n=16'),
    ('E75h4n32', 1408, 32, '4 heads, n=32'),
    ('E75h5n16', 1536, 16, '5 heads, n=16'),
    ('E75h5n24', 1408, 24, '5 heads, n=24'),
    ('E75h6n20', 1408, 20, '6 heads, n=20'),
    ('E75h6n24', 1280, 24, '6 heads, n=24'),
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

def run_wave(models, output_dir, gpus, wave_name):
    """Run a wave of models in parallel."""
    print(f"\n{'='*60}")
    print(f"Starting {wave_name}")
    print(f"{'='*60}")

    processes = []

    for i, (name, dim, n_state, desc) in enumerate(models):
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

        print(f"[GPU {gpu}] {name}: {desc}")

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

    print(f"\nAll {len(processes)} processes started. Waiting...")

    # Wait for all to complete
    for name, proc, log_file in processes:
        proc.wait()
        ret = proc.returncode
        status = "OK" if ret == 0 else f"FAILED ({ret})"
        print(f"  {name}: {status}")

    return processes

def print_results(output_dir, models):
    """Print results summary."""
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")

    results = []
    for name, _, _, desc in models:
        log_file = f'{output_dir}/{name}.log'
        try:
            # Get final checkpoint
            import glob
            chk_pattern = f'{output_dir}/{name}/level*/checkpoint_*.pt'
            chks = glob.glob(chk_pattern)
            if chks:
                latest = max(chks, key=os.path.getmtime)
                loss = os.path.basename(latest).split('loss_')[1].split('.pt')[0]
                steps = os.path.basename(latest).split('step_')[1].split('_')[0].lstrip('0') or '0'
                results.append((name, steps, loss, desc))
            else:
                results.append((name, '-', 'FAILED', desc))
        except Exception as e:
            results.append((name, '-', f'ERROR: {e}', desc))

    # Sort by loss
    results.sort(key=lambda x: float(x[2]) if x[2].replace('.','').isdigit() else 999)

    print(f"\n{'Model':<15} {'Steps':>6} {'Loss':>8} {'Description'}")
    print("-" * 60)
    for name, steps, loss, desc in results:
        print(f"{name:<15} {steps:>6} {loss:>8} {desc}")

    return results

def main():
    import sys

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'benchmark_results/e75_paramscan_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    gpus = get_available_gpus()
    print(f"Found {len(gpus)} GPUs: {gpus}")
    print(f"Output directory: {output_dir}")

    # Check for wave argument
    if len(sys.argv) > 1:
        wave = sys.argv[1]
        if wave == '1':
            run_wave(WAVE1_MODELS, output_dir, gpus, "Wave 1")
            print_results(output_dir, WAVE1_MODELS)
        elif wave == '2':
            run_wave(WAVE2_MODELS, output_dir, gpus, "Wave 2")
            print_results(output_dir, WAVE2_MODELS)
        else:
            print(f"Unknown wave: {wave}. Use 1 or 2.")
    else:
        # Run both waves sequentially
        run_wave(WAVE1_MODELS, output_dir, gpus, "Wave 1")
        run_wave(WAVE2_MODELS, output_dir, gpus, "Wave 2")
        print_results(output_dir, WAVE1_MODELS + WAVE2_MODELS)

    print(f"\n{'='*60}")
    print("Benchmark Complete!")
    print(f"Results in: {output_dir}")

if __name__ == '__main__':
    main()
