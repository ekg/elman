#!/usr/bin/env python3
"""
Run aligned benchmark for all promising model levels.
Ensures dimensions are 128-aligned for optimal GPU performance.
"""

import subprocess
import os
import sys
from pathlib import Path

# Models to benchmark (level -> description)
MODELS = {
    1: "E1: Mamba-gated (baseline)",
    33: "E33: Self-gate",
    42: "E42: Linear tied",
    44: "E44: Diagonal W_h",
    51: "E51: No self-gate",
    52: "E52: Quadratic gate",
    53: "E53: Sigmoid gate",
    56: "E56: Concat Elman",
    59: "E59: Highway (no W_h)",
    60: "E60: Residual nonlinear",
    "mamba2": "Mamba2 baseline",
}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', default='100m', help='Target params (e.g., 100m)')
    parser.add_argument('--minutes', type=int, default=10, help='Training minutes')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--chunk-size', type=int, default=256)
    parser.add_argument('--output-dir', default='benchmark_results/aligned_sweep')
    parser.add_argument('--gpus', type=int, default=8, help='Number of GPUs')
    parser.add_argument('--dry-run', action='store_true', help='Print commands only')
    args = parser.parse_args()

    # Dimension configs that ensure 128-alignment
    # dim must be 128-aligned, and dim*expansion must also be 128-aligned
    # expansion=1.5 means d_inner = dim * 1.5, so dim must be even multiple of 128
    # expansion=2.0 means d_inner = dim * 2.0, so dim can be any multiple of 128
    DIM_CONFIGS = {
        '50m': (512, 1.5),   # 512 * 1.5 = 768 ✓
        '100m': (768, 1.5),  # 768 * 1.5 = 1152 ✓
        '200m': (1024, 1.5), # 1024 * 1.5 = 1536 ✓
        '500m': (1280, 2.0), # 1280 * 2 = 2560 ✓
        '1b': (1536, 2.0),   # 1536 * 2 = 3072 ✓
    }

    params = args.params.lower()
    if params not in DIM_CONFIGS:
        print(f"Warning: {params} not in DIM_CONFIGS, using default")

    dim, expansion = DIM_CONFIGS.get(params, (768, 1.5))
    d_inner = int(dim * expansion)

    print(f"Benchmark config:")
    print(f"  Target params: {params}")
    print(f"  Base dim: {dim} (128-aligned: {dim % 128 == 0})")
    print(f"  Expansion: {expansion}")
    print(f"  d_inner: {d_inner} (128-aligned: {d_inner % 128 == 0})")
    print(f"  Training: {args.minutes} minutes")
    print(f"  GPUs: {args.gpus}")
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build commands
    models = list(MODELS.items())
    commands = []

    for i, (level, desc) in enumerate(models):
        gpu_id = i % args.gpus
        log_file = output_dir / f"level_{level}.log"

        cmd = (
            f"CUDA_VISIBLE_DEVICES={gpu_id} python train.py "
            f"--data data/pile.txt "
            f"--level {level} "
            f"--params {params} "
            f"--train_minutes {args.minutes} "
            f"--batch_size {args.batch_size} "
            f"--chunk_size {args.chunk_size} "
            f"2>&1 | tee {log_file}"
        )
        commands.append((level, desc, gpu_id, cmd))

    # Print plan
    print(f"Will run {len(commands)} models:")
    for level, desc, gpu, cmd in commands:
        print(f"  GPU {gpu}: {desc}")
    print()

    if args.dry_run:
        print("Commands (dry run):")
        for level, desc, gpu, cmd in commands:
            print(cmd)
        return

    # Run in batches of GPU count
    import time
    batch_size = args.gpus

    for batch_start in range(0, len(commands), batch_size):
        batch = commands[batch_start:batch_start + batch_size]
        print(f"\n{'='*60}")
        print(f"Starting batch {batch_start//batch_size + 1}: {len(batch)} models")
        print('='*60)

        # Launch all in batch
        procs = []
        for level, desc, gpu, cmd in batch:
            print(f"  Launching {desc} on GPU {gpu}")
            proc = subprocess.Popen(cmd, shell=True)
            procs.append((level, desc, proc))

        # Wait for batch to complete
        print(f"\nWaiting for batch to complete...")
        for level, desc, proc in procs:
            proc.wait()
            print(f"  {desc}: exit code {proc.returncode}")

    print("\n" + "="*60)
    print("All benchmarks complete!")
    print("="*60)

    # Summarize results
    print("\nResults summary:")
    print("-" * 80)
    print(f"{'Model':<35} {'Final Loss':>12} {'Steps':>8} {'Tok/s':>10}")
    print("-" * 80)

    for level, desc in MODELS.items():
        log_file = output_dir / f"level_{level}.log"
        if log_file.exists():
            with open(log_file) as f:
                lines = f.readlines()

            # Find last step line and final info
            last_loss = "N/A"
            last_step = "N/A"
            last_toks = "N/A"

            for line in reversed(lines):
                if "step" in line and "loss" in line:
                    parts = line.split("|")
                    for p in parts:
                        if "loss" in p:
                            last_loss = p.split()[-1]
                        if "step" in p:
                            last_step = p.split()[-1]
                        if "tok/s" in p:
                            last_toks = p.split()[-1]
                    break

            print(f"{desc:<35} {last_loss:>12} {last_step:>8} {last_toks:>10}")

    print("-" * 80)


if __name__ == "__main__":
    main()
