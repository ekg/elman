#!/usr/bin/env python3
"""Benchmark E59/E60 vs E1/E33/E42/E5x/Mamba2 at 100M params, 25 layers.

Runs in batches of 8 (one per GPU), waiting for each batch to complete.
"""

import subprocess
import os
import sys

# Target: ~4M params/layer, 25 layers, ~100M total
SEED = 42
LAYERS = 25
MINUTES = 10
NUM_GPUS = 7  # GPU 7 reserved for investigation agent

# Model configs: (level, dim, description)
# Dimensions calculated to hit ~100M params at 25 layers
MODELS = [
    # Batch 1: GPUs 0-7
    ("1", 536, "E1 (gated elman)"),
    ("33", 576, "E33 (multi-gate)"),
    ("42", 704, "E42 (linear tied)"),
    ("56", 536, "E56 (softsign)"),
    ("57", 536, "E57 (scalar learned radius)"),
    ("58", 536, "E58 (per-dim learned radii)"),
    ("59", 704, "E59 (highway)"),
    ("44", 1000, "E44 (diagonal W_h)"),

    # Batch 2: GPUs 0-4 (skip E51/E52/E53 until fixed, add mamba2)
    ("60", 576, "E60 (residual nonlinear)"),
    ("mamba2", 768, "Mamba2"),

    # Skip E51/E52/E53 - being investigated for GPU utilization issues
    # ("51", 704, "E51 (cubic gate)"),
    # ("52", 704, "E52 (squared gate)"),
    # ("53", 704, "E53 (abs gate)"),
]

def run_single_job(level, dim, desc, gpu_id, results_dir):
    """Run a single training job on specified GPU."""
    log_file = f"{results_dir}/{level}.log"

    cmd = [
        "python", "train.py",
        "--data=data/pile.txt",
        f"--level={level}",
        f"--dim={dim}",
        f"--depth={LAYERS}",
        f"--train_minutes={MINUTES}",
        f"--seed={SEED}",
        "--r_h_mode=auto",
        "--batch_size=16",
        "--chunk_size=512",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"  Starting GPU {gpu_id}: {desc} (level={level}, d={dim})", flush=True)

    with open(log_file, 'w') as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)

    return p, level, desc, gpu_id

def run_benchmark():
    """Launch benchmarks in batches of 8."""

    results_dir = "benchmark_results/e59_e60_comparison"
    os.makedirs(results_dir, exist_ok=True)

    # Clear old logs
    for level, _, _ in MODELS:
        log_file = f"{results_dir}/{level}.log"
        if os.path.exists(log_file):
            os.remove(log_file)

    print(f"Running {len(MODELS)} models on {NUM_GPUS} GPUs", flush=True)
    print(f"Each model trains for {MINUTES} minutes", flush=True)
    print("=" * 60, flush=True)

    # Process in batches of NUM_GPUS
    batch_num = 0
    for batch_start in range(0, len(MODELS), NUM_GPUS):
        batch_end = min(batch_start + NUM_GPUS, len(MODELS))
        batch = MODELS[batch_start:batch_end]
        batch_num += 1

        print(f"\nBatch {batch_num}: {len(batch)} models", flush=True)
        print("-" * 40, flush=True)

        # Start all jobs in this batch
        processes = []
        for i, (level, dim, desc) in enumerate(batch):
            gpu_id = i
            p, level, desc, gpu_id = run_single_job(level, dim, desc, gpu_id, results_dir)
            processes.append((p, level, desc, gpu_id))

        # Wait for all to complete
        print(f"\nWaiting for batch {batch_num} to complete...", flush=True)
        for p, level, desc, gpu_id in processes:
            ret = p.wait()
            status = "OK" if ret == 0 else f"FAIL({ret})"
            print(f"  Completed GPU {gpu_id}: {desc} - {status}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("All benchmarks complete!", flush=True)
    print(f"\nAnalyze with: python analyze_e59_e60_benchmark.py", flush=True)

if __name__ == "__main__":
    run_benchmark()
