#!/usr/bin/env python3
"""
Autopoetic Ladder Benchmark: E79 vs Matrix-State Family vs Baselines

All models configured for ~100M parameters with 128-aligned dimensions.
Matches E79 hyperparameters for fair comparison.
"""

import subprocess
import sys
import os
from datetime import datetime

# Benchmark configuration
TRAIN_MINUTES = 10
SEED = 42
BATCH_SIZE = 32
CHUNK_SIZE = 512
LR = 0.0003
DATA_PATH = "data/pile.txt"
OUTPUT_BASE = "benchmark_results/autopoetic_100m"

# Model configurations for ~100M params, 128-aligned dims
# Format: (level, dim, depth, n_state, expansion, extra_args)
# All configs verified to be 128-aligned and hit ~100M params
CONFIGS = {
    # Matrix-state models (E70-E79) - expansion=1.0, n_state=32
    "e79": (79, 2048, 22, 32, 1.0, ""),  # 100.06M - Coupled memory-modulation
    "e74v2": ("74v2-delta", 2176, 20, 32, 1.0, ""),  # 100.87M - Delta rule matrix memory
    "e73": (73, 2176, 20, 32, 1.0, ""),           # 100.87M - Nonlinear matrix with z-modulation
    "e70": (70, 2176, 20, 32, 1.0, ""),           # 100.87M - Linear matrix delta

    # Simpler baselines - expansion=2.0
    "e1": (1, 640, 17, None, 2.0, ""),            # 97.68M - Gated Elman baseline

    # External baselines
    "mamba2": ("mamba2", 896, 20, None, 2, ""),   # 99.63M - Mamba2 SSM
}

def run_benchmark(name, config, gpu_id):
    """Launch a single benchmark on specified GPU."""
    level, dim, depth, n_state, expansion, extra = config

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{OUTPUT_BASE}/{name}"
    log_file = f"{OUTPUT_BASE}/{name}.log"

    os.makedirs(output_dir, exist_ok=True)

    # Use appropriate Python for each model
    # - Python 3.10 (micromamba): Has CUDA extension for E-series
    # - Python 3.12 (system): Has mamba_ssm for Mamba2
    if str(level).startswith("mamba"):
        python_path = "/usr/bin/python3"
    else:
        python_path = "/home/erikg/micromamba/envs/wtf/bin/python3.10"
    cmd = [
        python_path, "train.py",
        "--data", DATA_PATH,
        "--level", str(level),
        "--dim", str(dim),
        "--depth", str(depth),
        "--expansion", str(expansion),
        "--train_minutes", str(TRAIN_MINUTES),
        "--output", output_dir,
        "--seed", str(SEED),
        "--batch_size", str(BATCH_SIZE),
        "--chunk_size", str(CHUNK_SIZE),
        "--lr", str(LR),
        "--bf16",
    ]

    if n_state is not None:
        cmd.extend(["--n_state", str(n_state)])

    # Add any extra model-specific args (can override defaults like --lr)
    if extra:
        cmd.extend(extra.split())

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[GPU {gpu_id}] Starting {name}: level={level}, dim={dim}, depth={depth}, n_state={n_state}")

    with open(log_file, "w") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            cwd="/home/erikg/elman"
        )

    return proc, name, log_file

def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # Get available GPUs
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        capture_output=True, text=True
    )
    gpus = [int(x.strip()) for x in result.stdout.strip().split("\n")]
    print(f"Available GPUs: {gpus}")

    models = list(CONFIGS.keys())
    if len(models) > len(gpus):
        print(f"Warning: {len(models)} models but only {len(gpus)} GPUs. Running in batches.")

    # Launch all benchmarks
    processes = []
    for i, name in enumerate(models):
        gpu_id = gpus[i % len(gpus)]
        proc, name, log_file = run_benchmark(name, CONFIGS[name], gpu_id)
        processes.append((proc, name, log_file))

    print(f"\nLaunched {len(processes)} benchmarks. Waiting for completion...")

    # Wait for all to complete
    for proc, name, log_file in processes:
        proc.wait()
        print(f"  {name} completed (exit code: {proc.returncode})")

    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)

    for name in models:
        log_file = f"{OUTPUT_BASE}/{name}.log"
        if os.path.exists(log_file):
            # Get last training line
            with open(log_file) as f:
                lines = f.readlines()

            # Find params
            params_line = [l for l in lines if "parameters" in l]
            params = params_line[0].strip() if params_line else "?"

            # Find last step
            step_lines = [l for l in lines if l.strip().startswith("step")]
            if step_lines:
                last = step_lines[-1].strip()
                print(f"{name:12s}: {last}")
            else:
                print(f"{name:12s}: No training data")
        else:
            print(f"{name:12s}: Log not found")

if __name__ == "__main__":
    main()
