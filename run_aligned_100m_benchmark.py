#!/usr/bin/env python3
"""
100M Parameter Benchmark with CUDA-Aligned Dimensions

Uses dim=768 (divisible by 128) for optimal GPU utilization.
Different models need different depths to reach ~100M params.
"""

import subprocess
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

NUM_GPUS = 8
SEED = 42
TRAIN_MINUTES = 10
DIM = 768  # CUDA-aligned: 768 % 128 = 0
EXPANSION = 1.5
BATCH_SIZE = 16
CHUNK_SIZE = 512

# Model configs: (level, depth, name, description)
# Depths calculated to hit ~100M params with dim=768, expansion=1.5
# All verified to be within 2% of 100M target
MODELS = [
    # E1: 5.31M/layer → depth=19 → 101M params
    (1, 19, "E1", "gated elman"),

    # E33: 4.43M/layer → depth=23 → 102M params
    (33, 23, "E33", "multi-gate"),

    # E42: 3.10M/layer (tied weights) → depth=32 → 99M params
    (42, 32, "E42", "linear tied"),

    # E44: 1.77M/layer (diagonal W_h) → depth=56 → 99M params
    (44, 56, "E44", "diagonal W_h"),

    # E51: 3.10M/layer (no self-gate) → depth=32 → 99M params
    (51, 32, "E51", "no self-gate"),

    # E56: 5.31M/layer (softsign) → depth=19 → 101M params
    (56, 19, "E56", "softsign"),

    # E59: 3.10M/layer (highway) → depth=32 → 99M params
    (59, 32, "E59", "highway"),

    # E60: 4.43M/layer (residual) → depth=23 → 102M params
    (60, 23, "E60", "residual nonlinear"),

    # Mamba2: 3.77M/layer (d_state=128) → depth=27 → 102M params
    ("mamba2", 27, "Mamba2", "baseline"),
]

OUTPUT_DIR = "benchmark_results/aligned_100m"


def run_model(gpu_id, level, depth, name, desc):
    """Run a single model on a specific GPU."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_file = f"{OUTPUT_DIR}/{level}.log"

    # Build command
    if level == "mamba2":
        cmd = [
            "python", "train.py",
            "--data", "data/pile.txt",
            "--level", "mamba2",
            "--dim", str(DIM),
            "--depth", str(depth),
            "--batch_size", str(BATCH_SIZE),
            "--chunk_size", str(CHUNK_SIZE),
            "--train_minutes", str(TRAIN_MINUTES),
            "--seed", str(SEED),
            "--output", f"output/aligned_{level}_{DIM}x{depth}",
        ]
    else:
        cmd = [
            "python", "train.py",
            "--data", "data/pile.txt",
            "--level", str(level),
            "--dim", str(DIM),
            "--expansion", str(EXPANSION),
            "--depth", str(depth),
            "--batch_size", str(BATCH_SIZE),
            "--chunk_size", str(CHUNK_SIZE),
            "--train_minutes", str(TRAIN_MINUTES),
            "--seed", str(SEED),
            "--output", f"output/aligned_{level}_{DIM}x{depth}",
        ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"  Starting GPU {gpu_id}: {name} ({desc}) (level={level}, d={DIM}, depth={depth})")

    with open(log_file, "w") as f:
        result = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)

    status = "OK" if result.returncode == 0 else f"FAILED (code {result.returncode})"
    print(f"  Completed GPU {gpu_id}: {name} ({desc}) - {status}")
    return level, status


def main():
    print(f"Running {len(MODELS)} models on {NUM_GPUS} GPUs")
    print(f"All models use dim={DIM} (CUDA-aligned: {DIM} % 128 = {DIM % 128})")
    print(f"Each model trains for {TRAIN_MINUTES} minutes")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Split into batches of NUM_GPUS
    batches = [MODELS[i:i+NUM_GPUS] for i in range(0, len(MODELS), NUM_GPUS)]

    for batch_idx, batch in enumerate(batches, 1):
        print(f"\nBatch {batch_idx}: {len(batch)} models")
        print("-" * 40)

        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = []
            for gpu_id, (level, depth, name, desc) in enumerate(batch):
                future = executor.submit(run_model, gpu_id, level, depth, name, desc)
                futures.append(future)

            # Wait for all in batch to complete
            print(f"\nWaiting for batch {batch_idx} to complete...")
            for future in as_completed(futures):
                try:
                    level, status = future.result()
                except Exception as e:
                    print(f"  Error: {e}")

    print("\n" + "=" * 60)
    print("All benchmarks complete!")
    print(f"\nResults in: {OUTPUT_DIR}/")
    print(f"Analyze with: python analyze_aligned_benchmark.py")


if __name__ == "__main__":
    main()
