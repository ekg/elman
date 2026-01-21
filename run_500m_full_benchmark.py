#!/usr/bin/env python3
"""
500M Parameter Full Benchmark - All Models
Compares E88 FLA Hybrid variants with established baselines.
"""

import subprocess
import os
import time
from datetime import datetime

# Benchmark settings
TRAIN_MINUTES = 10
BATCH_SIZE = 16
CHUNK_SIZE = 512
LR = 1e-4
WARMUP = 1000
SEED = 42
DATA = "data/pile.txt"

# Output directory
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_BASE = f"benchmark_results/500m_full_{TIMESTAMP}"

# Model configs targeting ~500M params
# Format: (name, level, dim, depth, extra_args, state_size_comment)
MODELS = [
    # === Established Baselines ===
    # Mamba2: d_model=1600, depth=32, d_state=128 (default), expand=2
    # State per layer: d_state * d_model * expand = 128 * 1600 * 2 = 409,600
    ("mamba2", "mamba2", None, None, "--params 500m", "state=409K/layer"),

    # FLA-GDN: dim=1536, depth=20, head_dim=128, expand=2
    # State per layer: (dim*expand/head_dim) * head_dim^2 = 24 * 16384 = 393,216
    ("fla-gdn", "fla-gdn", 1536, 20, "", "state=393K/layer"),

    # CUDA LSTM: dim=1280, depth=20 (4 gates * hidden^2 = 4 * 1280^2 per layer)
    ("cudalstm", "cudalstm", 1280, 20, "", "state=6.5M/layer"),

    # CUDA GRU: dim=1536, depth=20 (3 gates)
    ("cudagru", "cudagru", 1536, 20, "", "state=7.1M/layer"),

    # === E88 FLA Hybrid Variants ===
    # E88_h40n64: 40 heads × 64 state = 163,840 state (matches Mamba2 effective)
    # d_inner=2560, dim=2432 for ~500M
    ("e88_h40n64", "E88_h40n64", 2432, 20, "", "state=164K/layer"),

    # E88_h96n32: 96 heads × 32² = 98,304 state
    ("e88_h96n32", "E88_h96n32", 2048, 20, "", "state=98K/layer"),

    # E88_h80n32: 80 heads × 32² = 81,920 state (1/2 Mamba2)
    ("e88_h80n32", "E88_h80n32", 2560, 20, "", "state=82K/layer"),

    # E88_h128n32: 128 heads × 32² = 131,072 state
    ("e88_h128n32", "E88_h128n32", 2048, 20, "", "state=131K/layer"),

    # E88_h64n48: 64 heads × 48² = 147,456 state
    ("e88_h64n48", "E88_h64n48", 2304, 20, "", "state=147K/layer"),

    # === E88 with n_state=64 (larger per-head state) ===
    # E88_h64n64: 64 heads × 64² = 262,144 state
    ("e88_h64n64", "E88_h64n64", 2048, 20, "", "state=262K/layer"),

    # E88_h32n64: 32 heads × 64² = 131,072 state
    ("e88_h32n64", "E88_h32n64", 2560, 20, "", "state=131K/layer"),

    # E88_h24n64: 24 heads × 64² = 98,304 state
    ("e88_h24n64", "E88_h24n64", 2688, 20, "", "state=98K/layer"),
]

def run_benchmark(name, level, dim, depth, extra_args, state_comment, gpu_id):
    """Run a single benchmark."""
    output_dir = f"{OUTPUT_BASE}/{name}"
    log_file = f"{OUTPUT_BASE}/{name}.log"

    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python -u train.py --level {level}"
    if dim is not None:
        cmd += f" --dim {dim}"
    if depth is not None:
        cmd += f" --depth {depth}"
    cmd += f" --batch_size {BATCH_SIZE} --grad_accum 1 --chunk_size {CHUNK_SIZE}"
    cmd += f" --lr {LR} --warmup_steps {WARMUP} --train_minutes {TRAIN_MINUTES}"
    cmd += f" --data {DATA} --seed {SEED} --output {output_dir} --bf16"
    if extra_args:
        cmd += f" {extra_args}"

    # Run with nohup
    full_cmd = f"nohup {cmd} > {log_file} 2>&1 &"
    print(f"[GPU {gpu_id}] Starting {name} ({state_comment})")
    subprocess.run(full_cmd, shell=True)
    return log_file

def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    print(f"Starting 500M Full Benchmark")
    print(f"Output: {OUTPUT_BASE}")
    print(f"Models: {len(MODELS)}")
    print()

    # Save config
    with open(f"{OUTPUT_BASE}/config.txt", "w") as f:
        f.write(f"500M Full Benchmark - {TIMESTAMP}\n")
        f.write(f"Train minutes: {TRAIN_MINUTES}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Chunk size: {CHUNK_SIZE}\n")
        f.write(f"Learning rate: {LR}\n")
        f.write(f"Warmup steps: {WARMUP}\n")
        f.write(f"Seed: {SEED}\n\n")
        f.write("Models:\n")
        for name, level, dim, depth, extra, state in MODELS:
            f.write(f"  {name}: level={level}, dim={dim}, depth={depth}, {state}\n")

    # Launch in waves of 8 (8 GPUs available)
    NUM_GPUS = 8
    logs = []

    for wave_start in range(0, len(MODELS), NUM_GPUS):
        wave = MODELS[wave_start:wave_start + NUM_GPUS]
        print(f"\n=== Wave {wave_start // NUM_GPUS + 1} ({len(wave)} models) ===")

        for i, (name, level, dim, depth, extra, state) in enumerate(wave):
            gpu_id = i
            log = run_benchmark(name, level, dim, depth, extra, state, gpu_id)
            logs.append((name, log))

        # Wait for this wave to complete
        print(f"Waiting for wave to complete (max {TRAIN_MINUTES} min)...")
        time.sleep(TRAIN_MINUTES * 60 + 60)  # Extra minute buffer

    print(f"\n=== Benchmark Complete ===")
    print(f"Results in: {OUTPUT_BASE}/")

if __name__ == "__main__":
    main()
