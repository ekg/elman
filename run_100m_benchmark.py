#!/usr/bin/env python3
"""
Dynamic GPU scheduler for 100M parameter benchmarks.
Runs multiple training jobs in parallel, dynamically allocating GPUs as they become free.
"""

import subprocess
import time
import os
import sys
from pathlib import Path
from datetime import datetime

# Configuration
DATA_PATH = "/home/erikg/elman/data/pile.txt"
TRAIN_MINUTES = 10
BATCH_SIZE = 16
CHUNK_SIZE = 512
LR = 1e-4
WARMUP_STEPS = 1000
LOG_EVERY = 50
SEED = 42
OUTPUT_DIR = Path("/home/erikg/elman/benchmark_results/100m_systematic")

# Models to benchmark: (level, name, dim, depth) - all targeting ~100M params
# For models where --params 100m works well, dim/depth are None (auto)
MODELS = [
    # Tier 1: Essential baselines (auto-config works)
    (1, "E1_gated_elman", None, None),           # ~101M
    (42, "E42_linear_tied", None, None),         # ~99M
    ("mamba2", "Mamba2_ssm", None, None),        # ~101M

    # Tier 2: Standard models (auto-config works)
    (33, "E33_self_gate", None, None),           # ~102M
    (51, "E51_no_self_gate", None, None),        # ~99M
    (52, "E52_quadratic_gate", None, None),      # ~99M
    (53, "E53_sigmoid_gate", None, None),        # ~99M
    (56, "E56_concat_elman", None, None),        # ~101M

    # Tier 3: Models needing custom dim/depth for 100M
    (43, "E43_scalar_decay", 768, 84),           # ~99M (default 48 layers was 85M)
    (44, "E44_diagonal_w", 768, 84),             # ~99M
    (45, "E45_pure_accumulation", 768, 85),      # ~100M
    (46, "E46_no_in_proj", 768, 84),             # ~99M (no in_proj -> fewer params/layer)
    (48, "E48_no_projections", 1024, 95),        # ~100M (minimal -> needs 1024 dim)

    # Note: E54/E55 excluded - they're ultra-minimal (<5M params even at extreme configs)
    # They use scalar/diagonal decay with no projections, fundamentally different design
]


def get_free_gpus(min_memory_mb=40000):
    """Get list of GPU indices with at least min_memory_mb free."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        free_gpus = []
        for line in result.stdout.strip().split("\n"):
            if line:
                idx, mem = line.split(",")
                if int(mem.strip()) >= min_memory_mb:
                    free_gpus.append(int(idx.strip()))
        return free_gpus
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return []


def get_gpu_processes():
    """Get set of GPUs currently running processes."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        # Map GPU UUIDs to indices
        uuid_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        uuid_to_idx = {}
        for line in uuid_result.stdout.strip().split("\n"):
            if line:
                parts = line.split(",")
                if len(parts) >= 2:
                    uuid_to_idx[parts[1].strip()] = int(parts[0].strip())

        busy_gpus = set()
        for line in result.stdout.strip().split("\n"):
            if line:
                parts = line.split(",")
                if len(parts) >= 1:
                    uuid = parts[0].strip()
                    if uuid in uuid_to_idx:
                        busy_gpus.add(uuid_to_idx[uuid])
        return busy_gpus
    except Exception as e:
        return set()


def build_command(level, name, gpu_id, dim=None, depth=None):
    """Build the training command for a model."""
    log_file = OUTPUT_DIR / f"{name}.log"

    cmd = [
        "python", "/home/erikg/elman/train.py",
        "--data", DATA_PATH,
        "--level", str(level),
        "--batch_size", str(BATCH_SIZE),
        "--chunk_size", str(CHUNK_SIZE),
        "--train_minutes", str(TRAIN_MINUTES),
        "--lr", str(LR),
        "--warmup_steps", str(WARMUP_STEPS),
        "--log_every", str(LOG_EVERY),
        "--bf16",
        "--seed", str(SEED),
        "--output", str(OUTPUT_DIR / name),
    ]

    # Use custom dim/depth if provided, otherwise auto-target 100M
    if dim is not None and depth is not None:
        cmd.extend(["--dim", str(dim), "--depth", str(depth)])
    else:
        cmd.extend(["--params", "100m"])

    return cmd, log_file


def run_job(level, name, gpu_id, dim=None, depth=None):
    """Start a training job on a specific GPU."""
    cmd, log_file = build_command(level, name, gpu_id, dim, depth)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting {name} on GPU {gpu_id}")

    with open(log_file, "w") as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            cwd="/home/erikg/elman"
        )

    return process, gpu_id, name, log_file


def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("100M Parameter Systematic Benchmark")
    print(f"Models to test: {len(MODELS)}")
    print(f"Training time per model: {TRAIN_MINUTES} minutes")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)

    # Queue of models to run
    pending = list(MODELS)
    running = []  # List of (process, gpu_id, name, log_file)
    completed = []
    failed = []

    start_time = time.time()

    while pending or running:
        # Check for completed jobs
        still_running = []
        for proc, gpu_id, name, log_file in running:
            ret = proc.poll()
            if ret is None:
                still_running.append((proc, gpu_id, name, log_file))
            elif ret == 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Completed {name} (GPU {gpu_id})")
                completed.append(name)
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] FAILED {name} (GPU {gpu_id}, exit code {ret})")
                failed.append(name)
        running = still_running

        # Find free GPUs
        busy_gpus = {gpu_id for _, gpu_id, _, _ in running}
        all_gpus = set(range(8))
        free_gpus = sorted(all_gpus - busy_gpus)

        # Start new jobs on free GPUs
        while pending and free_gpus:
            gpu_id = free_gpus.pop(0)
            level, name, dim, depth = pending.pop(0)
            job = run_job(level, name, gpu_id, dim, depth)
            running.append(job)

        # Status update
        if running:
            running_names = [name for _, _, name, _ in running]
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Running: {len(running)}, Pending: {len(pending)}, Completed: {len(completed)}, Failed: {len(failed)}")
            print(f"  Active: {', '.join(running_names)}")

        # Wait before checking again
        if pending or running:
            time.sleep(30)

    # Final summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Completed: {len(completed)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print(f"Failed models: {', '.join(failed)}")
    print("=" * 70)

    # Parse results
    print("\nParsing results...")
    results = []
    for level, name, dim, depth in MODELS:
        log_file = OUTPUT_DIR / f"{name}.log"
        if log_file.exists():
            try:
                with open(log_file) as f:
                    lines = f.readlines()

                # Extract final metrics
                params = None
                final_loss = None
                throughput = None

                for line in lines:
                    if "Total parameters:" in line:
                        params = line.split(":")[-1].strip()
                    # Look for last training step with loss and tok/s
                    if "step" in line and "loss" in line and "tok/s" in line:
                        parts = line.split()
                        for i, p in enumerate(parts):
                            if p == "loss":
                                try:
                                    final_loss = float(parts[i+1].rstrip(","))
                                except:
                                    pass
                            if "tok/s" in p or (i > 0 and parts[i-1].replace(",", "").replace(".", "").isdigit() and "tok/s" in line):
                                try:
                                    # Find the number before tok/s
                                    for j in range(i-1, -1, -1):
                                        val = parts[j].rstrip(",")
                                        if val.replace(".", "").replace(",", "").isdigit():
                                            throughput = float(val.replace(",", ""))
                                            break
                                except:
                                    pass

                results.append({
                    "name": name,
                    "level": level,
                    "params": params,
                    "loss": final_loss,
                    "throughput": throughput
                })
            except Exception as e:
                print(f"Error parsing {name}: {e}")

    # Print results table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Model':<25} {'Params':<12} {'Loss':<10} {'Throughput':<15}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x.get("loss") or 999):
        params = r.get("params", "N/A")
        loss = f"{r['loss']:.4f}" if r.get("loss") else "N/A"
        tput = f"{r['throughput']:.0f} tok/s" if r.get("throughput") else "N/A"
        print(f"{r['name']:<25} {params:<12} {loss:<10} {tput:<15}")

    # Save results to JSON
    import json
    results_file = OUTPUT_DIR / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
