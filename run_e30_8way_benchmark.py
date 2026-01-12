#!/usr/bin/env python3
"""
8-way parallel benchmark: E30 vs key Elman variants and Mamba2.

Runs all models in parallel on 8 GPUs with identical seeded data loading.
10 minutes per model, byte-level (vocab=256), reports last-100 averaged loss.
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path


def run_benchmark():
    # Config matching CLAUDE.md best settings
    DIM = 1280
    DEPTH = 6
    BATCH = 32
    SEQ = 512
    TIMEOUT = 600  # 10 minutes
    SEED = 42  # Same seed for all models for fair comparison

    # Key models to compare
    MODELS = [
        ("E0", "0", {}),           # Stock Elman baseline
        ("E1", "1", {}),           # Best Elman (gated)
        ("E5", "5", {}),           # Low-rank
        ("E28", "28", {}),         # Conv Elman
        ("E30", "30", {}),         # NEW: Diagonal gated
        ("Mamba2", "mamba2", {}),  # External baseline
        ("E16", "16", {}),         # Diagonal state
        ("E21", "21", {}),         # Structured Elman
    ]

    # Output directory
    output_dir = Path("benchmark_results/e30_8way_10min")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"8-Way Parallel Benchmark")
    print(f"Config: dim={DIM}, depth={DEPTH}, batch={BATCH}, seq={SEQ}")
    print(f"Timeout: {TIMEOUT}s (10 minutes)")
    print(f"Seed: {SEED} (same for all)")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Launch all processes in parallel
    processes = []
    log_files = {}

    for gpu_id, (name, level, extra_args) in enumerate(MODELS):
        if gpu_id >= 8:
            print(f"Warning: Skipping {name}, only 8 GPUs available")
            continue

        log_path = output_dir / f"{name}.log"
        log_files[name] = log_path

        # Build command
        cmd = [
            sys.executable, "-c", f'''
import sys
import os
import time
import math
import torch
import torch.nn.functional as F
from datetime import datetime

sys.path.insert(0, "{os.getcwd()}")
from elman.models.ladder_lm import LadderLM
from elman.data.dataset import DocumentStreamDataset
from schedulefree import AdamWScheduleFree

# Suppress Triton warnings
import warnings
warnings.filterwarnings("ignore")

# Config
DIM = {DIM}
DEPTH = {DEPTH}
BATCH = {BATCH}
SEQ = {SEQ}
TIMEOUT = {TIMEOUT}
SEED = {SEED}
LEVEL = "{level}"
NAME = "{name}"

device = torch.device("cuda:0")
dtype = torch.bfloat16

print(f"Config: dim={{DIM}}, depth={{DEPTH}}, batch={{BATCH}}, seq={{SEQ}}")
print(f"Data: data/pile.txt")
print()

# Create dataset with fixed seed
dataset = DocumentStreamDataset("data/pile.txt", SEQ, rank=0, world_size=1, seed=SEED)

# Create model
if LEVEL == "mamba2":
    from elman.models.mamba2_baseline import create_mamba2_model
    model = create_mamba2_model(vocab_size=256, target_params="50m")
else:
    model = LadderLM(
        vocab_size=256,
        dim=DIM,
        depth=DEPTH,
        level=int(LEVEL) if LEVEL.isdigit() else LEVEL,
        mamba2_init=True,
    )
model = model.to(device=device, dtype=dtype)
num_params = model.get_num_params()

print("=" * 60)
print(f"{{NAME}}")
print("=" * 60)
print(f"Parameters: {{num_params/1e6:.1f}}M")

# Optimizer
optimizer = AdamWScheduleFree(model.parameters(), lr=3e-4, weight_decay=0.1)

# Training loop
model.train()
optimizer.train()
start_time = time.time()
tokens_seen = 0
step = 0
losses = []

while True:
    step += 1
    elapsed = time.time() - start_time
    if elapsed >= TIMEOUT:
        break

    # Get batch
    batch_chunks = []
    for _ in range(BATCH):
        chunk, _, _ = dataset[0]
        batch_chunks.append(chunk)
    batch = torch.stack(batch_chunks).to(device)

    optimizer.zero_grad()
    loss = model(batch, return_loss=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    tokens_seen += BATCH * SEQ
    losses.append(loss.item())

    if step % 50 == 0 or step == 1:
        tps = tokens_seen / elapsed
        print(f"  Step {{step}}: loss = {{loss.item():.4f}}")

elapsed = time.time() - start_time
final_loss = sum(losses[-100:]) / min(100, len(losses))
throughput = tokens_seen / elapsed

print(f"{{NAME}}: loss={{final_loss:.4f}}, throughput={{throughput/1000:.1f}}K tok/s")
'''
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        with open(log_path, 'w') as f:
            proc = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env
            )
            processes.append((name, proc, log_path))
            print(f"Started {name} on GPU {gpu_id}")

    print()
    print("Waiting for all models to complete...")
    print()

    # Wait for all processes
    for name, proc, log_path in processes:
        proc.wait()
        print(f"Completed: {name}")

    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    # Parse results
    results = []
    for name, _, log_path in processes:
        try:
            with open(log_path) as f:
                content = f.read()
            # Find the final line with results
            for line in content.split('\n')[::-1]:
                if f"{name}:" in line and "loss=" in line:
                    # Parse: "E1: loss=1.4300, throughput=254.0K tok/s"
                    parts = line.split("loss=")[1].split(",")
                    loss = float(parts[0])
                    throughput = float(parts[1].split("=")[1].replace("K tok/s", "").strip())
                    results.append((name, loss, throughput))
                    break
        except Exception as e:
            print(f"Error parsing {name}: {e}")

    # Sort by loss
    results.sort(key=lambda x: x[1])

    print(f"{'Model':<12} {'Loss':>8} {'Throughput':>14}")
    print("-" * 36)
    for name, loss, throughput in results:
        print(f"{name:<12} {loss:>8.4f} {throughput:>10.1f}K tok/s")

    # Save summary
    summary = {
        "config": {"dim": DIM, "depth": DEPTH, "batch": BATCH, "seq": SEQ, "timeout": TIMEOUT, "seed": SEED},
        "results": [{"model": name, "loss": loss, "throughput_ktps": throughput} for name, loss, throughput in results]
    }
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Full logs saved to: {output_dir}")


if __name__ == "__main__":
    run_benchmark()
