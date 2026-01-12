#!/usr/bin/env python3
"""
Per-step benchmark: Compare models at fixed step count (500 steps).

Measures quality per step, not per second - shows which models learn faster.
"""

import os
import sys
import subprocess
import json
from pathlib import Path


def run_benchmark():
    BATCH = 32
    SEQ = 512
    MAX_STEPS = 500
    SEED = 42

    # Matched ~50M param configs: (name, level, dim, depth)
    MODELS = [
        ("E0", "0", 1280, 6),        # 49.5M
        ("E1", "1", 1280, 6),        # 49.5M
        ("E5", "5", 3328, 6),        # 50.7M
        ("E16", "16", 1088, 6),      # 50.0M
        ("E21", "21", 704, 6),       # 50.3M
        ("E25", "25", 1024, 8),      # 50.6M (entmax dual memory)
        ("E30", "30", 1280, 6),      # 49.5M
        ("Mamba2", "mamba2", 0, 0),  # 50.9M
    ]

    output_dir = Path("benchmark_results/perstep_500")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Per-Step Benchmark ({MAX_STEPS} steps, ~50M params each)")
    print(f"Seed: {SEED}")
    print("=" * 60)

    processes = []

    for gpu_id, (name, level, dim, depth) in enumerate(MODELS):
        if gpu_id >= 8:
            break

        log_path = output_dir / f"{name}.log"

        cmd = [
            sys.executable, "-c", f'''
import sys
import os
import time
import torch

sys.path.insert(0, "{os.getcwd()}")

import warnings
warnings.filterwarnings("ignore")

from elman.models.ladder_lm import LadderLM
from elman.data.dataset import DocumentStreamDataset
from schedulefree import AdamWScheduleFree

BATCH = {BATCH}
SEQ = {SEQ}
MAX_STEPS = {MAX_STEPS}
SEED = {SEED}
LEVEL = "{level}"
NAME = "{name}"
DIM = {dim}
DEPTH = {depth}

device = torch.device("cuda:0")
dtype = torch.bfloat16

dataset = DocumentStreamDataset("data/pile.txt", SEQ, rank=0, world_size=1, seed=SEED)

if LEVEL == "mamba2":
    from elman.models.mamba2_baseline import create_mamba2_model
    model = create_mamba2_model(vocab_size=256, target_params="50m")
    DIM = model.dim
    DEPTH = len(model.layers)
else:
    model = LadderLM(
        vocab_size=256,
        dim=DIM,
        depth=DEPTH,
        level=int(LEVEL),
        mamba2_init=True,
    )

model = model.to(device=device, dtype=dtype)
num_params = sum(p.numel() for p in model.parameters())

print(f"{{NAME}}: dim={{DIM}}, depth={{DEPTH}}, params={{num_params/1e6:.1f}}M")

optimizer = AdamWScheduleFree(model.parameters(), lr=3e-4, weight_decay=0.1)

model.train()
optimizer.train()
start_time = time.time()
tokens_seen = 0
losses = []
checkpoints = {{}}  # step -> loss

for step in range(1, MAX_STEPS + 1):
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

    # Record at key checkpoints
    if step in [50, 100, 200, 300, 400, 500]:
        avg = sum(losses[-50:]) / min(50, len(losses))
        checkpoints[step] = avg
        print(f"  Step {{step}}: loss={{avg:.4f}}")

elapsed = time.time() - start_time
final_loss = sum(losses[-50:]) / 50
throughput = tokens_seen / elapsed

print(f"RESULT|{{NAME}}|{{num_params/1e6:.1f}}|{{final_loss:.4f}}|{{throughput/1000:.1f}}|{{checkpoints}}")
'''
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        with open(log_path, 'w') as f:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
            processes.append((name, proc, log_path))
            print(f"Started {name} on GPU {gpu_id}")

    print()
    print("Waiting...")

    for name, proc, _ in processes:
        proc.wait()
        print(f"Done: {name}")

    print()
    print("=" * 60)
    print(f"RESULTS @ {MAX_STEPS} steps (matched ~50M params)")
    print("=" * 60)

    results = []
    for name, _, log_path in processes:
        try:
            with open(log_path) as f:
                for line in f:
                    if line.startswith("RESULT|"):
                        parts = line.strip().split("|")
                        n, params, loss, tput, ckpts = parts[1], float(parts[2]), float(parts[3]), float(parts[4]), eval(parts[5])
                        results.append((n, params, loss, tput, ckpts))
        except Exception as e:
            print(f"Error parsing {name}: {e}")

    results.sort(key=lambda x: x[2])

    print(f"{'Model':<8} {'Params':>7} {'Loss@500':>9} {'Throughput':>12}")
    print("-" * 40)
    for name, params, loss, tput, _ in results:
        print(f"{name:<8} {params:>5.1f}M {loss:>9.4f} {tput:>9.1f}K tok/s")

    # Show progression
    print()
    print("Loss progression:")
    print(f"{'Model':<8} {'@50':>8} {'@100':>8} {'@200':>8} {'@300':>8} {'@400':>8} {'@500':>8}")
    print("-" * 60)
    for name, _, _, _, ckpts in results:
        row = f"{name:<8}"
        for step in [50, 100, 200, 300, 400, 500]:
            row += f" {ckpts.get(step, 0):>7.4f}"
        print(row)

    summary = {
        "config": {"batch": BATCH, "seq": SEQ, "max_steps": MAX_STEPS, "seed": SEED},
        "results": [{"model": n, "params_m": p, "loss": l, "throughput_ktps": t, "checkpoints": c}
                    for n, p, l, t, c in results]
    }
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    run_benchmark()
