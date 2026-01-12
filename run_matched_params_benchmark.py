#!/usr/bin/env python3
"""
8-way parallel benchmark with MATCHED parameters (~50M each).

All models configured to have approximately 50M parameters.
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path


def run_benchmark():
    BATCH = 32
    SEQ = 512
    TIMEOUT = 600  # 10 minutes
    SEED = 42

    # Matched ~50M param configs: (name, level, dim, depth)
    MODELS = [
        ("E0", "0", 1280, 6),        # 49.5M
        ("E1", "1", 1280, 6),        # 49.5M
        ("E5", "5", 3328, 6),        # 50.7M (low-rank needs larger dim)
        ("E16", "16", 1088, 6),      # 50.0M (diagonal state)
        ("E21", "21", 704, 6),       # 50.3M (structured needs smaller dim)
        ("E28", "28", 1280, 6),      # 49.5M
        ("E30", "30", 1280, 6),      # 49.5M
        ("Mamba2", "mamba2", 0, 0),  # 50.9M (auto-configured)
    ]

    output_dir = Path("benchmark_results/matched_50m_10min")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Matched-Params Benchmark (~50M each)")
    print(f"Timeout: {TIMEOUT}s (10 minutes)")
    print(f"Seed: {SEED}")
    print(f"Output: {output_dir}")
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
import torch.nn.functional as F

sys.path.insert(0, "{os.getcwd()}")

import warnings
warnings.filterwarnings("ignore")

from elman.models.ladder_lm import LadderLM
from elman.data.dataset import DocumentStreamDataset
from schedulefree import AdamWScheduleFree

BATCH = {BATCH}
SEQ = {SEQ}
TIMEOUT = {TIMEOUT}
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

print(f"Config: dim={{DIM}}, depth={{DEPTH}}, batch={{BATCH}}, seq={{SEQ}}")
print()
print("=" * 60)
print(f"{{NAME}}")
print("=" * 60)
print(f"Parameters: {{num_params:,}} ({{num_params/1e6:.1f}}M)")

optimizer = AdamWScheduleFree(model.parameters(), lr=3e-4, weight_decay=0.1)

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
        print(f"  Step {{step}}: loss = {{loss.item():.4f}}")

elapsed = time.time() - start_time
final_loss = sum(losses[-100:]) / min(100, len(losses))
throughput = tokens_seen / elapsed

print(f"{{NAME}}: loss={{final_loss:.4f}}, throughput={{throughput/1000:.1f}}K tok/s, params={{num_params/1e6:.1f}}M")
'''
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        with open(log_path, 'w') as f:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
            processes.append((name, proc, log_path))
            print(f"Started {name} on GPU {gpu_id} (dim={dim}, depth={depth})")

    print()
    print("Waiting for completion...")
    print()

    for name, proc, _ in processes:
        proc.wait()
        print(f"Completed: {name}")

    print()
    print("=" * 60)
    print("RESULTS (Matched ~50M params)")
    print("=" * 60)

    results = []
    for name, _, log_path in processes:
        try:
            with open(log_path) as f:
                content = f.read()
            for line in content.split('\n')[::-1]:
                if f"{name}:" in line and "loss=" in line:
                    parts = line.split("loss=")[1].split(",")
                    loss = float(parts[0])
                    throughput = float(parts[1].split("=")[1].replace("K tok/s", "").strip())
                    params = float(parts[2].split("=")[1].replace("M", "").strip())
                    results.append((name, loss, throughput, params))
                    break
        except Exception as e:
            print(f"Error parsing {name}: {e}")

    results.sort(key=lambda x: x[1])

    print(f"{'Model':<8} {'Params':>8} {'Loss':>8} {'Throughput':>14}")
    print("-" * 42)
    for name, loss, throughput, params in results:
        print(f"{name:<8} {params:>6.1f}M {loss:>8.4f} {throughput:>10.1f}K tok/s")

    summary = {
        "config": {"batch": BATCH, "seq": SEQ, "timeout": TIMEOUT, "seed": SEED},
        "results": [{"model": n, "params_m": p, "loss": l, "throughput_ktps": t} for n, l, t, p in results]
    }
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    run_benchmark()
