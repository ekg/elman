#!/usr/bin/env python3
"""
E29 Benchmark: Compare Mamba2, E1, E29a, E29b on byte-level modeling (10 min each)

Runs 4 models in parallel on 4 GPUs (can use more configs on 8 GPUs).
CRITICAL: Same data loader seed for ALL processes for fair comparison.
"""

import subprocess
import os
import sys
import time
import json
import re
from pathlib import Path
from datetime import datetime

DATA_PATH = "/home/erikg/elman/data/pile.txt"
OUTPUT_DIR = Path("/home/erikg/elman/benchmark_results/e29_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Common parameters - matched to E1's best config (~50M params)
# E29 Python is very slow, so we use smaller batch for it
DIM = 1280
DEPTH = 6
BATCH_SIZE = 64
BATCH_SIZE_E29 = 16  # Smaller batch for E29 Python (4x slower per step but more steps)
SEQ_LEN = 512
TIME_LIMIT = 600  # 10 minutes
DATA_SEED = 42  # SAME SEED FOR ALL - critical for fair comparison

# Model configurations
# (model_type, name, extra_args)
# Using 8 GPUs for parallel runs
# E29 CUDA supports dim up to 1024, so use dim=1024 for E29 models
CONFIGS = [
    ("mamba2", "mamba2_d1024_depth8", {"batch_size": BATCH_SIZE, "dim": 1024, "depth": 8}),
    ("e1", "e1_d1024_depth8", {"batch_size": BATCH_SIZE, "dim": 1024, "depth": 8}),
    ("e29a", "e29a_d1024_depth8_n8", {"n_slots": 8, "batch_size": BATCH_SIZE, "dim": 1024, "depth": 8}),
    ("e29b", "e29b_d1024_depth8_n8", {"n_slots": 8, "batch_size": BATCH_SIZE, "dim": 1024, "depth": 8}),
    # Additional configs - varied depths/slots
    ("e1", "e1_d1024_depth6", {"batch_size": BATCH_SIZE, "dim": 1024, "depth": 6}),
    ("e29a", "e29a_d1024_depth6_n16", {"n_slots": 16, "batch_size": BATCH_SIZE, "dim": 1024, "depth": 6}),
    ("e29b", "e29b_d1024_depth6_n16", {"n_slots": 16, "batch_size": BATCH_SIZE, "dim": 1024, "depth": 6}),
    ("mamba2", "mamba2_d1024_depth6", {"batch_size": BATCH_SIZE, "dim": 1024, "depth": 6}),
]


BENCHMARK_SCRIPT = '''
import sys; sys.path.insert(0, '/home/erikg/elman')
import os
os.environ['LD_LIBRARY_PATH'] = f"/home/erikg/.local/lib/python3.12/site-packages/torch/lib:{{os.environ.get('LD_LIBRARY_PATH', '')}}"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mmap
import time
from schedulefree import AdamWScheduleFree

# CRITICAL: Same seed for ALL models
DATA_SEED = {data_seed}
torch.manual_seed(DATA_SEED)
np.random.seed(DATA_SEED)

model_type = "{model_type}"
dim = {dim}
depth = {depth}
batch_size = {batch_size}
seq_len = {seq_len}
time_limit = {time_limit}
n_slots = {n_slots}

# Data setup
with open('{data_path}', 'rb') as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
data_len = len(mm)

# Pre-generate ALL batch positions for deterministic data loading
# This ensures every model sees EXACTLY the same data in the same order
max_steps = int(time_limit * 2)  # Upper bound on steps
np.random.seed(DATA_SEED)  # Reset seed before generating positions
all_positions = np.random.randint(0, data_len - seq_len - 1, size=(max_steps, batch_size))

buf = np.zeros((batch_size, seq_len + 1), dtype=np.uint8)

def get_batch_deterministic(step):
    """Get batch at specific step - deterministic across processes."""
    pos = all_positions[step % len(all_positions)]
    for j, p in enumerate(pos):
        buf[j] = np.frombuffer(mm[p:p+seq_len+1], dtype=np.uint8)
    return torch.from_numpy(buf.astype(np.int64)).cuda()

# Model-specific imports and creation
if model_type == "mamba2":
    from elman.models.mamba2_baseline import Mamba2LM, MAMBA2_AVAILABLE
    if not MAMBA2_AVAILABLE:
        print("ERROR: mamba_ssm not installed")
        sys.exit(1)
    model = Mamba2LM(vocab_size=256, dim=dim, depth=depth).cuda().bfloat16()

elif model_type == "e1":
    from elman.models.mamba_gated_elman import MambaGatedElman

    class E1LM(nn.Module):
        def __init__(self, vocab_size, dim, depth):
            super().__init__()
            self.vocab_size = vocab_size
            self.embed = nn.Embedding(vocab_size, dim)
            self.layers = nn.ModuleList([MambaGatedElman(dim, expansion=1.0) for _ in range(depth)])
            self.norm = nn.RMSNorm(dim)
            self.lm_head = nn.Linear(dim, vocab_size, bias=False)
            self.lm_head.weight = self.embed.weight
            nn.init.normal_(self.embed.weight, std=0.02)

        def forward(self, x, return_loss=False):
            if return_loss:
                targets = x[:, 1:].contiguous()
                x = x[:, :-1]
            h = self.embed(x)
            for layer in self.layers:
                out, _ = layer(h)
                h = h + out
            h = self.norm(h)
            logits = self.lm_head(h)
            if return_loss:
                return F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            return logits

    model = E1LM(256, dim, depth).cuda().bfloat16()

elif model_type in ("e29a", "e29b"):
    from elman.models.e29_selective import E29aSelectiveElmanCell, E29bSelectiveElmanCell

    Cell = E29aSelectiveElmanCell if model_type == "e29a" else E29bSelectiveElmanCell

    class E29LM(nn.Module):
        def __init__(self, vocab_size, dim, depth, n_slots):
            super().__init__()
            self.vocab_size = vocab_size
            self.embed = nn.Embedding(vocab_size, dim)
            self.layers = nn.ModuleList([Cell(dim, n_slots=n_slots) for _ in range(depth)])
            self.norm = nn.RMSNorm(dim)
            self.lm_head = nn.Linear(dim, vocab_size, bias=False)
            self.lm_head.weight = self.embed.weight
            nn.init.normal_(self.embed.weight, std=0.02)

        def forward(self, x, return_loss=False):
            if return_loss:
                targets = x[:, 1:].contiguous()
                x = x[:, :-1]
            h = self.embed(x)
            for layer in self.layers:
                out, _, _ = layer(h, use_cuda=True)  # CUDA forward, Python backward
                h = h + out
            h = self.norm(h)
            logits = self.lm_head(h)
            if return_loss:
                return F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            return logits

    model = E29LM(256, dim, depth, n_slots).cuda().bfloat16()

else:
    print(f"ERROR: Unknown model type: {{model_type}}")
    sys.exit(1)

# Count parameters
n_params = sum(p.numel() for p in model.parameters())
print(f'{{model_type.upper()}} D={{dim}} depth={{depth}}: params={{n_params:,}}', flush=True)

# Training setup
opt = AdamWScheduleFree(model.parameters(), lr=3e-4, weight_decay=0.1)
model.train()
opt.train()

losses = []
start = time.time()
step = 0

while time.time() - start < time_limit:
    step += 1
    batch = get_batch_deterministic(step - 1)  # 0-indexed
    opt.zero_grad()
    loss = model(batch, return_loss=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    losses.append(loss.item())

    if step % 50 == 0:
        elapsed = time.time() - start
        tokens = step * batch_size * seq_len
        avg100 = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
        print(f'Step {{step}} | {{elapsed:.0f}}s | Loss {{loss.item():.4f}} | Avg100 {{avg100:.4f}} | {{int(tokens/elapsed)/1000:.1f}}K tok/s', flush=True)

elapsed = time.time() - start
tokens = step * batch_size * seq_len
avg_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
tps = int(tokens / elapsed)
print(f'FINAL: steps={{step}}, params={{n_params/1e6:.1f}}M, loss={{avg_loss:.4f}}, tok/s={{tps/1000:.1f}}K', flush=True)
mm.close()
'''


def run_config(gpu_id, model_type, name, extra_args):
    """Run a single config on a specific GPU."""
    log_file = OUTPUT_DIR / f"{name}.log"
    script_file = OUTPUT_DIR / f"{name}.py"

    # Allow per-config overrides
    n_slots = extra_args.get("n_slots", 8)
    batch_size = extra_args.get("batch_size", BATCH_SIZE)
    dim = extra_args.get("dim", DIM)
    depth = extra_args.get("depth", DEPTH)

    script = BENCHMARK_SCRIPT.format(
        model_type=model_type,
        dim=dim,
        depth=depth,
        batch_size=batch_size,
        seq_len=SEQ_LEN,
        time_limit=TIME_LIMIT,
        data_path=DATA_PATH,
        data_seed=DATA_SEED,
        n_slots=n_slots,
    )

    with open(script_file, "w") as f:
        f.write(script)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["LD_LIBRARY_PATH"] = f"/home/erikg/.local/lib/python3.12/site-packages/torch/lib:{env.get('LD_LIBRARY_PATH', '')}"

    print(f"[GPU {gpu_id}] Starting {name}")

    with open(log_file, "w") as f:
        f.write(f"Config: {model_type.upper()} D={dim}, depth={depth}, batch={batch_size}\n")
        f.write(f"Data seed: {DATA_SEED}\n")
        f.write("=" * 60 + "\n\n")
        f.flush()

        proc = subprocess.Popen(
            ["python", str(script_file)],
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            cwd="/home/erikg/elman"
        )

    return proc, name, log_file


def extract_results(log_file):
    """Extract final loss and throughput from log file."""
    try:
        with open(log_file, "r") as f:
            lines = f.readlines()

        loss = None
        throughput = None
        params = None

        for line in lines:
            if "FINAL:" in line:
                match = re.search(r'loss=(\d+\.\d+)', line)
                if match:
                    loss = float(match.group(1))
                match = re.search(r'tok/s=(\d+\.?\d*)K', line)
                if match:
                    throughput = float(match.group(1)) * 1000
                match = re.search(r'params=(\d+\.?\d*)M', line)
                if match:
                    params = float(match.group(1))

        return loss, throughput, params
    except Exception as e:
        print(f"Error extracting from {log_file}: {e}")
        return None, None, None


def main():
    print("=" * 70)
    print("E29 Benchmark: Mamba2 vs E1 vs E29a vs E29b")
    print("=" * 70)
    print(f"Config: D={DIM}, depth={DEPTH}, batch={BATCH_SIZE}, seq_len={SEQ_LEN}")
    print(f"Time limit: {TIME_LIMIT}s ({TIME_LIMIT/60:.0f} minutes)")
    print(f"Data seed: {DATA_SEED} (SAME FOR ALL - deterministic comparison)")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Running {len(CONFIGS)} models")
    print()

    # Launch all processes
    processes = []
    for gpu_id, (model_type, name, extra_args) in enumerate(CONFIGS):
        proc, name, log_file = run_config(gpu_id, model_type, name, extra_args)
        dim = extra_args.get("dim", DIM)
        depth = extra_args.get("depth", DEPTH)
        batch_size = extra_args.get("batch_size", BATCH_SIZE)
        processes.append((proc, name, log_file, model_type, dim, depth, batch_size))

    print(f"\nLaunched {len(processes)} experiments. Waiting for completion...")
    print(f"Monitor with: tail -f {OUTPUT_DIR}/*.log")
    print()

    # Wait for all to complete
    start_time = time.time()
    while True:
        running = sum(1 for p, *_ in processes if p.poll() is None)
        if running == 0:
            break
        elapsed = time.time() - start_time
        print(f"\r[{elapsed/60:.1f}min] {running}/{len(processes)} still running...", end="", flush=True)
        time.sleep(10)

    print(f"\n\nAll experiments complete! Total time: {(time.time() - start_time)/60:.1f} minutes")

    # Collect results
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    print(f"{'Model':<25} | {'Params':>8} | {'Loss':>8} | {'Throughput':>12}")
    print("-" * 70)

    results = []
    for proc, name, log_file, model_type, dim, depth, batch_size in processes:
        loss, throughput, params = extract_results(log_file)
        results.append({
            "name": name,
            "model_type": model_type,
            "dim": dim,
            "depth": depth,
            "batch_size": batch_size,
            "params_m": params,
            "loss": loss,
            "throughput": throughput,
        })

        loss_str = f"{loss:.4f}" if loss else "N/A"
        tp_str = f"{throughput/1000:.1f}K tok/s" if throughput else "N/A"
        params_str = f"{params:.1f}M" if params else "N/A"

        print(f"{name:<25} | {params_str:>8} | {loss_str:>8} | {tp_str:>12}")

    # Analysis
    print("\n" + "=" * 70)
    print("Analysis")
    print("=" * 70)

    e1 = next((r for r in results if r["model_type"] == "e1"), None)
    mamba2 = next((r for r in results if r["model_type"] == "mamba2"), None)
    e29a = next((r for r in results if r["model_type"] == "e29a"), None)
    e29b = next((r for r in results if r["model_type"] == "e29b"), None)

    if e1 and e1["loss"]:
        print(f"\nE1 (baseline): loss={e1['loss']:.4f}, {e1['throughput']/1000:.1f}K tok/s")

        for name, model in [("Mamba2", mamba2), ("E29a", e29a), ("E29b", e29b)]:
            if model and model["loss"]:
                loss_diff = model["loss"] - e1["loss"]
                tp_ratio = model["throughput"] / e1["throughput"]
                better = "better" if loss_diff < 0 else "worse"
                faster = "faster" if tp_ratio > 1 else "slower"
                print(f"  vs {name}: {loss_diff:+.4f} loss ({better}), {tp_ratio:.2f}x speed ({faster})")

    # Save results
    results_file = OUTPUT_DIR / "summary.json"
    with open(results_file, "w") as f:
        json.dump({
            "config": {
                "dim": DIM,
                "depth": DEPTH,
                "batch_size": BATCH_SIZE,
                "seq_len": SEQ_LEN,
                "time_limit": TIME_LIMIT,
                "data_seed": DATA_SEED,
            },
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
