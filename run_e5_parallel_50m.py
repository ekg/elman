#!/usr/bin/env python3
"""Run E5 50M variants in parallel across GPUs - matching the winning config."""
import subprocess
import os
import sys
import time

# E5 50M configs (all ~50M params, depth=20, batch=256)
CONFIGS = [
    {"dim": 768,  "rank": 539, "gpu": 0},
    {"dim": 1024, "rank": 404, "gpu": 1},
    {"dim": 1536, "rank": 270, "gpu": 2},  # Winner
    {"dim": 2048, "rank": 200, "gpu": 3},
]

DEPTH = 20
BATCH_SIZE = 256
STEPS = 1000
DATA = "data/fineweb_100mb.txt"

SCRIPT_TEMPLATE = '''
import sys
sys.path.insert(0, '.')
import torch
import time
import math
from schedulefree import AdamWScheduleFree
from elman.models import LadderLM
from elman.data import FastTokenizedDataset
from elman.data.tokenizers import ByteTokenizer

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

dim = {dim}
rank = {rank}
depth = {depth}
batch_size = {batch_size}
steps = {steps}

model = LadderLM(
    vocab_size=256,
    dim=dim,
    depth=depth,
    level=5,
    rank=rank,
).cuda().bfloat16()

num_params = model.get_num_params()
print(f'e5_50m_d{{dim}}_r{{rank}}: params={{num_params:,}}, depth={{depth}}, batch={{batch_size}}')

tokenizer = ByteTokenizer()
dataset = FastTokenizedDataset('{data}', tokenizer, batch_size, 513, seed=42)

optimizer = AdamWScheduleFree(model.parameters(), lr=3e-4, weight_decay=0.1)
model.train()
optimizer.train()

start = time.time()
tokens = 0

for step in range(1, steps+1):
    batch, _, _ = dataset.get_batch(device='cuda')
    optimizer.zero_grad()
    loss = model(batch, return_loss=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    tokens += batch_size * 512

    if step % 100 == 0:
        elapsed = time.time() - start
        ppl = math.exp(min(loss.item(), 20))
        print(f'Step {{step}} | Loss {{loss.item():.4f}} | PPL {{ppl:.1f}} | {{int(tokens/elapsed)}} tok/s')

print(f'DONE: e5_50m_d{{dim}}_r{{rank}} | Final loss: {{loss.item():.4f}}')
'''

OUT_DIR = "benchmark_results/e5_parallel_50m"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("Running E5 50M variants in parallel")
print("=" * 60)

processes = []
for cfg in CONFIGS:
    dim = cfg["dim"]
    rank = cfg["rank"]
    gpu = cfg["gpu"]

    script = SCRIPT_TEMPLATE.format(
        dim=dim, rank=rank, depth=DEPTH,
        batch_size=BATCH_SIZE, steps=STEPS, data=DATA
    )

    name = f"e5_d{dim}_r{rank}"
    log = f"{OUT_DIR}/{name}.log"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    print(f"Launching {name} on GPU {gpu}...")

    proc = subprocess.Popen(
        ["python", "-u", "-c", script],
        stdout=open(log, "w"),
        stderr=subprocess.STDOUT,
        env=env
    )
    processes.append((name, proc, log))
    time.sleep(1)

print(f"\nAll {len(CONFIGS)} experiments launched!")
print("Waiting for completion...\n")

# Wait and show progress
while any(p.poll() is None for _, p, _ in processes):
    time.sleep(30)
    print("\n--- Progress ---")
    for name, proc, log in processes:
        if os.path.exists(log):
            with open(log) as f:
                lines = f.readlines()
                if lines:
                    last = lines[-1].strip()
                    status = "RUNNING" if proc.poll() is None else "DONE"
                    print(f"{name}: {status} | {last[:60]}")

# Final results
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
for name, proc, log in processes:
    if os.path.exists(log):
        with open(log) as f:
            for line in f:
                if "DONE:" in line:
                    print(line.strip())
                    break
