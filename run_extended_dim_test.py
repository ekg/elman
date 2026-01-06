#!/usr/bin/env python3
"""
Extended dimension tests for E5 and E1 variants.
Tests larger dims for E5 and depth/width tradeoffs for E1.
All ~50M params, 1000 steps, reports Last-100-step average loss.
"""
import subprocess
import os
import time

# E5 extended dims (larger than d2048) - reduce batch for memory
# E1 depth/width variants
EXPERIMENTS = [
    # GPU 0-2: E5 larger dims (batch=128 for memory)
    {"gpu": 0, "name": "e5_d3072_r133", "type": "e5", "dim": 3072, "rank": 133, "depth": 20, "batch": 128},
    {"gpu": 1, "name": "e5_d4096_r99",  "type": "e5", "dim": 4096, "rank": 99,  "depth": 20, "batch": 128},
    {"gpu": 2, "name": "e5_d6144_r65",  "type": "e5", "dim": 6144, "rank": 65,  "depth": 20, "batch": 64},

    # GPU 3-5: E1 depth/width variants
    {"gpu": 3, "name": "e1_d256_deep",  "type": "e1", "dim": 256,  "depth": 84, "batch": 256},  # thin & deep
    {"gpu": 4, "name": "e1_d512_mid",   "type": "e1", "dim": 512,  "depth": 21, "batch": 256},  # baseline
    {"gpu": 5, "name": "e1_d1024_wide", "type": "e1", "dim": 1024, "depth": 5,  "batch": 256},  # wide & shallow

    # GPU 6-7: More E5 reference points
    {"gpu": 6, "name": "e5_d1536_r270", "type": "e5", "dim": 1536, "rank": 270, "depth": 20, "batch": 256},
    {"gpu": 7, "name": "e5_d2048_r200", "type": "e5", "dim": 2048, "rank": 200, "depth": 20, "batch": 256},
]

STEPS = 1000
DATA = "data/pile.txt"

E5_SCRIPT = '''
import sys
sys.path.insert(0, '.')
import torch
import numpy as np
import mmap
import math
import time
from schedulefree import AdamWScheduleFree
from elman.models import LadderLM

torch.manual_seed(42)
np.random.seed(42)

dim, rank, depth = {dim}, {rank}, {depth}
batch_size, steps, seq_len = {batch_size}, {steps}, 512

model = LadderLM(vocab_size=256, dim=dim, depth=depth, level=5, rank=rank).cuda().bfloat16()
params = model.get_num_params()
print(f'{name}: params={{params:,}}, dim={{dim}}, rank={{rank}}, depth={{depth}}, batch={{batch_size}}')

with open('{data}', 'rb') as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
data_len = len(mm)

optimizer = AdamWScheduleFree(model.parameters(), lr=3e-4, weight_decay=0.1)
model.train()
optimizer.train()

batch_buffer = np.zeros((batch_size, seq_len + 1), dtype=np.uint8)
all_losses = []
start = time.time()
tokens = 0

for step in range(1, steps+1):
    positions = np.random.randint(0, data_len - seq_len - 1, size=batch_size)
    for j, pos in enumerate(positions):
        batch_buffer[j] = np.frombuffer(mm[pos:pos+seq_len+1], dtype=np.uint8)
    batch = torch.from_numpy(batch_buffer.astype(np.int64)).cuda()

    optimizer.zero_grad()
    loss = model(batch, return_loss=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    tokens += batch_size * seq_len
    all_losses.append(loss.item())

    if step % 100 == 0:
        elapsed = time.time() - start
        avg_100 = np.mean(all_losses[-100:])
        print(f'Step {{step}} | Loss {{loss.item():.4f}} | Avg100 {{avg_100:.4f}} | {{int(tokens/elapsed)}} tok/s')

avg_last_100 = np.mean(all_losses[-100:])
print(f'DONE: {name} | Last100={{avg_last_100:.4f}} | Final={{loss.item():.4f}} | Params={{params:,}}')
mm.close()
'''

E1_SCRIPT = '''
import sys
sys.path.insert(0, '.')
import torch
import numpy as np
import mmap
import math
import time
from schedulefree import AdamWScheduleFree
from elman.models import LadderLM

torch.manual_seed(42)
np.random.seed(42)

dim, depth = {dim}, {depth}
batch_size, steps, seq_len = {batch_size}, {steps}, 512

model = LadderLM(vocab_size=256, dim=dim, depth=depth, level=1).cuda().bfloat16()
params = model.get_num_params()
print(f'{name}: params={{params:,}}, dim={{dim}}, depth={{depth}}, batch={{batch_size}}')

with open('{data}', 'rb') as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
data_len = len(mm)

optimizer = AdamWScheduleFree(model.parameters(), lr=3e-4, weight_decay=0.1)
model.train()
optimizer.train()

batch_buffer = np.zeros((batch_size, seq_len + 1), dtype=np.uint8)
all_losses = []
start = time.time()
tokens = 0

for step in range(1, steps+1):
    positions = np.random.randint(0, data_len - seq_len - 1, size=batch_size)
    for j, pos in enumerate(positions):
        batch_buffer[j] = np.frombuffer(mm[pos:pos+seq_len+1], dtype=np.uint8)
    batch = torch.from_numpy(batch_buffer.astype(np.int64)).cuda()

    optimizer.zero_grad()
    loss = model(batch, return_loss=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    tokens += batch_size * seq_len
    all_losses.append(loss.item())

    if step % 100 == 0:
        elapsed = time.time() - start
        avg_100 = np.mean(all_losses[-100:])
        print(f'Step {{step}} | Loss {{loss.item():.4f}} | Avg100 {{avg_100:.4f}} | {{int(tokens/elapsed)}} tok/s')

avg_last_100 = np.mean(all_losses[-100:])
print(f'DONE: {name} | Last100={{avg_last_100:.4f}} | Final={{loss.item():.4f}} | Params={{params:,}}')
mm.close()
'''

OUT_DIR = "benchmark_results/extended_dim_test"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("Extended Dimension Tests - E5 large dims + E1 depth/width variants")
print("All ~50M params, 1000 steps, Last-100 averaged loss")
print("=" * 70)

processes = []
for exp in EXPERIMENTS:
    gpu = exp["gpu"]
    name = exp["name"]
    exp_type = exp["type"]
    batch = exp.get("batch", 256)
    log = f"{OUT_DIR}/{name}.log"

    if exp_type == "e5":
        script = E5_SCRIPT.format(
            dim=exp["dim"], rank=exp["rank"], depth=exp["depth"],
            batch_size=batch, steps=STEPS, data=DATA, name=name
        )
    else:  # e1
        script = E1_SCRIPT.format(
            dim=exp["dim"], depth=exp["depth"],
            batch_size=batch, steps=STEPS, data=DATA, name=name
        )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    print(f"Launching {name} on GPU {gpu} (batch={batch})...")
    proc = subprocess.Popen(
        ["python", "-u", "-c", script],
        stdout=open(log, "w"),
        stderr=subprocess.STDOUT,
        env=env
    )
    processes.append((name, proc, log))
    time.sleep(0.5)

print(f"\nAll {len(EXPERIMENTS)} experiments launched!")
print("Waiting for completion...\n")

# Monitor progress
while any(p.poll() is None for _, p, _ in processes):
    time.sleep(60)
    print("\n--- Progress ---")
    for name, proc, log in processes:
        if os.path.exists(log):
            with open(log) as f:
                lines = f.readlines()
                for line in reversed(lines):
                    if line.strip() and ("Step" in line or "DONE" in line or "Error" in line):
                        print(f"{name}: {line.strip()[:75]}")
                        break

# Final results
print("\n" + "=" * 70)
print("FINAL RESULTS - Last 100 Step Average Loss")
print("=" * 70)
for name, proc, log in processes:
    if os.path.exists(log):
        with open(log) as f:
            for line in f:
                if "DONE:" in line:
                    print(line.strip())

print("\n" + "=" * 70)
