#!/usr/bin/env python3
"""
Run E5 variants, E1, and Mamba2 in parallel across 8 GPUs.
All ~50M params, 1000 steps, batch=256, reports last-100-step average loss.
"""
import subprocess
import os
import time

# 6 experiments across 6 GPUs (avoiding OOM on large dim)
# E5 configs: ~50M params each, depth=20
EXPERIMENTS = [
    # GPU 0-3: E5 variants with increasing dim
    {"gpu": 0, "name": "e5_d768_r539",  "type": "e5", "dim": 768,  "rank": 539,  "depth": 20},
    {"gpu": 1, "name": "e5_d1024_r404", "type": "e5", "dim": 1024, "rank": 404,  "depth": 20},
    {"gpu": 2, "name": "e5_d1536_r270", "type": "e5", "dim": 1536, "rank": 270,  "depth": 20},
    {"gpu": 3, "name": "e5_d2048_r200", "type": "e5", "dim": 2048, "rank": 200,  "depth": 20},
    # GPU 4-5: Baselines
    {"gpu": 4, "name": "e1_50m",        "type": "e1"},
    {"gpu": 5, "name": "mamba2_50m",    "type": "mamba2"},
]

BATCH_SIZE = 256
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
print(f'{name}: params={{params:,}}, dim={{dim}}, rank={{rank}}, depth={{depth}}')

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
from elman.models import create_ladder_model

torch.manual_seed(42)
np.random.seed(42)

batch_size, steps, seq_len = {batch_size}, {steps}, 512

model = create_ladder_model(target_params='50m', level=1, vocab_size=256).cuda().bfloat16()
params = model.get_num_params()
print(f'e1_50m: params={{params:,}}')

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
print(f'DONE: e1_50m | Last100={{avg_last_100:.4f}} | Final={{loss.item():.4f}} | Params={{params:,}}')
mm.close()
'''

MAMBA2_SCRIPT = '''
import sys
sys.path.insert(0, '.')
import torch
import numpy as np
import mmap
import math
import time
from schedulefree import AdamWScheduleFree
from elman.models import create_mamba2_model

torch.manual_seed(42)
np.random.seed(42)

batch_size, steps, seq_len = {batch_size}, {steps}, 512

model = create_mamba2_model(target_params='50m', vocab_size=256).cuda().bfloat16()
params = model.get_num_params()
print(f'mamba2_50m: params={{params:,}}')

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
print(f'DONE: mamba2_50m | Last100={{avg_last_100:.4f}} | Final={{loss.item():.4f}} | Params={{params:,}}')
mm.close()
'''

OUT_DIR = "benchmark_results/e5_e1_mamba2_compare"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("E5 vs E1 vs Mamba2 Comparison - 50M params, 1000 steps, batch=256")
print("Reports last-100-step average loss")
print("=" * 70)

processes = []
for exp in EXPERIMENTS:
    gpu = exp["gpu"]
    name = exp["name"]
    exp_type = exp["type"]
    log = f"{OUT_DIR}/{name}.log"

    if exp_type == "e5":
        script = E5_SCRIPT.format(
            dim=exp["dim"], rank=exp["rank"], depth=exp["depth"],
            batch_size=BATCH_SIZE, steps=STEPS, data=DATA, name=name
        )
    elif exp_type == "e1":
        script = E1_SCRIPT.format(batch_size=BATCH_SIZE, steps=STEPS, data=DATA)
    else:  # mamba2
        script = MAMBA2_SCRIPT.format(batch_size=BATCH_SIZE, steps=STEPS, data=DATA)

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
                    if line.strip() and ("Step" in line or "DONE" in line):
                        status = "DONE" if "DONE" in line else "RUNNING"
                        print(f"{name}: {line.strip()[:70]}")
                        break

# Final results
print("\n" + "=" * 70)
print("FINAL RESULTS - Last 100 Step Average Loss")
print("=" * 70)
results = []
for name, proc, log in processes:
    if os.path.exists(log):
        with open(log) as f:
            for line in f:
                if "DONE:" in line:
                    results.append(line.strip())
                    print(line.strip())

print("\n" + "=" * 70)
