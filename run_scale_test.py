#!/usr/bin/env python3
"""
Scaling test: E1 vs E9 vs Mamba2 at 50M, 100M, 200M, 400M params.
10 minutes per run, 8 GPUs available.
12 total experiments, run in batches.
"""
import subprocess, os, time

TIME_LIMIT = 600  # 10 minutes

# Use conservative batch sizes for fair comparison (same tokens/step)
BATCH_SIZES = {50: 256, 100: 192, 200: 128, 400: 96}

# All experiments
EXPERIMENTS = [
    # 50M scale
    {"name": "e1_50m", "type": "e1", "dim": 1280, "depth": 6, "scale": 50},
    {"name": "e9_50m", "type": "e9", "dim": 1440, "depth": 3, "scale": 50},
    {"name": "mamba2_50m", "type": "mamba2", "scale": 50},
    # 100M scale
    {"name": "e1_100m", "type": "e1", "dim": 1280, "depth": 12, "scale": 100},
    {"name": "e9_100m", "type": "e9", "dim": 1440, "depth": 6, "scale": 100},
    {"name": "mamba2_100m", "type": "mamba2", "scale": 100},
    # 200M scale
    {"name": "e1_200m", "type": "e1", "dim": 1792, "depth": 12, "scale": 200},
    {"name": "e9_200m", "type": "e9", "dim": 2048, "depth": 6, "scale": 200},
    {"name": "mamba2_200m", "type": "mamba2", "scale": 200},
    # 400M scale
    {"name": "e1_400m", "type": "e1", "dim": 2304, "depth": 12, "scale": 400},
    {"name": "e9_400m", "type": "e9", "dim": 2560, "depth": 6, "scale": 400},
    {"name": "mamba2_400m", "type": "mamba2", "scale": 400},
]

E1_SCRIPT = '''
import sys; sys.path.insert(0, '.')
import torch, numpy as np, mmap, time
from schedulefree import AdamWScheduleFree
from elman.models import LadderLM

torch.manual_seed(42); np.random.seed(42)
batch_size, time_limit = {batch_size}, {time_limit}
dim, depth = {dim}, {depth}

model = LadderLM(vocab_size=256, dim=dim, depth=depth, level=1).cuda().bfloat16()
print(f'{name}: params={{model.get_num_params():,}}')

with open('data/pile.txt', 'rb') as f: mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
data_len = len(mm)

opt = AdamWScheduleFree(model.parameters(), lr=3e-4, weight_decay=0.1)
model.train(); opt.train()
buf = np.zeros((batch_size, 513), dtype=np.uint8)
losses = []; start = time.time(); step = 0

while time.time() - start < time_limit:
    step += 1
    pos = np.random.randint(0, data_len - 513, size=batch_size)
    for j, p in enumerate(pos): buf[j] = np.frombuffer(mm[p:p+513], dtype=np.uint8)
    batch = torch.from_numpy(buf.astype(np.int64)).cuda()
    opt.zero_grad(); loss = model(batch, return_loss=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    losses.append(loss.item())
    if step % 100 == 0:
        elapsed = time.time() - start
        tokens = step * batch_size * 512
        print(f'Step {{step}} | {{elapsed:.0f}}s | Loss {{loss.item():.4f}} | Avg100 {{np.mean(losses[-100:]):.4f}} | {{int(tokens/elapsed)}} tok/s')

elapsed = time.time() - start
tokens = step * batch_size * 512
print(f'DONE: {name} | Steps={{step}} | Time={{elapsed:.0f}}s | Last100={{np.mean(losses[-100:]):.4f}} | {{int(tokens/elapsed)}} tok/s')
mm.close()
'''

E9_SCRIPT = '''
import sys; sys.path.insert(0, '.')
import torch, numpy as np, mmap, time
from schedulefree import AdamWScheduleFree
from elman.models import LadderLM

torch.manual_seed(42); np.random.seed(42)
batch_size, time_limit = {batch_size}, {time_limit}
dim, depth = {dim}, {depth}

model = LadderLM(vocab_size=256, dim=dim, depth=depth, level=9,
                 expansion=2.0, core_ratio=0.5).cuda().bfloat16()
layer = model.layers[0]
print(f'{name}: core={{layer.core_dim}} mem={{layer.mem_dim}} params={{model.get_num_params():,}}')

with open('data/pile.txt', 'rb') as f: mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
data_len = len(mm)

opt = AdamWScheduleFree(model.parameters(), lr=3e-4, weight_decay=0.1)
model.train(); opt.train()
buf = np.zeros((batch_size, 513), dtype=np.uint8)
losses = []; start = time.time(); step = 0

while time.time() - start < time_limit:
    step += 1
    pos = np.random.randint(0, data_len - 513, size=batch_size)
    for j, p in enumerate(pos): buf[j] = np.frombuffer(mm[p:p+513], dtype=np.uint8)
    batch = torch.from_numpy(buf.astype(np.int64)).cuda()
    opt.zero_grad(); loss = model(batch, return_loss=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    losses.append(loss.item())
    if step % 100 == 0:
        elapsed = time.time() - start
        tokens = step * batch_size * 512
        print(f'Step {{step}} | {{elapsed:.0f}}s | Loss {{loss.item():.4f}} | Avg100 {{np.mean(losses[-100:]):.4f}} | {{int(tokens/elapsed)}} tok/s')

elapsed = time.time() - start
tokens = step * batch_size * 512
print(f'DONE: {name} | Steps={{step}} | Time={{elapsed:.0f}}s | Last100={{np.mean(losses[-100:]):.4f}} | {{int(tokens/elapsed)}} tok/s')
mm.close()
'''

MAMBA2_SCRIPT = '''
import sys; sys.path.insert(0, '.')
import torch, numpy as np, mmap, time
from schedulefree import AdamWScheduleFree
from elman.models import create_mamba2_model

torch.manual_seed(42); np.random.seed(42)
batch_size, time_limit = {batch_size}, {time_limit}

model = create_mamba2_model(target_params='{scale}m', vocab_size=256).cuda().bfloat16()
print(f'{name}: params={{model.get_num_params():,}}')

with open('data/pile.txt', 'rb') as f: mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
data_len = len(mm)

opt = AdamWScheduleFree(model.parameters(), lr=3e-4, weight_decay=0.1)
model.train(); opt.train()
buf = np.zeros((batch_size, 513), dtype=np.uint8)
losses = []; start = time.time(); step = 0

while time.time() - start < time_limit:
    step += 1
    pos = np.random.randint(0, data_len - 513, size=batch_size)
    for j, p in enumerate(pos): buf[j] = np.frombuffer(mm[p:p+513], dtype=np.uint8)
    batch = torch.from_numpy(buf.astype(np.int64)).cuda()
    opt.zero_grad(); loss = model(batch, return_loss=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    losses.append(loss.item())
    if step % 100 == 0:
        elapsed = time.time() - start
        tokens = step * batch_size * 512
        print(f'Step {{step}} | {{elapsed:.0f}}s | Loss {{loss.item():.4f}} | Avg100 {{np.mean(losses[-100:]):.4f}} | {{int(tokens/elapsed)}} tok/s')

elapsed = time.time() - start
tokens = step * batch_size * 512
print(f'DONE: {name} | Steps={{step}} | Time={{elapsed:.0f}}s | Last100={{np.mean(losses[-100:]):.4f}} | {{int(tokens/elapsed)}} tok/s')
mm.close()
'''

OUT = "benchmark_results/scale_test"; os.makedirs(OUT, exist_ok=True)

print("="*70)
print("SCALING TEST: E1 vs E9 vs Mamba2")
print("Scales: 50M, 100M, 200M, 400M params")
print(f"Time per run: {TIME_LIMIT}s (10 min)")
print("="*70)

# Run in batches of 8 (we have 8 GPUs)
# Batch 1: 50M (3) + 100M (3) + 200M (2) = 8
# Batch 2: 200M (1) + 400M (3) = 4
batch1 = EXPERIMENTS[:8]
batch2 = EXPERIMENTS[8:]

def run_batch(experiments, batch_name):
    print(f"\n--- {batch_name} ---")
    procs = []
    for i, e in enumerate(experiments):
        gpu = i
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        batch_size = BATCH_SIZES[e["scale"]]

        if e["type"] == "e1":
            script = E1_SCRIPT.format(
                dim=e["dim"], depth=e["depth"], batch_size=batch_size,
                name=e["name"], time_limit=TIME_LIMIT
            )
        elif e["type"] == "e9":
            script = E9_SCRIPT.format(
                dim=e["dim"], depth=e["depth"], batch_size=batch_size,
                name=e["name"], time_limit=TIME_LIMIT
            )
        else:  # mamba2
            script = MAMBA2_SCRIPT.format(
                scale=e["scale"], batch_size=batch_size,
                name=e["name"], time_limit=TIME_LIMIT
            )

        log_path = f"{OUT}/{e['name']}.log"
        print(f"GPU {gpu}: {e['name']} (batch={batch_size})")
        p = subprocess.Popen(["python", "-u", "-c", script],
                           stdout=open(log_path, "w"), stderr=subprocess.STDOUT, env=env)
        procs.append((e["name"], p, log_path))
        time.sleep(0.3)

    print(f"Launched {len(experiments)}. Waiting...")

    # Monitor progress
    while any(p.poll() is None for _, p, _ in procs):
        time.sleep(60)
        print("\n--- Progress ---")
        for name, proc, log in procs:
            status = "DONE" if proc.poll() is not None else "running"
            try:
                with open(log) as f:
                    lines = f.readlines()
                    for line in reversed(lines[-3:]):
                        if "Step" in line or "DONE" in line:
                            parts = line.strip().split("|")
                            short = " | ".join(p.strip() for p in parts[:2] + parts[-1:])
                            print(f"  [{status}] {name}: {short[-70:]}")
                            break
            except: print(f"  [{status}] {name}: starting...")

    return procs

# Run batches
all_procs = []
all_procs.extend(run_batch(batch1, "Batch 1: 50M + 100M + 200M[:2]"))
all_procs.extend(run_batch(batch2, "Batch 2: 200M[2:] + 400M"))

# Final results
print("\n" + "="*70)
print("FINAL RESULTS - SCALING TEST")
print("="*70)

results = []
for name, _, log in all_procs:
    with open(log) as f:
        for line in f:
            if "DONE:" in line:
                print(line.strip())
                parts = line.split("|")
                loss = float([p for p in parts if "Last100" in p][0].split("=")[1])
                toks = int([p for p in parts if "tok/s" in p][0].split()[0])
                # Parse scale from name
                scale = int(name.split("_")[1].replace("m", ""))
                model = name.split("_")[0]
                results.append((scale, model, loss, toks))

print("\n" + "="*70)
print("BY SCALE:")
print("="*70)
for scale in [50, 100, 200, 400]:
    print(f"\n{scale}M params:")
    scale_results = [(m, l, t) for s, m, l, t in results if s == scale]
    for model, loss, toks in sorted(scale_results, key=lambda x: x[1]):
        print(f"  {model}: loss={loss:.4f}, {toks//1000}K tok/s")
