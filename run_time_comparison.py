#!/usr/bin/env python3
"""
Time-based comparison: Run each model for 10 minutes, compare final loss.
Tests which architecture is truly faster to train.
"""
import subprocess, os, time

TIME_LIMIT = 600  # 10 minutes

EXPERIMENTS = [
    # Best E5
    {"gpu": 0, "name": "e5_d1536_r270", "type": "e5", "dim": 1536, "rank": 270, "depth": 20},
    {"gpu": 1, "name": "e5_d2048_r200", "type": "e5", "dim": 2048, "rank": 200, "depth": 20},
    # Best E1
    {"gpu": 2, "name": "e1_d1024_depth10", "type": "e1", "dim": 1024, "depth": 10},
    {"gpu": 3, "name": "e1_d1280_depth6", "type": "e1", "dim": 1280, "depth": 6},
    # Mamba2
    {"gpu": 4, "name": "mamba2_50m", "type": "mamba2"},
]

E5_SCRIPT = '''
import sys; sys.path.insert(0, '.')
import torch, numpy as np, mmap, time
from schedulefree import AdamWScheduleFree
from elman.models import LadderLM

torch.manual_seed(42); np.random.seed(42)
dim, rank, depth, batch_size = {dim}, {rank}, {depth}, 256
time_limit = {time_limit}

model = LadderLM(vocab_size=256, dim=dim, depth=depth, level=5, rank=rank).cuda().bfloat16()
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

E1_SCRIPT = '''
import sys; sys.path.insert(0, '.')
import torch, numpy as np, mmap, time
from schedulefree import AdamWScheduleFree
from elman.models import LadderLM

torch.manual_seed(42); np.random.seed(42)
dim, depth, batch_size = {dim}, {depth}, 256
time_limit = {time_limit}

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

MAMBA2_SCRIPT = '''
import sys; sys.path.insert(0, '.')
import torch, numpy as np, mmap, time
from schedulefree import AdamWScheduleFree
from elman.models import create_mamba2_model

torch.manual_seed(42); np.random.seed(42)
batch_size = 256
time_limit = {time_limit}

model = create_mamba2_model(target_params='50m', vocab_size=256).cuda().bfloat16()
print(f'mamba2_50m: params={{model.get_num_params():,}}')

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
print(f'DONE: mamba2_50m | Steps={{step}} | Time={{elapsed:.0f}}s | Last100={{np.mean(losses[-100:]):.4f}} | {{int(tokens/elapsed)}} tok/s')
mm.close()
'''

OUT = "benchmark_results/time_comparison"; os.makedirs(OUT, exist_ok=True)
print(f"Time-based comparison: {TIME_LIMIT}s (10 min) per model")
print("=" * 60)

procs = []
for e in EXPERIMENTS:
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(e["gpu"])
    if e["type"] == "e5":
        script = E5_SCRIPT.format(dim=e["dim"], rank=e["rank"], depth=e["depth"], name=e["name"], time_limit=TIME_LIMIT)
    elif e["type"] == "e1":
        script = E1_SCRIPT.format(dim=e["dim"], depth=e["depth"], name=e["name"], time_limit=TIME_LIMIT)
    else:
        script = MAMBA2_SCRIPT.format(time_limit=TIME_LIMIT)

    print(f"Launching {e['name']} on GPU {e['gpu']}...")
    p = subprocess.Popen(["python", "-u", "-c", script], stdout=open(f"{OUT}/{e['name']}.log", "w"), stderr=subprocess.STDOUT, env=env)
    procs.append((e["name"], p, f"{OUT}/{e['name']}.log")); time.sleep(0.5)

print(f"\nAll {len(EXPERIMENTS)} launched! Running for {TIME_LIMIT}s each...")

# Wait for all to complete
while any(p.poll() is None for _, p, _ in procs):
    time.sleep(30)

print("\n" + "=" * 60)
print("RESULTS - 10 minutes training time")
print("=" * 60)
for n, _, l in procs:
    with open(l) as f:
        for line in f:
            if "DONE:" in line: print(line.strip())
