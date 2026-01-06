#!/usr/bin/env python3
"""
Test wider/shallower E1 configurations at ~50M params.
"""
import subprocess, os, time

TIME_LIMIT = 600  # 10 minutes

EXPERIMENTS = [
    # Previous winner
    {"gpu": 0, "name": "e1_d1280_depth6", "dim": 1280, "depth": 6},   # 49.5M
    # Even wider
    {"gpu": 1, "name": "e1_d1580_depth4", "dim": 1580, "depth": 4},   # 50.4M
    {"gpu": 2, "name": "e1_d1820_depth3", "dim": 1820, "depth": 3},   # 50.2M
    {"gpu": 3, "name": "e1_d2200_depth2", "dim": 2200, "depth": 2},   # 49.0M
]

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

OUT = "benchmark_results/wide_shallow"; os.makedirs(OUT, exist_ok=True)
print(f"Wide/shallow E1 comparison: {TIME_LIMIT}s (10 min) per model")
print("=" * 60)

procs = []
for e in EXPERIMENTS:
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(e["gpu"])
    script = E1_SCRIPT.format(dim=e["dim"], depth=e["depth"], name=e["name"], time_limit=TIME_LIMIT)
    print(f"Launching {e['name']} on GPU {e['gpu']}...")
    p = subprocess.Popen(["python", "-u", "-c", script], stdout=open(f"{OUT}/{e['name']}.log", "w"), stderr=subprocess.STDOUT, env=env)
    procs.append((e["name"], p, f"{OUT}/{e['name']}.log")); time.sleep(0.5)

print(f"\nAll {len(EXPERIMENTS)} launched! Running for {TIME_LIMIT}s each...")

while any(p.poll() is None for _, p, _ in procs):
    time.sleep(30)

print("\n" + "=" * 60)
print("RESULTS - 10 minutes training time")
print("=" * 60)
for n, _, l in procs:
    with open(l) as f:
        for line in f:
            if "DONE:" in line: print(line.strip())
