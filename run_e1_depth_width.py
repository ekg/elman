#!/usr/bin/env python3
"""E1 depth/width sweep at ~50M params."""
import subprocess, os, time

EXPERIMENTS = [
    {"gpu": 0, "name": "e1_d512_depth40",  "dim": 512,  "depth": 40},  # thin & deep
    {"gpu": 1, "name": "e1_d640_depth26",  "dim": 640,  "depth": 26},
    {"gpu": 2, "name": "e1_d768_depth18",  "dim": 768,  "depth": 18},
    {"gpu": 3, "name": "e1_d1024_depth10", "dim": 1024, "depth": 10},
    {"gpu": 4, "name": "e1_d1280_depth6",  "dim": 1280, "depth": 6},   # wide & shallow
]

SCRIPT = '''
import sys; sys.path.insert(0, '.')
import torch, numpy as np, mmap, time
from schedulefree import AdamWScheduleFree
from elman.models import LadderLM

torch.manual_seed(42); np.random.seed(42)
dim, depth, batch_size, steps = {dim}, {depth}, 256, 1000

model = LadderLM(vocab_size=256, dim=dim, depth=depth, level=1).cuda().bfloat16()
print(f'{name}: params={{model.get_num_params():,}}, dim={{dim}}, depth={{depth}}')

with open('data/pile.txt', 'rb') as f: mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
data_len = len(mm)

opt = AdamWScheduleFree(model.parameters(), lr=3e-4, weight_decay=0.1)
model.train(); opt.train()
buf = np.zeros((batch_size, 513), dtype=np.uint8)
losses = []; start = time.time(); tokens = 0

for step in range(1, steps+1):
    pos = np.random.randint(0, data_len - 513, size=batch_size)
    for j, p in enumerate(pos): buf[j] = np.frombuffer(mm[p:p+513], dtype=np.uint8)
    batch = torch.from_numpy(buf.astype(np.int64)).cuda()
    opt.zero_grad(); loss = model(batch, return_loss=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    tokens += batch_size * 512; losses.append(loss.item())
    if step % 100 == 0:
        print(f'Step {{step}} | Loss {{loss.item():.4f}} | Avg100 {{np.mean(losses[-100:]):.4f}} | {{int(tokens/(time.time()-start))}} tok/s')

print(f'DONE: {name} | Last100={{np.mean(losses[-100:]):.4f}} | Params={{model.get_num_params():,}}')
mm.close()
'''

OUT = "benchmark_results/e1_depth_width"; os.makedirs(OUT, exist_ok=True)
print("E1 Depth/Width Sweep - ~50M params, 1000 steps")
procs = []
for e in EXPERIMENTS:
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(e["gpu"])
    script = SCRIPT.format(dim=e["dim"], depth=e["depth"], name=e["name"])
    print(f"Launching {e['name']} on GPU {e['gpu']}...")
    p = subprocess.Popen(["python", "-u", "-c", script], stdout=open(f"{OUT}/{e['name']}.log", "w"), stderr=subprocess.STDOUT, env=env)
    procs.append((e["name"], p, f"{OUT}/{e['name']}.log")); time.sleep(0.5)

print(f"\nAll {len(EXPERIMENTS)} launched! Waiting...")
while any(p.poll() is None for _, p, _ in procs): time.sleep(60)
print("\n=== RESULTS ===")
for n, _, l in procs:
    with open(l) as f:
        for line in f:
            if "DONE:" in line: print(line.strip())
