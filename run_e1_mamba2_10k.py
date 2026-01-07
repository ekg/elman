#!/usr/bin/env python3
"""
E1 vs Mamba2: 10,000 steps comparison.
Run on separate GPUs with nohup.
"""
import subprocess, os, time

STEPS = 10000

EXPERIMENTS = [
    {"gpu": 0, "name": "e1_d1280_depth6", "type": "e1", "dim": 1280, "depth": 6},
    {"gpu": 1, "name": "mamba2_50m", "type": "mamba2"},
]

E1_SCRIPT = '''
import sys; sys.path.insert(0, '.')
import torch, numpy as np, mmap, time
from schedulefree import AdamWScheduleFree
from elman.models import LadderLM

torch.manual_seed(42); np.random.seed(42)
dim, depth, batch_size = {dim}, {depth}, 256
max_steps = {max_steps}

model = LadderLM(vocab_size=256, dim=dim, depth=depth, level=1).cuda().bfloat16()
print(f'{name}: params={{model.get_num_params():,}}')

with open('data/pile.txt', 'rb') as f: mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
data_len = len(mm)

opt = AdamWScheduleFree(model.parameters(), lr=3e-4, weight_decay=0.1)
model.train(); opt.train()
buf = np.zeros((batch_size, 513), dtype=np.uint8)
losses = []; start = time.time()

for step in range(1, max_steps + 1):
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
tokens = max_steps * batch_size * 512
print(f'DONE: {name} | Steps={{max_steps}} | Time={{elapsed:.0f}}s | Last100={{np.mean(losses[-100:]):.4f}} | {{int(tokens/elapsed)}} tok/s')
mm.close()
'''

MAMBA2_SCRIPT = '''
import sys; sys.path.insert(0, '.')
import torch, numpy as np, mmap, time
from schedulefree import AdamWScheduleFree
from elman.models import create_mamba2_model

torch.manual_seed(42); np.random.seed(42)
batch_size = 256
max_steps = {max_steps}

model = create_mamba2_model(target_params='50m', vocab_size=256).cuda().bfloat16()
print(f'mamba2_50m: params={{model.get_num_params():,}}')

with open('data/pile.txt', 'rb') as f: mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
data_len = len(mm)

opt = AdamWScheduleFree(model.parameters(), lr=3e-4, weight_decay=0.1)
model.train(); opt.train()
buf = np.zeros((batch_size, 513), dtype=np.uint8)
losses = []; start = time.time()

for step in range(1, max_steps + 1):
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
tokens = max_steps * batch_size * 512
print(f'DONE: mamba2_50m | Steps={{max_steps}} | Time={{elapsed:.0f}}s | Last100={{np.mean(losses[-100:]):.4f}} | {{int(tokens/elapsed)}} tok/s')
mm.close()
'''

OUT = "benchmark_results/e1_vs_mamba2_10k"; os.makedirs(OUT, exist_ok=True)
print(f"E1 vs Mamba2: {STEPS} steps")
print("=" * 60)

procs = []
for e in EXPERIMENTS:
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(e["gpu"])
    if e["type"] == "e1":
        script = E1_SCRIPT.format(dim=e["dim"], depth=e["depth"], name=e["name"], max_steps=STEPS)
    else:
        script = MAMBA2_SCRIPT.format(max_steps=STEPS)

    log = f"{OUT}/{e['name']}.log"
    print(f"Launching {e['name']} on GPU {e['gpu']} -> {log}")
    p = subprocess.Popen(["python", "-u", "-c", script], stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env)
    procs.append((e["name"], p, log)); time.sleep(0.5)

print(f"\nAll {len(EXPERIMENTS)} launched!")
print("Monitor with: tail -f benchmark_results/e1_vs_mamba2_10k/*.log")
