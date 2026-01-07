#!/usr/bin/env python3
"""
E9 balanced scan: Equal core and memory dims, like E1 but with diagonal memory
Testing 3-6 layers with balanced dimensions
"""
import subprocess, os, time

TIME_LIMIT = 600  # 10 minutes

# Balanced configs: core_dim == mem_dim (roughly)
# d_inner = core + mem = 2*core, so expansion = 2*core/dim, core_ratio = 0.5
EXPERIMENTS = [
    # 1280+1280 = 2560 total hidden
    {"gpu": 0, "name": "e9_1280x1280_d3", "dim": 1280, "expansion": 2.0, "core": 1280, "depth": 3},
    {"gpu": 1, "name": "e9_1280x1280_d4", "dim": 1280, "expansion": 2.0, "core": 1280, "depth": 4},
    {"gpu": 2, "name": "e9_1280x1280_d5", "dim": 1280, "expansion": 2.0, "core": 1280, "depth": 5},
    {"gpu": 3, "name": "e9_1280x1280_d6", "dim": 1280, "expansion": 2.0, "core": 1280, "depth": 6},
    # 1024+1024 = 2048 total hidden (smaller core, more layers possible)
    {"gpu": 4, "name": "e9_1024x1024_d4", "dim": 1024, "expansion": 2.0, "core": 1024, "depth": 4},
    {"gpu": 5, "name": "e9_1024x1024_d5", "dim": 1024, "expansion": 2.0, "core": 1024, "depth": 5},
    {"gpu": 6, "name": "e9_1024x1024_d6", "dim": 1024, "expansion": 2.0, "core": 1024, "depth": 6},
    # 1536+1536 = 3072 total hidden (wider, shallower)
    {"gpu": 7, "name": "e9_1536x1536_d3", "dim": 1536, "expansion": 2.0, "core": 1536, "depth": 3},
]

SCRIPT = '''
import sys; sys.path.insert(0, '.')
import torch, numpy as np, mmap, time
from schedulefree import AdamWScheduleFree
from elman.models import LadderLM

torch.manual_seed(42); np.random.seed(42)
batch_size, time_limit = 256, {time_limit}
dim, expansion, core, depth = {dim}, {expansion}, {core}, {depth}
d_inner = int(dim * expansion)
core_ratio = core / d_inner

model = LadderLM(vocab_size=256, dim=dim, depth=depth, level=9,
                 expansion=expansion, core_ratio=core_ratio).cuda().bfloat16()
layer = model.layers[0]
print(f'{name}: core={{layer.core_dim}} mem={{layer.mem_dim}} total={{layer.core_dim+layer.mem_dim}} params={{model.get_num_params():,}}')

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

OUT = "benchmark_results/e9_balanced"; os.makedirs(OUT, exist_ok=True)
print(f"E9 Balanced Scan: {len(EXPERIMENTS)} configs, {TIME_LIMIT}s each")
print("=" * 70)

procs = []
for e in EXPERIMENTS:
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(e["gpu"])
    script = SCRIPT.format(
        dim=e["dim"], expansion=e["expansion"], core=e["core"],
        depth=e["depth"], name=e["name"], time_limit=TIME_LIMIT
    )
    log_path = f"{OUT}/{e['name']}.log"
    print(f"GPU {e['gpu']}: {e['name']}")
    p = subprocess.Popen(["python", "-u", "-c", script],
                         stdout=open(log_path, "w"), stderr=subprocess.STDOUT, env=env)
    procs.append((e["name"], p, log_path))
    time.sleep(0.3)

print(f"\nAll {len(EXPERIMENTS)} launched! Waiting {TIME_LIMIT}s...")

while any(p.poll() is None for _, p, _ in procs):
    time.sleep(60)
    print("\n--- Progress ---")
    for name, _, log in procs:
        try:
            with open(log) as f:
                lines = f.readlines()
                for line in reversed(lines[-5:]):
                    if "Step" in line or "DONE" in line:
                        # Extract key info
                        if "DONE" in line:
                            print(f"  {line.strip()}")
                        else:
                            parts = line.strip().split("|")
                            step = parts[0].strip()
                            loss = [p for p in parts if "Avg100" in p][0].strip() if any("Avg100" in p for p in parts) else ""
                            toks = [p for p in parts if "tok/s" in p][0].strip() if any("tok/s" in p for p in parts) else ""
                            print(f"  {name}: {step} | {loss} | {toks}")
                        break
        except: pass

print("\n" + "=" * 70)
print("RESULTS - E9 Balanced (core == mem)")
print("=" * 70)
results = []
for n, _, l in procs:
    with open(l) as f:
        content = f.read()
        # Get params
        for line in content.split('\n'):
            if 'params=' in line and 'core=' in line:
                print(line.strip())
            if "DONE:" in line:
                print(line.strip())
                parts = line.split("|")
                loss = float([p for p in parts if "Last100" in p][0].split("=")[1])
                toks = int([p for p in parts if "tok/s" in p][0].split()[0])
                results.append((n, loss, toks))

print("\n" + "=" * 70)
print("RANKED BY LOSS (compare to E1 d1280x6: 1.43 loss, 254K tok/s):")
print("=" * 70)
for name, loss, toks in sorted(results, key=lambda x: x[1]):
    print(f"  {name}: loss={loss:.4f}, {toks//1000}K tok/s")
