#!/usr/bin/env python3
"""E11 (Selective Memory) sweep: compare k=4,8,16,32,64 against E1 and E10 baselines.
All models ~50M params, batch sizes maximized for GPU utilization.
"""
import os, subprocess, time, sys
sys.path.insert(0, '/home/erikg/elman')

TIME_LIMIT = 600
OUT = "benchmark_results/e11_sweep"
os.makedirs(OUT, exist_ok=True)

print("=" * 70)
print("FINDING OPTIMAL CONFIGS (~50M params)")
print("=" * 70)

from elman.models import LadderLM

def get_params(level, dim, depth, expansion, n_banks, core_ratio):
    model = LadderLM(vocab_size=256, dim=dim, depth=depth, level=level,
                     expansion=expansion, n_banks=n_banks, core_ratio=core_ratio)
    p = model.get_num_params()
    del model
    return p

def find_dim_for_50m(level, depth, expansion, n_banks, core_ratio):
    """Search for dim giving ~50M params."""
    target = 50_000_000
    best_dim, best_diff = 512, float('inf')
    for dim in range(256, 2048, 32):
        try:
            p = get_params(level, dim, depth, expansion, n_banks, core_ratio)
            diff = abs(p - target)
            if diff < best_diff:
                best_diff = diff
                best_dim = dim
            if p > target * 1.2 and diff > best_diff:
                break
        except:
            continue
    return best_dim

configs = []

# E1 baseline (depth=6)
dim = find_dim_for_50m(1, 6, 1.0, 4, 0.5)
params = get_params(1, dim, 6, 1.0, 4, 0.5)
configs.append(("e1", 1, dim, 6, 1.0, 1, 0.5, params))
print(f"E1:       dim={dim}, depth=6, params={params:,}")

# E10 baseline with k=4 (for comparison)
dim = find_dim_for_50m(10, 6, 1.0, 4, 0.5)
params = get_params(10, dim, 6, 1.0, 4, 0.5)
configs.append(("e10_k4", 10, dim, 6, 1.0, 4, 0.5, params))
print(f"E10 k=4:  dim={dim}, depth=6, params={params:,}")

# E11 with varying k
for k in [4, 8, 16, 32, 64]:
    dim = find_dim_for_50m(11, 6, 1.0, k, 0.5)
    params = get_params(11, dim, 6, 1.0, k, 0.5)
    configs.append((f"e11_k{k}", 11, dim, 6, 1.0, k, 0.5, params))
    print(f"E11 k={k:<2}: dim={dim}, depth=6, params={params:,}")

# Batch sizes - E11 should be more memory-efficient due to fewer gates
batch_sizes = {
    "e1": 256,
    "e10_k4": 192,
    "e11_k4": 256,   # More efficient than E10 (fewer gates)
    "e11_k8": 224,
    "e11_k16": 192,
    "e11_k32": 160,
    "e11_k64": 128,
}

print(f"\nTotal: {len(configs)} configs")

SCRIPT = '''
import sys; sys.path.insert(0, '/home/erikg/elman')
import torch, numpy as np, mmap, time
from schedulefree import AdamWScheduleFree

# Disable CUDA kernel for E11 (has a bug, use PyTorch fallback)
import elman.models.selective_elman as se
se.SELECTIVE_CUDA_AVAILABLE = False

from elman.models import LadderLM

torch.manual_seed(42); np.random.seed(42)

model = LadderLM(vocab_size=256, dim={dim}, depth={depth}, level={level},
                 expansion={expansion}, n_banks={n_banks}, core_ratio={core_ratio}).cuda().bfloat16()
print(f'{name}: dim={dim} depth={depth} k={n_banks} batch={batch_size} params={{model.get_num_params():,}}', flush=True)

with open('/home/erikg/elman/data/pile.txt', 'rb') as f: mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
data_len = len(mm)

opt = AdamWScheduleFree(model.parameters(), lr=3e-4, weight_decay=0.1)
model.train(); opt.train()
buf = np.zeros(({batch_size}, 513), dtype=np.uint8)
losses = []; start = time.time(); step = 0; peak_mem = 0

while time.time() - start < {time_limit}:
    step += 1
    pos = np.random.randint(0, data_len - 513, size={batch_size})
    for j, p in enumerate(pos): buf[j] = np.frombuffer(mm[p:p+513], dtype=np.uint8)
    batch = torch.from_numpy(buf.astype(np.int64)).cuda()
    opt.zero_grad(); loss = model(batch, return_loss=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    losses.append(loss.item())
    if step % 100 == 0:
        elapsed = time.time() - start
        tokens = step * {batch_size} * 512
        mem_gb = torch.cuda.max_memory_allocated() / 1e9
        peak_mem = max(peak_mem, mem_gb)
        print(f'Step {{step}} | {{elapsed:.0f}}s | Loss {{loss.item():.4f}} | Avg100 {{np.mean(losses[-100:]):.4f}} | {{int(tokens/elapsed)}} tok/s | Mem {{mem_gb:.1f}}GB', flush=True)

elapsed = time.time() - start
tokens = step * {batch_size} * 512
print(f'DONE: {name} | Steps={{step}} | Time={{elapsed:.0f}}s | Last100={{np.mean(losses[-100:]):.4f}} | {{int(tokens/elapsed)}} tok/s | PeakMem={{peak_mem:.1f}}GB', flush=True)
mm.close()
'''

print("\n" + "=" * 70)
print(f"LAUNCHING {len(configs)} JOBS ON 8 GPUs")
print("=" * 70)

running = {}
pending = list(configs)
completed = []

while pending or running:
    for gpu in list(running.keys()):
        name, proc, start = running[gpu]
        if proc.poll() is not None:
            print(f"[DONE] GPU {gpu}: {name} ({time.time()-start:.0f}s)")
            completed.append(name)
            del running[gpu]

    free = [g for g in range(8) if g not in running]
    while pending and free:
        gpu = free.pop(0)
        name, level, dim, depth, expansion, n_banks, core_ratio, params = pending.pop(0)
        batch_size = batch_sizes[name]

        script = SCRIPT.format(name=name, level=level, dim=dim, depth=depth,
                               expansion=expansion, n_banks=n_banks, core_ratio=core_ratio,
                               batch_size=batch_size, time_limit=TIME_LIMIT)
        path = f"{OUT}/{name}.py"
        with open(path, 'w') as f: f.write(script)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        proc = subprocess.Popen(["python3", "-u", path], stdout=open(f"{OUT}/{name}.log", "w"),
                               stderr=subprocess.STDOUT, env=env)
        running[gpu] = (name, proc, time.time())
        print(f"[START] GPU {gpu}: {name} (dim={dim}, k={n_banks}, batch={batch_size})")

    if running:
        time.sleep(30)
        print(f"\n--- {len(completed)}/{len(configs)} done, {len(running)} running ---")
        for gpu, (name, _, start) in sorted(running.items()):
            try:
                with open(f"{OUT}/{name}.log") as f:
                    lines = [l for l in f.readlines() if "Step" in l or "DONE" in l]
                    if lines: print(f"  GPU {gpu} {name}: ...{lines[-1].strip()[-50:]}]")
            except: pass

print("\n" + "=" * 70)
print("FINAL RESULTS (sorted by loss)")
print("=" * 70)
results = []
for name, level, dim, depth, expansion, n_banks, core_ratio, params in configs:
    try:
        with open(f"{OUT}/{name}.log") as f:
            for line in f:
                if "DONE:" in line:
                    parts = line.split("|")
                    loss = float([p for p in parts if "Last100" in p][0].split("=")[1])
                    toks = int([p for p in parts if "tok/s" in p][0].split()[0])
                    mem = float([p for p in parts if "PeakMem" in p][0].split("=")[1].replace("GB",""))
                    results.append((name, level, n_banks, dim, params, loss, toks, mem))
    except: pass

print(f"{'Name':<12} {'Level':<6} {'k':<4} {'dim':<5} {'params':<12} {'Loss':<8} {'Tok/s':<10} {'Mem'}")
print("-" * 80)
for name, level, k, dim, params, loss, toks, mem in sorted(results, key=lambda x: x[5]):
    print(f"{name:<12} E{level:<5} {k:<4} {dim:<5} {params:<12,} {loss:<8.4f} {toks:<10,} {mem:.1f}GB")
