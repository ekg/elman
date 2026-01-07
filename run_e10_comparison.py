#!/usr/bin/env python3
"""
E10 Comparison Test: Vary number of EMA banks (k)
Compare against E1 and E9 baselines at ~50M params.
"""
import os
import subprocess
import time
import sys

sys.path.insert(0, '/home/erikg/elman')

TIME_LIMIT = 600  # 10 minutes
BATCH_SIZE = 128  # Conservative for all models
OUT = "benchmark_results/e10_comparison"
os.makedirs(OUT, exist_ok=True)

# First, find the right configs for ~50M params
print("=" * 70)
print("Finding optimal configs for ~50M params...")
print("=" * 70)

from elman.models import LadderLM
import torch

def count_params(level, dim, depth, n_banks=4, expansion=1.0, core_ratio=0.5):
    """Count params for a given config."""
    try:
        model = LadderLM(
            vocab_size=256, dim=dim, depth=depth, level=level,
            expansion=expansion, n_banks=n_banks, core_ratio=core_ratio
        )
        return model.get_num_params()
    except Exception as e:
        return None

def find_config(level, target=50_000_000, n_banks=4, expansion=1.0, core_ratio=0.5):
    """Find dim/depth that gives ~target params."""
    best = None
    best_diff = float('inf')

    for depth in range(3, 13):
        for dim in range(512, 2560, 64):
            params = count_params(level, dim, depth, n_banks, expansion, core_ratio)
            if params is None:
                continue
            diff = abs(params - target)
            if diff < best_diff:
                best_diff = diff
                best = (dim, depth, params)

    return best

# Find configs
print("\nE1 (Mamba-Gated Elman):")
e1_config = find_config(1, expansion=1.0)
print(f"  dim={e1_config[0]}, depth={e1_config[1]}, params={e1_config[2]:,}")

print("\nE9 (Hybrid Elman):")
e9_config = find_config(9, expansion=2.0, core_ratio=0.5)
print(f"  dim={e9_config[0]}, depth={e9_config[1]}, expansion=2.0, core_ratio=0.5, params={e9_config[2]:,}")

print("\nE10 with k=4 banks:")
e10_k4_config = find_config(10, n_banks=4, expansion=1.0)
print(f"  dim={e10_k4_config[0]}, depth={e10_k4_config[1]}, n_banks=4, params={e10_k4_config[2]:,}")

print("\nE10 with k=64 banks:")
e10_k64_config = find_config(10, n_banks=64, expansion=1.0)
print(f"  dim={e10_k64_config[0]}, depth={e10_k64_config[1]}, n_banks=64, params={e10_k64_config[2]:,}")

print("\nE10 with k=256 banks:")
e10_k256_config = find_config(10, n_banks=256, expansion=1.0)
print(f"  dim={e10_k256_config[0]}, depth={e10_k256_config[1]}, n_banks=256, params={e10_k256_config[2]:,}")

# Job definitions
JOBS = [
    ("e1_50m", "e1", e1_config[0], e1_config[1], 1.0, 4, 0.5),
    ("e9_50m", "e9", e9_config[0], e9_config[1], 2.0, 4, 0.5),
    ("e10_k4", "e10", e10_k4_config[0], e10_k4_config[1], 1.0, 4, 0.5),
    ("e10_k64", "e10", e10_k64_config[0], e10_k64_config[1], 1.0, 64, 0.5),
    ("e10_k256", "e10", e10_k256_config[0], e10_k256_config[1], 1.0, 256, 0.5),
]

SCRIPT_TEMPLATE = '''
import sys; sys.path.insert(0, '/home/erikg/elman')
import torch, numpy as np, mmap, time
from schedulefree import AdamWScheduleFree
from elman.models import LadderLM

torch.manual_seed(42); np.random.seed(42)
batch_size, time_limit = {batch_size}, {time_limit}

model = LadderLM(
    vocab_size=256, dim={dim}, depth={depth}, level={level},
    expansion={expansion}, n_banks={n_banks}, core_ratio={core_ratio}
).cuda().bfloat16()

layer = model.layers[0]
if hasattr(layer, 'n_banks'):
    print(f'{name}: dim={{model.dim}} depth={{model.depth}} n_banks={{layer.n_banks}} params={{model.get_num_params():,}}', flush=True)
elif hasattr(layer, 'core_dim'):
    print(f'{name}: dim={{model.dim}} depth={{model.depth}} core={{layer.core_dim}} mem={{layer.mem_dim}} params={{model.get_num_params():,}}', flush=True)
else:
    print(f'{name}: dim={{model.dim}} depth={{model.depth}} params={{model.get_num_params():,}}', flush=True)

with open('/home/erikg/elman/data/pile.txt', 'rb') as f: mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
data_len = len(mm)

opt = AdamWScheduleFree(model.parameters(), lr=3e-4, weight_decay=0.1)
model.train(); opt.train()
buf = np.zeros((batch_size, 513), dtype=np.uint8)
losses = []; start = time.time(); step = 0; peak_mem = 0

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
        mem_gb = torch.cuda.max_memory_allocated() / 1e9
        peak_mem = max(peak_mem, mem_gb)
        print(f'Step {{step}} | {{elapsed:.0f}}s | Loss {{loss.item():.4f}} | Avg100 {{np.mean(losses[-100:]):.4f}} | {{int(tokens/elapsed)}} tok/s | Mem {{mem_gb:.1f}}GB', flush=True)

elapsed = time.time() - start
tokens = step * batch_size * 512
print(f'DONE: {name} | Steps={{step}} | Time={{elapsed:.0f}}s | Last100={{np.mean(losses[-100:]):.4f}} | {{int(tokens/elapsed)}} tok/s | PeakMem={{peak_mem:.1f}}GB', flush=True)
mm.close()
'''

def make_script(name, mtype, dim, depth, expansion, n_banks, core_ratio):
    level = {"e1": 1, "e9": 9, "e10": 10}[mtype]
    return SCRIPT_TEMPLATE.format(
        name=name, dim=dim, depth=depth, level=level,
        expansion=expansion, n_banks=n_banks, core_ratio=core_ratio,
        batch_size=BATCH_SIZE, time_limit=TIME_LIMIT
    )

print("\n" + "=" * 70)
print("E10 COMPARISON TEST - GPU Job Scheduler")
print("=" * 70)

# Track running jobs: {gpu: (name, process, start_time)}
running = {}
pending = list(JOBS)
completed = []

while pending or running:
    # Check for completed jobs
    for gpu in list(running.keys()):
        name, proc, start = running[gpu]
        if proc.poll() is not None:
            elapsed = time.time() - start
            print(f"[DONE] GPU {gpu}: {name} ({elapsed:.0f}s)")
            completed.append(name)
            del running[gpu]

    # Launch new jobs on free GPUs
    free_gpus = [g for g in range(8) if g not in running]
    while pending and free_gpus:
        gpu = free_gpus.pop(0)
        job = pending.pop(0)
        name, mtype, dim, depth, expansion, n_banks, core_ratio = job

        script = make_script(name, mtype, dim, depth, expansion, n_banks, core_ratio)
        script_path = f"{OUT}/{name}.py"
        with open(script_path, 'w') as f:
            f.write(script)

        log_path = f"{OUT}/{name}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["LD_LIBRARY_PATH"] = "/home/erikg/.local/lib/python3.12/site-packages/torch/lib:" + env.get("LD_LIBRARY_PATH", "")

        proc = subprocess.Popen(
            ["python3", "-u", script_path],
            stdout=open(log_path, "w"),
            stderr=subprocess.STDOUT,
            env=env
        )
        running[gpu] = (name, proc, time.time())
        print(f"[START] GPU {gpu}: {name} (dim={dim}, depth={depth}, n_banks={n_banks})")

    # Status update
    if running:
        time.sleep(30)
        print(f"\n--- Status: {len(completed)} done, {len(running)} running, {len(pending)} pending ---")
        for gpu, (name, proc, start) in running.items():
            elapsed = time.time() - start
            try:
                with open(f"{OUT}/{name}.log") as f:
                    lines = f.readlines()
                    last = [l for l in lines if "Step" in l or "DONE" in l]
                    if last:
                        print(f"  GPU {gpu} ({name}): {last[-1].strip()[-70:]}")
                    else:
                        print(f"  GPU {gpu} ({name}): starting... ({elapsed:.0f}s)")
            except:
                print(f"  GPU {gpu} ({name}): {elapsed:.0f}s elapsed")

print("\n" + "=" * 70)
print("ALL JOBS COMPLETED")
print("=" * 70)

# Print final results
print("\nFINAL RESULTS:")
results = []
for name, mtype, dim, depth, expansion, n_banks, core_ratio in JOBS:
    try:
        with open(f"{OUT}/{name}.log") as f:
            content = f.read()
            for line in content.split('\n'):
                if "DONE:" in line:
                    print(line.strip())
                    parts = line.split("|")
                    loss = float([p for p in parts if "Last100" in p][0].split("=")[1])
                    toks = int([p for p in parts if "tok/s" in p][0].split()[0])
                    mem = float([p for p in parts if "PeakMem" in p][0].split("=")[1].replace("GB", ""))
                    results.append((name, mtype, n_banks, loss, toks, mem, dim, depth))
    except Exception as e:
        print(f"{name}: ERROR - {e}")

print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(f"{'Model':<12} {'Type':<6} {'Banks':<6} {'Dim':<6} {'Depth':<6} {'Loss':<8} {'Tok/s':<10} {'Mem':<6}")
print("-" * 70)
for name, mtype, n_banks, loss, toks, mem, dim, depth in sorted(results, key=lambda x: x[3]):
    print(f"{name:<12} {mtype:<6} {n_banks:<6} {dim:<6} {depth:<6} {loss:<8.4f} {toks:<10,} {mem:<6.1f}GB")
