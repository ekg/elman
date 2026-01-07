#!/usr/bin/env python3
"""
Simple GPU job scheduler for scale test.
Launches jobs as GPUs become available.
"""
import os, subprocess, time, json

TIME_LIMIT = 600
BATCH_SIZES = {50: 256, 100: 192, 200: 128, 400: 96}
OUT = "benchmark_results/scale_test"
os.makedirs(OUT, exist_ok=True)

# All jobs: (name, type, dim, depth, scale)
JOBS = [
    ("e1_50m", "e1", 1280, 6, 50),
    ("e9_50m", "e9", 1440, 3, 50),
    ("mamba2_50m", "mamba2", 0, 0, 50),
    ("e1_100m", "e1", 1280, 12, 100),
    ("e9_100m", "e9", 1440, 6, 100),
    ("mamba2_100m", "mamba2", 0, 0, 100),
    ("e1_200m", "e1", 1792, 12, 200),
    ("e9_200m", "e9", 2048, 6, 200),
    ("mamba2_200m", "mamba2", 0, 0, 200),
    ("e1_400m", "e1", 2304, 12, 400),
    ("e9_400m", "e9", 2560, 6, 400),
    ("mamba2_400m", "mamba2", 0, 0, 400),
]

SCRIPT_TEMPLATE = '''
import sys; sys.path.insert(0, '/home/erikg/elman')
import torch, numpy as np, mmap, time
from schedulefree import AdamWScheduleFree
{imports}

torch.manual_seed(42); np.random.seed(42)
batch_size, time_limit = {batch_size}, {time_limit}

{model_init}
print(f'{name}: params={{model.get_num_params():,}}', flush=True)

with open('/home/erikg/elman/data/pile.txt', 'rb') as f: mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
data_len = len(mm)

opt = AdamWScheduleFree(model.parameters(), lr=3e-4, weight_decay=0.1)
model.train(); opt.train()
buf = np.zeros((batch_size, 513), dtype=np.uint8)
losses = []; start = time.time(); step = 0

peak_mem = 0
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

def get_free_gpus():
    """Return list of GPUs with <1GB memory used."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        free = []
        for line in result.stdout.strip().split('\n'):
            idx, mem = line.split(',')
            if int(mem.strip()) < 1000:  # Less than 1GB used
                free.append(int(idx.strip()))
        return free
    except:
        return list(range(8))

def make_script(name, mtype, dim, depth, scale):
    batch_size = BATCH_SIZES[scale]
    if mtype == "e1":
        imports = "from elman.models import LadderLM"
        model_init = f"model = LadderLM(vocab_size=256, dim={dim}, depth={depth}, level=1).cuda().bfloat16()"
    elif mtype == "e9":
        imports = "from elman.models import LadderLM"
        model_init = f"model = LadderLM(vocab_size=256, dim={dim}, depth={depth}, level=9, expansion=2.0, core_ratio=0.5).cuda().bfloat16()"
    else:
        imports = "from elman.models import create_mamba2_model"
        model_init = f"model = create_mamba2_model(target_params='{scale}m', vocab_size=256).cuda().bfloat16()"

    return SCRIPT_TEMPLATE.format(
        imports=imports, model_init=model_init, batch_size=batch_size,
        time_limit=TIME_LIMIT, name=name
    ), batch_size

print("="*70)
print("SCALE TEST SCHEDULER - 12 jobs, 8 GPUs")
print("="*70)

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
        name, mtype, dim, depth, scale = job

        script, batch_size = make_script(name, mtype, dim, depth, scale)
        script_path = f"{OUT}/{name}.py"
        with open(script_path, 'w') as f:
            f.write(script)

        log_path = f"{OUT}/{name}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        proc = subprocess.Popen(
            ["python", "-u", script_path],
            stdout=open(log_path, "w"),
            stderr=subprocess.STDOUT,
            env=env
        )
        running[gpu] = (name, proc, time.time())
        print(f"[START] GPU {gpu}: {name} (batch={batch_size}, {scale}M params)")

    # Status update
    if running:
        time.sleep(30)
        print(f"\n--- Status: {len(completed)} done, {len(running)} running, {len(pending)} pending ---")
        for gpu, (name, proc, start) in running.items():
            elapsed = time.time() - start
            # Read last line of log
            try:
                with open(f"{OUT}/{name}.log") as f:
                    lines = f.readlines()
                    last = [l for l in lines if "Step" in l or "DONE" in l]
                    if last:
                        print(f"  GPU {gpu} ({name}): {last[-1].strip()[-60:]}")
                    else:
                        print(f"  GPU {gpu} ({name}): starting... ({elapsed:.0f}s)")
            except:
                print(f"  GPU {gpu} ({name}): {elapsed:.0f}s elapsed")

print("\n" + "="*70)
print("ALL JOBS COMPLETED")
print("="*70)

# Print final results
print("\nFINAL RESULTS:")
results = []
for name, _, _, _, scale in JOBS:
    try:
        with open(f"{OUT}/{name}.log") as f:
            for line in f:
                if "DONE:" in line:
                    print(line.strip())
                    parts = line.split("|")
                    loss = float([p for p in parts if "Last100" in p][0].split("=")[1])
                    toks = int([p for p in parts if "tok/s" in p][0].split()[0])
                    model = name.split("_")[0]
                    results.append((scale, model, loss, toks))
    except Exception as e:
        print(f"{name}: ERROR - {e}")

print("\n" + "="*70)
print("BY SCALE (sorted by loss):")
print("="*70)
for scale in [50, 100, 200, 400]:
    print(f"\n{scale}M params:")
    scale_results = [(m, l, t) for s, m, l, t in results if s == scale]
    for model, loss, toks in sorted(scale_results, key=lambda x: x[1]):
        print(f"  {model}: loss={loss:.4f}, {toks//1000}K tok/s")
