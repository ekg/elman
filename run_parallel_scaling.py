#!/usr/bin/env python3
"""Parallel scaling test: E1 vs Mamba2 vs minGRU vs minLSTM at 50M, 100M, 200M, 400M."""
import os
import subprocess
import time

TIME_LIMIT = 120  # 2 minutes per config
OUT = "scaling_results"
os.makedirs(OUT, exist_ok=True)

# Configs: (name, model_type, scale, batch_size)
# Use same batch sizes across models for fair comparison
configs = [
    # 50M scale
    ("e1_50m", "e1", "50M", 48),
    ("mamba2_50m", "mamba2", "50M", 48),
    ("mingru_50m", "mingru", "50M", 48),
    ("minlstm_50m", "minlstm", "50M", 48),
    # 100M scale
    ("e1_100m", "e1", "100M", 32),
    ("mamba2_100m", "mamba2", "100M", 32),
    ("mingru_100m", "mingru", "100M", 32),
    ("minlstm_100m", "minlstm", "100M", 32),
    # 200M scale
    ("e1_200m", "e1", "200M", 24),
    ("mamba2_200m", "mamba2", "200M", 24),
    ("mingru_200m", "mingru", "200M", 24),
    ("minlstm_200m", "minlstm", "200M", 24),
    # 400M scale
    ("e1_400m", "e1", "400M", 16),
    ("mamba2_400m", "mamba2", "400M", 16),
    ("mingru_400m", "mingru", "400M", 16),
    ("minlstm_400m", "minlstm", "400M", 16),
]

# E1 configs for each scale (depth=26 for 100M+)
E1_CONFIGS = {
    "50M": (1280, 6),    # 49.5M params
    "100M": (876, 26),   # 100.1M params
    "200M": (1248, 26),  # 202.9M params
    "400M": (1760, 26),  # 403.3M params
}

SCRIPT_TEMPLATE = '''
import sys; sys.path.insert(0, '/home/erikg/elman')
import torch, numpy as np, mmap, time
torch.manual_seed(42); np.random.seed(42)
torch.backends.cudnn.benchmark = True

# Data loading
with open('/home/erikg/elman/data/pile.txt', 'rb') as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
data_len = len(mm)

def get_batch(batch_size, seq_len):
    pos = np.random.randint(0, data_len - seq_len - 1, size=batch_size)
    buf = np.zeros((batch_size, seq_len + 1), dtype=np.uint8)
    for i, p in enumerate(pos):
        buf[i] = np.frombuffer(mm[p:p+seq_len+1], dtype=np.uint8)
    return torch.from_numpy(buf.astype(np.int64)).cuda()

batch_size = {batch_size}
seq_len = 512
time_limit = {time_limit}

{model_code}

print(f'{name}: {{model.get_num_params():,}} params', flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Warmup
for _ in range(5):
    x = get_batch(batch_size, seq_len)
    loss = model(x, return_loss=True)
    loss.backward()
    opt.step()
    opt.zero_grad()
torch.cuda.synchronize()

mem = torch.cuda.max_memory_allocated() / 1e9
print(f'Memory: {{mem:.1f}} GB', flush=True)

# Training
losses = []
step = 0
torch.cuda.synchronize()
start = time.time()

while time.time() - start < time_limit:
    x = get_batch(batch_size, seq_len)
    loss = model(x, return_loss=True)
    loss.backward()
    opt.step()
    opt.zero_grad()
    losses.append(loss.item())
    step += 1

    if step % 50 == 0:
        elapsed = time.time() - start
        avg_loss = sum(losses[-50:]) / 50
        tok_s = (step * batch_size * seq_len) / elapsed
        print(f'Step {{step}} ({{elapsed:.0f}}s): loss={{avg_loss:.4f}}, {{tok_s/1000:.1f}}K tok/s', flush=True)

torch.cuda.synchronize()
elapsed = time.time() - start
final_loss = sum(losses[-50:]) / min(50, len(losses))
tok_per_s = (step * batch_size * seq_len) / elapsed

print(f'FINAL: {{step}} steps, loss={{final_loss:.4f}}, {{tok_per_s/1000:.1f}}K tok/s, mem={{mem:.1f}}GB', flush=True)
mm.close()
'''

E1_MODEL = '''
from elman.models import LadderLM
name = "{name}"
dim, depth = {dim}, {depth}
model = LadderLM(vocab_size=256, dim=dim, depth=depth, level=1).cuda().bfloat16()
'''

MAMBA2_MODEL = '''
from elman.models import create_mamba2_model
name = "{name}"
model = create_mamba2_model(target_params="{scale}", vocab_size=256).cuda().bfloat16()
'''

MINGRU_MODEL = '''
from elman.models import MinGRULM
name = "{name}"
model = MinGRULM(vocab_size=256, target_params="{scale}").cuda().bfloat16()
'''

MINLSTM_MODEL = '''
from elman.models import MinLSTMLM
name = "{name}"
model = MinLSTMLM(vocab_size=256, target_params="{scale}").cuda().bfloat16()
'''

def get_model_code(name, model_type, scale):
    if model_type == "e1":
        dim, depth = E1_CONFIGS[scale]
        return E1_MODEL.format(name=name, dim=dim, depth=depth)
    elif model_type == "mamba2":
        return MAMBA2_MODEL.format(name=name, scale=scale)
    elif model_type == "mingru":
        return MINGRU_MODEL.format(name=name, scale=scale)
    elif model_type == "minlstm":
        return MINLSTM_MODEL.format(name=name, scale=scale)

print("=" * 70)
print(f"PARALLEL SCALING TEST: {len(configs)} configs")
print("=" * 70)

running = {}
pending = list(configs)
completed = []

while pending or running:
    # Check completed
    for gpu in list(running.keys()):
        name, proc, start_time = running[gpu]
        if proc.poll() is not None:
            elapsed = time.time() - start_time
            print(f"[DONE] GPU {gpu}: {name} ({elapsed:.0f}s)")
            completed.append(name)
            del running[gpu]

    # Launch new jobs on free GPUs
    free = [g for g in range(8) if g not in running]
    while pending and free:
        gpu = free.pop(0)
        name, model_type, scale, batch_size = pending.pop(0)

        model_code = get_model_code(name, model_type, scale)
        script = SCRIPT_TEMPLATE.format(
            batch_size=batch_size,
            time_limit=TIME_LIMIT,
            model_code=model_code
        )

        path = f"{OUT}/{name}.py"
        with open(path, 'w') as f:
            f.write(script)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        log_path = f"{OUT}/{name}.log"
        proc = subprocess.Popen(
            ["python3", "-u", path],
            stdout=open(log_path, "w"),
            stderr=subprocess.STDOUT,
            env=env
        )
        running[gpu] = (name, proc, time.time())
        print(f"[START] GPU {gpu}: {name} (batch={batch_size})")

    if running:
        time.sleep(10)

print("\n" + "=" * 70)
print("ALL DONE - Parsing results...")
print("=" * 70)

# Parse results
results = {}
for name, _, scale, batch in configs:
    log_path = f"{OUT}/{name}.log"
    try:
        with open(log_path) as f:
            content = f.read()

        # Parse FINAL line
        for line in content.split('\n'):
            if 'FINAL:' in line:
                parts = line.split(',')
                loss = float([p for p in parts if 'loss=' in p][0].split('=')[1])
                toks = float([p for p in parts if 'K tok/s' in p][0].split('K')[0].strip().split()[-1])
                results[name] = {'scale': scale, 'loss': loss, 'tok_s': toks * 1000}
            if 'params' in line.lower():
                params = int(line.split(':')[1].split()[0].replace(',', ''))
                if name in results:
                    results[name]['params'] = params
    except Exception as e:
        print(f"Error parsing {name}: {e}")

# Print comparison table
print("\n" + "=" * 70)
print("SCALING COMPARISON RESULTS")
print("=" * 70)

for scale in ["50M", "100M", "200M", "400M"]:
    print(f"\n=== {scale} ===")
    print(f"{'Model':<12} {'Params':>10} {'Tok/s':>10} {'Loss':>8}")
    print("-" * 45)

    scale_results = [(n, r) for n, r in results.items() if r.get('scale') == scale]
    scale_results.sort(key=lambda x: -x[1].get('tok_s', 0))

    for name, r in scale_results:
        model = name.split('_')[0].upper()
        params = r.get('params', 0) / 1e6
        toks = r.get('tok_s', 0) / 1000
        loss = r.get('loss', 0)
        print(f"{model:<12} {params:>9.1f}M {toks:>9.1f}K {loss:>8.4f}")

print("\n" + "=" * 70)
