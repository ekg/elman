#!/usr/bin/env python3
"""Quick 400M scale comparison: E1 vs Mamba2 vs minGRU vs minLSTM."""
import subprocess
import os
import time

OUT = "scaling_results"
os.makedirs(OUT, exist_ok=True)
TIME_LIMIT = 120

SCRIPT_TEMPLATE = '''
import sys; sys.path.insert(0, '/home/erikg/elman')
import torch, numpy as np, mmap, time
torch.manual_seed(42); np.random.seed(42)
torch.backends.cudnn.benchmark = True

with open('/home/erikg/elman/data/pile.txt', 'rb') as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
data_len = len(mm)

def get_batch(bs, sl):
    pos = np.random.randint(0, data_len - sl - 1, size=bs)
    buf = np.zeros((bs, sl + 1), dtype=np.uint8)
    for i, p in enumerate(pos): buf[i] = np.frombuffer(mm[p:p+sl+1], dtype=np.uint8)
    return torch.from_numpy(buf.astype(np.int64)).cuda()

batch_size, seq_len, time_limit = BATCH_SIZE, 512, TIME_LIMIT
MODEL_CODE
print(f"{name}: {model.get_num_params():,} params", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
for _ in range(5):
    loss = model(get_batch(batch_size, seq_len), return_loss=True)
    loss.backward(); opt.step(); opt.zero_grad()
torch.cuda.synchronize()
print(f"Memory: {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)
losses, step = [], 0
torch.cuda.synchronize()
start = time.time()
while time.time() - start < time_limit:
    loss = model(get_batch(batch_size, seq_len), return_loss=True)
    loss.backward(); opt.step(); opt.zero_grad()
    losses.append(loss.item()); step += 1
    if step % 50 == 0:
        el = time.time() - start
        print(f"Step {step} ({el:.0f}s): loss={sum(losses[-50:])/50:.4f}, {(step*batch_size*seq_len)/el/1000:.1f}K tok/s", flush=True)
torch.cuda.synchronize()
el = time.time() - start
print(f"FINAL: {step} steps, loss={sum(losses[-50:])/min(50,len(losses)):.4f}, {(step*batch_size*seq_len)/el/1000:.1f}K tok/s", flush=True)
mm.close()
'''

MODELS = {
    'e1': ('from elman.models import LadderLM\nname = "E1"\nmodel = LadderLM(vocab_size=256, dim=1760, depth=26, level=1).cuda().bfloat16()', 16),
    'mamba2': ('from elman.models import create_mamba2_model\nname = "Mamba2"\nmodel = create_mamba2_model(target_params="400M", vocab_size=256).cuda().bfloat16()', 16),
    'mingru': ('from elman.models import MinGRULM\nname = "minGRU"\nmodel = MinGRULM(vocab_size=256, dim=2752, depth=24).cuda().bfloat16()', 16),
    'minlstm': ('from elman.models import MinLSTMLM\nname = "minLSTM"\nmodel = MinLSTMLM(vocab_size=256, dim=2752, depth=24).cuda().bfloat16()', 16),
}

procs = {}
for gpu, (name, (code, batch)) in enumerate(MODELS.items()):
    script = SCRIPT_TEMPLATE.replace('BATCH_SIZE', str(batch)).replace('TIME_LIMIT', str(TIME_LIMIT)).replace('MODEL_CODE', code)
    path = f"{OUT}/{name}_400m.py"
    with open(path, 'w') as f: f.write(script)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log = f"{OUT}/{name}_400m.log"
    procs[name] = subprocess.Popen(["python3", "-u", path], stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env)
    print(f"[GPU {gpu}] {name} started")

print("Waiting for completion...")
for name, p in procs.items():
    p.wait()
    print(f"[DONE] {name}")

print("\n=== 400M RESULTS ===")
for name in MODELS:
    log = f"{OUT}/{name}_400m.log"
    with open(log) as f:
        for line in f:
            if 'FINAL:' in line or 'params' in line:
                print(f"{name}: {line.strip()}")
