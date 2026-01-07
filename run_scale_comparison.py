#!/usr/bin/env python3
"""Scale comparison: E1 vs E10 k=4 vs Mamba2 at 50M, 100M, 200M, 400M params."""
import os, subprocess, time

TIME_LIMIT = 600
OUT = "benchmark_results/scale_comparison"
os.makedirs(OUT, exist_ok=True)

# Empirically tested max batch sizes for 48GB GPUs
configs = [
    ("e1_50m", "e1", 1280, 6, 512),
    ("e10_50m", "e10", 960, 6, 288),
    ("mamba2_50m", "mamba2", 1152, 6, 512),
    ("e1_100m", "e1", 1792, 6, 448),
    ("e10_100m", "e10", 1344, 6, 256),
    ("mamba2_100m", "mamba2", 1600, 6, 448),
    ("e1_200m", "e1", 2560, 6, 256),
    ("e10_200m", "e10", 1920, 6, 160),
    ("mamba2_200m", "mamba2", 2304, 6, 256),
    ("e1_400m", "e1", 3584, 6, 192),
    ("e10_400m", "e10", 2688, 6, 112),
    ("mamba2_400m", "mamba2", 3264, 6, 208),
]

E1_SCRIPT = '''
import sys; sys.path.insert(0, '/home/erikg/elman')
import torch, numpy as np, mmap, time
from schedulefree import AdamWScheduleFree
from elman.models import LadderLM
torch.manual_seed(42); np.random.seed(42)
model = LadderLM(vocab_size=256, dim={dim}, depth={depth}, level=1).cuda().bfloat16()
batch_size = {batch_size}
print(f'{name}: dim={dim} depth={depth} batch={{batch_size}} params={{model.get_num_params():,}}', flush=True)
with open('/home/erikg/elman/data/pile.txt', 'rb') as f: mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
data_len = len(mm)
opt = AdamWScheduleFree(model.parameters(), lr=3e-4, weight_decay=0.1)
model.train(); opt.train()
buf = np.zeros((batch_size, 513), dtype=np.uint8)
losses = []; start = time.time(); step = 0; peak_mem = 0
while time.time() - start < {time_limit}:
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

E10_SCRIPT = '''
import sys; sys.path.insert(0, '/home/erikg/elman')
import torch, numpy as np, mmap, time
from schedulefree import AdamWScheduleFree
from elman.models import LadderLM
torch.manual_seed(42); np.random.seed(42)
model = LadderLM(vocab_size=256, dim={dim}, depth={depth}, level=10, n_banks=4).cuda().bfloat16()
batch_size = {batch_size}
print(f'{name}: dim={dim} depth={depth} k=4 batch={{batch_size}} params={{model.get_num_params():,}}', flush=True)
with open('/home/erikg/elman/data/pile.txt', 'rb') as f: mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
data_len = len(mm)
opt = AdamWScheduleFree(model.parameters(), lr=3e-4, weight_decay=0.1)
model.train(); opt.train()
buf = np.zeros((batch_size, 513), dtype=np.uint8)
losses = []; start = time.time(); step = 0; peak_mem = 0
while time.time() - start < {time_limit}:
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

MAMBA2_SCRIPT = '''
import torch, torch.nn as nn, numpy as np, mmap, time
from schedulefree import AdamWScheduleFree
from mamba_ssm import Mamba2
torch.manual_seed(42); np.random.seed(42)
class Mamba2LM(nn.Module):
    def __init__(self, vocab_size, dim, depth):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([Mamba2(dim, d_state=128, headdim=64) for _ in range(depth)])
        self.norm = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.embed.weight
    def forward(self, x, return_loss=False):
        if return_loss: x, targets = x[:, :-1], x[:, 1:]
        h = self.embed(x)
        for layer in self.layers: h = h + layer(h)
        h = self.norm(h)
        logits = self.head(h)
        if return_loss: return nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits
    def get_num_params(self): return sum(p.numel() for p in self.parameters())
model = Mamba2LM(vocab_size=256, dim={dim}, depth={depth}).cuda().bfloat16()
batch_size = {batch_size}
print(f'{name}: dim={dim} depth={depth} batch={{batch_size}} params={{model.get_num_params():,}}', flush=True)
with open('/home/erikg/elman/data/pile.txt', 'rb') as f: mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
data_len = len(mm)
opt = AdamWScheduleFree(model.parameters(), lr=3e-4, weight_decay=0.1)
model.train(); opt.train()
buf = np.zeros((batch_size, 513), dtype=np.uint8)
losses = []; start = time.time(); step = 0; peak_mem = 0
while time.time() - start < {time_limit}:
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

SCRIPTS = {"e1": E1_SCRIPT, "e10": E10_SCRIPT, "mamba2": MAMBA2_SCRIPT}

print("=" * 70)
print(f"SCALE COMPARISON: {len(configs)} configs on 8 GPUs (10 min each)")
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
        name, model_type, dim, depth, batch_size = pending.pop(0)
        script = SCRIPTS[model_type].format(name=name, dim=dim, depth=depth, batch_size=batch_size, time_limit=TIME_LIMIT)
        path = f"{OUT}/{name}.py"
        with open(path, 'w') as f: f.write(script)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        proc = subprocess.Popen(["python3", "-u", path], stdout=open(f"{OUT}/{name}.log", "w"), stderr=subprocess.STDOUT, env=env)
        running[gpu] = (name, proc, time.time())
        print(f"[START] GPU {gpu}: {name} (batch={batch_size})")

    if running:
        time.sleep(60)
        print(f"\n--- {len(completed)}/{len(configs)} done, {len(running)} running ---")

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)
