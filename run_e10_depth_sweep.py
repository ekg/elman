#!/usr/bin/env python3
"""
E10 (MultiScale EMA) depth sweep with k=4 banks.
batch=16, 1500 steps for all, same as the main 400M comparison.
Runs in parallel across GPUs.
"""
import subprocess
import os

OUT = "e10_depth_sweep"
os.makedirs(OUT, exist_ok=True)
BATCH_SIZE = 16
NUM_STEPS = 1500

SCRIPT_TEMPLATE = '''
import sys; sys.path.insert(0, '/home/erikg/elman')
import torch, numpy as np, mmap, time
from schedulefree import AdamWScheduleFree
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

batch_size, seq_len, num_steps = BATCH_SIZE, 512, NUM_STEPS
MODEL_CODE
params = model.get_num_params()
print(f"{name}: {params:,} params", flush=True)

opt = AdamWScheduleFree(model.parameters(), lr=3e-4, weight_decay=0.1)
model.train()
opt.train()

for _ in range(3):
    loss = model(get_batch(batch_size, seq_len), return_loss=True)
    loss.backward(); opt.step(); opt.zero_grad()
torch.cuda.synchronize()

torch.manual_seed(42); np.random.seed(42)
opt.eval()
with torch.no_grad():
    init_loss = sum(model(get_batch(batch_size, seq_len), return_loss=True).item() for _ in range(5)) / 5
opt.train()
print(f"Initial: {init_loss:.4f}, Memory: {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)

torch.manual_seed(42); np.random.seed(42)
losses = []
torch.cuda.synchronize()
start = time.time()

for step in range(1, num_steps + 1):
    loss = model(get_batch(batch_size, seq_len), return_loss=True)
    loss.backward()
    opt.step(); opt.zero_grad()
    losses.append(loss.item())
    if step % 300 == 0:
        avg = sum(losses[-100:]) / 100
        el = time.time() - start
        print(f"Step {step}: loss={avg:.4f}, {step*batch_size*seq_len/el/1000:.1f}K tok/s", flush=True)

torch.cuda.synchronize()
elapsed = time.time() - start
final_loss = sum(losses[-100:]) / 100
tok_s = (num_steps * batch_size * seq_len) / elapsed
print(f"FINAL: loss={final_loss:.4f}, tok/s={tok_s/1000:.1f}K, time={elapsed:.0f}s", flush=True)
mm.close()
'''

# E10 configs targeting ~400M params at different depths
# E10 has MultiScaleElman with n_banks parameter
# Params scale roughly as: embed(256*dim) + depth*(layer_params) + output(dim*256)
# Layer params for E10 include dim*dim (W_x, W_h) + bank params

CONFIGS = {
    # dim chosen to target ~400M params at each depth with n_banks=4
    'e10_d6':   'from elman.models import LadderLM\nname = "E10_k4_d6"\nmodel = LadderLM(vocab_size=256, dim=3584, depth=6, level=10, n_banks=4).cuda().bfloat16()',
    'e10_d12':  'from elman.models import LadderLM\nname = "E10_k4_d12"\nmodel = LadderLM(vocab_size=256, dim=2560, depth=12, level=10, n_banks=4).cuda().bfloat16()',
    'e10_d16':  'from elman.models import LadderLM\nname = "E10_k4_d16"\nmodel = LadderLM(vocab_size=256, dim=2208, depth=16, level=10, n_banks=4).cuda().bfloat16()',
    'e10_d20':  'from elman.models import LadderLM\nname = "E10_k4_d20"\nmodel = LadderLM(vocab_size=256, dim=1984, depth=20, level=10, n_banks=4).cuda().bfloat16()',
    'e10_d26':  'from elman.models import LadderLM\nname = "E10_k4_d26"\nmodel = LadderLM(vocab_size=256, dim=1760, depth=26, level=10, n_banks=4).cuda().bfloat16()',
}

print("=" * 70)
print(f"E10 DEPTH SWEEP (k=4 banks, batch={BATCH_SIZE}, steps={NUM_STEPS})")
print("=" * 70)

procs = {}
for gpu, (name, code) in enumerate(CONFIGS.items()):
    script = SCRIPT_TEMPLATE.replace('BATCH_SIZE', str(BATCH_SIZE)).replace('NUM_STEPS', str(NUM_STEPS)).replace('MODEL_CODE', code)
    path = f"{OUT}/{name}.py"
    with open(path, 'w') as f:
        f.write(script)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log = f"{OUT}/{name}.log"
    procs[name] = subprocess.Popen(["python3", "-u", path], stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env)
    print(f"[GPU {gpu}] {name} started")

print("\nWaiting...")
for name, p in procs.items():
    p.wait()
    print(f"[DONE] {name}")

print("\n" + "=" * 70)
print("RESULTS (E10 k=4 Depth Sweep)")
print("=" * 70)

results = []
for name in CONFIGS:
    log = f"{OUT}/{name}.log"
    with open(log) as f:
        content = f.read()
    params = loss = toks = time_s = 0
    for line in content.split('\n'):
        if 'params' in line and ':' in line:
            try:
                params = int(line.split(':')[1].split()[0].replace(',', ''))
            except:
                pass
        if 'FINAL:' in line:
            parts = line.split(',')
            for p in parts:
                if 'loss=' in p:
                    loss = float(p.split('=')[1])
                if 'tok/s=' in p:
                    toks = float(p.split('=')[1].replace('K', ''))
                if 'time=' in p:
                    time_s = float(p.split('=')[1].replace('s', ''))
    results.append((name, params, loss, toks, time_s))

results.sort(key=lambda x: x[2])
print(f"\n{'Model':<15} {'Params':>10} {'Loss':>8} {'Tok/s':>10} {'Time':>8}")
print("-" * 55)
for name, params, loss, toks, time_s in results:
    print(f"{name:<15} {params/1e6:>9.1f}M {loss:>8.4f} {toks:>9.1f}K {time_s:>7.0f}s")
