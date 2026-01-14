#!/usr/bin/env python3
"""
E1 Expansion Sweep - CORRECT param calculation for ~400M.

Layer params scale as: dim² * (3*exp + 2*exp²)
- exp=1.0: 5*dim²
- exp=1.5: 9*dim²
- exp=2.0: 14*dim²
- exp=2.5: 20*dim²

For ~400M params at depth=26:
- exp=1.0: dim≈1760 → 403M
- exp=1.5: dim≈1310 → ~400M
- exp=2.0: dim≈1050 → ~400M
- exp=2.5: dim≈880 → ~400M
"""
import subprocess
import os

OUT = "e1_expansion_400m"
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

# E1 configs - CORRECTED dims for ~400M params
# Layer params ≈ dim² * (3*exp + 2*exp²) * depth
CONFIGS = {
    # Baseline: expansion=1.0, dim=1760 → ~403M
    'e1_exp1.0': 'from elman.models import LadderLM\nname = "E1_exp1.0_d1760"\nmodel = LadderLM(vocab_size=256, dim=1760, depth=26, level=1, expansion=1.0).cuda().bfloat16()',

    # expansion=1.5, dim=1312 → ~400M (d_inner=1968)
    'e1_exp1.5': 'from elman.models import LadderLM\nname = "E1_exp1.5_d1312"\nmodel = LadderLM(vocab_size=256, dim=1312, depth=26, level=1, expansion=1.5).cuda().bfloat16()',

    # expansion=2.0, dim=1056 → ~400M (d_inner=2112)
    'e1_exp2.0': 'from elman.models import LadderLM\nname = "E1_exp2.0_d1056"\nmodel = LadderLM(vocab_size=256, dim=1056, depth=26, level=1, expansion=2.0).cuda().bfloat16()',

    # expansion=2.5, dim=880 → ~400M (d_inner=2200)
    'e1_exp2.5': 'from elman.models import LadderLM\nname = "E1_exp2.5_d880"\nmodel = LadderLM(vocab_size=256, dim=880, depth=26, level=1, expansion=2.5).cuda().bfloat16()',

    # Mamba2 for reference
    'mamba2': 'from elman.models import create_mamba2_model\nname = "Mamba2"\nmodel = create_mamba2_model(target_params="400M", vocab_size=256).cuda().bfloat16()',
}

print("=" * 70)
print(f"E1 EXPANSION SWEEP - EQUAL PARAMS ~400M (batch={BATCH_SIZE}, steps={NUM_STEPS})")
print("=" * 70)
print("All models have ~400M params. Testing if wider d_inner helps at equal param budget.")
print()

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
print("RESULTS (E1 Expansion at ~400M Params)")
print("=" * 70)

results = []
for name in CONFIGS:
    log = f"{OUT}/{name}.log"
    with open(log) as f:
        content = f.read()
    params = loss = toks = time_s = memory = 0
    for line in content.split('\n'):
        if 'params' in line and ':' in line:
            try:
                params = int(line.split(':')[1].split()[0].replace(',', ''))
            except:
                pass
        if 'Memory:' in line:
            try:
                memory = float(line.split('Memory:')[1].split()[0])
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
    results.append((name, params, loss, toks, time_s, memory))

results.sort(key=lambda x: x[2])
print(f"\n{'Model':<20} {'Params':>10} {'Loss':>8} {'Tok/s':>10} {'Memory':>8} {'Time':>8}")
print("-" * 70)
for name, params, loss, toks, time_s, memory in results:
    print(f"{name:<20} {params/1e6:>9.1f}M {loss:>8.4f} {toks:>9.1f}K {memory:>7.1f}GB {time_s:>7.0f}s")

print("\nKey question: At equal ~400M params, does expansion help?")
