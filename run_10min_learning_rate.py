#!/usr/bin/env python3
"""
10-minute learning rate comparison: E1 vs Mamba2 vs minGRU vs minLSTM at 400M.
Focus on LEARNING RATE (nats/s) - how fast loss decreases per second.
"""
import subprocess
import os
import time

OUT = "learning_rate_results"
os.makedirs(OUT, exist_ok=True)
TIME_LIMIT = 600  # 10 minutes

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
params = model.get_num_params()
print(f"{name}: {params:,} params", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

# Warmup
for _ in range(5):
    loss = model(get_batch(batch_size, seq_len), return_loss=True)
    loss.backward(); opt.step(); opt.zero_grad()
torch.cuda.synchronize()
print(f"Memory: {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)

# Get initial loss
init_losses = []
for _ in range(10):
    with torch.no_grad():
        loss = model(get_batch(batch_size, seq_len), return_loss=True)
        init_losses.append(loss.item())
initial_loss = sum(init_losses) / len(init_losses)
print(f"Initial loss: {initial_loss:.4f}", flush=True)

# Training
losses = []
step = 0
torch.cuda.synchronize()
start = time.time()

while time.time() - start < time_limit:
    loss = model(get_batch(batch_size, seq_len), return_loss=True)
    loss.backward(); opt.step(); opt.zero_grad()
    losses.append(loss.item()); step += 1

    if step % 100 == 0:
        el = time.time() - start
        avg_loss = sum(losses[-100:]) / 100
        tok_s = (step * batch_size * seq_len) / el
        # Learning rate = loss decrease per second
        nats_per_s = (initial_loss - avg_loss) / el
        print(f"Step {step} ({el:.0f}s): loss={avg_loss:.4f}, {tok_s/1000:.1f}K tok/s, nats/s={nats_per_s:.6f}", flush=True)

torch.cuda.synchronize()
elapsed = time.time() - start
final_loss = sum(losses[-100:]) / min(100, len(losses))
tok_s = (step * batch_size * seq_len) / elapsed
nats_per_s = (initial_loss - final_loss) / elapsed

print(f"FINAL: steps={step}, loss={final_loss:.4f}, tok/s={tok_s/1000:.1f}K, nats/s={nats_per_s:.6f}", flush=True)
print(f"LEARNING: initial={initial_loss:.4f}, final={final_loss:.4f}, delta={initial_loss-final_loss:.4f}", flush=True)
mm.close()
'''

# 400M configs with max batch sizes
CONFIGS = {
    'e1':      ('from elman.models import LadderLM\nname = "E1"\nmodel = LadderLM(vocab_size=256, dim=1760, depth=26, level=1).cuda().bfloat16()', 80),
    'mamba2':  ('from elman.models import create_mamba2_model\nname = "Mamba2"\nmodel = create_mamba2_model(target_params="400M", vocab_size=256).cuda().bfloat16()', 128),
    'mingru':  ('from elman.models import MinGRULM\nname = "minGRU"\nmodel = MinGRULM(vocab_size=256, dim=2752, depth=24).cuda().bfloat16()', 24),
    'minlstm': ('from elman.models import MinLSTMLM\nname = "minLSTM"\nmodel = MinLSTMLM(vocab_size=256, dim=2304, depth=24).cuda().bfloat16()', 16),
}

print("=" * 70)
print("10-MINUTE LEARNING RATE COMPARISON (400M scale)")
print("=" * 70)
print(f"Time limit: {TIME_LIMIT}s per model")
print("Metric: nats/s = (initial_loss - final_loss) / time")
print()

procs = {}
for gpu, (model_name, (code, batch)) in enumerate(CONFIGS.items()):
    script = SCRIPT_TEMPLATE.replace('BATCH_SIZE', str(batch)).replace('TIME_LIMIT', str(TIME_LIMIT)).replace('MODEL_CODE', code)
    path = f"{OUT}/{model_name}_400m.py"
    with open(path, 'w') as f:
        f.write(script)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log = f"{OUT}/{model_name}_400m.log"
    procs[model_name] = subprocess.Popen(["python3", "-u", path], stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env)
    print(f"[GPU {gpu}] {model_name} batch={batch} started")

print("\nWaiting for completion (~10 min)...")
for name, p in procs.items():
    p.wait()
    print(f"[DONE] {name}")

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

for model_name in CONFIGS:
    log = f"{OUT}/{model_name}_400m.log"
    print(f"\n=== {model_name.upper()} ===")
    with open(log) as f:
        for line in f:
            if any(k in line for k in ['params', 'FINAL:', 'LEARNING:', 'Memory:']):
                print(line.strip())
