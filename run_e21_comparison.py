#!/usr/bin/env python3
"""
E21 Comparison: 10-minute training time comparison.
Tests E1, Mamba2, minLSTM, E18-A, and E21 variants.

All models target ~50M params, same seed (42), same data (pile.txt).
"""
import subprocess, os, time

TIME_LIMIT = 600  # 10 minutes
BATCH_SIZE = 64   # Reasonable for all models
SEQ_LEN = 512

# Base training script template
TRAIN_SCRIPT = '''
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

batch_size, seq_len = {batch_size}, {seq_len}
time_limit = {time_limit}

{model_setup}

params = model.get_num_params()
print(f"{{name}}: {{params:,}} params", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

# Warmup
torch.manual_seed(42); np.random.seed(42)
for _ in range(3):
    loss = model(get_batch(batch_size, seq_len), return_loss=True)
    loss.backward(); opt.step(); opt.zero_grad()
torch.cuda.synchronize()

# Reset seed for training
torch.manual_seed(42); np.random.seed(42)

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
        avg_loss = sum(losses[-100:]) / min(100, len(losses))
        tok_s = (step * batch_size * seq_len) / el
        print(f"Step {{step}} ({{el:.0f}}s): loss={{avg_loss:.4f}}, {{tok_s/1000:.1f}}K tok/s", flush=True)

torch.cuda.synchronize()
elapsed = time.time() - start
final_loss = sum(losses[-100:]) / min(100, len(losses))
tok_s = (step * batch_size * seq_len) / elapsed
total_tokens = step * batch_size * seq_len

print(f"FINAL {{name}}: steps={{step}}, tokens={{total_tokens/1e6:.1f}}M, loss={{final_loss:.4f}}, tok/s={{tok_s/1000:.1f}}K", flush=True)
mm.close()
'''

# Model configurations (~50M params each)
EXPERIMENTS = [
    # E1 best config
    {
        "name": "E1_d1280x6",
        "gpu": 0,
        "setup": '''
from elman.models import LadderLM
name = "E1_d1280x6"
model = LadderLM(vocab_size=256, dim=1280, depth=6, level=1).cuda().bfloat16()
'''
    },
    # Mamba2 50M
    {
        "name": "Mamba2_50M",
        "gpu": 1,
        "setup": '''
from elman.models import create_mamba2_model
name = "Mamba2_50M"
model = create_mamba2_model(target_params='50m', vocab_size=256).cuda().bfloat16()
'''
    },
    # minLSTM
    {
        "name": "minLSTM",
        "gpu": 2,
        "setup": '''
from elman.models import MinLSTMLM
name = "minLSTM"
model = MinLSTMLM(vocab_size=256, dim=1280, depth=6).cuda().bfloat16()
'''
    },
    # E18-A (best E18)
    {
        "name": "E18A",
        "gpu": 3,
        "setup": '''
from elman.models import LadderLM
name = "E18A"
model = LadderLM(vocab_size=256, dim=1280, depth=6, level='18a').cuda().bfloat16()
'''
    },
    # E21 base (R=8, N=32)
    {
        "name": "E21",
        "gpu": 4,
        "setup": '''
from elman.models import LadderLM
name = "E21"
model = LadderLM(vocab_size=256, dim=896, depth=6, level=21).cuda().bfloat16()
'''
    },
    # E21-S (R=4, smaller)
    {
        "name": "E21S",
        "gpu": 5,
        "setup": '''
from elman.models import LadderLM
name = "E21S"
model = LadderLM(vocab_size=256, dim=1024, depth=6, level='21s').cuda().bfloat16()
'''
    },
    # E21-T (tanh instead of silu)
    {
        "name": "E21T",
        "gpu": 6,
        "setup": '''
from elman.models import LadderLM
name = "E21T"
model = LadderLM(vocab_size=256, dim=896, depth=6, level='21t').cuda().bfloat16()
'''
    },
    # E21-L (linear - ablation)
    {
        "name": "E21L",
        "gpu": 7,
        "setup": '''
from elman.models import LadderLM
name = "E21L"
model = LadderLM(vocab_size=256, dim=896, depth=6, level='21l').cuda().bfloat16()
'''
    },
]

OUT = "benchmark_results/e21_comparison"
os.makedirs(OUT, exist_ok=True)

print(f"E21 10-minute comparison: {TIME_LIMIT}s per model")
print(f"Batch size: {BATCH_SIZE}, Seq len: {SEQ_LEN}")
print("=" * 60)

procs = []
for e in EXPERIMENTS:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(e["gpu"])

    script = TRAIN_SCRIPT.format(
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        time_limit=TIME_LIMIT,
        model_setup=e["setup"]
    )

    log_path = f"{OUT}/{e['name']}.log"
    print(f"Launching {e['name']} on GPU {e['gpu']}...")
    p = subprocess.Popen(
        ["python", "-u", "-c", script],
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        env=env
    )
    procs.append((e["name"], p, log_path))
    time.sleep(0.5)

print(f"\nAll {len(EXPERIMENTS)} experiments launched! Running for {TIME_LIMIT}s each...")
print("Monitoring progress...")

# Wait for all to complete, periodically report
while any(p.poll() is None for _, p, _ in procs):
    time.sleep(60)
    done = sum(1 for _, p, _ in procs if p.poll() is not None)
    print(f"  {done}/{len(procs)} completed...")

print("\n" + "=" * 60)
print("FINAL RESULTS - 10 minutes training time")
print("=" * 60)

results = []
for name, _, log_path in procs:
    with open(log_path) as f:
        content = f.read()
        params = "unknown"
        # Find params line
        for line in content.split('\n'):
            if 'params' in line.lower() and name in line:
                params = line
        # Find final line
        for line in content.split('\n'):
            if 'FINAL' in line:
                results.append((name, line.strip(), params))
                print(line.strip())
                break
        else:
            print(f"{name}: FAILED (check {log_path})")
            lines = content.strip().split('\n')
            for l in lines[-5:]:
                print(f"  {l}")

print("\n" + "=" * 60)
print("SUMMARY TABLE")
print("=" * 60)
print(f"{'Model':<15} {'Loss':>8} {'Tok/s':>10} {'Params':>12}")
print("-" * 50)

for name, final, params in results:
    try:
        parts = final.split(',')
        loss = float([p for p in parts if 'loss=' in p][0].split('=')[1])
        tok_s = [p for p in parts if 'tok/s=' in p][0].split('=')[1]
        p_num = params.split(':')[1].strip().split()[0] if ':' in params else 'N/A'
        print(f"{name:<15} {loss:>8.4f} {tok_s:>10} {p_num:>12}")
    except:
        print(f"{name:<15} PARSE ERROR")

print("=" * 60)
