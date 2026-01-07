#!/usr/bin/env python3
"""Full scaling test: E1 vs Mamba2 vs minGRU vs minLSTM at 50M, 100M, 200M, 400M."""
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

# Configs: model_code, batch_size for each scale
# E1: depth=6 for 50M, depth=26 for 100M+
# Use max batch that fits in 48GB
CONFIGS = {
    '50M': {
        'e1':      ('from elman.models import LadderLM\nname = "E1"\nmodel = LadderLM(vocab_size=256, dim=1280, depth=6, level=1).cuda().bfloat16()', 48),
        'mamba2':  ('from elman.models import create_mamba2_model\nname = "Mamba2"\nmodel = create_mamba2_model(target_params="50M", vocab_size=256).cuda().bfloat16()', 48),
        'mingru':  ('from elman.models import MinGRULM\nname = "minGRU"\nmodel = MinGRULM(vocab_size=256, dim=1984, depth=6).cuda().bfloat16()', 48),
        'minlstm': ('from elman.models import MinLSTMLM\nname = "minLSTM"\nmodel = MinLSTMLM(vocab_size=256, dim=1984, depth=6).cuda().bfloat16()', 48),
    },
    '100M': {
        'e1':      ('from elman.models import LadderLM\nname = "E1"\nmodel = LadderLM(vocab_size=256, dim=876, depth=26, level=1).cuda().bfloat16()', 48),
        'mamba2':  ('from elman.models import create_mamba2_model\nname = "Mamba2"\nmodel = create_mamba2_model(target_params="100M", vocab_size=256).cuda().bfloat16()', 48),
        'mingru':  ('from elman.models import MinGRULM\nname = "minGRU"\nmodel = MinGRULM(vocab_size=256, dim=2752, depth=6).cuda().bfloat16()', 48),
        'minlstm': ('from elman.models import MinLSTMLM\nname = "minLSTM"\nmodel = MinLSTMLM(vocab_size=256, dim=2752, depth=6).cuda().bfloat16()', 48),
    },
    '200M': {
        'e1':      ('from elman.models import LadderLM\nname = "E1"\nmodel = LadderLM(vocab_size=256, dim=1248, depth=26, level=1).cuda().bfloat16()', 48),
        'mamba2':  ('from elman.models import create_mamba2_model\nname = "Mamba2"\nmodel = create_mamba2_model(target_params="200M", vocab_size=256).cuda().bfloat16()', 32),
        'mingru':  ('from elman.models import MinGRULM\nname = "minGRU"\nmodel = MinGRULM(vocab_size=256, dim=2752, depth=12).cuda().bfloat16()', 32),
        'minlstm': ('from elman.models import MinLSTMLM\nname = "minLSTM"\nmodel = MinLSTMLM(vocab_size=256, dim=2752, depth=12).cuda().bfloat16()', 24),
    },
    '400M': {
        'e1':      ('from elman.models import LadderLM\nname = "E1"\nmodel = LadderLM(vocab_size=256, dim=1760, depth=26, level=1).cuda().bfloat16()', 80),
        'mamba2':  ('from elman.models import create_mamba2_model\nname = "Mamba2"\nmodel = create_mamba2_model(target_params="400M", vocab_size=256).cuda().bfloat16()', 64),
        'mingru':  ('from elman.models import MinGRULM\nname = "minGRU"\nmodel = MinGRULM(vocab_size=256, dim=2752, depth=24).cuda().bfloat16()', 24),
        'minlstm': ('from elman.models import MinLSTMLM\nname = "minLSTM"\nmodel = MinLSTMLM(vocab_size=256, dim=2304, depth=24).cuda().bfloat16()', 16),
    },
}

def run_scale(scale):
    """Run all 4 models for a given scale in parallel on GPUs 0-3."""
    print(f"\n{'='*60}")
    print(f"SCALE: {scale}")
    print(f"{'='*60}")

    procs = {}
    for gpu, (model_name, (code, batch)) in enumerate(CONFIGS[scale].items()):
        script = SCRIPT_TEMPLATE.replace('BATCH_SIZE', str(batch)).replace('TIME_LIMIT', str(TIME_LIMIT)).replace('MODEL_CODE', code)
        path = f"{OUT}/{model_name}_{scale.lower()}.py"
        with open(path, 'w') as f:
            f.write(script)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log = f"{OUT}/{model_name}_{scale.lower()}.log"
        procs[model_name] = subprocess.Popen(["python3", "-u", path], stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env)
        print(f"[GPU {gpu}] {model_name} batch={batch} started")

    for name, p in procs.items():
        p.wait()
        print(f"[DONE] {name}")

    # Parse results
    results = {}
    for model_name in CONFIGS[scale]:
        log = f"{OUT}/{model_name}_{scale.lower()}.log"
        try:
            with open(log) as f:
                content = f.read()
            for line in content.split('\n'):
                if 'FINAL:' in line:
                    parts = line.split(',')
                    loss = float([p for p in parts if 'loss=' in p][0].split('=')[1])
                    toks = float([p for p in parts if 'K tok/s' in p][0].strip().split()[-2].replace('K', ''))
                    results[model_name] = {'loss': loss, 'tok_s': toks}
                if 'params' in line and ':' in line:
                    params = int(line.split(':')[1].split()[0].replace(',', ''))
                    if model_name not in results:
                        results[model_name] = {}
                    results[model_name]['params'] = params
        except Exception as e:
            print(f"Error parsing {model_name}: {e}")
            results[model_name] = {'error': str(e)}

    return results

# Run all scales
all_results = {}
for scale in ['50M', '100M', '200M', '400M']:
    all_results[scale] = run_scale(scale)

# Print final summary
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

for scale in ['50M', '100M', '200M', '400M']:
    print(f"\n=== {scale} ===")
    print(f"{'Model':<10} {'Params':>10} {'Tok/s':>10} {'Loss':>8}")
    print("-" * 42)

    results = all_results[scale]
    # Sort by throughput
    sorted_models = sorted(results.items(), key=lambda x: -x[1].get('tok_s', 0))

    for model, r in sorted_models:
        if 'error' in r:
            print(f"{model:<10} ERROR: {r['error'][:20]}")
        else:
            params = r.get('params', 0) / 1e6
            toks = r.get('tok_s', 0)
            loss = r.get('loss', 0)
            print(f"{model:<10} {params:>9.1f}M {toks:>9.1f}K {loss:>8.4f}")
