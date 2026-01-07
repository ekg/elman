#!/usr/bin/env python3
"""
DEEP MODELS comparison: depths 32, 40, 48 for E1, Mamba2, minGRU, minLSTM.
Same batch=16, same steps=1500.
"""
import subprocess
import os

OUT = "deep_models_comparison"
os.makedirs(OUT, exist_ok=True)
BATCH_SIZE = 16
NUM_STEPS = 1500

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

batch_size, seq_len, num_steps = BATCH_SIZE, 512, NUM_STEPS
MODEL_CODE
params = model.get_num_params()
print(f"{name}: {params:,} params", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

for _ in range(3):
    loss = model(get_batch(batch_size, seq_len), return_loss=True)
    loss.backward(); opt.step(); opt.zero_grad()
torch.cuda.synchronize()

torch.manual_seed(42); np.random.seed(42)
with torch.no_grad():
    init_loss = sum(model(get_batch(batch_size, seq_len), return_loss=True).item() for _ in range(5)) / 5
print(f"Initial: {init_loss:.4f}, Memory: {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)

torch.manual_seed(42); np.random.seed(42)
losses = []
torch.cuda.synchronize()
start = time.time()

for step in range(1, num_steps + 1):
    loss = model(get_batch(batch_size, seq_len), return_loss=True)
    loss.backward(); opt.step(); opt.zero_grad()
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

# ~400M param configs at different depths
CONFIGS = {
    # E1 configs (calculated for ~400M)
    'e1_d32': 'from elman.models import LadderLM\nname = "E1_d32"\nmodel = LadderLM(vocab_size=256, dim=1568, depth=32, level=1).cuda().bfloat16()',
    'e1_d40': 'from elman.models import LadderLM\nname = "E1_d40"\nmodel = LadderLM(vocab_size=256, dim=1408, depth=40, level=1).cuda().bfloat16()',
    'e1_d48': 'from elman.models import LadderLM\nname = "E1_d48"\nmodel = LadderLM(vocab_size=256, dim=1280, depth=48, level=1).cuda().bfloat16()',
    # Mamba2 deep
    'mamba2_d32': 'from mamba_ssm import Mamba2\nimport torch.nn as nn\nclass M2(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.embed = nn.Embedding(256, 1440)\n        self.layers = nn.ModuleList([Mamba2(1440, d_state=128, headdim=64) for _ in range(32)])\n        self.norm = nn.RMSNorm(1440)\n        self.head = nn.Linear(1440, 256, bias=False)\n        self.head.weight = self.embed.weight\n    def forward(self, x, return_loss=False):\n        if return_loss: x, t = x[:, :-1], x[:, 1:]\n        h = self.embed(x)\n        for l in self.layers: h = h + l(h)\n        h = self.norm(h)\n        logits = self.head(h)\n        if return_loss: return nn.functional.cross_entropy(logits.view(-1,256), t.reshape(-1))\n        return logits\n    def get_num_params(self): return sum(p.numel() for p in self.parameters())\nname = "Mamba2_d32"\nmodel = M2().cuda().bfloat16()',
    'mamba2_d40': 'from mamba_ssm import Mamba2\nimport torch.nn as nn\nclass M2(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.embed = nn.Embedding(256, 1280)\n        self.layers = nn.ModuleList([Mamba2(1280, d_state=128, headdim=64) for _ in range(40)])\n        self.norm = nn.RMSNorm(1280)\n        self.head = nn.Linear(1280, 256, bias=False)\n        self.head.weight = self.embed.weight\n    def forward(self, x, return_loss=False):\n        if return_loss: x, t = x[:, :-1], x[:, 1:]\n        h = self.embed(x)\n        for l in self.layers: h = h + l(h)\n        h = self.norm(h)\n        logits = self.head(h)\n        if return_loss: return nn.functional.cross_entropy(logits.view(-1,256), t.reshape(-1))\n        return logits\n    def get_num_params(self): return sum(p.numel() for p in self.parameters())\nname = "Mamba2_d40"\nmodel = M2().cuda().bfloat16()',
    # minGRU deep
    'mingru_d32': 'from elman.models import MinGRULM\nname = "minGRU_d32"\nmodel = MinGRULM(vocab_size=256, dim=2432, depth=32).cuda().bfloat16()',
    'mingru_d40': 'from elman.models import MinGRULM\nname = "minGRU_d40"\nmodel = MinGRULM(vocab_size=256, dim=2176, depth=40).cuda().bfloat16()',
    # minLSTM deep
    'minlstm_d32': 'from elman.models import MinLSTMLM\nname = "minLSTM_d32"\nmodel = MinLSTMLM(vocab_size=256, dim=2048, depth=32).cuda().bfloat16()',
    'minlstm_d40': 'from elman.models import MinLSTMLM\nname = "minLSTM_d40"\nmodel = MinLSTMLM(vocab_size=256, dim=1856, depth=40).cuda().bfloat16()',
}

print("=" * 70)
print(f"DEEP MODELS (batch={BATCH_SIZE}, steps={NUM_STEPS})")
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
print("RESULTS")
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
