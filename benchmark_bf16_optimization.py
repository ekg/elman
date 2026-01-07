"""Benchmark BF16 optimized E1 kernels vs baseline."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import mmap
import numpy as np

sys.path.insert(0, '/home/erikg/elman')

torch.manual_seed(42)
torch.backends.cudnn.benchmark = True

# Data loading
data_path = '/home/erikg/elman/data/pile.txt'
with open(data_path, 'rb') as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
data_len = len(mm)

def get_batch(batch_size, seq_len):
    positions = np.random.randint(0, data_len - seq_len - 1, size=batch_size)
    buf = np.zeros((batch_size, seq_len + 1), dtype=np.uint8)
    for i, pos in enumerate(positions):
        buf[i] = np.frombuffer(mm[pos:pos + seq_len + 1], dtype=np.uint8)
    return torch.from_numpy(buf.astype(np.int64)).cuda()

# Import models
from elman.models import LadderLM
from elman.models.mamba_gated_elman import MAMBA_CUDA_AVAILABLE

print("=" * 70)
print("BF16 OPTIMIZATION BENCHMARK")
print("=" * 70)
print(f"CUDA kernels available: {MAMBA_CUDA_AVAILABLE}")

# Test configurations matching documented baselines
configs = [
    # (dim, depth, batch, seq, name)
    (1280, 6, 48, 512, "50M baseline"),
    (1024, 10, 48, 512, "50M deep"),
    (1024, 26, 48, 512, "~140M deep"),
]

device = 'cuda'
dtype = torch.bfloat16
time_limit = 60  # seconds per test

print(f"\nRunning {time_limit}s training for each configuration...")
print(f"Comparing against documented baselines from CLAUDE.md\n")

results = []

for dim, depth, batch_size, seq_len, name in configs:
    print(f"{'='*60}")
    print(f"Config: {name} (d{dim}×{depth}, batch={batch_size})")
    
    torch.manual_seed(42)
    model = LadderLM(vocab_size=256, dim=dim, depth=depth, level=1).to(device).to(dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    params = model.get_num_params()
    print(f"Parameters: {params:,}")
    
    # Warmup
    for _ in range(3):
        x = get_batch(batch_size, seq_len)
        loss = model(x, return_loss=True)
        loss.backward()
        opt.step()
        opt.zero_grad()
    torch.cuda.synchronize()
    
    # Timed training
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
        
        if step % 20 == 0:
            elapsed = time.time() - start
            avg_loss = sum(losses[-20:]) / 20
            tok_s = (step * batch_size * seq_len) / elapsed
            print(f"  Step {step} ({elapsed:.0f}s): loss={avg_loss:.4f}, {tok_s/1000:.1f}K tok/s")
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    final_loss = sum(losses[-20:]) / 20
    tok_per_s = (step * batch_size * seq_len) / elapsed
    
    results.append({
        'name': name,
        'config': f'd{dim}×{depth}',
        'params': params,
        'steps': step,
        'loss': final_loss,
        'tok_per_s': tok_per_s,
    })
    
    print(f"\nFinal: {step} steps, loss={final_loss:.4f}, {tok_per_s/1000:.1f}K tok/s")
    
    # Clear memory
    del model, opt
    torch.cuda.empty_cache()

print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

# Documented baselines from CLAUDE.md
baselines = {
    'd1280×6': 254,   # K tok/s
    'd1024×10': 214,  # K tok/s
}

print(f"\n{'Config':<15} {'Params':>10} {'Steps':>8} {'Loss':>8} {'Tok/s':>10} {'vs Baseline':>12}")
print("-" * 70)

for r in results:
    config = r['config']
    baseline = baselines.get(config, None)
    if baseline:
        pct_diff = (r['tok_per_s']/1000 - baseline) / baseline * 100
        baseline_str = f"{pct_diff:+.1f}%"
    else:
        baseline_str = "N/A"
    
    print(f"{r['name']:<15} {r['params']/1e6:>9.1f}M {r['steps']:>8} {r['loss']:>8.4f} {r['tok_per_s']/1000:>9.1f}K {baseline_str:>12}")

print("\n" + "=" * 70)
print("DOCUMENTED BASELINES (from CLAUDE.md)")
print("=" * 70)
print("""
| Model | Loss | Throughput |
|-------|------|------------|
| E1 d1280×6 | 1.43 | 254K tok/s |
| E1 d1024×10 | 1.45 | 214K tok/s |
""")

mm.close()
