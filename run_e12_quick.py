#!/usr/bin/env python3
"""
Quick E12 vs E1 comparison using LadderLM.
batch=16, 500 steps to see initial learning behavior.
"""
import sys
sys.path.insert(0, '/home/erikg/elman')
import torch
import numpy as np
import mmap
import time
from schedulefree import AdamWScheduleFree

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.benchmark = True

# Data loading
with open('/home/erikg/elman/data/pile.txt', 'rb') as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
data_len = len(mm)

def get_batch(bs, sl):
    pos = np.random.randint(0, data_len - sl - 1, size=bs)
    buf = np.zeros((bs, sl + 1), dtype=np.uint8)
    for i, p in enumerate(pos):
        buf[i] = np.frombuffer(mm[p:p+sl+1], dtype=np.uint8)
    return torch.from_numpy(buf.astype(np.int64)).cuda()

batch_size = 16
seq_len = 512
num_steps = 500

from elman.models import LadderLM

def run_test(name, level, dim, depth):
    print(f"\n{'='*60}")
    print(f"Testing {name} (level={level}, dim={dim}, depth={depth})")
    print('='*60)

    torch.manual_seed(42)
    np.random.seed(42)

    model = LadderLM(vocab_size=256, dim=dim, depth=depth, level=level).cuda().bfloat16()
    params = model.get_num_params()
    print(f"Parameters: {params:,}")

    opt = AdamWScheduleFree(model.parameters(), lr=3e-4, weight_decay=0.1)
    model.train()
    opt.train()

    # Warmup
    for _ in range(3):
        loss = model(get_batch(batch_size, seq_len), return_loss=True)
        loss.backward()
        opt.step()
        opt.zero_grad()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Get initial loss
    torch.manual_seed(42)
    np.random.seed(42)
    opt.eval()
    with torch.no_grad():
        init_loss = sum(model(get_batch(batch_size, seq_len), return_loss=True).item() for _ in range(5)) / 5
    opt.train()
    print(f"Initial: {init_loss:.4f}, Memory: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")

    # Training loop
    torch.manual_seed(42)
    np.random.seed(42)
    losses = []
    torch.cuda.synchronize()
    start = time.time()

    for step in range(1, num_steps + 1):
        loss = model(get_batch(batch_size, seq_len), return_loss=True)
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
        if step % 100 == 0:
            avg = sum(losses[-100:]) / min(100, len(losses))
            el = time.time() - start
            print(f"Step {step}: loss={avg:.4f}, {step*batch_size*seq_len/el/1000:.1f}K tok/s")

    torch.cuda.synchronize()
    elapsed = time.time() - start
    final_loss = sum(losses[-100:]) / 100
    tok_s = (num_steps * batch_size * seq_len) / elapsed
    memory = torch.cuda.max_memory_allocated() / 1e9

    print(f"FINAL: loss={final_loss:.4f}, tok/s={tok_s/1000:.1f}K, time={elapsed:.0f}s, memory={memory:.1f}GB")

    del model
    torch.cuda.empty_cache()

    return {
        'name': name,
        'params': params,
        'init_loss': init_loss,
        'final_loss': final_loss,
        'tok_s': tok_s/1000,
        'memory': memory,
        'time': elapsed
    }

# Run tests
results = []

# E1 at dim=1760, depth=26 (~400M params)
results.append(run_test('E1', level=1, dim=1760, depth=26))

# E12 at dim=1760, depth=26 (will have more params due to W_g)
results.append(run_test('E12', level=12, dim=1760, depth=26))

# Also test smaller config to see throughput difference more clearly
results.append(run_test('E1_small', level=1, dim=1280, depth=6))
results.append(run_test('E12_small', level=12, dim=1280, depth=6))

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"{'Model':<12} {'Params':>10} {'Init':>8} {'Final':>8} {'Tok/s':>10} {'Memory':>8} {'Time':>8}")
print("-"*70)
for r in results:
    print(f"{r['name']:<12} {r['params']/1e6:>9.1f}M {r['init_loss']:>8.4f} {r['final_loss']:>8.4f} {r['tok_s']:>9.1f}K {r['memory']:>7.1f}GB {r['time']:>7.0f}s")

mm.close()
