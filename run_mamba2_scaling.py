"""Mamba2 scaling test for comparison with E1."""

import torch
import time
import sys
import mmap
import numpy as np

sys.path.insert(0, '/home/erikg/elman')

from elman.models import create_mamba2_model

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

print("=" * 70)
print("MAMBA2 SCALING TEST (for E1 comparison)")
print("=" * 70)
print(f"GPU: {torch.cuda.get_device_name()}")

device = 'cuda'
dtype = torch.bfloat16
seq_len = 512
time_limit = 120

# Mamba2 configs (batch sizes matched to E1's wide+shallow configs for fairness)
configs = [
    ("50M", 48),
    ("100M", 32),
    ("200M", 16),
    ("400M", 12),
]

results = []

for scale_name, batch_size in configs:
    print(f"\n{'='*60}")
    print(f"Scale: {scale_name} (batch={batch_size})")
    print("=" * 60)
    
    try:
        torch.cuda.empty_cache()
        torch.manual_seed(42)
        
        model = create_mamba2_model(target_params=scale_name, vocab_size=256).to(device).to(dtype)
        params = model.get_num_params()
        print(f"Parameters: {params:,} ({params/1e6:.1f}M)")
        
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        # Warmup
        print("Warming up...")
        for _ in range(3):
            x = get_batch(batch_size, seq_len)
            loss = model(x, return_loss=True)
            loss.backward()
            opt.step()
            opt.zero_grad()
        torch.cuda.synchronize()
        
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"Memory: {mem:.1f} GB")
        
        # Training
        print(f"Training for {time_limit}s...")
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
            
            if step % 50 == 0:
                elapsed = time.time() - start
                avg_loss = sum(losses[-50:]) / 50
                tok_s = (step * batch_size * seq_len) / elapsed
                print(f"  Step {step} ({elapsed:.0f}s): loss={avg_loss:.4f}, {tok_s/1000:.1f}K tok/s")
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        final_loss = sum(losses[-50:]) / 50
        tok_per_s = (step * batch_size * seq_len) / elapsed
        
        results.append({
            'scale': scale_name,
            'params': params,
            'batch': batch_size,
            'steps': step,
            'loss': final_loss,
            'tok_per_s': tok_per_s,
            'memory': mem,
        })
        
        print(f"\nFinal: {step} steps, loss={final_loss:.4f}, {tok_per_s/1000:.1f}K tok/s")
        
        del model, opt
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error at {scale_name}: {e}")
        results.append({'scale': scale_name, 'error': str(e)})
        torch.cuda.empty_cache()

# Summary
print("\n" + "=" * 70)
print("MAMBA2 SCALING RESULTS")
print("=" * 70)

print(f"\n{'Scale':<8} {'Params':>10} {'Batch':>6} {'Steps':>6} {'Loss':>8} {'Tok/s':>10} {'Memory':>8}")
print("-" * 60)

for r in results:
    if 'error' in r:
        print(f"{r['scale']:<8} ERROR: {r['error'][:30]}")
    else:
        print(f"{r['scale']:<8} {r['params']/1e6:>9.1f}M {r['batch']:>6} {r['steps']:>6} {r['loss']:>8.4f} {r['tok_per_s']/1000:>9.1f}K {r['memory']:>7.1f}GB")

mm.close()

# Load E1 results and compare
print("\n" + "=" * 70)
print("E1 vs MAMBA2 COMPARISON")
print("=" * 70)

e1_results = {
    '50M': {'tok_per_s': 189800, 'loss': 1.6808},
    '100M': {'tok_per_s': 77500, 'loss': 1.8370},
    '200M': {'tok_per_s': 42300, 'loss': 2.1067},
    '400M': {'tok_per_s': 20100, 'loss': 2.2928},
}

print(f"\n{'Scale':<8} {'E1 Tok/s':>12} {'Mamba2 Tok/s':>14} {'E1 Speedup':>12} {'E1 Loss':>10} {'M2 Loss':>10}")
print("-" * 75)

for r in results:
    if 'error' not in r:
        scale = r['scale']
        e1 = e1_results.get(scale, {})
        if e1:
            speedup = e1['tok_per_s'] / r['tok_per_s']
            print(f"{scale:<8} {e1['tok_per_s']/1000:>11.1f}K {r['tok_per_s']/1000:>13.1f}K {speedup:>11.2f}x {e1['loss']:>10.4f} {r['loss']:>10.4f}")
