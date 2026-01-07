"""Scaling test for BF16-optimized E1 kernels."""

import torch
import torch.nn.functional as F
import time
import sys
import mmap
import numpy as np

sys.path.insert(0, '/home/erikg/elman')

from elman.models import LadderLM, create_mamba2_model

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
print("E1 SCALING TEST (BF16 Optimized)")
print("=" * 70)
print(f"GPU: {torch.cuda.get_device_name()}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

device = 'cuda'
dtype = torch.bfloat16
seq_len = 512
time_limit = 120  # 2 minutes per scale

# Scaling configurations: (scale_name, dim, depth, batch_size)
# Two strategies: depth=6 (shallow+wide) or depth=26 (optimal for learning)
# Batch sizes empirically tested for 48GB GPU
configs = [
    ("50M", 1280, 6, 48),      # 49.5M params - shallow+wide (best for small)
    ("100M", 876, 26, 48),     # 100.1M params - deep
    ("200M", 1248, 26, 48),    # 202.9M params - deep
    ("400M", 1760, 26, 80),    # 403.3M params - deep (1.46x faster than Mamba2)
]

results = []

for scale_name, dim, depth, batch_size in configs:
    print(f"\n{'='*60}")
    print(f"Scale: {scale_name} (d{dim}×{depth}, batch={batch_size})")
    print("=" * 60)
    
    try:
        torch.cuda.empty_cache()
        torch.manual_seed(42)
        
        model = LadderLM(vocab_size=256, dim=dim, depth=depth, level=1).to(device).to(dtype)
        params = model.get_num_params()
        print(f"Parameters: {params:,} ({params/1e6:.1f}M)")
        
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        # Check memory
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated() / 1e9
        
        # Warmup
        print("Warming up...")
        for _ in range(3):
            x = get_batch(batch_size, seq_len)
            loss = model(x, return_loss=True)
            loss.backward()
            opt.step()
            opt.zero_grad()
        torch.cuda.synchronize()
        
        mem_after = torch.cuda.max_memory_allocated() / 1e9
        print(f"Memory: {mem_after:.1f} GB")
        
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
            'dim': dim,
            'depth': depth,
            'batch': batch_size,
            'steps': step,
            'loss': final_loss,
            'tok_per_s': tok_per_s,
            'memory': mem_after,
        })
        
        print(f"\nFinal: {step} steps, loss={final_loss:.4f}, {tok_per_s/1000:.1f}K tok/s")
        
        del model, opt
        torch.cuda.empty_cache()
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"OOM at {scale_name}: {e}")
        results.append({
            'scale': scale_name,
            'params': 0,
            'error': 'OOM',
        })
        torch.cuda.empty_cache()

# Summary
print("\n" + "=" * 70)
print("SCALING TEST RESULTS")
print("=" * 70)

print(f"\n{'Scale':<8} {'Params':>10} {'Config':>12} {'Batch':>6} {'Steps':>6} {'Loss':>8} {'Tok/s':>10} {'Memory':>8}")
print("-" * 75)

for r in results:
    if 'error' in r:
        print(f"{r['scale']:<8} {'OOM':>10}")
    else:
        config = f"d{r['dim']}×{r['depth']}"
        print(f"{r['scale']:<8} {r['params']/1e6:>9.1f}M {config:>12} {r['batch']:>6} {r['steps']:>6} {r['loss']:>8.4f} {r['tok_per_s']/1000:>9.1f}K {r['memory']:>7.1f}GB")

# Scaling analysis
print("\n" + "=" * 70)
print("SCALING ANALYSIS")
print("=" * 70)

valid_results = [r for r in results if 'error' not in r]
if len(valid_results) >= 2:
    # Calculate throughput scaling
    base = valid_results[0]
    print(f"\nBase: {base['scale']} at {base['tok_per_s']/1000:.1f}K tok/s")
    print("\nRelative throughput as model scales:")
    for r in valid_results[1:]:
        ratio = r['tok_per_s'] / base['tok_per_s']
        param_ratio = r['params'] / base['params']
        print(f"  {r['scale']}: {r['tok_per_s']/1000:.1f}K tok/s ({ratio:.2f}x base, {param_ratio:.1f}x params)")

mm.close()
print("\nDone!")
