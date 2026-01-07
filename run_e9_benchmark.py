#!/usr/bin/env python3
"""
E9 (Hybrid Elman) 10-minute benchmark at 50M params.
Compare with E1 and Mamba2.
"""
import sys; sys.path.insert(0, '.')
import torch, numpy as np, mmap, time
from schedulefree import AdamWScheduleFree
from elman.models import LadderLM

torch.manual_seed(42); np.random.seed(42)
batch_size = 256
time_limit = 600  # 10 minutes

# Create E9 model - need to find dim/depth for ~50M params
# E9 has small core + large diagonal memory, so should be efficient

# First, probe param count
def get_params(dim, depth, core_ratio=0.125, expansion=2.0):
    m = LadderLM(vocab_size=256, dim=dim, depth=depth, level=9)
    return m.get_num_params()

# Find config ~50M params
# E9 is more efficient per-param due to diagonal memory
# Try dim=1024, expansion=2.0, vary depth
best = None
for dim in [768, 896, 1024, 1152, 1280]:
    for depth in range(4, 24, 2):
        p = get_params(dim, depth)
        if abs(p - 50_000_000) < (abs(best[2] - 50_000_000) if best else float('inf')):
            best = (dim, depth, p)

dim, depth, _ = best
print(f"Using dim={dim}, depth={depth}")

model = LadderLM(vocab_size=256, dim=dim, depth=depth, level=9).cuda().bfloat16()
print(f'E9_50m: params={model.get_num_params():,}')
print(f'Core dim: {model.layers[0].core_dim}, Mem dim: {model.layers[0].mem_dim}')

with open('data/pile.txt', 'rb') as f: mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
data_len = len(mm)

opt = AdamWScheduleFree(model.parameters(), lr=3e-4, weight_decay=0.1)
model.train(); opt.train()
buf = np.zeros((batch_size, 513), dtype=np.uint8)
losses = []; start = time.time(); step = 0

while time.time() - start < time_limit:
    step += 1
    pos = np.random.randint(0, data_len - 513, size=batch_size)
    for j, p in enumerate(pos): buf[j] = np.frombuffer(mm[p:p+513], dtype=np.uint8)
    batch = torch.from_numpy(buf.astype(np.int64)).cuda()
    opt.zero_grad(); loss = model(batch, return_loss=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    losses.append(loss.item())
    if step % 100 == 0:
        elapsed = time.time() - start
        tokens = step * batch_size * 512
        print(f'Step {step} | {elapsed:.0f}s | Loss {loss.item():.4f} | Avg100 {np.mean(losses[-100:]):.4f} | {int(tokens/elapsed)} tok/s')

elapsed = time.time() - start
tokens = step * batch_size * 512
print(f'DONE: E9_50m | Steps={step} | Time={elapsed:.0f}s | Last100={np.mean(losses[-100:]):.4f} | {int(tokens/elapsed)} tok/s')
mm.close()
