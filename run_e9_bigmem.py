#!/usr/bin/env python3
"""
E9 with same core (160) but 4x memory (4480).
Total hidden = 4640, compute still dominated by core.
"""
import sys; sys.path.insert(0, '.')
import torch, numpy as np, mmap, time
from schedulefree import AdamWScheduleFree
from elman.models import LadderLM

torch.manual_seed(42); np.random.seed(42)
batch_size = 256
time_limit = 600  # 10 minutes

# Target: core_dim=160, mem_dim=4480 (4x previous 1120)
# d_inner = 160 + 4480 = 4640
# With dim=1280: expansion = 4640/1280 = 3.625
# core_ratio = 160/4640 = 0.0345

dim = 1280
expansion = 3.625
core_ratio = 160 / (dim * expansion)  # = 0.0345

# Find depth for ~50M params
print(f"Target: core=160, mem=4480, total hidden=4640")
print(f"Config: dim={dim}, expansion={expansion}, core_ratio={core_ratio:.4f}")

for depth in range(2, 20):
    model = LadderLM(vocab_size=256, dim=dim, depth=depth, level=9,
                     expansion=expansion, core_ratio=core_ratio).cuda().bfloat16()
    params = model.get_num_params()
    layer = model.layers[0]
    if depth == 2 or abs(params - 50_000_000) < 10_000_000:
        print(f"depth={depth}: core={layer.core_dim}, mem={layer.mem_dim}, params={params:,}")
    if params > 50_000_000:
        break

# Use the depth that gets us closest to 50M
target = 50_000_000
best_depth = 2
best_diff = float('inf')
for d in range(2, 20):
    m = LadderLM(vocab_size=256, dim=dim, depth=d, level=9,
                 expansion=expansion, core_ratio=core_ratio)
    diff = abs(m.get_num_params() - target)
    if diff < best_diff:
        best_diff = diff
        best_depth = d
    del m

model = LadderLM(vocab_size=256, dim=dim, depth=best_depth, level=9,
                 expansion=expansion, core_ratio=core_ratio).cuda().bfloat16()

layer = model.layers[0]
print(f"\nFinal config:")
print(f"dim={dim}, depth={model.depth}, expansion={expansion}, core_ratio={core_ratio:.4f}")
print(f"Core dim: {layer.core_dim}, Mem dim: {layer.mem_dim}")
print(f"Total hidden: {layer.core_dim + layer.mem_dim}")
print(f"Params: {model.get_num_params():,}")

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
print(f'DONE: E9_bigmem | Steps={step} | Time={elapsed:.0f}s | Last100={np.mean(losses[-100:]):.4f} | {int(tokens/elapsed)} tok/s')
mm.close()
