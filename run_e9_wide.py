#!/usr/bin/env python3
"""
E9 WIDE: Big core + massive memory + shallow
Maximize hidden state, minimize depth
"""
import sys; sys.path.insert(0, '.')
import torch, numpy as np, mmap, time
from schedulefree import AdamWScheduleFree
from elman.models import LadderLM

torch.manual_seed(42); np.random.seed(42)
batch_size = 256
time_limit = 600

# Try several configs - big core, huge memory, shallow
configs = [
    # (dim, expansion, core_ratio, target_core, target_mem)
    (1024, 9.0, 1024/9216, 1024, 8192),    # core=1024, mem=8192
    (1024, 5.0, 512/5120, 512, 4608),      # core=512, mem=4608
    (768, 8.0, 768/6144, 768, 5376),       # core=768, mem=5376
]

print("Exploring configs...")
for dim, expansion, core_ratio, target_core, target_mem in configs:
    d_inner = int(dim * expansion)
    actual_core = max(64, int(d_inner * core_ratio))
    actual_mem = d_inner - actual_core

    for depth in [2, 3, 4]:
        try:
            m = LadderLM(vocab_size=256, dim=dim, depth=depth, level=9,
                         expansion=expansion, core_ratio=core_ratio)
            params = m.get_num_params()
            l = m.layers[0]
            if 40_000_000 < params < 60_000_000:
                print(f"dim={dim} exp={expansion} d={depth}: core={l.core_dim} mem={l.mem_dim} total={l.core_dim+l.mem_dim} params={params:,}")
            del m
        except Exception as e:
            pass

# Pick best config: core=512, mem=4608, depth=3 should be ~50M
dim = 1024
expansion = 5.0
core_ratio = 512 / (dim * expansion)  # 512/5120 = 0.1
depth = 3

model = LadderLM(vocab_size=256, dim=dim, depth=depth, level=9,
                 expansion=expansion, core_ratio=core_ratio).cuda().bfloat16()

layer = model.layers[0]
print(f"\n=== RUNNING ===")
print(f"Config: dim={dim}, depth={depth}, expansion={expansion}")
print(f"Core: {layer.core_dim}, Memory: {layer.mem_dim}, Total hidden: {layer.core_dim + layer.mem_dim}")
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
print(f'DONE: E9_wide | Steps={step} | Time={elapsed:.0f}s | Last100={np.mean(losses[-100:]):.4f} | {int(tokens/elapsed)} tok/s')
mm.close()
