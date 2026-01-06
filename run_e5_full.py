#!/usr/bin/env python3
"""Run E5 with winning config on full pile.txt - byte-level without document delimiters."""
import sys
sys.path.insert(0, '.')
import torch
import time
import math
import numpy as np
import mmap
from schedulefree import AdamWScheduleFree
from elman.models import LadderLM

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Winning E5 config
dim = 1536
rank = 270
depth = 20
batch_size = 256
steps = 1000
seq_len = 512

print(f'Creating E5 model: dim={dim}, rank={rank}, depth={depth}')

model = LadderLM(
    vocab_size=256,
    dim=dim,
    depth=depth,
    level=5,
    rank=rank,
).cuda().bfloat16()

num_params = model.get_num_params()
print(f'e5_50m_d{dim}_r{rank}: params={num_params:,}, depth={depth}, batch={batch_size}')

# Memory-map pile.txt for efficient access
data_path = 'data/pile.txt'
print(f'Memory-mapping {data_path}...')
with open(data_path, 'rb') as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
data_len = len(mm)
print(f'Mapped {data_len:,} bytes ({data_len/1e9:.1f} GB)')

# Pre-allocate batch buffer
batch_buffer = np.zeros((batch_size, seq_len + 1), dtype=np.uint8)

optimizer = AdamWScheduleFree(model.parameters(), lr=3e-4, weight_decay=0.1)
model.train()
optimizer.train()

start = time.time()
tokens = 0
all_losses = []

for step in range(1, steps+1):
    # Sample random positions
    positions = np.random.randint(0, data_len - seq_len - 1, size=batch_size)

    # Read data efficiently
    for i, pos in enumerate(positions):
        batch_buffer[i] = np.frombuffer(mm[pos:pos+seq_len+1], dtype=np.uint8)

    batch = torch.from_numpy(batch_buffer.astype(np.int64)).cuda()

    optimizer.zero_grad()
    loss = model(batch, return_loss=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    tokens += batch_size * seq_len
    all_losses.append(loss.item())

    if step % 100 == 0:
        elapsed = time.time() - start
        avg_100 = np.mean(all_losses[-100:])
        ppl = math.exp(min(avg_100, 20))
        print(f'Step {step} | Loss {loss.item():.4f} | Avg100 {avg_100:.4f} | PPL {ppl:.1f} | {int(tokens/elapsed)} tok/s')

avg_last_100 = np.mean(all_losses[-100:])
print(f'DONE: e5_50m_d{dim}_r{rank} | Last100 avg: {avg_last_100:.4f} | Final: {loss.item():.4f}')
mm.close()
