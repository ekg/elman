#!/usr/bin/env python3
"""Run E5 with the winning configuration."""
import sys
sys.path.insert(0, '.')
import torch
import time
import math
from schedulefree import AdamWScheduleFree
from elman.models import LadderLM
from elman.data import DocumentStreamDataset

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Winning E5 config: dim=1536, rank=270, depth=20, batch=256
dim = 1536
rank = 270
depth = 20
batch_size = 256
steps = 1000

print(f'Creating E5 model: dim={dim}, rank={rank}, depth={depth}')

model = LadderLM(
    vocab_size=256,
    dim=dim,
    depth=depth,
    level=5,  # E5: Pure Low-Rank Elman
    rank=rank,
).cuda().bfloat16()

num_params = model.get_num_params()
print(f'e5_50m_d{dim}_r{rank}: params={num_params:,}, depth={depth}, batch={batch_size}')

dataset = DocumentStreamDataset('data/fineweb_100mb.txt', chunk_size=513, seed=42)

optimizer = AdamWScheduleFree(model.parameters(), lr=3e-4, weight_decay=0.1)
model.train()
optimizer.train()

start = time.time()
tokens = 0

for step in range(1, steps+1):
    batch = torch.stack([dataset[0][0] for _ in range(batch_size)]).cuda()
    optimizer.zero_grad()
    loss = model(batch, return_loss=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    tokens += batch_size * 512

    if step % 100 == 0:
        elapsed = time.time() - start
        ppl = math.exp(min(loss.item(), 20))
        print(f'Step {step} | Loss {loss.item():.4f} | PPL {ppl:.1f} | {int(tokens/elapsed)} tok/s')

print(f'DONE: e5_50m_d{dim}_r{rank} | Final loss: {loss.item():.4f}')
