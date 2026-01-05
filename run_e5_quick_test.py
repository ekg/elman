#!/usr/bin/env python3
"""Quick E5 test with small data to verify model works."""
import sys
sys.path.insert(0, '.')
import torch
import time
import math
import numpy as np
from schedulefree import AdamWScheduleFree
from elman.models import LadderLM

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Winning E5 config
dim = 1536
rank = 270
depth = 20
batch_size = 64  # Reduced for test
steps = 100

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

# Load data as raw bytes - no tokenization needed for byte-level
data_path = 'data/pile_1mb.txt'
with open(data_path, 'rb') as f:
    raw_bytes = f.read()
data = np.frombuffer(raw_bytes, dtype=np.uint8)
data_len = len(data)
print(f'Loaded {data_len:,} bytes from {data_path}')

optimizer = AdamWScheduleFree(model.parameters(), lr=3e-4, weight_decay=0.1)
model.train()
optimizer.train()

seq_len = 512
start = time.time()
tokens = 0

for step in range(1, steps+1):
    # Sample random positions
    positions = np.random.randint(0, data_len - seq_len - 1, size=batch_size)
    batch = torch.tensor([data[p:p+seq_len+1] for p in positions], dtype=torch.long, device='cuda')

    optimizer.zero_grad()
    loss = model(batch, return_loss=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    tokens += batch_size * seq_len

    if step % 10 == 0:
        elapsed = time.time() - start
        ppl = math.exp(min(loss.item(), 20))
        print(f'Step {step} | Loss {loss.item():.4f} | PPL {ppl:.1f} | {int(tokens/elapsed)} tok/s')

print(f'DONE: e5_50m_d{dim}_r{rank} | Final loss: {loss.item():.4f}')
