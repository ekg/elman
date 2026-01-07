"""Profile E1 to understand compute vs memory bottleneck."""

import torch
import torch.nn as nn
import time
import sys
sys.path.insert(0, '/home/erikg/elman')

from elman.models import LadderLM

# Setup
torch.backends.cudnn.benchmark = True
device = 'cuda'
dtype = torch.bfloat16

# Model similar to our 400M config
dim, depth = 1024, 26
batch_size = 48
seq_len = 512

model = LadderLM(vocab_size=256, dim=dim, depth=depth, level=1).to(device).to(dtype)
print(f"Model: E1 d{dim}×{depth}, params={model.get_num_params():,}")

# Warmup
x = torch.randint(0, 256, (batch_size, seq_len), device=device)
for _ in range(3):
    loss = model(x, return_loss=True)
    loss.backward()
torch.cuda.synchronize()

# Profile with torch profiler
print("\nProfiling forward + backward...")
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=False,
) as prof:
    for _ in range(5):
        loss = model(x, return_loss=True)
        loss.backward()
        torch.cuda.synchronize()

# Print top CUDA operations
print("\nTop 20 CUDA operations by time:")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# Also check memory bandwidth utilization
print("\n\nMemory stats:")
print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
print(f"Current memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
