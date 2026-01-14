
import sys; sys.path.insert(0, '/home/erikg/elman')
import os
os.environ['LD_LIBRARY_PATH'] = f"/home/erikg/.local/lib/python3.12/site-packages/torch/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mmap
import time
from schedulefree import AdamWScheduleFree

# CRITICAL: Same seed for ALL models
DATA_SEED = 42
torch.manual_seed(DATA_SEED)
np.random.seed(DATA_SEED)

model_type = "mamba2"
dim = 1024
depth = 8
batch_size = 64
seq_len = 512
time_limit = 600
n_slots = 8

# Data setup
with open('/home/erikg/elman/data/pile.txt', 'rb') as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
data_len = len(mm)

# Pre-generate ALL batch positions for deterministic data loading
# This ensures every model sees EXACTLY the same data in the same order
max_steps = int(time_limit * 2)  # Upper bound on steps
np.random.seed(DATA_SEED)  # Reset seed before generating positions
all_positions = np.random.randint(0, data_len - seq_len - 1, size=(max_steps, batch_size))

buf = np.zeros((batch_size, seq_len + 1), dtype=np.uint8)

def get_batch_deterministic(step):
    """Get batch at specific step - deterministic across processes."""
    pos = all_positions[step % len(all_positions)]
    for j, p in enumerate(pos):
        buf[j] = np.frombuffer(mm[p:p+seq_len+1], dtype=np.uint8)
    return torch.from_numpy(buf.astype(np.int64)).cuda()

# Model-specific imports and creation
if model_type == "mamba2":
    from elman.models.mamba2_baseline import Mamba2LM, MAMBA2_AVAILABLE
    if not MAMBA2_AVAILABLE:
        print("ERROR: mamba_ssm not installed")
        sys.exit(1)
    model = Mamba2LM(vocab_size=256, dim=dim, depth=depth).cuda().bfloat16()

elif model_type == "e1":
    from elman.models.mamba_gated_elman import MambaGatedElman

    class E1LM(nn.Module):
        def __init__(self, vocab_size, dim, depth):
            super().__init__()
            self.vocab_size = vocab_size
            self.embed = nn.Embedding(vocab_size, dim)
            self.layers = nn.ModuleList([MambaGatedElman(dim, expansion=1.0) for _ in range(depth)])
            self.norm = nn.RMSNorm(dim)
            self.lm_head = nn.Linear(dim, vocab_size, bias=False)
            self.lm_head.weight = self.embed.weight
            nn.init.normal_(self.embed.weight, std=0.02)

        def forward(self, x, return_loss=False):
            if return_loss:
                targets = x[:, 1:].contiguous()
                x = x[:, :-1]
            h = self.embed(x)
            for layer in self.layers:
                out, _ = layer(h)
                h = h + out
            h = self.norm(h)
            logits = self.lm_head(h)
            if return_loss:
                return F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            return logits

    model = E1LM(256, dim, depth).cuda().bfloat16()

elif model_type in ("e29a", "e29b"):
    from elman.models.e29_selective import E29aSelectiveElmanCell, E29bSelectiveElmanCell

    Cell = E29aSelectiveElmanCell if model_type == "e29a" else E29bSelectiveElmanCell

    class E29LM(nn.Module):
        def __init__(self, vocab_size, dim, depth, n_slots):
            super().__init__()
            self.vocab_size = vocab_size
            self.embed = nn.Embedding(vocab_size, dim)
            self.layers = nn.ModuleList([Cell(dim, n_slots=n_slots) for _ in range(depth)])
            self.norm = nn.RMSNorm(dim)
            self.lm_head = nn.Linear(dim, vocab_size, bias=False)
            self.lm_head.weight = self.embed.weight
            nn.init.normal_(self.embed.weight, std=0.02)

        def forward(self, x, return_loss=False):
            if return_loss:
                targets = x[:, 1:].contiguous()
                x = x[:, :-1]
            h = self.embed(x)
            for layer in self.layers:
                out, _, _ = layer(h, use_cuda=True)  # CUDA forward, Python backward
                h = h + out
            h = self.norm(h)
            logits = self.lm_head(h)
            if return_loss:
                return F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            return logits

    model = E29LM(256, dim, depth, n_slots).cuda().bfloat16()

else:
    print(f"ERROR: Unknown model type: {model_type}")
    sys.exit(1)

# Count parameters
n_params = sum(p.numel() for p in model.parameters())
print(f'{model_type.upper()} D={dim} depth={depth}: params={n_params:,}', flush=True)

# Training setup
opt = AdamWScheduleFree(model.parameters(), lr=3e-4, weight_decay=0.1)
model.train()
opt.train()

losses = []
start = time.time()
step = 0

while time.time() - start < time_limit:
    step += 1
    batch = get_batch_deterministic(step - 1)  # 0-indexed
    opt.zero_grad()
    loss = model(batch, return_loss=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    losses.append(loss.item())

    if step % 50 == 0:
        elapsed = time.time() - start
        tokens = step * batch_size * seq_len
        avg100 = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
        print(f'Step {step} | {elapsed:.0f}s | Loss {loss.item():.4f} | Avg100 {avg100:.4f} | {int(tokens/elapsed)/1000:.1f}K tok/s', flush=True)

elapsed = time.time() - start
tokens = step * batch_size * seq_len
avg_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
tps = int(tokens / elapsed)
print(f'FINAL: steps={step}, params={n_params/1e6:.1f}M, loss={avg_loss:.4f}, tok/s={tps/1000:.1f}K', flush=True)
mm.close()
