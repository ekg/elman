
import sys; sys.path.insert(0, '/home/erikg/elman')
import os
os.environ['LD_LIBRARY_PATH'] = f"/home/erikg/.local/lib/python3.12/site-packages/torch/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

import torch
import torch.nn as nn
import numpy as np
import mmap
import time
from schedulefree import AdamWScheduleFree

from elman.models.e28_conv_elman import E28ConvElman

torch.manual_seed(42); np.random.seed(42)

dim = 1280
depth = 6
batch_size = 64
seq_len = 512
time_limit = 300  # 5 min - E28 Python is slow

# Data
with open('/home/erikg/elman/data/pile.txt', 'rb') as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
data_len = len(mm)

def get_batch(buf, mm, data_len, batch_size, seq_len):
    pos = np.random.randint(0, data_len - seq_len - 1, size=batch_size)
    for j, p in enumerate(pos):
        buf[j] = np.frombuffer(mm[p:p+seq_len+1], dtype=np.uint8)
    return torch.from_numpy(buf.astype(np.int64)).cuda()

class ElmanLM(nn.Module):
    def __init__(self, vocab_size, dim, depth):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([E28ConvElman(dim, expansion=1.0, d_conv=4) for _ in range(depth)])
        self.norm = nn.RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        # Proper initialization
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
            loss = torch.nn.functional.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            return loss
        return logits

model = ElmanLM(256, dim, depth).cuda().bfloat16()
n_params = sum(p.numel() for p in model.parameters())
print(f'E28 D={dim} depth={depth} batch={batch_size}: params={n_params:,}', flush=True)

opt = AdamWScheduleFree(model.parameters(), lr=3e-4, weight_decay=0.1)
model.train(); opt.train()
buf = np.zeros((batch_size, seq_len+1), dtype=np.uint8)
losses = []; start = time.time(); step = 0

while time.time() - start < time_limit:
    step += 1
    batch = get_batch(buf, mm, data_len, batch_size, seq_len)
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
tps = int(tokens/elapsed)
print(f'FINAL: steps={step}, params={n_params/1e6:.1f}M, loss={avg_loss:.4f}, tok/s={tps/1000:.1f}K', flush=True)
mm.close()
