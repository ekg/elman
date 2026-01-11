
import sys; sys.path.insert(0, '/home/erikg/elman')
import os
os.environ['LD_LIBRARY_PATH'] = f"/home/erikg/.local/lib/python3.12/site-packages/torch/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

import torch
import torch.nn as nn
import numpy as np
import mmap
import time
from schedulefree import AdamWScheduleFree

sys.path.insert(0, '/home/erikg/elman/elman/cuda')
import hasty_pytorch_lib

torch.manual_seed(42); np.random.seed(42)

model_type = "e25"
dim = 512
n_slots = 32
depth = 32
batch_size = 64
seq_len = 512
time_limit = 600  # 10 minutes

# Data setup
with open('/home/erikg/elman/data/pile.txt', 'rb') as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
data_len = len(mm)

def get_batch(buf, mm, data_len, batch_size, seq_len):
    pos = np.random.randint(0, data_len - seq_len - 1, size=batch_size)
    for j, p in enumerate(pos):
        buf[j] = np.frombuffer(mm[p:p+seq_len+1], dtype=np.uint8)
    return torch.from_numpy(buf.astype(np.int64)).cuda()


class DualMemoryCell(nn.Module):
    """Dual-Memory Elman Cell for E25/E26"""
    def __init__(self, dim, n_slots=64, model_type="e25", w_h_init_scale=0.9):
        super().__init__()
        self.dim = dim
        self.n_slots = n_slots
        self.model_type = model_type

        self.W_h = nn.Linear(dim, dim, bias=False)
        self.W_x = nn.Linear(dim, dim, bias=False)
        self.b_h = nn.Parameter(torch.zeros(dim))
        self.W_write = nn.Linear(dim, dim, bias=False)

        with torch.no_grad():
            nn.init.orthogonal_(self.W_h.weight)
            self.W_h.weight.mul_(w_h_init_scale)
            nn.init.xavier_uniform_(self.W_x.weight)
            nn.init.xavier_uniform_(self.W_write.weight)

    def forward(self, x_seq, h_tape=None, h_work=None):
        B, T, D = x_seq.shape
        device = x_seq.device
        dtype = x_seq.dtype

        if h_tape is None:
            h_tape = torch.zeros(B, self.n_slots, D, device=device, dtype=dtype)
        if h_work is None:
            h_work = torch.zeros(B, D, device=device, dtype=dtype)

        if self.model_type == "e25":
            results = hasty_pytorch_lib.e25_entmax_forward(
                self.training,
                x_seq,
                h_tape,
                h_work,
                self.W_h.weight,
                self.W_x.weight,
                self.b_h,
                self.W_write.weight
            )
        else:  # e26
            results = hasty_pytorch_lib.e26_parallel_forward(
                self.training,
                x_seq,
                h_tape,
                h_work,
                self.W_h.weight,
                self.W_x.weight,
                self.b_h,
                self.W_write.weight
            )

        return results[0], results[1], results[0][:, -1]


class DualMemoryLayer(nn.Module):
    """Dual-Memory layer with Mamba2-style input/output projections."""
    def __init__(self, dim, expansion=1.0, n_slots=64, model_type="e25"):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.n_slots = n_slots

        self.in_proj = nn.Linear(dim, 2 * self.d_inner, bias=False)
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)
        self.cell = DualMemoryCell(self.d_inner, n_slots, model_type)

    def forward(self, x, hidden=None):
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)
        x_proj = torch.nn.functional.silu(x_proj)

        if hidden is not None:
            h_tape, h_work = hidden
        else:
            h_tape, h_work = None, None

        output, h_tape_final, h_work_final = self.cell(x_proj, h_tape, h_work)
        output = output * torch.nn.functional.silu(z)
        output = self.out_proj(output)
        return output, (h_tape_final, h_work_final)


class DualMemoryLM(nn.Module):
    """Dual-Memory Language Model."""
    def __init__(self, vocab_size, dim, depth, n_slots=64, model_type="e25", expansion=1.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.depth = depth
        self.n_slots = n_slots
        self.model_type = model_type

        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            DualMemoryLayer(dim, expansion, n_slots, model_type)
            for _ in range(depth)
        ])
        self.norm = nn.RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, x, return_loss=False):
        if return_loss:
            inputs = x[:, :-1]
            targets = x[:, 1:]
        else:
            inputs = x

        h = self.embed(inputs)
        for layer in self.layers:
            out, _ = layer(h)
            h = h + out
        h = self.norm(h)
        logits = self.lm_head(h)

        if return_loss:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.reshape(-1)
            )
            return loss
        return logits

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Create model
model = DualMemoryLM(
    vocab_size=256,
    dim=dim,
    depth=depth,
    n_slots=n_slots,
    model_type=model_type,
    expansion=1.0
)
model = model.cuda().bfloat16()
n_params = model.get_num_params()
print(f'{model_type.upper()} D={dim} N={n_slots} depth={depth} batch={batch_size}: params={n_params:,}', flush=True)

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
