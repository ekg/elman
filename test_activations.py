#!/usr/bin/env python3
"""
Experiment 1: Compare activation functions for E1
- tanh (baseline)
- softsign (cheap, smooth)
- hardtanh (cheapest, has dead units)
- silu (expensive but popular)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import mmap
import numpy as np
import os
import sys

def softsign(x):
    return x / (1 + torch.abs(x))

def train_model(gpu_id, activation_name):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    device = 'cuda'
    dtype = torch.bfloat16

    # Data setup
    data_path = 'data/pile.txt'
    with open(data_path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    data_len = len(mm)

    def get_batch(batch_size, seq_len):
        positions = np.random.randint(0, data_len - seq_len - 1, size=batch_size)
        buf = np.zeros((batch_size, seq_len + 1), dtype=np.uint8)
        for i, pos in enumerate(positions):
            buf[i] = np.frombuffer(mm[pos:pos+seq_len+1], dtype=np.uint8)
        return torch.from_numpy(buf).long().to(device)

    # Select activation function
    if activation_name == 'tanh':
        activation = torch.tanh
    elif activation_name == 'softsign':
        activation = softsign
    elif activation_name == 'hardtanh':
        activation = F.hardtanh
    elif activation_name == 'silu':
        activation = F.silu
    else:
        raise ValueError(f"Unknown activation: {activation_name}")

    class ElmanCell(nn.Module):
        def __init__(self, dim, activation_fn):
            super().__init__()
            self.dim = dim
            self.activation = activation_fn
            self.W_x = nn.Parameter(torch.empty(dim, dim))
            self.W_h = nn.Parameter(torch.empty(dim, dim))
            self.b = nn.Parameter(torch.zeros(dim))

            # Mamba2-style init
            nn.init.normal_(self.W_x, std=0.02)
            W_h_fp32 = torch.empty(dim, dim, dtype=torch.float32)
            nn.init.orthogonal_(W_h_fp32)
            W_h_fp32.mul_(0.999)
            self.W_h.data.copy_(W_h_fp32)

        def forward(self, x, z, h0=None):
            T, B, D = x.shape
            if h0 is None:
                h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)

            h = h0
            outputs = []
            for t in range(T):
                x_t = x[t]
                z_t = z[t]
                pre = x_t @ self.W_x.T + h @ self.W_h.T + self.b
                h = self.activation(pre)
                out = h * F.silu(z_t)
                outputs.append(out)

            return torch.stack(outputs, dim=0), h

    class E1Layer(nn.Module):
        def __init__(self, dim, activation_fn):
            super().__init__()
            self.in_proj = nn.Linear(dim, 2 * dim, bias=False)
            self.cell = ElmanCell(dim, activation_fn)
            self.out_proj = nn.Linear(dim, dim, bias=False)

            nn.init.normal_(self.in_proj.weight, std=0.02)
            nn.init.normal_(self.out_proj.weight, std=0.02)

        def forward(self, x, h0=None):
            B, T, D = x.shape
            xz = self.in_proj(x)
            x_proj, z = xz.chunk(2, dim=-1)
            x_proj = F.silu(x_proj)

            x_rnn = x_proj.permute(1, 0, 2).contiguous()
            z_rnn = z.permute(1, 0, 2).contiguous()

            out, h = self.cell(x_rnn, z_rnn, h0)
            out = out.permute(1, 0, 2).contiguous()
            return self.out_proj(out), h

    class E1LM(nn.Module):
        def __init__(self, vocab_size, dim, depth, activation_fn):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, dim)
            self.layers = nn.ModuleList([
                E1Layer(dim, activation_fn) for _ in range(depth)
            ])
            self.norm = nn.LayerNorm(dim)
            self.lm_head = nn.Linear(dim, vocab_size, bias=False)
            self.embed.weight = self.lm_head.weight
            self.vocab_size = vocab_size

            nn.init.normal_(self.embed.weight, std=0.02)

        def forward(self, x, return_loss=False):
            if return_loss:
                inp, target = x[:, :-1], x[:, 1:]
            else:
                inp = x

            x = self.embed(inp)
            for layer in self.layers:
                x_out, _ = layer(x)
                x = x + x_out
            x = self.norm(x)
            logits = self.lm_head(x)

            if return_loss:
                loss = F.cross_entropy(logits.view(-1, self.vocab_size), target.reshape(-1))
                return loss
            return logits

    B, T = 16, 512
    dim, depth = 512, 12
    steps = 3000
    lr = 3e-4

    model = E1LM(256, dim, depth, activation).to(device).to(dtype)
    params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    losses = []
    start = time.time()

    for step in range(steps):
        batch = get_batch(B, T)
        loss = model(batch, return_loss=True)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())

        if (step + 1) % 1000 == 0:
            recent_loss = sum(losses[-100:]) / 100
            print(f'[GPU {gpu_id}] {activation_name}: Step {step+1}, loss={recent_loss:.4f}', flush=True)

    elapsed = time.time() - start
    final_loss = sum(losses[-100:]) / 100
    tok_per_sec = (B * T * steps) / elapsed

    return final_loss, tok_per_sec


if __name__ == '__main__':
    gpu_id = int(sys.argv[1])
    activation_name = sys.argv[2]

    loss, tok_s = train_model(gpu_id, activation_name)
    print(f'RESULT: {activation_name}, loss={loss:.4f}, tok_s={tok_s/1000:.1f}K')
