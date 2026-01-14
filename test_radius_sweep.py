#!/usr/bin/env python3
"""Test higher spectral radius values for E1 W_h initialization."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import mmap
import numpy as np
import os
import sys

def train_model(gpu_id, radius):
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

    from elman.models.mamba_gated_elman import MambaGatedElmanCell, MambaGatedElman

    def custom_init(model, proj_std=0.02, wh_radius=0.99):
        for name, module in model.named_modules():
            if isinstance(module, MambaGatedElmanCell):
                with torch.no_grad():
                    W_x_fp32 = torch.empty_like(module.W_x, dtype=torch.float32)
                    nn.init.normal_(W_x_fp32, std=proj_std)
                    module.W_x.copy_(W_x_fp32.to(module.W_x.dtype))

                    W_h_fp32 = torch.empty_like(module.W_h, dtype=torch.float32)
                    nn.init.orthogonal_(W_h_fp32)
                    W_h_fp32.mul_(wh_radius)
                    module.W_h.copy_(W_h_fp32.to(module.W_h.dtype))

                    module.b.zero_()
            elif isinstance(module, MambaGatedElman):
                with torch.no_grad():
                    w_fp32 = torch.empty_like(module.in_proj.weight, dtype=torch.float32)
                    nn.init.normal_(w_fp32, std=proj_std)
                    module.in_proj.weight.copy_(w_fp32.to(module.in_proj.weight.dtype))

                    w_fp32 = torch.empty_like(module.out_proj.weight, dtype=torch.float32)
                    nn.init.normal_(w_fp32, std=proj_std)
                    module.out_proj.weight.copy_(w_fp32.to(module.out_proj.weight.dtype))
            elif isinstance(module, nn.Embedding):
                with torch.no_grad():
                    w_fp32 = torch.empty_like(module.weight, dtype=torch.float32)
                    nn.init.normal_(w_fp32, std=proj_std)
                    module.weight.copy_(w_fp32.to(module.weight.dtype))

    class E1LM(nn.Module):
        def __init__(self, vocab_size, dim, depth):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, dim)
            self.layers = nn.ModuleList([
                MambaGatedElman(dim=dim) for _ in range(depth)
            ])
            self.norm = nn.LayerNorm(dim)
            self.lm_head = nn.Linear(dim, vocab_size, bias=False)
            self.embed.weight = self.lm_head.weight
            self.vocab_size = vocab_size

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

    model = E1LM(256, dim, depth).to(device).to(dtype)
    custom_init(model, proj_std=0.02, wh_radius=radius)

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
            print(f'[GPU {gpu_id}] radius={radius}: Step {step+1}, loss={recent_loss:.4f}', flush=True)

    elapsed = time.time() - start
    final_loss = sum(losses[-100:]) / 100
    tok_per_sec = (B * T * steps) / elapsed

    return final_loss, tok_per_sec


if __name__ == '__main__':
    gpu_id = int(sys.argv[1])
    radius = float(sys.argv[2])

    loss, tok_s = train_model(gpu_id, radius)
    print(f'RESULT: radius={radius}, loss={loss:.4f}, tok_s={tok_s/1000:.1f}K')
