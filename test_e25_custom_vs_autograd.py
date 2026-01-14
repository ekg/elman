"""Compare E25 custom backward vs pure autograd on learnable pattern."""

import torch
import torch.nn as nn
import math

torch.manual_seed(42)

B = 4
T = 64
D = 256  # Must be supported by CUDA kernel
N = 8    # Must be supported by CUDA kernel

device = 'cuda'
dtype = torch.bfloat16  # CUDA kernel requires bfloat16

print("="*60)
print("E25: Custom Backward vs Pure Autograd")
print("="*60)

from elman.models.e25_entmax import E25DualMemoryElmanCell, entmax_1_5

class E25CellAutograd(nn.Module):
    """E25 cell using pure autograd (no custom backward)."""

    def __init__(self, dim, n_slots=64, w_h_init_scale=0.9):
        super().__init__()
        self.dim = dim
        self.n_slots = n_slots
        self.scale = 1.0 / math.sqrt(dim)

        self.W_h = nn.Parameter(torch.empty(dim, dim))
        self.W_x = nn.Parameter(torch.empty(dim, dim))
        self.b_h = nn.Parameter(torch.zeros(dim))
        self.W_write = nn.Parameter(torch.empty(dim, dim))

        self._init_weights(w_h_init_scale)

    def _init_weights(self, w_h_init_scale):
        W_h_fp32 = torch.empty_like(self.W_h, dtype=torch.float32)
        nn.init.orthogonal_(W_h_fp32)
        W_h_fp32.mul_(w_h_init_scale)
        with torch.no_grad():
            self.W_h.copy_(W_h_fp32.to(self.W_h.dtype))

        nn.init.xavier_uniform_(self.W_x)
        nn.init.xavier_uniform_(self.W_write)

    def forward(self, x_seq, h_tape=None, h_work=None):
        B, T, D = x_seq.shape

        if h_tape is None:
            h_tape = torch.zeros(B, self.n_slots, D, device=x_seq.device, dtype=x_seq.dtype)
        if h_work is None:
            h_work = torch.zeros(B, D, device=x_seq.device, dtype=x_seq.dtype)

        h_work_list = []

        for t in range(T):
            # Read
            read_scores = (h_tape * h_work[:, None, :]).sum(dim=-1) * self.scale
            read_attn = entmax_1_5(read_scores, dim=-1)
            read = (read_attn[:, :, None] * h_tape).sum(dim=1)

            # Update h_work
            pre_act = h_work @ self.W_h.T + x_seq[:, t] @ self.W_x.T + read + self.b_h
            h_work_new = torch.tanh(pre_act)

            # Write
            write_value = h_work_new @ self.W_write.T
            write_scores = (h_tape * h_work_new[:, None, :]).sum(dim=-1) * self.scale
            write_attn = entmax_1_5(write_scores, dim=-1)
            h_tape = (1 - write_attn[:, :, None]) * h_tape + write_attn[:, :, None] * write_value[:, None, :]

            h_work = h_work_new
            h_work_list.append(h_work)

        h_work_all = torch.stack(h_work_list, dim=1)
        return h_work_all, h_tape, h_work

# Learnable pattern: token at position t depends on t mod 8
vocab_size = 64

def generate_pattern_batch(batch_size, seq_len, vocab_size):
    """Generate tokens where target = (input + position%8) % vocab_size"""
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len+1), device=device)
    pos_offsets = torch.arange(seq_len, device=device) % 8
    for i in range(seq_len):
        tokens[:, i+1] = (tokens[:, i] + pos_offsets[i]) % vocab_size
    return tokens

def train_model(cell, embed, head, name, n_steps=200):
    optimizer = torch.optim.AdamW(
        list(cell.parameters()) + list(embed.parameters()) + list(head.parameters()),
        lr=1e-3
    )

    losses = []
    for step in range(n_steps):
        torch.manual_seed(1000 + step)  # Same data for both
        tokens = generate_pattern_batch(B, T, vocab_size)
        x = embed(tokens[:, :-1])
        target = tokens[:, 1:]

        optimizer.zero_grad()
        h_work_all, h_tape_final, h_work_final = cell(x)
        logits = head(h_work_all)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, vocab_size),
            target.reshape(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cell.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

        if step % 50 == 0:
            print(f"  {name} Step {step:3d}: Loss = {loss.item():.4f}")

    return losses

print("\n--- Training with Pure Autograd ---")
torch.manual_seed(42)
cell_auto = E25CellAutograd(dim=D, n_slots=N).to(device).to(dtype)
embed_auto = nn.Embedding(vocab_size, D).to(device).to(dtype)
head_auto = nn.Linear(D, vocab_size, bias=False).to(device).to(dtype)
losses_auto = train_model(cell_auto, embed_auto, head_auto, "Autograd")

print("\n--- Training with Custom Backward ---")
torch.manual_seed(42)
cell_custom = E25DualMemoryElmanCell(dim=D, n_slots=N).to(device).to(dtype)
embed_custom = nn.Embedding(vocab_size, D).to(device).to(dtype)
head_custom = nn.Linear(D, vocab_size, bias=False).to(device).to(dtype)
losses_custom = train_model(cell_custom, embed_custom, head_custom, "Custom", n_steps=200)

print("\n" + "="*60)
print("Results:")
print(f"  Autograd: {losses_auto[0]:.4f} -> {losses_auto[-1]:.4f} (improvement: {losses_auto[0] - losses_auto[-1]:.4f})")
print(f"  Custom:   {losses_custom[0]:.4f} -> {losses_custom[-1]:.4f} (improvement: {losses_custom[0] - losses_custom[-1]:.4f})")
print("="*60)
