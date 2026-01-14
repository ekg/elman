"""Test E25 cell training using pure autograd (no custom backward)."""

import torch
import torch.nn as nn
import math

torch.manual_seed(42)

B = 4
T = 32
D = 128
N = 16

device = 'cuda'
dtype = torch.float32

print("="*60)
print("E25 Cell Training Test (Pure Autograd)")
print("="*60)

from elman.models.e25_entmax import entmax_1_5

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

# Create cell
cell = E25CellAutograd(dim=D, n_slots=N).to(device).to(dtype)

# Simple target: predict next token from random embedding
vocab_size = 256
embed = nn.Embedding(vocab_size, D).to(device).to(dtype)
head = nn.Linear(D, vocab_size, bias=False).to(device).to(dtype)

optimizer = torch.optim.AdamW(
    list(cell.parameters()) + list(embed.parameters()) + list(head.parameters()),
    lr=1e-3
)

print(f"Cell params: {sum(p.numel() for p in cell.parameters())/1e3:.1f}K")

# Training loop
losses = []
for step in range(100):
    tokens = torch.randint(0, vocab_size, (B, T+1), device=device)
    x = embed(tokens[:, :-1])  # [B, T, D]
    target = tokens[:, 1:]  # [B, T]

    optimizer.zero_grad()

    # Forward through cell (pure autograd)
    h_work_all, h_tape_final, h_work_final = cell(x)

    # Predict
    logits = head(h_work_all)  # [B, T, vocab_size]

    # Loss
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, vocab_size),
        target.reshape(-1)
    )

    loss.backward()

    # Check gradients
    if step == 0:
        print(f"\nGradient check at step 0:")
        print(f"  W_h grad norm: {cell.W_h.grad.norm().item():.4f}")
        print(f"  W_x grad norm: {cell.W_x.grad.norm().item():.4f}")
        print(f"  W_write grad norm: {cell.W_write.grad.norm().item():.4f}")

    optimizer.step()
    losses.append(loss.item())

    if step % 20 == 0:
        print(f"Step {step:3d}: Loss = {loss.item():.4f}")

print(f"\nInitial loss: {losses[0]:.4f}")
print(f"Final loss:   {losses[-1]:.4f}")
print(f"Improvement:  {losses[0] - losses[-1]:.4f}")

if losses[-1] < losses[0] - 0.1:
    print("\nSUCCESS: Loss is clearly decreasing with pure autograd!")
elif losses[-1] < losses[0]:
    print("\nMARGINAL: Loss decreased slightly.")
else:
    print("\nFAILURE: Loss did not decrease even with pure autograd!")

print("\n" + "="*60)
