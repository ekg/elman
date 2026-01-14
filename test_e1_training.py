"""Test E1 (gated Elman) cell training as baseline."""

import torch
import torch.nn as nn

torch.manual_seed(42)

B = 4
T = 32
D = 128

device = 'cuda'
dtype = torch.float32

print("="*60)
print("E1 Cell Training Test (Baseline)")
print("="*60)

class E1CellSimple(nn.Module):
    """Simple E1 cell (gated Elman) using pure autograd."""

    def __init__(self, dim, w_h_init_scale=0.9):
        super().__init__()
        self.dim = dim

        self.W_h = nn.Parameter(torch.empty(dim, dim))
        self.W_x = nn.Parameter(torch.empty(dim, dim))
        self.b_h = nn.Parameter(torch.zeros(dim))
        self.W_g = nn.Parameter(torch.empty(dim, dim))
        self.b_g = nn.Parameter(torch.zeros(dim))

        self._init_weights(w_h_init_scale)

    def _init_weights(self, w_h_init_scale):
        W_h_fp32 = torch.empty_like(self.W_h, dtype=torch.float32)
        nn.init.orthogonal_(W_h_fp32)
        W_h_fp32.mul_(w_h_init_scale)
        with torch.no_grad():
            self.W_h.copy_(W_h_fp32.to(self.W_h.dtype))

        nn.init.xavier_uniform_(self.W_x)
        nn.init.xavier_uniform_(self.W_g)

    def forward(self, x_seq, h=None):
        B, T, D = x_seq.shape

        if h is None:
            h = torch.zeros(B, D, device=x_seq.device, dtype=x_seq.dtype)

        h_list = []

        for t in range(T):
            # Gated Elman update
            pre_act = h @ self.W_h.T + x_seq[:, t] @ self.W_x.T + self.b_h
            h_cand = torch.tanh(pre_act)

            # Simple gate
            gate = torch.sigmoid(x_seq[:, t] @ self.W_g.T + self.b_g)
            h = gate * h_cand + (1 - gate) * h

            h_list.append(h)

        h_all = torch.stack(h_list, dim=1)
        return h_all, h

# Create cell
cell = E1CellSimple(dim=D).to(device).to(dtype)

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

    # Forward through cell
    h_all, h_final = cell(x)

    # Predict
    logits = head(h_all)  # [B, T, vocab_size]

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

    optimizer.step()
    losses.append(loss.item())

    if step % 20 == 0:
        print(f"Step {step:3d}: Loss = {loss.item():.4f}")

print(f"\nInitial loss: {losses[0]:.4f}")
print(f"Final loss:   {losses[-1]:.4f}")
print(f"Improvement:  {losses[0] - losses[-1]:.4f}")

if losses[-1] < losses[0] - 0.1:
    print("\nSUCCESS: Loss is clearly decreasing!")
elif losses[-1] < losses[0]:
    print("\nMARGINAL: Loss decreased slightly.")
else:
    print("\nFAILURE: Loss did not decrease!")

print("\n" + "="*60)
