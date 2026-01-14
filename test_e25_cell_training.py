"""Test E25 cell training (not full ladder model)."""

import torch
import torch.nn as nn

torch.manual_seed(42)

B = 4
T = 32
D = 128
N = 16

device = 'cuda'
dtype = torch.float32  # Use float32 to eliminate precision issues

print("="*60)
print("E25 Cell Training Test (Python-only)")
print("="*60)

from elman.models.e25_entmax import E25DualMemoryElmanCell

# Create cell
cell = E25DualMemoryElmanCell(dim=D, n_slots=N).to(device).to(dtype)

# Simple target: predict next token from random embedding
vocab_size = 256
embed = nn.Embedding(vocab_size, D).to(device).to(dtype)
head = nn.Linear(D, vocab_size, bias=False).to(device).to(dtype)

optimizer = torch.optim.AdamW(
    list(cell.parameters()) + list(embed.parameters()) + list(head.parameters()),
    lr=1e-2  # Higher LR
)

print(f"Cell params: {sum(p.numel() for p in cell.parameters())/1e3:.1f}K")

# Training loop
losses = []
for step in range(200):
    tokens = torch.randint(0, vocab_size, (B, T+1), device=device)
    x = embed(tokens[:, :-1])  # [B, T, D]
    target = tokens[:, 1:]  # [B, T]

    optimizer.zero_grad()

    # Forward through cell (Python only)
    h_work_all, h_tape_final, h_work_final = cell(x, use_cuda=False)

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

    if step % 10 == 0:
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
