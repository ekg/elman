"""Quick training test for E25 to verify gradients work."""

import torch
import torch.nn.functional as F

torch.manual_seed(42)

print("="*60)
print("E25 Quick Training Test")
print("="*60)

from elman.models.ladder_lm import create_ladder_model

model = create_ladder_model('50m', vocab_size=256, level=25).cuda().bfloat16()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

print(f"\nModel: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

# Simple training loop
losses = []
for step in range(20):
    x = torch.randint(0, 256, (4, 128), device='cuda')

    optimizer.zero_grad()
    loss = model(x, return_loss=True)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if step % 5 == 0:
        print(f"Step {step:3d}: Loss = {loss.item():.4f}")

print(f"\nInitial loss: {losses[0]:.4f}")
print(f"Final loss:   {losses[-1]:.4f}")
print(f"Improvement:  {losses[0] - losses[-1]:.4f}")

if losses[-1] < losses[0]:
    print("\nSUCCESS: Loss is decreasing!")
else:
    print("\nWARNING: Loss is not decreasing - gradients may be broken")

print("\n" + "="*60)
