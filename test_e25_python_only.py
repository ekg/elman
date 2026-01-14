"""Test E25 training with Python-only implementation."""

import torch
import torch.nn.functional as F

torch.manual_seed(42)

print("="*60)
print("E25 Python-Only Training Test")
print("="*60)

from elman.models.ladder_lm import create_ladder_model

model = create_ladder_model('50m', vocab_size=256, level=25).cuda().bfloat16()

# Force Python path
for layer in model.layers:
    if hasattr(layer, 'mixer') and hasattr(layer.mixer, 'cell'):
        # Set a flag to use Python
        layer.mixer.cell._force_python = True

# Monkey-patch the forward to use Python
def patch_cell(cell):
    original_forward = cell.forward
    def patched_forward(x_seq, h_tape=None, h_work=None, use_cuda=True):
        return original_forward(x_seq, h_tape, h_work, use_cuda=False)  # Force Python
    cell.forward = patched_forward

for layer in model.layers:
    if hasattr(layer, 'mixer') and hasattr(layer.mixer, 'cell'):
        patch_cell(layer.mixer.cell)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

print(f"\nModel: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
print("Using Python-only backward (no CUDA kernel)")

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
    print("\nSUCCESS: Loss is decreasing with Python implementation!")
else:
    print("\nWARNING: Loss is not decreasing even with Python!")

print("\n" + "="*60)
