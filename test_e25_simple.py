"""Super simple test for E25 backward."""

import torch

torch.manual_seed(42)

B = 1
T = 1
D = 256
N = 8

device = 'cuda'
dtype = torch.bfloat16

print("Testing a very simple case: tanh(x @ W.T)")

# Simple test: y = tanh(x @ W.T), compute dy/dW
x = torch.randn(B, D, device=device, dtype=dtype)
W = torch.randn(D, D, device=device, dtype=dtype) * 0.1

# Forward
y = torch.tanh(x @ W.T)
print(f"y sample: {y[0, :4].float().tolist()}")

# Backward
dy = torch.ones_like(y)
d_pre = dy * (1 - y ** 2)
print(f"d_pre sample: {d_pre[0, :4].float().tolist()}")

dW = d_pre.float().T @ x.float()
print(f"dW sample: {dW[0, :4].float().tolist()}")
print(f"dW norm: {dW.norm().item():.4f}")

# Compare with autograd
W_auto = W.clone().detach().requires_grad_(True)
x_auto = x.clone().detach()
y_auto = torch.tanh(x_auto @ W_auto.T)
loss = y_auto.sum()
loss.backward()

print(f"\nAutograd dW sample: {W_auto.grad[0, :4].float().tolist()}")
print(f"Autograd dW norm: {W_auto.grad.float().norm().item():.4f}")

print(f"\nDiff: {(dW.to(dtype) - W_auto.grad).abs().max().item():.6f}")
