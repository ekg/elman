"""Test if pure bf16 element-wise ops maintain numerical stability."""

import torch
import torch.nn.functional as F
import numpy as np

torch.manual_seed(42)
device = 'cuda'

# Simulate E1's element-wise operations
batch, dim = 48, 1024
n_steps = 512

print("Testing bf16 vs f32 numerical stability for E1 ops\n")

# Test 1: tanh precision
print("=" * 60)
print("Test 1: tanh precision")
x_f32 = torch.randn(batch, dim, device=device, dtype=torch.float32)
x_bf16 = x_f32.bfloat16()

tanh_f32 = torch.tanh(x_f32)
tanh_bf16 = torch.tanh(x_bf16)
tanh_bf16_via_f32 = torch.tanh(x_bf16.float()).bfloat16()

diff_direct = (tanh_bf16.float() - tanh_f32).abs()
diff_via_f32 = (tanh_bf16_via_f32.float() - tanh_f32).abs()

print(f"Direct bf16 tanh:     max_err={diff_direct.max():.6f}, mean_err={diff_direct.mean():.6f}")
print(f"bf16→f32→tanh→bf16:  max_err={diff_via_f32.max():.6f}, mean_err={diff_via_f32.mean():.6f}")

# Test 2: sigmoid precision  
print("\n" + "=" * 60)
print("Test 2: sigmoid precision")
sig_f32 = torch.sigmoid(x_f32)
sig_bf16 = torch.sigmoid(x_bf16)
sig_bf16_via_f32 = torch.sigmoid(x_bf16.float()).bfloat16()

diff_direct = (sig_bf16.float() - sig_f32).abs()
diff_via_f32 = (sig_bf16_via_f32.float() - sig_f32).abs()

print(f"Direct bf16 sigmoid:     max_err={diff_direct.max():.6f}, mean_err={diff_direct.mean():.6f}")
print(f"bf16→f32→sigmoid→bf16:  max_err={diff_via_f32.max():.6f}, mean_err={diff_via_f32.mean():.6f}")

# Test 3: Accumulation error over sequence (critical for RNN)
print("\n" + "=" * 60)
print("Test 3: Recurrence accumulation error over 512 steps")

# Simulate: h_t = gate * h_{t-1} + (1-gate) * tanh(input)
h_f32 = torch.zeros(batch, dim, device=device, dtype=torch.float32)
h_bf16 = torch.zeros(batch, dim, device=device, dtype=torch.bfloat16)
h_bf16_careful = torch.zeros(batch, dim, device=device, dtype=torch.bfloat16)

# Fixed gate ~0.9 (typical for trained model)
gate = torch.full((batch, dim), 0.9, device=device)

for t in range(n_steps):
    # Random input each step
    inp = torch.randn(batch, dim, device=device)
    
    # f32 reference
    new_h_f32 = gate * h_f32 + (1 - gate) * torch.tanh(inp)
    h_f32 = new_h_f32
    
    # Pure bf16
    inp_bf16 = inp.bfloat16()
    gate_bf16 = gate.bfloat16()
    new_h_bf16 = gate_bf16 * h_bf16 + (1 - gate_bf16) * torch.tanh(inp_bf16)
    h_bf16 = new_h_bf16
    
    # bf16 with f32 internal (current approach)
    new_h_careful = (gate_bf16.float() * h_bf16_careful.float() + 
                     (1 - gate_bf16.float()) * torch.tanh(inp_bf16.float())).bfloat16()
    h_bf16_careful = new_h_careful

diff_pure = (h_bf16.float() - h_f32).abs()
diff_careful = (h_bf16_careful.float() - h_f32).abs()

print(f"Pure bf16 after {n_steps} steps:     max_err={diff_pure.max():.6f}, mean_err={diff_pure.mean():.6f}")
print(f"bf16 w/ f32 internal:               max_err={diff_careful.max():.6f}, mean_err={diff_careful.mean():.6f}")
print(f"Error ratio (pure/careful):         {(diff_pure.mean() / diff_careful.mean()):.2f}x")

# Test 4: Gradient accumulation
print("\n" + "=" * 60)
print("Test 4: Gradient accumulation (bias gradient pattern)")

# Simulate accumulating gradients over batch*seq_len elements
grads_f32 = torch.randn(batch * n_steps, dim, device=device, dtype=torch.float32) * 0.01
grads_bf16 = grads_f32.bfloat16()

sum_f32 = grads_f32.sum(dim=0)
sum_bf16 = grads_bf16.sum(dim=0)
sum_bf16_via_f32 = grads_bf16.float().sum(dim=0)

diff_direct = (sum_bf16.float() - sum_f32).abs()
diff_via_f32 = (sum_bf16_via_f32 - sum_f32).abs()

print(f"Direct bf16 sum:         max_err={diff_direct.max():.4f}, mean_err={diff_direct.mean():.4f}")
print(f"bf16→f32 sum:           max_err={diff_via_f32.max():.4f}, mean_err={diff_via_f32.mean():.4f}")
print(f"Error ratio:            {(diff_direct.mean() / (diff_via_f32.mean() + 1e-10)):.1f}x")

# Test 5: Timing comparison
print("\n" + "=" * 60)
print("Test 5: Throughput comparison")

x = torch.randn(batch, n_steps, dim, device=device, dtype=torch.bfloat16)
gate = torch.rand(batch, n_steps, dim, device=device, dtype=torch.bfloat16) * 0.2 + 0.8

# Warmup
for _ in range(10):
    _ = gate * torch.tanh(x) + (1 - gate) * x
torch.cuda.synchronize()

# Pure bf16
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(100):
    out_bf16 = gate * torch.tanh(x) + (1 - gate) * x
end.record()
torch.cuda.synchronize()
time_bf16 = start.elapsed_time(end)

# bf16 with f32 internal
start.record()
for _ in range(100):
    out_f32 = (gate.float() * torch.tanh(x.float()) + (1 - gate.float()) * x.float()).bfloat16()
end.record()
torch.cuda.synchronize()
time_f32 = start.elapsed_time(end)

print(f"Pure bf16:           {time_bf16:.2f} ms")
print(f"bf16→f32→bf16:      {time_f32:.2f} ms")
print(f"Speedup:             {time_f32/time_bf16:.2f}x")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"- Single-op precision: bf16 is fine for tanh/sigmoid")
print(f"- Recurrence: {n_steps}-step accumulation has {(diff_pure.mean() / diff_careful.mean()):.1f}x more error in pure bf16")
print(f"- Gradient sums: Direct bf16 sum has ~{(diff_direct.mean() / (diff_via_f32.mean() + 1e-10)):.0f}x more error")
print(f"- Throughput gain: {time_f32/time_bf16:.2f}x for element-wise ops")
