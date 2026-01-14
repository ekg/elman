#!/usr/bin/env python3
"""
Profile E23 layer stacking to understand where slowdown comes from.

Single layer: E23 is 2x slower than E1
Full model: E23 is 7x slower than E1

Is the difference just layer count? E23 has 17 layers vs E1's 6.
"""
import torch
import sys
sys.path.insert(0, 'elman/cuda')

from elman.models import LadderLM

batch_size = 64
seq_len = 512

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

print("=" * 70)
print("E23 vs E1 Layer Stacking Analysis")
print(f"batch={batch_size}, seq={seq_len}")
print("=" * 70)

# Create models with SAME depth to isolate per-layer difference
print("\n1. Same Depth Comparison (depth=6)")
print("-" * 70)

e1_d6 = LadderLM(vocab_size=256, dim=1280, depth=6, level=1).cuda().bfloat16()
e23_d6 = LadderLM(vocab_size=256, dim=512, depth=6, level=23, n_slots=8).cuda().bfloat16()

print(f"E1  depth=6: {e1_d6.get_num_params():,} params")
print(f"E23 depth=6: {e23_d6.get_num_params():,} params")

x = torch.randint(0, 256, (batch_size, seq_len + 1), device='cuda')

# Profile E1 depth=6
e1_d6.train()
for _ in range(3):
    e1_d6.zero_grad()
    loss = e1_d6(x, return_loss=True)
    loss.backward()

torch.cuda.synchronize()
start.record()
for _ in range(10):
    e1_d6.zero_grad()
    loss = e1_d6(x, return_loss=True)
    loss.backward()
end.record()
torch.cuda.synchronize()
e1_d6_ms = start.elapsed_time(end) / 10
print(f"E1  depth=6 fwd+bwd: {e1_d6_ms:.2f}ms")

# Profile E23 depth=6
e23_d6.train()
for _ in range(3):
    e23_d6.zero_grad()
    loss = e23_d6(x, return_loss=True)
    loss.backward()

torch.cuda.synchronize()
start.record()
for _ in range(10):
    e23_d6.zero_grad()
    loss = e23_d6(x, return_loss=True)
    loss.backward()
end.record()
torch.cuda.synchronize()
e23_d6_ms = start.elapsed_time(end) / 10
print(f"E23 depth=6 fwd+bwd: {e23_d6_ms:.2f}ms")
print(f"Ratio at same depth: {e23_d6_ms/e1_d6_ms:.2f}x")

del e1_d6, e23_d6
torch.cuda.empty_cache()

print("\n2. E23 at Different Depths")
print("-" * 70)

for depth in [6, 12, 17]:
    model = LadderLM(vocab_size=256, dim=512, depth=depth, level=23, n_slots=8).cuda().bfloat16()
    model.train()

    for _ in range(3):
        model.zero_grad()
        loss = model(x, return_loss=True)
        loss.backward()

    torch.cuda.synchronize()
    start.record()
    for _ in range(5):
        model.zero_grad()
        loss = model(x, return_loss=True)
        loss.backward()
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / 5
    per_layer = ms / depth
    print(f"E23 depth={depth:2d}: {ms:7.2f}ms total, {per_layer:.2f}ms/layer, {model.get_num_params():,} params")

    del model
    torch.cuda.empty_cache()

print("\n3. E1 at Different Depths")
print("-" * 70)

for depth in [6, 12, 17]:
    model = LadderLM(vocab_size=256, dim=1280, depth=depth, level=1).cuda().bfloat16()
    model.train()

    for _ in range(3):
        model.zero_grad()
        loss = model(x, return_loss=True)
        loss.backward()

    torch.cuda.synchronize()
    start.record()
    for _ in range(5):
        model.zero_grad()
        loss = model(x, return_loss=True)
        loss.backward()
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / 5
    per_layer = ms / depth
    print(f"E1  depth={depth:2d}: {ms:7.2f}ms total, {per_layer:.2f}ms/layer, {model.get_num_params():,} params")

    del model
    torch.cuda.empty_cache()

print("\n4. Standard Configs (50M params)")
print("-" * 70)

# E1 standard: dim=1280, depth=6
e1_std = LadderLM(vocab_size=256, dim=1280, depth=6, level=1).cuda().bfloat16()
e1_std.train()

for _ in range(3):
    e1_std.zero_grad()
    loss = e1_std(x, return_loss=True)
    loss.backward()

torch.cuda.synchronize()
start.record()
for _ in range(10):
    e1_std.zero_grad()
    loss = e1_std(x, return_loss=True)
    loss.backward()
end.record()
torch.cuda.synchronize()
e1_std_ms = start.elapsed_time(end) / 10

# E23 standard: dim=512, depth=17
e23_std = LadderLM(vocab_size=256, dim=512, depth=17, level=23, n_slots=8).cuda().bfloat16()
e23_std.train()

for _ in range(3):
    e23_std.zero_grad()
    loss = e23_std(x, return_loss=True)
    loss.backward()

torch.cuda.synchronize()
start.record()
for _ in range(10):
    e23_std.zero_grad()
    loss = e23_std(x, return_loss=True)
    loss.backward()
end.record()
torch.cuda.synchronize()
e23_std_ms = start.elapsed_time(end) / 10

print(f"E1  d1280×6:  {e1_std_ms:.2f}ms ({e1_std.get_num_params():,} params)")
print(f"E23 d512×17:  {e23_std_ms:.2f}ms ({e23_std.get_num_params():,} params)")
print(f"Ratio: {e23_std_ms/e1_std_ms:.2f}x")

print("\n5. Analysis")
print("-" * 70)
print(f"E1 per-layer (from depth scaling): ~{e1_std_ms/6:.2f}ms")
print(f"E23 per-layer (from depth scaling): ~{e23_std_ms/17:.2f}ms")
print(f"Per-layer ratio: {(e23_std_ms/17)/(e1_std_ms/6):.2f}x")
print()
print("If E23 used same depth as E1 (6 layers):")
e23_hypothetical = (e23_std_ms/17) * 6
print(f"  E23 d512×6 would be: ~{e23_hypothetical:.2f}ms")
print(f"  Ratio would be: {e23_hypothetical/e1_std_ms:.2f}x")
