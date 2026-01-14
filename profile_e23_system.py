#!/usr/bin/env python3
"""
Profile E23 full system to find integration overhead.

We see:
- Per-layer forward: E23 is 2.7x slower than E1 (24us vs 9us)
- Full training: E23 is 6.4x slower than E1 (31.5K vs 201.6K tok/s)

Where does the extra 2.4x slowdown come from?
"""
import torch
import torch.nn.functional as F
import sys
import time
sys.path.insert(0, 'elman/cuda')

from elman.models import LadderLM, create_ladder_model

batch_size = 64
seq_len = 512

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

print("=" * 70)
print("E23 vs E1 Full System Profiling")
print(f"batch={batch_size}, seq={seq_len}")
print("=" * 70)

# Create models
print("\nCreating models...")
e1_model = LadderLM(vocab_size=256, dim=1280, depth=6, level=1).cuda().bfloat16()
e23_model = create_ladder_model(target_params='50m', level=23, vocab_size=256).cuda().bfloat16()

e1_params = e1_model.get_num_params()
e23_params = e23_model.get_num_params()
print(f"E1:  dim=1280, depth=6, params={e1_params:,}")
print(f"E23: dim=512, depth=17, params={e23_params:,}")

# Create input
x = torch.randint(0, 256, (batch_size, seq_len + 1), device='cuda')

def profile_forward(model, name, x, n_iters=10):
    """Profile forward pass only."""
    model.eval()
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            logits = model(x)

    torch.cuda.synchronize()
    start.record()
    with torch.no_grad():
        for _ in range(n_iters):
            logits = model(x)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / n_iters
    return ms

def profile_forward_backward(model, name, x, n_iters=10):
    """Profile forward + backward pass."""
    model.train()
    # Warmup
    for _ in range(3):
        model.zero_grad()
        loss = model(x, return_loss=True)
        loss.backward()

    torch.cuda.synchronize()
    start.record()
    for _ in range(n_iters):
        model.zero_grad()
        loss = model(x, return_loss=True)
        loss.backward()
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / n_iters
    return ms

def profile_full_step(model, name, x, optimizer, n_iters=10):
    """Profile full training step including optimizer."""
    model.train()
    # Warmup
    for _ in range(3):
        optimizer.zero_grad()
        loss = model(x, return_loss=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    torch.cuda.synchronize()
    start.record()
    for _ in range(n_iters):
        optimizer.zero_grad()
        loss = model(x, return_loss=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / n_iters
    return ms

# Profile each component
print("\n" + "=" * 70)
print("1. Forward Pass Only (eval mode)")
print("=" * 70)

e1_fwd = profile_forward(e1_model, "E1", x)
e23_fwd = profile_forward(e23_model, "E23", x)
print(f"E1:  {e1_fwd:.2f}ms ({e1_fwd/seq_len*1000:.1f}us/step)")
print(f"E23: {e23_fwd:.2f}ms ({e23_fwd/seq_len*1000:.1f}us/step)")
print(f"Ratio: {e23_fwd/e1_fwd:.2f}x")

print("\n" + "=" * 70)
print("2. Forward + Backward Pass")
print("=" * 70)

e1_fb = profile_forward_backward(e1_model, "E1", x)
e23_fb = profile_forward_backward(e23_model, "E23", x)
print(f"E1:  {e1_fb:.2f}ms ({e1_fb/seq_len*1000:.1f}us/step)")
print(f"E23: {e23_fb:.2f}ms ({e23_fb/seq_len*1000:.1f}us/step)")
print(f"Ratio: {e23_fb/e1_fb:.2f}x")

# Backward only
e1_bwd = e1_fb - e1_fwd
e23_bwd = e23_fb - e23_fwd
print(f"\nBackward only (derived):")
print(f"E1:  {e1_bwd:.2f}ms")
print(f"E23: {e23_bwd:.2f}ms")
print(f"Ratio: {e23_bwd/e1_bwd:.2f}x")

print("\n" + "=" * 70)
print("3. Full Training Step (with optimizer)")
print("=" * 70)

from schedulefree import AdamWScheduleFree
e1_opt = AdamWScheduleFree(e1_model.parameters(), lr=3e-4, weight_decay=0.1)
e23_opt = AdamWScheduleFree(e23_model.parameters(), lr=3e-4, weight_decay=0.1)
e1_model.train(); e1_opt.train()
e23_model.train(); e23_opt.train()

e1_full = profile_full_step(e1_model, "E1", x, e1_opt)
e23_full = profile_full_step(e23_model, "E23", x, e23_opt)
print(f"E1:  {e1_full:.2f}ms ({e1_full/seq_len*1000:.1f}us/step)")
print(f"E23: {e23_full:.2f}ms ({e23_full/seq_len*1000:.1f}us/step)")
print(f"Ratio: {e23_full/e1_full:.2f}x")

# Optimizer overhead
e1_opt_time = e1_full - e1_fb
e23_opt_time = e23_full - e23_fb
print(f"\nOptimizer overhead (derived):")
print(f"E1:  {e1_opt_time:.2f}ms")
print(f"E23: {e23_opt_time:.2f}ms")

print("\n" + "=" * 70)
print("4. Per-Layer Analysis")
print("=" * 70)

# E1 has 6 layers, E23 has 17 layers
e1_per_layer = e1_fwd / 6
e23_per_layer = e23_fwd / 17
print(f"E1:  {e1_fwd:.2f}ms / 6 layers = {e1_per_layer:.2f}ms/layer")
print(f"E23: {e23_fwd:.2f}ms / 17 layers = {e23_per_layer:.2f}ms/layer")
print(f"Per-layer ratio: {e23_per_layer/e1_per_layer:.2f}x")

print("\n" + "=" * 70)
print("5. Throughput Calculation")
print("=" * 70)

tokens_per_step = batch_size * seq_len
e1_tps = tokens_per_step / e1_full * 1000
e23_tps = tokens_per_step / e23_full * 1000
print(f"E1:  {e1_tps/1000:.1f}K tok/s")
print(f"E23: {e23_tps/1000:.1f}K tok/s")
print(f"Ratio: {e1_tps/e23_tps:.2f}x (E1 faster)")

print("\n" + "=" * 70)
print("6. Breakdown Summary")
print("=" * 70)
print(f"{'Component':<25} {'E1':>10} {'E23':>10} {'Ratio':>10}")
print("-" * 55)
print(f"{'Forward (ms)':<25} {e1_fwd:>10.2f} {e23_fwd:>10.2f} {e23_fwd/e1_fwd:>10.2f}x")
print(f"{'Backward (ms)':<25} {e1_bwd:>10.2f} {e23_bwd:>10.2f} {e23_bwd/e1_bwd:>10.2f}x")
print(f"{'Optimizer (ms)':<25} {e1_opt_time:>10.2f} {e23_opt_time:>10.2f} {e23_opt_time/e1_opt_time if e1_opt_time > 0 else 0:>10.2f}x")
print(f"{'Full step (ms)':<25} {e1_full:>10.2f} {e23_full:>10.2f} {e23_full/e1_full:>10.2f}x")
print("-" * 55)
print(f"{'Throughput (K tok/s)':<25} {e1_tps/1000:>10.1f} {e23_tps/1000:>10.1f} {e1_tps/e23_tps:>10.2f}x")

print("\n" + "=" * 70)
print("7. Memory Analysis")
print("=" * 70)
torch.cuda.reset_peak_memory_stats()
_ = e1_model(x, return_loss=True)
e1_mem = torch.cuda.max_memory_allocated() / 1e9
torch.cuda.reset_peak_memory_stats()
_ = e23_model(x, return_loss=True)
e23_mem = torch.cuda.max_memory_allocated() / 1e9
print(f"E1 peak memory:  {e1_mem:.2f} GB")
print(f"E23 peak memory: {e23_mem:.2f} GB")
print(f"Ratio: {e23_mem/e1_mem:.2f}x")
