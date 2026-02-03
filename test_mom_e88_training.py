#!/usr/bin/env python
"""
Quick training test for MoM E88 to verify it works and measure throughput.
"""

import time
import torch
import sys
sys.path.insert(0, '/home/erikg/elman')

from elman.models import LadderLM

device = 'cuda:0'
dtype = torch.bfloat16

# Test configs - similar to E88 but with MoM routing
configs = [
    # (level, dim, depth, n_heads, top_k, n_state, description)
    ('MoME88', 512, 12, 64, 16, 16, 'Small: h=64, k=16, n=16'),
    ('MoME88', 768, 16, 96, 24, 16, 'Medium: h=96, k=24, n=16'),
    ('MoME88', 512, 12, 64, 16, 32, 'Small n=32: h=64, k=16, n=32'),
    ('E88', 512, 12, 64, None, 16, 'E88 baseline: h=64, n=16'),
]

batch_size = 8
chunk_size = 512
warmup_steps = 5
measure_steps = 20

print("=" * 70)
print("MoM E88 Training Test")
print("=" * 70)

for level, dim, depth, n_heads, top_k, n_state, desc in configs:
    print(f"\n{desc}")
    print("-" * 50)

    try:
        # Build model
        kwargs = {
            'vocab_size': 256,
            'dim': dim,
            'depth': depth,
            'n_heads': n_heads,
            'n_state': n_state,
        }
        if top_k is not None:
            kwargs['top_k'] = top_k

        model = LadderLM(level=level, **kwargs).to(device=device, dtype=dtype)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {n_params/1e6:.1f}M")

        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

        # Warmup
        for _ in range(warmup_steps):
            x = torch.randint(0, 256, (batch_size, chunk_size), device=device)
            loss = model(x, return_loss=True)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.synchronize()

        # Measure
        start_time = time.perf_counter()
        total_tokens = 0

        for step in range(measure_steps):
            x = torch.randint(0, 256, (batch_size, chunk_size), device=device)
            loss = model(x, return_loss=True)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_tokens += batch_size * chunk_size

            if step == measure_steps - 1:
                last_loss = loss.item()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        tok_per_sec = total_tokens / elapsed
        print(f"  Loss: {last_loss:.4f}")
        print(f"  Throughput: {tok_per_sec/1000:.1f}K tok/s")
        print(f"  Time: {elapsed:.2f}s for {measure_steps} steps")

        # Memory
        mem_used = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak memory: {mem_used:.2f} GB")

        del model, optimizer
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 70)
print("Test complete")
