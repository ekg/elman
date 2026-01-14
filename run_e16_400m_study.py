#!/usr/bin/env python3
"""
E16 400M Parameter Study - Varying State Expansion

Compare E16 with different state_expansion values against:
- E1 (from prior runs: best=1.54 loss @ exp2.5)
- Mamba2 (from prior runs: 1.50 loss @ 23.4K tok/s)
- minGRU, minLSTM (new runs)

All at ~400M parameters, 10 minutes training, same data.
"""

import torch
import torch.nn.functional as F
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from elman.models import LadderLM, create_ladder_model
from elman.models.min_rnn_baseline import create_mingru_model, create_minlstm_model

# Training settings
TRAIN_MINUTES = 10
BATCH_SIZE = 32
CHUNK_SIZE = 512
LR = 1e-4
DATA_PATH = "data/pile.txt"
LOG_EVERY = 300

def load_data():
    """Memory-map the data file."""
    import mmap
    import numpy as np

    with open(DATA_PATH, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    return mm, len(mm)

def get_batch(mm, data_len, batch_size, chunk_size, device):
    """Sample random chunks from data."""
    import numpy as np

    max_start = data_len - chunk_size - 2
    starts = np.random.randint(0, max_start, size=batch_size)

    chunks = []
    for start in starts:
        chunk = np.frombuffer(mm[start:start + chunk_size + 1], dtype=np.uint8).copy()
        chunks.append(torch.from_numpy(chunk.astype(np.int64)))

    return torch.stack(chunks).to(device)

def train_model(model, name, train_minutes):
    """Train model for specified minutes, return final loss and throughput."""
    device = 'cuda'
    model = model.to(device).bfloat16()

    params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"{name}: {params:,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    mm, data_len = load_data()

    # Warmup
    for _ in range(3):
        batch = get_batch(mm, data_len, BATCH_SIZE, CHUNK_SIZE, device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = model(batch, return_loss=True)
            if isinstance(loss, tuple):
                loss = loss[0]
        loss.backward()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    print(f"Memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

    # Training
    train_end = time.time() + train_minutes * 60
    step = 0
    total_tokens = 0
    total_loss = 0
    start_time = time.time()

    losses = []

    while time.time() < train_end:
        batch = get_batch(mm, data_len, BATCH_SIZE, CHUNK_SIZE, device)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = model(batch, return_loss=True)
            if isinstance(loss, tuple):
                loss = loss[0]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        step += 1
        total_tokens += BATCH_SIZE * CHUNK_SIZE
        losses.append(loss.item())

        if step % LOG_EVERY == 0:
            elapsed = time.time() - start_time
            tok_per_sec = total_tokens / elapsed
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            print(f"Step {step}: loss={avg_loss:.4f}, {tok_per_sec/1000:.1f}K tok/s")

    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    final_tok_per_sec = total_tokens / elapsed
    final_loss = sum(losses[-100:]) / len(losses[-100:])

    print(f"FINAL: loss={final_loss:.4f}, tok/s={final_tok_per_sec/1000:.1f}K, steps={step}")

    return final_loss, final_tok_per_sec, params

def create_e16_model(target_params, state_expansion):
    """Create E16 model with specified state expansion."""
    # E16 uses expansion and state_expansion
    # Need to find dim/depth to hit target params

    # Start with a base config and adjust
    from elman.models.ladder_lm import LadderLM
    from elman.models.diagonal_state_elman import DiagonalStateElman

    # Parse target
    if isinstance(target_params, str):
        if target_params.endswith('m'):
            target = int(float(target_params[:-1]) * 1e6)
        else:
            target = int(target_params)
    else:
        target = target_params

    # For E16: params per layer ≈ 2*dim*d_inner + d_inner*d_state + d_state*d_inner + d_state + 2*d_inner
    # d_inner = dim * expansion, d_state = d_inner * state_expansion
    # Try different dims
    expansion = 1.5

    best_dim, best_depth = 512, 12
    best_diff = float('inf')

    for dim in range(512, 2048, 64):
        for depth in range(4, 48):
            # Estimate params
            d_inner = int(dim * expansion)
            d_state = d_inner * state_expansion

            # Per layer: in_proj (dim -> 2*d_inner) + B (d_inner -> d_state) + C (d_state -> d_inner) + A (d_state) + out_proj (d_inner -> dim)
            per_layer = 2 * dim * d_inner + d_inner * d_state + d_state * d_inner + d_state + d_inner * dim
            # Add layer norm
            per_layer += 2 * dim

            # Base: embedding + output (tied)
            base = 256 * dim

            total = base + depth * per_layer
            diff = abs(total - target)

            if diff < best_diff:
                best_diff = diff
                best_dim = dim
                best_depth = depth

    model = LadderLM(
        vocab_size=256,
        dim=best_dim,
        depth=best_depth,
        level=16,
        expansion=expansion,
        state_expansion=state_expansion,
    )

    actual = sum(p.numel() for p in model.parameters())
    d_inner = int(best_dim * expansion)
    d_state = d_inner * state_expansion
    print(f"E16 state_exp={state_expansion}: dim={best_dim}, depth={best_depth}, d_state={d_state}, params={actual:,}")

    return model

def main():
    print("="*60)
    print("E16 400M Parameter Study")
    print("="*60)

    results = []

    # E16 variants with different state expansion
    for state_exp in [2, 4, 8]:
        name = f"E16_state{state_exp}x"
        model = create_e16_model("400m", state_exp)
        loss, tok_s, params = train_model(model, name, TRAIN_MINUTES)
        results.append((name, params, loss, tok_s))
        del model
        torch.cuda.empty_cache()

    # minGRU
    print("\n" + "="*60)
    model = create_mingru_model("400m")
    loss, tok_s, params = train_model(model, "minGRU", TRAIN_MINUTES)
    results.append(("minGRU", params, loss, tok_s))
    del model
    torch.cuda.empty_cache()

    # minLSTM
    print("\n" + "="*60)
    model = create_minlstm_model("400m")
    loss, tok_s, params = train_model(model, "minLSTM", TRAIN_MINUTES)
    results.append(("minLSTM", params, loss, tok_s))
    del model
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY (10 min training, 400M params)")
    print("="*60)
    print(f"{'Model':<20} {'Params':>12} {'Loss':>8} {'Tok/s':>10}")
    print("-"*60)

    # Add prior results
    prior = [
        ("E1 exp2.5 (prior)", 403_018_000, 1.5355, 13900),
        ("Mamba2 (prior)", 402_064_492, 1.5028, 23400),
    ]

    all_results = prior + results
    all_results.sort(key=lambda x: x[2])  # Sort by loss

    for name, params, loss, tok_s in all_results:
        print(f"{name:<20} {params:>12,} {loss:>8.4f} {tok_s/1000:>9.1f}K")

if __name__ == "__main__":
    main()
