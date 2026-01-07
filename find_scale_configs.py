#!/usr/bin/env python3
"""
Find optimal configs for E1, E9, Mamba2 at 50M, 100M, 200M, 400M params.
Then test max batch sizes.
"""
import sys; sys.path.insert(0, '.')
import torch
from elman.models import LadderLM, create_mamba2_model

SCALES = [50_000_000, 100_000_000, 200_000_000, 400_000_000]

print("="*70)
print("FINDING OPTIMAL CONFIGS")
print("="*70)

# E1 configs (level=1)
print("\n--- E1 (Mamba-Gated Elman) ---")
e1_configs = {}
for target in SCALES:
    best = None
    best_diff = float('inf')
    # E1 prefers wide+shallow based on previous experiments
    for dim in [768, 1024, 1280, 1536, 1792, 2048, 2304]:
        for depth in [4, 6, 8, 10, 12]:
            try:
                m = LadderLM(vocab_size=256, dim=dim, depth=depth, level=1)
                params = m.get_num_params()
                diff = abs(params - target)
                if diff < best_diff:
                    best_diff = diff
                    best = (dim, depth, params)
                del m
            except: pass
    if best:
        e1_configs[target] = best
        print(f"  {target//1_000_000}M: dim={best[0]}, depth={best[1]}, actual={best[2]:,}")

# E9 configs (level=9, balanced core=mem)
print("\n--- E9 (Hybrid Elman, balanced) ---")
e9_configs = {}
for target in SCALES:
    best = None
    best_diff = float('inf')
    # E9 balanced: expansion=2.0, core_ratio=0.5
    for dim in [1024, 1280, 1440, 1536, 1792, 2048, 2304, 2560]:
        for depth in [3, 4, 5, 6]:
            try:
                m = LadderLM(vocab_size=256, dim=dim, depth=depth, level=9,
                            expansion=2.0, core_ratio=0.5)
                params = m.get_num_params()
                diff = abs(params - target)
                if diff < best_diff:
                    best_diff = diff
                    best = (dim, depth, params)
                del m
            except: pass
    if best:
        e9_configs[target] = best
        print(f"  {target//1_000_000}M: dim={best[0]}, depth={best[1]}, actual={best[2]:,}")

# Mamba2 configs
print("\n--- Mamba2 ---")
mamba2_configs = {}
for target in SCALES:
    try:
        target_str = f"{target//1_000_000}m"
        m = create_mamba2_model(target_params=target_str, vocab_size=256)
        params = m.get_num_params()
        mamba2_configs[target] = params
        print(f"  {target//1_000_000}M: actual={params:,}")
        del m
    except Exception as e:
        print(f"  {target//1_000_000}M: ERROR - {e}")

print("\n" + "="*70)
print("TESTING MAX BATCH SIZES (GPU 0)")
print("="*70)

def test_batch_size(model_fn, name, start_batch=256):
    """Binary search for max batch size."""
    torch.cuda.empty_cache()
    low, high = 32, start_batch * 4
    best = low

    while low <= high:
        mid = (low + high) // 2
        torch.cuda.empty_cache()
        try:
            model = model_fn().cuda().bfloat16()
            x = torch.randint(0, 256, (mid, 513), device='cuda')
            loss = model(x, return_loss=True)
            loss.backward()
            del model, x, loss
            torch.cuda.empty_cache()
            best = mid
            low = mid + 1
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                high = mid - 1
                torch.cuda.empty_cache()
            else:
                raise
    return best

# Test each config
print("\nE1 batch sizes:")
for target, (dim, depth, params) in e1_configs.items():
    bs = test_batch_size(
        lambda d=dim, dp=depth: LadderLM(vocab_size=256, dim=d, depth=dp, level=1),
        f"E1-{target//1_000_000}M"
    )
    print(f"  {target//1_000_000}M (d{dim}x{depth}): max_batch={bs}")

print("\nE9 batch sizes:")
for target, (dim, depth, params) in e9_configs.items():
    bs = test_batch_size(
        lambda d=dim, dp=depth: LadderLM(vocab_size=256, dim=d, depth=dp, level=9, expansion=2.0, core_ratio=0.5),
        f"E9-{target//1_000_000}M"
    )
    print(f"  {target//1_000_000}M (d{dim}x{depth}): max_batch={bs}")

print("\nMamba2 batch sizes:")
for target in SCALES:
    target_str = f"{target//1_000_000}m"
    bs = test_batch_size(
        lambda t=target_str: create_mamba2_model(target_params=t, vocab_size=256),
        f"Mamba2-{target//1_000_000}M"
    )
    print(f"  {target//1_000_000}M: max_batch={bs}")
