#!/usr/bin/env python3
"""Calculate dimensions for 100M params at 25 layers for each model."""

import torch
import sys
sys.path.insert(0, '/home/erikg/elman')

from elman.models import LadderLM

def find_dim_for_target(level, depth, target_params, min_dim=256, max_dim=4096):
    """Binary search for dimension that gives target params."""
    best_dim = min_dim
    best_diff = float('inf')

    for dim in range(min_dim, max_dim, 16):  # Step by 16 for speed
        try:
            model = LadderLM(vocab_size=256, dim=dim, depth=depth, level=level)
            total = sum(p.numel() for p in model.parameters())
            diff = abs(total - target_params)
            if diff < best_diff:
                best_diff = diff
                best_dim = dim
            if total > target_params * 1.1:  # Gone too far
                break
        except Exception as e:
            continue

    # Fine tune
    for dim in range(max(min_dim, best_dim - 32), best_dim + 48, 2):
        try:
            model = LadderLM(vocab_size=256, dim=dim, depth=depth, level=level)
            total = sum(p.numel() for p in model.parameters())
            diff = abs(total - target_params)
            if diff < best_diff:
                best_diff = diff
                best_dim = dim
        except Exception as e:
            continue

    return best_dim

target = 100_000_000  # 100M params
depth = 25

configs = [
    ("E1", 1),
    ("E43", 43),
    ("E44", 44),
    ("E45", 45),
    ("E46", 46),
    ("E48", 48),
    ("E51", 51),
    ("E52", 52),
    ("E53", 53),
    ("E56", 56),
]

print("Finding dimensions for 100M params at 25 layers...")
print()
print(f"{'Model':<8} {'Level':<6} {'Dim':<6} {'Depth':<6} {'Total Params':>14} {'Per Layer':>12}")
print("-" * 60)

results = []
for name, level in configs:
    dim = find_dim_for_target(level, depth, target)
    model = LadderLM(vocab_size=256, dim=dim, depth=depth, level=level)
    total = sum(p.numel() for p in model.parameters())
    per_layer = total / depth
    print(f"{name:<8} {level:<6} {dim:<6} {depth:<6} {total:>14,} {per_layer:>12,.0f}")
    results.append((name, level, dim, depth, total))

# Print for bash script
print()
print("For benchmark script:")
for name, level, dim, depth, total in results:
    print(f"# {name}: dim={dim}, depth={depth}, params={total:,}")
