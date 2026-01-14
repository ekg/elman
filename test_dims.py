#!/usr/bin/env python3
"""Quick test of parameter counts at specific dimensions."""

import torch
import sys
sys.path.insert(0, '/home/erikg/elman')

from elman.models import LadderLM

depth = 25
target = 100_000_000

# Test specific dimensions for each model
test_configs = [
    # Model, level, [dims to test]
    ("E1", 1, [880, 900, 920, 940]),
    ("E43", 43, [1400, 1410, 1420]),
    ("E44", 44, [1400, 1410, 1420]),
    ("E45", 45, [1400, 1410, 1420]),
    ("E46", 46, [1400, 1410, 1420, 1440]),
    ("E48", 48, [1980, 1990, 2000]),
    ("E51", 51, [1140, 1150, 1160, 1170]),
    ("E52", 52, [1140, 1150, 1160, 1170]),
    ("E53", 53, [1140, 1150, 1160, 1170]),
    ("E56", 56, [940, 950, 960, 970]),
]

print(f"Testing dimensions for 100M params at {depth} layers")
print(f"{'Model':<8} {'Level':<6} {'Dim':<6} {'Total Params':>14} {'Per Layer':>12} {'Diff%':>8}")
print("-" * 65)

for name, level, dims in test_configs:
    for dim in dims:
        try:
            model = LadderLM(vocab_size=256, dim=dim, depth=depth, level=level)
            total = sum(p.numel() for p in model.parameters())
            per_layer = total / depth
            diff_pct = (total - target) / target * 100
            marker = " <--" if abs(diff_pct) < 1.5 else ""
            print(f"{name:<8} {level:<6} {dim:<6} {total:>14,} {per_layer:>12,.0f} {diff_pct:>7.1f}%{marker}")
        except Exception as e:
            print(f"{name:<8} {level:<6} {dim:<6} ERROR: {e}")
    print()
