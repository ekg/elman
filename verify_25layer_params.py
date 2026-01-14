#!/usr/bin/env python3
"""Verify parameter counts for 25-layer benchmark configs."""

import torch
import sys
sys.path.insert(0, '/home/erikg/elman')

from elman.models import LadderLM

# Target: 4M params/layer, 25 layers = 100M total
configs = [
    ("E1", 1, 706, 25),
    ("E43", 43, 998, 25),
    ("E44", 44, 998, 25),
    ("E45", 45, 998, 25),
    ("E46", 46, 1410, 25),
    ("E48", 48, 1990, 25),
    ("E51", 51, 706, 25),
    ("E52", 52, 706, 25),
    ("E53", 53, 706, 25),
    ("E56", 56, 577, 25),
]

print(f"{'Model':<8} {'Level':<6} {'Dim':<6} {'Depth':<6} {'Total Params':>14} {'Per Layer':>12}")
print("-" * 60)

for name, level, dim, depth in configs:
    try:
        model = LadderLM(vocab_size=256, dim=dim, depth=depth, level=level)
        total = sum(p.numel() for p in model.parameters())
        per_layer = total / depth
        status = "OK" if 95_000_000 <= total <= 105_000_000 else "ADJUST"
        print(f"{name:<8} {level:<6} {dim:<6} {depth:<6} {total:>14,} {per_layer:>12,.0f}  {status}")
    except Exception as e:
        print(f"{name:<8} {level:<6} {dim:<6} {depth:<6} ERROR: {e}")

# Also check Mamba2
print()
print("Mamba2 baseline:")
try:
    from elman.models.mamba2 import Mamba2LM
    mamba = Mamba2LM(vocab_size=256, dim=814, depth=25)
    total = sum(p.numel() for p in mamba.parameters())
    per_layer = total / 25
    print(f"Mamba2   -      814    25     {total:>14,} {per_layer:>12,.0f}")
except Exception as e:
    print(f"Mamba2 ERROR: {e}")
