#!/usr/bin/env python3
"""Analyze 25-layer benchmark results."""

import re
import os

results_dir = "benchmark_results/25layer_100m"

# Model configurations
configs = {
    "E1_gated": (1, 900, 25, "Gated Elman"),
    "E43_scalar_decay": (43, 1410, 25, "Scalar decay"),
    "E44_diagonal_w": (44, 1410, 25, "Diagonal W"),
    "E45_pure_accum": (45, 1410, 25, "Pure accumulation"),
    "E46_no_in_proj": (46, 1410, 25, "No input proj"),
    "E48_no_proj": (48, 1990, 25, "No projections"),
    "E51_no_self_gate": (51, 1150, 25, "No self-gate"),
    "E52_quadratic": (52, 1150, 25, "Quadratic gate"),
    "E53_sigmoid": (53, 1150, 25, "Sigmoid gate"),
    "E56_concat": (56, 890, 25, "Concat Elman"),
    "Mamba2": ("mamba2", 800, 25, "Mamba2 baseline"),
}

def parse_log(filepath):
    """Parse training log for loss and throughput."""
    losses = []
    throughputs = []

    with open(filepath, 'r') as f:
        for line in f:
            # Parse: step   1000 | loss 1.8457 | lr 9.99e-05 | grad 1.44 | tok/s 14071
            match = re.search(r'step\s+\d+\s+\|\s+loss\s+([\d.]+)\s+\|.*tok/s\s+([\d.]+)', line)
            if match:
                losses.append(float(match.group(1)))
                throughputs.append(float(match.group(2)))

    return losses, throughputs

print("=" * 100)
print("25-Layer Benchmark Results (100M params, 10 min training, depth=25)")
print("=" * 100)
print()
print(f"{'Model':<12} {'Level':<8} {'Dim':<6} {'Depth':<6} {'Params':>12} {'Loss (L20)':>12} {'Tok/s (L20)':>14} {'Description':<20}")
print("-" * 100)

results = []
for name, (level, dim, depth, desc) in sorted(configs.items()):
    log_path = os.path.join(results_dir, f"{name}.log")
    if not os.path.exists(log_path):
        print(f"{name:<12} - LOG NOT FOUND")
        continue

    losses, throughputs = parse_log(log_path)
    if not losses:
        print(f"{name:<12} - NO DATA IN LOG")
        continue

    # Last 20 average (or all if fewer)
    n = min(20, len(losses))
    avg_loss = sum(losses[-n:]) / n
    avg_toks = sum(throughputs[-n:]) / n

    # Get actual param count from log
    with open(log_path, 'r') as f:
        content = f.read()
        match = re.search(r'(\d+(?:,\d+)*)\s+parameters', content)
        if match:
            params = int(match.group(1).replace(',', ''))
        else:
            params = 0

    results.append((name, level, dim, depth, params, avg_loss, avg_toks, desc))
    print(f"{name:<12} {str(level):<8} {dim:<6} {depth:<6} {params:>12,} {avg_loss:>12.4f} {avg_toks:>14,.0f} {desc:<20}")

print()
print("=" * 100)

# Sort by loss
print("\nRanked by Loss (lower is better):")
print("-" * 60)
for name, level, dim, depth, params, loss, toks, desc in sorted(results, key=lambda x: x[5]):
    print(f"{name:<20} {loss:.4f}  ({toks:,.0f} tok/s)")

print()
print("Ranked by Throughput (higher is better):")
print("-" * 60)
for name, level, dim, depth, params, loss, toks, desc in sorted(results, key=lambda x: -x[6]):
    print(f"{name:<20} {toks:,.0f} tok/s  (loss {loss:.4f})")
