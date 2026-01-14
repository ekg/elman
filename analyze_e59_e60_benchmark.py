#!/usr/bin/env python3
"""Analyze E59/E60 benchmark results."""

import re
import os

def parse_log(filepath):
    """Parse training log for loss and throughput."""
    losses = []
    throughputs = []
    params = None

    if not os.path.exists(filepath):
        return None, [], []

    with open(filepath, 'r') as f:
        for line in f:
            # Get param count
            if 'parameters' in line.lower():
                match = re.search(r'([\d,]+)\s*parameters', line)
                if match:
                    params = int(match.group(1).replace(',', ''))

            # Get training metrics
            match = re.search(r'step\s+\d+\s+\|\s+loss\s+([\d.]+)\s+\|.*tok/s\s+([\d.]+)', line)
            if match:
                losses.append(float(match.group(1)))
                throughputs.append(float(match.group(2)))

    return params, losses, throughputs

results_dir = "benchmark_results/e59_e60_comparison"

models = [
    ("1", "E1 (gated elman)"),
    ("33", "E33 (multi-gate)"),
    ("42", "E42 (linear tied)"),
    ("51", "E51 (cubic gate)"),
    ("52", "E52 (squared gate)"),
    ("53", "E53 (abs gate)"),
    ("56", "E56 (softsign)"),
    ("57", "E57 (scalar learned)"),
    ("58", "E58 (per-dim learned)"),
    ("59", "E59 (highway)"),
    ("60", "E60 (residual nonlin)"),
    ("44", "E44 (diagonal)"),
    ("mamba2", "Mamba2"),
]

print("=" * 90)
print("E59/E60 Benchmark Results (100M params, 25 layers, 10 min, seed=42)")
print("=" * 90)
print()
print(f"{'Model':<24} {'Params':>12} {'Loss (L100)':>12} {'Tok/s (L100)':>14} {'Steps':>8}")
print("-" * 90)

results = []
for level, desc in models:
    log_path = os.path.join(results_dir, f"{level}.log")
    params, losses, toks = parse_log(log_path)

    if not losses:
        print(f"{desc:<24} - NO DATA")
        continue

    # Average last 100 steps (or all if fewer)
    n = min(100, len(losses))
    avg_loss = sum(losses[-n:]) / n
    avg_toks = sum(toks[-n:]) / n
    steps = len(losses) * 50  # Assuming log every 50 steps

    results.append((level, desc, params, avg_loss, avg_toks, steps))
    param_str = f"{params/1e6:.1f}M" if params else "?"
    print(f"{desc:<24} {param_str:>12} {avg_loss:>12.4f} {avg_toks:>14,.0f} {steps:>8}")

print()
print("=" * 90)
print("Rankings")
print("-" * 90)

if results:
    print("\nBy Loss (lower is better):")
    for i, (level, desc, params, loss, toks, steps) in enumerate(sorted(results, key=lambda x: x[3]), 1):
        print(f"  {i}. {desc:<24} {loss:.4f}  ({toks:,.0f} tok/s)")

    print("\nBy Throughput (higher is better):")
    for i, (level, desc, params, loss, toks, steps) in enumerate(sorted(results, key=lambda x: -x[4]), 1):
        print(f"  {i}. {desc:<24} {toks:,.0f} tok/s  (loss {loss:.4f})")

print()
print("=" * 90)
print("Key Comparisons")
print("-" * 90)

def get_result(level):
    for r in results:
        if r[0] == level:
            return r
    return None

e1 = get_result("1")
e59 = get_result("59")
e60 = get_result("60")
mamba = get_result("mamba2")
e44 = get_result("44")

if e1 and e59:
    delta = e59[3] - e1[3]
    speed_ratio = e59[4] / e1[4]
    print(f"E59 vs E1: loss {e59[3]:.4f} vs {e1[3]:.4f} ({delta:+.4f}), speed {speed_ratio:.2f}x")

if e1 and e60:
    delta = e60[3] - e1[3]
    speed_ratio = e60[4] / e1[4]
    print(f"E60 vs E1: loss {e60[3]:.4f} vs {e1[3]:.4f} ({delta:+.4f}), speed {speed_ratio:.2f}x")

if e59 and e60:
    delta = e60[3] - e59[3]
    print(f"E60 vs E59: loss {e60[3]:.4f} vs {e59[3]:.4f} ({delta:+.4f})")

if mamba:
    print(f"\nMamba2 baseline: loss {mamba[3]:.4f}, {mamba[4]:,.0f} tok/s")

if e44:
    print(f"E44 (fastest diagonal): loss {e44[3]:.4f}, {e44[4]:,.0f} tok/s")
