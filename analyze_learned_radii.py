#!/usr/bin/env python3
"""Analyze learned radii benchmark results."""

import re
import os

def parse_log(filepath):
    losses = []
    throughputs = []
    if not os.path.exists(filepath):
        return [], []
    with open(filepath, 'r') as f:
        for line in f:
            match = re.search(r'step\s+\d+\s+\|\s+loss\s+([\d.]+)\s+\|.*tok/s\s+([\d.]+)', line)
            if match:
                losses.append(float(match.group(1)))
                throughputs.append(float(match.group(2)))
    return losses, throughputs

results_dir = "benchmark_results/learned_radii"

models = [
    ("E1_auto", "E1 (auto=spectral_norm)"),
    ("E57_scalar", "E57 (scalar learned)"),
    ("E58_perdim", "E58 (per-dim learned)"),
    ("E33_auto", "E33 (auto=spectral_norm)"),
    ("E42_auto", "E42 (auto=spectral_norm)"),
    ("E44_auto", "E44 (auto=none, diagonal)"),
    ("Mamba2", "Mamba2 baseline"),
]

print("=" * 80)
print("Learned Radii Benchmark Results (100M params, 25 layers, 10 min)")
print("=" * 80)
print()
print(f"{'Model':<28} {'Loss (L20)':>12} {'Tok/s (L20)':>14} {'Steps':>8}")
print("-" * 80)

results = []
for name, desc in models:
    log_path = os.path.join(results_dir, f"{name}.log")
    losses, toks = parse_log(log_path)
    if not losses:
        print(f"{desc:<28} - NO DATA")
        continue

    n = min(20, len(losses))
    avg_loss = sum(losses[-n:]) / n
    avg_toks = sum(toks[-n:]) / n
    results.append((name, desc, avg_loss, avg_toks, len(losses)))
    print(f"{desc:<28} {avg_loss:>12.4f} {avg_toks:>14,.0f} {len(losses)*50:>8}")

print()
print("=" * 80)
print("Key Comparisons:")
print("-" * 80)

# Find results by name
def get_loss(name):
    for r in results:
        if r[0] == name:
            return r[2]
    return None

e1 = get_loss("E1_auto")
e57 = get_loss("E57_scalar")
e58 = get_loss("E58_perdim")
mamba = get_loss("Mamba2")
e44 = get_loss("E44_auto")

if e1 and e57:
    print(f"E1 fixed vs E57 scalar learned: {e1:.4f} vs {e57:.4f} (delta: {e57 - e1:+.4f})")
if e1 and e58:
    print(f"E1 fixed vs E58 per-dim learned: {e1:.4f} vs {e58:.4f} (delta: {e58 - e1:+.4f})")
if e57 and e58:
    print(f"E57 scalar vs E58 per-dim: {e57:.4f} vs {e58:.4f} (delta: {e58 - e57:+.4f})")
if mamba:
    print(f"\nMamba2 baseline: {mamba:.4f}")
if e44:
    print(f"E44 (diagonal, fastest): {e44:.4f}")

print()
print("Ranking by loss (lower is better):")
print("-" * 80)
for name, desc, loss, toks, steps in sorted(results, key=lambda x: x[2]):
    print(f"{desc:<28} {loss:.4f}  ({toks:,.0f} tok/s)")
