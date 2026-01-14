#!/usr/bin/env python3
"""Analyze spectral radius benchmark results."""

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

results_dir = "benchmark_results/specradius_100m"

models = [
    ("E1_specnorm", "E1 + SpecNorm (baseline)"),
    ("E57_learned", "E57 (learned radius)"),
    ("E33_specnorm", "E33 + SpecNorm"),
    ("E33_no_specnorm", "E33 (no SpecNorm)"),
    ("E42_specnorm", "E42 + SpecNorm"),
    ("E42_no_specnorm", "E42 (no SpecNorm)"),
]

print("=" * 80)
print("Spectral Radius Benchmark Results (100M params, 25 layers, 10 min)")
print("=" * 80)
print()
print(f"{'Model':<28} {'Loss (L20)':>12} {'Tok/s (L20)':>14}")
print("-" * 80)

for name, desc in models:
    log_path = os.path.join(results_dir, f"{name}.log")
    losses, toks = parse_log(log_path)
    if not losses:
        print(f"{desc:<28} - NO DATA")
        continue

    n = min(20, len(losses))
    avg_loss = sum(losses[-n:]) / n
    avg_toks = sum(toks[-n:]) / n
    print(f"{desc:<28} {avg_loss:>12.4f} {avg_toks:>14,.0f}")

print()
print("=" * 80)
print("Key Comparisons:")
print("-" * 80)

# E1 vs E57
e1_loss, e1_toks = parse_log(os.path.join(results_dir, "E1_specnorm.log"))
e57_loss, e57_toks = parse_log(os.path.join(results_dir, "E57_learned.log"))
if e1_loss and e57_loss:
    e1_l = sum(e1_loss[-20:]) / 20
    e57_l = sum(e57_loss[-20:]) / 20
    print(f"E1 fixed SpecNorm vs E57 learned: {e1_l:.4f} vs {e57_l:.4f} (delta: {e57_l - e1_l:+.4f})")

# E33 comparison
e33_sn, _ = parse_log(os.path.join(results_dir, "E33_specnorm.log"))
e33_no, _ = parse_log(os.path.join(results_dir, "E33_no_specnorm.log"))
if e33_sn and e33_no:
    sn_l = sum(e33_sn[-20:]) / 20
    no_l = sum(e33_no[-20:]) / 20
    print(f"E33 SpecNorm vs No SpecNorm: {sn_l:.4f} vs {no_l:.4f} (delta: {sn_l - no_l:+.4f})")

# E42 comparison
e42_sn, _ = parse_log(os.path.join(results_dir, "E42_specnorm.log"))
e42_no, _ = parse_log(os.path.join(results_dir, "E42_no_specnorm.log"))
if e42_sn and e42_no:
    sn_l = sum(e42_sn[-20:]) / 20
    no_l = sum(e42_no[-20:]) / 20
    print(f"E42 SpecNorm vs No SpecNorm: {sn_l:.4f} vs {no_l:.4f} (delta: {sn_l - no_l:+.4f})")
