#!/usr/bin/env python3
"""Compare spectral norm vs no spectral norm results."""

import re
import os

def parse_log(filepath):
    """Parse training log for loss and throughput."""
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

models = [
    "E1_gated",
    "E43_scalar_decay",
    "E44_diagonal_w",
    "E45_pure_accum",
    "E46_no_in_proj",
    "E48_no_proj",
    "E51_no_self_gate",
    "E52_quadratic",
    "E53_sigmoid",
    "E56_concat",
]

print("=" * 90)
print("Spectral Normalization Comparison (25 layers, ~100M params, 10 min training)")
print("=" * 90)
print()
print(f"{'Model':<20} {'No SpecNorm':>12} {'With SpecNorm':>14} {'Delta':>10} {'Better?':>10}")
print("-" * 90)

results = []
for name in models:
    # No spectral norm (previous run)
    no_sn_path = f"benchmark_results/25layer_100m/{name}.log"
    losses_no, toks_no = parse_log(no_sn_path)

    # With spectral norm
    sn_path = f"benchmark_results/25layer_specnorm/{name}.log"
    losses_sn, toks_sn = parse_log(sn_path)

    if not losses_no or not losses_sn:
        print(f"{name:<20} - MISSING DATA")
        continue

    # Last 20 average
    n_no = min(20, len(losses_no))
    n_sn = min(20, len(losses_sn))

    loss_no = sum(losses_no[-n_no:]) / n_no
    loss_sn = sum(losses_sn[-n_sn:]) / n_sn

    toks_no_avg = sum(toks_no[-n_no:]) / n_no
    toks_sn_avg = sum(toks_sn[-n_sn:]) / n_sn

    delta = loss_sn - loss_no
    better = "SpecNorm" if delta < -0.01 else ("NoSpec" if delta > 0.01 else "Same")

    results.append((name, loss_no, loss_sn, delta, toks_no_avg, toks_sn_avg, better))
    print(f"{name:<20} {loss_no:>12.4f} {loss_sn:>14.4f} {delta:>+10.4f} {better:>10}")

print()
print("=" * 90)
print("Summary:")
print("-" * 90)

sn_wins = sum(1 for r in results if r[6] == "SpecNorm")
no_wins = sum(1 for r in results if r[6] == "NoSpec")
same = sum(1 for r in results if r[6] == "Same")

print(f"Spectral Norm better: {sn_wins}")
print(f"No Spectral Norm better: {no_wins}")
print(f"Essentially same: {same}")

# Average delta
avg_delta = sum(r[3] for r in results) / len(results) if results else 0
print(f"\nAverage loss delta (SpecNorm - NoSpec): {avg_delta:+.4f}")
print("(Negative = SpecNorm is better)")

print()
print("Throughput comparison (tok/s):")
print("-" * 90)
for name, loss_no, loss_sn, delta, toks_no, toks_sn, better in results:
    toks_delta = (toks_sn - toks_no) / toks_no * 100
    print(f"{name:<20} NoSpec: {toks_no:>8,.0f}  SpecNorm: {toks_sn:>8,.0f}  ({toks_delta:+.1f}%)")
