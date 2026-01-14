#!/usr/bin/env python3
"""
E1 vs E16 Comparison: Dense Elman vs Diagonal State-Expanded Elman

E1: h = tanh(W_h @ h + W_x @ x)  - Dense W_h (d_inner × d_inner)
E16: h = tanh(A ⊙ h + B @ x)     - Diagonal A, state expansion d_state > d_inner

Key difference: E16 trades off dense W_h for:
1. Diagonal A (O(n) instead of O(n²) recurrence params)
2. Larger state (d_state = 2 * d_inner by default)
"""

import subprocess
import sys
import time

# Training parameters
TRAIN_MINUTES = 10
PARAMS = "50m"
BATCH_SIZE = 48
DATA_PATH = "data/pile.txt"

configs = [
    # E1 baseline (best known config)
    {"level": 1, "name": "E1 (Dense)", "expansion": 1.5},

    # E16 variants
    {"level": 16, "name": "E16 (Diag State)", "expansion": 1.5, "state_expansion": 2},
    {"level": 16, "name": "E16 (Diag State 3x)", "expansion": 1.5, "state_expansion": 3},
]

def run_config(cfg):
    """Run training with given config."""
    cmd = [
        "python", "train.py",
        "--data", DATA_PATH,
        "--params", PARAMS,
        "--level", str(cfg["level"]),
        "--batch_size", str(BATCH_SIZE),
        "--train_minutes", str(TRAIN_MINUTES),
        "--expansion", str(cfg.get("expansion", 1.0)),
        "--bf16",
    ]

    # Add state_expansion for E16
    if cfg["level"] == 16:
        cmd.extend(["--state_expansion", str(cfg.get("state_expansion", 2))])

    print(f"\n{'='*70}")
    print(f"Running: {cfg['name']}")
    print(f"Command: {' '.join(cmd)}")
    print('='*70)

    start = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start

    return result.returncode == 0, elapsed


if __name__ == "__main__":
    print("E1 vs E16 Comparison")
    print("="*70)
    print(f"Parameters: {PARAMS}, Training: {TRAIN_MINUTES} minutes each")
    print(f"Batch size: {BATCH_SIZE}")

    results = []
    for cfg in configs:
        success, elapsed = run_config(cfg)
        results.append((cfg['name'], success, elapsed))

        if not success:
            print(f"WARNING: {cfg['name']} failed!")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, success, elapsed in results:
        status = "✓" if success else "✗"
        print(f"{status} {name}: {elapsed/60:.1f} min")
