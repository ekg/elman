#!/usr/bin/env python3
"""
100M parameter benchmark: E61, E62, E63, E63m, E56, E42, Mamba2, E1, GRU, LSTM
30-minute training with expansion=2.0, last-100 average loss comparison.

Features:
- Dynamic GPU allocation (auto-detects available GPUs)
- Support for new E61-E63m gated delta variants
- Targets ~20 layers for all models
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path

# Target: 100M params with expansion=2.0
TARGET_PARAMS = 100_000_000
EXPANSION = 2.0
TRAIN_MINUTES = 10
TARGET_DEPTH = 20  # Target ~20 layers as requested
OUTPUT_DIR = Path("benchmark_results/100m_10min")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_available_gpus():
    """Detect available GPUs using nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(',')
                idx = int(parts[0].strip())
                free_mem = int(parts[1].strip())
                gpus.append((idx, free_mem))
        # Sort by free memory descending
        gpus.sort(key=lambda x: -x[1])
        return [g[0] for g in gpus]
    except Exception as e:
        print(f"Warning: Could not detect GPUs: {e}")
        return list(range(8))  # Default fallback

def align_to_128(x):
    """Round x to nearest multiple of 128."""
    return ((x + 63) // 128) * 128


def count_e61_params(dim, depth, vocab_size=256, expansion=2.0):
    """Count E61 (DecayGated) parameters.

    E61: α*h + (1-α)*Wx
    - W_alpha [d_inner, d_inner], b_alpha [d_inner]
    - W_v [d_inner, d_inner], b_v [d_inner]
    """
    d_inner = int(dim * expansion)

    per_layer = (
        dim * d_inner +          # in_proj
        d_inner * d_inner +      # W_alpha
        d_inner +                # b_alpha
        d_inner * d_inner +      # W_v
        d_inner +                # b_v
        d_inner * dim +          # out_proj
        2 * dim                  # LayerNorm
    )

    total = (
        vocab_size * dim +       # embedding
        per_layer * depth +      # layers
        2 * dim                  # final norm
    )
    return total


def count_e62_params(dim, depth, vocab_size=256, expansion=2.0):
    """Count E62 (SelectiveWrite) parameters.

    E62: (1-k)*h + k*v
    - W_k [d_inner, d_inner], b_k [d_inner]
    - W_v [d_inner, d_inner], b_v [d_inner]
    """
    d_inner = int(dim * expansion)

    per_layer = (
        dim * d_inner +          # in_proj
        d_inner * d_inner +      # W_k
        d_inner +                # b_k
        d_inner * d_inner +      # W_v
        d_inner +                # b_v
        d_inner * dim +          # out_proj
        2 * dim                  # LayerNorm
    )

    total = (
        vocab_size * dim +       # embedding
        per_layer * depth +      # layers
        2 * dim                  # final norm
    )
    return total


def count_e63_params(dim, depth, vocab_size=256, expansion=2.0):
    """Count E63 (NonlinearDelta) parameters.

    E63: α*h + (1-α)*tanh(W_h@h + W_x@x + b)
    - W_alpha [d_inner, d_inner], b_alpha [d_inner]
    - W_h [d_inner, d_inner] (h-dependent!)
    - W_x [d_inner, d_inner]
    - b [d_inner]
    """
    d_inner = int(dim * expansion)

    per_layer = (
        dim * d_inner +          # in_proj
        d_inner * d_inner +      # W_alpha
        d_inner +                # b_alpha
        d_inner * d_inner +      # W_h (nonlinear path!)
        d_inner * d_inner +      # W_x
        d_inner +                # b
        d_inner * dim +          # out_proj
        2 * dim                  # LayerNorm
    )

    total = (
        vocab_size * dim +       # embedding
        per_layer * depth +      # layers
        2 * dim                  # final norm
    )
    return total


def count_e63m_params(dim, depth, vocab_size=256, expansion=2.0, n_slots=64):
    """Count E63m (MatrixNonlinear) parameters.

    E63m: Matrix state S [n_slots, d_inner] with nonlinear retrieval
    - W_k, W_q [d_inner, d_inner]
    - W_x [n_slots, d_inner]
    - W_r [n_slots, n_slots]
    - W_alpha [n_slots, d_inner], b_alpha [n_slots]
    - b [n_slots]
    """
    d_inner = int(dim * expansion)

    per_layer = (
        dim * d_inner +          # in_proj
        d_inner * d_inner +      # W_k
        d_inner * d_inner +      # W_q
        n_slots * d_inner +      # W_x
        n_slots * n_slots +      # W_r
        n_slots * d_inner +      # W_alpha
        n_slots +                # b_alpha
        n_slots +                # b
        n_slots * dim +          # out_proj
        2 * dim                  # LayerNorm
    )

    total = (
        vocab_size * dim +       # embedding
        per_layer * depth +      # layers
        2 * dim                  # final norm
    )
    return total


def count_e1_params(dim, depth, vocab_size=256, expansion=2.0):
    """Count E1 (MambaGatedElman) parameters."""
    d_inner = int(dim * expansion)

    # Per layer:
    # - in_proj: dim -> 2 * d_inner (for x, z split)
    # - W_h: d_inner x d_inner
    # - b_h: d_inner
    # - W_gate: d_inner -> d_inner
    # - b_gate: d_inner
    # - out_proj: d_inner -> dim
    # - norm: 2 * dim (gamma, beta)

    per_layer = (
        dim * 2 * d_inner +      # in_proj
        d_inner * d_inner +      # W_h
        d_inner +                # b_h
        d_inner * d_inner +      # W_gate
        d_inner +                # b_gate
        d_inner * dim +          # out_proj
        2 * dim                  # LayerNorm
    )

    total = (
        vocab_size * dim +       # embedding
        per_layer * depth +      # layers
        2 * dim                  # final norm
    )
    return total


def count_e42_params(dim, depth, vocab_size=256, expansion=2.0):
    """Count E42 (LinearTiedElman) parameters - tied W_x = W_h."""
    d_inner = int(dim * expansion)

    # Per layer:
    # - in_proj: dim -> d_inner (single, not doubled)
    # - W_h: d_inner x d_inner (W_x is tied to this)
    # - b_h: d_inner
    # - out_proj: d_inner -> dim
    # - norm: 2 * dim

    per_layer = (
        dim * d_inner +          # in_proj
        d_inner * d_inner +      # W_h (W_x is tied)
        d_inner +                # b_h
        d_inner * dim +          # out_proj
        2 * dim                  # LayerNorm
    )

    total = (
        vocab_size * dim +       # embedding
        per_layer * depth +      # layers
        2 * dim                  # final norm
    )
    return total


def count_e56_params(dim, depth, vocab_size=256, expansion=2.0):
    """Count E56 (ConcatElman) parameters - [x;h] concat GEMM."""
    d_inner = int(dim * expansion)

    # Per layer (concat elman):
    # - in_proj: dim -> 2*d_inner (like E1, split into x,z)
    # - W: d_inner x 2*d_inner (for [x;h] concat)
    # - b: d_inner
    # - out_proj: d_inner -> dim
    # - norm: 2 * dim

    per_layer = (
        dim * 2 * d_inner +      # in_proj
        d_inner * 2 * d_inner +  # W for [x;h] concat
        d_inner +                # b
        d_inner * dim +          # out_proj
        2 * dim                  # LayerNorm
    )

    total = (
        vocab_size * dim +       # embedding
        per_layer * depth +      # layers
        2 * dim                  # final norm
    )
    return total


def count_gru_params(dim, depth, vocab_size=256, expansion=2.0):
    """Count GRU parameters."""
    d_inner = int(dim * expansion)

    # Per layer:
    # - in_proj: dim -> d_inner
    # - GRU: 3 gates (reset, update, candidate)
    #   - weight_ih: 3 * d_inner * d_inner
    #   - weight_hh: 3 * d_inner * d_inner
    #   - bias_ih: 3 * d_inner
    #   - bias_hh: 3 * d_inner
    # - out_proj: d_inner -> dim
    # - norm: 2 * dim

    per_layer = (
        dim * d_inner +          # in_proj
        3 * d_inner * d_inner +  # weight_ih
        3 * d_inner * d_inner +  # weight_hh
        3 * d_inner +            # bias_ih
        3 * d_inner +            # bias_hh
        d_inner * dim +          # out_proj
        2 * dim                  # LayerNorm
    )

    total = (
        vocab_size * dim +       # embedding
        per_layer * depth +      # layers
        2 * dim                  # final norm
    )
    return total


def count_lstm_params(dim, depth, vocab_size=256, expansion=2.0):
    """Count LSTM parameters."""
    d_inner = int(dim * expansion)

    # Per layer:
    # - in_proj: dim -> d_inner
    # - LSTM: 4 gates (input, forget, cell, output)
    #   - weight_ih: 4 * d_inner * d_inner
    #   - weight_hh: 4 * d_inner * d_inner
    #   - bias_ih: 4 * d_inner
    #   - bias_hh: 4 * d_inner
    # - out_proj: d_inner -> dim
    # - norm: 2 * dim

    per_layer = (
        dim * d_inner +          # in_proj
        4 * d_inner * d_inner +  # weight_ih
        4 * d_inner * d_inner +  # weight_hh
        4 * d_inner +            # bias_ih
        4 * d_inner +            # bias_hh
        d_inner * dim +          # out_proj
        2 * dim                  # LayerNorm
    )

    total = (
        vocab_size * dim +       # embedding
        per_layer * depth +      # layers
        2 * dim                  # final norm
    )
    return total


def count_e64_params(dim, depth, vocab_size=256, expansion=2.0):
    """Count E64 (Additive H) parameters.

    E64: v = tanh(h + W_x @ x) - O(d), no W_h matrix
    - W_alpha [d_inner, d_inner], b_alpha [d_inner]
    - W_x [d_inner, d_inner], b [d_inner]
    """
    d_inner = int(dim * expansion)

    per_layer = (
        dim * d_inner +          # in_proj
        d_inner * d_inner +      # W_alpha
        d_inner +                # b_alpha
        d_inner * d_inner +      # W_x
        d_inner +                # b
        d_inner * dim +          # out_proj
        2 * dim                  # LayerNorm
    )

    total = (
        vocab_size * dim +       # embedding
        per_layer * depth +      # layers
        2 * dim                  # final norm
    )
    return total


def count_e65_params(dim, depth, vocab_size=256, expansion=2.0):
    """Count E65 (Diagonal H) parameters.

    E65: v = tanh(d_h * h + W_x @ x) - O(d), learnable diagonal
    - W_alpha [d_inner, d_inner], b_alpha [d_inner]
    - d_h [d_inner] (diagonal scaling)
    - W_x [d_inner, d_inner], b [d_inner]
    """
    d_inner = int(dim * expansion)

    per_layer = (
        dim * d_inner +          # in_proj
        d_inner * d_inner +      # W_alpha
        d_inner +                # b_alpha
        d_inner +                # d_h (diagonal)
        d_inner * d_inner +      # W_x
        d_inner +                # b
        d_inner * dim +          # out_proj
        2 * dim                  # LayerNorm
    )

    total = (
        vocab_size * dim +       # embedding
        per_layer * depth +      # layers
        2 * dim                  # final norm
    )
    return total


def count_e66_params(dim, depth, vocab_size=256, expansion=2.0, rank=None):
    """Count E66 (Low-Rank H) parameters.

    E66: v = tanh(U @ (V @ h) + W_x @ x) - O(d*r), two small GEMMs
    - W_alpha [d_inner, d_inner], b_alpha [d_inner]
    - V [rank, d_inner], U [d_inner, rank]
    - W_x [d_inner, d_inner], b [d_inner]
    """
    d_inner = int(dim * expansion)
    if rank is None:
        rank = max(d_inner // 4, 16)

    per_layer = (
        dim * d_inner +          # in_proj
        d_inner * d_inner +      # W_alpha
        d_inner +                # b_alpha
        rank * d_inner +         # V (compress)
        d_inner * rank +         # U (expand)
        d_inner * d_inner +      # W_x
        d_inner +                # b
        d_inner * dim +          # out_proj
        2 * dim                  # LayerNorm
    )

    total = (
        vocab_size * dim +       # embedding
        per_layer * depth +      # layers
        2 * dim                  # final norm
    )
    return total


def count_e67_params(dim, depth, vocab_size=256, expansion=2.0):
    """Count E67 (H-Gated) parameters.

    E67: α = sigmoid(W @ x + d * h) - O(d), h-dependence in gate
    - W_alpha [d_inner, d_inner], d_alpha [d_inner], b_alpha [d_inner]
    - W_x [d_inner, d_inner], b_v [d_inner]
    """
    d_inner = int(dim * expansion)

    per_layer = (
        dim * d_inner +          # in_proj
        d_inner * d_inner +      # W_alpha
        d_inner +                # d_alpha (diagonal)
        d_inner +                # b_alpha
        d_inner * d_inner +      # W_x
        d_inner +                # b_v
        d_inner * dim +          # out_proj
        2 * dim                  # LayerNorm
    )

    total = (
        vocab_size * dim +       # embedding
        per_layer * depth +      # layers
        2 * dim                  # final norm
    )
    return total


def count_e68_params(dim, depth, vocab_size=256, expansion=2.0):
    """Count E68 (Self-Gating) parameters.

    E68: v = tanh(W @ x) * sigmoid(d_g * h + b_g) - O(d), multiplicative h
    - W_alpha [d_inner, d_inner], b_alpha [d_inner]
    - d_g [d_inner], b_g [d_inner]
    - W_x [d_inner, d_inner], b_v [d_inner]
    """
    d_inner = int(dim * expansion)

    per_layer = (
        dim * d_inner +          # in_proj
        d_inner * d_inner +      # W_alpha
        d_inner +                # b_alpha
        d_inner +                # d_g
        d_inner +                # b_g
        d_inner * d_inner +      # W_x
        d_inner +                # b_v
        d_inner * dim +          # out_proj
        2 * dim                  # LayerNorm
    )

    total = (
        vocab_size * dim +       # embedding
        per_layer * depth +      # layers
        2 * dim                  # final norm
    )
    return total


def count_mamba2_params(dim, depth, vocab_size=256, expand=2):
    """Approximate Mamba2 parameters."""
    d_inner = dim * expand
    d_state = 64  # SSM state size

    # Per layer (rough estimate):
    # - in_proj: dim -> d_inner * 2 (x, z)
    # - x_proj for dt, B, C: d_inner -> (1 + 2*d_state) per head
    # - dt_bias, A_log, D params
    # - out_proj: d_inner -> dim
    # - norm: dim

    per_layer = (
        dim * d_inner * 2 +      # in_proj
        d_inner * dim +          # out_proj (approx)
        dim                      # norm
    )

    # This is a rough estimate - Mamba2 has complex structure
    total = (
        vocab_size * dim +
        per_layer * depth +
        dim * 2
    )
    # Add ~20% for SSM-specific params
    return int(total * 1.2)


def find_config(count_fn, target_params, expansion, max_depth=50):
    """Find dim, depth to hit target_params with given expansion."""
    best = None
    best_diff = float('inf')

    for depth in range(4, max_depth):
        # Binary search for dim
        low, high = 128, 2048
        while low < high:
            mid = align_to_128((low + high) // 2)
            params = count_fn(mid, depth, expansion=expansion)
            if params < target_params:
                low = mid + 128
            else:
                high = mid

        dim = align_to_128(low)
        params = count_fn(dim, depth, expansion=expansion)
        diff = abs(params - target_params)

        if diff < best_diff:
            best_diff = diff
            best = (dim, depth, params)

        # Also try one step down
        if dim > 128:
            dim2 = dim - 128
            params2 = count_fn(dim2, depth, expansion=expansion)
            diff2 = abs(params2 - target_params)
            if diff2 < best_diff:
                best_diff = diff2
                best = (dim2, depth, params2)

    return best


def find_config_at_depth(count_fn, target_params, expansion, target_depth=20):
    """Find dim to hit target_params at a specific depth."""
    depth = target_depth

    # Binary search for dim
    low, high = 128, 2048
    while low < high:
        mid = align_to_128((low + high) // 2)
        params = count_fn(mid, depth, expansion=expansion)
        if params < target_params:
            low = mid + 128
        else:
            high = mid

    dim = align_to_128(low)
    params = count_fn(dim, depth, expansion=expansion)

    # Also try one step down
    if dim > 128:
        dim2 = dim - 128
        params2 = count_fn(dim2, depth, expansion=expansion)
        if abs(params2 - target_params) < abs(params - target_params):
            dim, params = dim2, params2

    return (dim, depth, params)


def find_mamba2_config(target_params, expand=2):
    """Find dim, depth for Mamba2."""
    # Mamba2 structure is different - use empirical values
    # At expand=2, roughly: params ≈ dim² * depth * 5 + vocab * dim
    best = None
    best_diff = float('inf')

    for depth in range(8, 40):
        # Binary search for dim
        low, high = 128, 2048
        while low < high:
            mid = align_to_128((low + high) // 2)
            params = count_mamba2_params(mid, depth, expand=expand)
            if params < target_params:
                low = mid + 128
            else:
                high = mid

        dim = align_to_128(low)
        params = count_mamba2_params(dim, depth, expand=expand)
        diff = abs(params - target_params)

        if diff < best_diff:
            best_diff = diff
            best = (dim, depth, params)

    return best


def main():
    print("=" * 60)
    print(f"Finding 100M parameter configs with expansion={EXPANSION}, target_depth={TARGET_DEPTH}")
    print("=" * 60)

    configs = {}

    # E61 (DecayGated) - parallelizable
    dim, depth, params = find_config_at_depth(count_e61_params, TARGET_PARAMS, EXPANSION, TARGET_DEPTH)
    configs['e61'] = {'dim': dim, 'depth': depth, 'params': params, 'expansion': EXPANSION, 'level': 61}
    print(f"E61:    dim={dim}, depth={depth}, params={params:,}")

    # E62 (SelectiveWrite) - parallelizable
    dim, depth, params = find_config_at_depth(count_e62_params, TARGET_PARAMS, EXPANSION, TARGET_DEPTH)
    configs['e62'] = {'dim': dim, 'depth': depth, 'params': params, 'expansion': EXPANSION, 'level': 62}
    print(f"E62:    dim={dim}, depth={depth}, params={params:,}")

    # E63 (NonlinearDelta) - UTM-class, sequential
    dim, depth, params = find_config_at_depth(count_e63_params, TARGET_PARAMS, EXPANSION, TARGET_DEPTH)
    configs['e63'] = {'dim': dim, 'depth': depth, 'params': params, 'expansion': EXPANSION, 'level': 63}
    print(f"E63:    dim={dim}, depth={depth}, params={params:,}")

    # E63m (MatrixNonlinear) - UTM-class with matrix state
    dim, depth, params = find_config_at_depth(
        lambda d, dp, expansion: count_e63m_params(d, dp, expansion=expansion, n_slots=64),
        TARGET_PARAMS, EXPANSION, TARGET_DEPTH
    )
    configs['e63m'] = {'dim': dim, 'depth': depth, 'params': params, 'expansion': EXPANSION, 'level': '63m', 'n_slots': 64}
    print(f"E63m:   dim={dim}, depth={depth}, params={params:,} (n_slots=64)")

    # E64 (Additive H) - fast h-dependence, no W_h
    dim, depth, params = find_config_at_depth(count_e64_params, TARGET_PARAMS, EXPANSION, TARGET_DEPTH)
    configs['e64'] = {'dim': dim, 'depth': depth, 'params': params, 'expansion': EXPANSION, 'level': 64}
    print(f"E64:    dim={dim}, depth={depth}, params={params:,}")

    # E65 (Diagonal H) - diagonal scaling
    dim, depth, params = find_config_at_depth(count_e65_params, TARGET_PARAMS, EXPANSION, TARGET_DEPTH)
    configs['e65'] = {'dim': dim, 'depth': depth, 'params': params, 'expansion': EXPANSION, 'level': 65}
    print(f"E65:    dim={dim}, depth={depth}, params={params:,}")

    # E66 (Low-Rank H) - O(d*r) h transform
    dim, depth, params = find_config_at_depth(count_e66_params, TARGET_PARAMS, EXPANSION, TARGET_DEPTH)
    configs['e66'] = {'dim': dim, 'depth': depth, 'params': params, 'expansion': EXPANSION, 'level': 66}
    print(f"E66:    dim={dim}, depth={depth}, params={params:,}")

    # E67 (H-Gated) - h-dependence in gate
    dim, depth, params = find_config_at_depth(count_e67_params, TARGET_PARAMS, EXPANSION, TARGET_DEPTH)
    configs['e67'] = {'dim': dim, 'depth': depth, 'params': params, 'expansion': EXPANSION, 'level': 67}
    print(f"E67:    dim={dim}, depth={depth}, params={params:,}")

    # E68 (Self-Gating) - multiplicative h gating
    dim, depth, params = find_config_at_depth(count_e68_params, TARGET_PARAMS, EXPANSION, TARGET_DEPTH)
    configs['e68'] = {'dim': dim, 'depth': depth, 'params': params, 'expansion': EXPANSION, 'level': 68}
    print(f"E68:    dim={dim}, depth={depth}, params={params:,}")

    # E1 (MambaGatedElman)
    dim, depth, params = find_config_at_depth(count_e1_params, TARGET_PARAMS, EXPANSION, TARGET_DEPTH)
    configs['e1'] = {'dim': dim, 'depth': depth, 'params': params, 'expansion': EXPANSION, 'level': 1}
    print(f"E1:     dim={dim}, depth={depth}, params={params:,}")

    # E42 (LinearTiedElman)
    dim, depth, params = find_config_at_depth(count_e42_params, TARGET_PARAMS, EXPANSION, TARGET_DEPTH)
    configs['e42'] = {'dim': dim, 'depth': depth, 'params': params, 'expansion': EXPANSION, 'level': 42}
    print(f"E42:    dim={dim}, depth={depth}, params={params:,}")

    # E56 (MinimalGatedElman)
    dim, depth, params = find_config_at_depth(count_e56_params, TARGET_PARAMS, EXPANSION, TARGET_DEPTH)
    configs['e56'] = {'dim': dim, 'depth': depth, 'params': params, 'expansion': EXPANSION, 'level': 56}
    print(f"E56:    dim={dim}, depth={depth}, params={params:,}")

    # GRU
    dim, depth, params = find_config_at_depth(count_gru_params, TARGET_PARAMS, EXPANSION, TARGET_DEPTH)
    configs['cudagru'] = {'dim': dim, 'depth': depth, 'params': params, 'expansion': EXPANSION, 'level': 'cudagru'}
    print(f"GRU:    dim={dim}, depth={depth}, params={params:,}")

    # LSTM
    dim, depth, params = find_config_at_depth(count_lstm_params, TARGET_PARAMS, EXPANSION, TARGET_DEPTH)
    configs['cudalstm'] = {'dim': dim, 'depth': depth, 'params': params, 'expansion': EXPANSION, 'level': 'cudalstm'}
    print(f"LSTM:   dim={dim}, depth={depth}, params={params:,}")

    # Mamba2 - use expand=2 to match
    dim, depth, params = find_mamba2_config(TARGET_PARAMS, expand=2)
    configs['mamba2'] = {'dim': dim, 'depth': depth, 'params': params, 'expand': 2, 'level': 'mamba2'}
    print(f"Mamba2: dim={dim}, depth={depth}, params={params:,} (estimated)")

    # Save configs
    with open(OUTPUT_DIR / "configs.json", "w") as f:
        json.dump(configs, f, indent=2)

    # Get available GPUs dynamically
    available_gpus = get_available_gpus()
    print(f"\nAvailable GPUs: {available_gpus}")

    # Models to benchmark (priority order)
    model_order = ['e61', 'e62', 'e63', 'e63m', 'e64', 'e65', 'e66', 'e67', 'e68', 'e1', 'e42', 'e56', 'cudagru', 'cudalstm', 'mamba2']

    print("\n" + "=" * 60)
    print(f"Starting {TRAIN_MINUTES}-minute benchmarks with dynamic GPU allocation")
    print(f"Models: {len(model_order)}, GPUs available: {len(available_gpus)}")
    print("=" * 60)

    def launch_job(model_name, gpu_id, config):
        """Launch a training job on a specific GPU."""
        level = config.get('level', model_name)

        # Build command with appropriate args
        cmd_parts = [
            f"python train.py",
            f"--data data/pile.txt",
            f"--level {level}",
            f"--dim {config['dim']}",
            f"--depth {config['depth']}",
            f"--batch_size 32",
            f"--chunk_size 512",
            f"--steps 999999",
            f"--train_minutes {TRAIN_MINUTES}",
            f"--lr 3e-4",
            f"--log_every 10",
            f"--output {OUTPUT_DIR}/{model_name}",
            f"--bf16",
        ]

        # Add expansion if present
        if 'expansion' in config:
            cmd_parts.append(f"--expansion {config['expansion']}")

        cmd = " ".join(cmd_parts)

        p = subprocess.Popen(
            f"CUDA_VISIBLE_DEVICES={gpu_id} PYTHONUNBUFFERED=1 {cmd}",
            shell=True,
            stdout=open(OUTPUT_DIR / f"{model_name}.log", "w"),
            stderr=subprocess.STDOUT
        )
        return p

    # Track active jobs: {model_name: (process, gpu_id)}
    active_jobs = {}
    pending_models = list(model_order)
    completed_models = []

    # Initial batch: launch as many jobs as GPUs available
    for gpu_id in available_gpus:
        if not pending_models:
            break
        model_name = pending_models.pop(0)
        if model_name in configs:
            p = launch_job(model_name, gpu_id, configs[model_name])
            active_jobs[model_name] = (p, gpu_id)
            print(f"Launched {model_name} on GPU {gpu_id}")

    # Monitor and launch remaining jobs as GPUs become free
    while active_jobs or pending_models:
        time.sleep(5)  # Check every 5 seconds

        # Check for completed jobs
        completed = []
        for model_name, (p, gpu_id) in active_jobs.items():
            ret = p.poll()
            if ret is not None:
                completed.append((model_name, gpu_id, ret))

        # Process completed jobs and potentially launch new ones
        for model_name, gpu_id, ret in completed:
            del active_jobs[model_name]
            completed_models.append(model_name)
            print(f"{model_name} completed with exit code {ret} (GPU {gpu_id} now free)")

            # Launch next pending model on this GPU
            if pending_models:
                next_model = pending_models.pop(0)
                if next_model in configs:
                    p = launch_job(next_model, gpu_id, configs[next_model])
                    active_jobs[next_model] = (p, gpu_id)
                    print(f"Launched {next_model} on GPU {gpu_id}")

        # Show progress
        if completed:
            print(f"Progress: {len(completed_models)}/{len(model_order)} complete, "
                  f"{len(active_jobs)} running, {len(pending_models)} pending")

    print("\n" + "=" * 60)
    print("All benchmarks complete!")
    print(f"Completed models: {completed_models}")
    print("=" * 60)


if __name__ == "__main__":
    main()
