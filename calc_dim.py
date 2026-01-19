#!/usr/bin/env python3
"""
Model Dimension Calculator

Calculates the correct 128-aligned dimension to hit target parameters.
Ensures all constraints are met (128-alignment, n_state multiples of 8, etc.)

Usage:
    python calc_dim.py --model E75h4n32 --params 100M --depth 20
    python calc_dim.py --model mamba2 --params 100M --depth 20
    python calc_dim.py --model fla-gdn --params 100M --depth 20

    # Show all standard 100M configs
    python calc_dim.py --standard
"""

import argparse
import sys

# Model parameter formulas (approximate, per layer)
# All dims must be 128-aligned
# n_state must be multiple of 8 for E75

def calc_e75_params(dim, n_heads, n_state, depth, expansion=1.0, vocab_size=256):
    """Calculate E75 MultiHead parameters."""
    d_inner = int(dim * expansion)

    # Per layer:
    # in_proj: dim * d_inner
    # W_k, W_v, W_q, W_beta: 4 * (n_heads * n_state) * d_inner
    # b_beta: n_heads * n_state
    # out_proj: (n_heads * n_state) * dim
    in_proj = dim * d_inner
    cell_W = 4 * (n_heads * n_state) * d_inner
    cell_b = n_heads * n_state
    out_proj = (n_heads * n_state) * dim
    per_layer = in_proj + cell_W + cell_b + out_proj

    layers_total = per_layer * depth
    embed = vocab_size * dim  # tied embeddings
    norms = dim * (depth + 1)  # RMSNorm

    return layers_total + embed + norms


def calc_mamba2_params(dim, depth, expand=2, d_state=128, vocab_size=256):
    """Calculate Mamba2 parameters (approximate)."""
    d_inner = dim * expand
    # Per layer (approximate from mamba2 code):
    # in_proj, out_proj, conv, dt_proj, A_log, D, etc.
    per_layer = (
        dim * d_inner * 2 +  # in_proj
        d_inner * dim +      # out_proj
        d_inner * 4 +        # conv1d
        d_inner * 2 +        # dt, A, D
        d_inner * d_state    # SSM state
    )
    layers_total = per_layer * depth
    embed = vocab_size * dim
    return layers_total + embed


def calc_fla_gdn_params(dim, depth, expansion=2.0, vocab_size=256):
    """Calculate FLA GatedDeltaNet parameters (approximate)."""
    d_inner = int(dim * expansion)
    # Per layer: similar to linear attention
    per_layer = (
        dim * d_inner * 3 +  # Q, K, V projections
        d_inner * dim +      # out_proj
        d_inner * 2          # gates
    )
    layers_total = per_layer * depth
    embed = vocab_size * dim
    return layers_total + embed


def find_dim_for_params(calc_func, target_params, **kwargs):
    """Binary search for 128-aligned dim that hits target params."""
    for dim in range(128, 4096, 128):
        params = calc_func(dim=dim, **kwargs)
        if params >= target_params:
            # Check if previous was closer
            prev_dim = dim - 128
            if prev_dim >= 128:
                prev_params = calc_func(dim=prev_dim, **kwargs)
                if abs(prev_params - target_params) < abs(params - target_params):
                    return prev_dim, prev_params
            return dim, params
    return 3968, calc_func(dim=3968, **kwargs)


def parse_params(s):
    """Parse param string like '100M' or '100000000'."""
    s = s.strip().upper()
    if s.endswith('M'):
        return int(float(s[:-1]) * 1_000_000)
    elif s.endswith('K'):
        return int(float(s[:-1]) * 1_000)
    elif s.endswith('B'):
        return int(float(s[:-1]) * 1_000_000_000)
    else:
        return int(s)


def print_standard_configs():
    """Print all standard 100M configurations."""
    print("Standard 100M Parameter Configurations")
    print("=" * 70)
    print(f"{'Model':<12} {'Dim':<6} {'Depth':<6} {'Extra':<20} {'Params':<12}")
    print("-" * 70)

    target = 100_000_000

    # Mamba2
    dim, params = find_dim_for_params(calc_mamba2_params, target, depth=20)
    print(f"{'mamba2':<12} {dim:<6} {20:<6} {'expand=2':<20} {params/1e6:.1f}M")

    # FLA-GDN
    dim, params = find_dim_for_params(calc_fla_gdn_params, target, depth=20, expansion=2.0)
    print(f"{'fla-gdn':<12} {dim:<6} {20:<6} {'expansion=2.0':<20} {params/1e6:.1f}M")

    # E75 variants
    e75_configs = [
        (4, 16), (4, 24), (4, 32), (4, 48),
        (8, 16), (8, 24), (8, 32),
        (6, 24), (6, 32),
    ]
    for n_heads, n_state in e75_configs:
        dim, params = find_dim_for_params(
            calc_e75_params, target,
            n_heads=n_heads, n_state=n_state, depth=20, expansion=1.0
        )
        name = f"E75h{n_heads}n{n_state}"
        extra = f"H={n_heads}, n={n_state}"
        print(f"{name:<12} {dim:<6} {20:<6} {extra:<20} {params/1e6:.1f}M")


def main():
    parser = argparse.ArgumentParser(description='Calculate model dimensions for target params')
    parser.add_argument('--model', type=str, help='Model type (E75h4n32, mamba2, fla-gdn, etc.)')
    parser.add_argument('--params', type=str, default='100M', help='Target parameters (e.g., 100M)')
    parser.add_argument('--depth', type=int, default=20, help='Number of layers')
    parser.add_argument('--expansion', type=float, default=1.0, help='Expansion factor')
    parser.add_argument('--standard', action='store_true', help='Print all standard 100M configs')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    args = parser.parse_args()

    if args.standard:
        print_standard_configs()
        return

    if not args.model:
        parser.print_help()
        return

    target = parse_params(args.params)
    model = args.model.lower()

    if model == 'mamba2':
        dim, params = find_dim_for_params(calc_mamba2_params, target, depth=args.depth)
        config = {'model': 'mamba2', 'dim': dim, 'depth': args.depth, 'params': params}

    elif model.startswith('fla') or model == 'gdn':
        dim, params = find_dim_for_params(
            calc_fla_gdn_params, target, depth=args.depth, expansion=args.expansion
        )
        config = {'model': 'fla-gdn', 'dim': dim, 'depth': args.depth,
                  'expansion': args.expansion, 'params': params}

    elif model.startswith('e75'):
        # Parse E75h4n32 format
        import re
        match = re.match(r'e75h(\d+)n(\d+)', model)
        if not match:
            print(f"Invalid E75 format: {model}. Use E75h4n32 style.")
            return
        n_heads = int(match.group(1))
        n_state = int(match.group(2))

        if n_state % 8 != 0:
            print(f"ERROR: n_state must be multiple of 8, got {n_state}")
            return

        dim, params = find_dim_for_params(
            calc_e75_params, target,
            n_heads=n_heads, n_state=n_state, depth=args.depth, expansion=args.expansion
        )
        config = {
            'model': f'E75h{n_heads}n{n_state}',
            'dim': dim, 'depth': args.depth, 'n_heads': n_heads, 'n_state': n_state,
            'expansion': args.expansion, 'params': params
        }
    else:
        print(f"Unknown model type: {model}")
        return

    if args.json:
        import json
        print(json.dumps(config))
    else:
        print(f"Model: {config['model']}")
        print(f"Dim: {config['dim']} (128-aligned)")
        print(f"Depth: {config['depth']}")
        if 'n_heads' in config:
            print(f"n_heads: {config['n_heads']}")
            print(f"n_state: {config['n_state']}")
        if 'expansion' in config:
            print(f"Expansion: {config['expansion']}")
        print(f"Parameters: {config['params']:,} ({config['params']/1e6:.1f}M)")


if __name__ == '__main__':
    main()
