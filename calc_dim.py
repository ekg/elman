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


def calc_transformer_params(dim, depth, n_heads=8, expansion=4.0, vocab_size=256):
    """Calculate Transformer (Llama-style) parameters."""
    # Self-attention: Q, K, V, O projections
    attn = dim * dim * 4  # 4 projections of dim x dim
    # FFN: up_proj, gate_proj, down_proj (SwiGLU style)
    ffn_dim = int(dim * expansion)
    ffn = dim * ffn_dim * 3  # 3 projections
    # RMSNorm
    norm = dim * 2
    per_layer = attn + ffn + norm
    layers_total = per_layer * depth
    embed = vocab_size * dim
    return layers_total + embed


def calc_gru_params(dim, depth, expansion=1.0, vocab_size=256):
    """Calculate GRU parameters (CudaGRU with expansion)."""
    dim_inner = int(dim * expansion)
    # Per layer: input_proj, output_proj, GRU gates on dim_inner
    # GRU gates: 3 gates * (dim_inner*dim_inner for W_hh + dim_inner for bias)
    # Plus input_proj (dim -> dim_inner) and output_proj (dim_inner -> dim)
    gru_params = 3 * (dim_inner * dim_inner + dim_inner)  # W_hh and bias for 3 gates
    input_proj = dim * dim_inner
    output_proj = dim_inner * dim
    layer_norm = 2 * dim  # weight and bias
    per_layer = gru_params + input_proj + output_proj + layer_norm
    layers_total = per_layer * depth
    embed = vocab_size * dim * 2  # embedding + lm_head (tied)
    return layers_total + embed


def calc_lstm_params(dim, depth, expansion=1.0, vocab_size=256):
    """Calculate LSTM parameters (CudaLSTM with expansion)."""
    dim_inner = int(dim * expansion)
    # Per layer: input_proj, output_proj, LSTM gates on dim_inner
    # LSTM gates: 4 gates * (dim_inner*dim_inner for W_hh + dim_inner for bias)
    lstm_params = 4 * (dim_inner * dim_inner + dim_inner)  # W_hh and bias for 4 gates
    input_proj = dim * dim_inner
    output_proj = dim_inner * dim
    layer_norm = 2 * dim  # weight and bias
    per_layer = lstm_params + input_proj + output_proj + layer_norm
    layers_total = per_layer * depth
    embed = vocab_size * dim * 2  # embedding + lm_head (tied)
    return layers_total + embed


def calc_mingru_params(dim, depth, expansion=2.0, vocab_size=256):
    """Calculate minGRU parameters (simplified GRU from Feng et al.)."""
    d_inner = int(dim * expansion)
    # minGRU: in_proj, out_proj, and simplified gates
    per_layer = (
        dim * d_inner +      # in_proj
        d_inner * dim +      # out_proj
        d_inner * d_inner * 2  # simplified W_z and W_h
    )
    layers_total = per_layer * depth
    embed = vocab_size * dim
    return layers_total + embed


def calc_minlstm_params(dim, depth, expansion=2.0, vocab_size=256):
    """Calculate minLSTM parameters (simplified LSTM from Feng et al.)."""
    d_inner = int(dim * expansion)
    # minLSTM: in_proj, out_proj, and simplified gates
    per_layer = (
        dim * d_inner +      # in_proj
        d_inner * dim +      # out_proj
        d_inner * d_inner * 3  # simplified W_i, W_f, W_o
    )
    layers_total = per_layer * depth
    embed = vocab_size * dim
    return layers_total + embed


def calc_e88_params(dim, n_heads, n_state, depth, expansion=1.0, vocab_size=256, use_gate=True):
    """Calculate E88 FLA Hybrid parameters.

    Args:
        use_gate: If True (default), includes g_proj for output gating.
                  Set False for "best" ablated config (no gate).
    """
    # Key dimensions
    key_dim = n_heads * n_state
    value_dim = int(n_heads * n_state * expansion)

    # Per layer:
    # qkv_proj: dim → 2*key_dim + value_dim = 3*H*n (when expansion=1.0)
    # a_proj: dim → n_heads (decay)
    # A_log: n_heads
    # dt_bias: n_heads
    # g_proj: dim → value_dim (only if use_gate=True)
    # o_proj: value_dim → dim
    # o_norm_weight: n_state (always created)
    qkv_proj = dim * (2 * key_dim + value_dim)
    decay_params = dim * n_heads + n_heads + n_heads  # a_proj + A_log + dt_bias
    gate_proj = dim * value_dim if use_gate else 0
    out_proj = value_dim * dim
    norm_weight = n_state  # head_v_dim

    per_layer = qkv_proj + decay_params + gate_proj + out_proj + norm_weight

    layers_total = per_layer * depth
    embed = vocab_size * dim  # tied embeddings
    norms = dim * (depth + 1)  # RMSNorm

    return layers_total + embed + norms


def calc_mom_e88_params(dim, n_heads, top_k, n_state, depth, expansion=1.0, vocab_size=256, use_gate=True):
    """Calculate MoM E88 (Mixture of Memory) parameters.

    Note: top_k doesn't affect param count - only H, n, and dim do.
    top_k affects compute and state size, not params.

    Args:
        dim: Model dimension
        n_heads: Total number of memory heads (H)
        top_k: Number of active heads per token (K) - doesn't affect params
        n_state: State dimension per head (n)
        depth: Number of layers
        expansion: Value dimension expansion (default 1.0)
        vocab_size: Vocabulary size
        use_gate: If True (default), includes g_proj for output gating.
    """
    # Key dimensions
    key_dim = n_heads * n_state
    value_dim = int(n_heads * n_state * expansion)
    head_v_dim = value_dim // n_heads  # n_state when expansion=1.0

    # Per layer:
    # router: dim → n_heads (for top-K selection)
    # qkv_proj: dim → 2*key_dim + value_dim = 3*H*n (when expansion=1.0)
    # a_proj: dim → n_heads (decay)
    # A_log: n_heads
    # dt_bias: n_heads
    # g_proj: dim → value_dim (only if use_gate=True)
    # o_proj: head_v_dim → dim (different from E88: projects per-head output)
    router = dim * n_heads
    qkv_proj = dim * (2 * key_dim + value_dim)
    decay_params = dim * n_heads + n_heads + n_heads  # a_proj + A_log + dt_bias
    gate_proj = dim * value_dim if use_gate else 0
    out_proj = head_v_dim * dim  # Note: o_proj is per head_v_dim, not full value_dim

    per_layer = router + qkv_proj + decay_params + gate_proj + out_proj

    layers_total = per_layer * depth
    embed = vocab_size * dim  # tied embeddings
    norms = dim * (depth + 1)  # RMSNorm

    return layers_total + embed + norms


def calc_e90_params(dim, n_heads, k_fast, k_slow, depth, vocab_size=256, use_gate=True):
    """Calculate E90 Dual-Rate State parameters.

    E90 has two memory systems per head:
    - Fast state: k_fast × k_fast per head (updated every step)
    - Slow state: k_slow × k_slow per head (gated update)

    Args:
        dim: Model dimension
        n_heads: Number of heads (H)
        k_fast: Fast state key/value dimension
        k_slow: Slow state key/value dimension
        depth: Number of layers
        vocab_size: Vocabulary size
        use_gate: If True (default), includes g_proj for output gating
    """
    # Fast state projections
    key_dim_fast = n_heads * k_fast
    value_dim_fast = n_heads * k_fast  # Square state
    qkv_fast = dim * (2 * key_dim_fast + value_dim_fast)  # q, k, v for fast

    # Slow state projections
    key_dim_slow = n_heads * k_slow
    value_dim_slow = n_heads * k_slow  # Square state
    qkv_slow = dim * (2 * key_dim_slow + value_dim_slow)  # q, k, v for slow

    # Output dimension is max(v_fast, v_slow) per head
    out_v_dim = max(k_fast, k_slow)

    # Per layer:
    # qkv_proj (fast): dim → 2*key_dim_fast + value_dim_fast
    # qkv_slow_proj: dim → 2*key_dim_slow + value_dim_slow
    # slow_gate_proj: dim → n_heads (with bias)
    # mix_proj: dim → n_heads * 2 (with bias)
    # a_proj (fast decay): dim → n_heads
    # a_slow_proj (slow decay): dim → n_heads
    # A_log, dt_bias for fast: n_heads each
    # A_slow_log, dt_slow_bias for slow: n_heads each
    # g_proj (gate): dim → n_heads * out_v_dim (if use_gate)
    # o_proj: n_heads * out_v_dim → dim

    decay_params_fast = dim * n_heads + n_heads + n_heads  # a_proj + A_log + dt_bias
    decay_params_slow = dim * n_heads + n_heads + n_heads  # a_slow_proj + A_slow_log + dt_slow_bias
    slow_gate = dim * n_heads + n_heads  # slow_gate_proj with bias
    mix_proj = dim * n_heads * 2 + n_heads * 2  # mix_proj with bias
    gate_proj = dim * (n_heads * out_v_dim) if use_gate else 0
    out_proj = (n_heads * out_v_dim) * dim

    per_layer = (qkv_fast + qkv_slow + decay_params_fast + decay_params_slow +
                 slow_gate + mix_proj + gate_proj + out_proj)

    layers_total = per_layer * depth
    embed = vocab_size * dim  # tied embeddings
    norms = dim * (depth + 1)  # RMSNorm

    return layers_total + embed + norms


def find_dim_for_params(calc_func, target_params, **kwargs):
    """Binary search for 128-aligned dim that hits target params."""
    max_dim = 8192  # Extended for 500M+ models
    for dim in range(128, max_dim + 1, 128):
        params = calc_func(dim=dim, **kwargs)
        if params >= target_params:
            # Check if previous was closer
            prev_dim = dim - 128
            if prev_dim >= 128:
                prev_params = calc_func(dim=prev_dim, **kwargs)
                if abs(prev_params - target_params) < abs(params - target_params):
                    return prev_dim, prev_params
            return dim, params
    return max_dim, calc_func(dim=max_dim, **kwargs)


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

    # E88 variants (best config: expansion=1.0, no conv, no gate)
    print()
    print("E88 FLA Hybrid (expansion=1.0, ablated):")
    e88_configs = [
        (4, 32), (6, 32), (8, 32), (12, 32), (16, 32), (20, 32),
        (24, 24), (32, 16),
        (8, 48), (8, 64),
    ]
    for n_heads, n_state in e88_configs:
        dim, params = find_dim_for_params(
            calc_e88_params, target,
            n_heads=n_heads, n_state=n_state, depth=20, expansion=1.0
        )
        name = f"E88h{n_heads}n{n_state}"
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

    elif model.startswith('e88'):
        # Parse E88h8n32 format
        import re
        match = re.match(r'e88h(\d+)n(\d+)', model)
        if not match:
            print(f"Invalid E88 format: {model}. Use E88h8n32 style.")
            return
        n_heads = int(match.group(1))
        n_state = int(match.group(2))

        if n_state % 8 != 0:
            print(f"ERROR: n_state must be multiple of 8, got {n_state}")
            return

        dim, params = find_dim_for_params(
            calc_e88_params, target,
            n_heads=n_heads, n_state=n_state, depth=args.depth, expansion=args.expansion
        )
        config = {
            'model': f'E88h{n_heads}n{n_state}',
            'dim': dim, 'depth': args.depth, 'n_heads': n_heads, 'n_state': n_state,
            'expansion': args.expansion, 'params': params,
            'state_per_layer': n_heads * n_state * n_state  # H × n² state
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
