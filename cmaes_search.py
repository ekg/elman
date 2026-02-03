#!/usr/bin/env python3
"""
CMA-ES Architecture/Hyperparameter Search for E88, FLA-GDN, and Mamba2

Uses CMA-ES to find optimal configurations by running short training runs
and using final loss as fitness.

Usage:
    pip install cma
    python cmaes_search.py --model e88 --generations 50 --train_minutes 2 --gpus 0,1,2,3

Fixed issues:
- Uses calc_dim.py functions for accurate param estimation
- Validates n_state against CUDA kernel support list
- Runs population in parallel across multiple GPUs
"""

import os
import sys
import argparse
import subprocess
import json
import re
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

try:
    import cma
except ImportError:
    print("Please install cma: pip install cma")
    sys.exit(1)

# Import param calculation functions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calc_dim import (
    calc_e88_params, calc_fla_gdn_params, calc_mamba2_params, find_dim_for_params,
    calc_transformer_params, calc_gru_params, calc_lstm_params,
    calc_mingru_params, calc_minlstm_params, calc_mom_e88_params, calc_e90_params,
    calc_e1_params, calc_e23_params, calc_e42_params, calc_e75_params
)

# Supported n_state values for E88 CUDA fused gate kernel (must match head_v_dim for default config)
# Only these sizes are efficiently supported without warnings/fallbacks: 16, 32, 48, 64
E88_SUPPORTED_N_STATE = [16, 32, 48, 64]

# E90 valid (k_fast, k_slow) configurations supported by CUDA kernel
# Note: k_slow=64 configs have backward kernel bug, excluded for now
E90_CONFIGS = [
    (8, 16),   # config_idx=0
    (8, 24),   # config_idx=1
    (16, 32),  # config_idx=2
    (16, 48),  # config_idx=3
]

# Search space definitions for each model
# All search spaces are 6D: dim + architecture params + learning rate (log scale)
# - dim: 1024-3072 (128-aligned) - model width
# - lr: 1e-5 to 1e-3 (log scale) - learning rate
SEARCH_SPACES = {
    'e88': {
        # 6D: capacity (dim, depth) + architecture (n_heads, n_state, use_gate) + lr
        # Note: expansion fixed at 1.0 (head_v_dim = n_state) to avoid shape issues
        'dim': (1024, 3072, 'int_mult128', 'Model dimension'),
        'n_heads': (32, 160, 'int', 'Number of attention heads'),
        'n_state': (16, 64, 'e88_n_state', 'State dimension (only 16,32,48,64 supported)'),
        'depth': (12, 40, 'int', 'Number of layers'),
        'use_gate': (0, 1, 'binary', 'Use output gating (0=no, 1=yes)'),
        'lr': (1e-5, 1e-3, 'log', 'Learning rate (log scale)'),
    },
    'fla-gdn': {
        # 6D: capacity (dim, depth) + architecture (expansion, n_heads, use_conv) + lr
        'dim': (1024, 3072, 'int_mult128', 'Model dimension'),
        'expansion': (1, 3, 'int', 'Value expansion factor'),
        'depth': (12, 40, 'int', 'Number of layers'),
        'n_heads': (8, 32, 'int', 'Number of heads'),
        'use_conv': (0, 1, 'binary', 'Use short convolution (0=no, 1=yes)'),
        'lr': (1e-5, 1e-3, 'log', 'Learning rate (log scale)'),
    },
    'mamba2': {
        # 6D: capacity (dim, depth) + state (d_state, headdim, expand) + lr
        'dim': (1024, 3072, 'int_mult128', 'Model dimension'),
        'd_state': (64, 256, 'int_mult16', 'SSM state dimension'),
        'headdim': (32, 128, 'int_pow2', 'Head dimension (32, 64, or 128)'),
        'expand': (1, 3, 'int', 'Expansion factor'),
        'depth': (16, 40, 'int', 'Number of layers'),
        'lr': (1e-5, 1e-3, 'log', 'Learning rate (log scale)'),
    },
    'transformer': {
        # 6D: capacity (dim, depth) + attention (n_heads, expansion) + regularization (dropout) + lr
        # Note: head_dim = dim / n_heads (computed automatically)
        'dim': (1024, 3072, 'int_mult128', 'Model dimension'),
        'n_heads': (8, 32, 'int', 'Number of attention heads'),
        'expansion': (2, 6, 'int', 'FFN expansion factor'),
        'depth': (12, 36, 'int', 'Number of layers'),
        'dropout': (0.0, 0.15, 'float', 'Dropout rate'),
        'lr': (1e-5, 1e-3, 'log', 'Learning rate (log scale)'),
    },
    'gru': {
        # 4D - GRU (may be unstable at large scale)
        'dim': (512, 2048, 'int_mult128', 'Model dimension'),
        'expansion': (1, 3, 'int', 'Expansion factor for dim_inner'),
        'depth': (12, 48, 'int', 'Number of layers'),
        'lr': (1e-5, 1e-3, 'log', 'Learning rate (log scale)'),
    },
    'lstm': {
        # 4D - LSTM (may be unstable at large scale)
        'dim': (512, 2048, 'int_mult128', 'Model dimension'),
        'expansion': (1, 3, 'int', 'Expansion factor for dim_inner'),
        'depth': (12, 48, 'int', 'Number of layers'),
        'lr': (1e-5, 1e-3, 'log', 'Learning rate (log scale)'),
    },
    'mingru': {
        # 6D: capacity (dim, expansion, depth) + architecture (use_conv, d_conv) + lr
        'dim': (1024, 3584, 'int_mult128', 'Model dimension'),
        'expansion': (1, 4, 'int', 'Expansion factor'),
        'depth': (12, 40, 'int', 'Number of layers'),
        'use_conv': (0, 1, 'binary', 'Use Conv1d (0=no, 1=yes)'),
        'd_conv': (3, 7, 'int', 'Conv kernel size (if use_conv=1)'),
        'lr': (1e-5, 1e-3, 'log', 'Learning rate (log scale)'),
    },
    'minlstm': {
        # 6D: capacity (dim, expansion, depth) + architecture (use_conv, d_conv) + lr
        'dim': (1024, 3584, 'int_mult128', 'Model dimension'),
        'expansion': (1, 4, 'int', 'Expansion factor'),
        'depth': (12, 40, 'int', 'Number of layers'),
        'use_conv': (0, 1, 'binary', 'Use Conv1d (0=no, 1=yes)'),
        'd_conv': (3, 7, 'int', 'Conv kernel size (if use_conv=1)'),
        'lr': (1e-5, 1e-3, 'log', 'Learning rate (log scale)'),
    },
    'mom-e88': {
        # 6D: capacity (dim, depth) + routing (n_heads, top_k) + state (n_state) + lr
        'dim': (1024, 3072, 'int_mult128', 'Model dimension'),
        'n_heads': (32, 256, 'int', 'Total number of memory heads'),
        'top_k': (8, 96, 'int', 'Active heads per token'),
        'n_state': (16, 64, 'e88_n_state', 'State dimension (only 16,32,48,64 supported)'),
        'depth': (8, 32, 'int', 'Number of layers'),
        'lr': (1e-5, 1e-3, 'log', 'Learning rate (log scale)'),
    },
    'e90': {
        # 6D: capacity (dim, depth) + architecture (n_heads, config_idx, use_gate) + lr
        'dim': (1024, 3072, 'int_mult128', 'Model dimension'),
        'n_heads': (32, 128, 'int', 'Number of heads'),
        'config_idx': (0, 3, 'e90_config', 'Fast/slow config index'),
        'depth': (12, 32, 'int', 'Number of layers'),
        'use_gate': (0, 1, 'binary', 'Use output gating (0=no, 1=yes)'),
        'lr': (1e-5, 1e-3, 'log', 'Learning rate (log scale)'),
    },
    'e1': {
        # 6D: capacity (dim, expansion, depth) + architecture (use_conv, mamba2_init) + lr
        'dim': (1024, 3072, 'int_mult128', 'Model dimension'),
        'expansion': (1, 3, 'int', 'Expansion factor for d_inner'),
        'depth': (12, 40, 'int', 'Number of layers'),
        'use_conv': (0, 1, 'binary', 'Use Conv1d for local context (0=no, 1=yes)'),
        'mamba2_init': (0, 1, 'binary', 'Use Mamba2-style init (0=xavier, 1=mamba2)'),
        'lr': (1e-5, 1e-3, 'log', 'Learning rate (log scale)'),
    },
    'e23': {
        # 6D: capacity (dim, expansion, depth) + architecture (n_slots) + stability (w_h_init_scale) + lr
        'dim': (1024, 3072, 'int_mult128', 'Model dimension'),
        'n_slots': (32, 128, 'int', 'Number of tape memory slots'),
        'expansion': (1, 3, 'float', 'Hidden expansion factor'),
        'depth': (12, 40, 'int', 'Number of layers'),
        'w_h_init_scale': (0.5, 1.0, 'float', 'W_h orthogonal init scale (stability)'),
        'lr': (1e-5, 1e-3, 'log', 'Learning rate (log scale)'),
    },
    'e42': {
        # 6D: capacity (dim, expansion, depth) + stability (spectral_radius) + init (mamba2_init) + lr
        'dim': (1024, 3072, 'int_mult128', 'Model dimension'),
        'expansion': (1, 3, 'int', 'Expansion factor for d_inner'),
        'depth': (12, 48, 'int', 'Number of layers'),
        'spectral_radius': (0.95, 0.9999, 'float', 'Target spectral radius for W (stability)'),
        'mamba2_init': (0, 1, 'binary', 'Init strategy (0=xavier, 1=mamba2)'),
        'lr': (1e-5, 1e-3, 'log', 'Learning rate (log scale)'),
    },
    'e75': {
        # 6D: capacity (dim, depth) + architecture (n_heads, n_state, expansion) + lr
        'dim': (1024, 3072, 'int_mult128', 'Model dimension'),
        'n_heads': (2, 16, 'int', 'Number of heads'),
        'n_state': (16, 64, 'int_mult8', 'State dimension per head (must be mult of 8)'),
        'depth': (12, 40, 'int', 'Number of layers'),
        'expansion': (0.5, 2.0, 'float', 'Value expansion factor'),
        'lr': (1e-5, 1e-3, 'log', 'Learning rate (log scale)'),
    },
}


def decode_params(x, model_type):
    """Convert CMA-ES vector to model parameters."""
    space = SEARCH_SPACES[model_type]
    params = {}

    for i, (name, (lo, hi, ptype, desc)) in enumerate(space.items()):
        val = x[i]
        # Clip to [0, 1] range (CMA-ES can go outside)
        val = np.clip(val, 0, 1)

        if ptype == 'int':
            params[name] = int(round(lo + val * (hi - lo)))
        elif ptype == 'binary':
            # Binary 0/1 parameter
            params[name] = 1 if val >= 0.5 else 0
        elif ptype == 'int_mult16':
            raw = lo + val * (hi - lo)
            params[name] = int(round(raw / 16) * 16)
            params[name] = max(16, params[name])
        elif ptype == 'int_mult8':
            raw = lo + val * (hi - lo)
            params[name] = int(round(raw / 8) * 8)
            params[name] = max(8, params[name])
        elif ptype == 'int_mult128':
            raw = lo + val * (hi - lo)
            params[name] = int(round(raw / 128) * 128)
            params[name] = max(128, params[name])
        elif ptype == 'int_pow2':
            # Map to nearest power of 2 within range
            raw = lo + val * (hi - lo)
            # Valid powers of 2 in typical ranges: 32, 64, 128
            powers = [p for p in [16, 32, 64, 128, 256] if lo <= p <= hi]
            if powers:
                closest = min(powers, key=lambda p: abs(p - raw))
                params[name] = closest
            else:
                params[name] = int(round(raw))
        elif ptype == 'e88_n_state':
            # Map to nearest supported n_state value
            raw = lo + val * (hi - lo)
            # Find closest supported value
            closest = min(E88_SUPPORTED_N_STATE, key=lambda x: abs(x - raw))
            params[name] = closest
        elif ptype == 'e90_config':
            # Map to valid E90 (k_fast, k_slow) config index
            idx = int(round(lo + val * (hi - lo)))
            idx = max(0, min(idx, len(E90_CONFIGS) - 1))
            params[name] = idx
        elif ptype == 'log':
            # Log-scale interpolation
            log_lo, log_hi = np.log10(lo), np.log10(hi)
            params[name] = 10 ** (log_lo + val * (log_hi - log_lo))
        else:  # float
            params[name] = lo + val * (hi - lo)

    # MoM E88 constraint: top_k must be <= n_heads
    if model_type == 'mom-e88' and 'top_k' in params and 'n_heads' in params:
        params['top_k'] = min(params['top_k'], params['n_heads'])

    return params


def encode_params(params, model_type):
    """Convert model parameters to CMA-ES vector [0,1]^n."""
    space = SEARCH_SPACES[model_type]
    x = []

    for name, (lo, hi, ptype, desc) in space.items():
        val = params.get(name, (lo + hi) / 2)  # Default to middle if not specified

        if ptype == 'binary':
            # Binary: 0 -> 0.25, 1 -> 0.75 (away from boundaries)
            x_val = 0.75 if val else 0.25
        elif ptype == 'int_pow2':
            # Power of 2: linear interpolation
            x_val = (val - lo) / (hi - lo)
        elif ptype in ('int', 'int_mult16', 'int_mult8', 'int_mult128', 'e88_n_state', 'e90_config'):
            # Linear interpolation
            x_val = (val - lo) / (hi - lo)
        elif ptype == 'log':
            # Log-scale interpolation
            log_lo, log_hi = np.log10(lo), np.log10(hi)
            x_val = (np.log10(val) - log_lo) / (log_hi - log_lo)
        else:  # float
            x_val = (val - lo) / (hi - lo)

        x.append(np.clip(x_val, 0, 1))

    return x


# Known best configs for warm-starting (6D: includes dim and lr)
BEST_CONFIGS = {
    'e88': {
        'dim': 2176,
        'n_heads': 98,
        'n_state': 32,
        'depth': 14,
        'use_gate': 1,  # SiLU gating helps at 480M scale
        'lr': 3e-4,
    },
    'fla-gdn': {
        'dim': 1920,
        'expansion': 2,
        'depth': 17,
        'n_heads': 24,
        'use_conv': 1,  # FLA-GDN typically uses short conv
        'lr': 3e-4,
    },
    'mamba2': {
        'dim': 1792,
        'd_state': 96,
        'headdim': 64,
        'expand': 2,
        'depth': 25,
        'lr': 3e-4,
    },
    'transformer': {
        'dim': 1536,
        'n_heads': 16,
        'expansion': 4,
        'depth': 24,
        'dropout': 0.0,  # No dropout for short training runs
        'lr': 1e-4,  # Transformers often need lower LR
    },
    'gru': {
        'dim': 1024,
        'expansion': 1,
        'depth': 20,
        'lr': 3e-4,
    },
    'lstm': {
        'dim': 1024,
        'expansion': 1,
        'depth': 20,
        'lr': 3e-4,
    },
    'mingru': {
        'dim': 2944,
        'expansion': 1,
        'depth': 14,
        'use_conv': 0,
        'd_conv': 4,
        'lr': 3e-4,
    },
    'minlstm': {
        'dim': 1792,
        'expansion': 1,
        'depth': 31,
        'use_conv': 0,
        'd_conv': 4,
        'lr': 3e-4,
    },
    'mom-e88': {
        'dim': 2048,
        'n_heads': 196,
        'top_k': 48,
        'n_state': 32,
        'depth': 20,
        'lr': 3e-4,
    },
    'e90': {
        'dim': 2048,
        'n_heads': 64,
        'config_idx': 3,
        'depth': 20,
        'use_gate': 1,
        'lr': 3e-4,
    },
    'e1': {
        'dim': 2048,
        'expansion': 2,
        'depth': 20,
        'use_conv': 0,
        'mamba2_init': 0,
        'lr': 3e-4,
    },
    'e23': {
        'dim': 2048,
        'n_slots': 64,
        'expansion': 1.0,
        'depth': 20,
        'w_h_init_scale': 0.9,
        'lr': 3e-4,
    },
    'e42': {
        'dim': 2944,
        'expansion': 1,
        'depth': 19,
        'spectral_radius': 0.999,
        'mamba2_init': 0,
        'lr': 3e-4,
    },
    'e75': {
        'dim': 2048,
        'n_heads': 8,
        'n_state': 32,
        'depth': 20,
        'expansion': 1.0,
        'lr': 3e-4,
    },
}


def estimate_params_for_dim(params, model_type, dim):
    """Estimate actual params for a given dim value."""
    depth = params.get('depth', 20)
    expansion = params.get('expansion', 2)

    if model_type == 'e88':
        return calc_e88_params(dim, depth=depth, n_heads=params.get('n_heads', 96),
                               n_state=params.get('n_state', 32), expansion=params.get('expansion', 1.0),
                               use_gate=True)
    elif model_type == 'fla-gdn':
        return calc_fla_gdn_params(dim, depth=depth, expansion=expansion)
    elif model_type == 'mamba2':
        return calc_mamba2_params(dim, depth=depth, expand=params.get('expand', 2))
    elif model_type == 'transformer':
        return calc_transformer_params(dim, depth=depth, n_heads=params.get('n_heads', 16),
                                       expansion=params.get('expansion', 4))
    elif model_type == 'gru':
        return calc_gru_params(dim, depth=depth, expansion=expansion)
    elif model_type == 'lstm':
        return calc_lstm_params(dim, depth=depth, expansion=expansion)
    elif model_type == 'mingru':
        return calc_mingru_params(dim, depth=depth, expansion=expansion)
    elif model_type == 'minlstm':
        return calc_minlstm_params(dim, depth=depth, expansion=expansion)
    elif model_type == 'mom-e88':
        return calc_mom_e88_params(dim, depth=depth, n_heads=params.get('n_heads', 196),
                                   top_k=params.get('top_k', 48), n_state=params.get('n_state', 32))
    elif model_type == 'e90':
        config_idx = params.get('config_idx', 3)
        k_fast, k_slow = E90_CONFIGS[config_idx]
        return calc_e90_params(dim, depth=depth, n_heads=params.get('n_heads', 64),
                               k_fast=k_fast, k_slow=k_slow)
    elif model_type == 'e1':
        return calc_e1_params(dim, depth=depth, expansion=expansion)
    elif model_type == 'e23':
        return calc_e23_params(dim, depth=depth, expansion=expansion, n_slots=params.get('n_slots', 64))
    elif model_type == 'e42':
        return calc_e42_params(dim, depth=depth, expansion=expansion)
    elif model_type == 'e75':
        return calc_e75_params(dim, depth=depth, n_heads=params.get('n_heads', 8),
                               n_state=params.get('n_state', 32), expansion=params.get('expansion', 1.0))
    else:
        # Rough estimate: ~4 * dim^2 * depth for typical RNNs
        return 4 * dim * dim * depth


def estimate_dim_and_params(params, model_type, target_params):
    """Calculate dim to hit target params, return (dim, actual_params)."""
    if model_type == 'e88':
        n_heads = params['n_heads']
        n_state = params['n_state']
        depth = params['depth']

        # Use accurate calc function from calc_dim.py
        dim, actual_params = find_dim_for_params(
            calc_e88_params,
            target_params,
            n_heads=n_heads,
            n_state=n_state,
            depth=depth,
            expansion=1.0,  # Best E88 config uses expansion=1.0
            use_gate=True   # We use SiLU gating which adds params
        )
        return dim, actual_params

    elif model_type == 'fla-gdn':
        expansion = params['expansion']
        depth = params['depth']

        dim, actual_params = find_dim_for_params(
            calc_fla_gdn_params,
            target_params,
            depth=depth,
            expansion=expansion
        )
        return dim, actual_params

    elif model_type == 'mamba2':
        depth = params['depth']
        expand = params.get('expand', 2)

        dim, actual_params = find_dim_for_params(
            calc_mamba2_params,
            target_params,
            depth=depth,
            expand=expand
        )
        return dim, actual_params

    elif model_type == 'transformer':
        depth = params['depth']
        n_heads = params.get('n_heads', 16)
        expansion = params.get('expansion', 4)

        dim, actual_params = find_dim_for_params(
            calc_transformer_params,
            target_params,
            depth=depth,
            n_heads=n_heads,
            expansion=expansion
        )
        return dim, actual_params

    elif model_type == 'gru':
        depth = params['depth']
        expansion = params.get('expansion', 1)

        dim, actual_params = find_dim_for_params(
            calc_gru_params,
            target_params,
            depth=depth,
            expansion=expansion
        )
        return dim, actual_params

    elif model_type == 'lstm':
        depth = params['depth']
        expansion = params.get('expansion', 1)

        dim, actual_params = find_dim_for_params(
            calc_lstm_params,
            target_params,
            depth=depth,
            expansion=expansion
        )
        return dim, actual_params

    elif model_type == 'mingru':
        depth = params['depth']
        expansion = params.get('expansion', 2)

        dim, actual_params = find_dim_for_params(
            calc_mingru_params,
            target_params,
            depth=depth,
            expansion=expansion
        )
        return dim, actual_params

    elif model_type == 'minlstm':
        depth = params['depth']
        expansion = params.get('expansion', 2)

        dim, actual_params = find_dim_for_params(
            calc_minlstm_params,
            target_params,
            depth=depth,
            expansion=expansion
        )
        return dim, actual_params

    elif model_type == 'mom-e88':
        n_heads = params['n_heads']
        top_k = params['top_k']
        n_state = params['n_state']
        depth = params['depth']

        # top_k doesn't affect params, only compute/state
        dim, actual_params = find_dim_for_params(
            calc_mom_e88_params,
            target_params,
            n_heads=n_heads,
            top_k=top_k,
            n_state=n_state,
            depth=depth,
            expansion=1.0,
            use_gate=True
        )
        return dim, actual_params

    elif model_type == 'e90':
        n_heads = params['n_heads']
        config_idx = params['config_idx']
        k_fast, k_slow = E90_CONFIGS[config_idx]
        depth = params['depth']

        dim, actual_params = find_dim_for_params(
            calc_e90_params,
            target_params,
            n_heads=n_heads,
            k_fast=k_fast,
            k_slow=k_slow,
            depth=depth,
            use_gate=True
        )
        return dim, actual_params

    elif model_type == 'e1':
        depth = params['depth']
        expansion = params.get('expansion', 2)

        dim, actual_params = find_dim_for_params(
            calc_e1_params,
            target_params,
            depth=depth,
            expansion=expansion
        )
        return dim, actual_params

    elif model_type == 'e23':
        depth = params['depth']
        n_slots = params.get('n_slots', 64)

        dim, actual_params = find_dim_for_params(
            calc_e23_params,
            target_params,
            depth=depth,
            n_slots=n_slots
        )
        return dim, actual_params

    elif model_type == 'e42':
        depth = params['depth']
        expansion = params.get('expansion', 1)

        dim, actual_params = find_dim_for_params(
            calc_e42_params,
            target_params,
            depth=depth,
            expansion=expansion
        )
        return dim, actual_params

    elif model_type == 'e75':
        n_heads = params['n_heads']
        n_state = params['n_state']
        depth = params['depth']

        dim, actual_params = find_dim_for_params(
            calc_e75_params,
            target_params,
            n_heads=n_heads,
            n_state=n_state,
            depth=depth,
            expansion=1.0
        )
        return dim, actual_params

    return 1024, target_params


def build_train_command(params, model_type, dim, train_minutes, output_dir, actual_params):
    """Build training command for a configuration."""
    # Adjust batch size based on actual model size (not target)
    if actual_params > 480_000_000:
        batch_size = 8   # 480M+ models need small batch
    elif actual_params > 350_000_000:
        batch_size = 16  # 350-480M models
    else:
        batch_size = 32

    # Use LR from search space (log scale 1e-5 to 1e-3)
    lr = params.get('lr', 3e-4)

    cmd = [
        'python', 'train.py',
        '--data', 'data/pile.txt',
        '--dim', str(dim),
        '--depth', str(params['depth']),
        '--lr', str(lr),
        '--bf16',
        '--batch_size', str(batch_size),
        '--chunk_size', '512',
        '--train_minutes', str(train_minutes),
        '--output', output_dir,
        '--optimizer', 'schedulefree',
        '--seed', '42',
    ]

    if model_type == 'e88':
        use_gate = params.get('use_gate', 1)
        cmd.extend([
            '--level', 'E88',
            '--n_heads', str(params['n_heads']),
            '--n_state', str(params['n_state']),
            '--expansion', str(params.get('expansion', 1.0)),
            '--use_gate', str(use_gate),
        ])
        if use_gate:
            cmd.extend(['--gate_activation', 'silu'])
    elif model_type == 'fla-gdn':
        use_conv = params.get('use_conv', 1)
        cmd.extend([
            '--level', 'fla-gdn',
            '--expansion', str(params['expansion']),
            '--n_heads', str(params.get('n_heads', 16)),
            '--use_conv', str(use_conv),
        ])
    elif model_type == 'mamba2':
        cmd.extend([
            '--level', 'mamba2',
        ])
        # Note: headdim passed via Mamba2 config, not train.py flag
        # Mamba2 calculates nheads from d_inner // headdim internally
    elif model_type == 'transformer':
        dropout = params.get('dropout', 0.0)
        cmd.extend([
            '--level', 'llama',
            '--n_heads', str(params.get('n_heads', 16)),
            '--expansion', str(params.get('expansion', 4)),
            '--dropout', str(dropout),
        ])
    elif model_type == 'gru':
        cmd.extend([
            '--level', 'cudagru',
            '--expansion', str(params.get('expansion', 1)),
        ])
    elif model_type == 'lstm':
        cmd.extend([
            '--level', 'cudalstm',
            '--expansion', str(params.get('expansion', 1)),
        ])
    elif model_type == 'mingru':
        cmd.extend([
            '--level', 'mingru',
            '--expansion', str(params.get('expansion', 2)),
        ])
        if params.get('use_conv', 0):
            cmd.extend(['--use_conv', '1', '--d_conv', str(params.get('d_conv', 4))])
    elif model_type == 'minlstm':
        cmd.extend([
            '--level', 'minlstm',
            '--expansion', str(params.get('expansion', 2)),
        ])
        if params.get('use_conv', 0):
            cmd.extend(['--use_conv', '1', '--d_conv', str(params.get('d_conv', 4))])
    elif model_type == 'mom-e88':
        cmd.extend([
            '--level', 'MoME88',
            '--n_heads', str(params['n_heads']),
            '--top_k', str(params['top_k']),
            '--n_state', str(params['n_state']),
            '--expansion', '1.0',
            '--use_gate', '1',
            '--gate_activation', 'silu',
        ])
    elif model_type == 'e90':
        config_idx = params['config_idx']
        k_fast, k_slow = E90_CONFIGS[config_idx]
        use_gate = params.get('use_gate', 1)
        cmd.extend([
            '--level', 'E90',
            '--n_heads', str(params['n_heads']),
            '--k_fast', str(k_fast),
            '--k_slow', str(k_slow),
            '--use_gate', str(use_gate),
        ])
        if use_gate:
            cmd.extend(['--gate_activation', 'silu'])
    elif model_type == 'e1':
        cmd.extend([
            '--level', '1',  # E1 is level 1 in ladder_lm.py
            '--expansion', str(params.get('expansion', 2)),
        ])
        if params.get('use_conv', 0):
            cmd.extend(['--use_conv', '1'])
        if params.get('mamba2_init', 0):
            cmd.extend(['--mamba2_init', '1'])
    elif model_type == 'e23':
        cmd.extend([
            '--level', '23',  # E23 is level 23 in ladder_lm.py
            '--n_slots', str(params.get('n_slots', 64)),
            '--expansion', str(params.get('expansion', 1.0)),
        ])
        # w_h_init_scale would need to be added to train.py to be passed
    elif model_type == 'e42':
        cmd.extend([
            '--level', '42',  # E42 is level 42 in ladder_lm.py
            '--expansion', str(params.get('expansion', 1)),
        ])
        if params.get('mamba2_init', 0):
            cmd.extend(['--mamba2_init', '1'])
        # spectral_radius would need to be added to train.py to be passed
    elif model_type == 'e75':
        # E75 MultiHead - need to construct the level string
        n_heads = params['n_heads']
        n_state = params['n_state']
        cmd.extend([
            '--level', f'E75h{n_heads}n{n_state}',
            '--expansion', str(params.get('expansion', 1.0)),
        ])

    return cmd


def extract_loss(log_file):
    """Extract last-100 average loss from log file."""
    losses = []
    try:
        with open(log_file, 'r') as f:
            for line in f:
                match = re.search(r'loss\s+([\d.]+)', line)
                if match:
                    loss_val = float(match.group(1))
                    # Skip NaN values
                    if not np.isnan(loss_val) and loss_val < 100:
                        losses.append(loss_val)
    except:
        return float('inf')

    if len(losses) >= 100:
        return sum(losses[-100:]) / 100
    elif len(losses) > 10:
        return sum(losses[-len(losses)//2:]) / (len(losses)//2)
    elif len(losses) > 0:
        return sum(losses) / len(losses)
    return float('inf')


def evaluate_config_worker(args):
    """Worker function for parallel evaluation."""
    x, model_type, train_minutes, gpu_id, work_dir, eval_id, target_params, tolerance = args

    params = decode_params(x, model_type)

    # Use dim from search space if provided, otherwise compute it
    if 'dim' in params:
        dim = params['dim']
        # Estimate actual params with this dim (for logging)
        # This is approximate - actual params depend on model-specific factors
        actual_params = estimate_params_for_dim(params, model_type, dim)
    else:
        dim, actual_params = estimate_dim_and_params(params, model_type, target_params)

    # Skip constraint when dim is directly searchable (we want to explore the full space)
    # Only skip if params are WAY off (e.g., 10x larger than could fit in memory)
    if 'dim' not in params and abs(actual_params - target_params) > tolerance:
        print(f"  [Eval {eval_id}] Skip - params {actual_params/1e6:.1f}M vs target {target_params/1e6:.0f}M")
        return {'eval_id': eval_id, 'loss': 10.0, 'params': params, 'skipped': True}

    output_dir = os.path.join(work_dir, f'eval_{eval_id}')
    log_file = os.path.join(work_dir, f'eval_{eval_id}.log')

    cmd = build_train_command(params, model_type, dim, train_minutes, output_dir, actual_params)

    config_str = ', '.join(f'{k}={v:.4g}' if isinstance(v, float) else f'{k}={v}'
                           for k, v in params.items())
    print(f"  [Eval {eval_id}] GPU {gpu_id} | {config_str} | dim={dim} | {actual_params/1e6:.1f}M params")

    # Run training
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=train_minutes * 60 + 180,  # Extra buffer
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
    except subprocess.TimeoutExpired:
        print(f"  [Eval {eval_id}] Timeout")
        return {'eval_id': eval_id, 'loss': 10.0, 'params': params, 'timeout': True}
    except Exception as e:
        print(f"  [Eval {eval_id}] Error: {e}")
        return {'eval_id': eval_id, 'loss': 10.0, 'params': params, 'error': str(e)}

    loss = extract_loss(log_file)

    # Check for NaN/divergence
    if np.isnan(loss) or np.isinf(loss) or loss > 10:
        print(f"  [Eval {eval_id}] NaN/diverged (loss={loss})")
        return {'eval_id': eval_id, 'loss': 10.0, 'params': params, 'diverged': True}

    print(f"  [Eval {eval_id}] Loss: {loss:.4f}")
    return {'eval_id': eval_id, 'loss': loss, 'params': params, 'dim': dim, 'actual_params': actual_params}


def run_cmaes_search(model_type, generations, train_minutes, gpu_ids, output_dir, target_params, tolerance, start_from_best=False, converge_threshold=None):
    """Run CMA-ES search for optimal configuration.

    Args:
        converge_threshold: If set, stop when best loss improvement < threshold between generations.
                          If None, run for fixed number of generations.
    """
    space = SEARCH_SPACES[model_type]
    n_dims = len(space)
    n_gpus = len(gpu_ids)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize CMA-ES
    if start_from_best and model_type in BEST_CONFIGS:
        x0 = encode_params(BEST_CONFIGS[model_type], model_type)
        sigma0 = 0.15  # Smaller sigma for local refinement around known best
        print(f"Starting from known best config: {BEST_CONFIGS[model_type]}")
        print(f"Encoded x0: {[f'{v:.3f}' for v in x0]}")
    else:
        x0 = [0.5] * n_dims
        sigma0 = 0.3

    # Population size = number of GPUs for parallel eval
    popsize = max(n_gpus, 4)

    # For convergence mode, set a very high max iterations
    max_generations = generations if converge_threshold is None else 1000

    opts = {
        'maxiter': max_generations,
        'popsize': popsize,
        'bounds': [[0] * n_dims, [1] * n_dims],
        'verb_disp': 1,
        'verb_log': 0,
    }

    print("=" * 70)
    print(f"CMA-ES Search for {model_type.upper()}")
    print("=" * 70)
    print(f"Search space ({n_dims} dimensions):")
    for name, (lo, hi, ptype, desc) in space.items():
        print(f"  {name}: [{lo}, {hi}] ({ptype}) - {desc}")
    print(f"Target params: {target_params/1e6:.0f}M ± {tolerance/1e6:.0f}M")
    print(f"Training time per config: {train_minutes} min")
    if converge_threshold is not None:
        print(f"Convergence mode: stop when improvement < {converge_threshold}")
        print(f"Max generations: {max_generations}")
    else:
        print(f"Generations: {generations}")
    print(f"Population: {popsize}")
    print(f"GPUs: {gpu_ids} ({n_gpus} parallel)")
    if converge_threshold is None:
        print(f"Total evaluations: ~{generations * popsize}")
        print(f"Estimated time: ~{generations * train_minutes:.0f} min ({generations * train_minutes / 60:.1f} hours)")
    print("=" * 70)

    # Track best
    eval_count = 0
    best_loss = float('inf')
    best_params = None
    history = []

    # For convergence tracking
    prev_best_loss = float('inf')
    generations_without_improvement = 0

    # Run CMA-ES
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    while not es.stop():
        gen = es.countiter + 1
        if converge_threshold is not None:
            print(f"\n--- Generation {gen} (converge threshold: {converge_threshold}) ---")
        else:
            print(f"\n--- Generation {gen}/{generations} ---")

        solutions = es.ask()

        # Prepare worker arguments
        worker_args = []
        for i, x in enumerate(solutions):
            eval_count += 1
            gpu_id = gpu_ids[i % n_gpus]
            worker_args.append((
                x, model_type, train_minutes, gpu_id,
                output_dir, eval_count, target_params, tolerance
            ))

        # Run in parallel
        fitness = []
        with ProcessPoolExecutor(max_workers=n_gpus) as executor:
            futures = {executor.submit(evaluate_config_worker, args): args[5]
                       for args in worker_args}
            results = {}
            for future in as_completed(futures):
                eval_id = futures[future]
                try:
                    result = future.result()
                    results[result['eval_id']] = result
                except Exception as e:
                    print(f"  [Eval {eval_id}] Worker failed: {e}")
                    results[eval_id] = {'eval_id': eval_id, 'loss': 10.0, 'error': str(e)}

        # Collect fitness in order
        for args in worker_args:
            eval_id = args[5]
            result = results.get(eval_id, {'loss': 10.0})
            loss = result['loss']
            fitness.append(loss)
            history.append(result)

            if loss < best_loss:
                best_loss = loss
                best_params = result.get('params', decode_params(args[0], model_type))
                best_dim = result.get('dim', 0)
                best_actual = result.get('actual_params', 0)
                print(f"  *** NEW BEST: {loss:.4f} | {best_params} | dim={best_dim} ***")

        es.tell(solutions, fitness)
        es.disp()

        # Convergence check
        if converge_threshold is not None:
            improvement = prev_best_loss - best_loss
            print(f"  Generation best: {min(fitness):.4f} | Overall best: {best_loss:.4f} | Improvement: {improvement:.4f}")

            if improvement < converge_threshold and improvement >= 0:
                generations_without_improvement += 1
                print(f"  Improvement {improvement:.4f} < threshold {converge_threshold} ({generations_without_improvement} consecutive)")
                if generations_without_improvement >= 2:  # Stop after 2 consecutive generations without sufficient improvement
                    print(f"\n*** CONVERGED: improvement {improvement:.4f} < {converge_threshold} ***")
                    break
            else:
                generations_without_improvement = 0

            prev_best_loss = best_loss

        # Save checkpoint
        checkpoint = {
            'generation': gen,
            'best_loss': best_loss,
            'best_params': best_params,
            'converge_threshold': converge_threshold,
            'history': history[-100:],  # Keep last 100 evals
        }
        with open(os.path.join(output_dir, 'checkpoint.json'), 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)

    # Final results
    print(f"\n" + "=" * 70)
    print(f"SEARCH COMPLETE")
    print("=" * 70)
    print(f"Best loss: {best_loss:.4f}")
    print(f"Best params: {best_params}")
    print(f"Total evaluations: {eval_count}")

    # Save final results
    results = {
        'model_type': model_type,
        'target_params': target_params,
        'best_loss': best_loss,
        'best_params': best_params,
        'total_evals': eval_count,
        'history': history,
    }
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Cleanup: delete checkpoint files to save disk space (keep logs and results)
    import glob
    pt_files = glob.glob(os.path.join(output_dir, '**', '*.pt'), recursive=True)
    if pt_files:
        print(f"\nCleaning up {len(pt_files)} checkpoint files...")
        for pt_file in pt_files:
            try:
                os.remove(pt_file)
            except OSError:
                pass
        # Remove empty directories
        for root, dirs, files in os.walk(output_dir, topdown=False):
            for d in dirs:
                dir_path = os.path.join(root, d)
                try:
                    if not os.listdir(dir_path):
                        os.rmdir(dir_path)
                except OSError:
                    pass
        print("Cleanup complete.")

    return best_params, best_loss


def main():
    parser = argparse.ArgumentParser(description='CMA-ES search for optimal model config')
    parser.add_argument('--model', type=str, required=True,
                        choices=['e88', 'fla-gdn', 'mamba2', 'transformer', 'gru', 'lstm', 'mingru', 'minlstm', 'mom-e88', 'e90', 'e1', 'e23', 'e42', 'e75'],
                        help='Model type to optimize')
    parser.add_argument('--generations', type=int, default=20,
                        help='Number of CMA-ES generations (max if using --converge)')
    parser.add_argument('--train_minutes', type=float, default=2,
                        help='Training time per config (minutes)')
    parser.add_argument('--converge', type=float, default=None,
                        help='Convergence threshold: stop when improvement < this value (e.g., 0.01)')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Comma-separated GPU IDs (e.g., 0,1,2,3)')
    parser.add_argument('--params', type=str, default='500M',
                        help='Target parameters (e.g., 100M, 500M)')
    parser.add_argument('--tolerance', type=str, default='50M',
                        help='Parameter tolerance (e.g., 50M)')
    parser.add_argument('--output', type=str, default='benchmark_results/cmaes_search',
                        help='Output directory')
    parser.add_argument('--start_from_best', action='store_true',
                        help='Start search from known best config (for local refinement)')
    args = parser.parse_args()

    # Parse GPU IDs
    gpu_ids = [int(g.strip()) for g in args.gpus.split(',')]

    # Parse target params
    def parse_size(s):
        s = s.strip().upper()
        if s.endswith('M'):
            return int(float(s[:-1]) * 1_000_000)
        elif s.endswith('B'):
            return int(float(s[:-1]) * 1_000_000_000)
        return int(s)

    target_params = parse_size(args.params)
    tolerance = parse_size(args.tolerance)

    # Output directory name reflects convergence mode
    if args.converge is not None:
        output_dir = os.path.join(
            args.output,
            f'{args.model}_{args.params}_converge{args.converge}_{time.strftime("%Y%m%d_%H%M%S")}'
        )
    else:
        output_dir = os.path.join(
            args.output,
            f'{args.model}_{args.params}_{args.generations}gen_{time.strftime("%Y%m%d_%H%M%S")}'
        )

    best_params, best_loss = run_cmaes_search(
        args.model,
        args.generations,
        args.train_minutes,
        gpu_ids,
        output_dir,
        target_params,
        tolerance,
        start_from_best=args.start_from_best,
        converge_threshold=args.converge
    )

    print(f"\nTo train with best config:")
    # Use dim from search if available, otherwise compute it
    if 'dim' in best_params:
        dim = best_params['dim']
    else:
        dim, _ = estimate_dim_and_params(best_params, args.model, target_params)
    lr = best_params.get('lr', 3e-4)
    if args.model == 'e88':
        print(f"python train.py --level E88 --dim {dim} --n_heads {best_params['n_heads']} "
              f"--n_state {best_params['n_state']} --depth {best_params['depth']} "
              f"--lr {lr} --expansion {best_params.get('expansion', 1.0)} --use_gate 1 --gate_activation silu --train_minutes 30")
    elif args.model == 'fla-gdn':
        print(f"python train.py --level fla-gdn --dim {dim} --depth {best_params['depth']} "
              f"--expansion {best_params['expansion']} --n_heads {best_params.get('n_heads', 16)} "
              f"--lr {lr} --train_minutes 30")
    elif args.model == 'mamba2':
        print(f"python train.py --level mamba2 --dim {dim} --depth {best_params['depth']} "
              f"--lr {lr} --train_minutes 30")
    elif args.model == 'transformer':
        print(f"python train.py --level llama --dim {dim} --depth {best_params['depth']} "
              f"--n_heads {best_params.get('n_heads', 16)} --expansion {best_params.get('expansion', 4)} "
              f"--lr {lr} --train_minutes 30")
    elif args.model in ['gru', 'lstm']:
        level = 'cudagru' if args.model == 'gru' else 'cudalstm'
        print(f"python train.py --level {level} --dim {dim} --depth {best_params['depth']} "
              f"--expansion {best_params.get('expansion', 1)} --lr {lr} --train_minutes 30")
    elif args.model in ['mingru', 'minlstm']:
        print(f"python train.py --level {args.model} --dim {dim} --depth {best_params['depth']} "
              f"--expansion {best_params.get('expansion', 2)} --lr {lr} --train_minutes 30")
    elif args.model == 'mom-e88':
        print(f"python train.py --level MoME88 --dim {dim} --n_heads {best_params['n_heads']} "
              f"--top_k {best_params['top_k']} --n_state {best_params['n_state']} "
              f"--depth {best_params['depth']} --lr {lr} --expansion 1.0 --use_gate 1 "
              f"--gate_activation silu --train_minutes 30")
    elif args.model == 'e90':
        config_idx = best_params['config_idx']
        k_fast, k_slow = E90_CONFIGS[config_idx]
        print(f"python train.py --level E90 --dim {dim} --n_heads {best_params['n_heads']} "
              f"--k_fast {k_fast} --k_slow {k_slow} "
              f"--depth {best_params['depth']} --lr {lr} --use_gate 1 "
              f"--gate_activation silu --train_minutes 30")
    elif args.model == 'e1':
        print(f"python train.py --level 1 --dim {dim} --depth {best_params['depth']} "
              f"--expansion {best_params.get('expansion', 2)} --lr {lr} --train_minutes 30")
    elif args.model == 'e23':
        print(f"python train.py --level 23 --dim {dim} --depth {best_params['depth']} "
              f"--n_slots {best_params.get('n_slots', 64)} --lr {lr} --train_minutes 30")
    elif args.model == 'e42':
        print(f"python train.py --level 42 --dim {dim} --depth {best_params['depth']} "
              f"--expansion {best_params.get('expansion', 1)} --lr {lr} --train_minutes 30")
    elif args.model == 'e75':
        print(f"python train.py --level E75h{best_params['n_heads']}n{best_params['n_state']} "
              f"--dim {dim} --depth {best_params['depth']} --lr {lr} --expansion {best_params.get('expansion', 1.0)} --train_minutes 30")


if __name__ == '__main__':
    main()
