#!/usr/bin/env python3
"""
CMA-ES v2: Improved Architecture/Hyperparameter Search

Improvements over v1:
1. Two-Phase Search: LHS exploration → CMA-ES refinement
2. Discrete parameter sweep (e.g., run separate search per n_state value)
3. Looser convergence: min_generations=6, consecutive=3, threshold=0.005
4. Larger sigma (0.35) for better exploration
5. Population of 16 (2 batches per generation)

Usage:
    # Full two-phase search
    python cmaes_search_v2.py --model e88 --train_minutes 30 --gpus 0,1,2,3,4,5,6,7

    # LHS exploration only (phase 1)
    python cmaes_search_v2.py --model e88 --phase lhs --lhs_samples 48

    # CMA-ES refinement only (phase 2, from existing LHS results)
    python cmaes_search_v2.py --model e88 --phase cmaes --warm_start_from results.json

    # Discrete parameter sweep (separate search per n_state)
    python cmaes_search_v2.py --model e88 --sweep_discrete n_state
"""

import os
import sys
import argparse
import subprocess
import json
import re
import shutil
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import glob
from datetime import datetime

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
    calc_e1_params, calc_e1h_params, calc_e23_params, calc_e42_params, calc_e75_params
)

# Supported n_state values for E88
E88_SUPPORTED_N_STATE = [16, 32]

# Global compile settings (set from args in main())
COMPILE_ENABLED = False
COMPILE_MODE = 'max-autotune'

# Global sequence length setting (set from args in main())
CHUNK_SIZE = 512

# Global long-sequence settings (set from args in main())
GRADIENT_CHECKPOINTING = False
PROJECTION_CHUNK_SIZE = 0

# Progressive training settings (set from args in main())
PROGRESSIVE = False
PHASE1_MINUTES = 10
PHASE2_MINUTES = 10
PHASE2_CHUNK_SIZE = 32768



# Models that need gradient checkpointing at long context
PHASE2_GRAD_CKPT_MODELS = {'e88', 'e88_fused', 'e1h'}

# Models that need projection chunking at long context
PHASE2_PROJ_CHUNK_MODELS = {'e88', 'e88_fused'}

# Known good configs from previous runs - inject into LHS to ensure exploration around them
# These configs are validated for 480M±10% with use_gate=True
# BEST FINDING: narrow dim + many heads + deep works better than wide + shallow
KNOWN_GOOD_CONFIGS = {
    'e88': {
        16: [  # n_state=16: 32K progressive CMA-ES (Mar 2026) + 512 CMA-ES seeds
            {'dim': 1152, 'n_heads': 269, 'depth': 18, 'lr': 9.07e-4},   # 32K best: 1.1000
            {'dim': 1280, 'n_heads': 152, 'depth': 28, 'lr': 1.70e-3},   # 32K #2: 1.1086
            {'dim': 1024, 'n_heads': 262, 'depth': 20, 'lr': 4.96e-4},   # 32K #3: 1.1093
            {'dim': 2560, 'n_heads': 153, 'depth': 17, 'lr': 0.0006},    # 512 CMA-ES best: 1.2794
        ],
        32: [  # n_state=32: 32K progressive CMA-ES (Mar 2026) — sub-1.0 nats!
            {'dim': 1152, 'n_heads': 60, 'depth': 39, 'lr': 1.21e-3},    # 32K best: 0.9858
            {'dim': 1024, 'n_heads': 63, 'depth': 46, 'lr': 1.54e-3},    # 32K #2: 0.9971
            {'dim': 1280, 'n_heads': 60, 'depth': 42, 'lr': 2.53e-3},    # 32K #3: 1.0183
            {'dim': 1024, 'n_heads': 62, 'depth': 47, 'lr': 6.09e-4},    # 32K #4: 1.0248
        ],
    },
    # e88_fused uses same configs as e88 (fused kernel, same semantics)
    'e88_fused': {
        16: [
            {'dim': 1152, 'n_heads': 269, 'depth': 18, 'lr': 9.07e-4},
            {'dim': 1280, 'n_heads': 152, 'depth': 28, 'lr': 1.70e-3},
            {'dim': 1024, 'n_heads': 262, 'depth': 20, 'lr': 4.96e-4},
            {'dim': 2560, 'n_heads': 153, 'depth': 17, 'lr': 0.0006},
        ],
        32: [
            {'dim': 1152, 'n_heads': 60, 'depth': 39, 'lr': 1.21e-3},
            {'dim': 1024, 'n_heads': 63, 'depth': 46, 'lr': 1.54e-3},
            {'dim': 1280, 'n_heads': 60, 'depth': 42, 'lr': 2.53e-3},
            {'dim': 1024, 'n_heads': 62, 'depth': 47, 'lr': 6.09e-4},
        ],
    },
    'fla-gdn': {
        None: [  # 32K progressive CMA-ES (Mar 2026)
            {'dim': 1664, 'expansion': 3, 'depth': 13, 'n_heads': 12, 'lr': 7.68e-4},  # 32K best: 1.1345
            {'dim': 1408, 'expansion': 3, 'depth': 20, 'n_heads': 15, 'lr': 6.11e-4},  # 32K #2: 1.1359
            {'dim': 1664, 'expansion': 3, 'depth': 13, 'n_heads': 8, 'lr': 7.54e-4},   # 32K #3: 1.1370
            {'dim': 1664, 'expansion': 3, 'depth': 15, 'n_heads': 10, 'lr': 6.68e-4},  # 32K #4: 1.1379
        ],
    },
    'mamba2': {
        None: [  # 32K progressive CMA-ES (Mar 2026) + 512 CMA-ES seeds
            {'dim': 1920, 'd_state': 208, 'expand': 2, 'depth': 21, 'lr': 3.12e-4},    # 32K best: 1.1882
            {'dim': 2176, 'd_state': 192, 'expand': 1, 'depth': 34, 'lr': 3.26e-4},    # 32K #2: 1.1885
            {'dim': 2176, 'd_state': 208, 'expand': 1, 'depth': 34, 'lr': 2.98e-4},    # 32K #3: 1.1895
            {'dim': 1664, 'd_state': 176, 'expand': 2, 'depth': 26, 'lr': 3.38e-4},    # 32K #4: 1.1938
        ],
    },
    'e1h': {
        16: [  # 32K progressive CMA-ES (Mar 2026)
            {'dim': 1408, 'n_heads': 184, 'depth': 40, 'lr': 6.54e-4},   # 32K best: 1.2542
            {'dim': 2304, 'n_heads': 191, 'depth': 23, 'lr': 4.15e-4},   # 32K #2: 1.2587
            {'dim': 1920, 'n_heads': 192, 'depth': 26, 'lr': 4.93e-4},   # 32K #3: 1.2642
            {'dim': 1280, 'n_heads': 204, 'depth': 39, 'lr': 5.26e-4},   # 32K #4: 1.2664
        ],
        32: [  # 32K progressive CMA-ES (Mar 2026)
            {'dim': 2048, 'n_heads': 74, 'depth': 35, 'lr': 3.69e-4},   # 32K best: 1.2322
            {'dim': 2688, 'n_heads': 44, 'depth': 39, 'lr': 6.03e-4},   # 32K #2: 1.2410
            {'dim': 2304, 'n_heads': 57, 'depth': 40, 'lr': 9.85e-4},   # 32K #3: 1.2641
            {'dim': 1792, 'n_heads': 159, 'depth': 17, 'lr': 5.18e-4},  # 32K #4: 1.2717
        ],
    },
}

# E90 valid (k_fast, k_slow) configurations
E90_CONFIGS = [
    (8, 16), (8, 24), (16, 32), (16, 48),
]

# =============================================================================
# SEARCH SPACES - 6D for all models
# =============================================================================

# E88 base search space (shared by ablation variants)
_E88_SEARCH_SPACE = {
    'dim': (1024, 3072, 'int_mult128', 'Model dimension'),
    'n_heads': (32, 400, 'int', 'Number of attention heads'),  # Expanded to 400 - n16 wants many small heads
    'n_state': (16, 64, 'e88_n_state', 'State dimension (16,32,48,64)'),
    'depth': (10, 50, 'int', 'Number of layers'),  # Expanded from 40 - deep networks work with many heads
    'lr': (1e-4, 3e-3, 'log', 'Learning rate'),  # Raised upper bound - models can handle higher LR
}  # 5D (n_state swept separately, batch_size probed via memory)

SEARCH_SPACES = {
    # Clean 5D/4D search spaces - no binary params (CMA-ES handles continuous better)
    'e88': _E88_SEARCH_SPACE,  # baseline: use_gate=1, linear_state=0
    'e88_fused': _E88_SEARCH_SPACE,  # E88 with fused CUDA kernel (faster training)
    'e88-linear': _E88_SEARCH_SPACE,  # ablation: remove tanh (linear_state=1)
    'e88-nogate': _E88_SEARCH_SPACE,  # ablation: remove gating (use_gate=0)
    'e88-minimal': _E88_SEARCH_SPACE,  # ablation: remove both
    'e88-wgate': _E88_SEARCH_SPACE,  # ablation: add write gate (beta) like FLA-GDN
    'fla-gdn': {
        'dim': (1024, 3072, 'int_mult128', 'Model dimension'),
        'expansion': (1, 3, 'int', 'Value expansion factor'),
        'depth': (10, 40, 'int', 'Number of layers'),
        'n_heads': (8, 32, 'int', 'Number of heads'),
        'lr': (1e-4, 3e-3, 'log', 'Learning rate'),
        },  # 6D
    'mamba2': {
        'dim': (1024, 3072, 'int_mult128', 'Model dimension'),
        'd_state': (64, 256, 'int_mult16', 'SSM state dimension'),
        'expand': (1, 3, 'int', 'Expansion factor'),
        'depth': (10, 40, 'int', 'Number of layers'),
        'lr': (1e-4, 3e-3, 'log', 'Learning rate'),
        },  # 6D
    'transformer': {
        'dim': (1024, 3072, 'int_mult128', 'Model dimension'),
        'n_heads': (8, 32, 'int', 'Number of attention heads'),
        'expansion': (2, 6, 'int', 'FFN expansion factor'),
        'depth': (10, 40, 'int', 'Number of layers'),
        'lr': (1e-4, 3e-3, 'log', 'Learning rate'),
        },  # 6D
    'mingru': {
        'dim': (1024, 3584, 'int_mult128', 'Model dimension'),
        'expansion': (1, 4, 'int', 'Expansion factor'),
        'depth': (10, 40, 'int', 'Number of layers'),
        'lr': (1e-4, 3e-3, 'log', 'Learning rate'),
        },  # 5D
    'minlstm': {
        'dim': (1024, 3584, 'int_mult128', 'Model dimension'),
        'expansion': (1, 4, 'int', 'Expansion factor'),
        'depth': (10, 40, 'int', 'Number of layers'),
        'lr': (1e-4, 3e-3, 'log', 'Learning rate'),
        },  # 5D
    'e1': {
        'dim': (1024, 3072, 'int_mult128', 'Model dimension'),
        'expansion': (1, 3, 'int', 'Expansion factor'),
        'depth': (10, 40, 'int', 'Number of layers'),
        'lr': (1e-4, 3e-3, 'log', 'Learning rate'),
        },  # 5D
    'e23': {
        'dim': (1024, 3072, 'int_mult128', 'Model dimension'),
        'n_slots': (32, 128, 'int', 'Number of tape memory slots'),
        'expansion': (1, 3, 'int', 'Expansion factor'),
        'depth': (10, 40, 'int', 'Number of layers'),
        'lr': (1e-4, 3e-3, 'log', 'Learning rate'),
        },  # 6D
    'e42': {
        'dim': (1024, 3584, 'int_mult128', 'Model dimension'),
        'expansion': (1, 3, 'int', 'Expansion factor'),
        'depth': (10, 40, 'int', 'Number of layers'),
        'spectral_radius': (0.9, 0.999, 'float', 'Spectral radius'),
        'lr': (1e-4, 3e-3, 'log', 'Learning rate'),
        },  # 6D
    'e75': {
        'dim': (1024, 3072, 'int_mult128', 'Model dimension'),
        'n_heads': (4, 32, 'int', 'Number of heads'),
        'n_state': (16, 64, 'int_mult8', 'State dimension'),
        'depth': (10, 40, 'int', 'Number of layers'),
        'lr': (1e-4, 3e-3, 'log', 'Learning rate'),
        },  # 6D
    'e1h': {
        'dim': (1024, 3584, 'int_mult128', 'Model dimension'),
        'n_heads': (16, 400, 'int', 'Number of independent Elman heads'),
        'n_state': (16, 64, 'e88_n_state', 'Per-head state dimension'),
        'depth': (10, 40, 'int', 'Number of layers'),
        'lr': (1e-4, 3e-3, 'log', 'Learning rate'),
        },  # 6D (n_state swept separately like E88)
}

# Discrete parameters that benefit from sweep (instead of CMA-ES interpolation)
DISCRETE_SWEEP_PARAMS = {
    'e88': {'n_state': [16, 32]},
    'e88_fused': {'n_state': [16, 32]},  # fused CUDA kernel variant
    'e88-linear': {'n_state': [16, 32]},  # ablation: remove tanh
    'e88-nogate': {'n_state': [16, 32]},  # ablation: remove gating
    'e88-minimal': {'n_state': [16, 32]},  # ablation: remove both
    'e88-wgate': {'n_state': [16, 32]},  # ablation: add write gate
    'e75': {'n_state': [16, 24, 32, 40, 48, 56, 64]},
    'e1h': {'n_state': [16, 32]},
}

# =============================================================================
# PARAMETER CONVERSION
# =============================================================================
def get_search_space(model_type, fixed_params=None):
    """Get search space with n_state-dependent bounds for E88."""
    space = SEARCH_SPACES[model_type].copy()
    fixed_params = fixed_params or {}

    # Adjust n_heads range for E88 based on n_state
    if model_type.startswith('e88') and 'n_heads' in space:
        n_state = fixed_params.get('n_state')
        if n_state == 16:
            # n_state=16 wants many small heads: 96-400
            space['n_heads'] = (96, 400, 'int', 'Number of attention heads (n16: many small)')
        elif n_state == 32:
            # n_state=32 prefers fewer larger heads: 32-160
            space['n_heads'] = (32, 160, 'int', 'Number of attention heads (n32: fewer large)')
        # else: use default range

    # Adjust n_heads range for E1H based on n_state
    if model_type == 'e1h' and 'n_heads' in space:
        n_state = fixed_params.get('n_state')
        if n_state == 16:
            space['n_heads'] = (64, 400, 'int', 'Number of Elman heads (n16: many small)')
        elif n_state == 32:
            space['n_heads'] = (32, 200, 'int', 'Number of Elman heads (n32: fewer large)')

    return space


def decode_params(x, model_type, fixed_params=None):
    """Convert CMA-ES vector [0,1]^n to model parameters."""
    space = get_search_space(model_type, fixed_params)
    params = {}
    fixed_params = fixed_params or {}

    x_idx = 0
    for name, (lo, hi, ptype, desc) in space.items():
        # Use fixed value if provided
        if name in fixed_params:
            params[name] = fixed_params[name]
            continue

        val = np.clip(x[x_idx], 0, 1)
        x_idx += 1

        if ptype == 'int':
            params[name] = int(round(lo + val * (hi - lo)))
        elif ptype == 'binary':
            params[name] = 1 if val >= 0.5 else 0
        elif ptype == 'int_mult16':
            raw = lo + val * (hi - lo)
            params[name] = max(16, int(round(raw / 16) * 16))
        elif ptype == 'int_mult8':
            raw = lo + val * (hi - lo)
            params[name] = max(8, int(round(raw / 8) * 8))
        elif ptype == 'int_mult128':
            raw = lo + val * (hi - lo)
            params[name] = max(128, int(round(raw / 128) * 128))
        elif ptype == 'int_pow2':
            raw = lo + val * (hi - lo)
            powers = [p for p in [16, 32, 64, 128, 256] if lo <= p <= hi]
            params[name] = min(powers, key=lambda p: abs(p - raw)) if powers else int(round(raw))
        elif ptype == 'e88_n_state':
            raw = lo + val * (hi - lo)
            params[name] = min(E88_SUPPORTED_N_STATE, key=lambda x: abs(x - raw))
        elif ptype == 'log':
            log_lo, log_hi = np.log10(lo), np.log10(hi)
            params[name] = 10 ** (log_lo + val * (log_hi - log_lo))
        else:  # float
            params[name] = lo + val * (hi - lo)

    return params


def encode_params(params, model_type, fixed_params=None):
    """Convert model parameters to CMA-ES vector [0,1]^n."""
    space = get_search_space(model_type, fixed_params)
    fixed_params = fixed_params or {}
    x = []

    for name, (lo, hi, ptype, desc) in space.items():
        if name in fixed_params:
            continue  # Skip fixed params

        val = params.get(name, (lo + hi) / 2)

        if ptype == 'binary':
            x_val = 0.75 if val else 0.25
        elif ptype == 'log':
            log_lo, log_hi = np.log10(lo), np.log10(hi)
            x_val = (np.log10(val) - log_lo) / (log_hi - log_lo)
        else:
            x_val = (val - lo) / (hi - lo)

        x.append(np.clip(x_val, 0, 1))

    return x


def get_search_dim(model_type, fixed_params=None):
    """Get number of dimensions to search (excluding fixed params)."""
    fixed_params = fixed_params or {}
    return len(get_search_space(model_type, fixed_params)) - len(fixed_params)


# =============================================================================
# LATIN HYPERCUBE SAMPLING
# =============================================================================
def latin_hypercube_sample(n_samples, n_dims, seed=None):
    """Generate Latin Hypercube samples in [0,1]^n_dims."""
    rng = np.random.default_rng(seed)

    # Create intervals
    samples = np.zeros((n_samples, n_dims))
    for d in range(n_dims):
        # Divide [0,1] into n_samples equal intervals
        intervals = np.linspace(0, 1, n_samples + 1)
        # Sample one point from each interval
        for i in range(n_samples):
            samples[i, d] = rng.uniform(intervals[i], intervals[i + 1])
        # Shuffle to break correlation
        rng.shuffle(samples[:, d])

    return samples


def generate_lhs_configs(model_type, n_samples, fixed_params=None, seed=42):
    """Generate LHS configurations for a model."""
    n_dims = get_search_dim(model_type, fixed_params)
    samples = latin_hypercube_sample(n_samples, n_dims, seed=seed)

    configs = []
    for i in range(n_samples):
        params = decode_params(samples[i], model_type, fixed_params)
        configs.append(params)

    return configs


# =============================================================================
# PARAM ESTIMATION AND DIM CALCULATION
# =============================================================================
def estimate_params_for_config(params, model_type):
    """Estimate parameter count for a configuration."""
    dim = params.get('dim', 1024)
    depth = params.get('depth', 20)

    if model_type in ('e88', 'e88_fused', 'e88-linear', 'e88-nogate', 'e88-minimal', 'e88-wgate'):
        # All E88 variants have ~same param count (ablations only affect computation, not params)
        # Note: e88-wgate adds small write_gate_proj (dim -> n_heads) but negligible
        use_gate = model_type not in ('e88-nogate', 'e88-minimal')
        return calc_e88_params(dim, depth=depth, n_heads=params.get('n_heads', 96),
                               n_state=params.get('n_state', 32),
                               expansion=params.get('expansion', 1.0), use_gate=use_gate)
    elif model_type == 'fla-gdn':
        return calc_fla_gdn_params(dim, depth=depth, expansion=params.get('expansion', 2))
    elif model_type == 'mamba2':
        return calc_mamba2_params(dim, depth=depth, expand=params.get('expand', 2),
                                    d_state=params.get('d_state', 64))
    elif model_type == 'transformer':
        return calc_transformer_params(dim, depth=depth, n_heads=params.get('n_heads', 16),
                                       expansion=params.get('expansion', 4))
    elif model_type == 'mingru':
        return calc_mingru_params(dim, depth=depth, expansion=params.get('expansion', 2))
    elif model_type == 'minlstm':
        return calc_minlstm_params(dim, depth=depth, expansion=params.get('expansion', 2))
    elif model_type == 'e1':
        return calc_e1_params(dim, depth=depth, expansion=params.get('expansion', 2))
    elif model_type == 'e23':
        return calc_e23_params(dim, depth=depth, n_slots=params.get('n_slots', 64))
    elif model_type == 'e42':
        return calc_e42_params(dim, depth=depth, expansion=params.get('expansion', 2))
    elif model_type == 'e75':
        return calc_e75_params(dim, depth=depth, n_heads=params.get('n_heads', 8),
                               n_state=params.get('n_state', 32), expansion=params.get('expansion', 1.0))
    elif model_type == 'e1h':
        return calc_e1h_params(dim, depth=depth, n_heads=params.get('n_heads', 16),
                               n_state=params.get('n_state', 32))
    else:
        return 4 * dim * dim * depth  # Rough estimate


def is_valid_param_count(params, model_type, target_params, tolerance=0.10):
    """Check if config is within tolerance of target params (default ±10%)."""
    actual = estimate_params_for_config(params, model_type)
    return abs(actual - target_params) / target_params <= tolerance


# =============================================================================
# TRAINING COMMAND BUILDER
# =============================================================================
def build_train_command(params, model_type, train_minutes, output_dir):
    """Build training command for a configuration."""
    dim = params['dim']
    actual_params = estimate_params_for_config(params, model_type)

    # Batch size: set by memory probing (not a CMA-ES search dimension)
    batch_size = params.get('batch_size', 16)

    lr = params.get('lr', 3e-4)

    cmd = [
        'python', 'train.py',
        '--data', '/mnt/nvme1n1/erikg/comma_v0.1_training_dataset/commapile.txt',
        '--dim', str(dim),
        '--depth', str(params['depth']),
        '--lr', str(lr),
        '--bf16',
        '--batch_size', str(batch_size),
        '--chunk_size', str(CHUNK_SIZE),
        '--train_minutes', str(train_minutes),
        '--output', output_dir,
        '--optimizer', 'schedulefree',
        '--seed', '42',
        '--save_every', '999999',  # Only save final checkpoint
        '--keep_checkpoints', '1',  # Keep final checkpoint for top-3 retention
    ]

    # Add torch.compile if enabled (global settings)
    if COMPILE_ENABLED:
        cmd.extend(['--compile', '--compile_mode', COMPILE_MODE])

    # Add long-sequence options
    if GRADIENT_CHECKPOINTING:
        cmd.append('--gradient_checkpointing')
    if PROJECTION_CHUNK_SIZE > 0:
        cmd.extend(['--projection_chunk_size', str(PROJECTION_CHUNK_SIZE)])

    if model_type == 'e88':
        cmd.extend([
            '--level', 'E88',
            '--n_heads', str(params['n_heads']),
            '--n_state', str(params['n_state']),
            '--expansion', '1.0',  # Fixed - E88 requires square state
            '--use_gate', '1',  # Gate enabled - best result (0.8272) was WITH gate
            '--gate_activation', 'silu',  # SiLU gating
        ])

    elif model_type == 'e88_fused':
        # E88 with fused CUDA kernel (faster training, same semantics)
        cmd.extend([
            '--level', 'e88_fused',
            '--n_heads', str(params['n_heads']),
            '--n_state', str(params['n_state']),
            '--expansion', '1.0',
            '--use_gate', '1',
            '--gate_activation', 'silu',
        ])

    elif model_type == 'e88-linear':
        # Ablation: remove tanh (linear state update)
        cmd.extend([
            '--level', 'E88',
            '--n_heads', str(params['n_heads']),
            '--n_state', str(params['n_state']),
            '--expansion', '1.0',
            '--use_gate', '1',
            '--gate_activation', 'silu',
            '--linear_state', '1',  # ABLATION: linear state (no tanh)
        ])

    elif model_type == 'e88-nogate':
        # Ablation: remove gating
        cmd.extend([
            '--level', 'E88',
            '--n_heads', str(params['n_heads']),
            '--n_state', str(params['n_state']),
            '--expansion', '1.0',
            '--use_gate', '0',  # ABLATION: no gating
        ])

    elif model_type == 'e88-minimal':
        # Ablation: remove both tanh and gating
        cmd.extend([
            '--level', 'E88',
            '--n_heads', str(params['n_heads']),
            '--n_state', str(params['n_state']),
            '--expansion', '1.0',
            '--use_gate', '0',  # ABLATION: no gating
            '--linear_state', '1',  # ABLATION: linear state (no tanh)
        ])

    elif model_type == 'e88-wgate':
        # Ablation: add write gate (like FLA-GDN's beta gate on delta)
        cmd.extend([
            '--level', 'E88',
            '--n_heads', str(params['n_heads']),
            '--n_state', str(params['n_state']),
            '--expansion', '1.0',
            '--use_gate', '1',
            '--gate_activation', 'silu',
            '--use_write_gate', '1',  # ABLATION: add write gate (FLA-GDN beta style)
        ])

    elif model_type == 'fla-gdn':
        cmd.extend([
            '--level', 'fla-gdn',
            '--expansion', str(params['expansion']),
            '--n_heads', str(params.get('n_heads', 16)),
        ])

    elif model_type == 'mamba2':
        cmd.extend([
            '--level', 'mamba2',
            '--mamba_d_state', str(params.get('d_state', 64)),
            '--mamba_expand', str(params.get('expand', 2)),
        ])

    elif model_type == 'transformer':
        cmd.extend([
            '--level', 'llama',
            '--n_heads', str(params.get('n_heads', 16)),
            '--expansion', str(params.get('expansion', 4)),
        ])

    elif model_type == 'mingru':
        cmd.extend([
            '--level', 'mingru',
            '--expansion', str(params.get('expansion', 2)),
        ])

    elif model_type == 'minlstm':
        cmd.extend([
            '--level', 'minlstm',
            '--expansion', str(params.get('expansion', 2)),
        ])

    elif model_type == 'e1':
        cmd.extend([
            '--level', '1',  # Integer level - parsed by train.py's parse_level()
            '--expansion', str(params.get('expansion', 2)),
        ])

    elif model_type == 'e23':
        cmd.extend([
            '--level', '23',  # Integer level - parsed by train.py's parse_level()
            '--n_slots', str(params.get('n_slots', 64)),
            '--expansion', str(params.get('expansion', 1)),
        ])

    elif model_type == 'e42':
        cmd.extend([
            '--level', '42',  # Integer level - parsed by train.py's parse_level()
            '--expansion', str(params.get('expansion', 2)),
        ])

    elif model_type == 'e75':
        cmd.extend([
            '--level', 'E75h{n}n{s}'.format(n=params.get('n_heads', 8), s=params.get('n_state', 32)),
            '--n_heads', str(params.get('n_heads', 8)),
            '--n_state', str(params.get('n_state', 32)),
        ])

    elif model_type == 'e1h':
        cmd.extend([
            '--level', 'E1H',
            '--n_heads', str(params.get('n_heads', 16)),
            '--n_state', str(params.get('n_state', 32)),
        ])

    return cmd, actual_params


# =============================================================================
# EVALUATION
# =============================================================================

def _is_oom(result):
    """Check if a subprocess result indicates CUDA OOM."""
    return (result.returncode != 0 and
            ('CUDA out of memory' in result.stderr or
             'OutOfMemoryError' in result.stderr))


def _probe_memory(cmd_no_bs, bs, env, cwd):
    """Run train.py --probe_memory at given batch size, return peak memory in MB or None on failure."""
    cmd = cmd_no_bs + ['--batch_size', str(bs), '--probe_memory']
    # Remove --train_minutes since probe exits after 1 step
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=120, env=env, cwd=cwd)
    except (subprocess.TimeoutExpired, Exception):
        return None

    if result.returncode != 0:
        return None

    for line in result.stdout.split('\n'):
        if 'PROBE_PEAK_MEMORY_MB:' in line:
            match = re.search(r'PROBE_PEAK_MEMORY_MB:\s*([0-9.]+)', line)
            if match:
                return float(match.group(1))
    return None



def probe_max_batch_size(cmd_no_bs, env, cwd, max_bs_cap=256):
    """Find max batch size via binary search with memory probes.

    Each probe runs 1 fwd+bwd step (~10-15s) — much faster than a full training run.
    Exponential search + binary search finds the exact max in O(log N) probes.

    The probe tests actual peak memory for 1 step. find_max_batch_size() provides
    OOM fallback (step down by 1) in case training uses slightly more than 1 step.

    Returns max batch size (int >= 1), or 0 if even bs=1 OOMs.
    """
    # Check bs=1 first
    if _probe_memory(cmd_no_bs, 1, env, cwd) is None:
        return 0  # Even bs=1 fails

    # Exponential search: double up until OOM to find upper bound
    lo = 1
    hi = 2
    while hi <= max_bs_cap:
        if _probe_memory(cmd_no_bs, hi, env, cwd) is None:
            break  # hi OOMs, max is in [lo, hi)
        lo = hi
        hi = hi * 2

    hi = min(hi, max_bs_cap + 1)

    # If max_bs_cap didn't OOM, return it
    if lo >= max_bs_cap:
        return max_bs_cap

    # Binary search between lo (works) and hi (OOMs)
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if _probe_memory(cmd_no_bs, mid, env, cwd) is not None:
            lo = mid
        else:
            hi = mid

    # lo is the largest bs that fits in 1 step
    # Subtract 1 for safety (fragmentation over many steps)
    return max(1, lo - 1)


def find_max_batch_size(cmd_no_bs, env, cwd, timeout, cleanup_fn=None):
    """Find max batch size via memory probing, then verify with actual training.

    Uses probe_max_batch_size() for fast estimation, then trains at that bs.
    If training OOMs, steps down by 1 until it works.

    Returns (max_bs, result) where result is the subprocess result from
    the final successful run, or (0, None) if even bs=1 OOMs.
    """
    # Memory probe to estimate max bs (fast — seconds, not minutes)
    bs = probe_max_batch_size(cmd_no_bs, env, cwd)
    if bs == 0:
        return (0, None)

    # Try training at estimated max, step down on OOM
    while bs >= 1:
        cmd = cmd_no_bs + ['--batch_size', str(bs)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=timeout, env=env, cwd=cwd)
        except (subprocess.TimeoutExpired, Exception):
            if bs == 1:
                return (0, None)
            bs -= 1
            if cleanup_fn:
                cleanup_fn()
            continue

        if _is_oom(result):
            if bs == 1:
                return (0, None)
            bs -= 1
            if cleanup_fn:
                cleanup_fn()
            continue

        # Success!
        return (bs, result)

    return (0, None)


def run_training_progressive(gpu_id, params, model_type, train_minutes, output_dir, eval_id):
    """Run 2-phase progressive training: Phase 1 @ 512, Phase 2 @ 32K.

    train_minutes is ignored — uses PHASE1_MINUTES and PHASE2_MINUTES instead.
    Returns result dict with loss = Phase 2 final loss (the fitness signal).
    """
    eval_dir = os.path.join(output_dir, f'eval_{eval_id}')
    phase1_dir = os.path.join(eval_dir, 'phase1')
    phase2_dir = os.path.join(eval_dir, 'phase2')
    os.makedirs(phase1_dir, exist_ok=True)
    os.makedirs(phase2_dir, exist_ok=True)

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    cwd = os.path.dirname(os.path.abspath(__file__))

    actual_params = estimate_params_for_config(params, model_type)

    # --- PHASE 1: Train at 512 (save checkpoint for resume) ---
    # Build Phase 1 command: short context, normal batch size, KEEP checkpoints
    global CHUNK_SIZE, GRADIENT_CHECKPOINTING, PROJECTION_CHUNK_SIZE
    saved_globals = (CHUNK_SIZE, GRADIENT_CHECKPOINTING, PROJECTION_CHUNK_SIZE)

    CHUNK_SIZE = 512
    GRADIENT_CHECKPOINTING = False
    PROJECTION_CHUNK_SIZE = 0
    cmd1, _ = build_train_command(params, model_type, PHASE1_MINUTES, phase1_dir)
    CHUNK_SIZE, GRADIENT_CHECKPOINTING, PROJECTION_CHUNK_SIZE = saved_globals

    # Phase 1 needs to SAVE a checkpoint — override keep_checkpoints and save_every
    # Remove --keep_checkpoints 0 and --save_every 999999 from cmd1
    cmd1_filtered = []
    skip_next = False
    for i, arg in enumerate(cmd1):
        if skip_next:
            skip_next = False
            continue
        if arg in ('--keep_checkpoints', '--save_every'):
            skip_next = True  # Skip this flag and its value
            continue
        cmd1_filtered.append(arg)
    cmd1 = cmd1_filtered + ['--keep_checkpoints', '1', '--save_every', '999999']
    # Final checkpoint is always saved by train.py at the end, so save_every=999999 is fine

    # Phase 1: probe memory to find max batch size, then train with OOM fallback
    cmd1_no_bs = []
    skip_next = False
    for arg in cmd1:
        if skip_next:
            skip_next = False
            continue
        if arg == '--batch_size':
            skip_next = True
            continue
        cmd1_no_bs.append(arg)

    phase1_timeout = PHASE1_MINUTES * 60 + 300

    def cleanup_phase1():
        for d in glob.glob(os.path.join(phase1_dir, 'level*')):
            shutil.rmtree(d, ignore_errors=True)

    phase1_max_bs, result1 = find_max_batch_size(cmd1_no_bs, env, cwd,
                                                  phase1_timeout, cleanup_phase1)

    # Record Phase 1 max batch size
    with open(os.path.join(eval_dir, 'phase1_max_batch_size.txt'), 'w') as f:
        f.write(str(phase1_max_bs))

    if phase1_max_bs == 0 or result1 is None:
        return {
            'params': params, 'actual_params': actual_params,
            'loss': float('inf'), 'eval_id': eval_id, 'gpu_id': gpu_id,
            'success': False, 'error': 'phase1_oom',
        }

    # Save Phase 1 stdout/stderr
    with open(os.path.join(eval_dir, 'phase1_stdout.txt'), 'w') as f:
        f.write(result1.stdout)
    if result1.stderr:
        with open(os.path.join(eval_dir, 'phase1_stderr.txt'), 'w') as f:
            f.write(result1.stderr)

    if result1.returncode != 0:
        return {
            'params': params, 'actual_params': actual_params,
            'loss': float('inf'), 'eval_id': eval_id, 'gpu_id': gpu_id,
            'success': False, 'error': f'phase1_returncode_{result1.returncode}',
        }

    # Find Phase 1 checkpoint
    ckpts = glob.glob(os.path.join(phase1_dir, '**', 'checkpoint_*.pt'), recursive=True)
    if not ckpts:
        return {
            'params': params, 'actual_params': actual_params,
            'loss': float('inf'), 'eval_id': eval_id, 'gpu_id': gpu_id,
            'success': False, 'error': 'no_phase1_checkpoint',
        }
    # Use the latest checkpoint (sorted alphabetically, step number is zero-padded)
    ckpt_path = sorted(ckpts)[-1]

    # --- PHASE 2: Resume at 32K with model-specific settings ---
    # Build base Phase 2 command (without batch_size — we'll set it per attempt)
    saved_globals = (CHUNK_SIZE, GRADIENT_CHECKPOINTING, PROJECTION_CHUNK_SIZE)

    CHUNK_SIZE = PHASE2_CHUNK_SIZE
    GRADIENT_CHECKPOINTING = model_type in PHASE2_GRAD_CKPT_MODELS
    PROJECTION_CHUNK_SIZE = 512 if model_type in PHASE2_PROJ_CHUNK_MODELS else 0
    cmd2_base, _ = build_train_command(params, model_type, PHASE2_MINUTES, phase2_dir)
    CHUNK_SIZE, GRADIENT_CHECKPOINTING, PROJECTION_CHUNK_SIZE = saved_globals

    # Strip batch_size from base command
    cmd2_no_bs = []
    skip_next = False
    for arg in cmd2_base:
        if skip_next:
            skip_next = False
            continue
        if arg == '--batch_size':
            skip_next = True
            continue
        cmd2_no_bs.append(arg)

    # Add --resume pointing to Phase 1 checkpoint
    cmd2_no_bs += ['--resume', ckpt_path]

    phase2_timeout = PHASE2_MINUTES * 60 + 300

    def cleanup_phase2():
        for d in glob.glob(os.path.join(phase2_dir, 'level*')):
            shutil.rmtree(d, ignore_errors=True)

    max_bs, result2 = find_max_batch_size(cmd2_no_bs, env, cwd,
                                           phase2_timeout, cleanup_phase2)

    # Save Phase 2 stdout/stderr
    if result2 is not None:
        with open(os.path.join(eval_dir, 'phase2_stdout.txt'), 'w') as f:
            f.write(result2.stdout)
        if result2.stderr:
            with open(os.path.join(eval_dir, 'phase2_stderr.txt'), 'w') as f:
                f.write(result2.stderr)

    # Record the max batch size that fit
    with open(os.path.join(eval_dir, 'phase2_max_batch_size.txt'), 'w') as f:
        f.write(str(max_bs))

    # Parse Phase 2 loss
    loss = float('inf')
    if result2 is not None:
        for line in result2.stdout.split('\n'):
            if 'FINAL_LOSS_LAST100:' in line:
                match = re.search(r'FINAL_LOSS_LAST100:\s*([0-9.]+)', line)
                if match:
                    try:
                        loss = float(match.group(1))
                        break
                    except:
                        pass

        if loss == float('inf'):
            p2_ckpts = glob.glob(os.path.join(phase2_dir, '**', 'checkpoint_*.pt'), recursive=True)
            for ckpt in p2_ckpts:
                match = re.search(r'loss_([0-9.]+)\.pt', ckpt)
                if match:
                    try:
                        ckpt_loss = float(match.group(1))
                        if ckpt_loss < loss:
                            loss = ckpt_loss
                    except:
                        pass

    # Delete Phase 1 checkpoint(s) — only needed for resume, not archival
    for ckpt in ckpts:
        try:
            os.remove(ckpt)
        except:
            pass
    # Keep Phase 2 checkpoints — retain_top_checkpoints() will prune non-top-3 later

    return {
        'params': params,
        'actual_params': actual_params,
        'loss': loss,
        'eval_id': eval_id,
        'gpu_id': gpu_id,
        'success': loss < 10.0,
    }


def run_training(gpu_id, params, model_type, train_minutes, output_dir, eval_id):
    """Run training for a single configuration."""
    if PROGRESSIVE:
        return run_training_progressive(gpu_id, params, model_type, train_minutes, output_dir, eval_id)

    eval_dir = os.path.join(output_dir, f'eval_{eval_id}')
    os.makedirs(eval_dir, exist_ok=True)

    cmd_base, actual_params = build_train_command(params, model_type, train_minutes, eval_dir)

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    cwd = os.path.dirname(os.path.abspath(__file__))

    # Strip batch_size from base command
    cmd_no_bs = []
    skip_next = False
    for arg in cmd_base:
        if skip_next:
            skip_next = False
            continue
        if arg == '--batch_size':
            skip_next = True
            continue
        cmd_no_bs.append(arg)

    timeout = train_minutes * 60 + 300

    def cleanup():
        for d in glob.glob(os.path.join(eval_dir, 'level*')):
            shutil.rmtree(d, ignore_errors=True)

    max_bs, result = find_max_batch_size(cmd_no_bs, env, cwd, timeout, cleanup)

    # Record max batch size that fit
    with open(os.path.join(eval_dir, 'max_batch_size.txt'), 'w') as f:
        f.write(str(max_bs))

    # Parse loss from result
    loss = float('inf')
    if result is not None:
        # Always save stdout/stderr for loss curve recovery
        with open(os.path.join(eval_dir, 'stdout.txt'), 'w') as f:
            f.write(result.stdout)
        if result.stderr:
            with open(os.path.join(eval_dir, 'stderr.txt'), 'w') as f:
                f.write(result.stderr)

        for line in result.stdout.split('\n'):
            if 'FINAL_LOSS_LAST100:' in line:
                match = re.search(r'FINAL_LOSS_LAST100:\s*([0-9.]+)', line)
                if match:
                    try:
                        loss = float(match.group(1))
                        break
                    except:
                        pass

        if loss == float('inf'):
            ckpts = glob.glob(os.path.join(eval_dir, '**', 'checkpoint_*.pt'), recursive=True)
            for ckpt in ckpts:
                match = re.search(r'loss_([0-9.]+)\.pt', ckpt)
                if match:
                    try:
                        ckpt_loss = float(match.group(1))
                        if ckpt_loss < loss:
                            loss = ckpt_loss
                    except:
                        pass

    return {
        'params': params,
        'actual_params': actual_params,
        'loss': loss,
        'eval_id': eval_id,
        'gpu_id': gpu_id,
        'success': loss < 10.0,
    }





def evaluate_batch(configs, model_type, train_minutes, output_dir, gpus, start_eval_id=0):
    """Evaluate a batch of configurations in parallel."""
    results = []

    with ProcessPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {}

        for i, params in enumerate(configs):
            gpu_id = gpus[i % len(gpus)]
            eval_id = start_eval_id + i
            future = executor.submit(
                run_training, gpu_id, params, model_type, train_minutes, output_dir, eval_id
            )
            futures[future] = (eval_id, params)

        for future in as_completed(futures):
            eval_id, params = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"  [Eval {eval_id}] GPU {result['gpu_id']} | {format_params(params)} | "
                      f"{result['actual_params']/1e6:.1f}M params | Loss: {result['loss']:.4f}")
            except Exception as e:
                print(f"  [Eval {eval_id}] FAILED: {e}")
                results.append({
                    'params': params,
                    'loss': float('inf'),
                    'eval_id': eval_id,
                    'success': False,
                    'error': str(e),
                })

    return results


def format_params(params):
    """Format params dict for display."""
    parts = []
    for k, v in params.items():
        if isinstance(v, float):
            if k == 'lr':
                parts.append(f"{k}={v:.4g}")
            else:
                parts.append(f"{k}={v:.2f}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


# =============================================================================
# PHASE 1: LHS EXPLORATION
# =============================================================================
def run_lhs_phase(model_type, n_samples, train_minutes, output_dir, gpus,
                  target_params=480_000_000, fixed_params=None, seed=42):
    """Run LHS exploration phase."""
    print(f"\n{'='*70}")
    print(f"PHASE 1: Latin Hypercube Sampling ({n_samples} valid samples)")
    print(f"{'='*70}")

    # Inject known good configs first (ensures we explore around them)
    valid_configs = []
    if model_type in KNOWN_GOOD_CONFIGS:
        # Get n_state from fixed_params if doing a sweep
        n_state = fixed_params.get('n_state') if fixed_params else None
        lookup_key = n_state if n_state is not None else None
        if lookup_key in KNOWN_GOOD_CONFIGS[model_type]:
            seed_configs = KNOWN_GOOD_CONFIGS[model_type][lookup_key]
            for sc in seed_configs:
                cfg = {**sc}
                if n_state is not None:
                    cfg['n_state'] = n_state
                if is_valid_param_count(cfg, model_type, target_params, 0.10):
                    valid_configs.append(cfg)
            if valid_configs:
                print(f"  Injected {len(valid_configs)} known good configs as seeds")

    # Keep sampling until we have enough valid configs
    attempt = 0
    max_attempts = 10

    while len(valid_configs) < n_samples and attempt < max_attempts:
        # Generate more samples each attempt to get enough valid ones
        batch_size = n_samples * (2 ** attempt)  # 64, 128, 256, ...
        configs = generate_lhs_configs(model_type, batch_size, fixed_params, seed + attempt)

        # Filter by param count (10% tolerance: 432M-528M for 480M target)
        for c in configs:
            if is_valid_param_count(c, model_type, target_params, 0.10):
                # Avoid duplicates by checking dim+depth+key params
                key = (c.get('dim'), c.get('depth'), c.get('n_heads', 0), c.get('n_state', 0))
                existing_keys = [(v.get('dim'), v.get('depth'), v.get('n_heads', 0), v.get('n_state', 0))
                                 for v in valid_configs]
                if key not in existing_keys:
                    valid_configs.append(c)
                    if len(valid_configs) >= n_samples:
                        break

        print(f"  Attempt {attempt + 1}: Generated {batch_size} samples, {len(valid_configs)}/{n_samples} valid so far")
        attempt += 1

    print(f"Total valid configs within ±10% of {target_params/1e6:.0f}M: {len(valid_configs)}")

    # Run evaluations in batches
    all_results = []
    batch_size = len(gpus) * 2  # 2 batches per "generation" for better GPU utilization

    for batch_start in range(0, len(valid_configs), batch_size):
        batch = valid_configs[batch_start:batch_start + batch_size]
        print(f"\n--- LHS Batch {batch_start // batch_size + 1} ({len(batch)} configs) ---")

        # Split into sub-batches for parallel execution
        for sub_start in range(0, len(batch), len(gpus)):
            sub_batch = batch[sub_start:sub_start + len(gpus)]
            results = evaluate_batch(sub_batch, model_type, train_minutes, output_dir,
                                     gpus, start_eval_id=batch_start + sub_start)
            all_results.extend(results)

        # Prune checkpoints after each LHS batch: keep only top-3
        retain_top_checkpoints(output_dir, all_results, top_n=3)

    # Sort by loss
    all_results.sort(key=lambda x: x['loss'])

    # Report top 10
    print(f"\n{'='*70}")
    print("LHS PHASE COMPLETE - Top 10 Configurations:")
    print(f"{'='*70}")
    for i, r in enumerate(all_results[:10]):
        print(f"  {i+1}. Loss={r['loss']:.4f} | {format_params(r['params'])}")

    return all_results


# =============================================================================
# PHASE 2: CMA-ES REFINEMENT
# =============================================================================
def run_cmaes_phase(model_type, train_minutes, output_dir, gpus,
                    warm_starts, target_params=480_000_000, fixed_params=None,
                    sigma0=0.35, min_generations=6, converge_threshold=0.005,
                    consecutive_required=3, max_generations=30):
    """Run CMA-ES refinement phase from warm starts."""
    print(f"\n{'='*70}")
    print(f"PHASE 2: CMA-ES Refinement")
    print(f"{'='*70}")
    print(f"  Warm starts: {len(warm_starts)}")
    print(f"  Sigma: {sigma0} (refinement: {sigma0 * 0.4:.2f})")
    print(f"  Min generations: {min_generations}")
    print(f"  Converge threshold: {converge_threshold}")
    print(f"  Consecutive required: {consecutive_required}")

    all_results = []

    for ws_idx, warm_start in enumerate(warm_starts):
        print(f"\n--- CMA-ES from warm start {ws_idx + 1}/{len(warm_starts)} ---")
        print(f"    Start config: {format_params(warm_start)}")

        n_dims = get_search_dim(model_type, fixed_params)
        x0 = encode_params(warm_start, model_type, fixed_params)

        # Use smaller sigma for refinement (sigma0 is for exploration, use 40% of it for refinement)
        refinement_sigma = sigma0 * 0.4  # 0.35 * 0.4 = 0.14
        es = cma.CMAEvolutionStrategy(x0, refinement_sigma, {
            'popsize': len(gpus) * 2,  # 16 configs per generation (2 batches)
            'bounds': [0, 1],
            'seed': 42 + ws_idx,
            'verbose': -1,
        })

        best_loss = float('inf')
        best_params = None
        generations_without_improvement = 0
        eval_counter = len(all_results)

        for gen in range(max_generations):
            # REJECTION SAMPLING: Generate enough valid configs to fill all GPUs
            target_evals = len(gpus) * 2  # Target 16 valid configs per generation
            valid_solutions = []
            valid_configs = []
            total_generated = 0
            max_attempts = 20  # Prevent infinite loop

            for attempt in range(max_attempts):
                # Ask for a batch of solutions
                batch_size = target_evals * 2  # Overgenererate 2x
                solutions_batch = es.ask(number=batch_size)
                total_generated += batch_size

                for sol in solutions_batch:
                    if len(valid_solutions) >= target_evals:
                        break
                    cfg = decode_params(sol, model_type, fixed_params)
                    if is_valid_param_count(cfg, model_type, target_params, 0.10):
                        # Check not duplicate
                        if not any(np.allclose(sol, vs) for vs in valid_solutions):
                            valid_solutions.append(sol)
                            valid_configs.append(cfg)

                if len(valid_solutions) >= target_evals:
                    break

            n_valid = len(valid_configs)
            print(f"\n  Generation {gen + 1}: {n_valid} valid configs (from {total_generated} generated)")

            # Evaluate all valid configs (should fill all GPUs)
            gen_results = []
            if valid_configs:
                for batch_start in range(0, len(valid_configs), len(gpus)):
                    batch_configs = valid_configs[batch_start:batch_start + len(gpus)]
                    batch_results = evaluate_batch(
                        batch_configs, model_type, train_minutes, output_dir,
                        gpus, start_eval_id=eval_counter
                    )
                    gen_results.extend(batch_results)
                    eval_counter += len(batch_results)

            # Tell CMA-ES only about the valid solutions we actually evaluated
            # This keeps the covariance matrix clean (no penalty pollution)
            fitnesses = [r['loss'] for r in gen_results]
            if valid_solutions and fitnesses:
                es.tell(valid_solutions[:len(fitnesses)], fitnesses)

            # Track best
            if not fitnesses:
                print(f"    No valid configs this generation, skipping...")
                continue

            gen_best_loss = min(fitnesses)
            gen_best_idx = fitnesses.index(gen_best_loss)

            if gen_best_loss < best_loss:
                improvement = best_loss - gen_best_loss
                best_loss = gen_best_loss
                best_params = valid_configs[gen_best_idx]
                print(f"    *** NEW BEST: {best_loss:.4f} | {format_params(best_params)} ***")

                if improvement < converge_threshold:
                    generations_without_improvement += 1
                else:
                    generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            print(f"    Gen best: {gen_best_loss:.4f} | Overall best: {best_loss:.4f} | "
                  f"No improvement: {generations_without_improvement}/{consecutive_required}")

            all_results.extend(gen_results)

            # Prune checkpoints: keep only top-3 across all evals so far
            retain_top_checkpoints(output_dir, all_results, top_n=3)

            # Check convergence (but not before min_generations)
            if gen >= min_generations - 1 and generations_without_improvement >= consecutive_required:
                print(f"\n    CONVERGED after {gen + 1} generations")
                break

        print(f"\n  Warm start {ws_idx + 1} complete. Best: {best_loss:.4f}")

    # Sort all results
    all_results.sort(key=lambda x: x['loss'])

    return all_results


# =============================================================================
# DISCRETE PARAMETER SWEEP
# =============================================================================
def run_discrete_sweep(model_type, sweep_param, train_minutes, output_dir, gpus,
                       target_params=480_000_000, lhs_samples=24, cmaes_refinements=2):
    """Run separate search for each discrete parameter value."""
    if model_type not in DISCRETE_SWEEP_PARAMS or sweep_param not in DISCRETE_SWEEP_PARAMS[model_type]:
        print(f"No discrete sweep defined for {model_type}.{sweep_param}")
        return []

    values = DISCRETE_SWEEP_PARAMS[model_type][sweep_param]
    print(f"\n{'='*70}")
    print(f"DISCRETE SWEEP: {sweep_param} = {values}")
    print(f"{'='*70}")

    all_results = []

    for val in values:
        print(f"\n{'='*70}")
        print(f"SWEEP: {sweep_param} = {val}")
        print(f"{'='*70}")

        fixed_params = {sweep_param: val}
        sweep_dir = os.path.join(output_dir, f'{sweep_param}_{val}')
        os.makedirs(sweep_dir, exist_ok=True)

        # Phase 1: LHS
        lhs_results = run_lhs_phase(
            model_type, lhs_samples, train_minutes, sweep_dir, gpus,
            target_params, fixed_params
        )

        # Phase 2: CMA-ES from top configs
        top_configs = [r['params'] for r in lhs_results[:cmaes_refinements] if r['loss'] < 5.0]
        if top_configs:
            cmaes_results = run_cmaes_phase(
                model_type, train_minutes, sweep_dir, gpus,
                top_configs, target_params, fixed_params
            )
            all_results.extend(cmaes_results)

        all_results.extend(lhs_results)

    # Sort all
    all_results.sort(key=lambda x: x['loss'])

    return all_results


# =============================================================================
# CLEANUP
# =============================================================================
def retain_top_checkpoints(output_dir, results, top_n=3):
    """Keep checkpoints only for the top N configs by loss. Delete the rest.

    Each eval_dir may contain checkpoint .pt files. We keep checkpoints for
    the top_n best losses and remove all others to save disk space.
    """
    # Sort results by loss (best first)
    sorted_results = sorted(
        [r for r in results if r.get('success') and r.get('loss', float('inf')) < 10.0],
        key=lambda r: r['loss']
    )

    # Eval IDs to keep
    keep_eval_ids = set()
    for r in sorted_results[:top_n]:
        keep_eval_ids.add(r.get('eval_id'))

    # Find and manage checkpoints
    kept = 0
    deleted = 0
    for eval_dir_path in glob.glob(os.path.join(output_dir, 'eval_*')):
        eval_id_match = re.search(r'eval_(\d+)', os.path.basename(eval_dir_path))
        if not eval_id_match:
            continue
        eval_id = int(eval_id_match.group(1))

        pt_files = glob.glob(os.path.join(eval_dir_path, '**', '*.pt'), recursive=True)
        # Exclude latest.pt symlinks from count
        pt_files = [f for f in pt_files if not os.path.islink(f)]

        if eval_id in keep_eval_ids:
            kept += len(pt_files)
        else:
            for f in pt_files:
                try:
                    os.remove(f)
                    deleted += 1
                except:
                    pass
            # Also remove symlinks
            for f in glob.glob(os.path.join(eval_dir_path, '**', 'latest.pt'), recursive=True):
                try:
                    os.remove(f)
                except:
                    pass

    if kept or deleted:
        top_losses = [f"{r['loss']:.4f}" for r in sorted_results[:top_n]]
        print(f"\nCheckpoint retention: kept {kept} files (top {top_n}: losses {', '.join(top_losses)}), "
              f"deleted {deleted} files")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='CMA-ES v2: Improved Architecture Search')
    parser.add_argument('--model', type=str, required=True,
                        choices=list(SEARCH_SPACES.keys()),
                        help='Model type to search')
    parser.add_argument('--phase', type=str, default='both',
                        choices=['lhs', 'cmaes', 'both', 'sweep'],
                        help='Search phase: lhs, cmaes, both, or sweep')
    parser.add_argument('--train_minutes', type=float, default=30,
                        help='Training time per config (minutes)')
    parser.add_argument('--gpus', type=str, default='0,1,2,3,4,5,6,7',
                        help='Comma-separated GPU IDs')
    parser.add_argument('--params', type=str, default='480M',
                        help='Target parameter count (e.g., 480M)')
    parser.add_argument('--output', type=str, default='benchmark_results/cmaes_v2',
                        help='Output directory')

    # LHS options
    parser.add_argument('--lhs_samples', type=int, default=128,
                        help='Number of LHS samples (phase 1)')

    # CMA-ES options
    parser.add_argument('--sigma', type=float, default=0.35,
                        help='Initial sigma for CMA-ES')
    parser.add_argument('--min_generations', type=int, default=6,
                        help='Minimum generations before convergence check')
    parser.add_argument('--converge', type=float, default=0.01,
                        help='Convergence threshold (1% of typical loss)')
    parser.add_argument('--consecutive', type=int, default=2,
                        help='Consecutive generations without improvement to converge')
    parser.add_argument('--cmaes_refinements', type=int, default=3,
                        help='Number of top LHS configs to refine with CMA-ES')

    # Sweep options
    parser.add_argument('--sweep_param', type=str, default=None,
                        help='Discrete parameter to sweep (e.g., n_state)')
    parser.add_argument('--fixed_n_state', type=int, default=None,
                        help='Fix n_state to this value (skip sweep)')

    # Warm start
    parser.add_argument('--warm_start', type=str, default=None,
                        help='JSON file with warm start configs')

    # torch.compile options
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile for training (recommended: +17% throughput)')
    parser.add_argument('--compile_mode', type=str, default='max-autotune',
                        help='torch.compile mode (default, reduce-overhead, max-autotune)')

    # Sequence length scaling
    parser.add_argument('--chunk_size', type=int, default=512,
                        help='Sequence chunk size (default: 512, for scaling: 1024, 2048)')

    # Long-sequence options
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Enable gradient checkpointing (needed for long sequences)')
    parser.add_argument('--projection_chunk_size', type=int, default=0,
                        help='Projection chunk size for memory savings (0=disabled)')

    # Progressive training (512→32K)
    parser.add_argument('--progressive', action='store_true',
                        help='Enable 2-phase progressive training: Phase 1 @ 512, Phase 2 @ 32K')
    parser.add_argument('--phase1_minutes', type=float, default=10,
                        help='Phase 1 training time at 512 (default: 10)')
    parser.add_argument('--phase2_minutes', type=float, default=10,
                        help='Phase 2 training time at 32K (default: 10)')
    parser.add_argument('--phase2_chunk_size', type=int, default=32768,
                        help='Phase 2 sequence length (default: 32768)')

    args = parser.parse_args()

    # Parse params
    target_params = int(args.params.lower().replace('m', '000000').replace('b', '000000000'))
    gpus = [int(g) for g in args.gpus.split(',')]

    # Set global compile and sequence settings
    global COMPILE_ENABLED, COMPILE_MODE, CHUNK_SIZE, GRADIENT_CHECKPOINTING, PROJECTION_CHUNK_SIZE
    global PROGRESSIVE, PHASE1_MINUTES, PHASE2_MINUTES, PHASE2_CHUNK_SIZE
    COMPILE_ENABLED = args.compile
    COMPILE_MODE = args.compile_mode
    GRADIENT_CHECKPOINTING = args.gradient_checkpointing
    PROJECTION_CHUNK_SIZE = args.projection_chunk_size
    CHUNK_SIZE = args.chunk_size

    # Progressive training settings
    PROGRESSIVE = args.progressive
    PHASE1_MINUTES = args.phase1_minutes
    PHASE2_MINUTES = args.phase2_minutes
    PHASE2_CHUNK_SIZE = args.phase2_chunk_size

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output, f'{args.model}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    print(f"{'='*70}")
    print(f"CMA-ES v2 Search for {args.model.upper()}")
    print(f"{'='*70}")
    print(f"Phase: {args.phase}")
    print(f"Target params: {target_params/1e6:.0f}M")
    if PROGRESSIVE:
        print(f"Progressive training: Phase 1 = {PHASE1_MINUTES} min @ 512, Phase 2 = {PHASE2_MINUTES} min @ {PHASE2_CHUNK_SIZE}")
        print(f"Effective time per eval: {PHASE1_MINUTES + PHASE2_MINUTES} min")
    else:
        print(f"Training time: {args.train_minutes} min/config")
    print(f"GPUs: {gpus}")
    print(f"Output: {output_dir}")
    if not PROGRESSIVE:
        print(f"Chunk size: {CHUNK_SIZE} (batch size auto-scaled)")
    print(f"torch.compile: {COMPILE_ENABLED} (mode: {COMPILE_MODE})")
    if args.phase in ['both', 'lhs']:
        print(f"LHS samples: {args.lhs_samples}")
    if args.phase in ['both', 'cmaes']:
        print(f"Sigma: {args.sigma}, Min gens: {args.min_generations}, "
              f"Converge: {args.converge}, Consecutive: {args.consecutive}")

    # Build fixed_params dict
    fixed_params = {}
    if args.fixed_n_state is not None:
        fixed_params['n_state'] = args.fixed_n_state
        print(f"Fixed n_state: {args.fixed_n_state}")

    # Log to file
    log_file = os.path.join(output_dir, 'search.log')

    start_time = time.time()

    if args.phase == 'sweep':
        sweep_param = args.sweep_param or list(DISCRETE_SWEEP_PARAMS.get(args.model, {}).keys())[0]
        results = run_discrete_sweep(
            args.model, sweep_param, args.train_minutes, output_dir, gpus,
            target_params, args.lhs_samples, args.cmaes_refinements
        )

    elif args.phase == 'lhs':
        results = run_lhs_phase(
            args.model, args.lhs_samples, args.train_minutes, output_dir, gpus,
            target_params, fixed_params=fixed_params if fixed_params else None
        )

    elif args.phase == 'cmaes':
        # Load warm starts
        if args.warm_start:
            with open(args.warm_start) as f:
                warm_starts = json.load(f)
        else:
            # Use default warm start
            warm_starts = [{}]  # Will use middle of search space

        results = run_cmaes_phase(
            args.model, args.train_minutes, output_dir, gpus,
            warm_starts, target_params,
            fixed_params=fixed_params if fixed_params else None,
            sigma0=args.sigma, min_generations=args.min_generations,
            converge_threshold=args.converge, consecutive_required=args.consecutive
        )

    else:  # both
        # Phase 1: LHS
        lhs_results = run_lhs_phase(
            args.model, args.lhs_samples, args.train_minutes, output_dir, gpus,
            target_params, fixed_params=fixed_params if fixed_params else None
        )

        # Phase 2: CMA-ES from top configs
        top_configs = [r['params'] for r in lhs_results[:args.cmaes_refinements] if r['loss'] < 5.0]

        if top_configs:
            cmaes_results = run_cmaes_phase(
                args.model, args.train_minutes, output_dir, gpus,
                top_configs, target_params,
                fixed_params=fixed_params if fixed_params else None,
                sigma0=args.sigma, min_generations=args.min_generations,
                converge_threshold=args.converge, consecutive_required=args.consecutive
            )
            results = lhs_results + cmaes_results
        else:
            print("No valid LHS configs found, skipping CMA-ES phase")
            results = lhs_results

        results.sort(key=lambda x: x['loss'])

    # Final report
    elapsed = (time.time() - start_time) / 3600

    print(f"\n{'='*70}")
    print(f"SEARCH COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {elapsed:.2f} hours")
    print(f"Total evaluations: {len(results)}")

    if results:
        best = results[0]
        print(f"\nBest loss: {best['loss']:.4f}")
        print(f"Best config: {format_params(best['params'])}")

        # Save results
        results_file = os.path.join(output_dir, 'results.json')
        with open(results_file, 'w') as f:
            json.dump({
                'model': args.model,
                'best_loss': best['loss'],
                'best_params': best['params'],
                'all_results': [{'params': r['params'], 'loss': r['loss'],
                                 'actual_params': r.get('actual_params'),
                                 'eval_id': r.get('eval_id')} for r in results],
                'elapsed_hours': elapsed,
                'total_evals': len(results),
            }, f, indent=2, default=str)
        print(f"\nResults saved to: {results_file}")

    # Keep only top-3 checkpoints, delete the rest
    retain_top_checkpoints(output_dir, results, top_n=3)

    return results


if __name__ == '__main__':
    main()
