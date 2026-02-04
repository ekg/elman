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
    calc_e1_params, calc_e23_params, calc_e42_params, calc_e75_params
)

# Supported n_state values for E88
E88_SUPPORTED_N_STATE = [16, 32, 48, 64]

# E90 valid (k_fast, k_slow) configurations
E90_CONFIGS = [
    (8, 16), (8, 24), (16, 32), (16, 48),
]

# =============================================================================
# SEARCH SPACES - 6D for all models
# =============================================================================
SEARCH_SPACES = {
    'e88': {
        'dim': (1024, 3072, 'int_mult128', 'Model dimension'),
        'n_heads': (32, 160, 'int', 'Number of attention heads'),
        'n_state': (16, 64, 'e88_n_state', 'State dimension (16,32,48,64)'),
        'depth': (12, 40, 'int', 'Number of layers'),
        'use_gate': (0, 1, 'binary', 'Use output gating'),
        'lr': (1e-5, 1e-3, 'log', 'Learning rate'),
    },
    'fla-gdn': {
        'dim': (1024, 3072, 'int_mult128', 'Model dimension'),
        'expansion': (1, 3, 'int', 'Value expansion factor'),
        'depth': (12, 40, 'int', 'Number of layers'),
        'n_heads': (8, 32, 'int', 'Number of heads'),
        'use_conv': (0, 1, 'binary', 'Use short convolution'),
        'lr': (1e-5, 1e-3, 'log', 'Learning rate'),
    },
    'mamba2': {
        'dim': (1024, 3072, 'int_mult128', 'Model dimension'),
        'd_state': (64, 256, 'int_mult16', 'SSM state dimension'),
        'headdim': (32, 128, 'int_pow2', 'Head dimension'),
        'expand': (1, 3, 'int', 'Expansion factor'),
        'depth': (16, 40, 'int', 'Number of layers'),
        'lr': (1e-5, 1e-3, 'log', 'Learning rate'),
    },
    'transformer': {
        'dim': (1024, 3072, 'int_mult128', 'Model dimension'),
        'n_heads': (8, 32, 'int', 'Number of attention heads'),
        'expansion': (2, 6, 'int', 'FFN expansion factor'),
        'depth': (12, 36, 'int', 'Number of layers'),
        'dropout': (0.0, 0.15, 'float', 'Dropout rate'),
        'lr': (1e-5, 1e-3, 'log', 'Learning rate'),
    },
    'mingru': {
        'dim': (1024, 3584, 'int_mult128', 'Model dimension'),
        'expansion': (1, 4, 'int', 'Expansion factor'),
        'depth': (12, 40, 'int', 'Number of layers'),
        'use_conv': (0, 1, 'binary', 'Use Conv1d'),
        'd_conv': (3, 7, 'int', 'Conv kernel size'),
        'lr': (1e-5, 1e-3, 'log', 'Learning rate'),
    },
    'minlstm': {
        'dim': (1024, 3584, 'int_mult128', 'Model dimension'),
        'expansion': (1, 4, 'int', 'Expansion factor'),
        'depth': (12, 40, 'int', 'Number of layers'),
        'use_conv': (0, 1, 'binary', 'Use Conv1d'),
        'd_conv': (3, 7, 'int', 'Conv kernel size'),
        'lr': (1e-5, 1e-3, 'log', 'Learning rate'),
    },
    'e1': {
        'dim': (1024, 3072, 'int_mult128', 'Model dimension'),
        'expansion': (1, 3, 'int', 'Expansion factor'),
        'depth': (12, 40, 'int', 'Number of layers'),
        'use_conv': (0, 1, 'binary', 'Use Conv1d'),
        'mamba2_init': (0, 1, 'binary', 'Use Mamba2-style init'),
        'lr': (1e-5, 1e-3, 'log', 'Learning rate'),
    },
    'e42': {
        'dim': (1024, 3584, 'int_mult128', 'Model dimension'),
        'expansion': (1, 3, 'int', 'Expansion factor'),
        'depth': (12, 40, 'int', 'Number of layers'),
        'spectral_radius': (0.9, 0.999, 'float', 'Spectral radius'),
        'mamba2_init': (0, 1, 'binary', 'Use Mamba2-style init'),
        'lr': (1e-5, 1e-3, 'log', 'Learning rate'),
    },
    'e75': {
        'dim': (1024, 3072, 'int_mult128', 'Model dimension'),
        'n_heads': (4, 32, 'int', 'Number of heads'),
        'n_state': (16, 64, 'int_mult8', 'State dimension'),
        'depth': (12, 40, 'int', 'Number of layers'),
        'expansion': (1, 2, 'float', 'Expansion factor'),
        'lr': (1e-5, 1e-3, 'log', 'Learning rate'),
    },
}

# Discrete parameters that benefit from sweep (instead of CMA-ES interpolation)
DISCRETE_SWEEP_PARAMS = {
    'e88': {'n_state': [16, 32, 48, 64]},
    'e75': {'n_state': [16, 24, 32, 40, 48, 56, 64]},
}

# =============================================================================
# PARAMETER CONVERSION
# =============================================================================
def decode_params(x, model_type, fixed_params=None):
    """Convert CMA-ES vector [0,1]^n to model parameters."""
    space = SEARCH_SPACES[model_type]
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
    space = SEARCH_SPACES[model_type]
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
    return len(SEARCH_SPACES[model_type]) - len(fixed_params)


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

    if model_type == 'e88':
        return calc_e88_params(dim, depth=depth, n_heads=params.get('n_heads', 96),
                               n_state=params.get('n_state', 32), expansion=1.0, use_gate=True)
    elif model_type == 'fla-gdn':
        return calc_fla_gdn_params(dim, depth=depth, expansion=params.get('expansion', 2))
    elif model_type == 'mamba2':
        return calc_mamba2_params(dim, depth=depth, expand=params.get('expand', 2))
    elif model_type == 'transformer':
        return calc_transformer_params(dim, depth=depth, n_heads=params.get('n_heads', 16),
                                       expansion=params.get('expansion', 4))
    elif model_type == 'mingru':
        return calc_mingru_params(dim, depth=depth, expansion=params.get('expansion', 2))
    elif model_type == 'minlstm':
        return calc_minlstm_params(dim, depth=depth, expansion=params.get('expansion', 2))
    elif model_type == 'e1':
        return calc_e1_params(dim, depth=depth, expansion=params.get('expansion', 2))
    elif model_type == 'e42':
        return calc_e42_params(dim, depth=depth, expansion=params.get('expansion', 2))
    elif model_type == 'e75':
        return calc_e75_params(dim, depth=depth, n_heads=params.get('n_heads', 8),
                               n_state=params.get('n_state', 32), expansion=params.get('expansion', 1.0))
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

    # Adjust batch size based on model size
    if actual_params > 600_000_000:
        batch_size = 8
    elif actual_params > 400_000_000:
        batch_size = 16
    else:
        batch_size = 32

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
            '--expansion', '1.0',
            '--use_gate', str(use_gate),
        ])
        if use_gate:
            cmd.extend(['--gate_activation', 'silu'])

    elif model_type == 'fla-gdn':
        cmd.extend([
            '--level', 'fla-gdn',
            '--expansion', str(params['expansion']),
            '--n_heads', str(params.get('n_heads', 16)),
            '--use_conv', str(params.get('use_conv', 1)),
        ])

    elif model_type == 'mamba2':
        cmd.extend(['--level', 'mamba2'])

    elif model_type == 'transformer':
        cmd.extend([
            '--level', 'llama',
            '--n_heads', str(params.get('n_heads', 16)),
            '--expansion', str(params.get('expansion', 4)),
            '--dropout', str(params.get('dropout', 0.0)),
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

    elif model_type == 'e1':
        cmd.extend([
            '--level', '1',
            '--expansion', str(params.get('expansion', 2)),
        ])
        if params.get('use_conv', 0):
            cmd.extend(['--use_conv', '1'])
        if params.get('mamba2_init', 0):
            cmd.extend(['--mamba2_init', '1'])

    elif model_type == 'e42':
        cmd.extend([
            '--level', '42',
            '--expansion', str(params.get('expansion', 2)),
        ])
        if params.get('mamba2_init', 0):
            cmd.extend(['--mamba2_init', '1'])

    elif model_type == 'e75':
        cmd.extend([
            '--level', 'E75h{n}n{s}'.format(n=params.get('n_heads', 8), s=params.get('n_state', 32)),
            '--n_heads', str(params.get('n_heads', 8)),
            '--n_state', str(params.get('n_state', 32)),
            '--expansion', str(params.get('expansion', 1.0)),
        ])

    return cmd, actual_params


# =============================================================================
# EVALUATION
# =============================================================================
def run_training(gpu_id, params, model_type, train_minutes, output_dir, eval_id):
    """Run training for a single configuration."""
    eval_dir = os.path.join(output_dir, f'eval_{eval_id}')
    os.makedirs(eval_dir, exist_ok=True)

    cmd, actual_params = build_train_command(params, model_type, train_minutes, eval_dir)

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=train_minutes * 60 + 300,  # 5 min buffer
            env=env,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )

        # Parse loss from output
        loss = float('inf')
        for line in result.stdout.split('\n'):
            if 'loss' in line.lower():
                match = re.search(r'loss[:\s]+([0-9.]+)', line, re.IGNORECASE)
                if match:
                    try:
                        loss = float(match.group(1))
                    except:
                        pass

        # Also check checkpoint files for loss
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

    except subprocess.TimeoutExpired:
        return {
            'params': params,
            'actual_params': actual_params,
            'loss': float('inf'),
            'eval_id': eval_id,
            'gpu_id': gpu_id,
            'success': False,
            'error': 'timeout',
        }
    except Exception as e:
        return {
            'params': params,
            'actual_params': actual_params,
            'loss': float('inf'),
            'eval_id': eval_id,
            'gpu_id': gpu_id,
            'success': False,
            'error': str(e),
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

    # Keep sampling until we have enough valid configs
    valid_configs = []
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
    print(f"  Sigma: {sigma0}")
    print(f"  Min generations: {min_generations}")
    print(f"  Converge threshold: {converge_threshold}")
    print(f"  Consecutive required: {consecutive_required}")

    all_results = []

    for ws_idx, warm_start in enumerate(warm_starts):
        print(f"\n--- CMA-ES from warm start {ws_idx + 1}/{len(warm_starts)} ---")
        print(f"    Start config: {format_params(warm_start)}")

        n_dims = get_search_dim(model_type, fixed_params)
        x0 = encode_params(warm_start, model_type, fixed_params)

        # Initialize CMA-ES with larger sigma
        es = cma.CMAEvolutionStrategy(x0, sigma0, {
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
            # Get population
            solutions = es.ask()

            # Convert to configs
            configs = [decode_params(s, model_type, fixed_params) for s in solutions]

            # Filter by param count - only train valid configs
            valid_mask = [is_valid_param_count(c, model_type, target_params, 0.10) for c in configs]
            valid_configs = [c for c, v in zip(configs, valid_mask) if v]
            valid_indices = [i for i, v in enumerate(valid_mask) if v]

            n_valid = len(valid_configs)
            n_invalid = len(configs) - n_valid
            print(f"\n  Generation {gen + 1}: {n_valid} valid configs, {n_invalid} skipped (wrong params)")

            # Only evaluate valid configs
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

            # Assign fitness: real loss for valid, penalty for invalid
            fitnesses = []
            result_idx = 0
            for i, (cfg, valid) in enumerate(zip(configs, valid_mask)):
                if valid and result_idx < len(gen_results):
                    fitnesses.append(gen_results[result_idx]['loss'])
                    result_idx += 1
                else:
                    fitnesses.append(10.0)  # Penalty for wrong param count (not trained)

            es.tell(solutions, fitnesses)

            # Track best
            gen_best_loss = min(fitnesses)
            gen_best_idx = fitnesses.index(gen_best_loss)

            if gen_best_loss < best_loss:
                improvement = best_loss - gen_best_loss
                best_loss = gen_best_loss
                best_params = configs[gen_best_idx]
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
def cleanup_checkpoints(output_dir):
    """Remove checkpoint files to save disk space."""
    pt_files = glob.glob(os.path.join(output_dir, '**', '*.pt'), recursive=True)
    if pt_files:
        print(f"\nCleaning up {len(pt_files)} checkpoint files...")
        for f in pt_files:
            try:
                os.remove(f)
            except:
                pass
        print("Cleanup complete.")


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
    parser.add_argument('--lhs_samples', type=int, default=48,
                        help='Number of LHS samples (phase 1)')

    # CMA-ES options
    parser.add_argument('--sigma', type=float, default=0.35,
                        help='Initial sigma for CMA-ES')
    parser.add_argument('--min_generations', type=int, default=6,
                        help='Minimum generations before convergence check')
    parser.add_argument('--converge', type=float, default=0.005,
                        help='Convergence threshold')
    parser.add_argument('--consecutive', type=int, default=3,
                        help='Consecutive generations without improvement to converge')
    parser.add_argument('--cmaes_refinements', type=int, default=3,
                        help='Number of top LHS configs to refine with CMA-ES')

    # Sweep options
    parser.add_argument('--sweep_param', type=str, default=None,
                        help='Discrete parameter to sweep (e.g., n_state)')

    # Warm start
    parser.add_argument('--warm_start', type=str, default=None,
                        help='JSON file with warm start configs')

    args = parser.parse_args()

    # Parse params
    target_params = int(args.params.lower().replace('m', '000000').replace('b', '000000000'))
    gpus = [int(g) for g in args.gpus.split(',')]

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output, f'{args.model}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    print(f"{'='*70}")
    print(f"CMA-ES v2 Search for {args.model.upper()}")
    print(f"{'='*70}")
    print(f"Phase: {args.phase}")
    print(f"Target params: {target_params/1e6:.0f}M")
    print(f"Training time: {args.train_minutes} min/config")
    print(f"GPUs: {gpus}")
    print(f"Output: {output_dir}")
    if args.phase in ['both', 'lhs']:
        print(f"LHS samples: {args.lhs_samples}")
    if args.phase in ['both', 'cmaes']:
        print(f"Sigma: {args.sigma}, Min gens: {args.min_generations}, "
              f"Converge: {args.converge}, Consecutive: {args.consecutive}")

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
            target_params
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
            sigma0=args.sigma, min_generations=args.min_generations,
            converge_threshold=args.converge, consecutive_required=args.consecutive
        )

    else:  # both
        # Phase 1: LHS
        lhs_results = run_lhs_phase(
            args.model, args.lhs_samples, args.train_minutes, output_dir, gpus,
            target_params
        )

        # Phase 2: CMA-ES from top configs
        top_configs = [r['params'] for r in lhs_results[:args.cmaes_refinements] if r['loss'] < 5.0]

        if top_configs:
            cmaes_results = run_cmaes_phase(
                args.model, args.train_minutes, output_dir, gpus,
                top_configs, target_params,
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
                'all_results': [{'params': r['params'], 'loss': r['loss']} for r in results[:50]],
                'elapsed_hours': elapsed,
                'total_evals': len(results),
            }, f, indent=2, default=str)
        print(f"\nResults saved to: {results_file}")

    # Cleanup
    cleanup_checkpoints(output_dir)

    return results


if __name__ == '__main__':
    main()
