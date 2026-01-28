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
    calc_mingru_params, calc_minlstm_params, calc_mom_e88_params
)

# Supported n_state values for E88 CUDA fused gate kernel (must match head_v_dim for default config)
# Only these sizes are efficiently supported without warnings/fallbacks: 16, 32, 48, 64
E88_SUPPORTED_N_STATE = [16, 32, 48, 64]

# Search space definitions for each model
# NOTE: LR is FIXED at 3e-4 for all models to match benchmark conditions
SEARCH_SPACES = {
    'e88': {
        # Parameter: (min, max, type, description)
        'n_heads': (32, 160, 'int', 'Number of attention heads'),  # Expanded range
        'n_state': (16, 64, 'e88_n_state', 'State dimension (only 16,32,48,64 supported by fused kernel)'),
        'depth': (12, 40, 'int', 'Number of layers'),  # Allow shallower networks
        # LR removed - fixed at 3e-4 for fair comparison
    },
    'fla-gdn': {
        'expansion': (1, 3, 'int', 'FFN expansion factor (must be int for FLA)'),
        'depth': (16, 40, 'int', 'Number of layers'),
        'n_heads': (8, 32, 'int', 'Number of heads'),
        # LR fixed at 3e-4
    },
    'mamba2': {
        'd_state': (64, 256, 'int_mult16', 'SSM state dimension'),
        'expand': (1, 3, 'int', 'Expansion factor'),
        'depth': (16, 40, 'int', 'Number of layers'),
        # LR fixed at 3e-4
    },
    'transformer': {
        'n_heads': (8, 32, 'int', 'Number of attention heads'),
        'expansion': (2, 6, 'int', 'FFN expansion factor'),
        'depth': (12, 36, 'int', 'Number of layers'),
        # LR fixed at 3e-4
    },
    'gru': {
        'expansion': (1, 3, 'int', 'Expansion factor for dim_inner'),
        'depth': (12, 48, 'int', 'Number of layers'),
    },
    'lstm': {
        'expansion': (1, 3, 'int', 'Expansion factor for dim_inner'),
        'depth': (12, 48, 'int', 'Number of layers'),
    },
    'mingru': {
        'expansion': (1, 4, 'int', 'Expansion factor'),
        'depth': (12, 40, 'int', 'Number of layers'),
    },
    'minlstm': {
        'expansion': (1, 4, 'int', 'Expansion factor'),
        'depth': (12, 40, 'int', 'Number of layers'),
    },
    'mom-e88': {
        # Mixture of Memory E88: sparse top-K routing to memory heads
        # Balanced search space - ensure dim stays reasonable (>512)
        'n_heads': (32, 256, 'int', 'Total number of memory heads'),
        'top_k': (8, 96, 'int', 'Active heads per token'),
        'n_state': (16, 64, 'e88_n_state', 'State dimension (only 16,32,48,64 supported)'),
        'depth': (8, 32, 'int', 'Number of layers'),
        # LR fixed at 3e-4
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
        elif ptype == 'int_mult16':
            raw = lo + val * (hi - lo)
            params[name] = int(round(raw / 16) * 16)
            params[name] = max(16, params[name])
        elif ptype == 'e88_n_state':
            # Map to nearest supported n_state value
            raw = lo + val * (hi - lo)
            # Find closest supported value
            closest = min(E88_SUPPORTED_N_STATE, key=lambda x: abs(x - raw))
            params[name] = closest
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

        if ptype == 'int' or ptype == 'int_mult16' or ptype == 'e88_n_state':
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


# Known best configs for warm-starting
BEST_CONFIGS = {
    'e88': {
        'n_heads': 98,
        'n_state': 32,
        'depth': 14,
    },
    'fla-gdn': {
        'expansion': 2,
        'depth': 24,
        'n_heads': 16,
    },
    'mamba2': {
        'd_state': 64,
        'expand': 2,
        'depth': 28,
    },
    'transformer': {
        'n_heads': 16,
        'expansion': 4,
        'depth': 24,
    },
    'gru': {
        'expansion': 1,
        'depth': 20,
    },
    'lstm': {
        'expansion': 1,
        'depth': 20,
    },
    'mingru': {
        'expansion': 2,
        'depth': 20,
    },
    'minlstm': {
        'expansion': 2,
        'depth': 20,
    },
    'mom-e88': {
        # Start from middle of search space for broad exploration
        'n_heads': 196,  # Middle of 32-512 range
        'top_k': 48,     # Middle of 4-128 range
        'n_state': 32,   # Sweet spot from E88
        'depth': 20,     # Middle of 6-48 range
    },
}


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

    # Fixed LR for all models at 3e-4
    lr = 3e-4

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
        cmd.extend([
            '--level', 'E88',
            '--n_heads', str(params['n_heads']),
            '--n_state', str(params['n_state']),
            '--expansion', '1.0',
            '--use_gate', '1',
            '--gate_activation', 'silu',
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
        ])
    elif model_type == 'transformer':
        cmd.extend([
            '--level', 'llama',
            '--n_heads', str(params.get('n_heads', 16)),
            '--expansion', str(params.get('expansion', 4)),
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
    elif model_type == 'minlstm':
        cmd.extend([
            '--level', 'minlstm',
            '--expansion', str(params.get('expansion', 2)),
        ])
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
    dim, actual_params = estimate_dim_and_params(params, model_type, target_params)

    # Skip if params too far from target
    if abs(actual_params - target_params) > tolerance:
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


def run_cmaes_search(model_type, generations, train_minutes, gpu_ids, output_dir, target_params, tolerance, start_from_best=False):
    """Run CMA-ES search for optimal configuration."""
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

    opts = {
        'maxiter': generations,
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
    print(f"Generations: {generations}")
    print(f"Population: {popsize}")
    print(f"GPUs: {gpu_ids} ({n_gpus} parallel)")
    print(f"Total evaluations: ~{generations * popsize}")
    print(f"Estimated time: ~{generations * train_minutes:.0f} min ({generations * train_minutes / 60:.1f} hours)")
    print("=" * 70)

    # Track best
    eval_count = 0
    best_loss = float('inf')
    best_params = None
    history = []

    # Run CMA-ES
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    while not es.stop():
        gen = es.countiter + 1
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

        # Save checkpoint
        checkpoint = {
            'generation': gen,
            'best_loss': best_loss,
            'best_params': best_params,
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

    return best_params, best_loss


def main():
    parser = argparse.ArgumentParser(description='CMA-ES search for optimal model config')
    parser.add_argument('--model', type=str, required=True,
                        choices=['e88', 'fla-gdn', 'mamba2', 'transformer', 'gru', 'lstm', 'mingru', 'minlstm', 'mom-e88'],
                        help='Model type to optimize')
    parser.add_argument('--generations', type=int, default=20,
                        help='Number of CMA-ES generations')
    parser.add_argument('--train_minutes', type=float, default=2,
                        help='Training time per config (minutes)')
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
        start_from_best=args.start_from_best
    )

    print(f"\nTo train with best config:")
    dim, _ = estimate_dim_and_params(best_params, args.model, target_params)
    if args.model == 'e88':
        print(f"python train.py --level E88 --dim {dim} --n_heads {best_params['n_heads']} "
              f"--n_state {best_params['n_state']} --depth {best_params['depth']} "
              f"--lr 3e-4 --expansion 1.0 --use_gate 1 --gate_activation silu --train_minutes 30")
    elif args.model == 'fla-gdn':
        print(f"python train.py --level fla-gdn --dim {dim} --depth {best_params['depth']} "
              f"--expansion {best_params['expansion']} --n_heads {best_params.get('n_heads', 16)} "
              f"--lr 3e-4 --train_minutes 30")
    elif args.model == 'mamba2':
        print(f"python train.py --level mamba2 --dim {dim} --depth {best_params['depth']} "
              f"--lr 3e-4 --train_minutes 30")
    elif args.model == 'transformer':
        print(f"python train.py --level llama --dim {dim} --depth {best_params['depth']} "
              f"--n_heads {best_params.get('n_heads', 16)} --expansion {best_params.get('expansion', 4)} "
              f"--lr 3e-4 --train_minutes 30")
    elif args.model in ['gru', 'lstm']:
        level = 'cudagru' if args.model == 'gru' else 'cudalstm'
        print(f"python train.py --level {level} --dim {dim} --depth {best_params['depth']} "
              f"--expansion {best_params.get('expansion', 1)} --lr 3e-4 --train_minutes 30")
    elif args.model in ['mingru', 'minlstm']:
        print(f"python train.py --level {args.model} --dim {dim} --depth {best_params['depth']} "
              f"--expansion {best_params.get('expansion', 2)} --lr 3e-4 --train_minutes 30")
    elif args.model == 'mom-e88':
        print(f"python train.py --level MoME88 --dim {dim} --n_heads {best_params['n_heads']} "
              f"--top_k {best_params['top_k']} --n_state {best_params['n_state']} "
              f"--depth {best_params['depth']} --lr 3e-4 --expansion 1.0 --use_gate 1 "
              f"--gate_activation silu --train_minutes 30")


if __name__ == '__main__':
    main()
