#!/usr/bin/env python3
"""
E74 Full Matrix 100M Benchmark

Comprehensive benchmark of E74 full matrix state variants vs baselines.
Tests all projection types (tied_kvq, tied_kq, no_z) across state sizes (32, 48, 64, 96).

Key question: Does full matrix state (O(n²)) provide meaningful improvement over
diagonal state (O(n)) to justify the computational cost?

Architecture:
    E74 Delta Rule: S = tanh(S + outer(v - S@k, k))
    - Erase before write (v - S@k), then write with normalized k
    - NOT E70's decay approach
    - CUDA kernel with gradient checkpointing for memory efficiency

Models (20 total):
    E74 Full Matrix: 12 variants (3 proj_types × 4 n_states)
    E74 Diagonal: 4 variants (2 proj_types × 2 n_states)
    Baselines: mamba2, fla-gdn, cudagru, cudalstm

Usage:
    python run_e74_fullmatrix_benchmark.py           # Run all 20 models
    python run_e74_fullmatrix_benchmark.py --batch 1 # First 8 models
    python run_e74_fullmatrix_benchmark.py --batch 2 # Second 8 models
    python run_e74_fullmatrix_benchmark.py --batch 3 # Third batch (4 E74 diag + 4 baselines)
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime


# Benchmark settings
TARGET_PARAMS = 100_000_000
TRAIN_MINUTES = 10
TARGET_DEPTH = 20
OUTPUT_DIR = Path("benchmark_results/e74_fullmatrix_100m")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# CUDA-supported n_state values (template specializations in kernel)
SUPPORTED_N_STATES = [32, 48, 64, 96]

# Test both expansion factors for E74 models
E74_EXPANSIONS = [1.0, 2.0]


def get_available_gpus():
    """Detect available GPUs."""
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
        gpus.sort(key=lambda x: -x[1])
        return [g[0] for g in gpus]
    except Exception as e:
        print(f"Warning: Could not detect GPUs: {e}")
        return list(range(8))


def align_to_128(x):
    """Round x to nearest multiple of 128."""
    return ((x + 63) // 128) * 128


def count_e74_fullmatrix_params(dim, depth, n_state, proj_type, expansion=1.0):
    """Count E74 full matrix parameters.

    E74 layer structure:
    - in_proj: dim -> d_inner (where d_inner = dim * expansion)
    - Cell projections depend on proj_type (from d_inner!):
      - tied_kvq: W_kvq [n_state, d_inner]
      - tied_kq: W_k [n_state, d_inner], W_v [n_state, d_inner]
      - no_z: W_k, W_v, W_q each [n_state, d_inner]
    - out_proj: n_state -> dim
    - RMSNorm: dim
    """
    vocab_size = 256
    d_inner = int(dim * expansion)

    # Base per-layer params
    per_layer = (
        dim * d_inner +          # in_proj
        n_state * dim +          # out_proj
        dim                      # RMSNorm
    )

    # Projection params (from d_inner, not dim!)
    if proj_type == 'tied_kvq':
        per_layer += n_state * d_inner
    elif proj_type == 'tied_kq':
        per_layer += 2 * n_state * d_inner
    elif proj_type == 'no_z':
        per_layer += 3 * n_state * d_inner

    total = (
        vocab_size * dim +       # embedding
        per_layer * depth +      # layers
        dim                      # final norm
    )
    return total


def count_baseline_params(model_type, dim, depth, expansion=2.0):
    """Count baseline model parameters."""
    d_inner = int(dim * expansion)
    vocab_size = 256

    if model_type == 'mamba2':
        # Mamba2 approximation
        embed = vocab_size * dim
        layer_params = depth * expansion * 3.05 * dim * dim
        norm_params = depth * 2 * dim + 2 * dim
        return int(embed + layer_params + norm_params)

    elif model_type == 'fla-gdn':
        # FLA GatedDeltaNet
        num_heads = max(1, dim // 128)
        per_layer = (
            dim * dim +              # q_proj
            dim * dim +              # k_proj
            dim * d_inner +          # v_proj
            dim * num_heads +        # a_proj
            dim * num_heads +        # b_proj
            dim * 4 +                # q_conv1d
            dim * 4 +                # k_conv1d
            d_inner * 4 +            # v_conv1d
            dim * d_inner +          # g_proj
            2 * d_inner +            # o_norm
            d_inner * dim +          # o_proj
            dim                      # layer RMSNorm
        )
        return vocab_size * dim + per_layer * depth + dim

    elif model_type == 'cudagru':
        # GRU
        per_layer = (
            dim * d_inner +          # in_proj
            3 * d_inner * d_inner +  # weight_ih
            3 * d_inner * d_inner +  # weight_hh
            6 * d_inner +            # biases
            d_inner * dim +          # out_proj
            2 * dim                  # LayerNorm
        )
        return vocab_size * dim + per_layer * depth + 2 * dim

    elif model_type == 'cudalstm':
        # LSTM
        per_layer = (
            dim * d_inner +          # in_proj
            4 * d_inner * d_inner +  # weight_ih
            4 * d_inner * d_inner +  # weight_hh
            8 * d_inner +            # biases
            d_inner * dim +          # out_proj
            2 * dim                  # LayerNorm
        )
        return vocab_size * dim + per_layer * depth + 2 * dim

    return 0


def find_dim_for_params(target_params, count_fn, depth=20):
    """Binary search for dim that hits target params."""
    best_dim = 128
    best_params = count_fn(128)
    best_diff = abs(best_params - target_params)

    for dim in range(128, 2561, 128):
        params = count_fn(dim)
        diff = abs(params - target_params)
        if diff < best_diff:
            best_diff = diff
            best_dim = dim
            best_params = params

    return best_dim, best_params


# ============================================================================
# Define all model configurations
# ============================================================================

def generate_all_configs():
    """Generate all benchmark configurations."""
    configs = {}

    # E74 Full Matrix variants - test both expansion factors
    # 12 variants per expansion × 2 expansions = 24 models
    for expansion in E74_EXPANSIONS:
        exp_suffix = f'-exp{expansion:.0f}' if expansion != 1.0 else ''
        for proj_type in ['tied_kvq', 'tied_kq', 'no_z']:
            for n_state in SUPPORTED_N_STATES:
                name = f'e74-full-{proj_type}-n{n_state}{exp_suffix}'
                dim, params = find_dim_for_params(
                    TARGET_PARAMS,
                    lambda d, ns=n_state, pt=proj_type, exp=expansion:
                        count_e74_fullmatrix_params(d, TARGET_DEPTH, ns, pt, exp)
                )
                configs[name] = {
                    'name': name,
                    'type': 'e74-full',
                    'dim': dim,
                    'depth': TARGET_DEPTH,
                    'n_state': n_state,
                    'proj_type': proj_type,
                    'state_type': 'full',
                    'expansion': expansion,
                    'params': params,
                }

    # E74 Diagonal variants - test both expansion factors
    # 4 variants per expansion × 2 expansions = 8 models
    for expansion in E74_EXPANSIONS:
        exp_suffix = f'-exp{expansion:.0f}' if expansion != 1.0 else ''
        for proj_type in ['tied_kvq', 'no_z']:
            for n_state in [64, 96]:
                name = f'e74-diag-{proj_type}-n{n_state}{exp_suffix}'
                dim, params = find_dim_for_params(
                    TARGET_PARAMS,
                    lambda d, ns=n_state, pt=proj_type, exp=expansion:
                        count_e74_fullmatrix_params(d, TARGET_DEPTH, ns, pt, exp)
                )
                configs[name] = {
                    'name': name,
                    'type': 'e74-diag',
                    'dim': dim,
                    'depth': TARGET_DEPTH,
                    'n_state': n_state,
                    'proj_type': proj_type,
                    'state_type': 'diagonal',
                    'expansion': expansion,
                    'params': params,
                }

    # Baselines
    # Mamba2: dim=896 (128-aligned), depth=20, expand=2 → ~98M params
    configs['mamba2'] = {
        'name': 'mamba2',
        'type': 'mamba2',
        'dim': 896,  # 7×128, 128-aligned
        'depth': 20,
        'expansion': 2.0,
        'params': 98_210_560,  # calculated: depth * expand * 3.05 * dim^2 + vocab*dim + norms
    }

    # FLA-GDN
    dim, params = find_dim_for_params(
        TARGET_PARAMS,
        lambda d: count_baseline_params('fla-gdn', d, TARGET_DEPTH)
    )
    configs['fla-gdn'] = {
        'name': 'fla-gdn',
        'type': 'fla-gdn',
        'dim': dim,
        'depth': TARGET_DEPTH,
        'expansion': 2.0,
        'params': params,
    }

    # CUDA GRU
    dim, params = find_dim_for_params(
        TARGET_PARAMS,
        lambda d: count_baseline_params('cudagru', d, TARGET_DEPTH)
    )
    configs['cudagru'] = {
        'name': 'cudagru',
        'type': 'cudagru',
        'dim': dim,
        'depth': TARGET_DEPTH,
        'expansion': 2.0,
        'params': params,
    }

    # CUDA LSTM
    dim, params = find_dim_for_params(
        TARGET_PARAMS,
        lambda d: count_baseline_params('cudalstm', d, TARGET_DEPTH)
    )
    configs['cudalstm'] = {
        'name': 'cudalstm',
        'type': 'cudalstm',
        'dim': dim,
        'depth': TARGET_DEPTH,
        'expansion': 2.0,
        'params': params,
    }

    return configs


def get_batch(batch_num, configs):
    """Get model names for a specific batch."""
    model_names = list(configs.keys())

    # Batch organization (36 total models):
    # Batch 1: E74 Full exp=1.0, tied_kvq + tied_kq (8)
    # Batch 2: E74 Full exp=1.0, no_z (4) + E74 Diag exp=1.0 (4) = 8
    # Batch 3: E74 Full exp=2.0, tied_kvq + tied_kq (8)
    # Batch 4: E74 Full exp=2.0, no_z (4) + E74 Diag exp=2.0 (4) = 8
    # Batch 5: Baselines (4)

    exp1_models = [n for n in model_names if 'exp2' not in n and n.startswith('e74')]
    exp2_models = [n for n in model_names if 'exp2' in n]
    baselines = [n for n in model_names if not n.startswith('e74')]

    if batch_num == 1:
        # E74 Full exp=1.0, tied_kvq + tied_kq
        return [n for n in exp1_models if 'e74-full' in n and ('tied_kvq' in n or 'tied_kq' in n)]
    elif batch_num == 2:
        # E74 Full exp=1.0 no_z + E74 Diag exp=1.0
        return [n for n in exp1_models if ('e74-full' in n and 'no_z' in n) or 'e74-diag' in n]
    elif batch_num == 3:
        # E74 Full exp=2.0, tied_kvq + tied_kq
        return [n for n in exp2_models if 'e74-full' in n and ('tied_kvq' in n or 'tied_kq' in n)]
    elif batch_num == 4:
        # E74 Full exp=2.0 no_z + E74 Diag exp=2.0
        return [n for n in exp2_models if ('e74-full' in n and 'no_z' in n) or 'e74-diag' in n]
    elif batch_num == 5:
        return baselines
    else:
        return model_names


def launch_job(model_name, gpu_id, config, train_minutes, output_dir):
    """Launch a training job on a specific GPU."""
    model_type = config['type']

    if model_type.startswith('e74'):
        # E74 models use train_e74.py
        expansion = config.get('expansion', 1.0)
        cmd_parts = [
            f"python train_e74.py",
            f"--data data/pile.txt",
            f"--dim {config['dim']}",
            f"--depth {config['depth']}",
            f"--n_state {config['n_state']}",
            f"--state_type {config['state_type']}",
            f"--proj_type {config['proj_type']}",
            f"--expansion {expansion}",
            f"--batch_size 32",
            f"--chunk_size 512",
            f"--steps 999999",
            f"--train_minutes {train_minutes}",
            f"--lr 3e-4",
            f"--log_every 10",
            f"--output {output_dir}/{model_name}",
            f"--bf16",
        ]
    elif model_type == 'mamba2':
        cmd_parts = [
            f"python train.py",
            f"--data data/pile.txt",
            f"--level mamba2",
            f"--dim {config['dim']}",
            f"--depth {config['depth']}",
            f"--batch_size 32",
            f"--chunk_size 512",
            f"--steps 999999",
            f"--train_minutes {train_minutes}",
            f"--lr 3e-4",
            f"--log_every 10",
            f"--output {output_dir}/{model_name}",
            f"--bf16",
        ]
    else:
        # Standard LadderLM models
        cmd_parts = [
            f"python train.py",
            f"--data data/pile.txt",
            f"--level {model_type}",
            f"--dim {config['dim']}",
            f"--depth {config['depth']}",
            f"--expansion {config.get('expansion', 2.0)}",
            f"--batch_size 32",
            f"--chunk_size 512",
            f"--steps 999999",
            f"--train_minutes {train_minutes}",
            f"--lr 3e-4",
            f"--log_every 10",
            f"--output {output_dir}/{model_name}",
            f"--bf16",
        ]

    cmd = " ".join(cmd_parts)

    log_file = output_dir / f"{model_name}.log"
    p = subprocess.Popen(
        f"CUDA_VISIBLE_DEVICES={gpu_id} PYTHONUNBUFFERED=1 {cmd}",
        shell=True,
        stdout=open(log_file, "w"),
        stderr=subprocess.STDOUT
    )
    return p


def extract_results(log_file):
    """Extract final loss and throughput from log file."""
    try:
        with open(log_file) as f:
            content = f.read()

        # Look for lines with step and loss
        lines = [l for l in content.split('\n') if 'step' in l and 'loss' in l]
        if not lines:
            return None, None

        # Get last 100 steps for average
        recent_losses = []
        recent_toks = []
        for line in lines[-100:]:
            parts = line.split('|')
            for p in parts:
                if 'loss' in p and 'loss/' not in p:
                    try:
                        recent_losses.append(float(p.split()[-1]))
                    except:
                        pass
                if 'tok/s' in p:
                    try:
                        recent_toks.append(float(p.split()[-1]))
                    except:
                        pass

        avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else None
        avg_toks = sum(recent_toks) / len(recent_toks) if recent_toks else None

        return avg_loss, avg_toks
    except Exception as e:
        return None, None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="E74 Full Matrix 100M Benchmark")
    parser.add_argument("--batch", type=int, choices=[1, 2, 3, 4, 5], help="Run specific batch (1-5)")
    parser.add_argument("--model", type=str, help="Run single model by name")
    parser.add_argument("--minutes", type=int, default=TRAIN_MINUTES, help="Training minutes")
    parser.add_argument("--list", action="store_true", help="List all configs and exit")
    args = parser.parse_args()

    # Generate configs
    configs = generate_all_configs()

    if args.list:
        print("=" * 80)
        print("E74 FULL MATRIX 100M BENCHMARK CONFIGURATIONS")
        print("=" * 80)

        # Group E74 models by expansion
        for expansion in E74_EXPANSIONS:
            exp_suffix = f'-exp{expansion:.0f}' if expansion != 1.0 else ''
            print(f"\n## E74 Full Matrix (expansion={expansion})")
            print("-" * 70)
            for name, cfg in sorted(configs.items()):
                if cfg['type'] == 'e74-full' and cfg.get('expansion', 1.0) == expansion:
                    print(f"{name:40s}: dim={cfg['dim']:4d}, n_state={cfg['n_state']:2d}, "
                          f"depth={cfg['depth']}, params={cfg['params']:>12,}")

            print(f"\n## E74 Diagonal (expansion={expansion})")
            print("-" * 70)
            for name, cfg in sorted(configs.items()):
                if cfg['type'] == 'e74-diag' and cfg.get('expansion', 1.0) == expansion:
                    print(f"{name:40s}: dim={cfg['dim']:4d}, n_state={cfg['n_state']:2d}, "
                          f"depth={cfg['depth']}, params={cfg['params']:>12,}")

        print()
        print("## Baselines (4 models)")
        print("-" * 70)
        for name, cfg in sorted(configs.items()):
            if cfg['type'] not in ['e74-full', 'e74-diag']:
                print(f"{name:40s}: dim={cfg['dim']:4d}, depth={cfg['depth']}, "
                      f"params={cfg['params']:>12,}")

        print()
        print("## Batch Organization (5 batches)")
        print("-" * 70)
        for i in range(1, 6):
            batch = get_batch(i, configs)
            print(f"Batch {i} ({len(batch)} models): {batch}")

        total = sum(len(get_batch(i, configs)) for i in range(1, 6))
        print(f"\nTotal: {total} models")

        return

    # Determine which models to run
    if args.model:
        if args.model not in configs:
            print(f"Error: Unknown model '{args.model}'")
            print(f"Available: {list(configs.keys())}")
            return
        models_to_run = [args.model]
    elif args.batch:
        models_to_run = get_batch(args.batch, configs)
    else:
        models_to_run = list(configs.keys())

    print("=" * 70)
    print("E74 Full Matrix 100M Benchmark")
    print(f"Training time: {args.minutes} minutes")
    print(f"Models: {len(models_to_run)}")
    print("=" * 70)

    for name in models_to_run:
        cfg = configs[name]
        print(f"{name:35s}: dim={cfg['dim']:4d}, params={cfg['params']:>12,}")

    # Save configs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_file = OUTPUT_DIR / f"configs_{timestamp}.json"
    with open(config_file, "w") as f:
        json.dump({name: configs[name] for name in models_to_run}, f, indent=2)
    print(f"\nSaved configs to {config_file}")

    # Get available GPUs
    available_gpus = get_available_gpus()
    print(f"Available GPUs: {available_gpus}")

    print("\n" + "=" * 70)
    print(f"Starting {args.minutes}-minute benchmarks")
    print("=" * 70)

    # Track jobs
    active_jobs = {}
    pending_models = list(models_to_run)
    completed_models = []

    # Launch initial batch
    for gpu_id in available_gpus:
        if not pending_models:
            break
        model_name = pending_models.pop(0)
        p = launch_job(model_name, gpu_id, configs[model_name], args.minutes, OUTPUT_DIR)
        active_jobs[model_name] = (p, gpu_id)
        print(f"Launched {model_name} on GPU {gpu_id}")

    # Monitor and launch remaining jobs
    while active_jobs or pending_models:
        time.sleep(5)

        completed = []
        for model_name, (p, gpu_id) in active_jobs.items():
            ret = p.poll()
            if ret is not None:
                completed.append((model_name, gpu_id, ret))

        for model_name, gpu_id, ret in completed:
            del active_jobs[model_name]
            completed_models.append(model_name)
            print(f"{model_name} completed with exit code {ret} (GPU {gpu_id} free)")

            if pending_models:
                next_model = pending_models.pop(0)
                p = launch_job(next_model, gpu_id, configs[next_model], args.minutes, OUTPUT_DIR)
                active_jobs[next_model] = (p, gpu_id)
                print(f"Launched {next_model} on GPU {gpu_id}")

        if completed:
            print(f"Progress: {len(completed_models)}/{len(models_to_run)} complete, "
                  f"{len(active_jobs)} running, {len(pending_models)} pending")

    print("\n" + "=" * 70)
    print("All benchmarks complete!")
    print("=" * 70)

    # Results summary
    print("\nResults Summary (Last-100 Avg):")
    print("-" * 70)
    print(f"{'Model':<35} {'Params':>12} {'Loss':>10} {'Tok/s':>10}")
    print("-" * 70)

    results = []
    for model_name in completed_models:
        log_file = OUTPUT_DIR / f"{model_name}.log"
        loss, toks = extract_results(log_file)
        cfg = configs[model_name]
        results.append((model_name, cfg['params'], loss, toks))

        loss_str = f"{loss:.4f}" if loss else "N/A"
        toks_str = f"{toks:.0f}" if toks else "N/A"
        print(f"{model_name:<35} {cfg['params']:>12,} {loss_str:>10} {toks_str:>10}")

    print("-" * 70)

    # Sort by loss
    print("\nRanking by Loss:")
    results_with_loss = [(n, p, l, t) for n, p, l, t in results if l is not None]
    results_with_loss.sort(key=lambda x: x[2])
    for i, (name, params, loss, toks) in enumerate(results_with_loss, 1):
        print(f"{i:2d}. {name:<35} loss={loss:.4f} tok/s={toks:.0f}")

    # Save results
    results_file = OUTPUT_DIR / f"results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump([
            {'name': n, 'params': p, 'loss': l, 'tok_s': t}
            for n, p, l, t in results
        ], f, indent=2)
    print(f"\nSaved results to {results_file}")


if __name__ == "__main__":
    main()
