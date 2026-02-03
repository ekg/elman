#!/usr/bin/env python3
"""
E74 CUDA Kernel Benchmark: Test all 16 configurations (2 update × 4 proj × 2 nonlin)

Runs 8 configs at a time across 8 GPUs, 10-minute training each.
Uses the same pattern as run_100m_benchmark.py.
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
from itertools import product

# Benchmark settings
TARGET_PARAMS = 100_000_000
TRAIN_MINUTES = 10
TARGET_DEPTH = 20
OUTPUT_DIR = Path("benchmark_results/e74_cuda_10min")
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
        gpus.sort(key=lambda x: -x[1])
        return [g[0] for g in gpus]
    except Exception as e:
        print(f"Warning: Could not detect GPUs: {e}")
        return list(range(8))


def align_to_128(x):
    return ((x + 63) // 128) * 128


def count_e74_params(dim, depth, n_state, proj_type, vocab_size=256, expansion=2.0):
    """Count E74 diagonal state parameters.

    Projection types:
    - tied_kvq (0): single W projection
    - tied_kq (1): W_kq + W_v (2 projections)
    - no_z (2): W_k, W_v, W_q (3 projections)
    - full (3): W_k, W_v, W_q, W_z (4 projections)
    """
    d_inner = int(dim * expansion)

    proj_params = {
        'tied_kvq': n_state * d_inner,           # single W
        'tied_kq': 2 * n_state * d_inner,        # W_kq, W_v
        'no_z': 3 * n_state * d_inner,           # W_k, W_v, W_q
        'full': 4 * n_state * d_inner + n_state, # W_k, W_v, W_q, W_z + b_z
    }

    per_layer = (
        dim * d_inner +              # in_proj
        proj_params[proj_type] +     # projection weights
        n_state * dim +              # out_proj
        2 * dim                      # LayerNorm
    )

    total = (
        vocab_size * dim +           # embedding
        per_layer * depth +          # layers
        2 * dim                      # final norm
    )
    return total


def find_e74_config(target_params, proj_type, n_state=256, target_depth=20, expansion=2.0):
    """Find dim to hit target_params for E74 with given proj_type."""
    depth = target_depth
    best_dim = 128
    best_params = count_e74_params(128, depth, n_state, proj_type, expansion=expansion)
    best_diff = abs(best_params - target_params)

    for dim in range(128, 2049, 128):
        params = count_e74_params(dim, depth, n_state, proj_type, expansion=expansion)
        diff = abs(params - target_params)
        if diff < best_diff:
            best_diff = diff
            best_dim = dim
            best_params = params

    return best_dim, depth, best_params


# E74 Configuration Space
UPDATE_TYPES = ['delta', 'simple']
PROJ_TYPES = ['tied_kvq', 'tied_kq', 'no_z', 'full']  # Maps to 0, 1, 2, 3
NONLIN_TYPES = ['tanh', 'linear']
N_STATE = 256  # Diagonal state size
EXPANSION = 2.0

# Map names to CUDA enum values
PROJ_TYPE_MAP = {'tied_kvq': 0, 'tied_kq': 1, 'no_z': 2, 'full': 3}


def main():
    print("=" * 70)
    print("E74 CUDA Kernel Benchmark")
    print(f"Configs: {len(UPDATE_TYPES)} update × {len(PROJ_TYPES)} proj × {len(NONLIN_TYPES)} nonlin = 16")
    print(f"Training: {TRAIN_MINUTES} minutes each, n_state={N_STATE}")
    print("=" * 70)

    configs = {}

    # Generate all 16 configurations
    for update, proj, nonlin in product(UPDATE_TYPES, PROJ_TYPES, NONLIN_TYPES):
        name = f"e74_{update}_{proj}_{nonlin}"
        dim, depth, params = find_e74_config(TARGET_PARAMS, proj, N_STATE, TARGET_DEPTH, EXPANSION)

        configs[name] = {
            'dim': dim,
            'depth': depth,
            'params': params,
            'n_state': N_STATE,
            'expansion': EXPANSION,
            'update_type': update,
            'proj_type': proj,
            'proj_type_id': PROJ_TYPE_MAP[proj],
            'nonlin_type': nonlin,
            'use_tanh': nonlin == 'tanh',
        }
        print(f"{name:40s} dim={dim:4d} depth={depth:2d} params={params:,}")

    # Save configs
    with open(OUTPUT_DIR / "configs.json", "w") as f:
        json.dump(configs, f, indent=2)

    # Get available GPUs
    available_gpus = get_available_gpus()
    print(f"\nAvailable GPUs: {available_gpus}")

    # Model order (priority)
    model_order = list(configs.keys())

    print("\n" + "=" * 70)
    print(f"Starting {TRAIN_MINUTES}-minute benchmarks")
    print(f"Models: {len(model_order)}, GPUs available: {len(available_gpus)}")
    print("=" * 70)

    def launch_job(model_name, gpu_id, config):
        """Launch E74 training job on a specific GPU."""
        # Build command
        cmd_parts = [
            "python train_e74.py",
            f"--data data/pile.txt",
            f"--dim {config['dim']}",
            f"--depth {config['depth']}",
            f"--n_state {config['n_state']}",
            f"--expansion {config['expansion']}",
            f"--update_type {config['update_type']}",
            f"--proj_type {config['proj_type']}",
            f"--nonlin_type {config['nonlin_type']}",
            f"--batch_size 32",
            f"--chunk_size 512",
            f"--steps 999999",
            f"--train_minutes {TRAIN_MINUTES}",
            f"--lr 3e-4",
            f"--log_every 10",
            f"--output {OUTPUT_DIR}/{model_name}",
            f"--bf16",
            f"--use_cuda",
        ]

        cmd = " ".join(cmd_parts)
        log_file = OUTPUT_DIR / f"{model_name}.log"

        p = subprocess.Popen(
            f"CUDA_VISIBLE_DEVICES={gpu_id} PYTHONUNBUFFERED=1 {cmd}",
            shell=True,
            stdout=open(log_file, "w"),
            stderr=subprocess.STDOUT
        )
        return p

    # Track jobs
    active_jobs = {}
    pending_models = list(model_order)
    completed_models = []

    # Initial batch
    for gpu_id in available_gpus:
        if not pending_models:
            break
        model_name = pending_models.pop(0)
        p = launch_job(model_name, gpu_id, configs[model_name])
        active_jobs[model_name] = (p, gpu_id)
        print(f"Launched {model_name} on GPU {gpu_id}")

    # Monitor and launch remaining
    while active_jobs or pending_models:
        time.sleep(5)

        # Check for completed
        completed = []
        for model_name, (p, gpu_id) in active_jobs.items():
            ret = p.poll()
            if ret is not None:
                completed.append((model_name, gpu_id, ret))

        # Process completed and launch new
        for model_name, gpu_id, ret in completed:
            del active_jobs[model_name]
            completed_models.append(model_name)
            status = "OK" if ret == 0 else f"FAIL({ret})"
            print(f"{model_name} completed [{status}] (GPU {gpu_id} free)")

            if pending_models:
                next_model = pending_models.pop(0)
                p = launch_job(next_model, gpu_id, configs[next_model])
                active_jobs[next_model] = (p, gpu_id)
                print(f"Launched {next_model} on GPU {gpu_id}")

        if completed:
            print(f"Progress: {len(completed_models)}/{len(model_order)} complete, "
                  f"{len(active_jobs)} running, {len(pending_models)} pending")

    print("\n" + "=" * 70)
    print("All E74 CUDA benchmarks complete!")
    print(f"Results in: {OUTPUT_DIR}")
    print("=" * 70)

    # Parse results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Config':45s} {'Loss':>10s} {'Tok/s':>12s}")
    print("-" * 70)

    results = []
    for name in model_order:
        log_file = OUTPUT_DIR / f"{name}.log"
        if log_file.exists():
            try:
                with open(log_file) as f:
                    lines = f.readlines()
                # Find last-100 average or final loss
                loss = None
                toks = None
                for line in reversed(lines):
                    if 'last-100' in line.lower() or 'loss' in line.lower():
                        parts = line.split()
                        for i, p in enumerate(parts):
                            if 'loss' in p.lower() and i+1 < len(parts):
                                try:
                                    loss = float(parts[i+1].rstrip(','))
                                except:
                                    pass
                            if 'tok/s' in p.lower() or 'tokens/s' in p.lower():
                                try:
                                    toks = float(parts[i-1].replace(',', ''))
                                except:
                                    pass
                        if loss:
                            break

                if loss:
                    results.append((name, loss, toks or 0))
                    print(f"{name:45s} {loss:10.4f} {toks or 0:12,.0f}")
            except Exception as e:
                print(f"{name:45s} {'ERROR':>10s} {str(e)[:20]}")

    # Sort by loss
    if results:
        print("\n" + "=" * 70)
        print("RANKED BY LOSS")
        print("=" * 70)
        results.sort(key=lambda x: x[1])
        for i, (name, loss, toks) in enumerate(results, 1):
            print(f"{i:2d}. {name:43s} {loss:10.4f} {toks:12,.0f}")


if __name__ == "__main__":
    main()
