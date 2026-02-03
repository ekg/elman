#!/usr/bin/env python3
"""
E74 Ablation 100M Benchmark

Systematic 10-minute benchmark of all E74 ablation configs at ~100M parameters.
Uses real data (pile.txt) and same training setup as run_100m_benchmark.py.

The key question: How minimal can we make E73 while maintaining performance?

Ablation dimensions:
- State: full (E73), diagonal, lowrank, blockdiag
- Projection: full (k,v,q,z), no_z, tied_kq, tied_kvq
- Nonlinearity: tanh, linear, rmsnorm
- Gate: output, retain, state

Usage:
    python run_e74_100m_benchmark.py           # Run all 20 configs
    python run_e74_100m_benchmark.py --quick   # Quick 2-min test
    python run_e74_100m_benchmark.py --phase 1 # State structure only
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime


# Target: 100M params with expansion=2.0, same as other benchmarks
TARGET_PARAMS = 100_000_000
EXPANSION = 2.0
TRAIN_MINUTES = 10
TARGET_DEPTH = 20
OUTPUT_DIR = Path("benchmark_results/e74_100m")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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


def count_e74_params(dim, depth, n_state, state_type, proj_type, expansion=2.0, rank=8, block_size=8):
    """Count E74 parameters for a specific config.

    E74 layer structure:
    - in_proj: dim -> d_inner
    - cell: depends on state_type and proj_type
    - out_proj: n_state -> dim
    - LayerNorm: 2 * dim
    """
    d_inner = int(dim * expansion)
    vocab_size = 256

    # Base per-layer params
    per_layer = (
        dim * d_inner +          # in_proj
        n_state * dim +          # out_proj
        2 * dim                  # LayerNorm
    )

    # Projection params depend on proj_type
    if proj_type == 'full':
        # W_k, W_v, W_q, W_z, b_z
        per_layer += (
            4 * n_state * d_inner +  # W_k, W_v, W_q, W_z
            n_state                   # b_z
        )
    elif proj_type == 'no_z':
        # W_k, W_v, W_q
        per_layer += 3 * n_state * d_inner
    elif proj_type == 'tied_kq':
        # W_k (=W_q), W_v
        per_layer += 2 * n_state * d_inner
    elif proj_type == 'tied_kvq':
        # W (k=v=q)
        per_layer += n_state * d_inner

    # Gate params (for retain, state gate types)
    # d_g, b_g: n_state each
    # Adding these for all configs since some use them

    # Low-rank has additional projection
    if state_type == 'lowrank':
        per_layer += rank * n_state  # W_kr

    total = (
        vocab_size * dim +       # embedding
        per_layer * depth +      # layers
        2 * dim                  # final norm
    )
    return total


def find_config_for_100m(state_type, proj_type, n_state, rank=8, block_size=8, target_depth=20):
    """Find dim to hit ~100M params for given E74 config."""
    depth = target_depth
    best_dim = 128
    best_params = count_e74_params(128, depth, n_state, state_type, proj_type, EXPANSION, rank, block_size)
    best_diff = abs(best_params - TARGET_PARAMS)

    for dim in range(128, 2049, 128):
        params = count_e74_params(dim, depth, n_state, state_type, proj_type, EXPANSION, rank, block_size)
        diff = abs(params - TARGET_PARAMS)
        if diff < best_diff:
            best_diff = diff
            best_dim = dim
            best_params = params

    return best_dim, depth, best_params


# Define all ablation configs (from e74_ablations.py)
ABLATION_CONFIGS = [
    # Phase 1: State structure with baseline projections (delta rule)
    {'id': 1, 'state': 'full', 'proj': 'full', 'nonlin': 'tanh', 'gate': 'output', 'desc': 'E73 baseline'},
    {'id': 2, 'state': 'full', 'proj': 'no_z', 'nonlin': 'tanh', 'gate': 'output', 'desc': 'Remove z'},
    {'id': 3, 'state': 'full', 'proj': 'tied_kq', 'nonlin': 'tanh', 'gate': 'output', 'desc': 'Tie k=q'},
    {'id': 4, 'state': 'full', 'proj': 'tied_kvq', 'nonlin': 'tanh', 'gate': 'output', 'desc': 'Tie k=v=q'},
    {'id': 5, 'state': 'diagonal', 'proj': 'full', 'nonlin': 'tanh', 'gate': 'output', 'desc': 'Diagonal baseline'},
    {'id': 6, 'state': 'diagonal', 'proj': 'no_z', 'nonlin': 'tanh', 'gate': 'output', 'desc': 'Diagonal, no z'},
    {'id': 7, 'state': 'diagonal', 'proj': 'tied_kq', 'nonlin': 'tanh', 'gate': 'output', 'desc': 'Diagonal, tied k=q'},
    {'id': 8, 'state': 'diagonal', 'proj': 'tied_kvq', 'nonlin': 'tanh', 'gate': 'output', 'desc': 'Diagonal, k=v=q'},
    {'id': 9, 'state': 'lowrank', 'proj': 'no_z', 'nonlin': 'tanh', 'gate': 'output', 'desc': 'Lowrank-4', 'rank': 4},
    {'id': 10, 'state': 'lowrank', 'proj': 'no_z', 'nonlin': 'tanh', 'gate': 'output', 'desc': 'Lowrank-8', 'rank': 8},
    {'id': 11, 'state': 'blockdiag', 'proj': 'no_z', 'nonlin': 'tanh', 'gate': 'output', 'desc': 'Block-diag'},

    # Phase 2: Nonlinearity variants
    {'id': 12, 'state': 'diagonal', 'proj': 'tied_kq', 'nonlin': 'linear', 'gate': 'output', 'desc': 'Diag, linear (E42-style)'},
    {'id': 13, 'state': 'diagonal', 'proj': 'tied_kq', 'nonlin': 'rmsnorm', 'gate': 'output', 'desc': 'Diag, rmsnorm'},
    {'id': 14, 'state': 'full', 'proj': 'no_z', 'nonlin': 'linear', 'gate': 'output', 'desc': 'Full, linear'},

    # Phase 3: Gate variants
    {'id': 15, 'state': 'diagonal', 'proj': 'no_z', 'nonlin': 'tanh', 'gate': 'retain', 'desc': 'Diag, retain gate'},
    {'id': 16, 'state': 'diagonal', 'proj': 'no_z', 'nonlin': 'tanh', 'gate': 'state', 'desc': 'Diag, state gate'},
    {'id': 17, 'state': 'diagonal', 'proj': 'tied_kq', 'nonlin': 'linear', 'gate': 'retain', 'desc': 'Diag, linear, retain'},

    # Phase 4: Best combo candidates (delta rule)
    {'id': 18, 'state': 'lowrank', 'proj': 'tied_kq', 'nonlin': 'tanh', 'gate': 'output', 'desc': 'Lowrank-4, tied k=q', 'rank': 4},
    {'id': 19, 'state': 'lowrank', 'proj': 'tied_kq', 'nonlin': 'linear', 'gate': 'output', 'desc': 'Lowrank-8, linear', 'rank': 8},
    {'id': 20, 'state': 'diagonal', 'proj': 'tied_kvq', 'nonlin': 'linear', 'gate': 'output', 'desc': 'Minimal: diag, tied, linear'},

    # Phase 5: Simple update (no delta rule) - just decay + write
    # Tests if "erase before write" (v - S@k) is necessary, or if simple accumulate works
    # S = f(α*S + outer(v, k)) instead of S = f(S + outer(v - S@k, k))
    {'id': 21, 'state': 'full', 'proj': 'tied_kvq', 'nonlin': 'tanh', 'gate': 'output', 'update': 'simple',
     'desc': 'Simple: full, tanh'},
    {'id': 22, 'state': 'diagonal', 'proj': 'tied_kvq', 'nonlin': 'tanh', 'gate': 'output', 'update': 'simple',
     'desc': 'Simple: diag, tanh'},
    {'id': 23, 'state': 'full', 'proj': 'tied_kvq', 'nonlin': 'linear', 'gate': 'output', 'update': 'simple',
     'desc': 'Simple: full, linear'},
    {'id': 24, 'state': 'diagonal', 'proj': 'tied_kvq', 'nonlin': 'linear', 'gate': 'output', 'update': 'simple',
     'desc': 'Simple: diag, linear'},
]


def get_configs_for_phase(phase):
    """Get config IDs for each phase."""
    if phase == 1:
        return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # State structure
    elif phase == 2:
        return [12, 13, 14]  # Nonlinearity
    elif phase == 3:
        return [15, 16, 17]  # Gates
    elif phase == 4:
        return [18, 19, 20]  # Best combos (delta)
    elif phase == 5:
        return [21, 22, 23, 24]  # Simple update (no delta)
    else:
        return list(range(1, 25))  # All 24 configs


def main():
    import argparse
    parser = argparse.ArgumentParser(description="E74 Ablation 100M Benchmark")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5], help="Run specific phase")
    parser.add_argument("--config", type=int, help="Run single config by ID (1-24)")
    parser.add_argument("--quick", action="store_true", help="Quick 2-min test")
    parser.add_argument("--n_state", type=int, default=96, help="State dimension for all configs")
    parser.add_argument("--minutes", type=int, default=TRAIN_MINUTES, help="Training minutes")
    args = parser.parse_args()

    train_minutes = 2 if args.quick else args.minutes
    n_state = args.n_state

    # Determine which configs to run
    if args.config is not None:
        config_ids = [args.config]
    elif args.phase is not None:
        config_ids = get_configs_for_phase(args.phase)
    else:
        config_ids = list(range(1, 25))  # All 24 configs

    configs_to_run = [c for c in ABLATION_CONFIGS if c['id'] in config_ids]

    print("=" * 70)
    print(f"E74 Ablation 100M Benchmark")
    print(f"Training time: {train_minutes} minutes per config")
    print(f"n_state: {n_state}")
    print(f"Configs: {len(configs_to_run)}")
    print("=" * 70)

    # Calculate dimensions for each config
    model_configs = {}
    for cfg in configs_to_run:
        state_type = cfg['state']
        proj_type = cfg['proj']
        rank = cfg.get('rank', 8)
        block_size = cfg.get('block_size', 8)
        update_type = cfg.get('update', 'delta')

        dim, depth, params = find_config_for_100m(
            state_type, proj_type, n_state, rank, block_size, TARGET_DEPTH
        )

        model_configs[cfg['id']] = {
            'dim': dim,
            'depth': depth,
            'params': params,
            'n_state': n_state,
            'state_type': state_type,
            'proj_type': proj_type,
            'nonlin_type': cfg['nonlin'],
            'gate_type': cfg['gate'],
            'update_type': update_type,
            'rank': rank,
            'block_size': block_size,
            'desc': cfg['desc'],
        }

        update_str = f" [{update_type}]" if update_type != 'delta' else ""
        print(f"Config {cfg['id']:2d} ({cfg['desc']:<25}): dim={dim}, depth={depth}, params={params:,}{update_str}")

    # Save configs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(OUTPUT_DIR / f"configs_{timestamp}.json", "w") as f:
        json.dump(model_configs, f, indent=2)

    # Get available GPUs
    available_gpus = get_available_gpus()
    print(f"\nAvailable GPUs: {available_gpus}")

    print("\n" + "=" * 70)
    print(f"Starting {train_minutes}-minute benchmarks")
    print("=" * 70)

    def launch_job(config_id, gpu_id, cfg):
        """Launch training job using E74 checkpointed layer."""

        # Build training command
        # Use a custom training script that wraps E74
        cmd_parts = [
            f"python train_e74_ablation.py",
            f"--data data/pile.txt",
            f"--dim {cfg['dim']}",
            f"--depth {cfg['depth']}",
            f"--n_state {cfg['n_state']}",
            f"--state_type {cfg['state_type']}",
            f"--proj_type {cfg['proj_type']}",
            f"--nonlin_type {cfg['nonlin_type']}",
            f"--gate_type {cfg['gate_type']}",
            f"--update_type {cfg['update_type']}",
            f"--rank {cfg['rank']}",
            f"--block_size {cfg['block_size']}",
            f"--batch_size 32",
            f"--chunk_size 512",
            f"--steps 999999",
            f"--train_minutes {train_minutes}",
            f"--lr 3e-4",
            f"--log_every 10",
            f"--output {OUTPUT_DIR}/config{config_id:02d}",
            f"--bf16",
            f"--expansion {EXPANSION}",
            f"--checkpointed",
        ]

        cmd = " ".join(cmd_parts)

        log_file = OUTPUT_DIR / f"config{config_id:02d}.log"
        p = subprocess.Popen(
            f"CUDA_VISIBLE_DEVICES={gpu_id} PYTHONUNBUFFERED=1 {cmd}",
            shell=True,
            stdout=open(log_file, "w"),
            stderr=subprocess.STDOUT
        )
        return p

    # Track jobs
    active_jobs = {}
    pending_configs = list(config_ids)
    completed_configs = []

    # Launch initial batch
    for gpu_id in available_gpus:
        if not pending_configs:
            break
        config_id = pending_configs.pop(0)
        if config_id in model_configs:
            p = launch_job(config_id, gpu_id, model_configs[config_id])
            active_jobs[config_id] = (p, gpu_id)
            print(f"Launched config {config_id} ({model_configs[config_id]['desc']}) on GPU {gpu_id}")

    # Monitor and launch remaining jobs
    while active_jobs or pending_configs:
        time.sleep(5)

        completed = []
        for config_id, (p, gpu_id) in active_jobs.items():
            ret = p.poll()
            if ret is not None:
                completed.append((config_id, gpu_id, ret))

        for config_id, gpu_id, ret in completed:
            del active_jobs[config_id]
            completed_configs.append(config_id)
            print(f"Config {config_id} completed with exit code {ret} (GPU {gpu_id} free)")

            if pending_configs:
                next_config = pending_configs.pop(0)
                if next_config in model_configs:
                    p = launch_job(next_config, gpu_id, model_configs[next_config])
                    active_jobs[next_config] = (p, gpu_id)
                    print(f"Launched config {next_config} ({model_configs[next_config]['desc']}) on GPU {gpu_id}")

        if completed:
            print(f"Progress: {len(completed_configs)}/{len(configs_to_run)} complete")

    print("\n" + "=" * 70)
    print("All benchmarks complete!")
    print("=" * 70)

    # Extract results from logs
    print("\nResults Summary:")
    print("-" * 70)
    print(f"{'ID':>3} {'Description':<30} {'Final Loss':>12} {'Tok/s':>10}")
    print("-" * 70)

    for config_id in sorted(completed_configs):
        log_file = OUTPUT_DIR / f"config{config_id:02d}.log"
        desc = model_configs[config_id]['desc']

        try:
            with open(log_file) as f:
                content = f.read()

            # Extract final loss and throughput from last step line
            lines = [l for l in content.split('\n') if 'step' in l and 'loss' in l]
            if lines:
                last_line = lines[-1]
                # Parse: step   XXX | loss X.XXXX | lr X.XXe-XX | grad X.XX | tok/s XXXX
                parts = last_line.split('|')
                loss_part = [p for p in parts if 'loss' in p]
                tok_part = [p for p in parts if 'tok/s' in p]

                loss = float(loss_part[0].split()[-1]) if loss_part else -1
                tok_s = float(tok_part[0].split()[-1]) if tok_part else -1

                print(f"{config_id:>3} {desc:<30} {loss:>12.4f} {tok_s:>10.0f}")
            else:
                print(f"{config_id:>3} {desc:<30} {'N/A':>12} {'N/A':>10}")
        except Exception as e:
            print(f"{config_id:>3} {desc:<30} {'ERROR':>12} {'ERROR':>10}")

    print("-" * 70)


if __name__ == "__main__":
    main()
