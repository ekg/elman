#!/usr/bin/env python3
"""Wave 2: E74v2 benchmarks with fixed CUDA kernels."""

import os
import sys
import subprocess
import time
import json
from pathlib import Path

TARGET_PARAMS = 100_000_000
EXPANSION = 2.0
TRAIN_MINUTES = 10
TARGET_DEPTH = 20
N_STATE = 64
OUTPUT_DIR = Path("benchmark_results/e75_e74v2_wave2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_available_gpus():
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


def count_e74_ablation_params(dim, depth, vocab_size=256, expansion=2.0, n_state=64,
                              update_type='delta', gate_type='output'):
    d_inner = int(dim * expansion)
    per_layer = (
        dim * d_inner +          # in_proj
        n_state * d_inner +      # W_k
        n_state * d_inner +      # W_v
        n_state * d_inner +      # W_q
        n_state * dim +          # out_proj
        2 * dim                  # LayerNorm
    )
    if update_type == 'residual':
        per_layer += n_state     # residual_scale
    elif update_type == 'ntm':
        per_layer += (n_state * d_inner + n_state + n_state * d_inner + n_state)
    elif update_type == 'retrieved_gate':
        per_layer += (n_state * d_inner + n_state)
    elif update_type == 'ema':
        per_layer += (n_state * d_inner + n_state)
    if gate_type == 'input':
        per_layer += (n_state * d_inner + n_state)
    total = vocab_size * dim + per_layer * depth + 2 * dim
    return total


def count_e75_params(dim, depth, vocab_size=256, expansion=2.0, n_state=64):
    d_inner = int(dim * expansion)
    per_layer = (
        dim * d_inner + n_state * d_inner * 4 + n_state + n_state * dim + 2 * dim
    )
    total = vocab_size * dim + per_layer * depth + 2 * dim
    return total


def find_config_at_depth(count_fn, target_params, target_depth=20, **kwargs):
    best_dim, best_params, best_diff = 128, count_fn(128, target_depth, **kwargs), float('inf')
    for dim in range(128, 2049, 128):
        params = count_fn(dim, target_depth, **kwargs)
        diff = abs(params - target_params)
        if diff < best_diff:
            best_diff, best_dim, best_params = diff, dim, params
    return (best_dim, target_depth, best_params)


def main():
    print("=" * 60)
    print(f"E74v2 Wave 2 Benchmark: Fixed CUDA kernels")
    print("=" * 60)

    configs = {}

    # E75 variants
    for ns in [32, 48, 64, 96]:
        dim, depth, params = find_config_at_depth(
            count_e75_params, TARGET_PARAMS, TARGET_DEPTH, expansion=EXPANSION, n_state=ns
        )
        key = f'e75n{ns}' if ns != 64 else 'e75'
        configs[key] = {
            'dim': dim, 'depth': depth, 'params': params,
            'expansion': EXPANSION, 'level': key if ns == 64 else f'75n{ns}', 'n_state': ns
        }

    # E74 ablations with all update types + gate type output
    for update_type in ['delta', 'residual', 'ntm', 'retrieved_gate', 'ema']:
        dim, depth, params = find_config_at_depth(
            count_e74_ablation_params, TARGET_PARAMS, TARGET_DEPTH,
            expansion=EXPANSION, n_state=N_STATE, update_type=update_type
        )
        configs[f'e74-{update_type}'] = {
            'dim': dim, 'depth': depth, 'params': params,
            'expansion': EXPANSION, 'n_state': N_STATE,
            'state_type': 'full', 'proj_type': 'no_z',
            'nonlin_type': 'tanh', 'gate_type': 'output',
            'update_type': update_type
        }

    # E74 with input gate for main types
    for update_type in ['delta', 'ema']:
        dim, depth, params = find_config_at_depth(
            count_e74_ablation_params, TARGET_PARAMS, TARGET_DEPTH,
            expansion=EXPANSION, n_state=N_STATE, gate_type='input', update_type=update_type
        )
        configs[f'e74-{update_type}-input'] = {
            'dim': dim, 'depth': depth, 'params': params,
            'expansion': EXPANSION, 'n_state': N_STATE,
            'state_type': 'full', 'proj_type': 'no_z',
            'nonlin_type': 'tanh', 'gate_type': 'input',
            'update_type': update_type
        }

    # E74 with linear (no tanh) for delta
    dim, depth, params = find_config_at_depth(
        count_e74_ablation_params, TARGET_PARAMS, TARGET_DEPTH,
        expansion=EXPANSION, n_state=N_STATE, update_type='delta'
    )
    configs['e74-delta-linear'] = {
        'dim': dim, 'depth': depth, 'params': params,
        'expansion': EXPANSION, 'n_state': N_STATE,
        'state_type': 'full', 'proj_type': 'no_z',
        'nonlin_type': 'linear', 'gate_type': 'output',
        'update_type': 'delta'
    }

    # Save configs
    with open(OUTPUT_DIR / "configs.json", "w") as f:
        json.dump(configs, f, indent=2)

    print(f"\n{len(configs)} configurations:")
    for name, c in configs.items():
        print(f"  {name}: {c['params']:,} params")

    available_gpus = get_available_gpus()
    print(f"\nAvailable GPUs: {available_gpus}")

    model_order = list(configs.keys())
    print(f"\nStarting {TRAIN_MINUTES}-minute benchmarks with {len(model_order)} models")

    def launch_job(model_name, gpu_id, config):
        level = config.get('level', model_name)
        if 'update_type' in config:
            cmd_parts = [
                f"python train_e74_ablation.py",
                f"--data data/pile.txt",
                f"--dim {config['dim']}",
                f"--depth {config['depth']}",
                f"--n_state {config['n_state']}",
                f"--expansion {config['expansion']}",
                f"--state_type {config['state_type']}",
                f"--proj_type {config['proj_type']}",
                f"--nonlin_type {config['nonlin_type']}",
                f"--gate_type {config['gate_type']}",
                f"--update_type {config['update_type']}",
                f"--batch_size 32",
                f"--chunk_size 512",
                f"--steps 999999",
                f"--train_minutes {TRAIN_MINUTES}",
                f"--lr 3e-4",
                f"--log_every 10",
                f"--output {OUTPUT_DIR}/{model_name}",
                f"--bf16",
            ]
        else:
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
            if 'expansion' in config:
                cmd_parts.append(f"--expansion {config['expansion']}")
            if 'n_state' in config:
                cmd_parts.append(f"--n_state {config['n_state']}")

        cmd = " ".join(cmd_parts)
        log_file = OUTPUT_DIR / f"{model_name}.log"
        p = subprocess.Popen(
            f"CUDA_VISIBLE_DEVICES={gpu_id} PYTHONUNBUFFERED=1 {cmd}",
            shell=True, stdout=open(log_file, "w"), stderr=subprocess.STDOUT
        )
        return p

    active_jobs = {}
    pending_models = list(model_order)
    completed_models = []

    for gpu_id in available_gpus:
        if not pending_models:
            break
        model_name = pending_models.pop(0)
        p = launch_job(model_name, gpu_id, configs[model_name])
        active_jobs[model_name] = (p, gpu_id)
        print(f"Launched {model_name} on GPU {gpu_id}")

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
            print(f"{model_name} completed (exit {ret})")
            if pending_models:
                next_model = pending_models.pop(0)
                p = launch_job(next_model, gpu_id, configs[next_model])
                active_jobs[next_model] = (p, gpu_id)
                print(f"Launched {next_model} on GPU {gpu_id}")

        if completed:
            print(f"Progress: {len(completed_models)}/{len(model_order)} complete")

    print("\n" + "=" * 60)
    print("All benchmarks complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
