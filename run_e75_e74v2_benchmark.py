#!/usr/bin/env python3
"""
Benchmark script for E75 Gated Delta and E74v2 ablation variants.

E75: Delta rule + forget gate (combines E74's delta rule with E61's active forgetting)
E74v2: Different update types for full matrix delta rule:
  - DELTA: Standard delta rule
  - RESIDUAL: ResNet-style residual updates
  - NTM: Neural Turing Machine style erase/write
  - RETRIEVED_GATE: Gate delta by retrieval quality
  - EMA: E61-style exponential moving average

Config: ~100M params, 10 minutes training, batch=32, chunk=512, lr=3e-4, bf16
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path

# Target: 100M params with expansion=2.0
TARGET_PARAMS = 100_000_000
EXPANSION = 2.0
TRAIN_MINUTES = 10
TARGET_DEPTH = 20
N_STATE = 64  # For E75 and E74 full matrix variants
OUTPUT_DIR = Path("benchmark_results/e75_e74v2_10min")
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
    """Round x to nearest multiple of 128."""
    return ((x + 63) // 128) * 128


def count_e75_params(dim, depth, vocab_size=256, expansion=2.0, n_state=64):
    """Count E75 (Gated Delta Matrix) parameters.

    E75: S = tanh(beta*S + outer(delta, k)) with beta = sigmoid(W_beta @ x + b_beta)
    Per layer:
    - in_proj: [d_inner, dim]
    - W_k, W_v, W_q: [n_state, d_inner]
    - W_beta: [n_state, d_inner], b_beta: [n_state]
    - out_proj: [n_state, dim]
    """
    d_inner = int(dim * expansion)

    per_layer = (
        dim * d_inner +          # in_proj
        n_state * d_inner +      # W_k
        n_state * d_inner +      # W_v
        n_state * d_inner +      # W_q
        n_state * d_inner +      # W_beta
        n_state +                # b_beta
        n_state * dim +          # out_proj
        2 * dim                  # LayerNorm
    )

    total = (
        vocab_size * dim +       # embedding
        per_layer * depth +      # layers
        2 * dim                  # final norm
    )
    return total


def count_e74_ablation_params(dim, depth, vocab_size=256, expansion=2.0, n_state=64,
                              update_type='delta', gate_type='output'):
    """Count E74 Ablation (Full Matrix) parameters.

    Base parameters for all update types:
    - in_proj: [d_inner, dim]
    - W_k, W_v, W_q: [n_state, d_inner]
    - out_proj: [n_state, dim]

    Additional parameters by update_type:
    - delta: none
    - residual: residual_scale [n_state]
    - ntm: W_erase, b_erase, W_write, b_write
    - retrieved_gate: W_gate, b_gate
    - ema: W_alpha, b_alpha

    Additional parameters by gate_type:
    - output: none (self-gating uses no extra params)
    - input: W_z_gate, b_z_gate
    """
    d_inner = int(dim * expansion)

    # Base params
    per_layer = (
        dim * d_inner +          # in_proj
        n_state * d_inner +      # W_k
        n_state * d_inner +      # W_v
        n_state * d_inner +      # W_q
        n_state * dim +          # out_proj
        2 * dim                  # LayerNorm
    )

    # Update type extras
    if update_type == 'residual':
        per_layer += n_state     # residual_scale
    elif update_type == 'ntm':
        per_layer += (
            n_state * d_inner +  # W_erase
            n_state +            # b_erase
            n_state * d_inner +  # W_write
            n_state              # b_write
        )
    elif update_type == 'retrieved_gate':
        per_layer += (
            n_state * d_inner +  # W_gate
            n_state              # b_gate
        )
    elif update_type == 'ema':
        per_layer += (
            n_state * d_inner +  # W_alpha
            n_state              # b_alpha
        )

    # Gate type extras
    if gate_type == 'input':
        per_layer += (
            n_state * d_inner +  # W_z_gate
            n_state              # b_z_gate
        )

    total = (
        vocab_size * dim +       # embedding
        per_layer * depth +      # layers
        2 * dim                  # final norm
    )
    return total


def find_config_at_depth(count_fn, target_params, target_depth=20, **kwargs):
    """Find dim to hit target_params at a specific depth."""
    depth = target_depth
    best_dim = 128
    best_params = count_fn(128, depth, **kwargs)
    best_diff = abs(best_params - target_params)

    for dim in range(128, 2049, 128):
        params = count_fn(dim, depth, **kwargs)
        diff = abs(params - target_params)
        if diff < best_diff:
            best_diff = diff
            best_dim = dim
            best_params = params

    return (best_dim, depth, best_params)


def main():
    print("=" * 60)
    print(f"E75/E74v2 Benchmark: 100M params, 10 minutes, depth={TARGET_DEPTH}")
    print("=" * 60)

    configs = {}

    # E75 Gated Delta - combines delta rule with forget gate
    dim, depth, params = find_config_at_depth(
        count_e75_params, TARGET_PARAMS, TARGET_DEPTH,
        expansion=EXPANSION, n_state=N_STATE
    )
    configs['e75'] = {
        'dim': dim, 'depth': depth, 'params': params,
        'expansion': EXPANSION, 'level': 75, 'n_state': N_STATE
    }
    print(f"E75 (Gated Delta):     dim={dim}, depth={depth}, params={params:,}")

    # E75 with different n_state values
    for ns in [32, 48, 96]:
        dim, depth, params = find_config_at_depth(
            count_e75_params, TARGET_PARAMS, TARGET_DEPTH,
            expansion=EXPANSION, n_state=ns
        )
        configs[f'e75n{ns}'] = {
            'dim': dim, 'depth': depth, 'params': params,
            'expansion': EXPANSION, 'level': f'75n{ns}', 'n_state': ns
        }
        print(f"E75 n={ns}:             dim={dim}, depth={depth}, params={params:,}")

    # E74 ablations with different update types
    update_types = [
        ('delta', 'DELTA'),
        ('residual', 'RESIDUAL'),
        ('ntm', 'NTM'),
        ('retrieved_gate', 'RETRIEVED_GATE'),
        ('ema', 'EMA'),
    ]

    for update_type, update_name in update_types:
        dim, depth, params = find_config_at_depth(
            count_e74_ablation_params, TARGET_PARAMS, TARGET_DEPTH,
            expansion=EXPANSION, n_state=N_STATE, update_type=update_type
        )
        key = f'e74-{update_type}'
        configs[key] = {
            'dim': dim, 'depth': depth, 'params': params,
            'expansion': EXPANSION, 'n_state': N_STATE,
            'state_type': 'full', 'proj_type': 'no_z',
            'nonlin_type': 'tanh', 'gate_type': 'output',
            'update_type': update_type
        }
        print(f"E74 {update_name:15s}:   dim={dim}, depth={depth}, params={params:,}")

    # E74 with INPUT gate (E1-style output gating) for main update types
    for update_type in ['delta', 'ema', 'retrieved_gate']:
        dim, depth, params = find_config_at_depth(
            count_e74_ablation_params, TARGET_PARAMS, TARGET_DEPTH,
            expansion=EXPANSION, n_state=N_STATE, gate_type='input', update_type=update_type
        )
        key = f'e74-{update_type}-input'
        configs[key] = {
            'dim': dim, 'depth': depth, 'params': params,
            'expansion': EXPANSION, 'n_state': N_STATE,
            'state_type': 'full', 'proj_type': 'no_z',
            'nonlin_type': 'tanh', 'gate_type': 'input',
            'update_type': update_type
        }
        print(f"E74 {update_type:12s} INPUT: dim={dim}, depth={depth}, params={params:,}")

    # E74 with no_tanh (linear state updates) for main update types
    for update_type in ['delta', 'ema', 'residual']:
        dim, depth, params = find_config_at_depth(
            count_e74_ablation_params, TARGET_PARAMS, TARGET_DEPTH,
            expansion=EXPANSION, n_state=N_STATE, update_type=update_type
        )
        key = f'e74-{update_type}-linear'
        configs[key] = {
            'dim': dim, 'depth': depth, 'params': params,
            'expansion': EXPANSION, 'n_state': N_STATE,
            'state_type': 'full', 'proj_type': 'no_z',
            'nonlin_type': 'linear', 'gate_type': 'output',  # no_tanh = linear
            'update_type': update_type
        }
        print(f"E74 {update_type:12s} linear:dim={dim}, depth={depth}, params={params:,}")

    # Save configs
    with open(OUTPUT_DIR / "configs.json", "w") as f:
        json.dump(configs, f, indent=2)

    # Get available GPUs
    available_gpus = get_available_gpus()
    print(f"\nAvailable GPUs: {available_gpus}")

    # Models to benchmark
    model_order = list(configs.keys())

    print("\n" + "=" * 60)
    print(f"Starting {TRAIN_MINUTES}-minute benchmarks")
    print(f"Models: {len(model_order)}, GPUs available: {len(available_gpus)}")
    print("=" * 60)

    def launch_job(model_name, gpu_id, config):
        """Launch a training job on a specific GPU."""
        level = config.get('level', model_name)

        # Check if this is an E74 ablation (has update_type)
        if 'update_type' in config:
            # Use train_e74_ablation.py
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
            # Use standard train.py with LadderLM
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
            shell=True,
            stdout=open(log_file, "w"),
            stderr=subprocess.STDOUT
        )
        return p

    # Track active jobs
    active_jobs = {}
    pending_models = list(model_order)
    completed_models = []

    # Initial batch
    for gpu_id in available_gpus:
        if not pending_models:
            break
        model_name = pending_models.pop(0)
        if model_name in configs:
            p = launch_job(model_name, gpu_id, configs[model_name])
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
            print(f"{model_name} completed with exit code {ret} (GPU {gpu_id} now free)")

            if pending_models:
                next_model = pending_models.pop(0)
                if next_model in configs:
                    p = launch_job(next_model, gpu_id, configs[next_model])
                    active_jobs[next_model] = (p, gpu_id)
                    print(f"Launched {next_model} on GPU {gpu_id}")

        if completed:
            print(f"Progress: {len(completed_models)}/{len(model_order)} complete, "
                  f"{len(active_jobs)} running, {len(pending_models)} pending")

    print("\n" + "=" * 60)
    print("All benchmarks complete!")
    print(f"Completed models: {completed_models}")
    print("=" * 60)


if __name__ == "__main__":
    main()
