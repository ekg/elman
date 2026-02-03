#!/usr/bin/env python3
"""
E76 vs Mamba2/FLA-GDN Benchmark
10-minute training, ~100M params, comparing E76 configurations.

Wave 1: Core comparison (8 models)
- mamba2, fla-gdn: Linear baselines
- e75: Original nonlinear delta
- e76 (4 configs): tanh/linear × log_gate/sigmoid

Wave 2: n_state sweep (if needed)
"""

import os
import subprocess
import time
from pathlib import Path

TRAIN_MINUTES = 10
OUTPUT_DIR = Path("benchmark_results/e76_benchmark")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Wave 1 configurations (8 models for 8 GPUs)
WAVE1_CONFIGS = {
    # Linear baselines
    'mamba2': {
        'level': 'mamba2',
        'dim': 896,
        'depth': 20,
    },
    'fla-gdn': {
        'level': 'fla-gdn',
        'dim': 768,
        'depth': 20,
        'expansion': 2.0,
    },
    # E75 original (nonlinear delta with sigmoid gate)
    'e75': {
        'level': 75,
        'dim': 1408,
        'depth': 20,
        'n_state': 64,
        'expansion': 2.0,
    },
    # E76 default: tanh + log_space_gate (nonlinear + Mamba2-style params)
    'e76-t-log': {
        'level': '76-t-log',
        'dim': 1408,
        'depth': 20,
        'n_state': 64,
        'expansion': 2.0,
    },
    # E76: tanh + sigmoid_gate (like E75 but with different init)
    'e76-t-sig': {
        'level': '76-t-sig',
        'dim': 1408,
        'depth': 20,
        'n_state': 64,
        'expansion': 2.0,
    },
    # E76: linear + log_space_gate (Mamba2-like but with matrix state)
    'e76-l-log': {
        'level': '76-l-log',
        'dim': 1408,
        'depth': 20,
        'n_state': 64,
        'expansion': 2.0,
    },
    # E76: linear + sigmoid_gate (fully linear comparison)
    'e76-l-sig': {
        'level': '76-l-sig',
        'dim': 1408,
        'depth': 20,
        'n_state': 64,
        'expansion': 2.0,
    },
    # E68: Best previous E-series
    'e68': {
        'level': 68,
        'dim': 640,
        'depth': 20,
        'expansion': 2.0,
    },
}

# Wave 2: n_state sweep for E76 default config
WAVE2_CONFIGS = {
    'e76-n32': {
        'level': '76n32',
        'dim': 1408,
        'depth': 20,
        'n_state': 32,
        'expansion': 2.0,
    },
    'e76-n48': {
        'level': '76n48',
        'dim': 1408,
        'depth': 20,
        'n_state': 48,
        'expansion': 2.0,
    },
    'e76-n64': {
        'level': '76n64',
        'dim': 1408,
        'depth': 20,
        'n_state': 64,
        'expansion': 2.0,
    },
    'e76-n96': {
        'level': '76n96',
        'dim': 1408,
        'depth': 20,
        'n_state': 96,
        'expansion': 2.0,
    },
}


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


def launch_job(model_name, config, gpu_id, output_dir):
    """Launch a training job on a specific GPU."""
    level = config.get('level', model_name)

    cmd_parts = [
        f"python train.py",
        f"--data data/pile.txt",
        f"--level {level}",
        f"--dim {config['dim']}",
        f"--depth {config['depth']}",
        f"--batch_size 32",
        f"--chunk_size 512",
        f"--lr 3e-4",
        f"--warmup_steps 1000",
        f"--train_minutes {TRAIN_MINUTES}",
        f"--seed 42",
        f"--bf16",  # Required for E76 CUDA kernel
        f"--output {output_dir / model_name}",
    ]

    if 'expansion' in config:
        cmd_parts.append(f"--expansion {config['expansion']}")
    if 'n_state' in config:
        cmd_parts.append(f"--n_state {config['n_state']}")
    if 'expand' in config:
        cmd_parts.append(f"--expand {config['expand']}")

    cmd = " ".join(cmd_parts)
    log_file = output_dir / f"{model_name}.log"

    print(f"  GPU {gpu_id}: {model_name} (level={level}, dim={config['dim']})")

    p = subprocess.Popen(
        f"CUDA_VISIBLE_DEVICES={gpu_id} PYTHONUNBUFFERED=1 {cmd}",
        shell=True,
        stdout=open(log_file, "w"),
        stderr=subprocess.STDOUT
    )
    return p


def run_wave(wave_name, configs, output_dir):
    """Run a wave of benchmarks."""
    available_gpus = get_available_gpus()
    print(f"\n{'='*60}")
    print(f"Running {wave_name}: {len(configs)} models on {len(available_gpus)} GPUs")
    print(f"{'='*60}")

    if len(available_gpus) < len(configs):
        print(f"Warning: Only {len(available_gpus)} GPUs available for {len(configs)} models")

    # Launch all jobs
    jobs = {}
    for i, (model_name, config) in enumerate(configs.items()):
        gpu_id = available_gpus[i % len(available_gpus)]
        p = launch_job(model_name, config, gpu_id, output_dir)
        jobs[model_name] = (p, gpu_id)

    print(f"\nAll {len(jobs)} jobs launched. Waiting for completion...")

    # Wait for all jobs
    start_time = time.time()
    while jobs:
        for model_name in list(jobs.keys()):
            p, gpu_id = jobs[model_name]
            ret = p.poll()
            if ret is not None:
                elapsed = time.time() - start_time
                status = "completed" if ret == 0 else f"failed (code {ret})"
                print(f"  {model_name}: {status} after {elapsed/60:.1f}min")
                del jobs[model_name]
        if jobs:
            time.sleep(10)

    print(f"\n{wave_name} complete!")


def extract_results(output_dir):
    """Extract final loss from logs."""
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")

    results = []
    for log_file in output_dir.glob("*.log"):
        model_name = log_file.stem
        try:
            with open(log_file) as f:
                content = f.read()

            # Find last loss value
            losses = []
            for line in content.split('\n'):
                if 'loss=' in line or 'Loss:' in line:
                    import re
                    match = re.search(r'loss[=:]\s*([0-9.]+)', line, re.IGNORECASE)
                    if match:
                        losses.append(float(match.group(1)))

            if losses:
                # Last 100 average
                final_loss = sum(losses[-100:]) / min(100, len(losses))
                results.append((model_name, final_loss, len(losses)))
        except Exception as e:
            print(f"  {model_name}: Error reading log - {e}")

    # Sort by loss
    results.sort(key=lambda x: x[1])

    print(f"\n{'Model':<15} {'Loss':>8} {'Steps':>8}")
    print("-" * 35)
    for model, loss, steps in results:
        print(f"{model:<15} {loss:>8.4f} {steps:>8}")

    return results


if __name__ == "__main__":
    import sys

    wave = sys.argv[1] if len(sys.argv) > 1 else "1"

    if wave == "1":
        run_wave("Wave 1: E76 vs Baselines", WAVE1_CONFIGS, OUTPUT_DIR)
    elif wave == "2":
        run_wave("Wave 2: E76 n_state sweep", WAVE2_CONFIGS, OUTPUT_DIR)
    elif wave == "results":
        extract_results(OUTPUT_DIR)
    else:
        print(f"Usage: {sys.argv[0]} [1|2|results]")
        sys.exit(1)

    # Always show results at the end
    if wave in ["1", "2"]:
        time.sleep(5)
        extract_results(OUTPUT_DIR)
