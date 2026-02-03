#!/usr/bin/env python3
"""
8-Model Comparison Benchmark
FLA-GDN, Mamba2, E42, E61, E68, E75 (n32, n48, n64)

Config: ~100M params, 10 minutes training, batch=32, chunk=512, lr=3e-4, bf16
"""

import os
import subprocess
import time
from pathlib import Path

TRAIN_MINUTES = 10
OUTPUT_DIR = Path("benchmark_results/8model_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model configurations - all ~100M params, depth=20
CONFIGS = {
    'fla-gdn': {
        'level': 'fla-gdn',
        'dim': 768,
        'depth': 20,
        'expansion': 2.0,
        'params': 95_020_016,
    },
    'mamba2': {
        'level': 'mamba2',
        'dim': 896,
        'depth': 20,
        'params': 101_900_688,
    },
    'e42': {
        'level': 42,
        'dim': 768,
        'depth': 20,
        'expansion': 2.0,
        'params': 94_615_296,
    },
    'e61': {
        'level': 61,
        'dim': 640,
        'depth': 20,
        'expansion': 2.0,
        'params': 98_532_480,
    },
    'e68': {
        'level': 68,
        'dim': 640,
        'depth': 20,
        'expansion': 2.0,
        'params': 98_583_680,
    },
    'e75n32': {
        'level': '75n32',
        'dim': 1536,
        'depth': 20,
        'expansion': 2.0,
        'n_state': 32,
        'params': 103_645_312,
    },
    'e75n48': {
        'level': '75n48',
        'dim': 1536,
        'depth': 20,
        'expansion': 2.0,
        'n_state': 48,
        'params': 108_069_312,
    },
    'e75n64': {
        'level': 75,
        'dim': 1408,
        'depth': 20,
        'expansion': 2.0,
        'n_state': 64,
        'params': 95_910_016,
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


def launch_job(model_name, gpu_id, config):
    """Launch a training job on a specific GPU."""
    cmd_parts = [
        f"python train.py",
        f"--data data/pile.txt",
        f"--level {config['level']}",
        f"--dim {config['dim']}",
        f"--depth {config['depth']}",
        f"--batch_size 32",
        f"--chunk_size 512",
        f"--steps 999999",
        f"--train_minutes {TRAIN_MINUTES}",
        f"--lr 3e-4",
        f"--log_every 10",
        f"--seed 42",
        f"--output {OUTPUT_DIR}/{model_name}",
        f"--bf16",
    ]

    if 'expansion' in config:
        cmd_parts.append(f"--expansion {config['expansion']}")
    if 'n_state' in config:
        cmd_parts.append(f"--n_state {config['n_state']}")

    cmd = " ".join(cmd_parts)
    log_file = OUTPUT_DIR / f"{model_name}.log"

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    env['PYTHONUNBUFFERED'] = '1'

    p = subprocess.Popen(
        cmd,
        shell=True,
        stdout=open(log_file, "w"),
        stderr=subprocess.STDOUT,
        env=env
    )
    return p


def main():
    print("=" * 70)
    print("8-Model Comparison Benchmark")
    print(f"Config: ~100M params, {TRAIN_MINUTES} min, batch=32, chunk=512, seed=42")
    print("=" * 70)

    # Print config table
    print("\nModel Configurations:")
    print("-" * 70)
    print(f"{'Model':<12} {'Params':>12} {'dim':>6} {'depth':>6} {'expansion':>10} {'n_state':>8}")
    print("-" * 70)
    for name, cfg in CONFIGS.items():
        exp = cfg.get('expansion', '-')
        ns = cfg.get('n_state', '-')
        print(f"{name:<12} {cfg['params']:>12,} {cfg['dim']:>6} {cfg['depth']:>6} {exp:>10} {ns:>8}")
    print("-" * 70)

    # Get GPUs
    available_gpus = get_available_gpus()
    print(f"\nAvailable GPUs: {available_gpus[:8]}")

    if len(available_gpus) < 8:
        print(f"WARNING: Only {len(available_gpus)} GPUs available, need 8 for full parallel run")

    # Launch all jobs
    print(f"\nLaunching {len(CONFIGS)} jobs...")
    active_jobs = {}

    for i, (model_name, config) in enumerate(CONFIGS.items()):
        gpu_id = available_gpus[i % len(available_gpus)]
        p = launch_job(model_name, gpu_id, config)
        active_jobs[model_name] = (p, gpu_id)
        print(f"  {model_name} -> GPU {gpu_id} (PID {p.pid})")

    # Wait for completion
    print(f"\nWaiting for {TRAIN_MINUTES} minutes...")
    completed = []

    while active_jobs:
        time.sleep(10)

        done = []
        for model_name, (p, gpu_id) in active_jobs.items():
            ret = p.poll()
            if ret is not None:
                done.append((model_name, gpu_id, ret))

        for model_name, gpu_id, ret in done:
            del active_jobs[model_name]
            completed.append(model_name)
            status = "OK" if ret == 0 else f"ERROR({ret})"
            print(f"  {model_name} completed: {status}")

    print("\n" + "=" * 70)
    print("All jobs complete! Extracting results...")
    print("=" * 70)

    # Extract results
    results = []
    for model_name in CONFIGS.keys():
        log_file = OUTPUT_DIR / f"{model_name}.log"
        if log_file.exists():
            with open(log_file) as f:
                lines = f.readlines()

            # Find last training line
            last_step = None
            for line in reversed(lines):
                if line.startswith('step'):
                    parts = line.split('|')
                    try:
                        step = int(parts[0].split()[1])
                        loss = float(parts[1].split()[1])
                        toks = int(parts[4].split()[1])
                        last_step = {'step': step, 'loss': loss, 'toks': toks}
                        break
                    except:
                        pass

            if last_step:
                results.append({
                    'model': model_name,
                    'params': CONFIGS[model_name]['params'],
                    **last_step
                })

    # Print results
    print("\nFinal Results (sorted by loss):")
    print("-" * 70)
    print(f"{'Model':<12} {'Params':>12} {'Steps':>8} {'Loss':>8} {'tok/s':>10}")
    print("-" * 70)

    for r in sorted(results, key=lambda x: x['loss']):
        print(f"{r['model']:<12} {r['params']:>12,} {r['step']:>8} {r['loss']:>8.4f} {r['toks']:>10,}")

    print("-" * 70)
    print(f"\nLogs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
