#!/usr/bin/env python3
"""
Quick 100M parameter benchmark: E77 vs E42 vs E68
10-minute training comparison
"""

import os
import subprocess
import time
from pathlib import Path

OUTPUT_DIR = Path("benchmark_results/e77_100m")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_MINUTES = 10

# Configs for ~100M params at depth=20
# From run_100m_benchmark.py calculations
configs = {
    'e42': {
        'dim': 768,  # LinearTiedElman - simpler so needs wider dim
        'depth': 20,
        'expansion': 2.0,
        'level': 42,
    },
    'e68': {
        'dim': 640,  # Self-gating h-dependence
        'depth': 20,
        'expansion': 2.0,
        'level': 68,
    },
    'e77': {
        'dim': 2048,  # Linear Matrix State - expansion=1.0, n_state=64
        'depth': 20,
        'expansion': 1.0,
        'n_state': 64,
        'level': 77,
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
        f"--output {OUTPUT_DIR}/{model_name}",
        f"--bf16",
    ]

    if 'expansion' in config:
        cmd_parts.append(f"--expansion {config['expansion']}")

    if 'n_state' in config:
        cmd_parts.append(f"--n_state {config['n_state']}")

    cmd = " ".join(cmd_parts)

    p = subprocess.Popen(
        f"CUDA_VISIBLE_DEVICES={gpu_id} PYTHONUNBUFFERED=1 {cmd}",
        shell=True,
        stdout=open(OUTPUT_DIR / f"{model_name}.log", "w"),
        stderr=subprocess.STDOUT
    )
    return p


def main():
    print("=" * 60)
    print(f"E77 vs E42 vs E68: 100M param benchmark ({TRAIN_MINUTES} min)")
    print("=" * 60)

    for name, cfg in configs.items():
        print(f"{name}: dim={cfg['dim']}, depth={cfg['depth']}, expansion={cfg.get('expansion', 1.0)}")

    gpus = get_available_gpus()[:3]  # Use up to 3 GPUs
    print(f"\nUsing GPUs: {gpus}")

    # Launch all 3 in parallel
    jobs = {}
    for i, (name, config) in enumerate(configs.items()):
        gpu = gpus[i % len(gpus)]
        p = launch_job(name, gpu, config)
        jobs[name] = (p, gpu)
        print(f"Launched {name} on GPU {gpu}")

    # Wait for completion
    print("\nWaiting for completion...")
    while jobs:
        time.sleep(10)
        completed = []
        for name, (p, gpu) in jobs.items():
            ret = p.poll()
            if ret is not None:
                completed.append(name)
                print(f"{name} completed (exit {ret})")

        for name in completed:
            del jobs[name]

    print("\n" + "=" * 60)
    print("Benchmark complete! Results:")
    print("=" * 60)

    # Parse results from logs
    for name in configs:
        log_file = OUTPUT_DIR / f"{name}.log"
        if log_file.exists():
            with open(log_file) as f:
                lines = f.readlines()

            # Find last loss report
            last_loss = None
            last_step = None
            throughput = None
            for line in lines:
                if "loss:" in line.lower():
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if "loss" in p.lower() and i + 1 < len(parts):
                            try:
                                last_loss = float(parts[i + 1].strip(','))
                            except:
                                pass
                        if "step" in p.lower() and i + 1 < len(parts):
                            try:
                                last_step = int(parts[i + 1].strip(','))
                            except:
                                pass
                        if "tok/s" in p.lower():
                            try:
                                throughput = parts[i - 1]
                            except:
                                pass

            print(f"{name}: step={last_step}, loss={last_loss}, throughput={throughput}")


if __name__ == "__main__":
    main()
