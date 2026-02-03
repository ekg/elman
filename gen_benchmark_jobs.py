#!/usr/bin/env python3
"""
Generate benchmark job commands for gpu_queue.py

Usage:
    # Generate jobs file for E75 100M benchmark
    python gen_benchmark_jobs.py e75_100m > jobs.txt
    python gpu_queue.py add-file jobs.txt
    python gpu_queue.py run

    # Or pipe directly
    python gen_benchmark_jobs.py e75_100m | while read cmd; do python gpu_queue.py add "$cmd"; done
    python gpu_queue.py run
"""

import sys
from datetime import datetime

# Standard benchmark configurations
BENCHMARKS = {
    'e75_100m': {
        'description': 'E75 Multi-Head 100M param benchmark',
        'common': '--data data/pile.txt --depth 20 --batch_size 32 --chunk_size 512 '
                  '--lr 3e-4 --warmup_steps 100 --seed 42 --expansion 1.0 --steps 3000 --bf16',
        'models': [
            ('mamba2', '--level mamba2 --dim 896'),
            ('fla-gdn', '--level fla-gdn --dim 768'),
            ('E75h4n16', '--level E75h4n16 --dim 2048 --n_state 16'),
            ('E75h4n24', '--level E75h4n24 --dim 2048 --n_state 24'),
            ('E75h4n32', '--level E75h4n32 --dim 1920 --n_state 32'),
            ('E75h8n16', '--level E75h8n16 --dim 1920 --n_state 16'),
            ('E75h8n24', '--level E75h8n24 --dim 1792 --n_state 24'),
        ],
    },
    'e75_quick': {
        'description': 'Quick E75 test (500 steps)',
        'common': '--data data/pile.txt --depth 20 --batch_size 32 --chunk_size 512 '
                  '--lr 3e-4 --warmup_steps 100 --seed 42 --expansion 1.0 --steps 500 --bf16',
        'models': [
            ('mamba2', '--level mamba2 --dim 896'),
            ('E75h8n24', '--level E75h8n24 --dim 1792 --n_state 24'),
        ],
    },
    'baselines': {
        'description': 'Baseline models comparison',
        'common': '--data data/pile.txt --depth 20 --batch_size 32 --chunk_size 512 '
                  '--lr 3e-4 --warmup_steps 100 --seed 42 --steps 3000 --bf16',
        'models': [
            ('mamba2', '--level mamba2 --dim 896'),
            ('fla-gdn', '--level fla-gdn --dim 768 --expansion 2.0'),
            ('e1', '--level 1 --dim 640 --expansion 2.0'),
            ('llama', '--level llama --dim 640 --expansion 2.0'),
        ],
    },
}


def generate_jobs(benchmark_name: str, output_dir: str = None):
    """Generate job commands for a benchmark."""
    if benchmark_name not in BENCHMARKS:
        print(f"Unknown benchmark: {benchmark_name}", file=sys.stderr)
        print(f"Available: {list(BENCHMARKS.keys())}", file=sys.stderr)
        return

    bench = BENCHMARKS[benchmark_name]

    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f'benchmark_results/{benchmark_name}_{timestamp}'

    print(f"# {bench['description']}", file=sys.stderr)
    print(f"# Output: {output_dir}", file=sys.stderr)
    print(f"# Models: {len(bench['models'])}", file=sys.stderr)

    for name, model_args in bench['models']:
        cmd = f"python train.py {model_args} {bench['common']} --output {output_dir}/{name}"
        print(cmd)


def main():
    if len(sys.argv) < 2:
        print("Usage: python gen_benchmark_jobs.py <benchmark_name> [output_dir]")
        print(f"Available benchmarks: {list(BENCHMARKS.keys())}")
        return

    benchmark_name = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    generate_jobs(benchmark_name, output_dir)


if __name__ == '__main__':
    main()
