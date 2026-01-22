#!/usr/bin/env python3
"""
Generate benchmark job files for sched.py

Usage:
    # Generate 100M benchmark jobs (10 min training)
    python gen_jobs.py e75_100m --minutes 10 > jobs.txt
    python sched.py jobs.txt

    # Generate quick test (1 min)
    python gen_jobs.py e75_100m --minutes 1 > jobs_quick.txt

    # List available benchmarks
    python gen_jobs.py --list
"""

import argparse
import sys
from datetime import datetime

# Standard common arguments
COMMON = (
    "--data data/pile.txt "
    "--batch_size 32 "
    "--chunk_size 512 "
    "--lr 3e-4 "
    "--warmup_steps 100 "
    "--seed 42 "
    "--bf16"
)

# Benchmark definitions: name -> list of (model_name, model_args)
BENCHMARKS = {
    'e75_100m': {
        'desc': 'E75 Multi-Head 100M param comparison',
        'models': [
            ('mamba2', '--level mamba2 --dim 896 --depth 20'),
            ('fla-gdn', '--level fla-gdn --dim 768 --depth 20 --expansion 2.0'),
            ('E75h4n16', '--level E75h4n16 --dim 2048 --depth 20 --n_state 16 --expansion 1.0'),
            ('E75h4n24', '--level E75h4n24 --dim 2048 --depth 20 --n_state 24 --expansion 1.0'),
            ('E75h4n32', '--level E75h4n32 --dim 1920 --depth 20 --n_state 32 --expansion 1.0'),
            ('E75h8n16', '--level E75h8n16 --dim 1920 --depth 20 --n_state 16 --expansion 1.0'),
            ('E75h8n24', '--level E75h8n24 --dim 1792 --depth 20 --n_state 24 --expansion 1.0'),
        ],
    },
    'e75_extended': {
        'desc': 'Extended E75 scan with more head/state combos',
        'models': [
            ('mamba2', '--level mamba2 --dim 896 --depth 20'),
            ('fla-gdn', '--level fla-gdn --dim 768 --depth 20 --expansion 2.0'),
            ('E75h4n16', '--level E75h4n16 --dim 2048 --depth 20 --n_state 16 --expansion 1.0'),
            ('E75h4n24', '--level E75h4n24 --dim 2048 --depth 20 --n_state 24 --expansion 1.0'),
            ('E75h4n32', '--level E75h4n32 --dim 1920 --depth 20 --n_state 32 --expansion 1.0'),
            ('E75h4n48', '--level E75h4n48 --dim 1792 --depth 20 --n_state 48 --expansion 1.0'),
            ('E75h6n24', '--level E75h6n24 --dim 1920 --depth 20 --n_state 24 --expansion 1.0'),
            ('E75h6n32', '--level E75h6n32 --dim 1792 --depth 20 --n_state 32 --expansion 1.0'),
            ('E75h8n16', '--level E75h8n16 --dim 1920 --depth 20 --n_state 16 --expansion 1.0'),
            ('E75h8n24', '--level E75h8n24 --dim 1792 --depth 20 --n_state 24 --expansion 1.0'),
        ],
    },
    'baselines': {
        'desc': 'Baseline models (mamba2, fla-gdn, e1, llama)',
        'models': [
            ('mamba2', '--level mamba2 --dim 896 --depth 20'),
            ('fla-gdn', '--level fla-gdn --dim 768 --depth 20 --expansion 2.0'),
            ('e1', '--level 1 --dim 640 --depth 20 --expansion 2.0'),
            ('llama', '--level llama --dim 640 --depth 20 --expansion 2.0'),
        ],
    },
    'quick_test': {
        'desc': 'Quick 2-model test',
        'models': [
            ('mamba2', '--level mamba2 --dim 896 --depth 20'),
            ('E75h8n24', '--level E75h8n24 --dim 1792 --depth 20 --n_state 24 --expansion 1.0'),
        ],
    },
    # E88 FLA Hybrid benchmarks
    # Best E88 config: expansion=1.0, use_conv=False, use_gate=False, use_output_norm=False
    # E88f_* = fully ablated (no conv, no gate, no norm) - best config
    # E88c_* = partial ablation (no conv, no norm, but WITH gate)
    'e88_nstate': {
        'desc': 'E88 n_state comparison (newly added templates)',
        'models': [
            ('fla-gdn', '--level fla-gdn --dim 768 --depth 20'),
            ('mamba2', '--level mamba2 --dim 896 --depth 20'),
            # n_state=16: 32 heads (newly added template) - full ablation
            ('E88_h32n16', '--level E88f_h32n16 --dim 1536 --depth 20'),
            # n_state=24: 24 heads (newly added template) - full ablation
            ('E88_h24n24', '--level E88f_h24n24 --dim 1536 --depth 20'),
            # n_state=32: 8 heads (baseline, known working) - full ablation
            ('E88_h8n32', '--level E88e_h8_75m --dim 1920 --depth 20'),
            # n_state=32: 16 heads - partial ablation (with gate)
            ('E88_h16n32_gate', '--level E88b_nonorm --dim 1536 --depth 20'),
        ],
    },
    'e88_heads': {
        'desc': 'E88 head count scaling at n_state=32 (full ablation)',
        'models': [
            ('fla-gdn', '--level fla-gdn --dim 768 --depth 20'),
            ('E88_h4n32', '--level E88f_h4 --dim 2176 --depth 20'),
            ('E88_h6n32', '--level E88f_h6 --dim 2048 --depth 20'),
            ('E88_h8n32', '--level E88e_h8_75m --dim 1920 --depth 20'),
            ('E88_h12n32', '--level E88d_h12 --dim 1664 --depth 20'),
            ('E88_h20n32', '--level E88d_h20 --dim 1536 --depth 20'),
        ],
    },
    'e88_vs_baselines': {
        'desc': 'E88 best configs vs baselines at ~100M params',
        'models': [
            ('mamba2', '--level mamba2 --dim 896 --depth 20'),
            ('fla-gdn', '--level fla-gdn --dim 768 --depth 20'),
            # E88 with n_state=32 (full ablation)
            ('E88_h8n32', '--level E88e_h8_75m --dim 1920 --depth 20'),
            ('E88_h12n32', '--level E88d_h12 --dim 1664 --depth 20'),
            # E88 with smaller n_state (newly added templates)
            ('E88_h32n16', '--level E88f_h32n16 --dim 1536 --depth 20'),
        ],
    },
}


def generate_jobs(benchmark: str, minutes: float, output_dir: str = None):
    """Generate job commands."""
    if benchmark not in BENCHMARKS:
        print(f"Unknown benchmark: {benchmark}", file=sys.stderr)
        print(f"Available: {list(BENCHMARKS.keys())}", file=sys.stderr)
        sys.exit(1)

    bench = BENCHMARKS[benchmark]

    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f'benchmark_results/{benchmark}_{timestamp}'

    print(f"# Benchmark: {bench['desc']}", file=sys.stderr)
    print(f"# Models: {len(bench['models'])}", file=sys.stderr)
    print(f"# Training: {minutes} minutes", file=sys.stderr)
    print(f"# Output: {output_dir}", file=sys.stderr)

    for name, model_args in bench['models']:
        cmd = (
            f"python train.py {model_args} "
            f"{COMMON} "
            f"--train_minutes {minutes} "
            f"--output {output_dir}/{name}"
        )
        print(cmd)


def main():
    parser = argparse.ArgumentParser(description='Generate benchmark jobs')
    parser.add_argument('benchmark', nargs='?', help='Benchmark name')
    parser.add_argument('--minutes', type=float, default=10, help='Training time per model')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--list', action='store_true', help='List available benchmarks')
    args = parser.parse_args()

    if args.list or not args.benchmark:
        print("Available benchmarks:")
        for name, bench in BENCHMARKS.items():
            print(f"  {name:<16} - {bench['desc']} ({len(bench['models'])} models)")
        return

    generate_jobs(args.benchmark, args.minutes, args.output)


if __name__ == '__main__':
    main()
