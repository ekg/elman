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

    # ============================================================
    # MEGA BENCHMARKS - Comprehensive param-matched comparisons
    # All configs verified with actual LadderLM instantiation
    # ============================================================

    'mega_100m': {
        'desc': 'Comprehensive 100M param benchmark (verified configs)',
        'models': [
            # Baselines (~100M)
            ('mamba2', '--level mamba2 --dim 896 --depth 20'),  # 101.4M (auto-config)
            ('fla-gdn', '--level fla-gdn --dim 1024 --depth 20'),  # 105.7M

            # E88 n_state=32 variants (fully ablated: no conv, no gate, no norm)
            ('E88_d12h32', '--level E88d_h12 --dim 3200 --depth 20'),  # 100.0M (12 heads)
            ('E88_d20h32', '--level E88d_h20 --dim 1920 --depth 20'),  # 99.6M (20 heads)

            # E88 smaller n_state (faster throughput)
            ('E88_h32n16', '--level E88f_h32n16 --dim 2432 --depth 20'),  # 101.8M
            ('E88_h24n24', '--level E88f_h24n24 --dim 2176 --depth 20'),  # 101.9M

            # E88 larger n_state (more state capacity)
            ('E88_h8n64', '--level E88c_h8n64 --dim 2432 --depth 20'),  # ~100M

            # E75 MultiHead for comparison
            ('E75h4n32', '--level E75h4n32 --dim 1920 --depth 20 --n_state 32 --expansion 1.0'),  # 98.8M
        ],
    },

    'mega_500m': {
        'desc': 'Comprehensive 500M param benchmark (verified configs)',
        'models': [
            # Baselines (~500M)
            ('mamba2', '--level mamba2 --dim 1600 --depth 32'),  # ~508M (auto depth)
            ('fla-gdn', '--level fla-gdn --dim 2304 --depth 20'),  # 533.7M

            # E88 many-heads n_state=32 (fully ablated)
            ('E88_h48n32', '--level E88_h48n32 --dim 3968 --depth 20'),  # 492.5M
            ('E88_h64n32', '--level E88_h64n32 --dim 3072 --depth 20'),  # 508.1M
            ('E88_h96n32', '--level E88_h96n32 --dim 2048 --depth 20'),  # 507.8M
            ('E88_h128n32', '--level E88_h128n32 --dim 1536 --depth 20'),  # 507.7M

            # E88 larger state variants
            ('E88_h32n64', '--level E88_h32n64 --dim 3072 --depth 20'),  # 506.1M
            ('E88_h24n64', '--level E88_h24n64 --dim 4096 --depth 20'),  # 506.4M
        ],
    },

    # Extended benchmarks for specific comparisons
    'e88_heads_100m': {
        'desc': 'E88 head count scaling at 100M params',
        'models': [
            ('fla-gdn', '--level fla-gdn --dim 1024 --depth 20'),
            ('E88_h8n32', '--level E88e_h8_75m --dim 3840 --depth 20'),  # 8 heads, ~100M
            ('E88_h12n32', '--level E88d_h12 --dim 3200 --depth 20'),  # 12 heads
            ('E88_h16n32', '--level E88c_nogate --dim 2560 --depth 20'),  # 16 heads
            ('E88_h20n32', '--level E88d_h20 --dim 1920 --depth 20'),  # 20 heads
        ],
    },

    'e88_state_100m': {
        'desc': 'E88 state size scaling at 100M params',
        'models': [
            ('fla-gdn', '--level fla-gdn --dim 1024 --depth 20'),
            ('E88_h32n16', '--level E88f_h32n16 --dim 2432 --depth 20'),  # n=16
            ('E88_h24n24', '--level E88f_h24n24 --dim 2176 --depth 20'),  # n=24
            ('E88_h16n32', '--level E88c_nogate --dim 2560 --depth 20'),  # n=32
            ('E88_h8n48', '--level E88h8n48 --dim 3200 --depth 20'),  # n=48
            ('E88_h8n64', '--level E88c_h8n64 --dim 2432 --depth 20'),  # n=64
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
