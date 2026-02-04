#!/usr/bin/env python3
"""
Run CMA-ES v2 search for all models.

This script runs the improved two-phase search:
1. LHS exploration (48 samples for broad coverage)
2. CMA-ES refinement (from top 3 LHS configs)

For models with discrete parameters (e88 n_state), runs sweep mode.

Usage:
    python run_cmaes_v2.py                    # Run all models
    python run_cmaes_v2.py --only e88,mamba2  # Run specific models
    python run_cmaes_v2.py --dry-run          # Show what would run
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

# Models to search and their optimal strategies
# Increased LHS samples for tighter ±10% param tolerance
MODEL_CONFIGS = {
    # Model: (phase, sweep_param, lhs_samples, extra_args)
    'e88': ('sweep', 'n_state', 64, []),  # Sweep n_state=[16,32,48,64], 64 samples each
    'fla-gdn': ('both', None, 64, []),
    'mamba2': ('both', None, 64, []),
    'transformer': ('both', None, 64, []),
    'e1': ('both', None, 64, []),
    'e42': ('both', None, 64, []),
    'mingru': ('both', None, 64, []),
    'minlstm': ('both', None, 64, []),
}

# Skip these by default (unstable at 480M)
UNSTABLE_MODELS = ['e75', 'e23']


def run_search(model, output_dir, gpus, train_minutes, phase, sweep_param, lhs_samples, extra_args, dry_run=False):
    """Run CMA-ES v2 search for a model."""
    cmd = [
        'python', '-u', 'cmaes_search_v2.py',
        '--model', model,
        '--phase', phase,
        '--train_minutes', str(train_minutes),
        '--gpus', gpus,
        '--params', '480M',
        '--output', output_dir,
        '--lhs_samples', str(lhs_samples),
        '--sigma', '0.35',
        '--min_generations', '6',
        '--converge', '0.005',
        '--consecutive', '3',
        '--cmaes_refinements', '3',
    ]

    if sweep_param:
        cmd.extend(['--sweep_param', sweep_param])

    cmd.extend(extra_args)

    print(f"\nCommand: {' '.join(cmd)}")

    if dry_run:
        return 0, "DRY RUN"

    log_file = os.path.join(output_dir, f'{model}_search.log')

    with open(log_file, 'w') as log:
        log.write(f"CMA-ES v2 Search for {model}\n")
        log.write(f"Started: {datetime.now().isoformat()}\n")
        log.write(f"Command: {' '.join(cmd)}\n")
        log.write("=" * 70 + "\n\n")
        log.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in proc.stdout:
            print(line, end='')
            log.write(line)
            log.flush()

        proc.wait()

        log.write(f"\n{'='*70}\n")
        log.write(f"Finished: {datetime.now().isoformat()}\n")
        log.write(f"Return code: {proc.returncode}\n")

    return proc.returncode, log_file


def main():
    parser = argparse.ArgumentParser(description='Run CMA-ES v2 for all models')
    parser.add_argument('--only', type=str, default=None,
                        help='Only run these models (comma-separated)')
    parser.add_argument('--skip', type=str, default=None,
                        help='Skip these models (comma-separated)')
    parser.add_argument('--gpus', type=str, default='0,1,2,3,4,5,6,7',
                        help='GPUs to use')
    parser.add_argument('--train_minutes', type=float, default=30,
                        help='Training time per config')
    parser.add_argument('--output', type=str, default='benchmark_results/cmaes_v2',
                        help='Output directory')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would run')
    parser.add_argument('--include-unstable', action='store_true',
                        help='Include unstable models (e75, e23)')

    args = parser.parse_args()

    # Determine models to run
    if args.only:
        models = [m.strip() for m in args.only.split(',')]
    else:
        models = list(MODEL_CONFIGS.keys())
        if args.include_unstable:
            models.extend(UNSTABLE_MODELS)

    if args.skip:
        skip = [m.strip() for m in args.skip.split(',')]
        models = [m for m in models if m not in skip]

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    print("=" * 70)
    print("CMA-ES v2 SEARCH - Full Model Suite")
    print("=" * 70)
    print(f"\nModels to run ({len(models)}): {models}")
    print(f"Strategy: LHS exploration (48 samples) → CMA-ES refinement (top 3)")
    print(f"Training per eval: {args.train_minutes} minutes")
    print(f"Convergence: min_gen=6, threshold=0.005, consecutive=3")
    print(f"GPUs: {args.gpus}")
    print(f"Output: {args.output}")

    # Write run log
    run_log = os.path.join(args.output, 'run_all.log')
    with open(run_log, 'w') as f:
        f.write(f"\nModels to run ({len(models)}): {models}\n")
        f.write(f"Strategy: Two-phase (LHS → CMA-ES)\n")
        f.write(f"Training per eval: {args.train_minutes} minutes\n")
        f.write(f"Convergence: min_gen=6, threshold=0.005, consecutive=3\n")
        f.write(f"GPUs: {args.gpus}\n")
        f.write(f"Output: {args.output}\n\n")

    results = {}
    start_time = datetime.now()

    for i, model in enumerate(models):
        print(f"\n[{i+1}/{len(models)}]")
        print("=" * 70)
        print(f"Starting search for {model.upper()}")
        print("=" * 70)

        # Get config for this model
        if model in MODEL_CONFIGS:
            phase, sweep_param, lhs_samples, extra_args = MODEL_CONFIGS[model]
        else:
            phase, sweep_param, lhs_samples, extra_args = 'both', None, 48, []

        model_start = datetime.now()

        returncode, log_file = run_search(
            model, args.output, args.gpus, args.train_minutes,
            phase, sweep_param, lhs_samples, extra_args, args.dry_run
        )

        elapsed = (datetime.now() - model_start).total_seconds() / 3600
        status = "OK" if returncode == 0 else f"FAILED ({returncode})"

        results[model] = {
            'status': status,
            'hours': elapsed,
            'log': log_file,
        }

        # Update run log
        with open(run_log, 'a') as f:
            f.write(f"[{status}] {model} completed in {elapsed:.2f} hours\n")

        print(f"\n[{status}] {model} completed in {elapsed:.2f} hours")

    # Final summary
    total_hours = (datetime.now() - start_time).total_seconds() / 3600

    print(f"\n{'='*70}")
    print("CMA-ES v2 SEARCH SUMMARY")
    print(f"{'='*70}\n")

    print(f"{'Model':<15} {'Status':<10} {'Hours':>10}")
    print("-" * 40)
    for model, r in results.items():
        print(f"{model:<15} {r['status']:<10} {r['hours']:>10.2f}")
    print("-" * 40)
    print(f"\nTotal time: {total_hours:.2f} hours")
    print(f"Completed: {sum(1 for r in results.values() if 'OK' in r['status'])}/{len(results)} models")

    # Write summary
    with open(run_log, 'a') as f:
        f.write(f"\n{'='*70}\n")
        f.write("SUMMARY\n")
        f.write(f"{'='*70}\n")
        f.write(f"Total time: {total_hours:.2f} hours\n")
        for model, r in results.items():
            f.write(f"{model}: {r['status']} ({r['hours']:.2f}h)\n")


if __name__ == '__main__':
    main()
