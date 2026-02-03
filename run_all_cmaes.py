#!/usr/bin/env python3
"""
Run CMA-ES searches for all models until convergence.

Chains all model searches together, running one at a time (using all 8 GPUs per search).
Each search runs until improvement < 0.01 between generations.

Usage:
    python run_all_cmaes.py                    # Run all models
    python run_all_cmaes.py --resume fla-gdn   # Resume from fla-gdn
    python run_all_cmaes.py --only e88,e1      # Only run specific models
"""

import os
import sys
import subprocess
import time
import json
import argparse
from datetime import datetime
from pathlib import Path

# Models in priority order (most important baselines first)
MODELS = [
    'mamba2',      # SSM baseline
    'fla-gdn',     # ICLR 2025 baseline
    'e88',         # Best Elman variant
    'e1',          # Original gated Elman
    'e42',         # Linear tied (best simple Elman)
    'e75',         # Gated delta multihead
    'e23',         # Dual memory
    'mingru',      # MinGRU
    'minlstm',     # MinLSTM
    'transformer', # Attention baseline
]

# Workgraph task IDs for each model
TASK_IDS = {
    'mamba2': 'cma-es-search',
    'fla-gdn': 'cma-es-search-2',
    'e88': 'cma-es-search-3',
    'e1': 'cma-es-search-4',
    'e23': 'cma-es-search-5',
    'e42': 'cma-es-search-6',
    'e75': 'cma-es-search-7',
    'mingru': 'cma-es-search-8',
    'minlstm': 'cma-es-search-9',
    'transformer': 'cma-es-search-10',
}

# Configuration
TRAIN_MINUTES = 30
CONVERGE_THRESHOLD = 0.01
TARGET_PARAMS = '480M'
GPUS = '0,1,2,3,4,5,6,7'
OUTPUT_DIR = 'benchmark_results/cmaes_converge'


def run_command(cmd, capture=False):
    """Run a shell command."""
    if capture:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip()
    else:
        return subprocess.run(cmd, shell=True)


def update_workgraph(task_id, action, message=None):
    """Update workgraph task status."""
    if action == 'claim':
        run_command(f'wg claim {task_id} --actor cmaes-runner')
    elif action == 'log':
        run_command(f'wg log {task_id} "{message}"')
    elif action == 'done':
        run_command(f'wg done {task_id}')
    elif action == 'fail':
        run_command(f'wg fail {task_id} --reason "{message}"')


def run_cmaes_search(model):
    """Run CMA-ES search for a single model."""
    task_id = TASK_IDS.get(model)
    log_file = f'{OUTPUT_DIR}/{model}_search.log'

    print(f"\n{'='*70}")
    print(f"Starting CMA-ES search for {model.upper()}")
    print(f"{'='*70}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log: {log_file}")
    print(f"Config: train_minutes={TRAIN_MINUTES}, converge={CONVERGE_THRESHOLD}")

    # Update workgraph
    if task_id:
        update_workgraph(task_id, 'claim')
        update_workgraph(task_id, 'log', f'Starting search at {datetime.now().isoformat()}')

    # Build command
    cmd = [
        'python', 'cmaes_search.py',
        '--model', model,
        '--train_minutes', str(TRAIN_MINUTES),
        '--converge', str(CONVERGE_THRESHOLD),
        '--gpus', GPUS,
        '--params', TARGET_PARAMS,
        '--output', OUTPUT_DIR,
        '--start_from_best',
    ]

    # Run the search
    start_time = time.time()

    with open(log_file, 'w') as f:
        f.write(f"CMA-ES Search for {model}\n")
        f.write(f"Started: {datetime.now().isoformat()}\n")
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write("="*70 + "\n\n")
        f.flush()

        try:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
            )

            # Wait for completion
            return_code = process.wait()

            elapsed = time.time() - start_time
            hours = elapsed / 3600

            f.write(f"\n{'='*70}\n")
            f.write(f"Finished: {datetime.now().isoformat()}\n")
            f.write(f"Duration: {hours:.2f} hours\n")
            f.write(f"Return code: {return_code}\n")

            if return_code == 0:
                print(f"✓ {model} completed in {hours:.2f} hours")
                if task_id:
                    update_workgraph(task_id, 'log', f'Completed in {hours:.2f} hours')
                    update_workgraph(task_id, 'done')
                return True, hours
            else:
                print(f"✗ {model} failed with return code {return_code}")
                if task_id:
                    update_workgraph(task_id, 'fail', f'Return code {return_code}')
                return False, hours

        except Exception as e:
            elapsed = time.time() - start_time
            hours = elapsed / 3600
            print(f"✗ {model} error: {e}")
            f.write(f"\nError: {e}\n")
            if task_id:
                update_workgraph(task_id, 'fail', str(e))
            return False, hours


def extract_best_result(model):
    """Extract best result from completed search."""
    # Find the results directory
    results_dirs = list(Path(OUTPUT_DIR).glob(f'{model}_*'))
    if not results_dirs:
        return None

    # Get most recent
    latest_dir = max(results_dirs, key=lambda p: p.stat().st_mtime)
    results_file = latest_dir / 'results.json'

    if not results_file.exists():
        return None

    with open(results_file) as f:
        return json.load(f)


def is_search_complete(model):
    """Check if a model's search has already completed successfully."""
    result = extract_best_result(model)
    if result is None:
        return False

    # Check if it has a valid best_loss (not inf or very high)
    best_loss = result.get('best_loss', float('inf'))
    if best_loss is None or best_loss > 5.0:
        return False

    # Check if it has best_params
    if not result.get('best_params'):
        return False

    return True


def get_completed_models():
    """Get list of models that have already completed."""
    completed = []
    for model in MODELS:
        if is_search_complete(model):
            completed.append(model)
    return completed


def print_summary(results):
    """Print summary of all results."""
    print(f"\n{'='*70}")
    print("CMA-ES SEARCH SUMMARY")
    print(f"{'='*70}\n")

    print(f"{'Model':<15} {'Status':<10} {'Hours':>8} {'Loss':>10} {'Config'}")
    print("-"*70)

    for model, (success, hours, best) in results.items():
        status = "✓" if success else "✗"
        loss = f"{best['best_loss']:.4f}" if best else "N/A"
        config = str(best.get('best_params', {})) if best else ""
        if len(config) > 40:
            config = config[:37] + "..."
        print(f"{model:<15} {status:<10} {hours:>8.2f} {loss:>10} {config}")

    print("-"*70)

    # Find best overall
    valid_results = [(m, r) for m, r in results.items() if r[2] and r[0]]
    if valid_results:
        best_model, best_data = min(valid_results, key=lambda x: x[1][2]['best_loss'])
        print(f"\nBest model: {best_model} with loss {best_data[2]['best_loss']:.4f}")
        print(f"Config: {best_data[2]['best_params']}")


def main():
    parser = argparse.ArgumentParser(description='Run CMA-ES searches for all models')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from this model (skip earlier ones)')
    parser.add_argument('--only', type=str, default=None,
                        help='Only run these models (comma-separated)')
    parser.add_argument('--skip', type=str, default=None,
                        help='Skip these models (comma-separated)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be run without executing')
    parser.add_argument('--force', action='store_true',
                        help='Force re-run even if already completed')
    parser.add_argument('--auto-resume', action='store_true', default=True,
                        help='Automatically skip already-completed models (default: True)')
    parser.add_argument('--no-auto-resume', action='store_false', dest='auto_resume',
                        help='Do not auto-skip completed models')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Determine which models to run
    models_to_run = MODELS.copy()

    if args.only:
        only_models = [m.strip() for m in args.only.split(',')]
        models_to_run = [m for m in models_to_run if m in only_models]

    if args.skip:
        skip_models = [m.strip() for m in args.skip.split(',')]
        models_to_run = [m for m in models_to_run if m not in skip_models]

    if args.resume:
        if args.resume in models_to_run:
            idx = models_to_run.index(args.resume)
            models_to_run = models_to_run[idx:]
        else:
            print(f"Error: resume model '{args.resume}' not in list")
            sys.exit(1)

    # Auto-detect and skip completed models
    if args.auto_resume and not args.force:
        completed = get_completed_models()
        if completed:
            print(f"Already completed: {completed}")
            for model in completed:
                result = extract_best_result(model)
                if result:
                    print(f"  {model}: loss={result['best_loss']:.4f}, params={result['best_params']}")
            models_to_run = [m for m in models_to_run if m not in completed]
            print(f"Skipping completed models, will run: {models_to_run}")

    if not models_to_run:
        print("All models already completed! Use --force to re-run.")
        return

    print(f"\nModels to run: {models_to_run}")
    print(f"Training time per evaluation: {TRAIN_MINUTES} minutes")
    print(f"Convergence threshold: {CONVERGE_THRESHOLD}")
    print(f"Target params: {TARGET_PARAMS}")
    print(f"GPUs: {GPUS}")

    if args.dry_run:
        print("\n[DRY RUN - not executing]")
        return

    # Run all searches
    results = {}
    total_start = time.time()

    for i, model in enumerate(models_to_run):
        print(f"\n[{i+1}/{len(models_to_run)}] ", end="")
        success, hours = run_cmaes_search(model)
        best = extract_best_result(model)
        results[model] = (success, hours, best)

        # Save intermediate results
        with open(f'{OUTPUT_DIR}/summary.json', 'w') as f:
            summary = {
                model: {
                    'success': success,
                    'hours': hours,
                    'best_loss': best['best_loss'] if best else None,
                    'best_params': best['best_params'] if best else None,
                }
                for model, (success, hours, best) in results.items()
            }
            json.dump(summary, f, indent=2)

    total_hours = (time.time() - total_start) / 3600

    # Print summary
    print_summary(results)
    print(f"\nTotal time: {total_hours:.2f} hours")

    # Update final summary task if all done
    if len(results) == len(MODELS):
        run_command('wg claim final-summary-compile --actor cmaes-runner')
        run_command(f'wg log final-summary-compile "All searches complete in {total_hours:.2f} hours"')
        run_command('wg done final-summary-compile')


if __name__ == '__main__':
    main()
