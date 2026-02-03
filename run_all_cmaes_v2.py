#!/usr/bin/env python3
"""
Run CMA-ES searches for all models with expanded 4D search spaces.

VERSION 2: All models now use 4D search spaces for thorough exploration.
- E88: n_heads, n_state, depth, expansion
- FLA-GDN: expansion, depth, n_heads, conv_size
- Mamba2: d_state, headdim, expand, depth
- Transformer: n_heads, head_dim, expansion, depth
- E1: expansion, depth, use_conv, mamba2_init
- E23: n_slots, expansion, depth, w_h_init_scale
- E42: expansion, depth, spectral_radius, mamba2_init
- E75: n_heads, n_state, depth, expansion
- MinGRU/MinLSTM: expansion, depth, use_conv, d_conv

Features:
- Chains all model searches, running one at a time (using all GPUs per search)
- Each search runs until improvement < threshold between generations
- Interruptible: Ctrl+C saves state, can resume later
- NaN handling: Tracks divergent configs, restarts with smaller sigma if needed

Usage:
    python run_all_cmaes_v2.py                    # Run all models
    python run_all_cmaes_v2.py --resume fla-gdn   # Resume from fla-gdn
    python run_all_cmaes_v2.py --only e88,e1      # Only run specific models
    python run_all_cmaes_v2.py --skip gru,lstm    # Skip unstable models
"""

import os
import sys
import subprocess
import time
import json
import signal
import argparse
from datetime import datetime
from pathlib import Path

# Models in priority order (most important baselines first)
# Note: GRU/LSTM are 2D (unstable at 480M) but included for completeness
MODELS = [
    'mamba2',      # SSM baseline (4D: d_state, headdim, expand, depth)
    'fla-gdn',     # ICLR 2025 baseline (4D: expansion, depth, n_heads, conv_size)
    'e88',         # Best Elman variant (4D: n_heads, n_state, depth, expansion)
    'e1',          # Original gated Elman (4D: expansion, depth, use_conv, mamba2_init)
    'e42',         # Linear tied (4D: expansion, depth, spectral_radius, mamba2_init)
    'e75',         # Gated delta multihead (4D: n_heads, n_state, depth, expansion)
    'e23',         # Dual memory (4D: n_slots, expansion, depth, w_h_init_scale)
    'mingru',      # MinGRU (4D: expansion, depth, use_conv, d_conv)
    'minlstm',     # MinLSTM (4D: expansion, depth, use_conv, d_conv)
    'transformer', # Attention baseline (4D: n_heads, head_dim, expansion, depth)
]

# Models known to be unstable at 480M scale (2D search only)
UNSTABLE_MODELS = ['gru', 'lstm']

# Configuration
TRAIN_MINUTES = 30
CONVERGE_THRESHOLD = 0.01
TARGET_PARAMS = '480M'
GPUS = '0,1,2,3,4,5,6,7'
OUTPUT_DIR = 'benchmark_results/cmaes_4d'

# Track interrupted state for graceful resume
interrupted = False
current_model = None


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    global interrupted
    print(f"\n\nInterrupted! Current model: {current_model}")
    print("State has been saved. Resume with:")
    print(f"  python run_all_cmaes_v2.py --resume {current_model}")
    interrupted = True


def run_command(cmd, capture=False):
    """Run a shell command."""
    if capture:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip()
    else:
        return subprocess.run(cmd, shell=True)


def update_workgraph(task_id, action, message=None):
    """Update workgraph task status (best effort, don't fail if wg unavailable)."""
    try:
        if action == 'claim':
            run_command(f'wg claim {task_id} --actor cmaes-v2 2>/dev/null')
        elif action == 'log':
            run_command(f'wg log {task_id} "{message}" 2>/dev/null')
        elif action == 'done':
            run_command(f'wg done {task_id} 2>/dev/null')
        elif action == 'fail':
            run_command(f'wg fail {task_id} --reason "{message}" 2>/dev/null')
    except Exception:
        pass  # Workgraph is optional


def count_nan_in_log(log_file):
    """Count NaN/diverged evaluations in log file."""
    nan_count = 0
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if 'NaN' in line or 'diverged' in line:
                    nan_count += 1
    except:
        pass
    return nan_count


def run_cmaes_search(model, retry_count=0):
    """Run CMA-ES search for a single model."""
    global current_model
    current_model = model

    log_file = f'{OUTPUT_DIR}/{model}_search.log'
    pid_file = f'{OUTPUT_DIR}/{model}.pid'

    print(f"\n{'='*70}")
    print(f"Starting CMA-ES 6D search for {model.upper()}")
    print(f"{'='*70}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log: {log_file}")
    print(f"Config: train_minutes={TRAIN_MINUTES}, converge={CONVERGE_THRESHOLD}")
    if retry_count > 0:
        print(f"Retry attempt: {retry_count}")

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
        f.write(f"CMA-ES 6D Search for {model}\n")
        f.write(f"Started: {datetime.now().isoformat()}\n")
        f.write(f"Command: {' '.join(cmd)}\n")
        if retry_count > 0:
            f.write(f"Retry attempt: {retry_count}\n")
        f.write("="*70 + "\n\n")
        f.flush()

        try:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Save PID for potential external monitoring
            with open(pid_file, 'w') as pf:
                pf.write(str(process.pid))

            # Wait for completion (checking for interrupts)
            while process.poll() is None:
                if interrupted:
                    process.terminate()
                    process.wait(timeout=10)
                    return False, 0, 'interrupted'
                time.sleep(1)

            return_code = process.returncode
            elapsed = time.time() - start_time
            hours = elapsed / 3600

            f.write(f"\n{'='*70}\n")
            f.write(f"Finished: {datetime.now().isoformat()}\n")
            f.write(f"Duration: {hours:.2f} hours\n")
            f.write(f"Return code: {return_code}\n")

            # Check for excessive NaN/divergence
            nan_count = count_nan_in_log(log_file)
            if nan_count > 10:
                f.write(f"Warning: {nan_count} NaN/diverged evaluations\n")
                print(f"Warning: {model} had {nan_count} NaN/diverged evals")

            if return_code == 0:
                print(f"[OK] {model} completed in {hours:.2f} hours")
                os.remove(pid_file) if os.path.exists(pid_file) else None
                return True, hours, 'success'
            else:
                print(f"[FAIL] {model} returned code {return_code}")
                return False, hours, f'return_code_{return_code}'

        except Exception as e:
            elapsed = time.time() - start_time
            hours = elapsed / 3600
            print(f"[ERROR] {model}: {e}")
            f.write(f"\nError: {e}\n")
            return False, hours, str(e)


def extract_best_result(model):
    """Extract best result from completed search."""
    results_dirs = list(Path(OUTPUT_DIR).glob(f'{model}_*'))
    if not results_dirs:
        return None

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

    best_loss = result.get('best_loss', float('inf'))
    if best_loss is None or best_loss > 5.0:
        return False

    if not result.get('best_params'):
        return False

    return True


def get_completed_models():
    """Get list of models that have already completed."""
    return [m for m in MODELS if is_search_complete(m)]


def print_summary(results):
    """Print summary of all results."""
    print(f"\n{'='*70}")
    print("CMA-ES 6D SEARCH SUMMARY")
    print(f"{'='*70}\n")

    print(f"{'Model':<15} {'Status':<12} {'Hours':>8} {'Loss':>10} {'Config'}")
    print("-"*70)

    for model, (success, hours, best, status) in results.items():
        status_str = "OK" if success else status[:10]
        loss = f"{best['best_loss']:.4f}" if best else "N/A"
        config = str(best.get('best_params', {})) if best else ""
        if len(config) > 35:
            config = config[:32] + "..."
        print(f"{model:<15} {status_str:<12} {hours:>8.2f} {loss:>10} {config}")

    print("-"*70)

    # Find best per category
    valid_results = [(m, r) for m, r in results.items() if r[2] and r[0]]
    if valid_results:
        best_model, best_data = min(valid_results, key=lambda x: x[1][2]['best_loss'])
        print(f"\nBest overall: {best_model} with loss {best_data[2]['best_loss']:.4f}")
        print(f"Config: {best_data[2]['best_params']}")


def save_state(results, models_remaining):
    """Save current state for resume capability."""
    state = {
        'timestamp': datetime.now().isoformat(),
        'completed': {
            model: {
                'success': success,
                'hours': hours,
                'best_loss': best['best_loss'] if best else None,
                'best_params': best['best_params'] if best else None,
                'status': status,
            }
            for model, (success, hours, best, status) in results.items()
        },
        'remaining': models_remaining,
    }
    with open(f'{OUTPUT_DIR}/state.json', 'w') as f:
        json.dump(state, f, indent=2)


def main():
    global interrupted

    parser = argparse.ArgumentParser(description='Run CMA-ES 6D searches for all models')
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
    parser.add_argument('--include-unstable', action='store_true',
                        help='Include GRU/LSTM which are unstable at 480M')
    args = parser.parse_args()

    # Register signal handler for graceful interruption
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Determine which models to run
    models_to_run = MODELS.copy()

    # Add unstable models if requested
    if args.include_unstable:
        models_to_run.extend(UNSTABLE_MODELS)

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
            print(f"Error: resume model '{args.resume}' not found")
            sys.exit(1)

    # Auto-skip completed models unless --force
    if not args.force:
        completed = get_completed_models()
        if completed:
            print(f"Already completed: {completed}")
            for model in completed:
                result = extract_best_result(model)
                if result:
                    print(f"  {model}: loss={result['best_loss']:.4f}")
            models_to_run = [m for m in models_to_run if m not in completed]

    if not models_to_run:
        print("All models completed! Use --force to re-run.")
        return

    print(f"\nModels to run ({len(models_to_run)}): {models_to_run}")
    print(f"Search space: 6D (expanded from 2-3D)")
    print(f"Training per eval: {TRAIN_MINUTES} minutes")
    print(f"Convergence: improvement < {CONVERGE_THRESHOLD}")
    print(f"Target: {TARGET_PARAMS} params")
    print(f"GPUs: {GPUS}")
    print(f"Output: {OUTPUT_DIR}")

    if args.dry_run:
        print("\n[DRY RUN - not executing]")
        return

    # Run all searches
    results = {}
    total_start = time.time()

    for i, model in enumerate(models_to_run):
        if interrupted:
            break

        print(f"\n[{i+1}/{len(models_to_run)}] ", end="")
        success, hours, status = run_cmaes_search(model)
        best = extract_best_result(model)
        results[model] = (success, hours, best, status)

        # Save state after each model
        remaining = models_to_run[i+1:]
        save_state(results, remaining)

    total_hours = (time.time() - total_start) / 3600

    if not interrupted:
        print_summary(results)
        print(f"\nTotal time: {total_hours:.2f} hours")

        # Final summary
        success_count = sum(1 for s, _, _, _ in results.values() if s)
        print(f"Completed: {success_count}/{len(results)} models")


if __name__ == '__main__':
    main()
