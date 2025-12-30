#!/usr/bin/env python3
"""
Analyze results from 500M comparison runs.

Usage:
    python analyze_comparison.py outputs/500m_comparison_TIMESTAMP

Outputs:
    - Summary table to stdout
    - summary.json with all metrics
    - loss_curves.png (if matplotlib available)
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


LEVEL_NAMES = {
    '0': 'StockElman',
    '1': 'GatedElman',
    '2': 'SelectiveElman',
    '3': 'DiagonalSelective',
    'log_0': 'LogSpacePolynomial',
    'log_1': 'LogSpaceSelective',
    'log_2': 'LogSpaceDiagonalSelective',
}


def load_steps(jsonl_path):
    """Load steps from JSONL file."""
    steps = []
    with open(jsonl_path) as f:
        for line in f:
            data = json.loads(line)
            if data.get('type') == 'step':
                steps.append(data)
    return steps


def analyze_level(level_dir):
    """Analyze a single level's training run."""
    level_dir = Path(level_dir)
    steps_path = level_dir / 'steps.jsonl'
    config_path = level_dir / 'config.json'

    if not steps_path.exists():
        return None

    steps = load_steps(steps_path)
    if not steps:
        return None

    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    # Extract metrics
    losses = [s['loss'] for s in steps]
    grad_norms = [s.get('grad_norm_total', 0) for s in steps]
    tokens_per_sec = [s.get('tokens_per_sec', 0) for s in steps]
    forward_times = [s.get('forward_time_ms', 0) for s in steps]
    backward_times = [s.get('backward_time_ms', 0) for s in steps]
    memory = [s.get('memory_allocated_mb', 0) for s in steps]

    # Compute summary stats
    final_loss = losses[-1] if losses else float('inf')
    min_loss = min(losses) if losses else float('inf')
    avg_tokens_per_sec = sum(tokens_per_sec) / len(tokens_per_sec) if tokens_per_sec else 0
    avg_forward_ms = sum(forward_times) / len(forward_times) if forward_times else 0
    avg_backward_ms = sum(backward_times) / len(backward_times) if backward_times else 0
    max_grad_norm = max(grad_norms) if grad_norms else 0
    peak_memory = max(memory) if memory else 0

    return {
        'level': config.get('level', level_dir.name),
        'num_params': config.get('num_params', 0),
        'num_steps': len(steps),
        'final_loss': final_loss,
        'min_loss': min_loss,
        'tokens_per_sec': avg_tokens_per_sec,
        'forward_ms': avg_forward_ms,
        'backward_ms': avg_backward_ms,
        'total_time_s': steps[-1].get('total_time_s', 0) if steps else 0,
        'tokens_seen': steps[-1].get('tokens_seen', 0) if steps else 0,
        'max_grad_norm': max_grad_norm,
        'peak_memory_mb': peak_memory,
        'losses': losses,
        'steps': [s['step'] for s in steps],
    }


def print_summary_table(results):
    """Print a summary table of results."""
    print("\n" + "=" * 100)
    print("500M PARAMETER COMPARISON SUMMARY")
    print("=" * 100)

    # Header
    print(f"{'Level':<8} {'Name':<25} {'Final Loss':>10} {'Min Loss':>10} "
          f"{'Tok/s':>10} {'Fwd(ms)':>8} {'Bwd(ms)':>8} {'Mem(MB)':>8}")
    print("-" * 100)

    # Sort by final loss
    sorted_results = sorted(results, key=lambda x: x['final_loss'])

    for r in sorted_results:
        level = r['level']
        name = LEVEL_NAMES.get(str(level), str(level))
        print(f"{level:<8} {name:<25} {r['final_loss']:>10.4f} {r['min_loss']:>10.4f} "
              f"{r['tokens_per_sec']:>10,.0f} {r['forward_ms']:>8.1f} {r['backward_ms']:>8.1f} "
              f"{r['peak_memory_mb']:>8.0f}")

    print("-" * 100)

    # Winner
    winner = sorted_results[0]
    print(f"\nBest: Level {winner['level']} ({LEVEL_NAMES.get(str(winner['level']), '')}) "
          f"with loss {winner['final_loss']:.4f}")


def plot_loss_curves(results, output_path):
    """Plot loss curves for all levels."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return

    plt.figure(figsize=(12, 6))

    colors = plt.cm.tab10(range(len(results)))

    for i, r in enumerate(results):
        level = str(r['level'])
        name = LEVEL_NAMES.get(level, level)
        plt.plot(r['steps'], r['losses'], label=f"{level}: {name}",
                 color=colors[i], alpha=0.8)

    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('500M Parameter Comparison: Training Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved loss curves to: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_comparison.py <output_dir>")
        print("Example: python analyze_comparison.py outputs/500m_comparison_20240101_120000")
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    if not output_dir.exists():
        print(f"Directory not found: {output_dir}")
        sys.exit(1)

    # Find all level directories
    results = []
    for level_dir in sorted(output_dir.iterdir()):
        if level_dir.is_dir() and (level_dir / 'steps.jsonl').exists():
            result = analyze_level(level_dir)
            if result:
                results.append(result)
                print(f"Loaded: {level_dir.name}")

    if not results:
        print("No results found!")
        sys.exit(1)

    # Print summary
    print_summary_table(results)

    # Save summary JSON
    summary_path = output_dir / 'summary.json'
    summary = {
        'levels': {r['level']: {k: v for k, v in r.items() if k not in ['losses', 'steps']}
                   for r in results},
        'best_level': min(results, key=lambda x: x['final_loss'])['level'],
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to: {summary_path}")

    # Plot loss curves
    plot_path = output_dir / 'loss_curves.png'
    plot_loss_curves(results, plot_path)


if __name__ == '__main__':
    main()
