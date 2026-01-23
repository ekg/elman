#!/usr/bin/env python3
"""
Overnight E88 Configuration Exploration

Systematically explores E88 configurations at 500M scale to find what works.
Runs wave after wave of experiments, tracking results and exploring promising directions.

Target: Beat Mamba2 (~1.81 last-100 loss) and approach FLA-GDN (~1.73)

Key dimensions to explore:
- depth: 20, 24, 28, 32, 36
- n_state: 16, 24, 32, 40, 48
- n_heads: calculated to hit ~500M params
- The balancing principle: n_heads × n_state ≈ dim

Usage:
    python overnight_e88_exploration.py --data /path/to/data.txt
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import re

# Add elman to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from calc_dim import calc_e88_params, find_dim_for_params


def calc_last_100_loss(log_file):
    """Calculate last-100 step average loss from log file."""
    losses = []
    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(r'loss\s+([\d.]+)', line)
            if match:
                losses.append(float(match.group(1)))

    if len(losses) >= 100:
        return sum(losses[-100:]) / 100
    elif len(losses) > 0:
        return sum(losses) / len(losses)
    return None


def run_experiment(config, data_path, output_dir, train_minutes=10):
    """Run a single E88 experiment and return results."""
    name = config['name']
    dim = config['dim']
    depth = config['depth']
    n_heads = config['n_heads']
    n_state = config['n_state']
    expansion = config.get('expansion', 1.0)

    # Create output directory
    exp_dir = Path(output_dir) / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_file = exp_dir / f"{name}.log"

    # Build command
    # use_gate=0 for "best" config (no output gating)
    # bf16 required for CUDA kernel
    # batch_size=16 to fit in GPU memory at 500M scale
    cmd = [
        'python', 'train.py',
        '--data', data_path,
        '--level', 'E88',
        '--dim', str(dim),
        '--depth', str(depth),
        '--n_heads', str(n_heads),
        '--n_state', str(n_state),
        '--expansion', str(expansion),
        '--use_gate', '0',  # Best E88 config has no gate
        '--bf16',  # Required for CUDA kernel
        '--batch_size', '16',
        '--chunk_size', '512',
        '--lr', '3e-4',
        '--optimizer', 'schedulefree',
        '--train_minutes', str(train_minutes),
        '--output', str(exp_dir),
    ]

    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"  dim={dim}, depth={depth}, n_heads={n_heads}, n_state={n_state}")
    print(f"  params≈{config['params']/1e6:.1f}M, state/layer={n_heads * n_state * n_state:,}")
    print(f"  balance ratio={(n_heads * n_state) / dim:.2f}")
    print(f"{'='*60}")

    start_time = time.time()

    with open(log_file, 'w') as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=os.path.dirname(os.path.abspath(__file__)))

    elapsed = time.time() - start_time

    # Calculate loss
    loss = calc_last_100_loss(log_file)

    result = {
        'name': name,
        'dim': dim,
        'depth': depth,
        'n_heads': n_heads,
        'n_state': n_state,
        'expansion': expansion,
        'params': config['params'],
        'state_per_layer': n_heads * n_state * n_state,
        'balance_ratio': (n_heads * n_state) / dim,
        'last_100_loss': loss,
        'elapsed_seconds': elapsed,
        'log_file': str(log_file),
    }

    if loss:
        print(f"  Result: last-100 loss = {loss:.4f}")
    else:
        print(f"  Result: FAILED (no loss data)")

    return result


def generate_configs(target_params=500_000_000):
    """Generate E88 configurations to explore."""
    configs = []

    # Wave 1: Different depths with balanced configs
    # Balance principle: n_heads × n_state ≈ dim
    print("Generating Wave 1: Depth exploration with balanced configs")

    for depth in [20, 24, 28, 32]:
        for n_state in [32, 40, 48]:
            # Find dim and n_heads for target params
            # Start with n_heads such that n_heads × n_state is close to reasonable dim
            for n_heads in range(16, 128, 4):
                dim, params = find_dim_for_params(
                    calc_e88_params, target_params,
                    n_heads=n_heads, n_state=n_state, depth=depth, expansion=1.0, use_gate=False
                )
                # Check if balanced (ratio 0.5 - 2.0)
                ratio = (n_heads * n_state) / dim
                if 0.7 <= ratio <= 1.5 and abs(params - target_params) < 50_000_000:
                    name = f"E88_d{depth}_h{n_heads}_n{n_state}"
                    configs.append({
                        'name': name,
                        'dim': dim,
                        'depth': depth,
                        'n_heads': n_heads,
                        'n_state': n_state,
                        'expansion': 1.0,
                        'params': params,
                        'wave': 1,
                    })
                    break  # Found a good config for this depth/n_state combo

    # Wave 2: More heads with smaller state
    print("Generating Wave 2: Many heads, small state")
    for depth in [24, 32]:
        for n_state in [16, 24]:
            for n_heads in range(32, 200, 8):
                dim, params = find_dim_for_params(
                    calc_e88_params, target_params,
                    n_heads=n_heads, n_state=n_state, depth=depth, expansion=1.0, use_gate=False
                )
                ratio = (n_heads * n_state) / dim
                if 0.7 <= ratio <= 1.5 and abs(params - target_params) < 50_000_000:
                    name = f"E88_d{depth}_h{n_heads}_n{n_state}"
                    # Avoid duplicates
                    if not any(c['name'] == name for c in configs):
                        configs.append({
                            'name': name,
                            'dim': dim,
                            'depth': depth,
                            'n_heads': n_heads,
                            'n_state': n_state,
                            'expansion': 1.0,
                            'params': params,
                            'wave': 2,
                        })
                    break

    # Wave 3: Deeper models
    print("Generating Wave 3: Deep models (36, 40 layers)")
    for depth in [36, 40]:
        for n_state in [32, 40]:
            for n_heads in range(16, 100, 4):
                dim, params = find_dim_for_params(
                    calc_e88_params, target_params,
                    n_heads=n_heads, n_state=n_state, depth=depth, expansion=1.0, use_gate=False
                )
                ratio = (n_heads * n_state) / dim
                if 0.7 <= ratio <= 1.5 and abs(params - target_params) < 50_000_000:
                    name = f"E88_d{depth}_h{n_heads}_n{n_state}"
                    if not any(c['name'] == name for c in configs):
                        configs.append({
                            'name': name,
                            'dim': dim,
                            'depth': depth,
                            'n_heads': n_heads,
                            'n_state': n_state,
                            'expansion': 1.0,
                            'params': params,
                            'wave': 3,
                        })
                    break

    # Wave 4: Unbalanced exploration (higher/lower ratios)
    print("Generating Wave 4: Ratio exploration")
    for depth in [24, 32]:
        for n_state in [32, 48]:
            # Try different ratios
            for target_ratio in [0.5, 0.6, 1.8, 2.0]:
                # n_heads × n_state / dim = target_ratio
                # So n_heads = target_ratio × dim / n_state
                for n_heads in range(8, 150, 4):
                    dim, params = find_dim_for_params(
                        calc_e88_params, target_params,
                        n_heads=n_heads, n_state=n_state, depth=depth, expansion=1.0, use_gate=False
                    )
                    ratio = (n_heads * n_state) / dim
                    if abs(ratio - target_ratio) < 0.15 and abs(params - target_params) < 50_000_000:
                        name = f"E88_d{depth}_h{n_heads}_n{n_state}_r{int(target_ratio*10)}"
                        if not any(c['name'] == name for c in configs):
                            configs.append({
                                'name': name,
                                'dim': dim,
                                'depth': depth,
                                'n_heads': n_heads,
                                'n_state': n_state,
                                'expansion': 1.0,
                                'params': params,
                                'wave': 4,
                            })
                        break

    # Wave 5: New n_state values (36, 40, 44)
    print("Generating Wave 5: New n_state values (36, 40, 44)")
    for depth in [24, 28, 32]:
        for n_state in [36, 40, 44]:
            for n_heads in range(16, 100, 4):
                dim, params = find_dim_for_params(
                    calc_e88_params, target_params,
                    n_heads=n_heads, n_state=n_state, depth=depth, expansion=1.0, use_gate=False
                )
                ratio = (n_heads * n_state) / dim
                if 0.7 <= ratio <= 1.5 and abs(params - target_params) < 50_000_000:
                    name = f"E88_d{depth}_h{n_heads}_n{n_state}"
                    if not any(c['name'] == name for c in configs):
                        configs.append({
                            'name': name,
                            'dim': dim,
                            'depth': depth,
                            'n_heads': n_heads,
                            'n_state': n_state,
                            'expansion': 1.0,
                            'params': params,
                            'wave': 5,
                        })
                    break

    return configs


def main():
    parser = argparse.ArgumentParser(description='Overnight E88 exploration')
    parser.add_argument('--data', type=str, default='data/pile.txt',
                        help='Training data path')
    parser.add_argument('--output', type=str, default='benchmark_results/overnight_e88',
                        help='Output directory')
    parser.add_argument('--train_minutes', type=float, default=10,
                        help='Training time per config (minutes)')
    parser.add_argument('--target_params', type=str, default='500M',
                        help='Target parameter count')
    parser.add_argument('--max_configs', type=int, default=None,
                        help='Maximum configs to run (for testing)')
    parser.add_argument('--wave', type=int, default=None,
                        help='Run only specific wave (1-5)')
    args = parser.parse_args()

    # Parse target params
    target_str = args.target_params.upper()
    if target_str.endswith('M'):
        target_params = int(float(target_str[:-1]) * 1_000_000)
    elif target_str.endswith('B'):
        target_params = int(float(target_str[:-1]) * 1_000_000_000)
    else:
        target_params = int(target_str)

    # Generate configurations
    configs = generate_configs(target_params)

    # Filter by wave if specified
    if args.wave:
        configs = [c for c in configs if c.get('wave') == args.wave]

    # Limit configs if specified
    if args.max_configs:
        configs = configs[:args.max_configs]

    print(f"\nGenerated {len(configs)} configurations to explore")
    print(f"Target: ~{target_params/1e6:.0f}M parameters")
    print(f"Training time: {args.train_minutes} minutes per config")
    print(f"Estimated total time: {len(configs) * args.train_minutes / 60:.1f} hours")

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config list
    with open(output_dir / 'configs.json', 'w') as f:
        json.dump(configs, f, indent=2)

    # Run experiments
    results = []
    best_loss = float('inf')
    best_config = None

    print(f"\n{'='*60}")
    print("STARTING OVERNIGHT EXPLORATION")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] ", end='')

        result = run_experiment(config, args.data, output_dir, args.train_minutes)
        results.append(result)

        # Track best
        if result['last_100_loss'] and result['last_100_loss'] < best_loss:
            best_loss = result['last_100_loss']
            best_config = result
            print(f"  *** NEW BEST: {best_loss:.4f} ***")

        # Save results after each experiment
        with open(output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)

        # Print summary table
        print(f"\n--- Current Results Summary ---")
        sorted_results = sorted([r for r in results if r['last_100_loss']],
                               key=lambda x: x['last_100_loss'])
        for j, r in enumerate(sorted_results[:5]):
            marker = "***" if j == 0 else ""
            print(f"  {j+1}. {r['name']}: {r['last_100_loss']:.4f} {marker}")
        print(f"  Baseline targets: Mamba2=1.81, FLA-GDN=1.73")

    # Final summary
    print(f"\n{'='*60}")
    print("EXPLORATION COMPLETE")
    print(f"{'='*60}")

    if best_config:
        print(f"\nBest configuration:")
        print(f"  Name: {best_config['name']}")
        print(f"  Loss: {best_config['last_100_loss']:.4f}")
        print(f"  dim={best_config['dim']}, depth={best_config['depth']}")
        print(f"  n_heads={best_config['n_heads']}, n_state={best_config['n_state']}")
        print(f"  balance_ratio={best_config['balance_ratio']:.2f}")

        # Compare to baselines
        print(f"\nVs Baselines:")
        print(f"  vs Mamba2 (1.81): {best_config['last_100_loss'] - 1.81:+.4f}")
        print(f"  vs FLA-GDN (1.73): {best_config['last_100_loss'] - 1.73:+.4f}")

    print(f"\nResults saved to: {output_dir}")

    # Create final report
    with open(output_dir / 'REPORT.md', 'w') as f:
        f.write(f"# E88 Overnight Exploration Report\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Target params: {target_params/1e6:.0f}M\n")
        f.write(f"Training time per config: {args.train_minutes} min\n")
        f.write(f"Total configs tested: {len(results)}\n\n")

        f.write(f"## Best Results (Top 10)\n\n")
        f.write(f"| Rank | Config | Loss | dim | depth | heads | n_state | ratio |\n")
        f.write(f"|------|--------|------|-----|-------|-------|---------|-------|\n")

        sorted_results = sorted([r for r in results if r['last_100_loss']],
                               key=lambda x: x['last_100_loss'])
        for j, r in enumerate(sorted_results[:10]):
            f.write(f"| {j+1} | {r['name']} | {r['last_100_loss']:.4f} | {r['dim']} | {r['depth']} | {r['n_heads']} | {r['n_state']} | {r['balance_ratio']:.2f} |\n")

        f.write(f"\n## Baselines\n\n")
        f.write(f"- Mamba2 (500M): 1.81 last-100 loss\n")
        f.write(f"- FLA-GDN (500M): 1.73 last-100 loss\n")

        if best_config:
            f.write(f"\n## Best E88 vs Baselines\n\n")
            f.write(f"- Best E88: {best_config['last_100_loss']:.4f}\n")
            f.write(f"- vs Mamba2: {best_config['last_100_loss'] - 1.81:+.4f}\n")
            f.write(f"- vs FLA-GDN: {best_config['last_100_loss'] - 1.73:+.4f}\n")


if __name__ == '__main__':
    main()
