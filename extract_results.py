#!/usr/bin/env python3
"""
Extract benchmark results from logs

Computes last-100 step average loss (smoother than point estimate).

Usage:
    # Extract from scheduler logs
    python extract_results.py sched_logs/20260119_*/

    # Extract from benchmark directory
    python extract_results.py benchmark_results/e75_100m_*/

    # Output as JSON
    python extract_results.py --json benchmark_results/*/
"""

import argparse
import re
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional


def extract_from_log(log_path: Path) -> Optional[Dict]:
    """Extract metrics from a single log file."""
    try:
        with open(log_path) as f:
            content = f.read()
    except:
        return None

    result = {
        'log': str(log_path),
        'name': log_path.stem,
    }

    # Extract model name from command or filename
    if '--level' in content:
        match = re.search(r'--level\s+(\S+)', content)
        if match:
            result['name'] = match.group(1)

    # Extract parameters
    match = re.search(r'([\d,]+)\s*parameters', content)
    if match:
        result['params'] = int(match.group(1).replace(',', ''))

    # Extract dim
    match = re.search(r'--dim\s+(\d+)', content)
    if match:
        result['dim'] = int(match.group(1))

    # Extract all step losses
    losses = []
    for match in re.finditer(r'step\s+(\d+)\s+\|\s+loss\s+([\d.]+)', content):
        step = int(match.group(1))
        loss = float(match.group(2))
        losses.append((step, loss))

    if not losses:
        result['status'] = 'no_data'
        return result

    # Check for NaN
    if any(loss != loss for _, loss in losses):  # NaN check
        result['status'] = 'nan'
        result['nan_step'] = next(s for s, l in losses if l != l)
        return result

    # Final step
    result['final_step'] = losses[-1][0]
    result['final_loss'] = losses[-1][1]

    # Last-100 average (or all if < 100)
    last_n = min(100, len(losses))
    last_losses = [l for _, l in losses[-last_n:]]
    result['last100_avg'] = sum(last_losses) / len(last_losses)

    # Last-50 average for comparison
    last_n = min(50, len(losses))
    last_losses = [l for _, l in losses[-last_n:]]
    result['last50_avg'] = sum(last_losses) / len(last_losses)

    # Throughput (exclude first 10 steps for compilation)
    toks = re.findall(r'tok/s\s+(\d+)', content)
    if len(toks) > 10:
        toks = [int(t) for t in toks[10:]]
        result['avg_throughput'] = sum(toks) / len(toks)
    elif toks:
        toks = [int(t) for t in toks]
        result['avg_throughput'] = sum(toks) / len(toks)

    result['status'] = 'ok'
    return result


def find_logs(paths: List[str]) -> List[Path]:
    """Find all log files in given paths."""
    logs = []
    for p in paths:
        path = Path(p)
        if path.is_file() and path.suffix == '.log':
            logs.append(path)
        elif path.is_dir():
            logs.extend(path.glob('*.log'))
            logs.extend(path.glob('**/*.log'))
    return sorted(set(logs))


def main():
    parser = argparse.ArgumentParser(description='Extract benchmark results from logs')
    parser.add_argument('paths', nargs='+', help='Log files or directories')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('--sort', choices=['loss', 'name', 'params', 'throughput'],
                        default='loss', help='Sort by field')
    args = parser.parse_args()

    logs = find_logs(args.paths)
    if not logs:
        print("No log files found.", file=sys.stderr)
        return

    results = []
    for log in logs:
        result = extract_from_log(log)
        if result:
            results.append(result)

    # Sort
    def sort_key(r):
        if args.sort == 'loss':
            return r.get('last100_avg', 999)
        elif args.sort == 'throughput':
            return -r.get('avg_throughput', 0)
        elif args.sort == 'params':
            return r.get('params', 0)
        else:
            return r.get('name', '')

    results.sort(key=sort_key)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"{'Model':<14} {'Params':>10} {'Steps':>6} {'Last100':>8} {'Last50':>8} {'Throughput':>12} {'Status'}")
        print("-" * 80)

        for r in results:
            params = f"{r.get('params', 0)/1e6:.1f}M" if r.get('params') else 'N/A'
            steps = str(r.get('final_step', '-'))
            last100 = f"{r.get('last100_avg', 0):.4f}" if r.get('last100_avg') else 'N/A'
            last50 = f"{r.get('last50_avg', 0):.4f}" if r.get('last50_avg') else 'N/A'
            toks = f"{r.get('avg_throughput', 0):.0f} tok/s" if r.get('avg_throughput') else 'N/A'
            status = r.get('status', 'unknown')

            print(f"{r['name']:<14} {params:>10} {steps:>6} {last100:>8} {last50:>8} {toks:>12} {status}")


if __name__ == '__main__':
    main()
