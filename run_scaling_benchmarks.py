#!/usr/bin/env python3
"""
Scaling benchmark runner for E1, E33, E42, Mamba2.
Runs 1-hour training at each scale, 4 models in parallel.

Usage:
    python run_scaling_benchmarks.py

Configuration:
    Reads from scaling_config.py which should define SCALING_CONFIG dict.
    See scaling_config.py for expected format.

Output:
    Results saved to scaling_results.json with loss and throughput per model/scale.
"""

import subprocess
import json
import time
import os
import re
import sys
from pathlib import Path
from datetime import datetime

# Wait for config file
CONFIG_PATH = '/home/erikg/elman/scaling_config.py'
RESULTS_PATH = '/home/erikg/elman/scaling_results.json'
DATA_PATH = '/home/erikg/elman/data/pile.txt'


def wait_for_config():
    """Wait for scaling_config.py to exist."""
    while not os.path.exists(CONFIG_PATH):
        print(f"Waiting for {CONFIG_PATH}...")
        time.sleep(30)
    print(f"Found {CONFIG_PATH}")


def load_config():
    """Load SCALING_CONFIG from scaling_config.py."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("scaling_config", CONFIG_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SCALING_CONFIG


def extract_last100_metrics(log_file):
    """
    Extract Last-100 average loss and throughput from training log.

    Returns:
        dict with 'loss' and 'throughput' keys, or None if extraction fails
    """
    if not os.path.exists(log_file):
        return None

    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()

        # Find all log lines with loss and tok/s
        # Format: step    123 | loss 1.4567 | lr 1.00e-04 | grad 0.50 | tok/s 123456
        log_pattern = re.compile(r'step\s+(\d+)\s+\|\s+loss\s+([\d.]+)\s+\|.*tok/s\s+([\d.]+)')

        entries = []
        for line in lines:
            match = log_pattern.search(line)
            if match:
                step = int(match.group(1))
                loss = float(match.group(2))
                throughput = float(match.group(3))
                entries.append({'step': step, 'loss': loss, 'throughput': throughput})

        if len(entries) < 10:
            # Not enough data
            print(f"Warning: Only {len(entries)} entries in {log_file}")
            if entries:
                return {
                    'loss': entries[-1]['loss'],
                    'throughput': entries[-1]['throughput'],
                    'final_step': entries[-1]['step'],
                    'n_samples': len(entries),
                }
            return None

        # Take last 100 entries (or all if fewer)
        last_n = min(100, len(entries))
        last_entries = entries[-last_n:]

        avg_loss = sum(e['loss'] for e in last_entries) / len(last_entries)
        avg_throughput = sum(e['throughput'] for e in last_entries) / len(last_entries)
        final_step = entries[-1]['step']

        return {
            'loss': avg_loss,
            'throughput': avg_throughput,
            'final_step': final_step,
            'n_samples': last_n,
        }
    except Exception as e:
        print(f"Error extracting metrics from {log_file}: {e}")
        return None


def build_command(model, cfg, gpu):
    """
    Build training command for a given model and config.

    Args:
        model: Model name ('E1', 'E33', 'E42', 'Mamba2')
        cfg: Config dict with 'dim', 'depth', 'batch' keys
        gpu: GPU index to use

    Returns:
        Command string
    """
    dim = cfg['dim']
    depth = cfg['depth']
    batch = cfg['batch']

    if model == 'Mamba2':
        # Mamba2 uses mamba2 level
        cmd = (
            f"CUDA_VISIBLE_DEVICES={gpu} python /home/erikg/elman/train.py "
            f"--data {DATA_PATH} "
            f"--level=mamba2 "
            f"--dim={dim} "
            f"--depth={depth} "
            f"--batch_size={batch} "
            f"--train_minutes=60 "
            f"--bf16 "
            f"--seed=42"
        )
    else:
        # E-series models use integer level
        level_map = {'E1': 1, 'E33': 33, 'E42': 42}
        level = level_map[model]
        cmd = (
            f"CUDA_VISIBLE_DEVICES={gpu} python /home/erikg/elman/train.py "
            f"--data {DATA_PATH} "
            f"--level={level} "
            f"--dim={dim} "
            f"--depth={depth} "
            f"--batch_size={batch} "
            f"--train_minutes=60 "
            f"--bf16 "
            f"--seed=42"
        )

    return cmd


def run_scale(scale_name, config, gpu_offset=0):
    """
    Run all 4 models for a given scale on GPUs gpu_offset to gpu_offset+3.

    Args:
        scale_name: Scale identifier (e.g., '50M', '100M')
        config: Dict mapping model names to their configs
        gpu_offset: Starting GPU index (0 or 4)

    Returns:
        Dict mapping model names to their results
    """
    models = ['E1', 'E33', 'E42', 'Mamba2']
    processes = []

    print(f"\n{'='*60}")
    print(f"Starting scale {scale_name} on GPUs {gpu_offset}-{gpu_offset+3}")
    print(f"{'='*60}")

    for i, model in enumerate(models):
        cfg = config[model]
        gpu = gpu_offset + i
        log_file = f'/tmp/scaling_{model}_{scale_name}.log'

        cmd = build_command(model, cfg, gpu)

        print(f"  Launching {model} on GPU {gpu}: dim={cfg['dim']}, depth={cfg['depth']}, batch={cfg['batch']}")
        print(f"    Log: {log_file}")

        # Launch process
        with open(log_file, 'w') as f:
            p = subprocess.Popen(
                cmd,
                shell=True,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd='/home/erikg/elman'
            )

        processes.append((model, p, log_file, cfg))

    # Wait for all to complete
    print(f"\n  Waiting for {scale_name} runs to complete (1 hour)...")
    start_time = time.time()

    for model, p, log_file, cfg in processes:
        p.wait()
        elapsed = (time.time() - start_time) / 60
        if p.returncode == 0:
            print(f"  {model} completed successfully (elapsed: {elapsed:.1f} min)")
        else:
            print(f"  {model} FAILED with code {p.returncode} (elapsed: {elapsed:.1f} min)")

    # Extract results
    results = {}
    for model, p, log_file, cfg in processes:
        metrics = extract_last100_metrics(log_file)
        if metrics:
            results[model] = {
                'loss': metrics['loss'],
                'throughput': metrics['throughput'],
                'final_step': metrics['final_step'],
                'n_samples': metrics['n_samples'],
                'config': cfg,
                'exit_code': p.returncode,
            }
            print(f"  {model}: loss={metrics['loss']:.4f}, throughput={metrics['throughput']:.0f} tok/s")
        else:
            results[model] = {
                'loss': None,
                'throughput': None,
                'config': cfg,
                'exit_code': p.returncode,
                'error': 'Failed to extract metrics',
            }
            print(f"  {model}: FAILED to extract metrics")

    return results


def save_results(all_results):
    """Save results to JSON file."""
    output = {
        'timestamp': datetime.now().isoformat(),
        'config_path': CONFIG_PATH,
        'scales': all_results,
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {RESULTS_PATH}")


def main():
    """Main entry point."""
    print("Scaling Benchmark Runner")
    print("=" * 60)
    print(f"Config: {CONFIG_PATH}")
    print(f"Output: {RESULTS_PATH}")
    print(f"Data: {DATA_PATH}")
    print()

    # Wait for config
    wait_for_config()

    # Load config
    config = load_config()
    scales = list(config.keys())
    print(f"Scales to run: {scales}")

    all_results = {}

    # Organize scales into waves
    # Wave 1: 50M (GPUs 0-3) + 100M (GPUs 4-7) in parallel
    # Wave 2: 250M (GPUs 0-3) + 500M (GPUs 4-7) in parallel
    # Wave 3: 1B (GPUs 0-3)

    waves = []
    scale_order = ['50M', '100M', '250M', '500M', '1B']

    # Filter to only scales present in config
    available_scales = [s for s in scale_order if s in config]

    # Pair scales into waves
    i = 0
    while i < len(available_scales):
        wave = []
        # First scale on GPUs 0-3
        wave.append((available_scales[i], 0))
        i += 1
        # Second scale (if available) on GPUs 4-7
        if i < len(available_scales):
            wave.append((available_scales[i], 4))
            i += 1
        waves.append(wave)

    print(f"\nExecution plan ({len(waves)} waves):")
    for wave_idx, wave in enumerate(waves):
        wave_desc = ', '.join([f"{s[0]} (GPUs {s[1]}-{s[1]+3})" for s in wave])
        print(f"  Wave {wave_idx + 1}: {wave_desc}")

    # Execute waves
    for wave_idx, wave in enumerate(waves):
        print(f"\n{'#'*60}")
        print(f"WAVE {wave_idx + 1} of {len(waves)}")
        print(f"{'#'*60}")

        # Launch all scales in this wave in parallel
        wave_processes = []
        for scale_name, gpu_offset in wave:
            # For parallel execution, we spawn each scale's runs
            # But within run_scale, models run in parallel, so we need
            # to run scales sequentially if they share GPUs (they don't in our wave design)
            pass

        # Since scales in a wave use different GPUs (0-3 vs 4-7), we can
        # run them in parallel using subprocess
        if len(wave) == 1:
            # Single scale, run directly
            scale_name, gpu_offset = wave[0]
            results = run_scale(scale_name, config[scale_name], gpu_offset)
            all_results[scale_name] = results
        else:
            # Two scales in parallel - we need to handle this differently
            # Use threading to run both scale groups simultaneously
            import threading

            wave_results = {}
            threads = []

            def run_and_store(scale_name, gpu_offset):
                results = run_scale(scale_name, config[scale_name], gpu_offset)
                wave_results[scale_name] = results

            for scale_name, gpu_offset in wave:
                t = threading.Thread(target=run_and_store, args=(scale_name, gpu_offset))
                threads.append(t)
                t.start()

            # Wait for all threads to complete
            for t in threads:
                t.join()

            # Store results
            all_results.update(wave_results)

        # Save intermediate results after each wave
        save_results(all_results)

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Scale':<10} {'Model':<10} {'Loss':<10} {'Throughput':<15} {'Steps':<10}")
    print("-" * 55)

    for scale in scale_order:
        if scale not in all_results:
            continue
        for model in ['E1', 'E33', 'E42', 'Mamba2']:
            if model not in all_results[scale]:
                continue
            r = all_results[scale][model]
            if r.get('loss') is not None:
                print(f"{scale:<10} {model:<10} {r['loss']:<10.4f} {r['throughput']:<15.0f} {r.get('final_step', 'N/A')}")
            else:
                print(f"{scale:<10} {model:<10} {'FAILED':<10} {'N/A':<15}")

    print("\n" + "=" * 60)
    print(f"Results saved to: {RESULTS_PATH}")
    print("=" * 60)


if __name__ == '__main__':
    main()
