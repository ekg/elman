#!/usr/bin/env python3
"""
Direct scaling benchmark runner.
Runs E1, E33, E42, Mamba2 at multiple scales with proper model handling.
"""

import subprocess
import time
import json
import os
import re
from datetime import datetime
from pathlib import Path

# Config
DATA_PATH = '/home/erikg/elman/data/pile.txt'
RESULTS_DIR = '/home/erikg/elman/scaling_results'

CONFIGS = {
    "50M": {
        "E1": {"dim": 1280, "depth": 6, "batch": 256, "level": 1},
        "E33": {"dim": 1440, "depth": 6, "batch": 192, "level": 33},  # Reduced batch to avoid CUDA bug
        "E42": {"dim": 1664, "depth": 6, "batch": 256, "level": 42},
        "Mamba2": {"dim": 1152, "depth": 6, "batch": 256},
    },
    "100M": {
        "E1": {"dim": 1824, "depth": 6, "batch": 160, "level": 1},
        "E33": {"dim": 2048, "depth": 6, "batch": 144, "level": 33},  # Reduced
        "E42": {"dim": 2352, "depth": 6, "batch": 160, "level": 42},
        "Mamba2": {"dim": 1664, "depth": 6, "batch": 160},
    },
    "250M": {
        "E1": {"dim": 2880, "depth": 6, "batch": 96, "level": 1},
        "E33": {"dim": 3200, "depth": 6, "batch": 80, "level": 33},  # Reduced
        "E42": {"dim": 3712, "depth": 6, "batch": 96, "level": 42},
        "Mamba2": {"dim": 2624, "depth": 6, "batch": 96},
    },
    "500M": {
        "E1": {"dim": 4096, "depth": 6, "batch": 64, "level": 1},
        "E33": {"dim": 4608, "depth": 6, "batch": 56, "level": 33},  # Reduced
        "E42": {"dim": 5376, "depth": 6, "batch": 64, "level": 42},
        "Mamba2": {"dim": 3712, "depth": 6, "batch": 64},
    },
    "1B": {
        "E1": {"dim": 5888, "depth": 6, "batch": 48, "level": 1},
        "E33": {"dim": 6400, "depth": 6, "batch": 40, "level": 33},  # Reduced
        "E42": {"dim": 7680, "depth": 6, "batch": 48, "level": 42},
        "Mamba2": {"dim": 5248, "depth": 6, "batch": 48},
    },
}


def run_training(model_name, cfg, gpu, scale, train_minutes=60):
    """Run single training job."""
    log_file = f'/tmp/scaling_{scale}_{model_name}.log'
    
    if model_name == 'Mamba2':
        # Use benchmark_baselines.py for Mamba2
        timeout_seconds = train_minutes * 60
        cmd = (
            f"CUDA_VISIBLE_DEVICES={gpu} python /home/erikg/elman/benchmark_baselines.py "
            f"--data {DATA_PATH} "
            f"--model mamba2 "
            f"--params {scale.lower()} "
            f"--batch_size {cfg['batch']} "
            f"--timeout {timeout_seconds} "
        )
    else:
        # Use train.py for Elman models
        cmd = (
            f"CUDA_VISIBLE_DEVICES={gpu} python /home/erikg/elman/train.py "
            f"--data {DATA_PATH} "
            f"--level={cfg['level']} "
            f"--dim={cfg['dim']} "
            f"--depth={cfg['depth']} "
            f"--batch_size={cfg['batch']} "
            f"--train_minutes={train_minutes} "
            f"--bf16 "
            f"--seed=42 "
        )
    
    print(f"  [{model_name}] Starting on GPU {gpu} with dim={cfg.get('dim', 'auto')}, batch={cfg['batch']}")
    print(f"    Log: {log_file}")
    
    with open(log_file, 'w') as f:
        proc = subprocess.Popen(
            cmd, shell=True, stdout=f, stderr=subprocess.STDOUT,
            cwd='/home/erikg/elman'
        )
    
    return proc, log_file


def extract_metrics(log_file):
    """Extract last-100 average loss and throughput from log."""
    if not os.path.exists(log_file):
        return None
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Pattern: step    123 | loss 1.4567 | lr 1.00e-04 | grad 0.50 | tok/s 123456
        pattern = re.compile(r'step\s+(\d+)\s+\|\s+loss\s+([\d.]+)\s+\|.*tok/s\s+([\d.]+)')
        
        entries = []
        for line in lines:
            m = pattern.search(line)
            if m:
                entries.append({
                    'step': int(m.group(1)),
                    'loss': float(m.group(2)),
                    'throughput': float(m.group(3)),
                })
        
        if not entries:
            return None
        
        # Last 100 average
        last_n = min(100, len(entries))
        last = entries[-last_n:]
        
        return {
            'loss': sum(e['loss'] for e in last) / len(last),
            'throughput': sum(e['throughput'] for e in last) / len(last),
            'final_step': entries[-1]['step'],
            'n_samples': last_n,
        }
    except Exception as e:
        print(f"Error extracting from {log_file}: {e}")
        return None


def run_scale(scale_name, train_minutes=60, gpu_offset=0):
    """Run all models for a scale."""
    config = CONFIGS[scale_name]
    models = ['E1', 'E33', 'E42', 'Mamba2']
    
    print(f"\n{'='*60}")
    print(f"Running {scale_name} scale on GPUs {gpu_offset}-{gpu_offset+3}")
    print(f"{'='*60}")
    
    procs = []
    for i, model in enumerate(models):
        gpu = gpu_offset + i
        cfg = config[model]
        proc, log_file = run_training(model, cfg, gpu, scale_name, train_minutes)
        procs.append((model, proc, log_file, cfg))
    
    # Wait for completion
    print(f"\n  Waiting for {scale_name} runs to complete ({train_minutes} min)...")
    for model, proc, log_file, cfg in procs:
        proc.wait()
        status = "OK" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
        print(f"  [{model}] {status}")
    
    # Collect results
    results = {}
    for model, proc, log_file, cfg in procs:
        metrics = extract_metrics(log_file)
        if metrics:
            results[model] = {
                'loss': metrics['loss'],
                'throughput': metrics['throughput'],
                'final_step': metrics['final_step'],
                'config': cfg,
                'exit_code': proc.returncode,
            }
            print(f"  [{model}] loss={metrics['loss']:.4f}, throughput={metrics['throughput']:.0f} tok/s")
        else:
            results[model] = {'error': 'Failed to extract metrics', 'config': cfg, 'exit_code': proc.returncode}
            print(f"  [{model}] FAILED to extract metrics")
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scales', type=str, default='50M,100M,250M,500M,1B',
                        help='Scales to run (comma-separated)')
    parser.add_argument('--train_minutes', type=int, default=60,
                        help='Training time per scale (default: 60)')
    parser.add_argument('--parallel', type=int, default=2,
                        help='Number of scales to run in parallel (1 or 2)')
    args = parser.parse_args()
    
    scales = [s.strip() for s in args.scales.split(',')]
    all_results = {}
    
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    
    print("="*60)
    print(f"Scaling Study: {', '.join(scales)}")
    print(f"Training time: {args.train_minutes} min per scale")
    print("="*60)
    
    if args.parallel == 2:
        # Run pairs in parallel
        import threading
        
        i = 0
        while i < len(scales):
            threads = []
            wave_results = {}
            
            def run_and_store(scale, gpu_offset):
                res = run_scale(scale, args.train_minutes, gpu_offset)
                wave_results[scale] = res
            
            # First scale on GPUs 0-3
            t1 = threading.Thread(target=run_and_store, args=(scales[i], 0))
            threads.append(t1)
            i += 1
            
            # Second scale on GPUs 4-7 (if available)
            if i < len(scales):
                t2 = threading.Thread(target=run_and_store, args=(scales[i], 4))
                threads.append(t2)
                i += 1
            
            # Start and wait
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            
            all_results.update(wave_results)
            
            # Save intermediate results
            with open(f'{RESULTS_DIR}/scaling_intermediate.json', 'w') as f:
                json.dump({'timestamp': datetime.now().isoformat(), 'results': all_results}, f, indent=2)
    else:
        # Sequential
        for scale in scales:
            all_results[scale] = run_scale(scale, args.train_minutes, 0)
    
    # Save final results
    output_file = f'{RESULTS_DIR}/scaling_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'train_minutes': args.train_minutes,
            'results': all_results,
        }, f, indent=2)
    
    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"{'Scale':<10} {'Model':<10} {'Loss':<10} {'Throughput':<15}")
    print("-"*50)
    
    for scale in scales:
        if scale not in all_results:
            continue
        for model in ['E1', 'E33', 'E42', 'Mamba2']:
            if model not in all_results[scale]:
                continue
            r = all_results[scale][model]
            if 'loss' in r:
                print(f"{scale:<10} {model:<10} {r['loss']:<10.4f} {r['throughput']:<15.0f}")
            else:
                print(f"{scale:<10} {model:<10} {'FAILED':<10}")
    
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
