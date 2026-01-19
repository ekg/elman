#!/usr/bin/env python3
"""
Simple GPU Job Scheduler

Reads commands from a file (one per line), runs them one-per-GPU.
Time-based training preserved - slow jobs time out naturally.

Usage:
    # Create jobs file
    cat > jobs.txt << 'EOF'
    python train.py --level mamba2 --dim 896 --train_minutes 10 --output out/mamba2 ...
    python train.py --level E75h8n24 --dim 1792 --train_minutes 10 --output out/e75 ...
    EOF

    # Run all jobs (8 GPUs, one job per GPU)
    python sched.py jobs.txt

    # Run with fewer GPUs
    python sched.py jobs.txt --gpus 4

    # Dry run (show what would run)
    python sched.py jobs.txt --dry
"""

import os
import sys
import subprocess
import time
import argparse
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def get_gpu_count():
    """Get number of available GPUs."""
    try:
        result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
        return len([l for l in result.stdout.strip().split('\n') if 'GPU' in l])
    except:
        return 0


def run_job(job_id: int, cmd: str, gpu: int, log_dir: Path) -> dict:
    """Run a single job on specified GPU."""
    # Extract job name from command (look for --output or --level)
    name = f"job_{job_id}"
    if '--output' in cmd:
        try:
            name = cmd.split('--output')[1].split()[0].split('/')[-1]
        except:
            pass
    elif '--level' in cmd:
        try:
            name = cmd.split('--level')[1].split()[0]
        except:
            pass

    log_file = log_dir / f"{job_id:02d}_{name}.log"

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)

    start = time.time()
    print(f"[GPU {gpu}] START: {name}")

    with open(log_file, 'w') as f:
        f.write(f"# Job {job_id}: {name}\n")
        f.write(f"# GPU: {gpu}\n")
        f.write(f"# Command: {cmd}\n")
        f.write(f"# Started: {datetime.now().isoformat()}\n")
        f.write("=" * 60 + "\n\n")
        f.flush()

        result = subprocess.run(
            cmd,
            shell=True,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
        )

    elapsed = time.time() - start
    status = "OK" if result.returncode == 0 else f"FAIL({result.returncode})"
    print(f"[GPU {gpu}] {status}: {name} ({elapsed/60:.1f} min)")

    return {
        'job_id': job_id,
        'name': name,
        'gpu': gpu,
        'status': status,
        'elapsed': elapsed,
        'log': str(log_file),
    }


def main():
    parser = argparse.ArgumentParser(description='Simple GPU job scheduler')
    parser.add_argument('jobs_file', help='File with commands (one per line)')
    parser.add_argument('--gpus', type=int, default=None, help='Number of GPUs to use')
    parser.add_argument('--dry', action='store_true', help='Dry run - show jobs without running')
    parser.add_argument('--log-dir', type=str, default=None, help='Log directory')
    args = parser.parse_args()

    # Read jobs
    jobs = []
    with open(args.jobs_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                jobs.append(line)

    if not jobs:
        print("No jobs found in file.")
        return

    # Get GPU count
    num_gpus = args.gpus or get_gpu_count()
    if num_gpus == 0:
        print("No GPUs available!")
        return

    # Setup logging
    if args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = Path(f'sched_logs/{timestamp}')
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Jobs: {len(jobs)}")
    print(f"GPUs: {num_gpus}")
    print(f"Logs: {log_dir}")
    print("=" * 60)

    if args.dry:
        for i, cmd in enumerate(jobs):
            gpu = i % num_gpus
            print(f"[GPU {gpu}] {cmd[:80]}...")
        return

    # Run jobs with process pool (one per GPU)
    results = []
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = {}
        job_queue = list(enumerate(jobs))
        running = 0

        while job_queue or futures:
            # Submit new jobs to free GPUs
            while job_queue and running < num_gpus:
                job_id, cmd = job_queue.pop(0)
                gpu = job_id % num_gpus

                # Wait if this GPU is busy
                busy_gpus = {f.result()['gpu'] for f in futures if f.done()} if futures else set()
                while gpu in [futures[f]['gpu'] for f in futures if not f.done()]:
                    gpu = (gpu + 1) % num_gpus

                future = executor.submit(run_job, job_id, cmd, gpu, log_dir)
                futures[future] = {'gpu': gpu, 'job_id': job_id}
                running += 1

            # Wait for at least one job to complete
            if futures:
                done = next(as_completed(futures))
                result = done.result()
                results.append(result)
                del futures[done]
                running -= 1

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    ok = [r for r in results if r['status'] == 'OK']
    failed = [r for r in results if r['status'] != 'OK']

    print(f"Completed: {len(ok)}/{len(results)}")
    if failed:
        print(f"Failed: {[r['name'] for r in failed]}")

    total_time = sum(r['elapsed'] for r in results)
    print(f"Total GPU-time: {total_time/60:.1f} min")
    print(f"Logs in: {log_dir}")


if __name__ == '__main__':
    main()
