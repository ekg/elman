#!/usr/bin/env python3
"""
Simple GPU Job Queue - One job per GPU, no contention

Usage:
    # Add jobs to queue
    python gpu_queue.py add "python train.py --level mamba2 --dim 896 ..."
    python gpu_queue.py add "python train.py --level E75h4n32 --dim 1920 ..."

    # Or add from a jobs file (one command per line)
    python gpu_queue.py add-file jobs.txt

    # Start processing queue (blocks until done)
    python gpu_queue.py run

    # Check status
    python gpu_queue.py status

    # Clear queue
    python gpu_queue.py clear

Jobs are stored in ~/.gpu_queue/jobs.txt
Logs go to ~/.gpu_queue/logs/
"""

import os
import sys
import json
import time
import fcntl
import subprocess
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

QUEUE_DIR = Path.home() / '.gpu_queue'
JOBS_FILE = QUEUE_DIR / 'jobs.json'
LOGS_DIR = QUEUE_DIR / 'logs'
LOCK_FILE = QUEUE_DIR / '.lock'


def init_queue():
    """Initialize queue directory."""
    QUEUE_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    if not JOBS_FILE.exists():
        save_jobs([])


def load_jobs() -> List[Dict]:
    """Load jobs from queue file."""
    if not JOBS_FILE.exists():
        return []
    with open(JOBS_FILE) as f:
        return json.load(f)


def save_jobs(jobs: List[Dict]):
    """Save jobs to queue file."""
    with open(JOBS_FILE, 'w') as f:
        json.dump(jobs, f, indent=2)


def get_num_gpus() -> int:
    """Get number of available GPUs."""
    result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
    return len([l for l in result.stdout.strip().split('\n') if 'GPU' in l])


def add_job(cmd: str, name: Optional[str] = None):
    """Add a job to the queue."""
    init_queue()
    jobs = load_jobs()

    job_id = len(jobs) + 1
    if name is None:
        # Extract name from command if possible
        if '--level' in cmd:
            parts = cmd.split('--level')[1].split()
            name = parts[0] if parts else f'job_{job_id}'
        else:
            name = f'job_{job_id}'

    job = {
        'id': job_id,
        'name': name,
        'cmd': cmd,
        'status': 'pending',
        'gpu': None,
        'added': datetime.now().isoformat(),
        'started': None,
        'finished': None,
    }
    jobs.append(job)
    save_jobs(jobs)
    print(f"Added job {job_id}: {name}")


def add_jobs_from_file(filepath: str):
    """Add jobs from a file (one command per line)."""
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                add_job(line)


def run_job(job: Dict, gpu: int) -> Dict:
    """Run a single job on specified GPU."""
    job['gpu'] = gpu
    job['status'] = 'running'
    job['started'] = datetime.now().isoformat()

    log_file = LOGS_DIR / f"{job['name']}_{job['id']}.log"

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)

    print(f"[GPU {gpu}] Starting: {job['name']}")

    with open(log_file, 'w') as log:
        log.write(f"# Job: {job['name']}\n")
        log.write(f"# Command: {job['cmd']}\n")
        log.write(f"# GPU: {gpu}\n")
        log.write(f"# Started: {job['started']}\n")
        log.write(f"{'='*60}\n\n")
        log.flush()

        result = subprocess.run(
            job['cmd'],
            shell=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(Path(__file__).parent),  # Run from elman directory
        )

    job['finished'] = datetime.now().isoformat()
    job['status'] = 'completed' if result.returncode == 0 else f'failed ({result.returncode})'
    job['log'] = str(log_file)

    print(f"[GPU {gpu}] Finished: {job['name']} - {job['status']}")
    return job


def run_queue():
    """Process all pending jobs in queue."""
    init_queue()
    jobs = load_jobs()

    pending = [j for j in jobs if j['status'] == 'pending']
    if not pending:
        print("No pending jobs in queue.")
        return

    num_gpus = get_num_gpus()
    print(f"Processing {len(pending)} jobs on {num_gpus} GPUs")
    print("="*60)

    # Process in batches of num_gpus
    gpu_procs = {}  # gpu -> (process, job)
    job_idx = 0

    while job_idx < len(pending) or gpu_procs:
        # Start jobs on free GPUs
        for gpu in range(num_gpus):
            if gpu not in gpu_procs and job_idx < len(pending):
                job = pending[job_idx]
                job_idx += 1

                # Start job in subprocess
                proc = mp.Process(target=run_job_wrapper, args=(job, gpu))
                proc.start()
                gpu_procs[gpu] = (proc, job)

        # Check for completed jobs
        completed_gpus = []
        for gpu, (proc, job) in gpu_procs.items():
            if not proc.is_alive():
                proc.join()
                completed_gpus.append(gpu)

                # Update job status in main jobs list
                for j in jobs:
                    if j['id'] == job['id']:
                        j.update(job)
                        break
                save_jobs(jobs)

        for gpu in completed_gpus:
            del gpu_procs[gpu]

        if gpu_procs:
            time.sleep(1)

    print("="*60)
    print("All jobs completed!")
    show_status()


def run_job_wrapper(job: Dict, gpu: int):
    """Wrapper to run job and handle results."""
    result = run_job(job, gpu)
    # Job dict is modified in place


def show_status():
    """Show queue status."""
    init_queue()
    jobs = load_jobs()

    if not jobs:
        print("Queue is empty.")
        return

    pending = [j for j in jobs if j['status'] == 'pending']
    running = [j for j in jobs if j['status'] == 'running']
    completed = [j for j in jobs if j['status'] == 'completed']
    failed = [j for j in jobs if j['status'].startswith('failed')]

    print(f"Queue Status: {len(pending)} pending, {len(running)} running, "
          f"{len(completed)} completed, {len(failed)} failed")
    print()

    for job in jobs:
        gpu_str = f"GPU {job['gpu']}" if job['gpu'] is not None else "---"
        print(f"  [{job['status']:<12}] {gpu_str:<6} {job['name']}")


def clear_queue():
    """Clear all jobs from queue."""
    init_queue()
    save_jobs([])
    print("Queue cleared.")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]

    if cmd == 'add':
        if len(sys.argv) < 3:
            print("Usage: gpu_queue.py add 'command to run'")
            return
        add_job(' '.join(sys.argv[2:]))

    elif cmd == 'add-file':
        if len(sys.argv) < 3:
            print("Usage: gpu_queue.py add-file jobs.txt")
            return
        add_jobs_from_file(sys.argv[2])

    elif cmd == 'run':
        run_queue()

    elif cmd == 'status':
        show_status()

    elif cmd == 'clear':
        clear_queue()

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == '__main__':
    main()
