#!/usr/bin/env python3
"""
E23 Configuration Sweep: Explore D vs N tradeoffs

Hypothesis: Smaller D with larger N (tape slots) may give better speed/quality tradeoff
- Smaller D = faster per-step computation
- Larger N = more tape memory capacity

Run 8 configs in parallel on 8 GPUs for 10 minutes each.
"""

import subprocess
import os
import sys
import time
from pathlib import Path

# Sweep configurations: (dim, n_slots, depth, name)
# Target ~50M params each. Formula: params ≈ depth * 6 * dim^2 + 512 * dim
# We manually tune depth to hit ~50M
CONFIGS = [
    # Smaller D, larger N - the hypothesis
    (384, 64, 56, "d384_n64"),   # 384^2 * 6 * 56 ≈ 49.5M
    (384, 128, 56, "d384_n128"), # Same params, more slots
    (448, 64, 40, "d448_n64"),   # 448^2 * 6 * 40 ≈ 48.2M
    (512, 64, 32, "d512_n64"),   # 512^2 * 6 * 32 ≈ 50.3M

    # Baseline configs for comparison
    (512, 32, 32, "d512_n32"),   # Same dim, fewer slots
    (512, 16, 32, "d512_n16"),   # E23 with minimal tape
    (640, 32, 20, "d640_n32"),   # Larger dim
    (768, 16, 14, "d768_n16"),   # Like E1 but with small tape
]

DATA_PATH = "/home/erikg/elman/data/pile.txt"
OUTPUT_DIR = Path("/home/erikg/elman/benchmark_results/e23_sweep")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Create the worker script
WORKER_SCRIPT = """
import sys; sys.path.insert(0, '/home/erikg/elman')
import torch, numpy as np, mmap, time
from schedulefree import AdamWScheduleFree
from elman.models import LadderLM

torch.manual_seed(42); np.random.seed(42)

dim = {dim}
n_slots = {n_slots}
depth = {depth}
batch_size = 64
seq_len = 512
time_limit = 600  # 10 minutes

# Data setup
with open('{data_path}', 'rb') as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
data_len = len(mm)

def get_batch(buf, mm, data_len, batch_size, seq_len):
    pos = np.random.randint(0, data_len - seq_len - 1, size=batch_size)
    for j, p in enumerate(pos):
        buf[j] = np.frombuffer(mm[p:p+seq_len+1], dtype=np.uint8)
    return torch.from_numpy(buf.astype(np.int64)).cuda()

# Create E23 model
model = LadderLM(vocab_size=256, dim=dim, depth=depth, level=23, n_slots=n_slots)
model = model.cuda().bfloat16()
n_params = model.get_num_params()
print(f'E23 D={{dim}} N={{n_slots}} depth={{depth}}: params={{n_params:,}}', flush=True)

opt = AdamWScheduleFree(model.parameters(), lr=3e-4, weight_decay=0.1)
model.train(); opt.train()
buf = np.zeros((batch_size, seq_len+1), dtype=np.uint8)
losses = []; start = time.time(); step = 0

while time.time() - start < time_limit:
    step += 1
    batch = get_batch(buf, mm, data_len, batch_size, seq_len)
    opt.zero_grad()
    loss = model(batch, return_loss=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    losses.append(loss.item())
    if step % 50 == 0:
        elapsed = time.time() - start
        tokens = step * batch_size * seq_len
        avg100 = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
        print(f'Step {{step}} | {{elapsed:.0f}}s | Loss {{loss.item():.4f}} | Avg100 {{avg100:.4f}} | {{int(tokens/elapsed)/1000:.1f}}K tok/s', flush=True)

elapsed = time.time() - start
tokens = step * batch_size * seq_len
avg_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
tps = int(tokens/elapsed)
print(f'FINAL: steps={{step}}, params={{n_params/1e6:.1f}}M, loss={{avg_loss:.4f}}, tok/s={{tps/1000:.1f}}K', flush=True)
mm.close()
"""

def run_config(gpu_id, dim, n_slots, depth, name):
    """Run a single config on a specific GPU."""
    log_file = OUTPUT_DIR / f"{name}.log"
    script_file = OUTPUT_DIR / f"{name}.py"

    # Write worker script
    script = WORKER_SCRIPT.format(
        dim=dim,
        n_slots=n_slots,
        depth=depth,
        data_path=DATA_PATH
    )
    with open(script_file, "w") as f:
        f.write(script)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[GPU {gpu_id}] Starting {name}: D={dim}, N={n_slots}, depth={depth}")

    with open(log_file, "w") as f:
        f.write(f"Config: D={dim}, N={n_slots}, depth={depth}\n")
        f.write("=" * 60 + "\n\n")
        f.flush()

        proc = subprocess.Popen(
            ["python", str(script_file)],
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            cwd="/home/erikg/elman"
        )

    return proc, name, log_file

def extract_results(log_file):
    """Extract final loss and throughput from log file."""
    try:
        with open(log_file, "r") as f:
            lines = f.readlines()

        # Find FINAL line
        loss = None
        throughput = None
        params = None

        for line in lines:
            if "FINAL:" in line:
                import re
                match = re.search(r'loss=(\d+\.\d+)', line)
                if match:
                    loss = float(match.group(1))
                match = re.search(r'tok/s=(\d+\.?\d*)K', line)
                if match:
                    throughput = float(match.group(1)) * 1000
                match = re.search(r'params=(\d+\.?\d*)M', line)
                if match:
                    params = float(match.group(1))

        return loss, throughput, params
    except Exception as e:
        return None, None, None

def main():
    print("=" * 70)
    print("E23 Configuration Sweep")
    print("=" * 70)
    print(f"Running {len(CONFIGS)} configs in parallel for 10 minutes each")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Launch all processes
    processes = []
    for gpu_id, (dim, n_slots, depth, name) in enumerate(CONFIGS):
        if gpu_id >= 8:
            print(f"Warning: Only 8 GPUs available, skipping {name}")
            continue
        proc, name, log_file = run_config(gpu_id, dim, n_slots, depth, name)
        processes.append((proc, name, log_file, dim, n_slots, depth))

    print(f"\nLaunched {len(processes)} experiments. Waiting for completion...")
    print("Check progress with: tail -f benchmark_results/e23_sweep/*.log")
    print()

    # Wait for all to complete
    start_time = time.time()
    while True:
        running = sum(1 for p, _, _, _, _, _ in processes if p.poll() is None)
        if running == 0:
            break
        elapsed = time.time() - start_time
        print(f"\r[{elapsed/60:.1f}min] {running}/{len(processes)} still running...", end="", flush=True)
        time.sleep(10)

    print(f"\n\nAll experiments complete! Total time: {(time.time() - start_time)/60:.1f} minutes")

    # Collect results
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    print(f"{'Config':<15} | {'D':>5} | {'N':>4} | {'Depth':>5} | {'Params':>8} | {'Loss':>8} | {'Throughput':>12}")
    print("-" * 70)

    results = []
    for proc, name, log_file, dim, n_slots, depth in processes:
        loss, throughput, params = extract_results(log_file)
        results.append({
            "name": name,
            "dim": dim,
            "n_slots": n_slots,
            "depth": depth,
            "params": params,
            "loss": loss,
            "throughput": throughput,
        })

        loss_str = f"{loss:.4f}" if loss else "N/A"
        tp_str = f"{throughput/1000:.1f}K tok/s" if throughput else "N/A"
        params_str = f"{params:.1f}M" if params else "N/A"

        print(f"{name:<15} | {dim:>5} | {n_slots:>4} | {depth:>5} | {params_str:>8} | {loss_str:>8} | {tp_str:>12}")

    # Sort by loss (best first)
    print("\n" + "=" * 70)
    print("Ranked by Loss (best first)")
    print("=" * 70)
    valid_results = [r for r in results if r["loss"] is not None]
    for i, r in enumerate(sorted(valid_results, key=lambda x: x["loss"]), 1):
        tp_str = f"{r['throughput']/1000:.1f}K" if r["throughput"] else "N/A"
        print(f"{i}. {r['name']:<15} loss={r['loss']:.4f}  throughput={tp_str} tok/s")

    # Save results
    import json
    results_file = OUTPUT_DIR / "summary.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    main()
