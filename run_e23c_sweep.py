#!/usr/bin/env python3
"""
E23c (Chunked Dual-Memory) Training Sweep

Uses the same configurations that worked for E23:
- d512_n32: best loss (1.998)
- d512_n16: fast, good loss
- d768_n64: large tape
- d512_n64: standard
- d512_n8: minimal tape, fastest
- d768_n8: E1-like + small tape
- d1024_n64: high capacity

Each config runs for 10 minutes, all in parallel on different GPUs.
"""

import subprocess
import os
import sys
import time
from pathlib import Path

# Configurations from E23 sweep that worked well
# (dim, n_slots, depth, batch_size, chunk_size, name)
# E23c should be ~3x faster than E23, so we can use larger batch sizes
CONFIGS = [
    # Best E23 configs - now with E23c can use larger batches
    (512, 32, 32, 96, 64, "d512_n32"),   # Best loss in E23
    (512, 16, 32, 128, 64, "d512_n16"),  # Good balance
    (768, 64, 14, 64, 64, "d768_n64"),   # Large tape
    (512, 64, 32, 64, 64, "d512_n64"),   # Standard large tape
    (512, 8, 32, 128, 64, "d512_n8"),    # Minimal tape, fastest
    (768, 8, 14, 128, 64, "d768_n8"),    # E1-like + tiny tape
    (1024, 64, 8, 32, 64, "d1024_n64"),  # High capacity
    # Additional: test different chunk sizes
    (512, 32, 32, 96, 128, "d512_n32_k128"),  # Larger chunk
]

DATA_PATH = "/home/erikg/elman/data/pile.txt"
OUTPUT_DIR = Path("/home/erikg/elman/benchmark_results/e23c_sweep")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Worker script for E23c
WORKER_SCRIPT = """
import sys; sys.path.insert(0, '/home/erikg/elman')
import os
os.environ['LD_LIBRARY_PATH'] = f"/home/erikg/.local/lib/python3.12/site-packages/torch/lib:{{os.environ.get('LD_LIBRARY_PATH', '')}}"

import torch
import torch.nn as nn
import numpy as np
import mmap
import time
from schedulefree import AdamWScheduleFree

# Import CUDA kernel
sys.path.insert(0, '/home/erikg/elman/elman/cuda')
import hasty_pytorch_lib

torch.manual_seed(42); np.random.seed(42)

dim = {dim}
n_slots = {n_slots}
depth = {depth}
batch_size = {batch_size}
chunk_size = {chunk_size}
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


class E23cCell(nn.Module):
    \"\"\"E23c: Chunked Dual-Memory Elman\"\"\"
    def __init__(self, dim, n_slots=64, chunk_size=64, w_h_init_scale=0.9):
        super().__init__()
        self.dim = dim
        self.n_slots = n_slots
        self.chunk_size = chunk_size

        self.W_h = nn.Linear(dim, dim, bias=False)
        self.W_x = nn.Linear(dim, dim, bias=False)
        self.b_h = nn.Parameter(torch.zeros(dim))
        self.W_write = nn.Linear(dim, dim, bias=False)

        with torch.no_grad():
            nn.init.orthogonal_(self.W_h.weight)
            self.W_h.weight.mul_(w_h_init_scale)
            nn.init.xavier_uniform_(self.W_x.weight)
            nn.init.xavier_uniform_(self.W_write.weight)

    def forward(self, x_seq, h_tape=None, h_work=None):
        B, T, D = x_seq.shape
        device = x_seq.device
        dtype = x_seq.dtype

        if h_tape is None:
            h_tape = torch.zeros(B, self.n_slots, D, device=device, dtype=dtype)
        if h_work is None:
            h_work = torch.zeros(B, D, device=device, dtype=dtype)

        results = hasty_pytorch_lib.e23c_chunked_forward(
            self.training,
            x_seq,
            h_tape,
            h_work,
            self.W_h.weight,
            self.W_x.weight,
            self.b_h,
            self.W_write.weight,
            self.chunk_size
        )

        return results[0], results[1], results[2][:, -1]


class E23cLayer(nn.Module):
    \"\"\"E23c layer with Mamba2-style input/output projections.\"\"\"
    def __init__(self, dim, expansion=1.0, n_slots=64, chunk_size=64):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.n_slots = n_slots
        self.chunk_size = chunk_size

        self.in_proj = nn.Linear(dim, 2 * self.d_inner, bias=False)
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)
        self.cell = E23cCell(self.d_inner, n_slots, chunk_size)

    def forward(self, x, hidden=None):
        B, T, _ = x.shape
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)
        x_proj = torch.nn.functional.silu(x_proj)

        if hidden is not None:
            h_tape, h_work = hidden
        else:
            h_tape, h_work = None, None

        output, h_tape_final, h_work_final = self.cell(x_proj, h_tape, h_work)
        output = output * torch.nn.functional.silu(z)
        output = self.out_proj(output)
        return output, (h_tape_final, h_work_final)


class E23cLM(nn.Module):
    \"\"\"E23c Language Model.\"\"\"
    def __init__(self, vocab_size, dim, depth, n_slots=64, chunk_size=64, expansion=1.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.depth = depth
        self.n_slots = n_slots
        self.chunk_size = chunk_size

        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            E23cLayer(dim, expansion, n_slots, chunk_size)
            for _ in range(depth)
        ])
        self.norm = nn.RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embed.weight

    def forward(self, x, return_loss=False):
        if return_loss:
            inputs = x[:, :-1]
            targets = x[:, 1:]
        else:
            inputs = x

        h = self.embed(inputs)
        for layer in self.layers:
            out, _ = layer(h)
            h = h + out
        h = self.norm(h)
        logits = self.lm_head(h)

        if return_loss:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.reshape(-1)
            )
            return loss
        return logits

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Create model
model = E23cLM(
    vocab_size=256,
    dim=dim,
    depth=depth,
    n_slots=n_slots,
    chunk_size=chunk_size,
    expansion=1.0
)
model = model.cuda().bfloat16()
n_params = model.get_num_params()
print(f'E23c D={{dim}} N={{n_slots}} depth={{depth}} K={{chunk_size}} batch={{batch_size}}: params={{n_params:,}}', flush=True)

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


def run_config(gpu_id, dim, n_slots, depth, batch_size, chunk_size, name):
    """Run a single config on a specific GPU."""
    log_file = OUTPUT_DIR / f"{name}.log"
    script_file = OUTPUT_DIR / f"{name}.py"

    script = WORKER_SCRIPT.format(
        dim=dim,
        n_slots=n_slots,
        depth=depth,
        batch_size=batch_size,
        chunk_size=chunk_size,
        data_path=DATA_PATH
    )
    with open(script_file, "w") as f:
        f.write(script)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["LD_LIBRARY_PATH"] = f"/home/erikg/.local/lib/python3.12/site-packages/torch/lib:{env.get('LD_LIBRARY_PATH', '')}"

    print(f"[GPU {gpu_id}] Starting {name}: D={dim}, N={n_slots}, depth={depth}, K={chunk_size}, batch={batch_size}")

    with open(log_file, "w") as f:
        f.write(f"Config: D={dim}, N={n_slots}, depth={depth}, K={chunk_size}, batch={batch_size}\n")
        f.write("=" * 60 + "\n\n")
        f.flush()

        proc = subprocess.Popen(
            ["python", str(script_file)],
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            cwd="/home/erikg/elman/elman/cuda"
        )

    return proc, name, log_file


def extract_results(log_file):
    """Extract final loss and throughput from log file."""
    try:
        with open(log_file, "r") as f:
            lines = f.readlines()

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
    print("E23c (Chunked Dual-Memory) Configuration Sweep")
    print("=" * 70)
    print(f"Running {len(CONFIGS)} configs in parallel for 10 minutes each")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Launch all processes
    processes = []
    for gpu_id, (dim, n_slots, depth, batch_size, chunk_size, name) in enumerate(CONFIGS):
        if gpu_id >= 8:
            print(f"Warning: Only 8 GPUs available, skipping {name}")
            continue
        proc, name, log_file = run_config(gpu_id, dim, n_slots, depth, batch_size, chunk_size, name)
        processes.append((proc, name, log_file, dim, n_slots, depth, batch_size, chunk_size))

    print(f"\nLaunched {len(processes)} experiments. Waiting for completion...")
    print("Check progress with: tail -f benchmark_results/e23c_sweep/*.log")
    print()

    # Wait for all to complete
    start_time = time.time()
    while True:
        running = sum(1 for p, *_ in processes if p.poll() is None)
        if running == 0:
            break
        elapsed = time.time() - start_time
        print(f"\r[{elapsed/60:.1f}min] {running}/{len(processes)} still running...", end="", flush=True)
        time.sleep(10)

    print(f"\n\nAll experiments complete! Total time: {(time.time() - start_time)/60:.1f} minutes")

    # Collect results
    print("\n" + "=" * 90)
    print("Results Summary")
    print("=" * 90)
    print(f"{'Config':<16} | {'D':>5} | {'N':>3} | {'Depth':>5} | {'K':>4} | {'Batch':>5} | {'Params':>8} | {'Loss':>8} | {'Throughput':>12}")
    print("-" * 90)

    results = []
    for proc, name, log_file, dim, n_slots, depth, batch_size, chunk_size in processes:
        loss, throughput, params = extract_results(log_file)
        results.append({
            "name": name,
            "dim": dim,
            "n_slots": n_slots,
            "depth": depth,
            "batch_size": batch_size,
            "chunk_size": chunk_size,
            "params": params,
            "loss": loss,
            "throughput": throughput,
        })

        loss_str = f"{loss:.4f}" if loss else "N/A"
        tp_str = f"{throughput/1000:.1f}K tok/s" if throughput else "N/A"
        params_str = f"{params:.1f}M" if params else "N/A"

        print(f"{name:<16} | {dim:>5} | {n_slots:>3} | {depth:>5} | {chunk_size:>4} | {batch_size:>5} | {params_str:>8} | {loss_str:>8} | {tp_str:>12}")

    # Sort by loss
    print("\n" + "=" * 90)
    print("Ranked by Loss (best first)")
    print("=" * 90)
    valid_results = [r for r in results if r["loss"] is not None]
    for i, r in enumerate(sorted(valid_results, key=lambda x: x["loss"]), 1):
        tp_str = f"{r['throughput']/1000:.1f}K" if r["throughput"] else "N/A"
        print(f"{i}. {r['name']:<16} loss={r['loss']:.4f}  throughput={tp_str} tok/s  (D={r['dim']}, N={r['n_slots']})")

    # Sort by throughput
    print("\n" + "=" * 90)
    print("Ranked by Throughput (fastest first)")
    print("=" * 90)
    for i, r in enumerate(sorted(valid_results, key=lambda x: -x["throughput"] if x["throughput"] else 0), 1):
        tp_str = f"{r['throughput']/1000:.1f}K" if r["throughput"] else "N/A"
        print(f"{i}. {r['name']:<16} throughput={tp_str} tok/s  loss={r['loss']:.4f}  (D={r['dim']}, N={r['n_slots']})")

    # Compare with E23 results
    print("\n" + "=" * 90)
    print("Comparison with E23 (from previous sweep)")
    print("=" * 90)
    e23_results = {
        "d512_n32": (1.998, 26.2),
        "d512_n16": (2.020, 32.3),
        "d768_n64": (2.032, 16.2),
        "d512_n64": (2.094, 10.1),
        "d512_n8": (2.087, 37.0),
        "d768_n8": (2.123, 61.2),
        "d1024_n64": (2.243, 11.8),
    }
    print(f"{'Config':<16} | {'E23 Loss':>10} | {'E23c Loss':>10} | {'E23 tok/s':>12} | {'E23c tok/s':>12} | {'Speedup':>8}")
    print("-" * 80)
    for r in valid_results:
        base_name = r['name'].split('_k')[0]  # Remove chunk size suffix if present
        if base_name in e23_results:
            e23_loss, e23_tp = e23_results[base_name]
            e23c_loss = r['loss']
            e23c_tp = r['throughput'] / 1000 if r['throughput'] else 0
            speedup = e23c_tp / e23_tp if e23_tp > 0 else 0
            print(f"{r['name']:<16} | {e23_loss:>10.4f} | {e23c_loss:>10.4f} | {e23_tp:>10.1f}K | {e23c_tp:>10.1f}K | {speedup:>7.2f}x")

    # Save results
    import json
    results_file = OUTPUT_DIR / "summary.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
