#!/usr/bin/env python3
"""
E28 vs E1 Benchmark (10 minutes each)

Uses the actual model implementations instead of inline code.
"""

import subprocess
import os
import sys
import time
from pathlib import Path

# Two configs: E1 (no conv) and E28 (with conv)
# Using E1's best config: d1280×6 at 50M params
CONFIGS = [
    ("e1", 1280, 6, 64, "e1_d1280_depth6"),
    ("e28", 1280, 6, 64, "e28_d1280_depth6"),
]

DATA_PATH = "/home/erikg/elman/data/pile.txt"
OUTPUT_DIR = Path("/home/erikg/elman/benchmark_results/e28_vs_e1_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

E1_SCRIPT = """
import sys; sys.path.insert(0, '/home/erikg/elman')
import os
os.environ['LD_LIBRARY_PATH'] = f"/home/erikg/.local/lib/python3.12/site-packages/torch/lib:{{os.environ.get('LD_LIBRARY_PATH', '')}}"

import torch
import torch.nn as nn
import numpy as np
import mmap
import time
from schedulefree import AdamWScheduleFree

from elman.models.mamba_gated_elman import MambaGatedElman

torch.manual_seed(42); np.random.seed(42)

dim = {dim}
depth = {depth}
batch_size = {batch_size}
seq_len = 512
time_limit = 300  # 5 min - E28 Python is slow

# Data
with open('{data_path}', 'rb') as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
data_len = len(mm)

def get_batch(buf, mm, data_len, batch_size, seq_len):
    pos = np.random.randint(0, data_len - seq_len - 1, size=batch_size)
    for j, p in enumerate(pos):
        buf[j] = np.frombuffer(mm[p:p+seq_len+1], dtype=np.uint8)
    return torch.from_numpy(buf.astype(np.int64)).cuda()

class ElmanLM(nn.Module):
    def __init__(self, vocab_size, dim, depth):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([MambaGatedElman(dim, expansion=1.0) for _ in range(depth)])
        self.norm = nn.RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        # Proper initialization
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, x, return_loss=False):
        if return_loss:
            targets = x[:, 1:].contiguous()
            x = x[:, :-1]

        h = self.embed(x)
        for layer in self.layers:
            out, _ = layer(h)
            h = h + out
        h = self.norm(h)
        logits = self.lm_head(h)

        if return_loss:
            loss = torch.nn.functional.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            return loss
        return logits

model = ElmanLM(256, dim, depth).cuda().bfloat16()
n_params = sum(p.numel() for p in model.parameters())
print(f'E1 D={{dim}} depth={{depth}} batch={{batch_size}}: params={{n_params:,}}', flush=True)

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

E28_SCRIPT = """
import sys; sys.path.insert(0, '/home/erikg/elman')
import os
os.environ['LD_LIBRARY_PATH'] = f"/home/erikg/.local/lib/python3.12/site-packages/torch/lib:{{os.environ.get('LD_LIBRARY_PATH', '')}}"

import torch
import torch.nn as nn
import numpy as np
import mmap
import time
from schedulefree import AdamWScheduleFree

from elman.models.e28_conv_elman import E28ConvElman

torch.manual_seed(42); np.random.seed(42)

dim = {dim}
depth = {depth}
batch_size = {batch_size}
seq_len = 512
time_limit = 300  # 5 min - E28 Python is slow

# Data
with open('{data_path}', 'rb') as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
data_len = len(mm)

def get_batch(buf, mm, data_len, batch_size, seq_len):
    pos = np.random.randint(0, data_len - seq_len - 1, size=batch_size)
    for j, p in enumerate(pos):
        buf[j] = np.frombuffer(mm[p:p+seq_len+1], dtype=np.uint8)
    return torch.from_numpy(buf.astype(np.int64)).cuda()

class ElmanLM(nn.Module):
    def __init__(self, vocab_size, dim, depth):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([E28ConvElman(dim, expansion=1.0, d_conv=4) for _ in range(depth)])
        self.norm = nn.RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        # Proper initialization
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, x, return_loss=False):
        if return_loss:
            targets = x[:, 1:].contiguous()
            x = x[:, :-1]

        h = self.embed(x)
        for layer in self.layers:
            out, _ = layer(h)
            h = h + out
        h = self.norm(h)
        logits = self.lm_head(h)

        if return_loss:
            loss = torch.nn.functional.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            return loss
        return logits

model = ElmanLM(256, dim, depth).cuda().bfloat16()
n_params = sum(p.numel() for p in model.parameters())
print(f'E28 D={{dim}} depth={{depth}} batch={{batch_size}}: params={{n_params:,}}', flush=True)

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


def run_config(gpu_id, model_type, dim, depth, batch_size, name):
    """Run a single config on a specific GPU."""
    log_file = OUTPUT_DIR / f"{name}.log"
    script_file = OUTPUT_DIR / f"{name}.py"

    if model_type == "e1":
        script = E1_SCRIPT.format(dim=dim, depth=depth, batch_size=batch_size, data_path=DATA_PATH)
    else:
        script = E28_SCRIPT.format(dim=dim, depth=depth, batch_size=batch_size, data_path=DATA_PATH)

    with open(script_file, "w") as f:
        f.write(script)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["LD_LIBRARY_PATH"] = f"/home/erikg/.local/lib/python3.12/site-packages/torch/lib:{env.get('LD_LIBRARY_PATH', '')}"

    print(f"[GPU {gpu_id}] Starting {name}: D={dim}, depth={depth}, batch={batch_size}")

    with open(log_file, "w") as f:
        f.write(f"Config: {model_type.upper()} D={dim}, depth={depth}, batch={batch_size}\n")
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
    print("E28 vs E1 Benchmark v2 (10 minutes each)")
    print("=" * 70)
    print(f"Running {len(CONFIGS)} configs in parallel")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Launch all processes
    processes = []
    for gpu_id, (model_type, dim, depth, batch_size, name) in enumerate(CONFIGS):
        proc, name, log_file = run_config(gpu_id, model_type, dim, depth, batch_size, name)
        processes.append((proc, name, log_file, model_type, dim, depth, batch_size))

    print(f"\nLaunched {len(processes)} experiments. Waiting for completion...")
    print(f"Check progress with: tail -f {OUTPUT_DIR}/*.log")
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
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    print(f"{'Config':<20} | {'Model':>5} | {'Params':>8} | {'Loss':>8} | {'Throughput':>12}")
    print("-" * 70)

    results = []
    for proc, name, log_file, model_type, dim, depth, batch_size in processes:
        loss, throughput, params = extract_results(log_file)
        results.append({
            "name": name,
            "model_type": model_type,
            "dim": dim,
            "depth": depth,
            "params": params,
            "loss": loss,
            "throughput": throughput,
        })

        loss_str = f"{loss:.4f}" if loss else "N/A"
        tp_str = f"{throughput/1000:.1f}K tok/s" if throughput else "N/A"
        params_str = f"{params:.1f}M" if params else "N/A"

        print(f"{name:<20} | {model_type.upper():>5} | {params_str:>8} | {loss_str:>8} | {tp_str:>12}")

    # Compare
    print("\n" + "=" * 70)
    print("Comparison")
    print("=" * 70)
    e1_result = next((r for r in results if r["model_type"] == "e1"), None)
    e28_result = next((r for r in results if r["model_type"] == "e28"), None)

    if e1_result and e28_result and e1_result["loss"] and e28_result["loss"]:
        loss_diff = e28_result["loss"] - e1_result["loss"]
        tp_ratio = e28_result["throughput"] / e1_result["throughput"] if e1_result["throughput"] else 0

        print(f"E1:  loss={e1_result['loss']:.4f}, throughput={e1_result['throughput']/1000:.1f}K tok/s")
        print(f"E28: loss={e28_result['loss']:.4f}, throughput={e28_result['throughput']/1000:.1f}K tok/s")
        print(f"")
        print(f"Loss difference: {loss_diff:+.4f} ({'E28 better' if loss_diff < 0 else 'E1 better'})")
        print(f"Throughput ratio: {tp_ratio:.2f}x ({'E28 faster' if tp_ratio > 1 else 'E1 faster'})")

    # Save results
    import json
    results_file = OUTPUT_DIR / "summary.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
