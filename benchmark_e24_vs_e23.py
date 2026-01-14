#!/usr/bin/env python3
"""
Compare E24 vs E23 throughput.
"""
import torch
import time
import sys
sys.path.insert(0, 'elman/cuda')

from elman.models.e24_single_gemm import E24Cell
from elman.models.dual_memory_elman import DualMemoryElmanCell  # E23

device = 'cuda'
dtype = torch.bfloat16

print("=" * 70)
print("E24 vs E23 Throughput Comparison")
print("=" * 70)

# Benchmark parameters - same configs for both
configs = [
    {"B": 4, "T": 512, "D": 256, "N": 16},
    {"B": 4, "T": 512, "D": 512, "N": 16},
    {"B": 8, "T": 512, "D": 512, "N": 16},
    {"B": 4, "T": 512, "D": 256, "N": 64},
    {"B": 4, "T": 512, "D": 512, "N": 64},
]

warmup_iters = 5
bench_iters = 20

results = []

for cfg in configs:
    B, T, D, N = cfg["B"], cfg["T"], cfg["D"], cfg["N"]
    print(f"\nConfig: B={B}, T={T}, D={D}, N={N}")
    print("-" * 50)

    # Create cells
    e24_cell = E24Cell(dim=D, n_slots=N).to(device).to(dtype)
    e23_cell = DualMemoryElmanCell(dim=D, n_slots=N).to(device).to(dtype)

    # Create input
    x = torch.randn(B, T, D, device=device, dtype=dtype)
    h_tape = torch.zeros(B, N, D, device=device, dtype=dtype)
    h_work = torch.zeros(B, D, device=device, dtype=dtype)

    # Count params
    e24_params = sum(p.numel() for p in e24_cell.parameters())
    e23_params = sum(p.numel() for p in e23_cell.parameters())

    # ===== E24 Benchmark =====
    # Warmup
    for _ in range(warmup_iters):
        x_tmp = x.detach().clone().requires_grad_(True)
        e24_cell.zero_grad()
        h_out, _, _ = e24_cell(x_tmp, h_tape.clone(), h_work.clone(), use_cuda=True)
        loss = h_out.sum()
        loss.backward()
    torch.cuda.synchronize()

    # Forward only
    e24_fwd_times = []
    for _ in range(bench_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            h_out, _, _ = e24_cell(x, h_tape.clone(), h_work.clone(), use_cuda=True)
        torch.cuda.synchronize()
        e24_fwd_times.append(time.perf_counter() - t0)

    e24_fwd_avg = sum(e24_fwd_times) / len(e24_fwd_times) * 1000
    e24_fwd_tok_s = (B * T) / (e24_fwd_avg / 1000)

    # Forward + backward
    e24_fwd_bwd_times = []
    for _ in range(bench_iters):
        x_tmp = x.detach().clone().requires_grad_(True)
        e24_cell.zero_grad()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        h_out, _, _ = e24_cell(x_tmp, h_tape.clone(), h_work.clone(), use_cuda=True)
        loss = h_out.sum()
        loss.backward()
        torch.cuda.synchronize()
        e24_fwd_bwd_times.append(time.perf_counter() - t0)

    e24_fwd_bwd_avg = sum(e24_fwd_bwd_times) / len(e24_fwd_bwd_times) * 1000
    e24_fwd_bwd_tok_s = (B * T) / (e24_fwd_bwd_avg / 1000)

    # ===== E23 Benchmark =====
    # Warmup
    for _ in range(warmup_iters):
        x_tmp = x.detach().clone().requires_grad_(True)
        e23_cell.zero_grad()
        h_out, _, _ = e23_cell(x_tmp, h_tape.clone(), h_work.clone(), use_cuda=True)
        loss = h_out.sum()
        loss.backward()
    torch.cuda.synchronize()

    # Forward only
    e23_fwd_times = []
    for _ in range(bench_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            h_out, _, _ = e23_cell(x, h_tape.clone(), h_work.clone(), use_cuda=True)
        torch.cuda.synchronize()
        e23_fwd_times.append(time.perf_counter() - t0)

    e23_fwd_avg = sum(e23_fwd_times) / len(e23_fwd_times) * 1000
    e23_fwd_tok_s = (B * T) / (e23_fwd_avg / 1000)

    # Forward + backward
    e23_fwd_bwd_times = []
    for _ in range(bench_iters):
        x_tmp = x.detach().clone().requires_grad_(True)
        e23_cell.zero_grad()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        h_out, _, _ = e23_cell(x_tmp, h_tape.clone(), h_work.clone(), use_cuda=True)
        loss = h_out.sum()
        loss.backward()
        torch.cuda.synchronize()
        e23_fwd_bwd_times.append(time.perf_counter() - t0)

    e23_fwd_bwd_avg = sum(e23_fwd_bwd_times) / len(e23_fwd_bwd_times) * 1000
    e23_fwd_bwd_tok_s = (B * T) / (e23_fwd_bwd_avg / 1000)

    # Calculate speedups
    fwd_speedup = e24_fwd_tok_s / e23_fwd_tok_s
    fwd_bwd_speedup = e24_fwd_bwd_tok_s / e23_fwd_bwd_tok_s

    print(f"  E24: {e24_params/1e6:.2f}M params")
    print(f"  E23: {e23_params/1e6:.2f}M params")
    print(f"  E24 Forward:     {e24_fwd_avg:.2f}ms ({e24_fwd_tok_s/1e3:.1f}K tok/s)")
    print(f"  E23 Forward:     {e23_fwd_avg:.2f}ms ({e23_fwd_tok_s/1e3:.1f}K tok/s)")
    print(f"  E24 Fwd+Bwd:     {e24_fwd_bwd_avg:.2f}ms ({e24_fwd_bwd_tok_s/1e3:.1f}K tok/s)")
    print(f"  E23 Fwd+Bwd:     {e23_fwd_bwd_avg:.2f}ms ({e23_fwd_bwd_tok_s/1e3:.1f}K tok/s)")
    print(f"  Speedup (fwd):   {fwd_speedup:.2f}x")
    print(f"  Speedup (f+b):   {fwd_bwd_speedup:.2f}x")

    results.append({
        "cfg": cfg,
        "e24_fwd": e24_fwd_tok_s,
        "e23_fwd": e23_fwd_tok_s,
        "e24_fwd_bwd": e24_fwd_bwd_tok_s,
        "e23_fwd_bwd": e23_fwd_bwd_tok_s,
        "fwd_speedup": fwd_speedup,
        "fwd_bwd_speedup": fwd_bwd_speedup,
    })

print("\n" + "=" * 70)
print("Summary Table")
print("=" * 70)
print(f"{'Config':<25} | {'E24 Fwd':>10} | {'E23 Fwd':>10} | {'E24 F+B':>10} | {'E23 F+B':>10} | {'Speedup':>7}")
print("-" * 90)
for r in results:
    cfg = r["cfg"]
    cfg_str = f"B={cfg['B']},T={cfg['T']},D={cfg['D']},N={cfg['N']}"
    print(f"{cfg_str:<25} | {r['e24_fwd']/1e3:>8.1f}K | {r['e23_fwd']/1e3:>8.1f}K | {r['e24_fwd_bwd']/1e3:>8.1f}K | {r['e23_fwd_bwd']/1e3:>8.1f}K | {r['fwd_bwd_speedup']:>6.2f}x")

print("\n" + "=" * 70)
print("E24 vs E23 comparison complete")
print("=" * 70)
