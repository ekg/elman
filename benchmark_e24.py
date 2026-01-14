#!/usr/bin/env python3
"""
Benchmark E24 CUDA kernel performance.
"""
import torch
import time
import sys
sys.path.insert(0, 'elman/cuda')

from elman.models.e24_single_gemm import E24Cell

device = 'cuda'
dtype = torch.bfloat16

print("=" * 70)
print("E24 CUDA Benchmark")
print("=" * 70)

# Benchmark parameters
configs = [
    {"B": 4, "T": 512, "D": 256, "N": 16},
    {"B": 4, "T": 512, "D": 512, "N": 16},
    {"B": 4, "T": 512, "D": 768, "N": 16},
    {"B": 4, "T": 512, "D": 1024, "N": 16},
    {"B": 8, "T": 512, "D": 512, "N": 16},
    {"B": 4, "T": 1024, "D": 512, "N": 16},
]

warmup_iters = 5
bench_iters = 20

for cfg in configs:
    B, T, D, N = cfg["B"], cfg["T"], cfg["D"], cfg["N"]
    print(f"\nConfig: B={B}, T={T}, D={D}, N={N}")
    print("-" * 50)

    # Create cell
    cell = E24Cell(dim=D, n_slots=N).to(device).to(dtype)

    # Create input
    x = torch.randn(B, T, D, device=device, dtype=dtype)
    h_tape = torch.zeros(B, N, D, device=device, dtype=dtype)
    h_work = torch.zeros(B, D, device=device, dtype=dtype)

    # Warmup
    for _ in range(warmup_iters):
        x_tmp = x.detach().clone().requires_grad_(True)
        cell.zero_grad()
        h_out, _, _ = cell(x_tmp, h_tape.clone(), h_work.clone(), use_cuda=True)
        loss = h_out.sum()
        loss.backward()

    torch.cuda.synchronize()

    # Benchmark forward
    fwd_times = []
    for _ in range(bench_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            h_out, _, _ = cell(x, h_tape.clone(), h_work.clone(), use_cuda=True)
        torch.cuda.synchronize()
        fwd_times.append(time.perf_counter() - t0)

    fwd_avg = sum(fwd_times) / len(fwd_times) * 1000  # ms
    fwd_tok_s = (B * T) / (fwd_avg / 1000)

    # Benchmark forward + backward
    fwd_bwd_times = []
    for _ in range(bench_iters):
        x_tmp = x.detach().clone().requires_grad_(True)
        cell.zero_grad()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        h_out, _, _ = cell(x_tmp, h_tape.clone(), h_work.clone(), use_cuda=True)
        loss = h_out.sum()
        loss.backward()
        torch.cuda.synchronize()
        fwd_bwd_times.append(time.perf_counter() - t0)

    fwd_bwd_avg = sum(fwd_bwd_times) / len(fwd_bwd_times) * 1000  # ms
    fwd_bwd_tok_s = (B * T) / (fwd_bwd_avg / 1000)

    # Count params
    n_params = sum(p.numel() for p in cell.parameters())

    print(f"  Params: {n_params/1e6:.2f}M")
    print(f"  Forward:     {fwd_avg:.2f}ms ({fwd_tok_s/1e3:.1f}K tok/s)")
    print(f"  Fwd+Bwd:     {fwd_bwd_avg:.2f}ms ({fwd_bwd_tok_s/1e3:.1f}K tok/s)")

print("\n" + "=" * 70)
print("Benchmark complete")
print("=" * 70)
