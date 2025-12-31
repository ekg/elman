#!/usr/bin/env python3
import sys
sys.path.insert(0, "/home/erikg/elman/elman/cuda")
import torch
import time

torch.set_default_device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import hasty_pytorch_lib as hasty

D = 2048

def fmt(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    return f"{n/1_000:.0f}K"

def benchmark(name, forward_fn, backward_fn, setup_fn, batch_size=16, seq_len=1024, dim=D, n_warmup=3, n_runs=5):
    torch.cuda.synchronize()
    tensors = setup_fn(batch_size, seq_len, dim)

    for _ in range(n_warmup):
        results = forward_fn(*tensors)
        if backward_fn:
            backward_fn(tensors, results)

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        results = forward_fn(*tensors)
        if backward_fn:
            backward_fn(tensors, results)
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    return (n_runs * batch_size * seq_len) / elapsed

def setup_stock(B, T, D):
    return (True, torch.randn(T,B,D), torch.zeros(B,D),
            torch.randn(D,D)/(D**0.5), torch.randn(D,D)/(D**0.5), torch.zeros(D))

def setup_gated(B, T, D):
    return (True, torch.randn(T,B,D), torch.zeros(B,D),
            torch.randn(D,D)/(D**0.5), torch.randn(D,D)/(D**0.5),
            torch.randn(D,D)/(D**0.5), torch.zeros(D), torch.zeros(D))

def setup_selective(B, T, D):
    return (True, torch.randn(T,B,D), torch.zeros(B,D),
            torch.randn(D,D)/(D**0.5), torch.randn(D,D)/(D**0.5),
            torch.randn(D,D)/(D**0.5), torch.randn(D,D)/(D**0.5),
            torch.zeros(D), torch.zeros(D), 4)

def setup_diagonal_selective(B, T, D):
    return (True, torch.randn(T,B,D), torch.zeros(B,D),
            torch.randn(D,D)/(D**0.5), torch.randn(D)/(D**0.5),
            torch.randn(D,D)/(D**0.5), torch.randn(D,D)/(D**0.5),
            torch.zeros(D), torch.zeros(D), 4)

def setup_full_recurrence(B, T, D):
    return (True, torch.randn(T,B,D), torch.zeros(B,D),
            torch.randn(D,D)/(D**0.5), torch.randn(D,D)/(D**0.5),
            torch.randn(D,D)/(D**0.5), torch.randn(D,D)/(D**0.5),
            torch.zeros(D), torch.zeros(D), 4)

def setup_linear_triple_r(B, T, D):
    return (True, torch.randn(T,B,D), torch.zeros(B,D),
            torch.randn(D,D)/(D**0.5), torch.randn(D,D)/(D**0.5),
            torch.randn(D,D)/(D**0.5), torch.randn(D,D)/(D**0.5),
            torch.randn(D,D)/(D**0.5), torch.zeros(D), torch.zeros(D), 4)

def setup_linear_polynomial(B, T, D):
    return (True, torch.randn(T,B,D), torch.zeros(B,D),
            torch.randn(D,D)/(D**0.5), torch.randn(D)/(D**0.5),
            torch.randn(D,D)/(D**0.5), torch.zeros(D),
            torch.randn(D,D)/(D**0.5), torch.randn(D,D)/(D**0.5),
            torch.zeros(D), torch.zeros(D), 4)

def setup_log_storage_diagonal(B, T, D):
    return (True, torch.randn(T,B,D), torch.zeros(B,D), torch.ones(B,D),
            torch.randn(D,D)/(D**0.5), torch.randn(D)/(D**0.5),
            torch.randn(D,D)/(D**0.5), torch.randn(D,D)/(D**0.5),
            torch.zeros(D), torch.zeros(D), 4)

def setup_log_compute_full(B, T, D):
    return (True, torch.randn(T,B,D), torch.zeros(B,D), torch.ones(B,D),
            torch.randn(D,D)/(D**0.5), torch.randn(D,D)/(D**0.5),
            torch.randn(D,D)/(D**0.5), torch.randn(D,D)/(D**0.5),
            torch.zeros(D), torch.zeros(D), 4)

def setup_logspace_triple_r(B, T, D):
    return (True, torch.randn(T,B,D), torch.zeros(B,D), torch.ones(B,D),
            torch.randn(D,D)/(D**0.5), torch.randn(D,D)/(D**0.5),
            torch.randn(D,D)/(D**0.5), torch.randn(D,D)/(D**0.5),
            torch.randn(D,D)/(D**0.5), torch.zeros(D), torch.zeros(D), 4)

def setup_logspace_polynomial(B, T, D):
    return (True, torch.randn(T,B,D), torch.zeros(B,D), torch.ones(B,D),
            torch.randn(D,D)/(D**0.5), torch.zeros(D), torch.ones(D),
            torch.randn(D,D)/(D**0.5), torch.zeros(D),
            torch.randn(D,D)/(D**0.5), torch.zeros(D), torch.zeros(D), torch.zeros(D))

params = {
    "stock_elman": 2*D*D + D,
    "gated_elman": 3*D*D + 2*D,
    "selective_elman": 4*D*D + 2*D,
    "diagonal_selective": 3*D*D + 3*D,
    "full_recurrence": 4*D*D + 2*D,
    "linear_triple_r": 5*D*D + 2*D,
    "linear_polynomial": 4*D*D + 4*D,
    "log_storage_diagonal": 3*D*D + 3*D,
    "log_compute_full": 4*D*D + 2*D,
    "logspace_triple_r": 5*D*D + 2*D,
    "logspace_polynomial": 3*D*D + 5*D,
}

print("=" * 60)
print("Elman Ladder Benchmarks @ dim=2048 (1x RTX 6000 Ada)")
print("Config: batch_size=16, seq_len=1024")
print("=" * 60)
print(f"{'Kernel':<25} {'Params':>10} {'tok/s':>12}")
print("-" * 60)

kernels = [
    ("stock_elman", setup_stock,
     lambda *a: hasty.stock_elman_forward(*a),
     lambda i, o: hasty.stock_elman_backward(i[3], i[4], i[1], o[0], o[1], torch.randn_like(o[0][1:]))),

    ("gated_elman", setup_gated,
     lambda *a: hasty.gated_elman_forward(*a),
     lambda i, o: hasty.gated_elman_backward(i[3], i[4], i[5], i[1], o[0], o[1], o[2], torch.randn_like(o[0][1:]))),

    ("selective_elman", setup_selective,
     lambda *a: hasty.selective_elman_forward(*a),
     lambda i, o: hasty.selective_elman_backward(i[3], i[4], i[5], i[6], i[1], o[0], o[2], o[3], o[4], torch.randn_like(o[1]), i[9])),

    ("diagonal_selective", setup_diagonal_selective,
     lambda *a: hasty.diagonal_selective_forward(*a),
     lambda i, o: hasty.diagonal_selective_backward(i[3], i[4], i[5], i[6], i[1], o[0], o[2], o[3], o[4], torch.randn_like(o[1]), i[9])),

    ("full_recurrence", setup_full_recurrence,
     lambda *a: hasty.full_recurrence_forward(*a),
     lambda i, o: hasty.full_recurrence_backward(i[3], i[4], i[5], i[6], i[1], o[0], o[2], o[3], o[4], torch.randn_like(o[1]), i[9])),

    ("linear_triple_r", setup_linear_triple_r,
     lambda *a: hasty.linear_triple_r_forward(*a),
     lambda i, o: hasty.linear_triple_r_backward(i[3], i[4], i[5], i[6], i[7], i[1], o[0], o[2], o[3], o[4], torch.randn_like(o[1]), i[10])),

    ("linear_polynomial", setup_linear_polynomial,
     lambda *a: hasty.linear_polynomial_forward(*a),
     lambda i, o: hasty.linear_polynomial_backward(i[3], i[4], i[5], i[7], i[8], i[1], o[0], o[2], o[3], o[4], o[5], torch.randn_like(o[1]), i[11])),

    ("log_storage_diagonal", setup_log_storage_diagonal,
     lambda *a: hasty.log_storage_diagonal_forward(*a),
     lambda i, o: hasty.log_storage_diagonal_backward(i[4], i[5], i[6], i[7], i[1], o[0], o[1], o[3], o[4], o[5], o[6], o[7], o[8], torch.randn_like(o[2]), i[10])),

    ("log_compute_full", setup_log_compute_full,
     lambda *a: hasty.log_compute_full_forward(*a),
     lambda i, o: hasty.log_compute_full_backward(i[4], i[5], i[6], i[7], i[1], o[0], o[1], o[3], o[4], o[5], torch.randn_like(o[2]), i[10])),

    ("logspace_triple_r", setup_logspace_triple_r,
     lambda *a: hasty.logspace_triple_r_forward(*a),
     lambda i, o: hasty.logspace_triple_r_backward(i[4], i[5], i[6], i[7], i[8], i[1], o[0], o[1], o[3], o[4], o[5], torch.randn_like(o[2]), i[11])),

    ("logspace_polynomial", setup_logspace_polynomial,
     lambda *a: hasty.logspace_polynomial_forward(*a),
     lambda i, o: hasty.logspace_polynomial_backward(i[4], i[5], i[6], i[7], i[9], i[12], i[1], o[0], o[1], o[3], o[4], o[5], o[9], o[6], o[7], o[8], o[2], torch.randn_like(o[2]))),
]

for name, setup, fwd, bwd in kernels:
    try:
        tok_s = benchmark(name, fwd, bwd, setup)
        p = params[name]
        print(f"{name:<25} {fmt(p):>10} {tok_s:>12,.0f}")
    except Exception as e:
        print(f"{name:<25} {'ERROR':>10} {str(e)[:25]}")

print("=" * 60)
print()
print("For ~500M model: stack 30 layers of linear_polynomial")
print(f"  -> 30 x {fmt(params['linear_polynomial'])} = {fmt(30*params['linear_polynomial'])} params")
