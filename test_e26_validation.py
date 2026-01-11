#!/usr/bin/env python3
"""Validation of E26 CUDA kernel vs Python reference."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['LD_LIBRARY_PATH'] = f"/home/erikg/.local/lib/python3.12/site-packages/torch/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

import torch
import torch.nn as nn
import hasty_pytorch_lib

from elman.models.e26_parallel import e26_forward_python


def test_cuda_forward():
    """Test CUDA E26 forward pass."""
    print("=" * 60)
    print("Test 1: E26 CUDA Forward Pass")
    print("=" * 60)

    B, T, D, N = 2, 8, 256, 8
    device = 'cuda'
    dtype = torch.bfloat16

    torch.manual_seed(42)
    x = torch.randn(B, T, D, device=device, dtype=dtype) * 0.1
    h_tape = torch.zeros(B, N, D, device=device, dtype=dtype)
    h_work = torch.zeros(B, D, device=device, dtype=dtype)

    W_h = torch.randn(D, D, device=device, dtype=dtype) * 0.01
    W_x = torch.randn(D, D, device=device, dtype=dtype) * 0.01
    b_h = torch.zeros(D, device=device, dtype=dtype)
    W_write = torch.randn(D, D, device=device, dtype=dtype) * 0.01

    with torch.no_grad():
        W_h_f32 = torch.empty_like(W_h, dtype=torch.float32)
        nn.init.orthogonal_(W_h_f32)
        W_h.copy_(W_h_f32.to(dtype) * 0.9)

    results = hasty_pytorch_lib.e26_parallel_forward(
        True, x, h_tape, h_work, W_h, W_x, b_h, W_write
    )

    h_work_out, h_tape_final, h_tape_all, read_attn, write_attn = results

    print(f"h_work_out shape: {h_work_out.shape} (expected [{B}, {T}, {D}])")
    print(f"h_tape_final shape: {h_tape_final.shape} (expected [{B}, {N}, {D}])")
    print(f"read_attn shape: {read_attn.shape} (expected [{B}, {T}, {N}])")
    print(f"write_attn shape: {write_attn.shape} (expected [{B}, {T}, {N}])")

    has_nan = torch.isnan(h_work_out).any().item()
    print(f"NaN in output: {has_nan}")

    # Check attention sums to 1
    read_sum = read_attn.sum(dim=-1)
    write_sum = write_attn.sum(dim=-1)
    print(f"Read attention sum range: [{read_sum.min().item():.4f}, {read_sum.max().item():.4f}]")
    print(f"Write attention sum range: [{write_sum.min().item():.4f}, {write_sum.max().item():.4f}]")

    if not has_nan:
        print("PASS: CUDA forward produces valid output\n")
        return True
    else:
        print("FAIL: NaN in output\n")
        return False


def test_python_vs_cuda():
    """Compare Python and CUDA implementations."""
    print("=" * 60)
    print("Test 2: Python vs CUDA Comparison")
    print("=" * 60)

    B, T, D, N = 2, 4, 256, 8
    device = 'cuda'
    dtype = torch.bfloat16

    torch.manual_seed(42)
    x = torch.randn(B, T, D, device=device, dtype=dtype) * 0.1
    h_tape = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
    h_work = torch.randn(B, D, device=device, dtype=dtype) * 0.1

    W_h = torch.randn(D, D, device=device, dtype=dtype) * 0.01
    W_x = torch.randn(D, D, device=device, dtype=dtype) * 0.01
    b_h = torch.zeros(D, device=device, dtype=dtype)
    W_write = torch.randn(D, D, device=device, dtype=dtype) * 0.01

    with torch.no_grad():
        W_h_f32 = torch.empty_like(W_h, dtype=torch.float32)
        nn.init.orthogonal_(W_h_f32)
        W_h.copy_(W_h_f32.to(dtype) * 0.9)

    # CUDA forward
    cuda_results = hasty_pytorch_lib.e26_parallel_forward(
        True, x, h_tape.clone(), h_work.clone(), W_h, W_x, b_h, W_write
    )
    cuda_h_work = cuda_results[0]
    cuda_read_attn = cuda_results[3]
    cuda_write_attn = cuda_results[4]

    # Python forward
    py_h_work, py_h_tape, py_read_attn, py_write_attn = e26_forward_python(
        x, h_tape.clone(), h_work.clone(), W_h, W_x, b_h, W_write
    )

    # Compare
    h_work_diff = (cuda_h_work.float() - py_h_work.float()).abs()
    read_diff = (cuda_read_attn.float() - py_read_attn.float()).abs()
    write_diff = (cuda_write_attn.float() - py_write_attn.float()).abs()

    print(f"h_work difference: max={h_work_diff.max().item():.6f}, mean={h_work_diff.mean().item():.6f}")
    print(f"read_attn difference: max={read_diff.max().item():.6f}, mean={read_diff.mean().item():.6f}")
    print(f"write_attn difference: max={write_diff.max().item():.6f}, mean={write_diff.mean().item():.6f}")

    tol = 0.1
    if h_work_diff.max().item() < tol and read_diff.max().item() < tol:
        print("PASS: Python and CUDA match\n")
        return True
    else:
        print("FAIL: Mismatch\n")
        return False


def test_backward():
    """Test backward pass produces valid gradients."""
    print("=" * 60)
    print("Test 3: E26 Backward Pass")
    print("=" * 60)

    B, T, D, N = 2, 4, 256, 8
    device = 'cuda'
    dtype = torch.bfloat16

    torch.manual_seed(42)
    x = torch.randn(B, T, D, device=device, dtype=dtype) * 0.1
    h_tape = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
    h_work = torch.zeros(B, D, device=device, dtype=dtype)

    W_h = torch.randn(D, D, device=device, dtype=dtype) * 0.01
    W_x = torch.randn(D, D, device=device, dtype=dtype) * 0.01
    b_h = torch.zeros(D, device=device, dtype=dtype)
    W_write = torch.randn(D, D, device=device, dtype=dtype) * 0.01

    with torch.no_grad():
        W_h_f32 = torch.empty_like(W_h, dtype=torch.float32)
        nn.init.orthogonal_(W_h_f32)
        W_h.copy_(W_h_f32.to(dtype) * 0.9)

    # Forward
    results = hasty_pytorch_lib.e26_parallel_forward(
        True, x, h_tape, h_work, W_h, W_x, b_h, W_write
    )
    h_work_out, h_tape_final, h_tape_all, read_attn, write_attn = results

    # Create gradients
    d_h_work_out = torch.randn_like(h_work_out) * 0.1
    d_h_tape_final = torch.randn_like(h_tape_final) * 0.1

    # Backward
    grads = hasty_pytorch_lib.e26_parallel_backward(
        x, h_work_out, h_work, h_tape_all,
        read_attn, write_attn,
        W_h, W_x, W_write,
        d_h_work_out, d_h_tape_final
    )

    dx, dW_h, dW_x, db_h, dW_write = grads

    print(f"dx shape: {dx.shape}, norm: {dx.norm().item():.6f}")
    print(f"dW_h norm: {dW_h.norm().item():.6f}")
    print(f"dW_x norm: {dW_x.norm().item():.6f}")
    print(f"db_h norm: {db_h.norm().item():.6f}")
    print(f"dW_write norm: {dW_write.norm().item():.6f}")

    has_nan = any([
        torch.isnan(dx).any().item(),
        torch.isnan(dW_h).any().item(),
        torch.isnan(dW_x).any().item(),
        torch.isnan(db_h).any().item(),
        torch.isnan(dW_write).any().item(),
    ])

    if not has_nan:
        print("PASS: Backward produces valid gradients\n")
        return True
    else:
        print("FAIL: NaN in gradients\n")
        return False


def benchmark():
    """Benchmark E26 vs E23c vs E25."""
    print("=" * 60)
    print("Benchmark: E26 vs E23c vs E25")
    print("=" * 60)

    import time

    B, T, D, N = 32, 512, 512, 32
    device = 'cuda'
    dtype = torch.bfloat16

    torch.manual_seed(42)
    x = torch.randn(B, T, D, device=device, dtype=dtype) * 0.1
    h_tape = torch.zeros(B, N, D, device=device, dtype=dtype)
    h_work = torch.zeros(B, D, device=device, dtype=dtype)

    W_h = torch.randn(D, D, device=device, dtype=dtype) * 0.01
    W_x = torch.randn(D, D, device=device, dtype=dtype) * 0.01
    b_h = torch.zeros(D, device=device, dtype=dtype)
    W_write = torch.randn(D, D, device=device, dtype=dtype) * 0.01

    with torch.no_grad():
        W_h_f32 = torch.empty_like(W_h, dtype=torch.float32)
        nn.init.orthogonal_(W_h_f32)
        W_h.copy_(W_h_f32.to(dtype) * 0.9)

    n_iters = 20
    tokens = B * T

    # E26
    for _ in range(3):
        with torch.no_grad():
            _ = hasty_pytorch_lib.e26_parallel_forward(
                False, x, h_tape, h_work, W_h, W_x, b_h, W_write)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            _ = hasty_pytorch_lib.e26_parallel_forward(
                False, x, h_tape, h_work, W_h, W_x, b_h, W_write)
    torch.cuda.synchronize()
    e26_ms = (time.perf_counter() - t0) / n_iters * 1000
    e26_toks = tokens / (e26_ms / 1000) / 1000

    # E23c
    for _ in range(3):
        with torch.no_grad():
            _ = hasty_pytorch_lib.e23c_chunked_forward(
                False, x, h_tape, h_work, W_h, W_x, b_h, W_write, 64)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            _ = hasty_pytorch_lib.e23c_chunked_forward(
                False, x, h_tape, h_work, W_h, W_x, b_h, W_write, 64)
    torch.cuda.synchronize()
    e23c_ms = (time.perf_counter() - t0) / n_iters * 1000
    e23c_toks = tokens / (e23c_ms / 1000) / 1000

    # E25
    for _ in range(3):
        with torch.no_grad():
            _ = hasty_pytorch_lib.e25_entmax_forward(
                False, x, h_tape, h_work, W_h, W_x, b_h, W_write)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            _ = hasty_pytorch_lib.e25_entmax_forward(
                False, x, h_tape, h_work, W_h, W_x, b_h, W_write)
    torch.cuda.synchronize()
    e25_ms = (time.perf_counter() - t0) / n_iters * 1000
    e25_toks = tokens / (e25_ms / 1000) / 1000

    print(f"Config: B={B}, T={T}, D={D}, N={N}")
    print(f"E26 (softmax):  {e26_ms:.2f}ms ({e26_toks:.1f}K tok/s)")
    print(f"E23c (softmax): {e23c_ms:.2f}ms ({e23c_toks:.1f}K tok/s)")
    print(f"E25 (entmax):   {e25_ms:.2f}ms ({e25_toks:.1f}K tok/s)")
    print(f"E26 vs E23c: {e23c_ms/e26_ms:.2f}x")
    print(f"E26 vs E25:  {e25_ms/e26_ms:.2f}x")


if __name__ == '__main__':
    test_cuda_forward()
    test_python_vs_cuda()
    test_backward()
    benchmark()
