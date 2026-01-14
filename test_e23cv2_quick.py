#!/usr/bin/env python3
"""Quick validation of E23c_v2 kernel."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['LD_LIBRARY_PATH'] = f"/home/erikg/.local/lib/python3.12/site-packages/torch/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

import torch
import torch.nn as nn
import hasty_pytorch_lib

def test_e23cv2_forward():
    """Test basic E23c_v2 forward pass."""
    B, T, D, N, K = 4, 64, 256, 32, 16
    device = 'cuda'
    dtype = torch.bfloat16

    print(f"Testing E23c_v2: B={B}, T={T}, D={D}, N={N}, K={K}")
    print("=" * 60)

    # Create inputs
    torch.manual_seed(42)
    x = torch.randn(B, T, D, device=device, dtype=dtype) * 0.1
    h_tape = torch.zeros(B, N, D, device=device, dtype=dtype)
    h_work = torch.zeros(B, D, device=device, dtype=dtype)

    # Create weights
    W_h = torch.randn(D, D, device=device, dtype=dtype) * 0.01
    W_x = torch.randn(D, D, device=device, dtype=dtype) * 0.01
    W_r = torch.randn(D, D, device=device, dtype=dtype) * 0.01  # NEW: read projection
    b_h = torch.zeros(D, device=device, dtype=dtype)
    W_write = torch.randn(D, D, device=device, dtype=dtype) * 0.01

    # Make W_h well-conditioned (need float32 for orthogonal init)
    with torch.no_grad():
        W_h_f32 = torch.empty_like(W_h, dtype=torch.float32)
        nn.init.orthogonal_(W_h_f32)
        W_h.copy_(W_h_f32.to(dtype) * 0.9)

    # Run E23c_v2
    try:
        results = hasty_pytorch_lib.e23cv2_chunked_forward(
            False,  # training
            x,
            h_tape,
            h_work,
            W_h,
            W_x,
            W_r,  # NEW parameter
            b_h,
            W_write,
            K
        )

        output = results[0]
        h_tape_final = results[1]
        h_work_all = results[2]

        print(f"Output shape: {output.shape} (expected [{B}, {T}, {D}])")
        print(f"Tape shape: {h_tape_final.shape} (expected [{B}, {N}, {D}])")
        print(f"h_work_all shape: {h_work_all.shape} (expected [{B}, {T}, {D}])")

        # Check no NaN
        has_nan = torch.isnan(output).any().item()
        print(f"NaN in output: {has_nan}")

        # Check reasonable values
        print(f"Output stats: min={output.min().item():.4f}, max={output.max().item():.4f}, mean={output.mean().item():.4f}")

        if not has_nan and output.shape == (B, T, D):
            print("\nE23c_v2 forward: PASS")
            return True
        else:
            print("\nE23c_v2 forward: FAIL")
            return False

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def benchmark_e23cv2():
    """Benchmark E23c_v2 throughput."""
    import time

    B, T, D, N, K = 32, 512, 768, 64, 64
    device = 'cuda'
    dtype = torch.bfloat16

    print(f"\nBenchmark: B={B}, T={T}, D={D}, N={N}, K={K}")
    print("=" * 60)

    torch.manual_seed(42)
    x = torch.randn(B, T, D, device=device, dtype=dtype) * 0.1
    h_tape = torch.zeros(B, N, D, device=device, dtype=dtype)
    h_work = torch.zeros(B, D, device=device, dtype=dtype)

    W_h = torch.randn(D, D, device=device, dtype=dtype) * 0.01
    W_x = torch.randn(D, D, device=device, dtype=dtype) * 0.01
    W_r = torch.randn(D, D, device=device, dtype=dtype) * 0.01
    b_h = torch.zeros(D, device=device, dtype=dtype)
    W_write = torch.randn(D, D, device=device, dtype=dtype) * 0.01

    with torch.no_grad():
        W_h_f32 = torch.empty_like(W_h, dtype=torch.float32)
        nn.init.orthogonal_(W_h_f32)
        W_h.copy_(W_h_f32.to(dtype) * 0.9)

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = hasty_pytorch_lib.e23cv2_chunked_forward(
                False, x, h_tape, h_work, W_h, W_x, W_r, b_h, W_write, K)
    torch.cuda.synchronize()

    # Benchmark
    n_iters = 20
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            _ = hasty_pytorch_lib.e23cv2_chunked_forward(
                False, x, h_tape, h_work, W_h, W_x, W_r, b_h, W_write, K)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / n_iters * 1000

    tokens = B * T
    tok_per_sec = tokens / (elapsed / 1000) / 1000

    print(f"E23c_v2: {elapsed:.2f}ms ({tok_per_sec:.1f}K tok/s)")

    # Compare with E23c (no read feedback)
    for _ in range(3):
        with torch.no_grad():
            _ = hasty_pytorch_lib.e23c_chunked_forward(
                False, x, h_tape, h_work, W_h, W_x, b_h, W_write, K)
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            _ = hasty_pytorch_lib.e23c_chunked_forward(
                False, x, h_tape, h_work, W_h, W_x, b_h, W_write, K)
    torch.cuda.synchronize()
    elapsed_c = (time.perf_counter() - t0) / n_iters * 1000
    tok_per_sec_c = tokens / (elapsed_c / 1000) / 1000

    print(f"E23c:    {elapsed_c:.2f}ms ({tok_per_sec_c:.1f}K tok/s)")
    print(f"E23c_v2 vs E23c: {elapsed_c/elapsed:.2f}x")


if __name__ == '__main__':
    if test_e23cv2_forward():
        benchmark_e23cv2()
