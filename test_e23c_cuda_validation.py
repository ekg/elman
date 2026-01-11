"""
E23c CUDA vs Python Validation Test

Validates mathematical equivalence between:
- Python: elman/models/dual_memory_elman_chunked.py (DualMemoryElmanChunkedCell)
- CUDA: elman/cuda/lib/e23c_chunked_gpu.cu.cc (E23cChunkedForward)

Key E23c architecture:
  h_work_t = tanh(W_h @ h_work_{t-1} + W_x @ x_t + b)  # No read dependency!
  output_t = h_work_t + read_t                         # Additive read
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set library path for torch
os.environ['LD_LIBRARY_PATH'] = f"/home/erikg/.local/lib/python3.12/site-packages/torch/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

import torch
import torch.nn.functional as F
import math
import numpy as np

# Import CUDA binding
import hasty_pytorch_lib

# Import Python implementation
from elman.models.dual_memory_elman_chunked import DualMemoryElmanChunkedCell


def test_e23c_forward_equivalence(
    B=4, T=128, D=256, N=64, K=32,
    dtype=torch.bfloat16,
    atol=1e-2, rtol=1e-2,
    verbose=True
):
    """
    Test that Python and CUDA E23c forward passes produce identical outputs.

    Args:
        B: batch size
        T: sequence length
        D: hidden dimension
        N: number of tape slots
        K: chunk size
        dtype: torch dtype to test
        atol: absolute tolerance
        rtol: relative tolerance
    """
    device = 'cuda'

    if verbose:
        print(f"\n{'='*70}")
        print(f"E23c Forward Validation: B={B}, T={T}, D={D}, N={N}, K={K}, dtype={dtype}")
        print(f"{'='*70}")

    # Create Python cell
    py_cell = DualMemoryElmanChunkedCell(dim=D, n_slots=N, chunk_size=K).to(device).to(dtype)

    # Initialize states
    torch.manual_seed(42)
    x_seq = torch.randn(B, T, D, device=device, dtype=dtype) * 0.1
    h_tape = torch.zeros(B, N, D, device=device, dtype=dtype)
    h_work = torch.zeros(B, D, device=device, dtype=dtype)

    # Run Python forward
    with torch.no_grad():
        py_output, py_h_tape, py_h_work = py_cell(x_seq, h_tape.clone(), h_work.clone())

    # Prepare weights for CUDA kernel (same weights, no transpose needed)
    W_h = py_cell.W_h.weight.contiguous()
    W_x = py_cell.W_x.weight.contiguous()
    b_h = py_cell.b_h
    W_write = py_cell.W_write.weight.contiguous()

    # Run CUDA forward
    with torch.no_grad():
        cuda_results = hasty_pytorch_lib.e23c_chunked_forward(
            False,  # training
            x_seq,
            h_tape.clone(),
            h_work.clone(),
            W_h,
            W_x,
            b_h,
            W_write,
            K
        )

    cuda_output = cuda_results[0]
    cuda_h_tape = cuda_results[1]

    # Compare outputs
    output_diff = torch.abs(py_output - cuda_output)
    tape_diff = torch.abs(py_h_tape - cuda_h_tape)

    max_output_diff = output_diff.max().item()
    mean_output_diff = output_diff.mean().item()
    max_tape_diff = tape_diff.max().item()
    mean_tape_diff = tape_diff.mean().item()

    if verbose:
        print(f"\nOutput comparison:")
        print(f"  Max diff:  {max_output_diff:.6e}")
        print(f"  Mean diff: {mean_output_diff:.6e}")
        print(f"  Py shape:  {py_output.shape}")
        print(f"  CUDA shape: {cuda_output.shape}")

        print(f"\nTape comparison:")
        print(f"  Max diff:  {max_tape_diff:.6e}")
        print(f"  Mean diff: {mean_tape_diff:.6e}")

        # Sample values
        print(f"\nSample output values (first 5 elements of [0,0,:]):")
        print(f"  Python: {py_output[0,0,:5].tolist()}")
        print(f"  CUDA:   {cuda_output[0,0,:5].tolist()}")

    # Check tolerance
    output_pass = torch.allclose(py_output, cuda_output, atol=atol, rtol=rtol)
    tape_pass = torch.allclose(py_h_tape, cuda_h_tape, atol=atol, rtol=rtol)

    if verbose:
        print(f"\nTest Results:")
        print(f"  Output equivalence: {'PASS' if output_pass else 'FAIL'}")
        print(f"  Tape equivalence:   {'PASS' if tape_pass else 'FAIL'}")

    return output_pass and tape_pass, {
        'max_output_diff': max_output_diff,
        'mean_output_diff': mean_output_diff,
        'max_tape_diff': max_tape_diff,
        'mean_tape_diff': mean_tape_diff,
    }


def test_h_work_sequential_equivalence(B=4, T=32, D=128, N=32, K=16, verbose=True):
    """
    Test that h_work updates are computed correctly in both implementations.
    The h_work update should be: h_work_t = tanh(W_h @ h_work_{t-1} + W_x @ x_t + b)
    """
    device = 'cuda'
    dtype = torch.float32  # Use float32 for higher precision comparison

    if verbose:
        print(f"\n{'='*70}")
        print(f"h_work Sequential Update Test (float32 for precision)")
        print(f"{'='*70}")

    # Create Python cell
    py_cell = DualMemoryElmanChunkedCell(dim=D, n_slots=N, chunk_size=K).to(device).to(dtype)

    # Initialize
    torch.manual_seed(123)
    x_seq = torch.randn(B, T, D, device=device, dtype=dtype) * 0.1
    h_tape = torch.zeros(B, N, D, device=device, dtype=dtype)
    h_work = torch.zeros(B, D, device=device, dtype=dtype)

    # Manually compute h_work sequence using the exact formula
    W_h = py_cell.W_h.weight
    W_x = py_cell.W_x.weight
    b_h = py_cell.b_h

    manual_h_work = []
    h_cur = h_work.clone()
    for t in range(T):
        Rh = F.linear(h_cur, W_h)
        Wx = F.linear(x_seq[:, t], W_x)
        h_cur = torch.tanh(Rh + Wx + b_h)
        manual_h_work.append(h_cur)
    manual_h_work = torch.stack(manual_h_work, dim=1)  # [B, T, D]

    # Get Python implementation result
    with torch.no_grad():
        py_output, py_h_tape, py_h_work = py_cell(x_seq, h_tape.clone(), h_work.clone())

    # In Python impl, we get output = h_work + read, but h_work_all is internal
    # We need to run CUDA to get h_work_all
    W_h = W_h.T.contiguous()
    W_x = W_x.T.contiguous()
    W_write = py_cell.W_write.weight.contiguous()

    with torch.no_grad():
        cuda_results = hasty_pytorch_lib.e23c_chunked_forward(
            False, x_seq, h_tape.clone(), h_work.clone(),
            W_h, W_x, b_h, W_write, K
        )

    cuda_h_work_all = cuda_results[2]  # [B, T, D]

    # Compare manual vs CUDA h_work
    diff = torch.abs(manual_h_work - cuda_h_work_all)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    if verbose:
        print(f"\nManual h_work vs CUDA h_work_all:")
        print(f"  Max diff:  {max_diff:.6e}")
        print(f"  Mean diff: {mean_diff:.6e}")
        print(f"  Sample [0,0,:5]:")
        print(f"    Manual: {manual_h_work[0,0,:5].tolist()}")
        print(f"    CUDA:   {cuda_h_work_all[0,0,:5].tolist()}")

    passed = max_diff < 1e-4  # Very tight tolerance for float32
    print(f"\nh_work equivalence: {'PASS' if passed else 'FAIL'}")
    return passed


def test_chunked_attention_correctness(B=2, T=64, D=128, N=32, K=16, verbose=True):
    """
    Test that chunked batched attention produces correct results.
    """
    device = 'cuda'
    dtype = torch.float32

    if verbose:
        print(f"\n{'='*70}")
        print(f"Chunked Attention Correctness Test")
        print(f"{'='*70}")

    # Create cell
    py_cell = DualMemoryElmanChunkedCell(dim=D, n_slots=N, chunk_size=K).to(device).to(dtype)

    torch.manual_seed(456)
    x_seq = torch.randn(B, T, D, device=device, dtype=dtype) * 0.1
    # Initialize tape with some content to test attention
    h_tape = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
    h_work = torch.zeros(B, D, device=device, dtype=dtype)

    # Run Python
    with torch.no_grad():
        py_output, py_h_tape, py_h_work = py_cell(x_seq, h_tape.clone(), h_work.clone())

    # Run CUDA
    W_h = py_cell.W_h.weight.contiguous()
    W_x = py_cell.W_x.weight.contiguous()
    W_write = py_cell.W_write.weight.contiguous()

    with torch.no_grad():
        cuda_results = hasty_pytorch_lib.e23c_chunked_forward(
            False, x_seq, h_tape.clone(), h_work.clone(),
            W_h, W_x, py_cell.b_h, W_write, K
        )

    cuda_output = cuda_results[0]
    cuda_h_tape = cuda_results[1]

    output_diff = torch.abs(py_output - cuda_output).max().item()
    tape_diff = torch.abs(py_h_tape - cuda_h_tape).max().item()

    if verbose:
        print(f"\nWith non-zero initial tape:")
        print(f"  Output max diff: {output_diff:.6e}")
        print(f"  Tape max diff:   {tape_diff:.6e}")

    passed = output_diff < 1e-3 and tape_diff < 1e-3
    print(f"\nChunked attention: {'PASS' if passed else 'FAIL'}")
    return passed


def test_various_configs(verbose=True):
    """
    Test multiple configurations to ensure robustness.
    """
    configs = [
        # (B, T, D, N, K)
        (1, 64, 64, 8, 8),     # Minimal
        (2, 128, 128, 16, 16), # Small
        (4, 256, 256, 32, 32), # Medium
        (8, 512, 512, 64, 64), # Large
        (4, 100, 256, 64, 25), # Non-power-of-2 T and K
    ]

    print(f"\n{'='*70}")
    print("Testing Various Configurations (bf16)")
    print(f"{'='*70}")

    all_passed = True
    for B, T, D, N, K in configs:
        passed, stats = test_e23c_forward_equivalence(
            B=B, T=T, D=D, N=N, K=K,
            dtype=torch.bfloat16,
            atol=0.1, rtol=0.1,  # Looser tolerance for bf16
            verbose=False
        )
        status = "PASS" if passed else "FAIL"
        print(f"  B={B:2}, T={T:3}, D={D:3}, N={N:2}, K={K:2}: {status} "
              f"(max_out={stats['max_output_diff']:.4f}, max_tape={stats['max_tape_diff']:.4f})")
        all_passed = all_passed and passed

    return all_passed


def benchmark_speedup(B=32, T=512, D=768, N=64, K=64):
    """
    Benchmark Python vs CUDA speed.
    """
    device = 'cuda'
    dtype = torch.bfloat16

    print(f"\n{'='*70}")
    print(f"Benchmark: B={B}, T={T}, D={D}, N={N}, K={K}")
    print(f"{'='*70}")

    py_cell = DualMemoryElmanChunkedCell(dim=D, n_slots=N, chunk_size=K).to(device).to(dtype)

    torch.manual_seed(789)
    x_seq = torch.randn(B, T, D, device=device, dtype=dtype)
    h_tape = torch.zeros(B, N, D, device=device, dtype=dtype)
    h_work = torch.zeros(B, D, device=device, dtype=dtype)

    W_h = py_cell.W_h.weight.contiguous()
    W_x = py_cell.W_x.weight.contiguous()
    W_write = py_cell.W_write.weight.contiguous()

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = py_cell(x_seq, h_tape.clone(), h_work.clone())
            _ = hasty_pytorch_lib.e23c_chunked_forward(
                False, x_seq, h_tape.clone(), h_work.clone(),
                W_h, W_x, py_cell.b_h, W_write, K
            )
    torch.cuda.synchronize()

    # Benchmark
    import time
    n_iters = 10

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            _ = py_cell(x_seq, h_tape.clone(), h_work.clone())
    torch.cuda.synchronize()
    py_time = (time.perf_counter() - t0) / n_iters * 1000

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            _ = hasty_pytorch_lib.e23c_chunked_forward(
                False, x_seq, h_tape.clone(), h_work.clone(),
                W_h, W_x, py_cell.b_h, W_write, K
            )
    torch.cuda.synchronize()
    cuda_time = (time.perf_counter() - t0) / n_iters * 1000

    tokens = B * T
    py_toks = tokens / (py_time / 1000) / 1000
    cuda_toks = tokens / (cuda_time / 1000) / 1000

    print(f"\nPython:  {py_time:.1f}ms ({py_toks:.1f}K tok/s)")
    print(f"CUDA:    {cuda_time:.1f}ms ({cuda_toks:.1f}K tok/s)")
    print(f"Speedup: {py_time/cuda_time:.2f}x")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("E23c CUDA vs Python Validation Suite")
    print("="*70)

    # Test 1: Basic forward equivalence with float32
    test_e23c_forward_equivalence(B=4, T=64, D=128, N=32, K=16,
                                   dtype=torch.float32, atol=1e-4, rtol=1e-4)

    # Test 2: h_work sequential computation
    test_h_work_sequential_equivalence()

    # Test 3: Chunked attention with non-zero tape
    test_chunked_attention_correctness()

    # Test 4: Various configurations (bf16)
    test_various_configs()

    # Test 5: Benchmark
    benchmark_speedup(B=32, T=512, D=768, N=64, K=64)

    print("\n" + "="*70)
    print("All validation tests complete!")
    print("="*70)
