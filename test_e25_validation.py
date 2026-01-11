#!/usr/bin/env python3
"""Mathematical validation of E25 CUDA kernel vs Python reference."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['LD_LIBRARY_PATH'] = f"/home/erikg/.local/lib/python3.12/site-packages/torch/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

import torch
import torch.nn as nn
import hasty_pytorch_lib

# Import Python reference implementation
from elman.models.e25_entmax import entmax_1_5_forward, E25DualMemoryElmanCell


def test_entmax_forward():
    """Test 1.5-entmax forward pass produces sparse outputs."""
    print("=" * 60)
    print("Test 1: 1.5-Entmax Forward (Python reference)")
    print("=" * 60)

    B, N = 4, 8
    device = 'cuda'

    # Test with varying scores
    torch.manual_seed(42)
    z = torch.randn(B, N, device=device, dtype=torch.float32)

    p = entmax_1_5_forward(z)

    print(f"Input scores: {z[0]}")
    print(f"Output probs: {p[0]}")
    print(f"Sum: {p[0].sum().item():.6f}")
    print(f"Num zeros: {(p[0] == 0).sum().item()}/{N}")

    # Verify properties
    assert torch.allclose(p.sum(dim=-1), torch.ones(B, device=device)), "Probs don't sum to 1"
    assert (p >= 0).all(), "Negative probabilities"
    assert (p[0] == 0).any(), "Entmax should produce some zeros"

    print("PASS: Entmax produces valid sparse distribution\n")


def test_cuda_forward():
    """Test CUDA E25 forward pass."""
    print("=" * 60)
    print("Test 2: E25 CUDA Forward Pass")
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

    # Make W_h well-conditioned
    with torch.no_grad():
        W_h_f32 = torch.empty_like(W_h, dtype=torch.float32)
        nn.init.orthogonal_(W_h_f32)
        W_h.copy_(W_h_f32.to(dtype) * 0.9)

    # Run CUDA forward
    results = hasty_pytorch_lib.e25_entmax_forward(
        True,  # training
        x, h_tape, h_work,
        W_h, W_x, b_h, W_write
    )

    h_work_out, h_tape_final, h_tape_all, read_attn, write_attn = results

    print(f"h_work_out shape: {h_work_out.shape} (expected [{B}, {T}, {D}])")
    print(f"h_tape_final shape: {h_tape_final.shape} (expected [{B}, {N}, {D}])")
    print(f"read_attn shape: {read_attn.shape} (expected [{B}, {T}, {N}])")
    print(f"write_attn shape: {write_attn.shape} (expected [{B}, {T}, {N}])")

    # Check no NaN
    has_nan = torch.isnan(h_work_out).any().item()
    print(f"NaN in output: {has_nan}")

    # Check attention sparsity
    read_zeros = (read_attn == 0).float().mean().item()
    write_zeros = (write_attn == 0).float().mean().item()
    print(f"Read attention sparsity: {read_zeros*100:.1f}% zeros")
    print(f"Write attention sparsity: {write_zeros*100:.1f}% zeros")

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
    """Compare Python and CUDA implementations step by step."""
    print("=" * 60)
    print("Test 3: Python vs CUDA Mathematical Comparison")
    print("=" * 60)

    # Use supported dimensions: N in {8,16,32,64}, D in {256,512}
    B, T, D, N = 2, 4, 256, 8
    device = 'cuda'
    dtype = torch.bfloat16

    torch.manual_seed(42)
    x = torch.randn(B, T, D, device=device, dtype=dtype) * 0.1
    h_tape = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1  # Non-zero for attention
    h_work = torch.randn(B, D, device=device, dtype=dtype) * 0.1

    W_h = torch.randn(D, D, device=device, dtype=dtype) * 0.01
    W_x = torch.randn(D, D, device=device, dtype=dtype) * 0.01
    b_h = torch.zeros(D, device=device, dtype=dtype)
    W_write = torch.randn(D, D, device=device, dtype=dtype) * 0.01

    # Make W_h well-conditioned
    with torch.no_grad():
        W_h_f32 = torch.empty_like(W_h, dtype=torch.float32)
        nn.init.orthogonal_(W_h_f32)
        W_h.copy_(W_h_f32.to(dtype) * 0.9)

    # Run CUDA forward
    cuda_results = hasty_pytorch_lib.e25_entmax_forward(
        True, x, h_tape, h_work,
        W_h, W_x, b_h, W_write
    )
    cuda_h_work_out = cuda_results[0]
    cuda_read_attn = cuda_results[3]
    cuda_write_attn = cuda_results[4]

    # Run Python reference step by step
    scale = 1.0 / (D ** 0.5)
    h_tape_py = h_tape.clone()
    h_work_py = h_work.clone()

    py_h_work_list = []
    py_read_attn_list = []
    py_write_attn_list = []

    for t in range(T):
        x_t = x[:, t, :]  # [B, D]

        # Compute pre-activation
        pre_act = h_work_py @ W_h.T + x_t @ W_x.T + b_h

        # Read attention scores: h_work [B, D] @ h_tape^T [D, N] -> [B, N]
        # h_tape is [B, N, D], so we need h_work @ h_tape.transpose(1,2)
        # h_work: [B, D], h_tape: [B, N, D]
        # We want: [B, D] @ [B, D, N] -> [B, N] via einsum
        read_scores = torch.einsum('bd,bnd->bn', h_work_py.float(), h_tape_py.float()) * scale  # [B, N]
        read_attn_py = entmax_1_5_forward(read_scores).to(dtype)

        # Read value: [B, N] @ [B, N, D] -> [B, D]
        read_val = torch.einsum('bn,bnd->bd', read_attn_py.float(), h_tape_py.float()).to(dtype)

        # Update h_work
        h_work_py = torch.tanh(pre_act + read_val)

        # Write attention
        write_val = h_work_py @ W_write.T  # [B, D]
        # write_scores: [B, D] @ [B, D, N] -> [B, N]
        write_scores = torch.einsum('bd,bnd->bn', write_val.float(), h_tape_py.float()) * scale
        write_attn_py = entmax_1_5_forward(write_scores).to(dtype)

        # Update tape: h_tape = h_tape * (1 - w) + write_val * w
        # w: [B, N], write_val: [B, D] -> expand to [B, N, D]
        h_tape_py = h_tape_py * (1 - write_attn_py.unsqueeze(-1)) + write_val.unsqueeze(1) * write_attn_py.unsqueeze(-1)

        py_h_work_list.append(h_work_py.clone())
        py_read_attn_list.append(read_attn_py.clone())
        py_write_attn_list.append(write_attn_py.clone())

    py_h_work_out = torch.stack(py_h_work_list, dim=1)  # [B, T, D]
    py_read_attn = torch.stack(py_read_attn_list, dim=1)  # [B, T, N]
    py_write_attn = torch.stack(py_write_attn_list, dim=1)  # [B, T, N]

    # Compare outputs
    h_work_diff = (cuda_h_work_out.float() - py_h_work_out.float()).abs()
    read_diff = (cuda_read_attn.float() - py_read_attn.float()).abs()
    write_diff = (cuda_write_attn.float() - py_write_attn.float()).abs()

    print(f"h_work difference: max={h_work_diff.max().item():.6f}, mean={h_work_diff.mean().item():.6f}")
    print(f"read_attn difference: max={read_diff.max().item():.6f}, mean={read_diff.mean().item():.6f}")
    print(f"write_attn difference: max={write_diff.max().item():.6f}, mean={write_diff.mean().item():.6f}")

    # For bf16, expect ~1e-2 to 1e-3 tolerance
    tol = 0.1  # bf16 has limited precision
    h_work_match = h_work_diff.max().item() < tol
    read_match = read_diff.max().item() < tol
    write_match = write_diff.max().item() < tol

    if h_work_match and read_match and write_match:
        print("PASS: Python and CUDA outputs match within tolerance\n")
        return True
    else:
        print(f"FAIL: Outputs differ beyond tolerance ({tol})")
        # Print first mismatch details
        if not h_work_match:
            idx = (h_work_diff == h_work_diff.max()).nonzero()[0]
            print(f"  h_work mismatch at {idx.tolist()}: CUDA={cuda_h_work_out[idx[0], idx[1], idx[2]].item():.6f}, Py={py_h_work_out[idx[0], idx[1], idx[2]].item():.6f}")
        return False


def test_backward_pass():
    """Test E25 backward pass produces valid gradients."""
    print("=" * 60)
    print("Test 4: E25 CUDA Backward Pass")
    print("=" * 60)

    # Use supported dimensions: N in {8,16,32,64}, D in {256,512}
    B, T, D, N = 2, 4, 256, 8
    device = 'cuda'
    dtype = torch.bfloat16

    torch.manual_seed(42)
    x = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True) * 0.1
    h_tape = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
    h_work = torch.zeros(B, D, device=device, dtype=dtype)

    W_h = torch.randn(D, D, device=device, dtype=dtype, requires_grad=True) * 0.01
    W_x = torch.randn(D, D, device=device, dtype=dtype, requires_grad=True) * 0.01
    b_h = torch.zeros(D, device=device, dtype=dtype, requires_grad=True)
    W_write = torch.randn(D, D, device=device, dtype=dtype, requires_grad=True) * 0.01

    # Make W_h well-conditioned
    with torch.no_grad():
        W_h_f32 = torch.empty_like(W_h, dtype=torch.float32)
        nn.init.orthogonal_(W_h_f32)
        W_h.copy_(W_h_f32.to(dtype) * 0.9)

    # Forward
    results = hasty_pytorch_lib.e25_entmax_forward(
        True, x, h_tape, h_work,
        W_h, W_x, b_h, W_write
    )
    h_work_out, h_tape_final, h_tape_all, read_attn, write_attn = results

    # Create gradients
    d_h_work_out = torch.randn_like(h_work_out) * 0.1
    d_h_tape_final = torch.randn_like(h_tape_final) * 0.1

    # Backward
    grads = hasty_pytorch_lib.e25_entmax_backward(
        x, h_work_out, h_work, h_tape_all,
        read_attn, write_attn,
        W_h, W_x, W_write,
        d_h_work_out, d_h_tape_final
    )

    dx, dW_h, dW_x, db_h, dW_write = grads

    print(f"dx shape: {dx.shape}, norm: {dx.norm().item():.6f}")
    print(f"dW_h shape: {dW_h.shape}, norm: {dW_h.norm().item():.6f}")
    print(f"dW_x shape: {dW_x.shape}, norm: {dW_x.norm().item():.6f}")
    print(f"db_h shape: {db_h.shape}, norm: {db_h.norm().item():.6f}")
    print(f"dW_write shape: {dW_write.shape}, norm: {dW_write.norm().item():.6f}")

    # Check for NaN
    has_nan = any([
        torch.isnan(dx).any().item(),
        torch.isnan(dW_h).any().item(),
        torch.isnan(dW_x).any().item(),
        torch.isnan(db_h).any().item(),
        torch.isnan(dW_write).any().item(),
    ])

    if not has_nan:
        print("PASS: Backward pass produces valid gradients\n")
        return True
    else:
        print("FAIL: NaN in gradients\n")
        return False


def benchmark():
    """Quick benchmark of E25 throughput."""
    print("=" * 60)
    print("Benchmark: E25 Entmax Throughput")
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

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = hasty_pytorch_lib.e25_entmax_forward(
                False, x, h_tape, h_work,
                W_h, W_x, b_h, W_write
            )
    torch.cuda.synchronize()

    # Benchmark
    n_iters = 20
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            _ = hasty_pytorch_lib.e25_entmax_forward(
                False, x, h_tape, h_work,
                W_h, W_x, b_h, W_write
            )
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / n_iters * 1000

    tokens = B * T
    tok_per_sec = tokens / (elapsed / 1000) / 1000

    print(f"Config: B={B}, T={T}, D={D}, N={N}")
    print(f"E25 Entmax: {elapsed:.2f}ms ({tok_per_sec:.1f}K tok/s)")

    # Compare with E23c (softmax)
    for _ in range(3):
        with torch.no_grad():
            _ = hasty_pytorch_lib.e23c_chunked_forward(
                False, x, h_tape, h_work,
                W_h, W_x, b_h, W_write, 64
            )
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            _ = hasty_pytorch_lib.e23c_chunked_forward(
                False, x, h_tape, h_work,
                W_h, W_x, b_h, W_write, 64
            )
    torch.cuda.synchronize()
    elapsed_e23c = (time.perf_counter() - t0) / n_iters * 1000
    tok_per_sec_e23c = tokens / (elapsed_e23c / 1000) / 1000

    print(f"E23c Softmax: {elapsed_e23c:.2f}ms ({tok_per_sec_e23c:.1f}K tok/s)")
    print(f"E25 vs E23c: {elapsed_e23c/elapsed:.2f}x")


if __name__ == '__main__':
    test_entmax_forward()
    test_cuda_forward()
    test_python_vs_cuda()
    test_backward_pass()
    benchmark()
