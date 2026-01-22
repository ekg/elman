#!/usr/bin/env python3
"""Test E88 fused projection CUDA kernel against Python reference."""

import torch
import torch.nn.functional as F


def test_fused_projection():
    """Compare CUDA fused projection against Python reference."""
    import hasty_pytorch_lib as lib

    # Test configuration matching E88 defaults
    T, B = 64, 4  # sequence length, batch size
    dim = 512
    n_heads = 4
    n_state = 32  # key_dim / n_heads
    head_v_dim = 32  # value_dim / n_heads
    d_conv = 4

    key_dim = n_heads * n_state  # 128
    value_dim = n_heads * head_v_dim  # 128

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Create test inputs
    torch.manual_seed(42)
    x = torch.randn(T, B, dim, device=device, dtype=dtype) * 0.1

    # Combined weight: [2*key_dim + value_dim + n_heads, dim]
    # Layout: [q, k, v, alpha]
    out_dim = 2 * key_dim + value_dim + n_heads
    W_qkva = torch.randn(out_dim, dim, device=device, dtype=dtype) * 0.02

    # Conv weights (groups=dim, so shape is [dim, d_conv] per-channel)
    # For depthwise conv: [channels, d_conv]
    conv_q = torch.randn(key_dim, d_conv, device=device, dtype=dtype) * 0.1
    conv_k = torch.randn(key_dim, d_conv, device=device, dtype=dtype) * 0.1
    conv_v = torch.randn(value_dim, d_conv, device=device, dtype=dtype) * 0.1

    # Decay parameters (float32 for numerical stability)
    A_log = torch.randn(n_heads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(n_heads, device=device, dtype=torch.float32)

    print(f"Testing E88 fused projection:")
    print(f"  T={T}, B={B}, dim={dim}")
    print(f"  n_heads={n_heads}, n_state={n_state}, head_v_dim={head_v_dim}")
    print(f"  key_dim={key_dim}, value_dim={value_dim}")

    # =========================================================================
    # Python reference implementation
    # =========================================================================
    print("\nComputing Python reference...")

    # GEMM: x @ W_qkva^T
    qkva_ref = F.linear(x.view(T * B, dim), W_qkva).view(T, B, out_dim)

    # Split into q, k, v, alpha
    q_raw = qkva_ref[..., :key_dim]  # [T, B, key_dim]
    k_raw = qkva_ref[..., key_dim:2*key_dim]
    v_raw = qkva_ref[..., 2*key_dim:2*key_dim+value_dim]
    alpha_raw = qkva_ref[..., 2*key_dim+value_dim:]  # [T, B, n_heads]

    # Apply depthwise conv (manual implementation matching CUDA kernel)
    # Conv expects [T, B, C] input, outputs [T, B, C]
    # For causal conv: pad with d_conv-1 zeros at the beginning
    def apply_depthwise_conv_causal(x, w, d_conv):
        """Apply causal depthwise conv manually.

        x: [T, B, C]
        w: [C, d_conv] - per-channel conv weights
        """
        T, B, C = x.shape
        output = torch.zeros_like(x)

        # Create padded input (d_conv-1 zeros at beginning)
        x_padded = torch.cat([
            torch.zeros(d_conv - 1, B, C, device=x.device, dtype=x.dtype),
            x
        ], dim=0)  # [T + d_conv - 1, B, C]

        for t in range(T):
            # Take d_conv elements ending at t (inclusive)
            # In padded coordinates: [t, t+1, ..., t+d_conv-1]
            for d in range(d_conv):
                output[t] += x_padded[t + d] * w[:, d]

        return output

    q_conv = apply_depthwise_conv_causal(q_raw, conv_q, d_conv)
    k_conv = apply_depthwise_conv_causal(k_raw, conv_k, d_conv)
    v_conv = apply_depthwise_conv_causal(v_raw, conv_v, d_conv)

    # Apply SiLU
    q_silu = F.silu(q_conv)
    k_silu = F.silu(k_conv)
    v_silu = F.silu(v_conv)

    # Reshape to per-head: [T, B, H, n_state]
    q_per_head = q_silu.view(T, B, n_heads, n_state)
    k_per_head = k_silu.view(T, B, n_heads, n_state)
    v_per_head = v_silu.view(T, B, n_heads, head_v_dim)

    # L2 normalize q and k
    q_ref = q_per_head / (q_per_head.norm(dim=-1, keepdim=True) + 1e-6)
    k_ref = k_per_head / (k_per_head.norm(dim=-1, keepdim=True) + 1e-6)
    v_ref = v_per_head

    # Compute decay from alpha
    # decay = exp(-exp(A_log) * softplus(alpha + dt_bias))
    alpha_per_head = alpha_raw  # [T, B, n_heads]
    x_decay = alpha_per_head.float() + dt_bias
    sp = F.softplus(x_decay)
    g = -A_log.float().exp() * sp
    decay_ref = g.exp().to(dtype)

    print(f"  q_ref shape: {q_ref.shape}")
    print(f"  k_ref shape: {k_ref.shape}")
    print(f"  v_ref shape: {v_ref.shape}")
    print(f"  decay_ref shape: {decay_ref.shape}")

    # =========================================================================
    # CUDA kernel implementation
    # =========================================================================
    print("\nRunning CUDA fused projection...")

    # Reshape conv weights to match CUDA kernel expectation
    # CUDA expects conv weights indexed by (h * n_state + i) * d_conv + d
    # Which is effectively [key_dim, d_conv] row-major
    # Our Python weights are already [key_dim, d_conv], so they match

    results = lib.e88_fused_projection(
        x, W_qkva,
        conv_q, conv_k, conv_v,
        A_log, dt_bias,
        n_heads, key_dim, value_dim,
        d_conv,
        True,  # use_conv
        True,  # use_silu
        True   # use_l2_norm
    )
    q_cuda, k_cuda, v_cuda, decay_cuda = results

    print(f"  q_cuda shape: {q_cuda.shape}")
    print(f"  k_cuda shape: {k_cuda.shape}")
    print(f"  v_cuda shape: {v_cuda.shape}")
    print(f"  decay_cuda shape: {decay_cuda.shape}")

    # =========================================================================
    # Compare results
    # =========================================================================
    print("\nComparing outputs...")

    def compare_tensors(name, ref, cuda, rtol=2e-2, atol=1e-2):
        """Compare tensors and report max difference."""
        ref_f = ref.float()
        cuda_f = cuda.float()
        diff = (ref_f - cuda_f).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        # Check relative difference
        rel_diff = diff / (ref_f.abs() + 1e-6)
        max_rel_diff = rel_diff.max().item()

        matches = torch.allclose(ref_f, cuda_f, rtol=rtol, atol=atol)
        status = "PASS" if matches else "FAIL"

        print(f"  {name}: {status}")
        print(f"    max_abs_diff: {max_diff:.6f}")
        print(f"    mean_abs_diff: {mean_diff:.6f}")
        print(f"    max_rel_diff: {max_rel_diff:.6f}")

        return matches

    all_pass = True
    all_pass &= compare_tensors("q", q_ref, q_cuda)
    all_pass &= compare_tensors("k", k_ref, k_cuda)
    all_pass &= compare_tensors("v", v_ref, v_cuda)
    all_pass &= compare_tensors("decay", decay_ref, decay_cuda)

    print()
    if all_pass:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")

        # Debug: Print sample values
        print("\nDebug - sample values at t=10, b=0, h=0:")
        print(f"  q_ref[10,0,0,:4]: {q_ref[10,0,0,:4].tolist()}")
        print(f"  q_cuda[10,0,0,:4]: {q_cuda[10,0,0,:4].tolist()}")
        print(f"  decay_ref[10,0,:]: {decay_ref[10,0,:].tolist()}")
        print(f"  decay_cuda[10,0,:]: {decay_cuda[10,0,:].tolist()}")

    return all_pass


def benchmark_fused_projection():
    """Benchmark CUDA fused projection vs Python reference."""
    import hasty_pytorch_lib as lib
    import time

    # Test configuration
    T, B = 512, 32  # larger for realistic benchmark
    dim = 1024
    n_heads = 8
    n_state = 64
    head_v_dim = 64
    d_conv = 4

    key_dim = n_heads * n_state
    value_dim = n_heads * head_v_dim

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Create test inputs
    torch.manual_seed(42)
    x = torch.randn(T, B, dim, device=device, dtype=dtype) * 0.1
    out_dim = 2 * key_dim + value_dim + n_heads
    W_qkva = torch.randn(out_dim, dim, device=device, dtype=dtype) * 0.02
    conv_q = torch.randn(key_dim, d_conv, device=device, dtype=dtype) * 0.1
    conv_k = torch.randn(key_dim, d_conv, device=device, dtype=dtype) * 0.1
    conv_v = torch.randn(value_dim, d_conv, device=device, dtype=dtype) * 0.1
    A_log = torch.randn(n_heads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(n_heads, device=device, dtype=torch.float32)

    print(f"\nBenchmark: T={T}, B={B}, dim={dim}, n_heads={n_heads}")
    print(f"  key_dim={key_dim}, value_dim={value_dim}\n")

    # Warmup
    for _ in range(5):
        lib.e88_fused_projection(
            x, W_qkva, conv_q, conv_k, conv_v, A_log, dt_bias,
            n_heads, key_dim, value_dim, d_conv,
            True, True, True
        )
    torch.cuda.synchronize()

    # Benchmark CUDA
    torch.cuda.synchronize()
    start = time.perf_counter()
    n_iter = 100
    for _ in range(n_iter):
        lib.e88_fused_projection(
            x, W_qkva, conv_q, conv_k, conv_v, A_log, dt_bias,
            n_heads, key_dim, value_dim, d_conv,
            True, True, True
        )
    torch.cuda.synchronize()
    cuda_time = (time.perf_counter() - start) / n_iter * 1000

    print(f"CUDA fused projection: {cuda_time:.3f} ms")
    print(f"  Throughput: {T * B / cuda_time * 1000:.0f} tokens/sec")


if __name__ == "__main__":
    test_fused_projection()
    benchmark_fused_projection()
