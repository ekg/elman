"""Test E88 multihead backward kernel against register_owned backward.

Both kernels should produce identical gradients given the same inputs.
The multihead kernel uses 4 warps per block (4 heads) for better occupancy,
while register_owned uses 1 warp per block.
"""

import torch
import time
import sys
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, '/home/erikg/elman/elman/cuda')
import hasty_pytorch_lib


def test_correctness(B, T, H, n_state, head_v_dim, has_gate=True, seed=42):
    """Compare multihead vs register_owned backward gradients."""
    torch.manual_seed(seed)
    device = 'cuda:0'

    # H must be divisible by 4 for multihead kernel
    assert H % 4 == 0, f"H={H} must be divisible by 4"

    # Create random inputs
    k = torch.randn(B, T, H, n_state, device=device, dtype=torch.bfloat16) * 0.1
    v = torch.randn(B, T, H, head_v_dim, device=device, dtype=torch.bfloat16) * 0.1
    q = torch.randn(B, T, H, n_state, device=device, dtype=torch.bfloat16) * 0.1
    decay = torch.randn(B, T, H, device=device, dtype=torch.bfloat16) * 0.1
    d_output = torch.randn(B, T, H, head_v_dim, device=device, dtype=torch.bfloat16) * 0.1

    if has_gate:
        g = torch.randn(B, T, H, head_v_dim, device=device, dtype=torch.bfloat16) * 0.1
    else:
        g = torch.empty(0, device=device, dtype=torch.bfloat16)

    # Construct S_cache: checkpoints + Sq_cache
    checkpoint_interval = 16
    num_checkpoints = (T + checkpoint_interval - 1) // checkpoint_interval + 1
    s_checkpoints_size = num_checkpoints * B * H * n_state * head_v_dim
    sq_cache_size = B * T * H * head_v_dim
    cache_size = s_checkpoints_size + sq_cache_size

    # Run the forward pass to get valid S_cache
    S0 = torch.zeros(B, H, n_state, head_v_dim, device=device, dtype=torch.bfloat16)
    output = torch.empty(B, T, H, head_v_dim, device=device, dtype=torch.bfloat16)
    S_cache = torch.empty(cache_size, device=device, dtype=torch.bfloat16)

    g_fwd = g if has_gate else torch.empty(0, device=device, dtype=torch.bfloat16)
    hasty_pytorch_lib.e88_warp_optimized_forward(
        True, k.contiguous(), v.contiguous(), q.contiguous(),
        decay.contiguous(), g_fwd.contiguous(),
        S0.contiguous(), output, S_cache, H, has_gate
    )

    # Segment cache for backward
    cache_entry_size = n_state * head_v_dim + n_state + head_v_dim + 1

    # --- Register-owned backward ---
    d_k_ref = torch.empty_like(k)
    d_v_ref = torch.empty_like(v)
    d_q_ref = torch.empty_like(q)
    d_decay_ref = torch.empty_like(decay)
    d_g_ref = torch.empty_like(g) if has_gate else torch.empty(0, device=device, dtype=torch.bfloat16)
    segment_cache_ref = torch.empty(
        B * H * checkpoint_interval * cache_entry_size,
        dtype=torch.bfloat16, device=device
    )

    hasty_pytorch_lib.e88_register_owned_backward(
        k, v, q, decay, g_fwd,
        S_cache, d_output.contiguous(),
        d_k_ref, d_v_ref, d_q_ref, d_decay_ref, d_g_ref,
        segment_cache_ref, H, has_gate
    )

    # --- Multihead backward ---
    d_k_mh = torch.empty_like(k)
    d_v_mh = torch.empty_like(v)
    d_q_mh = torch.empty_like(q)
    d_decay_mh = torch.empty_like(decay)
    d_g_mh = torch.empty_like(g) if has_gate else torch.empty(0, device=device, dtype=torch.bfloat16)
    segment_cache_mh = torch.empty(
        B * H * checkpoint_interval * cache_entry_size,
        dtype=torch.bfloat16, device=device
    )

    hasty_pytorch_lib.e88_multihead_backward(
        k, v, q, decay, g_fwd,
        S_cache, d_output.contiguous(),
        d_k_mh, d_v_mh, d_q_mh, d_decay_mh, d_g_mh,
        segment_cache_mh, H, has_gate
    )

    # Compare results
    def check(name, ref, test):
        if ref.numel() == 0:
            return True
        max_diff = (ref.float() - test.float()).abs().max().item()
        mean_diff = (ref.float() - test.float()).abs().mean().item()
        ref_norm = ref.float().abs().mean().item()
        rel_err = mean_diff / (ref_norm + 1e-8)
        ok = max_diff < 0.05 and rel_err < 0.01
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.8f}, ref_norm={ref_norm:.6f}, rel_err={rel_err:.6f} [{status}]")
        return ok

    print(f"\nConfig: B={B}, T={T}, H={H}, n_state={n_state}, head_v_dim={head_v_dim}, gate={has_gate}")
    all_pass = True
    all_pass &= check("d_k", d_k_ref, d_k_mh)
    all_pass &= check("d_v", d_v_ref, d_v_mh)
    all_pass &= check("d_q", d_q_ref, d_q_mh)
    all_pass &= check("d_decay", d_decay_ref, d_decay_mh)
    if has_gate:
        all_pass &= check("d_g", d_g_ref, d_g_mh)

    return all_pass


def test_performance(B, T, H, n_state, head_v_dim, has_gate=True, n_iters=100, seed=42):
    """Benchmark multihead vs register_owned backward."""
    torch.manual_seed(seed)
    device = 'cuda:0'

    k = torch.randn(B, T, H, n_state, device=device, dtype=torch.bfloat16) * 0.1
    v = torch.randn(B, T, H, head_v_dim, device=device, dtype=torch.bfloat16) * 0.1
    q = torch.randn(B, T, H, n_state, device=device, dtype=torch.bfloat16) * 0.1
    decay = torch.randn(B, T, H, device=device, dtype=torch.bfloat16) * 0.1
    d_output = torch.randn(B, T, H, head_v_dim, device=device, dtype=torch.bfloat16) * 0.1
    g = torch.randn(B, T, H, head_v_dim, device=device, dtype=torch.bfloat16) * 0.1 if has_gate else torch.empty(0, device=device, dtype=torch.bfloat16)

    checkpoint_interval = 16
    num_checkpoints = (T + checkpoint_interval - 1) // checkpoint_interval + 1
    s_checkpoints_size = num_checkpoints * B * H * n_state * head_v_dim
    sq_cache_size = B * T * H * head_v_dim
    cache_size = s_checkpoints_size + sq_cache_size

    S0 = torch.zeros(B, H, n_state, head_v_dim, device=device, dtype=torch.bfloat16)
    output = torch.empty(B, T, H, head_v_dim, device=device, dtype=torch.bfloat16)
    S_cache = torch.empty(cache_size, device=device, dtype=torch.bfloat16)

    hasty_pytorch_lib.e88_warp_optimized_forward(
        True, k.contiguous(), v.contiguous(), q.contiguous(),
        decay.contiguous(), g.contiguous(),
        S0.contiguous(), output, S_cache, H, has_gate
    )

    cache_entry_size = n_state * head_v_dim + n_state + head_v_dim + 1

    # Benchmark register_owned
    d_k = torch.empty_like(k)
    d_v = torch.empty_like(v)
    d_q = torch.empty_like(q)
    d_decay = torch.empty_like(decay)
    d_g_t = torch.empty_like(g) if has_gate else torch.empty(0, device=device, dtype=torch.bfloat16)
    seg_cache = torch.empty(B * H * checkpoint_interval * cache_entry_size, dtype=torch.bfloat16, device=device)

    # Warmup
    for _ in range(10):
        hasty_pytorch_lib.e88_register_owned_backward(
            k, v, q, decay, g, S_cache, d_output.contiguous(),
            d_k, d_v, d_q, d_decay, d_g_t, seg_cache, H, has_gate
        )
    torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(n_iters):
        hasty_pytorch_lib.e88_register_owned_backward(
            k, v, q, decay, g, S_cache, d_output.contiguous(),
            d_k, d_v, d_q, d_decay, d_g_t, seg_cache, H, has_gate
        )
    torch.cuda.synchronize(device)
    reg_time = (time.perf_counter() - start) / n_iters * 1000

    # Benchmark multihead
    for _ in range(10):
        hasty_pytorch_lib.e88_multihead_backward(
            k, v, q, decay, g, S_cache, d_output.contiguous(),
            d_k, d_v, d_q, d_decay, d_g_t, seg_cache, H, has_gate
        )
    torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(n_iters):
        hasty_pytorch_lib.e88_multihead_backward(
            k, v, q, decay, g, S_cache, d_output.contiguous(),
            d_k, d_v, d_q, d_decay, d_g_t, seg_cache, H, has_gate
        )
    torch.cuda.synchronize(device)
    mh_time = (time.perf_counter() - start) / n_iters * 1000

    speedup = reg_time / mh_time
    print(f"\nPerformance: B={B}, T={T}, H={H}, n={n_state}, v={head_v_dim}")
    print(f"  register_owned: {reg_time:.3f} ms")
    print(f"  multihead:      {mh_time:.3f} ms")
    print(f"  speedup:        {speedup:.2f}x")
    return speedup


if __name__ == '__main__':
    print("=" * 60)
    print("E88 Multihead Backward Kernel Tests")
    print("=" * 60)

    torch.cuda.set_device(0)

    # Correctness tests
    print("\n--- Correctness Tests ---")
    configs = [
        # B, T, H, n_state, head_v_dim, has_gate
        (2, 64, 8, 32, 32, True),     # Small, gated
        (2, 64, 8, 32, 32, False),    # Small, no gate
        (4, 128, 16, 32, 32, True),   # Medium
        (2, 512, 84, 32, 32, True),   # Production-like (H=84, divisible by 4)
        (2, 128, 16, 16, 16, True),   # Small state
        (4, 256, 80, 32, 32, True),   # Large H=80
        (2, 512, 104, 32, 32, True),  # H=104 (optimal config, divisible by 4)
    ]

    all_pass = True
    for B, T, H, n, v, gate in configs:
        try:
            ok = test_correctness(B, T, H, n, v, gate)
            all_pass &= ok
        except Exception as e:
            print(f"  ERROR: {e}")
            all_pass = False

    # Performance tests
    print("\n--- Performance Tests ---")
    perf_configs = [
        (4, 512, 84, 32, 32, True),   # Realistic: 84 heads
        (16, 512, 84, 32, 32, True),  # Larger batch
        (4, 512, 104, 32, 32, True),  # 104 heads (optimal config)
    ]

    for B, T, H, n, v, gate in perf_configs:
        try:
            test_performance(B, T, H, n, v, gate)
        except Exception as e:
            print(f"  ERROR with B={B},T={T},H={H}: {e}")

    print("\n" + "=" * 60)
    if all_pass:
        print("ALL CORRECTNESS TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
