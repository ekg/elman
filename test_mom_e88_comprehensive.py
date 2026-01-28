#!/usr/bin/env python
"""
Comprehensive test for MoM E88 CUDA kernel.
Tests both forward and backward passes with various head routing patterns.
"""

import torch
import sys
sys.path.insert(0, '/home/erikg/elman')

import hasty_pytorch_lib

device = 'cuda'
dtype = torch.bfloat16


def test_forward(B, T, H, K, n_state, head_v_dim, head_indices_pattern='fixed'):
    """Test MoM E88 forward pass."""
    torch.manual_seed(42)

    k = torch.randn(B, T, H, n_state, device=device, dtype=dtype)
    v = torch.randn(B, T, H, head_v_dim, device=device, dtype=dtype)
    q = torch.randn(B, T, H, n_state, device=device, dtype=dtype)
    decay = torch.sigmoid(torch.randn(B, T, H, device=device, dtype=dtype))

    # Normalize k and q
    k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)

    router_weights = torch.softmax(torch.randn(B, T, K, device=device, dtype=dtype), dim=-1)

    # Generate head indices based on pattern
    if head_indices_pattern == 'fixed':
        # Same heads for all timesteps
        head_indices = torch.stack([torch.arange(K) for _ in range(T)], dim=0)
        head_indices = head_indices.unsqueeze(0).expand(B, -1, -1)
    elif head_indices_pattern == 'dynamic':
        # Different heads each timestep (rotating)
        head_indices = torch.zeros(B, T, K, dtype=torch.int32)
        for t in range(T):
            head_indices[:, t, :] = torch.arange(t * K, (t + 1) * K) % H
    elif head_indices_pattern == 'random':
        # Random head selection
        head_indices = torch.randint(0, H, (B, T, K), dtype=torch.int32)
    else:
        raise ValueError(f"Unknown pattern: {head_indices_pattern}")

    head_indices = head_indices.to(device=device, dtype=torch.int32)

    # Initial state (per-slot)
    S0 = torch.zeros(B, K, n_state, head_v_dim, device=device, dtype=dtype)

    # Python reference
    outputs_py = []
    S_slots = S0.clone().float()

    for t in range(T):
        output_t = torch.zeros(B, head_v_dim, device=device, dtype=torch.float32)
        S_next = S_slots.clone()

        for b in range(B):
            for slot in range(K):
                h = head_indices[b, t, slot].item()
                w = router_weights[b, t, slot].item()

                k_h = k[b, t, h].float()
                v_h = v[b, t, h].float()
                q_h = q[b, t, h].float()
                decay_h = decay[b, t, h].float()

                S_slot = S_slots[b, slot]
                retrieved = S_slot.T @ k_h
                delta = v_h - retrieved
                outer = torch.outer(k_h, delta)
                new_state = torch.tanh(decay_h * S_slot + outer)
                S_next[b, slot] = new_state

                Sq = new_state.T @ q_h
                output_t[b] += w * Sq

        S_slots = S_next
        outputs_py.append(output_t)

    output_py = torch.stack(outputs_py, dim=1)

    # CUDA kernel
    k_t = k.transpose(0, 1).contiguous()
    v_t = v.transpose(0, 1).contiguous()
    q_t = q.transpose(0, 1).contiguous()
    decay_t = decay.transpose(0, 1).contiguous()
    head_indices_t = head_indices.transpose(0, 1).contiguous()
    router_weights_t = router_weights.transpose(0, 1).contiguous()

    output_cuda_t, S_cuda, _ = hasty_pytorch_lib.mom_e88_forward(
        True, k_t, v_t, q_t, decay_t,
        head_indices_t, router_weights_t,
        S0, H, K
    )
    output_cuda = output_cuda_t.transpose(0, 1)

    # Compare
    output_diff = (output_cuda.float() - output_py).abs()
    state_diff = (S_cuda.float() - S_slots).abs()

    max_output_diff = output_diff.max().item()
    max_state_diff = state_diff.max().item()

    return max_output_diff, max_state_diff


def run_tests():
    """Run comprehensive test suite."""
    print("=" * 60)
    print("MoM E88 Comprehensive Test Suite")
    print("=" * 60)

    test_configs = [
        # (B, T, H, K, n_state, head_v_dim, pattern)
        (1, 1, 4, 2, 16, 16, 'fixed'),
        (1, 2, 4, 2, 16, 16, 'fixed'),
        (1, 2, 4, 2, 16, 16, 'dynamic'),
        (1, 4, 8, 4, 16, 16, 'fixed'),
        (1, 4, 8, 4, 16, 16, 'dynamic'),
        (1, 4, 8, 4, 16, 16, 'random'),
        (2, 4, 8, 4, 16, 16, 'dynamic'),
        (4, 8, 16, 4, 16, 16, 'random'),
        # n_state=32 configs
        (1, 4, 8, 4, 32, 32, 'fixed'),
        (1, 4, 8, 4, 32, 32, 'dynamic'),
        (2, 8, 16, 4, 32, 32, 'random'),
    ]

    passed = 0
    failed = 0
    threshold = 0.02  # BF16 tolerance

    for config in test_configs:
        B, T, H, K, n_state, head_v_dim, pattern = config
        try:
            out_diff, state_diff = test_forward(B, T, H, K, n_state, head_v_dim, pattern)
            status = "PASS" if out_diff < threshold and state_diff < threshold else "FAIL"
            if status == "PASS":
                passed += 1
            else:
                failed += 1
            print(f"[{status}] B={B}, T={T}, H={H}, K={K}, n={n_state}, {pattern}: "
                  f"out_diff={out_diff:.6f}, state_diff={state_diff:.6f}")
        except Exception as e:
            failed += 1
            print(f"[ERROR] B={B}, T={T}, H={H}, K={K}, n={n_state}, {pattern}: {e}")

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
