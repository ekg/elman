"""
E88 FLA Hybrid Correctness Verification

Tests forward and backward pass consistency between:
1. PyTorch reference implementation
2. CUDA kernel implementation

Run with: python test_e88_correctness.py
"""

import torch
import torch.nn.functional as F
import numpy as np

def e88_forward_reference(k, v, q, decay, S0, use_tanh=True):
    """
    Pure PyTorch reference implementation of E88 forward pass.

    Args:
        k: [T, B, H, n_state] L2 normalized keys
        v: [T, B, H, head_v_dim] values
        q: [T, B, H, n_state] L2 normalized queries
        decay: [T, B, H] exponential decay factors
        S0: [B, H, n_state, head_v_dim] initial state

    Returns:
        S_final: [B, H, n_state, head_v_dim] final state
        output: [T, B, H, head_v_dim] outputs
    """
    T, B, H, n_state = k.shape
    head_v_dim = v.shape[-1]

    S = S0.clone()
    outputs = []

    for t in range(T):
        k_t = k[t]  # [B, H, n_state]
        v_t = v[t]  # [B, H, head_v_dim]
        q_t = q[t]  # [B, H, n_state]
        decay_t = decay[t]  # [B, H]

        # retrieved = S @ k: [B, H, n_state, head_v_dim] @ [B, H, n_state] -> [B, H, head_v_dim]
        retrieved = torch.einsum('bhiv,bhi->bhv', S, k_t)

        # delta = v - retrieved
        delta = v_t - retrieved

        # outer product: delta outer k -> [B, H, n_state, head_v_dim]
        outer = torch.einsum('bhv,bhi->bhiv', delta, k_t)

        # State update: S = tanh(decay * S + outer)
        pre_tanh = decay_t.unsqueeze(-1).unsqueeze(-1) * S + outer
        if use_tanh:
            S = torch.tanh(pre_tanh)
        else:
            S = pre_tanh

        # Output: Sq = S @ q
        Sq = torch.einsum('bhiv,bhi->bhv', S, q_t)
        outputs.append(Sq)

    output = torch.stack(outputs, dim=0)  # [T, B, H, head_v_dim]
    return S, output


def e88_backward_reference(k, v, q, decay, S0, d_output, use_tanh=True):
    """
    Pure PyTorch reference implementation of E88 backward pass.
    Uses autograd for correctness verification.
    """
    k = k.clone().detach().requires_grad_(True)
    v = v.clone().detach().requires_grad_(True)
    q = q.clone().detach().requires_grad_(True)
    decay = decay.clone().detach().requires_grad_(True)
    S0 = S0.clone().detach()  # S0 gradient not needed

    _, output = e88_forward_reference(k, v, q, decay, S0, use_tanh)

    # Backward
    output.backward(d_output)

    return k.grad, v.grad, q.grad, decay.grad


def test_forward_consistency():
    """Test that CUDA forward matches PyTorch reference."""
    print("=" * 60)
    print("Testing E88 Forward Pass Consistency")
    print("=" * 60)

    try:
        from elman.models.e88_fla_hybrid import E88FLAHybridCUDAFunction, E88_NATIVE_CUDA_AVAILABLE
        if not E88_NATIVE_CUDA_AVAILABLE:
            print("CUDA kernel not available, skipping")
            return False
    except ImportError as e:
        print(f"Import error: {e}")
        return False

    torch.manual_seed(42)

    # Test configuration
    T, B, H = 16, 4, 8
    n_state, head_v_dim = 32, 64

    print(f"Config: T={T}, B={B}, H={H}, n_state={n_state}, head_v_dim={head_v_dim}")

    # Generate test data
    k = torch.randn(T, B, H, n_state, device='cuda', dtype=torch.float32)
    v = torch.randn(T, B, H, head_v_dim, device='cuda', dtype=torch.float32)
    q = torch.randn(T, B, H, n_state, device='cuda', dtype=torch.float32)
    decay = torch.sigmoid(torch.randn(T, B, H, device='cuda', dtype=torch.float32))
    S0 = torch.zeros(B, H, n_state, head_v_dim, device='cuda', dtype=torch.float32)

    # L2 normalize k and q
    k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)

    # PyTorch reference (float32)
    S_ref, output_ref = e88_forward_reference(k, v, q, decay, S0)

    # CUDA kernel (bfloat16)
    k_bf16 = k.bfloat16()
    v_bf16 = v.bfloat16()
    q_bf16 = q.bfloat16()
    decay_bf16 = decay.bfloat16()
    S0_bf16 = S0.bfloat16()

    S_cuda, output_cuda = E88FLAHybridCUDAFunction.apply(
        True, k_bf16, v_bf16, q_bf16, decay_bf16, S0_bf16, H
    )

    # Compare (convert CUDA output to float32 for comparison)
    output_cuda_f32 = output_cuda.float()
    S_cuda_f32 = S_cuda.float()

    output_diff = (output_ref - output_cuda_f32).abs()
    S_diff = (S_ref - S_cuda_f32).abs()

    print(f"\nOutput comparison:")
    print(f"  Max diff: {output_diff.max().item():.6f}")
    print(f"  Mean diff: {output_diff.mean().item():.6f}")
    print(f"  Relative error: {(output_diff / (output_ref.abs() + 1e-6)).mean().item():.6f}")

    print(f"\nFinal state comparison:")
    print(f"  Max diff: {S_diff.max().item():.6f}")
    print(f"  Mean diff: {S_diff.mean().item():.6f}")

    # Check if differences are within bf16 precision tolerance
    # bf16 has ~3 decimal digits of precision
    tolerance = 0.05  # Allow 5% relative error for bf16
    rel_error = (output_diff / (output_ref.abs() + 1e-6)).mean().item()

    if rel_error < tolerance:
        print(f"\n✓ PASS: Forward pass matches within tolerance ({rel_error:.4f} < {tolerance})")
        return True
    else:
        print(f"\n✗ FAIL: Forward pass mismatch ({rel_error:.4f} >= {tolerance})")
        return False


def test_backward_consistency():
    """Test that CUDA backward matches PyTorch reference."""
    print("\n" + "=" * 60)
    print("Testing E88 Backward Pass Consistency")
    print("=" * 60)

    try:
        from elman.models.e88_fla_hybrid import E88FLAHybridCUDAFunction, E88_NATIVE_CUDA_AVAILABLE
        if not E88_NATIVE_CUDA_AVAILABLE:
            print("CUDA kernel not available, skipping")
            return False
    except ImportError as e:
        print(f"Import error: {e}")
        return False

    torch.manual_seed(42)

    # Test configuration
    T, B, H = 8, 2, 4
    n_state, head_v_dim = 32, 64

    print(f"Config: T={T}, B={B}, H={H}, n_state={n_state}, head_v_dim={head_v_dim}")

    # Generate test data
    k = torch.randn(T, B, H, n_state, device='cuda', dtype=torch.float32)
    v = torch.randn(T, B, H, head_v_dim, device='cuda', dtype=torch.float32)
    q = torch.randn(T, B, H, n_state, device='cuda', dtype=torch.float32)
    decay = torch.sigmoid(torch.randn(T, B, H, device='cuda', dtype=torch.float32))
    S0 = torch.zeros(B, H, n_state, head_v_dim, device='cuda', dtype=torch.float32)
    d_output = torch.randn(T, B, H, head_v_dim, device='cuda', dtype=torch.float32)

    # L2 normalize k and q
    k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)

    # PyTorch reference gradients
    dk_ref, dv_ref, dq_ref, ddecay_ref = e88_backward_reference(
        k, v, q, decay, S0, d_output
    )

    # CUDA kernel gradients
    k_bf16 = k.bfloat16().clone().detach().requires_grad_(True)
    v_bf16 = v.bfloat16().clone().detach().requires_grad_(True)
    q_bf16 = q.bfloat16().clone().detach().requires_grad_(True)
    decay_bf16 = decay.bfloat16().clone().detach().requires_grad_(True)
    S0_bf16 = S0.bfloat16()

    S_cuda, output_cuda = E88FLAHybridCUDAFunction.apply(
        True, k_bf16, v_bf16, q_bf16, decay_bf16, S0_bf16, H
    )

    output_cuda.backward(d_output.bfloat16())

    dk_cuda = k_bf16.grad.float()
    dv_cuda = v_bf16.grad.float()
    dq_cuda = q_bf16.grad.float()
    ddecay_cuda = decay_bf16.grad.float()

    # Compare gradients
    results = []
    for name, ref, cuda in [
        ("d_k", dk_ref, dk_cuda),
        ("d_v", dv_ref, dv_cuda),
        ("d_q", dq_ref, dq_cuda),
        ("d_decay", ddecay_ref, ddecay_cuda),
    ]:
        diff = (ref - cuda).abs()
        rel_err = (diff / (ref.abs() + 1e-6)).mean().item()
        max_diff = diff.max().item()

        print(f"\n{name}:")
        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Relative error: {rel_err:.6f}")

        results.append(rel_err)

    # Check tolerance
    tolerance = 0.1  # 10% relative error tolerance for bf16 backward
    all_pass = all(r < tolerance for r in results)

    if all_pass:
        print(f"\n✓ PASS: Backward pass matches within tolerance")
        return True
    else:
        print(f"\n✗ FAIL: Backward pass mismatch")
        return False


def test_gradient_flow():
    """Test that gradients flow correctly through the full model."""
    print("\n" + "=" * 60)
    print("Testing E88 Full Model Gradient Flow")
    print("=" * 60)

    try:
        from elman.models import LadderLM
    except ImportError as e:
        print(f"Import error: {e}")
        return False

    torch.manual_seed(42)

    # Create small model
    model = LadderLM(
        vocab_size=256,
        dim=256,
        depth=2,
        level=88,
        n_state=32
    ).cuda().bfloat16()

    print(f"Model params: {model.get_num_params() / 1e6:.2f}M")

    # Test input
    x = torch.randint(0, 256, (2, 32), device='cuda')

    # Forward
    model.train()
    loss = model(x, return_loss=True)
    print(f"Loss: {loss.item():.4f}")

    # Backward
    loss.backward()

    # Check gradients exist and are not NaN/Inf
    grad_stats = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            g = param.grad
            is_finite = torch.isfinite(g).all().item()
            norm = g.norm().item()
            grad_stats.append((name, is_finite, norm))

    all_finite = all(s[1] for s in grad_stats)

    print(f"\nGradient statistics:")
    for name, is_finite, norm in grad_stats[:5]:  # Show first 5
        status = "✓" if is_finite else "✗"
        print(f"  {status} {name}: norm={norm:.4f}")
    print(f"  ... ({len(grad_stats)} total parameters)")

    if all_finite:
        print(f"\n✓ PASS: All gradients are finite")
        return True
    else:
        print(f"\n✗ FAIL: Some gradients are NaN/Inf")
        return False


def test_numerical_gradients():
    """Test gradients with finite differences (slow but thorough)."""
    print("\n" + "=" * 60)
    print("Testing E88 Numerical Gradient Check")
    print("=" * 60)

    try:
        from elman.models.e88_fla_hybrid import E88FLAHybridCUDAFunction, E88_NATIVE_CUDA_AVAILABLE
        if not E88_NATIVE_CUDA_AVAILABLE:
            print("CUDA kernel not available, skipping")
            return False
    except ImportError as e:
        print(f"Import error: {e}")
        return False

    torch.manual_seed(42)

    # Small config for numerical gradient check
    T, B, H = 4, 1, 2
    n_state, head_v_dim = 8, 16

    print(f"Config: T={T}, B={B}, H={H}, n_state={n_state}, head_v_dim={head_v_dim}")

    # Generate test data (use float64 for numerical accuracy)
    k = torch.randn(T, B, H, n_state, device='cuda', dtype=torch.float64, requires_grad=True)
    v = torch.randn(T, B, H, head_v_dim, device='cuda', dtype=torch.float64, requires_grad=True)
    q = torch.randn(T, B, H, n_state, device='cuda', dtype=torch.float64, requires_grad=True)
    decay = torch.sigmoid(torch.randn(T, B, H, device='cuda', dtype=torch.float64, requires_grad=True))
    S0 = torch.zeros(B, H, n_state, head_v_dim, device='cuda', dtype=torch.float64)

    # L2 normalize
    k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
    q_norm = q / (q.norm(dim=-1, keepdim=True) + 1e-6)

    def forward_fn(k, v, q, decay):
        k_n = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
        q_n = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
        _, out = e88_forward_reference(k_n, v, q_n, decay, S0.clone())
        return out.sum()

    # Check gradients
    print("\nRunning torch.autograd.gradcheck (this may take a moment)...")
    try:
        result = torch.autograd.gradcheck(
            forward_fn,
            (k, v, q, decay),
            eps=1e-5,
            atol=1e-3,
            rtol=1e-2,
            raise_exception=False
        )
        if result:
            print("✓ PASS: Numerical gradient check passed")
            return True
        else:
            print("✗ FAIL: Numerical gradient check failed")
            return False
    except Exception as e:
        print(f"✗ FAIL: Gradient check error: {e}")
        return False


if __name__ == "__main__":
    print("E88 FLA Hybrid Correctness Verification")
    print("=" * 60)

    results = {}

    # Run tests
    results['forward'] = test_forward_consistency()
    results['backward'] = test_backward_consistency()
    results['gradient_flow'] = test_gradient_flow()
    # results['numerical'] = test_numerical_gradients()  # Slow, uncomment if needed

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_pass = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        all_pass = all_pass and passed

    if all_pass:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed")

    exit(0 if all_pass else 1)
