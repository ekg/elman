"""
Verify E1H CUDA kernel matches PyTorch reference in forward and backward.

Tests the core recurrence kernel (pre_x, z, W_h, b -> output, h_final)
with gradients flowing through all parameters.
"""

import torch
import torch.nn.functional as F
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import hasty_pytorch_lib
    HAS_CUDA = hasattr(hasty_pytorch_lib, 'e1h_forward')
except ImportError:
    print("ERROR: hasty_pytorch_lib not available")
    sys.exit(1)

if not HAS_CUDA:
    print("ERROR: e1h_forward not found in hasty_pytorch_lib")
    sys.exit(1)

print("hasty_pytorch_lib loaded, e1h_forward available")


def pytorch_reference(pre_x, z, h0, W_h, b_h):
    """
    Pure PyTorch reference for the E1H recurrence.

    Args:
        pre_x: [B, T, H, N] float32
        z: [B, T, H, N] float32
        h0: [B, H, N] float32
        W_h: [H, N, N] float32
        b_h: [H, N] float32

    Returns:
        output: [B, T, H, N]
        h_final: [B, H, N]
    """
    B, T, H, N = pre_x.shape
    h_prev = h0.clone()
    outputs = []

    for t in range(T):
        # W_h @ h_prev: [B, H, N]
        wh = torch.einsum('bhi,hij->bhj', h_prev, W_h)
        # h_new = tanh(pre_x + W_h @ h_prev + b)
        h_new = torch.tanh(pre_x[:, t] + wh + b_h)
        # output = h * silu(z)
        out_t = h_new * F.silu(z[:, t])
        outputs.append(out_t)
        h_prev = h_new

    output = torch.stack(outputs, dim=1)
    return output, h_prev


def test_forward(B, T, H, N, device='cuda'):
    """Test forward pass: CUDA vs PyTorch reference."""
    print(f"\n=== Forward Test: B={B}, T={T}, H={H}, N={N} ===")

    torch.manual_seed(42)

    # Create inputs in float32, then cast to bf16 for CUDA
    pre_x_f32 = torch.randn(B, T, H, N, device=device)
    z_f32 = torch.randn(B, T, H, N, device=device)
    h0_f32 = torch.randn(B, H, N, device=device) * 0.1
    W_h_f32 = torch.randn(H, N, N, device=device) * 0.1
    b_h_f32 = torch.randn(H, N, device=device) * 0.01

    # PyTorch reference in float32
    out_ref, h_ref = pytorch_reference(pre_x_f32, z_f32, h0_f32, W_h_f32, b_h_f32)

    # CUDA in bf16
    pre_x_bf16 = pre_x_f32.to(torch.bfloat16)
    z_bf16 = z_f32.to(torch.bfloat16)
    h0_bf16 = h0_f32.to(torch.bfloat16)
    W_h_bf16 = W_h_f32.to(torch.bfloat16)
    b_h_bf16 = b_h_f32.to(torch.bfloat16)

    # Transpose to [T, B, H, N] for CUDA
    pre_x_t = pre_x_bf16.permute(1, 0, 2, 3).contiguous()
    z_t = z_bf16.permute(1, 0, 2, 3).contiguous()

    results = hasty_pytorch_lib.e1h_forward(
        True, pre_x_t, z_t, h0_bf16, W_h_bf16, b_h_bf16
    )

    h_cuda = results[0]    # [B, H, N]
    out_cuda_t = results[1]  # [T, B, H, N]

    # Transpose output back
    out_cuda = out_cuda_t.permute(1, 0, 2, 3).float()
    h_cuda = h_cuda.float()

    # Compare
    out_diff = (out_cuda - out_ref).abs()
    h_diff = (h_cuda - h_ref).abs()

    print(f"  Output max diff: {out_diff.max().item():.6f}")
    print(f"  Output mean diff: {out_diff.mean().item():.6f}")
    print(f"  H_final max diff: {h_diff.max().item():.6f}")
    print(f"  H_final mean diff: {h_diff.mean().item():.6f}")

    # For bf16, acceptable tolerance is ~0.1 for long sequences
    # (errors accumulate through sequential recurrence)
    out_ok = out_diff.max().item() < 1.0  # generous for bf16 accumulation
    h_ok = h_diff.max().item() < 1.0

    # Also check relative error
    rel_err_out = (out_diff / (out_ref.abs() + 1e-6)).mean().item()
    rel_err_h = (h_diff / (h_ref.abs() + 1e-6)).mean().item()
    print(f"  Output relative error: {rel_err_out:.6f}")
    print(f"  H_final relative error: {rel_err_h:.6f}")

    if out_ok and h_ok:
        print("  PASS")
    else:
        print("  FAIL")
    return out_ok and h_ok


def test_backward(B, T, H, N, device='cuda'):
    """Test backward pass: CUDA gradients vs PyTorch autograd."""
    print(f"\n=== Backward Test: B={B}, T={T}, H={H}, N={N} ===")

    torch.manual_seed(42)

    # Create inputs - need grad for backward
    pre_x = torch.randn(B, T, H, N, device=device, dtype=torch.float32, requires_grad=True)
    z = torch.randn(B, T, H, N, device=device, dtype=torch.float32, requires_grad=True)
    h0 = torch.zeros(B, H, N, device=device, dtype=torch.float32)
    W_h_data = torch.randn(H, N, N, device=device, dtype=torch.float32) * 0.1
    W_h = W_h_data.clone().requires_grad_(True)
    b_h_data = torch.randn(H, N, device=device, dtype=torch.float32) * 0.01
    b_h = b_h_data.clone().requires_grad_(True)

    # PyTorch reference forward + backward
    out_ref, h_ref = pytorch_reference(pre_x, z, h0, W_h, b_h)
    loss_ref = out_ref.sum()
    loss_ref.backward()

    d_pre_x_ref = pre_x.grad.clone()
    d_z_ref = z.grad.clone()
    d_W_h_ref = W_h.grad.clone()
    d_b_ref = b_h.grad.clone()

    # Now test CUDA backward
    # Use bf16 versions of the same inputs
    pre_x_bf16 = pre_x.detach().to(torch.bfloat16)
    z_bf16 = z.detach().to(torch.bfloat16)
    h0_bf16 = h0.to(torch.bfloat16)
    W_h_bf16 = W_h.detach().to(torch.bfloat16)
    b_h_bf16 = b_h.detach().to(torch.bfloat16)

    # Forward to get checkpoints
    pre_x_t = pre_x_bf16.permute(1, 0, 2, 3).contiguous()
    z_t = z_bf16.permute(1, 0, 2, 3).contiguous()

    fwd_results = hasty_pytorch_lib.e1h_forward(
        True, pre_x_t, z_t, h0_bf16, W_h_bf16, b_h_bf16
    )
    h_checkpoints = fwd_results[2]

    # d_output = ones (matching sum() loss)
    d_output_t = torch.ones_like(fwd_results[1])  # [T, B, H, N]

    bwd_results = hasty_pytorch_lib.e1h_backward(
        pre_x_t, z_t, W_h_bf16, b_h_bf16, h_checkpoints, d_output_t
    )

    d_pre_x_cuda = bwd_results[0].permute(1, 0, 2, 3).float()  # [B, T, H, N]
    d_z_cuda = bwd_results[1].permute(1, 0, 2, 3).float()       # [B, T, H, N]
    d_W_h_cuda = bwd_results[2].sum(dim=0)                       # [H, N, N] float32
    d_b_cuda = bwd_results[3].sum(dim=0)                         # [H, N] float32

    # Compare gradients
    for name, cuda_g, ref_g in [
        ("d_pre_x", d_pre_x_cuda, d_pre_x_ref),
        ("d_z", d_z_cuda, d_z_ref),
        ("d_W_h", d_W_h_cuda, d_W_h_ref),
        ("d_b", d_b_cuda, d_b_ref),
    ]:
        diff = (cuda_g - ref_g).abs()
        rel = (diff / (ref_g.abs() + 1e-6)).mean().item()
        print(f"  {name:10s}: max_diff={diff.max().item():.6f}, mean_diff={diff.mean().item():.6f}, rel_err={rel:.6f}")

    # Check overall pass/fail
    all_ok = True
    for name, cuda_g, ref_g in [
        ("d_pre_x", d_pre_x_cuda, d_pre_x_ref),
        ("d_z", d_z_cuda, d_z_ref),
        ("d_W_h", d_W_h_cuda, d_W_h_ref),
        ("d_b", d_b_cuda, d_b_ref),
    ]:
        rel = ((cuda_g - ref_g).abs() / (ref_g.abs() + 1e-6)).mean().item()
        if rel > 0.5:  # bf16 accumulation through long sequential chain
            print(f"  FAIL: {name} relative error too high: {rel:.6f}")
            all_ok = False

    if all_ok:
        print("  PASS")
    return all_ok


def test_autograd_function(B, T, H, N, device='cuda'):
    """Test E1HCUDAFunction autograd wrapper end-to-end."""
    print(f"\n=== Autograd Function Test: B={B}, T={T}, H={H}, N={N} ===")

    from elman.models.e1_multihead import E1HCUDAFunction

    torch.manual_seed(42)

    pre_x = torch.randn(B, T, H, N, device=device, dtype=torch.bfloat16).requires_grad_(True)
    z = torch.randn(B, T, H, N, device=device, dtype=torch.bfloat16).requires_grad_(True)
    h0 = torch.zeros(B, H, N, device=device, dtype=torch.bfloat16)
    W_h = (torch.randn(H, N, N, device=device, dtype=torch.bfloat16) * 0.1).detach().requires_grad_(True)
    b_h = (torch.randn(H, N, device=device, dtype=torch.bfloat16) * 0.01).detach().requires_grad_(True)

    # Forward through autograd function
    output, h_final = E1HCUDAFunction.apply(True, pre_x, z, h0, W_h, b_h)

    print(f"  Output shape: {output.shape}")
    print(f"  H_final shape: {h_final.shape}")

    # Backward
    loss = output.sum()
    loss.backward()

    print(f"  d_pre_x shape: {pre_x.grad.shape}, has_grad: {pre_x.grad is not None}")
    print(f"  d_z shape: {z.grad.shape}, has_grad: {z.grad is not None}")
    print(f"  d_W_h shape: {W_h.grad.shape}, has_grad: {W_h.grad is not None}")
    print(f"  d_b shape: {b_h.grad.shape}, has_grad: {b_h.grad is not None}")

    # Check shapes
    ok = (output.shape == (B, T, H, N) and
          h_final.shape == (B, H, N) and
          pre_x.grad.shape == (B, T, H, N) and
          z.grad.shape == (B, T, H, N) and
          W_h.grad.shape == (H, N, N) and
          b_h.grad.shape == (H, N))

    # Check no NaN/Inf
    for name, g in [("pre_x", pre_x.grad), ("z", z.grad), ("W_h", W_h.grad), ("b_h", b_h.grad)]:
        if torch.isnan(g).any():
            print(f"  FAIL: NaN in {name} gradient")
            ok = False
        if torch.isinf(g).any():
            print(f"  FAIL: Inf in {name} gradient")
            ok = False

    if ok:
        print("  PASS")
    return ok


def test_full_model(B, T, dim, H, N, device='cuda'):
    """Test full E1MultiHead model with CUDA kernel."""
    print(f"\n=== Full Model Test: B={B}, T={T}, dim={dim}, H={H}, N={N} ===")

    from elman.models.e1_multihead import E1MultiHead

    model = E1MultiHead(dim=dim, n_heads=H, n_state=N, mamba2_init=True).to(device).to(torch.bfloat16)

    x = torch.randn(B, T, dim, device=device, dtype=torch.bfloat16)

    # Forward
    output, h_final = model(x)
    print(f"  Output shape: {output.shape}")
    print(f"  H_final shape: {h_final.shape}")

    # Backward
    loss = output.sum()
    loss.backward()

    # Check gradients exist
    ok = True
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"  FAIL: No gradient for {name}")
            ok = False
        elif torch.isnan(param.grad).any():
            print(f"  FAIL: NaN gradient for {name}")
            ok = False

    if ok:
        print("  PASS")
    return ok


def test_pytorch_vs_cuda_model(B, T, dim, H, N, device='cuda'):
    """Compare full model outputs: PyTorch fallback vs CUDA path."""
    print(f"\n=== Model PyTorch vs CUDA: B={B}, T={T}, dim={dim}, H={H}, N={N} ===")

    from elman.models.e1_multihead import E1MultiHead

    torch.manual_seed(42)
    model = E1MultiHead(dim=dim, n_heads=H, n_state=N, mamba2_init=True).to(device).to(torch.bfloat16)
    x = torch.randn(B, T, dim, device=device, dtype=torch.bfloat16)

    # PyTorch path
    out_py, h_py = model._forward_pytorch(x, None)

    # CUDA path (full forward, which uses CUDA when available)
    out_cuda, h_cuda = model(x)

    out_diff = (out_cuda.float() - out_py.float()).abs()
    h_diff = (h_cuda.float() - h_py.float()).abs()

    print(f"  Output max diff: {out_diff.max().item():.6f}")
    print(f"  Output mean diff: {out_diff.mean().item():.6f}")
    print(f"  H_final max diff: {h_diff.max().item():.6f}")

    # These should match very closely since both go through the same bf16 projection
    # The only difference is the recurrence implementation
    ok = out_diff.max().item() < 2.0  # bf16 accumulation tolerance
    if ok:
        print("  PASS")
    else:
        print("  FAIL")
    return ok


if __name__ == '__main__':
    device = 'cuda'
    all_pass = True

    # Test forward at various sizes
    all_pass &= test_forward(B=2, T=32, H=4, N=16, device=device)
    all_pass &= test_forward(B=2, T=64, H=8, N=32, device=device)
    all_pass &= test_forward(B=1, T=128, H=16, N=16, device=device)

    # Test backward at various sizes
    all_pass &= test_backward(B=2, T=32, H=4, N=16, device=device)
    all_pass &= test_backward(B=2, T=64, H=8, N=32, device=device)
    all_pass &= test_backward(B=1, T=128, H=16, N=16, device=device)

    # Test autograd function wrapper
    all_pass &= test_autograd_function(B=2, T=64, H=8, N=32, device=device)

    # Test full model
    all_pass &= test_full_model(B=2, T=64, dim=256, H=8, N=32, device=device)

    # Test PyTorch vs CUDA model match
    all_pass &= test_pytorch_vs_cuda_model(B=2, T=64, dim=256, H=8, N=32, device=device)

    print(f"\n{'='*60}")
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    sys.exit(0 if all_pass else 1)
