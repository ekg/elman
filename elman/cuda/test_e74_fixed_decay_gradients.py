"""
E74 Fixed Decay Gradient Tests

Tests:
1. Forward pass equivalence (CUDA vs Python) in both fp32 and bf16
2. Backward pass gradient equivalence

Tolerances:
- bf16: 1% relative tolerance
- fp32: 0.1% relative tolerance
"""

import torch
import torch.nn.functional as F
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.e74_fixed_decay import E74FixedDecayCell, E74_FIXED_DECAY_CUDA_AVAILABLE


def e74_python_forward(x, S, W_kvq, alpha):
    """Reference Python implementation for E74 Fixed Decay forward pass.

    S' = alpha * S + outer(v - S @ k_norm, k_norm)
    output = (S' @ q) * silu(S' @ q)
    """
    T, B, D = x.shape
    n = S.shape[1]

    x_flat = x.reshape(T * B, D)
    all_proj = (x_flat @ W_kvq.T).reshape(T, B, 3 * n)
    k_all = all_proj[:, :, :n]
    v_all = all_proj[:, :, n:2*n]
    q_all = all_proj[:, :, 2*n:]

    outputs = []

    for t in range(T):
        k = k_all[t]
        v = v_all[t]
        q = q_all[t]

        # Normalize k
        k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

        # Delta rule with fixed decay
        retrieved = torch.einsum('bij,bj->bi', S, k_norm)
        delta = v - retrieved
        S = alpha * S + torch.einsum('bi,bj->bij', delta, k_norm)

        # Output with self-gating
        Sq = torch.einsum('bij,bj->bi', S, q)
        out = Sq * F.silu(Sq)
        outputs.append(out)

    output = torch.stack(outputs, dim=0)
    return output, S


def test_forward_equivalence_fp32():
    """Test forward pass equivalence between CUDA and Python (FP32)."""
    print("Testing E74 Fixed Decay forward pass (FP32)...")

    torch.manual_seed(42)
    T, B, D = 16, 4, 256
    n_state = 32

    x = torch.randn(T, B, D, device='cuda', dtype=torch.float32)
    S0 = torch.randn(B, n_state, n_state, device='cuda', dtype=torch.float32) * 0.1
    W_kvq = torch.randn(3 * n_state, D, device='cuda', dtype=torch.float32) * 0.1
    alpha = 0.9

    # Python reference
    output_py, S_py = e74_python_forward(x, S0.clone(), W_kvq, alpha)

    # CUDA implementation
    cell = E74FixedDecayCell(D, n_state=n_state, alpha=alpha, use_cuda=True)
    cell.W_kvq.data = W_kvq
    cell = cell.cuda().float()
    cell.eval()

    with torch.no_grad():
        output_cuda, S_cuda = cell(x, S0.clone())

    # Compare
    output_diff = (output_cuda - output_py).abs()
    S_diff = (S_cuda - S_py).abs()

    output_rel_err = output_diff / (output_py.abs() + 1e-8)
    S_rel_err = S_diff / (S_py.abs() + 1e-8)

    max_output_rel_err = output_rel_err.max().item()
    max_S_rel_err = S_rel_err.max().item()

    print(f"  Max output relative error: {max_output_rel_err:.6f}")
    print(f"  Max S relative error: {max_S_rel_err:.6f}")

    # FP32 tolerance: 0.1%
    if max_output_rel_err < 0.001 and max_S_rel_err < 0.001:
        print("  PASSED!")
        return True
    else:
        print(f"  FAILED: output_err={max_output_rel_err}, S_err={max_S_rel_err}")
        return False


def test_forward_equivalence_bf16():
    """Test forward pass equivalence between CUDA and Python (BF16)."""
    print("Testing E74 Fixed Decay forward pass (BF16)...")

    torch.manual_seed(42)
    T, B, D = 16, 4, 256
    n_state = 32

    x = torch.randn(T, B, D, device='cuda', dtype=torch.bfloat16)
    S0 = torch.randn(B, n_state, n_state, device='cuda', dtype=torch.bfloat16) * 0.1
    W_kvq = torch.randn(3 * n_state, D, device='cuda', dtype=torch.bfloat16) * 0.1
    alpha = 0.9

    # Python reference (in float32 for accuracy)
    output_py, S_py = e74_python_forward(
        x.float(), S0.float().clone(), W_kvq.float(), alpha
    )
    output_py = output_py.bfloat16()
    S_py = S_py.bfloat16()

    # CUDA implementation
    cell = E74FixedDecayCell(D, n_state=n_state, alpha=alpha, use_cuda=True)
    cell.W_kvq.data = W_kvq
    cell = cell.cuda().bfloat16()
    cell.eval()

    with torch.no_grad():
        output_cuda, S_cuda = cell(x, S0.clone())

    # Compare
    output_diff = (output_cuda.float() - output_py.float()).abs()
    S_diff = (S_cuda.float() - S_py.float()).abs()

    output_rel_err = output_diff / (output_py.float().abs() + 1e-6)
    S_rel_err = S_diff / (S_py.float().abs() + 1e-6)

    max_output_rel_err = output_rel_err.max().item()
    max_S_rel_err = S_rel_err.max().item()

    print(f"  Max output relative error: {max_output_rel_err:.6f}")
    print(f"  Max S relative error: {max_S_rel_err:.6f}")

    # BF16 tolerance: 1%
    if max_output_rel_err < 0.01 and max_S_rel_err < 0.01:
        print("  PASSED!")
        return True
    else:
        print(f"  FAILED: output_err={max_output_rel_err}, S_err={max_S_rel_err}")
        return False


def test_backward_gradients_fp32():
    """Test backward pass gradient equivalence (FP32)."""
    print("Testing E74 Fixed Decay backward gradients (FP32)...")

    torch.manual_seed(42)
    T, B, D = 8, 2, 128
    n_state = 16

    # Create inputs with gradients
    x = torch.randn(T, B, D, device='cuda', dtype=torch.float32, requires_grad=True)
    S0 = torch.randn(B, n_state, n_state, device='cuda', dtype=torch.float32) * 0.1
    W_kvq = torch.randn(3 * n_state, D, device='cuda', dtype=torch.float32, requires_grad=True) * 0.1
    alpha = 0.9

    # CUDA implementation
    cell = E74FixedDecayCell(D, n_state=n_state, alpha=alpha, use_cuda=True)
    cell.W_kvq.data = W_kvq.clone()
    cell.W_kvq.requires_grad_(True)
    cell = cell.cuda().float()
    cell.train()

    output_cuda, _ = cell(x, S0.clone())
    loss_cuda = output_cuda.sum()
    loss_cuda.backward()

    grad_x_cuda = x.grad.clone()
    grad_W_cuda = cell.W_kvq.grad.clone()

    # Reset gradients
    x.grad = None
    cell.W_kvq.grad = None

    # Python reference with autograd
    x_py = x.detach().clone().requires_grad_(True)
    W_py = W_kvq.detach().clone().requires_grad_(True)

    S = S0.clone()
    outputs = []
    x_flat = x_py.reshape(T * B, D)
    all_proj = (x_flat @ W_py.T).reshape(T, B, 3 * n_state)

    for t in range(T):
        k = all_proj[t, :, :n_state]
        v = all_proj[t, :, n_state:2*n_state]
        q = all_proj[t, :, 2*n_state:]

        k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

        retrieved = torch.einsum('bij,bj->bi', S, k_norm)
        delta = v - retrieved
        S = alpha * S + torch.einsum('bi,bj->bij', delta, k_norm)

        Sq = torch.einsum('bij,bj->bi', S, q)
        out = Sq * F.silu(Sq)
        outputs.append(out)

    output_py = torch.stack(outputs, dim=0)
    loss_py = output_py.sum()
    loss_py.backward()

    grad_x_py = x_py.grad
    grad_W_py = W_py.grad

    # Compare gradients
    grad_x_diff = (grad_x_cuda - grad_x_py).abs()
    grad_W_diff = (grad_W_cuda - grad_W_py).abs()

    grad_x_rel_err = grad_x_diff / (grad_x_py.abs() + 1e-8)
    grad_W_rel_err = grad_W_diff / (grad_W_py.abs() + 1e-8)

    max_grad_x_rel_err = grad_x_rel_err.max().item()
    max_grad_W_rel_err = grad_W_rel_err.max().item()

    print(f"  Max grad_x relative error: {max_grad_x_rel_err:.6f}")
    print(f"  Max grad_W relative error: {max_grad_W_rel_err:.6f}")

    # FP32 tolerance: 0.1%
    if max_grad_x_rel_err < 0.001 and max_grad_W_rel_err < 0.001:
        print("  PASSED!")
        return True
    else:
        print(f"  FAILED: grad_x_err={max_grad_x_rel_err}, grad_W_err={max_grad_W_rel_err}")
        return False


def test_backward_gradients_bf16():
    """Test backward pass gradient equivalence (BF16)."""
    print("Testing E74 Fixed Decay backward gradients (BF16)...")

    torch.manual_seed(42)
    T, B, D = 8, 2, 128
    n_state = 16

    # Create inputs with gradients
    x = torch.randn(T, B, D, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    S0 = torch.randn(B, n_state, n_state, device='cuda', dtype=torch.bfloat16) * 0.1
    W_kvq = torch.randn(3 * n_state, D, device='cuda', dtype=torch.bfloat16, requires_grad=True) * 0.1
    alpha = 0.9

    # CUDA implementation
    cell = E74FixedDecayCell(D, n_state=n_state, alpha=alpha, use_cuda=True)
    cell.W_kvq.data = W_kvq.clone()
    cell.W_kvq.requires_grad_(True)
    cell = cell.cuda().bfloat16()
    cell.train()

    output_cuda, _ = cell(x, S0.clone())
    loss_cuda = output_cuda.sum()
    loss_cuda.backward()

    grad_x_cuda = x.grad.clone().float()
    grad_W_cuda = cell.W_kvq.grad.clone().float()

    # Reset gradients
    x.grad = None
    cell.W_kvq.grad = None

    # Python reference with autograd (in float32 for accuracy)
    x_py = x.float().detach().clone().requires_grad_(True)
    W_py = W_kvq.float().detach().clone().requires_grad_(True)

    S = S0.float().clone()
    outputs = []
    x_flat = x_py.reshape(T * B, D)
    all_proj = (x_flat @ W_py.T).reshape(T, B, 3 * n_state)

    for t in range(T):
        k = all_proj[t, :, :n_state]
        v = all_proj[t, :, n_state:2*n_state]
        q = all_proj[t, :, 2*n_state:]

        k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

        retrieved = torch.einsum('bij,bj->bi', S, k_norm)
        delta = v - retrieved
        S = alpha * S + torch.einsum('bi,bj->bij', delta, k_norm)

        Sq = torch.einsum('bij,bj->bi', S, q)
        out = Sq * F.silu(Sq)
        outputs.append(out)

    output_py = torch.stack(outputs, dim=0)
    loss_py = output_py.sum()
    loss_py.backward()

    grad_x_py = x_py.grad
    grad_W_py = W_py.grad

    # Compare gradients
    grad_x_diff = (grad_x_cuda - grad_x_py).abs()
    grad_W_diff = (grad_W_cuda - grad_W_py).abs()

    grad_x_rel_err = grad_x_diff / (grad_x_py.abs() + 1e-6)
    grad_W_rel_err = grad_W_diff / (grad_W_py.abs() + 1e-6)

    max_grad_x_rel_err = grad_x_rel_err.max().item()
    max_grad_W_rel_err = grad_W_rel_err.max().item()

    print(f"  Max grad_x relative error: {max_grad_x_rel_err:.6f}")
    print(f"  Max grad_W relative error: {max_grad_W_rel_err:.6f}")

    # BF16 tolerance: 1%
    if max_grad_x_rel_err < 0.01 and max_grad_W_rel_err < 0.01:
        print("  PASSED!")
        return True
    else:
        print(f"  FAILED: grad_x_err={max_grad_x_rel_err}, grad_W_err={max_grad_W_rel_err}")
        return False


def test_python_fallback():
    """Test Python fallback implementation (no CUDA)."""
    print("Testing E74 Fixed Decay Python fallback...")

    torch.manual_seed(42)
    T, B, D = 8, 2, 128
    n_state = 16

    x = torch.randn(T, B, D, device='cuda', dtype=torch.float32)
    S0 = torch.randn(B, n_state, n_state, device='cuda', dtype=torch.float32) * 0.1
    W_kvq = torch.randn(3 * n_state, D, device='cuda', dtype=torch.float32) * 0.1
    alpha = 0.9

    # Python reference
    output_py, S_py = e74_python_forward(x, S0.clone(), W_kvq, alpha)

    # Cell with CUDA disabled
    cell = E74FixedDecayCell(D, n_state=n_state, alpha=alpha, use_cuda=False)
    cell.W_kvq.data = W_kvq
    cell = cell.cuda().float()
    cell.eval()

    with torch.no_grad():
        output_cell, S_cell = cell(x, S0.clone())

    # Compare
    output_diff = (output_cell - output_py).abs()
    S_diff = (S_cell - S_py).abs()

    max_output_diff = output_diff.max().item()
    max_S_diff = S_diff.max().item()

    print(f"  Max output absolute error: {max_output_diff:.8f}")
    print(f"  Max S absolute error: {max_S_diff:.8f}")

    # Should be essentially identical (floating point precision only)
    if max_output_diff < 1e-5 and max_S_diff < 1e-5:
        print("  PASSED!")
        return True
    else:
        print(f"  FAILED: output_diff={max_output_diff}, S_diff={max_S_diff}")
        return False


def test_learnable_alpha():
    """Test that learnable alpha can train."""
    print("Testing E74 Fixed Decay learnable alpha...")

    torch.manual_seed(42)
    T, B, D = 8, 2, 128
    n_state = 16

    x = torch.randn(T, B, D, device='cuda', dtype=torch.float32)
    S0 = torch.randn(B, n_state, n_state, device='cuda', dtype=torch.float32) * 0.1

    # Cell with learnable alpha
    cell = E74FixedDecayCell(D, n_state=n_state, alpha=0.9, learnable_alpha=True, use_cuda=False)
    cell = cell.cuda().float()
    cell.train()

    initial_alpha = cell.get_alpha().item()
    print(f"  Initial alpha: {initial_alpha:.4f}")

    # Train for a few steps
    optimizer = torch.optim.Adam(cell.parameters(), lr=0.1)

    for i in range(5):
        optimizer.zero_grad()
        output, _ = cell(x, S0.clone())
        loss = output.sum()
        loss.backward()
        optimizer.step()

    final_alpha = cell.get_alpha().item()
    print(f"  Final alpha: {final_alpha:.4f}")

    # Alpha should have changed
    if abs(final_alpha - initial_alpha) > 1e-4:
        print("  PASSED!")
        return True
    else:
        print("  FAILED: Alpha did not change during training")
        return False


def main():
    print("=" * 60)
    print("E74 Fixed Decay Gradient Tests")
    print("=" * 60)
    print(f"CUDA kernel available: {E74_FIXED_DECAY_CUDA_AVAILABLE}")
    print()

    tests = [
        ("Python Fallback", test_python_fallback),
        ("Learnable Alpha", test_learnable_alpha),
    ]

    # Add CUDA tests if available
    if E74_FIXED_DECAY_CUDA_AVAILABLE:
        tests.extend([
            ("Forward FP32", test_forward_equivalence_fp32),
            ("Forward BF16", test_forward_equivalence_bf16),
            ("Backward FP32", test_backward_gradients_fp32),
            ("Backward BF16", test_backward_gradients_bf16),
        ])
    else:
        print("WARNING: CUDA kernel not available. Testing Python fallback only.")
        print()

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
        print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")

    return all(r for _, r in results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
