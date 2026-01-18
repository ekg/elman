"""
E82 Self-Gate Gradient Tests

Tests:
1. Forward pass equivalence (CUDA vs Python) in both fp32 and bf16
2. Backward pass gradient equivalence
3. Gate stability check (gates don't collapse to 0 or 1)

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

from models.e82_self_gate import E82SelfGateCell, E82_CUDA_AVAILABLE


def e82_python_forward(x, S, W_kvqm, alpha, epsilon):
    """Reference Python implementation for E82 forward pass."""
    T, B, D = x.shape
    n = S.shape[1]

    x_flat = x.reshape(T * B, D)
    all_proj = (x_flat @ W_kvqm.T).reshape(T, B, 4 * n)
    k_all = all_proj[:, :, :n]
    v_all = all_proj[:, :, n:2*n]
    q_all = all_proj[:, :, 2*n:3*n]
    m_all = all_proj[:, :, 3*n:]

    outputs = []
    gate_values = []

    for t in range(T):
        k = k_all[t]
        v = v_all[t]
        q = q_all[t]
        m_vec = m_all[t]

        # Normalize k and m
        k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
        m_norm = m_vec / (m_vec.norm(dim=-1, keepdim=True) + 1e-6)

        # Self-gating
        Sm = torch.einsum('bij,bj->bi', S, m_norm)
        gate_logits = torch.einsum('bi,bj->bij', Sm, k_norm) + alpha * S
        G = torch.sigmoid(gate_logits)
        if epsilon > 0:
            G = G + epsilon
        gate_values.append(G.detach().clone())

        # Delta rule
        s_retrieved = torch.einsum('bij,bj->bi', S, k_norm)
        s_delta = v - s_retrieved

        # Update S
        S = G * S + torch.einsum('bi,bj->bij', s_delta, k_norm)

        # Output
        Sq = torch.einsum('bij,bj->bi', S, q)
        out = Sq * F.silu(Sq)
        outputs.append(out)

    output = torch.stack(outputs, dim=0)
    return output, S, gate_values


def test_forward_equivalence_fp32():
    """Test forward pass equivalence between CUDA and Python (FP32)."""
    print("Testing E82 forward pass (FP32)...")

    torch.manual_seed(42)
    T, B, D = 16, 4, 256
    n_state = 32

    x = torch.randn(T, B, D, device='cuda', dtype=torch.float32)
    S0 = torch.randn(B, n_state, n_state, device='cuda', dtype=torch.float32) * 0.1
    W_kvqm = torch.randn(4 * n_state, D, device='cuda', dtype=torch.float32) * 0.1
    alpha = 0.1
    epsilon = 0.0

    # Python reference
    output_py, S_py, _ = e82_python_forward(x, S0.clone(), W_kvqm, alpha, epsilon)

    # CUDA implementation
    cell = E82SelfGateCell(D, n_state=n_state, alpha_init=alpha, epsilon=epsilon, use_cuda=True)
    cell.W_kvqm.data = W_kvqm
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
    assert max_output_rel_err < 0.001, f"FP32 output error too large: {max_output_rel_err}"
    assert max_S_rel_err < 0.001, f"FP32 S error too large: {max_S_rel_err}"

    print("  PASSED!")
    return True


def test_forward_equivalence_bf16():
    """Test forward pass equivalence between CUDA and Python (BF16)."""
    print("Testing E82 forward pass (BF16)...")

    torch.manual_seed(42)
    T, B, D = 16, 4, 256
    n_state = 32

    x = torch.randn(T, B, D, device='cuda', dtype=torch.bfloat16)
    S0 = torch.randn(B, n_state, n_state, device='cuda', dtype=torch.bfloat16) * 0.1
    W_kvqm = torch.randn(4 * n_state, D, device='cuda', dtype=torch.bfloat16) * 0.1
    alpha = 0.1
    epsilon = 0.0

    # Python reference (in float32 for accuracy)
    output_py, S_py, _ = e82_python_forward(
        x.float(), S0.float().clone(), W_kvqm.float(), alpha, epsilon
    )
    output_py = output_py.bfloat16()
    S_py = S_py.bfloat16()

    # CUDA implementation
    cell = E82SelfGateCell(D, n_state=n_state, alpha_init=alpha, epsilon=epsilon, use_cuda=True)
    cell.W_kvqm.data = W_kvqm
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
    assert max_output_rel_err < 0.01, f"BF16 output error too large: {max_output_rel_err}"
    assert max_S_rel_err < 0.01, f"BF16 S error too large: {max_S_rel_err}"

    print("  PASSED!")
    return True


def test_backward_gradients_fp32():
    """Test backward pass gradient equivalence (FP32)."""
    print("Testing E82 backward gradients (FP32)...")

    torch.manual_seed(42)
    T, B, D = 8, 2, 128
    n_state = 16

    # Create inputs with gradients
    x = torch.randn(T, B, D, device='cuda', dtype=torch.float32, requires_grad=True)
    S0 = torch.randn(B, n_state, n_state, device='cuda', dtype=torch.float32) * 0.1
    W_kvqm = torch.randn(4 * n_state, D, device='cuda', dtype=torch.float32, requires_grad=True) * 0.1
    alpha = 0.1
    epsilon = 0.0

    # CUDA implementation
    cell = E82SelfGateCell(D, n_state=n_state, alpha_init=alpha, epsilon=epsilon, use_cuda=True)
    cell.W_kvqm.data = W_kvqm.clone()
    cell.W_kvqm.requires_grad_(True)
    cell = cell.cuda().float()
    cell.train()

    output_cuda, _ = cell(x, S0.clone())
    loss_cuda = output_cuda.sum()
    loss_cuda.backward()

    grad_x_cuda = x.grad.clone()
    grad_W_cuda = cell.W_kvqm.grad.clone()
    grad_alpha_cuda = cell.alpha.grad.clone() if cell.alpha.grad is not None else torch.zeros(1)

    # Reset gradients
    x.grad = None
    cell.W_kvqm.grad = None
    cell.alpha.grad = None

    # Python reference with autograd
    x_py = x.detach().clone().requires_grad_(True)
    W_py = W_kvqm.detach().clone().requires_grad_(True)
    alpha_param = torch.tensor(alpha, device='cuda', dtype=torch.float32, requires_grad=True)

    S = S0.clone()
    outputs = []
    x_flat = x_py.reshape(T * B, D)
    all_proj = (x_flat @ W_py.T).reshape(T, B, 4 * n_state)

    for t in range(T):
        k = all_proj[t, :, :n_state]
        v = all_proj[t, :, n_state:2*n_state]
        q = all_proj[t, :, 2*n_state:3*n_state]
        m = all_proj[t, :, 3*n_state:]

        k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
        m_norm = m / (m.norm(dim=-1, keepdim=True) + 1e-6)

        Sm = torch.einsum('bij,bj->bi', S, m_norm)
        gate_logits = torch.einsum('bi,bj->bij', Sm, k_norm) + alpha_param * S
        G = torch.sigmoid(gate_logits)

        s_retrieved = torch.einsum('bij,bj->bi', S, k_norm)
        s_delta = v - s_retrieved
        S = G * S + torch.einsum('bi,bj->bij', s_delta, k_norm)

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
    assert max_grad_x_rel_err < 0.001, f"FP32 grad_x error too large: {max_grad_x_rel_err}"
    assert max_grad_W_rel_err < 0.001, f"FP32 grad_W error too large: {max_grad_W_rel_err}"

    print("  PASSED!")
    return True


def test_backward_gradients_bf16():
    """Test backward pass gradient equivalence (BF16)."""
    print("Testing E82 backward gradients (BF16)...")

    torch.manual_seed(42)
    T, B, D = 8, 2, 128
    n_state = 16

    # Create inputs with gradients
    x = torch.randn(T, B, D, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    S0 = torch.randn(B, n_state, n_state, device='cuda', dtype=torch.bfloat16) * 0.1
    W_kvqm = torch.randn(4 * n_state, D, device='cuda', dtype=torch.bfloat16, requires_grad=True) * 0.1
    alpha = 0.1
    epsilon = 0.0

    # CUDA implementation
    cell = E82SelfGateCell(D, n_state=n_state, alpha_init=alpha, epsilon=epsilon, use_cuda=True)
    cell.W_kvqm.data = W_kvqm.clone()
    cell.W_kvqm.requires_grad_(True)
    cell = cell.cuda().bfloat16()
    cell.train()

    output_cuda, _ = cell(x, S0.clone())
    loss_cuda = output_cuda.sum()
    loss_cuda.backward()

    grad_x_cuda = x.grad.clone().float()
    grad_W_cuda = cell.W_kvqm.grad.clone().float()

    # Reset gradients
    x.grad = None
    cell.W_kvqm.grad = None

    # Python reference with autograd (in float32 for accuracy)
    x_py = x.float().detach().clone().requires_grad_(True)
    W_py = W_kvqm.float().detach().clone().requires_grad_(True)
    alpha_param = torch.tensor(alpha, device='cuda', dtype=torch.float32, requires_grad=True)

    S = S0.float().clone()
    outputs = []
    x_flat = x_py.reshape(T * B, D)
    all_proj = (x_flat @ W_py.T).reshape(T, B, 4 * n_state)

    for t in range(T):
        k = all_proj[t, :, :n_state]
        v = all_proj[t, :, n_state:2*n_state]
        q = all_proj[t, :, 2*n_state:3*n_state]
        m = all_proj[t, :, 3*n_state:]

        k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
        m_norm = m / (m.norm(dim=-1, keepdim=True) + 1e-6)

        Sm = torch.einsum('bij,bj->bi', S, m_norm)
        gate_logits = torch.einsum('bi,bj->bij', Sm, k_norm) + alpha_param * S
        G = torch.sigmoid(gate_logits)

        s_retrieved = torch.einsum('bij,bj->bi', S, k_norm)
        s_delta = v - s_retrieved
        S = G * S + torch.einsum('bi,bj->bij', s_delta, k_norm)

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
    assert max_grad_x_rel_err < 0.01, f"BF16 grad_x error too large: {max_grad_x_rel_err}"
    assert max_grad_W_rel_err < 0.01, f"BF16 grad_W error too large: {max_grad_W_rel_err}"

    print("  PASSED!")
    return True


def test_gate_stability():
    """Test that gates don't collapse to 0 or 1 (stability check)."""
    print("Testing E82 gate stability...")

    torch.manual_seed(42)
    T, B, D = 32, 4, 256
    n_state = 32

    x = torch.randn(T, B, D, device='cuda', dtype=torch.float32)
    S0 = torch.randn(B, n_state, n_state, device='cuda', dtype=torch.float32) * 0.1
    W_kvqm = torch.randn(4 * n_state, D, device='cuda', dtype=torch.float32) * 0.1
    alpha = 0.1
    epsilon = 0.0

    # Get gate values
    _, _, gate_values = e82_python_forward(x, S0.clone(), W_kvqm, alpha, epsilon)

    # Check gate statistics
    all_gates = torch.stack(gate_values, dim=0)  # [T, B, n, n]
    mean_gate = all_gates.mean().item()
    min_gate = all_gates.min().item()
    max_gate = all_gates.max().item()
    std_gate = all_gates.std().item()

    print(f"  Gate mean: {mean_gate:.4f}")
    print(f"  Gate std: {std_gate:.4f}")
    print(f"  Gate min: {min_gate:.4f}")
    print(f"  Gate max: {max_gate:.4f}")

    # Check that gates are not collapsed
    # Gates should not all be near 0 or 1
    collapsed_to_zero = (all_gates < 0.1).float().mean().item()
    collapsed_to_one = (all_gates > 0.9).float().mean().item()

    print(f"  Fraction near 0 (<0.1): {collapsed_to_zero:.4f}")
    print(f"  Fraction near 1 (>0.9): {collapsed_to_one:.4f}")

    # Allow some extreme values but not majority
    assert collapsed_to_zero < 0.5, f"Too many gates collapsed to 0: {collapsed_to_zero}"
    assert collapsed_to_one < 0.5, f"Too many gates collapsed to 1: {collapsed_to_one}"

    print("  PASSED!")
    return True


def test_gate_stability_with_epsilon():
    """Test gate stability with epsilon skip connection."""
    print("Testing E82 gate stability with epsilon=0.01...")

    torch.manual_seed(42)
    T, B, D = 32, 4, 256
    n_state = 32

    x = torch.randn(T, B, D, device='cuda', dtype=torch.float32)
    S0 = torch.randn(B, n_state, n_state, device='cuda', dtype=torch.float32) * 0.1
    W_kvqm = torch.randn(4 * n_state, D, device='cuda', dtype=torch.float32) * 0.1
    alpha = 0.1
    epsilon = 0.01  # Small skip connection

    # Get gate values
    _, _, gate_values = e82_python_forward(x, S0.clone(), W_kvqm, alpha, epsilon)

    # Check gate statistics
    all_gates = torch.stack(gate_values, dim=0)  # [T, B, n, n]
    min_gate = all_gates.min().item()

    print(f"  Gate min with epsilon: {min_gate:.4f}")

    # With epsilon, minimum gate should be at least epsilon
    assert min_gate >= epsilon - 1e-6, f"Gate min should be >= epsilon: {min_gate} < {epsilon}"

    print("  PASSED!")
    return True


def main():
    print("=" * 60)
    print("E82 Self-Gate Gradient Tests")
    print("=" * 60)
    print(f"CUDA kernel available: {E82_CUDA_AVAILABLE}")
    print()

    if not E82_CUDA_AVAILABLE:
        print("WARNING: CUDA kernel not available. Testing Python fallback only.")

    tests = [
        ("Forward FP32", test_forward_equivalence_fp32),
        ("Forward BF16", test_forward_equivalence_bf16),
        ("Backward FP32", test_backward_gradients_fp32),
        ("Backward BF16", test_backward_gradients_bf16),
        ("Gate Stability", test_gate_stability),
        ("Gate Stability (epsilon)", test_gate_stability_with_epsilon),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"  FAILED: {e}")
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
