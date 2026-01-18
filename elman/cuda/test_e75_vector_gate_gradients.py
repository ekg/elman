"""
E75 Vector Gate Gradient Tests

Tests:
1. Forward pass equivalence (CUDA vs Python) in both fp32 and bf16
2. Backward pass gradient equivalence
3. Gate behavior check

Mathematical definition:
    g = sigmoid(W_beta @ x + b_beta)  # n-dimensional gate vector
    S' = diag(g) * S + outer(v - S @ k_norm, k_norm)
    output = (S' @ q) * silu(S' @ q)

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

from models.e75_vector_gate import E75VectorGateCell, E75_VECTOR_GATE_CUDA_AVAILABLE


def e75_python_forward(x, S, W_k, W_v, W_q, W_beta, b_beta):
    """Reference Python implementation for E75 Vector Gate forward pass."""
    T, B, D = x.shape
    n = S.shape[1]

    x_flat = x.reshape(T * B, D)
    k_all = (x_flat @ W_k.T).reshape(T, B, n)
    v_all = (x_flat @ W_v.T).reshape(T, B, n)
    q_all = (x_flat @ W_q.T).reshape(T, B, n)
    g_all = torch.sigmoid((x_flat @ W_beta.T + b_beta).reshape(T, B, n))

    outputs = []
    gate_values = []

    for t in range(T):
        k = k_all[t]
        v = v_all[t]
        q = q_all[t]
        g = g_all[t]  # [B, n] - per-row gate

        gate_values.append(g.detach().clone())

        # Normalize k
        k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

        # Retrieve from memory
        retrieved = torch.einsum('bij,bj->bi', S, k_norm)

        # Delta update with per-row decay
        delta = v - retrieved
        outer = torch.einsum('bi,bj->bij', delta, k_norm)

        # Row-wise decay: S = diag(g) * S + outer
        # diag(g) * S means S[i,:] *= g[i]
        S = g.unsqueeze(-1) * S + outer

        # Self-gating output
        Sq = torch.einsum('bij,bj->bi', S, q)
        out = Sq * F.silu(Sq)
        outputs.append(out)

    output = torch.stack(outputs, dim=0)
    return output, S, gate_values


def test_forward_equivalence_fp32():
    """Test forward pass equivalence between CUDA and Python (FP32)."""
    print("Testing E75 Vector Gate forward pass (FP32)...")

    if not E75_VECTOR_GATE_CUDA_AVAILABLE:
        print("  SKIPPED: CUDA kernel not available")
        return True

    torch.manual_seed(42)
    T, B, D = 16, 4, 256
    n_state = 32

    x = torch.randn(T, B, D, device='cuda', dtype=torch.float32)
    S0 = torch.randn(B, n_state, n_state, device='cuda', dtype=torch.float32) * 0.1
    W_k = torch.randn(n_state, D, device='cuda', dtype=torch.float32) * 0.1
    W_v = torch.randn(n_state, D, device='cuda', dtype=torch.float32) * 0.1
    W_q = torch.randn(n_state, D, device='cuda', dtype=torch.float32) * 0.1
    W_beta = torch.randn(n_state, D, device='cuda', dtype=torch.float32) * 0.1
    b_beta = torch.full((n_state,), 2.0, device='cuda', dtype=torch.float32)

    # Python reference
    output_py, S_py, _ = e75_python_forward(x, S0.clone(), W_k, W_v, W_q, W_beta, b_beta)

    # CUDA implementation
    cell = E75VectorGateCell(D, n_state=n_state, use_cuda=True)
    cell.W_k.data = W_k.clone()
    cell.W_v.data = W_v.clone()
    cell.W_q.data = W_q.clone()
    cell.W_beta.data = W_beta.clone()
    cell.b_beta.data = b_beta.clone()
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

    print("  PASS")
    return True


def test_forward_equivalence_bf16():
    """Test forward pass equivalence between CUDA and Python (BF16)."""
    print("Testing E75 Vector Gate forward pass (BF16)...")

    if not E75_VECTOR_GATE_CUDA_AVAILABLE:
        print("  SKIPPED: CUDA kernel not available")
        return True

    torch.manual_seed(42)
    T, B, D = 16, 4, 256
    n_state = 32

    x = torch.randn(T, B, D, device='cuda', dtype=torch.bfloat16)
    S0 = torch.randn(B, n_state, n_state, device='cuda', dtype=torch.bfloat16) * 0.1
    W_k = torch.randn(n_state, D, device='cuda', dtype=torch.bfloat16) * 0.1
    W_v = torch.randn(n_state, D, device='cuda', dtype=torch.bfloat16) * 0.1
    W_q = torch.randn(n_state, D, device='cuda', dtype=torch.bfloat16) * 0.1
    W_beta = torch.randn(n_state, D, device='cuda', dtype=torch.bfloat16) * 0.1
    b_beta = torch.full((n_state,), 2.0, device='cuda', dtype=torch.bfloat16)

    # Python reference (in float32 for accuracy)
    output_py, S_py, _ = e75_python_forward(
        x.float(), S0.float().clone(),
        W_k.float(), W_v.float(), W_q.float(), W_beta.float(), b_beta.float()
    )
    output_py = output_py.bfloat16()
    S_py = S_py.bfloat16()

    # CUDA implementation
    cell = E75VectorGateCell(D, n_state=n_state, use_cuda=True)
    cell.W_k.data = W_k.clone()
    cell.W_v.data = W_v.clone()
    cell.W_q.data = W_q.clone()
    cell.W_beta.data = W_beta.clone()
    cell.b_beta.data = b_beta.clone()
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

    print("  PASS")
    return True


def test_backward_gradients_fp32():
    """Test backward pass gradient equivalence (FP32)."""
    print("Testing E75 Vector Gate backward gradients (FP32)...")

    if not E75_VECTOR_GATE_CUDA_AVAILABLE:
        print("  SKIPPED: CUDA kernel not available")
        return True

    torch.manual_seed(42)
    T, B, D = 8, 2, 128
    n_state = 16

    # Create inputs with gradients
    x = torch.randn(T, B, D, device='cuda', dtype=torch.float32, requires_grad=True)
    S0 = torch.randn(B, n_state, n_state, device='cuda', dtype=torch.float32) * 0.1
    W_k = torch.randn(n_state, D, device='cuda', dtype=torch.float32, requires_grad=True) * 0.1
    W_v = torch.randn(n_state, D, device='cuda', dtype=torch.float32, requires_grad=True) * 0.1
    W_q = torch.randn(n_state, D, device='cuda', dtype=torch.float32, requires_grad=True) * 0.1
    W_beta = torch.randn(n_state, D, device='cuda', dtype=torch.float32, requires_grad=True) * 0.1
    b_beta = torch.full((n_state,), 2.0, device='cuda', dtype=torch.float32, requires_grad=True)

    # CUDA implementation
    cell = E75VectorGateCell(D, n_state=n_state, use_cuda=True)
    cell.W_k.data = W_k.clone()
    cell.W_v.data = W_v.clone()
    cell.W_q.data = W_q.clone()
    cell.W_beta.data = W_beta.clone()
    cell.b_beta.data = b_beta.clone()
    cell.W_k.requires_grad_(True)
    cell.W_v.requires_grad_(True)
    cell.W_q.requires_grad_(True)
    cell.W_beta.requires_grad_(True)
    cell.b_beta.requires_grad_(True)
    cell = cell.cuda().float()
    cell.train()

    output_cuda, _ = cell(x, S0.clone())
    loss_cuda = output_cuda.sum()
    loss_cuda.backward()

    grad_x_cuda = x.grad.clone()
    grad_W_k_cuda = cell.W_k.grad.clone()
    grad_W_v_cuda = cell.W_v.grad.clone()
    grad_W_q_cuda = cell.W_q.grad.clone()
    grad_W_beta_cuda = cell.W_beta.grad.clone()
    grad_b_beta_cuda = cell.b_beta.grad.clone()

    # Reset gradients
    x.grad = None

    # Python reference with autograd
    x_py = x.detach().clone().requires_grad_(True)
    W_k_py = W_k.detach().clone().requires_grad_(True)
    W_v_py = W_v.detach().clone().requires_grad_(True)
    W_q_py = W_q.detach().clone().requires_grad_(True)
    W_beta_py = W_beta.detach().clone().requires_grad_(True)
    b_beta_py = b_beta.detach().clone().requires_grad_(True)

    S = S0.clone()
    outputs = []
    x_flat = x_py.reshape(T * B, D)
    k_all = (x_flat @ W_k_py.T).reshape(T, B, n_state)
    v_all = (x_flat @ W_v_py.T).reshape(T, B, n_state)
    q_all = (x_flat @ W_q_py.T).reshape(T, B, n_state)
    g_all = torch.sigmoid((x_flat @ W_beta_py.T + b_beta_py).reshape(T, B, n_state))

    for t in range(T):
        k = k_all[t]
        v = v_all[t]
        q = q_all[t]
        g = g_all[t]

        k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
        retrieved = torch.einsum('bij,bj->bi', S, k_norm)
        delta = v - retrieved
        outer = torch.einsum('bi,bj->bij', delta, k_norm)
        S = g.unsqueeze(-1) * S + outer

        Sq = torch.einsum('bij,bj->bi', S, q)
        out = Sq * F.silu(Sq)
        outputs.append(out)

    output_py = torch.stack(outputs, dim=0)
    loss_py = output_py.sum()
    loss_py.backward()

    grad_x_py = x_py.grad
    grad_W_k_py = W_k_py.grad
    grad_W_v_py = W_v_py.grad
    grad_W_q_py = W_q_py.grad
    grad_W_beta_py = W_beta_py.grad
    grad_b_beta_py = b_beta_py.grad

    # Compare gradients
    def rel_err(a, b):
        return ((a - b).abs() / (b.abs() + 1e-8)).max().item()

    max_grad_x_rel_err = rel_err(grad_x_cuda, grad_x_py)
    max_grad_Wk_rel_err = rel_err(grad_W_k_cuda, grad_W_k_py)
    max_grad_Wv_rel_err = rel_err(grad_W_v_cuda, grad_W_v_py)
    max_grad_Wq_rel_err = rel_err(grad_W_q_cuda, grad_W_q_py)
    max_grad_Wbeta_rel_err = rel_err(grad_W_beta_cuda, grad_W_beta_py)
    max_grad_bbeta_rel_err = rel_err(grad_b_beta_cuda, grad_b_beta_py)

    print(f"  Max grad_x relative error: {max_grad_x_rel_err:.6f}")
    print(f"  Max grad_W_k relative error: {max_grad_Wk_rel_err:.6f}")
    print(f"  Max grad_W_v relative error: {max_grad_Wv_rel_err:.6f}")
    print(f"  Max grad_W_q relative error: {max_grad_Wq_rel_err:.6f}")
    print(f"  Max grad_W_beta relative error: {max_grad_Wbeta_rel_err:.6f}")
    print(f"  Max grad_b_beta relative error: {max_grad_bbeta_rel_err:.6f}")

    # FP32 tolerance: 0.1%
    assert max_grad_x_rel_err < 0.001, f"FP32 grad_x error too large: {max_grad_x_rel_err}"
    assert max_grad_Wk_rel_err < 0.001, f"FP32 grad_W_k error too large: {max_grad_Wk_rel_err}"
    assert max_grad_Wv_rel_err < 0.001, f"FP32 grad_W_v error too large: {max_grad_Wv_rel_err}"
    assert max_grad_Wq_rel_err < 0.001, f"FP32 grad_W_q error too large: {max_grad_Wq_rel_err}"
    assert max_grad_Wbeta_rel_err < 0.001, f"FP32 grad_W_beta error too large: {max_grad_Wbeta_rel_err}"
    assert max_grad_bbeta_rel_err < 0.001, f"FP32 grad_b_beta error too large: {max_grad_bbeta_rel_err}"

    print("  PASS")
    return True


def test_backward_gradients_bf16():
    """Test backward pass gradient equivalence (BF16)."""
    print("Testing E75 Vector Gate backward gradients (BF16)...")

    if not E75_VECTOR_GATE_CUDA_AVAILABLE:
        print("  SKIPPED: CUDA kernel not available")
        return True

    torch.manual_seed(42)
    T, B, D = 8, 2, 128
    n_state = 16

    # Create inputs with gradients
    x = torch.randn(T, B, D, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    S0 = torch.randn(B, n_state, n_state, device='cuda', dtype=torch.bfloat16) * 0.1
    W_k = torch.randn(n_state, D, device='cuda', dtype=torch.bfloat16, requires_grad=True) * 0.1
    W_v = torch.randn(n_state, D, device='cuda', dtype=torch.bfloat16, requires_grad=True) * 0.1
    W_q = torch.randn(n_state, D, device='cuda', dtype=torch.bfloat16, requires_grad=True) * 0.1
    W_beta = torch.randn(n_state, D, device='cuda', dtype=torch.bfloat16, requires_grad=True) * 0.1
    b_beta = torch.full((n_state,), 2.0, device='cuda', dtype=torch.bfloat16, requires_grad=True)

    # CUDA implementation
    cell = E75VectorGateCell(D, n_state=n_state, use_cuda=True)
    cell.W_k.data = W_k.clone()
    cell.W_v.data = W_v.clone()
    cell.W_q.data = W_q.clone()
    cell.W_beta.data = W_beta.clone()
    cell.b_beta.data = b_beta.clone()
    cell.W_k.requires_grad_(True)
    cell.W_v.requires_grad_(True)
    cell.W_q.requires_grad_(True)
    cell.W_beta.requires_grad_(True)
    cell.b_beta.requires_grad_(True)
    cell = cell.cuda().bfloat16()
    cell.train()

    output_cuda, _ = cell(x, S0.clone())
    loss_cuda = output_cuda.sum()
    loss_cuda.backward()

    grad_x_cuda = x.grad.clone().float()
    grad_W_k_cuda = cell.W_k.grad.clone().float()
    grad_W_v_cuda = cell.W_v.grad.clone().float()
    grad_W_q_cuda = cell.W_q.grad.clone().float()
    grad_W_beta_cuda = cell.W_beta.grad.clone().float()
    grad_b_beta_cuda = cell.b_beta.grad.clone().float()

    # Reset gradients
    x.grad = None

    # Python reference with autograd (in float32 for accuracy)
    x_py = x.float().detach().clone().requires_grad_(True)
    W_k_py = W_k.float().detach().clone().requires_grad_(True)
    W_v_py = W_v.float().detach().clone().requires_grad_(True)
    W_q_py = W_q.float().detach().clone().requires_grad_(True)
    W_beta_py = W_beta.float().detach().clone().requires_grad_(True)
    b_beta_py = b_beta.float().detach().clone().requires_grad_(True)

    S = S0.float().clone()
    outputs = []
    x_flat = x_py.reshape(T * B, D)
    k_all = (x_flat @ W_k_py.T).reshape(T, B, n_state)
    v_all = (x_flat @ W_v_py.T).reshape(T, B, n_state)
    q_all = (x_flat @ W_q_py.T).reshape(T, B, n_state)
    g_all = torch.sigmoid((x_flat @ W_beta_py.T + b_beta_py).reshape(T, B, n_state))

    for t in range(T):
        k = k_all[t]
        v = v_all[t]
        q = q_all[t]
        g = g_all[t]

        k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
        retrieved = torch.einsum('bij,bj->bi', S, k_norm)
        delta = v - retrieved
        outer = torch.einsum('bi,bj->bij', delta, k_norm)
        S = g.unsqueeze(-1) * S + outer

        Sq = torch.einsum('bij,bj->bi', S, q)
        out = Sq * F.silu(Sq)
        outputs.append(out)

    output_py = torch.stack(outputs, dim=0)
    loss_py = output_py.sum()
    loss_py.backward()

    grad_x_py = x_py.grad
    grad_W_k_py = W_k_py.grad
    grad_W_v_py = W_v_py.grad
    grad_W_q_py = W_q_py.grad
    grad_W_beta_py = W_beta_py.grad
    grad_b_beta_py = b_beta_py.grad

    # Compare gradients
    def rel_err(a, b):
        return ((a - b).abs() / (b.abs() + 1e-6)).max().item()

    max_grad_x_rel_err = rel_err(grad_x_cuda, grad_x_py)
    max_grad_Wk_rel_err = rel_err(grad_W_k_cuda, grad_W_k_py)
    max_grad_Wv_rel_err = rel_err(grad_W_v_cuda, grad_W_v_py)
    max_grad_Wq_rel_err = rel_err(grad_W_q_cuda, grad_W_q_py)
    max_grad_Wbeta_rel_err = rel_err(grad_W_beta_cuda, grad_W_beta_py)
    max_grad_bbeta_rel_err = rel_err(grad_b_beta_cuda, grad_b_beta_py)

    print(f"  Max grad_x relative error: {max_grad_x_rel_err:.6f}")
    print(f"  Max grad_W_k relative error: {max_grad_Wk_rel_err:.6f}")
    print(f"  Max grad_W_v relative error: {max_grad_Wv_rel_err:.6f}")
    print(f"  Max grad_W_q relative error: {max_grad_Wq_rel_err:.6f}")
    print(f"  Max grad_W_beta relative error: {max_grad_Wbeta_rel_err:.6f}")
    print(f"  Max grad_b_beta relative error: {max_grad_bbeta_rel_err:.6f}")

    # BF16 tolerance: 1%
    assert max_grad_x_rel_err < 0.01, f"BF16 grad_x error too large: {max_grad_x_rel_err}"
    assert max_grad_Wk_rel_err < 0.01, f"BF16 grad_W_k error too large: {max_grad_Wk_rel_err}"
    assert max_grad_Wv_rel_err < 0.01, f"BF16 grad_W_v error too large: {max_grad_Wv_rel_err}"
    assert max_grad_Wq_rel_err < 0.01, f"BF16 grad_W_q error too large: {max_grad_Wq_rel_err}"
    assert max_grad_Wbeta_rel_err < 0.01, f"BF16 grad_W_beta error too large: {max_grad_Wbeta_rel_err}"
    assert max_grad_bbeta_rel_err < 0.01, f"BF16 grad_b_beta error too large: {max_grad_bbeta_rel_err}"

    print("  PASS")
    return True


def test_gate_behavior():
    """Test that gates have expected behavior (sigmoid of W_beta@x + b_beta)."""
    print("Testing E75 Vector Gate gate behavior...")

    torch.manual_seed(42)
    T, B, D = 32, 4, 256
    n_state = 32

    x = torch.randn(T, B, D, device='cuda', dtype=torch.float32)
    S0 = torch.randn(B, n_state, n_state, device='cuda', dtype=torch.float32) * 0.1
    W_k = torch.randn(n_state, D, device='cuda', dtype=torch.float32) * 0.1
    W_v = torch.randn(n_state, D, device='cuda', dtype=torch.float32) * 0.1
    W_q = torch.randn(n_state, D, device='cuda', dtype=torch.float32) * 0.1
    W_beta = torch.randn(n_state, D, device='cuda', dtype=torch.float32) * 0.1
    b_beta = torch.full((n_state,), 2.0, device='cuda', dtype=torch.float32)  # Bias toward 0.88 retention

    # Get gate values
    _, _, gate_values = e75_python_forward(x, S0.clone(), W_k, W_v, W_q, W_beta, b_beta)

    # Check gate statistics
    all_gates = torch.stack(gate_values, dim=0)  # [T, B, n]
    mean_gate = all_gates.mean().item()
    min_gate = all_gates.min().item()
    max_gate = all_gates.max().item()
    std_gate = all_gates.std().item()

    print(f"  Gate mean: {mean_gate:.4f}")
    print(f"  Gate std: {std_gate:.4f}")
    print(f"  Gate min: {min_gate:.4f}")
    print(f"  Gate max: {max_gate:.4f}")

    # With b_beta=2.0, expected mean gate should be around sigmoid(2) ~= 0.88
    expected_mean = torch.sigmoid(torch.tensor(2.0)).item()
    print(f"  Expected mean (sigmoid(2)): {expected_mean:.4f}")

    # Gate mean should be reasonably close to expected
    assert abs(mean_gate - expected_mean) < 0.3, f"Gate mean too far from expected: {mean_gate} vs {expected_mean}"

    # Gates should be in valid range (0, 1)
    assert min_gate > 0.0, f"Gate min should be > 0: {min_gate}"
    assert max_gate < 1.0, f"Gate max should be < 1: {max_gate}"

    print("  PASS")
    return True


def test_python_fallback():
    """Test Python fallback implementation."""
    print("Testing E75 Vector Gate Python fallback...")

    torch.manual_seed(42)
    T, B, D = 8, 2, 128
    n_state = 16

    x = torch.randn(T, B, D, device='cuda', dtype=torch.float32)
    S0 = torch.randn(B, n_state, n_state, device='cuda', dtype=torch.float32) * 0.1

    # Python fallback
    cell = E75VectorGateCell(D, n_state=n_state, use_cuda=False)
    cell = cell.cuda().float()
    cell.eval()

    with torch.no_grad():
        output_py, S_py = cell(x, S0.clone())

    # Reference implementation
    output_ref, S_ref, _ = e75_python_forward(
        x, S0.clone(), cell.W_k.data, cell.W_v.data, cell.W_q.data,
        cell.W_beta.data, cell.b_beta.data
    )

    # Compare
    output_diff = (output_py - output_ref).abs().max().item()
    S_diff = (S_py - S_ref).abs().max().item()

    print(f"  Max output diff: {output_diff:.10f}")
    print(f"  Max S diff: {S_diff:.10f}")

    # Should be numerically identical
    assert output_diff < 1e-5, f"Python fallback output mismatch: {output_diff}"
    assert S_diff < 1e-5, f"Python fallback S mismatch: {S_diff}"

    print("  PASS")
    return True


def main():
    print("=" * 60)
    print("E75 Vector Gate Gradient Tests")
    print("=" * 60)
    print(f"CUDA kernel available: {E75_VECTOR_GATE_CUDA_AVAILABLE}")
    print()

    tests = [
        ("Python Fallback", test_python_fallback),
        ("Forward FP32", test_forward_equivalence_fp32),
        ("Forward BF16", test_forward_equivalence_bf16),
        ("Backward FP32", test_backward_gradients_fp32),
        ("Backward BF16", test_backward_gradients_bf16),
        ("Gate Behavior", test_gate_behavior),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            import traceback
            print(f"  FAIL: {e}")
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
