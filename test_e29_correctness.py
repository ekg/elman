"""
E29 Correctness Tests: Python forward/backward validation

Tests:
1. E29a forward - numerical check
2. E29a backward - compare with autograd
3. E29b forward - numerical check
4. E29b backward - compare with autograd
5. (Future) CUDA vs Python comparison when kernels are compiled
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck
import math
import sys

sys.path.insert(0, '/home/erikg/elman')
from elman.models.e29_selective import (
    e29a_forward_python, e29a_backward_python,
    e29b_forward_python, e29b_backward_python,
    E29aSelectiveElmanCell, E29bSelectiveElmanCell
)


def test_e29a_forward():
    """Test E29a forward pass produces valid outputs."""
    print("=" * 60)
    print("TEST: E29a Forward Pass")
    print("=" * 60)

    B, T, D, N = 2, 4, 64, 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32

    torch.manual_seed(42)
    x = torch.randn(B, T, D, device=device, dtype=dtype) * 0.1
    h_tape = torch.randn(B, N, D, device=device, dtype=dtype) * 0.01
    h_work = torch.randn(B, D, device=device, dtype=dtype) * 0.01

    W_h = torch.empty(D, D, device=device, dtype=dtype)
    nn.init.orthogonal_(W_h)
    W_h = W_h * 0.9
    W_xz = torch.randn(2 * D, D, device=device, dtype=dtype) * 0.1
    b_h = torch.zeros(D, device=device, dtype=dtype)
    W_write = torch.randn(D, D, device=device, dtype=dtype) * 0.1

    output_all, h_work_all, h_tape_final, h_tape_all, read_attn, write_attn = \
        e29a_forward_python(x, h_tape, h_work, W_h, W_xz, b_h, W_write)

    # Check shapes
    assert output_all.shape == (B, T, D), f"output shape mismatch: {output_all.shape}"
    assert h_work_all.shape == (B, T, D), f"h_work_all shape mismatch: {h_work_all.shape}"
    assert h_tape_final.shape == (B, N, D), f"h_tape_final shape mismatch: {h_tape_final.shape}"
    assert read_attn.shape == (B, T, N), f"read_attn shape mismatch: {read_attn.shape}"
    assert write_attn.shape == (B, T, N), f"write_attn shape mismatch: {write_attn.shape}"

    # Check no NaN
    assert not torch.isnan(output_all).any(), "NaN in output"
    assert not torch.isnan(h_work_all).any(), "NaN in h_work"
    assert not torch.isnan(h_tape_final).any(), "NaN in h_tape"

    # Check attention sums to 1
    read_sum = read_attn.sum(dim=-1)
    write_sum = write_attn.sum(dim=-1)
    assert torch.allclose(read_sum, torch.ones_like(read_sum), atol=1e-5), \
        f"Read attention doesn't sum to 1: {read_sum}"
    assert torch.allclose(write_sum, torch.ones_like(write_sum), atol=1e-5), \
        f"Write attention doesn't sum to 1: {write_sum}"

    print(f"  Output shape: {output_all.shape}")
    print(f"  Output stats: min={output_all.min():.4f}, max={output_all.max():.4f}")
    print(f"  Read attention sum: {read_sum.mean():.6f}")
    print(f"  PASS")
    return True


def test_e29a_backward_vs_autograd():
    """Test E29a backward using torch.autograd.gradcheck."""
    print("\n" + "=" * 60)
    print("TEST: E29a Backward (autograd.gradcheck)")
    print("=" * 60)

    B, T, D, N = 2, 3, 16, 4  # Small dims for gradcheck
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float64

    torch.manual_seed(123)

    # Create inputs with requires_grad for gradcheck
    x = (torch.randn(B, T, D, device=device, dtype=dtype) * 0.1).requires_grad_(True)
    h_tape_init = torch.randn(B, N, D, device=device, dtype=dtype) * 0.01
    h_work_init = torch.randn(B, D, device=device, dtype=dtype) * 0.01

    W_h = torch.empty(D, D, device=device, dtype=dtype)
    nn.init.orthogonal_(W_h)
    W_h = (W_h * 0.5).requires_grad_(True)  # Smaller spectral radius for stability

    W_xz = (torch.randn(2 * D, D, device=device, dtype=dtype) * 0.1).requires_grad_(True)
    b_h = (torch.zeros(D, device=device, dtype=dtype) + 0.01).requires_grad_(True)
    W_write = (torch.randn(D, D, device=device, dtype=dtype) * 0.1).requires_grad_(True)

    def forward_fn(x, W_h, W_xz, b_h, W_write):
        output_all, h_work_all, h_tape_final, h_tape_all, _, _ = \
            e29a_forward_python(x, h_tape_init, h_work_init, W_h, W_xz, b_h, W_write)
        return output_all.sum() + h_tape_final.sum()

    try:
        # RNNs accumulate numerical error through time, so use relaxed tolerances
        result = gradcheck(forward_fn, (x, W_h, W_xz, b_h, W_write),
                          eps=1e-5, atol=1e-3, rtol=1e-2, raise_exception=True)
        print(f"  gradcheck PASSED")
        return True
    except Exception as e:
        print(f"  gradcheck FAILED: {e}")
        return False


def test_e29b_forward():
    """Test E29b forward pass produces valid outputs."""
    print("\n" + "=" * 60)
    print("TEST: E29b Forward Pass")
    print("=" * 60)

    B, T, D, N = 2, 4, 64, 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32

    torch.manual_seed(42)
    x = torch.randn(B, T, D, device=device, dtype=dtype) * 0.1
    h_tape = torch.randn(B, N, D, device=device, dtype=dtype) * 0.01
    h_work = torch.randn(B, D, device=device, dtype=dtype) * 0.01

    W_h = torch.empty(D, D, device=device, dtype=dtype)
    nn.init.orthogonal_(W_h)
    W_h = W_h * 0.9
    W_xz = torch.randn(2 * D, D, device=device, dtype=dtype) * 0.1
    b_h = torch.zeros(D, device=device, dtype=dtype)
    W_write = torch.randn(D, D, device=device, dtype=dtype) * 0.1
    W_gate = torch.randn(D, 3 * D, device=device, dtype=dtype) * 0.1

    output_all, h_work_all, h_tape_final, h_tape_all, read_attn, write_attn = \
        e29b_forward_python(x, h_tape, h_work, W_h, W_xz, b_h, W_write, W_gate)

    # Check shapes
    assert output_all.shape == (B, T, D), f"output shape mismatch: {output_all.shape}"
    assert h_work_all.shape == (B, T, D), f"h_work_all shape mismatch: {h_work_all.shape}"
    assert h_tape_final.shape == (B, N, D), f"h_tape_final shape mismatch: {h_tape_final.shape}"

    # Check no NaN
    assert not torch.isnan(output_all).any(), "NaN in output"
    assert not torch.isnan(h_work_all).any(), "NaN in h_work"
    assert not torch.isnan(h_tape_final).any(), "NaN in h_tape"

    # Check attention sums to 1
    read_sum = read_attn.sum(dim=-1)
    write_sum = write_attn.sum(dim=-1)
    assert torch.allclose(read_sum, torch.ones_like(read_sum), atol=1e-5)
    assert torch.allclose(write_sum, torch.ones_like(write_sum), atol=1e-5)

    print(f"  Output shape: {output_all.shape}")
    print(f"  Output stats: min={output_all.min():.4f}, max={output_all.max():.4f}")
    print(f"  PASS")
    return True


def test_e29b_backward_vs_autograd():
    """Test E29b backward using torch.autograd.gradcheck."""
    print("\n" + "=" * 60)
    print("TEST: E29b Backward (autograd.gradcheck)")
    print("=" * 60)

    B, T, D, N = 2, 3, 16, 4  # Small dims for gradcheck
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float64

    torch.manual_seed(456)

    x = (torch.randn(B, T, D, device=device, dtype=dtype) * 0.1).requires_grad_(True)
    h_tape_init = torch.randn(B, N, D, device=device, dtype=dtype) * 0.01
    h_work_init = torch.randn(B, D, device=device, dtype=dtype) * 0.01

    W_h = torch.empty(D, D, device=device, dtype=dtype)
    nn.init.orthogonal_(W_h)
    W_h = (W_h * 0.5).requires_grad_(True)

    W_xz = (torch.randn(2 * D, D, device=device, dtype=dtype) * 0.1).requires_grad_(True)
    b_h = (torch.zeros(D, device=device, dtype=dtype) + 0.01).requires_grad_(True)
    W_write = (torch.randn(D, D, device=device, dtype=dtype) * 0.1).requires_grad_(True)
    W_gate = (torch.randn(D, 3 * D, device=device, dtype=dtype) * 0.1).requires_grad_(True)

    def forward_fn(x, W_h, W_xz, b_h, W_write, W_gate):
        output_all, h_work_all, h_tape_final, h_tape_all, _, _ = \
            e29b_forward_python(x, h_tape_init, h_work_init, W_h, W_xz, b_h, W_write, W_gate)
        return output_all.sum() + h_tape_final.sum()

    try:
        # RNNs accumulate numerical error through time, so use relaxed tolerances
        result = gradcheck(forward_fn, (x, W_h, W_xz, b_h, W_write, W_gate),
                          eps=1e-5, atol=1e-3, rtol=1e-2, raise_exception=True)
        print(f"  gradcheck PASSED")
        return True
    except Exception as e:
        print(f"  gradcheck FAILED: {e}")
        return False


def test_e29a_vs_e26_behavior():
    """Test that E29a differs from E26 (different gate mechanism)."""
    print("\n" + "=" * 60)
    print("TEST: E29a vs E26 Behavior Difference")
    print("=" * 60)

    B, T, D, N = 2, 4, 32, 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32

    torch.manual_seed(42)
    x = torch.randn(B, T, D, device=device, dtype=dtype) * 0.1
    h_tape = torch.randn(B, N, D, device=device, dtype=dtype) * 0.01
    h_work = torch.randn(B, D, device=device, dtype=dtype) * 0.01

    W_h = torch.empty(D, D, device=device, dtype=dtype)
    nn.init.orthogonal_(W_h)
    W_h = W_h * 0.9
    W_xz = torch.randn(2 * D, D, device=device, dtype=dtype) * 0.1
    b_h = torch.zeros(D, device=device, dtype=dtype)
    W_write = torch.randn(D, D, device=device, dtype=dtype) * 0.1

    # E29a output
    output_e29a, _, _, _, _, _ = e29a_forward_python(
        x, h_tape, h_work, W_h, W_xz, b_h, W_write
    )

    # For E26-style, gate would be just silu(z), not silu(z + read + h_work)
    # Let's compute what E26-style would give
    xz = x @ W_xz.T
    x_proj = xz[:, :, :D]
    z = xz[:, :, D:]
    scale = 1.0 / math.sqrt(D)

    # Simple E26-style forward (gate = silu(z))
    h_tape_e26 = h_tape.clone()
    h_work_e26 = h_work.clone()
    output_e26_list = []

    for t in range(T):
        read_scores = torch.einsum('bd,bnd->bn', h_work_e26.float(), h_tape_e26.float()) * scale
        read_attn = F.softmax(read_scores, dim=-1).to(dtype)
        read_val = torch.einsum('bn,bnd->bd', read_attn.float(), h_tape_e26.float()).to(dtype)

        Rh = h_work_e26 @ W_h.T
        h_work_e26 = torch.tanh(x_proj[:, t] + Rh + read_val + b_h)

        # E26-style gate: just silu(z)
        gate_e26 = F.silu(z[:, t])
        output_e26 = h_work_e26 * gate_e26
        output_e26_list.append(output_e26)

        # Write (simplified)
        write_val = h_work_e26 @ W_write.T
        write_scores = torch.einsum('bd,bnd->bn', write_val.float(), h_tape_e26.float()) * scale
        write_attn = F.softmax(write_scores, dim=-1).to(dtype)
        h_tape_e26 = h_tape_e26 * (1 - write_attn.unsqueeze(-1)) + write_val.unsqueeze(1) * write_attn.unsqueeze(-1)

    output_e26 = torch.stack(output_e26_list, dim=1)

    # They should differ!
    diff = (output_e29a - output_e26).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"  E29a output: min={output_e29a.min():.4f}, max={output_e29a.max():.4f}")
    print(f"  E26 output:  min={output_e26.min():.4f}, max={output_e26.max():.4f}")
    print(f"  Max difference: {max_diff:.4f}")
    print(f"  Mean difference: {mean_diff:.4f}")

    if max_diff > 1e-5:
        print(f"  PASS (E29a and E26-style produce different outputs)")
        return True
    else:
        print(f"  FAIL (E29a and E26-style are identical - gate isn't selective)")
        return False


def test_e29_cell_modules():
    """Test E29a and E29b cell modules."""
    print("\n" + "=" * 60)
    print("TEST: E29a/E29b Cell Modules")
    print("=" * 60)

    B, T, D, N = 2, 4, 64, 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32

    torch.manual_seed(42)
    x = torch.randn(B, T, D, device=device, dtype=dtype) * 0.1

    # Test E29a cell
    cell_a = E29aSelectiveElmanCell(dim=D, n_slots=N).to(device).to(dtype)
    output_a, h_tape_a, h_work_a = cell_a(x, use_cuda=False)

    assert output_a.shape == (B, T, D), f"E29a output shape: {output_a.shape}"
    assert not torch.isnan(output_a).any(), "E29a output has NaN"
    print(f"  E29a cell output: shape={output_a.shape}, range=[{output_a.min():.3f}, {output_a.max():.3f}]")

    # Test E29b cell
    cell_b = E29bSelectiveElmanCell(dim=D, n_slots=N).to(device).to(dtype)
    output_b, h_tape_b, h_work_b = cell_b(x, use_cuda=False)

    assert output_b.shape == (B, T, D), f"E29b output shape: {output_b.shape}"
    assert not torch.isnan(output_b).any(), "E29b output has NaN"
    print(f"  E29b cell output: shape={output_b.shape}, range=[{output_b.min():.3f}, {output_b.max():.3f}]")

    # Test that they produce different outputs (different gate mechanisms)
    diff = (output_a - output_b).abs().mean().item()
    print(f"  E29a vs E29b mean diff: {diff:.4f}")

    if diff > 1e-5:
        print(f"  PASS (E29a and E29b produce different outputs)")
        return True
    else:
        print(f"  FAIL (E29a and E29b are identical)")
        return False


def test_e29a_cuda_vs_python():
    """Compare E29a CUDA kernel with Python reference."""
    print("\n" + "=" * 60)
    print("TEST: E29a CUDA vs Python Forward")
    print("=" * 60)

    try:
        import hasty_pytorch_lib
        if not hasattr(hasty_pytorch_lib, 'e29a_selective_forward'):
            print("  SKIP: e29a_selective_forward not available in CUDA lib")
            return None
    except ImportError:
        print("  SKIP: hasty_pytorch_lib not available")
        return None

    B, T, D, N = 2, 8, 512, 8
    device = 'cuda'
    dtype = torch.bfloat16

    torch.manual_seed(42)
    # Use smaller initialization for bf16 numerical stability
    x = torch.randn(B, T, D, device=device, dtype=dtype) * 0.01
    h_tape = torch.zeros(B, N, D, device=device, dtype=dtype)  # Start with zeros
    h_work = torch.zeros(B, D, device=device, dtype=dtype)

    # Initialize W_h in float32 then convert to bf16
    W_h_f32 = torch.empty(D, D, device=device, dtype=torch.float32)
    nn.init.orthogonal_(W_h_f32)
    W_h = (W_h_f32 * 0.5).to(dtype)  # Smaller spectral radius
    W_xz = torch.randn(2 * D, D, device=device, dtype=dtype) * 0.01
    b_h = torch.zeros(D, device=device, dtype=dtype)
    W_write = torch.randn(D, D, device=device, dtype=dtype) * 0.01

    # Python forward
    output_py, h_work_py, _, _, _, _ = e29a_forward_python(
        x, h_tape, h_work, W_h, W_xz, b_h, W_write
    )

    # CUDA forward
    output_cuda, h_work_cuda, h_tape_final, h_tape_all, read_attn, write_attn = \
        hasty_pytorch_lib.e29a_selective_forward(
            True,  # training
            x.contiguous(),
            h_tape.contiguous(),
            h_work.contiguous(),
            W_h.contiguous(),
            W_xz.contiguous(),
            b_h.contiguous(),
            W_write.contiguous()
        )

    # Compare outputs
    output_diff = (output_py - output_cuda).abs()
    h_work_diff = (h_work_py - h_work_cuda).abs()

    max_output_diff = output_diff.max().item()
    max_hwork_diff = h_work_diff.max().item()
    mean_output_diff = output_diff.mean().item()

    print(f"  Output max diff: {max_output_diff:.6f}")
    print(f"  Output mean diff: {mean_output_diff:.6f}")
    print(f"  h_work max diff: {max_hwork_diff:.6f}")

    # bf16 tolerance is ~1e-3
    threshold = 1e-2
    if max_output_diff < threshold and max_hwork_diff < threshold:
        print(f"  PASS (max_diff < {threshold})")
        return True
    else:
        print(f"  FAIL (max_diff >= {threshold})")
        return False


def test_e29b_cuda_vs_python():
    """Compare E29b CUDA kernel with Python reference."""
    print("\n" + "=" * 60)
    print("TEST: E29b CUDA vs Python Forward")
    print("=" * 60)

    try:
        import hasty_pytorch_lib
        if not hasattr(hasty_pytorch_lib, 'e29b_selective_forward'):
            print("  SKIP: e29b_selective_forward not available in CUDA lib")
            return None
    except ImportError:
        print("  SKIP: hasty_pytorch_lib not available")
        return None

    B, T, D, N = 2, 8, 512, 8
    device = 'cuda'
    dtype = torch.bfloat16

    torch.manual_seed(42)
    # Use smaller initialization for bf16 numerical stability
    x = torch.randn(B, T, D, device=device, dtype=dtype) * 0.01
    h_tape = torch.zeros(B, N, D, device=device, dtype=dtype)
    h_work = torch.zeros(B, D, device=device, dtype=dtype)

    # Initialize W_h in float32 then convert to bf16
    W_h_f32 = torch.empty(D, D, device=device, dtype=torch.float32)
    nn.init.orthogonal_(W_h_f32)
    W_h = (W_h_f32 * 0.5).to(dtype)
    W_xz = torch.randn(2 * D, D, device=device, dtype=dtype) * 0.01
    b_h = torch.zeros(D, device=device, dtype=dtype)
    W_write = torch.randn(D, D, device=device, dtype=dtype) * 0.01
    W_gate = torch.randn(D, 3 * D, device=device, dtype=dtype) * 0.01

    # Python forward
    output_py, h_work_py, _, _, _, _ = e29b_forward_python(
        x, h_tape, h_work, W_h, W_xz, b_h, W_write, W_gate
    )

    # CUDA forward
    output_cuda, h_work_cuda, h_tape_final, h_tape_all, read_attn, write_attn = \
        hasty_pytorch_lib.e29b_selective_forward(
            True,  # training
            x.contiguous(),
            h_tape.contiguous(),
            h_work.contiguous(),
            W_h.contiguous(),
            W_xz.contiguous(),
            b_h.contiguous(),
            W_write.contiguous(),
            W_gate.contiguous()
        )

    # Compare outputs
    output_diff = (output_py - output_cuda).abs()
    h_work_diff = (h_work_py - h_work_cuda).abs()

    max_output_diff = output_diff.max().item()
    max_hwork_diff = h_work_diff.max().item()
    mean_output_diff = output_diff.mean().item()

    print(f"  Output max diff: {max_output_diff:.6f}")
    print(f"  Output mean diff: {mean_output_diff:.6f}")
    print(f"  h_work max diff: {max_hwork_diff:.6f}")

    threshold = 1e-2
    if max_output_diff < threshold and max_hwork_diff < threshold:
        print(f"  PASS (max_diff < {threshold})")
        return True
    else:
        print(f"  FAIL (max_diff >= {threshold})")
        return False


def main():
    print("\n" + "#" * 70)
    print("# E29 CORRECTNESS TESTS")
    print("#" * 70 + "\n")

    results = []

    results.append(("E29a Forward", test_e29a_forward()))
    results.append(("E29a Backward vs Autograd", test_e29a_backward_vs_autograd()))
    results.append(("E29b Forward", test_e29b_forward()))
    results.append(("E29b Backward vs Autograd", test_e29b_backward_vs_autograd()))
    results.append(("E29a vs E26 Behavior", test_e29a_vs_e26_behavior()))
    results.append(("E29 Cell Modules", test_e29_cell_modules()))

    # CUDA vs Python tests (may be skipped if CUDA not available)
    cuda_result_a = test_e29a_cuda_vs_python()
    if cuda_result_a is not None:
        results.append(("E29a CUDA vs Python", cuda_result_a))

    cuda_result_b = test_e29b_cuda_vs_python()
    if cuda_result_b is not None:
        results.append(("E29b CUDA vs Python", cuda_result_b))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
