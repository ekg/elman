#!/usr/bin/env python3
"""
Test E88 CUDA kernel numerical accuracy vs Python reference.

Tests across configuration space to find where numerical issues occur:
- Varying n_heads (4, 8, 16, 32, 64, 128)
- Varying n_state (16, 24, 32, 48, 64, 96, 128)
- Check for NaN, inf, and large differences
"""

import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '.')

from elman.models.e88_fla_hybrid import E88FLAHybrid, E88_NATIVE_CUDA_AVAILABLE

def test_config(n_heads, n_state, dim=512, T=64, B=4, expansion=1.0, verbose=False):
    """Test a single configuration, comparing CUDA vs Python."""
    device = 'cuda'
    dtype = torch.bfloat16

    # Create model
    model = E88FLAHybrid(
        dim=dim,
        n_state=n_state,
        n_heads=n_heads,
        expansion=expansion,
        use_conv=False,
        use_gate=False,
        use_output_norm=False,
    ).to(device).to(dtype)

    # Random input (use same seed for reproducibility)
    torch.manual_seed(42)
    x = torch.randn(B, T, dim, device=device, dtype=dtype)

    # --- CUDA forward ---
    model.train()
    try:
        out_cuda, S_cuda = model(x, use_cuda=True)
        cuda_ok = True
        cuda_nan = torch.isnan(out_cuda).any().item()
        cuda_inf = torch.isinf(out_cuda).any().item()
    except Exception as e:
        cuda_ok = False
        cuda_nan = True
        cuda_inf = True
        out_cuda = None
        if verbose:
            print(f"  CUDA error: {e}")

    # --- Python forward ---
    torch.manual_seed(42)  # Reset seed for same input
    x = torch.randn(B, T, dim, device=device, dtype=dtype)

    model.eval()  # Python fallback used in eval mode
    try:
        # Force Python path by temporarily disabling CUDA
        import elman.models.e88_fla_hybrid as e88_module
        orig_cuda = e88_module.E88_NATIVE_CUDA_AVAILABLE
        e88_module.E88_NATIVE_CUDA_AVAILABLE = False

        out_py, S_py = model(x)

        e88_module.E88_NATIVE_CUDA_AVAILABLE = orig_cuda

        py_ok = True
        py_nan = torch.isnan(out_py).any().item()
        py_inf = torch.isinf(out_py).any().item()
    except Exception as e:
        py_ok = False
        py_nan = True
        py_inf = True
        out_py = None
        if verbose:
            print(f"  Python error: {e}")

    # --- Compare ---
    if cuda_ok and py_ok and not cuda_nan and not py_nan:
        # Compute difference
        diff = (out_cuda.float() - out_py.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        rel_diff = (diff / (out_py.float().abs() + 1e-6)).mean().item()
    else:
        max_diff = float('inf')
        mean_diff = float('inf')
        rel_diff = float('inf')

    return {
        'n_heads': n_heads,
        'n_state': n_state,
        'dim': dim,
        'cuda_ok': cuda_ok,
        'py_ok': py_ok,
        'cuda_nan': cuda_nan,
        'py_nan': py_nan,
        'cuda_inf': cuda_inf,
        'py_inf': py_inf,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'rel_diff': rel_diff,
    }


def test_backward(n_heads, n_state, dim=512, T=32, B=2, expansion=1.0, verbose=False):
    """Test backward pass for a configuration."""
    device = 'cuda'
    dtype = torch.bfloat16

    # Create model
    model = E88FLAHybrid(
        dim=dim,
        n_state=n_state,
        n_heads=n_heads,
        expansion=expansion,
        use_conv=False,
        use_gate=False,
        use_output_norm=False,
    ).to(device).to(dtype)

    # --- CUDA backward ---
    torch.manual_seed(42)
    x = torch.randn(B, T, dim, device=device, dtype=dtype, requires_grad=True)

    model.train()
    model.zero_grad()

    try:
        out_cuda, _ = model(x, use_cuda=True)
        loss = out_cuda.sum()
        loss.backward()

        cuda_grad = x.grad.clone()
        cuda_ok = True
        cuda_grad_nan = torch.isnan(cuda_grad).any().item()
        cuda_grad_inf = torch.isinf(cuda_grad).any().item()
    except Exception as e:
        cuda_ok = False
        cuda_grad = None
        cuda_grad_nan = True
        cuda_grad_inf = True
        if verbose:
            print(f"  CUDA backward error: {e}")

    # --- Python backward ---
    import elman.models.e88_fla_hybrid as e88_module
    orig_cuda = e88_module.E88_NATIVE_CUDA_AVAILABLE
    e88_module.E88_NATIVE_CUDA_AVAILABLE = False

    torch.manual_seed(42)
    x = torch.randn(B, T, dim, device=device, dtype=dtype, requires_grad=True)

    model.zero_grad()

    try:
        out_py, _ = model(x)
        loss = out_py.sum()
        loss.backward()

        py_grad = x.grad.clone()
        py_ok = True
        py_grad_nan = torch.isnan(py_grad).any().item()
        py_grad_inf = torch.isinf(py_grad).any().item()
    except Exception as e:
        py_ok = False
        py_grad = None
        py_grad_nan = True
        py_grad_inf = True
        if verbose:
            print(f"  Python backward error: {e}")

    e88_module.E88_NATIVE_CUDA_AVAILABLE = orig_cuda

    # --- Compare ---
    if cuda_ok and py_ok and cuda_grad is not None and py_grad is not None:
        diff = (cuda_grad.float() - py_grad.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
    else:
        max_diff = float('inf')
        mean_diff = float('inf')

    return {
        'n_heads': n_heads,
        'n_state': n_state,
        'cuda_grad_nan': cuda_grad_nan,
        'py_grad_nan': py_grad_nan,
        'cuda_grad_inf': cuda_grad_inf,
        'py_grad_inf': py_grad_inf,
        'grad_max_diff': max_diff,
        'grad_mean_diff': mean_diff,
    }


def main():
    print("=" * 80)
    print("E88 CUDA vs Python Numerical Accuracy Test")
    print("=" * 80)

    if not E88_NATIVE_CUDA_AVAILABLE:
        print("ERROR: E88 CUDA kernel not available!")
        return

    # Test configurations
    n_heads_list = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    n_state_list = [16, 24, 32, 48, 64, 96, 128]

    print("\n" + "=" * 80)
    print("FORWARD PASS COMPARISON")
    print("=" * 80)
    print(f"{'n_heads':>8} {'n_state':>8} {'CUDA':>6} {'Python':>6} {'C_NaN':>6} {'P_NaN':>6} {'MaxDiff':>12} {'RelDiff':>12}")
    print("-" * 80)

    issues = []

    for n_state in n_state_list:
        for n_heads in n_heads_list:
            # Adjust dim to fit GPU memory
            if n_heads > 64:
                dim = 256
            elif n_heads > 32:
                dim = 384
            else:
                dim = 512

            result = test_config(n_heads, n_state, dim=dim)

            cuda_str = "OK" if result['cuda_ok'] and not result['cuda_nan'] else "FAIL"
            py_str = "OK" if result['py_ok'] and not result['py_nan'] else "FAIL"
            c_nan = "NaN" if result['cuda_nan'] else "-"
            p_nan = "NaN" if result['py_nan'] else "-"

            max_diff = f"{result['max_diff']:.2e}" if result['max_diff'] < 1e10 else "INF"
            rel_diff = f"{result['rel_diff']:.2e}" if result['rel_diff'] < 1e10 else "INF"

            # Flag issues
            is_issue = (result['cuda_nan'] or result['max_diff'] > 1e-2 or
                       (result['cuda_nan'] and not result['py_nan']))
            marker = " ***" if is_issue else ""

            print(f"{n_heads:>8} {n_state:>8} {cuda_str:>6} {py_str:>6} {c_nan:>6} {p_nan:>6} {max_diff:>12} {rel_diff:>12}{marker}")

            if is_issue:
                issues.append(result)

    print("\n" + "=" * 80)
    print("BACKWARD PASS COMPARISON (subset)")
    print("=" * 80)
    print(f"{'n_heads':>8} {'n_state':>8} {'C_grad_NaN':>10} {'P_grad_NaN':>10} {'GradMaxDiff':>12}")
    print("-" * 60)

    # Test backward for subset
    test_configs_bwd = [
        (8, 32),   # Known working
        (12, 32),  # Known working
        (16, 32),  # Boundary
        (32, 32),  # Failing
        (8, 48),   # Known working
        (36, 48),  # Failing
        (8, 64),   # Boundary
        (16, 64),  # Likely failing
    ]

    for n_heads, n_state in test_configs_bwd:
        dim = 384 if n_heads > 16 else 512
        result = test_backward(n_heads, n_state, dim=dim, T=16, B=2)

        c_nan = "NaN" if result['cuda_grad_nan'] else "OK"
        p_nan = "NaN" if result['py_grad_nan'] else "OK"
        max_diff = f"{result['grad_max_diff']:.2e}" if result['grad_max_diff'] < 1e10 else "INF"

        is_issue = result['cuda_grad_nan'] and not result['py_grad_nan']
        marker = " ***" if is_issue else ""

        print(f"{n_heads:>8} {n_state:>8} {c_nan:>10} {p_nan:>10} {max_diff:>12}{marker}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total forward issues found: {len(issues)}")
    if issues:
        print("\nProblematic configurations:")
        for r in issues[:10]:  # Show first 10
            print(f"  n_heads={r['n_heads']}, n_state={r['n_state']}: "
                  f"CUDA_NaN={r['cuda_nan']}, Py_NaN={r['py_nan']}, MaxDiff={r['max_diff']:.2e}")


if __name__ == '__main__':
    main()
