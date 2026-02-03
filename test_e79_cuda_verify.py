"""
E79 CUDA Kernel Verification Script

Compares forward and backward passes between Python fallback and CUDA kernel.
Reports max absolute differences for outputs and gradients.
"""

import torch
import torch.nn as nn
import sys

# Config - as specified in the task
B = 2
T = 16
dim = 256
n_state = 32
seed = 42
# Use float32 for verification (better numerical precision)
dtype = torch.float32

def main():
    print("=" * 70)
    print("E79 CUDA Kernel Verification")
    print("=" * 70)
    print(f"Config: B={B}, T={T}, dim={dim}, n_state={n_state}")
    print(f"dtype={dtype}")
    print()

    # Set seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Import E79 module
    from elman.models.e79_coupled_matrix import E79CoupledMatrixCell, E79_CUDA_AVAILABLE

    print(f"E79 CUDA kernel available: {E79_CUDA_AVAILABLE}")
    if not E79_CUDA_AVAILABLE:
        print("ERROR: CUDA kernel not available. Cannot perform comparison.")
        sys.exit(1)

    # Create identical input data
    x = torch.randn(T, B, dim, device='cuda', dtype=dtype, requires_grad=True)
    S0 = torch.randn(B, n_state, n_state, device='cuda', dtype=dtype) * 0.1
    M0 = torch.randn(B, n_state, n_state, device='cuda', dtype=dtype) * 0.1

    # Create gradient for backward pass
    d_output = torch.randn(T, B, n_state, device='cuda', dtype=dtype)

    # =========================================================================
    # Python fallback forward/backward
    # =========================================================================
    print("Running Python fallback forward/backward...")

    cell_py = E79CoupledMatrixCell(dim=dim, n_state=n_state, use_cuda=False).cuda().to(dtype)
    cell_py.train()

    # Clone inputs for Python
    x_py = x.detach().clone().requires_grad_(True)
    S0_py = S0.detach().clone()
    M0_py = M0.detach().clone()

    # Forward
    output_py, S_py, M_py = cell_py(x_py, S0_py, M0_py)

    # Backward
    output_py.backward(d_output)

    # Get gradients
    dW_kvqm_py = cell_py.W_kvqm.grad.clone()
    db_s_gate_py = cell_py.b_s_gate.grad.clone()
    db_m_gate_py = cell_py.b_m_gate.grad.clone()
    dx_py = x_py.grad.clone()

    print("  Forward output shape:", output_py.shape)
    print("  dW_kvqm shape:", dW_kvqm_py.shape)
    print("  db_s_gate shape:", db_s_gate_py.shape)
    print("  db_m_gate shape:", db_m_gate_py.shape)
    print()

    # =========================================================================
    # CUDA kernel forward/backward
    # =========================================================================
    print("Running CUDA kernel forward/backward...")

    cell_cuda = E79CoupledMatrixCell(dim=dim, n_state=n_state, use_cuda=True).cuda().to(dtype)
    cell_cuda.train()

    # Copy weights from Python cell to ensure identical parameters
    cell_cuda.W_kvqm.data.copy_(cell_py.W_kvqm.data)
    cell_cuda.b_s_gate.data.copy_(cell_py.b_s_gate.data)
    cell_cuda.b_m_gate.data.copy_(cell_py.b_m_gate.data)

    # Clone inputs for CUDA
    x_cuda = x.detach().clone().requires_grad_(True)
    S0_cuda = S0.detach().clone()
    M0_cuda = M0.detach().clone()

    # Forward
    output_cuda, S_cuda, M_cuda = cell_cuda(x_cuda, S0_cuda, M0_cuda)

    # Backward
    output_cuda.backward(d_output)

    # Get gradients
    dW_kvqm_cuda = cell_cuda.W_kvqm.grad.clone()
    db_s_gate_cuda = cell_cuda.b_s_gate.grad.clone()
    db_m_gate_cuda = cell_cuda.b_m_gate.grad.clone()
    dx_cuda = x_cuda.grad.clone()

    print("  Forward output shape:", output_cuda.shape)
    print("  dW_kvqm shape:", dW_kvqm_cuda.shape)
    print("  db_s_gate shape:", db_s_gate_cuda.shape)
    print("  db_m_gate shape:", db_m_gate_cuda.shape)
    print()

    # =========================================================================
    # Compare results
    # =========================================================================
    print("=" * 70)
    print("Comparison Results")
    print("=" * 70)

    def compare(name, py_tensor, cuda_tensor):
        diff = (py_tensor - cuda_tensor).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        py_norm = py_tensor.abs().mean().item()
        cuda_norm = cuda_tensor.abs().mean().item()
        rel_diff = max_diff / (max(py_norm, cuda_norm) + 1e-8)

        status = "PASS" if max_diff < 1e-3 else ("WARN" if max_diff < 1e-2 else "FAIL")
        print(f"{name:25s}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, rel_diff={rel_diff:.6e} [{status}]")
        return max_diff, status

    print("\nForward pass:")
    output_diff, output_status = compare("output", output_py, output_cuda)
    S_diff, S_status = compare("S (final state)", S_py, S_cuda)
    M_diff, M_status = compare("M (final state)", M_py, M_cuda)

    print("\nBackward pass (gradients):")
    dx_diff, dx_status = compare("dx (input grad)", dx_py, dx_cuda)
    dW_diff, dW_status = compare("dW_kvqm", dW_kvqm_py, dW_kvqm_cuda)
    db_s_diff, db_s_status = compare("db_s_gate", db_s_gate_py, db_s_gate_cuda)
    db_m_diff, db_m_status = compare("db_m_gate", db_m_gate_py, db_m_gate_cuda)

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    all_results = [
        ("Forward output", output_diff, output_status),
        ("Final S state", S_diff, S_status),
        ("Final M state", M_diff, M_status),
        ("dx gradient", dx_diff, dx_status),
        ("dW_kvqm gradient", dW_diff, dW_status),
        ("db_s_gate gradient", db_s_diff, db_s_status),
        ("db_m_gate gradient", db_m_diff, db_m_status),
    ]

    print("\nRequested metrics:")
    print(f"  Forward output max abs diff:  {output_diff:.6e}")
    print(f"  dW_kvqm max abs diff:         {dW_diff:.6e}")
    print(f"  db_s_gate max abs diff:       {db_s_diff:.6e}")
    print(f"  db_m_gate max abs diff:       {db_m_diff:.6e}")

    fail_count = sum(1 for _, _, s in all_results if s == "FAIL")
    warn_count = sum(1 for _, _, s in all_results if s == "WARN")
    pass_count = sum(1 for _, _, s in all_results if s == "PASS")

    print(f"\nOverall: {pass_count} PASS, {warn_count} WARN, {fail_count} FAIL")

    # Additional debugging: check gradient magnitudes
    print("\n" + "=" * 70)
    print("Gradient Magnitude Analysis")
    print("=" * 70)
    print(f"Python dW_kvqm:   mean={dW_kvqm_py.abs().mean().item():.4f}, max={dW_kvqm_py.abs().max().item():.4f}")
    print(f"CUDA dW_kvqm:     mean={dW_kvqm_cuda.abs().mean().item():.4f}, max={dW_kvqm_cuda.abs().max().item():.4f}")
    print(f"Python db_s_gate: mean={db_s_gate_py.abs().mean().item():.4f}, max={db_s_gate_py.abs().max().item():.4f}")
    print(f"CUDA db_s_gate:   mean={db_s_gate_cuda.abs().mean().item():.4f}, max={db_s_gate_cuda.abs().max().item():.4f}")
    print(f"Python db_m_gate: mean={db_m_gate_py.abs().mean().item():.4f}, max={db_m_gate_py.abs().max().item():.4f}")
    print(f"CUDA db_m_gate:   mean={db_m_gate_cuda.abs().mean().item():.4f}, max={db_m_gate_cuda.abs().max().item():.4f}")
    print(f"Python dx:        mean={dx_py.abs().mean().item():.4f}, max={dx_py.abs().max().item():.4f}")
    print(f"CUDA dx:          mean={dx_cuda.abs().mean().item():.4f}, max={dx_cuda.abs().max().item():.4f}")

    if fail_count > 0:
        print("\nVERIFICATION FAILED - CUDA kernel has significant differences from Python")
        return 1
    elif warn_count > 0:
        print("\nVERIFICATION WARNING - Some differences are above typical threshold but may be acceptable")
        return 0
    else:
        print("\nVERIFICATION PASSED - CUDA kernel matches Python fallback")
        return 0

if __name__ == "__main__":
    sys.exit(main())
