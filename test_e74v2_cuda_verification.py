#!/usr/bin/env python3
"""
E74v2 Full Matrix CUDA Kernel Verification

Compares forward and backward passes between Python fallback and CUDA kernel
implementations to verify correctness.

Test configuration:
- B=2, T=16, dim=256, n_state=32
- update_type='delta' (standard delta rule)
- gate_type='output' (self-gating)
- proj_type='no_z' (separate k, v, q projections)
"""

import torch
import torch.nn.functional as F
import sys

# Import E74v2
from elman.models.e74_v2 import (
    E74v2Cell,
    E74V2_CUDA_AVAILABLE,
    E74_CUDA_AVAILABLE,
)


def compare_tensors(name: str, py_tensor: torch.Tensor, cuda_tensor: torch.Tensor):
    """Compare two tensors and report differences."""
    if py_tensor is None and cuda_tensor is None:
        print(f"  {name}: Both None (OK)")
        return 0.0, 0.0

    if py_tensor is None or cuda_tensor is None:
        print(f"  {name}: One is None! py={py_tensor is not None}, cuda={cuda_tensor is not None}")
        return float('inf'), float('inf')

    diff = (py_tensor.float() - cuda_tensor.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Check for NaN/Inf
    has_nan_py = torch.isnan(py_tensor).any().item()
    has_nan_cuda = torch.isnan(cuda_tensor).any().item()
    has_inf_py = torch.isinf(py_tensor).any().item()
    has_inf_cuda = torch.isinf(cuda_tensor).any().item()

    status = "OK"
    if has_nan_py or has_nan_cuda:
        status = f"NaN! py={has_nan_py}, cuda={has_nan_cuda}"
    elif has_inf_py or has_inf_cuda:
        status = f"Inf! py={has_inf_py}, cuda={has_inf_cuda}"
    elif max_diff > 0.5:
        status = "HIGH DIFF"

    print(f"  {name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f} [{status}]")

    return max_diff, mean_diff


def test_e74v2_cuda(
    B=2,
    T=16,
    dim=256,
    n_state=32,
    update_type='delta',
    gate_type='output',
    proj_type='no_z',
    seed=42,
):
    """Test E74v2 CUDA kernel against Python fallback."""

    print("=" * 70)
    print(f"E74v2 CUDA Verification")
    print(f"  Config: B={B}, T={T}, dim={dim}, n_state={n_state}")
    print(f"  update_type={update_type}, gate_type={gate_type}, proj_type={proj_type}")
    print(f"  E74V2_CUDA_AVAILABLE: {E74V2_CUDA_AVAILABLE}")
    print(f"  E74_CUDA_AVAILABLE: {E74_CUDA_AVAILABLE}")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return False

    device = torch.device('cuda')
    dtype = torch.bfloat16

    # Create cell with CUDA enabled
    torch.manual_seed(seed)
    cuda_cell = E74v2Cell(
        dim=dim,
        n_state=n_state,
        proj_type=proj_type,
        update_type=update_type,
        gate_type=gate_type,
        use_tanh=True,
        use_cuda=True,
    ).to(device).to(dtype)

    # Create cell with CUDA disabled (Python fallback)
    torch.manual_seed(seed)
    py_cell = E74v2Cell(
        dim=dim,
        n_state=n_state,
        proj_type=proj_type,
        update_type=update_type,
        gate_type=gate_type,
        use_tanh=True,
        use_cuda=False,  # Force Python fallback
    ).to(device).to(dtype)

    # Copy weights from CUDA cell to Python cell to ensure identical weights
    with torch.no_grad():
        for (name_cuda, param_cuda), (name_py, param_py) in zip(
            cuda_cell.named_parameters(), py_cell.named_parameters()
        ):
            assert name_cuda == name_py, f"Parameter name mismatch: {name_cuda} vs {name_py}"
            param_py.copy_(param_cuda)

    # Verify CUDA cell will actually use CUDA
    print(f"\n  CUDA cell use_cuda flag: {cuda_cell.use_cuda}")
    print(f"  Python cell use_cuda flag: {py_cell.use_cuda}")

    # Create input data
    torch.manual_seed(seed + 1)
    x = torch.randn(T, B, dim, device=device, dtype=dtype)

    # Create separate inputs for gradient tracking
    x_cuda = x.clone().detach().requires_grad_(True)
    x_py = x.clone().detach().requires_grad_(True)

    # Initial state
    S0 = torch.zeros(B, n_state, n_state, device=device, dtype=dtype)

    # =========================================================================
    # FORWARD PASS
    # =========================================================================
    print("\n" + "-" * 50)
    print("FORWARD PASS")
    print("-" * 50)

    cuda_cell.train()
    py_cell.train()

    cuda_out, cuda_S = cuda_cell(x_cuda, S0.clone())
    py_out, py_S = py_cell(x_py, S0.clone())

    print("\nOutput comparison:")
    out_max, out_mean = compare_tensors("output", py_out, cuda_out)

    print("\nFinal state comparison:")
    S_max, S_mean = compare_tensors("final_S", py_S, cuda_S)

    # Sample values
    print("\nSample output values (first timestep, first batch, first 5 elements):")
    print(f"  Python: {py_out[0, 0, :5].float().tolist()}")
    print(f"  CUDA:   {cuda_out[0, 0, :5].float().tolist()}")

    print("\nSample state values (first batch, [0:3, 0:3]):")
    print(f"  Python:\n{py_S[0, :3, :3].float()}")
    print(f"  CUDA:\n{cuda_S[0, :3, :3].float()}")

    forward_ok = (out_max < 0.5 and S_max < 0.1 and
                  not torch.isnan(cuda_out).any() and
                  not torch.isnan(cuda_S).any())

    print(f"\nForward pass: {'PASS' if forward_ok else 'FAIL'}")

    # =========================================================================
    # BACKWARD PASS
    # =========================================================================
    print("\n" + "-" * 50)
    print("BACKWARD PASS")
    print("-" * 50)

    # Create gradient for backward
    torch.manual_seed(seed + 2)
    grad_out = torch.randn_like(cuda_out)

    # Backward pass
    cuda_out.backward(grad_out.clone())
    py_out.backward(grad_out.clone())

    # Compare input gradients
    print("\nInput gradient comparison:")
    dx_max, dx_mean = compare_tensors("dx", py_cell.W_k.grad is not None and x_py.grad,
                                       cuda_cell.W_k.grad is not None and x_cuda.grad)

    # Actually compare
    if x_py.grad is not None and x_cuda.grad is not None:
        dx_max, dx_mean = compare_tensors("dx", x_py.grad, x_cuda.grad)

    print("\nWeight gradient comparison:")

    # Compare weight gradients based on proj_type
    weight_grads = {}
    if proj_type == 'tied_kvq':
        weight_grads['W_kvq'] = (py_cell.W_kvq.grad, cuda_cell.W_kvq.grad)
    elif proj_type == 'tied_kq':
        weight_grads['W_k'] = (py_cell.W_k.grad, cuda_cell.W_k.grad)
        weight_grads['W_v'] = (py_cell.W_v.grad, cuda_cell.W_v.grad)
    else:  # no_z
        weight_grads['W_k'] = (py_cell.W_k.grad, cuda_cell.W_k.grad)
        weight_grads['W_v'] = (py_cell.W_v.grad, cuda_cell.W_v.grad)
        weight_grads['W_q'] = (py_cell.W_q.grad, cuda_cell.W_q.grad)

    # Check update-type specific gradients
    if update_type == 'residual':
        weight_grads['residual_scale'] = (py_cell.residual_scale.grad, cuda_cell.residual_scale.grad)
    elif update_type == 'ntm':
        weight_grads['W_erase'] = (py_cell.W_erase.grad, cuda_cell.W_erase.grad)
        weight_grads['b_erase'] = (py_cell.b_erase.grad, cuda_cell.b_erase.grad)
        weight_grads['W_write'] = (py_cell.W_write.grad, cuda_cell.W_write.grad)
        weight_grads['b_write'] = (py_cell.b_write.grad, cuda_cell.b_write.grad)
    elif update_type == 'retrieved_gate':
        weight_grads['W_gate'] = (py_cell.W_gate.grad, cuda_cell.W_gate.grad)
        weight_grads['b_gate'] = (py_cell.b_gate.grad, cuda_cell.b_gate.grad)
    elif update_type == 'ema':
        weight_grads['W_alpha'] = (py_cell.W_alpha.grad, cuda_cell.W_alpha.grad)
        weight_grads['b_alpha'] = (py_cell.b_alpha.grad, cuda_cell.b_alpha.grad)

    # Check gate-type specific gradients
    if gate_type == 'input':
        weight_grads['W_z_gate'] = (py_cell.W_z_gate.grad, cuda_cell.W_z_gate.grad)
        weight_grads['b_z_gate'] = (py_cell.b_z_gate.grad, cuda_cell.b_z_gate.grad)

    max_weight_diff = 0.0
    for name, (py_grad, cuda_grad) in weight_grads.items():
        w_max, w_mean = compare_tensors(f"d{name}", py_grad, cuda_grad)
        max_weight_diff = max(max_weight_diff, w_max)

    # Sample gradient values
    print("\nSample dx values (first timestep, first batch, first 5 elements):")
    if x_py.grad is not None and x_cuda.grad is not None:
        print(f"  Python: {x_py.grad[0, 0, :5].float().tolist()}")
        print(f"  CUDA:   {x_cuda.grad[0, 0, :5].float().tolist()}")

    backward_ok = (dx_max < 0.5 and max_weight_diff < 0.5 and
                   not torch.isnan(x_cuda.grad).any())

    print(f"\nBackward pass: {'PASS' if backward_ok else 'FAIL'}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Forward output max diff:  {out_max:.6f}")
    print(f"  Forward state max diff:   {S_max:.6f}")
    print(f"  Backward dx max diff:     {dx_max:.6f}")
    print(f"  Backward weight max diff: {max_weight_diff:.6f}")

    overall_ok = forward_ok and backward_ok
    print(f"\n  Overall: {'PASS' if overall_ok else 'FAIL'}")
    print("=" * 70)

    return overall_ok


def main():
    """Run E74v2 CUDA verification tests."""

    print("\n" + "#" * 70)
    print("# E74v2 CUDA Kernel Verification Suite")
    print("#" * 70 + "\n")

    results = {}

    # Test 1: Default configuration (delta + output gate)
    print("\n[TEST 1] Default config: delta update + output gate")
    results['delta_output'] = test_e74v2_cuda(
        B=2, T=16, dim=256, n_state=32,
        update_type='delta', gate_type='output', proj_type='no_z'
    )

    # Test 2: Input gate
    print("\n\n[TEST 2] Input gate")
    results['delta_input'] = test_e74v2_cuda(
        B=2, T=16, dim=256, n_state=32,
        update_type='delta', gate_type='input', proj_type='no_z'
    )

    # Test 3: Residual update
    print("\n\n[TEST 3] Residual update")
    results['residual'] = test_e74v2_cuda(
        B=2, T=16, dim=256, n_state=32,
        update_type='residual', gate_type='output', proj_type='no_z'
    )

    # Test 4: Retrieved gate update
    print("\n\n[TEST 4] Retrieved gate update")
    results['retrieved_gate'] = test_e74v2_cuda(
        B=2, T=16, dim=256, n_state=32,
        update_type='retrieved_gate', gate_type='output', proj_type='no_z'
    )

    # Test 5: EMA update
    print("\n\n[TEST 5] EMA update")
    results['ema'] = test_e74v2_cuda(
        B=2, T=16, dim=256, n_state=32,
        update_type='ema', gate_type='output', proj_type='no_z'
    )

    # Test 6: NTM update
    print("\n\n[TEST 6] NTM update")
    results['ntm'] = test_e74v2_cuda(
        B=2, T=16, dim=256, n_state=32,
        update_type='ntm', gate_type='output', proj_type='no_z'
    )

    # Final summary
    print("\n\n" + "#" * 70)
    print("# FINAL RESULTS SUMMARY")
    print("#" * 70)

    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:20s}: {status}")

    all_passed = all(results.values())
    print("\n" + "#" * 70)
    print(f"# OVERALL: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("#" * 70 + "\n")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
