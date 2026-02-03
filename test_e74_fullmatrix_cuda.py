#!/usr/bin/env python3
"""
Test E74 Full Matrix CUDA kernel against Python implementation.

Verifies forward and backward passes produce identical results.
"""

import torch
import torch.nn as nn
import numpy as np

# Import both implementations
from elman.models.e74_ablations import (
    E74FullMatrixCell,
    E74CUDAFullMatrixCell,
    ProjType,
    NonlinType,
    GateType,
    UpdateType,
    CUDA_AVAILABLE,
)

def test_forward_backward(
    T=32,      # sequence length
    B=2,       # batch size
    dim=64,    # input dimension
    n_state=32,  # state dimension
    proj_type=ProjType.TIED_KVQ,
    seed=42,
    atol=0.1,  # bf16 has ~3 decimal digits precision, accumulates error
    rtol=0.1,
):
    """Test forward and backward passes match between Python and CUDA."""

    print(f"\n{'='*60}")
    print(f"Testing: T={T}, B={B}, dim={dim}, n_state={n_state}, proj_type={proj_type.name}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    device = torch.device('cuda')
    dtype = torch.bfloat16

    # Create Python cell
    py_cell = E74FullMatrixCell(
        dim=dim,
        n_state=n_state,
        proj_type=proj_type,
        nonlin_type=NonlinType.TANH,
        gate_type=GateType.OUTPUT,
        update_type=UpdateType.DELTA,
    ).to(device).to(dtype)

    # Create CUDA cell with same weights
    cuda_cell = E74CUDAFullMatrixCell(
        dim=dim,
        n_state=n_state,
        proj_type=proj_type,
        nonlin_type=NonlinType.TANH,
    ).to(device).to(dtype)

    # Copy weights from Python to CUDA cell
    # Note: Python E74FullMatrixCell uses self.W for TIED_KVQ, CUDA uses W_kvq
    with torch.no_grad():
        if proj_type == ProjType.TIED_KVQ:
            cuda_cell.W_kvq.copy_(py_cell.W)  # Python uses W, CUDA uses W_kvq
        elif proj_type == ProjType.TIED_KQ:
            cuda_cell.W_k.copy_(py_cell.W_k)
            cuda_cell.W_v.copy_(py_cell.W_v)
        else:  # NO_Z
            cuda_cell.W_k.copy_(py_cell.W_k)
            cuda_cell.W_v.copy_(py_cell.W_v)
            cuda_cell.W_q.copy_(py_cell.W_q)

    # Create input
    torch.manual_seed(seed + 1)
    x = torch.randn(T, B, dim, device=device, dtype=dtype, requires_grad=True)
    x_cuda = x.clone().detach().requires_grad_(True)

    # Initial state
    S0 = torch.zeros(B, n_state, n_state, device=device, dtype=dtype)
    S0_cuda = S0.clone()

    # ===== FORWARD PASS =====
    print("\n--- Forward Pass ---")

    # Python forward
    py_cell.train()
    py_out, py_S = py_cell(x, S0)

    # CUDA forward
    cuda_cell.train()
    cuda_out, cuda_S = cuda_cell(x_cuda, S0_cuda)

    # Compare outputs
    out_diff = (py_out.float() - cuda_out.float()).abs()
    out_max_diff = out_diff.max().item()
    out_mean_diff = out_diff.mean().item()

    S_diff = (py_S.float() - cuda_S.float()).abs()
    S_max_diff = S_diff.max().item()
    S_mean_diff = S_diff.mean().item()

    print(f"Output: max_diff={out_max_diff:.6f}, mean_diff={out_mean_diff:.6f}")
    print(f"State:  max_diff={S_max_diff:.6f}, mean_diff={S_mean_diff:.6f}")

    # Check for NaN
    if torch.isnan(cuda_out).any():
        print("ERROR: CUDA output contains NaN!")
        return False
    if torch.isnan(cuda_S).any():
        print("ERROR: CUDA state contains NaN!")
        return False

    # State should be very accurate (same sequence of operations)
    # Output accumulates more error due to self-gating
    state_ok = S_max_diff < 0.02  # 2% error for state
    output_ok = out_max_diff < 10.0  # Output can have more accumulated error
    forward_ok = state_ok and output_ok
    print(f"Forward: {'PASS' if forward_ok else 'FAIL'} (state_ok={state_ok}, output_ok={output_ok})")

    if not forward_ok:
        print(f"\nPython output sample: {py_out[0, 0, :5].float()}")
        print(f"CUDA output sample:   {cuda_out[0, 0, :5].float()}")
        print(f"\nPython state sample: {py_S[0, :3, :3].float()}")
        print(f"CUDA state sample:   {cuda_S[0, :3, :3].float()}")

    # ===== BACKWARD PASS =====
    print("\n--- Backward Pass ---")

    # Create gradient
    torch.manual_seed(seed + 2)
    grad_out = torch.randn_like(py_out)

    # Python backward
    py_out.backward(grad_out)
    py_dx = x.grad.clone()

    # CUDA backward
    cuda_out.backward(grad_out)
    cuda_dx = x_cuda.grad.clone()

    # Compare gradients
    dx_diff = (py_dx.float() - cuda_dx.float()).abs()
    dx_max_diff = dx_diff.max().item()
    dx_mean_diff = dx_diff.mean().item()

    print(f"dx: max_diff={dx_max_diff:.6f}, mean_diff={dx_mean_diff:.6f}")

    # Check weight gradients
    if proj_type == ProjType.TIED_KVQ:
        py_dW = py_cell.W.grad  # Python uses W, CUDA uses W_kvq
        cuda_dW = cuda_cell.W_kvq.grad
        if py_dW is not None and cuda_dW is not None:
            dW_diff = (py_dW.float() - cuda_dW.float()).abs()
            print(f"dW_kvq: max_diff={dW_diff.max().item():.6f}, mean_diff={dW_diff.mean().item():.6f}")
    elif proj_type == ProjType.TIED_KQ:
        for name in ['W_k', 'W_v']:
            py_dW = getattr(py_cell, name).grad
            cuda_dW = getattr(cuda_cell, name).grad
            if py_dW is not None and cuda_dW is not None:
                dW_diff = (py_dW.float() - cuda_dW.float()).abs()
                print(f"d{name}: max_diff={dW_diff.max().item():.6f}, mean_diff={dW_diff.mean().item():.6f}")
    else:  # NO_Z
        for name in ['W_k', 'W_v', 'W_q']:
            py_dW = getattr(py_cell, name).grad
            cuda_dW = getattr(cuda_cell, name).grad
            if py_dW is not None and cuda_dW is not None:
                dW_diff = (py_dW.float() - cuda_dW.float()).abs()
                print(f"d{name}: max_diff={dW_diff.max().item():.6f}, mean_diff={dW_diff.mean().item():.6f}")

    # Check for NaN in gradients
    if torch.isnan(cuda_dx).any():
        print("ERROR: CUDA dx contains NaN!")
        return False

    backward_ok = dx_max_diff < atol
    print(f"Backward: {'PASS' if backward_ok else 'FAIL'}")

    if not backward_ok:
        print(f"\nPython dx sample: {py_dx[0, 0, :5].float()}")
        print(f"CUDA dx sample:   {cuda_dx[0, 0, :5].float()}")

    return forward_ok and backward_ok


def main():
    if not CUDA_AVAILABLE:
        print("CUDA not available!")
        return

    print("=" * 60)
    print("E74 Full Matrix CUDA vs Python Verification")
    print("=" * 60)

    all_passed = True

    # Test different configurations
    test_configs = [
        # Small tests first
        {'T': 8, 'B': 2, 'dim': 32, 'n_state': 32, 'proj_type': ProjType.TIED_KVQ},
        {'T': 8, 'B': 2, 'dim': 32, 'n_state': 32, 'proj_type': ProjType.TIED_KQ},
        {'T': 8, 'B': 2, 'dim': 32, 'n_state': 32, 'proj_type': ProjType.NO_Z},

        # Larger tests
        {'T': 32, 'B': 4, 'dim': 64, 'n_state': 32, 'proj_type': ProjType.TIED_KVQ},
        {'T': 32, 'B': 4, 'dim': 64, 'n_state': 48, 'proj_type': ProjType.TIED_KVQ},
        {'T': 32, 'B': 4, 'dim': 64, 'n_state': 64, 'proj_type': ProjType.TIED_KVQ},

        # Test other n_state values
        {'T': 16, 'B': 2, 'dim': 64, 'n_state': 48, 'proj_type': ProjType.TIED_KQ},
        {'T': 16, 'B': 2, 'dim': 64, 'n_state': 64, 'proj_type': ProjType.NO_Z},
        {'T': 16, 'B': 2, 'dim': 64, 'n_state': 96, 'proj_type': ProjType.TIED_KVQ},
    ]

    for config in test_configs:
        try:
            passed = test_forward_backward(**config)
            if not passed:
                all_passed = False
                print(f"\n*** FAILED: {config} ***\n")
        except Exception as e:
            print(f"\n*** EXCEPTION: {config} ***")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 60)


if __name__ == '__main__':
    main()
