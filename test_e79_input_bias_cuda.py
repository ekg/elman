"""
Test script to validate E79 input-bias CUDA kernel against Python reference implementation.
"""

import os
import sys

# Set up library path before importing torch
os.environ['LD_LIBRARY_PATH'] = '/home/erikg/.local/lib/python3.12/site-packages/torch/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

# Add paths for hasty_pytorch_lib
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, '/home/erikg/elman/elman/cuda')
sys.path.insert(0, '/home/erikg/elman/elman/cuda/build/lib.linux-x86_64-cpython-312')

import torch
import torch.nn.functional as F

# Import hasty_pytorch_lib directly first
try:
    import hasty_pytorch_lib
    E79_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e79_coupled_forward') and hasattr(hasty_pytorch_lib, 'e79_coupled_backward')
    E79_INPUT_BIAS_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e79_coupled_input_bias_forward') and hasattr(hasty_pytorch_lib, 'e79_coupled_input_bias_backward')
except ImportError as e:
    print(f"Failed to import hasty_pytorch_lib: {e}")
    E79_CUDA_AVAILABLE = False
    E79_INPUT_BIAS_CUDA_AVAILABLE = False

from elman.models.e79_coupled_matrix import E79CoupledMatrixCell

def test_e79_regular():
    """Test regular (non-input-bias) E79 kernel."""
    print("="*60)
    print("Testing REGULAR E79 (input_bias=False)")
    print("="*60)

    torch.manual_seed(42)
    T, B, dim, n_state = 1, 2, 64, 32

    x = torch.randn(T, B, dim, device='cuda', dtype=torch.float32)
    S0 = torch.zeros(B, n_state, n_state, device='cuda', dtype=torch.float32)
    M0 = torch.zeros(B, n_state, n_state, device='cuda', dtype=torch.float32)

    torch.manual_seed(42)
    cell = E79CoupledMatrixCell(
        dim=dim,
        n_state=n_state,
        use_cuda=True,
        input_bias=False,
        use_bias=True,
    ).cuda().float()

    # CUDA forward
    output_cuda, S_cuda, M_cuda = cell(x.clone(), S0.clone(), M0.clone())

    # Python forward
    cell.use_cuda = False
    output_python, S_python, M_python = cell(x.clone(), S0.clone(), M0.clone())
    cell.use_cuda = True

    print(f"Output diff: {(output_cuda - output_python).abs().max().item():.2e}")
    print(f"S diff: {(S_cuda - S_python).abs().max().item():.2e}")
    print(f"M diff: {(M_cuda - M_python).abs().max().item():.2e}")

def test_e79_input_bias():
    """Test input-bias E79 kernel."""
    print("\n" + "="*60)
    print("Testing INPUT-BIAS E79 (input_bias=True)")
    print("="*60)

    torch.manual_seed(42)
    T, B, dim, n_state = 8, 4, 64, 32

    x = torch.randn(T, B, dim, device='cuda', dtype=torch.float32)
    S0 = torch.randn(B, n_state, n_state, device='cuda', dtype=torch.float32) * 0.1
    M0 = torch.randn(B, n_state, n_state, device='cuda', dtype=torch.float32) * 0.1

    torch.manual_seed(42)
    cell = E79CoupledMatrixCell(
        dim=dim,
        n_state=n_state,
        use_cuda=True,
        input_bias=True,
    ).cuda().float()

    # CUDA forward
    x_cuda = x.clone().requires_grad_(True)
    output_cuda, S_cuda, M_cuda = cell(x_cuda, S0.clone(), M0.clone())

    # Backward
    loss_cuda = output_cuda.sum()
    loss_cuda.backward()
    dx_cuda = x_cuda.grad.clone()
    dW_kvqm_cuda = cell.W_kvqm.grad.clone()
    dW_bs_cuda = cell.W_bs.grad.clone()
    dW_bm_cuda = cell.W_bm.grad.clone()

    cell.zero_grad()

    # Python forward
    cell.use_cuda = False
    x_python = x.clone().requires_grad_(True)
    output_python, S_python, M_python = cell(x_python, S0.clone(), M0.clone())

    loss_python = output_python.sum()
    loss_python.backward()
    dx_python = x_python.grad.clone()
    dW_kvqm_python = cell.W_kvqm.grad.clone()
    dW_bs_python = cell.W_bs.grad.clone() if cell.W_bs.grad is not None else torch.zeros_like(cell.W_bs)
    dW_bm_python = cell.W_bm.grad.clone() if cell.W_bm.grad is not None else torch.zeros_like(cell.W_bm)

    cell.use_cuda = True

    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)

    output_diff = (output_cuda - output_python).abs().max().item()
    S_diff = (S_cuda - S_python).abs().max().item()
    M_diff = (M_cuda - M_python).abs().max().item()
    dx_diff = (dx_cuda - dx_python).abs().max().item()
    dW_kvqm_diff = (dW_kvqm_cuda - dW_kvqm_python).abs().max().item()
    dW_bs_diff = (dW_bs_cuda - dW_bs_python).abs().max().item()
    dW_bm_diff = (dW_bm_cuda - dW_bm_python).abs().max().item()

    print(f"Output max abs diff: {output_diff:.2e}")
    print(f"Final S max abs diff: {S_diff:.2e}")
    print(f"Final M max abs diff: {M_diff:.2e}")
    print(f"dx max abs diff: {dx_diff:.2e}")
    print(f"dW_kvqm max abs diff: {dW_kvqm_diff:.2e}")
    print(f"dW_bs max abs diff: {dW_bs_diff:.2e}")
    print(f"dW_bm max abs diff: {dW_bm_diff:.2e}")

    # Validation
    print("\n" + "="*60)
    print("VALIDATION")
    print("="*60)

    output_ok = output_diff < 1e-4
    S_ok = S_diff < 1e-4
    M_ok = M_diff < 1e-4
    dx_ok = dx_diff < 1e-3
    dW_kvqm_ok = dW_kvqm_diff < 1e-3
    dW_bs_ok = dW_bs_diff < 1e-3
    dW_bm_ok = dW_bm_diff < 1e-3

    print(f"Output diff < 1e-4: {'PASS' if output_ok else 'FAIL'}")
    print(f"Final S diff < 1e-4: {'PASS' if S_ok else 'FAIL'}")
    print(f"Final M diff < 1e-4: {'PASS' if M_ok else 'FAIL'}")
    print(f"dx diff < 1e-3: {'PASS' if dx_ok else 'FAIL'}")
    print(f"dW_kvqm diff < 1e-3: {'PASS' if dW_kvqm_ok else 'FAIL'}")
    print(f"dW_bs diff < 1e-3: {'PASS' if dW_bs_ok else 'FAIL'}")
    print(f"dW_bm diff < 1e-3: {'PASS' if dW_bm_ok else 'FAIL'}")

    all_pass = all([output_ok, S_ok, M_ok, dx_ok, dW_kvqm_ok, dW_bs_ok, dW_bm_ok])
    print(f"\n{'='*60}")
    print(f"OVERALL: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print(f"{'='*60}")

    return all_pass

if __name__ == "__main__":
    print(f"E79 CUDA available: {E79_CUDA_AVAILABLE}")
    print(f"E79 Input-Bias CUDA available: {E79_INPUT_BIAS_CUDA_AVAILABLE}")

    if not E79_CUDA_AVAILABLE or not E79_INPUT_BIAS_CUDA_AVAILABLE:
        print("\nERROR: CUDA kernels not available!")
        sys.exit(1)

    test_e79_regular()
    test_e79_input_bias()
