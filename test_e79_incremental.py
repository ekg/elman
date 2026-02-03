"""Test E79 input-bias with incremental complexity."""
import sys
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, '/home/erikg/elman/elman/cuda')

import torch
import torch.nn.functional as F
import hasty_pytorch_lib
from elman.models.e79_coupled_matrix import E79CoupledMatrixCell

def test_config(T, B, dim, n_state, init_scale=0.0, name=""):
    print(f"\n{name}: T={T}, B={B}, dim={dim}, n_state={n_state}, init_scale={init_scale}")
    print("-"*60)

    torch.manual_seed(42)

    x = torch.randn(T, B, dim, device='cuda', dtype=torch.float32)
    S0 = torch.randn(B, n_state, n_state, device='cuda', dtype=torch.float32) * init_scale
    M0 = torch.randn(B, n_state, n_state, device='cuda', dtype=torch.float32) * init_scale

    torch.manual_seed(42)
    cell = E79CoupledMatrixCell(
        dim=dim,
        n_state=n_state,
        use_cuda=True,
        input_bias=True,
    ).cuda().float()

    # CUDA forward/backward
    x_cuda = x.clone().requires_grad_(True)
    output_cuda, S_cuda, M_cuda = cell(x_cuda, S0.clone(), M0.clone())
    loss_cuda = output_cuda.sum()
    loss_cuda.backward()
    dx_cuda = x_cuda.grad.clone()
    dW_kvqm_cuda = cell.W_kvqm.grad.clone()

    cell.zero_grad()

    # Python forward/backward
    cell.use_cuda = False
    x_py = x.clone().requires_grad_(True)
    output_py, S_py, M_py = cell(x_py, S0.clone(), M0.clone())
    loss_py = output_py.sum()
    loss_py.backward()
    dx_py = x_py.grad.clone()
    dW_kvqm_py = cell.W_kvqm.grad.clone()

    output_diff = (output_cuda - output_py).abs().max().item()
    dx_diff = (dx_cuda - dx_py).abs().max().item()
    dW_diff = (dW_kvqm_cuda - dW_kvqm_py).abs().max().item()

    print(f"  Output diff: {output_diff:.2e}")
    print(f"  dx diff: {dx_diff:.2e}")
    print(f"  dW_kvqm diff: {dW_diff:.2e}")

    pass_thresh = 1e-4
    status = "PASS" if dx_diff < pass_thresh and dW_diff < pass_thresh else "FAIL"
    print(f"  Status: {status}")
    return dx_diff < pass_thresh and dW_diff < pass_thresh

if __name__ == "__main__":
    results = []

    # Test 1: Base case (should pass)
    results.append(test_config(2, 1, 32, 16, 0.0, "Test 1: Base"))

    # Test 2: Non-zero initial M0
    results.append(test_config(2, 1, 32, 16, 0.1, "Test 2: Non-zero M0"))

    # Test 3: Larger T
    results.append(test_config(4, 1, 32, 16, 0.0, "Test 3: Larger T"))

    # Test 4: Larger B
    results.append(test_config(2, 2, 32, 16, 0.0, "Test 4: Larger B"))

    # Test 5: Combined
    results.append(test_config(4, 2, 32, 16, 0.0, "Test 5: Combined"))

    # Test 6: Even larger
    results.append(test_config(8, 4, 64, 32, 0.0, "Test 6: Full zero init"))

    # Test 7: Full test with non-zero init
    results.append(test_config(8, 4, 64, 32, 0.1, "Test 7: Full non-zero init"))

    print("\n" + "="*60)
    print(f"Results: {sum(results)}/{len(results)} passed")
