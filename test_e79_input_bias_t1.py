"""Simplified test for E79 input-bias backward."""
import sys
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, '/home/erikg/elman/elman/cuda')

import torch
import hasty_pytorch_lib
from elman.models.e79_coupled_matrix import E79CoupledMatrixCell

def test_simple():
    print("E79 Input-Bias Simple Gradient Test (T=1, B=1)")
    print("="*60)
    
    torch.manual_seed(42)
    T, B, dim, n_state = 1, 1, 32, 16
    
    x = torch.randn(T, B, dim, device='cuda', dtype=torch.float32)
    S0 = torch.zeros(B, n_state, n_state, device='cuda', dtype=torch.float32)
    M0 = torch.zeros(B, n_state, n_state, device='cuda', dtype=torch.float32)
    
    torch.manual_seed(42)
    cell = E79CoupledMatrixCell(
        dim=dim,
        n_state=n_state,
        use_cuda=True,
        input_bias=True,
    ).cuda().float()
    
    # Python forward + backward FIRST to ensure gradients exist
    x_python = x.clone().requires_grad_(True)
    cell.use_cuda_input_bias = False
    cell.train()
    output_python, S_python, M_python = cell(x_python, S0.clone(), M0.clone())
    
    loss_python = output_python.sum()
    loss_python.backward()
    
    grad_x_python = x_python.grad.clone() if x_python.grad is not None else None
    grad_W_kvqm_python = cell.W_kvqm.grad.clone() if cell.W_kvqm.grad is not None else None
    grad_W_bs_python = cell.W_bs.grad.clone() if cell.W_bs.grad is not None else None
    grad_W_bm_python = cell.W_bm.grad.clone() if cell.W_bm.grad is not None else None
    
    print(f"Python grads exist: x={grad_x_python is not None}, W_kvqm={grad_W_kvqm_python is not None}, W_bs={grad_W_bs_python is not None}, W_bm={grad_W_bm_python is not None}")
    
    cell.zero_grad()
    
    # CUDA forward + backward
    x_cuda = x.clone().requires_grad_(True)
    cell.use_cuda_input_bias = True
    output_cuda, S_cuda, M_cuda = cell(x_cuda, S0.clone(), M0.clone())
    
    loss_cuda = output_cuda.sum()
    loss_cuda.backward()
    
    grad_x_cuda = x_cuda.grad.clone() if x_cuda.grad is not None else None
    grad_W_kvqm_cuda = cell.W_kvqm.grad.clone() if cell.W_kvqm.grad is not None else None
    grad_W_bs_cuda = cell.W_bs.grad.clone() if cell.W_bs.grad is not None else None
    grad_W_bm_cuda = cell.W_bm.grad.clone() if cell.W_bm.grad is not None else None
    
    print(f"CUDA grads exist: x={grad_x_cuda is not None}, W_kvqm={grad_W_kvqm_cuda is not None}, W_bs={grad_W_bs_cuda is not None}, W_bm={grad_W_bm_cuda is not None}")
    
    print(f"\nForward: output diff = {(output_cuda - output_python).abs().max().item():.2e}")
    
    if grad_x_cuda is not None and grad_x_python is not None:
        print(f"\nd_x diff: {(grad_x_cuda - grad_x_python).abs().max().item():.2e}")
        print(f"d_x CUDA first 8: {grad_x_cuda[0,0,:8].tolist()}")
        print(f"d_x Python first 8: {grad_x_python[0,0,:8].tolist()}")
    
    if grad_W_kvqm_cuda is not None and grad_W_kvqm_python is not None:
        print(f"\nd_W_kvqm diff: {(grad_W_kvqm_cuda - grad_W_kvqm_python).abs().max().item():.2e}")
        print(f"d_W_kvqm CUDA[0,:8]: {grad_W_kvqm_cuda[0,:8].tolist()}")
        print(f"d_W_kvqm Python[0,:8]: {grad_W_kvqm_python[0,:8].tolist()}")
    
    if grad_W_bs_cuda is not None and grad_W_bs_python is not None:
        print(f"\nd_W_bs diff: {(grad_W_bs_cuda - grad_W_bs_python).abs().max().item():.2e}")
    
    if grad_W_bm_cuda is not None and grad_W_bm_python is not None:
        print(f"d_W_bm diff: {(grad_W_bm_cuda - grad_W_bm_python).abs().max().item():.2e}")

if __name__ == "__main__":
    test_simple()
