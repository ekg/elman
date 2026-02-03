"""Test E79 input-bias with T=2."""
import sys
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, '/home/erikg/elman/elman/cuda')

import torch
import hasty_pytorch_lib
from elman.models.e79_coupled_matrix import E79CoupledMatrixCell

def test_t2():
    print("E79 Input-Bias Test T=2, B=1")
    print("="*60)
    
    torch.manual_seed(42)
    T, B, dim, n_state = 2, 1, 32, 16
    
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
    
    # Python forward + backward
    x_python = x.clone().requires_grad_(True)
    cell.use_cuda_input_bias = False
    cell.train()
    output_python, S_python, M_python = cell(x_python, S0.clone(), M0.clone())
    loss_python = output_python.sum()
    loss_python.backward()
    
    grad_x_python = x_python.grad.clone()
    grad_W_kvqm_python = cell.W_kvqm.grad.clone()
    
    cell.zero_grad()
    
    # CUDA forward + backward
    x_cuda = x.clone().requires_grad_(True)
    cell.use_cuda_input_bias = True
    output_cuda, S_cuda, M_cuda = cell(x_cuda, S0.clone(), M0.clone())
    loss_cuda = output_cuda.sum()
    loss_cuda.backward()
    
    grad_x_cuda = x_cuda.grad.clone()
    grad_W_kvqm_cuda = cell.W_kvqm.grad.clone()
    
    print(f"Forward: output diff = {(output_cuda - output_python).abs().max().item():.2e}")
    print(f"\nd_x diff: {(grad_x_cuda - grad_x_python).abs().max().item():.2e}")
    print(f"d_W_kvqm diff: {(grad_W_kvqm_cuda - grad_W_kvqm_python).abs().max().item():.2e}")
    
    # Check per-timestep gradients
    print(f"\nd_x[t=0] diff: {(grad_x_cuda[0] - grad_x_python[0]).abs().max().item():.2e}")
    print(f"d_x[t=1] diff: {(grad_x_cuda[1] - grad_x_python[1]).abs().max().item():.2e}")
    
    print(f"\nd_x CUDA[0,0,:8]: {grad_x_cuda[0,0,:8].tolist()}")
    print(f"d_x Python[0,0,:8]: {grad_x_python[0,0,:8].tolist()}")
    print(f"\nd_x CUDA[1,0,:8]: {grad_x_cuda[1,0,:8].tolist()}")
    print(f"d_x Python[1,0,:8]: {grad_x_python[1,0,:8].tolist()}")

if __name__ == "__main__":
    test_t2()
