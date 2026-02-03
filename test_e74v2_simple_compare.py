#!/usr/bin/env python3
"""
Simple E74v2 comparison test - focus on delta update with output gate.
Uses minimal configuration to isolate numerical differences.
"""

import torch
import torch.nn.functional as F

from elman.models.e74_v2 import E74v2Cell, E74V2_CUDA_AVAILABLE

def main():
    print("E74v2 Simple CUDA vs Python Comparison")
    print("=" * 60)
    print(f"E74V2_CUDA_AVAILABLE: {E74V2_CUDA_AVAILABLE}")

    # Configuration
    B = 2
    T = 16
    dim = 256
    n_state = 32
    seed = 42

    device = torch.device('cuda')
    dtype = torch.bfloat16

    # Create cells
    torch.manual_seed(seed)
    cuda_cell = E74v2Cell(
        dim=dim, n_state=n_state,
        proj_type='no_z', update_type='delta', gate_type='output',
        use_tanh=True, use_cuda=True
    ).to(device).to(dtype)

    torch.manual_seed(seed)
    py_cell = E74v2Cell(
        dim=dim, n_state=n_state,
        proj_type='no_z', update_type='delta', gate_type='output',
        use_tanh=True, use_cuda=False
    ).to(device).to(dtype)

    # Copy weights
    with torch.no_grad():
        py_cell.W_k.copy_(cuda_cell.W_k)
        py_cell.W_v.copy_(cuda_cell.W_v)
        py_cell.W_q.copy_(cuda_cell.W_q)

    print(f"\nCUDA cell uses CUDA: {cuda_cell.use_cuda}")
    print(f"Python cell uses CUDA: {py_cell.use_cuda}")

    # Input
    torch.manual_seed(seed + 1)
    x = torch.randn(T, B, dim, device=device, dtype=dtype)
    x_cuda = x.clone().detach().requires_grad_(True)
    x_py = x.clone().detach().requires_grad_(True)

    S0 = torch.zeros(B, n_state, n_state, device=device, dtype=dtype)

    # Forward
    cuda_cell.train()
    py_cell.train()

    cuda_out, cuda_S = cuda_cell(x_cuda, S0.clone())
    py_out, py_S = py_cell(x_py, S0.clone())

    # Compare
    out_diff = (cuda_out.float() - py_out.float()).abs()
    S_diff = (cuda_S.float() - py_S.float()).abs()

    print(f"\n--- Forward Results ---")
    print(f"Output max diff: {out_diff.max().item():.6f}")
    print(f"Output mean diff: {out_diff.mean().item():.6f}")
    print(f"State max diff: {S_diff.max().item():.6f}")
    print(f"State mean diff: {S_diff.mean().item():.6f}")

    # Per-timestep output diff
    print(f"\nPer-timestep output max diff:")
    for t in range(min(T, 8)):
        t_diff = out_diff[t].max().item()
        print(f"  t={t}: {t_diff:.6f}")

    # Backward
    torch.manual_seed(seed + 2)
    grad_out = torch.randn_like(cuda_out)

    cuda_out.backward(grad_out.clone())
    py_out.backward(grad_out.clone())

    dx_diff = (x_cuda.grad.float() - x_py.grad.float()).abs()
    dW_k_diff = (cuda_cell.W_k.grad.float() - py_cell.W_k.grad.float()).abs()
    dW_v_diff = (cuda_cell.W_v.grad.float() - py_cell.W_v.grad.float()).abs()
    dW_q_diff = (cuda_cell.W_q.grad.float() - py_cell.W_q.grad.float()).abs()

    print(f"\n--- Backward Results ---")
    print(f"dx max diff: {dx_diff.max().item():.6f}")
    print(f"dx mean diff: {dx_diff.mean().item():.6f}")
    print(f"dW_k max diff: {dW_k_diff.max().item():.6f}")
    print(f"dW_v max diff: {dW_v_diff.max().item():.6f}")
    print(f"dW_q max diff: {dW_q_diff.max().item():.6f}")

    # Per-timestep dx diff
    print(f"\nPer-timestep dx max diff:")
    for t in range(min(T, 8)):
        t_diff = dx_diff[t].max().item()
        print(f"  t={t}: {t_diff:.6f}")

    # Check relative error
    print(f"\n--- Relative Errors ---")
    out_rel = (out_diff / (py_out.float().abs() + 1e-6)).max().item()
    S_rel = (S_diff / (py_S.float().abs() + 1e-6)).max().item()
    dx_rel = (dx_diff / (x_py.grad.float().abs() + 1e-6)).max().item()
    print(f"Output max relative error: {out_rel:.2%}")
    print(f"State max relative error: {S_rel:.2%}")
    print(f"dx max relative error: {dx_rel:.2%}")

    # Report
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Forward output max absolute diff:  {out_diff.max().item():.6f}")
    print(f"Forward state max absolute diff:   {S_diff.max().item():.6f}")
    print(f"Backward dx max absolute diff:     {dx_diff.max().item():.6f}")
    print(f"Backward dW_k max absolute diff:   {dW_k_diff.max().item():.6f}")
    print(f"Backward dW_v max absolute diff:   {dW_v_diff.max().item():.6f}")
    print(f"Backward dW_q max absolute diff:   {dW_q_diff.max().item():.6f}")

if __name__ == '__main__':
    main()
