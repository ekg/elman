"""
Test that E37v2 produces the same outputs as E37 (mathematically equivalent).
"""

import torch
import torch.nn as nn
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

from elman.models.e37_tied_weights import E37TiedWeights
from elman.models.e37_tied_weights_v2 import E37TiedWeightsV2


def test_correctness():
    torch.manual_seed(42)

    device = 'cuda'
    dtype = torch.float32  # Use float32 for accurate comparison

    dim = 512
    expansion = 2.0
    batch_size = 4
    seq_len = 32

    # Create models with identical weights
    # Disable spectral normalization to get exact comparison
    model_e37 = E37TiedWeights(dim=dim, expansion=expansion, use_conv=False, r_h_mode='none').to(device).to(dtype)
    model_e37v2 = E37TiedWeightsV2(dim=dim, expansion=expansion, use_conv=False, r_h_mode='none').to(device).to(dtype)

    # Copy weights from E37 to E37v2
    with torch.no_grad():
        model_e37v2.in_proj.weight.copy_(model_e37.in_proj.weight)
        model_e37v2.out_proj.weight.copy_(model_e37.out_proj.weight)
        model_e37v2.cell.W.copy_(model_e37.cell.W)
        model_e37v2.cell.b.copy_(model_e37.cell.b)

    # Create input
    x = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype)

    # Forward pass
    model_e37.eval()
    model_e37v2.eval()

    with torch.no_grad():
        out_e37, h_e37 = model_e37(x)
        out_e37v2, h_e37v2 = model_e37v2(x)

    # Compare outputs
    out_diff = (out_e37 - out_e37v2).abs()
    h_diff = (h_e37 - h_e37v2).abs()

    print("E37v2 Correctness Test")
    print("=" * 60)
    print(f"Output max diff: {out_diff.max().item():.2e}")
    print(f"Output mean diff: {out_diff.mean().item():.2e}")
    print(f"Hidden max diff: {h_diff.max().item():.2e}")
    print(f"Hidden mean diff: {h_diff.mean().item():.2e}")

    # Test gradient correctness
    print("\nTesting gradient correctness...")
    model_e37.train()
    model_e37v2.train()

    x_e37 = x.clone().requires_grad_(True)
    x_e37v2 = x.clone().requires_grad_(True)

    out_e37, _ = model_e37(x_e37)
    out_e37v2, _ = model_e37v2(x_e37v2)

    loss_e37 = out_e37.sum()
    loss_e37v2 = out_e37v2.sum()

    loss_e37.backward()
    loss_e37v2.backward()

    # Compare input gradients
    grad_diff = (x_e37.grad - x_e37v2.grad).abs()
    print(f"Input grad max diff: {grad_diff.max().item():.2e}")
    print(f"Input grad mean diff: {grad_diff.mean().item():.2e}")

    # Compare weight gradients
    w_grad_diff = (model_e37.cell.W.grad - model_e37v2.cell.W.grad).abs()
    b_grad_diff = (model_e37.cell.b.grad - model_e37v2.cell.b.grad).abs()
    print(f"W grad max diff: {w_grad_diff.max().item():.2e}")
    print(f"b grad max diff: {b_grad_diff.max().item():.2e}")

    # Check if differences are within tolerance
    tol = 1e-4
    if out_diff.max() < tol and h_diff.max() < tol and grad_diff.max() < tol:
        print(f"\nPASS: All differences below tolerance ({tol})")
        return True
    else:
        print(f"\nFAIL: Some differences exceed tolerance ({tol})")
        return False


if __name__ == "__main__":
    test_correctness()
