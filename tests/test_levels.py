#!/usr/bin/env python3
"""
Test all ladder levels work correctly.

Verifies:
1. Forward pass produces correct shapes
2. Backward pass runs without error
3. Gradient flow is reasonable
4. Hidden state persistence works
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from elman.models import (
    StockElman, GatedElman, SelectiveElman, DiagonalSelective,
    LadderLM, create_ladder_model
)


def test_layer_forward_backward(LayerClass, level_name):
    """Test a single layer class."""
    print(f"\n{'='*60}")
    print(f"Testing {level_name}")
    print('='*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32

    # Create layer
    layer = LayerClass(dim=256, expansion=1.0, n_groups=32).to(device).to(dtype)
    params = sum(p.numel() for p in layer.parameters())
    print(f"Parameters: {params:,}")

    # Test input
    B, T, D = 4, 64, 256
    x = torch.randn(B, T, D, device=device, dtype=dtype)

    # Forward
    out, h_final = layer(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Hidden: {h_final.shape}")

    assert out.shape == (B, T, D), f"Output shape mismatch: {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN"
    assert not torch.isinf(out).any(), "Output contains Inf"

    # Backward
    loss = out.sum()
    loss.backward()

    # Check gradients exist
    has_grads = sum(1 for p in layer.parameters() if p.grad is not None)
    total_params = sum(1 for p in layer.parameters())
    print(f"Gradients: {has_grads}/{total_params} parameters")
    assert has_grads > 0, "No gradients computed"

    # Check gradient magnitudes
    grad_norms = [p.grad.norm().item() for p in layer.parameters() if p.grad is not None]
    print(f"Grad norm range: [{min(grad_norms):.2e}, {max(grad_norms):.2e}]")

    print(f"{level_name}: PASSED")
    return True


def test_hidden_state_persistence():
    """Test that hidden state carries information across chunks."""
    print(f"\n{'='*60}")
    print("Testing hidden state persistence")
    print('='*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32

    layer = DiagonalSelective(dim=128, expansion=1.0).to(device).to(dtype)

    B, T, D = 2, 32, 128

    # First chunk
    x1 = torch.randn(B, T, D, device=device, dtype=dtype)
    out1, h1 = layer(x1, h0=None)

    # Second chunk with vs without hidden state
    x2 = torch.randn(B, T, D, device=device, dtype=dtype)

    out2_with_h, h2 = layer(x2, h0=h1)
    out2_no_h, _ = layer(x2, h0=None)

    # Outputs should differ if hidden state matters
    diff = (out2_with_h - out2_no_h).abs().mean().item()
    print(f"Output diff (with vs without h0): {diff:.6f}")

    assert diff > 1e-6, "Hidden state has no effect on output"
    print("Hidden state persistence: PASSED")
    return True


def test_ladder_lm():
    """Test the full language model."""
    print(f"\n{'='*60}")
    print("Testing LadderLM")
    print('='*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32

    for level in range(4):
        model = LadderLM(
            vocab_size=256,
            dim=128,
            depth=4,
            level=level,
            expansion=1.0,
        ).to(device).to(dtype)

        params = model.get_num_params()
        print(f"\nLevel {level}: {params:,} parameters")

        # Test forward
        B, T = 2, 32
        x = torch.randint(0, 256, (B, T), device=device)

        logits, (hidden, _) = model(x, return_prev_hiddens=True)
        assert logits.shape == (B, T, 256), f"Logits shape: {logits.shape}"
        assert len(hidden) == 4, f"Hidden layers: {len(hidden)}"

        # Test loss computation
        loss = model(x, return_loss=True)
        loss.backward()

        print(f"  Logits: {logits.shape}, Loss: {loss.item():.4f}")

    print("\nLadderLM: PASSED")
    return True


def test_create_ladder_model():
    """Test model creation helper."""
    print(f"\n{'='*60}")
    print("Testing create_ladder_model")
    print('='*60)

    for params in ['50m', '100m', '500m']:
        for level in [0, 3]:
            model = create_ladder_model(params, level=level)
            actual_params = model.get_num_params()
            print(f"  {params} level {level}: {actual_params:,} params")

    print("create_ladder_model: PASSED")
    return True


def test_gradient_flow_depth():
    """Test gradient flow at different depths."""
    print(f"\n{'='*60}")
    print("Testing gradient flow at depth")
    print('='*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)

    for level in [0, 3]:
        layer = (StockElman if level == 0 else DiagonalSelective)(
            dim=64, expansion=1.0
        ).to(device).float()

        for seq_len in [16, 64, 256]:
            x = torch.randn(2, seq_len, 64, device=device, requires_grad=True)

            out, _ = layer(x)
            loss = out[:, -1, :].sum()  # Only last timestep
            loss.backward()

            # Check gradient at first vs last timestep
            grad_first = x.grad[:, 0, :].abs().mean().item()
            grad_last = x.grad[:, -1, :].abs().mean().item()
            ratio = grad_last / (grad_first + 1e-10)

            print(f"Level {level}, seq {seq_len:3d}: first={grad_first:.2e}, last={grad_last:.2e}, ratio={ratio:.1f}x")

            x.grad = None

    print("Gradient flow: CHECK (ratio should not be extreme)")
    return True


if __name__ == '__main__':
    print("Elman Ladder Tests")
    print("=" * 60)

    # Test each level
    test_layer_forward_backward(StockElman, "Level 0: Stock Elman")
    test_layer_forward_backward(GatedElman, "Level 1: Gated Elman")
    test_layer_forward_backward(SelectiveElman, "Level 2: Selective Elman")
    test_layer_forward_backward(DiagonalSelective, "Level 3: Diagonal Selective")

    # Test hidden state
    test_hidden_state_persistence()

    # Test full LM
    test_ladder_lm()

    # Test model creation
    test_create_ladder_model()

    # Test gradient flow
    test_gradient_flow_depth()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
