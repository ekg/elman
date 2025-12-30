"""
Test API consistency across all Elman levels.
These tests verify that all levels have consistent input/output shapes.
"""
import pytest
import torch
import torch.nn as nn


# Test parameters
DIM = 32
T = 8
B = 2


def get_all_levels():
    """Get all Elman levels for testing."""
    from elman.models.stock_elman import StockElman
    from elman.models.gated_elman import GatedElman
    from elman.models.selective_elman import SelectiveElman
    from elman.models.diagonal_selective import DiagonalSelective
    from elman.models.logspace_polynomial import LogSpacePolynomial
    from elman.models.logspace_selective import LogSpaceSelective
    from elman.models.logspace_diagonal_selective import LogSpaceDiagonalSelective

    return [
        ('level_0', StockElman),
        ('level_1', GatedElman),
        ('level_2', SelectiveElman),
        ('level_3', DiagonalSelective),
        ('log_0', LogSpacePolynomial),
        ('log_1', LogSpaceSelective),
        ('log_2', LogSpaceDiagonalSelective),
    ]


@pytest.mark.parametrize("name,LayerClass", get_all_levels())
def test_layer_instantiation(name, LayerClass):
    """Test that all layers can be instantiated."""
    layer = LayerClass(dim=DIM, expansion=1.0)
    assert isinstance(layer, nn.Module)


@pytest.mark.parametrize("name,LayerClass", get_all_levels())
def test_output_shape(name, LayerClass):
    """Test that all layers produce correct output shapes."""
    layer = LayerClass(dim=DIM, expansion=1.0)
    x = torch.randn(B, T, DIM)

    output, h_final = layer(x)

    assert output.shape == (B, T, DIM), f"{name}: output shape mismatch"
    # h_final can be a tensor or tuple depending on log-space vs linear


@pytest.mark.parametrize("name,LayerClass", get_all_levels())
def test_output_dtype(name, LayerClass):
    """Test that output dtype matches input dtype."""
    layer = LayerClass(dim=DIM, expansion=1.0)

    for dtype in [torch.float32, torch.float64]:
        layer = layer.to(dtype)
        x = torch.randn(B, T, DIM, dtype=dtype)
        output, _ = layer(x)
        assert output.dtype == dtype, f"{name}: dtype mismatch for {dtype}"


@pytest.mark.parametrize("name,LayerClass", get_all_levels())
def test_gradient_flow(name, LayerClass):
    """Test that gradients flow through all layers."""
    layer = LayerClass(dim=DIM, expansion=1.0)
    x = torch.randn(B, T, DIM, requires_grad=True)

    output, _ = layer(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None, f"{name}: no gradient to input"
    assert x.grad.shape == x.shape, f"{name}: gradient shape mismatch"

    # Check that some parameters have gradients
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in layer.parameters() if p.requires_grad)
    assert has_grad, f"{name}: no parameter gradients"


@pytest.mark.parametrize("name,LayerClass", get_all_levels())
def test_output_is_finite(name, LayerClass):
    """Test that outputs don't contain NaN or Inf."""
    layer = LayerClass(dim=DIM, expansion=1.0)
    x = torch.randn(B, T, DIM)

    output, _ = layer(x)

    assert torch.isfinite(output).all(), f"{name}: output contains NaN/Inf"


@pytest.mark.parametrize("name,LayerClass", get_all_levels())
def test_train_eval_modes(name, LayerClass):
    """Test that train/eval modes work correctly."""
    layer = LayerClass(dim=DIM, expansion=1.0)
    x = torch.randn(B, T, DIM)

    # Train mode
    layer.train()
    output_train, _ = layer(x)
    assert output_train.shape == (B, T, DIM)

    # Eval mode
    layer.eval()
    with torch.no_grad():
        output_eval, _ = layer(x)
    assert output_eval.shape == (B, T, DIM)


@pytest.mark.parametrize("name,LayerClass", get_all_levels())
def test_expansion_factor(name, LayerClass):
    """Test that expansion factor works correctly."""
    for expansion in [0.5, 1.0, 2.0]:
        layer = LayerClass(dim=DIM, expansion=expansion)
        x = torch.randn(B, T, DIM)
        output, _ = layer(x)
        assert output.shape == (B, T, DIM), f"{name}: expansion={expansion} failed"


@pytest.mark.parametrize("name,LayerClass", get_all_levels())
def test_variable_sequence_length(name, LayerClass):
    """Test that layers work with different sequence lengths."""
    layer = LayerClass(dim=DIM, expansion=1.0)

    for seq_len in [1, 4, 16, 32]:
        x = torch.randn(B, seq_len, DIM)
        output, _ = layer(x)
        assert output.shape == (B, seq_len, DIM), f"{name}: seq_len={seq_len} failed"


@pytest.mark.parametrize("name,LayerClass", get_all_levels())
def test_batch_independence(name, LayerClass):
    """Test that batch elements are processed independently."""
    layer = LayerClass(dim=DIM, expansion=1.0)
    layer.eval()

    x1 = torch.randn(1, T, DIM)
    x2 = torch.randn(1, T, DIM)
    x_combined = torch.cat([x1, x2], dim=0)

    with torch.no_grad():
        out1, _ = layer(x1)
        out2, _ = layer(x2)
        out_combined, _ = layer(x_combined)

    # Outputs should match (within numerical precision)
    assert torch.allclose(out1, out_combined[0:1], rtol=1e-4, atol=1e-5), \
        f"{name}: batch element 0 not independent"
    assert torch.allclose(out2, out_combined[1:2], rtol=1e-4, atol=1e-5), \
        f"{name}: batch element 1 not independent"
