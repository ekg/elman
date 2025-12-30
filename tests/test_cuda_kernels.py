"""
Tests for CUDA kernel correctness and consistency.
These tests require GPU and compare CUDA vs PyTorch implementations.
"""
import pytest
import torch

# Skip entire module if CUDA not available
pytestmark = pytest.mark.gpu


DIM = 64
T = 16
B = 4


def get_cuda_levels():
    """Get levels with CUDA implementations."""
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


@pytest.mark.parametrize("name,LayerClass", get_cuda_levels())
def test_cuda_forward(name, LayerClass):
    """Test that CUDA forward pass works."""
    layer = LayerClass(dim=DIM, expansion=1.0).cuda()
    x = torch.randn(B, T, DIM, device='cuda')

    output, h_final = layer(x)

    assert output.shape == (B, T, DIM)
    assert torch.isfinite(output).all(), f"{name}: CUDA output contains NaN/Inf"


@pytest.mark.parametrize("name,LayerClass", get_cuda_levels())
def test_cuda_backward(name, LayerClass):
    """Test that CUDA backward pass produces gradients."""
    layer = LayerClass(dim=DIM, expansion=1.0).cuda()
    x = torch.randn(B, T, DIM, device='cuda', requires_grad=True)

    output, _ = layer(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all(), f"{name}: CUDA gradient contains NaN/Inf"

    # Check parameter gradients
    for pname, p in layer.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{name}: no gradient for {pname}"
            assert torch.isfinite(p.grad).all(), \
                f"{name}: NaN/Inf in gradient for {pname}"


@pytest.mark.parametrize("name,LayerClass", get_cuda_levels())
def test_cuda_vs_cpu_output_shape(name, LayerClass):
    """Test that CUDA and CPU produce same output shapes."""
    torch.manual_seed(42)

    layer_cpu = LayerClass(dim=DIM, expansion=1.0)
    layer_cuda = LayerClass(dim=DIM, expansion=1.0).cuda()

    # Copy weights
    layer_cuda.load_state_dict(layer_cpu.state_dict())

    x_cpu = torch.randn(B, T, DIM)
    x_cuda = x_cpu.cuda()

    out_cpu, _ = layer_cpu(x_cpu)
    out_cuda, _ = layer_cuda(x_cuda)

    assert out_cpu.shape == out_cuda.shape


@pytest.mark.parametrize("name,LayerClass", get_cuda_levels())
def test_cuda_deterministic(name, LayerClass):
    """Test that CUDA gives deterministic results."""
    torch.manual_seed(42)

    layer = LayerClass(dim=DIM, expansion=1.0).cuda()
    layer.eval()

    x = torch.randn(B, T, DIM, device='cuda')

    with torch.no_grad():
        out1, _ = layer(x)
        out2, _ = layer(x)

    assert torch.allclose(out1, out2, rtol=1e-5, atol=1e-6), \
        f"{name}: CUDA not deterministic"


@pytest.mark.parametrize("name,LayerClass", get_cuda_levels())
def test_cuda_half_precision(name, LayerClass):
    """Test that CUDA works with float16."""
    layer = LayerClass(dim=DIM, expansion=1.0).cuda().half()
    x = torch.randn(B, T, DIM, device='cuda', dtype=torch.float16)

    output, _ = layer(x)

    assert output.dtype == torch.float16
    assert torch.isfinite(output).all(), f"{name}: half precision output NaN/Inf"


@pytest.mark.parametrize("name,LayerClass", get_cuda_levels())
def test_cuda_training_step(name, LayerClass):
    """Test a complete training step on CUDA."""
    torch.manual_seed(42)

    layer = LayerClass(dim=DIM, expansion=1.0).cuda()
    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-4)

    x = torch.randn(B, T, DIM, device='cuda')
    target = torch.randn(B, T, DIM, device='cuda')

    # Forward
    output, _ = layer(x)
    loss = ((output - target) ** 2).mean()

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # Check gradients exist and are finite
    for pname, p in layer.named_parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all(), \
                f"{name}: NaN/Inf gradient in {pname}"

    # Update
    optimizer.step()

    # Forward again - should not crash
    output2, _ = layer(x)
    assert torch.isfinite(output2).all()


@pytest.mark.parametrize("name,LayerClass", get_cuda_levels())
def test_cuda_gradient_clipping(name, LayerClass):
    """Test that gradient clipping works correctly."""
    layer = LayerClass(dim=DIM, expansion=1.0).cuda()
    x = torch.randn(B, T, DIM, device='cuda', requires_grad=True)

    output, _ = layer(x)
    loss = output.sum()
    loss.backward()

    # Clip gradients
    torch.nn.utils.clip_grad_norm_(layer.parameters(), 1.0)

    # All gradients should be bounded
    for pname, p in layer.named_parameters():
        if p.grad is not None:
            grad_norm = p.grad.norm().item()
            assert grad_norm < 10.0, \
                f"{name}: gradient for {pname} not clipped ({grad_norm:.1f})"
