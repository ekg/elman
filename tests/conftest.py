"""
Pytest configuration and fixtures for Elman tests.
"""
import pytest
import torch


def pytest_configure(config):
    """Add custom markers."""
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU (deselected by default on CPU)"
    )
    config.addinivalue_line(
        "markers", "cuda: mark test as requiring CUDA kernels"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Skip GPU tests when CUDA is not available."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords or "cuda" in item.keywords:
                item.add_marker(skip_gpu)


@pytest.fixture
def device():
    """Return appropriate device for testing."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def small_dim():
    """Small dimension for fast tests."""
    return 16


@pytest.fixture
def medium_dim():
    """Medium dimension for more thorough tests."""
    return 64


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    return 42
