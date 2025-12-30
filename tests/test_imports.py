"""
Test that all modules can be imported without errors.
These tests run on CPU without CUDA.
"""
import pytest


def test_import_models():
    """Test that model modules can be imported."""
    from elman import models
    assert hasattr(models, 'StockElman')
    assert hasattr(models, 'GatedElman')
    assert hasattr(models, 'SelectiveElman')
    assert hasattr(models, 'DiagonalSelective')
    assert hasattr(models, 'LogSpacePolynomial')
    assert hasattr(models, 'LogSpaceSelective')
    assert hasattr(models, 'LogSpaceDiagonalSelective')
    assert hasattr(models, 'LadderLM')


def test_import_stock_elman():
    """Test StockElman import."""
    from elman.models.stock_elman import StockElman, StockElmanCell
    assert StockElman is not None
    assert StockElmanCell is not None


def test_import_gated_elman():
    """Test GatedElman import."""
    from elman.models.gated_elman import GatedElman, GatedElmanCell
    assert GatedElman is not None
    assert GatedElmanCell is not None


def test_import_selective_elman():
    """Test SelectiveElman import."""
    from elman.models.selective_elman import SelectiveElman, SelectiveElmanCell
    assert SelectiveElman is not None
    assert SelectiveElmanCell is not None


def test_import_diagonal_selective():
    """Test DiagonalSelective import."""
    from elman.models.diagonal_selective import DiagonalSelective, DiagonalSelectiveCell
    assert DiagonalSelective is not None
    assert DiagonalSelectiveCell is not None


def test_import_logspace_polynomial():
    """Test LogSpacePolynomial import."""
    from elman.models.logspace_polynomial import LogSpacePolynomial, LogSpacePolynomialCell
    assert LogSpacePolynomial is not None
    assert LogSpacePolynomialCell is not None


def test_import_logspace_selective():
    """Test LogSpaceSelective import."""
    from elman.models.logspace_selective import LogSpaceSelective, LogSpaceSelectiveCell
    assert LogSpaceSelective is not None
    assert LogSpaceSelectiveCell is not None


def test_import_logspace_diagonal_selective():
    """Test LogSpaceDiagonalSelective import."""
    from elman.models.logspace_diagonal_selective import (
        LogSpaceDiagonalSelective, LogSpaceDiagonalSelectiveCell
    )
    assert LogSpaceDiagonalSelective is not None
    assert LogSpaceDiagonalSelectiveCell is not None


def test_import_ladder_lm():
    """Test LadderLM import."""
    from elman.models.ladder_lm import LadderLM
    assert LadderLM is not None


def test_import_dataset():
    """Test dataset import."""
    from elman.data.dataset import DocumentStreamDataset, TokenizedStreamDataset
    assert DocumentStreamDataset is not None
    assert TokenizedStreamDataset is not None
