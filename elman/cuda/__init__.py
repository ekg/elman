"""
Elman CUDA kernels.

Custom CUDA implementations for the Elman Ladder levels.
Forked from Haste (https://github.com/lmnt-com/haste).

Levels:
    0: Stock Elman - stock_elman_gpu.cu.cc
    1: Gated Elman - gated_elman_gpu.cu.cc
    2: Selective Elman - selective_elman_gpu.cu.cc
    3: Diagonal Selective - diagonal_selective_gpu.cu.cc
    4: Log-Storage - log_storage_diagonal_gpu.cu.cc
    5: Log-Compute - log_compute_full_gpu.cu.cc
    6: Log-Space Triple R - logspace_triple_r_gpu.cu.cc

Build with: make && pip install -e .
"""

try:
    from .pytorch.elman_variants import (
        StockElmanCUDA,
        GatedElmanCUDA,
        SelectiveElmanCUDA,
        DiagonalSelectiveCUDA,
    )
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    # Fallback to pure PyTorch implementations
    StockElmanCUDA = None
    GatedElmanCUDA = None
    SelectiveElmanCUDA = None
    DiagonalSelectiveCUDA = None


def get_cuda_layer(level: int):
    """Get CUDA-accelerated layer for a given level, or None if unavailable."""
    if not CUDA_AVAILABLE:
        return None

    layers = {
        0: StockElmanCUDA,
        1: GatedElmanCUDA,
        2: SelectiveElmanCUDA,
        3: DiagonalSelectiveCUDA,
    }
    return layers.get(level)
