"""
PyTorch bindings for Elman CUDA kernels.
"""

try:
    from .elman_variants import (
        StockElmanCUDA,
        GatedElmanCUDA,
        SelectiveElmanCUDA,
        DiagonalSelectiveCUDA,
    )
except ImportError:
    # CUDA library not built yet
    pass
