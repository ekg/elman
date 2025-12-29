# Elman CUDA Kernels

Custom CUDA kernels for the Elman Ladder, forked from Haste.

## Directory Structure

```
cuda/
  lib/                  # CUDA kernel source files
    stock_elman_gpu.cu.cc       # Level 0: Stock Elman
    gated_elman_gpu.cu.cc       # Level 1: Gated Elman
    selective_elman_gpu.cu.cc   # Level 2: Selective Elman
    diagonal_selective_gpu.cu.cc # Level 3: Diagonal Selective
    log_storage_diagonal_gpu.cu.cc # Level 4: Log-Storage
    log_compute_full_gpu.cu.cc  # Level 5: Log-Compute
    logspace_triple_r_gpu.cu.cc # Level 6: Log-Space Triple R
    *.h                         # Header files
  pytorch/              # PyTorch bindings
    elman_ladder.cc     # C++ bindings for all levels
    elman_variants.py   # Python wrappers
    base_rnn.py         # Base class
  Makefile              # Build configuration
  setup.py              # Python package build
```

## Building

From the `cuda/` directory:

```bash
# Build the library
make

# Install as Python package
pip install -e .
```

## Requirements

- CUDA 11.0+
- PyTorch 2.0+
- C++17 compatible compiler

## Usage

After building:

```python
from elman.cuda.pytorch import elman_variants

# Use CUDA-accelerated Elman layers
layer = elman_variants.DiagonalSelectiveElman(dim=512)
```

## Notes

- These kernels provide 10-50x speedup over pure PyTorch implementations
- Log-space kernels (Levels 4-6) are experimental
- Original Haste repository: https://github.com/lmnt-com/haste
