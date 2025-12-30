# Hasty - Elman CUDA Kernels

Custom CUDA kernels for the Elman Ladder (forked from Haste, renamed to Hasty).

## Directory Structure

```
cuda/
  lib/                  # CUDA kernel source files
    # Linear-space levels
    stock_elman_gpu.cu.cc       # Level 0: Stock Elman
    gated_elman_gpu.cu.cc       # Level 1: Gated Elman
    selective_elman_gpu.cu.cc   # Level 2: Selective Elman
    diagonal_selective_gpu.cu.cc # Level 3: Diagonal Selective

    # Log-space polynomial levels (RECOMMENDED)
    logspace_polynomial_gpu.cu.cc        # Level log_0: Log-Space Polynomial
    logspace_selective_gpu.cu.cc         # Level log_1: + compete×silu
    logspace_diagonal_selective_gpu.cu.cc # Level log_2: + diagonal r_h

    # Experimental
    log_storage_diagonal_gpu.cu.cc # Level 4: Log-Storage
    log_compute_full_gpu.cu.cc  # Level 5: Log-Compute
    logspace_triple_r_gpu.cu.cc # Level 6: Log-Space Triple R

    *.h                         # Header files
  pytorch/              # PyTorch bindings
    elman_ladder.cc     # C++ bindings for all levels
  Makefile              # Build configuration
  setup.py              # Python package build
```

## Building

From the `elman/cuda/` directory:

```bash
# Build the CUDA library
export PATH=/usr/local/cuda/bin:$PATH
make -j12 hasty

# Install as Python package
pip install -e . --break-system-packages
```

## Runtime

Set library path before running:
```bash
export LD_LIBRARY_PATH=$HOME/.local/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH
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
