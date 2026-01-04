# Elman Ladder Build Instructions

## System Info (erikg's workstation)

- **OS**: Ubuntu 24.04 (Linux 6.8.0-84-generic)
- **Python**: 3.12 (system python)
- **CUDA**: 12.8 (via __cuda virtual package)
- **PyTorch**: 2.9.1 (installed in `~/.local/lib/python3.12/site-packages/`)

## Environment Setup

The project uses system Python with user-level packages (no conda/venv).

### Required Environment Variable

PyTorch libraries must be in `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=/home/erikg/.local/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH
```

Add this to your shell rc file for persistence.

## Building the CUDA Extension

The CUDA kernels are in `elman/cuda/`. Three kernel files:
- `lib/stock_elman_gpu.cu.cc` - E0: Stock Elman with learned gate
- `lib/mamba_gated_elman_gpu.cu.cc` - E1: Mamba-style split gating
- `lib/slot_elman_gpu.cu.cc` - E2: Multi-slot memory (64 slots)

To rebuild:

```bash
cd /home/erikg/elman/elman/cuda

# Set CUDA paths
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8

# Build static library first (compiles all CUDA kernels)
make -j12 hasty

# Install Python extension (requires --break-system-packages for externally-managed Python)
pip3 install --user --break-system-packages -e .
```

### Verifying the Build

```bash
export LD_LIBRARY_PATH=/home/erikg/.local/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH

# Check all CUDA kernels are loaded
python3 -c "
import hasty_pytorch_lib
print('stock_elman_forward:', hasattr(hasty_pytorch_lib, 'stock_elman_forward'))
print('mamba_gated_elman_forward:', hasattr(hasty_pytorch_lib, 'mamba_gated_elman_forward'))
print('slot_elman_forward:', hasattr(hasty_pytorch_lib, 'slot_elman_forward'))
"

# Check Python models detect CUDA
python3 -c "
from elman.models.stock_elman import HASTE_AVAILABLE
from elman.models.mamba_gated_elman import MAMBA_CUDA_AVAILABLE
from elman.models.slot_elman import SLOT_CUDA_AVAILABLE
print(f'E0 CUDA: {HASTE_AVAILABLE}')
print(f'E1 CUDA: {MAMBA_CUDA_AVAILABLE}')
print(f'E2 CUDA: {SLOT_CUDA_AVAILABLE}')
"
```

All should print `True`. If not, the Python fallback (slower) will be used.

## Expected Performance

With CUDA kernels active (B=16, T=256, D=512, expansion=2.0):
- **E0 (Stock Elman)**: ~270k tok/s
- **E1 (Mamba-Gated)**: ~260k tok/s
- **E2 (Slot-Based)**: ~130k tok/s (64 slots = 64x more memory than e0/e1)

Without CUDA kernels (Python fallback):
- All models: ~6-10k tok/s

## Running Benchmarks

```bash
export LD_LIBRARY_PATH=/home/erikg/.local/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH

# Single GPU
python3 benchmark_baselines.py --models e0 --params 50m --steps 100 --data data/fineweb_100mb.txt

# Multi-GPU parallel (use separate CUDA_VISIBLE_DEVICES per process)
CUDA_VISIBLE_DEVICES=0 python3 -u benchmark_baselines.py --models e0 ... &
CUDA_VISIBLE_DEVICES=1 python3 -u benchmark_baselines.py --models e1 ... &
wait
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'hasty_pytorch_lib'`
- Rebuild with: `cd elman/cuda && pip3 install --user --break-system-packages -e .`

### `ImportError: libc10.so: cannot open shared object file`
- Set LD_LIBRARY_PATH as shown above

### `HASTE_AVAILABLE = False` (or MAMBA_CUDA_AVAILABLE, SLOT_CUDA_AVAILABLE)
- CUDA extension not built or not loadable
- Check `python3 -c "import hasty_pytorch_lib"` for errors
- Rebuild: `cd elman/cuda && make clean && make -j12 hasty && pip3 install --user --break-system-packages -e .`

### Slow throughput (~6k tok/s instead of ~270k tok/s)
- CUDA kernel not loaded, check the *_AVAILABLE flags above
- Rebuild extension if needed

### CUDA compilation errors
- Check CUDA version: `nvcc --version` (should be 12.8)
- Check CUDA_HOME is set: `echo $CUDA_HOME`
- Check arch flags in Makefile match your GPU (sm_80 for A100, sm_89 for 4090)
