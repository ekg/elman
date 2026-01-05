# Building Hasty CUDA Kernels for Elman Models

## Quick Reference

### Files to Create/Modify

1. **CUDA Kernel**: `elman/cuda/lib/<kernel_name>_gpu.cu.cc`
   - Template: Follow `lowrank_elman_gpu.cu.cc`
   - Use cuBLAS for GEMMs via `blas<T>::gemm()`
   - Elementwise ops use custom kernels

2. **Header**: `elman/cuda/lib/hasty/elman_ladder.h`
   - Add `template<typename T> struct YourKernelForward { ... };`
   - Add `template<typename T> struct YourKernelBackward { ... };`

3. **PyTorch Bindings**: `elman/cuda/pytorch/elman_ladder.cc`
   - Add forward/backward C++ functions in anonymous namespace
   - Register in `elman_ladder_init()` with `m.def()`

4. **Makefile**: `elman/cuda/Makefile`
   - Add `lib/<kernel_name>_gpu.o` to `CUDA_OBJS`

5. **Python Model**: `elman/models/<your_model>.py`
   - Create `Function` class with `@staticmethod forward/backward`
   - Check `HASTY_AVAILABLE and x.is_cuda` to use kernel

### Build Commands

```bash
cd elman/cuda
CUDA_HOME=/usr/local/cuda-12.8 make hasty
CUDA_HOME=/usr/local/cuda-12.8 python -m pip wheel . --no-deps -w dist/
python -m pip install --force-reinstall --break-system-packages dist/*.whl
```

### Testing

```bash
python -c "import torch; import hasty_pytorch_lib as h; print(dir(h))"
python elman/models/<your_model>.py
```

### Key Patterns

- Tensors: `[T, B, dim]` for RNN inputs (time-major)
- GEMM storage: cuBLAS is column-major, PyTorch is row-major
- Atomics: Use `float` accumulators, convert at end
- Workspace: Allocate via PyTorch, pass pointer to kernel
