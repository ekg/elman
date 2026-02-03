# E79 Input-Bias CUDA Kernel Handoff

## Summary

This document describes the work on implementing and fixing the E79 input-dependent bias CUDA kernel variant.

## Background

E79 (Coupled Memory-Modulation Matrix System) is a recurrent cell with two coupled n×n matrices:
- **S**: Content/association matrix
- **M**: Modulation/gating matrix

The gates use sigmoid activation with optional bias:
```
gate_s = sigmoid(M @ k_norm + bias_s)
gate_m = sigmoid(S @ q + M @ m_vec + bias_m)
```

Three bias variants exist:
1. **No bias** (`use_bias=False`): bias = 0
2. **Fixed bias** (`use_bias=True, input_bias=False`): bias = learned fixed parameters `b_s_gate`, `b_m_gate`
3. **Input-dependent bias** (`input_bias=True`): bias = `W_bs @ x`, `W_bm @ x` (per-timestep projections)

## Files Modified

### CUDA Kernel Implementation
- `/home/erikg/elman/elman/cuda/lib/e79_coupled_matrix_gpu.cu.cc`
  - Added `E79CoupledInputBiasForwardKernel_BF16` and `E79CoupledInputBiasForwardKernel_FP32`
  - Added `E79CoupledInputBiasBackwardKernel_BF16` and `E79CoupledInputBiasBackwardKernel_FP32`
  - Added `E79CoupledInputBiasForward<DataT>` and `E79CoupledInputBiasBackward<DataT>` wrapper classes

### Python Bindings
- `/home/erikg/elman/elman/cuda/pytorch/elman_ladder.cc`
  - Added `e79_coupled_input_bias_forward()` function (lines 12369-12477)
  - Added `e79_coupled_input_bias_backward()` function (lines 12479-12587)
  - Added m.def() registrations (lines 14078-14081)

### Python Model
- `/home/erikg/elman/elman/models/e79_coupled_matrix.py`
  - Added `E79_INPUT_BIAS_CUDA_AVAILABLE` check
  - Added `E79CUDAInputBiasFunction` autograd wrapper class
  - Updated `E79CoupledMatrixCell` to use CUDA kernel when available

## Critical Bug Fixed: Forward Pass GEMM

The original implementation used wrong transpose flags and leading dimensions for cuBLAS GEMMs.

### Before (BUGGY):
```cpp
// Forward: kvqm = x @ W_kvqm.T
cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
             4 * n, T * B, d,
             &alpha,
             W_kvqm, data_type, 4 * n,  // Wrong lda
             x, data_type, d,
             ...);
```

### After (FIXED):
```cpp
// Forward: kvqm = W_kvqm^T @ x (in column-major)
cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
             4 * n, T * B, d,
             &alpha,
             W_kvqm, data_type, d,  // Correct lda for transposed
             x, data_type, d,
             ...);
```

The same pattern was applied to W_bs and W_bm projections.

## Backward Pass GEMM Fix (Applied)

Similar fixes were applied to the backward pass:

```cpp
// d_x = W_kvqm @ d_kvqm_cache (not W_kvqm^T)
cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
             d, T * B, 4 * n,
             &alpha,
             W_kvqm, data_type, d,  // lda = d
             d_kvqm_cache, data_type, 4 * n,
             ...);

// d_W_kvqm = x @ d_kvqm_cache^T
cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
             d, 4 * n, T * B,
             &alpha,
             x, data_type, d,
             d_kvqm_cache, data_type, 4 * n,
             &beta_zero,
             d_W_kvqm, data_type, d,  // ldc = d
             ...);
```

## Current Status

### WORKING
- Forward pass: Output/S/M diffs are ~1e-5 to 1e-7 (acceptable numerical precision)

### NOT WORKING
- Backward pass gradients still differ significantly:
  - d_x diff: 6.18e-01
  - d_W_kvqm diff: 2.04e+00
  - d_W_bs diff: 1.56e-02
  - d_W_bm diff: 1.06e-03

## Root Cause Analysis (In Progress)

### What's Working
- Forward pass: correct for all T values
- Backward pass with T=1, B=1: **CORRECT** (diff ~1e-6)

### What's Broken
- Backward pass with T=2: only t=1 gradients are wrong, t=0 gradients are correct!

### Debugging Results

```
T=1, B=1:
  d_x diff: 1.91e-06  (CORRECT)
  d_W_kvqm diff: 3.81e-06  (CORRECT)

T=2, B=1:
  d_x[t=0] diff: 1.91e-06  (CORRECT)
  d_x[t=1] diff: 2.15e-01  (WRONG)
  d_W_kvqm diff: 4.07e-01  (WRONG - accumulated from t=1 error)
```

### Analysis

The backward kernel processes timesteps in reverse order (t=T-1 down to t=0). For each t:
1. Loads checkpoint (S0, M0 for segment 0)
2. Re-runs forward pass from t_start to t to reconstruct state
3. Computes gradients for timestep t
4. Propagates dS/dM for next iteration

The fact that t=0 gradients are correct but t=1 gradients are wrong suggests the issue is specifically in:
1. The initial gradient computation at t=T-1 (first iteration)
2. NOT in the dS/dM propagation (since t=0 is correct after processing t=1)
3. NOT in the checkpoint/forward re-run logic (dS/dM would be wrong for t=0 otherwise)

### Differences from Original E79 Kernel

Comparing the FP32 kernels:
- Original: Uses fixed b_s_gate, b_m_gate parameters
- Input-bias: Uses per-timestep bs_all[t], bm_all[t] from projections

The gradient computation for biases at lines 3323-3330 writes to d_bs_all and d_bm_all per-timestep:
```cpp
d_bs_all[bias_idx + tid] = d_s_gate;
d_bm_all[bias_idx + tid] = d_m_gate;
```

The dS/dM propagation (lines 3397-3418) is identical to the original kernel.

### Suspected Issues

1. **The forward re-run uses decay cache**: At lines 3184-3187, the backward kernel loads s_row_decay etc. from cache, not recomputing from biases. This is correct because the cache was computed with the right biases, but verify this.

2. **Initial dS/dM at t=T-1**: The kernel starts with dS=0, dM=0. Check if the initial gradient contribution from d_output is correctly computed.

3. **The d_Sq computation**: At line 3257-3261, the output gradient derivative might differ from what Python computes.

## Recommended Next Steps

1. **Focus on t=1 gradient computation**:
   - The issue is specifically in gradients at t=T-1 (first backward iteration)
   - T=1, B=1 works, so compare what's different when T=2

2. **Compare d_Sq computation**:
   - Check lines 3256-3263 where d_Sq is computed from d_output
   - The softsign derivative `2*Sq*sig + Sq^2*sig*(1-sig)` might have numerical issues

3. **Add debug prints**:
   - Print d_kvqm values per timestep to see exactly where they diverge
   - Compare d_k_raw, d_v_raw, d_q_raw, d_m_raw between CUDA and Python

4. **Check the forward re-run state**:
   - After loading checkpoint and re-running forward to t=1, verify S and M match what Python would compute
   - This determines if the issue is in state reconstruction or gradient computation

## Test Scripts

- `/home/erikg/elman/test_e79_input_bias_cuda.py` - Forward pass validation
- `/home/erikg/elman/test_e79_input_bias_t1.py` - Gradient test with T=1, B=1 (PASSES)
- `/home/erikg/elman/test_e79_input_bias_t2.py` - Gradient test with T=2, B=1 (FAILS at t=1)

Run tests with:
```bash
LD_LIBRARY_PATH=/home/erikg/.local/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH /usr/bin/python3 test_e79_input_bias_t2.py
```

## Build Instructions

```bash
cd /home/erikg/elman/elman/cuda
PATH=/usr/local/cuda/bin:/usr/bin:$PATH make
PATH=/usr/local/cuda/bin:/usr/bin:$PATH python3 setup.py build_ext --inplace
```

## Key Insight

The original bug happened because the subagent wrote new CUDA code from scratch instead of copying and modifying the working E79 kernel. Future kernel implementations should always start from a working kernel and make minimal modifications.

The cuBLAS convention is:
- Column-major storage (PyTorch row-major becomes transposed)
- PyTorch `[out_features, in_features]` becomes `[in_features, out_features]` in cuBLAS
- Forward: use `CUBLAS_OP_T` with `lda = in_features`
- Backward d_x: use `CUBLAS_OP_N` with `lda = in_features`
- Backward d_W: use `CUBLAS_OP_N, CUBLAS_OP_T` with `ldc = in_features`
