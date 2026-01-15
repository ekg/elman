# E74 CUDA Kernel Design Document

Design specification for efficient CUDA kernels supporting E74 ablation variants.

## 1. Overview

Based on ablation results, implement optimized CUDA kernels for winning configurations.
Start by copying and modifying `e73_checkpointed_gpu.cu.cc`.

## 2. Key Design Decisions

### 2.1 State Type Parameterization

Add `state_type` parameter to distinguish state structures:

```cpp
enum StateType {
    STATE_FULL = 0,      // [B, n, n] matrix
    STATE_DIAGONAL = 1,  // [B, n] vector
    STATE_LOWRANK = 2,   // [B, n, r] + [B, n, r]
    STATE_BLOCKDIAG = 3, // [B, n/b, b, b]
};
```

### 2.2 Projection Type Parameterization

```cpp
enum ProjType {
    PROJ_FULL = 0,       // k, v, q, z separate
    PROJ_NO_Z = 1,       // k, v, q separate
    PROJ_TIED_KQ = 2,    // k=q, v separate
    PROJ_TIED_KVQ = 3,   // k=v=q single
};
```

### 2.3 Nonlinearity Type

```cpp
enum NonlinType {
    NONLIN_TANH = 0,
    NONLIN_LINEAR = 1,
    NONLIN_RMSNORM = 2,
    NONLIN_FROBNORM = 3,
};
```

## 3. File Structure

```
elman/cuda/lib/
├── e74_checkpointed_gpu.cu.cc    # Main checkpointed implementation
├── e74_diagonal_kernels.cuh      # Diagonal-specific kernels
├── e74_full_kernels.cuh          # Full matrix kernels (from E73)
├── e74_lowrank_kernels.cuh       # Low-rank kernels
├── e74_common.cuh                # Shared utilities
```

## 4. Recommended Approach: Copy and Modify

### 4.1 Base: e73_checkpointed_gpu.cu.cc → e74_checkpointed_gpu.cu.cc

Copy the entire file, then:

1. Add enum parameters to forward/backward signatures
2. Add kernel dispatch based on state_type
3. Keep checkpoint logic generic (just varies in size)

### 4.2 Key Modifications

#### Forward Signature Change
```cpp
// Before (E73)
void E73CheckpointedForward::Run(
    int steps,
    const __nv_bfloat16* W_k,
    const __nv_bfloat16* W_v,
    const __nv_bfloat16* W_q,
    const __nv_bfloat16* W_z,  // Remove if PROJ_NO_Z
    const __nv_bfloat16* b_z,  // Remove if PROJ_NO_Z
    ...
);

// After (E74)
void E74CheckpointedForward::Run(
    int steps,
    StateType state_type,
    ProjType proj_type,
    NonlinType nonlin_type,
    int rank,                   // For lowrank state
    int block_size,            // For blockdiag state
    const __nv_bfloat16* W_k,
    const __nv_bfloat16* W_v,  // NULL if PROJ_TIED_KVQ
    const __nv_bfloat16* W_q,  // NULL if PROJ_TIED_KQ or PROJ_TIED_KVQ
    const __nv_bfloat16* W_z,  // NULL if PROJ_NO_Z, PROJ_TIED_*
    ...
);
```

#### Kernel Dispatch
```cpp
// In forward loop
switch (state_type) {
    case STATE_DIAGONAL:
        E74DiagonalDeltaInPlaceKernel_BF16<<<...>>>(
            S_diag, v_t, k_norm_t, B, n, use_tanh
        );
        E74DiagonalOutputKernel_BF16<<<...>>>(
            S_diag, q_t, out_t, Sq_cache_t, B, n
        );
        break;

    case STATE_FULL:
        E74FullRetrievalKernel_BF16<<<...>>>(
            S_full, z_t, k_norm_t, retrieved, B, n, variant
        );
        E74FullDeltaInPlaceKernel_BF16<<<...>>>(
            S_full, v_t, retrieved, k_norm_t, B, n, use_tanh
        );
        E74FullOutputKernel_BF16<<<...>>>(
            S_full, q_t, out_t, Sq_cache_t, B, n
        );
        break;

    case STATE_LOWRANK:
        // Separate U, V handling
        E74LowrankRetrievalKernel_BF16<<<...>>>(...);
        E74LowrankUpdateUKernel_BF16<<<...>>>(...);
        break;

    case STATE_BLOCKDIAG:
        // Block-wise operations
        E74BlockDiagDeltaKernel_BF16<<<...>>>(...);
        break;
}
```

#### Checkpoint Size Calculation
```cpp
int64_t get_checkpoint_size(StateType state_type, int B, int n, int r, int b) {
    switch (state_type) {
        case STATE_DIAGONAL:
            return B * n;
        case STATE_FULL:
            return B * n * n;
        case STATE_LOWRANK:
            return 2 * B * n * r;  // U + V
        case STATE_BLOCKDIAG:
            return B * (n / b) * b * b;
    }
}
```

## 5. Diagonal State Kernels (NEW)

Most efficient - O(n) per step instead of O(n²).

### 5.1 Delta Update (In-Place)
```cpp
__global__ void E74DiagonalDeltaInPlaceKernel_BF16(
    const int batch_size,
    const int n,
    __nv_bfloat16* __restrict__ S,      // [B, n] - updated IN-PLACE
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ k_norm,
    const bool use_tanh)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * n;

    if (idx < total) {
        float s = __bfloat162float(S[idx]);
        float v_i = __bfloat162float(v[idx]);
        float k_i = __bfloat162float(k_norm[idx]);

        // S[i] = f(S[i] * (1 - k²[i]) + v[i] * k[i])
        float k_sq = k_i * k_i;
        float s_new = s * (1.0f - k_sq) + v_i * k_i;

        if (use_tanh) {
            s_new = tanhf(s_new);
        }

        S[idx] = __float2bfloat16(s_new);
    }
}
```

### 5.2 Output (Diagonal)
```cpp
__global__ void E74DiagonalOutputKernel_BF16(
    const int batch_size,
    const int n,
    const __nv_bfloat16* __restrict__ S,
    const __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ Sq_cache)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * n;

    if (idx < total) {
        float s = __bfloat162float(S[idx]);
        float q_i = __bfloat162float(q[idx]);

        // Sq = S * q (element-wise for diagonal)
        float Sq = s * q_i;

        if (Sq_cache) {
            Sq_cache[idx] = __float2bfloat16(Sq);
        }

        // out = Sq * silu(Sq)
        float sigmoid_Sq = 1.0f / (1.0f + expf(-Sq));
        float silu_Sq = Sq * sigmoid_Sq;
        float out = Sq * silu_Sq;

        output[idx] = __float2bfloat16(out);
    }
}
```

### 5.3 Backward Kernels
```cpp
// Self-gate backward: out = Sq * silu(Sq)
__global__ void E74DiagonalSelfGateBackwardKernel_BF16(
    const int total,
    const __nv_bfloat16* __restrict__ Sq,
    const __nv_bfloat16* __restrict__ d_output,
    __nv_bfloat16* __restrict__ d_Sq)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        float sq = __bfloat162float(Sq[idx]);
        float d_out = __bfloat162float(d_output[idx]);

        float sigmoid_sq = 1.0f / (1.0f + expf(-sq));
        float silu_sq = sq * sigmoid_sq;

        // d_out/d_Sq = silu + Sq * d_silu/d_Sq
        float d_silu = sigmoid_sq + sq * sigmoid_sq * (1.0f - sigmoid_sq);
        float d_sq = d_out * (silu_sq + sq * d_silu);

        d_Sq[idx] = __float2bfloat16(d_sq);
    }
}

// Delta backward
__global__ void E74DiagonalDeltaBackwardKernel_BF16(
    const int batch_size,
    const int n,
    const __nv_bfloat16* __restrict__ S_prev,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ k_norm,
    const __nv_bfloat16* __restrict__ d_S_new,
    __nv_bfloat16* __restrict__ d_S_prev,
    float* __restrict__ d_v_f,
    float* __restrict__ d_k_f,
    const bool use_tanh)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * n;

    if (idx < total) {
        float s_prev = __bfloat162float(S_prev[idx]);
        float v_i = __bfloat162float(v[idx]);
        float k = __bfloat162float(k_norm[idx]);
        float d_s_new = __bfloat162float(d_S_new[idx]);

        float k_sq = k * k;
        float pre_tanh = s_prev * (1.0f - k_sq) + v_i * k;

        float d_pre;
        if (use_tanh) {
            float tanh_val = tanhf(pre_tanh);
            d_pre = d_s_new * (1.0f - tanh_val * tanh_val);
        } else {
            d_pre = d_s_new;
        }

        // d_S_prev = d_pre * (1 - k²)
        d_S_prev[idx] = __float2bfloat16(d_pre * (1.0f - k_sq));

        // d_v = d_pre * k
        atomicAdd(&d_v_f[idx], d_pre * k);

        // d_k = d_pre * (v - 2*S_prev*k)
        atomicAdd(&d_k_f[idx], d_pre * (v_i - 2.0f * s_prev * k));
    }
}
```

## 6. Memory Comparison

### Diagonal State (Recommended)
```
State storage:       O(n) per checkpoint
Retrieval:          O(n) - element-wise
Delta update:       O(n) - element-wise
Output:             O(n) - element-wise

Total per step:     O(n)
Total memory:       O(T*n) for caches + O((T/K)*n) for checkpoints

For B=32, T=512, n=64, K=32:
  Checkpoints: 17 * 32 * 64 * 2B = 70 KB
  Caches:      512 * 32 * 64 * 4 * 2B = 8.4 MB (k_norm, v, q, Sq)
  Total:       ~8.5 MB
```

### Full Matrix State (E73 baseline)
```
State storage:       O(n²) per checkpoint
Retrieval:          O(n²) - matrix-vector
Delta update:       O(n²) - outer product
Output:             O(n²) - matrix-vector

Total per step:     O(n²)
Total memory:       O(T*n) for caches + O((T/K)*n²) for checkpoints

For B=32, T=512, n=64, K=32:
  Checkpoints: 17 * 32 * 64 * 64 * 2B = 4.5 MB
  Caches:      512 * 32 * 64 * 5 * 2B = 10.5 MB
  Total:       ~15 MB
```

### Comparison
| State Type | Checkpoint Size | Ops per Step | Params |
|------------|-----------------|--------------|--------|
| Full       | O(n²)           | O(n²)        | O(nd)  |
| Diagonal   | O(n)            | O(n)         | O(nd)  |
| Lowrank-r  | O(nr)           | O(nr)        | O(nd)  |

**Diagonal gives 64x memory reduction for checkpoints and 64x compute reduction per step.**

## 7. Projection Simplification

### PROJ_NO_Z (Remove z modulation)
```cpp
// Forward: Skip z projection and modulation
if (proj_type != PROJ_NO_Z) {
    blas<T>::gemm(..., W_z, x, z_logit);
    BiasKernel<<<...>>>(z_logit, b_z, z);
}

// In retrieval: no z modulation
if (proj_type == PROJ_NO_Z) {
    // retrieved = S @ k_norm (no z modulation)
    E74RetrievalNoZKernel<<<...>>>(S, k_norm, retrieved);
} else {
    E74RetrievalWithZKernel<<<...>>>(S, z, k_norm, retrieved, variant);
}
```

### PROJ_TIED_KQ (k = q)
```cpp
if (proj_type == PROJ_TIED_KQ || proj_type == PROJ_TIED_KVQ) {
    // Skip q projection, use k
    q_all = k_all;  // Pointer alias
}
```

### PROJ_TIED_KVQ (k = v = q)
```cpp
if (proj_type == PROJ_TIED_KVQ) {
    // Single projection
    blas<T>::gemm(..., W_k, x, k_all);
    v_all = k_all;
    q_all = k_all;
}
```

## 8. Implementation Priority

### Phase 1: Diagonal with NO_Z (Highest Value)
1. Copy e73_checkpointed_gpu.cu.cc → e74_checkpointed_gpu.cu.cc
2. Add diagonal kernels (simple, O(n) per step)
3. Remove z modulation codepath
4. Test against Triton baseline

### Phase 2: Tied Projections
1. Add PROJ_TIED_KQ support
2. Add PROJ_TIED_KVQ support
3. Measure parameter/compute reduction

### Phase 3: Other State Types (If Needed)
1. Low-rank (if ablations show promise)
2. Block-diagonal (if ablations show promise)

## 9. Testing Strategy

### Unit Tests
1. Diagonal forward matches PyTorch reference
2. Diagonal backward matches PyTorch autograd
3. Checkpointing reproduces non-checkpointed results

### Integration Tests
1. Full training loop with checkpointing
2. Memory usage verification
3. Gradient accuracy over long sequences

### Performance Tests
1. Throughput comparison: Triton vs CUDA
2. Memory comparison: checkpointed vs non-checkpointed
3. Scaling: varying B, T, n

## 10. Expected Performance

| Implementation | Throughput | Memory | Complexity |
|----------------|------------|--------|------------|
| E73 CUDA       | 100%       | 100%   | High       |
| E74 Diagonal CUDA | 150-200% | 50%    | Medium     |
| E74 Diagonal Triton | 120-150% | 50%  | Low        |

The diagonal state with no z modulation should be:
- 50-100% faster (O(n) vs O(n²) per step)
- 50% less memory (smaller checkpoints)
- Simpler to implement and maintain

## 11. Migration Checklist

- [ ] Copy e73_checkpointed_gpu.cu.cc → e74_checkpointed_gpu.cu.cc
- [ ] Add StateType, ProjType enums
- [ ] Implement E74DiagonalDeltaInPlaceKernel_BF16
- [ ] Implement E74DiagonalOutputKernel_BF16
- [ ] Implement E74DiagonalSelfGateBackwardKernel_BF16
- [ ] Implement E74DiagonalDeltaBackwardKernel_BF16
- [ ] Add projection type dispatching
- [ ] Update workspace size calculations
- [ ] Add Python bindings
- [ ] Write unit tests
- [ ] Benchmark against Triton
