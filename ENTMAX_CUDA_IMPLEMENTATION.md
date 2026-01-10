# ENTMAX CUDA Implementation Strategy for E23 Dual-Memory RNN

## Document Overview

This document provides a comprehensive CUDA implementation strategy for replacing softmax attention with entmax (sparse attention) in the E23 dual-memory RNN architecture. The E23 kernel currently achieves ~24us/step with 2 GEMMs, and this design aims to maintain similar performance while adding sparsity benefits.

## Table of Contents

1. [Algorithm Selection](#1-algorithm-selection)
2. [Mathematical Foundations](#2-mathematical-foundations)
3. [Forward Pass Implementation](#3-forward-pass-implementation)
4. [Backward Pass Implementation](#4-backward-pass-implementation)
5. [Performance Considerations](#5-performance-considerations)
6. [Integration with E23 Kernel](#6-integration-with-e23-kernel)
7. [Pseudocode](#7-pseudocode)
8. [Testing and Validation](#8-testing-and-validation)

---

## 1. Algorithm Selection

### 1.1 Options Considered

| Algorithm | Sparsity | Computation | Backward Complexity | Recommendation |
|-----------|----------|-------------|---------------------|----------------|
| Softmax | None | O(N) | Simple | Current baseline |
| Sparsemax (alpha=2) | Aggressive | O(N log N) sort | Sparse Jacobian | Good for discrete |
| 1.5-entmax | Moderate | O(N log N) sort | Sparse Jacobian | **Recommended** |
| General alpha-entmax | Tunable | O(N log N) + bisection | Complex | Future extension |

### 1.2 Recommendation: Fixed 1.5-Entmax

**Primary choice: 1.5-entmax** for the following reasons:

1. **Sweet spot for sparsity**: Produces exact zeros for low-scoring slots without being as aggressive as sparsemax
2. **Closed-form solution**: Unlike general alpha-entmax, 1.5-entmax has an efficient closed-form threshold computation
3. **Simple backward pass**: The Jacobian has a clean analytical form with sparse structure
4. **Proven performance**: Widely validated in NLP attention mechanisms

**Secondary support: Sparsemax (alpha=2)** as a compile-time option for tasks requiring more aggressive sparsity.

### 1.3 Why Not General Alpha?

- General alpha-entmax requires bisection search for threshold, adding ~10-20 iterations
- Performance overhead not justified for E23's small N (8-64 slots)
- Learnable alpha adds complexity; fixed alpha sufficient for memory addressing

---

## 2. Mathematical Foundations

### 2.1 Sparsemax (alpha = 2)

Sparsemax solves the Euclidean projection onto the probability simplex:

```
sparsemax(z) = argmin_{p in simplex} ||p - z||^2
```

**Closed-form solution:**
```
sparsemax_i(z) = [z_i - tau(z)]+
```

where `[t]+ = max(0, t)` and `tau(z)` is the threshold ensuring outputs sum to 1.

**Threshold finding (sorted algorithm):**
```
Sort z in descending order: z[1] >= z[2] >= ... >= z[K]
Find k* = max{k : 1 + k*z[k] > cumsum(z[1:k])}
tau = (cumsum(z[1:k*]) - 1) / k*
```

### 2.2 1.5-Entmax (alpha = 1.5)

1.5-entmax maximizes Tsallis 1.5-entropy subject to moment constraints:

```
1.5-entmax(z) = argmax_{p in simplex} [p^T z + H_1.5(p)]
```

**Closed-form solution:**
```
1.5-entmax_i(z) = [z_i - tau(z)]^2_+
```

where the output is the **square** of the thresholded value (not just ReLU).

**Threshold finding (key insight: uses mean and variance):**
```
Sort z in descending order
For each candidate support size k:
    mean_k = mean(z[1:k])
    ss_k = sum((z[1:k] - mean_k)^2) / k^2  # scaled variance
    delta_k = (1 - ss_k) / k
    tau_k = mean_k - sqrt(max(0, delta_k))

Find k* = max{k : tau_k <= z[k]}
tau = tau_{k*}
```

### 2.3 Jacobian Structure

**Sparsemax Jacobian:**
```
Let S = support(sparsemax(z))  // indices where output > 0
d(sparsemax)/dz = I_S - (1/|S|) * 1_S * 1_S^T
```

This is a rank-1 update to a diagonal matrix, restricted to the support set.

**1.5-Entmax Jacobian:**
```
Let S = support(entmax(z)), p = entmax(z), g = sqrt(p)
d(entmax)/dz = 2 * diag(g)_S - (2 / sum(g)) * g_S * g_S^T
```

Also a rank-1 update structure, enabling efficient backward computation.

---

## 3. Forward Pass Implementation

### 3.1 Design for E23

The E23 kernel operates on:
- Batch size B (variable, typically 32-256)
- N_SLOTS: 8, 16, 32, or 64 tape slots
- DIM: 256, 512, 768, or 1024

**Key constraint**: N is small (max 64), so sorting overhead is acceptable.

### 3.2 Sorting Strategy for Small N

For N <= 64, use **bitonic sort** in shared memory:
- O(log^2 N) parallel steps
- No global memory access after initial load
- Well-suited for warp-level execution

**Alternative for N <= 32**: Use warp shuffle-based sorting (entire sort within a warp).

### 3.3 1.5-Entmax Forward Kernel Design

```
Phase 1: Compute attention scores (unchanged from E23)
    scores[n] = dot(h_tape[n], h_work) * scale

Phase 2: Find threshold tau
    a) Load scores into shared memory
    b) Bitonic sort (in-place, shared memory)
    c) Compute cumulative statistics:
       - Running mean
       - Running scaled variance (ss)
       - Candidate tau values
    d) Find support size k* via parallel scan
    e) Compute final tau

Phase 3: Apply transformation
    p[n] = max(0, scores[n] - tau)^2

Phase 4: Normalize (ensure sum = 1)
    p[n] /= sum(p)  // Numerical stability correction
```

### 3.4 Numerical Stability Considerations

1. **Pre-scaling**: Divide scores by max before sorting (like softmax max-subtraction)
2. **Epsilon guarding**: Add small epsilon to prevent division by zero in normalization
3. **Clamping delta**: Ensure `delta >= 0` before sqrt to avoid NaN

---

## 4. Backward Pass Implementation

### 4.1 Backward Pass Requirements

E23 backward pass needs gradients through:
1. Read attention: d_h_work, d_h_tape from d_read_value
2. Write attention: d_h_work_new, d_h_tape from d_tape_updated

### 4.2 1.5-Entmax Backward Algorithm

**Input:**
- `p`: forward pass output (attention weights)
- `dp`: gradient w.r.t. output

**Output:**
- `dz`: gradient w.r.t. input scores

**Algorithm:**
```
// Compute auxiliary variables
S = {i : p[i] > 0}  // support set
g[i] = sqrt(p[i]) for i in S, else 0

// Compute intermediate
v = sum(dp[i] * g[i] for i in S)  // scalar
g_sum = sum(g[i] for i in S)

// Compute gradient
dz[i] = 2 * g[i] * (dp[i] - v / g_sum) for i in S
dz[i] = 0 for i not in S
```

### 4.3 Sparse Gradient Efficiency

The backward pass only needs to compute gradients for non-zero outputs:
- For k non-zero attention weights, only k gradient terms
- Perfect for E23 where we want few active slots

**Memory pattern**: Store support mask alongside attention weights for backward.

### 4.4 Jacobian-Vector Product (JVP) Form

For integration with autograd, express as JVP:
```
// Forward saved: p (attention), S (support mask)
// Backward input: dp

g = sqrt(p)  // only compute for S
g_dp = g * dp  // elementwise, only for S
v = sum(g_dp) / sum(g)  // single scalar
dz = 2 * (g_dp - v * g)  // only for S
```

This requires:
- 1 sqrt per non-zero element
- 2 reductions (sum of g_dp, sum of g)
- 2 elementwise ops per non-zero element

---

## 5. Performance Considerations

### 5.1 Comparison with Softmax

| Operation | Softmax | 1.5-Entmax | Delta |
|-----------|---------|------------|-------|
| Score computation | Same | Same | 0 |
| Max finding | O(N) | O(N log N) sort | +2x |
| Normalization | O(N) exp + sum | O(N) ReLU + sq + sum | Similar |
| Backward | O(N) | O(k) where k <= N | Better if sparse |

**Expected overhead**: ~10-20% for forward pass, potentially faster backward if attention is sparse.

### 5.2 Memory Access Patterns

**Forward pass:**
- Scores loaded once to shared memory
- Sorted in-place in shared memory
- Output written coalesced to global memory

**Backward pass:**
- Need to store support mask (1 bit per slot, or use threshold comparison)
- Sparse gradient computation reduces memory bandwidth

### 5.3 Shared Memory Requirements

For N_SLOTS = 64, DIM = 1024:
```
Current E23:
- h_work_sh: 1024 floats = 4KB
- attn_sh: 64 floats = 256B
- h_work_new_sh: 1024 floats = 4KB
- write_val_sh: 1024 floats = 4KB
Total: ~12.3KB

With entmax additions:
- sorted_scores: 64 floats = 256B (reuse attn_sh)
- sorted_indices: 64 ints = 256B (new)
- cumsum_buffer: 64 floats = 256B (new)
- support_mask: 64 bits = 8B (new)
Total additional: ~520B
```

Fits comfortably within 48KB shared memory limit.

### 5.4 Warp Divergence

**Concern**: Support size varies per batch element, causing divergence.

**Mitigation**:
- Use warp-uniform branching where possible
- Compute all k candidates in parallel, select valid one
- For small N, divergence cost is minimal

### 5.5 Occupancy Analysis

Current E23 uses 256 threads per block with ~12KB shared memory:
- Theoretical occupancy: ~75% (limited by shared memory)
- Adding 520B for entmax: negligible impact on occupancy

---

## 6. Integration with E23 Kernel

### 6.1 Code Structure

Replace softmax sections in both Phase1 and Phase2 kernels:

```cpp
// Current softmax (lines 87-100 in E23Phase1Kernel_BF16)
if (tid == 0) {
    float max_score = attn_sh[0];
    // ... softmax computation
}

// Replace with:
if (tid < N_SLOTS) {
    // Parallel entmax (see pseudocode below)
}
```

### 6.2 Template Parameters

Add compile-time alpha selection:
```cpp
enum class AttentionType { SOFTMAX, SPARSEMAX, ENTMAX_1_5 };

template<int N_SLOTS, int DIM, AttentionType ATTN_TYPE = AttentionType::ENTMAX_1_5>
__global__ void E23Phase1Kernel_BF16(...);
```

### 6.3 Backward Pass Integration

Modify backward kernels to handle sparse Jacobian:

```cpp
// E23BackwardPhase3: Read attention backward
// Current: softmax backward
// New: entmax backward with support tracking
```

### 6.4 Python Interface

Add configuration option:
```python
class DualMemoryElman(nn.Module):
    def __init__(self, ..., attention_type='entmax_1.5'):
        # attention_type: 'softmax', 'sparsemax', 'entmax_1.5'
```

---

## 7. Pseudocode

### 7.1 Forward Pass: 1.5-Entmax Kernel

```cpp
/**
 * 1.5-Entmax attention computation
 * Replaces softmax in E23 Phase1/Phase2 kernels
 *
 * Input: attn_sh[N_SLOTS] - attention scores (pre-loaded)
 * Output: attn_sh[N_SLOTS] - attention weights (sparse)
 */
template<int N_SLOTS>
__device__ void entmax_1_5_forward(float* attn_sh, const int tid) {
    // Shared memory for sorting
    __shared__ float sorted_vals[N_SLOTS];
    __shared__ int sorted_idx[N_SLOTS];
    __shared__ float cumsum[N_SLOTS];
    __shared__ float tau_candidates[N_SLOTS];
    __shared__ int support_size;

    // ========== Phase 1: Initialize and find max for stability ==========
    if (tid < N_SLOTS) {
        sorted_vals[tid] = attn_sh[tid];
        sorted_idx[tid] = tid;
    }
    __syncthreads();

    // ========== Phase 2: Bitonic sort (descending) ==========
    // For N_SLOTS <= 64, this is efficient in shared memory
    for (int k = 2; k <= N_SLOTS; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            if (tid < N_SLOTS) {
                int ixj = tid ^ j;
                if (ixj > tid) {
                    bool ascending = ((tid & k) == 0);
                    // For descending sort, flip the comparison
                    if ((sorted_vals[tid] < sorted_vals[ixj]) != ascending) {
                        // Swap
                        float tmp_val = sorted_vals[tid];
                        sorted_vals[tid] = sorted_vals[ixj];
                        sorted_vals[ixj] = tmp_val;

                        int tmp_idx = sorted_idx[tid];
                        sorted_idx[tid] = sorted_idx[ixj];
                        sorted_idx[ixj] = tmp_idx;
                    }
                }
            }
            __syncthreads();
        }
    }

    // ========== Phase 3: Compute tau candidates ==========
    // Parallel prefix sum for cumulative sum
    if (tid < N_SLOTS) {
        cumsum[tid] = sorted_vals[tid];
    }
    __syncthreads();

    // Inclusive scan (Hillis-Steele for small N)
    for (int offset = 1; offset < N_SLOTS; offset <<= 1) {
        float val = (tid >= offset) ? cumsum[tid - offset] : 0.0f;
        __syncthreads();
        if (tid < N_SLOTS) {
            cumsum[tid] += val;
        }
        __syncthreads();
    }

    // Compute tau for each candidate k (1.5-entmax formula)
    if (tid < N_SLOTS) {
        int k = tid + 1;  // Support size candidate
        float sum_k = cumsum[tid];
        float mean_k = sum_k / k;

        // Compute scaled variance: ss = sum((z - mean)^2) / k^2
        // Use cumsum of squares minus (sum^2)/k, normalized by k^2
        float sum_sq = 0.0f;
        for (int i = 0; i <= tid; i++) {
            float diff = sorted_vals[i] - mean_k;
            sum_sq += diff * diff;
        }
        float ss_k = sum_sq / (k * k);

        // Compute delta and tau
        float delta_k = (1.0f - ss_k) / k;
        float tau_k = mean_k - sqrtf(fmaxf(0.0f, delta_k));
        tau_candidates[tid] = tau_k;
    }
    __syncthreads();

    // ========== Phase 4: Find support size k* ==========
    // k* = max{k : tau_k <= sorted_vals[k-1]}
    if (tid == 0) {
        support_size = N_SLOTS;  // Default: all in support
        for (int k = 1; k <= N_SLOTS; k++) {
            if (tau_candidates[k-1] > sorted_vals[k-1]) {
                support_size = k - 1;
                break;
            }
        }
        if (support_size == 0) support_size = 1;  // At least one element
    }
    __syncthreads();

    float tau_star = tau_candidates[support_size - 1];

    // ========== Phase 5: Apply transformation ==========
    if (tid < N_SLOTS) {
        float z = attn_sh[tid];  // Original unsorted scores
        float p = z - tau_star;
        p = fmaxf(0.0f, p);     // ReLU
        p = p * p;              // Square for 1.5-entmax
        attn_sh[tid] = p;
    }
    __syncthreads();

    // ========== Phase 6: Normalize ==========
    // Parallel reduction for sum
    __shared__ float sum_p;
    if (tid == 0) {
        sum_p = 0.0f;
        for (int i = 0; i < N_SLOTS; i++) {
            sum_p += attn_sh[i];
        }
        sum_p = fmaxf(sum_p, 1e-9f);  // Prevent division by zero
    }
    __syncthreads();

    if (tid < N_SLOTS) {
        attn_sh[tid] /= sum_p;
    }
}
```

### 7.2 Backward Pass: 1.5-Entmax Gradient

```cpp
/**
 * 1.5-Entmax backward pass
 * Computes gradient d_scores from d_attention
 *
 * Input:
 *   p[N_SLOTS] - forward pass attention weights
 *   dp[N_SLOTS] - gradient w.r.t. attention weights
 * Output:
 *   dz[N_SLOTS] - gradient w.r.t. input scores
 */
template<int N_SLOTS>
__device__ void entmax_1_5_backward(
    const float* p,    // Attention weights from forward
    const float* dp,   // Incoming gradient
    float* dz,         // Output gradient
    const int tid
) {
    __shared__ float g[N_SLOTS];      // sqrt(p)
    __shared__ float g_dp[N_SLOTS];   // g * dp
    __shared__ float sum_g;           // sum of g
    __shared__ float sum_g_dp;        // sum of g * dp

    // ========== Phase 1: Compute g = sqrt(p) and g*dp ==========
    if (tid < N_SLOTS) {
        float p_i = p[tid];
        float g_i = sqrtf(fmaxf(p_i, 0.0f));
        g[tid] = g_i;
        g_dp[tid] = g_i * dp[tid];
    }
    __syncthreads();

    // ========== Phase 2: Compute sums (reduction) ==========
    if (tid == 0) {
        float s_g = 0.0f;
        float s_gdp = 0.0f;
        for (int i = 0; i < N_SLOTS; i++) {
            s_g += g[i];
            s_gdp += g_dp[i];
        }
        sum_g = fmaxf(s_g, 1e-9f);  // Prevent division by zero
        sum_g_dp = s_gdp;
    }
    __syncthreads();

    // ========== Phase 3: Compute output gradient ==========
    // dz[i] = 2 * g[i] * (dp[i] - sum_g_dp / sum_g)
    // Only non-zero where p[i] > 0 (but sparse zeros handled naturally)
    if (tid < N_SLOTS) {
        float v = sum_g_dp / sum_g;
        dz[tid] = 2.0f * g[tid] * (dp[tid] - v);
    }
}
```

### 7.3 Sparsemax Alternative (alpha = 2)

```cpp
/**
 * Sparsemax attention (simpler than 1.5-entmax)
 * Output: p[i] = max(0, z[i] - tau)
 */
template<int N_SLOTS>
__device__ void sparsemax_forward(float* attn_sh, const int tid) {
    __shared__ float sorted_vals[N_SLOTS];
    __shared__ float cumsum[N_SLOTS];
    __shared__ int support_size;
    __shared__ float tau_star;

    // Initialize
    if (tid < N_SLOTS) {
        sorted_vals[tid] = attn_sh[tid];
    }
    __syncthreads();

    // Bitonic sort (descending) - same as above
    // ... sorting code ...
    __syncthreads();

    // Compute cumsum
    if (tid < N_SLOTS) {
        cumsum[tid] = sorted_vals[tid];
    }
    __syncthreads();

    for (int offset = 1; offset < N_SLOTS; offset <<= 1) {
        float val = (tid >= offset) ? cumsum[tid - offset] : 0.0f;
        __syncthreads();
        if (tid < N_SLOTS) cumsum[tid] += val;
        __syncthreads();
    }

    // Find k* and tau for sparsemax
    // k* = max{k : 1 + k*z[k] > cumsum[k]}
    if (tid == 0) {
        support_size = 1;
        for (int k = 1; k <= N_SLOTS; k++) {
            float rhs = 1.0f + k * sorted_vals[k-1];
            if (rhs > cumsum[k-1]) {
                support_size = k;
            }
        }
        tau_star = (cumsum[support_size-1] - 1.0f) / support_size;
    }
    __syncthreads();

    // Apply sparsemax: p = max(0, z - tau)
    if (tid < N_SLOTS) {
        attn_sh[tid] = fmaxf(0.0f, attn_sh[tid] - tau_star);
    }
}

/**
 * Sparsemax backward (simpler Jacobian)
 * dz[i] = dp[i] - mean(dp[S]) for i in support S
 * dz[i] = 0 otherwise
 */
template<int N_SLOTS>
__device__ void sparsemax_backward(
    const float* p,
    const float* dp,
    float* dz,
    const int tid
) {
    __shared__ float sum_dp;
    __shared__ int support_count;

    // Count support and sum dp over support
    if (tid == 0) {
        float s = 0.0f;
        int cnt = 0;
        for (int i = 0; i < N_SLOTS; i++) {
            if (p[i] > 0.0f) {
                s += dp[i];
                cnt++;
            }
        }
        sum_dp = s;
        support_count = fmaxf(cnt, 1);
    }
    __syncthreads();

    float v_hat = sum_dp / support_count;

    if (tid < N_SLOTS) {
        dz[tid] = (p[tid] > 0.0f) ? (dp[tid] - v_hat) : 0.0f;
    }
}
```

### 7.4 Integration into E23FusedKernel

```cpp
template<int N_SLOTS, int DIM, AttentionType ATTN_TYPE>
__global__ void E23FusedKernel_BF16(
    // ... existing parameters ...
) {
    // ... existing code for loading h_work, computing scores ...

    // ============================================
    // REPLACE SOFTMAX WITH ENTMAX
    // ============================================

    if constexpr (ATTN_TYPE == AttentionType::SOFTMAX) {
        // Original softmax code (single thread)
        if (tid == 0) {
            float max_score = attn_sh[0];
            for (int n = 1; n < N_SLOTS; n++) {
                max_score = fmaxf(max_score, attn_sh[n]);
            }
            float sum_exp = 0.0f;
            for (int n = 0; n < N_SLOTS; n++) {
                attn_sh[n] = expf(attn_sh[n] - max_score);
                sum_exp += attn_sh[n];
            }
            for (int n = 0; n < N_SLOTS; n++) {
                attn_sh[n] /= sum_exp;
            }
        }
    }
    else if constexpr (ATTN_TYPE == AttentionType::ENTMAX_1_5) {
        // 1.5-entmax (parallel)
        entmax_1_5_forward<N_SLOTS>(attn_sh, tid);
    }
    else if constexpr (ATTN_TYPE == AttentionType::SPARSEMAX) {
        // Sparsemax (parallel)
        sparsemax_forward<N_SLOTS>(attn_sh, tid);
    }
    __syncthreads();

    // ... rest of kernel unchanged ...
}
```

---

## 8. Testing and Validation

### 8.1 Numerical Correctness Tests

1. **Reference comparison**: Compare CUDA output against PyTorch entmax library
2. **Gradient check**: Numerical gradient vs analytical gradient
3. **Edge cases**:
   - All scores equal (should give uniform)
   - One score much larger (should give near-one-hot)
   - Negative scores
   - Very large/small scores (numerical stability)

### 8.2 Performance Benchmarks

1. **Microbenchmark**: Isolated entmax kernel vs softmax kernel
2. **End-to-end**: Full E23 step timing comparison
3. **Throughput**: Tokens/second on benchmark tasks

### 8.3 Sparsity Analysis

1. **Attention entropy**: Measure H(attention) across training
2. **Support size**: Track average number of non-zero attention weights
3. **Memory utilization**: Verify sparse patterns emerge

### 8.4 Training Stability

1. **Loss curves**: Compare softmax vs entmax training dynamics
2. **Gradient norms**: Monitor for exploding/vanishing gradients
3. **Convergence**: Verify comparable final performance

---

## Appendix A: References

- [Martins & Astudillo 2016](https://arxiv.org/abs/1602.02068): From Softmax to Sparsemax
- [Peters et al. 2019](https://arxiv.org/abs/1905.05702): Sparse Sequence-to-Sequence Models
- [Correia et al. 2019](https://arxiv.org/abs/1909.00015): Adaptively Sparse Transformers
- [deep-spin/entmax](https://github.com/deep-spin/entmax): Reference implementation

## Appendix B: Implementation Checklist

- [ ] Implement 1.5-entmax forward kernel
- [ ] Implement 1.5-entmax backward kernel
- [ ] Implement sparsemax forward kernel (optional)
- [ ] Implement sparsemax backward kernel (optional)
- [ ] Add template parameter for attention type
- [ ] Update Python bindings
- [ ] Add configuration option to DualMemoryElman
- [ ] Write unit tests
- [ ] Run performance benchmarks
- [ ] Validate on E23 benchmark tasks

## Appendix C: Performance Optimization Notes

### C.1 Warp-Level Sorting for N <= 32

For N_SLOTS <= 32, the entire sort can happen within a single warp using shuffle instructions:

```cpp
// Warp-level bitonic sort using __shfl_xor_sync
__device__ float warp_bitonic_sort(float val, int lane_id) {
    for (int k = 2; k <= 32; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            float other = __shfl_xor_sync(0xffffffff, val, j);
            bool ascending = ((lane_id & k) == 0);
            if ((val < other) != ascending) {
                val = other;
            }
        }
    }
    return val;
}
```

### C.2 Avoiding Sorting Entirely

For 1.5-entmax with very small N (8-16), an alternative approach:

1. Compute all N tau candidates in parallel (no sorting needed for tau computation)
2. Use parallel reduction to find the valid k*
3. This trades O(N^2) work for O(N) parallelism

May be faster for N=8 due to avoiding sorting overhead entirely.

### C.3 Fused Read-Value Computation

Currently, attention weights are stored, then read value is computed. With sparse attention, we can fuse:

```cpp
// Instead of:
// 1. Compute attention (store to shared)
// 2. For each d: sum over n of attn[n] * tape[n,d]

// Do:
// 1. Compute attention
// 2. Identify non-zero indices (small set)
// 3. For each d: sum only over non-zero indices

// This reduces memory bandwidth when attention is sparse
```
