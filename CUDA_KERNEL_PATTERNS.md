# CUDA Kernel Patterns: Hybrid Template + Dynamic Shared Memory

## The Problem

Current kernels template on both `N_SLOTS` and `DIM`:

```cpp
template<int N_SLOTS, int DIM>
__global__ void E29Kernel(...) {
    __shared__ float h_work_sh[DIM];
    __shared__ float attn_sh[N_SLOTS];
    // ...
}
```

This requires instantiating for every combination:
- N_SLOTS: 8, 16, 32, 64 (4 values)
- DIM: 512, 768, 1024, 1280 (4+ values)
- Total: 16+ specializations per kernel

Adding a new DIM requires touching every kernel file.

## The Solution: Hybrid Approach

**Template on N_SLOTS** (small, benefits from unrolling)
**Dynamic shared memory for DIM** (large, memory-bound anyway)

```cpp
template<int N_SLOTS>
__global__ void E29Kernel(const int DIM, ...) {
    extern __shared__ char shared_mem[];
    float* h_work_sh = (float*)shared_mem;
    float* attn_sh = h_work_sh + DIM;
    float* read_val_sh = attn_sh + N_SLOTS;
    // ...
}
```

Now only 4 specializations (one per N_SLOTS), and DIM is passed at runtime.

---

## Complete Example: E29a Hybrid Kernel

### Header

```cpp
/**
 * E29a: Selective Gating Dual-Memory Elman
 *
 * Hybrid kernel: N_SLOTS templated, DIM dynamic.
 * Reduces specialization burden while maintaining performance.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {
constexpr int BLOCK_SIZE = 256;

// Softmax - N_SLOTS templated for unrolling
template<int N_SLOTS>
__device__ void softmax_device(float* attn_sh, const float scale) {
    // Find max (unrolled for small N_SLOTS)
    float max_val = attn_sh[0] * scale;
    #pragma unroll
    for (int i = 1; i < N_SLOTS; i++) {
        float v = attn_sh[i] * scale;
        if (v > max_val) max_val = v;
    }

    // Compute exp and sum (unrolled)
    float sum_exp = 0.0f;
    #pragma unroll
    for (int i = 0; i < N_SLOTS; i++) {
        attn_sh[i] = expf(attn_sh[i] * scale - max_val);
        sum_exp += attn_sh[i];
    }

    // Normalize (unrolled)
    sum_exp = fmaxf(sum_exp, 1e-9f);
    #pragma unroll
    for (int i = 0; i < N_SLOTS; i++) {
        attn_sh[i] /= sum_exp;
    }
}

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}
```

### Phase 1 Kernel: Read + Update

```cpp
/**
 * E29a Phase 1: Read attention + h_work update
 *
 * Template: N_SLOTS (for attention unrolling)
 * Dynamic: DIM (via shared memory)
 *
 * Shared memory layout: [h_work_sh(DIM), h_tape_sh(N_SLOTS*DIM), attn_sh(N_SLOTS), read_val_sh(DIM)]
 * Total: (2 + N_SLOTS) * DIM + N_SLOTS floats
 */
template<int N_SLOTS>
__global__ void E29aPhase1Kernel_Hybrid(
    const int DIM,                              // Runtime dimension
    const int batch_size,
    const __nv_bfloat16* __restrict__ Rh,       // [B, DIM]
    const __nv_bfloat16* __restrict__ x_proj_t, // [B, DIM]
    const __nv_bfloat16* __restrict__ b_h,      // [DIM]
    const __nv_bfloat16* __restrict__ h_tape,   // [B, N_SLOTS, DIM]
    const __nv_bfloat16* __restrict__ h_work,   // [B, DIM]
    __nv_bfloat16* __restrict__ h_work_out,     // [B, DIM]
    __nv_bfloat16* __restrict__ read_attn_out,  // [B, N_SLOTS]
    __nv_bfloat16* __restrict__ read_val_out,   // [B, DIM]
    const float scale
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    // Dynamic shared memory layout
    extern __shared__ char shared_mem[];
    float* h_work_sh = (float*)shared_mem;
    float* attn_sh = h_work_sh + DIM;
    float* read_val_sh = attn_sh + N_SLOTS;

    // Load h_work into shared memory (DIM is runtime, use loop)
    for (int d = tid; d < DIM; d += BLOCK_SIZE) {
        h_work_sh[d] = __bfloat162float(h_work[b * DIM + d]);
    }
    __syncthreads();

    // Compute attention scores (N_SLOTS is compile-time, can hint unroll)
    if (tid < N_SLOTS) {
        float score = 0.0f;
        const __nv_bfloat16* tape_slot = h_tape + b * N_SLOTS * DIM + tid * DIM;
        // DIM loop - not unrolled (runtime), but memory-bound anyway
        for (int d = 0; d < DIM; d++) {
            score += __bfloat162float(tape_slot[d]) * h_work_sh[d];
        }
        attn_sh[tid] = score;
    }
    __syncthreads();

    // Softmax (N_SLOTS templated - unrolls efficiently)
    if (tid == 0) {
        softmax_device<N_SLOTS>(attn_sh, scale);
    }
    __syncthreads();

    // Compute read value: weighted sum over tape
    for (int d = tid; d < DIM; d += BLOCK_SIZE) {
        float val = 0.0f;
        #pragma unroll
        for (int n = 0; n < N_SLOTS; n++) {
            val += attn_sh[n] * __bfloat162float(h_tape[b * N_SLOTS * DIM + n * DIM + d]);
        }
        read_val_sh[d] = val;
    }
    __syncthreads();

    // Update h_work: tanh(x_proj + Rh + read + b)
    for (int d = tid; d < DIM; d += BLOCK_SIZE) {
        float x = __bfloat162float(x_proj_t[b * DIM + d]);
        float r = __bfloat162float(Rh[b * DIM + d]);
        float bias = __bfloat162float(b_h[d]);
        float h_new = tanhf(x + r + read_val_sh[d] + bias);

        h_work_out[b * DIM + d] = __float2bfloat16(h_new);
        read_val_out[b * DIM + d] = __float2bfloat16(read_val_sh[d]);
    }

    // Store attention weights
    if (tid < N_SLOTS) {
        read_attn_out[b * N_SLOTS + tid] = __float2bfloat16(attn_sh[tid]);
    }
}
```

### Phase 2 Kernel: Write + Output

```cpp
/**
 * E29a Phase 2: Write attention + selective output
 *
 * gate = silu(z + read_val + h_work_new)
 * output = h_work_new * gate
 */
template<int N_SLOTS>
__global__ void E29aPhase2Kernel_Hybrid(
    const int DIM,
    const int batch_size,
    const __nv_bfloat16* __restrict__ h_tape,      // [B, N_SLOTS, DIM]
    const __nv_bfloat16* __restrict__ h_work_new,  // [B, DIM]
    const __nv_bfloat16* __restrict__ z_t,         // [B, DIM]
    const __nv_bfloat16* __restrict__ read_val,    // [B, DIM]
    const __nv_bfloat16* __restrict__ write_val,   // [B, DIM]
    __nv_bfloat16* __restrict__ h_tape_out,        // [B, N_SLOTS, DIM]
    __nv_bfloat16* __restrict__ output,            // [B, DIM]
    __nv_bfloat16* __restrict__ write_attn_out,    // [B, N_SLOTS]
    const float scale
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    extern __shared__ char shared_mem[];
    float* write_val_sh = (float*)shared_mem;
    float* attn_sh = write_val_sh + DIM;

    // Load write_val
    for (int d = tid; d < DIM; d += BLOCK_SIZE) {
        write_val_sh[d] = __bfloat162float(write_val[b * DIM + d]);
    }
    __syncthreads();

    // Compute write attention scores
    if (tid < N_SLOTS) {
        float score = 0.0f;
        const __nv_bfloat16* tape_slot = h_tape + b * N_SLOTS * DIM + tid * DIM;
        for (int d = 0; d < DIM; d++) {
            score += __bfloat162float(tape_slot[d]) * write_val_sh[d];
        }
        attn_sh[tid] = score;
    }
    __syncthreads();

    // Softmax
    if (tid == 0) {
        softmax_device<N_SLOTS>(attn_sh, scale);
    }
    __syncthreads();

    // Update tape and compute output
    for (int d = tid; d < DIM; d += BLOCK_SIZE) {
        float wv = write_val_sh[d];

        // Update each tape slot
        #pragma unroll
        for (int n = 0; n < N_SLOTS; n++) {
            float old_val = __bfloat162float(h_tape[b * N_SLOTS * DIM + n * DIM + d]);
            float new_val = old_val * (1.0f - attn_sh[n]) + wv * attn_sh[n];
            h_tape_out[b * N_SLOTS * DIM + n * DIM + d] = __float2bfloat16(new_val);
        }

        // Selective output: gate = silu(z + read + h_work_new)
        float z = __bfloat162float(z_t[b * DIM + d]);
        float rv = __bfloat162float(read_val[b * DIM + d]);
        float hw = __bfloat162float(h_work_new[b * DIM + d]);
        float gate = silu(z + rv + hw);
        output[b * DIM + d] = __float2bfloat16(hw * gate);
    }

    // Store write attention
    if (tid < N_SLOTS) {
        write_attn_out[b * N_SLOTS + tid] = __float2bfloat16(attn_sh[tid]);
    }
}
```

### Launch Wrapper

```cpp
/**
 * Launch wrapper - dispatches based on N_SLOTS, DIM is runtime
 */
void launch_e29a_phase1(
    int N_SLOTS, int DIM, int batch_size,
    const __nv_bfloat16* Rh, const __nv_bfloat16* x_proj_t,
    const __nv_bfloat16* b_h, const __nv_bfloat16* h_tape,
    const __nv_bfloat16* h_work, __nv_bfloat16* h_work_out,
    __nv_bfloat16* read_attn_out, __nv_bfloat16* read_val_out,
    float scale, cudaStream_t stream
) {
    // Shared memory: h_work(DIM) + attn(N_SLOTS) + read_val(DIM)
    size_t shared_size = (2 * DIM + N_SLOTS) * sizeof(float);

    dim3 grid(batch_size);
    dim3 block(BLOCK_SIZE);

    // Dispatch on N_SLOTS only (4 cases instead of 16+)
    switch (N_SLOTS) {
        case 8:
            E29aPhase1Kernel_Hybrid<8><<<grid, block, shared_size, stream>>>(
                DIM, batch_size, Rh, x_proj_t, b_h, h_tape, h_work,
                h_work_out, read_attn_out, read_val_out, scale);
            break;
        case 16:
            E29aPhase1Kernel_Hybrid<16><<<grid, block, shared_size, stream>>>(
                DIM, batch_size, Rh, x_proj_t, b_h, h_tape, h_work,
                h_work_out, read_attn_out, read_val_out, scale);
            break;
        case 32:
            E29aPhase1Kernel_Hybrid<32><<<grid, block, shared_size, stream>>>(
                DIM, batch_size, Rh, x_proj_t, b_h, h_tape, h_work,
                h_work_out, read_attn_out, read_val_out, scale);
            break;
        case 64:
            E29aPhase1Kernel_Hybrid<64><<<grid, block, shared_size, stream>>>(
                DIM, batch_size, Rh, x_proj_t, b_h, h_tape, h_work,
                h_work_out, read_attn_out, read_val_out, scale);
            break;
        default:
            fprintf(stderr, "Unsupported N_SLOTS: %d\n", N_SLOTS);
    }
}
```

### Instantiation (Minimal!)

```cpp
// Only need to instantiate for N_SLOTS values
template __global__ void E29aPhase1Kernel_Hybrid<8>(int, int, ...);
template __global__ void E29aPhase1Kernel_Hybrid<16>(int, int, ...);
template __global__ void E29aPhase1Kernel_Hybrid<32>(int, int, ...);
template __global__ void E29aPhase1Kernel_Hybrid<64>(int, int, ...);

template __global__ void E29aPhase2Kernel_Hybrid<8>(int, int, ...);
template __global__ void E29aPhase2Kernel_Hybrid<16>(int, int, ...);
template __global__ void E29aPhase2Kernel_Hybrid<32>(int, int, ...);
template __global__ void E29aPhase2Kernel_Hybrid<64>(int, int, ...);

// That's it! 8 instantiations instead of 32+
```

---

## Performance Comparison

| Approach | Specializations | Compile Time | Runtime Overhead |
|----------|-----------------|--------------|------------------|
| Full template `<N,D>` | N_vals × D_vals | High | None |
| Hybrid `<N>` + dynamic D | N_vals only | **4x lower** | ~5% |
| Fully dynamic | 1 | Minimal | ~10-15% |

The hybrid approach is the sweet spot:
- Attention loops (N_SLOTS) are unrolled → fast softmax
- Dimension loops (DIM) are runtime → flexible, still memory-bound
- 4 instantiations per kernel instead of 16+

---

## Checklist for Converting Existing Kernels

1. [ ] Change template from `<N_SLOTS, DIM>` to `<N_SLOTS>`
2. [ ] Add `const int DIM` as first kernel parameter
3. [ ] Replace static shared memory with `extern __shared__ char shared_mem[]`
4. [ ] Calculate shared memory layout: `float* ptr1 = (float*)shared_mem; float* ptr2 = ptr1 + DIM;`
5. [ ] Pass shared memory size at launch: `<<<grid, block, shared_size, stream>>>`
6. [ ] Update launch wrapper to dispatch only on N_SLOTS
7. [ ] Remove DIM instantiations, keep only N_SLOTS instantiations
8. [ ] Test: verify CUDA output matches Python reference for various DIMs

---

## Files to Update

When converting to hybrid approach:

```
elman/cuda/lib/e25_entmax_gpu.cu.cc
elman/cuda/lib/e26_parallel_gpu.cu.cc
elman/cuda/lib/e29_selective_gpu.cu.cc
elman/cuda/lib/dual_memory_elman_gpu.cu.cc  (E23)
```

Each conversion removes ~75% of template instantiations.
