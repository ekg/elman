// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// Level 17: Selective W_h Elman - Input-dependent gating on recurrence
//
// Key insight: Mamba2's advantage may come from input-dependent A matrix.
// This adds diagonal selectivity to E1's dense W_h @ h recurrence.
//
// Architecture:
// x, z = split(in_proj(x))           # Pre-computed before kernel
// x = silu(x)                        # Pre-computed before kernel
// G = W_gate @ x + b_gate            # Gate projection
// gate = sigmoid(G)                  # [B, dim] diagonal gate
// Rh = W_h @ h_{t-1}                 # Dense recurrence
// h_t = tanh(W_x @ x_t + Rh * gate + b)  # Gated recurrence (like Mamba2's selective A)
// output = h * silu(z)               # Gate with z branch

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <algorithm>

#include "hasty/elman_ladder.h"
#include "blas.h"
#include "inline_ops.h"

namespace {

// =============================================================================
// Native BF16 operations (fast path for SM80+)
// =============================================================================

// Native bf16 add - uses hardware instruction on Ampere+
__device__ __forceinline__ __nv_bfloat16 bf16_add(__nv_bfloat16 a, __nv_bfloat16 b) {
#if __CUDA_ARCH__ >= 800
    return __hadd(a, b);
#else
    return __float2bfloat16(__bfloat162float(a) + __bfloat162float(b));
#endif
}

// Native bf16 multiply
__device__ __forceinline__ __nv_bfloat16 bf16_mul(__nv_bfloat16 a, __nv_bfloat16 b) {
#if __CUDA_ARCH__ >= 800
    return __hmul(a, b);
#else
    return __float2bfloat16(__bfloat162float(a) * __bfloat162float(b));
#endif
}

// Native bf16 fused multiply-add: a * b + c
__device__ __forceinline__ __nv_bfloat16 bf16_fma(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c) {
#if __CUDA_ARCH__ >= 800
    return __hfma(a, b, c);
#else
    return __float2bfloat16(__bfloat162float(a) * __bfloat162float(b) + __bfloat162float(c));
#endif
}

// =============================================================================
// Optimized Forward Kernels
// =============================================================================

// BF16-optimized: Fused Wx + Rh + bias + tanh
// Uses native bf16 adds, only converts to f32 for tanh (unavoidable)
__global__ void FusedTanhKernel_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ Wx,
    const __nv_bfloat16* __restrict__ Rh,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ h_out,
    __nv_bfloat16* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // Native bf16 additions (no f32 conversion)
        __nv_bfloat16 sum = bf16_add(bf16_add(Wx[idx], Rh[idx]), b[d]);

        // Store pre-activation in bf16
        if (v_cache) v_cache[idx] = sum;

        // Only convert to f32 for tanh (no bf16 tanh in CUDA)
        float val = __bfloat162float(sum);
        h_out[idx] = __float2bfloat16(tanhf(val));
    }
}

// Generic version for other types (float, half, double)
template<typename T>
__global__ void FusedTanhKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ Wx,
    const T* __restrict__ Rh,
    const T* __restrict__ b,
    T* __restrict__ h_out,
    T* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;
        float val = static_cast<float>(Wx[idx]) + static_cast<float>(Rh[idx]) + static_cast<float>(b[d]);
        if (v_cache) v_cache[idx] = static_cast<T>(val);
        h_out[idx] = static_cast<T>(tanhf(val));
    }
}

// BF16-optimized: output = h * silu(z)
__global__ void MambaGateForward_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ h,
    const __nv_bfloat16* __restrict__ z,
    __nv_bfloat16* __restrict__ output) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        float h_val = __bfloat162float(h[idx]);
        float z_val = __bfloat162float(z[idx]);

        // silu(z) = z * sigmoid(z) - need f32 for exp
        float sigmoid_z = 1.0f / (1.0f + __expf(-z_val));
        float silu_z = z_val * sigmoid_z;

        output[idx] = __float2bfloat16(h_val * silu_z);
    }
}

// Generic version
template<typename T>
__global__ void MambaGateForward(
    const int batch_size,
    const int dim,
    const T* __restrict__ h,
    const T* __restrict__ z,
    T* __restrict__ output) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        float h_val = static_cast<float>(h[idx]);
        float z_val = static_cast<float>(z[idx]);

        float sigmoid_z = 1.0f / (1.0f + expf(-z_val));
        float silu_z = z_val * sigmoid_z;

        output[idx] = static_cast<T>(h_val * silu_z);
    }
}

// =============================================================================
// FUSED Forward Kernel: selective tanh + gate in one pass
// E17: Adds input-dependent gating on Rh (like Mamba2's selective A)
// =============================================================================

// E17 selective kernel: h = tanh(Wx + Rh * sigmoid(G) + b), out = h * silu(z)
__global__ void SelectiveTanhGateKernel_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ Wx,
    const __nv_bfloat16* __restrict__ Rh,
    const __nv_bfloat16* __restrict__ G,       // W_gate @ x + b_gate (pre-computed)
    const __nv_bfloat16* __restrict__ b,
    const __nv_bfloat16* __restrict__ z,
    __nv_bfloat16* __restrict__ h_out,
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ v_cache,
    __nv_bfloat16* __restrict__ gate_cache) {  // Cache gate for backward

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float wx = __bfloat162float(Wx[idx]);
        float rh = __bfloat162float(Rh[idx]);
        float g = __bfloat162float(G[idx]);
        float bias = __bfloat162float(b[d]);

        // Input-dependent gate on recurrence (like Mamba2's selective A)
        float gate = 1.0f / (1.0f + __expf(-g));

        // Cache gate for backward
        if (gate_cache) gate_cache[idx] = __float2bfloat16(gate);

        // Gated recurrence: Rh * gate
        float pre = wx + rh * gate + bias;

        // Cache pre-activation for backward
        if (v_cache) v_cache[idx] = __float2bfloat16(pre);

        // tanh
        float h_val = tanhf(pre);
        h_out[idx] = __float2bfloat16(h_val);

        // Output gate: h * silu(z)
        float z_val = __bfloat162float(z[idx]);
        float sigmoid_z = 1.0f / (1.0f + __expf(-z_val));
        float silu_z = z_val * sigmoid_z;

        output[idx] = __float2bfloat16(h_val * silu_z);
    }
}

// Original E1 kernel kept for reference/fallback
__global__ void FusedTanhGateKernel_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ Wx,
    const __nv_bfloat16* __restrict__ Rh,
    const __nv_bfloat16* __restrict__ b,
    const __nv_bfloat16* __restrict__ z,
    __nv_bfloat16* __restrict__ h_out,
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // Native bf16 additions
        __nv_bfloat16 sum = bf16_add(bf16_add(Wx[idx], Rh[idx]), b[d]);

        // Store pre-activation
        if (v_cache) v_cache[idx] = sum;

        // tanh (need f32)
        float val = __bfloat162float(sum);
        float h_val = tanhf(val);
        h_out[idx] = __float2bfloat16(h_val);

        // Gate (fused - no extra memory read for h)
        float z_val = __bfloat162float(z[idx]);
        float sigmoid_z = 1.0f / (1.0f + __expf(-z_val));
        float silu_z = z_val * sigmoid_z;

        output[idx] = __float2bfloat16(h_val * silu_z);
    }
}

// Generic fused version
template<typename T>
__global__ void FusedTanhGateKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ Wx,
    const T* __restrict__ Rh,
    const T* __restrict__ b,
    const T* __restrict__ z,
    T* __restrict__ h_out,
    T* __restrict__ output,
    T* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float val = static_cast<float>(Wx[idx]) + static_cast<float>(Rh[idx]) + static_cast<float>(b[d]);
        if (v_cache) v_cache[idx] = static_cast<T>(val);

        float h_val = tanhf(val);
        h_out[idx] = static_cast<T>(h_val);

        float z_val = static_cast<float>(z[idx]);
        float sigmoid_z = 1.0f / (1.0f + expf(-z_val));
        float silu_z = z_val * sigmoid_z;

        output[idx] = static_cast<T>(h_val * silu_z);
    }
}

// E17 generic: selective tanh gate kernel
template<typename T>
__global__ void SelectiveTanhGateKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ Wx,
    const T* __restrict__ Rh,
    const T* __restrict__ G,
    const T* __restrict__ b,
    const T* __restrict__ z,
    T* __restrict__ h_out,
    T* __restrict__ output,
    T* __restrict__ v_cache,
    T* __restrict__ gate_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float wx = static_cast<float>(Wx[idx]);
        float rh = static_cast<float>(Rh[idx]);
        float g = static_cast<float>(G[idx]);
        float bias = static_cast<float>(b[d]);

        // Input-dependent gate on recurrence
        float gate = 1.0f / (1.0f + expf(-g));

        if (gate_cache) gate_cache[idx] = static_cast<T>(gate);

        float pre = wx + rh * gate + bias;

        if (v_cache) v_cache[idx] = static_cast<T>(pre);

        float h_val = tanhf(pre);
        h_out[idx] = static_cast<T>(h_val);

        float z_val = static_cast<float>(z[idx]);
        float sigmoid_z = 1.0f / (1.0f + expf(-z_val));
        float silu_z = z_val * sigmoid_z;

        output[idx] = static_cast<T>(h_val * silu_z);
    }
}

// =============================================================================
// Optimized Backward Kernels
// =============================================================================

// E17: Backward through selective tanh - produces dv, dRh (scaled by gate), dG
__global__ void SelectiveTanhBackwardKernel_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ v,           // pre-activation
    const __nv_bfloat16* __restrict__ gate_cache,  // cached sigmoid(G)
    const __nv_bfloat16* __restrict__ Rh_cache,    // cached W_h @ h_prev
    const __nv_bfloat16* __restrict__ dh,          // gradient from output gate
    const __nv_bfloat16* __restrict__ dh_recurrent,// recurrent gradient
    __nv_bfloat16* __restrict__ dv,                // gradient through tanh (for W_x)
    __nv_bfloat16* __restrict__ dRh,               // gradient to Rh (for W_h backward)
    __nv_bfloat16* __restrict__ dG,                // gradient to G (for W_gate backward)
    float* __restrict__ db) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // Combine gradients
        float grad = __bfloat162float(dh[idx]);
        if (dh_recurrent) grad += __bfloat162float(dh_recurrent[idx]);

        // Backward through tanh
        float pre = __bfloat162float(v[idx]);
        float h = tanhf(pre);
        float dtanh = 1.0f - h * h;
        float dv_val = grad * dtanh;
        dv[idx] = __float2bfloat16(dv_val);

        // Backward through gated recurrence: pre = wx + rh * gate + b
        float gate = __bfloat162float(gate_cache[idx]);
        float rh = __bfloat162float(Rh_cache[idx]);

        // d_rh = dv * gate (for W_h backward)
        dRh[idx] = __float2bfloat16(dv_val * gate);

        // d_gate = dv * rh, then d_G = d_gate * gate * (1 - gate)
        float d_gate = dv_val * rh;
        float dg = d_gate * gate * (1.0f - gate);
        dG[idx] = __float2bfloat16(dg);

        // d_b = dv
        atomicAdd(&db[d], dv_val);
    }
}

// BF16-optimized backward through tanh (original E1)
__global__ void TanhBackwardKernel_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ dh,
    const __nv_bfloat16* __restrict__ dh_recurrent,
    __nv_bfloat16* __restrict__ dv,
    float* __restrict__ db) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // Combine gradients - use native bf16 add if available
        float grad;
        if (dh_recurrent) {
            __nv_bfloat16 combined = bf16_add(dh[idx], dh_recurrent[idx]);
            grad = __bfloat162float(combined);
        } else {
            grad = __bfloat162float(dh[idx]);
        }

        // dtanh: need f32 for tanh computation
        float h = tanhf(__bfloat162float(v[idx]));
        float dtanh = 1.0f - h * h;
        float dv_val = grad * dtanh;

        dv[idx] = __float2bfloat16(dv_val);
        atomicAdd(&db[d], dv_val);
    }
}

// Generic version
template<typename T>
__global__ void TanhBackwardKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ v,
    const T* __restrict__ dh,
    const T* __restrict__ dh_recurrent,
    T* __restrict__ dv,
    float* __restrict__ db) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float grad = static_cast<float>(dh[idx]);
        if (dh_recurrent) grad += static_cast<float>(dh_recurrent[idx]);

        float h = tanhf(static_cast<float>(v[idx]));
        float dtanh = 1.0f - h * h;
        float dv_val = grad * dtanh;
        dv[idx] = static_cast<T>(dv_val);

        atomicAdd(&db[d], dv_val);
    }
}

// E17 generic: selective tanh backward kernel
template<typename T>
__global__ void SelectiveTanhBackwardKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ v,
    const T* __restrict__ gate_cache,
    const T* __restrict__ Rh_cache,
    const T* __restrict__ dh,
    T* __restrict__ dv,
    T* __restrict__ dG,
    T* __restrict__ dRh,
    float* __restrict__ db) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float grad = static_cast<float>(dh[idx]);

        // Backward through tanh
        float pre = static_cast<float>(v[idx]);
        float h = tanhf(pre);
        float dtanh = 1.0f - h * h;
        float dv_val = grad * dtanh;
        dv[idx] = static_cast<T>(dv_val);

        // Backward through gated recurrence
        float gate = static_cast<float>(gate_cache[idx]);
        float rh = static_cast<float>(Rh_cache[idx]);

        // d_rh = dv * gate
        dRh[idx] = static_cast<T>(dv_val * gate);

        // d_gate = dv * rh, d_G = d_gate * gate * (1 - gate)
        float d_gate = dv_val * rh;
        float dg = d_gate * gate * (1.0f - gate);
        dG[idx] = static_cast<T>(dg);

        // d_b = dv
        atomicAdd(&db[d], dv_val);
    }
}

// BF16-optimized backward through mamba gate
__global__ void MambaGateBackward_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ h,
    const __nv_bfloat16* __restrict__ z,
    const __nv_bfloat16* __restrict__ d_output,
    __nv_bfloat16* __restrict__ dh,
    __nv_bfloat16* __restrict__ dz) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        float h_val = __bfloat162float(h[idx]);
        float z_val = __bfloat162float(z[idx]);
        float dout = __bfloat162float(d_output[idx]);

        float sigmoid_z = 1.0f / (1.0f + __expf(-z_val));
        float silu_z = z_val * sigmoid_z;
        float dsilu = sigmoid_z * (1.0f + z_val * (1.0f - sigmoid_z));

        dh[idx] = __float2bfloat16(dout * silu_z);
        dz[idx] = __float2bfloat16(dout * h_val * dsilu);
    }
}

// Generic version
template<typename T>
__global__ void MambaGateBackward(
    const int batch_size,
    const int dim,
    const T* __restrict__ h,
    const T* __restrict__ z,
    const T* __restrict__ d_output,
    T* __restrict__ dh,
    T* __restrict__ dz) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        float h_val = static_cast<float>(h[idx]);
        float z_val = static_cast<float>(z[idx]);
        float dout = static_cast<float>(d_output[idx]);

        float sigmoid_z = 1.0f / (1.0f + expf(-z_val));
        float silu_z = z_val * sigmoid_z;
        float dsilu = sigmoid_z * (1.0f + z_val * (1.0f - sigmoid_z));

        dh[idx] = static_cast<T>(dout * silu_z);
        dz[idx] = static_cast<T>(dout * h_val * dsilu);
    }
}

// =============================================================================
// Utility Kernels
// =============================================================================

// BF16-optimized vector add inplace
__global__ void VectorAddInplace_BF16(
    const int n,
    __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = bf16_add(a[idx], b[idx]);
    }
}

// Generic version
template<typename T>
__global__ void VectorAddInplace(const int n, T* __restrict__ a, const T* __restrict__ b) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = static_cast<T>(static_cast<float>(a[idx]) + static_cast<float>(b[idx]));
    }
}

// Copy float to T
template<typename T>
__global__ void CopyFloatToT(const int n, const float* __restrict__ src, T* __restrict__ dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = static_cast<T>(src[idx]);
    }
}

}  // anonymous namespace


namespace hasty {
namespace v0 {
namespace elman_ladder {

// =============================================================================
// E17: Selective W_h Elman Forward - BF16 Specialization
// =============================================================================

template<>
SelectiveWhElmanForward<__nv_bfloat16>::SelectiveWhElmanForward(
    bool training,
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : training_(training),
      batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void SelectiveWhElmanForward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_x,
    const __nv_bfloat16* W_h,
    const __nv_bfloat16* W_gate,   // NEW: gate projection
    const __nv_bfloat16* b,
    const __nv_bfloat16* b_gate,   // NEW: gate bias
    const __nv_bfloat16* x,
    const __nv_bfloat16* z,
    __nv_bfloat16* h,
    __nv_bfloat16* output,
    __nv_bfloat16* v,
    __nv_bfloat16* gate_cache,     // NEW: cache gates for backward
    __nv_bfloat16* Rh_cache,       // NEW: cache Rh for backward
    __nv_bfloat16* workspace) {

    static const __nv_bfloat16 alpha = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);
    static const __nv_bfloat16 beta_one = __float2bfloat16(1.0f);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Workspace: [Wx: T*BD] [G: T*BD] [Rh: BD]
    __nv_bfloat16* tmp_Wx = workspace;
    __nv_bfloat16* tmp_G = workspace + steps * BD;
    __nv_bfloat16* tmp_Rh = workspace + 2 * steps * BD;

    // Pre-compute W_x @ x for all timesteps
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha,
        W_x, dim_,
        x, dim_,
        &beta_zero,
        tmp_Wx, dim_);

    // Pre-compute W_gate @ x for all timesteps (+ b_gate handled in kernel)
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha,
        W_gate, dim_,
        x, dim_,
        &beta_zero,
        tmp_G, dim_);

    // Add b_gate to tmp_G - simple broadcast kernel
    auto add_bias_kernel = [&]() {
        // Inline kernel to add b_gate
        int total = steps * BD;
        int nblocks = (total + 255) / 256;
        // We'll fold b_gate into the main kernel instead
    };

    // Process each timestep with FUSED selective kernel
    for (int t = 0; t < steps; ++t) {
        const __nv_bfloat16* Wx_t = tmp_Wx + t * BD;
        __nv_bfloat16* G_t = tmp_G + t * BD;
        const __nv_bfloat16* h_prev = h + t * BD;
        const __nv_bfloat16* z_t = z + t * BD;
        __nv_bfloat16* h_t = h + (t + 1) * BD;
        __nv_bfloat16* out_t = output + t * BD;
        __nv_bfloat16* v_t = training_ ? (v + t * BD) : nullptr;
        __nv_bfloat16* gate_t = training_ ? (gate_cache + t * BD) : nullptr;
        __nv_bfloat16* Rh_t = training_ ? (Rh_cache + t * BD) : nullptr;

        // tmp_Rh = h_prev @ W_h.T
        blas<__nv_bfloat16>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha,
            W_h, dim_,
            h_prev, dim_,
            &beta_zero,
            tmp_Rh, dim_);

        // Cache Rh for backward
        if (Rh_t) {
            cudaMemcpyAsync(Rh_t, tmp_Rh, BD * sizeof(__nv_bfloat16),
                           cudaMemcpyDeviceToDevice, stream_);
        }

        // FUSED: gate = sigmoid(G + b_gate), h = tanh(Wx + Rh*gate + b), out = h*silu(z)
        // Note: We pass b_gate instead of folding into G for simplicity
        SelectiveTanhGateKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, Wx_t, tmp_Rh, G_t, b, z_t, h_t, out_t, v_t, gate_t);
    }
}

// =============================================================================
// E17: Selective W_h Elman Backward - BF16 Specialization
// =============================================================================

template<>
SelectiveWhElmanBackward<__nv_bfloat16>::SelectiveWhElmanBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void SelectiveWhElmanBackward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_x,
    const __nv_bfloat16* W_h,
    const __nv_bfloat16* W_gate,
    const __nv_bfloat16* x,
    const __nv_bfloat16* z,
    const __nv_bfloat16* h,
    const __nv_bfloat16* v,
    const __nv_bfloat16* gate_cache,
    const __nv_bfloat16* Rh_cache,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* dx,
    __nv_bfloat16* dz,
    __nv_bfloat16* dW_x,
    __nv_bfloat16* dW_h,
    __nv_bfloat16* dW_gate,
    __nv_bfloat16* db,
    __nv_bfloat16* workspace) {

    static const __nv_bfloat16 alpha_one = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);
    static const __nv_bfloat16 beta_one = __float2bfloat16(1.0f);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Workspace: [dv_all: T*BD] [dG_all: T*BD] [dRh_all: T*BD] [dh: BD] [dh_rec: BD] [db_float: dim]
    __nv_bfloat16* dv_all = workspace;
    __nv_bfloat16* dG_all = workspace + steps * BD;
    __nv_bfloat16* dRh_all = workspace + 2 * steps * BD;
    __nv_bfloat16* dh = workspace + 3 * steps * BD;
    __nv_bfloat16* dh_recurrent = dh + BD;
    float* db_float = reinterpret_cast<float*>(dh_recurrent + BD);

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW_x, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_h, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_gate, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);

    // BPTT loop
    for (int t = steps - 1; t >= 0; --t) {
        const __nv_bfloat16* v_t = v + t * BD;
        const __nv_bfloat16* gate_t = gate_cache + t * BD;
        const __nv_bfloat16* Rh_t = Rh_cache + t * BD;
        const __nv_bfloat16* h_t = h + (t + 1) * BD;
        const __nv_bfloat16* z_t = z + t * BD;
        const __nv_bfloat16* d_out_t = d_output + t * BD;
        __nv_bfloat16* dv_t = dv_all + t * BD;
        __nv_bfloat16* dG_t = dG_all + t * BD;
        __nv_bfloat16* dRh_t = dRh_all + t * BD;
        __nv_bfloat16* dz_t = dz + t * BD;

        // Backward through output gate
        MambaGateBackward_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, z_t, d_out_t, dh, dz_t);

        // Backward through selective tanh (produces dv, dRh, dG)
        SelectiveTanhBackwardKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, v_t, gate_t, Rh_t, dh,
            (t < steps - 1) ? dh_recurrent : nullptr,
            dv_t, dRh_t, dG_t, db_float);

        // Recurrent gradient: dh_recurrent = W_h @ dRh (note: dRh already scaled by gate)
        if (t > 0) {
            blas<__nv_bfloat16>::gemm(
                blas_handle_,
                CUBLAS_OP_N, CUBLAS_OP_N,
                dim_, batch_size_, dim_,
                &alpha_one,
                W_h, dim_,
                dRh_t, dim_,
                &beta_zero,
                dh_recurrent, dim_);
        }
    }

    // Batch GEMMs for weight gradients
    // dx from W_x
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_x, dim_,
        dv_all, dim_,
        &beta_zero,
        dx, dim_);

    // dx also gets contribution from W_gate @ dG
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_gate, dim_,
        dG_all, dim_,
        &beta_one,  // accumulate
        dx, dim_);

    // dW_x = x.T @ dv
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        dv_all, dim_,
        &beta_one,
        dW_x, dim_);

    // dW_h = h[:-1].T @ dRh (note: dRh is already gate-scaled)
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        h, dim_,
        dRh_all, dim_,
        &beta_one,
        dW_h, dim_);

    // dW_gate = x.T @ dG
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        dG_all, dim_,
        &beta_one,
        dW_gate, dim_);

    // Copy float gradients to bf16
    CopyFloatToT<__nv_bfloat16><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_float, db);
}

// Explicit instantiations for E17
template struct SelectiveWhElmanForward<__nv_bfloat16>;
template struct SelectiveWhElmanBackward<__nv_bfloat16>;

// =============================================================================
// Generic Template Implementations for E17 (half, float, double)
// =============================================================================

template<typename T>
SelectiveWhElmanForward<T>::SelectiveWhElmanForward(
    bool training,
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : training_(training),
      batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void SelectiveWhElmanForward<T>::Run(
    int steps,
    const T* W_x,
    const T* W_h,
    const T* W_gate,
    const T* b,
    const T* b_gate,
    const T* x,
    const T* z,
    T* h,
    T* output,
    T* v,
    T* gate_cache,
    T* Rh_cache,
    T* workspace) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    T* tmp_Wx = workspace;
    T* tmp_G = workspace + steps * BD;
    T* tmp_Rh = workspace + 2 * steps * BD;

    // Pre-compute W_x @ x and W_gate @ x for all timesteps
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha,
        W_x, dim_,
        x, dim_,
        &beta_zero,
        tmp_Wx, dim_);

    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha,
        W_gate, dim_,
        x, dim_,
        &beta_zero,
        tmp_G, dim_);

    for (int t = 0; t < steps; ++t) {
        const T* Wx_t = tmp_Wx + t * BD;
        const T* G_t = tmp_G + t * BD;
        const T* h_prev = h + t * BD;
        const T* z_t = z + t * BD;
        T* h_t = h + (t + 1) * BD;
        T* out_t = output + t * BD;
        T* v_t = training_ ? (v + t * BD) : nullptr;
        T* gate_t = training_ ? (gate_cache + t * BD) : nullptr;
        T* Rh_t = training_ ? (Rh_cache + t * BD) : nullptr;

        // tmp_Rh = h_prev @ W_h.T
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha,
            W_h, dim_,
            h_prev, dim_,
            &beta_zero,
            tmp_Rh, dim_);

        // FUSED kernel with selective gating
        SelectiveTanhGateKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, Wx_t, tmp_Rh, G_t, b, z_t, h_t, out_t, v_t, gate_t);

        if (Rh_t) {
            cudaMemcpyAsync(Rh_t, tmp_Rh, BD * sizeof(T), cudaMemcpyDeviceToDevice, stream_);
        }
    }
}

template<typename T>
SelectiveWhElmanBackward<T>::SelectiveWhElmanBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void SelectiveWhElmanBackward<T>::Run(
    int steps,
    const T* W_x,
    const T* W_h,
    const T* W_gate,
    const T* x,
    const T* z,
    const T* h,
    const T* v,
    const T* gate_cache,
    const T* Rh_cache,
    const T* d_output,
    T* dx,
    T* dz,
    T* dW_x,
    T* dW_h,
    T* dW_gate,
    T* db,
    T* workspace) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    T* dv_all = workspace;
    T* dG_all = workspace + steps * BD;
    T* dRh_all = workspace + 2 * steps * BD;
    T* dh = workspace + 3 * steps * BD;
    T* dh_recurrent = workspace + (3 * steps + 1) * BD;
    float* db_float = reinterpret_cast<float*>(workspace + (3 * steps + 2) * BD);

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(T), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW_x, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dW_h, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dW_gate, 0, dim_ * dim_ * sizeof(T), stream_);

    // BPTT loop
    for (int t = steps - 1; t >= 0; --t) {
        const T* v_t = v + t * BD;
        const T* h_t = h + (t + 1) * BD;
        const T* z_t = z + t * BD;
        const T* gate_t = gate_cache + t * BD;
        const T* Rh_t = Rh_cache + t * BD;
        const T* d_out_t = d_output + t * BD;
        T* dv_t = dv_all + t * BD;
        T* dG_t = dG_all + t * BD;
        T* dRh_t = dRh_all + t * BD;
        T* dz_t = dz + t * BD;

        // Backward through gate
        MambaGateBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, z_t, d_out_t, dh, dz_t);

        VectorAddInplace<T><<<num_blocks, block_size, 0, stream_>>>(BD, dh, dh_recurrent);

        // Backward through selective tanh with gating
        SelectiveTanhBackwardKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, v_t, gate_t, Rh_t, dh, dv_t, dG_t, dRh_t, db_float);

        // dh_recurrent = W_h @ dRh
        if (t > 0) {
            blas<T>::gemm(
                blas_handle_,
                CUBLAS_OP_N, CUBLAS_OP_N,
                dim_, batch_size_, dim_,
                &alpha,
                W_h, dim_,
                dRh_t, dim_,
                &beta_zero,
                dh_recurrent, dim_);
        }
    }

    // Batch GEMMs for dx, dW_x, dW_h, dW_gate
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha,
        W_x, dim_,
        dv_all, dim_,
        &beta_zero,
        dx, dim_);

    // Add contribution from gate backward
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha,
        W_gate, dim_,
        dG_all, dim_,
        &beta_one,
        dx, dim_);

    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha,
        x, dim_,
        dv_all, dim_,
        &beta_one,
        dW_x, dim_);

    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha,
        h, dim_,
        dRh_all, dim_,
        &beta_one,
        dW_h, dim_);

    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha,
        x, dim_,
        dG_all, dim_,
        &beta_one,
        dW_gate, dim_);

    CopyFloatToT<T><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_float, db);
}

// Explicit template instantiations for generic types
template struct SelectiveWhElmanForward<__half>;
template struct SelectiveWhElmanForward<float>;
template struct SelectiveWhElmanForward<double>;

template struct SelectiveWhElmanBackward<__half>;
template struct SelectiveWhElmanBackward<float>;
template struct SelectiveWhElmanBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
