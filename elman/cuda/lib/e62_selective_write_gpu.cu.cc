// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E62: Selective Write Elman
//
// Vector analog of DeltaNet's selective memory updates.
//
// Architecture:
// k_t = sigmoid(W_k @ x_t + b_k)     # Selection mask (0-1 per dimension)
// v_t = tanh(W_v @ x_t + b_v)        # New values
// h_t = (1 - k_t) * h_{t-1} + k_t * v_t   # Selective replacement
// output_t = h_t * silu(h_t)         # Self-gating
//
// This is LINEAR in h - potentially parallelizable with associative scan.
// Scan form: A_t = (1 - k_t), B_t = k_t * v_t
//
// Jacobian: dh_t/dh_{t-1} = diag(1 - k_t)
// When k -> 0: preserve h (gradient = 1)
// When k -> 1: overwrite with v (gradient = 0, intentional forgetting)
//
// Key optimization: Batch compute W_k @ x and W_v @ x for ALL timesteps upfront.

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

__device__ __forceinline__ __nv_bfloat16 bf16_add(__nv_bfloat16 a, __nv_bfloat16 b) {
#if __CUDA_ARCH__ >= 800
    return __hadd(a, b);
#else
    return __float2bfloat16(__bfloat162float(a) + __bfloat162float(b));
#endif
}

__device__ __forceinline__ __nv_bfloat16 bf16_mul(__nv_bfloat16 a, __nv_bfloat16 b) {
#if __CUDA_ARCH__ >= 800
    return __hmul(a, b);
#else
    return __float2bfloat16(__bfloat162float(a) * __bfloat162float(b));
#endif
}

__device__ __forceinline__ __nv_bfloat16 bf16_sub(__nv_bfloat16 a, __nv_bfloat16 b) {
#if __CUDA_ARCH__ >= 800
    return __hsub(a, b);
#else
    return __float2bfloat16(__bfloat162float(a) - __bfloat162float(b));
#endif
}

// =============================================================================
// Forward Kernels
// =============================================================================

// Fused kernel: compute k = sigmoid(Wk_x + b_k), v = tanh(Wv_x + b_v)
// Then h_new = (1 - k) * h_prev + k * v, output = h_new * silu(h_new)
__global__ void E62SelectiveWriteKernel_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ Wk_x,   // Pre-computed W_k @ x_t
    const __nv_bfloat16* __restrict__ Wv_x,   // Pre-computed W_v @ x_t
    const __nv_bfloat16* __restrict__ b_k,
    const __nv_bfloat16* __restrict__ b_v,
    const __nv_bfloat16* __restrict__ h_prev,
    __nv_bfloat16* __restrict__ h_out,
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ k_cache,   // Save k for backward
    __nv_bfloat16* __restrict__ v_cache) { // Save v for backward

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // k = sigmoid(Wk_x + b_k)
        float k_pre = __bfloat162float(bf16_add(Wk_x[idx], b_k[d]));
        float k_val = 1.0f / (1.0f + __expf(-k_pre));

        // v = tanh(Wv_x + b_v)
        float v_pre = __bfloat162float(bf16_add(Wv_x[idx], b_v[d]));
        float v_val = tanhf(v_pre);

        // Save k and v for backward
        if (k_cache) k_cache[idx] = __float2bfloat16(k_val);
        if (v_cache) v_cache[idx] = __float2bfloat16(v_val);

        // h_new = (1 - k) * h_prev + k * v
        float h_prev_f = __bfloat162float(h_prev[idx]);
        float h_new_f = (1.0f - k_val) * h_prev_f + k_val * v_val;
        h_out[idx] = __float2bfloat16(h_new_f);

        // Self-gating output: h_new * silu(h_new)
        float sigmoid_h = 1.0f / (1.0f + __expf(-h_new_f));
        float silu_h = h_new_f * sigmoid_h;
        output[idx] = __float2bfloat16(h_new_f * silu_h);
    }
}

template<typename T>
__global__ void E62SelectiveWriteKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ Wk_x,
    const T* __restrict__ Wv_x,
    const T* __restrict__ b_k,
    const T* __restrict__ b_v,
    const T* __restrict__ h_prev,
    T* __restrict__ h_out,
    T* __restrict__ output,
    T* __restrict__ k_cache,
    T* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // k = sigmoid(Wk_x + b_k)
        float k_pre = static_cast<float>(Wk_x[idx]) + static_cast<float>(b_k[d]);
        float k_val = 1.0f / (1.0f + expf(-k_pre));

        // v = tanh(Wv_x + b_v)
        float v_pre = static_cast<float>(Wv_x[idx]) + static_cast<float>(b_v[d]);
        float v_val = tanhf(v_pre);

        // Save k and v for backward
        if (k_cache) k_cache[idx] = static_cast<T>(k_val);
        if (v_cache) v_cache[idx] = static_cast<T>(v_val);

        // h_new = (1 - k) * h_prev + k * v
        float h_prev_f = static_cast<float>(h_prev[idx]);
        float h_new_f = (1.0f - k_val) * h_prev_f + k_val * v_val;
        h_out[idx] = static_cast<T>(h_new_f);

        // Self-gating output
        float sigmoid_h = 1.0f / (1.0f + expf(-h_new_f));
        float silu_h = h_new_f * sigmoid_h;
        output[idx] = static_cast<T>(h_new_f * silu_h);
    }
}

// =============================================================================
// Backward Kernels
// =============================================================================

// Backward through self-gate: output = h * silu(h) = h^2 * sigmoid(h)
// d(output)/dh = silu(h) * (2 + h * (1 - sigmoid(h)))
__global__ void E62SelfGateBackwardKernel_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ h,
    const __nv_bfloat16* __restrict__ d_output,
    __nv_bfloat16* __restrict__ dh) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        float h_val = __bfloat162float(h[idx]);
        float dout = __bfloat162float(d_output[idx]);

        float sigmoid_h = 1.0f / (1.0f + __expf(-h_val));
        float silu_h = h_val * sigmoid_h;
        float grad_factor = silu_h * (2.0f + h_val * (1.0f - sigmoid_h));

        dh[idx] = __float2bfloat16(dout * grad_factor);
    }
}

template<typename T>
__global__ void E62SelfGateBackwardKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ h,
    const T* __restrict__ d_output,
    T* __restrict__ dh) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        float h_val = static_cast<float>(h[idx]);
        float dout = static_cast<float>(d_output[idx]);

        float sigmoid_h = 1.0f / (1.0f + expf(-h_val));
        float silu_h = h_val * sigmoid_h;
        float grad_factor = silu_h * (2.0f + h_val * (1.0f - sigmoid_h));

        dh[idx] = static_cast<T>(dout * grad_factor);
    }
}

// Backward through selective write: h_new = (1 - k) * h_prev + k * v
// dh_prev = dh * (1 - k)
// dk_total = dh * (-h_prev + v) = dh * (v - h_prev)
// dk_pre = dk_total * k * (1 - k)  [sigmoid derivative]
// dv_total = dh * k
// dv_pre = dv_total * (1 - v^2)    [tanh derivative]
__global__ void E62SelectiveWriteBackwardKernel_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ h_prev,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const __nv_bfloat16* __restrict__ dh,
    const __nv_bfloat16* __restrict__ dh_recurrent,
    __nv_bfloat16* __restrict__ dh_prev_out,
    __nv_bfloat16* __restrict__ dk_pre,   // Gradient for W_k @ x
    __nv_bfloat16* __restrict__ dv_pre,   // Gradient for W_v @ x
    float* __restrict__ db_k,
    float* __restrict__ db_v) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float h_prev_f = __bfloat162float(h_prev[idx]);
        float k_val = __bfloat162float(k_cache[idx]);
        float v_val = __bfloat162float(v_cache[idx]);

        // Combine output gradient with recurrent gradient
        float dh_total;
        if (dh_recurrent) {
            dh_total = __bfloat162float(dh[idx]) + __bfloat162float(dh_recurrent[idx]);
        } else {
            dh_total = __bfloat162float(dh[idx]);
        }

        // dh_prev = dh_total * (1 - k)
        float dh_prev_val = dh_total * (1.0f - k_val);
        dh_prev_out[idx] = __float2bfloat16(dh_prev_val);

        // dk_total = dh_total * (v - h_prev)
        float dk_total = dh_total * (v_val - h_prev_f);
        // dk_pre = dk_total * sigmoid'(k_pre) = dk_total * k * (1 - k)
        float dk_pre_val = dk_total * k_val * (1.0f - k_val);
        dk_pre[idx] = __float2bfloat16(dk_pre_val);

        // dv_total = dh_total * k
        float dv_total = dh_total * k_val;
        // dv_pre = dv_total * tanh'(v_pre) = dv_total * (1 - v^2)
        float dv_pre_val = dv_total * (1.0f - v_val * v_val);
        dv_pre[idx] = __float2bfloat16(dv_pre_val);

        // Accumulate bias gradients
        atomicAdd(&db_k[d], dk_pre_val);
        atomicAdd(&db_v[d], dv_pre_val);
    }
}

template<typename T>
__global__ void E62SelectiveWriteBackwardKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ h_prev,
    const T* __restrict__ k_cache,
    const T* __restrict__ v_cache,
    const T* __restrict__ dh,
    const T* __restrict__ dh_recurrent,
    T* __restrict__ dh_prev_out,
    T* __restrict__ dk_pre,
    T* __restrict__ dv_pre,
    float* __restrict__ db_k,
    float* __restrict__ db_v) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float h_prev_f = static_cast<float>(h_prev[idx]);
        float k_val = static_cast<float>(k_cache[idx]);
        float v_val = static_cast<float>(v_cache[idx]);

        // Combine output gradient with recurrent gradient
        float dh_total = static_cast<float>(dh[idx]);
        if (dh_recurrent) {
            dh_total += static_cast<float>(dh_recurrent[idx]);
        }

        // dh_prev = dh_total * (1 - k)
        float dh_prev_val = dh_total * (1.0f - k_val);
        dh_prev_out[idx] = static_cast<T>(dh_prev_val);

        // dk_total = dh_total * (v - h_prev)
        float dk_total = dh_total * (v_val - h_prev_f);
        float dk_pre_val = dk_total * k_val * (1.0f - k_val);
        dk_pre[idx] = static_cast<T>(dk_pre_val);

        // dv_total = dh_total * k
        float dv_total = dh_total * k_val;
        float dv_pre_val = dv_total * (1.0f - v_val * v_val);
        dv_pre[idx] = static_cast<T>(dv_pre_val);

        // Accumulate bias gradients
        atomicAdd(&db_k[d], dk_pre_val);
        atomicAdd(&db_v[d], dv_pre_val);
    }
}

// =============================================================================
// Utility Kernels
// =============================================================================

__global__ void VectorAddInplace_E62_BF16(
    const int n,
    __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = bf16_add(a[idx], b[idx]);
    }
}

template<typename T>
__global__ void VectorAddInplace_E62(const int n, T* __restrict__ a, const T* __restrict__ b) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = static_cast<T>(static_cast<float>(a[idx]) + static_cast<float>(b[idx]));
    }
}

template<typename T>
__global__ void CopyFloatToT_E62(const int n, const float* __restrict__ src, T* __restrict__ dst) {
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
// E62 Selective Write Forward - BF16 Specialization
// =============================================================================

template<>
E62SelectiveWriteForward<__nv_bfloat16>::E62SelectiveWriteForward(
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
void E62SelectiveWriteForward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_k,       // [dim, dim]
    const __nv_bfloat16* b_k,       // [dim]
    const __nv_bfloat16* W_v,       // [dim, dim]
    const __nv_bfloat16* b_v,       // [dim]
    const __nv_bfloat16* x,         // [T, B, dim]
    __nv_bfloat16* h,               // [T+1, B, dim] (output)
    __nv_bfloat16* output,          // [T, B, dim] (output)
    __nv_bfloat16* k_cache,         // [T, B, dim] (for backward)
    __nv_bfloat16* v_cache,         // [T, B, dim] (for backward)
    __nv_bfloat16* workspace) {

    static const __nv_bfloat16 alpha = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Workspace layout: [tmp_Wk_x: T*BD] [tmp_Wv_x: T*BD]
    __nv_bfloat16* tmp_Wk_x = workspace;
    __nv_bfloat16* tmp_Wv_x = workspace + steps * BD;

    // =========================================================================
    // KEY OPTIMIZATION: Pre-compute W_k @ x and W_v @ x for ALL timesteps
    // =========================================================================
    // W_k @ x for all timesteps
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha,
        W_k, dim_,
        x, dim_,
        &beta_zero,
        tmp_Wk_x, dim_);

    // W_v @ x for all timesteps
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha,
        W_v, dim_,
        x, dim_,
        &beta_zero,
        tmp_Wv_x, dim_);

    // Process each timestep with simple sequential recurrence
    for (int t = 0; t < steps; ++t) {
        const __nv_bfloat16* Wk_x_t = tmp_Wk_x + t * BD;
        const __nv_bfloat16* Wv_x_t = tmp_Wv_x + t * BD;
        const __nv_bfloat16* h_prev = h + t * BD;
        __nv_bfloat16* h_t = h + (t + 1) * BD;
        __nv_bfloat16* out_t = output + t * BD;
        __nv_bfloat16* k_t = training_ ? (k_cache + t * BD) : nullptr;
        __nv_bfloat16* v_t = training_ ? (v_cache + t * BD) : nullptr;

        E62SelectiveWriteKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, Wk_x_t, Wv_x_t, b_k, b_v, h_prev, h_t, out_t, k_t, v_t);
    }
}

// =============================================================================
// E62 Selective Write Backward - BF16 Specialization
// =============================================================================

template<>
E62SelectiveWriteBackward<__nv_bfloat16>::E62SelectiveWriteBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void E62SelectiveWriteBackward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_k,
    const __nv_bfloat16* W_v,
    const __nv_bfloat16* x,
    const __nv_bfloat16* h,
    const __nv_bfloat16* k_cache,
    const __nv_bfloat16* v_cache,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* dx,
    __nv_bfloat16* dW_k,
    __nv_bfloat16* db_k,
    __nv_bfloat16* dW_v,
    __nv_bfloat16* db_v,
    __nv_bfloat16* workspace) {

    static const __nv_bfloat16 alpha_one = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);
    static const __nv_bfloat16 beta_one = __float2bfloat16(1.0f);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Workspace layout:
    // [T*BD] dk_pre_all - gradients for W_k @ x
    // [T*BD] dv_pre_all - gradients for W_v @ x
    // [BD]   dh - gradient through self-gate
    // [BD]   dh_recurrent - accumulated recurrent gradient
    // [dim]  db_k_float
    // [dim]  db_v_float
    __nv_bfloat16* dk_pre_all = workspace;
    __nv_bfloat16* dv_pre_all = workspace + steps * BD;
    __nv_bfloat16* dh = workspace + 2 * steps * BD;
    __nv_bfloat16* dh_recurrent = workspace + (2 * steps + 1) * BD;
    float* db_k_float = reinterpret_cast<float*>(workspace + (2 * steps + 2) * BD);
    float* db_v_float = db_k_float + dim_;

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(db_k_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_v_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW_k, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_v, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);

    // BPTT loop
    for (int t = steps - 1; t >= 0; --t) {
        const __nv_bfloat16* h_t = h + (t + 1) * BD;      // h at timestep t+1
        const __nv_bfloat16* h_prev = h + t * BD;        // h at timestep t
        const __nv_bfloat16* k_t = k_cache + t * BD;
        const __nv_bfloat16* v_t = v_cache + t * BD;
        const __nv_bfloat16* d_out_t = d_output + t * BD;
        __nv_bfloat16* dk_pre_t = dk_pre_all + t * BD;
        __nv_bfloat16* dv_pre_t = dv_pre_all + t * BD;

        // Step 1: Backward through self-gate
        E62SelfGateBackwardKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, d_out_t, dh);

        // Step 2: Backward through selective write
        // Output: dh_recurrent (for next iteration), dk_pre_t, dv_pre_t
        __nv_bfloat16* dh_recurrent_use = (t == steps - 1) ? nullptr : dh_recurrent;
        __nv_bfloat16* dh_prev_out = (t > 0) ? dh_recurrent : dh;  // Reuse dh if t==0

        E62SelectiveWriteBackwardKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_prev, k_t, v_t, dh, dh_recurrent_use,
            dh_prev_out, dk_pre_t, dv_pre_t, db_k_float, db_v_float);
    }

    // =========================================================================
    // Batch GEMMs for weight and input gradients
    // =========================================================================

    // dx = W_k @ dk_pre_all + W_v @ dv_pre_all
    // First: dx = W_k @ dk_pre_all
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_k, dim_,
        dk_pre_all, dim_,
        &beta_zero,
        dx, dim_);

    // Then: dx += W_v @ dv_pre_all
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_v, dim_,
        dv_pre_all, dim_,
        &beta_one,
        dx, dim_);

    // dW_k = x @ dk_pre_all^T
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        dk_pre_all, dim_,
        &beta_one,
        dW_k, dim_);

    // dW_v = x @ dv_pre_all^T
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        dv_pre_all, dim_,
        &beta_one,
        dW_v, dim_);

    // Copy float gradients to output types
    CopyFloatToT_E62<__nv_bfloat16><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_k_float, db_k);
    CopyFloatToT_E62<__nv_bfloat16><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_v_float, db_v);
}

// =============================================================================
// Generic Template Implementations
// =============================================================================

template<typename T>
E62SelectiveWriteForward<T>::E62SelectiveWriteForward(
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
void E62SelectiveWriteForward<T>::Run(
    int steps,
    const T* W_k,
    const T* b_k,
    const T* W_v,
    const T* b_v,
    const T* x,
    T* h,
    T* output,
    T* k_cache,
    T* v_cache,
    T* workspace) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    T* tmp_Wk_x = workspace;
    T* tmp_Wv_x = workspace + steps * BD;

    // Pre-compute W_k @ x and W_v @ x for all timesteps
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha,
        W_k, dim_,
        x, dim_,
        &beta_zero,
        tmp_Wk_x, dim_);

    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha,
        W_v, dim_,
        x, dim_,
        &beta_zero,
        tmp_Wv_x, dim_);

    for (int t = 0; t < steps; ++t) {
        const T* Wk_x_t = tmp_Wk_x + t * BD;
        const T* Wv_x_t = tmp_Wv_x + t * BD;
        const T* h_prev = h + t * BD;
        T* h_t = h + (t + 1) * BD;
        T* out_t = output + t * BD;
        T* k_t = training_ ? (k_cache + t * BD) : nullptr;
        T* v_t = training_ ? (v_cache + t * BD) : nullptr;

        E62SelectiveWriteKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, Wk_x_t, Wv_x_t, b_k, b_v, h_prev, h_t, out_t, k_t, v_t);
    }
}

template<typename T>
E62SelectiveWriteBackward<T>::E62SelectiveWriteBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void E62SelectiveWriteBackward<T>::Run(
    int steps,
    const T* W_k,
    const T* W_v,
    const T* x,
    const T* h,
    const T* k_cache,
    const T* v_cache,
    const T* d_output,
    T* dx,
    T* dW_k,
    T* db_k,
    T* dW_v,
    T* db_v,
    T* workspace) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    T* dk_pre_all = workspace;
    T* dv_pre_all = workspace + steps * BD;
    T* dh = workspace + 2 * steps * BD;
    T* dh_recurrent = workspace + (2 * steps + 1) * BD;
    float* db_k_float = reinterpret_cast<float*>(workspace + (2 * steps + 2) * BD);
    float* db_v_float = db_k_float + dim_;

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(T), stream_);
    cudaMemsetAsync(db_k_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_v_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW_k, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dW_v, 0, dim_ * dim_ * sizeof(T), stream_);

    for (int t = steps - 1; t >= 0; --t) {
        const T* h_t = h + (t + 1) * BD;
        const T* h_prev = h + t * BD;
        const T* k_t = k_cache + t * BD;
        const T* v_t = v_cache + t * BD;
        const T* d_out_t = d_output + t * BD;
        T* dk_pre_t = dk_pre_all + t * BD;
        T* dv_pre_t = dv_pre_all + t * BD;

        E62SelfGateBackwardKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, d_out_t, dh);

        T* dh_recurrent_use = (t == steps - 1) ? nullptr : dh_recurrent;
        T* dh_prev_out = (t > 0) ? dh_recurrent : dh;

        E62SelectiveWriteBackwardKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_prev, k_t, v_t, dh, dh_recurrent_use,
            dh_prev_out, dk_pre_t, dv_pre_t, db_k_float, db_v_float);
    }

    // dx = W_k @ dk_pre_all + W_v @ dv_pre_all
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha,
        W_k, dim_,
        dk_pre_all, dim_,
        &beta_zero,
        dx, dim_);

    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha,
        W_v, dim_,
        dv_pre_all, dim_,
        &beta_one,
        dx, dim_);

    // dW_k = x @ dk_pre_all^T
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha,
        x, dim_,
        dk_pre_all, dim_,
        &beta_one,
        dW_k, dim_);

    // dW_v = x @ dv_pre_all^T
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha,
        x, dim_,
        dv_pre_all, dim_,
        &beta_one,
        dW_v, dim_);

    CopyFloatToT_E62<T><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_k_float, db_k);
    CopyFloatToT_E62<T><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_v_float, db_v);
}

// Explicit template instantiations
template struct E62SelectiveWriteForward<__half>;
template struct E62SelectiveWriteForward<float>;
template struct E62SelectiveWriteForward<double>;

template struct E62SelectiveWriteBackward<__half>;
template struct E62SelectiveWriteBackward<float>;
template struct E62SelectiveWriteBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
