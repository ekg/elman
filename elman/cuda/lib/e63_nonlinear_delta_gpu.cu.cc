// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E63: Nonlinear Delta Elman (UTM-Class Expressivity)
//
// The key insight: E63 adds nonlinear h-dependence while preserving gated gradient control.
// This makes E63 Turing complete while E61/E62 are not.
//
// E63a (Complementary) Architecture:
// alpha_t = sigmoid(W_alpha @ x_t + b_alpha)           # Retain gate (x-only)
// v_t = tanh(W_h @ h_{t-1} + W_x @ x_t + b)           # NONLINEAR value (h-dependent!)
// h_t = alpha_t * h_{t-1} + (1 - alpha_t) * v_t       # Gated mixing (linear for gradient flow)
// output = h * silu(h)                                 # Self-gating
//
// Key optimization:
// - Batch x @ W_alpha.T and x @ W_x.T for ALL timesteps upfront (single GEMM each)
// - Per-timestep h @ W_h.T GEMM (cannot batch due to h-dependence)
// - Fused kernel for sigmoid, tanh, element-wise ops, and self-gate

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

// =============================================================================
// Forward Kernel: Fused gate + value + update + self-gate
// Input: alpha_x = W_alpha @ x_t (pre-batched), Wx = W_x @ x_t (pre-batched),
//        Wh = W_h @ h_prev (per-step)
// Output: h_new, output, and optionally save v_pre (for backward)
// =============================================================================

__global__ void E63FusedForwardKernel_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ alpha_x,    // [B, D] pre-computed W_alpha @ x + b_alpha
    const __nv_bfloat16* __restrict__ Wx,         // [B, D] pre-computed W_x @ x
    const __nv_bfloat16* __restrict__ Wh,         // [B, D] per-step W_h @ h_prev
    const __nv_bfloat16* __restrict__ h_prev,     // [B, D] previous hidden state
    const __nv_bfloat16* __restrict__ b,          // [D] bias for value
    __nv_bfloat16* __restrict__ h_out,            // [B, D] new hidden state
    __nv_bfloat16* __restrict__ output,           // [B, D] output (h * silu(h))
    __nv_bfloat16* __restrict__ v_pre_cache,      // [B, D] pre-tanh value cache (for backward)
    __nv_bfloat16* __restrict__ alpha_cache) {    // [B, D] alpha cache (for backward)

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // Alpha gate: sigmoid(alpha_x)
        float alpha_val = __bfloat162float(alpha_x[idx]);
        float alpha = 1.0f / (1.0f + __expf(-alpha_val));

        // Value pre-activation: Wh + Wx + b
        float v_pre = __bfloat162float(Wh[idx]) + __bfloat162float(Wx[idx]) + __bfloat162float(b[d]);

        // Nonlinear value: tanh(v_pre)
        float v = tanhf(v_pre);

        // Gated update: h_new = alpha * h_prev + (1 - alpha) * v
        float h_prev_val = __bfloat162float(h_prev[idx]);
        float h_new = alpha * h_prev_val + (1.0f - alpha) * v;

        // Store hidden state
        h_out[idx] = __float2bfloat16(h_new);

        // Self-gate output: h * silu(h) = h^2 * sigmoid(h)
        float sigmoid_h = 1.0f / (1.0f + __expf(-h_new));
        float silu_h = h_new * sigmoid_h;
        output[idx] = __float2bfloat16(h_new * silu_h);

        // Cache for backward (training only)
        if (v_pre_cache) v_pre_cache[idx] = __float2bfloat16(v_pre);
        if (alpha_cache) alpha_cache[idx] = __float2bfloat16(alpha);
    }
}

template<typename T>
__global__ void E63FusedForwardKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ alpha_x,
    const T* __restrict__ Wx,
    const T* __restrict__ Wh,
    const T* __restrict__ h_prev,
    const T* __restrict__ b,
    T* __restrict__ h_out,
    T* __restrict__ output,
    T* __restrict__ v_pre_cache,
    T* __restrict__ alpha_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // Alpha gate
        float alpha_val = static_cast<float>(alpha_x[idx]);
        float alpha = 1.0f / (1.0f + expf(-alpha_val));

        // Value pre-activation
        float v_pre = static_cast<float>(Wh[idx]) + static_cast<float>(Wx[idx]) + static_cast<float>(b[d]);

        // Nonlinear value
        float v = tanhf(v_pre);

        // Gated update
        float h_prev_val = static_cast<float>(h_prev[idx]);
        float h_new = alpha * h_prev_val + (1.0f - alpha) * v;

        h_out[idx] = static_cast<T>(h_new);

        // Self-gate
        float sigmoid_h = 1.0f / (1.0f + expf(-h_new));
        float silu_h = h_new * sigmoid_h;
        output[idx] = static_cast<T>(h_new * silu_h);

        if (v_pre_cache) v_pre_cache[idx] = static_cast<T>(v_pre);
        if (alpha_cache) alpha_cache[idx] = static_cast<T>(alpha);
    }
}

// =============================================================================
// Backward Kernels
// =============================================================================

// Phase 1: Backward through self-gate: d_output -> dh_post_gate
// output = h * silu(h) = h^2 * sigmoid(h)
// d(output)/dh = silu(h) * (2 + h*(1-sigmoid(h)))
__global__ void E63SelfGateBackward_BF16(
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
__global__ void E63SelfGateBackward(
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

// Phase 2: Backward through gated update and value computation
// h_t = alpha * h_prev + (1 - alpha) * tanh(v_pre)
// Computes: dh_prev (for recurrence), dv (for weight grads), dalpha (for alpha weight grads)
__global__ void E63GatedUpdateBackward_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ h_prev,        // [B, D] h_{t-1}
    const __nv_bfloat16* __restrict__ v_pre,         // [B, D] pre-tanh value
    const __nv_bfloat16* __restrict__ alpha_cache,   // [B, D] sigmoid(alpha_x)
    const __nv_bfloat16* __restrict__ alpha_x_all_t, // [B, D] alpha_x (pre-sigmoid)
    const __nv_bfloat16* __restrict__ dh,            // [B, D] gradient from output (dh_t)
    const __nv_bfloat16* __restrict__ dh_recurrent,  // [B, D] gradient from next timestep (or null)
    __nv_bfloat16* __restrict__ dh_prev,             // [B, D] gradient to previous h
    __nv_bfloat16* __restrict__ dv_pre,              // [B, D] gradient through value (for W_h, W_x, b)
    __nv_bfloat16* __restrict__ dalpha_x,            // [B, D] gradient through alpha (for W_alpha)
    float* __restrict__ db_accum) {                  // [D] accumulator for bias gradient

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // Combined gradient from output and recurrent
        float dh_total;
        if (dh_recurrent) {
            dh_total = __bfloat162float(dh[idx]) + __bfloat162float(dh_recurrent[idx]);
        } else {
            dh_total = __bfloat162float(dh[idx]);
        }

        float alpha = __bfloat162float(alpha_cache[idx]);
        float h_prev_val = __bfloat162float(h_prev[idx]);
        float v_pre_val = __bfloat162float(v_pre[idx]);
        float v = tanhf(v_pre_val);

        // h_t = alpha * h_prev + (1 - alpha) * v
        // dh_prev = dh_total * alpha
        float dh_prev_val = dh_total * alpha;
        dh_prev[idx] = __float2bfloat16(dh_prev_val);

        // dv = dh_total * (1 - alpha)
        float dv = dh_total * (1.0f - alpha);

        // dv_pre = dv * tanh'(v_pre) = dv * (1 - v^2)
        float dv_pre_val = dv * (1.0f - v * v);
        dv_pre[idx] = __float2bfloat16(dv_pre_val);

        // dalpha = dh_total * (h_prev - v)
        float dalpha = dh_total * (h_prev_val - v);

        // dalpha_x = dalpha * sigmoid'(alpha_x) = dalpha * alpha * (1 - alpha)
        float dalpha_x_val = dalpha * alpha * (1.0f - alpha);
        dalpha_x[idx] = __float2bfloat16(dalpha_x_val);

        // Accumulate bias gradient
        atomicAdd(&db_accum[d], dv_pre_val);
    }
}

template<typename T>
__global__ void E63GatedUpdateBackward(
    const int batch_size,
    const int dim,
    const T* __restrict__ h_prev,
    const T* __restrict__ v_pre,
    const T* __restrict__ alpha_cache,
    const T* __restrict__ alpha_x_all_t,
    const T* __restrict__ dh,
    const T* __restrict__ dh_recurrent,
    T* __restrict__ dh_prev,
    T* __restrict__ dv_pre,
    T* __restrict__ dalpha_x,
    float* __restrict__ db_accum) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float dh_total = static_cast<float>(dh[idx]);
        if (dh_recurrent) dh_total += static_cast<float>(dh_recurrent[idx]);

        float alpha = static_cast<float>(alpha_cache[idx]);
        float h_prev_val = static_cast<float>(h_prev[idx]);
        float v_pre_val = static_cast<float>(v_pre[idx]);
        float v = tanhf(v_pre_val);

        // dh_prev
        float dh_prev_val = dh_total * alpha;
        dh_prev[idx] = static_cast<T>(dh_prev_val);

        // dv_pre
        float dv = dh_total * (1.0f - alpha);
        float dv_pre_val = dv * (1.0f - v * v);
        dv_pre[idx] = static_cast<T>(dv_pre_val);

        // dalpha_x
        float dalpha = dh_total * (h_prev_val - v);
        float dalpha_x_val = dalpha * alpha * (1.0f - alpha);
        dalpha_x[idx] = static_cast<T>(dalpha_x_val);

        atomicAdd(&db_accum[d], dv_pre_val);
    }
}

// Utility: Add dh_prev to dh_recurrent for next iteration
__global__ void VectorCopy_BF16(
    const int n,
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

template<typename T>
__global__ void VectorCopy(const int n, const T* __restrict__ src, T* __restrict__ dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

template<typename T>
__global__ void CopyFloatToT_E63(const int n, const float* __restrict__ src, T* __restrict__ dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = static_cast<T>(src[idx]);
    }
}

// Add bias to alpha_x (element-wise) - BF16 specialization
__global__ void AddAlphaBias_BF16(
    const int n,
    const int dim,
    __nv_bfloat16* __restrict__ data,
    const __nv_bfloat16* __restrict__ bias) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const int d = idx % dim;
        data[idx] = __float2bfloat16(__bfloat162float(data[idx]) + __bfloat162float(bias[d]));
    }
}

// Add bias to alpha_x (element-wise) - generic
template<typename T>
__global__ void AddAlphaBias(
    const int n,
    const int dim,
    T* __restrict__ data,
    const T* __restrict__ bias) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const int d = idx % dim;
        data[idx] = static_cast<T>(static_cast<float>(data[idx]) + static_cast<float>(bias[d]));
    }
}

// Accumulate db_alpha from dalpha_x - BF16 specialization
__global__ void AccumulateDbAlpha_BF16(
    const int n,
    const int dim,
    const __nv_bfloat16* __restrict__ dalpha_x_all,
    float* __restrict__ db_alpha_float) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const int d = idx % dim;
        atomicAdd(&db_alpha_float[d], __bfloat162float(dalpha_x_all[idx]));
    }
}

// Accumulate db_alpha from dalpha_x - generic
template<typename T>
__global__ void AccumulateDbAlpha(
    const int n,
    const int dim,
    const T* __restrict__ dalpha_x_all,
    float* __restrict__ db_alpha_float) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const int d = idx % dim;
        atomicAdd(&db_alpha_float[d], static_cast<float>(dalpha_x_all[idx]));
    }
}

}  // anonymous namespace


namespace hasty {
namespace v0 {
namespace elman_ladder {

// =============================================================================
// E63 Nonlinear Delta Forward - BF16 Specialization
// =============================================================================

template<>
E63NonlinearDeltaForward<__nv_bfloat16>::E63NonlinearDeltaForward(
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
void E63NonlinearDeltaForward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_alpha,  // [dim, dim]
    const __nv_bfloat16* b_alpha,  // [dim]
    const __nv_bfloat16* W_h,      // [dim, dim]
    const __nv_bfloat16* W_x,      // [dim, dim]
    const __nv_bfloat16* b,        // [dim]
    const __nv_bfloat16* x,        // [T, B, dim]
    __nv_bfloat16* h,              // [T+1, B, dim]
    __nv_bfloat16* output,         // [T, B, dim]
    __nv_bfloat16* v_pre_cache,    // [T, B, dim] (training)
    __nv_bfloat16* alpha_cache,    // [T, B, dim] (training)
    __nv_bfloat16* workspace) {

    static const __nv_bfloat16 alpha_one = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Workspace layout:
    // [alpha_x_all: T*BD] - pre-computed W_alpha @ x + b_alpha
    // [Wx_all: T*BD]      - pre-computed W_x @ x
    // [tmp_Wh: BD]        - per-step W_h @ h_prev
    __nv_bfloat16* alpha_x_all = workspace;
    __nv_bfloat16* Wx_all = workspace + steps * BD;
    __nv_bfloat16* tmp_Wh = workspace + 2 * steps * BD;

    // =========================================================================
    // BATCH GEMM 1: W_alpha @ x for ALL timesteps
    // Result: alpha_x_all[t] = x[t] @ W_alpha.T
    // =========================================================================
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_alpha, dim_,
        x, dim_,
        &beta_zero,
        alpha_x_all, dim_);

    // Add bias to alpha_x (element-wise)
    const int total_alpha = steps * BD;
    const int alpha_blocks = (total_alpha + block_size - 1) / block_size;
    AddAlphaBias_BF16<<<alpha_blocks, block_size, 0, stream_>>>(total_alpha, dim_, alpha_x_all, b_alpha);

    // =========================================================================
    // BATCH GEMM 2: W_x @ x for ALL timesteps
    // Result: Wx_all[t] = x[t] @ W_x.T
    // =========================================================================
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_x, dim_,
        x, dim_,
        &beta_zero,
        Wx_all, dim_);

    // =========================================================================
    // Sequential timestep loop (cannot parallelize due to h-dependence in v_t)
    // =========================================================================
    for (int t = 0; t < steps; ++t) {
        const __nv_bfloat16* alpha_x_t = alpha_x_all + t * BD;
        const __nv_bfloat16* Wx_t = Wx_all + t * BD;
        const __nv_bfloat16* h_prev = h + t * BD;
        __nv_bfloat16* h_t = h + (t + 1) * BD;
        __nv_bfloat16* out_t = output + t * BD;
        __nv_bfloat16* v_pre_t = training_ ? (v_pre_cache + t * BD) : nullptr;
        __nv_bfloat16* alpha_t = training_ ? (alpha_cache + t * BD) : nullptr;

        // W_h @ h_prev (per-step GEMM - the sequential bottleneck)
        blas<__nv_bfloat16>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha_one,
            W_h, dim_,
            h_prev, dim_,
            &beta_zero,
            tmp_Wh, dim_);

        // Fused forward kernel
        E63FusedForwardKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, alpha_x_t, Wx_t, tmp_Wh, h_prev, b, h_t, out_t, v_pre_t, alpha_t);
    }
}

// =============================================================================
// E63 Nonlinear Delta Backward - BF16 Specialization
// =============================================================================

template<>
E63NonlinearDeltaBackward<__nv_bfloat16>::E63NonlinearDeltaBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void E63NonlinearDeltaBackward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_alpha,
    const __nv_bfloat16* W_h,
    const __nv_bfloat16* W_x,
    const __nv_bfloat16* x,
    const __nv_bfloat16* h,
    const __nv_bfloat16* v_pre_cache,
    const __nv_bfloat16* alpha_cache,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* dx,
    __nv_bfloat16* dW_alpha,
    __nv_bfloat16* db_alpha,
    __nv_bfloat16* dW_h,
    __nv_bfloat16* dW_x,
    __nv_bfloat16* db,
    __nv_bfloat16* workspace) {

    static const __nv_bfloat16 alpha_one = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);
    static const __nv_bfloat16 beta_one = __float2bfloat16(1.0f);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Workspace layout:
    // [dh: BD]              - gradient after self-gate
    // [dh_recurrent: BD]    - gradient from next timestep
    // [dh_prev: BD]         - gradient to previous h
    // [dv_pre_all: T*BD]    - gradient through tanh for all timesteps
    // [dalpha_x_all: T*BD]  - gradient through alpha for all timesteps
    // [db_float: dim]       - float accumulator for bias
    // [db_alpha_float: dim] - float accumulator for alpha bias
    // [alpha_x_all: T*BD]   - need to recompute alpha_x for backward
    __nv_bfloat16* dh = workspace;
    __nv_bfloat16* dh_recurrent = workspace + BD;
    __nv_bfloat16* dh_prev = workspace + 2 * BD;
    __nv_bfloat16* dv_pre_all = workspace + 3 * BD;
    __nv_bfloat16* dalpha_x_all = workspace + 3 * BD + steps * BD;
    float* db_float = reinterpret_cast<float*>(workspace + 3 * BD + 2 * steps * BD);
    float* db_alpha_float = db_float + dim_;
    __nv_bfloat16* alpha_x_all = reinterpret_cast<__nv_bfloat16*>(db_alpha_float + dim_);

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_alpha_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW_alpha, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_h, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_x, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);

    // Recompute alpha_x_all for backward (x @ W_alpha.T + b_alpha)
    // This is more efficient than storing it during forward
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_alpha, dim_,
        x, dim_,
        &beta_zero,
        alpha_x_all, dim_);

    // =========================================================================
    // BPTT Loop (backward through time)
    // =========================================================================
    for (int t = steps - 1; t >= 0; --t) {
        const __nv_bfloat16* h_t = h + (t + 1) * BD;       // h[t+1]
        const __nv_bfloat16* h_prev_t = h + t * BD;        // h[t]
        const __nv_bfloat16* d_out_t = d_output + t * BD;
        const __nv_bfloat16* v_pre_t = v_pre_cache + t * BD;
        const __nv_bfloat16* alpha_t = alpha_cache + t * BD;
        const __nv_bfloat16* alpha_x_t = alpha_x_all + t * BD;
        __nv_bfloat16* dv_pre_t = dv_pre_all + t * BD;
        __nv_bfloat16* dalpha_x_t = dalpha_x_all + t * BD;

        // Step 1: Backward through self-gate
        E63SelfGateBackward_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, d_out_t, dh);

        // Step 2: Backward through gated update
        E63GatedUpdateBackward_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_prev_t, v_pre_t, alpha_t, alpha_x_t,
            dh, dh_recurrent, dh_prev, dv_pre_t, dalpha_x_t, db_float);

        // Step 3: Gradient through W_h path (dh_prev += dv_pre @ W_h)
        // dh_prev already has gradient from alpha path, add W_h contribution
        blas<__nv_bfloat16>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha_one,
            W_h, dim_,
            dv_pre_t, dim_,
            &beta_one,  // Accumulate into dh_prev
            dh_prev, dim_);

        // Step 4: Copy dh_prev to dh_recurrent for next iteration
        if (t > 0) {
            VectorCopy_BF16<<<num_blocks, block_size, 0, stream_>>>(BD, dh_prev, dh_recurrent);
        }
    }

    // =========================================================================
    // Weight gradients via batched GEMMs
    // =========================================================================

    // dW_h = sum_t h[t].T @ dv_pre[t]
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        h, dim_,  // h[0:T] (T*B matrices)
        dv_pre_all, dim_,
        &beta_one,
        dW_h, dim_);

    // dW_x = sum_t x[t].T @ dv_pre[t]
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        dv_pre_all, dim_,
        &beta_one,
        dW_x, dim_);

    // dW_alpha = sum_t x[t].T @ dalpha_x[t]
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        dalpha_x_all, dim_,
        &beta_one,
        dW_alpha, dim_);

    // dx = dv_pre @ W_x + dalpha_x @ W_alpha
    // First: dx = dv_pre @ W_x
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_x, dim_,
        dv_pre_all, dim_,
        &beta_zero,
        dx, dim_);

    // Then: dx += dalpha_x @ W_alpha
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_alpha, dim_,
        dalpha_x_all, dim_,
        &beta_one,
        dx, dim_);

    // db_alpha = sum(dalpha_x, dim=0)
    const int total_alpha = steps * BD;
    const int alpha_blocks = (total_alpha + block_size - 1) / block_size;
    AccumulateDbAlpha_BF16<<<alpha_blocks, block_size, 0, stream_>>>(
        total_alpha, dim_, dalpha_x_all, db_alpha_float);

    // Copy float accumulators to output
    CopyFloatToT_E63<__nv_bfloat16><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_float, db);
    CopyFloatToT_E63<__nv_bfloat16><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_alpha_float, db_alpha);
}

// =============================================================================
// Generic Template Implementations (float, half, double)
// =============================================================================

template<typename T>
E63NonlinearDeltaForward<T>::E63NonlinearDeltaForward(
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
void E63NonlinearDeltaForward<T>::Run(
    int steps,
    const T* W_alpha,
    const T* b_alpha,
    const T* W_h,
    const T* W_x,
    const T* b,
    const T* x,
    T* h,
    T* output,
    T* v_pre_cache,
    T* alpha_cache,
    T* workspace) {

    static const T alpha_one = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    T* alpha_x_all = workspace;
    T* Wx_all = workspace + steps * BD;
    T* tmp_Wh = workspace + 2 * steps * BD;

    // Batch GEMM for W_alpha @ x
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_alpha, dim_,
        x, dim_,
        &beta_zero,
        alpha_x_all, dim_);

    // Add alpha bias
    const int total_alpha = steps * BD;
    const int alpha_blocks = (total_alpha + block_size - 1) / block_size;
    AddAlphaBias<T><<<alpha_blocks, block_size, 0, stream_>>>(total_alpha, dim_, alpha_x_all, b_alpha);

    // Batch GEMM for W_x @ x
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_x, dim_,
        x, dim_,
        &beta_zero,
        Wx_all, dim_);

    // Sequential loop
    for (int t = 0; t < steps; ++t) {
        const T* alpha_x_t = alpha_x_all + t * BD;
        const T* Wx_t = Wx_all + t * BD;
        const T* h_prev = h + t * BD;
        T* h_t = h + (t + 1) * BD;
        T* out_t = output + t * BD;
        T* v_pre_t = training_ ? (v_pre_cache + t * BD) : nullptr;
        T* alpha_t = training_ ? (alpha_cache + t * BD) : nullptr;

        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha_one,
            W_h, dim_,
            h_prev, dim_,
            &beta_zero,
            tmp_Wh, dim_);

        E63FusedForwardKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, alpha_x_t, Wx_t, tmp_Wh, h_prev, b, h_t, out_t, v_pre_t, alpha_t);
    }
}

template<typename T>
E63NonlinearDeltaBackward<T>::E63NonlinearDeltaBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void E63NonlinearDeltaBackward<T>::Run(
    int steps,
    const T* W_alpha,
    const T* W_h,
    const T* W_x,
    const T* x,
    const T* h,
    const T* v_pre_cache,
    const T* alpha_cache,
    const T* d_output,
    T* dx,
    T* dW_alpha,
    T* db_alpha,
    T* dW_h,
    T* dW_x,
    T* db,
    T* workspace) {

    static const T alpha_one = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    T* dh = workspace;
    T* dh_recurrent = workspace + BD;
    T* dh_prev = workspace + 2 * BD;
    T* dv_pre_all = workspace + 3 * BD;
    T* dalpha_x_all = workspace + 3 * BD + steps * BD;
    float* db_float = reinterpret_cast<float*>(workspace + 3 * BD + 2 * steps * BD);
    float* db_alpha_float = db_float + dim_;
    T* alpha_x_all = reinterpret_cast<T*>(db_alpha_float + dim_);

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(T), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_alpha_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW_alpha, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dW_h, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dW_x, 0, dim_ * dim_ * sizeof(T), stream_);

    // Recompute alpha_x_all
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_alpha, dim_,
        x, dim_,
        &beta_zero,
        alpha_x_all, dim_);

    // BPTT loop
    for (int t = steps - 1; t >= 0; --t) {
        const T* h_t = h + (t + 1) * BD;
        const T* h_prev_t = h + t * BD;
        const T* d_out_t = d_output + t * BD;
        const T* v_pre_t = v_pre_cache + t * BD;
        const T* alpha_t = alpha_cache + t * BD;
        const T* alpha_x_t = alpha_x_all + t * BD;
        T* dv_pre_t = dv_pre_all + t * BD;
        T* dalpha_x_t = dalpha_x_all + t * BD;

        E63SelfGateBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, d_out_t, dh);

        E63GatedUpdateBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_prev_t, v_pre_t, alpha_t, alpha_x_t,
            dh, dh_recurrent, dh_prev, dv_pre_t, dalpha_x_t, db_float);

        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha_one,
            W_h, dim_,
            dv_pre_t, dim_,
            &beta_one,
            dh_prev, dim_);

        if (t > 0) {
            VectorCopy<T><<<num_blocks, block_size, 0, stream_>>>(BD, dh_prev, dh_recurrent);
        }
    }

    // Weight gradients
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        h, dim_,
        dv_pre_all, dim_,
        &beta_one,
        dW_h, dim_);

    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        dv_pre_all, dim_,
        &beta_one,
        dW_x, dim_);

    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        dalpha_x_all, dim_,
        &beta_one,
        dW_alpha, dim_);

    // dx
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_x, dim_,
        dv_pre_all, dim_,
        &beta_zero,
        dx, dim_);

    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_alpha, dim_,
        dalpha_x_all, dim_,
        &beta_one,
        dx, dim_);

    // db_alpha accumulation
    const int total_alpha = steps * BD;
    const int alpha_blocks = (total_alpha + block_size - 1) / block_size;
    AccumulateDbAlpha<T><<<alpha_blocks, block_size, 0, stream_>>>(
        total_alpha, dim_, dalpha_x_all, db_alpha_float);

    CopyFloatToT_E63<T><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_float, db);
    CopyFloatToT_E63<T><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_alpha_float, db_alpha);
}

// Explicit template instantiations
template struct E63NonlinearDeltaForward<__half>;
template struct E63NonlinearDeltaForward<float>;
template struct E63NonlinearDeltaForward<double>;

template struct E63NonlinearDeltaBackward<__half>;
template struct E63NonlinearDeltaBackward<float>;
template struct E63NonlinearDeltaBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
