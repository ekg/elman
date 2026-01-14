// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E61: Decay-Gated Elman - Mamba2-style Input-Dependent Decay
//
// Architecture:
// alpha_t = sigmoid(x @ W_alpha.T + b_alpha)    # Decay gate (input-dependent)
// v_t = x @ W_v.T + b_v                         # New value (linear, no tanh)
// h_t = alpha_t * h_{t-1} + (1 - alpha_t) * v_t # Gated update
// output = h * silu(h)                          # Self-gating
//
// Key Properties:
// - Linear in h: Jacobian dh_t/dh_{t-1} = diag(alpha_t)
// - Parallelizable via associative scan (future optimization)
// - When alpha -> 1: Preserve (gradient = 1)
// - When alpha -> 0: Replace with input (gradient = 0)
//
// Key Optimization: Batch both projections upfront since they only depend on x

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
// Forward Kernel: Decay-gated update + self-gating
// Computes: h_new = alpha * h_prev + (1 - alpha) * v
//          output = h_new * silu(h_new)
// =============================================================================

__global__ void E61DecayGateKernel_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ alpha_logits,  // Pre-sigmoid logits
    const __nv_bfloat16* __restrict__ v,             // New values
    const __nv_bfloat16* __restrict__ h_prev,
    const __nv_bfloat16* __restrict__ b_alpha,       // Bias for alpha
    const __nv_bfloat16* __restrict__ b_v,           // Bias for v
    __nv_bfloat16* __restrict__ h_out,
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ alpha_cache) {       // Cache alpha for backward

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // Compute alpha = sigmoid(alpha_logits + b_alpha)
        float alpha_logit = __bfloat162float(alpha_logits[idx]) + __bfloat162float(b_alpha[d]);
        float alpha = 1.0f / (1.0f + __expf(-alpha_logit));

        // Compute v with bias
        float v_val = __bfloat162float(v[idx]) + __bfloat162float(b_v[d]);

        // Decay-gated update: h = alpha * h_prev + (1 - alpha) * v
        float h_prev_val = __bfloat162float(h_prev[idx]);
        float h_new = alpha * h_prev_val + (1.0f - alpha) * v_val;

        // Store hidden state
        h_out[idx] = __float2bfloat16(h_new);

        // Cache alpha for backward
        if (alpha_cache) alpha_cache[idx] = __float2bfloat16(alpha);

        // Self-gate: output = h * silu(h)
        float sigmoid_h = 1.0f / (1.0f + __expf(-h_new));
        float silu_h = h_new * sigmoid_h;
        output[idx] = __float2bfloat16(h_new * silu_h);
    }
}

template<typename T>
__global__ void E61DecayGateKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ alpha_logits,
    const T* __restrict__ v,
    const T* __restrict__ h_prev,
    const T* __restrict__ b_alpha,
    const T* __restrict__ b_v,
    T* __restrict__ h_out,
    T* __restrict__ output,
    T* __restrict__ alpha_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float alpha_logit = static_cast<float>(alpha_logits[idx]) + static_cast<float>(b_alpha[d]);
        float alpha = 1.0f / (1.0f + expf(-alpha_logit));
        float v_val = static_cast<float>(v[idx]) + static_cast<float>(b_v[d]);
        float h_prev_val = static_cast<float>(h_prev[idx]);
        float h_new = alpha * h_prev_val + (1.0f - alpha) * v_val;

        h_out[idx] = static_cast<T>(h_new);
        if (alpha_cache) alpha_cache[idx] = static_cast<T>(alpha);

        float sigmoid_h = 1.0f / (1.0f + expf(-h_new));
        float silu_h = h_new * sigmoid_h;
        output[idx] = static_cast<T>(h_new * silu_h);
    }
}

// =============================================================================
// Backward Kernels
// =============================================================================

// Backward through self-gate: output = h * silu(h) = h^2 * sigmoid(h)
// d(output)/dh = silu(h) * (2 + h*(1-sigmoid(h)))
__global__ void E61SelfGateBackward_BF16(
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
__global__ void E61SelfGateBackward(
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

// Backward through decay gate: h = alpha * h_prev + (1 - alpha) * v
// dh_prev = dh * alpha
// dv = dh * (1 - alpha)
// d_alpha = dh * (h_prev - v)
// d_alpha_logit = d_alpha * alpha * (1 - alpha)  (sigmoid derivative)
__global__ void E61DecayGateBackward_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ alpha,         // Cached alpha values
    const __nv_bfloat16* __restrict__ h_prev,
    const __nv_bfloat16* __restrict__ v,             // v with bias already applied
    const __nv_bfloat16* __restrict__ b_v,
    const __nv_bfloat16* __restrict__ dh,            // Gradient from self-gate
    const __nv_bfloat16* __restrict__ dh_recurrent,  // Recurrent gradient from next step
    __nv_bfloat16* __restrict__ dh_prev,             // Output: gradient to previous h
    __nv_bfloat16* __restrict__ d_alpha_logit,       // Output: gradient to alpha logits
    __nv_bfloat16* __restrict__ dv,                  // Output: gradient to v (before bias)
    float* __restrict__ db_alpha,                    // Accumulated bias gradient
    float* __restrict__ db_v) {                      // Accumulated v bias gradient

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float alpha_val = __bfloat162float(alpha[idx]);
        float h_prev_val = __bfloat162float(h_prev[idx]);
        float v_val = __bfloat162float(v[idx]) + __bfloat162float(b_v[d]);

        // Combine gradients
        float grad = __bfloat162float(dh[idx]);
        if (dh_recurrent) {
            grad += __bfloat162float(dh_recurrent[idx]);
        }

        // Gradient to h_prev: dh_prev = dh * alpha
        dh_prev[idx] = __float2bfloat16(grad * alpha_val);

        // Gradient to v: dv = dh * (1 - alpha)
        float dv_val = grad * (1.0f - alpha_val);
        dv[idx] = __float2bfloat16(dv_val);

        // Gradient to alpha: d_alpha = dh * (h_prev - v)
        float d_alpha = grad * (h_prev_val - v_val);

        // Gradient through sigmoid: d_alpha_logit = d_alpha * alpha * (1 - alpha)
        float d_alpha_logit_val = d_alpha * alpha_val * (1.0f - alpha_val);
        d_alpha_logit[idx] = __float2bfloat16(d_alpha_logit_val);

        // Accumulate bias gradients
        atomicAdd(&db_alpha[d], d_alpha_logit_val);
        atomicAdd(&db_v[d], dv_val);
    }
}

template<typename T>
__global__ void E61DecayGateBackward(
    const int batch_size,
    const int dim,
    const T* __restrict__ alpha,
    const T* __restrict__ h_prev,
    const T* __restrict__ v,
    const T* __restrict__ b_v,
    const T* __restrict__ dh,
    const T* __restrict__ dh_recurrent,
    T* __restrict__ dh_prev,
    T* __restrict__ d_alpha_logit,
    T* __restrict__ dv,
    float* __restrict__ db_alpha,
    float* __restrict__ db_v) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float alpha_val = static_cast<float>(alpha[idx]);
        float h_prev_val = static_cast<float>(h_prev[idx]);
        float v_val = static_cast<float>(v[idx]) + static_cast<float>(b_v[d]);

        float grad = static_cast<float>(dh[idx]);
        if (dh_recurrent) {
            grad += static_cast<float>(dh_recurrent[idx]);
        }

        dh_prev[idx] = static_cast<T>(grad * alpha_val);

        float dv_val = grad * (1.0f - alpha_val);
        dv[idx] = static_cast<T>(dv_val);

        float d_alpha = grad * (h_prev_val - v_val);
        float d_alpha_logit_val = d_alpha * alpha_val * (1.0f - alpha_val);
        d_alpha_logit[idx] = static_cast<T>(d_alpha_logit_val);

        atomicAdd(&db_alpha[d], d_alpha_logit_val);
        atomicAdd(&db_v[d], dv_val);
    }
}

// =============================================================================
// Utility Kernels
// =============================================================================

template<typename T>
__global__ void CopyFloatToT_E61(const int n, const float* __restrict__ src, T* __restrict__ dst) {
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
// E61 Decay-Gated Forward - BF16 Specialization
// =============================================================================

template<>
E61DecayGatedForward<__nv_bfloat16>::E61DecayGatedForward(
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
void E61DecayGatedForward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_alpha,   // [dim, dim]
    const __nv_bfloat16* b_alpha,   // [dim]
    const __nv_bfloat16* W_v,       // [dim, dim]
    const __nv_bfloat16* b_v,       // [dim]
    const __nv_bfloat16* x,         // [T, B, dim]
    __nv_bfloat16* h,               // [T+1, B, dim]
    __nv_bfloat16* output,          // [T, B, dim]
    __nv_bfloat16* alpha_cache,     // [T, B, dim] for backward
    __nv_bfloat16* workspace) {     // [2*T*B*dim] for alpha_logits and v

    static const __nv_bfloat16 alpha_one = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);

    const int BD = batch_size_ * dim_;
    const int TBD = steps * BD;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Workspace layout: [alpha_logits: T*BD] [v: T*BD]
    __nv_bfloat16* alpha_logits_all = workspace;
    __nv_bfloat16* v_all = workspace + TBD;

    // =========================================================================
    // KEY OPTIMIZATION: Batch both projections for all timesteps
    // x @ W_alpha.T for all T (one GEMM)
    // x @ W_v.T for all T (one GEMM)
    // =========================================================================

    // Batch GEMM: alpha_logits = x @ W_alpha.T  (no bias yet, added in kernel)
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_alpha, dim_,
        x, dim_,
        &beta_zero,
        alpha_logits_all, dim_);

    // Batch GEMM: v = x @ W_v.T  (no bias yet, added in kernel)
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_v, dim_,
        x, dim_,
        &beta_zero,
        v_all, dim_);

    // Process each timestep sequentially (recurrence)
    for (int t = 0; t < steps; ++t) {
        const __nv_bfloat16* alpha_logits_t = alpha_logits_all + t * BD;
        const __nv_bfloat16* v_t = v_all + t * BD;
        const __nv_bfloat16* h_prev = h + t * BD;
        __nv_bfloat16* h_t = h + (t + 1) * BD;
        __nv_bfloat16* out_t = output + t * BD;
        __nv_bfloat16* alpha_t = training_ ? (alpha_cache + t * BD) : nullptr;

        E61DecayGateKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_,
            alpha_logits_t, v_t, h_prev,
            b_alpha, b_v,
            h_t, out_t, alpha_t);
    }
}

// =============================================================================
// E61 Decay-Gated Backward - BF16 Specialization
// =============================================================================

template<>
E61DecayGatedBackward<__nv_bfloat16>::E61DecayGatedBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void E61DecayGatedBackward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_alpha,
    const __nv_bfloat16* W_v,
    const __nv_bfloat16* b_v,
    const __nv_bfloat16* x,
    const __nv_bfloat16* h,
    const __nv_bfloat16* alpha_cache,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* dx,
    __nv_bfloat16* dW_alpha,
    __nv_bfloat16* db_alpha,
    __nv_bfloat16* dW_v,
    __nv_bfloat16* db_v,
    __nv_bfloat16* workspace) {

    static const __nv_bfloat16 alpha_one = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);
    static const __nv_bfloat16 beta_one = __float2bfloat16(1.0f);

    const int BD = batch_size_ * dim_;
    const int TBD = steps * BD;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Workspace layout:
    // [dh: BD] [dh_recurrent: BD] [d_alpha_logit_all: T*BD] [dv_all: T*BD]
    // [v_all: T*BD] [db_alpha_float: dim] [db_v_float: dim]
    __nv_bfloat16* dh = workspace;
    __nv_bfloat16* dh_recurrent = workspace + BD;
    __nv_bfloat16* d_alpha_logit_all = workspace + 2 * BD;
    __nv_bfloat16* dv_all = workspace + 2 * BD + TBD;
    __nv_bfloat16* v_all = workspace + 2 * BD + 2 * TBD;
    float* db_alpha_float = reinterpret_cast<float*>(workspace + 2 * BD + 3 * TBD);
    float* db_v_float = db_alpha_float + dim_;

    // Zero out accumulators
    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(db_alpha_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_v_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW_alpha, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_v, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);

    // Recompute v for backward (needed for gradient computation)
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_v, dim_,
        x, dim_,
        &beta_zero,
        v_all, dim_);

    // BPTT loop
    for (int t = steps - 1; t >= 0; --t) {
        const __nv_bfloat16* h_t = h + (t + 1) * BD;
        const __nv_bfloat16* h_prev = h + t * BD;
        const __nv_bfloat16* alpha_t = alpha_cache + t * BD;
        const __nv_bfloat16* v_t = v_all + t * BD;
        const __nv_bfloat16* d_out_t = d_output + t * BD;
        __nv_bfloat16* d_alpha_logit_t = d_alpha_logit_all + t * BD;
        __nv_bfloat16* dv_t = dv_all + t * BD;

        // Backward through self-gate
        E61SelfGateBackward_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, d_out_t, dh);

        // Backward through decay gate
        // Use dh_recurrent from next timestep (or zero if last step)
        __nv_bfloat16* dh_recurrent_in = (t < steps - 1) ? dh_recurrent : nullptr;

        // Temporary buffer for dh_prev output this step
        __nv_bfloat16* dh_prev_temp = workspace;  // Reuse dh slot temporarily

        E61DecayGateBackward_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_,
            alpha_t, h_prev, v_t, b_v,
            dh, dh_recurrent_in,
            dh_prev_temp, d_alpha_logit_t, dv_t,
            db_alpha_float, db_v_float);

        // Copy dh_prev to dh_recurrent for next iteration (if not first step)
        if (t > 0) {
            cudaMemcpyAsync(dh_recurrent, dh_prev_temp, BD * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToDevice, stream_);
        }
    }

    // =========================================================================
    // Batched GEMMs for weight gradients
    // =========================================================================

    // dW_alpha = d_alpha_logit_all.T @ x
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        d_alpha_logit_all, dim_,
        &beta_one,
        dW_alpha, dim_);

    // dW_v = dv_all.T @ x
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        dv_all, dim_,
        &beta_one,
        dW_v, dim_);

    // =========================================================================
    // Batched GEMM for dx: dx = W_alpha @ d_alpha_logit + W_v @ dv
    // =========================================================================

    // dx = W_alpha @ d_alpha_logit_all
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_alpha, dim_,
        d_alpha_logit_all, dim_,
        &beta_zero,
        dx, dim_);

    // dx += W_v @ dv_all
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_v, dim_,
        dv_all, dim_,
        &beta_one,
        dx, dim_);

    // Copy float gradients to output type
    CopyFloatToT_E61<__nv_bfloat16><<<(dim_ + 255) / 256, 256, 0, stream_>>>(
        dim_, db_alpha_float, db_alpha);
    CopyFloatToT_E61<__nv_bfloat16><<<(dim_ + 255) / 256, 256, 0, stream_>>>(
        dim_, db_v_float, db_v);
}

// =============================================================================
// Generic Template Implementations (float, half, double)
// =============================================================================

template<typename T>
E61DecayGatedForward<T>::E61DecayGatedForward(
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
void E61DecayGatedForward<T>::Run(
    int steps,
    const T* W_alpha,
    const T* b_alpha,
    const T* W_v,
    const T* b_v,
    const T* x,
    T* h,
    T* output,
    T* alpha_cache,
    T* workspace) {

    static const T alpha_one = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int BD = batch_size_ * dim_;
    const int TBD = steps * BD;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    T* alpha_logits_all = workspace;
    T* v_all = workspace + TBD;

    // Batch GEMMs
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_alpha, dim_,
        x, dim_,
        &beta_zero,
        alpha_logits_all, dim_);

    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_v, dim_,
        x, dim_,
        &beta_zero,
        v_all, dim_);

    for (int t = 0; t < steps; ++t) {
        const T* alpha_logits_t = alpha_logits_all + t * BD;
        const T* v_t = v_all + t * BD;
        const T* h_prev = h + t * BD;
        T* h_t = h + (t + 1) * BD;
        T* out_t = output + t * BD;
        T* alpha_t = training_ ? (alpha_cache + t * BD) : nullptr;

        E61DecayGateKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_,
            alpha_logits_t, v_t, h_prev,
            b_alpha, b_v,
            h_t, out_t, alpha_t);
    }
}

template<typename T>
E61DecayGatedBackward<T>::E61DecayGatedBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void E61DecayGatedBackward<T>::Run(
    int steps,
    const T* W_alpha,
    const T* W_v,
    const T* b_v,
    const T* x,
    const T* h,
    const T* alpha_cache,
    const T* d_output,
    T* dx,
    T* dW_alpha,
    T* db_alpha,
    T* dW_v,
    T* db_v,
    T* workspace) {

    static const T alpha_one = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int BD = batch_size_ * dim_;
    const int TBD = steps * BD;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    T* dh = workspace;
    T* dh_recurrent = workspace + BD;
    T* d_alpha_logit_all = workspace + 2 * BD;
    T* dv_all = workspace + 2 * BD + TBD;
    T* v_all = workspace + 2 * BD + 2 * TBD;
    float* db_alpha_float = reinterpret_cast<float*>(workspace + 2 * BD + 3 * TBD);
    float* db_v_float = db_alpha_float + dim_;

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(T), stream_);
    cudaMemsetAsync(db_alpha_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_v_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW_alpha, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dW_v, 0, dim_ * dim_ * sizeof(T), stream_);

    // Recompute v
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_v, dim_,
        x, dim_,
        &beta_zero,
        v_all, dim_);

    for (int t = steps - 1; t >= 0; --t) {
        const T* h_t = h + (t + 1) * BD;
        const T* h_prev = h + t * BD;
        const T* alpha_t = alpha_cache + t * BD;
        const T* v_t = v_all + t * BD;
        const T* d_out_t = d_output + t * BD;
        T* d_alpha_logit_t = d_alpha_logit_all + t * BD;
        T* dv_t = dv_all + t * BD;

        E61SelfGateBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, d_out_t, dh);

        T* dh_recurrent_in = (t < steps - 1) ? dh_recurrent : nullptr;
        T* dh_prev_temp = workspace;

        E61DecayGateBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_,
            alpha_t, h_prev, v_t, b_v,
            dh, dh_recurrent_in,
            dh_prev_temp, d_alpha_logit_t, dv_t,
            db_alpha_float, db_v_float);

        if (t > 0) {
            cudaMemcpyAsync(dh_recurrent, dh_prev_temp, BD * sizeof(T),
                          cudaMemcpyDeviceToDevice, stream_);
        }
    }

    // Weight gradients
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        d_alpha_logit_all, dim_,
        &beta_one,
        dW_alpha, dim_);

    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        dv_all, dim_,
        &beta_one,
        dW_v, dim_);

    // dx
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_alpha, dim_,
        d_alpha_logit_all, dim_,
        &beta_zero,
        dx, dim_);

    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_v, dim_,
        dv_all, dim_,
        &beta_one,
        dx, dim_);

    CopyFloatToT_E61<T><<<(dim_ + 255) / 256, 256, 0, stream_>>>(
        dim_, db_alpha_float, db_alpha);
    CopyFloatToT_E61<T><<<(dim_ + 255) / 256, 256, 0, stream_>>>(
        dim_, db_v_float, db_v);
}

// Explicit template instantiations
template struct E61DecayGatedForward<__half>;
template struct E61DecayGatedForward<float>;
template struct E61DecayGatedForward<double>;

template struct E61DecayGatedBackward<__half>;
template struct E61DecayGatedBackward<float>;
template struct E61DecayGatedBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
