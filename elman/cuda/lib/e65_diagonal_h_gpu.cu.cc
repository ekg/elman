// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E65: Diagonal H-Dependence - Learnable Per-Dimension Scaling
//
// Architecture:
// alpha_t = sigmoid(x @ W_alpha.T + b_alpha)        # Decay gate (x-only)
// v_t = tanh(d_h * h_{t-1} + W_x @ x_t + b)         # d_h is [dim] vector, element-wise!
// h_t = alpha_t * h_{t-1} + (1 - alpha_t) * v_t    # Gated update
// output = h * silu(h)                              # Self-gating
//
// Key Properties:
// - O(d) cost per timestep for h-dependence (diagonal instead of matrix)
// - UTM-class: h is inside the tanh nonlinearity
// - Learnable per-dimension importance of h contribution
//
// Jacobian: dh_t/dh_{t-1} = diag(alpha + (1-alpha)*(1-v^2)*d_h)
//
// Key Optimization: Batch both W_alpha and W_x projections upfront since they only depend on x

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
// Forward Kernel: Diagonal h-dependence + gated update + self-gating
// Computes: v = tanh(d_h * h_prev + Wx + b)
//          h_new = alpha * h_prev + (1 - alpha) * v
//          output = h_new * silu(h_new)
// =============================================================================

__global__ void E65DiagonalHKernel_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ alpha_logits,  // Pre-sigmoid logits from W_alpha @ x
    const __nv_bfloat16* __restrict__ Wx,            // W_x @ x (pre-computed)
    const __nv_bfloat16* __restrict__ h_prev,
    const __nv_bfloat16* __restrict__ d_h,           // [dim] diagonal vector
    const __nv_bfloat16* __restrict__ b_alpha,       // Bias for alpha
    const __nv_bfloat16* __restrict__ b,             // Bias for v
    __nv_bfloat16* __restrict__ h_out,
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ alpha_cache,         // Cache alpha for backward
    __nv_bfloat16* __restrict__ v_pre_cache) {       // Cache pre-tanh value for backward

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // Compute alpha = sigmoid(alpha_logits + b_alpha)
        float alpha_logit = __bfloat162float(alpha_logits[idx]) + __bfloat162float(b_alpha[d]);
        float alpha = 1.0f / (1.0f + __expf(-alpha_logit));

        // Compute v_pre = d_h * h_prev + Wx + b (DIAGONAL h-dependence!)
        float h_prev_val = __bfloat162float(h_prev[idx]);
        float d_h_val = __bfloat162float(d_h[d]);
        float Wx_val = __bfloat162float(Wx[idx]);
        float b_val = __bfloat162float(b[d]);
        float v_pre = d_h_val * h_prev_val + Wx_val + b_val;

        // v = tanh(v_pre)
        float v = tanhf(v_pre);

        // Gated update: h = alpha * h_prev + (1 - alpha) * v
        float h_new = alpha * h_prev_val + (1.0f - alpha) * v;

        // Store hidden state
        h_out[idx] = __float2bfloat16(h_new);

        // Cache for backward
        if (alpha_cache) alpha_cache[idx] = __float2bfloat16(alpha);
        if (v_pre_cache) v_pre_cache[idx] = __float2bfloat16(v_pre);

        // Self-gate: output = h * silu(h)
        float sigmoid_h = 1.0f / (1.0f + __expf(-h_new));
        float silu_h = h_new * sigmoid_h;
        output[idx] = __float2bfloat16(h_new * silu_h);
    }
}

template<typename T>
__global__ void E65DiagonalHKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ alpha_logits,
    const T* __restrict__ Wx,
    const T* __restrict__ h_prev,
    const T* __restrict__ d_h,
    const T* __restrict__ b_alpha,
    const T* __restrict__ b,
    T* __restrict__ h_out,
    T* __restrict__ output,
    T* __restrict__ alpha_cache,
    T* __restrict__ v_pre_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float alpha_logit = static_cast<float>(alpha_logits[idx]) + static_cast<float>(b_alpha[d]);
        float alpha = 1.0f / (1.0f + expf(-alpha_logit));

        float h_prev_val = static_cast<float>(h_prev[idx]);
        float d_h_val = static_cast<float>(d_h[d]);
        float Wx_val = static_cast<float>(Wx[idx]);
        float b_val = static_cast<float>(b[d]);
        float v_pre = d_h_val * h_prev_val + Wx_val + b_val;

        float v = tanhf(v_pre);
        float h_new = alpha * h_prev_val + (1.0f - alpha) * v;

        h_out[idx] = static_cast<T>(h_new);
        if (alpha_cache) alpha_cache[idx] = static_cast<T>(alpha);
        if (v_pre_cache) v_pre_cache[idx] = static_cast<T>(v_pre);

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
__global__ void E65SelfGateBackward_BF16(
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
__global__ void E65SelfGateBackward(
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

// Backward through gated update + diagonal h-dependence
// h = alpha * h_prev + (1 - alpha) * v
// v = tanh(d_h * h_prev + Wx + b)
//
// Gradients:
// dv = dh * (1 - alpha)
// d_v_pre = dv * (1 - v^2)  (tanh derivative)
// dh_prev = dh * alpha + d_v_pre * d_h  (two paths: direct + through v)
// d_alpha = dh * (h_prev - v)
// d_alpha_logit = d_alpha * alpha * (1 - alpha)
// dWx = d_v_pre (gradient flows to W_x @ x)
// d_d_h = d_v_pre * h_prev (accumulate across batch)
__global__ void E65DiagonalHBackward_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ alpha,
    const __nv_bfloat16* __restrict__ v_pre,
    const __nv_bfloat16* __restrict__ h_prev,
    const __nv_bfloat16* __restrict__ d_h,           // [dim] diagonal vector
    const __nv_bfloat16* __restrict__ dh,            // Gradient from self-gate
    const __nv_bfloat16* __restrict__ dh_recurrent,  // Recurrent gradient from next step
    __nv_bfloat16* __restrict__ dh_prev,             // Output: gradient to previous h
    __nv_bfloat16* __restrict__ d_alpha_logit,       // Output: gradient to alpha logits
    __nv_bfloat16* __restrict__ dWx,                 // Output: gradient to W_x @ x
    float* __restrict__ db_alpha,                    // Accumulated bias gradient
    float* __restrict__ db,                          // Accumulated v bias gradient
    float* __restrict__ d_d_h) {                     // Accumulated diagonal gradient

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float alpha_val = __bfloat162float(alpha[idx]);
        float v_pre_val = __bfloat162float(v_pre[idx]);
        float h_prev_val = __bfloat162float(h_prev[idx]);
        float d_h_val = __bfloat162float(d_h[d]);

        // Combine gradients
        float grad = __bfloat162float(dh[idx]);
        if (dh_recurrent) {
            grad += __bfloat162float(dh_recurrent[idx]);
        }

        // v = tanh(v_pre)
        float v = tanhf(v_pre_val);

        // Gradient to v: dv = dh * (1 - alpha)
        float dv = grad * (1.0f - alpha_val);

        // Gradient through tanh: d_v_pre = dv * (1 - v^2)
        float d_v_pre = dv * (1.0f - v * v);

        // Gradient to h_prev: through alpha path + through v path
        float dh_prev_val = grad * alpha_val + d_v_pre * d_h_val;
        dh_prev[idx] = __float2bfloat16(dh_prev_val);

        // Gradient to alpha: d_alpha = dh * (h_prev - v)
        float d_alpha = grad * (h_prev_val - v);

        // Gradient through sigmoid: d_alpha_logit = d_alpha * alpha * (1 - alpha)
        float d_alpha_logit_val = d_alpha * alpha_val * (1.0f - alpha_val);
        d_alpha_logit[idx] = __float2bfloat16(d_alpha_logit_val);

        // Gradient to W_x @ x (same as gradient to v_pre)
        dWx[idx] = __float2bfloat16(d_v_pre);

        // Accumulate bias gradients
        atomicAdd(&db_alpha[d], d_alpha_logit_val);
        atomicAdd(&db[d], d_v_pre);

        // Accumulate diagonal gradient: d_d_h = d_v_pre * h_prev
        atomicAdd(&d_d_h[d], d_v_pre * h_prev_val);
    }
}

template<typename T>
__global__ void E65DiagonalHBackwardKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ alpha,
    const T* __restrict__ v_pre,
    const T* __restrict__ h_prev,
    const T* __restrict__ d_h,
    const T* __restrict__ dh,
    const T* __restrict__ dh_recurrent,
    T* __restrict__ dh_prev,
    T* __restrict__ d_alpha_logit,
    T* __restrict__ dWx,
    float* __restrict__ db_alpha,
    float* __restrict__ db,
    float* __restrict__ d_d_h) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float alpha_val = static_cast<float>(alpha[idx]);
        float v_pre_val = static_cast<float>(v_pre[idx]);
        float h_prev_val = static_cast<float>(h_prev[idx]);
        float d_h_val = static_cast<float>(d_h[d]);

        float grad = static_cast<float>(dh[idx]);
        if (dh_recurrent) {
            grad += static_cast<float>(dh_recurrent[idx]);
        }

        float v = tanhf(v_pre_val);
        float dv = grad * (1.0f - alpha_val);
        float d_v_pre = dv * (1.0f - v * v);

        dh_prev[idx] = static_cast<T>(grad * alpha_val + d_v_pre * d_h_val);

        float d_alpha = grad * (h_prev_val - v);
        float d_alpha_logit_val = d_alpha * alpha_val * (1.0f - alpha_val);
        d_alpha_logit[idx] = static_cast<T>(d_alpha_logit_val);

        dWx[idx] = static_cast<T>(d_v_pre);

        atomicAdd(&db_alpha[d], d_alpha_logit_val);
        atomicAdd(&db[d], d_v_pre);
        atomicAdd(&d_d_h[d], d_v_pre * h_prev_val);
    }
}

// =============================================================================
// Utility Kernels
// =============================================================================

template<typename T>
__global__ void CopyFloatToT_E65(const int n, const float* __restrict__ src, T* __restrict__ dst) {
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
// E65 Diagonal H Forward - BF16 Specialization
// =============================================================================

template<>
E65DiagonalHForward<__nv_bfloat16>::E65DiagonalHForward(
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
void E65DiagonalHForward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_alpha,   // [dim, dim]
    const __nv_bfloat16* b_alpha,   // [dim]
    const __nv_bfloat16* d_h,       // [dim] diagonal vector
    const __nv_bfloat16* W_x,       // [dim, dim]
    const __nv_bfloat16* b,         // [dim]
    const __nv_bfloat16* x,         // [T, B, dim]
    __nv_bfloat16* h,               // [T+1, B, dim]
    __nv_bfloat16* output,          // [T, B, dim]
    __nv_bfloat16* v_pre_cache,     // [T, B, dim] for backward
    __nv_bfloat16* alpha_cache,     // [T, B, dim] for backward
    __nv_bfloat16* workspace) {     // [2*T*B*dim] for alpha_logits and Wx

    static const __nv_bfloat16 alpha_one = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);

    const int BD = batch_size_ * dim_;
    const int TBD = steps * BD;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Workspace layout: [alpha_logits: T*BD] [Wx: T*BD]
    __nv_bfloat16* alpha_logits_all = workspace;
    __nv_bfloat16* Wx_all = workspace + TBD;

    // =========================================================================
    // KEY OPTIMIZATION: Batch both projections for all timesteps
    // x @ W_alpha.T for all T (one GEMM)
    // x @ W_x.T for all T (one GEMM)
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

    // Batch GEMM: Wx = x @ W_x.T  (no bias yet, added in kernel)
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_x, dim_,
        x, dim_,
        &beta_zero,
        Wx_all, dim_);

    // Process each timestep sequentially (recurrence)
    for (int t = 0; t < steps; ++t) {
        const __nv_bfloat16* alpha_logits_t = alpha_logits_all + t * BD;
        const __nv_bfloat16* Wx_t = Wx_all + t * BD;
        const __nv_bfloat16* h_prev = h + t * BD;
        __nv_bfloat16* h_t = h + (t + 1) * BD;
        __nv_bfloat16* out_t = output + t * BD;
        __nv_bfloat16* alpha_t = training_ ? (alpha_cache + t * BD) : nullptr;
        __nv_bfloat16* v_pre_t = training_ ? (v_pre_cache + t * BD) : nullptr;

        E65DiagonalHKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_,
            alpha_logits_t, Wx_t, h_prev,
            d_h, b_alpha, b,
            h_t, out_t, alpha_t, v_pre_t);
    }
}

// =============================================================================
// E65 Diagonal H Backward - BF16 Specialization
// =============================================================================

template<>
E65DiagonalHBackward<__nv_bfloat16>::E65DiagonalHBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void E65DiagonalHBackward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_alpha,
    const __nv_bfloat16* d_h,
    const __nv_bfloat16* W_x,
    const __nv_bfloat16* x,
    const __nv_bfloat16* h,
    const __nv_bfloat16* v_pre_cache,
    const __nv_bfloat16* alpha_cache,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* dx,
    __nv_bfloat16* dW_alpha,
    __nv_bfloat16* db_alpha,
    __nv_bfloat16* d_d_h,
    __nv_bfloat16* dW_x,
    __nv_bfloat16* db,
    __nv_bfloat16* workspace) {

    static const __nv_bfloat16 alpha_one = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);
    static const __nv_bfloat16 beta_one = __float2bfloat16(1.0f);

    const int BD = batch_size_ * dim_;
    const int TBD = steps * BD;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Workspace layout:
    // [dh: BD] [dh_recurrent: BD] [d_alpha_logit_all: T*BD] [dWx_all: T*BD]
    // [db_alpha_float: dim] [db_float: dim] [d_d_h_float: dim]
    __nv_bfloat16* dh = workspace;
    __nv_bfloat16* dh_recurrent = workspace + BD;
    __nv_bfloat16* d_alpha_logit_all = workspace + 2 * BD;
    __nv_bfloat16* dWx_all = workspace + 2 * BD + TBD;
    float* db_alpha_float = reinterpret_cast<float*>(workspace + 2 * BD + 2 * TBD);
    float* db_float = db_alpha_float + dim_;
    float* d_d_h_float = db_float + dim_;

    // Zero out accumulators
    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(db_alpha_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(d_d_h_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW_alpha, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_x, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);

    // BPTT loop
    for (int t = steps - 1; t >= 0; --t) {
        const __nv_bfloat16* h_t = h + (t + 1) * BD;
        const __nv_bfloat16* h_prev = h + t * BD;
        const __nv_bfloat16* alpha_t = alpha_cache + t * BD;
        const __nv_bfloat16* v_pre_t = v_pre_cache + t * BD;
        const __nv_bfloat16* d_out_t = d_output + t * BD;
        __nv_bfloat16* d_alpha_logit_t = d_alpha_logit_all + t * BD;
        __nv_bfloat16* dWx_t = dWx_all + t * BD;

        // Backward through self-gate
        E65SelfGateBackward_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, d_out_t, dh);

        // Backward through gated update + diagonal h-dependence
        __nv_bfloat16* dh_recurrent_in = (t < steps - 1) ? dh_recurrent : nullptr;

        // Temporary buffer for dh_prev output this step
        __nv_bfloat16* dh_prev_temp = workspace;  // Reuse dh slot temporarily

        E65DiagonalHBackward_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_,
            alpha_t, v_pre_t, h_prev, d_h,
            dh, dh_recurrent_in,
            dh_prev_temp, d_alpha_logit_t, dWx_t,
            db_alpha_float, db_float, d_d_h_float);

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

    // dW_x = dWx_all.T @ x
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        dWx_all, dim_,
        &beta_one,
        dW_x, dim_);

    // =========================================================================
    // Batched GEMM for dx: dx = W_alpha @ d_alpha_logit + W_x @ dWx
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

    // dx += W_x @ dWx_all
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_x, dim_,
        dWx_all, dim_,
        &beta_one,
        dx, dim_);

    // Copy float gradients to output type
    CopyFloatToT_E65<__nv_bfloat16><<<(dim_ + 255) / 256, 256, 0, stream_>>>(
        dim_, db_alpha_float, db_alpha);
    CopyFloatToT_E65<__nv_bfloat16><<<(dim_ + 255) / 256, 256, 0, stream_>>>(
        dim_, db_float, db);
    CopyFloatToT_E65<__nv_bfloat16><<<(dim_ + 255) / 256, 256, 0, stream_>>>(
        dim_, d_d_h_float, d_d_h);
}

// =============================================================================
// Generic Template Implementations (float, half, double)
// =============================================================================

template<typename T>
E65DiagonalHForward<T>::E65DiagonalHForward(
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
void E65DiagonalHForward<T>::Run(
    int steps,
    const T* W_alpha,
    const T* b_alpha,
    const T* d_h,
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
    const int TBD = steps * BD;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    T* alpha_logits_all = workspace;
    T* Wx_all = workspace + TBD;

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
        W_x, dim_,
        x, dim_,
        &beta_zero,
        Wx_all, dim_);

    for (int t = 0; t < steps; ++t) {
        const T* alpha_logits_t = alpha_logits_all + t * BD;
        const T* Wx_t = Wx_all + t * BD;
        const T* h_prev = h + t * BD;
        T* h_t = h + (t + 1) * BD;
        T* out_t = output + t * BD;
        T* alpha_t = training_ ? (alpha_cache + t * BD) : nullptr;
        T* v_pre_t = training_ ? (v_pre_cache + t * BD) : nullptr;

        E65DiagonalHKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_,
            alpha_logits_t, Wx_t, h_prev,
            d_h, b_alpha, b,
            h_t, out_t, alpha_t, v_pre_t);
    }
}

template<typename T>
E65DiagonalHBackward<T>::E65DiagonalHBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void E65DiagonalHBackward<T>::Run(
    int steps,
    const T* W_alpha,
    const T* d_h,
    const T* W_x,
    const T* x,
    const T* h,
    const T* v_pre_cache,
    const T* alpha_cache,
    const T* d_output,
    T* dx,
    T* dW_alpha,
    T* db_alpha,
    T* d_d_h,
    T* dW_x,
    T* db,
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
    T* dWx_all = workspace + 2 * BD + TBD;
    float* db_alpha_float = reinterpret_cast<float*>(workspace + 2 * BD + 2 * TBD);
    float* db_float = db_alpha_float + dim_;
    float* d_d_h_float = db_float + dim_;

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(T), stream_);
    cudaMemsetAsync(db_alpha_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(d_d_h_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW_alpha, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dW_x, 0, dim_ * dim_ * sizeof(T), stream_);

    for (int t = steps - 1; t >= 0; --t) {
        const T* h_t = h + (t + 1) * BD;
        const T* h_prev = h + t * BD;
        const T* alpha_t = alpha_cache + t * BD;
        const T* v_pre_t = v_pre_cache + t * BD;
        const T* d_out_t = d_output + t * BD;
        T* d_alpha_logit_t = d_alpha_logit_all + t * BD;
        T* dWx_t = dWx_all + t * BD;

        E65SelfGateBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, d_out_t, dh);

        T* dh_recurrent_in = (t < steps - 1) ? dh_recurrent : nullptr;
        T* dh_prev_temp = workspace;

        E65DiagonalHBackwardKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_,
            alpha_t, v_pre_t, h_prev, d_h,
            dh, dh_recurrent_in,
            dh_prev_temp, d_alpha_logit_t, dWx_t,
            db_alpha_float, db_float, d_d_h_float);

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
        dWx_all, dim_,
        &beta_one,
        dW_x, dim_);

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
        W_x, dim_,
        dWx_all, dim_,
        &beta_one,
        dx, dim_);

    CopyFloatToT_E65<T><<<(dim_ + 255) / 256, 256, 0, stream_>>>(
        dim_, db_alpha_float, db_alpha);
    CopyFloatToT_E65<T><<<(dim_ + 255) / 256, 256, 0, stream_>>>(
        dim_, db_float, db);
    CopyFloatToT_E65<T><<<(dim_ + 255) / 256, 256, 0, stream_>>>(
        dim_, d_d_h_float, d_d_h);
}

// Explicit template instantiations
template struct E65DiagonalHForward<__half>;
template struct E65DiagonalHForward<float>;
template struct E65DiagonalHForward<double>;

template struct E65DiagonalHBackward<__half>;
template struct E65DiagonalHBackward<float>;
template struct E65DiagonalHBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
