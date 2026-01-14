// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E68: Self-Gating Elman - Multiplicative H-Dependence
//
// Architecture:
// alpha_t = sigmoid(x @ W_alpha.T + b_alpha)        # Retain gate (input-dependent)
// g_t = sigmoid(d_g * h_{t-1} + b_g)               # SELF-GATING: h gates the value!
// v_raw_t = tanh(x @ W_x.T + b_v)                  # Raw new value
// v_t = v_raw_t * g_t                              # Gated value
// h_t = alpha_t * h_{t-1} + (1 - alpha_t) * v_t    # Gated update
// output = h * silu(h)                             # Self-gating output
//
// Key Properties:
// - O(d) cost per timestep (no W_h matrix, just diagonal d_g)
// - UTM-class: h inside sigmoid creates nonlinear h-dependence
// - When h is large positive: sigmoid(d_g*h) -> 1, gate opens
// - When h is large negative: sigmoid(d_g*h) -> 0, gate closes
// - The stored state controls how much new info can be written (capacity-based gating)
//
// Key Optimization: Batch both projections upfront (W_alpha@x, W_x@x)

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
// Forward Kernel: Self-gating + decay-gated update + output self-gating
// Computes: g = sigmoid(d_g * h_prev + b_g)
//          v = tanh(v_raw + b_v) * g
//          h_new = alpha * h_prev + (1 - alpha) * v
//          output = h_new * silu(h_new)
// =============================================================================

__global__ void E68SelfGatingKernel_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ alpha_logits,   // Pre-sigmoid logits [B*D]
    const __nv_bfloat16* __restrict__ v_raw,          // Pre-tanh values [B*D]
    const __nv_bfloat16* __restrict__ h_prev,
    const __nv_bfloat16* __restrict__ b_alpha,        // [D]
    const __nv_bfloat16* __restrict__ d_g,            // [D] diagonal gating weights
    const __nv_bfloat16* __restrict__ b_g,            // [D] gating bias
    const __nv_bfloat16* __restrict__ b_v,            // [D] value bias
    __nv_bfloat16* __restrict__ h_out,
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ alpha_cache,          // Cache alpha for backward
    __nv_bfloat16* __restrict__ g_cache,              // Cache g for backward
    __nv_bfloat16* __restrict__ v_raw_tanh_cache) {   // Cache tanh(v_raw+b_v) for backward

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // Compute alpha = sigmoid(alpha_logits + b_alpha)
        float alpha_logit = __bfloat162float(alpha_logits[idx]) + __bfloat162float(b_alpha[d]);
        float alpha = 1.0f / (1.0f + __expf(-alpha_logit));

        // Compute g = sigmoid(d_g * h_prev + b_g) - SELF-GATING!
        float h_prev_val = __bfloat162float(h_prev[idx]);
        float d_g_val = __bfloat162float(d_g[d]);
        float b_g_val = __bfloat162float(b_g[d]);
        float g_logit = d_g_val * h_prev_val + b_g_val;
        float g = 1.0f / (1.0f + __expf(-g_logit));

        // Compute v = tanh(v_raw + b_v) * g
        float v_raw_val = __bfloat162float(v_raw[idx]) + __bfloat162float(b_v[d]);
        float v_raw_tanh = tanhf(v_raw_val);
        float v = v_raw_tanh * g;

        // Gated update: h = alpha * h_prev + (1 - alpha) * v
        float h_new = alpha * h_prev_val + (1.0f - alpha) * v;

        // Store hidden state
        h_out[idx] = __float2bfloat16(h_new);

        // Cache values for backward
        if (alpha_cache) alpha_cache[idx] = __float2bfloat16(alpha);
        if (g_cache) g_cache[idx] = __float2bfloat16(g);
        if (v_raw_tanh_cache) v_raw_tanh_cache[idx] = __float2bfloat16(v_raw_tanh);

        // Self-gate output: output = h * silu(h)
        float sigmoid_h = 1.0f / (1.0f + __expf(-h_new));
        float silu_h = h_new * sigmoid_h;
        output[idx] = __float2bfloat16(h_new * silu_h);
    }
}

template<typename T>
__global__ void E68SelfGatingKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ alpha_logits,
    const T* __restrict__ v_raw,
    const T* __restrict__ h_prev,
    const T* __restrict__ b_alpha,
    const T* __restrict__ d_g,
    const T* __restrict__ b_g,
    const T* __restrict__ b_v,
    T* __restrict__ h_out,
    T* __restrict__ output,
    T* __restrict__ alpha_cache,
    T* __restrict__ g_cache,
    T* __restrict__ v_raw_tanh_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float alpha_logit = static_cast<float>(alpha_logits[idx]) + static_cast<float>(b_alpha[d]);
        float alpha = 1.0f / (1.0f + expf(-alpha_logit));

        float h_prev_val = static_cast<float>(h_prev[idx]);
        float d_g_val = static_cast<float>(d_g[d]);
        float b_g_val = static_cast<float>(b_g[d]);
        float g_logit = d_g_val * h_prev_val + b_g_val;
        float g = 1.0f / (1.0f + expf(-g_logit));

        float v_raw_val = static_cast<float>(v_raw[idx]) + static_cast<float>(b_v[d]);
        float v_raw_tanh = tanhf(v_raw_val);
        float v = v_raw_tanh * g;

        float h_new = alpha * h_prev_val + (1.0f - alpha) * v;

        h_out[idx] = static_cast<T>(h_new);

        if (alpha_cache) alpha_cache[idx] = static_cast<T>(alpha);
        if (g_cache) g_cache[idx] = static_cast<T>(g);
        if (v_raw_tanh_cache) v_raw_tanh_cache[idx] = static_cast<T>(v_raw_tanh);

        float sigmoid_h = 1.0f / (1.0f + expf(-h_new));
        float silu_h = h_new * sigmoid_h;
        output[idx] = static_cast<T>(h_new * silu_h);
    }
}

// =============================================================================
// Backward Kernels
// =============================================================================

// Backward through self-gate output: output = h * silu(h) = h^2 * sigmoid(h)
// d(output)/dh = silu(h) * (2 + h*(1-sigmoid(h)))
__global__ void E68SelfGateOutputBackward_BF16(
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
__global__ void E68SelfGateOutputBackward(
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

// Backward through the self-gating recurrence:
// h = alpha * h_prev + (1 - alpha) * v
// where v = v_raw_tanh * g
// and g = sigmoid(d_g * h_prev + b_g)
//
// Gradients:
// dh_prev = dh * alpha + dh * (1-alpha) * v_raw_tanh * g * (1-g) * d_g  (through g)
// dv_raw_tanh = dh * (1 - alpha) * g
// dg = dh * (1 - alpha) * v_raw_tanh
// d_alpha = dh * (h_prev - v)
// d_alpha_logit = d_alpha * alpha * (1 - alpha)
__global__ void E68SelfGatingRecurrenceBackwardKernel_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ alpha,          // Cached
    const __nv_bfloat16* __restrict__ g,              // Cached
    const __nv_bfloat16* __restrict__ v_raw_tanh,     // Cached
    const __nv_bfloat16* __restrict__ h_prev,
    const __nv_bfloat16* __restrict__ d_g,            // [D] diagonal weights
    const __nv_bfloat16* __restrict__ dh,             // Gradient from output self-gate
    const __nv_bfloat16* __restrict__ dh_recurrent,   // Recurrent gradient from next step
    __nv_bfloat16* __restrict__ dh_prev,              // Output: gradient to previous h
    __nv_bfloat16* __restrict__ d_alpha_logit,        // Output: gradient to alpha logits
    __nv_bfloat16* __restrict__ dv_raw,               // Output: gradient to v_raw (pre-tanh)
    float* __restrict__ db_alpha,                     // Accumulated bias gradient
    float* __restrict__ dd_g,                         // Accumulated d_g gradient
    float* __restrict__ db_g,                         // Accumulated b_g gradient
    float* __restrict__ db_v) {                       // Accumulated v bias gradient

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float alpha_val = __bfloat162float(alpha[idx]);
        float g_val = __bfloat162float(g[idx]);
        float v_raw_tanh_val = __bfloat162float(v_raw_tanh[idx]);
        float h_prev_val = __bfloat162float(h_prev[idx]);
        float d_g_val = __bfloat162float(d_g[d]);

        // Combine gradients
        float grad = __bfloat162float(dh[idx]);
        if (dh_recurrent) {
            grad += __bfloat162float(dh_recurrent[idx]);
        }

        // v = v_raw_tanh * g
        float v_val = v_raw_tanh_val * g_val;

        // Gradient to h_prev from direct path: dh * alpha
        float dh_prev_direct = grad * alpha_val;

        // Gradient through g: g = sigmoid(d_g * h_prev + b_g)
        // dv/dg = v_raw_tanh
        // dh/dv = (1 - alpha)
        // dg/dh_prev = g * (1-g) * d_g
        float dv_val = grad * (1.0f - alpha_val);  // dh/dv
        float dg_val = dv_val * v_raw_tanh_val;    // dv/dg * dh/dv
        float dg_logit = dg_val * g_val * (1.0f - g_val);  // sigmoid derivative
        float dh_prev_through_g = dg_logit * d_g_val;

        // Total dh_prev
        dh_prev[idx] = __float2bfloat16(dh_prev_direct + dh_prev_through_g);

        // Gradient to v_raw_tanh: dh * (1 - alpha) * g
        float dv_raw_tanh_val = dv_val * g_val;
        // Gradient through tanh: d(tanh(x))/dx = 1 - tanh^2(x)
        float dv_raw_val = dv_raw_tanh_val * (1.0f - v_raw_tanh_val * v_raw_tanh_val);
        dv_raw[idx] = __float2bfloat16(dv_raw_val);

        // Gradient to alpha: d_alpha = dh * (h_prev - v)
        float d_alpha = grad * (h_prev_val - v_val);
        float d_alpha_logit_val = d_alpha * alpha_val * (1.0f - alpha_val);
        d_alpha_logit[idx] = __float2bfloat16(d_alpha_logit_val);

        // Accumulate bias/parameter gradients
        atomicAdd(&db_alpha[d], d_alpha_logit_val);
        atomicAdd(&dd_g[d], dg_logit * h_prev_val);  // d(g_logit)/d(d_g) = h_prev
        atomicAdd(&db_g[d], dg_logit);               // d(g_logit)/d(b_g) = 1
        atomicAdd(&db_v[d], dv_raw_val);
    }
}

template<typename T>
__global__ void E68SelfGatingRecurrenceBackwardKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ alpha,
    const T* __restrict__ g,
    const T* __restrict__ v_raw_tanh,
    const T* __restrict__ h_prev,
    const T* __restrict__ d_g,
    const T* __restrict__ dh,
    const T* __restrict__ dh_recurrent,
    T* __restrict__ dh_prev,
    T* __restrict__ d_alpha_logit,
    T* __restrict__ dv_raw,
    float* __restrict__ db_alpha,
    float* __restrict__ dd_g,
    float* __restrict__ db_g,
    float* __restrict__ db_v) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float alpha_val = static_cast<float>(alpha[idx]);
        float g_val = static_cast<float>(g[idx]);
        float v_raw_tanh_val = static_cast<float>(v_raw_tanh[idx]);
        float h_prev_val = static_cast<float>(h_prev[idx]);
        float d_g_val = static_cast<float>(d_g[d]);

        float grad = static_cast<float>(dh[idx]);
        if (dh_recurrent) {
            grad += static_cast<float>(dh_recurrent[idx]);
        }

        float v_val = v_raw_tanh_val * g_val;

        float dh_prev_direct = grad * alpha_val;
        float dv_val = grad * (1.0f - alpha_val);
        float dg_val = dv_val * v_raw_tanh_val;
        float dg_logit = dg_val * g_val * (1.0f - g_val);
        float dh_prev_through_g = dg_logit * d_g_val;

        dh_prev[idx] = static_cast<T>(dh_prev_direct + dh_prev_through_g);

        float dv_raw_tanh_val = dv_val * g_val;
        float dv_raw_val = dv_raw_tanh_val * (1.0f - v_raw_tanh_val * v_raw_tanh_val);
        dv_raw[idx] = static_cast<T>(dv_raw_val);

        float d_alpha = grad * (h_prev_val - v_val);
        float d_alpha_logit_val = d_alpha * alpha_val * (1.0f - alpha_val);
        d_alpha_logit[idx] = static_cast<T>(d_alpha_logit_val);

        atomicAdd(&db_alpha[d], d_alpha_logit_val);
        atomicAdd(&dd_g[d], dg_logit * h_prev_val);
        atomicAdd(&db_g[d], dg_logit);
        atomicAdd(&db_v[d], dv_raw_val);
    }
}

// =============================================================================
// Utility Kernels
// =============================================================================

template<typename T>
__global__ void CopyFloatToT_E68(const int n, const float* __restrict__ src, T* __restrict__ dst) {
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
// E68 Self-Gating Forward - BF16 Specialization
// =============================================================================

template<>
E68SelfGatingForward<__nv_bfloat16>::E68SelfGatingForward(
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
void E68SelfGatingForward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_alpha,   // [dim, dim]
    const __nv_bfloat16* b_alpha,   // [dim]
    const __nv_bfloat16* W_x,       // [dim, dim]
    const __nv_bfloat16* b_v,       // [dim]
    const __nv_bfloat16* d_g,       // [dim] diagonal gating weights
    const __nv_bfloat16* b_g,       // [dim] gating bias
    const __nv_bfloat16* x,         // [T, B, dim]
    __nv_bfloat16* h,               // [T+1, B, dim]
    __nv_bfloat16* output,          // [T, B, dim]
    __nv_bfloat16* alpha_cache,     // [T, B, dim] for backward
    __nv_bfloat16* g_cache,         // [T, B, dim] for backward
    __nv_bfloat16* v_raw_tanh_cache, // [T, B, dim] for backward
    __nv_bfloat16* workspace) {     // [2*T*B*dim] for alpha_logits and v_raw

    static const __nv_bfloat16 alpha_one = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);

    const int BD = batch_size_ * dim_;
    const int TBD = steps * BD;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Workspace layout: [alpha_logits: T*BD] [v_raw: T*BD]
    __nv_bfloat16* alpha_logits_all = workspace;
    __nv_bfloat16* v_raw_all = workspace + TBD;

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

    // Batch GEMM: v_raw = x @ W_x.T  (no bias yet, added in kernel)
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_x, dim_,
        x, dim_,
        &beta_zero,
        v_raw_all, dim_);

    // Process each timestep sequentially (recurrence)
    for (int t = 0; t < steps; ++t) {
        const __nv_bfloat16* alpha_logits_t = alpha_logits_all + t * BD;
        const __nv_bfloat16* v_raw_t = v_raw_all + t * BD;
        const __nv_bfloat16* h_prev = h + t * BD;
        __nv_bfloat16* h_t = h + (t + 1) * BD;
        __nv_bfloat16* out_t = output + t * BD;
        __nv_bfloat16* alpha_t = training_ ? (alpha_cache + t * BD) : nullptr;
        __nv_bfloat16* g_t = training_ ? (g_cache + t * BD) : nullptr;
        __nv_bfloat16* v_raw_tanh_t = training_ ? (v_raw_tanh_cache + t * BD) : nullptr;

        E68SelfGatingKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_,
            alpha_logits_t, v_raw_t, h_prev,
            b_alpha, d_g, b_g, b_v,
            h_t, out_t, alpha_t, g_t, v_raw_tanh_t);
    }
}

// =============================================================================
// E68 Self-Gating Backward - BF16 Specialization
// =============================================================================

template<>
E68SelfGatingBackward<__nv_bfloat16>::E68SelfGatingBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void E68SelfGatingBackward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_alpha,
    const __nv_bfloat16* W_x,
    const __nv_bfloat16* d_g,
    const __nv_bfloat16* x,
    const __nv_bfloat16* h,
    const __nv_bfloat16* alpha_cache,
    const __nv_bfloat16* g_cache,
    const __nv_bfloat16* v_raw_tanh_cache,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* dx,
    __nv_bfloat16* dW_alpha,
    __nv_bfloat16* db_alpha,
    __nv_bfloat16* dW_x,
    __nv_bfloat16* db_v,
    __nv_bfloat16* dd_g_out,
    __nv_bfloat16* db_g,
    __nv_bfloat16* workspace) {

    static const __nv_bfloat16 alpha_one = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);
    static const __nv_bfloat16 beta_one = __float2bfloat16(1.0f);

    const int BD = batch_size_ * dim_;
    const int TBD = steps * BD;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Workspace layout:
    // [dh: BD] [dh_recurrent: BD] [d_alpha_logit_all: T*BD] [dv_raw_all: T*BD]
    // [db_alpha_float: dim] [dd_g_float: dim] [db_g_float: dim] [db_v_float: dim]
    __nv_bfloat16* dh = workspace;
    __nv_bfloat16* dh_recurrent = workspace + BD;
    __nv_bfloat16* d_alpha_logit_all = workspace + 2 * BD;
    __nv_bfloat16* dv_raw_all = workspace + 2 * BD + TBD;
    float* db_alpha_float = reinterpret_cast<float*>(workspace + 2 * BD + 2 * TBD);
    float* dd_g_float = db_alpha_float + dim_;
    float* db_g_float = dd_g_float + dim_;
    float* db_v_float = db_g_float + dim_;

    // Zero out accumulators
    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(db_alpha_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dd_g_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_g_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_v_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW_alpha, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_x, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);

    // BPTT loop
    for (int t = steps - 1; t >= 0; --t) {
        const __nv_bfloat16* h_t = h + (t + 1) * BD;
        const __nv_bfloat16* h_prev = h + t * BD;
        const __nv_bfloat16* alpha_t = alpha_cache + t * BD;
        const __nv_bfloat16* g_t = g_cache + t * BD;
        const __nv_bfloat16* v_raw_tanh_t = v_raw_tanh_cache + t * BD;
        const __nv_bfloat16* d_out_t = d_output + t * BD;
        __nv_bfloat16* d_alpha_logit_t = d_alpha_logit_all + t * BD;
        __nv_bfloat16* dv_raw_t = dv_raw_all + t * BD;

        // Backward through self-gate output
        E68SelfGateOutputBackward_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, d_out_t, dh);

        // Backward through self-gating recurrence
        __nv_bfloat16* dh_recurrent_in = (t < steps - 1) ? dh_recurrent : nullptr;
        __nv_bfloat16* dh_prev_temp = workspace;  // Reuse dh slot temporarily

        E68SelfGatingRecurrenceBackwardKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_,
            alpha_t, g_t, v_raw_tanh_t, h_prev, d_g,
            dh, dh_recurrent_in,
            dh_prev_temp, d_alpha_logit_t, dv_raw_t,
            db_alpha_float, dd_g_float, db_g_float, db_v_float);

        // Copy dh_prev to dh_recurrent for next iteration
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

    // dW_x = dv_raw_all.T @ x
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        dv_raw_all, dim_,
        &beta_one,
        dW_x, dim_);

    // =========================================================================
    // Batched GEMM for dx: dx = W_alpha @ d_alpha_logit + W_x @ dv_raw
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

    // dx += W_x @ dv_raw_all
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_x, dim_,
        dv_raw_all, dim_,
        &beta_one,
        dx, dim_);

    // Copy float gradients to output type
    CopyFloatToT_E68<__nv_bfloat16><<<(dim_ + 255) / 256, 256, 0, stream_>>>(
        dim_, db_alpha_float, db_alpha);
    CopyFloatToT_E68<__nv_bfloat16><<<(dim_ + 255) / 256, 256, 0, stream_>>>(
        dim_, dd_g_float, dd_g_out);
    CopyFloatToT_E68<__nv_bfloat16><<<(dim_ + 255) / 256, 256, 0, stream_>>>(
        dim_, db_g_float, db_g);
    CopyFloatToT_E68<__nv_bfloat16><<<(dim_ + 255) / 256, 256, 0, stream_>>>(
        dim_, db_v_float, db_v);
}

// =============================================================================
// Generic Template Implementations (float, half, double)
// =============================================================================

template<typename T>
E68SelfGatingForward<T>::E68SelfGatingForward(
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
void E68SelfGatingForward<T>::Run(
    int steps,
    const T* W_alpha,
    const T* b_alpha,
    const T* W_x,
    const T* b_v,
    const T* d_g,
    const T* b_g,
    const T* x,
    T* h,
    T* output,
    T* alpha_cache,
    T* g_cache,
    T* v_raw_tanh_cache,
    T* workspace) {

    static const T alpha_one = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int BD = batch_size_ * dim_;
    const int TBD = steps * BD;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    T* alpha_logits_all = workspace;
    T* v_raw_all = workspace + TBD;

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
        v_raw_all, dim_);

    for (int t = 0; t < steps; ++t) {
        const T* alpha_logits_t = alpha_logits_all + t * BD;
        const T* v_raw_t = v_raw_all + t * BD;
        const T* h_prev = h + t * BD;
        T* h_t = h + (t + 1) * BD;
        T* out_t = output + t * BD;
        T* alpha_t = training_ ? (alpha_cache + t * BD) : nullptr;
        T* g_t = training_ ? (g_cache + t * BD) : nullptr;
        T* v_raw_tanh_t = training_ ? (v_raw_tanh_cache + t * BD) : nullptr;

        E68SelfGatingKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_,
            alpha_logits_t, v_raw_t, h_prev,
            b_alpha, d_g, b_g, b_v,
            h_t, out_t, alpha_t, g_t, v_raw_tanh_t);
    }
}

template<typename T>
E68SelfGatingBackward<T>::E68SelfGatingBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void E68SelfGatingBackward<T>::Run(
    int steps,
    const T* W_alpha,
    const T* W_x,
    const T* d_g,
    const T* x,
    const T* h,
    const T* alpha_cache,
    const T* g_cache,
    const T* v_raw_tanh_cache,
    const T* d_output,
    T* dx,
    T* dW_alpha,
    T* db_alpha,
    T* dW_x,
    T* db_v,
    T* dd_g_out,
    T* db_g,
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
    T* dv_raw_all = workspace + 2 * BD + TBD;
    float* db_alpha_float = reinterpret_cast<float*>(workspace + 2 * BD + 2 * TBD);
    float* dd_g_float = db_alpha_float + dim_;
    float* db_g_float = dd_g_float + dim_;
    float* db_v_float = db_g_float + dim_;

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(T), stream_);
    cudaMemsetAsync(db_alpha_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dd_g_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_g_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_v_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW_alpha, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dW_x, 0, dim_ * dim_ * sizeof(T), stream_);

    for (int t = steps - 1; t >= 0; --t) {
        const T* h_t = h + (t + 1) * BD;
        const T* h_prev = h + t * BD;
        const T* alpha_t = alpha_cache + t * BD;
        const T* g_t = g_cache + t * BD;
        const T* v_raw_tanh_t = v_raw_tanh_cache + t * BD;
        const T* d_out_t = d_output + t * BD;
        T* d_alpha_logit_t = d_alpha_logit_all + t * BD;
        T* dv_raw_t = dv_raw_all + t * BD;

        E68SelfGateOutputBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, d_out_t, dh);

        T* dh_recurrent_in = (t < steps - 1) ? dh_recurrent : nullptr;
        T* dh_prev_temp = workspace;

        E68SelfGatingRecurrenceBackwardKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_,
            alpha_t, g_t, v_raw_tanh_t, h_prev, d_g,
            dh, dh_recurrent_in,
            dh_prev_temp, d_alpha_logit_t, dv_raw_t,
            db_alpha_float, dd_g_float, db_g_float, db_v_float);

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
        dv_raw_all, dim_,
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
        dv_raw_all, dim_,
        &beta_one,
        dx, dim_);

    CopyFloatToT_E68<T><<<(dim_ + 255) / 256, 256, 0, stream_>>>(
        dim_, db_alpha_float, db_alpha);
    CopyFloatToT_E68<T><<<(dim_ + 255) / 256, 256, 0, stream_>>>(
        dim_, dd_g_float, dd_g_out);
    CopyFloatToT_E68<T><<<(dim_ + 255) / 256, 256, 0, stream_>>>(
        dim_, db_g_float, db_g);
    CopyFloatToT_E68<T><<<(dim_ + 255) / 256, 256, 0, stream_>>>(
        dim_, db_v_float, db_v);
}

// Explicit template instantiations
template struct E68SelfGatingForward<__half>;
template struct E68SelfGatingForward<float>;
template struct E68SelfGatingForward<double>;

template struct E68SelfGatingBackward<__half>;
template struct E68SelfGatingBackward<float>;
template struct E68SelfGatingBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
