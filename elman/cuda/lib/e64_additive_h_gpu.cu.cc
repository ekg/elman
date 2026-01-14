// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E64: Additive H-Dependence - Cheapest UTM-Class Recurrence
//
// The simplest way to get h into the nonlinearity:
//    v_t = tanh(h_{t-1} + W_x @ x_t + b)
//
// Cost: O(d) per step vs O(d^2) for E63's W_h @ h
// Still UTM-class: h is inside the tanh!
//
// Architecture:
// alpha_t = sigmoid(W_alpha @ x_t + b_alpha)    # Retain gate (x-only)
// v_t = tanh(h_{t-1} + W_x @ x_t + b)          # ADDITIVE h-dependence (no W_h!)
// h_t = alpha_t * h_{t-1} + (1 - alpha_t) * v_t  # Gated mixing
// output = h * silu(h)                          # Self-gating
//
// Key insight: h added directly to tanh is O(d) vs O(d^2) for W @ h,
// but still provides nonlinear h-dependence for UTM expressivity.
//
// Jacobian:
//    dh_t/dh_{t-1} = diag(alpha) + diag((1-alpha) * (1 - v^2))
//                  = diag(alpha + (1-alpha)*(1-v^2))
//
// This is DIAGONAL - very efficient for both forward and backward!
// But: no cross-dimension mixing through h. x provides mixing via W_x.
//
// Key optimization:
// - Batch W_alpha @ x and W_x @ x for ALL timesteps upfront (two big GEMMs)
// - Per-timestep: only element-wise ops (O(d), no GEMM!)
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
// Forward Kernel: Fused gate + additive value + update + self-gate
// Input: alpha_x = W_alpha @ x_t (pre-batched), Wx = W_x @ x_t (pre-batched)
// Output: h_new, output, and optionally save v_pre (for backward)
//
// Key difference from E63: NO per-step GEMM! Just element-wise h_prev + Wx
// =============================================================================

__global__ void E64FusedForwardKernel_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ alpha_x,    // [B, D] pre-computed W_alpha @ x + b_alpha
    const __nv_bfloat16* __restrict__ Wx,         // [B, D] pre-computed W_x @ x + b
    const __nv_bfloat16* __restrict__ h_prev,     // [B, D] previous hidden state
    __nv_bfloat16* __restrict__ h_out,            // [B, D] new hidden state
    __nv_bfloat16* __restrict__ output,           // [B, D] output (h * silu(h))
    __nv_bfloat16* __restrict__ v_pre_cache,      // [B, D] pre-tanh value cache (for backward)
    __nv_bfloat16* __restrict__ alpha_cache) {    // [B, D] alpha cache (for backward)

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        // Alpha gate: sigmoid(alpha_x) - alpha_x already has bias
        float alpha_val = __bfloat162float(alpha_x[idx]);
        float alpha = 1.0f / (1.0f + __expf(-alpha_val));

        // Value pre-activation: h_prev + Wx (ADDITIVE h-dependence!)
        // Note: Wx already includes bias from batched computation
        float h_prev_val = __bfloat162float(h_prev[idx]);
        float v_pre = h_prev_val + __bfloat162float(Wx[idx]);

        // Nonlinear value: tanh(v_pre)
        float v = tanhf(v_pre);

        // Gated update: h_new = alpha * h_prev + (1 - alpha) * v
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
__global__ void E64FusedForwardKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ alpha_x,
    const T* __restrict__ Wx,
    const T* __restrict__ h_prev,
    T* __restrict__ h_out,
    T* __restrict__ output,
    T* __restrict__ v_pre_cache,
    T* __restrict__ alpha_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        // Alpha gate
        float alpha_val = static_cast<float>(alpha_x[idx]);
        float alpha = 1.0f / (1.0f + expf(-alpha_val));

        // Value pre-activation: ADDITIVE h-dependence
        float h_prev_val = static_cast<float>(h_prev[idx]);
        float v_pre = h_prev_val + static_cast<float>(Wx[idx]);

        // Nonlinear value
        float v = tanhf(v_pre);

        // Gated update
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
__global__ void E64SelfGateBackward_BF16(
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
__global__ void E64SelfGateBackward(
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

// Phase 2: Backward through gated update and ADDITIVE value computation
// h_t = alpha * h_prev + (1 - alpha) * tanh(h_prev + Wx)
//
// Gradient derivation:
// Let v_pre = h_prev + Wx, v = tanh(v_pre)
// h_t = alpha * h_prev + (1 - alpha) * v
//
// dh_prev = dh * alpha                           (direct path through alpha*h_prev)
//         + dh * (1 - alpha) * (1 - v^2)         (path through tanh: dv_pre * d(h_prev)/d(v_pre))
//         = dh * (alpha + (1 - alpha) * (1 - v^2))
//
// dWx_pre = dh * (1 - alpha) * (1 - v^2)         (same as dv_pre)
//
// dalpha = dh * (h_prev - v)
// dalpha_x = dalpha * alpha * (1 - alpha)        (sigmoid derivative)
__global__ void E64GatedUpdateBackward_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ h_prev,        // [B, D] h_{t-1}
    const __nv_bfloat16* __restrict__ v_pre,         // [B, D] pre-tanh value (h_prev + Wx)
    const __nv_bfloat16* __restrict__ alpha_cache,   // [B, D] sigmoid(alpha_x)
    const __nv_bfloat16* __restrict__ dh,            // [B, D] gradient from output (dh_t)
    const __nv_bfloat16* __restrict__ dh_recurrent,  // [B, D] gradient from next timestep (or null)
    __nv_bfloat16* __restrict__ dh_prev,             // [B, D] gradient to previous h
    __nv_bfloat16* __restrict__ dWx_pre,             // [B, D] gradient through Wx (for W_x grad)
    __nv_bfloat16* __restrict__ dalpha_x,            // [B, D] gradient through alpha (for W_alpha grad)
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
        // where v = tanh(h_prev + Wx)

        // tanh derivative: dv/dv_pre = 1 - v^2
        float tanh_grad = 1.0f - v * v;

        // dv_pre = dh_total * (1 - alpha) * tanh_grad
        float dv_pre = dh_total * (1.0f - alpha) * tanh_grad;

        // dh_prev has two paths:
        // 1. Through alpha*h_prev: dh_total * alpha
        // 2. Through tanh(h_prev + Wx): dv_pre * 1 (since d(v_pre)/d(h_prev) = 1)
        float dh_prev_val = dh_total * alpha + dv_pre;
        dh_prev[idx] = __float2bfloat16(dh_prev_val);

        // dWx_pre = dv_pre (gradient through Wx path)
        dWx_pre[idx] = __float2bfloat16(dv_pre);

        // dalpha = dh_total * (h_prev - v)
        float dalpha = dh_total * (h_prev_val - v);

        // dalpha_x = dalpha * sigmoid'(alpha_x) = dalpha * alpha * (1 - alpha)
        float dalpha_x_val = dalpha * alpha * (1.0f - alpha);
        dalpha_x[idx] = __float2bfloat16(dalpha_x_val);

        // Accumulate bias gradient (dWx_pre flows through bias)
        atomicAdd(&db_accum[d], dv_pre);
    }
}

template<typename T>
__global__ void E64GatedUpdateBackward(
    const int batch_size,
    const int dim,
    const T* __restrict__ h_prev,
    const T* __restrict__ v_pre,
    const T* __restrict__ alpha_cache,
    const T* __restrict__ dh,
    const T* __restrict__ dh_recurrent,
    T* __restrict__ dh_prev,
    T* __restrict__ dWx_pre,
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

        float tanh_grad = 1.0f - v * v;
        float dv_pre = dh_total * (1.0f - alpha) * tanh_grad;

        float dh_prev_val = dh_total * alpha + dv_pre;
        dh_prev[idx] = static_cast<T>(dh_prev_val);

        dWx_pre[idx] = static_cast<T>(dv_pre);

        float dalpha = dh_total * (h_prev_val - v);
        float dalpha_x_val = dalpha * alpha * (1.0f - alpha);
        dalpha_x[idx] = static_cast<T>(dalpha_x_val);

        atomicAdd(&db_accum[d], dv_pre);
    }
}

// Utility: Copy vector
template<typename T>
__global__ void VectorCopy_E64(const int n, const T* __restrict__ src, T* __restrict__ dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// Copy float to T
template<typename T>
__global__ void CopyFloatToT_E64(const int n, const float* __restrict__ src, T* __restrict__ dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = static_cast<T>(src[idx]);
    }
}

// Add bias to projected values (element-wise)
__global__ void AddBias_E64_BF16(
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

template<typename T>
__global__ void AddBias_E64(
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

// Accumulate db_alpha from dalpha_x
__global__ void AccumulateDbAlpha_E64_BF16(
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

template<typename T>
__global__ void AccumulateDbAlpha_E64(
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
// E64 Additive H Forward - BF16 Specialization
// =============================================================================

template<>
E64AdditiveHForward<__nv_bfloat16>::E64AdditiveHForward(
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
void E64AdditiveHForward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_alpha,  // [dim, dim]
    const __nv_bfloat16* b_alpha,  // [dim]
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
    const int TBD = steps * BD;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Workspace layout:
    // [alpha_x_all: T*BD] - pre-computed W_alpha @ x + b_alpha
    // [Wx_all: T*BD]      - pre-computed W_x @ x + b
    __nv_bfloat16* alpha_x_all = workspace;
    __nv_bfloat16* Wx_all = workspace + TBD;

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
    const int total_elements = TBD;
    const int bias_blocks = (total_elements + block_size - 1) / block_size;
    AddBias_E64_BF16<<<bias_blocks, block_size, 0, stream_>>>(total_elements, dim_, alpha_x_all, b_alpha);

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

    // Add bias to Wx_all
    AddBias_E64_BF16<<<bias_blocks, block_size, 0, stream_>>>(total_elements, dim_, Wx_all, b);

    // =========================================================================
    // Sequential timestep loop (cannot parallelize due to h-dependence in v_t)
    // KEY ADVANTAGE: No per-step GEMM! Just element-wise ops.
    // =========================================================================
    for (int t = 0; t < steps; ++t) {
        const __nv_bfloat16* alpha_x_t = alpha_x_all + t * BD;
        const __nv_bfloat16* Wx_t = Wx_all + t * BD;
        const __nv_bfloat16* h_prev = h + t * BD;
        __nv_bfloat16* h_t = h + (t + 1) * BD;
        __nv_bfloat16* out_t = output + t * BD;
        __nv_bfloat16* v_pre_t = training_ ? (v_pre_cache + t * BD) : nullptr;
        __nv_bfloat16* alpha_t = training_ ? (alpha_cache + t * BD) : nullptr;

        // Fused forward kernel - NO GEMM, just element-wise!
        E64FusedForwardKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, alpha_x_t, Wx_t, h_prev, h_t, out_t, v_pre_t, alpha_t);
    }
}

// =============================================================================
// E64 Additive H Backward - BF16 Specialization
// =============================================================================

template<>
E64AdditiveHBackward<__nv_bfloat16>::E64AdditiveHBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void E64AdditiveHBackward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_alpha,
    const __nv_bfloat16* W_x,
    const __nv_bfloat16* x,
    const __nv_bfloat16* h,
    const __nv_bfloat16* v_pre_cache,
    const __nv_bfloat16* alpha_cache,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* dx,
    __nv_bfloat16* dW_alpha,
    __nv_bfloat16* db_alpha,
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
    // [dh: BD]              - gradient after self-gate
    // [dh_recurrent: BD]    - gradient from next timestep
    // [dh_prev: BD]         - gradient to previous h
    // [dWx_pre_all: T*BD]   - gradient through Wx for all timesteps
    // [dalpha_x_all: T*BD]  - gradient through alpha for all timesteps
    // [db_float: dim]       - float accumulator for bias
    // [db_alpha_float: dim] - float accumulator for alpha bias
    __nv_bfloat16* dh = workspace;
    __nv_bfloat16* dh_recurrent = workspace + BD;
    __nv_bfloat16* dh_prev = workspace + 2 * BD;
    __nv_bfloat16* dWx_pre_all = workspace + 3 * BD;
    __nv_bfloat16* dalpha_x_all = workspace + 3 * BD + TBD;
    float* db_float = reinterpret_cast<float*>(workspace + 3 * BD + 2 * TBD);
    float* db_alpha_float = db_float + dim_;

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_alpha_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW_alpha, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_x, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);

    // =========================================================================
    // BPTT Loop (backward through time)
    // =========================================================================
    for (int t = steps - 1; t >= 0; --t) {
        const __nv_bfloat16* h_t = h + (t + 1) * BD;       // h[t+1]
        const __nv_bfloat16* h_prev_t = h + t * BD;        // h[t]
        const __nv_bfloat16* d_out_t = d_output + t * BD;
        const __nv_bfloat16* v_pre_t = v_pre_cache + t * BD;
        const __nv_bfloat16* alpha_t = alpha_cache + t * BD;
        __nv_bfloat16* dWx_pre_t = dWx_pre_all + t * BD;
        __nv_bfloat16* dalpha_x_t = dalpha_x_all + t * BD;

        // Step 1: Backward through self-gate
        E64SelfGateBackward_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, d_out_t, dh);

        // Step 2: Backward through gated update
        // Note: dh_prev already combines both gradient paths (alpha*h and tanh path)
        E64GatedUpdateBackward_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_prev_t, v_pre_t, alpha_t,
            dh, dh_recurrent, dh_prev, dWx_pre_t, dalpha_x_t, db_float);

        // Step 3: Copy dh_prev to dh_recurrent for next iteration
        if (t > 0) {
            VectorCopy_E64<__nv_bfloat16><<<num_blocks, block_size, 0, stream_>>>(BD, dh_prev, dh_recurrent);
        }
    }

    // =========================================================================
    // Weight gradients via batched GEMMs
    // =========================================================================

    // dW_x = sum_t x[t].T @ dWx_pre[t]
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        dWx_pre_all, dim_,
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

    // dx = dWx_pre @ W_x + dalpha_x @ W_alpha
    // First: dx = dWx_pre @ W_x
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_x, dim_,
        dWx_pre_all, dim_,
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
    const int total_elements = TBD;
    const int alpha_blocks = (total_elements + block_size - 1) / block_size;
    AccumulateDbAlpha_E64_BF16<<<alpha_blocks, block_size, 0, stream_>>>(
        total_elements, dim_, dalpha_x_all, db_alpha_float);

    // Copy float accumulators to output
    CopyFloatToT_E64<__nv_bfloat16><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_float, db);
    CopyFloatToT_E64<__nv_bfloat16><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_alpha_float, db_alpha);
}

// =============================================================================
// Generic Template Implementations (float, half, double)
// =============================================================================

template<typename T>
E64AdditiveHForward<T>::E64AdditiveHForward(
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
void E64AdditiveHForward<T>::Run(
    int steps,
    const T* W_alpha,
    const T* b_alpha,
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

    T* alpha_x_all = workspace;
    T* Wx_all = workspace + TBD;

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
    const int total_elements = TBD;
    const int bias_blocks = (total_elements + block_size - 1) / block_size;
    AddBias_E64<T><<<bias_blocks, block_size, 0, stream_>>>(total_elements, dim_, alpha_x_all, b_alpha);

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

    // Add value bias
    AddBias_E64<T><<<bias_blocks, block_size, 0, stream_>>>(total_elements, dim_, Wx_all, b);

    // Sequential loop - NO per-step GEMM!
    for (int t = 0; t < steps; ++t) {
        const T* alpha_x_t = alpha_x_all + t * BD;
        const T* Wx_t = Wx_all + t * BD;
        const T* h_prev = h + t * BD;
        T* h_t = h + (t + 1) * BD;
        T* out_t = output + t * BD;
        T* v_pre_t = training_ ? (v_pre_cache + t * BD) : nullptr;
        T* alpha_t = training_ ? (alpha_cache + t * BD) : nullptr;

        E64FusedForwardKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, alpha_x_t, Wx_t, h_prev, h_t, out_t, v_pre_t, alpha_t);
    }
}

template<typename T>
E64AdditiveHBackward<T>::E64AdditiveHBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void E64AdditiveHBackward<T>::Run(
    int steps,
    const T* W_alpha,
    const T* W_x,
    const T* x,
    const T* h,
    const T* v_pre_cache,
    const T* alpha_cache,
    const T* d_output,
    T* dx,
    T* dW_alpha,
    T* db_alpha,
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
    T* dh_prev = workspace + 2 * BD;
    T* dWx_pre_all = workspace + 3 * BD;
    T* dalpha_x_all = workspace + 3 * BD + TBD;
    float* db_float = reinterpret_cast<float*>(workspace + 3 * BD + 2 * TBD);
    float* db_alpha_float = db_float + dim_;

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(T), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_alpha_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW_alpha, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dW_x, 0, dim_ * dim_ * sizeof(T), stream_);

    // BPTT loop
    for (int t = steps - 1; t >= 0; --t) {
        const T* h_t = h + (t + 1) * BD;
        const T* h_prev_t = h + t * BD;
        const T* d_out_t = d_output + t * BD;
        const T* v_pre_t = v_pre_cache + t * BD;
        const T* alpha_t = alpha_cache + t * BD;
        T* dWx_pre_t = dWx_pre_all + t * BD;
        T* dalpha_x_t = dalpha_x_all + t * BD;

        E64SelfGateBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, d_out_t, dh);

        E64GatedUpdateBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_prev_t, v_pre_t, alpha_t,
            dh, dh_recurrent, dh_prev, dWx_pre_t, dalpha_x_t, db_float);

        if (t > 0) {
            VectorCopy_E64<T><<<num_blocks, block_size, 0, stream_>>>(BD, dh_prev, dh_recurrent);
        }
    }

    // Weight gradients
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        dWx_pre_all, dim_,
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
        dWx_pre_all, dim_,
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
    const int total_elements = TBD;
    const int alpha_blocks = (total_elements + block_size - 1) / block_size;
    AccumulateDbAlpha_E64<T><<<alpha_blocks, block_size, 0, stream_>>>(
        total_elements, dim_, dalpha_x_all, db_alpha_float);

    CopyFloatToT_E64<T><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_float, db);
    CopyFloatToT_E64<T><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_alpha_float, db_alpha);
}

// Explicit template instantiations
template struct E64AdditiveHForward<__half>;
template struct E64AdditiveHForward<float>;
template struct E64AdditiveHForward<double>;

template struct E64AdditiveHBackward<__half>;
template struct E64AdditiveHBackward<float>;
template struct E64AdditiveHBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
