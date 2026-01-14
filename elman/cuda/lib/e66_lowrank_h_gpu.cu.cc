// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E66: Low-Rank H-Dependence (UTM-Class Expressivity)
//
// The key insight: E66 uses low-rank factorization of h-transformation
// to achieve cross-dimension mixing at O(d*rank) cost instead of O(d^2).
//
// E66 Architecture:
// alpha_t = sigmoid(W_alpha @ x_t + b_alpha)           # Retain gate (x-only)
// h_compressed = V @ h_{t-1}                           # [rank] compress h
// h_transformed = U @ h_compressed                     # [dim] expand back
// v_t = tanh(h_transformed + W_x @ x_t + b)           # NONLINEAR value
// h_t = alpha_t * h_{t-1} + (1 - alpha_t) * v_t       # Gated mixing
// output = h * silu(h)                                 # Self-gating
//
// Key optimization:
// - Batch x @ W_alpha.T and x @ W_x.T for ALL timesteps upfront (single GEMM each)
// - Per-timestep: V @ h (small GEMM [rank, dim] x [dim, B] -> [rank, B])
// - Per-timestep: U @ result (small GEMM [dim, rank] x [rank, B] -> [dim, B])
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
// Forward Kernel: Fused gate + value + update + self-gate
// Input: alpha_x = W_alpha @ x_t (pre-batched), Wx = W_x @ x_t (pre-batched),
//        Uh = U @ (V @ h_prev) (per-step, two small GEMMs)
// Output: h_new, output, and optionally save v_pre (for backward)
// =============================================================================

__global__ void E66FusedForwardKernel_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ alpha_x,    // [B, D] pre-computed W_alpha @ x + b_alpha
    const __nv_bfloat16* __restrict__ Wx,         // [B, D] pre-computed W_x @ x
    const __nv_bfloat16* __restrict__ Uh,         // [B, D] per-step U @ (V @ h_prev)
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

        // Value pre-activation: Uh + Wx + b
        float v_pre = __bfloat162float(Uh[idx]) + __bfloat162float(Wx[idx]) + __bfloat162float(b[d]);

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
__global__ void E66FusedForwardKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ alpha_x,
    const T* __restrict__ Wx,
    const T* __restrict__ Uh,
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
        float v_pre = static_cast<float>(Uh[idx]) + static_cast<float>(Wx[idx]) + static_cast<float>(b[d]);

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
__global__ void E66SelfGateBackward_BF16(
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
__global__ void E66SelfGateBackward(
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
// Computes: dh_prev (for recurrence), dv (for U,V weight grads), dalpha (for alpha weight grads)
__global__ void E66GatedUpdateBackward_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ h_prev,        // [B, D] h_{t-1}
    const __nv_bfloat16* __restrict__ v_pre,         // [B, D] pre-tanh value
    const __nv_bfloat16* __restrict__ alpha_cache,   // [B, D] sigmoid(alpha_x)
    const __nv_bfloat16* __restrict__ dh,            // [B, D] gradient from output (dh_t)
    const __nv_bfloat16* __restrict__ dh_recurrent,  // [B, D] gradient from next timestep (or null)
    __nv_bfloat16* __restrict__ dh_prev,             // [B, D] gradient to previous h
    __nv_bfloat16* __restrict__ dv_pre,              // [B, D] gradient through value (for U, V, W_x, b)
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
__global__ void E66GatedUpdateBackward(
    const int batch_size,
    const int dim,
    const T* __restrict__ h_prev,
    const T* __restrict__ v_pre,
    const T* __restrict__ alpha_cache,
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

// Utility: Vector copy
__global__ void VectorCopy_E66_BF16(
    const int n,
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

template<typename T>
__global__ void VectorCopy_E66(const int n, const T* __restrict__ src, T* __restrict__ dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

template<typename T>
__global__ void CopyFloatToT_E66(const int n, const float* __restrict__ src, T* __restrict__ dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = static_cast<T>(src[idx]);
    }
}

// Add bias to alpha_x (element-wise) - BF16 specialization
__global__ void AddAlphaBias_E66_BF16(
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
__global__ void AddAlphaBias_E66(
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
__global__ void AccumulateDbAlpha_E66_BF16(
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
__global__ void AccumulateDbAlpha_E66(
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
// E66 Low-Rank H Forward - BF16 Specialization
// =============================================================================

template<>
E66LowRankHForward<__nv_bfloat16>::E66LowRankHForward(
    bool training,
    int batch_size,
    int dim,
    int rank,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : training_(training),
      batch_size_(batch_size),
      dim_(dim),
      rank_(rank),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void E66LowRankHForward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_alpha,  // [dim, dim]
    const __nv_bfloat16* b_alpha,  // [dim]
    const __nv_bfloat16* U,        // [dim, rank]
    const __nv_bfloat16* V,        // [rank, dim]
    const __nv_bfloat16* W_x,      // [dim, dim]
    const __nv_bfloat16* b,        // [dim]
    const __nv_bfloat16* x,        // [T, B, dim]
    __nv_bfloat16* h,              // [T+1, B, dim]
    __nv_bfloat16* output,         // [T, B, dim]
    __nv_bfloat16* v_pre_cache,    // [T, B, dim] (training)
    __nv_bfloat16* alpha_cache,    // [T, B, dim] (training)
    __nv_bfloat16* Vh_cache,       // [T, B, rank] (training) - V @ h for backward
    __nv_bfloat16* workspace) {

    static const __nv_bfloat16 alpha_one = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);

    const int BD = batch_size_ * dim_;
    const int BR = batch_size_ * rank_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Workspace layout:
    // [alpha_x_all: T*BD] - pre-computed W_alpha @ x + b_alpha
    // [Wx_all: T*BD]      - pre-computed W_x @ x
    // [tmp_Vh: BR]        - per-step V @ h_prev
    // [tmp_Uh: BD]        - per-step U @ (V @ h_prev)
    __nv_bfloat16* alpha_x_all = workspace;
    __nv_bfloat16* Wx_all = workspace + steps * BD;
    __nv_bfloat16* tmp_Vh = workspace + 2 * steps * BD;
    __nv_bfloat16* tmp_Uh = workspace + 2 * steps * BD + BR;

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
    AddAlphaBias_E66_BF16<<<alpha_blocks, block_size, 0, stream_>>>(total_alpha, dim_, alpha_x_all, b_alpha);

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
    // Sequential timestep loop (cannot parallelize due to h-dependence)
    // =========================================================================
    for (int t = 0; t < steps; ++t) {
        const __nv_bfloat16* alpha_x_t = alpha_x_all + t * BD;
        const __nv_bfloat16* Wx_t = Wx_all + t * BD;
        const __nv_bfloat16* h_prev = h + t * BD;
        __nv_bfloat16* h_t = h + (t + 1) * BD;
        __nv_bfloat16* out_t = output + t * BD;
        __nv_bfloat16* v_pre_t = training_ ? (v_pre_cache + t * BD) : nullptr;
        __nv_bfloat16* alpha_t = training_ ? (alpha_cache + t * BD) : nullptr;
        __nv_bfloat16* Vh_t = training_ ? (Vh_cache + t * BR) : nullptr;

        // Step 1: V @ h_prev (compress: [rank, dim] x [dim, B] -> [rank, B])
        blas<__nv_bfloat16>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            rank_, batch_size_, dim_,
            &alpha_one,
            V, dim_,
            h_prev, dim_,
            &beta_zero,
            tmp_Vh, rank_);

        // Cache V @ h for backward if training
        if (training_ && Vh_t) {
            cudaMemcpyAsync(Vh_t, tmp_Vh, BR * sizeof(__nv_bfloat16),
                           cudaMemcpyDeviceToDevice, stream_);
        }

        // Step 2: U @ (V @ h_prev) (expand: [dim, rank] x [rank, B] -> [dim, B])
        blas<__nv_bfloat16>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, rank_,
            &alpha_one,
            U, rank_,
            tmp_Vh, rank_,
            &beta_zero,
            tmp_Uh, dim_);

        // Fused forward kernel
        E66FusedForwardKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, alpha_x_t, Wx_t, tmp_Uh, h_prev, b, h_t, out_t, v_pre_t, alpha_t);
    }
}

// =============================================================================
// E66 Low-Rank H Backward - BF16 Specialization
// =============================================================================

template<>
E66LowRankHBackward<__nv_bfloat16>::E66LowRankHBackward(
    int batch_size,
    int dim,
    int rank,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      rank_(rank),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void E66LowRankHBackward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_alpha,
    const __nv_bfloat16* U,
    const __nv_bfloat16* V,
    const __nv_bfloat16* W_x,
    const __nv_bfloat16* x,
    const __nv_bfloat16* h,
    const __nv_bfloat16* v_pre_cache,
    const __nv_bfloat16* alpha_cache,
    const __nv_bfloat16* Vh_cache,       // [T, B, rank] cached V @ h from forward
    const __nv_bfloat16* d_output,
    __nv_bfloat16* dx,
    __nv_bfloat16* dW_alpha,
    __nv_bfloat16* db_alpha,
    __nv_bfloat16* dU,
    __nv_bfloat16* dV,
    __nv_bfloat16* dW_x,
    __nv_bfloat16* db,
    __nv_bfloat16* workspace) {

    static const __nv_bfloat16 alpha_one = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);
    static const __nv_bfloat16 beta_one = __float2bfloat16(1.0f);

    const int BD = batch_size_ * dim_;
    const int BR = batch_size_ * rank_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Workspace layout:
    // [dh: BD]              - gradient after self-gate
    // [dh_recurrent: BD]    - gradient from next timestep
    // [dh_prev: BD]         - gradient to previous h
    // [dv_pre_all: T*BD]    - gradient through tanh for all timesteps
    // [dalpha_x_all: T*BD]  - gradient through alpha for all timesteps
    // [dUh_all: T*BD]       - gradient through U @ (V @ h) for all timesteps
    // [db_float: dim]       - float accumulator for bias
    // [db_alpha_float: dim] - float accumulator for alpha bias
    // [alpha_x_all: T*BD]   - need to recompute alpha_x for backward
    // [tmp_dVh: BR]         - per-step gradient through V @ h
    __nv_bfloat16* dh = workspace;
    __nv_bfloat16* dh_recurrent = workspace + BD;
    __nv_bfloat16* dh_prev = workspace + 2 * BD;
    __nv_bfloat16* dv_pre_all = workspace + 3 * BD;
    __nv_bfloat16* dalpha_x_all = workspace + 3 * BD + steps * BD;
    __nv_bfloat16* dUh_all = workspace + 3 * BD + 2 * steps * BD;
    float* db_float = reinterpret_cast<float*>(workspace + 3 * BD + 3 * steps * BD);
    float* db_alpha_float = db_float + dim_;
    __nv_bfloat16* alpha_x_all = reinterpret_cast<__nv_bfloat16*>(db_alpha_float + dim_);
    __nv_bfloat16* tmp_dVh = alpha_x_all + steps * BD;

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_alpha_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW_alpha, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dU, 0, dim_ * rank_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dV, 0, rank_ * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_x, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);

    // Recompute alpha_x_all for backward (x @ W_alpha.T + b_alpha)
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
        (void)Vh_cache;  // Used in weight gradient batched GEMM below
        __nv_bfloat16* dv_pre_t = dv_pre_all + t * BD;
        __nv_bfloat16* dalpha_x_t = dalpha_x_all + t * BD;
        __nv_bfloat16* dUh_t = dUh_all + t * BD;

        // Step 1: Backward through self-gate
        E66SelfGateBackward_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, d_out_t, dh);

        // Step 2: Backward through gated update
        E66GatedUpdateBackward_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_prev_t, v_pre_t, alpha_t,
            dh, dh_recurrent, dh_prev, dv_pre_t, dalpha_x_t, db_float);

        // dv_pre is the gradient w.r.t. (U @ V @ h + W_x @ x + b)
        // Copy to dUh for later U, V gradient computation
        cudaMemcpyAsync(dUh_t, dv_pre_t, BD * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToDevice, stream_);

        // Step 3: Gradient through U @ (V @ h_prev)
        // dUh = dv_pre (same gradient)
        // d(V@h) = U.T @ dUh
        // dh_prev_from_UV = V.T @ d(V@h)

        // Compute d(V@h) = U.T @ dv_pre: [rank, dim] x [dim, B] -> [rank, B]
        blas<__nv_bfloat16>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            rank_, batch_size_, dim_,
            &alpha_one,
            U, rank_,
            dv_pre_t, dim_,
            &beta_zero,
            tmp_dVh, rank_);

        // Gradient through V: dh_prev += V.T @ d(V@h): [dim, rank] x [rank, B] -> [dim, B]
        blas<__nv_bfloat16>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, rank_,
            &alpha_one,
            V, dim_,
            tmp_dVh, rank_,
            &beta_one,  // Accumulate into dh_prev
            dh_prev, dim_);

        // Step 4: Copy dh_prev to dh_recurrent for next iteration
        if (t > 0) {
            VectorCopy_E66_BF16<<<num_blocks, block_size, 0, stream_>>>(BD, dh_prev, dh_recurrent);
        }
    }

    // =========================================================================
    // Weight gradients via batched GEMMs
    // =========================================================================

    // dU = sum_t dUh[t] @ (V @ h[t]).T = sum_t dv_pre[t] @ Vh[t].T
    // dU: [dim, rank], dUh_all: [T*B, dim], Vh_cache: [T*B, rank]
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        rank_, dim_, steps * batch_size_,
        &alpha_one,
        Vh_cache, rank_,
        dUh_all, dim_,
        &beta_one,
        dU, rank_);

    // dV = sum_t d(V@h)[t] @ h[t].T
    // Compute dV in a loop (safer than trying to batch)
    for (int t = 0; t < steps; ++t) {
        const __nv_bfloat16* h_prev_t = h + t * BD;
        const __nv_bfloat16* dv_pre_t = dv_pre_all + t * BD;

        // d(V@h) = U.T @ dv_pre: [rank, dim] x [dim, B] -> [rank, B]
        blas<__nv_bfloat16>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            rank_, batch_size_, dim_,
            &alpha_one,
            U, rank_,
            dv_pre_t, dim_,
            &beta_zero,
            tmp_dVh, rank_);

        // dV += d(V@h) @ h.T: [rank, B] x [B, dim] -> [rank, dim]
        blas<__nv_bfloat16>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, rank_, batch_size_,
            &alpha_one,
            h_prev_t, dim_,
            tmp_dVh, rank_,
            &beta_one,
            dV, dim_);
    }

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
    AccumulateDbAlpha_E66_BF16<<<alpha_blocks, block_size, 0, stream_>>>(
        total_alpha, dim_, dalpha_x_all, db_alpha_float);

    // Copy float accumulators to output
    CopyFloatToT_E66<__nv_bfloat16><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_float, db);
    CopyFloatToT_E66<__nv_bfloat16><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_alpha_float, db_alpha);
}

// =============================================================================
// Generic Template Implementations (float, half, double)
// =============================================================================

template<typename T>
E66LowRankHForward<T>::E66LowRankHForward(
    bool training,
    int batch_size,
    int dim,
    int rank,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : training_(training),
      batch_size_(batch_size),
      dim_(dim),
      rank_(rank),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void E66LowRankHForward<T>::Run(
    int steps,
    const T* W_alpha,
    const T* b_alpha,
    const T* U,
    const T* V,
    const T* W_x,
    const T* b,
    const T* x,
    T* h,
    T* output,
    T* v_pre_cache,
    T* alpha_cache,
    T* Vh_cache,
    T* workspace) {

    static const T alpha_one = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int BD = batch_size_ * dim_;
    const int BR = batch_size_ * rank_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    T* alpha_x_all = workspace;
    T* Wx_all = workspace + steps * BD;
    T* tmp_Vh = workspace + 2 * steps * BD;
    T* tmp_Uh = workspace + 2 * steps * BD + BR;

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
    AddAlphaBias_E66<T><<<alpha_blocks, block_size, 0, stream_>>>(total_alpha, dim_, alpha_x_all, b_alpha);

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
        T* Vh_t = training_ ? (Vh_cache + t * BR) : nullptr;

        // V @ h_prev
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            rank_, batch_size_, dim_,
            &alpha_one,
            V, dim_,
            h_prev, dim_,
            &beta_zero,
            tmp_Vh, rank_);

        if (training_ && Vh_t) {
            cudaMemcpyAsync(Vh_t, tmp_Vh, BR * sizeof(T),
                           cudaMemcpyDeviceToDevice, stream_);
        }

        // U @ (V @ h_prev)
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, rank_,
            &alpha_one,
            U, rank_,
            tmp_Vh, rank_,
            &beta_zero,
            tmp_Uh, dim_);

        E66FusedForwardKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, alpha_x_t, Wx_t, tmp_Uh, h_prev, b, h_t, out_t, v_pre_t, alpha_t);
    }
}

template<typename T>
E66LowRankHBackward<T>::E66LowRankHBackward(
    int batch_size,
    int dim,
    int rank,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      rank_(rank),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void E66LowRankHBackward<T>::Run(
    int steps,
    const T* W_alpha,
    const T* U,
    const T* V,
    const T* W_x,
    const T* x,
    const T* h,
    const T* v_pre_cache,
    const T* alpha_cache,
    const T* Vh_cache,
    const T* d_output,
    T* dx,
    T* dW_alpha,
    T* db_alpha,
    T* dU,
    T* dV,
    T* dW_x,
    T* db,
    T* workspace) {

    static const T alpha_one = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int BD = batch_size_ * dim_;
    const int BR = batch_size_ * rank_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    T* dh = workspace;
    T* dh_recurrent = workspace + BD;
    T* dh_prev = workspace + 2 * BD;
    T* dv_pre_all = workspace + 3 * BD;
    T* dalpha_x_all = workspace + 3 * BD + steps * BD;
    T* dUh_all = workspace + 3 * BD + 2 * steps * BD;
    float* db_float = reinterpret_cast<float*>(workspace + 3 * BD + 3 * steps * BD);
    float* db_alpha_float = db_float + dim_;
    T* alpha_x_all = reinterpret_cast<T*>(db_alpha_float + dim_);
    T* tmp_dVh = alpha_x_all + steps * BD;

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(T), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_alpha_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW_alpha, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dU, 0, dim_ * rank_ * sizeof(T), stream_);
    cudaMemsetAsync(dV, 0, rank_ * dim_ * sizeof(T), stream_);
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
        const T* Vh_t = Vh_cache + t * BR;
        T* dv_pre_t = dv_pre_all + t * BD;
        T* dalpha_x_t = dalpha_x_all + t * BD;
        T* dUh_t = dUh_all + t * BD;

        E66SelfGateBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, d_out_t, dh);

        E66GatedUpdateBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_prev_t, v_pre_t, alpha_t,
            dh, dh_recurrent, dh_prev, dv_pre_t, dalpha_x_t, db_float);

        cudaMemcpyAsync(dUh_t, dv_pre_t, BD * sizeof(T),
                       cudaMemcpyDeviceToDevice, stream_);

        // d(V@h) = U.T @ dv_pre
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            rank_, batch_size_, dim_,
            &alpha_one,
            U, rank_,
            dv_pre_t, dim_,
            &beta_zero,
            tmp_dVh, rank_);

        // dh_prev += V.T @ d(V@h)
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, rank_,
            &alpha_one,
            V, dim_,
            tmp_dVh, rank_,
            &beta_one,
            dh_prev, dim_);

        if (t > 0) {
            VectorCopy_E66<T><<<num_blocks, block_size, 0, stream_>>>(BD, dh_prev, dh_recurrent);
        }
    }

    // dU
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        rank_, dim_, steps * batch_size_,
        &alpha_one,
        Vh_cache, rank_,
        dUh_all, dim_,
        &beta_one,
        dU, rank_);

    // dV (loop)
    for (int t = 0; t < steps; ++t) {
        const T* h_prev_t = h + t * BD;
        const T* dv_pre_t = dv_pre_all + t * BD;

        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            rank_, batch_size_, dim_,
            &alpha_one,
            U, rank_,
            dv_pre_t, dim_,
            &beta_zero,
            tmp_dVh, rank_);

        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, rank_, batch_size_,
            &alpha_one,
            h_prev_t, dim_,
            tmp_dVh, rank_,
            &beta_one,
            dV, dim_);
    }

    // dW_x
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        dv_pre_all, dim_,
        &beta_one,
        dW_x, dim_);

    // dW_alpha
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
    AccumulateDbAlpha_E66<T><<<alpha_blocks, block_size, 0, stream_>>>(
        total_alpha, dim_, dalpha_x_all, db_alpha_float);

    CopyFloatToT_E66<T><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_float, db);
    CopyFloatToT_E66<T><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_alpha_float, db_alpha);
}

// Explicit template instantiations
template struct E66LowRankHForward<__half>;
template struct E66LowRankHForward<float>;
template struct E66LowRankHForward<double>;

template struct E66LowRankHBackward<__half>;
template struct E66LowRankHBackward<float>;
template struct E66LowRankHBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
