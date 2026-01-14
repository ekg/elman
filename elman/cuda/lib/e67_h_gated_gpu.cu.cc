// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E67: H-Dependent Gate Only - Nonlinearity Through Gate Selection
//
// Put h-dependence in the GATE, not the value:
//     alpha_t = sigmoid(W_alpha @ x_t + d_alpha * h_{t-1} + b_alpha)   # h affects gate!
//     v_t = tanh(W_x @ x_t + b_v)                                      # v is h-independent
//     h_t = alpha_t * h_{t-1} + (1 - alpha_t) * v_t
//     output = h * silu(h)                                             # Self-gating
//
// Key insight: The gate controls WHAT to remember vs WHAT to write.
// Making the gate h-dependent means the model can make state-dependent
// decisions about memory management.
//
// Cost: O(d) per timestep (d_alpha is diagonal)
// UTM-class because gate depends nonlinearly on h through sigmoid
//
// Jacobian:
//     dh_t/dh_{t-1} = diag(alpha) + diag(h - v) * diag(alpha * (1 - alpha) * d_alpha)
//                   = diag(alpha + (h - v) * sigma'(gate) * d_alpha)
//
// Key optimization:
// - Batch x @ W_alpha.T and x @ W_x.T for ALL timesteps upfront (single GEMM each)
// - Per-timestep: only diagonal d_alpha * h (O(d) element-wise)
// - Much faster than E63 which has per-step W_h @ h GEMM

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
// Forward Kernel: Fused gate (with diagonal h) + value + update + self-gate
// =============================================================================

__global__ void E67FusedForwardKernel_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ alpha_x,    // [B, D] pre-computed W_alpha @ x + b_alpha
    const __nv_bfloat16* __restrict__ d_alpha,    // [D] diagonal for h contribution to gate
    const __nv_bfloat16* __restrict__ v_all,      // [B, D] pre-computed tanh(W_x @ x + b_v)
    const __nv_bfloat16* __restrict__ h_prev,     // [B, D] previous hidden state
    __nv_bfloat16* __restrict__ h_out,            // [B, D] new hidden state
    __nv_bfloat16* __restrict__ output,           // [B, D] output (h * silu(h))
    __nv_bfloat16* __restrict__ alpha_cache,      // [B, D] alpha cache (for backward)
    __nv_bfloat16* __restrict__ gate_pre_cache) { // [B, D] pre-sigmoid gate cache (for backward)

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float h_prev_val = __bfloat162float(h_prev[idx]);

        // H-DEPENDENT GATE: alpha = sigmoid(W_alpha @ x + d_alpha * h_prev + b_alpha)
        float gate_pre = __bfloat162float(alpha_x[idx]) + __bfloat162float(d_alpha[d]) * h_prev_val;
        float alpha = 1.0f / (1.0f + __expf(-gate_pre));

        // Value (pre-computed, no h dependence)
        float v = __bfloat162float(v_all[idx]);

        // Gated update: h_new = alpha * h_prev + (1 - alpha) * v
        float h_new = alpha * h_prev_val + (1.0f - alpha) * v;

        // Store hidden state
        h_out[idx] = __float2bfloat16(h_new);

        // Self-gate output: h * silu(h) = h^2 * sigmoid(h)
        float sigmoid_h = 1.0f / (1.0f + __expf(-h_new));
        float silu_h = h_new * sigmoid_h;
        output[idx] = __float2bfloat16(h_new * silu_h);

        // Cache for backward (training only)
        if (alpha_cache) alpha_cache[idx] = __float2bfloat16(alpha);
        if (gate_pre_cache) gate_pre_cache[idx] = __float2bfloat16(gate_pre);
    }
}

template<typename T>
__global__ void E67FusedForwardKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ alpha_x,
    const T* __restrict__ d_alpha,
    const T* __restrict__ v_all,
    const T* __restrict__ h_prev,
    T* __restrict__ h_out,
    T* __restrict__ output,
    T* __restrict__ alpha_cache,
    T* __restrict__ gate_pre_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float h_prev_val = static_cast<float>(h_prev[idx]);

        // H-dependent gate
        float gate_pre = static_cast<float>(alpha_x[idx]) + static_cast<float>(d_alpha[d]) * h_prev_val;
        float alpha = 1.0f / (1.0f + expf(-gate_pre));

        // Value
        float v = static_cast<float>(v_all[idx]);

        // Gated update
        float h_new = alpha * h_prev_val + (1.0f - alpha) * v;

        h_out[idx] = static_cast<T>(h_new);

        // Self-gate
        float sigmoid_h = 1.0f / (1.0f + expf(-h_new));
        float silu_h = h_new * sigmoid_h;
        output[idx] = static_cast<T>(h_new * silu_h);

        if (alpha_cache) alpha_cache[idx] = static_cast<T>(alpha);
        if (gate_pre_cache) gate_pre_cache[idx] = static_cast<T>(gate_pre);
    }
}

// =============================================================================
// Value pre-computation kernel: v = tanh(W_x @ x + b_v)
// =============================================================================

__global__ void E67ComputeValue_BF16(
    const int n,
    const int dim,
    const __nv_bfloat16* __restrict__ Wx,     // [T*B, D] pre-computed W_x @ x
    const __nv_bfloat16* __restrict__ b_v,    // [D]
    __nv_bfloat16* __restrict__ v_all) {      // [T*B, D] output

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const int d = idx % dim;
        float val = __bfloat162float(Wx[idx]) + __bfloat162float(b_v[d]);
        v_all[idx] = __float2bfloat16(tanhf(val));
    }
}

template<typename T>
__global__ void E67ComputeValue(
    const int n,
    const int dim,
    const T* __restrict__ Wx,
    const T* __restrict__ b_v,
    T* __restrict__ v_all) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const int d = idx % dim;
        float val = static_cast<float>(Wx[idx]) + static_cast<float>(b_v[d]);
        v_all[idx] = static_cast<T>(tanhf(val));
    }
}

// =============================================================================
// Add bias to alpha_x (element-wise)
// =============================================================================

__global__ void E67AddAlphaBias_BF16(
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
__global__ void E67AddAlphaBias(
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

// =============================================================================
// Backward Kernels
// =============================================================================

// Phase 1: Backward through self-gate: d_output -> dh_post_gate
__global__ void E67SelfGateBackward_BF16(
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
__global__ void E67SelfGateBackward(
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

// Phase 2: Backward through gated update
// h_t = alpha * h_prev + (1 - alpha) * v
// alpha = sigmoid(W_alpha @ x + d_alpha * h_prev + b_alpha)
// Computes: dh_prev, dalpha_x, dd_alpha_contrib
__global__ void E67GatedUpdateBackward_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ h_prev,        // [B, D]
    const __nv_bfloat16* __restrict__ v,             // [B, D] pre-computed tanh value
    const __nv_bfloat16* __restrict__ alpha_cache,   // [B, D] sigmoid(gate)
    const __nv_bfloat16* __restrict__ d_alpha,       // [D] diagonal
    const __nv_bfloat16* __restrict__ dh,            // [B, D] gradient from output
    const __nv_bfloat16* __restrict__ dh_recurrent,  // [B, D] gradient from next timestep
    __nv_bfloat16* __restrict__ dh_prev,             // [B, D] gradient to previous h
    __nv_bfloat16* __restrict__ dalpha_x,            // [B, D] gradient through alpha for W_alpha
    float* __restrict__ dd_alpha_accum,              // [D] accumulator for d_alpha gradient
    float* __restrict__ db_alpha_accum) {            // [D] accumulator for b_alpha gradient

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
        float v_val = __bfloat162float(v[idx]);

        // h_t = alpha * h_prev + (1 - alpha) * v
        // dalpha = dh_total * (h_prev - v)
        float dalpha = dh_total * (h_prev_val - v_val);

        // dalpha_pre = dalpha * sigmoid'(gate) = dalpha * alpha * (1 - alpha)
        float dalpha_pre = dalpha * alpha * (1.0f - alpha);

        // dh_prev from alpha path: dh_prev = dh_total * alpha
        // dh_prev from gate path: dh_prev += dalpha_pre * d_alpha
        float d_alpha_val = __bfloat162float(d_alpha[d]);
        float dh_prev_val = dh_total * alpha + dalpha_pre * d_alpha_val;
        dh_prev[idx] = __float2bfloat16(dh_prev_val);

        // dalpha_x = dalpha_pre (for W_alpha gradient)
        dalpha_x[idx] = __float2bfloat16(dalpha_pre);

        // Accumulate gradients
        // dd_alpha = dalpha_pre * h_prev
        atomicAdd(&dd_alpha_accum[d], dalpha_pre * h_prev_val);
        // db_alpha = dalpha_pre (sum over batch)
        atomicAdd(&db_alpha_accum[d], dalpha_pre);
    }
}

template<typename T>
__global__ void E67GatedUpdateBackward(
    const int batch_size,
    const int dim,
    const T* __restrict__ h_prev,
    const T* __restrict__ v,
    const T* __restrict__ alpha_cache,
    const T* __restrict__ d_alpha,
    const T* __restrict__ dh,
    const T* __restrict__ dh_recurrent,
    T* __restrict__ dh_prev,
    T* __restrict__ dalpha_x,
    float* __restrict__ dd_alpha_accum,
    float* __restrict__ db_alpha_accum) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float dh_total = static_cast<float>(dh[idx]);
        if (dh_recurrent) dh_total += static_cast<float>(dh_recurrent[idx]);

        float alpha = static_cast<float>(alpha_cache[idx]);
        float h_prev_val = static_cast<float>(h_prev[idx]);
        float v_val = static_cast<float>(v[idx]);

        float dalpha = dh_total * (h_prev_val - v_val);
        float dalpha_pre = dalpha * alpha * (1.0f - alpha);

        float d_alpha_val = static_cast<float>(d_alpha[d]);
        float dh_prev_val = dh_total * alpha + dalpha_pre * d_alpha_val;
        dh_prev[idx] = static_cast<T>(dh_prev_val);

        dalpha_x[idx] = static_cast<T>(dalpha_pre);

        atomicAdd(&dd_alpha_accum[d], dalpha_pre * h_prev_val);
        atomicAdd(&db_alpha_accum[d], dalpha_pre);
    }
}

// Backward through value: v = tanh(W_x @ x + b_v)
// dv -> dWx_pre (for weight gradients)
__global__ void E67ValueBackward_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ v,             // [B, D] tanh output
    const __nv_bfloat16* __restrict__ alpha_cache,   // [B, D]
    const __nv_bfloat16* __restrict__ dh,            // [B, D]
    const __nv_bfloat16* __restrict__ dh_recurrent,  // [B, D]
    __nv_bfloat16* __restrict__ dWx_pre,             // [B, D] gradient through tanh
    float* __restrict__ db_v_accum) {                // [D] accumulator for b_v gradient

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float dh_total;
        if (dh_recurrent) {
            dh_total = __bfloat162float(dh[idx]) + __bfloat162float(dh_recurrent[idx]);
        } else {
            dh_total = __bfloat162float(dh[idx]);
        }

        float alpha = __bfloat162float(alpha_cache[idx]);
        float v_val = __bfloat162float(v[idx]);

        // dv = dh_total * (1 - alpha)
        float dv = dh_total * (1.0f - alpha);

        // dWx_pre = dv * tanh'(Wx_pre) = dv * (1 - v^2)
        float dWx_pre_val = dv * (1.0f - v_val * v_val);
        dWx_pre[idx] = __float2bfloat16(dWx_pre_val);

        atomicAdd(&db_v_accum[d], dWx_pre_val);
    }
}

template<typename T>
__global__ void E67ValueBackward(
    const int batch_size,
    const int dim,
    const T* __restrict__ v,
    const T* __restrict__ alpha_cache,
    const T* __restrict__ dh,
    const T* __restrict__ dh_recurrent,
    T* __restrict__ dWx_pre,
    float* __restrict__ db_v_accum) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float dh_total = static_cast<float>(dh[idx]);
        if (dh_recurrent) dh_total += static_cast<float>(dh_recurrent[idx]);

        float alpha = static_cast<float>(alpha_cache[idx]);
        float v_val = static_cast<float>(v[idx]);

        float dv = dh_total * (1.0f - alpha);
        float dWx_pre_val = dv * (1.0f - v_val * v_val);
        dWx_pre[idx] = static_cast<T>(dWx_pre_val);

        atomicAdd(&db_v_accum[d], dWx_pre_val);
    }
}

// Utility kernels
__global__ void E67VectorCopy_BF16(
    const int n,
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

template<typename T>
__global__ void E67VectorCopy(const int n, const T* __restrict__ src, T* __restrict__ dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

template<typename T>
__global__ void E67CopyFloatToT(const int n, const float* __restrict__ src, T* __restrict__ dst) {
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
// E67 H-Gated Forward - BF16 Specialization
// =============================================================================

template<>
E67HGatedForward<__nv_bfloat16>::E67HGatedForward(
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
void E67HGatedForward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_alpha,  // [dim, dim]
    const __nv_bfloat16* d_alpha,  // [dim] diagonal for h in gate
    const __nv_bfloat16* b_alpha,  // [dim]
    const __nv_bfloat16* W_x,      // [dim, dim]
    const __nv_bfloat16* b_v,      // [dim]
    const __nv_bfloat16* x,        // [T, B, dim]
    __nv_bfloat16* h,              // [T+1, B, dim]
    __nv_bfloat16* output,         // [T, B, dim]
    __nv_bfloat16* v_cache,        // [T, B, dim] (training)
    __nv_bfloat16* alpha_cache,    // [T, B, dim] (training)
    __nv_bfloat16* workspace) {

    static const __nv_bfloat16 alpha_one = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Workspace layout:
    // [alpha_x_all: T*BD] - pre-computed W_alpha @ x + b_alpha
    // [v_all: T*BD]       - pre-computed tanh(W_x @ x + b_v)
    __nv_bfloat16* alpha_x_all = workspace;
    __nv_bfloat16* v_all = workspace + steps * BD;

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

    // Add bias to alpha_x
    const int total = steps * BD;
    const int total_blocks = (total + block_size - 1) / block_size;
    E67AddAlphaBias_BF16<<<total_blocks, block_size, 0, stream_>>>(total, dim_, alpha_x_all, b_alpha);

    // =========================================================================
    // BATCH GEMM 2: W_x @ x for ALL timesteps, then compute v = tanh(Wx + b_v)
    // =========================================================================
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_x, dim_,
        x, dim_,
        &beta_zero,
        v_all, dim_);

    // Compute v = tanh(Wx @ x + b_v)
    E67ComputeValue_BF16<<<total_blocks, block_size, 0, stream_>>>(total, dim_, v_all, b_v, v_all);

    // If training, copy v_all to v_cache
    if (training_ && v_cache) {
        cudaMemcpyAsync(v_cache, v_all, total * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream_);
    }

    // =========================================================================
    // Sequential timestep loop (only diagonal h in gate - O(d) per step!)
    // =========================================================================
    for (int t = 0; t < steps; ++t) {
        const __nv_bfloat16* alpha_x_t = alpha_x_all + t * BD;
        const __nv_bfloat16* v_t = v_all + t * BD;
        const __nv_bfloat16* h_prev = h + t * BD;
        __nv_bfloat16* h_t = h + (t + 1) * BD;
        __nv_bfloat16* out_t = output + t * BD;
        __nv_bfloat16* alpha_t = training_ ? (alpha_cache + t * BD) : nullptr;

        // Fused forward kernel (no per-step GEMM needed!)
        E67FusedForwardKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, alpha_x_t, d_alpha, v_t, h_prev, h_t, out_t, alpha_t, nullptr);
    }
}

// =============================================================================
// E67 H-Gated Backward - BF16 Specialization
// =============================================================================

template<>
E67HGatedBackward<__nv_bfloat16>::E67HGatedBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void E67HGatedBackward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_alpha,
    const __nv_bfloat16* d_alpha,
    const __nv_bfloat16* W_x,
    const __nv_bfloat16* x,
    const __nv_bfloat16* h,
    const __nv_bfloat16* v_cache,
    const __nv_bfloat16* alpha_cache,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* dx,
    __nv_bfloat16* dW_alpha,
    __nv_bfloat16* dd_alpha,
    __nv_bfloat16* db_alpha,
    __nv_bfloat16* dW_x,
    __nv_bfloat16* db_v,
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
    // [dalpha_x_all: T*BD]  - gradient for W_alpha
    // [dWx_pre_all: T*BD]   - gradient through tanh for W_x
    // [dd_alpha_float: dim] - float accumulator for d_alpha
    // [db_alpha_float: dim] - float accumulator for b_alpha
    // [db_v_float: dim]     - float accumulator for b_v
    __nv_bfloat16* dh = workspace;
    __nv_bfloat16* dh_recurrent = workspace + BD;
    __nv_bfloat16* dh_prev = workspace + 2 * BD;
    __nv_bfloat16* dalpha_x_all = workspace + 3 * BD;
    __nv_bfloat16* dWx_pre_all = workspace + 3 * BD + steps * BD;
    float* dd_alpha_float = reinterpret_cast<float*>(workspace + 3 * BD + 2 * steps * BD);
    float* db_alpha_float = dd_alpha_float + dim_;
    float* db_v_float = db_alpha_float + dim_;

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dd_alpha_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_alpha_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_v_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW_alpha, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_x, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);

    // =========================================================================
    // BPTT Loop (backward through time)
    // =========================================================================
    for (int t = steps - 1; t >= 0; --t) {
        const __nv_bfloat16* h_t = h + (t + 1) * BD;
        const __nv_bfloat16* h_prev_t = h + t * BD;
        const __nv_bfloat16* d_out_t = d_output + t * BD;
        const __nv_bfloat16* v_t = v_cache + t * BD;
        const __nv_bfloat16* alpha_t = alpha_cache + t * BD;
        __nv_bfloat16* dalpha_x_t = dalpha_x_all + t * BD;
        __nv_bfloat16* dWx_pre_t = dWx_pre_all + t * BD;

        // Step 1: Backward through self-gate
        E67SelfGateBackward_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, d_out_t, dh);

        // Step 2: Backward through gated update
        E67GatedUpdateBackward_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_prev_t, v_t, alpha_t, d_alpha,
            dh, dh_recurrent, dh_prev, dalpha_x_t, dd_alpha_float, db_alpha_float);

        // Step 3: Backward through value computation
        E67ValueBackward_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, v_t, alpha_t, dh, dh_recurrent, dWx_pre_t, db_v_float);

        // Step 4: Copy dh_prev to dh_recurrent for next iteration
        if (t > 0) {
            E67VectorCopy_BF16<<<num_blocks, block_size, 0, stream_>>>(BD, dh_prev, dh_recurrent);
        }
    }

    // =========================================================================
    // Weight gradients via batched GEMMs
    // =========================================================================

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

    // dx = dalpha_x @ W_alpha + dWx_pre @ W_x
    // First: dx = dalpha_x @ W_alpha
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_alpha, dim_,
        dalpha_x_all, dim_,
        &beta_zero,
        dx, dim_);

    // Then: dx += dWx_pre @ W_x
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_x, dim_,
        dWx_pre_all, dim_,
        &beta_one,
        dx, dim_);

    // Copy float accumulators to output
    const int dim_blocks = (dim_ + 255) / 256;
    E67CopyFloatToT<__nv_bfloat16><<<dim_blocks, 256, 0, stream_>>>(dim_, dd_alpha_float, dd_alpha);
    E67CopyFloatToT<__nv_bfloat16><<<dim_blocks, 256, 0, stream_>>>(dim_, db_alpha_float, db_alpha);
    E67CopyFloatToT<__nv_bfloat16><<<dim_blocks, 256, 0, stream_>>>(dim_, db_v_float, db_v);
}

// =============================================================================
// Generic Template Implementations
// =============================================================================

template<typename T>
E67HGatedForward<T>::E67HGatedForward(
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
void E67HGatedForward<T>::Run(
    int steps,
    const T* W_alpha,
    const T* d_alpha,
    const T* b_alpha,
    const T* W_x,
    const T* b_v,
    const T* x,
    T* h,
    T* output,
    T* v_cache,
    T* alpha_cache,
    T* workspace) {

    static const T alpha_one = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    T* alpha_x_all = workspace;
    T* v_all = workspace + steps * BD;

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
    const int total = steps * BD;
    const int total_blocks = (total + block_size - 1) / block_size;
    E67AddAlphaBias<T><<<total_blocks, block_size, 0, stream_>>>(total, dim_, alpha_x_all, b_alpha);

    // Batch GEMM for W_x @ x, then compute v
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_x, dim_,
        x, dim_,
        &beta_zero,
        v_all, dim_);

    E67ComputeValue<T><<<total_blocks, block_size, 0, stream_>>>(total, dim_, v_all, b_v, v_all);

    if (training_ && v_cache) {
        cudaMemcpyAsync(v_cache, v_all, total * sizeof(T), cudaMemcpyDeviceToDevice, stream_);
    }

    // Sequential loop
    for (int t = 0; t < steps; ++t) {
        const T* alpha_x_t = alpha_x_all + t * BD;
        const T* v_t = v_all + t * BD;
        const T* h_prev = h + t * BD;
        T* h_t = h + (t + 1) * BD;
        T* out_t = output + t * BD;
        T* alpha_t = training_ ? (alpha_cache + t * BD) : nullptr;

        E67FusedForwardKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, alpha_x_t, d_alpha, v_t, h_prev, h_t, out_t, alpha_t, nullptr);
    }
}

template<typename T>
E67HGatedBackward<T>::E67HGatedBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void E67HGatedBackward<T>::Run(
    int steps,
    const T* W_alpha,
    const T* d_alpha,
    const T* W_x,
    const T* x,
    const T* h,
    const T* v_cache,
    const T* alpha_cache,
    const T* d_output,
    T* dx,
    T* dW_alpha,
    T* dd_alpha,
    T* db_alpha,
    T* dW_x,
    T* db_v,
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
    T* dalpha_x_all = workspace + 3 * BD;
    T* dWx_pre_all = workspace + 3 * BD + steps * BD;
    float* dd_alpha_float = reinterpret_cast<float*>(workspace + 3 * BD + 2 * steps * BD);
    float* db_alpha_float = dd_alpha_float + dim_;
    float* db_v_float = db_alpha_float + dim_;

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(T), stream_);
    cudaMemsetAsync(dd_alpha_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_alpha_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_v_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW_alpha, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dW_x, 0, dim_ * dim_ * sizeof(T), stream_);

    // BPTT loop
    for (int t = steps - 1; t >= 0; --t) {
        const T* h_t = h + (t + 1) * BD;
        const T* h_prev_t = h + t * BD;
        const T* d_out_t = d_output + t * BD;
        const T* v_t = v_cache + t * BD;
        const T* alpha_t = alpha_cache + t * BD;
        T* dalpha_x_t = dalpha_x_all + t * BD;
        T* dWx_pre_t = dWx_pre_all + t * BD;

        E67SelfGateBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, d_out_t, dh);

        E67GatedUpdateBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_prev_t, v_t, alpha_t, d_alpha,
            dh, dh_recurrent, dh_prev, dalpha_x_t, dd_alpha_float, db_alpha_float);

        E67ValueBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, v_t, alpha_t, dh, dh_recurrent, dWx_pre_t, db_v_float);

        if (t > 0) {
            E67VectorCopy<T><<<num_blocks, block_size, 0, stream_>>>(BD, dh_prev, dh_recurrent);
        }
    }

    // Weight gradients
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        dalpha_x_all, dim_,
        &beta_one,
        dW_alpha, dim_);

    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        dWx_pre_all, dim_,
        &beta_one,
        dW_x, dim_);

    // dx
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_alpha, dim_,
        dalpha_x_all, dim_,
        &beta_zero,
        dx, dim_);

    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_x, dim_,
        dWx_pre_all, dim_,
        &beta_one,
        dx, dim_);

    // Copy accumulators
    const int dim_blocks = (dim_ + 255) / 256;
    E67CopyFloatToT<T><<<dim_blocks, 256, 0, stream_>>>(dim_, dd_alpha_float, dd_alpha);
    E67CopyFloatToT<T><<<dim_blocks, 256, 0, stream_>>>(dim_, db_alpha_float, db_alpha);
    E67CopyFloatToT<T><<<dim_blocks, 256, 0, stream_>>>(dim_, db_v_float, db_v);
}

// Explicit template instantiations
template struct E67HGatedForward<__half>;
template struct E67HGatedForward<float>;
template struct E67HGatedForward<double>;

template struct E67HGatedBackward<__half>;
template struct E67HGatedBackward<float>;
template struct E67HGatedBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
