// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// Level 1: Gated Elman - Input-dependent delta gate
// delta = sigmoid(W_delta @ x_t + b_delta)
// h_t = (1 - delta) * h_{t-1} + delta * tanh(W_x @ x_t + W_h @ h_{t-1} + b)

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#include "hasty/elman_ladder.h"
#include "blas.h"
#include "inline_ops.h"

namespace {

// Kernel: Compute gated update
// delta = sigmoid(delta_raw + b_delta)
// candidate = tanh(v + b)
// h_new = (1 - delta) * h_prev + delta * candidate
template<typename T>
__global__ void GatedElmanPointwise(
    const int batch_size,
    const int dim,
    const T* __restrict__ h_prev,      // [B, dim]
    const T* __restrict__ v_in,        // [B, dim] W_x @ x + W_h @ h_prev
    const T* __restrict__ delta_raw,   // [B, dim] W_delta @ x
    const T* __restrict__ b,           // [dim]
    const T* __restrict__ b_delta,     // [dim]
    T* __restrict__ h_out,             // [B, dim]
    T* __restrict__ v_cache,           // [B, dim] pre-activation
    T* __restrict__ delta_cache) {     // [B, dim] cached delta

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // Delta gate: sigmoid(delta_raw + b_delta)
        float delta_in = static_cast<float>(delta_raw[idx]) + static_cast<float>(b_delta[d]);
        float delta = 1.0f / (1.0f + expf(-delta_in));
        if (delta_cache) delta_cache[idx] = static_cast<T>(delta);

        // Candidate: tanh(v + b)
        float v = static_cast<float>(v_in[idx]) + static_cast<float>(b[d]);
        if (v_cache) v_cache[idx] = static_cast<T>(v);
        float candidate = tanhf(v);

        // Gated update: h = (1 - delta) * h_prev + delta * candidate
        float h_p = static_cast<float>(h_prev[idx]);
        float h_new = (1.0f - delta) * h_p + delta * candidate;
        h_out[idx] = static_cast<T>(h_new);
    }
}

// HASTE PATTERN: Fused kernel with separate wx_x and wh_h inputs
template<typename T>
__global__ void GatedElmanPointwiseFused(
    const int batch_size,
    const int dim,
    const T* __restrict__ h_prev,      // [B, dim]
    const T* __restrict__ wx_x,        // [B, dim] W_x @ x (pre-computed)
    const T* __restrict__ wh_h,        // [B, dim] W_h @ h_prev (just computed)
    const T* __restrict__ delta_raw,   // [B, dim] W_delta @ x (pre-computed)
    const T* __restrict__ b,           // [dim]
    const T* __restrict__ b_delta,     // [dim]
    T* __restrict__ h_out,             // [B, dim]
    T* __restrict__ v_cache,           // [B, dim] pre-activation
    T* __restrict__ delta_cache) {     // [B, dim] cached delta

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // Delta gate: sigmoid(delta_raw + b_delta)
        float delta_in = static_cast<float>(delta_raw[idx]) + static_cast<float>(b_delta[d]);
        float delta = 1.0f / (1.0f + expf(-delta_in));
        if (delta_cache) delta_cache[idx] = static_cast<T>(delta);

        // Candidate: tanh(wx_x + wh_h + b)
        float v = static_cast<float>(wx_x[idx]) + static_cast<float>(wh_h[idx]) + static_cast<float>(b[d]);
        if (v_cache) v_cache[idx] = static_cast<T>(v);
        float candidate = tanhf(v);

        // Gated update: h = (1 - delta) * h_prev + delta * candidate
        float h_p = static_cast<float>(h_prev[idx]);
        float h_new = (1.0f - delta) * h_p + delta * candidate;
        h_out[idx] = static_cast<T>(h_new);
    }
}

// Backward kernel for gated update
template<typename T>
__global__ void GatedElmanBackwardKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ h_prev,
    const T* __restrict__ v,
    const T* __restrict__ delta,
    const T* __restrict__ dh,
    const T* __restrict__ dh_recurrent,
    T* __restrict__ dv,
    T* __restrict__ d_delta_raw,
    T* __restrict__ dh_prev_out,
    float* __restrict__ db,
    float* __restrict__ db_delta) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // Total gradient on h
        float grad_h = static_cast<float>(dh[idx]);
        if (dh_recurrent) grad_h += static_cast<float>(dh_recurrent[idx]);

        // h = (1 - delta) * h_prev + delta * candidate
        // candidate = tanh(v)
        float cand = tanhf(static_cast<float>(v[idx]));
        float del = static_cast<float>(delta[idx]);
        float one_minus_del = 1.0f - del;

        // dL/d_candidate = dL/dh * delta
        float d_cand = grad_h * del;

        // dL/dv = dL/d_candidate * (1 - tanh^2(v))
        float dtanh = 1.0f - cand * cand;
        float dv_val = d_cand * dtanh;
        dv[idx] = static_cast<T>(dv_val);

        // dL/d_delta = dL/dh * (candidate - h_prev)
        float h_p = static_cast<float>(h_prev[idx]);
        float d_delta = grad_h * (cand - h_p);

        // dL/d_delta_raw = dL/d_delta * sigmoid'(delta_raw)
        // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x)) = delta * (1 - delta)
        float dsigmoid = del * one_minus_del;
        float d_delta_raw_val = d_delta * dsigmoid;
        d_delta_raw[idx] = static_cast<T>(d_delta_raw_val);

        // dL/dh_prev = dL/dh * (1 - delta)
        dh_prev_out[idx] = static_cast<T>(grad_h * one_minus_del);

        // Accumulate bias gradients
        atomicAdd(&db[d], dv_val);
        atomicAdd(&db_delta[d], d_delta_raw_val);
    }
}

// Kernel: Copy float array to type T (for bias gradients)
template<typename T>
__global__ void CopyFloatToT(
    const int n,
    const float* __restrict__ src,
    T* __restrict__ dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = static_cast<T>(src[idx]);
    }
}

// h+x Selective Output: output = h * silu(h + x + b_gate)
template<typename T>
__global__ void SelectiveOutputForward(
    const int batch_size,
    const int dim,
    const T* __restrict__ h,
    const T* __restrict__ x,
    const T* __restrict__ b_gate,
    T* __restrict__ output,
    T* __restrict__ gate_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float h_val = static_cast<float>(h[idx]);
        float x_val = static_cast<float>(x[idx]);
        float b_val = static_cast<float>(b_gate[d]);

        float gate_raw = h_val + x_val + b_val;
        float sigmoid_val = 1.0f / (1.0f + expf(-gate_raw));
        float silu_val = gate_raw * sigmoid_val;

        output[idx] = static_cast<T>(h_val * silu_val);
        if (gate_cache) gate_cache[idx] = static_cast<T>(silu_val);
    }
}

template<typename T>
__global__ void SelectiveOutputBackward(
    const int batch_size,
    const int dim,
    const T* __restrict__ h,
    const T* __restrict__ x,
    const T* __restrict__ b_gate,
    const T* __restrict__ d_output,
    T* __restrict__ dh,
    T* __restrict__ dx,
    float* __restrict__ d_b_gate) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float h_val = static_cast<float>(h[idx]);
        float x_val = static_cast<float>(x[idx]);
        float b_val = static_cast<float>(b_gate[d]);
        float dout = static_cast<float>(d_output[idx]);

        float gate_raw = h_val + x_val + b_val;
        float sigmoid_val = 1.0f / (1.0f + expf(-gate_raw));
        float silu_val = gate_raw * sigmoid_val;

        // d_silu/d_gate_raw = sigmoid * (1 + gate_raw * (1 - sigmoid))
        float dsilu = sigmoid_val * (1.0f + gate_raw * (1.0f - sigmoid_val));

        // d_output/d_h = silu + h * dsilu
        float dh_val = dout * (silu_val + h_val * dsilu);

        // d_output/d_x = h * dsilu
        float dx_val = dout * h_val * dsilu;

        // d_output/d_b_gate = h * dsilu
        float db_val = dout * h_val * dsilu;

        dh[idx] = static_cast<T>(dh_val);
        dx[idx] = static_cast<T>(dx_val);
        atomicAdd(&d_b_gate[d], db_val);
    }
}

// Kernel to add vectors: a += b
template<typename T>
__global__ void VectorAddInplace(const int n, T* __restrict__ a, const T* __restrict__ b) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = static_cast<T>(static_cast<float>(a[idx]) + static_cast<float>(b[idx]));
    }
}

}  // anonymous namespace


namespace hasty {
namespace v0 {
namespace elman_ladder {

// =============================================================================
// Gated Elman Forward
// =============================================================================

template<typename T>
GatedElmanForward<T>::GatedElmanForward(
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
void GatedElmanForward<T>::Run(
    int steps,
    const T* W_x,
    const T* W_h,
    const T* W_delta,
    const T* b,
    const T* b_delta,
    const T* b_gate,
    const T* x,
    T* h,
    T* output,
    T* v,
    T* delta_cache,
    T* gate_cache) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // =========================================================================
    // HASTE PATTERN: Pre-compute input projections for ALL timesteps
    // Reduces from 2*T GEMMs to 2 big GEMMs
    // =========================================================================
    T *all_wx_x, *all_delta_raw, *wh_h;
    cudaMalloc(&all_wx_x, steps * BD * sizeof(T));
    cudaMalloc(&all_delta_raw, steps * BD * sizeof(T));
    cudaMalloc(&wh_h, BD * sizeof(T));

    // Pre-compute W_x @ x for ALL timesteps
    blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_, &alpha, W_x, dim_, x, dim_, &beta_zero, all_wx_x, dim_);

    // Pre-compute W_delta @ x for ALL timesteps
    blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_, &alpha, W_delta, dim_, x, dim_, &beta_zero, all_delta_raw, dim_);

    // Only W_h @ h_prev per step (depends on h)
    for (int t = 0; t < steps; ++t) {
        const T* wx_x_t = all_wx_x + t * BD;
        const T* delta_raw_t = all_delta_raw + t * BD;
        const T* h_prev = h + t * BD;
        const T* x_t = x + t * BD;
        T* h_t = h + (t + 1) * BD;
        T* out_t = output + t * BD;
        T* v_t = training_ ? (v + t * BD) : nullptr;
        T* delta_t = training_ ? (delta_cache + t * BD) : nullptr;
        T* gate_t = training_ ? (gate_cache + t * BD) : nullptr;

        // wh_h = h_prev @ W_h.T
        blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, W_h, dim_, h_prev, dim_, &beta_zero, wh_h, dim_);

        // Apply gated update with fused kernel (no memcpy needed)
        GatedElmanPointwiseFused<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_prev, wx_x_t, wh_h, delta_raw_t,
            b, b_delta, h_t, v_t, delta_t);

        // h+x selective output: output = h * silu(h + x + b_gate)
        SelectiveOutputForward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, x_t, b_gate, out_t, gate_t);
    }

    cudaFree(all_wx_x);
    cudaFree(all_delta_raw);
    cudaFree(wh_h);
}

// =============================================================================
// Gated Elman Backward
// =============================================================================

template<typename T>
GatedElmanBackward<T>::GatedElmanBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void GatedElmanBackward<T>::Run(
    int steps,
    const T* W_x,
    const T* W_h,
    const T* W_delta,
    const T* b_gate,
    const T* x,
    const T* h,
    const T* v,
    const T* delta_cache,
    const T* gate_cache,
    const T* d_output,
    T* dx,
    T* dW_x,
    T* dW_h,
    T* dW_delta,
    T* db,
    T* db_delta,
    T* d_b_gate,
    T* workspace) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // ==========================================================================
    // WORKSPACE LAYOUT: [dv: BD] [d_delta_raw: BD] [dh_recurrent: BD] [dh_prev: BD]
    //                   [dh: BD] [dx_gate: BD]
    //                   [db_float: dim] [db_delta_float: dim] [db_gate_float: dim]
    // ==========================================================================
    T* dv = workspace;
    T* d_delta_raw = workspace + BD;
    T* dh_recurrent = workspace + 2 * BD;
    T* dh_prev = workspace + 3 * BD;
    T* dh = workspace + 4 * BD;
    T* dx_gate = workspace + 5 * BD;
    float* db_float = reinterpret_cast<float*>(workspace + 6 * BD);
    float* db_delta_float = db_float + dim_;
    float* db_gate_float = db_delta_float + dim_;

    // Initialize workspace (all async on same stream)
    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(T), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_delta_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_gate_float, 0, dim_ * sizeof(float), stream_);

    // Zero weight gradients (async)
    cudaMemsetAsync(dW_x, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dW_h, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dW_delta, 0, dim_ * dim_ * sizeof(T), stream_);

    for (int t = steps - 1; t >= 0; --t) {
        const T* x_t = x + t * BD;
        const T* h_prev = h + t * BD;
        const T* h_t = h + (t + 1) * BD;
        const T* v_t = v + t * BD;
        const T* delta_t = delta_cache + t * BD;
        const T* d_out_t = d_output + t * BD;
        T* dx_t = dx + t * BD;

        // Step 1: Backward through h+x selective output
        SelectiveOutputBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, x_t, b_gate, d_out_t,
            dh, dx_gate, db_gate_float);

        // Step 2: Add recurrent gradient: dh += dh_recurrent
        VectorAddInplace<T><<<num_blocks, block_size, 0, stream_>>>(BD, dh, dh_recurrent);

        // Step 3: Backward through gated update
        GatedElmanBackwardKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_prev, v_t, delta_t, dh, nullptr,
            dv, d_delta_raw, dh_prev, db_float, db_delta_float);

        // dx = dx_gate + dv @ W_x + d_delta_raw @ W_delta
        cudaMemcpyAsync(dx_t, dx_gate, BD * sizeof(T), cudaMemcpyDeviceToDevice, stream_);

        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha,
            W_x, dim_,
            dv, dim_,
            &beta_one,  // Add to existing
            dx_t, dim_);

        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha,
            W_delta, dim_,
            d_delta_raw, dim_,
            &beta_one,  // Add to existing
            dx_t, dim_);

        // dh_recurrent = dv @ W_h + dh_prev
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha,
            W_h, dim_,
            dv, dim_,
            &beta_zero,
            dh_recurrent, dim_);

        // Add dh_prev contribution
        VectorAddInplace<T><<<num_blocks, block_size, 0, stream_>>>(BD, dh_recurrent, dh_prev);

        // dW_x += dv @ x_t^T
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_,
            &alpha,
            dv, dim_,
            x_t, dim_,
            &alpha,
            dW_x, dim_);

        // dW_h += dv @ h_prev^T
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_,
            &alpha,
            dv, dim_,
            h_prev, dim_,
            &alpha,
            dW_h, dim_);

        // dW_delta += d_delta_raw @ x_t^T
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_,
            &alpha,
            d_delta_raw, dim_,
            x_t, dim_,
            &alpha,
            dW_delta, dim_);
    }

    // Copy float bias gradients to T type
    const int bias_blocks = (dim_ + block_size - 1) / block_size;
    CopyFloatToT<T><<<bias_blocks, block_size, 0, stream_>>>(dim_, db_float, db);
    CopyFloatToT<T><<<bias_blocks, block_size, 0, stream_>>>(dim_, db_delta_float, db_delta);
    CopyFloatToT<T><<<bias_blocks, block_size, 0, stream_>>>(dim_, db_gate_float, d_b_gate);
}

// Explicit template instantiations
template struct GatedElmanForward<__half>;
template struct GatedElmanForward<__nv_bfloat16>;
template struct GatedElmanForward<float>;
template struct GatedElmanForward<double>;

template struct GatedElmanBackward<__half>;
template struct GatedElmanBackward<__nv_bfloat16>;
template struct GatedElmanBackward<float>;
template struct GatedElmanBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
