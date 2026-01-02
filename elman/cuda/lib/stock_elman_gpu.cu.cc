// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// Level 0: Stock Elman - Basic tanh recurrence with h+x selective output
// h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b)
// output_t = h_t * silu(h_t + x_t + b_gate)  -- h+x input-dependent gating

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

// Kernel: Apply tanh activation and add bias (original version)
template<typename T>
__global__ void PointwiseTanhBias(
    const int batch_size,
    const int dim,
    const T* __restrict__ v_in,      // [B, dim] pre-bias
    const T* __restrict__ b,         // [dim]
    T* __restrict__ h_out,           // [B, dim] output
    T* __restrict__ v_cache) {       // [B, dim] pre-activation cache

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;
        float val = static_cast<float>(v_in[idx]) + static_cast<float>(b[d]);
        if (v_cache) v_cache[idx] = static_cast<T>(val);
        h_out[idx] = static_cast<T>(tanhf(val));
    }
}

// Kernel: Fused Wx + Rh + bias + tanh (HASTE pattern)
// Combines pre-computed input projection with recurrent result
template<typename T>
__global__ void FusedTanhKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ Wx,        // [B, dim] pre-computed W_x @ x
    const T* __restrict__ Rh,        // [B, dim] W_h @ h_prev (just computed)
    const T* __restrict__ b,         // [dim] bias
    T* __restrict__ h_out,           // [B, dim] output
    T* __restrict__ v_cache) {       // [B, dim] pre-activation cache (optional)

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;
        float val = static_cast<float>(Wx[idx]) + static_cast<float>(Rh[idx]) + static_cast<float>(b[d]);
        if (v_cache) v_cache[idx] = static_cast<T>(val);
        h_out[idx] = static_cast<T>(tanhf(val));
    }
}

// Kernel: Backward through tanh and compute gradients
template<typename T>
__global__ void StockElmanBackwardKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ v,           // [B, dim] pre-activation
    const T* __restrict__ dh,          // [B, dim] gradient from above
    const T* __restrict__ dh_recurrent,// [B, dim] gradient from next timestep (or null)
    T* __restrict__ dv,                // [B, dim] gradient w.r.t. pre-activation
    float* __restrict__ db) {          // [dim] gradient w.r.t. bias (atomic add, float for all types)

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // Combine gradients from output and recurrent path
        float grad = static_cast<float>(dh[idx]);
        if (dh_recurrent) grad += static_cast<float>(dh_recurrent[idx]);

        // dtanh: dL/dv = dL/dh * (1 - tanh(v)^2)
        float h = tanhf(static_cast<float>(v[idx]));
        float dtanh = 1.0f - h * h;
        float dv_val = grad * dtanh;
        dv[idx] = static_cast<T>(dv_val);

        // Accumulate bias gradient
        atomicAdd(&db[d], dv_val);
    }
}

// =============================================================================
// h+x Selective Output: output = h * silu(h + x + b_gate)
// =============================================================================

template<typename T>
__global__ void SelectiveOutputForward(
    const int batch_size,
    const int dim,
    const T* __restrict__ h,            // [B, dim]
    const T* __restrict__ x,            // [B, dim] input at this timestep
    const T* __restrict__ b_gate,       // [dim] learned gate bias
    T* __restrict__ output,             // [B, dim]
    T* __restrict__ gate_cache) {       // [B, dim] silu value for backward

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
    T* __restrict__ dh,              // gradient to h (add to existing)
    T* __restrict__ dx,              // gradient to x
    float* __restrict__ d_b_gate) {  // accumulated gradient for b_gate

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

        // d_output/d_h = silu + h * dsilu (h appears in both output and gate)
        float dh_val = dout * (silu_val + h_val * dsilu);

        // d_output/d_x = h * dsilu
        float dx_val = dout * h_val * dsilu;

        // d_output/d_b_gate = h * dsilu (same as d_x)
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
// Stock Elman Forward
// =============================================================================

template<typename T>
StockElmanForward<T>::StockElmanForward(
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
void StockElmanForward<T>::Run(
    int steps,
    const T* W_x,
    const T* W_h,
    const T* b,
    const T* b_gate,
    const T* x,
    T* h,
    T* output,
    T* v,
    T* gate_cache) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // =========================================================================
    // HASTE PATTERN: Pre-compute W_x @ x for ALL timesteps in ONE big GEMM!
    // =========================================================================
    T* tmp_Wx;
    T* tmp_Rh;
    cudaMalloc(&tmp_Wx, steps * BD * sizeof(T));
    cudaMalloc(&tmp_Rh, BD * sizeof(T));

    // One big GEMM: tmp_Wx = x @ W_x.T for all T*B rows at once
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha,
        W_x, dim_,
        x, dim_,
        &beta_zero,
        tmp_Wx, dim_);

    // Now process each timestep
    for (int t = 0; t < steps; ++t) {
        const T* Wx_t = tmp_Wx + t * BD;
        const T* h_prev = h + t * BD;
        const T* x_t = x + t * BD;  // Raw input for h+x gate
        T* h_t = h + (t + 1) * BD;
        T* out_t = output + t * BD;
        T* v_t = training_ ? (v + t * BD) : nullptr;
        T* gate_t = training_ ? (gate_cache + t * BD) : nullptr;

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

        // h_t = tanh(Wx_t + tmp_Rh + b)
        FusedTanhKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, Wx_t, tmp_Rh, b, h_t, v_t);

        // output = h * silu(h + x + b_gate)
        SelectiveOutputForward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, x_t, b_gate, out_t, gate_t);
    }

    cudaFree(tmp_Wx);
    cudaFree(tmp_Rh);
}

// =============================================================================
// Stock Elman Backward
// =============================================================================

template<typename T>
StockElmanBackward<T>::StockElmanBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      sync_stream_(stream) {
    stream_[0] = stream;  // Use the provided stream
    stream_[1] = nullptr;
    event_ = nullptr;
}

template<typename T>
StockElmanBackward<T>::~StockElmanBackward() {
    // No cleanup needed - we don't own the stream
}

template<typename T>
void StockElmanBackward<T>::Run(
    int steps,
    const T* W_x,
    const T* W_h,
    const T* b_gate,
    const T* x,
    const T* h,
    const T* v,
    const T* gate_cache,
    const T* d_output,
    T* dx,
    T* dW_x,
    T* dW_h,
    T* db,
    T* d_b_gate,
    T* workspace) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    const cudaStream_t stream = stream_[0];

    // ==========================================================================
    // Workspace layout: [dv_all: T*BD] [dh: BD] [dh_recurrent: BD]
    //                   [dx_gate: BD] [db_float: dim] [db_gate_float: dim]
    // ==========================================================================
    T* dv_all = workspace;
    T* dh = workspace + steps * BD;
    T* dh_recurrent = workspace + (steps + 1) * BD;
    T* dx_gate = workspace + (steps + 2) * BD;
    float* db_float = reinterpret_cast<float*>(workspace + (steps + 3) * BD);
    float* db_gate_float = db_float + dim_;

    // Initialize workspace
    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(T), stream);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream);
    cudaMemsetAsync(db_gate_float, 0, dim_ * sizeof(float), stream);
    cudaMemsetAsync(dW_x, 0, dim_ * dim_ * sizeof(T), stream);
    cudaMemsetAsync(dW_h, 0, dim_ * dim_ * sizeof(T), stream);

    // ==========================================================================
    // BPTT loop with h+x selective output
    // ==========================================================================
    for (int t = steps - 1; t >= 0; --t) {
        const T* v_t = v + t * BD;
        const T* h_t = h + (t + 1) * BD;  // h at output of this step
        const T* x_t = x + t * BD;
        const T* d_out_t = d_output + t * BD;
        T* dv_t = dv_all + t * BD;
        T* dx_t = dx + t * BD;

        // Backward through h+x selective output: output = h * silu(h + x + b_gate)
        SelectiveOutputBackward<T><<<num_blocks, block_size, 0, stream>>>(
            batch_size_, dim_, h_t, x_t, b_gate, d_out_t,
            dh, dx_gate, db_gate_float);

        // Add recurrent gradient from next timestep: dh += dh_recurrent
        VectorAddInplace<T><<<num_blocks, block_size, 0, stream>>>(BD, dh, dh_recurrent);

        // Backward through tanh: dv = dh * (1 - tanh(v)^2)
        StockElmanBackwardKernel<T><<<num_blocks, block_size, 0, stream>>>(
            batch_size_, dim_, v_t, dh, nullptr, dv_t, db_float);

        // dh_recurrent = W_h @ dv (for next iteration)
        if (t > 0) {
            blas<T>::gemm(
                blas_handle_,
                CUBLAS_OP_N, CUBLAS_OP_N,
                dim_, batch_size_, dim_,
                &alpha,
                W_h, dim_,
                dv_t, dim_,
                &beta_zero,
                dh_recurrent, dim_);
        }

        // dx from gate + dx from W_x will be combined below
        cudaMemcpyAsync(dx_t, dx_gate, BD * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    }

    // ==========================================================================
    // Batch GEMMs across all timesteps
    // ==========================================================================

    // dx += W_x @ dv_all (add to existing dx which has gate gradient)
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha,
        W_x, dim_,
        dv_all, dim_,
        &beta_one,  // Add to existing
        dx, dim_);

    // dW_x = x^T @ dv_all (swapped for correct transpose)
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha,
        x, dim_,
        dv_all, dim_,
        &beta_one,
        dW_x, dim_);

    // dW_h = h^T @ dv_all
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha,
        h, dim_,
        dv_all, dim_,
        &beta_one,
        dW_h, dim_);

    // Copy float gradients to output
    CopyFloatToT<T><<<(dim_ + 255) / 256, 256, 0, stream>>>(dim_, db_float, db);
    CopyFloatToT<T><<<(dim_ + 255) / 256, 256, 0, stream>>>(dim_, db_gate_float, d_b_gate);
}

// Explicit template instantiations
template struct StockElmanForward<__half>;
template struct StockElmanForward<__nv_bfloat16>;
template struct StockElmanForward<float>;
template struct StockElmanForward<double>;

template struct StockElmanBackward<__half>;
template struct StockElmanBackward<__nv_bfloat16>;
template struct StockElmanBackward<float>;
template struct StockElmanBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
