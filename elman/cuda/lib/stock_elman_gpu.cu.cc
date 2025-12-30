// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// Level 0: Stock Elman - Basic tanh recurrence
// h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b)

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
    const T* x,
    T* h,
    T* v) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // =========================================================================
    // HASTE PATTERN: Pre-compute W_x @ x for ALL timesteps in ONE big GEMM!
    // This is much more efficient than T separate GEMMs.
    // x is [T, B, dim] treated as [T*B, dim]
    // tmp_Wx is [T, B, dim] = [T*B, dim]
    // =========================================================================
    T* tmp_Wx;
    T* tmp_Rh;
    cudaMalloc(&tmp_Wx, steps * BD * sizeof(T));
    cudaMalloc(&tmp_Rh, BD * sizeof(T));  // Only need one buffer, reused each step

    // One big GEMM: tmp_Wx = x @ W_x.T for all T*B rows at once
    // x: [T*B, dim], W_x: [dim, dim], tmp_Wx: [T*B, dim]
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha,
        W_x, dim_,
        x, dim_,
        &beta_zero,
        tmp_Wx, dim_);

    // Now process each timestep - only need recurrence GEMM per step
    for (int t = 0; t < steps; ++t) {
        const T* Wx_t = tmp_Wx + t * BD;  // Pre-computed W_x @ x_t
        const T* h_prev = h + t * BD;
        T* h_t = h + (t + 1) * BD;
        T* v_t = training_ ? (v + t * BD) : nullptr;

        // tmp_Rh = h_prev @ W_h.T (only recurrence GEMM per step)
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha,
            W_h, dim_,
            h_prev, dim_,
            &beta_zero,
            tmp_Rh, dim_);

        // h_t = tanh(Wx_t + tmp_Rh + b), cache pre-activation in v_t
        FusedTanhKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, Wx_t, tmp_Rh, b, h_t, v_t);
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
      stream_(stream) {}

template<typename T>
void StockElmanBackward<T>::Run(
    int steps,
    const T* W_x,
    const T* W_h,
    const T* x,
    const T* h,
    const T* v,
    const T* dh_out,
    T* dx,
    T* dW_x,
    T* dW_h,
    T* db) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Workspace for dv and dh_recurrent
    T* dv;
    T* dh_recurrent;
    cudaMalloc(&dv, BD * sizeof(T));
    cudaMalloc(&dh_recurrent, BD * sizeof(T));
    cudaMemset(dh_recurrent, 0, BD * sizeof(T));

    // Use float buffer for bias gradients
    float* db_float;
    cudaMalloc(&db_float, dim_ * sizeof(float));
    cudaMemset(db_float, 0, dim_ * sizeof(float));

    // Zero gradients
    cudaMemset(dW_x, 0, dim_ * dim_ * sizeof(T));
    cudaMemset(dW_h, 0, dim_ * dim_ * sizeof(T));

    // Backward through time
    for (int t = steps - 1; t >= 0; --t) {
        const T* x_t = x + t * BD;
        const T* h_prev = h + t * BD;
        const T* v_t = v + t * BD;
        const T* dh_t = dh_out + t * BD;
        T* dx_t = dx + t * BD;

        // Backward through tanh
        StockElmanBackwardKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, v_t, dh_t,
            (t < steps - 1) ? dh_recurrent : nullptr,
            dv, db_float);

        // dx_t = dv @ W_x (backward of x @ W_x.T)
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha,
            W_x, dim_,
            dv, dim_,
            &beta_zero,
            dx_t, dim_);

        // dh_recurrent = dv @ W_h (backward of h @ W_h.T)
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha,
            W_h, dim_,
            dv, dim_,
            &beta_zero,
            dh_recurrent, dim_);

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
    }

    // Copy float bias gradient to T type
    // Simple conversion kernel
    cudaMemset(db, 0, dim_ * sizeof(T));
    // For now just copy the float values (would need a proper conversion kernel for half/bf16)
    if constexpr (std::is_same<T, float>::value) {
        cudaMemcpy(db, db_float, dim_ * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    cudaFree(dv);
    cudaFree(dh_recurrent);
    cudaFree(db_float);
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
