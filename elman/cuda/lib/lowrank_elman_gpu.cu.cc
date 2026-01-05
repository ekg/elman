// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E4: Low-Rank Elman - SVD-style low-rank W_h for fat hidden state
//
// h_t = tanh(W_x @ x_t + U @ V @ h_{t-1} + b)
// output = h_t * silu(z_t)
//
// Key insight: U is [D, R], V is [R, D]. With same param count as E1,
// E4 can have 2x the hidden dimension (fat hidden state).

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

// Kernel: Fused Wx + UVh + bias + tanh
template<typename T>
__global__ void FusedLowRankTanhKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ Wx,
    const T* __restrict__ UVh,
    const T* __restrict__ b,
    T* __restrict__ h_out,
    T* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;
        float val = static_cast<float>(Wx[idx]) + static_cast<float>(UVh[idx]) + static_cast<float>(b[d]);
        if (v_cache) v_cache[idx] = static_cast<T>(val);
        h_out[idx] = static_cast<T>(tanhf(val));
    }
}

// Kernel: output = h * silu(z)
template<typename T>
__global__ void LowRankGateForward(
    const int batch_size,
    const int dim,
    const T* __restrict__ h,
    const T* __restrict__ z,
    T* __restrict__ output) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        float h_val = static_cast<float>(h[idx]);
        float z_val = static_cast<float>(z[idx]);
        float sigmoid_z = 1.0f / (1.0f + expf(-z_val));
        float silu_z = z_val * sigmoid_z;
        output[idx] = static_cast<T>(h_val * silu_z);
    }
}

// Kernel: Backward through tanh
template<typename T>
__global__ void LowRankTanhBackwardKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ v,
    const T* __restrict__ dh,
    const T* __restrict__ dh_recurrent,
    T* __restrict__ dv,
    float* __restrict__ db) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float grad = static_cast<float>(dh[idx]);
        if (dh_recurrent) grad += static_cast<float>(dh_recurrent[idx]);

        float h = tanhf(static_cast<float>(v[idx]));
        float dtanh = 1.0f - h * h;
        float dv_val = grad * dtanh;
        dv[idx] = static_cast<T>(dv_val);

        atomicAdd(&db[d], dv_val);
    }
}

// Kernel: Backward through gate
template<typename T>
__global__ void LowRankGateBackward(
    const int batch_size,
    const int dim,
    const T* __restrict__ h,
    const T* __restrict__ z,
    const T* __restrict__ d_output,
    T* __restrict__ dh,
    T* __restrict__ dz) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        float h_val = static_cast<float>(h[idx]);
        float z_val = static_cast<float>(z[idx]);
        float dout = static_cast<float>(d_output[idx]);

        float sigmoid_z = 1.0f / (1.0f + expf(-z_val));
        float silu_z = z_val * sigmoid_z;
        float dsilu = sigmoid_z * (1.0f + z_val * (1.0f - sigmoid_z));

        dh[idx] = static_cast<T>(dout * silu_z);
        dz[idx] = static_cast<T>(dout * h_val * dsilu);
    }
}

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
// Low-Rank Elman Forward
// =============================================================================

template<typename T>
LowRankElmanForward<T>::LowRankElmanForward(
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
void LowRankElmanForward<T>::Run(
    int steps,
    const T* W_x,
    const T* U,
    const T* V,
    const T* b,
    const T* x,
    const T* z,
    T* h,
    T* output,
    T* v,
    T* workspace) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int BD = batch_size_ * dim_;
    const int BR = batch_size_ * rank_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    T* tmp_Wx = workspace;
    T* tmp_Vh = workspace + steps * BD;
    T* tmp_UVh = tmp_Vh + BR;

    // Pre-compute W_x @ x for all timesteps: [T*B, dim] = [T*B, dim] @ [dim, dim]^T
    // C = A @ B^T where A is [T*B, dim], B is [dim, dim]
    // cuBLAS: C^T = B^T @ A^T, so CUBLAS_OP_T on W_x, CUBLAS_OP_N on x
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha, W_x, dim_,
        x, dim_,
        &beta_zero, tmp_Wx, dim_);

    // Sequential recurrence
    for (int t = 0; t < steps; ++t) {
        const T* h_prev = h + t * BD;
        T* h_curr = h + (t + 1) * BD;
        const T* Wx_t = tmp_Wx + t * BD;
        const T* z_t = z + t * BD;
        T* v_t = training_ ? v + t * BD : nullptr;
        T* out_t = output + t * BD;

        // Step 1: V @ h_prev -> tmp_Vh  [B, rank] = [B, dim] @ [dim, rank]
        // V is [rank, dim], so V^T is [dim, rank]
        // C = A @ B^T where A is [B, dim], B is [rank, dim]
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            rank_, batch_size_, dim_,
            &alpha, V, dim_,
            h_prev, dim_,
            &beta_zero, tmp_Vh, rank_);

        // Step 2: U @ tmp_Vh -> tmp_UVh  [B, dim] = [B, rank] @ [rank, dim]
        // U is [dim, rank], so U^T is [rank, dim]
        // C = A @ B^T where A is [B, rank], B is [dim, rank]
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, rank_,
            &alpha, U, rank_,
            tmp_Vh, rank_,
            &beta_zero, tmp_UVh, dim_);

        // Step 3: Fused Wx + UVh + b -> tanh -> h_curr
        FusedLowRankTanhKernel<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, Wx_t, tmp_UVh, b, h_curr, v_t);

        // Step 4: h * silu(z) -> output
        LowRankGateForward<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_curr, z_t, out_t);
    }
}

template struct LowRankElmanForward<double>;
template struct LowRankElmanForward<float>;
template struct LowRankElmanForward<__nv_bfloat16>;
template struct LowRankElmanForward<__half>;

// =============================================================================
// Low-Rank Elman Backward
// =============================================================================

template<typename T>
LowRankElmanBackward<T>::LowRankElmanBackward(
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
void LowRankElmanBackward<T>::Run(
    int steps,
    const T* W_x,
    const T* U,
    const T* V,
    const T* x,
    const T* z,
    const T* h,
    const T* v,
    const T* d_output,
    T* dx,
    T* dz,
    T* dW_x,
    T* dU,
    T* dV,
    T* db,
    T* workspace) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int BD = batch_size_ * dim_;
    const int BR = batch_size_ * rank_;
    const int block_size = 256;
    const int num_blocks_d = (BD + block_size - 1) / block_size;
    const int num_blocks_dim = (dim_ + block_size - 1) / block_size;

    T* dh_curr = workspace;
    T* dh_next = dh_curr + BD;
    T* dv_t = dh_next + BD;
    T* dVh = dv_t + BD;
    float* db_f32 = reinterpret_cast<float*>(dVh + BR);

    cudaMemsetAsync(dW_x, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dU, 0, dim_ * rank_ * sizeof(T), stream_);
    cudaMemsetAsync(dV, 0, rank_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(db_f32, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dh_next, 0, BD * sizeof(T), stream_);

    for (int t = steps - 1; t >= 0; --t) {
        const T* h_t = h + (t + 1) * BD;
        const T* h_prev = h + t * BD;
        const T* z_t = z + t * BD;
        const T* v_t = v + t * BD;
        const T* d_out_t = d_output + t * BD;
        T* dx_t = dx + t * BD;
        T* dz_t = dz + t * BD;

        // Backward through gate
        LowRankGateBackward<<<num_blocks_d, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, z_t, d_out_t, dh_curr, dz_t);

        // Backward through tanh
        LowRankTanhBackwardKernel<<<num_blocks_d, block_size, 0, stream_>>>(
            batch_size_, dim_, v_t, dh_curr, (t < steps - 1) ? dh_next : nullptr, dv_t, db_f32);

        // dW_x += dv_t^T @ x_t: cuBLAS writes col-major, so compute x^T @ dv
        // When interpreted as row-major: (x^T @ dv)^T = dv^T @ x
        const T* x_t = x + t * BD;
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_,
            &alpha, x_t, dim_,
            dv_t, dim_,
            &beta_one, dW_x, dim_);

        // dx_t = dv_t @ W_x: [B, dim] = [B, dim] @ [dim, dim]
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha, W_x, dim_,
            dv_t, dim_,
            &beta_zero, dx_t, dim_);

        // Compute V @ h_prev again for dU
        T* tmp_Vh = dVh;
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            rank_, batch_size_, dim_,
            &alpha, V, dim_,
            h_prev, dim_,
            &beta_zero, tmp_Vh, rank_);

        // dU += tmp_Vh^T @ dv_t: [rank, dim] stored as [dim, rank] row-major
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_T,
            rank_, dim_, batch_size_,
            &alpha, tmp_Vh, rank_,
            dv_t, dim_,
            &beta_one, dU, rank_);

        // dVh = dv_t @ U: [B, rank] = [B, dim] @ [dim, rank]
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            rank_, batch_size_, dim_,
            &alpha, U, rank_,
            dv_t, dim_,
            &beta_zero, dVh, rank_);

        // dV += h_prev^T @ dVh: [dim, rank] stored as [rank, dim] row-major
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, rank_, batch_size_,
            &alpha, h_prev, dim_,
            dVh, rank_,
            &beta_one, dV, dim_);

        // dh_prev = dVh @ V: [B, dim] = [B, rank] @ [rank, dim]
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, rank_,
            &alpha, V, dim_,
            dVh, rank_,
            &beta_zero, dh_next, dim_);
    }

    CopyFloatToT<<<num_blocks_dim, block_size, 0, stream_>>>(dim_, db_f32, db);
}

template struct LowRankElmanBackward<double>;
template struct LowRankElmanBackward<float>;
template struct LowRankElmanBackward<__nv_bfloat16>;
template struct LowRankElmanBackward<__half>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
