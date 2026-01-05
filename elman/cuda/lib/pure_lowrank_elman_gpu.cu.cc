// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E5: Pure Low-Rank Elman - No projections, all low-rank on full dim.
//
// h_t = tanh(U_h @ V_h @ h_{t-1} + U_x @ V_x @ x_t + b)
// y_t = h_t * silu(U_z @ V_z @ x_t)
//
// Key insight: No in_proj/out_proj. Hidden state IS dim.
// All matrices factored as U @ V (low-rank).
// With rank=64, dim=512: 197k params/layer -> 252 layers at 50M.

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

// Kernel: Fused add + tanh
// h = tanh(UVh + UVx + b)
template<typename T>
__global__ void FusedLowRankTanhKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ UVh,       // [B, dim]
    const T* __restrict__ UVx,       // [B, dim]
    const T* __restrict__ b,         // [dim]
    T* __restrict__ h_out,           // [B, dim]
    T* __restrict__ v_cache) {       // [B, dim] or null

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;
        float val = static_cast<float>(UVh[idx]) +
                    static_cast<float>(UVx[idx]) +
                    static_cast<float>(b[d]);
        if (v_cache) v_cache[idx] = static_cast<T>(val);
        h_out[idx] = static_cast<T>(tanhf(val));
    }
}

// Kernel: output = h * silu(z)
template<typename T>
__global__ void PureLowRankGateForward(
    const int batch_size,
    const int dim,
    const T* __restrict__ h,         // [B, dim]
    const T* __restrict__ z,         // [B, dim] (U_z @ V_z @ x)
    T* __restrict__ output) {        // [B, dim]

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
__global__ void PureLowRankTanhBackwardKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ v,         // [B, dim] pre-activation
    const T* __restrict__ dh,        // [B, dim] gradient into h
    const T* __restrict__ dh_rec,    // [B, dim] recurrent gradient (or null)
    T* __restrict__ dv,              // [B, dim] gradient to pre-activation
    float* __restrict__ db) {        // [dim] accumulates

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float grad = static_cast<float>(dh[idx]);
        if (dh_rec) grad += static_cast<float>(dh_rec[idx]);

        float h = tanhf(static_cast<float>(v[idx]));
        float dtanh = 1.0f - h * h;
        float dv_val = grad * dtanh;
        dv[idx] = static_cast<T>(dv_val);

        atomicAdd(&db[d], dv_val);
    }
}

// Kernel: Backward through gate
template<typename T>
__global__ void PureLowRankGateBackward(
    const int batch_size,
    const int dim,
    const T* __restrict__ h,         // [B, dim]
    const T* __restrict__ z,         // [B, dim]
    const T* __restrict__ d_output,  // [B, dim]
    T* __restrict__ dh,              // [B, dim]
    T* __restrict__ dz) {            // [B, dim]

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
// Pure Low-Rank Elman Forward
// =============================================================================

template<typename T>
PureLowRankElmanForward<T>::PureLowRankElmanForward(
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
void PureLowRankElmanForward<T>::Run(
    int steps,
    const T* U_h,
    const T* V_h,
    const T* U_x,
    const T* V_x,
    const T* U_z,
    const T* V_z,
    const T* b,
    const T* x,
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

    // Workspace layout:
    // [tmp_Vx_all: T*BR] [tmp_UVx_all: T*BD] [tmp_Vz_all: T*BR] [tmp_UVz_all: T*BD]
    // [tmp_Vh: BR] [tmp_UVh: BD]
    T* tmp_Vx_all = workspace;
    T* tmp_UVx_all = tmp_Vx_all + steps * BR;
    T* tmp_Vz_all = tmp_UVx_all + steps * BD;
    T* tmp_UVz_all = tmp_Vz_all + steps * BR;
    T* tmp_Vh = tmp_UVz_all + steps * BD;
    T* tmp_UVh = tmp_Vh + BR;

    // Pre-compute V_x @ x for all timesteps: [T*B, rank] = [T*B, dim] @ [dim, rank]
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        rank_, steps * batch_size_, dim_,
        &alpha, V_x, dim_,
        x, dim_,
        &beta_zero, tmp_Vx_all, rank_);

    // Pre-compute U_x @ tmp_Vx for all timesteps: [T*B, dim] = [T*B, rank] @ [rank, dim]
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, rank_,
        &alpha, U_x, rank_,
        tmp_Vx_all, rank_,
        &beta_zero, tmp_UVx_all, dim_);

    // Pre-compute V_z @ x for all timesteps
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        rank_, steps * batch_size_, dim_,
        &alpha, V_z, dim_,
        x, dim_,
        &beta_zero, tmp_Vz_all, rank_);

    // Pre-compute U_z @ tmp_Vz for all timesteps
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, rank_,
        &alpha, U_z, rank_,
        tmp_Vz_all, rank_,
        &beta_zero, tmp_UVz_all, dim_);

    // Sequential recurrence
    for (int t = 0; t < steps; ++t) {
        const T* h_prev = h + t * BD;
        T* h_curr = h + (t + 1) * BD;
        const T* UVx_t = tmp_UVx_all + t * BD;
        const T* UVz_t = tmp_UVz_all + t * BD;
        T* v_t = training_ ? v + t * BD : nullptr;
        T* out_t = output + t * BD;

        // Step 1: V_h @ h_prev -> tmp_Vh
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            rank_, batch_size_, dim_,
            &alpha, V_h, dim_,
            h_prev, dim_,
            &beta_zero, tmp_Vh, rank_);

        // Step 2: U_h @ tmp_Vh -> tmp_UVh
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, rank_,
            &alpha, U_h, rank_,
            tmp_Vh, rank_,
            &beta_zero, tmp_UVh, dim_);

        // Step 3: Fused UVh + UVx + b -> tanh -> h_curr
        FusedLowRankTanhKernel<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, tmp_UVh, UVx_t, b, h_curr, v_t);

        // Step 4: h * silu(UVz) -> output
        PureLowRankGateForward<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_curr, UVz_t, out_t);
    }
}

template struct PureLowRankElmanForward<double>;
template struct PureLowRankElmanForward<float>;
template struct PureLowRankElmanForward<__nv_bfloat16>;
template struct PureLowRankElmanForward<__half>;

// =============================================================================
// Pure Low-Rank Elman Backward
// =============================================================================

template<typename T>
PureLowRankElmanBackward<T>::PureLowRankElmanBackward(
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
void PureLowRankElmanBackward<T>::Run(
    int steps,
    const T* U_h,
    const T* V_h,
    const T* U_x,
    const T* V_x,
    const T* U_z,
    const T* V_z,
    const T* x,
    const T* h,
    const T* v,
    const T* d_output,
    T* dx,
    T* dU_h,
    T* dV_h,
    T* dU_x,
    T* dV_x,
    T* dU_z,
    T* dV_z,
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

    // Workspace layout:
    // [dh_curr: BD] [dh_rec: BD] [dv_t: BD] [dz: BD]
    // [tmp_Vh: BR] [dVh: BR] [tmp_Vx: BR] [dVx: BR] [tmp_Vz: BR] [dVz: BR]
    // [db_f32: dim]
    T* dh_curr = workspace;
    T* dh_rec = dh_curr + BD;
    T* dv_t = dh_rec + BD;
    T* dz = dv_t + BD;
    T* tmp_Vh = dz + BD;
    T* dVh = tmp_Vh + BR;
    T* tmp_Vx = dVh + BR;
    T* dVx = tmp_Vx + BR;
    T* tmp_Vz = dVx + BR;
    T* dVz = tmp_Vz + BR;
    float* db_f32 = reinterpret_cast<float*>(dVz + BR);

    // Initialize gradients
    cudaMemsetAsync(dU_h, 0, dim_ * rank_ * sizeof(T), stream_);
    cudaMemsetAsync(dV_h, 0, rank_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dU_x, 0, dim_ * rank_ * sizeof(T), stream_);
    cudaMemsetAsync(dV_x, 0, rank_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dU_z, 0, dim_ * rank_ * sizeof(T), stream_);
    cudaMemsetAsync(dV_z, 0, rank_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(db_f32, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dh_rec, 0, BD * sizeof(T), stream_);
    cudaMemsetAsync(dx, 0, steps * BD * sizeof(T), stream_);

    for (int t = steps - 1; t >= 0; --t) {
        const T* h_t = h + (t + 1) * BD;
        const T* h_prev = h + t * BD;
        const T* v_t_in = v + t * BD;
        const T* x_t = x + t * BD;
        const T* d_out_t = d_output + t * BD;
        T* dx_t = dx + t * BD;

        // Recompute UVz for backward
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            rank_, batch_size_, dim_,
            &alpha, V_z, dim_,
            x_t, dim_,
            &beta_zero, tmp_Vz, rank_);

        T* tmp_UVz = dz;  // Reuse buffer
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, rank_,
            &alpha, U_z, rank_,
            tmp_Vz, rank_,
            &beta_zero, tmp_UVz, dim_);

        // Backward through gate: y = h * silu(z)
        PureLowRankGateBackward<<<num_blocks_d, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, tmp_UVz, d_out_t, dh_curr, dz);

        // Backward through tanh
        PureLowRankTanhBackwardKernel<<<num_blocks_d, block_size, 0, stream_>>>(
            batch_size_, dim_, v_t_in, dh_curr, (t < steps - 1) ? dh_rec : nullptr, dv_t, db_f32);

        // Backward through U_z @ V_z @ x
        // dVz = dz @ U_z: [B, rank] = [B, dim] @ [dim, rank]
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            rank_, batch_size_, dim_,
            &alpha, U_z, rank_,
            dz, dim_,
            &beta_zero, dVz, rank_);

        // dU_z += tmp_Vz^T @ dz
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_T,
            rank_, dim_, batch_size_,
            &alpha, tmp_Vz, rank_,
            dz, dim_,
            &beta_one, dU_z, rank_);

        // dV_z += x^T @ dVz
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, rank_, batch_size_,
            &alpha, x_t, dim_,
            dVz, rank_,
            &beta_one, dV_z, dim_);

        // dx += dVz @ V_z
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, rank_,
            &alpha, V_z, dim_,
            dVz, rank_,
            &beta_one, dx_t, dim_);

        // Backward through U_x @ V_x @ x
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            rank_, batch_size_, dim_,
            &alpha, V_x, dim_,
            x_t, dim_,
            &beta_zero, tmp_Vx, rank_);

        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            rank_, batch_size_, dim_,
            &alpha, U_x, rank_,
            dv_t, dim_,
            &beta_zero, dVx, rank_);

        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_T,
            rank_, dim_, batch_size_,
            &alpha, tmp_Vx, rank_,
            dv_t, dim_,
            &beta_one, dU_x, rank_);

        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, rank_, batch_size_,
            &alpha, x_t, dim_,
            dVx, rank_,
            &beta_one, dV_x, dim_);

        // dx += dVx @ V_x
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, rank_,
            &alpha, V_x, dim_,
            dVx, rank_,
            &beta_one, dx_t, dim_);

        // Backward through U_h @ V_h @ h_prev
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            rank_, batch_size_, dim_,
            &alpha, V_h, dim_,
            h_prev, dim_,
            &beta_zero, tmp_Vh, rank_);

        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            rank_, batch_size_, dim_,
            &alpha, U_h, rank_,
            dv_t, dim_,
            &beta_zero, dVh, rank_);

        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_T,
            rank_, dim_, batch_size_,
            &alpha, tmp_Vh, rank_,
            dv_t, dim_,
            &beta_one, dU_h, rank_);

        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, rank_, batch_size_,
            &alpha, h_prev, dim_,
            dVh, rank_,
            &beta_one, dV_h, dim_);

        // dh_rec = dVh @ V_h (gradient to h_prev for next iteration)
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, rank_,
            &alpha, V_h, dim_,
            dVh, rank_,
            &beta_zero, dh_rec, dim_);
    }

    // Convert float gradients to output type
    CopyFloatToT<<<num_blocks_dim, block_size, 0, stream_>>>(dim_, db_f32, db);
}

template struct PureLowRankElmanBackward<double>;
template struct PureLowRankElmanBackward<float>;
template struct PureLowRankElmanBackward<__nv_bfloat16>;
template struct PureLowRankElmanBackward<__half>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
