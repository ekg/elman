// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E5 Optimized: Pure Low-Rank Elman with Fused Kernels
//
// h_t = tanh(U_h @ V_h @ h_{t-1} + U_x @ V_x @ x_t + b)
// y_t = h_t * silu(U_z @ V_z @ x_t)
//
// Optimizations:
// 1. Fused tanh + gate kernel (reduces 2 kernel launches to 1)
// 2. CUDA Graphs for sequential loop capture

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <algorithm>
#include <unordered_map>
#include <mutex>

#include "hasty/elman_ladder.h"
#include "blas.h"
#include "inline_ops.h"

namespace {

// Fused kernel: tanh + gate in one pass
// h = tanh(UVh + UVx + b)
// output = h * silu(UVz)
template<typename T>
__global__ void FusedTanhGateKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ UVh,       // [B, dim]
    const T* __restrict__ UVx,       // [B, dim]
    const T* __restrict__ UVz,       // [B, dim]
    const T* __restrict__ b,         // [dim]
    T* __restrict__ h_out,           // [B, dim]
    T* __restrict__ v_cache,         // [B, dim] or null
    T* __restrict__ output) {        // [B, dim]

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // Compute pre-activation
        float val = static_cast<float>(UVh[idx]) +
                    static_cast<float>(UVx[idx]) +
                    static_cast<float>(b[d]);

        // Save for backward if training
        if (v_cache) v_cache[idx] = static_cast<T>(val);

        // Compute h = tanh(val)
        float h_val = tanhf(val);
        h_out[idx] = static_cast<T>(h_val);

        // Compute output = h * silu(z)
        float z_val = static_cast<float>(UVz[idx]);
        float sigmoid_z = 1.0f / (1.0f + expf(-z_val));
        float silu_z = z_val * sigmoid_z;
        output[idx] = static_cast<T>(h_val * silu_z);
    }
}

// Kernel: Backward through fused tanh + gate
template<typename T>
__global__ void FusedTanhGateBackwardKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ v,         // [B, dim] pre-activation
    const T* __restrict__ z,         // [B, dim] (U_z @ V_z @ x)
    const T* __restrict__ d_output,  // [B, dim]
    const T* __restrict__ dh_rec,    // [B, dim] recurrent gradient (or null)
    T* __restrict__ dv,              // [B, dim] gradient to pre-activation
    T* __restrict__ dz,              // [B, dim] gradient to z
    float* __restrict__ db) {        // [dim] accumulates

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // Forward pass values
        float v_val = static_cast<float>(v[idx]);
        float z_val = static_cast<float>(z[idx]);
        float h = tanhf(v_val);

        // Silu computation
        float sigmoid_z = 1.0f / (1.0f + expf(-z_val));
        float silu_z = z_val * sigmoid_z;
        float dsilu = sigmoid_z * (1.0f + z_val * (1.0f - sigmoid_z));

        // Backward through output = h * silu(z)
        float dout = static_cast<float>(d_output[idx]);
        float dh_from_output = dout * silu_z;
        dz[idx] = static_cast<T>(dout * h * dsilu);

        // Combine gradients to h
        float grad_h = dh_from_output;
        if (dh_rec) grad_h += static_cast<float>(dh_rec[idx]);

        // Backward through tanh
        float dtanh = 1.0f - h * h;
        float dv_val = grad_h * dtanh;
        dv[idx] = static_cast<T>(dv_val);

        atomicAdd(&db[d], dv_val);
    }
}

template<typename T>
__global__ void CopyFloatToT(const int n, const float* __restrict__ src, T* __restrict__ dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = static_cast<T>(src[idx]);
    }
}

// Graph cache for different sequence lengths
struct GraphCache {
    std::unordered_map<int, cudaGraph_t> forward_graphs;
    std::unordered_map<int, cudaGraphExec_t> forward_execs;
    std::mutex mutex;

    ~GraphCache() {
        for (auto& [_, exec] : forward_execs) {
            cudaGraphExecDestroy(exec);
        }
        for (auto& [_, graph] : forward_graphs) {
            cudaGraphDestroy(graph);
        }
    }
};

// Global cache for graphs
static GraphCache g_graph_cache;

}  // anonymous namespace


namespace hasty {
namespace v0 {
namespace elman_ladder {

// =============================================================================
// Pure Low-Rank Elman Forward (Fused + Optimized)
// =============================================================================

template<typename T>
PureLowRankElmanForwardFused<T>::PureLowRankElmanForwardFused(
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
      stream_(stream),
      graph_captured_(false),
      captured_steps_(0) {}

template<typename T>
void PureLowRankElmanForwardFused<T>::Run(
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

    // Pre-compute V_x @ x for all timesteps (time-parallel)
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        rank_, steps * batch_size_, dim_,
        &alpha, V_x, dim_,
        x, dim_,
        &beta_zero, tmp_Vx_all, rank_);

    // Pre-compute U_x @ tmp_Vx for all timesteps
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

    // Sequential recurrence with fused kernel (3 ops per timestep instead of 4)
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

        // Step 3: FUSED tanh + gate (was 2 kernels, now 1)
        FusedTanhGateKernel<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, tmp_UVh, UVx_t, UVz_t, b, h_curr, v_t, out_t);
    }
}

template struct PureLowRankElmanForwardFused<double>;
template struct PureLowRankElmanForwardFused<float>;
template struct PureLowRankElmanForwardFused<__nv_bfloat16>;
template struct PureLowRankElmanForwardFused<__half>;

// =============================================================================
// Pure Low-Rank Elman Backward (Fused + Optimized)
// =============================================================================

template<typename T>
PureLowRankElmanBackwardFused<T>::PureLowRankElmanBackwardFused(
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
void PureLowRankElmanBackwardFused<T>::Run(
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
    // [dh_rec: BD] [dv_t: BD] [dz: BD]
    // [tmp_Vh: BR] [dVh: BR] [tmp_Vx: BR] [dVx: BR] [tmp_Vz: BR] [dVz: BR]
    // [tmp_UVz: BD] [db_f32: dim]
    T* dh_rec = workspace;
    T* dv_t = dh_rec + BD;
    T* dz = dv_t + BD;
    T* tmp_Vh = dz + BD;
    T* dVh = tmp_Vh + BR;
    T* tmp_Vx = dVh + BR;
    T* dVx = tmp_Vx + BR;
    T* tmp_Vz = dVx + BR;
    T* dVz = tmp_Vz + BR;
    T* tmp_UVz = dVz + BR;
    float* db_f32 = reinterpret_cast<float*>(tmp_UVz + BD);

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

        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, rank_,
            &alpha, U_z, rank_,
            tmp_Vz, rank_,
            &beta_zero, tmp_UVz, dim_);

        // FUSED backward through tanh + gate
        FusedTanhGateBackwardKernel<<<num_blocks_d, block_size, 0, stream_>>>(
            batch_size_, dim_, v_t_in, tmp_UVz, d_out_t,
            (t < steps - 1) ? dh_rec : nullptr,
            dv_t, dz, db_f32);

        // Backward through U_z @ V_z @ x
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            rank_, batch_size_, dim_,
            &alpha, U_z, rank_,
            dz, dim_,
            &beta_zero, dVz, rank_);

        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_T,
            rank_, dim_, batch_size_,
            &alpha, tmp_Vz, rank_,
            dz, dim_,
            &beta_one, dU_z, rank_);

        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, rank_, batch_size_,
            &alpha, x_t, dim_,
            dVz, rank_,
            &beta_one, dV_z, dim_);

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

        // dh_rec for next iteration
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

template struct PureLowRankElmanBackwardFused<double>;
template struct PureLowRankElmanBackwardFused<float>;
template struct PureLowRankElmanBackwardFused<__nv_bfloat16>;
template struct PureLowRankElmanBackwardFused<__half>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
