// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E6: Diagonal Elman - Per-channel scalar recurrence + low-rank mixing
//
// h_t = gate * h_{t-1} + (1 - gate) * x_t   (per-channel EMA)
// y_t = U @ V @ h_t * silu(x_t)             (low-rank cross-channel mix)
//
// Key insight: Diagonal recurrence is O(dim) - no GEMM needed!
// Only mixing requires GEMMs. Allows MASSIVE depth (~755 layers at 50M).

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

// Kernel: Diagonal recurrence (per-channel EMA)
// h_t = gate * h_{t-1} + (1 - gate) * x_t
// gate = sigmoid(gate_logit)
template<typename T>
__global__ void DiagonalRecurrenceKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ gate_logit,  // [dim] raw logits
    const T* __restrict__ x_t,         // [B, dim]
    const T* __restrict__ h_prev,      // [B, dim]
    T* __restrict__ h_out) {           // [B, dim]

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;
        float logit = static_cast<float>(gate_logit[d]);
        float gate = 1.0f / (1.0f + expf(-logit));
        float h_p = static_cast<float>(h_prev[idx]);
        float x = static_cast<float>(x_t[idx]);
        h_out[idx] = static_cast<T>(gate * h_p + (1.0f - gate) * x);
    }
}

// Kernel: Gated output y = mixed * silu(x)
template<typename T>
__global__ void DiagonalGateForward(
    const int batch_size,
    const int dim,
    const T* __restrict__ mixed,     // [B, dim] from U @ V @ h
    const T* __restrict__ x,         // [B, dim]
    T* __restrict__ output) {        // [B, dim]

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        float m = static_cast<float>(mixed[idx]);
        float x_val = static_cast<float>(x[idx]);
        float sigmoid_x = 1.0f / (1.0f + expf(-x_val));
        float silu_x = x_val * sigmoid_x;
        output[idx] = static_cast<T>(m * silu_x);
    }
}

// Kernel: Backward through gate
// d_mixed = d_output * silu(x)
// dx_gate = d_output * mixed * dsilu(x)
template<typename T>
__global__ void DiagonalGateBackward(
    const int batch_size,
    const int dim,
    const T* __restrict__ mixed,     // [B, dim]
    const T* __restrict__ x,         // [B, dim]
    const T* __restrict__ d_output,  // [B, dim]
    T* __restrict__ d_mixed,         // [B, dim]
    T* __restrict__ dx_gate) {       // [B, dim] gate contribution to dx

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        float m = static_cast<float>(mixed[idx]);
        float x_val = static_cast<float>(x[idx]);
        float dout = static_cast<float>(d_output[idx]);

        float sigmoid_x = 1.0f / (1.0f + expf(-x_val));
        float silu_x = x_val * sigmoid_x;
        float dsilu = sigmoid_x * (1.0f + x_val * (1.0f - sigmoid_x));

        d_mixed[idx] = static_cast<T>(dout * silu_x);
        dx_gate[idx] = static_cast<T>(dout * m * dsilu);
    }
}

// Kernel: Backward through diagonal recurrence
// dh_prev = (dh_mix + dh_rec) * gate
// dx_rec = (dh_mix + dh_rec) * (1 - gate)
// d_gate_logit += sum_batch((dh_mix + dh_rec) * (h_prev - x) * gate * (1 - gate))
template<typename T>
__global__ void DiagonalRecurrenceBackward(
    const int batch_size,
    const int dim,
    const T* __restrict__ gate_logit,  // [dim] raw logits
    const T* __restrict__ h_prev,      // [B, dim]
    const T* __restrict__ x_t,         // [B, dim]
    const T* __restrict__ dh_mix,      // [B, dim] gradient from mixing
    const T* __restrict__ dh_rec,      // [B, dim] gradient from next timestep (or null)
    T* __restrict__ dh_prev_out,       // [B, dim] gradient to h_{t-1}
    T* __restrict__ dx_rec,            // [B, dim] recurrence contribution to dx
    float* __restrict__ d_gate_logit) {// [dim] accumulates gradient

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float logit = static_cast<float>(gate_logit[d]);
        float gate = 1.0f / (1.0f + expf(-logit));
        float h_p = static_cast<float>(h_prev[idx]);
        float x = static_cast<float>(x_t[idx]);

        // Total gradient to h_t
        float dh = static_cast<float>(dh_mix[idx]);
        if (dh_rec != nullptr) {
            dh += static_cast<float>(dh_rec[idx]);
        }

        // dh_prev = dh * gate
        dh_prev_out[idx] = static_cast<T>(dh * gate);

        // dx_rec = dh * (1 - gate)
        dx_rec[idx] = static_cast<T>(dh * (1.0f - gate));

        // d_gate_logit: d/d_logit of gate = gate * (1 - gate) (sigmoid derivative)
        // dL/d_logit = dL/d_h * d_h/d_gate * d_gate/d_logit
        //            = dh * (h_prev - x) * gate * (1 - gate)
        float d_logit = dh * (h_p - x) * gate * (1.0f - gate);
        atomicAdd(&d_gate_logit[d], d_logit);
    }
}

// Kernel: Add two tensors element-wise
template<typename T>
__global__ void AddTensors(
    const int n,
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ out) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = static_cast<T>(static_cast<float>(a[idx]) + static_cast<float>(b[idx]));
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
// Diagonal Elman Forward
// =============================================================================

template<typename T>
DiagonalElmanForward<T>::DiagonalElmanForward(
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
void DiagonalElmanForward<T>::Run(
    int steps,
    const T* gate_logit,
    const T* U,
    const T* V,
    const T* x,
    T* h,
    T* output,
    T* workspace) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int BD = batch_size_ * dim_;
    const int BR = batch_size_ * rank_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    T* tmp_Vh = workspace;         // [B, rank]
    T* tmp_UVh = tmp_Vh + BR;      // [B, dim]

    // Sequential recurrence
    for (int t = 0; t < steps; ++t) {
        const T* h_prev = h + t * BD;
        T* h_curr = h + (t + 1) * BD;
        const T* x_t = x + t * BD;
        T* out_t = output + t * BD;

        // Step 1: Diagonal recurrence h_t = gate * h_{t-1} + (1 - gate) * x_t
        DiagonalRecurrenceKernel<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, gate_logit, x_t, h_prev, h_curr);

        // Step 2: V @ h_curr -> tmp_Vh  [B, rank] = [B, dim] @ [dim, rank]
        // V is [rank, dim], so V^T is [dim, rank]
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            rank_, batch_size_, dim_,
            &alpha, V, dim_,
            h_curr, dim_,
            &beta_zero, tmp_Vh, rank_);

        // Step 3: U @ tmp_Vh -> tmp_UVh  [B, dim] = [B, rank] @ [rank, dim]
        // U is [dim, rank], so U^T is [rank, dim]
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, rank_,
            &alpha, U, rank_,
            tmp_Vh, rank_,
            &beta_zero, tmp_UVh, dim_);

        // Step 4: output = mixed * silu(x)
        DiagonalGateForward<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, tmp_UVh, x_t, out_t);
    }
}

template struct DiagonalElmanForward<double>;
template struct DiagonalElmanForward<float>;
template struct DiagonalElmanForward<__nv_bfloat16>;
template struct DiagonalElmanForward<__half>;

// =============================================================================
// Diagonal Elman Backward
// =============================================================================

template<typename T>
DiagonalElmanBackward<T>::DiagonalElmanBackward(
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
void DiagonalElmanBackward<T>::Run(
    int steps,
    const T* gate_logit,
    const T* U,
    const T* V,
    const T* x,
    const T* h,
    const T* d_output,
    T* dx,
    T* d_gate_logit,
    T* dU,
    T* dV,
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
    // [d_mixed: BD] [dh_mix: BD] [dh_rec: BD] [dx_gate: BD] [dx_rec: BD]
    // [tmp_Vh: BR] [dVh: BR] [d_gate_f32: dim]
    T* d_mixed = workspace;
    T* dh_mix = d_mixed + BD;
    T* dh_rec = dh_mix + BD;
    T* dx_gate = dh_rec + BD;
    T* dx_rec = dx_gate + BD;
    T* tmp_Vh = dx_rec + BD;
    T* dVh = tmp_Vh + BR;
    float* d_gate_f32 = reinterpret_cast<float*>(dVh + BR);

    // Initialize gradients
    cudaMemsetAsync(dU, 0, dim_ * rank_ * sizeof(T), stream_);
    cudaMemsetAsync(dV, 0, rank_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(d_gate_f32, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dh_rec, 0, BD * sizeof(T), stream_);

    for (int t = steps - 1; t >= 0; --t) {
        const T* h_t = h + (t + 1) * BD;
        const T* h_prev = h + t * BD;
        const T* x_t = x + t * BD;
        const T* d_out_t = d_output + t * BD;
        T* dx_t = dx + t * BD;

        // Recompute Vh = V @ h_t for backward
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            rank_, batch_size_, dim_,
            &alpha, V, dim_,
            h_t, dim_,
            &beta_zero, tmp_Vh, rank_);

        // Recompute mixed = U @ Vh
        T* tmp_mixed = d_mixed;  // Reuse buffer for mixed
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, rank_,
            &alpha, U, rank_,
            tmp_Vh, rank_,
            &beta_zero, tmp_mixed, dim_);

        // Backward through gate: y = mixed * silu(x)
        // d_mixed = d_output * silu(x)
        // dx_gate = d_output * mixed * dsilu(x)
        DiagonalGateBackward<<<num_blocks_d, block_size, 0, stream_>>>(
            batch_size_, dim_, tmp_mixed, x_t, d_out_t, d_mixed, dx_gate);

        // Backward through mixing: mixed = U @ V @ h_t
        // dVh = d_mixed @ U: [B, rank] = [B, dim] @ [dim, rank]
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            rank_, batch_size_, dim_,
            &alpha, U, rank_,
            d_mixed, dim_,
            &beta_zero, dVh, rank_);

        // dU += tmp_Vh^T @ d_mixed: [rank, dim] stored as [dim, rank]
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_T,
            rank_, dim_, batch_size_,
            &alpha, tmp_Vh, rank_,
            d_mixed, dim_,
            &beta_one, dU, rank_);

        // dV += h_t^T @ dVh: [dim, rank] stored as [rank, dim]
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, rank_, batch_size_,
            &alpha, h_t, dim_,
            dVh, rank_,
            &beta_one, dV, dim_);

        // dh_mix from mixing = dVh @ V: [B, dim] = [B, rank] @ [rank, dim]
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, rank_,
            &alpha, V, dim_,
            dVh, rank_,
            &beta_zero, dh_mix, dim_);

        // Backward through diagonal recurrence
        // dh_prev = (dh_mix + dh_rec) * gate
        // dx_rec = (dh_mix + dh_rec) * (1 - gate)
        DiagonalRecurrenceBackward<<<num_blocks_d, block_size, 0, stream_>>>(
            batch_size_, dim_, gate_logit, h_prev, x_t,
            dh_mix,
            (t < steps - 1) ? dh_rec : nullptr,
            dh_rec,  // Output: becomes dh_rec for next iteration (t-1)
            dx_rec,
            d_gate_f32);

        // dx = dx_gate + dx_rec
        AddTensors<<<num_blocks_d, block_size, 0, stream_>>>(
            BD, dx_gate, dx_rec, dx_t);
    }

    // Convert float gradients to output type
    CopyFloatToT<<<num_blocks_dim, block_size, 0, stream_>>>(dim_, d_gate_f32, d_gate_logit);
}

template struct DiagonalElmanBackward<double>;
template struct DiagonalElmanBackward<float>;
template struct DiagonalElmanBackward<__nv_bfloat16>;
template struct DiagonalElmanBackward<__half>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
