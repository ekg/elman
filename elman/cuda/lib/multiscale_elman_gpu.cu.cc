// Copyright 2026 Erik Garrison. Apache 2.0 License.
//
// Level 10: Multi-Scale EMA Elman
//
// Architecture:
//   h_t = tanh(W_x @ x_t + W_h @ h_prev + b)  -- same as E1
//   m_i_t = alpha_i * m_i_prev + (1 - alpha_i) * h_t  -- k EMA banks
//   out = h * silu(z) + sum_i(m_i * silu(z_i))
//
// Each memory bank has learned per-dimension decay alpha_i = sigmoid(a_i)
// This provides multi-timescale memory with zero additional GEMMs.

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

// Kernel: Fused tanh activation for RNN core (same as E1)
template<typename T>
__global__ void CoreTanhKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ Wx,      // [B, dim]
    const T* __restrict__ Rh,      // [B, dim]
    const T* __restrict__ b,       // [dim]
    T* __restrict__ h_out,         // [B, dim]
    T* __restrict__ v_cache) {     // [B, dim] pre-activation cache

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;
        float val = static_cast<float>(Wx[idx]) + static_cast<float>(Rh[idx]) + static_cast<float>(b[d]);
        if (v_cache) v_cache[idx] = static_cast<T>(val);
        h_out[idx] = static_cast<T>(tanhf(val));
    }
}

// Kernel: EMA memory update for all banks
// m_i = alpha_i * m_i_prev + (1 - alpha_i) * h
template<typename T>
__global__ void EMAUpdateKernel(
    const int batch_size,
    const int dim,
    const int n_banks,
    const T* __restrict__ a,           // [n_banks, dim] decay logits
    const T* __restrict__ h,           // [B, dim] current hidden state
    const T* __restrict__ m_prev,      // [n_banks, B, dim] previous memories
    T* __restrict__ m_out) {           // [n_banks, B, dim] output memories

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_banks * batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;
        const int b = (idx / dim) % batch_size;
        const int bank = idx / (batch_size * dim);

        const int h_idx = b * dim + d;
        const int a_idx = bank * dim + d;

        float a_val = static_cast<float>(a[a_idx]);
        float alpha = 1.0f / (1.0f + expf(-a_val));  // sigmoid
        float h_val = static_cast<float>(h[h_idx]);
        float m_prev_val = static_cast<float>(m_prev[idx]);

        m_out[idx] = static_cast<T>(alpha * m_prev_val + (1.0f - alpha) * h_val);
    }
}

// Kernel: Gated output combining h and all memory banks
// out = h * silu(z) + sum_i(m_i * silu(z_i))
template<typename T>
__global__ void MultiScaleGateKernel(
    const int batch_size,
    const int dim,
    const int n_banks,
    const T* __restrict__ h,       // [B, dim]
    const T* __restrict__ z,       // [B, (1+n_banks)*dim] gates for h and each m_i
    const T* __restrict__ m,       // [n_banks, B, dim]
    T* __restrict__ output) {      // [B, dim]

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;
        const int b = idx / dim;

        // Gate for h
        float z_h = static_cast<float>(z[b * (1 + n_banks) * dim + d]);
        float sigmoid_zh = 1.0f / (1.0f + expf(-z_h));
        float silu_zh = z_h * sigmoid_zh;
        float h_val = static_cast<float>(h[idx]);
        float out_val = h_val * silu_zh;

        // Add gated memories
        for (int i = 0; i < n_banks; i++) {
            int z_idx = b * (1 + n_banks) * dim + (1 + i) * dim + d;
            int m_idx = i * batch_size * dim + idx;

            float z_mi = static_cast<float>(z[z_idx]);
            float sigmoid_zmi = 1.0f / (1.0f + expf(-z_mi));
            float silu_zmi = z_mi * sigmoid_zmi;
            float m_val = static_cast<float>(m[m_idx]);
            out_val += m_val * silu_zmi;
        }

        output[idx] = static_cast<T>(out_val);
    }
}

// Backward through tanh
template<typename T>
__global__ void CoreTanhBackwardKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ v,       // [B, dim] pre-activation
    const T* __restrict__ dh,      // [B, dim] gradient to h
    T* __restrict__ dv,            // [B, dim] gradient to pre-activation
    float* __restrict__ db) {      // [dim] bias gradient (accumulated)

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;
        float h = tanhf(static_cast<float>(v[idx]));
        float dtanh = 1.0f - h * h;
        float dv_val = static_cast<float>(dh[idx]) * dtanh;
        dv[idx] = static_cast<T>(dv_val);
        atomicAdd(&db[d], dv_val);
    }
}

// Backward through EMA: m = alpha * m_prev + (1 - alpha) * h
template<typename T>
__global__ void EMABackwardKernel(
    const int batch_size,
    const int dim,
    const int n_banks,
    const T* __restrict__ a,           // [n_banks, dim] decay logits
    const T* __restrict__ h,           // [B, dim]
    const T* __restrict__ m_prev,      // [n_banks, B, dim]
    const T* __restrict__ dm,          // [n_banks, B, dim] gradient to m
    float* __restrict__ dh_accum,      // [B, dim] float buffer for accumulated gradient to h
    T* __restrict__ dm_prev_out,       // [n_banks, B, dim] gradient to m_prev
    float* __restrict__ da) {          // [n_banks, dim] gradient to decay logits

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_banks * batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;
        const int b = (idx / dim) % batch_size;
        const int bank = idx / (batch_size * dim);

        const int h_idx = b * dim + d;
        const int a_idx = bank * dim + d;

        float a_val = static_cast<float>(a[a_idx]);
        float alpha = 1.0f / (1.0f + expf(-a_val));
        float dalpha = alpha * (1.0f - alpha);  // sigmoid derivative

        float h_val = static_cast<float>(h[h_idx]);
        float m_prev_val = static_cast<float>(m_prev[idx]);
        float dm_val = static_cast<float>(dm[idx]);

        // d/d(m_prev) = alpha * dm
        dm_prev_out[idx] = static_cast<T>(alpha * dm_val);

        // d/d(h) = (1 - alpha) * dm (accumulated across banks in float precision)
        atomicAdd(&dh_accum[h_idx], (1.0f - alpha) * dm_val);

        // d/d(a) = (m_prev - h) * dalpha * dm
        float da_val = (m_prev_val - h_val) * dalpha * dm_val;
        atomicAdd(&da[a_idx], da_val);
    }
}

// Kernel to copy float buffer to type T and add to existing
template<typename T>
__global__ void FloatToTypeAddKernel(
    const int n,
    const float* __restrict__ src,
    T* __restrict__ dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = static_cast<T>(static_cast<float>(dst[idx]) + src[idx]);
    }
}

// Backward through multi-scale gate
template<typename T>
__global__ void MultiScaleGateBackwardKernel(
    const int batch_size,
    const int dim,
    const int n_banks,
    const T* __restrict__ h,           // [B, dim]
    const T* __restrict__ z,           // [B, (1+n_banks)*dim]
    const T* __restrict__ m,           // [n_banks, B, dim]
    const T* __restrict__ d_output,    // [B, dim]
    T* __restrict__ dh,                // [B, dim]
    T* __restrict__ dz,                // [B, (1+n_banks)*dim]
    T* __restrict__ dm) {              // [n_banks, B, dim]

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;
        const int b = idx / dim;

        float dout = static_cast<float>(d_output[idx]);

        // Gradient for h branch
        float z_h = static_cast<float>(z[b * (1 + n_banks) * dim + d]);
        float sigmoid_zh = 1.0f / (1.0f + expf(-z_h));
        float silu_zh = z_h * sigmoid_zh;
        float dsilu_zh = sigmoid_zh * (1.0f + z_h * (1.0f - sigmoid_zh));
        float h_val = static_cast<float>(h[idx]);

        dh[idx] = static_cast<T>(dout * silu_zh);
        dz[b * (1 + n_banks) * dim + d] = static_cast<T>(dout * h_val * dsilu_zh);

        // Gradients for memory branches
        for (int i = 0; i < n_banks; i++) {
            int z_idx = b * (1 + n_banks) * dim + (1 + i) * dim + d;
            int m_idx = i * batch_size * dim + idx;

            float z_mi = static_cast<float>(z[z_idx]);
            float sigmoid_zmi = 1.0f / (1.0f + expf(-z_mi));
            float silu_zmi = z_mi * sigmoid_zmi;
            float dsilu_zmi = sigmoid_zmi * (1.0f + z_mi * (1.0f - sigmoid_zmi));
            float m_val = static_cast<float>(m[m_idx]);

            dm[m_idx] = static_cast<T>(dout * silu_zmi);
            dz[z_idx] = static_cast<T>(dout * m_val * dsilu_zmi);
        }
    }
}

template<typename T>
__global__ void ZeroKernel(const int n, T* data) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = static_cast<T>(0);
}

template<typename T>
__global__ void AddKernel(const int n, T* a, const T* b) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = static_cast<T>(static_cast<float>(a[idx]) + static_cast<float>(b[idx]));
    }
}

template<typename T>
__global__ void CopyFloatToT(const int n, const float* src, T* dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = static_cast<T>(src[idx]);
}

}  // anonymous namespace


namespace hasty {
namespace v0 {
namespace elman_ladder {

// =============================================================================
// Multi-Scale EMA Elman Forward
// =============================================================================

template<typename T>
MultiScaleElmanForward<T>::MultiScaleElmanForward(
    bool training,
    int batch_size,
    int dim,
    int n_banks,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : training_(training),
      batch_size_(batch_size),
      dim_(dim),
      n_banks_(n_banks),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void MultiScaleElmanForward<T>::Run(
    int steps,
    const T* W_x,        // [dim, dim]
    const T* W_h,        // [dim, dim]
    const T* b,          // [dim]
    const T* a,          // [n_banks, dim] EMA decay logits
    const T* x,          // [T, B, dim]
    const T* z,          // [T, B, (1+n_banks)*dim] gates
    T* h,                // [T+1, B, dim] hidden states
    T* m,                // [T+1, n_banks, B, dim] memory banks
    T* output,           // [T, B, dim]
    T* v,                // [T, B, dim] pre-activation cache
    T* workspace) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks_bd = (BD + block_size - 1) / block_size;
    const int num_blocks_mem = (n_banks_ * BD + block_size - 1) / block_size;

    // Workspace: [tmp_Wx: T*B*dim] [tmp_Rh: B*dim]
    T* tmp_Wx = workspace;
    T* tmp_Rh = workspace + steps * BD;

    // Pre-compute W_x @ x for all timesteps
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha,
        W_x, dim_,
        x, dim_,
        &beta_zero,
        tmp_Wx, dim_);

    // Process each timestep
    for (int t = 0; t < steps; ++t) {
        const T* Wx_t = tmp_Wx + t * BD;
        const T* h_prev = h + t * BD;
        const T* z_t = z + t * batch_size_ * (1 + n_banks_) * dim_;
        const T* m_prev = m + t * n_banks_ * BD;

        T* h_t = h + (t + 1) * BD;
        T* m_t = m + (t + 1) * n_banks_ * BD;
        T* out_t = output + t * BD;
        T* v_t = training_ ? (v + t * BD) : nullptr;

        // RNN: tmp_Rh = W_h @ h_prev
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
        CoreTanhKernel<T><<<num_blocks_bd, block_size, 0, stream_>>>(
            batch_size_, dim_, Wx_t, tmp_Rh, b, h_t, v_t);

        // EMA update: m_i_t = alpha_i * m_i_prev + (1 - alpha_i) * h_t
        EMAUpdateKernel<T><<<num_blocks_mem, block_size, 0, stream_>>>(
            batch_size_, dim_, n_banks_, a, h_t, m_prev, m_t);

        // Gated output: out = h * silu(z) + sum_i(m_i * silu(z_i))
        MultiScaleGateKernel<T><<<num_blocks_bd, block_size, 0, stream_>>>(
            batch_size_, dim_, n_banks_, h_t, z_t, m_t, out_t);
    }
}

// =============================================================================
// Multi-Scale EMA Elman Backward
// =============================================================================

template<typename T>
MultiScaleElmanBackward<T>::MultiScaleElmanBackward(
    int batch_size,
    int dim,
    int n_banks,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      n_banks_(n_banks),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void MultiScaleElmanBackward<T>::Run(
    int steps,
    const T* W_x,
    const T* W_h,
    const T* a,
    const T* x,
    const T* z,
    const T* h,
    const T* m,
    const T* v,
    const T* d_output,
    T* dx,
    T* dz,
    T* dW_x,
    T* dW_h,
    T* db,
    T* da,
    T* workspace) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks_bd = (BD + block_size - 1) / block_size;
    const int num_blocks_mem = (n_banks_ * BD + block_size - 1) / block_size;

    // Workspace layout:
    // [dv_all: T*B*dim] [dh: B*dim] [dh_recurrent: B*dim]
    // [dm: n_banks*B*dim] [dm_recurrent: n_banks*B*dim]
    // [dh_ema_float: B*dim (as floats)] [db_float: dim] [da_float: n_banks*dim]
    T* dv_all = workspace;
    T* dh = workspace + steps * BD;
    T* dh_recurrent = workspace + (steps + 1) * BD;
    T* dm = workspace + (steps + 2) * BD;
    T* dm_recurrent = workspace + (steps + 2) * BD + n_banks_ * BD;
    // Float workspace needs proper alignment - compute in terms of T elements needed
    const int64_t float_offset = (steps + 2) * BD + 2 * n_banks_ * BD;
    float* dh_ema_float = reinterpret_cast<float*>(workspace + float_offset);
    float* db_float = dh_ema_float + BD;
    float* da_float = db_float + dim_;

    // Initialize
    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(T), stream_);
    cudaMemsetAsync(dm_recurrent, 0, n_banks_ * BD * sizeof(T), stream_);
    cudaMemsetAsync(dh_ema_float, 0, BD * sizeof(float), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(da_float, 0, n_banks_ * dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW_x, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dW_h, 0, dim_ * dim_ * sizeof(T), stream_);

    // BPTT loop
    for (int t = steps - 1; t >= 0; --t) {
        const T* h_t = h + (t + 1) * BD;
        const T* m_t = m + (t + 1) * n_banks_ * BD;
        const T* m_prev = m + t * n_banks_ * BD;
        const T* z_t = z + t * batch_size_ * (1 + n_banks_) * dim_;
        const T* v_t = v + t * BD;
        const T* d_out_t = d_output + t * BD;

        T* dv_t = dv_all + t * BD;
        T* dz_t = dz + t * batch_size_ * (1 + n_banks_) * dim_;

        // Backward through multi-scale gate
        MultiScaleGateBackwardKernel<T><<<num_blocks_bd, block_size, 0, stream_>>>(
            batch_size_, dim_, n_banks_, h_t, z_t, m_t, d_out_t, dh, dz_t, dm);

        // Add recurrent gradients
        AddKernel<T><<<num_blocks_bd, block_size, 0, stream_>>>(BD, dh, dh_recurrent);
        AddKernel<T><<<num_blocks_mem, block_size, 0, stream_>>>(n_banks_ * BD, dm, dm_recurrent);

        // Zero float buffer for EMA dh accumulation
        cudaMemsetAsync(dh_ema_float, 0, BD * sizeof(float), stream_);

        // Backward through EMA (accumulates into dh_ema_float)
        EMABackwardKernel<T><<<num_blocks_mem, block_size, 0, stream_>>>(
            batch_size_, dim_, n_banks_, a, h_t, m_prev, dm,
            dh_ema_float, dm_recurrent, da_float);

        // Add EMA contribution to dh (convert float to T and add)
        FloatToTypeAddKernel<T><<<num_blocks_bd, block_size, 0, stream_>>>(BD, dh_ema_float, dh);

        // Backward through tanh
        CoreTanhBackwardKernel<T><<<num_blocks_bd, block_size, 0, stream_>>>(
            batch_size_, dim_, v_t, dh, dv_t, db_float);

        // dh_recurrent = W_h @ dv for next iteration
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
    }

    // Batch GEMMs
    // dx = W_x @ dv_all
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha,
        W_x, dim_,
        dv_all, dim_,
        &beta_zero,
        dx, dim_);

    // dW_x = x^T @ dv_all
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha,
        x, dim_,
        dv_all, dim_,
        &beta_one,
        dW_x, dim_);

    // dW_h = h^T @ dv_all (using h[0:T], not h[1:T+1])
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
    CopyFloatToT<T><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_float, db);
    CopyFloatToT<T><<<(n_banks_ * dim_ + 255) / 256, 256, 0, stream_>>>(n_banks_ * dim_, da_float, da);
}

// Explicit template instantiations
template struct MultiScaleElmanForward<__half>;
template struct MultiScaleElmanForward<__nv_bfloat16>;
template struct MultiScaleElmanForward<float>;
template struct MultiScaleElmanForward<double>;

template struct MultiScaleElmanBackward<__half>;
template struct MultiScaleElmanBackward<__nv_bfloat16>;
template struct MultiScaleElmanBackward<float>;
template struct MultiScaleElmanBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
