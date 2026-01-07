// Copyright 2026 Erik Garrison. Apache 2.0 License.
//
// Level 11: Selective Multi-Scale EMA Elman (Mamba-inspired selectivity)
//
// Architecture:
//   h_t = tanh(W_x @ x_t + W_h @ h_prev + b)  -- same as E1
//
//   # Selective decay
//   a_scale = x_t @ W_a                       -- [B, k] per-bank modulation
//   alpha_i = sigmoid(a_i + a_scale[:, i])    -- input-dependent decay
//   m_i_t = alpha_i * m_i_prev + (1 - alpha_i) * h_t
//
//   # Selective read
//   w = softmax(x_t @ W_w)                    -- [B, k] attention over banks
//   memory_out = sum(w_i * m_i)
//
//   # Gated output (just 2 gates)
//   out = h * silu(z_h) + memory_out * silu(z_m)

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cfloat>

#include "hasty/elman_ladder.h"
#include "blas.h"
#include "inline_ops.h"

namespace {

// Kernel: Fused tanh activation for RNN core (same as E1/E10)
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

// Kernel: Selective EMA update for all banks
// m_i = alpha_i * m_i_prev + (1 - alpha_i) * h
// where alpha_i = sigmoid(a_i + a_scale[:, i])
template<typename T>
__global__ void SelectiveEMAUpdateKernel(
    const int batch_size,
    const int dim,
    const int n_banks,
    const T* __restrict__ a,           // [n_banks, dim] base decay logits
    const T* __restrict__ a_scale,     // [B, n_banks] per-batch bank modulation
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
        const int scale_idx = b * n_banks + bank;

        // Selective decay: alpha = sigmoid(base + modulation)
        float a_base = static_cast<float>(a[a_idx]);
        float a_mod = static_cast<float>(a_scale[scale_idx]);
        float alpha = 1.0f / (1.0f + expf(-(a_base + a_mod)));

        float h_val = static_cast<float>(h[h_idx]);
        float m_prev_val = static_cast<float>(m_prev[idx]);

        m_out[idx] = static_cast<T>(alpha * m_prev_val + (1.0f - alpha) * h_val);
    }
}

// Kernel: Compute softmax read weights from logits
// w = softmax(x @ W_w)
template<typename T>
__global__ void SoftmaxReadWeightsKernel(
    const int batch_size,
    const int n_banks,
    const T* __restrict__ logits,      // [B, n_banks] raw logits
    T* __restrict__ weights) {         // [B, n_banks] softmax weights

    const int b = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < batch_size) {
        // Find max for numerical stability
        float max_val = -FLT_MAX;
        for (int i = 0; i < n_banks; i++) {
            float v = static_cast<float>(logits[b * n_banks + i]);
            if (v > max_val) max_val = v;
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < n_banks; i++) {
            float v = static_cast<float>(logits[b * n_banks + i]);
            float e = expf(v - max_val);
            weights[b * n_banks + i] = static_cast<T>(e);
            sum += e;
        }

        // Normalize
        float inv_sum = 1.0f / (sum + 1e-8f);
        for (int i = 0; i < n_banks; i++) {
            float w = static_cast<float>(weights[b * n_banks + i]) * inv_sum;
            weights[b * n_banks + i] = static_cast<T>(w);
        }
    }
}

// Kernel: Weighted sum of memory banks and gated output
// memory_out = sum(w_i * m_i)
// out = h * silu(z_h) + memory_out * silu(z_m)
template<typename T>
__global__ void SelectiveGateKernel(
    const int batch_size,
    const int dim,
    const int n_banks,
    const T* __restrict__ h,           // [B, dim]
    const T* __restrict__ z,           // [B, 2*dim] gates for h and memory
    const T* __restrict__ m,           // [n_banks, B, dim]
    const T* __restrict__ read_weights,// [B, n_banks] softmax weights
    T* __restrict__ output) {          // [B, dim]

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;
        const int b = idx / dim;

        // Weighted sum of memory banks
        float memory_out = 0.0f;
        for (int i = 0; i < n_banks; i++) {
            float w = static_cast<float>(read_weights[b * n_banks + i]);
            float m_val = static_cast<float>(m[i * batch_size * dim + idx]);
            memory_out += w * m_val;
        }

        // Gate for h
        float z_h = static_cast<float>(z[b * 2 * dim + d]);
        float sigmoid_zh = 1.0f / (1.0f + expf(-z_h));
        float silu_zh = z_h * sigmoid_zh;
        float h_val = static_cast<float>(h[idx]);

        // Gate for memory
        float z_m = static_cast<float>(z[b * 2 * dim + dim + d]);
        float sigmoid_zm = 1.0f / (1.0f + expf(-z_m));
        float silu_zm = z_m * sigmoid_zm;

        output[idx] = static_cast<T>(h_val * silu_zh + memory_out * silu_zm);
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

// Backward through selective gate
template<typename T>
__global__ void SelectiveGateBackwardKernel(
    const int batch_size,
    const int dim,
    const int n_banks,
    const T* __restrict__ h,               // [B, dim]
    const T* __restrict__ z,               // [B, 2*dim]
    const T* __restrict__ m,               // [n_banks, B, dim]
    const T* __restrict__ read_weights,    // [B, n_banks]
    const T* __restrict__ d_output,        // [B, dim]
    T* __restrict__ dh,                    // [B, dim]
    T* __restrict__ dz,                    // [B, 2*dim]
    T* __restrict__ dm,                    // [n_banks, B, dim]
    float* __restrict__ d_read_weights) {  // [B, n_banks] float for accumulation

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;
        const int b = idx / dim;

        float dout = static_cast<float>(d_output[idx]);

        // Recompute memory_out for gradient
        float memory_out = 0.0f;
        for (int i = 0; i < n_banks; i++) {
            float w = static_cast<float>(read_weights[b * n_banks + i]);
            float m_val = static_cast<float>(m[i * batch_size * dim + idx]);
            memory_out += w * m_val;
        }

        // Gradient for h branch
        float z_h = static_cast<float>(z[b * 2 * dim + d]);
        float sigmoid_zh = 1.0f / (1.0f + expf(-z_h));
        float silu_zh = z_h * sigmoid_zh;
        float dsilu_zh = sigmoid_zh * (1.0f + z_h * (1.0f - sigmoid_zh));
        float h_val = static_cast<float>(h[idx]);

        dh[idx] = static_cast<T>(dout * silu_zh);
        dz[b * 2 * dim + d] = static_cast<T>(dout * h_val * dsilu_zh);

        // Gradient for memory branch
        float z_m = static_cast<float>(z[b * 2 * dim + dim + d]);
        float sigmoid_zm = 1.0f / (1.0f + expf(-z_m));
        float silu_zm = z_m * sigmoid_zm;
        float dsilu_zm = sigmoid_zm * (1.0f + z_m * (1.0f - sigmoid_zm));

        dz[b * 2 * dim + dim + d] = static_cast<T>(dout * memory_out * dsilu_zm);

        // Gradient for memory banks and read weights
        float d_mem_out = dout * silu_zm;
        for (int i = 0; i < n_banks; i++) {
            float w = static_cast<float>(read_weights[b * n_banks + i]);
            float m_val = static_cast<float>(m[i * batch_size * dim + idx]);

            // dm_i = d_mem_out * w_i
            dm[i * batch_size * dim + idx] = static_cast<T>(d_mem_out * w);

            // d_read_weights_i = d_mem_out * m_i (accumulated across dims)
            atomicAdd(&d_read_weights[b * n_banks + i], d_mem_out * m_val);
        }
    }
}

// Backward through softmax read weights
// d_logits = weights * (d_weights - sum(weights * d_weights))
template<typename T>
__global__ void SoftmaxBackwardKernel(
    const int batch_size,
    const int n_banks,
    const T* __restrict__ weights,         // [B, n_banks]
    const float* __restrict__ d_weights,   // [B, n_banks] from previous kernel
    T* __restrict__ d_logits) {            // [B, n_banks]

    const int b = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < batch_size) {
        // Compute weighted sum: sum(w_i * dw_i)
        float weighted_sum = 0.0f;
        for (int i = 0; i < n_banks; i++) {
            float w = static_cast<float>(weights[b * n_banks + i]);
            float dw = d_weights[b * n_banks + i];
            weighted_sum += w * dw;
        }

        // d_logits_i = w_i * (dw_i - weighted_sum)
        for (int i = 0; i < n_banks; i++) {
            float w = static_cast<float>(weights[b * n_banks + i]);
            float dw = d_weights[b * n_banks + i];
            d_logits[b * n_banks + i] = static_cast<T>(w * (dw - weighted_sum));
        }
    }
}

// Backward through selective EMA
template<typename T>
__global__ void SelectiveEMABackwardKernel(
    const int batch_size,
    const int dim,
    const int n_banks,
    const T* __restrict__ a,               // [n_banks, dim] base decay logits
    const T* __restrict__ a_scale,         // [B, n_banks] modulation
    const T* __restrict__ h,               // [B, dim]
    const T* __restrict__ m_prev,          // [n_banks, B, dim]
    const T* __restrict__ dm,              // [n_banks, B, dim] gradient to m
    float* __restrict__ dh_accum,          // [B, dim] float buffer
    T* __restrict__ dm_prev_out,           // [n_banks, B, dim] gradient to m_prev
    float* __restrict__ da,                // [n_banks, dim] gradient to base decay
    float* __restrict__ da_scale) {        // [B, n_banks] gradient to modulation

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_banks * batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;
        const int b = (idx / dim) % batch_size;
        const int bank = idx / (batch_size * dim);

        const int h_idx = b * dim + d;
        const int a_idx = bank * dim + d;
        const int scale_idx = b * n_banks + bank;

        float a_base = static_cast<float>(a[a_idx]);
        float a_mod = static_cast<float>(a_scale[scale_idx]);
        float alpha = 1.0f / (1.0f + expf(-(a_base + a_mod)));
        float dalpha = alpha * (1.0f - alpha);  // sigmoid derivative

        float h_val = static_cast<float>(h[h_idx]);
        float m_prev_val = static_cast<float>(m_prev[idx]);
        float dm_val = static_cast<float>(dm[idx]);

        // d/d(m_prev) = alpha * dm
        dm_prev_out[idx] = static_cast<T>(alpha * dm_val);

        // d/d(h) = (1 - alpha) * dm (accumulated across banks)
        atomicAdd(&dh_accum[h_idx], (1.0f - alpha) * dm_val);

        // d/d(alpha) = (m_prev - h) * dm
        // d/d(a_base) = d/d(alpha) * dalpha
        // d/d(a_scale) = d/d(alpha) * dalpha (same chain rule)
        float da_val = (m_prev_val - h_val) * dalpha * dm_val;
        atomicAdd(&da[a_idx], da_val);
        atomicAdd(&da_scale[scale_idx], da_val);
    }
}

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
// Selective EMA Elman Forward
// =============================================================================

template<typename T>
SelectiveElmanForward<T>::SelectiveElmanForward(
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
void SelectiveElmanForward<T>::Run(
    int steps,
    const T* W_x,        // [dim, dim]
    const T* W_h,        // [dim, dim]
    const T* b,          // [dim]
    const T* a,          // [n_banks, dim] base EMA decay logits
    const T* W_a,        // [dim, n_banks] decay modulation projection
    const T* W_w,        // [dim, n_banks] read weights projection
    const T* x,          // [T, B, dim]
    const T* z,          // [T, B, 2*dim] gates (h and memory)
    T* h,                // [T+1, B, dim] hidden states
    T* m,                // [T+1, n_banks, B, dim] memory banks
    T* output,           // [T, B, dim]
    T* v,                // [T, B, dim] pre-activation cache
    T* a_scale_cache,    // [T, B, n_banks] decay modulation cache
    T* read_weights_cache,// [T, B, n_banks] read weights cache
    T* workspace) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int BD = batch_size_ * dim_;
    const int BK = batch_size_ * n_banks_;
    const int block_size = 256;
    const int num_blocks_bd = (BD + block_size - 1) / block_size;
    const int num_blocks_mem = (n_banks_ * BD + block_size - 1) / block_size;
    const int num_blocks_batch = (batch_size_ + block_size - 1) / block_size;

    // Workspace: [tmp_Wx: T*B*dim] [tmp_Rh: B*dim] [a_scale: B*k] [read_logits: B*k]
    T* tmp_Wx = workspace;
    T* tmp_Rh = workspace + steps * BD;
    T* a_scale = workspace + steps * BD + BD;
    T* read_logits = workspace + steps * BD + BD + BK;

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
        const T* x_t = x + t * BD;
        const T* h_prev = h + t * BD;
        const T* z_t = z + t * batch_size_ * 2 * dim_;
        const T* m_prev = m + t * n_banks_ * BD;

        T* h_t = h + (t + 1) * BD;
        T* m_t = m + (t + 1) * n_banks_ * BD;
        T* out_t = output + t * BD;
        T* v_t = training_ ? (v + t * BD) : nullptr;
        T* a_scale_t = training_ ? (a_scale_cache + t * BK) : a_scale;
        T* read_weights_t = training_ ? (read_weights_cache + t * BK) : read_logits;

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

        // Compute decay modulation: a_scale = x @ W_a
        // x is [B, dim] row-major, W_a is [dim, n_banks] row-major
        // In col-major: x_col is [dim, B], W_a_col is [n_banks, dim]
        // Result [B, n_banks] row-major = [n_banks, B] col-major = W_a_col @ x_col
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n_banks_, batch_size_, dim_,
            &alpha,
            W_a, n_banks_,
            x_t, dim_,
            &beta_zero,
            a_scale_t, n_banks_);

        // Selective EMA update
        SelectiveEMAUpdateKernel<T><<<num_blocks_mem, block_size, 0, stream_>>>(
            batch_size_, dim_, n_banks_, a, a_scale_t, h_t, m_prev, m_t);

        // Compute read logits and softmax weights: read_logits = x @ W_w
        // Same layout as a_scale computation
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n_banks_, batch_size_, dim_,
            &alpha,
            W_w, n_banks_,
            x_t, dim_,
            &beta_zero,
            read_logits, n_banks_);

        SoftmaxReadWeightsKernel<T><<<num_blocks_batch, block_size, 0, stream_>>>(
            batch_size_, n_banks_, read_logits, read_weights_t);

        // Gated output with selective memory read
        SelectiveGateKernel<T><<<num_blocks_bd, block_size, 0, stream_>>>(
            batch_size_, dim_, n_banks_, h_t, z_t, m_t, read_weights_t, out_t);
    }
}

// =============================================================================
// Selective EMA Elman Backward
// =============================================================================

template<typename T>
SelectiveElmanBackward<T>::SelectiveElmanBackward(
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
void SelectiveElmanBackward<T>::Run(
    int steps,
    const T* W_x,
    const T* W_h,
    const T* a,
    const T* W_a,
    const T* W_w,
    const T* x,
    const T* z,
    const T* h,
    const T* m,
    const T* v,
    const T* a_scale_cache,
    const T* read_weights_cache,
    const T* d_output,
    T* dx,
    T* dz,
    T* dW_x,
    T* dW_h,
    T* db,
    T* da,
    T* dW_a,
    T* dW_w,
    T* workspace) {

    static const T alpha_one = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int BD = batch_size_ * dim_;
    const int BK = batch_size_ * n_banks_;
    const int block_size = 256;
    const int num_blocks_bd = (BD + block_size - 1) / block_size;
    const int num_blocks_mem = (n_banks_ * BD + block_size - 1) / block_size;
    const int num_blocks_batch = (batch_size_ + block_size - 1) / block_size;
    const int num_blocks_bk = (BK + block_size - 1) / block_size;

    // Workspace layout:
    // [dv_all: T*BD] [dh: BD] [dh_recurrent: BD] [dm: n_banks*BD] [dm_recurrent: n_banks*BD]
    // [d_read_logits: BK] [da_scale_T: BK] [dx_proj: BD] [dh_ema_float: BD] [d_read_weights_float: BK]
    // [db_float: dim] [da_float: n_banks*dim] [da_scale_float: BK]
    T* dv_all = workspace;
    T* dh = workspace + steps * BD;
    T* dh_recurrent = workspace + (steps + 1) * BD;
    T* dm = workspace + (steps + 2) * BD;
    T* dm_recurrent = workspace + (steps + 2) * BD + n_banks_ * BD;
    T* d_read_logits = workspace + (steps + 2) * BD + 2 * n_banks_ * BD;
    T* da_scale_T = workspace + (steps + 2) * BD + 2 * n_banks_ * BD + BK;
    T* dx_proj = workspace + (steps + 2) * BD + 2 * n_banks_ * BD + 2 * BK;

    const int64_t float_offset = (steps + 2) * BD + 2 * n_banks_ * BD + 2 * BK + BD;
    float* dh_ema_float = reinterpret_cast<float*>(workspace + float_offset);
    float* d_read_weights_float = dh_ema_float + BD;
    float* db_float = d_read_weights_float + BK;
    float* da_float = db_float + dim_;
    float* da_scale_float = da_float + n_banks_ * dim_;

    // Initialize
    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(T), stream_);
    cudaMemsetAsync(dm_recurrent, 0, n_banks_ * BD * sizeof(T), stream_);
    cudaMemsetAsync(dh_ema_float, 0, BD * sizeof(float), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(da_float, 0, n_banks_ * dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dx, 0, steps * BD * sizeof(T), stream_);  // dx gets accumulated in loop
    cudaMemsetAsync(dW_x, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dW_h, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dW_a, 0, dim_ * n_banks_ * sizeof(T), stream_);
    cudaMemsetAsync(dW_w, 0, dim_ * n_banks_ * sizeof(T), stream_);

    // BPTT loop
    for (int t = steps - 1; t >= 0; --t) {
        const T* h_t = h + (t + 1) * BD;
        const T* x_t = x + t * BD;
        const T* m_t = m + (t + 1) * n_banks_ * BD;
        const T* m_prev = m + t * n_banks_ * BD;
        const T* z_t = z + t * batch_size_ * 2 * dim_;
        const T* v_t = v + t * BD;
        const T* a_scale_t = a_scale_cache + t * BK;
        const T* read_weights_t = read_weights_cache + t * BK;
        const T* d_out_t = d_output + t * BD;

        T* dv_t = dv_all + t * BD;
        T* dz_t = dz + t * batch_size_ * 2 * dim_;

        // Zero float accumulators for this timestep
        cudaMemsetAsync(d_read_weights_float, 0, BK * sizeof(float), stream_);
        cudaMemsetAsync(da_scale_float, 0, BK * sizeof(float), stream_);

        // Backward through selective gate
        SelectiveGateBackwardKernel<T><<<num_blocks_bd, block_size, 0, stream_>>>(
            batch_size_, dim_, n_banks_, h_t, z_t, m_t, read_weights_t, d_out_t,
            dh, dz_t, dm, d_read_weights_float);

        // Add recurrent gradients
        AddKernel<T><<<num_blocks_bd, block_size, 0, stream_>>>(BD, dh, dh_recurrent);
        AddKernel<T><<<num_blocks_mem, block_size, 0, stream_>>>(n_banks_ * BD, dm, dm_recurrent);

        // Backward through softmax
        SoftmaxBackwardKernel<T><<<num_blocks_batch, block_size, 0, stream_>>>(
            batch_size_, n_banks_, read_weights_t, d_read_weights_float, d_read_logits);

        // dW_w += x^T @ d_read_logits
        // dW_w is row-major [dim, k] = col-major [k, dim]
        // Want: dW_w[d, i] = sum_b(x[b, d] * d_logits[b, i])
        // In cuBLAS col-major: dW_w_col[i, d] = sum_b(x_col[d, b] * d_logits_col[i, b])
        // This is C = d_logits_col @ x_col^T = [k, B] @ [B, dim] = [k, dim]
        // With CUBLAS_OP_N on d_logits (already [k, B] col-major) and CUBLAS_OP_T on x
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_T,
            n_banks_, dim_, batch_size_,
            &alpha_one,
            d_read_logits, n_banks_,
            x_t, dim_,
            &beta_one,
            dW_w, n_banks_);

        // dx += d_read_logits @ W_w^T  (contribution from read weights projection)
        // d_logits is [B, n_banks], W_w^T is [n_banks, dim], dx is [B, dim]
        // In col-major: W_w_col^T @ d_logits_col = [dim, n_banks] @ [n_banks, B] = [dim, B]
        T* dx_t = dx + t * BD;
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, n_banks_,
            &alpha_one,
            W_w, n_banks_,
            d_read_logits, n_banks_,
            &beta_one,
            dx_t, dim_);

        // Zero float buffer for EMA dh accumulation
        cudaMemsetAsync(dh_ema_float, 0, BD * sizeof(float), stream_);

        // Backward through selective EMA
        SelectiveEMABackwardKernel<T><<<num_blocks_mem, block_size, 0, stream_>>>(
            batch_size_, dim_, n_banks_, a, a_scale_t, h_t, m_prev, dm,
            dh_ema_float, dm_recurrent, da_float, da_scale_float);

        // Add EMA contribution to dh
        FloatToTypeAddKernel<T><<<num_blocks_bd, block_size, 0, stream_>>>(BD, dh_ema_float, dh);

        // Convert da_scale_float to T for GEMM
        CopyFloatToT<T><<<num_blocks_bk, block_size, 0, stream_>>>(BK, da_scale_float, da_scale_T);

        // dW_a += x^T @ da_scale
        // Same pattern as dW_w
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_T,
            n_banks_, dim_, batch_size_,
            &alpha_one,
            da_scale_T, n_banks_,
            x_t, dim_,
            &beta_one,
            dW_a, n_banks_);

        // dx += da_scale @ W_a^T  (contribution from decay modulation projection)
        // Same layout as dx from W_w computation
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, n_banks_,
            &alpha_one,
            W_a, n_banks_,
            da_scale_T, n_banks_,
            &beta_one,
            dx_t, dim_);

        // Backward through tanh
        CoreTanhBackwardKernel<T><<<num_blocks_bd, block_size, 0, stream_>>>(
            batch_size_, dim_, v_t, dh, dv_t, db_float);

        // dh_recurrent = W_h @ dv for next iteration
        if (t > 0) {
            blas<T>::gemm(
                blas_handle_,
                CUBLAS_OP_N, CUBLAS_OP_N,
                dim_, batch_size_, dim_,
                &alpha_one,
                W_h, dim_,
                dv_t, dim_,
                &beta_zero,
                dh_recurrent, dim_);
        }
    }

    // Batch GEMMs
    // dx += W_x @ dv_all (add to existing dx from W_a/W_w contributions)
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_x, dim_,
        dv_all, dim_,
        &beta_one,
        dx, dim_);

    // dW_x = x^T @ dv_all
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        dv_all, dim_,
        &beta_one,
        dW_x, dim_);

    // dW_h = h^T @ dv_all (using h[0:T])
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        h, dim_,
        dv_all, dim_,
        &beta_one,
        dW_h, dim_);

    // Copy float gradients to output
    CopyFloatToT<T><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_float, db);
    CopyFloatToT<T><<<(n_banks_ * dim_ + 255) / 256, 256, 0, stream_>>>(n_banks_ * dim_, da_float, da);
}

// Explicit template instantiations
template struct SelectiveElmanForward<__half>;
template struct SelectiveElmanForward<__nv_bfloat16>;
template struct SelectiveElmanForward<float>;
template struct SelectiveElmanForward<double>;

template struct SelectiveElmanBackward<__half>;
template struct SelectiveElmanBackward<__nv_bfloat16>;
template struct SelectiveElmanBackward<float>;
template struct SelectiveElmanBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
