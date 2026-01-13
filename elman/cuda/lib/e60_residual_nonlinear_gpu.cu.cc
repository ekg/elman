// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E60: Residual Nonlinear Elman
//
// Core Innovation: Residual RNN with nonlinear h-dependence
//
// Architecture:
// h_t = h_{t-1} + alpha * tanh(W_h @ h_{t-1} + W_x @ x_t + b)
// output_t = h_t * silu(h_t)
//
// Where alpha = exp(log_alpha) is a learned positive scalar.
//
// Jacobian: dh_t/dh_{t-1} = I + alpha * diag(1 - tanh²(...)) @ W_h
//
// Key insight: Identity path ALWAYS exists for gradient flow.
// Nonlinear path adds expressivity without killing gradients.

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
// Native BF16 operations (fast path for SM80+)
// =============================================================================

__device__ __forceinline__ __nv_bfloat16 bf16_add(__nv_bfloat16 a, __nv_bfloat16 b) {
#if __CUDA_ARCH__ >= 800
    return __hadd(a, b);
#else
    return __float2bfloat16(__bfloat162float(a) + __bfloat162float(b));
#endif
}

__device__ __forceinline__ __nv_bfloat16 bf16_mul(__nv_bfloat16 a, __nv_bfloat16 b) {
#if __CUDA_ARCH__ >= 800
    return __hmul(a, b);
#else
    return __float2bfloat16(__bfloat162float(a) * __bfloat162float(b));
#endif
}

// =============================================================================
// Forward Kernels
// =============================================================================

// BF16-optimized: Fused residual update with tanh + self-gating output
// h_new = h_prev + alpha * tanh(Wx + Rh + b)
// output = h_new * silu(h_new)
__global__ void E60FusedResidualGateKernel_BF16(
    const int batch_size,
    const int dim,
    const float alpha,
    const __nv_bfloat16* __restrict__ h_prev,
    const __nv_bfloat16* __restrict__ Wx,
    const __nv_bfloat16* __restrict__ Rh,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ h_out,
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ tanh_cache) {  // Store tanh values for backward

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // Compute pre-activation: Wx + Rh + b
        __nv_bfloat16 sum = bf16_add(bf16_add(Wx[idx], Rh[idx]), b[d]);

        // Compute tanh (need f32 for tanh)
        float val = __bfloat162float(sum);
        float tanh_val = tanhf(val);

        // Store tanh for backward (need 1 - tanh^2 derivative)
        if (tanh_cache) tanh_cache[idx] = __float2bfloat16(tanh_val);

        // Residual update: h_new = h_prev + alpha * tanh(...)
        float h_prev_f = __bfloat162float(h_prev[idx]);
        float h_new_f = h_prev_f + alpha * tanh_val;
        h_out[idx] = __float2bfloat16(h_new_f);

        // Self-gating output: h_new * silu(h_new)
        float sigmoid_h = 1.0f / (1.0f + __expf(-h_new_f));
        float silu_h = h_new_f * sigmoid_h;
        output[idx] = __float2bfloat16(h_new_f * silu_h);
    }
}

// Generic version for other types
template<typename T>
__global__ void E60FusedResidualGateKernel(
    const int batch_size,
    const int dim,
    const float alpha,
    const T* __restrict__ h_prev,
    const T* __restrict__ Wx,
    const T* __restrict__ Rh,
    const T* __restrict__ b,
    T* __restrict__ h_out,
    T* __restrict__ output,
    T* __restrict__ tanh_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float val = static_cast<float>(Wx[idx]) + static_cast<float>(Rh[idx]) + static_cast<float>(b[d]);
        float tanh_val = tanhf(val);

        if (tanh_cache) tanh_cache[idx] = static_cast<T>(tanh_val);

        float h_prev_f = static_cast<float>(h_prev[idx]);
        float h_new_f = h_prev_f + alpha * tanh_val;
        h_out[idx] = static_cast<T>(h_new_f);

        float sigmoid_h = 1.0f / (1.0f + expf(-h_new_f));
        float silu_h = h_new_f * sigmoid_h;
        output[idx] = static_cast<T>(h_new_f * silu_h);
    }
}

// =============================================================================
// Backward Kernels
// =============================================================================

// BF16-optimized backward through self-gate and residual update
// output = h * silu(h) = h² * sigmoid(h)
// d(output)/dh = silu(h) * (2 + h*(1-sigmoid(h)))
//
// h = h_prev + alpha * tanh(pre_act)
// dh_prev = dh * 1 (identity path)
// d_tanh = dh * alpha
// d_pre_act = d_tanh * (1 - tanh²)
// d_log_alpha = dh * tanh * alpha (since d(exp(log_alpha))/d(log_alpha) = alpha)
__global__ void E60BackwardKernel_BF16(
    const int batch_size,
    const int dim,
    const float alpha,
    const __nv_bfloat16* __restrict__ h_prev,
    const __nv_bfloat16* __restrict__ h_new,
    const __nv_bfloat16* __restrict__ tanh_cache,
    const __nv_bfloat16* __restrict__ d_output,
    const __nv_bfloat16* __restrict__ dh_recurrent,
    __nv_bfloat16* __restrict__ dh_prev_out,
    __nv_bfloat16* __restrict__ dv,
    float* __restrict__ db,
    float* __restrict__ d_log_alpha) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float h_new_f = __bfloat162float(h_new[idx]);
        float tanh_val = __bfloat162float(tanh_cache[idx]);
        float dout = __bfloat162float(d_output[idx]);

        // Backward through self-gate: output = h * silu(h)
        float sigmoid_h = 1.0f / (1.0f + __expf(-h_new_f));
        float silu_h = h_new_f * sigmoid_h;
        float d_silu_factor = silu_h * (2.0f + h_new_f * (1.0f - sigmoid_h));
        float dh_from_output = dout * d_silu_factor;

        // Add recurrent gradient
        float dh_total;
        if (dh_recurrent) {
            dh_total = dh_from_output + __bfloat162float(dh_recurrent[idx]);
        } else {
            dh_total = dh_from_output;
        }

        // Backward through residual: h = h_prev + alpha * tanh
        // dh_prev gets dh_total * 1 (identity path)
        // d_tanh gets dh_total * alpha
        float dtanh_val = dh_total * alpha;

        // Backward through tanh: d_pre_act = dtanh * (1 - tanh²)
        float dtanh_deriv = 1.0f - tanh_val * tanh_val;
        float dv_val = dtanh_val * dtanh_deriv;

        dv[idx] = __float2bfloat16(dv_val);
        dh_prev_out[idx] = __float2bfloat16(dh_total);  // identity path gradient

        atomicAdd(&db[d], dv_val);
        // d_log_alpha: d(alpha * tanh)/d(log_alpha) = alpha * tanh (chain rule with exp)
        atomicAdd(d_log_alpha, dh_total * tanh_val * alpha);
    }
}

// Generic version
template<typename T>
__global__ void E60BackwardKernel(
    const int batch_size,
    const int dim,
    const float alpha,
    const T* __restrict__ h_prev,
    const T* __restrict__ h_new,
    const T* __restrict__ tanh_cache,
    const T* __restrict__ d_output,
    const T* __restrict__ dh_recurrent,
    T* __restrict__ dh_prev_out,
    T* __restrict__ dv,
    float* __restrict__ db,
    float* __restrict__ d_log_alpha) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float h_new_f = static_cast<float>(h_new[idx]);
        float tanh_val = static_cast<float>(tanh_cache[idx]);
        float dout = static_cast<float>(d_output[idx]);

        // Backward through self-gate
        float sigmoid_h = 1.0f / (1.0f + expf(-h_new_f));
        float silu_h = h_new_f * sigmoid_h;
        float d_silu_factor = silu_h * (2.0f + h_new_f * (1.0f - sigmoid_h));
        float dh_from_output = dout * d_silu_factor;

        // Add recurrent gradient
        float dh_total;
        if (dh_recurrent) {
            dh_total = dh_from_output + static_cast<float>(dh_recurrent[idx]);
        } else {
            dh_total = dh_from_output;
        }

        // Backward through residual
        float dtanh_val = dh_total * alpha;
        float dtanh_deriv = 1.0f - tanh_val * tanh_val;
        float dv_val = dtanh_val * dtanh_deriv;

        dv[idx] = static_cast<T>(dv_val);
        dh_prev_out[idx] = static_cast<T>(dh_total);

        atomicAdd(&db[d], dv_val);
        atomicAdd(d_log_alpha, dh_total * tanh_val * alpha);
    }
}

// =============================================================================
// Utility Kernels
// =============================================================================

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
// E60 Residual Nonlinear Forward - BF16 Specialization
// =============================================================================

template<>
E60ResidualNonlinearForward<__nv_bfloat16>::E60ResidualNonlinearForward(
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
void E60ResidualNonlinearForward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_x,
    const __nv_bfloat16* W_h,
    const __nv_bfloat16* b,
    const float* log_alpha,
    const __nv_bfloat16* x,
    __nv_bfloat16* h,
    __nv_bfloat16* output,
    __nv_bfloat16* tanh_cache,
    __nv_bfloat16* workspace) {

    static const __nv_bfloat16 alpha_one = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);

    // Get alpha from log_alpha (on device)
    float log_alpha_host;
    cudaMemcpyAsync(&log_alpha_host, log_alpha, sizeof(float), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    float alpha = expf(log_alpha_host);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    __nv_bfloat16* tmp_Wx = workspace;
    __nv_bfloat16* tmp_Rh = workspace + steps * BD;

    // Pre-compute W_x @ x for all timesteps
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_x, dim_,
        x, dim_,
        &beta_zero,
        tmp_Wx, dim_);

    // Process each timestep
    for (int t = 0; t < steps; ++t) {
        const __nv_bfloat16* Wx_t = tmp_Wx + t * BD;
        const __nv_bfloat16* h_prev = h + t * BD;
        __nv_bfloat16* h_t = h + (t + 1) * BD;
        __nv_bfloat16* out_t = output + t * BD;
        __nv_bfloat16* tanh_t = training_ ? (tanh_cache + t * BD) : nullptr;

        // tmp_Rh = h_prev @ W_h.T
        blas<__nv_bfloat16>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha_one,
            W_h, dim_,
            h_prev, dim_,
            &beta_zero,
            tmp_Rh, dim_);

        // Fused: h_t = h_prev + alpha * tanh(Wx + Rh + b), output = h_t * silu(h_t)
        E60FusedResidualGateKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, alpha, h_prev, Wx_t, tmp_Rh, b, h_t, out_t, tanh_t);
    }
}

// =============================================================================
// E60 Residual Nonlinear Backward - BF16 Specialization
// =============================================================================

template<>
E60ResidualNonlinearBackward<__nv_bfloat16>::E60ResidualNonlinearBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void E60ResidualNonlinearBackward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_x,
    const __nv_bfloat16* W_h,
    const float* log_alpha,
    const __nv_bfloat16* x,
    const __nv_bfloat16* h,
    const __nv_bfloat16* tanh_cache,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* dx,
    __nv_bfloat16* dW_x,
    __nv_bfloat16* dW_h,
    __nv_bfloat16* db,
    float* d_log_alpha,
    __nv_bfloat16* workspace) {

    static const __nv_bfloat16 alpha_one = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);
    static const __nv_bfloat16 beta_one = __float2bfloat16(1.0f);

    // Get alpha from log_alpha (on device)
    float log_alpha_host;
    cudaMemcpyAsync(&log_alpha_host, log_alpha, sizeof(float), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    float alpha = expf(log_alpha_host);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Workspace layout:
    // [T * BD] dv_all
    // [BD] dh_prev_out (for each timestep, reused)
    // [BD] dh_recurrent
    // [dim] db_float
    // [1] d_log_alpha_float
    __nv_bfloat16* dv_all = workspace;
    __nv_bfloat16* dh_prev_out = workspace + steps * BD;
    __nv_bfloat16* dh_recurrent = workspace + (steps + 1) * BD;
    float* db_float = reinterpret_cast<float*>(workspace + (steps + 2) * BD);
    float* d_log_alpha_device = db_float + dim_;

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(d_log_alpha_device, 0, sizeof(float), stream_);
    cudaMemsetAsync(dW_x, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_h, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);

    // BPTT loop
    // The residual h_t = h_{t-1} + alpha * tanh(W_h @ h_{t-1} + ...)
    // means dh_{t-1} = dh_t + dh_t * alpha * (1-tanh²) * W_h^T
    //                = dh_prev_out + W_h @ dv
    for (int t = steps - 1; t >= 0; --t) {
        const __nv_bfloat16* h_prev = h + t * BD;
        const __nv_bfloat16* h_new = h + (t + 1) * BD;
        const __nv_bfloat16* tanh_t = tanh_cache + t * BD;
        const __nv_bfloat16* d_out_t = d_output + t * BD;
        __nv_bfloat16* dv_t = dv_all + t * BD;

        // Backward through self-gate and residual
        // dh_prev_out contains the full gradient for h_prev (identity + through tanh paths)
        E60BackwardKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, alpha, h_prev, h_new, tanh_t, d_out_t,
            (t < steps - 1) ? dh_recurrent : nullptr,
            dh_prev_out, dv_t, db_float, d_log_alpha_device);

        // dh_recurrent = dh_prev_out + W_h @ dv
        // = identity path gradient + W_h path gradient
        if (t > 0) {
            // Start with identity path: dh_prev_out
            cudaMemcpyAsync(dh_recurrent, dh_prev_out, BD * sizeof(__nv_bfloat16),
                           cudaMemcpyDeviceToDevice, stream_);
            // Add W_h path: dh_recurrent += W_h @ dv
            blas<__nv_bfloat16>::gemm(
                blas_handle_,
                CUBLAS_OP_N, CUBLAS_OP_N,
                dim_, batch_size_, dim_,
                &alpha_one,
                W_h, dim_,
                dv_t, dim_,
                &beta_one,  // IMPORTANT: add to existing dh_recurrent
                dh_recurrent, dim_);
        }
    }

    // Batch GEMMs for weight gradients
    // dx = W_x @ dv_all
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_x, dim_,
        dv_all, dim_,
        &beta_zero,
        dx, dim_);

    // dW_x = x.T @ dv_all
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        dv_all, dim_,
        &beta_one,
        dW_x, dim_);

    // dW_h = h[:-1].T @ dv_all
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        h, dim_,  // h[0:T] (excludes final state)
        dv_all, dim_,
        &beta_one,
        dW_h, dim_);

    // Copy float gradients to output types
    CopyFloatToT<__nv_bfloat16><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_float, db);

    // Copy d_log_alpha_device to d_log_alpha
    cudaMemcpyAsync(d_log_alpha, d_log_alpha_device, sizeof(float), cudaMemcpyDeviceToDevice, stream_);
}

// =============================================================================
// Generic Template Implementations
// =============================================================================

template<typename T>
E60ResidualNonlinearForward<T>::E60ResidualNonlinearForward(
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
void E60ResidualNonlinearForward<T>::Run(
    int steps,
    const T* W_x,
    const T* W_h,
    const T* b,
    const float* log_alpha,
    const T* x,
    T* h,
    T* output,
    T* tanh_cache,
    T* workspace) {

    static const T alpha_one = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    // Get alpha from log_alpha (on device)
    float log_alpha_host;
    cudaMemcpyAsync(&log_alpha_host, log_alpha, sizeof(float), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    float alpha = expf(log_alpha_host);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    T* tmp_Wx = workspace;
    T* tmp_Rh = workspace + steps * BD;

    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_x, dim_,
        x, dim_,
        &beta_zero,
        tmp_Wx, dim_);

    for (int t = 0; t < steps; ++t) {
        const T* Wx_t = tmp_Wx + t * BD;
        const T* h_prev = h + t * BD;
        T* h_t = h + (t + 1) * BD;
        T* out_t = output + t * BD;
        T* tanh_t = training_ ? (tanh_cache + t * BD) : nullptr;

        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha_one,
            W_h, dim_,
            h_prev, dim_,
            &beta_zero,
            tmp_Rh, dim_);

        E60FusedResidualGateKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, alpha, h_prev, Wx_t, tmp_Rh, b, h_t, out_t, tanh_t);
    }
}

template<typename T>
E60ResidualNonlinearBackward<T>::E60ResidualNonlinearBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void E60ResidualNonlinearBackward<T>::Run(
    int steps,
    const T* W_x,
    const T* W_h,
    const float* log_alpha,
    const T* x,
    const T* h,
    const T* tanh_cache,
    const T* d_output,
    T* dx,
    T* dW_x,
    T* dW_h,
    T* db,
    float* d_log_alpha,
    T* workspace) {

    static const T alpha_one = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    // Get alpha from log_alpha (on device)
    float log_alpha_host;
    cudaMemcpyAsync(&log_alpha_host, log_alpha, sizeof(float), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    float alpha = expf(log_alpha_host);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    T* dv_all = workspace;
    T* dh_prev_out = workspace + steps * BD;
    T* dh_recurrent = workspace + (steps + 1) * BD;
    float* db_float = reinterpret_cast<float*>(workspace + (steps + 2) * BD);
    float* d_log_alpha_device = db_float + dim_;

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(T), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(d_log_alpha_device, 0, sizeof(float), stream_);
    cudaMemsetAsync(dW_x, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dW_h, 0, dim_ * dim_ * sizeof(T), stream_);

    // BPTT loop with proper residual gradient handling
    for (int t = steps - 1; t >= 0; --t) {
        const T* h_prev = h + t * BD;
        const T* h_new = h + (t + 1) * BD;
        const T* tanh_t = tanh_cache + t * BD;
        const T* d_out_t = d_output + t * BD;
        T* dv_t = dv_all + t * BD;

        E60BackwardKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, alpha, h_prev, h_new, tanh_t, d_out_t,
            (t < steps - 1) ? dh_recurrent : nullptr,
            dh_prev_out, dv_t, db_float, d_log_alpha_device);

        // dh_recurrent = dh_prev_out + W_h @ dv
        // = identity path gradient + W_h path gradient
        if (t > 0) {
            // Start with identity path: dh_prev_out
            cudaMemcpyAsync(dh_recurrent, dh_prev_out, BD * sizeof(T),
                           cudaMemcpyDeviceToDevice, stream_);
            // Add W_h path: dh_recurrent += W_h @ dv
            blas<T>::gemm(
                blas_handle_,
                CUBLAS_OP_N, CUBLAS_OP_N,
                dim_, batch_size_, dim_,
                &alpha_one,
                W_h, dim_,
                dv_t, dim_,
                &beta_one,  // IMPORTANT: add to existing dh_recurrent
                dh_recurrent, dim_);
        }
    }

    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_x, dim_,
        dv_all, dim_,
        &beta_zero,
        dx, dim_);

    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        dv_all, dim_,
        &beta_one,
        dW_x, dim_);

    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        h, dim_,
        dv_all, dim_,
        &beta_one,
        dW_h, dim_);

    CopyFloatToT<T><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_float, db);
    cudaMemcpyAsync(d_log_alpha, d_log_alpha_device, sizeof(float), cudaMemcpyDeviceToDevice, stream_);
}

// Explicit template instantiations
template struct E60ResidualNonlinearForward<__half>;
template struct E60ResidualNonlinearForward<float>;
template struct E60ResidualNonlinearForward<double>;

template struct E60ResidualNonlinearBackward<__half>;
template struct E60ResidualNonlinearBackward<float>;
template struct E60ResidualNonlinearBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
