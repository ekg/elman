// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E39: No-Bias Elman - E38 without bias term
//
// Architecture:
// x = silu(in_proj(x))              # Pre-activation
// h_t = tanh(x_t + W_h @ h_{t-1})   # No W_x, no bias!
// output = h * silu(h)              # SELF-GATE: h gates itself
//
// Key simplification: removes W_x and bias entirely.
// The recurrence is just: tanh(x + W_h @ h_prev)

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

// =============================================================================
// Forward Kernels - No W_x, No Bias
// =============================================================================

// BF16-optimized: Fused x + Rh + tanh + self-gate (NO W_x, NO bias)
__global__ void FusedTanhGateNoBiasKernel_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ x_t,    // Already pre-activated input
    const __nv_bfloat16* __restrict__ Rh,     // W_h @ h_prev result
    __nv_bfloat16* __restrict__ h_out,
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        // E39: Just x + Rh, no W_x, no bias
        __nv_bfloat16 sum = bf16_add(x_t[idx], Rh[idx]);

        // Store pre-activation
        if (v_cache) v_cache[idx] = sum;

        // tanh (need f32)
        float val = __bfloat162float(sum);
        float h_val = tanhf(val);
        h_out[idx] = __float2bfloat16(h_val);

        // E39 SELF-GATE: output = h * silu(h)
        float sigmoid_h = 1.0f / (1.0f + __expf(-h_val));
        float silu_h = h_val * sigmoid_h;

        output[idx] = __float2bfloat16(h_val * silu_h);
    }
}

// Generic version (no W_x, no bias)
template<typename T>
__global__ void FusedTanhGateNoBiasKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ x_t,
    const T* __restrict__ Rh,
    T* __restrict__ h_out,
    T* __restrict__ output,
    T* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        // E39: x + Rh only (no W_x, no bias)
        float val = static_cast<float>(x_t[idx]) + static_cast<float>(Rh[idx]);
        if (v_cache) v_cache[idx] = static_cast<T>(val);

        float h_val = tanhf(val);
        h_out[idx] = static_cast<T>(h_val);

        // Self-gate
        float sigmoid_h = 1.0f / (1.0f + expf(-h_val));
        float silu_h = h_val * sigmoid_h;

        output[idx] = static_cast<T>(h_val * silu_h);
    }
}

// =============================================================================
// Backward Kernels - No W_x, No Bias
// =============================================================================

// BF16-optimized backward through tanh (no bias accumulation)
__global__ void TanhBackwardNoBiasKernel_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ dh,
    const __nv_bfloat16* __restrict__ dh_recurrent,
    __nv_bfloat16* __restrict__ dv) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        // Combine gradients
        float grad;
        if (dh_recurrent) {
            __nv_bfloat16 combined = bf16_add(dh[idx], dh_recurrent[idx]);
            grad = __bfloat162float(combined);
        } else {
            grad = __bfloat162float(dh[idx]);
        }

        // dtanh
        float h = tanhf(__bfloat162float(v[idx]));
        float dtanh = 1.0f - h * h;
        float dv_val = grad * dtanh;

        dv[idx] = __float2bfloat16(dv_val);
        // E39: No bias, so no atomicAdd to db
    }
}

// Generic version
template<typename T>
__global__ void TanhBackwardNoBiasKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ v,
    const T* __restrict__ dh,
    const T* __restrict__ dh_recurrent,
    T* __restrict__ dv) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        float grad = static_cast<float>(dh[idx]);
        if (dh_recurrent) grad += static_cast<float>(dh_recurrent[idx]);

        float h = tanhf(static_cast<float>(v[idx]));
        float dtanh = 1.0f - h * h;
        float dv_val = grad * dtanh;
        dv[idx] = static_cast<T>(dv_val);
        // E39: No bias
    }
}

// BF16-optimized backward through self-gate
// output = h * silu(h) = h^2 * sigmoid(h)
// d(output)/dh = silu(h) * (2 + h*(1-sigmoid(h)))
__global__ void SelfGateBackward_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ h,
    const __nv_bfloat16* __restrict__ d_output,
    __nv_bfloat16* __restrict__ dh) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        float h_val = __bfloat162float(h[idx]);
        float dout = __bfloat162float(d_output[idx]);

        // Self-gate backward
        float sigmoid_h = 1.0f / (1.0f + __expf(-h_val));
        float silu_h = h_val * sigmoid_h;
        // d(h * silu(h))/dh = silu(h) * (2 + h*(1-sigmoid(h)))
        float grad_factor = silu_h * (2.0f + h_val * (1.0f - sigmoid_h));

        dh[idx] = __float2bfloat16(dout * grad_factor);
    }
}

// Generic version
template<typename T>
__global__ void SelfGateBackward(
    const int batch_size,
    const int dim,
    const T* __restrict__ h,
    const T* __restrict__ d_output,
    T* __restrict__ dh) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        float h_val = static_cast<float>(h[idx]);
        float dout = static_cast<float>(d_output[idx]);

        float sigmoid_h = 1.0f / (1.0f + expf(-h_val));
        float silu_h = h_val * sigmoid_h;
        float grad_factor = silu_h * (2.0f + h_val * (1.0f - sigmoid_h));

        dh[idx] = static_cast<T>(dout * grad_factor);
    }
}

// =============================================================================
// Utility Kernels
// =============================================================================

// BF16-optimized vector add inplace
__global__ void VectorAddInplace_BF16(
    const int n,
    __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = bf16_add(a[idx], b[idx]);
    }
}

// Generic version
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
// E39 No-Bias Forward - BF16 Specialization
// =============================================================================

template<>
E39NoBiasForward<__nv_bfloat16>::E39NoBiasForward(
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
void E39NoBiasForward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_h,
    const __nv_bfloat16* x,
    __nv_bfloat16* h,
    __nv_bfloat16* output,
    __nv_bfloat16* v,
    __nv_bfloat16* workspace) {

    static const __nv_bfloat16 alpha = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    __nv_bfloat16* tmp_Rh = workspace;  // Only need Rh workspace (no Wx)

    // Process each timestep
    for (int t = 0; t < steps; ++t) {
        const __nv_bfloat16* x_t = x + t * BD;
        const __nv_bfloat16* h_prev = h + t * BD;
        __nv_bfloat16* h_t = h + (t + 1) * BD;
        __nv_bfloat16* out_t = output + t * BD;
        __nv_bfloat16* v_t = training_ ? (v + t * BD) : nullptr;

        // tmp_Rh = h_prev @ W_h.T
        blas<__nv_bfloat16>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha,
            W_h, dim_,
            h_prev, dim_,
            &beta_zero,
            tmp_Rh, dim_);

        // E39 FUSED: h_t = tanh(x_t + tmp_Rh), output = h_t * silu(h_t)
        // No W_x, no bias!
        FusedTanhGateNoBiasKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, x_t, tmp_Rh, h_t, out_t, v_t);
    }
}

// =============================================================================
// E39 No-Bias Backward - BF16 Specialization
// =============================================================================

template<>
E39NoBiasBackward<__nv_bfloat16>::E39NoBiasBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void E39NoBiasBackward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_h,
    const __nv_bfloat16* x,
    const __nv_bfloat16* h,
    const __nv_bfloat16* v,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* dx,
    __nv_bfloat16* dW_h,
    __nv_bfloat16* workspace) {

    static const __nv_bfloat16 alpha_one = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);
    static const __nv_bfloat16 beta_one = __float2bfloat16(1.0f);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    __nv_bfloat16* dv_all = workspace;
    __nv_bfloat16* dh = workspace + steps * BD;
    __nv_bfloat16* dh_recurrent = workspace + (steps + 1) * BD;

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_h, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);

    // BPTT loop
    for (int t = steps - 1; t >= 0; --t) {
        const __nv_bfloat16* v_t = v + t * BD;
        const __nv_bfloat16* h_t = h + (t + 1) * BD;
        const __nv_bfloat16* d_out_t = d_output + t * BD;
        __nv_bfloat16* dv_t = dv_all + t * BD;

        // Backward through self-gate
        SelfGateBackward_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, d_out_t, dh);

        // Add recurrent gradient
        VectorAddInplace_BF16<<<num_blocks, block_size, 0, stream_>>>(BD, dh, dh_recurrent);

        // Backward through tanh (no bias)
        TanhBackwardNoBiasKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, v_t, dh, nullptr, dv_t);

        // dh_recurrent = W_h @ dv
        if (t > 0) {
            blas<__nv_bfloat16>::gemm(
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

    // E39: dx = dv (no W_x multiplication!)
    // Just copy dv_all to dx
    cudaMemcpyAsync(dx, dv_all, steps * BD * sizeof(__nv_bfloat16),
                    cudaMemcpyDeviceToDevice, stream_);

    // dW_h = h[0:T-1].T @ dv_all (accumulated over timesteps)
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        h, dim_,  // h[0:T] - uses first T states (h_prev for each timestep)
        dv_all, dim_,
        &beta_one,
        dW_h, dim_);
}

// =============================================================================
// Generic Template Implementations (float, half, double)
// =============================================================================

template<typename T>
E39NoBiasForward<T>::E39NoBiasForward(
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
void E39NoBiasForward<T>::Run(
    int steps,
    const T* W_h,
    const T* x,
    T* h,
    T* output,
    T* v,
    T* workspace) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    T* tmp_Rh = workspace;

    for (int t = 0; t < steps; ++t) {
        const T* x_t = x + t * BD;
        const T* h_prev = h + t * BD;
        T* h_t = h + (t + 1) * BD;
        T* out_t = output + t * BD;
        T* v_t = training_ ? (v + t * BD) : nullptr;

        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha,
            W_h, dim_,
            h_prev, dim_,
            &beta_zero,
            tmp_Rh, dim_);

        // E39: No W_x, no bias
        FusedTanhGateNoBiasKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, x_t, tmp_Rh, h_t, out_t, v_t);
    }
}

template<typename T>
E39NoBiasBackward<T>::E39NoBiasBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void E39NoBiasBackward<T>::Run(
    int steps,
    const T* W_h,
    const T* x,
    const T* h,
    const T* v,
    const T* d_output,
    T* dx,
    T* dW_h,
    T* workspace) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    T* dv_all = workspace;
    T* dh = workspace + steps * BD;
    T* dh_recurrent = workspace + (steps + 1) * BD;

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(T), stream_);
    cudaMemsetAsync(dW_h, 0, dim_ * dim_ * sizeof(T), stream_);

    for (int t = steps - 1; t >= 0; --t) {
        const T* v_t = v + t * BD;
        const T* h_t = h + (t + 1) * BD;
        const T* d_out_t = d_output + t * BD;
        T* dv_t = dv_all + t * BD;

        SelfGateBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, d_out_t, dh);

        VectorAddInplace<T><<<num_blocks, block_size, 0, stream_>>>(BD, dh, dh_recurrent);

        TanhBackwardNoBiasKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, v_t, dh, nullptr, dv_t);

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

    // dx = dv (no W_x)
    cudaMemcpyAsync(dx, dv_all, steps * BD * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream_);

    // dW_h
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha,
        h, dim_,
        dv_all, dim_,
        &beta_one,
        dW_h, dim_);
}

// Explicit template instantiations
template struct E39NoBiasForward<__half>;
template struct E39NoBiasForward<float>;
template struct E39NoBiasForward<double>;

template struct E39NoBiasBackward<__half>;
template struct E39NoBiasBackward<float>;
template struct E39NoBiasBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
