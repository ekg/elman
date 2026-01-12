// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E34: Diagonal W_h Elman - W_h is diagonal (vector instead of matrix)
//
// Key difference from E33: W_h is a diagonal vector d instead of dense matrix
// This reduces W_h from O(dim^2) to O(dim) parameters.
//
// Architecture:
// x = in_proj(x)                       # Project input
// x = silu(x)                          # Pre-activation
// h_t = tanh(W_x @ x_t + d * h_{t-1} + b)  # d is [dim] vector, element-wise multiply
// output = h * silu(h)                 # Self-gating from E33

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

__device__ __forceinline__ __nv_bfloat16 bf16_fma(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c) {
#if __CUDA_ARCH__ >= 800
    return __hfma(a, b, c);
#else
    return __float2bfloat16(__bfloat162float(a) * __bfloat162float(b) + __bfloat162float(c));
#endif
}

// =============================================================================
// E34 Forward Kernels: Fused Wx + d*h + b + tanh + self-gate
// =============================================================================

// BF16-optimized: Fused Wx + d*h_prev + bias + tanh + self-gate
// d is a [dim] diagonal vector
__global__ void E34FusedKernel_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ Wx,
    const __nv_bfloat16* __restrict__ h_prev,
    const __nv_bfloat16* __restrict__ d,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ h_out,
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int dim_idx = idx % dim;

        // d * h_prev (element-wise diagonal)
        __nv_bfloat16 dh = bf16_mul(d[dim_idx], h_prev[idx]);

        // Wx + d*h + b
        __nv_bfloat16 sum = bf16_add(bf16_add(Wx[idx], dh), b[dim_idx]);

        // Store pre-activation
        if (v_cache) v_cache[idx] = sum;

        // tanh (need f32)
        float val = __bfloat162float(sum);
        float h_val = tanhf(val);
        h_out[idx] = __float2bfloat16(h_val);

        // Self-gate: output = h * silu(h)
        float sigmoid_h = 1.0f / (1.0f + __expf(-h_val));
        float silu_h = h_val * sigmoid_h;
        output[idx] = __float2bfloat16(h_val * silu_h);
    }
}

// Generic version
template<typename T>
__global__ void E34FusedKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ Wx,
    const T* __restrict__ h_prev,
    const T* __restrict__ d,
    const T* __restrict__ b,
    T* __restrict__ h_out,
    T* __restrict__ output,
    T* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int dim_idx = idx % dim;

        // d * h_prev (element-wise diagonal)
        float dh = static_cast<float>(d[dim_idx]) * static_cast<float>(h_prev[idx]);

        // Wx + d*h + b
        float val = static_cast<float>(Wx[idx]) + dh + static_cast<float>(b[dim_idx]);
        if (v_cache) v_cache[idx] = static_cast<T>(val);

        // tanh
        float h_val = tanhf(val);
        h_out[idx] = static_cast<T>(h_val);

        // Self-gate: output = h * silu(h)
        float sigmoid_h = 1.0f / (1.0f + expf(-h_val));
        float silu_h = h_val * sigmoid_h;
        output[idx] = static_cast<T>(h_val * silu_h);
    }
}

// =============================================================================
// E34 Backward Kernels
// =============================================================================

// BF16: Backward through self-gate
// output = h * silu(h) = h^2 * sigmoid(h)
// d(output)/dh = silu(h) * (2 + h*(1-sigmoid(h)))
__global__ void E34GateBackward_BF16(
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

        float sigmoid_h = 1.0f / (1.0f + __expf(-h_val));
        float silu_h = h_val * sigmoid_h;
        float grad_factor = silu_h * (2.0f + h_val * (1.0f - sigmoid_h));

        dh[idx] = __float2bfloat16(dout * grad_factor);
    }
}

// Generic version
template<typename T>
__global__ void E34GateBackward(
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

// BF16: Backward through tanh and compute dv, db, dd
// v = Wx + d*h_prev + b
// dv = dh * (1 - tanh(v)^2)
// dd += dv * h_prev (sum over batch)
// db += dv (sum over batch)
__global__ void E34TanhBackward_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ h_prev,
    const __nv_bfloat16* __restrict__ dh,
    const __nv_bfloat16* __restrict__ dh_recurrent,
    __nv_bfloat16* __restrict__ dv,
    float* __restrict__ db,
    float* __restrict__ dd) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int dim_idx = idx % dim;

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
        atomicAdd(&db[dim_idx], dv_val);

        // dd = dv * h_prev
        float h_prev_val = __bfloat162float(h_prev[idx]);
        atomicAdd(&dd[dim_idx], dv_val * h_prev_val);
    }
}

// Generic version
template<typename T>
__global__ void E34TanhBackward(
    const int batch_size,
    const int dim,
    const T* __restrict__ v,
    const T* __restrict__ h_prev,
    const T* __restrict__ dh,
    const T* __restrict__ dh_recurrent,
    T* __restrict__ dv,
    float* __restrict__ db,
    float* __restrict__ dd) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int dim_idx = idx % dim;

        float grad = static_cast<float>(dh[idx]);
        if (dh_recurrent) grad += static_cast<float>(dh_recurrent[idx]);

        float h = tanhf(static_cast<float>(v[idx]));
        float dtanh = 1.0f - h * h;
        float dv_val = grad * dtanh;
        dv[idx] = static_cast<T>(dv_val);

        atomicAdd(&db[dim_idx], dv_val);

        // dd = dv * h_prev
        float h_prev_val = static_cast<float>(h_prev[idx]);
        atomicAdd(&dd[dim_idx], dv_val * h_prev_val);
    }
}

// BF16: Compute dh_recurrent = d * dv (element-wise)
__global__ void E34RecurrentGrad_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ d,
    const __nv_bfloat16* __restrict__ dv,
    __nv_bfloat16* __restrict__ dh_recurrent) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int dim_idx = idx % dim;
        dh_recurrent[idx] = bf16_mul(d[dim_idx], dv[idx]);
    }
}

// Generic version
template<typename T>
__global__ void E34RecurrentGrad(
    const int batch_size,
    const int dim,
    const T* __restrict__ d,
    const T* __restrict__ dv,
    T* __restrict__ dh_recurrent) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int dim_idx = idx % dim;
        float d_val = static_cast<float>(d[dim_idx]);
        float dv_val = static_cast<float>(dv[idx]);
        dh_recurrent[idx] = static_cast<T>(d_val * dv_val);
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
// E34 Diagonal W_h Forward - BF16 Specialization
// =============================================================================

template<>
E34DiagonalWhForward<__nv_bfloat16>::E34DiagonalWhForward(
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
void E34DiagonalWhForward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_x,
    const __nv_bfloat16* d,      // [dim] diagonal vector (replaces W_h matrix)
    const __nv_bfloat16* b,
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

    __nv_bfloat16* tmp_Wx = workspace;

    // Pre-compute W_x @ x for all timesteps (one big GEMM)
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha,
        W_x, dim_,
        x, dim_,
        &beta_zero,
        tmp_Wx, dim_);

    // Process each timestep - now using element-wise d*h instead of W_h @ h GEMM
    for (int t = 0; t < steps; ++t) {
        const __nv_bfloat16* Wx_t = tmp_Wx + t * BD;
        const __nv_bfloat16* h_prev = h + t * BD;
        __nv_bfloat16* h_t = h + (t + 1) * BD;
        __nv_bfloat16* out_t = output + t * BD;
        __nv_bfloat16* v_t = training_ ? (v + t * BD) : nullptr;

        // E34 FUSED: h_t = tanh(Wx_t + d*h_prev + b), output = h_t * silu(h_t)
        E34FusedKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, Wx_t, h_prev, d, b, h_t, out_t, v_t);
    }
}

// =============================================================================
// E34 Diagonal W_h Backward - BF16 Specialization
// =============================================================================

template<>
E34DiagonalWhBackward<__nv_bfloat16>::E34DiagonalWhBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void E34DiagonalWhBackward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_x,
    const __nv_bfloat16* d,      // [dim] diagonal vector
    const __nv_bfloat16* x,
    const __nv_bfloat16* h,
    const __nv_bfloat16* v,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* dx,
    __nv_bfloat16* dW_x,
    __nv_bfloat16* dd,           // [dim] gradient for diagonal
    __nv_bfloat16* db,
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
    float* db_float = reinterpret_cast<float*>(workspace + (steps + 2) * BD);
    float* dd_float = db_float + dim_;  // After db_float

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dd_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW_x, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);

    // BPTT loop
    for (int t = steps - 1; t >= 0; --t) {
        const __nv_bfloat16* v_t = v + t * BD;
        const __nv_bfloat16* h_t = h + (t + 1) * BD;
        const __nv_bfloat16* h_prev = h + t * BD;
        const __nv_bfloat16* d_out_t = d_output + t * BD;
        __nv_bfloat16* dv_t = dv_all + t * BD;

        // Backward through self-gate
        E34GateBackward_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, d_out_t, dh);

        // Backward through tanh: computes dv, accumulates db and dd
        E34TanhBackward_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, v_t, h_prev, dh, t < steps - 1 ? dh_recurrent : nullptr,
            dv_t, db_float, dd_float);

        // Compute dh_recurrent = d * dv (element-wise, no GEMM needed!)
        if (t > 0) {
            E34RecurrentGrad_BF16<<<num_blocks, block_size, 0, stream_>>>(
                batch_size_, dim_, d, dv_t, dh_recurrent);
        }
    }

    // Batch GEMM for dx = W_x @ dv
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_x, dim_,
        dv_all, dim_,
        &beta_zero,
        dx, dim_);

    // Batch GEMM for dW_x = x.T @ dv
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        dv_all, dim_,
        &beta_one,
        dW_x, dim_);

    // Copy float gradients to bf16
    CopyFloatToT<__nv_bfloat16><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_float, db);
    CopyFloatToT<__nv_bfloat16><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, dd_float, dd);
}

// =============================================================================
// Generic Template Implementations (float, half, double)
// =============================================================================

template<typename T>
E34DiagonalWhForward<T>::E34DiagonalWhForward(
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
void E34DiagonalWhForward<T>::Run(
    int steps,
    const T* W_x,
    const T* d,      // [dim] diagonal vector
    const T* b,
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

    T* tmp_Wx = workspace;

    // Pre-compute W_x @ x
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha,
        W_x, dim_,
        x, dim_,
        &beta_zero,
        tmp_Wx, dim_);

    for (int t = 0; t < steps; ++t) {
        const T* Wx_t = tmp_Wx + t * BD;
        const T* h_prev = h + t * BD;
        T* h_t = h + (t + 1) * BD;
        T* out_t = output + t * BD;
        T* v_t = training_ ? (v + t * BD) : nullptr;

        E34FusedKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, Wx_t, h_prev, d, b, h_t, out_t, v_t);
    }
}

template<typename T>
E34DiagonalWhBackward<T>::E34DiagonalWhBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void E34DiagonalWhBackward<T>::Run(
    int steps,
    const T* W_x,
    const T* d,
    const T* x,
    const T* h,
    const T* v,
    const T* d_output,
    T* dx,
    T* dW_x,
    T* dd,
    T* db,
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
    float* db_float = reinterpret_cast<float*>(workspace + (steps + 2) * BD);
    float* dd_float = db_float + dim_;

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(T), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dd_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW_x, 0, dim_ * dim_ * sizeof(T), stream_);

    for (int t = steps - 1; t >= 0; --t) {
        const T* v_t = v + t * BD;
        const T* h_t = h + (t + 1) * BD;
        const T* h_prev = h + t * BD;
        const T* d_out_t = d_output + t * BD;
        T* dv_t = dv_all + t * BD;

        E34GateBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, d_out_t, dh);

        E34TanhBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, v_t, h_prev, dh, t < steps - 1 ? dh_recurrent : nullptr,
            dv_t, db_float, dd_float);

        if (t > 0) {
            E34RecurrentGrad<T><<<num_blocks, block_size, 0, stream_>>>(
                batch_size_, dim_, d, dv_t, dh_recurrent);
        }
    }

    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha,
        W_x, dim_,
        dv_all, dim_,
        &beta_zero,
        dx, dim_);

    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha,
        x, dim_,
        dv_all, dim_,
        &beta_one,
        dW_x, dim_);

    CopyFloatToT<T><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_float, db);
    CopyFloatToT<T><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, dd_float, dd);
}

// Explicit template instantiations
template struct E34DiagonalWhForward<__half>;
template struct E34DiagonalWhForward<float>;
template struct E34DiagonalWhForward<double>;

template struct E34DiagonalWhBackward<__half>;
template struct E34DiagonalWhBackward<float>;
template struct E34DiagonalWhBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
