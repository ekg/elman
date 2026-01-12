// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E41: Diagonal W_x Elman - W_x is diagonal (vector instead of matrix)
//
// Key difference from E33: W_x is a diagonal vector d_x instead of dense matrix
// This reduces W_x from O(dim^2) to O(dim) parameters.
// Different from E34 which has diagonal W_h - E41 keeps W_h matrix.
//
// Architecture:
// x_proj = silu(in_proj(x))                   # Project input
// h_t = tanh(d_x * x_t + W_h @ h_{t-1} + b)   # d_x is [dim] vector, element-wise multiply
// output = h * silu(h)                        # Self-gating from E33

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
// E41 Forward Kernels: Fused d_x*x + Rh + b + tanh + self-gate
// =============================================================================

// BF16-optimized: Fused d_x*x + Rh + bias + tanh + self-gate
// d_x is a [dim] diagonal vector, Rh is pre-computed W_h @ h_prev
__global__ void E41FusedKernel_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ Rh,
    const __nv_bfloat16* __restrict__ d_x,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ h_out,
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int dim_idx = idx % dim;

        // d_x * x (element-wise diagonal)
        __nv_bfloat16 dx_x = bf16_mul(d_x[dim_idx], x[idx]);

        // d_x*x + Rh + b
        __nv_bfloat16 sum = bf16_add(bf16_add(dx_x, Rh[idx]), b[dim_idx]);

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
__global__ void E41FusedKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ x,
    const T* __restrict__ Rh,
    const T* __restrict__ d_x,
    const T* __restrict__ b,
    T* __restrict__ h_out,
    T* __restrict__ output,
    T* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int dim_idx = idx % dim;

        // d_x * x (element-wise diagonal)
        float dx_x = static_cast<float>(d_x[dim_idx]) * static_cast<float>(x[idx]);

        // d_x*x + Rh + b
        float val = dx_x + static_cast<float>(Rh[idx]) + static_cast<float>(b[dim_idx]);
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
// E41 Backward Kernels
// =============================================================================

// BF16: Backward through self-gate
// output = h * silu(h) = h^2 * sigmoid(h)
// d(output)/dh = silu(h) * (2 + h*(1-sigmoid(h)))
__global__ void E41GateBackward_BF16(
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
__global__ void E41GateBackward(
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

// BF16: Backward through tanh and compute dv, db, dd_x
// v = d_x*x + Rh + b
// dv = dh * (1 - tanh(v)^2)
// dd_x += dv * x (sum over batch and time)
// db += dv (sum over batch)
__global__ void E41TanhBackward_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ dh,
    const __nv_bfloat16* __restrict__ dh_recurrent,
    __nv_bfloat16* __restrict__ dv,
    float* __restrict__ db,
    float* __restrict__ dd_x) {

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

        // dd_x = dv * x
        float x_val = __bfloat162float(x[idx]);
        atomicAdd(&dd_x[dim_idx], dv_val * x_val);
    }
}

// Generic version
template<typename T>
__global__ void E41TanhBackward(
    const int batch_size,
    const int dim,
    const T* __restrict__ v,
    const T* __restrict__ x,
    const T* __restrict__ dh,
    const T* __restrict__ dh_recurrent,
    T* __restrict__ dv,
    float* __restrict__ db,
    float* __restrict__ dd_x) {

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

        // dd_x = dv * x
        float x_val = static_cast<float>(x[idx]);
        atomicAdd(&dd_x[dim_idx], dv_val * x_val);
    }
}

// BF16: Compute dx = d_x * dv (element-wise)
__global__ void E41InputGrad_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ d_x,
    const __nv_bfloat16* __restrict__ dv,
    __nv_bfloat16* __restrict__ dx) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int dim_idx = idx % dim;
        dx[idx] = bf16_mul(d_x[dim_idx], dv[idx]);
    }
}

// Generic version
template<typename T>
__global__ void E41InputGrad(
    const int batch_size,
    const int dim,
    const T* __restrict__ d_x,
    const T* __restrict__ dv,
    T* __restrict__ dx) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int dim_idx = idx % dim;
        float d_x_val = static_cast<float>(d_x[dim_idx]);
        float dv_val = static_cast<float>(dv[idx]);
        dx[idx] = static_cast<T>(d_x_val * dv_val);
    }
}

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
// E41 Diagonal W_x Forward - BF16 Specialization
// =============================================================================

template<>
E41DiagonalWxForward<__nv_bfloat16>::E41DiagonalWxForward(
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
void E41DiagonalWxForward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* d_x,     // [dim] diagonal vector (replaces W_x matrix)
    const __nv_bfloat16* W_h,     // [dim, dim]
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

    __nv_bfloat16* tmp_Rh = workspace;

    // Process each timestep
    // E41: no batch W_x @ x GEMM since W_x is diagonal
    // Instead: per-step W_h @ h GEMM + element-wise d_x * x
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

        // E41 FUSED: h_t = tanh(d_x*x_t + tmp_Rh + b), output = h_t * silu(h_t)
        E41FusedKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, x_t, tmp_Rh, d_x, b, h_t, out_t, v_t);
    }
}

// =============================================================================
// E41 Diagonal W_x Backward - BF16 Specialization
// =============================================================================

template<>
E41DiagonalWxBackward<__nv_bfloat16>::E41DiagonalWxBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void E41DiagonalWxBackward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* d_x,     // [dim] diagonal vector
    const __nv_bfloat16* W_h,     // [dim, dim]
    const __nv_bfloat16* x,
    const __nv_bfloat16* h,
    const __nv_bfloat16* v,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* dx,
    __nv_bfloat16* dd_x,          // [dim] gradient for diagonal
    __nv_bfloat16* dW_h,
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
    float* dd_x_float = db_float + dim_;  // After db_float

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dd_x_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW_h, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);

    // BPTT loop
    for (int t = steps - 1; t >= 0; --t) {
        const __nv_bfloat16* v_t = v + t * BD;
        const __nv_bfloat16* x_t = x + t * BD;
        const __nv_bfloat16* h_t = h + (t + 1) * BD;
        const __nv_bfloat16* h_prev = h + t * BD;
        const __nv_bfloat16* d_out_t = d_output + t * BD;
        __nv_bfloat16* dv_t = dv_all + t * BD;
        __nv_bfloat16* dx_t = dx + t * BD;

        // Backward through self-gate
        E41GateBackward_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, d_out_t, dh);

        // Add recurrent gradient
        if (t < steps - 1) {
            VectorAddInplace_BF16<<<num_blocks, block_size, 0, stream_>>>(BD, dh, dh_recurrent);
        }

        // Backward through tanh: computes dv, accumulates db and dd_x
        E41TanhBackward_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, v_t, x_t, dh, nullptr,
            dv_t, db_float, dd_x_float);

        // Compute dx = d_x * dv (element-wise)
        E41InputGrad_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, d_x, dv_t, dx_t);

        // Compute dh_recurrent = W_h @ dv (standard GEMM since W_h is a matrix)
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

        // Accumulate dW_h = h_prev.T @ dv
        blas<__nv_bfloat16>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_,
            &alpha_one,
            h_prev, dim_,
            dv_t, dim_,
            &beta_one,
            dW_h, dim_);
    }

    // Copy float gradients to bf16
    CopyFloatToT<__nv_bfloat16><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_float, db);
    CopyFloatToT<__nv_bfloat16><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, dd_x_float, dd_x);
}

// =============================================================================
// Generic Template Implementations (float, half, double)
// =============================================================================

template<typename T>
E41DiagonalWxForward<T>::E41DiagonalWxForward(
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
void E41DiagonalWxForward<T>::Run(
    int steps,
    const T* d_x,    // [dim] diagonal vector
    const T* W_h,    // [dim, dim]
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

    T* tmp_Rh = workspace;

    for (int t = 0; t < steps; ++t) {
        const T* x_t = x + t * BD;
        const T* h_prev = h + t * BD;
        T* h_t = h + (t + 1) * BD;
        T* out_t = output + t * BD;
        T* v_t = training_ ? (v + t * BD) : nullptr;

        // W_h @ h_prev
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha,
            W_h, dim_,
            h_prev, dim_,
            &beta_zero,
            tmp_Rh, dim_);

        E41FusedKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, x_t, tmp_Rh, d_x, b, h_t, out_t, v_t);
    }
}

template<typename T>
E41DiagonalWxBackward<T>::E41DiagonalWxBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void E41DiagonalWxBackward<T>::Run(
    int steps,
    const T* d_x,
    const T* W_h,
    const T* x,
    const T* h,
    const T* v,
    const T* d_output,
    T* dx,
    T* dd_x,
    T* dW_h,
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
    float* dd_x_float = db_float + dim_;

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(T), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dd_x_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW_h, 0, dim_ * dim_ * sizeof(T), stream_);

    for (int t = steps - 1; t >= 0; --t) {
        const T* v_t = v + t * BD;
        const T* x_t = x + t * BD;
        const T* h_t = h + (t + 1) * BD;
        const T* h_prev = h + t * BD;
        const T* d_out_t = d_output + t * BD;
        T* dv_t = dv_all + t * BD;
        T* dx_t = dx + t * BD;

        E41GateBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, d_out_t, dh);

        if (t < steps - 1) {
            VectorAddInplace<T><<<num_blocks, block_size, 0, stream_>>>(BD, dh, dh_recurrent);
        }

        E41TanhBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, v_t, x_t, dh, nullptr,
            dv_t, db_float, dd_x_float);

        E41InputGrad<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, d_x, dv_t, dx_t);

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

        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_,
            &alpha,
            h_prev, dim_,
            dv_t, dim_,
            &beta_one,
            dW_h, dim_);
    }

    CopyFloatToT<T><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_float, db);
    CopyFloatToT<T><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, dd_x_float, dd_x);
}

// Explicit template instantiations
template struct E41DiagonalWxForward<__half>;
template struct E41DiagonalWxForward<float>;
template struct E41DiagonalWxForward<double>;

template struct E41DiagonalWxBackward<__half>;
template struct E41DiagonalWxBackward<float>;
template struct E41DiagonalWxBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
