// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E38: No W_x Elman - removes W_x matrix entirely
//
// Key difference from E33: No W_x @ x_t term.
// The recurrence becomes: raw = x_t + W_h @ h_{t-1} + b
//
// Architecture:
// x = in_proj(x)                     # Project (done in Python)
// x = silu(x)                        # Pre-activation (done in Python)
// h_t = tanh(x_t + W_h @ h_{t-1} + b)  # NO W_x! Direct add
// output = h * silu(h)               # SELF-GATE: h gates itself

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
// E38 Forward Kernels: x + Rh + b -> tanh -> h; h * silu(h) -> output
// Key change: No Wx term! Input x goes directly into the addition.
// =============================================================================

// BF16-optimized: Fused x + Rh + bias + tanh + self-gate
// NOTE: Unlike E33, there's no Wx - we use x_t directly
__global__ void E38FusedTanhGateKernel_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ x_t,   // Input directly (no W_x @ x)
    const __nv_bfloat16* __restrict__ Rh,    // W_h @ h_prev
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ h_out,
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // E38: Direct add x_t + Rh + b (no W_x @ x!)
        __nv_bfloat16 sum = bf16_add(bf16_add(x_t[idx], Rh[idx]), b[d]);

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

// Generic fused version
template<typename T>
__global__ void E38FusedTanhGateKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ x_t,   // Input directly (no W_x @ x)
    const T* __restrict__ Rh,
    const T* __restrict__ b,
    T* __restrict__ h_out,
    T* __restrict__ output,
    T* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // E38: Direct add x_t + Rh + b (no W_x @ x!)
        float val = static_cast<float>(x_t[idx]) + static_cast<float>(Rh[idx]) + static_cast<float>(b[d]);
        if (v_cache) v_cache[idx] = static_cast<T>(val);

        float h_val = tanhf(val);
        h_out[idx] = static_cast<T>(h_val);

        // Self-gate: output = h * silu(h)
        float sigmoid_h = 1.0f / (1.0f + expf(-h_val));
        float silu_h = h_val * sigmoid_h;

        output[idx] = static_cast<T>(h_val * silu_h);
    }
}

// =============================================================================
// E38 Backward Kernels
// =============================================================================

// BF16-optimized backward through tanh (same as E33)
__global__ void E38TanhBackwardKernel_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ dh,
    const __nv_bfloat16* __restrict__ dh_recurrent,
    __nv_bfloat16* __restrict__ dv,
    float* __restrict__ db) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

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
        atomicAdd(&db[d], dv_val);
    }
}

// Generic version
template<typename T>
__global__ void E38TanhBackwardKernel(
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

// BF16-optimized backward through self-gate
// output = h * silu(h) = h² * sigmoid(h)
// d(output)/dh = silu(h) * (2 + h*(1-sigmoid(h)))
__global__ void E38GateBackwardKernel_BF16(
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
__global__ void E38GateBackwardKernel(
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

        // Self-gate backward
        float sigmoid_h = 1.0f / (1.0f + expf(-h_val));
        float silu_h = h_val * sigmoid_h;
        float grad_factor = silu_h * (2.0f + h_val * (1.0f - sigmoid_h));

        dh[idx] = static_cast<T>(dout * grad_factor);
    }
}

// =============================================================================
// Utility Kernels
// =============================================================================

__global__ void VectorAddInplace_BF16(
    const int n,
    __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = bf16_add(a[idx], b[idx]);
    }
}

template<typename T>
__global__ void VectorAddInplace(const int n, T* __restrict__ a, const T* __restrict__ b) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = static_cast<T>(static_cast<float>(a[idx]) + static_cast<float>(b[idx]));
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
// E38 No-W_x Elman Forward - BF16 Specialization
// =============================================================================

template<>
E38NoWxForward<__nv_bfloat16>::E38NoWxForward(
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
void E38NoWxForward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_h,
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

    // E38: No pre-computed W_x @ x! Just workspace for tmp_Rh
    __nv_bfloat16* tmp_Rh = workspace;

    // Process each timestep with FUSED kernel
    for (int t = 0; t < steps; ++t) {
        const __nv_bfloat16* x_t = x + t * BD;  // Input directly (no W_x @ x)
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

        // E38 FUSED: h_t = tanh(x_t + tmp_Rh + b), output = h_t * silu(h_t)
        // Note: No W_x @ x_t term - x_t goes directly!
        E38FusedTanhGateKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, x_t, tmp_Rh, b, h_t, out_t, v_t);
    }
}

// =============================================================================
// E38 No-W_x Elman Backward - BF16 Specialization
// =============================================================================

template<>
E38NoWxBackward<__nv_bfloat16>::E38NoWxBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void E38NoWxBackward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_h,
    const __nv_bfloat16* x,
    const __nv_bfloat16* h,
    const __nv_bfloat16* v,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* dx,
    __nv_bfloat16* dW_h,
    __nv_bfloat16* db,
    __nv_bfloat16* workspace) {

    static const __nv_bfloat16 alpha_one = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);
    static const __nv_bfloat16 beta_one = __float2bfloat16(1.0f);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Workspace layout: [dv_all: T*BD] [dh: BD] [dh_recurrent: BD] [db_float: dim]
    __nv_bfloat16* dv_all = workspace;
    __nv_bfloat16* dh = workspace + steps * BD;
    __nv_bfloat16* dh_recurrent = workspace + (steps + 1) * BD;
    float* db_float = reinterpret_cast<float*>(workspace + (steps + 2) * BD);

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW_h, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);

    // BPTT loop
    for (int t = steps - 1; t >= 0; --t) {
        const __nv_bfloat16* v_t = v + t * BD;
        const __nv_bfloat16* h_t = h + (t + 1) * BD;
        const __nv_bfloat16* d_out_t = d_output + t * BD;
        __nv_bfloat16* dv_t = dv_all + t * BD;

        // Backward through self-gate
        E38GateBackwardKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, d_out_t, dh);

        // Add recurrent gradient
        VectorAddInplace_BF16<<<num_blocks, block_size, 0, stream_>>>(BD, dh, dh_recurrent);

        // Backward through tanh
        E38TanhBackwardKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, v_t, dh, nullptr, dv_t, db_float);

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

    // E38: dx = dv directly (no W_x to multiply through!)
    // Just copy dv_all to dx
    cudaMemcpyAsync(dx, dv_all, steps * BD * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream_);

    // dW_h gradient
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        h, dim_,
        dv_all, dim_,
        &beta_one,
        dW_h, dim_);

    // Copy float gradients to bf16
    CopyFloatToT<__nv_bfloat16><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_float, db);
}

// =============================================================================
// Generic Template Implementations (float, half, double)
// =============================================================================

template<typename T>
E38NoWxForward<T>::E38NoWxForward(
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
void E38NoWxForward<T>::Run(
    int steps,
    const T* W_h,
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

    // E38: No pre-computed W_x @ x! Just workspace for tmp_Rh
    T* tmp_Rh = workspace;

    for (int t = 0; t < steps; ++t) {
        const T* x_t = x + t * BD;  // Input directly
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

        // E38: Use fused kernel (x_t directly, no W_x)
        E38FusedTanhGateKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, x_t, tmp_Rh, b, h_t, out_t, v_t);
    }
}

template<typename T>
E38NoWxBackward<T>::E38NoWxBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void E38NoWxBackward<T>::Run(
    int steps,
    const T* W_h,
    const T* x,
    const T* h,
    const T* v,
    const T* d_output,
    T* dx,
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

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(T), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW_h, 0, dim_ * dim_ * sizeof(T), stream_);

    for (int t = steps - 1; t >= 0; --t) {
        const T* v_t = v + t * BD;
        const T* h_t = h + (t + 1) * BD;
        const T* d_out_t = d_output + t * BD;
        T* dv_t = dv_all + t * BD;

        E38GateBackwardKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, d_out_t, dh);

        VectorAddInplace<T><<<num_blocks, block_size, 0, stream_>>>(BD, dh, dh_recurrent);

        E38TanhBackwardKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, v_t, dh, nullptr, dv_t, db_float);

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

    // E38: dx = dv directly (no W_x!)
    cudaMemcpyAsync(dx, dv_all, steps * BD * sizeof(T), cudaMemcpyDeviceToDevice, stream_);

    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha,
        h, dim_,
        dv_all, dim_,
        &beta_one,
        dW_h, dim_);

    CopyFloatToT<T><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_float, db);
}

// Explicit template instantiations
template struct E38NoWxForward<__half>;
template struct E38NoWxForward<float>;
template struct E38NoWxForward<double>;

template struct E38NoWxBackward<__half>;
template struct E38NoWxBackward<float>;
template struct E38NoWxBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
