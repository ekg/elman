// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// Softsign Elman - E1 with softsign activation instead of tanh
//
// softsign(x) = x / (1 + |x|)
// - Bounded (-1, 1) like tanh
// - Cheaper: no exp(), just div and abs
// - Smoother gradients: derivative = 1/(1+|x|)^2
//
// Architecture:
// x, z = split(in_proj(x))           # Pre-computed before kernel
// x = silu(x)                        # Pre-computed before kernel
// h_t = softsign(W_x @ x_t + W_h @ h_{t-1} + b)  # Elman recurrence
// output = h * silu(z)               # Gate with z branch

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

// Native bf16 add - uses hardware instruction on Ampere+
__device__ __forceinline__ __nv_bfloat16 bf16_add(__nv_bfloat16 a, __nv_bfloat16 b) {
#if __CUDA_ARCH__ >= 800
    return __hadd(a, b);
#else
    return __float2bfloat16(__bfloat162float(a) + __bfloat162float(b));
#endif
}

// Native bf16 multiply
__device__ __forceinline__ __nv_bfloat16 bf16_mul(__nv_bfloat16 a, __nv_bfloat16 b) {
#if __CUDA_ARCH__ >= 800
    return __hmul(a, b);
#else
    return __float2bfloat16(__bfloat162float(a) * __bfloat162float(b));
#endif
}

// Native bf16 fused multiply-add: a * b + c
__device__ __forceinline__ __nv_bfloat16 bf16_fma(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c) {
#if __CUDA_ARCH__ >= 800
    return __hfma(a, b, c);
#else
    return __float2bfloat16(__bfloat162float(a) * __bfloat162float(b) + __bfloat162float(c));
#endif
}

// =============================================================================
// Optimized Forward Kernels
// =============================================================================

// BF16-optimized: Fused Wx + Rh + bias + softsign
// Uses native bf16 adds, converts to f32 for softsign
__global__ void FusedSoftsignKernel_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ Wx,
    const __nv_bfloat16* __restrict__ Rh,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ h_out,
    __nv_bfloat16* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // Native bf16 additions (no f32 conversion)
        __nv_bfloat16 sum = bf16_add(bf16_add(Wx[idx], Rh[idx]), b[d]);

        // Store pre-activation in bf16
        if (v_cache) v_cache[idx] = sum;

        // softsign: x / (1 + |x|) - cheaper than tanh, no exp()
        float val = __bfloat162float(sum);
        h_out[idx] = __float2bfloat16(val / (1.0f + fabsf(val)));
    }
}

// Generic version for other types (float, half, double)
template<typename T>
__global__ void FusedSoftsignKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ Wx,
    const T* __restrict__ Rh,
    const T* __restrict__ b,
    T* __restrict__ h_out,
    T* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;
        float val = static_cast<float>(Wx[idx]) + static_cast<float>(Rh[idx]) + static_cast<float>(b[d]);
        if (v_cache) v_cache[idx] = static_cast<T>(val);
        h_out[idx] = static_cast<T>(val / (1.0f + fabsf(val)));
    }
}

// BF16-optimized: output = h * silu(z)
__global__ void MambaGateForward_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ h,
    const __nv_bfloat16* __restrict__ z,
    __nv_bfloat16* __restrict__ output) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        float h_val = __bfloat162float(h[idx]);
        float z_val = __bfloat162float(z[idx]);

        // silu(z) = z * sigmoid(z) - need f32 for exp
        float sigmoid_z = 1.0f / (1.0f + __expf(-z_val));
        float silu_z = z_val * sigmoid_z;

        output[idx] = __float2bfloat16(h_val * silu_z);
    }
}

// Generic version
template<typename T>
__global__ void MambaGateForward(
    const int batch_size,
    const int dim,
    const T* __restrict__ h,
    const T* __restrict__ z,
    T* __restrict__ output) {

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

// =============================================================================
// FUSED Forward Kernel: softsign + gate in one pass (reduces memory traffic)
// =============================================================================

__global__ void FusedSoftsignGateKernel_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ Wx,
    const __nv_bfloat16* __restrict__ Rh,
    const __nv_bfloat16* __restrict__ b,
    const __nv_bfloat16* __restrict__ z,
    __nv_bfloat16* __restrict__ h_out,
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // Native bf16 additions
        __nv_bfloat16 sum = bf16_add(bf16_add(Wx[idx], Rh[idx]), b[d]);

        // Store pre-activation
        if (v_cache) v_cache[idx] = sum;

        // softsign (no exp needed!)
        float val = __bfloat162float(sum);
        float h_val = val / (1.0f + fabsf(val));
        h_out[idx] = __float2bfloat16(h_val);

        // Gate (fused - no extra memory read for h)
        float z_val = __bfloat162float(z[idx]);
        float sigmoid_z = 1.0f / (1.0f + __expf(-z_val));
        float silu_z = z_val * sigmoid_z;

        output[idx] = __float2bfloat16(h_val * silu_z);
    }
}

// Generic fused version
template<typename T>
__global__ void FusedSoftsignGateKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ Wx,
    const T* __restrict__ Rh,
    const T* __restrict__ b,
    const T* __restrict__ z,
    T* __restrict__ h_out,
    T* __restrict__ output,
    T* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float val = static_cast<float>(Wx[idx]) + static_cast<float>(Rh[idx]) + static_cast<float>(b[d]);
        if (v_cache) v_cache[idx] = static_cast<T>(val);

        float h_val = val / (1.0f + fabsf(val));
        h_out[idx] = static_cast<T>(h_val);

        float z_val = static_cast<float>(z[idx]);
        float sigmoid_z = 1.0f / (1.0f + expf(-z_val));
        float silu_z = z_val * sigmoid_z;

        output[idx] = static_cast<T>(h_val * silu_z);
    }
}

// =============================================================================
// Optimized Backward Kernels
// =============================================================================

// BF16-optimized backward through softsign
__global__ void SoftsignBackwardKernel_BF16(
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

        // Combine gradients - use native bf16 add if available
        float grad;
        if (dh_recurrent) {
            __nv_bfloat16 combined = bf16_add(dh[idx], dh_recurrent[idx]);
            grad = __bfloat162float(combined);
        } else {
            grad = __bfloat162float(dh[idx]);
        }

        // d/dx softsign(x) = 1 / (1 + |x|)^2
        float val = __bfloat162float(v[idx]);
        float denom = 1.0f + fabsf(val);
        float dsoftsign = 1.0f / (denom * denom);
        float dv_val = grad * dsoftsign;

        dv[idx] = __float2bfloat16(dv_val);
        atomicAdd(&db[d], dv_val);
    }
}

// Generic version
template<typename T>
__global__ void SoftsignBackwardKernel(
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

        // d/dx softsign(x) = 1 / (1 + |x|)^2
        float val = static_cast<float>(v[idx]);
        float denom = 1.0f + fabsf(val);
        float dsoftsign = 1.0f / (denom * denom);
        float dv_val = grad * dsoftsign;
        dv[idx] = static_cast<T>(dv_val);

        atomicAdd(&db[d], dv_val);
    }
}

// BF16-optimized backward through mamba gate
__global__ void MambaGateBackward_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ h,
    const __nv_bfloat16* __restrict__ z,
    const __nv_bfloat16* __restrict__ d_output,
    __nv_bfloat16* __restrict__ dh,
    __nv_bfloat16* __restrict__ dz) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        float h_val = __bfloat162float(h[idx]);
        float z_val = __bfloat162float(z[idx]);
        float dout = __bfloat162float(d_output[idx]);

        float sigmoid_z = 1.0f / (1.0f + __expf(-z_val));
        float silu_z = z_val * sigmoid_z;
        float dsilu = sigmoid_z * (1.0f + z_val * (1.0f - sigmoid_z));

        dh[idx] = __float2bfloat16(dout * silu_z);
        dz[idx] = __float2bfloat16(dout * h_val * dsilu);
    }
}

// Generic version
template<typename T>
__global__ void MambaGateBackward(
    const int batch_size,
    const int dim,
    const T* __restrict__ h,
    const T* __restrict__ z,
    const T* __restrict__ d_output,
    T* __restrict__ dh,
    T* __restrict__ dz) {

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

// Copy float to T
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
// Softsign Elman Forward - BF16 Specialization
// =============================================================================

template<>
SoftsignElmanForward<__nv_bfloat16>::SoftsignElmanForward(
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
void SoftsignElmanForward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_x,
    const __nv_bfloat16* W_h,
    const __nv_bfloat16* b,
    const __nv_bfloat16* x,
    const __nv_bfloat16* z,
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
    __nv_bfloat16* tmp_Rh = workspace + steps * BD;

    // Pre-compute W_x @ x for all timesteps
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha,
        W_x, dim_,
        x, dim_,
        &beta_zero,
        tmp_Wx, dim_);

    // Process each timestep with FUSED kernel
    for (int t = 0; t < steps; ++t) {
        const __nv_bfloat16* Wx_t = tmp_Wx + t * BD;
        const __nv_bfloat16* h_prev = h + t * BD;
        const __nv_bfloat16* z_t = z + t * BD;
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

        // FUSED: h_t = softsign(Wx_t + tmp_Rh + b), output = h_t * silu(z_t)
        FusedSoftsignGateKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, Wx_t, tmp_Rh, b, z_t, h_t, out_t, v_t);
    }
}

// =============================================================================
// Softsign Elman Backward - BF16 Specialization
// =============================================================================

template<>
SoftsignElmanBackward<__nv_bfloat16>::SoftsignElmanBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void SoftsignElmanBackward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_x,
    const __nv_bfloat16* W_h,
    const __nv_bfloat16* x,
    const __nv_bfloat16* z,
    const __nv_bfloat16* h,
    const __nv_bfloat16* v,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* dx,
    __nv_bfloat16* dz,
    __nv_bfloat16* dW_x,
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

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW_x, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_h, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);

    // BPTT loop with BF16-optimized kernels
    for (int t = steps - 1; t >= 0; --t) {
        const __nv_bfloat16* v_t = v + t * BD;
        const __nv_bfloat16* h_t = h + (t + 1) * BD;
        const __nv_bfloat16* z_t = z + t * BD;
        const __nv_bfloat16* d_out_t = d_output + t * BD;
        __nv_bfloat16* dv_t = dv_all + t * BD;
        __nv_bfloat16* dz_t = dz + t * BD;

        // Backward through gate (BF16 optimized)
        MambaGateBackward_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, z_t, d_out_t, dh, dz_t);

        // Add recurrent gradient (BF16 native add)
        VectorAddInplace_BF16<<<num_blocks, block_size, 0, stream_>>>(BD, dh, dh_recurrent);

        // Backward through softsign (BF16 optimized)
        SoftsignBackwardKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
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

    // Batch GEMMs
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_x, dim_,
        dv_all, dim_,
        &beta_zero,
        dx, dim_);

    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        dv_all, dim_,
        &beta_one,
        dW_x, dim_);

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
SoftsignElmanForward<T>::SoftsignElmanForward(
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
void SoftsignElmanForward<T>::Run(
    int steps,
    const T* W_x,
    const T* W_h,
    const T* b,
    const T* x,
    const T* z,
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
    T* tmp_Rh = workspace + steps * BD;

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
        const T* z_t = z + t * BD;
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

        // Use fused kernel for efficiency
        FusedSoftsignGateKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, Wx_t, tmp_Rh, b, z_t, h_t, out_t, v_t);
    }
}

template<typename T>
SoftsignElmanBackward<T>::SoftsignElmanBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void SoftsignElmanBackward<T>::Run(
    int steps,
    const T* W_x,
    const T* W_h,
    const T* x,
    const T* z,
    const T* h,
    const T* v,
    const T* d_output,
    T* dx,
    T* dz,
    T* dW_x,
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
    cudaMemsetAsync(dW_x, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dW_h, 0, dim_ * dim_ * sizeof(T), stream_);

    for (int t = steps - 1; t >= 0; --t) {
        const T* v_t = v + t * BD;
        const T* h_t = h + (t + 1) * BD;
        const T* z_t = z + t * BD;
        const T* d_out_t = d_output + t * BD;
        T* dv_t = dv_all + t * BD;
        T* dz_t = dz + t * BD;

        MambaGateBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, z_t, d_out_t, dh, dz_t);

        VectorAddInplace<T><<<num_blocks, block_size, 0, stream_>>>(BD, dh, dh_recurrent);

        SoftsignBackwardKernel<T><<<num_blocks, block_size, 0, stream_>>>(
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
template struct SoftsignElmanForward<__half>;
template struct SoftsignElmanForward<float>;
template struct SoftsignElmanForward<double>;

template struct SoftsignElmanBackward<__half>;
template struct SoftsignElmanBackward<float>;
template struct SoftsignElmanBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
