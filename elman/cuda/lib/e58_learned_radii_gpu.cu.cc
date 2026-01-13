// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E58: Per-Dimension Learned Radii Elman
//
// Key difference from E1: W_h is scaled by per-dimension learned radii
// W_h_scaled[i, j] = W_h[i, j] * radii[i]
// This allows different hidden dimensions to have different memory timescales.
//
// Architecture:
// x, z = split(in_proj(x))           # Pre-computed before kernel
// x = silu(x)                        # Pre-computed before kernel
// h_t = tanh(W_x @ x_t + (W_h * radii.unsqueeze(1)) @ h_{t-1} + b)  # Scaled recurrence
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
// E58 Forward Kernels
// =============================================================================

// BF16-optimized: Fused Wx + radii*Rh + bias + tanh + gate
// radii is a [dim] vector that scales each dimension of Rh
__global__ void E58FusedTanhGateKernel_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ Wx,
    const __nv_bfloat16* __restrict__ Rh,     // W_h @ h_prev (unscaled)
    const __nv_bfloat16* __restrict__ radii,  // [dim] per-dimension scaling
    const __nv_bfloat16* __restrict__ b,
    const __nv_bfloat16* __restrict__ z,
    __nv_bfloat16* __restrict__ h_out,
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // Scale Rh by radii (this is the per-dimension scaling)
        __nv_bfloat16 scaled_Rh = bf16_mul(Rh[idx], radii[d]);

        // Native bf16 additions: Wx + radii*Rh + b
        __nv_bfloat16 sum = bf16_add(bf16_add(Wx[idx], scaled_Rh), b[d]);

        // Store pre-activation
        if (v_cache) v_cache[idx] = sum;

        // tanh (need f32)
        float val = __bfloat162float(sum);
        float h_val = tanhf(val);
        h_out[idx] = __float2bfloat16(h_val);

        // Gate (fused - no extra memory read for h)
        float z_val = __bfloat162float(z[idx]);
        float sigmoid_z = 1.0f / (1.0f + __expf(-z_val));
        float silu_z = z_val * sigmoid_z;

        output[idx] = __float2bfloat16(h_val * silu_z);
    }
}

// Generic version
template<typename T>
__global__ void E58FusedTanhGateKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ Wx,
    const T* __restrict__ Rh,     // W_h @ h_prev (unscaled)
    const T* __restrict__ radii,  // [dim] per-dimension scaling
    const T* __restrict__ b,
    const T* __restrict__ z,
    T* __restrict__ h_out,
    T* __restrict__ output,
    T* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // Scale Rh by radii
        float scaled_Rh = static_cast<float>(Rh[idx]) * static_cast<float>(radii[d]);

        float val = static_cast<float>(Wx[idx]) + scaled_Rh + static_cast<float>(b[d]);
        if (v_cache) v_cache[idx] = static_cast<T>(val);

        float h_val = tanhf(val);
        h_out[idx] = static_cast<T>(h_val);

        float z_val = static_cast<float>(z[idx]);
        float sigmoid_z = 1.0f / (1.0f + expf(-z_val));
        float silu_z = z_val * sigmoid_z;

        output[idx] = static_cast<T>(h_val * silu_z);
    }
}

// =============================================================================
// E58 Backward Kernels
// =============================================================================

// BF16: Backward through mamba gate (same as E1)
__global__ void E58GateBackward_BF16(
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
__global__ void E58GateBackward(
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

// BF16: Backward through tanh with radii scaling
// v = Wx + radii * Rh + b
// dv = dh * (1 - tanh(v)^2)
// dRh = dv * radii[d]  (for backprop through W_h)
// d_radii[d] += sum over batch of (dv * Rh)
// db[d] += sum over batch of dv
__global__ void E58TanhBackward_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ Rh_cache,  // Cached W_h @ h_prev (unscaled)
    const __nv_bfloat16* __restrict__ radii,
    const __nv_bfloat16* __restrict__ dh,
    const __nv_bfloat16* __restrict__ dh_recurrent,
    __nv_bfloat16* __restrict__ dv,
    __nv_bfloat16* __restrict__ dRh,  // Gradient to backprop through W_h
    float* __restrict__ db,
    float* __restrict__ d_radii) {

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

        // dRh = dv * radii[d] (gradient flowing back through the scaling)
        float radii_val = __bfloat162float(radii[d]);
        dRh[idx] = __float2bfloat16(dv_val * radii_val);

        // d_radii = dv * Rh (gradient for the radii parameter)
        float Rh_val = __bfloat162float(Rh_cache[idx]);
        atomicAdd(&d_radii[d], dv_val * Rh_val);
    }
}

// Generic version
template<typename T>
__global__ void E58TanhBackward(
    const int batch_size,
    const int dim,
    const T* __restrict__ v,
    const T* __restrict__ Rh_cache,  // Cached W_h @ h_prev (unscaled)
    const T* __restrict__ radii,
    const T* __restrict__ dh,
    const T* __restrict__ dh_recurrent,
    T* __restrict__ dv,
    T* __restrict__ dRh,  // Gradient to backprop through W_h
    float* __restrict__ db,
    float* __restrict__ d_radii) {

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

        // dRh = dv * radii[d]
        float radii_val = static_cast<float>(radii[d]);
        dRh[idx] = static_cast<T>(dv_val * radii_val);

        // d_radii = dv * Rh
        float Rh_val = static_cast<float>(Rh_cache[idx]);
        atomicAdd(&d_radii[d], dv_val * Rh_val);
    }
}

// BF16: Vector add inplace
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
// E58 Learned Radii Forward - BF16 Specialization
// =============================================================================

template<>
E58LearnedRadiiForward<__nv_bfloat16>::E58LearnedRadiiForward(
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
void E58LearnedRadiiForward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_x,
    const __nv_bfloat16* W_h,
    const __nv_bfloat16* radii,   // [dim] per-dimension scaling
    const __nv_bfloat16* b,
    const __nv_bfloat16* x,
    const __nv_bfloat16* z,
    __nv_bfloat16* h,
    __nv_bfloat16* output,
    __nv_bfloat16* v,
    __nv_bfloat16* Rh_cache,      // [T, B, dim] cache for backward
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
        __nv_bfloat16* Rh_t = training_ ? (Rh_cache + t * BD) : nullptr;

        // tmp_Rh = W_h @ h_prev (unscaled - radii scaling happens in fused kernel)
        blas<__nv_bfloat16>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha,
            W_h, dim_,
            h_prev, dim_,
            &beta_zero,
            tmp_Rh, dim_);

        // Cache Rh for backward pass
        if (Rh_t) {
            cudaMemcpyAsync(Rh_t, tmp_Rh, BD * sizeof(__nv_bfloat16),
                           cudaMemcpyDeviceToDevice, stream_);
        }

        // FUSED: h_t = tanh(Wx_t + radii*tmp_Rh + b), output = h_t * silu(z_t)
        E58FusedTanhGateKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, Wx_t, tmp_Rh, radii, b, z_t, h_t, out_t, v_t);
    }
}

// =============================================================================
// E58 Learned Radii Backward - BF16 Specialization
// =============================================================================

template<>
E58LearnedRadiiBackward<__nv_bfloat16>::E58LearnedRadiiBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void E58LearnedRadiiBackward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_x,
    const __nv_bfloat16* W_h,
    const __nv_bfloat16* radii,
    const __nv_bfloat16* x,
    const __nv_bfloat16* z,
    const __nv_bfloat16* h,
    const __nv_bfloat16* v,
    const __nv_bfloat16* Rh_cache,  // [T, B, dim] cached from forward
    const __nv_bfloat16* d_output,
    __nv_bfloat16* dx,
    __nv_bfloat16* dz,
    __nv_bfloat16* dW_x,
    __nv_bfloat16* dW_h,
    __nv_bfloat16* d_radii,
    __nv_bfloat16* db,
    __nv_bfloat16* workspace) {

    static const __nv_bfloat16 alpha_one = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);
    static const __nv_bfloat16 beta_one = __float2bfloat16(1.0f);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Workspace layout:
    // [dv_all: T*BD] [dRh_all: T*BD] [dh: BD] [dh_recurrent: BD]
    // [db_float: dim] [d_radii_float: dim]
    __nv_bfloat16* dv_all = workspace;
    __nv_bfloat16* dRh_all = workspace + steps * BD;
    __nv_bfloat16* dh = workspace + 2 * steps * BD;
    __nv_bfloat16* dh_recurrent = workspace + (2 * steps + 1) * BD;
    float* db_float = reinterpret_cast<float*>(workspace + (2 * steps + 2) * BD);
    float* d_radii_float = db_float + dim_;

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(d_radii_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW_x, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_h, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);

    // BPTT loop
    for (int t = steps - 1; t >= 0; --t) {
        const __nv_bfloat16* v_t = v + t * BD;
        const __nv_bfloat16* h_t = h + (t + 1) * BD;
        const __nv_bfloat16* Rh_t = Rh_cache + t * BD;
        const __nv_bfloat16* z_t = z + t * BD;
        const __nv_bfloat16* d_out_t = d_output + t * BD;
        __nv_bfloat16* dv_t = dv_all + t * BD;
        __nv_bfloat16* dRh_t = dRh_all + t * BD;
        __nv_bfloat16* dz_t = dz + t * BD;

        // Backward through gate
        E58GateBackward_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, z_t, d_out_t, dh, dz_t);

        // Add recurrent gradient to dh (already computed from previous timestep)
        VectorAddInplace_BF16<<<num_blocks, block_size, 0, stream_>>>(BD, dh, dh_recurrent);

        // Backward through tanh with radii scaling
        // Pass nullptr for dh_recurrent since we already added it to dh above
        E58TanhBackward_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, v_t, Rh_t, radii, dh,
            nullptr,  // Already added to dh
            dv_t, dRh_t, db_float, d_radii_float);

        // dh_recurrent = W_h @ dRh (note: dRh already has radii scaling)
        if (t > 0) {
            blas<__nv_bfloat16>::gemm(
                blas_handle_,
                CUBLAS_OP_N, CUBLAS_OP_N,
                dim_, batch_size_, dim_,
                &alpha_one,
                W_h, dim_,
                dRh_t, dim_,
                &beta_zero,
                dh_recurrent, dim_);
        }
    }

    // Batch GEMMs for dx = W_x @ dv
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_x, dim_,
        dv_all, dim_,
        &beta_zero,
        dx, dim_);

    // dW_x = x.T @ dv
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        dv_all, dim_,
        &beta_one,
        dW_x, dim_);

    // dW_h = h.T @ dRh (using h[0:T] and dRh_all)
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        h, dim_,  // h[0:T*BD] = h_prev for each timestep
        dRh_all, dim_,
        &beta_one,
        dW_h, dim_);

    // Copy float gradients to bf16
    CopyFloatToT<__nv_bfloat16><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_float, db);
    CopyFloatToT<__nv_bfloat16><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, d_radii_float, d_radii);
}

// =============================================================================
// Generic Template Implementations (float, half, double)
// =============================================================================

template<typename T>
E58LearnedRadiiForward<T>::E58LearnedRadiiForward(
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
void E58LearnedRadiiForward<T>::Run(
    int steps,
    const T* W_x,
    const T* W_h,
    const T* radii,
    const T* b,
    const T* x,
    const T* z,
    T* h,
    T* output,
    T* v,
    T* Rh_cache,
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
        T* Rh_t = training_ ? (Rh_cache + t * BD) : nullptr;

        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha,
            W_h, dim_,
            h_prev, dim_,
            &beta_zero,
            tmp_Rh, dim_);

        // Cache Rh for backward
        if (Rh_t) {
            cudaMemcpyAsync(Rh_t, tmp_Rh, BD * sizeof(T),
                           cudaMemcpyDeviceToDevice, stream_);
        }

        // Fused kernel
        E58FusedTanhGateKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, Wx_t, tmp_Rh, radii, b, z_t, h_t, out_t, v_t);
    }
}

template<typename T>
E58LearnedRadiiBackward<T>::E58LearnedRadiiBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void E58LearnedRadiiBackward<T>::Run(
    int steps,
    const T* W_x,
    const T* W_h,
    const T* radii,
    const T* x,
    const T* z,
    const T* h,
    const T* v,
    const T* Rh_cache,
    const T* d_output,
    T* dx,
    T* dz,
    T* dW_x,
    T* dW_h,
    T* d_radii,
    T* db,
    T* workspace) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    T* dv_all = workspace;
    T* dRh_all = workspace + steps * BD;
    T* dh = workspace + 2 * steps * BD;
    T* dh_recurrent = workspace + (2 * steps + 1) * BD;
    float* db_float = reinterpret_cast<float*>(workspace + (2 * steps + 2) * BD);
    float* d_radii_float = db_float + dim_;

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(T), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(d_radii_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW_x, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dW_h, 0, dim_ * dim_ * sizeof(T), stream_);

    for (int t = steps - 1; t >= 0; --t) {
        const T* v_t = v + t * BD;
        const T* h_t = h + (t + 1) * BD;
        const T* Rh_t = Rh_cache + t * BD;
        const T* z_t = z + t * BD;
        const T* d_out_t = d_output + t * BD;
        T* dv_t = dv_all + t * BD;
        T* dRh_t = dRh_all + t * BD;
        T* dz_t = dz + t * BD;

        E58GateBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, z_t, d_out_t, dh, dz_t);

        // Add recurrent gradient to dh
        VectorAddInplace<T><<<num_blocks, block_size, 0, stream_>>>(BD, dh, dh_recurrent);

        // Pass nullptr since we already added dh_recurrent to dh
        E58TanhBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, v_t, Rh_t, radii, dh,
            nullptr,  // Already added to dh
            dv_t, dRh_t, db_float, d_radii_float);

        if (t > 0) {
            blas<T>::gemm(
                blas_handle_,
                CUBLAS_OP_N, CUBLAS_OP_N,
                dim_, batch_size_, dim_,
                &alpha,
                W_h, dim_,
                dRh_t, dim_,
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
        dRh_all, dim_,
        &beta_one,
        dW_h, dim_);

    CopyFloatToT<T><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_float, db);
    CopyFloatToT<T><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, d_radii_float, d_radii);
}

// Explicit template instantiations
template struct E58LearnedRadiiForward<__half>;
template struct E58LearnedRadiiForward<float>;
template struct E58LearnedRadiiForward<double>;

template struct E58LearnedRadiiBackward<__half>;
template struct E58LearnedRadiiBackward<float>;
template struct E58LearnedRadiiBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
