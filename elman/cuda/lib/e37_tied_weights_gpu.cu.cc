// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E37: Tied Weights Elman - Single W for both input and hidden
//
// Key difference from E33: W_x = W_h = W (single shared weight matrix)
//
// Architecture:
// h_t = tanh(W @ x_t + W @ h_{t-1} + b)  = tanh(W @ (x_t + h_{t-1}) + b)
// output = h * silu(h)                    # Self-gating from E33
//
// This reduces parameters by half (dim^2 instead of 2*dim^2 for recurrent weights)
// and uses single GEMM per timestep: W @ (x + h_prev) instead of W_x @ x + W_h @ h

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

// =============================================================================
// E37: Add x + h_prev for tied weight GEMM
// =============================================================================

__global__ void VectorAddKernel_BF16(
    const int n,
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ out) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = bf16_add(a[idx], b[idx]);
    }
}

template<typename T>
__global__ void VectorAddKernel(
    const int n,
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ out) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = static_cast<T>(static_cast<float>(a[idx]) + static_cast<float>(b[idx]));
    }
}

// =============================================================================
// FUSED Forward Kernel: bias + tanh + self-gate in one pass
// Input: Wsum = W @ (x + h_prev) already computed via GEMM
// =============================================================================

__global__ void FusedBiasTanhGateKernel_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ Wsum,  // W @ (x + h_prev)
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ h_out,
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // Add bias
        __nv_bfloat16 sum = bf16_add(Wsum[idx], b[d]);

        // Store pre-activation
        if (v_cache) v_cache[idx] = sum;

        // tanh (need f32)
        float val = __bfloat162float(sum);
        float h_val = tanhf(val);
        h_out[idx] = __float2bfloat16(h_val);

        // E37 SELF-GATE (same as E33): output = h * silu(h)
        float sigmoid_h = 1.0f / (1.0f + __expf(-h_val));
        float silu_h = h_val * sigmoid_h;

        output[idx] = __float2bfloat16(h_val * silu_h);
    }
}

template<typename T>
__global__ void FusedBiasTanhGateKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ Wsum,
    const T* __restrict__ b,
    T* __restrict__ h_out,
    T* __restrict__ output,
    T* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float val = static_cast<float>(Wsum[idx]) + static_cast<float>(b[d]);
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
// Backward Kernels
// =============================================================================

// Backward through tanh
__global__ void TanhBackwardKernel_BF16(
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

template<typename T>
__global__ void TanhBackwardKernel(
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

// Self-gate backward (same as E33)
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

        float sigmoid_h = 1.0f / (1.0f + __expf(-h_val));
        float silu_h = h_val * sigmoid_h;
        float grad_factor = silu_h * (2.0f + h_val * (1.0f - sigmoid_h));

        dh[idx] = __float2bfloat16(dout * grad_factor);
    }
}

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
// E37 Tied Weights Forward - BF16 Specialization
// =============================================================================

template<>
E37TiedWeightsForward<__nv_bfloat16>::E37TiedWeightsForward(
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
void E37TiedWeightsForward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W,      // [dim, dim] - SINGLE weight matrix
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

    // Workspace layout: [tmp_sum: BD] [tmp_Wsum: BD]
    __nv_bfloat16* tmp_sum = workspace;      // x_t + h_prev
    __nv_bfloat16* tmp_Wsum = workspace + BD; // W @ (x_t + h_prev)

    // Process each timestep
    for (int t = 0; t < steps; ++t) {
        const __nv_bfloat16* x_t = x + t * BD;
        const __nv_bfloat16* h_prev = h + t * BD;
        __nv_bfloat16* h_t = h + (t + 1) * BD;
        __nv_bfloat16* out_t = output + t * BD;
        __nv_bfloat16* v_t = training_ ? (v + t * BD) : nullptr;

        // E37 KEY: compute x_t + h_prev first
        VectorAddKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            BD, x_t, h_prev, tmp_sum);

        // Single GEMM: W @ (x_t + h_prev)
        blas<__nv_bfloat16>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha,
            W, dim_,
            tmp_sum, dim_,
            &beta_zero,
            tmp_Wsum, dim_);

        // Fused: add bias, tanh, self-gate
        FusedBiasTanhGateKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, tmp_Wsum, b, h_t, out_t, v_t);
    }
}

// =============================================================================
// E37 Tied Weights Backward - BF16 Specialization
// =============================================================================

template<>
E37TiedWeightsBackward<__nv_bfloat16>::E37TiedWeightsBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void E37TiedWeightsBackward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W,
    const __nv_bfloat16* x,
    const __nv_bfloat16* h,
    const __nv_bfloat16* v,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* dx,
    __nv_bfloat16* dW,       // Single weight gradient (accumulates from both paths)
    __nv_bfloat16* db,
    __nv_bfloat16* workspace) {

    static const __nv_bfloat16 alpha_one = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);
    static const __nv_bfloat16 beta_one = __float2bfloat16(1.0f);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Workspace layout: [dv_all: T*BD] [dh: BD] [dh_recurrent: BD] [tmp_sum: BD] [db_float: dim]
    __nv_bfloat16* dv_all = workspace;
    __nv_bfloat16* dh = workspace + steps * BD;
    __nv_bfloat16* dh_recurrent = workspace + (steps + 1) * BD;
    __nv_bfloat16* tmp_sum = workspace + (steps + 2) * BD;  // For x + h_prev
    float* db_float = reinterpret_cast<float*>(workspace + (steps + 3) * BD);

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);

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

        // Backward through tanh
        TanhBackwardKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, v_t, dh, nullptr, dv_t, db_float);

        // dh_recurrent = W @ dv (for next iteration)
        // E37: Same W used for both input and hidden, so gradient flows through same W
        if (t > 0) {
            blas<__nv_bfloat16>::gemm(
                blas_handle_,
                CUBLAS_OP_N, CUBLAS_OP_N,
                dim_, batch_size_, dim_,
                &alpha_one,
                W, dim_,
                dv_t, dim_,
                &beta_zero,
                dh_recurrent, dim_);
        }
    }

    // dx = W @ dv_all (batched across all timesteps)
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W, dim_,
        dv_all, dim_,
        &beta_zero,
        dx, dim_);

    // E37 KEY: dW accumulates from BOTH x and h_prev paths
    // dW = dv @ x.T + dv @ h_prev.T = dv @ (x + h_prev).T
    // We need to compute this per-timestep and accumulate

    for (int t = 0; t < steps; ++t) {
        const __nv_bfloat16* x_t = x + t * BD;
        const __nv_bfloat16* h_prev = h + t * BD;  // h[t] is h_prev for step t
        const __nv_bfloat16* dv_t = dv_all + t * BD;

        // Compute x_t + h_prev
        VectorAddKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            BD, x_t, h_prev, tmp_sum);

        // dW += (x_t + h_prev) @ dv_t.T
        blas<__nv_bfloat16>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_,
            &alpha_one,
            tmp_sum, dim_,
            dv_t, dim_,
            &beta_one,
            dW, dim_);
    }

    // Copy float gradients to bf16
    CopyFloatToT<__nv_bfloat16><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_float, db);
}

// =============================================================================
// Generic Template Implementations (float, half, double)
// =============================================================================

template<typename T>
E37TiedWeightsForward<T>::E37TiedWeightsForward(
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
void E37TiedWeightsForward<T>::Run(
    int steps,
    const T* W,
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

    T* tmp_sum = workspace;
    T* tmp_Wsum = workspace + BD;

    for (int t = 0; t < steps; ++t) {
        const T* x_t = x + t * BD;
        const T* h_prev = h + t * BD;
        T* h_t = h + (t + 1) * BD;
        T* out_t = output + t * BD;
        T* v_t = training_ ? (v + t * BD) : nullptr;

        // x_t + h_prev
        VectorAddKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            BD, x_t, h_prev, tmp_sum);

        // W @ (x_t + h_prev)
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha,
            W, dim_,
            tmp_sum, dim_,
            &beta_zero,
            tmp_Wsum, dim_);

        // bias + tanh + self-gate
        FusedBiasTanhGateKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, tmp_Wsum, b, h_t, out_t, v_t);
    }
}

template<typename T>
E37TiedWeightsBackward<T>::E37TiedWeightsBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void E37TiedWeightsBackward<T>::Run(
    int steps,
    const T* W,
    const T* x,
    const T* h,
    const T* v,
    const T* d_output,
    T* dx,
    T* dW,
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
    T* tmp_sum = workspace + (steps + 2) * BD;
    float* db_float = reinterpret_cast<float*>(workspace + (steps + 3) * BD);

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(T), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW, 0, dim_ * dim_ * sizeof(T), stream_);

    for (int t = steps - 1; t >= 0; --t) {
        const T* v_t = v + t * BD;
        const T* h_t = h + (t + 1) * BD;
        const T* d_out_t = d_output + t * BD;
        T* dv_t = dv_all + t * BD;

        SelfGateBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, d_out_t, dh);

        VectorAddInplace<T><<<num_blocks, block_size, 0, stream_>>>(BD, dh, dh_recurrent);

        TanhBackwardKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, v_t, dh, nullptr, dv_t, db_float);

        if (t > 0) {
            blas<T>::gemm(
                blas_handle_,
                CUBLAS_OP_N, CUBLAS_OP_N,
                dim_, batch_size_, dim_,
                &alpha,
                W, dim_,
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
        W, dim_,
        dv_all, dim_,
        &beta_zero,
        dx, dim_);

    for (int t = 0; t < steps; ++t) {
        const T* x_t = x + t * BD;
        const T* h_prev = h + t * BD;
        const T* dv_t = dv_all + t * BD;

        VectorAddKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            BD, x_t, h_prev, tmp_sum);

        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_,
            &alpha,
            tmp_sum, dim_,
            dv_t, dim_,
            &beta_one,
            dW, dim_);
    }

    CopyFloatToT<T><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_float, db);
}

// Explicit template instantiations
template struct E37TiedWeightsForward<__half>;
template struct E37TiedWeightsForward<float>;
template struct E37TiedWeightsForward<double>;

template struct E37TiedWeightsBackward<__half>;
template struct E37TiedWeightsBackward<float>;
template struct E37TiedWeightsBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
