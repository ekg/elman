// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E59: Highway Elman - Residual Recurrence with Perfect Gradient Flow
//
// The temporal analog of ResNet: just as residual connections revolutionized
// depth in feedforward networks, temporal skip connections revolutionize
// sequence length in recurrent networks.
//
// Architecture:
//     h_t = h_{t-1} + alpha * (W @ x_t + b)   # Residual accumulation (gradient = I)
//     output_t = h_t * silu(h_t)              # Nonlinearity at output only
//
// Where alpha = exp(log_alpha) is a learned positive scalar.
//
// Key insight: The Jacobian dh_t/dh_{t-1} = I (identity), providing
// perfect gradient preservation through time - no vanishing, no exploding.

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
// E59 Forward Kernel: Residual accumulation + self-gate
// h_new = h_prev + alpha * (Wx + b)
// output = h * silu(h)
// =============================================================================

__global__ void E59ResidualGateKernel_BF16(
    const int batch_size,
    const int dim,
    const float alpha,
    const __nv_bfloat16* __restrict__ Wx,       // [B, dim] W @ x_t + b (pre-computed)
    const __nv_bfloat16* __restrict__ h_prev,
    __nv_bfloat16* __restrict__ h_out,
    __nv_bfloat16* __restrict__ output) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        // E59: Residual accumulation h_new = h_prev + alpha * Wx
        float h_prev_f = __bfloat162float(h_prev[idx]);
        float Wx_f = __bfloat162float(Wx[idx]);
        float h_val = h_prev_f + alpha * Wx_f;
        h_out[idx] = __float2bfloat16(h_val);

        // Self-gate: output = h * silu(h)
        float sigmoid_h = 1.0f / (1.0f + __expf(-h_val));
        float silu_h = h_val * sigmoid_h;
        output[idx] = __float2bfloat16(h_val * silu_h);
    }
}

template<typename T>
__global__ void E59ResidualGateKernel(
    const int batch_size,
    const int dim,
    const float alpha,
    const T* __restrict__ Wx,
    const T* __restrict__ h_prev,
    T* __restrict__ h_out,
    T* __restrict__ output) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        // E59: Residual accumulation
        float h_val = static_cast<float>(h_prev[idx]) + alpha * static_cast<float>(Wx[idx]);
        h_out[idx] = static_cast<T>(h_val);

        // Self-gate: output = h * silu(h)
        float sigmoid_h = 1.0f / (1.0f + expf(-h_val));
        float silu_h = h_val * sigmoid_h;
        output[idx] = static_cast<T>(h_val * silu_h);
    }
}

// =============================================================================
// Backward Kernels
// =============================================================================

// Self-gate backward (same as E42/E45)
// output = h * silu(h) = h^2 * sigmoid(h)
// d(output)/dh = silu(h) * (2 + h*(1-sigmoid(h)))
__global__ void SelfGateBackward_E59_BF16(
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
__global__ void SelfGateBackward_E59(
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

// E59 Backward Kernel: Compute gradients through residual connection
// For E59: dh_t/dx_t = alpha (via W), dh_t/dh_{t-1} = 1 (perfect gradient flow)
// d_log_alpha contribution: d_log_alpha += sum(dh * (W @ x + b)) * alpha
__global__ void E59BackwardKernel_BF16(
    const int batch_size,
    const int dim,
    const float alpha,
    const __nv_bfloat16* __restrict__ Wx,       // W @ x_t + b (saved from forward)
    const __nv_bfloat16* __restrict__ dh,
    const __nv_bfloat16* __restrict__ dh_recurrent,
    __nv_bfloat16* __restrict__ dWx,            // Output: gradient w.r.t. Wx (to be backpropped through W)
    __nv_bfloat16* __restrict__ dh_recurrent_prev,
    float* __restrict__ d_log_alpha) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        // Combine gradients from output and recurrence
        float grad = __bfloat162float(dh[idx]);
        if (dh_recurrent) grad += __bfloat162float(dh_recurrent[idx]);

        // dWx = alpha * dh (gradient w.r.t. Wx = W @ x + b)
        dWx[idx] = __float2bfloat16(alpha * grad);

        // dh_recurrent_prev = dh (identity Jacobian - perfect gradient flow!)
        if (dh_recurrent_prev) dh_recurrent_prev[idx] = __float2bfloat16(grad);

        // d_log_alpha contribution: dL/d_log_alpha = sum(dh * Wx) * alpha
        // Because d(alpha)/d(log_alpha) = alpha (for alpha = exp(log_alpha))
        float Wx_val = __bfloat162float(Wx[idx]);
        atomicAdd(d_log_alpha, grad * Wx_val * alpha);
    }
}

template<typename T>
__global__ void E59BackwardKernel(
    const int batch_size,
    const int dim,
    const float alpha,
    const T* __restrict__ Wx,
    const T* __restrict__ dh,
    const T* __restrict__ dh_recurrent,
    T* __restrict__ dWx,
    T* __restrict__ dh_recurrent_prev,
    float* __restrict__ d_log_alpha) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        float grad = static_cast<float>(dh[idx]);
        if (dh_recurrent) grad += static_cast<float>(dh_recurrent[idx]);

        // dWx = alpha * dh
        dWx[idx] = static_cast<T>(alpha * grad);

        // dh_recurrent_prev = dh (identity Jacobian)
        if (dh_recurrent_prev) dh_recurrent_prev[idx] = static_cast<T>(grad);

        // d_log_alpha contribution
        float Wx_val = static_cast<float>(Wx[idx]);
        atomicAdd(d_log_alpha, grad * Wx_val * alpha);
    }
}

// Utility kernel: Add bias to Wx
__global__ void AddBiasKernel_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ Wx) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;
#if __CUDA_ARCH__ >= 800
        Wx[idx] = __hadd(Wx[idx], b[d]);
#else
        Wx[idx] = __float2bfloat16(__bfloat162float(Wx[idx]) + __bfloat162float(b[d]));
#endif
    }
}

template<typename T>
__global__ void AddBiasKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ b,
    T* __restrict__ Wx) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;
        Wx[idx] = static_cast<T>(static_cast<float>(Wx[idx]) + static_cast<float>(b[d]));
    }
}

// Bias gradient kernel
__global__ void BiasGradKernel_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ dWx,
    float* __restrict__ db) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;
        atomicAdd(&db[d], __bfloat162float(dWx[idx]));
    }
}

template<typename T>
__global__ void BiasGradKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ dWx,
    float* __restrict__ db) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;
        atomicAdd(&db[d], static_cast<float>(dWx[idx]));
    }
}

template<typename T>
__global__ void CopyFloatToT_E59(const int n, const float* __restrict__ src, T* __restrict__ dst) {
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
// E59 Highway Forward - BF16 Specialization
// =============================================================================

template<>
E59HighwayForward<__nv_bfloat16>::E59HighwayForward(
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
void E59HighwayForward<__nv_bfloat16>::Run(
    int steps,
    const float alpha,
    const __nv_bfloat16* W,      // [dim, dim]
    const __nv_bfloat16* b,      // [dim]
    const __nv_bfloat16* x,      // [T, B, dim]
    __nv_bfloat16* h,            // [T+1, B, dim]
    __nv_bfloat16* output,       // [T, B, dim]
    __nv_bfloat16* Wx_cache,     // [T, B, dim] cache of W@x+b for backward
    __nv_bfloat16* workspace) {  // [T*BD + BD] for Wx_all, tmp

    static const __nv_bfloat16 alpha_one = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Workspace layout: [Wx_all: T*BD]
    __nv_bfloat16* Wx_all = workspace;

    // =========================================================================
    // KEY OPTIMIZATION: Pre-compute W @ x for ALL timesteps in one batched GEMM
    // =========================================================================
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W, dim_,
        x, dim_,
        &beta_zero,
        Wx_all, dim_);

    // Add bias to all Wx values
    const int total_add = steps * BD;
    const int add_blocks = (total_add + block_size - 1) / block_size;
    for (int t = 0; t < steps; ++t) {
        AddBiasKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, b, Wx_all + t * BD);
    }

    // Copy Wx_all to cache if training
    if (training_ && Wx_cache) {
        cudaMemcpyAsync(Wx_cache, Wx_all, steps * BD * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToDevice, stream_);
    }

    // Process each timestep with residual accumulation
    for (int t = 0; t < steps; ++t) {
        const __nv_bfloat16* Wx_t = Wx_all + t * BD;
        const __nv_bfloat16* h_prev = h + t * BD;
        __nv_bfloat16* h_t = h + (t + 1) * BD;
        __nv_bfloat16* out_t = output + t * BD;

        // E59: h_new = h_prev + alpha * (W @ x + b), output = h_new * silu(h_new)
        E59ResidualGateKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, alpha, Wx_t, h_prev, h_t, out_t);
    }
}

// =============================================================================
// E59 Highway Backward - BF16 Specialization
// =============================================================================

template<>
E59HighwayBackward<__nv_bfloat16>::E59HighwayBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void E59HighwayBackward<__nv_bfloat16>::Run(
    int steps,
    const float alpha,
    const __nv_bfloat16* W,
    const __nv_bfloat16* x,
    const __nv_bfloat16* h,
    const __nv_bfloat16* Wx_cache,    // [T, B, dim] W@x+b from forward
    const __nv_bfloat16* d_output,
    __nv_bfloat16* dx,
    __nv_bfloat16* dW,
    __nv_bfloat16* db,
    float* d_log_alpha,
    __nv_bfloat16* workspace) {

    static const __nv_bfloat16 alpha_one = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);
    static const __nv_bfloat16 beta_one = __float2bfloat16(1.0f);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Workspace layout: [dWx_all: T*BD] [dh: BD] [dh_recurrent: BD] [db_float: dim]
    __nv_bfloat16* dWx_all = workspace;
    __nv_bfloat16* dh = workspace + steps * BD;
    __nv_bfloat16* dh_recurrent = workspace + (steps + 1) * BD;
    float* db_float = reinterpret_cast<float*>(workspace + (steps + 2) * BD);

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(d_log_alpha, 0, sizeof(float), stream_);

    // BPTT loop
    for (int t = steps - 1; t >= 0; --t) {
        const __nv_bfloat16* h_t = h + (t + 1) * BD;
        const __nv_bfloat16* Wx_t = Wx_cache + t * BD;
        const __nv_bfloat16* d_out_t = d_output + t * BD;
        __nv_bfloat16* dWx_t = dWx_all + t * BD;

        // Backward through self-gate: d_output -> dh
        SelfGateBackward_E59_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, d_out_t, dh);

        // E59 backward: compute dWx, accumulate d_log_alpha, propagate dh_recurrent
        E59BackwardKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, alpha, Wx_t, dh, dh_recurrent, dWx_t,
            (t > 0) ? dh_recurrent : nullptr, d_log_alpha);

        // Accumulate bias gradient
        BiasGradKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, dWx_t, db_float);
    }

    // =========================================================================
    // Batched GEMM for dx: dx = W^T @ dWx_all
    // =========================================================================
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W, dim_,
        dWx_all, dim_,
        &beta_zero,
        dx, dim_);

    // =========================================================================
    // Batched GEMM for dW: dW = x @ dWx_all^T
    // =========================================================================
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        dWx_all, dim_,
        &beta_one,
        dW, dim_);

    // Copy float gradients to bf16
    CopyFloatToT_E59<__nv_bfloat16><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_float, db);
}

// =============================================================================
// Generic Template Implementations (float, half, double)
// =============================================================================

template<typename T>
E59HighwayForward<T>::E59HighwayForward(
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
void E59HighwayForward<T>::Run(
    int steps,
    const float alpha,
    const T* W,
    const T* b,
    const T* x,
    T* h,
    T* output,
    T* Wx_cache,
    T* workspace) {

    static const T alpha_one = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    T* Wx_all = workspace;

    // Batch GEMM for W @ x across all timesteps
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W, dim_,
        x, dim_,
        &beta_zero,
        Wx_all, dim_);

    // Add bias
    for (int t = 0; t < steps; ++t) {
        AddBiasKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, b, Wx_all + t * BD);
    }

    // Copy to cache if training
    if (training_ && Wx_cache) {
        cudaMemcpyAsync(Wx_cache, Wx_all, steps * BD * sizeof(T),
                        cudaMemcpyDeviceToDevice, stream_);
    }

    for (int t = 0; t < steps; ++t) {
        const T* Wx_t = Wx_all + t * BD;
        const T* h_prev = h + t * BD;
        T* h_t = h + (t + 1) * BD;
        T* out_t = output + t * BD;

        E59ResidualGateKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, alpha, Wx_t, h_prev, h_t, out_t);
    }
}

template<typename T>
E59HighwayBackward<T>::E59HighwayBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void E59HighwayBackward<T>::Run(
    int steps,
    const float alpha,
    const T* W,
    const T* x,
    const T* h,
    const T* Wx_cache,
    const T* d_output,
    T* dx,
    T* dW,
    T* db,
    float* d_log_alpha,
    T* workspace) {

    static const T alpha_one = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    T* dWx_all = workspace;
    T* dh = workspace + steps * BD;
    T* dh_recurrent = workspace + (steps + 1) * BD;
    float* db_float = reinterpret_cast<float*>(workspace + (steps + 2) * BD);

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(T), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(d_log_alpha, 0, sizeof(float), stream_);

    for (int t = steps - 1; t >= 0; --t) {
        const T* h_t = h + (t + 1) * BD;
        const T* Wx_t = Wx_cache + t * BD;
        const T* d_out_t = d_output + t * BD;
        T* dWx_t = dWx_all + t * BD;

        SelfGateBackward_E59<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, d_out_t, dh);

        E59BackwardKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, alpha, Wx_t, dh, dh_recurrent, dWx_t,
            (t > 0) ? dh_recurrent : nullptr, d_log_alpha);

        BiasGradKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, dWx_t, db_float);
    }

    // Batched GEMM for dx
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W, dim_,
        dWx_all, dim_,
        &beta_zero,
        dx, dim_);

    // Batched GEMM for dW
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        dWx_all, dim_,
        &beta_one,
        dW, dim_);

    CopyFloatToT_E59<T><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_float, db);
}

// Explicit template instantiations
template struct E59HighwayForward<__half>;
template struct E59HighwayForward<float>;
template struct E59HighwayForward<double>;

template struct E59HighwayBackward<__half>;
template struct E59HighwayBackward<float>;
template struct E59HighwayBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
