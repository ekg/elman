// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E59: Highway Elman - RMSNorm-Bounded Residual Recurrence
//
// The temporal analog of ResNet with RMSNorm to bound hidden state.
//
// Architecture:
//     h_t = RMSNorm(h_{t-1} + alpha * (W @ x_t + b))  # Bounded residual
//     output_t = h_t * silu(h_t)                      # Nonlinearity at output
//
// Where alpha = exp(log_alpha) is a learned positive scalar.
//
// Key insight: RMSNorm bounds h to unit RMS, preventing explosion while
// preserving gradient direction through the residual pathway.

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

constexpr float RMSNORM_EPS = 1e-6f;

// =============================================================================
// E59 Forward Kernels: Residual + RMSNorm + Self-gate
// =============================================================================

// Phase 1: Residual accumulation (h_raw = h_prev + alpha * Wx)
// Writes to h_raw_cache for backward, then will be normalized in place
__global__ void E59ResidualKernel_BF16(
    const int batch_size,
    const int dim,
    const float alpha,
    const __nv_bfloat16* __restrict__ Wx,
    const __nv_bfloat16* __restrict__ h_prev,
    __nv_bfloat16* __restrict__ h_out) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        float h_prev_f = __bfloat162float(h_prev[idx]);
        float Wx_f = __bfloat162float(Wx[idx]);
        h_out[idx] = __float2bfloat16(h_prev_f + alpha * Wx_f);
    }
}

// Phase 2: RMSNorm (h = h / sqrt(mean(h^2) + eps))
// One block per batch element, uses shared memory for reduction
__global__ void E59RMSNormKernel_BF16(
    const int batch_size,
    const int dim,
    __nv_bfloat16* __restrict__ h) {

    extern __shared__ float sdata[];

    const int b = blockIdx.x;  // batch index
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    // Compute sum of squares for this batch element
    float sum_sq = 0.0f;
    for (int d = tid; d < dim; d += stride) {
        float val = __bfloat162float(h[b * dim + d]);
        sum_sq += val * val;
    }

    // Reduce within block
    sdata[tid] = sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Compute RMS and normalize
    float rms = sqrtf(sdata[0] / dim + RMSNORM_EPS);
    float inv_rms = 1.0f / rms;

    for (int d = tid; d < dim; d += stride) {
        float val = __bfloat162float(h[b * dim + d]);
        h[b * dim + d] = __float2bfloat16(val * inv_rms);
    }
}

// Phase 3: Self-gate (output = h * silu(h))
__global__ void E59SelfGateKernel_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ h,
    __nv_bfloat16* __restrict__ output) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        float h_val = __bfloat162float(h[idx]);
        float sigmoid_h = 1.0f / (1.0f + __expf(-h_val));
        float silu_h = h_val * sigmoid_h;
        output[idx] = __float2bfloat16(h_val * silu_h);
    }
}

// Generic template versions
template<typename T>
__global__ void E59ResidualKernel(
    const int batch_size,
    const int dim,
    const float alpha,
    const T* __restrict__ Wx,
    const T* __restrict__ h_prev,
    T* __restrict__ h_out) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        h_out[idx] = static_cast<T>(
            static_cast<float>(h_prev[idx]) + alpha * static_cast<float>(Wx[idx]));
    }
}

template<typename T>
__global__ void E59RMSNormKernel(
    const int batch_size,
    const int dim,
    T* __restrict__ h) {

    extern __shared__ float sdata[];

    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    float sum_sq = 0.0f;
    for (int d = tid; d < dim; d += stride) {
        float val = static_cast<float>(h[b * dim + d]);
        sum_sq += val * val;
    }

    sdata[tid] = sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    float rms = sqrtf(sdata[0] / dim + RMSNORM_EPS);
    float inv_rms = 1.0f / rms;

    for (int d = tid; d < dim; d += stride) {
        float val = static_cast<float>(h[b * dim + d]);
        h[b * dim + d] = static_cast<T>(val * inv_rms);
    }
}

template<typename T>
__global__ void E59SelfGateKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ h,
    T* __restrict__ output) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        float h_val = static_cast<float>(h[idx]);
        float sigmoid_h = 1.0f / (1.0f + expf(-h_val));
        float silu_h = h_val * sigmoid_h;
        output[idx] = static_cast<T>(h_val * silu_h);
    }
}

// Legacy combined kernel (kept for backward compatibility, but unused)
__global__ void E59ResidualGateKernel_BF16(
    const int batch_size,
    const int dim,
    const float alpha,
    const __nv_bfloat16* __restrict__ Wx,
    const __nv_bfloat16* __restrict__ h_prev,
    __nv_bfloat16* __restrict__ h_out,
    __nv_bfloat16* __restrict__ output) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        float h_prev_f = __bfloat162float(h_prev[idx]);
        float Wx_f = __bfloat162float(Wx[idx]);
        float h_val = h_prev_f + alpha * Wx_f;
        h_out[idx] = __float2bfloat16(h_val);

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
        float h_val = static_cast<float>(h_prev[idx]) + alpha * static_cast<float>(Wx[idx]);
        h_out[idx] = static_cast<T>(h_val);

        float sigmoid_h = 1.0f / (1.0f + expf(-h_val));
        float silu_h = h_val * sigmoid_h;
        output[idx] = static_cast<T>(h_val * silu_h);
    }
}

// =============================================================================
// Backward Kernels
// =============================================================================

// RMSNorm backward kernel
// Forward: h = h_raw / rms, where rms = sqrt(mean(h_raw^2) + eps)
// Backward: dh_raw = (dh - mean(dh * h) * h) / rms
// One block per batch element, uses shared memory for reduction
__global__ void E59RMSNormBackwardKernel_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ h_raw,  // Pre-normalized hidden state
    const __nv_bfloat16* __restrict__ h,       // Normalized hidden state (= h_raw / rms)
    const __nv_bfloat16* __restrict__ dh,      // Gradient w.r.t. normalized h
    __nv_bfloat16* __restrict__ dh_raw) {      // Output: gradient w.r.t. h_raw

    extern __shared__ float sdata[];

    const int b = blockIdx.x;  // batch index
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    // Step 1: Compute sum of h_raw^2 for RMS
    float sum_sq = 0.0f;
    for (int d = tid; d < dim; d += stride) {
        float val = __bfloat162float(h_raw[b * dim + d]);
        sum_sq += val * val;
    }

    // Reduce within block for sum_sq
    sdata[tid] = sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    float rms = sqrtf(sdata[0] / dim + RMSNORM_EPS);
    float inv_rms = 1.0f / rms;

    // Step 2: Compute mean(dh * h) for the correction term
    float dot_dh_h = 0.0f;
    for (int d = tid; d < dim; d += stride) {
        float dh_val = __bfloat162float(dh[b * dim + d]);
        float h_val = __bfloat162float(h[b * dim + d]);
        dot_dh_h += dh_val * h_val;
    }

    // Reduce within block for dot product
    sdata[tid] = dot_dh_h;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    float mean_dot = sdata[0] / dim;

    // Step 3: Compute dh_raw = (dh - mean_dot * h) / rms
    for (int d = tid; d < dim; d += stride) {
        float dh_val = __bfloat162float(dh[b * dim + d]);
        float h_val = __bfloat162float(h[b * dim + d]);
        float dh_raw_val = (dh_val - mean_dot * h_val) * inv_rms;
        dh_raw[b * dim + d] = __float2bfloat16(dh_raw_val);
    }
}

template<typename T>
__global__ void E59RMSNormBackwardKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ h_raw,
    const T* __restrict__ h,
    const T* __restrict__ dh,
    T* __restrict__ dh_raw) {

    extern __shared__ float sdata[];

    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    // Step 1: Compute RMS
    float sum_sq = 0.0f;
    for (int d = tid; d < dim; d += stride) {
        float val = static_cast<float>(h_raw[b * dim + d]);
        sum_sq += val * val;
    }

    sdata[tid] = sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    float rms = sqrtf(sdata[0] / dim + RMSNORM_EPS);
    float inv_rms = 1.0f / rms;

    // Step 2: Compute mean(dh * h)
    float dot_dh_h = 0.0f;
    for (int d = tid; d < dim; d += stride) {
        float dh_val = static_cast<float>(dh[b * dim + d]);
        float h_val = static_cast<float>(h[b * dim + d]);
        dot_dh_h += dh_val * h_val;
    }

    sdata[tid] = dot_dh_h;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    float mean_dot = sdata[0] / dim;

    // Step 3: Compute dh_raw
    for (int d = tid; d < dim; d += stride) {
        float dh_val = static_cast<float>(dh[b * dim + d]);
        float h_val = static_cast<float>(h[b * dim + d]);
        float dh_raw_val = (dh_val - mean_dot * h_val) * inv_rms;
        dh_raw[b * dim + d] = static_cast<T>(dh_raw_val);
    }
}

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

// Add two tensors: dst = a + b
__global__ void AddKernel_E59_BF16(
    const int total,
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        dst[idx] = __float2bfloat16(__bfloat162float(a[idx]) + __bfloat162float(b[idx]));
    }
}

template<typename T>
__global__ void AddKernel_E59(
    const int total,
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        dst[idx] = static_cast<T>(static_cast<float>(a[idx]) + static_cast<float>(b[idx]));
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

    // RMSNorm config: one block per batch element
    const int rmsnorm_threads = 256;
    const size_t rmsnorm_smem = rmsnorm_threads * sizeof(float);

    // Process each timestep: residual -> rmsnorm -> selfgate
    for (int t = 0; t < steps; ++t) {
        const __nv_bfloat16* Wx_t = Wx_all + t * BD;
        const __nv_bfloat16* h_prev = h + t * BD;
        __nv_bfloat16* h_t = h + (t + 1) * BD;
        __nv_bfloat16* out_t = output + t * BD;

        // Phase 1: Residual accumulation (h_new = h_prev + alpha * Wx)
        E59ResidualKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, alpha, Wx_t, h_prev, h_t);

        // Phase 2: RMSNorm (h = h / sqrt(mean(h^2) + eps))
        E59RMSNormKernel_BF16<<<batch_size_, rmsnorm_threads, rmsnorm_smem, stream_>>>(
            batch_size_, dim_, h_t);

        // Phase 3: Self-gate (output = h * silu(h))
        E59SelfGateKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, out_t);
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

    // RMSNorm backward config
    const int rmsnorm_threads = 256;
    const size_t rmsnorm_smem = rmsnorm_threads * sizeof(float);

    // Workspace layout: [dWx_all: T*BD] [dh: BD] [dh_raw: BD] [dh_recurrent: BD] [h_raw: BD] [db_float: dim]
    __nv_bfloat16* dWx_all = workspace;
    __nv_bfloat16* dh = workspace + steps * BD;
    __nv_bfloat16* dh_raw = workspace + (steps + 1) * BD;
    __nv_bfloat16* dh_recurrent = workspace + (steps + 2) * BD;
    __nv_bfloat16* h_raw = workspace + (steps + 3) * BD;
    float* db_float = reinterpret_cast<float*>(workspace + (steps + 4) * BD);

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(d_log_alpha, 0, sizeof(float), stream_);

    // BPTT loop with RMSNorm backward
    //
    // Forward pass per timestep:
    //   h_raw[t] = h[t-1] + alpha * Wx[t]       (residual accumulation)
    //   h[t] = h_raw[t] / rms(h_raw[t])         (RMSNorm)
    //   output[t] = h[t] * silu(h[t])           (self-gate)
    //
    // Backward pass per timestep (from t=T-1 to 0):
    //   dh[t] = d(output)/d(h) * d_output[t]   (self-gate backward)
    //   dh_total[t] = dh[t] + dh_recurrent[t]  (combine output and recurrent gradients)
    //   dh_raw[t] = RMSNorm_backward(dh_total[t], h_raw[t], h[t])
    //   dh_recurrent[t-1] = dh_raw[t]          (identity Jacobian for residual!)
    //   dWx[t] = alpha * dh_raw[t]
    //   d_log_alpha += sum(dh_raw[t] * Wx[t]) * alpha
    //
    for (int t = steps - 1; t >= 0; --t) {
        const __nv_bfloat16* h_t = h + (t + 1) * BD;        // Normalized h at timestep t+1
        const __nv_bfloat16* h_prev = h + t * BD;           // Normalized h at timestep t
        const __nv_bfloat16* Wx_t = Wx_cache + t * BD;
        const __nv_bfloat16* d_out_t = d_output + t * BD;
        __nv_bfloat16* dWx_t = dWx_all + t * BD;

        // Step 1: Backward through self-gate: d_output -> dh (from output path)
        SelfGateBackward_E59_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, d_out_t, dh);

        // Step 2: Combine output gradient with recurrent gradient
        // dh_total = dh + dh_recurrent (for t < steps-1)
        if (t < steps - 1) {
            AddKernel_E59_BF16<<<num_blocks, block_size, 0, stream_>>>(
                BD, dh, dh_recurrent, dh);
        }

        // Step 3: Recompute h_raw = h_prev + alpha * Wx_t
        E59ResidualKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, alpha, Wx_t, h_prev, h_raw);

        // Step 4: RMSNorm backward: dh_total -> dh_raw
        // dh_raw = (dh - mean(dh * h) * h) / rms
        E59RMSNormBackwardKernel_BF16<<<batch_size_, rmsnorm_threads, rmsnorm_smem, stream_>>>(
            batch_size_, dim_, h_raw, h_t, dh, dh_raw);

        // Step 5: Residual backward: dh_raw -> dWx, dh_recurrent (for prev timestep)
        // Since h_raw = h_prev + alpha * Wx:
        //   dh_prev = dh_raw (identity Jacobian - perfect gradient flow!)
        //   dWx = alpha * dh_raw
        //   d_log_alpha += sum(dh_raw * Wx) * alpha
        E59BackwardKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, alpha, Wx_t, dh_raw, nullptr, dWx_t,
            (t > 0) ? dh_recurrent : nullptr, d_log_alpha);

        // Step 6: Accumulate bias gradient
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

    // RMSNorm config
    const int rmsnorm_threads = 256;
    const size_t rmsnorm_smem = rmsnorm_threads * sizeof(float);

    // Process each timestep: residual -> rmsnorm -> selfgate
    for (int t = 0; t < steps; ++t) {
        const T* Wx_t = Wx_all + t * BD;
        const T* h_prev = h + t * BD;
        T* h_t = h + (t + 1) * BD;
        T* out_t = output + t * BD;

        // Phase 1: Residual
        E59ResidualKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, alpha, Wx_t, h_prev, h_t);

        // Phase 2: RMSNorm
        E59RMSNormKernel<T><<<batch_size_, rmsnorm_threads, rmsnorm_smem, stream_>>>(
            batch_size_, dim_, h_t);

        // Phase 3: Self-gate
        E59SelfGateKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, out_t);
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

    // RMSNorm backward config
    const int rmsnorm_threads = 256;
    const size_t rmsnorm_smem = rmsnorm_threads * sizeof(float);

    // Workspace layout: [dWx_all: T*BD] [dh: BD] [dh_raw: BD] [dh_recurrent: BD] [h_raw: BD] [db_float: dim]
    T* dWx_all = workspace;
    T* dh = workspace + steps * BD;
    T* dh_raw = workspace + (steps + 1) * BD;
    T* dh_recurrent = workspace + (steps + 2) * BD;
    T* h_raw = workspace + (steps + 3) * BD;
    float* db_float = reinterpret_cast<float*>(workspace + (steps + 4) * BD);

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(T), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(d_log_alpha, 0, sizeof(float), stream_);

    // BPTT loop with RMSNorm backward
    for (int t = steps - 1; t >= 0; --t) {
        const T* h_t = h + (t + 1) * BD;        // Normalized h at timestep t+1
        const T* h_prev = h + t * BD;           // Normalized h at timestep t
        const T* Wx_t = Wx_cache + t * BD;
        const T* d_out_t = d_output + t * BD;
        T* dWx_t = dWx_all + t * BD;

        // Step 1: Backward through self-gate
        SelfGateBackward_E59<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, d_out_t, dh);

        // Step 2: Combine with recurrent gradient
        if (t < steps - 1) {
            AddKernel_E59<T><<<num_blocks, block_size, 0, stream_>>>(
                BD, dh, dh_recurrent, dh);
        }

        // Step 3: Recompute h_raw = h_prev + alpha * Wx_t
        E59ResidualKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, alpha, Wx_t, h_prev, h_raw);

        // Step 4: RMSNorm backward
        E59RMSNormBackwardKernel<T><<<batch_size_, rmsnorm_threads, rmsnorm_smem, stream_>>>(
            batch_size_, dim_, h_raw, h_t, dh, dh_raw);

        // Step 5: Residual backward
        E59BackwardKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, alpha, Wx_t, dh_raw, nullptr, dWx_t,
            (t > 0) ? dh_recurrent : nullptr, d_log_alpha);

        // Step 6: Accumulate bias gradient
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
