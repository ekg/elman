// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E45: Pure Accumulation Elman
//
// The most extreme simplification: W = I (identity), just accumulate tokens.
//
// Architecture:
//     h_t = x_t + h_{t-1}              # Just add! No parameters in recurrence!
//     output = h_t * silu(h_t)          # Self-gating (only nonlinearity)
//
// E45b variant (with decay):
//     h_t = x_t + alpha * h_{t-1}       # Learned scalar decay
//     output = h_t * silu(h_t)          # Same self-gating
//
// Key: NO GEMM at all! Pure element-wise operations.

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
// E45 Forward Kernel: Pure accumulation + self-gate
// h_new = x + h_prev (NO GEMM!)
// output = h * silu(h)
// =============================================================================

__global__ void E45AccumulateGateKernel_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ x_t,
    const __nv_bfloat16* __restrict__ h_prev,
    __nv_bfloat16* __restrict__ h_out,
    __nv_bfloat16* __restrict__ output) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        // E45: Pure accumulation - just add! No GEMM!
#if __CUDA_ARCH__ >= 800
        __nv_bfloat16 h_val = __hadd(x_t[idx], h_prev[idx]);
#else
        float h_val_f = __bfloat162float(x_t[idx]) + __bfloat162float(h_prev[idx]);
        __nv_bfloat16 h_val = __float2bfloat16(h_val_f);
#endif
        h_out[idx] = h_val;

        // Self-gate: output = h * silu(h)
        float h_f = __bfloat162float(h_val);
        float sigmoid_h = 1.0f / (1.0f + __expf(-h_f));
        float silu_h = h_f * sigmoid_h;
        output[idx] = __float2bfloat16(h_f * silu_h);
    }
}

template<typename T>
__global__ void E45AccumulateGateKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ x_t,
    const T* __restrict__ h_prev,
    T* __restrict__ h_out,
    T* __restrict__ output) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        // E45: Pure accumulation
        float h_val = static_cast<float>(x_t[idx]) + static_cast<float>(h_prev[idx]);
        h_out[idx] = static_cast<T>(h_val);

        // Self-gate: output = h * silu(h)
        float sigmoid_h = 1.0f / (1.0f + expf(-h_val));
        float silu_h = h_val * sigmoid_h;
        output[idx] = static_cast<T>(h_val * silu_h);
    }
}

// =============================================================================
// E45b Forward Kernel: Accumulation with decay + self-gate
// h_new = x + alpha * h_prev
// output = h * silu(h)
// =============================================================================

__global__ void E45bDecayAccumulateGateKernel_BF16(
    const int batch_size,
    const int dim,
    const float alpha,
    const __nv_bfloat16* __restrict__ x_t,
    const __nv_bfloat16* __restrict__ h_prev,
    __nv_bfloat16* __restrict__ h_out,
    __nv_bfloat16* __restrict__ output) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        // E45b: Accumulation with decay
        float h_prev_f = __bfloat162float(h_prev[idx]);
        float x_f = __bfloat162float(x_t[idx]);
        float h_val = x_f + alpha * h_prev_f;
        h_out[idx] = __float2bfloat16(h_val);

        // Self-gate: output = h * silu(h)
        float sigmoid_h = 1.0f / (1.0f + __expf(-h_val));
        float silu_h = h_val * sigmoid_h;
        output[idx] = __float2bfloat16(h_val * silu_h);
    }
}

template<typename T>
__global__ void E45bDecayAccumulateGateKernel(
    const int batch_size,
    const int dim,
    const float alpha,
    const T* __restrict__ x_t,
    const T* __restrict__ h_prev,
    T* __restrict__ h_out,
    T* __restrict__ output) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        // E45b: Accumulation with decay
        float h_val = static_cast<float>(x_t[idx]) + alpha * static_cast<float>(h_prev[idx]);
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

// Self-gate backward (same as E42)
// output = h * silu(h) = h^2 * sigmoid(h)
// d(output)/dh = silu(h) * (2 + h*(1-sigmoid(h)))
__global__ void SelfGateBackward_E45_BF16(
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
__global__ void SelfGateBackward_E45(
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

// E45 Backward: accumulate dh_recurrent into dh, compute dx
// For E45: dh/dx = 1, dh/dh_prev = 1
// dx = dh, dh_recurrent_prev = dh
__global__ void E45BackwardKernel_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ dh,
    const __nv_bfloat16* __restrict__ dh_recurrent,
    __nv_bfloat16* __restrict__ dx,
    __nv_bfloat16* __restrict__ dh_recurrent_prev) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        // Combine gradients from output and recurrence
        float grad = __bfloat162float(dh[idx]);
        if (dh_recurrent) grad += __bfloat162float(dh_recurrent[idx]);

        // E45: dx = dh, dh_recurrent_prev = dh (identity Jacobians)
        dx[idx] = __float2bfloat16(grad);
        if (dh_recurrent_prev) dh_recurrent_prev[idx] = __float2bfloat16(grad);
    }
}

template<typename T>
__global__ void E45BackwardKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ dh,
    const T* __restrict__ dh_recurrent,
    T* __restrict__ dx,
    T* __restrict__ dh_recurrent_prev) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        float grad = static_cast<float>(dh[idx]);
        if (dh_recurrent) grad += static_cast<float>(dh_recurrent[idx]);

        dx[idx] = static_cast<T>(grad);
        if (dh_recurrent_prev) dh_recurrent_prev[idx] = static_cast<T>(grad);
    }
}

// E45b Backward: Same as E45 but with alpha scaling
// For E45b: dh/dx = 1, dh/dh_prev = alpha
// dx = dh, dh_recurrent_prev = alpha * dh
// dalpha = sum(dh * h_prev)
__global__ void E45bBackwardKernel_BF16(
    const int batch_size,
    const int dim,
    const float alpha,
    const __nv_bfloat16* __restrict__ h_prev,
    const __nv_bfloat16* __restrict__ dh,
    const __nv_bfloat16* __restrict__ dh_recurrent,
    __nv_bfloat16* __restrict__ dx,
    __nv_bfloat16* __restrict__ dh_recurrent_prev,
    float* __restrict__ d_alpha) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        float grad = __bfloat162float(dh[idx]);
        if (dh_recurrent) grad += __bfloat162float(dh_recurrent[idx]);

        // dx = dh
        dx[idx] = __float2bfloat16(grad);

        // dh_recurrent_prev = alpha * dh
        if (dh_recurrent_prev) dh_recurrent_prev[idx] = __float2bfloat16(alpha * grad);

        // d_alpha contribution: dL/dalpha = sum_t(dh_t * h_{t-1})
        float h_prev_val = __bfloat162float(h_prev[idx]);
        atomicAdd(d_alpha, grad * h_prev_val);
    }
}

template<typename T>
__global__ void E45bBackwardKernel(
    const int batch_size,
    const int dim,
    const float alpha,
    const T* __restrict__ h_prev,
    const T* __restrict__ dh,
    const T* __restrict__ dh_recurrent,
    T* __restrict__ dx,
    T* __restrict__ dh_recurrent_prev,
    float* __restrict__ d_alpha) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        float grad = static_cast<float>(dh[idx]);
        if (dh_recurrent) grad += static_cast<float>(dh_recurrent[idx]);

        dx[idx] = static_cast<T>(grad);
        if (dh_recurrent_prev) dh_recurrent_prev[idx] = static_cast<T>(alpha * grad);

        float h_prev_val = static_cast<float>(h_prev[idx]);
        atomicAdd(d_alpha, grad * h_prev_val);
    }
}

}  // anonymous namespace


namespace hasty {
namespace v0 {
namespace elman_ladder {

// =============================================================================
// E45 Pure Accumulation Forward - BF16 Specialization
// =============================================================================

template<>
E45PureAccumulationForward<__nv_bfloat16>::E45PureAccumulationForward(
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
void E45PureAccumulationForward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* x,
    __nv_bfloat16* h,
    __nv_bfloat16* output,
    __nv_bfloat16* workspace) {

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // E45: NO GEMM! Just loop with element-wise accumulation
    for (int t = 0; t < steps; ++t) {
        const __nv_bfloat16* x_t = x + t * BD;
        const __nv_bfloat16* h_prev = h + t * BD;
        __nv_bfloat16* h_t = h + (t + 1) * BD;
        __nv_bfloat16* out_t = output + t * BD;

        E45AccumulateGateKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, x_t, h_prev, h_t, out_t);
    }
}

// =============================================================================
// E45b With Decay Forward - BF16 Specialization
// =============================================================================

template<>
E45bWithDecayForward<__nv_bfloat16>::E45bWithDecayForward(
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
void E45bWithDecayForward<__nv_bfloat16>::Run(
    int steps,
    const float alpha,
    const __nv_bfloat16* x,
    __nv_bfloat16* h,
    __nv_bfloat16* output,
    __nv_bfloat16* workspace) {

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    for (int t = 0; t < steps; ++t) {
        const __nv_bfloat16* x_t = x + t * BD;
        const __nv_bfloat16* h_prev = h + t * BD;
        __nv_bfloat16* h_t = h + (t + 1) * BD;
        __nv_bfloat16* out_t = output + t * BD;

        E45bDecayAccumulateGateKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, alpha, x_t, h_prev, h_t, out_t);
    }
}

// =============================================================================
// E45 Pure Accumulation Backward - BF16 Specialization
// =============================================================================

template<>
E45PureAccumulationBackward<__nv_bfloat16>::E45PureAccumulationBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void E45PureAccumulationBackward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* h,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* dx,
    __nv_bfloat16* workspace) {

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Workspace: [dh: BD] [dh_recurrent: BD]
    __nv_bfloat16* dh = workspace;
    __nv_bfloat16* dh_recurrent = workspace + BD;

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(__nv_bfloat16), stream_);

    // BPTT loop
    for (int t = steps - 1; t >= 0; --t) {
        const __nv_bfloat16* h_t = h + (t + 1) * BD;
        const __nv_bfloat16* d_out_t = d_output + t * BD;
        __nv_bfloat16* dx_t = dx + t * BD;

        // Backward through self-gate: d_output -> dh
        SelfGateBackward_E45_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, d_out_t, dh);

        // E45 backward: dx = dh + dh_recurrent, dh_recurrent_prev = dh + dh_recurrent
        E45BackwardKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, dh, dh_recurrent, dx_t,
            (t > 0) ? dh_recurrent : nullptr);
    }
}

// =============================================================================
// E45b With Decay Backward - BF16 Specialization
// =============================================================================

template<>
E45bWithDecayBackward<__nv_bfloat16>::E45bWithDecayBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void E45bWithDecayBackward<__nv_bfloat16>::Run(
    int steps,
    const float alpha,
    const __nv_bfloat16* h,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* dx,
    float* d_alpha,
    __nv_bfloat16* workspace) {

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Workspace: [dh: BD] [dh_recurrent: BD]
    __nv_bfloat16* dh = workspace;
    __nv_bfloat16* dh_recurrent = workspace + BD;

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(d_alpha, 0, sizeof(float), stream_);

    // BPTT loop
    for (int t = steps - 1; t >= 0; --t) {
        const __nv_bfloat16* h_t = h + (t + 1) * BD;
        const __nv_bfloat16* h_prev = h + t * BD;
        const __nv_bfloat16* d_out_t = d_output + t * BD;
        __nv_bfloat16* dx_t = dx + t * BD;

        // Backward through self-gate
        SelfGateBackward_E45_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, d_out_t, dh);

        // E45b backward with decay
        E45bBackwardKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, alpha, h_prev, dh, dh_recurrent, dx_t,
            (t > 0) ? dh_recurrent : nullptr, d_alpha);
    }
}

// =============================================================================
// Generic Template Implementations
// =============================================================================

template<typename T>
E45PureAccumulationForward<T>::E45PureAccumulationForward(
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
void E45PureAccumulationForward<T>::Run(
    int steps,
    const T* x,
    T* h,
    T* output,
    T* workspace) {

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    for (int t = 0; t < steps; ++t) {
        const T* x_t = x + t * BD;
        const T* h_prev = h + t * BD;
        T* h_t = h + (t + 1) * BD;
        T* out_t = output + t * BD;

        E45AccumulateGateKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, x_t, h_prev, h_t, out_t);
    }
}

template<typename T>
E45bWithDecayForward<T>::E45bWithDecayForward(
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
void E45bWithDecayForward<T>::Run(
    int steps,
    const float alpha,
    const T* x,
    T* h,
    T* output,
    T* workspace) {

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    for (int t = 0; t < steps; ++t) {
        const T* x_t = x + t * BD;
        const T* h_prev = h + t * BD;
        T* h_t = h + (t + 1) * BD;
        T* out_t = output + t * BD;

        E45bDecayAccumulateGateKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, alpha, x_t, h_prev, h_t, out_t);
    }
}

template<typename T>
E45PureAccumulationBackward<T>::E45PureAccumulationBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void E45PureAccumulationBackward<T>::Run(
    int steps,
    const T* h,
    const T* d_output,
    T* dx,
    T* workspace) {

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    T* dh = workspace;
    T* dh_recurrent = workspace + BD;

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(T), stream_);

    for (int t = steps - 1; t >= 0; --t) {
        const T* h_t = h + (t + 1) * BD;
        const T* d_out_t = d_output + t * BD;
        T* dx_t = dx + t * BD;

        SelfGateBackward_E45<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, d_out_t, dh);

        E45BackwardKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, dh, dh_recurrent, dx_t,
            (t > 0) ? dh_recurrent : nullptr);
    }
}

template<typename T>
E45bWithDecayBackward<T>::E45bWithDecayBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void E45bWithDecayBackward<T>::Run(
    int steps,
    const float alpha,
    const T* h,
    const T* d_output,
    T* dx,
    float* d_alpha,
    T* workspace) {

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    T* dh = workspace;
    T* dh_recurrent = workspace + BD;

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(T), stream_);
    cudaMemsetAsync(d_alpha, 0, sizeof(float), stream_);

    for (int t = steps - 1; t >= 0; --t) {
        const T* h_t = h + (t + 1) * BD;
        const T* h_prev = h + t * BD;
        const T* d_out_t = d_output + t * BD;
        T* dx_t = dx + t * BD;

        SelfGateBackward_E45<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, d_out_t, dh);

        E45bBackwardKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, alpha, h_prev, dh, dh_recurrent, dx_t,
            (t > 0) ? dh_recurrent : nullptr, d_alpha);
    }
}

// Explicit template instantiations
template struct E45PureAccumulationForward<__half>;
template struct E45PureAccumulationForward<float>;
template struct E45PureAccumulationForward<double>;

template struct E45bWithDecayForward<__half>;
template struct E45bWithDecayForward<float>;
template struct E45bWithDecayForward<double>;

template struct E45PureAccumulationBackward<__half>;
template struct E45PureAccumulationBackward<float>;
template struct E45PureAccumulationBackward<double>;

template struct E45bWithDecayBackward<__half>;
template struct E45bWithDecayBackward<float>;
template struct E45bWithDecayBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
