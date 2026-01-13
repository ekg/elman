// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E55: Scalar Decay + No Projections Elman
//
// The ULTIMATE minimal recurrent layer - single scalar lambda.
// NO GEMM operations - pure element-wise ops.
//
// Architecture:
// h_t = lambda * (x_t + h_{t-1}) + b   # SINGLE scalar lambda for entire layer!
// output = h * silu(h)                  # Self-gating
//
// Total recurrence parameters per layer: 1 (lambda) + dim (b) = dim + 1
// This tests if ANY structure in W matters at all.

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
// E55 Forward Kernel: Scalar decay + self-gate
// h_t = lambda * (x + h_prev) + b, output = h * silu(h)
// =============================================================================

__global__ void E55ForwardKernel_BF16(
    const int batch_size,
    const int dim,
    const float lambda,                         // Single scalar!
    const __nv_bfloat16* __restrict__ b,        // [dim] - bias
    const __nv_bfloat16* __restrict__ x_t,      // [B, dim] - input at timestep t
    const __nv_bfloat16* __restrict__ h_prev,   // [B, dim] - hidden state at t-1
    __nv_bfloat16* __restrict__ h_out,          // [B, dim] - hidden state at t
    __nv_bfloat16* __restrict__ output,         // [B, dim] - output
    __nv_bfloat16* __restrict__ v_cache) {      // [B, dim] - pre-activation cache (optional)

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d_idx = idx % dim;

        // Load bias (broadcast across batch)
        float bias = __bfloat162float(b[d_idx]);

        // h_t = lambda * (x + h_prev) + b
        float x_val = __bfloat162float(x_t[idx]);
        float h_prev_val = __bfloat162float(h_prev[idx]);
        float h_new = lambda * (x_val + h_prev_val) + bias;

        // Store hidden state
        h_out[idx] = __float2bfloat16(h_new);

        // Cache pre-activation for backward
        if (v_cache) v_cache[idx] = __float2bfloat16(h_new);

        // Self-gate: output = h * silu(h) = h^2 * sigmoid(h)
        float sigmoid_h = 1.0f / (1.0f + __expf(-h_new));
        float silu_h = h_new * sigmoid_h;
        output[idx] = __float2bfloat16(h_new * silu_h);
    }
}

template<typename T>
__global__ void E55ForwardKernel(
    const int batch_size,
    const int dim,
    const float lambda,
    const T* __restrict__ b,
    const T* __restrict__ x_t,
    const T* __restrict__ h_prev,
    T* __restrict__ h_out,
    T* __restrict__ output,
    T* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d_idx = idx % dim;

        float bias = static_cast<float>(b[d_idx]);
        float x_val = static_cast<float>(x_t[idx]);
        float h_prev_val = static_cast<float>(h_prev[idx]);

        float h_new = lambda * (x_val + h_prev_val) + bias;

        h_out[idx] = static_cast<T>(h_new);
        if (v_cache) v_cache[idx] = static_cast<T>(h_new);

        float sigmoid_h = 1.0f / (1.0f + expf(-h_new));
        float silu_h = h_new * sigmoid_h;
        output[idx] = static_cast<T>(h_new * silu_h);
    }
}

// =============================================================================
// E55 Backward Kernels
// =============================================================================

// Backward through self-gate: output = h * silu(h) = h^2 * sigmoid(h)
// d(output)/dh = silu(h) * (2 + h*(1-sigmoid(h)))
__global__ void E55SelfGateBackward_BF16(
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
__global__ void E55SelfGateBackward(
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

// Backward through recurrence: h_t = lambda * (x + h_prev) + b
// dh_t is the gradient w.r.t. h_t (from output + recurrent path)
// dx_t = lambda * dh_t
// dh_prev = lambda * dh_t (added to dh_recurrent for next iteration)
// dlambda += dh_t * (x + h_prev)  (accumulated across batch, time, and dim!)
// db += dh_t (accumulated across batch and time)
__global__ void E55RecurrenceBackward_BF16(
    const int batch_size,
    const int dim,
    const float lambda,                         // Single scalar
    const __nv_bfloat16* __restrict__ x_t,      // [B, dim]
    const __nv_bfloat16* __restrict__ h_prev,   // [B, dim]
    const __nv_bfloat16* __restrict__ dh,       // [B, dim] gradient w.r.t. h_t
    const __nv_bfloat16* __restrict__ dh_recurrent, // [B, dim] from t+1 (nullptr for last step)
    __nv_bfloat16* __restrict__ dx,             // [B, dim] output: gradient w.r.t. x_t
    __nv_bfloat16* __restrict__ dh_prev_out,    // [B, dim] output: gradient for next iteration
    float* __restrict__ dlambda,                 // [1] accumulated: gradient w.r.t. lambda
    float* __restrict__ db) {                    // [dim] accumulated: gradient w.r.t. b

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d_idx = idx % dim;

        // Combine gradients from output path and recurrent path
        float dh_total;
        if (dh_recurrent) {
            dh_total = __bfloat162float(dh[idx]) + __bfloat162float(dh_recurrent[idx]);
        } else {
            dh_total = __bfloat162float(dh[idx]);
        }

        float x_val = __bfloat162float(x_t[idx]);
        float h_prev_val = __bfloat162float(h_prev[idx]);

        // dx = lambda * dh_total (gradient flows through scalar decay)
        dx[idx] = __float2bfloat16(lambda * dh_total);

        // dh_prev = lambda * dh_total (for recurrent backprop)
        dh_prev_out[idx] = __float2bfloat16(lambda * dh_total);

        // dlambda += dh_total * (x + h_prev) - atomic add for accumulation across all elements!
        atomicAdd(dlambda, dh_total * (x_val + h_prev_val));

        // db += dh_total
        atomicAdd(&db[d_idx], dh_total);
    }
}

template<typename T>
__global__ void E55RecurrenceBackward(
    const int batch_size,
    const int dim,
    const float lambda,
    const T* __restrict__ x_t,
    const T* __restrict__ h_prev,
    const T* __restrict__ dh,
    const T* __restrict__ dh_recurrent,
    T* __restrict__ dx,
    T* __restrict__ dh_prev_out,
    float* __restrict__ dlambda,
    float* __restrict__ db) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d_idx = idx % dim;

        float dh_total = static_cast<float>(dh[idx]);
        if (dh_recurrent) {
            dh_total += static_cast<float>(dh_recurrent[idx]);
        }

        float x_val = static_cast<float>(x_t[idx]);
        float h_prev_val = static_cast<float>(h_prev[idx]);

        dx[idx] = static_cast<T>(lambda * dh_total);
        dh_prev_out[idx] = static_cast<T>(lambda * dh_total);
        atomicAdd(dlambda, dh_total * (x_val + h_prev_val));
        atomicAdd(&db[d_idx], dh_total);
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
// E55 Forward - BF16 Specialization
// =============================================================================

template<>
E55ScalarNoProjForward<__nv_bfloat16>::E55ScalarNoProjForward(
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
void E55ScalarNoProjForward<__nv_bfloat16>::Run(
    int steps,
    const float lambda,          // Single scalar (already sigmoid)
    const __nv_bfloat16* b,      // [dim] - bias
    const __nv_bfloat16* x,      // [T, B, dim] - input
    __nv_bfloat16* h,            // [T+1, B, dim] - hidden states
    __nv_bfloat16* output,       // [T, B, dim] - output
    __nv_bfloat16* v,            // [T, B, dim] - pre-activation cache
    __nv_bfloat16* workspace) {  // unused for E55

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Process each timestep - NO GEMM, just element-wise!
    for (int t = 0; t < steps; ++t) {
        const __nv_bfloat16* x_t = x + t * BD;
        const __nv_bfloat16* h_prev = h + t * BD;
        __nv_bfloat16* h_t = h + (t + 1) * BD;
        __nv_bfloat16* out_t = output + t * BD;
        __nv_bfloat16* v_t = training_ ? (v + t * BD) : nullptr;

        E55ForwardKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, lambda, b, x_t, h_prev, h_t, out_t, v_t);
    }
}

// =============================================================================
// E55 Backward - BF16 Specialization
// =============================================================================

template<>
E55ScalarNoProjBackward<__nv_bfloat16>::E55ScalarNoProjBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void E55ScalarNoProjBackward<__nv_bfloat16>::Run(
    int steps,
    const float lambda,
    const __nv_bfloat16* x,
    const __nv_bfloat16* h,
    const __nv_bfloat16* v,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* dx,
    float* dlambda,              // [1] - gradient for scalar lambda (float output!)
    __nv_bfloat16* db,           // [dim] - gradient for bias
    __nv_bfloat16* workspace) {

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Workspace layout: [dh: BD] [dh_recurrent: BD] [db_float: dim]
    __nv_bfloat16* dh = workspace;
    __nv_bfloat16* dh_recurrent = workspace + BD;
    float* db_float = reinterpret_cast<float*>(workspace + 2 * BD);

    // Initialize accumulators
    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dlambda, 0, sizeof(float), stream_);  // Single scalar!
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);

    // BPTT loop - iterate backwards through time
    for (int t = steps - 1; t >= 0; --t) {
        const __nv_bfloat16* x_t = x + t * BD;
        const __nv_bfloat16* h_t = h + (t + 1) * BD;
        const __nv_bfloat16* h_prev = h + t * BD;
        const __nv_bfloat16* d_out_t = d_output + t * BD;
        __nv_bfloat16* dx_t = dx + t * BD;

        // Backward through self-gate: d_output -> dh
        E55SelfGateBackward_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, d_out_t, dh);

        // Backward through recurrence: dh + dh_recurrent -> dx, dh_prev, dlambda, db
        E55RecurrenceBackward_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, lambda, x_t, h_prev, dh,
            (t < steps - 1) ? dh_recurrent : nullptr,
            dx_t, dh_recurrent,
            dlambda, db_float);
    }

    // Copy float bias gradients to output
    CopyFloatToT<__nv_bfloat16><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_float, db);
    // dlambda is already float output, no copy needed
}

// =============================================================================
// Generic Template Implementations
// =============================================================================

template<typename T>
E55ScalarNoProjForward<T>::E55ScalarNoProjForward(
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
void E55ScalarNoProjForward<T>::Run(
    int steps,
    const float lambda,
    const T* b,
    const T* x,
    T* h,
    T* output,
    T* v,
    T* workspace) {

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    for (int t = 0; t < steps; ++t) {
        const T* x_t = x + t * BD;
        const T* h_prev = h + t * BD;
        T* h_t = h + (t + 1) * BD;
        T* out_t = output + t * BD;
        T* v_t = training_ ? (v + t * BD) : nullptr;

        E55ForwardKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, lambda, b, x_t, h_prev, h_t, out_t, v_t);
    }
}

template<typename T>
E55ScalarNoProjBackward<T>::E55ScalarNoProjBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void E55ScalarNoProjBackward<T>::Run(
    int steps,
    const float lambda,
    const T* x,
    const T* h,
    const T* v,
    const T* d_output,
    T* dx,
    float* dlambda,
    T* db,
    T* workspace) {

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    T* dh = workspace;
    T* dh_recurrent = workspace + BD;
    float* db_float = reinterpret_cast<float*>(workspace + 2 * BD);

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(T), stream_);
    cudaMemsetAsync(dlambda, 0, sizeof(float), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);

    for (int t = steps - 1; t >= 0; --t) {
        const T* x_t = x + t * BD;
        const T* h_t = h + (t + 1) * BD;
        const T* h_prev = h + t * BD;
        const T* d_out_t = d_output + t * BD;
        T* dx_t = dx + t * BD;

        E55SelfGateBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, d_out_t, dh);

        E55RecurrenceBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, lambda, x_t, h_prev, dh,
            (t < steps - 1) ? dh_recurrent : nullptr,
            dx_t, dh_recurrent,
            dlambda, db_float);
    }

    CopyFloatToT<T><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_float, db);
}

// Explicit template instantiations
template struct E55ScalarNoProjForward<__half>;
template struct E55ScalarNoProjForward<float>;
template struct E55ScalarNoProjForward<double>;

template struct E55ScalarNoProjBackward<__half>;
template struct E55ScalarNoProjBackward<float>;
template struct E55ScalarNoProjBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
