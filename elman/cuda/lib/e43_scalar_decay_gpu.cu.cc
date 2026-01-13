// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E43: Scalar Decay Elman
//
// The most radical W simplification: a single scalar λ replaces the entire d×d matrix.
//
// Architecture:
//     h_t = λ * (x_t + h_{t-1}) + b    # Scalar decay (NO matrix!)
//     output = h_t * silu(h_t)          # Self-gating (only nonlinearity)
//
// Key insight from E42:
//     - E42's W matrices collapse to spectral radius 0.3-0.7 in training
//     - Maybe only the decay rate matters, not the full mixing matrix
//     - This tests if dimension mixing is necessary at all
//
// Expected benefits:
//     - NO GEMM in recurrence! Just scalar multiply + add
//     - Orders of magnitude fewer parameters: 1 vs d²
//     - Potentially much faster than E42
//
// CRITICAL: λ is stored as log_lambda (pre-sigmoid), so λ = sigmoid(log_lambda) ∈ (0, 1)

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
// E43 Forward Kernel: h = λ * (x + h_prev) + b, output = h * silu(h)
// λ is a SINGLE scalar (same for all dimensions)
// =============================================================================

__global__ void E43ForwardKernel_BF16(
    const int batch_size,
    const int dim,
    const float lambda_val,     // λ = sigmoid(log_lambda), passed as float
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ h_prev,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ h_out,
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ v_cache) {  // stores x + h_prev for backward

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // x + h_prev
        float x_val = __bfloat162float(x[idx]);
        float hp_val = __bfloat162float(h_prev[idx]);
        float sum = x_val + hp_val;

        // Store x + h_prev for backward
        if (v_cache) v_cache[idx] = __float2bfloat16(sum);

        // h = λ * (x + h_prev) + b
        float b_val = __bfloat162float(b[d]);
        float h_val = lambda_val * sum + b_val;
        h_out[idx] = __float2bfloat16(h_val);

        // Self-gate: output = h * silu(h)
        float sigmoid_h = 1.0f / (1.0f + __expf(-h_val));
        float silu_h = h_val * sigmoid_h;
        output[idx] = __float2bfloat16(h_val * silu_h);
    }
}

template<typename T>
__global__ void E43ForwardKernel(
    const int batch_size,
    const int dim,
    const float lambda_val,
    const T* __restrict__ x,
    const T* __restrict__ h_prev,
    const T* __restrict__ b,
    T* __restrict__ h_out,
    T* __restrict__ output,
    T* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float x_val = static_cast<float>(x[idx]);
        float hp_val = static_cast<float>(h_prev[idx]);
        float sum = x_val + hp_val;

        if (v_cache) v_cache[idx] = static_cast<T>(sum);

        float b_val = static_cast<float>(b[d]);
        float h_val = lambda_val * sum + b_val;
        h_out[idx] = static_cast<T>(h_val);

        float sigmoid_h = 1.0f / (1.0f + expf(-h_val));
        float silu_h = h_val * sigmoid_h;
        output[idx] = static_cast<T>(h_val * silu_h);
    }
}

// =============================================================================
// E43 Backward Kernels
// =============================================================================

// Backward through self-gate
// output = h * silu(h) = h^2 * sigmoid(h)
// d(output)/dh = silu(h) * (2 + h*(1-sigmoid(h)))
__global__ void E43GateBackward_BF16(
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
__global__ void E43GateBackward(
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

// Backward through linear: h = λ * (x + h_prev) + b
// dh is the combined gradient (from output + recurrent)
// Outputs: dx, d_lambda (summed), db (summed)
// v_cache = x + h_prev (stored from forward)
// dh_recurrent_out = λ * dh (for next iteration)
__global__ void E43LinearBackward_BF16(
    const int batch_size,
    const int dim,
    const float lambda_val,
    const __nv_bfloat16* __restrict__ v_cache,  // x + h_prev
    const __nv_bfloat16* __restrict__ dh,
    const __nv_bfloat16* __restrict__ dh_recurrent_in,
    __nv_bfloat16* __restrict__ dx,
    __nv_bfloat16* __restrict__ dh_recurrent_out,
    float* __restrict__ db,
    float* __restrict__ d_lambda) {  // atomic sum

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // Combine gradients
        float grad = __bfloat162float(dh[idx]);
        if (dh_recurrent_in) {
            grad += __bfloat162float(dh_recurrent_in[idx]);
        }

        // dh/db = 1
        atomicAdd(&db[d], grad);

        // dh/d(x + h_prev) = λ
        float dx_val = lambda_val * grad;
        dx[idx] = __float2bfloat16(dx_val);

        // dh_recurrent = λ * grad (same as dx for tied decay)
        if (dh_recurrent_out) {
            dh_recurrent_out[idx] = __float2bfloat16(dx_val);
        }

        // dh/d_lambda = x + h_prev
        float v_val = __bfloat162float(v_cache[idx]);
        atomicAdd(d_lambda, grad * v_val);
    }
}

template<typename T>
__global__ void E43LinearBackward(
    const int batch_size,
    const int dim,
    const float lambda_val,
    const T* __restrict__ v_cache,
    const T* __restrict__ dh,
    const T* __restrict__ dh_recurrent_in,
    T* __restrict__ dx,
    T* __restrict__ dh_recurrent_out,
    float* __restrict__ db,
    float* __restrict__ d_lambda) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float grad = static_cast<float>(dh[idx]);
        if (dh_recurrent_in) {
            grad += static_cast<float>(dh_recurrent_in[idx]);
        }

        atomicAdd(&db[d], grad);

        float dx_val = lambda_val * grad;
        dx[idx] = static_cast<T>(dx_val);

        if (dh_recurrent_out) {
            dh_recurrent_out[idx] = static_cast<T>(dx_val);
        }

        float v_val = static_cast<float>(v_cache[idx]);
        atomicAdd(d_lambda, grad * v_val);
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
// E43 Scalar Decay Forward - BF16 Specialization
// =============================================================================

template<>
E43ScalarDecayForward<__nv_bfloat16>::E43ScalarDecayForward(
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
void E43ScalarDecayForward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* log_lambda,  // [1] - log(sigmoid^-1(λ))
    const __nv_bfloat16* b,           // [dim]
    const __nv_bfloat16* x,           // [T, B, dim] pre-activated input
    __nv_bfloat16* h,                 // [T+1, B, dim] hidden states
    __nv_bfloat16* output,            // [T, B, dim] output
    __nv_bfloat16* v,                 // [T, B, dim] cache for backward (stores x + h_prev)
    __nv_bfloat16* workspace) {       // unused

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Compute λ = sigmoid(log_lambda) on CPU/device
    // For simplicity, read log_lambda to host and compute sigmoid
    __nv_bfloat16 log_lambda_host;
    cudaMemcpy(&log_lambda_host, log_lambda, sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    float log_lam = __bfloat162float(log_lambda_host);
    float lambda_val = 1.0f / (1.0f + expf(-log_lam));

    // NO GEMM! Just element-wise ops per timestep
    for (int t = 0; t < steps; ++t) {
        const __nv_bfloat16* x_t = x + t * BD;
        const __nv_bfloat16* h_prev = h + t * BD;
        __nv_bfloat16* h_t = h + (t + 1) * BD;
        __nv_bfloat16* out_t = output + t * BD;
        __nv_bfloat16* v_t = training_ ? (v + t * BD) : nullptr;

        E43ForwardKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, lambda_val, x_t, h_prev, b, h_t, out_t, v_t);
    }
}

// =============================================================================
// E43 Scalar Decay Backward - BF16 Specialization
// =============================================================================

template<>
E43ScalarDecayBackward<__nv_bfloat16>::E43ScalarDecayBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void E43ScalarDecayBackward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* log_lambda,
    const __nv_bfloat16* x,
    const __nv_bfloat16* h,
    const __nv_bfloat16* v,           // x + h_prev for each timestep
    const __nv_bfloat16* d_output,
    __nv_bfloat16* dx,
    __nv_bfloat16* d_log_lambda,      // [1] gradient for log_lambda
    __nv_bfloat16* db,
    __nv_bfloat16* workspace) {

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Compute λ = sigmoid(log_lambda)
    __nv_bfloat16 log_lambda_host;
    cudaMemcpy(&log_lambda_host, log_lambda, sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    float log_lam = __bfloat162float(log_lambda_host);
    float lambda_val = 1.0f / (1.0f + expf(-log_lam));

    // Workspace layout: [dh: BD] [dh_recurrent: BD] [db_float: dim] [d_lambda_float: 1]
    __nv_bfloat16* dh = workspace;
    __nv_bfloat16* dh_recurrent = workspace + BD;
    float* db_float = reinterpret_cast<float*>(workspace + 2 * BD);
    float* d_lambda_float = db_float + dim_;

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(d_lambda_float, 0, sizeof(float), stream_);

    // BPTT loop
    for (int t = steps - 1; t >= 0; --t) {
        const __nv_bfloat16* h_t = h + (t + 1) * BD;
        const __nv_bfloat16* v_t = v + t * BD;
        const __nv_bfloat16* d_out_t = d_output + t * BD;
        __nv_bfloat16* dx_t = dx + t * BD;

        // Backward through self-gate
        E43GateBackward_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, d_out_t, dh);

        // Backward through linear: accumulates db and d_lambda, computes dx
        E43LinearBackward_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, lambda_val, v_t, dh,
            t < steps - 1 ? dh_recurrent : nullptr,
            dx_t,
            t > 0 ? dh_recurrent : nullptr,
            db_float, d_lambda_float);
    }

    // Convert d_lambda from float to bf16
    // d_log_lambda = d_lambda * lambda * (1 - lambda) (chain rule through sigmoid)
    float d_lambda_host;
    cudaMemcpy(&d_lambda_host, d_lambda_float, sizeof(float), cudaMemcpyDeviceToHost);
    float d_log_lambda_val = d_lambda_host * lambda_val * (1.0f - lambda_val);
    __nv_bfloat16 d_log_lambda_bf16 = __float2bfloat16(d_log_lambda_val);
    cudaMemcpy(d_log_lambda, &d_log_lambda_bf16, sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    // Copy float gradients to bf16
    CopyFloatToT<__nv_bfloat16><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_float, db);
}

// =============================================================================
// Generic Template Implementations (float, half, double)
// =============================================================================

template<typename T>
E43ScalarDecayForward<T>::E43ScalarDecayForward(
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
void E43ScalarDecayForward<T>::Run(
    int steps,
    const T* log_lambda,
    const T* b,
    const T* x,
    T* h,
    T* output,
    T* v,
    T* workspace) {

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Read log_lambda and compute sigmoid
    T log_lambda_host;
    cudaMemcpy(&log_lambda_host, log_lambda, sizeof(T), cudaMemcpyDeviceToHost);
    float log_lam = static_cast<float>(log_lambda_host);
    float lambda_val = 1.0f / (1.0f + expf(-log_lam));

    for (int t = 0; t < steps; ++t) {
        const T* x_t = x + t * BD;
        const T* h_prev = h + t * BD;
        T* h_t = h + (t + 1) * BD;
        T* out_t = output + t * BD;
        T* v_t = training_ ? (v + t * BD) : nullptr;

        E43ForwardKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, lambda_val, x_t, h_prev, b, h_t, out_t, v_t);
    }
}

template<typename T>
E43ScalarDecayBackward<T>::E43ScalarDecayBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void E43ScalarDecayBackward<T>::Run(
    int steps,
    const T* log_lambda,
    const T* x,
    const T* h,
    const T* v,
    const T* d_output,
    T* dx,
    T* d_log_lambda,
    T* db,
    T* workspace) {

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Compute lambda
    T log_lambda_host;
    cudaMemcpy(&log_lambda_host, log_lambda, sizeof(T), cudaMemcpyDeviceToHost);
    float log_lam = static_cast<float>(log_lambda_host);
    float lambda_val = 1.0f / (1.0f + expf(-log_lam));

    // Workspace layout: [dh: BD] [dh_recurrent: BD] [db_float: dim] [d_lambda_float: 1]
    T* dh = workspace;
    T* dh_recurrent = workspace + BD;
    float* db_float = reinterpret_cast<float*>(workspace + 2 * BD);
    float* d_lambda_float = db_float + dim_;

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(T), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(d_lambda_float, 0, sizeof(float), stream_);

    for (int t = steps - 1; t >= 0; --t) {
        const T* h_t = h + (t + 1) * BD;
        const T* v_t = v + t * BD;
        const T* d_out_t = d_output + t * BD;
        T* dx_t = dx + t * BD;

        E43GateBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, d_out_t, dh);

        E43LinearBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, lambda_val, v_t, dh,
            t < steps - 1 ? dh_recurrent : nullptr,
            dx_t,
            t > 0 ? dh_recurrent : nullptr,
            db_float, d_lambda_float);
    }

    // Convert d_lambda with sigmoid derivative
    float d_lambda_host;
    cudaMemcpy(&d_lambda_host, d_lambda_float, sizeof(float), cudaMemcpyDeviceToHost);
    float d_log_lambda_val = d_lambda_host * lambda_val * (1.0f - lambda_val);
    T d_log_lambda_T = static_cast<T>(d_log_lambda_val);
    cudaMemcpy(d_log_lambda, &d_log_lambda_T, sizeof(T), cudaMemcpyHostToDevice);

    CopyFloatToT<T><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_float, db);
}

// Explicit template instantiations
template struct E43ScalarDecayForward<__half>;
template struct E43ScalarDecayForward<float>;
template struct E43ScalarDecayForward<double>;

template struct E43ScalarDecayBackward<__half>;
template struct E43ScalarDecayBackward<float>;
template struct E43ScalarDecayBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
