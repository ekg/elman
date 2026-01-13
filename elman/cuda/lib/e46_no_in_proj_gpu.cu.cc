// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E46: No In-Projection Elman
//
// Removes the input projection - recurrence operates on raw embeddings.
//
// Architecture:
//     # NO in_proj! Direct on embeddings
//     h_t = W @ (x_t + h_{t-1}) + b    # W is dim*dim (not d_inner*d_inner!)
//     output = h_t * silu(h_t)          # Self-gating
//
// Key insight:
//     - If W mixes everything, why have a separate input projection?
//     - The W matrix can incorporate the projection's role
//     - Uses single GEMM per timestep: W @ (x + h)
//
// Based on E37's tied weights but without expansion (operates on embedding dim).

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
// Element-wise x + h kernel for E46
// =============================================================================

__global__ void AddKernel_BF16(
    const int n,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ h,
    __nv_bfloat16* __restrict__ out) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = bf16_add(x[idx], h[idx]);
    }
}

template<typename T>
__global__ void AddKernel(
    const int n,
    const T* __restrict__ x,
    const T* __restrict__ h,
    T* __restrict__ out) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = static_cast<T>(static_cast<float>(x[idx]) + static_cast<float>(h[idx]));
    }
}

// =============================================================================
// E46 Forward Kernel: W@(x+h) + b + self-gate (LINEAR recurrence)
// Input: Wsum = W @ (x_t + h_prev) (computed via GEMM)
// Output: h_new = Wsum + b (linear), output = h * silu(h)
// =============================================================================

__global__ void E46LinearGateKernel_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ Wsum,
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

        // Store pre-activation (for backward)
        if (v_cache) v_cache[idx] = sum;

        // E46: LINEAR recurrence (no tanh!)
        float h_val = __bfloat162float(sum);
        h_out[idx] = sum;

        // SELF-GATE: output = h * silu(h)
        float sigmoid_h = 1.0f / (1.0f + __expf(-h_val));
        float silu_h = h_val * sigmoid_h;
        output[idx] = __float2bfloat16(h_val * silu_h);
    }
}

template<typename T>
__global__ void E46LinearGateKernel(
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

        // E46: LINEAR recurrence
        float h_val = val;
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

// E46: Backward through LINEAR recurrence - gradient flows directly (derivative is 1)
__global__ void E46LinearBackwardKernel_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ dh,
    const __nv_bfloat16* __restrict__ dh_recurrent,
    __nv_bfloat16* __restrict__ dv,
    float* __restrict__ db) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float grad;
        if (dh_recurrent) {
            __nv_bfloat16 combined = bf16_add(dh[idx], dh_recurrent[idx]);
            grad = __bfloat162float(combined);
        } else {
            grad = __bfloat162float(dh[idx]);
        }

        // E46: LINEAR backward - gradient flows directly (derivative is 1)
        float dv_val = grad;
        dv[idx] = __float2bfloat16(dv_val);
        atomicAdd(&db[d], dv_val);
    }
}

template<typename T>
__global__ void E46LinearBackwardKernel(
    const int batch_size,
    const int dim,
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

        float dv_val = grad;
        dv[idx] = static_cast<T>(dv_val);
        atomicAdd(&db[d], dv_val);
    }
}

// Self-gate backward
// output = h * silu(h) = h^2 * sigmoid(h)
// d(output)/dh = silu(h) * (2 + h*(1-sigmoid(h)))
__global__ void E46SelfGateBackward_BF16(
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
__global__ void E46SelfGateBackward(
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

// Add in-place for dh accumulation
__global__ void VectorAddInplace_E46_BF16(
    const int n,
    __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = bf16_add(a[idx], b[idx]);
    }
}

template<typename T>
__global__ void VectorAddInplace_E46(const int n, T* __restrict__ a, const T* __restrict__ b) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = static_cast<T>(static_cast<float>(a[idx]) + static_cast<float>(b[idx]));
    }
}

template<typename T>
__global__ void CopyFloatToT_E46(const int n, const float* __restrict__ src, T* __restrict__ dst) {
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
// E46 No In-Proj Forward - BF16 Specialization
// Uses E37-style: compute W @ (x + h) per timestep
// =============================================================================

template<>
E46NoInProjForward<__nv_bfloat16>::E46NoInProjForward(
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
void E46NoInProjForward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W,      // [dim, dim] - weight matrix
    const __nv_bfloat16* b,      // [dim] - bias
    const __nv_bfloat16* x,      // [T, B, dim] - input (pre-activated with silu)
    __nv_bfloat16* h,            // [T+1, B, dim] - hidden states
    __nv_bfloat16* output,       // [T, B, dim] - output
    __nv_bfloat16* v,            // [T, B, dim] - pre-activation cache
    __nv_bfloat16* workspace) {  // [2*B*dim] for tmp_sum, tmp_Wsum

    static const __nv_bfloat16 alpha = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Workspace: [tmp_sum: BD] [tmp_Wsum: BD]
    __nv_bfloat16* tmp_sum = workspace;
    __nv_bfloat16* tmp_Wsum = workspace + BD;

    // E46: Process each timestep with W @ (x_t + h_prev)
    for (int t = 0; t < steps; ++t) {
        const __nv_bfloat16* x_t = x + t * BD;
        const __nv_bfloat16* h_prev = h + t * BD;
        __nv_bfloat16* h_t = h + (t + 1) * BD;
        __nv_bfloat16* out_t = output + t * BD;
        __nv_bfloat16* v_t = training_ ? (v + t * BD) : nullptr;

        // Step 1: Compute x_t + h_prev
        AddKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            BD, x_t, h_prev, tmp_sum);

        // Step 2: W @ (x_t + h_prev)
        blas<__nv_bfloat16>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha,
            W, dim_,
            tmp_sum, dim_,
            &beta_zero,
            tmp_Wsum, dim_);

        // Step 3: Add bias + self-gate
        E46LinearGateKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, tmp_Wsum, b, h_t, out_t, v_t);
    }
}

// =============================================================================
// E46 No In-Proj Backward - BF16 Specialization
// =============================================================================

template<>
E46NoInProjBackward<__nv_bfloat16>::E46NoInProjBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void E46NoInProjBackward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W,
    const __nv_bfloat16* x,
    const __nv_bfloat16* h,
    const __nv_bfloat16* v,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* dx,
    __nv_bfloat16* dW,
    __nv_bfloat16* db,
    __nv_bfloat16* workspace) {

    static const __nv_bfloat16 alpha_one = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);
    static const __nv_bfloat16 beta_one = __float2bfloat16(1.0f);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Workspace: [dv_all: T*BD] [dh: BD] [dh_recurrent: BD] [x_plus_h: T*BD] [db_float: dim]
    __nv_bfloat16* dv_all = workspace;
    __nv_bfloat16* dh = workspace + steps * BD;
    __nv_bfloat16* dh_recurrent = workspace + (steps + 1) * BD;
    __nv_bfloat16* x_plus_h = workspace + (steps + 2) * BD;
    float* db_float = reinterpret_cast<float*>(workspace + (2 * steps + 2) * BD);

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);

    // BPTT loop
    for (int t = steps - 1; t >= 0; --t) {
        const __nv_bfloat16* h_t = h + (t + 1) * BD;
        const __nv_bfloat16* d_out_t = d_output + t * BD;
        __nv_bfloat16* dv_t = dv_all + t * BD;

        // Backward through self-gate
        E46SelfGateBackward_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, d_out_t, dh);

        // Add recurrent gradient
        VectorAddInplace_E46_BF16<<<num_blocks, block_size, 0, stream_>>>(BD, dh, dh_recurrent);

        // E46: Backward through LINEAR recurrence
        E46LinearBackwardKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, dh, nullptr, dv_t, db_float);

        // dh_recurrent = W @ dv (for next iteration)
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

    // dx = W @ dv_all (batched GEMM across all timesteps)
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W, dim_,
        dv_all, dim_,
        &beta_zero,
        dx, dim_);

    // Compute x_plus_h = x + h[0:T] for dW GEMM
    const int total_elements = steps * BD;
    const int add_blocks = (total_elements + block_size - 1) / block_size;
    AddKernel_BF16<<<add_blocks, block_size, 0, stream_>>>(
        total_elements, x, h, x_plus_h);

    // dW = (x + h) @ dv_all.T (single GEMM)
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        x_plus_h, dim_,
        dv_all, dim_,
        &beta_one,
        dW, dim_);

    // Copy float gradients to bf16
    CopyFloatToT_E46<__nv_bfloat16><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_float, db);
}

// =============================================================================
// Generic Template Implementations
// =============================================================================

template<typename T>
E46NoInProjForward<T>::E46NoInProjForward(
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
void E46NoInProjForward<T>::Run(
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
        AddKernel<T><<<num_blocks, block_size, 0, stream_>>>(
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

        // Bias + self-gate
        E46LinearGateKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, tmp_Wsum, b, h_t, out_t, v_t);
    }
}

template<typename T>
E46NoInProjBackward<T>::E46NoInProjBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void E46NoInProjBackward<T>::Run(
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
    T* x_plus_h = workspace + (steps + 2) * BD;
    float* db_float = reinterpret_cast<float*>(workspace + (2 * steps + 2) * BD);

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(T), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW, 0, dim_ * dim_ * sizeof(T), stream_);

    for (int t = steps - 1; t >= 0; --t) {
        const T* h_t = h + (t + 1) * BD;
        const T* d_out_t = d_output + t * BD;
        T* dv_t = dv_all + t * BD;

        E46SelfGateBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, d_out_t, dh);

        VectorAddInplace_E46<T><<<num_blocks, block_size, 0, stream_>>>(BD, dh, dh_recurrent);

        E46LinearBackwardKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, dh, nullptr, dv_t, db_float);

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

    // dx
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha,
        W, dim_,
        dv_all, dim_,
        &beta_zero,
        dx, dim_);

    // x + h
    const int total_elements = steps * BD;
    const int add_blocks = (total_elements + block_size - 1) / block_size;
    AddKernel<T><<<add_blocks, block_size, 0, stream_>>>(
        total_elements, x, h, x_plus_h);

    // dW
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha,
        x_plus_h, dim_,
        dv_all, dim_,
        &beta_one,
        dW, dim_);

    CopyFloatToT_E46<T><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_float, db);
}

// Explicit template instantiations
template struct E46NoInProjForward<__half>;
template struct E46NoInProjForward<float>;
template struct E46NoInProjForward<double>;

template struct E46NoInProjBackward<__half>;
template struct E46NoInProjBackward<float>;
template struct E46NoInProjBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
