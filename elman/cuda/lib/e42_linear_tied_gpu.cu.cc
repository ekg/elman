// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E42: Linear Tied Self-Gated Elman
//
// Combines E36 (linear recurrence, no tanh) + E37 (tied weights)
// Uses E37v2's batched GEMM optimization for speed.
//
// Architecture:
// h_t = W @ x_t + W @ h_{t-1} + b   # LINEAR recurrence, tied weights (no tanh!)
// output = h * silu(h)              # Self-gating (only nonlinearity)
//
// Key optimization: Batch W @ x for ALL timesteps (like E37v2), then
// compute W @ h_prev per timestep (unavoidable due to recurrence).

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
// FUSED Forward Kernel: Wx + Wh + bias + self-gate (NO TANH - linear!)
// Input: Wx = W @ x_t (pre-batched), Wh = W @ h_prev (per-step)
// Output: h_new = Wx + Wh + b (linear), output = h * silu(h)
// =============================================================================

__global__ void FusedLinearGateKernel_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ Wx,
    const __nv_bfloat16* __restrict__ Wh,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ h_out,
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // Native bf16 additions: Wx + Wh + bias
        __nv_bfloat16 sum = bf16_add(bf16_add(Wx[idx], Wh[idx]), b[d]);

        // Store pre-activation (for backward, same as h_out for linear)
        if (v_cache) v_cache[idx] = sum;

        // E42: NO TANH - linear recurrence!
        float h_val = __bfloat162float(sum);
        h_out[idx] = sum;

        // SELF-GATE: output = h * silu(h) - the ONLY nonlinearity
        float sigmoid_h = 1.0f / (1.0f + __expf(-h_val));
        float silu_h = h_val * sigmoid_h;

        output[idx] = __float2bfloat16(h_val * silu_h);
    }
}

template<typename T>
__global__ void FusedLinearGateKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ Wx,
    const T* __restrict__ Wh,
    const T* __restrict__ b,
    T* __restrict__ h_out,
    T* __restrict__ output,
    T* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float val = static_cast<float>(Wx[idx]) + static_cast<float>(Wh[idx]) + static_cast<float>(b[d]);
        if (v_cache) v_cache[idx] = static_cast<T>(val);

        // E42: NO TANH - linear recurrence!
        float h_val = val;
        h_out[idx] = static_cast<T>(h_val);

        // Self-gate
        float sigmoid_h = 1.0f / (1.0f + expf(-h_val));
        float silu_h = h_val * sigmoid_h;

        output[idx] = static_cast<T>(h_val * silu_h);
    }
}

// =============================================================================
// Backward Kernels (E42: Linear recurrence - no tanh derivative!)
// =============================================================================

// E42: Backward through LINEAR recurrence - gradient flows directly (derivative is 1)
__global__ void LinearBackwardKernel_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ v,  // unused for linear but kept for API
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

        // E42: LINEAR backward - gradient flows directly (derivative is 1)
        // No tanh derivative needed! This is the key difference from E37v2
        float dv_val = grad;

        dv[idx] = __float2bfloat16(dv_val);
        atomicAdd(&db[d], dv_val);
    }
}

template<typename T>
__global__ void LinearBackwardKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ v,  // unused for linear
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

        // E42: LINEAR backward - gradient flows directly (derivative is 1)
        float dv_val = grad;
        dv[idx] = static_cast<T>(dv_val);

        atomicAdd(&db[d], dv_val);
    }
}

// Self-gate backward (same as E37v2)
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

// Fused x+h kernel for backward dW optimization
// Computes out = a + b (element-wise addition into a new buffer)
__global__ void VectorAdd_BF16(
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
__global__ void VectorAdd(
    const int n,
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = static_cast<T>(static_cast<float>(a[idx]) + static_cast<float>(b[idx]));
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
// E42 Linear Tied Forward - BF16 Specialization
// Uses same batched GEMM pattern as E37v2, but with linear recurrence (no tanh)
// =============================================================================

template<>
E42LinearTiedForward<__nv_bfloat16>::E42LinearTiedForward(
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
void E42LinearTiedForward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W,      // [dim, dim] - SINGLE weight matrix (tied)
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

    // Workspace layout: [tmp_Wx: T*BD] [tmp_Wh: BD]
    // tmp_Wx stores pre-computed W @ x for all timesteps (batched GEMM)
    // tmp_Wh is reused each timestep for W @ h_prev
    __nv_bfloat16* tmp_Wx = workspace;
    __nv_bfloat16* tmp_Wh = workspace + steps * BD;

    // =========================================================================
    // KEY OPTIMIZATION: Pre-compute W @ x for ALL timesteps in one batched GEMM
    // This is the E37v2 pattern that gives it 140K+ tok/s
    // =========================================================================
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,  // M, N, K - batch across all T*B
        &alpha,
        W, dim_,                           // Same W used for both x and h
        x, dim_,
        &beta_zero,
        tmp_Wx, dim_);

    // Process each timestep with sequential W @ h_prev
    for (int t = 0; t < steps; ++t) {
        const __nv_bfloat16* Wx_t = tmp_Wx + t * BD;  // Pre-computed W @ x_t
        const __nv_bfloat16* h_prev = h + t * BD;
        __nv_bfloat16* h_t = h + (t + 1) * BD;
        __nv_bfloat16* out_t = output + t * BD;
        __nv_bfloat16* v_t = training_ ? (v + t * BD) : nullptr;

        // W @ h_prev (same W, different data - sequential)
        blas<__nv_bfloat16>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha,
            W, dim_,          // Same W used for hidden state
            h_prev, dim_,
            &beta_zero,
            tmp_Wh, dim_);

        // E42: Fused LINEAR + self-gate (NO TANH!)
        // h_new = Wx + Wh + b (linear), output = h_new * silu(h_new)
        FusedLinearGateKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, Wx_t, tmp_Wh, b, h_t, out_t, v_t);
    }
}

// =============================================================================
// E42 Linear Tied Backward - BF16 Specialization
// =============================================================================

template<>
E42LinearTiedBackward<__nv_bfloat16>::E42LinearTiedBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void E42LinearTiedBackward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W,
    const __nv_bfloat16* x,
    const __nv_bfloat16* h,
    const __nv_bfloat16* v,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* dx,
    __nv_bfloat16* dW,       // Single weight gradient (tied)
    __nv_bfloat16* db,
    __nv_bfloat16* workspace) {

    static const __nv_bfloat16 alpha_one = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);
    static const __nv_bfloat16 beta_one = __float2bfloat16(1.0f);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Workspace layout: [dv_all: T*BD] [dh: BD] [dh_recurrent: BD] [x_plus_h: T*BD] [db_float: dim]
    // x_plus_h is for the fused dW GEMM: dW = (x + h) @ dv_all.T
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
        const __nv_bfloat16* v_t = v + t * BD;
        const __nv_bfloat16* h_t = h + (t + 1) * BD;
        const __nv_bfloat16* d_out_t = d_output + t * BD;
        __nv_bfloat16* dv_t = dv_all + t * BD;

        // Backward through self-gate
        SelfGateBackward_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, d_out_t, dh);

        // Add recurrent gradient
        VectorAddInplace_BF16<<<num_blocks, block_size, 0, stream_>>>(BD, dh, dh_recurrent);

        // E42: Backward through LINEAR recurrence (NO tanh derivative!)
        LinearBackwardKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, v_t, dh, nullptr, dv_t, db_float);

        // dh_recurrent = W @ dv (same W for both paths)
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

    // =========================================================================
    // KEY OPTIMIZATION: Batched GEMM for dx (same as E37v2)
    // dx = W @ dv_all across all timesteps
    // =========================================================================
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W, dim_,
        dv_all, dim_,
        &beta_zero,
        dx, dim_);

    // =========================================================================
    // FUSED dW OPTIMIZATION: Single GEMM instead of two
    // Old: dW = x @ dv_all.T + h @ dv_all.T (two GEMMs with beta=1 accumulation)
    // New: dW = (x + h) @ dv_all.T (one element-wise add + one GEMM)
    // This is faster because:
    // 1. Element-wise x+h is memory-bound but cheap (T*B*D elements)
    // 2. Single GEMM is more efficient than two GEMMs with accumulation
    // =========================================================================

    // Step 1: Compute x_plus_h = x + h[0:T] (element-wise)
    // h has shape [T+1, B, dim], we use h[0:T*B] as h_prev for each timestep
    const int total_elements = steps * BD;
    const int add_blocks = (total_elements + block_size - 1) / block_size;
    VectorAdd_BF16<<<add_blocks, block_size, 0, stream_>>>(
        total_elements, x, h, x_plus_h);

    // Step 2: Single GEMM: dW = (x + h) @ dv_all.T
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        x_plus_h, dim_,
        dv_all, dim_,
        &beta_one,  // Accumulate into dW (which was zeroed)
        dW, dim_);

    // Copy float gradients to bf16
    CopyFloatToT<__nv_bfloat16><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_float, db);
}

// =============================================================================
// Generic Template Implementations (float, half, double)
// =============================================================================

template<typename T>
E42LinearTiedForward<T>::E42LinearTiedForward(
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
void E42LinearTiedForward<T>::Run(
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

    T* tmp_Wx = workspace;
    T* tmp_Wh = workspace + steps * BD;

    // Batch GEMM for W @ x across all timesteps
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha,
        W, dim_,
        x, dim_,
        &beta_zero,
        tmp_Wx, dim_);

    for (int t = 0; t < steps; ++t) {
        const T* Wx_t = tmp_Wx + t * BD;
        const T* h_prev = h + t * BD;
        T* h_t = h + (t + 1) * BD;
        T* out_t = output + t * BD;
        T* v_t = training_ ? (v + t * BD) : nullptr;

        // Sequential GEMM for W @ h_prev
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha,
            W, dim_,
            h_prev, dim_,
            &beta_zero,
            tmp_Wh, dim_);

        // E42: Fused LINEAR + self-gate kernel (NO TANH!)
        FusedLinearGateKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, Wx_t, tmp_Wh, b, h_t, out_t, v_t);
    }
}

template<typename T>
E42LinearTiedBackward<T>::E42LinearTiedBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void E42LinearTiedBackward<T>::Run(
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

    // Workspace layout: [dv_all: T*BD] [dh: BD] [dh_recurrent: BD] [x_plus_h: T*BD] [db_float: dim]
    T* dv_all = workspace;
    T* dh = workspace + steps * BD;
    T* dh_recurrent = workspace + (steps + 1) * BD;
    T* x_plus_h = workspace + (steps + 2) * BD;
    float* db_float = reinterpret_cast<float*>(workspace + (2 * steps + 2) * BD);

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

        // E42: LINEAR backward (NO tanh derivative)
        LinearBackwardKernel<T><<<num_blocks, block_size, 0, stream_>>>(
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

    // Batched GEMM for dx
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha,
        W, dim_,
        dv_all, dim_,
        &beta_zero,
        dx, dim_);

    // FUSED dW OPTIMIZATION: Single GEMM instead of two
    // Step 1: Compute x_plus_h = x + h[0:T]
    const int total_elements = steps * BD;
    const int add_blocks = (total_elements + block_size - 1) / block_size;
    VectorAdd<T><<<add_blocks, block_size, 0, stream_>>>(
        total_elements, x, h, x_plus_h);

    // Step 2: Single GEMM: dW = (x + h) @ dv_all.T
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha,
        x_plus_h, dim_,
        dv_all, dim_,
        &beta_one,
        dW, dim_);

    CopyFloatToT<T><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_float, db);
}

// Explicit template instantiations
template struct E42LinearTiedForward<__half>;
template struct E42LinearTiedForward<float>;
template struct E42LinearTiedForward<double>;

template struct E42LinearTiedBackward<__half>;
template struct E42LinearTiedBackward<float>;
template struct E42LinearTiedBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
