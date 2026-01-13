// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E56: Concat Elman - Single GEMM on concatenated [x, h] input
//
// Instead of two GEMMs (W_x @ x + W_h @ h), we concatenate [x, h]
// and do a single GEMM with W @ [x; h] where W is [dim, 2*dim].
//
// Architecture:
// h_t = tanh(W @ [x_t; h_{t-1}] + b)  # Single GEMM!
// output = h * silu(z)                 # Gate with z branch
//
// Key insight: Same parameter count as E1, but one GEMM instead of two.
// The matrix W is [dim, 2*dim] instead of two [dim, dim] matrices.

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
// E56 Forward Kernels
// =============================================================================

// Concatenate x_t and h_prev into [x_t; h_prev] for the GEMM
// Input: x_t [B, dim], h_prev [B, dim]
// Output: xh [B, 2*dim] where first dim cols are x, second dim cols are h
__global__ void ConcatKernel_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ x_t,      // [B, dim]
    const __nv_bfloat16* __restrict__ h_prev,   // [B, dim]
    __nv_bfloat16* __restrict__ xh) {           // [B, 2*dim]

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int b = idx / dim;
        const int d = idx % dim;

        // Copy x to first half
        xh[b * 2 * dim + d] = x_t[idx];
        // Copy h to second half
        xh[b * 2 * dim + dim + d] = h_prev[idx];
    }
}

template<typename Scalar>
__global__ void ConcatKernel(
    const int batch_size,
    const int dim,
    const Scalar* __restrict__ x_t,
    const Scalar* __restrict__ h_prev,
    Scalar* __restrict__ xh) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int b = idx / dim;
        const int d = idx % dim;

        xh[b * 2 * dim + d] = x_t[idx];
        xh[b * 2 * dim + dim + d] = h_prev[idx];
    }
}

// Fused tanh + gate kernel: h = tanh(Wxh + b), output = h * silu(z)
__global__ void FusedTanhGateKernel_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ Wxh,     // [B, dim] - GEMM result
    const __nv_bfloat16* __restrict__ b,       // [dim] - bias
    const __nv_bfloat16* __restrict__ z,       // [B, dim] - gate input
    __nv_bfloat16* __restrict__ h_out,         // [B, dim] - hidden state
    __nv_bfloat16* __restrict__ output,        // [B, dim] - output
    __nv_bfloat16* __restrict__ v_cache) {     // [B, dim] - pre-activation cache

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // Add bias
        __nv_bfloat16 sum = bf16_add(Wxh[idx], b[d]);

        // Store pre-activation
        if (v_cache) v_cache[idx] = sum;

        // tanh (need f32)
        float val = __bfloat162float(sum);
        float h_val = tanhf(val);
        h_out[idx] = __float2bfloat16(h_val);

        // Gate: output = h * silu(z)
        float z_val = __bfloat162float(z[idx]);
        float sigmoid_z = 1.0f / (1.0f + __expf(-z_val));
        float silu_z = z_val * sigmoid_z;

        output[idx] = __float2bfloat16(h_val * silu_z);
    }
}

template<typename Scalar>
__global__ void FusedTanhGateKernel(
    const int batch_size,
    const int dim,
    const Scalar* __restrict__ Wxh,
    const Scalar* __restrict__ b,
    const Scalar* __restrict__ z,
    Scalar* __restrict__ h_out,
    Scalar* __restrict__ output,
    Scalar* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float val = static_cast<float>(Wxh[idx]) + static_cast<float>(b[d]);
        if (v_cache) v_cache[idx] = static_cast<Scalar>(val);

        float h_val = tanhf(val);
        h_out[idx] = static_cast<Scalar>(h_val);

        float z_val = static_cast<float>(z[idx]);
        float sigmoid_z = 1.0f / (1.0f + expf(-z_val));
        float silu_z = z_val * sigmoid_z;

        output[idx] = static_cast<Scalar>(h_val * silu_z);
    }
}

// =============================================================================
// E56 Backward Kernels
// =============================================================================

// Backward through gate: output = h * silu(z)
// Returns dh and dz
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

template<typename Scalar>
__global__ void MambaGateBackward(
    const int batch_size,
    const int dim,
    const Scalar* __restrict__ h,
    const Scalar* __restrict__ z,
    const Scalar* __restrict__ d_output,
    Scalar* __restrict__ dh,
    Scalar* __restrict__ dz) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        float h_val = static_cast<float>(h[idx]);
        float z_val = static_cast<float>(z[idx]);
        float dout = static_cast<float>(d_output[idx]);

        float sigmoid_z = 1.0f / (1.0f + expf(-z_val));
        float silu_z = z_val * sigmoid_z;
        float dsilu = sigmoid_z * (1.0f + z_val * (1.0f - sigmoid_z));

        dh[idx] = static_cast<Scalar>(dout * silu_z);
        dz[idx] = static_cast<Scalar>(dout * h_val * dsilu);
    }
}

// Backward through tanh: computes dv from dh + dh_recurrent, accumulates db
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

        float grad;
        if (dh_recurrent) {
            grad = __bfloat162float(dh[idx]) + __bfloat162float(dh_recurrent[idx]);
        } else {
            grad = __bfloat162float(dh[idx]);
        }

        float h = tanhf(__bfloat162float(v[idx]));
        float dtanh = 1.0f - h * h;
        float dv_val = grad * dtanh;

        dv[idx] = __float2bfloat16(dv_val);
        atomicAdd(&db[d], dv_val);
    }
}

template<typename Scalar>
__global__ void TanhBackwardKernel(
    const int batch_size,
    const int dim,
    const Scalar* __restrict__ v,
    const Scalar* __restrict__ dh,
    const Scalar* __restrict__ dh_recurrent,
    Scalar* __restrict__ dv,
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
        dv[idx] = static_cast<Scalar>(dv_val);

        atomicAdd(&db[d], dv_val);
    }
}

// Split dxh gradient back into dx and dh_recurrent
// dxh [B, 2*dim] -> dx [B, dim] (first half), dh_recurrent [B, dim] (second half)
__global__ void SplitGradKernel_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ dxh,      // [B, 2*dim]
    __nv_bfloat16* __restrict__ dx,              // [B, dim]
    __nv_bfloat16* __restrict__ dh_recurrent) { // [B, dim]

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int b = idx / dim;
        const int d = idx % dim;

        dx[idx] = dxh[b * 2 * dim + d];
        dh_recurrent[idx] = dxh[b * 2 * dim + dim + d];
    }
}

template<typename Scalar>
__global__ void SplitGradKernel(
    const int batch_size,
    const int dim,
    const Scalar* __restrict__ dxh,
    Scalar* __restrict__ dx,
    Scalar* __restrict__ dh_recurrent) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int b = idx / dim;
        const int d = idx % dim;

        dx[idx] = dxh[b * 2 * dim + d];
        dh_recurrent[idx] = dxh[b * 2 * dim + dim + d];
    }
}

// Vector add inplace
__global__ void VectorAddInplace_BF16(
    const int n,
    __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = bf16_add(a[idx], b[idx]);
    }
}

template<typename Scalar>
__global__ void VectorAddInplace(const int n, Scalar* __restrict__ a, const Scalar* __restrict__ b) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = static_cast<Scalar>(static_cast<float>(a[idx]) + static_cast<float>(b[idx]));
    }
}

// Copy float to Scalar
template<typename Scalar>
__global__ void CopyFloatToScalar(const int n, const float* __restrict__ src, Scalar* __restrict__ dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = static_cast<Scalar>(src[idx]);
    }
}

}  // anonymous namespace


namespace hasty {
namespace v0 {
namespace elman_ladder {

// =============================================================================
// E56 Concat Elman Forward - BF16 Specialization
// =============================================================================

template<>
E56ConcatElmanForward<__nv_bfloat16>::E56ConcatElmanForward(
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
void E56ConcatElmanForward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W,       // [dim, 2*dim]
    const __nv_bfloat16* b,       // [dim]
    const __nv_bfloat16* x,       // [T, B, dim] pre-activated input
    const __nv_bfloat16* z,       // [T, B, dim] gate input
    __nv_bfloat16* h,             // [T+1, B, dim] hidden states
    __nv_bfloat16* output,        // [T, B, dim] output
    __nv_bfloat16* v,             // [T, B, dim] pre-activation cache
    __nv_bfloat16* workspace) {   // [B*2*dim + B*dim] for xh_concat, Wxh

    static const __nv_bfloat16 alpha = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Workspace layout: xh_concat [B, 2*dim], Wxh [B, dim]
    __nv_bfloat16* xh_concat = workspace;
    __nv_bfloat16* Wxh = workspace + batch_size_ * 2 * dim_;

    // Process each timestep
    for (int t = 0; t < steps; ++t) {
        const __nv_bfloat16* x_t = x + t * BD;
        const __nv_bfloat16* h_prev = h + t * BD;
        const __nv_bfloat16* z_t = z + t * BD;
        __nv_bfloat16* h_t = h + (t + 1) * BD;
        __nv_bfloat16* out_t = output + t * BD;
        __nv_bfloat16* v_t = training_ ? (v + t * BD) : nullptr;

        // Concatenate [x_t; h_prev] -> xh_concat [B, 2*dim]
        ConcatKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, x_t, h_prev, xh_concat);

        // GEMM: Wxh = xh_concat @ W.T  (single GEMM!)
        // W is [dim, 2*dim], xh_concat is [B, 2*dim], result is [B, dim]
        // cuBLAS: C = alpha * A * B + beta * C
        // We want: [B, dim] = [B, 2*dim] @ [2*dim, dim]
        // With column-major: C(dim, B) = W.T(dim, 2*dim) @ xh_concat.T(2*dim, B)
        blas<__nv_bfloat16>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, 2 * dim_,
            &alpha,
            W, 2 * dim_,                // W stored as [dim, 2*dim] row-major -> [2*dim, dim] col-major
            xh_concat, 2 * dim_,        // xh_concat [B, 2*dim] row-major -> [2*dim, B] col-major
            &beta_zero,
            Wxh, dim_);

        // Fused: h_t = tanh(Wxh + b), output = h_t * silu(z_t)
        FusedTanhGateKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, Wxh, b, z_t, h_t, out_t, v_t);
    }
}

// =============================================================================
// E56 Concat Elman Backward - BF16 Specialization
// =============================================================================

template<>
E56ConcatElmanBackward<__nv_bfloat16>::E56ConcatElmanBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void E56ConcatElmanBackward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W,        // [dim, 2*dim]
    const __nv_bfloat16* x,        // [T, B, dim]
    const __nv_bfloat16* z,        // [T, B, dim]
    const __nv_bfloat16* h,        // [T+1, B, dim]
    const __nv_bfloat16* v,        // [T, B, dim]
    const __nv_bfloat16* d_output, // [T, B, dim]
    __nv_bfloat16* dx,             // [T, B, dim]
    __nv_bfloat16* dz,             // [T, B, dim]
    __nv_bfloat16* dW,             // [dim, 2*dim]
    __nv_bfloat16* db,             // [dim]
    __nv_bfloat16* workspace) {

    static const __nv_bfloat16 alpha_one = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);
    static const __nv_bfloat16 beta_one = __float2bfloat16(1.0f);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Workspace layout:
    // dv_all: [T, B, dim] = T * BD
    // dh: [B, dim] = BD
    // dh_recurrent: [B, dim] = BD
    // dxh: [B, 2*dim] = 2*BD
    // xh_concat: [B, 2*dim] = 2*BD (for dW computation)
    // db_float: [dim] floats = ceil(dim * sizeof(float) / sizeof(bf16))
    __nv_bfloat16* dv_all = workspace;
    __nv_bfloat16* dh = workspace + steps * BD;
    __nv_bfloat16* dh_recurrent = workspace + (steps + 1) * BD;
    __nv_bfloat16* dxh = workspace + (steps + 2) * BD;
    __nv_bfloat16* xh_concat = workspace + (steps + 4) * BD;
    float* db_float = reinterpret_cast<float*>(workspace + (steps + 6) * BD);

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW, 0, dim_ * 2 * dim_ * sizeof(__nv_bfloat16), stream_);

    // BPTT loop
    for (int t = steps - 1; t >= 0; --t) {
        const __nv_bfloat16* v_t = v + t * BD;
        const __nv_bfloat16* h_t = h + (t + 1) * BD;
        const __nv_bfloat16* h_prev = h + t * BD;
        const __nv_bfloat16* x_t = x + t * BD;
        const __nv_bfloat16* z_t = z + t * BD;
        const __nv_bfloat16* d_out_t = d_output + t * BD;
        __nv_bfloat16* dv_t = dv_all + t * BD;
        __nv_bfloat16* dz_t = dz + t * BD;
        __nv_bfloat16* dx_t = dx + t * BD;

        // Backward through gate: d_output -> dh, dz
        MambaGateBackward_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, z_t, d_out_t, dh, dz_t);

        // Add recurrent gradient
        if (t < steps - 1) {
            VectorAddInplace_BF16<<<num_blocks, block_size, 0, stream_>>>(BD, dh, dh_recurrent);
        }

        // Backward through tanh: (dh + dh_recurrent) -> dv, accumulate db
        TanhBackwardKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, v_t, dh, nullptr, dv_t, db_float);

        // Backward through GEMM: dv -> dxh, accumulate dW
        // Forward was: Wxh = xh_concat @ W.T
        // dxh = dv @ W (to get gradient w.r.t. xh_concat)
        // dW += dv.T @ xh_concat
        blas<__nv_bfloat16>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            2 * dim_, batch_size_, dim_,
            &alpha_one,
            W, 2 * dim_,
            dv_t, dim_,
            &beta_zero,
            dxh, 2 * dim_);

        // Split dxh -> dx_t, dh_recurrent
        SplitGradKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, dxh, dx_t, dh_recurrent);

        // Reconstruct xh_concat for dW accumulation
        ConcatKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, x_t, h_prev, xh_concat);

        // Accumulate dW: dW += dv_t.T @ xh_concat
        // dW is [dim, 2*dim], dv_t is [B, dim], xh_concat is [B, 2*dim]
        blas<__nv_bfloat16>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_T,
            2 * dim_, dim_, batch_size_,
            &alpha_one,
            xh_concat, 2 * dim_,
            dv_t, dim_,
            &beta_one,
            dW, 2 * dim_);
    }

    // Copy float gradients to bf16
    CopyFloatToScalar<__nv_bfloat16><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_float, db);
}

// =============================================================================
// Generic Template Implementations (float, half, double)
// =============================================================================

template<typename Scalar>
E56ConcatElmanForward<Scalar>::E56ConcatElmanForward(
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

template<typename Scalar>
void E56ConcatElmanForward<Scalar>::Run(
    int steps,
    const Scalar* W,
    const Scalar* b,
    const Scalar* x,
    const Scalar* z,
    Scalar* h,
    Scalar* output,
    Scalar* v,
    Scalar* workspace) {

    static const Scalar alpha = static_cast<Scalar>(1.0);
    static const Scalar beta_zero = static_cast<Scalar>(0.0);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    Scalar* xh_concat = workspace;
    Scalar* Wxh = workspace + batch_size_ * 2 * dim_;

    for (int t = 0; t < steps; ++t) {
        const Scalar* x_t = x + t * BD;
        const Scalar* h_prev = h + t * BD;
        const Scalar* z_t = z + t * BD;
        Scalar* h_t = h + (t + 1) * BD;
        Scalar* out_t = output + t * BD;
        Scalar* v_t = training_ ? (v + t * BD) : nullptr;

        ConcatKernel<Scalar><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, x_t, h_prev, xh_concat);

        blas<Scalar>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, 2 * dim_,
            &alpha,
            W, 2 * dim_,
            xh_concat, 2 * dim_,
            &beta_zero,
            Wxh, dim_);

        FusedTanhGateKernel<Scalar><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, Wxh, b, z_t, h_t, out_t, v_t);
    }
}

template<typename Scalar>
E56ConcatElmanBackward<Scalar>::E56ConcatElmanBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename Scalar>
void E56ConcatElmanBackward<Scalar>::Run(
    int steps,
    const Scalar* W,
    const Scalar* x,
    const Scalar* z,
    const Scalar* h,
    const Scalar* v,
    const Scalar* d_output,
    Scalar* dx,
    Scalar* dz,
    Scalar* dW,
    Scalar* db,
    Scalar* workspace) {

    static const Scalar alpha = static_cast<Scalar>(1.0);
    static const Scalar beta_zero = static_cast<Scalar>(0.0);
    static const Scalar beta_one = static_cast<Scalar>(1.0);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    Scalar* dv_all = workspace;
    Scalar* dh = workspace + steps * BD;
    Scalar* dh_recurrent = workspace + (steps + 1) * BD;
    Scalar* dxh = workspace + (steps + 2) * BD;
    Scalar* xh_concat = workspace + (steps + 4) * BD;
    float* db_float = reinterpret_cast<float*>(workspace + (steps + 6) * BD);

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(Scalar), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW, 0, dim_ * 2 * dim_ * sizeof(Scalar), stream_);

    for (int t = steps - 1; t >= 0; --t) {
        const Scalar* v_t = v + t * BD;
        const Scalar* h_t = h + (t + 1) * BD;
        const Scalar* h_prev = h + t * BD;
        const Scalar* x_t = x + t * BD;
        const Scalar* z_t = z + t * BD;
        const Scalar* d_out_t = d_output + t * BD;
        Scalar* dv_t = dv_all + t * BD;
        Scalar* dz_t = dz + t * BD;
        Scalar* dx_t = dx + t * BD;

        MambaGateBackward<Scalar><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, z_t, d_out_t, dh, dz_t);

        if (t < steps - 1) {
            VectorAddInplace<Scalar><<<num_blocks, block_size, 0, stream_>>>(BD, dh, dh_recurrent);
        }

        TanhBackwardKernel<Scalar><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, v_t, dh, nullptr, dv_t, db_float);

        blas<Scalar>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            2 * dim_, batch_size_, dim_,
            &alpha,
            W, 2 * dim_,
            dv_t, dim_,
            &beta_zero,
            dxh, 2 * dim_);

        SplitGradKernel<Scalar><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, dxh, dx_t, dh_recurrent);

        ConcatKernel<Scalar><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, x_t, h_prev, xh_concat);

        blas<Scalar>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_T,
            2 * dim_, dim_, batch_size_,
            &alpha,
            xh_concat, 2 * dim_,
            dv_t, dim_,
            &beta_one,
            dW, 2 * dim_);
    }

    CopyFloatToScalar<Scalar><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_float, db);
}

// Explicit template instantiations
template struct E56ConcatElmanForward<__half>;
template struct E56ConcatElmanForward<float>;
template struct E56ConcatElmanForward<double>;

template struct E56ConcatElmanBackward<__half>;
template struct E56ConcatElmanBackward<float>;
template struct E56ConcatElmanBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
