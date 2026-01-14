// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E12: Selective Gated Elman
//
// Minimal change from E1: gate depends on hidden state, not just input.
//
// E1:  output = h * silu(z)                    # gate = silu(z)
// E12: output = h * sigmoid(z + W_g @ h)       # gate = sigmoid(z + W_g @ h)
//
// This makes the gate "selective" - it depends on what the model has computed
// (hidden state h), similar to Mamba2's input-dependent gating.
//
// Architecture per timestep:
// h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b)    # Same as E1
// g_t = W_g @ h_t                              # NEW: project h for gating
// gate_t = sigmoid(z_t + g_t)                  # NEW: selective gate
// output_t = h_t * gate_t                      # Gated output

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

// bf16_mul not currently used in this kernel but kept for consistency

// =============================================================================
// Forward Kernels
// =============================================================================

// BF16-optimized: Fused Wx + Rh + bias + tanh (same as E1)
__global__ void FusedTanhKernel_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ Wx,
    const __nv_bfloat16* __restrict__ Rh,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ h_out,
    __nv_bfloat16* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;
        __nv_bfloat16 sum = bf16_add(bf16_add(Wx[idx], Rh[idx]), b[d]);
        if (v_cache) v_cache[idx] = sum;
        float val = __bfloat162float(sum);
        h_out[idx] = __float2bfloat16(tanhf(val));
    }
}

template<typename T>
__global__ void FusedTanhKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ Wx,
    const T* __restrict__ Rh,
    const T* __restrict__ b,
    T* __restrict__ h_out,
    T* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;
        float val = static_cast<float>(Wx[idx]) + static_cast<float>(Rh[idx]) + static_cast<float>(b[d]);
        if (v_cache) v_cache[idx] = static_cast<T>(val);
        h_out[idx] = static_cast<T>(tanhf(val));
    }
}

// BF16-optimized: Selective gate: output = h * sigmoid(z + Gh)
// Gh = W_g @ h is pre-computed via cuBLAS
__global__ void SelectiveGateForward_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ h,
    const __nv_bfloat16* __restrict__ z,
    const __nv_bfloat16* __restrict__ Gh,  // W_g @ h, pre-computed
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ gate_cache) {  // Cache gate pre-activation for backward

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        float h_val = __bfloat162float(h[idx]);
        float z_val = __bfloat162float(z[idx]);
        float gh_val = __bfloat162float(Gh[idx]);

        // gate_pre = z + W_g @ h
        float gate_pre = z_val + gh_val;

        // Cache for backward
        if (gate_cache) gate_cache[idx] = __float2bfloat16(gate_pre);

        // gate = sigmoid(gate_pre)
        float gate = 1.0f / (1.0f + __expf(-gate_pre));

        // output = h * gate
        output[idx] = __float2bfloat16(h_val * gate);
    }
}

template<typename T>
__global__ void SelectiveGateForward(
    const int batch_size,
    const int dim,
    const T* __restrict__ h,
    const T* __restrict__ z,
    const T* __restrict__ Gh,
    T* __restrict__ output,
    T* __restrict__ gate_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        float h_val = static_cast<float>(h[idx]);
        float z_val = static_cast<float>(z[idx]);
        float gh_val = static_cast<float>(Gh[idx]);

        float gate_pre = z_val + gh_val;
        if (gate_cache) gate_cache[idx] = static_cast<T>(gate_pre);

        float gate = 1.0f / (1.0f + expf(-gate_pre));
        output[idx] = static_cast<T>(h_val * gate);
    }
}

// =============================================================================
// Backward Kernels
// =============================================================================

// Backward through tanh with gradient accumulation
__global__ void TanhBackwardKernel_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ v,       // pre-activation
    const __nv_bfloat16* __restrict__ dh,      // gradient from output
    const __nv_bfloat16* __restrict__ dh_recurrent,  // gradient from next timestep
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

        // dtanh: need f32 for tanh computation
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

// Backward through selective gate
// Forward: output = h * sigmoid(z + Gh)
// Backward: d_h = d_output * gate
//           d_gate_pre = d_output * h * gate * (1 - gate)
//           d_z = d_gate_pre
//           d_Gh = d_gate_pre (for W_g gradient via cuBLAS)
__global__ void SelectiveGateBackward_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ h,
    const __nv_bfloat16* __restrict__ gate_cache,  // z + Gh (pre-sigmoid)
    const __nv_bfloat16* __restrict__ d_output,
    __nv_bfloat16* __restrict__ d_h,      // Output gradient for h (NOT accumulated)
    __nv_bfloat16* __restrict__ d_z,
    __nv_bfloat16* __restrict__ d_Gh) {   // For W_g gradient

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        float h_val = __bfloat162float(h[idx]);
        float gate_pre = __bfloat162float(gate_cache[idx]);
        float d_out = __bfloat162float(d_output[idx]);

        // Recompute gate
        float gate = 1.0f / (1.0f + __expf(-gate_pre));

        // d_h = d_output * gate (NOT accumulated - will be added to tanh grad later)
        d_h[idx] = __float2bfloat16(d_out * gate);

        // d_gate_pre = d_output * h * gate * (1 - gate)
        float d_gate_pre = d_out * h_val * gate * (1.0f - gate);

        // d_z = d_gate_pre
        d_z[idx] = __float2bfloat16(d_gate_pre);

        // d_Gh = d_gate_pre (used for W_g gradient via cuBLAS)
        d_Gh[idx] = __float2bfloat16(d_gate_pre);
    }
}

template<typename T>
__global__ void SelectiveGateBackward(
    const int batch_size,
    const int dim,
    const T* __restrict__ h,
    const T* __restrict__ gate_cache,
    const T* __restrict__ d_output,
    T* __restrict__ d_h,
    T* __restrict__ d_z,
    T* __restrict__ d_Gh) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        float h_val = static_cast<float>(h[idx]);
        float gate_pre = static_cast<float>(gate_cache[idx]);
        float d_out = static_cast<float>(d_output[idx]);

        float gate = 1.0f / (1.0f + expf(-gate_pre));

        d_h[idx] = static_cast<T>(d_out * gate);

        float d_gate_pre = d_out * h_val * gate * (1.0f - gate);
        d_z[idx] = static_cast<T>(d_gate_pre);
        d_Gh[idx] = static_cast<T>(d_gate_pre);
    }
}

// BF16-optimized vector add inplace
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
// SelectiveGatedElmanForward - BF16 Specialization
// =============================================================================

template<>
SelectiveGatedElmanForward<__nv_bfloat16>::SelectiveGatedElmanForward(
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
void SelectiveGatedElmanForward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_x,
    const __nv_bfloat16* W_h,
    const __nv_bfloat16* W_g,   // [dim, dim] NEW: gate projection
    const __nv_bfloat16* b,
    const __nv_bfloat16* x,
    const __nv_bfloat16* z,
    __nv_bfloat16* h,
    __nv_bfloat16* output,
    __nv_bfloat16* v,           // tanh pre-activation cache
    __nv_bfloat16* gate_cache,  // gate pre-activation cache
    __nv_bfloat16* workspace) {

    static const __nv_bfloat16 alpha = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Workspace layout: [tmp_Wx: T*BD] [tmp_Rh: BD] [tmp_Gh: BD]
    __nv_bfloat16* tmp_Wx = workspace;
    __nv_bfloat16* tmp_Rh = workspace + steps * BD;
    __nv_bfloat16* tmp_Gh = tmp_Rh + BD;

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

    // Process each timestep
    for (int t = 0; t < steps; ++t) {
        const __nv_bfloat16* Wx_t = tmp_Wx + t * BD;
        const __nv_bfloat16* h_prev = h + t * BD;
        const __nv_bfloat16* z_t = z + t * BD;
        __nv_bfloat16* h_t = h + (t + 1) * BD;
        __nv_bfloat16* out_t = output + t * BD;
        __nv_bfloat16* v_t = training_ ? (v + t * BD) : nullptr;
        __nv_bfloat16* gate_t = training_ ? (gate_cache + t * BD) : nullptr;

        // tmp_Rh = W_h @ h_prev
        blas<__nv_bfloat16>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha,
            W_h, dim_,
            h_prev, dim_,
            &beta_zero,
            tmp_Rh, dim_);

        // h_t = tanh(Wx_t + tmp_Rh + b)
        FusedTanhKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, Wx_t, tmp_Rh, b, h_t, v_t);

        // tmp_Gh = W_g @ h_t (NEW: project hidden state for gating)
        blas<__nv_bfloat16>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha,
            W_g, dim_,
            h_t, dim_,
            &beta_zero,
            tmp_Gh, dim_);

        // output = h * sigmoid(z + Gh) (NEW: selective gate)
        SelectiveGateForward_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, z_t, tmp_Gh, out_t, gate_t);
    }
}

// =============================================================================
// SelectiveGatedElmanBackward - BF16 Specialization
// =============================================================================

template<>
SelectiveGatedElmanBackward<__nv_bfloat16>::SelectiveGatedElmanBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void SelectiveGatedElmanBackward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_x,
    const __nv_bfloat16* W_h,
    const __nv_bfloat16* W_g,
    const __nv_bfloat16* x,
    const __nv_bfloat16* z,
    const __nv_bfloat16* h,
    const __nv_bfloat16* v,
    const __nv_bfloat16* gate_cache,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* dx,
    __nv_bfloat16* dz,
    __nv_bfloat16* dW_x,
    __nv_bfloat16* dW_h,
    __nv_bfloat16* dW_g,
    __nv_bfloat16* db,
    __nv_bfloat16* workspace) {

    static const __nv_bfloat16 alpha_one = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);
    static const __nv_bfloat16 beta_one = __float2bfloat16(1.0f);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Workspace layout:
    // [dv_all: T*BD] [dh_gate: BD] [dh_recurrent: BD] [d_Gh: BD] [db_float: dim]
    __nv_bfloat16* dv_all = workspace;
    __nv_bfloat16* dh_gate = workspace + steps * BD;
    __nv_bfloat16* dh_recurrent = dh_gate + BD;
    __nv_bfloat16* d_Gh = dh_recurrent + BD;
    float* db_float = reinterpret_cast<float*>(d_Gh + BD);

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW_x, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_h, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_g, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);

    // BPTT loop
    for (int t = steps - 1; t >= 0; --t) {
        const __nv_bfloat16* v_t = v + t * BD;
        const __nv_bfloat16* h_t = h + (t + 1) * BD;
        const __nv_bfloat16* gate_t = gate_cache + t * BD;
        const __nv_bfloat16* d_out_t = d_output + t * BD;
        __nv_bfloat16* dv_t = dv_all + t * BD;
        __nv_bfloat16* dz_t = dz + t * BD;

        // Backward through selective gate
        // Computes: dh_gate (grad w.r.t. h from gate), dz, d_Gh
        SelectiveGateBackward_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, gate_t, d_out_t, dh_gate, dz_t, d_Gh);

        // d_W_g += d_Gh @ h_t.T
        blas<__nv_bfloat16>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_,
            &alpha_one,
            d_Gh, dim_,
            h_t, dim_,
            &beta_one,
            dW_g, dim_);

        // dh_gate += W_g.T @ d_Gh (backprop through W_g @ h)
        blas<__nv_bfloat16>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha_one,
            W_g, dim_,
            d_Gh, dim_,
            &beta_one,
            dh_gate, dim_);

        // Add recurrent gradient from previous iteration
        VectorAddInplace_BF16<<<num_blocks, block_size, 0, stream_>>>(BD, dh_gate, dh_recurrent);

        // Backward through tanh
        TanhBackwardKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, v_t, dh_gate, nullptr, dv_t, db_float);

        // dh_recurrent = W_h @ dv (for next iteration)
        if (t > 0) {
            blas<__nv_bfloat16>::gemm(
                blas_handle_,
                CUBLAS_OP_N, CUBLAS_OP_N,
                dim_, batch_size_, dim_,
                &alpha_one,
                W_h, dim_,
                dv_t, dim_,
                &beta_zero,
                dh_recurrent, dim_);
        }
    }

    // Batch GEMMs for weight gradients
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_x, dim_,
        dv_all, dim_,
        &beta_zero,
        dx, dim_);

    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        dv_all, dim_,
        &beta_one,
        dW_x, dim_);

    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        h, dim_,
        dv_all, dim_,
        &beta_one,
        dW_h, dim_);

    // Copy float gradients to bf16
    CopyFloatToT<__nv_bfloat16><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_float, db);
}

// =============================================================================
// Generic Template Implementations (float, half, double)
// =============================================================================

template<typename T>
SelectiveGatedElmanForward<T>::SelectiveGatedElmanForward(
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
void SelectiveGatedElmanForward<T>::Run(
    int steps,
    const T* W_x,
    const T* W_h,
    const T* W_g,
    const T* b,
    const T* x,
    const T* z,
    T* h,
    T* output,
    T* v,
    T* gate_cache,
    T* workspace) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    T* tmp_Wx = workspace;
    T* tmp_Rh = workspace + steps * BD;
    T* tmp_Gh = tmp_Rh + BD;

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
        T* gate_t = training_ ? (gate_cache + t * BD) : nullptr;

        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha,
            W_h, dim_,
            h_prev, dim_,
            &beta_zero,
            tmp_Rh, dim_);

        FusedTanhKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, Wx_t, tmp_Rh, b, h_t, v_t);

        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha,
            W_g, dim_,
            h_t, dim_,
            &beta_zero,
            tmp_Gh, dim_);

        SelectiveGateForward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, z_t, tmp_Gh, out_t, gate_t);
    }
}

template<typename T>
SelectiveGatedElmanBackward<T>::SelectiveGatedElmanBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void SelectiveGatedElmanBackward<T>::Run(
    int steps,
    const T* W_x,
    const T* W_h,
    const T* W_g,
    const T* x,
    const T* z,
    const T* h,
    const T* v,
    const T* gate_cache,
    const T* d_output,
    T* dx,
    T* dz,
    T* dW_x,
    T* dW_h,
    T* dW_g,
    T* db,
    T* workspace) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    T* dv_all = workspace;
    T* dh_gate = workspace + steps * BD;
    T* dh_recurrent = dh_gate + BD;
    T* d_Gh = dh_recurrent + BD;
    float* db_float = reinterpret_cast<float*>(d_Gh + BD);

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(T), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW_x, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dW_h, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dW_g, 0, dim_ * dim_ * sizeof(T), stream_);

    for (int t = steps - 1; t >= 0; --t) {
        const T* v_t = v + t * BD;
        const T* h_t = h + (t + 1) * BD;
        const T* gate_t = gate_cache + t * BD;
        const T* d_out_t = d_output + t * BD;
        T* dv_t = dv_all + t * BD;
        T* dz_t = dz + t * BD;

        SelectiveGateBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, gate_t, d_out_t, dh_gate, dz_t, d_Gh);

        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_,
            &alpha,
            d_Gh, dim_,
            h_t, dim_,
            &beta_one,
            dW_g, dim_);

        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha,
            W_g, dim_,
            d_Gh, dim_,
            &beta_one,
            dh_gate, dim_);

        VectorAddInplace<T><<<num_blocks, block_size, 0, stream_>>>(BD, dh_gate, dh_recurrent);

        TanhBackwardKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, v_t, dh_gate, nullptr, dv_t, db_float);

        if (t > 0) {
            blas<T>::gemm(
                blas_handle_,
                CUBLAS_OP_N, CUBLAS_OP_N,
                dim_, batch_size_, dim_,
                &alpha,
                W_h, dim_,
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
        dv_all, dim_,
        &beta_one,
        dW_h, dim_);

    CopyFloatToT<T><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_float, db);
}

// Explicit template instantiations
template struct SelectiveGatedElmanForward<__half>;
template struct SelectiveGatedElmanForward<float>;
template struct SelectiveGatedElmanForward<double>;

template struct SelectiveGatedElmanBackward<__half>;
template struct SelectiveGatedElmanBackward<float>;
template struct SelectiveGatedElmanBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
