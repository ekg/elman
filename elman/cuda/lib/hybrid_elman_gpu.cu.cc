// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// Level 9: Hybrid Elman - Small dense core + large diagonal memory
//
// Architecture:
//   h_core: [core_dim] - dense RNN with full W_h matrix (nonlinear mixing)
//   h_mem:  [mem_dim]  - diagonal memory with learned decay (long-range storage)
//
// Forward:
//   h_core_t = tanh(W_x_core @ x_core_t + W_h @ h_core_prev + b_core)
//   h_mem_t  = a_mem * h_mem_prev + x_mem_t   (elementwise diagonal)
//   out_core = h_core_t * silu(z_core_t)
//   out_mem  = h_mem_t * silu(z_mem_t)
//   output   = [out_core, out_mem]  (concatenated, projected in Python)
//
// Benefit: Large hidden state (core_dim + mem_dim) with O(core_dim²) compute

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

// Kernel: Fused Wx + Rh + bias + tanh for core
template<typename T>
__global__ void CoreTanhKernel(
    const int batch_size,
    const int core_dim,
    const T* __restrict__ Wx,        // [B, core_dim] pre-computed
    const T* __restrict__ Rh,        // [B, core_dim] W_h @ h_prev
    const T* __restrict__ b,         // [core_dim] bias
    T* __restrict__ h_out,           // [B, core_dim] output
    T* __restrict__ v_cache) {       // [B, core_dim] pre-activation cache

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * core_dim;

    if (idx < total) {
        const int d = idx % core_dim;
        float val = static_cast<float>(Wx[idx]) + static_cast<float>(Rh[idx]) + static_cast<float>(b[d]);
        if (v_cache) v_cache[idx] = static_cast<T>(val);
        h_out[idx] = static_cast<T>(tanhf(val));
    }
}

// Kernel: Memory diagonal update: h_mem_t = a * h_mem_prev + x_mem_t
template<typename T>
__global__ void MemoryUpdateKernel(
    const int batch_size,
    const int mem_dim,
    const T* __restrict__ a,         // [mem_dim] learned decay (sigmoid applied)
    const T* __restrict__ h_prev,    // [B, mem_dim]
    const T* __restrict__ x_mem,     // [B, mem_dim]
    T* __restrict__ h_out) {         // [B, mem_dim]

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * mem_dim;

    if (idx < total) {
        const int d = idx % mem_dim;
        float a_val = static_cast<float>(a[d]);
        // Apply sigmoid to get decay in (0, 1)
        float decay = 1.0f / (1.0f + expf(-a_val));
        float h_prev_val = static_cast<float>(h_prev[idx]);
        float x_val = static_cast<float>(x_mem[idx]);
        h_out[idx] = static_cast<T>(decay * h_prev_val + x_val);
    }
}

// Kernel: Gated output = h * silu(z)
template<typename T>
__global__ void GateForwardKernel(
    const int n,
    const T* __restrict__ h,
    const T* __restrict__ z,
    T* __restrict__ output) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float h_val = static_cast<float>(h[idx]);
        float z_val = static_cast<float>(z[idx]);
        float sigmoid_z = 1.0f / (1.0f + expf(-z_val));
        float silu_z = z_val * sigmoid_z;
        output[idx] = static_cast<T>(h_val * silu_z);
    }
}

// Backward through tanh
template<typename T>
__global__ void CoreTanhBackwardKernel(
    const int batch_size,
    const int core_dim,
    const T* __restrict__ v,           // [B, core_dim] pre-activation
    const T* __restrict__ dh,          // [B, core_dim] gradient
    const T* __restrict__ dh_recurrent,// [B, core_dim] from next timestep
    T* __restrict__ dv,                // [B, core_dim] output gradient
    float* __restrict__ db) {          // [core_dim] bias gradient

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * core_dim;

    if (idx < total) {
        const int d = idx % core_dim;
        float grad = static_cast<float>(dh[idx]);
        if (dh_recurrent) grad += static_cast<float>(dh_recurrent[idx]);
        float h = tanhf(static_cast<float>(v[idx]));
        float dtanh = 1.0f - h * h;
        float dv_val = grad * dtanh;
        dv[idx] = static_cast<T>(dv_val);
        atomicAdd(&db[d], dv_val);
    }
}

// Backward through memory: h_mem_t = a * h_mem_prev + x_mem_t
template<typename T>
__global__ void MemoryBackwardKernel(
    const int batch_size,
    const int mem_dim,
    const T* __restrict__ a,           // [mem_dim] decay logits
    const T* __restrict__ h_prev,      // [B, mem_dim]
    const T* __restrict__ dh,          // [B, mem_dim] gradient
    const T* __restrict__ dh_recurrent,// [B, mem_dim] from next timestep
    T* __restrict__ dx_mem,            // [B, mem_dim] gradient to input
    T* __restrict__ dh_prev_out,       // [B, mem_dim] gradient to h_prev (for next iter)
    float* __restrict__ da) {          // [mem_dim] gradient to decay

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * mem_dim;

    if (idx < total) {
        const int d = idx % mem_dim;
        float grad = static_cast<float>(dh[idx]);
        if (dh_recurrent) grad += static_cast<float>(dh_recurrent[idx]);

        float a_val = static_cast<float>(a[d]);
        float sigmoid_a = 1.0f / (1.0f + expf(-a_val));
        float h_prev_val = static_cast<float>(h_prev[idx]);

        // d/d(x_mem) = 1
        dx_mem[idx] = static_cast<T>(grad);

        // d/d(h_prev) = sigmoid(a)
        dh_prev_out[idx] = static_cast<T>(grad * sigmoid_a);

        // d/d(a) = h_prev * sigmoid(a) * (1 - sigmoid(a)) * grad
        float da_val = grad * h_prev_val * sigmoid_a * (1.0f - sigmoid_a);
        atomicAdd(&da[d], da_val);
    }
}

// Backward through gate
template<typename T>
__global__ void GateBackwardKernel(
    const int n,
    const T* __restrict__ h,
    const T* __restrict__ z,
    const T* __restrict__ d_output,
    T* __restrict__ dh,
    T* __restrict__ dz) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float h_val = static_cast<float>(h[idx]);
        float z_val = static_cast<float>(z[idx]);
        float dout = static_cast<float>(d_output[idx]);

        float sigmoid_z = 1.0f / (1.0f + expf(-z_val));
        float silu_z = z_val * sigmoid_z;
        float dsilu = sigmoid_z * (1.0f + z_val * (1.0f - sigmoid_z));

        dh[idx] = static_cast<T>(dout * silu_z);
        dz[idx] = static_cast<T>(dout * h_val * dsilu);
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
// Hybrid Elman Forward
// =============================================================================

template<typename T>
HybridElmanForward<T>::HybridElmanForward(
    bool training,
    int batch_size,
    int core_dim,
    int mem_dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : training_(training),
      batch_size_(batch_size),
      core_dim_(core_dim),
      mem_dim_(mem_dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void HybridElmanForward<T>::Run(
    int steps,
    const T* W_x_core,   // [core_dim, core_dim]
    const T* W_h,        // [core_dim, core_dim]
    const T* b_core,     // [core_dim]
    const T* a_mem,      // [mem_dim] decay logits
    const T* x_core,     // [T, B, core_dim] pre-activated core input
    const T* z_core,     // [T, B, core_dim] core gate
    const T* x_mem,      // [T, B, mem_dim] memory input
    const T* z_mem,      // [T, B, mem_dim] memory gate
    T* h_core,           // [T+1, B, core_dim] core hidden states
    T* h_mem,            // [T+1, B, mem_dim] memory states
    T* out_core,         // [T, B, core_dim] gated core output
    T* out_mem,          // [T, B, mem_dim] gated memory output
    T* v_core,           // [T, B, core_dim] core pre-activation cache
    T* workspace) {      // [T*B*core_dim + B*core_dim] for tmp_Wx, tmp_Rh

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int B_core = batch_size_ * core_dim_;
    const int B_mem = batch_size_ * mem_dim_;
    const int block_size = 256;
    const int num_blocks_core = (B_core + block_size - 1) / block_size;
    const int num_blocks_mem = (B_mem + block_size - 1) / block_size;

    // Workspace: [tmp_Wx: T*B*core_dim] [tmp_Rh: B*core_dim]
    T* tmp_Wx = workspace;
    T* tmp_Rh = workspace + steps * B_core;

    // Pre-compute W_x @ x_core for all timesteps
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        core_dim_, steps * batch_size_, core_dim_,
        &alpha,
        W_x_core, core_dim_,
        x_core, core_dim_,
        &beta_zero,
        tmp_Wx, core_dim_);

    // Process each timestep
    for (int t = 0; t < steps; ++t) {
        // Core pointers
        const T* Wx_t = tmp_Wx + t * B_core;
        const T* h_core_prev = h_core + t * B_core;
        const T* z_core_t = z_core + t * B_core;
        T* h_core_t = h_core + (t + 1) * B_core;
        T* out_core_t = out_core + t * B_core;
        T* v_core_t = training_ ? (v_core + t * B_core) : nullptr;

        // Memory pointers
        const T* h_mem_prev = h_mem + t * B_mem;
        const T* x_mem_t = x_mem + t * B_mem;
        const T* z_mem_t = z_mem + t * B_mem;
        T* h_mem_t = h_mem + (t + 1) * B_mem;
        T* out_mem_t = out_mem + t * B_mem;

        // Core: tmp_Rh = h_core_prev @ W_h.T
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            core_dim_, batch_size_, core_dim_,
            &alpha,
            W_h, core_dim_,
            h_core_prev, core_dim_,
            &beta_zero,
            tmp_Rh, core_dim_);

        // Core: h_core_t = tanh(Wx_t + tmp_Rh + b_core)
        CoreTanhKernel<T><<<num_blocks_core, block_size, 0, stream_>>>(
            batch_size_, core_dim_, Wx_t, tmp_Rh, b_core, h_core_t, v_core_t);

        // Core gate: out_core = h_core * silu(z_core)
        GateForwardKernel<T><<<num_blocks_core, block_size, 0, stream_>>>(
            B_core, h_core_t, z_core_t, out_core_t);

        // Memory: h_mem_t = sigmoid(a_mem) * h_mem_prev + x_mem_t
        MemoryUpdateKernel<T><<<num_blocks_mem, block_size, 0, stream_>>>(
            batch_size_, mem_dim_, a_mem, h_mem_prev, x_mem_t, h_mem_t);

        // Memory gate: out_mem = h_mem * silu(z_mem)
        GateForwardKernel<T><<<num_blocks_mem, block_size, 0, stream_>>>(
            B_mem, h_mem_t, z_mem_t, out_mem_t);
    }
}

// =============================================================================
// Hybrid Elman Backward
// =============================================================================

template<typename T>
HybridElmanBackward<T>::HybridElmanBackward(
    int batch_size,
    int core_dim,
    int mem_dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      core_dim_(core_dim),
      mem_dim_(mem_dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void HybridElmanBackward<T>::Run(
    int steps,
    const T* W_x_core,
    const T* W_h,
    const T* a_mem,
    const T* x_core,
    const T* z_core,
    const T* x_mem,
    const T* z_mem,
    const T* h_core,
    const T* h_mem,
    const T* v_core,
    const T* d_out_core,
    const T* d_out_mem,
    T* dx_core,
    T* dz_core,
    T* dx_mem,
    T* dz_mem,
    T* dW_x_core,
    T* dW_h,
    T* db_core,
    T* da_mem,
    T* workspace) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int B_core = batch_size_ * core_dim_;
    const int B_mem = batch_size_ * mem_dim_;
    const int block_size = 256;
    const int num_blocks_core = (B_core + block_size - 1) / block_size;
    const int num_blocks_mem = (B_mem + block_size - 1) / block_size;

    // Workspace layout:
    // [dv_core_all: T*B*core_dim] [dh_core: B*core_dim] [dh_core_recurrent: B*core_dim]
    // [dh_mem: B*mem_dim] [dh_mem_recurrent: B*mem_dim] [db_float: core_dim] [da_float: mem_dim]
    T* dv_core_all = workspace;
    T* dh_core = workspace + steps * B_core;
    T* dh_core_recurrent = workspace + (steps + 1) * B_core;
    T* dh_mem = workspace + (steps + 2) * B_core;  // Separate buffer for memory gradients
    T* dh_mem_recurrent = workspace + (steps + 2) * B_core + B_mem;
    float* db_float = reinterpret_cast<float*>(workspace + (steps + 2) * B_core + 2 * B_mem);
    float* da_float = db_float + core_dim_;

    // Initialize
    cudaMemsetAsync(dh_core_recurrent, 0, B_core * sizeof(T), stream_);
    cudaMemsetAsync(dh_mem_recurrent, 0, B_mem * sizeof(T), stream_);
    cudaMemsetAsync(db_float, 0, core_dim_ * sizeof(float), stream_);
    cudaMemsetAsync(da_float, 0, mem_dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW_x_core, 0, core_dim_ * core_dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dW_h, 0, core_dim_ * core_dim_ * sizeof(T), stream_);

    // BPTT loop
    for (int t = steps - 1; t >= 0; --t) {
        // Core pointers
        const T* v_core_t = v_core + t * B_core;
        const T* h_core_t = h_core + (t + 1) * B_core;
        const T* z_core_t = z_core + t * B_core;
        const T* d_out_core_t = d_out_core + t * B_core;
        T* dv_core_t = dv_core_all + t * B_core;
        T* dz_core_t = dz_core + t * B_core;

        // Memory pointers
        const T* h_mem_prev = h_mem + t * B_mem;
        const T* h_mem_t = h_mem + (t + 1) * B_mem;
        const T* z_mem_t = z_mem + t * B_mem;
        const T* d_out_mem_t = d_out_mem + t * B_mem;
        T* dx_mem_t = dx_mem + t * B_mem;
        T* dz_mem_t = dz_mem + t * B_mem;

        // Core: Backward through gate
        GateBackwardKernel<T><<<num_blocks_core, block_size, 0, stream_>>>(
            B_core, h_core_t, z_core_t, d_out_core_t, dh_core, dz_core_t);

        // Add recurrent gradient
        VectorAddInplace<T><<<num_blocks_core, block_size, 0, stream_>>>(
            B_core, dh_core, dh_core_recurrent);

        // Core: Backward through tanh
        CoreTanhBackwardKernel<T><<<num_blocks_core, block_size, 0, stream_>>>(
            batch_size_, core_dim_, v_core_t, dh_core, nullptr, dv_core_t, db_float);

        // Core: dh_core_recurrent = W_h @ dv_core
        if (t > 0) {
            blas<T>::gemm(
                blas_handle_,
                CUBLAS_OP_N, CUBLAS_OP_N,
                core_dim_, batch_size_, core_dim_,
                &alpha,
                W_h, core_dim_,
                dv_core_t, core_dim_,
                &beta_zero,
                dh_core_recurrent, core_dim_);
        }

        // Memory: Backward through gate
        GateBackwardKernel<T><<<num_blocks_mem, block_size, 0, stream_>>>(
            B_mem, h_mem_t, z_mem_t, d_out_mem_t, dh_mem, dz_mem_t);

        // Add recurrent gradient
        VectorAddInplace<T><<<num_blocks_mem, block_size, 0, stream_>>>(
            B_mem, dh_mem, dh_mem_recurrent);

        // Memory: Backward through diagonal update
        MemoryBackwardKernel<T><<<num_blocks_mem, block_size, 0, stream_>>>(
            batch_size_, mem_dim_, a_mem, h_mem_prev, dh_mem,
            nullptr,  // recurrent gradient already added above
            dx_mem_t, dh_mem_recurrent, da_float);
    }

    // Batch GEMMs for core
    // dx_core = W_x_core @ dv_core_all
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        core_dim_, steps * batch_size_, core_dim_,
        &alpha,
        W_x_core, core_dim_,
        dv_core_all, core_dim_,
        &beta_zero,
        dx_core, core_dim_);

    // dW_x_core = x_core^T @ dv_core_all
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        core_dim_, core_dim_, steps * batch_size_,
        &alpha,
        x_core, core_dim_,
        dv_core_all, core_dim_,
        &beta_one,
        dW_x_core, core_dim_);

    // dW_h = h_core^T @ dv_core_all
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        core_dim_, core_dim_, steps * batch_size_,
        &alpha,
        h_core, core_dim_,
        dv_core_all, core_dim_,
        &beta_one,
        dW_h, core_dim_);

    // Copy float gradients
    CopyFloatToT<T><<<(core_dim_ + 255) / 256, 256, 0, stream_>>>(core_dim_, db_float, db_core);
    CopyFloatToT<T><<<(mem_dim_ + 255) / 256, 256, 0, stream_>>>(mem_dim_, da_float, da_mem);
}

// Explicit template instantiations
template struct HybridElmanForward<__half>;
template struct HybridElmanForward<__nv_bfloat16>;
template struct HybridElmanForward<float>;
template struct HybridElmanForward<double>;

template struct HybridElmanBackward<__half>;
template struct HybridElmanBackward<__nv_bfloat16>;
template struct HybridElmanBackward<float>;
template struct HybridElmanBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
