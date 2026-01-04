// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// Auto Elman - Autonomous hidden state with input-only gating
// h_t = tanh(W_h @ h_{t-1} + b_h)           -- hidden evolves autonomously
// output_t = h_t * silu(W_gate @ x_t + b_gate)  -- input only selects output
//
// Key insight: Tests whether input needs to affect state directly,
// or if it's sufficient for input to just SELECT which hidden dims to output.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#include "hasty/elman_ladder.h"
#include "blas.h"
#include "inline_ops.h"

namespace {

// Kernel: Apply tanh with bias to hidden state
// h_out = tanh(Rh + b_h) where Rh = W_h @ h_prev
template<typename T>
__global__ void AutoTanhKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ Rh,        // [B, dim] W_h @ h_prev
    const T* __restrict__ b_h,       // [dim] bias
    T* __restrict__ h_out,           // [B, dim] output
    T* __restrict__ v_cache) {       // [B, dim] pre-activation cache (optional)

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;
        float val = static_cast<float>(Rh[idx]) + static_cast<float>(b_h[d]);
        if (v_cache) v_cache[idx] = static_cast<T>(val);
        h_out[idx] = static_cast<T>(tanhf(val));
    }
}

// Kernel: Learned gate output
// output = h * silu(W_gate @ x + b_gate)
template<typename T>
__global__ void SelectiveOutputForward(
    const int batch_size,
    const int dim,
    const T* __restrict__ h,            // [B, dim]
    const T* __restrict__ gate_proj,    // [B, dim] pre-computed W_gate @ x
    const T* __restrict__ b_gate,       // [dim]
    T* __restrict__ output,             // [B, dim]
    T* __restrict__ gate_cache) {       // [B, dim] gate_raw for backward

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float h_val = static_cast<float>(h[idx]);
        float gp_val = static_cast<float>(gate_proj[idx]);
        float b_val = static_cast<float>(b_gate[d]);

        float gate_raw = gp_val + b_val;
        float sigmoid_val = 1.0f / (1.0f + expf(-gate_raw));
        float silu_val = gate_raw * sigmoid_val;

        output[idx] = static_cast<T>(h_val * silu_val);
        if (gate_cache) gate_cache[idx] = static_cast<T>(gate_raw);
    }
}

// Backward through selective output
template<typename T>
__global__ void SelectiveOutputBackward(
    const int batch_size,
    const int dim,
    const T* __restrict__ h,
    const T* __restrict__ gate_cache,
    const T* __restrict__ d_output,
    T* __restrict__ dh,
    T* __restrict__ d_gate_proj,
    float* __restrict__ d_b_gate) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float h_val = static_cast<float>(h[idx]);
        float gate_raw = static_cast<float>(gate_cache[idx]);
        float dout = static_cast<float>(d_output[idx]);

        float sigmoid_val = 1.0f / (1.0f + expf(-gate_raw));
        float silu_val = gate_raw * sigmoid_val;
        float dsilu = sigmoid_val * (1.0f + gate_raw * (1.0f - sigmoid_val));

        float dh_val = dout * silu_val;
        float dg_val = dout * h_val * dsilu;

        dh[idx] = static_cast<T>(dh_val);
        d_gate_proj[idx] = static_cast<T>(dg_val);
        atomicAdd(&d_b_gate[d], dg_val);
    }
}

// Backward through tanh
template<typename T>
__global__ void AutoElmanBackwardKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ v,           // [B, dim] pre-activation
    const T* __restrict__ dh,          // [B, dim] gradient from above
    const T* __restrict__ dh_recurrent,// [B, dim] gradient from next timestep (or null)
    T* __restrict__ dv,                // [B, dim] gradient w.r.t. pre-activation
    float* __restrict__ db_h) {        // [dim] gradient w.r.t. bias

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

        atomicAdd(&db_h[d], dv_val);
    }
}

// Add vectors in-place: a += b
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
// Auto Elman Forward
// =============================================================================

template<typename T>
AutoElmanForward<T>::AutoElmanForward(
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
void AutoElmanForward<T>::Run(
    int steps,
    const T* W_h,       // [dim, dim] hidden-to-hidden
    const T* W_gate,    // [dim, dim] input gate projection
    const T* b_h,       // [dim] hidden bias
    const T* b_gate,    // [dim] gate bias
    const T* x,         // [T, B, dim] input (only used for gating)
    T* h,               // [T+1, B, dim] hidden states
    T* output,          // [T, B, dim] output
    T* v,               // [T, B, dim] pre-activation cache
    T* gate_cache) {    // [T, B, dim] gate cache

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Pre-compute W_gate @ x for ALL timesteps (input only affects gate)
    T* gate_proj;
    T* tmp_Rh;
    cudaMalloc(&gate_proj, steps * BD * sizeof(T));
    cudaMalloc(&tmp_Rh, BD * sizeof(T));

    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha,
        W_gate, dim_,
        x, dim_,
        &beta_zero,
        gate_proj, dim_);

    // Per-timestep: autonomous hidden update + input-gated output
    for (int t = 0; t < steps; ++t) {
        const T* h_prev = h + t * BD;
        const T* gate_proj_t = gate_proj + t * BD;
        T* h_t = h + (t + 1) * BD;
        T* out_t = output + t * BD;
        T* v_t = training_ ? (v + t * BD) : nullptr;
        T* gate_t = training_ ? (gate_cache + t * BD) : nullptr;

        // tmp_Rh = h_prev @ W_h.T
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha,
            W_h, dim_,
            h_prev, dim_,
            &beta_zero,
            tmp_Rh, dim_);

        // h_t = tanh(tmp_Rh + b_h)  -- NO INPUT in state update
        AutoTanhKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, tmp_Rh, b_h, h_t, v_t);

        // output = h * silu(gate_proj + b_gate)  -- input only affects output
        SelectiveOutputForward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, gate_proj_t, b_gate, out_t, gate_t);
    }

    cudaFree(gate_proj);
    cudaFree(tmp_Rh);
}

// =============================================================================
// Auto Elman Backward
// =============================================================================

template<typename T>
AutoElmanBackward<T>::AutoElmanBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void AutoElmanBackward<T>::Run(
    int steps,
    const T* W_h,
    const T* W_gate,
    const T* x,
    const T* h,
    const T* v,
    const T* gate_cache,
    const T* d_output,
    T* dx,              // gradient flows only through gate
    T* dW_h,
    T* dW_gate,
    T* db_h,
    T* db_gate,
    T* workspace) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Workspace: [dv_all: T*BD] [d_gate_proj_all: T*BD] [dh: BD] [dh_recurrent: BD]
    //            [db_h_float: dim] [db_gate_float: dim]
    T* dv_all = workspace;
    T* d_gate_proj_all = workspace + steps * BD;
    T* dh = workspace + 2 * steps * BD;
    T* dh_recurrent = workspace + (2 * steps + 1) * BD;
    float* db_h_float = reinterpret_cast<float*>(workspace + (2 * steps + 2) * BD);
    float* db_gate_float = db_h_float + dim_;

    // Initialize
    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(T), stream_);
    cudaMemsetAsync(db_h_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_gate_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW_h, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dW_gate, 0, dim_ * dim_ * sizeof(T), stream_);

    // BPTT loop
    for (int t = steps - 1; t >= 0; --t) {
        const T* v_t = v + t * BD;
        const T* h_t = h + (t + 1) * BD;
        const T* gate_cache_t = gate_cache + t * BD;
        const T* d_out_t = d_output + t * BD;
        T* dv_t = dv_all + t * BD;
        T* d_gate_proj_t = d_gate_proj_all + t * BD;

        // Backward through selective output
        SelectiveOutputBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, gate_cache_t, d_out_t,
            dh, d_gate_proj_t, db_gate_float);

        // Add recurrent gradient
        VectorAddInplace<T><<<num_blocks, block_size, 0, stream_>>>(BD, dh, dh_recurrent);

        // Backward through tanh
        AutoElmanBackwardKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, v_t, dh, nullptr, dv_t, db_h_float);

        // dh_recurrent = W_h @ dv (gradient flows through h_prev)
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

    // dx = W_gate @ d_gate_proj_all (input gradient ONLY from gate path)
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha,
        W_gate, dim_,
        d_gate_proj_all, dim_,
        &beta_zero,  // Not beta_one - dx only comes from gate
        dx, dim_);

    // dW_h = h^T @ dv_all
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha,
        h, dim_,
        dv_all, dim_,
        &beta_one,
        dW_h, dim_);

    // dW_gate = x^T @ d_gate_proj_all
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha,
        x, dim_,
        d_gate_proj_all, dim_,
        &beta_one,
        dW_gate, dim_);

    // Copy float gradients
    CopyFloatToT<T><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_h_float, db_h);
    CopyFloatToT<T><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_gate_float, db_gate);
}

// Explicit template instantiations
template struct AutoElmanForward<__half>;
template struct AutoElmanForward<__nv_bfloat16>;
template struct AutoElmanForward<float>;
template struct AutoElmanForward<double>;

template struct AutoElmanBackward<__half>;
template struct AutoElmanBackward<__nv_bfloat16>;
template struct AutoElmanBackward<float>;
template struct AutoElmanBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
