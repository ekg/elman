// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// Diagonal Elman: tanh recurrence with diagonal decay + x-only selective output
// h_t = tanh(W_x @ x_t + α ⊙ h_{t-1} + b)  -- KEEP tanh, diagonal α instead of dense W_h
// output_t = h_t * silu(x_t + b_gate)  -- x-only selective gating
//
// Key differences from X-Gated Elman:
// - KEEP tanh nonlinearity (essential for expressivity!)
// - No dense W_h matrix - just diagonal α (1000x fewer params)
// - Much faster - no W_h @ h GEMM in recurrence loop

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

// Fused tanh recurrence with diagonal decay: h_new = tanh(Wx + α * h_prev + b)
// Key: KEEP tanh for expressivity, but use diagonal α instead of dense W_h
template<typename T>
__global__ void DiagonalRecurrenceKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ Wx,        // [B, dim] pre-computed W_x @ x
    const T* __restrict__ h_prev,    // [B, dim]
    const T* __restrict__ alpha,     // [dim] diagonal decay (pre-sigmoid)
    const T* __restrict__ b,         // [dim] bias
    T* __restrict__ h_out,           // [B, dim] output
    T* __restrict__ v_cache) {       // [B, dim] pre-activation cache for backward

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float wx_val = static_cast<float>(Wx[idx]);
        float h_prev_val = static_cast<float>(h_prev[idx]);
        float alpha_val = static_cast<float>(alpha[d]);
        float b_val = static_cast<float>(b[d]);

        // α is pre-sigmoid, apply sigmoid here to ensure α ∈ (0, 1)
        float alpha_stable = 1.0f / (1.0f + expf(-alpha_val));

        // Pre-activation: v = Wx + α * h_prev + b
        float v = wx_val + alpha_stable * h_prev_val + b_val;

        // Cache pre-activation for backward
        if (v_cache) v_cache[idx] = static_cast<T>(v);

        // Apply tanh: h = tanh(v)
        float h_new = tanhf(v);
        h_out[idx] = static_cast<T>(h_new);
    }
}

// Backward through tanh recurrence with diagonal decay
// h = tanh(v) where v = Wx + α * h_prev + b
// dv = dh * (1 - tanh(v)^2) = dh * (1 - h^2)
// dWx = dv, dh_prev = dv * α, dalpha = dv * h_prev * dsigmoid, db = dv
template<typename T>
__global__ void DiagonalRecurrenceBackwardKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ h_prev,      // [B, dim]
    const T* __restrict__ v_cache,     // [B, dim] pre-activation cache
    const T* __restrict__ alpha,       // [dim] pre-sigmoid alpha
    const T* __restrict__ dh,          // [B, dim] gradient from above
    const T* __restrict__ dh_recurrent,// [B, dim] gradient from next timestep (or null)
    T* __restrict__ dv_out,            // [B, dim] gradient w.r.t. pre-activation
    T* __restrict__ dh_prev_out,       // [B, dim] gradient to h_prev for next BPTT step
    float* __restrict__ dalpha,        // [dim] accumulated gradient for alpha
    float* __restrict__ db) {          // [dim] accumulated gradient for bias

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // Combine gradients from output and recurrent path
        float grad = static_cast<float>(dh[idx]);
        if (dh_recurrent) grad += static_cast<float>(dh_recurrent[idx]);

        float h_prev_val = static_cast<float>(h_prev[idx]);
        float v = static_cast<float>(v_cache[idx]);
        float alpha_raw = static_cast<float>(alpha[d]);
        float alpha_val = 1.0f / (1.0f + expf(-alpha_raw));  // sigmoid

        // h = tanh(v), so dv = dh * (1 - h^2)
        float h = tanhf(v);
        float dtanh = 1.0f - h * h;
        float dv = grad * dtanh;

        // dL/dv (for Wx gradient accumulation)
        dv_out[idx] = static_cast<T>(dv);

        // dL/dh_prev = dv * α
        dh_prev_out[idx] = static_cast<T>(dv * alpha_val);

        // dL/dalpha_raw = dv * h_prev * dsigmoid(alpha_raw)
        float dsigmoid = alpha_val * (1.0f - alpha_val);
        float dalpha_val = dv * h_prev_val * dsigmoid;
        atomicAdd(&dalpha[d], dalpha_val);

        // dL/db = dv
        atomicAdd(&db[d], dv);
    }
}

// X-Only Selective Output: output = h * silu(x + b_gate)
// Same as in x_gated_elman
template<typename T>
__global__ void XGatedOutputForward(
    const int batch_size,
    const int dim,
    const T* __restrict__ h,            // [B, dim]
    const T* __restrict__ x,            // [B, dim] input at this timestep
    const T* __restrict__ b_gate,       // [dim] learned gate bias
    T* __restrict__ output,             // [B, dim]
    T* __restrict__ gate_cache) {       // [B, dim] silu value for backward

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float h_val = static_cast<float>(h[idx]);
        float x_val = static_cast<float>(x[idx]);
        float b_val = static_cast<float>(b_gate[d]);

        // X-only gating: silu(x + b_gate)
        float gate_raw = x_val + b_val;
        float sigmoid_val = 1.0f / (1.0f + expf(-gate_raw));
        float silu_val = gate_raw * sigmoid_val;

        output[idx] = static_cast<T>(h_val * silu_val);
        if (gate_cache) gate_cache[idx] = static_cast<T>(silu_val);
    }
}

template<typename T>
__global__ void XGatedOutputBackward(
    const int batch_size,
    const int dim,
    const T* __restrict__ h,
    const T* __restrict__ x,
    const T* __restrict__ b_gate,
    const T* __restrict__ d_output,
    T* __restrict__ dh,              // gradient to h
    T* __restrict__ dx,              // gradient to x
    float* __restrict__ d_b_gate) {  // accumulated gradient for b_gate

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float h_val = static_cast<float>(h[idx]);
        float x_val = static_cast<float>(x[idx]);
        float b_val = static_cast<float>(b_gate[d]);
        float dout = static_cast<float>(d_output[idx]);

        // X-only gating: silu(x + b_gate)
        float gate_raw = x_val + b_val;
        float sigmoid_val = 1.0f / (1.0f + expf(-gate_raw));
        float silu_val = gate_raw * sigmoid_val;

        // d_silu/d_gate_raw = sigmoid * (1 + gate_raw * (1 - sigmoid))
        float dsilu = sigmoid_val * (1.0f + gate_raw * (1.0f - sigmoid_val));

        // d_output/d_h = silu(x + b_gate)
        float dh_val = dout * silu_val;

        // d_output/d_x = h * dsilu
        float dx_val = dout * h_val * dsilu;

        // d_output/d_b_gate = h * dsilu
        float db_val = dout * h_val * dsilu;

        dh[idx] = static_cast<T>(dh_val);
        dx[idx] = static_cast<T>(dx_val);
        atomicAdd(&d_b_gate[d], db_val);
    }
}

// Add vectors: a += b
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
// Diagonal Elman Forward
// =============================================================================

template<typename T>
DiagonalElmanForward<T>::DiagonalElmanForward(
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
void DiagonalElmanForward<T>::Run(
    int steps,
    const T* W_x,
    const T* alpha,      // [dim] diagonal decay (pre-sigmoid)
    const T* b,          // [dim] recurrence bias
    const T* b_gate,
    const T* x,
    T* h,
    T* output,
    T* v_cache,          // [T, B, dim] pre-activation cache for backward
    T* gate_cache) {

    static const T alpha_one = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Pre-compute W_x @ x for ALL timesteps in ONE big GEMM
    T* tmp_Wx;
    cudaMalloc(&tmp_Wx, steps * BD * sizeof(T));

    // One big GEMM: tmp_Wx = x @ W_x.T for all T*B rows at once
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_x, dim_,
        x, dim_,
        &beta_zero,
        tmp_Wx, dim_);

    // Sequential recurrence (but NO W_h @ h GEMM - just diagonal multiply!)
    for (int t = 0; t < steps; ++t) {
        const T* Wx_t = tmp_Wx + t * BD;
        const T* h_prev = h + t * BD;
        const T* x_t = x + t * BD;
        T* h_t = h + (t + 1) * BD;
        T* out_t = output + t * BD;
        T* v_t = training_ ? (v_cache + t * BD) : nullptr;
        T* gate_t = training_ ? (gate_cache + t * BD) : nullptr;

        // h_t = tanh(Wx_t + α ⊙ h_prev + b) -- WITH tanh!
        DiagonalRecurrenceKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, Wx_t, h_prev, alpha, b, h_t, v_t);

        // output = h * silu(x + b_gate) -- X-ONLY GATING
        XGatedOutputForward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, x_t, b_gate, out_t, gate_t);
    }

    cudaFree(tmp_Wx);
}

// =============================================================================
// Diagonal Elman Backward
// =============================================================================

template<typename T>
DiagonalElmanBackward<T>::DiagonalElmanBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      sync_stream_(stream) {
    stream_[0] = stream;
    stream_[1] = nullptr;
    event_ = nullptr;
}

template<typename T>
DiagonalElmanBackward<T>::~DiagonalElmanBackward() {
}

template<typename T>
void DiagonalElmanBackward<T>::Run(
    int steps,
    const T* W_x,
    const T* alpha,
    const T* b_gate,
    const T* x,
    const T* h,
    const T* v_cache,    // [T, B, dim] pre-activation cache
    const T* gate_cache,
    const T* d_output,
    T* dx,
    T* dW_x,
    T* dalpha,
    T* db,               // [dim] gradient for recurrence bias
    T* d_b_gate,
    T* workspace) {

    static const T alpha_one = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    const cudaStream_t stream = stream_[0];

    // Workspace layout: [dv_all: T*BD] [dh: BD] [dh_recurrent: BD]
    //                   [dx_gate: BD] [dalpha_float: dim] [db_float: dim] [db_gate_float: dim]
    T* dv_all = workspace;
    T* dh = workspace + steps * BD;
    T* dh_recurrent = workspace + (steps + 1) * BD;
    T* dx_gate = workspace + (steps + 2) * BD;
    float* dalpha_float = reinterpret_cast<float*>(workspace + (steps + 3) * BD);
    float* db_float = dalpha_float + dim_;
    float* db_gate_float = db_float + dim_;

    // Initialize workspace
    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(T), stream);
    cudaMemsetAsync(dalpha_float, 0, dim_ * sizeof(float), stream);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream);
    cudaMemsetAsync(db_gate_float, 0, dim_ * sizeof(float), stream);
    cudaMemsetAsync(dW_x, 0, dim_ * dim_ * sizeof(T), stream);

    // BPTT loop
    for (int t = steps - 1; t >= 0; --t) {
        const T* h_prev = h + t * BD;
        const T* h_t = h + (t + 1) * BD;
        const T* v_t = v_cache + t * BD;
        const T* x_t = x + t * BD;
        const T* d_out_t = d_output + t * BD;
        T* dv_t = dv_all + t * BD;
        T* dx_t = dx + t * BD;

        // Backward through x-only selective output: output = h * silu(x + b_gate)
        XGatedOutputBackward<T><<<num_blocks, block_size, 0, stream>>>(
            batch_size_, dim_, h_t, x_t, b_gate, d_out_t,
            dh, dx_gate, db_gate_float);

        // Add recurrent gradient from next timestep
        if (t < steps - 1) {
            VectorAddInplace<T><<<num_blocks, block_size, 0, stream>>>(BD, dh, dh_recurrent);
        }

        // Backward through tanh recurrence: h = tanh(Wx + α * h_prev + b)
        DiagonalRecurrenceBackwardKernel<T><<<num_blocks, block_size, 0, stream>>>(
            batch_size_, dim_, h_prev, v_t, alpha, dh,
            (t < steps - 1) ? dh_recurrent : nullptr,
            dv_t, dh_recurrent, dalpha_float, db_float);

        // Copy gate gradient to dx
        cudaMemcpyAsync(dx_t, dx_gate, BD * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    }

    // Batch GEMM for dx: dx += W_x @ dv_all
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_x, dim_,
        dv_all, dim_,
        &beta_one,
        dx, dim_);

    // dW_x = x^T @ dv_all
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        dv_all, dim_,
        &beta_one,
        dW_x, dim_);

    // Copy float gradients to output
    CopyFloatToT<T><<<(dim_ + 255) / 256, 256, 0, stream>>>(dim_, dalpha_float, dalpha);
    CopyFloatToT<T><<<(dim_ + 255) / 256, 256, 0, stream>>>(dim_, db_float, db);
    CopyFloatToT<T><<<(dim_ + 255) / 256, 256, 0, stream>>>(dim_, db_gate_float, d_b_gate);
}

// Explicit template instantiations
template struct DiagonalElmanForward<__half>;
template struct DiagonalElmanForward<__nv_bfloat16>;
template struct DiagonalElmanForward<float>;
template struct DiagonalElmanForward<double>;

template struct DiagonalElmanBackward<__half>;
template struct DiagonalElmanBackward<__nv_bfloat16>;
template struct DiagonalElmanBackward<float>;
template struct DiagonalElmanBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
