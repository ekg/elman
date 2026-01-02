// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// Level 3: Diagonal Selective Elman - Diagonal r_h (like Mamba2's diagonal A)
// delta = sigmoid(W_delta @ x_t + b_delta)
// h_t = (1 - delta) * h_{t-1} + delta * tanh(W_x @ x_t + r_h * h_{t-1} + b)
// where r_h is a VECTOR (diagonal), not full matrix
// output_t = h_t * silu(h_t + x_t + b_gate)  -- h+x selective gating

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cfloat>

#include "hasty/elman_ladder.h"
#include "blas.h"
#include "inline_ops.h"

namespace {

// Kernel: Compute gated update with DIAGONAL r_h
// v = W_x @ x + r_h * h_prev + b (element-wise r_h, not matrix)
template<typename T>
__global__ void DiagonalSelectiveGatedUpdate(
    const int batch_size,
    const int dim,
    const T* __restrict__ h_prev,
    const T* __restrict__ wx_x,        // [B, dim] W_x @ x (pre-computed)
    const T* __restrict__ r_h,         // [dim] diagonal decay
    const T* __restrict__ delta_raw,   // [B, dim] W_delta @ x
    const T* __restrict__ b,           // [dim]
    const T* __restrict__ b_delta,     // [dim]
    T* __restrict__ h_out,
    T* __restrict__ v_cache,
    T* __restrict__ delta_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // Delta gate: sigmoid(delta_raw + b_delta)
        float delta_in = static_cast<float>(delta_raw[idx]) + static_cast<float>(b_delta[d]);
        float delta = 1.0f / (1.0f + expf(-delta_in));
        if (delta_cache) delta_cache[idx] = static_cast<T>(delta);

        // Candidate with DIAGONAL r_h: v = W_x @ x + r_h * h_prev + b
        float h_p = static_cast<float>(h_prev[idx]);
        float v = static_cast<float>(wx_x[idx]) + static_cast<float>(r_h[d]) * h_p + static_cast<float>(b[d]);
        if (v_cache) v_cache[idx] = static_cast<T>(v);
        float candidate = tanhf(v);

        // Gated update: h = (1 - delta) * h_prev + delta * candidate
        float h_new = (1.0f - delta) * h_p + delta * candidate;
        h_out[idx] = static_cast<T>(h_new);
    }
}

// Kernel: h+x selective output
// output = h * silu(h + x + b_gate)
template<typename T>
__global__ void SelectiveOutputForward(
    const int batch_size,
    const int dim,
    const T* __restrict__ h,
    const T* __restrict__ x,
    const T* __restrict__ b_gate,
    T* __restrict__ output,
    T* __restrict__ gate_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * dim) {
        const int d = idx % dim;

        float h_val = static_cast<float>(h[idx]);
        float x_val = static_cast<float>(x[idx]);
        float b_val = static_cast<float>(b_gate[d]);

        float gate_raw = h_val + x_val + b_val;
        float sigmoid_val = 1.0f / (1.0f + expf(-gate_raw));
        float silu_val = gate_raw * sigmoid_val;

        output[idx] = static_cast<T>(h_val * silu_val);
        if (gate_cache) gate_cache[idx] = static_cast<T>(silu_val);
    }
}

// Backward through h+x selective output
template<typename T>
__global__ void SelectiveOutputBackward(
    const int batch_size,
    const int dim,
    const T* __restrict__ h,
    const T* __restrict__ x,
    const T* __restrict__ b_gate,
    const T* __restrict__ gate_cache,
    const T* __restrict__ d_output,
    T* __restrict__ dh,
    T* __restrict__ dx,
    float* __restrict__ db_gate) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * dim) {
        const int d = idx % dim;

        float h_val = static_cast<float>(h[idx]);
        float x_val = static_cast<float>(x[idx]);
        float b_val = static_cast<float>(b_gate[d]);
        float silu_val = static_cast<float>(gate_cache[idx]);
        float dout = static_cast<float>(d_output[idx]);

        float gate_raw = h_val + x_val + b_val;
        float sigmoid_val = 1.0f / (1.0f + expf(-gate_raw));
        float dsilu_dgate = sigmoid_val * (1.0f + gate_raw * (1.0f - sigmoid_val));

        float dh_val = dout * (silu_val + h_val * dsilu_dgate);
        dh[idx] = static_cast<T>(dh_val);

        float dx_val = dout * h_val * dsilu_dgate;
        dx[idx] = static_cast<T>(dx_val);

        atomicAdd(&db_gate[d], dx_val);
    }
}

// Kernel: Add gradients in-place
template<typename T>
__global__ void VectorAddInplace(
    const int n,
    T* __restrict__ a,
    const T* __restrict__ b) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = static_cast<T>(static_cast<float>(a[idx]) + static_cast<float>(b[idx]));
    }
}

// Backward through diagonal gated update
template<typename T>
__global__ void DiagonalSelectiveGatedBackward(
    const int batch_size,
    const int dim,
    const T* __restrict__ h_prev,
    const T* __restrict__ v,
    const T* __restrict__ delta,
    const T* __restrict__ r_h,
    const T* __restrict__ dh,
    const T* __restrict__ dh_recurrent,
    T* __restrict__ dv,
    T* __restrict__ d_delta_raw,
    T* __restrict__ dh_prev_out,
    float* __restrict__ dr_h,              // [dim] gradient for diagonal (float for atomicAdd)
    float* __restrict__ db,
    float* __restrict__ db_delta) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float grad_h = static_cast<float>(dh[idx]);
        if (dh_recurrent) grad_h += static_cast<float>(dh_recurrent[idx]);

        float cand = tanhf(static_cast<float>(v[idx]));
        float del = static_cast<float>(delta[idx]);
        float one_minus_del = 1.0f - del;

        float d_cand = grad_h * del;
        float dtanh = 1.0f - cand * cand;
        float dv_val = d_cand * dtanh;
        dv[idx] = static_cast<T>(dv_val);

        float h_p = static_cast<float>(h_prev[idx]);
        float d_delta = grad_h * (cand - h_p);
        float dsigmoid = del * one_minus_del;
        float d_delta_raw_val = d_delta * dsigmoid;
        d_delta_raw[idx] = static_cast<T>(d_delta_raw_val);

        // dh_prev from both gated path and r_h path
        // dh_prev += (1 - delta) * grad_h + dv * r_h
        float dh_prev_gated = one_minus_del * grad_h;
        float dh_prev_rh = dv_val * static_cast<float>(r_h[d]);
        dh_prev_out[idx] = static_cast<T>(dh_prev_gated + dh_prev_rh);

        // dr_h: gradient for diagonal element
        // v = W_x @ x + r_h * h_prev + b
        // dv/dr_h = h_prev
        float dr_h_val = dv_val * h_p;
        atomicAdd(&dr_h[d], dr_h_val);

        atomicAdd(&db[d], dv_val);
        atomicAdd(&db_delta[d], d_delta_raw_val);
    }
}

// Kernel: Copy float array to type T (for bias gradients)
template<typename T>
__global__ void CopyFloatToT(
    const int n,
    const float* __restrict__ src,
    T* __restrict__ dst) {
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
// Diagonal Selective Elman Forward
// =============================================================================

template<typename T>
DiagonalSelectiveElmanForward<T>::DiagonalSelectiveElmanForward(
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
void DiagonalSelectiveElmanForward<T>::Run(
    int steps,
    const T* W_x,
    const T* r_h,
    const T* W_delta,
    const T* b,
    const T* b_delta,
    const T* b_gate,
    const T* x,
    T* h,
    T* output,
    T* v,
    T* delta_cache,
    T* gate_cache) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // =========================================================================
    // HASTE PATTERN: Pre-compute input projections for ALL timesteps
    // Reduces from 2*T GEMMs to 2 big GEMMs (much more efficient)
    // =========================================================================
    T *all_wx_x, *all_delta_raw;
    cudaMalloc(&all_wx_x, steps * BD * sizeof(T));
    cudaMalloc(&all_delta_raw, steps * BD * sizeof(T));

    // Pre-compute W_x @ x for ALL timesteps in one GEMM
    blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_, &alpha, W_x, dim_, x, dim_, &beta_zero, all_wx_x, dim_);

    // Pre-compute W_delta @ x for ALL timesteps in one GEMM
    blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_, &alpha, W_delta, dim_, x, dim_, &beta_zero, all_delta_raw, dim_);

    // Per-timestep: diagonal gated update + h+x selective output
    for (int t = 0; t < steps; ++t) {
        const T* wx_x_t = all_wx_x + t * BD;
        const T* delta_raw_t = all_delta_raw + t * BD;
        const T* x_t = x + t * BD;
        const T* h_prev = h + t * BD;
        T* h_t = h + (t + 1) * BD;
        T* out_t = output + t * BD;
        T* v_t = training_ ? (v + t * BD) : nullptr;
        T* delta_t = training_ ? (delta_cache + t * BD) : nullptr;
        T* gate_t = training_ ? (gate_cache + t * BD) : nullptr;

        // Diagonal gated update (r_h is element-wise, not matrix)
        DiagonalSelectiveGatedUpdate<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_prev, wx_x_t, r_h, delta_raw_t, b, b_delta, h_t, v_t, delta_t);

        // h+x selective output: output = h * silu(h + x + b_gate)
        SelectiveOutputForward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, x_t, b_gate, out_t, gate_t);
    }

    cudaFree(all_wx_x);
    cudaFree(all_delta_raw);
}

// =============================================================================
// Diagonal Selective Elman Backward
// =============================================================================

template<typename T>
DiagonalSelectiveElmanBackward<T>::DiagonalSelectiveElmanBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void DiagonalSelectiveElmanBackward<T>::Run(
    int steps,
    const T* W_x,
    const T* r_h,
    const T* W_delta,
    const T* b_gate,
    const T* x,
    const T* h,
    const T* v,
    const T* delta_cache,
    const T* gate_cache,
    const T* d_output,
    T* dx,
    T* dW_x,
    T* dr_h,
    T* dW_delta,
    T* db,
    T* db_delta,
    T* db_gate,
    T* workspace) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // ==========================================================================
    // WORKSPACE LAYOUT: [dv: BD] [d_delta_raw: BD] [dh_recurrent: BD] [dh_prev: BD]
    //                   [dh_selective: BD] [dx_selective: BD]
    //                   [dr_h_float: dim] [db_float: dim] [db_delta_float: dim]
    //                   [db_gate_float: dim]
    // ==========================================================================
    T* dv = workspace;
    T* d_delta_raw = workspace + BD;
    T* dh_recurrent = workspace + 2 * BD;
    T* dh_prev = workspace + 3 * BD;
    T* dh_selective = workspace + 4 * BD;
    T* dx_selective = workspace + 5 * BD;
    float* dr_h_float = reinterpret_cast<float*>(workspace + 6 * BD);
    float* db_float = dr_h_float + dim_;
    float* db_delta_float = db_float + dim_;
    float* db_gate_float = db_delta_float + dim_;

    // Initialize workspace (all async)
    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(T), stream_);
    cudaMemsetAsync(dr_h_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_delta_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_gate_float, 0, dim_ * sizeof(float), stream_);

    // Zero weight gradients (async)
    cudaMemsetAsync(dW_x, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dW_delta, 0, dim_ * dim_ * sizeof(T), stream_);

    for (int t = steps - 1; t >= 0; --t) {
        const T* x_t = x + t * BD;
        const T* h_prev = h + t * BD;
        const T* h_t = h + (t + 1) * BD;
        const T* v_t = v + t * BD;
        const T* delta_t = delta_cache + t * BD;
        const T* gate_t = gate_cache + t * BD;
        const T* d_out_t = d_output + t * BD;
        T* dx_t = dx + t * BD;

        // Backward through h+x selective output
        SelectiveOutputBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, x_t, b_gate, gate_t,
            d_out_t, dh_selective, dx_selective, db_gate_float);

        // Add dh_recurrent to dh_selective
        if (t < steps - 1) {
            VectorAddInplace<T><<<num_blocks, block_size, 0, stream_>>>(
                BD, dh_selective, dh_recurrent);
        }

        // Backward through diagonal gated update
        DiagonalSelectiveGatedBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_prev, v_t, delta_t, r_h, dh_selective,
            nullptr,  // dh_recurrent added above
            dv, d_delta_raw, dh_prev, dr_h_float, db_float, db_delta_float);

        // dx = dx_selective + dv @ W_x + d_delta_raw @ W_delta
        cudaMemcpyAsync(dx_t, dx_selective, BD * sizeof(T), cudaMemcpyDeviceToDevice, stream_);
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, W_x, dim_, dv, dim_, &alpha, dx_t, dim_);
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, W_delta, dim_, d_delta_raw, dim_, &alpha, dx_t, dim_);

        // dh_recurrent = dv (through r_h) + dh_prev (through gated path)
        // DiagonalSelectiveGatedBackward already computes this in dh_prev
        VectorAddInplace<T><<<num_blocks, block_size, 0, stream_>>>(
            BD, dh_prev, dh_recurrent);  // Add old recurrent to dh_prev
        // Copy to dh_recurrent for next iteration
        cudaMemcpyAsync(dh_recurrent, dh_prev, BD * sizeof(T), cudaMemcpyDeviceToDevice, stream_);

        // Weight gradients
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_, &alpha, dv, dim_, x_t, dim_, &alpha, dW_x, dim_);
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_, &alpha, d_delta_raw, dim_, x_t, dim_, &alpha, dW_delta, dim_);
    }

    // Copy float gradients to T type using parallel kernel
    const int bias_blocks = (dim_ + block_size - 1) / block_size;
    CopyFloatToT<T><<<bias_blocks, block_size, 0, stream_>>>(dim_, dr_h_float, dr_h);
    CopyFloatToT<T><<<bias_blocks, block_size, 0, stream_>>>(dim_, db_float, db);
    CopyFloatToT<T><<<bias_blocks, block_size, 0, stream_>>>(dim_, db_delta_float, db_delta);
    CopyFloatToT<T><<<bias_blocks, block_size, 0, stream_>>>(dim_, db_gate_float, db_gate);
}

// Explicit template instantiations
template struct DiagonalSelectiveElmanForward<__half>;
template struct DiagonalSelectiveElmanForward<__nv_bfloat16>;
template struct DiagonalSelectiveElmanForward<float>;
template struct DiagonalSelectiveElmanForward<double>;

template struct DiagonalSelectiveElmanBackward<__half>;
template struct DiagonalSelectiveElmanBackward<__nv_bfloat16>;
template struct DiagonalSelectiveElmanBackward<float>;
template struct DiagonalSelectiveElmanBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
