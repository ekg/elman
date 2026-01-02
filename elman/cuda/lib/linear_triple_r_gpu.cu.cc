// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// Level 5: Linear Triple R Elman
// v = R_x @ x + R_h @ h_prev + b
// delta = sigmoid(W_delta @ x + R_delta @ h_prev + b_delta)
// h_new = (1-delta) * h_prev + delta * tanh(v)
//
// h+x Output selectivity:
// output = h * silu(h + x + b_gate)

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

// Simple kernel for y += x (element-wise vector addition)
template<typename T>
__global__ void VectorAdd(const int n, const T* __restrict__ x, T* __restrict__ y) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = static_cast<T>(static_cast<float>(y[idx]) + static_cast<float>(x[idx]));
    }
}

// Kernel: Compute gated update with Triple R matrices
template<typename T>
__global__ void LinearTripleRGatedUpdate(
    const int batch_size,
    const int dim,
    const T* __restrict__ h_prev,
    const T* __restrict__ rx_x,        // [B, dim] R_x @ x
    const T* __restrict__ rh_h,        // [B, dim] R_h @ h_prev
    const T* __restrict__ wdelta_x,    // [B, dim] W_delta @ x
    const T* __restrict__ rdelta_h,    // [B, dim] R_delta @ h_prev
    const T* __restrict__ b,           // [dim]
    const T* __restrict__ b_delta,     // [dim]
    T* __restrict__ h_out,
    T* __restrict__ v_cache,
    T* __restrict__ delta_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // Delta gate: sigmoid(W_delta @ x + R_delta @ h_prev + b_delta)
        float delta_in = static_cast<float>(wdelta_x[idx]) +
                         static_cast<float>(rdelta_h[idx]) +
                         static_cast<float>(b_delta[d]);
        float delta = 1.0f / (1.0f + expf(-delta_in));
        if (delta_cache) delta_cache[idx] = static_cast<T>(delta);

        // Candidate: v = R_x @ x + R_h @ h_prev + b
        float v = static_cast<float>(rx_x[idx]) +
                  static_cast<float>(rh_h[idx]) +
                  static_cast<float>(b[d]);
        if (v_cache) v_cache[idx] = static_cast<T>(v);
        float candidate = tanhf(v);

        // Gated update: h = (1 - delta) * h_prev + delta * candidate
        float h_p = static_cast<float>(h_prev[idx]);
        float h_new = (1.0f - delta) * h_p + delta * candidate;
        h_out[idx] = static_cast<T>(h_new);
    }
}

// Kernel: h+x selective output forward
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
    const int total = batch_size * dim;

    if (idx < total) {
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
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float h_val = static_cast<float>(h[idx]);
        float x_val = static_cast<float>(x[idx]);
        float b_val = static_cast<float>(b_gate[d]);
        float silu_val = static_cast<float>(gate_cache[idx]);
        float dout = static_cast<float>(d_output[idx]);

        float gate_raw = h_val + x_val + b_val;
        float sigmoid_val = 1.0f / (1.0f + expf(-gate_raw));

        float dsilu_dgate = sigmoid_val * (1.0f + gate_raw * (1.0f - sigmoid_val));
        float d_gate_raw = dout * h_val * dsilu_dgate;

        float dh_val = dout * silu_val + d_gate_raw;
        dh[idx] = static_cast<T>(dh_val);
        dx[idx] = static_cast<T>(d_gate_raw);

        atomicAdd(&db_gate[d], d_gate_raw);
    }
}

// In-place vector addition for gradient accumulation
template<typename T>
__global__ void VectorAddInplace(const int n, const T* __restrict__ src, T* __restrict__ dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = static_cast<T>(static_cast<float>(dst[idx]) + static_cast<float>(src[idx]));
    }
}

// Backward through Triple R gated update
template<typename T>
__global__ void LinearTripleRGatedBackward(
    const int batch_size,
    const int dim,
    const T* __restrict__ h_prev,
    const T* __restrict__ v,
    const T* __restrict__ delta,
    const T* __restrict__ dh,
    const T* __restrict__ dh_recurrent,
    T* __restrict__ dv,
    T* __restrict__ d_delta_raw,
    T* __restrict__ dh_prev_gated,
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

        float dh_prev_val = one_minus_del * grad_h;
        dh_prev_gated[idx] = static_cast<T>(dh_prev_val);

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
// Linear Triple R Forward
// =============================================================================

template<typename T>
LinearTripleRForward<T>::LinearTripleRForward(
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
void LinearTripleRForward<T>::Run(
    int steps,
    const T* R_h,
    const T* R_x,
    const T* R_delta,
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
    // =========================================================================
    T *all_rx_x, *all_wdelta_x, *rh_h, *rdelta_h;
    cudaMalloc(&all_rx_x, steps * BD * sizeof(T));
    cudaMalloc(&all_wdelta_x, steps * BD * sizeof(T));
    cudaMalloc(&rh_h, BD * sizeof(T));
    cudaMalloc(&rdelta_h, BD * sizeof(T));

    // Pre-compute R_x @ x for ALL timesteps
    blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_, &alpha, R_x, dim_, x, dim_, &beta_zero, all_rx_x, dim_);

    // Pre-compute W_delta @ x for ALL timesteps
    blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_, &alpha, W_delta, dim_, x, dim_, &beta_zero, all_wdelta_x, dim_);

    for (int t = 0; t < steps; ++t) {
        const T* rx_x_t = all_rx_x + t * BD;
        const T* wdelta_x_t = all_wdelta_x + t * BD;
        const T* x_t = x + t * BD;
        const T* h_prev = h + t * BD;
        T* h_t = h + (t + 1) * BD;
        T* out_t = output + t * BD;
        T* v_t = training_ ? (v + t * BD) : nullptr;
        T* delta_t = training_ ? (delta_cache + t * BD) : nullptr;
        T* gate_t = training_ ? (gate_cache + t * BD) : nullptr;

        // rh_h = h_prev @ R_h.T (per-step)
        blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, R_h, dim_, h_prev, dim_, &beta_zero, rh_h, dim_);

        // rdelta_h = h_prev @ R_delta.T (per-step)
        blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, R_delta, dim_, h_prev, dim_, &beta_zero, rdelta_h, dim_);

        // Triple R gated update
        LinearTripleRGatedUpdate<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_prev, rx_x_t, rh_h, wdelta_x_t, rdelta_h, b, b_delta, h_t, v_t, delta_t);

        // h+x selective output: output = h * silu(h + x + b_gate)
        SelectiveOutputForward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, x_t, b_gate, out_t, gate_t);
    }

    cudaFree(all_rx_x);
    cudaFree(all_wdelta_x);
    cudaFree(rh_h);
    cudaFree(rdelta_h);
}

// =============================================================================
// Linear Triple R Backward
// =============================================================================

template<typename T>
LinearTripleRBackward<T>::LinearTripleRBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void LinearTripleRBackward<T>::Run(
    int steps,
    const T* R_h,
    const T* R_x,
    const T* R_delta,
    const T* W_delta,
    const T* b_gate,
    const T* x,
    const T* h,
    const T* v,
    const T* delta_cache,
    const T* gate_cache,
    const T* d_output,
    T* dx,
    T* dR_h,
    T* dR_x,
    T* dR_delta,
    T* dW_delta,
    T* db,
    T* db_delta,
    T* db_gate) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Allocate temp buffers
    T *dh_output, *dx_output, *dv, *d_delta_raw, *dh_prev_gated, *dh_recurrent, *dh_rh, *dh_rdelta;
    float *db_f, *db_delta_f, *db_gate_f;

    cudaMalloc(&dh_output, BD * sizeof(T));
    cudaMalloc(&dx_output, BD * sizeof(T));
    cudaMalloc(&dv, BD * sizeof(T));
    cudaMalloc(&d_delta_raw, BD * sizeof(T));
    cudaMalloc(&dh_prev_gated, BD * sizeof(T));
    cudaMalloc(&dh_recurrent, BD * sizeof(T));
    cudaMalloc(&dh_rh, BD * sizeof(T));
    cudaMalloc(&dh_rdelta, BD * sizeof(T));
    cudaMalloc(&db_f, dim_ * sizeof(float));
    cudaMalloc(&db_delta_f, dim_ * sizeof(float));
    cudaMalloc(&db_gate_f, dim_ * sizeof(float));

    // Initialize
    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(T), stream_);
    cudaMemsetAsync(db_f, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_delta_f, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_gate_f, 0, dim_ * sizeof(float), stream_);

    // Zero out weight gradients
    cudaMemsetAsync(dR_h, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dR_x, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dR_delta, 0, dim_ * dim_ * sizeof(T), stream_);
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
            batch_size_, dim_, h_t, x_t, b_gate, gate_t, d_out_t,
            dh_output, dx_output, db_gate_f);

        // Backward through gated update
        LinearTripleRGatedBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_prev, v_t, delta_t, dh_output,
            (t < steps - 1) ? dh_recurrent : nullptr,
            dv, d_delta_raw, dh_prev_gated, db_f, db_delta_f);

        // dR_x += dv.T @ x_t
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_, &alpha, dv, dim_, x_t, dim_, &beta_one, dR_x, dim_);

        // dR_h += dv.T @ h_prev
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_, &alpha, dv, dim_, h_prev, dim_, &beta_one, dR_h, dim_);

        // dW_delta += d_delta_raw.T @ x_t
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_, &alpha, d_delta_raw, dim_, x_t, dim_, &beta_one, dW_delta, dim_);

        // dR_delta += d_delta_raw.T @ h_prev
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_, &alpha, d_delta_raw, dim_, h_prev, dim_, &beta_one, dR_delta, dim_);

        // dx = dv @ R_x + d_delta_raw @ W_delta + dx_output
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, R_x, dim_, dv, dim_, &beta_zero, dx_t, dim_);
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, W_delta, dim_, d_delta_raw, dim_, &beta_one, dx_t, dim_);
        VectorAddInplace<T><<<num_blocks, block_size, 0, stream_>>>(BD, dx_output, dx_t);

        // dh_prev from R_h and R_delta paths
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, R_h, dim_, dv, dim_, &beta_zero, dh_rh, dim_);
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, R_delta, dim_, d_delta_raw, dim_, &beta_zero, dh_rdelta, dim_);

        // dh_recurrent = dh_prev_gated + dh_rh + dh_rdelta
        cudaMemcpyAsync(dh_recurrent, dh_prev_gated, BD * sizeof(T), cudaMemcpyDeviceToDevice, stream_);
        VectorAddInplace<T><<<num_blocks, block_size, 0, stream_>>>(BD, dh_rh, dh_recurrent);
        VectorAddInplace<T><<<num_blocks, block_size, 0, stream_>>>(BD, dh_rdelta, dh_recurrent);
    }

    // Convert float gradients to T
    const int bias_blocks = (dim_ + block_size - 1) / block_size;
    CopyFloatToT<T><<<bias_blocks, block_size, 0, stream_>>>(dim_, db_f, db);
    CopyFloatToT<T><<<bias_blocks, block_size, 0, stream_>>>(dim_, db_delta_f, db_delta);
    CopyFloatToT<T><<<bias_blocks, block_size, 0, stream_>>>(dim_, db_gate_f, db_gate);

    // Cleanup
    cudaFree(dh_output);
    cudaFree(dx_output);
    cudaFree(dv);
    cudaFree(d_delta_raw);
    cudaFree(dh_prev_gated);
    cudaFree(dh_recurrent);
    cudaFree(dh_rh);
    cudaFree(dh_rdelta);
    cudaFree(db_f);
    cudaFree(db_delta_f);
    cudaFree(db_gate_f);
}

// Explicit instantiations
template class LinearTripleRForward<float>;
template class LinearTripleRForward<__half>;
template class LinearTripleRForward<__nv_bfloat16>;
template class LinearTripleRForward<double>;
template class LinearTripleRBackward<float>;
template class LinearTripleRBackward<__half>;
template class LinearTripleRBackward<__nv_bfloat16>;
template class LinearTripleRBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
