// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// Level 6: Linear Polynomial Elman
// alpha = 1 + softplus(W_alpha @ x + b_alpha)
// v = W_x @ x + r_h * h_prev + b
// candidate = sign(v) * |v|^alpha (bounded)
// delta = sigmoid(W_delta @ x + b_delta)
// h_new = (1-delta) * h_prev + delta * candidate
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

// Kernel: Add bias to each row (batch_size rows, dim columns)
template<typename T>
__global__ void AddBiasKernel(const int batch_size, const int dim,
                              T* __restrict__ data, const T* __restrict__ bias) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;
    if (idx < total) {
        const int d = idx % dim;
        data[idx] = static_cast<T>(static_cast<float>(data[idx]) + static_cast<float>(bias[d]));
    }
}

// Kernel: Compute polynomial gated update
template<typename T>
__global__ void LinearPolynomialGatedUpdate(
    const int batch_size,
    const int dim,
    const T* __restrict__ h_prev,
    const T* __restrict__ wx_x,        // [B, dim] W_x @ x
    const T* __restrict__ r_h,         // [dim] diagonal
    const T* __restrict__ alpha_raw,   // [B, dim] W_alpha @ x + b_alpha
    const T* __restrict__ delta_raw,   // [B, dim] W_delta @ x + b_delta
    const T* __restrict__ b,           // [dim]
    T* __restrict__ h_out,
    T* __restrict__ v_cache,
    T* __restrict__ alpha_cache,
    T* __restrict__ delta_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // Input-dependent alpha: 1 + softplus(alpha_raw), capped at 2.0 for stability
        float alpha_in = static_cast<float>(alpha_raw[idx]);
        float softplus_val = log1pf(expf(alpha_in));
        float alpha = 1.0f + fminf(softplus_val, 1.0f);  // Cap alpha to [1, 2]
        if (alpha_cache) alpha_cache[idx] = static_cast<T>(alpha);

        // Delta gate
        float delta_in = static_cast<float>(delta_raw[idx]);
        float delta = 1.0f / (1.0f + expf(-delta_in));
        if (delta_cache) delta_cache[idx] = static_cast<T>(delta);

        // v = W_x @ x + r_h * h_prev + b
        // Clamp r_h to <= 0.9 for stability
        float h_p = static_cast<float>(h_prev[idx]);
        float r_h_val = fminf(static_cast<float>(r_h[d]), 0.9f);
        float v = static_cast<float>(wx_x[idx]) +
                  r_h_val * h_p +
                  static_cast<float>(b[d]);
        if (v_cache) v_cache[idx] = static_cast<T>(v);

        // Polynomial activation with pre-squashing for stability
        // v_squashed = v / (1 + |v|) maps to (-1, 1), bounding polynomial output
        float sign_v = (v >= 0.0f) ? 1.0f : -1.0f;
        float abs_v = fabsf(v);
        float v_squashed = abs_v / (1.0f + abs_v);  // Squash to (0, 1)
        float candidate = sign_v * powf(v_squashed + 1e-6f, alpha);
        // No additional clamp needed: v_squashed ∈ (0,1), alpha ≥ 1 → candidate ∈ (-1, 1)

        // Gated update: h = (1 - delta) * h_prev + delta * candidate
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
    if (idx < batch_size * dim) {
        const int d = idx % dim;

        float h_val = static_cast<float>(h[idx]);
        float x_val = static_cast<float>(x[idx]);
        float b_val = static_cast<float>(b_gate[d]);

        // gate_raw = h + x + b_gate
        float gate_raw = h_val + x_val + b_val;

        // silu(gate_raw) = gate_raw * sigmoid(gate_raw)
        float sigmoid_val = 1.0f / (1.0f + expf(-gate_raw));
        float silu_val = gate_raw * sigmoid_val;

        // output = h * silu(h + x + b_gate)
        output[idx] = static_cast<T>(h_val * silu_val);

        // Cache silu value for backward
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
    const T* __restrict__ d_output,
    T* __restrict__ dh_out,
    T* __restrict__ dx_out,
    float* __restrict__ db_gate) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * dim) {
        const int d = idx % dim;

        float h_val = static_cast<float>(h[idx]);
        float x_val = static_cast<float>(x[idx]);
        float b_val = static_cast<float>(b_gate[d]);
        float dout = static_cast<float>(d_output[idx]);

        // Recompute forward
        float gate_raw = h_val + x_val + b_val;
        float sigmoid_val = 1.0f / (1.0f + expf(-gate_raw));
        float silu_val = gate_raw * sigmoid_val;

        // dsilu/dgate = sigmoid + gate * sigmoid * (1 - sigmoid)
        //             = sigmoid * (1 + gate * (1 - sigmoid))
        float dsilu_dgate = sigmoid_val * (1.0f + gate_raw * (1.0f - sigmoid_val));

        // d_output/dh = silu + h * dsilu_dgate (since dgate/dh = 1)
        float dh = dout * (silu_val + h_val * dsilu_dgate);
        dh_out[idx] = static_cast<T>(dh);

        // d_output/dx = h * dsilu_dgate (since dgate/dx = 1)
        float dx = dout * h_val * dsilu_dgate;
        dx_out[idx] = static_cast<T>(dx);

        // d_output/db_gate = h * dsilu_dgate (since dgate/db = 1)
        atomicAdd(&db_gate[d], dout * h_val * dsilu_dgate);
    }
}

// Kernel: Add dx to existing dx (for accumulating gradient from h+x path)
template<typename T>
__global__ void VectorAddInplace(const int n, const T* __restrict__ src, T* __restrict__ dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = static_cast<T>(static_cast<float>(dst[idx]) + static_cast<float>(src[idx]));
    }
}

// Backward through polynomial gated update
template<typename T>
__global__ void LinearPolynomialGatedBackward(
    const int batch_size,
    const int dim,
    const T* __restrict__ h_prev,
    const T* __restrict__ v,
    const T* __restrict__ alpha,
    const T* __restrict__ delta,
    const T* __restrict__ r_h,
    const T* __restrict__ dh,
    const T* __restrict__ dh_recurrent,
    T* __restrict__ dv,
    T* __restrict__ d_alpha_raw,
    T* __restrict__ d_delta_raw,
    T* __restrict__ dh_prev_out,
    float* __restrict__ dr_h,
    float* __restrict__ db,
    float* __restrict__ db_delta) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float grad_h = static_cast<float>(dh[idx]);
        if (dh_recurrent) grad_h += static_cast<float>(dh_recurrent[idx]);

        float v_val = static_cast<float>(v[idx]);
        float alpha_val = static_cast<float>(alpha[idx]);
        float del = static_cast<float>(delta[idx]);
        float one_minus_del = 1.0f - del;
        float h_p = static_cast<float>(h_prev[idx]);

        // Compute candidate and its derivatives with pre-squashing
        // v_squashed = |v| / (1 + |v|), candidate = sign(v) * v_squashed^alpha
        float sign_v = (v_val >= 0.0f) ? 1.0f : -1.0f;
        float abs_v = fabsf(v_val);
        float one_plus_abs_v = 1.0f + abs_v;
        float v_squashed = abs_v / one_plus_abs_v;  // ∈ (0, 1)
        float v_squashed_eps = v_squashed + 1e-6f;
        float candidate = sign_v * powf(v_squashed_eps, alpha_val);
        // candidate ∈ (-1, 1), no clamping needed

        // d_candidate/dv = alpha * sign(v) * v_squashed^(alpha-1) * d(v_squashed)/d|v|
        // where d(|v|/(1+|v|))/d|v| = 1/(1+|v|)^2
        float d_squash_dv = 1.0f / (one_plus_abs_v * one_plus_abs_v);
        float d_cand_dv = alpha_val * sign_v * powf(v_squashed_eps, alpha_val - 1.0f) * d_squash_dv;
        // Gradient is naturally bounded since v_squashed ∈ (0,1) and d_squash_dv ≤ 1

        // d_candidate/d_alpha = sign(v) * v_squashed^alpha * log(v_squashed)
        float log_v_squashed = logf(v_squashed_eps);
        float d_cand_dalpha = sign_v * powf(v_squashed_eps, alpha_val) * log_v_squashed;
        // log(v_squashed) ≤ 0 since v_squashed ≤ 1, so gradient is bounded

        // Clip grad_h early to prevent explosion
        grad_h = fmaxf(fminf(grad_h, 10.0f), -10.0f);

        float d_cand = grad_h * del;
        d_cand = fmaxf(fminf(d_cand, 10.0f), -10.0f);
        float dv_val = d_cand * d_cand_dv;
        dv_val = fmaxf(fminf(dv_val, 100.0f), -100.0f);
        dv[idx] = static_cast<T>(dv_val);

        // d_alpha = d_cand * d_cand_dalpha
        float d_alpha = d_cand * d_cand_dalpha;
        d_alpha = fmaxf(fminf(d_alpha, 100.0f), -100.0f);
        // d_alpha_raw = d_alpha * softplus'(alpha_raw) = d_alpha * sigmoid(alpha_raw)
        // alpha = 1 + softplus(alpha_raw)
        // softplus'(x) = sigmoid(x)
        float alpha_raw = alpha_val - 1.0f;  // Approximate
        float sig_alpha = 1.0f / (1.0f + expf(-alpha_raw));
        d_alpha_raw[idx] = static_cast<T>(fmaxf(fminf(d_alpha * sig_alpha, 10.0f), -10.0f));

        // d_delta - clip candidate and h_p difference first
        float cand_hp_diff = fmaxf(fminf(candidate - h_p, 10.0f), -10.0f);
        float d_delta = grad_h * cand_hp_diff;
        float dsigmoid = del * one_minus_del;
        float d_delta_raw_val = d_delta * dsigmoid;
        d_delta_raw[idx] = static_cast<T>(fmaxf(fminf(d_delta_raw_val, 10.0f), -10.0f));

        // dh_prev from gated path and r_h path
        // Clamp r_h to <= 0.9 for stability (must match forward)
        float dh_prev_gated = one_minus_del * grad_h;
        float r_h_val = fminf(static_cast<float>(r_h[d]), 0.9f);
        float dh_prev_rh = dv_val * r_h_val;
        float dh_prev_total = dh_prev_gated + dh_prev_rh;
        dh_prev_out[idx] = static_cast<T>(fmaxf(fminf(dh_prev_total, 10.0f), -10.0f));

        // dr_h: gradient for diagonal element - clip contribution
        float dr_h_val = fmaxf(fminf(dv_val * h_p, 1.0f), -1.0f);
        atomicAdd(&dr_h[d], dr_h_val);

        atomicAdd(&db[d], fmaxf(fminf(dv_val, 1.0f), -1.0f));
        atomicAdd(&db_delta[d], d_delta_raw_val);
    }
}

}  // anonymous namespace


namespace hasty {
namespace v0 {
namespace elman_ladder {

// =============================================================================
// Linear Polynomial Forward
// =============================================================================

template<typename T>
LinearPolynomialForward<T>::LinearPolynomialForward(
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
void LinearPolynomialForward<T>::Run(
    int steps,
    const T* W_x,
    const T* r_h,
    const T* W_alpha,
    const T* b_alpha,
    const T* W_delta,
    const T* b,
    const T* b_delta,
    const T* b_gate,
    const T* x,
    T* h,
    T* output,
    T* v,
    T* alpha_cache,
    T* delta_cache,
    T* gate_cache) {

    static const T alpha_blas = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int BD = batch_size_ * dim_;
    const int TBD = steps * BD;  // Total batch*dim for all timesteps
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // =========================================================================
    // Haste pattern: Pre-compute ALL input projections in big GEMMs
    // =========================================================================

    // Workspace for pre-computed projections (all timesteps)
    T *all_wx_x, *all_alpha_raw, *all_delta_raw;
    cudaMalloc(&all_wx_x, TBD * sizeof(T));
    cudaMalloc(&all_alpha_raw, TBD * sizeof(T));
    cudaMalloc(&all_delta_raw, TBD * sizeof(T));

    // Pre-compute W_x @ x for ALL timesteps in one GEMM
    // x is [T, B, dim], treat as [T*B, dim]
    // Result all_wx_x is [T*B, dim]
    blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_, &alpha_blas, W_x, dim_, x, dim_, &beta_zero, all_wx_x, dim_);

    // Pre-compute W_alpha @ x for ALL timesteps
    blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_, &alpha_blas, W_alpha, dim_, x, dim_, &beta_zero, all_alpha_raw, dim_);

    // Pre-compute W_delta @ x for ALL timesteps
    blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_, &alpha_blas, W_delta, dim_, x, dim_, &beta_zero, all_delta_raw, dim_);

    // Add biases to alpha_raw and delta_raw using GPU kernel (for all timesteps)
    for (int t = 0; t < steps; ++t) {
        AddBiasKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, all_alpha_raw + t * BD, b_alpha);
        AddBiasKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, all_delta_raw + t * BD, b_delta);
    }

    // =========================================================================
    // Sequential loop: Only recurrent operations (no input GEMMs per step)
    // =========================================================================

    for (int t = 0; t < steps; ++t) {
        const T* h_prev = h + t * BD;
        T* h_t = h + (t + 1) * BD;
        T* out_t = output + t * BD;
        const T* x_t = x + t * BD;
        T* v_t = training_ ? (v + t * BD) : nullptr;
        T* alpha_t = training_ ? (alpha_cache + t * BD) : nullptr;
        T* delta_t = training_ ? (delta_cache + t * BD) : nullptr;
        T* gate_t = training_ ? (gate_cache + t * BD) : nullptr;

        // Get pre-computed projections for this timestep
        const T* wx_x_t = all_wx_x + t * BD;
        const T* alpha_raw_t = all_alpha_raw + t * BD;
        const T* delta_raw_t = all_delta_raw + t * BD;

        // Polynomial gated update (no input GEMM needed - already pre-computed)
        LinearPolynomialGatedUpdate<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_prev, wx_x_t, r_h, alpha_raw_t, delta_raw_t, b, h_t, v_t, alpha_t, delta_t);

        // h+x selective output: output = h * silu(h + x + b_gate)
        SelectiveOutputForward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, x_t, b_gate, out_t, gate_t);
    }

    cudaFree(all_wx_x);
    cudaFree(all_alpha_raw);
    cudaFree(all_delta_raw);
}

// =============================================================================
// Linear Polynomial Backward
// =============================================================================

template<typename T>
LinearPolynomialBackward<T>::LinearPolynomialBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void LinearPolynomialBackward<T>::Run(
    int steps,
    const T* W_x,
    const T* r_h,
    const T* W_alpha,
    const T* W_delta,
    const T* b_gate,
    const T* x,
    const T* h,
    const T* v,
    const T* alpha_cache,
    const T* delta_cache,
    const T* gate_cache,
    const T* d_output,
    T* dx,
    T* dW_x,
    T* dr_h,
    T* dW_alpha,
    T* db_alpha,
    T* dW_delta,
    T* db,
    T* db_delta,
    T* db_gate,
    T* workspace) {

    static const T alpha_blas = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int BD = batch_size_ * dim_;
    const int TBD = steps * BD;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;
    const int dim_blocks = (dim_ + block_size - 1) / block_size;

    // ==========================================================================
    // HASTE PATTERN: Workspace from Python (PyTorch caching allocator)
    // Layout: [dv_all: TBD][d_alpha_all: TBD][d_delta_all: TBD][dx_gate: TBD]
    //         [dh_gate: BD][dh: BD][dh_prev_out: BD][dh_recurrent: BD]
    //         [dr_h_f: dim floats][db_f: dim][db_delta_f: dim][db_alpha_f: dim][db_gate_f: dim]
    // ==========================================================================

    T* dv_all = workspace;
    T* d_alpha_all = dv_all + TBD;
    T* d_delta_all = d_alpha_all + TBD;
    T* dx_gate = d_delta_all + TBD;
    T* dh_gate = dx_gate + TBD;
    T* dh = dh_gate + BD;
    T* dh_prev_out = dh + BD;
    T* dh_recurrent = dh_prev_out + BD;
    float* dr_h_f = reinterpret_cast<float*>(dh_recurrent + BD);
    float* db_f = dr_h_f + dim_;
    float* db_delta_f = db_f + dim_;
    float* db_alpha_f = db_delta_f + dim_;
    float* db_gate_f = db_alpha_f + dim_;

    // Zero out weight gradients and workspace (all async on same stream)
    cudaMemsetAsync(dW_x, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dW_alpha, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dW_delta, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(T), stream_);
    cudaMemsetAsync(dr_h_f, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_f, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_delta_f, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_alpha_f, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_gate_f, 0, dim_ * sizeof(float), stream_);

    // ==========================================================================
    // PASS 1: Compute gradients for all timesteps (sequential due to recurrent)
    // ==========================================================================
    for (int t = steps - 1; t >= 0; --t) {
        const T* h_prev = h + t * BD;
        const T* h_t = h + (t + 1) * BD;
        const T* x_t = x + t * BD;
        const T* v_t = v + t * BD;
        const T* alpha_t = alpha_cache + t * BD;
        const T* delta_t = delta_cache + t * BD;
        const T* d_out_t = d_output + t * BD;

        // Storage for this timestep
        T* dv_t = dv_all + t * BD;
        T* d_alpha_t = d_alpha_all + t * BD;
        T* d_delta_t = d_delta_all + t * BD;
        T* dx_gate_t = dx_gate + t * BD;

        // Backward through h+x selective output
        SelectiveOutputBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, x_t, b_gate, d_out_t, dh_gate, dx_gate_t, db_gate_f);

        // dh = dh_gate (from h+x path)
        cudaMemcpyAsync(dh, dh_gate, BD * sizeof(T), cudaMemcpyDeviceToDevice, stream_);

        // Add dh_recurrent to dh
        VectorAddInplace<T><<<num_blocks, block_size, 0, stream_>>>(BD, dh_recurrent, dh);

        // Backward through polynomial gated update
        LinearPolynomialGatedBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_prev, v_t, alpha_t, delta_t, r_h, dh, nullptr,
            dv_t, d_alpha_t, d_delta_t, dh_prev_out, dr_h_f, db_f, db_delta_f);

        // dh_recurrent = dh_prev_out for next iteration
        cudaMemcpyAsync(dh_recurrent, dh_prev_out, BD * sizeof(T), cudaMemcpyDeviceToDevice, stream_);
    }

    // ==========================================================================
    // PASS 2: Batch GEMMs across all timesteps (Haste pattern)
    // ==========================================================================

    // dx = W_x @ dv_all + W_alpha @ d_alpha_all + W_delta @ d_delta_all + dx_gate
    blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_, &alpha_blas, W_x, dim_, dv_all, dim_, &beta_zero, dx, dim_);
    blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_, &alpha_blas, W_alpha, dim_, d_alpha_all, dim_, &beta_one, dx, dim_);
    blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_, &alpha_blas, W_delta, dim_, d_delta_all, dim_, &beta_one, dx, dim_);

    // Add dx_gate to dx
    VectorAddInplace<T><<<(TBD + block_size - 1) / block_size, block_size, 0, stream_>>>(TBD, dx_gate, dx);

    // dW_x = dv_all @ x^T (all timesteps)
    blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_, &alpha_blas, dv_all, dim_, x, dim_, &beta_one, dW_x, dim_);

    // dW_alpha = d_alpha_all @ x^T (all timesteps)
    blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_, &alpha_blas, d_alpha_all, dim_, x, dim_, &beta_one, dW_alpha, dim_);

    // dW_delta = d_delta_all @ x^T (all timesteps)
    blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_, &alpha_blas, d_delta_all, dim_, x, dim_, &beta_one, dW_delta, dim_);

    // Copy from float buffers using kernel
    CopyFloatToT<T><<<dim_blocks, block_size, 0, stream_>>>(dim_, dr_h_f, dr_h);
    CopyFloatToT<T><<<dim_blocks, block_size, 0, stream_>>>(dim_, db_f, db);
    CopyFloatToT<T><<<dim_blocks, block_size, 0, stream_>>>(dim_, db_delta_f, db_delta);
    CopyFloatToT<T><<<dim_blocks, block_size, 0, stream_>>>(dim_, db_alpha_f, db_alpha);
    CopyFloatToT<T><<<dim_blocks, block_size, 0, stream_>>>(dim_, db_gate_f, db_gate);
}

// Explicit instantiations
template class LinearPolynomialForward<float>;
template class LinearPolynomialForward<__half>;
template class LinearPolynomialForward<__nv_bfloat16>;
template class LinearPolynomialForward<double>;
template class LinearPolynomialBackward<float>;
template class LinearPolynomialBackward<__half>;
template class LinearPolynomialBackward<__nv_bfloat16>;
template class LinearPolynomialBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
