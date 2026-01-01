// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// Level 6: Linear Polynomial Elman
// alpha = 1 + softplus(W_alpha @ x + b_alpha)
// v = W_x @ x + r_h * h_prev + b
// candidate = sign(v) * |v|^alpha (bounded)
// delta = sigmoid(W_delta @ x + b_delta)
// h_new = (1-delta) * h_prev + delta * candidate
// compete = softmax(h.reshape(groups), dim=-1)
// output = compete * silu(W_out @ h)

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

        // Polynomial activation: sign(v) * |v|^alpha, bounded
        float sign_v = (v >= 0.0f) ? 1.0f : -1.0f;
        float abs_v = fabsf(v);
        // Bound |v| to avoid explosion
        abs_v = fminf(abs_v, 10.0f);
        float candidate = sign_v * powf(abs_v + 1e-6f, alpha);
        // Bound candidate
        candidate = fmaxf(fminf(candidate, 10.0f), -10.0f);

        // Gated update: h = (1 - delta) * h_prev + delta * candidate
        float h_new = (1.0f - delta) * h_p + delta * candidate;
        h_out[idx] = static_cast<T>(h_new);
    }
}

// Kernel: Compute compete×silu output
template<typename T>
__global__ void LinearPolynomialOutput(
    const int batch_size,
    const int dim,
    const int n_groups,
    const int group_size,
    const T* __restrict__ h,
    const T* __restrict__ w_out_h,
    T* __restrict__ output,
    T* __restrict__ compete_cache) {

    extern __shared__ float smem[];

    const int b = blockIdx.x;
    const int g = blockIdx.y;

    if (b >= batch_size || g >= n_groups) return;

    const int base = b * dim + g * group_size;

    // Find max for softmax stability
    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        max_val = fmaxf(max_val, static_cast<float>(h[base + i]));
    }
    smem[threadIdx.x] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        __syncthreads();
    }
    max_val = smem[0];
    __syncthreads();

    // Compute exp sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        sum += expf(static_cast<float>(h[base + i]) - max_val);
    }
    float* sum_smem = smem + blockDim.x;
    sum_smem[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sum_smem[threadIdx.x] += sum_smem[threadIdx.x + s];
        __syncthreads();
    }
    sum = sum_smem[0];

    // Compute output = compete * silu(w_out_h)
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float compete = expf(static_cast<float>(h[base + i]) - max_val) / sum;
        if (compete_cache) compete_cache[base + i] = static_cast<T>(compete);

        float w = static_cast<float>(w_out_h[base + i]);
        float silu_val = w / (1.0f + expf(-w));
        output[base + i] = static_cast<T>(compete * silu_val);
    }
}

// Backward through selective output
template<typename T>
__global__ void LinearPolynomialOutputBackward(
    const int batch_size,
    const int dim,
    const int n_groups,
    const int group_size,
    const T* __restrict__ h,
    const T* __restrict__ w_out_h,
    const T* __restrict__ compete,
    const T* __restrict__ d_output,
    T* __restrict__ dh_compete,
    T* __restrict__ d_w_out_h) {

    extern __shared__ float smem[];

    const int b = blockIdx.x;
    const int g = blockIdx.y;

    if (b >= batch_size || g >= n_groups) return;

    const int base = b * dim + g * group_size;

    float sum_compete_dcompete = 0.0f;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float dout = static_cast<float>(d_output[base + i]);
        float w = static_cast<float>(w_out_h[base + i]);
        float sig = 1.0f / (1.0f + expf(-w));
        float silu_val = w * sig;
        float dsilu_dw = sig * (1.0f + w * (1.0f - sig));
        float comp = static_cast<float>(compete[base + i]);

        d_w_out_h[base + i] = static_cast<T>(dout * comp * dsilu_dw);
        sum_compete_dcompete += comp * dout * silu_val;
    }

    smem[threadIdx.x] = sum_compete_dcompete;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    sum_compete_dcompete = smem[0];

    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float dout = static_cast<float>(d_output[base + i]);
        float w = static_cast<float>(w_out_h[base + i]);
        float sig = 1.0f / (1.0f + expf(-w));
        float silu_val = w * sig;
        float comp = static_cast<float>(compete[base + i]);
        float d_comp = dout * silu_val;
        dh_compete[base + i] = static_cast<T>(comp * (d_comp - sum_compete_dcompete));
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

        // Compute candidate and its derivatives
        float sign_v = (v_val >= 0.0f) ? 1.0f : -1.0f;
        float abs_v = fabsf(v_val);
        abs_v = fminf(abs_v, 10.0f);
        float abs_v_eps = abs_v + 1e-6f;
        float candidate = sign_v * powf(abs_v_eps, alpha_val);
        candidate = fmaxf(fminf(candidate, 10.0f), -10.0f);

        // d_candidate/dv = alpha * sign(v) * |v|^(alpha-1)
        float d_cand_dv = alpha_val * sign_v * powf(abs_v_eps, alpha_val - 1.0f);
        // Bound gradient
        d_cand_dv = fmaxf(fminf(d_cand_dv, 100.0f), -100.0f);

        // d_candidate/d_alpha = sign(v) * |v|^alpha * log(|v|)
        float log_abs_v = logf(abs_v_eps);
        float d_cand_dalpha = sign_v * powf(abs_v_eps, alpha_val) * log_abs_v;
        d_cand_dalpha = fmaxf(fminf(d_cand_dalpha, 100.0f), -100.0f);

        float d_cand = grad_h * del;
        float dv_val = d_cand * d_cand_dv;
        dv[idx] = static_cast<T>(dv_val);

        // d_alpha = d_cand * d_cand_dalpha
        float d_alpha = d_cand * d_cand_dalpha;
        // d_alpha_raw = d_alpha * softplus'(alpha_raw) = d_alpha * sigmoid(alpha_raw)
        // alpha = 1 + softplus(alpha_raw)
        // softplus'(x) = sigmoid(x)
        float alpha_raw = alpha_val - 1.0f;  // Approximate
        float sig_alpha = 1.0f / (1.0f + expf(-alpha_raw));
        d_alpha_raw[idx] = static_cast<T>(d_alpha * sig_alpha);

        // d_delta
        float d_delta = grad_h * (candidate - h_p);
        float dsigmoid = del * one_minus_del;
        float d_delta_raw_val = d_delta * dsigmoid;
        d_delta_raw[idx] = static_cast<T>(d_delta_raw_val);

        // dh_prev from gated path and r_h path
        // Clamp r_h to <= 0.9 for stability (must match forward)
        float dh_prev_gated = one_minus_del * grad_h;
        float r_h_val = fminf(static_cast<float>(r_h[d]), 0.9f);
        float dh_prev_rh = dv_val * r_h_val;
        dh_prev_out[idx] = static_cast<T>(dh_prev_gated + dh_prev_rh);

        // dr_h: gradient for diagonal element
        float dr_h_val = dv_val * h_p;
        atomicAdd(&dr_h[d], dr_h_val);

        atomicAdd(&db[d], dv_val);
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
    int n_groups,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : training_(training),
      batch_size_(batch_size),
      dim_(dim),
      n_groups_(n_groups),
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
    const T* W_out,
    const T* b,
    const T* b_delta,
    const T* x,
    T* h,
    T* output,
    T* v,
    T* alpha_cache,
    T* delta_cache,
    T* compete_cache) {

    static const T alpha_blas = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int BD = batch_size_ * dim_;
    const int TBD = steps * BD;  // Total batch*dim for all timesteps
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;
    const int total_blocks = (TBD + block_size - 1) / block_size;
    const int group_size = dim_ / n_groups_;

    // =========================================================================
    // Haste pattern: Pre-compute ALL input projections in big GEMMs
    // =========================================================================

    // Workspace for pre-computed projections (all timesteps)
    T *all_wx_x, *all_alpha_raw, *all_delta_raw, *w_out_h;
    cudaMalloc(&all_wx_x, TBD * sizeof(T));
    cudaMalloc(&all_alpha_raw, TBD * sizeof(T));
    cudaMalloc(&all_delta_raw, TBD * sizeof(T));
    cudaMalloc(&w_out_h, BD * sizeof(T));

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
        T* v_t = training_ ? (v + t * BD) : nullptr;
        T* alpha_t = training_ ? (alpha_cache + t * BD) : nullptr;
        T* delta_t = training_ ? (delta_cache + t * BD) : nullptr;
        T* compete_t = training_ ? (compete_cache + t * BD) : nullptr;

        // Get pre-computed projections for this timestep
        const T* wx_x_t = all_wx_x + t * BD;
        const T* alpha_raw_t = all_alpha_raw + t * BD;
        const T* delta_raw_t = all_delta_raw + t * BD;

        // Polynomial gated update (no input GEMM needed - already pre-computed)
        LinearPolynomialGatedUpdate<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_prev, wx_x_t, r_h, alpha_raw_t, delta_raw_t, b, h_t, v_t, alpha_t, delta_t);

        // w_out_h = h_t @ W_out.T (output projection - depends on h_t so can't pre-compute)
        blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha_blas, W_out, dim_, h_t, dim_, &beta_zero, w_out_h, dim_);

        // Selective output with compete mechanism
        dim3 grid(batch_size_, n_groups_);
        int smem_size = 2 * block_size * sizeof(float);
        LinearPolynomialOutput<T><<<grid, block_size, smem_size, stream_>>>(
            batch_size_, dim_, n_groups_, group_size, h_t, w_out_h, out_t, compete_t);
    }

    cudaFree(all_wx_x);
    cudaFree(all_alpha_raw);
    cudaFree(all_delta_raw);
    cudaFree(w_out_h);
}

// =============================================================================
// Linear Polynomial Backward
// =============================================================================

template<typename T>
LinearPolynomialBackward<T>::LinearPolynomialBackward(
    int batch_size,
    int dim,
    int n_groups,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      n_groups_(n_groups),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void LinearPolynomialBackward<T>::Run(
    int steps,
    const T* W_x,
    const T* r_h,
    const T* W_alpha,
    const T* W_delta,
    const T* W_out,
    const T* x,
    const T* h,
    const T* v,
    const T* alpha_cache,
    const T* delta_cache,
    const T* compete_cache,
    const T* d_output,
    T* dx,
    T* dW_x,
    T* dr_h,
    T* dW_alpha,
    T* db_alpha,
    T* dW_delta,
    T* dW_out,
    T* db,
    T* db_delta,
    T* workspace) {

    static const T alpha_blas = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int BD = batch_size_ * dim_;
    const int TBD = steps * BD;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;
    const int dim_blocks = (dim_ + block_size - 1) / block_size;
    const int group_size = dim_ / n_groups_;

    // ==========================================================================
    // HASTE PATTERN: Workspace from Python (PyTorch caching allocator)
    // Layout: [dv_all: TBD][d_alpha_all: TBD][d_delta_all: TBD][d_w_out_h_all: TBD]
    //         [w_out_h: BD][dh_compete: BD][dh: BD][dh_prev_out: BD][dh_recurrent: BD]
    //         [dr_h_f: dim floats][db_f: dim][db_delta_f: dim][db_alpha_f: dim]
    // ==========================================================================

    T* dv_all = workspace;
    T* d_alpha_all = dv_all + TBD;
    T* d_delta_all = d_alpha_all + TBD;
    T* d_w_out_h_all = d_delta_all + TBD;
    T* w_out_h = d_w_out_h_all + TBD;
    T* dh_compete = w_out_h + BD;
    T* dh = dh_compete + BD;
    T* dh_prev_out = dh + BD;
    T* dh_recurrent = dh_prev_out + BD;
    float* dr_h_f = reinterpret_cast<float*>(dh_recurrent + BD);
    float* db_f = dr_h_f + dim_;
    float* db_delta_f = db_f + dim_;
    float* db_alpha_f = db_delta_f + dim_;

    // Zero out weight gradients and workspace (all async on same stream)
    cudaMemsetAsync(dW_x, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dW_alpha, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dW_delta, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dW_out, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(T), stream_);
    cudaMemsetAsync(dr_h_f, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_f, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_delta_f, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_alpha_f, 0, dim_ * sizeof(float), stream_);

    // ==========================================================================
    // PASS 1: Compute gradients for all timesteps (sequential due to recurrent)
    // ==========================================================================
    for (int t = steps - 1; t >= 0; --t) {
        const T* h_prev = h + t * BD;
        const T* h_t = h + (t + 1) * BD;
        const T* v_t = v + t * BD;
        const T* alpha_t = alpha_cache + t * BD;
        const T* delta_t = delta_cache + t * BD;
        const T* compete_t = compete_cache + t * BD;
        const T* d_out_t = d_output + t * BD;

        // Storage for this timestep
        T* dv_t = dv_all + t * BD;
        T* d_alpha_t = d_alpha_all + t * BD;
        T* d_delta_t = d_delta_all + t * BD;
        T* d_w_out_h_t = d_w_out_h_all + t * BD;

        // Recompute w_out_h = h_t @ W_out.T
        blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha_blas, W_out, dim_, h_t, dim_, &beta_zero, w_out_h, dim_);

        // Backward through output
        dim3 grid(batch_size_, n_groups_);
        int smem_size = block_size * sizeof(float);
        LinearPolynomialOutputBackward<T><<<grid, block_size, smem_size, stream_>>>(
            batch_size_, dim_, n_groups_, group_size, h_t, w_out_h, compete_t, d_out_t, dh_compete, d_w_out_h_t);

        // dh from W_out path: dh_wout = W_out @ d_w_out_h
        // Combine with dh_compete into dh
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha_blas, W_out, dim_, d_w_out_h_t, dim_, &beta_zero, dh, dim_);
        VectorAdd<T><<<num_blocks, block_size, 0, stream_>>>(BD, dh_compete, dh);

        // Backward through polynomial gated update
        LinearPolynomialGatedBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_prev, v_t, alpha_t, delta_t, r_h, dh, dh_recurrent,
            dv_t, d_alpha_t, d_delta_t, dh_prev_out, dr_h_f, db_f, db_delta_f);

        // dh_recurrent = dh_prev_out for next iteration
        cudaMemcpyAsync(dh_recurrent, dh_prev_out, BD * sizeof(T), cudaMemcpyDeviceToDevice, stream_);
    }

    // ==========================================================================
    // PASS 2: Batch GEMMs across all timesteps (Haste pattern)
    // ==========================================================================

    // dx = W_x @ dv_all + W_alpha @ d_alpha_all + W_delta @ d_delta_all
    blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_, &alpha_blas, W_x, dim_, dv_all, dim_, &beta_zero, dx, dim_);
    blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_, &alpha_blas, W_alpha, dim_, d_alpha_all, dim_, &beta_one, dx, dim_);
    blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_, &alpha_blas, W_delta, dim_, d_delta_all, dim_, &beta_one, dx, dim_);

    // dW_x = dv_all @ x^T (all timesteps)
    blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_, &alpha_blas, dv_all, dim_, x, dim_, &beta_one, dW_x, dim_);

    // dW_alpha = d_alpha_all @ x^T (all timesteps)
    blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_, &alpha_blas, d_alpha_all, dim_, x, dim_, &beta_one, dW_alpha, dim_);

    // dW_delta = d_delta_all @ x^T (all timesteps)
    blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_, &alpha_blas, d_delta_all, dim_, x, dim_, &beta_one, dW_delta, dim_);

    // dW_out = d_w_out_h_all @ h[1:T+1]^T (all timesteps)
    // h[1:T+1] is the output hidden states
    blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_, &alpha_blas, d_w_out_h_all, dim_, h + BD, dim_, &beta_one, dW_out, dim_);

    // Copy from float buffers using kernel
    CopyFloatToT<T><<<dim_blocks, block_size, 0, stream_>>>(dim_, dr_h_f, dr_h);
    CopyFloatToT<T><<<dim_blocks, block_size, 0, stream_>>>(dim_, db_f, db);
    CopyFloatToT<T><<<dim_blocks, block_size, 0, stream_>>>(dim_, db_delta_f, db_delta);
    CopyFloatToT<T><<<dim_blocks, block_size, 0, stream_>>>(dim_, db_alpha_f, db_alpha);
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
