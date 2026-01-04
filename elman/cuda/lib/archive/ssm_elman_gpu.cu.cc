// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// SSM Elman - Mamba2-style state space model with W_h mixing
//
// Core equations:
//   B_t = W_B @ x_t                           (input projection)
//   C_t = W_C @ x_t                           (output projection)
//   dt_t = softplus(W_dt @ x_t + b_dt)        (input-dependent timestep)
//   A = sigmoid(a_log)                        (learned diagonal decay)
//   h_t = A * h_{t-1} + W_h @ h_{t-1} + dt_t * B_t   (SSM + mixing)
//   y_t = C_t * h_t                           (selective output)
//
// Key features:
//   - Diagonal A (stable decay like Mamba2)
//   - W_h mixing (cross-dimension communication like Elman)
//   - Input-dependent B, C, dt (selectivity)
//   - No tanh - linear dynamics with selective gating

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#include "hasty/elman_ladder.h"
#include "blas.h"
#include "inline_ops.h"

namespace {

// Softplus: log(1 + exp(x))
__device__ __forceinline__ float softplus(float x) {
    if (x > 20.0f) return x;  // Avoid overflow
    if (x < -20.0f) return 0.0f;
    return logf(1.0f + expf(x));
}

// Softplus derivative: sigmoid(x)
__device__ __forceinline__ float softplus_grad(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Kernel: SSM forward step with W_h mixing
// h_t = A * h_{t-1} + H_proj + dt * B
// y_t = C * h_t
template<typename T>
__global__ void SSMElmanForwardKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ h_prev,       // [B, dim]
    const T* __restrict__ B_proj,       // [B, dim] pre-computed W_B @ x
    const T* __restrict__ C_proj,       // [B, dim] pre-computed W_C @ x
    const T* __restrict__ H_proj,       // [B, dim] pre-computed W_h @ h_prev
    const T* __restrict__ dt_proj,      // [B, dim] pre-computed W_dt @ x
    const T* __restrict__ a_log,        // [dim] log of decay (before sigmoid)
    const T* __restrict__ b_dt,         // [dim] dt bias
    T* __restrict__ h_out,              // [B, dim]
    T* __restrict__ y_out,              // [B, dim]
    T* __restrict__ dt_cache) {         // [B, dim] cache for backward (optional)

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float h_p = static_cast<float>(h_prev[idx]);
        float B_val = static_cast<float>(B_proj[idx]);
        float C_val = static_cast<float>(C_proj[idx]);
        float H_val = static_cast<float>(H_proj[idx]);  // W_h @ h_prev
        float dt_raw = static_cast<float>(dt_proj[idx]) + static_cast<float>(b_dt[d]);

        // A = sigmoid(a_log) - decay factor in (0, 1)
        float a_log_val = static_cast<float>(a_log[d]);
        float A_val = 1.0f / (1.0f + expf(-a_log_val));

        // dt = softplus(dt_raw)
        float dt_val = softplus(dt_raw);

        // h_t = A * h_{t-1} + W_h @ h_{t-1} + dt * B
        float h_new = A_val * h_p + H_val + dt_val * B_val;

        // y_t = C * h_t
        float y_val = C_val * h_new;

        h_out[idx] = static_cast<T>(h_new);
        y_out[idx] = static_cast<T>(y_val);

        if (dt_cache) dt_cache[idx] = static_cast<T>(dt_raw);
    }
}

// Kernel: SSM backward step with W_h mixing
// h = A * h_prev + H_proj + dt * B  (where H_proj = W_h @ h_prev)
// y = C * h
// dC = dy * h
// dh = dy * C + dh_next
// dH_proj = dh  (gradient for W_h @ h_prev term)
// dh_prev_from_A = dh * A
// Note: Total dh_prev = dh * A + W_h^T @ dH_proj (done via GEMM in Run)
template<typename T>
__global__ void SSMElmanBackwardKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ h_prev,       // [B, dim]
    const T* __restrict__ h_curr,       // [B, dim]
    const T* __restrict__ B_proj,       // [B, dim]
    const T* __restrict__ C_proj,       // [B, dim] C projection values
    const T* __restrict__ dt_cache,     // [B, dim] pre-softplus dt
    const T* __restrict__ a_log,        // [dim]
    const T* __restrict__ dy,           // [B, dim] gradient from output
    const T* __restrict__ dh_next,      // [B, dim] gradient from next timestep
    T* __restrict__ dh_prev_A,          // [B, dim] dh * A (partial, needs W_h^T @ dh added)
    T* __restrict__ dB_proj,            // [B, dim]
    T* __restrict__ dC_proj,            // [B, dim]
    T* __restrict__ dH_proj,            // [B, dim] gradient for W_h @ h_prev
    T* __restrict__ ddt_proj,           // [B, dim]
    float* __restrict__ da_log,         // [dim] accumulated
    float* __restrict__ db_dt) {        // [dim] accumulated

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float h_p = static_cast<float>(h_prev[idx]);
        float h_c = static_cast<float>(h_curr[idx]);
        float B_val = static_cast<float>(B_proj[idx]);
        float C_val = static_cast<float>(C_proj[idx]);
        float dt_raw = static_cast<float>(dt_cache[idx]);

        float a_log_val = static_cast<float>(a_log[d]);
        float A_val = 1.0f / (1.0f + expf(-a_log_val));
        float dt_val = softplus(dt_raw);

        float dy_val = static_cast<float>(dy[idx]);
        float dh_n = dh_next ? static_cast<float>(dh_next[idx]) : 0.0f;

        // y = C * h
        // dC = dy * h
        // dh = dy * C + dh_next
        float dC = dy_val * h_c;
        float dh = dy_val * C_val + dh_n;

        // h = A * h_prev + H_proj + dt * B
        // dh_prev_from_A = dh * A
        // dH_proj = dh  (for W_h gradient)
        // dA = dh * h_prev
        // ddt = dh * B
        // dB = dh * dt

        float dh_p_A = dh * A_val;  // Partial - W_h^T @ dh added later
        float dH = dh;              // Gradient for H_proj = W_h @ h_prev
        float dA = dh * h_p;
        float ddt = dh * B_val;
        float dB = dh * dt_val;

        // dt = softplus(dt_raw) => ddt_raw = ddt * sigmoid(dt_raw)
        float ddt_raw = ddt * softplus_grad(dt_raw);

        // A = sigmoid(a_log) => da_log = dA * A * (1 - A)
        float da_log_val = dA * A_val * (1.0f - A_val);

        dh_prev_A[idx] = static_cast<T>(dh_p_A);
        dB_proj[idx] = static_cast<T>(dB);
        dC_proj[idx] = static_cast<T>(dC);
        dH_proj[idx] = static_cast<T>(dH);
        ddt_proj[idx] = static_cast<T>(ddt_raw);

        atomicAdd(&da_log[d], da_log_val);
        atomicAdd(&db_dt[d], ddt_raw);
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

// Simple vector add: dst = a + b
template<typename T>
__global__ void VectorAddKernel(const int n, const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = a[idx] + b[idx];
    }
}

}  // anonymous namespace


namespace hasty {
namespace v0 {
namespace elman_ladder {

// =============================================================================
// SSM Elman Forward
// =============================================================================

template<typename T>
SSMElmanForward<T>::SSMElmanForward(
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
void SSMElmanForward<T>::Run(
    int steps,
    const T* W_B,       // [dim, dim] input -> B projection
    const T* W_C,       // [dim, dim] input -> C projection
    const T* W_h,       // [dim, dim] hidden -> hidden mixing
    const T* W_dt,      // [dim, dim] input -> dt projection
    const T* a_log,     // [dim] log decay (before sigmoid)
    const T* b_dt,      // [dim] dt bias
    const T* x,         // [T, B, dim] input sequence
    T* h,               // [T+1, B, dim] hidden states
    T* y,               // [T, B, dim] output
    T* B_proj_cache,    // [T, B, dim] B projections (for backward)
    T* C_proj_cache,    // [T, B, dim] C projections
    T* dt_cache) {      // [T, B, dim] pre-softplus dt values

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int BD = batch_size_ * dim_;
    const int TBD = steps * BD;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Pre-compute all projections for all timesteps
    // B_proj = W_B @ x for all t
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha,
        W_B, dim_,
        x, dim_,
        &beta_zero,
        B_proj_cache, dim_);

    // C_proj = W_C @ x for all t
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha,
        W_C, dim_,
        x, dim_,
        &beta_zero,
        C_proj_cache, dim_);

    // dt_proj = W_dt @ x for all t (bias added in kernel)
    T* dt_proj_all;
    cudaMalloc(&dt_proj_all, TBD * sizeof(T));

    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha,
        W_dt, dim_,
        x, dim_,
        &beta_zero,
        dt_proj_all, dim_);

    // Allocate H_proj for W_h @ h (one timestep at a time)
    T* H_proj;
    cudaMalloc(&H_proj, BD * sizeof(T));

    // Per-timestep forward
    for (int t = 0; t < steps; ++t) {
        const T* h_prev = h + t * BD;
        const T* B_proj_t = B_proj_cache + t * BD;
        const T* C_proj_t = C_proj_cache + t * BD;
        const T* dt_proj_t = dt_proj_all + t * BD;
        T* h_t = h + (t + 1) * BD;
        T* y_t = y + t * BD;
        T* dt_t = training_ ? (dt_cache + t * BD) : nullptr;

        // Compute H_proj = W_h @ h_prev
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha,
            W_h, dim_,
            h_prev, dim_,
            &beta_zero,
            H_proj, dim_);

        SSMElmanForwardKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_,
            h_prev, B_proj_t, C_proj_t, H_proj, dt_proj_t,
            a_log, b_dt,
            h_t, y_t, dt_t);
    }

    cudaFree(H_proj);
    cudaFree(dt_proj_all);
}

// =============================================================================
// SSM Elman Backward
// =============================================================================

template<typename T>
SSMElmanBackward<T>::SSMElmanBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void SSMElmanBackward<T>::Run(
    int steps,
    const T* W_B,
    const T* W_C,
    const T* W_h,
    const T* W_dt,
    const T* a_log,
    const T* x,
    const T* h,
    const T* B_proj_cache,
    const T* C_proj_cache,
    const T* dt_cache,
    const T* dy,
    T* dx,
    T* dW_B,
    T* dW_C,
    T* dW_h,
    T* dW_dt,
    T* da_log,
    T* db_dt,
    T* workspace) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int BD = batch_size_ * dim_;
    const int TBD = steps * BD;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Workspace layout:
    // [dB_proj_all: TBD] [dC_proj_all: TBD] [dH_proj_all: TBD] [ddt_proj_all: TBD]
    // [dh_prev_A: BD] [dh_next: BD] [dh_from_Wh: BD] [da_log_float: dim] [db_dt_float: dim]
    T* dB_proj_all = workspace;
    T* dC_proj_all = workspace + TBD;
    T* dH_proj_all = workspace + 2 * TBD;
    T* ddt_proj_all = workspace + 3 * TBD;
    T* dh_prev_A = workspace + 4 * TBD;
    T* dh_next = workspace + 4 * TBD + BD;
    T* dh_from_Wh = workspace + 4 * TBD + 2 * BD;
    float* da_log_float = reinterpret_cast<float*>(workspace + 4 * TBD + 3 * BD);
    float* db_dt_float = da_log_float + dim_;

    // Initialize accumulators
    cudaMemsetAsync(dh_next, 0, BD * sizeof(T), stream_);
    cudaMemsetAsync(da_log_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_dt_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW_B, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dW_C, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dW_h, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dW_dt, 0, dim_ * dim_ * sizeof(T), stream_);

    // BPTT loop
    for (int t = steps - 1; t >= 0; --t) {
        const T* h_prev_t = h + t * BD;
        const T* h_curr_t = h + (t + 1) * BD;
        const T* B_proj_t = B_proj_cache + t * BD;
        const T* C_proj_t = C_proj_cache + t * BD;
        const T* dt_t = dt_cache + t * BD;
        const T* dy_t = dy + t * BD;
        T* dB_proj_t = dB_proj_all + t * BD;
        T* dC_proj_t = dC_proj_all + t * BD;
        T* dH_proj_t = dH_proj_all + t * BD;
        T* ddt_proj_t = ddt_proj_all + t * BD;

        SSMElmanBackwardKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_,
            h_prev_t, h_curr_t, B_proj_t, C_proj_t, dt_t, a_log,
            dy_t, dh_next,
            dh_prev_A, dB_proj_t, dC_proj_t, dH_proj_t, ddt_proj_t,
            da_log_float, db_dt_float);

        // dh_prev = dh * A + W_h^T @ dH_proj
        // First compute W_h^T @ dH_proj
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha,
            W_h, dim_,
            dH_proj_t, dim_,
            &beta_zero,
            dh_from_Wh, dim_);

        // Add dh_prev_A + dh_from_Wh -> dh_next for next iteration
        if (t > 0) {
            int add_blocks = (BD + 255) / 256;
            VectorAddKernel<T><<<add_blocks, 256, 0, stream_>>>(BD, dh_prev_A, dh_from_Wh, dh_next);
        }

        // Accumulate dW_h = dH_proj @ h_prev^T
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_,
            &alpha,
            h_prev_t, dim_,
            dH_proj_t, dim_,
            &beta_one,
            dW_h, dim_);
    }

    // dx = W_B @ dB_proj + W_C @ dC_proj + W_dt @ ddt_proj
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha,
        W_B, dim_,
        dB_proj_all, dim_,
        &beta_zero,
        dx, dim_);

    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha,
        W_C, dim_,
        dC_proj_all, dim_,
        &beta_one,
        dx, dim_);

    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha,
        W_dt, dim_,
        ddt_proj_all, dim_,
        &beta_one,
        dx, dim_);

    // dW_B = x^T @ dB_proj
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha,
        x, dim_,
        dB_proj_all, dim_,
        &beta_one,
        dW_B, dim_);

    // dW_C = x^T @ dC_proj
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha,
        x, dim_,
        dC_proj_all, dim_,
        &beta_one,
        dW_C, dim_);

    // dW_dt = x^T @ ddt_proj
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha,
        x, dim_,
        ddt_proj_all, dim_,
        &beta_one,
        dW_dt, dim_);

    // Copy float gradients to output
    CopyFloatToT<T><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, da_log_float, da_log);
    CopyFloatToT<T><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_dt_float, db_dt);
}

// Explicit template instantiations
template struct SSMElmanForward<__half>;
template struct SSMElmanForward<__nv_bfloat16>;
template struct SSMElmanForward<float>;
template struct SSMElmanForward<double>;

template struct SSMElmanBackward<__half>;
template struct SSMElmanBackward<__nv_bfloat16>;
template struct SSMElmanBackward<float>;
template struct SSMElmanBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
