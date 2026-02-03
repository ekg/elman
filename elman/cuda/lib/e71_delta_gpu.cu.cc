// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E71 Delta: Matrix Gated Elman with Delta Rule - State-dependent learning rate
//
// Architecture:
//     k_t = W_k @ x_t                              # [B, n] key
//     v_t = W_v @ x_t                              # [B, n] value
//     q_t = W_q @ x_t                              # [B, n] query
//     beta_x_t = W_beta @ x_t                      # [B, n]
//
//     # Key normalization for stability
//     k_norm = k / (||k|| + eps)                   # [B, n] normalized key
//
//     # S-dependent learning rate
//     retrieved = S @ k_norm                       # [B, n] (matrix-vector product)
//     beta = sigmoid(beta_x + d_beta * retrieved + b_beta)  # State-dependent learning rate
//
//     # Delta rule update
//     delta = v - retrieved                        # Error signal
//     S_new = S + beta.unsqueeze(-1) * outer(delta, k_norm)
//
//     # Self-gating output
//     out = S_new @ q
//     output = out * silu(out)
//
// Key optimization strategy:
// 1. Batch projections upfront: k_all, v_all, q_all, beta_x_all via 4 batched GEMMs
// 2. Sequential kernel for the matrix state recurrence (cannot parallelize due to S-dependence)
//
// State S is [B, n_state, n_state] - square matrix

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
// Forward Kernels
// =============================================================================

// Key normalization: k_norm = k / (||k|| + eps)
// k: [B, N] -> k_norm: [B, N]
__global__ void KeyNormKernel_E71Delta_BF16(
    const int batch_size,
    const int N,
    const __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ k_norm) {

    const int b = blockIdx.x;

    if (b < batch_size) {
        // Compute L2 norm of k[b]
        float norm_sq = 0.0f;
        for (int j = 0; j < N; ++j) {
            float k_val = __bfloat162float(k[b * N + j]);
            norm_sq += k_val * k_val;
        }
        float norm = sqrtf(norm_sq) + 1e-6f;

        // Normalize
        for (int j = threadIdx.x; j < N; j += blockDim.x) {
            float k_val = __bfloat162float(k[b * N + j]);
            k_norm[b * N + j] = __float2bfloat16(k_val / norm);
        }
    }
}

// Compute S @ k_norm -> retrieved (batched matrix-vector multiply)
// S: [B, N, N], k_norm: [B, N] -> retrieved: [B, N]
__global__ void MatVecKernel_E71Delta_BF16(
    const int batch_size,
    const int N,
    const __nv_bfloat16* __restrict__ S,
    const __nv_bfloat16* __restrict__ k_norm,
    __nv_bfloat16* __restrict__ retrieved) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int b = idx / N;
        const int n = idx % N;

        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            float s_val = __bfloat162float(S[b * N * N + n * N + j]);
            float k_val = __bfloat162float(k_norm[b * N + j]);
            sum += s_val * k_val;
        }
        retrieved[idx] = __float2bfloat16(sum);
    }
}

// Compute beta = sigmoid(beta_x + d_beta * retrieved + b_beta)
// beta_x: [B, N], retrieved: [B, N], d_beta: [N], b_beta: [N]
// Output: beta: [B, N] (state-dependent learning rate)
__global__ void BetaKernel_E71Delta_BF16(
    const int batch_size,
    const int N,
    const __nv_bfloat16* __restrict__ beta_x,
    const __nv_bfloat16* __restrict__ retrieved,
    const __nv_bfloat16* __restrict__ d_beta,
    const __nv_bfloat16* __restrict__ b_beta,
    __nv_bfloat16* __restrict__ beta) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int n = idx % N;

        float bx = __bfloat162float(beta_x[idx]);
        float ret = __bfloat162float(retrieved[idx]);
        float db = __bfloat162float(d_beta[n]);
        float bb = __bfloat162float(b_beta[n]);

        float logit = bx + db * ret + bb;
        float b = 1.0f / (1.0f + __expf(-logit));
        beta[idx] = __float2bfloat16(b);
    }
}

// Delta rule state update + output computation (fused)
// delta[b, n] = v[b, n] - retrieved[b, n]
// S_new[b, i, j] = S_prev[b, i, j] + beta[b, i] * delta[b, i] * k_norm[b, j]
// out[b, n] = sum_j(S_new[b, n, j] * q[b, j])
// output[b, n] = out * silu(out)
__global__ void DeltaUpdateOutputKernel_E71Delta_BF16(
    const int batch_size,
    const int N,
    const __nv_bfloat16* __restrict__ S_prev,
    const __nv_bfloat16* __restrict__ beta,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ retrieved,
    const __nv_bfloat16* __restrict__ k_norm,
    const __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ S_new,
    __nv_bfloat16* __restrict__ output) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int b = idx / N;
        const int n = idx % N;

        float beta_val = __bfloat162float(beta[b * N + n]);
        float v_val = __bfloat162float(v[b * N + n]);
        float retrieved_val = __bfloat162float(retrieved[b * N + n]);
        float delta = v_val - retrieved_val;

        float out_sum = 0.0f;

        for (int j = 0; j < N; ++j) {
            int s_idx = b * N * N + n * N + j;
            float s_prev_val = __bfloat162float(S_prev[s_idx]);
            float k_norm_val = __bfloat162float(k_norm[b * N + j]);
            float q_val = __bfloat162float(q[b * N + j]);

            // Delta rule: S_new = S_prev + beta * outer(delta, k_norm)
            float s_new_val = s_prev_val + beta_val * delta * k_norm_val;
            S_new[s_idx] = __float2bfloat16(s_new_val);

            out_sum += s_new_val * q_val;
        }

        float sigmoid_out = 1.0f / (1.0f + __expf(-out_sum));
        float silu_out = out_sum * sigmoid_out;
        output[idx] = __float2bfloat16(out_sum * silu_out);
    }
}

// =============================================================================
// Backward Kernels
// =============================================================================

// Backward through self-gate output: output = out * silu(out) where out = S @ q
// d(output)/d(out) = silu(out) * (2 + out*(1-sigmoid(out)))
// Also computes d_S and d_q from out = S @ q
__global__ void OutputBackwardKernel_E71Delta_BF16(
    const int batch_size,
    const int N,
    const __nv_bfloat16* __restrict__ S,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ d_output,
    __nv_bfloat16* __restrict__ d_S,
    float* __restrict__ d_q_f) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int b = idx / N;
        const int n = idx % N;

        float out_sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            float s_val = __bfloat162float(S[b * N * N + n * N + j]);
            float q_val = __bfloat162float(q[b * N + j]);
            out_sum += s_val * q_val;
        }

        float dout_val = __bfloat162float(d_output[idx]);
        float sigmoid_out = 1.0f / (1.0f + __expf(-out_sum));
        float silu_out = out_sum * sigmoid_out;
        float d_out = dout_val * silu_out * (2.0f + out_sum * (1.0f - sigmoid_out));

        for (int j = 0; j < N; ++j) {
            int s_idx = b * N * N + n * N + j;
            float q_val = __bfloat162float(q[b * N + j]);
            float s_val = __bfloat162float(S[s_idx]);

            float d_S_curr = __bfloat162float(d_S[s_idx]);
            d_S[s_idx] = __float2bfloat16(d_S_curr + d_out * q_val);
            atomicAdd(&d_q_f[b * N + j], d_out * s_val);
        }
    }
}

// Backward through delta rule state update:
// S_new = S_prev + beta * outer(delta, k_norm)  where delta = v - retrieved
// Given d_S_new, compute d_S_prev, d_beta, d_v, d_retrieved, d_k_norm
__global__ void DeltaUpdateBackwardKernel_E71Delta_BF16(
    const int batch_size,
    const int N,
    const __nv_bfloat16* __restrict__ beta,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ retrieved,
    const __nv_bfloat16* __restrict__ k_norm,
    const __nv_bfloat16* __restrict__ d_S_new,
    __nv_bfloat16* __restrict__ d_S_prev,
    float* __restrict__ d_beta_f,
    float* __restrict__ d_v_f,
    float* __restrict__ d_retrieved_f,
    float* __restrict__ d_k_norm_f) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N * N;

    if (idx < total) {
        const int j = idx % N;
        const int n = (idx / N) % N;
        const int b = idx / (N * N);

        const int bn_idx = b * N + n;
        const int bj_idx = b * N + j;

        float beta_val = __bfloat162float(beta[bn_idx]);
        float v_val = __bfloat162float(v[bn_idx]);
        float retrieved_val = __bfloat162float(retrieved[bn_idx]);
        float k_norm_val = __bfloat162float(k_norm[bj_idx]);
        float delta = v_val - retrieved_val;
        float ds_new = __bfloat162float(d_S_new[idx]);

        // d_S_prev = d_S_new (gradient passes through unchanged)
        d_S_prev[idx] = __float2bfloat16(ds_new);

        // d_beta[b,n] += d_S_new[b,n,j] * delta[b,n] * k_norm[b,j]
        float d_beta_contrib = ds_new * delta * k_norm_val;
        atomicAdd(&d_beta_f[bn_idx], d_beta_contrib);

        // d_delta = d_S_new * beta * k_norm
        // d_v += d_delta (since delta = v - retrieved)
        // d_retrieved -= d_delta
        float d_delta = ds_new * beta_val * k_norm_val;
        atomicAdd(&d_v_f[bn_idx], d_delta);
        atomicAdd(&d_retrieved_f[bn_idx], -d_delta);

        // d_k_norm[b,j] += d_S_new[b,n,j] * beta[b,n] * delta[b,n]
        float d_k_norm_contrib = ds_new * beta_val * delta;
        atomicAdd(&d_k_norm_f[bj_idx], d_k_norm_contrib);
    }
}

// Backward through beta computation:
// beta = sigmoid(beta_x + d_beta * retrieved + b_beta)
// Given d_beta_in, compute d_beta_x, add to d_retrieved, accumulate dd_beta, db_beta
__global__ void BetaBackwardKernel_E71Delta_BF16(
    const int batch_size,
    const int N,
    const __nv_bfloat16* __restrict__ beta_x,
    const __nv_bfloat16* __restrict__ retrieved,
    const __nv_bfloat16* __restrict__ d_beta_param,
    const __nv_bfloat16* __restrict__ b_beta,
    const float* __restrict__ d_beta_f,
    __nv_bfloat16* __restrict__ d_beta_x,
    float* __restrict__ d_retrieved_f,
    float* __restrict__ dd_beta_f,
    float* __restrict__ db_beta_f) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int n = idx % N;

        float bx = __bfloat162float(beta_x[idx]);
        float ret = __bfloat162float(retrieved[idx]);
        float db = __bfloat162float(d_beta_param[n]);
        float bb = __bfloat162float(b_beta[n]);
        float logit = bx + db * ret + bb;
        float beta_val = 1.0f / (1.0f + __expf(-logit));

        float d_beta_in = d_beta_f[idx];
        float d_logit = d_beta_in * beta_val * (1.0f - beta_val);

        d_beta_x[idx] = __float2bfloat16(d_logit);
        atomicAdd(&d_retrieved_f[idx], d_logit * db);

        atomicAdd(&dd_beta_f[n], d_logit * ret);
        atomicAdd(&db_beta_f[n], d_logit);
    }
}

// Backward through retrieval: retrieved = S @ k_norm
// Given d_retrieved, compute d_S and accumulate d_k_norm
__global__ void RetrievalBackwardKernel_E71Delta_BF16(
    const int batch_size,
    const int N,
    const __nv_bfloat16* __restrict__ S,
    const __nv_bfloat16* __restrict__ k_norm,
    const float* __restrict__ d_retrieved_f,
    __nv_bfloat16* __restrict__ d_S,
    float* __restrict__ d_k_norm_f) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int b = idx / N;
        const int n = idx % N;

        float d_ret = d_retrieved_f[idx];

        for (int j = 0; j < N; ++j) {
            int s_idx = b * N * N + n * N + j;
            float k_val = __bfloat162float(k_norm[b * N + j]);
            float s_val = __bfloat162float(S[s_idx]);

            d_S[s_idx] = __float2bfloat16(__bfloat162float(d_S[s_idx]) + d_ret * k_val);
            atomicAdd(&d_k_norm_f[b * N + j], d_ret * s_val);
        }
    }
}

// Backward through key normalization: k_norm = k / (||k|| + eps)
// Given d_k_norm, compute d_k
__global__ void KeyNormBackwardKernel_E71Delta_BF16(
    const int batch_size,
    const int N,
    const __nv_bfloat16* __restrict__ k,
    const float* __restrict__ d_k_norm_f,
    float* __restrict__ d_k_f) {

    const int b = blockIdx.x;

    if (b < batch_size) {
        float norm_sq = 0.0f;
        for (int j = 0; j < N; ++j) {
            float k_val = __bfloat162float(k[b * N + j]);
            norm_sq += k_val * k_val;
        }
        float norm = sqrtf(norm_sq) + 1e-6f;
        float norm_inv = 1.0f / norm;

        float dot = 0.0f;
        for (int j = 0; j < N; ++j) {
            float k_val = __bfloat162float(k[b * N + j]);
            dot += k_val * d_k_norm_f[b * N + j];
        }

        for (int j = threadIdx.x; j < N; j += blockDim.x) {
            float k_val = __bfloat162float(k[b * N + j]);
            float d_kn = d_k_norm_f[b * N + j];
            float k_norm_j = k_val * norm_inv;
            float d_k_contrib = (d_kn - k_norm_j * dot * norm_inv) * norm_inv;
            atomicAdd(&d_k_f[b * N + j], d_k_contrib);
        }
    }
}

// Copy float to BF16
__global__ void CopyFloatToBF16_E71Delta(const int n, const float* __restrict__ src, __nv_bfloat16* __restrict__ dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2bfloat16(src[idx]);
    }
}

}  // anonymous namespace


namespace hasty {
namespace v0 {
namespace elman_ladder {

// =============================================================================
// E71 Delta Forward - BF16 Specialization
// =============================================================================

template<>
E71DeltaForward<__nv_bfloat16>::E71DeltaForward(
    bool training,
    int batch_size,
    int n_state,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : training_(training),
      batch_size_(batch_size),
      n_state_(n_state),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void E71DeltaForward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_k,
    const __nv_bfloat16* W_v,
    const __nv_bfloat16* W_q,
    const __nv_bfloat16* W_beta,
    const __nv_bfloat16* d_beta,
    const __nv_bfloat16* b_beta,
    const __nv_bfloat16* x,
    __nv_bfloat16* S,
    __nv_bfloat16* output,
    __nv_bfloat16* k_cache,
    __nv_bfloat16* v_cache,
    __nv_bfloat16* q_cache,
    __nv_bfloat16* beta_x_cache,
    __nv_bfloat16* retrieved_cache,
    __nv_bfloat16* beta_cache,
    __nv_bfloat16* k_norm_cache,
    __nv_bfloat16* workspace) {

    static const __nv_bfloat16 alpha_one = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);

    const int BN = batch_size_ * n_state_;
    const int BNN = batch_size_ * n_state_ * n_state_;
    const int block_size = 256;

    // Workspace layout:
    // k_all: [T, B, n_state]
    // v_all: [T, B, n_state]
    // q_all: [T, B, n_state]
    // beta_x_all: [T, B, n_state]
    // k_norm_tmp: [B, n_state]
    // retrieved_tmp: [B, n_state]
    // beta_tmp: [B, n_state]
    __nv_bfloat16* k_all = workspace;
    __nv_bfloat16* v_all = k_all + steps * BN;
    __nv_bfloat16* q_all = v_all + steps * BN;
    __nv_bfloat16* beta_x_all = q_all + steps * BN;
    __nv_bfloat16* k_norm_tmp = beta_x_all + steps * BN;
    __nv_bfloat16* retrieved_tmp = k_norm_tmp + BN;
    __nv_bfloat16* beta_tmp = retrieved_tmp + BN;

    // Pre-compute all x projections in batched GEMMs
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        n_state_, steps * batch_size_, dim_,
        &alpha_one,
        W_k, dim_,
        x, dim_,
        &beta_zero,
        k_all, n_state_);

    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        n_state_, steps * batch_size_, dim_,
        &alpha_one,
        W_v, dim_,
        x, dim_,
        &beta_zero,
        v_all, n_state_);

    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        n_state_, steps * batch_size_, dim_,
        &alpha_one,
        W_q, dim_,
        x, dim_,
        &beta_zero,
        q_all, n_state_);

    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        n_state_, steps * batch_size_, dim_,
        &alpha_one,
        W_beta, dim_,
        x, dim_,
        &beta_zero,
        beta_x_all, n_state_);

    // Process each timestep sequentially (cannot parallelize - S-dependent)
    for (int t = 0; t < steps; ++t) {
        const __nv_bfloat16* k_t = k_all + t * BN;
        const __nv_bfloat16* v_t = v_all + t * BN;
        const __nv_bfloat16* q_t = q_all + t * BN;
        const __nv_bfloat16* beta_x_t = beta_x_all + t * BN;
        const __nv_bfloat16* S_prev = S + t * BNN;
        __nv_bfloat16* S_t = S + (t + 1) * BNN;
        __nv_bfloat16* out_t = output + t * BN;

        // Cache pointers for backward
        __nv_bfloat16* k_norm_c = training_ ? (k_norm_cache + t * BN) : k_norm_tmp;
        __nv_bfloat16* retrieved_c = training_ ? (retrieved_cache + t * BN) : retrieved_tmp;
        __nv_bfloat16* beta_c = training_ ? (beta_cache + t * BN) : beta_tmp;

        // Copy to cache if training
        if (training_) {
            cudaMemcpyAsync(k_cache + t * BN, k_t, BN * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream_);
            cudaMemcpyAsync(v_cache + t * BN, v_t, BN * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream_);
            cudaMemcpyAsync(q_cache + t * BN, q_t, BN * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream_);
            cudaMemcpyAsync(beta_x_cache + t * BN, beta_x_t, BN * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream_);
        }

        // 1. Key normalization
        KeyNormKernel_E71Delta_BF16<<<batch_size_, 256, 0, stream_>>>(
            batch_size_, n_state_, k_t, k_norm_c);

        // 2. Retrieve: retrieved = S @ k_norm
        MatVecKernel_E71Delta_BF16<<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_state_, S_prev, k_norm_c, retrieved_c);

        // 3. Compute beta = sigmoid(beta_x + d_beta * retrieved + b_beta)
        BetaKernel_E71Delta_BF16<<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_state_, beta_x_t, retrieved_c, d_beta, b_beta, beta_c);

        // 4. Delta rule update + output (fused)
        DeltaUpdateOutputKernel_E71Delta_BF16<<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_state_, S_prev, beta_c, v_t, retrieved_c, k_norm_c, q_t, S_t, out_t);
    }
}

// =============================================================================
// E71 Delta Backward - BF16 Specialization
// =============================================================================

template<>
E71DeltaBackward<__nv_bfloat16>::E71DeltaBackward(
    int batch_size,
    int n_state,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      n_state_(n_state),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void E71DeltaBackward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_k,
    const __nv_bfloat16* W_v,
    const __nv_bfloat16* W_q,
    const __nv_bfloat16* W_beta,
    const __nv_bfloat16* d_beta,
    const __nv_bfloat16* b_beta,
    const __nv_bfloat16* x,
    const __nv_bfloat16* S,
    const __nv_bfloat16* k_cache,
    const __nv_bfloat16* v_cache,
    const __nv_bfloat16* q_cache,
    const __nv_bfloat16* beta_x_cache,
    const __nv_bfloat16* retrieved_cache,
    const __nv_bfloat16* beta_cache,
    const __nv_bfloat16* k_norm_cache,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* dx,
    __nv_bfloat16* dW_k,
    __nv_bfloat16* dW_v,
    __nv_bfloat16* dW_q,
    __nv_bfloat16* dW_beta,
    __nv_bfloat16* dd_beta_out,
    __nv_bfloat16* db_beta,
    __nv_bfloat16* workspace) {

    static const __nv_bfloat16 alpha_one = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);
    static const __nv_bfloat16 beta_one = __float2bfloat16(1.0f);

    const int BN = batch_size_ * n_state_;
    const int BNN = batch_size_ * n_state_ * n_state_;
    const int block_size = 256;

    // Workspace layout
    __nv_bfloat16* d_S = workspace;
    __nv_bfloat16* d_S_tmp = d_S + BNN;
    __nv_bfloat16* d_k_all = d_S_tmp + BNN;
    __nv_bfloat16* d_v_all = d_k_all + steps * BN;
    __nv_bfloat16* d_q_all = d_v_all + steps * BN;
    __nv_bfloat16* d_beta_x_all = d_q_all + steps * BN;

    float* float_ws = reinterpret_cast<float*>(d_beta_x_all + steps * BN);
    float* d_beta_f = float_ws;
    float* d_k_f = d_beta_f + BN;
    float* d_v_f = d_k_f + BN;
    float* d_q_f = d_v_f + BN;
    float* d_retrieved_f = d_q_f + BN;
    float* d_k_norm_f = d_retrieved_f + BN;
    float* dd_beta_f = d_k_norm_f + BN;
    float* db_beta_f = dd_beta_f + n_state_;

    // Initialize gradients to zero
    cudaMemsetAsync(d_S, 0, BNN * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_k, 0, n_state_ * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_v, 0, n_state_ * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_q, 0, n_state_ * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_beta, 0, n_state_ * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dd_beta_f, 0, n_state_ * sizeof(float), stream_);
    cudaMemsetAsync(db_beta_f, 0, n_state_ * sizeof(float), stream_);

    // BPTT loop
    for (int t = steps - 1; t >= 0; --t) {
        const __nv_bfloat16* S_t = S + (t + 1) * BNN;
        const __nv_bfloat16* S_prev = S + t * BNN;
        const __nv_bfloat16* k_t = k_cache + t * BN;
        const __nv_bfloat16* v_t = v_cache + t * BN;
        const __nv_bfloat16* q_t = q_cache + t * BN;
        const __nv_bfloat16* beta_x_t = beta_x_cache + t * BN;
        const __nv_bfloat16* retrieved_t = retrieved_cache + t * BN;
        const __nv_bfloat16* beta_t = beta_cache + t * BN;
        const __nv_bfloat16* k_norm_t = k_norm_cache + t * BN;
        const __nv_bfloat16* d_out_t = d_output + t * BN;

        __nv_bfloat16* d_k_t = d_k_all + t * BN;
        __nv_bfloat16* d_v_t = d_v_all + t * BN;
        __nv_bfloat16* d_q_t = d_q_all + t * BN;
        __nv_bfloat16* d_beta_x_t = d_beta_x_all + t * BN;

        // Zero per-timestep accumulators
        cudaMemsetAsync(d_beta_f, 0, BN * sizeof(float), stream_);
        cudaMemsetAsync(d_k_f, 0, BN * sizeof(float), stream_);
        cudaMemsetAsync(d_v_f, 0, BN * sizeof(float), stream_);
        cudaMemsetAsync(d_q_f, 0, BN * sizeof(float), stream_);
        cudaMemsetAsync(d_retrieved_f, 0, BN * sizeof(float), stream_);
        cudaMemsetAsync(d_k_norm_f, 0, BN * sizeof(float), stream_);

        // 1. Backward through output
        OutputBackwardKernel_E71Delta_BF16<<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_state_, S_t, q_t, d_out_t, d_S, d_q_f);

        // 2. Backward through delta rule state update
        DeltaUpdateBackwardKernel_E71Delta_BF16<<<(BNN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_state_, beta_t, v_t, retrieved_t, k_norm_t, d_S,
            d_S_tmp, d_beta_f, d_v_f, d_retrieved_f, d_k_norm_f);

        // 3. Backward through beta computation
        BetaBackwardKernel_E71Delta_BF16<<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_state_, beta_x_t, retrieved_t, d_beta, b_beta,
            d_beta_f, d_beta_x_t, d_retrieved_f, dd_beta_f, db_beta_f);

        // 4. Backward through retrieval: retrieved = S_prev @ k_norm
        RetrievalBackwardKernel_E71Delta_BF16<<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_state_, S_prev, k_norm_t, d_retrieved_f, d_S_tmp, d_k_norm_f);

        // 5. Backward through key normalization
        KeyNormBackwardKernel_E71Delta_BF16<<<batch_size_, 256, 0, stream_>>>(
            batch_size_, n_state_, k_t, d_k_norm_f, d_k_f);

        // Copy float accumulators to output
        CopyFloatToBF16_E71Delta<<<(BN + 255) / 256, 256, 0, stream_>>>(BN, d_k_f, d_k_t);
        CopyFloatToBF16_E71Delta<<<(BN + 255) / 256, 256, 0, stream_>>>(BN, d_v_f, d_v_t);
        CopyFloatToBF16_E71Delta<<<(BN + 255) / 256, 256, 0, stream_>>>(BN, d_q_f, d_q_t);

        // Swap d_S and d_S_tmp for next iteration
        std::swap(d_S, d_S_tmp);
    }

    // Batch compute weight gradients
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, n_state_,
        &alpha_one,
        W_k, dim_,
        d_k_all, n_state_,
        &beta_zero,
        dx, dim_);

    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, n_state_,
        &alpha_one,
        W_v, dim_,
        d_v_all, n_state_,
        &beta_one,
        dx, dim_);

    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, n_state_,
        &alpha_one,
        W_q, dim_,
        d_q_all, n_state_,
        &beta_one,
        dx, dim_);

    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, n_state_,
        &alpha_one,
        W_beta, dim_,
        d_beta_x_all, n_state_,
        &beta_one,
        dx, dim_);

    // Weight gradients
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, n_state_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        d_k_all, n_state_,
        &beta_one,
        dW_k, dim_);

    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, n_state_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        d_v_all, n_state_,
        &beta_one,
        dW_v, dim_);

    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, n_state_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        d_q_all, n_state_,
        &beta_one,
        dW_q, dim_);

    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, n_state_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        d_beta_x_all, n_state_,
        &beta_one,
        dW_beta, dim_);

    // Copy parameter gradients
    CopyFloatToBF16_E71Delta<<<(n_state_ + 255) / 256, 256, 0, stream_>>>(n_state_, dd_beta_f, dd_beta_out);
    CopyFloatToBF16_E71Delta<<<(n_state_ + 255) / 256, 256, 0, stream_>>>(n_state_, db_beta_f, db_beta);
}

// Explicit template instantiations
template struct E71DeltaForward<__nv_bfloat16>;
template struct E71DeltaBackward<__nv_bfloat16>;

// Stub implementations for float, double, and half (only bfloat16 is optimized)
// Forward::Run has 18 params, Backward::Run has 24 params
template<>
E71DeltaForward<float>::E71DeltaForward(bool, int, int, int, const cublasHandle_t&, const cudaStream_t&)
    : training_(false), batch_size_(0), n_state_(0), dim_(0), blas_handle_(nullptr), stream_(nullptr) {}
template<> void E71DeltaForward<float>::Run(int, const float*, const float*, const float*, const float*, const float*, const float*, const float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*) {}
template struct E71DeltaForward<float>;

template<>
E71DeltaForward<double>::E71DeltaForward(bool, int, int, int, const cublasHandle_t&, const cudaStream_t&)
    : training_(false), batch_size_(0), n_state_(0), dim_(0), blas_handle_(nullptr), stream_(nullptr) {}
template<> void E71DeltaForward<double>::Run(int, const double*, const double*, const double*, const double*, const double*, const double*, const double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*) {}
template struct E71DeltaForward<double>;

template<>
E71DeltaForward<__half>::E71DeltaForward(bool, int, int, int, const cublasHandle_t&, const cudaStream_t&)
    : training_(false), batch_size_(0), n_state_(0), dim_(0), blas_handle_(nullptr), stream_(nullptr) {}
template<> void E71DeltaForward<__half>::Run(int, const __half*, const __half*, const __half*, const __half*, const __half*, const __half*, const __half*, __half*, __half*, __half*, __half*, __half*, __half*, __half*, __half*, __half*, __half*) {}
template struct E71DeltaForward<__half>;

template<>
E71DeltaBackward<float>::E71DeltaBackward(int, int, int, const cublasHandle_t&, const cudaStream_t&)
    : batch_size_(0), n_state_(0), dim_(0), blas_handle_(nullptr), stream_(nullptr) {}
template<> void E71DeltaBackward<float>::Run(int, const float*, const float*, const float*, const float*, const float*, const float*, const float*, const float*, const float*, const float*, const float*, const float*, const float*, const float*, const float*, const float*, float*, float*, float*, float*, float*, float*, float*, float*) {}
template struct E71DeltaBackward<float>;

template<>
E71DeltaBackward<double>::E71DeltaBackward(int, int, int, const cublasHandle_t&, const cudaStream_t&)
    : batch_size_(0), n_state_(0), dim_(0), blas_handle_(nullptr), stream_(nullptr) {}
template<> void E71DeltaBackward<double>::Run(int, const double*, const double*, const double*, const double*, const double*, const double*, const double*, const double*, const double*, const double*, const double*, const double*, const double*, const double*, const double*, const double*, double*, double*, double*, double*, double*, double*, double*, double*) {}
template struct E71DeltaBackward<double>;

template<>
E71DeltaBackward<__half>::E71DeltaBackward(int, int, int, const cublasHandle_t&, const cudaStream_t&)
    : batch_size_(0), n_state_(0), dim_(0), blas_handle_(nullptr), stream_(nullptr) {}
template<> void E71DeltaBackward<__half>::Run(int, const __half*, const __half*, const __half*, const __half*, const __half*, const __half*, const __half*, const __half*, const __half*, const __half*, const __half*, const __half*, const __half*, const __half*, const __half*, const __half*, __half*, __half*, __half*, __half*, __half*, __half*, __half*, __half*) {}
template struct E71DeltaBackward<__half>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
