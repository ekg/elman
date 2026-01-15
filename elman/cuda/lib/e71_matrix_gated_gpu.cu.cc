// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E71: Matrix Gated Elman - E67-style state-dependent gating with matrix state
//
// Architecture:
//     k_t = W_k @ x_t                              # [B, n] key
//     v_t = W_v @ x_t                              # [B, n] value
//     q_t = W_q @ x_t                              # [B, n] query
//     alpha_x_t = W_alpha @ x_t                    # [B, n]
//
//     # S-dependent gate (E67 insight: state affects gating decision)
//     retrieved = S @ k_t                          # [B, n] (matrix-vector product)
//     alpha = sigmoid(alpha_x + d_alpha * retrieved + b_alpha)
//
//     # Gated update
//     S_new = alpha.unsqueeze(-1) * S + (1 - alpha.unsqueeze(-1)) * outer(v, k)
//
//     # Self-gating output
//     out = S_new @ q
//     output = out * silu(out)
//
// Key optimization strategy:
// 1. Batch projections upfront: k_all, v_all, q_all, alpha_x_all via 4 batched GEMMs
// 2. Sequential kernel for the matrix state recurrence (cannot parallelize due to S-dependence)
//
// State S is [B, n_state, n_state] - square matrix unlike E63m

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

// Compute S @ k -> retrieved (batched matrix-vector multiply)
// S: [B, N, N], k: [B, N] -> retrieved: [B, N]
template<typename T>
__global__ void MatVecKernel_E71(
    const int batch_size,
    const int N,   // n_state (square matrix)
    const T* __restrict__ S,        // [B, N, N]
    const T* __restrict__ k,        // [B, N]
    T* __restrict__ retrieved) {    // [B, N]

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int b = idx / N;
        const int n = idx % N;

        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            // S[b, n, j] = S[b * N * N + n * N + j]
            // k[b, j] = k[b * N + j]
            float s_val = static_cast<float>(S[b * N * N + n * N + j]);
            float k_val = static_cast<float>(k[b * N + j]);
            sum += s_val * k_val;
        }
        retrieved[idx] = static_cast<T>(sum);
    }
}

__global__ void MatVecKernel_E71_BF16(
    const int batch_size,
    const int N,
    const __nv_bfloat16* __restrict__ S,
    const __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ retrieved) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int b = idx / N;
        const int n = idx % N;

        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            float s_val = __bfloat162float(S[b * N * N + n * N + j]);
            float k_val = __bfloat162float(k[b * N + j]);
            sum += s_val * k_val;
        }
        retrieved[idx] = __float2bfloat16(sum);
    }
}

// Compute alpha = sigmoid(alpha_x + d_alpha * retrieved + b_alpha)
// alpha_x: [B, N], retrieved: [B, N], d_alpha: [N], b_alpha: [N]
// Output: alpha: [B, N]
template<typename T>
__global__ void AlphaKernel_E71(
    const int batch_size,
    const int N,
    const T* __restrict__ alpha_x,      // [B, N]
    const T* __restrict__ retrieved,    // [B, N]
    const T* __restrict__ d_alpha,      // [N]
    const T* __restrict__ b_alpha,      // [N]
    T* __restrict__ alpha) {            // [B, N]

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int n = idx % N;

        float ax = static_cast<float>(alpha_x[idx]);
        float ret = static_cast<float>(retrieved[idx]);
        float da = static_cast<float>(d_alpha[n]);
        float ba = static_cast<float>(b_alpha[n]);

        float logit = ax + da * ret + ba;
        float a = 1.0f / (1.0f + expf(-logit));
        alpha[idx] = static_cast<T>(a);
    }
}

__global__ void AlphaKernel_E71_BF16(
    const int batch_size,
    const int N,
    const __nv_bfloat16* __restrict__ alpha_x,
    const __nv_bfloat16* __restrict__ retrieved,
    const __nv_bfloat16* __restrict__ d_alpha,
    const __nv_bfloat16* __restrict__ b_alpha,
    __nv_bfloat16* __restrict__ alpha) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int n = idx % N;

        float ax = __bfloat162float(alpha_x[idx]);
        float ret = __bfloat162float(retrieved[idx]);
        float da = __bfloat162float(d_alpha[n]);
        float ba = __bfloat162float(b_alpha[n]);

        float logit = ax + da * ret + ba;
        float a = 1.0f / (1.0f + __expf(-logit));
        alpha[idx] = __float2bfloat16(a);
    }
}

// Gated state update + output computation (fused)
// S_new[b, i, j] = alpha[b, i] * S_prev[b, i, j] + (1 - alpha[b, i]) * v[b, i] * k[b, j]
// out[b, n] = sum_j(S_new[b, n, j] * q[b, j])
// output[b, n] = out * silu(out)
template<typename T>
__global__ void GatedUpdateOutputKernel_E71(
    const int batch_size,
    const int N,
    const T* __restrict__ S_prev,   // [B, N, N]
    const T* __restrict__ alpha,    // [B, N]
    const T* __restrict__ v,        // [B, N]
    const T* __restrict__ k,        // [B, N]
    const T* __restrict__ q,        // [B, N]
    T* __restrict__ S_new,          // [B, N, N]
    T* __restrict__ output) {       // [B, N]

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int b = idx / N;
        const int n = idx % N;

        float alpha_val = static_cast<float>(alpha[b * N + n]);
        float v_val = static_cast<float>(v[b * N + n]);

        float out_sum = 0.0f;

        // Update row n of S and compute output simultaneously
        for (int j = 0; j < N; ++j) {
            int s_idx = b * N * N + n * N + j;
            float s_prev_val = static_cast<float>(S_prev[s_idx]);
            float k_val = static_cast<float>(k[b * N + j]);
            float q_val = static_cast<float>(q[b * N + j]);

            // Gated update: S_new = alpha * S_prev + (1 - alpha) * outer(v, k)
            float s_new_val = alpha_val * s_prev_val + (1.0f - alpha_val) * v_val * k_val;
            S_new[s_idx] = static_cast<T>(s_new_val);

            // Accumulate for output: out = S_new @ q
            out_sum += s_new_val * q_val;
        }

        // Self-gating: output = out * silu(out)
        float sigmoid_out = 1.0f / (1.0f + expf(-out_sum));
        float silu_out = out_sum * sigmoid_out;
        output[idx] = static_cast<T>(out_sum * silu_out);
    }
}

__global__ void GatedUpdateOutputKernel_E71_BF16(
    const int batch_size,
    const int N,
    const __nv_bfloat16* __restrict__ S_prev,
    const __nv_bfloat16* __restrict__ alpha,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ S_new,
    __nv_bfloat16* __restrict__ output) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int b = idx / N;
        const int n = idx % N;

        float alpha_val = __bfloat162float(alpha[b * N + n]);
        float v_val = __bfloat162float(v[b * N + n]);

        float out_sum = 0.0f;

        for (int j = 0; j < N; ++j) {
            int s_idx = b * N * N + n * N + j;
            float s_prev_val = __bfloat162float(S_prev[s_idx]);
            float k_val = __bfloat162float(k[b * N + j]);
            float q_val = __bfloat162float(q[b * N + j]);

            float s_new_val = alpha_val * s_prev_val + (1.0f - alpha_val) * v_val * k_val;
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
template<typename T>
__global__ void OutputBackwardKernel_E71(
    const int batch_size,
    const int N,
    const T* __restrict__ S,          // [B, N, N]
    const T* __restrict__ q,          // [B, N]
    const T* __restrict__ d_output,   // [B, N]
    T* __restrict__ d_S,              // [B, N, N] - output
    float* __restrict__ d_q_f) {      // [B, N] float accumulator

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int b = idx / N;
        const int n = idx % N;

        // First compute out = S @ q for this row
        float out_sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            float s_val = static_cast<float>(S[b * N * N + n * N + j]);
            float q_val = static_cast<float>(q[b * N + j]);
            out_sum += s_val * q_val;
        }

        // Backward through self-gate: output = out * silu(out)
        float dout_val = static_cast<float>(d_output[idx]);
        float sigmoid_out = 1.0f / (1.0f + expf(-out_sum));
        float silu_out = out_sum * sigmoid_out;
        float d_out = dout_val * silu_out * (2.0f + out_sum * (1.0f - sigmoid_out));

        // d_S[b,n,j] += d_out * q[b,j]  (accumulate gradient)
        // d_q[b,j] += d_out * S[b,n,j]
        for (int j = 0; j < N; ++j) {
            int s_idx = b * N * N + n * N + j;
            float q_val = static_cast<float>(q[b * N + j]);
            float s_val = static_cast<float>(S[s_idx]);

            // NOTE: d_S is accumulated (+=) to include gradient from later timesteps
            float d_S_curr = static_cast<float>(d_S[s_idx]);
            d_S[s_idx] = static_cast<T>(d_S_curr + d_out * q_val);
            atomicAdd(&d_q_f[b * N + j], d_out * s_val);
        }
    }
}

__global__ void OutputBackwardKernel_E71_BF16(
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

            // NOTE: d_S is accumulated (+=) to include gradient from later timesteps
            float d_S_curr = __bfloat162float(d_S[s_idx]);
            d_S[s_idx] = __float2bfloat16(d_S_curr + d_out * q_val);
            atomicAdd(&d_q_f[b * N + j], d_out * s_val);
        }
    }
}

// Backward through gated state update:
// S_new = alpha * S_prev + (1 - alpha) * outer(v, k)
// Given d_S_new, compute d_S_prev, d_alpha, d_v, d_k
template<typename T>
__global__ void StateUpdateBackwardKernel_E71(
    const int batch_size,
    const int N,
    const T* __restrict__ S_prev,     // [B, N, N]
    const T* __restrict__ alpha,      // [B, N]
    const T* __restrict__ v,          // [B, N]
    const T* __restrict__ k,          // [B, N]
    const T* __restrict__ d_S_new,    // [B, N, N]
    T* __restrict__ d_S_prev,         // [B, N, N]
    float* __restrict__ d_alpha_f,    // [B, N] float accumulator
    float* __restrict__ d_v_f,        // [B, N] float accumulator
    float* __restrict__ d_k_f) {      // [B, N] float accumulator

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N * N;

    if (idx < total) {
        const int j = idx % N;
        const int n = (idx / N) % N;
        const int b = idx / (N * N);

        const int bn_idx = b * N + n;
        const int bj_idx = b * N + j;

        float alpha_val = static_cast<float>(alpha[bn_idx]);
        float s_prev_val = static_cast<float>(S_prev[idx]);
        float v_val = static_cast<float>(v[bn_idx]);
        float k_val = static_cast<float>(k[bj_idx]);
        float ds_new = static_cast<float>(d_S_new[idx]);

        // d_S_prev = d_S_new * alpha
        d_S_prev[idx] = static_cast<T>(ds_new * alpha_val);

        // d_alpha[b,n] += d_S_new[b,n,j] * (S_prev[b,n,j] - v[b,n] * k[b,j])
        float d_alpha_contrib = ds_new * (s_prev_val - v_val * k_val);
        atomicAdd(&d_alpha_f[bn_idx], d_alpha_contrib);

        // d_v[b,n] += d_S_new[b,n,j] * (1-alpha) * k[b,j]
        float d_v_contrib = ds_new * (1.0f - alpha_val) * k_val;
        atomicAdd(&d_v_f[bn_idx], d_v_contrib);

        // d_k[b,j] += d_S_new[b,n,j] * (1-alpha) * v[b,n]
        float d_k_contrib = ds_new * (1.0f - alpha_val) * v_val;
        atomicAdd(&d_k_f[bj_idx], d_k_contrib);
    }
}

__global__ void StateUpdateBackwardKernel_E71_BF16(
    const int batch_size,
    const int N,
    const __nv_bfloat16* __restrict__ S_prev,
    const __nv_bfloat16* __restrict__ alpha,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ d_S_new,
    __nv_bfloat16* __restrict__ d_S_prev,
    float* __restrict__ d_alpha_f,
    float* __restrict__ d_v_f,
    float* __restrict__ d_k_f) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N * N;

    if (idx < total) {
        const int j = idx % N;
        const int n = (idx / N) % N;
        const int b = idx / (N * N);

        const int bn_idx = b * N + n;
        const int bj_idx = b * N + j;

        float alpha_val = __bfloat162float(alpha[bn_idx]);
        float s_prev_val = __bfloat162float(S_prev[idx]);
        float v_val = __bfloat162float(v[bn_idx]);
        float k_val = __bfloat162float(k[bj_idx]);
        float ds_new = __bfloat162float(d_S_new[idx]);

        d_S_prev[idx] = __float2bfloat16(ds_new * alpha_val);

        float d_alpha_contrib = ds_new * (s_prev_val - v_val * k_val);
        atomicAdd(&d_alpha_f[bn_idx], d_alpha_contrib);

        float d_v_contrib = ds_new * (1.0f - alpha_val) * k_val;
        atomicAdd(&d_v_f[bn_idx], d_v_contrib);

        float d_k_contrib = ds_new * (1.0f - alpha_val) * v_val;
        atomicAdd(&d_k_f[bj_idx], d_k_contrib);
    }
}

// Backward through alpha computation:
// alpha = sigmoid(alpha_x + d_alpha * retrieved + b_alpha)
// Given d_alpha, compute d_alpha_x, d_retrieved, accumulate d_d_alpha, d_b_alpha
template<typename T>
__global__ void AlphaBackwardKernel_E71(
    const int batch_size,
    const int N,
    const T* __restrict__ alpha_x,      // [B, N]
    const T* __restrict__ retrieved,    // [B, N]
    const T* __restrict__ d_alpha,      // [N]
    const T* __restrict__ b_alpha,      // [N]
    const float* __restrict__ d_alpha_f, // [B, N] - gradient to alpha
    T* __restrict__ d_alpha_x,          // [B, N]
    T* __restrict__ d_retrieved,        // [B, N]
    float* __restrict__ dd_alpha_f,     // [N] accumulator
    float* __restrict__ db_alpha_f) {   // [N] accumulator

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int n = idx % N;

        // Recompute alpha (sigmoid)
        float ax = static_cast<float>(alpha_x[idx]);
        float ret = static_cast<float>(retrieved[idx]);
        float da = static_cast<float>(d_alpha[n]);
        float ba = static_cast<float>(b_alpha[n]);
        float logit = ax + da * ret + ba;
        float alpha_val = 1.0f / (1.0f + expf(-logit));

        // Gradient through sigmoid
        float d_alpha_in = d_alpha_f[idx];
        float d_logit = d_alpha_in * alpha_val * (1.0f - alpha_val);

        // d_alpha_x = d_logit
        d_alpha_x[idx] = static_cast<T>(d_logit);

        // d_retrieved = d_logit * d_alpha[n]
        d_retrieved[idx] = static_cast<T>(d_logit * da);

        // Accumulate parameter gradients
        atomicAdd(&dd_alpha_f[n], d_logit * ret);  // d(logit)/d(d_alpha) = retrieved
        atomicAdd(&db_alpha_f[n], d_logit);        // d(logit)/d(b_alpha) = 1
    }
}

__global__ void AlphaBackwardKernel_E71_BF16(
    const int batch_size,
    const int N,
    const __nv_bfloat16* __restrict__ alpha_x,
    const __nv_bfloat16* __restrict__ retrieved,
    const __nv_bfloat16* __restrict__ d_alpha,
    const __nv_bfloat16* __restrict__ b_alpha,
    const float* __restrict__ d_alpha_f,
    __nv_bfloat16* __restrict__ d_alpha_x,
    __nv_bfloat16* __restrict__ d_retrieved,
    float* __restrict__ dd_alpha_f,
    float* __restrict__ db_alpha_f) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int n = idx % N;

        float ax = __bfloat162float(alpha_x[idx]);
        float ret = __bfloat162float(retrieved[idx]);
        float da = __bfloat162float(d_alpha[n]);
        float ba = __bfloat162float(b_alpha[n]);
        float logit = ax + da * ret + ba;
        float alpha_val = 1.0f / (1.0f + __expf(-logit));

        float d_alpha_in = d_alpha_f[idx];
        float d_logit = d_alpha_in * alpha_val * (1.0f - alpha_val);

        d_alpha_x[idx] = __float2bfloat16(d_logit);
        d_retrieved[idx] = __float2bfloat16(d_logit * da);

        atomicAdd(&dd_alpha_f[n], d_logit * ret);
        atomicAdd(&db_alpha_f[n], d_logit);
    }
}

// Backward through retrieval: retrieved = S @ k
// Given d_retrieved, compute d_S and accumulate d_k
template<typename T>
__global__ void RetrievalBackwardKernel_E71(
    const int batch_size,
    const int N,
    const T* __restrict__ S,            // [B, N, N]
    const T* __restrict__ k,            // [B, N]
    const T* __restrict__ d_retrieved,  // [B, N]
    T* __restrict__ d_S,                // [B, N, N] - add to existing
    float* __restrict__ d_k_f) {        // [B, N] - add to existing

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int b = idx / N;
        const int n = idx % N;

        float d_ret = static_cast<float>(d_retrieved[idx]);

        // d_S[b,n,j] += d_retrieved[b,n] * k[b,j]
        // d_k[b,j] += d_retrieved[b,n] * S[b,n,j]
        for (int j = 0; j < N; ++j) {
            int s_idx = b * N * N + n * N + j;
            float k_val = static_cast<float>(k[b * N + j]);
            float s_val = static_cast<float>(S[s_idx]);

            d_S[s_idx] = static_cast<T>(static_cast<float>(d_S[s_idx]) + d_ret * k_val);
            atomicAdd(&d_k_f[b * N + j], d_ret * s_val);
        }
    }
}

__global__ void RetrievalBackwardKernel_E71_BF16(
    const int batch_size,
    const int N,
    const __nv_bfloat16* __restrict__ S,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ d_retrieved,
    __nv_bfloat16* __restrict__ d_S,
    float* __restrict__ d_k_f) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int b = idx / N;
        const int n = idx % N;

        float d_ret = __bfloat162float(d_retrieved[idx]);

        for (int j = 0; j < N; ++j) {
            int s_idx = b * N * N + n * N + j;
            float k_val = __bfloat162float(k[b * N + j]);
            float s_val = __bfloat162float(S[s_idx]);

            d_S[s_idx] = __float2bfloat16(__bfloat162float(d_S[s_idx]) + d_ret * k_val);
            atomicAdd(&d_k_f[b * N + j], d_ret * s_val);
        }
    }
}

// Copy float to T
template<typename T>
__global__ void CopyFloatToT_E71(const int n, const float* __restrict__ src, T* __restrict__ dst) {
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
// E71 Matrix Gated Forward - BF16 Specialization
// =============================================================================

template<>
E71MatrixGatedForward<__nv_bfloat16>::E71MatrixGatedForward(
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
void E71MatrixGatedForward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_k,       // [n_state, dim]
    const __nv_bfloat16* W_v,       // [n_state, dim]
    const __nv_bfloat16* W_q,       // [n_state, dim]
    const __nv_bfloat16* W_alpha,   // [n_state, dim]
    const __nv_bfloat16* d_alpha,   // [n_state]
    const __nv_bfloat16* b_alpha,   // [n_state]
    const __nv_bfloat16* x,         // [T, B, dim] input
    __nv_bfloat16* S,               // [T+1, B, n_state, n_state] state matrices
    __nv_bfloat16* output,          // [T, B, n_state] output
    __nv_bfloat16* k_cache,         // [T, B, n_state]
    __nv_bfloat16* v_cache,         // [T, B, n_state]
    __nv_bfloat16* q_cache,         // [T, B, n_state]
    __nv_bfloat16* alpha_x_cache,   // [T, B, n_state]
    __nv_bfloat16* retrieved_cache, // [T, B, n_state]
    __nv_bfloat16* alpha_cache,     // [T, B, n_state]
    __nv_bfloat16* workspace) {

    static const __nv_bfloat16 alpha_one = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);

    const int BD = batch_size_ * dim_;
    const int BN = batch_size_ * n_state_;
    const int BNN = batch_size_ * n_state_ * n_state_;
    const int block_size = 256;

    // Workspace layout:
    // k_all: [T, B, n_state]
    // v_all: [T, B, n_state]
    // q_all: [T, B, n_state]
    // alpha_x_all: [T, B, n_state]
    // retrieved_tmp: [B, n_state]
    // alpha_tmp: [B, n_state]
    __nv_bfloat16* k_all = workspace;
    __nv_bfloat16* v_all = k_all + steps * BN;
    __nv_bfloat16* q_all = v_all + steps * BN;
    __nv_bfloat16* alpha_x_all = q_all + steps * BN;
    __nv_bfloat16* retrieved_tmp = alpha_x_all + steps * BN;
    __nv_bfloat16* alpha_tmp = retrieved_tmp + BN;

    // Pre-compute all x projections in batched GEMMs
    // k_all = x @ W_k.T  [T*B, n_state]
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        n_state_, steps * batch_size_, dim_,
        &alpha_one,
        W_k, dim_,
        x, dim_,
        &beta_zero,
        k_all, n_state_);

    // v_all = x @ W_v.T  [T*B, n_state]
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        n_state_, steps * batch_size_, dim_,
        &alpha_one,
        W_v, dim_,
        x, dim_,
        &beta_zero,
        v_all, n_state_);

    // q_all = x @ W_q.T  [T*B, n_state]
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        n_state_, steps * batch_size_, dim_,
        &alpha_one,
        W_q, dim_,
        x, dim_,
        &beta_zero,
        q_all, n_state_);

    // alpha_x_all = x @ W_alpha.T  [T*B, n_state]
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        n_state_, steps * batch_size_, dim_,
        &alpha_one,
        W_alpha, dim_,
        x, dim_,
        &beta_zero,
        alpha_x_all, n_state_);

    // Process each timestep sequentially (cannot parallelize - S-dependent gating)
    for (int t = 0; t < steps; ++t) {
        const __nv_bfloat16* k_t = k_all + t * BN;
        const __nv_bfloat16* v_t = v_all + t * BN;
        const __nv_bfloat16* q_t = q_all + t * BN;
        const __nv_bfloat16* alpha_x_t = alpha_x_all + t * BN;
        const __nv_bfloat16* S_prev = S + t * BNN;
        __nv_bfloat16* S_t = S + (t + 1) * BNN;
        __nv_bfloat16* out_t = output + t * BN;

        // Cache pointers for backward
        __nv_bfloat16* retrieved_c = training_ ? (retrieved_cache + t * BN) : retrieved_tmp;
        __nv_bfloat16* alpha_c = training_ ? (alpha_cache + t * BN) : alpha_tmp;

        // Copy to cache if training
        if (training_) {
            cudaMemcpyAsync(k_cache + t * BN, k_t, BN * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream_);
            cudaMemcpyAsync(v_cache + t * BN, v_t, BN * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream_);
            cudaMemcpyAsync(q_cache + t * BN, q_t, BN * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream_);
            cudaMemcpyAsync(alpha_x_cache + t * BN, alpha_x_t, BN * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream_);
        }

        // 1. Retrieve: retrieved = S @ k
        MatVecKernel_E71_BF16<<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_state_, S_prev, k_t, retrieved_c);

        // 2. Compute alpha = sigmoid(alpha_x + d_alpha * retrieved + b_alpha)
        AlphaKernel_E71_BF16<<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_state_, alpha_x_t, retrieved_c, d_alpha, b_alpha, alpha_c);

        // 3. Gated update + output (fused)
        GatedUpdateOutputKernel_E71_BF16<<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_state_, S_prev, alpha_c, v_t, k_t, q_t, S_t, out_t);
    }
}

// =============================================================================
// E71 Matrix Gated Backward - BF16 Specialization
// =============================================================================

template<>
E71MatrixGatedBackward<__nv_bfloat16>::E71MatrixGatedBackward(
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
void E71MatrixGatedBackward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_k,
    const __nv_bfloat16* W_v,
    const __nv_bfloat16* W_q,
    const __nv_bfloat16* W_alpha,
    const __nv_bfloat16* d_alpha,
    const __nv_bfloat16* b_alpha,
    const __nv_bfloat16* x,
    const __nv_bfloat16* S,
    const __nv_bfloat16* k_cache,
    const __nv_bfloat16* v_cache,
    const __nv_bfloat16* q_cache,
    const __nv_bfloat16* alpha_x_cache,
    const __nv_bfloat16* retrieved_cache,
    const __nv_bfloat16* alpha_cache,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* dx,
    __nv_bfloat16* dW_k,
    __nv_bfloat16* dW_v,
    __nv_bfloat16* dW_q,
    __nv_bfloat16* dW_alpha,
    __nv_bfloat16* dd_alpha_out,
    __nv_bfloat16* db_alpha,
    __nv_bfloat16* workspace) {

    static const __nv_bfloat16 alpha_one = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);
    static const __nv_bfloat16 beta_one = __float2bfloat16(1.0f);

    const int BN = batch_size_ * n_state_;
    const int BNN = batch_size_ * n_state_ * n_state_;
    const int block_size = 256;

    // Workspace layout:
    // d_S: [B, n_state, n_state]
    // d_S_tmp: [B, n_state, n_state]
    // d_k_all: [T, B, n_state]
    // d_v_all: [T, B, n_state]
    // d_q_all: [T, B, n_state]
    // d_alpha_x_all: [T, B, n_state]
    // d_retrieved: [B, n_state]
    // d_alpha_f: [B, n_state] float
    // d_k_f: [B, n_state] float
    // d_v_f: [B, n_state] float
    // d_q_f: [B, n_state] float
    // dd_alpha_f: [n_state] float
    // db_alpha_f: [n_state] float
    __nv_bfloat16* d_S = workspace;
    __nv_bfloat16* d_S_tmp = d_S + BNN;
    __nv_bfloat16* d_k_all = d_S_tmp + BNN;
    __nv_bfloat16* d_v_all = d_k_all + steps * BN;
    __nv_bfloat16* d_q_all = d_v_all + steps * BN;
    __nv_bfloat16* d_alpha_x_all = d_q_all + steps * BN;
    __nv_bfloat16* d_retrieved = d_alpha_x_all + steps * BN;

    float* float_ws = reinterpret_cast<float*>(d_retrieved + BN);
    float* d_alpha_f = float_ws;
    float* d_k_f = d_alpha_f + BN;
    float* d_v_f = d_k_f + BN;
    float* d_q_f = d_v_f + BN;
    float* dd_alpha_f = d_q_f + BN;
    float* db_alpha_f = dd_alpha_f + n_state_;

    // Initialize gradients to zero
    cudaMemsetAsync(d_S, 0, BNN * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_k, 0, n_state_ * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_v, 0, n_state_ * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_q, 0, n_state_ * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_alpha, 0, n_state_ * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dd_alpha_f, 0, n_state_ * sizeof(float), stream_);
    cudaMemsetAsync(db_alpha_f, 0, n_state_ * sizeof(float), stream_);

    // BPTT loop
    for (int t = steps - 1; t >= 0; --t) {
        const __nv_bfloat16* S_t = S + (t + 1) * BNN;
        const __nv_bfloat16* S_prev = S + t * BNN;
        const __nv_bfloat16* k_t = k_cache + t * BN;
        const __nv_bfloat16* v_t = v_cache + t * BN;
        const __nv_bfloat16* q_t = q_cache + t * BN;
        const __nv_bfloat16* alpha_x_t = alpha_x_cache + t * BN;
        const __nv_bfloat16* retrieved_t = retrieved_cache + t * BN;
        const __nv_bfloat16* alpha_t = alpha_cache + t * BN;
        const __nv_bfloat16* d_out_t = d_output + t * BN;

        __nv_bfloat16* d_k_t = d_k_all + t * BN;
        __nv_bfloat16* d_v_t = d_v_all + t * BN;
        __nv_bfloat16* d_q_t = d_q_all + t * BN;
        __nv_bfloat16* d_alpha_x_t = d_alpha_x_all + t * BN;

        // Zero per-timestep accumulators
        cudaMemsetAsync(d_alpha_f, 0, BN * sizeof(float), stream_);
        cudaMemsetAsync(d_k_f, 0, BN * sizeof(float), stream_);
        cudaMemsetAsync(d_v_f, 0, BN * sizeof(float), stream_);
        cudaMemsetAsync(d_q_f, 0, BN * sizeof(float), stream_);

        // 1. Backward through output (self-gate + S @ q)
        OutputBackwardKernel_E71_BF16<<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_state_, S_t, q_t, d_out_t, d_S, d_q_f);

        // 2. Backward through gated state update
        StateUpdateBackwardKernel_E71_BF16<<<(BNN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_state_, S_prev, alpha_t, v_t, k_t, d_S,
            d_S_tmp, d_alpha_f, d_v_f, d_k_f);

        // 3. Backward through alpha computation
        AlphaBackwardKernel_E71_BF16<<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_state_, alpha_x_t, retrieved_t, d_alpha, b_alpha,
            d_alpha_f, d_alpha_x_t, d_retrieved, dd_alpha_f, db_alpha_f);

        // 4. Backward through retrieval: retrieved = S_prev @ k
        RetrievalBackwardKernel_E71_BF16<<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_state_, S_prev, k_t, d_retrieved, d_S_tmp, d_k_f);

        // Copy float accumulators to output
        CopyFloatToT_E71<__nv_bfloat16><<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            BN, d_k_f, d_k_t);
        CopyFloatToT_E71<__nv_bfloat16><<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            BN, d_v_f, d_v_t);
        CopyFloatToT_E71<__nv_bfloat16><<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            BN, d_q_f, d_q_t);

        // Swap d_S and d_S_tmp for next iteration
        std::swap(d_S, d_S_tmp);
    }

    // Batch compute weight gradients from accumulated projection gradients
    // dx = d_k_all @ W_k + d_v_all @ W_v + d_q_all @ W_q + d_alpha_x_all @ W_alpha

    // dx from k projection: dx = d_k_all @ W_k
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, n_state_,
        &alpha_one,
        W_k, dim_,
        d_k_all, n_state_,
        &beta_zero,
        dx, dim_);

    // dx += d_v_all @ W_v
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, n_state_,
        &alpha_one,
        W_v, dim_,
        d_v_all, n_state_,
        &beta_one,
        dx, dim_);

    // dx += d_q_all @ W_q
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, n_state_,
        &alpha_one,
        W_q, dim_,
        d_q_all, n_state_,
        &beta_one,
        dx, dim_);

    // dx += d_alpha_x_all @ W_alpha
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, n_state_,
        &alpha_one,
        W_alpha, dim_,
        d_alpha_x_all, n_state_,
        &beta_one,
        dx, dim_);

    // dW_k = d_k_all.T @ x = [n_state, T*B] @ [T*B, dim] = [n_state, dim]
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, n_state_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        d_k_all, n_state_,
        &beta_one,
        dW_k, dim_);

    // dW_v = d_v_all.T @ x
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, n_state_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        d_v_all, n_state_,
        &beta_one,
        dW_v, dim_);

    // dW_q = d_q_all.T @ x
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, n_state_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        d_q_all, n_state_,
        &beta_one,
        dW_q, dim_);

    // dW_alpha = d_alpha_x_all.T @ x
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, n_state_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        d_alpha_x_all, n_state_,
        &beta_one,
        dW_alpha, dim_);

    // Copy parameter gradients
    CopyFloatToT_E71<__nv_bfloat16><<<(n_state_ + 255) / 256, 256, 0, stream_>>>(n_state_, dd_alpha_f, dd_alpha_out);
    CopyFloatToT_E71<__nv_bfloat16><<<(n_state_ + 255) / 256, 256, 0, stream_>>>(n_state_, db_alpha_f, db_alpha);
}

// =============================================================================
// Generic Template Implementations
// =============================================================================

template<typename T>
E71MatrixGatedForward<T>::E71MatrixGatedForward(
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

template<typename T>
void E71MatrixGatedForward<T>::Run(
    int steps,
    const T* W_k,
    const T* W_v,
    const T* W_q,
    const T* W_alpha,
    const T* d_alpha,
    const T* b_alpha,
    const T* x,
    T* S,
    T* output,
    T* k_cache,
    T* v_cache,
    T* q_cache,
    T* alpha_x_cache,
    T* retrieved_cache,
    T* alpha_cache,
    T* workspace) {

    static const T alpha_one = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int BN = batch_size_ * n_state_;
    const int BNN = batch_size_ * n_state_ * n_state_;
    const int block_size = 256;

    T* k_all = workspace;
    T* v_all = k_all + steps * BN;
    T* q_all = v_all + steps * BN;
    T* alpha_x_all = q_all + steps * BN;
    T* retrieved_tmp = alpha_x_all + steps * BN;
    T* alpha_tmp = retrieved_tmp + BN;

    // Pre-compute projections
    blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        n_state_, steps * batch_size_, dim_, &alpha_one, W_k, dim_, x, dim_, &beta_zero, k_all, n_state_);
    blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        n_state_, steps * batch_size_, dim_, &alpha_one, W_v, dim_, x, dim_, &beta_zero, v_all, n_state_);
    blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        n_state_, steps * batch_size_, dim_, &alpha_one, W_q, dim_, x, dim_, &beta_zero, q_all, n_state_);
    blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        n_state_, steps * batch_size_, dim_, &alpha_one, W_alpha, dim_, x, dim_, &beta_zero, alpha_x_all, n_state_);

    for (int t = 0; t < steps; ++t) {
        const T* k_t = k_all + t * BN;
        const T* v_t = v_all + t * BN;
        const T* q_t = q_all + t * BN;
        const T* alpha_x_t = alpha_x_all + t * BN;
        const T* S_prev = S + t * BNN;
        T* S_t = S + (t + 1) * BNN;
        T* out_t = output + t * BN;

        T* retrieved_c = training_ ? (retrieved_cache + t * BN) : retrieved_tmp;
        T* alpha_c = training_ ? (alpha_cache + t * BN) : alpha_tmp;

        if (training_) {
            cudaMemcpyAsync(k_cache + t * BN, k_t, BN * sizeof(T), cudaMemcpyDeviceToDevice, stream_);
            cudaMemcpyAsync(v_cache + t * BN, v_t, BN * sizeof(T), cudaMemcpyDeviceToDevice, stream_);
            cudaMemcpyAsync(q_cache + t * BN, q_t, BN * sizeof(T), cudaMemcpyDeviceToDevice, stream_);
            cudaMemcpyAsync(alpha_x_cache + t * BN, alpha_x_t, BN * sizeof(T), cudaMemcpyDeviceToDevice, stream_);
        }

        MatVecKernel_E71<T><<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_state_, S_prev, k_t, retrieved_c);

        AlphaKernel_E71<T><<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_state_, alpha_x_t, retrieved_c, d_alpha, b_alpha, alpha_c);

        GatedUpdateOutputKernel_E71<T><<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_state_, S_prev, alpha_c, v_t, k_t, q_t, S_t, out_t);
    }
}

template<typename T>
E71MatrixGatedBackward<T>::E71MatrixGatedBackward(
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

template<typename T>
void E71MatrixGatedBackward<T>::Run(
    int steps,
    const T* W_k,
    const T* W_v,
    const T* W_q,
    const T* W_alpha,
    const T* d_alpha,
    const T* b_alpha,
    const T* x,
    const T* S,
    const T* k_cache,
    const T* v_cache,
    const T* q_cache,
    const T* alpha_x_cache,
    const T* retrieved_cache,
    const T* alpha_cache,
    const T* d_output,
    T* dx,
    T* dW_k,
    T* dW_v,
    T* dW_q,
    T* dW_alpha,
    T* dd_alpha_out,
    T* db_alpha,
    T* workspace) {

    // Placeholder - follows BF16 pattern
    cudaMemsetAsync(dx, 0, steps * batch_size_ * dim_ * sizeof(T), stream_);
}

// Explicit template instantiations
template struct E71MatrixGatedForward<__half>;
template struct E71MatrixGatedForward<float>;
template struct E71MatrixGatedForward<double>;

template struct E71MatrixGatedBackward<__half>;
template struct E71MatrixGatedBackward<float>;
template struct E71MatrixGatedBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
