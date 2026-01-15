// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E72: Matrix SelfGate Elman - Memory content controls writing
//
// Architecture:
// k = W_k @ x                              # [B, n]
// v = W_v @ x                              # [B, n]
// q = W_q @ x                              # [B, n]
// alpha = sigmoid(W_alpha @ x + b_alpha)   # [B, n] retain gate
//
// # S gates value (self-gating!)
// retrieved = S @ k                        # [B, n]
// g = sigmoid(d_g * retrieved + b_g)       # gate from memory content
// v_gated = v * g                          # memory controls writing
//
// # Gated update
// S = alpha.unsqueeze(-1) * S + (1 - alpha.unsqueeze(-1)) * outer(v_gated, k)
//
// out = S @ q                              # [B, n]
// out = out * silu(out)                    # self-gate
//
// Key insight: S gates what gets written! When retrieved content is large,
// the gate opens/closes (depending on variant), controlling writability.
//
// Variants:
// - standard: g = sigmoid(d_g * retrieved + b_g) - content enables writing
// - inverse:  g = sigmoid(-d_g * |retrieved| + b_g) - content resists writing
//
// State S is [B, n_state, n_state] matrix.

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
// Native BF16 operations (fast path for SM80+)
// =============================================================================

__device__ __forceinline__ __nv_bfloat16 bf16_add(__nv_bfloat16 a, __nv_bfloat16 b) {
#if __CUDA_ARCH__ >= 800
    return __hadd(a, b);
#else
    return __float2bfloat16(__bfloat162float(a) + __bfloat162float(b));
#endif
}

__device__ __forceinline__ __nv_bfloat16 bf16_mul(__nv_bfloat16 a, __nv_bfloat16 b) {
#if __CUDA_ARCH__ >= 800
    return __hmul(a, b);
#else
    return __float2bfloat16(__bfloat162float(a) * __bfloat162float(b));
#endif
}

// =============================================================================
// Forward Kernels
// =============================================================================

// Compute S @ k -> retrieved (batched matrix-vector multiply)
// S: [B, N, N], k: [B, N] -> retrieved: [B, N]
template<typename T>
__global__ void MatVecKernel(
    const int batch_size,
    const int N,
    const T* __restrict__ S,      // [B, N, N]
    const T* __restrict__ k,      // [B, N]
    T* __restrict__ retrieved) {  // [B, N]

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

// BF16 specialization
__global__ void MatVecKernel_BF16(
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

// Compute g = sigmoid(d_g * retrieved + b_g) and v_gated = v * g
// Standard variant: g increases with retrieved magnitude (content enables writing)
template<typename T>
__global__ void GateKernel(
    const int batch_size,
    const int N,
    const T* __restrict__ retrieved,   // [B, N]
    const T* __restrict__ v,           // [B, N]
    const T* __restrict__ d_g,         // [N]
    const T* __restrict__ b_g,         // [N]
    T* __restrict__ g_out,             // [B, N]
    T* __restrict__ v_gated) {         // [B, N]

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int n = idx % N;

        float r = static_cast<float>(retrieved[idx]);
        float dg = static_cast<float>(d_g[n]);
        float bg = static_cast<float>(b_g[n]);
        float v_val = static_cast<float>(v[idx]);

        // Standard: g = sigmoid(d_g * retrieved + b_g)
        float g_logit = dg * r + bg;
        float g = 1.0f / (1.0f + expf(-g_logit));

        g_out[idx] = static_cast<T>(g);
        v_gated[idx] = static_cast<T>(v_val * g);
    }
}

__global__ void GateKernel_BF16(
    const int batch_size,
    const int N,
    const __nv_bfloat16* __restrict__ retrieved,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ d_g,
    const __nv_bfloat16* __restrict__ b_g,
    __nv_bfloat16* __restrict__ g_out,
    __nv_bfloat16* __restrict__ v_gated) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int n = idx % N;

        float r = __bfloat162float(retrieved[idx]);
        float dg = __bfloat162float(d_g[n]);
        float bg = __bfloat162float(b_g[n]);
        float v_val = __bfloat162float(v[idx]);

        float g_logit = dg * r + bg;
        float g = 1.0f / (1.0f + __expf(-g_logit));

        g_out[idx] = __float2bfloat16(g);
        v_gated[idx] = __float2bfloat16(v_val * g);
    }
}

// Inverse variant: g = sigmoid(-d_g * |retrieved| + b_g)
// Large |retrieved| -> small g -> resist overwriting
template<typename T>
__global__ void GateKernelInverse(
    const int batch_size,
    const int N,
    const T* __restrict__ retrieved,
    const T* __restrict__ v,
    const T* __restrict__ d_g,
    const T* __restrict__ b_g,
    T* __restrict__ g_out,
    T* __restrict__ v_gated) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int n = idx % N;

        float r = static_cast<float>(retrieved[idx]);
        float dg = static_cast<float>(d_g[n]);
        float bg = static_cast<float>(b_g[n]);
        float v_val = static_cast<float>(v[idx]);

        // Inverse: g = sigmoid(-d_g * |retrieved| + b_g)
        float g_logit = -dg * fabsf(r) + bg;
        float g = 1.0f / (1.0f + expf(-g_logit));

        g_out[idx] = static_cast<T>(g);
        v_gated[idx] = static_cast<T>(v_val * g);
    }
}

__global__ void GateKernelInverse_BF16(
    const int batch_size,
    const int N,
    const __nv_bfloat16* __restrict__ retrieved,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ d_g,
    const __nv_bfloat16* __restrict__ b_g,
    __nv_bfloat16* __restrict__ g_out,
    __nv_bfloat16* __restrict__ v_gated) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int n = idx % N;

        float r = __bfloat162float(retrieved[idx]);
        float dg = __bfloat162float(d_g[n]);
        float bg = __bfloat162float(b_g[n]);
        float v_val = __bfloat162float(v[idx]);

        float g_logit = -dg * fabsf(r) + bg;
        float g = 1.0f / (1.0f + __expf(-g_logit));

        g_out[idx] = __float2bfloat16(g);
        v_gated[idx] = __float2bfloat16(v_val * g);
    }
}

// Gated matrix state update:
// S_new[b, i, j] = alpha[b, i] * S_prev[b, i, j] + (1 - alpha[b, i]) * v_gated[b, i] * k[b, j]
template<typename T>
__global__ void GatedStateUpdateKernel(
    const int batch_size,
    const int N,
    const T* __restrict__ S_prev,    // [B, N, N]
    const T* __restrict__ alpha,     // [B, N]
    const T* __restrict__ v_gated,   // [B, N]
    const T* __restrict__ k,         // [B, N]
    T* __restrict__ S_new) {         // [B, N, N]

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N * N;

    if (idx < total) {
        const int j = idx % N;
        const int i = (idx / N) % N;
        const int b = idx / (N * N);

        const int bi_idx = b * N + i;
        const int bj_idx = b * N + j;

        float alpha_val = static_cast<float>(alpha[bi_idx]);
        float s_prev_val = static_cast<float>(S_prev[idx]);
        float v_val = static_cast<float>(v_gated[bi_idx]);
        float k_val = static_cast<float>(k[bj_idx]);

        // S_new = alpha * S_prev + (1 - alpha) * v_gated * k
        float s_new_val = alpha_val * s_prev_val + (1.0f - alpha_val) * v_val * k_val;
        S_new[idx] = static_cast<T>(s_new_val);
    }
}

__global__ void GatedStateUpdateKernel_BF16(
    const int batch_size,
    const int N,
    const __nv_bfloat16* __restrict__ S_prev,
    const __nv_bfloat16* __restrict__ alpha,
    const __nv_bfloat16* __restrict__ v_gated,
    const __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ S_new) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N * N;

    if (idx < total) {
        const int j = idx % N;
        const int i = (idx / N) % N;
        const int b = idx / (N * N);

        const int bi_idx = b * N + i;
        const int bj_idx = b * N + j;

        float alpha_val = __bfloat162float(alpha[bi_idx]);
        float s_prev_val = __bfloat162float(S_prev[idx]);
        float v_val = __bfloat162float(v_gated[bi_idx]);
        float k_val = __bfloat162float(k[bj_idx]);

        float s_new_val = alpha_val * s_prev_val + (1.0f - alpha_val) * v_val * k_val;
        S_new[idx] = __float2bfloat16(s_new_val);
    }
}

// Compute output = S @ q with self-gating: out = out * silu(out)
// S: [B, N, N], q: [B, N] -> output: [B, N]
template<typename T>
__global__ void OutputKernel(
    const int batch_size,
    const int N,
    const T* __restrict__ S,      // [B, N, N]
    const T* __restrict__ q,      // [B, N]
    T* __restrict__ output) {     // [B, N]

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int b = idx / N;
        const int n = idx % N;

        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            float s_val = static_cast<float>(S[b * N * N + n * N + j]);
            float q_val = static_cast<float>(q[b * N + j]);
            sum += s_val * q_val;
        }

        // Self-gate: out = out * silu(out)
        float sigmoid_out = 1.0f / (1.0f + expf(-sum));
        float silu_out = sum * sigmoid_out;
        output[idx] = static_cast<T>(sum * silu_out);
    }
}

__global__ void OutputKernel_BF16(
    const int batch_size,
    const int N,
    const __nv_bfloat16* __restrict__ S,
    const __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ output) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int b = idx / N;
        const int n = idx % N;

        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            float s_val = __bfloat162float(S[b * N * N + n * N + j]);
            float q_val = __bfloat162float(q[b * N + j]);
            sum += s_val * q_val;
        }

        float sigmoid_out = 1.0f / (1.0f + __expf(-sum));
        float silu_out = sum * sigmoid_out;
        output[idx] = __float2bfloat16(sum * silu_out);
    }
}

// Compute alpha = sigmoid(alpha_x + b_alpha)
template<typename T>
__global__ void AlphaKernel(
    const int batch_size,
    const int N,
    const T* __restrict__ alpha_x,    // [B, N]
    const T* __restrict__ b_alpha,    // [N]
    T* __restrict__ alpha_out) {      // [B, N]

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int n = idx % N;

        float val = static_cast<float>(alpha_x[idx]) + static_cast<float>(b_alpha[n]);
        alpha_out[idx] = static_cast<T>(1.0f / (1.0f + expf(-val)));
    }
}

__global__ void AlphaKernel_BF16(
    const int batch_size,
    const int N,
    const __nv_bfloat16* __restrict__ alpha_x,
    const __nv_bfloat16* __restrict__ b_alpha,
    __nv_bfloat16* __restrict__ alpha_out) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int n = idx % N;

        float val = __bfloat162float(alpha_x[idx]) + __bfloat162float(b_alpha[n]);
        alpha_out[idx] = __float2bfloat16(1.0f / (1.0f + __expf(-val)));
    }
}

// =============================================================================
// Backward Kernels
// =============================================================================

// Backward through self-gate output: output = raw * silu(raw) = raw^2 * sigmoid(raw)
// d(output)/d(raw) = silu(raw) * (2 + raw*(1-sigmoid(raw)))
template<typename T>
__global__ void OutputBackwardKernel(
    const int batch_size,
    const int N,
    const T* __restrict__ S,          // [B, N, N]
    const T* __restrict__ q,          // [B, N]
    const T* __restrict__ d_output,   // [B, N]
    T* __restrict__ d_S,              // [B, N, N] - add to existing
    float* __restrict__ d_q_f) {      // [B, N] float accumulator

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int b = idx / N;
        const int n = idx % N;

        // Recompute raw = S @ q
        float raw = 0.0f;
        for (int j = 0; j < N; ++j) {
            float s_val = static_cast<float>(S[b * N * N + n * N + j]);
            float q_val = static_cast<float>(q[b * N + j]);
            raw += s_val * q_val;
        }

        // Derivative of output = raw * silu(raw) w.r.t. raw
        float sigmoid_raw = 1.0f / (1.0f + expf(-raw));
        float silu_raw = raw * sigmoid_raw;
        float grad_factor = silu_raw * (2.0f + raw * (1.0f - sigmoid_raw));

        float dout = static_cast<float>(d_output[idx]);
        float d_raw = dout * grad_factor;

        // d_S[b,n,j] += d_raw * q[j]
        // d_q[j] += d_raw * S[b,n,j]
        for (int j = 0; j < N; ++j) {
            int s_idx = b * N * N + n * N + j;
            float q_val = static_cast<float>(q[b * N + j]);
            float s_val = static_cast<float>(S[s_idx]);

            // Add to d_S
            d_S[s_idx] = static_cast<T>(static_cast<float>(d_S[s_idx]) + d_raw * q_val);

            // Add to d_q
            atomicAdd(&d_q_f[b * N + j], d_raw * s_val);
        }
    }
}

__global__ void OutputBackwardKernel_BF16(
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

        float raw = 0.0f;
        for (int j = 0; j < N; ++j) {
            float s_val = __bfloat162float(S[b * N * N + n * N + j]);
            float q_val = __bfloat162float(q[b * N + j]);
            raw += s_val * q_val;
        }

        float sigmoid_raw = 1.0f / (1.0f + __expf(-raw));
        float silu_raw = raw * sigmoid_raw;
        float grad_factor = silu_raw * (2.0f + raw * (1.0f - sigmoid_raw));

        float dout = __bfloat162float(d_output[idx]);
        float d_raw = dout * grad_factor;

        for (int j = 0; j < N; ++j) {
            int s_idx = b * N * N + n * N + j;
            float q_val = __bfloat162float(q[b * N + j]);
            float s_val = __bfloat162float(S[s_idx]);

            d_S[s_idx] = __float2bfloat16(__bfloat162float(d_S[s_idx]) + d_raw * q_val);
            atomicAdd(&d_q_f[b * N + j], d_raw * s_val);
        }
    }
}

// Element-wise multiply: out = a * b
__global__ void ElemwiseMulKernel_BF16(
    const int n,
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(__bfloat162float(a[idx]) * __bfloat162float(b[idx]));
    }
}

// Element-wise multiply with (1-b): out = a * (1 - b), for inverse mode
__global__ void ElemwiseMulOneMinusKernel_BF16(
    const int n,
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(__bfloat162float(a[idx]) * (1.0f - __bfloat162float(b[idx])));
    }
}

// Backward through gated state update:
// S_new = alpha * S_prev + (1 - alpha) * v_gated @ k^T
template<typename T>
__global__ void StateUpdateBackwardKernel(
    const int batch_size,
    const int N,
    const T* __restrict__ S_prev,     // [B, N, N]
    const T* __restrict__ alpha,      // [B, N]
    const T* __restrict__ v_gated,    // [B, N]
    const T* __restrict__ k,          // [B, N]
    const T* __restrict__ d_S_new,    // [B, N, N]
    T* __restrict__ d_S_prev,         // [B, N, N]
    float* __restrict__ d_alpha_f,    // [B, N] float accumulator
    float* __restrict__ d_v_gated_f,  // [B, N] float accumulator
    float* __restrict__ d_k_f) {      // [B, N] float accumulator

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N * N;

    if (idx < total) {
        const int j = idx % N;
        const int i = (idx / N) % N;
        const int b = idx / (N * N);

        const int bi_idx = b * N + i;
        const int bj_idx = b * N + j;

        float alpha_val = static_cast<float>(alpha[bi_idx]);
        float s_prev_val = static_cast<float>(S_prev[idx]);
        float v_val = static_cast<float>(v_gated[bi_idx]);
        float k_val = static_cast<float>(k[bj_idx]);
        float ds_new = static_cast<float>(d_S_new[idx]);

        // d_S_prev = d_S_new * alpha
        d_S_prev[idx] = static_cast<T>(ds_new * alpha_val);

        // d_alpha[b,i] += d_S_new[b,i,j] * (S_prev[b,i,j] - v_gated[b,i] * k[b,j])
        float d_alpha_contrib = ds_new * (s_prev_val - v_val * k_val);
        atomicAdd(&d_alpha_f[bi_idx], d_alpha_contrib);

        // d_v_gated[b,i] += d_S_new[b,i,j] * (1-alpha) * k[b,j]
        float d_v_contrib = ds_new * (1.0f - alpha_val) * k_val;
        atomicAdd(&d_v_gated_f[bi_idx], d_v_contrib);

        // d_k[b,j] += d_S_new[b,i,j] * (1-alpha) * v_gated[b,i]
        float d_k_contrib = ds_new * (1.0f - alpha_val) * v_val;
        atomicAdd(&d_k_f[bj_idx], d_k_contrib);
    }
}

__global__ void StateUpdateBackwardKernel_BF16(
    const int batch_size,
    const int N,
    const __nv_bfloat16* __restrict__ S_prev,
    const __nv_bfloat16* __restrict__ alpha,
    const __nv_bfloat16* __restrict__ v_gated,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ d_S_new,
    __nv_bfloat16* __restrict__ d_S_prev,
    float* __restrict__ d_alpha_f,
    float* __restrict__ d_v_gated_f,
    float* __restrict__ d_k_f) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N * N;

    if (idx < total) {
        const int j = idx % N;
        const int i = (idx / N) % N;
        const int b = idx / (N * N);

        const int bi_idx = b * N + i;
        const int bj_idx = b * N + j;

        float alpha_val = __bfloat162float(alpha[bi_idx]);
        float s_prev_val = __bfloat162float(S_prev[idx]);
        float v_val = __bfloat162float(v_gated[bi_idx]);
        float k_val = __bfloat162float(k[bj_idx]);
        float ds_new = __bfloat162float(d_S_new[idx]);

        d_S_prev[idx] = __float2bfloat16(ds_new * alpha_val);

        float d_alpha_contrib = ds_new * (s_prev_val - v_val * k_val);
        atomicAdd(&d_alpha_f[bi_idx], d_alpha_contrib);

        float d_v_contrib = ds_new * (1.0f - alpha_val) * k_val;
        atomicAdd(&d_v_gated_f[bi_idx], d_v_contrib);

        float d_k_contrib = ds_new * (1.0f - alpha_val) * v_val;
        atomicAdd(&d_k_f[bj_idx], d_k_contrib);
    }
}

// Backward through gate: v_gated = v * g
// Standard: g = sigmoid(d_g * retrieved + b_g)
template<typename T>
__global__ void GateBackwardKernel(
    const int batch_size,
    const int N,
    const T* __restrict__ retrieved,   // [B, N]
    const T* __restrict__ v,           // [B, N]
    const T* __restrict__ g,           // [B, N]
    const T* __restrict__ d_g_param,   // [N] parameter
    const float* __restrict__ d_v_gated_f,  // [B, N]
    float* __restrict__ d_v_f,         // [B, N]
    float* __restrict__ d_retrieved_f, // [B, N]
    float* __restrict__ dd_g_f,        // [N] gradient for d_g
    float* __restrict__ db_g_f) {      // [N] gradient for b_g

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int n = idx % N;

        float v_val = static_cast<float>(v[idx]);
        float g_val = static_cast<float>(g[idx]);
        float r_val = static_cast<float>(retrieved[idx]);
        float dg_param = static_cast<float>(d_g_param[n]);
        float dv_gated = d_v_gated_f[idx];

        // d_v = dv_gated * g
        d_v_f[idx] = dv_gated * g_val;

        // dg = dv_gated * v
        float dg = dv_gated * v_val;

        // g = sigmoid(d_g * r + b_g)
        // dg_logit = dg * g * (1-g)
        float dg_logit = dg * g_val * (1.0f - g_val);

        // d_retrieved = dg_logit * d_g
        d_retrieved_f[idx] = dg_logit * dg_param;

        // dd_g += dg_logit * retrieved
        atomicAdd(&dd_g_f[n], dg_logit * r_val);

        // db_g += dg_logit
        atomicAdd(&db_g_f[n], dg_logit);
    }
}

__global__ void GateBackwardKernel_BF16(
    const int batch_size,
    const int N,
    const __nv_bfloat16* __restrict__ retrieved,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ g,
    const __nv_bfloat16* __restrict__ d_g_param,
    const float* __restrict__ d_v_gated_f,
    float* __restrict__ d_v_f,
    float* __restrict__ d_retrieved_f,
    float* __restrict__ dd_g_f,
    float* __restrict__ db_g_f) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int n = idx % N;

        float v_val = __bfloat162float(v[idx]);
        float g_val = __bfloat162float(g[idx]);
        float r_val = __bfloat162float(retrieved[idx]);
        float dg_param = __bfloat162float(d_g_param[n]);
        float dv_gated = d_v_gated_f[idx];

        d_v_f[idx] = dv_gated * g_val;

        float dg = dv_gated * v_val;
        float dg_logit = dg * g_val * (1.0f - g_val);

        d_retrieved_f[idx] = dg_logit * dg_param;

        atomicAdd(&dd_g_f[n], dg_logit * r_val);
        atomicAdd(&db_g_f[n], dg_logit);
    }
}

// Inverse gate backward: g = sigmoid(-d_g * |retrieved| + b_g)
template<typename T>
__global__ void GateBackwardKernelInverse(
    const int batch_size,
    const int N,
    const T* __restrict__ retrieved,
    const T* __restrict__ v,
    const T* __restrict__ g,
    const T* __restrict__ d_g_param,
    const float* __restrict__ d_v_gated_f,
    float* __restrict__ d_v_f,
    float* __restrict__ d_retrieved_f,
    float* __restrict__ dd_g_f,
    float* __restrict__ db_g_f) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int n = idx % N;

        float v_val = static_cast<float>(v[idx]);
        float g_val = static_cast<float>(g[idx]);
        float r_val = static_cast<float>(retrieved[idx]);
        float dg_param = static_cast<float>(d_g_param[n]);
        float dv_gated = d_v_gated_f[idx];

        d_v_f[idx] = dv_gated * g_val;

        float dg = dv_gated * v_val;
        float dg_logit = dg * g_val * (1.0f - g_val);

        // g = sigmoid(-d_g * |r| + b_g)
        // d|r|/dr = sign(r)
        float sign_r = (r_val >= 0.0f) ? 1.0f : -1.0f;
        d_retrieved_f[idx] = dg_logit * (-dg_param) * sign_r;

        // dd_g += dg_logit * (-|r|)
        atomicAdd(&dd_g_f[n], dg_logit * (-fabsf(r_val)));

        atomicAdd(&db_g_f[n], dg_logit);
    }
}

__global__ void GateBackwardKernelInverse_BF16(
    const int batch_size,
    const int N,
    const __nv_bfloat16* __restrict__ retrieved,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ g,
    const __nv_bfloat16* __restrict__ d_g_param,
    const float* __restrict__ d_v_gated_f,
    float* __restrict__ d_v_f,
    float* __restrict__ d_retrieved_f,
    float* __restrict__ dd_g_f,
    float* __restrict__ db_g_f) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int n = idx % N;

        float v_val = __bfloat162float(v[idx]);
        float g_val = __bfloat162float(g[idx]);
        float r_val = __bfloat162float(retrieved[idx]);
        float dg_param = __bfloat162float(d_g_param[n]);
        float dv_gated = d_v_gated_f[idx];

        d_v_f[idx] = dv_gated * g_val;

        float dg = dv_gated * v_val;
        float dg_logit = dg * g_val * (1.0f - g_val);

        float sign_r = (r_val >= 0.0f) ? 1.0f : -1.0f;
        d_retrieved_f[idx] = dg_logit * (-dg_param) * sign_r;

        atomicAdd(&dd_g_f[n], dg_logit * (-fabsf(r_val)));
        atomicAdd(&db_g_f[n], dg_logit);
    }
}

// Backward through retrieval: retrieved = S @ k
template<typename T>
__global__ void RetrievalBackwardKernel(
    const int batch_size,
    const int N,
    const T* __restrict__ S,          // [B, N, N]
    const T* __restrict__ k,          // [B, N]
    const float* __restrict__ d_retrieved_f,  // [B, N]
    T* __restrict__ d_S,              // [B, N, N] - add to existing
    float* __restrict__ d_k_f) {      // [B, N] - add to existing

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int b = idx / N;
        const int n = idx % N;

        float d_ret = d_retrieved_f[idx];

        // d_S[b,n,j] += d_retrieved[b,n] * k[b,j]
        // d_k[b,j] += d_retrieved[b,n] * S[b,n,j]
        for (int j = 0; j < N; ++j) {
            int s_idx = b * N * N + n * N + j;
            float k_val = static_cast<float>(k[b * N + j]);
            float s_val = static_cast<float>(S[s_idx]);

            // Add to d_S
            d_S[s_idx] = static_cast<T>(static_cast<float>(d_S[s_idx]) + d_ret * k_val);

            // Add to d_k
            atomicAdd(&d_k_f[b * N + j], d_ret * s_val);
        }
    }
}

__global__ void RetrievalBackwardKernel_BF16(
    const int batch_size,
    const int N,
    const __nv_bfloat16* __restrict__ S,
    const __nv_bfloat16* __restrict__ k,
    const float* __restrict__ d_retrieved_f,
    __nv_bfloat16* __restrict__ d_S,
    float* __restrict__ d_k_f) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int b = idx / N;
        const int n = idx % N;

        float d_ret = d_retrieved_f[idx];

        for (int j = 0; j < N; ++j) {
            int s_idx = b * N * N + n * N + j;
            float k_val = __bfloat162float(k[b * N + j]);
            float s_val = __bfloat162float(S[s_idx]);

            d_S[s_idx] = __float2bfloat16(__bfloat162float(d_S[s_idx]) + d_ret * k_val);
            atomicAdd(&d_k_f[b * N + j], d_ret * s_val);
        }
    }
}

// Backward through alpha: alpha = sigmoid(alpha_x + b_alpha)
template<typename T>
__global__ void AlphaBackwardKernel(
    const int batch_size,
    const int N,
    const T* __restrict__ alpha,      // [B, N]
    const float* __restrict__ d_alpha_f,  // [B, N]
    float* __restrict__ d_alpha_x_f,  // [B, N]
    float* __restrict__ db_alpha_f) { // [N]

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int n = idx % N;

        float alpha_val = static_cast<float>(alpha[idx]);
        float d_alpha = d_alpha_f[idx];

        // d_logit = d_alpha * alpha * (1-alpha)
        float d_logit = d_alpha * alpha_val * (1.0f - alpha_val);

        d_alpha_x_f[idx] = d_logit;
        atomicAdd(&db_alpha_f[n], d_logit);
    }
}

__global__ void AlphaBackwardKernel_BF16(
    const int batch_size,
    const int N,
    const __nv_bfloat16* __restrict__ alpha,
    const float* __restrict__ d_alpha_f,
    float* __restrict__ d_alpha_x_f,
    float* __restrict__ db_alpha_f) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int n = idx % N;

        float alpha_val = __bfloat162float(alpha[idx]);
        float d_alpha = d_alpha_f[idx];

        float d_logit = d_alpha * alpha_val * (1.0f - alpha_val);

        d_alpha_x_f[idx] = d_logit;
        atomicAdd(&db_alpha_f[n], d_logit);
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
// E72 Matrix SelfGate Forward - BF16 Specialization
// =============================================================================

template<>
E72MatrixSelfGateForward<__nv_bfloat16>::E72MatrixSelfGateForward(
    bool training,
    int batch_size,
    int dim,
    int n_state,
    bool inverse_gate,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : training_(training),
      batch_size_(batch_size),
      dim_(dim),
      n_state_(n_state),
      inverse_gate_(inverse_gate),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void E72MatrixSelfGateForward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_k,       // [n_state, dim]
    const __nv_bfloat16* W_v,       // [n_state, dim]
    const __nv_bfloat16* W_q,       // [n_state, dim]
    const __nv_bfloat16* W_alpha,   // [n_state, dim]
    const __nv_bfloat16* b_alpha,   // [n_state]
    const __nv_bfloat16* d_g,       // [n_state] gating weight
    const __nv_bfloat16* b_g,       // [n_state] gating bias
    const __nv_bfloat16* x,         // [T, B, dim] input
    __nv_bfloat16* S,               // [T+1, B, n_state, n_state] state matrices
    __nv_bfloat16* output,          // [T, B, n_state] output
    __nv_bfloat16* k_cache,         // [T, B, n_state]
    __nv_bfloat16* v_cache,         // [T, B, n_state]
    __nv_bfloat16* q_cache,         // [T, B, n_state]
    __nv_bfloat16* alpha_cache,     // [T, B, n_state]
    __nv_bfloat16* retrieved_cache, // [T, B, n_state]
    __nv_bfloat16* g_cache,         // [T, B, n_state]
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
    // alpha_x_all: [T, B, n_state]
    // v_gated_tmp: [B, n_state]
    // alpha_tmp: [B, n_state]
    __nv_bfloat16* k_all = workspace;
    __nv_bfloat16* v_all = k_all + steps * BN;
    __nv_bfloat16* q_all = v_all + steps * BN;
    __nv_bfloat16* alpha_x_all = q_all + steps * BN;
    __nv_bfloat16* v_gated_tmp = alpha_x_all + steps * BN;
    __nv_bfloat16* alpha_tmp = v_gated_tmp + BN;

    // Pre-compute all x projections: k, v, q, alpha_x
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

    // v_all = x @ W_v.T
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        n_state_, steps * batch_size_, dim_,
        &alpha_one,
        W_v, dim_,
        x, dim_,
        &beta_zero,
        v_all, n_state_);

    // q_all = x @ W_q.T
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        n_state_, steps * batch_size_, dim_,
        &alpha_one,
        W_q, dim_,
        x, dim_,
        &beta_zero,
        q_all, n_state_);

    // alpha_x_all = x @ W_alpha.T
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        n_state_, steps * batch_size_, dim_,
        &alpha_one,
        W_alpha, dim_,
        x, dim_,
        &beta_zero,
        alpha_x_all, n_state_);

    // Process each timestep sequentially
    for (int t = 0; t < steps; ++t) {
        const __nv_bfloat16* k_t = k_all + t * BN;
        const __nv_bfloat16* v_t = v_all + t * BN;
        const __nv_bfloat16* q_t = q_all + t * BN;
        const __nv_bfloat16* alpha_x_t = alpha_x_all + t * BN;
        const __nv_bfloat16* S_prev = S + t * BNN;
        __nv_bfloat16* S_t = S + (t + 1) * BNN;
        __nv_bfloat16* out_t = output + t * BN;

        // Cache pointers
        __nv_bfloat16* k_c = training_ ? (k_cache + t * BN) : nullptr;
        __nv_bfloat16* v_c = training_ ? (v_cache + t * BN) : nullptr;
        __nv_bfloat16* q_c = training_ ? (q_cache + t * BN) : nullptr;
        __nv_bfloat16* alpha_c = training_ ? (alpha_cache + t * BN) : alpha_tmp;
        __nv_bfloat16* retrieved_c = training_ ? (retrieved_cache + t * BN) : v_gated_tmp;  // reuse
        __nv_bfloat16* g_c = training_ ? (g_cache + t * BN) : v_gated_tmp;  // reuse

        // Cache k, v, q if training
        if (training_) {
            cudaMemcpyAsync(k_c, k_t, BN * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream_);
            cudaMemcpyAsync(v_c, v_t, BN * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream_);
            cudaMemcpyAsync(q_c, q_t, BN * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream_);
        }

        // 1. Compute alpha = sigmoid(alpha_x + b_alpha)
        AlphaKernel_BF16<<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_state_, alpha_x_t, b_alpha, alpha_c);

        // 2. Compute retrieved = S @ k
        MatVecKernel_BF16<<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_state_, S_prev, k_t, retrieved_c);

        // 3. Compute g and v_gated
        if (inverse_gate_) {
            GateKernelInverse_BF16<<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
                batch_size_, n_state_, retrieved_c, v_t, d_g, b_g, g_c, v_gated_tmp);
        } else {
            GateKernel_BF16<<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
                batch_size_, n_state_, retrieved_c, v_t, d_g, b_g, g_c, v_gated_tmp);
        }

        // 4. Gated state update
        GatedStateUpdateKernel_BF16<<<(BNN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_state_, S_prev, alpha_c, v_gated_tmp, k_t, S_t);

        // 5. Output with self-gating
        OutputKernel_BF16<<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_state_, S_t, q_t, out_t);
    }
}

// =============================================================================
// E72 Matrix SelfGate Backward - BF16 Specialization
// =============================================================================

template<>
E72MatrixSelfGateBackward<__nv_bfloat16>::E72MatrixSelfGateBackward(
    int batch_size,
    int dim,
    int n_state,
    bool inverse_gate,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      n_state_(n_state),
      inverse_gate_(inverse_gate),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void E72MatrixSelfGateBackward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_k,
    const __nv_bfloat16* W_v,
    const __nv_bfloat16* W_q,
    const __nv_bfloat16* W_alpha,
    const __nv_bfloat16* d_g,
    const __nv_bfloat16* x,
    const __nv_bfloat16* S,
    const __nv_bfloat16* k_cache,
    const __nv_bfloat16* v_cache,
    const __nv_bfloat16* q_cache,
    const __nv_bfloat16* alpha_cache,
    const __nv_bfloat16* retrieved_cache,
    const __nv_bfloat16* g_cache,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* dx,
    __nv_bfloat16* dW_k,
    __nv_bfloat16* dW_v,
    __nv_bfloat16* dW_q,
    __nv_bfloat16* dW_alpha,
    __nv_bfloat16* db_alpha,
    __nv_bfloat16* dd_g_out,
    __nv_bfloat16* db_g,
    __nv_bfloat16* workspace) {

    static const __nv_bfloat16 alpha_one = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);
    static const __nv_bfloat16 beta_one = __float2bfloat16(1.0f);

    const int BN = batch_size_ * n_state_;
    const int BNN = batch_size_ * n_state_ * n_state_;
    const int block_size = 256;

    // Workspace layout:
    // d_S: [B, N, N]
    // d_S_tmp: [B, N, N]
    // d_k_all: [T, B, N]
    // d_v_all: [T, B, N]
    // d_q_all: [T, B, N]
    // d_alpha_x_all: [T, B, N]
    // Float accumulators:
    // d_k_f: [B, N]
    // d_v_f: [B, N]
    // d_q_f: [B, N]
    // d_alpha_f: [B, N]
    // d_v_gated_f: [B, N]
    // d_retrieved_f: [B, N]
    // d_alpha_x_f: [B, N]
    // dd_g_f: [N]
    // db_g_f: [N]
    // db_alpha_f: [N]
    __nv_bfloat16* d_S = workspace;
    __nv_bfloat16* d_S_tmp = d_S + BNN;
    __nv_bfloat16* d_k_all = d_S_tmp + BNN;
    __nv_bfloat16* d_v_all = d_k_all + steps * BN;
    __nv_bfloat16* d_q_all = d_v_all + steps * BN;
    __nv_bfloat16* d_alpha_x_all = d_q_all + steps * BN;

    // Temporary buffer for v_gated (recomputed in backward)
    __nv_bfloat16* v_gated_tmp = d_alpha_x_all + steps * BN;

    float* float_ws = reinterpret_cast<float*>(v_gated_tmp + BN);
    float* d_k_f = float_ws;
    float* d_v_f = d_k_f + BN;
    float* d_q_f = d_v_f + BN;
    float* d_alpha_f = d_q_f + BN;
    float* d_v_gated_f = d_alpha_f + BN;
    float* d_retrieved_f = d_v_gated_f + BN;
    float* d_alpha_x_f = d_retrieved_f + BN;
    float* dd_g_f = d_alpha_x_f + BN;
    float* db_g_f = dd_g_f + n_state_;
    float* db_alpha_f = db_g_f + n_state_;

    // Initialize gradients
    cudaMemsetAsync(d_S, 0, BNN * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_k, 0, n_state_ * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_v, 0, n_state_ * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_q, 0, n_state_ * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_alpha, 0, n_state_ * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dd_g_f, 0, n_state_ * sizeof(float), stream_);
    cudaMemsetAsync(db_g_f, 0, n_state_ * sizeof(float), stream_);
    cudaMemsetAsync(db_alpha_f, 0, n_state_ * sizeof(float), stream_);

    // BPTT loop
    for (int t = steps - 1; t >= 0; --t) {
        const __nv_bfloat16* S_t = S + (t + 1) * BNN;
        const __nv_bfloat16* S_prev = S + t * BNN;
        const __nv_bfloat16* k_t = k_cache + t * BN;
        const __nv_bfloat16* v_t = v_cache + t * BN;
        const __nv_bfloat16* q_t = q_cache + t * BN;
        const __nv_bfloat16* alpha_t = alpha_cache + t * BN;
        const __nv_bfloat16* retrieved_t = retrieved_cache + t * BN;
        const __nv_bfloat16* g_t = g_cache + t * BN;
        const __nv_bfloat16* d_out_t = d_output + t * BN;

        __nv_bfloat16* d_k_t = d_k_all + t * BN;
        __nv_bfloat16* d_v_t = d_v_all + t * BN;
        __nv_bfloat16* d_q_t = d_q_all + t * BN;
        __nv_bfloat16* d_alpha_x_t = d_alpha_x_all + t * BN;

        // Zero per-timestep float accumulators
        cudaMemsetAsync(d_k_f, 0, BN * sizeof(float), stream_);
        cudaMemsetAsync(d_v_f, 0, BN * sizeof(float), stream_);
        cudaMemsetAsync(d_q_f, 0, BN * sizeof(float), stream_);
        cudaMemsetAsync(d_alpha_f, 0, BN * sizeof(float), stream_);
        cudaMemsetAsync(d_v_gated_f, 0, BN * sizeof(float), stream_);
        cudaMemsetAsync(d_retrieved_f, 0, BN * sizeof(float), stream_);

        // 1. Backward through output (adds to d_S and d_q_f)
        OutputBackwardKernel_BF16<<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_state_, S_t, q_t, d_out_t, d_S, d_q_f);

        // 2. Recompute v_gated = v * g (g_cache has gate value for both modes)
        ElemwiseMulKernel_BF16<<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            BN, v_t, g_t, v_gated_tmp);

        // 3. Backward through state update (writes to d_S_tmp, accumulates d_alpha_f, d_v_gated_f, d_k_f)
        StateUpdateBackwardKernel_BF16<<<(BNN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_state_, S_prev, alpha_t, v_gated_tmp, k_t, d_S,
            d_S_tmp, d_alpha_f, d_v_gated_f, d_k_f);

        // 3. Backward through gate
        if (inverse_gate_) {
            GateBackwardKernelInverse_BF16<<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
                batch_size_, n_state_, retrieved_t, v_t, g_t, d_g,
                d_v_gated_f, d_v_f, d_retrieved_f, dd_g_f, db_g_f);
        } else {
            GateBackwardKernel_BF16<<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
                batch_size_, n_state_, retrieved_t, v_t, g_t, d_g,
                d_v_gated_f, d_v_f, d_retrieved_f, dd_g_f, db_g_f);
        }

        // 4. Backward through retrieval (adds to d_S_tmp and d_k_f)
        RetrievalBackwardKernel_BF16<<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_state_, S_prev, k_t, d_retrieved_f, d_S_tmp, d_k_f);

        // 5. Backward through alpha
        AlphaBackwardKernel_BF16<<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_state_, alpha_t, d_alpha_f, d_alpha_x_f, db_alpha_f);

        // Copy float accumulators to output
        CopyFloatToT<__nv_bfloat16><<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            BN, d_k_f, d_k_t);
        CopyFloatToT<__nv_bfloat16><<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            BN, d_v_f, d_v_t);
        CopyFloatToT<__nv_bfloat16><<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            BN, d_q_f, d_q_t);
        CopyFloatToT<__nv_bfloat16><<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            BN, d_alpha_x_f, d_alpha_x_t);

        // Swap d_S and d_S_tmp
        std::swap(d_S, d_S_tmp);
    }

    // Batch compute dx and weight gradients
    // dx = d_k @ W_k + d_v @ W_v + d_q @ W_q + d_alpha_x @ W_alpha

    // dx from k: dx = d_k_all @ W_k
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

    // Weight gradients: dW = x.T @ d_proj
    // dW_k = x.T @ d_k_all
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, n_state_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        d_k_all, n_state_,
        &beta_one,
        dW_k, dim_);

    // dW_v = x.T @ d_v_all
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, n_state_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        d_v_all, n_state_,
        &beta_one,
        dW_v, dim_);

    // dW_q = x.T @ d_q_all
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, n_state_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        d_q_all, n_state_,
        &beta_one,
        dW_q, dim_);

    // dW_alpha = x.T @ d_alpha_x_all
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, n_state_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        d_alpha_x_all, n_state_,
        &beta_one,
        dW_alpha, dim_);

    // Copy bias gradients
    CopyFloatToT<__nv_bfloat16><<<(n_state_ + 255) / 256, 256, 0, stream_>>>(n_state_, db_alpha_f, db_alpha);
    CopyFloatToT<__nv_bfloat16><<<(n_state_ + 255) / 256, 256, 0, stream_>>>(n_state_, dd_g_f, dd_g_out);
    CopyFloatToT<__nv_bfloat16><<<(n_state_ + 255) / 256, 256, 0, stream_>>>(n_state_, db_g_f, db_g);
}

// =============================================================================
// Generic Template Implementations
// =============================================================================

template<typename T>
E72MatrixSelfGateForward<T>::E72MatrixSelfGateForward(
    bool training,
    int batch_size,
    int dim,
    int n_state,
    bool inverse_gate,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : training_(training),
      batch_size_(batch_size),
      dim_(dim),
      n_state_(n_state),
      inverse_gate_(inverse_gate),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void E72MatrixSelfGateForward<T>::Run(
    int steps,
    const T* W_k,
    const T* W_v,
    const T* W_q,
    const T* W_alpha,
    const T* b_alpha,
    const T* d_g,
    const T* b_g,
    const T* x,
    T* S,
    T* output,
    T* k_cache,
    T* v_cache,
    T* q_cache,
    T* alpha_cache,
    T* retrieved_cache,
    T* g_cache,
    T* workspace) {

    // Generic implementation follows BF16 pattern
    static const T alpha_one = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int BN = batch_size_ * n_state_;
    const int BNN = batch_size_ * n_state_ * n_state_;
    const int block_size = 256;

    T* k_all = workspace;
    T* v_all = k_all + steps * BN;
    T* q_all = v_all + steps * BN;
    T* alpha_x_all = q_all + steps * BN;
    T* v_gated_tmp = alpha_x_all + steps * BN;
    T* alpha_tmp = v_gated_tmp + BN;

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

        T* alpha_c = training_ ? (alpha_cache + t * BN) : alpha_tmp;
        T* retrieved_c = training_ ? (retrieved_cache + t * BN) : v_gated_tmp;
        T* g_c = training_ ? (g_cache + t * BN) : v_gated_tmp;

        if (training_) {
            cudaMemcpyAsync(k_cache + t * BN, k_t, BN * sizeof(T), cudaMemcpyDeviceToDevice, stream_);
            cudaMemcpyAsync(v_cache + t * BN, v_t, BN * sizeof(T), cudaMemcpyDeviceToDevice, stream_);
            cudaMemcpyAsync(q_cache + t * BN, q_t, BN * sizeof(T), cudaMemcpyDeviceToDevice, stream_);
        }

        AlphaKernel<T><<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_state_, alpha_x_t, b_alpha, alpha_c);

        MatVecKernel<T><<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_state_, S_prev, k_t, retrieved_c);

        if (inverse_gate_) {
            GateKernelInverse<T><<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
                batch_size_, n_state_, retrieved_c, v_t, d_g, b_g, g_c, v_gated_tmp);
        } else {
            GateKernel<T><<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
                batch_size_, n_state_, retrieved_c, v_t, d_g, b_g, g_c, v_gated_tmp);
        }

        GatedStateUpdateKernel<T><<<(BNN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_state_, S_prev, alpha_c, v_gated_tmp, k_t, S_t);

        OutputKernel<T><<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_state_, S_t, q_t, out_t);
    }
}

template<typename T>
E72MatrixSelfGateBackward<T>::E72MatrixSelfGateBackward(
    int batch_size,
    int dim,
    int n_state,
    bool inverse_gate,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      n_state_(n_state),
      inverse_gate_(inverse_gate),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void E72MatrixSelfGateBackward<T>::Run(
    int steps,
    const T* W_k,
    const T* W_v,
    const T* W_q,
    const T* W_alpha,
    const T* d_g,
    const T* x,
    const T* S,
    const T* k_cache,
    const T* v_cache,
    const T* q_cache,
    const T* alpha_cache,
    const T* retrieved_cache,
    const T* g_cache,
    const T* d_output,
    T* dx,
    T* dW_k,
    T* dW_v,
    T* dW_q,
    T* dW_alpha,
    T* db_alpha,
    T* dd_g_out,
    T* db_g,
    T* workspace) {

    // Placeholder - follows BF16 pattern
    cudaMemsetAsync(dx, 0, steps * batch_size_ * dim_ * sizeof(T), stream_);
}

// Explicit template instantiations
template struct E72MatrixSelfGateForward<__half>;
template struct E72MatrixSelfGateForward<float>;
template struct E72MatrixSelfGateForward<double>;

template struct E72MatrixSelfGateBackward<__half>;
template struct E72MatrixSelfGateBackward<float>;
template struct E72MatrixSelfGateBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
