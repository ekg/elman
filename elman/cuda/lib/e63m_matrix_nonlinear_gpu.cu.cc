// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E63m: Matrix State Nonlinear Delta - Maximum Expressivity
//
// Matrix state S ∈ ℝ^(N×D) with NONLINEAR retrieval and update.
// This is UTM-class while DeltaNet is not.
//
// Forward:
//     k_t = x @ W_k.T                                # [B, D] key
//     q_t = x @ W_q.T                                # [B, D] query
//
//     # NONLINEAR retrieval (the key difference!)
//     Sk = bmm(S, k_t.unsqueeze(-1)).squeeze(-1)     # [B, N]
//     retrieved = tanh(Sk)                           # [B, N] nonlinear!
//
//     # Value computation
//     v_t = tanh(retrieved @ W_r.T + x @ W_x.T + b)  # [B, N]
//
//     # Gated update
//     alpha_t = sigmoid(x @ W_alpha.T + b_alpha)     # [B, N]
//     v_outer_k = bmm(v_t.unsqueeze(-1), k_t.unsqueeze(1))  # [B, N, D]
//     S_new = alpha_t.unsqueeze(-1) * S + (1 - alpha_t.unsqueeze(-1)) * v_outer_k
//
//     # Nonlinear output
//     Sq = bmm(S_new, q_t.unsqueeze(-1)).squeeze(-1) # [B, N]
//     output = tanh(Sq)                              # [B, N]
//
// Cannot parallelize due to S-dependence in retrieval (UTM-class).

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

// Compute S @ k -> Sk (batched matrix-vector multiply)
// S: [B, N, D], k: [B, D] -> Sk: [B, N]
template<typename T>
__global__ void MatVecKernel(
    const int batch_size,
    const int N,   // n_slots
    const int D,   // dim
    const T* __restrict__ S,      // [B, N, D]
    const T* __restrict__ k,      // [B, D]
    T* __restrict__ Sk) {         // [B, N]

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int b = idx / N;
        const int n = idx % N;

        float sum = 0.0f;
        for (int d = 0; d < D; ++d) {
            // S[b, n, d] = S[b * N * D + n * D + d]
            // k[b, d] = k[b * D + d]
            float s_val = static_cast<float>(S[b * N * D + n * D + d]);
            float k_val = static_cast<float>(k[b * D + d]);
            sum += s_val * k_val;
        }
        Sk[idx] = static_cast<T>(sum);
    }
}

// BF16 specialization
__global__ void MatVecKernel_BF16(
    const int batch_size,
    const int N,
    const int D,
    const __nv_bfloat16* __restrict__ S,
    const __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ Sk) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int b = idx / N;
        const int n = idx % N;

        float sum = 0.0f;
        for (int d = 0; d < D; ++d) {
            float s_val = __bfloat162float(S[b * N * D + n * D + d]);
            float k_val = __bfloat162float(k[b * D + d]);
            sum += s_val * k_val;
        }
        Sk[idx] = __float2bfloat16(sum);
    }
}

// Apply tanh to Sk -> retrieved
template<typename T>
__global__ void TanhKernel(
    const int n,
    const T* __restrict__ input,
    T* __restrict__ output) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = static_cast<T>(tanhf(static_cast<float>(input[idx])));
    }
}

__global__ void TanhKernel_BF16(
    const int n,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __float2bfloat16(tanhf(__bfloat162float(input[idx])));
    }
}

// Compute v_t = tanh(Wr_ret + Wx_t + b) and alpha_t = sigmoid(alpha_x_t + b_alpha)
// Wr_ret: [B, N], Wx_t: [B, N], b: [N]
// alpha_x_t: [B, N], b_alpha: [N]
// Output: v_t: [B, N], alpha_t: [B, N]
template<typename T>
__global__ void ValueAlphaKernel(
    const int batch_size,
    const int N,
    const T* __restrict__ Wr_ret,      // [B, N]
    const T* __restrict__ Wx_t,        // [B, N]
    const T* __restrict__ b,           // [N]
    const T* __restrict__ alpha_x_t,   // [B, N]
    const T* __restrict__ b_alpha,     // [N]
    T* __restrict__ v_t,               // [B, N]
    T* __restrict__ alpha_t) {         // [B, N]

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int n = idx % N;

        // v_t = tanh(Wr_ret + Wx_t + b)
        float v_pre = static_cast<float>(Wr_ret[idx]) +
                      static_cast<float>(Wx_t[idx]) +
                      static_cast<float>(b[n]);
        v_t[idx] = static_cast<T>(tanhf(v_pre));

        // alpha_t = sigmoid(alpha_x_t + b_alpha)
        float alpha_pre = static_cast<float>(alpha_x_t[idx]) +
                          static_cast<float>(b_alpha[n]);
        alpha_t[idx] = static_cast<T>(1.0f / (1.0f + expf(-alpha_pre)));
    }
}

__global__ void ValueAlphaKernel_BF16(
    const int batch_size,
    const int N,
    const __nv_bfloat16* __restrict__ Wr_ret,
    const __nv_bfloat16* __restrict__ Wx_t,
    const __nv_bfloat16* __restrict__ b,
    const __nv_bfloat16* __restrict__ alpha_x_t,
    const __nv_bfloat16* __restrict__ b_alpha,
    __nv_bfloat16* __restrict__ v_t,
    __nv_bfloat16* __restrict__ alpha_t) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int n = idx % N;

        float v_pre = __bfloat162float(Wr_ret[idx]) +
                      __bfloat162float(Wx_t[idx]) +
                      __bfloat162float(b[n]);
        v_t[idx] = __float2bfloat16(tanhf(v_pre));

        float alpha_pre = __bfloat162float(alpha_x_t[idx]) +
                          __bfloat162float(b_alpha[n]);
        alpha_t[idx] = __float2bfloat16(1.0f / (1.0f + __expf(-alpha_pre)));
    }
}

// Gated matrix state update:
// S_new[b, n, d] = alpha[b, n] * S_prev[b, n, d] + (1 - alpha[b, n]) * v[b, n] * k[b, d]
// S_prev: [B, N, D], alpha: [B, N], v: [B, N], k: [B, D]
// Output: S_new: [B, N, D]
template<typename T>
__global__ void GatedStateUpdateKernel(
    const int batch_size,
    const int N,
    const int D,
    const T* __restrict__ S_prev,    // [B, N, D]
    const T* __restrict__ alpha,     // [B, N]
    const T* __restrict__ v,         // [B, N]
    const T* __restrict__ k,         // [B, D]
    T* __restrict__ S_new) {         // [B, N, D]

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N * D;

    if (idx < total) {
        const int d = idx % D;
        const int n = (idx / D) % N;
        const int b = idx / (N * D);

        const int bn_idx = b * N + n;
        const int bd_idx = b * D + d;

        float alpha_val = static_cast<float>(alpha[bn_idx]);
        float s_prev_val = static_cast<float>(S_prev[idx]);
        float v_val = static_cast<float>(v[bn_idx]);
        float k_val = static_cast<float>(k[bd_idx]);

        // S_new = alpha * S_prev + (1 - alpha) * v * k
        float s_new_val = alpha_val * s_prev_val + (1.0f - alpha_val) * v_val * k_val;
        S_new[idx] = static_cast<T>(s_new_val);
    }
}

__global__ void GatedStateUpdateKernel_BF16(
    const int batch_size,
    const int N,
    const int D,
    const __nv_bfloat16* __restrict__ S_prev,
    const __nv_bfloat16* __restrict__ alpha,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ S_new) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N * D;

    if (idx < total) {
        const int d = idx % D;
        const int n = (idx / D) % N;
        const int b = idx / (N * D);

        const int bn_idx = b * N + n;
        const int bd_idx = b * D + d;

        float alpha_val = __bfloat162float(alpha[bn_idx]);
        float s_prev_val = __bfloat162float(S_prev[idx]);
        float v_val = __bfloat162float(v[bn_idx]);
        float k_val = __bfloat162float(k[bd_idx]);

        float s_new_val = alpha_val * s_prev_val + (1.0f - alpha_val) * v_val * k_val;
        S_new[idx] = __float2bfloat16(s_new_val);
    }
}

// Compute output = tanh(S @ q) (batched matrix-vector multiply with tanh)
// S: [B, N, D], q: [B, D] -> output: [B, N]
template<typename T>
__global__ void MatVecTanhKernel(
    const int batch_size,
    const int N,
    const int D,
    const T* __restrict__ S,      // [B, N, D]
    const T* __restrict__ q,      // [B, D]
    T* __restrict__ output) {     // [B, N]

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int b = idx / N;
        const int n = idx % N;

        float sum = 0.0f;
        for (int d = 0; d < D; ++d) {
            float s_val = static_cast<float>(S[b * N * D + n * D + d]);
            float q_val = static_cast<float>(q[b * D + d]);
            sum += s_val * q_val;
        }
        output[idx] = static_cast<T>(tanhf(sum));
    }
}

__global__ void MatVecTanhKernel_BF16(
    const int batch_size,
    const int N,
    const int D,
    const __nv_bfloat16* __restrict__ S,
    const __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ output) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int b = idx / N;
        const int n = idx % N;

        float sum = 0.0f;
        for (int d = 0; d < D; ++d) {
            float s_val = __bfloat162float(S[b * N * D + n * D + d]);
            float q_val = __bfloat162float(q[b * D + d]);
            sum += s_val * q_val;
        }
        output[idx] = __float2bfloat16(tanhf(sum));
    }
}

// =============================================================================
// Backward Kernels
// =============================================================================

// Backward through tanh(S @ q) -> d_S_from_out, d_q
// Given d_output [B, N], S [B, N, D], q [B, D], output [B, N] (tanh(S@q))
// Compute: d_Sq = d_output * (1 - output^2)
//          d_S[b,n,d] += d_Sq[b,n] * q[b,d]
//          d_q[b,d] += sum_n(d_Sq[b,n] * S[b,n,d])
template<typename T>
__global__ void OutputBackwardKernel(
    const int batch_size,
    const int N,
    const int D,
    const T* __restrict__ S,          // [B, N, D]
    const T* __restrict__ q,          // [B, D]
    const T* __restrict__ output,     // [B, N] (tanh output)
    const T* __restrict__ d_output,   // [B, N]
    T* __restrict__ d_S,              // [B, N, D] - add to existing
    float* __restrict__ d_q_f) {      // [B, D] float accumulator

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int b = idx / N;
        const int n = idx % N;

        float out_val = static_cast<float>(output[idx]);
        float dout = static_cast<float>(d_output[idx]);
        float d_Sq = dout * (1.0f - out_val * out_val);  // tanh derivative

        // d_S[b,n,d] += d_Sq * q[b,d]
        for (int d = 0; d < D; ++d) {
            float q_val = static_cast<float>(q[b * D + d]);
            float ds_add = d_Sq * q_val;
            int s_idx = b * N * D + n * D + d;
            d_S[s_idx] = static_cast<T>(static_cast<float>(d_S[s_idx]) + ds_add);
        }

        // d_q[b,d] += d_Sq * S[b,n,d]
        for (int d = 0; d < D; ++d) {
            float s_val = static_cast<float>(S[b * N * D + n * D + d]);
            atomicAdd(&d_q_f[b * D + d], d_Sq * s_val);
        }
    }
}

__global__ void OutputBackwardKernel_BF16(
    const int batch_size,
    const int N,
    const int D,
    const __nv_bfloat16* __restrict__ S,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ d_output,
    __nv_bfloat16* __restrict__ d_S,
    float* __restrict__ d_q_f) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int b = idx / N;
        const int n = idx % N;

        float out_val = __bfloat162float(output[idx]);
        float dout = __bfloat162float(d_output[idx]);
        float d_Sq = dout * (1.0f - out_val * out_val);

        for (int d = 0; d < D; ++d) {
            float q_val = __bfloat162float(q[b * D + d]);
            float ds_add = d_Sq * q_val;
            int s_idx = b * N * D + n * D + d;
            d_S[s_idx] = __float2bfloat16(__bfloat162float(d_S[s_idx]) + ds_add);
        }

        for (int d = 0; d < D; ++d) {
            float s_val = __bfloat162float(S[b * N * D + n * D + d]);
            atomicAdd(&d_q_f[b * D + d], d_Sq * s_val);
        }
    }
}

// Backward through gated state update:
// S_new = alpha * S_prev + (1 - alpha) * v @ k^T
// Given d_S_new [B, N, D], compute d_S_prev, d_alpha, d_v, d_k
template<typename T>
__global__ void StateUpdateBackwardKernel(
    const int batch_size,
    const int N,
    const int D,
    const T* __restrict__ S_prev,     // [B, N, D]
    const T* __restrict__ alpha,      // [B, N]
    const T* __restrict__ v,          // [B, N]
    const T* __restrict__ k,          // [B, D]
    const T* __restrict__ d_S_new,    // [B, N, D]
    T* __restrict__ d_S_prev,         // [B, N, D]
    float* __restrict__ d_alpha_f,    // [B, N] float accumulator
    float* __restrict__ d_v_f,        // [B, N] float accumulator
    float* __restrict__ d_k_f) {      // [B, D] float accumulator

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N * D;

    if (idx < total) {
        const int d = idx % D;
        const int n = (idx / D) % N;
        const int b = idx / (N * D);

        const int bn_idx = b * N + n;
        const int bd_idx = b * D + d;

        float alpha_val = static_cast<float>(alpha[bn_idx]);
        float s_prev_val = static_cast<float>(S_prev[idx]);
        float v_val = static_cast<float>(v[bn_idx]);
        float k_val = static_cast<float>(k[bd_idx]);
        float ds_new = static_cast<float>(d_S_new[idx]);

        // d_S_prev = d_S_new * alpha
        d_S_prev[idx] = static_cast<T>(ds_new * alpha_val);

        // d_alpha[b,n] += d_S_new[b,n,d] * (S_prev[b,n,d] - v[b,n] * k[b,d])
        float d_alpha_contrib = ds_new * (s_prev_val - v_val * k_val);
        atomicAdd(&d_alpha_f[bn_idx], d_alpha_contrib);

        // d_v[b,n] += d_S_new[b,n,d] * (1-alpha) * k[b,d]
        float d_v_contrib = ds_new * (1.0f - alpha_val) * k_val;
        atomicAdd(&d_v_f[bn_idx], d_v_contrib);

        // d_k[b,d] += d_S_new[b,n,d] * (1-alpha) * v[b,n]
        float d_k_contrib = ds_new * (1.0f - alpha_val) * v_val;
        atomicAdd(&d_k_f[bd_idx], d_k_contrib);
    }
}

__global__ void StateUpdateBackwardKernel_BF16(
    const int batch_size,
    const int N,
    const int D,
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
    const int total = batch_size * N * D;

    if (idx < total) {
        const int d = idx % D;
        const int n = (idx / D) % N;
        const int b = idx / (N * D);

        const int bn_idx = b * N + n;
        const int bd_idx = b * D + d;

        float alpha_val = __bfloat162float(alpha[bn_idx]);
        float s_prev_val = __bfloat162float(S_prev[idx]);
        float v_val = __bfloat162float(v[bn_idx]);
        float k_val = __bfloat162float(k[bd_idx]);
        float ds_new = __bfloat162float(d_S_new[idx]);

        d_S_prev[idx] = __float2bfloat16(ds_new * alpha_val);

        float d_alpha_contrib = ds_new * (s_prev_val - v_val * k_val);
        atomicAdd(&d_alpha_f[bn_idx], d_alpha_contrib);

        float d_v_contrib = ds_new * (1.0f - alpha_val) * k_val;
        atomicAdd(&d_v_f[bn_idx], d_v_contrib);

        float d_k_contrib = ds_new * (1.0f - alpha_val) * v_val;
        atomicAdd(&d_k_f[bd_idx], d_k_contrib);
    }
}

// Backward through value and alpha computation:
// v = tanh(Wr_ret + Wx + b)
// alpha = sigmoid(alpha_x + b_alpha)
// Given d_v, d_alpha, compute d_Wr_ret (for W_r backward), d_alpha_x (for W_alpha backward)
// and accumulate bias gradients
template<typename T>
__global__ void ValueAlphaBackwardKernel(
    const int batch_size,
    const int N,
    const T* __restrict__ Wr_ret,      // [B, N] (from forward)
    const T* __restrict__ Wx_t,        // [B, N] (from forward)
    const T* __restrict__ b,           // [N]
    const T* __restrict__ alpha_x_t,   // [B, N] (from forward)
    const T* __restrict__ b_alpha,     // [N]
    const float* __restrict__ d_v_f,   // [B, N]
    const float* __restrict__ d_alpha_f, // [B, N]
    T* __restrict__ d_Wr_ret,          // [B, N]
    T* __restrict__ d_alpha_x,         // [B, N]
    float* __restrict__ db_f,          // [N] float accumulator
    float* __restrict__ db_alpha_f) {  // [N] float accumulator

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int n = idx % N;

        // v = tanh(pre_v), d_pre_v = d_v * (1 - tanh^2)
        float pre_v = static_cast<float>(Wr_ret[idx]) +
                      static_cast<float>(Wx_t[idx]) +
                      static_cast<float>(b[n]);
        float v_val = tanhf(pre_v);
        float d_v = d_v_f[idx];
        float d_pre_v = d_v * (1.0f - v_val * v_val);

        d_Wr_ret[idx] = static_cast<T>(d_pre_v);

        // alpha = sigmoid(pre_alpha), d_pre_alpha = d_alpha * sigmoid * (1-sigmoid)
        float pre_alpha = static_cast<float>(alpha_x_t[idx]) +
                          static_cast<float>(b_alpha[n]);
        float alpha_val = 1.0f / (1.0f + expf(-pre_alpha));
        float d_alpha = d_alpha_f[idx];
        float d_pre_alpha = d_alpha * alpha_val * (1.0f - alpha_val);

        d_alpha_x[idx] = static_cast<T>(d_pre_alpha);

        // Bias gradients
        atomicAdd(&db_f[n], d_pre_v);
        atomicAdd(&db_alpha_f[n], d_pre_alpha);
    }
}

__global__ void ValueAlphaBackwardKernel_BF16(
    const int batch_size,
    const int N,
    const __nv_bfloat16* __restrict__ Wr_ret,
    const __nv_bfloat16* __restrict__ Wx_t,
    const __nv_bfloat16* __restrict__ b,
    const __nv_bfloat16* __restrict__ alpha_x_t,
    const __nv_bfloat16* __restrict__ b_alpha,
    const float* __restrict__ d_v_f,
    const float* __restrict__ d_alpha_f,
    __nv_bfloat16* __restrict__ d_Wr_ret,
    __nv_bfloat16* __restrict__ d_alpha_x,
    float* __restrict__ db_f,
    float* __restrict__ db_alpha_f) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int n = idx % N;

        float pre_v = __bfloat162float(Wr_ret[idx]) +
                      __bfloat162float(Wx_t[idx]) +
                      __bfloat162float(b[n]);
        float v_val = tanhf(pre_v);
        float d_v = d_v_f[idx];
        float d_pre_v = d_v * (1.0f - v_val * v_val);

        d_Wr_ret[idx] = __float2bfloat16(d_pre_v);

        float pre_alpha = __bfloat162float(alpha_x_t[idx]) +
                          __bfloat162float(b_alpha[n]);
        float alpha_val = 1.0f / (1.0f + __expf(-pre_alpha));
        float d_alpha = d_alpha_f[idx];
        float d_pre_alpha = d_alpha * alpha_val * (1.0f - alpha_val);

        d_alpha_x[idx] = __float2bfloat16(d_pre_alpha);

        atomicAdd(&db_f[n], d_pre_v);
        atomicAdd(&db_alpha_f[n], d_pre_alpha);
    }
}

// Backward through retrieval: retrieved = tanh(Sk)
// Given d_retrieved from W_r backward, compute d_Sk
// Then from Sk = S @ k, compute d_S_from_ret and d_k_from_ret
template<typename T>
__global__ void RetrievalBackwardKernel(
    const int batch_size,
    const int N,
    const int D,
    const T* __restrict__ S,          // [B, N, D]
    const T* __restrict__ k,          // [B, D]
    const T* __restrict__ Sk,         // [B, N] (pre-tanh)
    const T* __restrict__ d_retrieved, // [B, N]
    T* __restrict__ d_S,              // [B, N, D] - add to existing
    float* __restrict__ d_k_f) {      // [B, D] - add to existing

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int b = idx / N;
        const int n = idx % N;

        // retrieved = tanh(Sk), d_Sk = d_retrieved * (1 - tanh^2)
        float sk_val = static_cast<float>(Sk[idx]);
        float retrieved = tanhf(sk_val);
        float d_ret = static_cast<float>(d_retrieved[idx]);
        float d_Sk = d_ret * (1.0f - retrieved * retrieved);

        // Sk = S @ k, so d_S[b,n,d] += d_Sk * k[d], d_k[d] += d_Sk * S[b,n,d]
        for (int d = 0; d < D; ++d) {
            int s_idx = b * N * D + n * D + d;
            float k_val = static_cast<float>(k[b * D + d]);
            float s_val = static_cast<float>(S[s_idx]);

            // Add to d_S
            d_S[s_idx] = static_cast<T>(static_cast<float>(d_S[s_idx]) + d_Sk * k_val);

            // Add to d_k
            atomicAdd(&d_k_f[b * D + d], d_Sk * s_val);
        }
    }
}

__global__ void RetrievalBackwardKernel_BF16(
    const int batch_size,
    const int N,
    const int D,
    const __nv_bfloat16* __restrict__ S,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ Sk,
    const __nv_bfloat16* __restrict__ d_retrieved,
    __nv_bfloat16* __restrict__ d_S,
    float* __restrict__ d_k_f) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int b = idx / N;
        const int n = idx % N;

        float sk_val = __bfloat162float(Sk[idx]);
        float retrieved = tanhf(sk_val);
        float d_ret = __bfloat162float(d_retrieved[idx]);
        float d_Sk = d_ret * (1.0f - retrieved * retrieved);

        for (int d = 0; d < D; ++d) {
            int s_idx = b * N * D + n * D + d;
            float k_val = __bfloat162float(k[b * D + d]);
            float s_val = __bfloat162float(S[s_idx]);

            d_S[s_idx] = __float2bfloat16(__bfloat162float(d_S[s_idx]) + d_Sk * k_val);
            atomicAdd(&d_k_f[b * D + d], d_Sk * s_val);
        }
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

// Zero a buffer
template<typename T>
__global__ void ZeroKernel(const int n, T* __restrict__ data) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = static_cast<T>(0);
    }
}

}  // anonymous namespace


namespace hasty {
namespace v0 {
namespace elman_ladder {

// =============================================================================
// E63m Matrix Nonlinear Forward - BF16 Specialization
// =============================================================================

template<>
E63mMatrixNonlinearForward<__nv_bfloat16>::E63mMatrixNonlinearForward(
    bool training,
    int batch_size,
    int n_slots,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : training_(training),
      batch_size_(batch_size),
      n_slots_(n_slots),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void E63mMatrixNonlinearForward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_k,       // [D, D]
    const __nv_bfloat16* W_q,       // [D, D]
    const __nv_bfloat16* W_x,       // [N, D]
    const __nv_bfloat16* W_r,       // [N, N]
    const __nv_bfloat16* b,         // [N]
    const __nv_bfloat16* W_alpha,   // [N, D]
    const __nv_bfloat16* b_alpha,   // [N]
    const __nv_bfloat16* x,         // [T, B, D] input
    __nv_bfloat16* S,               // [T+1, B, N, D] state matrices
    __nv_bfloat16* output,          // [T, B, N] output
    __nv_bfloat16* k_cache,         // [T, B, D] for backward
    __nv_bfloat16* q_cache,         // [T, B, D] for backward
    __nv_bfloat16* Wx_cache,        // [T, B, N] for backward
    __nv_bfloat16* alpha_x_cache,   // [T, B, N] for backward
    __nv_bfloat16* Sk_cache,        // [T, B, N] for backward (pre-tanh)
    __nv_bfloat16* retrieved_cache, // [T, B, N] for backward
    __nv_bfloat16* Wr_ret_cache,    // [T, B, N] for backward
    __nv_bfloat16* v_cache,         // [T, B, N] for backward
    __nv_bfloat16* alpha_cache,     // [T, B, N] for backward
    __nv_bfloat16* workspace) {

    static const __nv_bfloat16 alpha_one = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);

    const int BD = batch_size_ * dim_;
    const int BN = batch_size_ * n_slots_;
    const int BND = batch_size_ * n_slots_ * dim_;
    const int block_size = 256;

    // Workspace layout:
    // k_all: [T, B, D]
    // q_all: [T, B, D]
    // Wx_all: [T, B, N]
    // alpha_x_all: [T, B, N]
    // Sk_tmp: [B, N]
    // retrieved_tmp: [B, N]
    // Wr_ret_tmp: [B, N]
    // v_tmp: [B, N]
    // alpha_tmp: [B, N]
    __nv_bfloat16* k_all = workspace;
    __nv_bfloat16* q_all = k_all + steps * BD;
    __nv_bfloat16* Wx_all = q_all + steps * BD;
    __nv_bfloat16* alpha_x_all = Wx_all + steps * BN;
    __nv_bfloat16* Sk_tmp = alpha_x_all + steps * BN;
    __nv_bfloat16* retrieved_tmp = Sk_tmp + BN;
    __nv_bfloat16* Wr_ret_tmp = retrieved_tmp + BN;
    __nv_bfloat16* v_tmp = Wr_ret_tmp + BN;
    __nv_bfloat16* alpha_tmp = v_tmp + BN;

    // Pre-compute all x projections in batched GEMMs
    // k_all = x @ W_k.T  [T*B, D]
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_k, dim_,
        x, dim_,
        &beta_zero,
        k_all, dim_);

    // q_all = x @ W_q.T  [T*B, D]
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_q, dim_,
        x, dim_,
        &beta_zero,
        q_all, dim_);

    // Wx_all = x @ W_x.T  [T*B, N]
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        n_slots_, steps * batch_size_, dim_,
        &alpha_one,
        W_x, dim_,
        x, dim_,
        &beta_zero,
        Wx_all, n_slots_);

    // alpha_x_all = x @ W_alpha.T  [T*B, N]
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        n_slots_, steps * batch_size_, dim_,
        &alpha_one,
        W_alpha, dim_,
        x, dim_,
        &beta_zero,
        alpha_x_all, n_slots_);

    // Process each timestep sequentially (cannot parallelize - S-dependent retrieval)
    for (int t = 0; t < steps; ++t) {
        const __nv_bfloat16* k_t = k_all + t * BD;
        const __nv_bfloat16* q_t = q_all + t * BD;
        const __nv_bfloat16* Wx_t = Wx_all + t * BN;
        const __nv_bfloat16* alpha_x_t = alpha_x_all + t * BN;
        const __nv_bfloat16* S_prev = S + t * BND;
        __nv_bfloat16* S_t = S + (t + 1) * BND;
        __nv_bfloat16* out_t = output + t * BN;

        // Cache pointers for backward
        __nv_bfloat16* k_c = training_ ? (k_cache + t * BD) : const_cast<__nv_bfloat16*>(k_t);
        __nv_bfloat16* q_c = training_ ? (q_cache + t * BD) : const_cast<__nv_bfloat16*>(q_t);
        __nv_bfloat16* Wx_c = training_ ? (Wx_cache + t * BN) : const_cast<__nv_bfloat16*>(Wx_t);
        __nv_bfloat16* alpha_x_c = training_ ? (alpha_x_cache + t * BN) : const_cast<__nv_bfloat16*>(alpha_x_t);
        __nv_bfloat16* Sk_c = training_ ? (Sk_cache + t * BN) : Sk_tmp;
        __nv_bfloat16* retrieved_c = training_ ? (retrieved_cache + t * BN) : retrieved_tmp;
        __nv_bfloat16* Wr_ret_c = training_ ? (Wr_ret_cache + t * BN) : Wr_ret_tmp;
        __nv_bfloat16* v_c = training_ ? (v_cache + t * BN) : v_tmp;
        __nv_bfloat16* alpha_c = training_ ? (alpha_cache + t * BN) : alpha_tmp;

        // Copy k, q, Wx, alpha_x to cache if training
        if (training_) {
            cudaMemcpyAsync(k_c, k_t, BD * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream_);
            cudaMemcpyAsync(q_c, q_t, BD * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream_);
            cudaMemcpyAsync(Wx_c, Wx_t, BN * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream_);
            cudaMemcpyAsync(alpha_x_c, alpha_x_t, BN * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream_);
        }

        // 1. Nonlinear retrieval: Sk = S @ k, retrieved = tanh(Sk)
        MatVecKernel_BF16<<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_slots_, dim_, S_prev, k_t, Sk_c);

        TanhKernel_BF16<<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            BN, Sk_c, retrieved_c);

        // 2. Wr_ret = retrieved @ W_r.T  [B, N]
        blas<__nv_bfloat16>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            n_slots_, batch_size_, n_slots_,
            &alpha_one,
            W_r, n_slots_,
            retrieved_c, n_slots_,
            &beta_zero,
            Wr_ret_c, n_slots_);

        // 3. Compute v = tanh(Wr_ret + Wx + b), alpha = sigmoid(alpha_x + b_alpha)
        ValueAlphaKernel_BF16<<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_slots_, Wr_ret_c, Wx_t, b, alpha_x_t, b_alpha, v_c, alpha_c);

        // 4. Gated state update: S_new = alpha * S_prev + (1-alpha) * v @ k^T
        GatedStateUpdateKernel_BF16<<<(BND + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_slots_, dim_, S_prev, alpha_c, v_c, k_t, S_t);

        // 5. Nonlinear output: output = tanh(S_new @ q)
        MatVecTanhKernel_BF16<<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_slots_, dim_, S_t, q_t, out_t);
    }
}

// =============================================================================
// E63m Matrix Nonlinear Backward - BF16 Specialization
// =============================================================================

template<>
E63mMatrixNonlinearBackward<__nv_bfloat16>::E63mMatrixNonlinearBackward(
    int batch_size,
    int n_slots,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      n_slots_(n_slots),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void E63mMatrixNonlinearBackward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_k,
    const __nv_bfloat16* W_q,
    const __nv_bfloat16* W_x,
    const __nv_bfloat16* W_r,
    const __nv_bfloat16* b,
    const __nv_bfloat16* W_alpha,
    const __nv_bfloat16* b_alpha,
    const __nv_bfloat16* x,
    const __nv_bfloat16* S,
    const __nv_bfloat16* output,
    const __nv_bfloat16* k_cache,
    const __nv_bfloat16* q_cache,
    const __nv_bfloat16* Wx_cache,
    const __nv_bfloat16* alpha_x_cache,
    const __nv_bfloat16* Sk_cache,
    const __nv_bfloat16* retrieved_cache,
    const __nv_bfloat16* Wr_ret_cache,
    const __nv_bfloat16* v_cache,
    const __nv_bfloat16* alpha_cache,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* dx,
    __nv_bfloat16* dW_k,
    __nv_bfloat16* dW_q,
    __nv_bfloat16* dW_x,
    __nv_bfloat16* dW_r,
    __nv_bfloat16* db,
    __nv_bfloat16* dW_alpha,
    __nv_bfloat16* db_alpha,
    __nv_bfloat16* workspace) {

    static const __nv_bfloat16 alpha_one = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);
    static const __nv_bfloat16 beta_one = __float2bfloat16(1.0f);

    const int BD = batch_size_ * dim_;
    const int BN = batch_size_ * n_slots_;
    const int BND = batch_size_ * n_slots_ * dim_;
    const int block_size = 256;

    // Workspace layout:
    // d_S: [B, N, D] - current state gradient (input)
    // d_S_tmp: [B, N, D] - state gradient (output) - need two buffers to avoid race
    // d_k_all: [T, B, D] - accumulated k gradients
    // d_q_all: [T, B, D] - accumulated q gradients
    // d_Wx_all: [T, B, N] - accumulated Wx gradients
    // d_alpha_x_all: [T, B, N] - accumulated alpha_x gradients
    // d_retrieved: [B, N]
    // d_Wr_ret: [B, N]
    // d_v_f: [B, N] float
    // d_alpha_f: [B, N] float
    // d_k_f: [B, D] float
    // d_q_f: [B, D] float
    // db_f: [N] float
    // db_alpha_f: [N] float
    __nv_bfloat16* d_S = workspace;
    __nv_bfloat16* d_S_tmp = d_S + BND;  // Separate buffer for state gradient output
    __nv_bfloat16* d_k_all = d_S_tmp + BND;
    __nv_bfloat16* d_q_all = d_k_all + steps * BD;
    __nv_bfloat16* d_Wx_all = d_q_all + steps * BD;
    __nv_bfloat16* d_alpha_x_all = d_Wx_all + steps * BN;
    __nv_bfloat16* d_retrieved = d_alpha_x_all + steps * BN;
    __nv_bfloat16* d_Wr_ret = d_retrieved + BN;

    float* float_ws = reinterpret_cast<float*>(d_Wr_ret + BN);
    float* d_v_f = float_ws;
    float* d_alpha_f = d_v_f + BN;
    float* d_k_f = d_alpha_f + BN;
    float* d_q_f = d_k_f + BD;
    float* db_f = d_q_f + BD;
    float* db_alpha_f = db_f + n_slots_;

    // Initialize gradients to zero
    cudaMemsetAsync(d_S, 0, BND * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_k, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_q, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_x, 0, n_slots_ * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_r, 0, n_slots_ * n_slots_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_alpha, 0, n_slots_ * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(db_f, 0, n_slots_ * sizeof(float), stream_);
    cudaMemsetAsync(db_alpha_f, 0, n_slots_ * sizeof(float), stream_);

    // BPTT loop
    for (int t = steps - 1; t >= 0; --t) {
        const __nv_bfloat16* S_t = S + (t + 1) * BND;
        const __nv_bfloat16* S_prev = S + t * BND;
        const __nv_bfloat16* k_t = k_cache + t * BD;
        const __nv_bfloat16* q_t = q_cache + t * BD;
        const __nv_bfloat16* Wx_t = Wx_cache + t * BN;
        const __nv_bfloat16* alpha_x_t = alpha_x_cache + t * BN;
        const __nv_bfloat16* Sk_t = Sk_cache + t * BN;
        const __nv_bfloat16* retrieved_t = retrieved_cache + t * BN;
        const __nv_bfloat16* Wr_ret_t = Wr_ret_cache + t * BN;
        const __nv_bfloat16* v_t = v_cache + t * BN;
        const __nv_bfloat16* alpha_t = alpha_cache + t * BN;
        const __nv_bfloat16* d_out_t = d_output + t * BN;
        const __nv_bfloat16* out_t = output + t * BN;

        __nv_bfloat16* d_k_t = d_k_all + t * BD;
        __nv_bfloat16* d_q_t = d_q_all + t * BD;
        __nv_bfloat16* d_Wx_t = d_Wx_all + t * BN;
        __nv_bfloat16* d_alpha_x_t = d_alpha_x_all + t * BN;

        // Zero per-timestep accumulators
        cudaMemsetAsync(d_v_f, 0, BN * sizeof(float), stream_);
        cudaMemsetAsync(d_alpha_f, 0, BN * sizeof(float), stream_);
        cudaMemsetAsync(d_k_f, 0, BD * sizeof(float), stream_);
        cudaMemsetAsync(d_q_f, 0, BD * sizeof(float), stream_);

        // 1. Backward through output: output = tanh(S @ q)
        // Adds gradient to d_S (grad w.r.t. S[t+1])
        OutputBackwardKernel_BF16<<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_slots_, dim_, S_t, q_t, out_t, d_out_t, d_S, d_q_f);

        // 2. Backward through gated state update
        // Reads from d_S (grad w.r.t. S[t+1]), writes to d_S_tmp (grad w.r.t. S[t])
        // IMPORTANT: d_S_tmp must be different from d_S to avoid race condition!
        StateUpdateBackwardKernel_BF16<<<(BND + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_slots_, dim_, S_prev, alpha_t, v_t, k_t, d_S,
            d_S_tmp, d_alpha_f, d_v_f, d_k_f);  // Output to separate buffer

        // 3. Backward through value/alpha computation
        ValueAlphaBackwardKernel_BF16<<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_slots_, Wr_ret_t, Wx_t, b, alpha_x_t, b_alpha,
            d_v_f, d_alpha_f, d_Wr_ret, d_alpha_x_t, db_f, db_alpha_f);

        // 4. Backward through W_r projection: Wr_ret = retrieved @ W_r.T
        // d_retrieved = d_Wr_ret @ W_r
        blas<__nv_bfloat16>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n_slots_, batch_size_, n_slots_,
            &alpha_one,
            W_r, n_slots_,
            d_Wr_ret, n_slots_,
            &beta_zero,
            d_retrieved, n_slots_);

        // dW_r += retrieved.T @ d_Wr_ret
        blas<__nv_bfloat16>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_T,
            n_slots_, n_slots_, batch_size_,
            &alpha_one,
            retrieved_t, n_slots_,
            d_Wr_ret, n_slots_,
            &beta_one,
            dW_r, n_slots_);

        // 5. Backward through retrieval: retrieved = tanh(Sk), Sk = S_prev @ k
        // Adds gradient to d_S_tmp (grad w.r.t. S[t] from retrieval path)
        RetrievalBackwardKernel_BF16<<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_slots_, dim_, S_prev, k_t, Sk_t, d_retrieved, d_S_tmp, d_k_f);

        // d_Wx = d_pre_v (already computed in ValueAlphaBackward)
        cudaMemcpyAsync(d_Wx_t, d_Wr_ret, BN * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream_);

        // Copy float accumulators to output
        CopyFloatToT<__nv_bfloat16><<<(BD + block_size - 1) / block_size, block_size, 0, stream_>>>(
            BD, d_k_f, d_k_t);
        CopyFloatToT<__nv_bfloat16><<<(BD + block_size - 1) / block_size, block_size, 0, stream_>>>(
            BD, d_q_f, d_q_t);

        // Swap d_S and d_S_tmp for next iteration
        std::swap(d_S, d_S_tmp);
    }

    // Batch compute weight gradients from accumulated projection gradients
    // dx = d_k_all @ W_k + d_q_all @ W_q + d_Wx_all @ W_x + d_alpha_x_all @ W_alpha

    // dx from k projection: dx = d_k_all @ W_k
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_k, dim_,
        d_k_all, dim_,
        &beta_zero,
        dx, dim_);

    // dx += d_q_all @ W_q
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_q, dim_,
        d_q_all, dim_,
        &beta_one,
        dx, dim_);

    // dx += d_Wx_all @ W_x
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, n_slots_,
        &alpha_one,
        W_x, dim_,
        d_Wx_all, n_slots_,
        &beta_one,
        dx, dim_);

    // dx += d_alpha_x_all @ W_alpha
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, n_slots_,
        &alpha_one,
        W_alpha, dim_,
        d_alpha_x_all, n_slots_,
        &beta_one,
        dx, dim_);

    // dW_k = x.T @ d_k_all
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        d_k_all, dim_,
        &beta_one,
        dW_k, dim_);

    // dW_q = x.T @ d_q_all
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        d_q_all, dim_,
        &beta_one,
        dW_q, dim_);

    // dW_x = d_Wx_all.T @ x = [N, T*B] @ [T*B, D] = [N, D]
    // W_x is [N, D], so dW_x must be [N, D]
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, n_slots_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        d_Wx_all, n_slots_,
        &beta_one,
        dW_x, dim_);

    // dW_alpha = x.T @ d_alpha_x_all
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, n_slots_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        d_alpha_x_all, n_slots_,
        &beta_one,
        dW_alpha, dim_);

    // Copy bias gradients
    CopyFloatToT<__nv_bfloat16><<<(n_slots_ + 255) / 256, 256, 0, stream_>>>(n_slots_, db_f, db);
    CopyFloatToT<__nv_bfloat16><<<(n_slots_ + 255) / 256, 256, 0, stream_>>>(n_slots_, db_alpha_f, db_alpha);
}

// =============================================================================
// Generic Template Implementations
// =============================================================================

template<typename T>
E63mMatrixNonlinearForward<T>::E63mMatrixNonlinearForward(
    bool training,
    int batch_size,
    int n_slots,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : training_(training),
      batch_size_(batch_size),
      n_slots_(n_slots),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void E63mMatrixNonlinearForward<T>::Run(
    int steps,
    const T* W_k,
    const T* W_q,
    const T* W_x,
    const T* W_r,
    const T* b,
    const T* W_alpha,
    const T* b_alpha,
    const T* x,
    T* S,
    T* output,
    T* k_cache,
    T* q_cache,
    T* Wx_cache,
    T* alpha_x_cache,
    T* Sk_cache,
    T* retrieved_cache,
    T* Wr_ret_cache,
    T* v_cache,
    T* alpha_cache,
    T* workspace) {

    // Generic implementation follows same pattern as BF16
    static const T alpha_one = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int BD = batch_size_ * dim_;
    const int BN = batch_size_ * n_slots_;
    const int BND = batch_size_ * n_slots_ * dim_;
    const int block_size = 256;

    T* k_all = workspace;
    T* q_all = k_all + steps * BD;
    T* Wx_all = q_all + steps * BD;
    T* alpha_x_all = Wx_all + steps * BN;
    T* Sk_tmp = alpha_x_all + steps * BN;
    T* retrieved_tmp = Sk_tmp + BN;
    T* Wr_ret_tmp = retrieved_tmp + BN;
    T* v_tmp = Wr_ret_tmp + BN;
    T* alpha_tmp = v_tmp + BN;

    // Pre-compute projections
    blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_, &alpha_one, W_k, dim_, x, dim_, &beta_zero, k_all, dim_);
    blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_, &alpha_one, W_q, dim_, x, dim_, &beta_zero, q_all, dim_);
    blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        n_slots_, steps * batch_size_, dim_, &alpha_one, W_x, dim_, x, dim_, &beta_zero, Wx_all, n_slots_);
    blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        n_slots_, steps * batch_size_, dim_, &alpha_one, W_alpha, dim_, x, dim_, &beta_zero, alpha_x_all, n_slots_);

    for (int t = 0; t < steps; ++t) {
        const T* k_t = k_all + t * BD;
        const T* q_t = q_all + t * BD;
        const T* Wx_t = Wx_all + t * BN;
        const T* alpha_x_t = alpha_x_all + t * BN;
        const T* S_prev = S + t * BND;
        T* S_t = S + (t + 1) * BND;
        T* out_t = output + t * BN;

        T* Sk_c = training_ ? (Sk_cache + t * BN) : Sk_tmp;
        T* retrieved_c = training_ ? (retrieved_cache + t * BN) : retrieved_tmp;
        T* Wr_ret_c = training_ ? (Wr_ret_cache + t * BN) : Wr_ret_tmp;
        T* v_c = training_ ? (v_cache + t * BN) : v_tmp;
        T* alpha_c = training_ ? (alpha_cache + t * BN) : alpha_tmp;

        if (training_) {
            cudaMemcpyAsync(k_cache + t * BD, k_t, BD * sizeof(T), cudaMemcpyDeviceToDevice, stream_);
            cudaMemcpyAsync(q_cache + t * BD, q_t, BD * sizeof(T), cudaMemcpyDeviceToDevice, stream_);
            cudaMemcpyAsync(Wx_cache + t * BN, Wx_t, BN * sizeof(T), cudaMemcpyDeviceToDevice, stream_);
            cudaMemcpyAsync(alpha_x_cache + t * BN, alpha_x_t, BN * sizeof(T), cudaMemcpyDeviceToDevice, stream_);
        }

        MatVecKernel<T><<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_slots_, dim_, S_prev, k_t, Sk_c);
        TanhKernel<T><<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            BN, Sk_c, retrieved_c);

        blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            n_slots_, batch_size_, n_slots_, &alpha_one, W_r, n_slots_, retrieved_c, n_slots_, &beta_zero, Wr_ret_c, n_slots_);

        ValueAlphaKernel<T><<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_slots_, Wr_ret_c, Wx_t, b, alpha_x_t, b_alpha, v_c, alpha_c);

        GatedStateUpdateKernel<T><<<(BND + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_slots_, dim_, S_prev, alpha_c, v_c, k_t, S_t);

        MatVecTanhKernel<T><<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_slots_, dim_, S_t, q_t, out_t);
    }
}

template<typename T>
E63mMatrixNonlinearBackward<T>::E63mMatrixNonlinearBackward(
    int batch_size,
    int n_slots,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      n_slots_(n_slots),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void E63mMatrixNonlinearBackward<T>::Run(
    int steps,
    const T* W_k,
    const T* W_q,
    const T* W_x,
    const T* W_r,
    const T* b,
    const T* W_alpha,
    const T* b_alpha,
    const T* x,
    const T* S,
    const T* output,
    const T* k_cache,
    const T* q_cache,
    const T* Wx_cache,
    const T* alpha_x_cache,
    const T* Sk_cache,
    const T* retrieved_cache,
    const T* Wr_ret_cache,
    const T* v_cache,
    const T* alpha_cache,
    const T* d_output,
    T* dx,
    T* dW_k,
    T* dW_q,
    T* dW_x,
    T* dW_r,
    T* db,
    T* dW_alpha,
    T* db_alpha,
    T* workspace) {

    // Placeholder - follows BF16 pattern
    cudaMemsetAsync(dx, 0, steps * batch_size_ * dim_ * sizeof(T), stream_);
}

// Explicit template instantiations
template struct E63mMatrixNonlinearForward<__half>;
template struct E63mMatrixNonlinearForward<float>;
template struct E63mMatrixNonlinearForward<double>;

template struct E63mMatrixNonlinearBackward<__half>;
template struct E63mMatrixNonlinearBackward<float>;
template struct E63mMatrixNonlinearBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
