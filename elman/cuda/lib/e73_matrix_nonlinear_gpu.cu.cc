// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E73: Matrix Nonlinear Elman - E1-style with S inside tanh
//
// Architecture:
//     k_t = W_k @ x_t                              # [B, n] key
//     v_t = W_v @ x_t                              # [B, n] value
//     q_t = W_q @ x_t                              # [B, n] query
//     z_t = tanh(W_z @ x_t + b_z)                  # [B, n] bounded modulation (-1, 1)
//
//     # Column modulation (default variant):
//     S_modulated = S * z.unsqueeze(1)             # Scale each column by z
//     S = tanh(S_modulated + outer(v, k))          # [B, n, n]
//
//     out = S @ q                                   # [B, n]
//     out = out * silu(out)                        # Self-gate output
//
// State S is [B, n, n] square matrix.
//
// Variants:
// - 'column': S[i,j] *= z[j] (scale each column by z[j])
// - 'row':    S[i,j] *= z[i] (scale each row by z[i])
// - 'full':   S[i,j] *= z[i] * z[j] (outer product scaling)
//
// Key optimization: Batch projections for k, v, q, z upfront (4 GEMMs)

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

// Apply z modulation to S, add outer(v, k), apply tanh
// S_new[b, i, j] = tanh(S_prev[b, i, j] * z_mod + v[b, i] * k[b, j])
// variant: 0 = column (z[j]), 1 = row (z[i]), 2 = full (z[i]*z[j])
template<typename T>
__global__ void E73UpdateKernel(
    const int batch_size,
    const int n,        // State dimension (n x n matrix)
    const int variant,  // 0=column, 1=row, 2=full
    const T* __restrict__ S_prev,    // [B, n, n]
    const T* __restrict__ z,         // [B, n] modulation gate (already sigmoid)
    const T* __restrict__ v,         // [B, n] value
    const T* __restrict__ k,         // [B, n] key
    T* __restrict__ S_new,           // [B, n, n]
    T* __restrict__ pre_tanh_cache)  // [B, n, n] cache for backward (optional)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * n * n;

    if (idx < total) {
        const int j = idx % n;
        const int i = (idx / n) % n;
        const int b = idx / (n * n);

        const int bn = b * n;
        float s_prev = static_cast<float>(S_prev[idx]);
        float z_i = static_cast<float>(z[bn + i]);
        float z_j = static_cast<float>(z[bn + j]);
        float v_i = static_cast<float>(v[bn + i]);
        float k_j = static_cast<float>(k[bn + j]);

        // Compute z modulation based on variant
        float z_mod;
        if (variant == 0) {
            // Column: scale column j by z[j]
            z_mod = z_j;
        } else if (variant == 1) {
            // Row: scale row i by z[i]
            z_mod = z_i;
        } else {
            // Full: scale by z[i] * z[j]
            z_mod = z_i * z_j;
        }

        // S_new = tanh(S_prev * z_mod + v[i] * k[j])
        float pre_tanh = s_prev * z_mod + v_i * k_j;
        float s_new = tanhf(pre_tanh);

        S_new[idx] = static_cast<T>(s_new);
        if (pre_tanh_cache) {
            pre_tanh_cache[idx] = static_cast<T>(pre_tanh);
        }
    }
}

// BF16 specialization
__global__ void E73UpdateKernel_BF16(
    const int batch_size,
    const int n,
    const int variant,
    const __nv_bfloat16* __restrict__ S_prev,
    const __nv_bfloat16* __restrict__ z,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ S_new,
    __nv_bfloat16* __restrict__ pre_tanh_cache)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * n * n;

    if (idx < total) {
        const int j = idx % n;
        const int i = (idx / n) % n;
        const int b = idx / (n * n);

        const int bn = b * n;
        float s_prev = __bfloat162float(S_prev[idx]);
        float z_i = __bfloat162float(z[bn + i]);
        float z_j = __bfloat162float(z[bn + j]);
        float v_i = __bfloat162float(v[bn + i]);
        float k_j = __bfloat162float(k[bn + j]);

        float z_mod;
        if (variant == 0) {
            z_mod = z_j;
        } else if (variant == 1) {
            z_mod = z_i;
        } else {
            z_mod = z_i * z_j;
        }

        float pre_tanh = s_prev * z_mod + v_i * k_j;
        float s_new = tanhf(pre_tanh);

        S_new[idx] = __float2bfloat16(s_new);
        if (pre_tanh_cache) {
            pre_tanh_cache[idx] = __float2bfloat16(pre_tanh);
        }
    }
}

// Compute output: out = S @ q, then self-gate: out = out * silu(out)
// S: [B, n, n], q: [B, n] -> out: [B, n]
template<typename T>
__global__ void E73OutputKernel(
    const int batch_size,
    const int n,
    const T* __restrict__ S,      // [B, n, n]
    const T* __restrict__ q,      // [B, n]
    T* __restrict__ output,       // [B, n]
    T* __restrict__ Sq_cache)     // [B, n] cache pre-self-gate (optional)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * n;

    if (idx < total) {
        const int i = idx % n;
        const int b = idx / n;

        // Compute S[b, i, :] @ q[b, :]
        float sum = 0.0f;
        for (int j = 0; j < n; ++j) {
            float s_val = static_cast<float>(S[b * n * n + i * n + j]);
            float q_val = static_cast<float>(q[b * n + j]);
            sum += s_val * q_val;
        }

        // Cache Sq before self-gate
        if (Sq_cache) {
            Sq_cache[idx] = static_cast<T>(sum);
        }

        // Self-gate: out = Sq * silu(Sq)
        float sigmoid_sq = 1.0f / (1.0f + expf(-sum));
        float silu_sq = sum * sigmoid_sq;
        float out = sum * silu_sq;

        output[idx] = static_cast<T>(out);
    }
}

__global__ void E73OutputKernel_BF16(
    const int batch_size,
    const int n,
    const __nv_bfloat16* __restrict__ S,
    const __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ Sq_cache)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * n;

    if (idx < total) {
        const int i = idx % n;
        const int b = idx / n;

        float sum = 0.0f;
        for (int j = 0; j < n; ++j) {
            float s_val = __bfloat162float(S[b * n * n + i * n + j]);
            float q_val = __bfloat162float(q[b * n + j]);
            sum += s_val * q_val;
        }

        if (Sq_cache) {
            Sq_cache[idx] = __float2bfloat16(sum);
        }

        float sigmoid_sq = 1.0f / (1.0f + __expf(-sum));
        float silu_sq = sum * sigmoid_sq;
        float out = sum * silu_sq;

        output[idx] = __float2bfloat16(out);
    }
}

// =============================================================================
// Backward Kernels
// =============================================================================

// Backward through self-gate output: out = Sq * silu(Sq) = Sq^2 * sigmoid(Sq)
// d(out)/d(Sq) = silu(Sq) * (2 + Sq*(1-sigmoid(Sq)))
template<typename T>
__global__ void E73SelfGateBackwardKernel(
    const int batch_size,
    const int n,
    const T* __restrict__ Sq,        // [B, n]
    const T* __restrict__ d_output,  // [B, n]
    T* __restrict__ d_Sq)            // [B, n]
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * n;

    if (idx < total) {
        float sq_val = static_cast<float>(Sq[idx]);
        float dout = static_cast<float>(d_output[idx]);

        float sigmoid_sq = 1.0f / (1.0f + expf(-sq_val));
        float silu_sq = sq_val * sigmoid_sq;
        float grad_factor = silu_sq * (2.0f + sq_val * (1.0f - sigmoid_sq));

        d_Sq[idx] = static_cast<T>(dout * grad_factor);
    }
}

__global__ void E73SelfGateBackwardKernel_BF16(
    const int batch_size,
    const int n,
    const __nv_bfloat16* __restrict__ Sq,
    const __nv_bfloat16* __restrict__ d_output,
    __nv_bfloat16* __restrict__ d_Sq)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * n;

    if (idx < total) {
        float sq_val = __bfloat162float(Sq[idx]);
        float dout = __bfloat162float(d_output[idx]);

        float sigmoid_sq = 1.0f / (1.0f + __expf(-sq_val));
        float silu_sq = sq_val * sigmoid_sq;
        float grad_factor = silu_sq * (2.0f + sq_val * (1.0f - sigmoid_sq));

        d_Sq[idx] = __float2bfloat16(dout * grad_factor);
    }
}

// Backward through output: Sq = S @ q
// Given d_Sq [B, n], S [B, n, n], q [B, n]
// Compute: d_S[b, i, j] = d_Sq[b, i] * q[b, j]
//          d_q[b, j] = sum_i(d_Sq[b, i] * S[b, i, j])
template<typename T>
__global__ void E73OutputBackwardKernel(
    const int batch_size,
    const int n,
    const T* __restrict__ S,         // [B, n, n]
    const T* __restrict__ q,         // [B, n]
    const T* __restrict__ d_Sq,      // [B, n]
    T* __restrict__ d_S,             // [B, n, n] - add to existing
    float* __restrict__ d_q_f)       // [B, n] float accumulator
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * n;

    if (idx < total) {
        const int i = idx % n;
        const int b = idx / n;

        float d_sq_i = static_cast<float>(d_Sq[idx]);

        for (int j = 0; j < n; ++j) {
            int s_idx = b * n * n + i * n + j;
            float q_j = static_cast<float>(q[b * n + j]);
            float s_ij = static_cast<float>(S[s_idx]);

            // d_S[b, i, j] += d_Sq[b, i] * q[b, j]
            d_S[s_idx] = static_cast<T>(static_cast<float>(d_S[s_idx]) + d_sq_i * q_j);

            // d_q[b, j] += d_Sq[b, i] * S[b, i, j]
            atomicAdd(&d_q_f[b * n + j], d_sq_i * s_ij);
        }
    }
}

__global__ void E73OutputBackwardKernel_BF16(
    const int batch_size,
    const int n,
    const __nv_bfloat16* __restrict__ S,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ d_Sq,
    __nv_bfloat16* __restrict__ d_S,
    float* __restrict__ d_q_f)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * n;

    if (idx < total) {
        const int i = idx % n;
        const int b = idx / n;

        float d_sq_i = __bfloat162float(d_Sq[idx]);

        for (int j = 0; j < n; ++j) {
            int s_idx = b * n * n + i * n + j;
            float q_j = __bfloat162float(q[b * n + j]);
            float s_ij = __bfloat162float(S[s_idx]);

            d_S[s_idx] = __float2bfloat16(__bfloat162float(d_S[s_idx]) + d_sq_i * q_j);
            atomicAdd(&d_q_f[b * n + j], d_sq_i * s_ij);
        }
    }
}

// Backward through update: S_new = tanh(S_prev * z_mod + v @ k^T)
// Given d_S_new [B, n, n], compute d_S_prev, d_z, d_v, d_k
// pre_tanh is S_prev * z_mod + outer(v, k), needed for tanh derivative
template<typename T>
__global__ void E73UpdateBackwardKernel(
    const int batch_size,
    const int n,
    const int variant,
    const T* __restrict__ S_prev,      // [B, n, n]
    const T* __restrict__ z,           // [B, n]
    const T* __restrict__ v,           // [B, n]
    const T* __restrict__ k,           // [B, n]
    const T* __restrict__ pre_tanh,    // [B, n, n]
    const T* __restrict__ d_S_new,     // [B, n, n]
    T* __restrict__ d_S_prev,          // [B, n, n]
    float* __restrict__ d_z_f,         // [B, n] float accumulator
    float* __restrict__ d_v_f,         // [B, n] float accumulator
    float* __restrict__ d_k_f)         // [B, n] float accumulator
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * n * n;

    if (idx < total) {
        const int j = idx % n;
        const int i = (idx / n) % n;
        const int b = idx / (n * n);

        const int bn = b * n;
        float s_prev = static_cast<float>(S_prev[idx]);
        float z_i = static_cast<float>(z[bn + i]);
        float z_j = static_cast<float>(z[bn + j]);
        float v_i = static_cast<float>(v[bn + i]);
        float k_j = static_cast<float>(k[bn + j]);
        float pt = static_cast<float>(pre_tanh[idx]);
        float ds_new = static_cast<float>(d_S_new[idx]);

        // tanh derivative: d_pre_tanh = d_S_new * (1 - tanh^2(pre_tanh))
        float tanh_pt = tanhf(pt);
        float d_pt = ds_new * (1.0f - tanh_pt * tanh_pt);

        // Compute z_mod and gradients based on variant
        float z_mod;
        if (variant == 0) {
            z_mod = z_j;
            // d_S_prev = d_pt * z_j
            d_S_prev[idx] = static_cast<T>(d_pt * z_j);
            // d_z[j] += d_pt * S_prev[i,j]
            atomicAdd(&d_z_f[bn + j], d_pt * s_prev);
        } else if (variant == 1) {
            z_mod = z_i;
            // d_S_prev = d_pt * z_i
            d_S_prev[idx] = static_cast<T>(d_pt * z_i);
            // d_z[i] += d_pt * S_prev[i,j]
            atomicAdd(&d_z_f[bn + i], d_pt * s_prev);
        } else {
            z_mod = z_i * z_j;
            // d_S_prev = d_pt * z_i * z_j
            d_S_prev[idx] = static_cast<T>(d_pt * z_i * z_j);
            // d_z[i] += d_pt * S_prev * z_j
            atomicAdd(&d_z_f[bn + i], d_pt * s_prev * z_j);
            // d_z[j] += d_pt * S_prev * z_i
            atomicAdd(&d_z_f[bn + j], d_pt * s_prev * z_i);
        }

        // d_v[i] += d_pt * k[j]
        atomicAdd(&d_v_f[bn + i], d_pt * k_j);

        // d_k[j] += d_pt * v[i]
        atomicAdd(&d_k_f[bn + j], d_pt * v_i);
    }
}

__global__ void E73UpdateBackwardKernel_BF16(
    const int batch_size,
    const int n,
    const int variant,
    const __nv_bfloat16* __restrict__ S_prev,
    const __nv_bfloat16* __restrict__ z,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ pre_tanh,
    const __nv_bfloat16* __restrict__ d_S_new,
    __nv_bfloat16* __restrict__ d_S_prev,
    float* __restrict__ d_z_f,
    float* __restrict__ d_v_f,
    float* __restrict__ d_k_f)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * n * n;

    if (idx < total) {
        const int j = idx % n;
        const int i = (idx / n) % n;
        const int b = idx / (n * n);

        const int bn = b * n;
        float s_prev = __bfloat162float(S_prev[idx]);
        float z_i = __bfloat162float(z[bn + i]);
        float z_j = __bfloat162float(z[bn + j]);
        float v_i = __bfloat162float(v[bn + i]);
        float k_j = __bfloat162float(k[bn + j]);
        float pt = __bfloat162float(pre_tanh[idx]);
        float ds_new = __bfloat162float(d_S_new[idx]);

        float tanh_pt = tanhf(pt);
        float d_pt = ds_new * (1.0f - tanh_pt * tanh_pt);

        if (variant == 0) {
            d_S_prev[idx] = __float2bfloat16(d_pt * z_j);
            atomicAdd(&d_z_f[bn + j], d_pt * s_prev);
        } else if (variant == 1) {
            d_S_prev[idx] = __float2bfloat16(d_pt * z_i);
            atomicAdd(&d_z_f[bn + i], d_pt * s_prev);
        } else {
            d_S_prev[idx] = __float2bfloat16(d_pt * z_i * z_j);
            atomicAdd(&d_z_f[bn + i], d_pt * s_prev * z_j);
            atomicAdd(&d_z_f[bn + j], d_pt * s_prev * z_i);
        }

        atomicAdd(&d_v_f[bn + i], d_pt * k_j);
        atomicAdd(&d_k_f[bn + j], d_pt * v_i);
    }
}

// Backward for tanh z: z = tanh(z_logit + b_z)
// d_z_logit = d_z * (1 - z^2) where z is post-tanh value
template<typename T>
__global__ void E73TanhBackwardKernel(
    const int batch_size,
    const int n,
    const T* __restrict__ z,           // [B, n] post-tanh value
    const float* __restrict__ d_z_f,   // [B, n] float
    T* __restrict__ d_z_logit)         // [B, n]
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * n;

    if (idx < total) {
        float z_val = static_cast<float>(z[idx]);
        float d_z = d_z_f[idx];
        float d_logit = d_z * (1.0f - z_val * z_val);  // tanh derivative
        d_z_logit[idx] = static_cast<T>(d_logit);
    }
}

__global__ void E73TanhBackwardKernel_BF16(
    const int batch_size,
    const int n,
    const __nv_bfloat16* __restrict__ z,
    const float* __restrict__ d_z_f,
    __nv_bfloat16* __restrict__ d_z_logit)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * n;

    if (idx < total) {
        float z_val = __bfloat162float(z[idx]);
        float d_z = d_z_f[idx];
        float d_logit = d_z * (1.0f - z_val * z_val);  // tanh derivative
        d_z_logit[idx] = __float2bfloat16(d_logit);
    }
}

// Copy float to T
template<typename T>
__global__ void CopyFloatToT_E73(const int n, const float* __restrict__ src, T* __restrict__ dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = static_cast<T>(src[idx]);
    }
}

// Accumulate d_z_logit into db_z using atomicAdd
// d_z_logit is [T*B, n], we sum across T*B to get db_z[n]
__global__ void AccumulateDBzKernel_BF16(
    const int total,           // T*B*n total elements
    const int n,               // state dimension
    const __nv_bfloat16* __restrict__ d_z_logit,  // [T*B, n]
    float* __restrict__ db_z)  // [n] accumulator
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        const int j = idx % n;  // Which bias element this contributes to
        atomicAdd(&db_z[j], __bfloat162float(d_z_logit[idx]));
    }
}

template<typename T>
__global__ void AccumulateDBzKernel(
    const int total,
    const int n,
    const T* __restrict__ d_z_logit,
    float* __restrict__ db_z)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        const int j = idx % n;
        atomicAdd(&db_z[j], static_cast<float>(d_z_logit[idx]));
    }
}

// Apply tanh to z_logit with bias (bounded modulation to (-1, 1))
template<typename T>
__global__ void TanhBiasKernel(
    const int n,
    const T* __restrict__ z_logit,     // [n] input
    const T* __restrict__ b_z,         // [N] bias
    T* __restrict__ z,                 // [n] output
    const int batch_size,
    const int N)                       // state size to get bias index
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const int d = idx % N;
        float val = static_cast<float>(z_logit[idx]) + static_cast<float>(b_z[d]);
        z[idx] = static_cast<T>(tanhf(val));
    }
}

__global__ void TanhBiasKernel_BF16(
    const int n,
    const __nv_bfloat16* __restrict__ z_logit,
    const __nv_bfloat16* __restrict__ b_z,
    __nv_bfloat16* __restrict__ z,
    const int batch_size,
    const int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const int d = idx % N;
        float val = __bfloat162float(z_logit[idx]) + __bfloat162float(b_z[d]);
        z[idx] = __float2bfloat16(tanhf(val));
    }
}

}  // anonymous namespace


namespace hasty {
namespace v0 {
namespace elman_ladder {

// =============================================================================
// E73 Matrix Nonlinear Forward - BF16 Specialization
// =============================================================================

template<>
E73MatrixNonlinearForward<__nv_bfloat16>::E73MatrixNonlinearForward(
    bool training,
    int batch_size,
    int n_state,
    int dim,
    int variant,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : training_(training),
      batch_size_(batch_size),
      n_state_(n_state),
      dim_(dim),
      variant_(variant),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void E73MatrixNonlinearForward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_k,       // [n, dim]
    const __nv_bfloat16* W_v,       // [n, dim]
    const __nv_bfloat16* W_q,       // [n, dim]
    const __nv_bfloat16* W_z,       // [n, dim]
    const __nv_bfloat16* b_z,       // [n]
    const __nv_bfloat16* x,         // [T, B, dim] input
    __nv_bfloat16* S,               // [T+1, B, n, n] state matrices
    __nv_bfloat16* output,          // [T, B, n] output
    __nv_bfloat16* k_cache,         // [T, B, n] for backward
    __nv_bfloat16* v_cache,         // [T, B, n] for backward
    __nv_bfloat16* q_cache,         // [T, B, n] for backward
    __nv_bfloat16* z_cache,         // [T, B, n] for backward (post-sigmoid)
    __nv_bfloat16* pre_tanh_cache,  // [T, B, n, n] for backward
    __nv_bfloat16* Sq_cache,        // [T, B, n] for backward (pre-self-gate)
    __nv_bfloat16* workspace)
{
    static const __nv_bfloat16 alpha_one = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);

    const int n = n_state_;
    const int BD = batch_size_ * dim_;
    const int BN = batch_size_ * n;
    const int BNN = batch_size_ * n * n;
    const int block_size = 256;

    // Workspace layout:
    // k_all: [T, B, n]
    // v_all: [T, B, n]
    // q_all: [T, B, n]
    // z_logit_all: [T, B, n]
    // z_all: [T, B, n] (post-sigmoid)
    __nv_bfloat16* k_all = workspace;
    __nv_bfloat16* v_all = k_all + steps * BN;
    __nv_bfloat16* q_all = v_all + steps * BN;
    __nv_bfloat16* z_logit_all = q_all + steps * BN;
    __nv_bfloat16* z_all = z_logit_all + steps * BN;

    // =========================================================================
    // Batch all 4 projections upfront
    // =========================================================================

    // k_all = x @ W_k.T  [T*B, n]
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        n, steps * batch_size_, dim_,
        &alpha_one,
        W_k, dim_,
        x, dim_,
        &beta_zero,
        k_all, n);

    // v_all = x @ W_v.T  [T*B, n]
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        n, steps * batch_size_, dim_,
        &alpha_one,
        W_v, dim_,
        x, dim_,
        &beta_zero,
        v_all, n);

    // q_all = x @ W_q.T  [T*B, n]
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        n, steps * batch_size_, dim_,
        &alpha_one,
        W_q, dim_,
        x, dim_,
        &beta_zero,
        q_all, n);

    // z_logit_all = x @ W_z.T  [T*B, n]
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        n, steps * batch_size_, dim_,
        &alpha_one,
        W_z, dim_,
        x, dim_,
        &beta_zero,
        z_logit_all, n);

    // Apply tanh with bias to z_logit_all -> z_all (bounded modulation)
    TanhBiasKernel_BF16<<<(steps * BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
        steps * BN, z_logit_all, b_z, z_all, batch_size_, n);

    // Process each timestep sequentially (recurrence)
    for (int t = 0; t < steps; ++t) {
        const __nv_bfloat16* k_t = k_all + t * BN;
        const __nv_bfloat16* v_t = v_all + t * BN;
        const __nv_bfloat16* q_t = q_all + t * BN;
        const __nv_bfloat16* z_t = z_all + t * BN;
        const __nv_bfloat16* S_prev = S + t * BNN;
        __nv_bfloat16* S_t = S + (t + 1) * BNN;
        __nv_bfloat16* out_t = output + t * BN;
        __nv_bfloat16* pre_tanh_t = training_ ? (pre_tanh_cache + t * BNN) : nullptr;
        __nv_bfloat16* Sq_t = training_ ? (Sq_cache + t * BN) : nullptr;

        // Cache k, v, q, z for backward
        if (training_) {
            cudaMemcpyAsync(k_cache + t * BN, k_t, BN * sizeof(__nv_bfloat16),
                           cudaMemcpyDeviceToDevice, stream_);
            cudaMemcpyAsync(v_cache + t * BN, v_t, BN * sizeof(__nv_bfloat16),
                           cudaMemcpyDeviceToDevice, stream_);
            cudaMemcpyAsync(q_cache + t * BN, q_t, BN * sizeof(__nv_bfloat16),
                           cudaMemcpyDeviceToDevice, stream_);
            cudaMemcpyAsync(z_cache + t * BN, z_t, BN * sizeof(__nv_bfloat16),
                           cudaMemcpyDeviceToDevice, stream_);
        }

        // Update: S_t = tanh(S_prev * z_mod + outer(v, k))
        E73UpdateKernel_BF16<<<(BNN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n, variant_, S_prev, z_t, v_t, k_t, S_t, pre_tanh_t);

        // Output: out = S @ q, then self-gate
        E73OutputKernel_BF16<<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n, S_t, q_t, out_t, Sq_t);
    }
}

// =============================================================================
// E73 Matrix Nonlinear Backward - BF16 Specialization
// =============================================================================

template<>
E73MatrixNonlinearBackward<__nv_bfloat16>::E73MatrixNonlinearBackward(
    int batch_size,
    int n_state,
    int dim,
    int variant,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      n_state_(n_state),
      dim_(dim),
      variant_(variant),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void E73MatrixNonlinearBackward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_k,
    const __nv_bfloat16* W_v,
    const __nv_bfloat16* W_q,
    const __nv_bfloat16* W_z,
    const __nv_bfloat16* x,
    const __nv_bfloat16* S,
    const __nv_bfloat16* k_cache,
    const __nv_bfloat16* v_cache,
    const __nv_bfloat16* q_cache,
    const __nv_bfloat16* z_cache,
    const __nv_bfloat16* pre_tanh_cache,
    const __nv_bfloat16* Sq_cache,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* dx,
    __nv_bfloat16* dW_k,
    __nv_bfloat16* dW_v,
    __nv_bfloat16* dW_q,
    __nv_bfloat16* dW_z,
    __nv_bfloat16* db_z,
    __nv_bfloat16* workspace)
{
    static const __nv_bfloat16 alpha_one = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);
    static const __nv_bfloat16 beta_one = __float2bfloat16(1.0f);

    const int n = n_state_;
    const int BD = batch_size_ * dim_;
    const int BN = batch_size_ * n;
    const int BNN = batch_size_ * n * n;
    const int block_size = 256;

    // Workspace layout:
    // d_S: [B, n, n] - current state gradient
    // d_S_tmp: [B, n, n] - state gradient temp
    // d_Sq: [B, n] - gradient through self-gate
    // d_k_all: [T, B, n]
    // d_v_all: [T, B, n]
    // d_q_all: [T, B, n]
    // d_z_logit_all: [T, B, n]
    // Float accumulators: d_k_f, d_v_f, d_q_f, d_z_f: 4*BN
    // db_z_f: [n]
    __nv_bfloat16* d_S = workspace;
    __nv_bfloat16* d_S_tmp = d_S + BNN;
    __nv_bfloat16* d_Sq = d_S_tmp + BNN;
    __nv_bfloat16* d_k_all = d_Sq + BN;
    __nv_bfloat16* d_v_all = d_k_all + steps * BN;
    __nv_bfloat16* d_q_all = d_v_all + steps * BN;
    __nv_bfloat16* d_z_logit_all = d_q_all + steps * BN;

    float* float_ws = reinterpret_cast<float*>(d_z_logit_all + steps * BN);
    float* d_k_f = float_ws;
    float* d_v_f = d_k_f + BN;
    float* d_q_f = d_v_f + BN;
    float* d_z_f = d_q_f + BN;
    float* db_z_f = d_z_f + BN;

    // Initialize
    cudaMemsetAsync(d_S, 0, BNN * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_k, 0, n * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_v, 0, n * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_q, 0, n * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_z, 0, n * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(db_z_f, 0, n * sizeof(float), stream_);

    // BPTT loop
    for (int t = steps - 1; t >= 0; --t) {
        const __nv_bfloat16* S_t = S + (t + 1) * BNN;
        const __nv_bfloat16* S_prev = S + t * BNN;
        const __nv_bfloat16* k_t = k_cache + t * BN;
        const __nv_bfloat16* v_t = v_cache + t * BN;
        const __nv_bfloat16* q_t = q_cache + t * BN;
        const __nv_bfloat16* z_t = z_cache + t * BN;
        const __nv_bfloat16* pre_tanh_t = pre_tanh_cache + t * BNN;
        const __nv_bfloat16* Sq_t = Sq_cache + t * BN;
        const __nv_bfloat16* d_out_t = d_output + t * BN;

        __nv_bfloat16* d_k_t = d_k_all + t * BN;
        __nv_bfloat16* d_v_t = d_v_all + t * BN;
        __nv_bfloat16* d_q_t = d_q_all + t * BN;
        __nv_bfloat16* d_z_logit_t = d_z_logit_all + t * BN;

        // Zero per-timestep float accumulators
        cudaMemsetAsync(d_k_f, 0, BN * sizeof(float), stream_);
        cudaMemsetAsync(d_v_f, 0, BN * sizeof(float), stream_);
        cudaMemsetAsync(d_q_f, 0, BN * sizeof(float), stream_);
        cudaMemsetAsync(d_z_f, 0, BN * sizeof(float), stream_);

        // 1. Backward through self-gate: out = Sq * silu(Sq)
        E73SelfGateBackwardKernel_BF16<<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n, Sq_t, d_out_t, d_Sq);

        // 2. Backward through output: Sq = S @ q
        // Adds to d_S, accumulates d_q
        E73OutputBackwardKernel_BF16<<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n, S_t, q_t, d_Sq, d_S, d_q_f);

        // 3. Backward through update: S_new = tanh(S_prev * z_mod + outer(v, k))
        // Output: d_S_tmp = gradient to S_prev
        E73UpdateBackwardKernel_BF16<<<(BNN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n, variant_, S_prev, z_t, v_t, k_t, pre_tanh_t, d_S,
            d_S_tmp, d_z_f, d_v_f, d_k_f);

        // 4. Backward through tanh z: d_z_logit = d_z * (1 - z^2)
        E73TanhBackwardKernel_BF16<<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n, z_t, d_z_f, d_z_logit_t);

        // Accumulate db_z
        for (int b_idx = 0; b_idx < batch_size_; ++b_idx) {
            // This is inefficient, but correct. Could use a reduction kernel.
            // For now, just use atomicAdd in a kernel
        }
        // Use a simple kernel to sum d_z_logit_t across batch to db_z_f
        // (skipped for simplicity, handled in separate reduction or atomics)

        // Copy float gradients to output
        CopyFloatToT_E73<__nv_bfloat16><<<(BN + 255) / 256, 256, 0, stream_>>>(BN, d_k_f, d_k_t);
        CopyFloatToT_E73<__nv_bfloat16><<<(BN + 255) / 256, 256, 0, stream_>>>(BN, d_v_f, d_v_t);
        CopyFloatToT_E73<__nv_bfloat16><<<(BN + 255) / 256, 256, 0, stream_>>>(BN, d_q_f, d_q_t);

        // Swap d_S and d_S_tmp for next iteration
        std::swap(d_S, d_S_tmp);
    }

    // =========================================================================
    // Batched GEMMs for weight gradients and dx
    // =========================================================================

    // dx = d_k_all @ W_k + d_v_all @ W_v + d_q_all @ W_q + d_z_logit_all @ W_z

    // dx from k: dx = d_k_all @ W_k
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, n,
        &alpha_one,
        W_k, dim_,
        d_k_all, n,
        &beta_zero,
        dx, dim_);

    // dx += d_v_all @ W_v
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, n,
        &alpha_one,
        W_v, dim_,
        d_v_all, n,
        &beta_one,
        dx, dim_);

    // dx += d_q_all @ W_q
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, n,
        &alpha_one,
        W_q, dim_,
        d_q_all, n,
        &beta_one,
        dx, dim_);

    // dx += d_z_logit_all @ W_z
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, n,
        &alpha_one,
        W_z, dim_,
        d_z_logit_all, n,
        &beta_one,
        dx, dim_);

    // dW_k = x.T @ d_k_all
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, n, steps * batch_size_,
        &alpha_one,
        x, dim_,
        d_k_all, n,
        &beta_one,
        dW_k, dim_);

    // dW_v = x.T @ d_v_all
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, n, steps * batch_size_,
        &alpha_one,
        x, dim_,
        d_v_all, n,
        &beta_one,
        dW_v, dim_);

    // dW_q = x.T @ d_q_all
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, n, steps * batch_size_,
        &alpha_one,
        x, dim_,
        d_q_all, n,
        &beta_one,
        dW_q, dim_);

    // dW_z = x.T @ d_z_logit_all
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, n, steps * batch_size_,
        &alpha_one,
        x, dim_,
        d_z_logit_all, n,
        &beta_one,
        dW_z, dim_);

    // Sum d_z_logit_all to db_z: db_z[j] = sum over (t, b) of d_z_logit_all[t, b, j]
    // Call the accumulation kernel defined in anonymous namespace
    {
        const int total_elements = steps * BN;
        const int block_sz = 256;
        AccumulateDBzKernel_BF16<<<(total_elements + block_sz - 1) / block_sz, block_sz, 0, stream_>>>(
            total_elements, n, d_z_logit_all, db_z_f);
    }
    // Copy accumulated db_z_f (float) to db_z (bf16)
    CopyFloatToT_E73<__nv_bfloat16><<<(n + 255) / 256, 256, 0, stream_>>>(n, db_z_f, db_z);
}

// =============================================================================
// Generic Template Implementations (float, half, double)
// =============================================================================

template<typename T>
E73MatrixNonlinearForward<T>::E73MatrixNonlinearForward(
    bool training,
    int batch_size,
    int n_state,
    int dim,
    int variant,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : training_(training),
      batch_size_(batch_size),
      n_state_(n_state),
      dim_(dim),
      variant_(variant),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void E73MatrixNonlinearForward<T>::Run(
    int steps,
    const T* W_k,
    const T* W_v,
    const T* W_q,
    const T* W_z,
    const T* b_z,
    const T* x,
    T* S,
    T* output,
    T* k_cache,
    T* v_cache,
    T* q_cache,
    T* z_cache,
    T* pre_tanh_cache,
    T* Sq_cache,
    T* workspace)
{
    static const T alpha_one = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int n = n_state_;
    const int BD = batch_size_ * dim_;
    const int BN = batch_size_ * n;
    const int BNN = batch_size_ * n * n;
    const int block_size = 256;

    T* k_all = workspace;
    T* v_all = k_all + steps * BN;
    T* q_all = v_all + steps * BN;
    T* z_logit_all = q_all + steps * BN;
    T* z_all = z_logit_all + steps * BN;

    // Batch projections
    blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        n, steps * batch_size_, dim_, &alpha_one, W_k, dim_, x, dim_, &beta_zero, k_all, n);
    blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        n, steps * batch_size_, dim_, &alpha_one, W_v, dim_, x, dim_, &beta_zero, v_all, n);
    blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        n, steps * batch_size_, dim_, &alpha_one, W_q, dim_, x, dim_, &beta_zero, q_all, n);
    blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        n, steps * batch_size_, dim_, &alpha_one, W_z, dim_, x, dim_, &beta_zero, z_logit_all, n);

    // Apply tanh with bias (bounded modulation)
    TanhBiasKernel<T><<<(steps * BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
        steps * BN, z_logit_all, b_z, z_all, batch_size_, n);

    for (int t = 0; t < steps; ++t) {
        const T* k_t = k_all + t * BN;
        const T* v_t = v_all + t * BN;
        const T* q_t = q_all + t * BN;
        const T* z_t = z_all + t * BN;
        const T* S_prev = S + t * BNN;
        T* S_t = S + (t + 1) * BNN;
        T* out_t = output + t * BN;
        T* pre_tanh_t = training_ ? (pre_tanh_cache + t * BNN) : nullptr;
        T* Sq_t = training_ ? (Sq_cache + t * BN) : nullptr;

        if (training_) {
            cudaMemcpyAsync(k_cache + t * BN, k_t, BN * sizeof(T), cudaMemcpyDeviceToDevice, stream_);
            cudaMemcpyAsync(v_cache + t * BN, v_t, BN * sizeof(T), cudaMemcpyDeviceToDevice, stream_);
            cudaMemcpyAsync(q_cache + t * BN, q_t, BN * sizeof(T), cudaMemcpyDeviceToDevice, stream_);
            cudaMemcpyAsync(z_cache + t * BN, z_t, BN * sizeof(T), cudaMemcpyDeviceToDevice, stream_);
        }

        E73UpdateKernel<T><<<(BNN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n, variant_, S_prev, z_t, v_t, k_t, S_t, pre_tanh_t);

        E73OutputKernel<T><<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n, S_t, q_t, out_t, Sq_t);
    }
}

template<typename T>
E73MatrixNonlinearBackward<T>::E73MatrixNonlinearBackward(
    int batch_size,
    int n_state,
    int dim,
    int variant,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      n_state_(n_state),
      dim_(dim),
      variant_(variant),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void E73MatrixNonlinearBackward<T>::Run(
    int steps,
    const T* W_k,
    const T* W_v,
    const T* W_q,
    const T* W_z,
    const T* x,
    const T* S,
    const T* k_cache,
    const T* v_cache,
    const T* q_cache,
    const T* z_cache,
    const T* pre_tanh_cache,
    const T* Sq_cache,
    const T* d_output,
    T* dx,
    T* dW_k,
    T* dW_v,
    T* dW_q,
    T* dW_z,
    T* db_z,
    T* workspace)
{
    // Placeholder - follows BF16 pattern
    cudaMemsetAsync(dx, 0, steps * batch_size_ * dim_ * sizeof(T), stream_);
}

// Explicit template instantiations
template struct E73MatrixNonlinearForward<__half>;
template struct E73MatrixNonlinearForward<float>;
template struct E73MatrixNonlinearForward<double>;

template struct E73MatrixNonlinearBackward<__half>;
template struct E73MatrixNonlinearBackward<float>;
template struct E73MatrixNonlinearBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
