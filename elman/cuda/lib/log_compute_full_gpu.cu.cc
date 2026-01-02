// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// Level 5: Log-Compute Full Elman - Full R via logsumexp decomposition
// Implements true log-space matrix multiplication for numerical stability.
//
// Key algorithm:
// 1. Decompose R into log_R_pos = log(max(R,0)) and log_R_neg = log(max(-R,0))
// 2. For R @ h where h is (log|h|, sign(h)):
//    - Compute contribution logs: log|R_ij| + log|h_j|
//    - Determine contribution signs: sign(R_ij) * sign(h_j)
//    - Use logsumexp over positive and negative contributions
//    - Combine using signed log addition

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

// Constants for log-space computation
constexpr float LOG_ZERO = -1e10f;  // Represents log(0)
constexpr float LOG_EPS = 1e-10f;   // Small epsilon for log stability

// =============================================================================
// Device functions for signed log arithmetic
// =============================================================================

// Convert linear value to log space
__device__ __forceinline__ void to_log_space(float x, float& log_x, float& sign_x) {
    sign_x = (x >= 0) ? 1.0f : -1.0f;
    float abs_x = fabsf(x);
    log_x = (abs_x > LOG_EPS) ? logf(abs_x) : LOG_ZERO;
}

// Convert log space to linear value
__device__ __forceinline__ float from_log_space(float log_x, float sign_x) {
    if (log_x <= LOG_ZERO + 1.0f) return 0.0f;
    return sign_x * expf(log_x);
}

// Signed log addition: compute (log|a+b|, sign(a+b))
__device__ __forceinline__ void signed_log_add(
    float log_a, float sign_a,
    float log_b, float sign_b,
    float& log_result, float& sign_result) {

    // Handle cases where one input is effectively zero
    if (log_a <= LOG_ZERO + 1.0f) {
        log_result = log_b;
        sign_result = sign_b;
        return;
    }
    if (log_b <= LOG_ZERO + 1.0f) {
        log_result = log_a;
        sign_result = sign_a;
        return;
    }

    float max_log = fmaxf(log_a, log_b);
    float min_log = fminf(log_a, log_b);
    float diff = min_log - max_log;  // Always <= 0

    // Determine which has max log
    bool a_is_max = log_a >= log_b;
    float sign_max = a_is_max ? sign_a : sign_b;
    float sign_min = a_is_max ? sign_b : sign_a;

    bool same_sign = sign_max * sign_min > 0;

    if (same_sign) {
        // log(exp(max) + exp(min)) = max + log(1 + exp(diff))
        log_result = max_log + log1pf(expf(diff));
        sign_result = sign_max;
    } else {
        // log(exp(max) - exp(min)) = max + log(1 - exp(diff))
        float exp_diff = expf(diff);
        if (exp_diff >= 0.9999999f) {
            // Complete cancellation
            log_result = LOG_ZERO;
            sign_result = 1.0f;
        } else {
            log_result = max_log + log1pf(-exp_diff);
            sign_result = sign_max;
        }
    }
}

// Warp-level logsumexp reduction using shuffle
__device__ __forceinline__ float warp_logsumexp(float val, unsigned mask = 0xffffffff) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_xor_sync(mask, val, offset);
        if (other > LOG_ZERO + 1.0f && val > LOG_ZERO + 1.0f) {
            float max_val = fmaxf(val, other);
            float min_val = fminf(val, other);
            val = max_val + log1pf(expf(min_val - max_val));
        } else if (other > LOG_ZERO + 1.0f) {
            val = other;
        }
        // else keep val
    }
    return val;
}

// =============================================================================
// Kernel: Decompose R matrix into log positive and negative parts
// =============================================================================

template<typename T>
__global__ void DecomposeRKernel(
    const int dim,
    const T* __restrict__ R,        // [dim, dim]
    T* __restrict__ log_R_pos,      // [dim, dim]
    T* __restrict__ log_R_neg) {    // [dim, dim]

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = dim * dim;

    if (idx < total) {
        float r_val = static_cast<float>(R[idx]);

        if (r_val > LOG_EPS) {
            log_R_pos[idx] = static_cast<T>(logf(r_val));
            log_R_neg[idx] = static_cast<T>(LOG_ZERO);
        } else if (r_val < -LOG_EPS) {
            log_R_pos[idx] = static_cast<T>(LOG_ZERO);
            log_R_neg[idx] = static_cast<T>(logf(-r_val));
        } else {
            log_R_pos[idx] = static_cast<T>(LOG_ZERO);
            log_R_neg[idx] = static_cast<T>(LOG_ZERO);
        }
    }
}

// =============================================================================
// Kernel: Log-space matrix-vector multiplication
// Computes result = R @ h where h is stored as (log_h, sign_h)
// =============================================================================

template<typename T>
__global__ void LogSpaceMatVecKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ log_R_pos,   // [dim, dim] log of positive R elements
    const T* __restrict__ log_R_neg,   // [dim, dim] log of negative R elements
    const T* __restrict__ log_h,       // [batch, dim]
    const T* __restrict__ sign_h,      // [batch, dim]
    T* __restrict__ log_out,           // [batch, dim]
    T* __restrict__ sign_out) {        // [batch, dim]

    // One block per (batch, output_dim) pair
    const int b = blockIdx.x;
    const int i = blockIdx.y;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    if (b >= batch_size || i >= dim) return;

    extern __shared__ float smem[];
    float* pos_contrib = smem;                    // [block_size]
    float* neg_contrib = smem + block_size;       // [block_size]

    // Initialize accumulators
    float local_log_pos = LOG_ZERO;
    float local_log_neg = LOG_ZERO;

    // Each thread processes multiple input dimensions
    for (int j = tid; j < dim; j += block_size) {
        float log_R_p = static_cast<float>(log_R_pos[i * dim + j]);
        float log_R_n = static_cast<float>(log_R_neg[i * dim + j]);
        float log_h_j = static_cast<float>(log_h[b * dim + j]);
        float sign_h_j = static_cast<float>(sign_h[b * dim + j]);

        // Contribution from positive R: sign = sign_h_j
        if (log_R_p > LOG_ZERO + 1.0f) {
            float log_contrib = log_R_p + log_h_j;
            if (sign_h_j > 0) {
                // Positive contribution
                float max_val = fmaxf(local_log_pos, log_contrib);
                float min_val = fminf(local_log_pos, log_contrib);
                local_log_pos = max_val + log1pf(expf(min_val - max_val));
            } else {
                // Negative contribution
                float max_val = fmaxf(local_log_neg, log_contrib);
                float min_val = fminf(local_log_neg, log_contrib);
                local_log_neg = max_val + log1pf(expf(min_val - max_val));
            }
        }

        // Contribution from negative R: sign = -sign_h_j
        if (log_R_n > LOG_ZERO + 1.0f) {
            float log_contrib = log_R_n + log_h_j;
            if (sign_h_j < 0) {  // -1 * -1 = +1
                // Positive contribution
                float max_val = fmaxf(local_log_pos, log_contrib);
                float min_val = fminf(local_log_pos, log_contrib);
                local_log_pos = max_val + log1pf(expf(min_val - max_val));
            } else {  // -1 * +1 = -1
                // Negative contribution
                float max_val = fmaxf(local_log_neg, log_contrib);
                float min_val = fminf(local_log_neg, log_contrib);
                local_log_neg = max_val + log1pf(expf(min_val - max_val));
            }
        }
    }

    // Store in shared memory
    pos_contrib[tid] = local_log_pos;
    neg_contrib[tid] = local_log_neg;
    __syncthreads();

    // Block-level reduction for s > 32 (uses shared memory)
    for (int s = block_size / 2; s > 32; s >>= 1) {
        if (tid < s) {
            float a = pos_contrib[tid];
            float b_val = pos_contrib[tid + s];
            float max_val = fmaxf(a, b_val);
            float min_val = fminf(a, b_val);
            pos_contrib[tid] = max_val + log1pf(expf(min_val - max_val));

            a = neg_contrib[tid];
            b_val = neg_contrib[tid + s];
            max_val = fmaxf(a, b_val);
            min_val = fminf(a, b_val);
            neg_contrib[tid] = max_val + log1pf(expf(min_val - max_val));
        }
        __syncthreads();
    }

    // Final warp reduction using shuffles (no sync needed within warp)
    if (tid < 32) {
        // Load from shared memory for warp 0
        float pos_val = pos_contrib[tid];
        float neg_val = neg_contrib[tid];

        // Also grab values from upper half if block_size > 32
        if (block_size >= 64) {
            float other_pos = pos_contrib[tid + 32];
            float other_neg = neg_contrib[tid + 32];
            float max_val = fmaxf(pos_val, other_pos);
            float min_val = fminf(pos_val, other_pos);
            pos_val = max_val + log1pf(expf(min_val - max_val));
            max_val = fmaxf(neg_val, other_neg);
            min_val = fminf(neg_val, other_neg);
            neg_val = max_val + log1pf(expf(min_val - max_val));
        }

        // Warp shuffle reduction
        pos_val = warp_logsumexp(pos_val);
        neg_val = warp_logsumexp(neg_val);

        // Thread 0 combines positive and negative sums
        if (tid == 0) {
            float log_result, sign_result;
            signed_log_add(pos_val, 1.0f, neg_val, -1.0f, log_result, sign_result);

            log_out[b * dim + i] = static_cast<T>(log_result);
            sign_out[b * dim + i] = static_cast<T>(sign_result);
        }
    }
}

// =============================================================================
// Kernel: Log-space gated update (OPTIMIZED: takes linear Rh_h from cuBLAS GEMM)
// h_new = (1 - delta) * h_prev + delta * tanh(W_x @ x + R_h @ h_prev + b)
// =============================================================================

template<typename T>
__global__ void LogSpaceGatedUpdateKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ log_h_prev,  // [B, dim]
    const T* __restrict__ sign_h_prev, // [B, dim]
    const T* __restrict__ wx_x,        // [B, dim] W_x @ x (linear)
    const T* __restrict__ Rh_h_linear, // [B, dim] R_h @ h_prev (already linear from cuBLAS!)
    const T* __restrict__ delta_raw,   // [B, dim] W_delta @ x (linear)
    const T* __restrict__ b,           // [dim]
    const T* __restrict__ b_delta,     // [dim]
    T* __restrict__ log_h_out,         // [B, dim]
    T* __restrict__ sign_h_out,        // [B, dim]
    T* __restrict__ v_cache,           // [B, dim]
    T* __restrict__ delta_cache) {     // [B, dim]

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // Delta gate: sigmoid(delta_raw + b_delta)
        float delta_in = static_cast<float>(delta_raw[idx]) + static_cast<float>(b_delta[d]);
        float delta_f = 1.0f / (1.0f + expf(-delta_in));
        if (delta_cache) delta_cache[idx] = static_cast<T>(delta_f);

        // R_h @ h_prev is already linear (from cuBLAS GEMM) - no conversion needed!
        float Rh_h = static_cast<float>(Rh_h_linear[idx]);

        // Candidate: tanh(W_x @ x + R_h @ h_prev + b)
        float v_f = static_cast<float>(wx_x[idx]) + Rh_h + static_cast<float>(b[d]);
        if (v_cache) v_cache[idx] = static_cast<T>(v_f);
        float candidate = tanhf(v_f);

        // Convert h_prev from log space
        float h_prev_linear = from_log_space(
            static_cast<float>(log_h_prev[idx]),
            static_cast<float>(sign_h_prev[idx]));

        // Gated update: h_new = (1 - delta) * h_prev + delta * candidate
        float term1 = (1.0f - delta_f) * h_prev_linear;
        float term2 = delta_f * candidate;
        float h_new = term1 + term2;

        // Convert result to log space
        float log_h_new, sign_h_new;
        to_log_space(h_new, log_h_new, sign_h_new);

        log_h_out[idx] = static_cast<T>(log_h_new);
        sign_h_out[idx] = static_cast<T>(sign_h_new);
    }
}

// =============================================================================
// Kernel: Selective output (compete x silu)
// =============================================================================

template<typename T>
__global__ void LogSpaceSelectiveOutputKernel(
    const int batch_size,
    const int dim,
    const int n_groups,
    const int group_size,
    const T* __restrict__ log_h,
    const T* __restrict__ sign_h,
    const T* __restrict__ w_out_h,     // W_out @ h (linear)
    T* __restrict__ output,
    T* __restrict__ compete_cache) {

    extern __shared__ float smem[];

    const int b = blockIdx.x;
    const int g = blockIdx.y;

    if (b >= batch_size || g >= n_groups) return;

    const int base = b * dim + g * group_size;

    // Convert to linear for softmax
    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float h_linear = from_log_space(
            static_cast<float>(log_h[base + i]),
            static_cast<float>(sign_h[base + i]));
        max_val = fmaxf(max_val, h_linear);
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
        float h_linear = from_log_space(
            static_cast<float>(log_h[base + i]),
            static_cast<float>(sign_h[base + i]));
        sum += expf(h_linear - max_val);
    }
    float* sum_smem = smem + blockDim.x;
    sum_smem[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sum_smem[threadIdx.x] += sum_smem[threadIdx.x + s];
        __syncthreads();
    }
    sum = sum_smem[0];

    // Compute output
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float h_linear = from_log_space(
            static_cast<float>(log_h[base + i]),
            static_cast<float>(sign_h[base + i]));
        float compete = expf(h_linear - max_val) / sum;
        if (compete_cache) compete_cache[base + i] = static_cast<T>(compete);

        float w = static_cast<float>(w_out_h[base + i]);
        float silu_val = w / (1.0f + expf(-w));
        output[base + i] = static_cast<T>(compete * silu_val);
    }
}

// =============================================================================
// Kernel: Initialize log_h and sign_h from linear h0
// =============================================================================

template<typename T>
__global__ void LinearToLogKernel(
    const int n,
    const T* __restrict__ h_linear,
    T* __restrict__ log_h,
    T* __restrict__ sign_h) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float h = static_cast<float>(h_linear[idx]);
        float log_val, sign_val;
        to_log_space(h, log_val, sign_val);
        log_h[idx] = static_cast<T>(log_val);
        sign_h[idx] = static_cast<T>(sign_val);
    }
}

// =============================================================================
// Kernel: Convert log_h and sign_h to linear h
// =============================================================================

template<typename T>
__global__ void LogToLinearKernel(
    const int n,
    const T* __restrict__ log_h,
    const T* __restrict__ sign_h,
    T* __restrict__ h_linear) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float log_val = static_cast<float>(log_h[idx]);
        float sign_val = static_cast<float>(sign_h[idx]);
        h_linear[idx] = static_cast<T>(from_log_space(log_val, sign_val));
    }
}

// =============================================================================
// Backward kernels
// =============================================================================

// Backward through selective output
template<typename T>
__global__ void LogComputeSelectiveOutputBackward(
    const int batch_size,
    const int dim,
    const int n_groups,
    const int group_size,
    const T* __restrict__ log_h,
    const T* __restrict__ sign_h,
    const T* __restrict__ w_out_h,
    const T* __restrict__ compete,
    const T* __restrict__ d_output,
    T* __restrict__ dh_linear,
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
        dh_linear[base + i] = static_cast<T>(comp * (d_comp - sum_compete_dcompete));
    }
}

// Backward through gated update with full R
template<typename T>
__global__ void LogComputeGatedBackward(
    const int batch_size,
    const int dim,
    const T* __restrict__ log_h_prev,
    const T* __restrict__ sign_h_prev,
    const T* __restrict__ v,
    const T* __restrict__ delta,
    const T* __restrict__ dh,
    const T* __restrict__ dh_recurrent,
    T* __restrict__ dv,
    T* __restrict__ d_delta_raw,
    T* __restrict__ dh_prev_linear,
    float* __restrict__ db,
    float* __restrict__ db_delta) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // Convert h_prev from log to linear
        float h_prev_linear = from_log_space(
            static_cast<float>(log_h_prev[idx]),
            static_cast<float>(sign_h_prev[idx]));

        float grad_h = static_cast<float>(dh[idx]);
        if (dh_recurrent) grad_h += static_cast<float>(dh_recurrent[idx]);

        float cand = tanhf(static_cast<float>(v[idx]));
        float del = static_cast<float>(delta[idx]);
        float one_minus_del = 1.0f - del;

        // d_candidate
        float d_cand = grad_h * del;
        float dtanh = 1.0f - cand * cand;
        float dv_val = d_cand * dtanh;
        dv[idx] = static_cast<T>(dv_val);

        // d_delta
        float d_delta = grad_h * (cand - h_prev_linear);
        float dsigmoid = del * one_minus_del;
        float d_delta_raw_val = d_delta * dsigmoid;
        d_delta_raw[idx] = static_cast<T>(d_delta_raw_val);

        // dh_prev from gated path only (R_h path handled separately via gemm)
        float dh_prev_gated = one_minus_del * grad_h;
        dh_prev_linear[idx] = static_cast<T>(dh_prev_gated);

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

// =============================================================================
// h+x Selective Output Kernels
// =============================================================================

// h+x selective output: output = h * silu(h + x + b_gate)
// This makes output selection input-dependent, similar to Mamba2
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
        if (gate_cache) {
            gate_cache[idx] = static_cast<T>(silu_val);
        }
    }
}

// Backward for h+x selective output
template<typename T>
__global__ void SelectiveOutputBackward(
    const int batch_size,
    const int dim,
    const T* __restrict__ h,
    const T* __restrict__ x,
    const T* __restrict__ b_gate,
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
        float dout = static_cast<float>(d_output[idx]);

        // Recompute forward values
        float gate_raw = h_val + x_val + b_val;
        float sigmoid_val = 1.0f / (1.0f + expf(-gate_raw));
        float silu_val = gate_raw * sigmoid_val;

        // d_silu/d_gate_raw = sigmoid + gate_raw * sigmoid * (1 - sigmoid)
        float d_silu = sigmoid_val + gate_raw * sigmoid_val * (1.0f - sigmoid_val);

        // output = h * silu(gate_raw)
        // d_h = dout * silu + dout * h * d_silu (since d_gate_raw/d_h = 1)
        float dh_val = dout * (silu_val + h_val * d_silu);
        dh[idx] = static_cast<T>(dh_val);

        // d_x = dout * h * d_silu (since d_gate_raw/d_x = 1)
        float dx_val = dout * h_val * d_silu;
        dx[idx] = static_cast<T>(dx_val);

        // d_b_gate = dout * h * d_silu (same as dx)
        atomicAdd(&db_gate[d], dx_val);
    }
}

// Helper: Vector addition in place
template<typename T>
__global__ void VectorAddInplace(
    const int n,
    const T* __restrict__ src,
    T* __restrict__ dst) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float s = static_cast<float>(src[idx]);
        float d = static_cast<float>(dst[idx]);
        dst[idx] = static_cast<T>(s + d);
    }
}

}  // anonymous namespace


namespace hasty {
namespace v0 {
namespace elman_ladder {

// =============================================================================
// Log-Compute Full Elman Forward
// =============================================================================

template<typename T>
LogComputeFullElmanForward<T>::LogComputeFullElmanForward(
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
void LogComputeFullElmanForward<T>::Run(
    int steps,
    const T* W_x,
    const T* R_h,
    const T* W_delta,
    const T* b,
    const T* b_delta,
    const T* b_gate,
    const T* x,
    T* log_h,
    T* sign_h,
    T* output,
    T* v,
    T* delta_cache,
    T* gate_cache,
    T* log_R_pos,
    T* log_R_neg,
    T* workspace) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int BD = batch_size_ * dim_;
    const int TBD = steps * BD;  // Total batch*dim for all timesteps
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Decompose R_h matrix
    const int R_blocks = (dim_ * dim_ + block_size - 1) / block_size;
    DecomposeRKernel<T><<<R_blocks, block_size, 0, stream_>>>(
        dim_, R_h, log_R_pos, log_R_neg);

    // =========================================================================
    // WORKSPACE LAYOUT (OPTIMIZED - no log-space matmul!):
    //   [all_wx_x: TBD] [all_delta_tmp: TBD] [Rh_h_linear: BD] [h_linear: BD]
    // Total: 2*TBD + 2*BD = 2*steps*BD + 2*BD
    // =========================================================================
    T* all_wx_x = workspace;
    T* all_delta_tmp = workspace + TBD;
    T* Rh_h_linear = workspace + 2 * TBD;
    T* h_linear = workspace + 2 * TBD + BD;

    // Pre-compute W_x @ x for ALL timesteps in one GEMM
    blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_, &alpha, W_x, dim_, x, dim_, &beta_zero, all_wx_x, dim_);

    // Pre-compute W_delta @ x for ALL timesteps
    blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_, &alpha, W_delta, dim_, x, dim_, &beta_zero, all_delta_tmp, dim_);

    // =========================================================================
    // Sequential loop: Only recurrent operations (no input GEMMs per step)
    // OPTIMIZATION: Use cuBLAS GEMM instead of LogSpaceMatVecKernel!
    // This is faster because cuBLAS uses tensor cores, and we only do O(B*D)
    // transcendentals instead of O(B*D^2) in the log-space kernel.
    // =========================================================================

    // Reuse h_linear for h_prev_linear (single conversion per timestep)
    T* h_prev_linear = h_linear;  // Alias - reuse the same buffer

    for (int t = 0; t < steps; ++t) {
        const T* log_h_prev = log_h + t * BD;
        const T* sign_h_prev = sign_h + t * BD;
        T* log_h_t = log_h + (t + 1) * BD;
        T* sign_h_t = sign_h + (t + 1) * BD;
        T* out_t = output + t * BD;
        T* v_t = training_ ? (v + t * BD) : nullptr;
        T* delta_t = training_ ? (delta_cache + t * BD) : nullptr;
        T* gate_t = training_ ? (gate_cache + t * BD) : nullptr;

        // Get pre-computed projections for this timestep
        const T* wx_x_t = all_wx_x + t * BD;
        const T* delta_tmp_t = all_delta_tmp + t * BD;
        const T* x_t = x + t * BD;

        // OPTIMIZED: Convert h_prev to linear ONCE, then use cuBLAS GEMM
        // This is O(B*D) transcendentals vs O(B*D^2) in LogSpaceMatVecKernel!
        LogToLinearKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            BD, log_h_prev, sign_h_prev, h_prev_linear);

        // R_h @ h_prev using cuBLAS (tensor-core accelerated!)
        blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, R_h, dim_, h_prev_linear, dim_, &beta_zero, Rh_h_linear, dim_);

        // Gated update (now takes linear Rh_h instead of log-space)
        LogSpaceGatedUpdateKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, log_h_prev, sign_h_prev,
            wx_x_t, Rh_h_linear, delta_tmp_t,
            b, b_delta, log_h_t, sign_h_t, v_t, delta_t);

        // Convert h to linear for h+x selective output
        LogToLinearKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            BD, log_h_t, sign_h_t, h_linear);

        // h+x selective output: output = h * silu(h + x + b_gate)
        SelectiveOutputForward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_linear, x_t, b_gate, out_t, gate_t);
    }
    // Workspace is managed by caller - no cudaFree needed
}

// =============================================================================
// Log-Compute Full Elman Backward
// =============================================================================

template<typename T>
LogComputeFullElmanBackward<T>::LogComputeFullElmanBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void LogComputeFullElmanBackward<T>::Run(
    int steps,
    const T* W_x,
    const T* R_h,
    const T* W_delta,
    const T* b_gate,
    const T* x,
    const T* log_h,
    const T* sign_h,
    const T* v,
    const T* delta_cache,
    const T* gate_cache,
    const T* log_R_pos,
    const T* log_R_neg,
    const T* d_output,
    T* dx,
    T* dW_x,
    T* dR_h,
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
    // WORKSPACE LAYOUT: [dv: BD] [d_delta_raw: BD] [dh_recurrent: BD]
    //                   [dh_prev_linear: BD] [dh_linear: BD] [dx_gate: BD]
    //                   [h_linear: BD] [h_prev_linear: BD]
    //                   [db_float: dim floats] [db_delta_float: dim floats]
    //                   [db_gate_float: dim floats]
    // Total: 8*BD + 3*dim floats
    // ==========================================================================
    T* dv = workspace;
    T* d_delta_raw = workspace + BD;
    T* dh_recurrent = workspace + 2 * BD;
    T* dh_prev_linear = workspace + 3 * BD;
    T* dh_linear = workspace + 4 * BD;
    T* dx_gate = workspace + 5 * BD;
    T* h_linear = workspace + 6 * BD;
    T* h_prev_linear = workspace + 7 * BD;
    float* db_float = reinterpret_cast<float*>(workspace + 8 * BD);
    float* db_delta_float = db_float + dim_;
    float* db_gate_float = db_delta_float + dim_;

    // Initialize workspace (all async)
    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(T), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_delta_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_gate_float, 0, dim_ * sizeof(float), stream_);

    // Zero weight gradients (async)
    cudaMemsetAsync(dW_x, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dR_h, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dW_delta, 0, dim_ * dim_ * sizeof(T), stream_);

    for (int t = steps - 1; t >= 0; --t) {
        const T* x_t = x + t * BD;
        const T* log_h_prev = log_h + t * BD;
        const T* sign_h_prev = sign_h + t * BD;
        const T* log_h_t = log_h + (t + 1) * BD;
        const T* sign_h_t = sign_h + (t + 1) * BD;
        const T* v_t = v + t * BD;
        const T* delta_t = delta_cache + t * BD;
        const T* d_out_t = d_output + t * BD;
        T* dx_t = dx + t * BD;

        // Convert h_t and h_prev from log to linear
        LogToLinearKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            BD, log_h_t, sign_h_t, h_linear);
        LogToLinearKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            BD, log_h_prev, sign_h_prev, h_prev_linear);

        // Backward through h+x selective output: output = h * silu(h + x + b_gate)
        SelectiveOutputBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_linear, x_t, b_gate, d_out_t,
            dh_linear, dx_gate, db_gate_float);

        // Add recurrent gradient to dh
        if (t < steps - 1) {
            VectorAddInplace<T><<<num_blocks, block_size, 0, stream_>>>(
                BD, dh_recurrent, dh_linear);
        }

        // Backward through gated update
        LogComputeGatedBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, log_h_prev, sign_h_prev, v_t, delta_t, dh_linear,
            nullptr,  // dh_recurrent handled above
            dv, d_delta_raw, dh_prev_linear, db_float, db_delta_float);

        // dh_prev += dv @ R_h (backward of h @ R_h.T, but R_h is not transposed in forward)
        // For the term R_h @ h_prev in forward: d(R_h @ h)/dh = R_h.T
        // So dh_prev += dv @ R_h.T, which is CUBLAS_OP_T, CUBLAS_OP_N (keeping R_h as-is since dv is transposed)
        blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, R_h, dim_, dv, dim_, &alpha, dh_prev_linear, dim_);

        // dR_h += dv @ h_prev^T
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_, &alpha, dv, dim_, h_prev_linear, dim_, &alpha, dR_h, dim_);

        // dx = dv @ W_x + d_delta_raw @ W_delta + dx_gate
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, W_x, dim_, dv, dim_, &beta_zero, dx_t, dim_);
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, W_delta, dim_, d_delta_raw, dim_, &alpha, dx_t, dim_);
        VectorAddInplace<T><<<num_blocks, block_size, 0, stream_>>>(BD, dx_gate, dx_t);

        // dh_recurrent for next iteration (async)
        cudaMemcpyAsync(dh_recurrent, dh_prev_linear, BD * sizeof(T), cudaMemcpyDeviceToDevice, stream_);

        // Weight gradients
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_, &alpha, dv, dim_, x_t, dim_, &alpha, dW_x, dim_);
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_, &alpha, d_delta_raw, dim_, x_t, dim_, &alpha, dW_delta, dim_);
    }

    // Copy float gradients to T type using parallel kernel
    const int bias_blocks = (dim_ + block_size - 1) / block_size;
    CopyFloatToT<T><<<bias_blocks, block_size, 0, stream_>>>(dim_, db_float, db);
    CopyFloatToT<T><<<bias_blocks, block_size, 0, stream_>>>(dim_, db_delta_float, db_delta);
    CopyFloatToT<T><<<bias_blocks, block_size, 0, stream_>>>(dim_, db_gate_float, db_gate);
}

// Explicit template instantiations
template struct LogComputeFullElmanForward<__half>;
template struct LogComputeFullElmanForward<__nv_bfloat16>;
template struct LogComputeFullElmanForward<float>;
template struct LogComputeFullElmanForward<double>;

template struct LogComputeFullElmanBackward<__half>;
template struct LogComputeFullElmanBackward<__nv_bfloat16>;
template struct LogComputeFullElmanBackward<float>;
template struct LogComputeFullElmanBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
