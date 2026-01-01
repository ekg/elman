// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// Level 6: Log-Space Triple R - Full log-space with R_delta modulation
//
// This level uses three R matrices for full recurrence control:
// - R_h: Hidden state recurrence
// - R_x: Input transformation (replaces W_x for symmetry)
// - R_delta: Delta gate modulation from hidden state
//
// Recurrence:
//   v_t = R_x @ x + R_h @ h_{t-1} + b
//   delta_raw = W_delta @ x + R_delta @ h_{t-1} + b_delta
//   delta_t = sigmoid(delta_raw)
//   h_t = (1 - delta_t) * h_{t-1} + delta_t * tanh(v_t)
//   output_t = compete(h_t) * silu(W_out @ h_t)

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
constexpr float LOG_ZERO = -1e10f;
constexpr float LOG_EPS = 1e-10f;

// =============================================================================
// Device functions for signed log arithmetic
// =============================================================================

__device__ __forceinline__ void to_log_space(float x, float& log_x, float& sign_x) {
    sign_x = (x >= 0) ? 1.0f : -1.0f;
    float abs_x = fabsf(x);
    log_x = (abs_x > LOG_EPS) ? logf(abs_x) : LOG_ZERO;
}

__device__ __forceinline__ float from_log_space(float log_x, float sign_x) {
    if (log_x <= LOG_ZERO + 1.0f) return 0.0f;
    return sign_x * expf(log_x);
}

__device__ __forceinline__ void signed_log_add(
    float log_a, float sign_a,
    float log_b, float sign_b,
    float& log_result, float& sign_result) {

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
    float diff = min_log - max_log;

    bool a_is_max = log_a >= log_b;
    float sign_max = a_is_max ? sign_a : sign_b;
    float sign_min = a_is_max ? sign_b : sign_a;

    bool same_sign = sign_max * sign_min > 0;

    if (same_sign) {
        log_result = max_log + log1pf(expf(diff));
        sign_result = sign_max;
    } else {
        float exp_diff = expf(diff);
        if (exp_diff >= 0.9999999f) {
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
    }
    return val;
}

// =============================================================================
// Kernel: Decompose R matrix into log positive and negative parts
// =============================================================================

template<typename T>
__global__ void DecomposeRKernel(
    const int total_elements,
    const T* __restrict__ R,
    T* __restrict__ log_R_pos,
    T* __restrict__ log_R_neg) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
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
// Computes result = R @ v where v is stored as (log_v, sign_v)
// =============================================================================

template<typename T>
__global__ void LogSpaceMatVecKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ log_R_pos,
    const T* __restrict__ log_R_neg,
    const T* __restrict__ log_v,
    const T* __restrict__ sign_v,
    T* __restrict__ log_out,
    T* __restrict__ sign_out) {

    const int b = blockIdx.x;
    const int i = blockIdx.y;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    if (b >= batch_size || i >= dim) return;

    extern __shared__ float smem[];
    float* pos_contrib = smem;
    float* neg_contrib = smem + block_size;

    float local_log_pos = LOG_ZERO;
    float local_log_neg = LOG_ZERO;

    for (int j = tid; j < dim; j += block_size) {
        float log_R_p = static_cast<float>(log_R_pos[i * dim + j]);
        float log_R_n = static_cast<float>(log_R_neg[i * dim + j]);
        float log_v_j = static_cast<float>(log_v[b * dim + j]);
        float sign_v_j = static_cast<float>(sign_v[b * dim + j]);

        if (log_R_p > LOG_ZERO + 1.0f) {
            float log_contrib = log_R_p + log_v_j;
            if (sign_v_j > 0) {
                float max_val = fmaxf(local_log_pos, log_contrib);
                float min_val = fminf(local_log_pos, log_contrib);
                local_log_pos = max_val + log1pf(expf(min_val - max_val));
            } else {
                float max_val = fmaxf(local_log_neg, log_contrib);
                float min_val = fminf(local_log_neg, log_contrib);
                local_log_neg = max_val + log1pf(expf(min_val - max_val));
            }
        }

        if (log_R_n > LOG_ZERO + 1.0f) {
            float log_contrib = log_R_n + log_v_j;
            if (sign_v_j < 0) {
                float max_val = fmaxf(local_log_pos, log_contrib);
                float min_val = fminf(local_log_pos, log_contrib);
                local_log_pos = max_val + log1pf(expf(min_val - max_val));
            } else {
                float max_val = fmaxf(local_log_neg, log_contrib);
                float min_val = fminf(local_log_neg, log_contrib);
                local_log_neg = max_val + log1pf(expf(min_val - max_val));
            }
        }
    }

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
        float pos_val = pos_contrib[tid];
        float neg_val = neg_contrib[tid];

        // Grab values from upper half if block_size > 32
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

        if (tid == 0) {
            float log_result, sign_result;
            signed_log_add(pos_val, 1.0f, neg_val, -1.0f, log_result, sign_result);

            log_out[b * dim + i] = static_cast<T>(log_result);
            sign_out[b * dim + i] = static_cast<T>(sign_result);
        }
    }
}

// =============================================================================
// Kernel: Gated update with triple R matrices (OPTIMIZED: takes linear inputs)
// v_t = R_x @ x + R_h @ h_{t-1} + b
// delta_raw = W_delta @ x + b_delta  (R_delta removed - causes instability!)
// h_t = (1 - delta) * h_{t-1} + delta * tanh(v_t)
// =============================================================================

template<typename T>
__global__ void TripleRGatedUpdateKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ log_h_prev,
    const T* __restrict__ sign_h_prev,
    const T* __restrict__ Rx_x_linear,   // R_x @ x (already linear from cuBLAS!)
    const T* __restrict__ Rh_h_linear,   // R_h @ h_{t-1} (already linear from cuBLAS!)
    const T* __restrict__ Wdelta_x,      // W_delta @ x (linear)
    const T* __restrict__ b,
    const T* __restrict__ b_delta,
    T* __restrict__ log_h_out,
    T* __restrict__ sign_h_out,
    T* __restrict__ v_cache,
    T* __restrict__ delta_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // All inputs are already linear from cuBLAS GEMM - no conversion needed!
        float Rh_h = static_cast<float>(Rh_h_linear[idx]);
        float Rx_x = static_cast<float>(Rx_x_linear[idx]);

        // v_t = R_x @ x + R_h @ h_{t-1} + b
        float v_f = Rx_x + Rh_h + static_cast<float>(b[d]);
        if (v_cache) v_cache[idx] = static_cast<T>(v_f);
        float candidate = tanhf(v_f);

        // Delta gate: sigmoid(W_delta @ x + b_delta) - NO R_delta for stability!
        float delta_in = static_cast<float>(Wdelta_x[idx]) + static_cast<float>(b_delta[d]);
        float delta_f = 1.0f / (1.0f + expf(-delta_in));
        if (delta_cache) delta_cache[idx] = static_cast<T>(delta_f);

        // h_{t-1} in linear (still need to convert from log space)
        float h_prev_linear = from_log_space(
            static_cast<float>(log_h_prev[idx]),
            static_cast<float>(sign_h_prev[idx]));

        // Gated update: h_t = (1 - delta) * h_{t-1} + delta * candidate
        float h_new = (1.0f - delta_f) * h_prev_linear + delta_f * candidate;

        // Convert to log space
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
__global__ void SelectiveOutputKernel(
    const int batch_size,
    const int dim,
    const int n_groups,
    const int group_size,
    const T* __restrict__ log_h,
    const T* __restrict__ sign_h,
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
// Kernel: Convert log/sign to linear
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
// Kernel: Convert linear input x to log space for R_x multiplication
// =============================================================================

template<typename T>
__global__ void LinearToLogKernel(
    const int n,
    const T* __restrict__ x_linear,
    T* __restrict__ log_x,
    T* __restrict__ sign_x) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = static_cast<float>(x_linear[idx]);
        float log_val, sign_val;
        to_log_space(x, log_val, sign_val);
        log_x[idx] = static_cast<T>(log_val);
        sign_x[idx] = static_cast<T>(sign_val);
    }
}

// =============================================================================
// Backward kernels
// =============================================================================

// Backward through selective output
template<typename T>
__global__ void TripleRSelectiveOutputBackward(
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

// Backward through Triple R gated update
template<typename T>
__global__ void TripleRGatedBackward(
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

        float h_prev_linear = from_log_space(
            static_cast<float>(log_h_prev[idx]),
            static_cast<float>(sign_h_prev[idx]));

        float grad_h = static_cast<float>(dh[idx]);
        if (dh_recurrent) grad_h += static_cast<float>(dh_recurrent[idx]);

        float cand = tanhf(static_cast<float>(v[idx]));
        float del = static_cast<float>(delta[idx]);
        float one_minus_del = 1.0f - del;

        float d_cand = grad_h * del;
        float dtanh = 1.0f - cand * cand;
        float dv_val = d_cand * dtanh;
        dv[idx] = static_cast<T>(dv_val);

        float d_delta = grad_h * (cand - h_prev_linear);
        float dsigmoid = del * one_minus_del;
        float d_delta_raw_val = d_delta * dsigmoid;
        d_delta_raw[idx] = static_cast<T>(d_delta_raw_val);

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

}  // anonymous namespace


namespace hasty {
namespace v0 {
namespace elman_ladder {

// =============================================================================
// Log-Space Triple R Forward
// =============================================================================

template<typename T>
LogSpaceTripleRForward<T>::LogSpaceTripleRForward(
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
void LogSpaceTripleRForward<T>::Run(
    int steps,
    const T* R_h,
    const T* R_x,
    const T* R_delta,
    const T* W_delta,
    const T* W_out,
    const T* b,
    const T* b_delta,
    const T* x,
    T* log_h,
    T* sign_h,
    T* output,
    T* v,
    T* delta_cache,
    T* compete_cache,
    T* workspace) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int BD = batch_size_ * dim_;
    const int TBD = steps * BD;  // Total batch*dim for all timesteps
    const int DD = dim_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;
    const int total_blocks = (TBD + block_size - 1) / block_size;
    const int group_size = dim_ / n_groups_;

    // =========================================================================
    // WORKSPACE LAYOUT (OPTIMIZED - uses cuBLAS GEMM instead of log-space matmul!):
    // Input projections:   [all_Rx_x: TBD] [all_Wdelta_x: TBD]
    // Per-step scratch:    [Rh_h_linear: BD] [h_prev_linear: BD] [w_out_h: BD] [h_linear: BD]
    // Total: 2*TBD + 4*BD  (R_delta removed for stability)
    // =========================================================================
    T* all_Rx_x = workspace;
    T* all_Wdelta_x = workspace + TBD;

    T* Rh_h_linear = workspace + 2 * TBD;
    T* h_prev_linear = workspace + 2 * TBD + BD;
    T* w_out_h = workspace + 2 * TBD + 2 * BD;
    T* h_linear = workspace + 2 * TBD + 3 * BD;

    // =========================================================================
    // Pre-compute ALL input projections using cuBLAS (tensor-core accelerated!)
    // No log-space decomposition needed - we stay in linear space!
    // =========================================================================

    // Pre-compute R_x @ x for ALL timesteps in one GEMM (x is linear input)
    blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_, &alpha, R_x, dim_, x, dim_, &beta_zero, all_Rx_x, dim_);

    // Pre-compute W_delta @ x for ALL timesteps in one GEMM
    blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_, &alpha, W_delta, dim_, x, dim_, &beta_zero, all_Wdelta_x, dim_);

    // =========================================================================
    // Sequential loop: Use cuBLAS for recurrent operations
    // OPTIMIZATION: Convert h to linear once, then use tensor-core accelerated GEMM
    // This is O(B*D) transcendentals instead of O(B*D^2) in LogSpaceMatVecKernel!
    // =========================================================================

    for (int t = 0; t < steps; ++t) {
        const T* log_h_prev = log_h + t * BD;
        const T* sign_h_prev = sign_h + t * BD;
        T* log_h_t = log_h + (t + 1) * BD;
        T* sign_h_t = sign_h + (t + 1) * BD;
        T* out_t = output + t * BD;
        T* v_t = training_ ? (v + t * BD) : nullptr;
        T* delta_t = training_ ? (delta_cache + t * BD) : nullptr;
        T* compete_t = training_ ? (compete_cache + t * BD) : nullptr;

        // Get pre-computed projections for this timestep
        const T* Rx_x_t = all_Rx_x + t * BD;
        const T* Wdelta_x_t = all_Wdelta_x + t * BD;

        // OPTIMIZED: Convert h_prev to linear ONCE, then use cuBLAS GEMM for both R matrices
        LogToLinearKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            BD, log_h_prev, sign_h_prev, h_prev_linear);

        // R_h @ h_prev using cuBLAS (tensor-core accelerated!)
        blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, R_h, dim_, h_prev_linear, dim_, &beta_zero, Rh_h_linear, dim_);

        // NOTE: R_delta @ h_prev REMOVED - causes training instability!

        // Gated update (now takes linear inputs from cuBLAS!)
        TripleRGatedUpdateKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, log_h_prev, sign_h_prev,
            Rx_x_t, Rh_h_linear,
            Wdelta_x_t,
            b, b_delta, log_h_t, sign_h_t, v_t, delta_t);

        // Convert h_t to linear for W_out
        LogToLinearKernel<T><<<num_blocks, block_size, 0, stream_>>>(BD, log_h_t, sign_h_t, h_linear);

        // h_linear @ W_out.T (depends on h_t, can't pre-compute)
        blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, W_out, dim_, h_linear, dim_, &beta_zero, w_out_h, dim_);

        // Selective output
        dim3 out_grid(batch_size_, n_groups_);
        int out_smem_size = 2 * block_size * sizeof(float);
        SelectiveOutputKernel<T><<<out_grid, block_size, out_smem_size, stream_>>>(
            batch_size_, dim_, n_groups_, group_size,
            log_h_t, sign_h_t, w_out_h, out_t, compete_t);
    }
    // Workspace is managed by caller - no cudaFree needed
}

// =============================================================================
// Log-Space Triple R Backward
// =============================================================================

template<typename T>
LogSpaceTripleRBackward<T>::LogSpaceTripleRBackward(
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
void LogSpaceTripleRBackward<T>::Run(
    int steps,
    const T* R_h,
    const T* R_x,
    const T* R_delta,
    const T* W_delta,
    const T* W_out,
    const T* x,
    const T* log_h,
    const T* sign_h,
    const T* v,
    const T* delta_cache,
    const T* compete_cache,
    const T* d_output,
    T* dx,
    T* dR_h,
    T* dR_x,
    T* dR_delta,
    T* dW_delta,
    T* dW_out,
    T* db,
    T* db_delta,
    T* workspace) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;
    const int group_size = dim_ / n_groups_;

    // Workspace layout: 10 * BD * sizeof(T) + 2 * dim_ * sizeof(float)
    T* dv = workspace;
    T* d_delta_raw = dv + BD;
    T* dh_recurrent = d_delta_raw + BD;
    T* dh_prev_linear = dh_recurrent + BD;
    T* dh_linear = dh_prev_linear + BD;
    T* d_w_out_h = dh_linear + BD;
    T* w_out_h = d_w_out_h + BD;
    T* h_linear = w_out_h + BD;
    T* h_prev_linear = h_linear + BD;
    T* x_linear = h_prev_linear + BD;

    // Float buffers for atomic gradients (after T buffers)
    float* db_float = reinterpret_cast<float*>(x_linear + BD);
    float* db_delta_float = db_float + dim_;

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(T), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_delta_float, 0, dim_ * sizeof(float), stream_);

    // Zero weight gradients
    cudaMemsetAsync(dR_h, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dR_x, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dR_delta, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dW_delta, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dW_out, 0, dim_ * dim_ * sizeof(T), stream_);

    for (int t = steps - 1; t >= 0; --t) {
        const T* x_t = x + t * BD;
        const T* log_h_prev = log_h + t * BD;
        const T* sign_h_prev = sign_h + t * BD;
        const T* log_h_t = log_h + (t + 1) * BD;
        const T* sign_h_t = sign_h + (t + 1) * BD;
        const T* v_t = v + t * BD;
        const T* delta_t = delta_cache + t * BD;
        const T* compete_t = compete_cache + t * BD;
        const T* d_out_t = d_output + t * BD;
        T* dx_t = dx + t * BD;

        // Convert h_t and h_prev from log to linear
        LogToLinearKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            BD, log_h_t, sign_h_t, h_linear);
        LogToLinearKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            BD, log_h_prev, sign_h_prev, h_prev_linear);

        // Copy x_t to workspace (already linear)
        cudaMemcpyAsync(x_linear, x_t, BD * sizeof(T), cudaMemcpyDeviceToDevice, stream_);

        // Recompute w_out_h = h_linear @ W_out.T (matching forward)
        blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, W_out, dim_, h_linear, dim_, &beta_zero, w_out_h, dim_);

        // Backward through selective output
        dim3 grid(batch_size_, n_groups_);
        int smem_size = block_size * sizeof(float);
        TripleRSelectiveOutputBackward<T><<<grid, block_size, smem_size, stream_>>>(
            batch_size_, dim_, n_groups_, group_size, log_h_t, sign_h_t, w_out_h, compete_t,
            d_out_t, dh_linear, d_w_out_h);

        // dW_out += h_linear @ d_w_out_h^T  (for Y = W.T @ X, dW = X @ dY.T)
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_, &alpha, h_linear, dim_, d_w_out_h, dim_, &alpha, dW_out, dim_);

        // dh_linear += d_w_out_h @ W_out (for Y = h @ W_out.T, dX = dY @ W_out)
        blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, W_out, dim_, d_w_out_h, dim_, &alpha, dh_linear, dim_);

        // Backward through gated update
        TripleRGatedBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, log_h_prev, sign_h_prev, v_t, delta_t, dh_linear,
            (t < steps - 1) ? dh_recurrent : nullptr,
            dv, d_delta_raw, dh_prev_linear, db_float, db_delta_float);

        // dh_prev += dv @ R_h (for Y = h @ R_h.T, dX = dY @ R_h)
        blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, R_h, dim_, dv, dim_, &alpha, dh_prev_linear, dim_);

        // NOTE: R_delta backward REMOVED - R_delta not used in forward anymore!

        // dR_h += h_prev @ dv^T  (for Y = W.T @ X, dW = X @ dY.T)
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_, &alpha, h_prev_linear, dim_, dv, dim_, &alpha, dR_h, dim_);

        // dR_x += x @ dv^T  (for Y = W.T @ X, dW = X @ dY.T)
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_, &alpha, x_linear, dim_, dv, dim_, &alpha, dR_x, dim_);

        // dx = dv @ R_x (for Y = x @ R_x.T, dX = dY @ R_x) + d_delta_raw @ W_delta
        // In cuBLAS col-major: dx_T = R_x.T @ dv.T, so dx = (R_x.T @ dv.T).T = dv @ R_x
        blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, R_x, dim_, dv, dim_, &beta_zero, dx_t, dim_);
        blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, W_delta, dim_, d_delta_raw, dim_, &alpha, dx_t, dim_);

        // dW_delta += x @ d_delta_raw^T  (for Y = W.T @ X, dW = X @ dY.T)
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_, &alpha, x_linear, dim_, d_delta_raw, dim_, &alpha, dW_delta, dim_);

        // dh_recurrent for next iteration
        cudaMemcpyAsync(dh_recurrent, dh_prev_linear, BD * sizeof(T), cudaMemcpyDeviceToDevice, stream_);
    }

    // Copy float gradients to output type using kernel
    const int copy_blocks = (dim_ + block_size - 1) / block_size;
    CopyFloatToT<T><<<copy_blocks, block_size, 0, stream_>>>(dim_, db_float, db);
    CopyFloatToT<T><<<copy_blocks, block_size, 0, stream_>>>(dim_, db_delta_float, db_delta);
}

// Explicit template instantiations
template struct LogSpaceTripleRForward<__half>;
template struct LogSpaceTripleRForward<__nv_bfloat16>;
template struct LogSpaceTripleRForward<float>;
template struct LogSpaceTripleRForward<double>;

template struct LogSpaceTripleRBackward<__half>;
template struct LogSpaceTripleRBackward<__nv_bfloat16>;
template struct LogSpaceTripleRBackward<float>;
template struct LogSpaceTripleRBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
