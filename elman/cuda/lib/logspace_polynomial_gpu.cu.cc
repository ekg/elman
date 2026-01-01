// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// Log-Space Polynomial Elman - True log-space RNN with polynomial activation
//
// Key innovations:
// 1. INPUT-DEPENDENT alpha: α_t = 1 + softplus(W_α @ x_t + b_α)  (guaranteed > 1)
// 2. Polynomial activation: log|h| = α * log|v|  (constant gradient α per element)
// 3. Soft bounding: log_bounded = -softplus(-log_h)  (magnitude ≤ 1, gradient ∈ [0,1])
// 4. All operations stay in log-space with bounded gradients
//
// Architecture:
//   α_t = 1 + softplus(W_α @ x_t + b_α)      -- input-dependent exponent, always > 1
//   v = r_h * h_prev + W_x @ x + b           -- pre-activation via signed log arithmetic
//   log|h_unbounded| = α_t * log|v|          -- polynomial nonlinearity (different α per element!)
//   log|h| = -softplus(-log|h_unbounded|)    -- soft bound magnitude to ≤ 1
//   sign(h) = sign(v)
//
// Gradient properties:
//   - Through softplus bound: sigmoid(-x) ∈ [0,1]
//   - Through polynomial: α (input-dependent but bounded > 1)
//   - Through logsumexp aggregation: softmax weights ∈ [0,1]
//   - α > 1 creates attractor dynamics: small → 0, large → 1

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cfloat>

#include "blas.h"
#include "inline_ops.h"

namespace {

// Constants for log-space computation
constexpr float LOG_ZERO = -40.0f;     // Represents log(~0), not too extreme for gradients
constexpr float LOG_EPS = 1e-10f;      // Small epsilon for log stability
constexpr float GRAD_CLIP = 1.0f;      // Aggressive gradient clipping for stability

// =============================================================================
// Device functions for signed log arithmetic
// =============================================================================

// Convert linear value to log space: x -> (log|x|, sign(x))
// Uses soft-abs: sqrt(x^2 + eps^2) to bound gradients near zero
// The gradient d/dx sqrt(x^2 + eps^2) = x / sqrt(x^2 + eps^2) is bounded by 1
constexpr float SOFT_ABS_EPS = 1e-6f;

__device__ __forceinline__ void to_log_space(float x, float& log_x, float& sign_x) {
    sign_x = (x >= 0.0f) ? 1.0f : -1.0f;
    // Soft-abs bounds gradient to 1 near zero, preventing explosion
    float soft_abs_x = sqrtf(x * x + SOFT_ABS_EPS * SOFT_ABS_EPS);
    log_x = logf(soft_abs_x);
}

// Convert log space to linear: (log|x|, sign(x)) -> x
__device__ __forceinline__ float from_log_space(float log_x, float sign_x) {
    if (log_x <= LOG_ZERO + 1.0f) return 0.0f;
    float result = sign_x * expf(fminf(log_x, 20.0f));  // Clamp to prevent overflow
    return result;
}

// Simple element-wise add kernel
template<typename T>
__global__ void AddKernel(const int n, const T* a, const T* b, T* c) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = static_cast<T>(static_cast<float>(a[idx]) + static_cast<float>(b[idx]));
    }
}

// Add bias kernel: result[b,d] = input[b,d] + bias[d]
template<typename T>
__global__ void AddBiasKernel(const int batch_size, const int dim,
                               const T* __restrict__ input,
                               const T* __restrict__ bias,
                               T* __restrict__ output) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;
    if (idx < total) {
        const int d = idx % dim;
        output[idx] = static_cast<T>(static_cast<float>(input[idx]) + static_cast<float>(bias[d]));
    }
}

// Signed log addition: compute (log|a+b|, sign(a+b)) from (log|a|, sign_a) and (log|b|, sign_b)
// Returns the softmax weight w_a = |a| / |a+b| for gradient computation
__device__ __forceinline__ void signed_log_add(
    float log_a, float sign_a,
    float log_b, float sign_b,
    float& log_result, float& sign_result,
    float& weight_a) {

    // Handle cases where one input is effectively zero
    if (log_a <= LOG_ZERO + 1.0f) {
        log_result = log_b;
        sign_result = sign_b;
        weight_a = 0.0f;
        return;
    }
    if (log_b <= LOG_ZERO + 1.0f) {
        log_result = log_a;
        sign_result = sign_a;
        weight_a = 1.0f;
        return;
    }

    float max_log = fmaxf(log_a, log_b);
    float min_log = fminf(log_a, log_b);
    float diff = min_log - max_log;  // Always <= 0

    bool a_is_max = log_a >= log_b;
    float sign_max = a_is_max ? sign_a : sign_b;
    float sign_min = a_is_max ? sign_b : sign_a;

    bool same_sign = sign_max * sign_min > 0.0f;

    if (same_sign) {
        // log(|a| + |b|) = max + log(1 + exp(diff))
        log_result = max_log + log1pf(expf(diff));
        sign_result = sign_max;
        // weight_a = |a| / (|a| + |b|) = softmax weight, bounded [0,1]
        weight_a = expf(log_a - log_result);
    } else {
        // log(||a| - |b||) = max + log(1 - exp(diff))
        float exp_diff = expf(diff);
        if (exp_diff >= 0.9999f) {
            // Near complete cancellation
            log_result = LOG_ZERO;
            sign_result = 1.0f;
            weight_a = 0.5f;
        } else {
            log_result = max_log + log1pf(-exp_diff);
            sign_result = sign_max;
            // For opposite signs, gradient can exceed 1 near cancellation
            float abs_result = expf(log_result);
            float abs_a = expf(log_a);
            weight_a = (abs_result > LOG_EPS) ? fminf(abs_a / abs_result, GRAD_CLIP) : 1.0f;
        }
    }
    weight_a = fminf(fmaxf(weight_a, 0.0f), GRAD_CLIP);
}

// Soft upper bound at magnitude 1: log_bounded = -softplus(-log_h)
// Maps log_h -> log_bounded such that exp(log_bounded) <= 1
__device__ __forceinline__ float soft_bound_log(float log_h) {
    if (log_h > 0.0f) {
        // log_h > 0 means |h| > 1, need to squash
        return -log1pf(expf(-log_h));
    } else {
        // log_h <= 0 means |h| <= 1, mostly preserve
        return log_h - log1pf(expf(log_h));
    }
}

// Gradient of soft_bound: d(-softplus(-x))/dx = sigmoid(-x) ∈ [0, 1]
__device__ __forceinline__ float soft_bound_grad(float log_h) {
    return 1.0f / (1.0f + expf(log_h));  // sigmoid(-log_h)
}

// softplus(x) = log(1 + exp(x)), numerically stable
__device__ __forceinline__ float stable_softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return 0.0f;
    return log1pf(expf(x));
}

// d/dx softplus(x) = sigmoid(x)
__device__ __forceinline__ float softplus_grad(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// =============================================================================
// Forward Kernel: Log-Space Polynomial Gated Update with Input-Dependent Alpha
// =============================================================================

template<typename T>
__global__ void LogPolyGatedUpdateKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ log_h_prev,       // [B, dim] log|h_prev|
    const T* __restrict__ sign_h_prev,      // [B, dim] sign(h_prev)
    const T* __restrict__ wx_x,             // [B, dim] W_x @ x (pre-computed, linear space)
    const T* __restrict__ alpha_raw,        // [B, dim] W_alpha @ x + b_alpha (pre-softplus)
    const T* __restrict__ log_r_h,          // [dim] log|r_h| (diagonal recurrence)
    const T* __restrict__ sign_r_h,         // [dim] sign(r_h)
    const T* __restrict__ b,                // [dim] bias (linear space)
    const T* __restrict__ delta_raw,        // [B, dim] W_delta @ x + b_delta (for gate)
    T* __restrict__ log_h_out,              // [B, dim] output log|h|
    T* __restrict__ sign_h_out,             // [B, dim] output sign(h)
    // Caches for backward pass
    T* __restrict__ log_v_cache,            // [B, dim] log|v| before polynomial
    T* __restrict__ sign_v_cache,           // [B, dim] sign(v)
    T* __restrict__ alpha_cache,            // [B, dim] computed alpha = 1 + softplus(alpha_raw)
    T* __restrict__ log_h_unbounded_cache,  // [B, dim] log|h| before bounding
    T* __restrict__ delta_cache,            // [B, dim] sigmoid(delta_raw)
    T* __restrict__ weight_rh_cache) {      // [B, dim] softmax weight for r_h*h term

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // === Load inputs ===
        float log_hp = static_cast<float>(log_h_prev[idx]);
        float sign_hp = static_cast<float>(sign_h_prev[idx]);
        // Clamp log_r_h to <= -0.1 (so r_h <= 0.9) for stability
        float log_rh = fminf(static_cast<float>(log_r_h[d]), -0.1f);
        float sign_rh = static_cast<float>(sign_r_h[d]);
        float wx = static_cast<float>(wx_x[idx]);
        float bias = static_cast<float>(b[d]);

        // === Compute input-dependent alpha ===
        // α = 1 + softplus(W_alpha @ x + b_alpha), capped at 2.0 for stability
        float alpha_raw_val = static_cast<float>(alpha_raw[idx]);
        float alpha = 1.0f + fminf(stable_softplus(alpha_raw_val), 1.0f);
        if (alpha_cache) alpha_cache[idx] = static_cast<T>(alpha);

        // === Compute r_h * h_prev in log space ===
        float log_rh_hp = log_rh + log_hp;
        float sign_rh_hp = sign_rh * sign_hp;

        // === Convert W_x @ x + b to log space ===
        float linear_input = wx + bias;
        float log_input, sign_input;
        to_log_space(linear_input, log_input, sign_input);

        // === Add: v = r_h * h_prev + (W_x @ x + b) ===
        float log_v, sign_v, weight_rh;
        signed_log_add(log_rh_hp, sign_rh_hp, log_input, sign_input,
                       log_v, sign_v, weight_rh);

        // Cache for backward
        if (log_v_cache) log_v_cache[idx] = static_cast<T>(log_v);
        if (sign_v_cache) sign_v_cache[idx] = static_cast<T>(sign_v);
        if (weight_rh_cache) weight_rh_cache[idx] = static_cast<T>(weight_rh);

        // === Polynomial activation: h_candidate = sign(v) * |v|^α ===
        // In log space: log|h_cand| = α * log|v|
        float log_cand = alpha * log_v;
        float sign_cand = sign_v;

        // === Soft bound to magnitude <= 1 ===
        float log_cand_bounded = soft_bound_log(log_cand);

        // Cache unbounded for backward
        if (log_h_unbounded_cache) log_h_unbounded_cache[idx] = static_cast<T>(log_cand);

        // === Delta gate (in linear space, like Mamba2) ===
        float delta_raw_val = static_cast<float>(delta_raw[idx]);
        float delta = 1.0f / (1.0f + expf(-delta_raw_val));  // sigmoid
        if (delta_cache) delta_cache[idx] = static_cast<T>(delta);

        // log(1-delta) and log(delta) for log-space gated update
        float log_one_minus_delta = -log1pf(expf(delta_raw_val));
        float log_delta = -log1pf(expf(-delta_raw_val));

        // === Gated update in log space: h = (1-δ)*h_prev + δ*candidate ===
        float log_term1 = log_one_minus_delta + log_hp;
        float sign_term1 = sign_hp;

        float log_term2 = log_delta + log_cand_bounded;
        float sign_term2 = sign_cand;

        float log_h_new, sign_h_new, weight_decay;
        signed_log_add(log_term1, sign_term1, log_term2, sign_term2,
                       log_h_new, sign_h_new, weight_decay);

        // === Clamp hidden state to prevent explosion ===
        log_h_new = fminf(fmaxf(log_h_new, -20.0f), 10.0f);
        if (!isfinite(log_h_new)) log_h_new = 0.0f;

        // === Output ===
        log_h_out[idx] = static_cast<T>(log_h_new);
        sign_h_out[idx] = static_cast<T>(sign_h_new);
    }
}

// =============================================================================
// Output Kernel: Convert log space to linear for output projection
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
// Selective Output Kernel (compete × silu) - Copied from log_1 for stability
// =============================================================================

template<typename T>
__global__ void LogSelectiveOutput(
    const int batch_size,
    const int dim,
    const int n_groups,
    const int group_size,
    const T* __restrict__ h_linear,   // [B, dim] in linear space (after RMSNorm)
    const T* __restrict__ w_out_h,    // [B, dim] W_out @ h
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
        max_val = fmaxf(max_val, static_cast<float>(h_linear[base + i]));
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
        sum += expf(static_cast<float>(h_linear[base + i]) - max_val);
    }
    float* sum_smem = smem + blockDim.x;
    sum_smem[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sum_smem[threadIdx.x] += sum_smem[threadIdx.x + s];
        __syncthreads();
    }
    sum = sum_smem[0];

    // output = compete * silu(w_out_h)
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float compete = expf(static_cast<float>(h_linear[base + i]) - max_val) / sum;
        if (compete_cache) compete_cache[base + i] = static_cast<T>(compete);

        float w = static_cast<float>(w_out_h[base + i]);
        float silu_val = w / (1.0f + expf(-w));
        output[base + i] = static_cast<T>(compete * silu_val);
    }
}

// =============================================================================
// Selective Output Backward Kernel
// =============================================================================

template<typename T>
__global__ void LogSelectiveOutputBackward(
    const int batch_size,
    const int dim,
    const int n_groups,
    const int group_size,
    const T* __restrict__ h_linear,
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

// =============================================================================
// Fused LogSpace RMSNorm + Linear Conversion Kernel
// Uses logsumexp for bounded gradients (softmax weights in [0,1])
// =============================================================================

template<typename T, int BLOCK_DIM = 256>
__global__ void LogSpaceRMSNormKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ log_h,      // [B, dim] log|h|
    const T* __restrict__ sign_h,     // [B, dim] sign(h)
    const T* __restrict__ log_gamma,  // [dim] learnable scale
    T* __restrict__ h_linear,         // [B, dim] output
    T* __restrict__ log_rms_cache) {  // [B, 1] cached for backward

    // Each block handles one batch element
    const int b = blockIdx.x;
    if (b >= batch_size) return;

    extern __shared__ float sdata[];

    // Step 1: Compute max for numerical stability (each thread handles multiple elements)
    float thread_max = -1e30f;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float log_h2 = 2.0f * static_cast<float>(log_h[b * dim + d]);
        thread_max = fmaxf(thread_max, log_h2);
    }

    // Block-wide max reduction
    sdata[threadIdx.x] = thread_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }
    float max_val = sdata[0];

    // Step 2: Compute sum of exp(log_h2 - max) for logsumexp
    float thread_sum = 0.0f;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float log_h2 = 2.0f * static_cast<float>(log_h[b * dim + d]);
        thread_sum += expf(log_h2 - max_val);
    }

    // Block-wide sum reduction
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    float sum_exp = sdata[0];

    // logsumexp(2*log_h) = max + log(sum_exp)
    float log_sum_h2 = max_val + logf(sum_exp);

    // log(rms) = log(mean(h^2)) / 2 = (log_sum_h2 - log(dim)) / 2
    float log_rms = (log_sum_h2 - logf(static_cast<float>(dim))) * 0.5f;

    // Cache log_rms for backward (only thread 0 writes)
    if (threadIdx.x == 0 && log_rms_cache != nullptr) {
        log_rms_cache[b] = static_cast<T>(log_rms);
    }

    // Step 3: Apply normalization and convert to linear
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        int idx = b * dim + d;
        float log_h_val = static_cast<float>(log_h[idx]);
        float sign_val = static_cast<float>(sign_h[idx]);
        float gamma = static_cast<float>(log_gamma[d]);

        // Normalized log: log_h - log_rms + log_gamma
        float log_normed = log_h_val - log_rms + gamma;

        // Convert to linear with clamping
        float clamped = fminf(fmaxf(log_normed, -40.0f), 20.0f);
        h_linear[idx] = static_cast<T>(sign_val * expf(clamped));
    }
}

// =============================================================================
// Backward Kernel: RMSNorm in log-space
// Gradient flow: d_h_linear -> d_log_h (+ d_log_gamma)
// Key insight: logsumexp gradient is softmax, giving bounded gradients!
// =============================================================================

template<typename T, int BLOCK_DIM = 256>
__global__ void LogSpaceRMSNormBackwardKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ log_h,          // [B, dim] log|h| (input to RMSNorm)
    const T* __restrict__ sign_h,         // [B, dim] sign(h)
    const T* __restrict__ log_gamma,      // [dim] learnable scale
    const T* __restrict__ h_linear,       // [B, dim] output from forward
    const T* __restrict__ d_h_linear,     // [B, dim] incoming gradient
    T* __restrict__ d_log_h,              // [B, dim] gradient w.r.t. log_h
    float* __restrict__ d_log_gamma) {    // [dim] gradient w.r.t. log_gamma (atomic add)

    // Each block handles one batch element
    const int b = blockIdx.x;
    if (b >= batch_size) return;

    extern __shared__ float sdata[];
    // sdata layout: [0..BLOCK_DIM-1] for reductions, [BLOCK_DIM..2*BLOCK_DIM-1] for softmax

    // Step 1: Compute logsumexp for softmax weights (same as forward)
    float thread_max = -1e30f;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float log_h2 = 2.0f * static_cast<float>(log_h[b * dim + d]);
        thread_max = fmaxf(thread_max, log_h2);
    }

    sdata[threadIdx.x] = thread_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }
    float max_val = sdata[0];

    float thread_sum = 0.0f;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float log_h2 = 2.0f * static_cast<float>(log_h[b * dim + d]);
        thread_sum += expf(log_h2 - max_val);
    }

    sdata[threadIdx.x] = thread_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    float sum_exp = sdata[0];

    // Step 2: Compute d_log_h_normed = d_h_linear * h_linear (chain rule for exp)
    // and sum for d_log_rms
    float thread_d_log_rms = 0.0f;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        int idx = b * dim + d;
        float h_lin = static_cast<float>(h_linear[idx]);
        float d_h_lin = static_cast<float>(d_h_linear[idx]);

        // d/d(log_normed) exp(log_normed) = exp(log_normed) = h_linear/sign
        // But h_linear = sign * exp(log_normed), so d_log_normed = d_h_linear * h_linear
        float d_log_normed = d_h_lin * h_lin;

        // d_log_gamma: accumulate with atomic (scaled by 0.0001 to match Python)
        atomicAdd(&d_log_gamma[d], d_log_normed * 0.0001f);

        // d_log_rms = -sum(d_log_normed) over dim
        thread_d_log_rms -= d_log_normed;
    }

    // Reduce d_log_rms across threads
    sdata[threadIdx.x] = thread_d_log_rms;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    float d_log_rms = sdata[0];

    // Step 3: Backward through logsumexp
    // d/d(log_h[i]) logsumexp(2*log_h) = softmax(2*log_h)[i] * 2
    // d/d(log_h[i]) log_rms = d/d(log_h[i]) (logsumexp(2*log_h)/2 - log(dim)/2)
    //                       = softmax(2*log_h)[i]
    // d_log_h from log_rms path = d_log_rms * softmax(2*log_h)
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        int idx = b * dim + d;
        float log_h2 = 2.0f * static_cast<float>(log_h[idx]);
        float h_lin = static_cast<float>(h_linear[idx]);
        float d_h_lin = static_cast<float>(d_h_linear[idx]);

        // Softmax weight for this element
        float softmax_w = expf(log_h2 - max_val) / sum_exp;

        // Direct gradient: d_log_h from normalization = d_log_normed
        float d_log_normed = d_h_lin * h_lin;

        // Gradient through log_rms: d_log_rms * softmax
        float d_from_rms = d_log_rms * softmax_w;

        // Total gradient (scaled by 0.0001 to match Python grad_scale and stabilize)
        d_log_h[idx] = static_cast<T>((d_log_normed + d_from_rms) * 0.0001f);
    }
}

// =============================================================================
// Backward Kernel: Gradient through log-space polynomial update
// =============================================================================

template<typename T>
__global__ void LogPolyGatedBackwardKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ log_h_prev,
    const T* __restrict__ sign_h_prev,
    const T* __restrict__ log_v,
    const T* __restrict__ sign_v,
    const T* __restrict__ alpha,            // Cached: 1 + softplus(alpha_raw)
    const T* __restrict__ alpha_raw,        // For gradient through softplus
    const T* __restrict__ log_h_unbounded,
    const T* __restrict__ delta,
    const T* __restrict__ weight_rh,
    const T* __restrict__ log_r_h,
    const T* __restrict__ sign_r_h,
    const T* __restrict__ d_log_h,
    const T* __restrict__ d_log_h_recurrent,
    T* __restrict__ d_log_h_prev,
    T* __restrict__ d_wx_x,
    T* __restrict__ d_alpha_raw,            // Gradient w.r.t. alpha pre-softplus
    T* __restrict__ d_delta_raw,
    float* __restrict__ d_log_r_h,
    float* __restrict__ d_b) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // Load gradients with NaN/Inf protection
        float grad_log_h = static_cast<float>(d_log_h[idx]);
        if (d_log_h_recurrent) {
            grad_log_h += static_cast<float>(d_log_h_recurrent[idx]);
        }
        // Sanitize: replace NaN/Inf with 0
        if (!isfinite(grad_log_h)) grad_log_h = 0.0f;
        // Early clipping of incoming gradient
        grad_log_h = fminf(fmaxf(grad_log_h, -1.0f), 1.0f);

        // Load cached values with NaN/Inf protection and clamping
        float log_hp = static_cast<float>(log_h_prev[idx]);
        float sign_hp = static_cast<float>(sign_h_prev[idx]);
        float log_v_val = static_cast<float>(log_v[idx]);
        float sign_v_val = static_cast<float>(sign_v[idx]);
        float alpha_val = static_cast<float>(alpha[idx]);
        float alpha_raw_val = static_cast<float>(alpha_raw[idx]);
        float log_h_unb = static_cast<float>(log_h_unbounded[idx]);
        float del = static_cast<float>(delta[idx]);
        float w_rh = static_cast<float>(weight_rh[idx]);

        // Sanitize cached values
        if (!isfinite(log_hp)) log_hp = LOG_ZERO;
        if (!isfinite(log_v_val)) log_v_val = LOG_ZERO;
        if (!isfinite(log_h_unb)) log_h_unb = LOG_ZERO;
        log_hp = fminf(fmaxf(log_hp, -40.0f), 20.0f);
        log_v_val = fminf(fmaxf(log_v_val, -40.0f), 20.0f);
        log_h_unb = fminf(fmaxf(log_h_unb, -40.0f), 20.0f);

        // Clamp log_r_h to <= -0.1 for stability (must match forward)
        float log_rh = fminf(static_cast<float>(log_r_h[d]), -0.1f);

        float one_minus_delta = 1.0f - del;

        // === Backward through gated update ===
        // Gradient to candidate term (through delta gate)
        float d_log_cand_bounded = grad_log_h * del;

        // Gradient to h_prev term (through decay path)
        float d_log_hp_decay = grad_log_h * one_minus_delta;

        // === Backward through soft bound ===
        // d/d(log_h_unb) = sigmoid(-log_h_unb) ∈ [0, 1]
        float bound_grad = soft_bound_grad(log_h_unb);
        float d_log_cand = d_log_cand_bounded * bound_grad;

        // === Backward through polynomial: log_cand = alpha * log_v ===
        // d_log_v = alpha * d_log_cand
        // d_alpha = log_v * d_log_cand
        // Clip d_log_cand first to prevent explosion
        float d_log_cand_clipped = fminf(fmaxf(d_log_cand, -10.0f), 10.0f);
        float d_log_v = alpha_val * d_log_cand_clipped;
        // Clip log_v when computing d_alpha to prevent explosion from extreme log values
        float log_v_clipped = fminf(fmaxf(log_v_val, -20.0f), 20.0f);
        float d_alpha = log_v_clipped * d_log_cand_clipped;

        // Gradient through alpha = 1 + softplus(alpha_raw)
        // d_alpha_raw = d_alpha * sigmoid(alpha_raw)
        float d_alpha_raw_val = d_alpha * softplus_grad(alpha_raw_val);
        d_alpha_raw[idx] = static_cast<T>(fminf(fmaxf(d_alpha_raw_val, -GRAD_CLIP), GRAD_CLIP));

        // === Backward through signed log addition v = r_h*h_prev + input ===
        // Clip d_log_v to prevent gradient explosion
        float d_log_v_clipped = fminf(fmaxf(d_log_v, -10.0f), 10.0f);
        float d_log_rh_hp = d_log_v_clipped * w_rh;
        float d_log_input = d_log_v_clipped * (1.0f - fminf(w_rh, 1.0f));

        // === Backward through r_h * h_prev ===
        float d_log_hp_rh = d_log_rh_hp;
        float d_log_rh_val = fminf(fmaxf(d_log_rh_hp, -GRAD_CLIP), GRAD_CLIP);

        // === Total gradient to log_h_prev ===
        float d_log_hp_total = d_log_hp_decay + d_log_hp_rh;
        // Clip total gradient to prevent explosion through time
        d_log_h_prev[idx] = static_cast<T>(fminf(fmaxf(d_log_hp_total, -10.0f), 10.0f));

        // === Backward through delta gate ===
        float h_prev_linear = from_log_space(fminf(log_hp, 20.0f), sign_hp);
        float cand_linear = from_log_space(fminf(log_h_unb, 20.0f), sign_v_val);
        cand_linear = cand_linear / (1.0f + fabsf(cand_linear));  // Approximate bounded
        h_prev_linear = fminf(fmaxf(h_prev_linear, -100.0f), 100.0f);  // Clamp linear values

        float d_delta = grad_log_h * (cand_linear - h_prev_linear);
        d_delta = fminf(fmaxf(d_delta, -100.0f), 100.0f);  // Clip before sigmoid derivative
        float d_delta_raw_val = d_delta * del * one_minus_delta;
        d_delta_raw[idx] = static_cast<T>(fminf(fmaxf(d_delta_raw_val, -GRAD_CLIP), GRAD_CLIP));

        // === Backward through W_x @ x + b ===
        // Clamp log argument to prevent explosion
        float log_input_arg = fminf(fmaxf(log_v_val - log_rh - log_hp, -20.0f), 20.0f);
        float input_linear = from_log_space(log_input_arg, 1.0f);
        input_linear = fminf(fmaxf(input_linear, -100.0f), 100.0f);
        float d_wx_linear = d_log_input * fmaxf(fabsf(input_linear), LOG_EPS);
        d_wx_x[idx] = static_cast<T>(fminf(fmaxf(d_wx_linear, -GRAD_CLIP), GRAD_CLIP));

        // === Accumulate parameter gradients ===
        atomicAdd(&d_log_r_h[d], d_log_rh_val);
        atomicAdd(&d_b[d], d_wx_linear);
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
// Log-Space Polynomial Elman Forward
// =============================================================================

template<typename T>
struct LogPolyElmanForward {
    bool training_;
    int batch_size_;
    int dim_;
    int n_groups_;  // NEW: for selective output
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;

    LogPolyElmanForward(
        bool training,
        int batch_size,
        int dim,
        int n_groups,  // NEW: for selective output
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream)
        : training_(training),
          batch_size_(batch_size),
          dim_(dim),
          n_groups_(n_groups),
          blas_handle_(blas_handle),
          stream_(stream) {}

    void Run(
        int steps,
        const T* W_x,           // [dim, dim]
        const T* log_r_h,       // [dim]
        const T* sign_r_h,      // [dim]
        const T* W_alpha,       // [dim, dim] for input-dependent alpha
        const T* b_alpha,       // [dim]
        const T* W_delta,       // [dim, dim]
        const T* W_out,         // [dim, dim] NEW: for selective output
        const T* b,             // [dim]
        const T* b_delta,       // [dim]
        const T* log_gamma,     // [dim] RMSNorm scale (learnable)
        const T* x,             // [steps, B, dim]
        T* log_h,               // [steps+1, B, dim]
        T* sign_h,              // [steps+1, B, dim]
        T* output,              // [steps, B, dim] RENAMED: final output after selective
        T* h_linear_cache,      // [steps, B, dim] intermediate h_linear for backward
        T* log_v_cache,
        T* sign_v_cache,
        T* alpha_cache,
        T* log_h_unbounded_cache,
        T* delta_cache,
        T* weight_rh_cache,
        T* alpha_raw_cache,
        T* log_rms_cache,
        T* compete_cache) {     // [steps, B, dim] NEW: for selective output backward

        static const T alpha_one = static_cast<T>(1.0);
        static const T beta_zero = static_cast<T>(0.0);

        const int BD = batch_size_ * dim_;
        const int TBD = steps * BD;  // Total batch*dim for all timesteps
        const int block_size = 256;
        const int num_blocks = (BD + block_size - 1) / block_size;
        const int group_size = dim_ / n_groups_;  // NEW: for selective output

        // =========================================================================
        // Haste pattern: Pre-compute ALL input projections in big GEMMs
        // =========================================================================

        T *all_wx_x, *all_alpha_raw, *all_delta_tmp, *w_out_h;
        cudaMalloc(&all_wx_x, TBD * sizeof(T));
        cudaMalloc(&all_alpha_raw, TBD * sizeof(T));
        cudaMalloc(&all_delta_tmp, TBD * sizeof(T));
        cudaMalloc(&w_out_h, BD * sizeof(T));  // NEW: for selective output

        // Pre-compute W_x @ x for ALL timesteps in one GEMM
        blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, steps * batch_size_, dim_, &alpha_one, W_x, dim_, x, dim_, &beta_zero, all_wx_x, dim_);

        // Pre-compute W_alpha @ x for ALL timesteps
        blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, steps * batch_size_, dim_, &alpha_one, W_alpha, dim_, x, dim_, &beta_zero, all_alpha_raw, dim_);

        // Pre-compute W_delta @ x for ALL timesteps
        blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, steps * batch_size_, dim_, &alpha_one, W_delta, dim_, x, dim_, &beta_zero, all_delta_tmp, dim_);

        // Add biases to alpha_raw and delta_tmp for all timesteps
        for (int t = 0; t < steps; ++t) {
            AddBiasKernel<T><<<num_blocks, block_size, 0, stream_>>>(
                batch_size_, dim_, all_alpha_raw + t * BD, b_alpha, all_alpha_raw + t * BD);
            AddBiasKernel<T><<<num_blocks, block_size, 0, stream_>>>(
                batch_size_, dim_, all_delta_tmp + t * BD, b_delta, all_delta_tmp + t * BD);
        }

        // =========================================================================
        // Sequential loop: Only recurrent operations (no input GEMMs per step)
        // =========================================================================

        for (int t = 0; t < steps; ++t) {
            const T* log_h_prev = log_h + t * BD;
            const T* sign_h_prev = sign_h + t * BD;
            T* log_h_t = log_h + (t + 1) * BD;
            T* sign_h_t = sign_h + (t + 1) * BD;
            T* out_t = output + t * BD;

            // Caches for this timestep
            T* log_v_t = training_ ? (log_v_cache + t * BD) : nullptr;
            T* sign_v_t = training_ ? (sign_v_cache + t * BD) : nullptr;
            T* alpha_t = training_ ? (alpha_cache + t * BD) : nullptr;
            T* log_h_unb_t = training_ ? (log_h_unbounded_cache + t * BD) : nullptr;
            T* delta_t = training_ ? (delta_cache + t * BD) : nullptr;
            T* weight_t = training_ ? (weight_rh_cache + t * BD) : nullptr;
            T* alpha_raw_t = training_ ? (alpha_raw_cache + t * BD) : nullptr;
            T* h_linear_t = training_ ? (h_linear_cache + t * BD) : nullptr;
            T* compete_t = training_ ? (compete_cache + t * BD) : nullptr;
            T* log_rms_t = training_ ? (log_rms_cache + t * batch_size_) : nullptr;

            // Get pre-computed projections for this timestep
            const T* wx_x_t = all_wx_x + t * BD;
            const T* alpha_raw_in = all_alpha_raw + t * BD;
            const T* delta_tmp_t = all_delta_tmp + t * BD;

            // Store alpha_raw for backward
            if (alpha_raw_t) {
                cudaMemcpyAsync(alpha_raw_t, alpha_raw_in, BD * sizeof(T), cudaMemcpyDeviceToDevice, stream_);
            }

            // Main update kernel (no input GEMM needed - already pre-computed)
            LogPolyGatedUpdateKernel<T><<<num_blocks, block_size, 0, stream_>>>(
                batch_size_, dim_,
                log_h_prev, sign_h_prev,
                wx_x_t, alpha_raw_in, log_r_h, sign_r_h, b, delta_tmp_t,
                log_h_t, sign_h_t,
                log_v_t, sign_v_t, alpha_t, log_h_unb_t, delta_t, weight_t);

            // Apply RMSNorm and convert to linear
            const int rmsnorm_block = 256;
            size_t rmsnorm_smem = rmsnorm_block * sizeof(float);

            // Use temporary buffer if not training (no cache needed)
            T* h_linear_buf = h_linear_t ? h_linear_t : w_out_h;  // Reuse w_out_h as temp

            LogSpaceRMSNormKernel<T, 256><<<batch_size_, rmsnorm_block, rmsnorm_smem, stream_>>>(
                batch_size_, dim_,
                log_h_t, sign_h_t, log_gamma,
                h_linear_buf, log_rms_t);

            // w_out_h = h_linear @ W_out.T
            blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                dim_, batch_size_, dim_, &alpha_one, W_out, dim_, h_linear_buf, dim_, &beta_zero, w_out_h, dim_);

            // NEW: Selective output with compete mechanism (stabilizer)
            dim3 grid(batch_size_, n_groups_);
            int smem_size = 2 * block_size * sizeof(float);
            LogSelectiveOutput<T><<<grid, block_size, smem_size, stream_>>>(
                batch_size_, dim_, n_groups_, group_size, h_linear_buf, w_out_h, out_t, compete_t);
        }

        cudaFree(all_wx_x);
        cudaFree(all_alpha_raw);
        cudaFree(all_delta_tmp);
        cudaFree(w_out_h);
    }
};

// =============================================================================
// Log-Space Polynomial Elman Backward
// =============================================================================

template<typename T>
struct LogPolyElmanBackward {
    int batch_size_;
    int dim_;
    int n_groups_;  // NEW: for selective output
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;

    LogPolyElmanBackward(
        int batch_size,
        int dim,
        int n_groups,  // NEW: for selective output
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream)
        : batch_size_(batch_size),
          dim_(dim),
          n_groups_(n_groups),
          blas_handle_(blas_handle),
          stream_(stream) {}

    void Run(
        int steps,
        const T* W_x,
        const T* log_r_h,
        const T* sign_r_h,
        const T* W_alpha,
        const T* W_delta,
        const T* W_out,          // NEW: for selective output
        const T* log_gamma,      // [dim] RMSNorm scale
        const T* x,
        const T* log_h,
        const T* sign_h,
        const T* log_v_cache,
        const T* sign_v_cache,
        const T* alpha_cache,
        const T* alpha_raw_cache,
        const T* log_h_unbounded_cache,
        const T* delta_cache,
        const T* weight_rh_cache,
        const T* h_linear_cache, // [T, B, dim] cached h_linear for RMSNorm backward
        const T* compete_cache,  // NEW: cached compete weights
        const T* log_rms_cache,  // [T, B] cached from forward
        const T* d_output,       // RENAMED: gradient from final output
        T* dx,
        T* dW_x,
        T* d_log_r_h,
        T* dW_alpha,
        T* db_alpha,
        T* dW_delta,
        T* dW_out,               // NEW: gradient for W_out
        T* db,
        T* db_delta,
        T* d_log_gamma,          // [dim] gradient for RMSNorm scale
        T* workspace) {

        static const T alpha_one = static_cast<T>(1.0);
        static const T beta_zero = static_cast<T>(0.0);

        const int BD = batch_size_ * dim_;
        const int block_size = 256;
        const int num_blocks = (BD + block_size - 1) / block_size;
        const int group_size = dim_ / n_groups_;  // NEW

        // Workspace layout: 8 * BD * sizeof(T) + 5 * dim_ * sizeof(float)
        T* d_log_h_recurrent = workspace;
        T* d_log_h_prev = d_log_h_recurrent + BD;
        T* d_wx_x = d_log_h_prev + BD;
        T* d_alpha_raw = d_wx_x + BD;
        T* d_delta_raw = d_alpha_raw + BD;
        T* d_log_h_from_rmsnorm = d_delta_raw + BD;
        T* d_h_linear = d_log_h_from_rmsnorm + BD;   // NEW: gradient for h_linear
        T* d_w_out_h = d_h_linear + BD;              // NEW: gradient for w_out_h
        T* w_out_h = d_w_out_h + BD;                 // NEW: recomputed w_out_h

        // Float buffers for atomic gradients (after T buffers)
        float* d_log_r_h_float = reinterpret_cast<float*>(w_out_h + BD);
        float* db_float = d_log_r_h_float + dim_;
        float* db_delta_float = db_float + dim_;
        float* db_alpha_float = db_delta_float + dim_;
        float* d_log_gamma_float = db_alpha_float + dim_;

        cudaMemsetAsync(d_log_h_recurrent, 0, BD * sizeof(T), stream_);
        cudaMemsetAsync(d_log_r_h_float, 0, dim_ * sizeof(float), stream_);
        cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
        cudaMemsetAsync(db_delta_float, 0, dim_ * sizeof(float), stream_);
        cudaMemsetAsync(db_alpha_float, 0, dim_ * sizeof(float), stream_);
        cudaMemsetAsync(d_log_gamma_float, 0, dim_ * sizeof(float), stream_);

        // Zero weight gradients
        cudaMemsetAsync(dW_x, 0, dim_ * dim_ * sizeof(T), stream_);
        cudaMemsetAsync(dW_alpha, 0, dim_ * dim_ * sizeof(T), stream_);
        cudaMemsetAsync(dW_delta, 0, dim_ * dim_ * sizeof(T), stream_);
        cudaMemsetAsync(dW_out, 0, dim_ * dim_ * sizeof(T), stream_);  // NEW

        // Kernel configs
        const int rmsnorm_block = 256;
        const int smem_size = rmsnorm_block * sizeof(float);

        for (int t = steps - 1; t >= 0; --t) {
            const T* x_t = x + t * BD;
            const T* log_h_prev = log_h + t * BD;
            const T* sign_h_prev = sign_h + t * BD;
            const T* log_h_new = log_h + (t + 1) * BD;   // log_h after step t
            const T* sign_h_new = sign_h + (t + 1) * BD;

            const T* log_v_t = log_v_cache + t * BD;
            const T* sign_v_t = sign_v_cache + t * BD;
            const T* alpha_t = alpha_cache + t * BD;
            const T* alpha_raw_t = alpha_raw_cache + t * BD;
            const T* log_h_unb_t = log_h_unbounded_cache + t * BD;
            const T* delta_t = delta_cache + t * BD;
            const T* weight_t = weight_rh_cache + t * BD;
            const T* h_linear_t = h_linear_cache + t * BD;
            const T* compete_t = compete_cache + t * BD;

            const T* d_output_t = d_output + t * BD;
            T* dx_t = dx + t * BD;

            // Step 0: Recompute w_out_h = W_out @ h_linear
            blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                dim_, batch_size_, dim_, &alpha_one, W_out, dim_, h_linear_t, dim_, &beta_zero, w_out_h, dim_);

            // Step 1: Selective output backward -> d_h_linear, d_w_out_h
            dim3 grid(batch_size_, n_groups_);
            int sel_smem = block_size * sizeof(float);
            LogSelectiveOutputBackward<T><<<grid, block_size, sel_smem, stream_>>>(
                batch_size_, dim_, n_groups_, group_size,
                h_linear_t, w_out_h, compete_t, d_output_t, d_h_linear, d_w_out_h);

            // dW_out
            blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
                dim_, dim_, batch_size_, &alpha_one, d_w_out_h, dim_, h_linear_t, dim_, &alpha_one, dW_out, dim_);

            // d_h_linear += W_out^T @ d_w_out_h
            blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                dim_, batch_size_, dim_, &alpha_one, W_out, dim_, d_w_out_h, dim_, &alpha_one, d_h_linear, dim_);

            // Step 2: RMSNorm backward - convert d_h_linear -> d_log_h
            LogSpaceRMSNormBackwardKernel<T, 256><<<batch_size_, rmsnorm_block, smem_size, stream_>>>(
                batch_size_, dim_,
                log_h_new, sign_h_new, log_gamma,
                h_linear_t, d_h_linear,
                d_log_h_from_rmsnorm, d_log_gamma_float);

            // Step 3: Add recurrent gradient from next timestep
            // d_log_h_total = d_log_h_from_rmsnorm + d_log_h_recurrent
            if (t < steps - 1) {
                // Simple element-wise add kernel
                AddKernel<T><<<num_blocks, block_size, 0, stream_>>>(
                    BD, d_log_h_from_rmsnorm, d_log_h_recurrent, d_log_h_from_rmsnorm);
            }

            // Step 4: Backward through polynomial update
            LogPolyGatedBackwardKernel<T><<<num_blocks, block_size, 0, stream_>>>(
                batch_size_, dim_,
                log_h_prev, sign_h_prev,
                log_v_t, sign_v_t, alpha_t, alpha_raw_t, log_h_unb_t, delta_t, weight_t,
                log_r_h, sign_r_h,
                d_log_h_from_rmsnorm,  // Use gradient after RMSNorm backward
                nullptr,  // recurrent gradient already added above
                d_log_h_prev, d_wx_x, d_alpha_raw, d_delta_raw,
                d_log_r_h_float, db_float);

            // dx through W_x path (backward of x @ W.T is d_result @ W)
            blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                dim_, batch_size_, dim_, &alpha_one, W_x, dim_, d_wx_x, dim_, &beta_zero, dx_t, dim_);

            // dx through W_alpha path
            blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                dim_, batch_size_, dim_, &alpha_one, W_alpha, dim_, d_alpha_raw, dim_, &alpha_one, dx_t, dim_);

            // dx through W_delta path
            blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                dim_, batch_size_, dim_, &alpha_one, W_delta, dim_, d_delta_raw, dim_, &alpha_one, dx_t, dim_);

            // Weight gradients
            blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
                dim_, dim_, batch_size_, &alpha_one, d_wx_x, dim_, x_t, dim_, &alpha_one, dW_x, dim_);

            blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
                dim_, dim_, batch_size_, &alpha_one, d_alpha_raw, dim_, x_t, dim_, &alpha_one, dW_alpha, dim_);

            blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
                dim_, dim_, batch_size_, &alpha_one, d_delta_raw, dim_, x_t, dim_, &alpha_one, dW_delta, dim_);

            // Copy for next iteration
            cudaMemcpyAsync(d_log_h_recurrent, d_log_h_prev, BD * sizeof(T), cudaMemcpyDeviceToDevice, stream_);
        }

        // Copy float gradients to output type using kernel
        const int copy_blocks = (dim_ + block_size - 1) / block_size;
        CopyFloatToT<T><<<copy_blocks, block_size, 0, stream_>>>(dim_, d_log_r_h_float, d_log_r_h);
        CopyFloatToT<T><<<copy_blocks, block_size, 0, stream_>>>(dim_, db_float, db);
        CopyFloatToT<T><<<copy_blocks, block_size, 0, stream_>>>(dim_, d_log_gamma_float, d_log_gamma);
    }
};

// Explicit template instantiations
template struct LogPolyElmanForward<__half>;
template struct LogPolyElmanForward<__nv_bfloat16>;
template struct LogPolyElmanForward<float>;
template struct LogPolyElmanForward<double>;

template struct LogPolyElmanBackward<__half>;
template struct LogPolyElmanBackward<__nv_bfloat16>;
template struct LogPolyElmanBackward<float>;
template struct LogPolyElmanBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
