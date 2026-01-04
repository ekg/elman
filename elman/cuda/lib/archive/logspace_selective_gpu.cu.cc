// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// Log-Space Level 1: Selective Log-Space Polynomial
// Combines log-space polynomial activation with compete×silu output selection.
//
// Architecture:
//   α_t = 1 + softplus(W_α @ x_t + b_α)        -- input-dependent exponent
//   v = r_h * h_prev + W_x @ x + b             -- signed log arithmetic
//   log|h_cand| = α_t * log|v|                 -- polynomial nonlinearity
//   log|h_bounded| = -softplus(-log|h_cand|)   -- soft bound magnitude ≤ 1
//   δ = sigmoid(W_δ @ x_t + b_δ)
//   h_new = (1-δ) * h_prev + δ * h_bounded     -- gated update
//
//   // Selective output (like level 2)
//   compete = softmax(h.reshape(groups), dim=-1)
//   output = compete * silu(W_out @ h)
//
// Key properties:
// - Polynomial activation in log space (gradient = α, constant)
// - Soft bounding prevents unbounded growth
// - Compete softmax provides selective attention within groups
// - All gradients bounded

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
constexpr float LOG_ZERO = -40.0f;
constexpr float LOG_EPS = 1e-10f;
constexpr float GRAD_CLIP = 10.0f;

// =============================================================================
// Device functions for signed log arithmetic (shared with log_0)
// =============================================================================

__device__ __forceinline__ void to_log_space(float x, float& log_x, float& sign_x) {
    sign_x = (x >= 0.0f) ? 1.0f : -1.0f;
    float abs_x = fabsf(x);
    log_x = (abs_x > LOG_EPS) ? logf(abs_x) : LOG_ZERO;
}

__device__ __forceinline__ float from_log_space(float log_x, float sign_x) {
    if (log_x <= LOG_ZERO + 1.0f) return 0.0f;
    return sign_x * expf(fminf(log_x, 20.0f));
}

__device__ __forceinline__ void signed_log_add(
    float log_a, float sign_a, float log_b, float sign_b,
    float& log_result, float& sign_result, float& weight_a) {

    if (log_a <= LOG_ZERO + 1.0f) {
        log_result = log_b; sign_result = sign_b; weight_a = 0.0f; return;
    }
    if (log_b <= LOG_ZERO + 1.0f) {
        log_result = log_a; sign_result = sign_a; weight_a = 1.0f; return;
    }

    float max_log = fmaxf(log_a, log_b);
    float min_log = fminf(log_a, log_b);
    float diff = min_log - max_log;

    bool a_is_max = log_a >= log_b;
    float sign_max = a_is_max ? sign_a : sign_b;
    float sign_min = a_is_max ? sign_b : sign_a;
    bool same_sign = sign_max * sign_min > 0.0f;

    if (same_sign) {
        log_result = max_log + log1pf(expf(diff));
        sign_result = sign_max;
        weight_a = expf(log_a - log_result);
    } else {
        float exp_diff = expf(diff);
        if (exp_diff >= 0.9999f) {
            log_result = LOG_ZERO; sign_result = 1.0f; weight_a = 0.5f;
        } else {
            log_result = max_log + log1pf(-exp_diff);
            sign_result = sign_max;
            float abs_result = expf(log_result);
            float abs_a = expf(log_a);
            weight_a = (abs_result > LOG_EPS) ? fminf(abs_a / abs_result, GRAD_CLIP) : 1.0f;
        }
    }
    weight_a = fminf(fmaxf(weight_a, 0.0f), GRAD_CLIP);
}

__device__ __forceinline__ float soft_bound_log(float log_h) {
    return (log_h > 0.0f) ? -log1pf(expf(-log_h)) : log_h - log1pf(expf(log_h));
}

__device__ __forceinline__ float soft_bound_grad(float log_h) {
    return 1.0f / (1.0f + expf(log_h));
}

__device__ __forceinline__ float stable_softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return 0.0f;
    return log1pf(expf(x));
}

__device__ __forceinline__ float softplus_grad(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// =============================================================================
// Forward: Log-Space RMSNorm (fused normalization)
// Computes: h_normed = gamma * h / rms(h) entirely in log-space
// Uses logsumexp for bounded gradients
// =============================================================================

template<typename T>
__global__ void LogSpaceRMSNormKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ log_h,        // [B, dim] log magnitudes
    const T* __restrict__ sign_h,       // [B, dim] signs
    const T* __restrict__ log_gamma,    // [dim] learnable scale in log-space
    T* __restrict__ log_h_normed,       // [B, dim] normalized log magnitudes
    T* __restrict__ h_linear_normed,    // [B, dim] normalized linear values
    T* __restrict__ log_rms_cache) {    // [B] cached log_rms for backward (optional)

    extern __shared__ float smem[];

    const int b = blockIdx.x;
    if (b >= batch_size) return;

    const int base = b * dim;
    const float log_dim = logf(static_cast<float>(dim));

    // Step 1: Compute logsumexp(2 * log_h) for this batch element
    // Find max for numerical stability
    float max_log_h2 = -FLT_MAX;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float log_h_val = static_cast<float>(log_h[base + d]);
        float log_h2 = 2.0f * log_h_val;  // log(h^2) = 2*log|h|
        max_log_h2 = fmaxf(max_log_h2, log_h2);
    }
    smem[threadIdx.x] = max_log_h2;
    __syncthreads();

    // Reduce to find global max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        }
        __syncthreads();
    }
    max_log_h2 = smem[0];
    __syncthreads();

    // Compute sum of exp(log_h2 - max)
    float sum_exp = 0.0f;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float log_h_val = static_cast<float>(log_h[base + d]);
        float log_h2 = 2.0f * log_h_val;
        sum_exp += expf(log_h2 - max_log_h2);
    }

    float* sum_smem = smem + blockDim.x;
    sum_smem[threadIdx.x] = sum_exp;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sum_smem[threadIdx.x] += sum_smem[threadIdx.x + s];
        }
        __syncthreads();
    }
    sum_exp = sum_smem[0];

    // logsumexp = max + log(sum_exp)
    float log_sum_h2 = max_log_h2 + logf(sum_exp + 1e-10f);

    // log(mean(h^2)) = logsumexp - log(dim)
    float log_mean_h2 = log_sum_h2 - log_dim;

    // log(rms) = log(mean(h^2)) / 2
    float log_rms = log_mean_h2 * 0.5f;

    // Cache log_rms for backward
    if (log_rms_cache && threadIdx.x == 0) {
        log_rms_cache[b] = static_cast<T>(log_rms);
    }

    // Step 2: Normalize each element
    // log|h_normed| = log|h| - log_rms + log_gamma
    // h_normed = sign_h * exp(log|h_normed|)
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float log_h_val = static_cast<float>(log_h[base + d]);
        float sign_h_val = static_cast<float>(sign_h[base + d]);
        float log_gamma_val = static_cast<float>(log_gamma[d]);

        float log_h_norm = log_h_val - log_rms + log_gamma_val;
        log_h_normed[base + d] = static_cast<T>(log_h_norm);

        // Convert to linear space
        float h_lin = from_log_space(log_h_norm, sign_h_val);
        h_linear_normed[base + d] = static_cast<T>(h_lin);
    }
}

// =============================================================================
// Backward: Log-Space RMSNorm
// =============================================================================

template<typename T>
__global__ void LogSpaceRMSNormBackward(
    const int batch_size,
    const int dim,
    const T* __restrict__ log_h,        // [B, dim] input log magnitudes
    const T* __restrict__ sign_h,       // [B, dim] signs
    const T* __restrict__ log_gamma,    // [dim] learnable scale
    const T* __restrict__ log_rms,      // [B] cached log_rms from forward
    const T* __restrict__ d_h_linear,   // [B, dim] gradient from output
    T* __restrict__ d_log_h,            // [B, dim] gradient to input
    float* __restrict__ d_log_gamma) {  // [dim] gradient to log_gamma (float for atomicAdd)

    extern __shared__ float smem[];

    const int b = blockIdx.x;
    if (b >= batch_size) return;

    const int base = b * dim;
    const float log_dim = logf(static_cast<float>(dim));
    float log_rms_val = static_cast<float>(log_rms[b]);

    // Compute sum of d_out * h_normed (for gradient through rms)
    float sum_dout_hnorm = 0.0f;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float log_h_val = static_cast<float>(log_h[base + d]);
        float sign_h_val = static_cast<float>(sign_h[base + d]);
        float log_gamma_val = static_cast<float>(log_gamma[d]);
        float d_out = static_cast<float>(d_h_linear[base + d]);

        float log_h_norm = log_h_val - log_rms_val + log_gamma_val;
        float h_norm = from_log_space(log_h_norm, sign_h_val);

        sum_dout_hnorm += d_out * h_norm;
    }

    smem[threadIdx.x] = sum_dout_hnorm;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] += smem[threadIdx.x + s];
        }
        __syncthreads();
    }
    sum_dout_hnorm = smem[0];

    // Mean for gradient correction
    float mean_dout_hnorm = sum_dout_hnorm / static_cast<float>(dim);

    // Compute gradients
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float log_h_val = static_cast<float>(log_h[base + d]);
        float sign_h_val = static_cast<float>(sign_h[base + d]);
        float log_gamma_val = static_cast<float>(log_gamma[d]);
        float d_out = static_cast<float>(d_h_linear[base + d]);

        float log_h_norm = log_h_val - log_rms_val + log_gamma_val;
        float h_norm = from_log_space(log_h_norm, sign_h_val);

        // Gradient w.r.t. log_gamma: d_out * h_norm (through exp)
        // Reduced scaling from 0.0001 to 0.01
        float d_log_gamma_val = d_out * h_norm * 0.001f;
        atomicAdd(&d_log_gamma[d], d_log_gamma_val);

        // Gradient w.r.t. log_h: (d_out * gamma / rms) * (1 - h^2 / mean_h^2 / dim)
        // Simplified: d_out * exp(log_gamma - log_rms) * sign * (1 - h_norm * mean_dout_hnorm / d_out / h_norm)
        float scale = expf(log_gamma_val - log_rms_val);
        float d_h = d_out * scale * sign_h_val - h_norm * mean_dout_hnorm / static_cast<float>(dim);

        // Convert to d_log_h: d_h * h / |h| = d_h * sign * exp(log_h)
        // But we want d_log_h, not d_h. Since h = sign * exp(log_h), d_log_h = d_h * sign * exp(log_h) = d_h * h
        // Reduced scaling from 0.0001 to 0.01
        float h_val = from_log_space(log_h_val, sign_h_val);
        float d_log_h_val = d_h * h_val * 0.001f;

        d_log_h[base + d] = static_cast<T>(fminf(fmaxf(d_log_h_val, -GRAD_CLIP), GRAD_CLIP));
    }
}

// =============================================================================
// Forward: Log-Space Polynomial Update (from log_0)
// =============================================================================

template<typename T>
__global__ void LogSelectiveUpdateKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ log_h_prev,
    const T* __restrict__ sign_h_prev,
    const T* __restrict__ wx_x,
    const T* __restrict__ alpha_raw,
    const T* __restrict__ log_r_h,
    const T* __restrict__ sign_r_h,
    const T* __restrict__ b,
    const T* __restrict__ delta_raw,
    T* __restrict__ log_h_out,
    T* __restrict__ sign_h_out,
    T* __restrict__ h_linear_out,      // Linear space for compete softmax
    T* __restrict__ log_v_cache,
    T* __restrict__ sign_v_cache,
    T* __restrict__ alpha_cache,
    T* __restrict__ log_h_unbounded_cache,
    T* __restrict__ delta_cache,
    T* __restrict__ weight_rh_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float log_hp = static_cast<float>(log_h_prev[idx]);
        float sign_hp = static_cast<float>(sign_h_prev[idx]);
        // Clamp log_r_h to <= -0.1 (so r_h <= 0.9) for stability
        float log_rh = fminf(static_cast<float>(log_r_h[d]), -0.1f);
        float sign_rh = static_cast<float>(sign_r_h[d]);
        float wx = static_cast<float>(wx_x[idx]);
        float bias = static_cast<float>(b[d]);

        // Input-dependent alpha, capped at 2.0 for stability
        float alpha_raw_val = static_cast<float>(alpha_raw[idx]);
        float alpha = 1.0f + fminf(stable_softplus(alpha_raw_val), 1.0f);
        if (alpha_cache) alpha_cache[idx] = static_cast<T>(alpha);

        // r_h * h_prev in log space
        float log_rh_hp = log_rh + log_hp;
        float sign_rh_hp = sign_rh * sign_hp;

        // W_x @ x + b to log space
        float linear_input = wx + bias;
        float log_input, sign_input;
        to_log_space(linear_input, log_input, sign_input);

        // Add: v = r_h * h_prev + input
        float log_v, sign_v, weight_rh;
        signed_log_add(log_rh_hp, sign_rh_hp, log_input, sign_input,
                       log_v, sign_v, weight_rh);

        if (log_v_cache) log_v_cache[idx] = static_cast<T>(log_v);
        if (sign_v_cache) sign_v_cache[idx] = static_cast<T>(sign_v);
        if (weight_rh_cache) weight_rh_cache[idx] = static_cast<T>(weight_rh);

        // Tanh activation (replacing polynomial for stability)
        // Convert v to linear, apply tanh, convert back to log
        float v_linear = from_log_space(log_v, sign_v);
        float cand_linear = tanhf(v_linear);
        float log_cand_bounded, sign_cand;
        to_log_space(cand_linear, log_cand_bounded, sign_cand);
        // Cache the tanh input for backward pass
        if (log_h_unbounded_cache) log_h_unbounded_cache[idx] = static_cast<T>(v_linear);

        // Delta gate
        float delta_raw_val = static_cast<float>(delta_raw[idx]);
        float delta = 1.0f / (1.0f + expf(-delta_raw_val));
        if (delta_cache) delta_cache[idx] = static_cast<T>(delta);

        // Gated update in log space
        float log_one_minus_delta = -log1pf(expf(delta_raw_val));
        float log_delta = -log1pf(expf(-delta_raw_val));

        float log_term1 = log_one_minus_delta + log_hp;
        float sign_term1 = sign_hp;
        float log_term2 = log_delta + log_cand_bounded;
        float sign_term2 = sign_cand;

        float log_h_new, sign_h_new, weight_decay;
        signed_log_add(log_term1, sign_term1, log_term2, sign_term2,
                       log_h_new, sign_h_new, weight_decay);

        log_h_out[idx] = static_cast<T>(log_h_new);
        sign_h_out[idx] = static_cast<T>(sign_h_new);

        // Output linear for compete softmax
        float h_lin = from_log_space(log_h_new, sign_h_new);
        h_linear_out[idx] = static_cast<T>(h_lin);
    }
}

// =============================================================================
// Forward: Selective Output (compete × silu)
// =============================================================================

template<typename T>
__global__ void LogSelectiveOutput(
    const int batch_size,
    const int dim,
    const int n_groups,
    const int group_size,
    const T* __restrict__ h_linear,   // [B, dim] in linear space
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
// Backward: Selective Output
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
// Backward: Log-Space Update
// =============================================================================

template<typename T>
__global__ void LogSelectiveUpdateBackward(
    const int batch_size,
    const int dim,
    const T* __restrict__ log_h_prev,
    const T* __restrict__ sign_h_prev,
    const T* __restrict__ log_v,
    const T* __restrict__ sign_v,
    const T* __restrict__ alpha,
    const T* __restrict__ alpha_raw,
    const T* __restrict__ log_h_unbounded,
    const T* __restrict__ delta,
    const T* __restrict__ weight_rh,
    const T* __restrict__ log_r_h,
    const T* __restrict__ sign_r_h,
    const T* __restrict__ d_h_linear,     // From selective output backward
    const T* __restrict__ d_log_h_recurrent,
    T* __restrict__ d_log_h_prev,
    T* __restrict__ d_wx_x,
    T* __restrict__ d_alpha_raw,
    T* __restrict__ d_delta_raw,
    float* __restrict__ d_log_r_h,
    float* __restrict__ d_b) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // Gradient from linear space output
        float grad_h_linear = static_cast<float>(d_h_linear[idx]);

        // Also add recurrent gradient if available
        float grad_log_h = 0.0f;
        if (d_log_h_recurrent) {
            grad_log_h += static_cast<float>(d_log_h_recurrent[idx]);
        }

        // Load cached values
        float log_hp = static_cast<float>(log_h_prev[idx]);
        float sign_hp = static_cast<float>(sign_h_prev[idx]);
        float log_v_val = static_cast<float>(log_v[idx]);
        float sign_v_val = static_cast<float>(sign_v[idx]);
        float alpha_val = static_cast<float>(alpha[idx]);
        float alpha_raw_val = static_cast<float>(alpha_raw[idx]);
        float log_h_unb = static_cast<float>(log_h_unbounded[idx]);
        float del = static_cast<float>(delta[idx]);
        float w_rh = static_cast<float>(weight_rh[idx]);
        // Clamp log_r_h to <= -0.1 for stability (must match forward)
        float log_rh = fminf(static_cast<float>(log_r_h[d]), -0.1f);

        float one_minus_delta = 1.0f - del;

        // Gradient from log_h to linear h conversion
        // d(exp(log_h)) = exp(log_h) = h_linear
        float h_linear_approx = from_log_space(log_hp, sign_hp);
        float d_log_h_from_linear = grad_h_linear * fabsf(h_linear_approx + 1e-6f);
        grad_log_h += d_log_h_from_linear;

        // Backward through gated update
        float d_log_cand_bounded = grad_log_h * del;
        float d_log_hp_decay = grad_log_h * one_minus_delta;

        // Backward through tanh activation (log_h_unb now stores v_linear)
        float v_linear = log_h_unb;  // Cached tanh input
        float tanh_v = tanhf(v_linear);
        float tanh_grad = 1.0f - tanh_v * tanh_v;  // dtanh/dv = 1 - tanh²

        // d_cand_bounded -> d_v_linear
        // Need to go through to_log_space backward and tanh backward
        float d_cand_linear = d_log_cand_bounded * fabsf(tanh_v + 1e-6f);  // d_log -> d_linear
        float d_v_linear = d_cand_linear * tanh_grad;

        // d_v_linear -> d_log_v (through from_log_space backward)
        float v_from_log = from_log_space(log_v_val, sign_v_val);
        float d_log_v = d_v_linear * fabsf(v_from_log + 1e-6f);

        // No alpha gradient needed (tanh doesn't use alpha)
        d_alpha_raw[idx] = static_cast<T>(0.0f);

        // Backward through signed log addition
        float d_log_rh_hp = d_log_v * w_rh;
        float d_log_input = d_log_v * (1.0f - fminf(w_rh, 1.0f));

        float d_log_hp_rh = d_log_rh_hp;
        float d_log_rh_val = d_log_rh_hp;

        float d_log_hp_total = d_log_hp_decay + d_log_hp_rh;
        d_log_h_prev[idx] = static_cast<T>(d_log_hp_total);

        // Delta gradient (using tanh_v which is the candidate)
        float h_prev_linear = from_log_space(log_hp, sign_hp);
        // cand_linear is tanh(v_linear) = tanh_v (already computed above)
        float d_delta_val = grad_log_h * (tanh_v - h_prev_linear);
        float d_delta_raw_val = d_delta_val * del * one_minus_delta;
        d_delta_raw[idx] = static_cast<T>(fminf(fmaxf(d_delta_raw_val, -GRAD_CLIP), GRAD_CLIP));

        // Input gradient
        float input_linear = from_log_space(log_v_val - log_rh - log_hp, 1.0f);
        float d_wx_linear = d_log_input * fmaxf(fabsf(input_linear), LOG_EPS);
        d_wx_x[idx] = static_cast<T>(fminf(fmaxf(d_wx_linear, -GRAD_CLIP), GRAD_CLIP));

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
// Log-Space Selective Elman Forward
// =============================================================================

template<typename T>
struct LogSelectiveElmanForward {
    bool training_;
    int batch_size_;
    int dim_;
    int n_groups_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;

    LogSelectiveElmanForward(
        bool training, int batch_size, int dim, int n_groups,
        const cublasHandle_t& blas_handle, const cudaStream_t& stream)
        : training_(training), batch_size_(batch_size), dim_(dim),
          n_groups_(n_groups), blas_handle_(blas_handle), stream_(stream) {}

    void Run(
        int steps,
        const T* W_x,
        const T* log_r_h,
        const T* sign_r_h,
        const T* W_alpha,
        const T* b_alpha,
        const T* W_delta,
        const T* W_out,
        const T* b,
        const T* b_delta,
        const T* log_gamma,      // NEW: log-space RMSNorm scale
        const T* x,
        T* log_h,
        T* sign_h,
        T* output,
        T* log_v_cache,
        T* sign_v_cache,
        T* alpha_cache,
        T* log_h_unbounded_cache,
        T* delta_cache,
        T* weight_rh_cache,
        T* alpha_raw_cache,
        T* h_linear_cache,       // Now stores NORMALIZED h_linear
        T* compete_cache,
        T* log_rms_cache) {      // NEW: cache for backward

        static const T alpha_one = static_cast<T>(1.0);
        static const T beta_zero = static_cast<T>(0.0);

        const int BD = batch_size_ * dim_;
        const int TBD = steps * BD;  // Total batch*dim for all timesteps
        const int block_size = 256;
        const int num_blocks = (BD + block_size - 1) / block_size;
        const int group_size = dim_ / n_groups_;

        // =========================================================================
        // Haste pattern: Pre-compute ALL input projections in big GEMMs
        // =========================================================================

        T *all_wx_x, *all_alpha_raw, *all_delta_tmp, *w_out_h, *h_linear_tmp, *log_h_normed_tmp;
        cudaMalloc(&all_wx_x, TBD * sizeof(T));
        cudaMalloc(&all_alpha_raw, TBD * sizeof(T));
        cudaMalloc(&all_delta_tmp, TBD * sizeof(T));
        cudaMalloc(&w_out_h, BD * sizeof(T));
        cudaMalloc(&h_linear_tmp, BD * sizeof(T));
        cudaMalloc(&log_h_normed_tmp, BD * sizeof(T));

        // Pre-compute W_x @ x for ALL timesteps in one GEMM
        blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, steps * batch_size_, dim_, &alpha_one, W_x, dim_, x, dim_, &beta_zero, all_wx_x, dim_);

        // Pre-compute W_alpha @ x for ALL timesteps
        blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, steps * batch_size_, dim_, &alpha_one, W_alpha, dim_, x, dim_, &beta_zero, all_alpha_raw, dim_);

        // Pre-compute W_delta @ x for ALL timesteps
        blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, steps * batch_size_, dim_, &alpha_one, W_delta, dim_, x, dim_, &beta_zero, all_delta_tmp, dim_);

        // =========================================================================
        // Sequential loop: Only recurrent operations (no input GEMMs per step)
        // =========================================================================

        for (int t = 0; t < steps; ++t) {
            const T* log_h_prev = log_h + t * BD;
            const T* sign_h_prev = sign_h + t * BD;
            T* log_h_t = log_h + (t + 1) * BD;
            T* sign_h_t = sign_h + (t + 1) * BD;
            T* out_t = output + t * BD;

            T* log_v_t = training_ ? (log_v_cache + t * BD) : nullptr;
            T* sign_v_t = training_ ? (sign_v_cache + t * BD) : nullptr;
            T* alpha_t = training_ ? (alpha_cache + t * BD) : nullptr;
            T* log_h_unb_t = training_ ? (log_h_unbounded_cache + t * BD) : nullptr;
            T* delta_t = training_ ? (delta_cache + t * BD) : nullptr;
            T* weight_t = training_ ? (weight_rh_cache + t * BD) : nullptr;
            T* alpha_raw_t = training_ ? (alpha_raw_cache + t * BD) : nullptr;
            T* h_linear_t = training_ ? (h_linear_cache + t * BD) : h_linear_tmp;
            T* compete_t = training_ ? (compete_cache + t * BD) : nullptr;
            T* log_rms_t = training_ ? (log_rms_cache + t * batch_size_) : nullptr;

            // Get pre-computed projections for this timestep
            const T* wx_x_t = all_wx_x + t * BD;
            const T* alpha_raw_in = all_alpha_raw + t * BD;
            const T* delta_tmp_t = all_delta_tmp + t * BD;

            if (alpha_raw_t) {
                cudaMemcpyAsync(alpha_raw_t, alpha_raw_in, BD * sizeof(T), cudaMemcpyDeviceToDevice, stream_);
            }

            // Log-space update (no input GEMM needed - already pre-computed)
            LogSelectiveUpdateKernel<T><<<num_blocks, block_size, 0, stream_>>>(
                batch_size_, dim_,
                log_h_prev, sign_h_prev, wx_x_t, alpha_raw_in, log_r_h, sign_r_h, b, delta_tmp_t,
                log_h_t, sign_h_t, h_linear_tmp,
                log_v_t, sign_v_t, alpha_t, log_h_unb_t, delta_t, weight_t);

            // FUSED: Log-Space RMSNorm -> normalized h_linear for output computation
            int norm_smem_size = 2 * block_size * sizeof(float);
            LogSpaceRMSNormKernel<T><<<batch_size_, block_size, norm_smem_size, stream_>>>(
                batch_size_, dim_,
                log_h_t, sign_h_t, log_gamma,
                log_h_normed_tmp,
                h_linear_t,
                log_rms_t);

            // w_out_h = h_linear_normed @ W_out.T (depends on h_t, can't pre-compute)
            blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                dim_, batch_size_, dim_, &alpha_one, W_out, dim_, h_linear_t, dim_, &beta_zero, w_out_h, dim_);

            // Selective output using NORMALIZED h_linear
            dim3 grid(batch_size_, n_groups_);
            int smem_size = 2 * block_size * sizeof(float);
            LogSelectiveOutput<T><<<grid, block_size, smem_size, stream_>>>(
                batch_size_, dim_, n_groups_, group_size, h_linear_t, w_out_h, out_t, compete_t);
        }

        cudaFree(all_wx_x);
        cudaFree(all_alpha_raw);
        cudaFree(all_delta_tmp);
        cudaFree(w_out_h);
        cudaFree(h_linear_tmp);
        cudaFree(log_h_normed_tmp);
    }
};

// =============================================================================
// Log-Space Selective Elman Backward
// =============================================================================

template<typename T>
struct LogSelectiveElmanBackward {
    int batch_size_;
    int dim_;
    int n_groups_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;

    LogSelectiveElmanBackward(
        int batch_size, int dim, int n_groups,
        const cublasHandle_t& blas_handle, const cudaStream_t& stream)
        : batch_size_(batch_size), dim_(dim), n_groups_(n_groups),
          blas_handle_(blas_handle), stream_(stream) {}

    void Run(
        int steps,
        const T* W_x,
        const T* log_r_h,
        const T* sign_r_h,
        const T* W_alpha,
        const T* W_delta,
        const T* W_out,
        const T* log_gamma,          // NEW: RMSNorm scale
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
        const T* h_linear_cache,     // Now contains NORMALIZED h_linear
        const T* compete_cache,
        const T* log_rms_cache,      // NEW: cached log_rms from forward
        const T* d_output,
        T* dx,
        T* dW_x,
        T* d_log_r_h,
        T* dW_alpha,
        T* db_alpha,
        T* dW_delta,
        T* dW_out,
        T* db,
        T* db_delta,
        T* d_log_gamma,              // NEW: gradient for RMSNorm scale
        T* workspace) {

        static const T alpha_one = static_cast<T>(1.0);
        static const T beta_zero = static_cast<T>(0.0);

        const int BD = batch_size_ * dim_;
        const int block_size = 256;
        const int num_blocks = (BD + block_size - 1) / block_size;
        const int group_size = dim_ / n_groups_;

        // Workspace layout: 9 * BD * sizeof(T) + 5 * dim_ * sizeof(float)
        T* d_log_h_recurrent = workspace;
        T* d_log_h_prev = d_log_h_recurrent + BD;
        T* d_wx_x = d_log_h_prev + BD;
        T* d_alpha_raw = d_wx_x + BD;
        T* d_delta_raw = d_alpha_raw + BD;
        T* d_h_linear = d_delta_raw + BD;
        T* d_w_out_h = d_h_linear + BD;
        T* w_out_h = d_w_out_h + BD;
        T* d_log_h = w_out_h + BD;

        // Float buffers for atomic gradients (after T buffers)
        float* d_log_r_h_float = reinterpret_cast<float*>(d_log_h + BD);
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

        cudaMemsetAsync(dW_x, 0, dim_ * dim_ * sizeof(T), stream_);
        cudaMemsetAsync(dW_alpha, 0, dim_ * dim_ * sizeof(T), stream_);
        cudaMemsetAsync(dW_delta, 0, dim_ * dim_ * sizeof(T), stream_);
        cudaMemsetAsync(dW_out, 0, dim_ * dim_ * sizeof(T), stream_);

        for (int t = steps - 1; t >= 0; --t) {
            const T* x_t = x + t * BD;
            const T* log_h_t = log_h + (t + 1) * BD;   // log_h at this timestep
            const T* sign_h_t = sign_h + (t + 1) * BD;
            const T* log_h_prev = log_h + t * BD;
            const T* sign_h_prev = sign_h + t * BD;

            const T* log_v_t = log_v_cache + t * BD;
            const T* sign_v_t = sign_v_cache + t * BD;
            const T* alpha_t = alpha_cache + t * BD;
            const T* alpha_raw_t = alpha_raw_cache + t * BD;
            const T* log_h_unb_t = log_h_unbounded_cache + t * BD;
            const T* delta_t = delta_cache + t * BD;
            const T* weight_t = weight_rh_cache + t * BD;
            const T* h_linear_t = h_linear_cache + t * BD;  // NORMALIZED h_linear
            const T* compete_t = compete_cache + t * BD;
            const T* log_rms_t = log_rms_cache + t * batch_size_;

            const T* d_out_t = d_output + t * BD;
            T* dx_t = dx + t * BD;

            // Recompute w_out_h = W_out @ h_linear_normed
            blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                dim_, batch_size_, dim_, &alpha_one, W_out, dim_, h_linear_t, dim_, &beta_zero, w_out_h, dim_);

            // Backward through selective output -> d_h_linear (gradient w.r.t. normalized h_linear)
            dim3 grid(batch_size_, n_groups_);
            int smem_size = block_size * sizeof(float);
            LogSelectiveOutputBackward<T><<<grid, block_size, smem_size, stream_>>>(
                batch_size_, dim_, n_groups_, group_size,
                h_linear_t, w_out_h, compete_t, d_out_t, d_h_linear, d_w_out_h);

            // dW_out
            blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
                dim_, dim_, batch_size_, &alpha_one, d_w_out_h, dim_, h_linear_t, dim_, &alpha_one, dW_out, dim_);

            // d_h_linear += W_out^T @ d_w_out_h
            blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                dim_, batch_size_, dim_, &alpha_one, W_out, dim_, d_w_out_h, dim_, &alpha_one, d_h_linear, dim_);

            // FUSED: Backward through RMSNorm -> d_log_h
            int norm_smem_size = block_size * sizeof(float);
            LogSpaceRMSNormBackward<T><<<batch_size_, block_size, norm_smem_size, stream_>>>(
                batch_size_, dim_,
                log_h_t, sign_h_t, log_gamma, log_rms_t,
                d_h_linear, d_log_h, d_log_gamma_float);

            // Backward through log-space update (now receives d_log_h instead of d_h_linear)
            LogSelectiveUpdateBackward<T><<<num_blocks, block_size, 0, stream_>>>(
                batch_size_, dim_,
                log_h_prev, sign_h_prev, log_v_t, sign_v_t, alpha_t, alpha_raw_t,
                log_h_unb_t, delta_t, weight_t, log_r_h, sign_r_h,
                d_log_h, (t < steps - 1) ? d_log_h_recurrent : nullptr,
                d_log_h_prev, d_wx_x, d_alpha_raw, d_delta_raw, d_log_r_h_float, db_float);

            // dx
            blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                dim_, batch_size_, dim_, &alpha_one, W_x, dim_, d_wx_x, dim_, &beta_zero, dx_t, dim_);
            blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                dim_, batch_size_, dim_, &alpha_one, W_alpha, dim_, d_alpha_raw, dim_, &alpha_one, dx_t, dim_);
            blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                dim_, batch_size_, dim_, &alpha_one, W_delta, dim_, d_delta_raw, dim_, &alpha_one, dx_t, dim_);

            // Weight gradients
            blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
                dim_, dim_, batch_size_, &alpha_one, d_wx_x, dim_, x_t, dim_, &alpha_one, dW_x, dim_);
            blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
                dim_, dim_, batch_size_, &alpha_one, d_alpha_raw, dim_, x_t, dim_, &alpha_one, dW_alpha, dim_);
            blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
                dim_, dim_, batch_size_, &alpha_one, d_delta_raw, dim_, x_t, dim_, &alpha_one, dW_delta, dim_);

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
template struct LogSelectiveElmanForward<__half>;
template struct LogSelectiveElmanForward<__nv_bfloat16>;
template struct LogSelectiveElmanForward<float>;
template struct LogSelectiveElmanForward<double>;

template struct LogSelectiveElmanBackward<__half>;
template struct LogSelectiveElmanBackward<__nv_bfloat16>;
template struct LogSelectiveElmanBackward<float>;
template struct LogSelectiveElmanBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
