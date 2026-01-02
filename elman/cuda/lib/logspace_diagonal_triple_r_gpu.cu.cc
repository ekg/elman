// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// Level 7: Log-Space Diagonal Triple R
//
// Like log_2 (diagonal selective) but with diagonal R_delta modulation
// for the delta gate, similar to triple_r but using diagonal matrices
// for efficiency.
//
// Architecture:
//   v = r_h * h_prev + W_x @ x + b              -- DIAGONAL r_h (element-wise)
//   delta_raw = W_delta @ x + r_delta * h_prev + b_delta  -- DIAGONAL r_delta
//   delta_t = sigmoid(delta_raw)
//   h_t = (1 - delta_t) * h_prev + delta_t * tanh(v)
//
//   // Selective output with RMSNorm
//   h_linear = RMSNorm(h_t)
//   compete = softmax(h_linear.reshape(groups), dim=-1)
//   output = compete * silu(W_out @ h_linear)
//
// Key features:
// - Diagonal r_h, r_delta stored in log-space for bounded gradients
// - R_delta provides h-dependent gate modulation without full matrix cost
// - RMSNorm in log-space for stable normalization

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

constexpr float LOG_ZERO = -40.0f;
constexpr float LOG_EPS = 1e-10f;
constexpr float GRAD_CLIP = 10.0f;

// =============================================================================
// Device functions for signed log arithmetic
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

__device__ __forceinline__ float stable_softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return 0.0f;
    return log1pf(expf(x));
}

// =============================================================================
// Log-Space RMSNorm Forward Kernel
// =============================================================================

template<typename T>
__global__ void LogDiagTripleRRMSNormKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ log_h,
    const T* __restrict__ sign_h,
    const T* __restrict__ log_gamma,
    T* __restrict__ log_h_normed,
    T* __restrict__ h_linear_normed,
    T* __restrict__ log_rms_cache) {

    extern __shared__ float smem[];

    const int b = blockIdx.x;
    if (b >= batch_size) return;

    const int base = b * dim;
    const float log_dim = logf(static_cast<float>(dim));

    // Compute logsumexp(2 * log_h) for RMS
    float max_log_h2 = -FLT_MAX;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float log_h_val = static_cast<float>(log_h[base + d]);
        float log_h2 = 2.0f * log_h_val;
        max_log_h2 = fmaxf(max_log_h2, log_h2);
    }
    smem[threadIdx.x] = max_log_h2;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        }
        __syncthreads();
    }
    max_log_h2 = smem[0];
    __syncthreads();

    // Sum exp(2*log_h - max)
    float sum_exp = 0.0f;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float log_h_val = static_cast<float>(log_h[base + d]);
        float log_h2 = 2.0f * log_h_val;
        sum_exp += expf(log_h2 - max_log_h2);
    }
    smem[threadIdx.x] = sum_exp;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] += smem[threadIdx.x + s];
        }
        __syncthreads();
    }
    sum_exp = smem[0];

    // log_rms = 0.5 * (max + log(sum_exp) - log(dim))
    float log_sum_h2 = max_log_h2 + logf(sum_exp + LOG_EPS);
    float log_rms = 0.5f * (log_sum_h2 - log_dim);

    if (log_rms_cache && threadIdx.x == 0) {
        log_rms_cache[b] = static_cast<T>(log_rms);
    }
    __syncthreads();

    // Normalize: h_normed = h / rms * gamma = exp(log_h - log_rms + log_gamma) * sign_h
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float log_h_val = static_cast<float>(log_h[base + d]);
        float sign_h_val = static_cast<float>(sign_h[base + d]);
        float log_gamma_val = static_cast<float>(log_gamma[d]);

        float log_h_norm = log_h_val - log_rms + log_gamma_val;
        log_h_normed[base + d] = static_cast<T>(log_h_norm);
        h_linear_normed[base + d] = static_cast<T>(from_log_space(log_h_norm, sign_h_val));
    }
}

// =============================================================================
// Log-Space RMSNorm Backward Kernel
// =============================================================================

template<typename T>
__global__ void LogDiagTripleRRMSNormBackward(
    const int batch_size,
    const int dim,
    const T* __restrict__ log_h,
    const T* __restrict__ sign_h,
    const T* __restrict__ log_gamma,
    const T* __restrict__ log_rms,
    const T* __restrict__ d_h_linear,
    T* __restrict__ d_log_h,
    float* __restrict__ d_log_gamma) {

    extern __shared__ float smem[];

    const int b = blockIdx.x;
    if (b >= batch_size) return;

    const int base = b * dim;
    float log_rms_val = static_cast<float>(log_rms[b]);

    // Compute sum of d_out * h_normed
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
    float mean_dout_hnorm = sum_dout_hnorm / static_cast<float>(dim);

    // Compute gradients
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float log_h_val = static_cast<float>(log_h[base + d]);
        float sign_h_val = static_cast<float>(sign_h[base + d]);
        float log_gamma_val = static_cast<float>(log_gamma[d]);
        float d_out = static_cast<float>(d_h_linear[base + d]);

        float log_h_norm = log_h_val - log_rms_val + log_gamma_val;
        float h_norm = from_log_space(log_h_norm, sign_h_val);

        float d_log_gamma_val = d_out * h_norm * 0.001f;
        atomicAdd(&d_log_gamma[d], d_log_gamma_val);

        float d_log_h_val = (d_out - mean_dout_hnorm) * fabsf(h_norm + 1e-6f) * 0.001f;
        d_log_h[base + d] = static_cast<T>(fminf(fmaxf(d_log_h_val, -GRAD_CLIP), GRAD_CLIP));
    }
}

// =============================================================================
// Forward: Diagonal Triple R Update (with R_delta modulation)
// =============================================================================

template<typename T>
__global__ void LogDiagTripleRUpdateKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ log_h_prev,
    const T* __restrict__ sign_h_prev,
    const T* __restrict__ wx_x,             // [B, dim] W_x @ x
    const T* __restrict__ log_r_h,          // [dim] diagonal recurrence
    const T* __restrict__ sign_r_h,         // [dim]
    const T* __restrict__ log_r_delta,      // [dim] diagonal delta modulation (NEW)
    const T* __restrict__ sign_r_delta,     // [dim] (NEW)
    const T* __restrict__ wdelta_x,         // [B, dim] W_delta @ x
    const T* __restrict__ b,
    const T* __restrict__ b_delta,
    T* __restrict__ log_h_out,
    T* __restrict__ sign_h_out,
    T* __restrict__ h_linear_out,
    T* __restrict__ log_v_cache,
    T* __restrict__ sign_v_cache,
    T* __restrict__ log_h_unbounded_cache,
    T* __restrict__ delta_cache,
    T* __restrict__ weight_rh_cache,
    T* __restrict__ rdelta_h_cache) {       // NEW: cache r_delta * h for backward

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float log_hp = static_cast<float>(log_h_prev[idx]);
        float sign_hp = static_cast<float>(sign_h_prev[idx]);
        float log_rh = fminf(static_cast<float>(log_r_h[d]), -0.1f);  // Clamp for stability
        float sign_rh = static_cast<float>(sign_r_h[d]);
        float log_rd = static_cast<float>(log_r_delta[d]);  // R_delta diagonal
        float sign_rd = static_cast<float>(sign_r_delta[d]);
        float wx = static_cast<float>(wx_x[idx]);
        float bias = static_cast<float>(b[d]);

        // Diagonal recurrence: r_h[d] * h_prev[d] in log space
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

        // tanh activation
        float v_linear = from_log_space(log_v, sign_v);
        float cand_linear = tanhf(v_linear);
        float log_cand_bounded, sign_cand;
        to_log_space(cand_linear, log_cand_bounded, sign_cand);
        if (log_h_unbounded_cache) log_h_unbounded_cache[idx] = static_cast<T>(v_linear);

        // Delta gate with R_delta modulation:
        // delta_raw = W_delta @ x + r_delta * h_prev + b_delta
        float wdelta = static_cast<float>(wdelta_x[idx]);
        float h_prev_linear = from_log_space(log_hp, sign_hp);
        float r_delta_linear = from_log_space(log_rd, sign_rd);
        float rdelta_h = r_delta_linear * h_prev_linear;  // r_delta * h_prev
        if (rdelta_h_cache) rdelta_h_cache[idx] = static_cast<T>(rdelta_h);

        float delta_raw_val = wdelta + rdelta_h + static_cast<float>(b_delta[d]);
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

        float h_lin = from_log_space(log_h_new, sign_h_new);
        h_linear_out[idx] = static_cast<T>(h_lin);
    }
}

// =============================================================================
// Forward: Selective Output (compete x silu)
// =============================================================================

template<typename T>
__global__ void LogDiagTripleRSelectiveOutput(
    const int batch_size,
    const int dim,
    const int n_groups,
    const int group_size,
    const T* __restrict__ h_linear,
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

    // Compute softmax
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

    // Output: compete * silu(W_out @ h)
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float compete = expf(static_cast<float>(h_linear[base + i]) - max_val) / sum;
        if (compete_cache) compete_cache[base + i] = static_cast<T>(compete);

        float w = static_cast<float>(w_out_h[base + i]);
        float silu_val = w / (1.0f + expf(-w));
        output[base + i] = static_cast<T>(compete * silu_val);
    }
}

// =============================================================================
// Backward Kernels
// =============================================================================

template<typename T>
__global__ void LogDiagTripleRSelectiveOutputBackward(
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

template<typename T>
__global__ void LogDiagTripleRUpdateBackward(
    const int batch_size,
    const int dim,
    const T* __restrict__ log_h_prev,
    const T* __restrict__ sign_h_prev,
    const T* __restrict__ log_v,
    const T* __restrict__ sign_v,
    const T* __restrict__ log_h_unbounded,
    const T* __restrict__ delta,
    const T* __restrict__ weight_rh,
    const T* __restrict__ rdelta_h,         // cached r_delta * h
    const T* __restrict__ log_r_h,
    const T* __restrict__ sign_r_h,
    const T* __restrict__ log_r_delta,
    const T* __restrict__ sign_r_delta,
    const T* __restrict__ d_h_linear,
    const T* __restrict__ d_log_h_recurrent,
    T* __restrict__ d_log_h_prev,
    T* __restrict__ d_wx_x,
    T* __restrict__ d_wdelta_x,             // NEW: gradient for W_delta @ x
    float* __restrict__ d_log_r_h,
    float* __restrict__ d_log_r_delta,      // NEW: gradient for diagonal R_delta
    float* __restrict__ d_b,
    float* __restrict__ d_b_delta) {        // NEW: gradient for b_delta

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float grad_h_linear = static_cast<float>(d_h_linear[idx]);
        float grad_log_h = 0.0f;
        if (d_log_h_recurrent) {
            grad_log_h += static_cast<float>(d_log_h_recurrent[idx]);
        }

        float log_hp = static_cast<float>(log_h_prev[idx]);
        float sign_hp = static_cast<float>(sign_h_prev[idx]);
        float log_v_val = static_cast<float>(log_v[idx]);
        float sign_v_val = static_cast<float>(sign_v[idx]);
        float log_h_unb = static_cast<float>(log_h_unbounded[idx]);
        float del = static_cast<float>(delta[idx]);
        float w_rh = static_cast<float>(weight_rh[idx]);
        float rdelta_h_val = static_cast<float>(rdelta_h[idx]);
        float log_rh = fminf(static_cast<float>(log_r_h[d]), -0.1f);
        float log_rd = static_cast<float>(log_r_delta[d]);
        float sign_rd = static_cast<float>(sign_r_delta[d]);

        float one_minus_delta = 1.0f - del;

        float h_linear_approx = from_log_space(log_hp, sign_hp);
        float d_log_h_from_linear = grad_h_linear * fabsf(h_linear_approx + 1e-6f);
        grad_log_h += d_log_h_from_linear;

        float d_log_cand_bounded = grad_log_h * del;
        float d_log_hp_decay = grad_log_h * one_minus_delta;

        // Backward through tanh
        float v_linear = log_h_unb;
        float tanh_v = tanhf(v_linear);
        float tanh_grad = 1.0f - tanh_v * tanh_v;

        float d_cand_linear = d_log_cand_bounded * fabsf(tanh_v + 1e-6f);
        float d_v_linear = d_cand_linear * tanh_grad;

        float v_from_log = from_log_space(log_v_val, sign_v_val);
        float d_log_v = d_v_linear * fabsf(v_from_log + 1e-6f);

        float d_log_rh_hp = d_log_v * w_rh;
        float d_log_input = d_log_v * (1.0f - fminf(w_rh, 1.0f));

        float d_log_hp_rh = d_log_rh_hp;
        float d_log_rh_val = d_log_rh_hp;

        // Delta gradient with R_delta contribution
        float h_prev_linear = from_log_space(log_hp, sign_hp);
        float d_delta_val = grad_log_h * (tanh_v - h_prev_linear);
        float d_delta_raw_val = d_delta_val * del * one_minus_delta;

        // Gradient for W_delta @ x
        d_wdelta_x[idx] = static_cast<T>(fminf(fmaxf(d_delta_raw_val, -GRAD_CLIP), GRAD_CLIP));

        // Gradient for r_delta (through r_delta * h_prev)
        float r_delta_linear = from_log_space(log_rd, sign_rd);
        float d_rdelta_h = d_delta_raw_val;  // d_delta_raw / d(r_delta * h_prev)
        float d_log_r_delta_val = d_rdelta_h * h_prev_linear;  // d/d(r_delta)
        atomicAdd(&d_log_r_delta[d], d_log_r_delta_val);

        // Gradient contribution to h_prev from R_delta
        float d_hp_from_rdelta = d_delta_raw_val * r_delta_linear;

        // Total h_prev gradient
        float d_log_hp_total = d_log_hp_decay + d_log_hp_rh;
        // Add contribution from R_delta path (in log space approximately)
        d_log_hp_total += d_hp_from_rdelta * fabsf(h_prev_linear + 1e-6f);
        d_log_h_prev[idx] = static_cast<T>(d_log_hp_total);

        // Input gradients
        float input_linear = from_log_space(log_v_val - log_rh - log_hp, 1.0f);
        float d_wx_linear = d_log_input * fmaxf(fabsf(input_linear), LOG_EPS);
        d_wx_x[idx] = static_cast<T>(fminf(fmaxf(d_wx_linear, -GRAD_CLIP), GRAD_CLIP));

        // Accumulate diagonal gradients
        atomicAdd(&d_log_r_h[d], d_log_rh_val);
        atomicAdd(&d_b[d], d_wx_linear);
        atomicAdd(&d_b_delta[d], d_delta_raw_val);
    }
}

// Kernel: Copy float array to type T
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
// LogDiagTripleRForward Implementation
// =============================================================================

template<typename T>
LogDiagTripleRForward<T>::LogDiagTripleRForward(
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
void LogDiagTripleRForward<T>::Run(
    int steps,
    const T* W_x,
    const T* log_r_h,
    const T* sign_r_h,
    const T* log_r_delta,
    const T* sign_r_delta,
    const T* W_delta,
    const T* W_out,
    const T* b,
    const T* b_delta,
    const T* log_gamma,
    const T* x,
    T* log_h,
    T* sign_h,
    T* output,
    T* h_linear_cache,
    T* log_v_cache,
    T* sign_v_cache,
    T* log_h_unbounded_cache,
    T* delta_cache,
    T* weight_rh_cache,
    T* rdelta_h_cache,
    T* compete_cache,
    T* log_rms_cache) {

    const int BD = batch_size_ * dim_;
    const int group_size = dim_ / n_groups_;
    const int block_size = 256;

    // BLAS constants
    const T alpha_one = static_cast<T>(1.0f);
    const T beta_zero = static_cast<T>(0.0f);

    // Pre-allocate workspace for all timesteps
    T* all_wx_x;
    T* all_wdelta_x;
    T* w_out_h;
    cudaMalloc(&all_wx_x, steps * BD * sizeof(T));
    cudaMalloc(&all_wdelta_x, steps * BD * sizeof(T));
    cudaMalloc(&w_out_h, BD * sizeof(T));

    // Pre-compute W_x @ x for ALL timesteps
    blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_, &alpha_one, W_x, dim_, x, dim_, &beta_zero, all_wx_x, dim_);

    // Pre-compute W_delta @ x for ALL timesteps
    blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_, &alpha_one, W_delta, dim_, x, dim_, &beta_zero, all_wdelta_x, dim_);

    for (int t = 0; t < steps; ++t) {
        const T* log_h_prev = log_h + t * BD;
        const T* sign_h_prev = sign_h + t * BD;
        T* log_h_out = log_h + (t + 1) * BD;
        T* sign_h_out = sign_h + (t + 1) * BD;
        T* output_t = output + t * BD;

        const T* wx_x_t = all_wx_x + t * BD;
        const T* wdelta_x_t = all_wdelta_x + t * BD;

        T* h_lin_t = training_ ? h_linear_cache + t * BD : nullptr;
        T* log_v_t = training_ ? log_v_cache + t * BD : nullptr;
        T* sign_v_t = training_ ? sign_v_cache + t * BD : nullptr;
        T* log_h_unb_t = training_ ? log_h_unbounded_cache + t * BD : nullptr;
        T* delta_t = training_ ? delta_cache + t * BD : nullptr;
        T* weight_rh_t = training_ ? weight_rh_cache + t * BD : nullptr;
        T* rdelta_h_t = training_ ? rdelta_h_cache + t * BD : nullptr;
        T* compete_t = training_ ? compete_cache + t * BD : nullptr;
        T* log_rms_t = training_ ? log_rms_cache + t * batch_size_ : nullptr;

        // Update kernel with R_delta modulation
        const int num_blocks = (BD + block_size - 1) / block_size;
        LogDiagTripleRUpdateKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_,
            log_h_prev, sign_h_prev,
            wx_x_t,
            log_r_h, sign_r_h,
            log_r_delta, sign_r_delta,
            wdelta_x_t,
            b, b_delta,
            log_h_out, sign_h_out,
            h_lin_t,
            log_v_t, sign_v_t,
            log_h_unb_t,
            delta_t,
            weight_rh_t,
            rdelta_h_t);

        // RMSNorm
        const int smem_size = block_size * sizeof(float);
        LogDiagTripleRRMSNormKernel<T><<<batch_size_, block_size, smem_size, stream_>>>(
            batch_size_, dim_,
            log_h_out, sign_h_out,
            log_gamma,
            log_h_out,  // in-place normalization of log_h
            h_lin_t ? h_lin_t : w_out_h,  // temp buffer if not training
            log_rms_t);

        // GEMM: W_out @ h_linear -> w_out_h
        T* h_source = h_lin_t ? h_lin_t : const_cast<T*>(log_h_out);  // Use log_h_out as temp if needed
        blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha_one, W_out, dim_, h_source, dim_, &beta_zero, w_out_h, dim_);

        // Selective output
        const int sel_smem = block_size * 2 * sizeof(float);
        dim3 sel_grid(batch_size_, n_groups_);
        LogDiagTripleRSelectiveOutput<T><<<sel_grid, block_size, sel_smem, stream_>>>(
            batch_size_, dim_, n_groups_, group_size,
            h_source,
            w_out_h,
            output_t,
            compete_t);
    }

    cudaFree(all_wx_x);
    cudaFree(all_wdelta_x);
    cudaFree(w_out_h);
}

// =============================================================================
// LogDiagTripleRBackward Implementation
// =============================================================================

template<typename T>
LogDiagTripleRBackward<T>::LogDiagTripleRBackward(
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
void LogDiagTripleRBackward<T>::Run(
    int steps,
    const T* W_x,
    const T* log_r_h,
    const T* sign_r_h,
    const T* log_r_delta,
    const T* sign_r_delta,
    const T* W_delta,
    const T* W_out,
    const T* log_gamma,
    const T* x,
    const T* log_h,
    const T* sign_h,
    const T* log_v_cache,
    const T* sign_v_cache,
    const T* log_h_unbounded_cache,
    const T* delta_cache,
    const T* weight_rh_cache,
    const T* rdelta_h_cache,
    const T* h_linear_cache,
    const T* compete_cache,
    const T* log_rms_cache,
    const T* d_output,
    T* dx,
    T* dW_x,
    T* d_log_r_h,
    T* d_log_r_delta,
    T* dW_delta,
    T* dW_out,
    T* db,
    T* db_delta,
    T* d_log_gamma,
    T* workspace) {

    const int BD = batch_size_ * dim_;
    const int group_size = dim_ / n_groups_;
    const int block_size = 256;

    // BLAS constants
    const T alpha_one = static_cast<T>(1.0f);
    const T beta_zero = static_cast<T>(0.0f);

    // Workspace layout
    T* dh_linear = workspace;                    // [BD]
    T* d_w_out_h = workspace + BD;               // [BD]
    T* d_log_h_recurrent = workspace + 2 * BD;   // [BD]
    T* d_log_h_prev = workspace + 3 * BD;        // [BD]
    T* d_wx_x = workspace + 4 * BD;              // [BD]
    T* d_wdelta_x = workspace + 5 * BD;          // [BD]
    T* w_out_h = workspace + 6 * BD;             // [BD] temp for GEMM
    float* d_log_r_h_f = reinterpret_cast<float*>(workspace + 7 * BD);  // [dim]
    float* d_log_r_delta_f = d_log_r_h_f + dim_; // [dim]
    float* db_f = d_log_r_delta_f + dim_;        // [dim]
    float* db_delta_f = db_f + dim_;             // [dim]
    float* d_log_gamma_f = db_delta_f + dim_;    // [dim]

    // Zero accumulators
    cudaMemsetAsync(d_log_r_h_f, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(d_log_r_delta_f, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_f, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_delta_f, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(d_log_gamma_f, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(d_log_h_recurrent, 0, BD * sizeof(T), stream_);

    for (int t = steps - 1; t >= 0; --t) {
        const T* x_t = x + t * BD;
        const T* log_h_prev = log_h + t * BD;
        const T* sign_h_prev = sign_h + t * BD;
        const T* log_h_out = log_h + (t + 1) * BD;
        const T* sign_h_out = sign_h + (t + 1) * BD;
        const T* d_output_t = d_output + t * BD;

        const T* h_lin_t = h_linear_cache + t * BD;
        const T* log_v_t = log_v_cache + t * BD;
        const T* sign_v_t = sign_v_cache + t * BD;
        const T* log_h_unb_t = log_h_unbounded_cache + t * BD;
        const T* delta_t = delta_cache + t * BD;
        const T* weight_rh_t = weight_rh_cache + t * BD;
        const T* rdelta_h_t = rdelta_h_cache + t * BD;
        const T* compete_t = compete_cache + t * BD;
        const T* log_rms_t = log_rms_cache + t * batch_size_;

        T* dx_t = dx + t * BD;

        // GEMM: W_out @ h_linear -> w_out_h
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha_one, W_out, dim_, h_lin_t, dim_, &beta_zero, w_out_h, dim_);

        // Backward through selective output
        const int sel_smem = block_size * sizeof(float);
        dim3 sel_grid(batch_size_, n_groups_);
        LogDiagTripleRSelectiveOutputBackward<T><<<sel_grid, block_size, sel_smem, stream_>>>(
            batch_size_, dim_, n_groups_, group_size,
            h_lin_t, w_out_h, compete_t, d_output_t,
            dh_linear, d_w_out_h);

        // Accumulate W_out gradient
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_, &alpha_one, d_w_out_h, dim_, h_lin_t, dim_, &alpha_one, dW_out, dim_);

        // Backward through RMSNorm
        const int norm_smem = block_size * sizeof(float);
        LogDiagTripleRRMSNormBackward<T><<<batch_size_, block_size, norm_smem, stream_>>>(
            batch_size_, dim_,
            log_h_out, sign_h_out,
            log_gamma, log_rms_t,
            dh_linear,
            d_log_h_prev,
            d_log_gamma_f);

        // Add gradient from d_w_out_h back to h_linear via W_out^T
        blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha_one, W_out, dim_, d_w_out_h, dim_, &alpha_one, dh_linear, dim_);

        // Backward through update kernel
        const int num_blocks = (BD + block_size - 1) / block_size;
        LogDiagTripleRUpdateBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_,
            log_h_prev, sign_h_prev,
            log_v_t, sign_v_t,
            log_h_unb_t,
            delta_t,
            weight_rh_t,
            rdelta_h_t,
            log_r_h, sign_r_h,
            log_r_delta, sign_r_delta,
            dh_linear, d_log_h_recurrent,
            d_log_h_prev,
            d_wx_x,
            d_wdelta_x,
            d_log_r_h_f,
            d_log_r_delta_f,
            db_f,
            db_delta_f);

        // Copy d_log_h_prev to d_log_h_recurrent for next iteration
        cudaMemcpyAsync(d_log_h_recurrent, d_log_h_prev, BD * sizeof(T), cudaMemcpyDeviceToDevice, stream_);

        // Accumulate W_x gradient: d_wx_x @ x^T
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_, &alpha_one, d_wx_x, dim_, x_t, dim_, &alpha_one, dW_x, dim_);

        // Accumulate W_delta gradient
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_, &alpha_one, d_wdelta_x, dim_, x_t, dim_, &alpha_one, dW_delta, dim_);

        // Compute dx: W_x^T @ d_wx_x + W_delta^T @ d_wdelta_x
        blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha_one, W_x, dim_, d_wx_x, dim_, &beta_zero, dx_t, dim_);

        blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha_one, W_delta, dim_, d_wdelta_x, dim_, &alpha_one, dx_t, dim_);
    }

    // Copy float gradients to output type
    const int copy_blocks = (dim_ + block_size - 1) / block_size;
    CopyFloatToT<T><<<copy_blocks, block_size, 0, stream_>>>(dim_, d_log_r_h_f, d_log_r_h);
    CopyFloatToT<T><<<copy_blocks, block_size, 0, stream_>>>(dim_, d_log_r_delta_f, d_log_r_delta);
    CopyFloatToT<T><<<copy_blocks, block_size, 0, stream_>>>(dim_, db_f, db);
    CopyFloatToT<T><<<copy_blocks, block_size, 0, stream_>>>(dim_, db_delta_f, db_delta);
    CopyFloatToT<T><<<copy_blocks, block_size, 0, stream_>>>(dim_, d_log_gamma_f, d_log_gamma);
}

// Explicit instantiations
template class LogDiagTripleRForward<float>;
template class LogDiagTripleRForward<__half>;
template class LogDiagTripleRForward<__nv_bfloat16>;
template class LogDiagTripleRForward<double>;
template class LogDiagTripleRBackward<float>;
template class LogDiagTripleRBackward<__half>;
template class LogDiagTripleRBackward<__nv_bfloat16>;
template class LogDiagTripleRBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
