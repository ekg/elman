// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// Log-Space Level 2: Diagonal Selective Log-Space Polynomial
// Like log_1 but uses diagonal recurrence (like Mamba2's diagonal A).
//
// Architecture:
//   α_t = 1 + softplus(W_α @ x_t + b_α)        -- input-dependent exponent
//   v = r_h * h_prev + W_x @ x + b             -- DIAGONAL r_h (element-wise)
//   log|h_cand| = α_t * log|v|                 -- polynomial nonlinearity
//   log|h_bounded| = -softplus(-log|h_cand|)   -- soft bound magnitude ≤ 1
//   δ = sigmoid(W_δ @ x_t + b_δ)
//   h_new = (1-δ) * h_prev + δ * h_bounded     -- gated update
//
//   // Selective output
//   compete = softmax(h.reshape(groups), dim=-1)
//   output = compete * silu(W_out @ h)
//
// Key difference from log_1:
// - Uses diagonal r_h vector instead of full W_h matrix
// - r_h is stored in log space: (log|r_h|, sign(r_h))

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cfloat>

#include "blas.h"
#include "inline_ops.h"

namespace {

constexpr float LOG_ZERO = -40.0f;
constexpr float LOG_EPS = 1e-10f;
constexpr float GRAD_CLIP = 10.0f;

// Device functions (same as log_0/log_1)
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
// Log-Space RMSNorm Forward Kernel (same as log_1)
// =============================================================================

template<typename T>
__global__ void LogDiagRMSNormKernel(
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

    // logsumexp(2*log_h) = max + log(sum_exp)
    float log_sum_h2 = max_log_h2 + logf(fmaxf(sum_exp, LOG_EPS));

    // log_rms = logsumexp(2*log_h) / 2 - log(dim) / 2
    //         = log(sqrt(sum(h^2) / dim))
    float log_rms = (log_sum_h2 - log_dim) * 0.5f;

    // Cache log_rms for backward pass
    if (log_rms_cache && threadIdx.x == 0) {
        log_rms_cache[b] = static_cast<T>(log_rms);
    }

    // Step 2: Normalize each element
    // h_normed = h / rms * gamma
    // log(h_normed) = log(h) - log(rms) + log(gamma)
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float log_h_val = static_cast<float>(log_h[base + d]);
        float sign_h_val = static_cast<float>(sign_h[base + d]);
        float log_gamma_val = static_cast<float>(log_gamma[d]);

        float log_h_norm = log_h_val - log_rms + log_gamma_val;
        log_h_normed[base + d] = static_cast<T>(log_h_norm);

        // Convert to linear space
        float h_norm = from_log_space(log_h_norm, sign_h_val);
        h_linear_normed[base + d] = static_cast<T>(h_norm);
    }
}

// =============================================================================
// Log-Space RMSNorm Backward Kernel
// =============================================================================

template<typename T>
__global__ void LogDiagRMSNormBackward(
    const int batch_size,
    const int dim,
    const T* __restrict__ log_h,        // [B, dim] input log magnitudes
    const T* __restrict__ sign_h,       // [B, dim] signs
    const T* __restrict__ log_gamma,    // [dim] learnable scale
    const T* __restrict__ log_rms,      // [B] cached log_rms from forward
    const T* __restrict__ d_h_linear,   // [B, dim] gradient from output
    T* __restrict__ d_log_h,            // [B, dim] gradient to input
    float* __restrict__ d_log_gamma) {  // [dim] gradient to log_gamma (atomic)

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
        // Scaled by 0.0001 to match Python grad_scale and stabilize training
        float d_log_gamma_val = d_out * h_norm * 0.0001f;
        atomicAdd(&d_log_gamma[d], d_log_gamma_val);

        // Gradient w.r.t. log_h (simplified for bounded gradients)
        // d_log_h = d_out * h_norm * (1 - h_norm^2 / sum_h2)
        // Approximated as: d_out * h_norm - mean_dout_hnorm * h_norm
        // Scaled by 0.0001 to match Python grad_scale and stabilize training
        float d_log_h_val = (d_out - mean_dout_hnorm) * fabsf(h_norm + 1e-6f) * 0.0001f;
        d_log_h[base + d] = static_cast<T>(fminf(fmaxf(d_log_h_val, -GRAD_CLIP), GRAD_CLIP));
    }
}

// =============================================================================
// Forward: Diagonal Log-Space Polynomial Update
// =============================================================================

template<typename T>
__global__ void LogDiagSelectiveUpdateKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ log_h_prev,
    const T* __restrict__ sign_h_prev,
    const T* __restrict__ wx_x,           // [B, dim] W_x @ x
    const T* __restrict__ alpha_raw,
    const T* __restrict__ log_r_h,        // [dim] diagonal recurrence (log space)
    const T* __restrict__ sign_r_h,       // [dim]
    const T* __restrict__ b,
    const T* __restrict__ delta_raw,
    T* __restrict__ log_h_out,
    T* __restrict__ sign_h_out,
    T* __restrict__ h_linear_out,
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
        float log_rh = static_cast<float>(log_r_h[d]);      // Diagonal element
        float sign_rh = static_cast<float>(sign_r_h[d]);
        float wx = static_cast<float>(wx_x[idx]);
        float bias = static_cast<float>(b[d]);

        // Input-dependent alpha
        float alpha_raw_val = static_cast<float>(alpha_raw[idx]);
        float alpha = 1.0f + stable_softplus(alpha_raw_val);
        if (alpha_cache) alpha_cache[idx] = static_cast<T>(alpha);

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

        // Polynomial activation
        float log_cand = alpha * log_v;
        float sign_cand = sign_v;

        // Soft bound
        float log_cand_bounded = soft_bound_log(log_cand);
        if (log_h_unbounded_cache) log_h_unbounded_cache[idx] = static_cast<T>(log_cand);

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

        float h_lin = from_log_space(log_h_new, sign_h_new);
        h_linear_out[idx] = static_cast<T>(h_lin);
    }
}

// =============================================================================
// Forward: Selective Output (same as log_1)
// =============================================================================

template<typename T>
__global__ void LogDiagSelectiveOutput(
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

    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float compete = expf(static_cast<float>(h_linear[base + i]) - max_val) / sum;
        if (compete_cache) compete_cache[base + i] = static_cast<T>(compete);

        float w = static_cast<float>(w_out_h[base + i]);
        float silu_val = w / (1.0f + expf(-w));
        output[base + i] = static_cast<T>(compete * silu_val);
    }
}

// =============================================================================
// Backward kernels
// =============================================================================

template<typename T>
__global__ void LogDiagSelectiveOutputBackward(
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
__global__ void LogDiagSelectiveUpdateBackward(
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
    const T* __restrict__ d_h_linear,
    const T* __restrict__ d_log_h_recurrent,
    T* __restrict__ d_log_h_prev,
    T* __restrict__ d_wx_x,
    T* __restrict__ d_alpha_raw,
    T* __restrict__ d_delta_raw,
    float* __restrict__ d_log_r_h,        // Gradient for diagonal (float for atomicAdd)
    float* __restrict__ d_b) {

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
        float alpha_val = static_cast<float>(alpha[idx]);
        float alpha_raw_val = static_cast<float>(alpha_raw[idx]);
        float log_h_unb = static_cast<float>(log_h_unbounded[idx]);
        float del = static_cast<float>(delta[idx]);
        float w_rh = static_cast<float>(weight_rh[idx]);
        float log_rh = static_cast<float>(log_r_h[d]);

        float one_minus_delta = 1.0f - del;

        float h_linear_approx = from_log_space(log_hp, sign_hp);
        float d_log_h_from_linear = grad_h_linear * fabsf(h_linear_approx + 1e-6f);
        grad_log_h += d_log_h_from_linear;

        float d_log_cand_bounded = grad_log_h * del;
        float d_log_hp_decay = grad_log_h * one_minus_delta;

        float bound_grad = soft_bound_grad(log_h_unb);
        float d_log_cand = d_log_cand_bounded * bound_grad;

        float d_log_v = alpha_val * d_log_cand;
        float d_alpha = log_v_val * d_log_cand;

        float d_alpha_raw_val = d_alpha * softplus_grad(alpha_raw_val);
        d_alpha_raw[idx] = static_cast<T>(fminf(fmaxf(d_alpha_raw_val, -GRAD_CLIP), GRAD_CLIP));

        float d_log_rh_hp = d_log_v * w_rh;
        float d_log_input = d_log_v * (1.0f - fminf(w_rh, 1.0f));

        float d_log_hp_rh = d_log_rh_hp;
        float d_log_rh_val = d_log_rh_hp;  // Gradient for diagonal r_h

        float d_log_hp_total = d_log_hp_decay + d_log_hp_rh;
        d_log_h_prev[idx] = static_cast<T>(d_log_hp_total);

        float h_prev_linear = from_log_space(log_hp, sign_hp);
        float cand_linear = from_log_space(log_h_unb, sign_v_val);
        cand_linear = cand_linear / (1.0f + fabsf(cand_linear));
        float d_delta_val = grad_log_h * (cand_linear - h_prev_linear);
        float d_delta_raw_val = d_delta_val * del * one_minus_delta;
        d_delta_raw[idx] = static_cast<T>(fminf(fmaxf(d_delta_raw_val, -GRAD_CLIP), GRAD_CLIP));

        float input_linear = from_log_space(log_v_val - log_rh - log_hp, 1.0f);
        float d_wx_linear = d_log_input * fmaxf(fabsf(input_linear), LOG_EPS);
        d_wx_x[idx] = static_cast<T>(fminf(fmaxf(d_wx_linear, -GRAD_CLIP), GRAD_CLIP));

        // Accumulate diagonal gradient
        atomicAdd(&d_log_r_h[d], d_log_rh_val);
        atomicAdd(&d_b[d], d_wx_linear);
    }
}

}  // anonymous namespace


namespace hasty {
namespace v0 {
namespace elman_ladder {

// =============================================================================
// Log-Space Diagonal Selective Elman Forward
// =============================================================================

template<typename T>
struct LogDiagSelectiveElmanForward {
    bool training_;
    int batch_size_;
    int dim_;
    int n_groups_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;

    LogDiagSelectiveElmanForward(
        bool training, int batch_size, int dim, int n_groups,
        const cublasHandle_t& blas_handle, const cudaStream_t& stream)
        : training_(training), batch_size_(batch_size), dim_(dim),
          n_groups_(n_groups), blas_handle_(blas_handle), stream_(stream) {}

    void Run(
        int steps,
        const T* W_x,
        const T* log_r_h,          // [dim] diagonal
        const T* sign_r_h,         // [dim]
        const T* W_alpha,
        const T* b_alpha,
        const T* W_delta,
        const T* W_out,
        const T* b,
        const T* b_delta,
        const T* log_gamma,        // NEW: RMSNorm scale in log-space
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
        T* h_linear_cache,
        T* compete_cache,
        T* log_rms_cache) {        // NEW: cache for backward

        static const T alpha_one = static_cast<T>(1.0);
        static const T beta_zero = static_cast<T>(0.0);

        const int BD = batch_size_ * dim_;
        const int block_size = 256;
        const int num_blocks = (BD + block_size - 1) / block_size;
        const int group_size = dim_ / n_groups_;
        const int norm_smem_size = 2 * block_size * sizeof(float);

        T *wx_x, *alpha_raw, *delta_tmp, *w_out_h, *h_linear_tmp, *log_h_normed_tmp;
        cudaMalloc(&wx_x, BD * sizeof(T));
        cudaMalloc(&alpha_raw, BD * sizeof(T));
        cudaMalloc(&delta_tmp, BD * sizeof(T));
        cudaMalloc(&w_out_h, BD * sizeof(T));
        cudaMalloc(&h_linear_tmp, BD * sizeof(T));
        cudaMalloc(&log_h_normed_tmp, BD * sizeof(T));

        for (int t = 0; t < steps; ++t) {
            const T* x_t = x + t * BD;
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

            // wx_x = x_t @ W_x.T (matching PyTorch convention)
            blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                dim_, batch_size_, dim_, &alpha_one, W_x, dim_, x_t, dim_, &beta_zero, wx_x, dim_);

            // alpha_raw = x_t @ W_alpha.T
            blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                dim_, batch_size_, dim_, &alpha_one, W_alpha, dim_, x_t, dim_, &beta_zero, alpha_raw, dim_);

            // delta_tmp = x_t @ W_delta.T
            blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                dim_, batch_size_, dim_, &alpha_one, W_delta, dim_, x_t, dim_, &beta_zero, delta_tmp, dim_);

            if (alpha_raw_t) {
                cudaMemcpyAsync(alpha_raw_t, alpha_raw, BD * sizeof(T), cudaMemcpyDeviceToDevice, stream_);
            }

            // Diagonal log-space update (produces unnormalized h_linear)
            LogDiagSelectiveUpdateKernel<T><<<num_blocks, block_size, 0, stream_>>>(
                batch_size_, dim_,
                log_h_prev, sign_h_prev, wx_x, alpha_raw, log_r_h, sign_r_h, b, delta_tmp,
                log_h_t, sign_h_t, h_linear_tmp,  // Use temp buffer for unnormalized
                log_v_t, sign_v_t, alpha_t, log_h_unb_t, delta_t, weight_t);

            // Apply LogSpaceRMSNorm - writes normalized h_linear to h_linear_t
            LogDiagRMSNormKernel<T><<<batch_size_, block_size, norm_smem_size, stream_>>>(
                batch_size_, dim_,
                log_h_t, sign_h_t, log_gamma,
                log_h_normed_tmp,  // normalized log magnitudes (scratch)
                h_linear_t,        // normalized h_linear (cached)
                log_rms_t);        // cache log_rms for backward

            // w_out_h = h_linear_normed @ W_out.T (matching PyTorch convention)
            blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                dim_, batch_size_, dim_, &alpha_one, W_out, dim_, h_linear_t, dim_, &beta_zero, w_out_h, dim_);

            // Selective output using NORMALIZED h_linear
            dim3 grid(batch_size_, n_groups_);
            int smem_size = 2 * block_size * sizeof(float);
            LogDiagSelectiveOutput<T><<<grid, block_size, smem_size, stream_>>>(
                batch_size_, dim_, n_groups_, group_size, h_linear_t, w_out_h, out_t, compete_t);
        }

        cudaFree(wx_x);
        cudaFree(alpha_raw);
        cudaFree(delta_tmp);
        cudaFree(w_out_h);
        cudaFree(h_linear_tmp);
        cudaFree(log_h_normed_tmp);
    }
};

// =============================================================================
// Log-Space Diagonal Selective Elman Backward
// =============================================================================

template<typename T>
struct LogDiagSelectiveElmanBackward {
    int batch_size_;
    int dim_;
    int n_groups_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;

    LogDiagSelectiveElmanBackward(
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
        const T* log_gamma,        // NEW: RMSNorm scale
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
        const T* h_linear_cache,
        const T* compete_cache,
        const T* log_rms_cache,    // NEW: cached log_rms from forward
        const T* d_output,
        T* dx,
        T* dW_x,
        T* d_log_r_h,              // [dim] gradient for diagonal
        T* dW_alpha,
        T* db_alpha,
        T* dW_delta,
        T* dW_out,
        T* db,
        T* db_delta,
        T* d_log_gamma) {          // NEW: gradient for log_gamma

        static const T alpha_one = static_cast<T>(1.0);
        static const T beta_zero = static_cast<T>(0.0);

        const int BD = batch_size_ * dim_;
        const int block_size = 256;
        const int num_blocks = (BD + block_size - 1) / block_size;
        const int group_size = dim_ / n_groups_;
        const int norm_smem_size = block_size * sizeof(float);

        T *d_log_h_recurrent, *d_log_h_prev, *d_wx_x, *d_alpha_raw, *d_delta_raw;
        T *d_h_linear, *d_w_out_h, *w_out_h, *d_log_h;
        cudaMalloc(&d_log_h_recurrent, BD * sizeof(T));
        cudaMalloc(&d_log_h_prev, BD * sizeof(T));
        cudaMalloc(&d_wx_x, BD * sizeof(T));
        cudaMalloc(&d_alpha_raw, BD * sizeof(T));
        cudaMalloc(&d_delta_raw, BD * sizeof(T));
        cudaMalloc(&d_h_linear, BD * sizeof(T));
        cudaMalloc(&d_w_out_h, BD * sizeof(T));
        cudaMalloc(&w_out_h, BD * sizeof(T));
        cudaMalloc(&d_log_h, BD * sizeof(T));
        cudaMemset(d_log_h_recurrent, 0, BD * sizeof(T));

        float *d_log_r_h_float, *db_float, *db_delta_float, *db_alpha_float, *d_log_gamma_float;
        cudaMalloc(&d_log_r_h_float, dim_ * sizeof(float));
        cudaMalloc(&db_float, dim_ * sizeof(float));
        cudaMalloc(&db_delta_float, dim_ * sizeof(float));
        cudaMalloc(&db_alpha_float, dim_ * sizeof(float));
        cudaMalloc(&d_log_gamma_float, dim_ * sizeof(float));
        cudaMemset(d_log_r_h_float, 0, dim_ * sizeof(float));
        cudaMemset(db_float, 0, dim_ * sizeof(float));
        cudaMemset(db_delta_float, 0, dim_ * sizeof(float));
        cudaMemset(db_alpha_float, 0, dim_ * sizeof(float));
        cudaMemset(d_log_gamma_float, 0, dim_ * sizeof(float));

        cudaMemset(dW_x, 0, dim_ * dim_ * sizeof(T));
        cudaMemset(dW_alpha, 0, dim_ * dim_ * sizeof(T));
        cudaMemset(dW_delta, 0, dim_ * dim_ * sizeof(T));
        cudaMemset(dW_out, 0, dim_ * dim_ * sizeof(T));

        for (int t = steps - 1; t >= 0; --t) {
            const T* x_t = x + t * BD;
            const T* log_h_prev = log_h + t * BD;
            const T* sign_h_prev = sign_h + t * BD;
            const T* log_h_t = log_h + (t + 1) * BD;
            const T* sign_h_t = sign_h + (t + 1) * BD;

            const T* log_v_t = log_v_cache + t * BD;
            const T* sign_v_t = sign_v_cache + t * BD;
            const T* alpha_t = alpha_cache + t * BD;
            const T* alpha_raw_t = alpha_raw_cache + t * BD;
            const T* log_h_unb_t = log_h_unbounded_cache + t * BD;
            const T* delta_t = delta_cache + t * BD;
            const T* weight_t = weight_rh_cache + t * BD;
            const T* h_linear_t = h_linear_cache + t * BD;
            const T* compete_t = compete_cache + t * BD;
            const T* log_rms_t = log_rms_cache + t * batch_size_;

            const T* d_out_t = d_output + t * BD;
            T* dx_t = dx + t * BD;

            blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                dim_, batch_size_, dim_, &alpha_one, W_out, dim_, h_linear_t, dim_, &beta_zero, w_out_h, dim_);

            dim3 grid(batch_size_, n_groups_);
            int smem_size = block_size * sizeof(float);
            LogDiagSelectiveOutputBackward<T><<<grid, block_size, smem_size, stream_>>>(
                batch_size_, dim_, n_groups_, group_size,
                h_linear_t, w_out_h, compete_t, d_out_t, d_h_linear, d_w_out_h);

            blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
                dim_, dim_, batch_size_, &alpha_one, d_w_out_h, dim_, h_linear_t, dim_, &alpha_one, dW_out, dim_);

            blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                dim_, batch_size_, dim_, &alpha_one, W_out, dim_, d_w_out_h, dim_, &alpha_one, d_h_linear, dim_);

            // Backward through RMSNorm
            LogDiagRMSNormBackward<T><<<batch_size_, block_size, norm_smem_size, stream_>>>(
                batch_size_, dim_,
                log_h_t, sign_h_t, log_gamma, log_rms_t,
                d_h_linear, d_log_h, d_log_gamma_float);

            // Backward through log-space update (receives d_log_h from RMSNorm backward)
            LogDiagSelectiveUpdateBackward<T><<<num_blocks, block_size, 0, stream_>>>(
                batch_size_, dim_,
                log_h_prev, sign_h_prev, log_v_t, sign_v_t, alpha_t, alpha_raw_t,
                log_h_unb_t, delta_t, weight_t, log_r_h, sign_r_h,
                d_log_h, (t < steps - 1) ? d_log_h_recurrent : nullptr,
                d_log_h_prev, d_wx_x, d_alpha_raw, d_delta_raw, d_log_r_h_float, db_float);

            blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                dim_, batch_size_, dim_, &alpha_one, W_x, dim_, d_wx_x, dim_, &beta_zero, dx_t, dim_);
            blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                dim_, batch_size_, dim_, &alpha_one, W_alpha, dim_, d_alpha_raw, dim_, &alpha_one, dx_t, dim_);
            blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                dim_, batch_size_, dim_, &alpha_one, W_delta, dim_, d_delta_raw, dim_, &alpha_one, dx_t, dim_);

            blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
                dim_, dim_, batch_size_, &alpha_one, d_wx_x, dim_, x_t, dim_, &alpha_one, dW_x, dim_);
            blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
                dim_, dim_, batch_size_, &alpha_one, d_alpha_raw, dim_, x_t, dim_, &alpha_one, dW_alpha, dim_);
            blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
                dim_, dim_, batch_size_, &alpha_one, d_delta_raw, dim_, x_t, dim_, &alpha_one, dW_delta, dim_);

            cudaMemcpy(d_log_h_recurrent, d_log_h_prev, BD * sizeof(T), cudaMemcpyDeviceToDevice);
        }

        // Copy accumulated gradients
        if constexpr (std::is_same<T, float>::value) {
            cudaMemcpy(d_log_r_h, d_log_r_h_float, dim_ * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(db, db_float, dim_ * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_log_gamma, d_log_gamma_float, dim_ * sizeof(float), cudaMemcpyDeviceToDevice);
        }

        cudaFree(d_log_h_recurrent);
        cudaFree(d_log_h_prev);
        cudaFree(d_wx_x);
        cudaFree(d_alpha_raw);
        cudaFree(d_delta_raw);
        cudaFree(d_h_linear);
        cudaFree(d_w_out_h);
        cudaFree(w_out_h);
        cudaFree(d_log_h);
        cudaFree(d_log_r_h_float);
        cudaFree(db_float);
        cudaFree(db_delta_float);
        cudaFree(db_alpha_float);
        cudaFree(d_log_gamma_float);
    }
};

// Explicit template instantiations
template struct LogDiagSelectiveElmanForward<__half>;
template struct LogDiagSelectiveElmanForward<__nv_bfloat16>;
template struct LogDiagSelectiveElmanForward<float>;
template struct LogDiagSelectiveElmanForward<double>;

template struct LogDiagSelectiveElmanBackward<__half>;
template struct LogDiagSelectiveElmanBackward<__nv_bfloat16>;
template struct LogDiagSelectiveElmanBackward<float>;
template struct LogDiagSelectiveElmanBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
