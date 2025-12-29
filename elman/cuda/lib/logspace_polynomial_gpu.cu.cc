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
constexpr float GRAD_CLIP = 10.0f;     // Gradient clipping for numerical stability

// =============================================================================
// Device functions for signed log arithmetic
// =============================================================================

// Convert linear value to log space: x -> (log|x|, sign(x))
__device__ __forceinline__ void to_log_space(float x, float& log_x, float& sign_x) {
    sign_x = (x >= 0.0f) ? 1.0f : -1.0f;
    float abs_x = fabsf(x);
    log_x = (abs_x > LOG_EPS) ? logf(abs_x) : LOG_ZERO;
}

// Convert log space to linear: (log|x|, sign(x)) -> x
__device__ __forceinline__ float from_log_space(float log_x, float sign_x) {
    if (log_x <= LOG_ZERO + 1.0f) return 0.0f;
    float result = sign_x * expf(fminf(log_x, 20.0f));  // Clamp to prevent overflow
    return result;
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
        float log_rh = static_cast<float>(log_r_h[d]);
        float sign_rh = static_cast<float>(sign_r_h[d]);
        float wx = static_cast<float>(wx_x[idx]);
        float bias = static_cast<float>(b[d]);

        // === Compute input-dependent alpha ===
        // α = 1 + softplus(W_alpha @ x + b_alpha), guaranteed > 1
        float alpha_raw_val = static_cast<float>(alpha_raw[idx]);
        float alpha = 1.0f + stable_softplus(alpha_raw_val);
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

        // Load gradients
        float grad_log_h = static_cast<float>(d_log_h[idx]);
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
        float log_rh = static_cast<float>(log_r_h[d]);

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
        float d_log_v = alpha_val * d_log_cand;
        float d_alpha = log_v_val * d_log_cand;

        // Gradient through alpha = 1 + softplus(alpha_raw)
        // d_alpha_raw = d_alpha * sigmoid(alpha_raw)
        float d_alpha_raw_val = d_alpha * softplus_grad(alpha_raw_val);
        d_alpha_raw[idx] = static_cast<T>(fminf(fmaxf(d_alpha_raw_val, -GRAD_CLIP), GRAD_CLIP));

        // === Backward through signed log addition v = r_h*h_prev + input ===
        float d_log_rh_hp = d_log_v * w_rh;
        float d_log_input = d_log_v * (1.0f - fminf(w_rh, 1.0f));

        // === Backward through r_h * h_prev ===
        float d_log_hp_rh = d_log_rh_hp;
        float d_log_rh_val = d_log_rh_hp;

        // === Total gradient to log_h_prev ===
        float d_log_hp_total = d_log_hp_decay + d_log_hp_rh;
        d_log_h_prev[idx] = static_cast<T>(d_log_hp_total);

        // === Backward through delta gate ===
        float h_prev_linear = from_log_space(log_hp, sign_hp);
        float cand_linear = from_log_space(log_h_unb, sign_v_val);
        cand_linear = cand_linear / (1.0f + fabsf(cand_linear));  // Approximate bounded

        float d_delta = grad_log_h * (cand_linear - h_prev_linear);
        float d_delta_raw_val = d_delta * del * one_minus_delta;
        d_delta_raw[idx] = static_cast<T>(fminf(fmaxf(d_delta_raw_val, -GRAD_CLIP), GRAD_CLIP));

        // === Backward through W_x @ x + b ===
        float input_linear = from_log_space(log_v_val - log_rh - log_hp, 1.0f);
        float d_wx_linear = d_log_input * fmaxf(fabsf(input_linear), LOG_EPS);
        d_wx_x[idx] = static_cast<T>(fminf(fmaxf(d_wx_linear, -GRAD_CLIP), GRAD_CLIP));

        // === Accumulate parameter gradients ===
        atomicAdd(&d_log_r_h[d], d_log_rh_val);
        atomicAdd(&d_b[d], d_wx_linear);
    }
}

}  // anonymous namespace


namespace haste {
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
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;

    LogPolyElmanForward(
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

    void Run(
        int steps,
        const T* W_x,           // [dim, dim]
        const T* log_r_h,       // [dim]
        const T* sign_r_h,      // [dim]
        const T* W_alpha,       // [dim, dim] for input-dependent alpha
        const T* b_alpha,       // [dim]
        const T* W_delta,       // [dim, dim]
        const T* b,             // [dim]
        const T* b_delta,       // [dim]
        const T* x,             // [steps, B, dim]
        T* log_h,               // [steps+1, B, dim]
        T* sign_h,              // [steps+1, B, dim]
        T* h_linear,            // [steps, B, dim] for output (optional)
        T* log_v_cache,
        T* sign_v_cache,
        T* alpha_cache,
        T* log_h_unbounded_cache,
        T* delta_cache,
        T* weight_rh_cache,
        T* alpha_raw_cache) {   // Store alpha_raw for backward

        static const T alpha_one = static_cast<T>(1.0);
        static const T beta_zero = static_cast<T>(0.0);

        const int BD = batch_size_ * dim_;
        const int block_size = 256;
        const int num_blocks = (BD + block_size - 1) / block_size;

        // Workspace for GEMM results
        T *wx_x, *alpha_raw, *delta_tmp;
        cudaMalloc(&wx_x, BD * sizeof(T));
        cudaMalloc(&alpha_raw, BD * sizeof(T));
        cudaMalloc(&delta_tmp, BD * sizeof(T));

        for (int t = 0; t < steps; ++t) {
            const T* x_t = x + t * BD;
            const T* log_h_prev = log_h + t * BD;
            const T* sign_h_prev = sign_h + t * BD;
            T* log_h_t = log_h + (t + 1) * BD;
            T* sign_h_t = sign_h + (t + 1) * BD;

            // Caches for this timestep
            T* log_v_t = training_ ? (log_v_cache + t * BD) : nullptr;
            T* sign_v_t = training_ ? (sign_v_cache + t * BD) : nullptr;
            T* alpha_t = training_ ? (alpha_cache + t * BD) : nullptr;
            T* log_h_unb_t = training_ ? (log_h_unbounded_cache + t * BD) : nullptr;
            T* delta_t = training_ ? (delta_cache + t * BD) : nullptr;
            T* weight_t = training_ ? (weight_rh_cache + t * BD) : nullptr;
            T* alpha_raw_t = training_ ? (alpha_raw_cache + t * BD) : nullptr;

            // wx_x = W_x @ x_t
            blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                dim_, batch_size_, dim_, &alpha_one, W_x, dim_, x_t, dim_, &beta_zero, wx_x, dim_);

            // alpha_raw = W_alpha @ x_t + b_alpha
            blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                dim_, batch_size_, dim_, &alpha_one, W_alpha, dim_, x_t, dim_, &beta_zero, alpha_raw, dim_);
            // Add bias (fused in kernel would be better, but this works)

            // delta_tmp = W_delta @ x_t + b_delta
            blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                dim_, batch_size_, dim_, &alpha_one, W_delta, dim_, x_t, dim_, &beta_zero, delta_tmp, dim_);

            // Store alpha_raw for backward
            if (alpha_raw_t) {
                cudaMemcpyAsync(alpha_raw_t, alpha_raw, BD * sizeof(T), cudaMemcpyDeviceToDevice, stream_);
            }

            // Main update kernel (biases added inside)
            LogPolyGatedUpdateKernel<T><<<num_blocks, block_size, 0, stream_>>>(
                batch_size_, dim_,
                log_h_prev, sign_h_prev,
                wx_x, alpha_raw, log_r_h, sign_r_h, b, delta_tmp,
                log_h_t, sign_h_t,
                log_v_t, sign_v_t, alpha_t, log_h_unb_t, delta_t, weight_t);

            // Convert to linear for output
            if (h_linear) {
                LogToLinearKernel<T><<<num_blocks, block_size, 0, stream_>>>(
                    BD, log_h_t, sign_h_t, h_linear + t * BD);
            }
        }

        cudaFree(wx_x);
        cudaFree(alpha_raw);
        cudaFree(delta_tmp);
    }
};

// =============================================================================
// Log-Space Polynomial Elman Backward
// =============================================================================

template<typename T>
struct LogPolyElmanBackward {
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;

    LogPolyElmanBackward(
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream)
        : batch_size_(batch_size),
          dim_(dim),
          blas_handle_(blas_handle),
          stream_(stream) {}

    void Run(
        int steps,
        const T* W_x,
        const T* log_r_h,
        const T* sign_r_h,
        const T* W_alpha,
        const T* W_delta,
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
        const T* d_h_linear,
        T* dx,
        T* dW_x,
        T* d_log_r_h,
        T* dW_alpha,
        T* db_alpha,
        T* dW_delta,
        T* db,
        T* db_delta) {

        static const T alpha_one = static_cast<T>(1.0);
        static const T beta_zero = static_cast<T>(0.0);

        const int BD = batch_size_ * dim_;
        const int block_size = 256;
        const int num_blocks = (BD + block_size - 1) / block_size;

        // Workspace
        T *d_log_h_recurrent, *d_log_h_prev, *d_wx_x, *d_alpha_raw, *d_delta_raw;
        cudaMalloc(&d_log_h_recurrent, BD * sizeof(T));
        cudaMalloc(&d_log_h_prev, BD * sizeof(T));
        cudaMalloc(&d_wx_x, BD * sizeof(T));
        cudaMalloc(&d_alpha_raw, BD * sizeof(T));
        cudaMalloc(&d_delta_raw, BD * sizeof(T));
        cudaMemset(d_log_h_recurrent, 0, BD * sizeof(T));

        // Float buffers for atomic gradients
        float *d_log_r_h_float, *db_float, *db_delta_float, *db_alpha_float;
        cudaMalloc(&d_log_r_h_float, dim_ * sizeof(float));
        cudaMalloc(&db_float, dim_ * sizeof(float));
        cudaMalloc(&db_delta_float, dim_ * sizeof(float));
        cudaMalloc(&db_alpha_float, dim_ * sizeof(float));
        cudaMemset(d_log_r_h_float, 0, dim_ * sizeof(float));
        cudaMemset(db_float, 0, dim_ * sizeof(float));
        cudaMemset(db_delta_float, 0, dim_ * sizeof(float));
        cudaMemset(db_alpha_float, 0, dim_ * sizeof(float));

        // Zero weight gradients
        cudaMemset(dW_x, 0, dim_ * dim_ * sizeof(T));
        cudaMemset(dW_alpha, 0, dim_ * dim_ * sizeof(T));
        cudaMemset(dW_delta, 0, dim_ * dim_ * sizeof(T));

        for (int t = steps - 1; t >= 0; --t) {
            const T* x_t = x + t * BD;
            const T* log_h_prev = log_h + t * BD;
            const T* sign_h_prev = sign_h + t * BD;

            const T* log_v_t = log_v_cache + t * BD;
            const T* sign_v_t = sign_v_cache + t * BD;
            const T* alpha_t = alpha_cache + t * BD;
            const T* alpha_raw_t = alpha_raw_cache + t * BD;
            const T* log_h_unb_t = log_h_unbounded_cache + t * BD;
            const T* delta_t = delta_cache + t * BD;
            const T* weight_t = weight_rh_cache + t * BD;

            const T* d_h_t = d_h_linear + t * BD;
            T* dx_t = dx + t * BD;

            // Backward kernel
            LogPolyGatedBackwardKernel<T><<<num_blocks, block_size, 0, stream_>>>(
                batch_size_, dim_,
                log_h_prev, sign_h_prev,
                log_v_t, sign_v_t, alpha_t, alpha_raw_t, log_h_unb_t, delta_t, weight_t,
                log_r_h, sign_r_h,
                d_h_t,
                (t < steps - 1) ? d_log_h_recurrent : nullptr,
                d_log_h_prev, d_wx_x, d_alpha_raw, d_delta_raw,
                d_log_r_h_float, db_float);

            // dx through W_x path
            blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                dim_, batch_size_, dim_, &alpha_one, W_x, dim_, d_wx_x, dim_, &beta_zero, dx_t, dim_);

            // dx through W_alpha path
            blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                dim_, batch_size_, dim_, &alpha_one, W_alpha, dim_, d_alpha_raw, dim_, &alpha_one, dx_t, dim_);

            // dx through W_delta path
            blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                dim_, batch_size_, dim_, &alpha_one, W_delta, dim_, d_delta_raw, dim_, &alpha_one, dx_t, dim_);

            // Weight gradients
            blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
                dim_, dim_, batch_size_, &alpha_one, d_wx_x, dim_, x_t, dim_, &alpha_one, dW_x, dim_);

            blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
                dim_, dim_, batch_size_, &alpha_one, d_alpha_raw, dim_, x_t, dim_, &alpha_one, dW_alpha, dim_);

            blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
                dim_, dim_, batch_size_, &alpha_one, d_delta_raw, dim_, x_t, dim_, &alpha_one, dW_delta, dim_);

            // Copy for next iteration
            cudaMemcpy(d_log_h_recurrent, d_log_h_prev, BD * sizeof(T), cudaMemcpyDeviceToDevice);
        }

        // Copy float gradients to output
        if constexpr (std::is_same<T, float>::value) {
            cudaMemcpy(d_log_r_h, d_log_r_h_float, dim_ * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(db, db_float, dim_ * sizeof(float), cudaMemcpyDeviceToDevice);
        }

        cudaFree(d_log_h_recurrent);
        cudaFree(d_log_h_prev);
        cudaFree(d_wx_x);
        cudaFree(d_alpha_raw);
        cudaFree(d_delta_raw);
        cudaFree(d_log_r_h_float);
        cudaFree(db_float);
        cudaFree(db_delta_float);
        cudaFree(db_alpha_float);
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
}  // namespace haste
