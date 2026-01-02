// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// Shared Log-Space RMSNorm Kernels
//
// These kernels compute RMSNorm directly in log-space using logsumexp
// for bounded gradients. The softmax weights in the backward pass are
// naturally bounded to [0, 1], preventing gradient explosion.
//
// Formula:
//   log(rms) = (logsumexp(2 * log_h) - log(dim)) / 2
//   h_normalized = sign * exp(log_h - log_rms + log_gamma)
//
// Used by: log_0, log_1, log_2, log_3, log_4, log_5, log_6

#ifndef HASTY_LOGSPACE_RMSNORM_CUH_
#define HASTY_LOGSPACE_RMSNORM_CUH_

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace hasty {
namespace logspace {

// =============================================================================
// Forward Kernel: Log-Space RMSNorm with Linear Output
// =============================================================================
//
// Inputs:
//   log_h: [B, dim] log magnitudes
//   sign_h: [B, dim] signs
//   log_gamma: [dim] learnable scale in log-space
//
// Outputs:
//   h_linear: [B, dim] normalized values in linear space
//   log_rms_cache: [B] cached log(rms) for backward (optional, can be nullptr)
//
// Each block handles one batch element.
// Shared memory: blockDim.x * sizeof(float)

template<typename T, int BLOCK_DIM = 256>
__global__ void LogSpaceRMSNormForwardKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ log_h,
    const T* __restrict__ sign_h,
    const T* __restrict__ log_gamma,
    T* __restrict__ h_linear,
    T* __restrict__ log_rms_cache) {

    const int b = blockIdx.x;
    if (b >= batch_size) return;

    extern __shared__ float sdata[];

    // Step 1: Compute max(2 * log_h) for numerical stability
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

    // Step 2: Compute sum(exp(2 * log_h - max)) for logsumexp
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

    // logsumexp(2 * log_h) = max + log(sum_exp)
    float log_sum_h2 = max_val + logf(sum_exp);

    // log(rms) = (logsumexp(2 * log_h) - log(dim)) / 2
    float log_rms = (log_sum_h2 - logf(static_cast<float>(dim))) * 0.5f;

    // Cache log_rms for backward
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

        // Convert to linear with clamping for numerical safety
        float clamped = fminf(fmaxf(log_normed, -40.0f), 20.0f);
        h_linear[idx] = static_cast<T>(sign_val * expf(clamped));
    }
}

// =============================================================================
// Backward Kernel: Log-Space RMSNorm Gradient
// =============================================================================
//
// Inputs:
//   log_h: [B, dim] log magnitudes (from forward)
//   sign_h: [B, dim] signs (from forward)
//   h_linear: [B, dim] normalized output (from forward)
//   d_h_linear: [B, dim] incoming gradient
//
// Outputs:
//   d_log_h: [B, dim] gradient w.r.t. log_h
//   d_log_gamma: [dim] gradient w.r.t. log_gamma (atomic add across batch)
//
// Each block handles one batch element.
// Shared memory: blockDim.x * sizeof(float)

template<typename T, int BLOCK_DIM = 256>
__global__ void LogSpaceRMSNormBackwardKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ log_h,
    const T* __restrict__ sign_h,
    const T* __restrict__ h_linear,
    const T* __restrict__ d_h_linear,
    T* __restrict__ d_log_h,
    float* __restrict__ d_log_gamma) {

    const int b = blockIdx.x;
    if (b >= batch_size) return;

    extern __shared__ float sdata[];

    // Step 1: Recompute logsumexp for softmax weights
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

    // Step 2: Compute sum of d_log_normed for d_log_rms
    // d_log_normed = d_h_linear * h_linear (chain rule for exp)
    float thread_d_log_rms = 0.0f;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        int idx = b * dim + d;
        float h_lin = static_cast<float>(h_linear[idx]);
        float d_h_lin = static_cast<float>(d_h_linear[idx]);
        float d_log_normed = d_h_lin * h_lin;
        thread_d_log_rms -= d_log_normed;  // Negative because log_rms is subtracted

        // Accumulate gradient for log_gamma
        atomicAdd(&d_log_gamma[d], d_log_normed);
    }

    sdata[threadIdx.x] = thread_d_log_rms;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    float d_log_rms = sdata[0];

    // Step 3: Compute d_log_h
    // d_log_h = d_log_normed + d_log_rms * softmax_weight
    // where softmax_weight = exp(2*log_h - logsumexp) is bounded to [0, 1]
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        int idx = b * dim + d;
        float log_h2 = 2.0f * static_cast<float>(log_h[idx]);
        float h_lin = static_cast<float>(h_linear[idx]);
        float d_h_lin = static_cast<float>(d_h_linear[idx]);

        // Softmax weight: bounded to [0, 1] - this is what prevents gradient explosion
        float softmax_w = expf(log_h2 - max_val) / sum_exp;

        float d_log_normed = d_h_lin * h_lin;
        float d_from_rms = d_log_rms * softmax_w;

        d_log_h[idx] = static_cast<T>(d_log_normed + d_from_rms);
    }
}

// =============================================================================
// Helper: Launch configuration
// =============================================================================

inline int get_rmsnorm_block_size(int dim) {
    if (dim <= 256) return 256;
    if (dim <= 512) return 256;
    if (dim <= 1024) return 256;
    return 256;  // Default
}

inline size_t get_rmsnorm_smem_size(int block_size) {
    return block_size * sizeof(float);
}

}  // namespace logspace
}  // namespace hasty

#endif  // HASTY_LOGSPACE_RMSNORM_CUH_
