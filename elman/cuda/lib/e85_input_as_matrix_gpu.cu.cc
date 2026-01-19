/**
 * E85 Input-as-Matrix CUDA Kernel
 *
 * A simpler approach: reshape input directly to matrix form.
 * - n_state typically 32 (so dim = 1024 = 32^2)
 * - Input x reshaped to [B, n_state, n_state] matrix A
 * - State M is [B, n_state, n_state]
 * - Update: M = M + scale * (A @ M)
 * - Normalize: M = M / norm(M)
 * - Output: flatten(M) with LayerNorm
 *
 * Key insight: ALL operations are n_state x n_state matrix ops that fit in shared memory!
 * No projections needed - input is directly used as a matrix.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include "hasty/elman_ladder.h"

#define E85_CHECKPOINT_INTERVAL 16

namespace elman {

// ============================================================================
// E85 Forward Kernel - BF16
// ============================================================================

template<int N_STATE>
__global__ void E85InputAsMatrixForwardKernel_BF16(
    int T,
    int B,
    float scale,
    const __nv_bfloat16* __restrict__ x,           // [T, B, N_STATE, N_STATE]
    const __nv_bfloat16* __restrict__ ln_gamma,    // [N_STATE * N_STATE]
    const __nv_bfloat16* __restrict__ ln_beta,     // [N_STATE * N_STATE]
    __nv_bfloat16* __restrict__ M,                 // [B, N_STATE, N_STATE]
    __nv_bfloat16* __restrict__ output,            // [T, B, N_STATE * N_STATE]
    __nv_bfloat16* __restrict__ M_checkpoints,     // [num_checkpoints, B, N_STATE, N_STATE]
    __nv_bfloat16* __restrict__ M_pre_norm_cache,  // [T, B, N_STATE, N_STATE] - pre-norm M for backward
    int checkpoint_interval
) {
    int b = blockIdx.x;
    if (b >= B) return;

    // Shared memory layout
    extern __shared__ float shared_mem[];
    float* M_shared = shared_mem;                            // [N_STATE * N_STATE]
    float* A_shared = M_shared + N_STATE * N_STATE;          // [N_STATE * N_STATE]
    float* AM_shared = A_shared + N_STATE * N_STATE;         // [N_STATE * N_STATE] - result of A @ M
    float* ln_gamma_shared = AM_shared + N_STATE * N_STATE;  // [N_STATE * N_STATE]
    float* ln_beta_shared = ln_gamma_shared + N_STATE * N_STATE;  // [N_STATE * N_STATE]

    int tid = threadIdx.x;
    const int n2 = N_STATE * N_STATE;

    // Load LayerNorm parameters
    for (int i = tid; i < n2; i += blockDim.x) {
        ln_gamma_shared[i] = __bfloat162float(ln_gamma[i]);
        ln_beta_shared[i] = __bfloat162float(ln_beta[i]);
    }
    __syncthreads();

    // Load initial state M
    for (int i = tid; i < n2; i += blockDim.x) {
        M_shared[i] = __bfloat162float(M[b * n2 + i]);
    }
    __syncthreads();

    // Save initial checkpoint (index 0)
    for (int i = tid; i < n2; i += blockDim.x) {
        M_checkpoints[b * n2 + i] = __float2bfloat16(M_shared[i]);
    }
    __syncthreads();

    // Process each timestep
    for (int t = 0; t < T; t++) {
        // Load A = reshape(x[t], [n_state, n_state])
        int x_offset = (t * B + b) * n2;
        for (int i = tid; i < n2; i += blockDim.x) {
            A_shared[i] = __bfloat162float(x[x_offset + i]);
        }
        __syncthreads();

        // Compute A @ M -> AM_shared
        // Each thread computes one element of the result
        // AM[i][j] = sum_k A[i][k] * M[k][j]
        for (int idx = tid; idx < n2; idx += blockDim.x) {
            int row = idx / N_STATE;
            int col = idx % N_STATE;
            float sum = 0.0f;
            #pragma unroll 8
            for (int k = 0; k < N_STATE; k++) {
                sum += A_shared[row * N_STATE + k] * M_shared[k * N_STATE + col];
            }
            AM_shared[idx] = sum;
        }
        __syncthreads();

        // Update: M = M + scale * (A @ M)
        for (int i = tid; i < n2; i += blockDim.x) {
            M_shared[i] = M_shared[i] + scale * AM_shared[i];
        }
        __syncthreads();

        // Cache M before normalization for backward pass
        for (int i = tid; i < n2; i += blockDim.x) {
            M_pre_norm_cache[(t * B + b) * n2 + i] = __float2bfloat16(M_shared[i]);
        }
        __syncthreads();

        // Normalize: M = M / ||M||_F
        __shared__ float norm_sum;
        if (tid == 0) {
            norm_sum = 0.0f;
            for (int i = 0; i < n2; i++) {
                norm_sum += M_shared[i] * M_shared[i];
            }
            norm_sum = sqrtf(norm_sum) + 1e-6f;
        }
        __syncthreads();

        for (int i = tid; i < n2; i += blockDim.x) {
            M_shared[i] /= norm_sum;
        }
        __syncthreads();

        // Save checkpoints if at boundary
        if ((t + 1) % checkpoint_interval == 0) {
            int cp_idx = (t + 1) / checkpoint_interval;
            for (int i = tid; i < n2; i += blockDim.x) {
                M_checkpoints[cp_idx * B * n2 + b * n2 + i] = __float2bfloat16(M_shared[i]);
            }
        }
        __syncthreads();

        // Output with LayerNorm: output = gamma * (M - mean) / std + beta
        // Compute mean and variance
        __shared__ float ln_mean, ln_var;
        if (tid == 0) {
            float sum = 0.0f;
            for (int i = 0; i < n2; i++) {
                sum += M_shared[i];
            }
            ln_mean = sum / n2;

            float var_sum = 0.0f;
            for (int i = 0; i < n2; i++) {
                float diff = M_shared[i] - ln_mean;
                var_sum += diff * diff;
            }
            ln_var = var_sum / n2;
        }
        __syncthreads();

        float inv_std = 1.0f / sqrtf(ln_var + 1e-5f);
        int out_offset = (t * B + b) * n2;
        for (int i = tid; i < n2; i += blockDim.x) {
            float normalized = (M_shared[i] - ln_mean) * inv_std;
            float out_val = ln_gamma_shared[i] * normalized + ln_beta_shared[i];
            output[out_offset + i] = __float2bfloat16(out_val);
        }
        __syncthreads();
    }

    // Write final state
    for (int i = tid; i < n2; i += blockDim.x) {
        M[b * n2 + i] = __float2bfloat16(M_shared[i]);
    }
}

// ============================================================================
// E85 Backward Kernel - BF16
// ============================================================================

template<int N_STATE>
__global__ void E85InputAsMatrixBackwardKernel_BF16(
    int T,
    int B,
    float scale,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ ln_gamma,
    const __nv_bfloat16* __restrict__ M_checkpoints,
    const __nv_bfloat16* __restrict__ M_pre_norm_cache,
    const __nv_bfloat16* __restrict__ d_output,
    __nv_bfloat16* __restrict__ d_x,
    float* __restrict__ d_ln_gamma_accum,
    float* __restrict__ d_ln_beta_accum,
    float* __restrict__ d_scale_accum,
    int checkpoint_interval
) {
    int b = blockIdx.x;
    if (b >= B) return;

    extern __shared__ float shared_mem[];
    float* M_shared = shared_mem;                              // [N_STATE * N_STATE]
    float* dM_shared = M_shared + N_STATE * N_STATE;           // [N_STATE * N_STATE]
    float* A_shared = dM_shared + N_STATE * N_STATE;           // [N_STATE * N_STATE]
    float* AM_shared = A_shared + N_STATE * N_STATE;           // [N_STATE * N_STATE]
    float* M_pre_norm = AM_shared + N_STATE * N_STATE;         // [N_STATE * N_STATE]
    float* d_AM = M_pre_norm + N_STATE * N_STATE;              // [N_STATE * N_STATE]
    float* d_A = d_AM + N_STATE * N_STATE;                     // [N_STATE * N_STATE]
    float* ln_gamma_shared = d_A + N_STATE * N_STATE;          // [N_STATE * N_STATE]
    float* d_ln_gamma_local = ln_gamma_shared + N_STATE * N_STATE;  // [N_STATE * N_STATE]
    float* d_ln_beta_local = d_ln_gamma_local + N_STATE * N_STATE;  // [N_STATE * N_STATE]

    int tid = threadIdx.x;
    const int n2 = N_STATE * N_STATE;

    // Load LayerNorm gamma
    for (int i = tid; i < n2; i += blockDim.x) {
        ln_gamma_shared[i] = __bfloat162float(ln_gamma[i]);
        d_ln_gamma_local[i] = 0.0f;
        d_ln_beta_local[i] = 0.0f;
    }
    __syncthreads();

    // Initialize dM accumulator
    for (int i = tid; i < n2; i += blockDim.x) {
        dM_shared[i] = 0.0f;
    }
    __syncthreads();

    // Local scale gradient accumulator
    __shared__ float d_scale_local;
    if (tid == 0) {
        d_scale_local = 0.0f;
    }
    __syncthreads();

    int num_segments = (T + checkpoint_interval - 1) / checkpoint_interval;

    for (int seg = num_segments - 1; seg >= 0; seg--) {
        int t_start = seg * checkpoint_interval;
        int t_end = min(t_start + checkpoint_interval, T);

        for (int t = t_end - 1; t >= t_start; t--) {
            // Reload checkpoint and recompute forward to step t
            for (int i = tid; i < n2; i += blockDim.x) {
                M_shared[i] = __bfloat162float(M_checkpoints[seg * B * n2 + b * n2 + i]);
            }
            __syncthreads();

            // Recompute forward from t_start to t
            for (int tt = t_start; tt <= t; tt++) {
                int x_offset = (tt * B + b) * n2;
                for (int i = tid; i < n2; i += blockDim.x) {
                    A_shared[i] = __bfloat162float(x[x_offset + i]);
                }
                __syncthreads();

                // A @ M
                for (int idx = tid; idx < n2; idx += blockDim.x) {
                    int row = idx / N_STATE;
                    int col = idx % N_STATE;
                    float sum = 0.0f;
                    #pragma unroll 8
                    for (int k = 0; k < N_STATE; k++) {
                        sum += A_shared[row * N_STATE + k] * M_shared[k * N_STATE + col];
                    }
                    AM_shared[idx] = sum;
                }
                __syncthreads();

                if (tt < t) {
                    // Update M for intermediate steps
                    for (int i = tid; i < n2; i += blockDim.x) {
                        M_shared[i] = M_shared[i] + scale * AM_shared[i];
                    }
                    __syncthreads();

                    // Normalize
                    __shared__ float norm_sum_tt;
                    if (tid == 0) {
                        norm_sum_tt = 0.0f;
                        for (int i = 0; i < n2; i++) {
                            norm_sum_tt += M_shared[i] * M_shared[i];
                        }
                        norm_sum_tt = sqrtf(norm_sum_tt) + 1e-6f;
                    }
                    __syncthreads();
                    for (int i = tid; i < n2; i += blockDim.x) {
                        M_shared[i] /= norm_sum_tt;
                    }
                    __syncthreads();
                }
            }

            // === BACKWARD PASS FOR STEP t ===

            // Load M_pre_norm for this timestep
            for (int i = tid; i < n2; i += blockDim.x) {
                M_pre_norm[i] = __bfloat162float(M_pre_norm_cache[(t * B + b) * n2 + i]);
            }
            __syncthreads();

            // Backward through LayerNorm
            // d_output -> d_M (post-norm)
            __shared__ float ln_mean, ln_var;
            if (tid == 0) {
                // Recompute LayerNorm stats from M_pre_norm after normalization
                // First, compute the normalized M
                float norm_sum = 0.0f;
                for (int i = 0; i < n2; i++) {
                    norm_sum += M_pre_norm[i] * M_pre_norm[i];
                }
                float norm_val = sqrtf(norm_sum) + 1e-6f;

                // Compute mean/var of normalized M
                float sum = 0.0f;
                for (int i = 0; i < n2; i++) {
                    sum += M_pre_norm[i] / norm_val;
                }
                ln_mean = sum / n2;

                float var_sum = 0.0f;
                for (int i = 0; i < n2; i++) {
                    float diff = M_pre_norm[i] / norm_val - ln_mean;
                    var_sum += diff * diff;
                }
                ln_var = var_sum / n2;
            }
            __syncthreads();

            float inv_std = 1.0f / sqrtf(ln_var + 1e-5f);

            // Compute norm of M_pre_norm for normalized M
            __shared__ float norm_val;
            if (tid == 0) {
                float ns = 0.0f;
                for (int i = 0; i < n2; i++) {
                    ns += M_pre_norm[i] * M_pre_norm[i];
                }
                norm_val = sqrtf(ns) + 1e-6f;
            }
            __syncthreads();

            // Accumulate d_ln_gamma, d_ln_beta and compute d_normalized
            int out_offset = (t * B + b) * n2;
            for (int i = tid; i < n2; i += blockDim.x) {
                float d_out = __bfloat162float(d_output[out_offset + i]);
                float M_normalized = M_pre_norm[i] / norm_val;
                float normalized = (M_normalized - ln_mean) * inv_std;

                // d_ln_gamma, d_ln_beta
                d_ln_gamma_local[i] += d_out * normalized;
                d_ln_beta_local[i] += d_out;

                // d_normalized
                float d_normalized = d_out * ln_gamma_shared[i];

                // Backprop through LayerNorm
                // d_M_normalized = d_normalized * inv_std - mean(d_normalized) * inv_std
                //                  - normalized * mean(d_normalized * normalized) * inv_std
                dM_shared[i] += d_normalized * inv_std;
            }
            __syncthreads();

            // LayerNorm backward correction terms
            __shared__ float d_mean_sum, d_var_sum;
            if (tid == 0) {
                d_mean_sum = 0.0f;
                d_var_sum = 0.0f;
                for (int i = 0; i < n2; i++) {
                    float d_out = __bfloat162float(d_output[out_offset + i]);
                    float d_normalized = d_out * ln_gamma_shared[i];
                    float M_normalized = M_pre_norm[i] / norm_val;
                    float normalized = (M_normalized - ln_mean) * inv_std;
                    d_mean_sum += d_normalized;
                    d_var_sum += d_normalized * normalized;
                }
            }
            __syncthreads();

            for (int i = tid; i < n2; i += blockDim.x) {
                float M_normalized = M_pre_norm[i] / norm_val;
                float normalized = (M_normalized - ln_mean) * inv_std;
                dM_shared[i] -= (d_mean_sum + normalized * d_var_sum) * inv_std / n2;
            }
            __syncthreads();

            // Backward through Frobenius normalization
            // M_out = M_pre_norm / ||M_pre_norm||
            // dM_pre_norm = dM_out / norm - M_pre_norm * (dM_out . M_out) / norm^2
            __shared__ float dot_prod;
            if (tid == 0) {
                dot_prod = 0.0f;
                for (int i = 0; i < n2; i++) {
                    dot_prod += dM_shared[i] * M_pre_norm[i] / norm_val;
                }
            }
            __syncthreads();

            for (int i = tid; i < n2; i += blockDim.x) {
                float dM_pre = dM_shared[i] / norm_val - M_pre_norm[i] * dot_prod / (norm_val * norm_val);
                dM_shared[i] = dM_pre;
            }
            __syncthreads();

            // Backward through update: M = M_prev + scale * (A @ M_prev)
            // dM_prev = dM + scale * A^T @ dM
            // d_scale += sum(dM * AM)
            // d_A = scale * dM @ M^T

            // Compute d_scale contribution
            if (tid == 0) {
                float ds = 0.0f;
                for (int i = 0; i < n2; i++) {
                    ds += dM_shared[i] * AM_shared[i];
                }
                d_scale_local += ds;
            }
            __syncthreads();

            // Compute d_A = scale * dM @ M^T
            for (int idx = tid; idx < n2; idx += blockDim.x) {
                int row = idx / N_STATE;
                int col = idx % N_STATE;
                float sum = 0.0f;
                #pragma unroll 8
                for (int k = 0; k < N_STATE; k++) {
                    sum += dM_shared[row * N_STATE + k] * M_shared[col * N_STATE + k];  // M^T
                }
                d_A[idx] = scale * sum;
            }
            __syncthreads();

            // Store d_x for this timestep
            int x_offset = (t * B + b) * n2;
            for (int i = tid; i < n2; i += blockDim.x) {
                d_x[x_offset + i] = __float2bfloat16(d_A[i]);
            }
            __syncthreads();

            // Compute dM_prev = dM + scale * A^T @ dM
            for (int idx = tid; idx < n2; idx += blockDim.x) {
                int row = idx / N_STATE;
                int col = idx % N_STATE;
                float sum = 0.0f;
                #pragma unroll 8
                for (int k = 0; k < N_STATE; k++) {
                    sum += A_shared[k * N_STATE + row] * dM_shared[k * N_STATE + col];  // A^T @ dM
                }
                d_AM[idx] = sum;
            }
            __syncthreads();

            for (int i = tid; i < n2; i += blockDim.x) {
                dM_shared[i] = dM_shared[i] + scale * d_AM[i];
            }
            __syncthreads();
        }
    }

    // Accumulate gradients
    for (int i = tid; i < n2; i += blockDim.x) {
        atomicAdd(&d_ln_gamma_accum[i], d_ln_gamma_local[i]);
        atomicAdd(&d_ln_beta_accum[i], d_ln_beta_local[i]);
    }
    if (tid == 0) {
        atomicAdd(d_scale_accum, d_scale_local);
    }
}

// ============================================================================
// E85 Forward Kernel - FP32
// ============================================================================

template<int N_STATE>
__global__ void E85InputAsMatrixForwardKernel_FP32(
    int T,
    int B,
    float scale,
    const float* __restrict__ x,
    const float* __restrict__ ln_gamma,
    const float* __restrict__ ln_beta,
    float* __restrict__ M,
    float* __restrict__ output,
    float* __restrict__ M_checkpoints,
    float* __restrict__ M_pre_norm_cache,
    int checkpoint_interval
) {
    int b = blockIdx.x;
    if (b >= B) return;

    extern __shared__ float shared_mem[];
    float* M_shared = shared_mem;
    float* A_shared = M_shared + N_STATE * N_STATE;
    float* AM_shared = A_shared + N_STATE * N_STATE;
    float* ln_gamma_shared = AM_shared + N_STATE * N_STATE;
    float* ln_beta_shared = ln_gamma_shared + N_STATE * N_STATE;

    int tid = threadIdx.x;
    const int n2 = N_STATE * N_STATE;

    for (int i = tid; i < n2; i += blockDim.x) {
        ln_gamma_shared[i] = ln_gamma[i];
        ln_beta_shared[i] = ln_beta[i];
    }
    __syncthreads();

    for (int i = tid; i < n2; i += blockDim.x) {
        M_shared[i] = M[b * n2 + i];
    }
    __syncthreads();

    for (int i = tid; i < n2; i += blockDim.x) {
        M_checkpoints[b * n2 + i] = M_shared[i];
    }
    __syncthreads();

    for (int t = 0; t < T; t++) {
        int x_offset = (t * B + b) * n2;
        for (int i = tid; i < n2; i += blockDim.x) {
            A_shared[i] = x[x_offset + i];
        }
        __syncthreads();

        for (int idx = tid; idx < n2; idx += blockDim.x) {
            int row = idx / N_STATE;
            int col = idx % N_STATE;
            float sum = 0.0f;
            for (int k = 0; k < N_STATE; k++) {
                sum += A_shared[row * N_STATE + k] * M_shared[k * N_STATE + col];
            }
            AM_shared[idx] = sum;
        }
        __syncthreads();

        for (int i = tid; i < n2; i += blockDim.x) {
            M_shared[i] = M_shared[i] + scale * AM_shared[i];
        }
        __syncthreads();

        for (int i = tid; i < n2; i += blockDim.x) {
            M_pre_norm_cache[(t * B + b) * n2 + i] = M_shared[i];
        }
        __syncthreads();

        __shared__ float norm_sum;
        if (tid == 0) {
            norm_sum = 0.0f;
            for (int i = 0; i < n2; i++) {
                norm_sum += M_shared[i] * M_shared[i];
            }
            norm_sum = sqrtf(norm_sum) + 1e-6f;
        }
        __syncthreads();

        for (int i = tid; i < n2; i += blockDim.x) {
            M_shared[i] /= norm_sum;
        }
        __syncthreads();

        if ((t + 1) % checkpoint_interval == 0) {
            int cp_idx = (t + 1) / checkpoint_interval;
            for (int i = tid; i < n2; i += blockDim.x) {
                M_checkpoints[cp_idx * B * n2 + b * n2 + i] = M_shared[i];
            }
        }
        __syncthreads();

        __shared__ float ln_mean, ln_var;
        if (tid == 0) {
            float sum = 0.0f;
            for (int i = 0; i < n2; i++) {
                sum += M_shared[i];
            }
            ln_mean = sum / n2;

            float var_sum = 0.0f;
            for (int i = 0; i < n2; i++) {
                float diff = M_shared[i] - ln_mean;
                var_sum += diff * diff;
            }
            ln_var = var_sum / n2;
        }
        __syncthreads();

        float inv_std = 1.0f / sqrtf(ln_var + 1e-5f);
        int out_offset = (t * B + b) * n2;
        for (int i = tid; i < n2; i += blockDim.x) {
            float normalized = (M_shared[i] - ln_mean) * inv_std;
            output[out_offset + i] = ln_gamma_shared[i] * normalized + ln_beta_shared[i];
        }
        __syncthreads();
    }

    for (int i = tid; i < n2; i += blockDim.x) {
        M[b * n2 + i] = M_shared[i];
    }
}

// ============================================================================
// E85 Backward Kernel - FP32
// ============================================================================

template<int N_STATE>
__global__ void E85InputAsMatrixBackwardKernel_FP32(
    int T,
    int B,
    float scale,
    const float* __restrict__ x,
    const float* __restrict__ ln_gamma,
    const float* __restrict__ M_checkpoints,
    const float* __restrict__ M_pre_norm_cache,
    const float* __restrict__ d_output,
    float* __restrict__ d_x,
    float* __restrict__ d_ln_gamma_accum,
    float* __restrict__ d_ln_beta_accum,
    float* __restrict__ d_scale_accum,
    int checkpoint_interval
) {
    int b = blockIdx.x;
    if (b >= B) return;

    extern __shared__ float shared_mem[];
    float* M_shared = shared_mem;
    float* dM_shared = M_shared + N_STATE * N_STATE;
    float* A_shared = dM_shared + N_STATE * N_STATE;
    float* AM_shared = A_shared + N_STATE * N_STATE;
    float* M_pre_norm = AM_shared + N_STATE * N_STATE;
    float* d_AM = M_pre_norm + N_STATE * N_STATE;
    float* d_A = d_AM + N_STATE * N_STATE;
    float* ln_gamma_shared = d_A + N_STATE * N_STATE;
    float* d_ln_gamma_local = ln_gamma_shared + N_STATE * N_STATE;
    float* d_ln_beta_local = d_ln_gamma_local + N_STATE * N_STATE;

    int tid = threadIdx.x;
    const int n2 = N_STATE * N_STATE;

    for (int i = tid; i < n2; i += blockDim.x) {
        ln_gamma_shared[i] = ln_gamma[i];
        d_ln_gamma_local[i] = 0.0f;
        d_ln_beta_local[i] = 0.0f;
    }
    __syncthreads();

    for (int i = tid; i < n2; i += blockDim.x) {
        dM_shared[i] = 0.0f;
    }
    __syncthreads();

    __shared__ float d_scale_local;
    if (tid == 0) {
        d_scale_local = 0.0f;
    }
    __syncthreads();

    int num_segments = (T + checkpoint_interval - 1) / checkpoint_interval;

    for (int seg = num_segments - 1; seg >= 0; seg--) {
        int t_start = seg * checkpoint_interval;
        int t_end = min(t_start + checkpoint_interval, T);

        for (int t = t_end - 1; t >= t_start; t--) {
            for (int i = tid; i < n2; i += blockDim.x) {
                M_shared[i] = M_checkpoints[seg * B * n2 + b * n2 + i];
            }
            __syncthreads();

            for (int tt = t_start; tt <= t; tt++) {
                int x_offset = (tt * B + b) * n2;
                for (int i = tid; i < n2; i += blockDim.x) {
                    A_shared[i] = x[x_offset + i];
                }
                __syncthreads();

                for (int idx = tid; idx < n2; idx += blockDim.x) {
                    int row = idx / N_STATE;
                    int col = idx % N_STATE;
                    float sum = 0.0f;
                    for (int k = 0; k < N_STATE; k++) {
                        sum += A_shared[row * N_STATE + k] * M_shared[k * N_STATE + col];
                    }
                    AM_shared[idx] = sum;
                }
                __syncthreads();

                if (tt < t) {
                    for (int i = tid; i < n2; i += blockDim.x) {
                        M_shared[i] = M_shared[i] + scale * AM_shared[i];
                    }
                    __syncthreads();

                    __shared__ float norm_sum_tt;
                    if (tid == 0) {
                        norm_sum_tt = 0.0f;
                        for (int i = 0; i < n2; i++) {
                            norm_sum_tt += M_shared[i] * M_shared[i];
                        }
                        norm_sum_tt = sqrtf(norm_sum_tt) + 1e-6f;
                    }
                    __syncthreads();
                    for (int i = tid; i < n2; i += blockDim.x) {
                        M_shared[i] /= norm_sum_tt;
                    }
                    __syncthreads();
                }
            }

            for (int i = tid; i < n2; i += blockDim.x) {
                M_pre_norm[i] = M_pre_norm_cache[(t * B + b) * n2 + i];
            }
            __syncthreads();

            __shared__ float ln_mean, ln_var, norm_val;
            if (tid == 0) {
                float ns = 0.0f;
                for (int i = 0; i < n2; i++) {
                    ns += M_pre_norm[i] * M_pre_norm[i];
                }
                norm_val = sqrtf(ns) + 1e-6f;

                float sum = 0.0f;
                for (int i = 0; i < n2; i++) {
                    sum += M_pre_norm[i] / norm_val;
                }
                ln_mean = sum / n2;

                float var_sum = 0.0f;
                for (int i = 0; i < n2; i++) {
                    float diff = M_pre_norm[i] / norm_val - ln_mean;
                    var_sum += diff * diff;
                }
                ln_var = var_sum / n2;
            }
            __syncthreads();

            float inv_std = 1.0f / sqrtf(ln_var + 1e-5f);

            int out_offset = (t * B + b) * n2;
            for (int i = tid; i < n2; i += blockDim.x) {
                float d_out = d_output[out_offset + i];
                float M_normalized = M_pre_norm[i] / norm_val;
                float normalized = (M_normalized - ln_mean) * inv_std;
                d_ln_gamma_local[i] += d_out * normalized;
                d_ln_beta_local[i] += d_out;
                dM_shared[i] += d_out * ln_gamma_shared[i] * inv_std;
            }
            __syncthreads();

            __shared__ float d_mean_sum, d_var_sum;
            if (tid == 0) {
                d_mean_sum = 0.0f;
                d_var_sum = 0.0f;
                for (int i = 0; i < n2; i++) {
                    float d_out = d_output[out_offset + i];
                    float d_normalized = d_out * ln_gamma_shared[i];
                    float M_normalized = M_pre_norm[i] / norm_val;
                    float normalized = (M_normalized - ln_mean) * inv_std;
                    d_mean_sum += d_normalized;
                    d_var_sum += d_normalized * normalized;
                }
            }
            __syncthreads();

            for (int i = tid; i < n2; i += blockDim.x) {
                float M_normalized = M_pre_norm[i] / norm_val;
                float normalized = (M_normalized - ln_mean) * inv_std;
                dM_shared[i] -= (d_mean_sum + normalized * d_var_sum) * inv_std / n2;
            }
            __syncthreads();

            __shared__ float dot_prod;
            if (tid == 0) {
                dot_prod = 0.0f;
                for (int i = 0; i < n2; i++) {
                    dot_prod += dM_shared[i] * M_pre_norm[i] / norm_val;
                }
            }
            __syncthreads();

            for (int i = tid; i < n2; i += blockDim.x) {
                float dM_pre = dM_shared[i] / norm_val - M_pre_norm[i] * dot_prod / (norm_val * norm_val);
                dM_shared[i] = dM_pre;
            }
            __syncthreads();

            if (tid == 0) {
                float ds = 0.0f;
                for (int i = 0; i < n2; i++) {
                    ds += dM_shared[i] * AM_shared[i];
                }
                d_scale_local += ds;
            }
            __syncthreads();

            for (int idx = tid; idx < n2; idx += blockDim.x) {
                int row = idx / N_STATE;
                int col = idx % N_STATE;
                float sum = 0.0f;
                for (int k = 0; k < N_STATE; k++) {
                    sum += dM_shared[row * N_STATE + k] * M_shared[col * N_STATE + k];
                }
                d_A[idx] = scale * sum;
            }
            __syncthreads();

            int x_offset = (t * B + b) * n2;
            for (int i = tid; i < n2; i += blockDim.x) {
                d_x[x_offset + i] = d_A[i];
            }
            __syncthreads();

            for (int idx = tid; idx < n2; idx += blockDim.x) {
                int row = idx / N_STATE;
                int col = idx % N_STATE;
                float sum = 0.0f;
                for (int k = 0; k < N_STATE; k++) {
                    sum += A_shared[k * N_STATE + row] * dM_shared[k * N_STATE + col];
                }
                d_AM[idx] = sum;
            }
            __syncthreads();

            for (int i = tid; i < n2; i += blockDim.x) {
                dM_shared[i] = dM_shared[i] + scale * d_AM[i];
            }
            __syncthreads();
        }
    }

    for (int i = tid; i < n2; i += blockDim.x) {
        atomicAdd(&d_ln_gamma_accum[i], d_ln_gamma_local[i]);
        atomicAdd(&d_ln_beta_accum[i], d_ln_beta_local[i]);
    }
    if (tid == 0) {
        atomicAdd(d_scale_accum, d_scale_local);
    }
}

// ============================================================================
// Dispatch functions
// ============================================================================

int calc_e85_forward_shared_mem(int n_state) {
    // M + A + AM + ln_gamma + ln_beta
    return (5 * n_state * n_state) * sizeof(float);
}

int calc_e85_backward_shared_mem(int n_state) {
    // M + dM + A + AM + M_pre_norm + d_AM + d_A + ln_gamma + d_ln_gamma + d_ln_beta
    return (10 * n_state * n_state) * sizeof(float);
}

void dispatch_e85_input_as_matrix_forward_bf16(
    int T, int B, int n_state, float scale,
    const __nv_bfloat16* x,
    const __nv_bfloat16* ln_gamma,
    const __nv_bfloat16* ln_beta,
    __nv_bfloat16* M,
    __nv_bfloat16* output,
    __nv_bfloat16* M_checkpoints,
    __nv_bfloat16* M_pre_norm_cache,
    int checkpoint_interval, cudaStream_t stream
) {
    int shared_size = calc_e85_forward_shared_mem(n_state);
    int block_size = 256;

    switch (n_state) {
        case 16:
            E85InputAsMatrixForwardKernel_BF16<16><<<B, block_size, shared_size, stream>>>(
                T, B, scale, x, ln_gamma, ln_beta, M, output,
                M_checkpoints, M_pre_norm_cache, checkpoint_interval);
            break;
        case 24:
            E85InputAsMatrixForwardKernel_BF16<24><<<B, block_size, shared_size, stream>>>(
                T, B, scale, x, ln_gamma, ln_beta, M, output,
                M_checkpoints, M_pre_norm_cache, checkpoint_interval);
            break;
        case 32:
            E85InputAsMatrixForwardKernel_BF16<32><<<B, block_size, shared_size, stream>>>(
                T, B, scale, x, ln_gamma, ln_beta, M, output,
                M_checkpoints, M_pre_norm_cache, checkpoint_interval);
            break;
        case 48:
            E85InputAsMatrixForwardKernel_BF16<48><<<B, block_size, shared_size, stream>>>(
                T, B, scale, x, ln_gamma, ln_beta, M, output,
                M_checkpoints, M_pre_norm_cache, checkpoint_interval);
            break;
        default:
            fprintf(stderr, "E85: Unsupported n_state=%d (use 16, 24, 32, or 48)\n", n_state);
    }
}

void dispatch_e85_input_as_matrix_backward_bf16(
    int T, int B, int n_state, float scale,
    const __nv_bfloat16* x,
    const __nv_bfloat16* ln_gamma,
    const __nv_bfloat16* M_checkpoints,
    const __nv_bfloat16* M_pre_norm_cache,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* d_x,
    float* d_ln_gamma_accum,
    float* d_ln_beta_accum,
    float* d_scale_accum,
    int checkpoint_interval, cudaStream_t stream
) {
    int shared_size = calc_e85_backward_shared_mem(n_state);
    int block_size = 256;

    switch (n_state) {
        case 16:
            E85InputAsMatrixBackwardKernel_BF16<16><<<B, block_size, shared_size, stream>>>(
                T, B, scale, x, ln_gamma, M_checkpoints, M_pre_norm_cache,
                d_output, d_x, d_ln_gamma_accum, d_ln_beta_accum, d_scale_accum,
                checkpoint_interval);
            break;
        case 24:
            E85InputAsMatrixBackwardKernel_BF16<24><<<B, block_size, shared_size, stream>>>(
                T, B, scale, x, ln_gamma, M_checkpoints, M_pre_norm_cache,
                d_output, d_x, d_ln_gamma_accum, d_ln_beta_accum, d_scale_accum,
                checkpoint_interval);
            break;
        case 32:
            E85InputAsMatrixBackwardKernel_BF16<32><<<B, block_size, shared_size, stream>>>(
                T, B, scale, x, ln_gamma, M_checkpoints, M_pre_norm_cache,
                d_output, d_x, d_ln_gamma_accum, d_ln_beta_accum, d_scale_accum,
                checkpoint_interval);
            break;
        case 48:
            E85InputAsMatrixBackwardKernel_BF16<48><<<B, block_size, shared_size, stream>>>(
                T, B, scale, x, ln_gamma, M_checkpoints, M_pre_norm_cache,
                d_output, d_x, d_ln_gamma_accum, d_ln_beta_accum, d_scale_accum,
                checkpoint_interval);
            break;
        default:
            fprintf(stderr, "E85: Unsupported n_state=%d (use 16, 24, 32, or 48)\n", n_state);
    }
}

void dispatch_e85_input_as_matrix_forward_fp32(
    int T, int B, int n_state, float scale,
    const float* x,
    const float* ln_gamma,
    const float* ln_beta,
    float* M,
    float* output,
    float* M_checkpoints,
    float* M_pre_norm_cache,
    int checkpoint_interval, cudaStream_t stream
) {
    int shared_size = calc_e85_forward_shared_mem(n_state);
    int block_size = 256;

    switch (n_state) {
        case 16:
            E85InputAsMatrixForwardKernel_FP32<16><<<B, block_size, shared_size, stream>>>(
                T, B, scale, x, ln_gamma, ln_beta, M, output,
                M_checkpoints, M_pre_norm_cache, checkpoint_interval);
            break;
        case 24:
            E85InputAsMatrixForwardKernel_FP32<24><<<B, block_size, shared_size, stream>>>(
                T, B, scale, x, ln_gamma, ln_beta, M, output,
                M_checkpoints, M_pre_norm_cache, checkpoint_interval);
            break;
        case 32:
            E85InputAsMatrixForwardKernel_FP32<32><<<B, block_size, shared_size, stream>>>(
                T, B, scale, x, ln_gamma, ln_beta, M, output,
                M_checkpoints, M_pre_norm_cache, checkpoint_interval);
            break;
        case 48:
            E85InputAsMatrixForwardKernel_FP32<48><<<B, block_size, shared_size, stream>>>(
                T, B, scale, x, ln_gamma, ln_beta, M, output,
                M_checkpoints, M_pre_norm_cache, checkpoint_interval);
            break;
        default:
            fprintf(stderr, "E85: Unsupported n_state=%d (use 16, 24, 32, or 48)\n", n_state);
    }
}

void dispatch_e85_input_as_matrix_backward_fp32(
    int T, int B, int n_state, float scale,
    const float* x,
    const float* ln_gamma,
    const float* M_checkpoints,
    const float* M_pre_norm_cache,
    const float* d_output,
    float* d_x,
    float* d_ln_gamma_accum,
    float* d_ln_beta_accum,
    float* d_scale_accum,
    int checkpoint_interval, cudaStream_t stream
) {
    int shared_size = calc_e85_backward_shared_mem(n_state);
    int block_size = 256;

    switch (n_state) {
        case 16:
            E85InputAsMatrixBackwardKernel_FP32<16><<<B, block_size, shared_size, stream>>>(
                T, B, scale, x, ln_gamma, M_checkpoints, M_pre_norm_cache,
                d_output, d_x, d_ln_gamma_accum, d_ln_beta_accum, d_scale_accum,
                checkpoint_interval);
            break;
        case 24:
            E85InputAsMatrixBackwardKernel_FP32<24><<<B, block_size, shared_size, stream>>>(
                T, B, scale, x, ln_gamma, M_checkpoints, M_pre_norm_cache,
                d_output, d_x, d_ln_gamma_accum, d_ln_beta_accum, d_scale_accum,
                checkpoint_interval);
            break;
        case 32:
            E85InputAsMatrixBackwardKernel_FP32<32><<<B, block_size, shared_size, stream>>>(
                T, B, scale, x, ln_gamma, M_checkpoints, M_pre_norm_cache,
                d_output, d_x, d_ln_gamma_accum, d_ln_beta_accum, d_scale_accum,
                checkpoint_interval);
            break;
        case 48:
            E85InputAsMatrixBackwardKernel_FP32<48><<<B, block_size, shared_size, stream>>>(
                T, B, scale, x, ln_gamma, M_checkpoints, M_pre_norm_cache,
                d_output, d_x, d_ln_gamma_accum, d_ln_beta_accum, d_scale_accum,
                checkpoint_interval);
            break;
        default:
            fprintf(stderr, "E85: Unsupported n_state=%d (use 16, 24, 32, or 48)\n", n_state);
    }
}

// ============================================================================
// Host-side wrapper classes
// ============================================================================

// Utility kernel for converting float accumulators to bfloat16
__global__ void ConvertFloatToBF16Kernel_E85(
    const float* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __float2bfloat16(src[i]);
    }
}

template<typename T>
E85InputAsMatrixForward<T>::E85InputAsMatrixForward(
    bool training,
    int batch_size,
    int n_state,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : training_(training),
      batch_size_(batch_size),
      n_state_(n_state),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void E85InputAsMatrixForward<T>::Run(
    int steps,
    float scale,
    const T* x,
    const T* ln_gamma,
    const T* ln_beta,
    T* M,
    T* output,
    T* M_checkpoints,
    T* M_pre_norm_cache
) {
    const int checkpoint_interval = E85_CHECKPOINT_INTERVAL;
    int n = n_state_;

    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        dispatch_e85_input_as_matrix_forward_bf16(
            steps, batch_size_, n, scale,
            x, ln_gamma, ln_beta, M, output,
            M_checkpoints, M_pre_norm_cache,
            checkpoint_interval, stream_);
    } else {
        dispatch_e85_input_as_matrix_forward_fp32(
            steps, batch_size_, n, scale,
            reinterpret_cast<const float*>(x),
            reinterpret_cast<const float*>(ln_gamma),
            reinterpret_cast<const float*>(ln_beta),
            reinterpret_cast<float*>(M),
            reinterpret_cast<float*>(output),
            reinterpret_cast<float*>(M_checkpoints),
            reinterpret_cast<float*>(M_pre_norm_cache),
            checkpoint_interval, stream_);
    }
}

template<typename T>
E85InputAsMatrixBackward<T>::E85InputAsMatrixBackward(
    int batch_size,
    int n_state,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      n_state_(n_state),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void E85InputAsMatrixBackward<T>::Run(
    int steps,
    float scale,
    const T* x,
    const T* ln_gamma,
    const T* M_checkpoints,
    const T* M_pre_norm_cache,
    const T* d_output,
    T* d_x,
    T* d_ln_gamma,
    T* d_ln_beta,
    T* d_scale,
    float* d_ln_gamma_accum,
    float* d_ln_beta_accum,
    float* d_scale_accum
) {
    const int checkpoint_interval = E85_CHECKPOINT_INTERVAL;
    int n = n_state_;
    int n2 = n * n;

    // Zero accumulators
    cudaMemsetAsync(d_ln_gamma_accum, 0, n2 * sizeof(float), stream_);
    cudaMemsetAsync(d_ln_beta_accum, 0, n2 * sizeof(float), stream_);
    cudaMemsetAsync(d_scale_accum, 0, sizeof(float), stream_);

    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        dispatch_e85_input_as_matrix_backward_bf16(
            steps, batch_size_, n, scale,
            x, ln_gamma, M_checkpoints, M_pre_norm_cache,
            d_output, d_x, d_ln_gamma_accum, d_ln_beta_accum, d_scale_accum,
            checkpoint_interval, stream_);

        // Convert accumulated gradients to bfloat16
        int threads = 256;
        int blocks = (n2 + threads - 1) / threads;
        ConvertFloatToBF16Kernel_E85<<<blocks, threads, 0, stream_>>>(
            d_ln_gamma_accum, d_ln_gamma, n2);
        ConvertFloatToBF16Kernel_E85<<<blocks, threads, 0, stream_>>>(
            d_ln_beta_accum, d_ln_beta, n2);
        ConvertFloatToBF16Kernel_E85<<<1, 1, 0, stream_>>>(
            d_scale_accum, d_scale, 1);
    } else {
        dispatch_e85_input_as_matrix_backward_fp32(
            steps, batch_size_, n, scale,
            reinterpret_cast<const float*>(x),
            reinterpret_cast<const float*>(ln_gamma),
            reinterpret_cast<const float*>(M_checkpoints),
            reinterpret_cast<const float*>(M_pre_norm_cache),
            reinterpret_cast<const float*>(d_output),
            reinterpret_cast<float*>(d_x),
            d_ln_gamma_accum,
            d_ln_beta_accum,
            d_scale_accum,
            checkpoint_interval, stream_);

        // Copy accumulators to output
        cudaMemcpyAsync(d_ln_gamma, d_ln_gamma_accum, n2 * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream_);
        cudaMemcpyAsync(d_ln_beta, d_ln_beta_accum, n2 * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream_);
        cudaMemcpyAsync(d_scale, d_scale_accum, sizeof(float),
                        cudaMemcpyDeviceToDevice, stream_);
    }
}

// Explicit template instantiations
template struct E85InputAsMatrixForward<__nv_bfloat16>;
template struct E85InputAsMatrixForward<float>;
template struct E85InputAsMatrixBackward<__nv_bfloat16>;
template struct E85InputAsMatrixBackward<float>;

}  // namespace elman
