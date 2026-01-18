/**
 * E74 Fixed Decay Delta Rule CUDA Kernel
 *
 * The simplest architecture in the autopoietic ladder.
 *
 * Mathematical Definition:
 *   k, v, q = W_kvq @ x  (FUSED projection)
 *   k_norm = k / ||k||
 *   delta = v - S @ k_norm
 *   S' = alpha * S + outer(delta, k_norm)
 *   output = (S' @ q) * silu(S' @ q)
 *
 * Where alpha in (0,1) is a FIXED scalar decay.
 *
 * Key simplifications vs E79:
 * - Single state matrix S (no M modulation matrix)
 * - Fixed scalar alpha for decay (no learned gating)
 * - No input-dependent gating
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include "hasty/elman_ladder.h"

#define E74_FD_CHECKPOINT_INTERVAL 16

namespace elman {

// ============================================================================
// E74 Fixed Decay Forward Kernel - BF16
// ============================================================================

template<int N_STATE>
__global__ void E74FixedDecayForwardKernel_BF16(
    int T,
    int B,
    float alpha,                                    // Fixed decay factor
    const __nv_bfloat16* __restrict__ kvq_all,      // [3*N_STATE, T*B] column-major (k, v, q fused)
    __nv_bfloat16* __restrict__ S,                  // [B, N_STATE, N_STATE]
    __nv_bfloat16* __restrict__ output,             // [T, B, N_STATE]
    __nv_bfloat16* __restrict__ S_checkpoints,      // [num_checkpoints, B, N_STATE, N_STATE]
    __nv_bfloat16* __restrict__ Sq_cache,           // [T, B, N_STATE]
    int checkpoint_interval
) {
    int b = blockIdx.x;
    if (b >= B) return;

    // Shared memory layout
    extern __shared__ float shared_mem[];
    float* S_shared = shared_mem;                         // [N_STATE * N_STATE]
    float* k_shared = S_shared + N_STATE * N_STATE;       // [N_STATE]
    float* v_shared = k_shared + N_STATE;                 // [N_STATE]
    float* q_shared = v_shared + N_STATE;                 // [N_STATE]
    float* retrieved = q_shared + N_STATE;                // [N_STATE]

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;
    const int STRIDE = 3 * N_STATE;

    // Load initial state
    for (int i = tid; i < n2; i += blockDim.x) {
        S_shared[i] = __bfloat162float(S[b * n2 + i]);
    }
    __syncthreads();

    // Save initial checkpoint (index 0)
    for (int i = tid; i < n2; i += blockDim.x) {
        S_checkpoints[b * n2 + i] = __float2bfloat16(S_shared[i]);
    }
    __syncthreads();

    // Process each timestep
    for (int t = 0; t < T; t++) {
        int col_idx = (t * B + b) * STRIDE;

        // Load k, v, q for this timestep
        if (tid < N_STATE) {
            k_shared[tid] = __bfloat162float(kvq_all[col_idx + tid]);
            v_shared[tid] = __bfloat162float(kvq_all[col_idx + N_STATE + tid]);
            q_shared[tid] = __bfloat162float(kvq_all[col_idx + 2 * N_STATE + tid]);
        }
        __syncthreads();

        // Normalize k
        __shared__ float k_norm_val;
        if (tid == 0) {
            float k_sum = 0.0f;
            for (int i = 0; i < N_STATE; i++) {
                k_sum += k_shared[i] * k_shared[i];
            }
            k_norm_val = sqrtf(k_sum) + 1e-6f;
        }
        __syncthreads();
        if (tid < N_STATE) {
            k_shared[tid] /= k_norm_val;
        }
        __syncthreads();

        // Compute retrieved = S @ k_norm
        if (tid < N_STATE) {
            float sum = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < N_STATE; j++) {
                sum += S_shared[tid * N_STATE + j] * k_shared[j];
            }
            retrieved[tid] = sum;
        }
        __syncthreads();

        // Update state: S' = alpha * S + outer(delta, k_norm)
        // where delta = v - retrieved
        for (int i = tid; i < n2; i += blockDim.x) {
            int row = i / N_STATE;
            int col = i % N_STATE;
            float delta_row = v_shared[row] - retrieved[row];
            S_shared[i] = alpha * S_shared[i] + delta_row * k_shared[col];
        }
        __syncthreads();

        // Save checkpoint if at boundary
        if ((t + 1) % checkpoint_interval == 0) {
            int cp_idx = (t + 1) / checkpoint_interval;
            for (int i = tid; i < n2; i += blockDim.x) {
                S_checkpoints[cp_idx * B * n2 + b * n2 + i] = __float2bfloat16(S_shared[i]);
            }
        }
        __syncthreads();

        // Output: Sq = S @ q, then Sq * silu(Sq)
        if (tid < N_STATE) {
            float Sq = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < N_STATE; j++) {
                Sq += S_shared[tid * N_STATE + j] * q_shared[j];
            }
            Sq_cache[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(Sq);
            // silu(Sq) = Sq * sigmoid(Sq)
            float sig = 1.0f / (1.0f + expf(-Sq));
            output[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(Sq * Sq * sig);
        }
        __syncthreads();
    }

    // Write final state
    for (int i = tid; i < n2; i += blockDim.x) {
        S[b * n2 + i] = __float2bfloat16(S_shared[i]);
    }
}

// ============================================================================
// E74 Fixed Decay Backward Kernel - BF16
// ============================================================================

template<int N_STATE>
__global__ void E74FixedDecayBackwardKernel_BF16(
    int T,
    int B,
    float alpha,
    const __nv_bfloat16* __restrict__ kvq_all,
    const __nv_bfloat16* __restrict__ S_checkpoints,
    const __nv_bfloat16* __restrict__ Sq_cache,
    const __nv_bfloat16* __restrict__ d_output,
    __nv_bfloat16* __restrict__ d_kvq_all,
    float* __restrict__ d_alpha_accum,              // For learnable alpha
    int checkpoint_interval
) {
    int b = blockIdx.x;
    if (b >= B) return;

    extern __shared__ float shared_mem[];
    // Layout for backward
    float* S = shared_mem;                                // [N_STATE * N_STATE]
    float* dS = S + N_STATE * N_STATE;                    // [N_STATE * N_STATE]
    float* k_raw = dS + N_STATE * N_STATE;                // [N_STATE]
    float* v_raw = k_raw + N_STATE;                       // [N_STATE]
    float* q_raw = v_raw + N_STATE;                       // [N_STATE]
    float* k_norm = q_raw + N_STATE;                      // [N_STATE]
    float* retrieved = k_norm + N_STATE;                  // [N_STATE]
    float* delta = retrieved + N_STATE;                   // [N_STATE]
    float* d_k_raw = delta + N_STATE;                     // [N_STATE]
    float* d_v_raw = d_k_raw + N_STATE;                   // [N_STATE]
    float* d_q_raw = d_v_raw + N_STATE;                   // [N_STATE]
    float* d_Sq = d_q_raw + N_STATE;                      // [N_STATE]
    float* d_delta = d_Sq + N_STATE;                      // [N_STATE]
    float* d_k_norm = d_delta + N_STATE;                  // [N_STATE]
    float* d_alpha_local = d_k_norm + N_STATE;            // [1]

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;
    const int STRIDE = 3 * N_STATE;

    // Initialize gradient accumulators
    for (int i = tid; i < n2; i += blockDim.x) {
        dS[i] = 0.0f;
    }
    if (tid == 0) {
        d_alpha_local[0] = 0.0f;
    }
    __syncthreads();

    int num_segments = (T + checkpoint_interval - 1) / checkpoint_interval;

    for (int seg = num_segments - 1; seg >= 0; seg--) {
        int t_start = seg * checkpoint_interval;
        int t_end = min(t_start + checkpoint_interval, T);

        for (int t = t_end - 1; t >= t_start; t--) {
            // Reload checkpoint
            for (int i = tid; i < n2; i += blockDim.x) {
                S[i] = __bfloat162float(S_checkpoints[seg * B * n2 + b * n2 + i]);
            }
            __syncthreads();

            // Recompute forward to step t
            __shared__ float k_norm_val_t;
            for (int tt = t_start; tt <= t; tt++) {
                int col_idx = (tt * B + b) * STRIDE;
                if (tid < N_STATE) {
                    k_raw[tid] = __bfloat162float(kvq_all[col_idx + tid]);
                    v_raw[tid] = __bfloat162float(kvq_all[col_idx + N_STATE + tid]);
                    q_raw[tid] = __bfloat162float(kvq_all[col_idx + 2 * N_STATE + tid]);
                }
                __syncthreads();

                // Normalize k
                if (tid == 0) {
                    float k_sum = 0.0f;
                    for (int i = 0; i < N_STATE; i++) {
                        k_sum += k_raw[i] * k_raw[i];
                    }
                    k_norm_val_t = sqrtf(k_sum) + 1e-6f;
                }
                __syncthreads();
                if (tid < N_STATE) {
                    k_norm[tid] = k_raw[tid] / k_norm_val_t;
                }
                __syncthreads();

                // Compute retrieved and delta BEFORE S update
                if (tid < N_STATE) {
                    float sum = 0.0f;
                    for (int j = 0; j < N_STATE; j++) {
                        sum += S[tid * N_STATE + j] * k_norm[j];
                    }
                    retrieved[tid] = sum;
                    delta[tid] = v_raw[tid] - sum;
                }
                __syncthreads();

                // Update S if not at target step
                if (tt < t) {
                    for (int i = tid; i < n2; i += blockDim.x) {
                        int row = i / N_STATE;
                        int col = i % N_STATE;
                        S[i] = alpha * S[i] + delta[row] * k_norm[col];
                    }
                    __syncthreads();
                }
            }

            // Now S holds state at t-1; delta is for step t

            // Backward through output: Sq * silu(Sq) = Sq * Sq * sigmoid(Sq)
            if (tid < N_STATE) {
                float d_out = __bfloat162float(d_output[t * B * N_STATE + b * N_STATE + tid]);
                float Sq = __bfloat162float(Sq_cache[t * B * N_STATE + b * N_STATE + tid]);
                // d/dSq of (Sq * silu(Sq)) = silu(Sq) + Sq * silu'(Sq)
                //                         = Sq*sigmoid(Sq) + Sq*sigmoid(Sq)*(1-sigmoid(Sq))
                float sig = 1.0f / (1.0f + expf(-Sq));
                float silu_Sq = Sq * sig;
                float d_Sq_val = d_out * (silu_Sq + Sq * sig * (1.0f - sig));
                d_Sq[tid] = d_Sq_val;
            }
            __syncthreads();

            // dS += outer(d_Sq, q)
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                dS[i] += d_Sq[row] * q_raw[col];
            }
            __syncthreads();

            // d_q = S_t^T @ d_Sq (where S_t = alpha*S + outer(delta, k_norm))
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    float S_t_ij = alpha * S[i * N_STATE + tid] + delta[i] * k_norm[tid];
                    sum += S_t_ij * d_Sq[i];
                }
                d_q_raw[tid] = sum;
            }
            __syncthreads();

            // Backward through S update: S_new = alpha * S + outer(delta, k_norm)
            // dS_old = alpha * dS_new
            // d_delta = dS_new @ k_norm
            // d_k_norm = dS_new.T @ delta
            // d_alpha += sum(dS_new * S)
            float d_alpha_step = 0.0f;
            if (tid < N_STATE) {
                float d_delta_local = 0.0f;
                float d_kn_local = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    float dS_ij = dS[tid * N_STATE + j];
                    d_delta_local += dS_ij * k_norm[j];
                    d_kn_local += dS[j * N_STATE + tid] * delta[j];
                    d_alpha_step += dS_ij * S[tid * N_STATE + j];  // Only count once per thread row
                }
                d_delta[tid] = d_delta_local;
                d_k_norm[tid] = d_kn_local;
            }
            __syncthreads();

            // Accumulate d_alpha (reduce across threads)
            if (tid == 0) {
                for (int i = 0; i < N_STATE; i++) {
                    for (int j = 0; j < N_STATE; j++) {
                        d_alpha_local[0] += dS[i * N_STATE + j] * S[i * N_STATE + j];
                    }
                }
            }
            // Note: The reduction above is done in thread 0 for simplicity
            // In practice, d_alpha is usually not learned, so this is fine

            // d_v = d_delta
            // d_retrieved = -d_delta
            if (tid < N_STATE) {
                d_v_raw[tid] = d_delta[tid];
            }
            __syncthreads();

            // d_k_norm contribution from retrieved: retrieved = S @ k_norm
            // d_k_norm += S^T @ (-d_delta)
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    sum += S[i * N_STATE + tid] * (-d_delta[i]);
                }
                d_k_norm[tid] += sum;
            }
            __syncthreads();

            // Convert d_k_norm to d_k_raw (through normalization)
            {
                __shared__ float k_dot_dk;
                if (tid == 0) {
                    k_dot_dk = 0.0f;
                    for (int i = 0; i < N_STATE; i++) {
                        k_dot_dk += k_raw[i] * d_k_norm[i];
                    }
                }
                __syncthreads();
                if (tid < N_STATE) {
                    float norm = k_norm_val_t;
                    float norm3 = norm * norm * norm;
                    d_k_raw[tid] = d_k_norm[tid] / norm - k_raw[tid] * k_dot_dk / norm3;
                }
                __syncthreads();
            }

            // Write gradients
            int col_idx_t = (t * B + b) * STRIDE;
            if (tid < N_STATE) {
                d_kvq_all[col_idx_t + tid] = __float2bfloat16(d_k_raw[tid]);
                d_kvq_all[col_idx_t + N_STATE + tid] = __float2bfloat16(d_v_raw[tid]);
                d_kvq_all[col_idx_t + 2 * N_STATE + tid] = __float2bfloat16(d_q_raw[tid]);
            }
            __syncthreads();

            // Update dS for next iteration: dS = alpha * dS + (-d_delta) @ k_norm^T
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                dS[i] = alpha * dS[i] + (-d_delta[row]) * k_norm[col];
            }
            __syncthreads();
        }
    }

    // Accumulate d_alpha (atomic add across batches)
    if (tid == 0) {
        atomicAdd(d_alpha_accum, d_alpha_local[0]);
    }
}

// ============================================================================
// E74 Fixed Decay Forward Kernel - FP32
// ============================================================================

template<int N_STATE>
__global__ void E74FixedDecayForwardKernel_FP32(
    int T,
    int B,
    float alpha,
    const float* __restrict__ kvq_all,
    float* __restrict__ S,
    float* __restrict__ output,
    float* __restrict__ S_checkpoints,
    float* __restrict__ Sq_cache,
    int checkpoint_interval
) {
    int b = blockIdx.x;
    if (b >= B) return;

    extern __shared__ float shared_mem[];
    float* S_shared = shared_mem;
    float* k_shared = S_shared + N_STATE * N_STATE;
    float* v_shared = k_shared + N_STATE;
    float* q_shared = v_shared + N_STATE;
    float* retrieved = q_shared + N_STATE;

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;
    const int STRIDE = 3 * N_STATE;

    // Load initial state
    for (int i = tid; i < n2; i += blockDim.x) {
        S_shared[i] = S[b * n2 + i];
    }
    __syncthreads();

    // Save initial checkpoint
    for (int i = tid; i < n2; i += blockDim.x) {
        S_checkpoints[b * n2 + i] = S_shared[i];
    }
    __syncthreads();

    for (int t = 0; t < T; t++) {
        int col_idx = (t * B + b) * STRIDE;

        if (tid < N_STATE) {
            k_shared[tid] = kvq_all[col_idx + tid];
            v_shared[tid] = kvq_all[col_idx + N_STATE + tid];
            q_shared[tid] = kvq_all[col_idx + 2 * N_STATE + tid];
        }
        __syncthreads();

        // Normalize k
        __shared__ float k_norm_val;
        if (tid == 0) {
            float k_sum = 0.0f;
            for (int i = 0; i < N_STATE; i++) {
                k_sum += k_shared[i] * k_shared[i];
            }
            k_norm_val = sqrtf(k_sum) + 1e-6f;
        }
        __syncthreads();
        if (tid < N_STATE) {
            k_shared[tid] /= k_norm_val;
        }
        __syncthreads();

        // Compute retrieved
        if (tid < N_STATE) {
            float sum = 0.0f;
            for (int j = 0; j < N_STATE; j++) {
                sum += S_shared[tid * N_STATE + j] * k_shared[j];
            }
            retrieved[tid] = sum;
        }
        __syncthreads();

        // Update state
        for (int i = tid; i < n2; i += blockDim.x) {
            int row = i / N_STATE;
            int col = i % N_STATE;
            float delta_row = v_shared[row] - retrieved[row];
            S_shared[i] = alpha * S_shared[i] + delta_row * k_shared[col];
        }
        __syncthreads();

        // Save checkpoint
        if ((t + 1) % checkpoint_interval == 0) {
            int cp_idx = (t + 1) / checkpoint_interval;
            for (int i = tid; i < n2; i += blockDim.x) {
                S_checkpoints[cp_idx * B * n2 + b * n2 + i] = S_shared[i];
            }
        }
        __syncthreads();

        // Output
        if (tid < N_STATE) {
            float Sq = 0.0f;
            for (int j = 0; j < N_STATE; j++) {
                Sq += S_shared[tid * N_STATE + j] * q_shared[j];
            }
            Sq_cache[t * B * N_STATE + b * N_STATE + tid] = Sq;
            float sig = 1.0f / (1.0f + expf(-Sq));
            output[t * B * N_STATE + b * N_STATE + tid] = Sq * Sq * sig;
        }
        __syncthreads();
    }

    // Write final state
    for (int i = tid; i < n2; i += blockDim.x) {
        S[b * n2 + i] = S_shared[i];
    }
}

// ============================================================================
// E74 Fixed Decay Backward Kernel - FP32
// ============================================================================

template<int N_STATE>
__global__ void E74FixedDecayBackwardKernel_FP32(
    int T,
    int B,
    float alpha,
    const float* __restrict__ kvq_all,
    const float* __restrict__ S_checkpoints,
    const float* __restrict__ Sq_cache,
    const float* __restrict__ d_output,
    float* __restrict__ d_kvq_all,
    float* __restrict__ d_alpha_accum,
    int checkpoint_interval
) {
    int b = blockIdx.x;
    if (b >= B) return;

    extern __shared__ float shared_mem[];
    float* S = shared_mem;
    float* dS = S + N_STATE * N_STATE;
    float* k_raw = dS + N_STATE * N_STATE;
    float* v_raw = k_raw + N_STATE;
    float* q_raw = v_raw + N_STATE;
    float* k_norm = q_raw + N_STATE;
    float* retrieved = k_norm + N_STATE;
    float* delta = retrieved + N_STATE;
    float* d_k_raw = delta + N_STATE;
    float* d_v_raw = d_k_raw + N_STATE;
    float* d_q_raw = d_v_raw + N_STATE;
    float* d_Sq = d_q_raw + N_STATE;
    float* d_delta = d_Sq + N_STATE;
    float* d_k_norm = d_delta + N_STATE;
    float* d_alpha_local = d_k_norm + N_STATE;

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;
    const int STRIDE = 3 * N_STATE;

    for (int i = tid; i < n2; i += blockDim.x) {
        dS[i] = 0.0f;
    }
    if (tid == 0) {
        d_alpha_local[0] = 0.0f;
    }
    __syncthreads();

    int num_segments = (T + checkpoint_interval - 1) / checkpoint_interval;

    for (int seg = num_segments - 1; seg >= 0; seg--) {
        int t_start = seg * checkpoint_interval;
        int t_end = min(t_start + checkpoint_interval, T);

        for (int t = t_end - 1; t >= t_start; t--) {
            // Reload checkpoint
            for (int i = tid; i < n2; i += blockDim.x) {
                S[i] = S_checkpoints[seg * B * n2 + b * n2 + i];
            }
            __syncthreads();

            // Recompute forward to step t
            __shared__ float k_norm_val_t;
            for (int tt = t_start; tt <= t; tt++) {
                int col_idx = (tt * B + b) * STRIDE;
                if (tid < N_STATE) {
                    k_raw[tid] = kvq_all[col_idx + tid];
                    v_raw[tid] = kvq_all[col_idx + N_STATE + tid];
                    q_raw[tid] = kvq_all[col_idx + 2 * N_STATE + tid];
                }
                __syncthreads();

                if (tid == 0) {
                    float k_sum = 0.0f;
                    for (int i = 0; i < N_STATE; i++) {
                        k_sum += k_raw[i] * k_raw[i];
                    }
                    k_norm_val_t = sqrtf(k_sum) + 1e-6f;
                }
                __syncthreads();
                if (tid < N_STATE) {
                    k_norm[tid] = k_raw[tid] / k_norm_val_t;
                }
                __syncthreads();

                if (tid < N_STATE) {
                    float sum = 0.0f;
                    for (int j = 0; j < N_STATE; j++) {
                        sum += S[tid * N_STATE + j] * k_norm[j];
                    }
                    retrieved[tid] = sum;
                    delta[tid] = v_raw[tid] - sum;
                }
                __syncthreads();

                if (tt < t) {
                    for (int i = tid; i < n2; i += blockDim.x) {
                        int row = i / N_STATE;
                        int col = i % N_STATE;
                        S[i] = alpha * S[i] + delta[row] * k_norm[col];
                    }
                    __syncthreads();
                }
            }

            // Backward through output
            if (tid < N_STATE) {
                float d_out = d_output[t * B * N_STATE + b * N_STATE + tid];
                float Sq = Sq_cache[t * B * N_STATE + b * N_STATE + tid];
                float sig = 1.0f / (1.0f + expf(-Sq));
                float silu_Sq = Sq * sig;
                d_Sq[tid] = d_out * (silu_Sq + Sq * sig * (1.0f - sig));
            }
            __syncthreads();

            // dS += outer(d_Sq, q)
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                dS[i] += d_Sq[row] * q_raw[col];
            }
            __syncthreads();

            // d_q
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    float S_t_ij = alpha * S[i * N_STATE + tid] + delta[i] * k_norm[tid];
                    sum += S_t_ij * d_Sq[i];
                }
                d_q_raw[tid] = sum;
            }
            __syncthreads();

            // Backward through S update
            if (tid < N_STATE) {
                float d_delta_local = 0.0f;
                float d_kn_local = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    float dS_ij = dS[tid * N_STATE + j];
                    d_delta_local += dS_ij * k_norm[j];
                    d_kn_local += dS[j * N_STATE + tid] * delta[j];
                }
                d_delta[tid] = d_delta_local;
                d_k_norm[tid] = d_kn_local;
            }
            __syncthreads();

            // d_v = d_delta
            if (tid < N_STATE) {
                d_v_raw[tid] = d_delta[tid];
            }
            __syncthreads();

            // d_k_norm contribution from retrieved
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    sum += S[i * N_STATE + tid] * (-d_delta[i]);
                }
                d_k_norm[tid] += sum;
            }
            __syncthreads();

            // Convert d_k_norm to d_k_raw
            {
                __shared__ float k_dot_dk;
                if (tid == 0) {
                    k_dot_dk = 0.0f;
                    for (int i = 0; i < N_STATE; i++) {
                        k_dot_dk += k_raw[i] * d_k_norm[i];
                    }
                }
                __syncthreads();
                if (tid < N_STATE) {
                    float norm = k_norm_val_t;
                    float norm3 = norm * norm * norm;
                    d_k_raw[tid] = d_k_norm[tid] / norm - k_raw[tid] * k_dot_dk / norm3;
                }
                __syncthreads();
            }

            // Write gradients
            int col_idx_t = (t * B + b) * STRIDE;
            if (tid < N_STATE) {
                d_kvq_all[col_idx_t + tid] = d_k_raw[tid];
                d_kvq_all[col_idx_t + N_STATE + tid] = d_v_raw[tid];
                d_kvq_all[col_idx_t + 2 * N_STATE + tid] = d_q_raw[tid];
            }
            __syncthreads();

            // Update dS
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                dS[i] = alpha * dS[i] + (-d_delta[row]) * k_norm[col];
            }
            __syncthreads();
        }
    }

    // Accumulate d_alpha
    if (tid == 0) {
        atomicAdd(d_alpha_accum, d_alpha_local[0]);
    }
}

// ============================================================================
// Launcher Functions
// ============================================================================

void LaunchE74FixedDecayForwardBF16(
    int T, int B, int n_state, float alpha,
    const __nv_bfloat16* kvq_all,
    __nv_bfloat16* S, __nv_bfloat16* output,
    __nv_bfloat16* S_checkpoints, __nv_bfloat16* Sq_cache,
    int checkpoint_interval, cudaStream_t stream
) {
    // Shared memory: n_state^2 (S) + 4*n_state (k,v,q,retrieved)
    int smem = (n_state * n_state + 4 * n_state) * sizeof(float);
    int threads = min(256, n_state * n_state);

    switch (n_state) {
        case 32:
            E74FixedDecayForwardKernel_BF16<32><<<B, threads, smem, stream>>>(
                T, B, alpha, kvq_all, S, output, S_checkpoints, Sq_cache, checkpoint_interval);
            break;
        case 64:
            E74FixedDecayForwardKernel_BF16<64><<<B, threads, smem, stream>>>(
                T, B, alpha, kvq_all, S, output, S_checkpoints, Sq_cache, checkpoint_interval);
            break;
        case 96:
            E74FixedDecayForwardKernel_BF16<96><<<B, threads, smem, stream>>>(
                T, B, alpha, kvq_all, S, output, S_checkpoints, Sq_cache, checkpoint_interval);
            break;
        case 128:
            E74FixedDecayForwardKernel_BF16<128><<<B, threads, smem, stream>>>(
                T, B, alpha, kvq_all, S, output, S_checkpoints, Sq_cache, checkpoint_interval);
            break;
        default:
            printf("E74 Fixed Decay: Unsupported n_state=%d\n", n_state);
    }
}

void LaunchE74FixedDecayBackwardBF16(
    int T, int B, int n_state, float alpha,
    const __nv_bfloat16* kvq_all,
    const __nv_bfloat16* S_checkpoints, const __nv_bfloat16* Sq_cache,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* d_kvq_all, float* d_alpha_accum,
    int checkpoint_interval, cudaStream_t stream
) {
    // Shared memory: 2*n_state^2 (S,dS) + 14*n_state (vectors) + 1 (d_alpha_local)
    int smem = (2 * n_state * n_state + 14 * n_state + 1) * sizeof(float);
    int threads = min(256, n_state * n_state);

    switch (n_state) {
        case 32:
            E74FixedDecayBackwardKernel_BF16<32><<<B, threads, smem, stream>>>(
                T, B, alpha, kvq_all, S_checkpoints, Sq_cache, d_output, d_kvq_all, d_alpha_accum, checkpoint_interval);
            break;
        case 64:
            E74FixedDecayBackwardKernel_BF16<64><<<B, threads, smem, stream>>>(
                T, B, alpha, kvq_all, S_checkpoints, Sq_cache, d_output, d_kvq_all, d_alpha_accum, checkpoint_interval);
            break;
        case 96:
            E74FixedDecayBackwardKernel_BF16<96><<<B, threads, smem, stream>>>(
                T, B, alpha, kvq_all, S_checkpoints, Sq_cache, d_output, d_kvq_all, d_alpha_accum, checkpoint_interval);
            break;
        case 128:
            E74FixedDecayBackwardKernel_BF16<128><<<B, threads, smem, stream>>>(
                T, B, alpha, kvq_all, S_checkpoints, Sq_cache, d_output, d_kvq_all, d_alpha_accum, checkpoint_interval);
            break;
        default:
            printf("E74 Fixed Decay: Unsupported n_state=%d\n", n_state);
    }
}

void LaunchE74FixedDecayForwardFP32(
    int T, int B, int n_state, float alpha,
    const float* kvq_all,
    float* S, float* output,
    float* S_checkpoints, float* Sq_cache,
    int checkpoint_interval, cudaStream_t stream
) {
    int smem = (n_state * n_state + 4 * n_state) * sizeof(float);
    int threads = min(256, n_state * n_state);

    switch (n_state) {
        case 32:
            E74FixedDecayForwardKernel_FP32<32><<<B, threads, smem, stream>>>(
                T, B, alpha, kvq_all, S, output, S_checkpoints, Sq_cache, checkpoint_interval);
            break;
        case 64:
            E74FixedDecayForwardKernel_FP32<64><<<B, threads, smem, stream>>>(
                T, B, alpha, kvq_all, S, output, S_checkpoints, Sq_cache, checkpoint_interval);
            break;
        case 96:
            E74FixedDecayForwardKernel_FP32<96><<<B, threads, smem, stream>>>(
                T, B, alpha, kvq_all, S, output, S_checkpoints, Sq_cache, checkpoint_interval);
            break;
        case 128:
            E74FixedDecayForwardKernel_FP32<128><<<B, threads, smem, stream>>>(
                T, B, alpha, kvq_all, S, output, S_checkpoints, Sq_cache, checkpoint_interval);
            break;
        default:
            printf("E74 Fixed Decay: Unsupported n_state=%d\n", n_state);
    }
}

void LaunchE74FixedDecayBackwardFP32(
    int T, int B, int n_state, float alpha,
    const float* kvq_all,
    const float* S_checkpoints, const float* Sq_cache,
    const float* d_output,
    float* d_kvq_all, float* d_alpha_accum,
    int checkpoint_interval, cudaStream_t stream
) {
    int smem = (2 * n_state * n_state + 14 * n_state + 1) * sizeof(float);
    int threads = min(256, n_state * n_state);

    switch (n_state) {
        case 32:
            E74FixedDecayBackwardKernel_FP32<32><<<B, threads, smem, stream>>>(
                T, B, alpha, kvq_all, S_checkpoints, Sq_cache, d_output, d_kvq_all, d_alpha_accum, checkpoint_interval);
            break;
        case 64:
            E74FixedDecayBackwardKernel_FP32<64><<<B, threads, smem, stream>>>(
                T, B, alpha, kvq_all, S_checkpoints, Sq_cache, d_output, d_kvq_all, d_alpha_accum, checkpoint_interval);
            break;
        case 96:
            E74FixedDecayBackwardKernel_FP32<96><<<B, threads, smem, stream>>>(
                T, B, alpha, kvq_all, S_checkpoints, Sq_cache, d_output, d_kvq_all, d_alpha_accum, checkpoint_interval);
            break;
        case 128:
            E74FixedDecayBackwardKernel_FP32<128><<<B, threads, smem, stream>>>(
                T, B, alpha, kvq_all, S_checkpoints, Sq_cache, d_output, d_kvq_all, d_alpha_accum, checkpoint_interval);
            break;
        default:
            printf("E74 Fixed Decay: Unsupported n_state=%d\n", n_state);
    }
}

// ============================================================================
// Template Class Implementation
// ============================================================================

template<typename T>
E74FixedDecayForward<T>::E74FixedDecayForward(
    bool training,
    int batch_size,
    int n_state,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : training_(training), batch_size_(batch_size), n_state_(n_state), dim_(dim),
      blas_handle_(blas_handle), stream_(stream) {}

template<typename T>
E74FixedDecayBackward<T>::E74FixedDecayBackward(
    int batch_size,
    int n_state,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size), n_state_(n_state), dim_(dim),
      blas_handle_(blas_handle), stream_(stream) {}

// BF16 specializations
template<>
void E74FixedDecayForward<__nv_bfloat16>::Run(
    int steps, float alpha,
    const __nv_bfloat16* W_kvq,
    const __nv_bfloat16* x,
    __nv_bfloat16* S,
    __nv_bfloat16* output,
    __nv_bfloat16* kvq_cache,
    __nv_bfloat16* S_checkpoints,
    __nv_bfloat16* Sq_cache
) {
    // Compute projections: kvq = W_kvq @ x
    // W_kvq: [3*n_state, dim]
    // x: [T, B, dim] -> flatten to [dim, T*B]
    // kvq: [3*n_state, T*B]
    const __nv_bfloat16 alpha_gemm = __float2bfloat16(1.0f);
    const __nv_bfloat16 beta_gemm = __float2bfloat16(0.0f);

    cublasGemmEx(blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        3 * n_state_, steps * batch_size_, dim_,
        &alpha_gemm,
        W_kvq, CUDA_R_16BF, dim_,
        x, CUDA_R_16BF, dim_,
        &beta_gemm,
        kvq_cache, CUDA_R_16BF, 3 * n_state_,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // Run forward kernel
    LaunchE74FixedDecayForwardBF16(
        steps, batch_size_, n_state_, alpha,
        kvq_cache, S, output, S_checkpoints, Sq_cache,
        E74_FD_CHECKPOINT_INTERVAL, stream_);
}

template<>
void E74FixedDecayBackward<__nv_bfloat16>::Run(
    int steps, float alpha,
    const __nv_bfloat16* W_kvq,
    const __nv_bfloat16* x,
    const __nv_bfloat16* kvq_cache,
    const __nv_bfloat16* S_checkpoints,
    const __nv_bfloat16* Sq_cache,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* d_x,
    __nv_bfloat16* d_W_kvq,
    __nv_bfloat16* d_kvq_cache,
    float* d_alpha_accum
) {
    // Run backward kernel
    LaunchE74FixedDecayBackwardBF16(
        steps, batch_size_, n_state_, alpha,
        kvq_cache, S_checkpoints, Sq_cache, d_output,
        d_kvq_cache, d_alpha_accum,
        E74_FD_CHECKPOINT_INTERVAL, stream_);

    // Compute dx = W_kvq^T @ d_kvq
    const __nv_bfloat16 alpha_gemm = __float2bfloat16(1.0f);
    const __nv_bfloat16 beta_gemm = __float2bfloat16(0.0f);

    cublasGemmEx(blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, 3 * n_state_,
        &alpha_gemm,
        W_kvq, CUDA_R_16BF, dim_,
        d_kvq_cache, CUDA_R_16BF, 3 * n_state_,
        &beta_gemm,
        d_x, CUDA_R_16BF, dim_,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // Compute dW_kvq = d_kvq @ x^T
    cublasGemmEx(blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        3 * n_state_, dim_, steps * batch_size_,
        &alpha_gemm,
        d_kvq_cache, CUDA_R_16BF, 3 * n_state_,
        x, CUDA_R_16BF, dim_,
        &beta_gemm,
        d_W_kvq, CUDA_R_16BF, 3 * n_state_,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

// FP32 specializations
template<>
void E74FixedDecayForward<float>::Run(
    int steps, float alpha,
    const float* W_kvq,
    const float* x,
    float* S,
    float* output,
    float* kvq_cache,
    float* S_checkpoints,
    float* Sq_cache
) {
    const float alpha_gemm = 1.0f;
    const float beta_gemm = 0.0f;

    cublasSgemm(blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        3 * n_state_, steps * batch_size_, dim_,
        &alpha_gemm,
        W_kvq, dim_,
        x, dim_,
        &beta_gemm,
        kvq_cache, 3 * n_state_);

    LaunchE74FixedDecayForwardFP32(
        steps, batch_size_, n_state_, alpha,
        kvq_cache, S, output, S_checkpoints, Sq_cache,
        E74_FD_CHECKPOINT_INTERVAL, stream_);
}

template<>
void E74FixedDecayBackward<float>::Run(
    int steps, float alpha,
    const float* W_kvq,
    const float* x,
    const float* kvq_cache,
    const float* S_checkpoints,
    const float* Sq_cache,
    const float* d_output,
    float* d_x,
    float* d_W_kvq,
    float* d_kvq_cache,
    float* d_alpha_accum
) {
    LaunchE74FixedDecayBackwardFP32(
        steps, batch_size_, n_state_, alpha,
        kvq_cache, S_checkpoints, Sq_cache, d_output,
        d_kvq_cache, d_alpha_accum,
        E74_FD_CHECKPOINT_INTERVAL, stream_);

    const float alpha_gemm = 1.0f;
    const float beta_gemm = 0.0f;

    // dx = W_kvq^T @ d_kvq
    cublasSgemm(blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, 3 * n_state_,
        &alpha_gemm,
        W_kvq, dim_,
        d_kvq_cache, 3 * n_state_,
        &beta_gemm,
        d_x, dim_);

    // dW_kvq = d_kvq @ x^T
    cublasSgemm(blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        3 * n_state_, dim_, steps * batch_size_,
        &alpha_gemm,
        d_kvq_cache, 3 * n_state_,
        x, dim_,
        &beta_gemm,
        d_W_kvq, 3 * n_state_);
}

// Explicit instantiations
template struct E74FixedDecayForward<__nv_bfloat16>;
template struct E74FixedDecayForward<float>;
template struct E74FixedDecayBackward<__nv_bfloat16>;
template struct E74FixedDecayBackward<float>;

}  // namespace elman
