/**
 * E75 Vector Gate CUDA Kernel - Input-Dependent Per-Row Decay
 *
 * Mathematical definition:
 *   g = sigmoid(W_beta @ x + b_beta)  # n-dimensional gate vector
 *   S' = diag(g) * S + outer(v - S @ k_norm, k_norm)
 *   output = (S' @ q) * silu(S' @ q)
 *
 * Each row of S gets its own decay controlled by the input.
 *
 * Key differences from E75 Gated Delta:
 * - E75 Gated Delta: S = tanh(beta * S + outer(delta, k_norm))
 * - E75 Vector Gate: S = diag(g) * S + outer(delta, k_norm)  [no tanh, row-wise decay]
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include "hasty/elman_ladder.h"

#define E75VG_CHECKPOINT_INTERVAL 16

namespace elman {

// ============================================================================
// Utility kernels
// ============================================================================

// Add bias and apply sigmoid: out[i] = sigmoid(out[i] + bias[i % n])
__global__ void E75VG_AddBiasSigmoid_BF16(
    __nv_bfloat16* __restrict__ data,
    const __nv_bfloat16* __restrict__ bias,
    int n,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int bias_idx = idx % n;
    float val = __bfloat162float(data[idx]) + __bfloat162float(bias[bias_idx]);
    float sig = 1.0f / (1.0f + expf(-val));
    data[idx] = __float2bfloat16(sig);
}

__global__ void E75VG_AddBiasSigmoid_FP32(
    float* __restrict__ data,
    const float* __restrict__ bias,
    int n,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int bias_idx = idx % n;
    float val = data[idx] + bias[bias_idx];
    float sig = 1.0f / (1.0f + expf(-val));
    data[idx] = sig;
}

// Reduce gradients for bias: db[i] = sum over (T*B) of d_data[j*n + i]
__global__ void E75VG_ReduceBiasGrad_BF16(
    const __nv_bfloat16* __restrict__ d_data,
    __nv_bfloat16* __restrict__ db,
    int n,
    int T_B
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float sum = 0.0f;
    for (int tb = 0; tb < T_B; tb++) {
        sum += __bfloat162float(d_data[tb * n + i]);
    }
    db[i] = __float2bfloat16(sum);
}

__global__ void E75VG_ReduceBiasGrad_FP32(
    const float* __restrict__ d_data,
    float* __restrict__ db,
    int n,
    int T_B
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float sum = 0.0f;
    for (int tb = 0; tb < T_B; tb++) {
        sum += d_data[tb * n + i];
    }
    db[i] = sum;
}

// Apply sigmoid derivative: d_out[i] *= sigmoid_val[i] * (1 - sigmoid_val[i])
__global__ void E75VG_ApplySigmoidDeriv_BF16(
    __nv_bfloat16* __restrict__ d_data,
    const __nv_bfloat16* __restrict__ sigmoid_val,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float d = __bfloat162float(d_data[idx]);
    float s = __bfloat162float(sigmoid_val[idx]);
    float d_pre = d * s * (1.0f - s);
    d_data[idx] = __float2bfloat16(d_pre);
}

__global__ void E75VG_ApplySigmoidDeriv_FP32(
    float* __restrict__ d_data,
    const float* __restrict__ sigmoid_val,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float d = d_data[idx];
    float s = sigmoid_val[idx];
    float d_pre = d * s * (1.0f - s);
    d_data[idx] = d_pre;
}

// ============================================================================
// E75 Vector Gate Forward Kernel (BF16)
// ============================================================================

template<int N_STATE>
__global__ void E75VectorGateForwardKernel_BF16(
    int T,
    int B,
    const __nv_bfloat16* __restrict__ k_all,      // [T, B, N_STATE]
    const __nv_bfloat16* __restrict__ v_all,      // [T, B, N_STATE]
    const __nv_bfloat16* __restrict__ q_all,      // [T, B, N_STATE]
    const __nv_bfloat16* __restrict__ g_all,      // [T, B, N_STATE] gate (post-sigmoid)
    __nv_bfloat16* __restrict__ S,                // [B, N_STATE, N_STATE]
    __nv_bfloat16* __restrict__ output,           // [T, B, N_STATE]
    __nv_bfloat16* __restrict__ S_checkpoints,    // [num_checkpoints, B, N_STATE, N_STATE]
    __nv_bfloat16* __restrict__ Sq_cache,         // [T, B, N_STATE]
    int checkpoint_interval
) {
    int b = blockIdx.x;
    if (b >= B) return;

    // Shared memory layout
    extern __shared__ float shared_mem[];
    float* S_shared = shared_mem;                    // [N_STATE * N_STATE]
    float* k_shared = S_shared + N_STATE * N_STATE;  // [N_STATE]
    float* v_shared = k_shared + N_STATE;            // [N_STATE]
    float* q_shared = v_shared + N_STATE;            // [N_STATE]
    float* retrieved = q_shared + N_STATE;           // [N_STATE]
    float* g_shared = retrieved + N_STATE;           // [N_STATE]

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;

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
        // Load k, v, q, g for this timestep
        if (tid < N_STATE) {
            k_shared[tid] = __bfloat162float(k_all[t * B * N_STATE + b * N_STATE + tid]);
            v_shared[tid] = __bfloat162float(v_all[t * B * N_STATE + b * N_STATE + tid]);
            q_shared[tid] = __bfloat162float(q_all[t * B * N_STATE + b * N_STATE + tid]);
            g_shared[tid] = __bfloat162float(g_all[t * B * N_STATE + b * N_STATE + tid]);
        }
        __syncthreads();

        // Normalize k
        __shared__ float k_norm_sq;
        if (tid == 0) {
            k_norm_sq = 0.0f;
            for (int i = 0; i < N_STATE; i++) {
                k_norm_sq += k_shared[i] * k_shared[i];
            }
            k_norm_sq = sqrtf(k_norm_sq) + 1e-6f;
        }
        __syncthreads();
        if (tid < N_STATE) {
            k_shared[tid] /= k_norm_sq;
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

        // Update state: S = diag(g) * S + outer(delta, k_norm)
        // where delta = v - retrieved
        // diag(g) * S means S[i,:] *= g[i]
        for (int i = tid; i < n2; i += blockDim.x) {
            int row = i / N_STATE;
            int col = i % N_STATE;

            float g_val = g_shared[row];
            float delta_i = v_shared[row] - retrieved[row];
            // Row-wise decay: g_val multiplies entire row
            float update = g_val * S_shared[i] + delta_i * k_shared[col];
            S_shared[i] = update;  // NO tanh!
        }
        __syncthreads();

        // Save checkpoint if at checkpoint boundary
        if ((t + 1) % checkpoint_interval == 0) {
            int cp_idx = (t + 1) / checkpoint_interval;
            for (int i = tid; i < n2; i += blockDim.x) {
                S_checkpoints[cp_idx * B * n2 + b * n2 + i] = __float2bfloat16(S_shared[i]);
            }
        }
        __syncthreads();

        // Compute output: Sq = S @ q, then self-gate
        if (tid < N_STATE) {
            float Sq = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < N_STATE; j++) {
                Sq += S_shared[tid * N_STATE + j] * q_shared[j];
            }
            // Cache Sq for backward
            Sq_cache[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(Sq);

            // Self-gating: Sq * silu(Sq) = Sq * Sq * sigmoid(Sq)
            float sig = 1.0f / (1.0f + expf(-Sq));
            float out_val = Sq * Sq * sig;
            output[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(out_val);
        }
        __syncthreads();
    }

    // Write final state back
    for (int i = tid; i < n2; i += blockDim.x) {
        S[b * n2 + i] = __float2bfloat16(S_shared[i]);
    }
}

// ============================================================================
// E75 Vector Gate Forward Kernel (FP32)
// ============================================================================

template<int N_STATE>
__global__ void E75VectorGateForwardKernel_FP32(
    int T,
    int B,
    const float* __restrict__ k_all,
    const float* __restrict__ v_all,
    const float* __restrict__ q_all,
    const float* __restrict__ g_all,
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
    float* g_shared = retrieved + N_STATE;

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;

    for (int i = tid; i < n2; i += blockDim.x) {
        S_shared[i] = S[b * n2 + i];
    }
    __syncthreads();

    for (int i = tid; i < n2; i += blockDim.x) {
        S_checkpoints[b * n2 + i] = S_shared[i];
    }
    __syncthreads();

    for (int t = 0; t < T; t++) {
        if (tid < N_STATE) {
            k_shared[tid] = k_all[t * B * N_STATE + b * N_STATE + tid];
            v_shared[tid] = v_all[t * B * N_STATE + b * N_STATE + tid];
            q_shared[tid] = q_all[t * B * N_STATE + b * N_STATE + tid];
            g_shared[tid] = g_all[t * B * N_STATE + b * N_STATE + tid];
        }
        __syncthreads();

        __shared__ float k_norm_sq;
        if (tid == 0) {
            k_norm_sq = 0.0f;
            for (int i = 0; i < N_STATE; i++) {
                k_norm_sq += k_shared[i] * k_shared[i];
            }
            k_norm_sq = sqrtf(k_norm_sq) + 1e-6f;
        }
        __syncthreads();
        if (tid < N_STATE) {
            k_shared[tid] /= k_norm_sq;
        }
        __syncthreads();

        if (tid < N_STATE) {
            float sum = 0.0f;
            for (int j = 0; j < N_STATE; j++) {
                sum += S_shared[tid * N_STATE + j] * k_shared[j];
            }
            retrieved[tid] = sum;
        }
        __syncthreads();

        for (int i = tid; i < n2; i += blockDim.x) {
            int row = i / N_STATE;
            int col = i % N_STATE;
            float g_val = g_shared[row];
            float delta_i = v_shared[row] - retrieved[row];
            S_shared[i] = g_val * S_shared[i] + delta_i * k_shared[col];
        }
        __syncthreads();

        if ((t + 1) % checkpoint_interval == 0) {
            int cp_idx = (t + 1) / checkpoint_interval;
            for (int i = tid; i < n2; i += blockDim.x) {
                S_checkpoints[cp_idx * B * n2 + b * n2 + i] = S_shared[i];
            }
        }
        __syncthreads();

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

    for (int i = tid; i < n2; i += blockDim.x) {
        S[b * n2 + i] = S_shared[i];
    }
}

// ============================================================================
// E75 Vector Gate Backward Kernel (BF16)
// ============================================================================

template<int N_STATE>
__global__ void E75VectorGateBackwardKernel_BF16(
    int T,
    int B,
    const __nv_bfloat16* __restrict__ k_all,
    const __nv_bfloat16* __restrict__ v_all,
    const __nv_bfloat16* __restrict__ q_all,
    const __nv_bfloat16* __restrict__ g_all,
    const __nv_bfloat16* __restrict__ S_checkpoints,
    const __nv_bfloat16* __restrict__ Sq_cache,
    const __nv_bfloat16* __restrict__ d_output,
    __nv_bfloat16* __restrict__ d_k_all,
    __nv_bfloat16* __restrict__ d_v_all,
    __nv_bfloat16* __restrict__ d_q_all,
    __nv_bfloat16* __restrict__ d_g_all,
    int checkpoint_interval
) {
    int b = blockIdx.x;
    if (b >= B) return;

    extern __shared__ float shared_mem[];
    // In-place update: only 2 matrices instead of 3
    float* S = shared_mem;                            // [N_STATE * N_STATE] - state (updated in-place)
    float* dS = S + N_STATE * N_STATE;                // [N_STATE * N_STATE] - gradient accumulator
    float* k_raw = dS + N_STATE * N_STATE;            // [N_STATE]
    float* v_raw = k_raw + N_STATE;                   // [N_STATE]
    float* q_raw = v_raw + N_STATE;                   // [N_STATE]
    float* k_norm = q_raw + N_STATE;                  // [N_STATE]
    float* delta = k_norm + N_STATE;                  // [N_STATE]
    float* retrieved = delta + N_STATE;               // [N_STATE]
    float* g = retrieved + N_STATE;                   // [N_STATE]
    float* d_k_raw = g + N_STATE;                     // [N_STATE]
    float* d_v_raw = d_k_raw + N_STATE;               // [N_STATE]
    float* d_q_raw = d_v_raw + N_STATE;               // [N_STATE]
    float* d_Sq_shared = d_q_raw + N_STATE;           // [N_STATE]
    float* d_delta = d_Sq_shared + N_STATE;           // [N_STATE]
    float* d_k_norm = d_delta + N_STATE;              // [N_STATE]

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;

    // Initialize dS to zero
    for (int i = tid; i < n2; i += blockDim.x) {
        dS[i] = 0.0f;
    }
    __syncthreads();

    int num_segments = (T + checkpoint_interval - 1) / checkpoint_interval;

    for (int seg = num_segments - 1; seg >= 0; seg--) {
        int t_start = seg * checkpoint_interval;
        int t_end = min(t_start + checkpoint_interval, T);

        // Backward through segment
        for (int t = t_end - 1; t >= t_start; t--) {
            // Reload checkpoint into S (will be updated in-place during recomputation)
            for (int i = tid; i < n2; i += blockDim.x) {
                S[i] = __bfloat162float(S_checkpoints[seg * B * n2 + b * n2 + i]);
            }
            __syncthreads();

            // Recompute forward to step t
            __shared__ float k_norm_val_t;
            for (int tt = t_start; tt <= t; tt++) {
                if (tid < N_STATE) {
                    k_raw[tid] = __bfloat162float(k_all[tt * B * N_STATE + b * N_STATE + tid]);
                    v_raw[tid] = __bfloat162float(v_all[tt * B * N_STATE + b * N_STATE + tid]);
                    q_raw[tid] = __bfloat162float(q_all[tt * B * N_STATE + b * N_STATE + tid]);
                    g[tid] = __bfloat162float(g_all[tt * B * N_STATE + b * N_STATE + tid]);
                }
                __syncthreads();

                // Normalize k
                if (tid == 0) {
                    float sum_sq = 0.0f;
                    for (int i = 0; i < N_STATE; i++) sum_sq += k_raw[i] * k_raw[i];
                    k_norm_val_t = sqrtf(sum_sq) + 1e-6f;
                }
                __syncthreads();
                if (tid < N_STATE) {
                    k_norm[tid] = k_raw[tid] / k_norm_val_t;
                }
                __syncthreads();

                // Compute retrieved = S @ k_norm (S is current state)
                if (tid < N_STATE) {
                    float sum = 0.0f;
                    for (int j = 0; j < N_STATE; j++) {
                        sum += S[tid * N_STATE + j] * k_norm[j];
                    }
                    retrieved[tid] = sum;
                    delta[tid] = v_raw[tid] - retrieved[tid];
                }
                __syncthreads();

                // Update state IN-PLACE only for tt < t (keep S = S_{t-1} for backward)
                if (tt < t) {
                    for (int i = tid; i < n2; i += blockDim.x) {
                        int row = i / N_STATE;
                        int col = i % N_STATE;
                        float update = g[row] * S[i] + delta[row] * k_norm[col];
                        S[i] = update;  // NO tanh
                    }
                    __syncthreads();
                }
            }

            // Now S holds S_{t-1} (state before step t)
            // k_norm, v_raw, q_raw, delta, g are loaded for step t

            // Backward through output: out = Sq * silu(Sq) = Sq^2 * sigmoid(Sq)
            if (tid < N_STATE) {
                float d_out = __bfloat162float(d_output[t * B * N_STATE + b * N_STATE + tid]);
                float Sq = __bfloat162float(Sq_cache[t * B * N_STATE + b * N_STATE + tid]);
                float sig = 1.0f / (1.0f + expf(-Sq));
                // d_out/d_Sq for out = Sq^2 * sigmoid(Sq)
                // d/dSq = 2*Sq*sig + Sq^2*sig*(1-sig) = Sq*sig*(2 + Sq*(1-sig))
                float d_Sq = d_out * (2.0f * Sq * sig + Sq * Sq * sig * (1.0f - sig));
                d_Sq_shared[tid] = d_Sq;
            }
            __syncthreads();

            // dS += outer(d_Sq, q) from Sq = S @ q
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                dS[i] += d_Sq_shared[row] * q_raw[col];
            }
            __syncthreads();

            // d_q = S_t^T @ d_Sq (compute S_t on-the-fly from S = S_{t-1})
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    // Compute S_t[i,tid] on-the-fly: S_t = g * S + outer(delta, k_norm)
                    float S_t_ij = g[i] * S[i * N_STATE + tid] + delta[i] * k_norm[tid];
                    sum += S_t_ij * d_Sq_shared[i];
                }
                d_q_raw[tid] = sum;
            }
            __syncthreads();

            // Backward through state update: S_t = g * S + outer(delta, k)
            // where S = S_{t-1}
            // d_g[i] = sum_j(dS[i,j] * S[i,j])  (uses S = S_{t-1})
            // d_delta[i] = sum_j(dS[i,j] * k_norm[j])
            // d_k_norm[j] = sum_i(dS[i,j] * delta[i])

            // Compute d_delta and d_g
            if (tid < N_STATE) {
                float d_delta_local = 0.0f;
                float d_g_local = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    float dS_ij = dS[tid * N_STATE + j];
                    d_delta_local += dS_ij * k_norm[j];
                    d_g_local += dS_ij * S[tid * N_STATE + j];  // Uses S = S_{t-1}
                }
                d_delta[tid] = d_delta_local;
                // Store d_g for this timestep
                d_g_all[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(d_g_local);
            }
            __syncthreads();

            // Compute d_k_norm
            if (tid < N_STATE) {
                float d_k_norm_local = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    d_k_norm_local += dS[i * N_STATE + tid] * delta[i];
                }
                d_k_norm[tid] = d_k_norm_local;
            }
            __syncthreads();

            // d_v = d_delta (from delta = v - retrieved)
            // d_retrieved = -d_delta
            if (tid < N_STATE) {
                d_v_raw[tid] = d_delta[tid];
            }

            // d_k_norm += S^T @ (-d_delta) (from retrieved = S @ k_norm, where S = S_{t-1})
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    sum += S[i * N_STATE + tid] * (-d_delta[i]);
                }
                d_k_norm[tid] += sum;
            }
            __syncthreads();

            // Convert d_k_norm to d_k_raw via normalization gradient
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

            // Write gradients for k, v, q
            if (tid < N_STATE) {
                d_k_all[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(d_k_raw[tid]);
                d_v_all[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(d_v_raw[tid]);
                d_q_all[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(d_q_raw[tid]);
            }
            __syncthreads();

            // Update dS for next iteration:
            // dS_prev has two contributions:
            // 1. From S_t = g * S + outer(delta, k_norm): dS * g
            // 2. From retrieved = S @ k_norm: outer(-d_delta, k_norm)
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                // Contribution 1: gradient through g * S (where S = S_{t-1})
                // Contribution 2: gradient through retrieved = S @ k_norm
                //                 d_retrieved = -d_delta, so dS += outer(-d_delta, k_norm)
                dS[i] = dS[i] * g[row] + (-d_delta[row]) * k_norm[col];
            }
            __syncthreads();
        }
    }
}

// ============================================================================
// E75 Vector Gate Backward Kernel (FP32)
// ============================================================================

template<int N_STATE>
__global__ void E75VectorGateBackwardKernel_FP32(
    int T,
    int B,
    const float* __restrict__ k_all,
    const float* __restrict__ v_all,
    const float* __restrict__ q_all,
    const float* __restrict__ g_all,
    const float* __restrict__ S_checkpoints,
    const float* __restrict__ Sq_cache,
    const float* __restrict__ d_output,
    float* __restrict__ d_k_all,
    float* __restrict__ d_v_all,
    float* __restrict__ d_q_all,
    float* __restrict__ d_g_all,
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
    float* delta = k_norm + N_STATE;
    float* retrieved = delta + N_STATE;
    float* g = retrieved + N_STATE;
    float* d_k_raw = g + N_STATE;
    float* d_v_raw = d_k_raw + N_STATE;
    float* d_q_raw = d_v_raw + N_STATE;
    float* d_Sq_shared = d_q_raw + N_STATE;
    float* d_delta = d_Sq_shared + N_STATE;
    float* d_k_norm = d_delta + N_STATE;

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;

    for (int i = tid; i < n2; i += blockDim.x) {
        dS[i] = 0.0f;
    }
    __syncthreads();

    int num_segments = (T + checkpoint_interval - 1) / checkpoint_interval;

    for (int seg = num_segments - 1; seg >= 0; seg--) {
        int t_start = seg * checkpoint_interval;
        int t_end = min(t_start + checkpoint_interval, T);

        for (int t = t_end - 1; t >= t_start; t--) {
            for (int i = tid; i < n2; i += blockDim.x) {
                S[i] = S_checkpoints[seg * B * n2 + b * n2 + i];
            }
            __syncthreads();

            __shared__ float k_norm_val_t;
            for (int tt = t_start; tt <= t; tt++) {
                if (tid < N_STATE) {
                    k_raw[tid] = k_all[tt * B * N_STATE + b * N_STATE + tid];
                    v_raw[tid] = v_all[tt * B * N_STATE + b * N_STATE + tid];
                    q_raw[tid] = q_all[tt * B * N_STATE + b * N_STATE + tid];
                    g[tid] = g_all[tt * B * N_STATE + b * N_STATE + tid];
                }
                __syncthreads();

                if (tid == 0) {
                    float sum_sq = 0.0f;
                    for (int i = 0; i < N_STATE; i++) sum_sq += k_raw[i] * k_raw[i];
                    k_norm_val_t = sqrtf(sum_sq) + 1e-6f;
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
                    delta[tid] = v_raw[tid] - retrieved[tid];
                }
                __syncthreads();

                if (tt < t) {
                    for (int i = tid; i < n2; i += blockDim.x) {
                        int row = i / N_STATE;
                        int col = i % N_STATE;
                        S[i] = g[row] * S[i] + delta[row] * k_norm[col];
                    }
                    __syncthreads();
                }
            }

            if (tid < N_STATE) {
                float d_out = d_output[t * B * N_STATE + b * N_STATE + tid];
                float Sq = Sq_cache[t * B * N_STATE + b * N_STATE + tid];
                float sig = 1.0f / (1.0f + expf(-Sq));
                float d_Sq = d_out * (2.0f * Sq * sig + Sq * Sq * sig * (1.0f - sig));
                d_Sq_shared[tid] = d_Sq;
            }
            __syncthreads();

            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                dS[i] += d_Sq_shared[row] * q_raw[col];
            }
            __syncthreads();

            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    float S_t_ij = g[i] * S[i * N_STATE + tid] + delta[i] * k_norm[tid];
                    sum += S_t_ij * d_Sq_shared[i];
                }
                d_q_raw[tid] = sum;
            }
            __syncthreads();

            if (tid < N_STATE) {
                float d_delta_local = 0.0f;
                float d_g_local = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    float dS_ij = dS[tid * N_STATE + j];
                    d_delta_local += dS_ij * k_norm[j];
                    d_g_local += dS_ij * S[tid * N_STATE + j];
                }
                d_delta[tid] = d_delta_local;
                d_g_all[t * B * N_STATE + b * N_STATE + tid] = d_g_local;
            }
            __syncthreads();

            if (tid < N_STATE) {
                float d_k_norm_local = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    d_k_norm_local += dS[i * N_STATE + tid] * delta[i];
                }
                d_k_norm[tid] = d_k_norm_local;
            }
            __syncthreads();

            if (tid < N_STATE) {
                d_v_raw[tid] = d_delta[tid];
            }

            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    sum += S[i * N_STATE + tid] * (-d_delta[i]);
                }
                d_k_norm[tid] += sum;
            }
            __syncthreads();

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

            if (tid < N_STATE) {
                d_k_all[t * B * N_STATE + b * N_STATE + tid] = d_k_raw[tid];
                d_v_all[t * B * N_STATE + b * N_STATE + tid] = d_v_raw[tid];
                d_q_all[t * B * N_STATE + b * N_STATE + tid] = d_q_raw[tid];
            }
            __syncthreads();

            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                dS[i] = dS[i] * g[row] + (-d_delta[row]) * k_norm[col];
            }
            __syncthreads();
        }
    }
}

// ============================================================================
// E75VectorGateForward Implementation
// ============================================================================

template<typename DataT>
E75VectorGateForward<DataT>::E75VectorGateForward(
    bool training,
    int batch_size,
    int n_state,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : training_(training),
      batch_size_(batch_size),
      n_state_(n_state),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void E75VectorGateForward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_k,
    const __nv_bfloat16* W_v,
    const __nv_bfloat16* W_q,
    const __nv_bfloat16* W_beta,
    const __nv_bfloat16* b_beta,
    const __nv_bfloat16* x,
    __nv_bfloat16* S,
    __nv_bfloat16* output,
    __nv_bfloat16* k_cache,
    __nv_bfloat16* v_cache,
    __nv_bfloat16* q_cache,
    __nv_bfloat16* g_cache,
    __nv_bfloat16* S_cache
) {
    int T = steps;
    int B = batch_size_;
    int n = n_state_;
    int d = dim_;

    const float alpha = 1.0f, beta_zero = 0.0f;

    // Project k, v, q
    cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        n, T * B, d, &alpha,
        W_k, CUDA_R_16BF, d, x, CUDA_R_16BF, d,
        &beta_zero, k_cache, CUDA_R_16BF, n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        n, T * B, d, &alpha,
        W_v, CUDA_R_16BF, d, x, CUDA_R_16BF, d,
        &beta_zero, v_cache, CUDA_R_16BF, n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        n, T * B, d, &alpha,
        W_q, CUDA_R_16BF, d, x, CUDA_R_16BF, d,
        &beta_zero, q_cache, CUDA_R_16BF, n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // Project gate = sigmoid(W_beta @ x + b_beta)
    cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        n, T * B, d, &alpha,
        W_beta, CUDA_R_16BF, d, x, CUDA_R_16BF, d,
        &beta_zero, g_cache, CUDA_R_16BF, n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // Apply bias and sigmoid to gate
    int total = T * B * n;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    E75VG_AddBiasSigmoid_BF16<<<blocks, threads, 0, stream_>>>(
        g_cache, b_beta, n, total);

    // Calculate workspace offsets for checkpoints
    int num_checkpoints = (T + E75VG_CHECKPOINT_INTERVAL - 1) / E75VG_CHECKPOINT_INTERVAL + 1;
    __nv_bfloat16* s_checkpoints = S_cache;
    __nv_bfloat16* sq_cache = S_cache + num_checkpoints * B * n * n;

    // Run forward kernel
    int shared_size = (n * n + 6 * n) * sizeof(float);  // S + k,v,q,retrieved,g
    int kernel_threads = min(256, n * n);

    #define DISPATCH_E75VG_FORWARD(N) \
        E75VectorGateForwardKernel_BF16<N><<<B, kernel_threads, shared_size, stream_>>>( \
            T, B, \
            k_cache, v_cache, q_cache, g_cache, \
            S, output, s_checkpoints, sq_cache, \
            E75VG_CHECKPOINT_INTERVAL)

    if (n == 1) { DISPATCH_E75VG_FORWARD(1); }
    else if (n == 2) { DISPATCH_E75VG_FORWARD(2); }
    else if (n == 4) { DISPATCH_E75VG_FORWARD(4); }
    else if (n == 8) { DISPATCH_E75VG_FORWARD(8); }
    else if (n == 16) { DISPATCH_E75VG_FORWARD(16); }
    else if (n == 24) { DISPATCH_E75VG_FORWARD(24); }
    else if (n == 28) { DISPATCH_E75VG_FORWARD(28); }
    else if (n == 32) { DISPATCH_E75VG_FORWARD(32); }
    else if (n == 48) { DISPATCH_E75VG_FORWARD(48); }
    else if (n == 64) { DISPATCH_E75VG_FORWARD(64); }
    else if (n == 96) { DISPATCH_E75VG_FORWARD(96); }
    else if (n == 128) { DISPATCH_E75VG_FORWARD(128); }
    else {
        fprintf(stderr, "E75 Vector Gate Forward: unsupported n_state=%d\n", n);
    }

    #undef DISPATCH_E75VG_FORWARD
}

template<>
void E75VectorGateForward<float>::Run(
    int steps,
    const float* W_k,
    const float* W_v,
    const float* W_q,
    const float* W_beta,
    const float* b_beta,
    const float* x,
    float* S,
    float* output,
    float* k_cache,
    float* v_cache,
    float* q_cache,
    float* g_cache,
    float* S_cache
) {
    int T = steps;
    int B = batch_size_;
    int n = n_state_;
    int d = dim_;

    const float alpha = 1.0f, beta_zero = 0.0f;

    // Project k, v, q
    cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        n, T * B, d, &alpha,
        W_k, CUDA_R_32F, d, x, CUDA_R_32F, d,
        &beta_zero, k_cache, CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        n, T * B, d, &alpha,
        W_v, CUDA_R_32F, d, x, CUDA_R_32F, d,
        &beta_zero, v_cache, CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        n, T * B, d, &alpha,
        W_q, CUDA_R_32F, d, x, CUDA_R_32F, d,
        &beta_zero, q_cache, CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // Project gate
    cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        n, T * B, d, &alpha,
        W_beta, CUDA_R_32F, d, x, CUDA_R_32F, d,
        &beta_zero, g_cache, CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // Apply bias and sigmoid
    int total = T * B * n;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    E75VG_AddBiasSigmoid_FP32<<<blocks, threads, 0, stream_>>>(
        g_cache, b_beta, n, total);

    int num_checkpoints = (T + E75VG_CHECKPOINT_INTERVAL - 1) / E75VG_CHECKPOINT_INTERVAL + 1;
    float* s_checkpoints = S_cache;
    float* sq_cache = S_cache + num_checkpoints * B * n * n;

    int shared_size = (n * n + 6 * n) * sizeof(float);
    int kernel_threads = min(256, n * n);

    #define DISPATCH_E75VG_FORWARD_FP32(N) \
        E75VectorGateForwardKernel_FP32<N><<<B, kernel_threads, shared_size, stream_>>>( \
            T, B, \
            k_cache, v_cache, q_cache, g_cache, \
            S, output, s_checkpoints, sq_cache, \
            E75VG_CHECKPOINT_INTERVAL)

    if (n == 1) { DISPATCH_E75VG_FORWARD_FP32(1); }
    else if (n == 2) { DISPATCH_E75VG_FORWARD_FP32(2); }
    else if (n == 4) { DISPATCH_E75VG_FORWARD_FP32(4); }
    else if (n == 8) { DISPATCH_E75VG_FORWARD_FP32(8); }
    else if (n == 16) { DISPATCH_E75VG_FORWARD_FP32(16); }
    else if (n == 24) { DISPATCH_E75VG_FORWARD_FP32(24); }
    else if (n == 28) { DISPATCH_E75VG_FORWARD_FP32(28); }
    else if (n == 32) { DISPATCH_E75VG_FORWARD_FP32(32); }
    else if (n == 48) { DISPATCH_E75VG_FORWARD_FP32(48); }
    else if (n == 64) { DISPATCH_E75VG_FORWARD_FP32(64); }
    else if (n == 96) { DISPATCH_E75VG_FORWARD_FP32(96); }
    else if (n == 128) { DISPATCH_E75VG_FORWARD_FP32(128); }
    else {
        fprintf(stderr, "E75 Vector Gate Forward FP32: unsupported n_state=%d\n", n);
    }

    #undef DISPATCH_E75VG_FORWARD_FP32
}

template struct E75VectorGateForward<__nv_bfloat16>;
template struct E75VectorGateForward<float>;

// ============================================================================
// E75VectorGateBackward Implementation
// ============================================================================

template<typename DataT>
E75VectorGateBackward<DataT>::E75VectorGateBackward(
    int batch_size,
    int n_state,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      n_state_(n_state),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void E75VectorGateBackward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_k,
    const __nv_bfloat16* W_v,
    const __nv_bfloat16* W_q,
    const __nv_bfloat16* W_beta,
    const __nv_bfloat16* x,
    const __nv_bfloat16* S_checkpoints,
    const __nv_bfloat16* Sq_cache,
    const __nv_bfloat16* k_cache,
    const __nv_bfloat16* v_cache,
    const __nv_bfloat16* q_cache,
    const __nv_bfloat16* g_cache,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* dx,
    __nv_bfloat16* dW_k,
    __nv_bfloat16* dW_v,
    __nv_bfloat16* dW_q,
    __nv_bfloat16* dW_beta,
    __nv_bfloat16* db_beta,
    __nv_bfloat16* workspace
) {
    int T = steps;
    int B = batch_size_;
    int n = n_state_;
    int d = dim_;

    // Workspace layout: [d_k_all: T*B*n] [d_v_all: T*B*n] [d_q_all: T*B*n] [d_g_all: T*B*n]
    __nv_bfloat16* d_k_all = workspace;
    __nv_bfloat16* d_v_all = d_k_all + T * B * n;
    __nv_bfloat16* d_q_all = d_v_all + T * B * n;
    __nv_bfloat16* d_g_all = d_q_all + T * B * n;

    // Shared memory: 2*n^2 + 13*n floats (in-place updates)
    int shared_size = (2 * n * n + 13 * n) * sizeof(float);
    int threads = min(256, n * n);

    #define SET_SHARED_MEM_AND_DISPATCH_E75VG_BACKWARD(N) \
        { \
            cudaError_t attr_err = cudaFuncSetAttribute(E75VectorGateBackwardKernel_BF16<N>, \
                cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size); \
            if (attr_err != cudaSuccess) { \
                fprintf(stderr, "E75 Vector Gate Backward: n_state=%d requires %d KB shared memory but GPU limit exceeded (error: %s)\n", \
                    N, shared_size / 1024, cudaGetErrorString(attr_err)); \
            } else { \
                E75VectorGateBackwardKernel_BF16<N><<<B, threads, shared_size, stream_>>>( \
                    T, B, \
                    k_cache, v_cache, q_cache, g_cache, \
                    S_checkpoints, Sq_cache, d_output, \
                    d_k_all, d_v_all, d_q_all, d_g_all, \
                    E75VG_CHECKPOINT_INTERVAL); \
            } \
        }

    #define DISPATCH_E75VG_BACKWARD(N) \
        E75VectorGateBackwardKernel_BF16<N><<<B, threads, shared_size, stream_>>>( \
            T, B, \
            k_cache, v_cache, q_cache, g_cache, \
            S_checkpoints, Sq_cache, d_output, \
            d_k_all, d_v_all, d_q_all, d_g_all, \
            E75VG_CHECKPOINT_INTERVAL)

    // For small n_state (<64), use regular dispatch; for n>=64, set extended shared memory
    if (n == 1) { DISPATCH_E75VG_BACKWARD(1); }
    else if (n == 2) { DISPATCH_E75VG_BACKWARD(2); }
    else if (n == 4) { DISPATCH_E75VG_BACKWARD(4); }
    else if (n == 8) { DISPATCH_E75VG_BACKWARD(8); }
    else if (n == 16) { DISPATCH_E75VG_BACKWARD(16); }
    else if (n == 24) { DISPATCH_E75VG_BACKWARD(24); }
    else if (n == 28) { DISPATCH_E75VG_BACKWARD(28); }
    else if (n == 32) { DISPATCH_E75VG_BACKWARD(32); }
    else if (n == 48) { DISPATCH_E75VG_BACKWARD(48); }
    else if (n == 64) { SET_SHARED_MEM_AND_DISPATCH_E75VG_BACKWARD(64); }
    else if (n == 96) { SET_SHARED_MEM_AND_DISPATCH_E75VG_BACKWARD(96); }
    else if (n == 128) { SET_SHARED_MEM_AND_DISPATCH_E75VG_BACKWARD(128); }
    else {
        fprintf(stderr, "E75 Vector Gate Backward: unsupported n_state=%d\n", n);
    }

    #undef SET_SHARED_MEM_AND_DISPATCH_E75VG_BACKWARD
    #undef DISPATCH_E75VG_BACKWARD

    // Apply sigmoid derivative to d_g_all (gate was sigmoid in forward)
    int total_g = T * B * n;
    int threads_deriv = 256;
    int blocks_deriv = (total_g + threads_deriv - 1) / threads_deriv;
    E75VG_ApplySigmoidDeriv_BF16<<<blocks_deriv, threads_deriv, 0, stream_>>>(
        d_g_all, g_cache, total_g);

    // Weight gradients via cuBLAS
    const float alpha = 1.0f, beta_zero = 0.0f;

    // dW_k = x @ d_k_all^T
    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        d, n, T * B, &alpha,
        x, CUDA_R_16BF, d, d_k_all, CUDA_R_16BF, n,
        &beta_zero, dW_k, CUDA_R_16BF, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        d, n, T * B, &alpha,
        x, CUDA_R_16BF, d, d_v_all, CUDA_R_16BF, n,
        &beta_zero, dW_v, CUDA_R_16BF, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        d, n, T * B, &alpha,
        x, CUDA_R_16BF, d, d_q_all, CUDA_R_16BF, n,
        &beta_zero, dW_q, CUDA_R_16BF, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        d, n, T * B, &alpha,
        x, CUDA_R_16BF, d, d_g_all, CUDA_R_16BF, n,
        &beta_zero, dW_beta, CUDA_R_16BF, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // db_beta = sum over (T*B) of d_g_all
    int threads_db = 256;
    int blocks_db = (n + threads_db - 1) / threads_db;
    E75VG_ReduceBiasGrad_BF16<<<blocks_db, threads_db, 0, stream_>>>(
        d_g_all, db_beta, n, T * B);

    // dx = W_k @ d_k_all + W_v @ d_v_all + W_q @ d_q_all + W_beta @ d_g_all
    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
        d, T * B, n, &alpha,
        W_k, CUDA_R_16BF, d, d_k_all, CUDA_R_16BF, n,
        &beta_zero, dx, CUDA_R_16BF, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    const float alpha_add = 1.0f, beta_add = 1.0f;
    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
        d, T * B, n, &alpha_add,
        W_v, CUDA_R_16BF, d, d_v_all, CUDA_R_16BF, n,
        &beta_add, dx, CUDA_R_16BF, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
        d, T * B, n, &alpha_add,
        W_q, CUDA_R_16BF, d, d_q_all, CUDA_R_16BF, n,
        &beta_add, dx, CUDA_R_16BF, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
        d, T * B, n, &alpha_add,
        W_beta, CUDA_R_16BF, d, d_g_all, CUDA_R_16BF, n,
        &beta_add, dx, CUDA_R_16BF, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

template<>
void E75VectorGateBackward<float>::Run(
    int steps,
    const float* W_k,
    const float* W_v,
    const float* W_q,
    const float* W_beta,
    const float* x,
    const float* S_checkpoints,
    const float* Sq_cache,
    const float* k_cache,
    const float* v_cache,
    const float* q_cache,
    const float* g_cache,
    const float* d_output,
    float* dx,
    float* dW_k,
    float* dW_v,
    float* dW_q,
    float* dW_beta,
    float* db_beta,
    float* workspace
) {
    int T = steps;
    int B = batch_size_;
    int n = n_state_;
    int d = dim_;

    float* d_k_all = workspace;
    float* d_v_all = d_k_all + T * B * n;
    float* d_q_all = d_v_all + T * B * n;
    float* d_g_all = d_q_all + T * B * n;

    int shared_size = (2 * n * n + 13 * n) * sizeof(float);
    int threads = min(256, n * n);

    #define SET_SHARED_MEM_AND_DISPATCH_E75VG_BACKWARD_FP32(N) \
        { \
            cudaError_t attr_err = cudaFuncSetAttribute(E75VectorGateBackwardKernel_FP32<N>, \
                cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size); \
            if (attr_err != cudaSuccess) { \
                fprintf(stderr, "E75 Vector Gate Backward FP32: n_state=%d requires %d KB shared memory (error: %s)\n", \
                    N, shared_size / 1024, cudaGetErrorString(attr_err)); \
            } else { \
                E75VectorGateBackwardKernel_FP32<N><<<B, threads, shared_size, stream_>>>( \
                    T, B, \
                    k_cache, v_cache, q_cache, g_cache, \
                    S_checkpoints, Sq_cache, d_output, \
                    d_k_all, d_v_all, d_q_all, d_g_all, \
                    E75VG_CHECKPOINT_INTERVAL); \
            } \
        }

    #define DISPATCH_E75VG_BACKWARD_FP32(N) \
        E75VectorGateBackwardKernel_FP32<N><<<B, threads, shared_size, stream_>>>( \
            T, B, \
            k_cache, v_cache, q_cache, g_cache, \
            S_checkpoints, Sq_cache, d_output, \
            d_k_all, d_v_all, d_q_all, d_g_all, \
            E75VG_CHECKPOINT_INTERVAL)

    if (n == 1) { DISPATCH_E75VG_BACKWARD_FP32(1); }
    else if (n == 2) { DISPATCH_E75VG_BACKWARD_FP32(2); }
    else if (n == 4) { DISPATCH_E75VG_BACKWARD_FP32(4); }
    else if (n == 8) { DISPATCH_E75VG_BACKWARD_FP32(8); }
    else if (n == 16) { DISPATCH_E75VG_BACKWARD_FP32(16); }
    else if (n == 24) { DISPATCH_E75VG_BACKWARD_FP32(24); }
    else if (n == 28) { DISPATCH_E75VG_BACKWARD_FP32(28); }
    else if (n == 32) { DISPATCH_E75VG_BACKWARD_FP32(32); }
    else if (n == 48) { DISPATCH_E75VG_BACKWARD_FP32(48); }
    else if (n == 64) { SET_SHARED_MEM_AND_DISPATCH_E75VG_BACKWARD_FP32(64); }
    else if (n == 96) { SET_SHARED_MEM_AND_DISPATCH_E75VG_BACKWARD_FP32(96); }
    else if (n == 128) { SET_SHARED_MEM_AND_DISPATCH_E75VG_BACKWARD_FP32(128); }
    else {
        fprintf(stderr, "E75 Vector Gate Backward FP32: unsupported n_state=%d\n", n);
    }

    #undef SET_SHARED_MEM_AND_DISPATCH_E75VG_BACKWARD_FP32
    #undef DISPATCH_E75VG_BACKWARD_FP32

    // Apply sigmoid derivative
    int total_g = T * B * n;
    int threads_deriv = 256;
    int blocks_deriv = (total_g + threads_deriv - 1) / threads_deriv;
    E75VG_ApplySigmoidDeriv_FP32<<<blocks_deriv, threads_deriv, 0, stream_>>>(
        d_g_all, g_cache, total_g);

    const float alpha = 1.0f, beta_zero = 0.0f;

    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        d, n, T * B, &alpha,
        x, CUDA_R_32F, d, d_k_all, CUDA_R_32F, n,
        &beta_zero, dW_k, CUDA_R_32F, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        d, n, T * B, &alpha,
        x, CUDA_R_32F, d, d_v_all, CUDA_R_32F, n,
        &beta_zero, dW_v, CUDA_R_32F, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        d, n, T * B, &alpha,
        x, CUDA_R_32F, d, d_q_all, CUDA_R_32F, n,
        &beta_zero, dW_q, CUDA_R_32F, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        d, n, T * B, &alpha,
        x, CUDA_R_32F, d, d_g_all, CUDA_R_32F, n,
        &beta_zero, dW_beta, CUDA_R_32F, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    int threads_db = 256;
    int blocks_db = (n + threads_db - 1) / threads_db;
    E75VG_ReduceBiasGrad_FP32<<<blocks_db, threads_db, 0, stream_>>>(
        d_g_all, db_beta, n, T * B);

    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
        d, T * B, n, &alpha,
        W_k, CUDA_R_32F, d, d_k_all, CUDA_R_32F, n,
        &beta_zero, dx, CUDA_R_32F, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    const float alpha_add = 1.0f, beta_add = 1.0f;
    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
        d, T * B, n, &alpha_add,
        W_v, CUDA_R_32F, d, d_v_all, CUDA_R_32F, n,
        &beta_add, dx, CUDA_R_32F, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
        d, T * B, n, &alpha_add,
        W_q, CUDA_R_32F, d, d_q_all, CUDA_R_32F, n,
        &beta_add, dx, CUDA_R_32F, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
        d, T * B, n, &alpha_add,
        W_beta, CUDA_R_32F, d, d_g_all, CUDA_R_32F, n,
        &beta_add, dx, CUDA_R_32F, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

template struct E75VectorGateBackward<__nv_bfloat16>;
template struct E75VectorGateBackward<float>;

}  // namespace elman
