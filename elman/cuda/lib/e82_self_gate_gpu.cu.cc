/**
 * E82 Self-Gating Matrix CUDA Kernel
 *
 * A single matrix S gates itself - maximum autopoiesis where the memory
 * controls its own forgetting without any external modulation matrix.
 *
 * Mathematical definition:
 *   FUSED projection: kvqm = W_kvqm @ x  (k, v, q, m)
 *
 *   # Self-gating
 *   G = sigmoid(outer(S @ m_norm, k_norm) + alpha * S) + epsilon
 *   s_delta = v - S @ k_norm
 *   S' = G * S + outer(s_delta, k_norm)
 *
 *   # Output
 *   Sq = S' @ q
 *   output = Sq * silu(Sq)
 *
 * Key insight: Minimal architecture - only ONE matrix that determines its own
 * forgetting. Much simpler than E79 (coupled matrices).
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include "hasty/elman_ladder.h"

#define E82_CHECKPOINT_INTERVAL 16

namespace elman {

// ============================================================================
// E82 Forward Kernel - Self-Gating Matrix System (BF16)
// ============================================================================

template<int N_STATE>
__global__ void E82SelfGateForwardKernel_BF16(
    int T,
    int B,
    const __nv_bfloat16* __restrict__ kvqm_all,   // [4*N_STATE, T*B] column-major
    float alpha,
    float epsilon,
    __nv_bfloat16* __restrict__ S,                // [B, N_STATE, N_STATE]
    __nv_bfloat16* __restrict__ output,           // [T, B, N_STATE]
    __nv_bfloat16* __restrict__ S_checkpoints,    // [num_checkpoints, B, N_STATE, N_STATE]
    __nv_bfloat16* __restrict__ Sq_cache,         // [T, B, N_STATE]
    __nv_bfloat16* __restrict__ gate_cache,       // [T, B, N_STATE, N_STATE]
    int checkpoint_interval
) {
    int b = blockIdx.x;
    if (b >= B) return;

    // Shared memory layout - only need space for one matrix plus vectors
    extern __shared__ float shared_mem[];
    float* S_shared = shared_mem;                         // [N_STATE * N_STATE]
    float* k_shared = S_shared + N_STATE * N_STATE;       // [N_STATE]
    float* v_shared = k_shared + N_STATE;                 // [N_STATE]
    float* q_shared = v_shared + N_STATE;                 // [N_STATE]
    float* m_vec_shared = q_shared + N_STATE;             // [N_STATE]
    float* Sm_shared = m_vec_shared + N_STATE;            // [N_STATE] - S @ m_norm
    float* s_retrieved = Sm_shared + N_STATE;             // [N_STATE]

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;
    const int STRIDE = 4 * N_STATE;

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

        // Load k, v, q, m for this timestep
        if (tid < N_STATE) {
            k_shared[tid] = __bfloat162float(kvqm_all[col_idx + tid]);
            v_shared[tid] = __bfloat162float(kvqm_all[col_idx + N_STATE + tid]);
            q_shared[tid] = __bfloat162float(kvqm_all[col_idx + 2 * N_STATE + tid]);
            m_vec_shared[tid] = __bfloat162float(kvqm_all[col_idx + 3 * N_STATE + tid]);
        }
        __syncthreads();

        // Normalize k and m
        __shared__ float k_norm_val, m_norm_val;
        if (tid == 0) {
            float k_sum = 0.0f, m_sum = 0.0f;
            for (int i = 0; i < N_STATE; i++) {
                k_sum += k_shared[i] * k_shared[i];
                m_sum += m_vec_shared[i] * m_vec_shared[i];
            }
            k_norm_val = sqrtf(k_sum) + 1e-6f;
            m_norm_val = sqrtf(m_sum) + 1e-6f;
        }
        __syncthreads();
        if (tid < N_STATE) {
            k_shared[tid] /= k_norm_val;
            m_vec_shared[tid] /= m_norm_val;
        }
        __syncthreads();

        // Compute Sm = S @ m_norm (for gate computation)
        if (tid < N_STATE) {
            float sum = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < N_STATE; j++) {
                sum += S_shared[tid * N_STATE + j] * m_vec_shared[j];
            }
            Sm_shared[tid] = sum;
        }
        __syncthreads();

        // Compute s_retrieved = S @ k_norm
        if (tid < N_STATE) {
            float sum = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < N_STATE; j++) {
                sum += S_shared[tid * N_STATE + j] * k_shared[j];
            }
            s_retrieved[tid] = sum;
        }
        __syncthreads();

        // Compute gate and update S in one pass
        // G[i,j] = sigmoid(Sm[i] * k[j] + alpha * S[i,j]) + epsilon
        // S'[i,j] = G[i,j] * S[i,j] + s_delta[i] * k[j]
        for (int i = tid; i < n2; i += blockDim.x) {
            int row = i / N_STATE;
            int col = i % N_STATE;
            float S_ij = S_shared[i];
            float gate_logit = Sm_shared[row] * k_shared[col] + alpha * S_ij;
            float G_ij = 1.0f / (1.0f + expf(-gate_logit)) + epsilon;

            // Cache gate for backward
            gate_cache[t * B * n2 + b * n2 + i] = __float2bfloat16(G_ij);

            float s_delta_row = v_shared[row] - s_retrieved[row];
            S_shared[i] = G_ij * S_ij + s_delta_row * k_shared[col];
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

        // Output: Sq = S @ q, then self-gate
        if (tid < N_STATE) {
            float Sq = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < N_STATE; j++) {
                Sq += S_shared[tid * N_STATE + j] * q_shared[j];
            }
            Sq_cache[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(Sq);
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
// E82 Backward Kernel - Self-Gating Matrix System (BF16)
// ============================================================================

template<int N_STATE>
__global__ void E82SelfGateBackwardKernel_BF16(
    int T,
    int B,
    const __nv_bfloat16* __restrict__ kvqm_all,
    float alpha,
    float epsilon,
    const __nv_bfloat16* __restrict__ S_checkpoints,
    const __nv_bfloat16* __restrict__ Sq_cache,
    const __nv_bfloat16* __restrict__ gate_cache,
    const __nv_bfloat16* __restrict__ d_output,
    __nv_bfloat16* __restrict__ d_kvqm_all,
    float* __restrict__ d_alpha_accum,
    int checkpoint_interval
) {
    int b = blockIdx.x;
    if (b >= B) return;

    extern __shared__ float shared_mem[];
    // Layout for backward - need more space
    float* S = shared_mem;                                // [N_STATE * N_STATE]
    float* dS = S + N_STATE * N_STATE;                    // [N_STATE * N_STATE]
    float* k_raw = dS + N_STATE * N_STATE;                // [N_STATE]
    float* v_raw = k_raw + N_STATE;                       // [N_STATE]
    float* q_raw = v_raw + N_STATE;                       // [N_STATE]
    float* m_vec_raw = q_raw + N_STATE;                   // [N_STATE]
    float* k_norm = m_vec_raw + N_STATE;                  // [N_STATE]
    float* m_norm = k_norm + N_STATE;                     // [N_STATE]
    float* Sm = m_norm + N_STATE;                         // [N_STATE]
    float* s_retrieved = Sm + N_STATE;                    // [N_STATE]
    float* s_delta = s_retrieved + N_STATE;               // [N_STATE]
    float* d_k_raw = s_delta + N_STATE;                   // [N_STATE]
    float* d_v_raw = d_k_raw + N_STATE;                   // [N_STATE]
    float* d_q_raw = d_v_raw + N_STATE;                   // [N_STATE]
    float* d_m_raw = d_q_raw + N_STATE;                   // [N_STATE]
    float* d_Sq_shared = d_m_raw + N_STATE;               // [N_STATE]
    float* d_s_delta = d_Sq_shared + N_STATE;             // [N_STATE]
    float* d_k_norm = d_s_delta + N_STATE;                // [N_STATE]
    float* d_m_norm = d_k_norm + N_STATE;                 // [N_STATE]
    float* d_Sm = d_m_norm + N_STATE;                     // [N_STATE]
    float* d_alpha_local = d_Sm + N_STATE;                // [1]

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;
    const int STRIDE = 4 * N_STATE;

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
            __shared__ float k_norm_val_t, m_norm_val_t;
            for (int tt = t_start; tt <= t; tt++) {
                int col_idx = (tt * B + b) * STRIDE;
                if (tid < N_STATE) {
                    k_raw[tid] = __bfloat162float(kvqm_all[col_idx + tid]);
                    v_raw[tid] = __bfloat162float(kvqm_all[col_idx + N_STATE + tid]);
                    q_raw[tid] = __bfloat162float(kvqm_all[col_idx + 2 * N_STATE + tid]);
                    m_vec_raw[tid] = __bfloat162float(kvqm_all[col_idx + 3 * N_STATE + tid]);
                }
                __syncthreads();

                // Normalize k and m
                if (tid == 0) {
                    float k_sum = 0.0f, m_sum = 0.0f;
                    for (int i = 0; i < N_STATE; i++) {
                        k_sum += k_raw[i] * k_raw[i];
                        m_sum += m_vec_raw[i] * m_vec_raw[i];
                    }
                    k_norm_val_t = sqrtf(k_sum) + 1e-6f;
                    m_norm_val_t = sqrtf(m_sum) + 1e-6f;
                }
                __syncthreads();
                if (tid < N_STATE) {
                    k_norm[tid] = k_raw[tid] / k_norm_val_t;
                    m_norm[tid] = m_vec_raw[tid] / m_norm_val_t;
                }
                __syncthreads();

                // Compute Sm = S @ m_norm
                if (tid < N_STATE) {
                    float sum = 0.0f;
                    for (int j = 0; j < N_STATE; j++) {
                        sum += S[tid * N_STATE + j] * m_norm[j];
                    }
                    Sm[tid] = sum;
                }
                __syncthreads();

                // Compute s_retrieved = S @ k_norm, s_delta = v - s_retrieved
                if (tid < N_STATE) {
                    float sum = 0.0f;
                    for (int j = 0; j < N_STATE; j++) {
                        sum += S[tid * N_STATE + j] * k_norm[j];
                    }
                    s_retrieved[tid] = sum;
                    s_delta[tid] = v_raw[tid] - sum;
                }
                __syncthreads();

                // Update S if not at target step
                if (tt < t) {
                    for (int i = tid; i < n2; i += blockDim.x) {
                        int row = i / N_STATE;
                        int col = i % N_STATE;
                        float S_ij = S[i];
                        float gate_logit = Sm[row] * k_norm[col] + alpha * S_ij;
                        float G_ij = 1.0f / (1.0f + expf(-gate_logit)) + epsilon;
                        S[i] = G_ij * S_ij + s_delta[row] * k_norm[col];
                    }
                    __syncthreads();
                }
            }

            // Now S holds state at t-1; s_delta, Sm, k_norm, m_norm are for step t

            // === BACKWARD PASS FOR STEP t ===

            // Backward through output: Sq = S @ q, output = Sq * silu(Sq)
            if (tid < N_STATE) {
                float d_out = __bfloat162float(d_output[t * B * N_STATE + b * N_STATE + tid]);
                float Sq_t = __bfloat162float(Sq_cache[t * B * N_STATE + b * N_STATE + tid]);
                // d_out = d(Sq * silu(Sq)) = d(Sq * Sq * sigmoid(Sq))
                // d_Sq = d_out * (2 * Sq * sigmoid(Sq) + Sq^2 * sigmoid(Sq) * (1 - sigmoid(Sq)))
                float sigmoid_Sq = 1.0f / (1.0f + expf(-Sq_t));
                float silu_Sq = Sq_t * sigmoid_Sq;
                float d_Sq_val = d_out * (silu_Sq + Sq_t * sigmoid_Sq * (1.0f - sigmoid_Sq));
                d_Sq_shared[tid] = d_Sq_val;
            }
            __syncthreads();

            // d_q from Sq = S_new @ q (need S_new = S after update)
            // First compute S_new for d_q calculation
            // Actually, Sq was computed using S_new, so we need to use gate_cache
            if (tid < N_STATE) {
                // S_new[i,j] = G[i,j] * S[i,j] + s_delta[i] * k[j]
                // d_q[col] = sum_i(S_new[i, col] * d_Sq[i])
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    float G_ij = __bfloat162float(gate_cache[t * B * n2 + b * n2 + i * N_STATE + tid]);
                    float S_ij = S[i * N_STATE + tid];
                    float S_new_ij = G_ij * S_ij + s_delta[i] * k_norm[tid];
                    sum += S_new_ij * d_Sq_shared[i];
                }
                d_q_raw[tid] = sum;
            }
            __syncthreads();

            // dS_new += outer(d_Sq, q)
            // Then propagate through S update: S_new = G * S + outer(s_delta, k)
            // dG[i,j] = dS_new[i,j] * S[i,j]
            // dS[i,j] += dS_new[i,j] * G[i,j]
            // d_s_delta[i] = sum_j(dS_new[i,j] * k[j])
            // d_k[j] += sum_i(dS_new[i,j] * s_delta[i])

            // Initialize d_s_delta, d_k_norm, d_m_norm, d_Sm
            if (tid < N_STATE) {
                d_s_delta[tid] = 0.0f;
                d_k_norm[tid] = 0.0f;
                d_m_norm[tid] = 0.0f;
                d_Sm[tid] = 0.0f;
            }
            __syncthreads();

            // Local accumulator for d_alpha
            float d_alpha_thread = 0.0f;

            // Process dS_new and propagate gradients
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;

                float dS_new_ij = dS[i] + d_Sq_shared[row] * q_raw[col];
                float G_ij = __bfloat162float(gate_cache[t * B * n2 + b * n2 + i]);
                float S_ij = S[i];

                // dG = dS_new * S
                float dG_ij = dS_new_ij * S_ij;

                // G = sigmoid(gate_logit) + epsilon
                // gate_logit = Sm[row] * k[col] + alpha * S[row,col]
                // dgate_logit = dG * sigmoid * (1 - sigmoid)
                // Note: G = sigmoid + epsilon, so sigmoid = G - epsilon
                float sigmoid_val = G_ij - epsilon;
                float dgate_logit = dG_ij * sigmoid_val * (1.0f - sigmoid_val);

                // d_Sm[row] += dgate_logit * k[col]
                atomicAdd(&d_Sm[row], dgate_logit * k_norm[col]);

                // d_k_norm[col] += dgate_logit * Sm[row]
                atomicAdd(&d_k_norm[col], dgate_logit * Sm[row]);

                // d_alpha += dgate_logit * S[row,col]
                d_alpha_thread += dgate_logit * S_ij;

                // dS[i] = dS_new * G (propagate to previous timestep)
                dS[i] = dS_new_ij * G_ij;

                // d_s_delta[row] += dS_new * k[col]
                atomicAdd(&d_s_delta[row], dS_new_ij * k_norm[col]);

                // d_k_norm[col] += dS_new * s_delta[row]
                atomicAdd(&d_k_norm[col], dS_new_ij * s_delta[row]);
            }

            // Accumulate d_alpha
            atomicAdd(d_alpha_local, d_alpha_thread);
            __syncthreads();

            // d_v = d_s_delta (from s_delta = v - s_retrieved)
            // d_s_retrieved = -d_s_delta
            if (tid < N_STATE) {
                d_v_raw[tid] = d_s_delta[tid];
            }
            __syncthreads();

            // d_k_norm contribution from s_retrieved = S @ k_norm
            // d_k_norm[j] += sum_i(S[i,j] * (-d_s_delta[i]))
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    sum += S[i * N_STATE + tid] * (-d_s_delta[i]);
                }
                d_k_norm[tid] += sum;
            }
            __syncthreads();

            // d_m_norm contribution from Sm = S @ m_norm
            // d_m_norm[j] += sum_i(S[i,j] * d_Sm[i])
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    sum += S[i * N_STATE + tid] * d_Sm[i];
                }
                d_m_norm[tid] += sum;
            }
            __syncthreads();

            // dS contribution from Sm = S @ m_norm
            // dS[i,j] += d_Sm[i] * m_norm[j]
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                dS[i] += d_Sm[row] * m_norm[col];
            }
            __syncthreads();

            // dS contribution from s_retrieved = S @ k_norm
            // dS[i,j] += (-d_s_delta[i]) * k_norm[j]
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                dS[i] += (-d_s_delta[row]) * k_norm[col];
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

            // Convert d_m_norm to d_m_raw
            {
                __shared__ float m_dot_dm;
                if (tid == 0) {
                    m_dot_dm = 0.0f;
                    for (int i = 0; i < N_STATE; i++) {
                        m_dot_dm += m_vec_raw[i] * d_m_norm[i];
                    }
                }
                __syncthreads();
                if (tid < N_STATE) {
                    float norm = m_norm_val_t;
                    float norm3 = norm * norm * norm;
                    d_m_raw[tid] = d_m_norm[tid] / norm - m_vec_raw[tid] * m_dot_dm / norm3;
                }
                __syncthreads();
            }

            // Write gradients
            int col_idx_t = (t * B + b) * STRIDE;
            if (tid < N_STATE) {
                d_kvqm_all[col_idx_t + tid] = __float2bfloat16(d_k_raw[tid]);
                d_kvqm_all[col_idx_t + N_STATE + tid] = __float2bfloat16(d_v_raw[tid]);
                d_kvqm_all[col_idx_t + 2 * N_STATE + tid] = __float2bfloat16(d_q_raw[tid]);
                d_kvqm_all[col_idx_t + 3 * N_STATE + tid] = __float2bfloat16(d_m_raw[tid]);
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
// E82 Forward Kernel - FP32 version
// ============================================================================

template<int N_STATE>
__global__ void E82SelfGateForwardKernel_FP32(
    int T,
    int B,
    const float* __restrict__ kvqm_all,
    float alpha,
    float epsilon,
    float* __restrict__ S,
    float* __restrict__ output,
    float* __restrict__ S_checkpoints,
    float* __restrict__ Sq_cache,
    float* __restrict__ gate_cache,
    int checkpoint_interval
) {
    int b = blockIdx.x;
    if (b >= B) return;

    extern __shared__ float shared_mem[];
    float* S_shared = shared_mem;
    float* k_shared = S_shared + N_STATE * N_STATE;
    float* v_shared = k_shared + N_STATE;
    float* q_shared = v_shared + N_STATE;
    float* m_vec_shared = q_shared + N_STATE;
    float* Sm_shared = m_vec_shared + N_STATE;
    float* s_retrieved = Sm_shared + N_STATE;

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;
    const int STRIDE = 4 * N_STATE;

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
            k_shared[tid] = kvqm_all[col_idx + tid];
            v_shared[tid] = kvqm_all[col_idx + N_STATE + tid];
            q_shared[tid] = kvqm_all[col_idx + 2 * N_STATE + tid];
            m_vec_shared[tid] = kvqm_all[col_idx + 3 * N_STATE + tid];
        }
        __syncthreads();

        // Normalize k and m
        __shared__ float k_norm_val, m_norm_val;
        if (tid == 0) {
            float k_sum = 0.0f, m_sum = 0.0f;
            for (int i = 0; i < N_STATE; i++) {
                k_sum += k_shared[i] * k_shared[i];
                m_sum += m_vec_shared[i] * m_vec_shared[i];
            }
            k_norm_val = sqrtf(k_sum) + 1e-6f;
            m_norm_val = sqrtf(m_sum) + 1e-6f;
        }
        __syncthreads();
        if (tid < N_STATE) {
            k_shared[tid] /= k_norm_val;
            m_vec_shared[tid] /= m_norm_val;
        }
        __syncthreads();

        // Compute Sm = S @ m_norm
        if (tid < N_STATE) {
            float sum = 0.0f;
            for (int j = 0; j < N_STATE; j++) {
                sum += S_shared[tid * N_STATE + j] * m_vec_shared[j];
            }
            Sm_shared[tid] = sum;
        }
        __syncthreads();

        // Compute s_retrieved = S @ k_norm
        if (tid < N_STATE) {
            float sum = 0.0f;
            for (int j = 0; j < N_STATE; j++) {
                sum += S_shared[tid * N_STATE + j] * k_shared[j];
            }
            s_retrieved[tid] = sum;
        }
        __syncthreads();

        // Compute gate and update S
        for (int i = tid; i < n2; i += blockDim.x) {
            int row = i / N_STATE;
            int col = i % N_STATE;
            float S_ij = S_shared[i];
            float gate_logit = Sm_shared[row] * k_shared[col] + alpha * S_ij;
            float G_ij = 1.0f / (1.0f + expf(-gate_logit)) + epsilon;

            gate_cache[t * B * n2 + b * n2 + i] = G_ij;

            float s_delta_row = v_shared[row] - s_retrieved[row];
            S_shared[i] = G_ij * S_ij + s_delta_row * k_shared[col];
        }
        __syncthreads();

        // Save checkpoint if at boundary
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
// E82 Backward Kernel - FP32 version
// ============================================================================

template<int N_STATE>
__global__ void E82SelfGateBackwardKernel_FP32(
    int T,
    int B,
    const float* __restrict__ kvqm_all,
    float alpha,
    float epsilon,
    const float* __restrict__ S_checkpoints,
    const float* __restrict__ Sq_cache,
    const float* __restrict__ gate_cache,
    const float* __restrict__ d_output,
    float* __restrict__ d_kvqm_all,
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
    float* m_vec_raw = q_raw + N_STATE;
    float* k_norm = m_vec_raw + N_STATE;
    float* m_norm = k_norm + N_STATE;
    float* Sm = m_norm + N_STATE;
    float* s_retrieved = Sm + N_STATE;
    float* s_delta = s_retrieved + N_STATE;
    float* d_k_raw = s_delta + N_STATE;
    float* d_v_raw = d_k_raw + N_STATE;
    float* d_q_raw = d_v_raw + N_STATE;
    float* d_m_raw = d_q_raw + N_STATE;
    float* d_Sq_shared = d_m_raw + N_STATE;
    float* d_s_delta = d_Sq_shared + N_STATE;
    float* d_k_norm = d_s_delta + N_STATE;
    float* d_m_norm = d_k_norm + N_STATE;
    float* d_Sm = d_m_norm + N_STATE;
    float* d_alpha_local = d_Sm + N_STATE;

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;
    const int STRIDE = 4 * N_STATE;

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
                S[i] = S_checkpoints[seg * B * n2 + b * n2 + i];
            }
            __syncthreads();

            // Recompute forward to step t
            __shared__ float k_norm_val_t, m_norm_val_t;
            for (int tt = t_start; tt <= t; tt++) {
                int col_idx = (tt * B + b) * STRIDE;
                if (tid < N_STATE) {
                    k_raw[tid] = kvqm_all[col_idx + tid];
                    v_raw[tid] = kvqm_all[col_idx + N_STATE + tid];
                    q_raw[tid] = kvqm_all[col_idx + 2 * N_STATE + tid];
                    m_vec_raw[tid] = kvqm_all[col_idx + 3 * N_STATE + tid];
                }
                __syncthreads();

                if (tid == 0) {
                    float k_sum = 0.0f, m_sum = 0.0f;
                    for (int i = 0; i < N_STATE; i++) {
                        k_sum += k_raw[i] * k_raw[i];
                        m_sum += m_vec_raw[i] * m_vec_raw[i];
                    }
                    k_norm_val_t = sqrtf(k_sum) + 1e-6f;
                    m_norm_val_t = sqrtf(m_sum) + 1e-6f;
                }
                __syncthreads();
                if (tid < N_STATE) {
                    k_norm[tid] = k_raw[tid] / k_norm_val_t;
                    m_norm[tid] = m_vec_raw[tid] / m_norm_val_t;
                }
                __syncthreads();

                // Compute Sm = S @ m_norm
                if (tid < N_STATE) {
                    float sum = 0.0f;
                    for (int j = 0; j < N_STATE; j++) {
                        sum += S[tid * N_STATE + j] * m_norm[j];
                    }
                    Sm[tid] = sum;
                }
                __syncthreads();

                // Compute s_retrieved and s_delta
                if (tid < N_STATE) {
                    float sum = 0.0f;
                    for (int j = 0; j < N_STATE; j++) {
                        sum += S[tid * N_STATE + j] * k_norm[j];
                    }
                    s_retrieved[tid] = sum;
                    s_delta[tid] = v_raw[tid] - sum;
                }
                __syncthreads();

                if (tt < t) {
                    for (int i = tid; i < n2; i += blockDim.x) {
                        int row = i / N_STATE;
                        int col = i % N_STATE;
                        float S_ij = S[i];
                        float gate_logit = Sm[row] * k_norm[col] + alpha * S_ij;
                        float G_ij = 1.0f / (1.0f + expf(-gate_logit)) + epsilon;
                        S[i] = G_ij * S_ij + s_delta[row] * k_norm[col];
                    }
                    __syncthreads();
                }
            }

            // === BACKWARD PASS FOR STEP t ===

            // Backward through output
            if (tid < N_STATE) {
                float d_out = d_output[t * B * N_STATE + b * N_STATE + tid];
                float Sq_t = Sq_cache[t * B * N_STATE + b * N_STATE + tid];
                float sigmoid_Sq = 1.0f / (1.0f + expf(-Sq_t));
                float silu_Sq = Sq_t * sigmoid_Sq;
                float d_Sq_val = d_out * (silu_Sq + Sq_t * sigmoid_Sq * (1.0f - sigmoid_Sq));
                d_Sq_shared[tid] = d_Sq_val;
            }
            __syncthreads();

            // d_q from Sq = S_new @ q
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    float G_ij = gate_cache[t * B * n2 + b * n2 + i * N_STATE + tid];
                    float S_ij = S[i * N_STATE + tid];
                    float S_new_ij = G_ij * S_ij + s_delta[i] * k_norm[tid];
                    sum += S_new_ij * d_Sq_shared[i];
                }
                d_q_raw[tid] = sum;
            }
            __syncthreads();

            // Initialize gradient accumulators
            if (tid < N_STATE) {
                d_s_delta[tid] = 0.0f;
                d_k_norm[tid] = 0.0f;
                d_m_norm[tid] = 0.0f;
                d_Sm[tid] = 0.0f;
            }
            __syncthreads();

            float d_alpha_thread = 0.0f;

            // Process dS_new and propagate gradients
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;

                float dS_new_ij = dS[i] + d_Sq_shared[row] * q_raw[col];
                float G_ij = gate_cache[t * B * n2 + b * n2 + i];
                float S_ij = S[i];

                float dG_ij = dS_new_ij * S_ij;
                float sigmoid_val = G_ij - epsilon;
                float dgate_logit = dG_ij * sigmoid_val * (1.0f - sigmoid_val);

                atomicAdd(&d_Sm[row], dgate_logit * k_norm[col]);
                atomicAdd(&d_k_norm[col], dgate_logit * Sm[row]);
                d_alpha_thread += dgate_logit * S_ij;

                dS[i] = dS_new_ij * G_ij;
                atomicAdd(&d_s_delta[row], dS_new_ij * k_norm[col]);
                atomicAdd(&d_k_norm[col], dS_new_ij * s_delta[row]);
            }

            atomicAdd(d_alpha_local, d_alpha_thread);
            __syncthreads();

            if (tid < N_STATE) {
                d_v_raw[tid] = d_s_delta[tid];
            }
            __syncthreads();

            // d_k_norm contribution from s_retrieved
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    sum += S[i * N_STATE + tid] * (-d_s_delta[i]);
                }
                d_k_norm[tid] += sum;
            }
            __syncthreads();

            // d_m_norm contribution from Sm
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    sum += S[i * N_STATE + tid] * d_Sm[i];
                }
                d_m_norm[tid] += sum;
            }
            __syncthreads();

            // dS contributions from Sm and s_retrieved
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                dS[i] += d_Sm[row] * m_norm[col] + (-d_s_delta[row]) * k_norm[col];
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

            // Convert d_m_norm to d_m_raw
            {
                __shared__ float m_dot_dm;
                if (tid == 0) {
                    m_dot_dm = 0.0f;
                    for (int i = 0; i < N_STATE; i++) {
                        m_dot_dm += m_vec_raw[i] * d_m_norm[i];
                    }
                }
                __syncthreads();
                if (tid < N_STATE) {
                    float norm = m_norm_val_t;
                    float norm3 = norm * norm * norm;
                    d_m_raw[tid] = d_m_norm[tid] / norm - m_vec_raw[tid] * m_dot_dm / norm3;
                }
                __syncthreads();
            }

            // Write gradients
            int col_idx_t = (t * B + b) * STRIDE;
            if (tid < N_STATE) {
                d_kvqm_all[col_idx_t + tid] = d_k_raw[tid];
                d_kvqm_all[col_idx_t + N_STATE + tid] = d_v_raw[tid];
                d_kvqm_all[col_idx_t + 2 * N_STATE + tid] = d_q_raw[tid];
                d_kvqm_all[col_idx_t + 3 * N_STATE + tid] = d_m_raw[tid];
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
// Template instantiations
// ============================================================================

#define INSTANTIATE_E82_KERNELS_BF16(N) \
    template __global__ void E82SelfGateForwardKernel_BF16<N>( \
        int, int, const __nv_bfloat16*, float, float, \
        __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int); \
    template __global__ void E82SelfGateBackwardKernel_BF16<N>( \
        int, int, const __nv_bfloat16*, float, float, \
        const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, \
        __nv_bfloat16*, float*, int);

#define INSTANTIATE_E82_KERNELS_FP32(N) \
    template __global__ void E82SelfGateForwardKernel_FP32<N>( \
        int, int, const float*, float, float, \
        float*, float*, float*, float*, float*, int); \
    template __global__ void E82SelfGateBackwardKernel_FP32<N>( \
        int, int, const float*, float, float, \
        const float*, const float*, const float*, const float*, \
        float*, float*, int);

INSTANTIATE_E82_KERNELS_BF16(8)
INSTANTIATE_E82_KERNELS_BF16(16)
INSTANTIATE_E82_KERNELS_BF16(32)
INSTANTIATE_E82_KERNELS_BF16(48)
INSTANTIATE_E82_KERNELS_BF16(64)
INSTANTIATE_E82_KERNELS_BF16(96)
INSTANTIATE_E82_KERNELS_BF16(128)

INSTANTIATE_E82_KERNELS_FP32(8)
INSTANTIATE_E82_KERNELS_FP32(16)
INSTANTIATE_E82_KERNELS_FP32(32)
INSTANTIATE_E82_KERNELS_FP32(48)
INSTANTIATE_E82_KERNELS_FP32(64)
INSTANTIATE_E82_KERNELS_FP32(96)
INSTANTIATE_E82_KERNELS_FP32(128)

// ============================================================================
// Dispatcher functions
// ============================================================================

void dispatch_e82_self_gate_forward(
    int T, int B, int n_state,
    const __nv_bfloat16* kvqm_all,
    float alpha, float epsilon,
    __nv_bfloat16* S, __nv_bfloat16* output,
    __nv_bfloat16* S_checkpoints, __nv_bfloat16* Sq_cache,
    __nv_bfloat16* gate_cache,
    int checkpoint_interval, cudaStream_t stream
) {
    // Shared memory: n^2 (S) + 7*n (vectors)
    int shared_size = (n_state * n_state + 7 * n_state) * sizeof(float);

    #define DISPATCH_E82_FWD(N) \
        E82SelfGateForwardKernel_BF16<N><<<B, 256, shared_size, stream>>>( \
            T, B, kvqm_all, alpha, epsilon, \
            S, output, S_checkpoints, Sq_cache, gate_cache, checkpoint_interval);

    #define DISPATCH_E82_FWD_EXT(N) \
        { \
            cudaError_t attr_err = cudaFuncSetAttribute( \
                E82SelfGateForwardKernel_BF16<N>, \
                cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size); \
            if (attr_err != cudaSuccess) { \
                fprintf(stderr, "E82 Forward: n_state=%d cudaFuncSetAttribute failed: %s\n", \
                        N, cudaGetErrorString(attr_err)); \
            } else { \
                DISPATCH_E82_FWD(N); \
            } \
        }

    switch (n_state) {
        case 8: DISPATCH_E82_FWD(8); break;
        case 16: DISPATCH_E82_FWD(16); break;
        case 32: DISPATCH_E82_FWD(32); break;
        case 48: DISPATCH_E82_FWD(48); break;
        case 64: DISPATCH_E82_FWD(64); break;
        case 96: DISPATCH_E82_FWD_EXT(96); break;
        case 128: DISPATCH_E82_FWD_EXT(128); break;
        default:
            fprintf(stderr, "E82: Unsupported n_state=%d (use 8, 16, 32, 48, 64, 96, or 128)\n", n_state);
    }
    #undef DISPATCH_E82_FWD
    #undef DISPATCH_E82_FWD_EXT
}

void dispatch_e82_self_gate_backward(
    int T, int B, int n_state,
    const __nv_bfloat16* kvqm_all,
    float alpha, float epsilon,
    const __nv_bfloat16* S_checkpoints,
    const __nv_bfloat16* Sq_cache, const __nv_bfloat16* gate_cache,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* d_kvqm_all,
    float* d_alpha_accum,
    int checkpoint_interval, cudaStream_t stream
) {
    // Shared memory for backward: 2*n^2 + 21*n + 1
    int shared_size = (2 * n_state * n_state + 21 * n_state + 1) * sizeof(float);

    #define DISPATCH_E82_BWD(N) \
        E82SelfGateBackwardKernel_BF16<N><<<B, 256, shared_size, stream>>>( \
            T, B, kvqm_all, alpha, epsilon, \
            S_checkpoints, Sq_cache, gate_cache, d_output, \
            d_kvqm_all, d_alpha_accum, checkpoint_interval);

    #define DISPATCH_E82_BWD_EXT(N) \
        { \
            cudaError_t attr_err = cudaFuncSetAttribute( \
                E82SelfGateBackwardKernel_BF16<N>, \
                cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size); \
            if (attr_err != cudaSuccess) { \
                fprintf(stderr, "E82 Backward: n_state=%d requires %d KB shared memory but GPU limit exceeded\n", \
                        N, shared_size / 1024); \
            } else { \
                DISPATCH_E82_BWD(N); \
            } \
        }

    switch (n_state) {
        case 8: DISPATCH_E82_BWD(8); break;
        case 16: DISPATCH_E82_BWD(16); break;
        case 32: DISPATCH_E82_BWD(32); break;
        case 48: DISPATCH_E82_BWD(48); break;
        case 64: DISPATCH_E82_BWD(64); break;
        case 96: DISPATCH_E82_BWD_EXT(96); break;
        case 128: DISPATCH_E82_BWD_EXT(128); break;
        default:
            fprintf(stderr, "E82: Unsupported n_state=%d (use 8, 16, 32, 48, 64, 96, or 128)\n", n_state);
    }
    #undef DISPATCH_E82_BWD
    #undef DISPATCH_E82_BWD_EXT
}

void dispatch_e82_self_gate_forward_fp32(
    int T, int B, int n_state,
    const float* kvqm_all,
    float alpha, float epsilon,
    float* S, float* output,
    float* S_checkpoints, float* Sq_cache,
    float* gate_cache,
    int checkpoint_interval, cudaStream_t stream
) {
    int shared_size = (n_state * n_state + 7 * n_state) * sizeof(float);

    #define DISPATCH_E82_FWD_FP32(N) \
        E82SelfGateForwardKernel_FP32<N><<<B, 256, shared_size, stream>>>( \
            T, B, kvqm_all, alpha, epsilon, \
            S, output, S_checkpoints, Sq_cache, gate_cache, checkpoint_interval);

    #define DISPATCH_E82_FWD_FP32_EXT(N) \
        { \
            cudaError_t attr_err = cudaFuncSetAttribute( \
                E82SelfGateForwardKernel_FP32<N>, \
                cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size); \
            if (attr_err != cudaSuccess) { \
                fprintf(stderr, "E82 FP32 Forward: n_state=%d cudaFuncSetAttribute failed: %s\n", \
                        N, cudaGetErrorString(attr_err)); \
            } else { \
                DISPATCH_E82_FWD_FP32(N); \
            } \
        }

    switch (n_state) {
        case 8: DISPATCH_E82_FWD_FP32(8); break;
        case 16: DISPATCH_E82_FWD_FP32(16); break;
        case 32: DISPATCH_E82_FWD_FP32(32); break;
        case 48: DISPATCH_E82_FWD_FP32(48); break;
        case 64: DISPATCH_E82_FWD_FP32(64); break;
        case 96: DISPATCH_E82_FWD_FP32_EXT(96); break;
        case 128: DISPATCH_E82_FWD_FP32_EXT(128); break;
        default:
            fprintf(stderr, "E82: Unsupported n_state=%d (use 8, 16, 32, 48, 64, 96, or 128)\n", n_state);
    }
    #undef DISPATCH_E82_FWD_FP32
    #undef DISPATCH_E82_FWD_FP32_EXT
}

void dispatch_e82_self_gate_backward_fp32(
    int T, int B, int n_state,
    const float* kvqm_all,
    float alpha, float epsilon,
    const float* S_checkpoints,
    const float* Sq_cache, const float* gate_cache,
    const float* d_output,
    float* d_kvqm_all,
    float* d_alpha_accum,
    int checkpoint_interval, cudaStream_t stream
) {
    int shared_size = (2 * n_state * n_state + 21 * n_state + 1) * sizeof(float);

    #define DISPATCH_E82_BWD_FP32(N) \
        E82SelfGateBackwardKernel_FP32<N><<<B, 256, shared_size, stream>>>( \
            T, B, kvqm_all, alpha, epsilon, \
            S_checkpoints, Sq_cache, gate_cache, d_output, \
            d_kvqm_all, d_alpha_accum, checkpoint_interval);

    #define DISPATCH_E82_BWD_FP32_EXT(N) \
        { \
            cudaError_t attr_err = cudaFuncSetAttribute( \
                E82SelfGateBackwardKernel_FP32<N>, \
                cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size); \
            if (attr_err != cudaSuccess) { \
                fprintf(stderr, "E82 FP32 Backward: n_state=%d requires %d KB shared memory but GPU limit exceeded\n", \
                        N, shared_size / 1024); \
            } else { \
                DISPATCH_E82_BWD_FP32(N); \
            } \
        }

    switch (n_state) {
        case 8: DISPATCH_E82_BWD_FP32(8); break;
        case 16: DISPATCH_E82_BWD_FP32(16); break;
        case 32: DISPATCH_E82_BWD_FP32(32); break;
        case 48: DISPATCH_E82_BWD_FP32(48); break;
        case 64: DISPATCH_E82_BWD_FP32(64); break;
        case 96: DISPATCH_E82_BWD_FP32_EXT(96); break;
        case 128: DISPATCH_E82_BWD_FP32_EXT(128); break;
        default:
            fprintf(stderr, "E82: Unsupported n_state=%d (use 8, 16, 32, 48, 64, 96, or 128)\n", n_state);
    }
    #undef DISPATCH_E82_BWD_FP32
    #undef DISPATCH_E82_BWD_FP32_EXT
}

// ============================================================================
// E82SelfGateForward Implementation (wrapper class for Python bindings)
// ============================================================================

template<typename DataT>
E82SelfGateForward<DataT>::E82SelfGateForward(
    bool training, int batch_size, int n_state, int dim,
    const cublasHandle_t& blas_handle, const cudaStream_t& stream)
    : training_(training), batch_size_(batch_size), n_state_(n_state),
      dim_(dim), blas_handle_(blas_handle), stream_(stream) {}

template<typename DataT>
void E82SelfGateForward<DataT>::Run(
    int steps,
    const DataT* W_kvqm,
    float alpha, float epsilon,
    const DataT* x,
    DataT* S,
    DataT* output,
    DataT* kvqm_cache,
    DataT* S_checkpoints,
    DataT* Sq_cache,
    DataT* gate_cache
) {
    int T = steps;
    int B = batch_size_;
    int n = n_state_;
    int d = dim_;
    int checkpoint_interval = E82_CHECKPOINT_INTERVAL;

    const float alpha_one = 1.0f, beta_zero = 0.0f;

    // FUSED projection: kvqm = W_kvqm @ x
    cudaDataType_t data_type = std::is_same<DataT, __nv_bfloat16>::value ? CUDA_R_16BF : CUDA_R_32F;
    cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                 4 * n, T * B, d,
                 &alpha_one,
                 W_kvqm, data_type, d,
                 x, data_type, d,
                 &beta_zero,
                 kvqm_cache, data_type, 4 * n,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    if constexpr (std::is_same<DataT, __nv_bfloat16>::value) {
        dispatch_e82_self_gate_forward(T, B, n, kvqm_cache, alpha, epsilon,
                                       S, output, S_checkpoints, Sq_cache,
                                       gate_cache, checkpoint_interval, stream_);
    } else {
        dispatch_e82_self_gate_forward_fp32(T, B, n,
                                            reinterpret_cast<const float*>(kvqm_cache),
                                            alpha, epsilon,
                                            reinterpret_cast<float*>(S),
                                            reinterpret_cast<float*>(output),
                                            reinterpret_cast<float*>(S_checkpoints),
                                            reinterpret_cast<float*>(Sq_cache),
                                            reinterpret_cast<float*>(gate_cache),
                                            checkpoint_interval, stream_);
    }
}

template<typename DataT>
E82SelfGateBackward<DataT>::E82SelfGateBackward(
    int batch_size, int n_state, int dim,
    const cublasHandle_t& blas_handle, const cudaStream_t& stream)
    : batch_size_(batch_size), n_state_(n_state), dim_(dim),
      blas_handle_(blas_handle), stream_(stream) {}

template<typename DataT>
void E82SelfGateBackward<DataT>::Run(
    int steps,
    const DataT* W_kvqm,
    float alpha, float epsilon,
    const DataT* x,
    const DataT* kvqm_cache,
    const DataT* S_checkpoints,
    const DataT* Sq_cache,
    const DataT* gate_cache,
    const DataT* d_output,
    DataT* d_x,
    DataT* d_W_kvqm,
    DataT* d_kvqm_cache,
    float* d_alpha_accum
) {
    int T = steps;
    int B = batch_size_;
    int n = n_state_;
    int d = dim_;
    int checkpoint_interval = E82_CHECKPOINT_INTERVAL;

    const float alpha_one = 1.0f, beta_zero = 0.0f;

    // Zero accumulators
    cudaMemsetAsync(d_alpha_accum, 0, sizeof(float), stream_);

    if constexpr (std::is_same<DataT, __nv_bfloat16>::value) {
        dispatch_e82_self_gate_backward(T, B, n, kvqm_cache, alpha, epsilon,
                                        S_checkpoints, Sq_cache, gate_cache, d_output,
                                        d_kvqm_cache, d_alpha_accum,
                                        checkpoint_interval, stream_);
    } else {
        dispatch_e82_self_gate_backward_fp32(T, B, n,
                                             reinterpret_cast<const float*>(kvqm_cache),
                                             alpha, epsilon,
                                             reinterpret_cast<const float*>(S_checkpoints),
                                             reinterpret_cast<const float*>(Sq_cache),
                                             reinterpret_cast<const float*>(gate_cache),
                                             reinterpret_cast<const float*>(d_output),
                                             reinterpret_cast<float*>(d_kvqm_cache),
                                             d_alpha_accum,
                                             checkpoint_interval, stream_);
    }

    cudaDataType_t data_type = std::is_same<DataT, __nv_bfloat16>::value ? CUDA_R_16BF : CUDA_R_32F;

    // d_x = W_kvqm @ d_kvqm_cache
    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                 d, T * B, 4 * n,
                 &alpha_one,
                 W_kvqm, data_type, d,
                 d_kvqm_cache, data_type, 4 * n,
                 &beta_zero,
                 d_x, data_type, d,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // d_W_kvqm = d_kvqm_cache @ x^T
    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
                 4 * n, d, T * B,
                 &alpha_one,
                 d_kvqm_cache, data_type, 4 * n,
                 x, data_type, d,
                 &beta_zero,
                 d_W_kvqm, data_type, 4 * n,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

// Explicit template instantiations
template class E82SelfGateForward<__nv_bfloat16>;
template class E82SelfGateForward<float>;
template class E82SelfGateBackward<__nv_bfloat16>;
template class E82SelfGateBackward<float>;

}  // namespace elman
