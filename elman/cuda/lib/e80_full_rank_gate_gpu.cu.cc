/**
 * E80 Full-Rank Mutual Gating CUDA Kernel
 *
 * Extends E79 with full n×n gate matrices instead of rank-1 outer products.
 *
 * Two coupled matrix states that mutually control each other's evolution:
 * - S [n_state x n_state]: Content memory - stores key-value associations
 * - M [n_state x n_state]: Modulation memory - controls how S updates
 *
 * Key difference from E79: The gate is a full n×n matrix, not a rank-1 outer product.
 *
 * Architecture:
 *   FUSED projection: kvqm = W_kvqm @ x  (k, v, q, m)
 *
 *   # M provides full-rank gate for S
 *   G_S = sigmoid(M + outer(M @ k_norm, k_norm) + B_S)  # Full n×n gate
 *   S = G_S ⊙ S + outer(v - S @ k_norm, k_norm)
 *
 *   # S provides full-rank gate for M
 *   G_M = sigmoid(S + outer(S @ m_norm, m_norm) + B_M)  # Full n×n gate
 *   M = G_M ⊙ M + outer(s_delta - M @ m_norm, m_norm)
 *
 *   # Output
 *   Sq = S @ q
 *   output = Sq * silu(Sq)
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include <vector>
#include "hasty/elman_ladder.h"

#define E80_CHECKPOINT_INTERVAL 16

namespace elman {

// ============================================================================
// E80 Forward Kernel - Full-Rank Mutual Gating System
// ============================================================================

template<int N_STATE>
__global__ void E80FullRankGateForwardKernel_BF16(
    int T,
    int B,
    const __nv_bfloat16* __restrict__ kvqm_all,   // [4*N_STATE, T*B] column-major
    const __nv_bfloat16* __restrict__ B_S,        // [N_STATE, N_STATE] S gate bias matrix
    const __nv_bfloat16* __restrict__ B_M,        // [N_STATE, N_STATE] M gate bias matrix
    __nv_bfloat16* __restrict__ S,                // [B, N_STATE, N_STATE]
    __nv_bfloat16* __restrict__ M,                // [B, N_STATE, N_STATE]
    __nv_bfloat16* __restrict__ output,           // [T, B, N_STATE]
    __nv_bfloat16* __restrict__ S_checkpoints,    // [num_checkpoints, B, N_STATE, N_STATE]
    __nv_bfloat16* __restrict__ M_checkpoints,    // [num_checkpoints, B, N_STATE, N_STATE]
    __nv_bfloat16* __restrict__ Sq_cache,         // [T, B, N_STATE]
    __nv_bfloat16* __restrict__ G_S_cache,        // [T, B, N_STATE, N_STATE]
    __nv_bfloat16* __restrict__ G_M_cache,        // [T, B, N_STATE, N_STATE]
    int checkpoint_interval
) {
    int b = blockIdx.x;
    if (b >= B) return;

    // Shared memory layout - need space for two matrices plus vectors
    extern __shared__ float shared_mem[];
    float* S_shared = shared_mem;                         // [N_STATE * N_STATE]
    float* M_shared = S_shared + N_STATE * N_STATE;       // [N_STATE * N_STATE]
    float* k_shared = M_shared + N_STATE * N_STATE;       // [N_STATE]
    float* v_shared = k_shared + N_STATE;                 // [N_STATE]
    float* q_shared = v_shared + N_STATE;                 // [N_STATE]
    float* m_vec_shared = q_shared + N_STATE;             // [N_STATE]
    float* M_k = m_vec_shared + N_STATE;                  // [N_STATE] - M @ k_norm
    float* S_m = M_k + N_STATE;                           // [N_STATE] - S @ m_norm
    float* s_retrieved = S_m + N_STATE;                   // [N_STATE]
    float* m_retrieved = s_retrieved + N_STATE;           // [N_STATE]
    float* B_S_shared = m_retrieved + N_STATE;            // [N_STATE * N_STATE]
    float* B_M_shared = B_S_shared + N_STATE * N_STATE;   // [N_STATE * N_STATE]

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;
    const int STRIDE = 4 * N_STATE;

    // Load gate bias matrices
    for (int i = tid; i < n2; i += blockDim.x) {
        B_S_shared[i] = __bfloat162float(B_S[i]);
        B_M_shared[i] = __bfloat162float(B_M[i]);
    }
    __syncthreads();

    // Load initial states
    for (int i = tid; i < n2; i += blockDim.x) {
        S_shared[i] = __bfloat162float(S[b * n2 + i]);
        M_shared[i] = __bfloat162float(M[b * n2 + i]);
    }
    __syncthreads();

    // Save initial checkpoints (index 0)
    for (int i = tid; i < n2; i += blockDim.x) {
        S_checkpoints[b * n2 + i] = __float2bfloat16(S_shared[i]);
        M_checkpoints[b * n2 + i] = __float2bfloat16(M_shared[i]);
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

        // --- S update (M-controlled full-rank gating) ---
        // M_k = M @ k_norm
        if (tid < N_STATE) {
            float sum = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < N_STATE; j++) {
                sum += M_shared[tid * N_STATE + j] * k_shared[j];
            }
            M_k[tid] = sum;
        }
        __syncthreads();

        // s_retrieved = S @ k_norm
        if (tid < N_STATE) {
            float sum = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < N_STATE; j++) {
                sum += S_shared[tid * N_STATE + j] * k_shared[j];
            }
            s_retrieved[tid] = sum;
        }
        __syncthreads();

        // Compute G_S and update S: G_S = sigmoid(M + outer(M_k, k) + B_S)
        // S = G_S * S + outer(s_delta, k)
        for (int i = tid; i < n2; i += blockDim.x) {
            int row = i / N_STATE;
            int col = i % N_STATE;

            // Full-rank gate: G_S[i,j] = sigmoid(M[i,j] + M_k[i] * k[j] + B_S[i,j])
            float gate_logit = M_shared[i] + M_k[row] * k_shared[col] + B_S_shared[i];
            float G_S_val = 1.0f / (1.0f + expf(-gate_logit));

            // Cache gate for backward
            G_S_cache[t * B * n2 + b * n2 + i] = __float2bfloat16(G_S_val);

            // Delta rule update
            float s_delta_row = v_shared[row] - s_retrieved[row];
            S_shared[i] = G_S_val * S_shared[i] + s_delta_row * k_shared[col];
        }
        __syncthreads();

        // --- M update (S-controlled full-rank gating) ---
        // S_m = S @ m_norm (after S update!)
        if (tid < N_STATE) {
            float sum = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < N_STATE; j++) {
                sum += S_shared[tid * N_STATE + j] * m_vec_shared[j];
            }
            S_m[tid] = sum;
        }
        __syncthreads();

        // m_retrieved = M @ m_norm
        if (tid < N_STATE) {
            float sum = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < N_STATE; j++) {
                sum += M_shared[tid * N_STATE + j] * m_vec_shared[j];
            }
            m_retrieved[tid] = sum;
        }
        __syncthreads();

        // Compute G_M and update M: G_M = sigmoid(S + outer(S_m, m) + B_M)
        // M = G_M * M + outer(m_delta, m)
        // m_delta = s_delta - m_retrieved
        for (int i = tid; i < n2; i += blockDim.x) {
            int row = i / N_STATE;
            int col = i % N_STATE;

            // Full-rank gate: G_M[i,j] = sigmoid(S[i,j] + S_m[i] * m[j] + B_M[i,j])
            float gate_logit = S_shared[i] + S_m[row] * m_vec_shared[col] + B_M_shared[i];
            float G_M_val = 1.0f / (1.0f + expf(-gate_logit));

            // Cache gate for backward
            G_M_cache[t * B * n2 + b * n2 + i] = __float2bfloat16(G_M_val);

            // m_delta = s_delta - m_retrieved (where s_delta was computed before S update)
            // We need s_delta from before S update but after S update we lost it...
            // Let's recompute: s_delta[row] = v[row] - (original s_retrieved[row])
            // But s_retrieved was computed BEFORE S update, so we can use it
            float s_delta_row = v_shared[row] - s_retrieved[row];
            float m_delta_row = s_delta_row - m_retrieved[row];

            M_shared[i] = G_M_val * M_shared[i] + m_delta_row * m_vec_shared[col];
        }
        __syncthreads();

        // Save checkpoints if at boundary
        if ((t + 1) % checkpoint_interval == 0) {
            int cp_idx = (t + 1) / checkpoint_interval;
            for (int i = tid; i < n2; i += blockDim.x) {
                S_checkpoints[cp_idx * B * n2 + b * n2 + i] = __float2bfloat16(S_shared[i]);
                M_checkpoints[cp_idx * B * n2 + b * n2 + i] = __float2bfloat16(M_shared[i]);
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

    // Write final states
    for (int i = tid; i < n2; i += blockDim.x) {
        S[b * n2 + i] = __float2bfloat16(S_shared[i]);
        M[b * n2 + i] = __float2bfloat16(M_shared[i]);
    }
}

// ============================================================================
// E80 Backward Kernel - Full-Rank Mutual Gating System
// ============================================================================

template<int N_STATE>
__global__ void E80FullRankGateBackwardKernel_BF16(
    int T,
    int B,
    const __nv_bfloat16* __restrict__ kvqm_all,
    const __nv_bfloat16* __restrict__ B_S,
    const __nv_bfloat16* __restrict__ B_M,
    const __nv_bfloat16* __restrict__ G_S_cache,
    const __nv_bfloat16* __restrict__ G_M_cache,
    const __nv_bfloat16* __restrict__ S_checkpoints,
    const __nv_bfloat16* __restrict__ M_checkpoints,
    const __nv_bfloat16* __restrict__ Sq_cache,
    const __nv_bfloat16* __restrict__ d_output,
    __nv_bfloat16* __restrict__ d_kvqm_all,
    float* __restrict__ d_B_S_accum,
    float* __restrict__ d_B_M_accum,
    int checkpoint_interval
) {
    int b = blockIdx.x;
    if (b >= B) return;

    extern __shared__ float shared_mem[];
    // Layout for backward - need more space
    float* S = shared_mem;                                // [N_STATE * N_STATE]
    float* M = S + N_STATE * N_STATE;                     // [N_STATE * N_STATE]
    float* dS = M + N_STATE * N_STATE;                    // [N_STATE * N_STATE]
    float* dM = dS + N_STATE * N_STATE;                   // [N_STATE * N_STATE]
    float* k_raw = dM + N_STATE * N_STATE;                // [N_STATE]
    float* v_raw = k_raw + N_STATE;                       // [N_STATE]
    float* q_raw = v_raw + N_STATE;                       // [N_STATE]
    float* m_vec_raw = q_raw + N_STATE;                   // [N_STATE]
    float* k_norm = m_vec_raw + N_STATE;                  // [N_STATE]
    float* m_norm = k_norm + N_STATE;                     // [N_STATE]
    float* M_k = m_norm + N_STATE;                        // [N_STATE]
    float* S_m = M_k + N_STATE;                           // [N_STATE]
    float* s_retrieved = S_m + N_STATE;                   // [N_STATE]
    float* m_retrieved = s_retrieved + N_STATE;           // [N_STATE]
    float* s_delta = m_retrieved + N_STATE;               // [N_STATE]
    float* m_delta = s_delta + N_STATE;                   // [N_STATE]
    float* d_k_raw = m_delta + N_STATE;                   // [N_STATE]
    float* d_v_raw = d_k_raw + N_STATE;                   // [N_STATE]
    float* d_q_raw = d_v_raw + N_STATE;                   // [N_STATE]
    float* d_m_raw = d_q_raw + N_STATE;                   // [N_STATE]
    float* d_Sq_shared = d_m_raw + N_STATE;               // [N_STATE]
    float* d_s_delta = d_Sq_shared + N_STATE;             // [N_STATE]
    float* d_m_delta = d_s_delta + N_STATE;               // [N_STATE]
    float* d_k_norm = d_m_delta + N_STATE;                // [N_STATE]
    float* d_m_norm = d_k_norm + N_STATE;                 // [N_STATE]
    float* d_M_k = d_m_norm + N_STATE;                    // [N_STATE]
    float* d_S_m = d_M_k + N_STATE;                       // [N_STATE]
    float* G_S_local = d_S_m + N_STATE;                   // [N_STATE * N_STATE] - loaded per step
    float* G_M_local = G_S_local + N_STATE * N_STATE;     // [N_STATE * N_STATE] - loaded per step

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;
    const int STRIDE = 4 * N_STATE;

    // Initialize gradient accumulators
    for (int i = tid; i < n2; i += blockDim.x) {
        dS[i] = 0.0f;
        dM[i] = 0.0f;
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
                M[i] = __bfloat162float(M_checkpoints[seg * B * n2 + b * n2 + i]);
            }
            __syncthreads();

            // Recompute forward to step t-1 (so S holds S_{t-1} for gate gradient)
            __shared__ float k_norm_val_t, m_norm_val_t;

            // First, recompute steps t_start to t-1, updating S and M at each step
            for (int tt = t_start; tt < t; tt++) {
                int col_idx = (tt * B + b) * STRIDE;
                if (tid < N_STATE) {
                    k_raw[tid] = __bfloat162float(kvqm_all[col_idx + tid]);
                    v_raw[tid] = __bfloat162float(kvqm_all[col_idx + N_STATE + tid]);
                    q_raw[tid] = __bfloat162float(kvqm_all[col_idx + 2 * N_STATE + tid]);
                    m_vec_raw[tid] = __bfloat162float(kvqm_all[col_idx + 3 * N_STATE + tid]);
                }
                __syncthreads();

                // Load cached gates
                for (int i = tid; i < n2; i += blockDim.x) {
                    G_S_local[i] = __bfloat162float(G_S_cache[tt * B * n2 + b * n2 + i]);
                    G_M_local[i] = __bfloat162float(G_M_cache[tt * B * n2 + b * n2 + i]);
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

                // Compute s_retrieved = S @ k_norm, s_delta
                if (tid < N_STATE) {
                    float sum = 0.0f;
                    for (int j = 0; j < N_STATE; j++) {
                        sum += S[tid * N_STATE + j] * k_norm[j];
                    }
                    s_retrieved[tid] = sum;
                    s_delta[tid] = v_raw[tid] - sum;
                }
                __syncthreads();

                // Update S
                for (int i = tid; i < n2; i += blockDim.x) {
                    int row = i / N_STATE;
                    int col = i % N_STATE;
                    S[i] = G_S_local[i] * S[i] + s_delta[row] * k_norm[col];
                }
                __syncthreads();

                // Compute m_retrieved and m_delta
                if (tid < N_STATE) {
                    float sum = 0.0f;
                    for (int j = 0; j < N_STATE; j++) {
                        sum += M[tid * N_STATE + j] * m_norm[j];
                    }
                    m_retrieved[tid] = sum;
                    m_delta[tid] = s_delta[tid] - m_retrieved[tid];
                }
                __syncthreads();

                // Update M
                for (int i = tid; i < n2; i += blockDim.x) {
                    int row = i / N_STATE;
                    int col = i % N_STATE;
                    M[i] = G_M_local[i] * M[i] + m_delta[row] * m_norm[col];
                }
                __syncthreads();
            }

            // Now handle step t: load inputs and compute intermediates, but DON'T update S/M
            // At this point S = S_{t-1}, M = M_{t-1}
            {
                int col_idx = (t * B + b) * STRIDE;
                if (tid < N_STATE) {
                    k_raw[tid] = __bfloat162float(kvqm_all[col_idx + tid]);
                    v_raw[tid] = __bfloat162float(kvqm_all[col_idx + N_STATE + tid]);
                    q_raw[tid] = __bfloat162float(kvqm_all[col_idx + 2 * N_STATE + tid]);
                    m_vec_raw[tid] = __bfloat162float(kvqm_all[col_idx + 3 * N_STATE + tid]);
                }
                __syncthreads();

                // Load cached gates for step t
                for (int i = tid; i < n2; i += blockDim.x) {
                    G_S_local[i] = __bfloat162float(G_S_cache[t * B * n2 + b * n2 + i]);
                    G_M_local[i] = __bfloat162float(G_M_cache[t * B * n2 + b * n2 + i]);
                }
                __syncthreads();

                // Normalize k and m for step t
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

                // Compute M_k = M @ k_norm (using M_{t-1})
                if (tid < N_STATE) {
                    float sum = 0.0f;
                    for (int j = 0; j < N_STATE; j++) {
                        sum += M[tid * N_STATE + j] * k_norm[j];
                    }
                    M_k[tid] = sum;
                }
                __syncthreads();

                // Compute s_retrieved = S_{t-1} @ k_norm, s_delta
                if (tid < N_STATE) {
                    float sum = 0.0f;
                    for (int j = 0; j < N_STATE; j++) {
                        sum += S[tid * N_STATE + j] * k_norm[j];
                    }
                    s_retrieved[tid] = sum;
                    s_delta[tid] = v_raw[tid] - sum;
                }
                __syncthreads();

                // Compute m_retrieved and m_delta (using M_{t-1})
                if (tid < N_STATE) {
                    float sum = 0.0f;
                    for (int j = 0; j < N_STATE; j++) {
                        sum += M[tid * N_STATE + j] * m_norm[j];
                    }
                    m_retrieved[tid] = sum;
                    m_delta[tid] = s_delta[tid] - m_retrieved[tid];
                }
                __syncthreads();
            }

            // Now S = S_{t-1}, M = M_{t-1}; s_delta, m_delta, k_norm, m_norm are for step t

            // Backward through output
            if (tid < N_STATE) {
                float d_out = __bfloat162float(d_output[t * B * N_STATE + b * N_STATE + tid]);
                float Sq = __bfloat162float(Sq_cache[t * B * N_STATE + b * N_STATE + tid]);
                float sig = 1.0f / (1.0f + expf(-Sq));
                float d_Sq = d_out * (2.0f * Sq * sig + Sq * Sq * sig * (1.0f - sig));
                d_Sq_shared[tid] = d_Sq;
            }
            __syncthreads();

            // dS += outer(d_Sq, q)
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                dS[i] += d_Sq_shared[row] * q_raw[col];
            }
            __syncthreads();

            // d_q = S_t^T @ d_Sq (where S_t is after update)
            // S_t = G_S * S + outer(s_delta, k)
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    float S_t_ij = G_S_local[i * N_STATE + tid] * S[i * N_STATE + tid] + s_delta[i] * k_norm[tid];
                    sum += S_t_ij * d_Sq_shared[i];
                }
                d_q_raw[tid] = sum;
            }
            __syncthreads();

            // --- Backward through M update ---
            // M_{t+1} = G_M_t * M_t + outer(m_delta_t, m_norm_t)
            // At this point, dM contains dL/dM_t (including G_S_{t+1} contributions from previous iteration)
            // d_m_delta[i] = sum_j(dM[i,j] * m_norm[j]) = (dM @ m_norm)[i]
            // d_m_norm[j] = sum_i(dM[i,j] * m_delta[i])
            if (tid < N_STATE) {
                float d_m_dt = 0.0f;
                float d_mn = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    float dM_ij = dM[tid * N_STATE + j];
                    d_m_dt += dM_ij * m_norm[j];
                    d_mn += dM[j * N_STATE + tid] * m_delta[j];
                }
                d_m_delta[tid] = d_m_dt;
                d_m_norm[tid] = d_mn;
            }
            __syncthreads();

            // Backward through G_M gate
            // Compute S_m = S_updated @ m_norm (needed for G_M backward)
            // S_updated[i,j] = G_S[i,j] * S[i,j] + s_delta[i] * k_norm[j]
            // Store S_m temporarily in s_retrieved (we're done with it)
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    float S_updated_ij = G_S_local[tid * N_STATE + j] * S[tid * N_STATE + j] + s_delta[tid] * k_norm[j];
                    sum += S_updated_ij * m_norm[j];
                }
                s_retrieved[tid] = sum;  // s_retrieved now holds S_m
            }
            __syncthreads();

            // G_M = sigmoid(S + outer(S_m, m) + B_M)
            // d_gate_logit = dM * M * G_M * (1 - G_M)
            // Accumulate gradients for B_M, S, and through S_m, m
            if (tid < N_STATE) {
                d_S_m[tid] = 0.0f;
            }
            __syncthreads();

            // G_M backward uses dM (full gradient including G_S contributions)
            // M_{t+1} = G_M * M_t + outer(m_delta, m_norm)
            // d_G_M = dL/dM_{t+1} * M_t (element-wise)
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                float dM_val = dM[i];  // Use full dM with G_S contributions
                float M_val = M[i];
                float G_M_val = G_M_local[i];

                // d_gate_logit = dM * M * G_M * (1 - G_M)
                float d_gate_logit = dM_val * M_val * G_M_val * (1.0f - G_M_val);

                // d_B_M[i,j] += d_gate_logit
                atomicAdd(&d_B_M_accum[i], d_gate_logit);

                // dS[i,j] += d_gate_logit (from S in gate)
                dS[i] += d_gate_logit;

                // d_S_m[row] += d_gate_logit * m_norm[col]
                atomicAdd(&d_S_m[row], d_gate_logit * m_norm[col]);

                // Contribution from outer(S_m, m_norm) to d_m_norm
                // d_m_norm[col] += d_gate_logit * S_m[row]
                // s_retrieved holds S_m
                atomicAdd(&d_m_norm[col], d_gate_logit * s_retrieved[row]);
            }
            __syncthreads();

            // d_S_m -> dS and d_m_norm contributions
            // S_m = S_updated @ m_norm (where S_updated is S after S update)
            // S_updated[i,j] = G_S[i,j] * S[i,j] + s_delta[i] * k_norm[j]
            // d_S_updated[i,j] += d_S_m[i] * m_norm[j]
            // d_m_norm[j] += S_updated[i,j]^T @ d_S_m = sum_i(S_updated[i,j] * d_S_m[i])
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                // Add dS contribution
                dS[i] += d_S_m[row] * m_norm[col];
            }
            __syncthreads();

            // d_m_norm contribution from S_m = S_updated @ m_norm
            // d_m_norm[j] += sum_i(S_updated[i,j] * d_S_m[i])
            // S_updated[i,j] = G_S[i,j] * S[i,j] + s_delta[i] * k_norm[j]
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    float S_updated_ij = G_S_local[i * N_STATE + tid] * S[i * N_STATE + tid] + s_delta[i] * k_norm[tid];
                    sum += S_updated_ij * d_S_m[i];
                }
                d_m_norm[tid] += sum;
            }
            __syncthreads();

            // --- Backward through S update ---
            // S_new = G_S * S + outer(s_delta, k_norm)
            // dS_old = dS_new * G_S
            // d_G_S[i,j] = dS_new[i,j] * S[i,j]
            // d_s_delta = dS_new @ k_norm
            // d_k_norm = dS_new.T @ s_delta
            if (tid < N_STATE) {
                float d_s_dt = 0.0f;
                float d_kn = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    float dS_ij = dS[tid * N_STATE + j];
                    d_s_dt += dS_ij * k_norm[j];
                    d_kn += dS[j * N_STATE + tid] * s_delta[j];
                }
                // Add contribution from M backward (m_delta = s_delta - M @ m_norm)
                d_s_delta[tid] = d_s_dt + d_m_delta[tid];
                d_k_norm[tid] = d_kn;
            }
            __syncthreads();

            // d_v = d_s_delta
            if (tid < N_STATE) {
                d_v_raw[tid] = d_s_delta[tid];
            }

            // d_k_norm contribution from s_retrieved: s_retrieved = S @ k_norm
            // d_k_norm += S^T @ (-d_s_delta)
            // IMPORTANT: Must be computed BEFORE we overwrite S with d_gate_logit
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    sum += S[i * N_STATE + tid] * (-d_s_delta[i]);
                }
                d_k_norm[tid] += sum;
            }
            __syncthreads();

            // Backward through G_S gate
            // G_S = sigmoid(M + outer(M_k, k) + B_S)
            // d_gate_logit = dS * S * G_S * (1 - G_S)
            // NOTE: G_S uses M_{t-1}, so contributions go to dM_{t-1}, not dM_t
            // We'll store d_gate_logit in S (done using it) and add to dM AFTER propagation
            if (tid < N_STATE) {
                d_M_k[tid] = 0.0f;
            }
            __syncthreads();

            // Compute d_B_S, d_M_k, and store d_gate_logit in S (we're done with S values)
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                float dS_val = dS[i];
                float S_val = S[i];
                float G_S_val = G_S_local[i];

                // d_gate_logit = dS * S * G_S * (1 - G_S)
                float d_gate_logit = dS_val * S_val * G_S_val * (1.0f - G_S_val);

                // d_B_S[i,j] += d_gate_logit
                atomicAdd(&d_B_S_accum[i], d_gate_logit);

                // d_M_k[row] += d_gate_logit * k_norm[col]
                atomicAdd(&d_M_k[row], d_gate_logit * k_norm[col]);

                // Store d_gate_logit in S for later (we're done with S values for this step)
                S[i] = d_gate_logit;
            }
            __syncthreads();

            // d_k_norm contribution from M_k = M @ k_norm (in gate computation)
            // d_k_norm += M^T @ d_M_k
            // Note: M here is M_{t-1} (we haven't overwritten it yet)
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    sum += M[i * N_STATE + tid] * d_M_k[i];
                }
                d_k_norm[tid] += sum;
            }
            __syncthreads();

            // d_k_norm contribution from outer(M_k, k_norm) in gate logit
            // gate_outer[i,j] = M_k[i] * k_norm[j]
            // d_k_norm[j] += sum_i(d_gate_logit[i,j] * M_k[i])
            // d_gate_logit is stored in S
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    sum += S[i * N_STATE + tid] * M_k[i];
                }
                d_k_norm[tid] += sum;
            }
            __syncthreads();

            // d_m_norm contribution from m_retrieved: m_retrieved = M @ m_norm
            // d_m_norm += M^T @ d_m_retrieved = M^T @ (-d_m_delta)
            // [from m_delta = s_delta - m_retrieved, so d_m_retrieved = -d_m_delta]
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    sum += M[i * N_STATE + tid] * (-d_m_delta[i]);
                }
                d_m_norm[tid] += sum;
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

            // Update dS for next iteration: dS_old = dS_new * G_S
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                dS[i] = dS[i] * G_S_local[i] + (-d_s_delta[row]) * k_norm[col];
            }
            __syncthreads();

            // Update dM for next iteration (propagate through M recurrence and add G_S contributions)
            // dL/dM_{t-1} = dL/dM_t * G_M_t + outer(-d_m_delta, m_norm) + G_S_t contributions
            // First: propagate through M recurrence
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                dM[i] = dM[i] * G_M_local[i] + (-d_m_delta[row]) * m_norm[col];
            }
            __syncthreads();

            // Then: add G_S contributions to dM
            // G_S at step t uses M_{t-1}, so its gradient goes to dL/dM_{t-1}
            // We stored d_gate_logit in S (array) and d_M_k contributions
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                // Add direct M contribution (stored in S)
                dM[i] += S[i];
                // Add M_k contribution (outer product term)
                dM[i] += d_M_k[row] * k_norm[col];
            }
            __syncthreads();
        }
    }
}

// ============================================================================
// FP32 versions
// ============================================================================

template<int N_STATE>
__global__ void E80FullRankGateForwardKernel_FP32(
    int T,
    int B,
    const float* __restrict__ kvqm_all,
    const float* __restrict__ B_S,
    const float* __restrict__ B_M,
    float* __restrict__ S,
    float* __restrict__ M,
    float* __restrict__ output,
    float* __restrict__ S_checkpoints,
    float* __restrict__ M_checkpoints,
    float* __restrict__ Sq_cache,
    float* __restrict__ G_S_cache,
    float* __restrict__ G_M_cache,
    int checkpoint_interval
) {
    int b = blockIdx.x;
    if (b >= B) return;

    extern __shared__ float shared_mem[];
    float* S_shared = shared_mem;
    float* M_shared = S_shared + N_STATE * N_STATE;
    float* k_shared = M_shared + N_STATE * N_STATE;
    float* v_shared = k_shared + N_STATE;
    float* q_shared = v_shared + N_STATE;
    float* m_vec_shared = q_shared + N_STATE;
    float* M_k = m_vec_shared + N_STATE;
    float* S_m = M_k + N_STATE;
    float* s_retrieved = S_m + N_STATE;
    float* m_retrieved = s_retrieved + N_STATE;
    float* B_S_shared = m_retrieved + N_STATE;
    float* B_M_shared = B_S_shared + N_STATE * N_STATE;

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;
    const int STRIDE = 4 * N_STATE;

    // Load gate bias matrices
    for (int i = tid; i < n2; i += blockDim.x) {
        B_S_shared[i] = B_S[i];
        B_M_shared[i] = B_M[i];
    }
    __syncthreads();

    for (int i = tid; i < n2; i += blockDim.x) {
        S_shared[i] = S[b * n2 + i];
        M_shared[i] = M[b * n2 + i];
    }
    __syncthreads();

    for (int i = tid; i < n2; i += blockDim.x) {
        S_checkpoints[b * n2 + i] = S_shared[i];
        M_checkpoints[b * n2 + i] = M_shared[i];
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

        // M_k = M @ k_norm
        if (tid < N_STATE) {
            float sum = 0.0f;
            for (int j = 0; j < N_STATE; j++) {
                sum += M_shared[tid * N_STATE + j] * k_shared[j];
            }
            M_k[tid] = sum;
        }
        __syncthreads();

        // s_retrieved = S @ k_norm
        if (tid < N_STATE) {
            float sum = 0.0f;
            for (int j = 0; j < N_STATE; j++) {
                sum += S_shared[tid * N_STATE + j] * k_shared[j];
            }
            s_retrieved[tid] = sum;
        }
        __syncthreads();

        // G_S and S update
        for (int i = tid; i < n2; i += blockDim.x) {
            int row = i / N_STATE;
            int col = i % N_STATE;
            float gate_logit = M_shared[i] + M_k[row] * k_shared[col] + B_S_shared[i];
            float G_S_val = 1.0f / (1.0f + expf(-gate_logit));
            G_S_cache[t * B * n2 + b * n2 + i] = G_S_val;
            float s_delta_row = v_shared[row] - s_retrieved[row];
            S_shared[i] = G_S_val * S_shared[i] + s_delta_row * k_shared[col];
        }
        __syncthreads();

        // S_m = S @ m_norm
        if (tid < N_STATE) {
            float sum = 0.0f;
            for (int j = 0; j < N_STATE; j++) {
                sum += S_shared[tid * N_STATE + j] * m_vec_shared[j];
            }
            S_m[tid] = sum;
        }
        __syncthreads();

        // m_retrieved = M @ m_norm
        if (tid < N_STATE) {
            float sum = 0.0f;
            for (int j = 0; j < N_STATE; j++) {
                sum += M_shared[tid * N_STATE + j] * m_vec_shared[j];
            }
            m_retrieved[tid] = sum;
        }
        __syncthreads();

        // G_M and M update
        for (int i = tid; i < n2; i += blockDim.x) {
            int row = i / N_STATE;
            int col = i % N_STATE;
            float gate_logit = S_shared[i] + S_m[row] * m_vec_shared[col] + B_M_shared[i];
            float G_M_val = 1.0f / (1.0f + expf(-gate_logit));
            G_M_cache[t * B * n2 + b * n2 + i] = G_M_val;
            float s_delta_row = v_shared[row] - s_retrieved[row];
            float m_delta_row = s_delta_row - m_retrieved[row];
            M_shared[i] = G_M_val * M_shared[i] + m_delta_row * m_vec_shared[col];
        }
        __syncthreads();

        if ((t + 1) % checkpoint_interval == 0) {
            int cp_idx = (t + 1) / checkpoint_interval;
            for (int i = tid; i < n2; i += blockDim.x) {
                S_checkpoints[cp_idx * B * n2 + b * n2 + i] = S_shared[i];
                M_checkpoints[cp_idx * B * n2 + b * n2 + i] = M_shared[i];
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
        M[b * n2 + i] = M_shared[i];
    }
}

template<int N_STATE>
__global__ void E80FullRankGateBackwardKernel_FP32(
    int T,
    int B,
    const float* __restrict__ kvqm_all,
    const float* __restrict__ B_S,
    const float* __restrict__ B_M,
    const float* __restrict__ G_S_cache,
    const float* __restrict__ G_M_cache,
    const float* __restrict__ S_checkpoints,
    const float* __restrict__ M_checkpoints,
    const float* __restrict__ Sq_cache,
    const float* __restrict__ d_output,
    float* __restrict__ d_kvqm_all,
    float* __restrict__ d_B_S_accum,
    float* __restrict__ d_B_M_accum,
    int checkpoint_interval
) {
    // FP32 backward - similar to BF16 but without conversions
    int b = blockIdx.x;
    if (b >= B) return;

    extern __shared__ float shared_mem[];
    float* S = shared_mem;
    float* M = S + N_STATE * N_STATE;
    float* dS = M + N_STATE * N_STATE;
    float* dM = dS + N_STATE * N_STATE;
    float* k_raw = dM + N_STATE * N_STATE;
    float* v_raw = k_raw + N_STATE;
    float* q_raw = v_raw + N_STATE;
    float* m_vec_raw = q_raw + N_STATE;
    float* k_norm = m_vec_raw + N_STATE;
    float* m_norm = k_norm + N_STATE;
    float* M_k = m_norm + N_STATE;
    float* S_m = M_k + N_STATE;
    float* s_retrieved = S_m + N_STATE;
    float* m_retrieved = s_retrieved + N_STATE;
    float* s_delta = m_retrieved + N_STATE;
    float* m_delta = s_delta + N_STATE;
    float* d_k_raw = m_delta + N_STATE;
    float* d_v_raw = d_k_raw + N_STATE;
    float* d_q_raw = d_v_raw + N_STATE;
    float* d_m_raw = d_q_raw + N_STATE;
    float* d_Sq_shared = d_m_raw + N_STATE;
    float* d_s_delta = d_Sq_shared + N_STATE;
    float* d_m_delta = d_s_delta + N_STATE;
    float* d_k_norm = d_m_delta + N_STATE;
    float* d_m_norm = d_k_norm + N_STATE;
    float* d_M_k = d_m_norm + N_STATE;
    float* d_S_m = d_M_k + N_STATE;
    float* G_S_local = d_S_m + N_STATE;
    float* G_M_local = G_S_local + N_STATE * N_STATE;

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;
    const int STRIDE = 4 * N_STATE;

    // Initialize gradient accumulators
    for (int i = tid; i < n2; i += blockDim.x) {
        dS[i] = 0.0f;
        dM[i] = 0.0f;
    }
    __syncthreads();

    int num_segments = (T + checkpoint_interval - 1) / checkpoint_interval;

    for (int seg = num_segments - 1; seg >= 0; seg--) {
        int t_start = seg * checkpoint_interval;
        int t_end = min(t_start + checkpoint_interval, T);

        for (int t = t_end - 1; t >= t_start; t--) {
            for (int i = tid; i < n2; i += blockDim.x) {
                S[i] = S_checkpoints[seg * B * n2 + b * n2 + i];
                M[i] = M_checkpoints[seg * B * n2 + b * n2 + i];
            }
            __syncthreads();

            // Recompute forward to step t-1 (so S holds S_{t-1} for gate gradient)
            __shared__ float k_norm_val_t, m_norm_val_t;

            // First, recompute steps t_start to t-1, updating S and M at each step
            for (int tt = t_start; tt < t; tt++) {
                int col_idx = (tt * B + b) * STRIDE;
                if (tid < N_STATE) {
                    k_raw[tid] = kvqm_all[col_idx + tid];
                    v_raw[tid] = kvqm_all[col_idx + N_STATE + tid];
                    q_raw[tid] = kvqm_all[col_idx + 2 * N_STATE + tid];
                    m_vec_raw[tid] = kvqm_all[col_idx + 3 * N_STATE + tid];
                }
                __syncthreads();

                for (int i = tid; i < n2; i += blockDim.x) {
                    G_S_local[i] = G_S_cache[tt * B * n2 + b * n2 + i];
                    G_M_local[i] = G_M_cache[tt * B * n2 + b * n2 + i];
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

                // Compute s_retrieved = S @ k_norm, s_delta
                if (tid < N_STATE) {
                    float sum = 0.0f;
                    for (int j = 0; j < N_STATE; j++) {
                        sum += S[tid * N_STATE + j] * k_norm[j];
                    }
                    s_retrieved[tid] = sum;
                    s_delta[tid] = v_raw[tid] - sum;
                }
                __syncthreads();

                // Update S
                for (int i = tid; i < n2; i += blockDim.x) {
                    int row = i / N_STATE;
                    int col = i % N_STATE;
                    S[i] = G_S_local[i] * S[i] + s_delta[row] * k_norm[col];
                }
                __syncthreads();

                // Compute m_retrieved and m_delta
                if (tid < N_STATE) {
                    float sum = 0.0f;
                    for (int j = 0; j < N_STATE; j++) {
                        sum += M[tid * N_STATE + j] * m_norm[j];
                    }
                    m_retrieved[tid] = sum;
                    m_delta[tid] = s_delta[tid] - m_retrieved[tid];
                }
                __syncthreads();

                // Update M
                for (int i = tid; i < n2; i += blockDim.x) {
                    int row = i / N_STATE;
                    int col = i % N_STATE;
                    M[i] = G_M_local[i] * M[i] + m_delta[row] * m_norm[col];
                }
                __syncthreads();
            }

            // Now handle step t: load inputs and compute intermediates, but DON'T update S/M
            // At this point S = S_{t-1}, M = M_{t-1}
            {
                int col_idx = (t * B + b) * STRIDE;
                if (tid < N_STATE) {
                    k_raw[tid] = kvqm_all[col_idx + tid];
                    v_raw[tid] = kvqm_all[col_idx + N_STATE + tid];
                    q_raw[tid] = kvqm_all[col_idx + 2 * N_STATE + tid];
                    m_vec_raw[tid] = kvqm_all[col_idx + 3 * N_STATE + tid];
                }
                __syncthreads();

                // Load cached gates for step t
                for (int i = tid; i < n2; i += blockDim.x) {
                    G_S_local[i] = G_S_cache[t * B * n2 + b * n2 + i];
                    G_M_local[i] = G_M_cache[t * B * n2 + b * n2 + i];
                }
                __syncthreads();

                // Normalize k and m for step t
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

                // Compute M_k = M @ k_norm (using M_{t-1})
                if (tid < N_STATE) {
                    float sum = 0.0f;
                    for (int j = 0; j < N_STATE; j++) {
                        sum += M[tid * N_STATE + j] * k_norm[j];
                    }
                    M_k[tid] = sum;
                }
                __syncthreads();

                // Compute s_retrieved = S_{t-1} @ k_norm, s_delta
                if (tid < N_STATE) {
                    float sum = 0.0f;
                    for (int j = 0; j < N_STATE; j++) {
                        sum += S[tid * N_STATE + j] * k_norm[j];
                    }
                    s_retrieved[tid] = sum;
                    s_delta[tid] = v_raw[tid] - sum;
                }
                __syncthreads();

                // Compute m_retrieved and m_delta (using M_{t-1})
                if (tid < N_STATE) {
                    float sum = 0.0f;
                    for (int j = 0; j < N_STATE; j++) {
                        sum += M[tid * N_STATE + j] * m_norm[j];
                    }
                    m_retrieved[tid] = sum;
                    m_delta[tid] = s_delta[tid] - m_retrieved[tid];
                }
                __syncthreads();
            }

            // Now S = S_{t-1}, M = M_{t-1}; s_delta, m_delta, k_norm, m_norm are for step t

            // Backward pass for step t
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
                    float S_t_ij = G_S_local[i * N_STATE + tid] * S[i * N_STATE + tid] + s_delta[i] * k_norm[tid];
                    sum += S_t_ij * d_Sq_shared[i];
                }
                d_q_raw[tid] = sum;
            }
            __syncthreads();

            // M backward - use full dM (includes G_S contributions from previous iteration)
            if (tid < N_STATE) {
                float d_m_dt = 0.0f;
                float d_mn = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    float dM_ij = dM[tid * N_STATE + j];
                    d_m_dt += dM_ij * m_norm[j];
                    d_mn += dM[j * N_STATE + tid] * m_delta[j];
                }
                d_m_delta[tid] = d_m_dt;
                d_m_norm[tid] = d_mn;
            }
            __syncthreads();

            // Compute S_m = S_updated @ m_norm (needed for G_M backward)
            // S_updated[i,j] = G_S[i,j] * S[i,j] + s_delta[i] * k_norm[j]
            // Store S_m temporarily in s_retrieved (we're done with it)
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    float S_updated_ij = G_S_local[tid * N_STATE + j] * S[tid * N_STATE + j] + s_delta[tid] * k_norm[j];
                    sum += S_updated_ij * m_norm[j];
                }
                s_retrieved[tid] = sum;  // s_retrieved now holds S_m
            }
            __syncthreads();

            // G_M backward - use full dM (includes G_S contributions)
            if (tid < N_STATE) {
                d_S_m[tid] = 0.0f;
            }
            __syncthreads();

            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                float dM_val = dM[i];  // Use full dM with G_S contributions
                float M_val = M[i];
                float G_M_val = G_M_local[i];
                float d_gate_logit = dM_val * M_val * G_M_val * (1.0f - G_M_val);
                atomicAdd(&d_B_M_accum[i], d_gate_logit);
                dS[i] += d_gate_logit;
                atomicAdd(&d_S_m[row], d_gate_logit * m_norm[col]);
                // Contribution from outer(S_m, m_norm) to d_m_norm
                // d_m_norm[col] += d_gate_logit * S_m[row]
                // s_retrieved holds S_m
                atomicAdd(&d_m_norm[col], d_gate_logit * s_retrieved[row]);
            }
            __syncthreads();

            // d_S_m -> dS and d_m_norm contributions
            // S_m = S_updated @ m_norm (where S_updated is S after S update)
            // S_updated[i,j] = G_S[i,j] * S[i,j] + s_delta[i] * k_norm[j]
            // d_S_updated[i,j] += d_S_m[i] * m_norm[j]
            // d_m_norm[j] += S_updated[i,j]^T @ d_S_m = sum_i(S_updated[i,j] * d_S_m[i])
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                // Add dS contribution
                dS[i] += d_S_m[row] * m_norm[col];
            }
            __syncthreads();

            // d_m_norm contribution from S_m = S_updated @ m_norm
            // d_m_norm[j] += sum_i(S_updated[i,j] * d_S_m[i])
            // S_updated[i,j] = G_S[i,j] * S[i,j] + s_delta[i] * k_norm[j]
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    float S_updated_ij = G_S_local[i * N_STATE + tid] * S[i * N_STATE + tid] + s_delta[i] * k_norm[tid];
                    sum += S_updated_ij * d_S_m[i];
                }
                d_m_norm[tid] += sum;
            }
            __syncthreads();

            // S backward
            if (tid < N_STATE) {
                float d_s_dt = 0.0f;
                float d_kn = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    float dS_ij = dS[tid * N_STATE + j];
                    d_s_dt += dS_ij * k_norm[j];
                    d_kn += dS[j * N_STATE + tid] * s_delta[j];
                }
                d_s_delta[tid] = d_s_dt + d_m_delta[tid];
                d_k_norm[tid] = d_kn;
            }
            __syncthreads();

            if (tid < N_STATE) {
                d_v_raw[tid] = d_s_delta[tid];
            }

            // d_k_norm contribution from s_retrieved: s_retrieved = S @ k_norm
            // d_k_norm += S^T @ (-d_s_delta)
            // IMPORTANT: Must be computed BEFORE we overwrite S with d_gate_logit
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    sum += S[i * N_STATE + tid] * (-d_s_delta[i]);
                }
                d_k_norm[tid] += sum;
            }
            __syncthreads();

            // G_S backward - compute gate gradient
            // (G_S uses M_{t-1}, so contributions should go to dM_{t-1}, not dM_t)
            // We'll store d_gate_logit in S (done using it) and add to dM after propagation
            if (tid < N_STATE) {
                d_M_k[tid] = 0.0f;
            }
            __syncthreads();

            // Compute d_B_S, d_M_k, and store d_gate_logit in S (we're done with S values)
            // Note: We'll add d_gate_logit to dM AFTER the dM propagation
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                float dS_val = dS[i];
                float S_val = S[i];
                float G_S_val = G_S_local[i];
                float d_gate_logit = dS_val * S_val * G_S_val * (1.0f - G_S_val);
                atomicAdd(&d_B_S_accum[i], d_gate_logit);
                atomicAdd(&d_M_k[row], d_gate_logit * k_norm[col]);
                // Store d_gate_logit in S for later (we're done with S values for this step)
                S[i] = d_gate_logit;
            }
            __syncthreads();

            // d_k_norm contribution from M_k = M @ k_norm (in gate computation)
            // d_k_norm += M^T @ d_M_k
            // Note: M here is M_{t-1} (we haven't overwritten it yet)
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    sum += M[i * N_STATE + tid] * d_M_k[i];
                }
                d_k_norm[tid] += sum;
            }
            __syncthreads();

            // d_k_norm contribution from outer(M_k, k_norm) in gate logit
            // gate_outer[i,j] = M_k[i] * k_norm[j]
            // d_k_norm[j] += sum_i(d_gate_logit[i,j] * M_k[i])
            // d_gate_logit is stored in S
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    sum += S[i * N_STATE + tid] * M_k[i];
                }
                d_k_norm[tid] += sum;
            }
            __syncthreads();

            // d_m_norm contribution from m_retrieved: m_retrieved = M @ m_norm
            // d_m_norm += M^T @ d_m_retrieved = M^T @ (-d_m_delta)
            // [from m_delta = s_delta - m_retrieved, so d_m_retrieved = -d_m_delta]
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    sum += M[i * N_STATE + tid] * (-d_m_delta[i]);
                }
                d_m_norm[tid] += sum;
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

            int col_idx_t = (t * B + b) * STRIDE;
            if (tid < N_STATE) {
                d_kvqm_all[col_idx_t + tid] = d_k_raw[tid];
                d_kvqm_all[col_idx_t + N_STATE + tid] = d_v_raw[tid];
                d_kvqm_all[col_idx_t + 2 * N_STATE + tid] = d_q_raw[tid];
                d_kvqm_all[col_idx_t + 3 * N_STATE + tid] = d_m_raw[tid];
            }
            __syncthreads();

            // Propagate dS: dS_{t-1} = G_S_t * dS_t + delta_contribution
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                dS[i] = dS[i] * G_S_local[i] + (-d_s_delta[row]) * k_norm[col];
            }
            __syncthreads();

            // Propagate dM: dM_{t-1} = G_M_t * dM_t + delta_contribution + G_S contributions
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                dM[i] = dM[i] * G_M_local[i] + (-d_m_delta[row]) * m_norm[col];
            }
            __syncthreads();

            // Add G_S contributions to dM
            // G_S at step t uses M_{t-1}, so its gradient goes to dM_{t-1}
            // We stored d_gate_logit in S (array) and d_M_k contributions
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                // Add direct M contribution (stored in S)
                dM[i] += S[i];
                // Add M_k contribution (outer product term)
                dM[i] += d_M_k[row] * k_norm[col];
            }
            __syncthreads();
        }
    }
}

// ============================================================================
// Template instantiations
// ============================================================================

#define INSTANTIATE_E80_KERNELS_BF16(N) \
    template __global__ void E80FullRankGateForwardKernel_BF16<N>( \
        int, int, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, \
        __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, \
        __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int); \
    template __global__ void E80FullRankGateBackwardKernel_BF16<N>( \
        int, int, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, \
        const __nv_bfloat16*, const __nv_bfloat16*, \
        const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, \
        __nv_bfloat16*, float*, float*, int);

#define INSTANTIATE_E80_KERNELS_FP32(N) \
    template __global__ void E80FullRankGateForwardKernel_FP32<N>( \
        int, int, const float*, const float*, const float*, \
        float*, float*, float*, float*, float*, \
        float*, float*, float*, int); \
    template __global__ void E80FullRankGateBackwardKernel_FP32<N>( \
        int, int, const float*, const float*, const float*, \
        const float*, const float*, \
        const float*, const float*, const float*, const float*, \
        float*, float*, float*, int);

// Standard kernel sizes (can use shared memory)
INSTANTIATE_E80_KERNELS_BF16(8)
INSTANTIATE_E80_KERNELS_BF16(16)
INSTANTIATE_E80_KERNELS_BF16(32)
INSTANTIATE_E80_KERNELS_BF16(48)
INSTANTIATE_E80_KERNELS_BF16(64)

INSTANTIATE_E80_KERNELS_FP32(8)
INSTANTIATE_E80_KERNELS_FP32(16)
INSTANTIATE_E80_KERNELS_FP32(32)
INSTANTIATE_E80_KERNELS_FP32(48)
INSTANTIATE_E80_KERNELS_FP32(64)

// ============================================================================
// Dispatcher functions
// ============================================================================

void dispatch_e80_full_rank_gate_forward(
    int T, int B, int n_state,
    const __nv_bfloat16* kvqm_all,
    const __nv_bfloat16* B_S,
    const __nv_bfloat16* B_M,
    __nv_bfloat16* S, __nv_bfloat16* M, __nv_bfloat16* output,
    __nv_bfloat16* S_checkpoints, __nv_bfloat16* M_checkpoints,
    __nv_bfloat16* Sq_cache,
    __nv_bfloat16* G_S_cache, __nv_bfloat16* G_M_cache,
    int checkpoint_interval, cudaStream_t stream
) {
    // Shared memory: 2*n^2 (S, M) + 2*n^2 (B_S, B_M) + 10*n (vectors)
    int shared_size = (4 * n_state * n_state + 10 * n_state) * sizeof(float);

    #define DISPATCH_E80_FWD(N) \
        E80FullRankGateForwardKernel_BF16<N><<<B, 256, shared_size, stream>>>( \
            T, B, kvqm_all, B_S, B_M, \
            S, M, output, S_checkpoints, M_checkpoints, Sq_cache, \
            G_S_cache, G_M_cache, checkpoint_interval);

    #define DISPATCH_E80_FWD_EXT(N) \
        { \
            cudaError_t attr_err = cudaFuncSetAttribute( \
                E80FullRankGateForwardKernel_BF16<N>, \
                cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size); \
            if (attr_err != cudaSuccess) { \
                fprintf(stderr, "E80 Forward: n_state=%d cudaFuncSetAttribute failed: %s\n", \
                        N, cudaGetErrorString(attr_err)); \
            } else { \
                DISPATCH_E80_FWD(N); \
            } \
        }

    switch (n_state) {
        case 8: DISPATCH_E80_FWD(8); break;
        case 16: DISPATCH_E80_FWD(16); break;
        case 32: DISPATCH_E80_FWD(32); break;
        case 48: DISPATCH_E80_FWD_EXT(48); break;
        case 64: DISPATCH_E80_FWD_EXT(64); break;
        default:
            fprintf(stderr, "E80: Unsupported n_state=%d (use 8, 16, 32, 48, or 64)\n", n_state);
    }
    #undef DISPATCH_E80_FWD
    #undef DISPATCH_E80_FWD_EXT
}

void dispatch_e80_full_rank_gate_backward(
    int T, int B, int n_state,
    const __nv_bfloat16* kvqm_all,
    const __nv_bfloat16* B_S, const __nv_bfloat16* B_M,
    const __nv_bfloat16* G_S_cache, const __nv_bfloat16* G_M_cache,
    const __nv_bfloat16* S_checkpoints, const __nv_bfloat16* M_checkpoints,
    const __nv_bfloat16* Sq_cache, const __nv_bfloat16* d_output,
    __nv_bfloat16* d_kvqm_all,
    float* d_B_S_accum, float* d_B_M_accum,
    int checkpoint_interval, cudaStream_t stream
) {
    // Shared memory for backward: 4*n^2 (S, M, dS, dM) + 2*n^2 (G_S, G_M) + 23*n (vectors)
    int shared_size = (6 * n_state * n_state + 23 * n_state) * sizeof(float);

    #define DISPATCH_E80_BWD(N) \
        E80FullRankGateBackwardKernel_BF16<N><<<B, 256, shared_size, stream>>>( \
            T, B, kvqm_all, B_S, B_M, \
            G_S_cache, G_M_cache, \
            S_checkpoints, M_checkpoints, Sq_cache, d_output, \
            d_kvqm_all, d_B_S_accum, d_B_M_accum, checkpoint_interval);

    #define DISPATCH_E80_BWD_EXT(N) \
        { \
            cudaError_t attr_err = cudaFuncSetAttribute( \
                E80FullRankGateBackwardKernel_BF16<N>, \
                cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size); \
            if (attr_err != cudaSuccess) { \
                fprintf(stderr, "E80 Backward: n_state=%d requires %d KB shared memory but GPU limit exceeded\n", \
                        N, shared_size / 1024); \
            } else { \
                DISPATCH_E80_BWD(N); \
            } \
        }

    switch (n_state) {
        case 8: DISPATCH_E80_BWD(8); break;
        case 16: DISPATCH_E80_BWD(16); break;
        case 32: DISPATCH_E80_BWD(32); break;
        case 48: DISPATCH_E80_BWD_EXT(48); break;
        case 64: DISPATCH_E80_BWD_EXT(64); break;
        default:
            fprintf(stderr, "E80: Unsupported n_state=%d (use 8, 16, 32, 48, or 64)\n", n_state);
    }
    #undef DISPATCH_E80_BWD
    #undef DISPATCH_E80_BWD_EXT
}

void dispatch_e80_full_rank_gate_forward_fp32(
    int T, int B, int n_state,
    const float* kvqm_all,
    const float* B_S, const float* B_M,
    float* S, float* M, float* output,
    float* S_checkpoints, float* M_checkpoints,
    float* Sq_cache,
    float* G_S_cache, float* G_M_cache,
    int checkpoint_interval, cudaStream_t stream
) {
    int shared_size = (4 * n_state * n_state + 10 * n_state) * sizeof(float);

    #define DISPATCH_E80_FWD_FP32(N) \
        E80FullRankGateForwardKernel_FP32<N><<<B, 256, shared_size, stream>>>( \
            T, B, kvqm_all, B_S, B_M, \
            S, M, output, S_checkpoints, M_checkpoints, Sq_cache, \
            G_S_cache, G_M_cache, checkpoint_interval);

    #define DISPATCH_E80_FWD_FP32_EXT(N) \
        { \
            cudaError_t attr_err = cudaFuncSetAttribute( \
                E80FullRankGateForwardKernel_FP32<N>, \
                cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size); \
            if (attr_err != cudaSuccess) { \
                fprintf(stderr, "E80 FP32 Forward: n_state=%d cudaFuncSetAttribute failed: %s\n", \
                        N, cudaGetErrorString(attr_err)); \
            } else { \
                DISPATCH_E80_FWD_FP32(N); \
            } \
        }

    switch (n_state) {
        case 8: DISPATCH_E80_FWD_FP32(8); break;
        case 16: DISPATCH_E80_FWD_FP32(16); break;
        case 32: DISPATCH_E80_FWD_FP32(32); break;
        case 48: DISPATCH_E80_FWD_FP32_EXT(48); break;
        case 64: DISPATCH_E80_FWD_FP32_EXT(64); break;
        default:
            fprintf(stderr, "E80: Unsupported n_state=%d (use 8, 16, 32, 48, or 64)\n", n_state);
    }
    #undef DISPATCH_E80_FWD_FP32
    #undef DISPATCH_E80_FWD_FP32_EXT
}

void dispatch_e80_full_rank_gate_backward_fp32(
    int T, int B, int n_state,
    const float* kvqm_all,
    const float* B_S, const float* B_M,
    const float* G_S_cache, const float* G_M_cache,
    const float* S_checkpoints, const float* M_checkpoints,
    const float* Sq_cache, const float* d_output,
    float* d_kvqm_all,
    float* d_B_S_accum, float* d_B_M_accum,
    int checkpoint_interval, cudaStream_t stream
) {
    // Shared memory for backward: 4*n^2 (S, M, dS, dM) + 2*n^2 (G_S, G_M) + 23*n (vectors)
    int shared_size = (6 * n_state * n_state + 23 * n_state) * sizeof(float);

    #define DISPATCH_E80_BWD_FP32(N) \
        E80FullRankGateBackwardKernel_FP32<N><<<B, 256, shared_size, stream>>>( \
            T, B, kvqm_all, B_S, B_M, \
            G_S_cache, G_M_cache, \
            S_checkpoints, M_checkpoints, Sq_cache, d_output, \
            d_kvqm_all, d_B_S_accum, d_B_M_accum, checkpoint_interval);

    #define DISPATCH_E80_BWD_FP32_EXT(N) \
        { \
            cudaError_t attr_err = cudaFuncSetAttribute( \
                E80FullRankGateBackwardKernel_FP32<N>, \
                cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size); \
            if (attr_err != cudaSuccess) { \
                fprintf(stderr, "E80 FP32 Backward: n_state=%d requires %d KB shared memory but GPU limit exceeded\n", \
                        N, shared_size / 1024); \
            } else { \
                DISPATCH_E80_BWD_FP32(N); \
            } \
        }

    switch (n_state) {
        case 8: DISPATCH_E80_BWD_FP32(8); break;
        case 16: DISPATCH_E80_BWD_FP32(16); break;
        case 32: DISPATCH_E80_BWD_FP32(32); break;
        case 48: DISPATCH_E80_BWD_FP32_EXT(48); break;
        case 64: DISPATCH_E80_BWD_FP32_EXT(64); break;
        default:
            fprintf(stderr, "E80: Unsupported n_state=%d (use 8, 16, 32, 48, or 64)\n", n_state);
    }
    #undef DISPATCH_E80_BWD_FP32
    #undef DISPATCH_E80_BWD_FP32_EXT
}

// ============================================================================
// E80FullRankGateForward Implementation (wrapper class for Python bindings)
// ============================================================================

template<typename DataT>
E80FullRankGateForward<DataT>::E80FullRankGateForward(
    bool training, int batch_size, int n_state, int dim,
    const cublasHandle_t& blas_handle, const cudaStream_t& stream)
    : training_(training), batch_size_(batch_size), n_state_(n_state),
      dim_(dim), blas_handle_(blas_handle), stream_(stream) {}

template<typename DataT>
void E80FullRankGateForward<DataT>::Run(
    int steps,
    const DataT* W_kvqm,
    const DataT* B_S,
    const DataT* B_M,
    const DataT* x,
    DataT* S, DataT* M,
    DataT* output,
    DataT* kvqm_cache,
    DataT* S_checkpoints, DataT* M_checkpoints,
    DataT* Sq_cache,
    DataT* G_S_cache, DataT* G_M_cache
) {
    int T = steps;
    int B = batch_size_;
    int n = n_state_;
    int d = dim_;
    int checkpoint_interval = E80_CHECKPOINT_INTERVAL;

    const float alpha = 1.0f, beta_zero = 0.0f;

    // FUSED projection: kvqm = W_kvqm @ x
    cudaDataType_t data_type = std::is_same<DataT, __nv_bfloat16>::value ? CUDA_R_16BF : CUDA_R_32F;
    cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                 4 * n, T * B, d,
                 &alpha,
                 W_kvqm, data_type, d,
                 x, data_type, d,
                 &beta_zero,
                 kvqm_cache, data_type, 4 * n,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    if constexpr (std::is_same<DataT, __nv_bfloat16>::value) {
        dispatch_e80_full_rank_gate_forward(T, B, n, kvqm_cache, B_S, B_M,
                                            S, M, output, S_checkpoints, M_checkpoints, Sq_cache,
                                            G_S_cache, G_M_cache,
                                            checkpoint_interval, stream_);
    } else {
        dispatch_e80_full_rank_gate_forward_fp32(T, B, n,
                                                  reinterpret_cast<const float*>(kvqm_cache),
                                                  reinterpret_cast<const float*>(B_S),
                                                  reinterpret_cast<const float*>(B_M),
                                                  reinterpret_cast<float*>(S),
                                                  reinterpret_cast<float*>(M),
                                                  reinterpret_cast<float*>(output),
                                                  reinterpret_cast<float*>(S_checkpoints),
                                                  reinterpret_cast<float*>(M_checkpoints),
                                                  reinterpret_cast<float*>(Sq_cache),
                                                  reinterpret_cast<float*>(G_S_cache),
                                                  reinterpret_cast<float*>(G_M_cache),
                                                  checkpoint_interval, stream_);
    }
}

template<typename DataT>
E80FullRankGateBackward<DataT>::E80FullRankGateBackward(
    int batch_size, int n_state, int dim,
    const cublasHandle_t& blas_handle, const cudaStream_t& stream)
    : batch_size_(batch_size), n_state_(n_state), dim_(dim),
      blas_handle_(blas_handle), stream_(stream) {}

template<typename DataT>
void E80FullRankGateBackward<DataT>::Run(
    int steps,
    const DataT* W_kvqm,
    const DataT* B_S, const DataT* B_M,
    const DataT* x,
    const DataT* kvqm_cache,
    const DataT* S_checkpoints, const DataT* M_checkpoints,
    const DataT* Sq_cache,
    const DataT* G_S_cache, const DataT* G_M_cache,
    const DataT* d_output,
    DataT* d_x,
    DataT* d_W_kvqm,
    DataT* d_B_S, DataT* d_B_M,
    DataT* d_kvqm_cache,
    float* d_B_S_accum, float* d_B_M_accum
) {
    int T = steps;
    int B = batch_size_;
    int n = n_state_;
    int d = dim_;
    int checkpoint_interval = E80_CHECKPOINT_INTERVAL;

    const float alpha = 1.0f, beta_zero = 0.0f;
    int n2 = n * n;

    // Zero accumulators
    cudaMemsetAsync(d_B_S_accum, 0, n2 * sizeof(float), stream_);
    cudaMemsetAsync(d_B_M_accum, 0, n2 * sizeof(float), stream_);

    if constexpr (std::is_same<DataT, __nv_bfloat16>::value) {
        dispatch_e80_full_rank_gate_backward(T, B, n, kvqm_cache, B_S, B_M,
                                             G_S_cache, G_M_cache,
                                             S_checkpoints, M_checkpoints, Sq_cache, d_output,
                                             d_kvqm_cache, d_B_S_accum, d_B_M_accum,
                                             checkpoint_interval, stream_);
    } else {
        dispatch_e80_full_rank_gate_backward_fp32(T, B, n,
                                                   reinterpret_cast<const float*>(kvqm_cache),
                                                   reinterpret_cast<const float*>(B_S),
                                                   reinterpret_cast<const float*>(B_M),
                                                   reinterpret_cast<const float*>(G_S_cache),
                                                   reinterpret_cast<const float*>(G_M_cache),
                                                   reinterpret_cast<const float*>(S_checkpoints),
                                                   reinterpret_cast<const float*>(M_checkpoints),
                                                   reinterpret_cast<const float*>(Sq_cache),
                                                   reinterpret_cast<const float*>(d_output),
                                                   reinterpret_cast<float*>(d_kvqm_cache),
                                                   d_B_S_accum, d_B_M_accum,
                                                   checkpoint_interval, stream_);
    }

    cudaDataType_t data_type = std::is_same<DataT, __nv_bfloat16>::value ? CUDA_R_16BF : CUDA_R_32F;

    // d_x = W_kvqm @ d_kvqm_cache
    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                 d, T * B, 4 * n,
                 &alpha,
                 W_kvqm, data_type, d,
                 d_kvqm_cache, data_type, 4 * n,
                 &beta_zero,
                 d_x, data_type, d,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // d_W_kvqm = x @ d_kvqm_cache^T
    // Forward was: kvqm = W^T @ x, where W is stored as [d, 4*n] column-major
    // So: d(W^T) = d_kvqm @ x^T, equivalently dW = x @ d_kvqm^T
    // We want dW with shape [d, 4*n] column-major (same as W)
    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
                 d, 4 * n, T * B,  // M=d, N=4*n, K=T*B
                 &alpha,
                 x, data_type, d,  // A[d, T*B]
                 d_kvqm_cache, data_type, 4 * n,  // B^T: B is [4*n, T*B], B^T is [T*B, 4*n]
                 &beta_zero,
                 d_W_kvqm, data_type, d,  // C[d, 4*n]
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // Copy accumulated bias gradients to output (convert float -> DataT if needed)
    if constexpr (std::is_same<DataT, __nv_bfloat16>::value) {
        // Convert float accumulators to bfloat16
        __nv_bfloat16* d_B_S_bf16 = d_B_S;
        __nv_bfloat16* d_B_M_bf16 = d_B_M;
        // Simple kernel to convert and copy - run on GPU
        // For simplicity, using cudaMemcpy with conversion on host side
        // A better approach would be a simple conversion kernel
        std::vector<float> h_B_S_accum(n2);
        std::vector<float> h_B_M_accum(n2);
        cudaMemcpyAsync(h_B_S_accum.data(), d_B_S_accum, n2 * sizeof(float), cudaMemcpyDeviceToHost, stream_);
        cudaMemcpyAsync(h_B_M_accum.data(), d_B_M_accum, n2 * sizeof(float), cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
        std::vector<__nv_bfloat16> h_B_S_bf16(n2);
        std::vector<__nv_bfloat16> h_B_M_bf16(n2);
        for (int i = 0; i < n2; i++) {
            h_B_S_bf16[i] = __float2bfloat16(h_B_S_accum[i]);
            h_B_M_bf16[i] = __float2bfloat16(h_B_M_accum[i]);
        }
        cudaMemcpyAsync(d_B_S_bf16, h_B_S_bf16.data(), n2 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(d_B_M_bf16, h_B_M_bf16.data(), n2 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice, stream_);
    } else {
        // FP32 - just copy directly
        cudaMemcpyAsync(d_B_S, d_B_S_accum, n2 * sizeof(float), cudaMemcpyDeviceToDevice, stream_);
        cudaMemcpyAsync(d_B_M, d_B_M_accum, n2 * sizeof(float), cudaMemcpyDeviceToDevice, stream_);
    }
}

// Explicit template instantiations
template class E80FullRankGateForward<__nv_bfloat16>;
template class E80FullRankGateForward<float>;
template class E80FullRankGateBackward<__nv_bfloat16>;
template class E80FullRankGateBackward<float>;

}  // namespace elman
