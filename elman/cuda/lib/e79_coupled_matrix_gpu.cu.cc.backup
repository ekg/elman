/**
 * E79 Coupled Memory-Modulation Matrix System CUDA Kernel
 *
 * Two coupled matrix states that mutually control each other's evolution:
 * - S [n_state x n_state]: Content memory - stores key-value associations
 * - M [n_state x n_state]: Modulation memory - controls how S updates
 *
 * Key insight: Self-modulation through mutual coupling.
 * - M controls S's row/col decay (what S forgets)
 * - S controls M's row/col decay (what M forgets)
 *
 * Architecture:
 *   FUSED projection: kvqm = W_kvqm @ x  (k, v, q, m)
 *
 *   # S update (M-controlled gating)
 *   s_row_decay = sigmoid(M @ k_norm + b_s_gate)
 *   s_col_decay = sigmoid(M.T @ k_norm + b_s_gate)
 *   s_delta = v - S @ k_norm
 *   S = (s_row_decay[:, None] * S * s_col_decay[None, :]) + outer(s_delta, k_norm)
 *
 *   # M update (S-controlled gating)
 *   m_row_decay = sigmoid(S @ m_norm + b_m_gate)
 *   m_col_decay = sigmoid(S.T @ m_norm + b_m_gate)
 *   m_delta = s_delta - M @ m_norm
 *   M = (m_row_decay[:, None] * M * m_col_decay[None, :]) + outer(m_delta, m_norm)
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
#include "hasty/elman_ladder.h"

#define E79_CHECKPOINT_INTERVAL 16

namespace elman {

// ============================================================================
// E79 Forward Kernel - Coupled Memory-Modulation System
// ============================================================================

template<int N_STATE>
__global__ void E79CoupledForwardKernel_BF16(
    int T,
    int B,
    const __nv_bfloat16* __restrict__ kvqm_all,   // [4*N_STATE, T*B] column-major
    const __nv_bfloat16* __restrict__ b_s_gate,   // [N_STATE] S gate bias
    const __nv_bfloat16* __restrict__ b_m_gate,   // [N_STATE] M gate bias
    __nv_bfloat16* __restrict__ S,                // [B, N_STATE, N_STATE]
    __nv_bfloat16* __restrict__ M,                // [B, N_STATE, N_STATE]
    __nv_bfloat16* __restrict__ output,           // [T, B, N_STATE]
    __nv_bfloat16* __restrict__ S_checkpoints,    // [num_checkpoints, B, N_STATE, N_STATE]
    __nv_bfloat16* __restrict__ M_checkpoints,    // [num_checkpoints, B, N_STATE, N_STATE]
    __nv_bfloat16* __restrict__ Sq_cache,         // [T, B, N_STATE]
    __nv_bfloat16* __restrict__ s_row_decay_cache,// [T, B, N_STATE]
    __nv_bfloat16* __restrict__ s_col_decay_cache,// [T, B, N_STATE]
    __nv_bfloat16* __restrict__ m_row_decay_cache,// [T, B, N_STATE]
    __nv_bfloat16* __restrict__ m_col_decay_cache,// [T, B, N_STATE]
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
    float* s_row_decay = m_vec_shared + N_STATE;          // [N_STATE]
    float* s_col_decay = s_row_decay + N_STATE;           // [N_STATE]
    float* m_row_decay = s_col_decay + N_STATE;           // [N_STATE]
    float* m_col_decay = m_row_decay + N_STATE;           // [N_STATE]
    float* s_retrieved = m_col_decay + N_STATE;           // [N_STATE]
    float* m_retrieved = s_retrieved + N_STATE;           // [N_STATE]
    float* b_s_gate_shared = m_retrieved + N_STATE;       // [N_STATE]
    float* b_m_gate_shared = b_s_gate_shared + N_STATE;   // [N_STATE]

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;
    const int STRIDE = 4 * N_STATE;

    // Load gate biases
    if (tid < N_STATE) {
        b_s_gate_shared[tid] = __bfloat162float(b_s_gate[tid]);
        b_m_gate_shared[tid] = __bfloat162float(b_m_gate[tid]);
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

        // --- S update (M-controlled gating) ---
        // s_row_decay = sigmoid(M @ k_norm + b_s_gate)
        // s_col_decay = sigmoid(M.T @ k_norm + b_s_gate)
        if (tid < N_STATE) {
            float row_sum = 0.0f, col_sum = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < N_STATE; j++) {
                row_sum += M_shared[tid * N_STATE + j] * k_shared[j];  // M @ k
                col_sum += M_shared[j * N_STATE + tid] * k_shared[j];  // M.T @ k
            }
            s_row_decay[tid] = 1.0f / (1.0f + expf(-(row_sum + b_s_gate_shared[tid])));
            s_col_decay[tid] = 1.0f / (1.0f + expf(-(col_sum + b_s_gate_shared[tid])));

            // Cache for backward
            s_row_decay_cache[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(s_row_decay[tid]);
            s_col_decay_cache[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(s_col_decay[tid]);
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

        // Update S: S = (s_row_decay * S * s_col_decay) + outer(s_delta, k_norm)
        for (int i = tid; i < n2; i += blockDim.x) {
            int row = i / N_STATE;
            int col = i % N_STATE;
            float s_delta_row = v_shared[row] - s_retrieved[row];
            float update = s_row_decay[row] * S_shared[i] * s_col_decay[col] + s_delta_row * k_shared[col];
            S_shared[i] = update;
        }
        __syncthreads();

        // --- M update (S-controlled gating) ---
        // m_row_decay = sigmoid(S @ m_norm + b_m_gate)
        // m_col_decay = sigmoid(S.T @ m_norm + b_m_gate)
        if (tid < N_STATE) {
            float row_sum = 0.0f, col_sum = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < N_STATE; j++) {
                row_sum += S_shared[tid * N_STATE + j] * m_vec_shared[j];  // S @ m (after S update!)
                col_sum += S_shared[j * N_STATE + tid] * m_vec_shared[j];  // S.T @ m
            }
            m_row_decay[tid] = 1.0f / (1.0f + expf(-(row_sum + b_m_gate_shared[tid])));
            m_col_decay[tid] = 1.0f / (1.0f + expf(-(col_sum + b_m_gate_shared[tid])));

            // Cache for backward
            m_row_decay_cache[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(m_row_decay[tid]);
            m_col_decay_cache[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(m_col_decay[tid]);
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

        // Update M: M = (m_row_decay * M * m_col_decay) + outer(m_delta, m_norm)
        // m_delta = s_delta - m_retrieved (M tries to predict S's delta)
        for (int i = tid; i < n2; i += blockDim.x) {
            int row = i / N_STATE;
            int col = i % N_STATE;
            float s_delta_row = v_shared[row] - (s_retrieved[row] / (s_row_decay[row] * s_col_decay[col] + 1e-8f));
            // Actually, we need s_delta from before S update. Let's recompute:
            // s_delta was v - S_old @ k_norm. After S update, S_old is gone.
            // We need to cache s_delta. For now, approximate with current s_retrieved.
            // Actually, let's use: m_delta = (v - s_retrieved_before_update) - m_retrieved
            // But we already updated S. Let me reconsider...
            // The Python code does: m_delta = s_delta - m_retrieved
            // where s_delta = v - S_old @ k_norm (BEFORE S update)
            // So we need s_retrieved BEFORE S update.
            // We computed s_retrieved before updating S, so we can use it.
        }

        // Let me redo M update correctly - need s_delta from before S update
        // s_delta[row] = v[row] - s_retrieved[row] (computed before S update)
        if (tid < N_STATE) {
            // s_retrieved still holds pre-update value
            float s_delta_row = v_shared[tid] - s_retrieved[tid];
            float m_delta_row = s_delta_row - m_retrieved[tid];

            // Now update M
            for (int col = 0; col < N_STATE; col++) {
                int idx = tid * N_STATE + col;
                M_shared[idx] = m_row_decay[tid] * M_shared[idx] * m_col_decay[col] + m_delta_row * m_vec_shared[col];
            }
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
// E79 Backward Kernel - Coupled Memory-Modulation System
// ============================================================================

template<int N_STATE>
__global__ void E79CoupledBackwardKernel_BF16(
    int T,
    int B,
    const __nv_bfloat16* __restrict__ kvqm_all,
    const __nv_bfloat16* __restrict__ b_s_gate,
    const __nv_bfloat16* __restrict__ b_m_gate,
    const __nv_bfloat16* __restrict__ s_row_decay_cache,
    const __nv_bfloat16* __restrict__ s_col_decay_cache,
    const __nv_bfloat16* __restrict__ m_row_decay_cache,
    const __nv_bfloat16* __restrict__ m_col_decay_cache,
    const __nv_bfloat16* __restrict__ S_checkpoints,
    const __nv_bfloat16* __restrict__ M_checkpoints,
    const __nv_bfloat16* __restrict__ Sq_cache,
    const __nv_bfloat16* __restrict__ d_output,
    __nv_bfloat16* __restrict__ d_kvqm_all,
    float* __restrict__ d_b_s_gate_accum,
    float* __restrict__ d_b_m_gate_accum,
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
    float* s_row_decay = m_norm + N_STATE;                // [N_STATE]
    float* s_col_decay = s_row_decay + N_STATE;           // [N_STATE]
    float* m_row_decay = s_col_decay + N_STATE;           // [N_STATE]
    float* m_col_decay = m_row_decay + N_STATE;           // [N_STATE]
    float* s_retrieved = m_col_decay + N_STATE;           // [N_STATE]
    float* m_retrieved = s_retrieved + N_STATE;           // [N_STATE]
    float* s_delta = m_retrieved + N_STATE;               // [N_STATE]
    float* m_delta = s_delta + N_STATE;                   // [N_STATE]
    float* b_s_gate_shared = m_delta + N_STATE;           // [N_STATE]
    float* b_m_gate_shared = b_s_gate_shared + N_STATE;   // [N_STATE]
    float* d_k_raw = b_m_gate_shared + N_STATE;           // [N_STATE]
    float* d_v_raw = d_k_raw + N_STATE;                   // [N_STATE]
    float* d_q_raw = d_v_raw + N_STATE;                   // [N_STATE]
    float* d_m_raw = d_q_raw + N_STATE;                   // [N_STATE]
    float* d_Sq_shared = d_m_raw + N_STATE;               // [N_STATE]
    float* d_s_delta = d_Sq_shared + N_STATE;             // [N_STATE]
    float* d_m_delta = d_s_delta + N_STATE;               // [N_STATE]
    float* d_k_norm = d_m_delta + N_STATE;                // [N_STATE]
    float* d_m_norm = d_k_norm + N_STATE;                 // [N_STATE]
    float* d_s_row_decay = d_m_norm + N_STATE;            // [N_STATE]
    float* d_s_col_decay = d_s_row_decay + N_STATE;       // [N_STATE]
    float* d_m_row_decay = d_s_col_decay + N_STATE;       // [N_STATE]
    float* d_m_col_decay = d_m_row_decay + N_STATE;       // [N_STATE]
    float* d_b_s_gate_local = d_m_col_decay + N_STATE;    // [N_STATE]
    float* d_b_m_gate_local = d_b_s_gate_local + N_STATE; // [N_STATE]

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;
    const int STRIDE = 4 * N_STATE;

    // Load gate biases
    if (tid < N_STATE) {
        b_s_gate_shared[tid] = __bfloat162float(b_s_gate[tid]);
        b_m_gate_shared[tid] = __bfloat162float(b_m_gate[tid]);
        d_b_s_gate_local[tid] = 0.0f;
        d_b_m_gate_local[tid] = 0.0f;
    }
    __syncthreads();

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

            // Recompute forward to step t
            __shared__ float k_norm_val_t, m_norm_val_t;
            for (int tt = t_start; tt <= t; tt++) {
                int col_idx = (tt * B + b) * STRIDE;
                if (tid < N_STATE) {
                    k_raw[tid] = __bfloat162float(kvqm_all[col_idx + tid]);
                    v_raw[tid] = __bfloat162float(kvqm_all[col_idx + N_STATE + tid]);
                    q_raw[tid] = __bfloat162float(kvqm_all[col_idx + 2 * N_STATE + tid]);
                    m_vec_raw[tid] = __bfloat162float(kvqm_all[col_idx + 3 * N_STATE + tid]);
                    s_row_decay[tid] = __bfloat162float(s_row_decay_cache[tt * B * N_STATE + b * N_STATE + tid]);
                    s_col_decay[tid] = __bfloat162float(s_col_decay_cache[tt * B * N_STATE + b * N_STATE + tid]);
                    m_row_decay[tid] = __bfloat162float(m_row_decay_cache[tt * B * N_STATE + b * N_STATE + tid]);
                    m_col_decay[tid] = __bfloat162float(m_col_decay_cache[tt * B * N_STATE + b * N_STATE + tid]);
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

                // Compute s_retrieved and s_delta BEFORE S update
                if (tid < N_STATE) {
                    float sum = 0.0f;
                    for (int j = 0; j < N_STATE; j++) {
                        sum += S[tid * N_STATE + j] * k_norm[j];
                    }
                    s_retrieved[tid] = sum;
                    s_delta[tid] = v_raw[tid] - s_retrieved[tid];
                }
                __syncthreads();

                // Update S if not at target step
                if (tt < t) {
                    for (int i = tid; i < n2; i += blockDim.x) {
                        int row = i / N_STATE;
                        int col = i % N_STATE;
                        S[i] = s_row_decay[row] * S[i] * s_col_decay[col] + s_delta[row] * k_norm[col];
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
                        M[i] = m_row_decay[row] * M[i] * m_col_decay[col] + m_delta[row] * m_norm[col];
                    }
                    __syncthreads();
                } else {
                    // At target step - compute m_retrieved and m_delta for backward
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
            }

            // Now S, M hold states at t-1; s_delta, m_delta are for step t

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

            // d_q = S_t^T @ d_Sq
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    float S_t_ij = s_row_decay[i] * S[i * N_STATE + tid] * s_col_decay[tid] + s_delta[i] * k_norm[tid];
                    sum += S_t_ij * d_Sq_shared[i];
                }
                d_q_raw[tid] = sum;
            }
            __syncthreads();

            // Backward through M update: M_new = m_row_decay * M * m_col_decay + outer(m_delta, m_norm)
            // dM_old = dM_new * m_row_decay * m_col_decay
            // d_m_delta = dM_new @ m_norm
            // d_m_norm = dM_new.T @ m_delta
            // d_m_row_decay = sum_j(dM[i,j] * M[i,j] * m_col_decay[j])
            // d_m_col_decay = sum_i(dM[i,j] * M[i,j] * m_row_decay[i])
            if (tid < N_STATE) {
                float d_m_delta_local = 0.0f;
                float d_m_norm_local = 0.0f;
                float d_m_row_local = 0.0f;
                float d_m_col_local = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    float dM_ij = dM[tid * N_STATE + j];
                    d_m_delta_local += dM_ij * m_norm[j];
                    d_m_row_local += dM_ij * M[tid * N_STATE + j] * m_col_decay[j];

                    float dM_ji = dM[j * N_STATE + tid];
                    d_m_norm_local += dM_ji * m_delta[j];
                    d_m_col_local += dM_ji * M[j * N_STATE + tid] * m_row_decay[j];
                }
                d_m_delta[tid] = d_m_delta_local;
                d_m_norm[tid] = d_m_norm_local;
                d_m_row_decay[tid] = d_m_row_local;
                d_m_col_decay[tid] = d_m_col_local;
            }
            __syncthreads();

            // d_m_delta = d_s_delta - d_m_delta (from m_delta = s_delta - m_retrieved)
            // So d_s_delta gets contribution from M update
            // d_m_retrieved = -d_m_delta

            // Backward through S update: S_new = s_row_decay * S * s_col_decay + outer(s_delta, k_norm)
            if (tid < N_STATE) {
                float d_s_delta_local = 0.0f;
                float d_k_norm_local = 0.0f;
                float d_s_row_local = 0.0f;
                float d_s_col_local = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    float dS_ij = dS[tid * N_STATE + j];
                    d_s_delta_local += dS_ij * k_norm[j];
                    d_s_row_local += dS_ij * S[tid * N_STATE + j] * s_col_decay[j];

                    float dS_ji = dS[j * N_STATE + tid];
                    d_k_norm_local += dS_ji * s_delta[j];
                    d_s_col_local += dS_ji * S[j * N_STATE + tid] * s_row_decay[j];
                }
                // Add contribution from m_delta = s_delta - m_retrieved
                d_s_delta_local += d_m_delta[tid];

                d_s_delta[tid] = d_s_delta_local;
                d_k_norm[tid] = d_k_norm_local;
                d_s_row_decay[tid] = d_s_row_local;
                d_s_col_decay[tid] = d_s_col_local;
            }
            __syncthreads();

            // Backward through decay computations (sigmoid derivatives)
            // s_row_decay = sigmoid(M @ k_norm + b_s_gate)
            // d_b_s_gate += (d_s_row_decay + d_s_col_decay) * decay * (1 - decay)
            float d_s_gate_local = 0.0f, d_m_gate_local = 0.0f;
            if (tid < N_STATE) {
                float d_row = d_s_row_decay[tid];
                float d_col = d_s_col_decay[tid];
                float dec_row = s_row_decay[tid];
                float dec_col = s_col_decay[tid];
                d_s_gate_local = d_row * dec_row * (1.0f - dec_row) + d_col * dec_col * (1.0f - dec_col);
                d_b_s_gate_local[tid] += d_s_gate_local;

                float d_m_row = d_m_row_decay[tid];
                float d_m_col = d_m_col_decay[tid];
                float m_dec_row = m_row_decay[tid];
                float m_dec_col = m_col_decay[tid];
                d_m_gate_local = d_m_row * m_dec_row * (1.0f - m_dec_row) + d_m_col * m_dec_col * (1.0f - m_dec_col);
                d_b_m_gate_local[tid] += d_m_gate_local;
            }
            __syncthreads();

            // d_v = d_s_delta
            if (tid < N_STATE) {
                d_v_raw[tid] = d_s_delta[tid];
            }

            // d_k_norm contribution from retrieved: retrieved = S @ k_norm
            // d_k_norm += S^T @ (-d_s_delta)
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    sum += S[i * N_STATE + tid] * (-d_s_delta[i]);
                }
                d_k_norm[tid] += sum;
            }
            __syncthreads();

            // d_m_norm contribution from m_retrieved: m_retrieved = M @ m_norm
            // d_m_norm += M^T @ (-d_m_delta)  [from m_delta = s_delta - m_retrieved]
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    sum += M[i * N_STATE + tid] * d_m_delta[i];  // d_m_retrieved = -d_m_delta, so += d_m_delta
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

            // Update dS for next iteration
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                float d_pre = dS[i];
                dS[i] = d_pre * s_row_decay[row] * s_col_decay[col] + (-d_s_delta[row]) * k_norm[col];
            }
            __syncthreads();

            // Update dM for next iteration
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                float d_pre = dM[i];
                dM[i] = d_pre * m_row_decay[row] * m_col_decay[col] + (-d_m_delta[row]) * m_norm[col];
            }
            __syncthreads();
        }
    }

    // Accumulate bias gradients
    if (tid < N_STATE) {
        atomicAdd(&d_b_s_gate_accum[tid], d_b_s_gate_local[tid]);
        atomicAdd(&d_b_m_gate_accum[tid], d_b_m_gate_local[tid]);
    }
}

// ============================================================================
// Global Memory Fallback Kernels for Large N_STATE
// Uses global memory for state matrices (S, M, dS, dM) to bypass shared memory limits.
// Only vectors are kept in shared memory (~34*N floats = 13KB for N=96).
// ============================================================================

template<int N_STATE>
__global__ void E79CoupledForwardGlobalMemKernel_BF16(
    int T,
    int B,
    const __nv_bfloat16* __restrict__ kvqm_all,
    const __nv_bfloat16* __restrict__ b_s_gate,
    const __nv_bfloat16* __restrict__ b_m_gate,
    __nv_bfloat16* __restrict__ S_out,
    __nv_bfloat16* __restrict__ M_out,
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ S_checkpoints,
    __nv_bfloat16* __restrict__ M_checkpoints,
    __nv_bfloat16* __restrict__ Sq_cache,
    __nv_bfloat16* __restrict__ s_row_decay_cache,
    __nv_bfloat16* __restrict__ s_col_decay_cache,
    __nv_bfloat16* __restrict__ m_row_decay_cache,
    __nv_bfloat16* __restrict__ m_col_decay_cache,
    float* __restrict__ state_workspace,  // [B, 2, N_STATE, N_STATE] for S and M
    int checkpoint_interval
) {
    int b = blockIdx.x;
    if (b >= B) return;

    int n2 = N_STATE * N_STATE;
    const int STRIDE = 4 * N_STATE;

    // State matrices in global memory
    float* S = state_workspace + b * 2 * n2;
    float* M = S + n2;

    // Only vectors in shared memory (14 * N_STATE floats)
    extern __shared__ float shared_mem[];
    float* k_shared = shared_mem;
    float* v_shared = k_shared + N_STATE;
    float* q_shared = v_shared + N_STATE;
    float* m_vec_shared = q_shared + N_STATE;
    float* s_row_decay = m_vec_shared + N_STATE;
    float* s_col_decay = s_row_decay + N_STATE;
    float* m_row_decay = s_col_decay + N_STATE;
    float* m_col_decay = m_row_decay + N_STATE;
    float* s_retrieved = m_col_decay + N_STATE;
    float* m_retrieved = s_retrieved + N_STATE;
    float* b_s_gate_shared = m_retrieved + N_STATE;
    float* b_m_gate_shared = b_s_gate_shared + N_STATE;
    float* Sq_shared = b_m_gate_shared + N_STATE;
    float* s_delta_shared = Sq_shared + N_STATE;

    int tid = threadIdx.x;

    // Load gate biases
    if (tid < N_STATE) {
        b_s_gate_shared[tid] = __bfloat162float(b_s_gate[tid]);
        b_m_gate_shared[tid] = __bfloat162float(b_m_gate[tid]);
    }
    __syncthreads();

    // Load initial states from input to global memory workspace
    for (int i = tid; i < n2; i += blockDim.x) {
        S[i] = __bfloat162float(S_out[b * n2 + i]);
        M[i] = __bfloat162float(M_out[b * n2 + i]);
    }
    __syncthreads();

    // Save initial checkpoints
    for (int i = tid; i < n2; i += blockDim.x) {
        S_checkpoints[b * n2 + i] = __float2bfloat16(S[i]);
        M_checkpoints[b * n2 + i] = __float2bfloat16(M[i]);
    }
    __syncthreads();

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

        // --- S update (M-controlled gating) ---
        // s_row_decay = sigmoid(M @ k_norm + b_s_gate)
        // s_col_decay = sigmoid(M.T @ k_norm + b_s_gate)
        if (tid < N_STATE) {
            float row_sum = 0.0f, col_sum = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < N_STATE; j++) {
                row_sum += M[tid * N_STATE + j] * k_shared[j];
                col_sum += M[j * N_STATE + tid] * k_shared[j];
            }
            s_row_decay[tid] = 1.0f / (1.0f + expf(-(row_sum + b_s_gate_shared[tid])));
            s_col_decay[tid] = 1.0f / (1.0f + expf(-(col_sum + b_s_gate_shared[tid])));
            s_row_decay_cache[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(s_row_decay[tid]);
            s_col_decay_cache[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(s_col_decay[tid]);
        }
        __syncthreads();

        // s_retrieved = S @ k_norm, s_delta = v - s_retrieved
        if (tid < N_STATE) {
            float sum = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < N_STATE; j++) {
                sum += S[tid * N_STATE + j] * k_shared[j];
            }
            s_retrieved[tid] = sum;
            s_delta_shared[tid] = v_shared[tid] - sum;
        }
        __syncthreads();

        // Update S in global memory
        for (int i = tid; i < n2; i += blockDim.x) {
            int row = i / N_STATE;
            int col = i % N_STATE;
            S[i] = s_row_decay[row] * S[i] * s_col_decay[col] + s_delta_shared[row] * k_shared[col];
        }
        __syncthreads();

        // --- M update (S-controlled gating) ---
        if (tid < N_STATE) {
            float row_sum = 0.0f, col_sum = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < N_STATE; j++) {
                row_sum += S[tid * N_STATE + j] * m_vec_shared[j];
                col_sum += S[j * N_STATE + tid] * m_vec_shared[j];
            }
            m_row_decay[tid] = 1.0f / (1.0f + expf(-(row_sum + b_m_gate_shared[tid])));
            m_col_decay[tid] = 1.0f / (1.0f + expf(-(col_sum + b_m_gate_shared[tid])));
            m_row_decay_cache[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(m_row_decay[tid]);
            m_col_decay_cache[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(m_col_decay[tid]);
        }
        __syncthreads();

        // m_retrieved = M @ m_norm, m_delta = s_delta - m_retrieved
        if (tid < N_STATE) {
            float sum = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < N_STATE; j++) {
                sum += M[tid * N_STATE + j] * m_vec_shared[j];
            }
            m_retrieved[tid] = s_delta_shared[tid] - sum;  // m_delta stored in m_retrieved
        }
        __syncthreads();

        // Update M in global memory
        for (int i = tid; i < n2; i += blockDim.x) {
            int row = i / N_STATE;
            int col = i % N_STATE;
            M[i] = m_row_decay[row] * M[i] * m_col_decay[col] + m_retrieved[row] * m_vec_shared[col];
        }
        __syncthreads();

        // --- Output ---
        // Sq = S @ q, output = Sq * silu(Sq)
        if (tid < N_STATE) {
            float sum = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < N_STATE; j++) {
                sum += S[tid * N_STATE + j] * q_shared[j];
            }
            Sq_shared[tid] = sum;
            float sigmoid_Sq = 1.0f / (1.0f + expf(-sum));
            float silu_Sq = sum * sigmoid_Sq;
            output[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(sum * silu_Sq);
            Sq_cache[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(sum);
        }
        __syncthreads();

        // Save checkpoints
        if ((t + 1) % checkpoint_interval == 0) {
            int cp_idx = (t + 1) / checkpoint_interval;
            for (int i = tid; i < n2; i += blockDim.x) {
                S_checkpoints[cp_idx * B * n2 + b * n2 + i] = __float2bfloat16(S[i]);
                M_checkpoints[cp_idx * B * n2 + b * n2 + i] = __float2bfloat16(M[i]);
            }
            __syncthreads();
        }
    }

    // Write final state back
    for (int i = tid; i < n2; i += blockDim.x) {
        S_out[b * n2 + i] = __float2bfloat16(S[i]);
        M_out[b * n2 + i] = __float2bfloat16(M[i]);
    }
}

template<int N_STATE>
__global__ void E79CoupledBackwardGlobalMemKernel_BF16(
    int T,
    int B,
    const __nv_bfloat16* __restrict__ kvqm_all,
    const __nv_bfloat16* __restrict__ b_s_gate,
    const __nv_bfloat16* __restrict__ b_m_gate,
    const __nv_bfloat16* __restrict__ s_row_decay_cache,
    const __nv_bfloat16* __restrict__ s_col_decay_cache,
    const __nv_bfloat16* __restrict__ m_row_decay_cache,
    const __nv_bfloat16* __restrict__ m_col_decay_cache,
    const __nv_bfloat16* __restrict__ S_checkpoints,
    const __nv_bfloat16* __restrict__ M_checkpoints,
    const __nv_bfloat16* __restrict__ Sq_cache,
    const __nv_bfloat16* __restrict__ d_output,
    __nv_bfloat16* __restrict__ d_kvqm_all,
    float* __restrict__ d_b_s_gate_accum,
    float* __restrict__ d_b_m_gate_accum,
    float* __restrict__ state_workspace,  // [B, 4, N_STATE, N_STATE] for S, M, dS, dM
    int checkpoint_interval
) {
    int b = blockIdx.x;
    if (b >= B) return;

    int n2 = N_STATE * N_STATE;
    const int STRIDE = 4 * N_STATE;

    // State matrices in global memory (4 matrices)
    float* S = state_workspace + b * 4 * n2;
    float* M = S + n2;
    float* dS = M + n2;
    float* dM = dS + n2;

    // Only vectors in shared memory (34 * N_STATE floats)
    extern __shared__ float shared_mem[];
    float* k_raw = shared_mem;
    float* v_raw = k_raw + N_STATE;
    float* q_raw = v_raw + N_STATE;
    float* m_vec_raw = q_raw + N_STATE;
    float* k_norm = m_vec_raw + N_STATE;
    float* m_norm = k_norm + N_STATE;
    float* s_row_decay = m_norm + N_STATE;
    float* s_col_decay = s_row_decay + N_STATE;
    float* m_row_decay = s_col_decay + N_STATE;
    float* m_col_decay = m_row_decay + N_STATE;
    float* s_retrieved = m_col_decay + N_STATE;
    float* m_retrieved = s_retrieved + N_STATE;
    float* s_delta = m_retrieved + N_STATE;
    float* m_delta = s_delta + N_STATE;
    float* b_s_gate_shared = m_delta + N_STATE;
    float* b_m_gate_shared = b_s_gate_shared + N_STATE;
    float* d_k_raw = b_m_gate_shared + N_STATE;
    float* d_v_raw = d_k_raw + N_STATE;
    float* d_q_raw = d_v_raw + N_STATE;
    float* d_m_raw = d_q_raw + N_STATE;
    float* d_Sq_shared = d_m_raw + N_STATE;
    float* d_s_delta = d_Sq_shared + N_STATE;
    float* d_m_delta = d_s_delta + N_STATE;
    float* d_k_norm = d_m_delta + N_STATE;
    float* d_m_norm = d_k_norm + N_STATE;
    float* d_s_row_decay = d_m_norm + N_STATE;
    float* d_s_col_decay = d_s_row_decay + N_STATE;
    float* d_m_row_decay = d_s_col_decay + N_STATE;
    float* d_m_col_decay = d_m_row_decay + N_STATE;
    float* d_b_s_gate_local = d_m_col_decay + N_STATE;
    float* d_b_m_gate_local = d_b_s_gate_local + N_STATE;

    int tid = threadIdx.x;

    // Load gate biases
    if (tid < N_STATE) {
        b_s_gate_shared[tid] = __bfloat162float(b_s_gate[tid]);
        b_m_gate_shared[tid] = __bfloat162float(b_m_gate[tid]);
        d_b_s_gate_local[tid] = 0.0f;
        d_b_m_gate_local[tid] = 0.0f;
    }
    __syncthreads();

    // Initialize gradient accumulators in global memory
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
            // Reload checkpoint to global memory
            for (int i = tid; i < n2; i += blockDim.x) {
                S[i] = __bfloat162float(S_checkpoints[seg * B * n2 + b * n2 + i]);
                M[i] = __bfloat162float(M_checkpoints[seg * B * n2 + b * n2 + i]);
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
                    s_row_decay[tid] = __bfloat162float(s_row_decay_cache[tt * B * N_STATE + b * N_STATE + tid]);
                    s_col_decay[tid] = __bfloat162float(s_col_decay_cache[tt * B * N_STATE + b * N_STATE + tid]);
                    m_row_decay[tid] = __bfloat162float(m_row_decay_cache[tt * B * N_STATE + b * N_STATE + tid]);
                    m_col_decay[tid] = __bfloat162float(m_col_decay_cache[tt * B * N_STATE + b * N_STATE + tid]);
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

                // Compute s_retrieved and s_delta BEFORE S update
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
                        S[i] = s_row_decay[row] * S[i] * s_col_decay[col] + s_delta[row] * k_norm[col];
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
                        M[i] = m_row_decay[row] * M[i] * m_col_decay[col] + m_delta[row] * m_norm[col];
                    }
                    __syncthreads();
                } else {
                    // At target step - compute m_retrieved and m_delta for backward
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
            }

            // === BACKWARD PASS FOR STEP t ===
            // Load d_output for step t
            if (tid < N_STATE) {
                float d_out = __bfloat162float(d_output[t * B * N_STATE + b * N_STATE + tid]);
                float Sq_t = __bfloat162float(Sq_cache[t * B * N_STATE + b * N_STATE + tid]);
                // d_out = d(Sq * silu(Sq))
                // d_Sq = d_out * (silu(Sq) + Sq * silu'(Sq))
                //      = d_out * (Sq * sigmoid(Sq) + Sq * sigmoid(Sq) * (1 - sigmoid(Sq)))
                //      = d_out * Sq * sigmoid(Sq) * (2 - sigmoid(Sq))
                float sigmoid_Sq = 1.0f / (1.0f + expf(-Sq_t));
                float silu_Sq = Sq_t * sigmoid_Sq;
                float d_Sq_val = d_out * (silu_Sq + Sq_t * sigmoid_Sq * (1.0f - sigmoid_Sq));
                d_Sq_shared[tid] = d_Sq_val;
            }
            __syncthreads();

            // d_q from Sq = S @ q
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    sum += S[j * N_STATE + tid] * d_Sq_shared[j];  // S.T @ d_Sq
                }
                d_q_raw[tid] = sum;
            }
            __syncthreads();

            // dS += outer(d_Sq, q)
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                dS[i] += d_Sq_shared[row] * q_raw[col];
            }
            __syncthreads();

            // --- M backward ---
            // From M update: M_new = m_row_decay * M * m_col_decay + outer(m_delta, m_norm)
            // dM_prev = dM * m_row_decay * m_col_decay
            // d_m_delta = dM @ m_norm
            // d_m_norm = dM.T @ m_delta
            // d_m_row_decay = sum_j(dM[i,j] * M[i,j] * m_col_decay[j])
            // d_m_col_decay = sum_i(dM[i,j] * M[i,j] * m_row_decay[i])
            if (tid < N_STATE) {
                float d_mdt = 0.0f;
                float d_mn = 0.0f;
                float d_mrd = 0.0f;
                float d_mcd = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    float dM_ij = dM[tid * N_STATE + j];
                    float M_ij = M[tid * N_STATE + j];
                    d_mdt += dM_ij * m_norm[j];
                    d_mn += dM[j * N_STATE + tid] * m_delta[j];
                    d_mrd += dM_ij * M_ij * m_col_decay[j];
                    d_mcd += dM[j * N_STATE + tid] * M[j * N_STATE + tid] * m_row_decay[j];
                }
                d_m_delta[tid] = d_mdt;
                d_m_norm[tid] = d_mn;
                d_m_row_decay[tid] = d_mrd;
                d_m_col_decay[tid] = d_mcd;

                // Bias gradients from decay gates
                float m_gate_row = m_row_decay[tid] * (1.0f - m_row_decay[tid]);
                float m_gate_col = m_col_decay[tid] * (1.0f - m_col_decay[tid]);
                d_b_m_gate_local[tid] += d_mrd * m_gate_row + d_mcd * m_gate_col;
            }
            __syncthreads();

            // d_m_delta flows to d_s_delta and d_m_retrieved
            // m_delta = s_delta - m_retrieved
            // d_s_delta += d_m_delta
            // d_m_retrieved = -d_m_delta
            // d_m_retrieved flows to dM (from M @ m_norm)

            // --- S backward ---
            // From S update: S_new = s_row_decay * S * s_col_decay + outer(s_delta, k_norm)
            if (tid < N_STATE) {
                float d_sdt = 0.0f;
                float d_kn = 0.0f;
                float d_srd = 0.0f;
                float d_scd = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    float dS_ij = dS[tid * N_STATE + j];
                    float S_ij = S[tid * N_STATE + j];
                    d_sdt += dS_ij * k_norm[j];
                    d_kn += dS[j * N_STATE + tid] * s_delta[j];
                    d_srd += dS_ij * S_ij * s_col_decay[j];
                    d_scd += dS[j * N_STATE + tid] * S[j * N_STATE + tid] * s_row_decay[j];
                }
                // Add contribution from M backward (m_delta = s_delta - M @ m_norm)
                d_s_delta[tid] = d_sdt + d_m_delta[tid];
                d_k_norm[tid] = d_kn;
                d_s_row_decay[tid] = d_srd;
                d_s_col_decay[tid] = d_scd;

                // Bias gradients from decay gates
                float s_gate_row = s_row_decay[tid] * (1.0f - s_row_decay[tid]);
                float s_gate_col = s_col_decay[tid] * (1.0f - s_col_decay[tid]);
                d_b_s_gate_local[tid] += d_srd * s_gate_row + d_scd * s_gate_col;
            }
            __syncthreads();

            // m_retrieved gradient contribution to dM is handled in the state propagation step below
            __syncthreads();

            // d_s_delta -> d_v and d_s_retrieved
            // s_delta = v - s_retrieved
            // d_v = d_s_delta
            // d_s_retrieved = -d_s_delta
            if (tid < N_STATE) {
                d_v_raw[tid] = d_s_delta[tid];
            }
            __syncthreads();

            // s_retrieved gradient contribution to dS is handled in the state propagation step below
            __syncthreads();

            // Add contribution from S @ k_norm to d_k_norm
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    sum += S[j * N_STATE + tid] * (-d_s_delta[j]);
                }
                d_k_norm[tid] += sum;
            }
            __syncthreads();

            // d_m_norm contribution from m_retrieved: m_retrieved = M @ m_norm
            // d_m_norm += M^T @ (-d_m_delta)  [from m_delta = s_delta - m_retrieved]
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    sum += M[j * N_STATE + tid] * d_m_delta[j];  // d_m_retrieved = -d_m_delta, so += d_m_delta
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

            // Update dS for next iteration
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                float d_pre = dS[i];
                dS[i] = d_pre * s_row_decay[row] * s_col_decay[col] + (-d_s_delta[row]) * k_norm[col];
            }
            __syncthreads();

            // Update dM for next iteration
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                float d_pre = dM[i];
                dM[i] = d_pre * m_row_decay[row] * m_col_decay[col] + (-d_m_delta[row]) * m_norm[col];
            }
            __syncthreads();
        }
    }

    // Accumulate bias gradients
    if (tid < N_STATE) {
        atomicAdd(&d_b_s_gate_accum[tid], d_b_s_gate_local[tid]);
        atomicAdd(&d_b_m_gate_accum[tid], d_b_m_gate_local[tid]);
    }
}

// ============================================================================
// FP32 versions
// ============================================================================

template<int N_STATE>
__global__ void E79CoupledForwardKernel_FP32(
    int T,
    int B,
    const float* __restrict__ kvqm_all,
    const float* __restrict__ b_s_gate,
    const float* __restrict__ b_m_gate,
    float* __restrict__ S,
    float* __restrict__ M,
    float* __restrict__ output,
    float* __restrict__ S_checkpoints,
    float* __restrict__ M_checkpoints,
    float* __restrict__ Sq_cache,
    float* __restrict__ s_row_decay_cache,
    float* __restrict__ s_col_decay_cache,
    float* __restrict__ m_row_decay_cache,
    float* __restrict__ m_col_decay_cache,
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
    float* s_row_decay = m_vec_shared + N_STATE;
    float* s_col_decay = s_row_decay + N_STATE;
    float* m_row_decay = s_col_decay + N_STATE;
    float* m_col_decay = m_row_decay + N_STATE;
    float* s_retrieved = m_col_decay + N_STATE;
    float* m_retrieved = s_retrieved + N_STATE;
    float* b_s_gate_shared = m_retrieved + N_STATE;
    float* b_m_gate_shared = b_s_gate_shared + N_STATE;

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;
    const int STRIDE = 4 * N_STATE;

    if (tid < N_STATE) {
        b_s_gate_shared[tid] = b_s_gate[tid];
        b_m_gate_shared[tid] = b_m_gate[tid];
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

        if (tid < N_STATE) {
            float row_sum = 0.0f, col_sum = 0.0f;
            for (int j = 0; j < N_STATE; j++) {
                row_sum += M_shared[tid * N_STATE + j] * k_shared[j];
                col_sum += M_shared[j * N_STATE + tid] * k_shared[j];
            }
            s_row_decay[tid] = 1.0f / (1.0f + expf(-(row_sum + b_s_gate_shared[tid])));
            s_col_decay[tid] = 1.0f / (1.0f + expf(-(col_sum + b_s_gate_shared[tid])));
            s_row_decay_cache[t * B * N_STATE + b * N_STATE + tid] = s_row_decay[tid];
            s_col_decay_cache[t * B * N_STATE + b * N_STATE + tid] = s_col_decay[tid];
        }
        __syncthreads();

        if (tid < N_STATE) {
            float sum = 0.0f;
            for (int j = 0; j < N_STATE; j++) {
                sum += S_shared[tid * N_STATE + j] * k_shared[j];
            }
            s_retrieved[tid] = sum;
        }
        __syncthreads();

        for (int i = tid; i < n2; i += blockDim.x) {
            int row = i / N_STATE;
            int col = i % N_STATE;
            float s_delta_row = v_shared[row] - s_retrieved[row];
            S_shared[i] = s_row_decay[row] * S_shared[i] * s_col_decay[col] + s_delta_row * k_shared[col];
        }
        __syncthreads();

        if (tid < N_STATE) {
            float row_sum = 0.0f, col_sum = 0.0f;
            for (int j = 0; j < N_STATE; j++) {
                row_sum += S_shared[tid * N_STATE + j] * m_vec_shared[j];
                col_sum += S_shared[j * N_STATE + tid] * m_vec_shared[j];
            }
            m_row_decay[tid] = 1.0f / (1.0f + expf(-(row_sum + b_m_gate_shared[tid])));
            m_col_decay[tid] = 1.0f / (1.0f + expf(-(col_sum + b_m_gate_shared[tid])));
            m_row_decay_cache[t * B * N_STATE + b * N_STATE + tid] = m_row_decay[tid];
            m_col_decay_cache[t * B * N_STATE + b * N_STATE + tid] = m_col_decay[tid];
        }
        __syncthreads();

        if (tid < N_STATE) {
            float sum = 0.0f;
            for (int j = 0; j < N_STATE; j++) {
                sum += M_shared[tid * N_STATE + j] * m_vec_shared[j];
            }
            m_retrieved[tid] = sum;
        }
        __syncthreads();

        if (tid < N_STATE) {
            float s_delta_row = v_shared[tid] - s_retrieved[tid];
            float m_delta_row = s_delta_row - m_retrieved[tid];
            for (int col = 0; col < N_STATE; col++) {
                int idx = tid * N_STATE + col;
                M_shared[idx] = m_row_decay[tid] * M_shared[idx] * m_col_decay[col] + m_delta_row * m_vec_shared[col];
            }
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
__global__ void E79CoupledBackwardKernel_FP32(
    int T,
    int B,
    const float* __restrict__ kvqm_all,
    const float* __restrict__ b_s_gate,
    const float* __restrict__ b_m_gate,
    const float* __restrict__ s_row_decay_cache,
    const float* __restrict__ s_col_decay_cache,
    const float* __restrict__ m_row_decay_cache,
    const float* __restrict__ m_col_decay_cache,
    const float* __restrict__ S_checkpoints,
    const float* __restrict__ M_checkpoints,
    const float* __restrict__ Sq_cache,
    const float* __restrict__ d_output,
    float* __restrict__ d_kvqm_all,
    float* __restrict__ d_b_s_gate_accum,
    float* __restrict__ d_b_m_gate_accum,
    int checkpoint_interval
) {
    // FP32 backward - same logic as BF16 but without conversions
    // For brevity, implementing minimal version that delegates to BF16 pattern
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
    float* s_row_decay = m_norm + N_STATE;
    float* s_col_decay = s_row_decay + N_STATE;
    float* m_row_decay = s_col_decay + N_STATE;
    float* m_col_decay = m_row_decay + N_STATE;
    float* s_retrieved = m_col_decay + N_STATE;
    float* m_retrieved = s_retrieved + N_STATE;
    float* s_delta = m_retrieved + N_STATE;
    float* m_delta = s_delta + N_STATE;
    float* b_s_gate_shared = m_delta + N_STATE;
    float* b_m_gate_shared = b_s_gate_shared + N_STATE;
    float* d_k_raw = b_m_gate_shared + N_STATE;
    float* d_v_raw = d_k_raw + N_STATE;
    float* d_q_raw = d_v_raw + N_STATE;
    float* d_m_raw = d_q_raw + N_STATE;
    float* d_Sq_shared = d_m_raw + N_STATE;
    float* d_s_delta = d_Sq_shared + N_STATE;
    float* d_m_delta = d_s_delta + N_STATE;
    float* d_k_norm = d_m_delta + N_STATE;
    float* d_m_norm = d_k_norm + N_STATE;
    float* d_s_row_decay = d_m_norm + N_STATE;
    float* d_s_col_decay = d_s_row_decay + N_STATE;
    float* d_m_row_decay = d_s_col_decay + N_STATE;
    float* d_m_col_decay = d_m_row_decay + N_STATE;
    float* d_b_s_gate_local = d_m_col_decay + N_STATE;
    float* d_b_m_gate_local = d_b_s_gate_local + N_STATE;

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;
    const int STRIDE = 4 * N_STATE;

    if (tid < N_STATE) {
        b_s_gate_shared[tid] = b_s_gate[tid];
        b_m_gate_shared[tid] = b_m_gate[tid];
        d_b_s_gate_local[tid] = 0.0f;
        d_b_m_gate_local[tid] = 0.0f;
    }
    __syncthreads();

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

            __shared__ float k_norm_val_t, m_norm_val_t;
            for (int tt = t_start; tt <= t; tt++) {
                int col_idx = (tt * B + b) * STRIDE;
                if (tid < N_STATE) {
                    k_raw[tid] = kvqm_all[col_idx + tid];
                    v_raw[tid] = kvqm_all[col_idx + N_STATE + tid];
                    q_raw[tid] = kvqm_all[col_idx + 2 * N_STATE + tid];
                    m_vec_raw[tid] = kvqm_all[col_idx + 3 * N_STATE + tid];
                    s_row_decay[tid] = s_row_decay_cache[tt * B * N_STATE + b * N_STATE + tid];
                    s_col_decay[tid] = s_col_decay_cache[tt * B * N_STATE + b * N_STATE + tid];
                    m_row_decay[tid] = m_row_decay_cache[tt * B * N_STATE + b * N_STATE + tid];
                    m_col_decay[tid] = m_col_decay_cache[tt * B * N_STATE + b * N_STATE + tid];
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

                if (tid < N_STATE) {
                    float sum = 0.0f;
                    for (int j = 0; j < N_STATE; j++) {
                        sum += S[tid * N_STATE + j] * k_norm[j];
                    }
                    s_retrieved[tid] = sum;
                    s_delta[tid] = v_raw[tid] - s_retrieved[tid];
                }
                __syncthreads();

                if (tt < t) {
                    for (int i = tid; i < n2; i += blockDim.x) {
                        int row = i / N_STATE;
                        int col = i % N_STATE;
                        S[i] = s_row_decay[row] * S[i] * s_col_decay[col] + s_delta[row] * k_norm[col];
                    }
                    __syncthreads();

                    if (tid < N_STATE) {
                        float sum = 0.0f;
                        for (int j = 0; j < N_STATE; j++) {
                            sum += M[tid * N_STATE + j] * m_norm[j];
                        }
                        m_retrieved[tid] = sum;
                        m_delta[tid] = s_delta[tid] - m_retrieved[tid];
                    }
                    __syncthreads();

                    for (int i = tid; i < n2; i += blockDim.x) {
                        int row = i / N_STATE;
                        int col = i % N_STATE;
                        M[i] = m_row_decay[row] * M[i] * m_col_decay[col] + m_delta[row] * m_norm[col];
                    }
                    __syncthreads();
                } else {
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
                    float S_t_ij = s_row_decay[i] * S[i * N_STATE + tid] * s_col_decay[tid] + s_delta[i] * k_norm[tid];
                    sum += S_t_ij * d_Sq_shared[i];
                }
                d_q_raw[tid] = sum;
            }
            __syncthreads();

            if (tid < N_STATE) {
                float d_m_delta_local = 0.0f, d_m_norm_local = 0.0f;
                float d_m_row_local = 0.0f, d_m_col_local = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    float dM_ij = dM[tid * N_STATE + j];
                    d_m_delta_local += dM_ij * m_norm[j];
                    d_m_row_local += dM_ij * M[tid * N_STATE + j] * m_col_decay[j];
                    float dM_ji = dM[j * N_STATE + tid];
                    d_m_norm_local += dM_ji * m_delta[j];
                    d_m_col_local += dM_ji * M[j * N_STATE + tid] * m_row_decay[j];
                }
                d_m_delta[tid] = d_m_delta_local;
                d_m_norm[tid] = d_m_norm_local;
                d_m_row_decay[tid] = d_m_row_local;
                d_m_col_decay[tid] = d_m_col_local;
            }
            __syncthreads();

            if (tid < N_STATE) {
                float d_s_delta_local = 0.0f, d_k_norm_local = 0.0f;
                float d_s_row_local = 0.0f, d_s_col_local = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    float dS_ij = dS[tid * N_STATE + j];
                    d_s_delta_local += dS_ij * k_norm[j];
                    d_s_row_local += dS_ij * S[tid * N_STATE + j] * s_col_decay[j];
                    float dS_ji = dS[j * N_STATE + tid];
                    d_k_norm_local += dS_ji * s_delta[j];
                    d_s_col_local += dS_ji * S[j * N_STATE + tid] * s_row_decay[j];
                }
                d_s_delta_local += d_m_delta[tid];
                d_s_delta[tid] = d_s_delta_local;
                d_k_norm[tid] = d_k_norm_local;
                d_s_row_decay[tid] = d_s_row_local;
                d_s_col_decay[tid] = d_s_col_local;
            }
            __syncthreads();

            if (tid < N_STATE) {
                float d_row = d_s_row_decay[tid];
                float d_col = d_s_col_decay[tid];
                float dec_row = s_row_decay[tid];
                float dec_col = s_col_decay[tid];
                d_b_s_gate_local[tid] += d_row * dec_row * (1.0f - dec_row) + d_col * dec_col * (1.0f - dec_col);

                float d_m_row = d_m_row_decay[tid];
                float d_m_col = d_m_col_decay[tid];
                float m_dec_row = m_row_decay[tid];
                float m_dec_col = m_col_decay[tid];
                d_b_m_gate_local[tid] += d_m_row * m_dec_row * (1.0f - m_dec_row) + d_m_col * m_dec_col * (1.0f - m_dec_col);
            }
            __syncthreads();

            if (tid < N_STATE) {
                d_v_raw[tid] = d_s_delta[tid];
            }

            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    sum += S[i * N_STATE + tid] * (-d_s_delta[i]);
                }
                d_k_norm[tid] += sum;
            }
            __syncthreads();

            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    sum += M[i * N_STATE + tid] * d_m_delta[i];
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

            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                dS[i] = dS[i] * s_row_decay[row] * s_col_decay[col] + (-d_s_delta[row]) * k_norm[col];
            }
            __syncthreads();

            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                dM[i] = dM[i] * m_row_decay[row] * m_col_decay[col] + (-d_m_delta[row]) * m_norm[col];
            }
            __syncthreads();
        }
    }

    if (tid < N_STATE) {
        atomicAdd(&d_b_s_gate_accum[tid], d_b_s_gate_local[tid]);
        atomicAdd(&d_b_m_gate_accum[tid], d_b_m_gate_local[tid]);
    }
}

// ============================================================================
// Template instantiations
// ============================================================================

#define INSTANTIATE_E79_KERNELS_BF16(N) \
    template __global__ void E79CoupledForwardKernel_BF16<N>( \
        int, int, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, \
        __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, \
        __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int); \
    template __global__ void E79CoupledBackwardKernel_BF16<N>( \
        int, int, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, \
        const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, \
        const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, \
        __nv_bfloat16*, float*, float*, int);

#define INSTANTIATE_E79_KERNELS_FP32(N) \
    template __global__ void E79CoupledForwardKernel_FP32<N>( \
        int, int, const float*, const float*, const float*, \
        float*, float*, float*, float*, float*, \
        float*, float*, float*, float*, float*, int); \
    template __global__ void E79CoupledBackwardKernel_FP32<N>( \
        int, int, const float*, const float*, const float*, \
        const float*, const float*, const float*, const float*, \
        const float*, const float*, const float*, const float*, \
        float*, float*, float*, int);

#define INSTANTIATE_E79_GLOBALMEM_BF16(N) \
    template __global__ void E79CoupledForwardGlobalMemKernel_BF16<N>( \
        int, int, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, \
        __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, \
        __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, \
        float*, int); \
    template __global__ void E79CoupledBackwardGlobalMemKernel_BF16<N>( \
        int, int, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, \
        const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, \
        const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, \
        __nv_bfloat16*, float*, float*, float*, int);

// Standard kernels (shared memory)
INSTANTIATE_E79_KERNELS_BF16(8)
INSTANTIATE_E79_KERNELS_BF16(16)
INSTANTIATE_E79_KERNELS_BF16(32)
INSTANTIATE_E79_KERNELS_BF16(48)
INSTANTIATE_E79_KERNELS_BF16(64)
INSTANTIATE_E79_KERNELS_BF16(96)
INSTANTIATE_E79_KERNELS_BF16(128)

INSTANTIATE_E79_KERNELS_FP32(8)
INSTANTIATE_E79_KERNELS_FP32(16)
INSTANTIATE_E79_KERNELS_FP32(32)
INSTANTIATE_E79_KERNELS_FP32(48)
INSTANTIATE_E79_KERNELS_FP32(64)
INSTANTIATE_E79_KERNELS_FP32(96)
INSTANTIATE_E79_KERNELS_FP32(128)

// Global memory fallback kernels (for large n_state)
INSTANTIATE_E79_GLOBALMEM_BF16(96)
INSTANTIATE_E79_GLOBALMEM_BF16(128)

// ============================================================================
// Dispatcher functions
// ============================================================================

void dispatch_e79_coupled_forward(
    int T, int B, int n_state,
    const __nv_bfloat16* kvqm_all,
    const __nv_bfloat16* b_s_gate,
    const __nv_bfloat16* b_m_gate,
    __nv_bfloat16* S, __nv_bfloat16* M, __nv_bfloat16* output,
    __nv_bfloat16* S_checkpoints, __nv_bfloat16* M_checkpoints,
    __nv_bfloat16* Sq_cache,
    __nv_bfloat16* s_row_decay_cache, __nv_bfloat16* s_col_decay_cache,
    __nv_bfloat16* m_row_decay_cache, __nv_bfloat16* m_col_decay_cache,
    float* state_workspace,  // For global memory fallback (n_state >= 128)
    int checkpoint_interval, cudaStream_t stream
) {
    // Shared memory: 2*n^2 (S, M) + 14*n (vectors)
    int shared_size = (2 * n_state * n_state + 14 * n_state) * sizeof(float);
    // Global memory kernel only needs vectors in shared memory: 14*n
    int shared_size_globalmem = 14 * n_state * sizeof(float);

    #define DISPATCH_E79_FWD(N) \
        E79CoupledForwardKernel_BF16<N><<<B, 256, shared_size, stream>>>( \
            T, B, kvqm_all, b_s_gate, b_m_gate, \
            S, M, output, S_checkpoints, M_checkpoints, Sq_cache, \
            s_row_decay_cache, s_col_decay_cache, m_row_decay_cache, m_col_decay_cache, \
            checkpoint_interval);

    #define DISPATCH_E79_FWD_EXT(N) \
        { \
            cudaError_t attr_err = cudaFuncSetAttribute( \
                E79CoupledForwardKernel_BF16<N>, \
                cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size); \
            if (attr_err != cudaSuccess) { \
                fprintf(stderr, "E79 Forward: n_state=%d cudaFuncSetAttribute failed: %s\n", \
                        N, cudaGetErrorString(attr_err)); \
            } else { \
                DISPATCH_E79_FWD(N); \
            } \
        }

    #define DISPATCH_E79_FWD_GLOBALMEM(N) \
        E79CoupledForwardGlobalMemKernel_BF16<N><<<B, 256, shared_size_globalmem, stream>>>( \
            T, B, kvqm_all, b_s_gate, b_m_gate, \
            S, M, output, S_checkpoints, M_checkpoints, Sq_cache, \
            s_row_decay_cache, s_col_decay_cache, m_row_decay_cache, m_col_decay_cache, \
            state_workspace, checkpoint_interval);

    switch (n_state) {
        case 8: DISPATCH_E79_FWD(8); break;
        case 16: DISPATCH_E79_FWD(16); break;
        case 32: DISPATCH_E79_FWD(32); break;
        case 48: DISPATCH_E79_FWD(48); break;
        case 64: DISPATCH_E79_FWD(64); break;
        case 96: DISPATCH_E79_FWD_EXT(96); break;
        case 128: DISPATCH_E79_FWD_GLOBALMEM(128); break;
        default:
            fprintf(stderr, "E79: Unsupported n_state=%d (use 8, 16, 32, 48, 64, 96, or 128)\n", n_state);
    }
    #undef DISPATCH_E79_FWD
    #undef DISPATCH_E79_FWD_EXT
    #undef DISPATCH_E79_FWD_GLOBALMEM
}

void dispatch_e79_coupled_backward(
    int T, int B, int n_state,
    const __nv_bfloat16* kvqm_all,
    const __nv_bfloat16* b_s_gate, const __nv_bfloat16* b_m_gate,
    const __nv_bfloat16* s_row_decay_cache, const __nv_bfloat16* s_col_decay_cache,
    const __nv_bfloat16* m_row_decay_cache, const __nv_bfloat16* m_col_decay_cache,
    const __nv_bfloat16* S_checkpoints, const __nv_bfloat16* M_checkpoints,
    const __nv_bfloat16* Sq_cache, const __nv_bfloat16* d_output,
    __nv_bfloat16* d_kvqm_all,
    float* d_b_s_gate_accum, float* d_b_m_gate_accum,
    float* state_workspace,  // For global memory fallback (n_state >= 96)
    int checkpoint_interval, cudaStream_t stream
) {
    // Shared memory for backward: 4*n^2 + 34*n
    int shared_size = (4 * n_state * n_state + 34 * n_state) * sizeof(float);
    // Global memory kernel only needs vectors in shared memory: 34*n
    int shared_size_globalmem = 34 * n_state * sizeof(float);

    #define DISPATCH_E79_BWD(N) \
        E79CoupledBackwardKernel_BF16<N><<<B, 256, shared_size, stream>>>( \
            T, B, kvqm_all, b_s_gate, b_m_gate, \
            s_row_decay_cache, s_col_decay_cache, m_row_decay_cache, m_col_decay_cache, \
            S_checkpoints, M_checkpoints, Sq_cache, d_output, \
            d_kvqm_all, d_b_s_gate_accum, d_b_m_gate_accum, checkpoint_interval);

    #define DISPATCH_E79_BWD_EXT(N) \
        { \
            cudaError_t attr_err = cudaFuncSetAttribute( \
                E79CoupledBackwardKernel_BF16<N>, \
                cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size); \
            if (attr_err != cudaSuccess) { \
                fprintf(stderr, "E79 Backward: n_state=%d requires %d KB shared memory but GPU limit exceeded\n", \
                        N, shared_size / 1024); \
            } else { \
                DISPATCH_E79_BWD(N); \
            } \
        }

    #define DISPATCH_E79_BWD_GLOBALMEM(N) \
        E79CoupledBackwardGlobalMemKernel_BF16<N><<<B, 256, shared_size_globalmem, stream>>>( \
            T, B, kvqm_all, b_s_gate, b_m_gate, \
            s_row_decay_cache, s_col_decay_cache, m_row_decay_cache, m_col_decay_cache, \
            S_checkpoints, M_checkpoints, Sq_cache, d_output, \
            d_kvqm_all, d_b_s_gate_accum, d_b_m_gate_accum, state_workspace, checkpoint_interval);

    switch (n_state) {
        case 8: DISPATCH_E79_BWD(8); break;
        case 16: DISPATCH_E79_BWD(16); break;
        case 32: DISPATCH_E79_BWD(32); break;
        case 48: DISPATCH_E79_BWD(48); break;
        case 64: DISPATCH_E79_BWD_EXT(64); break;
        case 96: DISPATCH_E79_BWD_GLOBALMEM(96); break;
        case 128: DISPATCH_E79_BWD_GLOBALMEM(128); break;
        default:
            fprintf(stderr, "E79: Unsupported n_state=%d (use 8, 16, 32, 48, 64, 96, or 128)\n", n_state);
    }
    #undef DISPATCH_E79_BWD
    #undef DISPATCH_E79_BWD_EXT
    #undef DISPATCH_E79_BWD_GLOBALMEM
}

void dispatch_e79_coupled_forward_fp32(
    int T, int B, int n_state,
    const float* kvqm_all,
    const float* b_s_gate, const float* b_m_gate,
    float* S, float* M, float* output,
    float* S_checkpoints, float* M_checkpoints,
    float* Sq_cache,
    float* s_row_decay_cache, float* s_col_decay_cache,
    float* m_row_decay_cache, float* m_col_decay_cache,
    int checkpoint_interval, cudaStream_t stream
) {
    int shared_size = (2 * n_state * n_state + 14 * n_state) * sizeof(float);

    #define DISPATCH_E79_FWD_FP32(N) \
        E79CoupledForwardKernel_FP32<N><<<B, 256, shared_size, stream>>>( \
            T, B, kvqm_all, b_s_gate, b_m_gate, \
            S, M, output, S_checkpoints, M_checkpoints, Sq_cache, \
            s_row_decay_cache, s_col_decay_cache, m_row_decay_cache, m_col_decay_cache, \
            checkpoint_interval);

    #define DISPATCH_E79_FWD_FP32_EXT(N) \
        { \
            cudaError_t attr_err = cudaFuncSetAttribute( \
                E79CoupledForwardKernel_FP32<N>, \
                cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size); \
            if (attr_err != cudaSuccess) { \
                fprintf(stderr, "E79 FP32 Forward: n_state=%d cudaFuncSetAttribute failed: %s\n", \
                        N, cudaGetErrorString(attr_err)); \
            } else { \
                DISPATCH_E79_FWD_FP32(N); \
            } \
        }

    switch (n_state) {
        case 8: DISPATCH_E79_FWD_FP32(8); break;
        case 16: DISPATCH_E79_FWD_FP32(16); break;
        case 32: DISPATCH_E79_FWD_FP32(32); break;
        case 48: DISPATCH_E79_FWD_FP32(48); break;
        case 64: DISPATCH_E79_FWD_FP32(64); break;
        case 96: DISPATCH_E79_FWD_FP32_EXT(96); break;
        case 128: DISPATCH_E79_FWD_FP32_EXT(128); break;
        default:
            fprintf(stderr, "E79: Unsupported n_state=%d (use 8, 16, 32, 48, 64, 96, or 128)\n", n_state);
    }
    #undef DISPATCH_E79_FWD_FP32
    #undef DISPATCH_E79_FWD_FP32_EXT
}

void dispatch_e79_coupled_backward_fp32(
    int T, int B, int n_state,
    const float* kvqm_all,
    const float* b_s_gate, const float* b_m_gate,
    const float* s_row_decay_cache, const float* s_col_decay_cache,
    const float* m_row_decay_cache, const float* m_col_decay_cache,
    const float* S_checkpoints, const float* M_checkpoints,
    const float* Sq_cache, const float* d_output,
    float* d_kvqm_all,
    float* d_b_s_gate_accum, float* d_b_m_gate_accum,
    int checkpoint_interval, cudaStream_t stream
) {
    int shared_size = (4 * n_state * n_state + 34 * n_state) * sizeof(float);

    #define DISPATCH_E79_BWD_FP32(N) \
        E79CoupledBackwardKernel_FP32<N><<<B, 256, shared_size, stream>>>( \
            T, B, kvqm_all, b_s_gate, b_m_gate, \
            s_row_decay_cache, s_col_decay_cache, m_row_decay_cache, m_col_decay_cache, \
            S_checkpoints, M_checkpoints, Sq_cache, d_output, \
            d_kvqm_all, d_b_s_gate_accum, d_b_m_gate_accum, checkpoint_interval);

    #define DISPATCH_E79_BWD_FP32_EXT(N) \
        { \
            cudaError_t attr_err = cudaFuncSetAttribute( \
                E79CoupledBackwardKernel_FP32<N>, \
                cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size); \
            if (attr_err != cudaSuccess) { \
                fprintf(stderr, "E79 FP32 Backward: n_state=%d requires %d KB shared memory but GPU limit exceeded\n", \
                        N, shared_size / 1024); \
            } else { \
                DISPATCH_E79_BWD_FP32(N); \
            } \
        }

    switch (n_state) {
        case 8: DISPATCH_E79_BWD_FP32(8); break;
        case 16: DISPATCH_E79_BWD_FP32(16); break;
        case 32: DISPATCH_E79_BWD_FP32(32); break;
        case 48: DISPATCH_E79_BWD_FP32(48); break;
        case 64: DISPATCH_E79_BWD_FP32_EXT(64); break;
        case 96: DISPATCH_E79_BWD_FP32_EXT(96); break;
        case 128: DISPATCH_E79_BWD_FP32_EXT(128); break;
        default:
            fprintf(stderr, "E79: Unsupported n_state=%d (use 8, 16, 32, 48, 64, 96, or 128)\n", n_state);
    }
    #undef DISPATCH_E79_BWD_FP32
    #undef DISPATCH_E79_BWD_FP32_EXT
}

// ============================================================================
// E79CoupledForward Implementation (wrapper class for Python bindings)
// ============================================================================

template<typename DataT>
E79CoupledForward<DataT>::E79CoupledForward(
    bool training, int batch_size, int n_state, int dim,
    const cublasHandle_t& blas_handle, const cudaStream_t& stream)
    : training_(training), batch_size_(batch_size), n_state_(n_state),
      dim_(dim), blas_handle_(blas_handle), stream_(stream) {}

template<typename DataT>
void E79CoupledForward<DataT>::Run(
    int steps,
    const DataT* W_kvqm,
    const DataT* b_s_gate,
    const DataT* b_m_gate,
    const DataT* x,
    DataT* S, DataT* M,
    DataT* output,
    DataT* kvqm_cache,
    DataT* S_checkpoints, DataT* M_checkpoints,
    DataT* Sq_cache,
    DataT* s_row_decay_cache, DataT* s_col_decay_cache,
    DataT* m_row_decay_cache, DataT* m_col_decay_cache
) {
    int T = steps;
    int B = batch_size_;
    int n = n_state_;
    int d = dim_;
    int checkpoint_interval = E79_CHECKPOINT_INTERVAL;

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

    // Allocate state workspace for global memory kernels (n >= 128 for forward)
    float* state_workspace = nullptr;
    if (n >= 128) {
        // Forward needs [B, 2, n, n] for S and M
        cudaMalloc(&state_workspace, B * 2 * n * n * sizeof(float));
        cudaMemsetAsync(state_workspace, 0, B * 2 * n * n * sizeof(float), stream_);
    }

    if constexpr (std::is_same<DataT, __nv_bfloat16>::value) {
        dispatch_e79_coupled_forward(T, B, n, kvqm_cache, b_s_gate, b_m_gate,
                                     S, M, output, S_checkpoints, M_checkpoints, Sq_cache,
                                     s_row_decay_cache, s_col_decay_cache,
                                     m_row_decay_cache, m_col_decay_cache,
                                     state_workspace, checkpoint_interval, stream_);
    } else {
        dispatch_e79_coupled_forward_fp32(T, B, n,
                                          reinterpret_cast<const float*>(kvqm_cache),
                                          reinterpret_cast<const float*>(b_s_gate),
                                          reinterpret_cast<const float*>(b_m_gate),
                                          reinterpret_cast<float*>(S),
                                          reinterpret_cast<float*>(M),
                                          reinterpret_cast<float*>(output),
                                          reinterpret_cast<float*>(S_checkpoints),
                                          reinterpret_cast<float*>(M_checkpoints),
                                          reinterpret_cast<float*>(Sq_cache),
                                          reinterpret_cast<float*>(s_row_decay_cache),
                                          reinterpret_cast<float*>(s_col_decay_cache),
                                          reinterpret_cast<float*>(m_row_decay_cache),
                                          reinterpret_cast<float*>(m_col_decay_cache),
                                          checkpoint_interval, stream_);
    }

    // Free workspace if allocated
    if (state_workspace != nullptr) {
        cudaFree(state_workspace);
    }
}

template<typename DataT>
E79CoupledBackward<DataT>::E79CoupledBackward(
    int batch_size, int n_state, int dim,
    const cublasHandle_t& blas_handle, const cudaStream_t& stream)
    : batch_size_(batch_size), n_state_(n_state), dim_(dim),
      blas_handle_(blas_handle), stream_(stream) {}

template<typename DataT>
void E79CoupledBackward<DataT>::Run(
    int steps,
    const DataT* W_kvqm,
    const DataT* b_s_gate, const DataT* b_m_gate,
    const DataT* x,
    const DataT* kvqm_cache,
    const DataT* S_checkpoints, const DataT* M_checkpoints,
    const DataT* Sq_cache,
    const DataT* s_row_decay_cache, const DataT* s_col_decay_cache,
    const DataT* m_row_decay_cache, const DataT* m_col_decay_cache,
    const DataT* d_output,
    DataT* d_x,
    DataT* d_W_kvqm,
    DataT* d_b_s_gate, DataT* d_b_m_gate,
    DataT* d_kvqm_cache,
    float* d_b_s_gate_accum, float* d_b_m_gate_accum
) {
    int T = steps;
    int B = batch_size_;
    int n = n_state_;
    int d = dim_;
    int checkpoint_interval = E79_CHECKPOINT_INTERVAL;

    const float alpha = 1.0f, beta_zero = 0.0f, beta_one = 1.0f;

    // Zero accumulators
    cudaMemsetAsync(d_b_s_gate_accum, 0, n * sizeof(float), stream_);
    cudaMemsetAsync(d_b_m_gate_accum, 0, n * sizeof(float), stream_);

    // Allocate state workspace for global memory kernels (n >= 96 for backward)
    float* state_workspace = nullptr;
    if (n >= 96) {
        // Backward needs [B, 4, n, n] for S, M, dS, dM
        cudaMalloc(&state_workspace, B * 4 * n * n * sizeof(float));
        cudaMemsetAsync(state_workspace, 0, B * 4 * n * n * sizeof(float), stream_);
    }

    if constexpr (std::is_same<DataT, __nv_bfloat16>::value) {
        dispatch_e79_coupled_backward(T, B, n, kvqm_cache, b_s_gate, b_m_gate,
                                      s_row_decay_cache, s_col_decay_cache,
                                      m_row_decay_cache, m_col_decay_cache,
                                      S_checkpoints, M_checkpoints, Sq_cache, d_output,
                                      d_kvqm_cache, d_b_s_gate_accum, d_b_m_gate_accum,
                                      state_workspace, checkpoint_interval, stream_);
    } else {
        dispatch_e79_coupled_backward_fp32(T, B, n,
                                           reinterpret_cast<const float*>(kvqm_cache),
                                           reinterpret_cast<const float*>(b_s_gate),
                                           reinterpret_cast<const float*>(b_m_gate),
                                           reinterpret_cast<const float*>(s_row_decay_cache),
                                           reinterpret_cast<const float*>(s_col_decay_cache),
                                           reinterpret_cast<const float*>(m_row_decay_cache),
                                           reinterpret_cast<const float*>(m_col_decay_cache),
                                           reinterpret_cast<const float*>(S_checkpoints),
                                           reinterpret_cast<const float*>(M_checkpoints),
                                           reinterpret_cast<const float*>(Sq_cache),
                                           reinterpret_cast<const float*>(d_output),
                                           reinterpret_cast<float*>(d_kvqm_cache),
                                           d_b_s_gate_accum, d_b_m_gate_accum,
                                           checkpoint_interval, stream_);
    }

    // Free workspace if allocated
    if (state_workspace != nullptr) {
        cudaFree(state_workspace);
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

    // d_W_kvqm = d_kvqm_cache @ x^T
    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
                 4 * n, d, T * B,
                 &alpha,
                 d_kvqm_cache, data_type, 4 * n,
                 x, data_type, d,
                 &beta_zero,
                 d_W_kvqm, data_type, 4 * n,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

// Explicit template instantiations
template class E79CoupledForward<__nv_bfloat16>;
template class E79CoupledForward<float>;
template class E79CoupledBackward<__nv_bfloat16>;
template class E79CoupledBackward<float>;

}  // namespace elman
