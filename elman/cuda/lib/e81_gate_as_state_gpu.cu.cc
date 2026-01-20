/**
 * E81 Gate As State CUDA Kernel
 *
 * Key insight: The gate itself (G) is a hidden state that EVOLVES over time.
 * Unlike E79 where M computes gates dynamically (M @ k), here G IS the gate directly.
 *
 * Two coupled matrix states:
 * - S [n_state x n_state]: Content memory - stores key-value associations
 * - G [n_state x n_state]: Gate state - directly provides gates via sigmoid(G)
 *
 * Architecture:
 *   FUSED projection: kvqm = W_kvqm @ x  (k, v, q, m)
 *
 *   # S update (G-gated) - G directly is the gate
 *   gate_S = sigmoid(G + b_s_gate)  # G provides gate directly (not M @ k!)
 *   s_delta = v - S @ k_norm
 *   S = gate_S * S + outer(s_delta, k_norm)
 *
 *   # G update (S-gated)
 *   gate_G = sigmoid(S + b_g_gate)  # S gates G
 *   g_delta = s_delta - G @ m_norm  # G predicts S's changes
 *   G = gate_G * G + outer(g_delta, m_norm)
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

#define E81_CHECKPOINT_INTERVAL 16

namespace elman {

// Utility kernel for converting float accumulators to bfloat16
__global__ void ConvertFloatToBF16Kernel_E81(
    const float* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __float2bfloat16(src[i]);
    }
}

// ============================================================================
// E81 Forward Kernel - Gate As State
// ============================================================================

template<int N_STATE>
__global__ void E81GateAsStateForwardKernel_BF16(
    int T,
    int B,
    const __nv_bfloat16* __restrict__ kvqm_all,   // [4*N_STATE, T*B] column-major
    const __nv_bfloat16* __restrict__ b_s_gate,   // [N_STATE] S gate bias
    const __nv_bfloat16* __restrict__ b_g_gate,   // [N_STATE] G gate bias
    __nv_bfloat16* __restrict__ S,                // [B, N_STATE, N_STATE]
    __nv_bfloat16* __restrict__ G,                // [B, N_STATE, N_STATE]
    __nv_bfloat16* __restrict__ output,           // [T, B, N_STATE]
    __nv_bfloat16* __restrict__ S_checkpoints,    // [num_checkpoints, B, N_STATE, N_STATE]
    __nv_bfloat16* __restrict__ G_checkpoints,    // [num_checkpoints, B, N_STATE, N_STATE]
    __nv_bfloat16* __restrict__ Sq_cache,         // [T, B, N_STATE]
    __nv_bfloat16* __restrict__ gate_S_cache,     // [T, B, N_STATE, N_STATE] sigmoid(G) cache
    __nv_bfloat16* __restrict__ gate_G_cache,     // [T, B, N_STATE, N_STATE] sigmoid(S) cache
    int checkpoint_interval
) {
    int b = blockIdx.x;
    if (b >= B) return;

    // Shared memory layout - need space for two matrices plus vectors
    extern __shared__ float shared_mem[];
    float* S_shared = shared_mem;                         // [N_STATE * N_STATE]
    float* G_shared = S_shared + N_STATE * N_STATE;       // [N_STATE * N_STATE]
    float* k_shared = G_shared + N_STATE * N_STATE;       // [N_STATE]
    float* v_shared = k_shared + N_STATE;                 // [N_STATE]
    float* q_shared = v_shared + N_STATE;                 // [N_STATE]
    float* m_vec_shared = q_shared + N_STATE;             // [N_STATE]
    float* s_retrieved = m_vec_shared + N_STATE;          // [N_STATE]
    float* g_retrieved = s_retrieved + N_STATE;           // [N_STATE]
    float* s_delta = g_retrieved + N_STATE;               // [N_STATE]
    float* b_s_gate_shared = s_delta + N_STATE;           // [N_STATE]
    float* b_g_gate_shared = b_s_gate_shared + N_STATE;   // [N_STATE]

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;
    const int STRIDE = 4 * N_STATE;

    // Load gate biases
    if (tid < N_STATE) {
        b_s_gate_shared[tid] = __bfloat162float(b_s_gate[tid]);
        b_g_gate_shared[tid] = __bfloat162float(b_g_gate[tid]);
    }
    __syncthreads();

    // Load initial states
    for (int i = tid; i < n2; i += blockDim.x) {
        S_shared[i] = __bfloat162float(S[b * n2 + i]);
        G_shared[i] = __bfloat162float(G[b * n2 + i]);
    }
    __syncthreads();

    // Save initial checkpoints (index 0)
    for (int i = tid; i < n2; i += blockDim.x) {
        S_checkpoints[b * n2 + i] = __float2bfloat16(S_shared[i]);
        G_checkpoints[b * n2 + i] = __float2bfloat16(G_shared[i]);
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

        // --- S update (G-gated) ---
        // gate_S = sigmoid(G + b_s_gate) - G directly provides gate!
        // s_retrieved = S @ k_norm
        if (tid < N_STATE) {
            float sum = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < N_STATE; j++) {
                sum += S_shared[tid * N_STATE + j] * k_shared[j];
            }
            s_retrieved[tid] = sum;
            s_delta[tid] = v_shared[tid] - sum;
        }
        __syncthreads();

        // Update S: gate_S * S + outer(s_delta, k_norm)
        // Gate is sigmoid(G[i,j] + b_s_gate[i]) - row-wise bias
        for (int i = tid; i < n2; i += blockDim.x) {
            int row = i / N_STATE;
            int col = i % N_STATE;
            float gate_val = 1.0f / (1.0f + expf(-(G_shared[i] + b_s_gate_shared[row])));
            float update = gate_val * S_shared[i] + s_delta[row] * k_shared[col];
            S_shared[i] = update;

            // Cache gate_S for backward
            gate_S_cache[t * B * n2 + b * n2 + i] = __float2bfloat16(gate_val);
        }
        __syncthreads();

        // --- G update (S-gated) ---
        // gate_G = sigmoid(S + b_g_gate) - S gates G
        // g_retrieved = G @ m_norm
        if (tid < N_STATE) {
            float sum = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < N_STATE; j++) {
                sum += G_shared[tid * N_STATE + j] * m_vec_shared[j];
            }
            g_retrieved[tid] = sum;
        }
        __syncthreads();

        // Update G: gate_G * G + outer(g_delta, m_norm)
        // g_delta = s_delta - g_retrieved (G predicts S's changes)
        for (int i = tid; i < n2; i += blockDim.x) {
            int row = i / N_STATE;
            int col = i % N_STATE;
            // Use S AFTER update for gating G
            float gate_val = 1.0f / (1.0f + expf(-(S_shared[i] + b_g_gate_shared[row])));
            float g_delta_row = s_delta[row] - g_retrieved[row];
            float update = gate_val * G_shared[i] + g_delta_row * m_vec_shared[col];
            G_shared[i] = update;

            // Cache gate_G for backward
            gate_G_cache[t * B * n2 + b * n2 + i] = __float2bfloat16(gate_val);
        }
        __syncthreads();

        // Save checkpoints if at boundary
        if ((t + 1) % checkpoint_interval == 0) {
            int cp_idx = (t + 1) / checkpoint_interval;
            for (int i = tid; i < n2; i += blockDim.x) {
                S_checkpoints[cp_idx * B * n2 + b * n2 + i] = __float2bfloat16(S_shared[i]);
                G_checkpoints[cp_idx * B * n2 + b * n2 + i] = __float2bfloat16(G_shared[i]);
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
        G[b * n2 + i] = __float2bfloat16(G_shared[i]);
    }
}

// ============================================================================
// E81 Backward Kernel - Gate As State
// ============================================================================

template<int N_STATE>
__global__ void E81GateAsStateBackwardKernel_BF16(
    int T,
    int B,
    const __nv_bfloat16* __restrict__ kvqm_all,
    const __nv_bfloat16* __restrict__ b_s_gate,
    const __nv_bfloat16* __restrict__ b_g_gate,
    const __nv_bfloat16* __restrict__ gate_S_cache,   // [T, B, N_STATE, N_STATE]
    const __nv_bfloat16* __restrict__ gate_G_cache,   // [T, B, N_STATE, N_STATE]
    const __nv_bfloat16* __restrict__ S_checkpoints,
    const __nv_bfloat16* __restrict__ G_checkpoints,
    const __nv_bfloat16* __restrict__ Sq_cache,
    const __nv_bfloat16* __restrict__ d_output,
    __nv_bfloat16* __restrict__ d_kvqm_all,
    float* __restrict__ d_b_s_gate_accum,
    float* __restrict__ d_b_g_gate_accum,
    int checkpoint_interval
) {
    int b = blockIdx.x;
    if (b >= B) return;

    extern __shared__ float shared_mem[];
    // Layout for backward - need space for states and gradients
    float* S = shared_mem;                                // [N_STATE * N_STATE]
    float* G = S + N_STATE * N_STATE;                     // [N_STATE * N_STATE]
    float* dS = G + N_STATE * N_STATE;                    // [N_STATE * N_STATE]
    float* dG = dS + N_STATE * N_STATE;                   // [N_STATE * N_STATE]
    float* k_raw = dG + N_STATE * N_STATE;                // [N_STATE]
    float* v_raw = k_raw + N_STATE;                       // [N_STATE]
    float* q_raw = v_raw + N_STATE;                       // [N_STATE]
    float* m_vec_raw = q_raw + N_STATE;                   // [N_STATE]
    float* k_norm = m_vec_raw + N_STATE;                  // [N_STATE]
    float* m_norm = k_norm + N_STATE;                     // [N_STATE]
    float* s_retrieved = m_norm + N_STATE;                // [N_STATE]
    float* g_retrieved = s_retrieved + N_STATE;           // [N_STATE]
    float* s_delta = g_retrieved + N_STATE;               // [N_STATE]
    float* g_delta = s_delta + N_STATE;                   // [N_STATE]
    float* b_s_gate_shared = g_delta + N_STATE;           // [N_STATE]
    float* b_g_gate_shared = b_s_gate_shared + N_STATE;   // [N_STATE]
    float* d_k_raw = b_g_gate_shared + N_STATE;           // [N_STATE]
    float* d_v_raw = d_k_raw + N_STATE;                   // [N_STATE]
    float* d_q_raw = d_v_raw + N_STATE;                   // [N_STATE]
    float* d_m_raw = d_q_raw + N_STATE;                   // [N_STATE]
    float* d_Sq_shared = d_m_raw + N_STATE;               // [N_STATE]
    float* d_s_delta = d_Sq_shared + N_STATE;             // [N_STATE]
    float* d_g_delta = d_s_delta + N_STATE;               // [N_STATE]
    float* d_k_norm = d_g_delta + N_STATE;                // [N_STATE]
    float* d_m_norm = d_k_norm + N_STATE;                 // [N_STATE]
    float* d_b_s_gate_local = d_m_norm + N_STATE;         // [N_STATE]
    float* d_b_g_gate_local = d_b_s_gate_local + N_STATE; // [N_STATE]

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;
    const int STRIDE = 4 * N_STATE;

    // Load gate biases
    if (tid < N_STATE) {
        b_s_gate_shared[tid] = __bfloat162float(b_s_gate[tid]);
        b_g_gate_shared[tid] = __bfloat162float(b_g_gate[tid]);
        d_b_s_gate_local[tid] = 0.0f;
        d_b_g_gate_local[tid] = 0.0f;
    }
    __syncthreads();

    // Initialize gradient accumulators
    for (int i = tid; i < n2; i += blockDim.x) {
        dS[i] = 0.0f;
        dG[i] = 0.0f;
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
                G[i] = __bfloat162float(G_checkpoints[seg * B * n2 + b * n2 + i]);
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
                        float gate_val = 1.0f / (1.0f + expf(-(G[i] + b_s_gate_shared[row])));
                        S[i] = gate_val * S[i] + s_delta[row] * k_norm[col];
                    }
                    __syncthreads();

                    // Compute g_retrieved and g_delta
                    if (tid < N_STATE) {
                        float sum = 0.0f;
                        for (int j = 0; j < N_STATE; j++) {
                            sum += G[tid * N_STATE + j] * m_norm[j];
                        }
                        g_retrieved[tid] = sum;
                        g_delta[tid] = s_delta[tid] - g_retrieved[tid];
                    }
                    __syncthreads();

                    // Update G
                    for (int i = tid; i < n2; i += blockDim.x) {
                        int row = i / N_STATE;
                        int col = i % N_STATE;
                        float gate_val = 1.0f / (1.0f + expf(-(S[i] + b_g_gate_shared[row])));
                        G[i] = gate_val * G[i] + g_delta[row] * m_norm[col];
                    }
                    __syncthreads();
                } else {
                    // At target step - compute g_retrieved and g_delta for backward
                    if (tid < N_STATE) {
                        float sum = 0.0f;
                        for (int j = 0; j < N_STATE; j++) {
                            sum += G[tid * N_STATE + j] * m_norm[j];
                        }
                        g_retrieved[tid] = sum;
                        g_delta[tid] = s_delta[tid] - g_retrieved[tid];
                    }
                    __syncthreads();
                }
            }

            // Now S, G hold states at t-1; s_delta, g_delta are for step t

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
                for (int j = 0; j < N_STATE; j++) {
                    // S_t after update
                    float gate_val = __bfloat162float(gate_S_cache[t * B * n2 + b * n2 + j * N_STATE + tid]);
                    float S_t_jt = gate_val * S[j * N_STATE + tid] + s_delta[j] * k_norm[tid];
                    sum += S_t_jt * d_Sq_shared[j];
                }
                d_q_raw[tid] = sum;
            }
            __syncthreads();

            // Load cached gates
            __shared__ float gate_S_local[N_STATE * N_STATE];
            __shared__ float gate_G_local[N_STATE * N_STATE];
            for (int i = tid; i < n2; i += blockDim.x) {
                gate_S_local[i] = __bfloat162float(gate_S_cache[t * B * n2 + b * n2 + i]);
                gate_G_local[i] = __bfloat162float(gate_G_cache[t * B * n2 + b * n2 + i]);
            }
            __syncthreads();

            // Backward through G update: G_new = gate_G * G + outer(g_delta, m_norm)
            // dG_old = dG_new * gate_G
            // d_gate_G = dG_new * G_old
            // d_g_delta = dG_new @ m_norm
            // d_m_norm = dG_new.T @ g_delta
            if (tid < N_STATE) {
                float d_gdt = 0.0f;
                float d_mn = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    float dG_ij = dG[tid * N_STATE + j];
                    d_gdt += dG_ij * m_norm[j];
                    d_mn += dG[j * N_STATE + tid] * g_delta[j];
                }
                d_g_delta[tid] = d_gdt;
                d_m_norm[tid] = d_mn;
            }
            __syncthreads();

            // d_g_delta flows to d_s_delta (from g_delta = s_delta - g_retrieved)
            // d_s_delta += d_g_delta
            // d_g_retrieved = -d_g_delta

            // Backward through S update: S_new = gate_S * S + outer(s_delta, k_norm)
            if (tid < N_STATE) {
                float d_sdt = 0.0f;
                float d_kn = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    float dS_ij = dS[tid * N_STATE + j];
                    d_sdt += dS_ij * k_norm[j];
                    d_kn += dS[j * N_STATE + tid] * s_delta[j];
                }
                // Add contribution from g_delta = s_delta - g_retrieved
                d_s_delta[tid] = d_sdt + d_g_delta[tid];
                d_k_norm[tid] = d_kn;
            }
            __syncthreads();

            // Backward through gate_G sigmoid
            // gate_G = sigmoid(S + b_g_gate)
            // d_S += d_gate_G * gate_G * (1 - gate_G)
            // d_b_g_gate += sum_col(d_gate_G * gate_G * (1 - gate_G))
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                float dG_i = dG[i];
                float gate_G_i = gate_G_local[i];
                float d_gate_G_i = dG_i * G[i];  // G_old is what's gated
                float d_pre_sigmoid = d_gate_G_i * gate_G_i * (1.0f - gate_G_i);
                dS[i] += d_pre_sigmoid;  // d_S from gate_G computation
                // Note: bias gradient accumulated per row
                if (i % N_STATE == 0) {  // First column of each row
                    float row_sum = 0.0f;
                    for (int c = 0; c < N_STATE; c++) {
                        int idx = row * N_STATE + c;
                        float dG_rc = dG[idx];
                        float gate_G_rc = gate_G_local[idx];
                        float d_gate_G_rc = dG_rc * G[idx];
                        row_sum += d_gate_G_rc * gate_G_rc * (1.0f - gate_G_rc);
                    }
                    atomicAdd(&d_b_g_gate_local[row], row_sum);
                }
            }
            __syncthreads();

            // Backward through gate_S sigmoid
            // gate_S = sigmoid(G + b_s_gate)
            // dG += d_gate_S * gate_S * (1 - gate_S)
            // d_b_s_gate += sum_col(d_gate_S * gate_S * (1 - gate_S))
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                float dS_i = dS[i];
                float gate_S_i = gate_S_local[i];
                float d_gate_S_i = dS_i * S[i];  // S_old is what's gated
                float d_pre_sigmoid = d_gate_S_i * gate_S_i * (1.0f - gate_S_i);
                dG[i] += d_pre_sigmoid;  // dG from gate_S computation
                if (i % N_STATE == 0) {
                    float row_sum = 0.0f;
                    for (int c = 0; c < N_STATE; c++) {
                        int idx = row * N_STATE + c;
                        float dS_rc = dS[idx];
                        float gate_S_rc = gate_S_local[idx];
                        float d_gate_S_rc = dS_rc * S[idx];
                        row_sum += d_gate_S_rc * gate_S_rc * (1.0f - gate_S_rc);
                    }
                    atomicAdd(&d_b_s_gate_local[row], row_sum);
                }
            }
            __syncthreads();

            // d_v = d_s_delta
            if (tid < N_STATE) {
                d_v_raw[tid] = d_s_delta[tid];
            }

            // d_k_norm contribution from retrieved: s_retrieved = S @ k_norm
            // d_k_norm += S^T @ (-d_s_delta)
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    sum += S[j * N_STATE + tid] * (-d_s_delta[j]);
                }
                d_k_norm[tid] += sum;
            }
            __syncthreads();

            // d_m_norm contribution from g_retrieved: g_retrieved = G @ m_norm
            // d_m_norm += G^T @ d_g_delta (from g_delta = s_delta - g_retrieved)
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    sum += G[j * N_STATE + tid] * d_g_delta[j];
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
                float gate_S_i = gate_S_local[i];
                dS[i] = dS[i] * gate_S_i + (-d_s_delta[row]) * k_norm[col];
            }
            __syncthreads();

            // Update dG for next iteration
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                float gate_G_i = gate_G_local[i];
                dG[i] = dG[i] * gate_G_i + (-d_g_delta[row]) * m_norm[col];
            }
            __syncthreads();
        }
    }

    // Accumulate bias gradients
    if (tid < N_STATE) {
        atomicAdd(&d_b_s_gate_accum[tid], d_b_s_gate_local[tid]);
        atomicAdd(&d_b_g_gate_accum[tid], d_b_g_gate_local[tid]);
    }
}

// ============================================================================
// FP32 versions
// ============================================================================

template<int N_STATE>
__global__ void E81GateAsStateForwardKernel_FP32(
    int T,
    int B,
    const float* __restrict__ kvqm_all,
    const float* __restrict__ b_s_gate,
    const float* __restrict__ b_g_gate,
    float* __restrict__ S,
    float* __restrict__ G,
    float* __restrict__ output,
    float* __restrict__ S_checkpoints,
    float* __restrict__ G_checkpoints,
    float* __restrict__ Sq_cache,
    float* __restrict__ gate_S_cache,
    float* __restrict__ gate_G_cache,
    int checkpoint_interval
) {
    int b = blockIdx.x;
    if (b >= B) return;

    extern __shared__ float shared_mem[];
    float* S_shared = shared_mem;
    float* G_shared = S_shared + N_STATE * N_STATE;
    float* k_shared = G_shared + N_STATE * N_STATE;
    float* v_shared = k_shared + N_STATE;
    float* q_shared = v_shared + N_STATE;
    float* m_vec_shared = q_shared + N_STATE;
    float* s_retrieved = m_vec_shared + N_STATE;
    float* g_retrieved = s_retrieved + N_STATE;
    float* s_delta = g_retrieved + N_STATE;
    float* b_s_gate_shared = s_delta + N_STATE;
    float* b_g_gate_shared = b_s_gate_shared + N_STATE;

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;
    const int STRIDE = 4 * N_STATE;

    if (tid < N_STATE) {
        b_s_gate_shared[tid] = b_s_gate[tid];
        b_g_gate_shared[tid] = b_g_gate[tid];
    }
    __syncthreads();

    for (int i = tid; i < n2; i += blockDim.x) {
        S_shared[i] = S[b * n2 + i];
        G_shared[i] = G[b * n2 + i];
    }
    __syncthreads();

    for (int i = tid; i < n2; i += blockDim.x) {
        S_checkpoints[b * n2 + i] = S_shared[i];
        G_checkpoints[b * n2 + i] = G_shared[i];
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

        // s_retrieved = S @ k_norm
        if (tid < N_STATE) {
            float sum = 0.0f;
            for (int j = 0; j < N_STATE; j++) {
                sum += S_shared[tid * N_STATE + j] * k_shared[j];
            }
            s_retrieved[tid] = sum;
            s_delta[tid] = v_shared[tid] - sum;
        }
        __syncthreads();

        // Update S with G-gating
        for (int i = tid; i < n2; i += blockDim.x) {
            int row = i / N_STATE;
            int col = i % N_STATE;
            float gate_val = 1.0f / (1.0f + expf(-(G_shared[i] + b_s_gate_shared[row])));
            S_shared[i] = gate_val * S_shared[i] + s_delta[row] * k_shared[col];
            gate_S_cache[t * B * n2 + b * n2 + i] = gate_val;
        }
        __syncthreads();

        // g_retrieved = G @ m_norm
        if (tid < N_STATE) {
            float sum = 0.0f;
            for (int j = 0; j < N_STATE; j++) {
                sum += G_shared[tid * N_STATE + j] * m_vec_shared[j];
            }
            g_retrieved[tid] = sum;
        }
        __syncthreads();

        // Update G with S-gating
        for (int i = tid; i < n2; i += blockDim.x) {
            int row = i / N_STATE;
            int col = i % N_STATE;
            float gate_val = 1.0f / (1.0f + expf(-(S_shared[i] + b_g_gate_shared[row])));
            float g_delta_row = s_delta[row] - g_retrieved[row];
            G_shared[i] = gate_val * G_shared[i] + g_delta_row * m_vec_shared[col];
            gate_G_cache[t * B * n2 + b * n2 + i] = gate_val;
        }
        __syncthreads();

        if ((t + 1) % checkpoint_interval == 0) {
            int cp_idx = (t + 1) / checkpoint_interval;
            for (int i = tid; i < n2; i += blockDim.x) {
                S_checkpoints[cp_idx * B * n2 + b * n2 + i] = S_shared[i];
                G_checkpoints[cp_idx * B * n2 + b * n2 + i] = G_shared[i];
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
        G[b * n2 + i] = G_shared[i];
    }
}

template<int N_STATE>
__global__ void E81GateAsStateBackwardKernel_FP32(
    int T,
    int B,
    const float* __restrict__ kvqm_all,
    const float* __restrict__ b_s_gate,
    const float* __restrict__ b_g_gate,
    const float* __restrict__ gate_S_cache,
    const float* __restrict__ gate_G_cache,
    const float* __restrict__ S_checkpoints,
    const float* __restrict__ G_checkpoints,
    const float* __restrict__ Sq_cache,
    const float* __restrict__ d_output,
    float* __restrict__ d_kvqm_all,
    float* __restrict__ d_b_s_gate_accum,
    float* __restrict__ d_b_g_gate_accum,
    int checkpoint_interval
) {
    // FP32 backward - same logic as BF16 but without conversions
    int b = blockIdx.x;
    if (b >= B) return;

    extern __shared__ float shared_mem[];
    float* S = shared_mem;
    float* G = S + N_STATE * N_STATE;
    float* dS = G + N_STATE * N_STATE;
    float* dG = dS + N_STATE * N_STATE;
    float* k_raw = dG + N_STATE * N_STATE;
    float* v_raw = k_raw + N_STATE;
    float* q_raw = v_raw + N_STATE;
    float* m_vec_raw = q_raw + N_STATE;
    float* k_norm = m_vec_raw + N_STATE;
    float* m_norm = k_norm + N_STATE;
    float* s_retrieved = m_norm + N_STATE;
    float* g_retrieved = s_retrieved + N_STATE;
    float* s_delta = g_retrieved + N_STATE;
    float* g_delta = s_delta + N_STATE;
    float* b_s_gate_shared = g_delta + N_STATE;
    float* b_g_gate_shared = b_s_gate_shared + N_STATE;
    float* d_k_raw = b_g_gate_shared + N_STATE;
    float* d_v_raw = d_k_raw + N_STATE;
    float* d_q_raw = d_v_raw + N_STATE;
    float* d_m_raw = d_q_raw + N_STATE;
    float* d_Sq_shared = d_m_raw + N_STATE;
    float* d_s_delta = d_Sq_shared + N_STATE;
    float* d_g_delta = d_s_delta + N_STATE;
    float* d_k_norm = d_g_delta + N_STATE;
    float* d_m_norm = d_k_norm + N_STATE;
    float* d_b_s_gate_local = d_m_norm + N_STATE;
    float* d_b_g_gate_local = d_b_s_gate_local + N_STATE;
    float* gate_S_local = d_b_g_gate_local + N_STATE;
    float* gate_G_local = gate_S_local + N_STATE * N_STATE;

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;
    const int STRIDE = 4 * N_STATE;

    if (tid < N_STATE) {
        b_s_gate_shared[tid] = b_s_gate[tid];
        b_g_gate_shared[tid] = b_g_gate[tid];
        d_b_s_gate_local[tid] = 0.0f;
        d_b_g_gate_local[tid] = 0.0f;
    }
    __syncthreads();

    for (int i = tid; i < n2; i += blockDim.x) {
        dS[i] = 0.0f;
        dG[i] = 0.0f;
    }
    __syncthreads();

    int num_segments = (T + checkpoint_interval - 1) / checkpoint_interval;

    for (int seg = num_segments - 1; seg >= 0; seg--) {
        int t_start = seg * checkpoint_interval;
        int t_end = min(t_start + checkpoint_interval, T);

        for (int t = t_end - 1; t >= t_start; t--) {
            for (int i = tid; i < n2; i += blockDim.x) {
                S[i] = S_checkpoints[seg * B * n2 + b * n2 + i];
                G[i] = G_checkpoints[seg * B * n2 + b * n2 + i];
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
                    s_delta[tid] = v_raw[tid] - sum;
                }
                __syncthreads();

                if (tt < t) {
                    for (int i = tid; i < n2; i += blockDim.x) {
                        int row = i / N_STATE;
                        int col = i % N_STATE;
                        float gate_val = 1.0f / (1.0f + expf(-(G[i] + b_s_gate_shared[row])));
                        S[i] = gate_val * S[i] + s_delta[row] * k_norm[col];
                    }
                    __syncthreads();

                    if (tid < N_STATE) {
                        float sum = 0.0f;
                        for (int j = 0; j < N_STATE; j++) {
                            sum += G[tid * N_STATE + j] * m_norm[j];
                        }
                        g_retrieved[tid] = sum;
                        g_delta[tid] = s_delta[tid] - g_retrieved[tid];
                    }
                    __syncthreads();

                    for (int i = tid; i < n2; i += blockDim.x) {
                        int row = i / N_STATE;
                        int col = i % N_STATE;
                        float gate_val = 1.0f / (1.0f + expf(-(S[i] + b_g_gate_shared[row])));
                        G[i] = gate_val * G[i] + g_delta[row] * m_norm[col];
                    }
                    __syncthreads();
                } else {
                    if (tid < N_STATE) {
                        float sum = 0.0f;
                        for (int j = 0; j < N_STATE; j++) {
                            sum += G[tid * N_STATE + j] * m_norm[j];
                        }
                        g_retrieved[tid] = sum;
                        g_delta[tid] = s_delta[tid] - g_retrieved[tid];
                    }
                    __syncthreads();
                }
            }

            // Load cached gates
            for (int i = tid; i < n2; i += blockDim.x) {
                gate_S_local[i] = gate_S_cache[t * B * n2 + b * n2 + i];
                gate_G_local[i] = gate_G_cache[t * B * n2 + b * n2 + i];
            }
            __syncthreads();

            // Backward through output
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
                for (int j = 0; j < N_STATE; j++) {
                    float gate_val = gate_S_local[j * N_STATE + tid];
                    float S_t_jt = gate_val * S[j * N_STATE + tid] + s_delta[j] * k_norm[tid];
                    sum += S_t_jt * d_Sq_shared[j];
                }
                d_q_raw[tid] = sum;
            }
            __syncthreads();

            // G backward
            if (tid < N_STATE) {
                float d_gdt = 0.0f, d_mn = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    float dG_ij = dG[tid * N_STATE + j];
                    d_gdt += dG_ij * m_norm[j];
                    d_mn += dG[j * N_STATE + tid] * g_delta[j];
                }
                d_g_delta[tid] = d_gdt;
                d_m_norm[tid] = d_mn;
            }
            __syncthreads();

            // S backward
            if (tid < N_STATE) {
                float d_sdt = 0.0f, d_kn = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    float dS_ij = dS[tid * N_STATE + j];
                    d_sdt += dS_ij * k_norm[j];
                    d_kn += dS[j * N_STATE + tid] * s_delta[j];
                }
                d_s_delta[tid] = d_sdt + d_g_delta[tid];
                d_k_norm[tid] = d_kn;
            }
            __syncthreads();

            // Bias gradients (simplified - accumulate row-wise)
            if (tid < N_STATE) {
                float s_bias_grad = 0.0f, g_bias_grad = 0.0f;
                for (int col = 0; col < N_STATE; col++) {
                    int idx = tid * N_STATE + col;
                    // d_gate_S contribution
                    float gate_S_i = gate_S_local[idx];
                    float d_gate_S = dS[idx] * S[idx];
                    s_bias_grad += d_gate_S * gate_S_i * (1.0f - gate_S_i);
                    // d_gate_G contribution
                    float gate_G_i = gate_G_local[idx];
                    float d_gate_G = dG[idx] * G[idx];
                    g_bias_grad += d_gate_G * gate_G_i * (1.0f - gate_G_i);
                }
                d_b_s_gate_local[tid] += s_bias_grad;
                d_b_g_gate_local[tid] += g_bias_grad;
            }
            __syncthreads();

            // dS from gate_G, dG from gate_S
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                float dG_i = dG[i];
                float gate_G_i = gate_G_local[i];
                float d_gate_G_i = dG_i * G[i];
                dS[i] += d_gate_G_i * gate_G_i * (1.0f - gate_G_i);

                float dS_i = dS[i];
                float gate_S_i = gate_S_local[i];
                float d_gate_S_i = dS_i * S[i];
                dG[i] += d_gate_S_i * gate_S_i * (1.0f - gate_S_i);
            }
            __syncthreads();

            if (tid < N_STATE) {
                d_v_raw[tid] = d_s_delta[tid];
            }

            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    sum += S[j * N_STATE + tid] * (-d_s_delta[j]);
                }
                d_k_norm[tid] += sum;
            }
            __syncthreads();

            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    sum += G[j * N_STATE + tid] * d_g_delta[j];
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
                dS[i] = dS[i] * gate_S_local[i] + (-d_s_delta[row]) * k_norm[col];
            }
            __syncthreads();

            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                dG[i] = dG[i] * gate_G_local[i] + (-d_g_delta[row]) * m_norm[col];
            }
            __syncthreads();
        }
    }

    if (tid < N_STATE) {
        atomicAdd(&d_b_s_gate_accum[tid], d_b_s_gate_local[tid]);
        atomicAdd(&d_b_g_gate_accum[tid], d_b_g_gate_local[tid]);
    }
}

// ============================================================================
// Template instantiations
// ============================================================================

#define INSTANTIATE_E81_KERNELS_BF16(N) \
    template __global__ void E81GateAsStateForwardKernel_BF16<N>( \
        int, int, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, \
        __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, \
        __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int); \
    template __global__ void E81GateAsStateBackwardKernel_BF16<N>( \
        int, int, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, \
        const __nv_bfloat16*, const __nv_bfloat16*, \
        const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, \
        __nv_bfloat16*, float*, float*, int);

#define INSTANTIATE_E81_KERNELS_FP32(N) \
    template __global__ void E81GateAsStateForwardKernel_FP32<N>( \
        int, int, const float*, const float*, const float*, \
        float*, float*, float*, float*, float*, \
        float*, float*, float*, int); \
    template __global__ void E81GateAsStateBackwardKernel_FP32<N>( \
        int, int, const float*, const float*, const float*, \
        const float*, const float*, \
        const float*, const float*, const float*, const float*, \
        float*, float*, float*, int);

// Standard kernels
INSTANTIATE_E81_KERNELS_BF16(8)
INSTANTIATE_E81_KERNELS_BF16(16)
INSTANTIATE_E81_KERNELS_BF16(32)
INSTANTIATE_E81_KERNELS_BF16(48)
INSTANTIATE_E81_KERNELS_BF16(64)

INSTANTIATE_E81_KERNELS_FP32(8)
INSTANTIATE_E81_KERNELS_FP32(16)
INSTANTIATE_E81_KERNELS_FP32(32)
INSTANTIATE_E81_KERNELS_FP32(48)
INSTANTIATE_E81_KERNELS_FP32(64)

// ============================================================================
// Dispatcher functions
// ============================================================================

void dispatch_e81_gate_as_state_forward(
    int T, int B, int n_state,
    const __nv_bfloat16* kvqm_all,
    const __nv_bfloat16* b_s_gate,
    const __nv_bfloat16* b_g_gate,
    __nv_bfloat16* S, __nv_bfloat16* G, __nv_bfloat16* output,
    __nv_bfloat16* S_checkpoints, __nv_bfloat16* G_checkpoints,
    __nv_bfloat16* Sq_cache,
    __nv_bfloat16* gate_S_cache, __nv_bfloat16* gate_G_cache,
    int checkpoint_interval, cudaStream_t stream
) {
    // Shared memory: 2*n^2 (S, G) + 11*n (vectors)
    int shared_size = (2 * n_state * n_state + 11 * n_state) * sizeof(float);

    #define DISPATCH_E81_FWD(N) \
        E81GateAsStateForwardKernel_BF16<N><<<B, 256, shared_size, stream>>>( \
            T, B, kvqm_all, b_s_gate, b_g_gate, \
            S, G, output, S_checkpoints, G_checkpoints, Sq_cache, \
            gate_S_cache, gate_G_cache, checkpoint_interval);

    #define DISPATCH_E81_FWD_EXT(N) \
        { \
            cudaError_t attr_err = cudaFuncSetAttribute( \
                E81GateAsStateForwardKernel_BF16<N>, \
                cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size); \
            if (attr_err != cudaSuccess) { \
                fprintf(stderr, "E81 Forward: n_state=%d cudaFuncSetAttribute failed: %s\n", \
                        N, cudaGetErrorString(attr_err)); \
            } else { \
                DISPATCH_E81_FWD(N); \
            } \
        }

    switch (n_state) {
        case 8: DISPATCH_E81_FWD(8); break;
        case 16: DISPATCH_E81_FWD(16); break;
        case 32: DISPATCH_E81_FWD(32); break;
        case 48: DISPATCH_E81_FWD(48); break;
        case 64: DISPATCH_E81_FWD_EXT(64); break;
        default:
            fprintf(stderr, "E81: Unsupported n_state=%d (use 8, 16, 32, 48, or 64)\n", n_state);
    }
    #undef DISPATCH_E81_FWD
    #undef DISPATCH_E81_FWD_EXT
}

void dispatch_e81_gate_as_state_backward(
    int T, int B, int n_state,
    const __nv_bfloat16* kvqm_all,
    const __nv_bfloat16* b_s_gate, const __nv_bfloat16* b_g_gate,
    const __nv_bfloat16* gate_S_cache, const __nv_bfloat16* gate_G_cache,
    const __nv_bfloat16* S_checkpoints, const __nv_bfloat16* G_checkpoints,
    const __nv_bfloat16* Sq_cache, const __nv_bfloat16* d_output,
    __nv_bfloat16* d_kvqm_all,
    float* d_b_s_gate_accum, float* d_b_g_gate_accum,
    int checkpoint_interval, cudaStream_t stream
) {
    // Shared memory for backward: 4*n^2 + 26*n + 2*n^2 for gate caches
    int shared_size = (4 * n_state * n_state + 26 * n_state) * sizeof(float);

    #define DISPATCH_E81_BWD(N) \
        E81GateAsStateBackwardKernel_BF16<N><<<B, 256, shared_size, stream>>>( \
            T, B, kvqm_all, b_s_gate, b_g_gate, \
            gate_S_cache, gate_G_cache, \
            S_checkpoints, G_checkpoints, Sq_cache, d_output, \
            d_kvqm_all, d_b_s_gate_accum, d_b_g_gate_accum, checkpoint_interval);

    #define DISPATCH_E81_BWD_EXT(N) \
        { \
            cudaError_t attr_err = cudaFuncSetAttribute( \
                E81GateAsStateBackwardKernel_BF16<N>, \
                cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size); \
            if (attr_err != cudaSuccess) { \
                fprintf(stderr, "E81 Backward: n_state=%d requires %d KB shared memory but GPU limit exceeded\n", \
                        N, shared_size / 1024); \
            } else { \
                DISPATCH_E81_BWD(N); \
            } \
        }

    switch (n_state) {
        case 8: DISPATCH_E81_BWD(8); break;
        case 16: DISPATCH_E81_BWD(16); break;
        case 32: DISPATCH_E81_BWD(32); break;
        case 48: DISPATCH_E81_BWD_EXT(48); break;
        case 64: DISPATCH_E81_BWD_EXT(64); break;
        default:
            fprintf(stderr, "E81: Unsupported n_state=%d (use 8, 16, 32, 48, or 64)\n", n_state);
    }
    #undef DISPATCH_E81_BWD
    #undef DISPATCH_E81_BWD_EXT
}

void dispatch_e81_gate_as_state_forward_fp32(
    int T, int B, int n_state,
    const float* kvqm_all,
    const float* b_s_gate, const float* b_g_gate,
    float* S, float* G, float* output,
    float* S_checkpoints, float* G_checkpoints,
    float* Sq_cache,
    float* gate_S_cache, float* gate_G_cache,
    int checkpoint_interval, cudaStream_t stream
) {
    int shared_size = (2 * n_state * n_state + 11 * n_state) * sizeof(float);

    #define DISPATCH_E81_FWD_FP32(N) \
        E81GateAsStateForwardKernel_FP32<N><<<B, 256, shared_size, stream>>>( \
            T, B, kvqm_all, b_s_gate, b_g_gate, \
            S, G, output, S_checkpoints, G_checkpoints, Sq_cache, \
            gate_S_cache, gate_G_cache, checkpoint_interval);

    #define DISPATCH_E81_FWD_FP32_EXT(N) \
        { \
            cudaError_t attr_err = cudaFuncSetAttribute( \
                E81GateAsStateForwardKernel_FP32<N>, \
                cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size); \
            if (attr_err != cudaSuccess) { \
                fprintf(stderr, "E81 FP32 Forward: n_state=%d cudaFuncSetAttribute failed: %s\n", \
                        N, cudaGetErrorString(attr_err)); \
            } else { \
                DISPATCH_E81_FWD_FP32(N); \
            } \
        }

    switch (n_state) {
        case 8: DISPATCH_E81_FWD_FP32(8); break;
        case 16: DISPATCH_E81_FWD_FP32(16); break;
        case 32: DISPATCH_E81_FWD_FP32(32); break;
        case 48: DISPATCH_E81_FWD_FP32(48); break;
        case 64: DISPATCH_E81_FWD_FP32_EXT(64); break;
        default:
            fprintf(stderr, "E81: Unsupported n_state=%d (use 8, 16, 32, 48, or 64)\n", n_state);
    }
    #undef DISPATCH_E81_FWD_FP32
    #undef DISPATCH_E81_FWD_FP32_EXT
}

void dispatch_e81_gate_as_state_backward_fp32(
    int T, int B, int n_state,
    const float* kvqm_all,
    const float* b_s_gate, const float* b_g_gate,
    const float* gate_S_cache, const float* gate_G_cache,
    const float* S_checkpoints, const float* G_checkpoints,
    const float* Sq_cache, const float* d_output,
    float* d_kvqm_all,
    float* d_b_s_gate_accum, float* d_b_g_gate_accum,
    int checkpoint_interval, cudaStream_t stream
) {
    // FP32 backward needs extra space for gate caches in shared memory
    int shared_size = (4 * n_state * n_state + 26 * n_state + 2 * n_state * n_state) * sizeof(float);

    #define DISPATCH_E81_BWD_FP32(N) \
        E81GateAsStateBackwardKernel_FP32<N><<<B, 256, shared_size, stream>>>( \
            T, B, kvqm_all, b_s_gate, b_g_gate, \
            gate_S_cache, gate_G_cache, \
            S_checkpoints, G_checkpoints, Sq_cache, d_output, \
            d_kvqm_all, d_b_s_gate_accum, d_b_g_gate_accum, checkpoint_interval);

    #define DISPATCH_E81_BWD_FP32_EXT(N) \
        { \
            cudaError_t attr_err = cudaFuncSetAttribute( \
                E81GateAsStateBackwardKernel_FP32<N>, \
                cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size); \
            if (attr_err != cudaSuccess) { \
                fprintf(stderr, "E81 FP32 Backward: n_state=%d requires %d KB shared memory but GPU limit exceeded\n", \
                        N, shared_size / 1024); \
            } else { \
                DISPATCH_E81_BWD_FP32(N); \
            } \
        }

    switch (n_state) {
        case 8: DISPATCH_E81_BWD_FP32(8); break;
        case 16: DISPATCH_E81_BWD_FP32(16); break;
        case 32: DISPATCH_E81_BWD_FP32_EXT(32); break;
        case 48: DISPATCH_E81_BWD_FP32_EXT(48); break;
        case 64: DISPATCH_E81_BWD_FP32_EXT(64); break;
        default:
            fprintf(stderr, "E81: Unsupported n_state=%d (use 8, 16, 32, 48, or 64)\n", n_state);
    }
    #undef DISPATCH_E81_BWD_FP32
    #undef DISPATCH_E81_BWD_FP32_EXT
}

// ============================================================================
// E81GateAsStateForward Implementation (wrapper class for Python bindings)
// ============================================================================

template<typename DataT>
E81GateAsStateForward<DataT>::E81GateAsStateForward(
    bool training, int batch_size, int n_state, int dim,
    const cublasHandle_t& blas_handle, const cudaStream_t& stream)
    : training_(training), batch_size_(batch_size), n_state_(n_state),
      dim_(dim), blas_handle_(blas_handle), stream_(stream) {}

template<typename DataT>
void E81GateAsStateForward<DataT>::Run(
    int steps,
    const DataT* W_kvqm,
    const DataT* b_s_gate,
    const DataT* b_g_gate,
    const DataT* x,
    DataT* S, DataT* G,
    DataT* output,
    DataT* kvqm_cache,
    DataT* S_checkpoints, DataT* G_checkpoints,
    DataT* Sq_cache,
    DataT* gate_S_cache, DataT* gate_G_cache
) {
    int T = steps;
    int B = batch_size_;
    int n = n_state_;
    int d = dim_;
    int checkpoint_interval = E81_CHECKPOINT_INTERVAL;

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
        dispatch_e81_gate_as_state_forward(T, B, n, kvqm_cache, b_s_gate, b_g_gate,
                                           S, G, output, S_checkpoints, G_checkpoints, Sq_cache,
                                           gate_S_cache, gate_G_cache,
                                           checkpoint_interval, stream_);
    } else {
        dispatch_e81_gate_as_state_forward_fp32(T, B, n,
                                                reinterpret_cast<const float*>(kvqm_cache),
                                                reinterpret_cast<const float*>(b_s_gate),
                                                reinterpret_cast<const float*>(b_g_gate),
                                                reinterpret_cast<float*>(S),
                                                reinterpret_cast<float*>(G),
                                                reinterpret_cast<float*>(output),
                                                reinterpret_cast<float*>(S_checkpoints),
                                                reinterpret_cast<float*>(G_checkpoints),
                                                reinterpret_cast<float*>(Sq_cache),
                                                reinterpret_cast<float*>(gate_S_cache),
                                                reinterpret_cast<float*>(gate_G_cache),
                                                checkpoint_interval, stream_);
    }
}

template<typename DataT>
E81GateAsStateBackward<DataT>::E81GateAsStateBackward(
    int batch_size, int n_state, int dim,
    const cublasHandle_t& blas_handle, const cudaStream_t& stream)
    : batch_size_(batch_size), n_state_(n_state), dim_(dim),
      blas_handle_(blas_handle), stream_(stream) {}

template<typename DataT>
void E81GateAsStateBackward<DataT>::Run(
    int steps,
    const DataT* W_kvqm,
    const DataT* b_s_gate, const DataT* b_g_gate,
    const DataT* x,
    const DataT* kvqm_cache,
    const DataT* S_checkpoints, const DataT* G_checkpoints,
    const DataT* Sq_cache,
    const DataT* gate_S_cache, const DataT* gate_G_cache,
    const DataT* d_output,
    DataT* d_x,
    DataT* d_W_kvqm,
    DataT* d_b_s_gate, DataT* d_b_g_gate,
    DataT* d_kvqm_cache,
    float* d_b_s_gate_accum, float* d_b_g_gate_accum
) {
    int T = steps;
    int B = batch_size_;
    int n = n_state_;
    int d = dim_;
    int checkpoint_interval = E81_CHECKPOINT_INTERVAL;

    const float alpha = 1.0f, beta_zero = 0.0f;

    // Zero accumulators
    cudaMemsetAsync(d_b_s_gate_accum, 0, n * sizeof(float), stream_);
    cudaMemsetAsync(d_b_g_gate_accum, 0, n * sizeof(float), stream_);

    if constexpr (std::is_same<DataT, __nv_bfloat16>::value) {
        dispatch_e81_gate_as_state_backward(T, B, n, kvqm_cache, b_s_gate, b_g_gate,
                                            gate_S_cache, gate_G_cache,
                                            S_checkpoints, G_checkpoints, Sq_cache, d_output,
                                            d_kvqm_cache, d_b_s_gate_accum, d_b_g_gate_accum,
                                            checkpoint_interval, stream_);
    } else {
        dispatch_e81_gate_as_state_backward_fp32(T, B, n,
                                                 reinterpret_cast<const float*>(kvqm_cache),
                                                 reinterpret_cast<const float*>(b_s_gate),
                                                 reinterpret_cast<const float*>(b_g_gate),
                                                 reinterpret_cast<const float*>(gate_S_cache),
                                                 reinterpret_cast<const float*>(gate_G_cache),
                                                 reinterpret_cast<const float*>(S_checkpoints),
                                                 reinterpret_cast<const float*>(G_checkpoints),
                                                 reinterpret_cast<const float*>(Sq_cache),
                                                 reinterpret_cast<const float*>(d_output),
                                                 reinterpret_cast<float*>(d_kvqm_cache),
                                                 d_b_s_gate_accum, d_b_g_gate_accum,
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

    // d_W_kvqm = d_kvqm_cache @ x^T
    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
                 4 * n, d, T * B,
                 &alpha,
                 d_kvqm_cache, data_type, 4 * n,
                 x, data_type, d,
                 &beta_zero,
                 d_W_kvqm, data_type, 4 * n,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // Convert accumulated bias gradients (float) to output dtype
    if constexpr (std::is_same<DataT, __nv_bfloat16>::value) {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        ConvertFloatToBF16Kernel_E81<<<blocks, threads, 0, stream_>>>(
            d_b_s_gate_accum, d_b_s_gate, n);
        ConvertFloatToBF16Kernel_E81<<<blocks, threads, 0, stream_>>>(
            d_b_g_gate_accum, d_b_g_gate, n);
    } else {
        cudaMemcpyAsync(d_b_s_gate, d_b_s_gate_accum, n * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream_);
        cudaMemcpyAsync(d_b_g_gate, d_b_g_gate_accum, n * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream_);
    }
}

// Explicit template instantiations
template class E81GateAsStateForward<__nv_bfloat16>;
template class E81GateAsStateForward<float>;
template class E81GateAsStateBackward<__nv_bfloat16>;
template class E81GateAsStateBackward<float>;

}  // namespace elman
