/**
 * E84 Neural ODE CUDA Kernel - Continuous-time self-modulation
 *
 * Two coupled matrix states that evolve according to ODE dynamics:
 * - S [n_state x n_state]: Content memory
 * - G [n_state x n_state]: Modulation memory (controls S evolution)
 *
 * Mathematical Definition:
 *   dS/dt = -S + sigmoid(G @ k) * S + outer(v - S @ k_norm, k_norm)
 *   dG/dt = -G + sigmoid(S @ m) * G + outer(delta_S - G @ m_norm, m_norm)
 *
 *   Integrate from t=0 to t=1 using RK4 for n_steps
 *
 *   output = (S @ q) * silu(S @ q)
 *
 * Key insight: Adaptive computation through continuous-time integration.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include "hasty/elman_ladder.h"

#define E84_CHECKPOINT_INTERVAL 16

namespace elman {

// ============================================================================
// Helper device functions for ODE dynamics
// ============================================================================

// Compute gate values: sigmoid(M @ v) where M is [n x n] and v is [n]
template<int N_STATE>
__device__ __forceinline__ float compute_gate(
    const float* __restrict__ M,
    const float* __restrict__ v,
    int row
) {
    float sum = 0.0f;
    #pragma unroll 8
    for (int j = 0; j < N_STATE; j++) {
        sum += M[row * N_STATE + j] * v[j];
    }
    return 1.0f / (1.0f + expf(-sum));
}

// Compute retrieved value: M @ v
template<int N_STATE>
__device__ __forceinline__ float compute_retrieved(
    const float* __restrict__ M,
    const float* __restrict__ v,
    int row
) {
    float sum = 0.0f;
    #pragma unroll 8
    for (int j = 0; j < N_STATE; j++) {
        sum += M[row * N_STATE + j] * v[j];
    }
    return sum;
}

// ============================================================================
// E84 Forward Kernel with RK4 Integration
// ============================================================================

template<int N_STATE>
__global__ void E84NeuralODEForwardKernel_BF16(
    int T,
    int B,
    int n_steps,
    float dt,
    const __nv_bfloat16* __restrict__ kvqm_all,   // [4*N_STATE, T*B] column-major
    __nv_bfloat16* __restrict__ S,                // [B, N_STATE, N_STATE]
    __nv_bfloat16* __restrict__ G,                // [B, N_STATE, N_STATE]
    __nv_bfloat16* __restrict__ output,           // [T, B, N_STATE]
    __nv_bfloat16* __restrict__ S_checkpoints,    // [num_checkpoints, B, N_STATE, N_STATE]
    __nv_bfloat16* __restrict__ G_checkpoints,    // [num_checkpoints, B, N_STATE, N_STATE]
    __nv_bfloat16* __restrict__ Sq_cache,         // [T, B, N_STATE]
    int checkpoint_interval
) {
    int b = blockIdx.x;
    if (b >= B) return;

    // Shared memory layout
    extern __shared__ float shared_mem[];
    float* S_shared = shared_mem;                         // [N_STATE * N_STATE]
    float* G_shared = S_shared + N_STATE * N_STATE;       // [N_STATE * N_STATE]
    // Temporary storage for RK4 intermediate states
    float* S_k1 = G_shared + N_STATE * N_STATE;           // [N_STATE * N_STATE]
    float* G_k1 = S_k1 + N_STATE * N_STATE;               // [N_STATE * N_STATE]
    float* S_tmp = G_k1 + N_STATE * N_STATE;              // [N_STATE * N_STATE] for intermediate state
    float* G_tmp = S_tmp + N_STATE * N_STATE;             // [N_STATE * N_STATE]
    float* k_shared = G_tmp + N_STATE * N_STATE;          // [N_STATE]
    float* v_shared = k_shared + N_STATE;                 // [N_STATE]
    float* q_shared = v_shared + N_STATE;                 // [N_STATE]
    float* m_vec_shared = q_shared + N_STATE;             // [N_STATE]
    float* s_gate = m_vec_shared + N_STATE;               // [N_STATE]
    float* g_gate = s_gate + N_STATE;                     // [N_STATE]
    float* s_retrieved = g_gate + N_STATE;                // [N_STATE]
    float* g_retrieved = s_retrieved + N_STATE;           // [N_STATE]
    float* delta_S = g_retrieved + N_STATE;               // [N_STATE]
    float* delta_G = delta_S + N_STATE;                   // [N_STATE]

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;
    const int STRIDE = 4 * N_STATE;

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

        // RK4 integration for n_steps
        for (int step = 0; step < n_steps; step++) {
            // ======== k1 ========
            // Compute gates: s_gate = sigmoid(G @ k), g_gate = sigmoid(S @ m)
            if (tid < N_STATE) {
                float g_row_sum = 0.0f, s_row_sum = 0.0f;
                #pragma unroll 8
                for (int j = 0; j < N_STATE; j++) {
                    g_row_sum += G_shared[tid * N_STATE + j] * k_shared[j];
                    s_row_sum += S_shared[tid * N_STATE + j] * m_vec_shared[j];
                }
                s_gate[tid] = 1.0f / (1.0f + expf(-g_row_sum));
                g_gate[tid] = 1.0f / (1.0f + expf(-s_row_sum));
            }
            __syncthreads();

            // Compute retrieved: s_retrieved = S @ k, g_retrieved = G @ m
            if (tid < N_STATE) {
                float s_sum = 0.0f, g_sum = 0.0f;
                #pragma unroll 8
                for (int j = 0; j < N_STATE; j++) {
                    s_sum += S_shared[tid * N_STATE + j] * k_shared[j];
                    g_sum += G_shared[tid * N_STATE + j] * m_vec_shared[j];
                }
                s_retrieved[tid] = s_sum;
                delta_S[tid] = v_shared[tid] - s_sum;
                delta_G[tid] = delta_S[tid] - g_sum;
            }
            __syncthreads();

            // Compute k1 = f(S, G): dS/dt, dG/dt
            // dS/dt = -S + s_gate * S + outer(delta_S, k)
            // dG/dt = -G + g_gate * G + outer(delta_G, m)
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                S_k1[i] = -S_shared[i] + s_gate[row] * S_shared[i] + delta_S[row] * k_shared[col];
                G_k1[i] = -G_shared[i] + g_gate[row] * G_shared[i] + delta_G[row] * m_vec_shared[col];
            }
            __syncthreads();

            // ======== k2: f(S + 0.5*dt*k1, G + 0.5*dt*k1) ========
            // Compute temporary state
            for (int i = tid; i < n2; i += blockDim.x) {
                S_tmp[i] = S_shared[i] + 0.5f * dt * S_k1[i];
                G_tmp[i] = G_shared[i] + 0.5f * dt * G_k1[i];
            }
            __syncthreads();

            // Compute gates for k2
            if (tid < N_STATE) {
                float g_row_sum = 0.0f, s_row_sum = 0.0f;
                #pragma unroll 8
                for (int j = 0; j < N_STATE; j++) {
                    g_row_sum += G_tmp[tid * N_STATE + j] * k_shared[j];
                    s_row_sum += S_tmp[tid * N_STATE + j] * m_vec_shared[j];
                }
                s_gate[tid] = 1.0f / (1.0f + expf(-g_row_sum));
                g_gate[tid] = 1.0f / (1.0f + expf(-s_row_sum));
            }
            __syncthreads();

            if (tid < N_STATE) {
                float s_sum = 0.0f, g_sum = 0.0f;
                #pragma unroll 8
                for (int j = 0; j < N_STATE; j++) {
                    s_sum += S_tmp[tid * N_STATE + j] * k_shared[j];
                    g_sum += G_tmp[tid * N_STATE + j] * m_vec_shared[j];
                }
                delta_S[tid] = v_shared[tid] - s_sum;
                delta_G[tid] = delta_S[tid] - g_sum;
            }
            __syncthreads();

            // Accumulate k2 into running sum (stored in S_k1, G_k1)
            // k1 + 2*k2 for final combination
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                float dS = -S_tmp[i] + s_gate[row] * S_tmp[i] + delta_S[row] * k_shared[col];
                float dG = -G_tmp[i] + g_gate[row] * G_tmp[i] + delta_G[row] * m_vec_shared[col];
                S_k1[i] += 2.0f * dS;
                G_k1[i] += 2.0f * dG;
                // Update S_tmp for k3
                S_tmp[i] = S_shared[i] + 0.5f * dt * dS;
                G_tmp[i] = G_shared[i] + 0.5f * dt * dG;
            }
            __syncthreads();

            // ======== k3: f(S + 0.5*dt*k2, G + 0.5*dt*k2) ========
            if (tid < N_STATE) {
                float g_row_sum = 0.0f, s_row_sum = 0.0f;
                #pragma unroll 8
                for (int j = 0; j < N_STATE; j++) {
                    g_row_sum += G_tmp[tid * N_STATE + j] * k_shared[j];
                    s_row_sum += S_tmp[tid * N_STATE + j] * m_vec_shared[j];
                }
                s_gate[tid] = 1.0f / (1.0f + expf(-g_row_sum));
                g_gate[tid] = 1.0f / (1.0f + expf(-s_row_sum));
            }
            __syncthreads();

            if (tid < N_STATE) {
                float s_sum = 0.0f, g_sum = 0.0f;
                #pragma unroll 8
                for (int j = 0; j < N_STATE; j++) {
                    s_sum += S_tmp[tid * N_STATE + j] * k_shared[j];
                    g_sum += G_tmp[tid * N_STATE + j] * m_vec_shared[j];
                }
                delta_S[tid] = v_shared[tid] - s_sum;
                delta_G[tid] = delta_S[tid] - g_sum;
            }
            __syncthreads();

            // Accumulate k3 and prepare for k4
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                float dS = -S_tmp[i] + s_gate[row] * S_tmp[i] + delta_S[row] * k_shared[col];
                float dG = -G_tmp[i] + g_gate[row] * G_tmp[i] + delta_G[row] * m_vec_shared[col];
                S_k1[i] += 2.0f * dS;
                G_k1[i] += 2.0f * dG;
                // Update S_tmp for k4
                S_tmp[i] = S_shared[i] + dt * dS;
                G_tmp[i] = G_shared[i] + dt * dG;
            }
            __syncthreads();

            // ======== k4: f(S + dt*k3, G + dt*k3) ========
            if (tid < N_STATE) {
                float g_row_sum = 0.0f, s_row_sum = 0.0f;
                #pragma unroll 8
                for (int j = 0; j < N_STATE; j++) {
                    g_row_sum += G_tmp[tid * N_STATE + j] * k_shared[j];
                    s_row_sum += S_tmp[tid * N_STATE + j] * m_vec_shared[j];
                }
                s_gate[tid] = 1.0f / (1.0f + expf(-g_row_sum));
                g_gate[tid] = 1.0f / (1.0f + expf(-s_row_sum));
            }
            __syncthreads();

            if (tid < N_STATE) {
                float s_sum = 0.0f, g_sum = 0.0f;
                #pragma unroll 8
                for (int j = 0; j < N_STATE; j++) {
                    s_sum += S_tmp[tid * N_STATE + j] * k_shared[j];
                    g_sum += G_tmp[tid * N_STATE + j] * m_vec_shared[j];
                }
                delta_S[tid] = v_shared[tid] - s_sum;
                delta_G[tid] = delta_S[tid] - g_sum;
            }
            __syncthreads();

            // Final RK4 update: S += (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                float dS_k4 = -S_tmp[i] + s_gate[row] * S_tmp[i] + delta_S[row] * k_shared[col];
                float dG_k4 = -G_tmp[i] + g_gate[row] * G_tmp[i] + delta_G[row] * m_vec_shared[col];
                S_shared[i] += (dt / 6.0f) * (S_k1[i] + dS_k4);
                G_shared[i] += (dt / 6.0f) * (G_k1[i] + dG_k4);
            }
            __syncthreads();
        }

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
// E84 Backward Kernel with RK4 Adjoint Method
// ============================================================================

template<int N_STATE>
__global__ void E84NeuralODEBackwardKernel_BF16(
    int T,
    int B,
    int n_steps,
    float dt,
    const __nv_bfloat16* __restrict__ kvqm_all,
    const __nv_bfloat16* __restrict__ S_checkpoints,
    const __nv_bfloat16* __restrict__ G_checkpoints,
    const __nv_bfloat16* __restrict__ Sq_cache,
    const __nv_bfloat16* __restrict__ d_output,
    __nv_bfloat16* __restrict__ d_kvqm_all,
    int checkpoint_interval
) {
    int b = blockIdx.x;
    if (b >= B) return;

    extern __shared__ float shared_mem[];
    // Layout for backward - need more space
    float* S = shared_mem;                                // [N_STATE * N_STATE]
    float* G = S + N_STATE * N_STATE;                     // [N_STATE * N_STATE]
    float* dS = G + N_STATE * N_STATE;                    // [N_STATE * N_STATE]
    float* dG = dS + N_STATE * N_STATE;                   // [N_STATE * N_STATE]
    // RK4 storage for backward
    float* dS_k1 = dG + N_STATE * N_STATE;                // [N_STATE * N_STATE]
    float* dG_k1 = dS_k1 + N_STATE * N_STATE;             // [N_STATE * N_STATE]
    float* S_tmp = dG_k1 + N_STATE * N_STATE;             // [N_STATE * N_STATE]
    float* G_tmp = S_tmp + N_STATE * N_STATE;             // [N_STATE * N_STATE]
    float* k_raw = G_tmp + N_STATE * N_STATE;             // [N_STATE]
    float* v_raw = k_raw + N_STATE;                       // [N_STATE]
    float* q_raw = v_raw + N_STATE;                       // [N_STATE]
    float* m_vec_raw = q_raw + N_STATE;                   // [N_STATE]
    float* k_norm = m_vec_raw + N_STATE;                  // [N_STATE]
    float* m_norm = k_norm + N_STATE;                     // [N_STATE]
    float* s_gate = m_norm + N_STATE;                     // [N_STATE]
    float* g_gate = s_gate + N_STATE;                     // [N_STATE]
    float* delta_S = g_gate + N_STATE;                    // [N_STATE]
    float* delta_G = delta_S + N_STATE;                   // [N_STATE]
    float* d_k_raw = delta_G + N_STATE;                   // [N_STATE]
    float* d_v_raw = d_k_raw + N_STATE;                   // [N_STATE]
    float* d_q_raw = d_v_raw + N_STATE;                   // [N_STATE]
    float* d_m_raw = d_q_raw + N_STATE;                   // [N_STATE]
    float* d_Sq_shared = d_m_raw + N_STATE;               // [N_STATE]
    float* d_delta_S = d_Sq_shared + N_STATE;             // [N_STATE]
    float* d_delta_G = d_delta_S + N_STATE;               // [N_STATE]
    float* d_k_norm = d_delta_G + N_STATE;                // [N_STATE]
    float* d_m_norm = d_k_norm + N_STATE;                 // [N_STATE]

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;
    const int STRIDE = 4 * N_STATE;

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

                // Forward pass for RK4 to time tt (or recompute if tt < t)
                if (tt < t) {
                    // Full RK4 forward for this step
                    for (int step = 0; step < n_steps; step++) {
                        // Simplified RK4 - compute k1-k4 and update
                        // (same structure as forward kernel)
                        if (tid < N_STATE) {
                            float g_row_sum = 0.0f, s_row_sum = 0.0f;
                            for (int j = 0; j < N_STATE; j++) {
                                g_row_sum += G[tid * N_STATE + j] * k_norm[j];
                                s_row_sum += S[tid * N_STATE + j] * m_norm[j];
                            }
                            s_gate[tid] = 1.0f / (1.0f + expf(-g_row_sum));
                            g_gate[tid] = 1.0f / (1.0f + expf(-s_row_sum));
                        }
                        __syncthreads();

                        if (tid < N_STATE) {
                            float s_sum = 0.0f, g_sum = 0.0f;
                            for (int j = 0; j < N_STATE; j++) {
                                s_sum += S[tid * N_STATE + j] * k_norm[j];
                                g_sum += G[tid * N_STATE + j] * m_norm[j];
                            }
                            delta_S[tid] = v_raw[tid] - s_sum;
                            delta_G[tid] = delta_S[tid] - g_sum;
                        }
                        __syncthreads();

                        // RK4 k1
                        for (int i = tid; i < n2; i += blockDim.x) {
                            int row = i / N_STATE;
                            int col = i % N_STATE;
                            dS_k1[i] = -S[i] + s_gate[row] * S[i] + delta_S[row] * k_norm[col];
                            dG_k1[i] = -G[i] + g_gate[row] * G[i] + delta_G[row] * m_norm[col];
                            S_tmp[i] = S[i] + 0.5f * dt * dS_k1[i];
                            G_tmp[i] = G[i] + 0.5f * dt * dG_k1[i];
                        }
                        __syncthreads();

                        // k2
                        if (tid < N_STATE) {
                            float g_row_sum = 0.0f, s_row_sum = 0.0f;
                            for (int j = 0; j < N_STATE; j++) {
                                g_row_sum += G_tmp[tid * N_STATE + j] * k_norm[j];
                                s_row_sum += S_tmp[tid * N_STATE + j] * m_norm[j];
                            }
                            s_gate[tid] = 1.0f / (1.0f + expf(-g_row_sum));
                            g_gate[tid] = 1.0f / (1.0f + expf(-s_row_sum));
                        }
                        __syncthreads();

                        if (tid < N_STATE) {
                            float s_sum = 0.0f, g_sum = 0.0f;
                            for (int j = 0; j < N_STATE; j++) {
                                s_sum += S_tmp[tid * N_STATE + j] * k_norm[j];
                                g_sum += G_tmp[tid * N_STATE + j] * m_norm[j];
                            }
                            delta_S[tid] = v_raw[tid] - s_sum;
                            delta_G[tid] = delta_S[tid] - g_sum;
                        }
                        __syncthreads();

                        for (int i = tid; i < n2; i += blockDim.x) {
                            int row = i / N_STATE;
                            int col = i % N_STATE;
                            float dS2 = -S_tmp[i] + s_gate[row] * S_tmp[i] + delta_S[row] * k_norm[col];
                            float dG2 = -G_tmp[i] + g_gate[row] * G_tmp[i] + delta_G[row] * m_norm[col];
                            dS_k1[i] += 2.0f * dS2;
                            dG_k1[i] += 2.0f * dG2;
                            S_tmp[i] = S[i] + 0.5f * dt * dS2;
                            G_tmp[i] = G[i] + 0.5f * dt * dG2;
                        }
                        __syncthreads();

                        // k3
                        if (tid < N_STATE) {
                            float g_row_sum = 0.0f, s_row_sum = 0.0f;
                            for (int j = 0; j < N_STATE; j++) {
                                g_row_sum += G_tmp[tid * N_STATE + j] * k_norm[j];
                                s_row_sum += S_tmp[tid * N_STATE + j] * m_norm[j];
                            }
                            s_gate[tid] = 1.0f / (1.0f + expf(-g_row_sum));
                            g_gate[tid] = 1.0f / (1.0f + expf(-s_row_sum));
                        }
                        __syncthreads();

                        if (tid < N_STATE) {
                            float s_sum = 0.0f, g_sum = 0.0f;
                            for (int j = 0; j < N_STATE; j++) {
                                s_sum += S_tmp[tid * N_STATE + j] * k_norm[j];
                                g_sum += G_tmp[tid * N_STATE + j] * m_norm[j];
                            }
                            delta_S[tid] = v_raw[tid] - s_sum;
                            delta_G[tid] = delta_S[tid] - g_sum;
                        }
                        __syncthreads();

                        for (int i = tid; i < n2; i += blockDim.x) {
                            int row = i / N_STATE;
                            int col = i % N_STATE;
                            float dS3 = -S_tmp[i] + s_gate[row] * S_tmp[i] + delta_S[row] * k_norm[col];
                            float dG3 = -G_tmp[i] + g_gate[row] * G_tmp[i] + delta_G[row] * m_norm[col];
                            dS_k1[i] += 2.0f * dS3;
                            dG_k1[i] += 2.0f * dG3;
                            S_tmp[i] = S[i] + dt * dS3;
                            G_tmp[i] = G[i] + dt * dG3;
                        }
                        __syncthreads();

                        // k4
                        if (tid < N_STATE) {
                            float g_row_sum = 0.0f, s_row_sum = 0.0f;
                            for (int j = 0; j < N_STATE; j++) {
                                g_row_sum += G_tmp[tid * N_STATE + j] * k_norm[j];
                                s_row_sum += S_tmp[tid * N_STATE + j] * m_norm[j];
                            }
                            s_gate[tid] = 1.0f / (1.0f + expf(-g_row_sum));
                            g_gate[tid] = 1.0f / (1.0f + expf(-s_row_sum));
                        }
                        __syncthreads();

                        if (tid < N_STATE) {
                            float s_sum = 0.0f, g_sum = 0.0f;
                            for (int j = 0; j < N_STATE; j++) {
                                s_sum += S_tmp[tid * N_STATE + j] * k_norm[j];
                                g_sum += G_tmp[tid * N_STATE + j] * m_norm[j];
                            }
                            delta_S[tid] = v_raw[tid] - s_sum;
                            delta_G[tid] = delta_S[tid] - g_sum;
                        }
                        __syncthreads();

                        for (int i = tid; i < n2; i += blockDim.x) {
                            int row = i / N_STATE;
                            int col = i % N_STATE;
                            float dS4 = -S_tmp[i] + s_gate[row] * S_tmp[i] + delta_S[row] * k_norm[col];
                            float dG4 = -G_tmp[i] + g_gate[row] * G_tmp[i] + delta_G[row] * m_norm[col];
                            S[i] += (dt / 6.0f) * (dS_k1[i] + dS4);
                            G[i] += (dt / 6.0f) * (dG_k1[i] + dG4);
                        }
                        __syncthreads();
                    }
                }
            }

            // === BACKWARD PASS FOR STEP t ===
            // Load output gradient
            if (tid < N_STATE) {
                float d_out = __bfloat162float(d_output[t * B * N_STATE + b * N_STATE + tid]);
                float Sq_t = __bfloat162float(Sq_cache[t * B * N_STATE + b * N_STATE + tid]);
                // d_out = d(Sq * silu(Sq))
                // d_Sq = d_out * (silu(Sq) + Sq * silu'(Sq))
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

            // dS += outer(d_Sq, q) (after RK4 integration)
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                dS[i] += d_Sq_shared[row] * q_raw[col];
            }
            __syncthreads();

            // Backward through RK4 integration
            // For simplicity, we use a single-step approximation for the adjoint
            // More accurate would be to integrate the adjoint ODE backward in time
            // Here we approximate: dS/dS_prev = I - dt*(dS/dt terms)

            // Compute current gates and deltas for backward
            if (tid < N_STATE) {
                float g_row_sum = 0.0f, s_row_sum = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    g_row_sum += G[tid * N_STATE + j] * k_norm[j];
                    s_row_sum += S[tid * N_STATE + j] * m_norm[j];
                }
                s_gate[tid] = 1.0f / (1.0f + expf(-g_row_sum));
                g_gate[tid] = 1.0f / (1.0f + expf(-s_row_sum));
            }
            __syncthreads();

            if (tid < N_STATE) {
                float s_sum = 0.0f, g_sum = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    s_sum += S[tid * N_STATE + j] * k_norm[j];
                    g_sum += G[tid * N_STATE + j] * m_norm[j];
                }
                delta_S[tid] = v_raw[tid] - s_sum;
                delta_G[tid] = delta_S[tid] - g_sum;
            }
            __syncthreads();

            // Backward through ODE dynamics
            // dS/dt = -S + s_gate * S + outer(delta_S, k)
            // d_delta_S = dS @ k_norm
            // d_k_norm contribution from outer product
            // d_s_gate contribution from s_gate * S

            if (tid < N_STATE) {
                float d_delta_S_local = 0.0f;
                float d_k_norm_local = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    float dS_ij = dS[tid * N_STATE + j];
                    d_delta_S_local += dS_ij * k_norm[j];
                    d_k_norm_local += dS[j * N_STATE + tid] * delta_S[j];
                }
                d_delta_S[tid] = d_delta_S_local;
                d_k_norm[tid] = d_k_norm_local;
            }
            __syncthreads();

            // d_delta_G = dG @ m_norm
            if (tid < N_STATE) {
                float d_delta_G_local = 0.0f;
                float d_m_norm_local = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    float dG_ij = dG[tid * N_STATE + j];
                    d_delta_G_local += dG_ij * m_norm[j];
                    d_m_norm_local += dG[j * N_STATE + tid] * delta_G[j];
                }
                d_delta_G[tid] = d_delta_G_local;
                d_m_norm[tid] = d_m_norm_local;
            }
            __syncthreads();

            // delta_S = v - S @ k, delta_G = delta_S - G @ m
            // d_v = d_delta_S + d_delta_G (chain through delta_G = delta_S - ...)
            // d_S_retrieved = -d_delta_S
            // d_G_retrieved = -d_delta_G
            if (tid < N_STATE) {
                d_v_raw[tid] = d_delta_S[tid] + d_delta_G[tid];
            }
            __syncthreads();

            // Add contribution from S @ k_norm to d_k_norm
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    sum += S[j * N_STATE + tid] * (-(d_delta_S[j] + d_delta_G[j]));
                }
                d_k_norm[tid] += sum;
            }
            __syncthreads();

            // Add contribution from G @ m_norm to d_m_norm
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    sum += G[j * N_STATE + tid] * d_delta_G[j];
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

            // Update dS and dG for next iteration
            // Approximate adjoint: propagate through ODE step
            // dS_prev = dS * (1 - dt) + contributions from gates
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                // Simplified adjoint: scale by factor approximating Jacobian
                float decay_factor = 1.0f - dt * n_steps * (1.0f - s_gate[row]);
                if (decay_factor < 0.1f) decay_factor = 0.1f;  // Clamp for stability
                dS[i] = dS[i] * decay_factor + (-(d_delta_S[row] + d_delta_G[row])) * k_norm[col];
            }
            __syncthreads();

            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                float decay_factor = 1.0f - dt * n_steps * (1.0f - g_gate[row]);
                if (decay_factor < 0.1f) decay_factor = 0.1f;
                dG[i] = dG[i] * decay_factor + d_delta_G[row] * m_norm[col];
            }
            __syncthreads();
        }
    }
}

// ============================================================================
// FP32 Kernels
// ============================================================================

template<int N_STATE>
__global__ void E84NeuralODEForwardKernel_FP32(
    int T,
    int B,
    int n_steps,
    float dt,
    const float* __restrict__ kvqm_all,
    float* __restrict__ S,
    float* __restrict__ G,
    float* __restrict__ output,
    float* __restrict__ S_checkpoints,
    float* __restrict__ G_checkpoints,
    float* __restrict__ Sq_cache,
    int checkpoint_interval
) {
    int b = blockIdx.x;
    if (b >= B) return;

    extern __shared__ float shared_mem[];
    float* S_shared = shared_mem;
    float* G_shared = S_shared + N_STATE * N_STATE;
    float* S_k1 = G_shared + N_STATE * N_STATE;
    float* G_k1 = S_k1 + N_STATE * N_STATE;
    float* S_tmp = G_k1 + N_STATE * N_STATE;
    float* G_tmp = S_tmp + N_STATE * N_STATE;
    float* k_shared = G_tmp + N_STATE * N_STATE;
    float* v_shared = k_shared + N_STATE;
    float* q_shared = v_shared + N_STATE;
    float* m_vec_shared = q_shared + N_STATE;
    float* s_gate = m_vec_shared + N_STATE;
    float* g_gate = s_gate + N_STATE;
    float* s_retrieved = g_gate + N_STATE;
    float* g_retrieved = s_retrieved + N_STATE;
    float* delta_S = g_retrieved + N_STATE;
    float* delta_G = delta_S + N_STATE;

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;
    const int STRIDE = 4 * N_STATE;

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

        // RK4 integration
        for (int step = 0; step < n_steps; step++) {
            // k1
            if (tid < N_STATE) {
                float g_row_sum = 0.0f, s_row_sum = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    g_row_sum += G_shared[tid * N_STATE + j] * k_shared[j];
                    s_row_sum += S_shared[tid * N_STATE + j] * m_vec_shared[j];
                }
                s_gate[tid] = 1.0f / (1.0f + expf(-g_row_sum));
                g_gate[tid] = 1.0f / (1.0f + expf(-s_row_sum));
            }
            __syncthreads();

            if (tid < N_STATE) {
                float s_sum = 0.0f, g_sum = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    s_sum += S_shared[tid * N_STATE + j] * k_shared[j];
                    g_sum += G_shared[tid * N_STATE + j] * m_vec_shared[j];
                }
                delta_S[tid] = v_shared[tid] - s_sum;
                delta_G[tid] = delta_S[tid] - g_sum;
            }
            __syncthreads();

            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                S_k1[i] = -S_shared[i] + s_gate[row] * S_shared[i] + delta_S[row] * k_shared[col];
                G_k1[i] = -G_shared[i] + g_gate[row] * G_shared[i] + delta_G[row] * m_vec_shared[col];
                S_tmp[i] = S_shared[i] + 0.5f * dt * S_k1[i];
                G_tmp[i] = G_shared[i] + 0.5f * dt * G_k1[i];
            }
            __syncthreads();

            // k2
            if (tid < N_STATE) {
                float g_row_sum = 0.0f, s_row_sum = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    g_row_sum += G_tmp[tid * N_STATE + j] * k_shared[j];
                    s_row_sum += S_tmp[tid * N_STATE + j] * m_vec_shared[j];
                }
                s_gate[tid] = 1.0f / (1.0f + expf(-g_row_sum));
                g_gate[tid] = 1.0f / (1.0f + expf(-s_row_sum));
            }
            __syncthreads();

            if (tid < N_STATE) {
                float s_sum = 0.0f, g_sum = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    s_sum += S_tmp[tid * N_STATE + j] * k_shared[j];
                    g_sum += G_tmp[tid * N_STATE + j] * m_vec_shared[j];
                }
                delta_S[tid] = v_shared[tid] - s_sum;
                delta_G[tid] = delta_S[tid] - g_sum;
            }
            __syncthreads();

            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                float dS = -S_tmp[i] + s_gate[row] * S_tmp[i] + delta_S[row] * k_shared[col];
                float dG = -G_tmp[i] + g_gate[row] * G_tmp[i] + delta_G[row] * m_vec_shared[col];
                S_k1[i] += 2.0f * dS;
                G_k1[i] += 2.0f * dG;
                S_tmp[i] = S_shared[i] + 0.5f * dt * dS;
                G_tmp[i] = G_shared[i] + 0.5f * dt * dG;
            }
            __syncthreads();

            // k3
            if (tid < N_STATE) {
                float g_row_sum = 0.0f, s_row_sum = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    g_row_sum += G_tmp[tid * N_STATE + j] * k_shared[j];
                    s_row_sum += S_tmp[tid * N_STATE + j] * m_vec_shared[j];
                }
                s_gate[tid] = 1.0f / (1.0f + expf(-g_row_sum));
                g_gate[tid] = 1.0f / (1.0f + expf(-s_row_sum));
            }
            __syncthreads();

            if (tid < N_STATE) {
                float s_sum = 0.0f, g_sum = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    s_sum += S_tmp[tid * N_STATE + j] * k_shared[j];
                    g_sum += G_tmp[tid * N_STATE + j] * m_vec_shared[j];
                }
                delta_S[tid] = v_shared[tid] - s_sum;
                delta_G[tid] = delta_S[tid] - g_sum;
            }
            __syncthreads();

            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                float dS = -S_tmp[i] + s_gate[row] * S_tmp[i] + delta_S[row] * k_shared[col];
                float dG = -G_tmp[i] + g_gate[row] * G_tmp[i] + delta_G[row] * m_vec_shared[col];
                S_k1[i] += 2.0f * dS;
                G_k1[i] += 2.0f * dG;
                S_tmp[i] = S_shared[i] + dt * dS;
                G_tmp[i] = G_shared[i] + dt * dG;
            }
            __syncthreads();

            // k4
            if (tid < N_STATE) {
                float g_row_sum = 0.0f, s_row_sum = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    g_row_sum += G_tmp[tid * N_STATE + j] * k_shared[j];
                    s_row_sum += S_tmp[tid * N_STATE + j] * m_vec_shared[j];
                }
                s_gate[tid] = 1.0f / (1.0f + expf(-g_row_sum));
                g_gate[tid] = 1.0f / (1.0f + expf(-s_row_sum));
            }
            __syncthreads();

            if (tid < N_STATE) {
                float s_sum = 0.0f, g_sum = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    s_sum += S_tmp[tid * N_STATE + j] * k_shared[j];
                    g_sum += G_tmp[tid * N_STATE + j] * m_vec_shared[j];
                }
                delta_S[tid] = v_shared[tid] - s_sum;
                delta_G[tid] = delta_S[tid] - g_sum;
            }
            __syncthreads();

            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                float dS4 = -S_tmp[i] + s_gate[row] * S_tmp[i] + delta_S[row] * k_shared[col];
                float dG4 = -G_tmp[i] + g_gate[row] * G_tmp[i] + delta_G[row] * m_vec_shared[col];
                S_shared[i] += (dt / 6.0f) * (S_k1[i] + dS4);
                G_shared[i] += (dt / 6.0f) * (G_k1[i] + dG4);
            }
            __syncthreads();
        }

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
__global__ void E84NeuralODEBackwardKernel_FP32(
    int T,
    int B,
    int n_steps,
    float dt,
    const float* __restrict__ kvqm_all,
    const float* __restrict__ S_checkpoints,
    const float* __restrict__ G_checkpoints,
    const float* __restrict__ Sq_cache,
    const float* __restrict__ d_output,
    float* __restrict__ d_kvqm_all,
    int checkpoint_interval
) {
    // FP32 backward kernel - same structure as BF16
    // Implementing simplified version for brevity
    int b = blockIdx.x;
    if (b >= B) return;

    extern __shared__ float shared_mem[];
    float* S = shared_mem;
    float* G = S + N_STATE * N_STATE;
    float* dS = G + N_STATE * N_STATE;
    float* dG = dS + N_STATE * N_STATE;
    float* dS_k1 = dG + N_STATE * N_STATE;
    float* dG_k1 = dS_k1 + N_STATE * N_STATE;
    float* S_tmp = dG_k1 + N_STATE * N_STATE;
    float* G_tmp = S_tmp + N_STATE * N_STATE;
    float* k_raw = G_tmp + N_STATE * N_STATE;
    float* v_raw = k_raw + N_STATE;
    float* q_raw = v_raw + N_STATE;
    float* m_vec_raw = q_raw + N_STATE;
    float* k_norm = m_vec_raw + N_STATE;
    float* m_norm = k_norm + N_STATE;
    float* s_gate = m_norm + N_STATE;
    float* g_gate = s_gate + N_STATE;
    float* delta_S = g_gate + N_STATE;
    float* delta_G = delta_S + N_STATE;
    float* d_k_raw = delta_G + N_STATE;
    float* d_v_raw = d_k_raw + N_STATE;
    float* d_q_raw = d_v_raw + N_STATE;
    float* d_m_raw = d_q_raw + N_STATE;
    float* d_Sq_shared = d_m_raw + N_STATE;
    float* d_delta_S = d_Sq_shared + N_STATE;
    float* d_delta_G = d_delta_S + N_STATE;
    float* d_k_norm = d_delta_G + N_STATE;
    float* d_m_norm = d_k_norm + N_STATE;

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;
    const int STRIDE = 4 * N_STATE;

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

                if (tt < t) {
                    // RK4 forward step
                    for (int step = 0; step < n_steps; step++) {
                        if (tid < N_STATE) {
                            float g_row_sum = 0.0f, s_row_sum = 0.0f;
                            for (int j = 0; j < N_STATE; j++) {
                                g_row_sum += G[tid * N_STATE + j] * k_norm[j];
                                s_row_sum += S[tid * N_STATE + j] * m_norm[j];
                            }
                            s_gate[tid] = 1.0f / (1.0f + expf(-g_row_sum));
                            g_gate[tid] = 1.0f / (1.0f + expf(-s_row_sum));
                        }
                        __syncthreads();

                        if (tid < N_STATE) {
                            float s_sum = 0.0f, g_sum = 0.0f;
                            for (int j = 0; j < N_STATE; j++) {
                                s_sum += S[tid * N_STATE + j] * k_norm[j];
                                g_sum += G[tid * N_STATE + j] * m_norm[j];
                            }
                            delta_S[tid] = v_raw[tid] - s_sum;
                            delta_G[tid] = delta_S[tid] - g_sum;
                        }
                        __syncthreads();

                        for (int i = tid; i < n2; i += blockDim.x) {
                            int row = i / N_STATE;
                            int col = i % N_STATE;
                            dS_k1[i] = -S[i] + s_gate[row] * S[i] + delta_S[row] * k_norm[col];
                            dG_k1[i] = -G[i] + g_gate[row] * G[i] + delta_G[row] * m_norm[col];
                            S_tmp[i] = S[i] + 0.5f * dt * dS_k1[i];
                            G_tmp[i] = G[i] + 0.5f * dt * dG_k1[i];
                        }
                        __syncthreads();

                        // k2-k4 (simplified for space)
                        for (int k_iter = 0; k_iter < 3; k_iter++) {
                            if (tid < N_STATE) {
                                float g_row_sum = 0.0f, s_row_sum = 0.0f;
                                for (int j = 0; j < N_STATE; j++) {
                                    g_row_sum += G_tmp[tid * N_STATE + j] * k_norm[j];
                                    s_row_sum += S_tmp[tid * N_STATE + j] * m_norm[j];
                                }
                                s_gate[tid] = 1.0f / (1.0f + expf(-g_row_sum));
                                g_gate[tid] = 1.0f / (1.0f + expf(-s_row_sum));
                            }
                            __syncthreads();

                            if (tid < N_STATE) {
                                float s_sum = 0.0f, g_sum = 0.0f;
                                for (int j = 0; j < N_STATE; j++) {
                                    s_sum += S_tmp[tid * N_STATE + j] * k_norm[j];
                                    g_sum += G_tmp[tid * N_STATE + j] * m_norm[j];
                                }
                                delta_S[tid] = v_raw[tid] - s_sum;
                                delta_G[tid] = delta_S[tid] - g_sum;
                            }
                            __syncthreads();

                            float mult = (k_iter < 2) ? 2.0f : 1.0f;
                            float dt_mult = (k_iter == 0) ? 0.5f : ((k_iter == 1) ? 0.5f : 1.0f);
                            for (int i = tid; i < n2; i += blockDim.x) {
                                int row = i / N_STATE;
                                int col = i % N_STATE;
                                float dSk = -S_tmp[i] + s_gate[row] * S_tmp[i] + delta_S[row] * k_norm[col];
                                float dGk = -G_tmp[i] + g_gate[row] * G_tmp[i] + delta_G[row] * m_norm[col];
                                if (k_iter < 2) {
                                    dS_k1[i] += mult * dSk;
                                    dG_k1[i] += mult * dGk;
                                    S_tmp[i] = S[i] + dt_mult * dt * dSk;
                                    G_tmp[i] = G[i] + dt_mult * dt * dGk;
                                } else {
                                    S[i] += (dt / 6.0f) * (dS_k1[i] + dSk);
                                    G[i] += (dt / 6.0f) * (dG_k1[i] + dGk);
                                }
                            }
                            __syncthreads();
                        }
                    }
                }
            }

            // Backward for step t
            if (tid < N_STATE) {
                float d_out = d_output[t * B * N_STATE + b * N_STATE + tid];
                float Sq_t = Sq_cache[t * B * N_STATE + b * N_STATE + tid];
                float sigmoid_Sq = 1.0f / (1.0f + expf(-Sq_t));
                float silu_Sq = Sq_t * sigmoid_Sq;
                d_Sq_shared[tid] = d_out * (silu_Sq + Sq_t * sigmoid_Sq * (1.0f - sigmoid_Sq));
            }
            __syncthreads();

            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    sum += S[j * N_STATE + tid] * d_Sq_shared[j];
                }
                d_q_raw[tid] = sum;
            }
            __syncthreads();

            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                dS[i] += d_Sq_shared[row] * q_raw[col];
            }
            __syncthreads();

            if (tid < N_STATE) {
                float g_row_sum = 0.0f, s_row_sum = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    g_row_sum += G[tid * N_STATE + j] * k_norm[j];
                    s_row_sum += S[tid * N_STATE + j] * m_norm[j];
                }
                s_gate[tid] = 1.0f / (1.0f + expf(-g_row_sum));
                g_gate[tid] = 1.0f / (1.0f + expf(-s_row_sum));
            }
            __syncthreads();

            if (tid < N_STATE) {
                float s_sum = 0.0f, g_sum = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    s_sum += S[tid * N_STATE + j] * k_norm[j];
                    g_sum += G[tid * N_STATE + j] * m_norm[j];
                }
                delta_S[tid] = v_raw[tid] - s_sum;
                delta_G[tid] = delta_S[tid] - g_sum;
            }
            __syncthreads();

            if (tid < N_STATE) {
                float d_delta_S_local = 0.0f;
                float d_k_norm_local = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    d_delta_S_local += dS[tid * N_STATE + j] * k_norm[j];
                    d_k_norm_local += dS[j * N_STATE + tid] * delta_S[j];
                }
                d_delta_S[tid] = d_delta_S_local;
                d_k_norm[tid] = d_k_norm_local;
            }
            __syncthreads();

            if (tid < N_STATE) {
                float d_delta_G_local = 0.0f;
                float d_m_norm_local = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    d_delta_G_local += dG[tid * N_STATE + j] * m_norm[j];
                    d_m_norm_local += dG[j * N_STATE + tid] * delta_G[j];
                }
                d_delta_G[tid] = d_delta_G_local;
                d_m_norm[tid] = d_m_norm_local;
            }
            __syncthreads();

            if (tid < N_STATE) {
                d_v_raw[tid] = d_delta_S[tid] + d_delta_G[tid];
            }
            __syncthreads();

            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    sum += S[j * N_STATE + tid] * (-(d_delta_S[j] + d_delta_G[j]));
                }
                d_k_norm[tid] += sum;
            }
            __syncthreads();

            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    sum += G[j * N_STATE + tid] * d_delta_G[j];
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
                float decay_factor = 1.0f - dt * n_steps * (1.0f - s_gate[row]);
                if (decay_factor < 0.1f) decay_factor = 0.1f;
                dS[i] = dS[i] * decay_factor + (-(d_delta_S[row] + d_delta_G[row])) * k_norm[col];
            }
            __syncthreads();

            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                float decay_factor = 1.0f - dt * n_steps * (1.0f - g_gate[row]);
                if (decay_factor < 0.1f) decay_factor = 0.1f;
                dG[i] = dG[i] * decay_factor + d_delta_G[row] * m_norm[col];
            }
            __syncthreads();
        }
    }
}

// ============================================================================
// Dispatcher functions
// ============================================================================

void dispatch_e84_neural_ode_forward_bf16(
    int T, int B, int n_state, int n_steps, float dt,
    const __nv_bfloat16* kvqm_all,
    __nv_bfloat16* S, __nv_bfloat16* G, __nv_bfloat16* output,
    __nv_bfloat16* S_checkpoints, __nv_bfloat16* G_checkpoints,
    __nv_bfloat16* Sq_cache,
    int checkpoint_interval, cudaStream_t stream
) {
    // Shared memory size: 6 * n_state^2 + 10 * n_state floats for RK4
    int smem_size = (6 * n_state * n_state + 10 * n_state) * sizeof(float);
    int block_size = 256;

    switch (n_state) {
        case 32:
            E84NeuralODEForwardKernel_BF16<32><<<B, block_size, smem_size, stream>>>(
                T, B, n_steps, dt, kvqm_all, S, G, output,
                S_checkpoints, G_checkpoints, Sq_cache, checkpoint_interval);
            break;
        case 64:
            E84NeuralODEForwardKernel_BF16<64><<<B, block_size, smem_size, stream>>>(
                T, B, n_steps, dt, kvqm_all, S, G, output,
                S_checkpoints, G_checkpoints, Sq_cache, checkpoint_interval);
            break;
        default:
            // For other sizes, fall back to n_state=32 template with runtime check
            if (n_state <= 32) {
                E84NeuralODEForwardKernel_BF16<32><<<B, block_size, smem_size, stream>>>(
                    T, B, n_steps, dt, kvqm_all, S, G, output,
                    S_checkpoints, G_checkpoints, Sq_cache, checkpoint_interval);
            } else if (n_state <= 64) {
                E84NeuralODEForwardKernel_BF16<64><<<B, block_size, smem_size, stream>>>(
                    T, B, n_steps, dt, kvqm_all, S, G, output,
                    S_checkpoints, G_checkpoints, Sq_cache, checkpoint_interval);
            }
            break;
    }
}

void dispatch_e84_neural_ode_backward_bf16(
    int T, int B, int n_state, int n_steps, float dt,
    const __nv_bfloat16* kvqm_all,
    const __nv_bfloat16* S_checkpoints, const __nv_bfloat16* G_checkpoints,
    const __nv_bfloat16* Sq_cache, const __nv_bfloat16* d_output,
    __nv_bfloat16* d_kvqm_all,
    int checkpoint_interval, cudaStream_t stream
) {
    // Shared memory: 8 * n_state^2 + 26 * n_state floats
    int smem_size = (8 * n_state * n_state + 26 * n_state) * sizeof(float);
    int block_size = 256;

    switch (n_state) {
        case 32:
            E84NeuralODEBackwardKernel_BF16<32><<<B, block_size, smem_size, stream>>>(
                T, B, n_steps, dt, kvqm_all, S_checkpoints, G_checkpoints,
                Sq_cache, d_output, d_kvqm_all, checkpoint_interval);
            break;
        case 64:
            E84NeuralODEBackwardKernel_BF16<64><<<B, block_size, smem_size, stream>>>(
                T, B, n_steps, dt, kvqm_all, S_checkpoints, G_checkpoints,
                Sq_cache, d_output, d_kvqm_all, checkpoint_interval);
            break;
        default:
            if (n_state <= 32) {
                E84NeuralODEBackwardKernel_BF16<32><<<B, block_size, smem_size, stream>>>(
                    T, B, n_steps, dt, kvqm_all, S_checkpoints, G_checkpoints,
                    Sq_cache, d_output, d_kvqm_all, checkpoint_interval);
            } else if (n_state <= 64) {
                E84NeuralODEBackwardKernel_BF16<64><<<B, block_size, smem_size, stream>>>(
                    T, B, n_steps, dt, kvqm_all, S_checkpoints, G_checkpoints,
                    Sq_cache, d_output, d_kvqm_all, checkpoint_interval);
            }
            break;
    }
}

void dispatch_e84_neural_ode_forward_fp32(
    int T, int B, int n_state, int n_steps, float dt,
    const float* kvqm_all,
    float* S, float* G, float* output,
    float* S_checkpoints, float* G_checkpoints,
    float* Sq_cache,
    int checkpoint_interval, cudaStream_t stream
) {
    int smem_size = (6 * n_state * n_state + 10 * n_state) * sizeof(float);
    int block_size = 256;

    switch (n_state) {
        case 32:
            E84NeuralODEForwardKernel_FP32<32><<<B, block_size, smem_size, stream>>>(
                T, B, n_steps, dt, kvqm_all, S, G, output,
                S_checkpoints, G_checkpoints, Sq_cache, checkpoint_interval);
            break;
        case 64:
            E84NeuralODEForwardKernel_FP32<64><<<B, block_size, smem_size, stream>>>(
                T, B, n_steps, dt, kvqm_all, S, G, output,
                S_checkpoints, G_checkpoints, Sq_cache, checkpoint_interval);
            break;
        default:
            if (n_state <= 32) {
                E84NeuralODEForwardKernel_FP32<32><<<B, block_size, smem_size, stream>>>(
                    T, B, n_steps, dt, kvqm_all, S, G, output,
                    S_checkpoints, G_checkpoints, Sq_cache, checkpoint_interval);
            } else if (n_state <= 64) {
                E84NeuralODEForwardKernel_FP32<64><<<B, block_size, smem_size, stream>>>(
                    T, B, n_steps, dt, kvqm_all, S, G, output,
                    S_checkpoints, G_checkpoints, Sq_cache, checkpoint_interval);
            }
            break;
    }
}

void dispatch_e84_neural_ode_backward_fp32(
    int T, int B, int n_state, int n_steps, float dt,
    const float* kvqm_all,
    const float* S_checkpoints, const float* G_checkpoints,
    const float* Sq_cache, const float* d_output,
    float* d_kvqm_all,
    int checkpoint_interval, cudaStream_t stream
) {
    int smem_size = (8 * n_state * n_state + 26 * n_state) * sizeof(float);
    int block_size = 256;

    switch (n_state) {
        case 32:
            E84NeuralODEBackwardKernel_FP32<32><<<B, block_size, smem_size, stream>>>(
                T, B, n_steps, dt, kvqm_all, S_checkpoints, G_checkpoints,
                Sq_cache, d_output, d_kvqm_all, checkpoint_interval);
            break;
        case 64:
            E84NeuralODEBackwardKernel_FP32<64><<<B, block_size, smem_size, stream>>>(
                T, B, n_steps, dt, kvqm_all, S_checkpoints, G_checkpoints,
                Sq_cache, d_output, d_kvqm_all, checkpoint_interval);
            break;
        default:
            if (n_state <= 32) {
                E84NeuralODEBackwardKernel_FP32<32><<<B, block_size, smem_size, stream>>>(
                    T, B, n_steps, dt, kvqm_all, S_checkpoints, G_checkpoints,
                    Sq_cache, d_output, d_kvqm_all, checkpoint_interval);
            } else if (n_state <= 64) {
                E84NeuralODEBackwardKernel_FP32<64><<<B, block_size, smem_size, stream>>>(
                    T, B, n_steps, dt, kvqm_all, S_checkpoints, G_checkpoints,
                    Sq_cache, d_output, d_kvqm_all, checkpoint_interval);
            }
            break;
    }
}

// ============================================================================
// Host-side wrapper classes
// ============================================================================

template<typename T>
E84NeuralODEForward<T>::E84NeuralODEForward(
    bool training,
    int batch_size,
    int n_state,
    int dim,
    int n_steps,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : training_(training),
      batch_size_(batch_size),
      n_state_(n_state),
      dim_(dim),
      n_steps_(n_steps),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void E84NeuralODEForward<T>::Run(
    int steps,
    const T* W_kvqm,
    const T* x,
    T* S,
    T* G,
    T* output,
    T* kvqm_cache,
    T* S_checkpoints,
    T* G_checkpoints,
    T* Sq_cache
) {
    const int checkpoint_interval = E84_CHECKPOINT_INTERVAL;
    float dt = 1.0f / n_steps_;

    // Compute kvqm = W_kvqm @ x for all timesteps
    // kvqm_cache: [4*n_state, T*B] in column-major
    const int M = 4 * n_state_;
    const int N = steps * batch_size_;
    const int K = dim_;

    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasGemmEx(blas_handle_,
                     CUBLAS_OP_T, CUBLAS_OP_N,
                     M, N, K,
                     &alpha,
                     W_kvqm, CUDA_R_16BF, K,
                     x, CUDA_R_16BF, K,
                     &beta,
                     kvqm_cache, CUDA_R_16BF, M,
                     CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT);

        dispatch_e84_neural_ode_forward_bf16(
            steps, batch_size_, n_state_, n_steps_, dt,
            kvqm_cache, S, G, output,
            S_checkpoints, G_checkpoints, Sq_cache,
            checkpoint_interval, stream_);
    } else {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasSgemm(blas_handle_,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    M, N, K,
                    &alpha,
                    reinterpret_cast<const float*>(W_kvqm), K,
                    reinterpret_cast<const float*>(x), K,
                    &beta,
                    reinterpret_cast<float*>(kvqm_cache), M);

        dispatch_e84_neural_ode_forward_fp32(
            steps, batch_size_, n_state_, n_steps_, dt,
            reinterpret_cast<const float*>(kvqm_cache),
            reinterpret_cast<float*>(S), reinterpret_cast<float*>(G),
            reinterpret_cast<float*>(output),
            reinterpret_cast<float*>(S_checkpoints), reinterpret_cast<float*>(G_checkpoints),
            reinterpret_cast<float*>(Sq_cache),
            checkpoint_interval, stream_);
    }
}

template<typename T>
E84NeuralODEBackward<T>::E84NeuralODEBackward(
    int batch_size,
    int n_state,
    int dim,
    int n_steps,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      n_state_(n_state),
      dim_(dim),
      n_steps_(n_steps),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void E84NeuralODEBackward<T>::Run(
    int steps,
    const T* W_kvqm,
    const T* x,
    const T* kvqm_cache,
    const T* S_checkpoints,
    const T* G_checkpoints,
    const T* Sq_cache,
    const T* d_output,
    T* d_x,
    T* d_W_kvqm,
    T* d_kvqm_cache
) {
    const int checkpoint_interval = E84_CHECKPOINT_INTERVAL;
    float dt = 1.0f / n_steps_;

    // Backward through kernel
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        dispatch_e84_neural_ode_backward_bf16(
            steps, batch_size_, n_state_, n_steps_, dt,
            kvqm_cache, S_checkpoints, G_checkpoints,
            Sq_cache, d_output, d_kvqm_cache,
            checkpoint_interval, stream_);
    } else {
        dispatch_e84_neural_ode_backward_fp32(
            steps, batch_size_, n_state_, n_steps_, dt,
            reinterpret_cast<const float*>(kvqm_cache),
            reinterpret_cast<const float*>(S_checkpoints),
            reinterpret_cast<const float*>(G_checkpoints),
            reinterpret_cast<const float*>(Sq_cache),
            reinterpret_cast<const float*>(d_output),
            reinterpret_cast<float*>(d_kvqm_cache),
            checkpoint_interval, stream_);
    }

    // Backward through projections
    // d_x = d_kvqm @ W_kvqm
    // d_W_kvqm = d_kvqm @ x.T
    const int M = dim_;
    const int N = steps * batch_size_;
    const int K = 4 * n_state_;

    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        const float alpha = 1.0f;
        const float beta = 0.0f;

        // d_x = W_kvqm.T @ d_kvqm
        cublasGemmEx(blas_handle_,
                     CUBLAS_OP_N, CUBLAS_OP_N,
                     M, N, K,
                     &alpha,
                     W_kvqm, CUDA_R_16BF, M,
                     d_kvqm_cache, CUDA_R_16BF, K,
                     &beta,
                     d_x, CUDA_R_16BF, M,
                     CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT);

        // d_W_kvqm = d_kvqm @ x.T
        cublasGemmEx(blas_handle_,
                     CUBLAS_OP_N, CUBLAS_OP_T,
                     K, M, N,
                     &alpha,
                     d_kvqm_cache, CUDA_R_16BF, K,
                     x, CUDA_R_16BF, M,
                     &beta,
                     d_W_kvqm, CUDA_R_16BF, K,
                     CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT);
    } else {
        const float alpha = 1.0f;
        const float beta = 0.0f;

        cublasSgemm(blas_handle_,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    M, N, K,
                    &alpha,
                    reinterpret_cast<const float*>(W_kvqm), M,
                    reinterpret_cast<const float*>(d_kvqm_cache), K,
                    &beta,
                    reinterpret_cast<float*>(d_x), M);

        cublasSgemm(blas_handle_,
                    CUBLAS_OP_N, CUBLAS_OP_T,
                    K, M, N,
                    &alpha,
                    reinterpret_cast<const float*>(d_kvqm_cache), K,
                    reinterpret_cast<const float*>(x), M,
                    &beta,
                    reinterpret_cast<float*>(d_W_kvqm), K);
    }
}

// Explicit instantiations
template struct E84NeuralODEForward<__nv_bfloat16>;
template struct E84NeuralODEForward<float>;
template struct E84NeuralODEBackward<__nv_bfloat16>;
template struct E84NeuralODEBackward<float>;

}  // namespace elman
