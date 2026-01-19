/**
 * E83 Circular K-Tower Memory System CUDA Kernel
 *
 * K matrices M_0, M_1, ..., M_{K-1}, each n_state x n_state.
 * Each matrix is gated by the NEXT one (modulo K) in a circular pattern:
 *   - M_0 is gated by M_1
 *   - M_1 is gated by M_2
 *   - ...
 *   - M_{K-1} is gated by M_0
 *
 * Key insight: No "top" level - circular dependency creates a fully symmetric
 * system where every matrix is both controller and controlled.
 *
 * Architecture (for K=3):
 *   # Input projections
 *   k_0, v_0 = W_kv[0:2n] @ x
 *   k_1, v_1 = W_kv[2n:4n] @ x
 *   k_2, v_2 = W_kv[4n:6n] @ x
 *   q = W_q @ x
 *
 *   # Circular update for each matrix
 *   for i in range(K):
 *     gater = M[(i+1) % K]  # Next matrix gates this one
 *     row_gate = sigmoid(gater @ k_norm[i] + B[i])
 *     col_gate = sigmoid(gater.T @ k_norm[i] + B[i])
 *     delta = v[i] - M[i] @ k_norm[i]
 *     M[i] = row_gate * M[i] * col_gate + outer(delta, k_norm[i])
 *
 *   # Output from M_0
 *   Sq = M_0 @ q
 *   output = Sq * silu(Sq)
 *
 * Default: K=3 matrices.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include "hasty/elman_ladder.h"

#define E83_CHECKPOINT_INTERVAL 16
#define E83_MAX_K 8  // Maximum supported K value

namespace elman {

// Utility kernel for converting float accumulators to bfloat16
__global__ void ConvertFloatToBF16Kernel_E83(
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
// E83 Forward Kernel - Circular K-Tower System (K=3 default)
// ============================================================================

template<int N_STATE, int K>
__global__ void E83CircularForwardKernel_BF16(
    int T,
    int B,
    const __nv_bfloat16* __restrict__ kv_all,       // [K * 2 * N_STATE, T*B] column-major
    const __nv_bfloat16* __restrict__ q_all,        // [N_STATE, T*B] column-major
    const __nv_bfloat16* __restrict__ B_gates,      // [K, N_STATE] gate biases
    __nv_bfloat16* __restrict__ M_states,           // [K, B, N_STATE, N_STATE]
    __nv_bfloat16* __restrict__ output,             // [T, B, N_STATE]
    __nv_bfloat16* __restrict__ M_checkpoints,      // [num_checkpoints, K, B, N_STATE, N_STATE]
    __nv_bfloat16* __restrict__ Sq_cache,           // [T, B, N_STATE]
    __nv_bfloat16* __restrict__ row_gate_cache,     // [K, T, B, N_STATE]
    __nv_bfloat16* __restrict__ col_gate_cache,     // [K, T, B, N_STATE]
    int checkpoint_interval
) {
    int b = blockIdx.x;
    if (b >= B) return;

    // Shared memory layout - need space for K matrices plus vectors
    // K matrices: K * N_STATE * N_STATE
    // Vectors per level: k, v (K sets) + q + row_gate, col_gate (K sets) + B_gates (K sets) + retrieved (K)
    extern __shared__ float shared_mem[];
    float* M_shared[K];  // Pointers to K matrices in shared memory
    float* temp_ptr = shared_mem;
    for (int i = 0; i < K; i++) {
        M_shared[i] = temp_ptr;
        temp_ptr += N_STATE * N_STATE;
    }

    // Vector storage
    float* k_shared[K];
    float* v_shared[K];
    for (int i = 0; i < K; i++) {
        k_shared[i] = temp_ptr;
        temp_ptr += N_STATE;
        v_shared[i] = temp_ptr;
        temp_ptr += N_STATE;
    }
    float* q_shared = temp_ptr; temp_ptr += N_STATE;

    float* row_gate[K];
    float* col_gate[K];
    for (int i = 0; i < K; i++) {
        row_gate[i] = temp_ptr;
        temp_ptr += N_STATE;
        col_gate[i] = temp_ptr;
        temp_ptr += N_STATE;
    }

    float* B_gate_shared[K];
    for (int i = 0; i < K; i++) {
        B_gate_shared[i] = temp_ptr;
        temp_ptr += N_STATE;
    }

    float* retrieved[K];
    for (int i = 0; i < K; i++) {
        retrieved[i] = temp_ptr;
        temp_ptr += N_STATE;
    }

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;
    const int KV_STRIDE = K * 2 * N_STATE;

    // Load gate biases for all K levels
    for (int i = 0; i < K; i++) {
        if (tid < N_STATE) {
            B_gate_shared[i][tid] = __bfloat162float(B_gates[i * N_STATE + tid]);
        }
    }
    __syncthreads();

    // Load initial states for all K matrices
    for (int i = 0; i < K; i++) {
        for (int j = tid; j < n2; j += blockDim.x) {
            M_shared[i][j] = __bfloat162float(M_states[i * B * n2 + b * n2 + j]);
        }
    }
    __syncthreads();

    // Save initial checkpoints (index 0)
    for (int i = 0; i < K; i++) {
        for (int j = tid; j < n2; j += blockDim.x) {
            M_checkpoints[i * B * n2 + b * n2 + j] = __float2bfloat16(M_shared[i][j]);
        }
    }
    __syncthreads();

    // Process each timestep
    for (int t = 0; t < T; t++) {
        int kv_col_idx = (t * B + b) * KV_STRIDE;
        int q_col_idx = (t * B + b) * N_STATE;

        // Load k, v for all K levels and q for this timestep
        for (int i = 0; i < K; i++) {
            if (tid < N_STATE) {
                int offset = i * 2 * N_STATE;
                k_shared[i][tid] = __bfloat162float(kv_all[kv_col_idx + offset + tid]);
                v_shared[i][tid] = __bfloat162float(kv_all[kv_col_idx + offset + N_STATE + tid]);
            }
        }
        if (tid < N_STATE) {
            q_shared[tid] = __bfloat162float(q_all[q_col_idx + tid]);
        }
        __syncthreads();

        // Normalize all k vectors
        __shared__ float k_norm_vals[K];
        if (tid == 0) {
            for (int i = 0; i < K; i++) {
                float k_sum = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    k_sum += k_shared[i][j] * k_shared[i][j];
                }
                k_norm_vals[i] = sqrtf(k_sum) + 1e-6f;
            }
        }
        __syncthreads();

        for (int i = 0; i < K; i++) {
            if (tid < N_STATE) {
                k_shared[i][tid] /= k_norm_vals[i];
            }
        }
        __syncthreads();

        // Compute gates and retrieved values for all K levels
        // Gate computation: gater = M[(i+1) % K], row_gate = sigmoid(gater @ k + B)
        for (int i = 0; i < K; i++) {
            int gater_idx = (i + 1) % K;
            float* gater = M_shared[gater_idx];

            if (tid < N_STATE) {
                float row_sum = 0.0f, col_sum = 0.0f;
                #pragma unroll 8
                for (int j = 0; j < N_STATE; j++) {
                    row_sum += gater[tid * N_STATE + j] * k_shared[i][j];  // gater @ k
                    col_sum += gater[j * N_STATE + tid] * k_shared[i][j];  // gater.T @ k
                }
                row_gate[i][tid] = 1.0f / (1.0f + expf(-(row_sum + B_gate_shared[i][tid])));
                col_gate[i][tid] = 1.0f / (1.0f + expf(-(col_sum + B_gate_shared[i][tid])));

                // Cache gates for backward
                row_gate_cache[i * T * B * N_STATE + t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(row_gate[i][tid]);
                col_gate_cache[i * T * B * N_STATE + t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(col_gate[i][tid]);
            }
        }
        __syncthreads();

        // Compute retrieved values: retrieved[i] = M[i] @ k_norm[i]
        for (int i = 0; i < K; i++) {
            if (tid < N_STATE) {
                float sum = 0.0f;
                #pragma unroll 8
                for (int j = 0; j < N_STATE; j++) {
                    sum += M_shared[i][tid * N_STATE + j] * k_shared[i][j];
                }
                retrieved[i][tid] = sum;
            }
        }
        __syncthreads();

        // Update all K matrices: M[i] = row_gate * M[i] * col_gate + outer(delta, k)
        for (int i = 0; i < K; i++) {
            for (int j = tid; j < n2; j += blockDim.x) {
                int row = j / N_STATE;
                int col = j % N_STATE;
                float delta_row = v_shared[i][row] - retrieved[i][row];
                float update = row_gate[i][row] * M_shared[i][j] * col_gate[i][col] + delta_row * k_shared[i][col];
                M_shared[i][j] = update;
            }
        }
        __syncthreads();

        // Save checkpoints if at boundary
        if ((t + 1) % checkpoint_interval == 0) {
            int cp_idx = (t + 1) / checkpoint_interval;
            for (int i = 0; i < K; i++) {
                for (int j = tid; j < n2; j += blockDim.x) {
                    M_checkpoints[cp_idx * K * B * n2 + i * B * n2 + b * n2 + j] = __float2bfloat16(M_shared[i][j]);
                }
            }
        }
        __syncthreads();

        // Output: Sq = M_0 @ q, then self-gate
        if (tid < N_STATE) {
            float Sq = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < N_STATE; j++) {
                Sq += M_shared[0][tid * N_STATE + j] * q_shared[j];
            }
            Sq_cache[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(Sq);
            float sig = 1.0f / (1.0f + expf(-Sq));
            output[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(Sq * Sq * sig);
        }
        __syncthreads();
    }

    // Write final states for all K matrices
    for (int i = 0; i < K; i++) {
        for (int j = tid; j < n2; j += blockDim.x) {
            M_states[i * B * n2 + b * n2 + j] = __float2bfloat16(M_shared[i][j]);
        }
    }
}

// ============================================================================
// E83 Backward Kernel - Circular K-Tower System
// ============================================================================

template<int N_STATE, int K>
__global__ void E83CircularBackwardKernel_BF16(
    int T,
    int B,
    const __nv_bfloat16* __restrict__ kv_all,
    const __nv_bfloat16* __restrict__ q_all,
    const __nv_bfloat16* __restrict__ B_gates,
    const __nv_bfloat16* __restrict__ row_gate_cache,
    const __nv_bfloat16* __restrict__ col_gate_cache,
    const __nv_bfloat16* __restrict__ M_checkpoints,
    const __nv_bfloat16* __restrict__ Sq_cache,
    const __nv_bfloat16* __restrict__ d_output,
    __nv_bfloat16* __restrict__ d_kv_all,
    __nv_bfloat16* __restrict__ d_q_all,
    float* __restrict__ d_B_gates_accum,
    int checkpoint_interval
) {
    int b = blockIdx.x;
    if (b >= B) return;

    extern __shared__ float shared_mem[];

    // Layout for backward - need K matrices + K dM matrices + vectors
    float* M_shared[K];
    float* dM_shared[K];
    float* temp_ptr = shared_mem;

    for (int i = 0; i < K; i++) {
        M_shared[i] = temp_ptr;
        temp_ptr += N_STATE * N_STATE;
    }
    for (int i = 0; i < K; i++) {
        dM_shared[i] = temp_ptr;
        temp_ptr += N_STATE * N_STATE;
    }

    // Vector storage
    float* k_raw[K];
    float* v_raw[K];
    float* k_norm[K];
    for (int i = 0; i < K; i++) {
        k_raw[i] = temp_ptr; temp_ptr += N_STATE;
        v_raw[i] = temp_ptr; temp_ptr += N_STATE;
        k_norm[i] = temp_ptr; temp_ptr += N_STATE;
    }
    float* q_raw = temp_ptr; temp_ptr += N_STATE;

    float* row_gate[K];
    float* col_gate[K];
    for (int i = 0; i < K; i++) {
        row_gate[i] = temp_ptr; temp_ptr += N_STATE;
        col_gate[i] = temp_ptr; temp_ptr += N_STATE;
    }

    float* B_gate_shared[K];
    for (int i = 0; i < K; i++) {
        B_gate_shared[i] = temp_ptr;
        temp_ptr += N_STATE;
    }

    float* retrieved[K];
    float* delta[K];
    for (int i = 0; i < K; i++) {
        retrieved[i] = temp_ptr; temp_ptr += N_STATE;
        delta[i] = temp_ptr; temp_ptr += N_STATE;
    }

    // Gradient vectors
    float* d_k_raw[K];
    float* d_v_raw[K];
    float* d_k_norm[K];
    for (int i = 0; i < K; i++) {
        d_k_raw[i] = temp_ptr; temp_ptr += N_STATE;
        d_v_raw[i] = temp_ptr; temp_ptr += N_STATE;
        d_k_norm[i] = temp_ptr; temp_ptr += N_STATE;
    }
    float* d_q_raw = temp_ptr; temp_ptr += N_STATE;
    float* d_Sq_shared = temp_ptr; temp_ptr += N_STATE;

    float* d_row_gate[K];
    float* d_col_gate[K];
    float* d_delta[K];
    for (int i = 0; i < K; i++) {
        d_row_gate[i] = temp_ptr; temp_ptr += N_STATE;
        d_col_gate[i] = temp_ptr; temp_ptr += N_STATE;
        d_delta[i] = temp_ptr; temp_ptr += N_STATE;
    }

    float* d_B_gate_local[K];
    for (int i = 0; i < K; i++) {
        d_B_gate_local[i] = temp_ptr;
        temp_ptr += N_STATE;
    }

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;
    const int KV_STRIDE = K * 2 * N_STATE;

    // Load gate biases
    for (int i = 0; i < K; i++) {
        if (tid < N_STATE) {
            B_gate_shared[i][tid] = __bfloat162float(B_gates[i * N_STATE + tid]);
            d_B_gate_local[i][tid] = 0.0f;
        }
    }
    __syncthreads();

    // Initialize gradient accumulators
    for (int i = 0; i < K; i++) {
        for (int j = tid; j < n2; j += blockDim.x) {
            dM_shared[i][j] = 0.0f;
        }
    }
    __syncthreads();

    int num_segments = (T + checkpoint_interval - 1) / checkpoint_interval;

    for (int seg = num_segments - 1; seg >= 0; seg--) {
        int t_start = seg * checkpoint_interval;
        int t_end = min(t_start + checkpoint_interval, T);

        for (int t = t_end - 1; t >= t_start; t--) {
            // Reload checkpoint for all K matrices
            for (int i = 0; i < K; i++) {
                for (int j = tid; j < n2; j += blockDim.x) {
                    M_shared[i][j] = __bfloat162float(M_checkpoints[seg * K * B * n2 + i * B * n2 + b * n2 + j]);
                }
            }
            __syncthreads();

            // Recompute forward to step t
            __shared__ float k_norm_vals[K];
            for (int tt = t_start; tt <= t; tt++) {
                int kv_col_idx = (tt * B + b) * KV_STRIDE;
                int q_col_idx = (tt * B + b) * N_STATE;

                // Load vectors
                for (int i = 0; i < K; i++) {
                    if (tid < N_STATE) {
                        int offset = i * 2 * N_STATE;
                        k_raw[i][tid] = __bfloat162float(kv_all[kv_col_idx + offset + tid]);
                        v_raw[i][tid] = __bfloat162float(kv_all[kv_col_idx + offset + N_STATE + tid]);
                        row_gate[i][tid] = __bfloat162float(row_gate_cache[i * T * B * N_STATE + tt * B * N_STATE + b * N_STATE + tid]);
                        col_gate[i][tid] = __bfloat162float(col_gate_cache[i * T * B * N_STATE + tt * B * N_STATE + b * N_STATE + tid]);
                    }
                }
                if (tid < N_STATE) {
                    q_raw[tid] = __bfloat162float(q_all[q_col_idx + tid]);
                }
                __syncthreads();

                // Normalize k vectors
                if (tid == 0) {
                    for (int i = 0; i < K; i++) {
                        float k_sum = 0.0f;
                        for (int j = 0; j < N_STATE; j++) {
                            k_sum += k_raw[i][j] * k_raw[i][j];
                        }
                        k_norm_vals[i] = sqrtf(k_sum) + 1e-6f;
                    }
                }
                __syncthreads();

                for (int i = 0; i < K; i++) {
                    if (tid < N_STATE) {
                        k_norm[i][tid] = k_raw[i][tid] / k_norm_vals[i];
                    }
                }
                __syncthreads();

                // Compute retrieved and delta BEFORE update
                for (int i = 0; i < K; i++) {
                    if (tid < N_STATE) {
                        float sum = 0.0f;
                        for (int j = 0; j < N_STATE; j++) {
                            sum += M_shared[i][tid * N_STATE + j] * k_norm[i][j];
                        }
                        retrieved[i][tid] = sum;
                        delta[i][tid] = v_raw[i][tid] - sum;
                    }
                }
                __syncthreads();

                // Update matrices if not at target step
                if (tt < t) {
                    for (int i = 0; i < K; i++) {
                        for (int j = tid; j < n2; j += blockDim.x) {
                            int row = j / N_STATE;
                            int col = j % N_STATE;
                            M_shared[i][j] = row_gate[i][row] * M_shared[i][j] * col_gate[i][col] + delta[i][row] * k_norm[i][col];
                        }
                    }
                    __syncthreads();
                }
            }

            // === BACKWARD PASS FOR STEP t ===

            // Backward through output: Sq = M_0 @ q, output = Sq * silu(Sq)
            if (tid < N_STATE) {
                float d_out = __bfloat162float(d_output[t * B * N_STATE + b * N_STATE + tid]);
                float Sq = __bfloat162float(Sq_cache[t * B * N_STATE + b * N_STATE + tid]);
                float sig = 1.0f / (1.0f + expf(-Sq));
                float d_Sq = d_out * (2.0f * Sq * sig + Sq * Sq * sig * (1.0f - sig));
                d_Sq_shared[tid] = d_Sq;
            }
            __syncthreads();

            // dM_0 += outer(d_Sq, q)
            for (int j = tid; j < n2; j += blockDim.x) {
                int row = j / N_STATE;
                int col = j % N_STATE;
                dM_shared[0][j] += d_Sq_shared[row] * q_raw[col];
            }
            __syncthreads();

            // d_q = M_0_t^T @ d_Sq (need to compute M_0 after update)
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    float M_0_ti = row_gate[0][i] * M_shared[0][i * N_STATE + tid] * col_gate[0][tid] + delta[0][i] * k_norm[0][tid];
                    sum += M_0_ti * d_Sq_shared[i];
                }
                d_q_raw[tid] = sum;
            }
            __syncthreads();

            // Backward through each matrix update (in reverse order to handle circular dependencies)
            for (int i = K - 1; i >= 0; i--) {
                // M_new = row_gate * M * col_gate + outer(delta, k_norm)
                // dM_old = dM_new * row_gate * col_gate
                // d_delta = dM_new @ k_norm
                // d_k_norm = dM_new.T @ delta + contributions from gate computation
                // d_row_gate = sum_j(dM[i,j] * M[i,j] * col_gate[j])
                // d_col_gate = sum_i(dM[i,j] * M[i,j] * row_gate[i])

                if (tid < N_STATE) {
                    float d_delta_local = 0.0f;
                    float d_k_norm_local = 0.0f;
                    float d_row_local = 0.0f;
                    float d_col_local = 0.0f;

                    for (int j = 0; j < N_STATE; j++) {
                        float dM_ij = dM_shared[i][tid * N_STATE + j];
                        d_delta_local += dM_ij * k_norm[i][j];
                        d_row_local += dM_ij * M_shared[i][tid * N_STATE + j] * col_gate[i][j];

                        float dM_ji = dM_shared[i][j * N_STATE + tid];
                        d_k_norm_local += dM_ji * delta[i][j];
                        d_col_local += dM_ji * M_shared[i][j * N_STATE + tid] * row_gate[i][j];
                    }

                    d_delta[i][tid] = d_delta_local;
                    d_k_norm[i][tid] = d_k_norm_local;
                    d_row_gate[i][tid] = d_row_local;
                    d_col_gate[i][tid] = d_col_local;
                }
                __syncthreads();

                // Gate gradients flow to gater matrix (next in circular order)
                // row_gate = sigmoid(gater @ k + B), where gater = M[(i+1) % K]
                // d_gater contribution: outer(d_row_gate * gate_deriv, k_norm) + outer(k_norm, d_col_gate * gate_deriv)
                int gater_idx = (i + 1) % K;

                // Bias gradients
                if (tid < N_STATE) {
                    float gate_deriv_row = row_gate[i][tid] * (1.0f - row_gate[i][tid]);
                    float gate_deriv_col = col_gate[i][tid] * (1.0f - col_gate[i][tid]);
                    d_B_gate_local[i][tid] += d_row_gate[i][tid] * gate_deriv_row + d_col_gate[i][tid] * gate_deriv_col;

                    // d_k_norm contribution from gate computation
                    float d_gater_k_row = d_row_gate[i][tid] * gate_deriv_row;
                    float d_gater_k_col = d_col_gate[i][tid] * gate_deriv_col;

                    // d_k_norm += gater.T @ d_gater_k_row + gater @ d_gater_k_col
                    float gater_contrib = 0.0f;
                    for (int j = 0; j < N_STATE; j++) {
                        gater_contrib += M_shared[gater_idx][j * N_STATE + tid] * d_gater_k_row;  // gater.T contribution
                        gater_contrib += M_shared[gater_idx][tid * N_STATE + j] * d_gater_k_col;  // gater contribution (transposed indexing)
                    }
                    // Actually, this needs to be reformulated - let's use proper chain rule
                }
                __syncthreads();

                // Add dM contribution to gater from gate computation
                // d_gater += outer(d_row_gate * gate_deriv, k_norm) for row gate
                // d_gater += outer(k_norm, d_col_gate * gate_deriv) for col gate (transposed)
                for (int j = tid; j < n2; j += blockDim.x) {
                    int row = j / N_STATE;
                    int col = j % N_STATE;
                    float gate_deriv_row = row_gate[i][row] * (1.0f - row_gate[i][row]);
                    float gate_deriv_col = col_gate[i][col] * (1.0f - col_gate[i][col]);

                    // Row gate: d/d(gater[row,col]) of sigmoid(gater @ k + B)[row] = gate_deriv * k[col]
                    // d_gater[row,col] += d_row_gate[row] * gate_deriv_row * k[col]
                    dM_shared[gater_idx][j] += d_row_gate[i][row] * gate_deriv_row * k_norm[i][col];

                    // Col gate: d/d(gater[col,row]) of sigmoid(gater.T @ k + B)[col] = gate_deriv * k[row]
                    // d_gater[col,row] += d_col_gate[col] * gate_deriv_col * k[row]
                    // This is: d_gater[row,col] += d_col_gate[row] * gate_deriv_at_row * k[col]
                    // Wait, col_gate[tid] comes from gater[j, tid] * k[j], so:
                    // d_gater[j,tid] += d_col_gate[tid] * gate_deriv_col * k[j]
                    // In matrix form at [row,col]: d_gater += k[row] * d_col_gate[col] * gate_deriv_col
                    float gate_deriv_col_at_col = col_gate[i][col] * (1.0f - col_gate[i][col]);
                    dM_shared[gater_idx][j] += k_norm[i][row] * d_col_gate[i][col] * gate_deriv_col_at_col;
                }
                __syncthreads();

                // d_v = d_delta (from delta = v - retrieved)
                if (tid < N_STATE) {
                    d_v_raw[i][tid] = d_delta[i][tid];
                }

                // d_k_norm contribution from retrieved: retrieved = M @ k_norm
                // d_k_norm += M^T @ (-d_delta)
                if (tid < N_STATE) {
                    float sum = 0.0f;
                    for (int j = 0; j < N_STATE; j++) {
                        sum += M_shared[i][j * N_STATE + tid] * (-d_delta[i][j]);
                    }
                    d_k_norm[i][tid] += sum;
                }
                __syncthreads();

                // Convert d_k_norm to d_k_raw
                {
                    __shared__ float k_dot_dk;
                    if (tid == 0) {
                        k_dot_dk = 0.0f;
                        for (int j = 0; j < N_STATE; j++) {
                            k_dot_dk += k_raw[i][j] * d_k_norm[i][j];
                        }
                    }
                    __syncthreads();
                    if (tid < N_STATE) {
                        float norm = k_norm_vals[i];
                        float norm3 = norm * norm * norm;
                        d_k_raw[i][tid] = d_k_norm[i][tid] / norm - k_raw[i][tid] * k_dot_dk / norm3;
                    }
                    __syncthreads();
                }

                // Update dM for next iteration (propagate through decay)
                for (int j = tid; j < n2; j += blockDim.x) {
                    int row = j / N_STATE;
                    int col = j % N_STATE;
                    float d_pre = dM_shared[i][j];
                    dM_shared[i][j] = d_pre * row_gate[i][row] * col_gate[i][col] + (-d_delta[i][row]) * k_norm[i][col];
                }
                __syncthreads();
            }

            // Write gradients for this timestep
            int kv_col_idx_t = (t * B + b) * KV_STRIDE;
            int q_col_idx_t = (t * B + b) * N_STATE;

            for (int i = 0; i < K; i++) {
                if (tid < N_STATE) {
                    int offset = i * 2 * N_STATE;
                    d_kv_all[kv_col_idx_t + offset + tid] = __float2bfloat16(d_k_raw[i][tid]);
                    d_kv_all[kv_col_idx_t + offset + N_STATE + tid] = __float2bfloat16(d_v_raw[i][tid]);
                }
            }
            if (tid < N_STATE) {
                d_q_all[q_col_idx_t + tid] = __float2bfloat16(d_q_raw[tid]);
            }
            __syncthreads();
        }
    }

    // Accumulate bias gradients
    for (int i = 0; i < K; i++) {
        if (tid < N_STATE) {
            atomicAdd(&d_B_gates_accum[i * N_STATE + tid], d_B_gate_local[i][tid]);
        }
    }
}

// ============================================================================
// FP32 Forward Kernel
// ============================================================================

template<int N_STATE, int K>
__global__ void E83CircularForwardKernel_FP32(
    int T,
    int B,
    const float* __restrict__ kv_all,
    const float* __restrict__ q_all,
    const float* __restrict__ B_gates,
    float* __restrict__ M_states,
    float* __restrict__ output,
    float* __restrict__ M_checkpoints,
    float* __restrict__ Sq_cache,
    float* __restrict__ row_gate_cache,
    float* __restrict__ col_gate_cache,
    int checkpoint_interval
) {
    int b = blockIdx.x;
    if (b >= B) return;

    extern __shared__ float shared_mem[];
    float* M_shared[K];
    float* temp_ptr = shared_mem;
    for (int i = 0; i < K; i++) {
        M_shared[i] = temp_ptr;
        temp_ptr += N_STATE * N_STATE;
    }

    float* k_shared[K];
    float* v_shared[K];
    for (int i = 0; i < K; i++) {
        k_shared[i] = temp_ptr; temp_ptr += N_STATE;
        v_shared[i] = temp_ptr; temp_ptr += N_STATE;
    }
    float* q_shared = temp_ptr; temp_ptr += N_STATE;

    float* row_gate[K];
    float* col_gate[K];
    for (int i = 0; i < K; i++) {
        row_gate[i] = temp_ptr; temp_ptr += N_STATE;
        col_gate[i] = temp_ptr; temp_ptr += N_STATE;
    }

    float* B_gate_shared[K];
    for (int i = 0; i < K; i++) {
        B_gate_shared[i] = temp_ptr;
        temp_ptr += N_STATE;
    }

    float* retrieved[K];
    for (int i = 0; i < K; i++) {
        retrieved[i] = temp_ptr;
        temp_ptr += N_STATE;
    }

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;
    const int KV_STRIDE = K * 2 * N_STATE;

    // Load gate biases
    for (int i = 0; i < K; i++) {
        if (tid < N_STATE) {
            B_gate_shared[i][tid] = B_gates[i * N_STATE + tid];
        }
    }
    __syncthreads();

    // Load initial states
    for (int i = 0; i < K; i++) {
        for (int j = tid; j < n2; j += blockDim.x) {
            M_shared[i][j] = M_states[i * B * n2 + b * n2 + j];
        }
    }
    __syncthreads();

    // Save initial checkpoints
    for (int i = 0; i < K; i++) {
        for (int j = tid; j < n2; j += blockDim.x) {
            M_checkpoints[i * B * n2 + b * n2 + j] = M_shared[i][j];
        }
    }
    __syncthreads();

    for (int t = 0; t < T; t++) {
        int kv_col_idx = (t * B + b) * KV_STRIDE;
        int q_col_idx = (t * B + b) * N_STATE;

        // Load vectors
        for (int i = 0; i < K; i++) {
            if (tid < N_STATE) {
                int offset = i * 2 * N_STATE;
                k_shared[i][tid] = kv_all[kv_col_idx + offset + tid];
                v_shared[i][tid] = kv_all[kv_col_idx + offset + N_STATE + tid];
            }
        }
        if (tid < N_STATE) {
            q_shared[tid] = q_all[q_col_idx + tid];
        }
        __syncthreads();

        // Normalize k vectors
        __shared__ float k_norm_vals[K];
        if (tid == 0) {
            for (int i = 0; i < K; i++) {
                float k_sum = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    k_sum += k_shared[i][j] * k_shared[i][j];
                }
                k_norm_vals[i] = sqrtf(k_sum) + 1e-6f;
            }
        }
        __syncthreads();
        for (int i = 0; i < K; i++) {
            if (tid < N_STATE) {
                k_shared[i][tid] /= k_norm_vals[i];
            }
        }
        __syncthreads();

        // Compute gates
        for (int i = 0; i < K; i++) {
            int gater_idx = (i + 1) % K;
            float* gater = M_shared[gater_idx];

            if (tid < N_STATE) {
                float row_sum = 0.0f, col_sum = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    row_sum += gater[tid * N_STATE + j] * k_shared[i][j];
                    col_sum += gater[j * N_STATE + tid] * k_shared[i][j];
                }
                row_gate[i][tid] = 1.0f / (1.0f + expf(-(row_sum + B_gate_shared[i][tid])));
                col_gate[i][tid] = 1.0f / (1.0f + expf(-(col_sum + B_gate_shared[i][tid])));
                row_gate_cache[i * T * B * N_STATE + t * B * N_STATE + b * N_STATE + tid] = row_gate[i][tid];
                col_gate_cache[i * T * B * N_STATE + t * B * N_STATE + b * N_STATE + tid] = col_gate[i][tid];
            }
        }
        __syncthreads();

        // Compute retrieved
        for (int i = 0; i < K; i++) {
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    sum += M_shared[i][tid * N_STATE + j] * k_shared[i][j];
                }
                retrieved[i][tid] = sum;
            }
        }
        __syncthreads();

        // Update matrices
        for (int i = 0; i < K; i++) {
            for (int j = tid; j < n2; j += blockDim.x) {
                int row = j / N_STATE;
                int col = j % N_STATE;
                float delta_row = v_shared[i][row] - retrieved[i][row];
                M_shared[i][j] = row_gate[i][row] * M_shared[i][j] * col_gate[i][col] + delta_row * k_shared[i][col];
            }
        }
        __syncthreads();

        // Save checkpoints
        if ((t + 1) % checkpoint_interval == 0) {
            int cp_idx = (t + 1) / checkpoint_interval;
            for (int i = 0; i < K; i++) {
                for (int j = tid; j < n2; j += blockDim.x) {
                    M_checkpoints[cp_idx * K * B * n2 + i * B * n2 + b * n2 + j] = M_shared[i][j];
                }
            }
        }
        __syncthreads();

        // Output
        if (tid < N_STATE) {
            float Sq = 0.0f;
            for (int j = 0; j < N_STATE; j++) {
                Sq += M_shared[0][tid * N_STATE + j] * q_shared[j];
            }
            Sq_cache[t * B * N_STATE + b * N_STATE + tid] = Sq;
            float sig = 1.0f / (1.0f + expf(-Sq));
            output[t * B * N_STATE + b * N_STATE + tid] = Sq * Sq * sig;
        }
        __syncthreads();
    }

    // Write final states
    for (int i = 0; i < K; i++) {
        for (int j = tid; j < n2; j += blockDim.x) {
            M_states[i * B * n2 + b * n2 + j] = M_shared[i][j];
        }
    }
}

// ============================================================================
// FP32 Backward Kernel
// ============================================================================

template<int N_STATE, int K>
__global__ void E83CircularBackwardKernel_FP32(
    int T,
    int B,
    const float* __restrict__ kv_all,
    const float* __restrict__ q_all,
    const float* __restrict__ B_gates,
    const float* __restrict__ row_gate_cache,
    const float* __restrict__ col_gate_cache,
    const float* __restrict__ M_checkpoints,
    const float* __restrict__ Sq_cache,
    const float* __restrict__ d_output,
    float* __restrict__ d_kv_all,
    float* __restrict__ d_q_all,
    float* __restrict__ d_B_gates_accum,
    int checkpoint_interval
) {
    int b = blockIdx.x;
    if (b >= B) return;

    extern __shared__ float shared_mem[];
    float* M_shared[K];
    float* dM_shared[K];
    float* temp_ptr = shared_mem;

    for (int i = 0; i < K; i++) {
        M_shared[i] = temp_ptr;
        temp_ptr += N_STATE * N_STATE;
    }
    for (int i = 0; i < K; i++) {
        dM_shared[i] = temp_ptr;
        temp_ptr += N_STATE * N_STATE;
    }

    float* k_raw[K];
    float* v_raw[K];
    float* k_norm[K];
    for (int i = 0; i < K; i++) {
        k_raw[i] = temp_ptr; temp_ptr += N_STATE;
        v_raw[i] = temp_ptr; temp_ptr += N_STATE;
        k_norm[i] = temp_ptr; temp_ptr += N_STATE;
    }
    float* q_raw = temp_ptr; temp_ptr += N_STATE;

    float* row_gate[K];
    float* col_gate[K];
    for (int i = 0; i < K; i++) {
        row_gate[i] = temp_ptr; temp_ptr += N_STATE;
        col_gate[i] = temp_ptr; temp_ptr += N_STATE;
    }

    float* B_gate_shared[K];
    for (int i = 0; i < K; i++) {
        B_gate_shared[i] = temp_ptr;
        temp_ptr += N_STATE;
    }

    float* retrieved[K];
    float* delta[K];
    for (int i = 0; i < K; i++) {
        retrieved[i] = temp_ptr; temp_ptr += N_STATE;
        delta[i] = temp_ptr; temp_ptr += N_STATE;
    }

    float* d_k_raw[K];
    float* d_v_raw[K];
    float* d_k_norm[K];
    for (int i = 0; i < K; i++) {
        d_k_raw[i] = temp_ptr; temp_ptr += N_STATE;
        d_v_raw[i] = temp_ptr; temp_ptr += N_STATE;
        d_k_norm[i] = temp_ptr; temp_ptr += N_STATE;
    }
    float* d_q_raw = temp_ptr; temp_ptr += N_STATE;
    float* d_Sq_shared = temp_ptr; temp_ptr += N_STATE;

    float* d_row_gate[K];
    float* d_col_gate[K];
    float* d_delta[K];
    for (int i = 0; i < K; i++) {
        d_row_gate[i] = temp_ptr; temp_ptr += N_STATE;
        d_col_gate[i] = temp_ptr; temp_ptr += N_STATE;
        d_delta[i] = temp_ptr; temp_ptr += N_STATE;
    }

    float* d_B_gate_local[K];
    for (int i = 0; i < K; i++) {
        d_B_gate_local[i] = temp_ptr;
        temp_ptr += N_STATE;
    }

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;
    const int KV_STRIDE = K * 2 * N_STATE;

    // Load gate biases and init local accumulators
    for (int i = 0; i < K; i++) {
        if (tid < N_STATE) {
            B_gate_shared[i][tid] = B_gates[i * N_STATE + tid];
            d_B_gate_local[i][tid] = 0.0f;
        }
    }
    __syncthreads();

    // Initialize dM
    for (int i = 0; i < K; i++) {
        for (int j = tid; j < n2; j += blockDim.x) {
            dM_shared[i][j] = 0.0f;
        }
    }
    __syncthreads();

    int num_segments = (T + checkpoint_interval - 1) / checkpoint_interval;

    for (int seg = num_segments - 1; seg >= 0; seg--) {
        int t_start = seg * checkpoint_interval;
        int t_end = min(t_start + checkpoint_interval, T);

        for (int t = t_end - 1; t >= t_start; t--) {
            // Reload checkpoints
            for (int i = 0; i < K; i++) {
                for (int j = tid; j < n2; j += blockDim.x) {
                    M_shared[i][j] = M_checkpoints[seg * K * B * n2 + i * B * n2 + b * n2 + j];
                }
            }
            __syncthreads();

            // Recompute forward
            __shared__ float k_norm_vals[K];
            for (int tt = t_start; tt <= t; tt++) {
                int kv_col_idx = (tt * B + b) * KV_STRIDE;
                int q_col_idx = (tt * B + b) * N_STATE;

                for (int i = 0; i < K; i++) {
                    if (tid < N_STATE) {
                        int offset = i * 2 * N_STATE;
                        k_raw[i][tid] = kv_all[kv_col_idx + offset + tid];
                        v_raw[i][tid] = kv_all[kv_col_idx + offset + N_STATE + tid];
                        row_gate[i][tid] = row_gate_cache[i * T * B * N_STATE + tt * B * N_STATE + b * N_STATE + tid];
                        col_gate[i][tid] = col_gate_cache[i * T * B * N_STATE + tt * B * N_STATE + b * N_STATE + tid];
                    }
                }
                if (tid < N_STATE) {
                    q_raw[tid] = q_all[q_col_idx + tid];
                }
                __syncthreads();

                if (tid == 0) {
                    for (int i = 0; i < K; i++) {
                        float k_sum = 0.0f;
                        for (int j = 0; j < N_STATE; j++) {
                            k_sum += k_raw[i][j] * k_raw[i][j];
                        }
                        k_norm_vals[i] = sqrtf(k_sum) + 1e-6f;
                    }
                }
                __syncthreads();
                for (int i = 0; i < K; i++) {
                    if (tid < N_STATE) {
                        k_norm[i][tid] = k_raw[i][tid] / k_norm_vals[i];
                    }
                }
                __syncthreads();

                for (int i = 0; i < K; i++) {
                    if (tid < N_STATE) {
                        float sum = 0.0f;
                        for (int j = 0; j < N_STATE; j++) {
                            sum += M_shared[i][tid * N_STATE + j] * k_norm[i][j];
                        }
                        retrieved[i][tid] = sum;
                        delta[i][tid] = v_raw[i][tid] - sum;
                    }
                }
                __syncthreads();

                if (tt < t) {
                    for (int i = 0; i < K; i++) {
                        for (int j = tid; j < n2; j += blockDim.x) {
                            int row = j / N_STATE;
                            int col = j % N_STATE;
                            M_shared[i][j] = row_gate[i][row] * M_shared[i][j] * col_gate[i][col] + delta[i][row] * k_norm[i][col];
                        }
                    }
                    __syncthreads();
                }
            }

            // Backward through output
            if (tid < N_STATE) {
                float d_out = d_output[t * B * N_STATE + b * N_STATE + tid];
                float Sq = Sq_cache[t * B * N_STATE + b * N_STATE + tid];
                float sig = 1.0f / (1.0f + expf(-Sq));
                float d_Sq = d_out * (2.0f * Sq * sig + Sq * Sq * sig * (1.0f - sig));
                d_Sq_shared[tid] = d_Sq;
            }
            __syncthreads();

            for (int j = tid; j < n2; j += blockDim.x) {
                int row = j / N_STATE;
                int col = j % N_STATE;
                dM_shared[0][j] += d_Sq_shared[row] * q_raw[col];
            }
            __syncthreads();

            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    float M_0_ti = row_gate[0][i] * M_shared[0][i * N_STATE + tid] * col_gate[0][tid] + delta[0][i] * k_norm[0][tid];
                    sum += M_0_ti * d_Sq_shared[i];
                }
                d_q_raw[tid] = sum;
            }
            __syncthreads();

            // Backward through each matrix
            for (int i = K - 1; i >= 0; i--) {
                if (tid < N_STATE) {
                    float d_delta_local = 0.0f;
                    float d_k_norm_local = 0.0f;
                    float d_row_local = 0.0f;
                    float d_col_local = 0.0f;

                    for (int j = 0; j < N_STATE; j++) {
                        float dM_ij = dM_shared[i][tid * N_STATE + j];
                        d_delta_local += dM_ij * k_norm[i][j];
                        d_row_local += dM_ij * M_shared[i][tid * N_STATE + j] * col_gate[i][j];

                        float dM_ji = dM_shared[i][j * N_STATE + tid];
                        d_k_norm_local += dM_ji * delta[i][j];
                        d_col_local += dM_ji * M_shared[i][j * N_STATE + tid] * row_gate[i][j];
                    }

                    d_delta[i][tid] = d_delta_local;
                    d_k_norm[i][tid] = d_k_norm_local;
                    d_row_gate[i][tid] = d_row_local;
                    d_col_gate[i][tid] = d_col_local;
                }
                __syncthreads();

                int gater_idx = (i + 1) % K;

                if (tid < N_STATE) {
                    float gate_deriv_row = row_gate[i][tid] * (1.0f - row_gate[i][tid]);
                    float gate_deriv_col = col_gate[i][tid] * (1.0f - col_gate[i][tid]);
                    d_B_gate_local[i][tid] += d_row_gate[i][tid] * gate_deriv_row + d_col_gate[i][tid] * gate_deriv_col;
                }
                __syncthreads();

                for (int j = tid; j < n2; j += blockDim.x) {
                    int row = j / N_STATE;
                    int col = j % N_STATE;
                    float gate_deriv_row = row_gate[i][row] * (1.0f - row_gate[i][row]);
                    float gate_deriv_col_at_col = col_gate[i][col] * (1.0f - col_gate[i][col]);
                    dM_shared[gater_idx][j] += d_row_gate[i][row] * gate_deriv_row * k_norm[i][col];
                    dM_shared[gater_idx][j] += k_norm[i][row] * d_col_gate[i][col] * gate_deriv_col_at_col;
                }
                __syncthreads();

                if (tid < N_STATE) {
                    d_v_raw[i][tid] = d_delta[i][tid];
                }

                if (tid < N_STATE) {
                    float sum = 0.0f;
                    for (int j = 0; j < N_STATE; j++) {
                        sum += M_shared[i][j * N_STATE + tid] * (-d_delta[i][j]);
                    }
                    d_k_norm[i][tid] += sum;
                }
                __syncthreads();

                {
                    __shared__ float k_dot_dk;
                    if (tid == 0) {
                        k_dot_dk = 0.0f;
                        for (int j = 0; j < N_STATE; j++) {
                            k_dot_dk += k_raw[i][j] * d_k_norm[i][j];
                        }
                    }
                    __syncthreads();
                    if (tid < N_STATE) {
                        float norm = k_norm_vals[i];
                        float norm3 = norm * norm * norm;
                        d_k_raw[i][tid] = d_k_norm[i][tid] / norm - k_raw[i][tid] * k_dot_dk / norm3;
                    }
                    __syncthreads();
                }

                for (int j = tid; j < n2; j += blockDim.x) {
                    int row = j / N_STATE;
                    int col = j % N_STATE;
                    float d_pre = dM_shared[i][j];
                    dM_shared[i][j] = d_pre * row_gate[i][row] * col_gate[i][col] + (-d_delta[i][row]) * k_norm[i][col];
                }
                __syncthreads();
            }

            // Write gradients
            int kv_col_idx_t = (t * B + b) * KV_STRIDE;
            int q_col_idx_t = (t * B + b) * N_STATE;

            for (int i = 0; i < K; i++) {
                if (tid < N_STATE) {
                    int offset = i * 2 * N_STATE;
                    d_kv_all[kv_col_idx_t + offset + tid] = d_k_raw[i][tid];
                    d_kv_all[kv_col_idx_t + offset + N_STATE + tid] = d_v_raw[i][tid];
                }
            }
            if (tid < N_STATE) {
                d_q_all[q_col_idx_t + tid] = d_q_raw[tid];
            }
            __syncthreads();
        }
    }

    // Accumulate bias gradients
    for (int i = 0; i < K; i++) {
        if (tid < N_STATE) {
            atomicAdd(&d_B_gates_accum[i * N_STATE + tid], d_B_gate_local[i][tid]);
        }
    }
}

// ============================================================================
// E83 INPUT-BIAS Forward Kernel - Per-timestep biases
// ============================================================================

template<int N_STATE, int K>
__global__ void E83CircularInputBiasForwardKernel_BF16(
    int T,
    int B,
    const __nv_bfloat16* __restrict__ kv_all,       // [K * 2 * N_STATE, T*B] column-major
    const __nv_bfloat16* __restrict__ q_all,        // [N_STATE, T*B] column-major
    const __nv_bfloat16* __restrict__ b_all,        // [T*B, K*N_STATE] per-timestep biases
    __nv_bfloat16* __restrict__ M_states,           // [K, B, N_STATE, N_STATE]
    __nv_bfloat16* __restrict__ output,             // [T, B, N_STATE]
    __nv_bfloat16* __restrict__ M_checkpoints,      // [num_checkpoints, K, B, N_STATE, N_STATE]
    __nv_bfloat16* __restrict__ Sq_cache,           // [T, B, N_STATE]
    __nv_bfloat16* __restrict__ row_gate_cache,     // [K, T, B, N_STATE]
    __nv_bfloat16* __restrict__ col_gate_cache,     // [K, T, B, N_STATE]
    int checkpoint_interval
) {
    int b = blockIdx.x;
    if (b >= B) return;

    extern __shared__ float shared_mem[];
    float* M_shared[K];
    float* temp_ptr = shared_mem;
    for (int i = 0; i < K; i++) {
        M_shared[i] = temp_ptr;
        temp_ptr += N_STATE * N_STATE;
    }

    float* k_shared[K];
    float* v_shared[K];
    for (int i = 0; i < K; i++) {
        k_shared[i] = temp_ptr;
        temp_ptr += N_STATE;
        v_shared[i] = temp_ptr;
        temp_ptr += N_STATE;
    }
    float* q_shared = temp_ptr; temp_ptr += N_STATE;

    float* row_gate[K];
    float* col_gate[K];
    for (int i = 0; i < K; i++) {
        row_gate[i] = temp_ptr;
        temp_ptr += N_STATE;
        col_gate[i] = temp_ptr;
        temp_ptr += N_STATE;
    }

    float* B_gate_shared[K];  // Per-timestep biases loaded each step
    for (int i = 0; i < K; i++) {
        B_gate_shared[i] = temp_ptr;
        temp_ptr += N_STATE;
    }

    float* retrieved[K];
    for (int i = 0; i < K; i++) {
        retrieved[i] = temp_ptr;
        temp_ptr += N_STATE;
    }

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;
    const int KV_STRIDE = K * 2 * N_STATE;

    // Load initial states for all K matrices
    for (int i = 0; i < K; i++) {
        for (int j = tid; j < n2; j += blockDim.x) {
            M_shared[i][j] = __bfloat162float(M_states[i * B * n2 + b * n2 + j]);
        }
    }
    __syncthreads();

    // Save initial checkpoints (index 0)
    for (int i = 0; i < K; i++) {
        for (int j = tid; j < n2; j += blockDim.x) {
            M_checkpoints[i * B * n2 + b * n2 + j] = __float2bfloat16(M_shared[i][j]);
        }
    }
    __syncthreads();

    // Process each timestep
    for (int t = 0; t < T; t++) {
        int kv_col_idx = (t * B + b) * KV_STRIDE;
        int q_col_idx = (t * B + b) * N_STATE;
        int bias_idx = (t * B + b) * K * N_STATE;  // Per-timestep bias index

        // Load k, v for all K levels, q, and per-timestep biases
        for (int i = 0; i < K; i++) {
            if (tid < N_STATE) {
                int offset = i * 2 * N_STATE;
                k_shared[i][tid] = __bfloat162float(kv_all[kv_col_idx + offset + tid]);
                v_shared[i][tid] = __bfloat162float(kv_all[kv_col_idx + offset + N_STATE + tid]);
                B_gate_shared[i][tid] = __bfloat162float(b_all[bias_idx + i * N_STATE + tid]);
            }
        }
        if (tid < N_STATE) {
            q_shared[tid] = __bfloat162float(q_all[q_col_idx + tid]);
        }
        __syncthreads();

        // Normalize all k vectors
        __shared__ float k_norm_vals[K];
        if (tid == 0) {
            for (int i = 0; i < K; i++) {
                float k_sum = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    k_sum += k_shared[i][j] * k_shared[i][j];
                }
                k_norm_vals[i] = sqrtf(k_sum) + 1e-6f;
            }
        }
        __syncthreads();

        for (int i = 0; i < K; i++) {
            if (tid < N_STATE) {
                k_shared[i][tid] /= k_norm_vals[i];
            }
        }
        __syncthreads();

        // Compute gates and retrieved values for all K levels
        for (int i = 0; i < K; i++) {
            int gater_idx = (i + 1) % K;
            float* gater = M_shared[gater_idx];

            if (tid < N_STATE) {
                float row_sum = 0.0f, col_sum = 0.0f;
                #pragma unroll 8
                for (int j = 0; j < N_STATE; j++) {
                    row_sum += gater[tid * N_STATE + j] * k_shared[i][j];
                    col_sum += gater[j * N_STATE + tid] * k_shared[i][j];
                }
                row_gate[i][tid] = 1.0f / (1.0f + expf(-(row_sum + B_gate_shared[i][tid])));
                col_gate[i][tid] = 1.0f / (1.0f + expf(-(col_sum + B_gate_shared[i][tid])));

                row_gate_cache[i * T * B * N_STATE + t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(row_gate[i][tid]);
                col_gate_cache[i * T * B * N_STATE + t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(col_gate[i][tid]);
            }
        }
        __syncthreads();

        // Compute retrieved values
        for (int i = 0; i < K; i++) {
            if (tid < N_STATE) {
                float sum = 0.0f;
                #pragma unroll 8
                for (int j = 0; j < N_STATE; j++) {
                    sum += M_shared[i][tid * N_STATE + j] * k_shared[i][j];
                }
                retrieved[i][tid] = sum;
            }
        }
        __syncthreads();

        // Update all K matrices
        for (int i = 0; i < K; i++) {
            for (int j = tid; j < n2; j += blockDim.x) {
                int row = j / N_STATE;
                int col = j % N_STATE;
                float delta_row = v_shared[i][row] - retrieved[i][row];
                float update = row_gate[i][row] * M_shared[i][j] * col_gate[i][col] + delta_row * k_shared[i][col];
                M_shared[i][j] = update;
            }
        }
        __syncthreads();

        // Save checkpoints if at boundary
        if ((t + 1) % checkpoint_interval == 0) {
            int cp_idx = (t + 1) / checkpoint_interval;
            for (int i = 0; i < K; i++) {
                for (int j = tid; j < n2; j += blockDim.x) {
                    M_checkpoints[cp_idx * K * B * n2 + i * B * n2 + b * n2 + j] = __float2bfloat16(M_shared[i][j]);
                }
            }
        }
        __syncthreads();

        // Output: Sq = M_0 @ q, then self-gate
        if (tid < N_STATE) {
            float Sq = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < N_STATE; j++) {
                Sq += M_shared[0][tid * N_STATE + j] * q_shared[j];
            }
            Sq_cache[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(Sq);
            float sig = 1.0f / (1.0f + expf(-Sq));
            output[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(Sq * Sq * sig);
        }
        __syncthreads();
    }

    // Write final states
    for (int i = 0; i < K; i++) {
        for (int j = tid; j < n2; j += blockDim.x) {
            M_states[i * B * n2 + b * n2 + j] = __float2bfloat16(M_shared[i][j]);
        }
    }
}

// ============================================================================
// E83 INPUT-BIAS Backward Kernel - Per-timestep biases
// ============================================================================

template<int N_STATE, int K>
__global__ void E83CircularInputBiasBackwardKernel_BF16(
    int T,
    int B,
    const __nv_bfloat16* __restrict__ kv_all,
    const __nv_bfloat16* __restrict__ q_all,
    const __nv_bfloat16* __restrict__ b_all,        // [T*B, K*N_STATE] per-timestep biases
    const __nv_bfloat16* __restrict__ row_gate_cache,
    const __nv_bfloat16* __restrict__ col_gate_cache,
    const __nv_bfloat16* __restrict__ M_checkpoints,
    const __nv_bfloat16* __restrict__ Sq_cache,
    const __nv_bfloat16* __restrict__ d_output,
    __nv_bfloat16* __restrict__ d_kv_all,
    __nv_bfloat16* __restrict__ d_q_all,
    __nv_bfloat16* __restrict__ d_b_all,            // [T*B, K*N_STATE] per-timestep bias gradients
    int checkpoint_interval
) {
    int b = blockIdx.x;
    if (b >= B) return;

    extern __shared__ float shared_mem[];

    float* M_shared[K];
    float* dM_shared[K];
    float* temp_ptr = shared_mem;

    for (int i = 0; i < K; i++) {
        M_shared[i] = temp_ptr;
        temp_ptr += N_STATE * N_STATE;
    }
    for (int i = 0; i < K; i++) {
        dM_shared[i] = temp_ptr;
        temp_ptr += N_STATE * N_STATE;
    }

    float* k_raw[K];
    float* v_raw[K];
    float* k_norm[K];
    for (int i = 0; i < K; i++) {
        k_raw[i] = temp_ptr; temp_ptr += N_STATE;
        v_raw[i] = temp_ptr; temp_ptr += N_STATE;
        k_norm[i] = temp_ptr; temp_ptr += N_STATE;
    }
    float* q_raw = temp_ptr; temp_ptr += N_STATE;

    float* row_gate[K];
    float* col_gate[K];
    for (int i = 0; i < K; i++) {
        row_gate[i] = temp_ptr; temp_ptr += N_STATE;
        col_gate[i] = temp_ptr; temp_ptr += N_STATE;
    }

    float* B_gate_shared[K];  // Per-timestep biases
    for (int i = 0; i < K; i++) {
        B_gate_shared[i] = temp_ptr;
        temp_ptr += N_STATE;
    }

    float* retrieved[K];
    float* delta[K];
    for (int i = 0; i < K; i++) {
        retrieved[i] = temp_ptr; temp_ptr += N_STATE;
        delta[i] = temp_ptr; temp_ptr += N_STATE;
    }

    float* d_k_raw[K];
    float* d_v_raw[K];
    float* d_k_norm[K];
    for (int i = 0; i < K; i++) {
        d_k_raw[i] = temp_ptr; temp_ptr += N_STATE;
        d_v_raw[i] = temp_ptr; temp_ptr += N_STATE;
        d_k_norm[i] = temp_ptr; temp_ptr += N_STATE;
    }
    float* d_q_raw = temp_ptr; temp_ptr += N_STATE;
    float* d_Sq_shared = temp_ptr; temp_ptr += N_STATE;

    float* d_row_gate[K];
    float* d_col_gate[K];
    float* d_delta[K];
    for (int i = 0; i < K; i++) {
        d_row_gate[i] = temp_ptr; temp_ptr += N_STATE;
        d_col_gate[i] = temp_ptr; temp_ptr += N_STATE;
        d_delta[i] = temp_ptr; temp_ptr += N_STATE;
    }

    float* d_B_gate_local[K];  // Per-timestep bias gradient (written per timestep)
    for (int i = 0; i < K; i++) {
        d_B_gate_local[i] = temp_ptr;
        temp_ptr += N_STATE;
    }

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;
    const int KV_STRIDE = K * 2 * N_STATE;

    // Initialize gradient accumulators
    for (int i = 0; i < K; i++) {
        for (int j = tid; j < n2; j += blockDim.x) {
            dM_shared[i][j] = 0.0f;
        }
    }
    __syncthreads();

    int num_segments = (T + checkpoint_interval - 1) / checkpoint_interval;

    for (int seg = num_segments - 1; seg >= 0; seg--) {
        int t_start = seg * checkpoint_interval;
        int t_end = min(t_start + checkpoint_interval, T);

        for (int t = t_end - 1; t >= t_start; t--) {
            // Reload checkpoint for all K matrices
            for (int i = 0; i < K; i++) {
                for (int j = tid; j < n2; j += blockDim.x) {
                    M_shared[i][j] = __bfloat162float(M_checkpoints[seg * K * B * n2 + i * B * n2 + b * n2 + j]);
                }
            }
            __syncthreads();

            // Recompute forward to step t
            __shared__ float k_norm_vals[K];
            for (int tt = t_start; tt <= t; tt++) {
                int kv_col_idx = (tt * B + b) * KV_STRIDE;
                int q_col_idx = (tt * B + b) * N_STATE;
                int bias_idx = (tt * B + b) * K * N_STATE;

                // Load vectors and per-timestep biases
                for (int i = 0; i < K; i++) {
                    if (tid < N_STATE) {
                        int offset = i * 2 * N_STATE;
                        k_raw[i][tid] = __bfloat162float(kv_all[kv_col_idx + offset + tid]);
                        v_raw[i][tid] = __bfloat162float(kv_all[kv_col_idx + offset + N_STATE + tid]);
                        row_gate[i][tid] = __bfloat162float(row_gate_cache[i * T * B * N_STATE + tt * B * N_STATE + b * N_STATE + tid]);
                        col_gate[i][tid] = __bfloat162float(col_gate_cache[i * T * B * N_STATE + tt * B * N_STATE + b * N_STATE + tid]);
                        B_gate_shared[i][tid] = __bfloat162float(b_all[bias_idx + i * N_STATE + tid]);
                    }
                }
                if (tid < N_STATE) {
                    q_raw[tid] = __bfloat162float(q_all[q_col_idx + tid]);
                }
                __syncthreads();

                // Normalize k vectors
                if (tid == 0) {
                    for (int i = 0; i < K; i++) {
                        float k_sum = 0.0f;
                        for (int j = 0; j < N_STATE; j++) {
                            k_sum += k_raw[i][j] * k_raw[i][j];
                        }
                        k_norm_vals[i] = sqrtf(k_sum) + 1e-6f;
                    }
                }
                __syncthreads();

                for (int i = 0; i < K; i++) {
                    if (tid < N_STATE) {
                        k_norm[i][tid] = k_raw[i][tid] / k_norm_vals[i];
                    }
                }
                __syncthreads();

                // Compute retrieved and delta BEFORE update
                for (int i = 0; i < K; i++) {
                    if (tid < N_STATE) {
                        float sum = 0.0f;
                        for (int j = 0; j < N_STATE; j++) {
                            sum += M_shared[i][tid * N_STATE + j] * k_norm[i][j];
                        }
                        retrieved[i][tid] = sum;
                        delta[i][tid] = v_raw[i][tid] - sum;
                    }
                }
                __syncthreads();

                // Update matrices if not at target step
                if (tt < t) {
                    for (int i = 0; i < K; i++) {
                        for (int j = tid; j < n2; j += blockDim.x) {
                            int row = j / N_STATE;
                            int col = j % N_STATE;
                            M_shared[i][j] = row_gate[i][row] * M_shared[i][j] * col_gate[i][col] + delta[i][row] * k_norm[i][col];
                        }
                    }
                    __syncthreads();
                }
            }

            // === BACKWARD PASS FOR STEP t ===

            // Initialize per-timestep bias gradients to zero
            for (int i = 0; i < K; i++) {
                if (tid < N_STATE) {
                    d_B_gate_local[i][tid] = 0.0f;
                }
            }
            __syncthreads();

            // Backward through output
            if (tid < N_STATE) {
                float d_out = __bfloat162float(d_output[t * B * N_STATE + b * N_STATE + tid]);
                float Sq = __bfloat162float(Sq_cache[t * B * N_STATE + b * N_STATE + tid]);
                float sig = 1.0f / (1.0f + expf(-Sq));
                float d_Sq = d_out * (2.0f * Sq * sig + Sq * Sq * sig * (1.0f - sig));
                d_Sq_shared[tid] = d_Sq;
            }
            __syncthreads();

            // dM_0 += outer(d_Sq, q)
            for (int j = tid; j < n2; j += blockDim.x) {
                int row = j / N_STATE;
                int col = j % N_STATE;
                dM_shared[0][j] += d_Sq_shared[row] * q_raw[col];
            }
            __syncthreads();

            // d_q = M_0_t^T @ d_Sq
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    float M_0_ti = row_gate[0][i] * M_shared[0][i * N_STATE + tid] * col_gate[0][tid] + delta[0][i] * k_norm[0][tid];
                    sum += M_0_ti * d_Sq_shared[i];
                }
                d_q_raw[tid] = sum;
            }
            __syncthreads();

            // Backward through each matrix update
            for (int i = K - 1; i >= 0; i--) {
                if (tid < N_STATE) {
                    float d_delta_local = 0.0f;
                    float d_k_norm_local = 0.0f;
                    float d_row_local = 0.0f;
                    float d_col_local = 0.0f;

                    for (int j = 0; j < N_STATE; j++) {
                        float dM_ij = dM_shared[i][tid * N_STATE + j];
                        d_delta_local += dM_ij * k_norm[i][j];
                        d_row_local += dM_ij * M_shared[i][tid * N_STATE + j] * col_gate[i][j];

                        float dM_ji = dM_shared[i][j * N_STATE + tid];
                        d_k_norm_local += dM_ji * delta[i][j];
                        d_col_local += dM_ji * M_shared[i][j * N_STATE + tid] * row_gate[i][j];
                    }

                    d_delta[i][tid] = d_delta_local;
                    d_k_norm[i][tid] = d_k_norm_local;
                    d_row_gate[i][tid] = d_row_local;
                    d_col_gate[i][tid] = d_col_local;
                }
                __syncthreads();

                int gater_idx = (i + 1) % K;

                // Bias gradients - accumulate locally
                if (tid < N_STATE) {
                    float gate_deriv_row = row_gate[i][tid] * (1.0f - row_gate[i][tid]);
                    float gate_deriv_col = col_gate[i][tid] * (1.0f - col_gate[i][tid]);
                    d_B_gate_local[i][tid] += d_row_gate[i][tid] * gate_deriv_row + d_col_gate[i][tid] * gate_deriv_col;
                }
                __syncthreads();

                // Add dM contribution to gater from gate computation
                for (int j = tid; j < n2; j += blockDim.x) {
                    int row = j / N_STATE;
                    int col = j % N_STATE;
                    float gate_deriv_row = row_gate[i][row] * (1.0f - row_gate[i][row]);
                    float gate_deriv_col_at_col = col_gate[i][col] * (1.0f - col_gate[i][col]);

                    dM_shared[gater_idx][j] += d_row_gate[i][row] * gate_deriv_row * k_norm[i][col];
                    dM_shared[gater_idx][j] += k_norm[i][row] * d_col_gate[i][col] * gate_deriv_col_at_col;
                }
                __syncthreads();

                // d_v = d_delta
                if (tid < N_STATE) {
                    d_v_raw[i][tid] = d_delta[i][tid];
                }

                // d_k_norm contribution from retrieved
                if (tid < N_STATE) {
                    float sum = 0.0f;
                    for (int j = 0; j < N_STATE; j++) {
                        sum += M_shared[i][j * N_STATE + tid] * (-d_delta[i][j]);
                    }
                    d_k_norm[i][tid] += sum;
                }
                __syncthreads();

                // Convert d_k_norm to d_k_raw
                {
                    __shared__ float k_dot_dk;
                    if (tid == 0) {
                        k_dot_dk = 0.0f;
                        for (int j = 0; j < N_STATE; j++) {
                            k_dot_dk += k_raw[i][j] * d_k_norm[i][j];
                        }
                    }
                    __syncthreads();
                    if (tid < N_STATE) {
                        float norm = k_norm_vals[i];
                        float norm3 = norm * norm * norm;
                        d_k_raw[i][tid] = d_k_norm[i][tid] / norm - k_raw[i][tid] * k_dot_dk / norm3;
                    }
                    __syncthreads();
                }

                // Update dM for next iteration
                for (int j = tid; j < n2; j += blockDim.x) {
                    int row = j / N_STATE;
                    int col = j % N_STATE;
                    float d_pre = dM_shared[i][j];
                    dM_shared[i][j] = d_pre * row_gate[i][row] * col_gate[i][col] + (-d_delta[i][row]) * k_norm[i][col];
                }
                __syncthreads();
            }

            // Write gradients for this timestep
            int kv_col_idx_t = (t * B + b) * KV_STRIDE;
            int q_col_idx_t = (t * B + b) * N_STATE;
            int bias_idx_t = (t * B + b) * K * N_STATE;

            for (int i = 0; i < K; i++) {
                if (tid < N_STATE) {
                    int offset = i * 2 * N_STATE;
                    d_kv_all[kv_col_idx_t + offset + tid] = __float2bfloat16(d_k_raw[i][tid]);
                    d_kv_all[kv_col_idx_t + offset + N_STATE + tid] = __float2bfloat16(d_v_raw[i][tid]);
                    // Write per-timestep bias gradients
                    d_b_all[bias_idx_t + i * N_STATE + tid] = __float2bfloat16(d_B_gate_local[i][tid]);
                }
            }
            if (tid < N_STATE) {
                d_q_all[q_col_idx_t + tid] = __float2bfloat16(d_q_raw[tid]);
            }
            __syncthreads();
        }
    }
}

// ============================================================================
// Template instantiations
// ============================================================================

#define INSTANTIATE_E83_IB_KERNELS_BF16(N, K_VAL) \
    template __global__ void E83CircularInputBiasForwardKernel_BF16<N, K_VAL>( \
        int, int, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, \
        __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, \
        __nv_bfloat16*, __nv_bfloat16*, int); \
    template __global__ void E83CircularInputBiasBackwardKernel_BF16<N, K_VAL>( \
        int, int, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, \
        const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, \
        const __nv_bfloat16*, const __nv_bfloat16*, \
        __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int);

#define INSTANTIATE_E83_KERNELS_BF16(N, K_VAL) \
    template __global__ void E83CircularForwardKernel_BF16<N, K_VAL>( \
        int, int, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, \
        __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, \
        __nv_bfloat16*, __nv_bfloat16*, int); \
    template __global__ void E83CircularBackwardKernel_BF16<N, K_VAL>( \
        int, int, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, \
        const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, \
        const __nv_bfloat16*, const __nv_bfloat16*, \
        __nv_bfloat16*, __nv_bfloat16*, float*, int);

#define INSTANTIATE_E83_KERNELS_FP32(N, K_VAL) \
    template __global__ void E83CircularForwardKernel_FP32<N, K_VAL>( \
        int, int, const float*, const float*, const float*, \
        float*, float*, float*, float*, float*, float*, int); \
    template __global__ void E83CircularBackwardKernel_FP32<N, K_VAL>( \
        int, int, const float*, const float*, const float*, \
        const float*, const float*, const float*, \
        const float*, const float*, \
        float*, float*, float*, int);

// Fixed-bias kernels: K=2, K=3, K=4
INSTANTIATE_E83_KERNELS_BF16(16, 2)
INSTANTIATE_E83_KERNELS_BF16(24, 2)
INSTANTIATE_E83_KERNELS_BF16(32, 2)
INSTANTIATE_E83_KERNELS_BF16(48, 2)

INSTANTIATE_E83_KERNELS_FP32(16, 2)
INSTANTIATE_E83_KERNELS_FP32(24, 2)
INSTANTIATE_E83_KERNELS_FP32(32, 2)
INSTANTIATE_E83_KERNELS_FP32(48, 2)

INSTANTIATE_E83_KERNELS_BF16(16, 3)
INSTANTIATE_E83_KERNELS_BF16(24, 3)
INSTANTIATE_E83_KERNELS_BF16(32, 3)
INSTANTIATE_E83_KERNELS_BF16(48, 3)

INSTANTIATE_E83_KERNELS_FP32(16, 3)
INSTANTIATE_E83_KERNELS_FP32(24, 3)
INSTANTIATE_E83_KERNELS_FP32(32, 3)
INSTANTIATE_E83_KERNELS_FP32(48, 3)

INSTANTIATE_E83_KERNELS_BF16(16, 4)
INSTANTIATE_E83_KERNELS_BF16(24, 4)
INSTANTIATE_E83_KERNELS_BF16(32, 4)

INSTANTIATE_E83_KERNELS_FP32(16, 4)
INSTANTIATE_E83_KERNELS_FP32(24, 4)
INSTANTIATE_E83_KERNELS_FP32(32, 4)

// Fixed-bias kernels: K=8 (many heads)
INSTANTIATE_E83_KERNELS_BF16(16, 8)
INSTANTIATE_E83_KERNELS_BF16(24, 8)

INSTANTIATE_E83_KERNELS_FP32(16, 8)
INSTANTIATE_E83_KERNELS_FP32(24, 8)

// Input-bias kernels: K=2, K=3, K=4
INSTANTIATE_E83_IB_KERNELS_BF16(16, 2)
INSTANTIATE_E83_IB_KERNELS_BF16(24, 2)
INSTANTIATE_E83_IB_KERNELS_BF16(32, 2)
INSTANTIATE_E83_IB_KERNELS_BF16(48, 2)

INSTANTIATE_E83_IB_KERNELS_BF16(16, 3)
INSTANTIATE_E83_IB_KERNELS_BF16(24, 3)
INSTANTIATE_E83_IB_KERNELS_BF16(32, 3)
INSTANTIATE_E83_IB_KERNELS_BF16(48, 3)

INSTANTIATE_E83_IB_KERNELS_BF16(16, 4)
INSTANTIATE_E83_IB_KERNELS_BF16(24, 4)
INSTANTIATE_E83_IB_KERNELS_BF16(32, 4)

// Input-bias kernels: K=8 (many heads)
INSTANTIATE_E83_IB_KERNELS_BF16(16, 8)
INSTANTIATE_E83_IB_KERNELS_BF16(24, 8)

// ============================================================================
// Dispatcher functions
// ============================================================================

// Calculate shared memory needed for forward: K matrices + vectors
inline int calc_e83_forward_shared_mem(int n_state, int K) {
    // K matrices: K * n^2
    // Vectors: K*2 (k,v) + 1 (q) + K*2 (row/col gate) + K (B_gate) + K (retrieved) = 6K + 1
    return (K * n_state * n_state + (6 * K + 1) * n_state) * sizeof(float);
}

// Calculate shared memory needed for backward: 2K matrices + vectors
inline int calc_e83_backward_shared_mem(int n_state, int K) {
    // 2K matrices: 2K * n^2 (M and dM)
    // Vectors: K*3 (k_raw, v_raw, k_norm) + 1 (q) + K*2 (gates) + K (B_gate) + K*2 (retrieved, delta)
    //        + K*3 (d_k_raw, d_v_raw, d_k_norm) + 1 (d_q) + 1 (d_Sq)
    //        + K*3 (d_row_gate, d_col_gate, d_delta) + K (d_B_gate_local)
    // Total vectors: 3K + 1 + 2K + K + 2K + 3K + 2 + 3K + K = 15K + 3
    return (2 * K * n_state * n_state + (15 * K + 3) * n_state) * sizeof(float);
}

void dispatch_e83_circular_forward(
    int T, int B, int n_state, int K,
    const __nv_bfloat16* kv_all,
    const __nv_bfloat16* q_all,
    const __nv_bfloat16* B_gates,
    __nv_bfloat16* M_states,
    __nv_bfloat16* output,
    __nv_bfloat16* M_checkpoints,
    __nv_bfloat16* Sq_cache,
    __nv_bfloat16* row_gate_cache,
    __nv_bfloat16* col_gate_cache,
    int checkpoint_interval, cudaStream_t stream
) {
    int shared_size = calc_e83_forward_shared_mem(n_state, K);

    #define DISPATCH_E83_FWD_BF16(N, K_VAL) \
        E83CircularForwardKernel_BF16<N, K_VAL><<<B, 256, shared_size, stream>>>( \
            T, B, kv_all, q_all, B_gates, M_states, output, M_checkpoints, \
            Sq_cache, row_gate_cache, col_gate_cache, checkpoint_interval);

    if (K == 2) {
        switch (n_state) {
            case 16: DISPATCH_E83_FWD_BF16(16, 2); break;
            case 24: DISPATCH_E83_FWD_BF16(24, 2); break;
            case 32: DISPATCH_E83_FWD_BF16(32, 2); break;
            case 48: DISPATCH_E83_FWD_BF16(48, 2); break;
            default: fprintf(stderr, "E83: Unsupported n_state=%d for K=2\n", n_state);
        }
    } else if (K == 3) {
        switch (n_state) {
            case 16: DISPATCH_E83_FWD_BF16(16, 3); break;
            case 24: DISPATCH_E83_FWD_BF16(24, 3); break;
            case 32: DISPATCH_E83_FWD_BF16(32, 3); break;
            case 48: DISPATCH_E83_FWD_BF16(48, 3); break;
            default: fprintf(stderr, "E83: Unsupported n_state=%d for K=3\n", n_state);
        }
    } else if (K == 4) {
        switch (n_state) {
            case 16: DISPATCH_E83_FWD_BF16(16, 4); break;
            case 24: DISPATCH_E83_FWD_BF16(24, 4); break;
            case 32: DISPATCH_E83_FWD_BF16(32, 4); break;
            default: fprintf(stderr, "E83: Unsupported n_state=%d for K=4\n", n_state);
        }
    } else if (K == 8) {
        switch (n_state) {
            case 16: DISPATCH_E83_FWD_BF16(16, 8); break;
            case 24: DISPATCH_E83_FWD_BF16(24, 8); break;
            default: fprintf(stderr, "E83: Unsupported n_state=%d for K=8\n", n_state);
        }
    } else {
        fprintf(stderr, "E83: Unsupported K=%d (use 2, 3, 4, or 8)\n", K);
    }

    #undef DISPATCH_E83_FWD_BF16
}

void dispatch_e83_circular_backward(
    int T, int B, int n_state, int K,
    const __nv_bfloat16* kv_all,
    const __nv_bfloat16* q_all,
    const __nv_bfloat16* B_gates,
    const __nv_bfloat16* row_gate_cache,
    const __nv_bfloat16* col_gate_cache,
    const __nv_bfloat16* M_checkpoints,
    const __nv_bfloat16* Sq_cache,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* d_kv_all,
    __nv_bfloat16* d_q_all,
    float* d_B_gates_accum,
    int checkpoint_interval, cudaStream_t stream
) {
    int shared_size = calc_e83_backward_shared_mem(n_state, K);

    #define DISPATCH_E83_BWD_BF16(N, K_VAL) \
        E83CircularBackwardKernel_BF16<N, K_VAL><<<B, 256, shared_size, stream>>>( \
            T, B, kv_all, q_all, B_gates, row_gate_cache, col_gate_cache, \
            M_checkpoints, Sq_cache, d_output, d_kv_all, d_q_all, d_B_gates_accum, \
            checkpoint_interval);

    if (K == 2) {
        switch (n_state) {
            case 16: DISPATCH_E83_BWD_BF16(16, 2); break;
            case 24: DISPATCH_E83_BWD_BF16(24, 2); break;
            case 32: DISPATCH_E83_BWD_BF16(32, 2); break;
            case 48: DISPATCH_E83_BWD_BF16(48, 2); break;
            default: fprintf(stderr, "E83: Unsupported n_state=%d for K=2\n", n_state);
        }
    } else if (K == 3) {
        switch (n_state) {
            case 16: DISPATCH_E83_BWD_BF16(16, 3); break;
            case 24: DISPATCH_E83_BWD_BF16(24, 3); break;
            case 32: DISPATCH_E83_BWD_BF16(32, 3); break;
            case 48: DISPATCH_E83_BWD_BF16(48, 3); break;
            default: fprintf(stderr, "E83: Unsupported n_state=%d for K=3\n", n_state);
        }
    } else if (K == 4) {
        switch (n_state) {
            case 16: DISPATCH_E83_BWD_BF16(16, 4); break;
            case 24: DISPATCH_E83_BWD_BF16(24, 4); break;
            case 32: DISPATCH_E83_BWD_BF16(32, 4); break;
            default: fprintf(stderr, "E83: Unsupported n_state=%d for K=4\n", n_state);
        }
    } else if (K == 8) {
        switch (n_state) {
            case 16: DISPATCH_E83_BWD_BF16(16, 8); break;
            case 24: DISPATCH_E83_BWD_BF16(24, 8); break;
            default: fprintf(stderr, "E83: Unsupported n_state=%d for K=8\n", n_state);
        }
    } else {
        fprintf(stderr, "E83: Unsupported K=%d (use 2, 3, 4, or 8)\n", K);
    }

    #undef DISPATCH_E83_BWD_BF16
}

void dispatch_e83_circular_forward_fp32(
    int T, int B, int n_state, int K,
    const float* kv_all,
    const float* q_all,
    const float* B_gates,
    float* M_states,
    float* output,
    float* M_checkpoints,
    float* Sq_cache,
    float* row_gate_cache,
    float* col_gate_cache,
    int checkpoint_interval, cudaStream_t stream
) {
    int shared_size = calc_e83_forward_shared_mem(n_state, K);

    #define DISPATCH_E83_FWD_FP32(N, K_VAL) \
        E83CircularForwardKernel_FP32<N, K_VAL><<<B, 256, shared_size, stream>>>( \
            T, B, kv_all, q_all, B_gates, M_states, output, M_checkpoints, \
            Sq_cache, row_gate_cache, col_gate_cache, checkpoint_interval);

    if (K == 2) {
        switch (n_state) {
            case 16: DISPATCH_E83_FWD_FP32(16, 2); break;
            case 24: DISPATCH_E83_FWD_FP32(24, 2); break;
            case 32: DISPATCH_E83_FWD_FP32(32, 2); break;
            case 48: DISPATCH_E83_FWD_FP32(48, 2); break;
            default: fprintf(stderr, "E83: Unsupported n_state=%d for K=2\n", n_state);
        }
    } else if (K == 3) {
        switch (n_state) {
            case 16: DISPATCH_E83_FWD_FP32(16, 3); break;
            case 24: DISPATCH_E83_FWD_FP32(24, 3); break;
            case 32: DISPATCH_E83_FWD_FP32(32, 3); break;
            case 48: DISPATCH_E83_FWD_FP32(48, 3); break;
            default: fprintf(stderr, "E83: Unsupported n_state=%d for K=3\n", n_state);
        }
    } else if (K == 4) {
        switch (n_state) {
            case 16: DISPATCH_E83_FWD_FP32(16, 4); break;
            case 24: DISPATCH_E83_FWD_FP32(24, 4); break;
            case 32: DISPATCH_E83_FWD_FP32(32, 4); break;
            default: fprintf(stderr, "E83: Unsupported n_state=%d for K=4\n", n_state);
        }
    } else if (K == 8) {
        switch (n_state) {
            case 16: DISPATCH_E83_FWD_FP32(16, 8); break;
            case 24: DISPATCH_E83_FWD_FP32(24, 8); break;
            default: fprintf(stderr, "E83: Unsupported n_state=%d for K=8\n", n_state);
        }
    } else {
        fprintf(stderr, "E83: Unsupported K=%d (use 2, 3, 4, or 8)\n", K);
    }

    #undef DISPATCH_E83_FWD_FP32
}

void dispatch_e83_circular_backward_fp32(
    int T, int B, int n_state, int K,
    const float* kv_all,
    const float* q_all,
    const float* B_gates,
    const float* row_gate_cache,
    const float* col_gate_cache,
    const float* M_checkpoints,
    const float* Sq_cache,
    const float* d_output,
    float* d_kv_all,
    float* d_q_all,
    float* d_B_gates_accum,
    int checkpoint_interval, cudaStream_t stream
) {
    int shared_size = calc_e83_backward_shared_mem(n_state, K);

    #define DISPATCH_E83_BWD_FP32(N, K_VAL) \
        E83CircularBackwardKernel_FP32<N, K_VAL><<<B, 256, shared_size, stream>>>( \
            T, B, kv_all, q_all, B_gates, row_gate_cache, col_gate_cache, \
            M_checkpoints, Sq_cache, d_output, d_kv_all, d_q_all, d_B_gates_accum, \
            checkpoint_interval);

    if (K == 2) {
        switch (n_state) {
            case 16: DISPATCH_E83_BWD_FP32(16, 2); break;
            case 24: DISPATCH_E83_BWD_FP32(24, 2); break;
            case 32: DISPATCH_E83_BWD_FP32(32, 2); break;
            case 48: DISPATCH_E83_BWD_FP32(48, 2); break;
            default: fprintf(stderr, "E83: Unsupported n_state=%d for K=2\n", n_state);
        }
    } else if (K == 3) {
        switch (n_state) {
            case 16: DISPATCH_E83_BWD_FP32(16, 3); break;
            case 24: DISPATCH_E83_BWD_FP32(24, 3); break;
            case 32: DISPATCH_E83_BWD_FP32(32, 3); break;
            case 48: DISPATCH_E83_BWD_FP32(48, 3); break;
            default: fprintf(stderr, "E83: Unsupported n_state=%d for K=3\n", n_state);
        }
    } else if (K == 4) {
        switch (n_state) {
            case 16: DISPATCH_E83_BWD_FP32(16, 4); break;
            case 24: DISPATCH_E83_BWD_FP32(24, 4); break;
            case 32: DISPATCH_E83_BWD_FP32(32, 4); break;
            default: fprintf(stderr, "E83: Unsupported n_state=%d for K=4\n", n_state);
        }
    } else if (K == 8) {
        switch (n_state) {
            case 16: DISPATCH_E83_BWD_FP32(16, 8); break;
            case 24: DISPATCH_E83_BWD_FP32(24, 8); break;
            default: fprintf(stderr, "E83: Unsupported n_state=%d for K=8\n", n_state);
        }
    } else {
        fprintf(stderr, "E83: Unsupported K=%d (use 2, 3, 4, or 8)\n", K);
    }

    #undef DISPATCH_E83_BWD_FP32
}

// ============================================================================
// Input-Bias Dispatcher functions
// ============================================================================

void dispatch_e83_circular_input_bias_forward(
    int T, int B, int n_state, int K,
    const __nv_bfloat16* kv_all,
    const __nv_bfloat16* q_all,
    const __nv_bfloat16* b_all,
    __nv_bfloat16* M_states,
    __nv_bfloat16* output,
    __nv_bfloat16* M_checkpoints,
    __nv_bfloat16* Sq_cache,
    __nv_bfloat16* row_gate_cache,
    __nv_bfloat16* col_gate_cache,
    int checkpoint_interval, cudaStream_t stream
) {
    int shared_size = calc_e83_forward_shared_mem(n_state, K);

    #define DISPATCH_E83_IB_FWD_BF16(N, K_VAL) \
        E83CircularInputBiasForwardKernel_BF16<N, K_VAL><<<B, 256, shared_size, stream>>>( \
            T, B, kv_all, q_all, b_all, M_states, output, M_checkpoints, \
            Sq_cache, row_gate_cache, col_gate_cache, checkpoint_interval);

    if (K == 2) {
        switch (n_state) {
            case 16: DISPATCH_E83_IB_FWD_BF16(16, 2); break;
            case 24: DISPATCH_E83_IB_FWD_BF16(24, 2); break;
            case 32: DISPATCH_E83_IB_FWD_BF16(32, 2); break;
            case 48: DISPATCH_E83_IB_FWD_BF16(48, 2); break;
            default: fprintf(stderr, "E83 IB: Unsupported n_state=%d for K=2\n", n_state);
        }
    } else if (K == 3) {
        switch (n_state) {
            case 16: DISPATCH_E83_IB_FWD_BF16(16, 3); break;
            case 24: DISPATCH_E83_IB_FWD_BF16(24, 3); break;
            case 32: DISPATCH_E83_IB_FWD_BF16(32, 3); break;
            case 48: DISPATCH_E83_IB_FWD_BF16(48, 3); break;
            default: fprintf(stderr, "E83 IB: Unsupported n_state=%d for K=3\n", n_state);
        }
    } else if (K == 4) {
        switch (n_state) {
            case 16: DISPATCH_E83_IB_FWD_BF16(16, 4); break;
            case 24: DISPATCH_E83_IB_FWD_BF16(24, 4); break;
            case 32: DISPATCH_E83_IB_FWD_BF16(32, 4); break;
            default: fprintf(stderr, "E83 IB: Unsupported n_state=%d for K=4\n", n_state);
        }
    } else if (K == 8) {
        switch (n_state) {
            case 16: DISPATCH_E83_IB_FWD_BF16(16, 8); break;
            case 24: DISPATCH_E83_IB_FWD_BF16(24, 8); break;
            default: fprintf(stderr, "E83 IB: Unsupported n_state=%d for K=8\n", n_state);
        }
    } else {
        fprintf(stderr, "E83 IB: Unsupported K=%d (use 2, 3, 4, or 8)\n", K);
    }

    #undef DISPATCH_E83_IB_FWD_BF16
}

void dispatch_e83_circular_input_bias_backward(
    int T, int B, int n_state, int K,
    const __nv_bfloat16* kv_all,
    const __nv_bfloat16* q_all,
    const __nv_bfloat16* b_all,
    const __nv_bfloat16* row_gate_cache,
    const __nv_bfloat16* col_gate_cache,
    const __nv_bfloat16* M_checkpoints,
    const __nv_bfloat16* Sq_cache,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* d_kv_all,
    __nv_bfloat16* d_q_all,
    __nv_bfloat16* d_b_all,
    int checkpoint_interval, cudaStream_t stream
) {
    int shared_size = calc_e83_backward_shared_mem(n_state, K);

    #define DISPATCH_E83_IB_BWD_BF16(N, K_VAL) \
        E83CircularInputBiasBackwardKernel_BF16<N, K_VAL><<<B, 256, shared_size, stream>>>( \
            T, B, kv_all, q_all, b_all, row_gate_cache, col_gate_cache, \
            M_checkpoints, Sq_cache, d_output, d_kv_all, d_q_all, d_b_all, \
            checkpoint_interval);

    if (K == 2) {
        switch (n_state) {
            case 16: DISPATCH_E83_IB_BWD_BF16(16, 2); break;
            case 24: DISPATCH_E83_IB_BWD_BF16(24, 2); break;
            case 32: DISPATCH_E83_IB_BWD_BF16(32, 2); break;
            case 48: DISPATCH_E83_IB_BWD_BF16(48, 2); break;
            default: fprintf(stderr, "E83 IB: Unsupported n_state=%d for K=2\n", n_state);
        }
    } else if (K == 3) {
        switch (n_state) {
            case 16: DISPATCH_E83_IB_BWD_BF16(16, 3); break;
            case 24: DISPATCH_E83_IB_BWD_BF16(24, 3); break;
            case 32: DISPATCH_E83_IB_BWD_BF16(32, 3); break;
            case 48: DISPATCH_E83_IB_BWD_BF16(48, 3); break;
            default: fprintf(stderr, "E83 IB: Unsupported n_state=%d for K=3\n", n_state);
        }
    } else if (K == 4) {
        switch (n_state) {
            case 16: DISPATCH_E83_IB_BWD_BF16(16, 4); break;
            case 24: DISPATCH_E83_IB_BWD_BF16(24, 4); break;
            case 32: DISPATCH_E83_IB_BWD_BF16(32, 4); break;
            default: fprintf(stderr, "E83 IB: Unsupported n_state=%d for K=4\n", n_state);
        }
    } else if (K == 8) {
        switch (n_state) {
            case 16: DISPATCH_E83_IB_BWD_BF16(16, 8); break;
            case 24: DISPATCH_E83_IB_BWD_BF16(24, 8); break;
            default: fprintf(stderr, "E83 IB: Unsupported n_state=%d for K=8\n", n_state);
        }
    } else {
        fprintf(stderr, "E83 IB: Unsupported K=%d (use 2, 3, 4, or 8)\n", K);
    }

    #undef DISPATCH_E83_IB_BWD_BF16
}

// ============================================================================
// E83CircularForward/Backward Implementation (wrapper class for Python bindings)
// ============================================================================

template<typename DataT>
E83CircularForward<DataT>::E83CircularForward(
    bool training, int batch_size, int n_state, int dim, int K,
    const cublasHandle_t& blas_handle, const cudaStream_t& stream)
    : training_(training), batch_size_(batch_size), n_state_(n_state),
      dim_(dim), K_(K), blas_handle_(blas_handle), stream_(stream) {}

template<typename DataT>
void E83CircularForward<DataT>::Run(
    int steps,
    const DataT* W_kv,
    const DataT* W_q,
    const DataT* B_gates,
    const DataT* x,
    DataT* M_states,
    DataT* output,
    DataT* kv_cache,
    DataT* q_cache,
    DataT* M_checkpoints,
    DataT* Sq_cache,
    DataT* row_gate_cache,
    DataT* col_gate_cache
) {
    int T = steps;
    int B = batch_size_;
    int n = n_state_;
    int d = dim_;
    int K = K_;
    int checkpoint_interval = E83_CHECKPOINT_INTERVAL;

    const float alpha = 1.0f, beta_zero = 0.0f;

    cudaDataType_t data_type = std::is_same<DataT, __nv_bfloat16>::value ? CUDA_R_16BF : CUDA_R_32F;

    // kv projection: kv_cache = W_kv @ x  [K*2*n, T*B]
    cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                 K * 2 * n, T * B, d,
                 &alpha,
                 W_kv, data_type, d,
                 x, data_type, d,
                 &beta_zero,
                 kv_cache, data_type, K * 2 * n,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // q projection: q_cache = W_q @ x  [n, T*B]
    cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                 n, T * B, d,
                 &alpha,
                 W_q, data_type, d,
                 x, data_type, d,
                 &beta_zero,
                 q_cache, data_type, n,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    if constexpr (std::is_same<DataT, __nv_bfloat16>::value) {
        dispatch_e83_circular_forward(T, B, n, K, kv_cache, q_cache, B_gates,
                                      M_states, output, M_checkpoints, Sq_cache,
                                      row_gate_cache, col_gate_cache,
                                      checkpoint_interval, stream_);
    } else {
        dispatch_e83_circular_forward_fp32(T, B, n, K,
                                           reinterpret_cast<const float*>(kv_cache),
                                           reinterpret_cast<const float*>(q_cache),
                                           reinterpret_cast<const float*>(B_gates),
                                           reinterpret_cast<float*>(M_states),
                                           reinterpret_cast<float*>(output),
                                           reinterpret_cast<float*>(M_checkpoints),
                                           reinterpret_cast<float*>(Sq_cache),
                                           reinterpret_cast<float*>(row_gate_cache),
                                           reinterpret_cast<float*>(col_gate_cache),
                                           checkpoint_interval, stream_);
    }
}

template<typename DataT>
E83CircularBackward<DataT>::E83CircularBackward(
    int batch_size, int n_state, int dim, int K,
    const cublasHandle_t& blas_handle, const cudaStream_t& stream)
    : batch_size_(batch_size), n_state_(n_state), dim_(dim), K_(K),
      blas_handle_(blas_handle), stream_(stream) {}

template<typename DataT>
void E83CircularBackward<DataT>::Run(
    int steps,
    const DataT* W_kv,
    const DataT* W_q,
    const DataT* B_gates,
    const DataT* x,
    const DataT* kv_cache,
    const DataT* q_cache,
    const DataT* M_checkpoints,
    const DataT* Sq_cache,
    const DataT* row_gate_cache,
    const DataT* col_gate_cache,
    const DataT* d_output,
    DataT* d_x,
    DataT* d_W_kv,
    DataT* d_W_q,
    DataT* d_B_gates,
    DataT* d_kv_cache,
    DataT* d_q_cache,
    float* d_B_gates_accum
) {
    int T = steps;
    int B = batch_size_;
    int n = n_state_;
    int d = dim_;
    int K = K_;
    int checkpoint_interval = E83_CHECKPOINT_INTERVAL;

    const float alpha = 1.0f, beta_zero = 0.0f;

    // Zero accumulators
    cudaMemsetAsync(d_B_gates_accum, 0, K * n * sizeof(float), stream_);

    cudaDataType_t data_type = std::is_same<DataT, __nv_bfloat16>::value ? CUDA_R_16BF : CUDA_R_32F;

    if constexpr (std::is_same<DataT, __nv_bfloat16>::value) {
        dispatch_e83_circular_backward(T, B, n, K, kv_cache, q_cache, B_gates,
                                       row_gate_cache, col_gate_cache,
                                       M_checkpoints, Sq_cache, d_output,
                                       d_kv_cache, d_q_cache, d_B_gates_accum,
                                       checkpoint_interval, stream_);
    } else {
        dispatch_e83_circular_backward_fp32(T, B, n, K,
                                            reinterpret_cast<const float*>(kv_cache),
                                            reinterpret_cast<const float*>(q_cache),
                                            reinterpret_cast<const float*>(B_gates),
                                            reinterpret_cast<const float*>(row_gate_cache),
                                            reinterpret_cast<const float*>(col_gate_cache),
                                            reinterpret_cast<const float*>(M_checkpoints),
                                            reinterpret_cast<const float*>(Sq_cache),
                                            reinterpret_cast<const float*>(d_output),
                                            reinterpret_cast<float*>(d_kv_cache),
                                            reinterpret_cast<float*>(d_q_cache),
                                            d_B_gates_accum,
                                            checkpoint_interval, stream_);
    }

    // d_x from kv: d_x += W_kv @ d_kv_cache
    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                 d, T * B, K * 2 * n,
                 &alpha,
                 W_kv, data_type, d,
                 d_kv_cache, data_type, K * 2 * n,
                 &beta_zero,
                 d_x, data_type, d,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // d_x from q: d_x += W_q @ d_q_cache
    const float beta_one = 1.0f;
    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                 d, T * B, n,
                 &alpha,
                 W_q, data_type, d,
                 d_q_cache, data_type, n,
                 &beta_one,
                 d_x, data_type, d,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // d_W_kv = d_kv_cache @ x^T
    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
                 K * 2 * n, d, T * B,
                 &alpha,
                 d_kv_cache, data_type, K * 2 * n,
                 x, data_type, d,
                 &beta_zero,
                 d_W_kv, data_type, K * 2 * n,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // d_W_q = d_q_cache @ x^T
    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
                 n, d, T * B,
                 &alpha,
                 d_q_cache, data_type, n,
                 x, data_type, d,
                 &beta_zero,
                 d_W_q, data_type, n,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // Convert accumulated bias gradients (float) to output dtype
    int bias_size = K * n;
    if constexpr (std::is_same<DataT, __nv_bfloat16>::value) {
        int threads = 256;
        int blocks = (bias_size + threads - 1) / threads;
        ConvertFloatToBF16Kernel_E83<<<blocks, threads, 0, stream_>>>(
            d_B_gates_accum, d_B_gates, bias_size);
    } else {
        cudaMemcpyAsync(d_B_gates, d_B_gates_accum, bias_size * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream_);
    }
}

// Explicit template instantiations
template class E83CircularForward<__nv_bfloat16>;
template class E83CircularForward<float>;
template class E83CircularBackward<__nv_bfloat16>;
template class E83CircularBackward<float>;

// ============================================================================
// E83CircularInputBiasForward/Backward Implementation
// ============================================================================

template<typename DataT>
E83CircularInputBiasForward<DataT>::E83CircularInputBiasForward(
    bool training, int batch_size, int n_state, int dim, int K,
    const cublasHandle_t& blas_handle, const cudaStream_t& stream)
    : training_(training), batch_size_(batch_size), n_state_(n_state),
      dim_(dim), K_(K), blas_handle_(blas_handle), stream_(stream) {}

template<typename DataT>
void E83CircularInputBiasForward<DataT>::Run(
    int steps,
    const DataT* W_kv,
    const DataT* W_q,
    const DataT* W_b,
    const DataT* x,
    DataT* M_states,
    DataT* output,
    DataT* kv_cache,
    DataT* q_cache,
    DataT* b_cache,
    DataT* M_checkpoints,
    DataT* Sq_cache,
    DataT* row_gate_cache,
    DataT* col_gate_cache
) {
    int T = steps;
    int B = batch_size_;
    int n = n_state_;
    int d = dim_;
    int K = K_;
    int checkpoint_interval = E83_CHECKPOINT_INTERVAL;

    const float alpha = 1.0f, beta_zero = 0.0f;

    cudaDataType_t data_type = std::is_same<DataT, __nv_bfloat16>::value ? CUDA_R_16BF : CUDA_R_32F;

    // kv projection: kv_cache = W_kv @ x  [K*2*n, T*B]
    cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                 K * 2 * n, T * B, d,
                 &alpha,
                 W_kv, data_type, d,
                 x, data_type, d,
                 &beta_zero,
                 kv_cache, data_type, K * 2 * n,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // q projection: q_cache = W_q @ x  [n, T*B]
    cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                 n, T * B, d,
                 &alpha,
                 W_q, data_type, d,
                 x, data_type, d,
                 &beta_zero,
                 q_cache, data_type, n,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // bias projection: b_cache = W_b @ x  [K*n, T*B] -> transpose to [T*B, K*n]
    // Actually, we store as [T*B, K*n] for the kernel to access per-timestep
    // cuBLAS outputs column-major, so [K*n, T*B] column-major = [T*B, K*n] row-major
    cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                 K * n, T * B, d,
                 &alpha,
                 W_b, data_type, d,
                 x, data_type, d,
                 &beta_zero,
                 b_cache, data_type, K * n,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    if constexpr (std::is_same<DataT, __nv_bfloat16>::value) {
        dispatch_e83_circular_input_bias_forward(T, B, n, K, kv_cache, q_cache, b_cache,
                                                  M_states, output, M_checkpoints, Sq_cache,
                                                  row_gate_cache, col_gate_cache,
                                                  checkpoint_interval, stream_);
    } else {
        // FP32 input-bias not implemented - fall back to error
        fprintf(stderr, "E83 Input-Bias: FP32 not implemented\n");
    }
}

template<typename DataT>
E83CircularInputBiasBackward<DataT>::E83CircularInputBiasBackward(
    int batch_size, int n_state, int dim, int K,
    const cublasHandle_t& blas_handle, const cudaStream_t& stream)
    : batch_size_(batch_size), n_state_(n_state), dim_(dim), K_(K),
      blas_handle_(blas_handle), stream_(stream) {}

template<typename DataT>
void E83CircularInputBiasBackward<DataT>::Run(
    int steps,
    const DataT* W_kv,
    const DataT* W_q,
    const DataT* W_b,
    const DataT* x,
    const DataT* kv_cache,
    const DataT* q_cache,
    const DataT* b_cache,
    const DataT* M_checkpoints,
    const DataT* Sq_cache,
    const DataT* row_gate_cache,
    const DataT* col_gate_cache,
    const DataT* d_output,
    DataT* d_x,
    DataT* d_W_kv,
    DataT* d_W_q,
    DataT* d_W_b,
    DataT* d_kv_cache,
    DataT* d_q_cache,
    DataT* d_b_cache
) {
    int T = steps;
    int B = batch_size_;
    int n = n_state_;
    int d = dim_;
    int K = K_;
    int checkpoint_interval = E83_CHECKPOINT_INTERVAL;

    const float alpha = 1.0f, beta_zero = 0.0f;

    cudaDataType_t data_type = std::is_same<DataT, __nv_bfloat16>::value ? CUDA_R_16BF : CUDA_R_32F;

    if constexpr (std::is_same<DataT, __nv_bfloat16>::value) {
        dispatch_e83_circular_input_bias_backward(T, B, n, K, kv_cache, q_cache, b_cache,
                                                   row_gate_cache, col_gate_cache,
                                                   M_checkpoints, Sq_cache, d_output,
                                                   d_kv_cache, d_q_cache, d_b_cache,
                                                   checkpoint_interval, stream_);
    } else {
        // FP32 input-bias not implemented
        fprintf(stderr, "E83 Input-Bias: FP32 backward not implemented\n");
        return;
    }

    // d_x from kv: d_x = W_kv @ d_kv_cache
    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                 d, T * B, K * 2 * n,
                 &alpha,
                 W_kv, data_type, d,
                 d_kv_cache, data_type, K * 2 * n,
                 &beta_zero,
                 d_x, data_type, d,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // d_x from q: d_x += W_q @ d_q_cache
    const float beta_one = 1.0f;
    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                 d, T * B, n,
                 &alpha,
                 W_q, data_type, d,
                 d_q_cache, data_type, n,
                 &beta_one,
                 d_x, data_type, d,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // d_x from b: d_x += W_b @ d_b_cache
    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                 d, T * B, K * n,
                 &alpha,
                 W_b, data_type, d,
                 d_b_cache, data_type, K * n,
                 &beta_one,
                 d_x, data_type, d,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // d_W_kv = d_kv_cache @ x^T
    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
                 K * 2 * n, d, T * B,
                 &alpha,
                 d_kv_cache, data_type, K * 2 * n,
                 x, data_type, d,
                 &beta_zero,
                 d_W_kv, data_type, K * 2 * n,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // d_W_q = d_q_cache @ x^T
    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
                 n, d, T * B,
                 &alpha,
                 d_q_cache, data_type, n,
                 x, data_type, d,
                 &beta_zero,
                 d_W_q, data_type, n,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // d_W_b = d_b_cache @ x^T
    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
                 K * n, d, T * B,
                 &alpha,
                 d_b_cache, data_type, K * n,
                 x, data_type, d,
                 &beta_zero,
                 d_W_b, data_type, K * n,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

// Explicit template instantiations for input-bias
template class E83CircularInputBiasForward<__nv_bfloat16>;
template class E83CircularInputBiasForward<float>;
template class E83CircularInputBiasBackward<__nv_bfloat16>;
template class E83CircularInputBiasBackward<float>;

}  // namespace elman
