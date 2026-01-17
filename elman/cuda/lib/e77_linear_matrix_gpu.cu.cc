/**
 * E77 Linear Matrix State CUDA Kernel
 *
 * Combines E42's insights with E76's matrix state:
 * - E42: Linear recurrence (no tanh) + self-gating output
 * - E76: Matrix state with delta rule
 *
 * Key difference: Put the nonlinearity at OUTPUT, not in the state update.
 * This allows gradients to flow through the matrix state unimpeded.
 *
 * Architecture:
 *   FUSED projection: kvqg = W_kvqg @ x  (single GEMM for k, v, q, gate)
 *   k = kvqg[:n], v = kvqg[n:2n], q = kvqg[2n:3n], gate = kvqg[3n:]
 *
 *   decay = sigmoid(gate + b_gate)   [simple sigmoid, no log-space]
 *   k_norm = k / ||k||
 *   retrieved = S @ k_norm
 *   delta = v - retrieved
 *   S = decay * S + outer(delta, k_norm)   [LINEAR - no tanh!]
 *
 *   Sq = S @ q
 *   output = Sq * silu(Sq)   [self-gating output]
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include "hasty/elman_ladder.h"

#define CHECKPOINT_INTERVAL 16

namespace elman {

// ============================================================================
// E77 Forward Kernel - Linear Matrix State with Self-Gating Output
// ============================================================================

template<int N_STATE>
__global__ void E77LinearForwardKernel_BF16(
    int T,
    int B,
    const __nv_bfloat16* __restrict__ kvqg_all,   // [4*N_STATE, T*B] column-major from GEMM
    const __nv_bfloat16* __restrict__ b_gate,     // [N_STATE] gate bias
    __nv_bfloat16* __restrict__ S,                // [B, N_STATE, N_STATE]
    __nv_bfloat16* __restrict__ output,           // [T, B, N_STATE]
    __nv_bfloat16* __restrict__ S_checkpoints,    // [num_checkpoints, B, N_STATE, N_STATE]
    __nv_bfloat16* __restrict__ Sq_cache,         // [T, B, N_STATE]
    __nv_bfloat16* __restrict__ decay_cache,      // [T, B, N_STATE]
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
    float* gate_shared = retrieved + N_STATE;        // [N_STATE]
    float* decay_shared = gate_shared + N_STATE;     // [N_STATE]
    float* b_gate_shared = decay_shared + N_STATE;   // [N_STATE]

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;
    const int STRIDE = 4 * N_STATE;  // Column stride in GEMM output

    // Load b_gate to shared memory (constant per batch)
    if (tid < N_STATE) {
        b_gate_shared[tid] = __bfloat162float(b_gate[tid]);
    }
    __syncthreads();

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
        // GEMM output is [4*N, T*B] column-major
        // For position (t, b), column index is (t*B + b)
        // k[i] at row i, v[i] at row N+i, q[i] at row 2N+i, gate[i] at row 3N+i
        int col_idx = (t * B + b) * STRIDE;

        // Load k, v, q, gate for this timestep with correct strided access
        if (tid < N_STATE) {
            k_shared[tid] = __bfloat162float(kvqg_all[col_idx + tid]);
            v_shared[tid] = __bfloat162float(kvqg_all[col_idx + N_STATE + tid]);
            q_shared[tid] = __bfloat162float(kvqg_all[col_idx + 2 * N_STATE + tid]);
            gate_shared[tid] = __bfloat162float(kvqg_all[col_idx + 3 * N_STATE + tid]);
        }
        __syncthreads();

        // Compute decay = sigmoid(gate + b_gate)
        if (tid < N_STATE) {
            float val = gate_shared[tid] + b_gate_shared[tid];
            decay_shared[tid] = 1.0f / (1.0f + expf(-val));
            // Cache decay for backward
            decay_cache[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(decay_shared[tid]);
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

        // Update state: S = decay * S + outer(delta, k_norm)  [LINEAR - no tanh!]
        for (int i = tid; i < n2; i += blockDim.x) {
            int row = i / N_STATE;
            int col = i % N_STATE;

            float decay_val = decay_shared[row];
            float delta_i = v_shared[row] - retrieved[row];
            float update = decay_val * S_shared[i] + delta_i * k_shared[col];

            // LINEAR: no tanh
            S_shared[i] = update;
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
// E77 Backward Kernel - Linear Matrix State
// ============================================================================

template<int N_STATE>
__global__ void E77LinearBackwardKernel_BF16(
    int T,
    int B,
    const __nv_bfloat16* __restrict__ kvqg_all,   // [4*N_STATE, T*B] column-major from GEMM
    const __nv_bfloat16* __restrict__ b_gate,
    const __nv_bfloat16* __restrict__ decay_cache,
    const __nv_bfloat16* __restrict__ S_checkpoints,
    const __nv_bfloat16* __restrict__ Sq_cache,
    const __nv_bfloat16* __restrict__ d_output,
    __nv_bfloat16* __restrict__ d_kvqg_all,    // [4*N_STATE, T*B] output gradients
    float* __restrict__ d_b_gate_accum,     // [N_STATE] accumulated across batches
    int checkpoint_interval
) {
    int b = blockIdx.x;
    if (b >= B) return;

    extern __shared__ float shared_mem[];
    float* S = shared_mem;                            // [N_STATE * N_STATE]
    float* dS = S + N_STATE * N_STATE;                // [N_STATE * N_STATE]
    float* k_raw = dS + N_STATE * N_STATE;            // [N_STATE]
    float* v_raw = k_raw + N_STATE;                   // [N_STATE]
    float* q_raw = v_raw + N_STATE;                   // [N_STATE]
    float* k_norm = q_raw + N_STATE;                  // [N_STATE]
    float* delta = k_norm + N_STATE;                  // [N_STATE]
    float* retrieved = delta + N_STATE;               // [N_STATE]
    float* gate = retrieved + N_STATE;                // [N_STATE]
    float* decay = gate + N_STATE;                    // [N_STATE]
    float* b_gate_shared = decay + N_STATE;           // [N_STATE]
    float* d_k_raw = b_gate_shared + N_STATE;         // [N_STATE]
    float* d_v_raw = d_k_raw + N_STATE;               // [N_STATE]
    float* d_q_raw = d_v_raw + N_STATE;               // [N_STATE]
    float* d_Sq_shared = d_q_raw + N_STATE;           // [N_STATE]
    float* d_delta = d_Sq_shared + N_STATE;           // [N_STATE]
    float* d_k_norm = d_delta + N_STATE;              // [N_STATE]
    float* d_decay = d_k_norm + N_STATE;              // [N_STATE]
    float* d_b_gate_local = d_decay + N_STATE;        // [N_STATE]

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;
    const int STRIDE = 4 * N_STATE;

    // Load b_gate
    if (tid < N_STATE) {
        b_gate_shared[tid] = __bfloat162float(b_gate[tid]);
        d_b_gate_local[tid] = 0.0f;
    }
    __syncthreads();

    // Initialize dS to zero
    for (int i = tid; i < n2; i += blockDim.x) {
        dS[i] = 0.0f;
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
                    k_raw[tid] = __bfloat162float(kvqg_all[col_idx + tid]);
                    v_raw[tid] = __bfloat162float(kvqg_all[col_idx + N_STATE + tid]);
                    q_raw[tid] = __bfloat162float(kvqg_all[col_idx + 2 * N_STATE + tid]);
                    gate[tid] = __bfloat162float(kvqg_all[col_idx + 3 * N_STATE + tid]);
                    decay[tid] = __bfloat162float(decay_cache[tt * B * N_STATE + b * N_STATE + tid]);
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

                // Compute retrieved
                if (tid < N_STATE) {
                    float sum = 0.0f;
                    for (int j = 0; j < N_STATE; j++) {
                        sum += S[tid * N_STATE + j] * k_norm[j];
                    }
                    retrieved[tid] = sum;
                    delta[tid] = v_raw[tid] - retrieved[tid];
                }
                __syncthreads();

                // Update S if not at target step (LINEAR - no tanh)
                if (tt < t) {
                    for (int i = tid; i < n2; i += blockDim.x) {
                        int row = i / N_STATE;
                        int col = i % N_STATE;
                        float update = decay[row] * S[i] + delta[row] * k_norm[col];
                        S[i] = update;  // LINEAR
                    }
                    __syncthreads();
                }
            }

            // Now S holds S_{t-1}, k_norm/v_raw/q_raw/delta/decay are for step t

            // Backward through output: out = Sq * silu(Sq) = Sq * Sq * sigmoid(Sq)
            if (tid < N_STATE) {
                float d_out = __bfloat162float(d_output[t * B * N_STATE + b * N_STATE + tid]);
                float Sq = __bfloat162float(Sq_cache[t * B * N_STATE + b * N_STATE + tid]);
                float sig = 1.0f / (1.0f + expf(-Sq));
                // d_out/d_Sq for Sq * Sq * sig = 2*Sq*sig + Sq*Sq*sig*(1-sig)
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

            // d_q = S_t^T @ d_Sq (S_t is after update)
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    // S_t[i, tid] = decay[i] * S[i, tid] + delta[i] * k_norm[tid]
                    float S_t_ij = decay[i] * S[i * N_STATE + tid] + delta[i] * k_norm[tid];
                    sum += S_t_ij * d_Sq_shared[i];
                }
                d_q_raw[tid] = sum;
            }
            __syncthreads();

            // Backward through state update (LINEAR - gradient passes through directly)
            // S_t = decay * S + outer(delta, k_norm)
            // dS gets multiplied by decay, d_delta and d_k_norm from outer product
            if (tid < N_STATE) {
                float d_delta_local = 0.0f;
                float d_decay_local = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    // LINEAR: d_pre = dS[tid, j] (no tanh derivative)
                    float d_pre = dS[tid * N_STATE + j];
                    d_delta_local += d_pre * k_norm[j];
                    d_decay_local += d_pre * S[tid * N_STATE + j];
                }
                d_delta[tid] = d_delta_local;
                d_decay[tid] = d_decay_local;
            }
            __syncthreads();

            // Compute d_k_norm
            if (tid < N_STATE) {
                float d_k_norm_local = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    // LINEAR: d_pre = dS[i, tid]
                    float d_pre = dS[i * N_STATE + tid];
                    d_k_norm_local += d_pre * delta[i];
                }
                d_k_norm[tid] = d_k_norm_local;
            }
            __syncthreads();

            // Backward through decay: decay = sigmoid(gate + b_gate)
            // d_gate = d_decay * decay * (1 - decay)
            float d_gate_val_local = 0.0f;
            if (tid < N_STATE) {
                float d_dec = d_decay[tid];
                float dec_val = decay[tid];
                d_gate_val_local = d_dec * dec_val * (1.0f - dec_val);

                // Accumulate d_b_gate
                d_b_gate_local[tid] += d_gate_val_local;
            }
            __syncthreads();

            // d_v = d_delta
            if (tid < N_STATE) {
                d_v_raw[tid] = d_delta[tid];
            }

            // d_k_norm += S^T @ (-d_delta)  (from retrieved computation)
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    sum += S[i * N_STATE + tid] * (-d_delta[i]);
                }
                d_k_norm[tid] += sum;
            }
            __syncthreads();

            // Convert d_k_norm to d_k_raw (backward through normalization)
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

            // Write gradients with fused layout
            int col_idx_t = (t * B + b) * STRIDE;
            if (tid < N_STATE) {
                d_kvqg_all[col_idx_t + tid] = __float2bfloat16(d_k_raw[tid]);
                d_kvqg_all[col_idx_t + N_STATE + tid] = __float2bfloat16(d_v_raw[tid]);
                d_kvqg_all[col_idx_t + 2 * N_STATE + tid] = __float2bfloat16(d_q_raw[tid]);
                d_kvqg_all[col_idx_t + 3 * N_STATE + tid] = __float2bfloat16(d_gate_val_local);
            }
            __syncthreads();

            // Update dS for next iteration (LINEAR: gradient passes through scaled by decay)
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                // LINEAR: d_pre = dS[i]
                float d_pre = dS[i];
                // dS_{t-1} = d_pre * decay + contribution from retrieved gradient
                dS[i] = d_pre * decay[row] + (-d_delta[row]) * k_norm[col];
            }
            __syncthreads();
        }
    }

    // Atomically accumulate b_gate gradients
    if (tid < N_STATE) {
        atomicAdd(&d_b_gate_accum[tid], d_b_gate_local[tid]);
    }
}

// ============================================================================
// E77 Forward Kernel - FP32 version
// ============================================================================

template<int N_STATE>
__global__ void E77LinearForwardKernel_FP32(
    int T,
    int B,
    const float* __restrict__ kvqg_all,   // [4*N_STATE, T*B] column-major from GEMM
    const float* __restrict__ b_gate,     // [N_STATE] gate bias
    float* __restrict__ S,                // [B, N_STATE, N_STATE]
    float* __restrict__ output,           // [T, B, N_STATE]
    float* __restrict__ S_checkpoints,    // [num_checkpoints, B, N_STATE, N_STATE]
    float* __restrict__ Sq_cache,         // [T, B, N_STATE]
    float* __restrict__ decay_cache,      // [T, B, N_STATE]
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
    float* gate_shared = retrieved + N_STATE;
    float* decay_shared = gate_shared + N_STATE;
    float* b_gate_shared = decay_shared + N_STATE;

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;
    const int STRIDE = 4 * N_STATE;

    // Load b_gate to shared memory
    if (tid < N_STATE) {
        b_gate_shared[tid] = b_gate[tid];
    }
    __syncthreads();

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
            k_shared[tid] = kvqg_all[col_idx + tid];
            v_shared[tid] = kvqg_all[col_idx + N_STATE + tid];
            q_shared[tid] = kvqg_all[col_idx + 2 * N_STATE + tid];
            gate_shared[tid] = kvqg_all[col_idx + 3 * N_STATE + tid];
        }
        __syncthreads();

        // Compute decay = sigmoid(gate + b_gate)
        if (tid < N_STATE) {
            float val = gate_shared[tid] + b_gate_shared[tid];
            decay_shared[tid] = 1.0f / (1.0f + expf(-val));
            decay_cache[t * B * N_STATE + b * N_STATE + tid] = decay_shared[tid];
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

        // Update state: S = decay * S + outer(delta, k_norm)
        for (int i = tid; i < n2; i += blockDim.x) {
            int row = i / N_STATE;
            int col = i % N_STATE;
            float decay_val = decay_shared[row];
            float delta_i = v_shared[row] - retrieved[row];
            S_shared[i] = decay_val * S_shared[i] + delta_i * k_shared[col];
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

        // Compute output: Sq = S @ q, then self-gate
        if (tid < N_STATE) {
            float Sq = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < N_STATE; j++) {
                Sq += S_shared[tid * N_STATE + j] * q_shared[j];
            }
            Sq_cache[t * B * N_STATE + b * N_STATE + tid] = Sq;
            float sig = 1.0f / (1.0f + expf(-Sq));
            output[t * B * N_STATE + b * N_STATE + tid] = Sq * Sq * sig;
        }
        __syncthreads();
    }

    // Write final state back
    for (int i = tid; i < n2; i += blockDim.x) {
        S[b * n2 + i] = S_shared[i];
    }
}

// ============================================================================
// E77 Backward Kernel - FP32 version
// ============================================================================

template<int N_STATE>
__global__ void E77LinearBackwardKernel_FP32(
    int T,
    int B,
    const float* __restrict__ kvqg_all,
    const float* __restrict__ b_gate,
    const float* __restrict__ decay_cache,
    const float* __restrict__ S_checkpoints,
    const float* __restrict__ Sq_cache,
    const float* __restrict__ d_output,
    float* __restrict__ d_kvqg_all,
    float* __restrict__ d_b_gate_accum,
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
    float* gate = retrieved + N_STATE;
    float* decay = gate + N_STATE;
    float* b_gate_shared = decay + N_STATE;
    float* d_k_raw = b_gate_shared + N_STATE;
    float* d_v_raw = d_k_raw + N_STATE;
    float* d_q_raw = d_v_raw + N_STATE;
    float* d_Sq_shared = d_q_raw + N_STATE;
    float* d_delta = d_Sq_shared + N_STATE;
    float* d_k_norm = d_delta + N_STATE;
    float* d_decay = d_k_norm + N_STATE;
    float* d_b_gate_local = d_decay + N_STATE;

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;
    const int STRIDE = 4 * N_STATE;

    if (tid < N_STATE) {
        b_gate_shared[tid] = b_gate[tid];
        d_b_gate_local[tid] = 0.0f;
    }
    __syncthreads();

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
                int col_idx = (tt * B + b) * STRIDE;
                if (tid < N_STATE) {
                    k_raw[tid] = kvqg_all[col_idx + tid];
                    v_raw[tid] = kvqg_all[col_idx + N_STATE + tid];
                    q_raw[tid] = kvqg_all[col_idx + 2 * N_STATE + tid];
                    gate[tid] = kvqg_all[col_idx + 3 * N_STATE + tid];
                    decay[tid] = decay_cache[tt * B * N_STATE + b * N_STATE + tid];
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
                        S[i] = decay[row] * S[i] + delta[row] * k_norm[col];
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
                    float S_t_ij = decay[i] * S[i * N_STATE + tid] + delta[i] * k_norm[tid];
                    sum += S_t_ij * d_Sq_shared[i];
                }
                d_q_raw[tid] = sum;
            }
            __syncthreads();

            if (tid < N_STATE) {
                float d_delta_local = 0.0f;
                float d_decay_local = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    float d_pre = dS[tid * N_STATE + j];
                    d_delta_local += d_pre * k_norm[j];
                    d_decay_local += d_pre * S[tid * N_STATE + j];
                }
                d_delta[tid] = d_delta_local;
                d_decay[tid] = d_decay_local;
            }
            __syncthreads();

            if (tid < N_STATE) {
                float d_k_norm_local = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    float d_pre = dS[i * N_STATE + tid];
                    d_k_norm_local += d_pre * delta[i];
                }
                d_k_norm[tid] = d_k_norm_local;
            }
            __syncthreads();

            float d_gate_val_local = 0.0f;
            if (tid < N_STATE) {
                float d_dec = d_decay[tid];
                float dec_val = decay[tid];
                d_gate_val_local = d_dec * dec_val * (1.0f - dec_val);
                d_b_gate_local[tid] += d_gate_val_local;
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

            int col_idx_t = (t * B + b) * STRIDE;
            if (tid < N_STATE) {
                d_kvqg_all[col_idx_t + tid] = d_k_raw[tid];
                d_kvqg_all[col_idx_t + N_STATE + tid] = d_v_raw[tid];
                d_kvqg_all[col_idx_t + 2 * N_STATE + tid] = d_q_raw[tid];
                d_kvqg_all[col_idx_t + 3 * N_STATE + tid] = d_gate_val_local;
            }
            __syncthreads();

            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                float d_pre = dS[i];
                dS[i] = d_pre * decay[row] + (-d_delta[row]) * k_norm[col];
            }
            __syncthreads();
        }
    }

    if (tid < N_STATE) {
        atomicAdd(&d_b_gate_accum[tid], d_b_gate_local[tid]);
    }
}

// ============================================================================
// Template instantiations for different N_STATE values
// ============================================================================

#define INSTANTIATE_E77_KERNELS_BF16(N) \
    template __global__ void E77LinearForwardKernel_BF16<N>( \
        int, int, const __nv_bfloat16*, const __nv_bfloat16*, \
        __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int); \
    template __global__ void E77LinearBackwardKernel_BF16<N>( \
        int, int, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, \
        const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, \
        __nv_bfloat16*, float*, int);

#define INSTANTIATE_E77_KERNELS_FP32(N) \
    template __global__ void E77LinearForwardKernel_FP32<N>( \
        int, int, const float*, const float*, \
        float*, float*, float*, float*, float*, int); \
    template __global__ void E77LinearBackwardKernel_FP32<N>( \
        int, int, const float*, const float*, const float*, \
        const float*, const float*, const float*, \
        float*, float*, int);

INSTANTIATE_E77_KERNELS_BF16(32)
INSTANTIATE_E77_KERNELS_BF16(48)
INSTANTIATE_E77_KERNELS_BF16(64)
INSTANTIATE_E77_KERNELS_BF16(96)

INSTANTIATE_E77_KERNELS_FP32(32)
INSTANTIATE_E77_KERNELS_FP32(48)
INSTANTIATE_E77_KERNELS_FP32(64)
INSTANTIATE_E77_KERNELS_FP32(96)

// ============================================================================
// Dispatcher functions
// ============================================================================

void dispatch_e77_linear_forward(
    int T, int B, int n_state,
    const __nv_bfloat16* kvqg_all,   // [4*n_state, T*B] fused projection output
    const __nv_bfloat16* b_gate,
    __nv_bfloat16* S, __nv_bfloat16* output,
    __nv_bfloat16* S_checkpoints, __nv_bfloat16* Sq_cache,
    __nv_bfloat16* decay_cache, int checkpoint_interval,
    cudaStream_t stream
) {
    int shared_size = (n_state * n_state + 8 * n_state) * sizeof(float);

    #define DISPATCH_E77_FWD(N) \
        E77LinearForwardKernel_BF16<N><<<B, 256, shared_size, stream>>>( \
            T, B, kvqg_all, b_gate, \
            S, output, S_checkpoints, Sq_cache, decay_cache, checkpoint_interval);

    switch (n_state) {
        case 32: DISPATCH_E77_FWD(32); break;
        case 48: DISPATCH_E77_FWD(48); break;
        case 64: DISPATCH_E77_FWD(64); break;
        case 96: DISPATCH_E77_FWD(96); break;
        default:
            fprintf(stderr, "E77: Unsupported n_state=%d (use 32, 48, 64, or 96)\n", n_state);
    }
    #undef DISPATCH_E77_FWD
}

void dispatch_e77_linear_backward(
    int T, int B, int n_state,
    const __nv_bfloat16* kvqg_all,    // [4*n_state, T*B] fused projection output
    const __nv_bfloat16* b_gate, const __nv_bfloat16* decay_cache,
    const __nv_bfloat16* S_checkpoints, const __nv_bfloat16* Sq_cache,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* d_kvqg_all,        // [4*n_state, T*B] fused gradients
    float* d_b_gate_accum,
    int checkpoint_interval, cudaStream_t stream
) {
    // More shared memory for backward: 2*n2 + 19*n
    int shared_size = (2 * n_state * n_state + 19 * n_state) * sizeof(float);

    #define DISPATCH_E77_BWD(N) \
        E77LinearBackwardKernel_BF16<N><<<B, 256, shared_size, stream>>>( \
            T, B, kvqg_all, b_gate, decay_cache, \
            S_checkpoints, Sq_cache, d_output, \
            d_kvqg_all, d_b_gate_accum, checkpoint_interval);

    switch (n_state) {
        case 32: DISPATCH_E77_BWD(32); break;
        case 48: DISPATCH_E77_BWD(48); break;
        case 64: DISPATCH_E77_BWD(64); break;
        case 96: DISPATCH_E77_BWD(96); break;
        default:
            fprintf(stderr, "E77: Unsupported n_state=%d (use 32, 48, 64, or 96)\n", n_state);
    }
    #undef DISPATCH_E77_BWD
}

// ============================================================================
// FP32 Dispatcher functions
// ============================================================================

void dispatch_e77_linear_forward_fp32(
    int T, int B, int n_state,
    const float* kvqg_all,
    const float* b_gate,
    float* S, float* output,
    float* S_checkpoints, float* Sq_cache,
    float* decay_cache, int checkpoint_interval,
    cudaStream_t stream
) {
    int shared_size = (n_state * n_state + 8 * n_state) * sizeof(float);

    #define DISPATCH_E77_FWD_FP32(N) \
        E77LinearForwardKernel_FP32<N><<<B, 256, shared_size, stream>>>( \
            T, B, kvqg_all, b_gate, \
            S, output, S_checkpoints, Sq_cache, decay_cache, checkpoint_interval);

    switch (n_state) {
        case 32: DISPATCH_E77_FWD_FP32(32); break;
        case 48: DISPATCH_E77_FWD_FP32(48); break;
        case 64: DISPATCH_E77_FWD_FP32(64); break;
        case 96: DISPATCH_E77_FWD_FP32(96); break;
        default:
            fprintf(stderr, "E77: Unsupported n_state=%d (use 32, 48, 64, or 96)\n", n_state);
    }
    #undef DISPATCH_E77_FWD_FP32
}

void dispatch_e77_linear_backward_fp32(
    int T, int B, int n_state,
    const float* kvqg_all,
    const float* b_gate, const float* decay_cache,
    const float* S_checkpoints, const float* Sq_cache,
    const float* d_output,
    float* d_kvqg_all,
    float* d_b_gate_accum,
    int checkpoint_interval, cudaStream_t stream
) {
    int shared_size = (2 * n_state * n_state + 19 * n_state) * sizeof(float);

    #define DISPATCH_E77_BWD_FP32(N) \
        E77LinearBackwardKernel_FP32<N><<<B, 256, shared_size, stream>>>( \
            T, B, kvqg_all, b_gate, decay_cache, \
            S_checkpoints, Sq_cache, d_output, \
            d_kvqg_all, d_b_gate_accum, checkpoint_interval);

    switch (n_state) {
        case 32: DISPATCH_E77_BWD_FP32(32); break;
        case 48: DISPATCH_E77_BWD_FP32(48); break;
        case 64: DISPATCH_E77_BWD_FP32(64); break;
        case 96: DISPATCH_E77_BWD_FP32(96); break;
        default:
            fprintf(stderr, "E77: Unsupported n_state=%d (use 32, 48, 64, or 96)\n", n_state);
    }
    #undef DISPATCH_E77_BWD_FP32
}

// ============================================================================
// E77LinearForward Implementation
// ============================================================================

template<typename DataT>
E77LinearForward<DataT>::E77LinearForward(
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

template<typename DataT>
void E77LinearForward<DataT>::Run(
    int steps,
    const DataT* W_kvqg,   // [4*n_state, dim] FUSED projection
    const DataT* b_gate,   // [n_state]
    const DataT* x,        // [T, B, dim]
    DataT* S,              // [B, n_state, n_state]
    DataT* output,         // [T, B, n_state]
    DataT* kvqg_cache,     // [4*n_state, T*B] for backward (column-major from GEMM)
    DataT* S_cache,        // Checkpoints + Sq cache
    DataT* decay_cache     // [T, B, n_state]
) {
    int T = steps;
    int B = batch_size_;
    int n = n_state_;
    int d = dim_;

    const float alpha = 1.0f, beta_zero = 0.0f;

    // FUSED projection: kvqg = W_kvqg @ x
    // W_kvqg: [4*n, d], x: [d, T*B] -> kvqg: [4*n, T*B] column-major
    cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        4 * n, T * B, d, &alpha,
        W_kvqg, CUDA_R_16BF, d, x, CUDA_R_16BF, d,
        &beta_zero, kvqg_cache, CUDA_R_16BF, 4 * n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // Calculate workspace offsets for checkpoints
    int num_checkpoints = (T + CHECKPOINT_INTERVAL - 1) / CHECKPOINT_INTERVAL + 1;
    DataT* s_checkpoints = S_cache;
    DataT* sq_cache = S_cache + num_checkpoints * B * n * n;

    // Run forward kernel with fused kvqg layout
    // Kernel handles the strided access internally
    dispatch_e77_linear_forward(
        T, B, n,
        (const __nv_bfloat16*)kvqg_cache,   // Fused [4*n, T*B] column-major
        (const __nv_bfloat16*)b_gate,
        (__nv_bfloat16*)S,
        (__nv_bfloat16*)output,
        (__nv_bfloat16*)s_checkpoints,
        (__nv_bfloat16*)sq_cache,
        (__nv_bfloat16*)decay_cache,
        CHECKPOINT_INTERVAL,
        stream_);
}

template struct E77LinearForward<__nv_bfloat16>;

// ============================================================================
// E77LinearForward - FP32 specialization
// ============================================================================

template<>
void E77LinearForward<float>::Run(
    int steps,
    const float* W_kvqg,
    const float* b_gate,
    const float* x,
    float* S,
    float* output,
    float* kvqg_cache,
    float* S_cache,
    float* decay_cache
) {
    int T = steps;
    int B = batch_size_;
    int n = n_state_;
    int d = dim_;

    const float alpha = 1.0f, beta_zero = 0.0f;

    // FUSED projection: kvqg = W_kvqg @ x
    cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        4 * n, T * B, d, &alpha,
        W_kvqg, CUDA_R_32F, d, x, CUDA_R_32F, d,
        &beta_zero, kvqg_cache, CUDA_R_32F, 4 * n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    int num_checkpoints = (T + CHECKPOINT_INTERVAL - 1) / CHECKPOINT_INTERVAL + 1;
    float* s_checkpoints = S_cache;
    float* sq_cache = S_cache + num_checkpoints * B * n * n;

    dispatch_e77_linear_forward_fp32(
        T, B, n,
        kvqg_cache, b_gate,
        S, output,
        s_checkpoints, sq_cache,
        decay_cache,
        CHECKPOINT_INTERVAL,
        stream_);
}

template struct E77LinearForward<float>;

// ============================================================================
// Reduction kernel for b_gate gradient
// ============================================================================

__global__ void e77_reduce_b_gate_kernel(const float* accum, __nv_bfloat16* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    output[i] = __float2bfloat16(accum[i]);
}

__global__ void e77_reduce_b_gate_kernel_fp32(const float* accum, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    output[i] = accum[i];
}

// ============================================================================
// E77LinearBackward Implementation
// ============================================================================

template<typename DataT>
E77LinearBackward<DataT>::E77LinearBackward(
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

template<typename DataT>
void E77LinearBackward<DataT>::Run(
    int steps,
    const DataT* W_kvqg,
    const DataT* b_gate,
    const DataT* x,
    const DataT* S_checkpoints,
    const DataT* Sq_cache,
    const DataT* kvqg_cache,
    const DataT* decay_cache,
    const DataT* d_output,
    DataT* dx,
    DataT* dW_kvqg,
    DataT* db_gate,
    DataT* workspace
) {
    int T = steps;
    int B = batch_size_;
    int n = n_state_;
    int d = dim_;

    // Workspace layout:
    // [d_kvqg_all: 4*n * T*B] [d_b_gate_accum: n (float)]
    DataT* d_kvqg_all = workspace;
    float* d_b_gate_accum = reinterpret_cast<float*>(d_kvqg_all + 4 * n * T * B);

    // Zero the accumulator
    cudaMemsetAsync(d_b_gate_accum, 0, n * sizeof(float), stream_);

    // Run backward kernel with fused kvqg layout
    dispatch_e77_linear_backward(
        T, B, n,
        (const __nv_bfloat16*)kvqg_cache,   // Fused [4*n, T*B] column-major
        (const __nv_bfloat16*)b_gate,
        (const __nv_bfloat16*)decay_cache,
        (const __nv_bfloat16*)S_checkpoints,
        (const __nv_bfloat16*)Sq_cache,
        (const __nv_bfloat16*)d_output,
        (__nv_bfloat16*)d_kvqg_all,   // Output: fused [4*n, T*B] column-major
        d_b_gate_accum,
        CHECKPOINT_INTERVAL,
        stream_);

    const float alpha = 1.0f, beta_zero = 0.0f, beta_one = 1.0f;

    // Back-project to dx: dx = W_kvqg.T @ d_kvqg_all
    // W_kvqg: [4*n, d], d_kvqg_all: [4*n, T*B] -> dx: [d, T*B]
    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
        d, T * B, 4 * n, &alpha,
        W_kvqg, CUDA_R_16BF, d, d_kvqg_all, CUDA_R_16BF, 4 * n,
        &beta_zero, dx, CUDA_R_16BF, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // Weight gradient: dW_kvqg += d_kvqg_all @ x.T
    // d_kvqg_all: [4*n, T*B], x: [d, T*B] -> dW_kvqg: [4*n, d]
    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        4 * n, d, T * B, &alpha,
        d_kvqg_all, CUDA_R_16BF, 4 * n, x, CUDA_R_16BF, d,
        &beta_one, dW_kvqg, CUDA_R_16BF, 4 * n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // Reduce b_gate gradient
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    e77_reduce_b_gate_kernel<<<blocks, threads, 0, stream_>>>(
        d_b_gate_accum, (__nv_bfloat16*)db_gate, n);
}

template struct E77LinearBackward<__nv_bfloat16>;

// ============================================================================
// E77LinearBackward - FP32 specialization
// ============================================================================

template<>
void E77LinearBackward<float>::Run(
    int steps,
    const float* W_kvqg,
    const float* b_gate,
    const float* x,
    const float* S_checkpoints,
    const float* Sq_cache,
    const float* kvqg_cache,
    const float* decay_cache,
    const float* d_output,
    float* dx,
    float* dW_kvqg,
    float* db_gate,
    float* workspace
) {
    int T = steps;
    int B = batch_size_;
    int n = n_state_;
    int d = dim_;

    // Workspace layout: [d_kvqg_all: 4*n * T*B] [d_b_gate_accum: n (float)]
    float* d_kvqg_all = workspace;
    float* d_b_gate_accum = d_kvqg_all + 4 * n * T * B;

    cudaMemsetAsync(d_b_gate_accum, 0, n * sizeof(float), stream_);

    dispatch_e77_linear_backward_fp32(
        T, B, n,
        kvqg_cache, b_gate, decay_cache,
        S_checkpoints, Sq_cache, d_output,
        d_kvqg_all, d_b_gate_accum,
        CHECKPOINT_INTERVAL, stream_);

    const float alpha = 1.0f, beta_zero = 0.0f, beta_one = 1.0f;

    // Back-project to dx
    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
        d, T * B, 4 * n, &alpha,
        W_kvqg, CUDA_R_32F, d, d_kvqg_all, CUDA_R_32F, 4 * n,
        &beta_zero, dx, CUDA_R_32F, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // Weight gradient
    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        4 * n, d, T * B, &alpha,
        d_kvqg_all, CUDA_R_32F, 4 * n, x, CUDA_R_32F, d,
        &beta_one, dW_kvqg, CUDA_R_32F, 4 * n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // Reduce b_gate gradient
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    e77_reduce_b_gate_kernel_fp32<<<blocks, threads, 0, stream_>>>(
        d_b_gate_accum, db_gate, n);
}

template struct E77LinearBackward<float>;

}  // namespace elman
