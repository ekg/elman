/**
 * E76 Log-Space Gated Delta Matrix CUDA Kernel
 *
 * E75's nonlinear recurrence + Mamba2/FLA-GDN stability techniques:
 * - Log-space A parameter for decay (A_log)
 * - Inverse softplus dt_bias for update magnitude
 * - Configurable tanh nonlinearity
 *
 * Template parameters:
 *   USE_TANH: Apply tanh to state update (default: true for nonlinear)
 *   LOG_SPACE_GATE: Use log-space A/dt parameterization (default: true)
 *
 * Update equation:
 *   if LOG_SPACE_GATE:
 *     decay = exp(-exp(A_log) * softplus(gate + dt_bias))
 *   else:
 *     decay = sigmoid(gate + b_gate)
 *
 *   if USE_TANH:
 *     S = tanh(decay * S + outer(delta, k_norm))
 *   else:
 *     S = decay * S + outer(delta, k_norm)
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
// Device helper functions
// ============================================================================

__device__ __forceinline__ float softplus_f(float x) {
    // softplus(x) = log(1 + exp(x))
    // For numerical stability, use: x + log(1 + exp(-x)) for x > 0
    if (x > 20.0f) return x;  // Avoid overflow
    return logf(1.0f + expf(x));
}

__device__ __forceinline__ float compute_log_space_decay(float A_val, float gate, float dt_bias) {
    // decay = exp(-A * softplus(gate + dt_bias))
    // A is already exp(A_log), so A_val > 0
    float dt = softplus_f(gate + dt_bias);
    return expf(-A_val * dt);
}

// ============================================================================
// E76 Forward Kernel - Configurable Nonlinearity
// ============================================================================

template<int N_STATE, bool USE_TANH, bool LOG_SPACE_GATE>
__global__ void E76LogSpaceForwardKernel_BF16(
    int T,
    int B,
    const __nv_bfloat16* __restrict__ k_all,      // [T, B, N_STATE]
    const __nv_bfloat16* __restrict__ v_all,      // [T, B, N_STATE]
    const __nv_bfloat16* __restrict__ q_all,      // [T, B, N_STATE]
    const __nv_bfloat16* __restrict__ gate_all,   // [T, B, N_STATE]
    const __nv_bfloat16* __restrict__ A_log,      // [N_STATE] (log-space) or b_gate (sigmoid)
    const __nv_bfloat16* __restrict__ dt_bias,    // [N_STATE] (only for LOG_SPACE_GATE)
    __nv_bfloat16* __restrict__ S,                // [B, N_STATE, N_STATE]
    __nv_bfloat16* __restrict__ output,           // [T, B, N_STATE]
    __nv_bfloat16* __restrict__ S_checkpoints,    // [num_checkpoints, B, N_STATE, N_STATE]
    __nv_bfloat16* __restrict__ Sq_cache,         // [T, B, N_STATE]
    __nv_bfloat16* __restrict__ decay_cache,      // [T, B, N_STATE] (for backward)
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
    float* A_shared = decay_shared + N_STATE;        // [N_STATE]
    float* dt_shared = A_shared + N_STATE;           // [N_STATE]

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;

    // Load A_log and dt_bias to shared memory (constant per batch)
    if (tid < N_STATE) {
        if constexpr (LOG_SPACE_GATE) {
            A_shared[tid] = expf(__bfloat162float(A_log[tid]));  // A = exp(A_log)
            dt_shared[tid] = __bfloat162float(dt_bias[tid]);
        } else {
            // For sigmoid mode, A_log holds b_gate
            A_shared[tid] = __bfloat162float(A_log[tid]);  // This is b_gate
        }
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
        // Load k, v, q, gate for this timestep
        if (tid < N_STATE) {
            k_shared[tid] = __bfloat162float(k_all[t * B * N_STATE + b * N_STATE + tid]);
            v_shared[tid] = __bfloat162float(v_all[t * B * N_STATE + b * N_STATE + tid]);
            q_shared[tid] = __bfloat162float(q_all[t * B * N_STATE + b * N_STATE + tid]);
            gate_shared[tid] = __bfloat162float(gate_all[t * B * N_STATE + b * N_STATE + tid]);
        }
        __syncthreads();

        // Compute decay based on mode
        if (tid < N_STATE) {
            if constexpr (LOG_SPACE_GATE) {
                // Log-space: decay = exp(-A * softplus(gate + dt_bias))
                decay_shared[tid] = compute_log_space_decay(A_shared[tid], gate_shared[tid], dt_shared[tid]);
            } else {
                // Sigmoid: decay = sigmoid(gate + b_gate)
                float val = gate_shared[tid] + A_shared[tid];  // A_shared holds b_gate here
                decay_shared[tid] = 1.0f / (1.0f + expf(-val));
            }
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

        // Update state: S = [tanh](decay * S + outer(delta, k_norm))
        for (int i = tid; i < n2; i += blockDim.x) {
            int row = i / N_STATE;
            int col = i % N_STATE;

            float decay_val = decay_shared[row];
            float delta_i = v_shared[row] - retrieved[row];
            float update = decay_val * S_shared[i] + delta_i * k_shared[col];

            if constexpr (USE_TANH) {
                S_shared[i] = tanhf(update);
            } else {
                S_shared[i] = update;  // Linear mode
            }
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
// E76 Backward Kernel - Configurable Nonlinearity
// ============================================================================

template<int N_STATE, bool USE_TANH, bool LOG_SPACE_GATE>
__global__ void E76LogSpaceBackwardKernel_BF16(
    int T,
    int B,
    const __nv_bfloat16* __restrict__ k_all,
    const __nv_bfloat16* __restrict__ v_all,
    const __nv_bfloat16* __restrict__ q_all,
    const __nv_bfloat16* __restrict__ gate_all,
    const __nv_bfloat16* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ dt_bias,
    const __nv_bfloat16* __restrict__ decay_cache,
    const __nv_bfloat16* __restrict__ S_checkpoints,
    const __nv_bfloat16* __restrict__ Sq_cache,
    const __nv_bfloat16* __restrict__ d_output,
    __nv_bfloat16* __restrict__ d_k_all,
    __nv_bfloat16* __restrict__ d_v_all,
    __nv_bfloat16* __restrict__ d_q_all,
    __nv_bfloat16* __restrict__ d_gate_all,
    float* __restrict__ d_A_log_accum,     // [N_STATE] accumulated across batches
    float* __restrict__ d_dt_bias_accum,   // [N_STATE] accumulated across batches
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
    float* A_val = decay + N_STATE;                   // [N_STATE]
    float* dt_val = A_val + N_STATE;                  // [N_STATE]
    float* d_k_raw = dt_val + N_STATE;                // [N_STATE]
    float* d_v_raw = d_k_raw + N_STATE;               // [N_STATE]
    float* d_q_raw = d_v_raw + N_STATE;               // [N_STATE]
    float* d_Sq_shared = d_q_raw + N_STATE;           // [N_STATE]
    float* d_delta = d_Sq_shared + N_STATE;           // [N_STATE]
    float* d_k_norm = d_delta + N_STATE;              // [N_STATE]
    float* d_decay = d_k_norm + N_STATE;              // [N_STATE]
    float* d_A_local = d_decay + N_STATE;             // [N_STATE]
    float* d_dt_local = d_A_local + N_STATE;          // [N_STATE]

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;

    // Load A and dt values
    if (tid < N_STATE) {
        if constexpr (LOG_SPACE_GATE) {
            A_val[tid] = expf(__bfloat162float(A_log[tid]));
            dt_val[tid] = __bfloat162float(dt_bias[tid]);
        } else {
            A_val[tid] = __bfloat162float(A_log[tid]);  // b_gate
        }
        d_A_local[tid] = 0.0f;
        d_dt_local[tid] = 0.0f;
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
                if (tid < N_STATE) {
                    k_raw[tid] = __bfloat162float(k_all[tt * B * N_STATE + b * N_STATE + tid]);
                    v_raw[tid] = __bfloat162float(v_all[tt * B * N_STATE + b * N_STATE + tid]);
                    q_raw[tid] = __bfloat162float(q_all[tt * B * N_STATE + b * N_STATE + tid]);
                    gate[tid] = __bfloat162float(gate_all[tt * B * N_STATE + b * N_STATE + tid]);
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

                // Update S if not at target step
                if (tt < t) {
                    for (int i = tid; i < n2; i += blockDim.x) {
                        int row = i / N_STATE;
                        int col = i % N_STATE;
                        float update = decay[row] * S[i] + delta[row] * k_norm[col];
                        if constexpr (USE_TANH) {
                            S[i] = tanhf(update);
                        } else {
                            S[i] = update;
                        }
                    }
                    __syncthreads();
                }
            }

            // Now S holds S_{t-1}, k_norm/v_raw/q_raw/delta/decay are for step t

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
                    float update = decay[i] * S[i * N_STATE + tid] + delta[i] * k_norm[tid];
                    float S_t_ij;
                    if constexpr (USE_TANH) {
                        S_t_ij = tanhf(update);
                    } else {
                        S_t_ij = update;
                    }
                    sum += S_t_ij * d_Sq_shared[i];
                }
                d_q_raw[tid] = sum;
            }
            __syncthreads();

            // Backward through state update
            if (tid < N_STATE) {
                float d_delta_local = 0.0f;
                float d_decay_local = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    float update = decay[tid] * S[tid * N_STATE + j] + delta[tid] * k_norm[j];
                    float d_pre;
                    if constexpr (USE_TANH) {
                        float S_t_ij = tanhf(update);
                        d_pre = dS[tid * N_STATE + j] * (1.0f - S_t_ij * S_t_ij);
                    } else {
                        d_pre = dS[tid * N_STATE + j];  // Linear: gradient passes through
                    }
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
                    float update = decay[i] * S[i * N_STATE + tid] + delta[i] * k_norm[tid];
                    float d_pre;
                    if constexpr (USE_TANH) {
                        float S_t_ij = tanhf(update);
                        d_pre = dS[i * N_STATE + tid] * (1.0f - S_t_ij * S_t_ij);
                    } else {
                        d_pre = dS[i * N_STATE + tid];
                    }
                    d_k_norm_local += d_pre * delta[i];
                }
                d_k_norm[tid] = d_k_norm_local;
            }
            __syncthreads();

            // Backward through decay computation
            if (tid < N_STATE) {
                float d_dec = d_decay[tid];
                if constexpr (LOG_SPACE_GATE) {
                    // decay = exp(-A * softplus(gate + dt_bias))
                    // d_decay/d_gate = decay * (-A) * sigmoid(gate + dt_bias)
                    // d_decay/d_A = decay * (-softplus(gate + dt_bias))
                    // d_decay/d_dt_bias = decay * (-A) * sigmoid(gate + dt_bias)
                    float x = gate[tid] + dt_val[tid];
                    float sp = softplus_f(x);
                    float sig_x = 1.0f / (1.0f + expf(-x));  // derivative of softplus
                    float dec_val = decay[tid];
                    float A = A_val[tid];

                    float d_gate_val = d_dec * dec_val * (-A) * sig_x;
                    d_gate_all[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(d_gate_val);

                    // Accumulate gradients for A_log and dt_bias
                    // d_loss/d_A_log = d_loss/d_A * d_A/d_A_log = d_loss/d_A * A
                    float d_A = d_dec * dec_val * (-sp);
                    d_A_local[tid] += d_A * A;  // Chain rule: d/d_A_log = d/dA * A

                    float d_dt = d_dec * dec_val * (-A) * sig_x;
                    d_dt_local[tid] += d_dt;
                } else {
                    // decay = sigmoid(gate + b_gate)
                    // d_decay/d_gate = decay * (1 - decay)
                    float d_gate_val = d_dec * decay[tid] * (1.0f - decay[tid]);
                    d_gate_all[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(d_gate_val);

                    // d_b_gate same as d_gate
                    d_A_local[tid] += d_gate_val;  // A_log holds b_gate in sigmoid mode
                }
            }
            __syncthreads();

            // d_v = d_delta
            if (tid < N_STATE) {
                d_v_raw[tid] = d_delta[tid];
            }

            // d_k_norm += S^T @ (-d_delta)
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
            if (tid < N_STATE) {
                d_k_all[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(d_k_raw[tid]);
                d_v_all[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(d_v_raw[tid]);
                d_q_all[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(d_q_raw[tid]);
            }
            __syncthreads();

            // Update dS for next iteration
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                float update = decay[row] * S[i] + delta[row] * k_norm[col];
                float d_pre;
                if constexpr (USE_TANH) {
                    float S_t_ij = tanhf(update);
                    d_pre = dS[i] * (1.0f - S_t_ij * S_t_ij);
                } else {
                    d_pre = dS[i];
                }
                dS[i] = d_pre * decay[row] + (-d_delta[row]) * k_norm[col];
            }
            __syncthreads();
        }
    }

    // Atomically accumulate A_log and dt_bias gradients
    if (tid < N_STATE) {
        atomicAdd(&d_A_log_accum[tid], d_A_local[tid]);
        if constexpr (LOG_SPACE_GATE) {
            atomicAdd(&d_dt_bias_accum[tid], d_dt_local[tid]);
        }
    }
}

// ============================================================================
// Template instantiations for different N_STATE values and modes
// ============================================================================

#define INSTANTIATE_E76_KERNELS(N) \
    template __global__ void E76LogSpaceForwardKernel_BF16<N, true, true>( \
        int, int, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, \
        const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, \
        __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int); \
    template __global__ void E76LogSpaceForwardKernel_BF16<N, true, false>( \
        int, int, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, \
        const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, \
        __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int); \
    template __global__ void E76LogSpaceForwardKernel_BF16<N, false, true>( \
        int, int, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, \
        const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, \
        __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int); \
    template __global__ void E76LogSpaceForwardKernel_BF16<N, false, false>( \
        int, int, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, \
        const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, \
        __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int); \
    template __global__ void E76LogSpaceBackwardKernel_BF16<N, true, true>( \
        int, int, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, \
        const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, \
        const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, \
        __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, float*, float*, int); \
    template __global__ void E76LogSpaceBackwardKernel_BF16<N, true, false>( \
        int, int, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, \
        const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, \
        const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, \
        __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, float*, float*, int); \
    template __global__ void E76LogSpaceBackwardKernel_BF16<N, false, true>( \
        int, int, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, \
        const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, \
        const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, \
        __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, float*, float*, int); \
    template __global__ void E76LogSpaceBackwardKernel_BF16<N, false, false>( \
        int, int, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, \
        const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, \
        const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, \
        __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, float*, float*, int);

INSTANTIATE_E76_KERNELS(32)
INSTANTIATE_E76_KERNELS(48)
INSTANTIATE_E76_KERNELS(64)
INSTANTIATE_E76_KERNELS(96)

// ============================================================================
// Dispatcher function to select kernel based on runtime parameters
// ============================================================================

void dispatch_e76_forward(
    int T, int B, int n_state,
    bool use_tanh, bool log_space_gate,
    const __nv_bfloat16* k_all, const __nv_bfloat16* v_all,
    const __nv_bfloat16* q_all, const __nv_bfloat16* gate_all,
    const __nv_bfloat16* A_log, const __nv_bfloat16* dt_bias,
    __nv_bfloat16* S, __nv_bfloat16* output,
    __nv_bfloat16* S_checkpoints, __nv_bfloat16* Sq_cache,
    __nv_bfloat16* decay_cache, int checkpoint_interval,
    cudaStream_t stream
) {
    int shared_size = (n_state * n_state + 9 * n_state) * sizeof(float);

    #define DISPATCH_E76_FWD(N) \
        if (use_tanh && log_space_gate) { \
            E76LogSpaceForwardKernel_BF16<N, true, true><<<B, 256, shared_size, stream>>>( \
                T, B, k_all, v_all, q_all, gate_all, A_log, dt_bias, \
                S, output, S_checkpoints, Sq_cache, decay_cache, checkpoint_interval); \
        } else if (use_tanh && !log_space_gate) { \
            E76LogSpaceForwardKernel_BF16<N, true, false><<<B, 256, shared_size, stream>>>( \
                T, B, k_all, v_all, q_all, gate_all, A_log, dt_bias, \
                S, output, S_checkpoints, Sq_cache, decay_cache, checkpoint_interval); \
        } else if (!use_tanh && log_space_gate) { \
            E76LogSpaceForwardKernel_BF16<N, false, true><<<B, 256, shared_size, stream>>>( \
                T, B, k_all, v_all, q_all, gate_all, A_log, dt_bias, \
                S, output, S_checkpoints, Sq_cache, decay_cache, checkpoint_interval); \
        } else { \
            E76LogSpaceForwardKernel_BF16<N, false, false><<<B, 256, shared_size, stream>>>( \
                T, B, k_all, v_all, q_all, gate_all, A_log, dt_bias, \
                S, output, S_checkpoints, Sq_cache, decay_cache, checkpoint_interval); \
        }

    switch (n_state) {
        case 32: DISPATCH_E76_FWD(32); break;
        case 48: DISPATCH_E76_FWD(48); break;
        case 64: DISPATCH_E76_FWD(64); break;
        case 96: DISPATCH_E76_FWD(96); break;
        default:
            fprintf(stderr, "E76: Unsupported n_state=%d (use 32, 48, 64, or 96)\n", n_state);
    }
    #undef DISPATCH_E76_FWD
}

void dispatch_e76_backward(
    int T, int B, int n_state,
    bool use_tanh, bool log_space_gate,
    const __nv_bfloat16* k_all, const __nv_bfloat16* v_all,
    const __nv_bfloat16* q_all, const __nv_bfloat16* gate_all,
    const __nv_bfloat16* A_log, const __nv_bfloat16* dt_bias,
    const __nv_bfloat16* decay_cache, const __nv_bfloat16* S_checkpoints,
    const __nv_bfloat16* Sq_cache, const __nv_bfloat16* d_output,
    __nv_bfloat16* d_k_all, __nv_bfloat16* d_v_all,
    __nv_bfloat16* d_q_all, __nv_bfloat16* d_gate_all,
    float* d_A_log_accum, float* d_dt_bias_accum,
    int checkpoint_interval, cudaStream_t stream
) {
    // More shared memory for backward
    int shared_size = (2 * n_state * n_state + 20 * n_state) * sizeof(float);

    #define DISPATCH_E76_BWD(N) \
        if (use_tanh && log_space_gate) { \
            E76LogSpaceBackwardKernel_BF16<N, true, true><<<B, 256, shared_size, stream>>>( \
                T, B, k_all, v_all, q_all, gate_all, A_log, dt_bias, \
                decay_cache, S_checkpoints, Sq_cache, d_output, \
                d_k_all, d_v_all, d_q_all, d_gate_all, \
                d_A_log_accum, d_dt_bias_accum, checkpoint_interval); \
        } else if (use_tanh && !log_space_gate) { \
            E76LogSpaceBackwardKernel_BF16<N, true, false><<<B, 256, shared_size, stream>>>( \
                T, B, k_all, v_all, q_all, gate_all, A_log, dt_bias, \
                decay_cache, S_checkpoints, Sq_cache, d_output, \
                d_k_all, d_v_all, d_q_all, d_gate_all, \
                d_A_log_accum, d_dt_bias_accum, checkpoint_interval); \
        } else if (!use_tanh && log_space_gate) { \
            E76LogSpaceBackwardKernel_BF16<N, false, true><<<B, 256, shared_size, stream>>>( \
                T, B, k_all, v_all, q_all, gate_all, A_log, dt_bias, \
                decay_cache, S_checkpoints, Sq_cache, d_output, \
                d_k_all, d_v_all, d_q_all, d_gate_all, \
                d_A_log_accum, d_dt_bias_accum, checkpoint_interval); \
        } else { \
            E76LogSpaceBackwardKernel_BF16<N, false, false><<<B, 256, shared_size, stream>>>( \
                T, B, k_all, v_all, q_all, gate_all, A_log, dt_bias, \
                decay_cache, S_checkpoints, Sq_cache, d_output, \
                d_k_all, d_v_all, d_q_all, d_gate_all, \
                d_A_log_accum, d_dt_bias_accum, checkpoint_interval); \
        }

    switch (n_state) {
        case 32: DISPATCH_E76_BWD(32); break;
        case 48: DISPATCH_E76_BWD(48); break;
        case 64: DISPATCH_E76_BWD(64); break;
        case 96: DISPATCH_E76_BWD(96); break;
        default:
            fprintf(stderr, "E76: Unsupported n_state=%d (use 32, 48, 64, or 96)\n", n_state);
    }
    #undef DISPATCH_E76_BWD
}

// ============================================================================
// E76LogSpaceForward Implementation
// ============================================================================

template<typename DataT>
E76LogSpaceForward<DataT>::E76LogSpaceForward(
    bool training,
    int batch_size,
    int n_state,
    int dim,
    bool use_tanh,
    bool log_space_gate,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : training_(training),
      batch_size_(batch_size),
      n_state_(n_state),
      dim_(dim),
      use_tanh_(use_tanh),
      log_space_gate_(log_space_gate),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename DataT>
void E76LogSpaceForward<DataT>::Run(
    int steps,
    const DataT* W_k,
    const DataT* W_v,
    const DataT* W_q,
    const DataT* W_gate,
    const DataT* A_log,
    const DataT* dt_bias,
    const DataT* x,
    DataT* S,
    DataT* output,
    DataT* k_cache,
    DataT* v_cache,
    DataT* q_cache,
    DataT* gate_cache,
    DataT* S_cache,
    DataT* decay_cache
) {
    int T = steps;
    int B = batch_size_;
    int n = n_state_;
    int d = dim_;

    const float alpha = 1.0f, beta_zero = 0.0f;

    // Project k, v, q, gate using cuBLAS
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

    cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        n, T * B, d, &alpha,
        W_gate, CUDA_R_16BF, d, x, CUDA_R_16BF, d,
        &beta_zero, gate_cache, CUDA_R_16BF, n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // Calculate workspace offsets for checkpoints
    int num_checkpoints = (T + CHECKPOINT_INTERVAL - 1) / CHECKPOINT_INTERVAL + 1;
    DataT* s_checkpoints = S_cache;
    DataT* sq_cache = S_cache + num_checkpoints * B * n * n;

    // Run forward kernel via dispatcher
    dispatch_e76_forward(
        T, B, n, use_tanh_, log_space_gate_,
        (const __nv_bfloat16*)k_cache,
        (const __nv_bfloat16*)v_cache,
        (const __nv_bfloat16*)q_cache,
        (const __nv_bfloat16*)gate_cache,
        (const __nv_bfloat16*)A_log,
        (const __nv_bfloat16*)dt_bias,
        (__nv_bfloat16*)S,
        (__nv_bfloat16*)output,
        (__nv_bfloat16*)s_checkpoints,
        (__nv_bfloat16*)sq_cache,
        (__nv_bfloat16*)decay_cache,
        CHECKPOINT_INTERVAL,
        stream_);
}

template struct E76LogSpaceForward<__nv_bfloat16>;

// ============================================================================
// Reduction kernel for parameter gradients (used by backward pass)
// ============================================================================

__global__ void reduce_accum_kernel(const float* accum, __nv_bfloat16* output, int B, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float sum = 0.0f;
    for (int b = 0; b < B; b++) {
        sum += accum[b * n + i];
    }
    output[i] = __float2bfloat16(sum);
}

void reduce_bf16_gradients(const float* accum, __nv_bfloat16* output, int B, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    reduce_accum_kernel<<<blocks, threads, 0, stream>>>(accum, output, B, n);
}

// ============================================================================
// E76LogSpaceBackward Implementation
// ============================================================================

template<typename DataT>
E76LogSpaceBackward<DataT>::E76LogSpaceBackward(
    int batch_size,
    int n_state,
    int dim,
    bool use_tanh,
    bool log_space_gate,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      n_state_(n_state),
      dim_(dim),
      use_tanh_(use_tanh),
      log_space_gate_(log_space_gate),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename DataT>
void E76LogSpaceBackward<DataT>::Run(
    int steps,
    const DataT* W_k,
    const DataT* W_v,
    const DataT* W_q,
    const DataT* W_gate,
    const DataT* A_log,
    const DataT* dt_bias,
    const DataT* x,
    const DataT* S_checkpoints,
    const DataT* Sq_cache,
    const DataT* k_cache,
    const DataT* v_cache,
    const DataT* q_cache,
    const DataT* gate_cache,
    const DataT* decay_cache,
    const DataT* d_output,
    DataT* dx,
    DataT* dW_k,
    DataT* dW_v,
    DataT* dW_q,
    DataT* dW_gate,
    DataT* dA_log,
    DataT* ddt_bias,
    DataT* workspace
) {
    int T = steps;
    int B = batch_size_;
    int n = n_state_;
    int d = dim_;

    // Workspace layout: [d_k_all: T*B*n] [d_v_all: T*B*n] [d_q_all: T*B*n] [d_gate_all: T*B*n] [d_A_log_accum: B*n] [d_dt_bias_accum: B*n]
    DataT* d_k_all = workspace;
    DataT* d_v_all = d_k_all + T * B * n;
    DataT* d_q_all = d_v_all + T * B * n;
    DataT* d_gate_all = d_q_all + T * B * n;
    float* d_A_log_accum = reinterpret_cast<float*>(d_gate_all + T * B * n);
    float* d_dt_bias_accum = d_A_log_accum + B * n;

    // Zero the accumulators
    cudaMemsetAsync(d_A_log_accum, 0, B * n * sizeof(float), stream_);
    cudaMemsetAsync(d_dt_bias_accum, 0, B * n * sizeof(float), stream_);

    // Run backward kernel via dispatcher
    dispatch_e76_backward(
        T, B, n, use_tanh_, log_space_gate_,
        (const __nv_bfloat16*)k_cache,
        (const __nv_bfloat16*)v_cache,
        (const __nv_bfloat16*)q_cache,
        (const __nv_bfloat16*)gate_cache,
        (const __nv_bfloat16*)A_log,
        (const __nv_bfloat16*)dt_bias,
        (const __nv_bfloat16*)decay_cache,
        (const __nv_bfloat16*)S_checkpoints,
        (const __nv_bfloat16*)Sq_cache,
        (const __nv_bfloat16*)d_output,
        (__nv_bfloat16*)d_k_all,
        (__nv_bfloat16*)d_v_all,
        (__nv_bfloat16*)d_q_all,
        (__nv_bfloat16*)d_gate_all,
        d_A_log_accum,
        d_dt_bias_accum,
        CHECKPOINT_INTERVAL,
        stream_);

    const float alpha = 1.0f, beta_zero = 0.0f, beta_one = 1.0f;

    // Back-project to dx: dx = W_k.T @ d_k + W_v.T @ d_v + W_q.T @ d_q + W_gate.T @ d_gate
    // dx = W_k @ d_k_all (GEMM: [d, n] @ [n, T*B] = [d, T*B])
    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
        d, T * B, n, &alpha,
        W_k, CUDA_R_16BF, d, d_k_all, CUDA_R_16BF, n,
        &beta_zero, dx, CUDA_R_16BF, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
        d, T * B, n, &alpha,
        W_v, CUDA_R_16BF, d, d_v_all, CUDA_R_16BF, n,
        &beta_one, dx, CUDA_R_16BF, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
        d, T * B, n, &alpha,
        W_q, CUDA_R_16BF, d, d_q_all, CUDA_R_16BF, n,
        &beta_one, dx, CUDA_R_16BF, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
        d, T * B, n, &alpha,
        W_gate, CUDA_R_16BF, d, d_gate_all, CUDA_R_16BF, n,
        &beta_one, dx, CUDA_R_16BF, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // Weight gradients: dW = d_proj @ x.T (GEMM: [n, T*B] @ [T*B, d] = [n, d])
    // dW_k += d_k_all @ x.T
    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        n, d, T * B, &alpha,
        d_k_all, CUDA_R_16BF, n, x, CUDA_R_16BF, d,
        &beta_one, dW_k, CUDA_R_16BF, n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        n, d, T * B, &alpha,
        d_v_all, CUDA_R_16BF, n, x, CUDA_R_16BF, d,
        &beta_one, dW_v, CUDA_R_16BF, n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        n, d, T * B, &alpha,
        d_q_all, CUDA_R_16BF, n, x, CUDA_R_16BF, d,
        &beta_one, dW_q, CUDA_R_16BF, n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        n, d, T * B, &alpha,
        d_gate_all, CUDA_R_16BF, n, x, CUDA_R_16BF, d,
        &beta_one, dW_gate, CUDA_R_16BF, n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // Reduce A_log and dt_bias gradients across batch
    // (The kernel accumulates per-batch, need to sum to get per-param gradients)
    // Simple reduction: sum d_A_log_accum[B, n] -> dA_log[n]
    reduce_bf16_gradients(d_A_log_accum, (__nv_bfloat16*)dA_log, B, n, stream_);
    reduce_bf16_gradients(d_dt_bias_accum, (__nv_bfloat16*)ddt_bias, B, n, stream_);
}

template struct E76LogSpaceBackward<__nv_bfloat16>;

}  // namespace elman
