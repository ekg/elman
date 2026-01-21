/**
 * E88 FLA Hybrid CUDA Kernel
 *
 * Combines:
 * 1. Mamba2-style exponential decay (passed as decay tensor)
 * 2. Nonlinear matrix state: S = tanh(decay * S + outer(delta, k_norm))
 * 3. L2-normalized k and q (expected to be normalized externally)
 * 4. Rectangular state [n_state x head_v_dim] for value expansion
 *
 * Per head h:
 *   k_h = L2_normalized(W_k_h @ x)  [n_state] - normalized externally
 *   v_h = W_v_h @ x                 [head_v_dim]
 *   q_h = L2_normalized(W_q_h @ x)  [n_state] - normalized externally
 *   decay_h = exp(-softplus(W_decay_h @ x))  [scalar per head]
 *
 *   retrieved = S_h @ k_h           [head_v_dim]
 *   delta = v_h - retrieved         [head_v_dim]
 *   S_h = tanh(decay_h * S_h + outer(delta, k_h))  [n_state x head_v_dim]
 *   out_h = S_h^T @ q_h             [head_v_dim]
 *   (gating applied externally via g_proj in Python layer)
 *
 * Output: concat(out_0, out_1, ..., out_{H-1})  [T, B, H * head_v_dim]
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include "hasty/elman_ladder.h"

#define E88_CHECKPOINT_INTERVAL 16

namespace elman {

// ============================================================================
// E88 FLA Hybrid Forward Kernel
// Each block handles one (batch, head) pair
// ============================================================================

template<int N_STATE, int HEAD_V_DIM>
__global__ void E88FLAHybridForwardKernel_BF16(
    int T,
    int B,
    int H,
    const __nv_bfloat16* __restrict__ k_all,      // [T, B, H, N_STATE]
    const __nv_bfloat16* __restrict__ v_all,      // [T, B, H, HEAD_V_DIM]
    const __nv_bfloat16* __restrict__ q_all,      // [T, B, H, N_STATE]
    const __nv_bfloat16* __restrict__ decay_all,  // [T, B, H]
    __nv_bfloat16* __restrict__ S,                // [B, H, N_STATE, HEAD_V_DIM]
    __nv_bfloat16* __restrict__ output,           // [T, B, H, HEAD_V_DIM]
    __nv_bfloat16* __restrict__ S_checkpoints,    // [num_checkpoints, B, H, N_STATE, HEAD_V_DIM]
    __nv_bfloat16* __restrict__ Sq_cache,         // [T, B, H, HEAD_V_DIM]
    int checkpoint_interval
) {
    int block_idx = blockIdx.x;
    int b = block_idx / H;
    int h = block_idx % H;
    if (b >= B) return;

    extern __shared__ float shared_mem[];
    // Shared memory layout:
    // S_shared: N_STATE * HEAD_V_DIM
    // k_shared: N_STATE
    // v_shared: HEAD_V_DIM
    // q_shared: N_STATE
    // retrieved: HEAD_V_DIM
    float* S_shared = shared_mem;
    float* k_shared = S_shared + N_STATE * HEAD_V_DIM;
    float* v_shared = k_shared + N_STATE;
    float* q_shared = v_shared + HEAD_V_DIM;
    float* retrieved = q_shared + N_STATE;

    int tid = threadIdx.x;
    int state_size = N_STATE * HEAD_V_DIM;

    // State offset for this (batch, head)
    int state_offset = (b * H + h) * state_size;

    // Load initial state for this head
    for (int i = tid; i < state_size; i += blockDim.x) {
        S_shared[i] = __bfloat162float(S[state_offset + i]);
    }
    __syncthreads();

    // Save initial checkpoint (index 0)
    for (int i = tid; i < state_size; i += blockDim.x) {
        S_checkpoints[(b * H + h) * state_size + i] = __float2bfloat16(S_shared[i]);
    }
    __syncthreads();

    for (int t = 0; t < T; t++) {
        // Load k, v, q for this timestep
        int k_offset = ((t * B + b) * H + h) * N_STATE;
        int v_offset = ((t * B + b) * H + h) * HEAD_V_DIM;
        int decay_offset = (t * B + b) * H + h;

        if (tid < N_STATE) {
            k_shared[tid] = __bfloat162float(k_all[k_offset + tid]);
            q_shared[tid] = __bfloat162float(q_all[k_offset + tid]);
        }
        if (tid < HEAD_V_DIM) {
            v_shared[tid] = __bfloat162float(v_all[v_offset + tid]);
        }
        __syncthreads();

        // Load decay (single scalar per head per timestep)
        __shared__ float decay_val;
        if (tid == 0) {
            decay_val = __bfloat162float(decay_all[decay_offset]);
        }
        __syncthreads();

        // retrieved = S @ k (S is [N_STATE, HEAD_V_DIM], k is [N_STATE])
        // retrieved[j] = sum_i S[i,j] * k[i]
        if (tid < HEAD_V_DIM) {
            float sum = 0.0f;
            #pragma unroll 8
            for (int i = 0; i < N_STATE; i++) {
                sum += S_shared[i * HEAD_V_DIM + tid] * k_shared[i];
            }
            retrieved[tid] = sum;
        }
        __syncthreads();

        // S = tanh(decay * S + outer(delta, k))
        // delta = v - retrieved
        // outer(delta, k)[i,j] = delta[j] * k[i]
        // S[i,j] = tanh(decay * S[i,j] + (v[j] - retrieved[j]) * k[i])
        for (int idx = tid; idx < state_size; idx += blockDim.x) {
            int i = idx / HEAD_V_DIM;  // row index (k dimension)
            int j = idx % HEAD_V_DIM;  // col index (v dimension)
            float delta_j = v_shared[j] - retrieved[j];
            float update = decay_val * S_shared[idx] + delta_j * k_shared[i];
            S_shared[idx] = tanhf(update);
        }
        __syncthreads();

        // Save checkpoint
        if ((t + 1) % checkpoint_interval == 0) {
            int cp_idx = (t + 1) / checkpoint_interval;
            int cp_offset = (cp_idx * B * H + b * H + h) * state_size;
            for (int i = tid; i < state_size; i += blockDim.x) {
                S_checkpoints[cp_offset + i] = __float2bfloat16(S_shared[i]);
            }
        }
        __syncthreads();

        // Compute output: Sq = S^T @ q (FLA-GDN style - no self-gating)
        // Sq[j] = sum_i S[i,j] * q[i]
        if (tid < HEAD_V_DIM) {
            float Sq = 0.0f;
            #pragma unroll 8
            for (int i = 0; i < N_STATE; i++) {
                Sq += S_shared[i * HEAD_V_DIM + tid] * q_shared[i];
            }
            Sq_cache[v_offset + tid] = __float2bfloat16(Sq);

            // Output directly (gating done in Python layer via g_proj)
            output[v_offset + tid] = __float2bfloat16(Sq);
        }
        __syncthreads();
    }

    // Write final state back
    for (int i = tid; i < state_size; i += blockDim.x) {
        S[state_offset + i] = __float2bfloat16(S_shared[i]);
    }
}

// ============================================================================
// E88 FLA Hybrid Backward Kernel
// ============================================================================

template<int N_STATE, int HEAD_V_DIM>
__global__ void E88FLAHybridBackwardKernel_BF16(
    int T,
    int B,
    int H,
    const __nv_bfloat16* __restrict__ k_all,
    const __nv_bfloat16* __restrict__ v_all,
    const __nv_bfloat16* __restrict__ q_all,
    const __nv_bfloat16* __restrict__ decay_all,
    const __nv_bfloat16* __restrict__ S_checkpoints,
    const __nv_bfloat16* __restrict__ Sq_cache,
    const __nv_bfloat16* __restrict__ d_output,
    __nv_bfloat16* __restrict__ d_k_all,
    __nv_bfloat16* __restrict__ d_v_all,
    __nv_bfloat16* __restrict__ d_q_all,
    __nv_bfloat16* __restrict__ d_decay_all,
    int checkpoint_interval
) {
    int block_idx = blockIdx.x;
    int b = block_idx / H;
    int h = block_idx % H;
    if (b >= B) return;

    extern __shared__ float shared_mem[];
    // Memory layout:
    // S: N_STATE * HEAD_V_DIM
    // dS: N_STATE * HEAD_V_DIM
    // k: N_STATE
    // v: HEAD_V_DIM
    // q: N_STATE
    // delta: HEAD_V_DIM
    // retrieved: HEAD_V_DIM
    // d_k: N_STATE
    // d_v: HEAD_V_DIM
    // d_q: N_STATE
    // d_Sq: HEAD_V_DIM
    // d_delta: HEAD_V_DIM
    float* S = shared_mem;
    float* dS = S + N_STATE * HEAD_V_DIM;
    float* k = dS + N_STATE * HEAD_V_DIM;
    float* v = k + N_STATE;
    float* q = v + HEAD_V_DIM;
    float* delta = q + N_STATE;
    float* retrieved = delta + HEAD_V_DIM;
    float* d_k = retrieved + HEAD_V_DIM;
    float* d_v = d_k + N_STATE;
    float* d_q = d_v + HEAD_V_DIM;
    float* d_Sq = d_q + N_STATE;
    float* d_delta = d_Sq + HEAD_V_DIM;

    int tid = threadIdx.x;
    int state_size = N_STATE * HEAD_V_DIM;

    // Initialize dS to zero
    for (int i = tid; i < state_size; i += blockDim.x) {
        dS[i] = 0.0f;
    }
    __syncthreads();

    int num_segments = (T + checkpoint_interval - 1) / checkpoint_interval;

    for (int seg = num_segments - 1; seg >= 0; seg--) {
        int t_start = seg * checkpoint_interval;
        int t_end = min(t_start + checkpoint_interval, T);

        for (int t = t_end - 1; t >= t_start; t--) {
            // Reload checkpoint for this segment
            int cp_offset = (seg * B * H + b * H + h) * state_size;
            for (int i = tid; i < state_size; i += blockDim.x) {
                S[i] = __bfloat162float(S_checkpoints[cp_offset + i]);
            }
            __syncthreads();

            // Replay forward from checkpoint to timestep t
            __shared__ float decay_val_replay;
            for (int tt = t_start; tt <= t; tt++) {
                int k_offset = ((tt * B + b) * H + h) * N_STATE;
                int v_offset = ((tt * B + b) * H + h) * HEAD_V_DIM;
                int decay_offset = (tt * B + b) * H + h;

                if (tid < N_STATE) {
                    k[tid] = __bfloat162float(k_all[k_offset + tid]);
                    q[tid] = __bfloat162float(q_all[k_offset + tid]);
                }
                if (tid < HEAD_V_DIM) {
                    v[tid] = __bfloat162float(v_all[v_offset + tid]);
                }
                __syncthreads();

                if (tid == 0) {
                    decay_val_replay = __bfloat162float(decay_all[decay_offset]);
                }
                __syncthreads();

                // Compute retrieved = S @ k
                if (tid < HEAD_V_DIM) {
                    float sum = 0.0f;
                    for (int i = 0; i < N_STATE; i++) {
                        sum += S[i * HEAD_V_DIM + tid] * k[i];
                    }
                    retrieved[tid] = sum;
                    delta[tid] = v[tid] - retrieved[tid];
                }
                __syncthreads();

                // Update S if not yet at target timestep t
                if (tt < t) {
                    for (int idx = tid; idx < state_size; idx += blockDim.x) {
                        int i = idx / HEAD_V_DIM;
                        int j = idx % HEAD_V_DIM;
                        float update = decay_val_replay * S[idx] + delta[j] * k[i];
                        S[idx] = tanhf(update);
                    }
                    __syncthreads();
                }
            }

            // Now S contains S_{t-1} after update (or checkpoint S if t == t_start)
            // We have k[t], v[t], q[t], delta[t], retrieved[t], decay[t] in shared memory

            // Load decay for timestep t
            __shared__ float decay_val;
            int decay_offset = (t * B + b) * H + h;
            if (tid == 0) {
                decay_val = __bfloat162float(decay_all[decay_offset]);
            }
            __syncthreads();

            // Backward through output (no self-gating - FLA-GDN style)
            int v_offset = ((t * B + b) * H + h) * HEAD_V_DIM;
            if (tid < HEAD_V_DIM) {
                float d_out = __bfloat162float(d_output[v_offset + tid]);
                // output = Sq directly (no self-gating)
                // d_Sq = d_output directly
                d_Sq[tid] = d_out;
            }
            __syncthreads();

            // d_q from output: Sq[j] = sum_i S_t[i,j] * q[i]
            // d_q[i] = sum_j S_t[i,j] * d_Sq[j]
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int j = 0; j < HEAD_V_DIM; j++) {
                    // S_t[i,j] after tanh update
                    float S_t_ij = tanhf(decay_val * S[tid * HEAD_V_DIM + j] + delta[j] * k[tid]);
                    sum += S_t_ij * d_Sq[j];
                }
                d_q[tid] = sum;
            }
            __syncthreads();

            // dS += outer(q, d_Sq) for the output computation
            // dS[i,j] += q[i] * d_Sq[j]
            for (int idx = tid; idx < state_size; idx += blockDim.x) {
                int i = idx / HEAD_V_DIM;
                int j = idx % HEAD_V_DIM;
                dS[idx] += q[i] * d_Sq[j];
            }
            __syncthreads();

            // Backward through state update: S_t = tanh(decay * S_{t-1} + outer(delta, k))
            // Let pre = decay * S_{t-1} + delta * k^T
            // S_t = tanh(pre)
            // dS_t flows back through tanh: d_pre = dS * (1 - S_t^2)
            // d_delta[j] = sum_i d_pre[i,j] * k[i]
            // d_k[i] = sum_j d_pre[i,j] * delta[j]
            // d_decay = sum_{i,j} d_pre[i,j] * S_{t-1}[i,j]
            // dS_{t-1} = d_pre * decay

            __shared__ float d_decay_accum;
            if (tid == 0) d_decay_accum = 0.0f;
            __syncthreads();

            // Compute d_delta and d_k contributions
            if (tid < HEAD_V_DIM) {
                float d_delta_local = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    float S_t_ij = tanhf(decay_val * S[i * HEAD_V_DIM + tid] + delta[tid] * k[i]);
                    float d_pre = dS[i * HEAD_V_DIM + tid] * (1.0f - S_t_ij * S_t_ij);
                    d_delta_local += d_pre * k[i];
                }
                d_delta[tid] = d_delta_local;
            }
            __syncthreads();

            if (tid < N_STATE) {
                float d_k_local = 0.0f;
                for (int j = 0; j < HEAD_V_DIM; j++) {
                    float S_t_ij = tanhf(decay_val * S[tid * HEAD_V_DIM + j] + delta[j] * k[tid]);
                    float d_pre = dS[tid * HEAD_V_DIM + j] * (1.0f - S_t_ij * S_t_ij);
                    d_k_local += d_pre * delta[j];
                }
                d_k[tid] = d_k_local;
            }
            __syncthreads();

            // Compute d_decay (sum over all elements)
            {
                float d_decay_local = 0.0f;
                for (int idx = tid; idx < state_size; idx += blockDim.x) {
                    int i = idx / HEAD_V_DIM;
                    int j = idx % HEAD_V_DIM;
                    float S_t_ij = tanhf(decay_val * S[idx] + delta[j] * k[i]);
                    float d_pre = dS[idx] * (1.0f - S_t_ij * S_t_ij);
                    d_decay_local += d_pre * S[idx];
                }
                atomicAdd(&d_decay_accum, d_decay_local);
            }
            __syncthreads();

            // d_v = d_delta (since delta = v - retrieved)
            // d_retrieved = -d_delta
            // d_S from retrieved: retrieved[j] = sum_i S[i,j] * k[i]
            // d_S[i,j] += k[i] * d_retrieved[j] = -k[i] * d_delta[j]
            if (tid < N_STATE) {
                float d_k_from_retrieved = 0.0f;
                for (int j = 0; j < HEAD_V_DIM; j++) {
                    d_k_from_retrieved += S[tid * HEAD_V_DIM + j] * (-d_delta[j]);
                }
                d_k[tid] += d_k_from_retrieved;
            }
            __syncthreads();

            // Write gradients for this timestep
            int k_offset = ((t * B + b) * H + h) * N_STATE;
            if (tid < N_STATE) {
                d_k_all[k_offset + tid] = __float2bfloat16(d_k[tid]);
                d_q_all[k_offset + tid] = __float2bfloat16(d_q[tid]);
            }
            if (tid < HEAD_V_DIM) {
                d_v_all[v_offset + tid] = __float2bfloat16(d_delta[tid]);
            }
            if (tid == 0) {
                d_decay_all[decay_offset] = __float2bfloat16(d_decay_accum);
            }
            __syncthreads();

            // Update dS for next iteration (propagate through decay)
            for (int idx = tid; idx < state_size; idx += blockDim.x) {
                int i = idx / HEAD_V_DIM;
                int j = idx % HEAD_V_DIM;
                float S_t_ij = tanhf(decay_val * S[idx] + delta[j] * k[i]);
                float d_pre = dS[idx] * (1.0f - S_t_ij * S_t_ij);
                // dS_{t-1} = d_pre * decay + contribution from retrieved gradient
                dS[idx] = d_pre * decay_val + (-d_delta[j]) * k[i];
            }
            __syncthreads();
        }
    }
}

// ============================================================================
// E88 FLA Hybrid Backward Kernel - Global Memory Version
// For large configurations (n_state=96/128, head_v_dim=128) that exceed shared memory limits
// S and dS are stored in per-block global memory instead of shared memory
// ============================================================================

template<int N_STATE, int HEAD_V_DIM>
__global__ void E88FLAHybridBackwardKernel_GlobalMem_BF16(
    int T,
    int B,
    int H,
    const __nv_bfloat16* __restrict__ k_all,
    const __nv_bfloat16* __restrict__ v_all,
    const __nv_bfloat16* __restrict__ q_all,
    const __nv_bfloat16* __restrict__ decay_all,
    const __nv_bfloat16* __restrict__ S_checkpoints,
    const __nv_bfloat16* __restrict__ Sq_cache,
    const __nv_bfloat16* __restrict__ d_output,
    __nv_bfloat16* __restrict__ d_k_all,
    __nv_bfloat16* __restrict__ d_v_all,
    __nv_bfloat16* __restrict__ d_q_all,
    __nv_bfloat16* __restrict__ d_decay_all,
    float* __restrict__ S_global,   // [B*H, N_STATE, HEAD_V_DIM] per-block state
    float* __restrict__ dS_global,  // [B*H, N_STATE, HEAD_V_DIM] per-block gradient
    int checkpoint_interval
) {
    int block_idx = blockIdx.x;
    int b = block_idx / H;
    int h = block_idx % H;
    if (b >= B) return;

    // Global memory pointers for this block's S and dS
    const int state_size = N_STATE * HEAD_V_DIM;
    float* S = S_global + block_idx * state_size;
    float* dS = dS_global + block_idx * state_size;

    // Shared memory for small buffers only
    extern __shared__ float shared_mem[];
    // Memory layout (small buffers):
    // k: N_STATE
    // v: HEAD_V_DIM
    // q: N_STATE
    // delta: HEAD_V_DIM
    // retrieved: HEAD_V_DIM
    // d_k: N_STATE
    // d_v: HEAD_V_DIM
    // d_q: N_STATE
    // d_Sq: HEAD_V_DIM
    // d_delta: HEAD_V_DIM
    float* k = shared_mem;
    float* v = k + N_STATE;
    float* q = v + HEAD_V_DIM;
    float* delta = q + N_STATE;
    float* retrieved = delta + HEAD_V_DIM;
    float* d_k = retrieved + HEAD_V_DIM;
    float* d_v = d_k + N_STATE;
    float* d_q = d_v + HEAD_V_DIM;
    float* d_Sq = d_q + N_STATE;
    float* d_delta = d_Sq + HEAD_V_DIM;

    int tid = threadIdx.x;

    // Initialize dS to zero (in global memory)
    for (int i = tid; i < state_size; i += blockDim.x) {
        dS[i] = 0.0f;
    }
    __syncthreads();

    int num_segments = (T + checkpoint_interval - 1) / checkpoint_interval;

    for (int seg = num_segments - 1; seg >= 0; seg--) {
        int t_start = seg * checkpoint_interval;
        int t_end = min(t_start + checkpoint_interval, T);

        for (int t = t_end - 1; t >= t_start; t--) {
            // Reload checkpoint for this segment
            int cp_offset = (seg * B * H + b * H + h) * state_size;
            for (int i = tid; i < state_size; i += blockDim.x) {
                S[i] = __bfloat162float(S_checkpoints[cp_offset + i]);
            }
            __syncthreads();

            // Replay forward from checkpoint to timestep t
            __shared__ float decay_val_replay;
            for (int tt = t_start; tt <= t; tt++) {
                int k_offset = ((tt * B + b) * H + h) * N_STATE;
                int v_offset = ((tt * B + b) * H + h) * HEAD_V_DIM;
                int decay_offset = (tt * B + b) * H + h;

                if (tid < N_STATE) {
                    k[tid] = __bfloat162float(k_all[k_offset + tid]);
                    q[tid] = __bfloat162float(q_all[k_offset + tid]);
                }
                if (tid < HEAD_V_DIM) {
                    v[tid] = __bfloat162float(v_all[v_offset + tid]);
                }
                __syncthreads();

                if (tid == 0) {
                    decay_val_replay = __bfloat162float(decay_all[decay_offset]);
                }
                __syncthreads();

                // Compute retrieved = S @ k
                if (tid < HEAD_V_DIM) {
                    float sum = 0.0f;
                    for (int i = 0; i < N_STATE; i++) {
                        sum += S[i * HEAD_V_DIM + tid] * k[i];
                    }
                    retrieved[tid] = sum;
                    delta[tid] = v[tid] - retrieved[tid];
                }
                __syncthreads();

                // Update S if not yet at target timestep t
                if (tt < t) {
                    for (int idx = tid; idx < state_size; idx += blockDim.x) {
                        int i = idx / HEAD_V_DIM;
                        int j = idx % HEAD_V_DIM;
                        float update = decay_val_replay * S[idx] + delta[j] * k[i];
                        S[idx] = tanhf(update);
                    }
                    __syncthreads();
                }
            }

            // Now S contains S_{t-1} after update (or checkpoint S if t == t_start)
            // We have k[t], v[t], q[t], delta[t], retrieved[t], decay[t] in shared memory

            // Load decay for timestep t
            __shared__ float decay_val;
            int decay_offset = (t * B + b) * H + h;
            if (tid == 0) {
                decay_val = __bfloat162float(decay_all[decay_offset]);
            }
            __syncthreads();

            // Backward through output (no self-gating - FLA-GDN style)
            int v_offset = ((t * B + b) * H + h) * HEAD_V_DIM;
            if (tid < HEAD_V_DIM) {
                float d_out = __bfloat162float(d_output[v_offset + tid]);
                // output = Sq directly (no self-gating)
                // d_Sq = d_output directly
                d_Sq[tid] = d_out;
            }
            __syncthreads();

            // d_q from output: Sq[j] = sum_i S_t[i,j] * q[i]
            // d_q[i] = sum_j S_t[i,j] * d_Sq[j]
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int j = 0; j < HEAD_V_DIM; j++) {
                    // S_t[i,j] after tanh update
                    float S_t_ij = tanhf(decay_val * S[tid * HEAD_V_DIM + j] + delta[j] * k[tid]);
                    sum += S_t_ij * d_Sq[j];
                }
                d_q[tid] = sum;
            }
            __syncthreads();

            // dS += outer(q, d_Sq) for the output computation
            // dS[i,j] += q[i] * d_Sq[j]
            for (int idx = tid; idx < state_size; idx += blockDim.x) {
                int i = idx / HEAD_V_DIM;
                int j = idx % HEAD_V_DIM;
                dS[idx] += q[i] * d_Sq[j];
            }
            __syncthreads();

            // Backward through state update: S_t = tanh(decay * S_{t-1} + outer(delta, k))
            // Let pre = decay * S_{t-1} + delta * k^T
            // S_t = tanh(pre)
            // dS_t flows back through tanh: d_pre = dS * (1 - S_t^2)
            // d_delta[j] = sum_i d_pre[i,j] * k[i]
            // d_k[i] = sum_j d_pre[i,j] * delta[j]
            // d_decay = sum_{i,j} d_pre[i,j] * S_{t-1}[i,j]
            // dS_{t-1} = d_pre * decay

            __shared__ float d_decay_accum;
            if (tid == 0) d_decay_accum = 0.0f;
            __syncthreads();

            // Compute d_delta and d_k contributions
            if (tid < HEAD_V_DIM) {
                float d_delta_local = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    float S_t_ij = tanhf(decay_val * S[i * HEAD_V_DIM + tid] + delta[tid] * k[i]);
                    float d_pre = dS[i * HEAD_V_DIM + tid] * (1.0f - S_t_ij * S_t_ij);
                    d_delta_local += d_pre * k[i];
                }
                d_delta[tid] = d_delta_local;
            }
            __syncthreads();

            if (tid < N_STATE) {
                float d_k_local = 0.0f;
                for (int j = 0; j < HEAD_V_DIM; j++) {
                    float S_t_ij = tanhf(decay_val * S[tid * HEAD_V_DIM + j] + delta[j] * k[tid]);
                    float d_pre = dS[tid * HEAD_V_DIM + j] * (1.0f - S_t_ij * S_t_ij);
                    d_k_local += d_pre * delta[j];
                }
                d_k[tid] = d_k_local;
            }
            __syncthreads();

            // Compute d_decay (sum over all elements)
            {
                float d_decay_local = 0.0f;
                for (int idx = tid; idx < state_size; idx += blockDim.x) {
                    int i = idx / HEAD_V_DIM;
                    int j = idx % HEAD_V_DIM;
                    float S_t_ij = tanhf(decay_val * S[idx] + delta[j] * k[i]);
                    float d_pre = dS[idx] * (1.0f - S_t_ij * S_t_ij);
                    d_decay_local += d_pre * S[idx];
                }
                atomicAdd(&d_decay_accum, d_decay_local);
            }
            __syncthreads();

            // d_v = d_delta (since delta = v - retrieved)
            // d_retrieved = -d_delta
            // d_S from retrieved: retrieved[j] = sum_i S[i,j] * k[i]
            // d_S[i,j] += k[i] * d_retrieved[j] = -k[i] * d_delta[j]
            if (tid < N_STATE) {
                float d_k_from_retrieved = 0.0f;
                for (int j = 0; j < HEAD_V_DIM; j++) {
                    d_k_from_retrieved += S[tid * HEAD_V_DIM + j] * (-d_delta[j]);
                }
                d_k[tid] += d_k_from_retrieved;
            }
            __syncthreads();

            // Write gradients for this timestep
            int k_offset = ((t * B + b) * H + h) * N_STATE;
            if (tid < N_STATE) {
                d_k_all[k_offset + tid] = __float2bfloat16(d_k[tid]);
                d_q_all[k_offset + tid] = __float2bfloat16(d_q[tid]);
            }
            if (tid < HEAD_V_DIM) {
                d_v_all[v_offset + tid] = __float2bfloat16(d_delta[tid]);
            }
            if (tid == 0) {
                d_decay_all[decay_offset] = __float2bfloat16(d_decay_accum);
            }
            __syncthreads();

            // Update dS for next iteration (propagate through decay)
            for (int idx = tid; idx < state_size; idx += blockDim.x) {
                int i = idx / HEAD_V_DIM;
                int j = idx % HEAD_V_DIM;
                float S_t_ij = tanhf(decay_val * S[idx] + delta[j] * k[i]);
                float d_pre = dS[idx] * (1.0f - S_t_ij * S_t_ij);
                // dS_{t-1} = d_pre * decay + contribution from retrieved gradient
                dS[idx] = d_pre * decay_val + (-d_delta[j]) * k[i];
            }
            __syncthreads();
        }
    }
}

// ============================================================================
// Dispatcher functions
// ============================================================================

void dispatch_e88_fla_hybrid_forward(
    int T, int B, int H, int n_state, int head_v_dim,
    const __nv_bfloat16* k_all, const __nv_bfloat16* v_all,
    const __nv_bfloat16* q_all, const __nv_bfloat16* decay_all,
    __nv_bfloat16* S, __nv_bfloat16* output,
    __nv_bfloat16* S_checkpoints, __nv_bfloat16* Sq_cache,
    int checkpoint_interval, cudaStream_t stream
) {
    int state_size = n_state * head_v_dim;
    int shared_size = (state_size + n_state + head_v_dim + n_state + head_v_dim) * sizeof(float);
    int threads = min(256, state_size);
    int num_blocks = B * H;

    // Dispatch based on n_state and head_v_dim combinations
    // For configs requiring >48KB shared memory, request extended shared memory
    #define DISPATCH_E88_FWD(N, V) do { \
        auto kernel = E88FLAHybridForwardKernel_BF16<N, V>; \
        if (shared_size > 48 * 1024) { \
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size); \
        } \
        kernel<<<num_blocks, threads, shared_size, stream>>>( \
            T, B, H, k_all, v_all, q_all, decay_all, S, output, \
            S_checkpoints, Sq_cache, checkpoint_interval); \
    } while(0)

    // Common configurations - ordered by n_state for clarity
    // n_state=8 (tiny state for many-head experiments)
    if (n_state == 8 && head_v_dim == 8) { DISPATCH_E88_FWD(8, 8); }
    else if (n_state == 8 && head_v_dim == 16) { DISPATCH_E88_FWD(8, 16); }
    else if (n_state == 8 && head_v_dim == 32) { DISPATCH_E88_FWD(8, 32); }
    else if (n_state == 8 && head_v_dim == 64) { DISPATCH_E88_FWD(8, 64); }
    // n_state=16 (small state, low shared memory)
    else if (n_state == 16 && head_v_dim == 16) { DISPATCH_E88_FWD(16, 16); }
    else if (n_state == 16 && head_v_dim == 32) { DISPATCH_E88_FWD(16, 32); }
    else if (n_state == 16 && head_v_dim == 64) { DISPATCH_E88_FWD(16, 64); }
    else if (n_state == 16 && head_v_dim == 128) { DISPATCH_E88_FWD(16, 128); }
    // n_state=24
    else if (n_state == 24 && head_v_dim == 24) { DISPATCH_E88_FWD(24, 24); }
    else if (n_state == 24 && head_v_dim == 32) { DISPATCH_E88_FWD(24, 32); }
    else if (n_state == 24 && head_v_dim == 48) { DISPATCH_E88_FWD(24, 48); }
    else if (n_state == 24 && head_v_dim == 64) { DISPATCH_E88_FWD(24, 64); }
    else if (n_state == 24 && head_v_dim == 96) { DISPATCH_E88_FWD(24, 96); }
    else if (n_state == 24 && head_v_dim == 128) { DISPATCH_E88_FWD(24, 128); }
    // n_state=32
    else if (n_state == 32 && head_v_dim == 32) { DISPATCH_E88_FWD(32, 32); }
    else if (n_state == 32 && head_v_dim == 64) { DISPATCH_E88_FWD(32, 64); }
    else if (n_state == 32 && head_v_dim == 128) { DISPATCH_E88_FWD(32, 128); }
    // n_state=64
    else if (n_state == 64 && head_v_dim == 32) { DISPATCH_E88_FWD(64, 32); }
    else if (n_state == 64 && head_v_dim == 64) { DISPATCH_E88_FWD(64, 64); }
    else if (n_state == 64 && head_v_dim == 128) { DISPATCH_E88_FWD(64, 128); }
    // n_state=96
    else if (n_state == 96 && head_v_dim == 32) { DISPATCH_E88_FWD(96, 32); }
    else if (n_state == 96 && head_v_dim == 64) { DISPATCH_E88_FWD(96, 64); }
    else if (n_state == 96 && head_v_dim == 96) { DISPATCH_E88_FWD(96, 96); }
    else if (n_state == 96 && head_v_dim == 128) { DISPATCH_E88_FWD(96, 128); }
    // n_state=128
    else if (n_state == 128 && head_v_dim == 32) { DISPATCH_E88_FWD(128, 32); }
    else if (n_state == 128 && head_v_dim == 64) { DISPATCH_E88_FWD(128, 64); }
    else if (n_state == 128 && head_v_dim == 128) { DISPATCH_E88_FWD(128, 128); }
    else {
        fprintf(stderr, "E88 FLA Hybrid Forward: unsupported n_state=%d, head_v_dim=%d\n", n_state, head_v_dim);
    }

    #undef DISPATCH_E88_FWD
}

// Check if configuration needs global memory fallback
// Returns true if shared memory requirement exceeds ~96KB limit
inline bool e88_needs_global_mem_backward(int n_state, int head_v_dim) {
    int state_size = n_state * head_v_dim;
    int shared_size = (2 * state_size + 3 * n_state + 5 * head_v_dim) * sizeof(float);
    return shared_size > 96 * 1024;  // 96KB limit for safety margin
}

void dispatch_e88_fla_hybrid_backward(
    int T, int B, int H, int n_state, int head_v_dim,
    const __nv_bfloat16* k_all, const __nv_bfloat16* v_all,
    const __nv_bfloat16* q_all, const __nv_bfloat16* decay_all,
    const __nv_bfloat16* S_checkpoints, const __nv_bfloat16* Sq_cache,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* d_k_all, __nv_bfloat16* d_v_all,
    __nv_bfloat16* d_q_all, __nv_bfloat16* d_decay_all,
    float* S_global, float* dS_global,  // For global memory fallback (can be nullptr if not needed)
    int checkpoint_interval, cudaStream_t stream
) {
    int state_size = n_state * head_v_dim;
    // Shared memory: 2*state_size + 2*n_state + 4*head_v_dim + n_state + head_v_dim + n_state + head_v_dim
    int shared_size = (2 * state_size + 3 * n_state + 5 * head_v_dim) * sizeof(float);
    int threads = min(256, state_size);
    int num_blocks = B * H;

    // For configs requiring >48KB shared memory, we need to request extended shared memory
    // SM89 (Ada) supports up to 100KB per block
    // Configs exceeding 96KB use global memory fallback
    #define DISPATCH_E88_BWD(N, V) do { \
        auto kernel = E88FLAHybridBackwardKernel_BF16<N, V>; \
        if (shared_size > 48 * 1024) { \
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size); \
        } \
        kernel<<<num_blocks, threads, shared_size, stream>>>( \
            T, B, H, k_all, v_all, q_all, decay_all, \
            S_checkpoints, Sq_cache, d_output, \
            d_k_all, d_v_all, d_q_all, d_decay_all, checkpoint_interval); \
    } while(0)

    // Global memory kernel dispatch - uses small buffers only in shared memory
    // Shared mem for global version: 3*n_state + 5*head_v_dim floats
    #define DISPATCH_E88_BWD_GLOBAL(N, V) do { \
        int global_shared_size = (3 * N + 5 * V) * sizeof(float); \
        E88FLAHybridBackwardKernel_GlobalMem_BF16<N, V><<<num_blocks, threads, global_shared_size, stream>>>( \
            T, B, H, k_all, v_all, q_all, decay_all, \
            S_checkpoints, Sq_cache, d_output, \
            d_k_all, d_v_all, d_q_all, d_decay_all, \
            S_global, dS_global, checkpoint_interval); \
    } while(0)

    // n_state=8 (tiny state for many-head experiments)
    if (n_state == 8 && head_v_dim == 8) { DISPATCH_E88_BWD(8, 8); }
    else if (n_state == 8 && head_v_dim == 16) { DISPATCH_E88_BWD(8, 16); }
    else if (n_state == 8 && head_v_dim == 32) { DISPATCH_E88_BWD(8, 32); }
    else if (n_state == 8 && head_v_dim == 64) { DISPATCH_E88_BWD(8, 64); }
    // n_state=16 (small state, low shared memory)
    else if (n_state == 16 && head_v_dim == 16) { DISPATCH_E88_BWD(16, 16); }
    else if (n_state == 16 && head_v_dim == 32) { DISPATCH_E88_BWD(16, 32); }
    else if (n_state == 16 && head_v_dim == 64) { DISPATCH_E88_BWD(16, 64); }
    else if (n_state == 16 && head_v_dim == 128) { DISPATCH_E88_BWD(16, 128); }
    // n_state=24
    else if (n_state == 24 && head_v_dim == 24) { DISPATCH_E88_BWD(24, 24); }
    else if (n_state == 24 && head_v_dim == 32) { DISPATCH_E88_BWD(24, 32); }
    else if (n_state == 24 && head_v_dim == 48) { DISPATCH_E88_BWD(24, 48); }
    else if (n_state == 24 && head_v_dim == 64) { DISPATCH_E88_BWD(24, 64); }
    else if (n_state == 24 && head_v_dim == 96) { DISPATCH_E88_BWD(24, 96); }
    else if (n_state == 24 && head_v_dim == 128) { DISPATCH_E88_BWD(24, 128); }
    // n_state=32
    else if (n_state == 32 && head_v_dim == 32) { DISPATCH_E88_BWD(32, 32); }
    else if (n_state == 32 && head_v_dim == 64) { DISPATCH_E88_BWD(32, 64); }
    else if (n_state == 32 && head_v_dim == 128) { DISPATCH_E88_BWD(32, 128); }
    // n_state=64
    else if (n_state == 64 && head_v_dim == 32) { DISPATCH_E88_BWD(64, 32); }
    else if (n_state == 64 && head_v_dim == 64) { DISPATCH_E88_BWD(64, 64); }
    else if (n_state == 64 && head_v_dim == 128) { DISPATCH_E88_BWD(64, 128); }
    // n_state=96
    else if (n_state == 96 && head_v_dim == 32) { DISPATCH_E88_BWD(96, 32); }
    else if (n_state == 96 && head_v_dim == 64) { DISPATCH_E88_BWD(96, 64); }
    else if (n_state == 96 && head_v_dim == 96) { DISPATCH_E88_BWD(96, 96); }  // 75KB shared, within limit
    else if (n_state == 96 && head_v_dim == 128) {
        // ~100KB shared needed, use global memory fallback
        DISPATCH_E88_BWD_GLOBAL(96, 128);
    }
    // n_state=128
    else if (n_state == 128 && head_v_dim == 32) { DISPATCH_E88_BWD(128, 32); }
    else if (n_state == 128 && head_v_dim == 64) { DISPATCH_E88_BWD(128, 64); }
    else if (n_state == 128 && head_v_dim == 128) {
        // ~132KB shared needed, use global memory fallback
        DISPATCH_E88_BWD_GLOBAL(128, 128);
    }
    else {
        fprintf(stderr, "E88 FLA Hybrid Backward: unsupported n_state=%d, head_v_dim=%d\n", n_state, head_v_dim);
    }

    #undef DISPATCH_E88_BWD
    #undef DISPATCH_E88_BWD_GLOBAL
}

// ============================================================================
// E88FLAHybridForward Implementation
// ============================================================================

template<typename DataT>
E88FLAHybridForward<DataT>::E88FLAHybridForward(
    bool training,
    int batch_size,
    int n_state,
    int head_v_dim,
    int n_heads,
    const cudaStream_t& stream)
    : training_(training),
      batch_size_(batch_size),
      n_state_(n_state),
      head_v_dim_(head_v_dim),
      n_heads_(n_heads),
      stream_(stream) {}

template<typename DataT>
void E88FLAHybridForward<DataT>::Run(
    int steps,
    const DataT* k,         // [T, B, H, n_state] L2 normalized keys
    const DataT* v,         // [T, B, H, head_v_dim] values
    const DataT* q,         // [T, B, H, n_state] L2 normalized queries
    const DataT* decay,     // [T, B, H] exponential decay factors
    DataT* S,               // [B, H, n_state, head_v_dim]
    DataT* output,          // [T, B, H, head_v_dim]
    DataT* S_cache          // checkpoints + Sq_cache
) {
    int T = steps;
    int B = batch_size_;
    int n = n_state_;
    int v_dim = head_v_dim_;
    int H = n_heads_;

    int num_checkpoints = (T + E88_CHECKPOINT_INTERVAL - 1) / E88_CHECKPOINT_INTERVAL + 1;
    DataT* s_checkpoints = S_cache;
    DataT* sq_cache = S_cache + num_checkpoints * B * H * n * v_dim;

    dispatch_e88_fla_hybrid_forward(
        T, B, H, n, v_dim,
        (const __nv_bfloat16*)k, (const __nv_bfloat16*)v,
        (const __nv_bfloat16*)q, (const __nv_bfloat16*)decay,
        (__nv_bfloat16*)S, (__nv_bfloat16*)output,
        (__nv_bfloat16*)s_checkpoints, (__nv_bfloat16*)sq_cache,
        E88_CHECKPOINT_INTERVAL, stream_);
}

template struct E88FLAHybridForward<__nv_bfloat16>;

// ============================================================================
// E88FLAHybridBackward Implementation
// ============================================================================

template<typename DataT>
E88FLAHybridBackward<DataT>::E88FLAHybridBackward(
    int batch_size,
    int n_state,
    int head_v_dim,
    int n_heads,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      n_state_(n_state),
      head_v_dim_(head_v_dim),
      n_heads_(n_heads),
      stream_(stream),
      use_global_mem_(false),
      S_global_(nullptr),
      dS_global_(nullptr) {
    // Check if we need global memory fallback
    use_global_mem_ = e88_needs_global_mem_backward(n_state, head_v_dim);
    if (use_global_mem_) {
        // Allocate global memory for S and dS: [B*H, n_state, head_v_dim] each
        size_t state_size = batch_size * n_heads * n_state * head_v_dim * sizeof(float);
        cudaMalloc(&S_global_, state_size);
        cudaMalloc(&dS_global_, state_size);
    }
}

template<typename DataT>
E88FLAHybridBackward<DataT>::~E88FLAHybridBackward() {
    if (S_global_) {
        cudaFree(S_global_);
        S_global_ = nullptr;
    }
    if (dS_global_) {
        cudaFree(dS_global_);
        dS_global_ = nullptr;
    }
}

template<typename DataT>
void E88FLAHybridBackward<DataT>::Run(
    int steps,
    const DataT* k,
    const DataT* v,
    const DataT* q,
    const DataT* decay,
    const DataT* S_checkpoints,
    const DataT* Sq_cache,
    const DataT* d_output,
    DataT* d_k,
    DataT* d_v,
    DataT* d_q,
    DataT* d_decay
) {
    int T = steps;
    int B = batch_size_;
    int n = n_state_;
    int v_dim = head_v_dim_;
    int H = n_heads_;

    dispatch_e88_fla_hybrid_backward(
        T, B, H, n, v_dim,
        (const __nv_bfloat16*)k, (const __nv_bfloat16*)v,
        (const __nv_bfloat16*)q, (const __nv_bfloat16*)decay,
        (const __nv_bfloat16*)S_checkpoints, (const __nv_bfloat16*)Sq_cache,
        (const __nv_bfloat16*)d_output,
        (__nv_bfloat16*)d_k, (__nv_bfloat16*)d_v,
        (__nv_bfloat16*)d_q, (__nv_bfloat16*)d_decay,
        S_global_, dS_global_,  // Pass global memory buffers (nullptr if not using global fallback)
        E88_CHECKPOINT_INTERVAL, stream_);
}

template struct E88FLAHybridBackward<__nv_bfloat16>;

}  // namespace elman
