/**
 * E89 Residual State CUDA Kernel
 *
 * Key difference from E88: RESIDUAL state update
 *   E88: S = tanh(decay * S + outer(delta, k))  -- tanh on entire update
 *   E89: S = decay * S + tanh(outer(delta, k))  -- tanh only on outer product
 *
 * This allows gradients to flow through the decay term without passing through
 * tanh's saturating derivative. Better gradient flow for longer sequences.
 *
 * Combines:
 * 1. Mamba2-style exponential decay (passed as decay tensor)
 * 2. RESIDUAL state update: S = decay * S + tanh(outer(delta, k_norm))
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
 *   S_h = decay_h * S_h + tanh(outer(delta, k_h))  [n_state x head_v_dim]  <-- RESIDUAL
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

#define E89_CHECKPOINT_INTERVAL 32  // Checkpoint every 32 steps

namespace elman {

// ============================================================================
// E89 FLA Hybrid Forward Kernel
// Each block handles one (batch, head) pair
// ============================================================================

template<int N_STATE, int HEAD_V_DIM>
__global__ void E89ResidualStateForwardKernel_BF16(
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

        // E89 TRUE RESIDUAL state update: S = S + tanh(decay * S + outer(delta, k))
        // This is like E88's update but added as a residual to preserve gradient flow
        // delta = v - retrieved
        // outer(delta, k)[i,j] = delta[j] * k[i]
        // pre[i,j] = decay * S[i,j] + outer[i,j]
        // S_new[i,j] = S[i,j] + tanh(pre[i,j])
        for (int idx = tid; idx < state_size; idx += blockDim.x) {
            int i = idx / HEAD_V_DIM;  // row index (k dimension)
            int j = idx % HEAD_V_DIM;  // col index (v dimension)
            float delta_j = v_shared[j] - retrieved[j];
            float outer_term = delta_j * k_shared[i];
            float pre = decay_val * S_shared[idx] + outer_term;
            S_shared[idx] = S_shared[idx] + tanhf(pre);  // Residual connection!
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
// E89 FLA Hybrid Backward Kernel (Optimized with Segment-Level Caching)
//
// Previous version was O(T * checkpoint_interval) - recomputed from checkpoint
// for EACH timestep within a segment.
//
// This version is O(T) - for each segment:
//   1. Replay forward ONCE, caching all S_{t-1} states to global memory
//   2. Process backward using cached states
// ============================================================================

template<int N_STATE, int HEAD_V_DIM>
__global__ void E89ResidualStateBackwardKernel_BF16(
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
    __nv_bfloat16* __restrict__ segment_state_cache,  // [B*H, checkpoint_interval, N_STATE, HEAD_V_DIM]
    int checkpoint_interval
) {
    int block_idx = blockIdx.x;
    int b = block_idx / H;
    int h = block_idx % H;
    if (b >= B) return;

    extern __shared__ float shared_mem[];
    // Memory layout:
    // S: N_STATE * HEAD_V_DIM (S_{t-1})
    // dS: N_STATE * HEAD_V_DIM
    // S_t: N_STATE * HEAD_V_DIM (cached tanh result)
    // dtanh: N_STATE * HEAD_V_DIM (cached 1 - S_t^2)
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
    float* S_t = dS + N_STATE * HEAD_V_DIM;
    float* dtanh = S_t + N_STATE * HEAD_V_DIM;
    float* k = dtanh + N_STATE * HEAD_V_DIM;
    float* v = k + N_STATE;
    float* q = v + HEAD_V_DIM;
    float* delta = q + N_STATE;
    float* retrieved = delta + HEAD_V_DIM;
    float* d_k = retrieved + HEAD_V_DIM;
    float* d_v = d_k + N_STATE;
    float* d_q = d_v + HEAD_V_DIM;
    float* d_Sq = d_q + N_STATE;
    float* d_delta = d_Sq + HEAD_V_DIM;
    float* warp_results = d_delta + HEAD_V_DIM;  // [8 floats for up to 256 threads / 32]

    int tid = threadIdx.x;
    int state_size = N_STATE * HEAD_V_DIM;

    // Segment cache offset for this (batch, head) pair
    // Layout: [B*H, checkpoint_interval, N_STATE, HEAD_V_DIM]
    __nv_bfloat16* seg_cache_base = segment_state_cache + (size_t)block_idx * checkpoint_interval * state_size;

    // Initialize dS to zero
    for (int i = tid; i < state_size; i += blockDim.x) {
        dS[i] = 0.0f;
    }
    __syncthreads();

    int num_segments = (T + checkpoint_interval - 1) / checkpoint_interval;

    for (int seg = num_segments - 1; seg >= 0; seg--) {
        int t_start = seg * checkpoint_interval;
        int t_end = min(t_start + checkpoint_interval, T);
        int seg_len = t_end - t_start;

        // ================================================================
        // PHASE 1: Forward replay through segment, caching all S_{t-1} states
        // ================================================================

        // Load checkpoint for this segment
        int cp_offset = (seg * B * H + b * H + h) * state_size;
        for (int i = tid; i < state_size; i += blockDim.x) {
            S[i] = __bfloat162float(S_checkpoints[cp_offset + i]);
        }
        __syncthreads();

        // Replay forward through entire segment, saving S_{t-1} for each t
        __shared__ float decay_val_replay;
        for (int local_t = 0; local_t < seg_len; local_t++) {
            int t = t_start + local_t;

            // Save S_{t-1} to segment cache BEFORE the update
            // This is the state we need for backward at timestep t
            __nv_bfloat16* cache_slot = seg_cache_base + (size_t)local_t * state_size;
            for (int i = tid; i < state_size; i += blockDim.x) {
                cache_slot[i] = __float2bfloat16(S[i]);
            }
            __syncthreads();

            // Load inputs for timestep t
            int k_offset = ((t * B + b) * H + h) * N_STATE;
            int v_offset = ((t * B + b) * H + h) * HEAD_V_DIM;
            int decay_offset = (t * B + b) * H + h;

            if (tid < N_STATE) {
                k[tid] = __bfloat162float(k_all[k_offset + tid]);
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
                #pragma unroll 8
                for (int i = 0; i < N_STATE; i++) {
                    sum += S[i * HEAD_V_DIM + tid] * k[i];
                }
                retrieved[tid] = sum;
                delta[tid] = v[tid] - retrieved[tid];
            }
            __syncthreads();

            // E89 RESIDUAL: S_t = decay * S_{t-1} + tanh(outer(delta, k))
            for (int idx = tid; idx < state_size; idx += blockDim.x) {
                int i = idx / HEAD_V_DIM;
                int j = idx % HEAD_V_DIM;
                float outer_term = delta[j] * k[i];
                S[idx] = decay_val_replay * S[idx] + tanhf(outer_term);
            }
            __syncthreads();
        }

        // ================================================================
        // PHASE 2: Backward pass through segment using cached states
        // ================================================================

        for (int local_t = seg_len - 1; local_t >= 0; local_t--) {
            int t = t_start + local_t;

            // Load cached S_{t-1} from segment cache
            __nv_bfloat16* cache_slot = seg_cache_base + (size_t)local_t * state_size;
            for (int i = tid; i < state_size; i += blockDim.x) {
                S[i] = __bfloat162float(cache_slot[i]);
            }
            __syncthreads();

            // Load k, v, q for timestep t
            int k_offset = ((t * B + b) * H + h) * N_STATE;
            int v_offset = ((t * B + b) * H + h) * HEAD_V_DIM;
            int decay_offset = (t * B + b) * H + h;

            if (tid < N_STATE) {
                k[tid] = __bfloat162float(k_all[k_offset + tid]);
                q[tid] = __bfloat162float(q_all[k_offset + tid]);
            }
            if (tid < HEAD_V_DIM) {
                v[tid] = __bfloat162float(v_all[v_offset + tid]);
            }
            __syncthreads();

            // Compute retrieved and delta for this timestep
            if (tid < HEAD_V_DIM) {
                float sum = 0.0f;
                #pragma unroll 8
                for (int i = 0; i < N_STATE; i++) {
                    sum += S[i * HEAD_V_DIM + tid] * k[i];
                }
                retrieved[tid] = sum;
                delta[tid] = v[tid] - retrieved[tid];
            }
            __syncthreads();

            // Load decay for timestep t
            __shared__ float decay_val;
            if (tid == 0) {
                decay_val = __bfloat162float(decay_all[decay_offset]);
            }
            __syncthreads();

            // *** E89 TRUE RESIDUAL: S_t = S_{t-1} + tanh(decay * S_{t-1} + outer) ***
            for (int idx = tid; idx < state_size; idx += blockDim.x) {
                int i = idx / HEAD_V_DIM;
                int j = idx % HEAD_V_DIM;
                float outer_term = delta[j] * k[i];
                float pre = decay_val * S[idx] + outer_term;
                float tanh_pre = tanhf(pre);
                S_t[idx] = S[idx] + tanh_pre;  // True residual: S + tanh(pre)
                dtanh[idx] = 1.0f - tanh_pre * tanh_pre;  // dtanh of pre
            }
            __syncthreads();

            // Backward through output (no self-gating - FLA-GDN style)
            if (tid < HEAD_V_DIM) {
                float d_out = __bfloat162float(d_output[v_offset + tid]);
                d_Sq[tid] = d_out;
            }
            __syncthreads();

            // d_q from output: Sq[j] = sum_i S_t[i,j] * q[i]
            // d_q[i] = sum_j S_t[i,j] * d_Sq[j]
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int j = 0; j < HEAD_V_DIM; j++) {
                    sum += S_t[tid * HEAD_V_DIM + j] * d_Sq[j];
                }
                d_q[tid] = sum;
            }
            __syncthreads();

            // dS += outer(q, d_Sq) for the output computation
            for (int idx = tid; idx < state_size; idx += blockDim.x) {
                int i = idx / HEAD_V_DIM;
                int j = idx % HEAD_V_DIM;
                dS[idx] += q[i] * d_Sq[j];
            }
            __syncthreads();

            // Backward through state update using cached dtanh

            // Compute d_delta using cached dtanh
            if (tid < HEAD_V_DIM) {
                float d_delta_local = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    float d_pre = dS[i * HEAD_V_DIM + tid] * dtanh[i * HEAD_V_DIM + tid];
                    d_delta_local += d_pre * k[i];
                }
                d_delta[tid] = d_delta_local;
            }
            __syncthreads();

            // Compute d_k using cached dtanh
            if (tid < N_STATE) {
                float d_k_local = 0.0f;
                for (int j = 0; j < HEAD_V_DIM; j++) {
                    float d_pre = dS[tid * HEAD_V_DIM + j] * dtanh[tid * HEAD_V_DIM + j];
                    d_k_local += d_pre * delta[j];
                }
                d_k[tid] = d_k_local;
            }
            __syncthreads();

            // E89 TRUE RESIDUAL: d_decay = sum(dS * dtanh * S_{t-1})
            // For S_t = S_{t-1} + tanh(pre) where pre = decay * S_{t-1} + outer:
            // d_pre = dS_t * dtanh, so d_decay = sum(d_pre * S_{t-1}) = sum(dS * dtanh * S_{t-1})
            float d_decay_local = 0.0f;
            for (int idx = tid; idx < state_size; idx += blockDim.x) {
                d_decay_local += dS[idx] * dtanh[idx] * S[idx];
            }

            // Warp reduction using shuffle intrinsics
            #pragma unroll
            for (int offset = 16; offset >= 1; offset /= 2) {
                d_decay_local += __shfl_xor_sync(0xFFFFFFFF, d_decay_local, offset);
            }

            // Lane 0 of each warp writes partial sum to shared memory
            const int warp_id = tid / 32;
            const int lane_id = tid % 32;
            const int num_warps = (blockDim.x + 31) / 32;
            if (lane_id == 0) {
                warp_results[warp_id] = d_decay_local;
            }
            __syncthreads();

            // First warp reduces across all warp results
            float d_decay_accum = 0.0f;
            if (warp_id == 0) {
                float load_val = (tid < num_warps) ? warp_results[tid] : 0.0f;
                #pragma unroll
                for (int offset = 16; offset >= 1; offset /= 2) {
                    load_val += __shfl_xor_sync(0xFFFFFFFF, load_val, offset);
                }
                d_decay_accum = load_val;
            }
            __syncthreads();

            // d_k contribution from retrieved gradient
            if (tid < N_STATE) {
                float d_k_from_retrieved = 0.0f;
                for (int j = 0; j < HEAD_V_DIM; j++) {
                    d_k_from_retrieved += S[tid * HEAD_V_DIM + j] * (-d_delta[j]);
                }
                d_k[tid] += d_k_from_retrieved;
            }
            __syncthreads();

            // Write gradients for this timestep
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

            // E89 TRUE RESIDUAL: Update dS for next iteration
            // S_t = S_{t-1} + tanh(pre) where pre = decay * S_{t-1} + outer
            // dS_{t-1} = dS_t * (1 + dtanh * decay) + retrieval_gradient
            // The identity gradient (dS_t) flows directly, plus the gradient through pre
            for (int idx = tid; idx < state_size; idx += blockDim.x) {
                int i = idx / HEAD_V_DIM;
                int j = idx % HEAD_V_DIM;
                float identity_grad = dS[idx];  // Gradient through identity connection
                float pre_grad = dS[idx] * dtanh[idx] * decay_val;  // Gradient through pre
                float retrieval_grad = (-d_delta[j]) * k[i];
                dS[idx] = identity_grad + pre_grad + retrieval_grad;
            }
            __syncthreads();
        }
    }
}

// ============================================================================
// E89 FLA Hybrid Backward Kernel - Cached Version (k, v, decay cached)
//
// Optimization over base kernel: caches k, v, decay during forward replay
// to avoid re-reading them from global memory during backward phase.
//
// Cache layout per (batch, head):
//   [0..checkpoint_interval-1]: S_{t-1} states [N_STATE * HEAD_V_DIM each]
//   [checkpoint_interval * N_STATE * HEAD_V_DIM + local_t * N_STATE]: k vectors
//   [checkpoint_interval * N_STATE * HEAD_V_DIM + checkpoint_interval * N_STATE + local_t * HEAD_V_DIM]: v vectors
//   [checkpoint_interval * N_STATE * HEAD_V_DIM + checkpoint_interval * N_STATE + checkpoint_interval * HEAD_V_DIM + local_t]: decay values
// ============================================================================

template<int N_STATE, int HEAD_V_DIM>
__global__ void E89ResidualStateBackwardKernel_Cached_BF16(
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
    __nv_bfloat16* __restrict__ segment_cache,  // Extended: [B*H, checkpoint_interval, N_STATE*HEAD_V_DIM + N_STATE + HEAD_V_DIM + 1]
    int checkpoint_interval
) {
    int block_idx = blockIdx.x;
    int b = block_idx / H;
    int h = block_idx % H;
    if (b >= B) return;

    extern __shared__ float shared_mem[];
    // Memory layout:
    // S: N_STATE * HEAD_V_DIM (S_{t-1})
    // dS: N_STATE * HEAD_V_DIM
    // S_t: N_STATE * HEAD_V_DIM (cached tanh result)
    // dtanh: N_STATE * HEAD_V_DIM (cached 1 - S_t^2)
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
    float* S_t = dS + N_STATE * HEAD_V_DIM;
    float* dtanh = S_t + N_STATE * HEAD_V_DIM;
    float* k = dtanh + N_STATE * HEAD_V_DIM;
    float* v = k + N_STATE;
    float* q = v + HEAD_V_DIM;
    float* delta = q + N_STATE;
    float* retrieved = delta + HEAD_V_DIM;
    float* d_k = retrieved + HEAD_V_DIM;
    float* d_v = d_k + N_STATE;
    float* d_q = d_v + HEAD_V_DIM;
    float* d_Sq = d_q + N_STATE;
    float* d_delta = d_Sq + HEAD_V_DIM;
    float* warp_results = d_delta + HEAD_V_DIM;  // [8 floats for up to 256 threads / 32]

    int tid = threadIdx.x;
    int state_size = N_STATE * HEAD_V_DIM;

    // Segment cache offsets for this (batch, head) pair
    // Extended layout: [S_states, k_vectors, v_vectors, decay_values]
    int cache_entry_size = state_size + N_STATE + HEAD_V_DIM + 1;
    __nv_bfloat16* seg_cache_base = segment_cache + (size_t)block_idx * checkpoint_interval * cache_entry_size;

    // Sub-cache offsets within each segment
    // Per-timestep: S at offset 0, then k, v, decay packed after all S entries
    // Actually: for each local_t, we have [S_{t-1}] contiguous, then [k, v, decay] after
    // Layout: [S_0, S_1, ..., S_{ci-1}] [k_0, k_1, ..., k_{ci-1}] [v_0, ...] [decay_0, ...]
    __nv_bfloat16* S_cache_base = seg_cache_base;  // [checkpoint_interval * state_size]
    __nv_bfloat16* k_cache_base = seg_cache_base + (size_t)checkpoint_interval * state_size;  // [checkpoint_interval * N_STATE]
    __nv_bfloat16* v_cache_base = k_cache_base + (size_t)checkpoint_interval * N_STATE;  // [checkpoint_interval * HEAD_V_DIM]
    __nv_bfloat16* decay_cache_base = v_cache_base + (size_t)checkpoint_interval * HEAD_V_DIM;  // [checkpoint_interval * 1]

    // Initialize dS to zero
    for (int i = tid; i < state_size; i += blockDim.x) {
        dS[i] = 0.0f;
    }
    __syncthreads();

    int num_segments = (T + checkpoint_interval - 1) / checkpoint_interval;

    for (int seg = num_segments - 1; seg >= 0; seg--) {
        int t_start = seg * checkpoint_interval;
        int t_end = min(t_start + checkpoint_interval, T);
        int seg_len = t_end - t_start;

        // ================================================================
        // PHASE 1: Forward replay through segment, caching S_{t-1}, k, v, decay
        // ================================================================

        // Load checkpoint for this segment
        int cp_offset = (seg * B * H + b * H + h) * state_size;
        for (int i = tid; i < state_size; i += blockDim.x) {
            S[i] = __bfloat162float(S_checkpoints[cp_offset + i]);
        }
        __syncthreads();

        // Replay forward through entire segment, saving S_{t-1}, k, v, decay for each t
        __shared__ float decay_val_replay;
        for (int local_t = 0; local_t < seg_len; local_t++) {
            int t = t_start + local_t;

            // Save S_{t-1} to segment cache BEFORE the update
            __nv_bfloat16* S_cache_slot = S_cache_base + (size_t)local_t * state_size;
            for (int i = tid; i < state_size; i += blockDim.x) {
                S_cache_slot[i] = __float2bfloat16(S[i]);
            }

            // Load inputs for timestep t from global memory
            int k_offset = ((t * B + b) * H + h) * N_STATE;
            int v_offset = ((t * B + b) * H + h) * HEAD_V_DIM;
            int decay_offset = (t * B + b) * H + h;

            if (tid < N_STATE) {
                k[tid] = __bfloat162float(k_all[k_offset + tid]);
            }
            if (tid < HEAD_V_DIM) {
                v[tid] = __bfloat162float(v_all[v_offset + tid]);
            }
            __syncthreads();

            if (tid == 0) {
                decay_val_replay = __bfloat162float(decay_all[decay_offset]);
            }
            __syncthreads();

            // Cache k, v, decay for backward phase
            __nv_bfloat16* k_cache_slot = k_cache_base + (size_t)local_t * N_STATE;
            __nv_bfloat16* v_cache_slot = v_cache_base + (size_t)local_t * HEAD_V_DIM;
            if (tid < N_STATE) {
                k_cache_slot[tid] = __float2bfloat16(k[tid]);
            }
            if (tid < HEAD_V_DIM) {
                v_cache_slot[tid] = __float2bfloat16(v[tid]);
            }
            if (tid == 0) {
                decay_cache_base[local_t] = __float2bfloat16(decay_val_replay);
            }
            __syncthreads();

            // Compute retrieved = S @ k
            if (tid < HEAD_V_DIM) {
                float sum = 0.0f;
                #pragma unroll 8
                for (int i = 0; i < N_STATE; i++) {
                    sum += S[i * HEAD_V_DIM + tid] * k[i];
                }
                retrieved[tid] = sum;
                delta[tid] = v[tid] - retrieved[tid];
            }
            __syncthreads();

            // E89 RESIDUAL: S_t = decay * S_{t-1} + tanh(outer(delta, k))
            for (int idx = tid; idx < state_size; idx += blockDim.x) {
                int i = idx / HEAD_V_DIM;
                int j = idx % HEAD_V_DIM;
                float outer_term = delta[j] * k[i];
                S[idx] = decay_val_replay * S[idx] + tanhf(outer_term);
            }
            __syncthreads();
        }

        // ================================================================
        // PHASE 2: Backward pass through segment using cached states and inputs
        // ================================================================

        for (int local_t = seg_len - 1; local_t >= 0; local_t--) {
            int t = t_start + local_t;

            // Load cached S_{t-1} from segment cache
            __nv_bfloat16* S_cache_slot = S_cache_base + (size_t)local_t * state_size;
            for (int i = tid; i < state_size; i += blockDim.x) {
                S[i] = __bfloat162float(S_cache_slot[i]);
            }

            // Load cached k, v from segment cache (NOT from global memory)
            __nv_bfloat16* k_cache_slot = k_cache_base + (size_t)local_t * N_STATE;
            __nv_bfloat16* v_cache_slot = v_cache_base + (size_t)local_t * HEAD_V_DIM;
            if (tid < N_STATE) {
                k[tid] = __bfloat162float(k_cache_slot[tid]);
            }
            if (tid < HEAD_V_DIM) {
                v[tid] = __bfloat162float(v_cache_slot[tid]);
            }
            __syncthreads();

            // Load q from global memory (still needed, not cached)
            int k_offset = ((t * B + b) * H + h) * N_STATE;
            int v_offset = ((t * B + b) * H + h) * HEAD_V_DIM;
            int decay_offset = (t * B + b) * H + h;

            if (tid < N_STATE) {
                q[tid] = __bfloat162float(q_all[k_offset + tid]);
            }
            __syncthreads();

            // Compute retrieved and delta for this timestep
            if (tid < HEAD_V_DIM) {
                float sum = 0.0f;
                #pragma unroll 8
                for (int i = 0; i < N_STATE; i++) {
                    sum += S[i * HEAD_V_DIM + tid] * k[i];
                }
                retrieved[tid] = sum;
                delta[tid] = v[tid] - retrieved[tid];
            }
            __syncthreads();

            // Load cached decay (NOT from global memory)
            __shared__ float decay_val;
            if (tid == 0) {
                decay_val = __bfloat162float(decay_cache_base[local_t]);
            }
            __syncthreads();

            // *** E89 TRUE RESIDUAL: S_t = S_{t-1} + tanh(decay * S_{t-1} + outer) ***
            for (int idx = tid; idx < state_size; idx += blockDim.x) {
                int i = idx / HEAD_V_DIM;
                int j = idx % HEAD_V_DIM;
                float outer_term = delta[j] * k[i];
                float pre = decay_val * S[idx] + outer_term;
                float tanh_pre = tanhf(pre);
                S_t[idx] = S[idx] + tanh_pre;  // True residual: S + tanh(pre)
                dtanh[idx] = 1.0f - tanh_pre * tanh_pre;  // dtanh of pre
            }
            __syncthreads();

            // Backward through output (no self-gating - FLA-GDN style)
            if (tid < HEAD_V_DIM) {
                float d_out = __bfloat162float(d_output[v_offset + tid]);
                d_Sq[tid] = d_out;
            }
            __syncthreads();

            // d_q from output: Sq[j] = sum_i S_t[i,j] * q[i]
            // d_q[i] = sum_j S_t[i,j] * d_Sq[j]
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int j = 0; j < HEAD_V_DIM; j++) {
                    sum += S_t[tid * HEAD_V_DIM + j] * d_Sq[j];
                }
                d_q[tid] = sum;
            }
            __syncthreads();

            // dS += outer(q, d_Sq) for the output computation
            for (int idx = tid; idx < state_size; idx += blockDim.x) {
                int i = idx / HEAD_V_DIM;
                int j = idx % HEAD_V_DIM;
                dS[idx] += q[i] * d_Sq[j];
            }
            __syncthreads();

            // Backward through state update using cached dtanh

            // Compute d_delta using cached dtanh
            if (tid < HEAD_V_DIM) {
                float d_delta_local = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    float d_pre = dS[i * HEAD_V_DIM + tid] * dtanh[i * HEAD_V_DIM + tid];
                    d_delta_local += d_pre * k[i];
                }
                d_delta[tid] = d_delta_local;
            }
            __syncthreads();

            // Compute d_k using cached dtanh
            if (tid < N_STATE) {
                float d_k_local = 0.0f;
                for (int j = 0; j < HEAD_V_DIM; j++) {
                    float d_pre = dS[tid * HEAD_V_DIM + j] * dtanh[tid * HEAD_V_DIM + j];
                    d_k_local += d_pre * delta[j];
                }
                d_k[tid] = d_k_local;
            }
            __syncthreads();

            // E89 TRUE RESIDUAL: d_decay = sum(dS * dtanh * S_{t-1})
            // For S_t = S_{t-1} + tanh(pre) where pre = decay * S_{t-1} + outer:
            // d_pre = dS_t * dtanh, so d_decay = sum(d_pre * S_{t-1}) = sum(dS * dtanh * S_{t-1})
            float d_decay_local = 0.0f;
            for (int idx = tid; idx < state_size; idx += blockDim.x) {
                d_decay_local += dS[idx] * dtanh[idx] * S[idx];
            }

            // Warp reduction using shuffle intrinsics
            #pragma unroll
            for (int offset = 16; offset >= 1; offset /= 2) {
                d_decay_local += __shfl_xor_sync(0xFFFFFFFF, d_decay_local, offset);
            }

            // Lane 0 of each warp writes partial sum to shared memory
            const int warp_id = tid / 32;
            const int lane_id = tid % 32;
            const int num_warps = (blockDim.x + 31) / 32;
            if (lane_id == 0) {
                warp_results[warp_id] = d_decay_local;
            }
            __syncthreads();

            // First warp reduces across all warp results
            float d_decay_accum = 0.0f;
            if (warp_id == 0) {
                float load_val = (tid < num_warps) ? warp_results[tid] : 0.0f;
                #pragma unroll
                for (int offset = 16; offset >= 1; offset /= 2) {
                    load_val += __shfl_xor_sync(0xFFFFFFFF, load_val, offset);
                }
                d_decay_accum = load_val;
            }
            __syncthreads();

            // d_k contribution from retrieved gradient
            if (tid < N_STATE) {
                float d_k_from_retrieved = 0.0f;
                for (int j = 0; j < HEAD_V_DIM; j++) {
                    d_k_from_retrieved += S[tid * HEAD_V_DIM + j] * (-d_delta[j]);
                }
                d_k[tid] += d_k_from_retrieved;
            }
            __syncthreads();

            // Write gradients for this timestep
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

            // E89 TRUE RESIDUAL: Update dS for next iteration
            // S_t = S_{t-1} + tanh(pre) where pre = decay * S_{t-1} + outer
            // dS_{t-1} = dS_t * (1 + dtanh * decay) + retrieval_gradient
            // The identity gradient (dS_t) flows directly, plus the gradient through pre
            for (int idx = tid; idx < state_size; idx += blockDim.x) {
                int i = idx / HEAD_V_DIM;
                int j = idx % HEAD_V_DIM;
                float identity_grad = dS[idx];  // Gradient through identity connection
                float pre_grad = dS[idx] * dtanh[idx] * decay_val;  // Gradient through pre
                float retrieval_grad = (-d_delta[j]) * k[i];
                dS[idx] = identity_grad + pre_grad + retrieval_grad;
            }
            __syncthreads();
        }
    }
}

// ============================================================================
// E89 FLA Hybrid Backward Kernel - Global Memory Version (Optimized)
// For large configurations (n_state=96/128, head_v_dim=128) that exceed shared memory limits
// S and dS are stored in per-block global memory instead of shared memory
// Uses segment-level caching for O(T) complexity instead of O(T * checkpoint_interval)
// ============================================================================

template<int N_STATE, int HEAD_V_DIM>
__global__ void E89ResidualStateBackwardKernel_GlobalMem_BF16(
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
    __nv_bfloat16* __restrict__ segment_cache,  // Extended: [B*H, checkpoint_interval, N_STATE*HEAD_V_DIM + N_STATE + HEAD_V_DIM + 1]
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

    // Segment cache offsets for this (batch, head) pair
    // Extended layout: [S_states, k_vectors, v_vectors, decay_values]
    int cache_entry_size = state_size + N_STATE + HEAD_V_DIM + 1;
    __nv_bfloat16* seg_cache_base = segment_cache + (size_t)block_idx * checkpoint_interval * cache_entry_size;

    // Sub-cache offsets within each segment
    __nv_bfloat16* S_cache_base = seg_cache_base;  // [checkpoint_interval * state_size]
    __nv_bfloat16* k_cache_base = seg_cache_base + (size_t)checkpoint_interval * state_size;  // [checkpoint_interval * N_STATE]
    __nv_bfloat16* v_cache_base = k_cache_base + (size_t)checkpoint_interval * N_STATE;  // [checkpoint_interval * HEAD_V_DIM]
    __nv_bfloat16* decay_cache_base = v_cache_base + (size_t)checkpoint_interval * HEAD_V_DIM;  // [checkpoint_interval * 1]

    // Shared memory for small buffers only
    extern __shared__ float shared_mem[];
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
    float* warp_results = d_delta + HEAD_V_DIM;  // [8 floats for warp reduction]

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
        int seg_len = t_end - t_start;

        // ================================================================
        // PHASE 1: Forward replay through segment, caching S_{t-1}, k, v, decay
        // ================================================================

        // Load checkpoint for this segment
        int cp_offset = (seg * B * H + b * H + h) * state_size;
        for (int i = tid; i < state_size; i += blockDim.x) {
            S[i] = __bfloat162float(S_checkpoints[cp_offset + i]);
        }
        __syncthreads();

        // Replay forward through entire segment, saving S_{t-1}, k, v, decay for each t
        __shared__ float decay_val_replay;
        for (int local_t = 0; local_t < seg_len; local_t++) {
            int t = t_start + local_t;

            // Save S_{t-1} to segment cache BEFORE the update
            __nv_bfloat16* S_cache_slot = S_cache_base + (size_t)local_t * state_size;
            for (int i = tid; i < state_size; i += blockDim.x) {
                S_cache_slot[i] = __float2bfloat16(S[i]);
            }

            // Load inputs for timestep t from global memory
            int k_offset = ((t * B + b) * H + h) * N_STATE;
            int v_offset = ((t * B + b) * H + h) * HEAD_V_DIM;
            int decay_offset = (t * B + b) * H + h;

            if (tid < N_STATE) {
                k[tid] = __bfloat162float(k_all[k_offset + tid]);
            }
            if (tid < HEAD_V_DIM) {
                v[tid] = __bfloat162float(v_all[v_offset + tid]);
            }
            __syncthreads();

            if (tid == 0) {
                decay_val_replay = __bfloat162float(decay_all[decay_offset]);
            }
            __syncthreads();

            // Cache k, v, decay for backward phase
            __nv_bfloat16* k_cache_slot = k_cache_base + (size_t)local_t * N_STATE;
            __nv_bfloat16* v_cache_slot = v_cache_base + (size_t)local_t * HEAD_V_DIM;
            if (tid < N_STATE) {
                k_cache_slot[tid] = __float2bfloat16(k[tid]);
            }
            if (tid < HEAD_V_DIM) {
                v_cache_slot[tid] = __float2bfloat16(v[tid]);
            }
            if (tid == 0) {
                decay_cache_base[local_t] = __float2bfloat16(decay_val_replay);
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

            // E89 RESIDUAL: S_t = decay * S_{t-1} + tanh(outer(delta, k))
            for (int idx = tid; idx < state_size; idx += blockDim.x) {
                int i = idx / HEAD_V_DIM;
                int j = idx % HEAD_V_DIM;
                float outer_term = delta[j] * k[i];
                S[idx] = decay_val_replay * S[idx] + tanhf(outer_term);
            }
            __syncthreads();
        }

        // ================================================================
        // PHASE 2: Backward pass through segment using cached states and inputs
        // ================================================================

        for (int local_t = seg_len - 1; local_t >= 0; local_t--) {
            int t = t_start + local_t;

            // Load cached S_{t-1} from segment cache
            __nv_bfloat16* S_cache_slot = S_cache_base + (size_t)local_t * state_size;
            for (int i = tid; i < state_size; i += blockDim.x) {
                S[i] = __bfloat162float(S_cache_slot[i]);
            }

            // Load cached k, v from segment cache (NOT from global memory)
            __nv_bfloat16* k_cache_slot = k_cache_base + (size_t)local_t * N_STATE;
            __nv_bfloat16* v_cache_slot = v_cache_base + (size_t)local_t * HEAD_V_DIM;
            if (tid < N_STATE) {
                k[tid] = __bfloat162float(k_cache_slot[tid]);
            }
            if (tid < HEAD_V_DIM) {
                v[tid] = __bfloat162float(v_cache_slot[tid]);
            }
            __syncthreads();

            // Load q from global memory (still needed, not cached)
            int k_offset = ((t * B + b) * H + h) * N_STATE;
            int v_offset = ((t * B + b) * H + h) * HEAD_V_DIM;
            int decay_offset = (t * B + b) * H + h;

            if (tid < N_STATE) {
                q[tid] = __bfloat162float(q_all[k_offset + tid]);
            }
            __syncthreads();

            // Compute retrieved and delta for this timestep
            if (tid < HEAD_V_DIM) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    sum += S[i * HEAD_V_DIM + tid] * k[i];
                }
                retrieved[tid] = sum;
                delta[tid] = v[tid] - retrieved[tid];
            }
            __syncthreads();

            // Load cached decay (NOT from global memory)
            __shared__ float decay_val;
            if (tid == 0) {
                decay_val = __bfloat162float(decay_cache_base[local_t]);
            }
            __syncthreads();

            // Backward through output
            if (tid < HEAD_V_DIM) {
                float d_out = __bfloat162float(d_output[v_offset + tid]);
                d_Sq[tid] = d_out;
            }
            __syncthreads();

            // d_q from output - E89 RESIDUAL: S_t = decay * S + tanh(outer)
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int j = 0; j < HEAD_V_DIM; j++) {
                    float outer_term = delta[j] * k[tid];
                    float pre = decay_val * S[tid * HEAD_V_DIM + j] + outer_term;
                    float S_t_ij = S[tid * HEAD_V_DIM + j] + tanhf(pre);  // True residual
                    sum += S_t_ij * d_Sq[j];
                }
                d_q[tid] = sum;
            }
            __syncthreads();

            // dS += outer(q, d_Sq)
            for (int idx = tid; idx < state_size; idx += blockDim.x) {
                int i = idx / HEAD_V_DIM;
                int j = idx % HEAD_V_DIM;
                dS[idx] += q[i] * d_Sq[j];
            }
            __syncthreads();

            // Backward through state update

            // E89 TRUE RESIDUAL: Compute d_delta and d_k contributions
            // S_t = S + tanh(decay * S + outer), so dtanh applies to (decay * S + outer)
            if (tid < HEAD_V_DIM) {
                float d_delta_local = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    float outer_term = delta[tid] * k[i];
                    float pre = decay_val * S[i * HEAD_V_DIM + tid] + outer_term;
                    float tanh_pre = tanhf(pre);
                    float dtanh = 1.0f - tanh_pre * tanh_pre;
                    float d_pre = dS[i * HEAD_V_DIM + tid] * dtanh;
                    d_delta_local += d_pre * k[i];
                }
                d_delta[tid] = d_delta_local;
            }
            __syncthreads();

            if (tid < N_STATE) {
                float d_k_local = 0.0f;
                for (int j = 0; j < HEAD_V_DIM; j++) {
                    float outer_term = delta[j] * k[tid];
                    float pre = decay_val * S[tid * HEAD_V_DIM + j] + outer_term;
                    float tanh_pre = tanhf(pre);
                    float dtanh = 1.0f - tanh_pre * tanh_pre;
                    float d_pre = dS[tid * HEAD_V_DIM + j] * dtanh;
                    d_k_local += d_pre * delta[j];
                }
                d_k[tid] = d_k_local;
            }
            __syncthreads();

            // E89 TRUE RESIDUAL: d_decay = sum(dS * dtanh * S_{t-1})
            float d_decay_local = 0.0f;
            for (int idx = tid; idx < state_size; idx += blockDim.x) {
                int i = idx / HEAD_V_DIM;
                int j = idx % HEAD_V_DIM;
                float outer_term = delta[j] * k[i];
                float pre = decay_val * S[idx] + outer_term;
                float tanh_pre = tanhf(pre);
                float dtanh = 1.0f - tanh_pre * tanh_pre;
                d_decay_local += dS[idx] * dtanh * S[idx];
            }

            // Warp reduction using shuffle intrinsics
            #pragma unroll
            for (int offset = 16; offset >= 1; offset /= 2) {
                d_decay_local += __shfl_xor_sync(0xFFFFFFFF, d_decay_local, offset);
            }

            // Lane 0 of each warp writes partial sum to shared memory
            const int warp_id = tid / 32;
            const int lane_id = tid % 32;
            const int num_warps = (blockDim.x + 31) / 32;
            if (lane_id == 0) {
                warp_results[warp_id] = d_decay_local;
            }
            __syncthreads();

            // First warp reduces across all warp results
            float d_decay_accum = 0.0f;
            if (warp_id == 0) {
                float load_val = (tid < num_warps) ? warp_results[tid] : 0.0f;
                #pragma unroll
                for (int offset = 16; offset >= 1; offset /= 2) {
                    load_val += __shfl_xor_sync(0xFFFFFFFF, load_val, offset);
                }
                d_decay_accum = load_val;
            }
            __syncthreads();

            // d_k contribution from retrieved gradient
            if (tid < N_STATE) {
                float d_k_from_retrieved = 0.0f;
                for (int j = 0; j < HEAD_V_DIM; j++) {
                    d_k_from_retrieved += S[tid * HEAD_V_DIM + j] * (-d_delta[j]);
                }
                d_k[tid] += d_k_from_retrieved;
            }
            __syncthreads();

            // Write gradients for this timestep
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

            // E89 TRUE RESIDUAL: Update dS for next iteration
            // S_t = S_{t-1} + tanh(pre) where pre = decay * S_{t-1} + outer
            // dS_{t-1} = dS_t * (1 + dtanh * decay) + retrieval_gradient
            for (int idx = tid; idx < state_size; idx += blockDim.x) {
                int i = idx / HEAD_V_DIM;
                int j = idx % HEAD_V_DIM;
                float outer_term = delta[j] * k[i];
                float pre = decay_val * S[idx] + outer_term;
                float tanh_pre = tanhf(pre);
                float dtanh = 1.0f - tanh_pre * tanh_pre;
                float identity_grad = dS[idx];
                float pre_grad = dS[idx] * dtanh * decay_val;
                float retrieval_grad = (-d_delta[j]) * k[i];
                dS[idx] = identity_grad + pre_grad + retrieval_grad;
            }
            __syncthreads();
        }
    }
}

// ============================================================================
// Dispatcher functions
// ============================================================================

void dispatch_e89_fla_hybrid_forward(
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
    #define DISPATCH_E89_FWD(N, V) do { \
        auto kernel = E89ResidualStateForwardKernel_BF16<N, V>; \
        if (shared_size > 48 * 1024) { \
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size); \
        } \
        kernel<<<num_blocks, threads, shared_size, stream>>>( \
            T, B, H, k_all, v_all, q_all, decay_all, S, output, \
            S_checkpoints, Sq_cache, checkpoint_interval); \
    } while(0)

    // Common configurations - ordered by n_state for clarity
    // n_state=4 (minimal state for extreme many-head experiments)
    if (n_state == 4 && head_v_dim == 4) { DISPATCH_E89_FWD(4, 4); }
    // n_state=8 (tiny state for many-head experiments)
    else if (n_state == 8 && head_v_dim == 8) { DISPATCH_E89_FWD(8, 8); }
    else if (n_state == 8 && head_v_dim == 16) { DISPATCH_E89_FWD(8, 16); }
    else if (n_state == 8 && head_v_dim == 32) { DISPATCH_E89_FWD(8, 32); }
    else if (n_state == 8 && head_v_dim == 64) { DISPATCH_E89_FWD(8, 64); }
    // n_state=16 (small state, low shared memory)
    else if (n_state == 16 && head_v_dim == 16) { DISPATCH_E89_FWD(16, 16); }
    else if (n_state == 16 && head_v_dim == 32) { DISPATCH_E89_FWD(16, 32); }
    else if (n_state == 16 && head_v_dim == 64) { DISPATCH_E89_FWD(16, 64); }
    else if (n_state == 16 && head_v_dim == 128) { DISPATCH_E89_FWD(16, 128); }
    // n_state=24
    else if (n_state == 24 && head_v_dim == 24) { DISPATCH_E89_FWD(24, 24); }
    else if (n_state == 24 && head_v_dim == 32) { DISPATCH_E89_FWD(24, 32); }
    else if (n_state == 24 && head_v_dim == 48) { DISPATCH_E89_FWD(24, 48); }
    else if (n_state == 24 && head_v_dim == 64) { DISPATCH_E89_FWD(24, 64); }
    else if (n_state == 24 && head_v_dim == 96) { DISPATCH_E89_FWD(24, 96); }
    else if (n_state == 24 && head_v_dim == 128) { DISPATCH_E89_FWD(24, 128); }
    // n_state=32
    else if (n_state == 32 && head_v_dim == 32) { DISPATCH_E89_FWD(32, 32); }
    else if (n_state == 32 && head_v_dim == 48) { DISPATCH_E89_FWD(32, 48); }
    else if (n_state == 32 && head_v_dim == 64) { DISPATCH_E89_FWD(32, 64); }
    else if (n_state == 32 && head_v_dim == 96) { DISPATCH_E89_FWD(32, 96); }
    else if (n_state == 32 && head_v_dim == 128) { DISPATCH_E89_FWD(32, 128); }
    // n_state=36 (for balanced configs)
    else if (n_state == 36 && head_v_dim == 36) { DISPATCH_E89_FWD(36, 36); }
    else if (n_state == 36 && head_v_dim == 48) { DISPATCH_E89_FWD(36, 48); }
    else if (n_state == 36 && head_v_dim == 64) { DISPATCH_E89_FWD(36, 64); }
    else if (n_state == 36 && head_v_dim == 72) { DISPATCH_E89_FWD(36, 72); }
    // n_state=40 (for balanced configs)
    else if (n_state == 40 && head_v_dim == 40) { DISPATCH_E89_FWD(40, 40); }
    else if (n_state == 40 && head_v_dim == 48) { DISPATCH_E89_FWD(40, 48); }
    else if (n_state == 40 && head_v_dim == 64) { DISPATCH_E89_FWD(40, 64); }
    else if (n_state == 40 && head_v_dim == 80) { DISPATCH_E89_FWD(40, 80); }
    // n_state=44 (for balanced configs)
    else if (n_state == 44 && head_v_dim == 44) { DISPATCH_E89_FWD(44, 44); }
    else if (n_state == 44 && head_v_dim == 48) { DISPATCH_E89_FWD(44, 48); }
    else if (n_state == 44 && head_v_dim == 64) { DISPATCH_E89_FWD(44, 64); }
    else if (n_state == 44 && head_v_dim == 88) { DISPATCH_E89_FWD(44, 88); }
    // n_state=48 (for proper scaling at 500M)
    else if (n_state == 48 && head_v_dim == 48) { DISPATCH_E89_FWD(48, 48); }
    else if (n_state == 48 && head_v_dim == 64) { DISPATCH_E89_FWD(48, 64); }
    else if (n_state == 48 && head_v_dim == 96) { DISPATCH_E89_FWD(48, 96); }
    // n_state=56 (for proper scaling at 500M)
    else if (n_state == 56 && head_v_dim == 56) { DISPATCH_E89_FWD(56, 56); }
    else if (n_state == 56 && head_v_dim == 64) { DISPATCH_E89_FWD(56, 64); }
    // n_state=64
    else if (n_state == 64 && head_v_dim == 32) { DISPATCH_E89_FWD(64, 32); }
    else if (n_state == 64 && head_v_dim == 64) { DISPATCH_E89_FWD(64, 64); }
    else if (n_state == 64 && head_v_dim == 128) { DISPATCH_E89_FWD(64, 128); }
    // n_state=72 (for h32n72 = 165,888 state/layer ≈ Mamba2)
    else if (n_state == 72 && head_v_dim == 72) { DISPATCH_E89_FWD(72, 72); }
    else if (n_state == 72 && head_v_dim == 64) { DISPATCH_E89_FWD(72, 64); }
    else if (n_state == 72 && head_v_dim == 96) { DISPATCH_E89_FWD(72, 96); }
    // n_state=80
    else if (n_state == 80 && head_v_dim == 80) { DISPATCH_E89_FWD(80, 80); }
    else if (n_state == 80 && head_v_dim == 64) { DISPATCH_E89_FWD(80, 64); }
    // n_state=88 (for balanced configs at various dims)
    else if (n_state == 88 && head_v_dim == 88) { DISPATCH_E89_FWD(88, 88); }
    else if (n_state == 88 && head_v_dim == 64) { DISPATCH_E89_FWD(88, 64); }
    // n_state=96
    else if (n_state == 96 && head_v_dim == 32) { DISPATCH_E89_FWD(96, 32); }
    else if (n_state == 96 && head_v_dim == 64) { DISPATCH_E89_FWD(96, 64); }
    else if (n_state == 96 && head_v_dim == 96) { DISPATCH_E89_FWD(96, 96); }
    else if (n_state == 96 && head_v_dim == 128) { DISPATCH_E89_FWD(96, 128); }
    // n_state=128
    else if (n_state == 128 && head_v_dim == 32) { DISPATCH_E89_FWD(128, 32); }
    else if (n_state == 128 && head_v_dim == 64) { DISPATCH_E89_FWD(128, 64); }
    else if (n_state == 128 && head_v_dim == 128) { DISPATCH_E89_FWD(128, 128); }
    else {
        fprintf(stderr, "E89 FLA Hybrid Forward: unsupported n_state=%d, head_v_dim=%d\n", n_state, head_v_dim);
    }

    #undef DISPATCH_E89_FWD
}

// Check if configuration needs global memory fallback
// Returns true if shared memory requirement exceeds ~96KB limit
inline bool e89_needs_global_mem_backward(int n_state, int head_v_dim) {
    int state_size = n_state * head_v_dim;
    // Shared mem: S, dS, S_t, dtanh (4*state_size) + k,q,d_k,d_q (4*n_state) + v,delta,retrieved,d_v,d_Sq,d_delta (6*head_v_dim) + warp_results (8)
    int shared_size = (4 * state_size + 4 * n_state + 6 * head_v_dim + 8) * sizeof(float);
    return shared_size > 96 * 1024;  // 96KB limit for safety margin
}

void dispatch_e89_fla_hybrid_backward(
    int T, int B, int H, int n_state, int head_v_dim,
    const __nv_bfloat16* k_all, const __nv_bfloat16* v_all,
    const __nv_bfloat16* q_all, const __nv_bfloat16* decay_all,
    const __nv_bfloat16* S_checkpoints, const __nv_bfloat16* Sq_cache,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* d_k_all, __nv_bfloat16* d_v_all,
    __nv_bfloat16* d_q_all, __nv_bfloat16* d_decay_all,
    float* S_global, float* dS_global,  // For global memory fallback (can be nullptr if not needed)
    __nv_bfloat16* segment_state_cache,  // [B*H, checkpoint_interval, n_state, head_v_dim] for O(T) caching
    int checkpoint_interval, cudaStream_t stream
) {
    int state_size = n_state * head_v_dim;
    // Shared memory: S, dS, S_t, dtanh (4*state_size) + k,q,d_k,d_q (4*n_state) + v,delta,retrieved,d_v,d_Sq,d_delta (6*head_v_dim) + warp_results (8)
    int shared_size = (4 * state_size + 4 * n_state + 6 * head_v_dim + 8) * sizeof(float);
    int threads = min(256, state_size);
    int num_blocks = B * H;

    // For configs requiring >48KB shared memory, we need to request extended shared memory
    // SM89 (Ada) supports up to 100KB per block
    // Configs exceeding 96KB use global memory fallback
    // Use cached kernel variant which caches k, v, decay during forward replay
    #define DISPATCH_E89_BWD(N, V) do { \
        auto kernel = E89ResidualStateBackwardKernel_Cached_BF16<N, V>; \
        if (shared_size > 48 * 1024) { \
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size); \
        } \
        kernel<<<num_blocks, threads, shared_size, stream>>>( \
            T, B, H, k_all, v_all, q_all, decay_all, \
            S_checkpoints, Sq_cache, d_output, \
            d_k_all, d_v_all, d_q_all, d_decay_all, \
            segment_state_cache, checkpoint_interval); \
    } while(0)

    // Global memory kernel dispatch - uses small buffers only in shared memory
    // Shared mem for global version: k,q,d_k,d_q (4*n_state) + v,delta,retrieved,d_v,d_Sq,d_delta (6*head_v_dim) + warp_results (8)
    #define DISPATCH_E89_BWD_GLOBAL(N, V) do { \
        int global_shared_size = (4 * N + 6 * V + 8) * sizeof(float); \
        E89ResidualStateBackwardKernel_GlobalMem_BF16<N, V><<<num_blocks, threads, global_shared_size, stream>>>( \
            T, B, H, k_all, v_all, q_all, decay_all, \
            S_checkpoints, Sq_cache, d_output, \
            d_k_all, d_v_all, d_q_all, d_decay_all, \
            S_global, dS_global, segment_state_cache, checkpoint_interval); \
    } while(0)

    // n_state=4 (minimal state for extreme many-head experiments)
    if (n_state == 4 && head_v_dim == 4) { DISPATCH_E89_BWD(4, 4); }
    // n_state=8 (tiny state for many-head experiments)
    else if (n_state == 8 && head_v_dim == 8) { DISPATCH_E89_BWD(8, 8); }
    else if (n_state == 8 && head_v_dim == 16) { DISPATCH_E89_BWD(8, 16); }
    else if (n_state == 8 && head_v_dim == 32) { DISPATCH_E89_BWD(8, 32); }
    else if (n_state == 8 && head_v_dim == 64) { DISPATCH_E89_BWD(8, 64); }
    // n_state=16 (small state, low shared memory)
    else if (n_state == 16 && head_v_dim == 16) { DISPATCH_E89_BWD(16, 16); }
    else if (n_state == 16 && head_v_dim == 32) { DISPATCH_E89_BWD(16, 32); }
    else if (n_state == 16 && head_v_dim == 64) { DISPATCH_E89_BWD(16, 64); }
    else if (n_state == 16 && head_v_dim == 128) { DISPATCH_E89_BWD(16, 128); }
    // n_state=24
    else if (n_state == 24 && head_v_dim == 24) { DISPATCH_E89_BWD(24, 24); }
    else if (n_state == 24 && head_v_dim == 32) { DISPATCH_E89_BWD(24, 32); }
    else if (n_state == 24 && head_v_dim == 48) { DISPATCH_E89_BWD(24, 48); }
    else if (n_state == 24 && head_v_dim == 64) { DISPATCH_E89_BWD(24, 64); }
    else if (n_state == 24 && head_v_dim == 96) { DISPATCH_E89_BWD(24, 96); }
    else if (n_state == 24 && head_v_dim == 128) { DISPATCH_E89_BWD(24, 128); }
    // n_state=32
    else if (n_state == 32 && head_v_dim == 32) { DISPATCH_E89_BWD(32, 32); }
    else if (n_state == 32 && head_v_dim == 48) { DISPATCH_E89_BWD(32, 48); }  // ~40KB, OK
    else if (n_state == 32 && head_v_dim == 64) { DISPATCH_E89_BWD(32, 64); }
    else if (n_state == 32 && head_v_dim == 96) { DISPATCH_E89_BWD(32, 96); }  // ~54KB, OK
    else if (n_state == 32 && head_v_dim == 128) { DISPATCH_E89_BWD(32, 128); }
    // n_state=36 (for balanced configs)
    else if (n_state == 36 && head_v_dim == 36) { DISPATCH_E89_BWD(36, 36); }
    else if (n_state == 36 && head_v_dim == 48) { DISPATCH_E89_BWD(36, 48); }
    else if (n_state == 36 && head_v_dim == 64) { DISPATCH_E89_BWD(36, 64); }
    else if (n_state == 36 && head_v_dim == 72) { DISPATCH_E89_BWD(36, 72); }
    // n_state=40 (for balanced configs)
    else if (n_state == 40 && head_v_dim == 40) { DISPATCH_E89_BWD(40, 40); }
    else if (n_state == 40 && head_v_dim == 48) { DISPATCH_E89_BWD(40, 48); }
    else if (n_state == 40 && head_v_dim == 64) { DISPATCH_E89_BWD(40, 64); }
    else if (n_state == 40 && head_v_dim == 80) { DISPATCH_E89_BWD(40, 80); }
    // n_state=44 (for balanced configs)
    else if (n_state == 44 && head_v_dim == 44) { DISPATCH_E89_BWD(44, 44); }
    else if (n_state == 44 && head_v_dim == 48) { DISPATCH_E89_BWD(44, 48); }
    else if (n_state == 44 && head_v_dim == 64) { DISPATCH_E89_BWD(44, 64); }
    else if (n_state == 44 && head_v_dim == 88) { DISPATCH_E89_BWD(44, 88); }
    // n_state=48 (for proper scaling at 500M)
    else if (n_state == 48 && head_v_dim == 48) { DISPATCH_E89_BWD(48, 48); }
    else if (n_state == 48 && head_v_dim == 64) { DISPATCH_E89_BWD(48, 64); }
    else if (n_state == 48 && head_v_dim == 96) { DISPATCH_E89_BWD(48, 96); }
    // n_state=56 (for proper scaling at 500M)
    else if (n_state == 56 && head_v_dim == 56) { DISPATCH_E89_BWD(56, 56); }
    else if (n_state == 56 && head_v_dim == 64) { DISPATCH_E89_BWD(56, 64); }
    // n_state=64
    else if (n_state == 64 && head_v_dim == 32) { DISPATCH_E89_BWD(64, 32); }
    else if (n_state == 64 && head_v_dim == 64) { DISPATCH_E89_BWD(64, 64); }
    else if (n_state == 64 && head_v_dim == 128) { DISPATCH_E89_BWD(64, 128); }
    // n_state=72 (for h32n72 = 165,888 state/layer ≈ Mamba2)
    else if (n_state == 72 && head_v_dim == 72) { DISPATCH_E89_BWD(72, 72); }
    else if (n_state == 72 && head_v_dim == 64) { DISPATCH_E89_BWD(72, 64); }
    else if (n_state == 72 && head_v_dim == 96) { DISPATCH_E89_BWD(72, 96); }
    // n_state=80 (~103KB shared for 80x80, use global memory)
    else if (n_state == 80 && head_v_dim == 80) {
        DISPATCH_E89_BWD_GLOBAL(80, 80);
    }
    else if (n_state == 80 && head_v_dim == 64) { DISPATCH_E89_BWD(80, 64); }  // ~68KB, OK
    // n_state=88 (~124KB shared for 88x88, use global memory)
    else if (n_state == 88 && head_v_dim == 88) {
        DISPATCH_E89_BWD_GLOBAL(88, 88);
    }
    else if (n_state == 88 && head_v_dim == 64) { DISPATCH_E89_BWD(88, 64); }  // ~70KB, OK
    // n_state=96 (~148KB shared for 96x96, use global memory)
    else if (n_state == 96 && head_v_dim == 32) { DISPATCH_E89_BWD(96, 32); }  // ~49KB, OK
    else if (n_state == 96 && head_v_dim == 64) { DISPATCH_E89_BWD(96, 64); }  // ~80KB, borderline
    else if (n_state == 96 && head_v_dim == 96) {
        DISPATCH_E89_BWD_GLOBAL(96, 96);  // ~148KB shared, needs global
    }
    else if (n_state == 96 && head_v_dim == 128) {
        DISPATCH_E89_BWD_GLOBAL(96, 128);
    }
    // n_state=128
    else if (n_state == 128 && head_v_dim == 32) { DISPATCH_E89_BWD(128, 32); }
    else if (n_state == 128 && head_v_dim == 64) { DISPATCH_E89_BWD(128, 64); }
    else if (n_state == 128 && head_v_dim == 128) {
        // ~132KB shared needed, use global memory fallback
        DISPATCH_E89_BWD_GLOBAL(128, 128);
    }
    else {
        fprintf(stderr, "E89 FLA Hybrid Backward: unsupported n_state=%d, head_v_dim=%d\n", n_state, head_v_dim);
    }

    #undef DISPATCH_E89_BWD
    #undef DISPATCH_E89_BWD_GLOBAL
}

// ============================================================================
// E89ResidualStateForward Implementation
// ============================================================================

template<typename DataT>
E89ResidualStateForward<DataT>::E89ResidualStateForward(
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
void E89ResidualStateForward<DataT>::Run(
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

    int num_checkpoints = (T + E89_CHECKPOINT_INTERVAL - 1) / E89_CHECKPOINT_INTERVAL + 1;
    DataT* s_checkpoints = S_cache;
    DataT* sq_cache = S_cache + num_checkpoints * B * H * n * v_dim;

    dispatch_e89_fla_hybrid_forward(
        T, B, H, n, v_dim,
        (const __nv_bfloat16*)k, (const __nv_bfloat16*)v,
        (const __nv_bfloat16*)q, (const __nv_bfloat16*)decay,
        (__nv_bfloat16*)S, (__nv_bfloat16*)output,
        (__nv_bfloat16*)s_checkpoints, (__nv_bfloat16*)sq_cache,
        E89_CHECKPOINT_INTERVAL, stream_);
}

template struct E89ResidualStateForward<__nv_bfloat16>;

// ============================================================================
// E89ResidualStateBackward Implementation
// ============================================================================

template<typename DataT>
E89ResidualStateBackward<DataT>::E89ResidualStateBackward(
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
      dS_global_(nullptr),
      segment_state_cache_(nullptr) {
    // Check if we need global memory fallback for S/dS
    use_global_mem_ = e89_needs_global_mem_backward(n_state, head_v_dim);
    if (use_global_mem_) {
        // Allocate global memory for S and dS: [B*H, n_state, head_v_dim] each
        size_t state_size = batch_size * n_heads * n_state * head_v_dim * sizeof(float);
        cudaMalloc(&S_global_, state_size);
        cudaMalloc(&dS_global_, state_size);
    }

    // Allocate extended segment cache: [B*H, checkpoint_interval * (n_state*head_v_dim + n_state + head_v_dim + 1)]
    // This enables O(T) backward instead of O(T * checkpoint_interval)
    // Extended layout per (batch, head):
    //   [0..checkpoint_interval-1]: S_{t-1} states [n_state * head_v_dim each]
    //   [checkpoint_interval * state_size..]: k vectors [n_state each]
    //   [...]: v vectors [head_v_dim each]
    //   [...]: decay values [1 each]
    size_t state_size_elem = (size_t)n_state * head_v_dim;
    size_t cache_entry_size = state_size_elem + n_state + head_v_dim + 1;  // S + k + v + decay
    size_t seg_cache_size = (size_t)batch_size * n_heads * E89_CHECKPOINT_INTERVAL * cache_entry_size * sizeof(DataT);
    cudaMalloc(&segment_state_cache_, seg_cache_size);
}

template<typename DataT>
E89ResidualStateBackward<DataT>::~E89ResidualStateBackward() {
    if (S_global_) {
        cudaFree(S_global_);
        S_global_ = nullptr;
    }
    if (dS_global_) {
        cudaFree(dS_global_);
        dS_global_ = nullptr;
    }
    if (segment_state_cache_) {
        cudaFree(segment_state_cache_);
        segment_state_cache_ = nullptr;
    }
}

template<typename DataT>
void E89ResidualStateBackward<DataT>::Run(
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

    dispatch_e89_fla_hybrid_backward(
        T, B, H, n, v_dim,
        (const __nv_bfloat16*)k, (const __nv_bfloat16*)v,
        (const __nv_bfloat16*)q, (const __nv_bfloat16*)decay,
        (const __nv_bfloat16*)S_checkpoints, (const __nv_bfloat16*)Sq_cache,
        (const __nv_bfloat16*)d_output,
        (__nv_bfloat16*)d_k, (__nv_bfloat16*)d_v,
        (__nv_bfloat16*)d_q, (__nv_bfloat16*)d_decay,
        S_global_, dS_global_,  // Pass global memory buffers (nullptr if not using global fallback)
        (__nv_bfloat16*)segment_state_cache_,  // Segment cache for O(T) backward
        E89_CHECKPOINT_INTERVAL, stream_);
}

template struct E89ResidualStateBackward<__nv_bfloat16>;

}  // namespace elman
