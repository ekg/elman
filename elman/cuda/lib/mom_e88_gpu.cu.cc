/**
 * MoM E88 CUDA Kernel - Mixture of Memory (Per-Slot State Design)
 *
 * Sparse routing to memory heads (MoE-style for memory).
 * Each token routes to top-K heads instead of all H heads.
 *
 * CRITICAL DESIGN: Per-slot state
 * - State shape: [B, K, n_state, head_v_dim]
 * - Each slot maintains its own independent state trajectory
 * - No shared state between slots = no race conditions
 * - Head indices only affect which k/v/q/decay to READ, not which state to write
 *
 * Per slot i at timestep t:
 *   h = head_indices[t, b, i]
 *   k_h = k_all[t, b, h]       [n_state]
 *   v_h = v_all[t, b, h]       [head_v_dim]
 *   q_h = q_all[t, b, h]       [n_state]
 *   decay_h = decay_all[t, b, h]  [scalar]
 *
 *   retrieved = S_slot @ k_h    [head_v_dim]
 *   delta = v_h - retrieved     [head_v_dim]
 *   S_slot = tanh(decay_h * S_slot + outer(delta, k_h))  [n_state x head_v_dim]
 *   out_slot = router_weight * (S_slot^T @ q_h)  [head_v_dim]
 *
 * Output: sum of weighted slot outputs for each batch
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include "hasty/elman_ladder.h"

#define MOM_E88_CHECKPOINT_INTERVAL 16

namespace elman {

__device__ __forceinline__ float mom_e88_tanh(float x) {
    return tanhf(x);
}

// ============================================================================
// MoM E88 Forward Kernel (Per-Slot State Design)
// Each block handles one (batch, slot) pair
// State is per-slot, no shared state between slots
// ============================================================================

template<int N_STATE, int HEAD_V_DIM>
__global__ void MoME88ForwardKernel_BF16(
    int T,                    // Sequence length
    int B,                    // Batch size
    int H,                    // Total number of heads (for indexing k/v/q/decay)
    int K,                    // Top-k (number of slots)
    const __nv_bfloat16* __restrict__ k_all,      // [T, B, H, N_STATE]
    const __nv_bfloat16* __restrict__ v_all,      // [T, B, H, HEAD_V_DIM]
    const __nv_bfloat16* __restrict__ q_all,      // [T, B, H, N_STATE]
    const __nv_bfloat16* __restrict__ decay_all,  // [T, B, H]
    const int* __restrict__ head_indices,         // [T, B, K] - which head for each slot
    const __nv_bfloat16* __restrict__ router_weights, // [T, B, K] - routing weights
    __nv_bfloat16* __restrict__ S,                // [B, K, N_STATE, HEAD_V_DIM] - per-slot states
    __nv_bfloat16* __restrict__ output,           // [T, B, HEAD_V_DIM] - combined output
    __nv_bfloat16* __restrict__ S_checkpoints,    // [num_checkpoints, B, K, N_STATE, HEAD_V_DIM]
    __nv_bfloat16* __restrict__ Sq_cache,         // [T, B, K, HEAD_V_DIM] - per-slot weighted outputs
    int checkpoint_interval
) {
    // Block handles one (batch, slot) pair
    int block_idx = blockIdx.x;
    int b = block_idx / K;
    int slot = block_idx % K;
    if (b >= B) return;

    extern __shared__ float shared_mem[];
    float* S_shared = shared_mem;
    float* k_shared = S_shared + N_STATE * HEAD_V_DIM;
    float* v_shared = k_shared + N_STATE;
    float* q_shared = v_shared + HEAD_V_DIM;
    float* retrieved = q_shared + N_STATE;

    int tid = threadIdx.x;
    constexpr int state_size = N_STATE * HEAD_V_DIM;

    __shared__ float decay_val;
    __shared__ int current_head;
    __shared__ float router_weight;

    // State offset for this (batch, slot) pair - per-slot state!
    int state_offset = (b * K + slot) * state_size;

    // Load initial state for this slot
    for (int i = tid; i < state_size; i += blockDim.x) {
        S_shared[i] = __bfloat162float(S[state_offset + i]);
    }
    __syncthreads();

    // Save initial checkpoint (t=0)
    int cp_base_offset = (0 * B * K + b * K + slot) * state_size;  // checkpoint 0
    for (int i = tid; i < state_size; i += blockDim.x) {
        S_checkpoints[cp_base_offset + i] = __float2bfloat16(S_shared[i]);
    }
    __syncthreads();

    for (int t = 0; t < T; t++) {
        // Get head index and router weight for this timestep and slot
        int idx_offset = (t * B + b) * K + slot;
        if (tid == 0) {
            current_head = head_indices[idx_offset];
            router_weight = __bfloat162float(router_weights[idx_offset]);
        }
        __syncthreads();

        int h = current_head;

        // Load k, v, q, decay for this timestep and the selected head
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
        if (tid == 0) {
            decay_val = __bfloat162float(decay_all[decay_offset]);
        }
        __syncthreads();

        // retrieved = S @ k (S stored as [N_STATE, HEAD_V_DIM])
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
        // where delta = v - retrieved
        for (int idx = tid; idx < state_size; idx += blockDim.x) {
            int i = idx / HEAD_V_DIM;
            int j = idx % HEAD_V_DIM;
            float delta_j = v_shared[j] - retrieved[j];
            S_shared[idx] = mom_e88_tanh(decay_val * S_shared[idx] + delta_j * k_shared[i]);
        }
        __syncthreads();

        // Save checkpoint after update (so checkpoint t contains S after processing timestep t-1)
        if ((t + 1) % checkpoint_interval == 0) {
            int cp_idx = (t + 1) / checkpoint_interval;
            int cp_offset = (cp_idx * B * K + b * K + slot) * state_size;
            for (int i = tid; i < state_size; i += blockDim.x) {
                S_checkpoints[cp_offset + i] = __float2bfloat16(S_shared[i]);
            }
        }

        // Compute output: Sq = S^T @ q, scaled by router weight
        int sq_offset = ((t * B + b) * K + slot) * HEAD_V_DIM;
        if (tid < HEAD_V_DIM) {
            float Sq = 0.0f;
            #pragma unroll 8
            for (int i = 0; i < N_STATE; i++) {
                Sq += S_shared[i * HEAD_V_DIM + tid] * q_shared[i];
            }
            // Cache the weighted output
            Sq_cache[sq_offset + tid] = __float2bfloat16(Sq * router_weight);
        }
        __syncthreads();
    }

    // Write final state back to global memory (per-slot state)
    for (int i = tid; i < state_size; i += blockDim.x) {
        S[state_offset + i] = __float2bfloat16(S_shared[i]);
    }
}

// ============================================================================
// Output Reduction Kernel
// Sums the weighted outputs from all K slots for each (batch, timestep) pair
// ============================================================================

template<int HEAD_V_DIM>
__global__ void MoME88OutputReductionKernel_BF16(
    int T,
    int B,
    int K,
    const __nv_bfloat16* __restrict__ Sq_cache,  // [T, B, K, HEAD_V_DIM]
    __nv_bfloat16* __restrict__ output           // [T, B, HEAD_V_DIM]
) {
    // Each block handles one (timestep, batch) pair
    int block_idx = blockIdx.x;
    int t = block_idx / B;
    int b = block_idx % B;
    if (t >= T || b >= B) return;

    int tid = threadIdx.x;

    // Sum weighted outputs from all K slots
    if (tid < HEAD_V_DIM) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            int sq_offset = ((t * B + b) * K + k) * HEAD_V_DIM + tid;
            sum += __bfloat162float(Sq_cache[sq_offset]);
        }
        int out_offset = (t * B + b) * HEAD_V_DIM + tid;
        output[out_offset] = __float2bfloat16(sum);
    }
}

// ============================================================================
// Dispatcher functions
// ============================================================================

void dispatch_mom_e88_forward(
    int T, int B, int H, int K, int n_state, int head_v_dim,
    const __nv_bfloat16* k_all, const __nv_bfloat16* v_all,
    const __nv_bfloat16* q_all, const __nv_bfloat16* decay_all,
    const int* head_indices, const __nv_bfloat16* router_weights,
    __nv_bfloat16* S, __nv_bfloat16* output,
    __nv_bfloat16* S_checkpoints, __nv_bfloat16* Sq_cache,
    int checkpoint_interval, cudaStream_t stream
) {
    int state_size = n_state * head_v_dim;
    int shared_size = (state_size + n_state + head_v_dim + n_state + head_v_dim) * sizeof(float);
    int threads = min(256, state_size);
    int num_blocks = B * K;  // One block per (batch, slot) pair

    #define DISPATCH_MOM_FWD(N, V) do { \
        auto kernel = MoME88ForwardKernel_BF16<N, V>; \
        if (shared_size > 48 * 1024) { \
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size); \
        } \
        kernel<<<num_blocks, threads, shared_size, stream>>>( \
            T, B, H, K, k_all, v_all, q_all, decay_all, \
            head_indices, router_weights, S, output, \
            S_checkpoints, Sq_cache, checkpoint_interval); \
    } while(0)

    // Common configurations (matching E88 supported configs)
    if (n_state == 16 && head_v_dim == 16) { DISPATCH_MOM_FWD(16, 16); }
    else if (n_state == 32 && head_v_dim == 32) { DISPATCH_MOM_FWD(32, 32); }
    else if (n_state == 32 && head_v_dim == 64) { DISPATCH_MOM_FWD(32, 64); }
    else if (n_state == 48 && head_v_dim == 48) { DISPATCH_MOM_FWD(48, 48); }
    else if (n_state == 64 && head_v_dim == 64) { DISPATCH_MOM_FWD(64, 64); }
    else {
        fprintf(stderr, "MoM E88 Forward: unsupported n_state=%d, head_v_dim=%d\n", n_state, head_v_dim);
        return;
    }

    #undef DISPATCH_MOM_FWD

    // Run output reduction kernel to sum weighted slot outputs
    int reduction_threads = min(256, head_v_dim);
    int reduction_blocks = T * B;

    #define DISPATCH_MOM_REDUCE(V) do { \
        MoME88OutputReductionKernel_BF16<V><<<reduction_blocks, reduction_threads, 0, stream>>>( \
            T, B, K, Sq_cache, output); \
    } while(0)

    if (head_v_dim == 16) { DISPATCH_MOM_REDUCE(16); }
    else if (head_v_dim == 32) { DISPATCH_MOM_REDUCE(32); }
    else if (head_v_dim == 48) { DISPATCH_MOM_REDUCE(48); }
    else if (head_v_dim == 64) { DISPATCH_MOM_REDUCE(64); }

    #undef DISPATCH_MOM_REDUCE
}

// ============================================================================
// MoM E88 Backward Kernel (Per-Slot State Design)
// Each block handles one (batch, slot) pair
// Uses segment-level caching for efficient gradient computation
//
// Forward equations (for reference):
//   h = head_indices[t, b, slot]
//   retrieved = S_{t-1}.T @ k_h        [HEAD_V_DIM]
//   delta = v_h - retrieved            [HEAD_V_DIM]
//   S_t = tanh(decay_h * S_{t-1} + outer(k_h, delta))  [N_STATE x HEAD_V_DIM]
//   Sq = S_t.T @ q_h                   [HEAD_V_DIM]
//   out_slot = router_weight * Sq      [HEAD_V_DIM]
//
// Backward equations:
//   d_Sq = d_output * router_weight    (from reduction)
//   d_router_weight = dot(Sq, d_output)
//   d_q[i] = sum_j S_t[i,j] * d_Sq[j]
//   dS_t[i,j] += q[i] * d_Sq[j]
//   d_pre = dS_t * (1 - S_t^2)         (tanh derivative)
//   d_decay = sum(d_pre * S_{t-1})
//   d_delta[j] = sum_i d_pre[i,j] * k[i]
//   d_k[i] = sum_j d_pre[i,j] * delta[j]
//   d_v = d_delta
//   d_retrieved = -d_delta
//   d_k[i] += sum_j S_{t-1}[i,j] * d_retrieved[j]
//   dS_{t-1}[i,j] = d_pre[i,j] * decay + k[i] * d_retrieved[j]
// ============================================================================

template<int N_STATE, int HEAD_V_DIM>
__global__ void MoME88BackwardKernel_BF16(
    int T,
    int B,
    int H,
    int K,
    const __nv_bfloat16* __restrict__ k_all,       // [T, B, H, N_STATE]
    const __nv_bfloat16* __restrict__ v_all,       // [T, B, H, HEAD_V_DIM]
    const __nv_bfloat16* __restrict__ q_all,       // [T, B, H, N_STATE]
    const __nv_bfloat16* __restrict__ decay_all,   // [T, B, H]
    const int* __restrict__ head_indices,          // [T, B, K]
    const __nv_bfloat16* __restrict__ router_weights, // [T, B, K]
    const __nv_bfloat16* __restrict__ S_checkpoints,  // [num_checkpoints, B, K, N_STATE, HEAD_V_DIM]
    const __nv_bfloat16* __restrict__ Sq_cache,    // [T, B, K, HEAD_V_DIM] - weighted from forward
    const __nv_bfloat16* __restrict__ d_output,    // [T, B, HEAD_V_DIM]
    __nv_bfloat16* __restrict__ d_k_all,           // [T, B, H, N_STATE]
    __nv_bfloat16* __restrict__ d_v_all,           // [T, B, H, HEAD_V_DIM]
    __nv_bfloat16* __restrict__ d_q_all,           // [T, B, H, N_STATE]
    __nv_bfloat16* __restrict__ d_decay_all,       // [T, B, H]
    __nv_bfloat16* __restrict__ d_router_weights,  // [T, B, K]
    __nv_bfloat16* __restrict__ segment_cache,     // [B*K, checkpoint_interval, state_size + N_STATE + HEAD_V_DIM + 1]
    int checkpoint_interval
) {
    int block_idx = blockIdx.x;
    int b = block_idx / K;
    int slot = block_idx % K;
    if (b >= B) return;

    constexpr int state_size = N_STATE * HEAD_V_DIM;

    // Shared memory layout:
    // S: N_STATE * HEAD_V_DIM (S_{t-1})
    // dS: N_STATE * HEAD_V_DIM
    // S_t: N_STATE * HEAD_V_DIM (cached tanh result)
    // dtanh: N_STATE * HEAD_V_DIM (1 - S_t^2)
    // k: N_STATE
    // v: HEAD_V_DIM
    // q: N_STATE
    // delta: HEAD_V_DIM
    // retrieved: HEAD_V_DIM
    // d_k: N_STATE
    // d_q: N_STATE
    // d_Sq: HEAD_V_DIM
    // d_delta: HEAD_V_DIM
    // warp_results: 8 floats for reduction
    extern __shared__ float shared_mem[];
    float* S = shared_mem;
    float* dS = S + state_size;
    float* S_t = dS + state_size;
    float* dtanh_arr = S_t + state_size;
    float* k = dtanh_arr + state_size;
    float* v = k + N_STATE;
    float* q = v + HEAD_V_DIM;
    float* delta = q + N_STATE;
    float* retrieved = delta + HEAD_V_DIM;
    float* d_k = retrieved + HEAD_V_DIM;
    float* d_q = d_k + N_STATE;
    float* d_Sq = d_q + N_STATE;
    float* d_delta = d_Sq + HEAD_V_DIM;
    float* warp_results = d_delta + HEAD_V_DIM;

    int tid = threadIdx.x;

    // Segment cache for this (batch, slot) pair
    int cache_entry_size = state_size + N_STATE + HEAD_V_DIM + 1;
    __nv_bfloat16* seg_cache_base = segment_cache + (size_t)block_idx * checkpoint_interval * cache_entry_size;
    __nv_bfloat16* S_cache_base = seg_cache_base;
    __nv_bfloat16* k_cache_base = seg_cache_base + (size_t)checkpoint_interval * state_size;
    __nv_bfloat16* v_cache_base = k_cache_base + (size_t)checkpoint_interval * N_STATE;
    __nv_bfloat16* decay_cache_base = v_cache_base + (size_t)checkpoint_interval * HEAD_V_DIM;

    // Shared variables
    __shared__ float decay_val;
    __shared__ int current_head;
    __shared__ float router_weight_val;

    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = (blockDim.x + 31) / 32;

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

        // Load checkpoint for this segment (per-slot checkpoint!)
        int cp_offset = (seg * B * K + b * K + slot) * state_size;
        for (int i = tid; i < state_size; i += blockDim.x) {
            S[i] = __bfloat162float(S_checkpoints[cp_offset + i]);
        }
        __syncthreads();

        __shared__ float decay_val_replay;
        for (int local_t = 0; local_t < seg_len; local_t++) {
            int t = t_start + local_t;

            // Save S_{t-1} to segment cache BEFORE the update
            __nv_bfloat16* S_cache_slot = S_cache_base + (size_t)local_t * state_size;
            for (int i = tid; i < state_size; i += blockDim.x) {
                S_cache_slot[i] = __float2bfloat16(S[i]);
            }

            // Get head index for this timestep and slot
            int idx_offset = (t * B + b) * K + slot;
            if (tid == 0) {
                current_head = head_indices[idx_offset];
            }
            __syncthreads();

            int h = current_head;

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

            // Compute retrieved = S.T @ k
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

            // Update S: S_t = tanh(decay * S_{t-1} + outer(k, delta))
            for (int idx = tid; idx < state_size; idx += blockDim.x) {
                int i = idx / HEAD_V_DIM;
                int j = idx % HEAD_V_DIM;
                float update = decay_val_replay * S[idx] + k[i] * delta[j];
                S[idx] = mom_e88_tanh(update);
            }
            __syncthreads();
        }

        // ================================================================
        // PHASE 2: Backward pass through segment using cached states
        // ================================================================

        for (int local_t = seg_len - 1; local_t >= 0; local_t--) {
            int t = t_start + local_t;

            // Load cached S_{t-1}
            __nv_bfloat16* S_cache_slot = S_cache_base + (size_t)local_t * state_size;
            for (int i = tid; i < state_size; i += blockDim.x) {
                S[i] = __bfloat162float(S_cache_slot[i]);
            }

            // Load cached k, v
            __nv_bfloat16* k_cache_slot = k_cache_base + (size_t)local_t * N_STATE;
            __nv_bfloat16* v_cache_slot = v_cache_base + (size_t)local_t * HEAD_V_DIM;
            if (tid < N_STATE) {
                k[tid] = __bfloat162float(k_cache_slot[tid]);
            }
            if (tid < HEAD_V_DIM) {
                v[tid] = __bfloat162float(v_cache_slot[tid]);
            }
            __syncthreads();

            // Get head index and router weight for this timestep
            int idx_offset = (t * B + b) * K + slot;
            if (tid == 0) {
                current_head = head_indices[idx_offset];
                router_weight_val = __bfloat162float(router_weights[idx_offset]);
                decay_val = __bfloat162float(decay_cache_base[local_t]);
            }
            __syncthreads();

            int h = current_head;

            // Load q from global memory
            int k_offset = ((t * B + b) * H + h) * N_STATE;
            int v_offset = ((t * B + b) * H + h) * HEAD_V_DIM;
            int decay_offset = (t * B + b) * H + h;

            if (tid < N_STATE) {
                q[tid] = __bfloat162float(q_all[k_offset + tid]);
            }
            __syncthreads();

            // Recompute retrieved, delta, and S_t with dtanh
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

            // Compute S_t and dtanh
            for (int idx = tid; idx < state_size; idx += blockDim.x) {
                int i = idx / HEAD_V_DIM;
                int j = idx % HEAD_V_DIM;
                float pre_tanh = decay_val * S[idx] + k[i] * delta[j];
                float tanh_val = mom_e88_tanh(pre_tanh);
                S_t[idx] = tanh_val;
                dtanh_arr[idx] = 1.0f - tanh_val * tanh_val;
            }
            __syncthreads();

            // === Backward through output ===
            // out_slot = router_weight * (S_t.T @ q)
            // d_Sq = d_output (gradients flow from all slots equally since reduction is sum)
            // d_router_weight = dot(Sq, d_output)

            int out_offset = (t * B + b) * HEAD_V_DIM;
            if (tid < HEAD_V_DIM) {
                d_Sq[tid] = __bfloat162float(d_output[out_offset + tid]);
            }
            __syncthreads();

            // Compute d_router_weight = dot(S_t.T @ q, d_output)
            // First compute Sq = S_t.T @ q
            float d_rw_local = 0.0f;
            if (tid < HEAD_V_DIM) {
                float Sq_j = 0.0f;
                #pragma unroll 8
                for (int i = 0; i < N_STATE; i++) {
                    Sq_j += S_t[i * HEAD_V_DIM + tid] * q[i];
                }
                d_rw_local = Sq_j * d_Sq[tid];
            }

            // Warp reduction for d_router_weight
            #pragma unroll
            for (int offset = 16; offset >= 1; offset /= 2) {
                d_rw_local += __shfl_xor_sync(0xFFFFFFFF, d_rw_local, offset);
            }

            if (lane_id == 0) {
                warp_results[warp_id] = d_rw_local;
            }
            __syncthreads();

            float d_rw_accum = 0.0f;
            if (warp_id == 0) {
                float load_val = (tid < num_warps) ? warp_results[tid] : 0.0f;
                #pragma unroll
                for (int offset = 16; offset >= 1; offset /= 2) {
                    load_val += __shfl_xor_sync(0xFFFFFFFF, load_val, offset);
                }
                d_rw_accum = load_val;
            }
            __syncthreads();

            // Write d_router_weight
            if (tid == 0) {
                d_router_weights[idx_offset] = __float2bfloat16(d_rw_accum);
            }

            // Scale d_Sq by router_weight for downstream gradients
            if (tid < HEAD_V_DIM) {
                d_Sq[tid] *= router_weight_val;
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

            // dS_t += outer(q, d_Sq) for the output computation
            for (int idx = tid; idx < state_size; idx += blockDim.x) {
                int i = idx / HEAD_V_DIM;
                int j = idx % HEAD_V_DIM;
                dS[idx] += q[i] * d_Sq[j];
            }
            __syncthreads();

            // === Backward through state update ===
            // S_t = tanh(decay * S_{t-1} + outer(k, delta))

            // Compute d_delta using dtanh
            if (tid < HEAD_V_DIM) {
                float d_delta_local = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    float d_pre = dS[i * HEAD_V_DIM + tid] * dtanh_arr[i * HEAD_V_DIM + tid];
                    d_delta_local += d_pre * k[i];
                }
                d_delta[tid] = d_delta_local;
            }
            __syncthreads();

            // Compute d_k using dtanh
            if (tid < N_STATE) {
                float d_k_local = 0.0f;
                for (int j = 0; j < HEAD_V_DIM; j++) {
                    float d_pre = dS[tid * HEAD_V_DIM + j] * dtanh_arr[tid * HEAD_V_DIM + j];
                    d_k_local += d_pre * delta[j];
                }
                d_k[tid] = d_k_local;
            }
            __syncthreads();

            // Compute d_decay using warp reduction
            float d_decay_local = 0.0f;
            for (int idx = tid; idx < state_size; idx += blockDim.x) {
                float d_pre = dS[idx] * dtanh_arr[idx];
                d_decay_local += d_pre * S[idx];
            }

            #pragma unroll
            for (int offset = 16; offset >= 1; offset /= 2) {
                d_decay_local += __shfl_xor_sync(0xFFFFFFFF, d_decay_local, offset);
            }

            if (lane_id == 0) {
                warp_results[warp_id] = d_decay_local;
            }
            __syncthreads();

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

            // d_k contribution from retrieved gradient: d_retrieved = -d_delta
            // retrieved[j] = sum_i S[i,j] * k[i]
            // d_k[i] += sum_j S[i,j] * d_retrieved[j] = sum_j S[i,j] * (-d_delta[j])
            if (tid < N_STATE) {
                float d_k_from_retrieved = 0.0f;
                for (int j = 0; j < HEAD_V_DIM; j++) {
                    d_k_from_retrieved += S[tid * HEAD_V_DIM + j] * (-d_delta[j]);
                }
                d_k[tid] += d_k_from_retrieved;
            }
            __syncthreads();

            // Write gradients for this timestep (atomic add for k since multiple slots may target same head)
            for (int i = tid; i < N_STATE; i += blockDim.x) {
                atomicAdd(reinterpret_cast<__nv_bfloat16*>(&d_k_all[k_offset + i]),
                          __float2bfloat16(d_k[i]));
                atomicAdd(reinterpret_cast<__nv_bfloat16*>(&d_q_all[k_offset + i]),
                          __float2bfloat16(d_q[i]));
            }
            for (int j = tid; j < HEAD_V_DIM; j += blockDim.x) {
                // d_v = d_delta (from delta = v - retrieved)
                atomicAdd(reinterpret_cast<__nv_bfloat16*>(&d_v_all[v_offset + j]),
                          __float2bfloat16(d_delta[j]));
            }
            if (tid == 0) {
                atomicAdd(reinterpret_cast<__nv_bfloat16*>(&d_decay_all[decay_offset]),
                          __float2bfloat16(d_decay_accum));
            }
            __syncthreads();

            // Update dS for next iteration (going backward in time)
            // dS_{t-1} = d_pre * decay + outer(k, -d_delta)
            for (int idx = tid; idx < state_size; idx += blockDim.x) {
                int i = idx / HEAD_V_DIM;
                int j = idx % HEAD_V_DIM;
                float d_pre = dS[idx] * dtanh_arr[idx];
                dS[idx] = d_pre * decay_val + k[i] * (-d_delta[j]);
            }
            __syncthreads();
        }
    }
}

// ============================================================================
// Dispatcher for MoM E88 Backward
// ============================================================================

void dispatch_mom_e88_backward(
    int T, int B, int H, int K, int n_state, int head_v_dim,
    const __nv_bfloat16* k_all, const __nv_bfloat16* v_all,
    const __nv_bfloat16* q_all, const __nv_bfloat16* decay_all,
    const int* head_indices, const __nv_bfloat16* router_weights,
    const __nv_bfloat16* S_checkpoints, const __nv_bfloat16* Sq_cache,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* d_k_all, __nv_bfloat16* d_v_all,
    __nv_bfloat16* d_q_all, __nv_bfloat16* d_decay_all,
    __nv_bfloat16* d_router_weights,
    __nv_bfloat16* segment_state_cache,
    int checkpoint_interval, cudaStream_t stream
) {
    int state_size = n_state * head_v_dim;
    // Shared memory: 4 * state_size + 2*N_STATE + 4*HEAD_V_DIM + N_STATE + 8 (warp_results)
    int shared_size = (4 * state_size + 3 * n_state + 4 * head_v_dim + 8) * sizeof(float);
    int threads = min(256, state_size);
    int num_blocks = B * K;  // One block per (batch, slot) pair

    #define DISPATCH_MOM_BWD(N, V) do { \
        auto kernel = MoME88BackwardKernel_BF16<N, V>; \
        if (shared_size > 48 * 1024) { \
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size); \
        } \
        kernel<<<num_blocks, threads, shared_size, stream>>>( \
            T, B, H, K, k_all, v_all, q_all, decay_all, \
            head_indices, router_weights, S_checkpoints, Sq_cache, d_output, \
            d_k_all, d_v_all, d_q_all, d_decay_all, d_router_weights, \
            segment_state_cache, checkpoint_interval); \
    } while(0)

    // Common configurations (matching forward kernel)
    if (n_state == 16 && head_v_dim == 16) { DISPATCH_MOM_BWD(16, 16); }
    else if (n_state == 32 && head_v_dim == 32) { DISPATCH_MOM_BWD(32, 32); }
    else if (n_state == 32 && head_v_dim == 64) { DISPATCH_MOM_BWD(32, 64); }
    else if (n_state == 48 && head_v_dim == 48) { DISPATCH_MOM_BWD(48, 48); }
    else if (n_state == 64 && head_v_dim == 64) { DISPATCH_MOM_BWD(64, 64); }
    else {
        fprintf(stderr, "MoM E88 Backward: unsupported n_state=%d, head_v_dim=%d\n", n_state, head_v_dim);
        return;
    }

    #undef DISPATCH_MOM_BWD
}

} // namespace elman
