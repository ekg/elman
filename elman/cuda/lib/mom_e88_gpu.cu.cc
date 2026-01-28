/**
 * MoM E88 CUDA Kernel - Mixture of Memory
 *
 * Sparse routing to memory heads (MoE-style for memory).
 * Each token routes to top-K heads instead of all H heads.
 *
 * Key differences from E88:
 * 1. Block indexing: (batch, slot) instead of (batch, head)
 * 2. Head lookup: actual head = head_indices[batch, slot]
 * 3. Output weighting: output scaled by router_weights
 * 4. State access: gather from non-contiguous heads
 *
 * Per selected head h = head_indices[b, slot]:
 *   k_h = L2_normalized(W_k_h @ x)  [n_state]
 *   v_h = W_v_h @ x                 [head_v_dim]
 *   q_h = L2_normalized(W_q_h @ x)  [n_state]
 *   decay_h = exp(-softplus(W_decay_h @ x))  [scalar]
 *
 *   retrieved = S_h @ k_h           [head_v_dim]
 *   delta = v_h - retrieved         [head_v_dim]
 *   S_h = tanh(decay_h * S_h + outer(delta, k_h))  [n_state x head_v_dim]
 *   out_slot = router_weight * (S_h^T @ q_h)  [head_v_dim]
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
// MoM E88 Forward Kernel
// Each block handles one (batch, slot) pair
// Slot maps to different heads based on routing
// ============================================================================

template<int N_STATE, int HEAD_V_DIM>
__global__ void MoME88ForwardKernel_BF16(
    int T,                    // Sequence length
    int B,                    // Batch size
    int H,                    // Total number of heads
    int K,                    // Top-k (number of active heads per token)
    const __nv_bfloat16* __restrict__ k_all,      // [T, B, H, N_STATE]
    const __nv_bfloat16* __restrict__ v_all,      // [T, B, H, HEAD_V_DIM]
    const __nv_bfloat16* __restrict__ q_all,      // [T, B, H, N_STATE]
    const __nv_bfloat16* __restrict__ decay_all,  // [T, B, H]
    const int* __restrict__ head_indices,         // [T, B, K] - which head for each slot
    const __nv_bfloat16* __restrict__ router_weights, // [T, B, K] - routing weights
    __nv_bfloat16* __restrict__ S,                // [B, H, N_STATE, HEAD_V_DIM] - all head states
    __nv_bfloat16* __restrict__ output,           // [T, B, HEAD_V_DIM] - combined output
    __nv_bfloat16* __restrict__ S_checkpoints,    // For gradient checkpointing
    __nv_bfloat16* __restrict__ Sq_cache,         // [T, B, K, HEAD_V_DIM] - per-slot outputs
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

    // Get initial head for slot 0 (will be updated per timestep)
    // Note: This is just for initialization; actual head changes per timestep
    int init_head = head_indices[b * K + slot];
    int state_offset = (b * H + init_head) * state_size;

    // Load initial state for this slot's initial head
    for (int i = tid; i < state_size; i += blockDim.x) {
        S_shared[i] = __bfloat162float(S[state_offset + i]);
    }
    __syncthreads();

    // Save initial checkpoint (slot-specific checkpoint index)
    int cp_base_offset = (b * K + slot) * state_size;
    for (int i = tid; i < state_size; i += blockDim.x) {
        S_checkpoints[cp_base_offset + i] = __float2bfloat16(S_shared[i]);
    }
    __syncthreads();

    // Track which head we're currently working with
    int prev_head = init_head;

    for (int t = 0; t < T; t++) {
        // Get head index for this timestep and slot
        int idx_offset = (t * B + b) * K + slot;
        if (tid == 0) {
            current_head = head_indices[idx_offset];
            router_weight = __bfloat162float(router_weights[idx_offset]);
        }
        __syncthreads();

        int h = current_head;

        // If head changed, need to load state from the new head
        // (This handles the case where routing changes heads mid-sequence)
        if (h != prev_head) {
            int new_state_offset = (b * H + h) * state_size;
            for (int i = tid; i < state_size; i += blockDim.x) {
                // First, save current state back to previous head
                int old_state_offset = (b * H + prev_head) * state_size;
                S[old_state_offset + i] = __float2bfloat16(S_shared[i]);
                // Then load state from new head
                S_shared[i] = __bfloat162float(S[new_state_offset + i]);
            }
            __syncthreads();
            prev_head = h;
        }

        // Load k, v, q, decay for this timestep and head
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

        // retrieved = S @ k
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
        for (int idx = tid; idx < state_size; idx += blockDim.x) {
            int i = idx / HEAD_V_DIM;
            int j = idx % HEAD_V_DIM;
            float delta_j = v_shared[j] - retrieved[j];
            S_shared[idx] = mom_e88_tanh(decay_val * S_shared[idx] + delta_j * k_shared[i]);
        }
        __syncthreads();

        // Save checkpoint
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

    // Write final state back to the last head we were working with
    int final_state_offset = (b * H + prev_head) * state_size;
    for (int i = tid; i < state_size; i += blockDim.x) {
        S[final_state_offset + i] = __float2bfloat16(S_shared[i]);
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

// Placeholder for backward - TODO: implement
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
    // TODO: Implement MoM backward kernel
    fprintf(stderr, "MoM E88 Backward: not yet implemented, use PyTorch autograd\n");
}

} // namespace elman
