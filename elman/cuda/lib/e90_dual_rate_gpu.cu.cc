/**
 * E90 Dual-Rate CUDA Kernel with Gradient Checkpointing
 *
 * Extends E88 with dual-rate factorized state:
 * - Fast state: Small (k_fast × v_fast), updated every timestep
 * - Slow state: Larger (k_slow × v_slow), updated via learned soft gate
 *
 * Per head h:
 *   # Fast state (updated every step)
 *   retrieved_fast = S_fast @ k_fast
 *   delta_fast = v_fast - retrieved_fast
 *   S_fast = tanh(decay_fast * S_fast + outer(k_fast, delta_fast))
 *   out_fast = S_fast @ q_fast
 *
 *   # Slow state (gated update)
 *   retrieved_slow = S_slow @ k_slow
 *   delta_slow = v_slow - retrieved_slow
 *   S_slow = tanh(decay_slow * S_slow + slow_gate * outer(k_slow, delta_slow))
 *   out_slow = S_slow @ q_slow
 *
 *   # Mix outputs
 *   out_h = mix_fast * out_fast + mix_slow * out_slow
 *
 * Checkpointing strategy (same as E88):
 * - Forward: Save state checkpoints every CHECKPOINT_INTERVAL steps
 * - Backward: Process segments, replay forward once per segment, then backward
 * - Complexity: O(T) instead of O(T²)
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include "hasty/elman_ladder.h"

#define E90_CHECKPOINT_INTERVAL 16

namespace elman {

__device__ __forceinline__ float e90_tanh(float x) {
    return tanhf(x);
}

// ============================================================================
// E90 Dual-Rate Forward Kernel with Checkpointing
// Each block handles one (batch, head) pair
// Template parameters: K_FAST, V_FAST, K_SLOW, V_SLOW
// ============================================================================

template<int K_FAST, int V_FAST, int K_SLOW, int V_SLOW>
__global__ void E90DualRateForwardKernel_BF16(
    int T,
    int B,
    int H,
    // Fast state inputs
    const __nv_bfloat16* __restrict__ k_fast_all,      // [T, B, H, K_FAST]
    const __nv_bfloat16* __restrict__ v_fast_all,      // [T, B, H, V_FAST]
    const __nv_bfloat16* __restrict__ q_fast_all,      // [T, B, H, K_FAST]
    const __nv_bfloat16* __restrict__ decay_fast_all,  // [T, B, H]
    // Slow state inputs
    const __nv_bfloat16* __restrict__ k_slow_all,      // [T, B, H, K_SLOW]
    const __nv_bfloat16* __restrict__ v_slow_all,      // [T, B, H, V_SLOW]
    const __nv_bfloat16* __restrict__ q_slow_all,      // [T, B, H, K_SLOW]
    const __nv_bfloat16* __restrict__ decay_slow_all,  // [T, B, H]
    const __nv_bfloat16* __restrict__ slow_gate_all,   // [T, B, H]
    // Mixing weights
    const __nv_bfloat16* __restrict__ mix_fast_all,    // [T, B, H]
    const __nv_bfloat16* __restrict__ mix_slow_all,    // [T, B, H]
    // States (in/out)
    __nv_bfloat16* __restrict__ S_fast,                // [B, H, K_FAST, V_FAST]
    __nv_bfloat16* __restrict__ S_slow,                // [B, H, K_SLOW, V_SLOW]
    // Output (max of V_FAST, V_SLOW)
    __nv_bfloat16* __restrict__ output,                // [T, B, H, max(V_FAST, V_SLOW)]
    // Checkpoints for backward pass
    __nv_bfloat16* __restrict__ S_fast_checkpoints,    // [num_cp, B, H, K_FAST, V_FAST]
    __nv_bfloat16* __restrict__ S_slow_checkpoints,    // [num_cp, B, H, K_SLOW, V_SLOW]
    int out_v_dim,  // max(V_FAST, V_SLOW)
    int checkpoint_interval
) {
    int block_idx = blockIdx.x;
    int b = block_idx / H;
    int h = block_idx % H;
    if (b >= B) return;

    extern __shared__ float shared_mem[];

    // Shared memory layout for FAST state
    constexpr int fast_state_size = K_FAST * V_FAST;
    float* S_fast_shared = shared_mem;
    float* k_fast_shared = S_fast_shared + fast_state_size;
    float* v_fast_shared = k_fast_shared + K_FAST;
    float* q_fast_shared = v_fast_shared + V_FAST;
    float* retrieved_fast = q_fast_shared + K_FAST;

    // Shared memory layout for SLOW state (after fast)
    constexpr int slow_state_size = K_SLOW * V_SLOW;
    float* S_slow_shared = retrieved_fast + V_FAST;
    float* k_slow_shared = S_slow_shared + slow_state_size;
    float* v_slow_shared = k_slow_shared + K_SLOW;
    float* q_slow_shared = v_slow_shared + V_SLOW;
    float* retrieved_slow = q_slow_shared + K_SLOW;

    // Output buffers
    float* out_fast_buf = retrieved_slow + V_SLOW;
    float* out_slow_buf = out_fast_buf + V_FAST;

    // Shared scalars
    __shared__ float decay_fast_val, decay_slow_val, slow_gate_val, mix_fast_val, mix_slow_val;

    int tid = threadIdx.x;

    // State offsets for this (batch, head)
    int fast_state_offset = (b * H + h) * fast_state_size;
    int slow_state_offset = (b * H + h) * slow_state_size;

    // Load initial fast state
    for (int i = tid; i < fast_state_size; i += blockDim.x) {
        S_fast_shared[i] = __bfloat162float(S_fast[fast_state_offset + i]);
    }
    // Load initial slow state
    for (int i = tid; i < slow_state_size; i += blockDim.x) {
        S_slow_shared[i] = __bfloat162float(S_slow[slow_state_offset + i]);
    }
    __syncthreads();

    // Save initial checkpoint (index 0)
    for (int i = tid; i < fast_state_size; i += blockDim.x) {
        S_fast_checkpoints[(b * H + h) * fast_state_size + i] = __float2bfloat16(S_fast_shared[i]);
    }
    for (int i = tid; i < slow_state_size; i += blockDim.x) {
        S_slow_checkpoints[(b * H + h) * slow_state_size + i] = __float2bfloat16(S_slow_shared[i]);
    }
    __syncthreads();

    for (int t = 0; t < T; t++) {
        // Calculate offsets for this timestep
        int fast_k_offset = ((t * B + b) * H + h) * K_FAST;
        int fast_v_offset = ((t * B + b) * H + h) * V_FAST;
        int slow_k_offset = ((t * B + b) * H + h) * K_SLOW;
        int slow_v_offset = ((t * B + b) * H + h) * V_SLOW;
        int scalar_offset = (t * B + b) * H + h;
        int out_offset = ((t * B + b) * H + h) * out_v_dim;

        // Load all inputs for this timestep
        if (tid < K_FAST) {
            k_fast_shared[tid] = __bfloat162float(k_fast_all[fast_k_offset + tid]);
            q_fast_shared[tid] = __bfloat162float(q_fast_all[fast_k_offset + tid]);
        }
        if (tid < V_FAST) {
            v_fast_shared[tid] = __bfloat162float(v_fast_all[fast_v_offset + tid]);
        }
        if (tid < K_SLOW) {
            k_slow_shared[tid] = __bfloat162float(k_slow_all[slow_k_offset + tid]);
            q_slow_shared[tid] = __bfloat162float(q_slow_all[slow_k_offset + tid]);
        }
        if (tid < V_SLOW) {
            v_slow_shared[tid] = __bfloat162float(v_slow_all[slow_v_offset + tid]);
        }
        if (tid == 0) {
            decay_fast_val = __bfloat162float(decay_fast_all[scalar_offset]);
            decay_slow_val = __bfloat162float(decay_slow_all[scalar_offset]);
            slow_gate_val = __bfloat162float(slow_gate_all[scalar_offset]);
            mix_fast_val = __bfloat162float(mix_fast_all[scalar_offset]);
            mix_slow_val = __bfloat162float(mix_slow_all[scalar_offset]);
        }
        __syncthreads();

        // ==================== FAST STATE UPDATE ====================
        // retrieved_fast = S_fast @ k_fast (S is [K_FAST, V_FAST], k is [K_FAST])
        if (tid < V_FAST) {
            float sum = 0.0f;
            for (int i = 0; i < K_FAST; i++) {
                sum += S_fast_shared[i * V_FAST + tid] * k_fast_shared[i];
            }
            retrieved_fast[tid] = sum;
        }
        __syncthreads();

        // S_fast = tanh(decay_fast * S_fast + outer(k_fast, delta_fast))
        for (int idx = tid; idx < fast_state_size; idx += blockDim.x) {
            int i = idx / V_FAST;
            int j = idx % V_FAST;
            float delta_j = v_fast_shared[j] - retrieved_fast[j];
            S_fast_shared[idx] = e90_tanh(decay_fast_val * S_fast_shared[idx] + k_fast_shared[i] * delta_j);
        }
        __syncthreads();

        // out_fast = S_fast @ q_fast
        if (tid < V_FAST) {
            float sum = 0.0f;
            for (int i = 0; i < K_FAST; i++) {
                sum += S_fast_shared[i * V_FAST + tid] * q_fast_shared[i];
            }
            out_fast_buf[tid] = sum;
        }
        __syncthreads();

        // ==================== SLOW STATE UPDATE (GATED) ====================
        // retrieved_slow = S_slow @ k_slow
        if (tid < V_SLOW) {
            float sum = 0.0f;
            for (int i = 0; i < K_SLOW; i++) {
                sum += S_slow_shared[i * V_SLOW + tid] * k_slow_shared[i];
            }
            retrieved_slow[tid] = sum;
        }
        __syncthreads();

        // S_slow = tanh(decay_slow * S_slow + slow_gate * outer(k_slow, delta_slow))
        for (int idx = tid; idx < slow_state_size; idx += blockDim.x) {
            int i = idx / V_SLOW;
            int j = idx % V_SLOW;
            float delta_j = v_slow_shared[j] - retrieved_slow[j];
            // Gated update: slow_gate multiplies the outer product contribution
            S_slow_shared[idx] = e90_tanh(decay_slow_val * S_slow_shared[idx] + slow_gate_val * k_slow_shared[i] * delta_j);
        }
        __syncthreads();

        // out_slow = S_slow @ q_slow
        if (tid < V_SLOW) {
            float sum = 0.0f;
            for (int i = 0; i < K_SLOW; i++) {
                sum += S_slow_shared[i * V_SLOW + tid] * q_slow_shared[i];
            }
            out_slow_buf[tid] = sum;
        }
        __syncthreads();

        // ==================== MIX OUTPUTS ====================
        // output = mix_fast * out_fast + mix_slow * out_slow
        // Handle dimension mismatch by padding smaller output
        if (tid < out_v_dim) {
            float fast_val = (tid < V_FAST) ? out_fast_buf[tid] : 0.0f;
            float slow_val = (tid < V_SLOW) ? out_slow_buf[tid] : 0.0f;
            float mixed = mix_fast_val * fast_val + mix_slow_val * slow_val;
            output[out_offset + tid] = __float2bfloat16(mixed);
        }

        // Save checkpoint after update
        if ((t + 1) % checkpoint_interval == 0) {
            int cp_idx = (t + 1) / checkpoint_interval;
            int cp_fast_offset = (cp_idx * B * H + b * H + h) * fast_state_size;
            int cp_slow_offset = (cp_idx * B * H + b * H + h) * slow_state_size;
            for (int i = tid; i < fast_state_size; i += blockDim.x) {
                S_fast_checkpoints[cp_fast_offset + i] = __float2bfloat16(S_fast_shared[i]);
            }
            for (int i = tid; i < slow_state_size; i += blockDim.x) {
                S_slow_checkpoints[cp_slow_offset + i] = __float2bfloat16(S_slow_shared[i]);
            }
        }
        __syncthreads();
    }

    // Write final states back
    for (int i = tid; i < fast_state_size; i += blockDim.x) {
        S_fast[fast_state_offset + i] = __float2bfloat16(S_fast_shared[i]);
    }
    for (int i = tid; i < slow_state_size; i += blockDim.x) {
        S_slow[slow_state_offset + i] = __float2bfloat16(S_slow_shared[i]);
    }
}


// ============================================================================
// E90 Dual-Rate Backward Kernel with Segment-Level Caching
//
// For each segment:
//   1. Replay forward ONCE, caching all S_{t-1} states
//   2. Process backward using cached states
// This is O(T) instead of O(T²)
// ============================================================================

template<int K_FAST, int V_FAST, int K_SLOW, int V_SLOW>
__global__ void E90DualRateBackwardKernel_BF16(
    int T,
    int B,
    int H,
    // Fast state inputs
    const __nv_bfloat16* __restrict__ k_fast_all,
    const __nv_bfloat16* __restrict__ v_fast_all,
    const __nv_bfloat16* __restrict__ q_fast_all,
    const __nv_bfloat16* __restrict__ decay_fast_all,
    // Slow state inputs
    const __nv_bfloat16* __restrict__ k_slow_all,
    const __nv_bfloat16* __restrict__ v_slow_all,
    const __nv_bfloat16* __restrict__ q_slow_all,
    const __nv_bfloat16* __restrict__ decay_slow_all,
    const __nv_bfloat16* __restrict__ slow_gate_all,
    // Mixing weights
    const __nv_bfloat16* __restrict__ mix_fast_all,
    const __nv_bfloat16* __restrict__ mix_slow_all,
    // Upstream gradient
    const __nv_bfloat16* __restrict__ d_output,  // [T, B, H, out_v_dim]
    // Checkpoints
    const __nv_bfloat16* __restrict__ S_fast_checkpoints,
    const __nv_bfloat16* __restrict__ S_slow_checkpoints,
    // Segment state cache
    __nv_bfloat16* __restrict__ seg_fast_cache,  // [B*H, checkpoint_interval, K_FAST*V_FAST]
    __nv_bfloat16* __restrict__ seg_slow_cache,  // [B*H, checkpoint_interval, K_SLOW*V_SLOW]
    // Output gradients
    __nv_bfloat16* __restrict__ d_k_fast_all,
    __nv_bfloat16* __restrict__ d_v_fast_all,
    __nv_bfloat16* __restrict__ d_q_fast_all,
    __nv_bfloat16* __restrict__ d_decay_fast_all,
    __nv_bfloat16* __restrict__ d_k_slow_all,
    __nv_bfloat16* __restrict__ d_v_slow_all,
    __nv_bfloat16* __restrict__ d_q_slow_all,
    __nv_bfloat16* __restrict__ d_decay_slow_all,
    __nv_bfloat16* __restrict__ d_slow_gate_all,
    __nv_bfloat16* __restrict__ d_mix_fast_all,
    __nv_bfloat16* __restrict__ d_mix_slow_all,
    int out_v_dim,
    int checkpoint_interval
) {
    int block_idx = blockIdx.x;
    int b = block_idx / H;
    int h = block_idx % H;
    if (b >= B) return;

    extern __shared__ float shared_mem[];

    // ====================== SHARED MEMORY LAYOUT ======================
    // Fast state variables
    constexpr int fast_state_size = K_FAST * V_FAST;
    float* S_fast = shared_mem;
    float* dS_fast = S_fast + fast_state_size;
    float* k_fast = dS_fast + fast_state_size;
    float* v_fast = k_fast + K_FAST;
    float* q_fast = v_fast + V_FAST;
    float* delta_fast = q_fast + K_FAST;
    float* retrieved_fast = delta_fast + V_FAST;
    float* d_k_fast = retrieved_fast + V_FAST;
    float* d_v_fast = d_k_fast + K_FAST;
    float* d_q_fast = d_v_fast + V_FAST;
    float* out_fast = d_q_fast + K_FAST;
    float* d_out_fast = out_fast + V_FAST;
    float* S_fast_t = d_out_fast + V_FAST;
    float* dtanh_fast = S_fast_t + fast_state_size;

    // Slow state variables
    constexpr int slow_state_size = K_SLOW * V_SLOW;
    float* S_slow = dtanh_fast + fast_state_size;
    float* dS_slow = S_slow + slow_state_size;
    float* k_slow = dS_slow + slow_state_size;
    float* v_slow = k_slow + K_SLOW;
    float* q_slow = v_slow + V_SLOW;
    float* delta_slow = q_slow + K_SLOW;
    float* retrieved_slow = delta_slow + V_SLOW;
    float* d_k_slow = retrieved_slow + V_SLOW;
    float* d_v_slow = d_k_slow + K_SLOW;
    float* d_q_slow = d_v_slow + V_SLOW;
    float* out_slow = d_q_slow + K_SLOW;
    float* d_out_slow = out_slow + V_SLOW;
    float* S_slow_t = d_out_slow + V_SLOW;
    float* dtanh_slow = S_slow_t + slow_state_size;

    // Output and reduction buffers
    float* d_out = dtanh_slow + slow_state_size;
    float* warp_results = d_out + out_v_dim;  // For reduction

    // Shared scalars
    __shared__ float decay_fast_val, decay_slow_val, slow_gate_val, mix_fast_val, mix_slow_val;

    int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = (blockDim.x + 31) / 32;

    // Segment cache base pointers for this (batch, head)
    __nv_bfloat16* fast_cache_base = seg_fast_cache + (size_t)block_idx * checkpoint_interval * fast_state_size;
    __nv_bfloat16* slow_cache_base = seg_slow_cache + (size_t)block_idx * checkpoint_interval * slow_state_size;

    // Initialize dS to zero
    for (int i = tid; i < fast_state_size; i += blockDim.x) {
        dS_fast[i] = 0.0f;
    }
    for (int i = tid; i < slow_state_size; i += blockDim.x) {
        dS_slow[i] = 0.0f;
    }
    __syncthreads();

    int num_segments = (T + checkpoint_interval - 1) / checkpoint_interval;

    for (int seg = num_segments - 1; seg >= 0; seg--) {
        int t_start = seg * checkpoint_interval;
        int t_end = min(t_start + checkpoint_interval, T);
        int seg_len = t_end - t_start;

        // ================================================================
        // PHASE 1: Forward replay through segment, caching S_{t-1} for each t
        // ================================================================

        // Load checkpoint for this segment
        int cp_fast_offset = (seg * B * H + b * H + h) * fast_state_size;
        int cp_slow_offset = (seg * B * H + b * H + h) * slow_state_size;
        for (int i = tid; i < fast_state_size; i += blockDim.x) {
            S_fast[i] = __bfloat162float(S_fast_checkpoints[cp_fast_offset + i]);
        }
        for (int i = tid; i < slow_state_size; i += blockDim.x) {
            S_slow[i] = __bfloat162float(S_slow_checkpoints[cp_slow_offset + i]);
        }
        __syncthreads();

        // Replay forward through entire segment
        for (int local_t = 0; local_t < seg_len; local_t++) {
            int t = t_start + local_t;

            // Save S_{t-1} to segment cache BEFORE the update
            __nv_bfloat16* fast_cache_slot = fast_cache_base + (size_t)local_t * fast_state_size;
            __nv_bfloat16* slow_cache_slot = slow_cache_base + (size_t)local_t * slow_state_size;
            for (int i = tid; i < fast_state_size; i += blockDim.x) {
                fast_cache_slot[i] = __float2bfloat16(S_fast[i]);
            }
            for (int i = tid; i < slow_state_size; i += blockDim.x) {
                slow_cache_slot[i] = __float2bfloat16(S_slow[i]);
            }
            __syncthreads();

            // Load inputs for timestep t
            int fast_k_offset = ((t * B + b) * H + h) * K_FAST;
            int fast_v_offset = ((t * B + b) * H + h) * V_FAST;
            int slow_k_offset = ((t * B + b) * H + h) * K_SLOW;
            int slow_v_offset = ((t * B + b) * H + h) * V_SLOW;
            int scalar_offset = (t * B + b) * H + h;

            if (tid < K_FAST) k_fast[tid] = __bfloat162float(k_fast_all[fast_k_offset + tid]);
            if (tid < V_FAST) v_fast[tid] = __bfloat162float(v_fast_all[fast_v_offset + tid]);
            if (tid < K_SLOW) k_slow[tid] = __bfloat162float(k_slow_all[slow_k_offset + tid]);
            if (tid < V_SLOW) v_slow[tid] = __bfloat162float(v_slow_all[slow_v_offset + tid]);
            if (tid == 0) {
                decay_fast_val = __bfloat162float(decay_fast_all[scalar_offset]);
                decay_slow_val = __bfloat162float(decay_slow_all[scalar_offset]);
                slow_gate_val = __bfloat162float(slow_gate_all[scalar_offset]);
            }
            __syncthreads();

            // Fast state: compute retrieved, then update
            if (tid < V_FAST) {
                float sum = 0.0f;
                for (int i = 0; i < K_FAST; i++) sum += S_fast[i * V_FAST + tid] * k_fast[i];
                retrieved_fast[tid] = sum;
            }
            __syncthreads();

            for (int idx = tid; idx < fast_state_size; idx += blockDim.x) {
                int i = idx / V_FAST;
                int j = idx % V_FAST;
                float delta_j = v_fast[j] - retrieved_fast[j];
                S_fast[idx] = e90_tanh(decay_fast_val * S_fast[idx] + k_fast[i] * delta_j);
            }
            __syncthreads();

            // Slow state: compute retrieved, then update
            if (tid < V_SLOW) {
                float sum = 0.0f;
                for (int i = 0; i < K_SLOW; i++) sum += S_slow[i * V_SLOW + tid] * k_slow[i];
                retrieved_slow[tid] = sum;
            }
            __syncthreads();

            for (int idx = tid; idx < slow_state_size; idx += blockDim.x) {
                int i = idx / V_SLOW;
                int j = idx % V_SLOW;
                float delta_j = v_slow[j] - retrieved_slow[j];
                S_slow[idx] = e90_tanh(decay_slow_val * S_slow[idx] + slow_gate_val * k_slow[i] * delta_j);
            }
            __syncthreads();
        }

        // ================================================================
        // PHASE 2: Backward pass through segment using cached states
        // ================================================================

        for (int local_t = seg_len - 1; local_t >= 0; local_t--) {
            int t = t_start + local_t;

            // Load cached S_{t-1} from segment cache
            __nv_bfloat16* fast_cache_slot = fast_cache_base + (size_t)local_t * fast_state_size;
            __nv_bfloat16* slow_cache_slot = slow_cache_base + (size_t)local_t * slow_state_size;
            for (int i = tid; i < fast_state_size; i += blockDim.x) {
                S_fast[i] = __bfloat162float(fast_cache_slot[i]);
            }
            for (int i = tid; i < slow_state_size; i += blockDim.x) {
                S_slow[i] = __bfloat162float(slow_cache_slot[i]);
            }
            __syncthreads();

            // Load all inputs for timestep t
            int fast_k_offset = ((t * B + b) * H + h) * K_FAST;
            int fast_v_offset = ((t * B + b) * H + h) * V_FAST;
            int slow_k_offset = ((t * B + b) * H + h) * K_SLOW;
            int slow_v_offset = ((t * B + b) * H + h) * V_SLOW;
            int scalar_offset = (t * B + b) * H + h;
            int out_offset = ((t * B + b) * H + h) * out_v_dim;

            if (tid < K_FAST) {
                k_fast[tid] = __bfloat162float(k_fast_all[fast_k_offset + tid]);
                q_fast[tid] = __bfloat162float(q_fast_all[fast_k_offset + tid]);
            }
            if (tid < V_FAST) {
                v_fast[tid] = __bfloat162float(v_fast_all[fast_v_offset + tid]);
            }
            if (tid < K_SLOW) {
                k_slow[tid] = __bfloat162float(k_slow_all[slow_k_offset + tid]);
                q_slow[tid] = __bfloat162float(q_slow_all[slow_k_offset + tid]);
            }
            if (tid < V_SLOW) {
                v_slow[tid] = __bfloat162float(v_slow_all[slow_v_offset + tid]);
            }
            if (tid == 0) {
                decay_fast_val = __bfloat162float(decay_fast_all[scalar_offset]);
                decay_slow_val = __bfloat162float(decay_slow_all[scalar_offset]);
                slow_gate_val = __bfloat162float(slow_gate_all[scalar_offset]);
                mix_fast_val = __bfloat162float(mix_fast_all[scalar_offset]);
                mix_slow_val = __bfloat162float(mix_slow_all[scalar_offset]);
            }
            __syncthreads();

            // Compute retrieved and delta for both states
            if (tid < V_FAST) {
                float sum = 0.0f;
                for (int i = 0; i < K_FAST; i++) sum += S_fast[i * V_FAST + tid] * k_fast[i];
                retrieved_fast[tid] = sum;
                delta_fast[tid] = v_fast[tid] - retrieved_fast[tid];
            }
            if (tid < V_SLOW) {
                float sum = 0.0f;
                for (int i = 0; i < K_SLOW; i++) sum += S_slow[i * V_SLOW + tid] * k_slow[i];
                retrieved_slow[tid] = sum;
                delta_slow[tid] = v_slow[tid] - retrieved_slow[tid];
            }
            __syncthreads();

            // Compute S_t and dtanh for both states
            for (int idx = tid; idx < fast_state_size; idx += blockDim.x) {
                int i = idx / V_FAST;
                int j = idx % V_FAST;
                float pre_tanh = decay_fast_val * S_fast[idx] + k_fast[i] * delta_fast[j];
                float tanh_val = e90_tanh(pre_tanh);
                S_fast_t[idx] = tanh_val;
                dtanh_fast[idx] = 1.0f - tanh_val * tanh_val;
            }
            for (int idx = tid; idx < slow_state_size; idx += blockDim.x) {
                int i = idx / V_SLOW;
                int j = idx % V_SLOW;
                float pre_tanh = decay_slow_val * S_slow[idx] + slow_gate_val * k_slow[i] * delta_slow[j];
                float tanh_val = e90_tanh(pre_tanh);
                S_slow_t[idx] = tanh_val;
                dtanh_slow[idx] = 1.0f - tanh_val * tanh_val;
            }
            __syncthreads();

            // Compute out_fast and out_slow for d_mix computation
            if (tid < V_FAST) {
                float sum = 0.0f;
                for (int i = 0; i < K_FAST; i++) sum += S_fast_t[i * V_FAST + tid] * q_fast[i];
                out_fast[tid] = sum;
            }
            if (tid < V_SLOW) {
                float sum = 0.0f;
                for (int i = 0; i < K_SLOW; i++) sum += S_slow_t[i * V_SLOW + tid] * q_slow[i];
                out_slow[tid] = sum;
            }
            __syncthreads();

            // Load d_output
            if (tid < out_v_dim) {
                d_out[tid] = __bfloat162float(d_output[out_offset + tid]);
            }
            __syncthreads();

            // ==================== BACKWARD THROUGH MIXING ====================
            // d_mix_fast = sum(d_output * out_fast)
            // d_mix_slow = sum(d_output * out_slow)
            float d_mix_fast_local = 0.0f, d_mix_slow_local = 0.0f;
            for (int j = tid; j < out_v_dim; j += blockDim.x) {
                float out_f = (j < V_FAST) ? out_fast[j] : 0.0f;
                float out_s = (j < V_SLOW) ? out_slow[j] : 0.0f;
                d_mix_fast_local += d_out[j] * out_f;
                d_mix_slow_local += d_out[j] * out_s;
            }

            // Warp reduction
            #pragma unroll
            for (int offset = 16; offset >= 1; offset /= 2) {
                d_mix_fast_local += __shfl_xor_sync(0xFFFFFFFF, d_mix_fast_local, offset);
                d_mix_slow_local += __shfl_xor_sync(0xFFFFFFFF, d_mix_slow_local, offset);
            }
            if (lane_id == 0) {
                warp_results[warp_id] = d_mix_fast_local;
                warp_results[num_warps + warp_id] = d_mix_slow_local;
            }
            __syncthreads();

            float d_mix_fast_accum = 0.0f, d_mix_slow_accum = 0.0f;
            if (warp_id == 0) {
                // All warp 0 threads participate, but only first num_warps have data
                d_mix_fast_accum = (tid < num_warps) ? warp_results[tid] : 0.0f;
                d_mix_slow_accum = (tid < num_warps) ? warp_results[num_warps + tid] : 0.0f;
                #pragma unroll
                for (int offset = 16; offset >= 1; offset /= 2) {
                    d_mix_fast_accum += __shfl_xor_sync(0xFFFFFFFF, d_mix_fast_accum, offset);
                    d_mix_slow_accum += __shfl_xor_sync(0xFFFFFFFF, d_mix_slow_accum, offset);
                }
            }
            __syncthreads();

            // Write d_mix gradients
            if (tid == 0) {
                d_mix_fast_all[scalar_offset] = __float2bfloat16(d_mix_fast_accum);
                d_mix_slow_all[scalar_offset] = __float2bfloat16(d_mix_slow_accum);
            }

            // d_out_fast = d_output * mix_fast, d_out_slow = d_output * mix_slow
            if (tid < V_FAST) {
                d_out_fast[tid] = d_out[tid] * mix_fast_val;
            }
            if (tid < V_SLOW) {
                d_out_slow[tid] = d_out[tid] * mix_slow_val;
            }
            __syncthreads();

            // ==================== BACKWARD THROUGH FAST OUTPUT ====================
            // d_q_fast from out_fast = S_fast_t @ q_fast
            if (tid < K_FAST) {
                float sum = 0.0f;
                for (int j = 0; j < V_FAST; j++) {
                    sum += S_fast_t[tid * V_FAST + j] * d_out_fast[j];
                }
                d_q_fast[tid] = sum;
            }
            __syncthreads();

            // dS_fast += outer(q_fast, d_out_fast)
            for (int idx = tid; idx < fast_state_size; idx += blockDim.x) {
                int i = idx / V_FAST;
                int j = idx % V_FAST;
                dS_fast[idx] += q_fast[i] * d_out_fast[j];
            }
            __syncthreads();

            // ==================== BACKWARD THROUGH FAST STATE UPDATE ====================
            // d_delta_fast using dtanh
            if (tid < V_FAST) {
                float d_delta_local = 0.0f;
                for (int i = 0; i < K_FAST; i++) {
                    float d_pre = dS_fast[i * V_FAST + tid] * dtanh_fast[i * V_FAST + tid];
                    d_delta_local += d_pre * k_fast[i];
                }
                d_v_fast[tid] = d_delta_local;  // d_v = d_delta
            }
            __syncthreads();

            // d_k_fast using dtanh
            if (tid < K_FAST) {
                float d_k_local = 0.0f;
                for (int j = 0; j < V_FAST; j++) {
                    float d_pre = dS_fast[tid * V_FAST + j] * dtanh_fast[tid * V_FAST + j];
                    d_k_local += d_pre * delta_fast[j];
                }
                // d_k from retrieved
                float d_k_from_ret = 0.0f;
                for (int j = 0; j < V_FAST; j++) {
                    d_k_from_ret += S_fast[tid * V_FAST + j] * (-d_v_fast[j]);
                }
                d_k_fast[tid] = d_k_local + d_k_from_ret;
            }
            __syncthreads();

            // d_decay_fast
            float d_decay_fast_local = 0.0f;
            for (int idx = tid; idx < fast_state_size; idx += blockDim.x) {
                float d_pre = dS_fast[idx] * dtanh_fast[idx];
                d_decay_fast_local += d_pre * S_fast[idx];
            }
            #pragma unroll
            for (int offset = 16; offset >= 1; offset /= 2) {
                d_decay_fast_local += __shfl_xor_sync(0xFFFFFFFF, d_decay_fast_local, offset);
            }
            if (lane_id == 0) warp_results[warp_id] = d_decay_fast_local;
            __syncthreads();

            float d_decay_fast_accum = 0.0f;
            if (warp_id == 0) {
                d_decay_fast_accum = (tid < num_warps) ? warp_results[tid] : 0.0f;
                #pragma unroll
                for (int offset = 16; offset >= 1; offset /= 2) {
                    d_decay_fast_accum += __shfl_xor_sync(0xFFFFFFFF, d_decay_fast_accum, offset);
                }
            }
            __syncthreads();

            // Write fast gradients
            if (tid < K_FAST) {
                d_k_fast_all[fast_k_offset + tid] = __float2bfloat16(d_k_fast[tid]);
                d_q_fast_all[fast_k_offset + tid] = __float2bfloat16(d_q_fast[tid]);
            }
            if (tid < V_FAST) {
                d_v_fast_all[fast_v_offset + tid] = __float2bfloat16(d_v_fast[tid]);
            }
            if (tid == 0) {
                d_decay_fast_all[scalar_offset] = __float2bfloat16(d_decay_fast_accum);
            }
            __syncthreads();

            // Update dS_fast for next iteration
            for (int idx = tid; idx < fast_state_size; idx += blockDim.x) {
                int i = idx / V_FAST;
                int j = idx % V_FAST;
                float d_pre = dS_fast[idx] * dtanh_fast[idx];
                dS_fast[idx] = d_pre * decay_fast_val + (-d_v_fast[j]) * k_fast[i];
            }
            __syncthreads();

            // ==================== BACKWARD THROUGH SLOW OUTPUT ====================
            // d_q_slow from out_slow = S_slow_t @ q_slow
            if (tid < K_SLOW) {
                float sum = 0.0f;
                for (int j = 0; j < V_SLOW; j++) {
                    sum += S_slow_t[tid * V_SLOW + j] * d_out_slow[j];
                }
                d_q_slow[tid] = sum;
            }
            __syncthreads();

            // dS_slow += outer(q_slow, d_out_slow)
            for (int idx = tid; idx < slow_state_size; idx += blockDim.x) {
                int i = idx / V_SLOW;
                int j = idx % V_SLOW;
                dS_slow[idx] += q_slow[i] * d_out_slow[j];
            }
            __syncthreads();

            // ==================== BACKWARD THROUGH SLOW STATE UPDATE ====================
            // d_delta_slow using dtanh (note: includes slow_gate)
            if (tid < V_SLOW) {
                float d_delta_local = 0.0f;
                for (int i = 0; i < K_SLOW; i++) {
                    float d_pre = dS_slow[i * V_SLOW + tid] * dtanh_slow[i * V_SLOW + tid];
                    d_delta_local += d_pre * slow_gate_val * k_slow[i];
                }
                d_v_slow[tid] = d_delta_local;  // d_v = d_delta
            }
            __syncthreads();

            // d_k_slow using dtanh
            if (tid < K_SLOW) {
                float d_k_local = 0.0f;
                for (int j = 0; j < V_SLOW; j++) {
                    float d_pre = dS_slow[tid * V_SLOW + j] * dtanh_slow[tid * V_SLOW + j];
                    d_k_local += d_pre * slow_gate_val * delta_slow[j];
                }
                // d_k from retrieved
                float d_k_from_ret = 0.0f;
                for (int j = 0; j < V_SLOW; j++) {
                    d_k_from_ret += S_slow[tid * V_SLOW + j] * (-d_v_slow[j]);
                }
                d_k_slow[tid] = d_k_local + d_k_from_ret;
            }
            __syncthreads();

            // d_decay_slow and d_slow_gate
            float d_decay_slow_local = 0.0f;
            float d_slow_gate_local = 0.0f;
            for (int idx = tid; idx < slow_state_size; idx += blockDim.x) {
                int i = idx / V_SLOW;
                int j = idx % V_SLOW;
                float d_pre = dS_slow[idx] * dtanh_slow[idx];
                d_decay_slow_local += d_pre * S_slow[idx];
                d_slow_gate_local += d_pre * k_slow[i] * delta_slow[j];
            }
            #pragma unroll
            for (int offset = 16; offset >= 1; offset /= 2) {
                d_decay_slow_local += __shfl_xor_sync(0xFFFFFFFF, d_decay_slow_local, offset);
                d_slow_gate_local += __shfl_xor_sync(0xFFFFFFFF, d_slow_gate_local, offset);
            }
            if (lane_id == 0) {
                warp_results[warp_id] = d_decay_slow_local;
                warp_results[num_warps + warp_id] = d_slow_gate_local;
            }
            __syncthreads();

            float d_decay_slow_accum = 0.0f, d_slow_gate_accum = 0.0f;
            if (warp_id == 0) {
                d_decay_slow_accum = (tid < num_warps) ? warp_results[tid] : 0.0f;
                d_slow_gate_accum = (tid < num_warps) ? warp_results[num_warps + tid] : 0.0f;
                #pragma unroll
                for (int offset = 16; offset >= 1; offset /= 2) {
                    d_decay_slow_accum += __shfl_xor_sync(0xFFFFFFFF, d_decay_slow_accum, offset);
                    d_slow_gate_accum += __shfl_xor_sync(0xFFFFFFFF, d_slow_gate_accum, offset);
                }
            }
            __syncthreads();

            // Write slow gradients
            if (tid < K_SLOW) {
                d_k_slow_all[slow_k_offset + tid] = __float2bfloat16(d_k_slow[tid]);
                d_q_slow_all[slow_k_offset + tid] = __float2bfloat16(d_q_slow[tid]);
            }
            if (tid < V_SLOW) {
                d_v_slow_all[slow_v_offset + tid] = __float2bfloat16(d_v_slow[tid]);
            }
            if (tid == 0) {
                d_decay_slow_all[scalar_offset] = __float2bfloat16(d_decay_slow_accum);
                d_slow_gate_all[scalar_offset] = __float2bfloat16(d_slow_gate_accum);
            }
            __syncthreads();

            // Update dS_slow for next iteration
            for (int idx = tid; idx < slow_state_size; idx += blockDim.x) {
                int i = idx / V_SLOW;
                int j = idx % V_SLOW;
                float d_pre = dS_slow[idx] * dtanh_slow[idx];
                dS_slow[idx] = d_pre * decay_slow_val + (-d_v_slow[j]) * slow_gate_val * k_slow[i];
            }
            __syncthreads();
        }
    }
}


// ============================================================================
// Dispatch functions
// ============================================================================

void dispatch_e90_dual_rate_forward(
    int T, int B, int H,
    int k_fast, int v_fast, int k_slow, int v_slow,
    const __nv_bfloat16* k_fast_all, const __nv_bfloat16* v_fast_all,
    const __nv_bfloat16* q_fast_all, const __nv_bfloat16* decay_fast_all,
    const __nv_bfloat16* k_slow_all, const __nv_bfloat16* v_slow_all,
    const __nv_bfloat16* q_slow_all, const __nv_bfloat16* decay_slow_all,
    const __nv_bfloat16* slow_gate_all,
    const __nv_bfloat16* mix_fast_all, const __nv_bfloat16* mix_slow_all,
    __nv_bfloat16* S_fast, __nv_bfloat16* S_slow,
    __nv_bfloat16* output,
    __nv_bfloat16* S_fast_checkpoints, __nv_bfloat16* S_slow_checkpoints,
    int out_v_dim,
    cudaStream_t stream
) {
    int num_blocks = B * H;
    int threads = 256;
    int checkpoint_interval = E90_CHECKPOINT_INTERVAL;

    // Calculate shared memory requirement
    size_t fast_shared = (k_fast * v_fast + 2 * k_fast + 2 * v_fast) * sizeof(float);
    size_t slow_shared = (k_slow * v_slow + 2 * k_slow + 2 * v_slow) * sizeof(float);
    size_t out_shared = (v_fast + v_slow) * sizeof(float);
    size_t shared_mem_size = fast_shared + slow_shared + out_shared + 32;

    #define DISPATCH_E90_FWD(KF, VF, KS, VS) \
        E90DualRateForwardKernel_BF16<KF, VF, KS, VS><<<num_blocks, threads, shared_mem_size, stream>>>( \
            T, B, H, \
            k_fast_all, v_fast_all, q_fast_all, decay_fast_all, \
            k_slow_all, v_slow_all, q_slow_all, decay_slow_all, slow_gate_all, \
            mix_fast_all, mix_slow_all, \
            S_fast, S_slow, output, \
            S_fast_checkpoints, S_slow_checkpoints, \
            out_v_dim, checkpoint_interval)

    // Common configurations
    if (k_fast == 16 && v_fast == 16 && k_slow == 48 && v_slow == 48) {
        DISPATCH_E90_FWD(16, 16, 48, 48);
    }
    else if (k_fast == 16 && v_fast == 16 && k_slow == 32 && v_slow == 32) {
        DISPATCH_E90_FWD(16, 16, 32, 32);
    }
    else if (k_fast == 16 && v_fast == 16 && k_slow == 64 && v_slow == 64) {
        DISPATCH_E90_FWD(16, 16, 64, 64);
    }
    else if (k_fast == 8 && v_fast == 8 && k_slow == 24 && v_slow == 24) {
        DISPATCH_E90_FWD(8, 8, 24, 24);
    }
    else if (k_fast == 8 && v_fast == 8 && k_slow == 16 && v_slow == 16) {
        DISPATCH_E90_FWD(8, 8, 16, 16);
    }
    else if (k_fast == 24 && v_fast == 24 && k_slow == 64 && v_slow == 64) {
        DISPATCH_E90_FWD(24, 24, 64, 64);
    }
    else {
        fprintf(stderr, "E90 Forward: unsupported config k_fast=%d, v_fast=%d, k_slow=%d, v_slow=%d\n",
                k_fast, v_fast, k_slow, v_slow);
    }

    #undef DISPATCH_E90_FWD
}

void dispatch_e90_dual_rate_backward(
    int T, int B, int H,
    int k_fast, int v_fast, int k_slow, int v_slow,
    const __nv_bfloat16* k_fast_all, const __nv_bfloat16* v_fast_all,
    const __nv_bfloat16* q_fast_all, const __nv_bfloat16* decay_fast_all,
    const __nv_bfloat16* k_slow_all, const __nv_bfloat16* v_slow_all,
    const __nv_bfloat16* q_slow_all, const __nv_bfloat16* decay_slow_all,
    const __nv_bfloat16* slow_gate_all,
    const __nv_bfloat16* mix_fast_all, const __nv_bfloat16* mix_slow_all,
    const __nv_bfloat16* d_output,
    const __nv_bfloat16* S_fast_checkpoints, const __nv_bfloat16* S_slow_checkpoints,
    __nv_bfloat16* seg_fast_cache, __nv_bfloat16* seg_slow_cache,
    __nv_bfloat16* d_k_fast_all, __nv_bfloat16* d_v_fast_all,
    __nv_bfloat16* d_q_fast_all, __nv_bfloat16* d_decay_fast_all,
    __nv_bfloat16* d_k_slow_all, __nv_bfloat16* d_v_slow_all,
    __nv_bfloat16* d_q_slow_all, __nv_bfloat16* d_decay_slow_all,
    __nv_bfloat16* d_slow_gate_all,
    __nv_bfloat16* d_mix_fast_all, __nv_bfloat16* d_mix_slow_all,
    int out_v_dim,
    cudaStream_t stream
) {
    int num_blocks = B * H;
    int threads = 256;
    int checkpoint_interval = E90_CHECKPOINT_INTERVAL;

    // Calculate shared memory for backward (much larger than forward)
    // Fast: 2*state + S_t + dtanh + k + v + q + delta + retrieved + d_k + d_v + d_q + out + d_out
    // = 4*state + 5*K_FAST + 6*V_FAST
    size_t fast_shared = (4 * k_fast * v_fast + 5 * k_fast + 6 * v_fast) * sizeof(float);
    // Slow: same pattern
    size_t slow_shared = (4 * k_slow * v_slow + 5 * k_slow + 6 * v_slow) * sizeof(float);
    // Plus output and warp reduction
    size_t out_shared = (out_v_dim + 32) * sizeof(float);
    size_t shared_mem_size = fast_shared + slow_shared + out_shared;

    #define DISPATCH_E90_BWD(KF, VF, KS, VS) \
        E90DualRateBackwardKernel_BF16<KF, VF, KS, VS><<<num_blocks, threads, shared_mem_size, stream>>>( \
            T, B, H, \
            k_fast_all, v_fast_all, q_fast_all, decay_fast_all, \
            k_slow_all, v_slow_all, q_slow_all, decay_slow_all, slow_gate_all, \
            mix_fast_all, mix_slow_all, \
            d_output, S_fast_checkpoints, S_slow_checkpoints, \
            seg_fast_cache, seg_slow_cache, \
            d_k_fast_all, d_v_fast_all, d_q_fast_all, d_decay_fast_all, \
            d_k_slow_all, d_v_slow_all, d_q_slow_all, d_decay_slow_all, \
            d_slow_gate_all, d_mix_fast_all, d_mix_slow_all, \
            out_v_dim, checkpoint_interval)

    if (k_fast == 16 && v_fast == 16 && k_slow == 48 && v_slow == 48) {
        DISPATCH_E90_BWD(16, 16, 48, 48);
    }
    else if (k_fast == 16 && v_fast == 16 && k_slow == 32 && v_slow == 32) {
        DISPATCH_E90_BWD(16, 16, 32, 32);
    }
    else if (k_fast == 16 && v_fast == 16 && k_slow == 64 && v_slow == 64) {
        DISPATCH_E90_BWD(16, 16, 64, 64);
    }
    else if (k_fast == 8 && v_fast == 8 && k_slow == 24 && v_slow == 24) {
        DISPATCH_E90_BWD(8, 8, 24, 24);
    }
    else if (k_fast == 8 && v_fast == 8 && k_slow == 16 && v_slow == 16) {
        DISPATCH_E90_BWD(8, 8, 16, 16);
    }
    else if (k_fast == 24 && v_fast == 24 && k_slow == 64 && v_slow == 64) {
        DISPATCH_E90_BWD(24, 24, 64, 64);
    }
    else {
        fprintf(stderr, "E90 Backward: unsupported config k_fast=%d, v_fast=%d, k_slow=%d, v_slow=%d\n",
                k_fast, v_fast, k_slow, v_slow);
    }

    #undef DISPATCH_E90_BWD
}

}  // namespace elman
