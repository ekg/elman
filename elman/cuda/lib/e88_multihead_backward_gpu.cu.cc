/**
 * E88 Multi-Head Backward Kernel
 *
 * Optimization: Process 4 heads per block using 4 warps instead of 1 head/1 warp.
 *
 * Current register-owned kernel:
 *   - 1 block per (batch, head) pair → B*H blocks
 *   - 1 warp (32 threads) per block
 *   - Low occupancy (only 1 warp active per SM)
 *
 * This kernel:
 *   - 1 block per (batch, head_group) pair → B*(H/4) blocks
 *   - 4 warps (128 threads) per block
 *   - Each warp handles one head independently
 *   - 4x better occupancy
 *
 * Requirements:
 *   - H must be divisible by HEADS_PER_BLOCK (4)
 *   - n_state, head_v_dim <= 32 (fits in single warp)
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include "hasty/elman_ladder.h"

#define HEADS_PER_BLOCK 4
#define THREADS_PER_BLOCK (32 * HEADS_PER_BLOCK)  // 128 threads = 4 warps

namespace elman {

__device__ __forceinline__ float multihead_tanh(float x) {
    return tanhf(x);
}

/**
 * Multi-head register-owned backward kernel
 *
 * Block layout: 4 warps per block, each warp handles 1 head
 * Thread layout: warp j handles head (block_head_offset + j)
 *                within each warp, thread k owns column k of state matrix
 */
template<int N_STATE, int HEAD_V_DIM>
__global__ void E88MultiHeadBackwardKernel_BF16(
    int T,
    int B,
    int H,
    const __nv_bfloat16* __restrict__ k_all,
    const __nv_bfloat16* __restrict__ v_all,
    const __nv_bfloat16* __restrict__ q_all,
    const __nv_bfloat16* __restrict__ decay_all,
    const __nv_bfloat16* __restrict__ g_all,
    const __nv_bfloat16* __restrict__ S_checkpoints,
    const __nv_bfloat16* __restrict__ Sq_cache,
    const __nv_bfloat16* __restrict__ d_output,
    __nv_bfloat16* __restrict__ d_k_all,
    __nv_bfloat16* __restrict__ d_v_all,
    __nv_bfloat16* __restrict__ d_q_all,
    __nv_bfloat16* __restrict__ d_decay_all,
    __nv_bfloat16* __restrict__ d_g_all,
    __nv_bfloat16* __restrict__ segment_cache,
    int checkpoint_interval,
    bool has_gate
) {
    static_assert(HEAD_V_DIM <= 32, "HEAD_V_DIM must be <= 32 for single-warp processing");
    static_assert(N_STATE <= 64, "N_STATE must be <= 64 for register budget");

    // Determine which warp this thread belongs to
    int tid = threadIdx.x;
    int warp_id = tid / 32;           // 0..3
    int lane_id = tid % 32;           // 0..31

    // Determine batch and head group
    int block_idx = blockIdx.x;
    int heads_per_group = HEADS_PER_BLOCK;
    int num_head_groups = (H + heads_per_group - 1) / heads_per_group;

    int b = block_idx / num_head_groups;
    int head_group = block_idx % num_head_groups;
    int h = head_group * heads_per_group + warp_id;  // Actual head index

    // Early exit if batch or head out of bounds
    if (b >= B || h >= H) return;

    // Thread lane_id owns column lane_id of state matrix (if active)
    const bool is_active = (lane_id < HEAD_V_DIM);

    int state_size = N_STATE * HEAD_V_DIM;

    // Register state: thread owns one column
    float S_reg[N_STATE];
    float dS_reg[N_STATE];
    float S_t_reg[N_STATE];
    float dtanh_reg[N_STATE];

    #pragma unroll
    for (int i = 0; i < N_STATE; i++) {
        S_reg[i] = 0.0f;
        dS_reg[i] = 0.0f;
    }

    // Shared memory: separate bank for each warp to avoid conflicts
    // Layout: [HEADS_PER_BLOCK][max(N_STATE, HEAD_V_DIM) + 1]
    extern __shared__ float shared_mem[];

    // Each warp gets its own shared memory region
    int shared_stride = (N_STATE > HEAD_V_DIM ? N_STATE : HEAD_V_DIM) + 2;  // +2 for decay + padding
    float* warp_k_shared = shared_mem + warp_id * shared_stride;
    float* warp_decay_ptr = warp_k_shared + (N_STATE > HEAD_V_DIM ? N_STATE : HEAD_V_DIM);

    // Segment cache: each (b, h) has its own cache region
    int cache_entry_size = state_size + N_STATE + HEAD_V_DIM + 1;
    __nv_bfloat16* seg_cache_base = segment_cache + (size_t)(b * H + h) * checkpoint_interval * cache_entry_size;
    __nv_bfloat16* S_cache_base = seg_cache_base;
    __nv_bfloat16* k_cache_base = seg_cache_base + (size_t)checkpoint_interval * state_size;
    __nv_bfloat16* v_cache_base = k_cache_base + (size_t)checkpoint_interval * N_STATE;
    __nv_bfloat16* decay_cache_base = v_cache_base + (size_t)checkpoint_interval * HEAD_V_DIM;

    int num_segments = (T + checkpoint_interval - 1) / checkpoint_interval;

    for (int seg = num_segments - 1; seg >= 0; seg--) {
        int t_start = seg * checkpoint_interval;
        int t_end = min(t_start + checkpoint_interval, T);
        int seg_len = t_end - t_start;

        // Phase 1: Load checkpoint into registers
        int cp_offset = (seg * B * H + b * H + h) * state_size;
        if (is_active) {
            #pragma unroll
            for (int i = 0; i < N_STATE; i++) {
                S_reg[i] = __bfloat162float(S_checkpoints[cp_offset + i * HEAD_V_DIM + lane_id]);
            }
        }

        // Phase 2: Forward replay through segment
        for (int local_t = 0; local_t < seg_len; local_t++) {
            int t = t_start + local_t;

            // Cache S_{t-1}
            __nv_bfloat16* S_cache_slot = S_cache_base + (size_t)local_t * state_size;
            if (is_active) {
                #pragma unroll
                for (int i = 0; i < N_STATE; i++) {
                    S_cache_slot[i * HEAD_V_DIM + lane_id] = __float2bfloat16(S_reg[i]);
                }
            }

            // Load inputs cooperatively within warp
            int k_offset = ((b * T + t) * H + h) * N_STATE;
            int v_offset = ((b * T + t) * H + h) * HEAD_V_DIM;
            int decay_offset = (b * T + t) * H + h;

            // Warp-cooperative load
            if (lane_id < N_STATE) {
                warp_k_shared[lane_id] = __bfloat162float(k_all[k_offset + lane_id]);
            }
            __syncwarp();

            float v_val = 0.0f;
            if (lane_id < HEAD_V_DIM) {
                v_val = __bfloat162float(v_all[v_offset + lane_id]);
            }

            float decay_val;
            if (lane_id == 0) {
                decay_val = __bfloat162float(decay_all[decay_offset]);
                *warp_decay_ptr = decay_val;
            }
            __syncwarp();
            decay_val = *warp_decay_ptr;

            // Cache k, v, decay
            __nv_bfloat16* k_cache_slot = k_cache_base + (size_t)local_t * N_STATE;
            __nv_bfloat16* v_cache_slot = v_cache_base + (size_t)local_t * HEAD_V_DIM;
            if (lane_id < N_STATE) {
                k_cache_slot[lane_id] = __float2bfloat16(warp_k_shared[lane_id]);
            }
            if (lane_id < HEAD_V_DIM) {
                v_cache_slot[lane_id] = __float2bfloat16(v_val);
            }
            if (lane_id == 0) {
                decay_cache_base[local_t] = __float2bfloat16(decay_val);
            }
            __syncwarp();

            // Compute retrieved and delta
            float retrieved = 0.0f;
            float delta = 0.0f;
            if (is_active) {
                #pragma unroll
                for (int i = 0; i < N_STATE; i++) {
                    retrieved += S_reg[i] * warp_k_shared[i];
                }
                delta = v_val - retrieved;

                // State update: S = tanh(decay * S + delta * k)
                #pragma unroll
                for (int i = 0; i < N_STATE; i++) {
                    float pre_tanh = decay_val * S_reg[i] + delta * warp_k_shared[i];
                    S_reg[i] = multihead_tanh(pre_tanh);
                }
            }
        }

        // Phase 3: Backward pass through segment
        for (int local_t = seg_len - 1; local_t >= 0; local_t--) {
            int t = t_start + local_t;

            // Load cached S_{t-1}
            __nv_bfloat16* S_cache_slot = S_cache_base + (size_t)local_t * state_size;
            if (is_active) {
                #pragma unroll
                for (int i = 0; i < N_STATE; i++) {
                    S_reg[i] = __bfloat162float(S_cache_slot[i * HEAD_V_DIM + lane_id]);
                }
            }

            // Load cached k, v, decay
            __nv_bfloat16* k_cache_slot = k_cache_base + (size_t)local_t * N_STATE;
            __nv_bfloat16* v_cache_slot = v_cache_base + (size_t)local_t * HEAD_V_DIM;

            if (lane_id < N_STATE) {
                warp_k_shared[lane_id] = __bfloat162float(k_cache_slot[lane_id]);
            }
            __syncwarp();

            float v_val = (lane_id < HEAD_V_DIM) ? __bfloat162float(v_cache_slot[lane_id]) : 0.0f;
            float decay_val;
            if (lane_id == 0) {
                decay_val = __bfloat162float(decay_cache_base[local_t]);
                *warp_decay_ptr = decay_val;
            }
            __syncwarp();
            decay_val = *warp_decay_ptr;

            // Load q
            int k_offset = ((b * T + t) * H + h) * N_STATE;
            int v_offset = ((b * T + t) * H + h) * HEAD_V_DIM;
            int decay_offset = (b * T + t) * H + h;
            float q_val = (lane_id < N_STATE) ? __bfloat162float(q_all[k_offset + lane_id]) : 0.0f;

            // Recompute forward values
            float retrieved = 0.0f;
            float delta = 0.0f;
            if (is_active) {
                #pragma unroll
                for (int i = 0; i < N_STATE; i++) {
                    retrieved += S_reg[i] * warp_k_shared[i];
                }
                delta = v_val - retrieved;

                #pragma unroll
                for (int i = 0; i < N_STATE; i++) {
                    float pre_tanh = decay_val * S_reg[i] + delta * warp_k_shared[i];
                    float tanh_val = multihead_tanh(pre_tanh);
                    S_t_reg[i] = tanh_val;
                    dtanh_reg[i] = 1.0f - tanh_val * tanh_val;
                }
            }

            // Load d_output and handle gating
            float d_out = is_active ? __bfloat162float(d_output[v_offset + lane_id]) : 0.0f;
            float d_Sq = 0.0f;

            if (is_active) {
                if (has_gate && g_all != nullptr) {
                    float g_val = __bfloat162float(g_all[v_offset + lane_id]);
                    float Sq_val = __bfloat162float(Sq_cache[v_offset + lane_id]);

                    float sig_g = 1.0f / (1.0f + expf(-g_val));
                    float silu_g = g_val * sig_g;

                    d_Sq = d_out * silu_g;

                    float d_silu = sig_g * (1.0f + g_val * (1.0f - sig_g));
                    float Sq_before_gate = Sq_val / (silu_g + 1e-8f);
                    d_g_all[v_offset + lane_id] = __float2bfloat16(d_out * Sq_before_gate * d_silu);
                } else {
                    d_Sq = d_out;
                }
            }

            // d_q[i] = sum_j(S_t[i,j] * d_Sq[j])
            float d_q_contrib[N_STATE];
            #pragma unroll
            for (int i = 0; i < N_STATE; i++) {
                d_q_contrib[i] = is_active ? (S_t_reg[i] * d_Sq) : 0.0f;
            }

            // Warp reduction for d_q
            float d_q_local[N_STATE];
            #pragma unroll
            for (int i = 0; i < N_STATE; i++) {
                float val = d_q_contrib[i];
                #pragma unroll
                for (int offset = 16; offset >= 1; offset /= 2) {
                    val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
                }
                d_q_local[i] = val;
            }
            if (lane_id < N_STATE) {
                d_q_all[k_offset + lane_id] = __float2bfloat16(d_q_local[lane_id]);
            }

            // dS contribution from output
            #pragma unroll
            for (int i = 0; i < N_STATE; i++) {
                float q_i = __shfl_sync(0xFFFFFFFF, q_val, i);
                if (is_active) {
                    dS_reg[i] += q_i * d_Sq;
                }
            }

            // d_delta
            float d_delta = 0.0f;
            if (is_active) {
                #pragma unroll
                for (int i = 0; i < N_STATE; i++) {
                    float d_pre = dS_reg[i] * dtanh_reg[i];
                    d_delta += d_pre * warp_k_shared[i];
                }
                d_v_all[v_offset + lane_id] = __float2bfloat16(d_delta);
            }

            // d_k
            float d_k_contrib[N_STATE];
            #pragma unroll
            for (int i = 0; i < N_STATE; i++) {
                if (is_active) {
                    float d_pre = dS_reg[i] * dtanh_reg[i];
                    d_k_contrib[i] = d_pre * delta + S_reg[i] * (-d_delta);
                } else {
                    d_k_contrib[i] = 0.0f;
                }
            }

            #pragma unroll
            for (int i = 0; i < N_STATE; i++) {
                float val = d_k_contrib[i];
                #pragma unroll
                for (int offset = 16; offset >= 1; offset /= 2) {
                    val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
                }
                if (lane_id == i && i < N_STATE) {
                    d_k_all[k_offset + i] = __float2bfloat16(val);
                }
            }

            // d_decay
            float d_decay_local = 0.0f;
            if (is_active) {
                #pragma unroll
                for (int i = 0; i < N_STATE; i++) {
                    d_decay_local += dS_reg[i] * dtanh_reg[i] * S_reg[i];
                }
            }
            #pragma unroll
            for (int offset = 16; offset >= 1; offset /= 2) {
                d_decay_local += __shfl_xor_sync(0xFFFFFFFF, d_decay_local, offset);
            }
            if (lane_id == 0) {
                d_decay_all[decay_offset] = __float2bfloat16(d_decay_local);
            }

            // Update dS for next iteration
            if (is_active) {
                #pragma unroll
                for (int i = 0; i < N_STATE; i++) {
                    float d_pre = dS_reg[i] * dtanh_reg[i];
                    dS_reg[i] = d_pre * decay_val + (-d_delta) * warp_k_shared[i];
                }
            }
        }
    }
}

// Dispatch macro
#define LAUNCH_MULTIHEAD_KERNEL(NS, HVD) \
    do { \
        int shared_mem = HEADS_PER_BLOCK * ((NS > HVD ? NS : HVD) + 2) * sizeof(float); \
        E88MultiHeadBackwardKernel_BF16<NS, HVD><<<num_blocks, THREADS_PER_BLOCK, shared_mem, stream>>>( \
            T, B, H, k, v, q, decay, g, S_checkpoints, Sq_cache, d_output, \
            d_k, d_v, d_q, d_decay, d_g, segment_cache, \
            checkpoint_interval, has_gate \
        ); \
    } while(0)

// Dispatch function
void dispatch_e88_multihead_backward(
    int T, int B, int H, int n_state, int head_v_dim,
    const __nv_bfloat16* k, const __nv_bfloat16* v, const __nv_bfloat16* q,
    const __nv_bfloat16* decay, const __nv_bfloat16* g,
    const __nv_bfloat16* S_checkpoints, const __nv_bfloat16* Sq_cache,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* d_k, __nv_bfloat16* d_v, __nv_bfloat16* d_q,
    __nv_bfloat16* d_decay, __nv_bfloat16* d_g,
    __nv_bfloat16* segment_cache,
    int checkpoint_interval, bool has_gate, cudaStream_t stream
) {
    if (head_v_dim > 32) {
        fprintf(stderr, "MultiHead kernel requires head_v_dim<=32, got %d\n", head_v_dim);
        return;
    }
    if (n_state > 64) {
        fprintf(stderr, "MultiHead kernel requires n_state<=64, got %d\n", n_state);
        return;
    }
    if (H % HEADS_PER_BLOCK != 0) {
        fprintf(stderr, "MultiHead kernel requires H divisible by %d, got H=%d\n", HEADS_PER_BLOCK, H);
        return;
    }

    int num_head_groups = H / HEADS_PER_BLOCK;
    int num_blocks = B * num_head_groups;

    // Dispatch based on size
    if (n_state == 32 && head_v_dim == 32) {
        LAUNCH_MULTIHEAD_KERNEL(32, 32);
    }
    else if (n_state == 24 && head_v_dim == 24) {
        LAUNCH_MULTIHEAD_KERNEL(24, 24);
    }
    else if (n_state == 16 && head_v_dim == 16) {
        LAUNCH_MULTIHEAD_KERNEL(16, 16);
    }
    else if (n_state == 8 && head_v_dim == 8) {
        LAUNCH_MULTIHEAD_KERNEL(8, 8);
    }
    else if (n_state == 64 && head_v_dim == 32) {
        LAUNCH_MULTIHEAD_KERNEL(64, 32);
    }
    else if (n_state == 48 && head_v_dim == 32) {
        LAUNCH_MULTIHEAD_KERNEL(48, 32);
    }
    else if (n_state == 40 && head_v_dim == 32) {
        LAUNCH_MULTIHEAD_KERNEL(40, 32);
    }
    else if (n_state == 16 && head_v_dim == 32) {
        LAUNCH_MULTIHEAD_KERNEL(16, 32);
    }
    else if (n_state == 24 && head_v_dim == 32) {
        LAUNCH_MULTIHEAD_KERNEL(24, 32);
    }
    else {
        fprintf(stderr, "Unsupported n_state=%d, head_v_dim=%d for multihead kernel\n",
                n_state, head_v_dim);
    }
}

#undef LAUNCH_MULTIHEAD_KERNEL

}  // namespace elman
