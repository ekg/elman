/**
 * E88 Register-Owned State Backward Kernel
 *
 * Key optimization: Each thread owns ONE FULL COLUMN of the 32x32 state matrix
 * in registers (32 floats = 128 bytes per thread, well within register limits).
 *
 * With 32 threads per block:
 * - Thread j owns S[:,j] (column j of the state matrix)
 * - Thread j computes retrieved[j], delta[j], d_k[j], d_v[j], etc.
 * - Cross-thread communication via warp shuffles (no __syncthreads needed!)
 *
 * Benefits:
 * - No shared memory for state (state lives in registers)
 * - No __syncthreads for state operations (warp is lockstep)
 * - Full utilization: all 32 threads active for all operations
 * - Expected speedup: 2-3x over fused_backward
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include "hasty/elman_ladder.h"

#define E88_REG_CHECKPOINT_INTERVAL 16

namespace elman {

__device__ __forceinline__ float e88_reg_tanh(float x) {
    return tanhf(x);
}

__device__ __forceinline__ float e88_reg_silu(float x) {
    return x / (1.0f + expf(-x));
}

/**
 * Register-owned backward kernel for N_STATE=32, HEAD_V_DIM=32
 *
 * Thread layout: 32 threads per block (1 warp)
 * Thread j owns column j of S (32 elements in registers)
 */
template<int N_STATE, int HEAD_V_DIM>
__global__ void E88RegisterOwnedBackwardKernel_BF16(
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
    static_assert(N_STATE == 32 && HEAD_V_DIM == 32,
                  "Register-owned kernel only supports N_STATE=32, HEAD_V_DIM=32");

    int block_idx = blockIdx.x;
    int b = block_idx / H;
    int h = block_idx % H;
    if (b >= B) return;

    int tid = threadIdx.x;  // tid in [0, 31]
    int state_size = N_STATE * HEAD_V_DIM;  // 1024

    // Thread j owns column j of the state matrix
    // S_reg[i] = S[i, j] where j = tid
    float S_reg[N_STATE];     // State column (32 floats)
    float dS_reg[N_STATE];    // Gradient of state column (32 floats)
    float S_t_reg[N_STATE];   // Post-tanh state column
    float dtanh_reg[N_STATE]; // 1 - tanh^2 values

    // Initialize dS to zero
    #pragma unroll
    for (int i = 0; i < N_STATE; i++) {
        dS_reg[i] = 0.0f;
    }

    // Small shared memory for cached values during replay
    __shared__ float k_shared[N_STATE];
    __shared__ float v_shared[HEAD_V_DIM];
    __shared__ float decay_shared;

    // Segment cache pointers
    int cache_entry_size = state_size + N_STATE + HEAD_V_DIM + 1;
    __nv_bfloat16* seg_cache_base = segment_cache + (size_t)block_idx * checkpoint_interval * cache_entry_size;
    __nv_bfloat16* S_cache_base = seg_cache_base;
    __nv_bfloat16* k_cache_base = seg_cache_base + (size_t)checkpoint_interval * state_size;
    __nv_bfloat16* v_cache_base = k_cache_base + (size_t)checkpoint_interval * N_STATE;
    __nv_bfloat16* decay_cache_base = v_cache_base + (size_t)checkpoint_interval * HEAD_V_DIM;

    int num_segments = (T + checkpoint_interval - 1) / checkpoint_interval;

    for (int seg = num_segments - 1; seg >= 0; seg--) {
        int t_start = seg * checkpoint_interval;
        int t_end = min(t_start + checkpoint_interval, T);
        int seg_len = t_end - t_start;

        // Phase 1: Load checkpoint into registers (32 threads load 32 elements each)
        int cp_offset = (seg * B * H + b * H + h) * state_size;
        #pragma unroll
        for (int i = 0; i < N_STATE; i++) {
            S_reg[i] = __bfloat162float(S_checkpoints[cp_offset + i * HEAD_V_DIM + tid]);
        }

        // Phase 2: Forward replay through segment to cache intermediate values
        for (int local_t = 0; local_t < seg_len; local_t++) {
            int t = t_start + local_t;

            // Cache S_{t-1} before update
            __nv_bfloat16* S_cache_slot = S_cache_base + (size_t)local_t * state_size;
            #pragma unroll
            for (int i = 0; i < N_STATE; i++) {
                S_cache_slot[i * HEAD_V_DIM + tid] = __float2bfloat16(S_reg[i]);
            }

            // Load inputs (cooperative load into shared memory)
            int k_offset = ((b * T + t) * H + h) * N_STATE;
            int v_offset = ((b * T + t) * H + h) * HEAD_V_DIM;
            int decay_offset = (b * T + t) * H + h;

            // Warp-cooperative load: thread tid loads k[tid] and v[tid]
            k_shared[tid] = __bfloat162float(k_all[k_offset + tid]);
            v_shared[tid] = __bfloat162float(v_all[v_offset + tid]);
            if (tid == 0) {
                decay_shared = __bfloat162float(decay_all[decay_offset]);
            }
            __syncwarp();  // Warp sync is very cheap

            // Cache k, v, decay
            __nv_bfloat16* k_cache_slot = k_cache_base + (size_t)local_t * N_STATE;
            __nv_bfloat16* v_cache_slot = v_cache_base + (size_t)local_t * HEAD_V_DIM;
            k_cache_slot[tid] = __float2bfloat16(k_shared[tid]);
            v_cache_slot[tid] = __float2bfloat16(v_shared[tid]);
            if (tid == 0) {
                decay_cache_base[local_t] = __float2bfloat16(decay_shared);
            }
            __syncwarp();

            // Compute retrieved[tid] = sum_i(S[i, tid] * k[i])
            float retrieved = 0.0f;
            #pragma unroll
            for (int i = 0; i < N_STATE; i++) {
                retrieved += S_reg[i] * k_shared[i];
            }

            // delta[tid] = v[tid] - retrieved[tid]
            float delta = v_shared[tid] - retrieved;

            // Update state: S[i, tid] = tanh(decay * S[i, tid] + delta * k[i])
            #pragma unroll
            for (int i = 0; i < N_STATE; i++) {
                float pre_tanh = decay_shared * S_reg[i] + delta * k_shared[i];
                S_reg[i] = e88_reg_tanh(pre_tanh);
            }
        }

        // Phase 3: Backward pass through segment
        for (int local_t = seg_len - 1; local_t >= 0; local_t--) {
            int t = t_start + local_t;

            // Load cached S_{t-1} into registers
            __nv_bfloat16* S_cache_slot = S_cache_base + (size_t)local_t * state_size;
            #pragma unroll
            for (int i = 0; i < N_STATE; i++) {
                S_reg[i] = __bfloat162float(S_cache_slot[i * HEAD_V_DIM + tid]);
            }

            // Load cached k, v into shared memory
            __nv_bfloat16* k_cache_slot = k_cache_base + (size_t)local_t * N_STATE;
            __nv_bfloat16* v_cache_slot = v_cache_base + (size_t)local_t * HEAD_V_DIM;
            k_shared[tid] = __bfloat162float(k_cache_slot[tid]);
            v_shared[tid] = __bfloat162float(v_cache_slot[tid]);
            if (tid == 0) {
                decay_shared = __bfloat162float(decay_cache_base[local_t]);
            }
            __syncwarp();

            // Load q (reuse k loading pattern)
            int k_offset = ((b * T + t) * H + h) * N_STATE;
            int v_offset = ((b * T + t) * H + h) * HEAD_V_DIM;
            int decay_offset = (b * T + t) * H + h;
            float q_val = __bfloat162float(q_all[k_offset + tid]);

            // Recompute forward values
            // retrieved[tid] = sum_i(S[i, tid] * k[i])
            float retrieved = 0.0f;
            #pragma unroll
            for (int i = 0; i < N_STATE; i++) {
                retrieved += S_reg[i] * k_shared[i];
            }
            float delta = v_shared[tid] - retrieved;

            // Compute S_t and dtanh
            #pragma unroll
            for (int i = 0; i < N_STATE; i++) {
                float pre_tanh = decay_shared * S_reg[i] + delta * k_shared[i];
                float tanh_val = e88_reg_tanh(pre_tanh);
                S_t_reg[i] = tanh_val;
                dtanh_reg[i] = 1.0f - tanh_val * tanh_val;
            }

            // Load d_output and handle gating
            float d_out = __bfloat162float(d_output[v_offset + tid]);
            float d_Sq;

            if (has_gate && g_all != nullptr) {
                float g_val = __bfloat162float(g_all[v_offset + tid]);
                float Sq_val = __bfloat162float(Sq_cache[v_offset + tid]);

                float sig_g = 1.0f / (1.0f + expf(-g_val));
                float silu_g = g_val * sig_g;

                d_Sq = d_out * silu_g;

                float d_silu = sig_g * (1.0f + g_val * (1.0f - sig_g));
                float Sq_before_gate = Sq_val / (silu_g + 1e-8f);
                d_g_all[v_offset + tid] = __float2bfloat16(d_out * Sq_before_gate * d_silu);
            } else {
                d_Sq = d_out;
            }

            // d_q[i] = sum_j(S_t[i, j] * d_Sq[j])
            // Thread tid has d_Sq[tid], need to compute d_q[i] for all i
            // Use warp shuffle to share d_Sq values

            // Each thread computes d_q[tid] = sum_j(S_t[tid, j] * d_Sq[j])
            // But thread tid doesn't have S_t[tid, j] for j != tid...
            // Actually thread j has S_t_reg[i] = S_t[i, j]
            // So d_q[i] = sum_j(S_t[i, j] * d_Sq[j])
            // Thread j contributes S_t[i, j] * d_Sq[j] to d_q[i]

            // First, compute local contribution to d_q
            float d_q_contrib[N_STATE];
            #pragma unroll
            for (int i = 0; i < N_STATE; i++) {
                d_q_contrib[i] = S_t_reg[i] * d_Sq;
            }

            // Now reduce across threads: d_q[i] = sum over j of d_q_contrib[i]
            // Each thread tid has d_q_contrib[i] for i in [0, 31]
            // After reduction, thread i should have d_q[i]

            // Use warp shuffle to sum contributions
            float d_q_local[N_STATE];
            #pragma unroll
            for (int i = 0; i < N_STATE; i++) {
                float val = d_q_contrib[i];
                // Warp reduce
                #pragma unroll
                for (int offset = 16; offset >= 1; offset /= 2) {
                    val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
                }
                d_q_local[i] = val;
            }
            // Now d_q_local[i] is the same across all threads (the full sum)
            // Write d_q[tid]
            d_q_all[k_offset + tid] = __float2bfloat16(d_q_local[tid]);

            // Add dS contribution from output: dS[i, j] += q[i] * d_Sq[j]
            // Thread j has d_Sq[j], needs to add q[i] * d_Sq to dS_reg[i]
            // Need q from other threads via shuffle
            #pragma unroll
            for (int i = 0; i < N_STATE; i++) {
                float q_i = __shfl_sync(0xFFFFFFFF, q_val, i);
                dS_reg[i] += q_i * d_Sq;
            }

            // Backward through state update
            // d_delta[j] = sum_i(dS[i,j] * dtanh[i,j] * k[i])
            float d_delta = 0.0f;
            #pragma unroll
            for (int i = 0; i < N_STATE; i++) {
                float d_pre = dS_reg[i] * dtanh_reg[i];
                d_delta += d_pre * k_shared[i];
            }

            // d_v[j] = d_delta[j]
            d_v_all[v_offset + tid] = __float2bfloat16(d_delta);

            // d_k[i] = sum_j(dS[i,j] * dtanh[i,j] * delta[j]) + sum_j(S[i,j] * (-d_delta[j]))
            // Thread j has dS_reg[i], dtanh_reg[i], delta, S_reg[i], d_delta for column j
            // d_k[i] needs contributions from all j

            // Compute local contribution for each i
            float d_k_contrib[N_STATE];
            #pragma unroll
            for (int i = 0; i < N_STATE; i++) {
                float d_pre = dS_reg[i] * dtanh_reg[i];
                d_k_contrib[i] = d_pre * delta + S_reg[i] * (-d_delta);
            }

            // Warp reduce to get d_k[i] = sum_j(d_k_contrib[i])
            #pragma unroll
            for (int i = 0; i < N_STATE; i++) {
                float val = d_k_contrib[i];
                #pragma unroll
                for (int offset = 16; offset >= 1; offset /= 2) {
                    val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
                }
                if (tid == i) {
                    d_k_all[k_offset + i] = __float2bfloat16(val);
                }
            }

            // d_decay = sum_{i,j}(dS[i,j] * dtanh[i,j] * S[i,j])
            // Thread j contributes sum_i(dS_reg[i] * dtanh_reg[i] * S_reg[i])
            float d_decay_local = 0.0f;
            #pragma unroll
            for (int i = 0; i < N_STATE; i++) {
                d_decay_local += dS_reg[i] * dtanh_reg[i] * S_reg[i];
            }
            // Warp reduce
            #pragma unroll
            for (int offset = 16; offset >= 1; offset /= 2) {
                d_decay_local += __shfl_xor_sync(0xFFFFFFFF, d_decay_local, offset);
            }
            if (tid == 0) {
                d_decay_all[decay_offset] = __float2bfloat16(d_decay_local);
            }

            // Update dS for next iteration (t-1)
            // dS_{t-1}[i,j] = dS_t[i,j] * dtanh[i,j] * decay + (-d_delta[j]) * k[i]
            // Need d_delta[j] from all threads, but we computed d_delta locally
            // Actually d_delta[j] is the local thread's value (thread j has d_delta[j])

            // But for the update, thread j needs -d_delta[j] * k[i] for all i
            // d_delta is local to thread j, k[i] is in shared memory
            #pragma unroll
            for (int i = 0; i < N_STATE; i++) {
                float d_pre = dS_reg[i] * dtanh_reg[i];
                dS_reg[i] = d_pre * decay_shared + (-d_delta) * k_shared[i];
            }
        }
    }
}

// Dispatch function matching existing pattern
void dispatch_e88_register_owned_backward(
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
    if (n_state != 32 || head_v_dim != 32) {
        fprintf(stderr, "Register-owned kernel requires n_state=32 and head_v_dim=32\n");
        return;
    }

    int num_blocks = B * H;
    int threads = 32;  // One warp per block!

    // Small shared memory for k, v, decay
    int shared_mem = (32 + 32 + 1) * sizeof(float);  // k + v + decay

    E88RegisterOwnedBackwardKernel_BF16<32, 32><<<num_blocks, threads, shared_mem, stream>>>(
        T, B, H, k, v, q, decay, g, S_checkpoints, Sq_cache, d_output,
        d_k, d_v, d_q, d_decay, d_g, segment_cache,
        checkpoint_interval, has_gate
    );
}

}  // namespace elman
