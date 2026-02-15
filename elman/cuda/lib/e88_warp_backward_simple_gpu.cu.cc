/**
 * E88 Warp Backward SIMPLIFIED - Fixed version with chunk prefetching
 *
 * Fixes applied (from V2 investigation):
 * 1. Per-timestep loading in forward replay (not bulk prefetch) to match fused kernel
 * 2. Read k, v from segment cache in backward phase (not stale prefetch buffers)
 * 3. Add __syncthreads() between d_delta and d_k computation
 *
 * Optimization: Chunk-based prefetching for backward phase
 * - Bulk load q, d_output, g, Sq_cache for entire segment into shared memory
 *   AFTER forward replay completes (fresh data, not stale)
 * - Backward loop reads from shared memory instead of global memory
 * - Mirrors the forward kernel's chunk prefetch pattern
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include "hasty/elman_ladder.h"

#define E88_WARP_BACKWARD_SIMPLE_CHUNK_SIZE 16

namespace elman {

__device__ __forceinline__ float e88_simple_tanh(float x) {
    return tanhf(x);
}

__device__ __forceinline__ float e88_simple_silu(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float e88_simple_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

template<int N_STATE, int HEAD_V_DIM, int CHUNK_SIZE>
__global__ void E88WarpBackwardSimpleKernel_BF16(
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
    int block_idx = blockIdx.x;
    int b = block_idx / H;
    int h = block_idx % H;
    if (b >= B) return;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    constexpr int state_size = N_STATE * HEAD_V_DIM;

    extern __shared__ float shared_mem[];
    float* S = shared_mem;
    float* dS = S + state_size;
    float* S_t = dS + state_size;
    float* dtanh_buf = S_t + state_size;
    float* k = dtanh_buf + state_size;          // Single timestep k buffer
    float* v = k + N_STATE;                     // Single timestep v buffer
    float* q = v + HEAD_V_DIM;                  // Single timestep q buffer (still used as scratch)
    float* delta_buf = q + N_STATE;
    float* d_delta_buf = delta_buf + HEAD_V_DIM;
    float* d_k_buf = d_delta_buf + HEAD_V_DIM;
    float* d_q_buf = d_k_buf + N_STATE;
    float* d_Sq_buf = d_q_buf + N_STATE;
    float* warp_results = d_Sq_buf + HEAD_V_DIM;

    // Chunk prefetch buffers for backward phase (loaded once per segment)
    float* q_chunk = warp_results + 8;                          // CHUNK_SIZE * N_STATE
    float* d_out_chunk = q_chunk + CHUNK_SIZE * N_STATE;        // CHUNK_SIZE * HEAD_V_DIM
    float* Sq_chunk = d_out_chunk + CHUNK_SIZE * HEAD_V_DIM;    // CHUNK_SIZE * HEAD_V_DIM
    float* g_chunk = Sq_chunk + CHUNK_SIZE * HEAD_V_DIM;        // CHUNK_SIZE * HEAD_V_DIM (conditional, but always allocated)
    float* decay_bwd_chunk = g_chunk + CHUNK_SIZE * HEAD_V_DIM; // CHUNK_SIZE

    // Segment cache
    int cache_entry_size = state_size + N_STATE + HEAD_V_DIM + 1;
    __nv_bfloat16* seg_cache_base = segment_cache + (size_t)block_idx * checkpoint_interval * cache_entry_size;
    __nv_bfloat16* S_cache_base = seg_cache_base;
    __nv_bfloat16* k_cache_base = seg_cache_base + (size_t)checkpoint_interval * state_size;
    __nv_bfloat16* v_cache_base = k_cache_base + (size_t)checkpoint_interval * N_STATE;
    __nv_bfloat16* decay_cache_base = v_cache_base + (size_t)checkpoint_interval * HEAD_V_DIM;

    // Initialize dS to zero
    for (int i = tid; i < state_size; i += num_threads) {
        dS[i] = 0.0f;
    }
    __syncthreads();

    int num_segments = (T + checkpoint_interval - 1) / checkpoint_interval;

    for (int seg = num_segments - 1; seg >= 0; seg--) {
        int t_start = seg * checkpoint_interval;
        int t_end = min(t_start + checkpoint_interval, T);
        int seg_len = t_end - t_start;

        // Phase 1: Forward replay
        int cp_offset = (seg * B * H + b * H + h) * state_size;
        for (int i = tid; i < state_size; i += num_threads) {
            S[i] = __bfloat162float(S_checkpoints[cp_offset + i]);
        }
        __syncthreads();

        // Forward replay - load per-timestep like fused kernel (FIX #1)
        __shared__ float decay_val_replay;
        for (int local_t = 0; local_t < seg_len; local_t++) {
            int t = t_start + local_t;

            // Save S_{t-1} before update
            __nv_bfloat16* S_cache_slot = S_cache_base + (size_t)local_t * state_size;
            for (int i = tid; i < state_size; i += num_threads) {
                S_cache_slot[i] = __float2bfloat16(S[i]);
            }

            // Load inputs per-timestep from global memory (matching fused kernel exactly)
            int k_offset = ((b * T + t) * H + h) * N_STATE;
            int v_offset = ((b * T + t) * H + h) * HEAD_V_DIM;
            int decay_offset = (b * T + t) * H + h;

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

            if (tid < HEAD_V_DIM) {
                float sum = 0.0f;
                #pragma unroll 8
                for (int i = 0; i < N_STATE; i++) {
                    sum += S[i * HEAD_V_DIM + tid] * k[i];
                }
                delta_buf[tid] = v[tid] - sum;
            }
            __syncthreads();

            // Simple loop-based state update (like fused backward)
            for (int idx = tid; idx < state_size; idx += num_threads) {
                int i = idx / HEAD_V_DIM;
                int j = idx % HEAD_V_DIM;
                S[idx] = e88_simple_tanh(decay_val_replay * S[idx] + k[i] * delta_buf[j]);
            }
            __syncthreads();
        }

        // Phase 2: Bulk prefetch q, d_output, g, Sq_cache for entire segment
        // These are read-only during backward and benefit from shared memory caching
        for (int i = tid; i < seg_len * N_STATE; i += num_threads) {
            int lt = i / N_STATE;
            int qi = i % N_STATE;
            int t = t_start + lt;
            q_chunk[lt * N_STATE + qi] = __bfloat162float(q_all[((b * T + t) * H + h) * N_STATE + qi]);
        }
        for (int i = tid; i < seg_len * HEAD_V_DIM; i += num_threads) {
            int lt = i / HEAD_V_DIM;
            int vi = i % HEAD_V_DIM;
            int t = t_start + lt;
            d_out_chunk[lt * HEAD_V_DIM + vi] = __bfloat162float(d_output[((b * T + t) * H + h) * HEAD_V_DIM + vi]);
        }
        if (has_gate && g_all != nullptr) {
            for (int i = tid; i < seg_len * HEAD_V_DIM; i += num_threads) {
                int lt = i / HEAD_V_DIM;
                int gi = i % HEAD_V_DIM;
                int t = t_start + lt;
                g_chunk[lt * HEAD_V_DIM + gi] = __bfloat162float(g_all[((b * T + t) * H + h) * HEAD_V_DIM + gi]);
            }
            for (int i = tid; i < seg_len * HEAD_V_DIM; i += num_threads) {
                int lt = i / HEAD_V_DIM;
                int si = i % HEAD_V_DIM;
                int t = t_start + lt;
                Sq_chunk[lt * HEAD_V_DIM + si] = __bfloat162float(Sq_cache[((b * T + t) * H + h) * HEAD_V_DIM + si]);
            }
        }
        for (int i = tid; i < seg_len; i += num_threads) {
            decay_bwd_chunk[i] = __bfloat162float(decay_cache_base[i]);
        }
        __syncthreads();

        // Phase 2: Backward pass - S, k, v from segment cache; q, d_output, g from chunk buffers
        __shared__ float decay_val;
        for (int local_t = seg_len - 1; local_t >= 0; local_t--) {
            int t = t_start + local_t;

            // Load S, k, v from segment cache (like fused backward)
            __nv_bfloat16* S_cache_slot = S_cache_base + (size_t)local_t * state_size;
            __nv_bfloat16* k_cache_slot = k_cache_base + (size_t)local_t * N_STATE;
            __nv_bfloat16* v_cache_slot = v_cache_base + (size_t)local_t * HEAD_V_DIM;

            for (int i = tid; i < state_size; i += num_threads) {
                S[i] = __bfloat162float(S_cache_slot[i]);
            }
            if (tid < N_STATE) {
                k[tid] = __bfloat162float(k_cache_slot[tid]);
            }
            if (tid < HEAD_V_DIM) {
                v[tid] = __bfloat162float(v_cache_slot[tid]);
            }

            // Load q from chunk buffer (replaces global memory read)
            if (tid < N_STATE) {
                q[tid] = q_chunk[local_t * N_STATE + tid];
            }

            // Load decay from chunk buffer
            if (tid == 0) {
                decay_val = decay_bwd_chunk[local_t];
            }
            __syncthreads();

            if (tid < HEAD_V_DIM) {
                float sum = 0.0f;
                #pragma unroll 8
                for (int i = 0; i < N_STATE; i++) {
                    sum += S[i * HEAD_V_DIM + tid] * k[i];
                }
                delta_buf[tid] = v[tid] - sum;
            }
            __syncthreads();

            // Simple loop-based S_t and dtanh computation
            for (int idx = tid; idx < state_size; idx += num_threads) {
                int i = idx / HEAD_V_DIM;
                int j = idx % HEAD_V_DIM;
                float k_val = k[i];
                float d_val = delta_buf[j];
                float pre_tanh = decay_val * S[idx] + k_val * d_val;
                float tv = e88_simple_tanh(pre_tanh);
                S_t[idx] = tv;
                dtanh_buf[idx] = 1.0f - tv * tv;
            }
            __syncthreads();

            int v_offset = ((b * T + t) * H + h) * HEAD_V_DIM;
            int k_offset = ((b * T + t) * H + h) * N_STATE;
            if (tid < HEAD_V_DIM) {
                // Read d_output and g from chunk buffers (replaces global memory reads)
                float d_out = d_out_chunk[local_t * HEAD_V_DIM + tid];

                if (has_gate && g_all != nullptr) {
                    float g_val = g_chunk[local_t * HEAD_V_DIM + tid];
                    float Sq_val = Sq_chunk[local_t * HEAD_V_DIM + tid];
                    float sig_g = e88_simple_sigmoid(g_val);
                    float silu_g = g_val * sig_g;
                    d_Sq_buf[tid] = d_out * silu_g;
                    float d_silu = sig_g * (1.0f + g_val * (1.0f - sig_g));
                    float Sq_before_gate = Sq_val / (silu_g + 1e-8f);
                    d_g_all[v_offset + tid] = __float2bfloat16(d_out * Sq_before_gate * d_silu);
                } else {
                    d_Sq_buf[tid] = d_out;
                }
            }
            __syncthreads();

            if (tid < N_STATE) {
                float sum = 0.0f;
                #pragma unroll 8
                for (int j = 0; j < HEAD_V_DIM; j++) {
                    sum += S_t[tid * HEAD_V_DIM + j] * d_Sq_buf[j];
                }
                d_q_buf[tid] = sum;
            }

            // Simple loop-based dS contribution
            for (int idx = tid; idx < state_size; idx += num_threads) {
                int i = idx / HEAD_V_DIM;
                int j = idx % HEAD_V_DIM;
                dS[idx] += q[i] * d_Sq_buf[j];
            }
            __syncthreads();

            if (tid < HEAD_V_DIM) {
                float d_delta_local = 0.0f;
                #pragma unroll 8
                for (int i = 0; i < N_STATE; i++) {
                    float d_pre = dS[i * HEAD_V_DIM + tid] * dtanh_buf[i * HEAD_V_DIM + tid];
                    d_delta_local += d_pre * k[i];
                }
                d_delta_buf[tid] = d_delta_local;
            }
            __syncthreads();  // FIX #3: sync between d_delta and d_k

            // Fused d_k: both dS*dtanh contribution and retrieved gradient in single loop
            if (tid < N_STATE) {
                float d_k_local = 0.0f;
                #pragma unroll 8
                for (int j = 0; j < HEAD_V_DIM; j++) {
                    float d_pre = dS[tid * HEAD_V_DIM + j] * dtanh_buf[tid * HEAD_V_DIM + j];
                    d_k_local += d_pre * delta_buf[j];
                    d_k_local += S[tid * HEAD_V_DIM + j] * (-d_delta_buf[j]);
                }
                d_k_buf[tid] = d_k_local;
            }
            __syncthreads();

            float d_decay_local = 0.0f;
            for (int idx = tid; idx < state_size; idx += num_threads) {
                float d_pre = dS[idx] * dtanh_buf[idx];
                d_decay_local += d_pre * S[idx];
            }

            #pragma unroll
            for (int offset = 16; offset >= 1; offset /= 2) {
                d_decay_local += __shfl_xor_sync(0xFFFFFFFF, d_decay_local, offset);
            }

            const int warp_id = tid / 32;
            const int lane_id = tid % 32;
            const int num_warps = (num_threads + 31) / 32;
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

            int decay_offset = (b * T + t) * H + h;
            if (tid < N_STATE) {
                d_k_all[k_offset + tid] = __float2bfloat16(d_k_buf[tid]);
                d_q_all[k_offset + tid] = __float2bfloat16(d_q_buf[tid]);
            }
            if (tid < HEAD_V_DIM) {
                d_v_all[v_offset + tid] = __float2bfloat16(d_delta_buf[tid]);
            }
            if (tid == 0) {
                d_decay_all[decay_offset] = __float2bfloat16(d_decay_accum);
            }

            // Simple loop-based dS update
            for (int idx = tid; idx < state_size; idx += num_threads) {
                int i = idx / HEAD_V_DIM;
                int j = idx % HEAD_V_DIM;
                float d_pre = dS[idx] * dtanh_buf[idx];
                dS[idx] = d_pre * decay_val + (-d_delta_buf[j]) * k[i];
            }
            __syncthreads();
        }
    }
}

void dispatch_e88_warp_backward_simple(
    int T, int B, int H, int n_state, int head_v_dim,
    const __nv_bfloat16* k, const __nv_bfloat16* v, const __nv_bfloat16* q,
    const __nv_bfloat16* decay, const __nv_bfloat16* g,
    const __nv_bfloat16* S_checkpoints, const __nv_bfloat16* Sq_cache,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* d_k, __nv_bfloat16* d_v, __nv_bfloat16* d_q,
    __nv_bfloat16* d_decay, __nv_bfloat16* d_g,
    __nv_bfloat16* segment_cache, int checkpoint_interval,
    bool has_gate, cudaStream_t stream
) {
    int num_blocks = B * H;
    // Adaptive thread count: reduce idle threads in guarded phases
    // Most phases use (tid < N_STATE), so scale threads with n_state, not state_size
    // n_state=16 → 64 threads (25% guarded utilization)
    // n_state=32 → 128 threads (25% guarded utilization)
    // n_state=64+ → 256 threads (capped)
    int threads_per_block = min(256, max(64, n_state * 4));

    constexpr int CHUNK_SIZE = E88_WARP_BACKWARD_SIMPLE_CHUNK_SIZE;

    // Shared memory: 4 state buffers + single-timestep k,v,q + work buffers + chunk prefetch buffers
    #define DISPATCH_SIMPLE(NS, HV) \
        do { \
            size_t shmem_size = (NS * HV) * 4 + \
                                NS + HV + NS + \
                                HV + HV + NS + NS + HV + 8 + \
                                CHUNK_SIZE * NS + \
                                CHUNK_SIZE * HV + \
                                CHUNK_SIZE * HV + \
                                CHUNK_SIZE * HV + \
                                CHUNK_SIZE; \
            shmem_size *= sizeof(float); \
            auto kernel = E88WarpBackwardSimpleKernel_BF16<NS, HV, CHUNK_SIZE>; \
            if (shmem_size > 48 * 1024) { \
                cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size); \
            } \
            kernel<<<num_blocks, threads_per_block, shmem_size, stream>>>( \
                T, B, H, k, v, q, decay, g, S_checkpoints, Sq_cache, d_output, \
                d_k, d_v, d_q, d_decay, d_g, segment_cache, checkpoint_interval, has_gate \
            ); \
        } while(0)

    if (n_state == 32 && head_v_dim == 32) {
        DISPATCH_SIMPLE(32, 32);
    } else if (n_state == 16 && head_v_dim == 16) {
        DISPATCH_SIMPLE(16, 16);
    } else if (n_state == 48 && head_v_dim == 48) {
        DISPATCH_SIMPLE(48, 48);
    } else if (n_state == 64 && head_v_dim == 64) {
        DISPATCH_SIMPLE(64, 64);
    } else {
        printf("E88 Warp Backward Simple: unsupported n_state=%d head_v_dim=%d\n", n_state, head_v_dim);
    }

    #undef DISPATCH_SIMPLE
}

}  // namespace elman
