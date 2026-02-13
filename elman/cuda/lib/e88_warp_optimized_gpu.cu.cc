/**
 * E88 Warp-Optimized CUDA Kernel
 *
 * Key optimizations over chunked kernel:
 * 1. Full thread utilization: All 128 threads active during compute
 * 2. Warp shuffle reductions: No shared memory atomics for dot products
 * 3. Parallel state update: Multiple threads per state element
 * 4. Register tiling: Keep frequently accessed data in registers
 *
 * Thread layout for n_state=32, head_v_dim=32:
 * - 128 threads total
 * - 4 threads per column for reduction (128 / 32 = 4)
 * - Each thread handles 8 state elements for update (1024 / 128 = 8)
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include "hasty/elman_ladder.h"

#define E88_WARP_CHUNK_SIZE 16  // Timesteps per chunk

namespace elman {

// =============================================================================
// Warp-level primitives
// =============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float e88_warp_tanh(float x) {
    return tanhf(x);
}

__device__ __forceinline__ float e88_warp_silu(float x) {
    return x / (1.0f + expf(-x));
}

// =============================================================================
// E88 Warp-Optimized Forward Kernel
// =============================================================================

/**
 * Simplified warp-optimized kernel:
 * - Use same column-per-thread as chunked (no parallel reduction)
 * - But use all threads for parallel state update
 * - Only HEAD_V_DIM threads (32) do the recurrence, rest help with memory ops
 */
template<int N_STATE, int HEAD_V_DIM, int CHUNK_SIZE>
__global__ void E88WarpOptimizedForwardKernel_BF16(
    int T,
    int B,
    int H,
    const __nv_bfloat16* __restrict__ k_all,
    const __nv_bfloat16* __restrict__ v_all,
    const __nv_bfloat16* __restrict__ q_all,
    const __nv_bfloat16* __restrict__ decay_all,
    const __nv_bfloat16* __restrict__ g_all,
    __nv_bfloat16* __restrict__ S,
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ S_checkpoints,
    __nv_bfloat16* __restrict__ Sq_cache,
    int checkpoint_interval,
    bool apply_gate
) {
    int block_idx = blockIdx.x;
    int b = block_idx / H;
    int h = block_idx % H;
    if (b >= B) return;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    // Shared memory
    extern __shared__ float shared_mem[];

    constexpr int state_size = N_STATE * HEAD_V_DIM;
    constexpr int k_chunk_size = CHUNK_SIZE * N_STATE;
    constexpr int v_chunk_size = CHUNK_SIZE * HEAD_V_DIM;
    constexpr int q_chunk_size = CHUNK_SIZE * N_STATE;
    constexpr int g_chunk_size = CHUNK_SIZE * HEAD_V_DIM;

    float* S_shared = shared_mem;
    float* k_chunk = S_shared + state_size;
    float* v_chunk = k_chunk + k_chunk_size;
    float* q_chunk = v_chunk + v_chunk_size;
    float* g_chunk = q_chunk + q_chunk_size;
    float* decay_chunk = g_chunk + (apply_gate ? g_chunk_size : 0);
    // delta buffer for broadcasting - one value per column
    float* delta_buf = decay_chunk + CHUNK_SIZE;

    int state_offset = (b * H + h) * state_size;

    // Load initial state
    for (int i = tid; i < state_size; i += num_threads) {
        S_shared[i] = __bfloat162float(S[state_offset + i]);
    }
    __syncthreads();

    // Save initial checkpoint
    for (int i = tid; i < state_size; i += num_threads) {
        S_checkpoints[(b * H + h) * state_size + i] = __float2bfloat16(S_shared[i]);
    }
    __syncthreads();

    int num_chunks = (T + CHUNK_SIZE - 1) / CHUNK_SIZE;

    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int t_start = chunk * CHUNK_SIZE;
        int t_end = min(t_start + CHUNK_SIZE, T);
        int chunk_len = t_end - t_start;

        // PREFETCH
        for (int i = tid; i < chunk_len * N_STATE; i += num_threads) {
            int t_local = i / N_STATE;
            int k_idx = i % N_STATE;
            int t = t_start + t_local;
            k_chunk[t_local * N_STATE + k_idx] = __bfloat162float(k_all[((b * T + t) * H + h) * N_STATE + k_idx]);
        }
        for (int i = tid; i < chunk_len * HEAD_V_DIM; i += num_threads) {
            int t_local = i / HEAD_V_DIM;
            int v_idx = i % HEAD_V_DIM;
            int t = t_start + t_local;
            v_chunk[t_local * HEAD_V_DIM + v_idx] = __bfloat162float(v_all[((b * T + t) * H + h) * HEAD_V_DIM + v_idx]);
        }
        for (int i = tid; i < chunk_len * N_STATE; i += num_threads) {
            int t_local = i / N_STATE;
            int q_idx = i % N_STATE;
            int t = t_start + t_local;
            q_chunk[t_local * N_STATE + q_idx] = __bfloat162float(q_all[((b * T + t) * H + h) * N_STATE + q_idx]);
        }
        for (int i = tid; i < chunk_len; i += num_threads) {
            decay_chunk[i] = __bfloat162float(decay_all[(b * T + t_start + i) * H + h]);
        }
        if (apply_gate && g_all != nullptr) {
            for (int i = tid; i < chunk_len * HEAD_V_DIM; i += num_threads) {
                int t_local = i / HEAD_V_DIM;
                int g_idx = i % HEAD_V_DIM;
                int t = t_start + t_local;
                g_chunk[t_local * HEAD_V_DIM + g_idx] = __bfloat162float(g_all[((b * T + t) * H + h) * HEAD_V_DIM + g_idx]);
            }
        }
        __syncthreads();

        // PROCESS
        for (int t_local = 0; t_local < chunk_len; t_local++) {
            int t = t_start + t_local;
            float decay_val = decay_chunk[t_local];

            // Step 1 & 2: Only first HEAD_V_DIM threads compute retrieval, delta
            // Each thread handles one column completely
            if (tid < HEAD_V_DIM) {
                int j = tid;

                // Retrieval
                float ret_j = 0.0f;
                #pragma unroll
                for (int i = 0; i < N_STATE; i++) {
                    ret_j += S_shared[i * HEAD_V_DIM + j] * k_chunk[t_local * N_STATE + i];
                }

                // Delta
                float delta_j = v_chunk[t_local * HEAD_V_DIM + j] - ret_j;
                delta_buf[j] = delta_j;
            }
            __syncthreads();

            // Step 3: ALL threads participate in state update
            for (int idx = tid; idx < state_size; idx += num_threads) {
                int i = idx / HEAD_V_DIM;
                int j = idx % HEAD_V_DIM;
                float s_old = S_shared[idx];
                float k_i = k_chunk[t_local * N_STATE + i];
                float d_j = delta_buf[j];
                float s_new = e88_warp_tanh(decay_val * s_old + k_i * d_j);
                S_shared[idx] = s_new;
            }
            __syncthreads();

            // Step 4 & 5: First HEAD_V_DIM threads compute query and write output
            if (tid < HEAD_V_DIM) {
                int j = tid;

                float sq_j = 0.0f;
                #pragma unroll
                for (int i = 0; i < N_STATE; i++) {
                    sq_j += S_shared[i * HEAD_V_DIM + j] * q_chunk[t_local * N_STATE + i];
                }

                float out_val = sq_j;
                if (apply_gate && g_all != nullptr) {
                    out_val = sq_j * e88_warp_silu(g_chunk[t_local * HEAD_V_DIM + j]);
                }

                output[((b * T + t) * H + h) * HEAD_V_DIM + j] = __float2bfloat16(out_val);

                if (Sq_cache != nullptr) {
                    Sq_cache[(b * T + t) * H * HEAD_V_DIM + h * HEAD_V_DIM + j] = __float2bfloat16(sq_j);
                }
            }
            __syncthreads();

            // Checkpoint
            if ((t + 1) % checkpoint_interval == 0 && t + 1 < T) {
                int cp_idx = (t + 1) / checkpoint_interval;
                int cp_offset = (cp_idx * B * H + b * H + h) * state_size;
                for (int i = tid; i < state_size; i += num_threads) {
                    S_checkpoints[cp_offset + i] = __float2bfloat16(S_shared[i]);
                }
            }
        }
        __syncthreads();
    }

    // Write final state
    for (int i = tid; i < state_size; i += num_threads) {
        S[state_offset + i] = __float2bfloat16(S_shared[i]);
    }
}

// =============================================================================
// Dispatch function for warp-optimized kernel
// =============================================================================

void dispatch_e88_warp_optimized_forward(
    int T, int B, int H, int n_state, int head_v_dim,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    const __nv_bfloat16* q,
    const __nv_bfloat16* decay,
    const __nv_bfloat16* g,
    __nv_bfloat16* S,
    __nv_bfloat16* output,
    __nv_bfloat16* S_checkpoints,
    __nv_bfloat16* Sq_cache,
    int checkpoint_interval,
    bool apply_gate,
    cudaStream_t stream
) {
    int num_blocks = B * H;
    int threads_per_block = 128;

    constexpr int CHUNK_SIZE = E88_WARP_CHUNK_SIZE;

    #define DISPATCH_WARP_KERNEL(NS, HV) \
        do { \
            size_t shmem_size = (NS * HV) +  /* state */ \
                                (CHUNK_SIZE * NS) +  /* k chunk */ \
                                (CHUNK_SIZE * HV) +  /* v chunk */ \
                                (CHUNK_SIZE * NS) +  /* q chunk */ \
                                (apply_gate ? CHUNK_SIZE * HV : 0) +  /* g chunk */ \
                                CHUNK_SIZE +  /* decay chunk */ \
                                threads_per_block;  /* reduction buffer */ \
            shmem_size *= sizeof(float); \
            E88WarpOptimizedForwardKernel_BF16<NS, HV, CHUNK_SIZE><<<num_blocks, threads_per_block, shmem_size, stream>>>( \
                T, B, H, k, v, q, decay, g, S, output, S_checkpoints, Sq_cache, checkpoint_interval, apply_gate \
            ); \
        } while(0)

    if (n_state == 32 && head_v_dim == 32) {
        DISPATCH_WARP_KERNEL(32, 32);
    } else if (n_state == 16 && head_v_dim == 16) {
        DISPATCH_WARP_KERNEL(16, 16);
    } else if (n_state == 48 && head_v_dim == 48) {
        DISPATCH_WARP_KERNEL(48, 48);
    } else if (n_state == 64 && head_v_dim == 64) {
        DISPATCH_WARP_KERNEL(64, 64);
    } else {
        printf("E88 Warp Optimized: unsupported n_state=%d head_v_dim=%d\n", n_state, head_v_dim);
    }

    #undef DISPATCH_WARP_KERNEL
}

}  // namespace elman
