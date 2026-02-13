/**
 * E88 Chunked Prefetch CUDA Kernel
 *
 * Key optimization: Load CHUNK_SIZE timesteps of k,v,q into shared memory
 * BEFORE processing, reducing global memory latency impact.
 *
 * Current kernel: 512 separate global memory reads (one per timestep)
 * Chunked kernel: 32 bulk reads of 16 timesteps each, processed from shared memory
 *
 * This is a stepping stone to full CUTLASS B2B fusion.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include "hasty/elman_ladder.h"

#define E88_CHUNK_SIZE 16  // Timesteps per chunk

namespace elman {

// =============================================================================
// Device Helper Functions
// =============================================================================

__device__ __forceinline__ float e88_chunked_tanh(float x) {
    return tanhf(x);
}

__device__ __forceinline__ float e88_chunked_silu(float x) {
    return x / (1.0f + expf(-x));
}

// =============================================================================
// E88 Chunked Forward Kernel
// Processes timesteps in chunks, prefetching each chunk into shared memory
// =============================================================================

template<int N_STATE, int HEAD_V_DIM, int CHUNK_SIZE>
__global__ void E88ChunkedForwardKernel_BF16(
    int T,
    int B,
    int H,
    const __nv_bfloat16* __restrict__ k_all,      // [B, T, H, N_STATE]
    const __nv_bfloat16* __restrict__ v_all,      // [B, T, H, HEAD_V_DIM]
    const __nv_bfloat16* __restrict__ q_all,      // [B, T, H, N_STATE]
    const __nv_bfloat16* __restrict__ decay_all,  // [B, T, H]
    const __nv_bfloat16* __restrict__ g_all,      // [B, T, H, HEAD_V_DIM] (can be nullptr)
    __nv_bfloat16* __restrict__ S,                // [B, H, N_STATE, HEAD_V_DIM]
    __nv_bfloat16* __restrict__ output,           // [B, T, H, HEAD_V_DIM]
    __nv_bfloat16* __restrict__ S_checkpoints,    // [num_checkpoints, B, H, N_STATE, HEAD_V_DIM]
    __nv_bfloat16* __restrict__ Sq_cache,         // [B, T, H, HEAD_V_DIM]
    int checkpoint_interval,
    bool apply_gate
) {
    // Block assignment: each block handles one (batch, head) pair
    int block_idx = blockIdx.x;
    int b = block_idx / H;
    int h = block_idx % H;
    if (b >= B) return;

    // Shared memory layout:
    // - S_shared: N_STATE * HEAD_V_DIM (state matrix, persistent)
    // - k_chunk: CHUNK_SIZE * N_STATE (prefetched keys)
    // - v_chunk: CHUNK_SIZE * HEAD_V_DIM (prefetched values)
    // - q_chunk: CHUNK_SIZE * N_STATE (prefetched queries)
    // - g_chunk: CHUNK_SIZE * HEAD_V_DIM (prefetched gates, if used)
    // - decay_chunk: CHUNK_SIZE (prefetched decay values)

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

    int tid = threadIdx.x;
    int state_offset = (b * H + h) * state_size;

    // Load initial state
    for (int i = tid; i < state_size; i += blockDim.x) {
        S_shared[i] = __bfloat162float(S[state_offset + i]);
    }
    __syncthreads();

    // Save initial checkpoint (index 0)
    for (int i = tid; i < state_size; i += blockDim.x) {
        S_checkpoints[(b * H + h) * state_size + i] = __float2bfloat16(S_shared[i]);
    }
    __syncthreads();

    int num_checkpoints = (T + checkpoint_interval - 1) / checkpoint_interval;
    int num_chunks = (T + CHUNK_SIZE - 1) / CHUNK_SIZE;

    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int t_start = chunk * CHUNK_SIZE;
        int t_end = min(t_start + CHUNK_SIZE, T);
        int chunk_len = t_end - t_start;

        // ========================================
        // PREFETCH: Load entire chunk into shared memory
        // ========================================

        // Load k for chunk
        for (int i = tid; i < chunk_len * N_STATE; i += blockDim.x) {
            int t_local = i / N_STATE;
            int k_idx = i % N_STATE;
            int t = t_start + t_local;
            int k_offset = ((b * T + t) * H + h) * N_STATE + k_idx;
            k_chunk[t_local * N_STATE + k_idx] = __bfloat162float(k_all[k_offset]);
        }

        // Load v for chunk
        for (int i = tid; i < chunk_len * HEAD_V_DIM; i += blockDim.x) {
            int t_local = i / HEAD_V_DIM;
            int v_idx = i % HEAD_V_DIM;
            int t = t_start + t_local;
            int v_offset = ((b * T + t) * H + h) * HEAD_V_DIM + v_idx;
            v_chunk[t_local * HEAD_V_DIM + v_idx] = __bfloat162float(v_all[v_offset]);
        }

        // Load q for chunk
        for (int i = tid; i < chunk_len * N_STATE; i += blockDim.x) {
            int t_local = i / N_STATE;
            int q_idx = i % N_STATE;
            int t = t_start + t_local;
            int q_offset = ((b * T + t) * H + h) * N_STATE + q_idx;
            q_chunk[t_local * N_STATE + q_idx] = __bfloat162float(q_all[q_offset]);
        }

        // Load decay for chunk
        for (int i = tid; i < chunk_len; i += blockDim.x) {
            int t = t_start + i;
            int decay_offset = (b * T + t) * H + h;
            decay_chunk[i] = __bfloat162float(decay_all[decay_offset]);
        }

        // Load gate for chunk (if applicable)
        if (apply_gate && g_all != nullptr) {
            for (int i = tid; i < chunk_len * HEAD_V_DIM; i += blockDim.x) {
                int t_local = i / HEAD_V_DIM;
                int g_idx = i % HEAD_V_DIM;
                int t = t_start + t_local;
                int g_offset = ((b * T + t) * H + h) * HEAD_V_DIM + g_idx;
                g_chunk[t_local * HEAD_V_DIM + g_idx] = __bfloat162float(g_all[g_offset]);
            }
        }

        __syncthreads();

        // ========================================
        // PROCESS: Run recurrence on chunk from shared memory
        // ========================================

        for (int t_local = 0; t_local < chunk_len; t_local++) {
            int t = t_start + t_local;

            float decay_val = decay_chunk[t_local];

            // Each thread handles one column of the state matrix
            if (tid < HEAD_V_DIM) {
                // Retrieval: sum_i S[i,j] * k[i]
                float ret_sum = 0.0f;
                #pragma unroll
                for (int i = 0; i < N_STATE; i++) {
                    ret_sum += S_shared[i * HEAD_V_DIM + tid] * k_chunk[t_local * N_STATE + i];
                }

                // Delta
                float delta_j = v_chunk[t_local * HEAD_V_DIM + tid] - ret_sum;

                // State update: S[i,j] = tanh(decay * S[i,j] + k[i] * delta[j])
                #pragma unroll
                for (int i = 0; i < N_STATE; i++) {
                    float s_old = S_shared[i * HEAD_V_DIM + tid];
                    float s_new = e88_chunked_tanh(
                        decay_val * s_old + k_chunk[t_local * N_STATE + i] * delta_j
                    );
                    S_shared[i * HEAD_V_DIM + tid] = s_new;
                }
            }
            __syncthreads();

            // Query: Sq[j] = sum_i S[i,j] * q[i]
            if (tid < HEAD_V_DIM) {
                float sq_sum = 0.0f;
                #pragma unroll
                for (int i = 0; i < N_STATE; i++) {
                    sq_sum += S_shared[i * HEAD_V_DIM + tid] * q_chunk[t_local * N_STATE + i];
                }

                // Apply gate if needed
                float out_val = sq_sum;
                if (apply_gate && g_all != nullptr) {
                    float gate_val = g_chunk[t_local * HEAD_V_DIM + tid];
                    out_val = sq_sum * e88_chunked_silu(gate_val);
                }

                // Write output
                int out_offset = ((b * T + t) * H + h) * HEAD_V_DIM + tid;
                output[out_offset] = __float2bfloat16(out_val);

                // Cache Sq for backward
                if (Sq_cache != nullptr) {
                    Sq_cache[(b * T + t) * H * HEAD_V_DIM + h * HEAD_V_DIM + tid] =
                        __float2bfloat16(sq_sum);
                }
            }
            __syncthreads();

            // Save checkpoint if needed
            if ((t + 1) % checkpoint_interval == 0 && t + 1 < T) {
                int cp_idx = (t + 1) / checkpoint_interval;
                int cp_offset = (cp_idx * B * H + b * H + h) * state_size;
                for (int i = tid; i < state_size; i += blockDim.x) {
                    S_checkpoints[cp_offset + i] = __float2bfloat16(S_shared[i]);
                }
            }
        }

        __syncthreads();  // Ensure all processing done before next chunk prefetch
    }

    // Write final state
    for (int i = tid; i < state_size; i += blockDim.x) {
        S[state_offset + i] = __float2bfloat16(S_shared[i]);
    }
}

// =============================================================================
// Dispatch function for chunked kernel
// =============================================================================

void dispatch_e88_chunked_forward(
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

    // Calculate shared memory size
    constexpr int CHUNK_SIZE = E88_CHUNK_SIZE;

    // Macro to dispatch based on n_state and head_v_dim
    #define DISPATCH_CHUNKED_KERNEL(NS, HV) \
        do { \
            size_t shmem_size = (NS * HV) +  /* state */ \
                                (CHUNK_SIZE * NS) +  /* k chunk */ \
                                (CHUNK_SIZE * HV) +  /* v chunk */ \
                                (CHUNK_SIZE * NS) +  /* q chunk */ \
                                (apply_gate ? CHUNK_SIZE * HV : 0) +  /* g chunk */ \
                                CHUNK_SIZE;  /* decay chunk */ \
            shmem_size *= sizeof(float); \
            E88ChunkedForwardKernel_BF16<NS, HV, CHUNK_SIZE><<<num_blocks, threads_per_block, shmem_size, stream>>>( \
                T, B, H, k, v, q, decay, g, S, output, S_checkpoints, Sq_cache, checkpoint_interval, apply_gate \
            ); \
        } while(0)

    if (n_state == 32 && head_v_dim == 32) {
        DISPATCH_CHUNKED_KERNEL(32, 32);
    } else if (n_state == 16 && head_v_dim == 16) {
        DISPATCH_CHUNKED_KERNEL(16, 16);
    } else if (n_state == 48 && head_v_dim == 48) {
        DISPATCH_CHUNKED_KERNEL(48, 48);
    } else if (n_state == 64 && head_v_dim == 64) {
        DISPATCH_CHUNKED_KERNEL(64, 64);
    } else {
        printf("E88 Chunked: unsupported n_state=%d head_v_dim=%d\n", n_state, head_v_dim);
    }

    #undef DISPATCH_CHUNKED_KERNEL
}

}  // namespace elman
