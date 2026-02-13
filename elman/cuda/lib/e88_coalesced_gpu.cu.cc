/**
 * E88 Coalesced Memory Access CUDA Kernel
 *
 * Key optimizations:
 * 1. Transpose state matrix for coalesced access: [HEAD_V_DIM, N_STATE] instead of [N_STATE, HEAD_V_DIM]
 * 2. Each thread handles one column (like chunked kernel) but with better memory layout
 * 3. Use 32 threads (one per column) for optimal warp utilization
 * 4. Reduce shared memory bank conflicts
 *
 * This is simpler than warp-parallel reduction but focuses on memory efficiency.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include "hasty/elman_ladder.h"

#define E88_COAL_CHUNK_SIZE 16

namespace elman {

__device__ __forceinline__ float e88_coal_tanh(float x) {
    return tanhf(x);
}

__device__ __forceinline__ float e88_coal_silu(float x) {
    return x / (1.0f + expf(-x));
}

// =============================================================================
// E88 Coalesced Forward Kernel
// State transposed: S[j][i] for coalesced column access
// =============================================================================

template<int N_STATE, int HEAD_V_DIM, int CHUNK_SIZE>
__global__ void E88CoalescedForwardKernel_BF16(
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
    int block_idx = blockIdx.x;
    int b = block_idx / H;
    int h = block_idx % H;
    if (b >= B) return;

    // Use exactly HEAD_V_DIM threads (32 for standard config)
    // Each thread owns one column j and iterates over all rows i
    const int j = threadIdx.x;
    if (j >= HEAD_V_DIM) return;

    // Shared memory: state + prefetch buffers
    extern __shared__ float shared_mem[];

    constexpr int state_size = N_STATE * HEAD_V_DIM;
    constexpr int k_chunk_size = CHUNK_SIZE * N_STATE;
    constexpr int v_chunk_size = CHUNK_SIZE * HEAD_V_DIM;
    constexpr int q_chunk_size = CHUNK_SIZE * N_STATE;
    constexpr int g_chunk_size = CHUNK_SIZE * HEAD_V_DIM;

    // State stored TRANSPOSED: S_T[j][i] = S[i][j] for coalesced column reads
    // Layout: S_T[j * N_STATE + i] = S[i, j]
    float* S_T = shared_mem;
    float* k_chunk = S_T + state_size;
    float* v_chunk = k_chunk + k_chunk_size;
    float* q_chunk = v_chunk + v_chunk_size;
    float* g_chunk = q_chunk + q_chunk_size;
    float* decay_chunk = g_chunk + (apply_gate ? g_chunk_size : 0);

    int state_offset = (b * H + h) * state_size;

    // Load initial state (transposed)
    // Original: S[state_offset + i * HEAD_V_DIM + j] = S[i, j]
    // Transposed: S_T[j * N_STATE + i] = S[i, j]
    for (int i = 0; i < N_STATE; i++) {
        S_T[j * N_STATE + i] = __bfloat162float(S[state_offset + i * HEAD_V_DIM + j]);
    }
    __syncthreads();

    // Save initial checkpoint (in original layout)
    for (int i = 0; i < N_STATE; i++) {
        S_checkpoints[(b * H + h) * state_size + i * HEAD_V_DIM + j] =
            __float2bfloat16(S_T[j * N_STATE + i]);
    }
    __syncthreads();

    int num_chunks = (T + CHUNK_SIZE - 1) / CHUNK_SIZE;

    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int t_start = chunk * CHUNK_SIZE;
        int t_end = min(t_start + CHUNK_SIZE, T);
        int chunk_len = t_end - t_start;

        // ========================================
        // PREFETCH: Load chunk data
        // ========================================

        // Each thread loads multiple elements to maximize memory bandwidth
        // k_chunk layout: [t_local, i] - each thread loads elements for different t
        for (int t_local = 0; t_local < chunk_len; t_local++) {
            int t = t_start + t_local;
            // Load k[t, i] for all i - but each thread only loads its slice
            for (int i = j; i < N_STATE; i += HEAD_V_DIM) {
                int k_offset = ((b * T + t) * H + h) * N_STATE + i;
                k_chunk[t_local * N_STATE + i] = __bfloat162float(k_all[k_offset]);
            }
            // Load v[t, j]
            int v_offset = ((b * T + t) * H + h) * HEAD_V_DIM + j;
            v_chunk[t_local * HEAD_V_DIM + j] = __bfloat162float(v_all[v_offset]);

            // Load q[t, i] for all i
            for (int i = j; i < N_STATE; i += HEAD_V_DIM) {
                int q_offset = ((b * T + t) * H + h) * N_STATE + i;
                q_chunk[t_local * N_STATE + i] = __bfloat162float(q_all[q_offset]);
            }
        }

        // Load decay
        for (int t_local = j; t_local < chunk_len; t_local += HEAD_V_DIM) {
            int t = t_start + t_local;
            int decay_offset = (b * T + t) * H + h;
            decay_chunk[t_local] = __bfloat162float(decay_all[decay_offset]);
        }

        // Load gate
        if (apply_gate && g_all != nullptr) {
            for (int t_local = 0; t_local < chunk_len; t_local++) {
                int t = t_start + t_local;
                int g_offset = ((b * T + t) * H + h) * HEAD_V_DIM + j;
                g_chunk[t_local * HEAD_V_DIM + j] = __bfloat162float(g_all[g_offset]);
            }
        }

        __syncthreads();

        // ========================================
        // PROCESS: Run recurrence
        // ========================================

        for (int t_local = 0; t_local < chunk_len; t_local++) {
            int t = t_start + t_local;
            float decay_val = decay_chunk[t_local];

            // Thread j handles column j of state matrix

            // Retrieval: ret[j] = sum_i S[i,j] * k[i]
            // With transposed layout: S_T[j * N_STATE + i] = S[i, j]
            float ret_j = 0.0f;
            #pragma unroll
            for (int i = 0; i < N_STATE; i++) {
                ret_j += S_T[j * N_STATE + i] * k_chunk[t_local * N_STATE + i];
            }

            // Delta
            float delta_j = v_chunk[t_local * HEAD_V_DIM + j] - ret_j;

            // State update: S[i,j] = tanh(decay * S[i,j] + k[i] * delta[j])
            #pragma unroll
            for (int i = 0; i < N_STATE; i++) {
                float s_old = S_T[j * N_STATE + i];
                float s_new = e88_coal_tanh(
                    decay_val * s_old + k_chunk[t_local * N_STATE + i] * delta_j
                );
                S_T[j * N_STATE + i] = s_new;
            }
            // No sync needed - each thread only touches its own column

            // Query: Sq[j] = sum_i S[i,j] * q[i]
            float sq_j = 0.0f;
            #pragma unroll
            for (int i = 0; i < N_STATE; i++) {
                sq_j += S_T[j * N_STATE + i] * q_chunk[t_local * N_STATE + i];
            }

            // Apply gate
            float out_val = sq_j;
            if (apply_gate && g_all != nullptr) {
                float gate_val = g_chunk[t_local * HEAD_V_DIM + j];
                out_val = sq_j * e88_coal_silu(gate_val);
            }

            // Write output
            int out_offset = ((b * T + t) * H + h) * HEAD_V_DIM + j;
            output[out_offset] = __float2bfloat16(out_val);

            // Cache Sq
            if (Sq_cache != nullptr) {
                Sq_cache[(b * T + t) * H * HEAD_V_DIM + h * HEAD_V_DIM + j] =
                    __float2bfloat16(sq_j);
            }

            // Save checkpoint
            if ((t + 1) % checkpoint_interval == 0 && t + 1 < T) {
                int cp_idx = (t + 1) / checkpoint_interval;
                // Write in original layout
                for (int i = 0; i < N_STATE; i++) {
                    int cp_offset = (cp_idx * B * H + b * H + h) * state_size + i * HEAD_V_DIM + j;
                    S_checkpoints[cp_offset] = __float2bfloat16(S_T[j * N_STATE + i]);
                }
            }
        }

        __syncthreads();
    }

    // Write final state (convert back from transposed)
    for (int i = 0; i < N_STATE; i++) {
        S[state_offset + i * HEAD_V_DIM + j] = __float2bfloat16(S_T[j * N_STATE + i]);
    }
}

// =============================================================================
// Dispatch function
// =============================================================================

void dispatch_e88_coalesced_forward(
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

    constexpr int CHUNK_SIZE = E88_COAL_CHUNK_SIZE;

    #define DISPATCH_COAL_KERNEL(NS, HV) \
        do { \
            int threads = HV; \
            size_t shmem_size = (NS * HV) +  /* state transposed */ \
                                (CHUNK_SIZE * NS) +  /* k chunk */ \
                                (CHUNK_SIZE * HV) +  /* v chunk */ \
                                (CHUNK_SIZE * NS) +  /* q chunk */ \
                                (apply_gate ? CHUNK_SIZE * HV : 0) +  /* g chunk */ \
                                CHUNK_SIZE;  /* decay chunk */ \
            shmem_size *= sizeof(float); \
            E88CoalescedForwardKernel_BF16<NS, HV, CHUNK_SIZE><<<num_blocks, threads, shmem_size, stream>>>( \
                T, B, H, k, v, q, decay, g, S, output, S_checkpoints, Sq_cache, checkpoint_interval, apply_gate \
            ); \
        } while(0)

    if (n_state == 32 && head_v_dim == 32) {
        DISPATCH_COAL_KERNEL(32, 32);
    } else if (n_state == 16 && head_v_dim == 16) {
        DISPATCH_COAL_KERNEL(16, 16);
    } else if (n_state == 48 && head_v_dim == 48) {
        DISPATCH_COAL_KERNEL(48, 48);
    } else if (n_state == 64 && head_v_dim == 64) {
        DISPATCH_COAL_KERNEL(64, 64);
    } else {
        printf("E88 Coalesced: unsupported n_state=%d head_v_dim=%d\n", n_state, head_v_dim);
    }

    #undef DISPATCH_COAL_KERNEL
}

}  // namespace elman
