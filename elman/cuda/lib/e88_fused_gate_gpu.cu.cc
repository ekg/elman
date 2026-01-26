/**
 * E88 FLA Hybrid Forward Kernel with Fused Output Gating
 *
 * This kernel fuses the output gating operation into the E88 forward pass:
 * - Original: output = Sq (write to memory) -> read Sq -> compute silu(g) -> output * silu(g) (write)
 * - Fused:    output = Sq * silu(g) (single write)
 *
 * Saves 1 memory read + 1 memory write per element per timestep.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include "hasty/elman_ladder.h"

#define E88_CHECKPOINT_INTERVAL 16

namespace elman {

__device__ __forceinline__ float e88_tanh(float x) {
    return tanhf(x);
}

// SiLU (swish) activation: x * sigmoid(x)
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

/**
 * E88 Forward Kernel with Fused Output Gating
 *
 * Template parameters:
 *   N_STATE: dimension of k, q (n_state)
 *   HEAD_V_DIM: dimension of v, output (head_v_dim)
 *
 * Inputs:
 *   k_all: [T, B, H, N_STATE] - key projections (L2 normalized)
 *   v_all: [T, B, H, HEAD_V_DIM] - value projections
 *   q_all: [T, B, H, N_STATE] - query projections (L2 normalized)
 *   decay_all: [T, B, H] - decay factors
 *   g_all: [T, B, H, HEAD_V_DIM] - gate projections (for fused gating)
 *   S: [B, H, N_STATE, HEAD_V_DIM] - initial state (also output final state)
 *
 * Outputs:
 *   output: [T, B, H, HEAD_V_DIM] - final gated output (Sq * silu(g))
 *   S_checkpoints: checkpoints for backward
 *   Sq_cache: Sq values for backward
 */
template<int N_STATE, int HEAD_V_DIM>
__global__ void E88FLAHybridForwardKernel_FusedGate_BF16(
    int T,
    int B,
    int H,
    const __nv_bfloat16* __restrict__ k_all,
    const __nv_bfloat16* __restrict__ v_all,
    const __nv_bfloat16* __restrict__ q_all,
    const __nv_bfloat16* __restrict__ decay_all,
    const __nv_bfloat16* __restrict__ g_all,  // NEW: gate projections
    __nv_bfloat16* __restrict__ S,
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ S_checkpoints,
    __nv_bfloat16* __restrict__ Sq_cache,
    int checkpoint_interval
) {
    int block_idx = blockIdx.x;
    int b = block_idx / H;
    int h = block_idx % H;
    if (b >= B) return;

    extern __shared__ float shared_mem[];

    // Shared memory layout
    constexpr int state_size = N_STATE * HEAD_V_DIM;
    float* S_shared = shared_mem;
    float* k_shared = S_shared + state_size;
    float* v_shared = k_shared + N_STATE;
    float* q_shared = v_shared + HEAD_V_DIM;
    float* retrieved = q_shared + N_STATE;
    float* g_shared = retrieved + HEAD_V_DIM;  // NEW: gate values

    int tid = threadIdx.x;
    int state_offset = (b * H + h) * state_size;

    // Precompute row/col indices to avoid division
    int my_start = tid;
    int my_stride = blockDim.x;
    int row0 = my_start / HEAD_V_DIM;
    int col0 = my_start % HEAD_V_DIM;
    int row1 = (my_start + my_stride) / HEAD_V_DIM;
    int col1 = (my_start + my_stride) % HEAD_V_DIM;
    int row2 = (my_start + 2*my_stride) / HEAD_V_DIM;
    int col2 = (my_start + 2*my_stride) % HEAD_V_DIM;
    int row3 = (my_start + 3*my_stride) / HEAD_V_DIM;
    int col3 = (my_start + 3*my_stride) % HEAD_V_DIM;

    __shared__ float decay_val;

    // Load initial state
    for (int i = tid; i < state_size; i += blockDim.x) {
        S_shared[i] = __bfloat162float(S[state_offset + i]);
    }
    __syncthreads();

    // Save initial checkpoint (only if training)
    if (S_checkpoints != nullptr) {
        for (int i = tid; i < state_size; i += blockDim.x) {
            S_checkpoints[(b * H + h) * state_size + i] = __float2bfloat16(S_shared[i]);
        }
    }
    __syncthreads();

    for (int t = 0; t < T; t++) {
        // Load k, v, q, decay, g for this timestep
        int k_offset = ((t * B + b) * H + h) * N_STATE;
        int v_offset = ((t * B + b) * H + h) * HEAD_V_DIM;
        int decay_offset = (t * B + b) * H + h;

        if (tid < N_STATE) {
            k_shared[tid] = __bfloat162float(k_all[k_offset + tid]);
            q_shared[tid] = __bfloat162float(q_all[k_offset + tid]);
        }
        if (tid < HEAD_V_DIM) {
            v_shared[tid] = __bfloat162float(v_all[v_offset + tid]);
            g_shared[tid] = __bfloat162float(g_all[v_offset + tid]);  // NEW: load gate
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
        if (my_start < state_size) {
            float delta0 = v_shared[col0] - retrieved[col0];
            S_shared[my_start] = e88_tanh(decay_val * S_shared[my_start] + delta0 * k_shared[row0]);
        }
        if (my_start + my_stride < state_size) {
            float delta1 = v_shared[col1] - retrieved[col1];
            S_shared[my_start + my_stride] = e88_tanh(decay_val * S_shared[my_start + my_stride] + delta1 * k_shared[row1]);
        }
        if (my_start + 2*my_stride < state_size) {
            float delta2 = v_shared[col2] - retrieved[col2];
            S_shared[my_start + 2*my_stride] = e88_tanh(decay_val * S_shared[my_start + 2*my_stride] + delta2 * k_shared[row2]);
        }
        if (my_start + 3*my_stride < state_size) {
            float delta3 = v_shared[col3] - retrieved[col3];
            S_shared[my_start + 3*my_stride] = e88_tanh(decay_val * S_shared[my_start + 3*my_stride] + delta3 * k_shared[row3]);
        }
        for (int idx = my_start + 4*my_stride; idx < state_size; idx += my_stride) {
            int i = idx / HEAD_V_DIM;
            int j = idx % HEAD_V_DIM;
            float delta_j = v_shared[j] - retrieved[j];
            S_shared[idx] = e88_tanh(decay_val * S_shared[idx] + delta_j * k_shared[i]);
        }
        __syncthreads();

        // Save checkpoint (only if training)
        if (S_checkpoints != nullptr && (t + 1) % checkpoint_interval == 0) {
            int cp_idx = (t + 1) / checkpoint_interval;
            int cp_offset = (cp_idx * B * H + b * H + h) * state_size;
            for (int i = tid; i < state_size; i += blockDim.x) {
                S_checkpoints[cp_offset + i] = __float2bfloat16(S_shared[i]);
            }
        }

        // Compute Sq and FUSED GATING: output = Sq * silu(g)
        if (tid < HEAD_V_DIM) {
            float Sq = 0.0f;
            #pragma unroll 8
            for (int i = 0; i < N_STATE; i++) {
                Sq += S_shared[i * HEAD_V_DIM + tid] * q_shared[i];
            }
            // Cache un-gated Sq for backward (only if training)
            if (Sq_cache != nullptr) {
                Sq_cache[v_offset + tid] = __float2bfloat16(Sq);
            }

            // FUSED GATING: apply silu(g) directly
            float g_val = g_shared[tid];
            float gated_output = Sq * silu(g_val);
            output[v_offset + tid] = __float2bfloat16(gated_output);
        }
        __syncthreads();
    }

    // Write final state
    for (int i = tid; i < state_size; i += blockDim.x) {
        S[state_offset + i] = __float2bfloat16(S_shared[i]);
    }
}

// ============================================================================
// Dispatch function
// ============================================================================

void dispatch_e88_fused_gate_forward(
    int T, int B, int H, int n_state, int head_v_dim,
    const __nv_bfloat16* k_all, const __nv_bfloat16* v_all,
    const __nv_bfloat16* q_all, const __nv_bfloat16* decay_all,
    const __nv_bfloat16* g_all,
    __nv_bfloat16* S, __nv_bfloat16* output,
    __nv_bfloat16* S_checkpoints, __nv_bfloat16* Sq_cache,
    int checkpoint_interval, cudaStream_t stream
) {
    int state_size = n_state * head_v_dim;
    // Shared: S + k + v + q + retrieved + g
    int shared_size = (state_size + n_state + head_v_dim + n_state + head_v_dim + head_v_dim) * sizeof(float);
    int threads = min(256, state_size);
    int num_blocks = B * H;

    #define DISPATCH_E88_FG(N, V) do { \
        auto kernel = E88FLAHybridForwardKernel_FusedGate_BF16<N, V>; \
        if (shared_size > 48 * 1024) { \
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size); \
        } \
        kernel<<<num_blocks, threads, shared_size, stream>>>( \
            T, B, H, k_all, v_all, q_all, decay_all, g_all, S, output, \
            S_checkpoints, Sq_cache, checkpoint_interval); \
    } while(0)

    // Common configurations
    if (n_state == 32 && head_v_dim == 32) { DISPATCH_E88_FG(32, 32); }
    else if (n_state == 16 && head_v_dim == 16) { DISPATCH_E88_FG(16, 16); }
    else if (n_state == 64 && head_v_dim == 64) { DISPATCH_E88_FG(64, 64); }
    else if (n_state == 48 && head_v_dim == 48) { DISPATCH_E88_FG(48, 48); }
    else {
        fprintf(stderr, "E88 Fused Gate Forward: unsupported n_state=%d, head_v_dim=%d\n", n_state, head_v_dim);
    }

    #undef DISPATCH_E88_FG
}

} // namespace elman

// Template instantiations
template __global__ void elman::E88FLAHybridForwardKernel_FusedGate_BF16<32, 32>(
    int, int, int, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int);

template __global__ void elman::E88FLAHybridForwardKernel_FusedGate_BF16<16, 16>(
    int, int, int, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int);

template __global__ void elman::E88FLAHybridForwardKernel_FusedGate_BF16<64, 64>(
    int, int, int, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int);

template __global__ void elman::E88FLAHybridForwardKernel_FusedGate_BF16<48, 48>(
    int, int, int, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int);
