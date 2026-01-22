/**
 * E88 FLA Hybrid - Parallel Segment Processing
 *
 * Splits backward pass into two kernels:
 * 1. ForwardReplayAll: Parallel forward replay for all segments (stores all S states)
 * 2. BackwardOnly: Sequential backward using pre-computed states (no replay)
 *
 * This parallelizes the forward replay phase which was previously sequential.
 * Memory cost: T * B * H * n_state * head_v_dim (stores all S_{t-1} states)
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include "hasty/elman_ladder.h"

namespace elman {

// ============================================================================
// Kernel 1: Parallel Forward Replay
// Each block handles one (batch, head, segment) triple
// Stores S_{t-1} for all timesteps t
// ============================================================================

template<int N_STATE, int HEAD_V_DIM>
__global__ void E88ForwardReplayAllKernel_BF16(
    int T,
    int B,
    int H,
    int num_segments,
    const __nv_bfloat16* __restrict__ k_all,
    const __nv_bfloat16* __restrict__ v_all,
    const __nv_bfloat16* __restrict__ decay_all,
    const __nv_bfloat16* __restrict__ S_checkpoints,
    __nv_bfloat16* __restrict__ S_all,  // [T, B, H, N_STATE, HEAD_V_DIM] output
    int checkpoint_interval
) {
    // Block handles one (b, h, seg) triple
    int total_bh = B * H;
    int block_idx = blockIdx.x;
    int seg = block_idx / total_bh;
    int bh_idx = block_idx % total_bh;
    int b = bh_idx / H;
    int h = bh_idx % H;

    if (seg >= num_segments || b >= B) return;

    extern __shared__ float shared_mem[];
    float* S = shared_mem;
    float* k = S + N_STATE * HEAD_V_DIM;
    float* v = k + N_STATE;
    float* delta = v + HEAD_V_DIM;
    float* retrieved = delta + HEAD_V_DIM;

    int tid = threadIdx.x;
    int state_size = N_STATE * HEAD_V_DIM;

    __shared__ float decay_val;

    int t_start = seg * checkpoint_interval;
    int t_end = min(t_start + checkpoint_interval, T);

    // Load checkpoint for this segment
    int cp_offset = (seg * B * H + b * H + h) * state_size;
    for (int i = tid; i < state_size; i += blockDim.x) {
        S[i] = __bfloat162float(S_checkpoints[cp_offset + i]);
    }
    __syncthreads();

    // Replay forward through segment, storing S_{t-1} for each t
    for (int t = t_start; t < t_end; t++) {
        // Store S_{t-1} BEFORE the update
        int s_out_offset = ((t * B + b) * H + h) * state_size;
        for (int i = tid; i < state_size; i += blockDim.x) {
            S_all[s_out_offset + i] = __float2bfloat16(S[i]);
        }

        // Load k, v, decay for timestep t
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
            decay_val = __bfloat162float(decay_all[decay_offset]);
        }
        __syncthreads();

        // Compute retrieved = S @ k and delta = v - retrieved
        if (tid < HEAD_V_DIM) {
            float sum = 0.0f;
            #pragma unroll 8
            for (int i = 0; i < N_STATE; i++) {
                sum += S[i * HEAD_V_DIM + tid] * k[i];
            }
            retrieved[tid] = sum;
            delta[tid] = v[tid] - sum;
        }
        __syncthreads();

        // Update S: S_t = tanh(decay * S + outer(delta, k))
        for (int idx = tid; idx < state_size; idx += blockDim.x) {
            int i = idx / HEAD_V_DIM;
            int j = idx % HEAD_V_DIM;
            S[idx] = tanhf(decay_val * S[idx] + delta[j] * k[i]);
        }
        __syncthreads();
    }
}

// ============================================================================
// Kernel 2: Backward Only (No Forward Replay)
// Each block handles one (batch, head) pair
// Uses pre-computed S states from ForwardReplayAll
// ============================================================================

template<int N_STATE, int HEAD_V_DIM>
__global__ void E88BackwardOnlyKernel_BF16(
    int T,
    int B,
    int H,
    const __nv_bfloat16* __restrict__ k_all,
    const __nv_bfloat16* __restrict__ v_all,
    const __nv_bfloat16* __restrict__ q_all,
    const __nv_bfloat16* __restrict__ decay_all,
    const __nv_bfloat16* __restrict__ S_all,  // [T, B, H, N_STATE, HEAD_V_DIM] pre-computed
    const __nv_bfloat16* __restrict__ d_output,
    __nv_bfloat16* __restrict__ d_k_all,
    __nv_bfloat16* __restrict__ d_v_all,
    __nv_bfloat16* __restrict__ d_q_all,
    __nv_bfloat16* __restrict__ d_decay_all
) {
    int block_idx = blockIdx.x;
    int b = block_idx / H;
    int h = block_idx % H;
    if (b >= B) return;

    extern __shared__ float shared_mem[];

    float* S = shared_mem;
    float* dS = S + N_STATE * HEAD_V_DIM;
    float* S_t = dS + N_STATE * HEAD_V_DIM;
    float* dtanh = S_t + N_STATE * HEAD_V_DIM;
    float* k = dtanh + N_STATE * HEAD_V_DIM;
    float* v = k + N_STATE;
    float* q = v + HEAD_V_DIM;
    float* delta = q + N_STATE;
    float* retrieved = delta + HEAD_V_DIM;
    float* d_k = retrieved + HEAD_V_DIM;
    float* d_v = d_k + N_STATE;
    float* d_q = d_v + HEAD_V_DIM;
    float* d_Sq = d_q + N_STATE;
    float* d_delta = d_Sq + HEAD_V_DIM;
    float* warp_results = d_delta + HEAD_V_DIM;

    int tid = threadIdx.x;
    int state_size = N_STATE * HEAD_V_DIM;

    __shared__ float decay_val;

    // Initialize dS to zero
    for (int i = tid; i < state_size; i += blockDim.x) {
        dS[i] = 0.0f;
    }
    __syncthreads();

    // Process all timesteps backward (no segment structure needed)
    for (int t = T - 1; t >= 0; t--) {
        int k_offset = ((t * B + b) * H + h) * N_STATE;
        int v_offset = ((t * B + b) * H + h) * HEAD_V_DIM;
        int decay_offset = (t * B + b) * H + h;
        int s_offset = ((t * B + b) * H + h) * state_size;

        // Load pre-computed S_{t-1}, k, v, q, d_output in one sync
        for (int i = tid; i < state_size; i += blockDim.x) {
            S[i] = __bfloat162float(S_all[s_offset + i]);
        }
        if (tid < N_STATE) {
            k[tid] = __bfloat162float(k_all[k_offset + tid]);
            q[tid] = __bfloat162float(q_all[k_offset + tid]);
        }
        if (tid < HEAD_V_DIM) {
            v[tid] = __bfloat162float(v_all[v_offset + tid]);
            d_Sq[tid] = __bfloat162float(d_output[v_offset + tid]);
        }
        if (tid == 0) {
            decay_val = __bfloat162float(decay_all[decay_offset]);
        }
        __syncthreads();

        // Compute retrieved and delta
        if (tid < HEAD_V_DIM) {
            float sum = 0.0f;
            #pragma unroll 8
            for (int i = 0; i < N_STATE; i++) {
                sum += S[i * HEAD_V_DIM + tid] * k[i];
            }
            retrieved[tid] = sum;
            delta[tid] = v[tid] - sum;
        }
        __syncthreads();

        // Compute S_t and dtanh
        for (int idx = tid; idx < state_size; idx += blockDim.x) {
            int i = idx / HEAD_V_DIM;
            int j = idx % HEAD_V_DIM;
            float pre_tanh = decay_val * S[idx] + delta[j] * k[i];
            float tanh_val = tanhf(pre_tanh);
            S_t[idx] = tanh_val;
            dtanh[idx] = 1.0f - tanh_val * tanh_val;
        }
        __syncthreads();

        // Compute d_q and update dS
        if (tid < N_STATE) {
            float sum = 0.0f;
            for (int j = 0; j < HEAD_V_DIM; j++) {
                sum += S_t[tid * HEAD_V_DIM + j] * d_Sq[j];
            }
            d_q[tid] = sum;
        }
        for (int idx = tid; idx < state_size; idx += blockDim.x) {
            int i = idx / HEAD_V_DIM;
            int j = idx % HEAD_V_DIM;
            dS[idx] += q[i] * d_Sq[j];
        }
        __syncthreads();

        // Compute d_delta and d_k
        if (tid < HEAD_V_DIM) {
            float d_delta_local = 0.0f;
            for (int i = 0; i < N_STATE; i++) {
                float d_pre = dS[i * HEAD_V_DIM + tid] * dtanh[i * HEAD_V_DIM + tid];
                d_delta_local += d_pre * k[i];
            }
            d_delta[tid] = d_delta_local;
        }
        if (tid < N_STATE) {
            float d_k_local = 0.0f;
            for (int j = 0; j < HEAD_V_DIM; j++) {
                float d_pre = dS[tid * HEAD_V_DIM + j] * dtanh[tid * HEAD_V_DIM + j];
                d_k_local += d_pre * delta[j];
            }
            d_k[tid] = d_k_local;
        }
        __syncthreads();

        // Compute d_decay with warp reduction
        float d_decay_local = 0.0f;
        for (int idx = tid; idx < state_size; idx += blockDim.x) {
            float d_pre = dS[idx] * dtanh[idx];
            d_decay_local += d_pre * S[idx];
        }

        #pragma unroll
        for (int offset = 16; offset >= 1; offset /= 2) {
            d_decay_local += __shfl_xor_sync(0xFFFFFFFF, d_decay_local, offset);
        }

        const int warp_id = tid / 32;
        const int lane_id = tid % 32;
        const int num_warps = (blockDim.x + 31) / 32;
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

        // d_k contribution from retrieved gradient
        if (tid < N_STATE) {
            float d_k_from_retrieved = 0.0f;
            for (int j = 0; j < HEAD_V_DIM; j++) {
                d_k_from_retrieved += S[tid * HEAD_V_DIM + j] * (-d_delta[j]);
            }
            d_k[tid] += d_k_from_retrieved;
        }
        __syncthreads();

        // Write gradients
        if (tid < N_STATE) {
            d_k_all[k_offset + tid] = __float2bfloat16(d_k[tid]);
            d_q_all[k_offset + tid] = __float2bfloat16(d_q[tid]);
        }
        if (tid < HEAD_V_DIM) {
            d_v_all[v_offset + tid] = __float2bfloat16(d_delta[tid]);
        }
        if (tid == 0) {
            d_decay_all[decay_offset] = __float2bfloat16(d_decay_accum);
        }

        // Update dS for next iteration
        for (int idx = tid; idx < state_size; idx += blockDim.x) {
            int i = idx / HEAD_V_DIM;
            int j = idx % HEAD_V_DIM;
            float d_pre = dS[idx] * dtanh[idx];
            dS[idx] = d_pre * decay_val + (-d_delta[j]) * k[i];
        }
        __syncthreads();
    }
}

// ============================================================================
// Dispatch functions
// ============================================================================

void dispatch_e88_forward_replay_all(
    int T, int B, int H, int n_state, int head_v_dim,
    const __nv_bfloat16* k_all, const __nv_bfloat16* v_all,
    const __nv_bfloat16* decay_all, const __nv_bfloat16* S_checkpoints,
    __nv_bfloat16* S_all,
    int checkpoint_interval, cudaStream_t stream
) {
    int num_segments = (T + checkpoint_interval - 1) / checkpoint_interval;
    int num_blocks = num_segments * B * H;
    int threads = 256;

    #define DISPATCH_FWD_REPLAY(N, V) do { \
        int state_size = N * V; \
        int shared_size = (state_size + N + 3 * V) * sizeof(float); \
        E88ForwardReplayAllKernel_BF16<N, V><<<num_blocks, threads, shared_size, stream>>>( \
            T, B, H, num_segments, k_all, v_all, decay_all, S_checkpoints, \
            S_all, checkpoint_interval); \
    } while(0)

    if (n_state == 32 && head_v_dim == 64) { DISPATCH_FWD_REPLAY(32, 64); }
    else if (n_state == 32 && head_v_dim == 128) { DISPATCH_FWD_REPLAY(32, 128); }
    else if (n_state == 64 && head_v_dim == 64) { DISPATCH_FWD_REPLAY(64, 64); }
    else if (n_state == 64 && head_v_dim == 128) { DISPATCH_FWD_REPLAY(64, 128); }
    else if (n_state == 48 && head_v_dim == 96) { DISPATCH_FWD_REPLAY(48, 96); }
    else {
        fprintf(stderr, "E88 Forward Replay All: unsupported n_state=%d, head_v_dim=%d\n",
                n_state, head_v_dim);
    }

    #undef DISPATCH_FWD_REPLAY
}

void dispatch_e88_backward_only(
    int T, int B, int H, int n_state, int head_v_dim,
    const __nv_bfloat16* k_all, const __nv_bfloat16* v_all,
    const __nv_bfloat16* q_all, const __nv_bfloat16* decay_all,
    const __nv_bfloat16* S_all, const __nv_bfloat16* d_output,
    __nv_bfloat16* d_k_all, __nv_bfloat16* d_v_all,
    __nv_bfloat16* d_q_all, __nv_bfloat16* d_decay_all,
    cudaStream_t stream
) {
    int num_blocks = B * H;
    int threads = 256;

    #define DISPATCH_BWD_ONLY(N, V) do { \
        int state_size = N * V; \
        int shared_size = (4 * state_size + 4 * N + 6 * V + 8) * sizeof(float); \
        auto kernel = E88BackwardOnlyKernel_BF16<N, V>; \
        if (shared_size > 48 * 1024) { \
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size); \
        } \
        kernel<<<num_blocks, threads, shared_size, stream>>>( \
            T, B, H, k_all, v_all, q_all, decay_all, S_all, d_output, \
            d_k_all, d_v_all, d_q_all, d_decay_all); \
    } while(0)

    if (n_state == 32 && head_v_dim == 64) { DISPATCH_BWD_ONLY(32, 64); }
    else if (n_state == 32 && head_v_dim == 128) { DISPATCH_BWD_ONLY(32, 128); }
    else if (n_state == 64 && head_v_dim == 64) { DISPATCH_BWD_ONLY(64, 64); }
    else if (n_state == 64 && head_v_dim == 128) { DISPATCH_BWD_ONLY(64, 128); }
    else if (n_state == 48 && head_v_dim == 96) { DISPATCH_BWD_ONLY(48, 96); }
    else {
        fprintf(stderr, "E88 Backward Only: unsupported n_state=%d, head_v_dim=%d\n",
                n_state, head_v_dim);
    }

    #undef DISPATCH_BWD_ONLY
}

// Template instantiations
template __global__ void E88ForwardReplayAllKernel_BF16<32, 64>(int, int, int, int, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int);
template __global__ void E88ForwardReplayAllKernel_BF16<32, 128>(int, int, int, int, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int);
template __global__ void E88ForwardReplayAllKernel_BF16<64, 64>(int, int, int, int, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int);
template __global__ void E88ForwardReplayAllKernel_BF16<64, 128>(int, int, int, int, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int);
template __global__ void E88ForwardReplayAllKernel_BF16<48, 96>(int, int, int, int, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int);

template __global__ void E88BackwardOnlyKernel_BF16<32, 64>(int, int, int, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*);
template __global__ void E88BackwardOnlyKernel_BF16<32, 128>(int, int, int, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*);
template __global__ void E88BackwardOnlyKernel_BF16<64, 64>(int, int, int, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*);
template __global__ void E88BackwardOnlyKernel_BF16<64, 128>(int, int, int, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*);
template __global__ void E88BackwardOnlyKernel_BF16<48, 96>(int, int, int, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*);

}  // namespace elman
