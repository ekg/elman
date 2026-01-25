/**
 * E88 FLA Hybrid Backward Kernel with Reduced Synchronization
 *
 * Based on E88FLAHybridBackwardKernel_Cached_BF16 but with fewer __syncthreads().
 *
 * Key optimizations:
 * 1. Merge adjacent loads (S, k, v, q) into single sync block
 * 2. Combine d_delta and d_k computation to share one sync
 * 3. Remove redundant sync after gradient writes (loop sync handles it)
 *
 * Original cached kernel: ~6 syncs/timestep (forward) + ~15 syncs/timestep (backward) = 21 total
 * Reduced sync kernel:    ~4 syncs/timestep (forward) + ~11 syncs/timestep (backward) = 15 total
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include "hasty/elman_ladder.h"

// Use hardware tanhf - Pade approximation is actually slower on A100
#define USE_FAST_TANH 0

namespace elman {

#if USE_FAST_TANH
__device__ __forceinline__ float e88_tanh(float x) {
    x = fmaxf(-5.0f, fminf(5.0f, x));
    float x2 = x * x;
    float num = x * (135135.0f + x2 * (17325.0f + x2 * 378.0f));
    float den = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
    return num / den;
}
#else
__device__ __forceinline__ float e88_tanh(float x) {
    return tanhf(x);
}
#endif

template<int N_STATE, int HEAD_V_DIM>
__global__ void E88FLAHybridBackwardKernel_ReducedSync_BF16(
    int T,
    int B,
    int H,
    const __nv_bfloat16* __restrict__ k_all,
    const __nv_bfloat16* __restrict__ v_all,
    const __nv_bfloat16* __restrict__ q_all,
    const __nv_bfloat16* __restrict__ decay_all,
    const __nv_bfloat16* __restrict__ S_checkpoints,
    const __nv_bfloat16* __restrict__ Sq_cache,
    const __nv_bfloat16* __restrict__ d_output,
    __nv_bfloat16* __restrict__ d_k_all,
    __nv_bfloat16* __restrict__ d_v_all,
    __nv_bfloat16* __restrict__ d_q_all,
    __nv_bfloat16* __restrict__ d_decay_all,
    __nv_bfloat16* __restrict__ segment_cache,
    int checkpoint_interval
) {
    int block_idx = blockIdx.x;
    int b = block_idx / H;
    int h = block_idx % H;
    if (b >= B) return;

    extern __shared__ float shared_mem[];

    // Shared memory layout (same as cached kernel)
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
    constexpr int state_size = N_STATE * HEAD_V_DIM;

    // OPTIMIZATION: Precompute row/col indices for this thread's work items
    // Avoids integer division in inner loops (expensive on GPU)
    int my_start = tid;
    int my_stride = blockDim.x;

    // Precompute indices for first 4 elements
    int row0 = my_start / HEAD_V_DIM;
    int col0 = my_start % HEAD_V_DIM;
    int row1 = (my_start + my_stride) / HEAD_V_DIM;
    int col1 = (my_start + my_stride) % HEAD_V_DIM;
    int row2 = (my_start + 2*my_stride) / HEAD_V_DIM;
    int col2 = (my_start + 2*my_stride) % HEAD_V_DIM;
    int row3 = (my_start + 3*my_stride) / HEAD_V_DIM;
    int col3 = (my_start + 3*my_stride) % HEAD_V_DIM;

    // Segment cache offsets
    int cache_entry_size = state_size + N_STATE + HEAD_V_DIM + 1;
    __nv_bfloat16* seg_cache_base = segment_cache + (size_t)block_idx * checkpoint_interval * cache_entry_size;
    __nv_bfloat16* S_cache_base = seg_cache_base;
    __nv_bfloat16* k_cache_base = seg_cache_base + (size_t)checkpoint_interval * state_size;
    __nv_bfloat16* v_cache_base = k_cache_base + (size_t)checkpoint_interval * N_STATE;
    __nv_bfloat16* decay_cache_base = v_cache_base + (size_t)checkpoint_interval * HEAD_V_DIM;

    // Shared variable for decay (used by all threads)
    __shared__ float decay_val;

    // Initialize dS to zero
    for (int i = tid; i < state_size; i += blockDim.x) {
        dS[i] = 0.0f;
    }
    __syncthreads();  // SYNC 1

    int num_segments = (T + checkpoint_interval - 1) / checkpoint_interval;

    for (int seg = num_segments - 1; seg >= 0; seg--) {
        int t_start = seg * checkpoint_interval;
        int t_end = min(t_start + checkpoint_interval, T);
        int seg_len = t_end - t_start;

        // ================================================================
        // PHASE 1: Forward replay (REDUCED SYNCS)
        // ================================================================

        // Load checkpoint
        int cp_offset = (seg * B * H + b * H + h) * state_size;
        for (int i = tid; i < state_size; i += blockDim.x) {
            S[i] = __bfloat162float(S_checkpoints[cp_offset + i]);
        }
        __syncthreads();  // SYNC 2

        for (int local_t = 0; local_t < seg_len; local_t++) {
            int t = t_start + local_t;
            int k_offset = ((t * B + b) * H + h) * N_STATE;
            int v_offset = ((t * B + b) * H + h) * HEAD_V_DIM;
            int decay_offset = (t * B + b) * H + h;

            __nv_bfloat16* S_cache_slot = S_cache_base + (size_t)local_t * state_size;
            __nv_bfloat16* k_cache_slot = k_cache_base + (size_t)local_t * N_STATE;
            __nv_bfloat16* v_cache_slot = v_cache_base + (size_t)local_t * HEAD_V_DIM;

            // MERGED: Save S to cache + load k,v,decay + cache k,v,decay (was 4 syncs, now 1)
            for (int i = tid; i < state_size; i += blockDim.x) {
                S_cache_slot[i] = __float2bfloat16(S[i]);
            }
            if (tid < N_STATE) {
                float k_val = __bfloat162float(k_all[k_offset + tid]);
                k[tid] = k_val;
                k_cache_slot[tid] = __float2bfloat16(k_val);
            }
            if (tid < HEAD_V_DIM) {
                float v_val = __bfloat162float(v_all[v_offset + tid]);
                v[tid] = v_val;
                v_cache_slot[tid] = __float2bfloat16(v_val);
            }
            if (tid == 0) {
                float d = __bfloat162float(decay_all[decay_offset]);
                decay_val = d;
                decay_cache_base[local_t] = __float2bfloat16(d);
            }
            __syncthreads();  // SYNC 3

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
            __syncthreads();  // SYNC 4

            // Update S: S_t = tanh(decay * S + outer(delta, k))
            // OPTIMIZATION: Use precomputed indices
            if (my_start < state_size) {
                S[my_start] = e88_tanh(decay_val * S[my_start] + delta[col0] * k[row0]);
            }
            if (my_start + my_stride < state_size) {
                S[my_start + my_stride] = e88_tanh(decay_val * S[my_start + my_stride] + delta[col1] * k[row1]);
            }
            if (my_start + 2*my_stride < state_size) {
                S[my_start + 2*my_stride] = e88_tanh(decay_val * S[my_start + 2*my_stride] + delta[col2] * k[row2]);
            }
            if (my_start + 3*my_stride < state_size) {
                S[my_start + 3*my_stride] = e88_tanh(decay_val * S[my_start + 3*my_stride] + delta[col3] * k[row3]);
            }
            for (int idx = my_start + 4*my_stride; idx < state_size; idx += my_stride) {
                int i = idx / HEAD_V_DIM;
                int j = idx % HEAD_V_DIM;
                S[idx] = e88_tanh(decay_val * S[idx] + delta[j] * k[i]);
            }
            __syncthreads();  // SYNC 5 (needed for next iteration)
        }
        // Forward replay: 4 syncs per timestep (was 6)

        // ================================================================
        // PHASE 2: Backward pass (REDUCED SYNCS)
        // ================================================================

        for (int local_t = seg_len - 1; local_t >= 0; local_t--) {
            int t = t_start + local_t;
            int k_offset = ((t * B + b) * H + h) * N_STATE;
            int v_offset = ((t * B + b) * H + h) * HEAD_V_DIM;
            int decay_offset = (t * B + b) * H + h;

            __nv_bfloat16* S_cache_slot = S_cache_base + (size_t)local_t * state_size;
            __nv_bfloat16* k_cache_slot = k_cache_base + (size_t)local_t * N_STATE;
            __nv_bfloat16* v_cache_slot = v_cache_base + (size_t)local_t * HEAD_V_DIM;

            // MERGED: Load S, k, v from cache + q from global + decay from cache (was 3 syncs, now 1)
            for (int i = tid; i < state_size; i += blockDim.x) {
                S[i] = __bfloat162float(S_cache_slot[i]);
            }
            if (tid < N_STATE) {
                k[tid] = __bfloat162float(k_cache_slot[tid]);
                q[tid] = __bfloat162float(q_all[k_offset + tid]);
            }
            if (tid < HEAD_V_DIM) {
                v[tid] = __bfloat162float(v_cache_slot[tid]);
                d_Sq[tid] = __bfloat162float(d_output[v_offset + tid]);
            }
            if (tid == 0) {
                decay_val = __bfloat162float(decay_cache_base[local_t]);
            }
            __syncthreads();  // SYNC 6

            // Compute retrieved, delta, S_t, dtanh
            if (tid < HEAD_V_DIM) {
                float sum = 0.0f;
                #pragma unroll 8
                for (int i = 0; i < N_STATE; i++) {
                    sum += S[i * HEAD_V_DIM + tid] * k[i];
                }
                retrieved[tid] = sum;
                delta[tid] = v[tid] - sum;
            }
            __syncthreads();  // SYNC 7

            // Compute S_t and dtanh - OPTIMIZATION: Use precomputed indices
            #define COMPUTE_ST_DTANH(IDX, ROW, COL) do { \
                float pre_tanh = decay_val * S[IDX] + delta[COL] * k[ROW]; \
                float tanh_val = e88_tanh(pre_tanh); \
                S_t[IDX] = tanh_val; \
                dtanh[IDX] = 1.0f - tanh_val * tanh_val; \
            } while(0)

            if (my_start < state_size) COMPUTE_ST_DTANH(my_start, row0, col0);
            if (my_start + my_stride < state_size) COMPUTE_ST_DTANH(my_start + my_stride, row1, col1);
            if (my_start + 2*my_stride < state_size) COMPUTE_ST_DTANH(my_start + 2*my_stride, row2, col2);
            if (my_start + 3*my_stride < state_size) COMPUTE_ST_DTANH(my_start + 3*my_stride, row3, col3);
            for (int idx = my_start + 4*my_stride; idx < state_size; idx += my_stride) {
                int i = idx / HEAD_V_DIM;
                int j = idx % HEAD_V_DIM;
                float pre_tanh = decay_val * S[idx] + delta[j] * k[i];
                float tanh_val = e88_tanh(pre_tanh);
                S_t[idx] = tanh_val;
                dtanh[idx] = 1.0f - tanh_val * tanh_val;
            }
            #undef COMPUTE_ST_DTANH
            __syncthreads();  // SYNC 8

            // Compute d_q and update dS with output gradient
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int j = 0; j < HEAD_V_DIM; j++) {
                    sum += S_t[tid * HEAD_V_DIM + j] * d_Sq[j];
                }
                d_q[tid] = sum;
            }
            // OPTIMIZATION: Use precomputed indices
            if (my_start < state_size) dS[my_start] += q[row0] * d_Sq[col0];
            if (my_start + my_stride < state_size) dS[my_start + my_stride] += q[row1] * d_Sq[col1];
            if (my_start + 2*my_stride < state_size) dS[my_start + 2*my_stride] += q[row2] * d_Sq[col2];
            if (my_start + 3*my_stride < state_size) dS[my_start + 3*my_stride] += q[row3] * d_Sq[col3];
            for (int idx = my_start + 4*my_stride; idx < state_size; idx += my_stride) {
                int i = idx / HEAD_V_DIM;
                int j = idx % HEAD_V_DIM;
                dS[idx] += q[i] * d_Sq[j];
            }
            __syncthreads();  // SYNC 9

            // MERGED: Compute d_delta AND d_k together (was 2 syncs, now 1)
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
            __syncthreads();  // SYNC 10

            // Compute d_decay with warp reduction
            float d_decay_local = 0.0f;
            for (int idx = tid; idx < state_size; idx += blockDim.x) {
                float d_pre = dS[idx] * dtanh[idx];
                d_decay_local += d_pre * S[idx];
            }

            // Warp reduction
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
            __syncthreads();  // SYNC 11

            // Final reduction and d_k correction
            float d_decay_accum = 0.0f;
            if (warp_id == 0) {
                float load_val = (tid < num_warps) ? warp_results[tid] : 0.0f;
                #pragma unroll
                for (int offset = 16; offset >= 1; offset /= 2) {
                    load_val += __shfl_xor_sync(0xFFFFFFFF, load_val, offset);
                }
                d_decay_accum = load_val;
            }

            // d_k contribution from retrieved gradient (runs in parallel with reduction)
            if (tid < N_STATE) {
                float d_k_from_retrieved = 0.0f;
                for (int j = 0; j < HEAD_V_DIM; j++) {
                    d_k_from_retrieved += S[tid * HEAD_V_DIM + j] * (-d_delta[j]);
                }
                d_k[tid] += d_k_from_retrieved;
            }
            __syncthreads();  // SYNC 12

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

            // Update dS for next iteration - OPTIMIZATION: Use precomputed indices
            #define UPDATE_DS(IDX, ROW, COL) do { \
                float d_pre = dS[IDX] * dtanh[IDX]; \
                dS[IDX] = d_pre * decay_val + (-d_delta[COL]) * k[ROW]; \
            } while(0)

            if (my_start < state_size) UPDATE_DS(my_start, row0, col0);
            if (my_start + my_stride < state_size) UPDATE_DS(my_start + my_stride, row1, col1);
            if (my_start + 2*my_stride < state_size) UPDATE_DS(my_start + 2*my_stride, row2, col2);
            if (my_start + 3*my_stride < state_size) UPDATE_DS(my_start + 3*my_stride, row3, col3);
            for (int idx = my_start + 4*my_stride; idx < state_size; idx += my_stride) {
                int i = idx / HEAD_V_DIM;
                int j = idx % HEAD_V_DIM;
                float d_pre = dS[idx] * dtanh[idx];
                dS[idx] = d_pre * decay_val + (-d_delta[j]) * k[i];
            }
            #undef UPDATE_DS
            __syncthreads();  // SYNC 13
        }
        // Backward: 8 syncs per timestep (was 15)
    }
}

// Dispatch function for reduced sync kernel
void dispatch_e88_reduced_sync_backward(
    int T, int B, int H, int n_state, int head_v_dim,
    const __nv_bfloat16* k_all, const __nv_bfloat16* v_all,
    const __nv_bfloat16* q_all, const __nv_bfloat16* decay_all,
    const __nv_bfloat16* S_checkpoints, const __nv_bfloat16* Sq_cache,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* d_k_all, __nv_bfloat16* d_v_all,
    __nv_bfloat16* d_q_all, __nv_bfloat16* d_decay_all,
    __nv_bfloat16* segment_cache,
    int checkpoint_interval, cudaStream_t stream
) {
    int num_blocks = B * H;
    int threads = 256;

    // Shared memory: same as cached kernel
    // 4 * state_size (S, dS, S_t, dtanh) + 4 * N_STATE (k, q, d_k, d_q) + 6 * HEAD_V_DIM (v, delta, retrieved, d_v, d_Sq, d_delta) + 8 (warp_results)
    #define DISPATCH_REDUCED_SYNC(N, V) do { \
        int state_size = N * V; \
        int shared_size = (4 * state_size + 4 * N + 6 * V + 8) * sizeof(float); \
        auto kernel = E88FLAHybridBackwardKernel_ReducedSync_BF16<N, V>; \
        if (shared_size > 48 * 1024) { \
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size); \
        } \
        kernel<<<num_blocks, threads, shared_size, stream>>>( \
            T, B, H, k_all, v_all, q_all, decay_all, \
            S_checkpoints, Sq_cache, d_output, \
            d_k_all, d_v_all, d_q_all, d_decay_all, \
            segment_cache, checkpoint_interval); \
    } while(0)

    // Symmetric configurations (n_state == head_v_dim)
    if (n_state == 4 && head_v_dim == 4) { DISPATCH_REDUCED_SYNC(4, 4); }
    else if (n_state == 8 && head_v_dim == 8) { DISPATCH_REDUCED_SYNC(8, 8); }
    else if (n_state == 16 && head_v_dim == 16) { DISPATCH_REDUCED_SYNC(16, 16); }
    else if (n_state == 32 && head_v_dim == 32) { DISPATCH_REDUCED_SYNC(32, 32); }
    else if (n_state == 48 && head_v_dim == 48) { DISPATCH_REDUCED_SYNC(48, 48); }
    else if (n_state == 64 && head_v_dim == 64) { DISPATCH_REDUCED_SYNC(64, 64); }
    else if (n_state == 96 && head_v_dim == 96) { DISPATCH_REDUCED_SYNC(96, 96); }
    else if (n_state == 128 && head_v_dim == 128) { DISPATCH_REDUCED_SYNC(128, 128); }
    // Asymmetric configurations
    else if (n_state == 32 && head_v_dim == 64) { DISPATCH_REDUCED_SYNC(32, 64); }
    else if (n_state == 32 && head_v_dim == 128) { DISPATCH_REDUCED_SYNC(32, 128); }
    else if (n_state == 64 && head_v_dim == 128) { DISPATCH_REDUCED_SYNC(64, 128); }
    else if (n_state == 48 && head_v_dim == 96) { DISPATCH_REDUCED_SYNC(48, 96); }
    else {
        fprintf(stderr, "E88 Reduced Sync Backward: unsupported n_state=%d, head_v_dim=%d\n",
                n_state, head_v_dim);
    }

    #undef DISPATCH_REDUCED_SYNC
}

// Template instantiations - Symmetric configurations
template __global__ void E88FLAHybridBackwardKernel_ReducedSync_BF16<4, 4>(
    int, int, int, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int);

template __global__ void E88FLAHybridBackwardKernel_ReducedSync_BF16<8, 8>(
    int, int, int, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int);

template __global__ void E88FLAHybridBackwardKernel_ReducedSync_BF16<16, 16>(
    int, int, int, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int);

template __global__ void E88FLAHybridBackwardKernel_ReducedSync_BF16<32, 32>(
    int, int, int, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int);

template __global__ void E88FLAHybridBackwardKernel_ReducedSync_BF16<48, 48>(
    int, int, int, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int);

template __global__ void E88FLAHybridBackwardKernel_ReducedSync_BF16<64, 64>(
    int, int, int, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int);

template __global__ void E88FLAHybridBackwardKernel_ReducedSync_BF16<96, 96>(
    int, int, int, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int);

template __global__ void E88FLAHybridBackwardKernel_ReducedSync_BF16<128, 128>(
    int, int, int, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int);

// Template instantiations - Asymmetric configurations
template __global__ void E88FLAHybridBackwardKernel_ReducedSync_BF16<32, 64>(
    int, int, int, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int);

template __global__ void E88FLAHybridBackwardKernel_ReducedSync_BF16<32, 128>(
    int, int, int, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int);

template __global__ void E88FLAHybridBackwardKernel_ReducedSync_BF16<64, 128>(
    int, int, int, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int);

template __global__ void E88FLAHybridBackwardKernel_ReducedSync_BF16<48, 96>(
    int, int, int, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int);

}  // namespace elman
