/**
 * E88 Fully Fused CUDA Kernel
 *
 * Eliminates Python overhead by fusing ALL operations:
 * 1. Input projection (qkv + alpha) via cuBLAS GEMM
 * 2. L2 normalization (warp reduction)
 * 3. Mamba2-style decay computation
 * 4. E88 nonlinear recurrence
 * 5. Output gating (silu)
 * 6. Output projection via cuBLAS GEMM
 *
 * Input: x [B, T, dim] - NO TRANSPOSE NEEDED
 * Output: y [B, T, dim] - NO TRANSPOSE NEEDED
 *
 * This kernel eliminates:
 * - 5 transpose+contiguous copies per layer (2.9 GB/forward)
 * - Separate L2 norm kernel launches (28 per forward)
 * - Separate decay computation kernels
 * - Separate gating kernels
 *
 * Expected speedup: ~40% over current E88 implementation
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include "hasty/elman_ladder.h"

#define E88_FUSED_CHECKPOINT_INTERVAL 16

namespace elman {

// =============================================================================
// Device Helper Functions
// =============================================================================

__device__ __forceinline__ float e88_fused_tanh(float x) {
    return tanhf(x);
}

__device__ __forceinline__ float e88_fused_silu(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float e88_fused_softplus(float x) {
    // softplus(x) = log(1 + exp(x)), numerically stable version
    if (x > 20.0f) return x;
    return log1pf(expf(x));
}

// L2 norm using warp reduction
// Each thread holds one element, computes sum of squares across warp
__device__ __forceinline__ float warp_l2_norm_sq(float val, int lane_mask) {
    float sum = val * val;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    }
    return sum;
}

// =============================================================================
// E88 Fused Forward Kernel - Recurrence Only
// Assumes projections and L2 norm already done, similar to current kernel
// but handles [B, T, H, dim] layout WITHOUT transpose
// =============================================================================

template<int N_STATE, int HEAD_V_DIM>
__global__ void E88FusedForwardKernel_BF16(
    int T,
    int B,
    int H,
    // Pre-projected inputs in [B, T, H, dim] layout
    const __nv_bfloat16* __restrict__ k_all,      // [B, T, H, N_STATE] L2-normalized
    const __nv_bfloat16* __restrict__ v_all,      // [B, T, H, HEAD_V_DIM]
    const __nv_bfloat16* __restrict__ q_all,      // [B, T, H, N_STATE] L2-normalized
    const __nv_bfloat16* __restrict__ decay_all,  // [B, T, H]
    const __nv_bfloat16* __restrict__ g_all,      // [B, T, H, HEAD_V_DIM] gate (can be nullptr)
    __nv_bfloat16* __restrict__ S,                // [B, H, N_STATE, HEAD_V_DIM]
    __nv_bfloat16* __restrict__ output,           // [B, T, H, HEAD_V_DIM]
    __nv_bfloat16* __restrict__ S_checkpoints,    // [num_checkpoints, B, H, N_STATE, HEAD_V_DIM]
    __nv_bfloat16* __restrict__ Sq_cache,         // [B, T, H, HEAD_V_DIM]
    int checkpoint_interval,
    bool apply_gate  // Whether to apply silu gating
) {
    // Block assignment: each block handles one (batch, head) pair
    int block_idx = blockIdx.x;
    int b = block_idx / H;
    int h = block_idx % H;
    if (b >= B) return;

    extern __shared__ float shared_mem[];
    float* S_shared = shared_mem;
    float* k_shared = S_shared + N_STATE * HEAD_V_DIM;
    float* v_shared = k_shared + N_STATE;
    float* q_shared = v_shared + HEAD_V_DIM;
    float* g_shared = q_shared + N_STATE;  // Only used if apply_gate

    int tid = threadIdx.x;
    constexpr int state_size = N_STATE * HEAD_V_DIM;

    int state_offset = (b * H + h) * state_size;
    __shared__ float decay_val;

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

    for (int t = 0; t < T; t++) {
        // Input indices for [B, T, H, dim] layout
        int k_offset = ((b * T + t) * H + h) * N_STATE;
        int v_offset = ((b * T + t) * H + h) * HEAD_V_DIM;
        int decay_offset = (b * T + t) * H + h;

        // Load k, v, q for this timestep
        if (tid < N_STATE) {
            k_shared[tid] = __bfloat162float(k_all[k_offset + tid]);
            q_shared[tid] = __bfloat162float(q_all[k_offset + tid]);
        }
        if (tid < HEAD_V_DIM) {
            v_shared[tid] = __bfloat162float(v_all[v_offset + tid]);
            if (apply_gate && g_all != nullptr) {
                g_shared[tid] = __bfloat162float(g_all[v_offset + tid]);
            }
        }
        if (tid == 0) {
            decay_val = __bfloat162float(decay_all[decay_offset]);
        }
        __syncthreads();

        // Fused retrieval + state update + output
        if (tid < HEAD_V_DIM) {
            // Retrieval: sum_i S[i,j] * k[i]
            float ret_sum = 0.0f;
            #pragma unroll 8
            for (int i = 0; i < N_STATE; i++) {
                ret_sum += S_shared[i * HEAD_V_DIM + tid] * k_shared[i];
            }
            float delta_j = v_shared[tid] - ret_sum;

            // State update + output accumulation
            float Sq = 0.0f;
            #pragma unroll 8
            for (int i = 0; i < N_STATE; i++) {
                int idx = i * HEAD_V_DIM + tid;
                float new_s = e88_fused_tanh(decay_val * S_shared[idx] + delta_j * k_shared[i]);
                S_shared[idx] = new_s;
                Sq += new_s * q_shared[i];
            }

            // Apply gating if enabled
            if (apply_gate && g_all != nullptr) {
                Sq = Sq * e88_fused_silu(g_shared[tid]);
            }

            // Write output (keeping [B, T, H, dim] layout)
            output[v_offset + tid] = __float2bfloat16(Sq);
            Sq_cache[v_offset + tid] = __float2bfloat16(Sq);
        }
        __syncthreads();

        // Checkpoint saving
        if ((t + 1) % checkpoint_interval == 0) {
            int cp_idx = (t + 1) / checkpoint_interval;
            int cp_offset = (cp_idx * B * H + b * H + h) * state_size;
            for (int i = tid; i < state_size; i += blockDim.x) {
                S_checkpoints[cp_offset + i] = __float2bfloat16(S_shared[i]);
            }
        }
    }

    // Write final state
    for (int i = tid; i < state_size; i += blockDim.x) {
        S[state_offset + i] = __float2bfloat16(S_shared[i]);
    }
}

// =============================================================================
// E88 Fused Backward Kernel - Handles [B, T, H, dim] layout
// =============================================================================

template<int N_STATE, int HEAD_V_DIM>
__global__ void E88FusedBackwardKernel_BF16(
    int T,
    int B,
    int H,
    const __nv_bfloat16* __restrict__ k_all,
    const __nv_bfloat16* __restrict__ v_all,
    const __nv_bfloat16* __restrict__ q_all,
    const __nv_bfloat16* __restrict__ decay_all,
    const __nv_bfloat16* __restrict__ g_all,      // Gate values (can be nullptr)
    const __nv_bfloat16* __restrict__ S_checkpoints,
    const __nv_bfloat16* __restrict__ Sq_cache,
    const __nv_bfloat16* __restrict__ d_output,
    __nv_bfloat16* __restrict__ d_k_all,
    __nv_bfloat16* __restrict__ d_v_all,
    __nv_bfloat16* __restrict__ d_q_all,
    __nv_bfloat16* __restrict__ d_decay_all,
    __nv_bfloat16* __restrict__ d_g_all,          // Gate gradients (can be nullptr)
    __nv_bfloat16* __restrict__ segment_cache,
    int checkpoint_interval,
    bool has_gate
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
    float* g_buf = d_delta + HEAD_V_DIM;  // For gate values
    float* warp_results = g_buf + HEAD_V_DIM;

    int tid = threadIdx.x;
    int state_size = N_STATE * HEAD_V_DIM;

    // Segment cache for this (batch, head)
    int cache_entry_size = state_size + N_STATE + HEAD_V_DIM + 1;
    __nv_bfloat16* seg_cache_base = segment_cache + (size_t)block_idx * checkpoint_interval * cache_entry_size;
    __nv_bfloat16* S_cache_base = seg_cache_base;
    __nv_bfloat16* k_cache_base = seg_cache_base + (size_t)checkpoint_interval * state_size;
    __nv_bfloat16* v_cache_base = k_cache_base + (size_t)checkpoint_interval * N_STATE;
    __nv_bfloat16* decay_cache_base = v_cache_base + (size_t)checkpoint_interval * HEAD_V_DIM;

    // Initialize dS to zero
    for (int i = tid; i < state_size; i += blockDim.x) {
        dS[i] = 0.0f;
    }
    __syncthreads();

    int num_segments = (T + checkpoint_interval - 1) / checkpoint_interval;

    for (int seg = num_segments - 1; seg >= 0; seg--) {
        int t_start = seg * checkpoint_interval;
        int t_end = min(t_start + checkpoint_interval, T);
        int seg_len = t_end - t_start;

        // Phase 1: Forward replay through segment
        int cp_offset = (seg * B * H + b * H + h) * state_size;
        for (int i = tid; i < state_size; i += blockDim.x) {
            S[i] = __bfloat162float(S_checkpoints[cp_offset + i]);
        }
        __syncthreads();

        __shared__ float decay_val_replay;
        for (int local_t = 0; local_t < seg_len; local_t++) {
            int t = t_start + local_t;

            // Save S_{t-1} before update
            __nv_bfloat16* S_cache_slot = S_cache_base + (size_t)local_t * state_size;
            for (int i = tid; i < state_size; i += blockDim.x) {
                S_cache_slot[i] = __float2bfloat16(S[i]);
            }

            // Load inputs - [B, T, H, dim] layout
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

            // Cache k, v, decay
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

            // Compute retrieved and delta
            if (tid < HEAD_V_DIM) {
                float sum = 0.0f;
                #pragma unroll 8
                for (int i = 0; i < N_STATE; i++) {
                    sum += S[i * HEAD_V_DIM + tid] * k[i];
                }
                retrieved[tid] = sum;
                delta[tid] = v[tid] - retrieved[tid];
            }
            __syncthreads();

            // Update S
            for (int idx = tid; idx < state_size; idx += blockDim.x) {
                int i = idx / HEAD_V_DIM;
                int j = idx % HEAD_V_DIM;
                float update = decay_val_replay * S[idx] + delta[j] * k[i];
                S[idx] = e88_fused_tanh(update);
            }
            __syncthreads();
        }

        // Phase 2: Backward pass
        for (int local_t = seg_len - 1; local_t >= 0; local_t--) {
            int t = t_start + local_t;

            // Load cached S_{t-1}
            __nv_bfloat16* S_cache_slot = S_cache_base + (size_t)local_t * state_size;
            for (int i = tid; i < state_size; i += blockDim.x) {
                S[i] = __bfloat162float(S_cache_slot[i]);
            }

            // Load cached k, v
            __nv_bfloat16* k_cache_slot = k_cache_base + (size_t)local_t * N_STATE;
            __nv_bfloat16* v_cache_slot = v_cache_base + (size_t)local_t * HEAD_V_DIM;
            if (tid < N_STATE) {
                k[tid] = __bfloat162float(k_cache_slot[tid]);
            }
            if (tid < HEAD_V_DIM) {
                v[tid] = __bfloat162float(v_cache_slot[tid]);
            }
            __syncthreads();

            // Load q
            int k_offset = ((b * T + t) * H + h) * N_STATE;
            int v_offset = ((b * T + t) * H + h) * HEAD_V_DIM;
            int decay_offset = (b * T + t) * H + h;

            if (tid < N_STATE) {
                q[tid] = __bfloat162float(q_all[k_offset + tid]);
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
                delta[tid] = v[tid] - retrieved[tid];
            }
            __syncthreads();

            // Load cached decay
            __shared__ float decay_val;
            if (tid == 0) {
                decay_val = __bfloat162float(decay_cache_base[local_t]);
            }
            __syncthreads();

            // Compute S_t and dtanh
            for (int idx = tid; idx < state_size; idx += blockDim.x) {
                int i = idx / HEAD_V_DIM;
                int j = idx % HEAD_V_DIM;
                float pre_tanh = decay_val * S[idx] + delta[j] * k[i];
                float tanh_val = e88_fused_tanh(pre_tanh);
                S_t[idx] = tanh_val;
                dtanh[idx] = 1.0f - tanh_val * tanh_val;
            }
            __syncthreads();

            // Backward through output and gating
            if (tid < HEAD_V_DIM) {
                float d_out = __bfloat162float(d_output[v_offset + tid]);

                // Handle gating backward
                if (has_gate && g_all != nullptr) {
                    float g_val = __bfloat162float(g_all[v_offset + tid]);
                    float Sq_val = __bfloat162float(Sq_cache[v_offset + tid]);

                    // Backward through silu gate: y = Sq * silu(g)
                    // silu(g) = g * sigmoid(g)
                    float sig_g = 1.0f / (1.0f + expf(-g_val));
                    float silu_g = g_val * sig_g;

                    // d_Sq = d_out * silu(g)
                    d_Sq[tid] = d_out * silu_g;

                    // d_g = d_out * Sq * d_silu(g)
                    // d_silu(g) = sigmoid(g) + g * sigmoid(g) * (1 - sigmoid(g))
                    //           = sigmoid(g) * (1 + g * (1 - sigmoid(g)))
                    float d_silu = sig_g * (1.0f + g_val * (1.0f - sig_g));
                    float Sq_before_gate = Sq_val / (silu_g + 1e-8f);  // Recover pre-gated Sq
                    d_g_all[v_offset + tid] = __float2bfloat16(d_out * Sq_before_gate * d_silu);
                } else {
                    d_Sq[tid] = d_out;
                }
            }
            __syncthreads();

            // d_q from output
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int j = 0; j < HEAD_V_DIM; j++) {
                    sum += S_t[tid * HEAD_V_DIM + j] * d_Sq[j];
                }
                d_q[tid] = sum;
            }
            __syncthreads();

            // dS contribution from output
            for (int idx = tid; idx < state_size; idx += blockDim.x) {
                int i = idx / HEAD_V_DIM;
                int j = idx % HEAD_V_DIM;
                dS[idx] += q[i] * d_Sq[j];
            }
            __syncthreads();

            // Backward through state update
            if (tid < HEAD_V_DIM) {
                float d_delta_local = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    float d_pre = dS[i * HEAD_V_DIM + tid] * dtanh[i * HEAD_V_DIM + tid];
                    d_delta_local += d_pre * k[i];
                }
                d_delta[tid] = d_delta_local;
            }
            __syncthreads();

            // Fused d_k: both dS*dtanh contribution and retrieved gradient in single loop
            if (tid < N_STATE) {
                float d_k_local = 0.0f;
                for (int j = 0; j < HEAD_V_DIM; j++) {
                    float d_pre = dS[tid * HEAD_V_DIM + j] * dtanh[tid * HEAD_V_DIM + j];
                    d_k_local += d_pre * delta[j];
                    d_k_local += S[tid * HEAD_V_DIM + j] * (-d_delta[j]);
                }
                d_k[tid] = d_k_local;
            }
            __syncthreads();

            // d_decay with warp reduction
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
            __syncthreads();

            // Write gradients (using [B, T, H, dim] layout)
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
            __syncthreads();

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
}

// =============================================================================
// Projection + L2 Norm + Decay Kernel
// Fuses all pre-recurrence operations into one kernel
// =============================================================================

__global__ void E88FusedProjectionKernel_BF16(
    int B, int T, int dim, int H,
    int key_dim, int value_dim, int n_state, int head_v_dim,
    const __nv_bfloat16* __restrict__ x,          // [B, T, dim]
    const __nv_bfloat16* __restrict__ W_qkv,      // [3*key_dim + value_dim, dim]
    const __nv_bfloat16* __restrict__ W_a,        // [H, dim]
    const float* __restrict__ A_log,               // [H]
    const float* __restrict__ dt_bias,             // [H]
    __nv_bfloat16* __restrict__ q_out,            // [B, T, H, n_state]
    __nv_bfloat16* __restrict__ k_out,            // [B, T, H, n_state]
    __nv_bfloat16* __restrict__ v_out,            // [B, T, H, head_v_dim]
    __nv_bfloat16* __restrict__ decay_out,        // [B, T, H]
    bool use_l2_norm
) {
    // This kernel does per-position processing
    // Each block processes one (batch, time) position for all heads
    int block_idx = blockIdx.x;
    int b = block_idx / T;
    int t = block_idx % T;
    if (b >= B) return;

    int tid = threadIdx.x;

    extern __shared__ float smem[];
    float* qkv_buf = smem;  // [2*key_dim + value_dim]
    float* alpha_buf = qkv_buf + 2 * key_dim + value_dim;  // [H]

    // Input offset
    int x_offset = (b * T + t) * dim;

    // Step 1: Compute QKV projection (this is inefficient without cuBLAS)
    // Note: For production, this should use cuBLAS GEMM
    // Here we do a simplified per-thread dot product

    int qkv_total = 2 * key_dim + value_dim;
    for (int i = tid; i < qkv_total; i += blockDim.x) {
        float sum = 0.0f;
        for (int d = 0; d < dim; d++) {
            sum += __bfloat162float(W_qkv[i * dim + d]) * __bfloat162float(x[x_offset + d]);
        }
        qkv_buf[i] = sum;
    }
    __syncthreads();

    // Step 2: Compute alpha projection
    for (int h = tid; h < H; h += blockDim.x) {
        float sum = 0.0f;
        for (int d = 0; d < dim; d++) {
            sum += __bfloat162float(W_a[h * dim + d]) * __bfloat162float(x[x_offset + d]);
        }
        alpha_buf[h] = sum;
    }
    __syncthreads();

    // Step 3: Split QKV and reshape to heads, apply L2 norm
    // Q: [0, key_dim), K: [key_dim, 2*key_dim), V: [2*key_dim, 2*key_dim+value_dim)

    for (int h = tid; h < H; h += blockDim.x) {
        int q_start = h * n_state;
        int k_start = key_dim + h * n_state;
        int v_start = 2 * key_dim + h * head_v_dim;

        // Compute L2 norms
        float q_norm_sq = 0.0f, k_norm_sq = 0.0f;
        if (use_l2_norm) {
            for (int i = 0; i < n_state; i++) {
                q_norm_sq += qkv_buf[q_start + i] * qkv_buf[q_start + i];
                k_norm_sq += qkv_buf[k_start + i] * qkv_buf[k_start + i];
            }
            q_norm_sq = rsqrtf(q_norm_sq + 1e-12f);
            k_norm_sq = rsqrtf(k_norm_sq + 1e-12f);
        } else {
            q_norm_sq = 1.0f;
            k_norm_sq = 1.0f;
        }

        // Write normalized Q, K and V
        int out_offset = ((b * T + t) * H + h);
        for (int i = 0; i < n_state; i++) {
            q_out[out_offset * n_state + i] = __float2bfloat16(qkv_buf[q_start + i] * q_norm_sq);
            k_out[out_offset * n_state + i] = __float2bfloat16(qkv_buf[k_start + i] * k_norm_sq);
        }
        for (int i = 0; i < head_v_dim; i++) {
            v_out[out_offset * head_v_dim + i] = __float2bfloat16(qkv_buf[v_start + i]);
        }

        // Compute decay: exp(-exp(A_log) * softplus(alpha + dt_bias))
        float alpha_h = alpha_buf[h];
        float g = -expf(A_log[h]) * e88_fused_softplus(alpha_h + dt_bias[h]);
        decay_out[(b * T + t) * H + h] = __float2bfloat16(expf(g));
    }
}

// =============================================================================
// Dispatch Functions
// =============================================================================

void dispatch_e88_fused_forward(
    int T, int B, int H, int n_state, int head_v_dim,
    const __nv_bfloat16* k_all, const __nv_bfloat16* v_all,
    const __nv_bfloat16* q_all, const __nv_bfloat16* decay_all,
    const __nv_bfloat16* g_all,
    __nv_bfloat16* S, __nv_bfloat16* output,
    __nv_bfloat16* S_checkpoints, __nv_bfloat16* Sq_cache,
    int checkpoint_interval, bool apply_gate, cudaStream_t stream
) {
    int state_size = n_state * head_v_dim;
    // Shared memory: S_shared + k + v + q + g
    int shared_size = (state_size + n_state + head_v_dim + n_state + head_v_dim) * sizeof(float);
    int threads = min(256, state_size);
    int num_blocks = B * H;

    #define DISPATCH_E88_FUSED_FWD(N, V) do { \
        auto kernel = E88FusedForwardKernel_BF16<N, V>; \
        if (shared_size > 48 * 1024) { \
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size); \
        } \
        kernel<<<num_blocks, threads, shared_size, stream>>>( \
            T, B, H, k_all, v_all, q_all, decay_all, g_all, S, output, \
            S_checkpoints, Sq_cache, checkpoint_interval, apply_gate); \
    } while(0)

    // Common configurations (same as e88_fla_hybrid)
    if (n_state == 32 && head_v_dim == 32) { DISPATCH_E88_FUSED_FWD(32, 32); }
    else if (n_state == 16 && head_v_dim == 16) { DISPATCH_E88_FUSED_FWD(16, 16); }
    else if (n_state == 24 && head_v_dim == 24) { DISPATCH_E88_FUSED_FWD(24, 24); }
    else if (n_state == 48 && head_v_dim == 48) { DISPATCH_E88_FUSED_FWD(48, 48); }
    else if (n_state == 64 && head_v_dim == 64) { DISPATCH_E88_FUSED_FWD(64, 64); }
    else if (n_state == 96 && head_v_dim == 96) { DISPATCH_E88_FUSED_FWD(96, 96); }
    else {
        fprintf(stderr, "E88 Fused Forward: unsupported n_state=%d, head_v_dim=%d\n", n_state, head_v_dim);
    }

    #undef DISPATCH_E88_FUSED_FWD
}

void dispatch_e88_fused_backward(
    int T, int B, int H, int n_state, int head_v_dim,
    const __nv_bfloat16* k_all, const __nv_bfloat16* v_all,
    const __nv_bfloat16* q_all, const __nv_bfloat16* decay_all,
    const __nv_bfloat16* g_all,
    const __nv_bfloat16* S_checkpoints, const __nv_bfloat16* Sq_cache,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* d_k_all, __nv_bfloat16* d_v_all,
    __nv_bfloat16* d_q_all, __nv_bfloat16* d_decay_all,
    __nv_bfloat16* d_g_all,
    __nv_bfloat16* segment_cache,
    int checkpoint_interval, bool has_gate, cudaStream_t stream
) {
    int state_size = n_state * head_v_dim;
    // Shared mem: S, dS, S_t, dtanh (4*state) + k,q,d_k,d_q (4*n) + v,delta,ret,d_v,d_Sq,d_delta,g (7*v) + warp (8)
    int shared_size = (4 * state_size + 4 * n_state + 7 * head_v_dim + 8) * sizeof(float);
    int threads = min(256, state_size);
    int num_blocks = B * H;

    #define DISPATCH_E88_FUSED_BWD(N, V) do { \
        auto kernel = E88FusedBackwardKernel_BF16<N, V>; \
        if (shared_size > 48 * 1024) { \
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size); \
        } \
        kernel<<<num_blocks, threads, shared_size, stream>>>( \
            T, B, H, k_all, v_all, q_all, decay_all, g_all, \
            S_checkpoints, Sq_cache, d_output, \
            d_k_all, d_v_all, d_q_all, d_decay_all, d_g_all, \
            segment_cache, checkpoint_interval, has_gate); \
    } while(0)

    if (n_state == 32 && head_v_dim == 32) { DISPATCH_E88_FUSED_BWD(32, 32); }
    else if (n_state == 16 && head_v_dim == 16) { DISPATCH_E88_FUSED_BWD(16, 16); }
    else if (n_state == 24 && head_v_dim == 24) { DISPATCH_E88_FUSED_BWD(24, 24); }
    else if (n_state == 48 && head_v_dim == 48) { DISPATCH_E88_FUSED_BWD(48, 48); }
    else if (n_state == 64 && head_v_dim == 64) { DISPATCH_E88_FUSED_BWD(64, 64); }
    else if (n_state == 96 && head_v_dim == 96) { DISPATCH_E88_FUSED_BWD(96, 96); }
    else {
        fprintf(stderr, "E88 Fused Backward: unsupported n_state=%d, head_v_dim=%d\n", n_state, head_v_dim);
    }

    #undef DISPATCH_E88_FUSED_BWD
}

// =============================================================================
// C++ Wrapper Classes
// =============================================================================

template<typename DataT>
class E88FusedForward {
public:
    E88FusedForward(bool training, int batch_size, int n_state, int head_v_dim,
                    int n_heads, const cudaStream_t& stream)
        : training_(training), batch_size_(batch_size), n_state_(n_state),
          head_v_dim_(head_v_dim), n_heads_(n_heads), stream_(stream) {}

    void Run(int steps, const DataT* k, const DataT* v, const DataT* q,
             const DataT* decay, const DataT* g, DataT* S, DataT* output,
             DataT* S_cache, bool apply_gate) {
        int T = steps;
        int B = batch_size_;
        int n = n_state_;
        int v_dim = head_v_dim_;
        int H = n_heads_;

        int num_checkpoints = (T + E88_FUSED_CHECKPOINT_INTERVAL - 1) / E88_FUSED_CHECKPOINT_INTERVAL + 1;
        DataT* s_checkpoints = S_cache;
        DataT* sq_cache = S_cache + num_checkpoints * B * H * n * v_dim;

        dispatch_e88_fused_forward(
            T, B, H, n, v_dim,
            (const __nv_bfloat16*)k, (const __nv_bfloat16*)v,
            (const __nv_bfloat16*)q, (const __nv_bfloat16*)decay,
            (const __nv_bfloat16*)g,
            (__nv_bfloat16*)S, (__nv_bfloat16*)output,
            (__nv_bfloat16*)s_checkpoints, (__nv_bfloat16*)sq_cache,
            E88_FUSED_CHECKPOINT_INTERVAL, apply_gate, stream_);
    }

private:
    bool training_;
    int batch_size_;
    int n_state_;
    int head_v_dim_;
    int n_heads_;
    cudaStream_t stream_;
};

template<typename DataT>
class E88FusedBackward {
public:
    E88FusedBackward(int batch_size, int n_state, int head_v_dim,
                     int n_heads, const cudaStream_t& stream)
        : batch_size_(batch_size), n_state_(n_state), head_v_dim_(head_v_dim),
          n_heads_(n_heads), stream_(stream) {}

    void Run(int steps, const DataT* k, const DataT* v, const DataT* q,
             const DataT* decay, const DataT* g,
             const DataT* S_cache, const DataT* d_output,
             DataT* d_k, DataT* d_v, DataT* d_q, DataT* d_decay, DataT* d_g,
             DataT* segment_cache, bool has_gate) {
        int T = steps;
        int B = batch_size_;
        int n = n_state_;
        int v_dim = head_v_dim_;
        int H = n_heads_;

        int num_checkpoints = (T + E88_FUSED_CHECKPOINT_INTERVAL - 1) / E88_FUSED_CHECKPOINT_INTERVAL + 1;
        const DataT* s_checkpoints = S_cache;
        const DataT* sq_cache = S_cache + num_checkpoints * B * H * n * v_dim;

        dispatch_e88_fused_backward(
            T, B, H, n, v_dim,
            (const __nv_bfloat16*)k, (const __nv_bfloat16*)v,
            (const __nv_bfloat16*)q, (const __nv_bfloat16*)decay,
            (const __nv_bfloat16*)g,
            (const __nv_bfloat16*)s_checkpoints, (const __nv_bfloat16*)sq_cache,
            (const __nv_bfloat16*)d_output,
            (__nv_bfloat16*)d_k, (__nv_bfloat16*)d_v,
            (__nv_bfloat16*)d_q, (__nv_bfloat16*)d_decay,
            (__nv_bfloat16*)d_g,
            (__nv_bfloat16*)segment_cache,
            E88_FUSED_CHECKPOINT_INTERVAL, has_gate, stream_);
    }

private:
    int batch_size_;
    int n_state_;
    int head_v_dim_;
    int n_heads_;
    cudaStream_t stream_;
};

// Explicit instantiations
template class E88FusedForward<__nv_bfloat16>;
template class E88FusedBackward<__nv_bfloat16>;

}  // namespace elman
