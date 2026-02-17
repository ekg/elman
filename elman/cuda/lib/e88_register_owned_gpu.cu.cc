/**
 * E88 Register-Owned State Backward Kernel
 *
 * Key optimization: Each thread owns ONE FULL COLUMN of the state matrix
 * in registers. For N_STATE=32, HEAD_V_DIM=32: 32 floats = 128 bytes per thread.
 *
 * With 32 threads per block (1 warp):
 * - Thread j owns S[:,j] (column j of the state matrix)
 * - Thread j computes retrieved[j], delta[j], d_k[j], d_v[j], etc.
 * - Cross-thread communication via warp shuffles (no __syncthreads needed!)
 *
 * Generalization (Tier 1):
 * - Supports N_STATE in {4, 8, 16, 24, 32} and HEAD_V_DIM in {4, 8, 16, 24, 32}
 * - For HEAD_V_DIM < 32: threads >= HEAD_V_DIM are inactive
 * - Uses ACTIVE_MASK for warp shuffle operations
 *
 * Benefits:
 * - No shared memory for state (state lives in registers)
 * - No __syncthreads for state operations (warp is lockstep)
 * - Expected speedup: 2-8x over fused_backward depending on size
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
 * Register-owned backward kernel for arbitrary N_STATE, HEAD_V_DIM <= 32
 *
 * Thread layout: 32 threads per block (1 warp)
 * - Thread j owns column j of S (N_STATE elements in registers)
 * - For HEAD_V_DIM < 32: threads j >= HEAD_V_DIM are inactive
 *
 * Template parameters:
 *   N_STATE:    rows in state matrix (key dimension)
 *   HEAD_V_DIM: columns in state matrix (value dimension), max 32
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
    bool has_gate,
    bool normalize_kq
) {
    static_assert(HEAD_V_DIM <= 32,
                  "Tier 1 register-owned kernel requires HEAD_V_DIM <= 32");
    static_assert(N_STATE <= 64,
                  "Tier 1 register-owned kernel requires N_STATE <= 64");

    int block_idx = blockIdx.x;
    int b = block_idx / H;
    int h = block_idx % H;
    if (b >= B) return;

    int tid = threadIdx.x;  // tid in [0, 31]

    // Active mask for warp operations: only threads 0..HEAD_V_DIM-1 participate
    // Special case: HEAD_V_DIM == 32 means all threads active (avoid UB from 1u << 32)
    constexpr unsigned ACTIVE_MASK = (HEAD_V_DIM >= 32) ? 0xFFFFFFFFu : ((1u << HEAD_V_DIM) - 1);
    const bool is_active = (tid < HEAD_V_DIM);

    int state_size = N_STATE * HEAD_V_DIM;

    // Thread j owns column j of the state matrix (if active)
    // S_reg[i] = S[i, j] where j = tid
    float S_reg[N_STATE];
    float dS_reg[N_STATE];
    float S_t_reg[N_STATE];
    float dtanh_reg[N_STATE];

    // Initialize dS to zero
    #pragma unroll
    for (int i = 0; i < N_STATE; i++) {
        S_reg[i] = 0.0f;
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

        // Phase 1: Load checkpoint into registers (active threads load their columns)
        int cp_offset = (seg * B * H + b * H + h) * state_size;
        if (is_active) {
            #pragma unroll
            for (int i = 0; i < N_STATE; i++) {
                S_reg[i] = __bfloat162float(S_checkpoints[cp_offset + i * HEAD_V_DIM + tid]);
            }
        }

        // Phase 2: Forward replay through segment to cache intermediate values
        for (int local_t = 0; local_t < seg_len; local_t++) {
            int t = t_start + local_t;

            // Cache S_{t-1} before update (only active threads)
            __nv_bfloat16* S_cache_slot = S_cache_base + (size_t)local_t * state_size;
            if (is_active) {
                #pragma unroll
                for (int i = 0; i < N_STATE; i++) {
                    S_cache_slot[i * HEAD_V_DIM + tid] = __float2bfloat16(S_reg[i]);
                }
            }

            // Load inputs (cooperative load into shared memory)
            int k_offset = ((b * T + t) * H + h) * N_STATE;
            int v_offset = ((b * T + t) * H + h) * HEAD_V_DIM;
            int decay_offset = (b * T + t) * H + h;

            // Warp-cooperative load: thread tid loads k[tid] and v[tid] if valid
            if (tid < N_STATE) {
                k_shared[tid] = __bfloat162float(k_all[k_offset + tid]);
            }
            if (tid < HEAD_V_DIM) {
                v_shared[tid] = __bfloat162float(v_all[v_offset + tid]);
            }
            if (tid == 0) {
                decay_shared = __bfloat162float(decay_all[decay_offset]);
            }
            __syncwarp();

            // In-kernel L2 normalization of k
            if (normalize_kq) {
                // Compute ||k||^2 via warp reduction
                float k_sq = (tid < N_STATE) ? (k_shared[tid] * k_shared[tid]) : 0.0f;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    k_sq += __shfl_xor_sync(0xFFFFFFFF, k_sq, offset);
                }
                // All threads have k_sq = ||k||^2 now
                float k_inv = rsqrtf(k_sq + 1e-12f);
                if (tid < N_STATE) {
                    k_shared[tid] *= k_inv;
                }
                __syncwarp();
            }

            // Cache k, v, decay
            __nv_bfloat16* k_cache_slot = k_cache_base + (size_t)local_t * N_STATE;
            __nv_bfloat16* v_cache_slot = v_cache_base + (size_t)local_t * HEAD_V_DIM;
            if (tid < N_STATE) {
                k_cache_slot[tid] = __float2bfloat16(k_shared[tid]);
            }
            if (tid < HEAD_V_DIM) {
                v_cache_slot[tid] = __float2bfloat16(v_shared[tid]);
            }
            if (tid == 0) {
                decay_cache_base[local_t] = __float2bfloat16(decay_shared);
            }
            __syncwarp();

            // Compute retrieved[tid] = sum_i(S[i, tid] * k[i]) (active threads only)
            float retrieved = 0.0f;
            float delta = 0.0f;
            if (is_active) {
                #pragma unroll
                for (int i = 0; i < N_STATE; i++) {
                    retrieved += S_reg[i] * k_shared[i];
                }

                // delta[tid] = v[tid] - retrieved[tid]
                delta = v_shared[tid] - retrieved;

                // Update state: S[i, tid] = tanh(decay * S[i, tid] + delta * k[i])
                #pragma unroll
                for (int i = 0; i < N_STATE; i++) {
                    float pre_tanh = decay_shared * S_reg[i] + delta * k_shared[i];
                    S_reg[i] = e88_reg_tanh(pre_tanh);
                }
            }
        }

        // Phase 3: Backward pass through segment
        for (int local_t = seg_len - 1; local_t >= 0; local_t--) {
            int t = t_start + local_t;

            // Load cached S_{t-1} into registers (active threads only)
            __nv_bfloat16* S_cache_slot = S_cache_base + (size_t)local_t * state_size;
            if (is_active) {
                #pragma unroll
                for (int i = 0; i < N_STATE; i++) {
                    S_reg[i] = __bfloat162float(S_cache_slot[i * HEAD_V_DIM + tid]);
                }
            }

            // Load cached k, v into shared memory
            __nv_bfloat16* k_cache_slot = k_cache_base + (size_t)local_t * N_STATE;
            __nv_bfloat16* v_cache_slot = v_cache_base + (size_t)local_t * HEAD_V_DIM;
            if (tid < N_STATE) {
                k_shared[tid] = __bfloat162float(k_cache_slot[tid]);
            }
            if (tid < HEAD_V_DIM) {
                v_shared[tid] = __bfloat162float(v_cache_slot[tid]);
            }
            if (tid == 0) {
                decay_shared = __bfloat162float(decay_cache_base[local_t]);
            }
            __syncwarp();

            // Load q (only threads < N_STATE)
            int k_offset = ((b * T + t) * H + h) * N_STATE;
            int v_offset = ((b * T + t) * H + h) * HEAD_V_DIM;
            int decay_offset = (b * T + t) * H + h;
            float q_val = (tid < N_STATE) ? __bfloat162float(q_all[k_offset + tid]) : 0.0f;

            // In-kernel L2 normalization of q (keep raw value for backward chain rule)
            float q_raw_val = q_val;
            float q_norm_inv = 1.0f;  // 1/||q_raw||
            if (normalize_kq) {
                float q_sq = (tid < N_STATE) ? (q_val * q_val) : 0.0f;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    q_sq += __shfl_xor_sync(0xFFFFFFFF, q_sq, offset);
                }
                q_norm_inv = rsqrtf(q_sq + 1e-12f);
                q_val *= q_norm_inv;
            }

            // Load unnormalized k from global memory for L2 norm backward chain rule
            float k_raw_val = 0.0f;
            float k_norm_inv = 1.0f;  // 1/||k_raw||
            if (normalize_kq) {
                k_raw_val = (tid < N_STATE) ? __bfloat162float(k_all[k_offset + tid]) : 0.0f;
                float k_sq = (tid < N_STATE) ? (k_raw_val * k_raw_val) : 0.0f;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    k_sq += __shfl_xor_sync(0xFFFFFFFF, k_sq, offset);
                }
                k_norm_inv = rsqrtf(k_sq + 1e-12f);
            }

            // Recompute forward values (active threads only)
            float retrieved = 0.0f;
            float delta = 0.0f;
            if (is_active) {
                #pragma unroll
                for (int i = 0; i < N_STATE; i++) {
                    retrieved += S_reg[i] * k_shared[i];
                }
                delta = v_shared[tid] - retrieved;

                // Compute S_t and dtanh
                #pragma unroll
                for (int i = 0; i < N_STATE; i++) {
                    float pre_tanh = decay_shared * S_reg[i] + delta * k_shared[i];
                    float tanh_val = e88_reg_tanh(pre_tanh);
                    S_t_reg[i] = tanh_val;
                    dtanh_reg[i] = 1.0f - tanh_val * tanh_val;
                }
            }

            // Load d_output and handle gating (active threads only)
            float d_out = is_active ? __bfloat162float(d_output[v_offset + tid]) : 0.0f;
            float d_Sq = 0.0f;

            if (is_active) {
                if (has_gate && g_all != nullptr) {
                    float g_val = __bfloat162float(g_all[v_offset + tid]);
                    // Sq_cache now stores PRE-GATED Sq (no division needed)
                    float Sq_pre_gate = __bfloat162float(Sq_cache[v_offset + tid]);

                    float sig_g = 1.0f / (1.0f + expf(-g_val));
                    float silu_g = g_val * sig_g;

                    // d_Sq = d_out * silu(g)
                    d_Sq = d_out * silu_g;

                    // d_g = d_out * Sq_pre_gate * d_silu(g)
                    // d_silu(g) = sigmoid(g) * (1 + g * (1 - sigmoid(g)))
                    float d_silu = sig_g * (1.0f + g_val * (1.0f - sig_g));
                    d_g_all[v_offset + tid] = __float2bfloat16(d_out * Sq_pre_gate * d_silu);
                } else {
                    d_Sq = d_out;
                }
            }

            // d_q[i] = sum_j(S_t[i, j] * d_Sq[j])
            // Thread j has S_t_reg[i] = S_t[i, j] and d_Sq[j]
            // Thread j contributes S_t[i, j] * d_Sq[j] to d_q[i]

            // First, compute local contribution to d_q (active threads only)
            float d_q_contrib[N_STATE];
            #pragma unroll
            for (int i = 0; i < N_STATE; i++) {
                d_q_contrib[i] = is_active ? (S_t_reg[i] * d_Sq) : 0.0f;
            }

            // Warp shuffle reduction across all threads (inactive have 0)
            // Use full mask - inactive threads have 0 contributions
            float d_q_local[N_STATE];
            #pragma unroll
            for (int i = 0; i < N_STATE; i++) {
                float val = d_q_contrib[i];
                // Full warp reduce (inactive threads contribute 0)
                #pragma unroll
                for (int offset = 16; offset >= 1; offset /= 2) {
                    val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
                }
                d_q_local[i] = val;
            }
            // Apply L2 norm backward chain rule for q if normalize_kq
            // d_q_raw[i] = (d_q_norm[i] - q_norm[i] * dot(d_q_norm, q_norm)) / ||q_raw||
            if (normalize_kq) {
                // d_q_local[tid] is d_q w.r.t. normalized q (all threads have all values after xor reduce)
                // q_val is the normalized q[tid], q_norm_inv = 1/||q_raw||
                float dq_dot_qn = (tid < N_STATE) ? (d_q_local[tid] * q_val) : 0.0f;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    dq_dot_qn += __shfl_xor_sync(0xFFFFFFFF, dq_dot_qn, offset);
                }
                // All threads have dot(d_q_norm, q_norm)
                if (tid < N_STATE) {
                    float d_q_raw = (d_q_local[tid] - q_val * dq_dot_qn) * q_norm_inv;
                    d_q_all[k_offset + tid] = __float2bfloat16(d_q_raw);
                }
            } else {
                // Write d_q[tid] directly (already w.r.t. normalized q which was input)
                if (tid < N_STATE) {
                    d_q_all[k_offset + tid] = __float2bfloat16(d_q_local[tid]);
                }
            }

            // Add dS contribution from output: dS[i, j] += q[i] * d_Sq[j]
            // Thread j has d_Sq[j], needs to add q[i] * d_Sq to dS_reg[i]
            // Need q from other threads via shuffle (q lives in threads 0..N_STATE-1)
            // ALL threads must participate in the shuffle, but only active threads update dS
            #pragma unroll
            for (int i = 0; i < N_STATE; i++) {
                // ALL threads shuffle (required by __shfl_sync with full mask)
                float q_i = __shfl_sync(0xFFFFFFFF, q_val, i);
                if (is_active) {
                    dS_reg[i] += q_i * d_Sq;
                }
            }

            // Backward through state update (active threads only)
            // d_delta[j] = sum_i(dS[i,j] * dtanh[i,j] * k[i])
            float d_delta = 0.0f;
            if (is_active) {
                #pragma unroll
                for (int i = 0; i < N_STATE; i++) {
                    float d_pre = dS_reg[i] * dtanh_reg[i];
                    d_delta += d_pre * k_shared[i];
                }

                // d_v[j] = d_delta[j]
                d_v_all[v_offset + tid] = __float2bfloat16(d_delta);
            }

            // d_k[i] = sum_j(dS[i,j] * dtanh[i,j] * delta[j]) + sum_j(S[i,j] * (-d_delta[j]))
            // Thread j has dS_reg[i], dtanh_reg[i], delta, S_reg[i], d_delta for column j
            // d_k[i] needs contributions from all active j

            // Compute local contribution for each i (active threads only)
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

            // Warp reduce to get d_k_norm[i] = sum_j(d_k_contrib[i])
            // After xor reduction, all threads have the same sum for each i.
            // Thread tid captures d_k_norm[tid] from the i==tid iteration.
            float my_dk_norm = 0.0f;
            #pragma unroll
            for (int i = 0; i < N_STATE; i++) {
                float val = d_k_contrib[i];
                #pragma unroll
                for (int offset = 16; offset >= 1; offset /= 2) {
                    val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
                }
                if (tid == i) my_dk_norm = val;
            }

            // Apply L2 norm backward chain rule for k if normalize_kq
            if (normalize_kq) {
                // k_shared[tid] contains normalized k[tid], k_norm_inv = 1/||k_raw||
                // Compute dot(d_k_norm, k_norm) = sum_i d_k_norm[i] * k_norm[i]
                float dk_dot_kn = (tid < N_STATE) ? (my_dk_norm * k_shared[tid]) : 0.0f;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    dk_dot_kn += __shfl_xor_sync(0xFFFFFFFF, dk_dot_kn, offset);
                }
                if (tid < N_STATE) {
                    float d_k_raw = (my_dk_norm - k_shared[tid] * dk_dot_kn) * k_norm_inv;
                    d_k_all[k_offset + tid] = __float2bfloat16(d_k_raw);
                }
            } else {
                if (tid < N_STATE) {
                    d_k_all[k_offset + tid] = __float2bfloat16(my_dk_norm);
                }
            }

            // d_decay = sum_{i,j}(dS[i,j] * dtanh[i,j] * S[i,j])
            // Thread j contributes sum_i(dS_reg[i] * dtanh_reg[i] * S_reg[i]) (active only)
            float d_decay_local = 0.0f;
            if (is_active) {
                #pragma unroll
                for (int i = 0; i < N_STATE; i++) {
                    d_decay_local += dS_reg[i] * dtanh_reg[i] * S_reg[i];
                }
            }
            // Full warp reduce - inactive have d_decay_local=0
            #pragma unroll
            for (int offset = 16; offset >= 1; offset /= 2) {
                d_decay_local += __shfl_xor_sync(0xFFFFFFFF, d_decay_local, offset);
            }
            if (tid == 0) {
                d_decay_all[decay_offset] = __float2bfloat16(d_decay_local);
            }

            // Update dS for next iteration (t-1) (active threads only)
            // dS_{t-1}[i,j] = dS_t[i,j] * dtanh[i,j] * decay + (-d_delta[j]) * k[i]
            if (is_active) {
                #pragma unroll
                for (int i = 0; i < N_STATE; i++) {
                    float d_pre = dS_reg[i] * dtanh_reg[i];
                    dS_reg[i] = d_pre * decay_shared + (-d_delta) * k_shared[i];
                }
            }
        }
    }
}

// Helper macro for kernel launch
#define LAUNCH_REGISTER_OWNED_KERNEL(NS, HVD) \
    do { \
        int shared_mem = ((NS) + (HVD) + 1) * sizeof(float); \
        E88RegisterOwnedBackwardKernel_BF16<NS, HVD><<<num_blocks, threads, shared_mem, stream>>>( \
            T, B, H, k, v, q, decay, g, S_checkpoints, Sq_cache, d_output, \
            d_k, d_v, d_q, d_decay, d_g, segment_cache, \
            checkpoint_interval, has_gate, normalize_kq \
        ); \
    } while(0)

// Dispatch function with support for multiple sizes
// Tier 1: Single warp (head_v_dim <= 32)
// Thread j owns column j, so we need exactly head_v_dim threads
// For head_v_dim <= 32: single warp with inactive lanes
// n_state can be up to 64 (each thread holds larger arrays)
void dispatch_e88_register_owned_backward(
    int T, int B, int H, int n_state, int head_v_dim,
    const __nv_bfloat16* k, const __nv_bfloat16* v, const __nv_bfloat16* q,
    const __nv_bfloat16* decay, const __nv_bfloat16* g,
    const __nv_bfloat16* S_checkpoints, const __nv_bfloat16* Sq_cache,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* d_k, __nv_bfloat16* d_v, __nv_bfloat16* d_q,
    __nv_bfloat16* d_decay, __nv_bfloat16* d_g,
    __nv_bfloat16* segment_cache,
    int checkpoint_interval, bool has_gate, bool normalize_kq, cudaStream_t stream
) {
    // Tier 1 constraints:
    // - head_v_dim <= 32 (single warp)
    // - n_state <= 64 (register budget: 4 arrays × n_state floats <= 255 regs)
    if (head_v_dim > 32) {
        fprintf(stderr, "Register-owned Tier 1 requires head_v_dim<=32, got %d (needs Tier 2 multi-warp)\n",
                head_v_dim);
        return;
    }
    if (n_state > 64) {
        fprintf(stderr, "Register-owned kernel requires n_state<=64, got %d (register pressure)\n",
                n_state);
        return;
    }

    int num_blocks = B * H;
    int threads = 32;  // One warp per block

    // Dispatch based on (n_state, head_v_dim) combination
    // Square states (most common for E88)
    if (n_state == 32 && head_v_dim == 32) {
        LAUNCH_REGISTER_OWNED_KERNEL(32, 32);
    }
    else if (n_state == 24 && head_v_dim == 24) {
        LAUNCH_REGISTER_OWNED_KERNEL(24, 24);
    }
    else if (n_state == 16 && head_v_dim == 16) {
        LAUNCH_REGISTER_OWNED_KERNEL(16, 16);
    }
    else if (n_state == 8 && head_v_dim == 8) {
        LAUNCH_REGISTER_OWNED_KERNEL(8, 8);
    }
    else if (n_state == 4 && head_v_dim == 4) {
        LAUNCH_REGISTER_OWNED_KERNEL(4, 4);
    }
    // Rectangular states: tall (n_state > head_v_dim)
    // Each thread holds larger column arrays - more register usage
    else if (n_state == 64 && head_v_dim == 32) {
        // 64 floats × 4 arrays = 256 floats = 1024 bytes - at register limit
        LAUNCH_REGISTER_OWNED_KERNEL(64, 32);
    }
    else if (n_state == 48 && head_v_dim == 32) {
        LAUNCH_REGISTER_OWNED_KERNEL(48, 32);
    }
    else if (n_state == 40 && head_v_dim == 32) {
        LAUNCH_REGISTER_OWNED_KERNEL(40, 32);
    }
    else if (n_state == 36 && head_v_dim == 32) {
        LAUNCH_REGISTER_OWNED_KERNEL(36, 32);
    }
    else if (n_state == 32 && head_v_dim == 16) {
        LAUNCH_REGISTER_OWNED_KERNEL(32, 16);
    }
    else if (n_state == 24 && head_v_dim == 16) {
        LAUNCH_REGISTER_OWNED_KERNEL(24, 16);
    }
    else if (n_state == 16 && head_v_dim == 8) {
        LAUNCH_REGISTER_OWNED_KERNEL(16, 8);
    }
    // Rectangular states: wide (n_state < head_v_dim)
    // Each thread holds smaller column arrays - less register usage
    else if (n_state == 16 && head_v_dim == 32) {
        LAUNCH_REGISTER_OWNED_KERNEL(16, 32);
    }
    else if (n_state == 24 && head_v_dim == 32) {
        LAUNCH_REGISTER_OWNED_KERNEL(24, 32);
    }
    else if (n_state == 32 && head_v_dim == 24) {
        LAUNCH_REGISTER_OWNED_KERNEL(32, 24);
    }
    else if (n_state == 8 && head_v_dim == 16) {
        LAUNCH_REGISTER_OWNED_KERNEL(8, 16);
    }
    else if (n_state == 8 && head_v_dim == 32) {
        LAUNCH_REGISTER_OWNED_KERNEL(8, 32);
    }
    else {
        fprintf(stderr, "Unsupported n_state=%d, head_v_dim=%d for register-owned kernel\n",
                n_state, head_v_dim);
        // Caller should fall back to fused_backward
    }
}

#undef LAUNCH_REGISTER_OWNED_KERNEL

}  // namespace elman
