/**
 * E25 Dual-Memory Elman with 1.5-Entmax Attention CUDA Kernels
 *
 * Based on E23 but replaces softmax with 1.5-entmax for sparse attention.
 *
 * 1.5-entmax properties:
 *   - Produces exact zeros for low-scoring slots (sparse)
 *   - Closed-form: p_i = max(0, z_i - tau)^2 / sum(...)
 *   - Threshold tau found via sorting and statistical computation
 *   - Backward: dz = g * (dp - (g @ dp) / sum(g)) where g = sqrt(p)
 *
 * Uses hybrid template pattern: N_SLOTS is compile-time, DIM is runtime with dynamic shared memory.
 */

#include "hasty/elman_ladder.h"
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

namespace {

constexpr int E25_BLOCK_SIZE = 256;
constexpr int N_SLOTS_MAX = 64;

/**
 * 1.5-Entmax device function
 *
 * Computes sparse attention weights from scores.
 * Input: attn_sh[N_SLOTS] - attention scores
 * Output: attn_sh[N_SLOTS] - sparse attention weights (sum to 1)
 *
 * Uses dynamic shared memory for sorting arrays (passed via sorted_vals, etc.)
 */
template<int N_SLOTS>
__device__ void entmax_1_5_device(float* attn_sh, const int tid,
                                   float* sorted_vals, int* sorted_idx,
                                   float* cumsum, float* tau_candidates,
                                   int* support_size_ptr, float* tau_star_ptr) {
    // Phase 1: Initialize for sorting
    if (tid < N_SLOTS) {
        sorted_vals[tid] = attn_sh[tid];
        sorted_idx[tid] = tid;
    }
    __syncthreads();

    // Phase 2: Simple selection sort for small N (more stable than bitonic)
    // Single thread does the sort for correctness
    if (tid == 0) {
        // Selection sort descending
        #pragma unroll
        for (int i = 0; i < N_SLOTS - 1; i++) {
            int max_idx = i;
            for (int j = i + 1; j < N_SLOTS; j++) {
                if (sorted_vals[j] > sorted_vals[max_idx]) {
                    max_idx = j;
                }
            }
            if (max_idx != i) {
                float tmp_val = sorted_vals[i];
                sorted_vals[i] = sorted_vals[max_idx];
                sorted_vals[max_idx] = tmp_val;
                int tmp_idx = sorted_idx[i];
                sorted_idx[i] = sorted_idx[max_idx];
                sorted_idx[max_idx] = tmp_idx;
            }
        }
    }
    __syncthreads();

    // Phase 3: Compute cumulative sum
    if (tid == 0) {
        cumsum[0] = sorted_vals[0];
        #pragma unroll
        for (int i = 1; i < N_SLOTS; i++) {
            cumsum[i] = cumsum[i-1] + sorted_vals[i];
        }
    }
    __syncthreads();

    // Phase 4: Compute tau candidates for each support size k
    // For 1.5-entmax: tau_k = mean_k - sqrt(max(0, delta_k))
    // where delta_k = (1 - ss_k) / k, ss_k = scaled variance
    if (tid == 0) {
        #pragma unroll
        for (int k = 1; k <= N_SLOTS; k++) {
            float sum_k = cumsum[k-1];
            float mean_k = sum_k / k;

            // Compute sum of squared deviations
            float sum_sq = 0.0f;
            for (int i = 0; i < k; i++) {
                float diff = sorted_vals[i] - mean_k;
                sum_sq += diff * diff;
            }
            float ss_k = sum_sq / (float)(k * k);

            float delta_k = (1.0f - ss_k) / k;
            float tau_k = mean_k - sqrtf(fmaxf(0.0f, delta_k));
            tau_candidates[k-1] = tau_k;
        }

        // Find support size: k* = max{k : tau_k <= sorted_vals[k-1]}
        *support_size_ptr = N_SLOTS;
        #pragma unroll
        for (int k = 1; k <= N_SLOTS; k++) {
            if (tau_candidates[k-1] > sorted_vals[k-1]) {
                *support_size_ptr = k - 1;
                break;
            }
        }
        if (*support_size_ptr == 0) *support_size_ptr = 1;  // At least one element
        *tau_star_ptr = tau_candidates[*support_size_ptr - 1];
    }
    __syncthreads();

    float tau_star = *tau_star_ptr;

    // Phase 5: Apply transformation p = max(0, z - tau)^2
    // Use original unsorted scores
    if (tid < N_SLOTS) {
        float z = attn_sh[tid];
        float p = fmaxf(0.0f, z - tau_star);
        p = p * p;  // Square for 1.5-entmax
        attn_sh[tid] = p;
    }
    __syncthreads();

    // Phase 6: Normalize to sum to 1
    __shared__ float sum_p;
    if (tid == 0) {
        sum_p = 0.0f;
        #pragma unroll
        for (int i = 0; i < N_SLOTS; i++) {
            sum_p += attn_sh[i];
        }
        sum_p = fmaxf(sum_p, 1e-9f);  // Prevent division by zero
    }
    __syncthreads();

    if (tid < N_SLOTS) {
        attn_sh[tid] /= sum_p;
    }
}


/**
 * E25 Fused Kernel Phase 1: Read attention (entmax) + Update h_work
 *
 * Dynamic shared memory layout:
 *   - attn_sh[N_SLOTS]
 *   - h_work_sh[DIM]
 *   - sorted_vals[N_SLOTS]
 *   - sorted_idx[N_SLOTS] (as int)
 *   - cumsum[N_SLOTS]
 *   - tau_candidates[N_SLOTS]
 *   - support_size (int)
 *   - tau_star (float)
 */
template<int N_SLOTS>
__global__ void E25Phase1Kernel_BF16(
    const int batch_size,
    const int DIM,
    const __nv_bfloat16* __restrict__ Rh,
    const __nv_bfloat16* __restrict__ x_proj_t,
    const __nv_bfloat16* __restrict__ b_h,
    const __nv_bfloat16* __restrict__ h_tape,
    const __nv_bfloat16* __restrict__ h_work,
    __nv_bfloat16* __restrict__ h_work_out,
    __nv_bfloat16* __restrict__ read_attn_out,
    const float scale
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    extern __shared__ char shared_mem[];
    float* attn_sh = (float*)shared_mem;
    float* h_work_sh = attn_sh + N_SLOTS;
    float* sorted_vals = h_work_sh + DIM;
    int* sorted_idx = (int*)(sorted_vals + N_SLOTS);
    float* cumsum = (float*)(sorted_idx + N_SLOTS);
    float* tau_candidates = cumsum + N_SLOTS;
    int* support_size_ptr = (int*)(tau_candidates + N_SLOTS);
    float* tau_star_ptr = (float*)(support_size_ptr + 1);

    // Load h_work into shared memory
    for (int d = tid; d < DIM; d += E25_BLOCK_SIZE) {
        h_work_sh[d] = __bfloat162float(h_work[b * DIM + d]);
    }
    __syncthreads();

    // Compute attention scores: score[n] = h_tape[n] @ h_work * scale
    if (tid < N_SLOTS) {
        float score = 0.0f;
        const __nv_bfloat16* tape_n = h_tape + b * N_SLOTS * DIM + tid * DIM;
        for (int d = 0; d < DIM; d++) {
            score += __bfloat162float(tape_n[d]) * h_work_sh[d];
        }
        attn_sh[tid] = score * scale;
    }
    __syncthreads();

    // 1.5-Entmax instead of softmax
    entmax_1_5_device<N_SLOTS>(attn_sh, tid, sorted_vals, sorted_idx, cumsum, tau_candidates, support_size_ptr, tau_star_ptr);
    __syncthreads();

    // Store read attention
    if (tid < N_SLOTS) {
        read_attn_out[b * N_SLOTS + tid] = __float2bfloat16(attn_sh[tid]);
    }

    // Compute h_work_new: tanh(Rh + x_proj_t + read_val + b_h)
    for (int d = tid; d < DIM; d += E25_BLOCK_SIZE) {
        float read_d = 0.0f;
        #pragma unroll
        for (int n = 0; n < N_SLOTS; n++) {
            read_d += attn_sh[n] * __bfloat162float(h_tape[b * N_SLOTS * DIM + n * DIM + d]);
        }

        float val = __bfloat162float(Rh[b * DIM + d])
                  + __bfloat162float(x_proj_t[b * DIM + d])
                  + read_d
                  + __bfloat162float(b_h[d]);

        h_work_out[b * DIM + d] = __float2bfloat16(tanhf(val));
    }
}


/**
 * E25 Fused Kernel Phase 2: Write attention (entmax) + Update tape
 *
 * Dynamic shared memory layout:
 *   - attn_sh[N_SLOTS]
 *   - h_work_sh[DIM]
 *   - write_val_sh[DIM]
 *   - sorted_vals[N_SLOTS]
 *   - sorted_idx[N_SLOTS] (as int)
 *   - cumsum[N_SLOTS]
 *   - tau_candidates[N_SLOTS]
 *   - support_size (int)
 *   - tau_star (float)
 */
template<int N_SLOTS>
__global__ void E25Phase2Kernel_BF16(
    const int batch_size,
    const int DIM,
    const __nv_bfloat16* __restrict__ write_val,
    const __nv_bfloat16* __restrict__ h_work_new,
    __nv_bfloat16* __restrict__ h_tape,
    __nv_bfloat16* __restrict__ write_attn_out,
    const float scale
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    extern __shared__ char shared_mem[];
    float* attn_sh = (float*)shared_mem;
    float* h_work_sh = attn_sh + N_SLOTS;
    float* write_val_sh = h_work_sh + DIM;
    float* sorted_vals = write_val_sh + DIM;
    int* sorted_idx = (int*)(sorted_vals + N_SLOTS);
    float* cumsum = (float*)(sorted_idx + N_SLOTS);
    float* tau_candidates = cumsum + N_SLOTS;
    int* support_size_ptr = (int*)(tau_candidates + N_SLOTS);
    float* tau_star_ptr = (float*)(support_size_ptr + 1);

    // Load h_work_new and write_val into shared memory
    for (int d = tid; d < DIM; d += E25_BLOCK_SIZE) {
        h_work_sh[d] = __bfloat162float(h_work_new[b * DIM + d]);
        write_val_sh[d] = __bfloat162float(write_val[b * DIM + d]);
    }
    __syncthreads();

    // Compute write attention scores
    __nv_bfloat16* tape_b = h_tape + b * N_SLOTS * DIM;
    if (tid < N_SLOTS) {
        float score = 0.0f;
        for (int d = 0; d < DIM; d++) {
            score += __bfloat162float(tape_b[tid * DIM + d]) * h_work_sh[d];
        }
        attn_sh[tid] = score * scale;
    }
    __syncthreads();

    // 1.5-Entmax
    entmax_1_5_device<N_SLOTS>(attn_sh, tid, sorted_vals, sorted_idx, cumsum, tau_candidates, support_size_ptr, tau_star_ptr);
    __syncthreads();

    // Store write attention
    if (tid < N_SLOTS) {
        write_attn_out[b * N_SLOTS + tid] = __float2bfloat16(attn_sh[tid]);
    }

    // Update tape: h_tape = (1 - attn) * h_tape + attn * write_val
    for (int i = tid; i < N_SLOTS * DIM; i += E25_BLOCK_SIZE) {
        int n = i / DIM;
        int d = i % DIM;
        float attn_n = attn_sh[n];
        float old_val = __bfloat162float(tape_b[i]);
        float new_val = (1.0f - attn_n) * old_val + attn_n * write_val_sh[d];
        tape_b[i] = __float2bfloat16(new_val);
    }
}


/**
 * E25 Backward Phase 1: Write attention backward (with entmax gradient)
 *
 * Python reference:
 *   d_write_value = (d_h_tape * write_attn[:, :, None]).sum(dim=1)
 *   write_value = h_work_t @ W_write.T
 *   d_write_attn = (d_h_tape * (write_value - h_tape_t)).sum(dim=-1)
 *   d_h_tape_pre_write = d_h_tape * (1 - write_attn[:, :, None])
 *   d_write_scores = entmax_backward(write_attn, d_write_attn) * scale
 *   d_h_work_from_write_attn = (d_write_scores[:, :, None] * h_tape_t).sum(dim=1)
 *   d_h_tape_from_write_attn = d_write_scores[:, :, None] * h_work_t[:, None, :]
 *
 * Dynamic shared memory layout:
 *   - attn_sh[N_SLOTS]
 *   - g_sh[N_SLOTS]
 *   - d_write_attn_sh[N_SLOTS]
 *   - d_write_scores_sh[N_SLOTS]
 *   - h_work_sh[DIM]
 *   - write_val_sh[DIM]
 */
template<int N_SLOTS>
__global__ void E25BackwardPhase1_BF16(
    const int batch_size,
    const int DIM,
    const __nv_bfloat16* __restrict__ write_attn_t,
    const __nv_bfloat16* __restrict__ d_h_work_out_t,
    const __nv_bfloat16* __restrict__ h_tape_t,      // Added: tape state before write
    const __nv_bfloat16* __restrict__ h_work_t,      // Added: h_work at current timestep
    const __nv_bfloat16* __restrict__ write_val_t,   // Added: write_value = h_work @ W_write^T
    const float scale,                                // Added: 1/sqrt(D)
    __nv_bfloat16* __restrict__ d_h_work,
    __nv_bfloat16* __restrict__ d_h_tape,
    __nv_bfloat16* __restrict__ d_write_val_t
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    extern __shared__ char shared_mem[];
    float* attn_sh = (float*)shared_mem;
    float* g_sh = attn_sh + N_SLOTS;
    float* d_write_attn_sh = g_sh + N_SLOTS;
    float* d_write_scores_sh = d_write_attn_sh + N_SLOTS;
    float* h_work_sh = d_write_scores_sh + N_SLOTS;
    float* write_val_sh = h_work_sh + DIM;

    const __nv_bfloat16* h_tape_b = h_tape_t + b * N_SLOTS * DIM;
    __nv_bfloat16* d_h_tape_b = d_h_tape + b * N_SLOTS * DIM;

    // Load attention weights and compute g = sqrt(attn)
    if (tid < N_SLOTS) {
        float a = __bfloat162float(write_attn_t[b * N_SLOTS + tid]);
        attn_sh[tid] = a;
        g_sh[tid] = sqrtf(fmaxf(a, 0.0f));
    }

    // Load h_work and write_val
    for (int d = tid; d < DIM; d += E25_BLOCK_SIZE) {
        h_work_sh[d] = __bfloat162float(h_work_t[b * DIM + d]);
        write_val_sh[d] = __bfloat162float(write_val_t[b * DIM + d]);
    }
    __syncthreads();

    // Step 1: Accumulate d_h_work from upstream gradient
    // Step 2: Compute d_write_val = sum_n(attn[n] * d_h_tape[n])
    for (int d = tid; d < DIM; d += E25_BLOCK_SIZE) {
        float d_h = __bfloat162float(d_h_work[b * DIM + d])
                  + __bfloat162float(d_h_work_out_t[b * DIM + d]);
        d_h_work[b * DIM + d] = __float2bfloat16(d_h);

        float d_wv = 0.0f;
        #pragma unroll
        for (int n = 0; n < N_SLOTS; n++) {
            d_wv += attn_sh[n] * __bfloat162float(d_h_tape_b[n * DIM + d]);
        }
        d_write_val_t[b * DIM + d] = __float2bfloat16(d_wv);
    }
    __syncthreads();

    // Step 3: Compute d_write_attn = sum_d(d_h_tape[n,d] * (write_val[d] - h_tape[n,d]))
    if (tid < N_SLOTS) {
        float d_attn = 0.0f;
        for (int d = 0; d < DIM; d++) {
            float d_tape = __bfloat162float(d_h_tape_b[tid * DIM + d]);
            float wv = write_val_sh[d];
            float ht = __bfloat162float(h_tape_b[tid * DIM + d]);
            d_attn += d_tape * (wv - ht);
        }
        d_write_attn_sh[tid] = d_attn;
    }
    __syncthreads();

    // Step 4: Entmax backward for write attention
    // dz = g * (dp - (g @ dp) / sum(g)) * scale
    if (tid == 0) {
        float g_dp_sum = 0.0f;
        float g_sum = 0.0f;
        #pragma unroll
        for (int n = 0; n < N_SLOTS; n++) {
            g_dp_sum += g_sh[n] * d_write_attn_sh[n];
            g_sum += g_sh[n];
        }
        g_sum = fmaxf(g_sum, 1e-9f);

        #pragma unroll
        for (int n = 0; n < N_SLOTS; n++) {
            d_write_scores_sh[n] = g_sh[n] * (d_write_attn_sh[n] - g_dp_sum / g_sum) * scale;
        }
    }
    __syncthreads();

    // Step 5: Compute d_h_work_from_write_attn = sum_n(d_write_scores[n] * h_tape[n])
    for (int d = tid; d < DIM; d += E25_BLOCK_SIZE) {
        float d_work_contrib = 0.0f;
        #pragma unroll
        for (int n = 0; n < N_SLOTS; n++) {
            d_work_contrib += d_write_scores_sh[n] * __bfloat162float(h_tape_b[n * DIM + d]);
        }
        float current = __bfloat162float(d_h_work[b * DIM + d]);
        d_h_work[b * DIM + d] = __float2bfloat16(current + d_work_contrib);
    }
    __syncthreads();

    // Step 6: Update d_h_tape
    // d_h_tape = (1 - attn) * d_h_tape + d_write_scores * h_work
    for (int i = tid; i < N_SLOTS * DIM; i += E25_BLOCK_SIZE) {
        int n = i / DIM;
        int d = i % DIM;
        float d_tape = __bfloat162float(d_h_tape_b[i]);
        float d_tape_new = (1.0f - attn_sh[n]) * d_tape + d_write_scores_sh[n] * h_work_sh[d];
        d_h_tape_b[i] = __float2bfloat16(d_tape_new);
    }
}


/**
 * E25 Backward Phase 2: Tanh backward
 *
 * No shared memory needed for DIM arrays - uses simple per-element computation.
 */
template<int N_SLOTS>
__global__ void E25BackwardPhase2_BF16(
    const int batch_size,
    const int DIM,
    const __nv_bfloat16* __restrict__ h_work_t,
    const __nv_bfloat16* __restrict__ d_h_work,
    __nv_bfloat16* __restrict__ dx_proj_t,
    __nv_bfloat16* __restrict__ d_pre_act_t,
    float* __restrict__ db_h
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    for (int d = tid; d < DIM; d += E25_BLOCK_SIZE) {
        float h = __bfloat162float(h_work_t[b * DIM + d]);
        float d_h = __bfloat162float(d_h_work[b * DIM + d]);
        float d_pre_act = d_h * (1.0f - h * h);

        dx_proj_t[b * DIM + d] = __float2bfloat16(d_pre_act);
        d_pre_act_t[b * DIM + d] = __float2bfloat16(d_pre_act);

        atomicAdd(&db_h[d], d_pre_act);
    }
}


/**
 * E25 Backward Phase 3: Read attention backward (entmax gradient)
 *
 * Entmax backward: dz = g * (dp - (g @ dp) / sum(g)) where g = sqrt(p)
 *
 * Dynamic shared memory layout:
 *   - attn_sh[N_SLOTS]
 *   - g_sh[N_SLOTS]
 *   - d_pre_act_sh[DIM]
 *   - h_work_prev_sh[DIM]
 *   - d_read_attn_sh[N_SLOTS]
 *   - d_read_scores_sh[N_SLOTS]
 */
template<int N_SLOTS>
__global__ void E25BackwardPhase3_BF16(
    const int batch_size,
    const int DIM,
    const __nv_bfloat16* __restrict__ read_attn_t,
    const __nv_bfloat16* __restrict__ d_pre_act_t,
    const __nv_bfloat16* __restrict__ h_tape_t,
    const __nv_bfloat16* __restrict__ h_work_prev_t,
    const float scale,
    __nv_bfloat16* __restrict__ d_h_tape,
    __nv_bfloat16* __restrict__ d_h_work
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    extern __shared__ char shared_mem[];
    float* attn_sh = (float*)shared_mem;
    float* g_sh = attn_sh + N_SLOTS;
    float* d_pre_act_sh = g_sh + N_SLOTS;
    float* h_work_prev_sh = d_pre_act_sh + DIM;
    float* d_read_attn_sh = h_work_prev_sh + DIM;
    float* d_read_scores_sh = d_read_attn_sh + N_SLOTS;

    const __nv_bfloat16* h_tape_b = h_tape_t + b * N_SLOTS * DIM;
    __nv_bfloat16* d_h_tape_b = d_h_tape + b * N_SLOTS * DIM;

    // Load attention and compute g = sqrt(attn)
    if (tid < N_SLOTS) {
        float a = __bfloat162float(read_attn_t[b * N_SLOTS + tid]);
        attn_sh[tid] = a;
        g_sh[tid] = sqrtf(fmaxf(a, 0.0f));
    }

    for (int d = tid; d < DIM; d += E25_BLOCK_SIZE) {
        d_pre_act_sh[d] = __bfloat162float(d_pre_act_t[b * DIM + d]);
        h_work_prev_sh[d] = __bfloat162float(h_work_prev_t[b * DIM + d]);
    }
    __syncthreads();

    // Step 1: Compute d_read_attn = (d_pre_act @ h_tape^T) per slot
    if (tid < N_SLOTS) {
        float d_attn = 0.0f;
        for (int d = 0; d < DIM; d++) {
            d_attn += d_pre_act_sh[d] * __bfloat162float(h_tape_b[tid * DIM + d]);
        }
        d_read_attn_sh[tid] = d_attn;
    }
    __syncthreads();

    // Step 2: Entmax backward
    // dz = g * (dp - (g @ dp) / sum(g))
    if (tid == 0) {
        float g_dp_sum = 0.0f;
        float g_sum = 0.0f;
        #pragma unroll
        for (int n = 0; n < N_SLOTS; n++) {
            g_dp_sum += g_sh[n] * d_read_attn_sh[n];
            g_sum += g_sh[n];
        }
        g_sum = fmaxf(g_sum, 1e-9f);

        #pragma unroll
        for (int n = 0; n < N_SLOTS; n++) {
            d_read_scores_sh[n] = g_sh[n] * (d_read_attn_sh[n] - g_dp_sum / g_sum) * scale;
        }
    }
    __syncthreads();

    // Step 3: Update d_h_tape
    for (int i = tid; i < N_SLOTS * DIM; i += E25_BLOCK_SIZE) {
        int n = i / DIM;
        int d = i % DIM;

        float d_tape_direct = attn_sh[n] * d_pre_act_sh[d];
        float d_tape_scores = d_read_scores_sh[n] * h_work_prev_sh[d];

        float current = __bfloat162float(d_h_tape_b[i]);
        d_h_tape_b[i] = __float2bfloat16(current + d_tape_direct + d_tape_scores);
    }

    // Step 4: Update d_h_work
    for (int d = tid; d < DIM; d += E25_BLOCK_SIZE) {
        float d_work_contrib = 0.0f;
        #pragma unroll
        for (int n = 0; n < N_SLOTS; n++) {
            d_work_contrib += d_read_scores_sh[n] * __bfloat162float(h_tape_b[n * DIM + d]);
        }
        float current = __bfloat162float(d_h_work[b * DIM + d]);
        d_h_work[b * DIM + d] = __float2bfloat16(current + d_work_contrib);
    }
}


/**
 * Initialize backward pass
 */
template<int N_SLOTS>
__global__ void E25BackwardInit_BF16(
    const int batch_size,
    const int DIM,
    const __nv_bfloat16* __restrict__ d_h_tape_final,
    __nv_bfloat16* __restrict__ d_h_tape,
    __nv_bfloat16* __restrict__ d_h_work
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    for (int i = tid; i < N_SLOTS * DIM; i += E25_BLOCK_SIZE) {
        d_h_tape[b * N_SLOTS * DIM + i] = d_h_tape_final[b * N_SLOTS * DIM + i];
    }

    for (int d = tid; d < DIM; d += E25_BLOCK_SIZE) {
        d_h_work[b * DIM + d] = __float2bfloat16(0.0f);
    }
}


// Helper functions to compute shared memory sizes
inline size_t phase1_shared_size(int n_slots, int dim) {
    // attn_sh[N_SLOTS] + h_work_sh[DIM] + sorted_vals[N_SLOTS] + sorted_idx[N_SLOTS] +
    // cumsum[N_SLOTS] + tau_candidates[N_SLOTS] + support_size (int) + tau_star (float)
    return (n_slots + dim + n_slots + n_slots + n_slots) * sizeof(float) +
           n_slots * sizeof(int) +
           sizeof(int) + sizeof(float);
}

inline size_t phase2_shared_size(int n_slots, int dim) {
    // attn_sh[N_SLOTS] + h_work_sh[DIM] + write_val_sh[DIM] + sorted_vals[N_SLOTS] +
    // sorted_idx[N_SLOTS] + cumsum[N_SLOTS] + tau_candidates[N_SLOTS] + support_size (int) + tau_star (float)
    return (n_slots + dim + dim + n_slots + n_slots + n_slots) * sizeof(float) +
           n_slots * sizeof(int) +
           sizeof(int) + sizeof(float);
}

inline size_t bwd_phase1_shared_size(int n_slots, int dim) {
    // attn_sh[N_SLOTS] + g_sh[N_SLOTS] + d_write_attn_sh[N_SLOTS] + d_write_scores_sh[N_SLOTS] +
    // h_work_sh[DIM] + write_val_sh[DIM]
    return (n_slots * 4 + dim * 2) * sizeof(float);
}

inline size_t bwd_phase3_shared_size(int n_slots, int dim) {
    // attn_sh[N_SLOTS] + g_sh[N_SLOTS] + d_pre_act_sh[DIM] + h_work_prev_sh[DIM] +
    // d_read_attn_sh[N_SLOTS] + d_read_scores_sh[N_SLOTS]
    return (n_slots * 4 + dim * 2) * sizeof(float);
}


}  // namespace


namespace hasty { namespace v0 { namespace elman_ladder {

// =============================================================================
// E25 Forward
// =============================================================================

template<typename T>
E25EntmaxForward<T>::E25EntmaxForward(
    bool training, int batch_size, int n_slots, int dim,
    const cublasHandle_t& blas_handle, const cudaStream_t& stream)
    : training_(training), batch_size_(batch_size), n_slots_(n_slots), dim_(dim),
      stream_(stream), blas_handle_(blas_handle) {}

template<>
void E25EntmaxForward<__nv_bfloat16>::Run(
    int seq_len,
    const __nv_bfloat16* x_proj,
    const __nv_bfloat16* W_h,
    const __nv_bfloat16* b_h,
    const __nv_bfloat16* W_write,
    const __nv_bfloat16* h_tape_init,
    const __nv_bfloat16* h_work_init,
    __nv_bfloat16* h_work_out,
    __nv_bfloat16* h_tape_final,
    __nv_bfloat16* h_tape_all,
    __nv_bfloat16* read_attn,
    __nv_bfloat16* write_attn,
    __nv_bfloat16* workspace
) {
    seq_len_ = seq_len;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    const int BD = batch_size_ * dim_;

    __nv_bfloat16* tmp_Rh = workspace;
    __nv_bfloat16* tmp_write_val = tmp_Rh + BD;

    cudaMemcpyAsync(h_tape_final, h_tape_init,
                    batch_size_ * n_slots_ * dim_ * sizeof(__nv_bfloat16),
                    cudaMemcpyDeviceToDevice, stream_);

    if (training_ && h_tape_all) {
        cudaMemcpyAsync(h_tape_all, h_tape_init,
                        batch_size_ * n_slots_ * dim_ * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToDevice, stream_);
    }

    const float scale = 1.0f / sqrtf(static_cast<float>(dim_));
    const int num_blocks = batch_size_;

    // Calculate shared memory sizes
    const size_t p1_smem = phase1_shared_size(n_slots_, dim_);
    const size_t p2_smem = phase2_shared_size(n_slots_, dim_);

    #define LAUNCH_E25_PHASE1(N) \
        E25Phase1Kernel_BF16<N><<<num_blocks, E25_BLOCK_SIZE, p1_smem, stream_>>>( \
            batch_size_, dim_, tmp_Rh, x_proj_t, b_h, h_tape_final, h_work_prev, \
            h_work_cur, read_attn_t, scale)

    #define LAUNCH_E25_PHASE2(N) \
        E25Phase2Kernel_BF16<N><<<num_blocks, E25_BLOCK_SIZE, p2_smem, stream_>>>( \
            batch_size_, dim_, tmp_write_val, h_work_cur, h_tape_final, \
            write_attn_t, scale)

    for (int t = 0; t < seq_len; ++t) {
        const __nv_bfloat16* x_proj_t = x_proj + t * BD;
        const __nv_bfloat16* h_work_prev = (t == 0) ? h_work_init : (h_work_out + (t - 1) * BD);
        __nv_bfloat16* h_work_cur = h_work_out + t * BD;
        __nv_bfloat16* read_attn_t = read_attn + t * batch_size_ * n_slots_;
        __nv_bfloat16* write_attn_t = write_attn + t * batch_size_ * n_slots_;

        // W_h @ h_work_prev
        cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha,
            W_h, CUDA_R_16BF, dim_, h_work_prev, CUDA_R_16BF, dim_,
            &beta, tmp_Rh, CUDA_R_16BF, dim_,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // Phase 1: read entmax + h_work update
        if (n_slots_ == 8) { LAUNCH_E25_PHASE1(8); }
        else if (n_slots_ == 16) { LAUNCH_E25_PHASE1(16); }
        else if (n_slots_ == 32) { LAUNCH_E25_PHASE1(32); }
        else if (n_slots_ == 64) { LAUNCH_E25_PHASE1(64); }
        else { fprintf(stderr, "E25 CUDA: unsupported n_slots=%d\n", n_slots_); }

        // W_write @ h_work_new
        cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha,
            W_write, CUDA_R_16BF, dim_, h_work_cur, CUDA_R_16BF, dim_,
            &beta, tmp_write_val, CUDA_R_16BF, dim_,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // Phase 2: write entmax + tape update
        if (n_slots_ == 8) { LAUNCH_E25_PHASE2(8); }
        else if (n_slots_ == 16) { LAUNCH_E25_PHASE2(16); }
        else if (n_slots_ == 32) { LAUNCH_E25_PHASE2(32); }
        else if (n_slots_ == 64) { LAUNCH_E25_PHASE2(64); }

        if (training_ && h_tape_all) {
            cudaMemcpyAsync(h_tape_all + (t + 1) * batch_size_ * n_slots_ * dim_,
                h_tape_final, batch_size_ * n_slots_ * dim_ * sizeof(__nv_bfloat16),
                cudaMemcpyDeviceToDevice, stream_);
        }
    }

    #undef LAUNCH_E25_PHASE1
    #undef LAUNCH_E25_PHASE2
}


// =============================================================================
// E25 Backward
// =============================================================================

template<typename T>
E25EntmaxBackward<T>::E25EntmaxBackward(
    int batch_size, int n_slots, int dim,
    const cublasHandle_t& blas_handle, const cudaStream_t& stream)
    : batch_size_(batch_size), n_slots_(n_slots), dim_(dim),
      stream_(stream), blas_handle_(blas_handle) {}

template<>
void E25EntmaxBackward<__nv_bfloat16>::Run(
    int seq_len,
    const __nv_bfloat16* h_work_all,
    const __nv_bfloat16* h_work_init,
    const __nv_bfloat16* h_tape_all,
    const __nv_bfloat16* read_attn,
    const __nv_bfloat16* write_attn,
    const __nv_bfloat16* W_h,
    const __nv_bfloat16* W_write,
    const __nv_bfloat16* d_h_work_out,
    const __nv_bfloat16* d_h_tape_final,
    __nv_bfloat16* dx_proj,
    __nv_bfloat16* d_pre_act_all,
    __nv_bfloat16* d_write_val_all,
    float* db_h,
    __nv_bfloat16* d_h_tape,
    float* dW_h,
    float* dW_write
) {
    const int num_blocks = batch_size_;
    const int TB = seq_len * batch_size_;
    const int BD = batch_size_ * dim_;
    const int BN = batch_size_ * n_slots_;
    const float alpha_one = 1.0f;
    const float beta_zero = 0.0f;
    const float beta_one = 1.0f;

    __nv_bfloat16* d_h_work;
    __nv_bfloat16* write_val_t_buf;  // Buffer for write_value = h_work @ W_write^T
    cudaMalloc(&d_h_work, BD * sizeof(__nv_bfloat16));
    cudaMalloc(&write_val_t_buf, BD * sizeof(__nv_bfloat16));

    // Calculate shared memory sizes
    const size_t bp1_smem = bwd_phase1_shared_size(n_slots_, dim_);
    const size_t bp3_smem = bwd_phase3_shared_size(n_slots_, dim_);

    #define LAUNCH_E25_BWD_INIT(N) \
        E25BackwardInit_BF16<N><<<num_blocks, E25_BLOCK_SIZE, 0, stream_>>>( \
            batch_size_, dim_, d_h_tape_final, d_h_tape, d_h_work)

    #define LAUNCH_E25_BWD_PHASE1(N) \
        E25BackwardPhase1_BF16<N><<<num_blocks, E25_BLOCK_SIZE, bp1_smem, stream_>>>( \
            batch_size_, dim_, write_attn_t, d_h_work_out_t, h_tape_t_ptr, h_work_t, write_val_t_buf, scale, \
            d_h_work, d_h_tape, d_write_val_t)

    #define LAUNCH_E25_BWD_PHASE2(N) \
        E25BackwardPhase2_BF16<N><<<num_blocks, E25_BLOCK_SIZE, 0, stream_>>>( \
            batch_size_, dim_, h_work_t, d_h_work, dx_proj_t, d_pre_act_t, db_h)

    #define LAUNCH_E25_BWD_PHASE3(N) \
        E25BackwardPhase3_BF16<N><<<num_blocks, E25_BLOCK_SIZE, bp3_smem, stream_>>>( \
            batch_size_, dim_, read_attn_t, d_pre_act_t, h_tape_t_ptr, h_work_prev_t, scale, d_h_tape, d_h_work)

    // Initialize
    if (n_slots_ == 8) { LAUNCH_E25_BWD_INIT(8); }
    else if (n_slots_ == 16) { LAUNCH_E25_BWD_INIT(16); }
    else if (n_slots_ == 32) { LAUNCH_E25_BWD_INIT(32); }
    else if (n_slots_ == 64) { LAUNCH_E25_BWD_INIT(64); }
    else {
        fprintf(stderr, "E25 CUDA backward: unsupported n_slots=%d\n", n_slots_);
        cudaFree(d_h_work);
        cudaFree(write_val_t_buf);
        return;
    }

    const float scale = 1.0f / sqrtf(static_cast<float>(dim_));
    const int BND = batch_size_ * n_slots_ * dim_;

    for (int t = seq_len - 1; t >= 0; t--) {
        const __nv_bfloat16* write_attn_t = write_attn + t * BN;
        const __nv_bfloat16* read_attn_t = read_attn + t * BN;
        const __nv_bfloat16* d_h_work_out_t = d_h_work_out + t * BD;
        const __nv_bfloat16* h_work_t = h_work_all + t * BD;
        __nv_bfloat16* dx_proj_t = dx_proj + t * BD;
        __nv_bfloat16* d_pre_act_t = d_pre_act_all + t * BD;
        __nv_bfloat16* d_write_val_t = d_write_val_all + t * BD;

        const __nv_bfloat16* h_tape_t_ptr = h_tape_all + t * BND;
        const __nv_bfloat16* h_work_prev_t = (t > 0) ? (h_work_all + (t - 1) * BD) : h_work_init;

        // Compute write_val_t = h_work_t @ W_write^T (needed for write attention backward)
        cublasGemmEx(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha_one,
            W_write, CUDA_R_16BF, dim_,
            h_work_t, CUDA_R_16BF, dim_,
            &beta_zero,
            write_val_t_buf, CUDA_R_16BF, dim_,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
        );

        // Phase 1
        if (n_slots_ == 8) { LAUNCH_E25_BWD_PHASE1(8); }
        else if (n_slots_ == 16) { LAUNCH_E25_BWD_PHASE1(16); }
        else if (n_slots_ == 32) { LAUNCH_E25_BWD_PHASE1(32); }
        else if (n_slots_ == 64) { LAUNCH_E25_BWD_PHASE1(64); }

        // cuBLAS: d_h_work += W_write.T @ d_write_val
        cublasGemmEx(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha_one,
            W_write, CUDA_R_16BF, dim_,
            d_write_val_t, CUDA_R_16BF, dim_,
            &beta_one,
            d_h_work, CUDA_R_16BF, dim_,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
        );

        // Phase 2
        if (n_slots_ == 8) { LAUNCH_E25_BWD_PHASE2(8); }
        else if (n_slots_ == 16) { LAUNCH_E25_BWD_PHASE2(16); }
        else if (n_slots_ == 32) { LAUNCH_E25_BWD_PHASE2(32); }
        else if (n_slots_ == 64) { LAUNCH_E25_BWD_PHASE2(64); }

        // cuBLAS: d_h_work = W_h.T @ d_pre_act
        cublasGemmEx(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha_one,
            W_h, CUDA_R_16BF, dim_,
            d_pre_act_t, CUDA_R_16BF, dim_,
            &beta_zero,
            d_h_work, CUDA_R_16BF, dim_,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
        );

        // Phase 3
        if (n_slots_ == 8) { LAUNCH_E25_BWD_PHASE3(8); }
        else if (n_slots_ == 16) { LAUNCH_E25_BWD_PHASE3(16); }
        else if (n_slots_ == 32) { LAUNCH_E25_BWD_PHASE3(32); }
        else if (n_slots_ == 64) { LAUNCH_E25_BWD_PHASE3(64); }
    }

    #undef LAUNCH_E25_BWD_INIT
    #undef LAUNCH_E25_BWD_PHASE1
    #undef LAUNCH_E25_BWD_PHASE2
    #undef LAUNCH_E25_BWD_PHASE3

    cudaFree(d_h_work);
    cudaFree(write_val_t_buf);

    // Weight gradients
    cublasGemmEx(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, TB,
        &alpha_one,
        d_write_val_all, CUDA_R_16BF, dim_,
        h_work_all, CUDA_R_16BF, dim_,
        &beta_zero,
        dW_write, CUDA_R_32F, dim_,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
    );

    // dW_h computation:
    // For t=0: dW_h += d_pre_act[0]^T @ h_work_init
    // For t=1..T-1: dW_h += d_pre_act[t]^T @ h_work[t-1]

    // First, compute contribution from t=0 (uses h_work_init)
    cublasGemmEx(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, batch_size_,
        &alpha_one,
        d_pre_act_all, CUDA_R_16BF, dim_,       // d_pre_act[0]
        h_work_init, CUDA_R_16BF, dim_,         // h_work_init
        &beta_zero,                              // Start fresh
        dW_h, CUDA_R_32F, dim_,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
    );

    // Then accumulate contributions from t=1..T-1
    const __nv_bfloat16* h_work_prev = h_work_all;
    const __nv_bfloat16* d_pre_act_next = d_pre_act_all + BD;
    const int TB_minus_1 = (seq_len - 1) * batch_size_;

    if (TB_minus_1 > 0) {
        cublasGemmEx(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, TB_minus_1,
            &alpha_one,
            d_pre_act_next, CUDA_R_16BF, dim_,
            h_work_prev, CUDA_R_16BF, dim_,
            &beta_one,                           // Accumulate (not reset)
            dW_h, CUDA_R_32F, dim_,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
        );
    }
}

// Explicit instantiations
template class E25EntmaxForward<__nv_bfloat16>;
template class E25EntmaxBackward<__nv_bfloat16>;

}}}  // namespace hasty::v0::elman_ladder
