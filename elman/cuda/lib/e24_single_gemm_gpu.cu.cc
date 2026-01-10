/**
 * E24 Single-GEMM Dual Memory CUDA Kernels
 *
 * Architecture:
 *   - Tape: [B, N, D] - Large linear storage (N slots)
 *   - Working Memory: [B, D] - Small nonlinear compute
 *
 * Per timestep (1 GEMM!):
 *   0. SINGLE GEMM: [h_work; x] @ W_all.T -> [h_update; write_val]
 *   1. Read: h_work queries tape via attention → read value
 *   2. Update: h_work_new = tanh(h_update + read + b)
 *   3. Write: h_tape_new = (1-attn)*h_tape + attn*write_val
 *
 * Key insight: Concatenate [h_work; x] and use single [2D, 2D] GEMM
 * to produce both h_update and write_val in one operation.
 */

#include "hasty/elman_ladder.h"
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

namespace {

constexpr int E24_BLOCK_SIZE = 256;

/**
 * E24 Concatenation Kernel: Concatenate h_work and x for GEMM input
 *
 * input_concat = [h_work; x] : [B, 2D]
 */
template<int DIM>
__global__ void E24ConcatKernel_BF16(
    const int batch_size,
    const __nv_bfloat16* __restrict__ h_work,  // [B, D]
    const __nv_bfloat16* __restrict__ x_t,     // [B, D]
    __nv_bfloat16* __restrict__ input_concat   // [B, 2D]
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    // Copy h_work to first D elements
    for (int d = tid; d < DIM; d += E24_BLOCK_SIZE) {
        input_concat[b * 2 * DIM + d] = h_work[b * DIM + d];
    }

    // Copy x to last D elements
    for (int d = tid; d < DIM; d += E24_BLOCK_SIZE) {
        input_concat[b * 2 * DIM + DIM + d] = x_t[b * DIM + d];
    }
}

/**
 * E24 Fused Kernel: Read attention + h_work update + Write attention + Tape update
 *
 * Inputs (from cuBLAS):
 *   - gemm_output: [B, 2D] - result of [h_work; x] @ W_all.T
 *
 * Operations:
 *   1. Split: h_update = output[:, :D], write_val = output[:, D:]
 *   2. Read attention: attn = softmax(h_tape @ h_work * scale)
 *   3. Read value: read = attn @ h_tape
 *   4. Update: h_work_new = tanh(h_update + read + b)
 *   5. Write attention: attn = softmax(h_tape @ h_work_new * scale)
 *   6. Update tape: h_tape = (1-attn) * h_tape + attn * write_val
 */
template<int N_SLOTS, int DIM>
__global__ void E24FusedKernel_BF16(
    const int batch_size,
    // From cuBLAS
    const __nv_bfloat16* __restrict__ gemm_output,  // [B, 2D] - [h_update; write_val]
    const __nv_bfloat16* __restrict__ b_h,          // [D]
    // State
    const __nv_bfloat16* __restrict__ h_work_prev,  // [B, D] - previous h_work (for read attention)
    __nv_bfloat16* __restrict__ h_tape,             // [B, N, D] - modified in place
    // Outputs
    __nv_bfloat16* __restrict__ h_work_out,         // [B, D]
    __nv_bfloat16* __restrict__ read_attn_out,      // [B, N]
    __nv_bfloat16* __restrict__ write_attn_out,     // [B, N]
    __nv_bfloat16* __restrict__ write_val_out,      // [B, D] - save for backward
    const float scale
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    // Shared memory
    __shared__ float h_work_prev_sh[DIM];
    __shared__ float attn_sh[N_SLOTS];
    __shared__ float h_update_sh[DIM];
    __shared__ float write_val_sh[DIM];
    __shared__ float h_work_new_sh[DIM];

    __nv_bfloat16* tape_b = h_tape + b * N_SLOTS * DIM;

    // Load h_work_prev into shared memory
    for (int d = tid; d < DIM; d += E24_BLOCK_SIZE) {
        h_work_prev_sh[d] = __bfloat162float(h_work_prev[b * DIM + d]);
    }

    // Load and split GEMM output: h_update = output[:D], write_val = output[D:]
    for (int d = tid; d < DIM; d += E24_BLOCK_SIZE) {
        h_update_sh[d] = __bfloat162float(gemm_output[b * 2 * DIM + d]);
        float wv = __bfloat162float(gemm_output[b * 2 * DIM + DIM + d]);
        write_val_sh[d] = wv;
        write_val_out[b * DIM + d] = __float2bfloat16(wv);  // Save for backward
    }
    __syncthreads();

    // ============================================
    // PHASE 1: Read attention
    // ============================================

    // Compute read attention scores: score[n] = h_tape[n] @ h_work_prev
    if (tid < N_SLOTS) {
        float score = 0.0f;
        for (int d = 0; d < DIM; d++) {
            score += __bfloat162float(tape_b[tid * DIM + d]) * h_work_prev_sh[d];
        }
        attn_sh[tid] = score * scale;
    }
    __syncthreads();

    // Softmax over N_SLOTS (single thread, N is small)
    if (tid == 0) {
        float max_score = attn_sh[0];
        for (int n = 1; n < N_SLOTS; n++) {
            max_score = fmaxf(max_score, attn_sh[n]);
        }
        float sum_exp = 0.0f;
        for (int n = 0; n < N_SLOTS; n++) {
            attn_sh[n] = expf(attn_sh[n] - max_score);
            sum_exp += attn_sh[n];
        }
        for (int n = 0; n < N_SLOTS; n++) {
            attn_sh[n] /= sum_exp;
        }
    }
    __syncthreads();

    // Store read attention
    if (tid < N_SLOTS) {
        read_attn_out[b * N_SLOTS + tid] = __float2bfloat16(attn_sh[tid]);
    }

    // ============================================
    // PHASE 2: h_work update
    // h_work_new = tanh(h_update + read + b_h)
    // ============================================

    for (int d = tid; d < DIM; d += E24_BLOCK_SIZE) {
        // Weighted read: read[d] = sum_n attn[n] * h_tape[n, d]
        float read_d = 0.0f;
        for (int n = 0; n < N_SLOTS; n++) {
            read_d += attn_sh[n] * __bfloat162float(tape_b[n * DIM + d]);
        }

        // Combine: h_update + read + b
        float val = h_update_sh[d] + read_d + __bfloat162float(b_h[d]);
        float h_new = tanhf(val);
        h_work_new_sh[d] = h_new;
        h_work_out[b * DIM + d] = __float2bfloat16(h_new);
    }
    __syncthreads();

    // ============================================
    // PHASE 3: Write attention
    // ============================================

    // Compute write attention scores: score[n] = h_tape[n] @ h_work_new
    if (tid < N_SLOTS) {
        float score = 0.0f;
        for (int d = 0; d < DIM; d++) {
            score += __bfloat162float(tape_b[tid * DIM + d]) * h_work_new_sh[d];
        }
        attn_sh[tid] = score * scale;
    }
    __syncthreads();

    // Softmax
    if (tid == 0) {
        float max_score = attn_sh[0];
        for (int n = 1; n < N_SLOTS; n++) {
            max_score = fmaxf(max_score, attn_sh[n]);
        }
        float sum_exp = 0.0f;
        for (int n = 0; n < N_SLOTS; n++) {
            attn_sh[n] = expf(attn_sh[n] - max_score);
            sum_exp += attn_sh[n];
        }
        for (int n = 0; n < N_SLOTS; n++) {
            attn_sh[n] /= sum_exp;
        }
    }
    __syncthreads();

    // Store write attention
    if (tid < N_SLOTS) {
        write_attn_out[b * N_SLOTS + tid] = __float2bfloat16(attn_sh[tid]);
    }

    // ============================================
    // PHASE 4: Update tape
    // h_tape = (1 - attn) * h_tape + attn * write_val
    // ============================================

    for (int i = tid; i < N_SLOTS * DIM; i += E24_BLOCK_SIZE) {
        int n = i / DIM;
        int d = i % DIM;
        float attn_n = attn_sh[n];
        float old_val = __bfloat162float(tape_b[i]);
        float new_val = (1.0f - attn_n) * old_val + attn_n * write_val_sh[d];
        tape_b[i] = __float2bfloat16(new_val);
    }
}


// =============================================================================
// E24 Backward Kernels
// =============================================================================

/**
 * E24 Backward Init: Initialize d_h_tape from final gradient and zero d_h_work
 */
template<int N_SLOTS, int DIM>
__global__ void E24BackwardInit_BF16(
    const int batch_size,
    const __nv_bfloat16* __restrict__ d_h_tape_final, // [B, N, D]
    __nv_bfloat16* __restrict__ d_h_tape,             // [B, N, D]
    __nv_bfloat16* __restrict__ d_h_work              // [B, D]
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    // Copy d_h_tape from final gradient
    for (int i = tid; i < N_SLOTS * DIM; i += E24_BLOCK_SIZE) {
        d_h_tape[b * N_SLOTS * DIM + i] = d_h_tape_final[b * N_SLOTS * DIM + i];
    }

    // Zero d_h_work
    for (int d = tid; d < DIM; d += E24_BLOCK_SIZE) {
        d_h_work[b * DIM + d] = __float2bfloat16(0.0f);
    }
}

/**
 * E24 Backward Phase 1: Write attention backward
 *
 * Computes d_write_val and d_h_work_t_total (for tanh backward).
 * Updates d_h_tape for write operation.
 *
 * Key change: d_h_work_t_total is a SEPARATE output, not accumulated into d_h_work.
 * d_h_work is only used as input (accumulated gradient from later timesteps).
 */
template<int N_SLOTS, int DIM>
__global__ void E24BackwardPhase1_BF16(
    const int batch_size,
    const __nv_bfloat16* __restrict__ write_attn_t,   // [B, N]
    const __nv_bfloat16* __restrict__ write_val_t,    // [B, D]
    const __nv_bfloat16* __restrict__ h_tape_t,       // [B, N, D] - tape BEFORE update
    const __nv_bfloat16* __restrict__ h_work_t,       // [B, D] - h_work_new at timestep t
    const __nv_bfloat16* __restrict__ d_h_work_out_t, // [B, D] - output gradient for this timestep
    const __nv_bfloat16* __restrict__ d_h_work_accum, // [B, D] - accumulated gradient from later timesteps (INPUT ONLY)
    const float scale,
    __nv_bfloat16* __restrict__ d_h_work_t_total,     // [B, D] - OUTPUT: total gradient for tanh backward
    __nv_bfloat16* __restrict__ d_h_tape,             // [B, N, D] - tape gradient (modified)
    __nv_bfloat16* __restrict__ d_write_val_t         // [B, D] - output: d_write_val for GEMM backward
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    __shared__ float attn_sh[N_SLOTS];
    __shared__ float write_val_sh[DIM];
    __shared__ float h_work_sh[DIM];
    __shared__ float d_write_attn_sh[N_SLOTS];
    __shared__ float d_write_scores_sh[N_SLOTS];

    const __nv_bfloat16* h_tape_b = h_tape_t + b * N_SLOTS * DIM;
    __nv_bfloat16* d_h_tape_b = d_h_tape + b * N_SLOTS * DIM;

    // Load write attention and write_val into shared memory
    if (tid < N_SLOTS) {
        attn_sh[tid] = __bfloat162float(write_attn_t[b * N_SLOTS + tid]);
    }
    for (int d = tid; d < DIM; d += E24_BLOCK_SIZE) {
        write_val_sh[d] = __bfloat162float(write_val_t[b * DIM + d]);
        h_work_sh[d] = __bfloat162float(h_work_t[b * DIM + d]);
    }
    __syncthreads();

    // Compute d_write_val and initial d_h_work_t
    __shared__ float d_h_work_t_sh[DIM];
    for (int d = tid; d < DIM; d += E24_BLOCK_SIZE) {
        // Combine incoming gradients: accumulated + output gradient
        float d_h = __bfloat162float(d_h_work_accum[b * DIM + d])
                  + __bfloat162float(d_h_work_out_t[b * DIM + d]);
        d_h_work_t_sh[d] = d_h;

        // d_write_val = sum_n(attn[n] * d_h_tape[n])
        float d_wv = 0.0f;
        for (int n = 0; n < N_SLOTS; n++) {
            d_wv += attn_sh[n] * __bfloat162float(d_h_tape_b[n * DIM + d]);
        }
        d_write_val_t[b * DIM + d] = __float2bfloat16(d_wv);
    }
    __syncthreads();

    // Compute d_write_attn = (d_h_tape * (write_val - h_tape)).sum(dim=-1)
    if (tid < N_SLOTS) {
        float d_attn = 0.0f;
        for (int d = 0; d < DIM; d++) {
            float d_tape = __bfloat162float(d_h_tape_b[tid * DIM + d]);
            float tape_val = __bfloat162float(h_tape_b[tid * DIM + d]);
            d_attn += d_tape * (write_val_sh[d] - tape_val);
        }
        d_write_attn_sh[tid] = d_attn;
    }
    __syncthreads();

    // Softmax backward for write attention
    if (tid == 0) {
        float dot_sum = 0.0f;
        for (int n = 0; n < N_SLOTS; n++) {
            dot_sum += d_write_attn_sh[n] * attn_sh[n];
        }
        for (int n = 0; n < N_SLOTS; n++) {
            d_write_scores_sh[n] = attn_sh[n] * (d_write_attn_sh[n] - dot_sum) * scale;
        }
    }
    __syncthreads();

    // Add gradient from write attention to d_h_work_t and output total
    for (int d = tid; d < DIM; d += E24_BLOCK_SIZE) {
        float d_work_from_attn = 0.0f;
        for (int n = 0; n < N_SLOTS; n++) {
            d_work_from_attn += d_write_scores_sh[n] * __bfloat162float(h_tape_b[n * DIM + d]);
        }
        // d_h_work_t_total = d_h_work_t + d_work_from_write_attn
        d_h_work_t_total[b * DIM + d] = __float2bfloat16(d_h_work_t_sh[d] + d_work_from_attn);
    }

    // Update d_h_tape: (1-attn) * d_h_tape + d_write_scores * h_work
    for (int i = tid; i < N_SLOTS * DIM; i += E24_BLOCK_SIZE) {
        int n = i / DIM;
        int d = i % DIM;
        float d_tape = __bfloat162float(d_h_tape_b[i]);
        float d_from_attn = d_write_scores_sh[n] * h_work_sh[d];
        d_h_tape_b[i] = __float2bfloat16((1.0f - attn_sh[n]) * d_tape + d_from_attn);
    }
}

/**
 * E24 Backward Phase 2: Tanh backward
 *
 * Computes d_pre_act = d_h_work_t_total * (1 - h_work^2)
 * This is d_h_update which will be combined with d_write_val for GEMM backward.
 */
template<int N_SLOTS, int DIM>
__global__ void E24BackwardPhase2_BF16(
    const int batch_size,
    const __nv_bfloat16* __restrict__ h_work_t,         // [B, D] - h_work for this timestep
    const __nv_bfloat16* __restrict__ d_h_work_t_total, // [B, D] - total gradient for tanh (from Phase1)
    __nv_bfloat16* __restrict__ d_h_update_t,           // [B, D] - output: d_h_update for this timestep
    float* __restrict__ db_h                             // [D] - bias gradient (accumulated)
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    for (int d = tid; d < DIM; d += E24_BLOCK_SIZE) {
        float h = __bfloat162float(h_work_t[b * DIM + d]);
        float d_h = __bfloat162float(d_h_work_t_total[b * DIM + d]);
        float d_pre_act = d_h * (1.0f - h * h);  // tanh derivative

        d_h_update_t[b * DIM + d] = __float2bfloat16(d_pre_act);
        atomicAdd(&db_h[d], d_pre_act);
    }
}

/**
 * E24 Backward Phase 3: Read attention backward
 *
 * Computes gradient to h_tape and d_h_work_from_read (gradient to h_work_prev).
 * Key change: Outputs d_h_work_from_read separately instead of adding to d_h_work.
 */
template<int N_SLOTS, int DIM>
__global__ void E24BackwardPhase3_BF16(
    const int batch_size,
    const __nv_bfloat16* __restrict__ read_attn_t,    // [B, N] - read attention for this timestep
    const __nv_bfloat16* __restrict__ d_h_update_t,   // [B, D] - d_h_update (also d_read after tanh bwd)
    const __nv_bfloat16* __restrict__ h_tape_t,       // [B, N, D] - tape state at timestep t
    const __nv_bfloat16* __restrict__ h_work_prev_t,  // [B, D] - h_work at timestep t-1
    const float scale,                                 // 1/sqrt(D)
    __nv_bfloat16* __restrict__ d_h_tape,             // [B, N, D] - tape gradient (modified)
    __nv_bfloat16* __restrict__ d_h_work_from_read    // [B, D] - OUTPUT: gradient to h_work_prev from read attn
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    __shared__ float attn_sh[N_SLOTS];
    __shared__ float d_h_update_sh[DIM];
    __shared__ float h_work_prev_sh[DIM];
    __shared__ float d_read_attn_sh[N_SLOTS];
    __shared__ float d_read_scores_sh[N_SLOTS];

    const __nv_bfloat16* h_tape_b = h_tape_t + b * N_SLOTS * DIM;
    __nv_bfloat16* d_h_tape_b = d_h_tape + b * N_SLOTS * DIM;

    // Load read attention into shared memory
    if (tid < N_SLOTS) {
        attn_sh[tid] = __bfloat162float(read_attn_t[b * N_SLOTS + tid]);
    }

    // Load d_h_update and h_work_prev into shared memory
    for (int d = tid; d < DIM; d += E24_BLOCK_SIZE) {
        d_h_update_sh[d] = __bfloat162float(d_h_update_t[b * DIM + d]);
        h_work_prev_sh[d] = __bfloat162float(h_work_prev_t[b * DIM + d]);
    }
    __syncthreads();

    // Compute d_read_attn = (d_h_update[:, None, :] * h_tape_t).sum(dim=-1)
    if (tid < N_SLOTS) {
        float d_attn = 0.0f;
        for (int d = 0; d < DIM; d++) {
            d_attn += d_h_update_sh[d] * __bfloat162float(h_tape_b[tid * DIM + d]);
        }
        d_read_attn_sh[tid] = d_attn;
    }
    __syncthreads();

    // Softmax backward
    if (tid == 0) {
        float dot_sum = 0.0f;
        for (int n = 0; n < N_SLOTS; n++) {
            dot_sum += d_read_attn_sh[n] * attn_sh[n];
        }
        for (int n = 0; n < N_SLOTS; n++) {
            d_read_scores_sh[n] = attn_sh[n] * (d_read_attn_sh[n] - dot_sum) * scale;
        }
    }
    __syncthreads();

    // Update d_h_tape with both contributions
    for (int i = tid; i < N_SLOTS * DIM; i += E24_BLOCK_SIZE) {
        int n = i / DIM;
        int d = i % DIM;

        // Direct gradient: d_h_update * attn
        float d_tape_direct = attn_sh[n] * d_h_update_sh[d];

        // Gradient from softmax: d_read_scores * h_work_prev
        float d_tape_softmax = d_read_scores_sh[n] * h_work_prev_sh[d];

        float current = __bfloat162float(d_h_tape_b[i]);
        d_h_tape_b[i] = __float2bfloat16(current + d_tape_direct + d_tape_softmax);
    }

    // Output d_h_work_from_read: gradient to h_work_prev from read attention softmax
    for (int d = tid; d < DIM; d += E24_BLOCK_SIZE) {
        float d_work_contrib = 0.0f;
        for (int n = 0; n < N_SLOTS; n++) {
            d_work_contrib += d_read_scores_sh[n] * __bfloat162float(h_tape_b[n * DIM + d]);
        }
        d_h_work_from_read[b * DIM + d] = __float2bfloat16(d_work_contrib);
    }
}

/**
 * E24 Backward Phase 4a: Prepare d_gemm_output
 *
 * Concatenates d_h_update and d_write_val into d_gemm_output [B, 2D]
 * for the GEMM backward and W_all gradient computation.
 */
template<int DIM>
__global__ void E24BackwardPhase4a_BF16(
    const int batch_size,
    const __nv_bfloat16* __restrict__ d_h_update_t,    // [B, D]
    const __nv_bfloat16* __restrict__ d_write_val_t,   // [B, D]
    __nv_bfloat16* __restrict__ d_gemm_output          // [B, 2D] - for GEMM backward
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    // Concatenate d_h_update and d_write_val into d_gemm_output
    for (int d = tid; d < DIM; d += E24_BLOCK_SIZE) {
        d_gemm_output[b * 2 * DIM + d] = d_h_update_t[b * DIM + d];
        d_gemm_output[b * 2 * DIM + DIM + d] = d_write_val_t[b * DIM + d];
    }
}

/**
 * E24 Backward Phase 4b: Split d_input_concat and combine gradients for h_work_prev
 *
 * After GEMM: d_input_concat = W_all @ d_gemm_output
 * Split into dx and d_h_work_from_gemm.
 * Combine with d_h_work_from_read to get total d_h_work for next iteration.
 *
 * Key change: SET d_h_work (not add) to d_h_work_from_gemm + d_h_work_from_read
 */
template<int DIM>
__global__ void E24BackwardPhase4b_BF16(
    const int batch_size,
    const __nv_bfloat16* __restrict__ d_input_concat,    // [B, 2D] - from W_all @ d_gemm_output
    const __nv_bfloat16* __restrict__ d_h_work_from_read, // [B, D] - from Phase3
    __nv_bfloat16* __restrict__ d_h_work,                 // [B, D] - OUTPUT: SET (not add) for next iteration
    __nv_bfloat16* __restrict__ dx_t                      // [B, D] - input gradient for this timestep
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    // Split d_input_concat and combine gradients
    for (int d = tid; d < DIM; d += E24_BLOCK_SIZE) {
        // d_h_work_from_gemm = d_input_concat[:, :D]
        float d_h_from_gemm = __bfloat162float(d_input_concat[b * 2 * DIM + d]);

        // d_h_work_from_read (from Phase3)
        float d_h_from_read = __bfloat162float(d_h_work_from_read[b * DIM + d]);

        // SET d_h_work = d_h_from_gemm + d_h_from_read (for next iteration)
        d_h_work[b * DIM + d] = __float2bfloat16(d_h_from_gemm + d_h_from_read);

        // dx = d_input_concat[:, D:]
        dx_t[b * DIM + d] = d_input_concat[b * 2 * DIM + DIM + d];
    }
}

// =============================================================================
// E24 Optimized Fused Backward Kernel
// =============================================================================

/**
 * E24 Fused Backward Kernel: Combines Phase1 + Phase2 + Phase3 + Phase4a prep
 *
 * This kernel does everything before the GEMM in a single launch:
 * - Write attention backward
 * - Tanh backward
 * - Read attention backward
 * - Prepares d_gemm_output = [d_h_update; d_write_val]
 * - Prepares input_concat = [h_work_prev; x] for dW_all
 *
 * Outputs:
 * - d_gemm_output: [B, 2D] for GEMM backward
 * - input_concat: [B, 2D] for dW_all computation
 * - d_h_work_from_read: [B, D] for combining with GEMM result
 * - d_h_tape: updated in place
 * - db_h: accumulated atomically
 */
template<int N_SLOTS, int DIM>
__global__ void E24FusedBackward_BF16(
    const int batch_size,
    // Saved tensors from forward
    const __nv_bfloat16* __restrict__ write_attn_t,   // [B, N]
    const __nv_bfloat16* __restrict__ read_attn_t,    // [B, N]
    const __nv_bfloat16* __restrict__ write_val_t,    // [B, D]
    const __nv_bfloat16* __restrict__ h_tape_t,       // [B, N, D] - tape BEFORE update
    const __nv_bfloat16* __restrict__ h_work_t,       // [B, D] - h_work_new at timestep t
    const __nv_bfloat16* __restrict__ h_work_prev_t,  // [B, D] - h_work at timestep t-1
    const __nv_bfloat16* __restrict__ x_t,            // [B, D] - input at timestep t
    // Incoming gradients
    const __nv_bfloat16* __restrict__ d_h_work_out_t, // [B, D] - output gradient for this timestep
    const __nv_bfloat16* __restrict__ d_h_work_accum, // [B, D] - accumulated from later timesteps
    const float scale,
    // Outputs
    __nv_bfloat16* __restrict__ d_h_tape,             // [B, N, D] - tape gradient (modified)
    __nv_bfloat16* __restrict__ d_h_work_from_read,   // [B, D] - gradient to h_work_prev from read
    __nv_bfloat16* __restrict__ d_gemm_output,        // [B, 2D] - [d_h_update; d_write_val]
    __nv_bfloat16* __restrict__ input_concat,         // [B, 2D] - [h_work_prev; x]
    float* __restrict__ db_h                          // [D] - bias gradient (accumulated)
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    // Shared memory for this batch element
    __shared__ float write_attn_sh[N_SLOTS];
    __shared__ float read_attn_sh[N_SLOTS];
    __shared__ float write_val_sh[DIM];
    __shared__ float h_work_sh[DIM];
    __shared__ float h_work_prev_sh[DIM];
    __shared__ float d_h_work_t_sh[DIM];
    __shared__ float d_write_scores_sh[N_SLOTS];
    __shared__ float d_read_scores_sh[N_SLOTS];
    __shared__ float d_h_update_sh[DIM];

    const __nv_bfloat16* h_tape_b = h_tape_t + b * N_SLOTS * DIM;
    __nv_bfloat16* d_h_tape_b = d_h_tape + b * N_SLOTS * DIM;

    // ========== LOAD DATA INTO SHARED MEMORY ==========
    if (tid < N_SLOTS) {
        write_attn_sh[tid] = __bfloat162float(write_attn_t[b * N_SLOTS + tid]);
        read_attn_sh[tid] = __bfloat162float(read_attn_t[b * N_SLOTS + tid]);
    }
    for (int d = tid; d < DIM; d += E24_BLOCK_SIZE) {
        write_val_sh[d] = __bfloat162float(write_val_t[b * DIM + d]);
        h_work_sh[d] = __bfloat162float(h_work_t[b * DIM + d]);
        h_work_prev_sh[d] = __bfloat162float(h_work_prev_t[b * DIM + d]);
        // Combine incoming gradients
        d_h_work_t_sh[d] = __bfloat162float(d_h_work_accum[b * DIM + d])
                        + __bfloat162float(d_h_work_out_t[b * DIM + d]);
    }
    __syncthreads();

    // ========== PHASE 1: WRITE ATTENTION BACKWARD ==========
    // Compute d_write_val
    __shared__ float d_write_val_sh[DIM];
    for (int d = tid; d < DIM; d += E24_BLOCK_SIZE) {
        float d_wv = 0.0f;
        for (int n = 0; n < N_SLOTS; n++) {
            d_wv += write_attn_sh[n] * __bfloat162float(d_h_tape_b[n * DIM + d]);
        }
        d_write_val_sh[d] = d_wv;
    }
    __syncthreads();

    // Compute d_write_attn
    __shared__ float d_write_attn_sh[N_SLOTS];
    if (tid < N_SLOTS) {
        float d_attn = 0.0f;
        for (int d = 0; d < DIM; d++) {
            float d_tape = __bfloat162float(d_h_tape_b[tid * DIM + d]);
            float tape_val = __bfloat162float(h_tape_b[tid * DIM + d]);
            d_attn += d_tape * (write_val_sh[d] - tape_val);
        }
        d_write_attn_sh[tid] = d_attn;
    }
    __syncthreads();

    // Softmax backward for write attention
    if (tid == 0) {
        float dot_sum = 0.0f;
        for (int n = 0; n < N_SLOTS; n++) {
            dot_sum += d_write_attn_sh[n] * write_attn_sh[n];
        }
        for (int n = 0; n < N_SLOTS; n++) {
            d_write_scores_sh[n] = write_attn_sh[n] * (d_write_attn_sh[n] - dot_sum) * scale;
        }
    }
    __syncthreads();

    // Add write attention gradient to d_h_work_t
    __shared__ float d_h_work_t_total_sh[DIM];
    for (int d = tid; d < DIM; d += E24_BLOCK_SIZE) {
        float d_work_from_attn = 0.0f;
        for (int n = 0; n < N_SLOTS; n++) {
            d_work_from_attn += d_write_scores_sh[n] * __bfloat162float(h_tape_b[n * DIM + d]);
        }
        d_h_work_t_total_sh[d] = d_h_work_t_sh[d] + d_work_from_attn;
    }

    // Update d_h_tape for write: (1-attn) * d_h_tape + d_write_scores * h_work
    for (int i = tid; i < N_SLOTS * DIM; i += E24_BLOCK_SIZE) {
        int n = i / DIM;
        int d = i % DIM;
        float d_tape = __bfloat162float(d_h_tape_b[i]);
        float d_from_attn = d_write_scores_sh[n] * h_work_sh[d];
        d_h_tape_b[i] = __float2bfloat16((1.0f - write_attn_sh[n]) * d_tape + d_from_attn);
    }
    __syncthreads();

    // ========== PHASE 2: TANH BACKWARD ==========
    for (int d = tid; d < DIM; d += E24_BLOCK_SIZE) {
        float h = h_work_sh[d];
        float d_pre_act = d_h_work_t_total_sh[d] * (1.0f - h * h);
        d_h_update_sh[d] = d_pre_act;
        atomicAdd(&db_h[d], d_pre_act);
    }
    __syncthreads();

    // ========== PHASE 3: READ ATTENTION BACKWARD ==========
    // Compute d_read_attn
    __shared__ float d_read_attn_sh[N_SLOTS];
    if (tid < N_SLOTS) {
        float d_attn = 0.0f;
        for (int d = 0; d < DIM; d++) {
            d_attn += d_h_update_sh[d] * __bfloat162float(h_tape_b[tid * DIM + d]);
        }
        d_read_attn_sh[tid] = d_attn;
    }
    __syncthreads();

    // Softmax backward for read attention
    if (tid == 0) {
        float dot_sum = 0.0f;
        for (int n = 0; n < N_SLOTS; n++) {
            dot_sum += d_read_attn_sh[n] * read_attn_sh[n];
        }
        for (int n = 0; n < N_SLOTS; n++) {
            d_read_scores_sh[n] = read_attn_sh[n] * (d_read_attn_sh[n] - dot_sum) * scale;
        }
    }
    __syncthreads();

    // Update d_h_tape for read
    for (int i = tid; i < N_SLOTS * DIM; i += E24_BLOCK_SIZE) {
        int n = i / DIM;
        int d = i % DIM;
        float d_tape_direct = read_attn_sh[n] * d_h_update_sh[d];
        float d_tape_softmax = d_read_scores_sh[n] * h_work_prev_sh[d];
        float current = __bfloat162float(d_h_tape_b[i]);
        d_h_tape_b[i] = __float2bfloat16(current + d_tape_direct + d_tape_softmax);
    }

    // Output d_h_work_from_read
    for (int d = tid; d < DIM; d += E24_BLOCK_SIZE) {
        float d_work_contrib = 0.0f;
        for (int n = 0; n < N_SLOTS; n++) {
            d_work_contrib += d_read_scores_sh[n] * __bfloat162float(h_tape_b[n * DIM + d]);
        }
        d_h_work_from_read[b * DIM + d] = __float2bfloat16(d_work_contrib);
    }

    // ========== OUTPUT: d_gemm_output and input_concat ==========
    for (int d = tid; d < DIM; d += E24_BLOCK_SIZE) {
        // d_gemm_output = [d_h_update; d_write_val]
        d_gemm_output[b * 2 * DIM + d] = __float2bfloat16(d_h_update_sh[d]);
        d_gemm_output[b * 2 * DIM + DIM + d] = __float2bfloat16(d_write_val_sh[d]);

        // input_concat = [h_work_prev; x]
        input_concat[b * 2 * DIM + d] = __float2bfloat16(h_work_prev_sh[d]);
        input_concat[b * 2 * DIM + DIM + d] = x_t[b * DIM + d];
    }
}

/**
 * E24 Post-GEMM Backward: Split d_input_concat and combine gradients
 * Also accumulates dW_all for this timestep
 */
template<int DIM>
__global__ void E24PostGemmBackward_BF16(
    const int batch_size,
    const __nv_bfloat16* __restrict__ d_input_concat,    // [B, 2D]
    const __nv_bfloat16* __restrict__ d_h_work_from_read, // [B, D]
    __nv_bfloat16* __restrict__ d_h_work,                 // [B, D]
    __nv_bfloat16* __restrict__ dx_t                      // [B, D]
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    for (int d = tid; d < DIM; d += E24_BLOCK_SIZE) {
        float d_h_from_gemm = __bfloat162float(d_input_concat[b * 2 * DIM + d]);
        float d_h_from_read = __bfloat162float(d_h_work_from_read[b * DIM + d]);
        d_h_work[b * DIM + d] = __float2bfloat16(d_h_from_gemm + d_h_from_read);
        dx_t[b * DIM + d] = d_input_concat[b * 2 * DIM + DIM + d];
    }
}

}  // namespace


namespace hasty { namespace v0 { namespace elman_ladder {

// =============================================================================
// E24 Forward
// =============================================================================

template<typename T>
E24SingleGemmForward<T>::E24SingleGemmForward(
    bool training, int batch_size, int n_slots, int dim,
    const cublasHandle_t& blas_handle, const cudaStream_t& stream)
    : training_(training), batch_size_(batch_size), n_slots_(n_slots), dim_(dim),
      stream_(stream), blas_handle_(blas_handle) {}

template<>
void E24SingleGemmForward<__nv_bfloat16>::Run(
    int seq_len,
    const __nv_bfloat16* x_seq,          // [T, B, D] - input sequence
    const __nv_bfloat16* W_all,          // [2D, 2D] - fused weight matrix
    const __nv_bfloat16* b_h,            // [D]
    const __nv_bfloat16* h_tape_init,    // [B, N, D]
    const __nv_bfloat16* h_work_init,    // [B, D]
    __nv_bfloat16* h_work_out,           // [T, B, D]
    __nv_bfloat16* h_tape_final,         // [B, N, D]
    __nv_bfloat16* h_tape_all,           // [T+1, B, N, D] - tape history (null if inference)
    __nv_bfloat16* read_attn,            // [T, B, N]
    __nv_bfloat16* write_attn,           // [T, B, N]
    __nv_bfloat16* write_val_all,        // [T, B, D] - write values for backward
    __nv_bfloat16* workspace             // Workspace: input_concat [B, 2D] + gemm_output [B, 2D]
) {
    seq_len_ = seq_len;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    const int BD = batch_size_ * dim_;
    const int B2D = batch_size_ * 2 * dim_;
    const int BND = batch_size_ * n_slots_ * dim_;
    const int BN = batch_size_ * n_slots_;

    // Workspace layout:
    // - input_concat: [B, 2D] - concatenation of [h_work; x]
    // - gemm_output: [B, 2D] - result of W_all @ input_concat
    __nv_bfloat16* input_concat = workspace;
    __nv_bfloat16* gemm_output = workspace + B2D;

    // Copy h_tape_init to h_tape_final (will be modified in place)
    cudaMemcpyAsync(h_tape_final, h_tape_init, BND * sizeof(__nv_bfloat16),
                    cudaMemcpyDeviceToDevice, stream_);

    // Save initial tape if training
    if (training_ && h_tape_all) {
        cudaMemcpyAsync(h_tape_all, h_tape_init, BND * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToDevice, stream_);
    }

    const float scale = 1.0f / sqrtf(static_cast<float>(dim_));
    const int num_blocks = batch_size_;

    // Dispatch macros
    #define LAUNCH_E24_CONCAT(D) \
        E24ConcatKernel_BF16<D><<<num_blocks, E24_BLOCK_SIZE, 0, stream_>>>( \
            batch_size_, h_work_prev, x_t, input_concat)

    #define LAUNCH_E24_FUSED(N, D) \
        E24FusedKernel_BF16<N, D><<<num_blocks, E24_BLOCK_SIZE, 0, stream_>>>( \
            batch_size_, gemm_output, b_h, h_work_prev, h_tape_final, \
            h_work_cur, read_attn_t, write_attn_t, write_val_t, scale)

    // Process each timestep
    for (int t = 0; t < seq_len; ++t) {
        const __nv_bfloat16* x_t = x_seq + t * BD;
        const __nv_bfloat16* h_work_prev = (t == 0) ? h_work_init : (h_work_out + (t - 1) * BD);
        __nv_bfloat16* h_work_cur = h_work_out + t * BD;
        __nv_bfloat16* read_attn_t = read_attn + t * BN;
        __nv_bfloat16* write_attn_t = write_attn + t * BN;
        __nv_bfloat16* write_val_t = (training_ && write_val_all) ? (write_val_all + t * BD) : nullptr;

        // Step 1: Concatenate [h_work; x]
        if (dim_ == 256) { LAUNCH_E24_CONCAT(256); }
        else if (dim_ == 512) { LAUNCH_E24_CONCAT(512); }
        else if (dim_ == 768) { LAUNCH_E24_CONCAT(768); }
        else if (dim_ == 1024) { LAUNCH_E24_CONCAT(1024); }
        else { fprintf(stderr, "E24 CUDA: unsupported dim=%d\n", dim_); }

        // Step 2: THE SINGLE GEMM - W_all @ input_concat
        // W_all is [2D, 2D], input_concat is [B, 2D] (col-major: [2D, B])
        // Result: gemm_output [B, 2D] (col-major: [2D, B])
        cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            2 * dim_, batch_size_, 2 * dim_, &alpha,
            W_all, CUDA_R_16BF, 2 * dim_,
            input_concat, CUDA_R_16BF, 2 * dim_,
            &beta, gemm_output, CUDA_R_16BF, 2 * dim_,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // Step 3: Fused kernel - split, read attention, update, write attention, tape update
        if (n_slots_ == 8 && dim_ == 256) { LAUNCH_E24_FUSED(8, 256); }
        else if (n_slots_ == 8 && dim_ == 512) { LAUNCH_E24_FUSED(8, 512); }
        else if (n_slots_ == 8 && dim_ == 768) { LAUNCH_E24_FUSED(8, 768); }
        else if (n_slots_ == 8 && dim_ == 1024) { LAUNCH_E24_FUSED(8, 1024); }
        else if (n_slots_ == 16 && dim_ == 256) { LAUNCH_E24_FUSED(16, 256); }
        else if (n_slots_ == 16 && dim_ == 512) { LAUNCH_E24_FUSED(16, 512); }
        else if (n_slots_ == 16 && dim_ == 768) { LAUNCH_E24_FUSED(16, 768); }
        else if (n_slots_ == 16 && dim_ == 1024) { LAUNCH_E24_FUSED(16, 1024); }
        else if (n_slots_ == 32 && dim_ == 256) { LAUNCH_E24_FUSED(32, 256); }
        else if (n_slots_ == 32 && dim_ == 512) { LAUNCH_E24_FUSED(32, 512); }
        else if (n_slots_ == 32 && dim_ == 768) { LAUNCH_E24_FUSED(32, 768); }
        else if (n_slots_ == 32 && dim_ == 1024) { LAUNCH_E24_FUSED(32, 1024); }
        else if (n_slots_ == 64 && dim_ == 256) { LAUNCH_E24_FUSED(64, 256); }
        else if (n_slots_ == 64 && dim_ == 512) { LAUNCH_E24_FUSED(64, 512); }
        else if (n_slots_ == 64 && dim_ == 768) { LAUNCH_E24_FUSED(64, 768); }
        else if (n_slots_ == 64 && dim_ == 1024) { LAUNCH_E24_FUSED(64, 1024); }
        else { fprintf(stderr, "E24 CUDA: unsupported n_slots=%d, dim=%d\n", n_slots_, dim_); }

        // Save tape state if training
        if (training_ && h_tape_all) {
            cudaMemcpyAsync(h_tape_all + (t + 1) * BND, h_tape_final,
                            BND * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream_);
        }
    }

    #undef LAUNCH_E24_CONCAT
    #undef LAUNCH_E24_FUSED
}


// =============================================================================
// E24 Backward
// =============================================================================

template<typename T>
E24SingleGemmBackward<T>::E24SingleGemmBackward(
    int batch_size, int n_slots, int dim,
    const cublasHandle_t& blas_handle, const cudaStream_t& stream)
    : batch_size_(batch_size), n_slots_(n_slots), dim_(dim),
      stream_(stream), blas_handle_(blas_handle) {}

template<>
void E24SingleGemmBackward<__nv_bfloat16>::Run(
    int seq_len,
    const __nv_bfloat16* x_seq,          // [T, B, D]
    const __nv_bfloat16* h_work_all,     // [T, B, D]
    const __nv_bfloat16* h_work_init,    // [B, D]
    const __nv_bfloat16* h_tape_all,     // [T+1, B, N, D]
    const __nv_bfloat16* read_attn,      // [T, B, N]
    const __nv_bfloat16* write_attn,     // [T, B, N]
    const __nv_bfloat16* write_val_all,  // [T, B, D]
    const __nv_bfloat16* W_all,          // [2D, 2D]
    const __nv_bfloat16* d_h_work_out,   // [T, B, D]
    const __nv_bfloat16* d_h_tape_final, // [B, N, D]
    __nv_bfloat16* dx,                   // [T, B, D]
    float* db_h,                          // [D] - accumulated
    float* dW_all,                        // [2D, 2D] - accumulated
    __nv_bfloat16* workspace             // See layout below
) {
    const int num_blocks = batch_size_;
    const int BD = batch_size_ * dim_;
    const int B2D = batch_size_ * 2 * dim_;
    const int BN = batch_size_ * n_slots_;
    const int BND = batch_size_ * n_slots_ * dim_;
    const float alpha_one = 1.0f;
    const float beta_zero = 0.0f;
    const float scale = 1.0f / sqrtf(static_cast<float>(dim_));

    // OPTIMIZED Workspace layout (reduced from original):
    // d_h_tape: [B, N, D]
    // d_h_work: [B, D] - accumulated gradient to h_work_prev
    // d_h_work_from_read: [B, D] - gradient from read attention
    // d_gemm_output_all: [T, B, 2D] - stored for final dW_all GEMM
    // input_concat_all: [T, B, 2D] - stored for final dW_all GEMM
    // d_input_concat: [B, 2D] - temporary for per-timestep GEMM
    __nv_bfloat16* d_h_tape = workspace;
    __nv_bfloat16* d_h_work = d_h_tape + BND;
    __nv_bfloat16* d_h_work_from_read = d_h_work + BD;
    __nv_bfloat16* d_gemm_output_all = d_h_work_from_read + BD;
    __nv_bfloat16* input_concat_all = d_gemm_output_all + seq_len * B2D;
    __nv_bfloat16* d_input_concat = input_concat_all + seq_len * B2D;

    // Dispatch macros for fused backward
    #define LAUNCH_E24_BWD_INIT(N, D) \
        E24BackwardInit_BF16<N, D><<<num_blocks, E24_BLOCK_SIZE, 0, stream_>>>( \
            batch_size_, d_h_tape_final, d_h_tape, d_h_work)

    #define LAUNCH_E24_FUSED_BWD(N, D) \
        E24FusedBackward_BF16<N, D><<<num_blocks, E24_BLOCK_SIZE, 0, stream_>>>( \
            batch_size_, write_attn_t, read_attn_t, write_val_t, h_tape_t, \
            h_work_t, h_work_prev_t, x_t, d_h_work_out_t, d_h_work, scale, \
            d_h_tape, d_h_work_from_read, d_gemm_output_t, input_concat_t, db_h)

    #define LAUNCH_E24_POST_GEMM(D) \
        E24PostGemmBackward_BF16<D><<<num_blocks, E24_BLOCK_SIZE, 0, stream_>>>( \
            batch_size_, d_input_concat, d_h_work_from_read, d_h_work, dx_t)

    // Initialize d_h_tape and d_h_work
    if (n_slots_ == 8 && dim_ == 256) { LAUNCH_E24_BWD_INIT(8, 256); }
    else if (n_slots_ == 8 && dim_ == 512) { LAUNCH_E24_BWD_INIT(8, 512); }
    else if (n_slots_ == 8 && dim_ == 768) { LAUNCH_E24_BWD_INIT(8, 768); }
    else if (n_slots_ == 8 && dim_ == 1024) { LAUNCH_E24_BWD_INIT(8, 1024); }
    else if (n_slots_ == 16 && dim_ == 256) { LAUNCH_E24_BWD_INIT(16, 256); }
    else if (n_slots_ == 16 && dim_ == 512) { LAUNCH_E24_BWD_INIT(16, 512); }
    else if (n_slots_ == 16 && dim_ == 768) { LAUNCH_E24_BWD_INIT(16, 768); }
    else if (n_slots_ == 16 && dim_ == 1024) { LAUNCH_E24_BWD_INIT(16, 1024); }
    else if (n_slots_ == 32 && dim_ == 256) { LAUNCH_E24_BWD_INIT(32, 256); }
    else if (n_slots_ == 32 && dim_ == 512) { LAUNCH_E24_BWD_INIT(32, 512); }
    else if (n_slots_ == 32 && dim_ == 768) { LAUNCH_E24_BWD_INIT(32, 768); }
    else if (n_slots_ == 32 && dim_ == 1024) { LAUNCH_E24_BWD_INIT(32, 1024); }
    else if (n_slots_ == 64 && dim_ == 256) { LAUNCH_E24_BWD_INIT(64, 256); }
    else if (n_slots_ == 64 && dim_ == 512) { LAUNCH_E24_BWD_INIT(64, 512); }
    else if (n_slots_ == 64 && dim_ == 768) { LAUNCH_E24_BWD_INIT(64, 768); }
    else if (n_slots_ == 64 && dim_ == 1024) { LAUNCH_E24_BWD_INIT(64, 1024); }
    else {
        fprintf(stderr, "E24 CUDA backward: unsupported n_slots=%d, dim=%d\n", n_slots_, dim_);
        return;
    }

    // Backward pass: iterate in reverse
    // OPTIMIZED: 1 fused kernel + 1 GEMM + 1 post-GEMM kernel per timestep
    // (vs. 5 kernels + 1 GEMM + 1 kernel in original)
    for (int t = seq_len - 1; t >= 0; t--) {
        const __nv_bfloat16* x_t = x_seq + t * BD;
        const __nv_bfloat16* write_attn_t = write_attn + t * BN;
        const __nv_bfloat16* read_attn_t = read_attn + t * BN;
        const __nv_bfloat16* write_val_t = write_val_all + t * BD;
        const __nv_bfloat16* d_h_work_out_t = d_h_work_out + t * BD;
        const __nv_bfloat16* h_work_t = h_work_all + t * BD;
        const __nv_bfloat16* h_tape_t = h_tape_all + t * BND;
        const __nv_bfloat16* h_work_prev_t = (t > 0) ? (h_work_all + (t - 1) * BD) : h_work_init;
        __nv_bfloat16* dx_t = dx + t * BD;
        __nv_bfloat16* d_gemm_output_t = d_gemm_output_all + t * B2D;
        __nv_bfloat16* input_concat_t = input_concat_all + t * B2D;

        // FUSED: Phase1 + Phase2 + Phase3 + input_concat prep + d_gemm_output prep
        if (n_slots_ == 8 && dim_ == 256) { LAUNCH_E24_FUSED_BWD(8, 256); }
        else if (n_slots_ == 8 && dim_ == 512) { LAUNCH_E24_FUSED_BWD(8, 512); }
        else if (n_slots_ == 8 && dim_ == 768) { LAUNCH_E24_FUSED_BWD(8, 768); }
        else if (n_slots_ == 8 && dim_ == 1024) { LAUNCH_E24_FUSED_BWD(8, 1024); }
        else if (n_slots_ == 16 && dim_ == 256) { LAUNCH_E24_FUSED_BWD(16, 256); }
        else if (n_slots_ == 16 && dim_ == 512) { LAUNCH_E24_FUSED_BWD(16, 512); }
        else if (n_slots_ == 16 && dim_ == 768) { LAUNCH_E24_FUSED_BWD(16, 768); }
        else if (n_slots_ == 16 && dim_ == 1024) { LAUNCH_E24_FUSED_BWD(16, 1024); }
        else if (n_slots_ == 32 && dim_ == 256) { LAUNCH_E24_FUSED_BWD(32, 256); }
        else if (n_slots_ == 32 && dim_ == 512) { LAUNCH_E24_FUSED_BWD(32, 512); }
        else if (n_slots_ == 32 && dim_ == 768) { LAUNCH_E24_FUSED_BWD(32, 768); }
        else if (n_slots_ == 32 && dim_ == 1024) { LAUNCH_E24_FUSED_BWD(32, 1024); }
        else if (n_slots_ == 64 && dim_ == 256) { LAUNCH_E24_FUSED_BWD(64, 256); }
        else if (n_slots_ == 64 && dim_ == 512) { LAUNCH_E24_FUSED_BWD(64, 512); }
        else if (n_slots_ == 64 && dim_ == 768) { LAUNCH_E24_FUSED_BWD(64, 768); }
        else if (n_slots_ == 64 && dim_ == 1024) { LAUNCH_E24_FUSED_BWD(64, 1024); }

        // cuBLAS: d_input_concat = W_all @ d_gemm_output
        cublasGemmEx(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            2 * dim_, batch_size_, 2 * dim_,
            &alpha_one,
            W_all, CUDA_R_16BF, 2 * dim_,
            d_gemm_output_t, CUDA_R_16BF, 2 * dim_,
            &beta_zero,
            d_input_concat, CUDA_R_16BF, 2 * dim_,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
        );

        // POST-GEMM: Split d_input_concat -> d_h_work and dx
        if (dim_ == 256) { LAUNCH_E24_POST_GEMM(256); }
        else if (dim_ == 512) { LAUNCH_E24_POST_GEMM(512); }
        else if (dim_ == 768) { LAUNCH_E24_POST_GEMM(768); }
        else if (dim_ == 1024) { LAUNCH_E24_POST_GEMM(1024); }
    }

    #undef LAUNCH_E24_BWD_INIT
    #undef LAUNCH_E24_FUSED_BWD
    #undef LAUNCH_E24_POST_GEMM

    // Compute dW_all via single batched GEMM over all timesteps
    // dW_all = d_gemm_output_all.T @ input_concat_all
    const int TB = seq_len * batch_size_;
    cublasGemmEx(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        2 * dim_, 2 * dim_, TB,
        &alpha_one,
        d_gemm_output_all, CUDA_R_16BF, 2 * dim_,
        input_concat_all, CUDA_R_16BF, 2 * dim_,
        &beta_zero,
        dW_all, CUDA_R_32F, 2 * dim_,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
    );
}

// Explicit instantiations
template class E24SingleGemmForward<__nv_bfloat16>;
template class E24SingleGemmBackward<__nv_bfloat16>;

}}}  // namespace hasty::v0::elman_ladder
