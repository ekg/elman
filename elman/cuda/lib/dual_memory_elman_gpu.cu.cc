/**
 * E23 Dual-Memory Elman CUDA Kernels
 *
 * Architecture:
 *   - Tape: [B, N, D] - Large linear storage (N slots)
 *   - Working Memory: [B, D] - Small nonlinear compute
 *
 * Per timestep:
 *   1. Read: h_work queries tape via attention → read value
 *   2. Update: h_work_new = tanh(W_h @ h_work + W_x @ x + read + b)
 *   3. Write: h_tape_new = (1-attn)*h_tape + attn*write_value
 *
 * Optimization strategy:
 *   - Pre-compute W_x @ x for all T (batch GEMM via cuBLAS)
 *   - Per-timestep W_h @ h and W_write @ h via cuBLAS GEMM
 *   - Fused kernel only handles attention + element-wise ops
 *
 * HYBRID TEMPLATE PATTERN:
 *   - Template on N_SLOTS (small, benefits from unrolling)
 *   - Dynamic shared memory for DIM (large, memory-bound anyway)
 *   - Only 4 specializations per kernel (N_SLOTS: 8, 16, 32, 64)
 */

#include "hasty/elman_ladder.h"
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

namespace {

// Block size for the fused kernel
constexpr int E23_BLOCK_SIZE = 256;

/**
 * E23 Fused Kernel Phase 1: Read attention + Update h_work
 *
 * Inputs (from cuBLAS):
 *   - Rh: W_h @ h_work [B, D]
 *   - x_proj_t: W_x @ x[t] [B, D]
 *
 * Operations:
 *   1. Read attention: attn = softmax(h_tape @ h_work * scale)
 *   2. Read value: read = attn @ h_tape
 *   3. Update: h_work_new = tanh(Rh + x_proj_t + read + b)
 *
 * Shared memory layout: [h_work_sh(DIM), attn_sh(N_SLOTS)]
 * Total: (DIM + N_SLOTS) * sizeof(float)
 */
template<int N_SLOTS>
__global__ void E23Phase1Kernel_BF16(
    const int batch_size,
    const int DIM,
    // Pre-computed via cuBLAS
    const __nv_bfloat16* __restrict__ Rh,        // [B, D] - W_h @ h_work
    const __nv_bfloat16* __restrict__ x_proj_t,  // [B, D] - W_x @ x[t]
    const __nv_bfloat16* __restrict__ b_h,       // [D]
    // State
    const __nv_bfloat16* __restrict__ h_tape,    // [B, N, D]
    const __nv_bfloat16* __restrict__ h_work,    // [B, D]
    // Outputs
    __nv_bfloat16* __restrict__ h_work_out,      // [B, D]
    __nv_bfloat16* __restrict__ read_attn_out,   // [B, N]
    const float scale
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    // Dynamic shared memory layout
    extern __shared__ char shared_mem[];
    float* h_work_sh = (float*)shared_mem;
    float* attn_sh = h_work_sh + DIM;

    // Load h_work into shared memory
    for (int d = tid; d < DIM; d += E23_BLOCK_SIZE) {
        h_work_sh[d] = __bfloat162float(h_work[b * DIM + d]);
    }
    __syncthreads();

    // Compute read attention scores: score[n] = h_tape[n] @ h_work
    if (tid < N_SLOTS) {
        float score = 0.0f;
        const __nv_bfloat16* tape_n = h_tape + b * N_SLOTS * DIM + tid * DIM;
        for (int d = 0; d < DIM; d++) {
            score += __bfloat162float(tape_n[d]) * h_work_sh[d];
        }
        attn_sh[tid] = score * scale;
    }
    __syncthreads();

    // Softmax over N_SLOTS (single thread for simplicity, N is small)
    if (tid == 0) {
        float max_score = attn_sh[0];
        #pragma unroll
        for (int n = 1; n < N_SLOTS; n++) {
            max_score = fmaxf(max_score, attn_sh[n]);
        }
        float sum_exp = 0.0f;
        #pragma unroll
        for (int n = 0; n < N_SLOTS; n++) {
            attn_sh[n] = expf(attn_sh[n] - max_score);
            sum_exp += attn_sh[n];
        }
        #pragma unroll
        for (int n = 0; n < N_SLOTS; n++) {
            attn_sh[n] /= sum_exp;
        }
    }
    __syncthreads();

    // Store read attention
    if (tid < N_SLOTS) {
        read_attn_out[b * N_SLOTS + tid] = __float2bfloat16(attn_sh[tid]);
    }

    // Compute h_work_new: tanh(Rh + x_proj_t + read_val + b_h)
    for (int d = tid; d < DIM; d += E23_BLOCK_SIZE) {
        // Weighted read: read[d] = sum_n attn[n] * h_tape[n, d]
        float read_d = 0.0f;
        #pragma unroll
        for (int n = 0; n < N_SLOTS; n++) {
            read_d += attn_sh[n] * __bfloat162float(h_tape[b * N_SLOTS * DIM + n * DIM + d]);
        }

        // Combine: Rh + x_proj_t + read + b
        float val = __bfloat162float(Rh[b * DIM + d])
                  + __bfloat162float(x_proj_t[b * DIM + d])
                  + read_d
                  + __bfloat162float(b_h[d]);

        h_work_out[b * DIM + d] = __float2bfloat16(tanhf(val));
    }
}


/**
 * E23 Fused Kernel Phase 2: Write attention + Update tape
 *
 * Inputs (from cuBLAS):
 *   - write_val: W_write @ h_work_new [B, D]
 *
 * Operations:
 *   1. Write attention: attn = softmax(h_tape @ h_work_new * scale)
 *   2. Update tape: h_tape = (1-attn) * h_tape + attn * write_val
 *
 * Shared memory layout: [h_work_sh(DIM), write_val_sh(DIM), attn_sh(N_SLOTS)]
 * Total: (2*DIM + N_SLOTS) * sizeof(float)
 */
template<int N_SLOTS>
__global__ void E23Phase2Kernel_BF16(
    const int batch_size,
    const int DIM,
    // Pre-computed via cuBLAS
    const __nv_bfloat16* __restrict__ write_val, // [B, D] - W_write @ h_work_new
    // State
    const __nv_bfloat16* __restrict__ h_work_new,// [B, D]
    __nv_bfloat16* __restrict__ h_tape,          // [B, N, D] - modified in place
    // Outputs
    __nv_bfloat16* __restrict__ write_attn_out,  // [B, N]
    const float scale
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    // Dynamic shared memory layout
    extern __shared__ char shared_mem[];
    float* h_work_sh = (float*)shared_mem;
    float* write_val_sh = h_work_sh + DIM;
    float* attn_sh = write_val_sh + DIM;

    // Load h_work_new and write_val into shared memory
    for (int d = tid; d < DIM; d += E23_BLOCK_SIZE) {
        h_work_sh[d] = __bfloat162float(h_work_new[b * DIM + d]);
        write_val_sh[d] = __bfloat162float(write_val[b * DIM + d]);
    }
    __syncthreads();

    // Compute write attention scores: score[n] = h_tape[n] @ h_work_new
    __nv_bfloat16* tape_b = h_tape + b * N_SLOTS * DIM;
    if (tid < N_SLOTS) {
        float score = 0.0f;
        for (int d = 0; d < DIM; d++) {
            score += __bfloat162float(tape_b[tid * DIM + d]) * h_work_sh[d];
        }
        attn_sh[tid] = score * scale;
    }
    __syncthreads();

    // Softmax
    if (tid == 0) {
        float max_score = attn_sh[0];
        #pragma unroll
        for (int n = 1; n < N_SLOTS; n++) {
            max_score = fmaxf(max_score, attn_sh[n]);
        }
        float sum_exp = 0.0f;
        #pragma unroll
        for (int n = 0; n < N_SLOTS; n++) {
            attn_sh[n] = expf(attn_sh[n] - max_score);
            sum_exp += attn_sh[n];
        }
        #pragma unroll
        for (int n = 0; n < N_SLOTS; n++) {
            attn_sh[n] /= sum_exp;
        }
    }
    __syncthreads();

    // Store write attention
    if (tid < N_SLOTS) {
        write_attn_out[b * N_SLOTS + tid] = __float2bfloat16(attn_sh[tid]);
    }

    // Update tape: h_tape = (1 - attn) * h_tape + attn * write_val
    for (int i = tid; i < N_SLOTS * DIM; i += E23_BLOCK_SIZE) {
        int n = i / DIM;
        int d = i % DIM;
        float attn_n = attn_sh[n];
        float old_val = __bfloat162float(tape_b[i]);
        float new_val = (1.0f - attn_n) * old_val + attn_n * write_val_sh[d];
        tape_b[i] = __float2bfloat16(new_val);
    }
}


/**
 * E23 Fused Kernel: Phase1 + inline W_write GEMM + Phase2
 *
 * Combines read attention, h_work update, W_write matmul, write attention, and tape update
 * into a single kernel launch to minimize overhead.
 *
 * Operations:
 *   1. Read attention: attn = softmax(h_tape @ h_work * scale)
 *   2. Read value: read = sum(attn[n] * h_tape[n])
 *   3. Update: h_work_new = tanh(Rh + x_proj_t + read + b)
 *   4. INLINE: write_val = W_write @ h_work_new
 *   5. Write attention: attn = softmax(h_tape @ h_work_new * scale)
 *   6. Update tape: h_tape = (1-attn) * h_tape + attn * write_val
 *
 * Shared memory layout: [h_work_sh(DIM), attn_sh(N_SLOTS), h_work_new_sh(DIM), write_val_sh(DIM)]
 * Total: (3*DIM + N_SLOTS) * sizeof(float)
 */
template<int N_SLOTS>
__global__ void E23FusedKernel_BF16(
    const int batch_size,
    const int DIM,
    // Pre-computed via cuBLAS
    const __nv_bfloat16* __restrict__ Rh,        // [B, D] - W_h @ h_work
    const __nv_bfloat16* __restrict__ x_proj_t,  // [B, D] - W_x @ x[t]
    const __nv_bfloat16* __restrict__ b_h,       // [D]
    const __nv_bfloat16* __restrict__ W_write,   // [D, D] - W_write matrix
    // State
    const __nv_bfloat16* __restrict__ h_work,    // [B, D] - previous h_work
    __nv_bfloat16* __restrict__ h_tape,          // [B, N, D] - modified in place
    // Outputs
    __nv_bfloat16* __restrict__ h_work_out,      // [B, D]
    __nv_bfloat16* __restrict__ read_attn_out,   // [B, N]
    __nv_bfloat16* __restrict__ write_attn_out,  // [B, N]
    const float scale
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    // Dynamic shared memory layout
    extern __shared__ char shared_mem[];
    float* h_work_sh = (float*)shared_mem;
    float* attn_sh = h_work_sh + DIM;
    float* h_work_new_sh = attn_sh + N_SLOTS;
    float* write_val_sh = h_work_new_sh + DIM;

    // ============================================
    // PHASE 1: Read attention + h_work update
    // ============================================

    // Load h_work into shared memory
    for (int d = tid; d < DIM; d += E23_BLOCK_SIZE) {
        h_work_sh[d] = __bfloat162float(h_work[b * DIM + d]);
    }
    __syncthreads();

    // Compute read attention scores: score[n] = h_tape[n] @ h_work
    __nv_bfloat16* tape_b = h_tape + b * N_SLOTS * DIM;
    if (tid < N_SLOTS) {
        float score = 0.0f;
        for (int d = 0; d < DIM; d++) {
            score += __bfloat162float(tape_b[tid * DIM + d]) * h_work_sh[d];
        }
        attn_sh[tid] = score * scale;
    }
    __syncthreads();

    // Softmax over N_SLOTS (single thread, N is small)
    if (tid == 0) {
        float max_score = attn_sh[0];
        #pragma unroll
        for (int n = 1; n < N_SLOTS; n++) {
            max_score = fmaxf(max_score, attn_sh[n]);
        }
        float sum_exp = 0.0f;
        #pragma unroll
        for (int n = 0; n < N_SLOTS; n++) {
            attn_sh[n] = expf(attn_sh[n] - max_score);
            sum_exp += attn_sh[n];
        }
        #pragma unroll
        for (int n = 0; n < N_SLOTS; n++) {
            attn_sh[n] /= sum_exp;
        }
    }
    __syncthreads();

    // Store read attention
    if (tid < N_SLOTS) {
        read_attn_out[b * N_SLOTS + tid] = __float2bfloat16(attn_sh[tid]);
    }

    // Compute h_work_new: tanh(Rh + x_proj_t + read_val + b_h)
    for (int d = tid; d < DIM; d += E23_BLOCK_SIZE) {
        // Weighted read: read[d] = sum_n attn[n] * h_tape[n, d]
        float read_d = 0.0f;
        #pragma unroll
        for (int n = 0; n < N_SLOTS; n++) {
            read_d += attn_sh[n] * __bfloat162float(tape_b[n * DIM + d]);
        }

        // Combine: Rh + x_proj_t + read + b
        float val = __bfloat162float(Rh[b * DIM + d])
                  + __bfloat162float(x_proj_t[b * DIM + d])
                  + read_d
                  + __bfloat162float(b_h[d]);

        float h_new = tanhf(val);
        h_work_new_sh[d] = h_new;
        h_work_out[b * DIM + d] = __float2bfloat16(h_new);
    }
    __syncthreads();

    // ============================================
    // INLINE W_WRITE GEMM: write_val = W_write @ h_work_new
    // Warp-collaborative: each warp computes one output element
    // 8 warps = 8 outputs per iteration, D/8 = 64 iterations for D=512
    // Uses coalesced reads + warp shuffle reduction (no explicit syncs in loop)
    // ============================================
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    constexpr int NUM_WARPS = E23_BLOCK_SIZE / 32;  // 8 warps

    // Process output elements in batches of NUM_WARPS
    for (int base_i = 0; base_i < DIM; base_i += NUM_WARPS) {
        int i = base_i + warp_id;
        if (i < DIM) {
            const __nv_bfloat16* W_row = W_write + i * DIM;

            // All 32 threads in warp read consecutive elements (COALESCED!)
            float partial = 0.0f;
            #pragma unroll 4
            for (int j = lane_id; j < DIM; j += 32) {
                partial += __bfloat162float(W_row[j]) * h_work_new_sh[j];
            }

            // Warp shuffle reduction (no __syncthreads needed!)
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                partial += __shfl_down_sync(0xffffffff, partial, offset);
            }

            // Lane 0 of each warp writes the result
            if (lane_id == 0) {
                write_val_sh[i] = partial;
            }
        }
    }
    __syncthreads();  // Single sync after all outputs computed

    // ============================================
    // PHASE 2: Write attention + tape update
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
        #pragma unroll
        for (int n = 1; n < N_SLOTS; n++) {
            max_score = fmaxf(max_score, attn_sh[n]);
        }
        float sum_exp = 0.0f;
        #pragma unroll
        for (int n = 0; n < N_SLOTS; n++) {
            attn_sh[n] = expf(attn_sh[n] - max_score);
            sum_exp += attn_sh[n];
        }
        #pragma unroll
        for (int n = 0; n < N_SLOTS; n++) {
            attn_sh[n] /= sum_exp;
        }
    }
    __syncthreads();

    // Store write attention
    if (tid < N_SLOTS) {
        write_attn_out[b * N_SLOTS + tid] = __float2bfloat16(attn_sh[tid]);
    }

    // Update tape: h_tape = (1 - attn) * h_tape + attn * write_val
    for (int i = tid; i < N_SLOTS * DIM; i += E23_BLOCK_SIZE) {
        int n = i / DIM;
        int d = i % DIM;
        float attn_n = attn_sh[n];
        float old_val = __bfloat162float(tape_b[i]);
        float new_val = (1.0f - attn_n) * old_val + attn_n * write_val_sh[d];
        tape_b[i] = __float2bfloat16(new_val);
    }
}


/**
 * E23 Backward Phase 1: Write attention backward
 *
 * Computes d_write_val from attention-weighted d_h_tape and updates d_h_tape.
 * Also adds d_output[t] to d_h_work.
 *
 * Shared memory layout: [attn_sh(N_SLOTS)]
 * Total: N_SLOTS * sizeof(float)
 */
template<int N_SLOTS>
__global__ void E23BackwardPhase1_BF16(
    const int batch_size,
    const int DIM,
    const __nv_bfloat16* __restrict__ write_attn_t, // [B, N] - attention for this timestep
    const __nv_bfloat16* __restrict__ d_h_work_out_t, // [B, D] - output gradient for this timestep
    __nv_bfloat16* __restrict__ d_h_work,           // [B, D] - accumulated work gradient (modified)
    __nv_bfloat16* __restrict__ d_h_tape,           // [B, N, D] - tape gradient (modified)
    __nv_bfloat16* __restrict__ d_write_val_t       // [B, D] - output: d_write_val for cuBLAS
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    // Dynamic shared memory layout
    extern __shared__ char shared_mem[];
    float* attn_sh = (float*)shared_mem;

    // Load write attention into shared memory
    if (tid < N_SLOTS) {
        attn_sh[tid] = __bfloat162float(write_attn_t[b * N_SLOTS + tid]);
    }
    __syncthreads();

    __nv_bfloat16* d_h_tape_b = d_h_tape + b * N_SLOTS * DIM;

    // Add output gradient to d_h_work
    // d_write_val = sum_n(attn[n] * d_h_tape[n])
    for (int d = tid; d < DIM; d += E23_BLOCK_SIZE) {
        // Add output gradient
        float d_h = __bfloat162float(d_h_work[b * DIM + d])
                  + __bfloat162float(d_h_work_out_t[b * DIM + d]);
        d_h_work[b * DIM + d] = __float2bfloat16(d_h);

        // Compute d_write_val
        float d_wv = 0.0f;
        #pragma unroll
        for (int n = 0; n < N_SLOTS; n++) {
            d_wv += attn_sh[n] * __bfloat162float(d_h_tape_b[n * DIM + d]);
        }
        d_write_val_t[b * DIM + d] = __float2bfloat16(d_wv);
    }
    __syncthreads();

    // Update d_h_tape: d_h_tape = (1-attn) * d_h_tape (gate for write backward)
    for (int i = tid; i < N_SLOTS * DIM; i += E23_BLOCK_SIZE) {
        int n = i / DIM;
        float d_tape = __bfloat162float(d_h_tape_b[i]);
        d_h_tape_b[i] = __float2bfloat16((1.0f - attn_sh[n]) * d_tape);
    }
}

/**
 * E23 Backward Phase 2: Tanh backward
 *
 * Computes d_pre_act from d_h_work and h_work, saves to d_pre_act_all.
 *
 * No shared memory needed for this kernel.
 */
template<int N_SLOTS>
__global__ void E23BackwardPhase2_BF16(
    const int batch_size,
    const int DIM,
    const __nv_bfloat16* __restrict__ h_work_t,     // [B, D] - h_work for this timestep
    const __nv_bfloat16* __restrict__ d_h_work,     // [B, D] - accumulated work gradient
    __nv_bfloat16* __restrict__ dx_proj_t,          // [B, D] - output: dx_proj for this timestep
    __nv_bfloat16* __restrict__ d_pre_act_t,        // [B, D] - output: d_pre_act for cuBLAS
    float* __restrict__ db_h                        // [D] - bias gradient (accumulated)
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    for (int d = tid; d < DIM; d += E23_BLOCK_SIZE) {
        float h = __bfloat162float(h_work_t[b * DIM + d]);
        float d_h = __bfloat162float(d_h_work[b * DIM + d]);
        float d_pre_act = d_h * (1.0f - h * h);  // tanh derivative

        dx_proj_t[b * DIM + d] = __float2bfloat16(d_pre_act);
        d_pre_act_t[b * DIM + d] = __float2bfloat16(d_pre_act);

        atomicAdd(&db_h[d], d_pre_act);
    }
}

/**
 * E23 Backward Phase 3: Read attention backward (COMPLETE)
 *
 * Computes the full read attention backward pass:
 * 1. d_h_tape += d_pre_act[:, None, :] * read_attn[:, :, None]  (direct gradient)
 * 2. Softmax backward for read attention scores
 * 3. d_h_work += (d_read_scores[:, :, None] * h_tape_t).sum(dim=1)
 * 4. d_h_tape += d_read_scores[:, :, None] * h_work_prev[:, None, :]
 *
 * Shared memory layout: [attn_sh(N_SLOTS), d_pre_act_sh(DIM), h_work_prev_sh(DIM),
 *                        d_read_attn_sh(N_SLOTS), d_read_scores_sh(N_SLOTS)]
 * Total: (2*DIM + 3*N_SLOTS) * sizeof(float)
 */
template<int N_SLOTS>
__global__ void E23BackwardPhase3_BF16(
    const int batch_size,
    const int DIM,
    const __nv_bfloat16* __restrict__ read_attn_t,   // [B, N] - read attention for this timestep
    const __nv_bfloat16* __restrict__ d_pre_act_t,   // [B, D] - d_pre_act (also d_read)
    const __nv_bfloat16* __restrict__ h_tape_t,      // [B, N, D] - tape state at timestep t
    const __nv_bfloat16* __restrict__ h_work_prev_t, // [B, D] - h_work at timestep t-1
    const float scale,                                // 1/sqrt(D)
    __nv_bfloat16* __restrict__ d_h_tape,            // [B, N, D] - tape gradient (modified)
    __nv_bfloat16* __restrict__ d_h_work             // [B, D] - work gradient (modified, added to)
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    // Dynamic shared memory layout
    extern __shared__ char shared_mem[];
    float* attn_sh = (float*)shared_mem;
    float* d_pre_act_sh = attn_sh + N_SLOTS;
    float* h_work_prev_sh = d_pre_act_sh + DIM;
    float* d_read_attn_sh = h_work_prev_sh + DIM;
    float* d_read_scores_sh = d_read_attn_sh + N_SLOTS;

    const __nv_bfloat16* h_tape_b = h_tape_t + b * N_SLOTS * DIM;
    __nv_bfloat16* d_h_tape_b = d_h_tape + b * N_SLOTS * DIM;

    // Load read attention into shared memory
    if (tid < N_SLOTS) {
        attn_sh[tid] = __bfloat162float(read_attn_t[b * N_SLOTS + tid]);
    }

    // Load d_pre_act and h_work_prev into shared memory
    for (int d = tid; d < DIM; d += E23_BLOCK_SIZE) {
        d_pre_act_sh[d] = __bfloat162float(d_pre_act_t[b * DIM + d]);
        h_work_prev_sh[d] = __bfloat162float(h_work_prev_t[b * DIM + d]);
    }
    __syncthreads();

    // ============================================
    // Step 1: Compute d_read_attn = (d_pre_act[:, None, :] * h_tape_t).sum(dim=-1)
    // Each thread computes partial sum for one slot
    // ============================================
    if (tid < N_SLOTS) {
        float d_attn = 0.0f;
        for (int d = 0; d < DIM; d++) {
            d_attn += d_pre_act_sh[d] * __bfloat162float(h_tape_b[tid * DIM + d]);
        }
        d_read_attn_sh[tid] = d_attn;
    }
    __syncthreads();

    // ============================================
    // Step 2: Softmax backward
    // d_read_scores = read_attn * (d_read_attn - sum(d_read_attn * read_attn)) * scale
    // ============================================
    if (tid == 0) {
        // Compute sum(d_read_attn * read_attn)
        float dot_sum = 0.0f;
        #pragma unroll
        for (int n = 0; n < N_SLOTS; n++) {
            dot_sum += d_read_attn_sh[n] * attn_sh[n];
        }
        // Compute d_read_scores for each slot
        #pragma unroll
        for (int n = 0; n < N_SLOTS; n++) {
            d_read_scores_sh[n] = attn_sh[n] * (d_read_attn_sh[n] - dot_sum) * scale;
        }
    }
    __syncthreads();

    // ============================================
    // Step 3: Update d_h_tape with both contributions
    // d_h_tape += d_pre_act[:, None, :] * read_attn[:, :, None]  (outer product)
    // d_h_tape += d_read_scores[:, :, None] * h_work_prev[:, None, :]  (from softmax)
    // ============================================
    for (int i = tid; i < N_SLOTS * DIM; i += E23_BLOCK_SIZE) {
        int n = i / DIM;
        int d = i % DIM;

        // Direct gradient: d_pre_act * attn
        float d_tape_direct = attn_sh[n] * d_pre_act_sh[d];

        // Gradient from softmax: d_read_scores * h_work_prev
        float d_tape_softmax = d_read_scores_sh[n] * h_work_prev_sh[d];

        float current = __bfloat162float(d_h_tape_b[i]);
        d_h_tape_b[i] = __float2bfloat16(current + d_tape_direct + d_tape_softmax);
    }

    // ============================================
    // Step 4: Update d_h_work with gradient from softmax
    // d_h_work += (d_read_scores[:, :, None] * h_tape_t).sum(dim=1)
    // ============================================
    for (int d = tid; d < DIM; d += E23_BLOCK_SIZE) {
        float d_work_contrib = 0.0f;
        #pragma unroll
        for (int n = 0; n < N_SLOTS; n++) {
            d_work_contrib += d_read_scores_sh[n] * __bfloat162float(h_tape_b[n * DIM + d]);
        }
        // Add to existing d_h_work
        float current = __bfloat162float(d_h_work[b * DIM + d]);
        d_h_work[b * DIM + d] = __float2bfloat16(current + d_work_contrib);
    }
}

/**
 * Initialize d_h_tape from final gradient and zero d_h_work
 *
 * No shared memory needed for this kernel.
 */
template<int N_SLOTS>
__global__ void E23BackwardInit_BF16(
    const int batch_size,
    const int DIM,
    const __nv_bfloat16* __restrict__ d_h_tape_final, // [B, N, D]
    __nv_bfloat16* __restrict__ d_h_tape,             // [B, N, D]
    __nv_bfloat16* __restrict__ d_h_work              // [B, D]
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    // Copy d_h_tape from final gradient
    for (int i = tid; i < N_SLOTS * DIM; i += E23_BLOCK_SIZE) {
        d_h_tape[b * N_SLOTS * DIM + i] = d_h_tape_final[b * N_SLOTS * DIM + i];
    }

    // Zero d_h_work
    for (int d = tid; d < DIM; d += E23_BLOCK_SIZE) {
        d_h_work[b * DIM + d] = __float2bfloat16(0.0f);
    }
}

}  // namespace

namespace hasty { namespace v0 { namespace elman_ladder {

// =============================================================================
// E23 Forward
// =============================================================================

template<typename T>
DualMemoryElmanForward<T>::DualMemoryElmanForward(
    bool training, int batch_size, int n_slots, int dim,
    const cublasHandle_t& blas_handle, const cudaStream_t& stream)
    : training_(training), batch_size_(batch_size), n_slots_(n_slots), dim_(dim),
      stream_(stream), blas_handle_(blas_handle) {}

template<>
void DualMemoryElmanForward<__nv_bfloat16>::Run(
    int seq_len,
    const __nv_bfloat16* x_proj,      // [T, B, D] - pre-computed x @ W_x^T
    const __nv_bfloat16* W_h,         // [D, D]
    const __nv_bfloat16* b_h,         // [D]
    const __nv_bfloat16* W_write,     // [D, D]
    const __nv_bfloat16* h_tape_init, // [B, N, D]
    const __nv_bfloat16* h_work_init, // [B, D]
    __nv_bfloat16* h_work_out,        // [T, B, D]
    __nv_bfloat16* h_tape_final,      // [B, N, D]
    __nv_bfloat16* h_tape_all,        // [T+1, B, N, D] - tape history for backward (null if inference)
    __nv_bfloat16* read_attn,         // [T, B, N]
    __nv_bfloat16* write_attn,        // [T, B, N]
    __nv_bfloat16* workspace          // Workspace: tmp_Rh [B, D] + tmp_write_val [B, D]
) {
    seq_len_ = seq_len;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    const int BD = batch_size_ * dim_;

    // Workspace layout:
    // - tmp_Rh: [B, D] - W_h @ h_work computed per timestep
    // - tmp_write_val: [B, D] - W_write @ h_work_new computed per timestep
    __nv_bfloat16* tmp_Rh = workspace;
    __nv_bfloat16* tmp_write_val = tmp_Rh + BD;

    // Copy h_tape_init to h_tape_final (will be modified in place)
    cudaMemcpyAsync(h_tape_final, h_tape_init,
                    batch_size_ * n_slots_ * dim_ * sizeof(__nv_bfloat16),
                    cudaMemcpyDeviceToDevice, stream_);

    // Save initial tape if training
    if (training_ && h_tape_all) {
        cudaMemcpyAsync(h_tape_all, h_tape_init,
                        batch_size_ * n_slots_ * dim_ * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToDevice, stream_);
    }

    const float scale = 1.0f / sqrtf(static_cast<float>(dim_));
    const int num_blocks = batch_size_;

    // Shared memory sizes for Phase1 and Phase2
    // Phase1: h_work_sh(DIM) + attn_sh(N_SLOTS)
    // Phase2: h_work_sh(DIM) + write_val_sh(DIM) + attn_sh(N_SLOTS)
    const size_t shmem_phase1 = (dim_ + n_slots_) * sizeof(float);
    const size_t shmem_phase2 = (2 * dim_ + n_slots_) * sizeof(float);

    // Phase1 + Phase2 kernels (cuBLAS for W_write is faster)
    #define LAUNCH_E23_PHASE1(N) \
        E23Phase1Kernel_BF16<N><<<num_blocks, E23_BLOCK_SIZE, shmem_phase1, stream_>>>( \
            batch_size_, dim_, tmp_Rh, x_proj_t, b_h, h_tape_final, h_work_prev, \
            h_work_cur, read_attn_t, scale)

    #define LAUNCH_E23_PHASE2(N) \
        E23Phase2Kernel_BF16<N><<<num_blocks, E23_BLOCK_SIZE, shmem_phase2, stream_>>>( \
            batch_size_, dim_, tmp_write_val, h_work_cur, h_tape_final, \
            write_attn_t, scale)

    // Process each timestep
    for (int t = 0; t < seq_len; ++t) {
        const __nv_bfloat16* x_proj_t = x_proj + t * BD;
        const __nv_bfloat16* h_work_prev = (t == 0) ? h_work_init : (h_work_out + (t - 1) * BD);
        __nv_bfloat16* h_work_cur = h_work_out + t * BD;
        __nv_bfloat16* read_attn_t = read_attn + t * batch_size_ * n_slots_;
        __nv_bfloat16* write_attn_t = write_attn + t * batch_size_ * n_slots_;

        // cuBLAS GEMM for W_h @ h_work_prev
        cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha,
            W_h, CUDA_R_16BF, dim_, h_work_prev, CUDA_R_16BF, dim_,
            &beta, tmp_Rh, CUDA_R_16BF, dim_,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // Phase 1: read attention + h_work update
        switch (n_slots_) {
            case 8:  LAUNCH_E23_PHASE1(8);  break;
            case 16: LAUNCH_E23_PHASE1(16); break;
            case 32: LAUNCH_E23_PHASE1(32); break;
            case 64: LAUNCH_E23_PHASE1(64); break;
            default:
                fprintf(stderr, "E23 CUDA: unsupported n_slots=%d\n", n_slots_);
        }

        // cuBLAS GEMM for W_write @ h_work_new
        cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha,
            W_write, CUDA_R_16BF, dim_, h_work_cur, CUDA_R_16BF, dim_,
            &beta, tmp_write_val, CUDA_R_16BF, dim_,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // Phase 2: write attention + tape update
        switch (n_slots_) {
            case 8:  LAUNCH_E23_PHASE2(8);  break;
            case 16: LAUNCH_E23_PHASE2(16); break;
            case 32: LAUNCH_E23_PHASE2(32); break;
            case 64: LAUNCH_E23_PHASE2(64); break;
        }

        // Save tape state if training
        if (training_ && h_tape_all) {
            cudaMemcpyAsync(h_tape_all + (t + 1) * batch_size_ * n_slots_ * dim_,
                h_tape_final, batch_size_ * n_slots_ * dim_ * sizeof(__nv_bfloat16),
                cudaMemcpyDeviceToDevice, stream_);
        }
    }

    #undef LAUNCH_E23_PHASE1
    #undef LAUNCH_E23_PHASE2
}


// =============================================================================
// E23 Backward
// =============================================================================

template<typename T>
DualMemoryElmanBackward<T>::DualMemoryElmanBackward(
    int batch_size, int n_slots, int dim,
    const cublasHandle_t& blas_handle, const cudaStream_t& stream)
    : batch_size_(batch_size), n_slots_(n_slots), dim_(dim),
      stream_(stream), blas_handle_(blas_handle) {}

template<>
void DualMemoryElmanBackward<__nv_bfloat16>::Run(
    int seq_len,
    const __nv_bfloat16* h_work_all,    // [T, B, D]
    const __nv_bfloat16* h_work_init,   // [B, D] - initial working memory (for t=0)
    const __nv_bfloat16* h_tape_all,    // [T+1, B, N, D]
    const __nv_bfloat16* read_attn,     // [T, B, N]
    const __nv_bfloat16* write_attn,    // [T, B, N]
    const __nv_bfloat16* W_h,
    const __nv_bfloat16* W_write,
    const __nv_bfloat16* d_h_work_out,  // [T, B, D]
    const __nv_bfloat16* d_h_tape_final,
    __nv_bfloat16* dx_proj,             // [T, B, D]
    __nv_bfloat16* d_pre_act_all,       // [T, B, D] - workspace for kernel
    __nv_bfloat16* d_write_val_all,     // [T, B, D] - workspace for kernel
    float* db_h,                        // [D] - accumulated in kernel
    __nv_bfloat16* d_h_tape,            // [B, N, D]
    float* dW_h,                        // [D, D] - computed via GEMM
    float* dW_write                     // [D, D] - computed via GEMM
) {
    const int num_blocks = batch_size_;
    const int TB = seq_len * batch_size_;
    const int BD = batch_size_ * dim_;
    const int BN = batch_size_ * n_slots_;
    const float alpha_one = 1.0f;
    const float beta_zero = 0.0f;
    const float beta_one = 1.0f;

    // Allocate workspace for d_h_work [B, D]
    __nv_bfloat16* d_h_work;
    cudaMalloc(&d_h_work, BD * sizeof(__nv_bfloat16));

    // Shared memory sizes for backward phases
    // Phase1: attn_sh(N_SLOTS)
    const size_t shmem_bwd_phase1 = n_slots_ * sizeof(float);
    // Phase2: no shared memory needed
    const size_t shmem_bwd_phase2 = 0;
    // Phase3: attn_sh(N_SLOTS) + d_pre_act_sh(DIM) + h_work_prev_sh(DIM) + d_read_attn_sh(N_SLOTS) + d_read_scores_sh(N_SLOTS)
    const size_t shmem_bwd_phase3 = (2 * dim_ + 3 * n_slots_) * sizeof(float);
    // Init: no shared memory needed
    const size_t shmem_bwd_init = 0;

    // Macro to dispatch phase kernels based on n_slots
    #define LAUNCH_E23_BWD_INIT(N) \
        E23BackwardInit_BF16<N><<<num_blocks, E23_BLOCK_SIZE, shmem_bwd_init, stream_>>>( \
            batch_size_, dim_, d_h_tape_final, d_h_tape, d_h_work)

    #define LAUNCH_E23_BWD_PHASE1(N) \
        E23BackwardPhase1_BF16<N><<<num_blocks, E23_BLOCK_SIZE, shmem_bwd_phase1, stream_>>>( \
            batch_size_, dim_, write_attn_t, d_h_work_out_t, d_h_work, d_h_tape, d_write_val_t)

    #define LAUNCH_E23_BWD_PHASE2(N) \
        E23BackwardPhase2_BF16<N><<<num_blocks, E23_BLOCK_SIZE, shmem_bwd_phase2, stream_>>>( \
            batch_size_, dim_, h_work_t, d_h_work, dx_proj_t, d_pre_act_t, db_h)

    #define LAUNCH_E23_BWD_PHASE3(N) \
        E23BackwardPhase3_BF16<N><<<num_blocks, E23_BLOCK_SIZE, shmem_bwd_phase3, stream_>>>( \
            batch_size_, dim_, read_attn_t, d_pre_act_t, h_tape_t_ptr, h_work_prev_t, scale, d_h_tape, d_h_work)

    // Initialize d_h_tape and d_h_work
    switch (n_slots_) {
        case 8:  LAUNCH_E23_BWD_INIT(8);  break;
        case 16: LAUNCH_E23_BWD_INIT(16); break;
        case 32: LAUNCH_E23_BWD_INIT(32); break;
        case 64: LAUNCH_E23_BWD_INIT(64); break;
        default:
            fprintf(stderr, "E23 CUDA backward: unsupported n_slots=%d\n", n_slots_);
            cudaFree(d_h_work);
            return;
    }

    // Attention scale
    const float scale = 1.0f / sqrtf(static_cast<float>(dim_));
    const int BND = batch_size_ * n_slots_ * dim_;

    // Backward pass: iterate in reverse with cuBLAS GEMMs between phases
    for (int t = seq_len - 1; t >= 0; t--) {
        // Pointers for this timestep
        const __nv_bfloat16* write_attn_t = write_attn + t * BN;
        const __nv_bfloat16* read_attn_t = read_attn + t * BN;
        const __nv_bfloat16* d_h_work_out_t = d_h_work_out + t * BD;
        const __nv_bfloat16* h_work_t = h_work_all + t * BD;
        __nv_bfloat16* dx_proj_t = dx_proj + t * BD;
        __nv_bfloat16* d_pre_act_t = d_pre_act_all + t * BD;
        __nv_bfloat16* d_write_val_t = d_write_val_all + t * BD;

        // Tape state at timestep t (BEFORE update)
        const __nv_bfloat16* h_tape_t_ptr = h_tape_all + t * BND;

        // h_work at timestep t-1 (needed for Phase3)
        const __nv_bfloat16* h_work_prev_t = (t > 0) ? (h_work_all + (t - 1) * BD) : h_work_init;

        // Phase 1: Add output gradient, compute d_write_val, update d_h_tape for write
        switch (n_slots_) {
            case 8:  LAUNCH_E23_BWD_PHASE1(8);  break;
            case 16: LAUNCH_E23_BWD_PHASE1(16); break;
            case 32: LAUNCH_E23_BWD_PHASE1(32); break;
            case 64: LAUNCH_E23_BWD_PHASE1(64); break;
        }

        // cuBLAS: d_h_work += W_write.T @ d_write_val
        // In cuBLAS terms: d_h_work = W_write @ d_write_val + d_h_work (with transposed W_write)
        // W_write is [D, D], d_write_val is [B, D] -> [D, B] in col-major
        // Result is [D, B] = d_h_work
        cublasGemmEx(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,  // W_write transposed
            dim_, batch_size_, dim_,
            &alpha_one,
            W_write, CUDA_R_16BF, dim_,
            d_write_val_t, CUDA_R_16BF, dim_,
            &beta_one,  // Add to existing d_h_work
            d_h_work, CUDA_R_16BF, dim_,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
        );

        // Phase 2: Tanh backward, compute d_pre_act
        switch (n_slots_) {
            case 8:  LAUNCH_E23_BWD_PHASE2(8);  break;
            case 16: LAUNCH_E23_BWD_PHASE2(16); break;
            case 32: LAUNCH_E23_BWD_PHASE2(32); break;
            case 64: LAUNCH_E23_BWD_PHASE2(64); break;
        }

        // cuBLAS: d_h_work = W_h.T @ d_pre_act (recurrent gradient for next iteration)
        cublasGemmEx(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,  // W_h transposed
            dim_, batch_size_, dim_,
            &alpha_one,
            W_h, CUDA_R_16BF, dim_,
            d_pre_act_t, CUDA_R_16BF, dim_,
            &beta_zero,  // Replace d_h_work
            d_h_work, CUDA_R_16BF, dim_,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
        );

        // Phase 3: Read attention backward, update d_h_tape
        switch (n_slots_) {
            case 8:  LAUNCH_E23_BWD_PHASE3(8);  break;
            case 16: LAUNCH_E23_BWD_PHASE3(16); break;
            case 32: LAUNCH_E23_BWD_PHASE3(32); break;
            case 64: LAUNCH_E23_BWD_PHASE3(64); break;
        }
    }

    #undef LAUNCH_E23_BWD_INIT
    #undef LAUNCH_E23_BWD_PHASE1
    #undef LAUNCH_E23_BWD_PHASE2
    #undef LAUNCH_E23_BWD_PHASE3

    // Free workspace
    cudaFree(d_h_work);

    // Compute weight gradients via cuBLAS GEMM
    // dW_write = d_write_val_all.T @ h_work_all = [D, T*B] @ [T*B, D] = [D, D]
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

    // dW_h = d_pre_act_all.T @ h_work_prev_all
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
            &beta_zero,
            dW_h, CUDA_R_32F, dim_,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
        );
    }
}

// Explicit instantiations
template class DualMemoryElmanForward<__nv_bfloat16>;
template class DualMemoryElmanBackward<__nv_bfloat16>;

}}}  // namespace hasty::v0::elman_ladder

// =============================================================================
// OPTIMIZED E23: Uses cuBLAS batched GEMM for attention (tensor cores)
// =============================================================================

namespace e23_opt {

// Optimized softmax kernel - one thread per batch element
template<int N_SLOTS>
__global__ void OptSoftmax_BF16(
    const int batch_size,
    const __nv_bfloat16* __restrict__ scores,  // [B, N]
    __nv_bfloat16* __restrict__ attn,          // [B, N]
    const float scale
) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    const __nv_bfloat16* s = scores + b * N_SLOTS;
    __nv_bfloat16* a = attn + b * N_SLOTS;

    // Load and find max
    float vals[N_SLOTS];
    float max_v = -1e30f;
    #pragma unroll
    for (int n = 0; n < N_SLOTS; n++) {
        vals[n] = __bfloat162float(s[n]) * scale;
        max_v = fmaxf(max_v, vals[n]);
    }

    // Exp and sum
    float sum_exp = 0.0f;
    #pragma unroll
    for (int n = 0; n < N_SLOTS; n++) {
        vals[n] = expf(vals[n] - max_v);
        sum_exp += vals[n];
    }

    // Normalize
    const float inv = 1.0f / sum_exp;
    #pragma unroll
    for (int n = 0; n < N_SLOTS; n++) {
        a[n] = __float2bfloat16(vals[n] * inv);
    }
}

// Fused tanh update: h_new = tanh(Rh + Wx + read + b)
__global__ void OptTanhUpdate_BF16(
    const int size,  // B * D
    const int dim,
    const __nv_bfloat16* __restrict__ Rh,
    const __nv_bfloat16* __restrict__ Wx,
    const __nv_bfloat16* __restrict__ read,
    const __nv_bfloat16* __restrict__ b_h,
    __nv_bfloat16* __restrict__ h_out
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    const int d = idx % dim;
    float v = __bfloat162float(Rh[idx]) + __bfloat162float(Wx[idx])
            + __bfloat162float(read[idx]) + __bfloat162float(b_h[d]);
    h_out[idx] = __float2bfloat16(tanhf(v));
}

// Tape update: h_tape = (1-attn)*h_tape + attn*write_val
template<int N_SLOTS>
__global__ void OptTapeUpdate_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ attn,       // [B, N]
    const __nv_bfloat16* __restrict__ write_val,  // [B, D]
    __nv_bfloat16* __restrict__ h_tape            // [B, N, D]
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N_SLOTS * dim;
    if (idx >= total) return;

    const int b = idx / (N_SLOTS * dim);
    const int n = (idx / dim) % N_SLOTS;
    const int d = idx % dim;

    const float a = __bfloat162float(attn[b * N_SLOTS + n]);
    const float t = __bfloat162float(h_tape[idx]);
    const float w = __bfloat162float(write_val[b * dim + d]);
    h_tape[idx] = __float2bfloat16((1.0f - a) * t + a * w);
}

} // namespace e23_opt

namespace hasty { namespace v0 { namespace elman_ladder {

// Optimized forward implementation
template<>
void DualMemoryElmanForwardOpt<__nv_bfloat16>::Run(
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
    const float alpha = 1.0f, beta = 0.0f;
    const float scale = 1.0f / sqrtf(static_cast<float>(dim_));

    const int BD = batch_size_ * dim_;
    const int BN = batch_size_ * n_slots_;
    const int BND = batch_size_ * n_slots_ * dim_;

    // Workspace: tmp_Rh[BD], tmp_write_val[BD], tmp_scores[BN], tmp_read[BD]
    __nv_bfloat16* tmp_Rh = workspace;
    __nv_bfloat16* tmp_write_val = tmp_Rh + BD;
    __nv_bfloat16* tmp_scores = tmp_write_val + BD;
    __nv_bfloat16* tmp_read = tmp_scores + BN;

    // Initialize tape
    cudaMemcpyAsync(h_tape_final, h_tape_init, BND * sizeof(__nv_bfloat16),
                    cudaMemcpyDeviceToDevice, stream_);
    if (training_ && h_tape_all) {
        cudaMemcpyAsync(h_tape_all, h_tape_init, BND * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToDevice, stream_);
    }

    const int threads = 256;

    for (int t = 0; t < seq_len; ++t) {
        const __nv_bfloat16* x_proj_t = x_proj + t * BD;
        const __nv_bfloat16* h_work_prev = (t == 0) ? h_work_init : (h_work_out + (t - 1) * BD);
        __nv_bfloat16* h_work_cur = h_work_out + t * BD;
        __nv_bfloat16* read_attn_t = read_attn + t * BN;
        __nv_bfloat16* write_attn_t = write_attn + t * BN;

        // 1. W_h @ h_work -> tmp_Rh
        cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha,
            W_h, CUDA_R_16BF, dim_, h_work_prev, CUDA_R_16BF, dim_,
            &beta, tmp_Rh, CUDA_R_16BF, dim_,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // 2. Read attention scores: h_tape @ h_work^T -> [B, N]
        // Use strided batched GEMM: for each b, [N,D] @ [D,1] -> [N,1]
        cublasGemmStridedBatchedEx(blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            n_slots_, 1, dim_,
            &alpha,
            h_tape_final, CUDA_R_16BF, dim_, n_slots_ * dim_,
            h_work_prev, CUDA_R_16BF, dim_, dim_,
            &beta,
            tmp_scores, CUDA_R_16BF, n_slots_, n_slots_,
            batch_size_,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // 3. Softmax
        int sm_blocks = (batch_size_ + threads - 1) / threads;
        switch (n_slots_) {
            case 8:  e23_opt::OptSoftmax_BF16<8><<<sm_blocks, threads, 0, stream_>>>(batch_size_, tmp_scores, read_attn_t, scale);  break;
            case 16: e23_opt::OptSoftmax_BF16<16><<<sm_blocks, threads, 0, stream_>>>(batch_size_, tmp_scores, read_attn_t, scale); break;
            case 32: e23_opt::OptSoftmax_BF16<32><<<sm_blocks, threads, 0, stream_>>>(batch_size_, tmp_scores, read_attn_t, scale); break;
            case 64: e23_opt::OptSoftmax_BF16<64><<<sm_blocks, threads, 0, stream_>>>(batch_size_, tmp_scores, read_attn_t, scale); break;
        }

        // 4. Weighted read: attn @ h_tape -> [B, D]
        // For each b: [1,N] @ [N,D] -> [1,D], i.e., [D,N] @ [N,1] in col-major
        cublasGemmStridedBatchedEx(blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, 1, n_slots_,
            &alpha,
            h_tape_final, CUDA_R_16BF, dim_, n_slots_ * dim_,
            read_attn_t, CUDA_R_16BF, n_slots_, n_slots_,
            &beta,
            tmp_read, CUDA_R_16BF, dim_, dim_,
            batch_size_,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // 5. Update: h_work_new = tanh(Rh + Wx + read + b)
        int up_blocks = (BD + threads - 1) / threads;
        e23_opt::OptTanhUpdate_BF16<<<up_blocks, threads, 0, stream_>>>(
            BD, dim_, tmp_Rh, x_proj_t, tmp_read, b_h, h_work_cur);

        // 6. W_write @ h_work_new -> write_val
        cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha,
            W_write, CUDA_R_16BF, dim_, h_work_cur, CUDA_R_16BF, dim_,
            &beta, tmp_write_val, CUDA_R_16BF, dim_,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // 7. Write attention scores
        cublasGemmStridedBatchedEx(blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            n_slots_, 1, dim_,
            &alpha,
            h_tape_final, CUDA_R_16BF, dim_, n_slots_ * dim_,
            h_work_cur, CUDA_R_16BF, dim_, dim_,
            &beta,
            tmp_scores, CUDA_R_16BF, n_slots_, n_slots_,
            batch_size_,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // 8. Softmax
        switch (n_slots_) {
            case 8:  e23_opt::OptSoftmax_BF16<8><<<sm_blocks, threads, 0, stream_>>>(batch_size_, tmp_scores, write_attn_t, scale);  break;
            case 16: e23_opt::OptSoftmax_BF16<16><<<sm_blocks, threads, 0, stream_>>>(batch_size_, tmp_scores, write_attn_t, scale); break;
            case 32: e23_opt::OptSoftmax_BF16<32><<<sm_blocks, threads, 0, stream_>>>(batch_size_, tmp_scores, write_attn_t, scale); break;
            case 64: e23_opt::OptSoftmax_BF16<64><<<sm_blocks, threads, 0, stream_>>>(batch_size_, tmp_scores, write_attn_t, scale); break;
        }

        // 9. Tape update
        int tape_blocks = (BND + threads - 1) / threads;
        switch (n_slots_) {
            case 8:  e23_opt::OptTapeUpdate_BF16<8><<<tape_blocks, threads, 0, stream_>>>(batch_size_, dim_, write_attn_t, tmp_write_val, h_tape_final);  break;
            case 16: e23_opt::OptTapeUpdate_BF16<16><<<tape_blocks, threads, 0, stream_>>>(batch_size_, dim_, write_attn_t, tmp_write_val, h_tape_final); break;
            case 32: e23_opt::OptTapeUpdate_BF16<32><<<tape_blocks, threads, 0, stream_>>>(batch_size_, dim_, write_attn_t, tmp_write_val, h_tape_final); break;
            case 64: e23_opt::OptTapeUpdate_BF16<64><<<tape_blocks, threads, 0, stream_>>>(batch_size_, dim_, write_attn_t, tmp_write_val, h_tape_final); break;
        }

        // Save tape if training
        if (training_ && h_tape_all) {
            cudaMemcpyAsync(h_tape_all + (t + 1) * BND, h_tape_final,
                BND * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream_);
        }
    }
}

// Constructor
template<>
DualMemoryElmanForwardOpt<__nv_bfloat16>::DualMemoryElmanForwardOpt(
    bool training, int batch_size, int n_slots, int dim,
    const cublasHandle_t& blas_handle, const cudaStream_t& stream)
    : training_(training), batch_size_(batch_size), n_slots_(n_slots), dim_(dim),
      stream_(stream), blas_handle_(blas_handle) {}

// Explicit instantiation
template class DualMemoryElmanForwardOpt<__nv_bfloat16>;

}}}  // namespace hasty::v0::elman_ladder
