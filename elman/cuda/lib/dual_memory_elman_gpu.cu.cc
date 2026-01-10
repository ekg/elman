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

// Max dimensions for shared memory sizing
constexpr int N_SLOTS_MAX = 64;
constexpr int DIM_MAX = 1024;

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
 */
template<int N_SLOTS, int DIM>
__global__ void E23Phase1Kernel_BF16(
    const int batch_size,
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

    // Shared memory for attention scores
    __shared__ float attn_sh[N_SLOTS];
    __shared__ float h_work_sh[DIM];

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

    // Compute h_work_new: tanh(Rh + x_proj_t + read_val + b_h)
    for (int d = tid; d < DIM; d += E23_BLOCK_SIZE) {
        // Weighted read: read[d] = sum_n attn[n] * h_tape[n, d]
        float read_d = 0.0f;
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
 */
template<int N_SLOTS, int DIM>
__global__ void E23Phase2Kernel_BF16(
    const int batch_size,
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

    // Shared memory
    __shared__ float attn_sh[N_SLOTS];
    __shared__ float h_work_sh[DIM];
    __shared__ float write_val_sh[DIM];

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
 * E23 Backward Kernel (BF16) - Global Memory Version
 *
 * Uses global memory for h_tape to support large N×D configurations.
 * Shared memory only for: h_work[D], d_h_work[D], attn[N], d_write_val[D]
 */
template<int N_SLOTS, int DIM>
__global__ void E23BackwardKernel_BF16(
    const int seq_len,
    const int batch_size,
    // Forward saved tensors
    const __nv_bfloat16* __restrict__ x_proj,      // [B, T, D]
    const __nv_bfloat16* __restrict__ h_work_all,  // [B, T, D]
    const __nv_bfloat16* __restrict__ h_tape_all,  // [T+1, B, N, D]
    const __nv_bfloat16* __restrict__ read_attn,   // [B, T, N]
    const __nv_bfloat16* __restrict__ write_attn,  // [B, T, N]
    // Weights
    const __nv_bfloat16* __restrict__ W_h,
    const __nv_bfloat16* __restrict__ W_write,
    // Gradient inputs
    const __nv_bfloat16* __restrict__ d_h_work_out,  // [B, T, D]
    const __nv_bfloat16* __restrict__ d_h_tape_final,// [B, N, D]
    // Gradient outputs (d_h_tape is in global memory, mutable)
    __nv_bfloat16* __restrict__ dx_proj,      // [B, T, D]
    float* __restrict__ dW_h,                 // [D, D] - accumulated
    float* __restrict__ db_h,                 // [D] - accumulated
    float* __restrict__ dW_write,             // [D, D] - accumulated
    __nv_bfloat16* __restrict__ d_h_tape,     // [B, N, D] - gradient accumulator in global mem
    // Scale
    const float scale
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;

    const int tid = threadIdx.x;

    // Shared memory
    extern __shared__ float smem[];
    float* h_work_sh = smem;                        // [DIM]
    float* d_h_work_sh = h_work_sh + DIM;           // [DIM]
    float* attn_sh = d_h_work_sh + DIM;             // [N_SLOTS]
    float* d_write_val_sh = attn_sh + N_SLOTS;      // [DIM]

    // Global memory pointers for this batch's tape gradients
    __nv_bfloat16* d_h_tape_b = d_h_tape + b * N_SLOTS * DIM;

    // Initialize d_h_tape from final gradient
    for (int i = tid; i < N_SLOTS * DIM; i += E23_BLOCK_SIZE) {
        d_h_tape_b[i] = d_h_tape_final[b * N_SLOTS * DIM + i];
    }

    // Initialize d_h_work to zero
    for (int d = tid; d < DIM; d += E23_BLOCK_SIZE) {
        d_h_work_sh[d] = 0.0f;
    }
    __syncthreads();

    // Backward pass: iterate in reverse
    for (int t = seq_len - 1; t >= 0; t--) {
        // Load current h_work
        for (int d = tid; d < DIM; d += E23_BLOCK_SIZE) {
            h_work_sh[d] = __bfloat162float(h_work_all[b * seq_len * DIM + t * DIM + d]);
        }
        __syncthreads();

        // Add gradient from output
        for (int d = tid; d < DIM; d += E23_BLOCK_SIZE) {
            d_h_work_sh[d] += __bfloat162float(d_h_work_out[b * seq_len * DIM + t * DIM + d]);
        }
        __syncthreads();

        // ============================================
        // BACKWARD THROUGH WRITE
        // ============================================
        // Load write attention for this timestep
        if (tid < N_SLOTS) {
            attn_sh[tid] = __bfloat162float(write_attn[b * seq_len * N_SLOTS + t * N_SLOTS + tid]);
        }
        __syncthreads();

        // h_tape at time t (before write)
        const __nv_bfloat16* h_tape_t = h_tape_all + t * batch_size * N_SLOTS * DIM + b * N_SLOTS * DIM;

        // Gradient through tape update: h_tape = (1-attn)*h_tape_old + attn*write_val
        // d_write_val = attn * d_h_tape
        for (int d = tid; d < DIM; d += E23_BLOCK_SIZE) {
            float d_wv = 0.0f;
            for (int n = 0; n < N_SLOTS; n++) {
                d_wv += attn_sh[n] * __bfloat162float(d_h_tape_b[n * DIM + d]);
            }
            d_write_val_sh[d] = d_wv;
        }
        __syncthreads();

        // Gradient w.r.t. W_write: dW_write += d_write_val @ h_work.T
        for (int i = tid; i < DIM; i += E23_BLOCK_SIZE) {
            float d_wv_i = d_write_val_sh[i];
            for (int j = 0; j < DIM; j++) {
                atomicAdd(&dW_write[i * DIM + j], d_wv_i * h_work_sh[j]);
            }
        }

        // Gradient through W_write @ h_work: d_h_work += W_write.T @ d_write_val
        for (int j = tid; j < DIM; j += E23_BLOCK_SIZE) {
            float d_h = 0.0f;
            for (int i = 0; i < DIM; i++) {
                d_h += __bfloat162float(W_write[i * DIM + j]) * d_write_val_sh[i];
            }
            d_h_work_sh[j] += d_h;
        }
        __syncthreads();

        // Update d_h_tape: d_h_tape_old = (1-attn) * d_h_tape
        for (int i = tid; i < N_SLOTS * DIM; i += E23_BLOCK_SIZE) {
            int n = i / DIM;
            float d_tape = __bfloat162float(d_h_tape_b[i]);
            d_h_tape_b[i] = __float2bfloat16((1.0f - attn_sh[n]) * d_tape);
        }
        __syncthreads();

        // ============================================
        // BACKWARD THROUGH H_WORK UPDATE
        // ============================================
        // h_work = tanh(pre_act)
        // d_pre_act = d_h_work * (1 - h_work^2)
        for (int d = tid; d < DIM; d += E23_BLOCK_SIZE) {
            float h = h_work_sh[d];
            float d_pre_act = d_h_work_sh[d] * (1.0f - h * h);

            // Gradient w.r.t. x_proj
            dx_proj[b * seq_len * DIM + t * DIM + d] = __float2bfloat16(d_pre_act);

            // Gradient w.r.t. b_h (accumulated)
            atomicAdd(&db_h[d], d_pre_act);

            // Store d_pre_act in d_write_val_sh for W_h gradient
            d_write_val_sh[d] = d_pre_act;
        }
        __syncthreads();

        // Load h_work_prev for W_h gradient
        if (t > 0) {
            for (int d = tid; d < DIM; d += E23_BLOCK_SIZE) {
                h_work_sh[d] = __bfloat162float(h_work_all[b * seq_len * DIM + (t-1) * DIM + d]);
            }
        } else {
            for (int d = tid; d < DIM; d += E23_BLOCK_SIZE) {
                h_work_sh[d] = 0.0f;
            }
        }
        __syncthreads();

        // Gradient w.r.t. W_h: dW_h += d_pre_act @ h_work_prev.T
        for (int i = tid; i < DIM; i += E23_BLOCK_SIZE) {
            float d_pre_act_i = d_write_val_sh[i];
            for (int j = 0; j < DIM; j++) {
                atomicAdd(&dW_h[i * DIM + j], d_pre_act_i * h_work_sh[j]);
            }
        }

        // Gradient through W_h @ h_work_prev: d_h_work_prev = W_h.T @ d_pre_act
        for (int j = tid; j < DIM; j += E23_BLOCK_SIZE) {
            float d_h_prev = 0.0f;
            for (int k = 0; k < DIM; k++) {
                d_h_prev += d_write_val_sh[k] * __bfloat162float(W_h[k * DIM + j]);
            }
            d_h_work_sh[j] = d_h_prev;
        }
        __syncthreads();

        // ============================================
        // BACKWARD THROUGH READ
        // ============================================
        // (Simplified - gradient through read attention updates d_h_tape and d_h_work)
        // Load read attention
        if (tid < N_SLOTS) {
            attn_sh[tid] = __bfloat162float(read_attn[b * seq_len * N_SLOTS + t * N_SLOTS + tid]);
        }
        __syncthreads();

        // d_pre_act already propagated read contribution via d_h_work
        // Gradient through read = attn @ h_tape goes to d_h_tape
        // Each thread handles unique (n, d) pairs, so no atomics needed
        for (int d = tid; d < DIM; d += E23_BLOCK_SIZE) {
            float d_read_d = d_write_val_sh[d];  // d_pre_act flows through read
            for (int n = 0; n < N_SLOTS; n++) {
                float d_tape = attn_sh[n] * d_read_d;
                // Read-modify-write without atomics (each thread owns unique indices)
                float current = __bfloat162float(d_h_tape_b[n * DIM + d]);
                d_h_tape_b[n * DIM + d] = __float2bfloat16(current + d_tape);
            }
        }
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
    const __nv_bfloat16* x,           // [B, T, D]
    const __nv_bfloat16* W_h,         // [D, D]
    const __nv_bfloat16* W_x,         // [D, D]
    const __nv_bfloat16* b_h,         // [D]
    const __nv_bfloat16* W_write,     // [D, D]
    const __nv_bfloat16* h_tape_init, // [B, N, D]
    const __nv_bfloat16* h_work_init, // [B, D]
    __nv_bfloat16* h_work_out,        // [B, T, D]
    __nv_bfloat16* h_tape_final,      // [B, N, D]
    __nv_bfloat16* h_tape_all,        // [T+1, B, N, D] - tape history for backward (null if inference)
    __nv_bfloat16* read_attn,         // [B, T, N]
    __nv_bfloat16* write_attn,        // [B, T, N]
    __nv_bfloat16* workspace          // Workspace for temporaries
) {
    seq_len_ = seq_len;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    const int BD = batch_size_ * dim_;

    // Workspace layout:
    // - tmp_x: [B, D] - contiguous copy of x[:, t, :] for each timestep
    // - tmp_Wx: [B, D] - W_x @ x[t] computed per timestep
    // - tmp_Rh: [B, D] - W_h @ h_work computed per timestep
    // - tmp_write_val: [B, D]
    __nv_bfloat16* tmp_x = workspace;
    __nv_bfloat16* tmp_Wx = workspace + BD;
    __nv_bfloat16* tmp_Rh = tmp_Wx + BD;
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

    // Shared memory sizes
    size_t smem_phase1 = (N_SLOTS_MAX + DIM_MAX) * sizeof(float);  // attn + h_work
    size_t smem_phase2 = (N_SLOTS_MAX + 2 * DIM_MAX) * sizeof(float);  // attn + h_work + write_val

    // Macro for kernel dispatch
    #define LAUNCH_E23_PHASE1(N, D) \
        E23Phase1Kernel_BF16<N, D><<<num_blocks, E23_BLOCK_SIZE, 0, stream_>>>( \
            batch_size_, tmp_Rh, tmp_Wx, b_h, h_tape_final, h_work_prev, \
            h_work_cur, read_attn_t, scale)

    #define LAUNCH_E23_PHASE2(N, D) \
        E23Phase2Kernel_BF16<N, D><<<num_blocks, E23_BLOCK_SIZE, 0, stream_>>>( \
            batch_size_, tmp_write_val, h_work_cur, h_tape_final, \
            write_attn_t, scale)

    // Process each timestep
    for (int t = 0; t < seq_len; ++t) {
        // x is [B, T, D] in row-major, so x[b, t, d] is at offset (b * seq_len + t) * dim_ + d
        // For timestep t, we need x[:, t, :] which is NOT contiguous
        // Copy x[:, t, :] to tmp_x for contiguous access
        for (int b = 0; b < batch_size_; ++b) {
            cudaMemcpyAsync(
                tmp_x + b * dim_,
                x + (b * seq_len + t) * dim_,
                dim_ * sizeof(__nv_bfloat16),
                cudaMemcpyDeviceToDevice, stream_);
        }

        const __nv_bfloat16* h_work_prev = (t == 0) ? h_work_init : (h_work_out + (t - 1) * BD);
        __nv_bfloat16* h_work_cur = h_work_out + t * BD;
        __nv_bfloat16* read_attn_t = read_attn + t * batch_size_ * n_slots_;
        __nv_bfloat16* write_attn_t = write_attn + t * batch_size_ * n_slots_;

        // Step 2a: cuBLAS GEMM for x_t @ W_x^T
        // Now tmp_x contains contiguous [B, D] data
        cublasGemmEx(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha,
            W_x, CUDA_R_16BF, dim_,
            tmp_x, CUDA_R_16BF, dim_,
            &beta,
            tmp_Wx, CUDA_R_16BF, dim_,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
        );

        // Step 2b: cuBLAS GEMM for W_h @ h_work_prev
        cublasGemmEx(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha,
            W_h, CUDA_R_16BF, dim_,
            h_work_prev, CUDA_R_16BF, dim_,
            &beta,
            tmp_Rh, CUDA_R_16BF, dim_,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
        );

        // Step 2b: Phase 1 kernel - read attention + h_work update
        if (n_slots_ == 8 && dim_ == 256) { LAUNCH_E23_PHASE1(8, 256); }
        else if (n_slots_ == 8 && dim_ == 512) { LAUNCH_E23_PHASE1(8, 512); }
        else if (n_slots_ == 8 && dim_ == 768) { LAUNCH_E23_PHASE1(8, 768); }
        else if (n_slots_ == 8 && dim_ == 1024) { LAUNCH_E23_PHASE1(8, 1024); }
        else if (n_slots_ == 16 && dim_ == 256) { LAUNCH_E23_PHASE1(16, 256); }
        else if (n_slots_ == 32 && dim_ == 512) { LAUNCH_E23_PHASE1(32, 512); }
        else if (n_slots_ == 64 && dim_ == 256) { LAUNCH_E23_PHASE1(64, 256); }
        else if (n_slots_ == 64 && dim_ == 512) { LAUNCH_E23_PHASE1(64, 512); }
        else if (n_slots_ == 64 && dim_ == 768) { LAUNCH_E23_PHASE1(64, 768); }
        else if (n_slots_ == 64 && dim_ == 1024) { LAUNCH_E23_PHASE1(64, 1024); }
        else {
            fprintf(stderr, "E23 CUDA: unsupported n_slots=%d, dim=%d\n", n_slots_, dim_);
        }

        // Step 2c: cuBLAS GEMM for W_write @ h_work_new
        cublasGemmEx(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha,
            W_write, CUDA_R_16BF, dim_,
            h_work_cur, CUDA_R_16BF, dim_,
            &beta,
            tmp_write_val, CUDA_R_16BF, dim_,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
        );

        // Step 2d: Phase 2 kernel - write attention + tape update
        if (n_slots_ == 8 && dim_ == 256) { LAUNCH_E23_PHASE2(8, 256); }
        else if (n_slots_ == 8 && dim_ == 512) { LAUNCH_E23_PHASE2(8, 512); }
        else if (n_slots_ == 8 && dim_ == 768) { LAUNCH_E23_PHASE2(8, 768); }
        else if (n_slots_ == 8 && dim_ == 1024) { LAUNCH_E23_PHASE2(8, 1024); }
        else if (n_slots_ == 16 && dim_ == 256) { LAUNCH_E23_PHASE2(16, 256); }
        else if (n_slots_ == 32 && dim_ == 512) { LAUNCH_E23_PHASE2(32, 512); }
        else if (n_slots_ == 64 && dim_ == 256) { LAUNCH_E23_PHASE2(64, 256); }
        else if (n_slots_ == 64 && dim_ == 512) { LAUNCH_E23_PHASE2(64, 512); }
        else if (n_slots_ == 64 && dim_ == 768) { LAUNCH_E23_PHASE2(64, 768); }
        else if (n_slots_ == 64 && dim_ == 1024) { LAUNCH_E23_PHASE2(64, 1024); }

        // Save tape state if training
        if (training_ && h_tape_all) {
            cudaMemcpyAsync(
                h_tape_all + (t + 1) * batch_size_ * n_slots_ * dim_,
                h_tape_final,
                batch_size_ * n_slots_ * dim_ * sizeof(__nv_bfloat16),
                cudaMemcpyDeviceToDevice, stream_
            );
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
    const __nv_bfloat16* x_proj,
    const __nv_bfloat16* h_work_all,
    const __nv_bfloat16* h_tape_all,
    const __nv_bfloat16* read_attn,
    const __nv_bfloat16* write_attn,
    const __nv_bfloat16* W_h,
    const __nv_bfloat16* W_write,
    const __nv_bfloat16* d_h_work_out,
    const __nv_bfloat16* d_h_tape_final,
    __nv_bfloat16* dx_proj,
    float* dW_h,
    float* db_h,
    float* dW_write,
    __nv_bfloat16* d_h_tape
) {
    const float scale = 1.0f / sqrtf(static_cast<float>(dim_));
    int num_blocks = batch_size_;

    // Shared memory for backward
    size_t smem_size = (3 * dim_ + n_slots_) * sizeof(float);

    #define LAUNCH_E23_BACKWARD(N, D) \
        E23BackwardKernel_BF16<N, D><<<num_blocks, E23_BLOCK_SIZE, smem_size, stream_>>>( \
            seq_len, batch_size_, \
            x_proj, h_work_all, h_tape_all, read_attn, write_attn, \
            W_h, W_write, d_h_work_out, d_h_tape_final, \
            dx_proj, dW_h, db_h, dW_write, d_h_tape, \
            scale)

    if (n_slots_ == 8 && dim_ == 256) { LAUNCH_E23_BACKWARD(8, 256); }
    else if (n_slots_ == 8 && dim_ == 512) { LAUNCH_E23_BACKWARD(8, 512); }
    else if (n_slots_ == 8 && dim_ == 768) { LAUNCH_E23_BACKWARD(8, 768); }
    else if (n_slots_ == 8 && dim_ == 1024) { LAUNCH_E23_BACKWARD(8, 1024); }
    else if (n_slots_ == 16 && dim_ == 256) { LAUNCH_E23_BACKWARD(16, 256); }
    else if (n_slots_ == 32 && dim_ == 512) { LAUNCH_E23_BACKWARD(32, 512); }
    else if (n_slots_ == 64 && dim_ == 256) { LAUNCH_E23_BACKWARD(64, 256); }
    else if (n_slots_ == 64 && dim_ == 512) { LAUNCH_E23_BACKWARD(64, 512); }
    else if (n_slots_ == 64 && dim_ == 768) { LAUNCH_E23_BACKWARD(64, 768); }
    else if (n_slots_ == 64 && dim_ == 1024) { LAUNCH_E23_BACKWARD(64, 1024); }
    else {
        fprintf(stderr, "E23 CUDA backward: unsupported n_slots=%d, dim=%d\n", n_slots_, dim_);
    }

    #undef LAUNCH_E23_BACKWARD
}

// Explicit instantiations
template class DualMemoryElmanForward<__nv_bfloat16>;
template class DualMemoryElmanBackward<__nv_bfloat16>;

}}}  // namespace hasty::v0::elman_ladder
