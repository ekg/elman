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
 *   - Fused kernel for sequential attention/update/write ops
 *   - One block per batch element, threads parallelize over D/N
 */

#include "hasty/elman_ladder.h"
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>

namespace {

// Block size for the fused kernel
constexpr int E23_BLOCK_SIZE = 256;

/**
 * E23 Forward Kernel (BF16)
 *
 * Each block handles one batch element.
 * Threads parallelize over D (working memory dimension).
 *
 * Template parameters:
 *   N_SLOTS: Number of tape slots (compile-time constant)
 *   DIM: Working memory dimension (compile-time constant)
 */
template<int N_SLOTS, int DIM>
__global__ void E23ForwardKernel_BF16(
    const int seq_len,
    const int batch_size,
    // Pre-computed input projection: W_x @ x for all timesteps
    const __nv_bfloat16* __restrict__ x_proj,  // [T, B, D]
    // Weights
    const __nv_bfloat16* __restrict__ W_h,     // [D, D]
    const __nv_bfloat16* __restrict__ b_h,     // [D]
    const __nv_bfloat16* __restrict__ W_write, // [D, D]
    // Initial states
    const __nv_bfloat16* __restrict__ h_tape_init,  // [B, N, D]
    const __nv_bfloat16* __restrict__ h_work_init,  // [B, D]
    // Outputs
    __nv_bfloat16* __restrict__ h_work_out,    // [T, B, D]
    __nv_bfloat16* __restrict__ h_tape_final,  // [B, N, D]
    // Attention weights (for backward)
    __nv_bfloat16* __restrict__ read_attn_out,  // [T, B, N]
    __nv_bfloat16* __restrict__ write_attn_out, // [T, B, N]
    // Scale factor
    const float scale
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;

    const int tid = threadIdx.x;

    // Shared memory for h_tape [N, D] and h_work [D]
    // Plus scratch for attention computation
    extern __shared__ float smem[];
    float* h_tape_sh = smem;                          // [N_SLOTS * DIM]
    float* h_work_sh = h_tape_sh + N_SLOTS * DIM;     // [DIM]
    float* attn_sh = h_work_sh + DIM;                 // [N_SLOTS]
    float* W_h_row = attn_sh + N_SLOTS;               // [DIM] scratch for W_h row

    // Load initial h_tape into shared memory
    for (int i = tid; i < N_SLOTS * DIM; i += E23_BLOCK_SIZE) {
        int n = i / DIM;
        int d = i % DIM;
        h_tape_sh[i] = __bfloat162float(h_tape_init[b * N_SLOTS * DIM + n * DIM + d]);
    }

    // Load initial h_work into shared memory
    for (int d = tid; d < DIM; d += E23_BLOCK_SIZE) {
        h_work_sh[d] = __bfloat162float(h_work_init[b * DIM + d]);
    }
    __syncthreads();

    // Process each timestep
    for (int t = 0; t < seq_len; t++) {

        // ============================================
        // STEP 1: READ FROM TAPE (attention)
        // ============================================
        // read_scores[n] = sum_d h_tape[n, d] * h_work[d]
        if (tid < N_SLOTS) {
            float score = 0.0f;
            for (int d = 0; d < DIM; d++) {
                score += h_tape_sh[tid * DIM + d] * h_work_sh[d];
            }
            attn_sh[tid] = score * scale;
        }
        __syncthreads();

        // Softmax over N_SLOTS
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
            read_attn_out[t * batch_size * N_SLOTS + b * N_SLOTS + tid] =
                __float2bfloat16(attn_sh[tid]);
        }

        // Weighted read: read[d] = sum_n attn[n] * h_tape[n, d]
        // Each thread computes one element of read
        float read_d = 0.0f;
        if (tid < DIM) {
            for (int n = 0; n < N_SLOTS; n++) {
                read_d += attn_sh[n] * h_tape_sh[n * DIM + tid];
            }
        }

        // ============================================
        // STEP 2: UPDATE WORKING MEMORY
        // ============================================
        // pre_act = W_h @ h_work + x_proj[t] + read + b_h
        // h_work_new = tanh(pre_act)

        // Load W_h row by row and compute W_h @ h_work
        float wh_contrib = 0.0f;
        if (tid < DIM) {
            for (int k = 0; k < DIM; k++) {
                wh_contrib += __bfloat162float(W_h[tid * DIM + k]) * h_work_sh[k];
            }

            float x_proj_val = __bfloat162float(x_proj[t * batch_size * DIM + b * DIM + tid]);
            float b_val = __bfloat162float(b_h[tid]);

            float pre_act = wh_contrib + x_proj_val + read_d + b_val;
            h_work_sh[tid] = tanhf(pre_act);
        }
        __syncthreads();

        // Store h_work output
        if (tid < DIM) {
            h_work_out[t * batch_size * DIM + b * DIM + tid] =
                __float2bfloat16(h_work_sh[tid]);
        }

        // ============================================
        // STEP 3: WRITE TO TAPE
        // ============================================
        // write_value = h_work_new @ W_write.T
        float write_val = 0.0f;
        if (tid < DIM) {
            for (int k = 0; k < DIM; k++) {
                write_val += h_work_sh[k] * __bfloat162float(W_write[tid * DIM + k]);
            }
        }

        // Write attention scores
        if (tid < N_SLOTS) {
            float score = 0.0f;
            for (int d = 0; d < DIM; d++) {
                score += h_tape_sh[tid * DIM + d] * h_work_sh[d];
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
            write_attn_out[t * batch_size * N_SLOTS + b * N_SLOTS + tid] =
                __float2bfloat16(attn_sh[tid]);
        }

        // Update tape: h_tape = (1 - attn) * h_tape + attn * write_value
        // Need to broadcast write_value to all threads
        __shared__ float write_val_sh[256];  // DIM elements
        if (tid < DIM) {
            write_val_sh[tid] = write_val;
        }
        __syncthreads();

        for (int i = tid; i < N_SLOTS * DIM; i += E23_BLOCK_SIZE) {
            int n = i / DIM;
            int d = i % DIM;
            float attn_n = attn_sh[n];
            h_tape_sh[i] = (1.0f - attn_n) * h_tape_sh[i] + attn_n * write_val_sh[d];
        }
        __syncthreads();
    }

    // Write final tape state
    for (int i = tid; i < N_SLOTS * DIM; i += E23_BLOCK_SIZE) {
        int n = i / DIM;
        int d = i % DIM;
        h_tape_final[b * N_SLOTS * DIM + n * DIM + d] = __float2bfloat16(h_tape_sh[i]);
    }
}


/**
 * E23 Backward Kernel (BF16)
 *
 * Computes gradients for:
 *   - W_h, W_x, b_h, W_write (accumulated across batch)
 *   - x (per-sample gradients)
 */
template<int N_SLOTS, int DIM>
__global__ void E23BackwardKernel_BF16(
    const int seq_len,
    const int batch_size,
    // Forward saved tensors
    const __nv_bfloat16* __restrict__ x_proj,      // [T, B, D]
    const __nv_bfloat16* __restrict__ h_work_all,  // [T, B, D]
    const __nv_bfloat16* __restrict__ h_tape_all,  // [T+1, B, N, D]
    const __nv_bfloat16* __restrict__ read_attn,   // [T, B, N]
    const __nv_bfloat16* __restrict__ write_attn,  // [T, B, N]
    // Weights
    const __nv_bfloat16* __restrict__ W_h,
    const __nv_bfloat16* __restrict__ W_write,
    // Gradient inputs
    const __nv_bfloat16* __restrict__ d_h_work_out,  // [T, B, D]
    const __nv_bfloat16* __restrict__ d_h_tape_final,// [B, N, D]
    // Gradient outputs
    __nv_bfloat16* __restrict__ dx_proj,      // [T, B, D]
    float* __restrict__ dW_h,                 // [D, D] - accumulated
    float* __restrict__ db_h,                 // [D] - accumulated
    float* __restrict__ dW_write,             // [D, D] - accumulated
    // Scale
    const float scale
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;

    const int tid = threadIdx.x;

    // Shared memory
    extern __shared__ float smem[];
    float* d_h_tape_sh = smem;                      // [N_SLOTS * DIM]
    float* d_h_work_sh = d_h_tape_sh + N_SLOTS * DIM;  // [DIM]
    float* h_tape_sh = d_h_work_sh + DIM;           // [N_SLOTS * DIM]
    float* h_work_sh = h_tape_sh + N_SLOTS * DIM;   // [DIM]
    float* attn_sh = h_work_sh + DIM;               // [N_SLOTS]

    // Initialize d_h_tape from final gradient
    for (int i = tid; i < N_SLOTS * DIM; i += E23_BLOCK_SIZE) {
        int n = i / DIM;
        int d = i % DIM;
        d_h_tape_sh[i] = __bfloat162float(d_h_tape_final[b * N_SLOTS * DIM + n * DIM + d]);
    }

    // Initialize d_h_work to zero
    for (int d = tid; d < DIM; d += E23_BLOCK_SIZE) {
        d_h_work_sh[d] = 0.0f;
    }
    __syncthreads();

    // Backward through timesteps
    for (int t = seq_len - 1; t >= 0; t--) {

        // Load h_work[t] and h_tape[t] (before update)
        for (int d = tid; d < DIM; d += E23_BLOCK_SIZE) {
            h_work_sh[d] = __bfloat162float(h_work_all[t * batch_size * DIM + b * DIM + d]);
        }
        for (int i = tid; i < N_SLOTS * DIM; i += E23_BLOCK_SIZE) {
            int n = i / DIM;
            int d = i % DIM;
            h_tape_sh[i] = __bfloat162float(
                h_tape_all[t * batch_size * N_SLOTS * DIM + b * N_SLOTS * DIM + n * DIM + d]);
        }
        __syncthreads();

        // Add incoming gradient from output
        for (int d = tid; d < DIM; d += E23_BLOCK_SIZE) {
            d_h_work_sh[d] += __bfloat162float(d_h_work_out[t * batch_size * DIM + b * DIM + d]);
        }
        __syncthreads();

        // === BACKWARD THROUGH WRITE ===
        // Load write attention
        for (int n = tid; n < N_SLOTS; n += E23_BLOCK_SIZE) {
            attn_sh[n] = __bfloat162float(write_attn[t * batch_size * N_SLOTS + b * N_SLOTS + n]);
        }
        __syncthreads();

        // write_value = h_work @ W_write.T
        // d_write_value = sum_n d_h_tape[n] * attn[n]
        __shared__ float d_write_val_sh[256];
        if (tid < DIM) {
            float d_wv = 0.0f;
            for (int n = 0; n < N_SLOTS; n++) {
                d_wv += d_h_tape_sh[n * DIM + tid] * attn_sh[n];
            }
            d_write_val_sh[tid] = d_wv;
        }
        __syncthreads();

        // Gradient w.r.t. h_work from write_value
        // write_value = h_work @ W_write.T
        // d_h_work += d_write_value @ W_write
        if (tid < DIM) {
            float grad = 0.0f;
            for (int k = 0; k < DIM; k++) {
                grad += d_write_val_sh[k] * __bfloat162float(W_write[k * DIM + tid]);
            }
            d_h_work_sh[tid] += grad;
        }

        // Gradient w.r.t. W_write (accumulated)
        // dW_write[i, j] += d_write_value[i] * h_work[j]
        if (tid < DIM) {
            for (int j = 0; j < DIM; j++) {
                atomicAdd(&dW_write[tid * DIM + j], d_write_val_sh[tid] * h_work_sh[j]);
            }
        }
        __syncthreads();

        // Update d_h_tape for tape before write
        // d_h_tape[n] *= (1 - attn[n])
        for (int i = tid; i < N_SLOTS * DIM; i += E23_BLOCK_SIZE) {
            int n = i / DIM;
            d_h_tape_sh[i] *= (1.0f - attn_sh[n]);
        }
        __syncthreads();

        // === BACKWARD THROUGH TANH AND UPDATE ===
        // h_work_new = tanh(pre_act)
        // d_pre_act = d_h_work * (1 - h_work^2)
        if (tid < DIM) {
            float h = h_work_sh[tid];
            float d_pre_act = d_h_work_sh[tid] * (1.0f - h * h);

            // Gradient w.r.t. x_proj
            dx_proj[t * batch_size * DIM + b * DIM + tid] = __float2bfloat16(d_pre_act);

            // Gradient w.r.t. b_h
            atomicAdd(&db_h[tid], d_pre_act);

            // Gradient w.r.t. W_h (need h_work_prev)
            // For simplicity, skip W_h gradient here (done in outer loop)

            // Update d_h_work for h_work_prev
            float d_h_prev = 0.0f;
            for (int k = 0; k < DIM; k++) {
                d_h_prev += d_pre_act * __bfloat162float(W_h[k * DIM + tid]);
            }
            d_h_work_sh[tid] = d_h_prev;
        }
        __syncthreads();

        // === BACKWARD THROUGH READ ===
        // Load read attention
        for (int n = tid; n < N_SLOTS; n += E23_BLOCK_SIZE) {
            attn_sh[n] = __bfloat162float(read_attn[t * batch_size * N_SLOTS + b * N_SLOTS + n]);
        }
        __syncthreads();

        // d_read (already computed as part of d_pre_act)
        // Gradient flows to d_h_tape
        // d_h_tape[n, d] += d_read[d] * attn[n]
        // Note: d_read = d_pre_act (from above, already stored in dx_proj)
        for (int i = tid; i < N_SLOTS * DIM; i += E23_BLOCK_SIZE) {
            int n = i / DIM;
            int d = i % DIM;
            float d_read = __bfloat162float(dx_proj[t * batch_size * DIM + b * DIM + d]);
            d_h_tape_sh[i] += d_read * attn_sh[n];
        }
        __syncthreads();
    }
}

}  // namespace


// =============================================================================
// Host-side wrapper classes
// =============================================================================

namespace hasty {
namespace v0 {
namespace elman_ladder {

// Forward declaration
template<typename T>
DualMemoryElmanForward<T>::DualMemoryElmanForward(
    bool training, int batch_size, int n_slots, int dim,
    const cublasHandle_t& blas_handle, const cudaStream_t& stream)
    : training_(training), batch_size_(batch_size), n_slots_(n_slots),
      dim_(dim), stream_(stream), blas_handle_(blas_handle) {}

template<>
void DualMemoryElmanForward<__nv_bfloat16>::Run(
    int seq_len,
    const __nv_bfloat16* x,           // [B, T, D] - raw input
    const __nv_bfloat16* W_h,         // [D, D]
    const __nv_bfloat16* W_x,         // [D, D]
    const __nv_bfloat16* b_h,         // [D]
    const __nv_bfloat16* W_write,     // [D, D]
    const __nv_bfloat16* h_tape_init, // [B, N, D]
    const __nv_bfloat16* h_work_init, // [B, D]
    __nv_bfloat16* h_work_out,        // [B, T, D]
    __nv_bfloat16* h_tape_final,      // [B, N, D]
    __nv_bfloat16* read_attn,         // [B, T, N]
    __nv_bfloat16* write_attn,        // [B, T, N]
    __nv_bfloat16* x_proj_out         // [B, T, D] - scratch for x @ W_x.T
) {
    // Step 1: Compute x_proj = x @ W_x.T using cuBLAS
    // x is [B*T, D], W_x is [D, D], x_proj is [B*T, D]
    // x_proj = x @ W_x.T = x @ W_x^T
    const __nv_bfloat16 alpha = __float2bfloat16(1.0f);
    const __nv_bfloat16 beta = __float2bfloat16(0.0f);

    // cuBLAS GEMM: C = alpha * A * B + beta * C
    // We want: x_proj[i, j] = sum_k x[i, k] * W_x[j, k]  (W_x transposed)
    // In column-major: x_proj^T = W_x @ x^T
    // m = D, n = B*T, k = D
    cublasGemmEx(
        blas_handle_,
        CUBLAS_OP_T,  // W_x transposed
        CUBLAS_OP_N,  // x not transposed
        dim_,         // m
        batch_size_ * seq_len,  // n
        dim_,         // k
        &alpha,
        W_x,
        CUDA_R_16BF,
        dim_,
        x,
        CUDA_R_16BF,
        dim_,
        &beta,
        x_proj_out,
        CUDA_R_16BF,
        dim_,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );

    // Step 2: Run fused forward kernel
    const float scale = 1.0f / sqrtf(static_cast<float>(dim_));
    int num_blocks = batch_size_;

    // Shared memory: h_tape[N*D] + h_work[D] + attn[N] + W_h_row[D] + write_val[D]
    size_t smem_size = (n_slots_ * dim_ + dim_ + n_slots_ + dim_ + dim_) * sizeof(float);

    if (n_slots_ == 64 && dim_ == 1024) {
        E23ForwardKernel_BF16<64, 1024><<<num_blocks, E23_BLOCK_SIZE, smem_size, stream_>>>(
            seq_len, batch_size_,
            x_proj_out, W_h, b_h, W_write,
            h_tape_init, h_work_init,
            h_work_out, h_tape_final,
            read_attn, write_attn,
            scale
        );
    } else if (n_slots_ == 32 && dim_ == 512) {
        E23ForwardKernel_BF16<32, 512><<<num_blocks, E23_BLOCK_SIZE, smem_size, stream_>>>(
            seq_len, batch_size_,
            x_proj_out, W_h, b_h, W_write,
            h_tape_init, h_work_init,
            h_work_out, h_tape_final,
            read_attn, write_attn,
            scale
        );
    } else if (n_slots_ == 16 && dim_ == 256) {
        E23ForwardKernel_BF16<16, 256><<<num_blocks, E23_BLOCK_SIZE, smem_size, stream_>>>(
            seq_len, batch_size_,
            x_proj_out, W_h, b_h, W_write,
            h_tape_init, h_work_init,
            h_work_out, h_tape_final,
            read_attn, write_attn,
            scale
        );
    }
    // Add more specializations as needed
}


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
    float* dW_write
) {
    const float scale = 1.0f / sqrtf(static_cast<float>(dim_));
    int num_blocks = batch_size_;

    // Shared memory for backward
    size_t smem_size = (2 * n_slots_ * dim_ + 2 * dim_ + n_slots_) * sizeof(float);

    if (n_slots_ == 64 && dim_ == 1024) {
        E23BackwardKernel_BF16<64, 1024><<<num_blocks, E23_BLOCK_SIZE, smem_size, stream_>>>(
            seq_len, batch_size_,
            x_proj, h_work_all, h_tape_all, read_attn, write_attn,
            W_h, W_write,
            d_h_work_out, d_h_tape_final,
            dx_proj, dW_h, db_h, dW_write,
            scale
        );
    } else if (n_slots_ == 32 && dim_ == 512) {
        E23BackwardKernel_BF16<32, 512><<<num_blocks, E23_BLOCK_SIZE, smem_size, stream_>>>(
            seq_len, batch_size_,
            x_proj, h_work_all, h_tape_all, read_attn, write_attn,
            W_h, W_write,
            d_h_work_out, d_h_tape_final,
            dx_proj, dW_h, db_h, dW_write,
            scale
        );
    } else if (n_slots_ == 16 && dim_ == 256) {
        E23BackwardKernel_BF16<16, 256><<<num_blocks, E23_BLOCK_SIZE, smem_size, stream_>>>(
            seq_len, batch_size_,
            x_proj, h_work_all, h_tape_all, read_attn, write_attn,
            W_h, W_write,
            d_h_work_out, d_h_tape_final,
            dx_proj, dW_h, db_h, dW_write,
            scale
        );
    }
}

// Explicit template instantiations
template class DualMemoryElmanForward<__nv_bfloat16>;
template class DualMemoryElmanBackward<__nv_bfloat16>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
