/**
 * E23c_v2: Chunked Dual-Memory Elman with Read Feedback
 *
 * Key difference from E23c: read_{t-1} feeds back into h_work_t
 *   h_work_t = tanh(W_h @ h_work_{t-1} + W_x @ x_t + W_r @ read_{t-1} + b)
 *   read_t = attention(h_work_t, tape)
 *   output_t = h_work_t
 *
 * This makes per-timestep computation sequential, but we still batch:
 *   - Tape updates at end of chunk (parallel formula)
 *   - Keep computation on GPU, fused operations
 *
 * Benefits vs E23c:
 *   - Restored read feedback for better loss
 *   - More expressive model (read influences hidden state)
 */

#include "hasty/elman_ladder.h"
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

namespace {

constexpr int BLOCK_SIZE = 256;

/**
 * RNN update with read feedback
 * h_work_new = tanh(Rh + x_proj_t + Rr + b)
 * where Rh = W_h @ h_work_prev, Rr = W_r @ read_prev
 */
__global__ void E23cv2_TanhUpdateWithRead_BF16(
    const int size,
    const __nv_bfloat16* __restrict__ Rh,        // [B, D]
    const __nv_bfloat16* __restrict__ x_proj_t,  // [B, D]
    const __nv_bfloat16* __restrict__ Rr,        // [B, D]
    const __nv_bfloat16* __restrict__ b_h,       // [D]
    __nv_bfloat16* __restrict__ h_work_out,      // [B, D]
    const int dim
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    const int d = idx % dim;
    const float rh = __bfloat162float(Rh[idx]);
    const float xp = __bfloat162float(x_proj_t[idx]);
    const float rr = __bfloat162float(Rr[idx]);
    const float b = __bfloat162float(b_h[d]);

    h_work_out[idx] = __float2bfloat16(tanhf(rh + xp + rr + b));
}

/**
 * Softmax for attention over N slots
 */
template<int N_SLOTS>
__global__ void E23cv2_Softmax_BF16(
    const int batch_size,
    const __nv_bfloat16* __restrict__ scores,  // [B, N]
    __nv_bfloat16* __restrict__ attn,          // [B, N]
    const float scale
) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    const __nv_bfloat16* s = scores + b * N_SLOTS;
    __nv_bfloat16* a = attn + b * N_SLOTS;

    float vals[N_SLOTS];
    float max_v = -1e30f;
    #pragma unroll
    for (int n = 0; n < N_SLOTS; n++) {
        vals[n] = __bfloat162float(s[n]) * scale;
        max_v = fmaxf(max_v, vals[n]);
    }

    float sum_exp = 0.0f;
    #pragma unroll
    for (int n = 0; n < N_SLOTS; n++) {
        vals[n] = expf(vals[n] - max_v);
        sum_exp += vals[n];
    }

    const float inv = 1.0f / sum_exp;
    #pragma unroll
    for (int n = 0; n < N_SLOTS; n++) {
        a[n] = __float2bfloat16(vals[n] * inv);
    }
}

/**
 * Scatter h_work to chunk buffer: h_work[B,D] -> tmp_chunk[B,K,D] at position t
 */
__global__ void E23cv2_ScatterToChunk_BF16(
    const int batch_size,
    const int dim,
    const int K,
    const int t,
    const __nv_bfloat16* __restrict__ h_work,
    __nv_bfloat16* __restrict__ tmp_chunk
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = batch_size * dim;
    if (idx >= size) return;

    const int b = idx / dim;
    const int d = idx % dim;
    tmp_chunk[b * K * dim + t * dim + d] = h_work[idx];
}

/**
 * Scatter attention weights to chunk buffer
 */
__global__ void E23cv2_ScatterAttn_BF16(
    const int batch_size,
    const int n_slots,
    const int K,
    const int t,
    const __nv_bfloat16* __restrict__ attn,
    __nv_bfloat16* __restrict__ tmp_attn_chunk
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = batch_size * n_slots;
    if (idx >= size) return;

    const int b = idx / n_slots;
    const int n = idx % n_slots;
    tmp_attn_chunk[b * K * n_slots + t * n_slots + n] = attn[idx];
}

/**
 * Add read to h_work for output: output = h_work + read
 * This provides both:
 * - Read feedback via h_work (from read_{t-1})
 * - Direct read access in output (from read_t)
 */
__global__ void E23cv2_AdditiveOutput_BF16(
    const int size,
    const __nv_bfloat16* __restrict__ h_work,
    const __nv_bfloat16* __restrict__ read,
    __nv_bfloat16* __restrict__ output
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    const float h = __bfloat162float(h_work[idx]);
    const float r = __bfloat162float(read[idx]);
    output[idx] = __float2bfloat16(h + r);
}

/**
 * Cumulative product of (1 - attn) from the END
 */
__global__ void E23cv2_CumprodRev_BF16(
    const int K,
    const int B,
    const int N,
    const __nv_bfloat16* __restrict__ attn,
    __nv_bfloat16* __restrict__ cumprod_rev
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * N;
    if (idx >= total) return;

    const int b = idx / N;
    const int n = idx % N;

    float prod = 1.0f;
    for (int k = K - 1; k >= 0; k--) {
        const int attn_idx = b * K * N + k * N + n;
        const float a = __bfloat162float(attn[attn_idx]);
        prod *= (1.0f - a);
        cumprod_rev[attn_idx] = __float2bfloat16(prod);
    }
}

/**
 * Parallel tape update
 */
__global__ void E23cv2_ParallelTapeUpdate_BF16(
    const int K,
    const int B,
    const int N,
    const int D,
    const __nv_bfloat16* __restrict__ tape_0,
    const __nv_bfloat16* __restrict__ write_vals,   // [B, K, D]
    const __nv_bfloat16* __restrict__ write_attn,   // [B, K, N]
    const __nv_bfloat16* __restrict__ cumprod_rev,
    __nv_bfloat16* __restrict__ tape_out
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * N * D;
    if (idx >= total) return;

    const int d = idx % D;
    const int n = (idx / D) % N;
    const int b = idx / (N * D);

    const float tape_0_coeff = __bfloat162float(cumprod_rev[b * K * N + 0 * N + n]);
    const float tape_0_val = __bfloat162float(tape_0[b * N * D + n * D + d]);

    float result = tape_0_val * tape_0_coeff;

    for (int k = 0; k < K; k++) {
        const float a = __bfloat162float(write_attn[b * K * N + k * N + n]);
        const float v = __bfloat162float(write_vals[b * K * D + k * D + d]);

        float cumprod_shifted;
        if (k < K - 1) {
            cumprod_shifted = __bfloat162float(cumprod_rev[b * K * N + (k + 1) * N + n]);
        } else {
            cumprod_shifted = 1.0f;
        }

        result += v * a * cumprod_shifted;
    }

    tape_out[idx] = __float2bfloat16(result);
}

}  // namespace

namespace hasty { namespace v0 { namespace elman_ladder {

/**
 * E23c_v2 Forward with read feedback
 */
template<>
void E23cv2ChunkedForward<__nv_bfloat16>::Run(
    int seq_len,
    int chunk_size,
    const __nv_bfloat16* x_proj,
    const __nv_bfloat16* W_h,
    const __nv_bfloat16* W_r,          // NEW: read projection weight [D, D]
    const __nv_bfloat16* b_h,
    const __nv_bfloat16* W_write,
    const __nv_bfloat16* h_tape_init,
    const __nv_bfloat16* h_work_init,
    __nv_bfloat16* output,
    __nv_bfloat16* h_tape_final,
    __nv_bfloat16* h_work_all,
    __nv_bfloat16* workspace
) {
    const float alpha = 1.0f, beta = 0.0f;
    const float scale = 1.0f / sqrtf(static_cast<float>(dim_));

    const int BD = batch_size_ * dim_;
    const int BN = batch_size_ * n_slots_;
    const int BND = batch_size_ * n_slots_ * dim_;
    const int K = chunk_size;

    // Workspace layout:
    // - tmp_Rh: [B, D]
    // - tmp_Rr: [B, D]
    // - tmp_scores: [B, N]
    // - tmp_attn: [B, N]
    // - tmp_read: [B, D]
    // - tmp_h_chunk: [B, K, D]
    // - tmp_attn_chunk: [B, K, N]
    // - tmp_write_vals: [B, K, D]
    // - tmp_cumprod: [B, K, N]
    __nv_bfloat16* tmp_Rh = workspace;
    __nv_bfloat16* tmp_Rr = tmp_Rh + BD;
    __nv_bfloat16* tmp_scores = tmp_Rr + BD;
    __nv_bfloat16* tmp_attn = tmp_scores + BN;
    __nv_bfloat16* tmp_read = tmp_attn + BN;
    __nv_bfloat16* tmp_h_chunk = tmp_read + BD;
    __nv_bfloat16* tmp_attn_chunk = tmp_h_chunk + K * BD;
    __nv_bfloat16* tmp_write_vals = tmp_attn_chunk + K * BN;
    __nv_bfloat16* tmp_cumprod = tmp_write_vals + K * BD;

    // Initialize tape
    cudaMemcpyAsync(h_tape_final, h_tape_init, BND * sizeof(__nv_bfloat16),
                    cudaMemcpyDeviceToDevice, stream_);

    // Initialize read to zeros (for t=0)
    cudaMemsetAsync(tmp_read, 0, BD * sizeof(__nv_bfloat16), stream_);

    const int threads = BLOCK_SIZE;

    // Process in chunks
    for (int chunk_start = 0; chunk_start < seq_len; chunk_start += K) {
        const int chunk_end = min(chunk_start + K, seq_len);
        const int chunk_len = chunk_end - chunk_start;

        // ================================================================
        // Step 1: Sequential RNN with read feedback
        // h_work_t = tanh(W_h @ h_work_{t-1} + W_x @ x_t + W_r @ read_{t-1} + b)
        // read_t = attention(h_work_t, tape)
        // ================================================================
        for (int t = 0; t < chunk_len; ++t) {
            const int global_t = chunk_start + t;
            const __nv_bfloat16* x_proj_t = x_proj + global_t * BD;
            const __nv_bfloat16* h_work_prev = (global_t == 0) ?
                h_work_init : (h_work_all + (global_t - 1) * BD);
            __nv_bfloat16* h_work_cur = h_work_all + global_t * BD;

            // W_h @ h_work_prev -> tmp_Rh
            cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                dim_, batch_size_, dim_, &alpha,
                W_h, CUDA_R_16BF, dim_, h_work_prev, CUDA_R_16BF, dim_,
                &beta, tmp_Rh, CUDA_R_16BF, dim_,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

            // W_r @ read_prev -> tmp_Rr
            cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                dim_, batch_size_, dim_, &alpha,
                W_r, CUDA_R_16BF, dim_, tmp_read, CUDA_R_16BF, dim_,
                &beta, tmp_Rr, CUDA_R_16BF, dim_,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

            // h_work = tanh(Rh + x_proj + Rr + b)
            int blocks = (BD + threads - 1) / threads;
            E23cv2_TanhUpdateWithRead_BF16<<<blocks, threads, 0, stream_>>>(
                BD, tmp_Rh, x_proj_t, tmp_Rr, b_h, h_work_cur, dim_);

            // Compute attention scores: h_work @ tape^T -> [B, N]
            // tape is [B, N, D], h_work is [B, D]
            // For each batch: score[n] = sum_d h_work[d] * tape[n, d]
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

            // Softmax
            int softmax_blocks = (batch_size_ + threads - 1) / threads;
            if (n_slots_ == 8) E23cv2_Softmax_BF16<8><<<softmax_blocks, threads, 0, stream_>>>(
                batch_size_, tmp_scores, tmp_attn, scale);
            else if (n_slots_ == 16) E23cv2_Softmax_BF16<16><<<softmax_blocks, threads, 0, stream_>>>(
                batch_size_, tmp_scores, tmp_attn, scale);
            else if (n_slots_ == 32) E23cv2_Softmax_BF16<32><<<softmax_blocks, threads, 0, stream_>>>(
                batch_size_, tmp_scores, tmp_attn, scale);
            else if (n_slots_ == 64) E23cv2_Softmax_BF16<64><<<softmax_blocks, threads, 0, stream_>>>(
                batch_size_, tmp_scores, tmp_attn, scale);

            // Weighted read: attn @ tape -> read [B, D]
            cublasGemmStridedBatchedEx(blas_handle_,
                CUBLAS_OP_N, CUBLAS_OP_N,
                dim_, 1, n_slots_,
                &alpha,
                h_tape_final, CUDA_R_16BF, dim_, n_slots_ * dim_,
                tmp_attn, CUDA_R_16BF, n_slots_, n_slots_,
                &beta,
                tmp_read, CUDA_R_16BF, dim_, dim_,
                batch_size_,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

            // Scatter h_work and attn to chunk buffers for tape update
            E23cv2_ScatterToChunk_BF16<<<blocks, threads, 0, stream_>>>(
                batch_size_, dim_, chunk_len, t, h_work_cur, tmp_h_chunk);

            int attn_blocks = (BN + threads - 1) / threads;
            E23cv2_ScatterAttn_BF16<<<attn_blocks, threads, 0, stream_>>>(
                batch_size_, n_slots_, chunk_len, t, tmp_attn, tmp_attn_chunk);

            // Compute output = h_work + read (additive read like E23c)
            E23cv2_AdditiveOutput_BF16<<<blocks, threads, 0, stream_>>>(
                BD, h_work_cur, tmp_read, output + global_t * BD);
        }

        // ================================================================
        // Step 2: Batched tape update
        // ================================================================
        // Compute write_vals: W_write @ h_chunk [B, K, D]
        cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, chunk_len * batch_size_, dim_, &alpha,
            W_write, CUDA_R_16BF, dim_,
            tmp_h_chunk, CUDA_R_16BF, dim_,
            &beta,
            tmp_write_vals, CUDA_R_16BF, dim_,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // Cumulative product for parallel tape update
        int cumprod_blocks = (BN + threads - 1) / threads;
        E23cv2_CumprodRev_BF16<<<cumprod_blocks, threads, 0, stream_>>>(
            chunk_len, batch_size_, n_slots_, tmp_attn_chunk, tmp_cumprod);

        // Parallel tape update
        int tape_blocks = (BND + threads - 1) / threads;
        E23cv2_ParallelTapeUpdate_BF16<<<tape_blocks, threads, 0, stream_>>>(
            chunk_len, batch_size_, n_slots_, dim_,
            h_tape_final, tmp_write_vals, tmp_attn_chunk, tmp_cumprod, h_tape_final);
    }
}

// Constructor
template<>
E23cv2ChunkedForward<__nv_bfloat16>::E23cv2ChunkedForward(
    bool training, int batch_size, int n_slots, int dim,
    const cublasHandle_t& blas_handle, const cudaStream_t& stream)
    : training_(training), batch_size_(batch_size), n_slots_(n_slots), dim_(dim),
      stream_(stream), blas_handle_(blas_handle) {}

// Explicit instantiation
template class E23cv2ChunkedForward<__nv_bfloat16>;

}}}  // namespace hasty::v0::elman_ladder
