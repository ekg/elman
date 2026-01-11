/**
 * E23c: Chunked Dual-Memory Elman CUDA Kernels
 *
 * Key architectural change from E23:
 *   h_work_t = tanh(W_h @ h_work_{t-1} + W_x @ x_t + b)  -- NO read dependency!
 *   output_t = h_work_t + read_t                         -- Additive read
 *
 * This allows batching reads and writes within chunks:
 *   1. Pre-compute h_work for K steps (sequential RNN)
 *   2. Batch ALL read attentions: [B, K, D] @ [B, D, N] - ONE BIG GEMM
 *   3. Batch ALL write attentions from frozen tape
 *   4. Parallel tape update via cumulative products
 *
 * Benefits:
 *   - Read attention: T tiny GEMMs -> T/K big GEMMs
 *   - Better tensor core utilization
 *   - Reduced kernel launch overhead
 */

#include "hasty/elman_ladder.h"
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

namespace {

constexpr int E23C_BLOCK_SIZE = 256;

/**
 * Pure RNN update kernel (no read dependency)
 * h_work_new = tanh(Rh + x_proj_t + b)
 */
__global__ void E23c_TanhUpdate_BF16(
    const int size,
    const __nv_bfloat16* __restrict__ Rh,        // [B, D]
    const __nv_bfloat16* __restrict__ x_proj_t,  // [B, D]
    const __nv_bfloat16* __restrict__ b_h,       // [D]
    __nv_bfloat16* __restrict__ h_work_out,      // [B, D]
    const int dim
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    const int d = idx % dim;
    const float rh = __bfloat162float(Rh[idx]);
    const float xp = __bfloat162float(x_proj_t[idx]);
    const float b = __bfloat162float(b_h[d]);

    h_work_out[idx] = __float2bfloat16(tanhf(rh + xp + b));
}

/**
 * Scatter kernel: copy h_work [B, D] to chunk buffer at timestep t
 * tmp_h_chunk layout: [B, K, D] for batched GEMM compatibility
 *
 * h_work[b, d] -> tmp_h_chunk[b, t, d]
 * Index: b * K * D + t * D + d
 */
__global__ void E23c_ScatterToChunk_BF16(
    const int batch_size,
    const int dim,
    const int K,              // chunk size
    const int t,              // timestep within chunk
    const __nv_bfloat16* __restrict__ h_work,      // [B, D]
    __nv_bfloat16* __restrict__ tmp_h_chunk        // [B, K, D]
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = batch_size * dim;
    if (idx >= size) return;

    const int b = idx / dim;
    const int d = idx % dim;

    // h_work[b, d] at offset b*D + d
    // tmp_h_chunk[b, t, d] at offset b*K*D + t*D + d
    tmp_h_chunk[b * K * dim + t * dim + d] = h_work[idx];
}

/**
 * Fused gather+add kernel: compute output = h_chunk + read_vals with layout transformation
 *
 * h_chunk layout: [B, K, D] (from scatter)
 * read_vals layout: [B, K, D] (from batched GEMM)
 * output layout: [K, B, D] (for final output tensor)
 *
 * output[t, b, d] = h_chunk[b, t, d] + read_vals[b, t, d]
 */
__global__ void E23c_GatherAddOutput_BF16(
    const int batch_size,
    const int dim,
    const int K,
    const __nv_bfloat16* __restrict__ h_chunk,      // [B, K, D]
    const __nv_bfloat16* __restrict__ read_vals,    // [B, K, D]
    __nv_bfloat16* __restrict__ output              // [K, B, D]
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = K * batch_size * dim;
    if (idx >= size) return;

    // Output index [t, b, d] in [K, B, D] layout
    const int d = idx % dim;
    const int b = (idx / dim) % batch_size;
    const int t = idx / (batch_size * dim);

    // Source index in [B, K, D] layout: b*K*D + t*D + d
    const int src_idx = b * K * dim + t * dim + d;

    const float h = __bfloat162float(h_chunk[src_idx]);
    const float r = __bfloat162float(read_vals[src_idx]);
    output[idx] = __float2bfloat16(h + r);
}

/**
 * Additive output kernel: output = h_work + read
 */
__global__ void E23c_AdditiveOutput_BF16(
    const int size,
    const __nv_bfloat16* __restrict__ h_work,  // [K*B, D]
    const __nv_bfloat16* __restrict__ read,    // [K*B, D]
    __nv_bfloat16* __restrict__ output         // [K*B, D]
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    const float h = __bfloat162float(h_work[idx]);
    const float r = __bfloat162float(read[idx]);
    output[idx] = __float2bfloat16(h + r);
}

/**
 * Softmax kernel for batched attention scores
 * Input: [B*K, N] scores
 * Output: [B*K, N] attention weights
 */
template<int N_SLOTS>
__global__ void E23c_BatchedSoftmax_BF16(
    const int batch_k,  // B * K
    const __nv_bfloat16* __restrict__ scores,  // [B*K, N]
    __nv_bfloat16* __restrict__ attn,          // [B*K, N]
    const float scale
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_k) return;

    const __nv_bfloat16* s = scores + idx * N_SLOTS;
    __nv_bfloat16* a = attn + idx * N_SLOTS;

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

/**
 * Compute cumulative product of (1 - attn) from the END
 * cumprod_rev[i] = prod_{j=i}^{K-1}(1 - attn[j])
 *
 * Input: attn [B, K, N] - contiguous as [B*K, N]
 * Output: cumprod_rev [B, K, N]
 */
__global__ void E23c_CumprodRev_BF16(
    const int K,
    const int B,
    const int N,
    const __nv_bfloat16* __restrict__ attn,      // [B, K, N]
    __nv_bfloat16* __restrict__ cumprod_rev      // [B, K, N]
) {
    // Each thread handles one (b, n) pair, computes cumprod over K
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * N;
    if (idx >= total) return;

    const int b = idx / N;
    const int n = idx % N;

    // Compute cumulative product from the end
    // attn[b, k, n] at offset b*K*N + k*N + n
    float prod = 1.0f;
    for (int k = K - 1; k >= 0; k--) {
        const int attn_idx = b * K * N + k * N + n;
        const float a = __bfloat162float(attn[attn_idx]);
        prod *= (1.0f - a);
        cumprod_rev[attn_idx] = __float2bfloat16(prod);
    }
}

/**
 * Parallel tape update using precomputed coefficients
 *
 * tape_K = tape_0 * tape_0_coeff + sum_i(v_i * v_coeffs[i])
 *
 * All inputs now use [B, K, ...] layout:
 * - write_vals: [B, K, D]
 * - write_attn: [B, K, N]
 * - cumprod_rev: [B, K, N]
 *
 * tape_0_coeff[b,n] = cumprod_rev[b, 0, n]
 * v_coeffs[b,k,n] = attn[b,k,n] * cumprod_shifted[b,k,n]
 * where cumprod_shifted[b,k,n] = cumprod_rev[b,k+1,n] for k < K-1, else 1
 */
__global__ void E23c_ParallelTapeUpdate_BF16(
    const int K,
    const int B,
    const int N,
    const int D,
    const __nv_bfloat16* __restrict__ tape_0,       // [B, N, D]
    const __nv_bfloat16* __restrict__ write_vals,   // [B, K, D]
    const __nv_bfloat16* __restrict__ write_attn,   // [B, K, N]
    const __nv_bfloat16* __restrict__ cumprod_rev,  // [B, K, N]
    __nv_bfloat16* __restrict__ tape_out            // [B, N, D]
) {
    // Each thread handles one (b, n, d) element
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * N * D;
    if (idx >= total) return;

    const int d = idx % D;
    const int n = (idx / D) % N;
    const int b = idx / (N * D);

    // tape_0 coefficient: cumprod_rev[b, 0, n] at offset b*K*N + 0*N + n
    const float tape_0_coeff = __bfloat162float(cumprod_rev[b * K * N + 0 * N + n]);
    const float tape_0_val = __bfloat162float(tape_0[b * N * D + n * D + d]);

    float result = tape_0_val * tape_0_coeff;

    // Sum contributions from write values
    for (int k = 0; k < K; k++) {
        // write_attn[b, k, n] at offset b*K*N + k*N + n
        const float a = __bfloat162float(write_attn[b * K * N + k * N + n]);
        // write_vals[b, k, d] at offset b*K*D + k*D + d
        const float v = __bfloat162float(write_vals[b * K * D + k * D + d]);

        // cumprod_shifted[b,k,n] = cumprod_rev[b,k+1,n] if k < K-1, else 1.0
        float cumprod_shifted;
        if (k < K - 1) {
            cumprod_shifted = __bfloat162float(cumprod_rev[b * K * N + (k + 1) * N + n]);
        } else {
            cumprod_shifted = 1.0f;
        }

        // v_coeff = a * cumprod_shifted
        result += v * a * cumprod_shifted;
    }

    tape_out[idx] = __float2bfloat16(result);
}

}  // namespace

namespace hasty { namespace v0 { namespace elman_ladder {

/**
 * E23c Chunked Forward Implementation
 */
template<>
void E23cChunkedForward<__nv_bfloat16>::Run(
    int seq_len,
    int chunk_size,
    const __nv_bfloat16* x_proj,       // [T, B, D] - pre-projected input
    const __nv_bfloat16* W_h,          // [D, D]
    const __nv_bfloat16* b_h,          // [D]
    const __nv_bfloat16* W_write,      // [D, D]
    const __nv_bfloat16* h_tape_init,  // [B, N, D]
    const __nv_bfloat16* h_work_init,  // [B, D]
    __nv_bfloat16* output,             // [T, B, D]
    __nv_bfloat16* h_tape_final,       // [B, N, D]
    __nv_bfloat16* h_work_all,         // [T, B, D] - all h_work states
    __nv_bfloat16* workspace           // workspace for intermediate results
) {
    const float alpha = 1.0f, beta = 0.0f;
    const float scale = 1.0f / sqrtf(static_cast<float>(dim_));

    const int BD = batch_size_ * dim_;
    const int BN = batch_size_ * n_slots_;
    const int BND = batch_size_ * n_slots_ * dim_;
    const int K = chunk_size;

    // Workspace layout:
    // - tmp_Rh: [B, D]
    // - tmp_h_chunk: [K, B, D] - h_work for current chunk
    // - tmp_read_scores: [B, K, N] - read attention scores
    // - tmp_read_attn: [B, K, N] - read attention weights
    // - tmp_read_vals: [K, B, D] - read values
    // - tmp_write_vals: [K, B, D] - write values
    // - tmp_write_attn: [K, B, N] - write attention weights
    // - tmp_cumprod: [K, B, N] - cumulative products
    __nv_bfloat16* tmp_Rh = workspace;
    __nv_bfloat16* tmp_h_chunk = tmp_Rh + BD;
    __nv_bfloat16* tmp_read_scores = tmp_h_chunk + K * BD;
    __nv_bfloat16* tmp_read_attn = tmp_read_scores + batch_size_ * K * n_slots_;
    __nv_bfloat16* tmp_read_vals = tmp_read_attn + batch_size_ * K * n_slots_;
    __nv_bfloat16* tmp_write_vals = tmp_read_vals + K * BD;
    __nv_bfloat16* tmp_write_attn = tmp_write_vals + K * BD;
    __nv_bfloat16* tmp_cumprod = tmp_write_attn + K * BN;

    // Initialize tape
    cudaMemcpyAsync(h_tape_final, h_tape_init, BND * sizeof(__nv_bfloat16),
                    cudaMemcpyDeviceToDevice, stream_);

    const int threads = E23C_BLOCK_SIZE;

    // Process in chunks
    for (int chunk_start = 0; chunk_start < seq_len; chunk_start += K) {
        const int chunk_end = min(chunk_start + K, seq_len);
        const int chunk_len = chunk_end - chunk_start;

        // ================================================================
        // Step 1: Sequential RNN for h_work (no read dependency)
        // h_work_t = tanh(W_h @ h_work_{t-1} + W_x @ x_t + b)
        // Store h_chunk in [B, K, D] layout for batched GEMM
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

            // h_work_cur = tanh(tmp_Rh + x_proj_t + b)
            int blocks = (BD + threads - 1) / threads;
            E23c_TanhUpdate_BF16<<<blocks, threads, 0, stream_>>>(
                BD, tmp_Rh, x_proj_t, b_h, h_work_cur, dim_);

            // Scatter to chunk buffer in [B, chunk_len, D] layout
            E23c_ScatterToChunk_BF16<<<blocks, threads, 0, stream_>>>(
                batch_size_, dim_, chunk_len, t, h_work_cur, tmp_h_chunk);
        }

        // ================================================================
        // Step 2: Batched read attention from frozen tape
        // tmp_h_chunk is [B, K, D], tape is [B, N, D]
        // scores = H_chunk @ tape^T = [B, K, D] @ [B, D, N] -> [B, K, N]
        //
        // cuBLAS column-major: For each batch b:
        //   C = A @ B where A=[D, N], B=[D, K], C=[N, K] in col-major
        //   = [N, D]^T @ [K, D]^T in row-major interpretation
        //   So with OP_T on A: C[n,k] = sum_d A[d,n] * B[d,k]
        //   = sum_d tape[b,n,d] * h_chunk[b,k,d] (exactly what we want!)
        // ================================================================
        cublasGemmStridedBatchedEx(blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            n_slots_, chunk_len, dim_,  // m, n, k (output is [N, K] in col-major = [K, N] row-major)
            &alpha,
            h_tape_final, CUDA_R_16BF, dim_, n_slots_ * dim_,  // A: tape [B, N, D], lda=D, stride=N*D
            tmp_h_chunk, CUDA_R_16BF, dim_, chunk_len * dim_,  // B: h_chunk [B, K, D], ldb=D, stride=K*D
            &beta,
            tmp_read_scores, CUDA_R_16BF, n_slots_, chunk_len * n_slots_,  // C: [B, K, N], ldc=N, stride=K*N
            batch_size_,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // Softmax over N: scores [B*K, N] -> attn [B*K, N]
        // Note: tmp_read_scores is [B, K, N] which is [B*K, N] contiguously
        int softmax_blocks = (batch_size_ * chunk_len + threads - 1) / threads;
        if (n_slots_ == 8) E23c_BatchedSoftmax_BF16<8><<<softmax_blocks, threads, 0, stream_>>>(
            batch_size_ * chunk_len, tmp_read_scores, tmp_read_attn, scale);
        else if (n_slots_ == 16) E23c_BatchedSoftmax_BF16<16><<<softmax_blocks, threads, 0, stream_>>>(
            batch_size_ * chunk_len, tmp_read_scores, tmp_read_attn, scale);
        else if (n_slots_ == 32) E23c_BatchedSoftmax_BF16<32><<<softmax_blocks, threads, 0, stream_>>>(
            batch_size_ * chunk_len, tmp_read_scores, tmp_read_attn, scale);
        else if (n_slots_ == 64) E23c_BatchedSoftmax_BF16<64><<<softmax_blocks, threads, 0, stream_>>>(
            batch_size_ * chunk_len, tmp_read_scores, tmp_read_attn, scale);

        // Weighted read: attn @ tape = [B, K, N] @ [B, N, D] -> [B, K, D]
        // cuBLAS: C = A @ B where A=[D, N], B=[N, K], C=[D, K] in col-major
        //   With OP_N on both: C[d,k] = sum_n A[d,n] * B[n,k]
        //   = sum_n tape[b,n,d] * attn[b,k,n] (exactly what we want!)
        cublasGemmStridedBatchedEx(blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, chunk_len, n_slots_,  // m, n, k
            &alpha,
            h_tape_final, CUDA_R_16BF, dim_, n_slots_ * dim_,  // A: tape [B, N, D], lda=D
            tmp_read_attn, CUDA_R_16BF, n_slots_, chunk_len * n_slots_,  // B: attn [B, K, N], ldb=N
            &beta,
            tmp_read_vals, CUDA_R_16BF, dim_, chunk_len * dim_,  // C: read_vals [B, K, D], ldc=D
            batch_size_,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // ================================================================
        // Step 3: Compute output = h_work + read with layout transformation
        // tmp_h_chunk: [B, K, D], tmp_read_vals: [B, K, D]
        // output: [T, B, D] where we write chunk at [chunk_start:chunk_end, :, :]
        // Use fused gather+add kernel that handles [B,K,D] -> [K,B,D] permutation
        // ================================================================
        int output_blocks = (chunk_len * BD + threads - 1) / threads;
        E23c_GatherAddOutput_BF16<<<output_blocks, threads, 0, stream_>>>(
            batch_size_, dim_, chunk_len,
            tmp_h_chunk, tmp_read_vals, output + chunk_start * BD);

        // ================================================================
        // Step 4: Batched write computation and parallel tape update
        // ================================================================
        // Compute write_vals: [B, K, D] = W_write @ h_chunk
        // h_chunk is [B, K, D] = contiguous [B*K, D], result is also [B, K, D]
        cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, chunk_len * batch_size_, dim_, &alpha,
            W_write, CUDA_R_16BF, dim_,
            tmp_h_chunk, CUDA_R_16BF, dim_,
            &beta,
            tmp_write_vals, CUDA_R_16BF, dim_,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // Compute write attention scores from FROZEN tape (same as read)
        // Reuse tmp_read_scores layout, just store in tmp_write_attn after softmax
        // Actually we can recompute since we need fresh softmax
        cublasGemmStridedBatchedEx(blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            n_slots_, chunk_len, dim_,
            &alpha,
            h_tape_final, CUDA_R_16BF, dim_, n_slots_ * dim_,
            tmp_h_chunk, CUDA_R_16BF, dim_, chunk_len * dim_,
            &beta,
            tmp_read_scores, CUDA_R_16BF, n_slots_, chunk_len * n_slots_,
            batch_size_,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // Softmax for write attention
        if (n_slots_ == 8) E23c_BatchedSoftmax_BF16<8><<<softmax_blocks, threads, 0, stream_>>>(
            batch_size_ * chunk_len, tmp_read_scores, tmp_write_attn, scale);
        else if (n_slots_ == 16) E23c_BatchedSoftmax_BF16<16><<<softmax_blocks, threads, 0, stream_>>>(
            batch_size_ * chunk_len, tmp_read_scores, tmp_write_attn, scale);
        else if (n_slots_ == 32) E23c_BatchedSoftmax_BF16<32><<<softmax_blocks, threads, 0, stream_>>>(
            batch_size_ * chunk_len, tmp_read_scores, tmp_write_attn, scale);
        else if (n_slots_ == 64) E23c_BatchedSoftmax_BF16<64><<<softmax_blocks, threads, 0, stream_>>>(
            batch_size_ * chunk_len, tmp_read_scores, tmp_write_attn, scale);

        // Compute cumulative product of (1 - write_attn) from the END
        int cumprod_blocks = (BN + threads - 1) / threads;
        E23c_CumprodRev_BF16<<<cumprod_blocks, threads, 0, stream_>>>(
            chunk_len, batch_size_, n_slots_, tmp_write_attn, tmp_cumprod);

        // Parallel tape update
        int tape_blocks = (BND + threads - 1) / threads;
        E23c_ParallelTapeUpdate_BF16<<<tape_blocks, threads, 0, stream_>>>(
            chunk_len, batch_size_, n_slots_, dim_,
            h_tape_final, tmp_write_vals, tmp_write_attn, tmp_cumprod, h_tape_final);
    }
}

// Constructor
template<>
E23cChunkedForward<__nv_bfloat16>::E23cChunkedForward(
    bool training, int batch_size, int n_slots, int dim,
    const cublasHandle_t& blas_handle, const cudaStream_t& stream)
    : training_(training), batch_size_(batch_size), n_slots_(n_slots), dim_(dim),
      stream_(stream), blas_handle_(blas_handle) {}

// Explicit instantiation
template class E23cChunkedForward<__nv_bfloat16>;

}}}  // namespace hasty::v0::elman_ladder
