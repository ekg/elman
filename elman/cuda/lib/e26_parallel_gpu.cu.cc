/**
 * E26: Parallel Dual-Memory Elman CUDA Kernels
 *
 * Architecture: Separate "what" (content) from "where" (routing)
 *
 * PARALLEL PHASE (done in bindings):
 *   x_proj[0:T] = x[0:T] @ W_x.T   -- One big GEMM
 *
 * SEQUENTIAL PHASE (this kernel):
 *   for t in range(T):
 *     read = softmax(h_work @ tape.T) @ tape    -- O(N×D) dots
 *     h_work = tanh(x_proj[t] + W_h @ h_work + read + b)
 *     tape = write(tape, h_work)                 -- O(N×D) dots
 *
 * Key insight: Attention is cheap (O(N×D)), GEMM is expensive (O(D²)).
 * Pre-computing x_proj batches the expensive part.
 */

#include "hasty/elman_ladder.h"
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

namespace {

constexpr int E26_BLOCK_SIZE = 256;

/**
 * Softmax device function for small N
 */
template<int N_SLOTS>
__device__ void softmax_device(float* attn_sh, const int tid, const float scale) {
    // Find max for numerical stability
    __shared__ float max_val;
    if (tid == 0) {
        max_val = attn_sh[0] * scale;
        for (int i = 1; i < N_SLOTS; i++) {
            float v = attn_sh[i] * scale;
            if (v > max_val) max_val = v;
        }
    }
    __syncthreads();

    // Compute exp and sum
    __shared__ float sum_exp;
    if (tid == 0) {
        sum_exp = 0.0f;
        for (int i = 0; i < N_SLOTS; i++) {
            float e = expf(attn_sh[i] * scale - max_val);
            attn_sh[i] = e;
            sum_exp += e;
        }
        sum_exp = fmaxf(sum_exp, 1e-9f);
    }
    __syncthreads();

    // Normalize
    if (tid < N_SLOTS) {
        attn_sh[tid] /= sum_exp;
    }
}

/**
 * E26 Phase 1: Read attention (softmax) + Update h_work
 */
template<int N_SLOTS, int DIM>
__global__ void E26Phase1Kernel_BF16(
    const int batch_size,
    const __nv_bfloat16* __restrict__ Rh,          // [B, D] W_h @ h_work_prev
    const __nv_bfloat16* __restrict__ x_proj_t,    // [B, D] pre-computed
    const __nv_bfloat16* __restrict__ b_h,         // [D]
    const __nv_bfloat16* __restrict__ h_tape,      // [B, N, D]
    const __nv_bfloat16* __restrict__ h_work,      // [B, D]
    __nv_bfloat16* __restrict__ h_work_out,        // [B, D]
    __nv_bfloat16* __restrict__ read_attn_out,     // [B, N]
    const float scale
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    __shared__ float attn_sh[N_SLOTS];
    __shared__ float h_work_sh[DIM];

    // Load h_work into shared memory
    for (int d = tid; d < DIM; d += E26_BLOCK_SIZE) {
        h_work_sh[d] = __bfloat162float(h_work[b * DIM + d]);
    }
    __syncthreads();

    // Compute attention scores: score[n] = h_tape[n] @ h_work
    if (tid < N_SLOTS) {
        float score = 0.0f;
        const __nv_bfloat16* tape_n = h_tape + b * N_SLOTS * DIM + tid * DIM;
        for (int d = 0; d < DIM; d++) {
            score += __bfloat162float(tape_n[d]) * h_work_sh[d];
        }
        attn_sh[tid] = score;
    }
    __syncthreads();

    // Softmax
    softmax_device<N_SLOTS>(attn_sh, tid, scale);
    __syncthreads();

    // Store read attention
    if (tid < N_SLOTS) {
        read_attn_out[b * N_SLOTS + tid] = __float2bfloat16(attn_sh[tid]);
    }

    // Compute h_work_new: tanh(Rh + x_proj_t + read_val + b_h)
    for (int d = tid; d < DIM; d += E26_BLOCK_SIZE) {
        float read_d = 0.0f;
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
 * E26 Phase 2: Write attention (softmax) + Update tape
 */
template<int N_SLOTS, int DIM>
__global__ void E26Phase2Kernel_BF16(
    const int batch_size,
    const __nv_bfloat16* __restrict__ write_val,   // [B, D] W_write @ h_work_new
    const __nv_bfloat16* __restrict__ h_work_new,  // [B, D]
    __nv_bfloat16* __restrict__ h_tape,            // [B, N, D]
    __nv_bfloat16* __restrict__ write_attn_out,    // [B, N]
    const float scale
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    __shared__ float attn_sh[N_SLOTS];
    __shared__ float h_work_sh[DIM];
    __shared__ float write_val_sh[DIM];

    // Load into shared memory
    for (int d = tid; d < DIM; d += E26_BLOCK_SIZE) {
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
        attn_sh[tid] = score;
    }
    __syncthreads();

    // Softmax
    softmax_device<N_SLOTS>(attn_sh, tid, scale);
    __syncthreads();

    // Store write attention
    if (tid < N_SLOTS) {
        write_attn_out[b * N_SLOTS + tid] = __float2bfloat16(attn_sh[tid]);
    }

    // Update tape: h_tape = (1 - attn) * h_tape + attn * write_val
    for (int i = tid; i < N_SLOTS * DIM; i += E26_BLOCK_SIZE) {
        int n = i / DIM;
        int d = i % DIM;
        float attn_n = attn_sh[n];
        float old_val = __bfloat162float(tape_b[i]);
        float new_val = (1.0f - attn_n) * old_val + attn_n * write_val_sh[d];
        tape_b[i] = __float2bfloat16(new_val);
    }
}

/**
 * E26 Backward Phase 1: Gradient w.r.t. write attention and tape
 */
template<int N_SLOTS, int DIM>
__global__ void E26BackwardInit_BF16(
    const int batch_size,
    const __nv_bfloat16* __restrict__ d_h_tape_final,
    __nv_bfloat16* __restrict__ d_h_tape,
    __nv_bfloat16* __restrict__ d_h_work
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = idx / (N_SLOTS * DIM);
    if (b >= batch_size) return;

    const int local_idx = idx % (N_SLOTS * DIM);
    d_h_tape[idx] = d_h_tape_final[idx];

    // Zero out d_h_work
    if (local_idx < DIM) {
        d_h_work[b * DIM + local_idx] = __float2bfloat16(0.0f);
    }
}

/**
 * E26 Backward Phase 2: Gradient through write
 */
template<int N_SLOTS, int DIM>
__global__ void E26BackwardPhase1_BF16(
    const int batch_size,
    const __nv_bfloat16* __restrict__ write_attn,
    const __nv_bfloat16* __restrict__ d_h_work_out,
    __nv_bfloat16* __restrict__ d_h_work,
    __nv_bfloat16* __restrict__ d_h_tape,
    __nv_bfloat16* __restrict__ d_write_val
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    __shared__ float attn_sh[N_SLOTS];
    __shared__ float d_h_work_sh[DIM];

    // Load write attention
    if (tid < N_SLOTS) {
        attn_sh[tid] = __bfloat162float(write_attn[b * N_SLOTS + tid]);
    }
    __syncthreads();

    // Accumulate d_h_work from output gradient
    for (int d = tid; d < DIM; d += E26_BLOCK_SIZE) {
        d_h_work_sh[d] = __bfloat162float(d_h_work_out[b * DIM + d]);
    }
    __syncthreads();

    // Gradient through tape update: new_tape = (1-attn)*old_tape + attn*write_val
    // d_write_val = d_tape @ attn (summed over slots)
    // d_old_tape = d_tape * (1 - attn)
    __nv_bfloat16* d_tape_b = d_h_tape + b * N_SLOTS * DIM;
    for (int d = tid; d < DIM; d += E26_BLOCK_SIZE) {
        float d_wv = 0.0f;
        for (int n = 0; n < N_SLOTS; n++) {
            float d_t = __bfloat162float(d_tape_b[n * DIM + d]);
            float a = attn_sh[n];
            d_wv += d_t * a;
            // Update d_tape for old_tape gradient
            d_tape_b[n * DIM + d] = __float2bfloat16(d_t * (1.0f - a));
        }
        d_write_val[b * DIM + d] = __float2bfloat16(d_wv);
    }
}

/**
 * E26 Backward Phase 3: Gradient through tanh and read attention
 */
template<int N_SLOTS, int DIM>
__global__ void E26BackwardPhase2_BF16(
    const int batch_size,
    const __nv_bfloat16* __restrict__ h_work,
    __nv_bfloat16* __restrict__ d_h_work,
    __nv_bfloat16* __restrict__ dx_proj,
    __nv_bfloat16* __restrict__ d_pre_act,
    float* __restrict__ db_h
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    // d_pre_act = d_h_work * (1 - h_work^2)  (tanh derivative)
    for (int d = tid; d < DIM; d += E26_BLOCK_SIZE) {
        float h = __bfloat162float(h_work[b * DIM + d]);
        float dh = __bfloat162float(d_h_work[b * DIM + d]);
        float dpa = dh * (1.0f - h * h);

        dx_proj[b * DIM + d] = __float2bfloat16(dpa);
        d_pre_act[b * DIM + d] = __float2bfloat16(dpa);

        // Accumulate db_h
        atomicAdd(&db_h[d], dpa);
    }
}

/**
 * E26 Backward Phase 4: Gradient through read attention
 */
template<int N_SLOTS, int DIM>
__global__ void E26BackwardPhase3_BF16(
    const int batch_size,
    const __nv_bfloat16* __restrict__ read_attn,
    const __nv_bfloat16* __restrict__ d_pre_act,
    const __nv_bfloat16* __restrict__ h_tape,
    const __nv_bfloat16* __restrict__ h_work_prev,
    const float scale,
    __nv_bfloat16* __restrict__ d_h_tape,
    __nv_bfloat16* __restrict__ d_h_work_prev
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    __shared__ float attn_sh[N_SLOTS];
    __shared__ float d_pre_act_sh[DIM];

    // Load
    if (tid < N_SLOTS) {
        attn_sh[tid] = __bfloat162float(read_attn[b * N_SLOTS + tid]);
    }
    for (int d = tid; d < DIM; d += E26_BLOCK_SIZE) {
        d_pre_act_sh[d] = __bfloat162float(d_pre_act[b * DIM + d]);
    }
    __syncthreads();

    // d_read_val = d_pre_act (since read_val is added directly)
    // read_val = sum_n(attn_n * tape_n)
    // d_tape_n += d_read_val * attn_n
    __nv_bfloat16* d_tape_b = d_h_tape + b * N_SLOTS * DIM;
    for (int i = tid; i < N_SLOTS * DIM; i += E26_BLOCK_SIZE) {
        int n = i / DIM;
        int d = i % DIM;
        float d_t = __bfloat162float(d_tape_b[i]);
        d_t += d_pre_act_sh[d] * attn_sh[n];
        d_tape_b[i] = __float2bfloat16(d_t);
    }
    __syncthreads();

    // Gradient through softmax attention (simplified - assume attn is fixed)
    // For proper gradient, need full Jacobian, but this is approximate
    // d_h_work_prev gets gradient from W_h @ h_work_prev term (done via GEMM)
}

}  // namespace


namespace hasty { namespace v0 { namespace elman_ladder {

// =============================================================================
// E26 Forward
// =============================================================================

template<typename T>
E26ParallelForward<T>::E26ParallelForward(
    bool training, int batch_size, int n_slots, int dim,
    const cublasHandle_t& blas_handle, const cudaStream_t& stream)
    : training_(training), batch_size_(batch_size), n_slots_(n_slots), dim_(dim),
      stream_(stream), blas_handle_(blas_handle) {}

template<>
void E26ParallelForward<__nv_bfloat16>::Run(
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

    #define LAUNCH_E26_PHASE1(N, D) \
        E26Phase1Kernel_BF16<N, D><<<num_blocks, E26_BLOCK_SIZE, 0, stream_>>>( \
            batch_size_, tmp_Rh, x_proj_t, b_h, h_tape_final, h_work_prev, \
            h_work_cur, read_attn_t, scale)

    #define LAUNCH_E26_PHASE2(N, D) \
        E26Phase2Kernel_BF16<N, D><<<num_blocks, E26_BLOCK_SIZE, 0, stream_>>>( \
            batch_size_, tmp_write_val, h_work_cur, h_tape_final, \
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

        // Phase 1: read attention + h_work update
        if (n_slots_ == 8 && dim_ == 256) { LAUNCH_E26_PHASE1(8, 256); }
        else if (n_slots_ == 8 && dim_ == 512) { LAUNCH_E26_PHASE1(8, 512); }
        else if (n_slots_ == 16 && dim_ == 256) { LAUNCH_E26_PHASE1(16, 256); }
        else if (n_slots_ == 16 && dim_ == 512) { LAUNCH_E26_PHASE1(16, 512); }
        else if (n_slots_ == 32 && dim_ == 256) { LAUNCH_E26_PHASE1(32, 256); }
        else if (n_slots_ == 32 && dim_ == 512) { LAUNCH_E26_PHASE1(32, 512); }
        else if (n_slots_ == 64 && dim_ == 256) { LAUNCH_E26_PHASE1(64, 256); }
        else if (n_slots_ == 64 && dim_ == 512) { LAUNCH_E26_PHASE1(64, 512); }
        else { fprintf(stderr, "E26 CUDA: unsupported n_slots=%d, dim=%d\n", n_slots_, dim_); }

        // W_write @ h_work_new
        cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha,
            W_write, CUDA_R_16BF, dim_, h_work_cur, CUDA_R_16BF, dim_,
            &beta, tmp_write_val, CUDA_R_16BF, dim_,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // Phase 2: write attention + tape update
        if (n_slots_ == 8 && dim_ == 256) { LAUNCH_E26_PHASE2(8, 256); }
        else if (n_slots_ == 8 && dim_ == 512) { LAUNCH_E26_PHASE2(8, 512); }
        else if (n_slots_ == 16 && dim_ == 256) { LAUNCH_E26_PHASE2(16, 256); }
        else if (n_slots_ == 16 && dim_ == 512) { LAUNCH_E26_PHASE2(16, 512); }
        else if (n_slots_ == 32 && dim_ == 256) { LAUNCH_E26_PHASE2(32, 256); }
        else if (n_slots_ == 32 && dim_ == 512) { LAUNCH_E26_PHASE2(32, 512); }
        else if (n_slots_ == 64 && dim_ == 256) { LAUNCH_E26_PHASE2(64, 256); }
        else if (n_slots_ == 64 && dim_ == 512) { LAUNCH_E26_PHASE2(64, 512); }

        if (training_ && h_tape_all) {
            cudaMemcpyAsync(h_tape_all + (t + 1) * batch_size_ * n_slots_ * dim_,
                h_tape_final, batch_size_ * n_slots_ * dim_ * sizeof(__nv_bfloat16),
                cudaMemcpyDeviceToDevice, stream_);
        }
    }

    #undef LAUNCH_E26_PHASE1
    #undef LAUNCH_E26_PHASE2
}


// =============================================================================
// E26 Backward
// =============================================================================

template<typename T>
E26ParallelBackward<T>::E26ParallelBackward(
    int batch_size, int n_slots, int dim,
    const cublasHandle_t& blas_handle, const cudaStream_t& stream)
    : batch_size_(batch_size), n_slots_(n_slots), dim_(dim),
      stream_(stream), blas_handle_(blas_handle) {}

template<>
void E26ParallelBackward<__nv_bfloat16>::Run(
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
    const int BD = batch_size_ * dim_;
    const int BN = batch_size_ * n_slots_;
    const float alpha_one = 1.0f;
    const float beta_zero = 0.0f;
    const float beta_one = 1.0f;

    __nv_bfloat16* d_h_work;
    cudaMalloc(&d_h_work, BD * sizeof(__nv_bfloat16));

    #define LAUNCH_E26_BWD_INIT(N, D) \
        E26BackwardInit_BF16<N, D><<<(batch_size_ * N * D + 255) / 256, 256, 0, stream_>>>( \
            batch_size_, d_h_tape_final, d_h_tape, d_h_work)

    #define LAUNCH_E26_BWD_PHASE1(N, D) \
        E26BackwardPhase1_BF16<N, D><<<num_blocks, E26_BLOCK_SIZE, 0, stream_>>>( \
            batch_size_, write_attn_t, d_h_work_out_t, d_h_work, d_h_tape, d_write_val_t)

    #define LAUNCH_E26_BWD_PHASE2(N, D) \
        E26BackwardPhase2_BF16<N, D><<<num_blocks, E26_BLOCK_SIZE, 0, stream_>>>( \
            batch_size_, h_work_t, d_h_work, dx_proj_t, d_pre_act_t, db_h)

    #define LAUNCH_E26_BWD_PHASE3(N, D) \
        E26BackwardPhase3_BF16<N, D><<<num_blocks, E26_BLOCK_SIZE, 0, stream_>>>( \
            batch_size_, read_attn_t, d_pre_act_t, h_tape_t_ptr, h_work_prev_t, scale, d_h_tape, d_h_work)

    // Initialize
    if (n_slots_ == 8 && dim_ == 256) { LAUNCH_E26_BWD_INIT(8, 256); }
    else if (n_slots_ == 8 && dim_ == 512) { LAUNCH_E26_BWD_INIT(8, 512); }
    else if (n_slots_ == 16 && dim_ == 256) { LAUNCH_E26_BWD_INIT(16, 256); }
    else if (n_slots_ == 16 && dim_ == 512) { LAUNCH_E26_BWD_INIT(16, 512); }
    else if (n_slots_ == 32 && dim_ == 256) { LAUNCH_E26_BWD_INIT(32, 256); }
    else if (n_slots_ == 32 && dim_ == 512) { LAUNCH_E26_BWD_INIT(32, 512); }
    else if (n_slots_ == 64 && dim_ == 256) { LAUNCH_E26_BWD_INIT(64, 256); }
    else if (n_slots_ == 64 && dim_ == 512) { LAUNCH_E26_BWD_INIT(64, 512); }

    const float scale = 1.0f / sqrtf(static_cast<float>(dim_));

    // Backward through time
    for (int t = seq_len - 1; t >= 0; --t) {
        const __nv_bfloat16* h_work_t = h_work_all + t * BD;
        const __nv_bfloat16* h_work_prev_t = (t == 0) ? h_work_init : (h_work_all + (t - 1) * BD);
        const __nv_bfloat16* h_tape_t_ptr = h_tape_all + t * batch_size_ * n_slots_ * dim_;
        const __nv_bfloat16* read_attn_t = read_attn + t * BN;
        const __nv_bfloat16* write_attn_t = write_attn + t * BN;
        const __nv_bfloat16* d_h_work_out_t = d_h_work_out + t * BD;
        __nv_bfloat16* dx_proj_t = dx_proj + t * BD;
        __nv_bfloat16* d_pre_act_t = d_pre_act_all + t * BD;
        __nv_bfloat16* d_write_val_t = d_write_val_all + t * BD;

        // Phase 1: backward through write
        if (n_slots_ == 8 && dim_ == 256) { LAUNCH_E26_BWD_PHASE1(8, 256); }
        else if (n_slots_ == 8 && dim_ == 512) { LAUNCH_E26_BWD_PHASE1(8, 512); }
        else if (n_slots_ == 16 && dim_ == 256) { LAUNCH_E26_BWD_PHASE1(16, 256); }
        else if (n_slots_ == 16 && dim_ == 512) { LAUNCH_E26_BWD_PHASE1(16, 512); }
        else if (n_slots_ == 32 && dim_ == 256) { LAUNCH_E26_BWD_PHASE1(32, 256); }
        else if (n_slots_ == 32 && dim_ == 512) { LAUNCH_E26_BWD_PHASE1(32, 512); }
        else if (n_slots_ == 64 && dim_ == 256) { LAUNCH_E26_BWD_PHASE1(64, 256); }
        else if (n_slots_ == 64 && dim_ == 512) { LAUNCH_E26_BWD_PHASE1(64, 512); }

        // dW_write += d_write_val^T @ h_work
        cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_, &alpha_one,
            d_write_val_t, CUDA_R_16BF, dim_,
            h_work_t, CUDA_R_16BF, dim_,
            &beta_one, dW_write, CUDA_R_32F, dim_,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // d_h_work += d_write_val @ W_write
        cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha_one,
            W_write, CUDA_R_16BF, dim_,
            d_write_val_t, CUDA_R_16BF, dim_,
            &beta_one, d_h_work, CUDA_R_16BF, dim_,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // Phase 2: backward through tanh
        if (n_slots_ == 8 && dim_ == 256) { LAUNCH_E26_BWD_PHASE2(8, 256); }
        else if (n_slots_ == 8 && dim_ == 512) { LAUNCH_E26_BWD_PHASE2(8, 512); }
        else if (n_slots_ == 16 && dim_ == 256) { LAUNCH_E26_BWD_PHASE2(16, 256); }
        else if (n_slots_ == 16 && dim_ == 512) { LAUNCH_E26_BWD_PHASE2(16, 512); }
        else if (n_slots_ == 32 && dim_ == 256) { LAUNCH_E26_BWD_PHASE2(32, 256); }
        else if (n_slots_ == 32 && dim_ == 512) { LAUNCH_E26_BWD_PHASE2(32, 512); }
        else if (n_slots_ == 64 && dim_ == 256) { LAUNCH_E26_BWD_PHASE2(64, 256); }
        else if (n_slots_ == 64 && dim_ == 512) { LAUNCH_E26_BWD_PHASE2(64, 512); }

        // Phase 3: backward through read attention
        if (n_slots_ == 8 && dim_ == 256) { LAUNCH_E26_BWD_PHASE3(8, 256); }
        else if (n_slots_ == 8 && dim_ == 512) { LAUNCH_E26_BWD_PHASE3(8, 512); }
        else if (n_slots_ == 16 && dim_ == 256) { LAUNCH_E26_BWD_PHASE3(16, 256); }
        else if (n_slots_ == 16 && dim_ == 512) { LAUNCH_E26_BWD_PHASE3(16, 512); }
        else if (n_slots_ == 32 && dim_ == 256) { LAUNCH_E26_BWD_PHASE3(32, 256); }
        else if (n_slots_ == 32 && dim_ == 512) { LAUNCH_E26_BWD_PHASE3(32, 512); }
        else if (n_slots_ == 64 && dim_ == 256) { LAUNCH_E26_BWD_PHASE3(64, 256); }
        else if (n_slots_ == 64 && dim_ == 512) { LAUNCH_E26_BWD_PHASE3(64, 512); }

        // dW_h += d_pre_act^T @ h_work_prev
        cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_, &alpha_one,
            d_pre_act_t, CUDA_R_16BF, dim_,
            h_work_prev_t, CUDA_R_16BF, dim_,
            &beta_one, dW_h, CUDA_R_32F, dim_,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // d_h_work_prev += d_pre_act @ W_h
        cudaMemsetAsync(d_h_work, 0, BD * sizeof(__nv_bfloat16), stream_);
        cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha_one,
            W_h, CUDA_R_16BF, dim_,
            d_pre_act_t, CUDA_R_16BF, dim_,
            &beta_zero, d_h_work, CUDA_R_16BF, dim_,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }

    cudaFree(d_h_work);

    #undef LAUNCH_E26_BWD_INIT
    #undef LAUNCH_E26_BWD_PHASE1
    #undef LAUNCH_E26_BWD_PHASE2
    #undef LAUNCH_E26_BWD_PHASE3
}

// Explicit instantiations
template class E26ParallelForward<__nv_bfloat16>;
template class E26ParallelBackward<__nv_bfloat16>;

}}}  // namespace hasty::v0::elman_ladder
