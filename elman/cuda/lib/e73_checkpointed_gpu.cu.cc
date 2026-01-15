// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E73CP: Matrix Nonlinear Elman with GRADIENT CHECKPOINTING
//
// Key difference from E73:
// - S is updated IN-PLACE (single [B,n,n] buffer)
// - Checkpoints stored every K steps (default K=32)
// - Backward recomputes S from checkpoints
// - 50x+ memory reduction for S storage
//
// Memory comparison (B=32, T=512, n=768):
//   Original E73: S = [T+1, B, n, n] = 19.4 GB
//   E73CP:        S = [B, n, n] + checkpoints = ~0.6 GB
//
// Architecture (same as E73):
//     k_norm = k / (||k|| + eps)
//     retrieved = (S * z_mod) @ k_norm
//     S = tanh(S + outer(v - retrieved, k_norm))
//     out = (S @ q) * silu(S @ q)

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cstdio>

#include "hasty/elman_ladder.h"
#include "blas.h"

namespace {

// Default checkpoint interval
constexpr int DEFAULT_CHECKPOINT_INTERVAL = 32;

// =============================================================================
// Kernel implementations (duplicated from e73_matrix_nonlinear_gpu.cu.cc)
// =============================================================================

__global__ void E73CPNormalizeKKernel_BF16(
    const int batch_size,
    const int n,
    const __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ k_norm,
    __nv_bfloat16* __restrict__ k_norm_factor)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size) {
        const int bn = b * n;
        float sum_sq = 0.0f;
        for (int i = 0; i < n; ++i) {
            float ki = __bfloat162float(k[bn + i]);
            sum_sq += ki * ki;
        }
        float norm = sqrtf(sum_sq) + 1e-6f;
        float inv_norm = 1.0f / norm;
        for (int i = 0; i < n; ++i) {
            float ki = __bfloat162float(k[bn + i]);
            k_norm[bn + i] = __float2bfloat16(ki * inv_norm);
        }
        if (k_norm_factor) {
            k_norm_factor[b] = __float2bfloat16(norm);
        }
    }
}

__global__ void E73CPRetrievalKernel_BF16(
    const int batch_size,
    const int n,
    const int variant,
    const __nv_bfloat16* __restrict__ S,
    const __nv_bfloat16* __restrict__ z,
    const __nv_bfloat16* __restrict__ k_norm,
    __nv_bfloat16* __restrict__ retrieved)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * n;

    if (idx < total) {
        const int i = idx % n;
        const int b = idx / n;
        const int bn = b * n;
        const int bnn = b * n * n;

        float z_i = __bfloat162float(z[bn + i]);
        float sum = 0.0f;
        for (int j = 0; j < n; ++j) {
            float s_ij = __bfloat162float(S[bnn + i * n + j]);
            float z_j = __bfloat162float(z[bn + j]);
            float k_j = __bfloat162float(k_norm[bn + j]);

            float z_mod;
            if (variant == 0) z_mod = z_j;
            else if (variant == 1) z_mod = z_i;
            else z_mod = z_i * z_j;

            sum += s_ij * z_mod * k_j;
        }
        retrieved[idx] = __float2bfloat16(sum);
    }
}

__global__ void E73CPOutputKernel_BF16(
    const int batch_size,
    const int n,
    const __nv_bfloat16* __restrict__ S,
    const __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ Sq_cache)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * n;

    if (idx < total) {
        const int i = idx % n;
        const int b = idx / n;

        float sum = 0.0f;
        for (int j = 0; j < n; ++j) {
            float s_val = __bfloat162float(S[b * n * n + i * n + j]);
            float q_val = __bfloat162float(q[b * n + j]);
            sum += s_val * q_val;
        }

        if (Sq_cache) {
            Sq_cache[idx] = __float2bfloat16(sum);
        }

        float sigmoid_sq = 1.0f / (1.0f + expf(-sum));
        float silu_sq = sum * sigmoid_sq;
        float out = sum * silu_sq;

        output[idx] = __float2bfloat16(out);
    }
}

// Add bias kernel
__global__ void BiasOnlyKernel_BF16(
    const int n,
    const __nv_bfloat16* __restrict__ z_logit,
    const __nv_bfloat16* __restrict__ b_z,
    __nv_bfloat16* __restrict__ z,
    const int batch_size,
    const int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const int j = idx % N;
        float z_val = __bfloat162float(z_logit[idx]) + __bfloat162float(b_z[j]);
        z[idx] = __float2bfloat16(z_val);
    }
}

// Copy S state kernel
__global__ void CopyStateKernel_BF16(
    const int size,
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = src[idx];
    }
}

// In-place delta update (S updated in same buffer)
__global__ void E73DeltaUpdateInPlaceKernel_BF16(
    const int batch_size,
    const int n,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ retrieved,
    const __nv_bfloat16* __restrict__ k_norm,
    __nv_bfloat16* __restrict__ S,  // Updated in-place
    __nv_bfloat16* __restrict__ pre_tanh)  // Optional cache
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * n * n;

    if (idx < total) {
        const int j = idx % n;
        const int i = (idx / n) % n;
        const int b = idx / (n * n);
        const int bn = b * n;

        float s_ij = __bfloat162float(S[idx]);
        float v_i = __bfloat162float(v[bn + i]);
        float ret_i = __bfloat162float(retrieved[bn + i]);
        float k_j = __bfloat162float(k_norm[bn + j]);

        float delta_i = v_i - ret_i;
        float new_val = s_ij + delta_i * k_j;

        if (pre_tanh) {
            pre_tanh[idx] = __float2bfloat16(new_val);
        }

        S[idx] = __float2bfloat16(tanhf(new_val));
    }
}

// Delta update with separate S_prev and S_new (for recomputation during backward)
__global__ void E73CPDeltaUpdateKernel_BF16(
    const int batch_size,
    const int n,
    const __nv_bfloat16* __restrict__ S_prev,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ retrieved,
    const __nv_bfloat16* __restrict__ k_norm,
    __nv_bfloat16* __restrict__ S_new,
    __nv_bfloat16* __restrict__ pre_tanh)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * n * n;

    if (idx < total) {
        const int j = idx % n;
        const int i = (idx / n) % n;
        const int b = idx / (n * n);
        const int bn = b * n;

        float s_prev = __bfloat162float(S_prev[idx]);
        float v_i = __bfloat162float(v[bn + i]);
        float ret_i = __bfloat162float(retrieved[bn + i]);
        float k_j = __bfloat162float(k_norm[bn + j]);

        float delta_i = v_i - ret_i;
        float pt = s_prev + delta_i * k_j;

        if (pre_tanh) {
            pre_tanh[idx] = __float2bfloat16(pt);
        }
        S_new[idx] = __float2bfloat16(tanhf(pt));
    }
}

// =============================================================================
// Backward Kernels for Checkpointed E73
// =============================================================================

// Backward through self-gate: out = Sq * silu(Sq)
__global__ void E73CPSelfGateBackwardKernel_BF16(
    const int batch_size,
    const int n,
    const __nv_bfloat16* __restrict__ Sq,
    const __nv_bfloat16* __restrict__ d_output,
    __nv_bfloat16* __restrict__ d_Sq)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * n;

    if (idx < total) {
        float sq_val = __bfloat162float(Sq[idx]);
        float dout = __bfloat162float(d_output[idx]);

        float sigmoid_sq = 1.0f / (1.0f + __expf(-sq_val));
        float silu_sq = sq_val * sigmoid_sq;
        float grad_factor = silu_sq * (2.0f + sq_val * (1.0f - sigmoid_sq));

        d_Sq[idx] = __float2bfloat16(dout * grad_factor);
    }
}

// Backward through output: Sq = S @ q
__global__ void E73CPOutputBackwardKernel_BF16(
    const int batch_size,
    const int n,
    const __nv_bfloat16* __restrict__ S,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ d_Sq,
    __nv_bfloat16* __restrict__ d_S,
    float* __restrict__ d_q_f)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * n;

    if (idx < total) {
        const int i = idx % n;
        const int b = idx / n;

        float d_sq_i = __bfloat162float(d_Sq[idx]);

        for (int j = 0; j < n; ++j) {
            int s_idx = b * n * n + i * n + j;
            float q_j = __bfloat162float(q[b * n + j]);
            float s_ij = __bfloat162float(S[s_idx]);

            d_S[s_idx] = __float2bfloat16(__bfloat162float(d_S[s_idx]) + d_sq_i * q_j);
            atomicAdd(&d_q_f[b * n + j], d_sq_i * s_ij);
        }
    }
}

// Backward through delta update: S_new = tanh(S_prev + outer(v - retrieved, k_norm))
__global__ void E73CPDeltaUpdateBackwardKernel_BF16(
    const int batch_size,
    const int n,
    const __nv_bfloat16* __restrict__ S_prev,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ retrieved,
    const __nv_bfloat16* __restrict__ k_norm,
    const __nv_bfloat16* __restrict__ pre_tanh,
    const __nv_bfloat16* __restrict__ d_S_new,
    __nv_bfloat16* __restrict__ d_S_prev,
    float* __restrict__ d_v_f,
    float* __restrict__ d_retrieved_f,
    float* __restrict__ d_k_norm_f)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * n * n;

    if (idx < total) {
        const int j = idx % n;
        const int i = (idx / n) % n;
        const int b = idx / (n * n);
        const int bn = b * n;

        float pt = __bfloat162float(pre_tanh[idx]);
        float ds_new = __bfloat162float(d_S_new[idx]);

        float tanh_pt = tanhf(pt);
        float d_pt = ds_new * (1.0f - tanh_pt * tanh_pt);

        d_S_prev[idx] = __float2bfloat16(d_pt);

        float v_i = __bfloat162float(v[bn + i]);
        float ret_i = __bfloat162float(retrieved[bn + i]);
        float delta_i = v_i - ret_i;
        float k_norm_j = __bfloat162float(k_norm[bn + j]);

        atomicAdd(&d_v_f[bn + i], d_pt * k_norm_j);
        atomicAdd(&d_retrieved_f[bn + i], -d_pt * k_norm_j);
        atomicAdd(&d_k_norm_f[bn + j], d_pt * delta_i);
    }
}

// Backward through retrieval: retrieved = (S * z_mod) @ k_norm
__global__ void E73CPRetrievalBackwardKernel_BF16(
    const int batch_size,
    const int n,
    const int variant,
    const __nv_bfloat16* __restrict__ S,
    const __nv_bfloat16* __restrict__ z,
    const __nv_bfloat16* __restrict__ k_norm,
    const float* __restrict__ d_retrieved_f,
    __nv_bfloat16* __restrict__ d_S,
    float* __restrict__ d_z_f,
    float* __restrict__ d_k_norm_f)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * n * n;

    if (idx < total) {
        const int j = idx % n;
        const int i = (idx / n) % n;
        const int b = idx / (n * n);
        const int bn = b * n;

        float d_ret_i = d_retrieved_f[bn + i];
        float s_ij = __bfloat162float(S[idx]);
        float z_i = __bfloat162float(z[bn + i]);
        float z_j = __bfloat162float(z[bn + j]);
        float k_norm_j = __bfloat162float(k_norm[bn + j]);

        float z_mod;
        if (variant == 0) {
            z_mod = z_j;
        } else if (variant == 1) {
            z_mod = z_i;
        } else {
            z_mod = z_i * z_j;
        }

        float ds_ij = __bfloat162float(d_S[idx]) + d_ret_i * z_mod * k_norm_j;
        d_S[idx] = __float2bfloat16(ds_ij);

        if (variant == 0) {
            atomicAdd(&d_z_f[bn + j], d_ret_i * s_ij * k_norm_j);
        } else if (variant == 1) {
            atomicAdd(&d_z_f[bn + i], d_ret_i * s_ij * k_norm_j);
        } else {
            atomicAdd(&d_z_f[bn + i], d_ret_i * s_ij * z_j * k_norm_j);
            atomicAdd(&d_z_f[bn + j], d_ret_i * s_ij * z_i * k_norm_j);
        }

        atomicAdd(&d_k_norm_f[bn + j], d_ret_i * s_ij * z_mod);
    }
}

// Backward through k normalization: k_norm = k / (||k|| + eps)
__global__ void E73CPNormalizeKBackwardKernel_BF16(
    const int batch_size,
    const int n,
    const __nv_bfloat16* __restrict__ k,
    const float* __restrict__ d_k_norm_f,
    float* __restrict__ d_k_f)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size) {
        const int bn = b * n;

        float sum_sq = 0.0f;
        for (int i = 0; i < n; ++i) {
            float ki = __bfloat162float(k[bn + i]);
            sum_sq += ki * ki;
        }
        float norm = sqrtf(sum_sq) + 1e-6f;
        float inv_norm = 1.0f / norm;
        float inv_norm3 = inv_norm * inv_norm * inv_norm;

        float dot = 0.0f;
        for (int j = 0; j < n; ++j) {
            float kj = __bfloat162float(k[bn + j]);
            dot += kj * d_k_norm_f[bn + j];
        }

        for (int i = 0; i < n; ++i) {
            float ki = __bfloat162float(k[bn + i]);
            float dk_i = d_k_norm_f[bn + i] * inv_norm - ki * dot * inv_norm3;
            atomicAdd(&d_k_f[bn + i], dk_i);
        }
    }
}

// Copy float to bf16
__global__ void CopyFloatToBF16Kernel(const int n, const float* __restrict__ src, __nv_bfloat16* __restrict__ dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2bfloat16(src[idx]);
    }
}

// Accumulate d_z_logit into db_z
__global__ void E73CPAccumulateDBzKernel_BF16(
    const int total,
    const int n,
    const __nv_bfloat16* __restrict__ d_z_logit,
    float* __restrict__ db_z)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        const int j = idx % n;
        atomicAdd(&db_z[j], __bfloat162float(d_z_logit[idx]));
    }
}

}  // anonymous namespace

namespace elman {

// =============================================================================
// E73CP Forward - In-place S with checkpointing
// =============================================================================

template<>
E73CheckpointedForward<__nv_bfloat16>::E73CheckpointedForward(
    int batch_size, int n_state, int dim, int variant, int checkpoint_interval,
    const cublasHandle_t& blas_handle, const cudaStream_t& stream)
    : batch_size_(batch_size), n_state_(n_state), dim_(dim), variant_(variant),
      checkpoint_interval_(checkpoint_interval > 0 ? checkpoint_interval : DEFAULT_CHECKPOINT_INTERVAL),
      blas_handle_(blas_handle), stream_(stream) {}

template<>
void E73CheckpointedForward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_k,
    const __nv_bfloat16* W_v,
    const __nv_bfloat16* W_q,
    const __nv_bfloat16* W_z,
    const __nv_bfloat16* b_z,
    const __nv_bfloat16* x,
    __nv_bfloat16* S,               // [B, n, n] - updated in-place!
    __nv_bfloat16* output,          // [T, B, n]
    __nv_bfloat16* S_checkpoints,   // [num_checkpoints, B, n, n]
    __nv_bfloat16* k_norm_cache,    // [T, B, n]
    __nv_bfloat16* v_cache,         // [T, B, n]
    __nv_bfloat16* q_cache,         // [T, B, n]
    __nv_bfloat16* z_cache,         // [T, B, n]
    __nv_bfloat16* Sq_cache,        // [T, B, n]
    __nv_bfloat16* workspace,
    bool training)
{
    static const __nv_bfloat16 alpha_one = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);

    const int n = n_state_;
    const int BN = batch_size_ * n;
    const int BNN = batch_size_ * n * n;
    const int block_size = 256;
    const int K = checkpoint_interval_;

    // Workspace layout:
    // k_all: [T*B, n] (unnormalized)
    // z_logit_all: [T*B, n] (temp)
    // retrieved: [B, n] (per-timestep temp)
    __nv_bfloat16* k_all = workspace;
    __nv_bfloat16* z_logit_all = k_all + steps * BN;
    __nv_bfloat16* retrieved = z_logit_all + steps * BN;

    // Pointers for projections (use caches if training)
    __nv_bfloat16* k_norm_all = training ? k_norm_cache : (retrieved + BN);
    __nv_bfloat16* v_all = training ? v_cache : (k_norm_all + steps * BN);
    __nv_bfloat16* q_all = training ? q_cache : (v_all + steps * BN);
    __nv_bfloat16* z_all = training ? z_cache : (q_all + steps * BN);

    // =========================================================================
    // Batch all 4 projections upfront (same as E73)
    // =========================================================================
    blas<__nv_bfloat16>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        n, steps * batch_size_, dim_, &alpha_one, W_k, dim_, x, dim_, &beta_zero, k_all, n);

    blas<__nv_bfloat16>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        n, steps * batch_size_, dim_, &alpha_one, W_v, dim_, x, dim_, &beta_zero, v_all, n);

    blas<__nv_bfloat16>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        n, steps * batch_size_, dim_, &alpha_one, W_q, dim_, x, dim_, &beta_zero, q_all, n);

    blas<__nv_bfloat16>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        n, steps * batch_size_, dim_, &alpha_one, W_z, dim_, x, dim_, &beta_zero, z_logit_all, n);

    // Add bias to z
    BiasOnlyKernel_BF16<<<(steps * BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
        steps * BN, z_logit_all, b_z, z_all, batch_size_, n);

    // =========================================================================
    // Save initial checkpoint (S at t=0)
    // =========================================================================
    if (training && S_checkpoints) {
        CopyStateKernel_BF16<<<(BNN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            BNN, S, S_checkpoints);
    }

    // =========================================================================
    // Sequential recurrence with IN-PLACE S update
    // =========================================================================
    for (int t = 0; t < steps; ++t) {
        const __nv_bfloat16* k_t = k_all + t * BN;
        __nv_bfloat16* k_norm_t = k_norm_all + t * BN;
        const __nv_bfloat16* v_t = v_all + t * BN;
        const __nv_bfloat16* q_t = q_all + t * BN;
        const __nv_bfloat16* z_t = z_all + t * BN;
        __nv_bfloat16* out_t = output + t * BN;
        __nv_bfloat16* Sq_t = training ? (Sq_cache + t * BN) : nullptr;

        // Step 1: Normalize k
        E73CPNormalizeKKernel_BF16<<<(batch_size_ + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n, k_t, k_norm_t, nullptr);

        // Step 2: Compute retrieved = (S * z_mod) @ k_norm
        E73CPRetrievalKernel_BF16<<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n, variant_, S, z_t, k_norm_t, retrieved);

        // Step 3: Delta update IN-PLACE (no pre_tanh cache in checkpointed mode)
        E73DeltaUpdateInPlaceKernel_BF16<<<(BNN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n, v_t, retrieved, k_norm_t, S, nullptr);

        // Step 4: Output
        E73CPOutputKernel_BF16<<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n, S, q_t, out_t, Sq_t);

        // Save checkpoint at interval boundaries (after processing step t)
        // Checkpoint index: t/K + 1 (since index 0 is initial state)
        if (training && S_checkpoints && ((t + 1) % K == 0)) {
            int cp_idx = (t + 1) / K;
            __nv_bfloat16* S_cp = S_checkpoints + cp_idx * BNN;
            CopyStateKernel_BF16<<<(BNN + block_size - 1) / block_size, block_size, 0, stream_>>>(
                BNN, S, S_cp);
        }
    }

    // Save final checkpoint if needed
    if (training && S_checkpoints) {
        int num_checkpoints = (steps + K) / K;  // Including initial
        int final_cp_idx = num_checkpoints - 1;
        if (steps % K != 0) {  // Final state not at checkpoint boundary
            __nv_bfloat16* S_final = S_checkpoints + final_cp_idx * BNN;
            CopyStateKernel_BF16<<<(BNN + block_size - 1) / block_size, block_size, 0, stream_>>>(
                BNN, S, S_final);
        }
    }
}

template<>
int64_t E73CheckpointedForward<__nv_bfloat16>::WorkspaceSize(
    int steps, int batch_size, int n_state, int checkpoint_interval, bool training)
{
    int64_t BN = batch_size * n_state;

    // k_all: T*B*n
    // z_logit_all: T*B*n
    // retrieved: B*n
    int64_t size = 2 * steps * BN + BN;

    // If not training, also need space for k_norm, v, q, z
    if (!training) {
        size += 4 * steps * BN;
    }

    return size;
}

// =============================================================================
// E73CP Backward - Recompute from checkpoints
// =============================================================================

template<>
E73CheckpointedBackward<__nv_bfloat16>::E73CheckpointedBackward(
    int batch_size, int n_state, int dim, int variant, int checkpoint_interval,
    const cublasHandle_t& blas_handle, const cudaStream_t& stream)
    : batch_size_(batch_size), n_state_(n_state), dim_(dim), variant_(variant),
      checkpoint_interval_(checkpoint_interval > 0 ? checkpoint_interval : DEFAULT_CHECKPOINT_INTERVAL),
      blas_handle_(blas_handle), stream_(stream) {}

template<>
void E73CheckpointedBackward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_k,
    const __nv_bfloat16* W_v,
    const __nv_bfloat16* W_q,
    const __nv_bfloat16* W_z,
    const __nv_bfloat16* x,
    const __nv_bfloat16* S_checkpoints,  // [num_checkpoints, B, n, n]
    const __nv_bfloat16* k_norm_cache,   // [T, B, n]
    const __nv_bfloat16* v_cache,        // [T, B, n]
    const __nv_bfloat16* q_cache,        // [T, B, n]
    const __nv_bfloat16* z_cache,        // [T, B, n]
    const __nv_bfloat16* Sq_cache,       // [T, B, n]
    const __nv_bfloat16* d_output,       // [T, B, n]
    __nv_bfloat16* dx,
    __nv_bfloat16* dW_k,
    __nv_bfloat16* dW_v,
    __nv_bfloat16* dW_q,
    __nv_bfloat16* dW_z,
    __nv_bfloat16* db_z,
    __nv_bfloat16* workspace)
{
    static const __nv_bfloat16 alpha_one = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);
    static const __nv_bfloat16 beta_one = __float2bfloat16(1.0f);

    const int n = n_state_;
    const int BN = batch_size_ * n;
    const int BNN = batch_size_ * n * n;
    const int block_size = 256;
    const int K = checkpoint_interval_;

    // Workspace layout:
    // S_interval: [K+1, B, n, n] - recomputed S states for current interval
    // pre_tanh_interval: [K, B, n, n] - pre-tanh values for current interval
    // retrieved_interval: [K, B, n] - retrieved values for current interval
    // d_S: [B, n, n] - accumulated gradient of S (passes across intervals)
    // d_S_tmp: [B, n, n] - temp for gradient computation
    // d_Sq: [B, n] - gradient through self-gate
    // d_k_all, d_v_all, d_q_all, d_z_all: [T, B, n] - per-timestep gradients
    // Float accumulators

    __nv_bfloat16* S_interval = workspace;  // [K+1, B, n, n]
    __nv_bfloat16* pre_tanh_interval = S_interval + (K + 1) * BNN;  // [K, B, n, n]
    __nv_bfloat16* retrieved_interval = pre_tanh_interval + K * BNN;  // [K, B, n]
    __nv_bfloat16* d_S = retrieved_interval + K * BN;  // [B, n, n]
    __nv_bfloat16* d_S_tmp = d_S + BNN;  // [B, n, n]
    __nv_bfloat16* d_Sq = d_S_tmp + BNN;  // [B, n]
    __nv_bfloat16* d_k_all = d_Sq + BN;  // [T, B, n]
    __nv_bfloat16* d_v_all = d_k_all + steps * BN;
    __nv_bfloat16* d_q_all = d_v_all + steps * BN;
    __nv_bfloat16* d_z_all = d_q_all + steps * BN;

    // Float accumulators (per-timestep, reused)
    float* float_ws = reinterpret_cast<float*>(d_z_all + steps * BN);
    float* d_k_f = float_ws;
    float* d_v_f = d_k_f + BN;
    float* d_q_f = d_v_f + BN;
    float* d_z_f = d_q_f + BN;
    float* d_retrieved_f = d_z_f + BN;
    float* d_k_norm_f = d_retrieved_f + BN;
    float* db_z_f = d_k_norm_f + BN;

    // k_unnorm workspace for k normalization backward
    __nv_bfloat16* k_unnorm_all = reinterpret_cast<__nv_bfloat16*>(db_z_f + n);

    // Initialize gradients
    cudaMemsetAsync(d_S, 0, BNN * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_k, 0, n * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_v, 0, n * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_q, 0, n * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_z, 0, n * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(db_z_f, 0, n * sizeof(float), stream_);
    cudaMemsetAsync(d_k_all, 0, steps * BN * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(d_v_all, 0, steps * BN * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(d_q_all, 0, steps * BN * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(d_z_all, 0, steps * BN * sizeof(__nv_bfloat16), stream_);

    // Recompute unnormalized k for normalization backward
    blas<__nv_bfloat16>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        n, steps * batch_size_, dim_, &alpha_one, W_k, dim_, x, dim_, &beta_zero, k_unnorm_all, n);

    // Process backward through checkpoint intervals (reverse order)
    int num_full_intervals = (steps + K - 1) / K;

    for (int interval_idx = num_full_intervals - 1; interval_idx >= 0; --interval_idx) {
        int t_start = interval_idx * K;
        int t_end = std::min((interval_idx + 1) * K, steps);
        int interval_len = t_end - t_start;
        if (interval_len <= 0) continue;

        // =====================================================================
        // Step 1: Load checkpoint and forward recompute through interval
        // =====================================================================
        const __nv_bfloat16* S_cp = S_checkpoints + interval_idx * BNN;
        CopyStateKernel_BF16<<<(BNN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            BNN, S_cp, S_interval);  // S_interval[0] = checkpoint

        for (int t_local = 0; t_local < interval_len; ++t_local) {
            int t = t_start + t_local;
            const __nv_bfloat16* k_norm_t = k_norm_cache + t * BN;
            const __nv_bfloat16* v_t = v_cache + t * BN;
            const __nv_bfloat16* z_t = z_cache + t * BN;

            __nv_bfloat16* S_prev_local = S_interval + t_local * BNN;
            __nv_bfloat16* S_next_local = S_interval + (t_local + 1) * BNN;
            __nv_bfloat16* pre_tanh_local = pre_tanh_interval + t_local * BNN;
            __nv_bfloat16* retrieved_local = retrieved_interval + t_local * BN;

            // Recompute: retrieved = (S * z_mod) @ k_norm
            E73CPRetrievalKernel_BF16<<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
                batch_size_, n, variant_, S_prev_local, z_t, k_norm_t, retrieved_local);

            // Recompute: S_next = tanh(S_prev + outer(v - retrieved, k_norm))
            E73CPDeltaUpdateKernel_BF16<<<(BNN + block_size - 1) / block_size, block_size, 0, stream_>>>(
                batch_size_, n, S_prev_local, v_t, retrieved_local, k_norm_t, S_next_local, pre_tanh_local);
        }

        // =====================================================================
        // Step 2: Backward through interval (reverse order)
        // =====================================================================
        for (int t_local = interval_len - 1; t_local >= 0; --t_local) {
            int t = t_start + t_local;

            // Get recomputed values for this timestep
            const __nv_bfloat16* S_prev_local = S_interval + t_local * BNN;
            const __nv_bfloat16* S_t_local = S_interval + (t_local + 1) * BNN;
            const __nv_bfloat16* pre_tanh_local = pre_tanh_interval + t_local * BNN;
            const __nv_bfloat16* retrieved_local = retrieved_interval + t_local * BN;

            // Get cached values
            const __nv_bfloat16* k_norm_t = k_norm_cache + t * BN;
            const __nv_bfloat16* k_unnorm_t = k_unnorm_all + t * BN;
            const __nv_bfloat16* v_t = v_cache + t * BN;
            const __nv_bfloat16* q_t = q_cache + t * BN;
            const __nv_bfloat16* z_t = z_cache + t * BN;
            const __nv_bfloat16* Sq_t = Sq_cache + t * BN;
            const __nv_bfloat16* d_out_t = d_output + t * BN;

            __nv_bfloat16* d_k_t = d_k_all + t * BN;
            __nv_bfloat16* d_v_t = d_v_all + t * BN;
            __nv_bfloat16* d_q_t = d_q_all + t * BN;
            __nv_bfloat16* d_z_t = d_z_all + t * BN;

            // Zero per-timestep float accumulators
            cudaMemsetAsync(d_k_f, 0, BN * sizeof(float), stream_);
            cudaMemsetAsync(d_v_f, 0, BN * sizeof(float), stream_);
            cudaMemsetAsync(d_q_f, 0, BN * sizeof(float), stream_);
            cudaMemsetAsync(d_z_f, 0, BN * sizeof(float), stream_);
            cudaMemsetAsync(d_retrieved_f, 0, BN * sizeof(float), stream_);
            cudaMemsetAsync(d_k_norm_f, 0, BN * sizeof(float), stream_);

            // 1. Backward through self-gate: out = Sq * silu(Sq)
            E73CPSelfGateBackwardKernel_BF16<<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
                batch_size_, n, Sq_t, d_out_t, d_Sq);

            // 2. Backward through output: Sq = S @ q
            // Adds to d_S, accumulates d_q_f
            E73CPOutputBackwardKernel_BF16<<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
                batch_size_, n, S_t_local, q_t, d_Sq, d_S, d_q_f);

            // 3. Backward through delta update: S_new = tanh(S_prev + outer(v - retrieved, k_norm))
            // Produces d_S_tmp (gradient w.r.t. S_prev), d_v_f, d_retrieved_f, d_k_norm_f
            E73CPDeltaUpdateBackwardKernel_BF16<<<(BNN + block_size - 1) / block_size, block_size, 0, stream_>>>(
                batch_size_, n, S_prev_local, v_t, retrieved_local, k_norm_t, pre_tanh_local, d_S,
                d_S_tmp, d_v_f, d_retrieved_f, d_k_norm_f);

            // 4. Backward through retrieval: retrieved = (S * z_mod) @ k_norm
            // Adds to d_S_tmp, d_z_f, d_k_norm_f
            E73CPRetrievalBackwardKernel_BF16<<<(BNN + block_size - 1) / block_size, block_size, 0, stream_>>>(
                batch_size_, n, variant_, S_prev_local, z_t, k_norm_t, d_retrieved_f,
                d_S_tmp, d_z_f, d_k_norm_f);

            // 5. Backward through k normalization: k_norm = k / (||k|| + eps)
            E73CPNormalizeKBackwardKernel_BF16<<<(batch_size_ + block_size - 1) / block_size, block_size, 0, stream_>>>(
                batch_size_, n, k_unnorm_t, d_k_norm_f, d_k_f);

            // Copy float gradients to bf16 output arrays
            CopyFloatToBF16Kernel<<<(BN + 255) / 256, 256, 0, stream_>>>(BN, d_k_f, d_k_t);
            CopyFloatToBF16Kernel<<<(BN + 255) / 256, 256, 0, stream_>>>(BN, d_v_f, d_v_t);
            CopyFloatToBF16Kernel<<<(BN + 255) / 256, 256, 0, stream_>>>(BN, d_q_f, d_q_t);
            CopyFloatToBF16Kernel<<<(BN + 255) / 256, 256, 0, stream_>>>(BN, d_z_f, d_z_t);

            // Swap d_S and d_S_tmp: d_S now holds gradient w.r.t. S_prev_local
            std::swap(d_S, d_S_tmp);
        }

        // After processing interval, d_S contains gradient w.r.t. S at t_start (the checkpoint)
        // This gradient passes to the previous interval automatically
    }

    // =========================================================================
    // Final: Batched GEMMs for weight gradients and dx
    // =========================================================================

    // dx = d_k_all @ W_k + d_v_all @ W_v + d_q_all @ W_q + d_z_all @ W_z
    blas<__nv_bfloat16>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, n, &alpha_one, W_k, dim_, d_k_all, n, &beta_zero, dx, dim_);
    blas<__nv_bfloat16>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, n, &alpha_one, W_v, dim_, d_v_all, n, &beta_one, dx, dim_);
    blas<__nv_bfloat16>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, n, &alpha_one, W_q, dim_, d_q_all, n, &beta_one, dx, dim_);
    blas<__nv_bfloat16>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, n, &alpha_one, W_z, dim_, d_z_all, n, &beta_one, dx, dim_);

    // dW_k = x.T @ d_k_all (note: need to transpose properly)
    blas<__nv_bfloat16>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, n, steps * batch_size_, &alpha_one, x, dim_, d_k_all, n, &beta_one, dW_k, dim_);
    blas<__nv_bfloat16>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, n, steps * batch_size_, &alpha_one, x, dim_, d_v_all, n, &beta_one, dW_v, dim_);
    blas<__nv_bfloat16>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, n, steps * batch_size_, &alpha_one, x, dim_, d_q_all, n, &beta_one, dW_q, dim_);
    blas<__nv_bfloat16>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, n, steps * batch_size_, &alpha_one, x, dim_, d_z_all, n, &beta_one, dW_z, dim_);

    // Accumulate db_z from d_z_all
    E73CPAccumulateDBzKernel_BF16<<<(steps * BN + 255) / 256, 256, 0, stream_>>>(
        steps * BN, n, d_z_all, db_z_f);
    CopyFloatToBF16Kernel<<<(n + 255) / 256, 256, 0, stream_>>>(n, db_z_f, db_z);
}

template<>
int64_t E73CheckpointedBackward<__nv_bfloat16>::WorkspaceSize(
    int steps, int batch_size, int n_state, int checkpoint_interval, int dim)
{
    int K = checkpoint_interval > 0 ? checkpoint_interval : DEFAULT_CHECKPOINT_INTERVAL;
    int64_t BN = batch_size * n_state;
    int64_t BNN = batch_size * n_state * n_state;

    // Workspace layout:
    // S_interval: [K+1, B, n, n]
    // pre_tanh_interval: [K, B, n, n]
    // retrieved_interval: [K, B, n]
    // d_S, d_S_tmp: 2 * BNN
    // d_Sq: BN
    // d_k_all, d_v_all, d_q_all, d_z_all: 4 * T * BN
    // Float accumulators: 7 * BN + n_state
    // k_unnorm_all: T * BN

    int64_t interval_size = (K + 1) * BNN + K * BNN + K * BN;  // S, pre_tanh, retrieved
    int64_t grad_size = 2 * BNN + BN;  // d_S, d_S_tmp, d_Sq
    int64_t per_step_grads = 4 * steps * BN;  // d_k_all, d_v_all, d_q_all, d_z_all
    int64_t float_bytes = (7 * BN + n_state) * sizeof(float);
    int64_t float_in_bf16 = (float_bytes + sizeof(__nv_bfloat16) - 1) / sizeof(__nv_bfloat16);
    int64_t k_unnorm = steps * BN;

    return interval_size + grad_size + per_step_grads + float_in_bf16 + k_unnorm;
}

template<>
int E73CheckpointedForward<__nv_bfloat16>::NumCheckpoints(int steps, int checkpoint_interval) {
    int K = checkpoint_interval > 0 ? checkpoint_interval : DEFAULT_CHECKPOINT_INTERVAL;
    // Checkpoints at: 0, K, 2K, ..., and final if not at boundary
    return (steps + K) / K;
}

}  // namespace elman
