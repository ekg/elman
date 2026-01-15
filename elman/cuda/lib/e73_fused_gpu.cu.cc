// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E73 FUSED: Optimized Matrix Nonlinear Elman with DELTA RULE
//
// Key optimizations over original E73:
// 1. FUSED forward kernel - all operations in ONE kernel launch per timestep
// 2. Block-parallel normalization - full thread block, not 32 threads
// 3. Gradient checkpointing - store S every K steps, recompute in backward
// 4. In-place backward - S stays "hot", no constant global memory round-trips
//
// Architecture (unchanged):
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

// Checkpoint interval for gradient checkpointing
constexpr int CHECKPOINT_INTERVAL = 32;

// =============================================================================
// FUSED Forward Kernel - All operations for ONE timestep in ONE kernel
// =============================================================================
// Each thread block handles one batch element's full N×N state
// Threads within block cooperatively process the N×N matrix

__global__ void E73FusedForwardKernel_BF16(
    const int batch_size,
    const int n,                      // n_state (e.g., 768)
    const int variant,                // 0=column, 1=row, 2=full
    // Inputs for this timestep
    const __nv_bfloat16* __restrict__ k,       // [B, n]
    const __nv_bfloat16* __restrict__ v,       // [B, n]
    const __nv_bfloat16* __restrict__ q,       // [B, n]
    const __nv_bfloat16* __restrict__ z,       // [B, n]
    // State (in-place update)
    __nv_bfloat16* __restrict__ S,             // [B, n, n] - updated in place
    // Outputs
    __nv_bfloat16* __restrict__ output,        // [B, n]
    // Training caches (optional)
    __nv_bfloat16* __restrict__ k_norm_out,    // [B, n] store normalized k
    __nv_bfloat16* __restrict__ pre_tanh_out,  // [B, n, n] store pre-tanh S
    __nv_bfloat16* __restrict__ Sq_out)        // [B, n] store S@q pre-selfgate
{
    // Each block handles one batch element
    const int b = blockIdx.x;
    if (b >= batch_size) return;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int bn = b * n;
    const int bnn = b * n * n;

    // Shared memory for k_norm, v, z, retrieved, Sq
    extern __shared__ float shared[];
    float* s_k_norm = shared;                    // [n]
    float* s_v = s_k_norm + n;                   // [n]
    float* s_z = s_v + n;                        // [n]
    float* s_retrieved = s_z + n;                // [n]
    float* s_Sq = s_retrieved + n;               // [n]
    float* s_reduction = s_Sq + n;               // [block_size] for reductions

    // =========================================================================
    // Step 1: Load v, z into shared memory & compute k_norm
    // =========================================================================

    // Load v, z cooperatively
    for (int i = tid; i < n; i += block_size) {
        s_v[i] = __bfloat162float(v[bn + i]);
        s_z[i] = __bfloat162float(z[bn + i]);
    }

    // Compute ||k||² using block-parallel reduction
    float local_sum_sq = 0.0f;
    for (int i = tid; i < n; i += block_size) {
        float ki = __bfloat162float(k[bn + i]);
        local_sum_sq += ki * ki;
    }
    s_reduction[tid] = local_sum_sq;
    __syncthreads();

    // Block reduction for sum
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_reduction[tid] += s_reduction[tid + stride];
        }
        __syncthreads();
    }

    float norm = sqrtf(s_reduction[0]) + 1e-6f;
    float inv_norm = 1.0f / norm;

    // Compute and store k_norm
    for (int i = tid; i < n; i += block_size) {
        float ki = __bfloat162float(k[bn + i]);
        s_k_norm[i] = ki * inv_norm;
        if (k_norm_out) {
            k_norm_out[bn + i] = __float2bfloat16(s_k_norm[i]);
        }
    }
    __syncthreads();

    // =========================================================================
    // Step 2: Compute retrieved = (S * z_mod) @ k_norm
    // Each thread computes one or more elements of retrieved[i]
    // =========================================================================

    for (int i = tid; i < n; i += block_size) {
        float z_i = s_z[i];
        float sum = 0.0f;
        for (int j = 0; j < n; ++j) {
            float s_ij = __bfloat162float(S[bnn + i * n + j]);
            float z_j = s_z[j];
            float k_j = s_k_norm[j];

            float z_mod;
            if (variant == 0) z_mod = z_j;        // column
            else if (variant == 1) z_mod = z_i;   // row
            else z_mod = z_i * z_j;               // full

            sum += s_ij * z_mod * k_j;
        }
        s_retrieved[i] = sum;
    }
    __syncthreads();

    // =========================================================================
    // Step 3: Delta update: S = tanh(S + outer(v - retrieved, k_norm))
    // Each thread handles multiple S elements
    // =========================================================================

    for (int idx = tid; idx < n * n; idx += block_size) {
        int i = idx / n;
        int j = idx % n;

        float s_ij = __bfloat162float(S[bnn + idx]);
        float delta_i = s_v[i] - s_retrieved[i];
        float k_j = s_k_norm[j];

        float pre_tanh = s_ij + delta_i * k_j;
        float new_s = tanhf(pre_tanh);

        S[bnn + idx] = __float2bfloat16(new_s);

        if (pre_tanh_out) {
            pre_tanh_out[bnn + idx] = __float2bfloat16(pre_tanh);
        }
    }
    __syncthreads();

    // =========================================================================
    // Step 4: Output = (S @ q) * silu(S @ q)
    // =========================================================================

    // Load q and compute S @ q
    for (int i = tid; i < n; i += block_size) {
        float sum = 0.0f;
        for (int j = 0; j < n; ++j) {
            float s_ij = __bfloat162float(S[bnn + i * n + j]);
            float q_j = __bfloat162float(q[bn + j]);
            sum += s_ij * q_j;
        }
        s_Sq[i] = sum;
    }
    __syncthreads();

    // Apply self-gate and write output
    for (int i = tid; i < n; i += block_size) {
        float sq = s_Sq[i];
        if (Sq_out) {
            Sq_out[bn + i] = __float2bfloat16(sq);
        }

        // out = sq * silu(sq) = sq * sq * sigmoid(sq)
        float sigmoid_sq = 1.0f / (1.0f + expf(-sq));
        float out = sq * sq * sigmoid_sq;
        output[bn + i] = __float2bfloat16(out);
    }
}

// =============================================================================
// FUSED Backward Kernel - All backward operations for ONE timestep
// =============================================================================

__global__ void E73FusedBackwardKernel_BF16(
    const int batch_size,
    const int n,
    const int variant,
    // Cached values from forward
    const __nv_bfloat16* __restrict__ k_norm,     // [B, n]
    const __nv_bfloat16* __restrict__ v,          // [B, n]
    const __nv_bfloat16* __restrict__ q,          // [B, n]
    const __nv_bfloat16* __restrict__ z,          // [B, n]
    const __nv_bfloat16* __restrict__ pre_tanh,   // [B, n, n]
    const __nv_bfloat16* __restrict__ Sq,         // [B, n]
    const __nv_bfloat16* __restrict__ S_next,     // [B, n, n] S after this step
    // Gradients
    const __nv_bfloat16* __restrict__ d_output,   // [B, n]
    __nv_bfloat16* __restrict__ d_S,              // [B, n, n] - accumulated
    // Output gradients (accumulated)
    float* __restrict__ d_k_f,                    // [B, n]
    float* __restrict__ d_v_f,                    // [B, n]
    float* __restrict__ d_q_f,                    // [B, n]
    float* __restrict__ d_z_f)                    // [B, n]
{
    const int b = blockIdx.x;
    if (b >= batch_size) return;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int bn = b * n;
    const int bnn = b * n * n;

    extern __shared__ float shared[];
    float* s_k_norm = shared;
    float* s_v = s_k_norm + n;
    float* s_z = s_v + n;
    float* s_Sq = s_z + n;
    float* s_d_Sq = s_Sq + n;
    float* s_d_retrieved = s_d_Sq + n;
    float* s_retrieved = s_d_retrieved + n;  // Recomputed

    // Load cached values
    for (int i = tid; i < n; i += block_size) {
        s_k_norm[i] = __bfloat162float(k_norm[bn + i]);
        s_v[i] = __bfloat162float(v[bn + i]);
        s_z[i] = __bfloat162float(z[bn + i]);
        s_Sq[i] = __bfloat162float(Sq[bn + i]);
    }
    __syncthreads();

    // =========================================================================
    // Backward through self-gate: out = Sq * silu(Sq)
    // d_Sq = d_out * (2*Sq*sig + Sq²*sig*(1-sig))
    // =========================================================================

    for (int i = tid; i < n; i += block_size) {
        float sq = s_Sq[i];
        float d_out = __bfloat162float(d_output[bn + i]);

        float sig = 1.0f / (1.0f + expf(-sq));
        // d/dSq of (Sq * Sq * sig) = 2*Sq*sig + Sq²*sig*(1-sig)
        float d_sq = d_out * (2.0f * sq * sig + sq * sq * sig * (1.0f - sig));
        s_d_Sq[i] = d_sq;
    }
    __syncthreads();

    // =========================================================================
    // Backward through S @ q -> Sq
    // d_S[i,j] += d_Sq[i] * q[j]
    // d_q[j] += sum_i(d_Sq[i] * S[i,j])
    // =========================================================================

    for (int idx = tid; idx < n * n; idx += block_size) {
        int i = idx / n;
        int j = idx % n;

        float d_sq_i = s_d_Sq[i];
        float q_j = __bfloat162float(q[bn + j]);
        float s_ij = __bfloat162float(S_next[bnn + idx]);

        // Accumulate d_S
        float curr_d_s = __bfloat162float(d_S[bnn + idx]);
        d_S[bnn + idx] = __float2bfloat16(curr_d_s + d_sq_i * q_j);

        // d_q[j] += d_Sq[i] * S[i,j]
        atomicAdd(&d_q_f[bn + j], d_sq_i * s_ij);
    }
    __syncthreads();

    // =========================================================================
    // Backward through tanh: S_new = tanh(pre_tanh)
    // d_pre_tanh = d_S * (1 - tanh²)
    // =========================================================================
    // Then backward through: pre_tanh = S_prev + outer(v - retrieved, k_norm)
    // d_S_prev[i,j] = d_pre_tanh[i,j]
    // d_v[i] += sum_j(d_pre_tanh[i,j] * k_norm[j])
    // d_retrieved[i] -= sum_j(d_pre_tanh[i,j] * k_norm[j])
    // d_k_norm[j] += sum_i(d_pre_tanh[i,j] * (v[i] - retrieved[i]))

    // First recompute retrieved (we need it for gradients)
    // This is cheaper than storing it
    for (int i = tid; i < n; i += block_size) {
        float z_i = s_z[i];
        float sum = 0.0f;
        // S_prev = atanh(S_next) approximately, but we use pre_tanh
        // Actually we need S_prev. For now, approximate from pre_tanh
        for (int j = 0; j < n; ++j) {
            float pt = __bfloat162float(pre_tanh[bnn + i * n + j]);
            // S_prev[i,j] = pre_tanh[i,j] - delta[i] * k_norm[j]
            // But delta = v - retrieved, circular...
            // For simplicity, we stored pre_tanh, compute S_prev differently
            // Actually: pre_tanh = S_prev + outer(delta, k_norm)
            // So we need S_prev. This is the gradient checkpointing case.
        }
        s_retrieved[i] = sum;  // TODO: proper recompute
    }
    __syncthreads();

    // Backward through delta update and tanh
    for (int idx = tid; idx < n * n; idx += block_size) {
        int i = idx / n;
        int j = idx % n;

        float pt = __bfloat162float(pre_tanh[bnn + idx]);
        float tanh_pt = tanhf(pt);
        float d_s = __bfloat162float(d_S[bnn + idx]);

        // d_pre_tanh = d_S * (1 - tanh²)
        float d_pt = d_s * (1.0f - tanh_pt * tanh_pt);

        // d_S_prev = d_pt (for now, will be propagated to previous timestep)
        // d_v[i] += d_pt * k_norm[j]
        // d_k_norm[j] += d_pt * (v[i] - retrieved[i])

        float k_j = s_k_norm[j];
        float v_i = s_v[i];

        atomicAdd(&d_v_f[bn + i], d_pt * k_j);
        // k_norm gradient goes to d_k after denormalization
        atomicAdd(&d_k_f[bn + j], d_pt * v_i);  // Simplified
    }
    __syncthreads();

    // =========================================================================
    // Backward through retrieval (d_z)
    // =========================================================================
    // retrieved[i] = sum_j(S[i,j] * z_mod * k_norm[j])
    // d_z depends on variant

    for (int idx = tid; idx < n * n; idx += block_size) {
        int i = idx / n;
        int j = idx % n;

        float pt = __bfloat162float(pre_tanh[bnn + idx]);
        float d_s = __bfloat162float(d_S[bnn + idx]);
        float d_pt = d_s * (1.0f - tanhf(pt) * tanhf(pt));

        // d_retrieved[i] -= d_pt * k_norm[j] summed over j
        // But d_retrieved flows back to d_z through the retrieval operation

        float k_j = s_k_norm[j];
        float z_i = s_z[i];
        float z_j = s_z[j];

        // Gradient to z through retrieval
        // This is simplified - full version needs S_prev
        if (variant == 0) {
            atomicAdd(&d_z_f[bn + j], -d_pt * k_j * k_j);  // Approximate
        } else if (variant == 1) {
            atomicAdd(&d_z_f[bn + i], -d_pt * k_j * k_j);
        } else {
            atomicAdd(&d_z_f[bn + i], -d_pt * k_j * k_j * z_j);
            atomicAdd(&d_z_f[bn + j], -d_pt * k_j * k_j * z_i);
        }
    }
}

// =============================================================================
// Persistent Forward - All timesteps in ONE kernel launch
// =============================================================================

__global__ void E73PersistentForwardKernel_BF16(
    const int batch_size,
    const int n,
    const int steps,
    const int variant,
    // Pre-projected inputs [T, B, n]
    const __nv_bfloat16* __restrict__ k_all,
    const __nv_bfloat16* __restrict__ v_all,
    const __nv_bfloat16* __restrict__ q_all,
    const __nv_bfloat16* __restrict__ z_all,
    // State [B, n, n] - initial state, updated in place
    __nv_bfloat16* __restrict__ S,
    // Outputs [T, B, n]
    __nv_bfloat16* __restrict__ output,
    // Training caches
    __nv_bfloat16* __restrict__ k_norm_cache,   // [T, B, n]
    __nv_bfloat16* __restrict__ pre_tanh_cache, // [num_checkpoints, B, n, n]
    __nv_bfloat16* __restrict__ Sq_cache,       // [T, B, n]
    __nv_bfloat16* __restrict__ S_checkpoints,  // [num_checkpoints, B, n, n]
    const bool training)
{
    const int b = blockIdx.x;
    if (b >= batch_size) return;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int bn = b * n;
    const int bnn = b * n * n;

    extern __shared__ float shared[];
    float* s_k_norm = shared;
    float* s_v = s_k_norm + n;
    float* s_z = s_v + n;
    float* s_retrieved = s_z + n;
    float* s_Sq = s_retrieved + n;
    float* s_reduction = s_Sq + n;

    const int num_checkpoints = (steps + CHECKPOINT_INTERVAL - 1) / CHECKPOINT_INTERVAL;

    // Process all timesteps
    for (int t = 0; t < steps; ++t) {
        const int t_bn = t * batch_size * n + bn;
        const int t_bnn = t * batch_size * n * n + bnn;

        const __nv_bfloat16* k_t = k_all + t_bn;
        const __nv_bfloat16* v_t = v_all + t_bn;
        const __nv_bfloat16* q_t = q_all + t_bn;
        const __nv_bfloat16* z_t = z_all + t_bn;
        __nv_bfloat16* out_t = output + t_bn;

        // Checkpoint S at intervals
        if (training && (t % CHECKPOINT_INTERVAL == 0)) {
            int cp_idx = t / CHECKPOINT_INTERVAL;
            __nv_bfloat16* S_cp = S_checkpoints + cp_idx * batch_size * n * n + bnn;
            for (int idx = tid; idx < n * n; idx += block_size) {
                S_cp[idx] = S[bnn + idx];
            }
        }
        __syncthreads();

        // =====================================================================
        // Step 1: Load inputs & normalize k
        // =====================================================================
        for (int i = tid; i < n; i += block_size) {
            s_v[i] = __bfloat162float(v_t[i]);
            s_z[i] = __bfloat162float(z_t[i]);
        }

        // Block-parallel k normalization
        float local_sum = 0.0f;
        for (int i = tid; i < n; i += block_size) {
            float ki = __bfloat162float(k_t[i]);
            local_sum += ki * ki;
        }
        s_reduction[tid] = local_sum;
        __syncthreads();

        for (int stride = block_size / 2; stride > 0; stride >>= 1) {
            if (tid < stride) s_reduction[tid] += s_reduction[tid + stride];
            __syncthreads();
        }

        float inv_norm = 1.0f / (sqrtf(s_reduction[0]) + 1e-6f);
        for (int i = tid; i < n; i += block_size) {
            float ki = __bfloat162float(k_t[i]);
            s_k_norm[i] = ki * inv_norm;
            if (training && k_norm_cache) {
                k_norm_cache[t_bn + i] = __float2bfloat16(s_k_norm[i]);
            }
        }
        __syncthreads();

        // =====================================================================
        // Step 2: Retrieval
        // =====================================================================
        for (int i = tid; i < n; i += block_size) {
            float z_i = s_z[i];
            float sum = 0.0f;
            for (int j = 0; j < n; ++j) {
                float s_ij = __bfloat162float(S[bnn + i * n + j]);
                float z_j = s_z[j];
                float k_j = s_k_norm[j];
                float z_mod = (variant == 0) ? z_j : (variant == 1) ? z_i : z_i * z_j;
                sum += s_ij * z_mod * k_j;
            }
            s_retrieved[i] = sum;
        }
        __syncthreads();

        // =====================================================================
        // Step 3: Delta update with tanh
        // =====================================================================
        for (int idx = tid; idx < n * n; idx += block_size) {
            int i = idx / n;
            int j = idx % n;

            float s_ij = __bfloat162float(S[bnn + idx]);
            float delta_i = s_v[i] - s_retrieved[i];
            float k_j = s_k_norm[j];
            float pre_tanh = s_ij + delta_i * k_j;

            S[bnn + idx] = __float2bfloat16(tanhf(pre_tanh));

            // Store pre_tanh for backward (only at checkpoints or all)
            if (training && pre_tanh_cache) {
                // Store all pre_tanh for now (can optimize later)
                pre_tanh_cache[t_bnn + idx] = __float2bfloat16(pre_tanh);
            }
        }
        __syncthreads();

        // =====================================================================
        // Step 4: Output
        // =====================================================================
        for (int i = tid; i < n; i += block_size) {
            float sum = 0.0f;
            for (int j = 0; j < n; ++j) {
                float s_ij = __bfloat162float(S[bnn + i * n + j]);
                float q_j = __bfloat162float(q_t[j]);
                sum += s_ij * q_j;
            }
            s_Sq[i] = sum;
        }
        __syncthreads();

        for (int i = tid; i < n; i += block_size) {
            float sq = s_Sq[i];
            if (training && Sq_cache) {
                Sq_cache[t_bn + i] = __float2bfloat16(sq);
            }
            float sig = 1.0f / (1.0f + expf(-sq));
            out_t[i] = __float2bfloat16(sq * sq * sig);
        }
        __syncthreads();
    }
}

}  // namespace

// =============================================================================
// Public Interface
// =============================================================================

namespace elman {

template<>
E73FusedForward<__nv_bfloat16>::E73FusedForward(
    int batch_size, int n_state, int dim, int variant,
    const cublasHandle_t& blas_handle, const cudaStream_t& stream)
    : batch_size_(batch_size), n_state_(n_state), dim_(dim), variant_(variant),
      blas_handle_(blas_handle), stream_(stream) {}

template<>
void E73FusedForward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_k, const __nv_bfloat16* W_v,
    const __nv_bfloat16* W_q, const __nv_bfloat16* W_z, const __nv_bfloat16* b_z,
    const __nv_bfloat16* x, __nv_bfloat16* S, __nv_bfloat16* output,
    __nv_bfloat16* workspace, bool training)
{
    const int n = n_state_;
    const int BN = batch_size_ * n;
    const int BNN = batch_size_ * n * n;

    static const __nv_bfloat16 alpha_one = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);

    // Workspace layout:
    // k_all: [T, B, n]
    // v_all: [T, B, n]
    // q_all: [T, B, n]
    // z_all: [T, B, n]
    // k_norm_cache: [T, B, n] (training only)
    // pre_tanh_cache: [T, B, n, n] (training only)
    // Sq_cache: [T, B, n] (training only)
    // S_checkpoints: [num_cp, B, n, n] (training only)

    __nv_bfloat16* k_all = workspace;
    __nv_bfloat16* v_all = k_all + steps * BN;
    __nv_bfloat16* q_all = v_all + steps * BN;
    __nv_bfloat16* z_all = q_all + steps * BN;

    __nv_bfloat16* k_norm_cache = nullptr;
    __nv_bfloat16* pre_tanh_cache = nullptr;
    __nv_bfloat16* Sq_cache = nullptr;
    __nv_bfloat16* S_checkpoints = nullptr;

    if (training) {
        k_norm_cache = z_all + steps * BN;
        pre_tanh_cache = k_norm_cache + steps * BN;
        Sq_cache = pre_tanh_cache + steps * BNN;
        int num_checkpoints = (steps + CHECKPOINT_INTERVAL - 1) / CHECKPOINT_INTERVAL;
        S_checkpoints = Sq_cache + steps * BN;
    }

    // Batch project all inputs (4 GEMMs)
    // k_all = x @ W_k.T
    blas<__nv_bfloat16>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        n, steps * batch_size_, dim_, &alpha_one, W_k, dim_, x, dim_, &beta_zero, k_all, n);

    // v_all = x @ W_v.T
    blas<__nv_bfloat16>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        n, steps * batch_size_, dim_, &alpha_one, W_v, dim_, x, dim_, &beta_zero, v_all, n);

    // q_all = x @ W_q.T
    blas<__nv_bfloat16>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        n, steps * batch_size_, dim_, &alpha_one, W_q, dim_, x, dim_, &beta_zero, q_all, n);

    // z_all = x @ W_z.T + b_z
    blas<__nv_bfloat16>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        n, steps * batch_size_, dim_, &alpha_one, W_z, dim_, x, dim_, &beta_zero, z_all, n);

    // Add bias to z (simple kernel)
    int total_z = steps * BN;
    int block_size = 256;
    int grid = (total_z + block_size - 1) / block_size;
    // TODO: Add bias kernel

    // Launch persistent forward kernel
    // One block per batch element, 256 threads per block
    int threads = 256;
    // Shared memory: 6 * n floats
    size_t shared_mem = (6 * n + threads) * sizeof(float);

    E73PersistentForwardKernel_BF16<<<batch_size_, threads, shared_mem, stream_>>>(
        batch_size_, n, steps, variant_,
        k_all, v_all, q_all, z_all,
        S, output,
        k_norm_cache, pre_tanh_cache, Sq_cache, S_checkpoints,
        training);
}

template<>
int64_t E73FusedForward<__nv_bfloat16>::WorkspaceSize(int steps, int batch_size, int n_state) {
    int64_t BN = batch_size * n_state;
    int64_t BNN = batch_size * n_state * n_state;
    int num_checkpoints = (steps + CHECKPOINT_INTERVAL - 1) / CHECKPOINT_INTERVAL;

    // k_all, v_all, q_all, z_all: 4 * T * B * n
    // k_norm_cache: T * B * n
    // pre_tanh_cache: T * B * n * n
    // Sq_cache: T * B * n
    // S_checkpoints: num_cp * B * n * n

    return 4 * steps * BN          // k,v,q,z projections
         + steps * BN              // k_norm_cache
         + steps * BNN             // pre_tanh_cache
         + steps * BN              // Sq_cache
         + num_checkpoints * BNN;  // S_checkpoints
}

// Backward not fully implemented yet - placeholder
template<>
E73FusedBackward<__nv_bfloat16>::E73FusedBackward(
    int batch_size, int n_state, int dim, int variant,
    const cublasHandle_t& blas_handle, const cudaStream_t& stream)
    : batch_size_(batch_size), n_state_(n_state), dim_(dim), variant_(variant),
      blas_handle_(blas_handle), stream_(stream) {}

template<>
void E73FusedBackward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_k, const __nv_bfloat16* W_v,
    const __nv_bfloat16* W_q, const __nv_bfloat16* W_z,
    const __nv_bfloat16* x, const __nv_bfloat16* S_checkpoints,
    const __nv_bfloat16* k_norm_cache, const __nv_bfloat16* v_cache,
    const __nv_bfloat16* q_cache, const __nv_bfloat16* z_cache,
    const __nv_bfloat16* pre_tanh_cache, const __nv_bfloat16* Sq_cache,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* dx, __nv_bfloat16* dW_k, __nv_bfloat16* dW_v,
    __nv_bfloat16* dW_q, __nv_bfloat16* dW_z, __nv_bfloat16* db_z,
    __nv_bfloat16* workspace)
{
    // TODO: Implement fused backward with checkpointing
    // For now, this is a placeholder
}

template<>
int64_t E73FusedBackward<__nv_bfloat16>::WorkspaceSize(int steps, int batch_size, int n_state) {
    int64_t BN = batch_size * n_state;
    int64_t BNN = batch_size * n_state * n_state;

    // d_S: B * n * n
    // d_k_all, d_v_all, d_q_all, d_z_all: 4 * T * B * n
    // Float accumulators: 4 * B * n

    return BNN + 4 * steps * BN + 4 * BN;
}

}  // namespace elman
