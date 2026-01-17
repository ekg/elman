/**
 * E75 Gated Delta Matrix CUDA Kernel
 *
 * Key innovation: Add forget gate to delta rule update
 *   β = sigmoid(W_β @ x + b_β)           # Per-row forget gate
 *   retrieved = S @ k_norm
 *   delta = v - retrieved
 *   S = tanh(β * S + outer(delta, k_norm))  # Forget + delta + tanh
 *
 * Based on E61/E68 insight: active forgetting is critical for performance.
 * Combines E74's associative memory with E61's gating mechanism.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include "hasty/elman_ladder.h"

#define CHECKPOINT_INTERVAL 16

namespace elman {

// ============================================================================
// Utility kernels
// ============================================================================

// Add bias and apply sigmoid: out[i] = sigmoid(out[i] + bias[i % n])
__global__ void E75_AddBiasSigmoid_BF16(
    __nv_bfloat16* __restrict__ data,
    const __nv_bfloat16* __restrict__ bias,
    int n,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int bias_idx = idx % n;
    float val = __bfloat162float(data[idx]) + __bfloat162float(bias[bias_idx]);
    float sig = 1.0f / (1.0f + expf(-val));
    data[idx] = __float2bfloat16(sig);
}

// Reduce gradients for bias: db[i] = sum over (T*B) of d_data[j*n + i]
__global__ void E75_ReduceBiasGrad_BF16(
    const __nv_bfloat16* __restrict__ d_data,
    __nv_bfloat16* __restrict__ db,
    int n,
    int T_B
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float sum = 0.0f;
    for (int tb = 0; tb < T_B; tb++) {
        sum += __bfloat162float(d_data[tb * n + i]);
    }
    db[i] = __float2bfloat16(sum);
}

// Apply sigmoid derivative: d_out[i] *= sigmoid_val[i] * (1 - sigmoid_val[i])
__global__ void E75_ApplySigmoidDeriv_BF16(
    __nv_bfloat16* __restrict__ d_data,
    const __nv_bfloat16* __restrict__ sigmoid_val,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float d = __bfloat162float(d_data[idx]);
    float s = __bfloat162float(sigmoid_val[idx]);
    float d_pre = d * s * (1.0f - s);
    d_data[idx] = __float2bfloat16(d_pre);
}

// ============================================================================
// E75 Forward Kernel - Gated Delta Update
// ============================================================================

template<int N_STATE>
__global__ void E75GatedDeltaForwardKernel_BF16(
    int T,
    int B,
    const __nv_bfloat16* __restrict__ k_all,      // [T, B, N_STATE]
    const __nv_bfloat16* __restrict__ v_all,      // [T, B, N_STATE]
    const __nv_bfloat16* __restrict__ q_all,      // [T, B, N_STATE]
    const __nv_bfloat16* __restrict__ beta_all,   // [T, B, N_STATE] forget gate (post-sigmoid)
    __nv_bfloat16* __restrict__ S,                // [B, N_STATE, N_STATE]
    __nv_bfloat16* __restrict__ output,           // [T, B, N_STATE]
    __nv_bfloat16* __restrict__ S_checkpoints,    // [num_checkpoints, B, N_STATE, N_STATE]
    __nv_bfloat16* __restrict__ Sq_cache,         // [T, B, N_STATE]
    int checkpoint_interval
) {
    int b = blockIdx.x;
    if (b >= B) return;

    // Shared memory layout
    extern __shared__ float shared_mem[];
    float* S_shared = shared_mem;                    // [N_STATE * N_STATE]
    float* k_shared = S_shared + N_STATE * N_STATE;  // [N_STATE]
    float* v_shared = k_shared + N_STATE;            // [N_STATE]
    float* q_shared = v_shared + N_STATE;            // [N_STATE]
    float* retrieved = q_shared + N_STATE;           // [N_STATE]
    float* beta_shared = retrieved + N_STATE;        // [N_STATE]

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;

    // Load initial state
    for (int i = tid; i < n2; i += blockDim.x) {
        S_shared[i] = __bfloat162float(S[b * n2 + i]);
    }
    __syncthreads();

    // Save initial checkpoint (index 0)
    for (int i = tid; i < n2; i += blockDim.x) {
        S_checkpoints[b * n2 + i] = __float2bfloat16(S_shared[i]);
    }
    __syncthreads();

    // Process each timestep
    for (int t = 0; t < T; t++) {
        // Load k, v, q, beta for this timestep
        if (tid < N_STATE) {
            k_shared[tid] = __bfloat162float(k_all[t * B * N_STATE + b * N_STATE + tid]);
            v_shared[tid] = __bfloat162float(v_all[t * B * N_STATE + b * N_STATE + tid]);
            q_shared[tid] = __bfloat162float(q_all[t * B * N_STATE + b * N_STATE + tid]);
            beta_shared[tid] = __bfloat162float(beta_all[t * B * N_STATE + b * N_STATE + tid]);
        }
        __syncthreads();

        // Normalize k
        __shared__ float k_norm_sq;
        if (tid == 0) {
            k_norm_sq = 0.0f;
            for (int i = 0; i < N_STATE; i++) {
                k_norm_sq += k_shared[i] * k_shared[i];
            }
            k_norm_sq = sqrtf(k_norm_sq) + 1e-6f;
        }
        __syncthreads();
        if (tid < N_STATE) {
            k_shared[tid] /= k_norm_sq;
        }
        __syncthreads();

        // Compute retrieved = S @ k_norm
        if (tid < N_STATE) {
            float sum = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < N_STATE; j++) {
                sum += S_shared[tid * N_STATE + j] * k_shared[j];
            }
            retrieved[tid] = sum;
        }
        __syncthreads();

        // Update state: S = tanh(β * S + outer(delta, k_norm))
        // where delta = v - retrieved
        for (int i = tid; i < n2; i += blockDim.x) {
            int row = i / N_STATE;
            int col = i % N_STATE;

            float beta_val = beta_shared[row];
            float delta_i = v_shared[row] - retrieved[row];
            float update = beta_val * S_shared[i] + delta_i * k_shared[col];
            S_shared[i] = tanhf(update);
        }
        __syncthreads();

        // Save checkpoint if at checkpoint boundary
        if ((t + 1) % checkpoint_interval == 0) {
            int cp_idx = (t + 1) / checkpoint_interval;
            for (int i = tid; i < n2; i += blockDim.x) {
                S_checkpoints[cp_idx * B * n2 + b * n2 + i] = __float2bfloat16(S_shared[i]);
            }
        }
        __syncthreads();

        // Compute output: Sq = S @ q, then self-gate
        if (tid < N_STATE) {
            float Sq = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < N_STATE; j++) {
                Sq += S_shared[tid * N_STATE + j] * q_shared[j];
            }
            // Cache Sq for backward
            Sq_cache[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(Sq);

            // Self-gating: Sq * silu(Sq) = Sq * Sq * sigmoid(Sq)
            float sig = 1.0f / (1.0f + expf(-Sq));
            float out_val = Sq * Sq * sig;
            output[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(out_val);
        }
        __syncthreads();
    }

    // Write final state back
    for (int i = tid; i < n2; i += blockDim.x) {
        S[b * n2 + i] = __float2bfloat16(S_shared[i]);
    }
}

// ============================================================================
// E75 Backward Kernel - Gated Delta Update
// ============================================================================

template<int N_STATE>
__global__ void E75GatedDeltaBackwardKernel_BF16(
    int T,
    int B,
    const __nv_bfloat16* __restrict__ k_all,
    const __nv_bfloat16* __restrict__ v_all,
    const __nv_bfloat16* __restrict__ q_all,
    const __nv_bfloat16* __restrict__ beta_all,
    const __nv_bfloat16* __restrict__ S_checkpoints,
    const __nv_bfloat16* __restrict__ Sq_cache,
    const __nv_bfloat16* __restrict__ d_output,
    __nv_bfloat16* __restrict__ d_k_all,
    __nv_bfloat16* __restrict__ d_v_all,
    __nv_bfloat16* __restrict__ d_q_all,
    __nv_bfloat16* __restrict__ d_beta_all,
    int checkpoint_interval
) {
    int b = blockIdx.x;
    if (b >= B) return;

    extern __shared__ float shared_mem[];
    // In-place update: only 2 matrices instead of 3
    float* S = shared_mem;                            // [N_STATE * N_STATE] - state (updated in-place)
    float* dS = S + N_STATE * N_STATE;                // [N_STATE * N_STATE] - gradient accumulator
    float* k_raw = dS + N_STATE * N_STATE;            // [N_STATE]
    float* v_raw = k_raw + N_STATE;                   // [N_STATE]
    float* q_raw = v_raw + N_STATE;                   // [N_STATE]
    float* k_norm = q_raw + N_STATE;                  // [N_STATE]
    float* delta = k_norm + N_STATE;                  // [N_STATE]
    float* retrieved = delta + N_STATE;               // [N_STATE]
    float* beta = retrieved + N_STATE;                // [N_STATE]
    float* d_k_raw = beta + N_STATE;                  // [N_STATE]
    float* d_v_raw = d_k_raw + N_STATE;               // [N_STATE]
    float* d_q_raw = d_v_raw + N_STATE;               // [N_STATE]
    float* d_Sq_shared = d_q_raw + N_STATE;           // [N_STATE]
    float* d_delta = d_Sq_shared + N_STATE;           // [N_STATE]
    float* d_k_norm = d_delta + N_STATE;              // [N_STATE]

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;

    // Initialize dS to zero
    for (int i = tid; i < n2; i += blockDim.x) {
        dS[i] = 0.0f;
    }
    __syncthreads();

    int num_segments = (T + checkpoint_interval - 1) / checkpoint_interval;

    for (int seg = num_segments - 1; seg >= 0; seg--) {
        int t_start = seg * checkpoint_interval;
        int t_end = min(t_start + checkpoint_interval, T);

        // Backward through segment
        for (int t = t_end - 1; t >= t_start; t--) {
            // Reload checkpoint into S (will be updated in-place during recomputation)
            for (int i = tid; i < n2; i += blockDim.x) {
                S[i] = __bfloat162float(S_checkpoints[seg * B * n2 + b * n2 + i]);
            }
            __syncthreads();

            // Recompute forward to step t
            __shared__ float k_norm_val_t;
            for (int tt = t_start; tt <= t; tt++) {
                if (tid < N_STATE) {
                    k_raw[tid] = __bfloat162float(k_all[tt * B * N_STATE + b * N_STATE + tid]);
                    v_raw[tid] = __bfloat162float(v_all[tt * B * N_STATE + b * N_STATE + tid]);
                    q_raw[tid] = __bfloat162float(q_all[tt * B * N_STATE + b * N_STATE + tid]);
                    beta[tid] = __bfloat162float(beta_all[tt * B * N_STATE + b * N_STATE + tid]);
                }
                __syncthreads();

                // Normalize k
                if (tid == 0) {
                    float sum_sq = 0.0f;
                    for (int i = 0; i < N_STATE; i++) sum_sq += k_raw[i] * k_raw[i];
                    k_norm_val_t = sqrtf(sum_sq) + 1e-6f;
                }
                __syncthreads();
                if (tid < N_STATE) {
                    k_norm[tid] = k_raw[tid] / k_norm_val_t;
                }
                __syncthreads();

                // Compute retrieved = S @ k_norm (S is current state)
                if (tid < N_STATE) {
                    float sum = 0.0f;
                    for (int j = 0; j < N_STATE; j++) {
                        sum += S[tid * N_STATE + j] * k_norm[j];
                    }
                    retrieved[tid] = sum;
                    delta[tid] = v_raw[tid] - retrieved[tid];
                }
                __syncthreads();

                // Update state IN-PLACE only for tt < t (keep S = S_{t-1} for backward)
                // When tt == t, we keep k, v, q, delta, k_norm, beta for backward but don't update S
                if (tt < t) {
                    for (int i = tid; i < n2; i += blockDim.x) {
                        int row = i / N_STATE;
                        int col = i % N_STATE;
                        float update = beta[row] * S[i] + delta[row] * k_norm[col];
                        S[i] = tanhf(update);
                    }
                    __syncthreads();
                }
            }

            // Now S holds S_{t-1} (state before step t)
            // k_norm, v_raw, q_raw, delta, beta are loaded for step t
            // We compute S_t on-the-fly when needed

            // Backward through output: out = Sq * silu(Sq) = Sq² * sigmoid(Sq)
            if (tid < N_STATE) {
                float d_out = __bfloat162float(d_output[t * B * N_STATE + b * N_STATE + tid]);
                float Sq = __bfloat162float(Sq_cache[t * B * N_STATE + b * N_STATE + tid]);
                float sig = 1.0f / (1.0f + expf(-Sq));
                // d_out/d_Sq for out = Sq² * sigmoid(Sq)
                // d/dSq = 2*Sq*sig + Sq²*sig*(1-sig) = Sq*sig*(2 + Sq*(1-sig))
                float d_Sq = d_out * (2.0f * Sq * sig + Sq * Sq * sig * (1.0f - sig));
                d_Sq_shared[tid] = d_Sq;
            }
            __syncthreads();

            // dS += outer(d_Sq, q) from Sq = S @ q
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                dS[i] += d_Sq_shared[row] * q_raw[col];
            }
            __syncthreads();

            // d_q = S_t^T @ d_Sq (compute S_t on-the-fly from S = S_{t-1})
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    // Compute S_t[i,tid] on-the-fly: S_t = tanh(β * S + outer(delta, k_norm))
                    float S_t_ij = tanhf(beta[i] * S[i * N_STATE + tid] + delta[i] * k_norm[tid]);
                    sum += S_t_ij * d_Sq_shared[i];
                }
                d_q_raw[tid] = sum;
            }
            __syncthreads();

            // Backward through state update: S_t = tanh(β * S + outer(delta, k))
            // where S = S_{t-1}
            // d_pre_tanh = dS * (1 - S_t²)
            // d_beta[i] = sum_j(d_pre_tanh[i,j] * S[i,j])  (uses S = S_{t-1})
            // d_delta[i] = sum_j(d_pre_tanh[i,j] * k_norm[j])
            // d_k_norm[j] = sum_i(d_pre_tanh[i,j] * delta[i])

            // Compute d_delta and d_beta (compute S_t on-the-fly for tanh derivative)
            if (tid < N_STATE) {
                float d_delta_local = 0.0f;
                float d_beta_local = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    // Compute S_t[tid,j] on-the-fly
                    float S_t_ij = tanhf(beta[tid] * S[tid * N_STATE + j] + delta[tid] * k_norm[j]);
                    float d_pre = dS[tid * N_STATE + j] * (1.0f - S_t_ij * S_t_ij);
                    d_delta_local += d_pre * k_norm[j];
                    d_beta_local += d_pre * S[tid * N_STATE + j];  // Uses S = S_{t-1}
                }
                d_delta[tid] = d_delta_local;
                // Store d_beta for this timestep
                d_beta_all[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(d_beta_local);
            }
            __syncthreads();

            // Compute d_k_norm (compute S_t on-the-fly for tanh derivative)
            if (tid < N_STATE) {
                float d_k_norm_local = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    // Compute S_t[i,tid] on-the-fly
                    float S_t_ij = tanhf(beta[i] * S[i * N_STATE + tid] + delta[i] * k_norm[tid]);
                    float d_pre = dS[i * N_STATE + tid] * (1.0f - S_t_ij * S_t_ij);
                    d_k_norm_local += d_pre * delta[i];
                }
                d_k_norm[tid] = d_k_norm_local;
            }
            __syncthreads();

            // d_v = d_delta (from delta = v - retrieved)
            // d_retrieved = -d_delta
            if (tid < N_STATE) {
                d_v_raw[tid] = d_delta[tid];
            }

            // d_k_norm += S^T @ (-d_delta) (from retrieved = S @ k_norm, where S = S_{t-1})
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    sum += S[i * N_STATE + tid] * (-d_delta[i]);
                }
                d_k_norm[tid] += sum;
            }
            __syncthreads();

            // Convert d_k_norm to d_k_raw via normalization gradient
            {
                __shared__ float k_dot_dk;
                if (tid == 0) {
                    k_dot_dk = 0.0f;
                    for (int i = 0; i < N_STATE; i++) {
                        k_dot_dk += k_raw[i] * d_k_norm[i];
                    }
                }
                __syncthreads();

                if (tid < N_STATE) {
                    float norm = k_norm_val_t;
                    float norm3 = norm * norm * norm;
                    d_k_raw[tid] = d_k_norm[tid] / norm - k_raw[tid] * k_dot_dk / norm3;
                }
                __syncthreads();
            }

            // Write gradients for k, v, q
            if (tid < N_STATE) {
                d_k_all[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(d_k_raw[tid]);
                d_v_all[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(d_v_raw[tid]);
                d_q_all[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(d_q_raw[tid]);
            }
            __syncthreads();

            // Update dS for next iteration:
            // dS_prev has two contributions:
            // 1. From pre_tanh = beta * S + outer(delta, k_norm): d_pre_tanh * beta
            // 2. From retrieved = S @ k_norm: outer(-d_delta, k_norm)
            // Need to compute S_t on-the-fly for tanh derivative
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                // Compute S_t on-the-fly
                float S_t_ij = tanhf(beta[row] * S[i] + delta[row] * k_norm[col]);
                float d_pre = dS[i] * (1.0f - S_t_ij * S_t_ij);
                // Contribution 1: gradient through beta * S (where S = S_{t-1})
                // Contribution 2: gradient through retrieved = S @ k_norm
                //                 d_retrieved = -d_delta, so dS += outer(-d_delta, k_norm)
                dS[i] = d_pre * beta[row] + (-d_delta[row]) * k_norm[col];
            }
            __syncthreads();
        }
    }
}

// ============================================================================
// E75GatedDeltaForward Implementation
// ============================================================================

template<typename DataT>
E75GatedDeltaForward<DataT>::E75GatedDeltaForward(
    bool training,
    int batch_size,
    int n_state,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : training_(training),
      batch_size_(batch_size),
      n_state_(n_state),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename DataT>
void E75GatedDeltaForward<DataT>::Run(
    int steps,
    const DataT* W_k,
    const DataT* W_v,
    const DataT* W_q,
    const DataT* W_beta,
    const DataT* b_beta,
    const DataT* x,
    DataT* S,
    DataT* output,
    DataT* k_cache,
    DataT* v_cache,
    DataT* q_cache,
    DataT* beta_cache,
    DataT* S_cache
) {
    int T = steps;
    int B = batch_size_;
    int n = n_state_;
    int d = dim_;

    const float alpha = 1.0f, beta_zero = 0.0f;

    // Project k, v, q
    cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        n, T * B, d, &alpha,
        W_k, CUDA_R_16BF, d, x, CUDA_R_16BF, d,
        &beta_zero, k_cache, CUDA_R_16BF, n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        n, T * B, d, &alpha,
        W_v, CUDA_R_16BF, d, x, CUDA_R_16BF, d,
        &beta_zero, v_cache, CUDA_R_16BF, n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        n, T * B, d, &alpha,
        W_q, CUDA_R_16BF, d, x, CUDA_R_16BF, d,
        &beta_zero, q_cache, CUDA_R_16BF, n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // Project beta = sigmoid(W_beta @ x + b_beta)
    cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        n, T * B, d, &alpha,
        W_beta, CUDA_R_16BF, d, x, CUDA_R_16BF, d,
        &beta_zero, beta_cache, CUDA_R_16BF, n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // Apply bias and sigmoid to beta
    int total = T * B * n;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    E75_AddBiasSigmoid_BF16<<<blocks, threads, 0, stream_>>>(
        (__nv_bfloat16*)beta_cache, (const __nv_bfloat16*)b_beta, n, total);

    // Calculate workspace offsets for checkpoints
    int num_checkpoints = (T + CHECKPOINT_INTERVAL - 1) / CHECKPOINT_INTERVAL + 1;
    DataT* s_checkpoints = S_cache;
    DataT* sq_cache = S_cache + num_checkpoints * B * n * n;

    // Run forward kernel
    int shared_size = (n * n + 6 * n) * sizeof(float);  // S + k,v,q,retrieved,beta
    int kernel_threads = min(256, n * n);

    #define DISPATCH_E75_FORWARD(N) \
        E75GatedDeltaForwardKernel_BF16<N><<<B, kernel_threads, shared_size, stream_>>>( \
            T, B, \
            (const __nv_bfloat16*)k_cache, \
            (const __nv_bfloat16*)v_cache, \
            (const __nv_bfloat16*)q_cache, \
            (const __nv_bfloat16*)beta_cache, \
            (__nv_bfloat16*)S, \
            (__nv_bfloat16*)output, \
            (__nv_bfloat16*)s_checkpoints, \
            (__nv_bfloat16*)sq_cache, \
            CHECKPOINT_INTERVAL)

    if (n == 1) { DISPATCH_E75_FORWARD(1); }
    else if (n == 2) { DISPATCH_E75_FORWARD(2); }
    else if (n == 4) { DISPATCH_E75_FORWARD(4); }
    else if (n == 8) { DISPATCH_E75_FORWARD(8); }
    else if (n == 16) { DISPATCH_E75_FORWARD(16); }
    else if (n == 24) { DISPATCH_E75_FORWARD(24); }
    else if (n == 28) { DISPATCH_E75_FORWARD(28); }
    else if (n == 32) { DISPATCH_E75_FORWARD(32); }
    else if (n == 48) { DISPATCH_E75_FORWARD(48); }
    else if (n == 64) { DISPATCH_E75_FORWARD(64); }
    else if (n == 96) { DISPATCH_E75_FORWARD(96); }
    else if (n == 128) { DISPATCH_E75_FORWARD(128); }
    else {
        fprintf(stderr, "E75 Forward: unsupported n_state=%d\n", n);
    }

    #undef DISPATCH_E75_FORWARD
}

template struct E75GatedDeltaForward<__nv_bfloat16>;

// ============================================================================
// E75GatedDeltaBackward Implementation
// ============================================================================

template<typename DataT>
E75GatedDeltaBackward<DataT>::E75GatedDeltaBackward(
    int batch_size,
    int n_state,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      n_state_(n_state),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename DataT>
void E75GatedDeltaBackward<DataT>::Run(
    int steps,
    const DataT* W_k,
    const DataT* W_v,
    const DataT* W_q,
    const DataT* W_beta,
    const DataT* x,
    const DataT* S_checkpoints,
    const DataT* Sq_cache,
    const DataT* k_cache,
    const DataT* v_cache,
    const DataT* q_cache,
    const DataT* beta_cache,
    const DataT* d_output,
    DataT* dx,
    DataT* dW_k,
    DataT* dW_v,
    DataT* dW_q,
    DataT* dW_beta,
    DataT* db_beta,
    DataT* workspace
) {
    int T = steps;
    int B = batch_size_;
    int n = n_state_;
    int d = dim_;

    // Workspace layout: [d_k_all: T*B*n] [d_v_all: T*B*n] [d_q_all: T*B*n] [d_beta_all: T*B*n]
    DataT* d_k_all = workspace;
    DataT* d_v_all = d_k_all + T * B * n;
    DataT* d_q_all = d_v_all + T * B * n;
    DataT* d_beta_all = d_q_all + T * B * n;

    // Shared memory: 2*n² + 13*n floats (in-place updates)
    // n=32: ~10KB, n=48: ~21KB, n=64: ~37KB, n=96: ~78KB
    int shared_size = (2 * n * n + 13 * n) * sizeof(float);
    int threads = min(256, n * n);

    // For n_state >= 64, need to set extended shared memory (exceeds default 48KB)
    // Use cudaFuncSetAttribute to request more shared memory
    // Note: n=96 requires 114KB (exceeds Ada 99KB), n=128 requires 200KB (exceeds Ampere 164KB)
    #define SET_SHARED_MEM_AND_DISPATCH_E75_BACKWARD(N) \
        { \
            cudaError_t attr_err = cudaFuncSetAttribute(E75GatedDeltaBackwardKernel_BF16<N>, \
                cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size); \
            if (attr_err != cudaSuccess) { \
                fprintf(stderr, "E75 Backward: n_state=%d requires %d KB shared memory but GPU limit exceeded (error: %s)\n", \
                    N, shared_size / 1024, cudaGetErrorString(attr_err)); \
                fprintf(stderr, "Try n_state <= 64 on Ada GPUs, or use Ampere for n_state=%d\n", N); \
            } else { \
                E75GatedDeltaBackwardKernel_BF16<N><<<B, threads, shared_size, stream_>>>( \
            T, B, \
            (const __nv_bfloat16*)k_cache, \
            (const __nv_bfloat16*)v_cache, \
            (const __nv_bfloat16*)q_cache, \
            (const __nv_bfloat16*)beta_cache, \
            (const __nv_bfloat16*)S_checkpoints, \
            (const __nv_bfloat16*)Sq_cache, \
            (const __nv_bfloat16*)d_output, \
            (__nv_bfloat16*)d_k_all, \
            (__nv_bfloat16*)d_v_all, \
            (__nv_bfloat16*)d_q_all, \
            (__nv_bfloat16*)d_beta_all, \
            CHECKPOINT_INTERVAL); \
            } \
        }

    // Run backward kernel
    #define DISPATCH_E75_BACKWARD(N) \
        E75GatedDeltaBackwardKernel_BF16<N><<<B, threads, shared_size, stream_>>>( \
            T, B, \
            (const __nv_bfloat16*)k_cache, \
            (const __nv_bfloat16*)v_cache, \
            (const __nv_bfloat16*)q_cache, \
            (const __nv_bfloat16*)beta_cache, \
            (const __nv_bfloat16*)S_checkpoints, \
            (const __nv_bfloat16*)Sq_cache, \
            (const __nv_bfloat16*)d_output, \
            (__nv_bfloat16*)d_k_all, \
            (__nv_bfloat16*)d_v_all, \
            (__nv_bfloat16*)d_q_all, \
            (__nv_bfloat16*)d_beta_all, \
            CHECKPOINT_INTERVAL)

    // For small n_state (<64), use regular dispatch; for n>=64, set extended shared memory (>48KB)
    if (n == 1) { DISPATCH_E75_BACKWARD(1); }
    else if (n == 2) { DISPATCH_E75_BACKWARD(2); }
    else if (n == 4) { DISPATCH_E75_BACKWARD(4); }
    else if (n == 8) { DISPATCH_E75_BACKWARD(8); }
    else if (n == 16) { DISPATCH_E75_BACKWARD(16); }
    else if (n == 24) { DISPATCH_E75_BACKWARD(24); }
    else if (n == 28) { DISPATCH_E75_BACKWARD(28); }
    else if (n == 32) { DISPATCH_E75_BACKWARD(32); }
    else if (n == 48) { DISPATCH_E75_BACKWARD(48); }  // 30KB fits in default 48KB
    // n >= 64 needs extended shared memory (>48KB default limit)
    else if (n == 64) { SET_SHARED_MEM_AND_DISPATCH_E75_BACKWARD(64); }
    else if (n == 96) { SET_SHARED_MEM_AND_DISPATCH_E75_BACKWARD(96); }
    else if (n == 128) { SET_SHARED_MEM_AND_DISPATCH_E75_BACKWARD(128); }
    else {
        fprintf(stderr, "E75 Backward: unsupported n_state=%d\n", n);
    }

    #undef SET_SHARED_MEM_AND_DISPATCH_E75_BACKWARD
    #undef DISPATCH_E75_BACKWARD

    // Apply sigmoid derivative to d_beta_all (beta was sigmoid in forward)
    int total_beta = T * B * n;
    int threads_deriv = 256;
    int blocks_deriv = (total_beta + threads_deriv - 1) / threads_deriv;
    E75_ApplySigmoidDeriv_BF16<<<blocks_deriv, threads_deriv, 0, stream_>>>(
        (__nv_bfloat16*)d_beta_all, (const __nv_bfloat16*)beta_cache, total_beta);

    // Weight gradients via cuBLAS
    const float alpha = 1.0f, beta_zero = 0.0f;

    // dW_k = x @ d_k_all^T
    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        d, n, T * B, &alpha,
        x, CUDA_R_16BF, d, d_k_all, CUDA_R_16BF, n,
        &beta_zero, dW_k, CUDA_R_16BF, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        d, n, T * B, &alpha,
        x, CUDA_R_16BF, d, d_v_all, CUDA_R_16BF, n,
        &beta_zero, dW_v, CUDA_R_16BF, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        d, n, T * B, &alpha,
        x, CUDA_R_16BF, d, d_q_all, CUDA_R_16BF, n,
        &beta_zero, dW_q, CUDA_R_16BF, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        d, n, T * B, &alpha,
        x, CUDA_R_16BF, d, d_beta_all, CUDA_R_16BF, n,
        &beta_zero, dW_beta, CUDA_R_16BF, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // db_beta = sum over (T*B) of d_beta_all
    int threads_db = 256;
    int blocks_db = (n + threads_db - 1) / threads_db;
    E75_ReduceBiasGrad_BF16<<<blocks_db, threads_db, 0, stream_>>>(
        (const __nv_bfloat16*)d_beta_all, (__nv_bfloat16*)db_beta, n, T * B);

    // dx = W_k @ d_k_all + W_v @ d_v_all + W_q @ d_q_all + W_beta @ d_beta_all
    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
        d, T * B, n, &alpha,
        W_k, CUDA_R_16BF, d, d_k_all, CUDA_R_16BF, n,
        &beta_zero, dx, CUDA_R_16BF, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    const float alpha_add = 1.0f, beta_add = 1.0f;
    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
        d, T * B, n, &alpha_add,
        W_v, CUDA_R_16BF, d, d_v_all, CUDA_R_16BF, n,
        &beta_add, dx, CUDA_R_16BF, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
        d, T * B, n, &alpha_add,
        W_q, CUDA_R_16BF, d, d_q_all, CUDA_R_16BF, n,
        &beta_add, dx, CUDA_R_16BF, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
        d, T * B, n, &alpha_add,
        W_beta, CUDA_R_16BF, d, d_beta_all, CUDA_R_16BF, n,
        &beta_add, dx, CUDA_R_16BF, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

template struct E75GatedDeltaBackward<__nv_bfloat16>;

}  // namespace elman
