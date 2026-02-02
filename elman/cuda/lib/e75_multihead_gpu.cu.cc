/**
 * E75 Multi-Head Gated Delta Matrix CUDA Kernel
 *
 * Multi-head version of E75 where each head maintains its own n_state x n_state
 * matrix state. H independent memory systems that can specialize.
 *
 * Per head h:
 *   k_h = W_k_h @ x, v_h = W_v_h @ x, q_h = W_q_h @ x
 *   beta_h = sigmoid(W_beta_h @ x + b_beta_h)
 *   k_norm = k_h / ||k_h||
 *   retrieved = S_h @ k_norm
 *   delta = v_h - retrieved
 *   S_h = tanh(beta_h * S_h + outer(delta, k_norm))
 *   out_h = S_h @ q_h
 *   out_h = out_h * silu(out_h)
 *
 * Output: concat(out_0, out_1, ..., out_{H-1})  [T, B, H * n_state]
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
// Utility kernels (same as E75 single-head)
// ============================================================================

__global__ void E75MH_AddBiasSigmoid_BF16(
    __nv_bfloat16* __restrict__ data,
    const __nv_bfloat16* __restrict__ bias,
    int H,
    int n,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // bias is [H, n], data is [T*B, H, n] (flattened)
    int bias_idx = idx % (H * n);
    float val = __bfloat162float(data[idx]) + __bfloat162float(bias[bias_idx]);
    float sig = 1.0f / (1.0f + expf(-val));
    data[idx] = __float2bfloat16(sig);
}

__global__ void E75MH_ReduceBiasGrad_BF16(
    const __nv_bfloat16* __restrict__ d_data,
    __nv_bfloat16* __restrict__ db,
    int H,
    int n,
    int T_B
) {
    // db has shape [H, n]
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H * n) return;

    float sum = 0.0f;
    for (int tb = 0; tb < T_B; tb++) {
        sum += __bfloat162float(d_data[tb * H * n + i]);
    }
    db[i] = __float2bfloat16(sum);
}

__global__ void E75MH_ApplySigmoidDeriv_BF16(
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
// E75 Multi-Head Forward Kernel
// Each block handles one (batch, head) pair
// ============================================================================

template<int N_STATE>
__global__ void E75MultiHeadForwardKernel_BF16(
    int T,
    int B,
    int H,
    const __nv_bfloat16* __restrict__ k_all,      // [T, B, H, N_STATE]
    const __nv_bfloat16* __restrict__ v_all,      // [T, B, H, N_STATE]
    const __nv_bfloat16* __restrict__ q_all,      // [T, B, H, N_STATE]
    const __nv_bfloat16* __restrict__ beta_all,   // [T, B, H, N_STATE]
    __nv_bfloat16* __restrict__ S,                // [B, H, N_STATE, N_STATE]
    __nv_bfloat16* __restrict__ output,           // [T, B, H, N_STATE]
    __nv_bfloat16* __restrict__ S_checkpoints,    // [num_checkpoints, B, H, N_STATE, N_STATE]
    __nv_bfloat16* __restrict__ Sq_cache,         // [T, B, H, N_STATE]
    int checkpoint_interval
) {
    int block_idx = blockIdx.x;
    int b = block_idx / H;
    int h = block_idx % H;
    if (b >= B) return;

    extern __shared__ float shared_mem[];
    float* S_shared = shared_mem;
    float* k_shared = S_shared + N_STATE * N_STATE;
    float* v_shared = k_shared + N_STATE;
    float* q_shared = v_shared + N_STATE;
    float* retrieved = q_shared + N_STATE;
    float* beta_shared = retrieved + N_STATE;

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;

    // State offset for this (batch, head)
    int state_offset = (b * H + h) * n2;

    // Load initial state for this head
    for (int i = tid; i < n2; i += blockDim.x) {
        S_shared[i] = __bfloat162float(S[state_offset + i]);
    }
    __syncthreads();

    // Save initial checkpoint (index 0)
    for (int i = tid; i < n2; i += blockDim.x) {
        S_checkpoints[(b * H + h) * n2 + i] = __float2bfloat16(S_shared[i]);
    }
    __syncthreads();

    for (int t = 0; t < T; t++) {
        // Data offset: [t, b, h, :]
        int data_offset = ((t * B + b) * H + h) * N_STATE;

        if (tid < N_STATE) {
            k_shared[tid] = __bfloat162float(k_all[data_offset + tid]);
            v_shared[tid] = __bfloat162float(v_all[data_offset + tid]);
            q_shared[tid] = __bfloat162float(q_all[data_offset + tid]);
            beta_shared[tid] = __bfloat162float(beta_all[data_offset + tid]);
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

        // retrieved = S @ k_norm
        if (tid < N_STATE) {
            float sum = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < N_STATE; j++) {
                sum += S_shared[tid * N_STATE + j] * k_shared[j];
            }
            retrieved[tid] = sum;
        }
        __syncthreads();

        // S = tanh(beta * S + outer(delta, k_norm))
        for (int i = tid; i < n2; i += blockDim.x) {
            int row = i / N_STATE;
            int col = i % N_STATE;
            float beta_val = beta_shared[row];
            float delta_i = v_shared[row] - retrieved[row];
            float update = beta_val * S_shared[i] + delta_i * k_shared[col];
            S_shared[i] = tanhf(update);
        }
        __syncthreads();

        // Save checkpoint
        if ((t + 1) % checkpoint_interval == 0) {
            int cp_idx = (t + 1) / checkpoint_interval;
            int cp_offset = (cp_idx * B * H + b * H + h) * n2;
            for (int i = tid; i < n2; i += blockDim.x) {
                S_checkpoints[cp_offset + i] = __float2bfloat16(S_shared[i]);
            }
        }
        __syncthreads();

        // Compute output: Sq = S @ q, self-gate
        if (tid < N_STATE) {
            float Sq = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < N_STATE; j++) {
                Sq += S_shared[tid * N_STATE + j] * q_shared[j];
            }
            Sq_cache[data_offset + tid] = __float2bfloat16(Sq);

            float sig = 1.0f / (1.0f + expf(-Sq));
            float out_val = Sq * Sq * sig;
            output[data_offset + tid] = __float2bfloat16(out_val);
        }
        __syncthreads();
    }

    // Write final state back
    for (int i = tid; i < n2; i += blockDim.x) {
        S[state_offset + i] = __float2bfloat16(S_shared[i]);
    }
}

// ============================================================================
// E75 Multi-Head Backward Kernel
// ============================================================================

template<int N_STATE>
__global__ void E75MultiHeadBackwardKernel_BF16(
    int T,
    int B,
    int H,
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
    int block_idx = blockIdx.x;
    int b = block_idx / H;
    int h = block_idx % H;
    if (b >= B) return;

    extern __shared__ float shared_mem[];
    float* S = shared_mem;
    float* dS = S + N_STATE * N_STATE;
    float* k_raw = dS + N_STATE * N_STATE;
    float* v_raw = k_raw + N_STATE;
    float* q_raw = v_raw + N_STATE;
    float* k_norm = q_raw + N_STATE;
    float* delta = k_norm + N_STATE;
    float* retrieved = delta + N_STATE;
    float* beta = retrieved + N_STATE;
    float* d_k_raw = beta + N_STATE;
    float* d_v_raw = d_k_raw + N_STATE;
    float* d_q_raw = d_v_raw + N_STATE;
    float* d_Sq_shared = d_q_raw + N_STATE;
    float* d_delta = d_Sq_shared + N_STATE;
    float* d_k_norm = d_delta + N_STATE;

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

        for (int t = t_end - 1; t >= t_start; t--) {
            // Reload checkpoint
            int cp_offset = (seg * B * H + b * H + h) * n2;
            for (int i = tid; i < n2; i += blockDim.x) {
                S[i] = __bfloat162float(S_checkpoints[cp_offset + i]);
            }
            __syncthreads();

            __shared__ float k_norm_val_t;
            for (int tt = t_start; tt <= t; tt++) {
                int data_offset = ((tt * B + b) * H + h) * N_STATE;

                if (tid < N_STATE) {
                    k_raw[tid] = __bfloat162float(k_all[data_offset + tid]);
                    v_raw[tid] = __bfloat162float(v_all[data_offset + tid]);
                    q_raw[tid] = __bfloat162float(q_all[data_offset + tid]);
                    beta[tid] = __bfloat162float(beta_all[data_offset + tid]);
                }
                __syncthreads();

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

                if (tid < N_STATE) {
                    float sum = 0.0f;
                    for (int j = 0; j < N_STATE; j++) {
                        sum += S[tid * N_STATE + j] * k_norm[j];
                    }
                    retrieved[tid] = sum;
                    delta[tid] = v_raw[tid] - retrieved[tid];
                }
                __syncthreads();

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

            // Backward through output
            int data_offset = ((t * B + b) * H + h) * N_STATE;
            if (tid < N_STATE) {
                float d_out = __bfloat162float(d_output[data_offset + tid]);
                float Sq = __bfloat162float(Sq_cache[data_offset + tid]);
                float sig = 1.0f / (1.0f + expf(-Sq));
                float d_Sq = d_out * (2.0f * Sq * sig + Sq * Sq * sig * (1.0f - sig));
                d_Sq_shared[tid] = d_Sq;
            }
            __syncthreads();

            // dS += outer(d_Sq, q)
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                dS[i] += d_Sq_shared[row] * q_raw[col];
            }
            __syncthreads();

            // d_q = S_t^T @ d_Sq
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    float S_t_ij = tanhf(beta[i] * S[i * N_STATE + tid] + delta[i] * k_norm[tid]);
                    sum += S_t_ij * d_Sq_shared[i];
                }
                d_q_raw[tid] = sum;
            }
            __syncthreads();

            // Backward through state update
            if (tid < N_STATE) {
                float d_delta_local = 0.0f;
                float d_beta_local = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    float S_t_ij = tanhf(beta[tid] * S[tid * N_STATE + j] + delta[tid] * k_norm[j]);
                    float d_pre = dS[tid * N_STATE + j] * (1.0f - S_t_ij * S_t_ij);
                    d_delta_local += d_pre * k_norm[j];
                    d_beta_local += d_pre * S[tid * N_STATE + j];
                }
                d_delta[tid] = d_delta_local;
                d_beta_all[data_offset + tid] = __float2bfloat16(d_beta_local);
            }
            __syncthreads();

            if (tid < N_STATE) {
                float d_k_norm_local = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    float S_t_ij = tanhf(beta[i] * S[i * N_STATE + tid] + delta[i] * k_norm[tid]);
                    float d_pre = dS[i * N_STATE + tid] * (1.0f - S_t_ij * S_t_ij);
                    d_k_norm_local += d_pre * delta[i];
                }
                d_k_norm[tid] = d_k_norm_local;
            }
            __syncthreads();

            if (tid < N_STATE) {
                d_v_raw[tid] = d_delta[tid];
            }

            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    sum += S[i * N_STATE + tid] * (-d_delta[i]);
                }
                d_k_norm[tid] += sum;
            }
            __syncthreads();

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

            // Write gradients
            if (tid < N_STATE) {
                d_k_all[data_offset + tid] = __float2bfloat16(d_k_raw[tid]);
                d_v_all[data_offset + tid] = __float2bfloat16(d_v_raw[tid]);
                d_q_all[data_offset + tid] = __float2bfloat16(d_q_raw[tid]);
            }
            __syncthreads();

            // Update dS for next iteration
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                float S_t_ij = tanhf(beta[row] * S[i] + delta[row] * k_norm[col]);
                float d_pre = dS[i] * (1.0f - S_t_ij * S_t_ij);
                dS[i] = d_pre * beta[row] + (-d_delta[row]) * k_norm[col];
            }
            __syncthreads();
        }
    }
}

// ============================================================================
// E75MultiHeadForward Implementation
// ============================================================================

template<typename DataT>
E75MultiHeadForward<DataT>::E75MultiHeadForward(
    bool training,
    int batch_size,
    int n_state,
    int n_heads,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : training_(training),
      batch_size_(batch_size),
      n_state_(n_state),
      n_heads_(n_heads),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename DataT>
void E75MultiHeadForward<DataT>::Run(
    int steps,
    const DataT* W_k,       // [H * n_state, dim]
    const DataT* W_v,       // [H * n_state, dim]
    const DataT* W_q,       // [H * n_state, dim]
    const DataT* W_beta,    // [H * n_state, dim]
    const DataT* b_beta,    // [H, n_state]
    const DataT* x,         // [T, B, dim]
    DataT* S,               // [B, H, n_state, n_state]
    DataT* output,          // [T, B, H * n_state]
    DataT* k_cache,         // [T, B, H, n_state]
    DataT* v_cache,
    DataT* q_cache,
    DataT* beta_cache,
    DataT* S_cache          // checkpoints + Sq_cache
) {
    int T = steps;
    int B = batch_size_;
    int n = n_state_;
    int H = n_heads_;
    int d = dim_;

    const float alpha = 1.0f, beta_zero = 0.0f;

    // Project k, v, q, beta for all heads at once
    // W_k is [H * n, d], x is [T*B, d], k_cache is [T*B, H*n] (then reshape)
    cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        H * n, T * B, d, &alpha,
        W_k, CUDA_R_16BF, d, x, CUDA_R_16BF, d,
        &beta_zero, k_cache, CUDA_R_16BF, H * n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        H * n, T * B, d, &alpha,
        W_v, CUDA_R_16BF, d, x, CUDA_R_16BF, d,
        &beta_zero, v_cache, CUDA_R_16BF, H * n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        H * n, T * B, d, &alpha,
        W_q, CUDA_R_16BF, d, x, CUDA_R_16BF, d,
        &beta_zero, q_cache, CUDA_R_16BF, H * n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        H * n, T * B, d, &alpha,
        W_beta, CUDA_R_16BF, d, x, CUDA_R_16BF, d,
        &beta_zero, beta_cache, CUDA_R_16BF, H * n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // Apply bias and sigmoid to beta
    int total = T * B * H * n;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    E75MH_AddBiasSigmoid_BF16<<<blocks, threads, 0, stream_>>>(
        (__nv_bfloat16*)beta_cache, (const __nv_bfloat16*)b_beta, H, n, total);

    // Workspace layout
    int num_checkpoints = (T + CHECKPOINT_INTERVAL - 1) / CHECKPOINT_INTERVAL + 1;
    DataT* s_checkpoints = S_cache;
    DataT* sq_cache = S_cache + num_checkpoints * B * H * n * n;

    // Run forward kernel - one block per (batch, head)
    int shared_size = (n * n + 6 * n) * sizeof(float);
    int kernel_threads = min(256, n * n);
    int num_blocks = B * H;

    #define DISPATCH_E75MH_FORWARD(N) \
        E75MultiHeadForwardKernel_BF16<N><<<num_blocks, kernel_threads, shared_size, stream_>>>( \
            T, B, H, \
            (const __nv_bfloat16*)k_cache, \
            (const __nv_bfloat16*)v_cache, \
            (const __nv_bfloat16*)q_cache, \
            (const __nv_bfloat16*)beta_cache, \
            (__nv_bfloat16*)S, \
            (__nv_bfloat16*)output, \
            (__nv_bfloat16*)s_checkpoints, \
            (__nv_bfloat16*)sq_cache, \
            CHECKPOINT_INTERVAL)

    // Supported n_state values: 8, 16, 24, 32, 40, 48, 56, 64
    // Note: n_state > 64 requires >48KB shared memory in backward pass, which exceeds
    // default limits on most GPUs. Use E88 for larger state sizes.
    if (n == 8) { DISPATCH_E75MH_FORWARD(8); }
    else if (n == 16) { DISPATCH_E75MH_FORWARD(16); }
    else if (n == 24) { DISPATCH_E75MH_FORWARD(24); }
    else if (n == 32) { DISPATCH_E75MH_FORWARD(32); }
    else if (n == 40) { DISPATCH_E75MH_FORWARD(40); }
    else if (n == 48) { DISPATCH_E75MH_FORWARD(48); }
    else if (n == 56) { DISPATCH_E75MH_FORWARD(56); }
    else if (n == 64) { DISPATCH_E75MH_FORWARD(64); }
    else {
        fprintf(stderr, "E75MultiHead Forward: unsupported n_state=%d. Supported values: 8, 16, 24, 32, 40, 48, 56, 64\n", n);
    }

    #undef DISPATCH_E75MH_FORWARD
}

template struct E75MultiHeadForward<__nv_bfloat16>;

// ============================================================================
// E75MultiHeadBackward Implementation
// ============================================================================

template<typename DataT>
E75MultiHeadBackward<DataT>::E75MultiHeadBackward(
    int batch_size,
    int n_state,
    int n_heads,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      n_state_(n_state),
      n_heads_(n_heads),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename DataT>
void E75MultiHeadBackward<DataT>::Run(
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
    int H = n_heads_;
    int d = dim_;

    // Workspace: [d_k: T*B*H*n] [d_v: T*B*H*n] [d_q: T*B*H*n] [d_beta: T*B*H*n]
    DataT* d_k_all = workspace;
    DataT* d_v_all = d_k_all + T * B * H * n;
    DataT* d_q_all = d_v_all + T * B * H * n;
    DataT* d_beta_all = d_q_all + T * B * H * n;

    int shared_size = (2 * n * n + 13 * n) * sizeof(float);
    int threads = min(256, n * n);
    int num_blocks = B * H;

    #define DISPATCH_E75MH_BACKWARD(N) \
        E75MultiHeadBackwardKernel_BF16<N><<<num_blocks, threads, shared_size, stream_>>>( \
            T, B, H, \
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

    // Supported n_state values: 8, 16, 24, 32, 40, 48, 56, 64
    // Note: n_state > 64 requires >48KB shared memory, exceeding default GPU limits.
    if (n == 8) { DISPATCH_E75MH_BACKWARD(8); }
    else if (n == 16) { DISPATCH_E75MH_BACKWARD(16); }
    else if (n == 24) { DISPATCH_E75MH_BACKWARD(24); }
    else if (n == 32) { DISPATCH_E75MH_BACKWARD(32); }
    else if (n == 40) { DISPATCH_E75MH_BACKWARD(40); }
    else if (n == 48) { DISPATCH_E75MH_BACKWARD(48); }
    else if (n == 56) { DISPATCH_E75MH_BACKWARD(56); }
    else if (n == 64) { DISPATCH_E75MH_BACKWARD(64); }
    else {
        fprintf(stderr, "E75MultiHead Backward: unsupported n_state=%d. Supported values: 8, 16, 24, 32, 40, 48, 56, 64\n", n);
    }

    #undef DISPATCH_E75MH_BACKWARD

    // Apply sigmoid derivative
    int total_beta = T * B * H * n;
    int threads_deriv = 256;
    int blocks_deriv = (total_beta + threads_deriv - 1) / threads_deriv;
    E75MH_ApplySigmoidDeriv_BF16<<<blocks_deriv, threads_deriv, 0, stream_>>>(
        (__nv_bfloat16*)d_beta_all, (const __nv_bfloat16*)beta_cache, total_beta);

    // Weight gradients
    const float alpha = 1.0f, beta_zero = 0.0f;

    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        d, H * n, T * B, &alpha,
        x, CUDA_R_16BF, d, d_k_all, CUDA_R_16BF, H * n,
        &beta_zero, dW_k, CUDA_R_16BF, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        d, H * n, T * B, &alpha,
        x, CUDA_R_16BF, d, d_v_all, CUDA_R_16BF, H * n,
        &beta_zero, dW_v, CUDA_R_16BF, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        d, H * n, T * B, &alpha,
        x, CUDA_R_16BF, d, d_q_all, CUDA_R_16BF, H * n,
        &beta_zero, dW_q, CUDA_R_16BF, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        d, H * n, T * B, &alpha,
        x, CUDA_R_16BF, d, d_beta_all, CUDA_R_16BF, H * n,
        &beta_zero, dW_beta, CUDA_R_16BF, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // db_beta = sum over T*B
    int threads_db = 256;
    int blocks_db = (H * n + threads_db - 1) / threads_db;
    E75MH_ReduceBiasGrad_BF16<<<blocks_db, threads_db, 0, stream_>>>(
        (const __nv_bfloat16*)d_beta_all, (__nv_bfloat16*)db_beta, H, n, T * B);

    // dx = W_k @ d_k + W_v @ d_v + W_q @ d_q + W_beta @ d_beta
    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
        d, T * B, H * n, &alpha,
        W_k, CUDA_R_16BF, d, d_k_all, CUDA_R_16BF, H * n,
        &beta_zero, dx, CUDA_R_16BF, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    const float alpha_add = 1.0f, beta_add = 1.0f;
    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
        d, T * B, H * n, &alpha_add,
        W_v, CUDA_R_16BF, d, d_v_all, CUDA_R_16BF, H * n,
        &beta_add, dx, CUDA_R_16BF, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
        d, T * B, H * n, &alpha_add,
        W_q, CUDA_R_16BF, d, d_q_all, CUDA_R_16BF, H * n,
        &beta_add, dx, CUDA_R_16BF, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
        d, T * B, H * n, &alpha_add,
        W_beta, CUDA_R_16BF, d, d_beta_all, CUDA_R_16BF, H * n,
        &beta_add, dx, CUDA_R_16BF, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

template struct E75MultiHeadBackward<__nv_bfloat16>;

// ============================================================================
// E75MultiHeadPrecomputedForward Implementation
// For post-projection convolutions: accepts pre-computed k, v, q, beta
// ============================================================================

template<typename DataT>
E75MultiHeadPrecomputedForward<DataT>::E75MultiHeadPrecomputedForward(
    bool training,
    int batch_size,
    int n_state,
    int n_heads,
    const cudaStream_t& stream)
    : training_(training),
      batch_size_(batch_size),
      n_state_(n_state),
      n_heads_(n_heads),
      stream_(stream) {}

template<typename DataT>
void E75MultiHeadPrecomputedForward<DataT>::Run(
    int steps,
    const DataT* k,         // [T, B, H, n_state] pre-computed
    const DataT* v,         // [T, B, H, n_state] pre-computed
    const DataT* q,         // [T, B, H, n_state] pre-computed
    const DataT* beta,      // [T, B, H, n_state] pre-computed (sigmoid already applied)
    DataT* S,               // [B, H, n_state, n_state]
    DataT* output,          // [T, B, H, n_state]
    DataT* S_cache          // checkpoints + Sq_cache
) {
    int T = steps;
    int B = batch_size_;
    int n = n_state_;
    int H = n_heads_;

    // Workspace layout
    int num_checkpoints = (T + CHECKPOINT_INTERVAL - 1) / CHECKPOINT_INTERVAL + 1;
    DataT* s_checkpoints = S_cache;
    DataT* sq_cache = S_cache + num_checkpoints * B * H * n * n;

    // Run forward kernel - one block per (batch, head)
    // Re-use the existing forward kernel since it already accepts k, v, q, beta as inputs
    int shared_size = (n * n + 6 * n) * sizeof(float);
    int kernel_threads = min(256, n * n);
    int num_blocks = B * H;

    #define DISPATCH_E75MH_PRECOMPUTED_FWD(N) \
        E75MultiHeadForwardKernel_BF16<N><<<num_blocks, kernel_threads, shared_size, stream_>>>( \
            T, B, H, \
            (const __nv_bfloat16*)k, \
            (const __nv_bfloat16*)v, \
            (const __nv_bfloat16*)q, \
            (const __nv_bfloat16*)beta, \
            (__nv_bfloat16*)S, \
            (__nv_bfloat16*)output, \
            (__nv_bfloat16*)s_checkpoints, \
            (__nv_bfloat16*)sq_cache, \
            CHECKPOINT_INTERVAL)

    // Supported n_state values: 8, 16, 24, 32, 40, 48, 56, 64
    if (n == 8) { DISPATCH_E75MH_PRECOMPUTED_FWD(8); }
    else if (n == 16) { DISPATCH_E75MH_PRECOMPUTED_FWD(16); }
    else if (n == 24) { DISPATCH_E75MH_PRECOMPUTED_FWD(24); }
    else if (n == 32) { DISPATCH_E75MH_PRECOMPUTED_FWD(32); }
    else if (n == 40) { DISPATCH_E75MH_PRECOMPUTED_FWD(40); }
    else if (n == 48) { DISPATCH_E75MH_PRECOMPUTED_FWD(48); }
    else if (n == 56) { DISPATCH_E75MH_PRECOMPUTED_FWD(56); }
    else if (n == 64) { DISPATCH_E75MH_PRECOMPUTED_FWD(64); }
    else {
        fprintf(stderr, "E75MultiHeadPrecomputed Forward: unsupported n_state=%d. Supported values: 8, 16, 24, 32, 40, 48, 56, 64\n", n);
    }

    #undef DISPATCH_E75MH_PRECOMPUTED_FWD
}

template struct E75MultiHeadPrecomputedForward<__nv_bfloat16>;

// ============================================================================
// E75MultiHeadPrecomputedBackward Implementation
// Computes gradients for pre-computed k, v, q, beta
// ============================================================================

template<typename DataT>
E75MultiHeadPrecomputedBackward<DataT>::E75MultiHeadPrecomputedBackward(
    int batch_size,
    int n_state,
    int n_heads,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      n_state_(n_state),
      n_heads_(n_heads),
      stream_(stream) {}

template<typename DataT>
void E75MultiHeadPrecomputedBackward<DataT>::Run(
    int steps,
    const DataT* k,
    const DataT* v,
    const DataT* q,
    const DataT* beta,
    const DataT* S_checkpoints,
    const DataT* Sq_cache,
    const DataT* d_output,
    DataT* d_k,
    DataT* d_v,
    DataT* d_q,
    DataT* d_beta
) {
    int T = steps;
    int B = batch_size_;
    int n = n_state_;
    int H = n_heads_;

    int shared_size = (2 * n * n + 13 * n) * sizeof(float);
    int threads = min(256, n * n);
    int num_blocks = B * H;

    // Re-use the existing backward kernel
    #define DISPATCH_E75MH_PRECOMPUTED_BWD(N) \
        E75MultiHeadBackwardKernel_BF16<N><<<num_blocks, threads, shared_size, stream_>>>( \
            T, B, H, \
            (const __nv_bfloat16*)k, \
            (const __nv_bfloat16*)v, \
            (const __nv_bfloat16*)q, \
            (const __nv_bfloat16*)beta, \
            (const __nv_bfloat16*)S_checkpoints, \
            (const __nv_bfloat16*)Sq_cache, \
            (const __nv_bfloat16*)d_output, \
            (__nv_bfloat16*)d_k, \
            (__nv_bfloat16*)d_v, \
            (__nv_bfloat16*)d_q, \
            (__nv_bfloat16*)d_beta, \
            CHECKPOINT_INTERVAL)

    // Supported n_state values: 8, 16, 24, 32, 40, 48, 56, 64
    if (n == 8) { DISPATCH_E75MH_PRECOMPUTED_BWD(8); }
    else if (n == 16) { DISPATCH_E75MH_PRECOMPUTED_BWD(16); }
    else if (n == 24) { DISPATCH_E75MH_PRECOMPUTED_BWD(24); }
    else if (n == 32) { DISPATCH_E75MH_PRECOMPUTED_BWD(32); }
    else if (n == 40) { DISPATCH_E75MH_PRECOMPUTED_BWD(40); }
    else if (n == 48) { DISPATCH_E75MH_PRECOMPUTED_BWD(48); }
    else if (n == 56) { DISPATCH_E75MH_PRECOMPUTED_BWD(56); }
    else if (n == 64) { DISPATCH_E75MH_PRECOMPUTED_BWD(64); }
    else {
        fprintf(stderr, "E75MultiHeadPrecomputed Backward: unsupported n_state=%d. Supported values: 8, 16, 24, 32, 40, 48, 56, 64\n", n);
    }

    #undef DISPATCH_E75MH_PRECOMPUTED_BWD
}

template struct E75MultiHeadPrecomputedBackward<__nv_bfloat16>;

}  // namespace elman
