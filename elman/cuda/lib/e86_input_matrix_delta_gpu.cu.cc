/**
 * E86 Input-as-Matrix Delta Rule CUDA Kernel (Multi-Head)
 *
 * Combines E85's input-as-matrix with E75's delta rule.
 * Supports multiple heads for capacity scaling.
 *
 * Architecture (per head h):
 *   A_h = x_h.view(n, n)           # Each head's input matrix
 *   k_h = A_h.mean(dim=1)          # Row means as key
 *   v_h = A_h.mean(dim=0)          # Col means as value
 *   beta_h = sigmoid(scale * A_h.mean() + bias)
 *   k_norm_h = k_h / ||k_h||
 *   retrieved_h = S_h @ k_norm_h
 *   delta_h = v_h - retrieved_h
 *   S_h = tanh(beta_h * S_h + outer(delta_h, k_norm_h))
 *   Sq_h = S_h @ k_norm_h
 *   out_h = Sq_h * silu(Sq_h)
 *
 * Memory layout:
 *   x: [T, B, H * n²]
 *   S: [B, H * n²] (flattened from [B, H, n, n])
 *   output: [T, B, H * n]
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include "hasty/elman_ladder.h"

#define CHECKPOINT_INTERVAL 16

namespace elman {

// ============================================================================
// E86 Forward Kernel - Input-as-Matrix Delta Rule (Multi-Head)
// One block per (batch, head) pair
// ============================================================================

template<int N_STATE>
__global__ void E86InputMatrixDeltaForwardKernel_BF16(
    int T,
    int B,
    int H,                                    // Number of heads
    const __nv_bfloat16* __restrict__ x_all,  // [T, B, H * N_STATE * N_STATE]
    float scale,
    float bias,
    __nv_bfloat16* __restrict__ S,            // [B, H * N_STATE * N_STATE]
    __nv_bfloat16* __restrict__ output,       // [T, B, H * N_STATE]
    __nv_bfloat16* __restrict__ k_cache,      // [T, B, H * N_STATE]
    __nv_bfloat16* __restrict__ v_cache,      // [T, B, H * N_STATE]
    __nv_bfloat16* __restrict__ beta_cache,   // [T, B, H]
    __nv_bfloat16* __restrict__ S_checkpoints,// [num_checkpoints, B, H * N_STATE * N_STATE]
    __nv_bfloat16* __restrict__ Sq_cache,     // [T, B, H * N_STATE]
    int checkpoint_interval
) {
    // Each block handles one (batch, head) pair
    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;
    if (b >= B) return;

    // Shared memory layout
    extern __shared__ float shared_mem[];
    float* S_shared = shared_mem;                     // [N_STATE * N_STATE]
    float* A_shared = S_shared + N_STATE * N_STATE;   // [N_STATE * N_STATE]
    float* k_shared = A_shared + N_STATE * N_STATE;   // [N_STATE]
    float* v_shared = k_shared + N_STATE;             // [N_STATE]
    float* retrieved = v_shared + N_STATE;            // [N_STATE]

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;
    int H_n2 = H * n2;
    int H_n = H * N_STATE;

    // Load initial state for this head
    for (int i = tid; i < n2; i += blockDim.x) {
        S_shared[i] = __bfloat162float(S[b * H_n2 + h * n2 + i]);
    }
    __syncthreads();

    // Save initial checkpoint (index 0)
    for (int i = tid; i < n2; i += blockDim.x) {
        S_checkpoints[b * H_n2 + h * n2 + i] = __float2bfloat16(S_shared[i]);
    }
    __syncthreads();

    // Process each timestep
    for (int t = 0; t < T; t++) {
        // Load input matrix A for this head: x[t, b, h*n²:(h+1)*n²]
        for (int i = tid; i < n2; i += blockDim.x) {
            A_shared[i] = __bfloat162float(x_all[t * B * H_n2 + b * H_n2 + h * n2 + i]);
        }
        __syncthreads();

        // Compute k = A.mean(dim=1) (row means) and v = A.mean(dim=0) (col means)
        // Also compute A_mean for beta
        __shared__ float A_sum;
        if (tid == 0) {
            A_sum = 0.0f;
        }
        __syncthreads();

        if (tid < N_STATE) {
            // k[tid] = mean of row tid
            float row_sum = 0.0f;
            for (int j = 0; j < N_STATE; j++) {
                row_sum += A_shared[tid * N_STATE + j];
            }
            k_shared[tid] = row_sum / N_STATE;

            // v[tid] = mean of col tid
            float col_sum = 0.0f;
            for (int i = 0; i < N_STATE; i++) {
                col_sum += A_shared[i * N_STATE + tid];
            }
            v_shared[tid] = col_sum / N_STATE;

            // Accumulate for A_mean
            atomicAdd(&A_sum, row_sum);
        }
        __syncthreads();

        // Compute beta = sigmoid(scale * A_mean + bias)
        __shared__ float beta_val;
        __shared__ float k_norm_val;
        if (tid == 0) {
            float A_mean = A_sum / n2;
            beta_val = 1.0f / (1.0f + expf(-(scale * A_mean + bias)));
            beta_cache[t * B * H + b * H + h] = __float2bfloat16(beta_val);

            // Compute k norm
            float k_sum_sq = 0.0f;
            for (int i = 0; i < N_STATE; i++) {
                k_sum_sq += k_shared[i] * k_shared[i];
            }
            k_norm_val = sqrtf(k_sum_sq) + 1e-6f;
        }
        __syncthreads();

        // Normalize k and cache
        if (tid < N_STATE) {
            k_shared[tid] /= k_norm_val;
            k_cache[t * B * H_n + b * H_n + h * N_STATE + tid] = __float2bfloat16(k_shared[tid] * k_norm_val);
            v_cache[t * B * H_n + b * H_n + h * N_STATE + tid] = __float2bfloat16(v_shared[tid]);
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

        // Update state: S = tanh(beta * S + outer(delta, k_norm))
        for (int i = tid; i < n2; i += blockDim.x) {
            int row = i / N_STATE;
            int col = i % N_STATE;

            float delta_i = v_shared[row] - retrieved[row];
            float update = beta_val * S_shared[i] + delta_i * k_shared[col];
            S_shared[i] = tanhf(update);
        }
        __syncthreads();

        // Save checkpoint if at checkpoint boundary
        if ((t + 1) % checkpoint_interval == 0) {
            int cp_idx = (t + 1) / checkpoint_interval;
            for (int i = tid; i < n2; i += blockDim.x) {
                S_checkpoints[cp_idx * B * H_n2 + b * H_n2 + h * n2 + i] = __float2bfloat16(S_shared[i]);
            }
        }
        __syncthreads();

        // Compute output: Sq = S @ k_norm, then self-gate
        if (tid < N_STATE) {
            float Sq = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < N_STATE; j++) {
                Sq += S_shared[tid * N_STATE + j] * k_shared[j];
            }
            // Cache Sq for backward
            Sq_cache[t * B * H_n + b * H_n + h * N_STATE + tid] = __float2bfloat16(Sq);

            // Self-gating: Sq * silu(Sq) = Sq * Sq * sigmoid(Sq)
            float sig = 1.0f / (1.0f + expf(-Sq));
            float out_val = Sq * Sq * sig;
            output[t * B * H_n + b * H_n + h * N_STATE + tid] = __float2bfloat16(out_val);
        }
        __syncthreads();
    }

    // Write final state back
    for (int i = tid; i < n2; i += blockDim.x) {
        S[b * H_n2 + h * n2 + i] = __float2bfloat16(S_shared[i]);
    }
}

// ============================================================================
// E86 Backward Kernel - Input-as-Matrix Delta Rule (Multi-Head)
// ============================================================================

template<int N_STATE>
__global__ void E86InputMatrixDeltaBackwardKernel_BF16(
    int T,
    int B,
    int H,
    const __nv_bfloat16* __restrict__ x_all,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const __nv_bfloat16* __restrict__ beta_cache,
    const __nv_bfloat16* __restrict__ S_checkpoints,
    const __nv_bfloat16* __restrict__ Sq_cache,
    const __nv_bfloat16* __restrict__ d_output,
    float scale,
    float bias,
    __nv_bfloat16* __restrict__ dx_all,
    float* __restrict__ d_scale_accum,
    float* __restrict__ d_bias_accum,
    int checkpoint_interval
) {
    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;
    if (b >= B) return;

    extern __shared__ float shared_mem[];
    float* S = shared_mem;                             // [N_STATE * N_STATE]
    float* dS = S + N_STATE * N_STATE;                 // [N_STATE * N_STATE]
    float* A_shared = dS + N_STATE * N_STATE;          // [N_STATE * N_STATE]
    float* dA_shared = A_shared + N_STATE * N_STATE;   // [N_STATE * N_STATE]
    float* k_raw = dA_shared + N_STATE * N_STATE;      // [N_STATE]
    float* v_raw = k_raw + N_STATE;                    // [N_STATE]
    float* k_norm = v_raw + N_STATE;                   // [N_STATE]
    float* delta = k_norm + N_STATE;                   // [N_STATE]
    float* retrieved = delta + N_STATE;                // [N_STATE]
    float* d_k_norm = retrieved + N_STATE;             // [N_STATE]
    float* d_v_raw = d_k_norm + N_STATE;               // [N_STATE]
    float* d_delta = d_v_raw + N_STATE;                // [N_STATE]
    float* d_Sq_shared = d_delta + N_STATE;            // [N_STATE]

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;
    int H_n2 = H * n2;
    int H_n = H * N_STATE;

    // Initialize dS and accumulators
    for (int i = tid; i < n2; i += blockDim.x) {
        dS[i] = 0.0f;
    }
    __shared__ float d_scale_local;
    __shared__ float d_bias_local;
    if (tid == 0) {
        d_scale_local = 0.0f;
        d_bias_local = 0.0f;
    }
    __syncthreads();

    int num_segments = (T + checkpoint_interval - 1) / checkpoint_interval;

    for (int seg = num_segments - 1; seg >= 0; seg--) {
        int t_start = seg * checkpoint_interval;
        int t_end = min(t_start + checkpoint_interval, T);

        for (int t = t_end - 1; t >= t_start; t--) {
            // Reload checkpoint
            for (int i = tid; i < n2; i += blockDim.x) {
                S[i] = __bfloat162float(S_checkpoints[seg * B * H_n2 + b * H_n2 + h * n2 + i]);
            }
            __syncthreads();

            // Recompute forward to step t
            __shared__ float k_norm_val_t;
            __shared__ float beta_val_t;

            for (int tt = t_start; tt <= t; tt++) {
                // Load A
                for (int i = tid; i < n2; i += blockDim.x) {
                    A_shared[i] = __bfloat162float(x_all[tt * B * H_n2 + b * H_n2 + h * n2 + i]);
                }
                __syncthreads();

                // Compute k, v, beta
                __shared__ float A_sum_t;
                if (tid == 0) A_sum_t = 0.0f;
                __syncthreads();

                if (tid < N_STATE) {
                    float row_sum = 0.0f;
                    for (int j = 0; j < N_STATE; j++) {
                        row_sum += A_shared[tid * N_STATE + j];
                    }
                    k_raw[tid] = row_sum / N_STATE;

                    float col_sum = 0.0f;
                    for (int i = 0; i < N_STATE; i++) {
                        col_sum += A_shared[i * N_STATE + tid];
                    }
                    v_raw[tid] = col_sum / N_STATE;

                    atomicAdd(&A_sum_t, row_sum);
                }
                __syncthreads();

                if (tid == 0) {
                    float A_mean = A_sum_t / n2;
                    beta_val_t = 1.0f / (1.0f + expf(-(scale * A_mean + bias)));

                    float k_sum_sq = 0.0f;
                    for (int i = 0; i < N_STATE; i++) {
                        k_sum_sq += k_raw[i] * k_raw[i];
                    }
                    k_norm_val_t = sqrtf(k_sum_sq) + 1e-6f;
                }
                __syncthreads();

                if (tid < N_STATE) {
                    k_norm[tid] = k_raw[tid] / k_norm_val_t;
                }
                __syncthreads();

                // Compute retrieved and delta
                if (tid < N_STATE) {
                    float sum = 0.0f;
                    for (int j = 0; j < N_STATE; j++) {
                        sum += S[tid * N_STATE + j] * k_norm[j];
                    }
                    retrieved[tid] = sum;
                    delta[tid] = v_raw[tid] - retrieved[tid];
                }
                __syncthreads();

                // Update S (except at tt == t, keep S = S_{t-1})
                if (tt < t) {
                    for (int i = tid; i < n2; i += blockDim.x) {
                        int row = i / N_STATE;
                        int col = i % N_STATE;
                        float update = beta_val_t * S[i] + delta[row] * k_norm[col];
                        S[i] = tanhf(update);
                    }
                    __syncthreads();
                }
            }

            // Now S = S_{t-1}, k_norm, v_raw, delta, beta_val_t are for step t

            // Backward through output
            if (tid < N_STATE) {
                float d_out = __bfloat162float(d_output[t * B * H_n + b * H_n + h * N_STATE + tid]);
                float Sq = __bfloat162float(Sq_cache[t * B * H_n + b * H_n + h * N_STATE + tid]);
                float sig = 1.0f / (1.0f + expf(-Sq));
                float d_Sq = d_out * (2.0f * Sq * sig + Sq * Sq * sig * (1.0f - sig));
                d_Sq_shared[tid] = d_Sq;
            }
            __syncthreads();

            // dS += outer(d_Sq, k_norm)
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                dS[i] += d_Sq_shared[row] * k_norm[col];
            }
            __syncthreads();

            // Initialize dA
            for (int i = tid; i < n2; i += blockDim.x) {
                dA_shared[i] = 0.0f;
            }
            __syncthreads();

            // Compute d_delta and accumulate d_beta
            __shared__ float d_beta_sum;
            if (tid == 0) d_beta_sum = 0.0f;
            __syncthreads();

            if (tid < N_STATE) {
                float d_delta_local = 0.0f;
                float d_beta_local_row = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    float S_t_ij = tanhf(beta_val_t * S[tid * N_STATE + j] + delta[tid] * k_norm[j]);
                    float d_pre = dS[tid * N_STATE + j] * (1.0f - S_t_ij * S_t_ij);
                    d_delta_local += d_pre * k_norm[j];
                    d_beta_local_row += d_pre * S[tid * N_STATE + j];
                }
                d_delta[tid] = d_delta_local;
                atomicAdd(&d_beta_sum, d_beta_local_row);
            }
            __syncthreads();

            // Compute d_k_norm
            if (tid < N_STATE) {
                float d_k_norm_local = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    float S_t_ij = tanhf(beta_val_t * S[i * N_STATE + tid] + delta[i] * k_norm[tid]);
                    float d_pre = dS[i * N_STATE + tid] * (1.0f - S_t_ij * S_t_ij);
                    d_k_norm_local += d_pre * delta[i];
                }
                for (int i = 0; i < N_STATE; i++) {
                    float S_t_ij = tanhf(beta_val_t * S[i * N_STATE + tid] + delta[i] * k_norm[tid]);
                    d_k_norm_local += S_t_ij * d_Sq_shared[i];
                }
                for (int i = 0; i < N_STATE; i++) {
                    d_k_norm_local += S[i * N_STATE + tid] * (-d_delta[i]);
                }
                d_k_norm[tid] = d_k_norm_local;
            }
            __syncthreads();

            if (tid < N_STATE) {
                d_v_raw[tid] = d_delta[tid];
            }
            __syncthreads();

            // Propagate gradients to dA
            __shared__ float k_dot_dk;
            if (tid == 0) {
                k_dot_dk = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    k_dot_dk += k_raw[i] * d_k_norm[i];
                }
            }
            __syncthreads();

            if (tid < N_STATE) {
                float norm3 = k_norm_val_t * k_norm_val_t * k_norm_val_t;
                float d_k_raw_i = d_k_norm[tid] / k_norm_val_t - k_raw[tid] * k_dot_dk / norm3;

                for (int j = 0; j < N_STATE; j++) {
                    atomicAdd(&dA_shared[tid * N_STATE + j], d_k_raw_i / N_STATE);
                }
                for (int j = 0; j < N_STATE; j++) {
                    atomicAdd(&dA_shared[j * N_STATE + tid], d_v_raw[tid] / N_STATE);
                }
            }
            __syncthreads();

            // d_beta propagates to d_scale and d_bias
            if (tid == 0) {
                float d_pre_sigmoid = d_beta_sum * beta_val_t * (1.0f - beta_val_t);
                float A_sum_local = 0.0f;
                for (int i = 0; i < n2; i++) {
                    A_sum_local += A_shared[i];
                }
                float A_mean_local = A_sum_local / n2;

                d_scale_local += d_pre_sigmoid * A_mean_local;
                d_bias_local += d_pre_sigmoid;

                float d_A_mean = d_pre_sigmoid * scale;
                for (int i = 0; i < n2; i++) {
                    dA_shared[i] += d_A_mean / n2;
                }
            }
            __syncthreads();

            // Write dA to dx_all
            for (int i = tid; i < n2; i += blockDim.x) {
                dx_all[t * B * H_n2 + b * H_n2 + h * n2 + i] = __float2bfloat16(dA_shared[i]);
            }
            __syncthreads();

            // Update dS for previous timestep
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                float S_t_ij = tanhf(beta_val_t * S[i] + delta[row] * k_norm[col]);
                float d_pre = dS[i] * (1.0f - S_t_ij * S_t_ij);
                dS[i] = d_pre * beta_val_t + (-d_delta[row]) * k_norm[col];
            }
            __syncthreads();
        }
    }

    // Atomically accumulate d_scale and d_bias
    if (tid == 0) {
        atomicAdd(&d_scale_accum[0], d_scale_local);
        atomicAdd(&d_bias_accum[0], d_bias_local);
    }
}

// ============================================================================
// E86InputMatrixDeltaForward Implementation (Multi-Head)
// ============================================================================

template<typename DataT>
E86InputMatrixDeltaForward<DataT>::E86InputMatrixDeltaForward(
    bool training,
    int batch_size,
    int n_state,
    const cudaStream_t& stream)
    : training_(training),
      batch_size_(batch_size),
      n_state_(n_state),
      stream_(stream) {}

template<typename DataT>
void E86InputMatrixDeltaForward<DataT>::Run(
    int steps,
    int n_heads,
    float scale,
    float bias,
    const DataT* x,
    DataT* S,
    DataT* output,
    DataT* k_cache,
    DataT* v_cache,
    DataT* beta_cache,
    DataT* S_cache
) {
    int T = steps;
    int B = batch_size_;
    int n = n_state_;
    int H = n_heads;
    int n2 = n * n;
    int H_n2 = H * n2;

    int num_checkpoints = (T + CHECKPOINT_INTERVAL - 1) / CHECKPOINT_INTERVAL + 1;
    DataT* s_checkpoints = S_cache;
    DataT* sq_cache = S_cache + num_checkpoints * B * H_n2;

    // Shared memory: 2*n² + 5*n floats (S, A, k, v, retrieved)
    int shared_size = (2 * n2 + 5 * n) * sizeof(float);
    int kernel_threads = min(256, n2);

    // Launch B*H blocks (one per batch-head pair)
    int num_blocks = B * H;

    #define DISPATCH_E86_FORWARD(N) \
        E86InputMatrixDeltaForwardKernel_BF16<N><<<num_blocks, kernel_threads, shared_size, stream_>>>( \
            T, B, H, \
            (const __nv_bfloat16*)x, \
            scale, bias, \
            (__nv_bfloat16*)S, \
            (__nv_bfloat16*)output, \
            (__nv_bfloat16*)k_cache, \
            (__nv_bfloat16*)v_cache, \
            (__nv_bfloat16*)beta_cache, \
            (__nv_bfloat16*)s_checkpoints, \
            (__nv_bfloat16*)sq_cache, \
            CHECKPOINT_INTERVAL)

    if (n == 16) { DISPATCH_E86_FORWARD(16); }
    else if (n == 24) { DISPATCH_E86_FORWARD(24); }
    else if (n == 32) { DISPATCH_E86_FORWARD(32); }
    else if (n == 48) { DISPATCH_E86_FORWARD(48); }
    else if (n == 64) { DISPATCH_E86_FORWARD(64); }
    else {
        fprintf(stderr, "E86 Forward: unsupported n_state=%d\n", n);
    }

    #undef DISPATCH_E86_FORWARD
}

template struct E86InputMatrixDeltaForward<__nv_bfloat16>;

// ============================================================================
// E86InputMatrixDeltaBackward Implementation (Multi-Head)
// ============================================================================

template<typename DataT>
E86InputMatrixDeltaBackward<DataT>::E86InputMatrixDeltaBackward(
    int batch_size,
    int n_state,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      n_state_(n_state),
      stream_(stream) {}

template<typename DataT>
void E86InputMatrixDeltaBackward<DataT>::Run(
    int steps,
    int n_heads,
    float scale,
    float bias,
    const DataT* x,
    const DataT* S_checkpoints,
    const DataT* Sq_cache,
    const DataT* k_cache,
    const DataT* v_cache,
    const DataT* beta_cache,
    const DataT* d_output,
    DataT* dx,
    float* d_scale,
    float* d_bias,
    DataT* workspace
) {
    int T = steps;
    int B = batch_size_;
    int n = n_state_;
    int H = n_heads;
    int n2 = n * n;

    // Zero out gradient accumulators
    cudaMemsetAsync(d_scale, 0, sizeof(float), stream_);
    cudaMemsetAsync(d_bias, 0, sizeof(float), stream_);

    // Shared memory: 4*n² + 10*n floats
    int shared_size = (4 * n2 + 10 * n) * sizeof(float);
    int threads = min(256, n2);

    int num_blocks = B * H;

    #define DISPATCH_E86_BACKWARD(N) \
        E86InputMatrixDeltaBackwardKernel_BF16<N><<<num_blocks, threads, shared_size, stream_>>>( \
            T, B, H, \
            (const __nv_bfloat16*)x, \
            (const __nv_bfloat16*)k_cache, \
            (const __nv_bfloat16*)v_cache, \
            (const __nv_bfloat16*)beta_cache, \
            (const __nv_bfloat16*)S_checkpoints, \
            (const __nv_bfloat16*)Sq_cache, \
            (const __nv_bfloat16*)d_output, \
            scale, bias, \
            (__nv_bfloat16*)dx, \
            d_scale, \
            d_bias, \
            CHECKPOINT_INTERVAL)

    if (n == 16) { DISPATCH_E86_BACKWARD(16); }
    else if (n == 24) { DISPATCH_E86_BACKWARD(24); }
    else if (n == 32) { DISPATCH_E86_BACKWARD(32); }
    else if (n == 48) { DISPATCH_E86_BACKWARD(48); }
    else if (n == 64) { DISPATCH_E86_BACKWARD(64); }
    else {
        fprintf(stderr, "E86 Backward: unsupported n_state=%d\n", n);
    }

    #undef DISPATCH_E86_BACKWARD
}

template struct E86InputMatrixDeltaBackward<__nv_bfloat16>;

}  // namespace elman
