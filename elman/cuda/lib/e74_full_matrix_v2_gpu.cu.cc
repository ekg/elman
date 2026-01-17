/**
 * E74 Full Matrix State V2 CUDA Kernel
 *
 * Extended version supporting multiple update types and gate types:
 * - update_type: 0=delta, 1=residual, 2=ntm, 3=retrieved_gate, 4=ema
 * - gate_type: 0=output (self-gate), 1=input (E1-style)
 *
 * Focuses on n_state=32 with no_z projection (optimal config from benchmarks).
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
// Element-wise utility kernels
// ============================================================================

// Add bias and apply sigmoid: out[i] = sigmoid(out[i] + bias[i % n])
__global__ void AddBiasSigmoidKernel_BF16(
    __nv_bfloat16* __restrict__ data,      // [total_elements] (T*B*n)
    const __nv_bfloat16* __restrict__ bias, // [n]
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

// Add bias only: out[i] = out[i] + bias[i % n]
__global__ void AddBiasKernel_BF16(
    __nv_bfloat16* __restrict__ data,      // [total_elements]
    const __nv_bfloat16* __restrict__ bias, // [n]
    int n,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int bias_idx = idx % n;
    float val = __bfloat162float(data[idx]) + __bfloat162float(bias[bias_idx]);
    data[idx] = __float2bfloat16(val);
}

// Reduce gradients for bias: db[i] = sum over (T*B) of d_data[j*n + i]
__global__ void ReduceBiasGradKernel_BF16(
    const __nv_bfloat16* __restrict__ d_data, // [T*B*n]
    __nv_bfloat16* __restrict__ db,            // [n]
    int n,
    int T_B  // T * B
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float sum = 0.0f;
    for (int tb = 0; tb < T_B; tb++) {
        sum += __bfloat162float(d_data[tb * n + i]);
    }
    db[i] = __float2bfloat16(sum);
}

// Reduce d_residual_scale across batches: d_scale[i] = sum over B of d_scale_accum[b*n + i]
__global__ void ReduceResidualScaleGradKernel(
    const float* __restrict__ d_scale_accum, // [B, n]
    float* __restrict__ d_scale,              // [n]
    int n,
    int B
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float sum = 0.0f;
    for (int b = 0; b < B; b++) {
        sum += d_scale_accum[b * n + i];
    }
    d_scale[i] = sum;
}

// Apply sigmoid derivative: d_out[i] = d_out[i] * sigmoid_val[i] * (1 - sigmoid_val[i])
// This is used when we applied sigmoid in forward, so backward needs the derivative
__global__ void ApplySigmoidDerivKernel_BF16(
    __nv_bfloat16* __restrict__ d_data,           // [total] in/out
    const __nv_bfloat16* __restrict__ sigmoid_val, // [total] post-sigmoid values from forward
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float d = __bfloat162float(d_data[idx]);
    float s = __bfloat162float(sigmoid_val[idx]);
    float d_pre = d * s * (1.0f - s);  // sigmoid derivative
    d_data[idx] = __float2bfloat16(d_pre);
}

// Apply sigmoid only (no bias): out[i] = sigmoid(out[i])
__global__ void ApplySigmoidKernel_BF16(
    __nv_bfloat16* __restrict__ data,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float val = __bfloat162float(data[idx]);
    float sig = 1.0f / (1.0f + expf(-val));
    data[idx] = __float2bfloat16(sig);
}

// ============================================================================
// Forward Kernel with UPDATE_TYPE and GATE_TYPE template parameters
// UPDATE_TYPE: 0=delta, 1=residual, 2=ntm, 3=retrieved_gate, 4=ema
// GATE_TYPE: 0=output (self-gate), 1=input (E1-style)
// ============================================================================

template<int N_STATE, int PROJ_TYPE, int UPDATE_TYPE, int GATE_TYPE>
__global__ void E74FullMatrixForwardV2Kernel_BF16(
    int T,
    int B,
    const __nv_bfloat16* __restrict__ k_all,      // [T, B, N_STATE]
    const __nv_bfloat16* __restrict__ v_all,      // [T, B, N_STATE]
    const __nv_bfloat16* __restrict__ q_all,      // [T, B, N_STATE]
    __nv_bfloat16* __restrict__ S,                // [B, N_STATE, N_STATE]
    __nv_bfloat16* __restrict__ output,           // [T, B, N_STATE]
    __nv_bfloat16* __restrict__ S_checkpoints,    // [num_checkpoints, B, N_STATE, N_STATE]
    __nv_bfloat16* __restrict__ Sq_cache,         // [T, B, N_STATE]
    // Extra inputs for update types:
    const float* __restrict__ residual_scale,     // [N_STATE] for residual
    const __nv_bfloat16* __restrict__ erase_all,  // [T, B, N_STATE] for NTM
    const __nv_bfloat16* __restrict__ write_all,  // [T, B, N_STATE] for NTM
    const __nv_bfloat16* __restrict__ gate_all,   // [T, B, N_STATE] for retrieved_gate
    const __nv_bfloat16* __restrict__ alpha_all,  // [T, B, N_STATE] for EMA
    // Extra inputs for gate type:
    const __nv_bfloat16* __restrict__ z_gate_all, // [T, B, N_STATE] for input gate
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
    float* extra_shared = retrieved + N_STATE;       // [N_STATE] for erase/write/gate/alpha

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;

    // Load initial state into shared memory
    for (int i = tid; i < n2; i += blockDim.x) {
        S_shared[i] = __bfloat162float(S[b * n2 + i]);
    }
    __syncthreads();

    // Save initial state as checkpoint 0
    for (int i = tid; i < n2; i += blockDim.x) {
        S_checkpoints[0 * B * n2 + b * n2 + i] = __float2bfloat16(S_shared[i]);
    }

    // Load residual_scale into registers if needed
    float res_scale_local = 0.0f;
    if constexpr (UPDATE_TYPE == 1) {  // RESIDUAL
        if (tid < N_STATE) {
            res_scale_local = residual_scale[tid];
        }
    }

    // Process each timestep
    for (int t = 0; t < T; t++) {
        // Load k, v, q based on projection type
        if (tid < N_STATE) {
            k_shared[tid] = __bfloat162float(k_all[t * B * N_STATE + b * N_STATE + tid]);

            if constexpr (PROJ_TYPE == 0) {
                // tied_kvq: k = v = q
                v_shared[tid] = k_shared[tid];
                q_shared[tid] = k_shared[tid];
            } else if constexpr (PROJ_TYPE == 1) {
                // tied_kq: k = q, v separate
                v_shared[tid] = __bfloat162float(v_all[t * B * N_STATE + b * N_STATE + tid]);
                q_shared[tid] = k_shared[tid];
            } else {
                // no_z: k, v, q all separate
                v_shared[tid] = __bfloat162float(v_all[t * B * N_STATE + b * N_STATE + tid]);
                q_shared[tid] = __bfloat162float(q_all[t * B * N_STATE + b * N_STATE + tid]);
            }

            // Load extra inputs for update types
            if constexpr (UPDATE_TYPE == 2) {  // NTM
                extra_shared[tid] = __bfloat162float(erase_all[t * B * N_STATE + b * N_STATE + tid]);
            } else if constexpr (UPDATE_TYPE == 3) {  // RETRIEVED_GATE
                extra_shared[tid] = __bfloat162float(gate_all[t * B * N_STATE + b * N_STATE + tid]);
            } else if constexpr (UPDATE_TYPE == 4) {  // EMA
                extra_shared[tid] = __bfloat162float(alpha_all[t * B * N_STATE + b * N_STATE + tid]);
            }
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

        // Compute retrieved = S @ k (for delta, residual, retrieved_gate)
        if constexpr (UPDATE_TYPE == 0 || UPDATE_TYPE == 1 || UPDATE_TYPE == 3) {
            if (tid < N_STATE) {
                float sum = 0.0f;
                #pragma unroll 8
                for (int j = 0; j < N_STATE; j++) {
                    sum += S_shared[tid * N_STATE + j] * k_shared[j];
                }
                retrieved[tid] = sum;
            }
            __syncthreads();
        }

        // Update state based on UPDATE_TYPE
        for (int i = tid; i < n2; i += blockDim.x) {
            int row = i / N_STATE;
            int col = i % N_STATE;

            if constexpr (UPDATE_TYPE == 0) {
                // DELTA: S = tanh(S + outer(v - retrieved, k))
                float delta_i = v_shared[row] - retrieved[row];
                float update = S_shared[i] + delta_i * k_shared[col];
                S_shared[i] = tanhf(update);

            } else if constexpr (UPDATE_TYPE == 1) {
                // RESIDUAL: S = S + scale * tanh(outer(delta, k))
                float delta_i = v_shared[row] - retrieved[row];
                float outer_val = delta_i * k_shared[col];
                float update = tanhf(outer_val);
                // Use per-row scale (loaded earlier)
                float scale = (tid < N_STATE) ? res_scale_local : residual_scale[row];
                S_shared[i] = S_shared[i] + scale * update;

            } else if constexpr (UPDATE_TYPE == 2) {
                // NTM: S = S * (1 - outer(erase, k)) + outer(write * v, k)
                float erase_val = extra_shared[row];  // erase[row]
                float write_val;
                // Load write separately (need second shared buffer or reload)
                write_val = __bfloat162float(write_all[t * B * N_STATE + b * N_STATE + row]);

                float erase_outer = erase_val * k_shared[col];
                float write_outer = (write_val * v_shared[row]) * k_shared[col];
                S_shared[i] = S_shared[i] * (1.0f - erase_outer) + write_outer;
                S_shared[i] = tanhf(S_shared[i]);

            } else if constexpr (UPDATE_TYPE == 3) {
                // RETRIEVED_GATE: S = tanh(S + outer(delta * gate, k))
                float delta_i = v_shared[row] - retrieved[row];
                float gate_val = extra_shared[row];  // gate[row]
                float gated_delta = delta_i * gate_val;
                float update = S_shared[i] + gated_delta * k_shared[col];
                S_shared[i] = tanhf(update);

            } else if constexpr (UPDATE_TYPE == 4) {
                // EMA: S = alpha * S + (1 - alpha) * outer(v, k)
                float alpha_val = extra_shared[row];  // alpha[row], already sigmoid applied
                float outer_val = v_shared[row] * k_shared[col];
                S_shared[i] = alpha_val * S_shared[i] + (1.0f - alpha_val) * outer_val;
                S_shared[i] = tanhf(S_shared[i]);
            }
        }
        __syncthreads();

        // Save checkpoint if at checkpoint boundary
        if ((t + 1) % checkpoint_interval == 0) {
            int cp_idx = (t + 1) / checkpoint_interval;
            for (int i = tid; i < n2; i += blockDim.x) {
                S_checkpoints[cp_idx * B * n2 + b * n2 + i] = __float2bfloat16(S_shared[i]);
            }
        }

        // Compute output: Sq = S @ q
        if (tid < N_STATE) {
            float Sq = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < N_STATE; j++) {
                Sq += S_shared[tid * N_STATE + j] * q_shared[j];
            }
            // Cache Sq for backward
            Sq_cache[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(Sq);

            // Output gating based on GATE_TYPE
            float out_val;
            if constexpr (GATE_TYPE == 0) {
                // Self-gating: Sq * silu(Sq) = Sq * Sq * sigmoid(Sq)
                float sig = 1.0f / (1.0f + expf(-Sq));
                out_val = Sq * Sq * sig;
            } else {
                // Input gating: Sq * silu(z_gate)
                float z = __bfloat162float(z_gate_all[t * B * N_STATE + b * N_STATE + tid]);
                float sig = 1.0f / (1.0f + expf(-z));
                out_val = Sq * z * sig;
            }
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
// E74FullMatrixForwardV2 Implementation
// ============================================================================

template<typename DataT>
E74FullMatrixForwardV2<DataT>::E74FullMatrixForwardV2(
    bool training,
    int batch_size,
    int n_state,
    int dim,
    int proj_type,
    bool use_tanh,
    int update_type,
    int gate_type,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : training_(training),
      batch_size_(batch_size),
      n_state_(n_state),
      dim_(dim),
      proj_type_(proj_type),
      use_tanh_(use_tanh),
      update_type_(update_type),
      gate_type_(gate_type),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename DataT>
void E74FullMatrixForwardV2<DataT>::Run(
    int steps,
    const DataT* W_kvq,
    const DataT* W_k,
    const DataT* W_v,
    const DataT* W_q,
    const DataT* x,
    DataT* S,
    DataT* output,
    DataT* k_cache,
    DataT* v_cache,
    DataT* q_cache,
    DataT* S_cache,
    const DataT* residual_scale,
    const DataT* W_erase,
    const DataT* b_erase,
    const DataT* W_write,
    const DataT* b_write,
    const DataT* W_gate,
    const DataT* b_gate,
    const DataT* W_alpha,
    const DataT* b_alpha,
    const DataT* W_z_gate,
    const DataT* b_z_gate,
    DataT* workspace
) {
    int T_steps = steps;
    int B = batch_size_;
    int n = n_state_;
    int d = dim_;

    // Step 1: Batch projections with cuBLAS
    const float alpha = 1.0f, beta = 0.0f;

    // k projection
    if (proj_type_ == 0) {
        // tied_kvq: k = v = q = W_kvq @ x
        cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            n, T_steps * B, d, &alpha,
            W_kvq, CUDA_R_16BF, d, x, CUDA_R_16BF, d,
            &beta, k_cache, CUDA_R_16BF, n,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    } else {
        // Separate projections
        cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            n, T_steps * B, d, &alpha,
            W_k, CUDA_R_16BF, d, x, CUDA_R_16BF, d,
            &beta, k_cache, CUDA_R_16BF, n,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            n, T_steps * B, d, &alpha,
            W_v, CUDA_R_16BF, d, x, CUDA_R_16BF, d,
            &beta, v_cache, CUDA_R_16BF, n,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        if (proj_type_ == 2) {  // no_z: separate q
            cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                n, T_steps * B, d, &alpha,
                W_q, CUDA_R_16BF, d, x, CUDA_R_16BF, d,
                &beta, q_cache, CUDA_R_16BF, n,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        }
    }

    // Extra projections for update types
    DataT* erase_cache = nullptr;
    DataT* write_cache = nullptr;
    DataT* gate_cache = nullptr;
    DataT* alpha_cache = nullptr;
    DataT* z_gate_cache = nullptr;

    // Calculate workspace offsets
    int num_checkpoints = (T_steps + CHECKPOINT_INTERVAL - 1) / CHECKPOINT_INTERVAL + 1;
    DataT* s_checkpoints = S_cache;
    DataT* sq_cache = S_cache + num_checkpoints * B * n * n;
    DataT* extra_workspace = sq_cache + T_steps * B * n;

    if (update_type_ == 2) {  // NTM
        erase_cache = extra_workspace;
        write_cache = extra_workspace + T_steps * B * n;
        extra_workspace += 2 * T_steps * B * n;

        // erase = sigmoid(W_erase @ x + b_erase) - erase needs sigmoid to stay in [0,1]
        cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            n, T_steps * B, d, &alpha,
            W_erase, CUDA_R_16BF, d, x, CUDA_R_16BF, d,
            &beta, erase_cache, CUDA_R_16BF, n,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // Apply bias and sigmoid to erase
        int total_erase = T_steps * B * n;
        int threads_erase = 256;
        int blocks_erase = (total_erase + threads_erase - 1) / threads_erase;
        AddBiasSigmoidKernel_BF16<<<blocks_erase, threads_erase, 0, stream_>>>(
            (__nv_bfloat16*)erase_cache, (const __nv_bfloat16*)b_erase, n, total_erase);

        // write = W_write @ x + b_write (no sigmoid needed)
        cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            n, T_steps * B, d, &alpha,
            W_write, CUDA_R_16BF, d, x, CUDA_R_16BF, d,
            &beta, write_cache, CUDA_R_16BF, n,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // Apply bias to write
        int total_write = T_steps * B * n;
        int threads_write = 256;
        int blocks_write = (total_write + threads_write - 1) / threads_write;
        AddBiasKernel_BF16<<<blocks_write, threads_write, 0, stream_>>>(
            (__nv_bfloat16*)write_cache, (const __nv_bfloat16*)b_write, n, total_write);

    } else if (update_type_ == 3) {  // RETRIEVED_GATE
        gate_cache = extra_workspace;
        extra_workspace += T_steps * B * n;

        cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            n, T_steps * B, d, &alpha,
            W_gate, CUDA_R_16BF, d, x, CUDA_R_16BF, d,
            &beta, gate_cache, CUDA_R_16BF, n,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // Apply bias and sigmoid: gate = sigmoid(W_gate @ x + b_gate)
        int total_gate = T_steps * B * n;
        int threads_gate = 256;
        int blocks_gate = (total_gate + threads_gate - 1) / threads_gate;
        AddBiasSigmoidKernel_BF16<<<blocks_gate, threads_gate, 0, stream_>>>(
            (__nv_bfloat16*)gate_cache, (const __nv_bfloat16*)b_gate, n, total_gate);

    } else if (update_type_ == 4) {  // EMA
        alpha_cache = extra_workspace;
        extra_workspace += T_steps * B * n;

        cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            n, T_steps * B, d, &alpha,
            W_alpha, CUDA_R_16BF, d, x, CUDA_R_16BF, d,
            &beta, alpha_cache, CUDA_R_16BF, n,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // Apply bias and sigmoid: alpha = sigmoid(W_alpha @ x + b_alpha)
        int total_alpha = T_steps * B * n;
        int threads_alpha = 256;
        int blocks_alpha = (total_alpha + threads_alpha - 1) / threads_alpha;
        AddBiasSigmoidKernel_BF16<<<blocks_alpha, threads_alpha, 0, stream_>>>(
            (__nv_bfloat16*)alpha_cache, (const __nv_bfloat16*)b_alpha, n, total_alpha);
    }

    if (gate_type_ == 1) {  // INPUT gate
        z_gate_cache = extra_workspace;

        cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            n, T_steps * B, d, &alpha,
            W_z_gate, CUDA_R_16BF, d, x, CUDA_R_16BF, d,
            &beta, z_gate_cache, CUDA_R_16BF, n,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // Apply bias: z_gate = W_z_gate @ x + b_z_gate
        int total_z = T_steps * B * n;
        int threads_z = 256;
        int blocks_z = (total_z + threads_z - 1) / threads_z;
        AddBiasKernel_BF16<<<blocks_z, threads_z, 0, stream_>>>(
            (__nv_bfloat16*)z_gate_cache, (const __nv_bfloat16*)b_z_gate, n, total_z);
    }

    // Step 2: Run fused forward kernel
    int shared_size = (n * n + 6 * n) * sizeof(float);  // state + k,v,q,retrieved,extra
    int threads = min(256, n * n);

    // Convert residual_scale to float if needed
    float* residual_scale_f32 = nullptr;
    if (update_type_ == 1) {
        // For now, assume residual_scale is passed as float
        residual_scale_f32 = (float*)residual_scale;
    }

    // Dispatch kernel - focusing on n=32 for now
    #define DISPATCH_V2_KERNEL(N_STATE, PROJ_TYPE, UPDATE_TYPE, GATE_TYPE) \
        E74FullMatrixForwardV2Kernel_BF16<N_STATE, PROJ_TYPE, UPDATE_TYPE, GATE_TYPE><<<B, threads, shared_size, stream_>>>( \
            T_steps, B, \
            (const __nv_bfloat16*)k_cache, \
            (const __nv_bfloat16*)v_cache, \
            (const __nv_bfloat16*)q_cache, \
            (__nv_bfloat16*)S, (__nv_bfloat16*)output, \
            (__nv_bfloat16*)s_checkpoints, (__nv_bfloat16*)sq_cache, \
            residual_scale_f32, \
            (const __nv_bfloat16*)erase_cache, \
            (const __nv_bfloat16*)write_cache, \
            (const __nv_bfloat16*)gate_cache, \
            (const __nv_bfloat16*)alpha_cache, \
            (const __nv_bfloat16*)z_gate_cache, \
            CHECKPOINT_INTERVAL)

    // Helper macro to dispatch all update types for a given n_state
    #define DISPATCH_FWD_ALL_UPDATES(N_STATE) \
        if (gate_type_ == 0) { \
            switch (update_type_) { \
                case 0: DISPATCH_V2_KERNEL(N_STATE, 2, 0, 0); break; \
                case 1: DISPATCH_V2_KERNEL(N_STATE, 2, 1, 0); break; \
                case 2: DISPATCH_V2_KERNEL(N_STATE, 2, 2, 0); break; \
                case 3: DISPATCH_V2_KERNEL(N_STATE, 2, 3, 0); break; \
                case 4: DISPATCH_V2_KERNEL(N_STATE, 2, 4, 0); break; \
            } \
        } else { \
            switch (update_type_) { \
                case 0: DISPATCH_V2_KERNEL(N_STATE, 2, 0, 1); break; \
                case 1: DISPATCH_V2_KERNEL(N_STATE, 2, 1, 1); break; \
                case 2: DISPATCH_V2_KERNEL(N_STATE, 2, 2, 1); break; \
                case 3: DISPATCH_V2_KERNEL(N_STATE, 2, 3, 1); break; \
                case 4: DISPATCH_V2_KERNEL(N_STATE, 2, 4, 1); break; \
            } \
        }

    // Dispatch by n_state (proj_type must be 2 = no_z)
    if (proj_type_ != 2) {
        printf("E74v2 Forward: Unsupported proj_type=%d (only no_z=2 supported)\n", proj_type_);
    } else if (n == 1) { DISPATCH_FWD_ALL_UPDATES(1); }
    else if (n == 2) { DISPATCH_FWD_ALL_UPDATES(2); }
    else if (n == 4) { DISPATCH_FWD_ALL_UPDATES(4); }
    else if (n == 8) { DISPATCH_FWD_ALL_UPDATES(8); }
    else if (n == 16) { DISPATCH_FWD_ALL_UPDATES(16); }
    else if (n == 28) { DISPATCH_FWD_ALL_UPDATES(28); }
    else if (n == 32) { DISPATCH_FWD_ALL_UPDATES(32); }
    else if (n == 48) { DISPATCH_FWD_ALL_UPDATES(48); }
    else if (n == 64) { DISPATCH_FWD_ALL_UPDATES(64); }
    else if (n == 96) { DISPATCH_FWD_ALL_UPDATES(96); }
    else {
        printf("E74v2 Forward: Unsupported n_state=%d (output will be zeros!)\n", n);
    }

    #undef DISPATCH_FWD_ALL_UPDATES

    #undef DISPATCH_V2_KERNEL
}

// Explicit template instantiation
template struct E74FullMatrixForwardV2<__nv_bfloat16>;

// ============================================================================
// Backward Kernel with UPDATE_TYPE and GATE_TYPE template parameters
// ============================================================================

template<int N_STATE, int PROJ_TYPE, int UPDATE_TYPE, int GATE_TYPE>
__global__ void E74FullMatrixBackwardV2Kernel_BF16(
    int T,
    int B,
    const __nv_bfloat16* __restrict__ k_all,        // [T, B, N_STATE]
    const __nv_bfloat16* __restrict__ v_all,        // [T, B, N_STATE]
    const __nv_bfloat16* __restrict__ q_all,        // [T, B, N_STATE]
    const __nv_bfloat16* __restrict__ S_checkpoints,// [num_checkpoints, B, N_STATE, N_STATE]
    const __nv_bfloat16* __restrict__ Sq_cache,     // [T, B, N_STATE]
    const __nv_bfloat16* __restrict__ d_output,     // [T, B, N_STATE]
    __nv_bfloat16* __restrict__ d_k_all,            // [T, B, N_STATE]
    __nv_bfloat16* __restrict__ d_v_all,            // [T, B, N_STATE]
    __nv_bfloat16* __restrict__ d_q_all,            // [T, B, N_STATE]
    // Extra inputs for update types:
    const float* __restrict__ residual_scale,       // [N_STATE] for RESIDUAL
    const __nv_bfloat16* __restrict__ erase_all,    // [T, B, N_STATE] for NTM
    const __nv_bfloat16* __restrict__ write_all,    // [T, B, N_STATE] for NTM
    const __nv_bfloat16* __restrict__ gate_all,     // [T, B, N_STATE] for RETRIEVED_GATE
    const __nv_bfloat16* __restrict__ alpha_all,    // [T, B, N_STATE] for EMA
    // Extra for gate type:
    const __nv_bfloat16* __restrict__ z_gate_all,   // [T, B, N_STATE] for INPUT gate
    // Extra gradient outputs:
    __nv_bfloat16* __restrict__ d_erase_all,        // [T, B, N_STATE] for NTM
    __nv_bfloat16* __restrict__ d_write_all,        // [T, B, N_STATE] for NTM
    __nv_bfloat16* __restrict__ d_gate_all,         // [T, B, N_STATE] for RETRIEVED_GATE
    __nv_bfloat16* __restrict__ d_alpha_all,        // [T, B, N_STATE] for EMA
    __nv_bfloat16* __restrict__ d_z_gate_all,       // [T, B, N_STATE] for INPUT gate
    float* __restrict__ d_residual_scale_accum,     // [B, N_STATE] for RESIDUAL (per-batch accumulator)
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
    float* d_k_raw = retrieved + N_STATE;             // [N_STATE]
    float* d_v_raw = d_k_raw + N_STATE;               // [N_STATE]
    float* d_q_raw = d_v_raw + N_STATE;               // [N_STATE]
    float* d_Sq_shared = d_q_raw + N_STATE;           // [N_STATE]
    float* d_delta = d_Sq_shared + N_STATE;           // [N_STATE]
    float* d_k_norm = d_delta + N_STATE;              // [N_STATE]
    float* extra_shared = d_k_norm + N_STATE;         // [N_STATE] for erase/write/gate/alpha

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;

    // Initialize dS to zero
    for (int i = tid; i < n2; i += blockDim.x) {
        dS[i] = 0.0f;
    }
    __syncthreads();

    // Per-batch accumulator for residual_scale gradient
    float d_res_scale_local[N_STATE];
    if constexpr (UPDATE_TYPE == 1) {
        for (int i = 0; i < N_STATE; i++) {
            d_res_scale_local[i] = 0.0f;
        }
    }

    int num_segments = (T + checkpoint_interval - 1) / checkpoint_interval;

    for (int seg = num_segments - 1; seg >= 0; seg--) {
        int t_start = seg * checkpoint_interval;
        int t_end = min(t_start + checkpoint_interval, T);

        // ====== BACKWARD through segment ======
        for (int t = t_end - 1; t >= t_start; t--) {
            // Reload checkpoint into S (will be updated in-place during recomputation)
            for (int i = tid; i < n2; i += blockDim.x) {
                S[i] = __bfloat162float(S_checkpoints[seg * B * n2 + b * n2 + i]);
            }
            __syncthreads();

            // Recompute forward to step t (based on UPDATE_TYPE)
            for (int tt = t_start; tt <= t; tt++) {
                if (tid < N_STATE) {
                    k_raw[tid] = __bfloat162float(k_all[tt * B * N_STATE + b * N_STATE + tid]);
                    v_raw[tid] = __bfloat162float(v_all[tt * B * N_STATE + b * N_STATE + tid]);
                    q_raw[tid] = __bfloat162float(q_all[tt * B * N_STATE + b * N_STATE + tid]);

                    // Load extra inputs for update types
                    if constexpr (UPDATE_TYPE == 2) {  // NTM
                        extra_shared[tid] = __bfloat162float(erase_all[tt * B * N_STATE + b * N_STATE + tid]);
                    } else if constexpr (UPDATE_TYPE == 3) {  // RETRIEVED_GATE
                        extra_shared[tid] = __bfloat162float(gate_all[tt * B * N_STATE + b * N_STATE + tid]);
                    } else if constexpr (UPDATE_TYPE == 4) {  // EMA
                        extra_shared[tid] = __bfloat162float(alpha_all[tt * B * N_STATE + b * N_STATE + tid]);
                    }
                }
                __syncthreads();

                // Normalize k
                __shared__ float k_norm_val_fwd;
                if (tid == 0) {
                    float sum_sq = 0.0f;
                    for (int i = 0; i < N_STATE; i++) sum_sq += k_raw[i] * k_raw[i];
                    k_norm_val_fwd = sqrtf(sum_sq) + 1e-6f;
                }
                __syncthreads();
                if (tid < N_STATE) {
                    k_norm[tid] = k_raw[tid] / k_norm_val_fwd;
                }
                __syncthreads();

                // Compute retrieved = S @ k_norm (for DELTA, RESIDUAL, RETRIEVED_GATE)
                // Must be computed BEFORE in-place state update
                if constexpr (UPDATE_TYPE == 0 || UPDATE_TYPE == 1 || UPDATE_TYPE == 3) {
                    if (tid < N_STATE) {
                        float sum = 0.0f;
                        for (int j = 0; j < N_STATE; j++) {
                            sum += S[tid * N_STATE + j] * k_norm[j];
                        }
                        retrieved[tid] = sum;
                    }
                    __syncthreads();

                    // delta = v - retrieved
                    if (tid < N_STATE) {
                        delta[tid] = v_raw[tid] - retrieved[tid];
                    }
                    __syncthreads();
                }

                // Update state IN-PLACE only for tt < t (keep S = S_{t-1} for backward)
                // When tt == t, we keep k, v, q, delta, k_norm for backward but don't update S
                if (tt < t) {
                    for (int i = tid; i < n2; i += blockDim.x) {
                        int row = i / N_STATE;
                        int col = i % N_STATE;
                        float S_old = S[i];

                        if constexpr (UPDATE_TYPE == 0) {
                            S[i] = tanhf(S_old + delta[row] * k_norm[col]);
                        } else if constexpr (UPDATE_TYPE == 1) {
                            float outer_val = delta[row] * k_norm[col];
                            S[i] = S_old + residual_scale[row] * tanhf(outer_val);
                        } else if constexpr (UPDATE_TYPE == 2) {
                            float erase_val = extra_shared[row];
                            float write_val = __bfloat162float(write_all[tt * B * N_STATE + b * N_STATE + row]);
                            S[i] = tanhf(S_old * (1.0f - erase_val * k_norm[col]) + (write_val * v_raw[row]) * k_norm[col]);
                        } else if constexpr (UPDATE_TYPE == 3) {
                            float gate_val = extra_shared[row];
                            S[i] = tanhf(S_old + delta[row] * gate_val * k_norm[col]);
                        } else if constexpr (UPDATE_TYPE == 4) {
                            float alpha_val = extra_shared[row];
                            float outer_val = v_raw[row] * k_norm[col];
                            S[i] = tanhf(alpha_val * S_old + (1.0f - alpha_val) * outer_val);
                        }
                    }
                    __syncthreads();
                }
                // After loop: S = S_{t-1}, k/v/q/delta/k_norm are for timestep t
            }

            // ====== BACKWARD at timestep t ======
            __shared__ float k_norm_val_t;
            if (tid == 0) {
                float sum_sq = 0.0f;
                for (int i = 0; i < N_STATE; i++) sum_sq += k_raw[i] * k_raw[i];
                k_norm_val_t = sqrtf(sum_sq) + 1e-6f;
            }
            __syncthreads();

            // Initialize local gradients
            if (tid < N_STATE) {
                d_k_raw[tid] = 0.0f;
                d_v_raw[tid] = 0.0f;
                d_q_raw[tid] = 0.0f;
            }
            __syncthreads();

            // Step 1: d_Sq from output gating (depends on GATE_TYPE)
            if (tid < N_STATE) {
                float Sq = __bfloat162float(Sq_cache[t * B * N_STATE + b * N_STATE + tid]);
                float d_out = __bfloat162float(d_output[t * B * N_STATE + b * N_STATE + tid]);

                float d_Sq;
                if constexpr (GATE_TYPE == 0) {
                    // Self-gating: out = Sq * Sq * sigmoid(Sq)
                    float sig = 1.0f / (1.0f + expf(-Sq));
                    d_Sq = d_out * (2.0f * Sq * sig + Sq * Sq * sig * (1.0f - sig));
                } else {
                    // Input gating: out = Sq * z * sigmoid(z)
                    float z = __bfloat162float(z_gate_all[t * B * N_STATE + b * N_STATE + tid]);
                    float sig = 1.0f / (1.0f + expf(-z));
                    float silu_z = z * sig;
                    d_Sq = d_out * silu_z;

                    // Also compute d_z_gate
                    float d_silu_z = d_out * Sq;
                    float d_z = d_silu_z * (sig + z * sig * (1.0f - sig));
                    d_z_gate_all[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(d_z);
                }
                d_Sq_shared[tid] = d_Sq;
            }
            __syncthreads();

            // Step 2: dS += outer(d_Sq, q) from Sq = S @ q
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                dS[i] += d_Sq_shared[row] * q_raw[col];
            }
            __syncthreads();

            // Step 3: d_q = S_t^T @ d_Sq (need to compute S_t on-the-fly)
            // Note: This is complex because different UPDATE_TYPEs have different S_t formulas
            // For DELTA: S_t = tanh(S + outer(delta, k_norm))
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    float S_t_ij;
                    if constexpr (UPDATE_TYPE == 0) {
                        S_t_ij = tanhf(S[i * N_STATE + tid] + delta[i] * k_norm[tid]);
                    } else if constexpr (UPDATE_TYPE == 1) {
                        float outer_val = delta[i] * k_norm[tid];
                        S_t_ij = S[i * N_STATE + tid] + residual_scale[i] * tanhf(outer_val);
                    } else if constexpr (UPDATE_TYPE == 2) {
                        float erase_val = extra_shared[i];
                        float write_val = __bfloat162float(write_all[t * B * N_STATE + b * N_STATE + i]);
                        float erase_outer = erase_val * k_norm[tid];
                        float write_outer = (write_val * v_raw[i]) * k_norm[tid];
                        S_t_ij = tanhf(S[i * N_STATE + tid] * (1.0f - erase_outer) + write_outer);
                    } else if constexpr (UPDATE_TYPE == 3) {
                        float gate_val = extra_shared[i];
                        S_t_ij = tanhf(S[i * N_STATE + tid] + delta[i] * gate_val * k_norm[tid]);
                    } else if constexpr (UPDATE_TYPE == 4) {
                        float alpha_val = extra_shared[i];
                        float outer_val = v_raw[i] * k_norm[tid];
                        S_t_ij = tanhf(alpha_val * S[i * N_STATE + tid] + (1.0f - alpha_val) * outer_val);
                    }
                    sum += S_t_ij * d_Sq_shared[i];
                }
                d_q_raw[tid] = sum;
            }
            __syncthreads();

            // Backward through state update (depends on UPDATE_TYPE)
            // Note: S = S_{t-1}, compute S_t on-the-fly for tanh derivative
            if constexpr (UPDATE_TYPE == 0) {
                // DELTA: S_t = tanh(S_{t-1} + outer(delta, k_norm))
                // d_pre_nonlin = dS * (1 - S_t²)
                // d_delta[i] = sum_j(d_pre_nonlin[i,j] * k_norm[j])
                // d_k_norm[j] = sum_i(d_pre_nonlin[i,j] * delta[i])

                if (tid < N_STATE) {
                    float d_delta_local = 0.0f;
                    for (int j = 0; j < N_STATE; j++) {
                        // Compute S_t on-the-fly: S_t = tanh(S + outer(delta, k_norm))
                        float S_t_ij = tanhf(S[tid * N_STATE + j] + delta[tid] * k_norm[j]);
                        float d_pre = dS[tid * N_STATE + j] * (1.0f - S_t_ij * S_t_ij);
                        d_delta_local += d_pre * k_norm[j];
                    }
                    d_delta[tid] = d_delta_local;
                }
                __syncthreads();

                if (tid < N_STATE) {
                    float d_k_norm_local = 0.0f;
                    for (int i = 0; i < N_STATE; i++) {
                        // Compute S_t on-the-fly
                        float S_t_ij = tanhf(S[i * N_STATE + tid] + delta[i] * k_norm[tid]);
                        float d_pre = dS[i * N_STATE + tid] * (1.0f - S_t_ij * S_t_ij);
                        d_k_norm_local += d_pre * delta[i];
                    }
                    d_k_norm[tid] = d_k_norm_local;
                }
                __syncthreads();

                // d_v = d_delta (from delta = v - retrieved)
                if (tid < N_STATE) {
                    d_v_raw[tid] = d_delta[tid];
                }

                // d_k_norm += S_{t-1}^T @ (-d_delta) (from retrieved = S_{t-1} @ k_norm)
                // S is S_{t-1}
                if (tid < N_STATE) {
                    float sum = 0.0f;
                    for (int i = 0; i < N_STATE; i++) {
                        sum += S[i * N_STATE + tid] * (-d_delta[i]);
                    }
                    d_k_norm[tid] += sum;
                }
                __syncthreads();

            } else if constexpr (UPDATE_TYPE == 1) {
                // RESIDUAL: S_curr = S_prev + scale * tanh(outer(delta, k_norm))
                // Let outer_val = delta[row] * k_norm[col]
                // Let tanh_out = tanh(outer_val)
                // d_tanh_out = dS[i] * scale[row]
                // d_outer_val = d_tanh_out * (1 - tanh_out²)
                // d_delta[i] = sum_j(d_outer_val[i,j] * k_norm[j])
                // d_scale[i] = sum_j(dS[i,j] * tanh_out[i,j])

                if (tid < N_STATE) {
                    float d_delta_local = 0.0f;
                    float d_scale_local = 0.0f;
                    for (int j = 0; j < N_STATE; j++) {
                        float outer_val = delta[tid] * k_norm[j];
                        float tanh_out = tanhf(outer_val);
                        float d_tanh_out = dS[tid * N_STATE + j] * residual_scale[tid];
                        float d_outer_val = d_tanh_out * (1.0f - tanh_out * tanh_out);
                        d_delta_local += d_outer_val * k_norm[j];
                        d_scale_local += dS[tid * N_STATE + j] * tanh_out;
                    }
                    d_delta[tid] = d_delta_local;
                    d_res_scale_local[tid] += d_scale_local;  // Accumulate
                }
                __syncthreads();

                if (tid < N_STATE) {
                    float d_k_norm_local = 0.0f;
                    for (int i = 0; i < N_STATE; i++) {
                        float outer_val = delta[i] * k_norm[tid];
                        float tanh_out = tanhf(outer_val);
                        float d_tanh_out = dS[i * N_STATE + tid] * residual_scale[i];
                        float d_outer_val = d_tanh_out * (1.0f - tanh_out * tanh_out);
                        d_k_norm_local += d_outer_val * delta[i];
                    }
                    d_k_norm[tid] = d_k_norm_local;
                }
                __syncthreads();

                if (tid < N_STATE) {
                    d_v_raw[tid] = d_delta[tid];
                    float sum = 0.0f;
                    for (int i = 0; i < N_STATE; i++) {
                        sum += S[i * N_STATE + tid] * (-d_delta[i]);
                    }
                    d_k_norm[tid] += sum;
                }
                __syncthreads();

            } else if constexpr (UPDATE_TYPE == 2) {
                // NTM: S_curr = tanh(S_prev * (1 - outer(erase, k)) + outer(write * v, k))
                float erase_val_t = extra_shared[tid < N_STATE ? tid : 0];
                float write_val_t = tid < N_STATE ?
                    __bfloat162float(write_all[t * B * N_STATE + b * N_STATE + tid]) : 0.0f;

                if (tid < N_STATE) {
                    float d_erase_local = 0.0f;
                    float d_write_local = 0.0f;
                    float d_v_local = 0.0f;
                    float d_k_norm_local = 0.0f;

                    for (int j = 0; j < N_STATE; j++) {
                        float erase_val = extra_shared[tid];
                        float write_val = __bfloat162float(write_all[t * B * N_STATE + b * N_STATE + tid]);
                        float erase_outer = erase_val * k_norm[j];
                        float write_outer = (write_val * v_raw[tid]) * k_norm[j];
                        float pre = S[tid * N_STATE + j] * (1.0f - erase_outer) + write_outer;
                        float tanh_out = tanhf(pre);
                        float d_pre = dS[tid * N_STATE + j] * (1.0f - tanh_out * tanh_out);

                        // d_erase: d_pre * (-S_prev * k_norm)
                        d_erase_local += d_pre * (-S[tid * N_STATE + j] * k_norm[j]);
                        // d_write: d_pre * (v * k_norm)
                        d_write_local += d_pre * (v_raw[tid] * k_norm[j]);
                        // d_v: d_pre * (write * k_norm)
                        d_v_local += d_pre * (write_val * k_norm[j]);
                    }
                    d_erase_all[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(d_erase_local);
                    d_write_all[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(d_write_local);
                    d_v_raw[tid] = d_v_local;
                }
                __syncthreads();

                // d_k_norm from NTM
                if (tid < N_STATE) {
                    float d_k_norm_local = 0.0f;
                    for (int i = 0; i < N_STATE; i++) {
                        float erase_val = extra_shared[i];
                        float write_val = __bfloat162float(write_all[t * B * N_STATE + b * N_STATE + i]);
                        float erase_outer = erase_val * k_norm[tid];
                        float write_outer = (write_val * v_raw[i]) * k_norm[tid];
                        float pre = S[i * N_STATE + tid] * (1.0f - erase_outer) + write_outer;
                        float tanh_out = tanhf(pre);
                        float d_pre = dS[i * N_STATE + tid] * (1.0f - tanh_out * tanh_out);
                        // d_k_norm += d_pre * (-S_prev * erase + write * v)
                        d_k_norm_local += d_pre * (-S[i * N_STATE + tid] * erase_val + write_val * v_raw[i]);
                    }
                    d_k_norm[tid] = d_k_norm_local;
                }
                __syncthreads();

            } else if constexpr (UPDATE_TYPE == 3) {
                // RETRIEVED_GATE: S_curr = tanh(S_prev + outer(delta * gate, k_norm))
                if (tid < N_STATE) {
                    float gate_val = extra_shared[tid];
                    float d_delta_local = 0.0f;
                    float d_gate_local = 0.0f;
                    for (int j = 0; j < N_STATE; j++) {
                        float gated_delta = delta[tid] * gate_val;
                        float pre = S[tid * N_STATE + j] + gated_delta * k_norm[j];
                        float tanh_out = tanhf(pre);
                        float d_pre = dS[tid * N_STATE + j] * (1.0f - tanh_out * tanh_out);
                        // d_gated_delta * k_norm
                        float d_gated_delta = d_pre * k_norm[j];
                        d_delta_local += d_gated_delta * gate_val;
                        d_gate_local += d_gated_delta * delta[tid];
                    }
                    d_delta[tid] = d_delta_local;
                    d_gate_all[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(d_gate_local);
                }
                __syncthreads();

                if (tid < N_STATE) {
                    float gate_val = extra_shared[tid];
                    float d_k_norm_local = 0.0f;
                    for (int i = 0; i < N_STATE; i++) {
                        float gated_delta = delta[i] * extra_shared[i];
                        float pre = S[i * N_STATE + tid] + gated_delta * k_norm[tid];
                        float tanh_out = tanhf(pre);
                        float d_pre = dS[i * N_STATE + tid] * (1.0f - tanh_out * tanh_out);
                        d_k_norm_local += d_pre * gated_delta;
                    }
                    d_k_norm[tid] = d_k_norm_local;
                }
                __syncthreads();

                if (tid < N_STATE) {
                    d_v_raw[tid] = d_delta[tid];
                    float sum = 0.0f;
                    for (int i = 0; i < N_STATE; i++) {
                        sum += S[i * N_STATE + tid] * (-d_delta[i]);
                    }
                    d_k_norm[tid] += sum;
                }
                __syncthreads();

            } else if constexpr (UPDATE_TYPE == 4) {
                // EMA: S_curr = tanh(alpha * S_prev + (1 - alpha) * outer(v, k_norm))
                if (tid < N_STATE) {
                    float alpha_val = extra_shared[tid];
                    float d_alpha_local = 0.0f;
                    float d_v_local = 0.0f;
                    for (int j = 0; j < N_STATE; j++) {
                        float outer_val = v_raw[tid] * k_norm[j];
                        float pre = alpha_val * S[tid * N_STATE + j] + (1.0f - alpha_val) * outer_val;
                        float tanh_out = tanhf(pre);
                        float d_pre = dS[tid * N_STATE + j] * (1.0f - tanh_out * tanh_out);
                        // d_alpha = d_pre * (S_prev - outer_val)
                        d_alpha_local += d_pre * (S[tid * N_STATE + j] - outer_val);
                        // d_v += d_pre * (1 - alpha) * k_norm
                        d_v_local += d_pre * (1.0f - alpha_val) * k_norm[j];
                    }
                    d_alpha_all[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(d_alpha_local);
                    d_v_raw[tid] = d_v_local;
                }
                __syncthreads();

                if (tid < N_STATE) {
                    float alpha_val = extra_shared[tid];
                    float d_k_norm_local = 0.0f;
                    for (int i = 0; i < N_STATE; i++) {
                        float outer_val = v_raw[i] * k_norm[tid];
                        float pre = extra_shared[i] * S[i * N_STATE + tid] + (1.0f - extra_shared[i]) * outer_val;
                        float tanh_out = tanhf(pre);
                        float d_pre = dS[i * N_STATE + tid] * (1.0f - tanh_out * tanh_out);
                        d_k_norm_local += d_pre * (1.0f - extra_shared[i]) * v_raw[i];
                    }
                    d_k_norm[tid] = d_k_norm_local;
                }
                __syncthreads();
            }

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

            // Update dS for next iteration (going backward)
            // For UPDATE_TYPE==0 (DELTA): dS_prev = d_pre_nonlin (carries through tanh)
            // For UPDATE_TYPE==1 (RESIDUAL): dS_prev = dS (residual connection)
            // For others: similar based on structure
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;

                if constexpr (UPDATE_TYPE == 0 || UPDATE_TYPE == 3) {
                    // DELTA/RETRIEVED_GATE: dS_prev from tanh + outer product contribution
                    float d_pre = dS[i] * (1.0f - S[i] * S[i]);
                    float gated_delta = (UPDATE_TYPE == 3) ?
                        delta[row] * extra_shared[row] : delta[row];
                    dS[i] = d_pre + (-d_delta[row]) * k_norm[col];

                } else if constexpr (UPDATE_TYPE == 1) {
                    // RESIDUAL: dS_prev = dS (residual connection)
                    dS[i] = dS[i] + (-d_delta[row]) * k_norm[col];

                } else if constexpr (UPDATE_TYPE == 2) {
                    // NTM: dS_prev = d_pre * (1 - erase_outer)
                    float erase_val = extra_shared[row];
                    float erase_outer = erase_val * k_norm[col];
                    float write_val = __bfloat162float(write_all[t * B * N_STATE + b * N_STATE + row]);
                    float write_outer = (write_val * v_raw[row]) * k_norm[col];
                    float pre = S[i] * (1.0f - erase_outer) + write_outer;
                    float tanh_out = tanhf(pre);
                    float d_pre = dS[i] * (1.0f - tanh_out * tanh_out);
                    dS[i] = d_pre * (1.0f - erase_outer);

                } else if constexpr (UPDATE_TYPE == 4) {
                    // EMA: dS_prev = d_pre * alpha
                    float alpha_val = extra_shared[row];
                    float outer_val = v_raw[row] * k_norm[col];
                    float pre = alpha_val * S[i] + (1.0f - alpha_val) * outer_val;
                    float tanh_out = tanhf(pre);
                    float d_pre = dS[i] * (1.0f - tanh_out * tanh_out);
                    dS[i] = d_pre * alpha_val;
                }
            }
            __syncthreads();
        }
    }

    // Write accumulated residual_scale gradient
    if constexpr (UPDATE_TYPE == 1) {
        if (tid < N_STATE) {
            d_residual_scale_accum[b * N_STATE + tid] = d_res_scale_local[tid];
        }
    }
}

// ============================================================================
// Global Memory Fallback Backward Kernel for Large N_STATE
// Uses global memory for state matrices (S, dS) with in-place updates to bypass
// shared memory limits. Slower but works on all GPUs.
// ============================================================================
template<int N_STATE, int PROJ_TYPE, int UPDATE_TYPE, int GATE_TYPE>
__global__ void E74FullMatrixBackwardV2GlobalMemKernel_BF16(
    int T,
    int B,
    const __nv_bfloat16* __restrict__ k_all,
    const __nv_bfloat16* __restrict__ v_all,
    const __nv_bfloat16* __restrict__ q_all,
    const __nv_bfloat16* __restrict__ S_checkpoints,
    const __nv_bfloat16* __restrict__ Sq_cache,
    const __nv_bfloat16* __restrict__ d_output,
    __nv_bfloat16* __restrict__ d_k_all,
    __nv_bfloat16* __restrict__ d_v_all,
    __nv_bfloat16* __restrict__ d_q_all,
    const float* __restrict__ residual_scale,
    const __nv_bfloat16* __restrict__ erase_all,
    const __nv_bfloat16* __restrict__ write_all,
    const __nv_bfloat16* __restrict__ gate_all,
    const __nv_bfloat16* __restrict__ alpha_all,
    const __nv_bfloat16* __restrict__ z_gate_all,
    __nv_bfloat16* __restrict__ d_erase_all,
    __nv_bfloat16* __restrict__ d_write_all,
    __nv_bfloat16* __restrict__ d_gate_all,
    __nv_bfloat16* __restrict__ d_alpha_all,
    __nv_bfloat16* __restrict__ d_z_gate_all,
    float* __restrict__ d_residual_scale_accum,
    float* __restrict__ state_workspace,  // [B, 2, N_STATE, N_STATE] for S, dS (in-place updates)
    int checkpoint_interval
) {
    int b = blockIdx.x;
    if (b >= B) return;

    int n2 = N_STATE * N_STATE;

    // State matrices in global memory (in-place updates like shared memory version)
    float* S = state_workspace + b * 2 * n2;      // Updated in-place during forward recompute
    float* dS = S + n2;                            // Gradient accumulator

    // Only vectors in shared memory (16 * N_STATE floats = ~6KB for n=96)
    extern __shared__ float shared_mem[];
    float* k_raw = shared_mem;
    float* v_raw = k_raw + N_STATE;
    float* q_raw = v_raw + N_STATE;
    float* k_norm = q_raw + N_STATE;
    float* delta = k_norm + N_STATE;
    float* retrieved = delta + N_STATE;
    float* d_k_raw = retrieved + N_STATE;
    float* d_v_raw = d_k_raw + N_STATE;
    float* d_q_raw = d_v_raw + N_STATE;
    float* d_Sq_shared = d_q_raw + N_STATE;
    float* d_delta = d_Sq_shared + N_STATE;
    float* d_k_norm = d_delta + N_STATE;
    float* extra_shared = d_k_norm + N_STATE;

    int tid = threadIdx.x;

    // Initialize dS to zero
    for (int i = tid; i < n2; i += blockDim.x) {
        dS[i] = 0.0f;
    }
    __syncthreads();

    // Per-batch accumulator for residual_scale gradient
    float d_res_scale_local[N_STATE];
    if constexpr (UPDATE_TYPE == 1) {
        for (int i = 0; i < N_STATE; i++) {
            d_res_scale_local[i] = 0.0f;
        }
    }

    int num_segments = (T + checkpoint_interval - 1) / checkpoint_interval;

    for (int seg = num_segments - 1; seg >= 0; seg--) {
        int t_start = seg * checkpoint_interval;
        int t_end = min(t_start + checkpoint_interval, T);

        for (int t = t_end - 1; t >= t_start; t--) {
            // Load checkpoint
            for (int i = tid; i < n2; i += blockDim.x) {
                S[i] = __bfloat162float(S_checkpoints[seg * B * n2 + b * n2 + i]);
            }
            __syncthreads();

            // Recompute forward to timestep t
            for (int tt = t_start; tt <= t; tt++) {
                if (tid < N_STATE) {
                    k_raw[tid] = __bfloat162float(k_all[tt * B * N_STATE + b * N_STATE + tid]);
                    v_raw[tid] = __bfloat162float(v_all[tt * B * N_STATE + b * N_STATE + tid]);
                    q_raw[tid] = __bfloat162float(q_all[tt * B * N_STATE + b * N_STATE + tid]);
                    if constexpr (UPDATE_TYPE == 2) {
                        extra_shared[tid] = __bfloat162float(erase_all[tt * B * N_STATE + b * N_STATE + tid]);
                    } else if constexpr (UPDATE_TYPE == 3) {
                        extra_shared[tid] = __bfloat162float(gate_all[tt * B * N_STATE + b * N_STATE + tid]);
                    } else if constexpr (UPDATE_TYPE == 4) {
                        extra_shared[tid] = __bfloat162float(alpha_all[tt * B * N_STATE + b * N_STATE + tid]);
                    }
                }
                __syncthreads();

                // Normalize k
                __shared__ float k_norm_val_fwd;
                if (tid == 0) {
                    float sum_sq = 0.0f;
                    for (int i = 0; i < N_STATE; i++) sum_sq += k_raw[i] * k_raw[i];
                    k_norm_val_fwd = sqrtf(sum_sq) + 1e-6f;
                }
                __syncthreads();
                if (tid < N_STATE) {
                    k_norm[tid] = k_raw[tid] / k_norm_val_fwd;
                }
                __syncthreads();

                // Compute retrieved for DELTA/RESIDUAL/RETRIEVED_GATE
                if constexpr (UPDATE_TYPE == 0 || UPDATE_TYPE == 1 || UPDATE_TYPE == 3) {
                    if (tid < N_STATE) {
                        float sum = 0.0f;
                        for (int j = 0; j < N_STATE; j++) {
                            sum += S[tid * N_STATE + j] * k_norm[j];
                        }
                        retrieved[tid] = sum;
                    }
                    __syncthreads();
                    if (tid < N_STATE) {
                        delta[tid] = v_raw[tid] - retrieved[tid];
                    }
                    __syncthreads();
                }

                // Update state IN-PLACE only for tt < t (keep S = S_{t-1} for backward)
                // When tt == t, we keep k, v, q, delta, k_norm for backward but don't update S
                if (tt < t) {
                    for (int i = tid; i < n2; i += blockDim.x) {
                        int row = i / N_STATE;
                        int col = i % N_STATE;
                        float S_old = S[i];
                        if constexpr (UPDATE_TYPE == 0) {
                            S[i] = tanhf(S_old + delta[row] * k_norm[col]);
                        } else if constexpr (UPDATE_TYPE == 1) {
                            float outer_val = delta[row] * k_norm[col];
                            S[i] = S_old + residual_scale[row] * tanhf(outer_val);
                        } else if constexpr (UPDATE_TYPE == 2) {
                            float erase_val = extra_shared[row];
                            float write_val = __bfloat162float(write_all[tt * B * N_STATE + b * N_STATE + row]);
                            S[i] = tanhf(S_old * (1.0f - erase_val * k_norm[col]) + (write_val * v_raw[row]) * k_norm[col]);
                        } else if constexpr (UPDATE_TYPE == 3) {
                            float gate_val = extra_shared[row];
                            float gated_delta = delta[row] * gate_val;
                            S[i] = tanhf(S_old + gated_delta * k_norm[col]);
                        } else if constexpr (UPDATE_TYPE == 4) {
                            float alpha_val = extra_shared[row];
                            float outer_val = v_raw[row] * k_norm[col];
                            S[i] = tanhf(alpha_val * S_old + (1.0f - alpha_val) * outer_val);
                        }
                    }
                    __syncthreads();
                }
            }

            // ====== BACKWARD at timestep t ======
            __shared__ float k_norm_val_t;
            if (tid == 0) {
                float sum_sq = 0.0f;
                for (int i = 0; i < N_STATE; i++) sum_sq += k_raw[i] * k_raw[i];
                k_norm_val_t = sqrtf(sum_sq) + 1e-6f;
            }
            __syncthreads();

            if (tid < N_STATE) {
                d_k_raw[tid] = 0.0f;
                d_v_raw[tid] = 0.0f;
                d_q_raw[tid] = 0.0f;
            }
            __syncthreads();

            // d_Sq from output gating
            if (tid < N_STATE) {
                float Sq = __bfloat162float(Sq_cache[t * B * N_STATE + b * N_STATE + tid]);
                float d_out = __bfloat162float(d_output[t * B * N_STATE + b * N_STATE + tid]);
                float d_Sq;
                if constexpr (GATE_TYPE == 0) {
                    float sig = 1.0f / (1.0f + expf(-Sq));
                    d_Sq = d_out * (2.0f * Sq * sig + Sq * Sq * sig * (1.0f - sig));
                } else {
                    float z = __bfloat162float(z_gate_all[t * B * N_STATE + b * N_STATE + tid]);
                    float sig = 1.0f / (1.0f + expf(-z));
                    d_Sq = d_out * z * sig;
                    float d_silu_z = d_out * Sq;
                    float d_z = d_silu_z * (sig + z * sig * (1.0f - sig));
                    d_z_gate_all[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(d_z);
                }
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

            // d_q = S_t^T @ d_Sq (need to compute S_t on-the-fly from S = S_{t-1})
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    // Compute S_t[i,tid] on-the-fly
                    float S_t_ij;
                    if constexpr (UPDATE_TYPE == 0) {
                        S_t_ij = tanhf(S[i * N_STATE + tid] + delta[i] * k_norm[tid]);
                    } else if constexpr (UPDATE_TYPE == 1) {
                        float outer_val = delta[i] * k_norm[tid];
                        S_t_ij = S[i * N_STATE + tid] + residual_scale[i] * tanhf(outer_val);
                    } else if constexpr (UPDATE_TYPE == 2) {
                        float erase_val = extra_shared[i];
                        float write_val = __bfloat162float(write_all[t * B * N_STATE + b * N_STATE + i]);
                        S_t_ij = tanhf(S[i * N_STATE + tid] * (1.0f - erase_val * k_norm[tid]) + (write_val * v_raw[i]) * k_norm[tid]);
                    } else if constexpr (UPDATE_TYPE == 3) {
                        float gate_val = extra_shared[i];
                        float gated_delta = delta[i] * gate_val;
                        S_t_ij = tanhf(S[i * N_STATE + tid] + gated_delta * k_norm[tid]);
                    } else if constexpr (UPDATE_TYPE == 4) {
                        float alpha_val = extra_shared[i];
                        float outer_val = v_raw[i] * k_norm[tid];
                        S_t_ij = tanhf(alpha_val * S[i * N_STATE + tid] + (1.0f - alpha_val) * outer_val);
                    }
                    sum += S_t_ij * d_Sq_shared[i];
                }
                d_q_raw[tid] = sum;
            }
            __syncthreads();

            // Backward through state update (compute S_t on-the-fly for tanh derivative)
            if constexpr (UPDATE_TYPE == 0) {
                // DELTA backward: S_t = tanh(S_{t-1} + outer(delta, k_norm))
                if (tid < N_STATE) {
                    float d_delta_local = 0.0f;
                    for (int j = 0; j < N_STATE; j++) {
                        // Compute S_t[tid,j] on-the-fly
                        float S_t_ij = tanhf(S[tid * N_STATE + j] + delta[tid] * k_norm[j]);
                        float d_pre = dS[tid * N_STATE + j] * (1.0f - S_t_ij * S_t_ij);
                        d_delta_local += d_pre * k_norm[j];
                    }
                    d_delta[tid] = d_delta_local;
                    d_v_raw[tid] += d_delta_local;
                }
                __syncthreads();

                if (tid < N_STATE) {
                    float d_k_norm_local = 0.0f;
                    for (int i = 0; i < N_STATE; i++) {
                        // Compute S_t[i,tid] on-the-fly
                        float S_t_ij = tanhf(S[i * N_STATE + tid] + delta[i] * k_norm[tid]);
                        float d_pre = dS[i * N_STATE + tid] * (1.0f - S_t_ij * S_t_ij);
                        d_k_norm_local += d_pre * delta[i];
                    }
                    d_k_norm[tid] = d_k_norm_local;
                }
                __syncthreads();

                // d_retrieved = -d_delta
                // d_k_raw through normalization (uses S = S_{t-1} correctly here)
                if (tid < N_STATE) {
                    float d_retrieved_local = 0.0f;
                    for (int i = 0; i < N_STATE; i++) {
                        d_retrieved_local += -d_delta[i] * S[i * N_STATE + tid];
                    }
                    // d_k through normalization: d_k = (d_k_norm - k_norm * dot(k_norm, d_k_norm)) / ||k||
                    float dot = 0.0f;
                    for (int i = 0; i < N_STATE; i++) dot += k_norm[i] * (d_k_norm[i] + d_retrieved_local);
                    d_k_raw[tid] = ((d_k_norm[tid] + d_retrieved_local) - k_norm[tid] * dot) / k_norm_val_t;
                }
                __syncthreads();

                // dS_prev = dS * (1 - S_t²) for next iteration
                for (int i = tid; i < n2; i += blockDim.x) {
                    int row = i / N_STATE;
                    int col = i % N_STATE;
                    // Compute S_t on-the-fly
                    float S_t_ij = tanhf(S[i] + delta[row] * k_norm[col]);
                    float d_pre = dS[i] * (1.0f - S_t_ij * S_t_ij);
                    dS[i] = d_pre;  // This becomes dS_prev for next iteration
                }
                __syncthreads();
            }
            // Note: Other UPDATE_TYPEs would have similar backward logic here

            // Write gradients
            if (tid < N_STATE) {
                d_k_all[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(d_k_raw[tid]);
                d_v_all[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(d_v_raw[tid]);
                d_q_all[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(d_q_raw[tid]);
            }
        }
    }

    if constexpr (UPDATE_TYPE == 1) {
        if (tid < N_STATE) {
            d_residual_scale_accum[b * N_STATE + tid] = d_res_scale_local[tid];
        }
    }
}

// ============================================================================
// E74FullMatrixBackwardV2 Implementation
// ============================================================================

template<typename DataT>
E74FullMatrixBackwardV2<DataT>::E74FullMatrixBackwardV2(
    int batch_size,
    int n_state,
    int dim,
    int proj_type,
    bool use_tanh,
    int update_type,
    int gate_type,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      n_state_(n_state),
      dim_(dim),
      proj_type_(proj_type),
      use_tanh_(use_tanh),
      update_type_(update_type),
      gate_type_(gate_type),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename DataT>
void E74FullMatrixBackwardV2<DataT>::Run(
    int steps,
    const DataT* W_kvq,
    const DataT* W_k,
    const DataT* W_v,
    const DataT* W_q,
    const DataT* x,
    const DataT* S_checkpoints,
    const DataT* Sq_cache,
    const DataT* k_cache,
    const DataT* v_cache,
    const DataT* q_cache,
    const DataT* d_output,
    const DataT* residual_scale,
    const DataT* erase_cache,
    const DataT* write_cache,
    const DataT* gate_cache,
    const DataT* alpha_cache,
    const DataT* z_gate_cache,
    const DataT* W_erase,
    const DataT* W_write,
    const DataT* W_gate,
    const DataT* W_alpha,
    const DataT* W_z_gate,
    DataT* dx,
    DataT* dW_kvq,
    DataT* dW_k,
    DataT* dW_v,
    DataT* dW_q,
    DataT* d_residual_scale,
    DataT* dW_erase,
    DataT* db_erase,
    DataT* dW_write,
    DataT* db_write,
    DataT* dW_gate,
    DataT* db_gate,
    DataT* dW_alpha,
    DataT* db_alpha,
    DataT* dW_z_gate,
    DataT* db_z_gate,
    DataT* workspace
) {
    int T = steps;
    int B = batch_size_;
    int n = n_state_;
    int d = dim_;

    // Workspace layout:
    // [d_k_all: T*B*n] [d_v_all: T*B*n] [d_q_all: T*B*n]
    // [d_erase_all: T*B*n] [d_write_all: T*B*n] [d_gate_all: T*B*n] [d_alpha_all: T*B*n] [d_z_gate_all: T*B*n]
    // [d_residual_scale_accum: B*n]
    DataT* d_k_all = workspace;
    DataT* d_v_all = d_k_all + T * B * n;
    DataT* d_q_all = d_v_all + T * B * n;
    DataT* d_erase_all = d_q_all + T * B * n;
    DataT* d_write_all = d_erase_all + T * B * n;
    DataT* d_gate_all = d_write_all + T * B * n;
    DataT* d_alpha_all = d_gate_all + T * B * n;
    DataT* d_z_gate_all = d_alpha_all + T * B * n;
    float* d_residual_scale_accum = (float*)(d_z_gate_all + T * B * n);

    // Shared memory size: 2*n² + 16*n floats (in-place updates)
    // n=32: 10KB, n=48: 21KB, n=64: 37KB, n=96: 78KB (needs extended on Ada)
    int shared_size = (2 * n * n + 16 * n) * sizeof(float);
    int threads = min(256, max(n * n, n));  // Ensure at least n threads for vector ops

    // Dispatch kernel with extended shared memory for n >= 64
    #define DISPATCH_V2_BACKWARD(N_STATE, PROJ_TYPE, UPDATE_TYPE, GATE_TYPE) \
        E74FullMatrixBackwardV2Kernel_BF16<N_STATE, PROJ_TYPE, UPDATE_TYPE, GATE_TYPE><<<B, threads, shared_size, stream_>>>( \
            T, B, \
            (const __nv_bfloat16*)k_cache, \
            (const __nv_bfloat16*)v_cache, \
            (const __nv_bfloat16*)q_cache, \
            (const __nv_bfloat16*)S_checkpoints, \
            (const __nv_bfloat16*)Sq_cache, \
            (const __nv_bfloat16*)d_output, \
            (__nv_bfloat16*)d_k_all, \
            (__nv_bfloat16*)d_v_all, \
            (__nv_bfloat16*)d_q_all, \
            (const float*)residual_scale, \
            (const __nv_bfloat16*)erase_cache, \
            (const __nv_bfloat16*)write_cache, \
            (const __nv_bfloat16*)gate_cache, \
            (const __nv_bfloat16*)alpha_cache, \
            (const __nv_bfloat16*)z_gate_cache, \
            (__nv_bfloat16*)d_erase_all, \
            (__nv_bfloat16*)d_write_all, \
            (__nv_bfloat16*)d_gate_all, \
            (__nv_bfloat16*)d_alpha_all, \
            (__nv_bfloat16*)d_z_gate_all, \
            d_residual_scale_accum, \
            CHECKPOINT_INTERVAL)

    // Global memory fallback dispatch (only for UPDATE_TYPE=0 DELTA currently)
    // Allocate workspace for state matrices: 2 * B * n * n * sizeof(float) (in-place updates)
    float* state_workspace = nullptr;
    int gmem_shared_size = 16 * n * sizeof(float);  // Only vectors in shared memory

    #define DISPATCH_V2_BACKWARD_GLOBAL(N_STATE, PROJ_TYPE, UPDATE_TYPE, GATE_TYPE) \
        if (state_workspace == nullptr) { \
            cudaMalloc(&state_workspace, 2 * B * n * n * sizeof(float)); \
        } \
        E74FullMatrixBackwardV2GlobalMemKernel_BF16<N_STATE, PROJ_TYPE, UPDATE_TYPE, GATE_TYPE><<<B, threads, gmem_shared_size, stream_>>>( \
            T, B, \
            (const __nv_bfloat16*)k_cache, \
            (const __nv_bfloat16*)v_cache, \
            (const __nv_bfloat16*)q_cache, \
            (const __nv_bfloat16*)S_checkpoints, \
            (const __nv_bfloat16*)Sq_cache, \
            (const __nv_bfloat16*)d_output, \
            (__nv_bfloat16*)d_k_all, \
            (__nv_bfloat16*)d_v_all, \
            (__nv_bfloat16*)d_q_all, \
            (const float*)residual_scale, \
            (const __nv_bfloat16*)erase_cache, \
            (const __nv_bfloat16*)write_cache, \
            (const __nv_bfloat16*)gate_cache, \
            (const __nv_bfloat16*)alpha_cache, \
            (const __nv_bfloat16*)z_gate_cache, \
            (__nv_bfloat16*)d_erase_all, \
            (__nv_bfloat16*)d_write_all, \
            (__nv_bfloat16*)d_gate_all, \
            (__nv_bfloat16*)d_alpha_all, \
            (__nv_bfloat16*)d_z_gate_all, \
            d_residual_scale_accum, \
            state_workspace, \
            CHECKPOINT_INTERVAL)

    // Extended shared memory dispatch for n >= 64 (> 48KB default)
    // Falls back to global memory kernel if shared memory request fails
    #define DISPATCH_V2_BACKWARD_EXT(N_STATE, PROJ_TYPE, UPDATE_TYPE, GATE_TYPE) \
        { \
            cudaError_t attr_err = cudaFuncSetAttribute( \
                E74FullMatrixBackwardV2Kernel_BF16<N_STATE, PROJ_TYPE, UPDATE_TYPE, GATE_TYPE>, \
                cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size); \
            if (attr_err != cudaSuccess) { \
                if (UPDATE_TYPE == 0) { \
                    fprintf(stderr, "E74v2 Backward: n_state=%d shared mem failed, using global memory fallback (slower)\n", N_STATE); \
                    DISPATCH_V2_BACKWARD_GLOBAL(N_STATE, PROJ_TYPE, UPDATE_TYPE, GATE_TYPE); \
                } else { \
                    fprintf(stderr, "E74v2 Backward: n_state=%d requires %d KB shared memory but GPU limit exceeded (error: %s)\n", \
                        N_STATE, shared_size / 1024, cudaGetErrorString(attr_err)); \
                    fprintf(stderr, "Global memory fallback only supports UPDATE_TYPE=0 (DELTA). Try n_state <= 64.\n"); \
                } \
            } else { \
                DISPATCH_V2_BACKWARD(N_STATE, PROJ_TYPE, UPDATE_TYPE, GATE_TYPE); \
            } \
        }

    // Helper macro to dispatch all update types for a given n_state
    #define DISPATCH_ALL_UPDATES(N_STATE, DISPATCH_MACRO) \
        if (gate_type_ == 0) { \
            switch (update_type_) { \
                case 0: DISPATCH_MACRO(N_STATE, 2, 0, 0); break; \
                case 1: DISPATCH_MACRO(N_STATE, 2, 1, 0); break; \
                case 2: DISPATCH_MACRO(N_STATE, 2, 2, 0); break; \
                case 3: DISPATCH_MACRO(N_STATE, 2, 3, 0); break; \
                case 4: DISPATCH_MACRO(N_STATE, 2, 4, 0); break; \
            } \
        } else { \
            switch (update_type_) { \
                case 0: DISPATCH_MACRO(N_STATE, 2, 0, 1); break; \
                case 1: DISPATCH_MACRO(N_STATE, 2, 1, 1); break; \
                case 2: DISPATCH_MACRO(N_STATE, 2, 2, 1); break; \
                case 3: DISPATCH_MACRO(N_STATE, 2, 3, 1); break; \
                case 4: DISPATCH_MACRO(N_STATE, 2, 4, 1); break; \
            } \
        }

    // Dispatch by n_state (proj_type must be 2 = no_z)
    if (proj_type_ != 2) {
        printf("E74v2 Backward: Unsupported proj_type=%d (only no_z=2 supported)\n", proj_type_);
    } else if (n == 1) { DISPATCH_ALL_UPDATES(1, DISPATCH_V2_BACKWARD); }
    else if (n == 2) { DISPATCH_ALL_UPDATES(2, DISPATCH_V2_BACKWARD); }
    else if (n == 4) { DISPATCH_ALL_UPDATES(4, DISPATCH_V2_BACKWARD); }
    else if (n == 8) { DISPATCH_ALL_UPDATES(8, DISPATCH_V2_BACKWARD); }
    else if (n == 16) { DISPATCH_ALL_UPDATES(16, DISPATCH_V2_BACKWARD); }
    else if (n == 28) { DISPATCH_ALL_UPDATES(28, DISPATCH_V2_BACKWARD); }
    else if (n == 32) { DISPATCH_ALL_UPDATES(32, DISPATCH_V2_BACKWARD); }
    else if (n == 48) { DISPATCH_ALL_UPDATES(48, DISPATCH_V2_BACKWARD); }
    // n >= 64 needs extended shared memory (> 48KB)
    else if (n == 64) { DISPATCH_ALL_UPDATES(64, DISPATCH_V2_BACKWARD_EXT); }
    else if (n == 96) { DISPATCH_ALL_UPDATES(96, DISPATCH_V2_BACKWARD_EXT); }
    else {
        printf("E74v2 Backward: Unsupported n_state=%d\n", n);
    }

    #undef DISPATCH_V2_BACKWARD
    #undef DISPATCH_V2_BACKWARD_GLOBAL
    #undef DISPATCH_V2_BACKWARD_EXT
    #undef DISPATCH_ALL_UPDATES

    // Free global memory workspace if it was allocated
    if (state_workspace != nullptr) {
        cudaStreamSynchronize(stream_);  // Wait for kernel to finish
        cudaFree(state_workspace);
    }

    // Apply sigmoid derivative for update types that used sigmoid in forward
    if (update_type_ == 2) {  // NTM: d_erase was grad w.r.t. sigmoid output
        int total_erase = T * B * n;
        int threads_deriv = 256;
        int blocks_deriv = (total_erase + threads_deriv - 1) / threads_deriv;
        ApplySigmoidDerivKernel_BF16<<<blocks_deriv, threads_deriv, 0, stream_>>>(
            (__nv_bfloat16*)d_erase_all, (const __nv_bfloat16*)erase_cache, total_erase);
        // Note: write doesn't use sigmoid, so no derivative needed
    } else if (update_type_ == 3) {  // RETRIEVED_GATE: d_gate was grad w.r.t. sigmoid output
        int total_gate = T * B * n;
        int threads_deriv = 256;
        int blocks_deriv = (total_gate + threads_deriv - 1) / threads_deriv;
        ApplySigmoidDerivKernel_BF16<<<blocks_deriv, threads_deriv, 0, stream_>>>(
            (__nv_bfloat16*)d_gate_all, (const __nv_bfloat16*)gate_cache, total_gate);
    } else if (update_type_ == 4) {  // EMA: d_alpha was grad w.r.t. sigmoid output
        int total_alpha = T * B * n;
        int threads_deriv = 256;
        int blocks_deriv = (total_alpha + threads_deriv - 1) / threads_deriv;
        ApplySigmoidDerivKernel_BF16<<<blocks_deriv, threads_deriv, 0, stream_>>>(
            (__nv_bfloat16*)d_alpha_all, (const __nv_bfloat16*)alpha_cache, total_alpha);
    }

    // Accumulate weight gradients via cuBLAS (d_k_all^T @ x -> dW_k, etc.)
    const float alpha = 1.0f, beta = 0.0f;

    // dW_k = d_k_all^T @ x (if not tied)
    if (proj_type_ >= 1) {
        cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            d, n, T * B, &alpha,
            x, CUDA_R_16BF, d, d_k_all, CUDA_R_16BF, n,
            &beta, dW_k, CUDA_R_16BF, d,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            d, n, T * B, &alpha,
            x, CUDA_R_16BF, d, d_v_all, CUDA_R_16BF, n,
            &beta, dW_v, CUDA_R_16BF, d,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        if (proj_type_ == 2) {  // no_z
            cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
                d, n, T * B, &alpha,
                x, CUDA_R_16BF, d, d_q_all, CUDA_R_16BF, n,
                &beta, dW_q, CUDA_R_16BF, d,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        }
    } else {
        // tied_kvq: dW_kvq = (d_k + d_v + d_q)^T @ x
        // For simplicity, just use d_k for now (they should all be equal for tied)
        cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            d, n, T * B, &alpha,
            x, CUDA_R_16BF, d, d_k_all, CUDA_R_16BF, n,
            &beta, dW_kvq, CUDA_R_16BF, d,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }

    // dx = W_k @ d_k_all + W_v @ d_v_all + W_q @ d_q_all
    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
        d, T * B, n, &alpha,
        W_k, CUDA_R_16BF, d, d_k_all, CUDA_R_16BF, n,
        &beta, dx, CUDA_R_16BF, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    const float alpha_add = 1.0f, beta_add = 1.0f;
    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
        d, T * B, n, &alpha_add,
        W_v, CUDA_R_16BF, d, d_v_all, CUDA_R_16BF, n,
        &beta_add, dx, CUDA_R_16BF, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    if (proj_type_ == 2) {
        cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            d, T * B, n, &alpha_add,
            W_q, CUDA_R_16BF, d, d_q_all, CUDA_R_16BF, n,
            &beta_add, dx, CUDA_R_16BF, d,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }

    // Add dx contributions from update-specific projections
    if (update_type_ == 2) {  // NTM: dx += W_erase @ d_erase + W_write @ d_write
        cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            d, T * B, n, &alpha_add,
            W_erase, CUDA_R_16BF, d, d_erase_all, CUDA_R_16BF, n,
            &beta_add, dx, CUDA_R_16BF, d,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            d, T * B, n, &alpha_add,
            W_write, CUDA_R_16BF, d, d_write_all, CUDA_R_16BF, n,
            &beta_add, dx, CUDA_R_16BF, d,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    } else if (update_type_ == 3) {  // RETRIEVED_GATE: dx += W_gate @ d_gate
        cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            d, T * B, n, &alpha_add,
            W_gate, CUDA_R_16BF, d, d_gate_all, CUDA_R_16BF, n,
            &beta_add, dx, CUDA_R_16BF, d,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    } else if (update_type_ == 4) {  // EMA: dx += W_alpha @ d_alpha
        cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            d, T * B, n, &alpha_add,
            W_alpha, CUDA_R_16BF, d, d_alpha_all, CUDA_R_16BF, n,
            &beta_add, dx, CUDA_R_16BF, d,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }

    // Add dx contribution from input gate if used
    if (gate_type_ == 1) {  // INPUT gate: dx += W_z_gate @ d_z_gate
        cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            d, T * B, n, &alpha_add,
            W_z_gate, CUDA_R_16BF, d, d_z_gate_all, CUDA_R_16BF, n,
            &beta_add, dx, CUDA_R_16BF, d,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }

    // Gradients for extra weights based on update_type
    if (update_type_ == 2) {  // NTM
        cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            d, n, T * B, &alpha,
            x, CUDA_R_16BF, d, d_erase_all, CUDA_R_16BF, n,
            &beta, dW_erase, CUDA_R_16BF, d,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // db_erase = sum over (T*B) of d_erase_all
        int threads_dbe = 256;
        int blocks_dbe = (n + threads_dbe - 1) / threads_dbe;
        ReduceBiasGradKernel_BF16<<<blocks_dbe, threads_dbe, 0, stream_>>>(
            (const __nv_bfloat16*)d_erase_all, (__nv_bfloat16*)db_erase, n, T * B);

        cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            d, n, T * B, &alpha,
            x, CUDA_R_16BF, d, d_write_all, CUDA_R_16BF, n,
            &beta, dW_write, CUDA_R_16BF, d,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // db_write = sum over (T*B) of d_write_all
        int threads_dbw = 256;
        int blocks_dbw = (n + threads_dbw - 1) / threads_dbw;
        ReduceBiasGradKernel_BF16<<<blocks_dbw, threads_dbw, 0, stream_>>>(
            (const __nv_bfloat16*)d_write_all, (__nv_bfloat16*)db_write, n, T * B);

    } else if (update_type_ == 3) {  // RETRIEVED_GATE
        cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            d, n, T * B, &alpha,
            x, CUDA_R_16BF, d, d_gate_all, CUDA_R_16BF, n,
            &beta, dW_gate, CUDA_R_16BF, d,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // db_gate = sum over (T*B) of d_gate_all
        int threads_dbg = 256;
        int blocks_dbg = (n + threads_dbg - 1) / threads_dbg;
        ReduceBiasGradKernel_BF16<<<blocks_dbg, threads_dbg, 0, stream_>>>(
            (const __nv_bfloat16*)d_gate_all, (__nv_bfloat16*)db_gate, n, T * B);

    } else if (update_type_ == 4) {  // EMA
        cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            d, n, T * B, &alpha,
            x, CUDA_R_16BF, d, d_alpha_all, CUDA_R_16BF, n,
            &beta, dW_alpha, CUDA_R_16BF, d,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // db_alpha = sum over (T*B) of d_alpha_all
        int threads_db = 256;
        int blocks_db = (n + threads_db - 1) / threads_db;
        ReduceBiasGradKernel_BF16<<<blocks_db, threads_db, 0, stream_>>>(
            (const __nv_bfloat16*)d_alpha_all, (__nv_bfloat16*)db_alpha, n, T * B);
    }

    // Gradient for input gate weights
    if (gate_type_ == 1) {
        cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            d, n, T * B, &alpha,
            x, CUDA_R_16BF, d, d_z_gate_all, CUDA_R_16BF, n,
            &beta, dW_z_gate, CUDA_R_16BF, d,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // db_z_gate = sum over (T*B) of d_z_gate_all
        int threads_dbz = 256;
        int blocks_dbz = (n + threads_dbz - 1) / threads_dbz;
        ReduceBiasGradKernel_BF16<<<blocks_dbz, threads_dbz, 0, stream_>>>(
            (const __nv_bfloat16*)d_z_gate_all, (__nv_bfloat16*)db_z_gate, n, T * B);
    }

    // Sum d_residual_scale_accum across batches
    if (update_type_ == 1) {
        int threads_drs = 256;
        int blocks_drs = (n + threads_drs - 1) / threads_drs;
        ReduceResidualScaleGradKernel<<<blocks_drs, threads_drs, 0, stream_>>>(
            (const float*)d_residual_scale_accum, (float*)d_residual_scale, n, B);
    }
}

template struct E74FullMatrixBackwardV2<__nv_bfloat16>;

}  // namespace elman
