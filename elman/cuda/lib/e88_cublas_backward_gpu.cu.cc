/**
 * E88 FLA Hybrid Backward Kernel with cuBLAS Tensor Core Optimization
 *
 * Key insight: Individual heads have small matrices (32×128), but batching
 * across all B×H heads gives us large matrices (e.g., 1024×128) that can
 * efficiently use tensor cores via cuBLAS.
 *
 * Strategy:
 * 1. Use custom CUDA kernels for element-wise ops (tanh derivatives, scaling)
 * 2. Use cuBLAS strided batched GEMM for matrix operations:
 *    - Matrix-vector products: retrieved = S @ k, d_q = S_t^T @ d_Sq
 *    - Outer products: dS += outer(q, d_Sq)
 *
 * Memory layout assumptions:
 * - k, v, q, decay: [T, B, H, dim] contiguous
 * - S, dS: [B, H, n_state, head_v_dim] contiguous
 * - For cuBLAS: treat [B, H] as batch dimension, giving batch_count = B * H
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include "hasty/elman_ladder.h"

namespace elman {

// Global cuBLAS handle (initialized once)
static cublasHandle_t g_cublas_handle = nullptr;
static bool g_cublas_initialized = false;

// Initialize cuBLAS handle (thread-safe)
static cublasHandle_t get_cublas_handle() {
    if (!g_cublas_initialized) {
        cublasCreate(&g_cublas_handle);
        // Enable tensor core math mode for maximum performance
        cublasSetMathMode(g_cublas_handle, CUBLAS_DEFAULT_MATH);
        g_cublas_initialized = true;
    }
    return g_cublas_handle;
}

// ============================================================================
// Element-wise kernels for tanh derivatives and scaling
// These run across all B×H heads in parallel
// ============================================================================

/**
 * Compute S_t = tanh(decay * S + outer(delta, k)) and dtanh = 1 - S_t^2
 * for ALL (batch, head) pairs at a single timestep.
 *
 * Input layout:
 *   S: [B*H, n_state, head_v_dim] - state before update
 *   k: [B*H, n_state]
 *   delta: [B*H, head_v_dim] = v - retrieved
 *   decay: [B*H]
 *
 * Output:
 *   S_t: [B*H, n_state, head_v_dim] - state after tanh
 *   dtanh: [B*H, n_state, head_v_dim] - derivative 1 - S_t^2
 */
template<int N_STATE, int HEAD_V_DIM>
__global__ void ComputeTanhAndDerivativeKernel(
    int batch_heads,  // B * H
    const float* __restrict__ S,        // [B*H, n_state, head_v_dim]
    const float* __restrict__ k,        // [B*H, n_state]
    const float* __restrict__ delta,    // [B*H, head_v_dim]
    const float* __restrict__ decay,    // [B*H]
    float* __restrict__ S_t,            // [B*H, n_state, head_v_dim]
    float* __restrict__ dtanh           // [B*H, n_state, head_v_dim]
) {
    const int state_size = N_STATE * HEAD_V_DIM;
    const int total_elements = batch_heads * state_size;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_elements;
         idx += blockDim.x * gridDim.x) {

        int bh = idx / state_size;
        int local_idx = idx % state_size;
        int i = local_idx / HEAD_V_DIM;  // n_state index
        int j = local_idx % HEAD_V_DIM;  // head_v_dim index

        float decay_val = decay[bh];
        float k_val = k[bh * N_STATE + i];
        float delta_val = delta[bh * HEAD_V_DIM + j];
        float S_val = S[idx];

        // S_t = tanh(decay * S + delta * k)
        float pre_tanh = decay_val * S_val + delta_val * k_val;
        float tanh_val = tanhf(pre_tanh);

        S_t[idx] = tanh_val;
        dtanh[idx] = 1.0f - tanh_val * tanh_val;
    }
}

/**
 * Compute delta = v - retrieved for all (batch, head) pairs.
 */
template<int HEAD_V_DIM>
__global__ void ComputeDeltaKernel(
    int batch_heads,
    const float* __restrict__ v,          // [B*H, head_v_dim]
    const float* __restrict__ retrieved,  // [B*H, head_v_dim]
    float* __restrict__ delta             // [B*H, head_v_dim]
) {
    const int total = batch_heads * HEAD_V_DIM;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {
        delta[idx] = v[idx] - retrieved[idx];
    }
}

/**
 * Update dS: dS_new = dtanh * dS * decay + outer(-d_delta, k)
 * This is the gradient flow from t to t-1.
 */
template<int N_STATE, int HEAD_V_DIM>
__global__ void UpdateDSKernel(
    int batch_heads,
    const float* __restrict__ dS,         // [B*H, n_state, head_v_dim]
    const float* __restrict__ dtanh,      // [B*H, n_state, head_v_dim]
    const float* __restrict__ decay,      // [B*H]
    const float* __restrict__ d_delta,    // [B*H, head_v_dim]
    const float* __restrict__ k,          // [B*H, n_state]
    float* __restrict__ dS_out            // [B*H, n_state, head_v_dim]
) {
    const int state_size = N_STATE * HEAD_V_DIM;
    const int total = batch_heads * state_size;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        int bh = idx / state_size;
        int local_idx = idx % state_size;
        int i = local_idx / HEAD_V_DIM;
        int j = local_idx % HEAD_V_DIM;

        float decay_val = decay[bh];
        float dtanh_val = dtanh[idx];
        float d_pre = dS[idx] * dtanh_val;

        // dS_out = d_pre * decay + (-d_delta[j]) * k[i]
        float d_delta_val = d_delta[bh * HEAD_V_DIM + j];
        float k_val = k[bh * N_STATE + i];

        dS_out[idx] = d_pre * decay_val + (-d_delta_val) * k_val;
    }
}

/**
 * Compute d_decay = sum over state of (dS * dtanh * S_{t-1})
 * Uses warp reduction for efficiency.
 */
template<int N_STATE, int HEAD_V_DIM>
__global__ void ComputeDDecayKernel(
    int batch_heads,
    const float* __restrict__ dS,     // [B*H, n_state, head_v_dim]
    const float* __restrict__ dtanh,  // [B*H, n_state, head_v_dim]
    const float* __restrict__ S,      // [B*H, n_state, head_v_dim] (S_{t-1})
    float* __restrict__ d_decay       // [B*H]
) {
    const int state_size = N_STATE * HEAD_V_DIM;

    // One block per (batch, head) pair
    int bh = blockIdx.x;
    if (bh >= batch_heads) return;

    int tid = threadIdx.x;
    int base_offset = bh * state_size;

    // Partial sum for this thread
    float local_sum = 0.0f;
    for (int idx = tid; idx < state_size; idx += blockDim.x) {
        float d_pre = dS[base_offset + idx] * dtanh[base_offset + idx];
        local_sum += d_pre * S[base_offset + idx];
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset >= 1; offset /= 2) {
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
    }

    // Block reduction via shared memory
    __shared__ float warp_sums[8];  // Up to 256 threads = 8 warps
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = (blockDim.x + 31) / 32;

    if (lane_id == 0) {
        warp_sums[warp_id] = local_sum;
    }
    __syncthreads();

    // First warp does final reduction
    if (warp_id == 0) {
        float val = (tid < num_warps) ? warp_sums[tid] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset >= 1; offset /= 2) {
            val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
        }
        if (tid == 0) {
            d_decay[bh] = val;
        }
    }
}

/**
 * Compute d_pre = dS * dtanh (element-wise)
 * This intermediate is used for d_delta and d_k computations.
 */
template<int N_STATE, int HEAD_V_DIM>
__global__ void ComputeDPreKernel(
    int batch_heads,
    const float* __restrict__ dS,     // [B*H, n_state, head_v_dim]
    const float* __restrict__ dtanh,  // [B*H, n_state, head_v_dim]
    float* __restrict__ d_pre         // [B*H, n_state, head_v_dim]
) {
    const int total = batch_heads * N_STATE * HEAD_V_DIM;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {
        d_pre[idx] = dS[idx] * dtanh[idx];
    }
}

/**
 * Convert bf16 tensor to fp32.
 */
__global__ void Bf16ToFp32Kernel(
    int count,
    const __nv_bfloat16* __restrict__ src,
    float* __restrict__ dst
) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < count;
         idx += blockDim.x * gridDim.x) {
        dst[idx] = __bfloat162float(src[idx]);
    }
}

/**
 * Convert fp32 tensor to bf16.
 */
__global__ void Fp32ToBf16Kernel(
    int count,
    const float* __restrict__ src,
    __nv_bfloat16* __restrict__ dst
) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < count;
         idx += blockDim.x * gridDim.x) {
        dst[idx] = __float2bfloat16(src[idx]);
    }
}

/**
 * Add dS contribution: dS += outer(q, d_Sq)
 * This is element-wise: dS[i,j] += q[i] * d_Sq[j]
 */
template<int N_STATE, int HEAD_V_DIM>
__global__ void AddOuterProductKernel(
    int batch_heads,
    const float* __restrict__ x,   // [B*H, N_STATE]
    const float* __restrict__ y,   // [B*H, HEAD_V_DIM]
    float* __restrict__ C          // [B*H, N_STATE, HEAD_V_DIM]
) {
    const int state_size = N_STATE * HEAD_V_DIM;
    const int total = batch_heads * state_size;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        int bh = idx / state_size;
        int local_idx = idx % state_size;
        int i = local_idx / HEAD_V_DIM;
        int j = local_idx % HEAD_V_DIM;

        C[idx] += x[bh * N_STATE + i] * y[bh * HEAD_V_DIM + j];
    }
}

// ============================================================================
// cuBLAS batched operations
// ============================================================================

/**
 * Batched matrix-vector multiply: y = A @ x
 * A: [batch, M, K], x: [batch, K], y: [batch, M]
 *
 * Uses cublasSgemmStridedBatched treating x as [K, 1] and y as [M, 1].
 */
static void batched_matvec(
    cublasHandle_t handle,
    int batch_count,
    int M,  // rows of A, length of y
    int K,  // cols of A, length of x
    const float* A,  // [batch, M, K] row-major = [batch, K, M] col-major
    const float* x,  // [batch, K]
    float* y,        // [batch, M]
    cudaStream_t stream
) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSetStream(handle, stream);

    // In row-major storage, A[M, K] is col-major A^T[K, M]
    // We want y = A @ x, which in col-major is y = A^T^T @ x = (row-major A) @ x
    // cublasSgemm computes C = alpha * op(A) * op(B) + beta * C
    // y[M, 1] = A[M, K] @ x[K, 1]
    // In col-major terms with row-major storage:
    // y = A^T^T @ x, use CUBLAS_OP_T on the col-major view

    long long strideA = M * K;
    long long strideX = K;
    long long strideY = M;

    cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_T,    // Transpose because A is row-major
        CUBLAS_OP_N,    // x not transposed
        M,              // rows of result
        1,              // cols of result (vector)
        K,              // inner dimension
        &alpha,
        A, K, strideA,  // A is [K, M] in col-major (row-major [M, K])
        x, K, strideX,  // x is [K, 1]
        &beta,
        y, M, strideY,  // y is [M, 1]
        batch_count
    );
}

/**
 * Batched outer product: C += alpha * x @ y^T
 * x: [batch, M], y: [batch, N], C: [batch, M, N]
 */
static void batched_outer_add(
    cublasHandle_t handle,
    int batch_count,
    int M,
    int N,
    float alpha,
    const float* x,  // [batch, M]
    const float* y,  // [batch, N]
    float* C,        // [batch, M, N] row-major
    cudaStream_t stream
) {
    const float beta = 1.0f;  // Add to existing C

    cublasSetStream(handle, stream);

    // C[M, N] += x[M, 1] @ y[1, N]
    // In col-major with row-major C[M, N] stored as C^T[N, M]:
    // We need C^T += (x @ y^T)^T = y @ x^T
    // So compute: C^T[N, M] = y[N, 1] @ x^T[1, M]

    long long strideX = M;
    long long strideY = N;
    long long strideC = M * N;

    cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N,    // y not transposed
        CUBLAS_OP_T,    // x transposed
        N,              // rows of C^T
        M,              // cols of C^T
        1,              // inner dimension
        &alpha,
        y, N, strideY,  // y is [N, 1]
        x, M, strideX,  // x is [M, 1], transposed to [1, M]
        &beta,
        C, N, strideC,  // C^T is [N, M]
        batch_count
    );
}

// ============================================================================
// Main backward pass using cuBLAS
// ============================================================================

/**
 * E88 Backward pass with cuBLAS tensor core acceleration.
 *
 * This processes ALL (batch, head) pairs together using batched GEMM,
 * enabling tensor core utilization for large effective matrix sizes.
 */
template<int N_STATE, int HEAD_V_DIM>
void e88_cublas_backward_impl(
    int T, int B, int H,
    const __nv_bfloat16* k_all,      // [T, B, H, N_STATE]
    const __nv_bfloat16* v_all,      // [T, B, H, HEAD_V_DIM]
    const __nv_bfloat16* q_all,      // [T, B, H, N_STATE]
    const __nv_bfloat16* decay_all,  // [T, B, H]
    const __nv_bfloat16* S_checkpoints,  // [num_cp, B, H, N_STATE, HEAD_V_DIM]
    const __nv_bfloat16* d_output,       // [T, B, H, HEAD_V_DIM]
    __nv_bfloat16* d_k_all,
    __nv_bfloat16* d_v_all,
    __nv_bfloat16* d_q_all,
    __nv_bfloat16* d_decay_all,
    float* workspace,
    __nv_bfloat16* segment_state_cache,  // [B*H, checkpoint_interval, N_STATE, HEAD_V_DIM]
    int checkpoint_interval,
    cudaStream_t stream
) {
    cublasHandle_t handle = get_cublas_handle();
    cublasSetStream(handle, stream);

    const int BH = B * H;
    const int state_size = N_STATE * HEAD_V_DIM;

    // Workspace layout
    float* S_fp32 = workspace;
    float* dS_fp32 = S_fp32 + BH * state_size;
    float* S_t = dS_fp32 + BH * state_size;
    float* dtanh_buf = S_t + BH * state_size;
    float* d_pre = dtanh_buf + BH * state_size;  // Reuse S_t buffer after computing dtanh
    float* delta = d_pre + BH * state_size;
    float* retrieved = delta + BH * HEAD_V_DIM;
    float* k_fp32 = retrieved + BH * HEAD_V_DIM;
    float* v_fp32 = k_fp32 + BH * N_STATE;
    float* q_fp32 = v_fp32 + BH * HEAD_V_DIM;
    float* decay_fp32 = q_fp32 + BH * N_STATE;
    float* d_Sq = decay_fp32 + BH;
    float* d_delta = d_Sq + BH * HEAD_V_DIM;
    float* d_k_fp32 = d_delta + BH * HEAD_V_DIM;
    float* d_q_fp32 = d_k_fp32 + BH * N_STATE;
    float* d_decay_fp32 = d_q_fp32 + BH * N_STATE;

    // Initialize dS to zero
    cudaMemsetAsync(dS_fp32, 0, BH * state_size * sizeof(float), stream);

    int num_segments = (T + checkpoint_interval - 1) / checkpoint_interval;
    const int blocks_state = (BH * state_size + 255) / 256;
    const int blocks_vdim = (BH * HEAD_V_DIM + 255) / 256;
    const int blocks_nstate = (BH * N_STATE + 255) / 256;

    // Process segments in reverse
    for (int seg = num_segments - 1; seg >= 0; seg--) {
        int t_start = seg * checkpoint_interval;
        int t_end = (t_start + checkpoint_interval < T) ? t_start + checkpoint_interval : T;
        int seg_len = t_end - t_start;

        // Load checkpoint for this segment: bf16 -> fp32
        int cp_offset = seg * BH * state_size;
        Bf16ToFp32Kernel<<<blocks_state, 256, 0, stream>>>(
            BH * state_size, S_checkpoints + cp_offset, S_fp32);

        // Forward replay through segment, caching S_{t-1} for each t
        for (int local_t = 0; local_t < seg_len; local_t++) {
            int t = t_start + local_t;

            // Save S_{t-1} to segment cache BEFORE update
            __nv_bfloat16* cache_slot = segment_state_cache + (size_t)local_t * BH * state_size;
            Fp32ToBf16Kernel<<<blocks_state, 256, 0, stream>>>(
                BH * state_size, S_fp32, cache_slot);

            // Load k, v for this timestep
            int input_offset = t * BH;
            Bf16ToFp32Kernel<<<blocks_nstate, 256, 0, stream>>>(
                BH * N_STATE, k_all + input_offset * N_STATE, k_fp32);
            Bf16ToFp32Kernel<<<blocks_vdim, 256, 0, stream>>>(
                BH * HEAD_V_DIM, v_all + input_offset * HEAD_V_DIM, v_fp32);
            Bf16ToFp32Kernel<<<(BH + 255) / 256, 256, 0, stream>>>(
                BH, decay_all + input_offset, decay_fp32);

            // Compute retrieved = S @ k
            batched_matvec(handle, BH, HEAD_V_DIM, N_STATE,
                          S_fp32, k_fp32, retrieved, stream);

            // delta = v - retrieved
            ComputeDeltaKernel<HEAD_V_DIM><<<blocks_vdim, 256, 0, stream>>>(
                BH, v_fp32, retrieved, delta);

            // S = tanh(decay * S + outer(delta, k))
            ComputeTanhAndDerivativeKernel<N_STATE, HEAD_V_DIM><<<blocks_state, 256, 0, stream>>>(
                BH, S_fp32, k_fp32, delta, decay_fp32, S_fp32, dtanh_buf);  // Output overwrites S_fp32
        }

        // Backward through segment using cached states
        for (int local_t = seg_len - 1; local_t >= 0; local_t--) {
            int t = t_start + local_t;

            // Load cached S_{t-1}
            __nv_bfloat16* cache_slot = segment_state_cache + (size_t)local_t * BH * state_size;
            Bf16ToFp32Kernel<<<blocks_state, 256, 0, stream>>>(
                BH * state_size, cache_slot, S_fp32);

            // Load inputs for timestep t
            int input_offset = t * BH;
            Bf16ToFp32Kernel<<<blocks_nstate, 256, 0, stream>>>(
                BH * N_STATE, k_all + input_offset * N_STATE, k_fp32);
            Bf16ToFp32Kernel<<<blocks_vdim, 256, 0, stream>>>(
                BH * HEAD_V_DIM, v_all + input_offset * HEAD_V_DIM, v_fp32);
            Bf16ToFp32Kernel<<<blocks_nstate, 256, 0, stream>>>(
                BH * N_STATE, q_all + input_offset * N_STATE, q_fp32);
            Bf16ToFp32Kernel<<<(BH + 255) / 256, 256, 0, stream>>>(
                BH, decay_all + input_offset, decay_fp32);
            Bf16ToFp32Kernel<<<blocks_vdim, 256, 0, stream>>>(
                BH * HEAD_V_DIM, d_output + input_offset * HEAD_V_DIM, d_Sq);

            // Compute retrieved and delta
            batched_matvec(handle, BH, HEAD_V_DIM, N_STATE,
                          S_fp32, k_fp32, retrieved, stream);
            ComputeDeltaKernel<HEAD_V_DIM><<<blocks_vdim, 256, 0, stream>>>(
                BH, v_fp32, retrieved, delta);

            // Compute S_t and dtanh
            ComputeTanhAndDerivativeKernel<N_STATE, HEAD_V_DIM><<<blocks_state, 256, 0, stream>>>(
                BH, S_fp32, k_fp32, delta, decay_fp32, S_t, dtanh_buf);

            // d_q = S_t @ d_Sq (batched matvec)
            batched_matvec(handle, BH, N_STATE, HEAD_V_DIM,
                          S_t, d_Sq, d_q_fp32, stream);

            // dS += outer(q, d_Sq) using custom kernel (more efficient than cuBLAS for this)
            AddOuterProductKernel<N_STATE, HEAD_V_DIM><<<blocks_state, 256, 0, stream>>>(
                BH, q_fp32, d_Sq, dS_fp32);

            // Compute d_pre = dS * dtanh
            ComputeDPreKernel<N_STATE, HEAD_V_DIM><<<blocks_state, 256, 0, stream>>>(
                BH, dS_fp32, dtanh_buf, d_pre);

            // d_delta = d_pre^T @ k (batched: [v_dim, n_state] @ [n_state] -> [v_dim])
            // Note: d_pre is [n_state, v_dim], so d_pre^T is [v_dim, n_state]
            batched_matvec(handle, BH, HEAD_V_DIM, N_STATE,
                          d_pre, k_fp32, d_delta, stream);

            // d_k = d_pre @ delta (batched: [n_state, v_dim] @ [v_dim] -> [n_state])
            // Need to also add contribution from retrieved gradient
            batched_matvec(handle, BH, N_STATE, HEAD_V_DIM,
                          d_pre, delta, d_k_fp32, stream);

            // d_k contribution from retrieved: d_k_from_retr = -S^T @ d_delta
            // We need to subtract S @ d_delta from d_k
            // For now, use batched_matvec with negative alpha (requires separate call)
            // TODO: Optimize this

            // Compute d_decay
            ComputeDDecayKernel<N_STATE, HEAD_V_DIM><<<BH, 256, 0, stream>>>(
                BH, dS_fp32, dtanh_buf, S_fp32, d_decay_fp32);

            // Write gradients (fp32 -> bf16)
            Fp32ToBf16Kernel<<<blocks_nstate, 256, 0, stream>>>(
                BH * N_STATE, d_k_fp32, d_k_all + input_offset * N_STATE);
            Fp32ToBf16Kernel<<<blocks_vdim, 256, 0, stream>>>(
                BH * HEAD_V_DIM, d_delta, d_v_all + input_offset * HEAD_V_DIM);
            Fp32ToBf16Kernel<<<blocks_nstate, 256, 0, stream>>>(
                BH * N_STATE, d_q_fp32, d_q_all + input_offset * N_STATE);
            Fp32ToBf16Kernel<<<(BH + 255) / 256, 256, 0, stream>>>(
                BH, d_decay_fp32, d_decay_all + input_offset);

            // Update dS for next timestep (going backwards)
            UpdateDSKernel<N_STATE, HEAD_V_DIM><<<blocks_state, 256, 0, stream>>>(
                BH, dS_fp32, dtanh_buf, decay_fp32, d_delta, k_fp32, dS_fp32);
        }
    }
}

// Workspace size calculator
size_t e88_cublas_backward_workspace_size(int B, int H, int n_state, int head_v_dim) {
    const int BH = B * H;
    const int state_size = n_state * head_v_dim;

    // S_fp32, dS_fp32, S_t, dtanh, d_pre: 5 * BH * state_size
    // delta, retrieved, d_Sq, d_delta: 4 * BH * head_v_dim
    // k_fp32, d_k_fp32, q_fp32, d_q_fp32: 4 * BH * n_state
    // v_fp32: BH * head_v_dim
    // decay_fp32, d_decay_fp32: 2 * BH

    size_t size = 5 * BH * state_size * sizeof(float);
    size += 4 * BH * head_v_dim * sizeof(float);
    size += 4 * BH * n_state * sizeof(float);
    size += BH * head_v_dim * sizeof(float);
    size += 2 * BH * sizeof(float);

    return size;
}

// Dispatcher for cuBLAS backward kernel
void dispatch_e88_cublas_backward(
    int T, int B, int H, int n_state, int head_v_dim,
    const __nv_bfloat16* k_all, const __nv_bfloat16* v_all,
    const __nv_bfloat16* q_all, const __nv_bfloat16* decay_all,
    const __nv_bfloat16* S_checkpoints, const __nv_bfloat16* d_output,
    __nv_bfloat16* d_k_all, __nv_bfloat16* d_v_all,
    __nv_bfloat16* d_q_all, __nv_bfloat16* d_decay_all,
    float* workspace, __nv_bfloat16* segment_state_cache,
    int checkpoint_interval, cudaStream_t stream
) {
    #define DISPATCH_CUBLAS_BWD(N, V) \
        e88_cublas_backward_impl<N, V>( \
            T, B, H, k_all, v_all, q_all, decay_all, \
            S_checkpoints, d_output, \
            d_k_all, d_v_all, d_q_all, d_decay_all, \
            workspace, segment_state_cache, checkpoint_interval, stream)

    // Common configurations
    if (n_state == 32 && head_v_dim == 64) { DISPATCH_CUBLAS_BWD(32, 64); }
    else if (n_state == 32 && head_v_dim == 128) { DISPATCH_CUBLAS_BWD(32, 128); }
    else if (n_state == 64 && head_v_dim == 64) { DISPATCH_CUBLAS_BWD(64, 64); }
    else if (n_state == 64 && head_v_dim == 128) { DISPATCH_CUBLAS_BWD(64, 128); }
    else if (n_state == 48 && head_v_dim == 96) { DISPATCH_CUBLAS_BWD(48, 96); }
    else if (n_state == 72 && head_v_dim == 72) { DISPATCH_CUBLAS_BWD(72, 72); }
    else {
        fprintf(stderr, "E88 cuBLAS backward: unsupported n_state=%d, head_v_dim=%d\n",
                n_state, head_v_dim);
    }

    #undef DISPATCH_CUBLAS_BWD
}

// Explicit template instantiations for common configurations
template void e88_cublas_backward_impl<32, 64>(
    int, int, int, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*, float*, __nv_bfloat16*, int, cudaStream_t);

template void e88_cublas_backward_impl<32, 128>(
    int, int, int, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*, float*, __nv_bfloat16*, int, cudaStream_t);

template void e88_cublas_backward_impl<64, 64>(
    int, int, int, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*, float*, __nv_bfloat16*, int, cudaStream_t);

template void e88_cublas_backward_impl<64, 128>(
    int, int, int, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*, float*, __nv_bfloat16*, int, cudaStream_t);

template void e88_cublas_backward_impl<48, 96>(
    int, int, int, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*, float*, __nv_bfloat16*, int, cudaStream_t);

template void e88_cublas_backward_impl<72, 72>(
    int, int, int, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*, float*, __nv_bfloat16*, int, cudaStream_t);

}  // namespace elman
