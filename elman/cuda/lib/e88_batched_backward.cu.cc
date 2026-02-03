/**
 * E88 FLA Hybrid Optimized Backward Kernel
 *
 * Key optimizations:
 * 1. Batch matrix operations across all heads for cuBLAS efficiency
 * 2. Use cuBLAS GEMM for large batched matrix-vector products
 * 3. Minimize memory traffic by keeping intermediate results in registers
 *
 * The standard backward processes one head per block with 32x32 matrices.
 * This version batches across heads to create larger matrices suitable for tensor cores.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include "hasty/elman_ladder.h"

namespace elman {

// Helper: Batched matrix-vector multiply using cuBLAS
// Computes y[b,h,:] = A[b,h,:,:] @ x[b,h,:] for all (b,h) pairs
// A: [B*H, M, K], x: [B*H, K], y: [B*H, M]
static void batched_matvec_cublas(
    cublasHandle_t handle,
    int batch_count,  // B * H
    int M,            // output dim
    int K,            // input dim
    const float* A,   // [batch_count, M, K]
    const float* x,   // [batch_count, K]
    float* y,         // [batch_count, M]
    cudaStream_t stream
) {
    // Use strided batched GEMV
    // y = A @ x is equivalent to y^T = x^T @ A^T
    // cuBLAS GEMV: y = alpha * op(A) * x + beta * y

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Set stream
    cublasSetStream(handle, stream);

    // For each batch element, compute y = A @ x
    // Using strided batched gemv
    // A is stored as [batch, M, K] in row-major = [batch, K, M] in column-major
    // So we compute: y = A^T @ x in column-major terms

    long long strideA = M * K;
    long long strideX = K;
    long long strideY = M;

    // cublasSgemvStridedBatched doesn't exist, so we use cublasSgemmStridedBatched
    // Treating x as [K, 1] matrix and y as [M, 1] matrix
    // y[M,1] = A[M,K] @ x[K,1]

    cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N,  // A not transposed (in col-major = row-major transposed)
        CUBLAS_OP_N,  // x not transposed
        M,            // rows of result
        1,            // cols of result (x is a vector)
        K,            // inner dimension
        &alpha,
        A, M, strideA,      // A is [M, K] with leading dim M
        x, K, strideX,      // x is [K, 1] with leading dim K
        &beta,
        y, M, strideY,      // y is [M, 1] with leading dim M
        batch_count
    );
}

// Batched outer product: C[b,h,:,:] += alpha * x[b,h,:] outer y[b,h,:]
// C: [B*H, M, N], x: [B*H, M], y: [B*H, N]
// This is equivalent to: C = x @ y^T
static void batched_outer_product_cublas(
    cublasHandle_t handle,
    int batch_count,
    int M,
    int N,
    float alpha,
    const float* x,   // [batch_count, M]
    const float* y,   // [batch_count, N]
    float* C,         // [batch_count, M, N]
    cudaStream_t stream
) {
    const float beta = 1.0f;  // Add to existing C

    cublasSetStream(handle, stream);

    long long strideX = M;
    long long strideY = N;
    long long strideC = M * N;

    // C = x @ y^T
    // In column-major: C[M,N] = x[M,1] @ y[1,N]
    cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N,  // x not transposed
        CUBLAS_OP_T,  // y transposed (to get y^T)
        M,            // rows of C
        N,            // cols of C
        1,            // inner dimension (vectors are treated as Mx1 and 1xN)
        &alpha,
        x, M, strideX,
        y, N, strideY,
        &beta,
        C, M, strideC,
        batch_count
    );
}

/**
 * Optimized backward kernel that uses cuBLAS for batched operations.
 *
 * Key insight: Instead of each block doing tiny 32x32 matvecs,
 * we launch fewer blocks that coordinate cuBLAS calls across all heads.
 *
 * This is a hybrid approach:
 * - Use a kernel for element-wise ops (tanh derivatives, decay scaling)
 * - Use cuBLAS for matrix-vector and outer products
 */

// Element-wise kernel: compute S_t = tanh(decay*S + outer(delta,k)) and dtanh = 1-S_t^2
template<int N_STATE, int HEAD_V_DIM>
__global__ void E88ComputeTanhAndDerivative(
    int T, int B, int H,
    const float* __restrict__ S_prev,      // [B, H, N_STATE, HEAD_V_DIM]
    const float* __restrict__ k,           // [T, B, H, N_STATE]
    const float* __restrict__ delta,       // [T, B, H, HEAD_V_DIM]
    const float* __restrict__ decay,       // [T, B, H]
    float* __restrict__ S_next,            // [B, H, N_STATE, HEAD_V_DIM]
    float* __restrict__ dtanh_out,         // [B, H, N_STATE, HEAD_V_DIM]
    int t  // Current timestep
) {
    // Each thread handles one element of the state matrix
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int state_size = N_STATE * HEAD_V_DIM;
    int total_elements = B * H * state_size;

    if (idx >= total_elements) return;

    // Decompose index
    int bh = idx / state_size;
    int local_idx = idx % state_size;
    int b = bh / H;
    int h = bh % H;
    int i = local_idx / HEAD_V_DIM;  // N_STATE index
    int j = local_idx % HEAD_V_DIM;  // HEAD_V_DIM index

    // Load values
    float S_val = S_prev[idx];
    float decay_val = decay[(t * B + b) * H + h];
    float k_val = k[((t * B + b) * H + h) * N_STATE + i];
    float delta_val = delta[((t * B + b) * H + h) * HEAD_V_DIM + j];

    // Compute tanh update
    float pre_tanh = decay_val * S_val + delta_val * k_val;
    float tanh_val = tanhf(pre_tanh);

    // Store results
    S_next[idx] = tanh_val;
    dtanh_out[idx] = 1.0f - tanh_val * tanh_val;
}

// This is a framework for the optimized backward pass
// Full implementation would require significant refactoring of the calling code

}  // namespace elman
