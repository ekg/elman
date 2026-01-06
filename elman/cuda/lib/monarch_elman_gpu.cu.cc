// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E7: Monarch Elman - O(n*sqrt(n)) hidden state updates via Monarch matrices
//
// h_t = tanh(monarch(B1_h, B2_h) @ h_{t-1} + monarch(B1_x, B2_x) @ x_t + b)
// output_t = h_t * silu(W_gate @ x_t + b_gate)
//
// Monarch matrix multiplication (n = m^2, so sqrt(n) = m):
// B1, B2 are block-diagonal: [m, m, m] (m blocks of m x m)
// monarch(B1, B2) @ v:
//   1. Reshape v: [B, n] -> [B, m, m]
//   2. Block matmul: z = einsum('kij,bkj->bki', B1, v_reshaped)  # [B, m, m]
//   3. Transpose: z = z.transpose(-1, -2)  # This is the "permutation"
//   4. Block matmul: out = einsum('kij,bkj->bki', B2, z)  # [B, m, m]
//   5. Flatten: [B, m, m] -> [B, n]
//
// Key: Block-diagonal matmul = batched GEMM with stride

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <algorithm>

#include "hasty/elman_ladder.h"
#include "blas.h"
#include "inline_ops.h"

namespace {

// =============================================================================
// Monarch Matrix Operations
// =============================================================================

// Kernel: Reshape [B, n] -> [B, m, m] (in-place reinterpret, no actual copy needed)
// But we need to do the transpose between the two block matmuls.

// Kernel: Transpose [B, m, m] -> [B, m, m] with transposed inner dimensions
// Input layout: out[b, i, j] = in[b, j, i]
// Since we're doing batched GEMM, we need to physically transpose the intermediate result
template<typename T>
__global__ void BatchTranspose2D(
    const int batch_size,
    const int m,
    const T* __restrict__ in,   // [B, m, m]
    T* __restrict__ out) {      // [B, m, m]

    const int b = blockIdx.z;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < batch_size && i < m && j < m) {
        const int mm = m * m;
        // in[b, j, i] -> out[b, i, j]
        out[b * mm + i * m + j] = in[b * mm + j * m + i];
    }
}

// Fused kernel: tanh(Mh + Mx + b) where Mh, Mx are monarch matmul results
template<typename T>
__global__ void FusedMonarchTanhKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ Mh,       // [B, dim] monarch(B1_h, B2_h) @ h_prev
    const T* __restrict__ Mx,       // [B, dim] monarch(B1_x, B2_x) @ x
    const T* __restrict__ b,        // [dim] bias
    T* __restrict__ h_out,          // [B, dim] output
    T* __restrict__ v_cache) {      // [B, dim] pre-activation cache (optional)

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;
        float val = static_cast<float>(Mh[idx]) + static_cast<float>(Mx[idx]) + static_cast<float>(b[d]);
        if (v_cache) v_cache[idx] = static_cast<T>(val);
        h_out[idx] = static_cast<T>(tanhf(val));
    }
}

// Kernel: output = h * silu(gate_proj + b_gate) (same as E0)
template<typename T>
__global__ void SelectiveOutputForward(
    const int batch_size,
    const int dim,
    const T* __restrict__ h,            // [B, dim]
    const T* __restrict__ gate_proj,    // [B, dim] pre-computed W_gate @ x
    const T* __restrict__ b_gate,       // [dim] learned gate bias
    T* __restrict__ output,             // [B, dim]
    T* __restrict__ gate_cache) {       // [B, dim] gate_raw for backward

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float h_val = static_cast<float>(h[idx]);
        float gp_val = static_cast<float>(gate_proj[idx]);
        float b_val = static_cast<float>(b_gate[d]);

        float gate_raw = gp_val + b_val;
        float sigmoid_val = 1.0f / (1.0f + expf(-gate_raw));
        float silu_val = gate_raw * sigmoid_val;

        output[idx] = static_cast<T>(h_val * silu_val);
        if (gate_cache) gate_cache[idx] = static_cast<T>(gate_raw);
    }
}

// Backward through selective output
template<typename T>
__global__ void SelectiveOutputBackward(
    const int batch_size,
    const int dim,
    const T* __restrict__ h,
    const T* __restrict__ gate_cache,
    const T* __restrict__ d_output,
    T* __restrict__ dh,
    T* __restrict__ d_gate_proj,
    float* __restrict__ d_b_gate) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float h_val = static_cast<float>(h[idx]);
        float gate_raw = static_cast<float>(gate_cache[idx]);
        float dout = static_cast<float>(d_output[idx]);

        float sigmoid_val = 1.0f / (1.0f + expf(-gate_raw));
        float silu_val = gate_raw * sigmoid_val;
        float dsilu = sigmoid_val * (1.0f + gate_raw * (1.0f - sigmoid_val));

        float dh_val = dout * silu_val;
        float dg_val = dout * h_val * dsilu;

        dh[idx] = static_cast<T>(dh_val);
        d_gate_proj[idx] = static_cast<T>(dg_val);
        atomicAdd(&d_b_gate[d], dg_val);
    }
}

// Backward through tanh
template<typename T>
__global__ void MonarchTanhBackwardKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ v,           // [B, dim] pre-activation
    const T* __restrict__ dh,          // [B, dim] gradient from above
    const T* __restrict__ dh_recurrent,// [B, dim] gradient from next timestep (or null)
    T* __restrict__ dv,                // [B, dim] gradient w.r.t. pre-activation
    float* __restrict__ db) {          // [dim] gradient w.r.t. bias (atomic)

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float grad = static_cast<float>(dh[idx]);
        if (dh_recurrent) grad += static_cast<float>(dh_recurrent[idx]);

        float h = tanhf(static_cast<float>(v[idx]));
        float dtanh = 1.0f - h * h;
        float dv_val = grad * dtanh;
        dv[idx] = static_cast<T>(dv_val);

        atomicAdd(&db[d], dv_val);
    }
}

// Vector add inplace
template<typename T>
__global__ void VectorAddInplace(const int n, T* __restrict__ a, const T* __restrict__ b) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = static_cast<T>(static_cast<float>(a[idx]) + static_cast<float>(b[idx]));
    }
}

// Copy float to T
template<typename T>
__global__ void CopyFloatToT(const int n, const float* __restrict__ src, T* __restrict__ dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = static_cast<T>(src[idx]);
    }
}

// =============================================================================
// Monarch Matmul Helper: monarch(B1, B2) @ v
// =============================================================================
// Performs the two-stage block-diagonal matmul with permutation between.
//
// v: [batch_size, n] where n = m * m
// B1, B2: [m, m, m] (m blocks of m x m)
// Result: [batch_size, n]
//
// The key insight: block-diagonal matmul is equivalent to batched GEMM
// where we treat batch dimension and block dimension together.
//
// Step 1: B1 @ v_reshaped
//   v_reshaped: [B, m, m] (just reinterpret)
//   For each block k in [0, m): z[b, k, :] = B1[k] @ v[b, k, :]
//   This is: [m, m] @ [m] -> [m] for each (b, k) pair
//   As batched GEMM: batchCount = B * m, each is a [m, m] @ [m, 1] -> [m, 1]
//   But cuBLAS batched GEMM with stride is better:
//     A stride = m*m (B1 blocks), B stride = m (within each batch's blocks)
//
// Step 2: Transpose z
//   z: [B, m, m] -> z_t: [B, m, m] where z_t[b, i, j] = z[b, j, i]
//
// Step 3: B2 @ z_t
//   Same pattern as step 1

template<typename T>
void MonarchMatmul(
    cublasHandle_t blas_handle,
    cudaStream_t stream,
    int batch_size,     // B
    int m,              // sqrt(n)
    const T* B1,        // [m, m, m] block-diagonal
    const T* B2,        // [m, m, m] block-diagonal
    const T* v,         // [B, n] where n = m*m
    T* tmp,             // [B, m, m] workspace for intermediate
    T* out) {           // [B, n] output

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int n = m * m;

    // Step 1: z = einsum('kij,bkj->bki', B1, v_reshaped)
    // Equivalently: for each block k, z[:, k, :] = v[:, k, :] @ B1[k].T
    // As strided batched GEMM:
    // - We want: z[b,k,i] = sum_j B1[k,i,j] * v[b,k,j]
    // - In matrix form for block k: z[:, k, :] = v[:, k, :] @ B1[k]^T
    //   where v[:, k, :] is [B, m] and B1[k] is [m, m]
    // - Batched across k with B1 having stride m*m, v having stride m, z having stride m
    //
    // cuBLAS: C = alpha * op(A) * op(B) + beta * C
    // We want: z[:, k, :] = v[:, k, :] @ B1[k]^T
    // = op(B1[k]) @ op(v[:, k, :])^T if we think in column-major
    //
    // Actually let's think carefully:
    // - v is [B, n] = [B, m, m] in row-major = [B, m*m]
    // - We want to compute: for each batch b and block k:
    //   z[b, k, i] = sum_j B1[k, i, j] * v[b, k, j]
    //
    // This is effectively: z_bk = B1_k @ v_bk where z_bk, v_bk are vectors of length m
    // We have B*m such operations, each is [m, m] @ [m, 1] -> [m, 1]
    //
    // For strided batched GEMM, we need to batch over both B and k.
    // Let's batch over k for each b, then we do B*m total GEMMs.
    // But wait, strided batched can only have one stride pattern.
    //
    // Better approach: batch over k only (m GEMMs), each GEMM does all B batches:
    // For block k: z[:, k, :] = B1[k] @ v[:, k, :]^T (thinking in terms of what CUBLAS sees)
    //
    // cuBLAS in column-major:
    // - A = B1[k], shape [m, m], stride between blocks = m*m
    // - B = v[:, k, :], shape [m, B] in column-major, but we have [B, m] stored
    //   Actually v[b, k, j] is at position b*n + k*m + j = b*m*m + k*m + j
    //   So for fixed k, v[:, k, :] has shape [B, m] with stride m*m between rows
    // - We want C = A @ B in mathematical terms, but in cuBLAS column-major:
    //   C = B^T @ A^T = (A @ B)^T
    //
    // Let me reconsider. In row-major:
    // v[b, k, j] stored at b*m*m + k*m + j
    // We want z[b, k, i] = sum_j B1[k, i, j] * v[b, k, j]
    //
    // For strided batched GEMM treating data as column-major:
    // - Treat v[:, k, :] as [m, B] matrix (column-major: consecutive in j, then b)
    //   But row-major: consecutive in j, then k, then b. So stride between b's is m*m.
    // - Treating as column-major with lda=1... no that doesn't work.
    //
    // Simplest approach: use m*B batches, each is [m, m] @ [m, 1]
    // A = B1, strideA = m*m (cycling through blocks)
    // B = v,  strideB = m (moving through blocks within a batch, then to next batch)
    // C = z,  strideC = m
    // batchCount = m * B? No, that's not right for stride.
    //
    // Actually strided batched iterates: A_i = A + i*strideA, etc.
    // So if strideB = m, after m iterations we've covered one batch (b=0, k=0..m-1)
    // But then we need strideB = m for the first m, then jump to next batch...
    //
    // The trick is to interleave. We can do m separate batched GEMMs, one per block.
    // For block k (k=0..m-1):
    //   A = B1 + k*m*m, shape [m, m]
    //   B = v + k*m, strideB = m*m (stride between batches), shape [m, B] in col-major view
    //   C = tmp + k*m, strideC = m*m
    //   batchCount = B
    //
    // This does B GEMMs per block, m blocks total = m*B total GEMM calls... expensive!
    //
    // Better: reorganize into one big GEMM.
    // We can view v as [B*m, m] and B1 as [m, m] (tiled m times).
    // Then z = v @ B1^T gives [B*m, m] @ [m, m] = [B*m, m]
    // But this doesn't account for different B1 blocks!
    //
    // The most efficient is strided batched with the right layout.
    // For monarch, the standard trick is:
    //
    // Reshape v: [B, m, m] -> process as B*m vectors of length m
    // Use batched GEMM with:
    //   - batchCount = B * m
    //   - But B1 only has m different matrices that repeat every m batches
    //
    // Unfortunately cuBLAS doesn't support this directly. We need either:
    // 1. m separate strided batched calls (one per block)
    // 2. Use gemmBatched with explicit pointers
    // 3. Tile B1 to have B*m copies
    //
    // Option 1 is cleanest. Let's do m strided batched GEMMs:

    for (int k = 0; k < m; ++k) {
        // For block k, compute z[:, k, :] = B1[k] @ v[:, k, :]^T
        // A = B1[k], shape [m, m]
        // B = v[all batches, k, :], shape [m] per batch, B batches total
        // C = tmp[all batches, k, :], shape [m] per batch
        //
        // In column-major terms for cuBLAS:
        // We want C = A @ B where A is [m, m] and B is [m, B] (m rows, B cols in col-major)
        // B's memory: v[:, k, :] means for batch b, we access v[b*m*m + k*m : b*m*m + k*m + m]
        // This is [m] elements, stride between batches = m*m
        //
        // cuBLAS: C = alpha * op(A) * op(B) + beta * C
        // With A = [m, m], B = [m, B cols stored with stride m*m], C = [m, B cols]
        // transa = CUBLAS_OP_N (A is m x m)
        // transb = CUBLAS_OP_N (B is m x 1 per batch, stacked as [m, B] in memory with stride)
        //
        // But strided batched does separate GEMMs, not one big one.
        // gemmStridedBatched: for i in range(batchCount): C[i] = A[i] @ B[i]
        // A[i] = A + i * strideA
        // B[i] = B + i * strideB
        // C[i] = C + i * strideC
        //
        // For our case with fixed A (one block):
        // A = B1 + k*m*m, strideA = 0 (same matrix for all batches)
        // B = v + k*m,    strideB = m*m (next batch is m*m away)
        // C = tmp + k*m,  strideC = m*m
        // batchCount = batch_size
        // GEMM: [m, m] @ [m, 1] = [m, 1]

        blas<T>::gemmStridedBatched(
            blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, 1, m,                          // M, N, K
            &alpha,
            B1 + k * m * m, m, 0,             // A, lda, strideA (fixed block)
            v + k * m, m, n,                  // B, ldb, strideB (jump by n=m*m between batches)
            &beta_zero,
            tmp + k * m, m, n,                // C, ldc, strideC
            batch_size);
    }

    // Step 2: Transpose tmp
    // tmp[b, i, j] -> tmp_transposed[b, j, i]
    // We need to transpose within each [m, m] block
    // Launch kernel with [m, m, B] grid
    dim3 block(16, 16, 1);
    dim3 grid((m + 15) / 16, (m + 15) / 16, batch_size);
    BatchTranspose2D<T><<<grid, block, 0, stream>>>(batch_size, m, tmp, out);

    // Step 3: B2 @ transposed = second monarch multiply
    // Same pattern as step 1, but now input is 'out' (transposed) and we write to 'tmp'
    for (int k = 0; k < m; ++k) {
        blas<T>::gemmStridedBatched(
            blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, 1, m,
            &alpha,
            B2 + k * m * m, m, 0,
            out + k * m, m, n,
            &beta_zero,
            tmp + k * m, m, n,
            batch_size);
    }

    // Copy tmp to out (or swap if we had double buffering)
    // Actually we can just use tmp as the result by swapping pointers in caller
    // But for now let's copy
    cudaMemcpyAsync(out, tmp, batch_size * n * sizeof(T), cudaMemcpyDeviceToDevice, stream);
}

// Backward for Monarch matmul: given d_out, compute d_v, d_B1, d_B2
// d_v = monarch_backward(B1, B2, d_out)
// d_B1, d_B2 = monarch_weight_backward(B1, B2, v, d_out)
template<typename T>
void MonarchMatmulBackward(
    cublasHandle_t blas_handle,
    cudaStream_t stream,
    int batch_size,
    int m,
    const T* B1,
    const T* B2,
    const T* v,         // input from forward
    const T* d_out,     // [B, n] gradient w.r.t. output
    T* d_v,             // [B, n] gradient w.r.t. input
    T* d_B1,            // [m, m, m] gradient w.r.t. B1 (accumulated)
    T* d_B2,            // [m, m, m] gradient w.r.t. B2 (accumulated)
    T* tmp1,            // [B, n] workspace
    T* tmp2) {          // [B, n] workspace

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int n = m * m;

    // Forward pass was:
    // z1 = B1 @ v_reshaped  (per block)
    // z2 = z1.transpose(-1, -2)
    // out = B2 @ z2 (per block)

    // Backward:
    // d_z2 = B2^T @ d_out (per block)
    // d_B2 += d_out @ z2^T (per block)  -- need z2 from forward!

    // First, recompute z1 (we didn't cache it)
    for (int k = 0; k < m; ++k) {
        blas<T>::gemmStridedBatched(
            blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, 1, m,
            &alpha,
            B1 + k * m * m, m, 0,
            v + k * m, m, n,
            &beta_zero,
            tmp1 + k * m, m, n,
            batch_size);
    }

    // Transpose z1 to get z2
    dim3 block(16, 16, 1);
    dim3 grid((m + 15) / 16, (m + 15) / 16, batch_size);
    BatchTranspose2D<T><<<grid, block, 0, stream>>>(batch_size, m, tmp1, tmp2);  // tmp2 = z2

    // d_z2 = B2^T @ d_out
    // d_out: [B, n] = [B, m, m]
    // For each block k: d_z2[:, k, :] = B2[k]^T @ d_out[:, k, :]
    for (int k = 0; k < m; ++k) {
        blas<T>::gemmStridedBatched(
            blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,  // Transpose B2
            m, 1, m,
            &alpha,
            B2 + k * m * m, m, 0,
            d_out + k * m, m, n,
            &beta_zero,
            tmp1 + k * m, m, n,  // tmp1 = d_z2
            batch_size);
    }

    // d_B2: for each block k, d_B2[k] += sum_b d_out[b, k, :] @ z2[b, k, :]^T
    // This is [m, 1] @ [1, m] = [m, m] summed over batches
    // Equivalently: d_B2[k] = d_out[:, k, :]^T @ z2[:, k, :] where shapes are [m, B] and [B, m]
    // = [m, B] @ [B, m] = [m, m]
    for (int k = 0; k < m; ++k) {
        // d_out[:, k, :] has shape [B, m], stored with stride n between batches
        // z2[:, k, :] has shape [B, m], stored with stride n between batches
        // We want: [m, m] += [m, B] @ [B, m]
        // In cuBLAS: C = A^T @ B where A is [B, m] and B is [B, m], both column-major

        // Actually for strided data, we do this as batched outer products then sum?
        // Or we can reshape: treat as single GEMM
        // d_out_k: [B, m] at d_out + k*m, stride m*m between rows
        // z2_k: [B, m] at tmp2 + k*m, stride m*m between rows

        // cuBLAS gemm with:
        // A = d_out + k*m, shape [B, m], lda = m*m (stride between rows), but we want [m, B]
        // If stored row-major as [B, m], in col-major view it's [m, B] with lda = m*m? No...

        // Let's just loop over batches and accumulate
        for (int b = 0; b < batch_size; ++b) {
            // d_B2[k] += d_out[b, k, :] @ z2[b, k, :]^T
            // = outer product of two m-vectors
            blas<T>::gemm(
                blas_handle,
                CUBLAS_OP_N, CUBLAS_OP_T,
                m, m, 1,
                &alpha,
                d_out + b * n + k * m, m,    // [m, 1]
                tmp2 + b * n + k * m, m,     // [m, 1] -> [1, m] when transposed
                &beta_one,
                d_B2 + k * m * m, m);        // [m, m]
        }
    }

    // d_z1 = d_z2.transpose(-1, -2)
    // d_z2 is in tmp1, transpose to get d_z1
    BatchTranspose2D<T><<<grid, block, 0, stream>>>(batch_size, m, tmp1, tmp2);  // tmp2 = d_z1

    // d_v = B1^T @ d_z1
    for (int k = 0; k < m; ++k) {
        blas<T>::gemmStridedBatched(
            blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            m, 1, m,
            &alpha,
            B1 + k * m * m, m, 0,
            tmp2 + k * m, m, n,
            &beta_zero,
            d_v + k * m, m, n,
            batch_size);
    }

    // d_B1: for each block k, d_B1[k] += sum_b d_z1[b, k, :] @ v[b, k, :]^T
    for (int k = 0; k < m; ++k) {
        for (int b = 0; b < batch_size; ++b) {
            blas<T>::gemm(
                blas_handle,
                CUBLAS_OP_N, CUBLAS_OP_T,
                m, m, 1,
                &alpha,
                tmp2 + b * n + k * m, m,
                v + b * n + k * m, m,
                &beta_one,
                d_B1 + k * m * m, m);
        }
    }
}

}  // anonymous namespace


namespace hasty {
namespace v0 {
namespace elman_ladder {

// =============================================================================
// Monarch Elman Forward
// =============================================================================

template<typename T>
MonarchElmanForward<T>::MonarchElmanForward(
    bool training,
    int batch_size,
    int dim,
    int m,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : training_(training),
      batch_size_(batch_size),
      dim_(dim),
      m_(m),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void MonarchElmanForward<T>::Run(
    int steps,
    const T* B1_h,      // [m, m, m] monarch blocks for hidden
    const T* B2_h,      // [m, m, m]
    const T* B1_x,      // [m, m, m] monarch blocks for input
    const T* B2_x,      // [m, m, m]
    const T* W_gate,    // [dim, dim] gate projection
    const T* b,         // [dim]
    const T* b_gate,    // [dim]
    const T* x,         // [T, B, dim]
    T* h,               // [T+1, B, dim]
    T* output,          // [T, B, dim]
    T* v,               // [T, B, dim] pre-activation cache
    T* gate_cache,      // [T, B, dim] gate cache
    T* workspace) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Workspace layout:
    // [Mx_all: T*BD] [gate_proj: T*BD] [tmp1: BD] [tmp2: BD] [Mh: BD]
    T* Mx_all = workspace;
    T* gate_proj = workspace + steps * BD;
    T* tmp1 = workspace + 2 * steps * BD;
    T* tmp2 = workspace + 2 * steps * BD + BD;
    T* Mh = workspace + 2 * steps * BD + 2 * BD;

    // Pre-compute monarch(B1_x, B2_x) @ x for ALL timesteps
    for (int t = 0; t < steps; ++t) {
        MonarchMatmul<T>(
            blas_handle_, stream_,
            batch_size_, m_,
            B1_x, B2_x,
            x + t * BD,
            tmp1,
            Mx_all + t * BD);
    }

    // Pre-compute gate_proj = x @ W_gate.T for all timesteps
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha,
        W_gate, dim_,
        x, dim_,
        &beta_zero,
        gate_proj, dim_);

    // Process each timestep
    for (int t = 0; t < steps; ++t) {
        const T* Mx_t = Mx_all + t * BD;
        const T* h_prev = h + t * BD;
        const T* gate_proj_t = gate_proj + t * BD;
        T* h_t = h + (t + 1) * BD;
        T* out_t = output + t * BD;
        T* v_t = training_ ? (v + t * BD) : nullptr;
        T* gate_t = training_ ? (gate_cache + t * BD) : nullptr;

        // Mh = monarch(B1_h, B2_h) @ h_prev
        MonarchMatmul<T>(
            blas_handle_, stream_,
            batch_size_, m_,
            B1_h, B2_h,
            h_prev,
            tmp1,
            Mh);

        // h_t = tanh(Mh + Mx + b)
        FusedMonarchTanhKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, Mh, Mx_t, b, h_t, v_t);

        // output = h * silu(gate_proj + b_gate)
        SelectiveOutputForward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, gate_proj_t, b_gate, out_t, gate_t);
    }
}

// =============================================================================
// Monarch Elman Backward
// =============================================================================

template<typename T>
MonarchElmanBackward<T>::MonarchElmanBackward(
    int batch_size,
    int dim,
    int m,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      m_(m),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void MonarchElmanBackward<T>::Run(
    int steps,
    const T* B1_h,
    const T* B2_h,
    const T* B1_x,
    const T* B2_x,
    const T* W_gate,
    const T* x,
    const T* h,
    const T* v,
    const T* gate_cache,
    const T* d_output,
    T* dx,
    T* dB1_h,
    T* dB2_h,
    T* dB1_x,
    T* dB2_x,
    T* dW_gate,
    T* db,
    T* d_b_gate,
    T* workspace) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int BD = batch_size_ * dim_;
    const int m3 = m_ * m_ * m_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Workspace layout:
    // [dv_all: T*BD] [d_gate_proj_all: T*BD] [dh: BD] [dh_recurrent: BD]
    // [dh_monarch: BD] [tmp1: BD] [tmp2: BD] [db_float: dim] [db_gate_float: dim]
    T* dv_all = workspace;
    T* d_gate_proj_all = workspace + steps * BD;
    T* dh = workspace + 2 * steps * BD;
    T* dh_recurrent = workspace + 2 * steps * BD + BD;
    T* dh_monarch = workspace + 2 * steps * BD + 2 * BD;
    T* tmp1 = workspace + 2 * steps * BD + 3 * BD;
    T* tmp2 = workspace + 2 * steps * BD + 4 * BD;
    float* db_float = reinterpret_cast<float*>(workspace + 2 * steps * BD + 5 * BD);
    float* db_gate_float = db_float + dim_;

    // Initialize
    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(T), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_gate_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dB1_h, 0, m3 * sizeof(T), stream_);
    cudaMemsetAsync(dB2_h, 0, m3 * sizeof(T), stream_);
    cudaMemsetAsync(dB1_x, 0, m3 * sizeof(T), stream_);
    cudaMemsetAsync(dB2_x, 0, m3 * sizeof(T), stream_);
    cudaMemsetAsync(dW_gate, 0, dim_ * dim_ * sizeof(T), stream_);

    // BPTT loop
    for (int t = steps - 1; t >= 0; --t) {
        const T* v_t = v + t * BD;
        const T* h_t = h + (t + 1) * BD;
        const T* h_prev = h + t * BD;
        const T* x_t = x + t * BD;
        const T* gate_cache_t = gate_cache + t * BD;
        const T* d_out_t = d_output + t * BD;
        T* dv_t = dv_all + t * BD;
        T* d_gate_proj_t = d_gate_proj_all + t * BD;

        // Backward through selective output
        SelectiveOutputBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, gate_cache_t, d_out_t,
            dh, d_gate_proj_t, db_gate_float);

        // Add recurrent gradient
        VectorAddInplace<T><<<num_blocks, block_size, 0, stream_>>>(BD, dh, dh_recurrent);

        // Backward through tanh
        MonarchTanhBackwardKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, v_t, dh, nullptr, dv_t, db_float);

        // dh_recurrent = monarch_backward(B1_h, B2_h) @ dv
        // Also accumulate dB1_h, dB2_h
        if (t > 0) {
            MonarchMatmulBackward<T>(
                blas_handle_, stream_,
                batch_size_, m_,
                B1_h, B2_h,
                h_prev,
                dv_t,
                dh_recurrent,
                dB1_h, dB2_h,
                tmp1, tmp2);
        } else {
            // Still need weight gradients for t=0
            MonarchMatmulBackward<T>(
                blas_handle_, stream_,
                batch_size_, m_,
                B1_h, B2_h,
                h_prev,
                dv_t,
                dh_monarch,  // dummy, not used
                dB1_h, dB2_h,
                tmp1, tmp2);
        }

        // Backward through monarch(B1_x, B2_x) @ x
        // dx_t = monarch_backward(B1_x, B2_x) @ dv_t
        // Accumulate dB1_x, dB2_x
        T* dx_t = dx + t * BD;
        MonarchMatmulBackward<T>(
            blas_handle_, stream_,
            batch_size_, m_,
            B1_x, B2_x,
            x_t,
            dv_t,
            dx_t,
            dB1_x, dB2_x,
            tmp1, tmp2);
    }

    // dx += W_gate @ d_gate_proj_all (from gate backward)
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha,
        W_gate, dim_,
        d_gate_proj_all, dim_,
        &beta_one,
        dx, dim_);

    // dW_gate = x^T @ d_gate_proj_all
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha,
        x, dim_,
        d_gate_proj_all, dim_,
        &beta_one,
        dW_gate, dim_);

    // Copy float gradients
    CopyFloatToT<T><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_float, db);
    CopyFloatToT<T><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_gate_float, d_b_gate);
}

// Explicit template instantiations
template struct MonarchElmanForward<__half>;
template struct MonarchElmanForward<__nv_bfloat16>;
template struct MonarchElmanForward<float>;
template struct MonarchElmanForward<double>;

template struct MonarchElmanBackward<__half>;
template struct MonarchElmanBackward<__nv_bfloat16>;
template struct MonarchElmanBackward<float>;
template struct MonarchElmanBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
