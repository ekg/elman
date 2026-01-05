// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// Custom Tiled B2B GEMM Kernel for Low-Rank Elman Recurrence
//
// Computes: result = U_h @ V_h @ h_prev
// Using transpose trick: result^T = h_prev^T @ V_h^T @ U_h^T
//
// Design:
// - GEMM0: [batch, dim] @ [dim, rank] = [batch, rank]  (iterate over K=dim)
// - GEMM1: [batch, rank] @ [rank, dim] = [batch, dim]  (iterate over N=dim)
//
// Key insight: Keep intermediate [TILE_M, rank] in shared memory, tile over:
// - K dimension of GEMM0 (input dim)
// - N dimension of GEMM1 (output dim)
//
// This removes the CUTLASS B2B constraint that output dim <= ThreadblockShape::kN
//

#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>

namespace hasty {
namespace v0 {
namespace elman_ladder {

// =============================================================================
// Configuration constants
// =============================================================================

// Threadblock tile sizes - optimized for shared memory usage
constexpr int TILE_M = 32;      // Batch tile size (rows of output) - reduced from 64
constexpr int TILE_N = 64;      // Output dim tile size (cols of output)
constexpr int TILE_K = 32;      // K-tile for GEMM0 and GEMM1 (reduction dimension)

// Warp-level tile sizes for tensor cores (16x8x16 for fp16/bf16)
constexpr int WARP_M = 16;
constexpr int WARP_N = 16;
constexpr int WARP_K = 16;

// Number of warps per threadblock - adjusted for smaller TILE_M
constexpr int WARPS_M = TILE_M / WARP_M;  // 2
constexpr int WARPS_N = TILE_N / WARP_N;  // 4
constexpr int NUM_WARPS = WARPS_M * WARPS_N;  // 8
constexpr int THREADS_PER_BLOCK = NUM_WARPS * 32;  // 256

// Maximum rank supported - reduced to fit in shared memory
// Shared memory budget ~100KB (to leave room for other uses)
// intermediate (float): TILE_M * MAX_RANK * 4 = 32 * 260 * 4 = 33KB (float for precision)
// a0_tile (bf16): TILE_M * TILE_K * 2 = 32 * 36 * 2 = 2.25KB
// b0_tile (bf16): MAX_RANK * TILE_K * 2 = 256 * 36 * 2 = 18KB
// b1_tile (bf16): TILE_N * MAX_RANK * 2 = 64 * 260 * 2 = 33KB
// Total: ~87KB + padding
constexpr int MAX_RANK = 256;

// =============================================================================
// Shared memory layout
// =============================================================================

// Shared memory for intermediate result [TILE_M, rank]
// Plus staging buffers for input tiles

template<typename T>
struct TiledB2bSharedStorage {
    // Intermediate accumulator from GEMM0: [TILE_M, MAX_RANK]
    // IMPORTANT: Use float for accumulation to avoid precision loss across K-tiles
    // With dim=1536, TILE_K=32, we have 48 K-tiles - bf16 would lose too much precision
    float intermediate[TILE_M][MAX_RANK + 4];  // +4 padding to avoid bank conflicts

    // Staging for GEMM0 inputs
    T a0_tile[TILE_M][TILE_K + 4];   // h_prev tile [TILE_M, TILE_K]
    T b0_tile[MAX_RANK][TILE_K + 4]; // V_h tile [rank, TILE_K]

    // Staging for GEMM1 B input
    T b1_tile[TILE_N][MAX_RANK + 4]; // U_h tile [TILE_N, rank]
};

// =============================================================================
// Helper: Load tile from global memory to shared memory
// =============================================================================

template<typename T>
__device__ __forceinline__ void load_tile_a0(
    T* smem_dst,        // [TILE_M][TILE_K+4]
    const T* gmem_src,  // [batch, dim]
    int batch_size,
    int dim,
    int m_offset,       // Batch offset
    int k_offset,       // Dim offset
    int smem_stride) {

    const int tid = threadIdx.x;
    const int total_elements = TILE_M * TILE_K;

    for (int i = tid; i < total_elements; i += blockDim.x) {
        int m = i / TILE_K;
        int k = i % TILE_K;

        int global_m = m_offset + m;
        int global_k = k_offset + k;

        T val = T(0);
        if (global_m < batch_size && global_k < dim) {
            val = gmem_src[global_m * dim + global_k];
        }

        smem_dst[m * smem_stride + k] = val;
    }
}

template<typename T>
__device__ __forceinline__ void load_tile_b0(
    T* smem_dst,        // [rank][TILE_K+4]
    const T* gmem_src,  // [rank, dim] (V_h stored row-major)
    int rank,
    int dim,
    int k_offset,       // Dim offset
    int smem_stride) {

    const int tid = threadIdx.x;
    const int total_elements = rank * TILE_K;

    for (int i = tid; i < total_elements; i += blockDim.x) {
        int r = i / TILE_K;
        int k = i % TILE_K;

        int global_k = k_offset + k;

        T val = T(0);
        if (r < rank && global_k < dim) {
            val = gmem_src[r * dim + global_k];
        }

        smem_dst[r * smem_stride + k] = val;
    }
}

template<typename T>
__device__ __forceinline__ void load_tile_b1(
    T* smem_dst,        // [TILE_N][rank+4]
    const T* gmem_src,  // [dim, rank] (U_h stored row-major)
    int dim,
    int rank,
    int n_offset,       // Output dim offset
    int smem_stride) {

    const int tid = threadIdx.x;
    const int total_elements = TILE_N * rank;

    for (int i = tid; i < total_elements; i += blockDim.x) {
        int n = i / rank;
        int r = i % rank;

        int global_n = n_offset + n;

        T val = T(0);
        if (global_n < dim && r < rank) {
            val = gmem_src[global_n * rank + r];
        }

        smem_dst[n * smem_stride + r] = val;
    }
}

// =============================================================================
// GEMM0 accumulate: intermediate += A0 @ B0^T
// A0: [TILE_M, TILE_K], B0: [rank, TILE_K]
// Result: [TILE_M, rank]
// =============================================================================

template<typename T>
__device__ __forceinline__ void gemm0_accumulate(
    float* intermediate,  // [TILE_M][rank+4] in smem (float for accumulation precision)
    const T* a0_tile,    // [TILE_M][TILE_K+4] in smem
    const T* b0_tile,    // [rank][TILE_K+4] in smem
    int rank,
    int smem_stride_inter,
    int smem_stride_a,
    int smem_stride_b) {

    // Each thread computes a subset of the output
    const int tid = threadIdx.x;
    const int total_output = TILE_M * rank;

    for (int i = tid; i < total_output; i += blockDim.x) {
        int m = i / rank;
        int r = i % rank;

        float acc = 0.0f;
        for (int k = 0; k < TILE_K; ++k) {
            float a_val = static_cast<float>(a0_tile[m * smem_stride_a + k]);
            float b_val = static_cast<float>(b0_tile[r * smem_stride_b + k]);
            acc += a_val * b_val;
        }

        // Accumulate to intermediate (stays in float - no precision loss!)
        intermediate[m * smem_stride_inter + r] += acc;
    }
}

// =============================================================================
// GEMM1 compute: output_tile = intermediate @ B1^T
// intermediate: [TILE_M, rank], B1: [TILE_N, rank]
// Result: [TILE_M, TILE_N]
// =============================================================================

template<typename T>
__device__ __forceinline__ void gemm1_compute_and_store(
    T* output,           // [batch, dim] in global memory
    const float* intermediate,  // [TILE_M][rank+4] in smem (float for precision)
    const T* b1_tile,    // [TILE_N][rank+4] in smem
    int batch_size,
    int dim,
    int rank,
    int m_offset,
    int n_offset,
    int smem_stride_inter,
    int smem_stride_b) {

    // Each thread computes and stores a subset of the output tile
    const int tid = threadIdx.x;
    const int total_output = TILE_M * TILE_N;

    for (int i = tid; i < total_output; i += blockDim.x) {
        int m = i / TILE_N;
        int n = i % TILE_N;

        int global_m = m_offset + m;
        int global_n = n_offset + n;

        if (global_m >= batch_size || global_n >= dim) continue;

        float acc = 0.0f;
        for (int r = 0; r < rank; ++r) {
            float inter_val = intermediate[m * smem_stride_inter + r];  // Already float
            float b_val = static_cast<float>(b1_tile[n * smem_stride_b + r]);
            acc += inter_val * b_val;
        }

        output[global_m * dim + global_n] = static_cast<T>(acc);
    }
}

// =============================================================================
// Main kernel: Tiled B2B GEMM
// Computes: output = input @ V_h^T @ U_h^T (using row-major convention)
// =============================================================================

template<typename T>
__global__ void TiledB2bGemmKernel(
    const int batch_size,
    const int dim,
    const int rank,
    const T* __restrict__ input,   // [batch, dim] h_prev
    const T* __restrict__ V_h,     // [rank, dim]
    const T* __restrict__ U_h,     // [dim, rank]
    T* __restrict__ output) {      // [batch, dim]

    extern __shared__ char smem_raw[];
    TiledB2bSharedStorage<T>& smem = *reinterpret_cast<TiledB2bSharedStorage<T>*>(smem_raw);

    // This threadblock handles one M-tile (batch chunk)
    const int m_tile_idx = blockIdx.x;
    const int m_offset = m_tile_idx * TILE_M;

    if (m_offset >= batch_size) return;

    // Initialize float intermediate to zero (float for accumulation precision)
    const int tid = threadIdx.x;
    for (int i = tid; i < TILE_M * rank; i += blockDim.x) {
        int m = i / rank;
        int r = i % rank;
        smem.intermediate[m][r] = 0.0f;
    }
    __syncthreads();

    // =================================================================
    // GEMM0: intermediate = input[m_tile] @ V_h^T
    // Iterate over K (dim) in tiles
    // =================================================================

    for (int k_offset = 0; k_offset < dim; k_offset += TILE_K) {
        // Load input tile [TILE_M, TILE_K]
        load_tile_a0<T>(
            &smem.a0_tile[0][0],
            input,
            batch_size, dim,
            m_offset, k_offset,
            TILE_K + 4);

        // Load V_h tile [rank, TILE_K]
        load_tile_b0<T>(
            &smem.b0_tile[0][0],
            V_h,
            rank, dim,
            k_offset,
            TILE_K + 4);

        __syncthreads();

        // Accumulate: intermediate += A0 @ B0^T
        gemm0_accumulate<T>(
            &smem.intermediate[0][0],
            &smem.a0_tile[0][0],
            &smem.b0_tile[0][0],
            rank,
            MAX_RANK + 4,  // intermediate stride
            TILE_K + 4,    // a0 stride
            TILE_K + 4);   // b0 stride

        __syncthreads();
    }

    // =================================================================
    // GEMM1: output[m_tile] = intermediate @ U_h^T
    // Iterate over N (output dim) in tiles
    // =================================================================

    for (int n_offset = 0; n_offset < dim; n_offset += TILE_N) {
        // Load U_h tile [TILE_N, rank]
        load_tile_b1<T>(
            &smem.b1_tile[0][0],
            U_h,
            dim, rank,
            n_offset,
            MAX_RANK + 4);

        __syncthreads();

        // Compute and store output tile
        gemm1_compute_and_store<T>(
            output,
            &smem.intermediate[0][0],
            &smem.b1_tile[0][0],
            batch_size, dim, rank,
            m_offset, n_offset,
            MAX_RANK + 4,  // intermediate stride
            MAX_RANK + 4); // b1 stride

        __syncthreads();
    }
}

// =============================================================================
// Launch wrapper
// =============================================================================

template<typename T>
void LaunchTiledB2bGemm(
    int batch_size,
    int dim,
    int rank,
    const T* input,
    const T* V_h,
    const T* U_h,
    T* output,
    cudaStream_t stream) {

    // Number of M-tiles
    int num_m_tiles = (batch_size + TILE_M - 1) / TILE_M;

    // Shared memory size
    size_t smem_size = sizeof(TiledB2bSharedStorage<T>);

    // Request extended shared memory if needed (SM80 default is 48KB)
    if (smem_size >= (48 << 10)) {
        cudaFuncSetAttribute(TiledB2bGemmKernel<T>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                            smem_size);
    }

    // Launch kernel
    TiledB2bGemmKernel<T><<<num_m_tiles, THREADS_PER_BLOCK, smem_size, stream>>>(
        batch_size, dim, rank, input, V_h, U_h, output);
}

// Explicit instantiations
template void LaunchTiledB2bGemm<__half>(int, int, int, const __half*, const __half*, const __half*, __half*, cudaStream_t);
template void LaunchTiledB2bGemm<__nv_bfloat16>(int, int, int, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, cudaStream_t);
template void LaunchTiledB2bGemm<float>(int, int, int, const float*, const float*, const float*, float*, cudaStream_t);

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
