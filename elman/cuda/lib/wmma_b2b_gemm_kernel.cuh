// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// WMMA Tensor Core B2B GEMM Kernel for Low-Rank Elman Recurrence
//
// Computes: result = h_prev @ V_h^T @ U_h^T
//
// Uses WMMA (Warp Matrix Multiply Accumulate) for tensor core acceleration.
// Key: Keep intermediate [TILE_M, rank] in shared memory between GEMM0 and GEMM1.
//

#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>

using namespace nvcuda;

namespace hasty {
namespace v0 {
namespace elman_ladder {

// =============================================================================
// Configuration - must match WMMA requirements
// =============================================================================

// WMMA tile size (fixed by hardware for fp16/bf16 with float accumulators)
static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

// Block tile sizes - tuned for SM89 shared memory limit (~99KB max)
// Total smem: intermediate(32*260*4=33KB) + b1_tile(256*40*2=20KB) + inter_half(768B) = ~54KB
// Requires extended shared memory (cudaFuncAttributeMaxDynamicSharedMemorySize)
static constexpr int BLOCK_TILE_M = 32;   // Rows per block
static constexpr int BLOCK_TILE_N = 32;   // Cols per output tile (reduced from 64)
static constexpr int BLOCK_TILE_K = 32;   // K-tile for loads

// Warps per block (2x2 warp tile grid = 4 warps for 32x32 output)
static constexpr int WMMA_WARPS_PER_BLOCK = 4;
static constexpr int WMMA_THREADS_PER_BLOCK = WMMA_WARPS_PER_BLOCK * 32;

// Padding for shared memory - MUST be multiple of 8 for WMMA 16-byte alignment
// With __half (2 bytes), stride * 2 must be multiple of 16, so stride multiple of 8
static constexpr int SMEM_PAD = 8;  // Changed back to 8 for WMMA alignment

// Max rank supported
static constexpr int MAX_RANK_WMMA = 256;

// =============================================================================
// Shared memory structure
// =============================================================================

template<typename T>
struct WmmaB2bSmem {
    // Intermediate result from GEMM0: [BLOCK_TILE_M, MAX_RANK]
    // Use float for WMMA accumulator storage (float accumulator -> float storage -> half load)
    float intermediate[BLOCK_TILE_M][MAX_RANK_WMMA + SMEM_PAD];

    // Tile buffers - reused between GEMM0 and GEMM1
    // GEMM0: a_tile=[BLOCK_TILE_M, BLOCK_TILE_K], b0_tile=[BLOCK_TILE_K, rank]
    // GEMM1: b1_tile=[rank, BLOCK_TILE_N]
    // Using union to share memory since GEMM0 and GEMM1 don't overlap
    union {
        struct {
            __half a_tile[BLOCK_TILE_M][BLOCK_TILE_K + SMEM_PAD];
            __half b0_tile[BLOCK_TILE_K][MAX_RANK_WMMA + SMEM_PAD];
        };
        __half b1_tile[MAX_RANK_WMMA][BLOCK_TILE_N + SMEM_PAD];
    };
    // Temp buffer for float->half conversion in GEMM1 - ONE PER WARP to avoid race
    // Use exactly WMMA_K stride (16) for 16-byte alignment in WMMA loads
    __half inter_half[WMMA_WARPS_PER_BLOCK][WMMA_M][WMMA_K];
    // Temp buffer for WMMA output before copying to global (one per warp)
    // Each warp needs 16x16 floats = 256 floats
    float out_buf[WMMA_WARPS_PER_BLOCK][WMMA_M * WMMA_N];
};

// =============================================================================
// Convert bf16 to half (for WMMA compatibility)
// =============================================================================

__device__ __forceinline__ __half to_half(__nv_bfloat16 x) {
    return __float2half(__bfloat162float(x));
}

__device__ __forceinline__ __half to_half(__half x) {
    return x;
}

__device__ __forceinline__ __half to_half(float x) {
    return __float2half(x);
}

template<typename T>
__device__ __forceinline__ T from_float(float x);

template<>
__device__ __forceinline__ __half from_float<__half>(float x) {
    return __float2half(x);
}

template<>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}

template<>
__device__ __forceinline__ float from_float<float>(float x) {
    return x;
}

template<>
__device__ __forceinline__ double from_float<double>(float x) {
    return static_cast<double>(x);
}

// =============================================================================
// Load tiles to shared memory with conversion to half
// =============================================================================

template<typename T>
__device__ void load_a_tile(
    __half* dst,
    const T* src,
    int batch_size, int dim,
    int m_off, int k_off,
    int dst_stride) {

    const int tid = threadIdx.x;
    const int total = BLOCK_TILE_M * BLOCK_TILE_K;

    for (int i = tid; i < total; i += blockDim.x) {
        int m = i / BLOCK_TILE_K;
        int k = i % BLOCK_TILE_K;
        int gm = m_off + m;
        int gk = k_off + k;

        float val = 0.0f;
        if (gm < batch_size && gk < dim) {
            val = static_cast<float>(src[gm * dim + gk]);
        }
        dst[m * dst_stride + k] = __float2half(val);
    }
}

template<typename T>
__device__ void load_b0_tile(
    __half* dst,
    const T* V_h,  // [rank, dim]
    int rank, int dim,
    int k_off,     // Offset in dim
    int dst_stride) {

    const int tid = threadIdx.x;
    const int total = BLOCK_TILE_K * rank;

    // Load V_h^T[k_off:k_off+TILE_K, 0:rank]
    // V_h^T[k, r] = V_h[r, k]
    for (int i = tid; i < total; i += blockDim.x) {
        int k = i / rank;
        int r = i % rank;
        int gk = k_off + k;

        float val = 0.0f;
        if (gk < dim && r < rank) {
            val = static_cast<float>(V_h[r * dim + gk]);
        }
        dst[k * dst_stride + r] = __float2half(val);
    }
}

template<typename T>
__device__ void load_b1_tile(
    __half* dst,
    const T* U_h,  // [dim, rank]
    int dim, int rank,
    int n_off,     // Offset in output dim
    int dst_stride) {

    const int tid = threadIdx.x;
    const int total = rank * BLOCK_TILE_N;

    // Load U_h^T[0:rank, n_off:n_off+TILE_N]
    // U_h^T[r, n] = U_h[n, r]
    for (int i = tid; i < total; i += blockDim.x) {
        int r = i / BLOCK_TILE_N;
        int n = i % BLOCK_TILE_N;
        int gn = n_off + n;

        float val = 0.0f;
        if (r < rank && gn < dim) {
            val = static_cast<float>(U_h[gn * rank + r]);
        }
        dst[r * dst_stride + n] = __float2half(val);
    }
}

// =============================================================================
// WMMA-based GEMM for one warp: C[16,16] += A[16,K] @ B[K,16]
// =============================================================================

__device__ void warp_gemm_16x16(
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>& c_frag,
    const __half* a_ptr, int a_stride,  // [16, K]
    const __half* b_ptr, int b_stride,  // [K, 16]
    int K) {

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;

    // Accumulate over K dimension
    for (int k = 0; k < K; k += WMMA_K) {
        // Load A fragment from row k
        wmma::load_matrix_sync(a_frag, a_ptr + k, a_stride);

        // Load B fragment from row k
        wmma::load_matrix_sync(b_frag, b_ptr + k * b_stride, b_stride);

        // MMA: C += A @ B
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
}

// =============================================================================
// Main kernel
// =============================================================================

template<typename T>
__global__ void WmmaB2bGemmKernel(
    int batch_size, int dim, int rank,
    const T* __restrict__ input,
    const T* __restrict__ V_h,
    const T* __restrict__ U_h,
    T* __restrict__ output) {

    extern __shared__ char smem_raw[];
    WmmaB2bSmem<T>& smem = *reinterpret_cast<WmmaB2bSmem<T>*>(smem_raw);

    const int block_m = blockIdx.x * BLOCK_TILE_M;
    if (block_m >= batch_size) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;

    // Each warp computes one or more 16x16 tiles
    // For intermediate [32, rank], we have 2 * (rank/16) tiles
    // With 4 warps, we tile: 2 warps in M dimension, 2 warps in N dimension
    // Each warp handles half of the M tiles and half of the N tiles

    // Compute warp's position in the 2x2 warp grid
    const int warp_m = warp_id / 2;  // 0-1, each handles 16 rows
    const int warp_n_base = (warp_id % 2);  // 0 or 1
    // Total N-tiles = rank/16, each warp in N dimension handles half
    const int num_n_tiles = rank / WMMA_N;
    const int n_tiles_per_warp = (num_n_tiles + 1) / 2;  // Ceiling division
    const int warp_n_start = warp_n_base * n_tiles_per_warp;

    // =================================================================
    // GEMM0: intermediate = input @ V_h^T
    // =================================================================

    // Initialize intermediate to zero (float storage)
    for (int i = tid; i < BLOCK_TILE_M * rank; i += blockDim.x) {
        int m = i / rank;
        int r = i % rank;
        smem.intermediate[m][r] = 0.0f;
    }
    __syncthreads();

    // Fragment accumulators for each warp's tiles
    // Each warp handles tiles at (warp_m*16, warp_n_start*16 + i*16) for i=0..7
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frags[8];

    for (int i = 0; i < 8; ++i) {
        wmma::fill_fragment(c_frags[i], 0.0f);
    }

    // Iterate over K dimension (input dim)
    for (int k_off = 0; k_off < dim; k_off += BLOCK_TILE_K) {
        // Load A tile [BLOCK_TILE_M, BLOCK_TILE_K]
        load_a_tile<T>(&smem.a_tile[0][0], input, batch_size, dim,
                       block_m, k_off, BLOCK_TILE_K + SMEM_PAD);

        // Load B tile V_h^T [BLOCK_TILE_K, rank]
        load_b0_tile<T>(&smem.b0_tile[0][0], V_h, rank, dim,
                        k_off, MAX_RANK_WMMA + SMEM_PAD);

        __syncthreads();

        // Each warp processes its tiles
        int m_start = warp_m * WMMA_M;

        for (int tile_n = 0; tile_n < n_tiles_per_warp && (warp_n_start + tile_n) < num_n_tiles; ++tile_n) {
            int n_start = (warp_n_start + tile_n) * WMMA_N;

            // Get pointers to A and B for this tile
            const __half* a_ptr = &smem.a_tile[m_start][0];
            const __half* b_ptr = &smem.b0_tile[0][n_start];

            // WMMA GEMM: C[16,16] += A[16,K] @ B[K,16]
            warp_gemm_16x16(c_frags[tile_n],
                           a_ptr, BLOCK_TILE_K + SMEM_PAD,
                           b_ptr, MAX_RANK_WMMA + SMEM_PAD,
                           min(BLOCK_TILE_K, dim - k_off));
        }

        __syncthreads();
    }

    // Store intermediate results to shared memory
    int m_start = warp_m * WMMA_M;
    for (int tile_n = 0; tile_n < n_tiles_per_warp && (warp_n_start + tile_n) < num_n_tiles; ++tile_n) {
        int n_start = (warp_n_start + tile_n) * WMMA_N;
        wmma::store_matrix_sync(&smem.intermediate[m_start][n_start],
                               c_frags[tile_n], MAX_RANK_WMMA + SMEM_PAD,
                               wmma::mem_row_major);
    }

    __syncthreads();

    // =================================================================
    // GEMM1: output = intermediate @ U_h^T
    // =================================================================

    // Use smem.inter_half for float->half conversion [WMMA_M, WMMA_K]
    // Stride is exactly WMMA_K (16) for 16-byte alignment
    const int inter_half_stride = WMMA_K;

    // Iterate over output N dimension
    for (int n_off = 0; n_off < dim; n_off += BLOCK_TILE_N) {
        // Load U_h^T [rank, BLOCK_TILE_N] into b1_tile
        load_b1_tile<T>(&smem.b1_tile[0][0], U_h, dim, rank,
                        n_off, BLOCK_TILE_N + SMEM_PAD);

        __syncthreads();

        // Each warp computes one 16x16 output tile
        // Output grid is [BLOCK_TILE_M/16, BLOCK_TILE_N/16] = [2, 2] = 4 tiles
        // With 4 warps, each warp handles 1 tile

        for (int tile_idx = warp_id; tile_idx < 4; tile_idx += WMMA_WARPS_PER_BLOCK) {
            int tile_m = tile_idx / 2;
            int tile_n = tile_idx % 2;

            int m_local = tile_m * WMMA_M;
            int n_local = tile_n * WMMA_N;

            // Initialize accumulator
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> out_frag;
            wmma::fill_fragment(out_frag, 0.0f);

            // GEMM1: C[16,16] = A[16,rank] @ B[rank,16]
            // Process rank in chunks of WMMA_K
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a1_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b1_frag;

            for (int k = 0; k < rank; k += WMMA_K) {
                // Convert this chunk of intermediate from float to half
                // Use per-warp buffer to avoid race condition
                for (int i = tid % 32; i < WMMA_M * WMMA_K; i += 32) {
                    int m = i / WMMA_K;
                    int kk = i % WMMA_K;
                    smem.inter_half[warp_id][m][kk] =
                        __float2half(smem.intermediate[m_local + m][k + kk]);
                }
                __syncwarp();

                // Load converted intermediate chunk from per-warp buffer
                wmma::load_matrix_sync(a1_frag, &smem.inter_half[warp_id][0][0], inter_half_stride);

                // Load B chunk from b1_tile
                wmma::load_matrix_sync(b1_frag, &smem.b1_tile[k][n_local], BLOCK_TILE_N + SMEM_PAD);

                // MMA
                wmma::mma_sync(out_frag, a1_frag, b1_frag, out_frag);
            }

            // Store result to dedicated per-warp output buffer
            float* warp_out_buf = &smem.out_buf[warp_id][0];
            wmma::store_matrix_sync(warp_out_buf, out_frag, WMMA_N, wmma::mem_row_major);

            // Copy to global with bounds check
            const int lane = threadIdx.x % 32;
            for (int idx = lane; idx < WMMA_M * WMMA_N; idx += 32) {
                int local_m = idx / WMMA_N;
                int local_n = idx % WMMA_N;
                int global_m = block_m + m_local + local_m;
                int global_n = n_off + n_local + local_n;

                if (global_m < batch_size && global_n < dim) {
                    output[global_m * dim + global_n] =
                        from_float<T>(warp_out_buf[local_m * WMMA_N + local_n]);
                }
            }
            __syncwarp();  // Ensure all threads done before next tile
        }

        __syncthreads();
    }
}

// =============================================================================
// Launch wrapper
// =============================================================================

// Check if type is supported by WMMA kernel
template<typename T>
struct WmmaSupportedType {
    static constexpr bool value = false;
};

template<>
struct WmmaSupportedType<__half> {
    static constexpr bool value = true;
};

template<>
struct WmmaSupportedType<__nv_bfloat16> {
    static constexpr bool value = true;
};

template<typename T>
void LaunchWmmaB2bGemm(
    int batch_size, int dim, int rank,
    const T* input, const T* V_h, const T* U_h,
    T* output, cudaStream_t stream) {

    // WMMA only supports half and bfloat16
    // Use if constexpr to prevent kernel compilation for unsupported types
    if constexpr (WmmaSupportedType<T>::value) {
        // Constraints
        if (rank % WMMA_K != 0 || rank > MAX_RANK_WMMA) {
            return;  // Unsupported config
        }

        int num_blocks = (batch_size + BLOCK_TILE_M - 1) / BLOCK_TILE_M;
        size_t smem_size = sizeof(WmmaB2bSmem<T>);

        // Request extended shared memory if needed (SM80+ supports up to 99-164KB)
        if (smem_size > 48 * 1024) {
            cudaError_t err = cudaFuncSetAttribute(WmmaB2bGemmKernel<T>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                smem_size);
            if (err != cudaSuccess) {
                // Fall back to caller's backup path
                return;
            }
        }

        WmmaB2bGemmKernel<T><<<num_blocks, WMMA_THREADS_PER_BLOCK, smem_size, stream>>>(
            batch_size, dim, rank, input, V_h, U_h, output);
    }
    // For float/double, do nothing - caller should fall back to cuBLAS
}

// Explicit instantiations - only for supported types
template void LaunchWmmaB2bGemm<__half>(int, int, int, const __half*, const __half*, const __half*, __half*, cudaStream_t);
template void LaunchWmmaB2bGemm<__nv_bfloat16>(int, int, int, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, cudaStream_t);

// Stub for unsupported types (float/double) - required for template instantiation
template void LaunchWmmaB2bGemm<float>(int, int, int, const float*, const float*, const float*, float*, cudaStream_t);
template void LaunchWmmaB2bGemm<double>(int, int, int, const double*, const double*, const double*, double*, cudaStream_t);

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
