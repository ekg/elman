/**
 * Brent-Kung tree scan for Newton affine-op composition — prototype.
 *
 * Stage 1 (this file): sequential-within-block CUDA kernel with shared
 * memory. Verifies the augmented-matrix math and CUDA build.
 *
 * Stage 2 (next): Brent-Kung parallel scan inside block (still one block
 * per chain), log(T_block) depth.
 *
 * Stage 3: WMMA tensor cores for the N×N matmul combine.
 *
 * Stage 4: Hierarchical (inter-block scan for full T).
 *
 * Each block handles one (b, h, row) chain. T positions inside block.
 * Augmented matrix: M = [[A, b], [0^T, 1]] packed into M_DIM × M_DIM.
 * Combine: M_new = M_right @ M_left  (matrix mult).
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>

namespace {

// -----------------------------------------------------------------------------
// Sequential-within-block CUDA scan, shared-memory cumulative state.
// -----------------------------------------------------------------------------

__global__ void intra_block_scan_sequential(
    const float* __restrict__ M_in,     // [B, H, N_row, T, M_DIM, M_DIM]
    float* __restrict__ delta_out,      // [B, H, N_row, T, N]
    const int B, const int H, const int N_row, const int T,
    const int N, const int M_DIM
) {
    const int pid = blockIdx.x;
    const int row = pid % N_row;
    const int h = (pid / N_row) % H;
    const int b = pid / (H * N_row);

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    // Shared memory: two M_DIM × M_DIM tiles (ping-pong)
    extern __shared__ float smem[];
    float* A_cum = smem;                       // cumulative prefix matrix
    float* A_new = &smem[M_DIM * M_DIM];        // scratch for matmul output

    // Initialize A_cum = identity
    for (int i = tid; i < M_DIM * M_DIM; i += nthreads) {
        const int r = i / M_DIM;
        const int c = i % M_DIM;
        A_cum[i] = (r == c) ? 1.0f : 0.0f;
    }
    __syncthreads();

    // Base offsets for this (b, h, row)
    const long M_BH_stride = (long)H * N_row * T * M_DIM * M_DIM;
    const long M_H_stride  = (long)N_row * T * M_DIM * M_DIM;
    const long M_R_stride  = (long)T * M_DIM * M_DIM;
    const long M_T_stride  = (long)M_DIM * M_DIM;
    const long M_base = (long)b * M_BH_stride + (long)h * M_H_stride + (long)row * M_R_stride;

    const long O_BH_stride = (long)H * N_row * T * N;
    const long O_H_stride  = (long)N_row * T * N;
    const long O_R_stride  = (long)T * N;
    const long O_T_stride  = (long)N;
    const long O_base = (long)b * O_BH_stride + (long)h * O_H_stride + (long)row * O_R_stride;

    // Sequential scan
    for (int t = 0; t < T; t++) {
        const float* M_t = M_in + M_base + (long)t * M_T_stride;

        // Compute A_new = M_t @ A_cum   (M_t is "later", A_cum is "earlier")
        // Each thread computes a subset of the M_DIM × M_DIM result.
        const int total = M_DIM * M_DIM;
        for (int idx = tid; idx < total; idx += nthreads) {
            const int rr = idx / M_DIM;
            const int cc = idx % M_DIM;
            float acc = 0.0f;
            #pragma unroll 16
            for (int k = 0; k < M_DIM; k++) {
                acc += M_t[rr * M_DIM + k] * A_cum[k * M_DIM + cc];
            }
            A_new[idx] = acc;
        }
        __syncthreads();

        // Swap A_cum <- A_new
        for (int i = tid; i < total; i += nthreads) {
            A_cum[i] = A_new[i];
        }
        __syncthreads();

        // Extract δ[t] = A_cum[:N, N]  (column N, first N rows = b portion)
        if (tid < N) {
            delta_out[O_base + (long)t * O_T_stride + tid] = A_cum[tid * M_DIM + N];
        }
    }
}

// -----------------------------------------------------------------------------
// Stage 2: Hillis-Steele parallel prefix scan within block.
// -----------------------------------------------------------------------------
//
// T_BLOCK positions scanned in log(T_BLOCK) depth.
// Each position has an (M_DIM × M_DIM) augmented matrix in shared memory.
// Threads cooperate on matmul of two matrices per level, stride d.
//
// Memory: T_BLOCK × M_DIM² floats in shared memory. For T_BLOCK=8, M_DIM=24:
//   8 × 576 × 4 = 18 KB. Fits.
// For T_BLOCK=16, M_DIM=24: 36 KB. Fits.
// For T_BLOCK=8, M_DIM=32: 32 KB. Fits (for N=32 case).
//
// Threads: 1 warp per block. Each thread handles one ROW of the matmul output.
// For M_DIM ≤ 32, 32 threads suffice.

template<int T_BLOCK, int M_DIM>
__global__ void intra_block_hillis_steele(
    const float* __restrict__ M_in,     // [B, H, N_row, num_blocks, T_BLOCK, M_DIM, M_DIM]
    float* __restrict__ delta_out,      // [B, H, N_row, num_blocks * T_BLOCK, N]
    const int B, const int H, const int N_row, const int T_full,
    const int N, const int num_blocks_t
) {
    // Each block handles one (b, h, row, t_block) combination.
    // Grid: B * H * N_row * num_blocks_t
    const int pid = blockIdx.x;
    const int t_block = pid % num_blocks_t;
    const int row = (pid / num_blocks_t) % N_row;
    const int h = (pid / num_blocks_t / N_row) % H;
    const int b = pid / num_blocks_t / N_row / H;

    const int tid = threadIdx.x;

    // Shared memory: two ping-pong buffers for [T_BLOCK, M_DIM, M_DIM]
    extern __shared__ float smem[];
    float* cur = smem;                                       // current scan state
    float* scratch = &smem[T_BLOCK * M_DIM * M_DIM];         // result buffer

    // Base offsets
    const long M_full_bh_stride = (long)H * N_row * T_full * M_DIM * M_DIM;
    const long M_full_h_stride  = (long)N_row * T_full * M_DIM * M_DIM;
    const long M_full_r_stride  = (long)T_full * M_DIM * M_DIM;
    const long t_block_start = (long)t_block * T_BLOCK;
    const long M_full_base = (long)b * M_full_bh_stride + (long)h * M_full_h_stride +
                             (long)row * M_full_r_stride + t_block_start * M_DIM * M_DIM;

    const long O_bh_stride = (long)H * N_row * T_full * N;
    const long O_h_stride  = (long)N_row * T_full * N;
    const long O_r_stride  = (long)T_full * N;
    const long O_base = (long)b * O_bh_stride + (long)h * O_h_stride + (long)row * O_r_stride
                        + t_block_start * N;

    // Load T_BLOCK matrices into shared memory. Each thread loads across multiple elems.
    const int total_elems = T_BLOCK * M_DIM * M_DIM;
    for (int idx = tid; idx < total_elems; idx += blockDim.x) {
        cur[idx] = M_in[M_full_base + idx];
    }
    __syncthreads();

    // Hillis-Steele scan: for d = 1, 2, 4, ..., T_BLOCK/2
    // at each level, for t >= d: cur[t] = cur[t] @ cur[t - d]
    // "later @ earlier" — matches our convention.

    for (int d = 1; d < T_BLOCK; d *= 2) {
        // For positions t in [d, T_BLOCK - 1], combine cur[t-d] and cur[t].
        // Each position's combine is (M_DIM × M_DIM) @ (M_DIM × M_DIM) = (M_DIM × M_DIM).
        // Distribute the work: each thread handles one row-column pair.
        // Total outputs per level: (T_BLOCK - d) positions × M_DIM² elements each.

        // For simplicity: 32-thread warp, each thread handles M_DIM outputs per position.
        // Iterate positions t one at a time (reduces shared mem pressure).
        for (int t = d; t < T_BLOCK; t++) {
            // Compute cur_new[t] = cur[t] @ cur[t - d]
            // Output is M_DIM × M_DIM. Each thread handles one row.
            const float* M_later   = &cur[t * M_DIM * M_DIM];
            const float* M_earlier = &cur[(t - d) * M_DIM * M_DIM];
            float* M_out = &scratch[t * M_DIM * M_DIM];

            for (int rr = tid; rr < M_DIM; rr += blockDim.x) {
                for (int cc = 0; cc < M_DIM; cc++) {
                    float acc = 0.0f;
                    #pragma unroll 16
                    for (int k = 0; k < M_DIM; k++) {
                        acc += M_later[rr * M_DIM + k] * M_earlier[k * M_DIM + cc];
                    }
                    M_out[rr * M_DIM + cc] = acc;
                }
            }
        }
        __syncthreads();

        // Copy scratch back to cur for combined positions
        for (int t = d; t < T_BLOCK; t++) {
            const int base = t * M_DIM * M_DIM;
            for (int idx = tid; idx < M_DIM * M_DIM; idx += blockDim.x) {
                cur[base + idx] = scratch[base + idx];
            }
        }
        __syncthreads();
    }

    // Extract δ[t] from each cur[t]: column N, rows 0..N-1.
    // Write to global output.
    for (int t = 0; t < T_BLOCK; t++) {
        if (tid < N) {
            delta_out[O_base + (long)t * N + tid] = cur[t * M_DIM * M_DIM + tid * M_DIM + N];
        }
    }
}

}  // anonymous namespace

// -----------------------------------------------------------------------------
// PyTorch bindings
// -----------------------------------------------------------------------------

torch::Tensor intra_block_sequential(
    torch::Tensor M,           // [B, H, N_row, T, M_DIM, M_DIM] fp32
    int N                      // effective N (the A part of augmented matrix)
) {
    TORCH_CHECK(M.is_cuda() && M.is_contiguous(), "M must be contiguous CUDA tensor");
    TORCH_CHECK(M.dtype() == torch::kFloat32, "M must be fp32");
    TORCH_CHECK(M.dim() == 6, "M must be 6D");

    const int B = M.size(0);
    const int H = M.size(1);
    const int N_row = M.size(2);
    const int T = M.size(3);
    const int M_DIM = M.size(4);
    TORCH_CHECK(M.size(5) == M_DIM, "M last two dims must be equal");
    TORCH_CHECK(N < M_DIM, "N must be < M_DIM");

    auto delta = torch::zeros({B, H, N_row, T, N},
                              torch::dtype(torch::kFloat32).device(M.device()));

    const int num_blocks = B * H * N_row;
    const int num_threads = 256;
    const size_t shmem = 2 * M_DIM * M_DIM * sizeof(float);

    intra_block_scan_sequential<<<num_blocks, num_threads, shmem>>>(
        M.data_ptr<float>(), delta.data_ptr<float>(),
        B, H, N_row, T, N, M_DIM
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    return delta;
}

// -----------------------------------------------------------------------------
// Stage 2 launcher: Hillis-Steele with T_BLOCK templated.
// -----------------------------------------------------------------------------

torch::Tensor intra_block_hillis(
    torch::Tensor M,           // [B, H, N_row, T, M_DIM, M_DIM] fp32
    int N,                     // effective N
    int T_BLOCK                // block length for intra-block scan
) {
    TORCH_CHECK(M.is_cuda() && M.is_contiguous(), "M must be contiguous CUDA fp32");
    TORCH_CHECK(M.dtype() == torch::kFloat32, "M must be fp32");
    TORCH_CHECK(M.dim() == 6, "M must be 6D");

    const int B = M.size(0);
    const int H = M.size(1);
    const int N_row = M.size(2);
    const int T = M.size(3);
    const int M_DIM = M.size(4);
    TORCH_CHECK(M.size(5) == M_DIM);
    TORCH_CHECK(T % T_BLOCK == 0, "T must be multiple of T_BLOCK");

    const int num_blocks_t = T / T_BLOCK;
    auto delta = torch::zeros({B, H, N_row, T, N},
                              torch::dtype(torch::kFloat32).device(M.device()));

    const int grid = B * H * N_row * num_blocks_t;
    const int threads = 32;  // one warp

    // Shared mem: 2 × T_BLOCK × M_DIM²
    const size_t shmem = 2ull * T_BLOCK * M_DIM * M_DIM * sizeof(float);

    // Opt in to larger dynamic shared memory.
    // A100 (SM80) supports up to 164KB per block with opt-in.
#define DISPATCH(TB, MD) do { \
    auto kfn = intra_block_hillis_steele<TB, MD>; \
    /* A100 SM80 supports up to 99328 bytes per block with opt-in (100KB-ish).
     * H100 SM90 supports up to 228KB. Use 99KB to stay safe on SM80. */ \
    cudaError_t attr_err = cudaFuncSetAttribute( \
        kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, 99328); \
    if (attr_err != cudaSuccess) { \
        /* Not fatal — might already be sufficient */ \
        cudaGetLastError(); /* clear error */ \
    } \
    kfn<<<grid, threads, shmem>>>( \
        M.data_ptr<float>(), delta.data_ptr<float>(), \
        B, H, N_row, T, N, num_blocks_t); \
} while (0)

    if (T_BLOCK == 8 && M_DIM == 32)  DISPATCH(8, 32);
    else if (T_BLOCK == 8 && M_DIM == 24)  DISPATCH(8, 24);
    else if (T_BLOCK == 8 && M_DIM == 40)  DISPATCH(8, 40);
    else if (T_BLOCK == 16 && M_DIM == 24) DISPATCH(16, 24);
    else if (T_BLOCK == 16 && M_DIM == 32) DISPATCH(16, 32);
    else if (T_BLOCK == 16 && M_DIM == 40) DISPATCH(16, 40);
    else if (T_BLOCK == 4 && M_DIM == 32)  DISPATCH(4, 32);
    else if (T_BLOCK == 4 && M_DIM == 24)  DISPATCH(4, 24);
    else if (T_BLOCK == 4 && M_DIM == 40)  DISPATCH(4, 40);
    else TORCH_CHECK(false, "Unsupported (T_BLOCK, M_DIM) combination.");

#undef DISPATCH

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA hillis kernel failed: ", cudaGetErrorString(err));
    return delta;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("intra_block_sequential", &intra_block_sequential,
          "Sequential scan within block using shared memory.");
    m.def("intra_block_hillis", &intra_block_hillis,
          "Hillis-Steele parallel scan within block.");
}
