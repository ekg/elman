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

// Stage 3: parallelized Hillis-Steele with ping-pong buffers.
//
// T_BLOCK warps per block, one per pair at level 0. log2(T_BLOCK) sync barriers.
// Ping-pong between two buffers means we skip the copy-back step entirely.
//
// Inclusive prefix scan: result[t] = combine(input[0], input[1], ..., input[t]).
// Hillis algorithm: at level d, output[t] = combine(input[t-d], input[t]) for t>=d.

template<int T_BLOCK, int M_DIM>
__global__ void intra_block_hillis_steele(
    const float* __restrict__ M_in,
    float* __restrict__ delta_out,
    const int B, const int H, const int N_row, const int T_full,
    const int N, const int num_blocks_t
) {
    const int pid = blockIdx.x;
    const int t_block = pid % num_blocks_t;
    const int row = (pid / num_blocks_t) % N_row;
    const int h = (pid / num_blocks_t / N_row) % H;
    const int b = pid / num_blocks_t / N_row / H;

    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x & 31;

    // Two ping-pong buffers. Pointers swap each level.
    extern __shared__ float smem[];
    float* buf_A = smem;
    float* buf_B = &smem[T_BLOCK * M_DIM * M_DIM];

    // Offsets
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

    // Cooperative load into buf_A.
    const int total_elems = T_BLOCK * M_DIM * M_DIM;
    for (int idx = threadIdx.x; idx < total_elems; idx += blockDim.x) {
        buf_A[idx] = M_in[M_full_base + idx];
    }
    __syncthreads();

    float* src = buf_A;
    float* dst = buf_B;

    for (int d = 1; d < T_BLOCK; d *= 2) {
        // For t < d: output[t] = input[t] (unchanged). Copy src -> dst.
        // For t >= d: output[t] = combine(input[t-d], input[t]).

        const int t_this = warp_id + d;
        const bool active_combine = (t_this < T_BLOCK);

        // Warp warp_id < d handles copying src[warp_id] -> dst[warp_id] (pass-through).
        // Warp warp_id >= d handles combine for t = warp_id + d... wait that's wrong.
        // Let me re-index: warp_id directly maps to the OUTPUT position.
        // Output pos = warp_id. For warp_id < d: dst[warp_id] = src[warp_id].
        //              For warp_id >= d: dst[warp_id] = combine(src[warp_id - d], src[warp_id]).
        const int t_out = warp_id;
        if (t_out < T_BLOCK) {
            if (t_out < d) {
                // Copy src -> dst at position t_out
                const int base = t_out * M_DIM * M_DIM;
                for (int idx = lane; idx < M_DIM * M_DIM; idx += 32) {
                    dst[base + idx] = src[base + idx];
                }
            } else {
                const float* M_later   = &src[t_out * M_DIM * M_DIM];
                const float* M_earlier = &src[(t_out - d) * M_DIM * M_DIM];
                float* M_out = &dst[t_out * M_DIM * M_DIM];

                const int total_out = M_DIM * M_DIM;
                for (int idx = lane; idx < total_out; idx += 32) {
                    const int rr = idx / M_DIM;
                    const int cc = idx % M_DIM;
                    float acc = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < M_DIM; k++) {
                        acc += M_later[rr * M_DIM + k] * M_earlier[k * M_DIM + cc];
                    }
                    M_out[idx] = acc;
                }
            }
        }
        __syncthreads();

        // Swap src <-> dst
        float* tmp = src; src = dst; dst = tmp;
    }

    // After the loop, src holds the final inclusive prefix.
    // Extract δ[t, i] = src[t, i, N] for i < N.
    const int tot_out = T_BLOCK * N;
    for (int idx = threadIdx.x; idx < tot_out; idx += blockDim.x) {
        const int t = idx / N;
        const int i = idx % N;
        delta_out[O_base + (long)t * N + i] = src[t * M_DIM * M_DIM + i * M_DIM + N];
    }
}

// -----------------------------------------------------------------------------
// Stage 4: Intra-block scan with explicit INITIAL STATE input.
// This is the building block for hierarchical scan — with init=identity we
// get local prefixes; with init=block_prefix we get global prefixes.
//
// Produces BOTH:
//   - δ_out[t]: per-position δ in the block
//   - block_summary: the final cumulative state at end of block
//
// Host orchestration:
//   Pass 1: run with init=identity, collect block_summary per block.
//   Pass 2: scan block_summary across blocks (another Hillis call).
//   Pass 3: re-run with init=block_cum_excl[b], output δ.
// -----------------------------------------------------------------------------

template<int T_BLOCK, int M_DIM>
__global__ void intra_block_with_init(
    const float* __restrict__ M_in,            // input per-position operators
    const float* __restrict__ init_state,      // per-block initial state [num_blocks, M_DIM, M_DIM]
    float* __restrict__ delta_out,             // per-position δ output [..., N]
    float* __restrict__ block_summary_out,     // final cumulative state [num_blocks, M_DIM, M_DIM]
    const int B, const int H, const int N_row, const int T_full,
    const int N, const int num_blocks_t,
    const bool have_init                       // if false, init = identity
) {
    const int pid = blockIdx.x;
    const int t_block = pid % num_blocks_t;
    const int row = (pid / num_blocks_t) % N_row;
    const int h = (pid / num_blocks_t / N_row) % H;
    const int b = pid / num_blocks_t / N_row / H;

    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x & 31;

    extern __shared__ float smem[];
    float* buf_A = smem;
    float* buf_B = &smem[T_BLOCK * M_DIM * M_DIM];

    const long M_full_bh_stride = (long)H * N_row * T_full * M_DIM * M_DIM;
    const long M_full_h_stride  = (long)N_row * T_full * M_DIM * M_DIM;
    const long M_full_r_stride  = (long)T_full * M_DIM * M_DIM;
    const long t_block_start = (long)t_block * T_BLOCK;
    const long M_full_base = (long)b * M_full_bh_stride + (long)h * M_full_h_stride +
                             (long)row * M_full_r_stride + t_block_start * M_DIM * M_DIM;

    const long summary_per_chain = (long)num_blocks_t * M_DIM * M_DIM;
    const long summary_bh_stride = (long)H * N_row * summary_per_chain;
    const long summary_h_stride  = (long)N_row * summary_per_chain;
    const long summary_r_stride  = summary_per_chain;
    const long summary_base = (long)b * summary_bh_stride + (long)h * summary_h_stride +
                              (long)row * summary_r_stride + (long)t_block * M_DIM * M_DIM;

    const long O_bh_stride = (long)H * N_row * T_full * N;
    const long O_h_stride  = (long)N_row * T_full * N;
    const long O_r_stride  = (long)T_full * N;
    const long O_base = (long)b * O_bh_stride + (long)h * O_h_stride + (long)row * O_r_stride
                        + t_block_start * N;

    // Load T_BLOCK matrices.
    const int total_elems = T_BLOCK * M_DIM * M_DIM;
    for (int idx = threadIdx.x; idx < total_elems; idx += blockDim.x) {
        buf_A[idx] = M_in[M_full_base + idx];
    }
    __syncthreads();

    // If init_state is given, "compose" it with position 0: buf_A[0] = buf_A[0] @ init
    // Everything else is as before.
    if (have_init) {
        // Need scratch for the composition. Use a small portion of buf_B.
        // Compute comp = buf_A[0] @ init, store in buf_B[0]
        const float* init_M = &init_state[summary_base];
        const float* later  = &buf_A[0];
        float* out = &buf_B[0];
        const int total_out = M_DIM * M_DIM;
        for (int idx = threadIdx.x; idx < total_out; idx += blockDim.x) {
            const int rr = idx / M_DIM;
            const int cc = idx % M_DIM;
            float acc = 0.0f;
            #pragma unroll
            for (int k = 0; k < M_DIM; k++) {
                acc += later[rr * M_DIM + k] * init_M[k * M_DIM + cc];
            }
            out[idx] = acc;
        }
        __syncthreads();
        // Copy buf_B[0] back to buf_A[0]
        for (int idx = threadIdx.x; idx < total_out; idx += blockDim.x) {
            buf_A[idx] = buf_B[idx];
        }
        __syncthreads();
    }

    float* src = buf_A;
    float* dst = buf_B;

    for (int d = 1; d < T_BLOCK; d *= 2) {
        const int t_out = warp_id;
        if (t_out < T_BLOCK) {
            if (t_out < d) {
                const int base = t_out * M_DIM * M_DIM;
                for (int idx = lane; idx < M_DIM * M_DIM; idx += 32) {
                    dst[base + idx] = src[base + idx];
                }
            } else {
                const float* M_later   = &src[t_out * M_DIM * M_DIM];
                const float* M_earlier = &src[(t_out - d) * M_DIM * M_DIM];
                float* M_out = &dst[t_out * M_DIM * M_DIM];
                const int total_out = M_DIM * M_DIM;
                for (int idx = lane; idx < total_out; idx += 32) {
                    const int rr = idx / M_DIM;
                    const int cc = idx % M_DIM;
                    float acc = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < M_DIM; k++) {
                        acc += M_later[rr * M_DIM + k] * M_earlier[k * M_DIM + cc];
                    }
                    M_out[idx] = acc;
                }
            }
        }
        __syncthreads();
        float* tmp = src; src = dst; dst = tmp;
    }

    // Store block summary: the LAST position's cumulative state.
    if (block_summary_out != nullptr) {
        const float* last = &src[(T_BLOCK - 1) * M_DIM * M_DIM];
        const int total = M_DIM * M_DIM;
        for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
            block_summary_out[summary_base + idx] = last[idx];
        }
    }

    // Store δ per-position.
    if (delta_out != nullptr) {
        const int tot_out = T_BLOCK * N;
        for (int idx = threadIdx.x; idx < tot_out; idx += blockDim.x) {
            const int t = idx / N;
            const int i = idx % N;
            delta_out[O_base + (long)t * N + i] = src[t * M_DIM * M_DIM + i * M_DIM + N];
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
    // Thread count: T_BLOCK warps — one per pair at level 0 — each warp 32 threads.
    const int threads = 32 * T_BLOCK;

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

// Stage 4 launcher: intra-block with init + summary output.
std::vector<torch::Tensor> intra_block_with_init_py(
    torch::Tensor M,
    torch::Tensor init_state,   // [B, H, N_row, num_blocks_t, M_DIM, M_DIM] or empty
    int N,
    int T_BLOCK
) {
    TORCH_CHECK(M.is_cuda() && M.is_contiguous(), "M must be contiguous CUDA fp32");
    TORCH_CHECK(M.dtype() == torch::kFloat32, "M must be fp32");
    TORCH_CHECK(M.dim() == 6, "M must be 6D");

    const int B = M.size(0);
    const int H = M.size(1);
    const int N_row = M.size(2);
    const int T = M.size(3);
    const int M_DIM = M.size(4);
    TORCH_CHECK(T % T_BLOCK == 0, "T must be multiple of T_BLOCK");

    const int num_blocks_t = T / T_BLOCK;
    auto delta = torch::zeros({B, H, N_row, T, N},
                              torch::dtype(torch::kFloat32).device(M.device()));
    auto summary = torch::zeros({B, H, N_row, num_blocks_t, M_DIM, M_DIM},
                                 torch::dtype(torch::kFloat32).device(M.device()));

    const bool have_init = (init_state.numel() > 0);
    const float* init_ptr = have_init ? init_state.data_ptr<float>() : nullptr;

    const int grid = B * H * N_row * num_blocks_t;
    const int threads = 32 * T_BLOCK;
    const size_t shmem = 2ull * T_BLOCK * M_DIM * M_DIM * sizeof(float);

#define DISPATCH2(TB, MD) do { \
    auto kfn = intra_block_with_init<TB, MD>; \
    cudaError_t attr_err = cudaFuncSetAttribute( \
        kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, 99328); \
    if (attr_err != cudaSuccess) cudaGetLastError(); \
    kfn<<<grid, threads, shmem>>>( \
        M.data_ptr<float>(), init_ptr, \
        delta.data_ptr<float>(), summary.data_ptr<float>(), \
        B, H, N_row, T, N, num_blocks_t, have_init); \
} while (0)

    if (T_BLOCK == 16 && M_DIM == 24) DISPATCH2(16, 24);
    else if (T_BLOCK == 8  && M_DIM == 24) DISPATCH2(8, 24);
    else if (T_BLOCK == 8  && M_DIM == 32) DISPATCH2(8, 32);
    else if (T_BLOCK == 8  && M_DIM == 40) DISPATCH2(8, 40);
    else if (T_BLOCK == 4  && M_DIM == 40) DISPATCH2(4, 40);
    else if (T_BLOCK == 4  && M_DIM == 32) DISPATCH2(4, 32);
    else if (T_BLOCK == 4  && M_DIM == 24) DISPATCH2(4, 24);
    else TORCH_CHECK(false, "Unsupported (T_BLOCK, M_DIM) combination: ", T_BLOCK, " ", M_DIM);

#undef DISPATCH2

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "kernel failed: ", cudaGetErrorString(err));

    return {delta, summary};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("intra_block_sequential", &intra_block_sequential,
          "Sequential scan within block using shared memory.");
    m.def("intra_block_hillis", &intra_block_hillis,
          "Hillis-Steele parallel scan within block.");
    m.def("intra_block_with_init", &intra_block_with_init_py,
          "Hillis-Steele within block with explicit initial state + "
          "output block summary.");
}
