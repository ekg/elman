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
#include <mma.h>
#include <cstdio>
#include <cstdint>

using namespace nvcuda;

namespace {

// -----------------------------------------------------------------------------
// Stage 6: WMMA tensor core matmul helper for 32×32 matmul.
//
// Assumes A and B are 32×32 matrices in shared memory, row-major, fp32.
// Output C (32×32) is accumulated in fp32.
// Uses bf16 WMMA (m16n16k16) for 32×32 as 2×2 output tiles × 2 K-steps = 8 mmas.
//
// Each warp does the full 32×32 × 32×32 matmul.
// A_smem, B_smem, C_smem are 32×32 tiles in shared memory.
// -----------------------------------------------------------------------------

__device__ __forceinline__ void wmma_matmul_32x32_bf16(
    const float* __restrict__ A_smem,
    const float* __restrict__ B_smem,
    float* __restrict__ C_smem
) {
    // First convert A, B fp32 → bf16 in local buffer (in shared mem we'd need
    // extra shared buffer). For simplicity, convert on load via direct
    // fragment init from fp32 buffer.
    //
    // Ampere WMMA: load_matrix_sync converts fp32 → bf16 if fragment is bf16.
    // Actually no — load_matrix_sync expects matching type. Must convert first.
    //
    // Approach: use TF32 fragments (m16n16k8) which accept fp32 directly.

    // Cast A_smem, B_smem via TF32 fragments
    using FragA = wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major>;
    using FragB = wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major>;
    using FragC = wmma::fragment<wmma::accumulator, 16, 16, 8, float>;

    // 2×2 output tiles, each 16×16. K dim = 32, so 4 K-steps of 8 each.
    // Total: 4 output tiles × 4 K-steps = 16 mma calls.
    // (For bf16 with k=16 it'd be 4×2=8, but bf16 needs conversion.)

    FragC acc00, acc01, acc10, acc11;
    wmma::fill_fragment(acc00, 0.0f);
    wmma::fill_fragment(acc01, 0.0f);
    wmma::fill_fragment(acc10, 0.0f);
    wmma::fill_fragment(acc11, 0.0f);

    // K-steps over the 32-dim contraction axis, stride 8 (TF32 k=8).
    #pragma unroll
    for (int k0 = 0; k0 < 32; k0 += 8) {
        FragA a0, a1;
        FragB b0, b1;
        // A rows 0..15 and 16..31 at K=[k0, k0+8)
        wmma::load_matrix_sync(a0, A_smem + 0 * 32 + k0, 32);
        wmma::load_matrix_sync(a1, A_smem + 16 * 32 + k0, 32);
        // B rows [k0, k0+8), cols 0..15 and 16..31
        wmma::load_matrix_sync(b0, B_smem + k0 * 32 + 0, 32);
        wmma::load_matrix_sync(b1, B_smem + k0 * 32 + 16, 32);

        // Round-to-nearest conversion fp32 → TF32 happens inside WMMA.
        // Convert each fragment element explicitly (required before mma_sync).
        #pragma unroll
        for (int i = 0; i < a0.num_elements; i++) {
            a0.x[i] = wmma::__float_to_tf32(a0.x[i]);
            a1.x[i] = wmma::__float_to_tf32(a1.x[i]);
        }
        #pragma unroll
        for (int i = 0; i < b0.num_elements; i++) {
            b0.x[i] = wmma::__float_to_tf32(b0.x[i]);
            b1.x[i] = wmma::__float_to_tf32(b1.x[i]);
        }

        wmma::mma_sync(acc00, a0, b0, acc00);
        wmma::mma_sync(acc01, a0, b1, acc01);
        wmma::mma_sync(acc10, a1, b0, acc10);
        wmma::mma_sync(acc11, a1, b1, acc11);
    }

    // Store output 2×2 tiles back to shared memory (row-major, stride 32)
    wmma::store_matrix_sync(C_smem + 0 * 32 + 0,   acc00, 32, wmma::mem_row_major);
    wmma::store_matrix_sync(C_smem + 0 * 32 + 16,  acc01, 32, wmma::mem_row_major);
    wmma::store_matrix_sync(C_smem + 16 * 32 + 0,  acc10, 32, wmma::mem_row_major);
    wmma::store_matrix_sync(C_smem + 16 * 32 + 16, acc11, 32, wmma::mem_row_major);
}

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

// -----------------------------------------------------------------------------
// Stage 7: fused-build intra-block scan.
//
// Avoids the huge [B, H, T, N_row, N, N] materialized A tensor by building
// each position's augmented matrix IN SHARED MEMORY from raw inputs:
//   S_prev (from S0 for t=0, else S_var[t-1]), K, V, decay.
//
// Same math, same output, but no PyTorch overhead.
//
// Inputs:
//   S0: [B, H, N, N]                  initial state
//   S_var: [B, H, T, N, N]            current Newton iterate (for S_prev[t] = S_var[t-1])
//   K, V: [B, H, T, N]                projections
//   decay: [B, H, T]                  scalar per position
// Output:
//   delta: [B, H, T, N_row, N]        per-position δ = -(J^{-1} r)
//
// For each (b, h, row, t): build A_t = diag(D_t[row]) - u_t[row] v_t^T
// on-the-fly into shared-mem, combined with scan.
// -----------------------------------------------------------------------------

template<int T_BLOCK, int M_DIM, int N>
__global__ void intra_block_fused_build(
    const float* __restrict__ S0,       // [B, H, N, N]
    const float* __restrict__ S_var,    // [B, H, T, N, N]
    const float* __restrict__ K_ptr,    // [B, H, T, N]
    const float* __restrict__ V_ptr,    // [B, H, T, N]
    const float* __restrict__ decay_ptr,// [B, H, T]
    const float* __restrict__ init_state, // [B, H, N_row, num_blocks_t, M_DIM, M_DIM] or null
    float* __restrict__ delta_out,      // [B, H, N_row, T, N]
    float* __restrict__ summary_out,    // [B, H, N_row, num_blocks_t, M_DIM, M_DIM] or null
    const int B, const int H, const int T,
    const int num_blocks_t,
    const bool have_init
) {
    // Grid: B * H * N_row * num_blocks_t
    // blockDim.x = 32 * T_BLOCK (one warp per scan position)
    const int pid = blockIdx.x;
    const int t_block = pid % num_blocks_t;
    const int row = (pid / num_blocks_t) % N;      // row within head
    const int h = (pid / num_blocks_t / N) % H;
    const int b = pid / num_blocks_t / N / H;

    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x & 31;

    extern __shared__ float smem[];
    float* buf_A = smem;
    float* buf_B = &smem[T_BLOCK * M_DIM * M_DIM];

    // Offsets for raw inputs
    const long S0_base = ((long)b * H + h) * N * N + (long)row * N;
    const long Sv_bh_stride = (long)H * T * N * N;
    const long Sv_h_stride  = (long)T * N * N;
    const long Sv_t_stride  = (long)N * N;
    const long Sv_base_bh = (long)b * Sv_bh_stride + (long)h * Sv_h_stride;

    const long K_bh_stride = (long)H * T * N;
    const long K_h_stride  = (long)T * N;
    const long K_t_stride  = N;
    const long K_base_bh = (long)b * K_bh_stride + (long)h * K_h_stride;

    const long dec_bh_stride = (long)H * T;
    const long dec_h_stride  = (long)T;
    const long dec_base_bh = (long)b * dec_bh_stride + (long)h * dec_h_stride;

    const long t_block_start = (long)t_block * T_BLOCK;

    const long O_base = ((((long)b * H + h) * N + row) * T + t_block_start) * N;

    const long summary_per_chain = (long)num_blocks_t * M_DIM * M_DIM;
    const long summary_base = (((long)b * H + h) * N + row) * summary_per_chain
                              + (long)t_block * M_DIM * M_DIM;

    // Step A: each warp builds one position's augmented matrix M[t] into buf_A.
    // M[t] = [[A[t], b[t]], [0, 1], [0, 0, I_pad]]
    // A[t, i, j] = delta_ij * D[t, row] - u[t, row] * K[t, j]   (row is fixed per program)
    // Wait: A is N×N matrix, but we're on row=row. So actually A is A[t] for that row:
    //   A[t] = diag(D[t, row, :]) - u[t, row, :] * v[t, :]^T
    // where D, u, v are N-dim (per row i=row, these have N entries).
    //
    // Per-position build:
    //   pre[t, :] = decay[t] * S_prev[t, :] + (V[t, row] - K[t] · S_prev[t, :]) * K[t]
    //     where S_prev[t, :] = S_var[t-1, row, :] or S0[row, :] if t==0
    //   tanh_deriv[t, :] = 1 - tanh(pre[t, :])^2
    //   D[t, :] = decay[t] * tanh_deriv[t, :]
    //   u[t, :] = tanh_deriv[t, :] * K[t, :]
    //   b_vec[t, :] = -(S_var[t, row, :] - tanh(pre[t, :]))
    // Then M[t, i, j] = D[i]*delta_ij - u[i]*K[j]  for i,j < N
    //      M[t, i, N] = b_vec[i]  (column N holds b)
    //      M[t, N, N] = 1 (identity pivot)
    //      M[t, p, p] = 1 for p > N (padding identity)

    const int t_out = warp_id;
    const int t_glob = t_block_start + t_out;

    if (t_out < T_BLOCK && t_glob < T) {
        // Load S_prev[t, row, :] into a register array (N = M_DIM - 1 or similar)
        float S_prev[N];
        if (t_glob == 0) {
            // Load from S0[row, :]
            for (int j = lane; j < N; j += 32) {
                // Only lanes < N actually need the value, but we need all threads to have it.
                // Use shared memory or shuffle. Simpler: every lane loads.
            }
            for (int j = 0; j < N; j++) {
                S_prev[j] = S0[S0_base + j];
            }
        } else {
            long off = Sv_base_bh + (long)(t_glob - 1) * Sv_t_stride + (long)row * N;
            for (int j = 0; j < N; j++) {
                S_prev[j] = S_var[off + j];
            }
        }

        // Load K[t, :] — broadcast
        float K_vec[N];
        long K_off = K_base_bh + (long)t_glob * K_t_stride;
        for (int j = 0; j < N; j++) K_vec[j] = K_ptr[K_off + j];

        // V[t, row]
        float V_val = V_ptr[K_base_bh + (long)t_glob * K_t_stride + row];
        float dec = decay_ptr[dec_base_bh + t_glob];

        // retrieved = sum_j S_prev[j] * K_vec[j]
        float retrieved = 0.0f;
        #pragma unroll
        for (int j = 0; j < N; j++) retrieved += S_prev[j] * K_vec[j];

        float delta_scalar = V_val - retrieved;

        // pre[j] = dec * S_prev[j] + delta_scalar * K_vec[j]
        // f_val[j] = tanh(pre[j])
        // tanh_deriv[j] = 1 - f_val^2
        // D[j] = dec * tanh_deriv[j]
        // u[j] = tanh_deriv[j] * K_vec[j]
        // b_vec[j] = -(S_var[t, row, j] - f_val[j])

        float pre[N], f_val[N], tanh_deriv[N];
        #pragma unroll
        for (int j = 0; j < N; j++) {
            pre[j] = dec * S_prev[j] + delta_scalar * K_vec[j];
            float e2x = __expf(2.0f * pre[j]);
            f_val[j] = (e2x - 1.0f) / (e2x + 1.0f);
            tanh_deriv[j] = 1.0f - f_val[j] * f_val[j];
        }

        // Load S_var[t, row, j] to compute b_vec[j] = -(S_var - f_val)
        float S_var_row[N];
        long Sv_off = Sv_base_bh + (long)t_glob * Sv_t_stride + (long)row * N;
        for (int j = 0; j < N; j++) S_var_row[j] = S_var[Sv_off + j];

        // Write M[t] into buf_A[t_out]
        float* M_my = &buf_A[t_out * M_DIM * M_DIM];
        // Clear the whole M_DIM x M_DIM tile first (identity on padding diag)
        const int total = M_DIM * M_DIM;
        for (int idx = lane; idx < total; idx += 32) {
            const int r = idx / M_DIM;
            const int c = idx % M_DIM;
            float val;
            if (r < N && c < N) {
                // A[r, c] = D[r] * (r==c) - u[r] * K_vec[c]   (u[r] = tanh_deriv[r] * K_vec[r])
                float D_r = dec * tanh_deriv[r];
                float u_r = tanh_deriv[r] * K_vec[r];
                val = (r == c ? D_r : 0.0f) - u_r * K_vec[c];
            } else if (r < N && c == N) {
                // b column: b_vec[r] = f_val[r] - S_var[r]
                val = f_val[r] - S_var_row[r];
            } else if (r == c) {
                val = 1.0f;
            } else {
                val = 0.0f;
            }
            M_my[idx] = val;
        }
    }
    __syncthreads();

    // Step B: apply init_state at position 0 if provided
    if (have_init) {
        if (warp_id == 0) {
            float* my0 = &buf_A[0];
            const float* init_M = &init_state[summary_base];
            float* my0_out = &buf_B[0];
            const int total_out = M_DIM * M_DIM;
            for (int idx = lane; idx < total_out; idx += 32) {
                const int rr = idx / M_DIM;
                const int cc = idx % M_DIM;
                float acc = 0.0f;
                #pragma unroll
                for (int k = 0; k < M_DIM; k++) {
                    acc += my0[rr * M_DIM + k] * init_M[k * M_DIM + cc];
                }
                my0_out[idx] = acc;
            }
        }
        __syncthreads();
        if (warp_id == 0) {
            const int total = M_DIM * M_DIM;
            for (int idx = lane; idx < total; idx += 32) {
                buf_A[idx] = buf_B[idx];
            }
        }
        __syncthreads();
    }

    // Step C: Hillis-Steele tree scan (ping-pong)
    float* src = buf_A;
    float* dst = buf_B;

    for (int d = 1; d < T_BLOCK; d *= 2) {
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

    // Step D: write outputs
    if (summary_out != nullptr && t_out == T_BLOCK - 1) {
        const float* last = &src[(T_BLOCK - 1) * M_DIM * M_DIM];
        const int total = M_DIM * M_DIM;
        for (int idx = lane; idx < total; idx += 32) {
            summary_out[summary_base + idx] = last[idx];
        }
    }

    if (delta_out != nullptr && t_out < T_BLOCK && t_glob < T) {
        // Extract δ[t, row, i] from src[t, i, N] for i < N
        for (int i = lane; i < N; i += 32) {
            delta_out[O_base + (long)t_out * N + i] =
                src[t_out * M_DIM * M_DIM + i * M_DIM + N];
        }
    }
}

// -----------------------------------------------------------------------------
// Stage 6: WMMA-accelerated intra-block scan for M_DIM=32, N=16.
//
// Uses the wmma_matmul_32x32_bf16 (actually TF32) helper. Each warp handles
// one pair's matmul via WMMA instead of scalar FMA.
//
// T_BLOCK=8 is the largest that fits in 99KB shmem: 2 * 8 * 32² * 4 = 64KB.
// -----------------------------------------------------------------------------

template<int T_BLOCK>
__global__ void intra_block_fused_wmma(
    const float* __restrict__ S0,
    const float* __restrict__ S_var,
    const float* __restrict__ K_ptr,
    const float* __restrict__ V_ptr,
    const float* __restrict__ decay_ptr,
    const float* __restrict__ init_state,
    float* __restrict__ delta_out,
    float* __restrict__ summary_out,
    const int B, const int H, const int T,
    const int num_blocks_t,
    const bool have_init
) {
    constexpr int N = 16;
    constexpr int M_DIM = 32;
    const int pid = blockIdx.x;
    const int t_block = pid % num_blocks_t;
    const int row = (pid / num_blocks_t) % N;
    const int h = (pid / num_blocks_t / N) % H;
    const int b = pid / num_blocks_t / N / H;

    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x & 31;

    extern __shared__ float smem[];
    float* buf_A = smem;
    float* buf_B = &smem[T_BLOCK * M_DIM * M_DIM];

    const long S0_base = ((long)b * H + h) * N * N + (long)row * N;
    const long Sv_base_bh = ((long)b * H + h) * T * N * N;
    const long Sv_t_stride = (long)N * N;
    const long K_base_bh = ((long)b * H + h) * T * N;
    const long K_t_stride = N;
    const long dec_base_bh = ((long)b * H + h) * T;
    const long t_block_start = (long)t_block * T_BLOCK;

    const long O_base = ((((long)b * H + h) * N + row) * T + t_block_start) * N;
    const long summary_base = (((long)b * H + h) * N + row) * num_blocks_t * M_DIM * M_DIM
                              + (long)t_block * M_DIM * M_DIM;

    const int t_out = warp_id;
    const int t_glob = t_block_start + t_out;

    // Build augmented matrix for this warp's position
    if (t_out < T_BLOCK && t_glob < T) {
        float S_prev_reg[N];
        if (t_glob == 0) {
            for (int j = 0; j < N; j++) S_prev_reg[j] = S0[S0_base + j];
        } else {
            long off = Sv_base_bh + (long)(t_glob - 1) * Sv_t_stride + (long)row * N;
            for (int j = 0; j < N; j++) S_prev_reg[j] = S_var[off + j];
        }

        float K_vec[N];
        long K_off = K_base_bh + (long)t_glob * K_t_stride;
        for (int j = 0; j < N; j++) K_vec[j] = K_ptr[K_off + j];

        float V_val = V_ptr[K_base_bh + (long)t_glob * K_t_stride + row];
        float dec = decay_ptr[dec_base_bh + t_glob];

        float retrieved = 0.0f;
        #pragma unroll
        for (int j = 0; j < N; j++) retrieved += S_prev_reg[j] * K_vec[j];
        float delta_scalar = V_val - retrieved;

        float f_val[N], tanh_deriv[N];
        #pragma unroll
        for (int j = 0; j < N; j++) {
            float pre_j = dec * S_prev_reg[j] + delta_scalar * K_vec[j];
            float e2x = __expf(2.0f * pre_j);
            f_val[j] = (e2x - 1.0f) / (e2x + 1.0f);
            tanh_deriv[j] = 1.0f - f_val[j] * f_val[j];
        }

        float S_var_row[N];
        long Sv_off = Sv_base_bh + (long)t_glob * Sv_t_stride + (long)row * N;
        for (int j = 0; j < N; j++) S_var_row[j] = S_var[Sv_off + j];

        float* M_my = &buf_A[t_out * M_DIM * M_DIM];
        const int total = M_DIM * M_DIM;
        for (int idx = lane; idx < total; idx += 32) {
            const int r = idx / M_DIM;
            const int c = idx % M_DIM;
            float val;
            if (r < N && c < N) {
                float D_r = dec * tanh_deriv[r];
                float u_r = tanh_deriv[r] * K_vec[r];
                val = (r == c ? D_r : 0.0f) - u_r * K_vec[c];
            } else if (r < N && c == N) {
                val = f_val[r] - S_var_row[r];
            } else if (r == c) {
                val = 1.0f;
            } else {
                val = 0.0f;
            }
            M_my[idx] = val;
        }
    }
    __syncthreads();

    // Apply init state if provided
    if (have_init) {
        if (warp_id == 0) {
            // Use WMMA for init composition too.
            // C = buf_A[0] @ init_state (scan order: init first, so buf_A @ init)
            wmma_matmul_32x32_bf16(&buf_A[0], &init_state[summary_base], &buf_B[0]);
        }
        __syncthreads();
        if (warp_id == 0) {
            for (int idx = lane; idx < M_DIM * M_DIM; idx += 32) {
                buf_A[idx] = buf_B[idx];
            }
        }
        __syncthreads();
    }

    float* src = buf_A;
    float* dst = buf_B;

    // Hillis-Steele with WMMA combine
    for (int d = 1; d < T_BLOCK; d *= 2) {
        if (t_out < T_BLOCK) {
            if (t_out < d) {
                // Pass-through copy
                const int base = t_out * M_DIM * M_DIM;
                for (int idx = lane; idx < M_DIM * M_DIM; idx += 32) {
                    dst[base + idx] = src[base + idx];
                }
            } else {
                // WMMA matmul: dst[t_out] = src[t_out] @ src[t_out - d]
                wmma_matmul_32x32_bf16(
                    &src[t_out * M_DIM * M_DIM],
                    &src[(t_out - d) * M_DIM * M_DIM],
                    &dst[t_out * M_DIM * M_DIM]
                );
            }
        }
        __syncthreads();
        float* tmp = src; src = dst; dst = tmp;
    }

    // Write outputs
    if (summary_out != nullptr && t_out == T_BLOCK - 1) {
        const float* last = &src[(T_BLOCK - 1) * M_DIM * M_DIM];
        for (int idx = lane; idx < M_DIM * M_DIM; idx += 32) {
            summary_out[summary_base + idx] = last[idx];
        }
    }

    if (delta_out != nullptr && t_out < T_BLOCK && t_glob < T) {
        for (int i = lane; i < N; i += 32) {
            delta_out[O_base + (long)t_out * N + i] =
                src[t_out * M_DIM * M_DIM + i * M_DIM + N];
        }
    }
}

// -----------------------------------------------------------------------------
// Stage 5: single-level parallel combine kernel for inter-block scan.
//
// Given a chain of N matrices, one LEVEL of Hillis-Steele does:
//   for each position p: M_out[p] = (p >= d) ? M_in[p] @ M_in[p - d] : M_in[p]
//
// All positions combine in parallel. Host calls this log2(N) times with
// d=1, 2, 4, ... Each call is O(N) parallel matmuls.
//
// For block summary scan: input shape [B, H, N_row, num_blocks, M_DIM, M_DIM].
// Grid: B * H * N_row * num_blocks. Each block processes one matmul.
// Threads cooperate on the M_DIM × M_DIM result.
// -----------------------------------------------------------------------------

template<int M_DIM>
__global__ void scan_one_level(
    const float* __restrict__ M_in,
    float* __restrict__ M_out,
    const int num_pos,
    const int d,
    const long chain_stride  // = num_pos * M_DIM * M_DIM
) {
    const int pid = blockIdx.x;
    const int pos = pid % num_pos;
    const long chain_base = (long)(pid / num_pos) * chain_stride;

    const float* src_pos = &M_in[chain_base + (long)pos * M_DIM * M_DIM];
    float* dst_pos = &M_out[chain_base + (long)pos * M_DIM * M_DIM];

    if (pos < d) {
        // Pass through
        const int total = M_DIM * M_DIM;
        for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
            dst_pos[idx] = src_pos[idx];
        }
    } else {
        const float* src_left = &M_in[chain_base + (long)(pos - d) * M_DIM * M_DIM];
        // M_out[pos] = M_in[pos] @ M_in[pos - d]
        const int total = M_DIM * M_DIM;
        for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
            const int rr = idx / M_DIM;
            const int cc = idx % M_DIM;
            float acc = 0.0f;
            #pragma unroll
            for (int k = 0; k < M_DIM; k++) {
                acc += src_pos[rr * M_DIM + k] * src_left[k * M_DIM + cc];
            }
            dst_pos[idx] = acc;
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

// Stage 7 launcher: fused-build intra-block scan.
std::vector<torch::Tensor> intra_block_fused_py(
    torch::Tensor S0, torch::Tensor S_var,
    torch::Tensor K, torch::Tensor V, torch::Tensor decay,
    torch::Tensor init_state,
    int T_BLOCK
) {
    TORCH_CHECK(S_var.is_cuda() && S_var.is_contiguous(), "S_var must be CUDA fp32");
    TORCH_CHECK(S_var.dtype() == torch::kFloat32, "fp32 only for now");
    TORCH_CHECK(S_var.dim() == 5, "S_var shape [B, H, T, N, N]");

    const int B = S_var.size(0);
    const int H = S_var.size(1);
    const int T = S_var.size(2);
    const int N = S_var.size(3);
    TORCH_CHECK(S_var.size(4) == N);
    TORCH_CHECK(T % T_BLOCK == 0, "T must be multiple of T_BLOCK");

    const int num_blocks_t = T / T_BLOCK;
    const int M_DIM = ((N + 1 + 7) / 8) * 8;  // round up to multiple of 8

    auto delta = torch::zeros({B, H, N, T, N},
                              torch::dtype(torch::kFloat32).device(S_var.device()));
    auto summary = torch::zeros({B, H, N, num_blocks_t, M_DIM, M_DIM},
                                 torch::dtype(torch::kFloat32).device(S_var.device()));

    const bool have_init = (init_state.numel() > 0);
    const float* init_ptr = have_init ? init_state.data_ptr<float>() : nullptr;

    const int grid = B * H * N * num_blocks_t;
    const int threads = 32 * T_BLOCK;
    const size_t shmem = 2ull * T_BLOCK * M_DIM * M_DIM * sizeof(float);

#define DISPATCH3(TB, MD, NN) do { \
    auto kfn = intra_block_fused_build<TB, MD, NN>; \
    cudaError_t attr_err = cudaFuncSetAttribute( \
        kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, 99328); \
    if (attr_err != cudaSuccess) cudaGetLastError(); \
    kfn<<<grid, threads, shmem>>>( \
        S0.data_ptr<float>(), S_var.data_ptr<float>(), \
        K.data_ptr<float>(), V.data_ptr<float>(), decay.data_ptr<float>(), \
        init_ptr, delta.data_ptr<float>(), summary.data_ptr<float>(), \
        B, H, T, num_blocks_t, have_init); \
} while (0)

    if (T_BLOCK == 16 && N == 16) DISPATCH3(16, 24, 16);
    else if (T_BLOCK == 8 && N == 16) DISPATCH3(8, 24, 16);
    else if (T_BLOCK == 8 && N == 32) DISPATCH3(8, 40, 32);
    else if (T_BLOCK == 4 && N == 16) DISPATCH3(4, 24, 16);
    else if (T_BLOCK == 4 && N == 32) DISPATCH3(4, 40, 32);
    else TORCH_CHECK(false, "Unsupported (T_BLOCK, N) combination: ", T_BLOCK, " ", N);

#undef DISPATCH3

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "fused kernel failed: ", cudaGetErrorString(err));
    return {delta, summary};
}

// Stage 6 launcher: WMMA variant for N=16.
std::vector<torch::Tensor> intra_block_fused_wmma_py(
    torch::Tensor S0, torch::Tensor S_var,
    torch::Tensor K, torch::Tensor V, torch::Tensor decay,
    torch::Tensor init_state,
    int T_BLOCK
) {
    TORCH_CHECK(S_var.dtype() == torch::kFloat32);
    TORCH_CHECK(S_var.is_cuda() && S_var.is_contiguous());
    TORCH_CHECK(S_var.dim() == 5);

    const int B = S_var.size(0);
    const int H = S_var.size(1);
    const int T = S_var.size(2);
    const int N = S_var.size(3);
    TORCH_CHECK(N == 16, "WMMA variant: only N=16 for now");
    TORCH_CHECK(T % T_BLOCK == 0);
    const int M_DIM = 32;
    const int num_blocks_t = T / T_BLOCK;

    auto delta = torch::zeros({B, H, N, T, N},
                              torch::dtype(torch::kFloat32).device(S_var.device()));
    auto summary = torch::zeros({B, H, N, num_blocks_t, M_DIM, M_DIM},
                                 torch::dtype(torch::kFloat32).device(S_var.device()));

    const bool have_init = (init_state.numel() > 0);
    const float* init_ptr = have_init ? init_state.data_ptr<float>() : nullptr;

    const int grid = B * H * N * num_blocks_t;
    const int threads = 32 * T_BLOCK;
    const size_t shmem = 2ull * T_BLOCK * M_DIM * M_DIM * sizeof(float);

#define DISPATCH_WMMA(TB) do { \
    auto kfn = intra_block_fused_wmma<TB>; \
    cudaError_t attr_err = cudaFuncSetAttribute( \
        kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, 99328); \
    if (attr_err != cudaSuccess) cudaGetLastError(); \
    kfn<<<grid, threads, shmem>>>( \
        S0.data_ptr<float>(), S_var.data_ptr<float>(), \
        K.data_ptr<float>(), V.data_ptr<float>(), decay.data_ptr<float>(), \
        init_ptr, delta.data_ptr<float>(), summary.data_ptr<float>(), \
        B, H, T, num_blocks_t, have_init); \
} while (0)

    if (T_BLOCK == 8) DISPATCH_WMMA(8);
    else if (T_BLOCK == 4) DISPATCH_WMMA(4);
    else TORCH_CHECK(false, "WMMA: Unsupported T_BLOCK=", T_BLOCK, " (only 4, 8)");

#undef DISPATCH_WMMA

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "WMMA kernel failed: ", cudaGetErrorString(err));
    return {delta, summary};
}

// Stage 5 launcher: scan a chain of matrices in-place via repeated one-level calls.
torch::Tensor inclusive_matrix_prefix_scan(
    torch::Tensor M    // [chains..., num_pos, M_DIM, M_DIM]
) {
    TORCH_CHECK(M.is_cuda() && M.is_contiguous(), "M must be contiguous CUDA fp32");
    TORCH_CHECK(M.dtype() == torch::kFloat32, "M must be fp32");

    auto sizes = M.sizes().vec();
    TORCH_CHECK(sizes.size() >= 3, "Input must have at least 3 dims");
    const int M_DIM = sizes[sizes.size() - 1];
    const int M_DIM2 = sizes[sizes.size() - 2];
    TORCH_CHECK(M_DIM == M_DIM2, "Last two dims must be equal");
    const int num_pos = sizes[sizes.size() - 3];

    int num_chains = 1;
    for (size_t i = 0; i + 3 < sizes.size(); i++) num_chains *= sizes[i];

    const long chain_stride = (long)num_pos * M_DIM * M_DIM;
    const int grid = num_chains * num_pos;
    const int threads = 128;

    auto scratch = torch::empty_like(M);
    float* src = M.data_ptr<float>();
    float* dst = scratch.data_ptr<float>();

#define DISPATCH_LEVEL(MD) do { \
    scan_one_level<MD><<<grid, threads>>>(src, dst, num_pos, d, chain_stride); \
} while (0)

    for (int d = 1; d < num_pos; d *= 2) {
        if (M_DIM == 24) DISPATCH_LEVEL(24);
        else if (M_DIM == 32) DISPATCH_LEVEL(32);
        else if (M_DIM == 40) DISPATCH_LEVEL(40);
        else TORCH_CHECK(false, "Unsupported M_DIM: ", M_DIM);
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "level kernel failed: ", cudaGetErrorString(err));
        std::swap(src, dst);
    }
#undef DISPATCH_LEVEL

    // src now points to the result. Copy into M_out if it's scratch.
    if (src != M.data_ptr<float>()) {
        cudaMemcpyAsync(M.data_ptr<float>(), src,
                        (size_t)num_chains * chain_stride * sizeof(float),
                        cudaMemcpyDeviceToDevice);
    }

    return M;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("intra_block_sequential", &intra_block_sequential,
          "Sequential scan within block using shared memory.");
    m.def("intra_block_hillis", &intra_block_hillis,
          "Hillis-Steele parallel scan within block.");
    m.def("intra_block_with_init", &intra_block_with_init_py,
          "Hillis-Steele within block with explicit initial state + "
          "output block summary.");
    m.def("inclusive_matrix_prefix_scan", &inclusive_matrix_prefix_scan,
          "In-place Hillis-Steele prefix scan on a chain of matrices, "
          "O(log N) kernel launches.");
    m.def("intra_block_fused", &intra_block_fused_py,
          "Fused-build intra-block scan — takes raw S0/S_var/K/V/decay, "
          "no pre-materialized A tensor.");
    m.def("intra_block_fused_wmma", &intra_block_fused_wmma_py,
          "Stage 6: fused intra-block scan with TF32 tensor-core combines (N=16).");
}
