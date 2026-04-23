/**
 * Pararnn fused backward — CUDA C++ with proper multi-warp parallelism + cp.async.
 *
 * Design:
 *   - Block per (b, h). 128 threads (4 warps).
 *   - g tile [N, N] in shared memory (fp32).
 *   - Each WARP handles ⌈N / num_warps⌉ rows. Within a warp, 32 lanes
 *     handle the 32 columns of a row.
 *   - Row-level reductions via warp shuffle.
 *   - Column reductions (dK) via shared memory accumulator + cross-warp sync.
 *   - cp.async prefetch of next-step S_traj[t-1] while computing step t.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cstdio>
#include <cstdint>

namespace {

__device__ __forceinline__ float bf2f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ __nv_bfloat16 f2bf(float x) { return __float2bfloat16(x); }

// Warp-level sum reduction (32 lanes)
__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, d);
    }
    return v;
}

// ------------------------------------------------------------
// Stage 1: properly-threaded CUDA kernel (no cp.async yet)
// ------------------------------------------------------------
//
// NUM_WARPS warps, warp size 32. Each warp handles WARP_ROWS rows.
// For N=32, NUM_WARPS=4 → WARP_ROWS=8.
// For N=16, NUM_WARPS=2 → WARP_ROWS=8, or NUM_WARPS=4 → WARP_ROWS=4.

// cp.async.cg PTX wrapper: async copy from global to shared, 16-byte granularity
__device__ __forceinline__ void cp_async_16B(void* smem_dst, const void* gmem_src) {
    unsigned smem_ptr = __cvta_generic_to_shared(smem_dst);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_ptr), "l"(gmem_src)
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template<int N_MINUS>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N_MINUS));
}


template<int N, int NUM_WARPS>
__global__ void fused_backward_warp_parallel(
    const __nv_bfloat16* __restrict__ S_traj,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ decay,
    const __nv_bfloat16* __restrict__ g_T,
    const __nv_bfloat16* __restrict__ dL_dout,
    const __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ g_out,
    __nv_bfloat16* __restrict__ dK_out,
    __nv_bfloat16* __restrict__ dV_out,
    __nv_bfloat16* __restrict__ ddecay_out,
    int B, int H, int T
) {
    constexpr int WARP_ROWS = (N + NUM_WARPS - 1) / NUM_WARPS;
    constexpr int BLOCK_THREADS = NUM_WARPS * 32;

    const int bh = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid & 31;

    const long NN = (long)N * N;
    const long S_traj_base = (long)bh * (T + 1) * NN;
    const long K_base = (long)bh * T * N;
    const long dec_base = (long)bh * T;
    const long g_head_base = (long)bh * NN;

    extern __shared__ float smem[];
    float* g_smem = smem;                          // [N, N] fp32
    float* S_prev_smem = &smem[NN];                // [N, N] fp32
    float* K_smem = &smem[2 * NN];                 // [N] fp32
    float* V_smem = &smem[2 * NN + N];
    float* dL_smem = &smem[2 * NN + 2 * N];
    float* q_smem = &smem[2 * NN + 3 * N];
    // Per-warp dK and ddec accumulators to avoid atomicAdd contention.
    float* dK_perwarp = &smem[2 * NN + 4 * N];     // [NUM_WARPS, N]
    float* ddec_perwarp = &smem[2 * NN + 4 * N + NUM_WARPS * N];  // [NUM_WARPS]
    float* dec_smem_p = &smem[2 * NN + 4 * N + NUM_WARPS * N + NUM_WARPS];

    // Load g_T into shared memory (fp32)
    for (int i = tid; i < N * N; i += BLOCK_THREADS) {
        g_smem[i] = bf2f(g_T[g_head_base + i]);
    }

    // Reverse scan
    for (int t = T - 1; t >= 0; t--) {
        // Load per-step data (broadcasts across all threads)
        if (tid < N) {
            K_smem[tid] = bf2f(K[K_base + (long)t * N + tid]);
            V_smem[tid] = bf2f(V[K_base + (long)t * N + tid]);
            dL_smem[tid] = bf2f(dL_dout[K_base + (long)t * N + tid]);
            q_smem[tid] = bf2f(q[K_base + (long)t * N + tid]);
        }
        if (tid == 0) {
            *dec_smem_p = bf2f(decay[dec_base + t]);
        }
        // Reset per-warp accumulators
        if (tid < NUM_WARPS * N) {
            dK_perwarp[tid] = 0.0f;
        }
        if (tid < NUM_WARPS) {
            ddec_perwarp[tid] = 0.0f;
        }

        // Load S_traj[t] (S_prev) into shared memory (scalar bf16 for now)
        for (int i = tid; i < N * N; i += BLOCK_THREADS) {
            S_prev_smem[i] = bf2f(S_traj[S_traj_base + (long)t * NN + i]);
        }
        __syncthreads();

        const float dec = *dec_smem_p;

        // Each warp handles WARP_ROWS rows. Lane is the column index (lane < N).
        // lane>=N idle.
        #pragma unroll
        for (int r_local = 0; r_local < WARP_ROWS; r_local++) {
            const int r = warp_id * WARP_ROWS + r_local;
            if (r >= N) continue;

            const float K_c = (lane < N) ? K_smem[lane] : 0.0f;
            const float S_prev_rc = (lane < N) ? S_prev_smem[r * N + lane] : 0.0f;
            const float V_r = V_smem[r];
            const float g_rc = (lane < N) ? g_smem[r * N + lane] : 0.0f;

            // retrieved[r] = sum_c S_prev[r, c] * K[c]  (warp reduction)
            float retrieved_partial = S_prev_rc * K_c;
            float retrieved = warp_sum(retrieved_partial);

            float delta_row = V_r - retrieved;

            // pre[r, c] = dec * S_prev[r, c] + delta_row * K[c]
            float pre_rc = (lane < N) ? (dec * S_prev_rc + delta_row * K_c) : 0.0f;

            float e2x = __expf(2.0f * pre_rc);
            float tanh_val = (e2x - 1.0f) / (e2x + 1.0f);
            float tanh_deriv = 1.0f - tanh_val * tanh_val;
            float u_rc = tanh_deriv * K_c;

            // gu_r = sum_c g[r, c] * u[r, c]
            float gu_partial = g_rc * u_rc;
            float gu_r = warp_sum(gu_partial);

            // ddec contribution: g * tanh_deriv * S_prev (sum over cols for this row)
            float ddec_partial = g_rc * tanh_deriv * S_prev_rc;
            float ddec_r = warp_sum(ddec_partial);

            // Write dV[t, r] (only lane 0)
            if (lane == 0) {
                dV_out[K_base + (long)t * N + r] = f2bf(gu_r);
            }
            // ddec: accumulate into per-warp slot (no atomics)
            if (lane == 0) {
                ddec_perwarp[warp_id] += ddec_r;
            }

            // dK_contrib[r, c] = delta_row * (g * tanh_deriv)[r, c] - S_prev[r, c] * gu_r
            // Accumulate into per-warp dK slot (no atomics — each warp has own row, lane=col)
            if (lane < N) {
                float dk_rc = delta_row * (g_rc * tanh_deriv) - S_prev_rc * gu_r;
                dK_perwarp[warp_id * N + lane] += dk_rc;
            }

            // Update g[r, c]: g_new = D * g - K_t * gu_r, then add outer(dL, q)
            if (lane < N) {
                float D_rc = dec * tanh_deriv;
                float g_new_rc = D_rc * g_rc - K_c * gu_r;
                float ext_rc = dL_smem[r] * q_smem[lane];
                g_smem[r * N + lane] = g_new_rc + ext_rc;
            }
        }
        __syncthreads();

        // Reduce per-warp accumulators → final dK and ddec
        if (tid < N) {
            float acc = 0.0f;
            #pragma unroll
            for (int w = 0; w < NUM_WARPS; w++) acc += dK_perwarp[w * N + tid];
            dK_out[K_base + (long)t * N + tid] = f2bf(acc);
        }
        if (tid == 0) {
            float acc = 0.0f;
            #pragma unroll
            for (int w = 0; w < NUM_WARPS; w++) acc += ddec_perwarp[w];
            ddecay_out[dec_base + t] = f2bf(acc);
        }
        __syncthreads();
    }

    // Write final g → dL/dS_0
    for (int i = tid; i < N * N; i += BLOCK_THREADS) {
        g_out[g_head_base + i] = f2bf(g_smem[i]);
    }
}

}  // anonymous namespace


std::vector<torch::Tensor> cuda_fused_backward(
    torch::Tensor S_traj, torch::Tensor K, torch::Tensor V, torch::Tensor decay,
    torch::Tensor g_T, torch::Tensor dL_dout, torch::Tensor q
) {
    TORCH_CHECK(S_traj.dtype() == torch::kBFloat16);
    TORCH_CHECK(S_traj.is_cuda() && S_traj.is_contiguous());
    const int B = S_traj.size(0);
    const int H = S_traj.size(1);
    const int T = S_traj.size(2) - 1;
    const int N = S_traj.size(3);

    auto opts = torch::dtype(torch::kBFloat16).device(S_traj.device());
    auto g_out = torch::zeros({B, H, N, N}, opts);
    auto dK = torch::zeros({B, H, T, N}, opts);
    auto dV = torch::zeros({B, H, T, N}, opts);
    auto ddec = torch::zeros({B, H, T}, opts);

    const int grid = B * H;

    // Shared memory: 2*NN (g, S_prev) + 4*N (K, V, dL, q) + NW*N (dK_perwarp)
    // + NW (ddec_perwarp) + 1 (dec) + N (dK_accum) + 1 (ddec_accum)
    // = 2*NN + 5*N + NW*(N+1) + 2
    const int NW_guess = 4;
    const int shmem_f32_count = 2 * N * N + 5 * N + NW_guess * (N + 1) + 2;
    size_t shmem = shmem_f32_count * sizeof(float);

    auto S_ptr = reinterpret_cast<const __nv_bfloat16*>(S_traj.data_ptr());
    auto K_ptr = reinterpret_cast<const __nv_bfloat16*>(K.data_ptr());
    auto V_ptr = reinterpret_cast<const __nv_bfloat16*>(V.data_ptr());
    auto dec_ptr = reinterpret_cast<const __nv_bfloat16*>(decay.data_ptr());
    auto gT_ptr = reinterpret_cast<const __nv_bfloat16*>(g_T.data_ptr());
    auto dL_ptr = reinterpret_cast<const __nv_bfloat16*>(dL_dout.data_ptr());
    auto q_ptr = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
    auto go_ptr = reinterpret_cast<__nv_bfloat16*>(g_out.data_ptr());
    auto dK_ptr = reinterpret_cast<__nv_bfloat16*>(dK.data_ptr());
    auto dV_ptr = reinterpret_cast<__nv_bfloat16*>(dV.data_ptr());
    auto dd_ptr = reinterpret_cast<__nv_bfloat16*>(ddec.data_ptr());

    // Dispatch by N. NUM_WARPS chosen so each warp handles roughly equal rows.
    if (N == 16) {
        constexpr int NW = 4;  // 4 warps, 4 rows each
        fused_backward_warp_parallel<16, NW><<<grid, 32 * NW, shmem>>>(
            S_ptr, K_ptr, V_ptr, dec_ptr, gT_ptr, dL_ptr, q_ptr,
            go_ptr, dK_ptr, dV_ptr, dd_ptr, B, H, T);
    } else if (N == 32) {
        constexpr int NW = 4;  // 4 warps, 8 rows each
        fused_backward_warp_parallel<32, NW><<<grid, 32 * NW, shmem>>>(
            S_ptr, K_ptr, V_ptr, dec_ptr, gT_ptr, dL_ptr, q_ptr,
            go_ptr, dK_ptr, dV_ptr, dd_ptr, B, H, T);
    } else {
        TORCH_CHECK(false, "Unsupported N=", N);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA backward failed: ", cudaGetErrorString(err));

    return {g_out, dK, dV, ddec};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cuda_fused_backward", &cuda_fused_backward,
          "CUDA fused Pararnn backward — warp-parallel version (no cp.async yet).");
}
