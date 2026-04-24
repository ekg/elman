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

// Vectorized load: 8 bf16 (16 bytes) in one HBM transaction.
// Source pointer must be 16-byte aligned.
__device__ __forceinline__ void load_8bf16_to_f32(const __nv_bfloat16* __restrict__ src,
                                                    float* __restrict__ dst) {
    uint4 vec = *reinterpret_cast<const uint4*>(src);
    __nv_bfloat162 b0 = reinterpret_cast<__nv_bfloat162&>(vec.x);
    __nv_bfloat162 b1 = reinterpret_cast<__nv_bfloat162&>(vec.y);
    __nv_bfloat162 b2 = reinterpret_cast<__nv_bfloat162&>(vec.z);
    __nv_bfloat162 b3 = reinterpret_cast<__nv_bfloat162&>(vec.w);
    float2 f0 = __bfloat1622float2(b0);
    float2 f1 = __bfloat1622float2(b1);
    float2 f2 = __bfloat1622float2(b2);
    float2 f3 = __bfloat1622float2(b3);
    dst[0] = f0.x; dst[1] = f0.y;
    dst[2] = f1.x; dst[3] = f1.y;
    dst[4] = f2.x; dst[5] = f2.y;
    dst[6] = f3.x; dst[7] = f3.y;
}

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


// Version 2: cp.async double-buffered S_prev prefetch.
// Reads step t's S_prev while computing step t+1 (reverse scan).

template<int N, int NUM_WARPS>
__launch_bounds__(NUM_WARPS * 32, 4)  // target 4 blocks per SM = ≤128 regs/thread
__global__ void fused_backward_cpasync(
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

    extern __shared__ unsigned char smem_raw[];

    // Layout:
    //   g_smem:          NN fp32
    //   S_prev_bf16[0]:  NN bf16   (prefetch buffer 0)
    //   S_prev_bf16[1]:  NN bf16   (prefetch buffer 1)
    //   K_smem:          N fp32
    //   V_smem:          N fp32
    //   dL_smem:         N fp32
    //   q_smem:          N fp32
    //   dK_perwarp:      NUM_WARPS * N fp32
    //   ddec_perwarp:    NUM_WARPS fp32
    //   dec_smem:        1 fp32

    // g lives in PER-THREAD REGISTERS. Each thread holds WARP_ROWS values
    // (one per row the thread's warp handles). Thread lane = column index.
    // Only lanes < N hold valid g data; others are padding.
    __nv_bfloat16* S_prev_bf16[2];
    S_prev_bf16[0] = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    S_prev_bf16[1] = reinterpret_cast<__nv_bfloat16*>(&S_prev_bf16[0][NN]);
    float* K_smem = reinterpret_cast<float*>(&S_prev_bf16[1][NN]);
    float* V_smem = &K_smem[N];
    float* dL_smem = &V_smem[N];
    float* q_smem = &dL_smem[N];
    float* dK_perwarp = &q_smem[N];
    float* ddec_perwarp = &dK_perwarp[NUM_WARPS * N];
    float* dec_smem_p = &ddec_perwarp[NUM_WARPS];

    // g in registers: g_reg[r_local] is the element at (warp_id * WARP_ROWS + r_local, lane)
    float g_reg[WARP_ROWS];

    // Load g_T into registers
    #pragma unroll
    for (int r_local = 0; r_local < WARP_ROWS; r_local++) {
        const int r = warp_id * WARP_ROWS + r_local;
        if (r < N && lane < N) {
            g_reg[r_local] = bf2f(g_T[g_head_base + r * N + lane]);
        } else {
            g_reg[r_local] = 0.0f;
        }
    }

    // Init accumulators (reset before first iter)
    if (tid < NUM_WARPS * N) dK_perwarp[tid] = 0.0f;
    if (tid < NUM_WARPS) ddec_perwarp[tid] = 0.0f;

    // Prefetch step t=T-1 into buffer 0
    const int vec_count = NN / 8;
    {
        const int t_init = T - 1;
        const __nv_bfloat16* src_base = &S_traj[S_traj_base + (long)t_init * NN];
        for (int vi = tid; vi < vec_count; vi += BLOCK_THREADS) {
            cp_async_16B(&S_prev_bf16[0][vi * 8], &src_base[vi * 8]);
        }
    }
    cp_async_commit();

    int cur_buf = 0;
    // ========================================================================
    // Reverse scan
    // ========================================================================
    for (int t = T - 1; t >= 0; t--) {
        int next_buf = 1 - cur_buf;

        // Prefetch S_traj[t-1] into next buffer (if not last iteration)
        if (t > 0) {
            const __nv_bfloat16* src_base = &S_traj[S_traj_base + (long)(t - 1) * NN];
            for (int vi = tid; vi < vec_count; vi += BLOCK_THREADS) {
                cp_async_16B(&S_prev_bf16[next_buf][vi * 8], &src_base[vi * 8]);
            }
            cp_async_commit();
        }

        // Load per-step broadcasts (K, V, dL, q, decay).
        // Use __ldg for hardware-cached global loads (goes through L1 tex cache).
        if (tid < N) {
            K_smem[tid] = bf2f(__ldg(&K[K_base + (long)t * N + tid]));
            V_smem[tid] = bf2f(__ldg(&V[K_base + (long)t * N + tid]));
            dL_smem[tid] = bf2f(__ldg(&dL_dout[K_base + (long)t * N + tid]));
            q_smem[tid] = bf2f(__ldg(&q[K_base + (long)t * N + tid]));
        }
        if (tid == 0) {
            *dec_smem_p = bf2f(__ldg(&decay[dec_base + t]));
        }

        // Wait for S_prev load to complete
        if (t > 0) {
            cp_async_wait<1>();  // keep next_buf's load in flight
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();

        // Read S_prev from current buffer (bf16)
        const __nv_bfloat16* S_prev_cur = S_prev_bf16[cur_buf];
        const float dec = *dec_smem_p;

        // Per-warp processing of WARP_ROWS rows
        #pragma unroll
        for (int r_local = 0; r_local < WARP_ROWS; r_local++) {
            const int r = warp_id * WARP_ROWS + r_local;
            if (r >= N) continue;

            const float K_c = (lane < N) ? K_smem[lane] : 0.0f;
            const float S_prev_rc = (lane < N) ? bf2f(S_prev_cur[r * N + lane]) : 0.0f;
            const float V_r = V_smem[r];
            // Inject output_t grad: output_t = S_{t+1} @ q_t, so
            // dL/dS_{t+1} receives dL_dout[t,r] * q[t,c].  g_reg holds
            // dL/dS_{t+1} at iter t pre-injection.
            const float ext_rc = (lane < N) ? (dL_smem[r] * q_smem[lane]) : 0.0f;
            const float g_rc = g_reg[r_local] + ext_rc;

            float retrieved_partial = S_prev_rc * K_c;
            float retrieved = warp_sum(retrieved_partial);
            float delta_row = V_r - retrieved;

            float pre_rc = (lane < N) ? (dec * S_prev_rc + delta_row * K_c) : 0.0f;
            float e2x = __expf(2.0f * pre_rc);
            float tanh_val = (e2x - 1.0f) / (e2x + 1.0f);
            float tanh_deriv = 1.0f - tanh_val * tanh_val;
            float u_rc = tanh_deriv * K_c;

            float gu_partial = g_rc * u_rc;
            float gu_r = warp_sum(gu_partial);
            float ddec_partial = g_rc * tanh_deriv * S_prev_rc;
            float ddec_r = warp_sum(ddec_partial);

            if (lane == 0) {
                dV_out[K_base + (long)t * N + r] = f2bf(gu_r);
                ddec_perwarp[warp_id] += ddec_r;
            }
            if (lane < N) {
                float dk_rc = delta_row * (g_rc * tanh_deriv) - S_prev_rc * gu_r;
                dK_perwarp[warp_id * N + lane] += dk_rc;
            }
            {
                float D_rc = dec * tanh_deriv;
                // g_rc now == dL/dS_{t+1} including output_t's contribution.
                // g_new_rc = dL/dS_t (will get output_{t-1}'s contribution at next iter).
                g_reg[r_local] = D_rc * g_rc - K_c * gu_r;
            }
        }
        __syncthreads();

        // Reduce dK and ddec, write outputs, AND RESET accumulators in one pass.
        // This avoids the final sync before next iter — reset is done in the
        // same thread that just read the values.
        if (tid < N) {
            float acc = 0.0f;
            #pragma unroll
            for (int w = 0; w < NUM_WARPS; w++) {
                acc += dK_perwarp[w * N + tid];
                dK_perwarp[w * N + tid] = 0.0f;
            }
            dK_out[K_base + (long)t * N + tid] = f2bf(acc);
        }
        if (tid == 0) {
            float acc = 0.0f;
            #pragma unroll
            for (int w = 0; w < NUM_WARPS; w++) {
                acc += ddec_perwarp[w];
                ddec_perwarp[w] = 0.0f;
            }
            ddecay_out[dec_base + t] = f2bf(acc);
        }
        // No sync here — next iter's compute doesn't touch dK_perwarp until
        // after its own cp.async wait + sync, at which point these writes are
        // visible. The reset-after-read pattern is race-free.

        cur_buf = next_buf;
    }

    // Write final g → dL/dS_0 (from registers)
    #pragma unroll
    for (int r_local = 0; r_local < WARP_ROWS; r_local++) {
        const int r = warp_id * WARP_ROWS + r_local;
        if (r < N && lane < N) {
            g_out[g_head_base + r * N + lane] = f2bf(g_reg[r_local]);
        }
    }
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

        // Load S_traj[t] (S_prev) vectorized — 8 bf16 / thread / iteration.
        {
            const __nv_bfloat16* src_base = &S_traj[S_traj_base + (long)t * NN];
            const int vec_count = (N * N) / 8;
            for (int vi = tid; vi < vec_count; vi += BLOCK_THREADS) {
                load_8bf16_to_f32(&src_base[vi * 8], &S_prev_smem[vi * 8]);
            }
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
            // Inject output_t grad before using g as dL/dS_{t+1}:
            // output_t = S_{t+1} @ q_t → dL/dS_{t+1} += dL_dout[t,r] * q[t,c].
            const float ext_inj = (lane < N) ? (dL_smem[r] * q_smem[lane]) : 0.0f;
            const float g_rc = ((lane < N) ? g_smem[r * N + lane] : 0.0f) + ext_inj;

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

            // Update g[r, c]: g_new = D * g - K_t * gu_r.  Output_t injection
            // was already folded into g_rc at read-time.
            if (lane < N) {
                float D_rc = dec * tanh_deriv;
                g_smem[r * N + lane] = D_rc * g_rc - K_c * gu_r;
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


std::vector<torch::Tensor> cuda_fused_backward_cpasync(
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

    // Shared mem layout:
    //   g (fp32): NN floats → 4*NN bytes
    //   S_prev_bf16[2]: 2*NN bf16 = 4*NN bytes
    //   K, V, dL, q: 4*N floats = 16*N bytes
    //   dK_perwarp: NUM_WARPS * N floats
    //   ddec_perwarp: NUM_WARPS floats
    //   dec: 1 float
    // Total: 8*NN + 4*N*(1+?) + ... Over-allocate generously.
    const int NW = 16;  // enough for both paths (over-allocates slightly for N=32)
    const int bytes_shmem = 2 * 2 * N * N   // S_prev_bf16 double buffer (g is in regs now)
                           + 4 * 4 * N      // K,V,dL,q fp32
                           + 4 * NW * N     // dK_perwarp
                           + 4 * NW         // ddec_perwarp
                           + 4              // dec
                           + 64;            // padding
    size_t shmem = bytes_shmem;

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

    // Dispatch: more warps = more row-level parallelism, but more smem/regs.
    // Measured sweet spot at production configs:
    //   N=16 → NW=8 (2 rows/warp)
    //   N=32 → NW=8 (4 rows/warp)
    if (N == 16) {
        fused_backward_cpasync<16, 16><<<grid, 32 * 16, shmem>>>(
            S_ptr, K_ptr, V_ptr, dec_ptr, gT_ptr, dL_ptr, q_ptr,
            go_ptr, dK_ptr, dV_ptr, dd_ptr, B, H, T);
    } else if (N == 32) {
        fused_backward_cpasync<32, 8><<<grid, 32 * 8, shmem>>>(
            S_ptr, K_ptr, V_ptr, dec_ptr, gT_ptr, dL_ptr, q_ptr,
            go_ptr, dK_ptr, dV_ptr, dd_ptr, B, H, T);
    } else {
        TORCH_CHECK(false, "Unsupported N=", N);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA cpasync backward failed: ", cudaGetErrorString(err));

    return {g_out, dK, dV, ddec};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cuda_fused_backward", &cuda_fused_backward,
          "CUDA fused Pararnn backward — warp-parallel version.");
    m.def("cuda_fused_backward_cpasync", &cuda_fused_backward_cpasync,
          "CUDA fused backward with cp.async double-buffered S_prev prefetch.");
}
