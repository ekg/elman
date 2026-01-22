/**
 * E88 Fused Projection CUDA Kernel
 *
 * Fuses all input projections and post-processing into efficient CUDA calls:
 * 1. cuBLAS GEMM: Combined qkva = W_qkva @ x (q, k, v, alpha projections)
 * 2. Custom kernel: Apply depthwise conv + SiLU + L2 norm + decay computation
 *
 * This avoids multiple Python dispatch overheads and intermediate memory traffic.
 *
 * Input layout for combined projection:
 *   W_qkva: [2*key_dim + value_dim + n_heads, dim]
 *   Output: [q (key_dim), k (key_dim), v (value_dim), alpha (n_heads)]
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include "hasty/elman_ladder.h"

namespace {

// ============================================================================
// Post-Projection Kernel: Conv + SiLU + L2 Norm + Decay
// ============================================================================

/**
 * Applies depthwise conv (d_conv=4), SiLU, L2 normalization, and decay computation.
 *
 * Each block handles one (b, h) pair, iterates over t.
 * For q and k: Apply conv, SiLU, L2 normalize
 * For v: Apply conv, SiLU (no L2 norm)
 * For alpha: Compute decay = exp(-exp(A_log) * softplus(alpha + dt_bias))
 */
template<int D_CONV, bool USE_CONV, bool USE_SILU, bool USE_L2_NORM>
__global__ void E88PostProjectionKernel_BF16(
    int T,
    int B,
    int H,
    int key_dim,
    int value_dim,
    int n_state,          // key_dim / H
    int head_v_dim,       // value_dim / H
    const __nv_bfloat16* __restrict__ qkva_all,  // [T, B, 2*key_dim + value_dim + H]
    const __nv_bfloat16* __restrict__ conv_q,    // [key_dim, D_CONV] or nullptr
    const __nv_bfloat16* __restrict__ conv_k,    // [key_dim, D_CONV] or nullptr
    const __nv_bfloat16* __restrict__ conv_v,    // [value_dim, D_CONV] or nullptr
    const float* __restrict__ A_log,             // [H]
    const float* __restrict__ dt_bias,           // [H]
    __nv_bfloat16* __restrict__ q_out,           // [T, B, H, n_state]
    __nv_bfloat16* __restrict__ k_out,           // [T, B, H, n_state]
    __nv_bfloat16* __restrict__ v_out,           // [T, B, H, head_v_dim]
    __nv_bfloat16* __restrict__ decay_out        // [T, B, H]
) {
    // Each block handles one (b, h) pair, iterates over t
    int bh_idx = blockIdx.x;
    int b = bh_idx / H;
    int h = bh_idx % H;
    if (b >= B) return;

    int tid = threadIdx.x;
    int qkva_stride = 2 * key_dim + value_dim + H;

    // Offsets for this head in qkva
    int q_start = h * n_state;
    int k_start = key_dim + h * n_state;
    int v_start = 2 * key_dim + h * head_v_dim;
    int alpha_idx = 2 * key_dim + value_dim + h;

    // Load A_log and dt_bias for this head
    float a_log_h = A_log[h];
    float dt_bias_h = dt_bias[h];

    // Shared memory for conv history (D_CONV timesteps per channel)
    extern __shared__ float shared_mem[];
    float* q_hist = shared_mem;  // [D_CONV, n_state]
    float* k_hist = q_hist + D_CONV * n_state;  // [D_CONV, n_state]
    float* v_hist = k_hist + D_CONV * n_state;  // [D_CONV, head_v_dim]
    float* q_norm_buf = v_hist + D_CONV * head_v_dim;  // [n_state]
    float* k_norm_buf = q_norm_buf + n_state;  // [n_state]

    // Initialize history to zero
    for (int d = 0; d < D_CONV; d++) {
        for (int i = tid; i < n_state; i += blockDim.x) {
            q_hist[d * n_state + i] = 0.0f;
            k_hist[d * n_state + i] = 0.0f;
        }
        for (int i = tid; i < head_v_dim; i += blockDim.x) {
            v_hist[d * head_v_dim + i] = 0.0f;
        }
    }
    __syncthreads();

    // Process each timestep
    for (int t = 0; t < T; t++) {
        int qkva_base = (t * B + b) * qkva_stride;

        // Shift history: move hist[1..D_CONV-1] to hist[0..D_CONV-2]
        for (int d = 0; d < D_CONV - 1; d++) {
            for (int i = tid; i < n_state; i += blockDim.x) {
                q_hist[d * n_state + i] = q_hist[(d + 1) * n_state + i];
                k_hist[d * n_state + i] = k_hist[(d + 1) * n_state + i];
            }
            for (int i = tid; i < head_v_dim; i += blockDim.x) {
                v_hist[d * head_v_dim + i] = v_hist[(d + 1) * head_v_dim + i];
            }
        }
        __syncthreads();

        // Load current values into newest slot
        for (int i = tid; i < n_state; i += blockDim.x) {
            q_hist[(D_CONV - 1) * n_state + i] = __bfloat162float(qkva_all[qkva_base + q_start + i]);
            k_hist[(D_CONV - 1) * n_state + i] = __bfloat162float(qkva_all[qkva_base + k_start + i]);
        }
        for (int i = tid; i < head_v_dim; i += blockDim.x) {
            v_hist[(D_CONV - 1) * head_v_dim + i] = __bfloat162float(qkva_all[qkva_base + v_start + i]);
        }
        __syncthreads();

        // Apply depthwise conv and SiLU to q, k
        float q_sum_sq = 0.0f;
        float k_sum_sq = 0.0f;

        for (int i = tid; i < n_state; i += blockDim.x) {
            float q_val, k_val;

            if constexpr (USE_CONV) {
                q_val = 0.0f;
                k_val = 0.0f;
                int q_conv_base = (h * n_state + i) * D_CONV;
                int k_conv_base = (h * n_state + i) * D_CONV;

                #pragma unroll
                for (int d = 0; d < D_CONV; d++) {
                    float q_h = q_hist[d * n_state + i];
                    float k_h = k_hist[d * n_state + i];
                    float q_w = __bfloat162float(conv_q[q_conv_base + d]);
                    float k_w = __bfloat162float(conv_k[k_conv_base + d]);
                    q_val += q_h * q_w;
                    k_val += k_h * k_w;
                }
            } else {
                q_val = q_hist[(D_CONV - 1) * n_state + i];
                k_val = k_hist[(D_CONV - 1) * n_state + i];
            }

            // Apply SiLU: x * sigmoid(x)
            if constexpr (USE_SILU) {
                float q_sig = 1.0f / (1.0f + expf(-q_val));
                float k_sig = 1.0f / (1.0f + expf(-k_val));
                q_val = q_val * q_sig;
                k_val = k_val * k_sig;
            }

            q_norm_buf[i] = q_val;
            k_norm_buf[i] = k_val;

            if constexpr (USE_L2_NORM) {
                q_sum_sq += q_val * q_val;
                k_sum_sq += k_val * k_val;
            }
        }
        __syncthreads();

        // L2 normalize q and k using block reduction
        if constexpr (USE_L2_NORM) {
            __shared__ float q_norm_sq_total;
            __shared__ float k_norm_sq_total;

            if (tid == 0) {
                q_norm_sq_total = 0.0f;
                k_norm_sq_total = 0.0f;
            }
            __syncthreads();

            atomicAdd(&q_norm_sq_total, q_sum_sq);
            atomicAdd(&k_norm_sq_total, k_sum_sq);
            __syncthreads();

            float q_norm_inv = rsqrtf(q_norm_sq_total + 1e-6f);
            float k_norm_inv = rsqrtf(k_norm_sq_total + 1e-6f);

            // Write normalized q, k
            int q_out_offset = ((t * B + b) * H + h) * n_state;
            int k_out_offset = ((t * B + b) * H + h) * n_state;

            for (int i = tid; i < n_state; i += blockDim.x) {
                q_out[q_out_offset + i] = __float2bfloat16(q_norm_buf[i] * q_norm_inv);
                k_out[k_out_offset + i] = __float2bfloat16(k_norm_buf[i] * k_norm_inv);
            }
        } else {
            // Write q, k without normalization
            int q_out_offset = ((t * B + b) * H + h) * n_state;
            int k_out_offset = ((t * B + b) * H + h) * n_state;

            for (int i = tid; i < n_state; i += blockDim.x) {
                q_out[q_out_offset + i] = __float2bfloat16(q_norm_buf[i]);
                k_out[k_out_offset + i] = __float2bfloat16(k_norm_buf[i]);
            }
        }

        // Process v (conv + SiLU only, no L2 norm)
        int v_out_offset = ((t * B + b) * H + h) * head_v_dim;
        for (int i = tid; i < head_v_dim; i += blockDim.x) {
            float v_val;

            if constexpr (USE_CONV) {
                v_val = 0.0f;
                int v_conv_base = (h * head_v_dim + i) * D_CONV;

                #pragma unroll
                for (int d = 0; d < D_CONV; d++) {
                    float v_h = v_hist[d * head_v_dim + i];
                    float v_w = __bfloat162float(conv_v[v_conv_base + d]);
                    v_val += v_h * v_w;
                }
            } else {
                v_val = v_hist[(D_CONV - 1) * head_v_dim + i];
            }

            // Apply SiLU
            if constexpr (USE_SILU) {
                float v_sig = 1.0f / (1.0f + expf(-v_val));
                v_val = v_val * v_sig;
            }

            v_out[v_out_offset + i] = __float2bfloat16(v_val);
        }

        // Compute decay from alpha
        // decay = exp(-exp(A_log) * softplus(alpha + dt_bias))
        if (tid == 0) {
            float alpha = __bfloat162float(qkva_all[qkva_base + alpha_idx]);
            float x = alpha + dt_bias_h;
            // softplus(x) = log(1 + exp(x)), numerically stable version
            float sp = (x > 20.0f) ? x : logf(1.0f + expf(x));
            float g = -expf(a_log_h) * sp;
            float decay = expf(g);

            int decay_offset = (t * B + b) * H + h;
            decay_out[decay_offset] = __float2bfloat16(decay);
        }
        __syncthreads();
    }
}

}  // anonymous namespace


// ============================================================================
// Dispatch Functions (in elman namespace for linkage with header)
// ============================================================================

namespace elman {

/**
 * Compute fused qkva projection using cuBLAS GEMM.
 *
 * x: [T, B, dim]
 * W_qkva: [2*key_dim + value_dim + n_heads, dim]
 * qkva_out: [T, B, 2*key_dim + value_dim + n_heads]
 */
static void e88_fused_qkva_gemm(
    cublasHandle_t handle,
    int T, int B, int dim,
    int key_dim, int value_dim, int n_heads,
    const __nv_bfloat16* x,
    const __nv_bfloat16* W_qkva,
    __nv_bfloat16* qkva_out,
    cudaStream_t stream
) {
    int out_dim = 2 * key_dim + value_dim + n_heads;
    int M = T * B;  // batch dimension
    int N = out_dim;  // output dimension
    int K = dim;  // input dimension

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasSetStream(handle, stream);

    // For row-major data with cuBLAS (column-major):
    // We have x [M, K] and W [N, K], want out [M, N]
    // out = x @ W^T
    // In cuBLAS: out^T [N, M] = W [N, K] @ x^T [K, M]
    cublasGemmEx(
        handle,
        CUBLAS_OP_T,   // op(A) = W^T
        CUBLAS_OP_N,   // op(B) = x
        N, M, K,       // m, n, k
        &alpha,
        W_qkva, CUDA_R_16BF, K,
        x, CUDA_R_16BF, K,
        &beta,
        qkva_out, CUDA_R_16BF, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );
}

void dispatch_e88_post_projection(
    int T, int B, int H,
    int key_dim, int value_dim,
    int n_state, int head_v_dim,
    int d_conv,
    bool use_conv, bool use_silu, bool use_l2_norm,
    const __nv_bfloat16* qkva_all,
    const __nv_bfloat16* conv_q,
    const __nv_bfloat16* conv_k,
    const __nv_bfloat16* conv_v,
    const float* A_log,
    const float* dt_bias,
    __nv_bfloat16* q_out,
    __nv_bfloat16* k_out,
    __nv_bfloat16* v_out,
    __nv_bfloat16* decay_out,
    cudaStream_t stream
) {
    int num_blocks = B * H;
    int threads = 128;

    // Shared memory: conv history for q, k, v + normalization buffers
    int shared_size = (d_conv * n_state + d_conv * n_state + d_conv * head_v_dim +
                      n_state + n_state) * sizeof(float);

    // Template dispatch (only d_conv=4 supported)
    if (d_conv != 4) {
        fprintf(stderr, "E88 fused projection only supports d_conv=4\n");
        return;
    }

    #define LAUNCH_KERNEL(CONV, SILU, L2) \
        E88PostProjectionKernel_BF16<4, CONV, SILU, L2><<<num_blocks, threads, shared_size, stream>>>( \
            T, B, H, key_dim, value_dim, n_state, head_v_dim, \
            qkva_all, conv_q, conv_k, conv_v, A_log, dt_bias, \
            q_out, k_out, v_out, decay_out)

    if (use_conv && use_silu && use_l2_norm) {
        LAUNCH_KERNEL(true, true, true);
    } else if (use_conv && use_silu && !use_l2_norm) {
        LAUNCH_KERNEL(true, true, false);
    } else if (use_conv && !use_silu && use_l2_norm) {
        LAUNCH_KERNEL(true, false, true);
    } else if (use_conv && !use_silu && !use_l2_norm) {
        LAUNCH_KERNEL(true, false, false);
    } else if (!use_conv && use_silu && use_l2_norm) {
        LAUNCH_KERNEL(false, true, true);
    } else if (!use_conv && use_silu && !use_l2_norm) {
        LAUNCH_KERNEL(false, true, false);
    } else if (!use_conv && !use_silu && use_l2_norm) {
        LAUNCH_KERNEL(false, false, true);
    } else {
        LAUNCH_KERNEL(false, false, false);
    }

    #undef LAUNCH_KERNEL
}

void e88_fused_projection(
    cublasHandle_t handle,
    int T, int B, int dim,
    int H, int key_dim, int value_dim,
    int n_state, int head_v_dim,
    int d_conv,
    bool use_conv, bool use_silu, bool use_l2_norm,
    const __nv_bfloat16* x,
    const __nv_bfloat16* W_qkva,
    const __nv_bfloat16* conv_q,
    const __nv_bfloat16* conv_k,
    const __nv_bfloat16* conv_v,
    const float* A_log,
    const float* dt_bias,
    __nv_bfloat16* qkva_workspace,
    __nv_bfloat16* q_out,
    __nv_bfloat16* k_out,
    __nv_bfloat16* v_out,
    __nv_bfloat16* decay_out,
    cudaStream_t stream
) {
    // Step 1: Compute qkva = x @ W_qkva^T using cuBLAS
    e88_fused_qkva_gemm(handle, T, B, dim, key_dim, value_dim, H,
                        x, W_qkva, qkva_workspace, stream);

    // Step 2: Apply post-projection processing
    dispatch_e88_post_projection(
        T, B, H, key_dim, value_dim, n_state, head_v_dim, d_conv,
        use_conv, use_silu, use_l2_norm,
        qkva_workspace, conv_q, conv_k, conv_v, A_log, dt_bias,
        q_out, k_out, v_out, decay_out, stream);
}

}  // namespace elman
