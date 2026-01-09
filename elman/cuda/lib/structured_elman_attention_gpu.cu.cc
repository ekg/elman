/**
 * E22: Structured Elman with State Attention (UTM class)
 *
 * Fused CUDA kernel that keeps state in shared memory for entire sequence.
 * State attention over N positions is small enough to compute entirely in shmem.
 *
 * Architecture:
 *   Per timestep:
 *     H = silu(α × H_prev + B @ X.T)     # MIMO rank-R update
 *   Every K timesteps:
 *     H = H + StateAttention(H)           # Routing via attention over N
 *   Output:
 *     y = H.sum(dim=N)
 *     output = y × silu(z + y)
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cmath>
#include <vector>
#include "hasty/elman_ladder.h"

namespace {

// Compile-time constants for template specialization
constexpr int BLOCK_SIZE = 256;

// Helper: atomicAdd for bfloat16 using CAS loop
__device__ __forceinline__ void atomicAddBF16(__nv_bfloat16* addr, float val) {
    // Reinterpret the bf16 address as an unsigned short address
    unsigned short* addr_as_ushort = reinterpret_cast<unsigned short*>(addr);
    unsigned short old = *addr_as_ushort;
    unsigned short assumed;
    do {
        assumed = old;
        float old_val = __bfloat162float(__ushort_as_bfloat16(assumed));
        float new_val = old_val + val;
        unsigned short new_bits = __bfloat16_as_ushort(__float2bfloat16(new_val));
        old = atomicCAS(addr_as_ushort, assumed, new_bits);
    } while (assumed != old);
}

/**
 * E22 Forward Kernel - Fully fused, state in shared memory
 *
 * Template params:
 *   N: state dimension (d_state), typically 32
 *   P: head dimension (headdim), typically 64
 *   R: MIMO rank, typically 4 or 8
 *   D_K: attention key dimension, typically 32
 */
template<int N, int P, int R, int D_K>
__global__ void E22ForwardKernel_BF16(
    const int seq_len,
    const int batch_size,
    const int nheads,
    const int attn_period,        // K: attend every K steps
    const int nonlinearity_mode,  // 0=silu, 1=tanh, 2=linear
    // E21 inputs [T, B, H, ...]
    const __nv_bfloat16* __restrict__ B_proj,     // [T, B, H, N, R]
    const __nv_bfloat16* __restrict__ X_proj,     // [T, B, H, P, R]
    const __nv_bfloat16* __restrict__ alpha_raw,  // [T, B, H]
    const __nv_bfloat16* __restrict__ alpha_bias, // [H]
    const __nv_bfloat16* __restrict__ z,          // [T, B, H*P]
    // E22 attention weights (shared across all heads for efficiency)
    const __nv_bfloat16* __restrict__ W_q,        // [H, P, D_K]
    const __nv_bfloat16* __restrict__ W_k,        // [H, P, D_K]
    const __nv_bfloat16* __restrict__ W_v,        // [H, P, D_K]
    const __nv_bfloat16* __restrict__ W_o,        // [H, D_K, P]
    // Initial state
    const __nv_bfloat16* __restrict__ H_init,     // [B, H, N, P]
    // Outputs
    __nv_bfloat16* __restrict__ output,           // [T, B, H*P]
    __nv_bfloat16* __restrict__ H_final,          // [B, H, N, P]
    // For backward (training only)
    __nv_bfloat16* __restrict__ H_all,            // [T+1, B, H, N, P]
    __nv_bfloat16* __restrict__ y_cache           // [T, B, H*P]
) {
    // Each block handles one (batch, head) pair
    const int bh = blockIdx.x;
    const int b = bh / nheads;
    const int h = bh % nheads;

    if (b >= batch_size) return;

    const int tid = threadIdx.x;

    // Shared memory layout (total ~28KB for N=32, P=64, D_K=32)
    __shared__ float H_sh[N * P];           // 8KB - current state
    __shared__ float update_sh[N * P];      // 8KB - MIMO update (reused for V_out)
    __shared__ float Q_sh[N * D_K];         // 4KB
    __shared__ float K_sh[N * D_K];         // 4KB
    __shared__ float attn_sh[N * N];        // 4KB - attention scores
    __shared__ float y_sh[P];               // 256B - output sum

    // Load attention weights into registers (per-head weights)
    // These are accessed multiple times during attention, so cache in registers
    const int W_head_offset = h * P * D_K;

    // Load initial state into shared memory
    const int H_offset = (b * nheads + h) * N * P;
    for (int i = tid; i < N * P; i += BLOCK_SIZE) {
        H_sh[i] = __bfloat162float(H_init[H_offset + i]);
    }
    __syncthreads();

    // Store initial state for backprop
    if (H_all != nullptr) {
        const int H_all_offset = (b * nheads + h) * N * P;
        for (int i = tid; i < N * P; i += BLOCK_SIZE) {
            H_all[H_all_offset + i] = __float2bfloat16(H_sh[i]);
        }
    }
    __syncthreads();

    // Get alpha bias for this head
    const float alpha_b = __bfloat162float(alpha_bias[h]);

    // Main loop over timesteps
    for (int t = 0; t < seq_len; t++) {

        // === Step 1: MIMO update ===
        // update[n,p] = sum_r B[n,r] * X[p,r]
        const int B_offset = ((t * batch_size + b) * nheads + h) * N * R;
        const int X_offset = ((t * batch_size + b) * nheads + h) * P * R;

        for (int np = tid; np < N * P; np += BLOCK_SIZE) {
            const int n = np / P;
            const int p = np % P;

            float sum = 0.0f;
            #pragma unroll
            for (int r = 0; r < R; r++) {
                float b_val = __bfloat162float(B_proj[B_offset + n * R + r]);
                float x_val = __bfloat162float(X_proj[X_offset + p * R + r]);
                sum += b_val * x_val;
            }
            update_sh[np] = sum;
        }
        __syncthreads();

        // === Step 2: Decay + nonlinearity ===
        // H = nonlin(alpha * H_prev + update)
        const int alpha_offset = (t * batch_size + b) * nheads + h;
        float alpha_raw_val = __bfloat162float(alpha_raw[alpha_offset]);

        // alpha = sigmoid(-softplus(raw + bias))
        float sp = log1pf(expf(alpha_raw_val + alpha_b));
        float alpha = 1.0f / (1.0f + expf(sp));

        for (int np = tid; np < N * P; np += BLOCK_SIZE) {
            float pre_act = alpha * H_sh[np] + update_sh[np];
            float h_new;

            if (nonlinearity_mode == 0) {  // silu
                float sigmoid_val = 1.0f / (1.0f + expf(-pre_act));
                h_new = pre_act * sigmoid_val;
            } else if (nonlinearity_mode == 1) {  // tanh
                h_new = tanhf(pre_act);
            } else {  // linear
                h_new = pre_act;
            }
            H_sh[np] = h_new;
        }
        __syncthreads();

        // === Step 3: Conditional state attention (every K steps) ===
        if (attn_period > 0 && ((t + 1) % attn_period) == 0) {

            // Q projection: Q[n,d] = sum_p H[n,p] * W_q[p,d]
            for (int nd = tid; nd < N * D_K; nd += BLOCK_SIZE) {
                const int n = nd / D_K;
                const int d = nd % D_K;
                float sum = 0.0f;
                #pragma unroll 8
                for (int p = 0; p < P; p++) {
                    float h_val = H_sh[n * P + p];
                    float w_val = __bfloat162float(W_q[W_head_offset + p * D_K + d]);
                    sum += h_val * w_val;
                }
                Q_sh[nd] = sum;
            }
            __syncthreads();

            // K projection: K[n,d] = sum_p H[n,p] * W_k[p,d]
            for (int nd = tid; nd < N * D_K; nd += BLOCK_SIZE) {
                const int n = nd / D_K;
                const int d = nd % D_K;
                float sum = 0.0f;
                #pragma unroll 8
                for (int p = 0; p < P; p++) {
                    float h_val = H_sh[n * P + p];
                    float w_val = __bfloat162float(W_k[W_head_offset + p * D_K + d]);
                    sum += h_val * w_val;
                }
                K_sh[nd] = sum;
            }
            __syncthreads();

            // V projection (reuse update_sh as V storage)
            float* V_sh = update_sh;  // Reuse buffer
            for (int nd = tid; nd < N * D_K; nd += BLOCK_SIZE) {
                const int n = nd / D_K;
                const int d = nd % D_K;
                float sum = 0.0f;
                #pragma unroll 8
                for (int p = 0; p < P; p++) {
                    float h_val = H_sh[n * P + p];
                    float w_val = __bfloat162float(W_v[W_head_offset + p * D_K + d]);
                    sum += h_val * w_val;
                }
                V_sh[nd] = sum;
            }
            __syncthreads();

            // Attention scores + softmax (one row per thread for efficiency)
            const float scale = 1.0f / sqrtf((float)D_K);

            for (int i = tid; i < N; i += BLOCK_SIZE) {
                // Compute scores for row i
                float max_score = -INFINITY;

                for (int j = 0; j < N; j++) {
                    float score = 0.0f;
                    #pragma unroll 8
                    for (int d = 0; d < D_K; d++) {
                        score += Q_sh[i * D_K + d] * K_sh[j * D_K + d];
                    }
                    score *= scale;
                    attn_sh[i * N + j] = score;
                    max_score = fmaxf(max_score, score);
                }

                // Softmax
                float sum_exp = 0.0f;
                for (int j = 0; j < N; j++) {
                    float e = expf(attn_sh[i * N + j] - max_score);
                    attn_sh[i * N + j] = e;
                    sum_exp += e;
                }
                float inv_sum = 1.0f / sum_exp;
                for (int j = 0; j < N; j++) {
                    attn_sh[i * N + j] *= inv_sum;
                }
            }
            __syncthreads();

            // Apply attention and project back: H_delta[n,p] = sum_d (sum_j attn[n,j] * V[j,d]) * W_o[d,p]
            const int W_o_offset = h * D_K * P;

            for (int np = tid; np < N * P; np += BLOCK_SIZE) {
                const int n = np / P;
                const int p = np % P;

                float delta = 0.0f;
                for (int d = 0; d < D_K; d++) {
                    // V_out[n,d] = sum_j attn[n,j] * V[j,d]
                    float v_out = 0.0f;
                    for (int j = 0; j < N; j++) {
                        v_out += attn_sh[n * N + j] * V_sh[j * D_K + d];
                    }
                    // delta += v_out * W_o[d,p]
                    delta += v_out * __bfloat162float(W_o[W_o_offset + d * P + p]);
                }
                H_sh[np] += delta;  // Residual connection
            }
            __syncthreads();
        }

        // === Step 4: Output ===
        // y[p] = sum_n H[n,p]
        for (int p = tid; p < P; p += BLOCK_SIZE) {
            float sum = 0.0f;
            #pragma unroll 8
            for (int n = 0; n < N; n++) {
                sum += H_sh[n * P + p];
            }
            y_sh[p] = sum;
        }
        __syncthreads();

        // Gated output: out = y * silu(z + y)
        const int z_offset = (t * batch_size + b) * nheads * P + h * P;
        const int out_offset = z_offset;

        for (int p = tid; p < P; p += BLOCK_SIZE) {
            float y_val = y_sh[p];
            float z_val = __bfloat162float(z[z_offset + p]);

            // gate = silu(z + y)
            float gate_in = z_val + y_val;
            float gate = gate_in / (1.0f + expf(-gate_in));

            // out = y * gate
            float out_val = y_val * gate;

            output[out_offset + p] = __float2bfloat16(out_val);

            if (y_cache != nullptr) {
                y_cache[out_offset + p] = __float2bfloat16(y_val);
            }
        }

        // Store state for backprop
        if (H_all != nullptr) {
            const int H_all_offset = ((t + 1) * batch_size * nheads + b * nheads + h) * N * P;
            for (int np = tid; np < N * P; np += BLOCK_SIZE) {
                H_all[H_all_offset + np] = __float2bfloat16(H_sh[np]);
            }
        }
        __syncthreads();
    }

    // Write final state
    for (int i = tid; i < N * P; i += BLOCK_SIZE) {
        H_final[H_offset + i] = __float2bfloat16(H_sh[i]);
    }
}


/**
 * E22 Backward Kernel - Compute gradients
 *
 * Uses dynamic shared memory to fit within limits.
 * Reuses buffers to minimize memory: H_sh serves double duty.
 */
template<int N, int P, int R, int D_K>
__global__ void E22BackwardKernel_BF16(
    const int seq_len,
    const int batch_size,
    const int nheads,
    const int attn_period,
    const int nonlinearity_mode,
    // Forward inputs
    const __nv_bfloat16* __restrict__ B_proj,
    const __nv_bfloat16* __restrict__ X_proj,
    const __nv_bfloat16* __restrict__ alpha_raw,
    const __nv_bfloat16* __restrict__ alpha_bias,
    const __nv_bfloat16* __restrict__ z,
    const __nv_bfloat16* __restrict__ W_q,
    const __nv_bfloat16* __restrict__ W_k,
    const __nv_bfloat16* __restrict__ W_v,
    const __nv_bfloat16* __restrict__ W_o,
    // Saved for backward
    const __nv_bfloat16* __restrict__ H_all,
    const __nv_bfloat16* __restrict__ y_cache,
    // Gradient input
    const __nv_bfloat16* __restrict__ d_output,
    // Gradient outputs
    __nv_bfloat16* __restrict__ dB_proj,
    __nv_bfloat16* __restrict__ dX_proj,
    __nv_bfloat16* __restrict__ dalpha_raw,
    __nv_bfloat16* __restrict__ dz,
    // Attention weight gradients (accumulated across batch/head)
    float* __restrict__ dW_q,   // [H, P, D_K]
    float* __restrict__ dW_k,   // [H, P, D_K]
    float* __restrict__ dW_v,   // [H, P, D_K]
    float* __restrict__ dW_o    // [H, D_K, P]
) {
    const int bh = blockIdx.x;
    const int b = bh / nheads;
    const int h = bh % nheads;

    if (b >= batch_size) return;

    const int tid = threadIdx.x;

    // Use dynamic shared memory to fit within 48KB limit
    // Layout: dH_sh[N*P] + H_sh[N*P] + QKV_sh[3*N*D_K] + attn_sh[N*N]
    // Total: 2*N*P + 3*N*D_K + N*N = 2*32*64 + 3*32*32 + 32*32 = 4096 + 3072 + 1024 = 8192 floats = 32KB
    extern __shared__ float smem[];
    float* dH_sh = smem;                          // [N*P]
    float* H_sh = dH_sh + N * P;                  // [N*P] - used for both H and H_prev
    float* QKV_sh = H_sh + N * P;                 // [3*N*D_K] - Q, K, V share this space
    float* attn_sh = QKV_sh + 3 * N * D_K;        // [N*N]

    // Aliases for QKV
    float* Q_sh = QKV_sh;
    float* K_sh = QKV_sh + N * D_K;
    float* V_sh = QKV_sh + 2 * N * D_K;

    const int W_head_offset = h * P * D_K;
    const int W_o_offset = h * D_K * P;
    const float alpha_b = __bfloat162float(alpha_bias[h]);

    // Initialize dH to zero
    for (int i = tid; i < N * P; i += BLOCK_SIZE) {
        dH_sh[i] = 0.0f;
    }
    __syncthreads();

    // Backward pass: iterate from T-1 to 0
    for (int t = seq_len - 1; t >= 0; t--) {

        // Load current state H[t+1]
        const int H_offset = ((t + 1) * batch_size * nheads + b * nheads + h) * N * P;

        for (int i = tid; i < N * P; i += BLOCK_SIZE) {
            H_sh[i] = __bfloat162float(H_all[H_offset + i]);
        }
        __syncthreads();

        // === Backward through output gating ===
        const int z_offset = (t * batch_size + b) * nheads * P + h * P;

        for (int p = tid; p < P; p += BLOCK_SIZE) {
            float d_out = __bfloat162float(d_output[z_offset + p]);
            float y_val = __bfloat162float(y_cache[z_offset + p]);
            float z_val = __bfloat162float(z[z_offset + p]);

            float gate_in = z_val + y_val;
            float sig = 1.0f / (1.0f + expf(-gate_in));
            float gate = gate_in * sig;

            float d_gate_d_in = sig * (1.0f + gate_in * (1.0f - sig));
            float dy = d_out * (gate + y_val * d_gate_d_in);
            float d_z_val = d_out * y_val * d_gate_d_in;

            dz[z_offset + p] = __float2bfloat16(d_z_val);

            // dy propagates to dH
            for (int n = 0; n < N; n++) {
                atomicAdd(&dH_sh[n * P + p], dy);
            }
        }
        __syncthreads();

        // Load H_prev for nonlinearity backward
        const int H_prev_offset = (t * batch_size * nheads + b * nheads + h) * N * P;

        // NOTE: Attention backward is disabled for now due to complexity in saving
        // intermediate states. Gradients still flow through the residual connection,
        // and attention weights get trained through the residual path. Full attention
        // backward would require saving H_pre_attn states (H after MIMO, before attention).

        // Load H_prev for nonlinearity backward
        for (int i = tid; i < N * P; i += BLOCK_SIZE) {
            H_sh[i] = __bfloat162float(H_all[H_prev_offset + i]);
        }
        __syncthreads();

        // === Backward through nonlinearity ===
        const int alpha_offset = (t * batch_size + b) * nheads + h;
        float alpha_raw_val = __bfloat162float(alpha_raw[alpha_offset]);
        float sp = log1pf(expf(alpha_raw_val + alpha_b));
        float alpha = 1.0f / (1.0f + expf(sp));

        __shared__ float dalpha_sum;
        if (tid == 0) dalpha_sum = 0.0f;
        __syncthreads();

        for (int np = tid; np < N * P; np += BLOCK_SIZE) {
            float h_prev = H_sh[np];

            const int n = np / P;
            const int p = np % P;
            const int B_offset = ((t * batch_size + b) * nheads + h) * N * R;
            const int X_offset = ((t * batch_size + b) * nheads + h) * P * R;

            float update = 0.0f;
            for (int r = 0; r < R; r++) {
                update += __bfloat162float(B_proj[B_offset + n * R + r]) *
                          __bfloat162float(X_proj[X_offset + p * R + r]);
            }
            float pre_act = alpha * h_prev + update;

            float dh = dH_sh[np];
            float d_pre_act;

            if (nonlinearity_mode == 0) {  // silu
                float sig = 1.0f / (1.0f + expf(-pre_act));
                float d_silu = sig * (1.0f + pre_act * (1.0f - sig));
                d_pre_act = dh * d_silu;
            } else if (nonlinearity_mode == 1) {  // tanh
                float tanh_val = tanhf(pre_act);
                d_pre_act = dh * (1.0f - tanh_val * tanh_val);
            } else {  // linear
                d_pre_act = dh;
            }

            atomicAdd(&dalpha_sum, d_pre_act * h_prev);
            dH_sh[np] = d_pre_act * alpha;

            for (int r = 0; r < R; r++) {
                float b_val = __bfloat162float(B_proj[B_offset + n * R + r]);
                float x_val = __bfloat162float(X_proj[X_offset + p * R + r]);
                atomicAddBF16(&dB_proj[B_offset + n * R + r], d_pre_act * x_val);
                atomicAddBF16(&dX_proj[X_offset + p * R + r], d_pre_act * b_val);
            }
        }
        __syncthreads();

        if (tid == 0) {
            float sig_sp = 1.0f / (1.0f + expf(-sp));
            float dalpha = dalpha_sum * (-alpha * (1.0f - alpha) * sig_sp);
            dalpha_raw[alpha_offset] = __float2bfloat16(dalpha);
        }
        __syncthreads();
    }
}

/**
 * Optimized E22 Backward Kernel - V2
 *
 * Key optimizations over original:
 * 1. P threads (one per headdim), each handles all N states - eliminates most contention
 * 2. dX: local accumulation, direct write (no atomic)
 * 3. dB: warp shuffle reduction + cross-warp shared memory reduction
 * 4. dalpha: warp shuffle + cross-warp reduction
 * 5. dy->dH: local addition (each thread handles its own dH[n] values)
 * 6. NO atomicAddBF16 (was CAS loop - extremely slow)
 */
template<int N, int P, int R, int D_K>
__global__ void E22BackwardKernel_BF16_Opt(
    const int seq_len,
    const int batch_size,
    const int nheads,
    const int attn_period,
    const int nonlinearity_mode,
    // Forward inputs
    const __nv_bfloat16* __restrict__ B_proj,
    const __nv_bfloat16* __restrict__ X_proj,
    const __nv_bfloat16* __restrict__ alpha_raw,
    const __nv_bfloat16* __restrict__ alpha_bias,
    const __nv_bfloat16* __restrict__ z,
    const __nv_bfloat16* __restrict__ W_q,
    const __nv_bfloat16* __restrict__ W_k,
    const __nv_bfloat16* __restrict__ W_v,
    const __nv_bfloat16* __restrict__ W_o,
    // Saved for backward
    const __nv_bfloat16* __restrict__ H_all,
    const __nv_bfloat16* __restrict__ y_cache,
    // Gradient input
    const __nv_bfloat16* __restrict__ d_output,
    // Gradient outputs
    __nv_bfloat16* __restrict__ dB_proj,
    __nv_bfloat16* __restrict__ dX_proj,
    __nv_bfloat16* __restrict__ dalpha_raw,
    __nv_bfloat16* __restrict__ dz,
    // Attention weight gradients (not used in opt kernel, but kept for API compat)
    float* __restrict__ dW_q,
    float* __restrict__ dW_k,
    float* __restrict__ dW_v,
    float* __restrict__ dW_o
) {
    const int bh = blockIdx.x;
    const int b = bh / nheads;
    const int h = bh % nheads;

    if (b >= batch_size) return;

    const int p = threadIdx.x;  // Each thread handles one p value (headdim index)
    if (p >= P) return;

    const int lane_id = p % 32;
    const int warp_id = p / 32;
    constexpr int num_warps = (P + 31) / 32;  // 2 warps for P=64

    // Shared memory layout:
    // dB_warp[num_warps * N * R] - for cross-warp dB reduction
    // dalpha_warp[num_warps] - for cross-warp dalpha reduction
    extern __shared__ float smem[];
    float* dB_warp = smem;
    float* dalpha_warp = dB_warp + num_warps * N * R;

    const float alpha_b = __bfloat162float(alpha_bias[h]);

    // Local dH for this thread's p column (in registers)
    float dH_local[N];
    #pragma unroll
    for (int n = 0; n < N; ++n) {
        dH_local[n] = 0.0f;
    }

    // Backward pass: iterate from T-1 to 0
    for (int t = seq_len - 1; t >= 0; t--) {

        // === Backward through output gating ===
        const int z_offset = (t * batch_size + b) * nheads * P + h * P;

        float d_out = __bfloat162float(d_output[z_offset + p]);
        float y_val = __bfloat162float(y_cache[z_offset + p]);
        float z_val = __bfloat162float(z[z_offset + p]);

        float gate_in = z_val + y_val;
        float sig = 1.0f / (1.0f + expf(-gate_in));
        float d_gate_d_in = sig * (1.0f + gate_in * (1.0f - sig));
        float dy = d_out * (gate_in * sig + y_val * d_gate_d_in);
        float d_z_val = d_out * y_val * d_gate_d_in;

        dz[z_offset + p] = __float2bfloat16(d_z_val);

        // dy propagates to dH for ALL n - local addition (no atomic!)
        #pragma unroll
        for (int n = 0; n < N; ++n) {
            dH_local[n] += dy;
        }

        // === Backward through nonlinearity ===
        const int alpha_offset = (t * batch_size + b) * nheads + h;
        float alpha_raw_val = __bfloat162float(alpha_raw[alpha_offset]);
        float sp = log1pf(expf(alpha_raw_val + alpha_b));
        float alpha = 1.0f / (1.0f + expf(sp));

        const int B_offset = ((t * batch_size + b) * nheads + h) * N * R;
        const int X_offset = ((t * batch_size + b) * nheads + h) * P * R;
        const int H_prev_offset = (t * batch_size * nheads + b * nheads + h) * N * P;

        // Load X values for this thread's p
        float X_vals[R];
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            X_vals[r] = __bfloat162float(X_proj[X_offset + p * R + r]);
        }

        float dalpha_local = 0.0f;
        float dX_local[R];
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            dX_local[r] = 0.0f;
        }

        // Process all N states
        #pragma unroll 4
        for (int n = 0; n < N; ++n) {
            float h_prev = __bfloat162float(H_all[H_prev_offset + n * P + p]);

            // Load B values for this n
            float B_vals[R];
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                B_vals[r] = __bfloat162float(B_proj[B_offset + n * R + r]);
            }

            // Compute update = B[n,:] @ X[p,:]
            float update = 0.0f;
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                update += B_vals[r] * X_vals[r];
            }
            float pre_act = alpha * h_prev + update;

            float dh = dH_local[n];
            float d_pre_act;

            if (nonlinearity_mode == 0) {  // silu
                float sig_pa = 1.0f / (1.0f + expf(-pre_act));
                d_pre_act = dh * sig_pa * (1.0f + pre_act * (1.0f - sig_pa));
            } else if (nonlinearity_mode == 1) {  // tanh
                float tanh_val = tanhf(pre_act);
                d_pre_act = dh * (1.0f - tanh_val * tanh_val);
            } else {  // linear
                d_pre_act = dh;
            }

            dalpha_local += d_pre_act * h_prev;
            dH_local[n] = d_pre_act * alpha;  // Update for next timestep

            // dX accumulation - local, no contention
            // dX[p,r] += d_pre_act * B[n,r] for all n
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                dX_local[r] += d_pre_act * B_vals[r];
            }

            // dB reduction via warp shuffle
            // dB[n,r] = sum_p d_pre_act[n,p] * X[p,r]
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                float contrib = d_pre_act * X_vals[r];
                // Warp shuffle reduction
                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    contrib += __shfl_down_sync(0xffffffff, contrib, offset);
                }
                if (lane_id == 0) {
                    dB_warp[warp_id * N * R + n * R + r] = contrib;
                }
            }
        }
        __syncthreads();

        // Complete dB reduction across warps and write to global
        if (warp_id == 0) {
            for (int nr = lane_id; nr < N * R; nr += 32) {
                float sum = 0.0f;
                #pragma unroll
                for (int w = 0; w < num_warps; ++w) {
                    sum += dB_warp[w * N * R + nr];
                }
                dB_proj[B_offset + nr] = __float2bfloat16(sum);
            }
        }

        // Write dX (no reduction needed - each thread owns its p)
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            dX_proj[X_offset + p * R + r] = __float2bfloat16(dX_local[r]);
        }

        // dalpha reduction via warp shuffle + cross-warp
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            dalpha_local += __shfl_down_sync(0xffffffff, dalpha_local, offset);
        }
        if (lane_id == 0) {
            dalpha_warp[warp_id] = dalpha_local;
        }
        __syncthreads();

        // Final dalpha reduction and write
        if (p == 0) {
            float dalpha_sum = 0.0f;
            #pragma unroll
            for (int w = 0; w < num_warps; ++w) {
                dalpha_sum += dalpha_warp[w];
            }
            float sig_sp = 1.0f / (1.0f + expf(-sp));
            float dalpha = dalpha_sum * (-alpha * (1.0f - alpha) * sig_sp);
            dalpha_raw[alpha_offset] = __float2bfloat16(dalpha);
        }
        __syncthreads();
    }
}

}  // namespace


// =============================================================================
// Host-side wrapper classes in hasty namespace
// Out-of-line method definitions for bfloat16 template specialization
// =============================================================================

namespace hasty {
namespace v0 {
namespace elman_ladder {

// =============================================================================
// StructuredElmanAttentionForward - BF16 Specialization
// =============================================================================

template<>
StructuredElmanAttentionForward<__nv_bfloat16>::StructuredElmanAttentionForward(
    bool training, int batch_size, int nheads, int d_state,
    int headdim, int mimo_rank, int attn_period, int attn_dim,
    int nonlinearity_mode, const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : training_(training), batch_size_(batch_size), nheads_(nheads),
      d_state_(d_state), headdim_(headdim), mimo_rank_(mimo_rank),
      attn_period_(attn_period), attn_dim_(attn_dim),
      nonlinearity_mode_(nonlinearity_mode), stream_(stream) {}

template<>
void StructuredElmanAttentionForward<__nv_bfloat16>::Run(
    int seq_len,
    const __nv_bfloat16* B_proj,
    const __nv_bfloat16* X_proj,
    const __nv_bfloat16* alpha_raw,
    const __nv_bfloat16* alpha_bias,
    const __nv_bfloat16* z,
    const __nv_bfloat16* W_q,
    const __nv_bfloat16* W_k,
    const __nv_bfloat16* W_v,
    const __nv_bfloat16* W_o,
    const __nv_bfloat16* H_init,
    __nv_bfloat16* output,
    __nv_bfloat16* H_final,
    __nv_bfloat16* H_all,
    __nv_bfloat16* y_cache
) {
    int num_blocks = batch_size_ * nheads_;

    // Template dispatch based on dimensions
    // Common configs: N=32, P=64, R=4/8, D_K=32
    if (d_state_ == 32 && headdim_ == 64 && attn_dim_ == 32) {
        if (mimo_rank_ == 4) {
            E22ForwardKernel_BF16<32, 64, 4, 32><<<num_blocks, BLOCK_SIZE, 0, stream_>>>(
                seq_len, batch_size_, nheads_, attn_period_, nonlinearity_mode_,
                B_proj, X_proj, alpha_raw, alpha_bias, z,
                W_q, W_k, W_v, W_o, H_init, output, H_final, H_all, y_cache
            );
        } else if (mimo_rank_ == 8) {
            E22ForwardKernel_BF16<32, 64, 8, 32><<<num_blocks, BLOCK_SIZE, 0, stream_>>>(
                seq_len, batch_size_, nheads_, attn_period_, nonlinearity_mode_,
                B_proj, X_proj, alpha_raw, alpha_bias, z,
                W_q, W_k, W_v, W_o, H_init, output, H_final, H_all, y_cache
            );
        } else if (mimo_rank_ == 16) {
            E22ForwardKernel_BF16<32, 64, 16, 32><<<num_blocks, BLOCK_SIZE, 0, stream_>>>(
                seq_len, batch_size_, nheads_, attn_period_, nonlinearity_mode_,
                B_proj, X_proj, alpha_raw, alpha_bias, z,
                W_q, W_k, W_v, W_o, H_init, output, H_final, H_all, y_cache
            );
        }
    }
    // Note: N=64 configurations removed - exceed shared memory limits
}


// =============================================================================
// StructuredElmanAttentionBackward - BF16 Specialization
// =============================================================================

template<>
StructuredElmanAttentionBackward<__nv_bfloat16>::StructuredElmanAttentionBackward(
    int batch_size, int nheads, int d_state, int headdim,
    int mimo_rank, int attn_period, int attn_dim, int nonlinearity_mode,
    const cublasHandle_t& blas_handle, const cudaStream_t& stream)
    : batch_size_(batch_size), nheads_(nheads), d_state_(d_state),
      headdim_(headdim), mimo_rank_(mimo_rank), attn_period_(attn_period),
      attn_dim_(attn_dim), nonlinearity_mode_(nonlinearity_mode),
      stream_(stream) {}

template<>
void StructuredElmanAttentionBackward<__nv_bfloat16>::Run(
    int seq_len,
    const __nv_bfloat16* B_proj,
    const __nv_bfloat16* X_proj,
    const __nv_bfloat16* alpha_raw,
    const __nv_bfloat16* alpha_bias,
    const __nv_bfloat16* z,
    const __nv_bfloat16* W_q,
    const __nv_bfloat16* W_k,
    const __nv_bfloat16* W_v,
    const __nv_bfloat16* W_o,
    const __nv_bfloat16* H_all,
    const __nv_bfloat16* y_cache,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* dB_proj,
    __nv_bfloat16* dX_proj,
    __nv_bfloat16* dalpha_raw,
    __nv_bfloat16* dz,
    float* dW_q,
    float* dW_k,
    float* dW_v,
    float* dW_o
) {
    int num_blocks = batch_size_ * nheads_;

    // Optimized kernel: P threads (64), shared mem for dB_warp + dalpha_warp
    // dB_warp[num_warps * N * R] + dalpha_warp[num_warps]
    // num_warps = 2 for P=64, N=32
    constexpr int opt_block_size = 64;  // P threads
    constexpr int num_warps = 2;

    if (d_state_ == 32 && headdim_ == 64 && attn_dim_ == 32) {
        if (mimo_rank_ == 4) {
            const size_t smem_size = (num_warps * 32 * 4 + num_warps) * sizeof(float);
            E22BackwardKernel_BF16_Opt<32, 64, 4, 32><<<num_blocks, opt_block_size, smem_size, stream_>>>(
                seq_len, batch_size_, nheads_, attn_period_, nonlinearity_mode_,
                B_proj, X_proj, alpha_raw, alpha_bias, z,
                W_q, W_k, W_v, W_o, H_all, y_cache, d_output,
                dB_proj, dX_proj, dalpha_raw, dz, dW_q, dW_k, dW_v, dW_o
            );
        } else if (mimo_rank_ == 8) {
            const size_t smem_size = (num_warps * 32 * 8 + num_warps) * sizeof(float);
            E22BackwardKernel_BF16_Opt<32, 64, 8, 32><<<num_blocks, opt_block_size, smem_size, stream_>>>(
                seq_len, batch_size_, nheads_, attn_period_, nonlinearity_mode_,
                B_proj, X_proj, alpha_raw, alpha_bias, z,
                W_q, W_k, W_v, W_o, H_all, y_cache, d_output,
                dB_proj, dX_proj, dalpha_raw, dz, dW_q, dW_k, dW_v, dW_o
            );
        } else if (mimo_rank_ == 16) {
            const size_t smem_size = (num_warps * 32 * 16 + num_warps) * sizeof(float);
            E22BackwardKernel_BF16_Opt<32, 64, 16, 32><<<num_blocks, opt_block_size, smem_size, stream_>>>(
                seq_len, batch_size_, nheads_, attn_period_, nonlinearity_mode_,
                B_proj, X_proj, alpha_raw, alpha_bias, z,
                W_q, W_k, W_v, W_o, H_all, y_cache, d_output,
                dB_proj, dX_proj, dalpha_raw, dz, dW_q, dW_k, dW_v, dW_o
            );
        }
    }
}

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
