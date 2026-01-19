/**
 * E87 Content-Gated Sparse Block Memory CUDA Kernel
 *
 * B blocks of n_state x n_state matrices with content-based routing.
 * Based on E75 multi-head but with:
 * - Router layer for soft top-k selection
 * - All blocks updated, but blended based on routing weights (GPU-efficient)
 * - Single shared query projection across all blocks
 * - Weighted output aggregation based on router softmax
 *
 * Forward:
 *   router_logits = W_router @ x              [B_batch, n_blocks]
 *   update_weights = softmax_topk(router_logits, k)  (sparse mask * softmax)
 *   read_weights = softmax(router_logits)     (dense for output)
 *
 *   For each block b:
 *     k_b = W_k[b] @ x, v_b = W_v[b] @ x
 *     beta_b = sigmoid(W_beta[b] @ x + b_beta[b])
 *     k_norm = k_b / ||k_b||
 *     retrieved = S_b @ k_norm
 *     delta = v_b - retrieved
 *     S_updated = tanh(beta_b * S_b + outer(delta, k_norm))
 *     S_b = (1 - w_b) * S_b + w_b * S_updated  (blend by routing weight)
 *
 *   q = W_q @ x                               (shared query)
 *   For each block b:
 *     Sq_b = S_b @ q
 *     out_b = Sq_b * silu(Sq_b)
 *   output = sum(read_weights[b] * out_b)
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include "hasty/elman_ladder.h"

#define CHECKPOINT_INTERVAL 16

namespace elman {

// ============================================================================
// Utility kernels
// ============================================================================

// Apply bias and sigmoid: data[i] = sigmoid(data[i] + bias[i % (B * n)])
__global__ void E87_AddBiasSigmoid_BF16(
    __nv_bfloat16* __restrict__ data,
    const __nv_bfloat16* __restrict__ bias,
    int n_blocks,
    int n,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // bias is [n_blocks, n], data is [T*B, n_blocks, n] (flattened)
    int bias_idx = idx % (n_blocks * n);
    float val = __bfloat162float(data[idx]) + __bfloat162float(bias[bias_idx]);
    float sig = 1.0f / (1.0f + expf(-val));
    data[idx] = __float2bfloat16(sig);
}

// Compute router softmax and top-k mask
// Input: router_logits [T*B, n_blocks]
// Output: update_weights [T*B, n_blocks] - sparse softmax (top-k only)
//         read_weights [T*B, n_blocks] - dense softmax
__global__ void E87_ComputeRoutingWeights_BF16(
    const __nv_bfloat16* __restrict__ router_logits,
    __nv_bfloat16* __restrict__ update_weights,
    __nv_bfloat16* __restrict__ read_weights,
    int n_blocks,
    int top_k,
    float router_temp,
    int T_B
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T_B) return;

    const __nv_bfloat16* logits = router_logits + idx * n_blocks;
    __nv_bfloat16* update_w = update_weights + idx * n_blocks;
    __nv_bfloat16* read_w = read_weights + idx * n_blocks;

    // Load logits and apply temperature
    float logits_f[16];  // max 16 blocks
    for (int b = 0; b < n_blocks; b++) {
        logits_f[b] = __bfloat162float(logits[b]) / router_temp;
    }

    // Find top-k indices (simple selection sort for small n_blocks)
    int topk_idx[8];  // max k=8
    float topk_val[8];
    for (int i = 0; i < top_k; i++) {
        topk_val[i] = -1e9f;
        topk_idx[i] = 0;
    }

    for (int b = 0; b < n_blocks; b++) {
        // Check if this is larger than smallest in topk
        if (logits_f[b] > topk_val[top_k - 1]) {
            // Insert in sorted order
            int pos = top_k - 1;
            while (pos > 0 && logits_f[b] > topk_val[pos - 1]) {
                topk_val[pos] = topk_val[pos - 1];
                topk_idx[pos] = topk_idx[pos - 1];
                pos--;
            }
            topk_val[pos] = logits_f[b];
            topk_idx[pos] = b;
        }
    }

    // Create top-k mask
    float mask[16];
    for (int b = 0; b < n_blocks; b++) {
        mask[b] = 0.0f;
    }
    for (int i = 0; i < top_k; i++) {
        mask[topk_idx[i]] = 1.0f;
    }

    // Compute softmax for read weights (dense)
    float max_logit = logits_f[0];
    for (int b = 1; b < n_blocks; b++) {
        max_logit = fmaxf(max_logit, logits_f[b]);
    }

    float sum_exp = 0.0f;
    float exp_vals[16];
    for (int b = 0; b < n_blocks; b++) {
        exp_vals[b] = expf(logits_f[b] - max_logit);
        sum_exp += exp_vals[b];
    }

    for (int b = 0; b < n_blocks; b++) {
        float softmax_val = exp_vals[b] / sum_exp;
        read_w[b] = __float2bfloat16(softmax_val);
    }

    // Update weights = masked softmax, renormalized
    float sum_masked = 0.0f;
    for (int b = 0; b < n_blocks; b++) {
        sum_masked += mask[b] * exp_vals[b];
    }

    for (int b = 0; b < n_blocks; b++) {
        float w = (sum_masked > 1e-8f) ? (mask[b] * exp_vals[b] / sum_masked) : 0.0f;
        update_w[b] = __float2bfloat16(w);
    }
}

// Reduce bias gradients
__global__ void E87_ReduceBiasGrad_BF16(
    const __nv_bfloat16* __restrict__ d_data,
    __nv_bfloat16* __restrict__ db,
    int n_blocks,
    int n,
    int T_B
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_blocks * n) return;

    float sum = 0.0f;
    for (int tb = 0; tb < T_B; tb++) {
        sum += __bfloat162float(d_data[tb * n_blocks * n + i]);
    }
    db[i] = __float2bfloat16(sum);
}

// Reduce router gradients
__global__ void E87_ReduceRouterGrad_BF16(
    const __nv_bfloat16* __restrict__ d_router,
    __nv_bfloat16* __restrict__ dW_router_out,
    int n_blocks,
    int T_B
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;

    float sum = 0.0f;
    for (int tb = 0; tb < T_B; tb++) {
        sum += __bfloat162float(d_router[tb * n_blocks + b]);
    }
    // This is for bias; actual dW_router is computed via GEMM
    dW_router_out[b] = __float2bfloat16(sum);
}

// Apply sigmoid derivative for beta gradients
__global__ void E87_ApplySigmoidDeriv_BF16(
    __nv_bfloat16* __restrict__ d_data,
    const __nv_bfloat16* __restrict__ sigmoid_val,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float d = __bfloat162float(d_data[idx]);
    float s = __bfloat162float(sigmoid_val[idx]);
    float d_pre = d * s * (1.0f - s);
    d_data[idx] = __float2bfloat16(d_pre);
}

// Reduce d_q across blocks: d_q_reduced[T,B,n] = sum over blocks of d_q_all[T,B,nb,n]
__global__ void E87_ReduceDq_BF16(
    const __nv_bfloat16* __restrict__ d_q_all,  // [T*B*n_blocks*n_state]
    __nv_bfloat16* __restrict__ d_q_reduced,    // [T*B*n_state]
    int n_blocks,
    int n_state,
    int T_B
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T_B * n_state;
    if (idx >= total) return;

    int n = idx % n_state;
    int tb = idx / n_state;

    float sum = 0.0f;
    for (int blk = 0; blk < n_blocks; blk++) {
        sum += __bfloat162float(d_q_all[(tb * n_blocks + blk) * n_state + n]);
    }
    d_q_reduced[idx] = __float2bfloat16(sum);
}

// ============================================================================
// E87 Forward Kernel
// Each CUDA block handles one (batch_item, block_idx) pair
// ============================================================================

template<int N_STATE>
__global__ void E87ForwardKernel_BF16(
    int T,
    int B,
    int n_blocks,
    const __nv_bfloat16* __restrict__ k_all,           // [T, B, n_blocks, N_STATE]
    const __nv_bfloat16* __restrict__ v_all,           // [T, B, n_blocks, N_STATE]
    const __nv_bfloat16* __restrict__ q_all,           // [T, B, N_STATE] (shared)
    const __nv_bfloat16* __restrict__ beta_all,        // [T, B, n_blocks, N_STATE]
    const __nv_bfloat16* __restrict__ update_weights,  // [T, B, n_blocks]
    const __nv_bfloat16* __restrict__ read_weights,    // [T, B, n_blocks]
    __nv_bfloat16* __restrict__ S,                     // [B, n_blocks, N_STATE, N_STATE]
    __nv_bfloat16* __restrict__ block_outputs,         // [T, B, n_blocks, N_STATE]
    __nv_bfloat16* __restrict__ S_checkpoints,         // [num_cp, B, n_blocks, N_STATE, N_STATE]
    __nv_bfloat16* __restrict__ Sq_cache,              // [T, B, n_blocks, N_STATE]
    int checkpoint_interval
) {
    int block_idx = blockIdx.x;
    int b_idx = block_idx / n_blocks;
    int blk = block_idx % n_blocks;
    if (b_idx >= B) return;

    extern __shared__ float shared_mem[];
    float* S_shared = shared_mem;
    float* k_shared = S_shared + N_STATE * N_STATE;
    float* v_shared = k_shared + N_STATE;
    float* q_shared = v_shared + N_STATE;
    float* retrieved = q_shared + N_STATE;
    float* beta_shared = retrieved + N_STATE;

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;

    // State offset for this (batch, block)
    int state_offset = (b_idx * n_blocks + blk) * n2;

    // Load initial state
    for (int i = tid; i < n2; i += blockDim.x) {
        S_shared[i] = __bfloat162float(S[state_offset + i]);
    }
    __syncthreads();

    // Save initial checkpoint (index 0)
    for (int i = tid; i < n2; i += blockDim.x) {
        S_checkpoints[(b_idx * n_blocks + blk) * n2 + i] = __float2bfloat16(S_shared[i]);
    }
    __syncthreads();

    for (int t = 0; t < T; t++) {
        // Data offsets
        int kv_offset = ((t * B + b_idx) * n_blocks + blk) * N_STATE;
        int q_offset = (t * B + b_idx) * N_STATE;  // Shared query
        int weight_offset = (t * B + b_idx) * n_blocks + blk;

        if (tid < N_STATE) {
            k_shared[tid] = __bfloat162float(k_all[kv_offset + tid]);
            v_shared[tid] = __bfloat162float(v_all[kv_offset + tid]);
            q_shared[tid] = __bfloat162float(q_all[q_offset + tid]);
            beta_shared[tid] = __bfloat162float(beta_all[kv_offset + tid]);
        }
        __syncthreads();

        // Get routing weight for this block
        float update_w = __bfloat162float(update_weights[weight_offset]);

        // Normalize k
        __shared__ float k_norm_sq;
        if (tid == 0) {
            k_norm_sq = 0.0f;
            for (int i = 0; i < N_STATE; i++) {
                k_norm_sq += k_shared[i] * k_shared[i];
            }
            k_norm_sq = sqrtf(k_norm_sq) + 1e-6f;
        }
        __syncthreads();
        if (tid < N_STATE) {
            k_shared[tid] /= k_norm_sq;
        }
        __syncthreads();

        // retrieved = S @ k_norm
        if (tid < N_STATE) {
            float sum = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < N_STATE; j++) {
                sum += S_shared[tid * N_STATE + j] * k_shared[j];
            }
            retrieved[tid] = sum;
        }
        __syncthreads();

        // S_updated = tanh(beta * S + outer(delta, k_norm))
        // S = (1 - w) * S + w * S_updated
        for (int i = tid; i < n2; i += blockDim.x) {
            int row = i / N_STATE;
            int col = i % N_STATE;
            float beta_val = beta_shared[row];
            float delta_i = v_shared[row] - retrieved[row];
            float update = beta_val * S_shared[i] + delta_i * k_shared[col];
            float S_updated = tanhf(update);
            // Blend based on routing weight
            S_shared[i] = (1.0f - update_w) * S_shared[i] + update_w * S_updated;
        }
        __syncthreads();

        // Save checkpoint
        if ((t + 1) % checkpoint_interval == 0) {
            int cp_idx = (t + 1) / checkpoint_interval;
            int cp_offset = (cp_idx * B * n_blocks + b_idx * n_blocks + blk) * n2;
            for (int i = tid; i < n2; i += blockDim.x) {
                S_checkpoints[cp_offset + i] = __float2bfloat16(S_shared[i]);
            }
        }
        __syncthreads();

        // Compute block output: Sq = S @ q, self-gate
        int out_offset = ((t * B + b_idx) * n_blocks + blk) * N_STATE;
        if (tid < N_STATE) {
            float Sq = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < N_STATE; j++) {
                Sq += S_shared[tid * N_STATE + j] * q_shared[j];
            }
            Sq_cache[out_offset + tid] = __float2bfloat16(Sq);

            float sig = 1.0f / (1.0f + expf(-Sq));
            float out_val = Sq * Sq * sig;  // Sq * silu(Sq)
            block_outputs[out_offset + tid] = __float2bfloat16(out_val);
        }
        __syncthreads();
    }

    // Write final state back
    for (int i = tid; i < n2; i += blockDim.x) {
        S[state_offset + i] = __float2bfloat16(S_shared[i]);
    }
}

// Aggregate block outputs with read weights
__global__ void E87_AggregateOutputs_BF16(
    const __nv_bfloat16* __restrict__ block_outputs,  // [T, B, n_blocks, n_state]
    const __nv_bfloat16* __restrict__ read_weights,   // [T, B, n_blocks]
    __nv_bfloat16* __restrict__ output,               // [T, B, n_state]
    int T,
    int B,
    int n_blocks,
    int n_state
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * B * n_state;
    if (idx >= total) return;

    int n = idx % n_state;
    int tb = idx / n_state;
    int t = tb / B;
    int b = tb % B;

    float sum = 0.0f;
    for (int blk = 0; blk < n_blocks; blk++) {
        float w = __bfloat162float(read_weights[(t * B + b) * n_blocks + blk]);
        float out_blk = __bfloat162float(block_outputs[((t * B + b) * n_blocks + blk) * n_state + n]);
        sum += w * out_blk;
    }
    output[idx] = __float2bfloat16(sum);
}

// ============================================================================
// E87 Backward Kernel
// ============================================================================

template<int N_STATE>
__global__ void E87BackwardKernel_BF16(
    int T,
    int B,
    int n_blocks,
    const __nv_bfloat16* __restrict__ k_all,
    const __nv_bfloat16* __restrict__ v_all,
    const __nv_bfloat16* __restrict__ q_all,
    const __nv_bfloat16* __restrict__ beta_all,
    const __nv_bfloat16* __restrict__ update_weights,
    const __nv_bfloat16* __restrict__ read_weights,
    const __nv_bfloat16* __restrict__ S_checkpoints,
    const __nv_bfloat16* __restrict__ Sq_cache,
    const __nv_bfloat16* __restrict__ d_block_outputs,  // [T, B, n_blocks, N_STATE]
    __nv_bfloat16* __restrict__ d_k_all,
    __nv_bfloat16* __restrict__ d_v_all,
    __nv_bfloat16* __restrict__ d_q_all,
    __nv_bfloat16* __restrict__ d_beta_all,
    __nv_bfloat16* __restrict__ d_update_weights,       // [T, B, n_blocks]
    int checkpoint_interval
) {
    int block_idx = blockIdx.x;
    int b_idx = block_idx / n_blocks;
    int blk = block_idx % n_blocks;
    if (b_idx >= B) return;

    extern __shared__ float shared_mem[];
    float* S = shared_mem;
    float* dS = S + N_STATE * N_STATE;
    float* k_raw = dS + N_STATE * N_STATE;
    float* v_raw = k_raw + N_STATE;
    float* q_raw = v_raw + N_STATE;
    float* k_norm = q_raw + N_STATE;
    float* delta = k_norm + N_STATE;
    float* retrieved = delta + N_STATE;
    float* beta = retrieved + N_STATE;
    float* d_k_raw = beta + N_STATE;
    float* d_v_raw = d_k_raw + N_STATE;
    float* d_q_raw = d_v_raw + N_STATE;
    float* d_Sq_shared = d_q_raw + N_STATE;
    float* d_delta = d_Sq_shared + N_STATE;
    float* d_k_norm = d_delta + N_STATE;

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;

    // Initialize dS to zero
    for (int i = tid; i < n2; i += blockDim.x) {
        dS[i] = 0.0f;
    }
    __syncthreads();

    int num_segments = (T + checkpoint_interval - 1) / checkpoint_interval;

    for (int seg = num_segments - 1; seg >= 0; seg--) {
        int t_start = seg * checkpoint_interval;
        int t_end = min(t_start + checkpoint_interval, T);

        for (int t = t_end - 1; t >= t_start; t--) {
            // Reload checkpoint and replay forward
            int cp_offset = (seg * B * n_blocks + b_idx * n_blocks + blk) * n2;
            for (int i = tid; i < n2; i += blockDim.x) {
                S[i] = __bfloat162float(S_checkpoints[cp_offset + i]);
            }
            __syncthreads();

            __shared__ float k_norm_val_t;
            __shared__ float update_w_t;

            // Replay forward from t_start to t
            for (int tt = t_start; tt <= t; tt++) {
                int kv_offset = ((tt * B + b_idx) * n_blocks + blk) * N_STATE;
                int q_offset = (tt * B + b_idx) * N_STATE;
                int weight_offset = (tt * B + b_idx) * n_blocks + blk;

                if (tid < N_STATE) {
                    k_raw[tid] = __bfloat162float(k_all[kv_offset + tid]);
                    v_raw[tid] = __bfloat162float(v_all[kv_offset + tid]);
                    q_raw[tid] = __bfloat162float(q_all[q_offset + tid]);
                    beta[tid] = __bfloat162float(beta_all[kv_offset + tid]);
                }
                if (tid == 0) {
                    update_w_t = __bfloat162float(update_weights[weight_offset]);
                }
                __syncthreads();

                if (tid == 0) {
                    float sum_sq = 0.0f;
                    for (int i = 0; i < N_STATE; i++) sum_sq += k_raw[i] * k_raw[i];
                    k_norm_val_t = sqrtf(sum_sq) + 1e-6f;
                }
                __syncthreads();
                if (tid < N_STATE) {
                    k_norm[tid] = k_raw[tid] / k_norm_val_t;
                }
                __syncthreads();

                if (tid < N_STATE) {
                    float sum = 0.0f;
                    for (int j = 0; j < N_STATE; j++) {
                        sum += S[tid * N_STATE + j] * k_norm[j];
                    }
                    retrieved[tid] = sum;
                    delta[tid] = v_raw[tid] - retrieved[tid];
                }
                __syncthreads();

                if (tt < t) {
                    // Apply blended update
                    for (int i = tid; i < n2; i += blockDim.x) {
                        int row = i / N_STATE;
                        int col = i % N_STATE;
                        float update = beta[row] * S[i] + delta[row] * k_norm[col];
                        float S_updated = tanhf(update);
                        S[i] = (1.0f - update_w_t) * S[i] + update_w_t * S_updated;
                    }
                    __syncthreads();
                }
            }

            // Now at timestep t with correct S state
            int kv_offset = ((t * B + b_idx) * n_blocks + blk) * N_STATE;
            int weight_offset = (t * B + b_idx) * n_blocks + blk;
            float update_w = __bfloat162float(update_weights[weight_offset]);

            // Backward through output: out = Sq * silu(Sq)
            if (tid < N_STATE) {
                float d_out = __bfloat162float(d_block_outputs[kv_offset + tid]);
                float Sq = __bfloat162float(Sq_cache[kv_offset + tid]);
                float sig = 1.0f / (1.0f + expf(-Sq));
                float d_Sq = d_out * (2.0f * Sq * sig + Sq * Sq * sig * (1.0f - sig));
                d_Sq_shared[tid] = d_Sq;
            }
            __syncthreads();

            // Compute S_t (the state after blended update at time t)
            // S_t = (1 - w) * S + w * tanh(beta * S + outer(delta, k_norm))
            // dS contribution from output
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                dS[i] += d_Sq_shared[row] * q_raw[col];
            }
            __syncthreads();

            // d_q = S_t^T @ d_Sq
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    // Compute S_t[i, tid]
                    float update = beta[i] * S[i * N_STATE + tid] + delta[i] * k_norm[tid];
                    float S_updated = tanhf(update);
                    float S_t_ij = (1.0f - update_w) * S[i * N_STATE + tid] + update_w * S_updated;
                    sum += S_t_ij * d_Sq_shared[i];
                }
                d_q_raw[tid] = sum;
            }
            __syncthreads();

            // Backward through blended update:
            // S_t = (1-w)*S + w*S_updated, where S_updated = tanh(beta*S + outer(delta, k))
            // dS_prev = (1-w)*dS + w * dS_updated * (1-S_updated^2) * beta
            // d_update_w = sum(dS * (S_updated - S))

            __shared__ float d_update_w_local;
            if (tid == 0) d_update_w_local = 0.0f;
            __syncthreads();

            // First compute d_update_w
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                float update = beta[row] * S[i] + delta[row] * k_norm[col];
                float S_updated = tanhf(update);
                atomicAdd(&d_update_w_local, dS[i] * (S_updated - S[i]));
            }
            __syncthreads();

            if (tid == 0) {
                d_update_weights[weight_offset] = __float2bfloat16(d_update_w_local);
            }

            // Backward through delta rule inside the blended update
            if (tid < N_STATE) {
                float d_delta_local = 0.0f;
                float d_beta_local = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    float update = beta[tid] * S[tid * N_STATE + j] + delta[tid] * k_norm[j];
                    float S_updated = tanhf(update);
                    float d_pre = dS[tid * N_STATE + j] * update_w * (1.0f - S_updated * S_updated);
                    d_delta_local += d_pre * k_norm[j];
                    d_beta_local += d_pre * S[tid * N_STATE + j];
                }
                d_delta[tid] = d_delta_local;
                d_beta_all[kv_offset + tid] = __float2bfloat16(d_beta_local);
            }
            __syncthreads();

            // d_k_norm contribution from delta rule
            if (tid < N_STATE) {
                float d_k_norm_local = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    float update = beta[i] * S[i * N_STATE + tid] + delta[i] * k_norm[tid];
                    float S_updated = tanhf(update);
                    float d_pre = dS[i * N_STATE + tid] * update_w * (1.0f - S_updated * S_updated);
                    d_k_norm_local += d_pre * delta[i];
                }
                d_k_norm[tid] = d_k_norm_local;
            }
            __syncthreads();

            // d_v = d_delta
            if (tid < N_STATE) {
                d_v_raw[tid] = d_delta[tid];
            }

            // d_k_norm from retrieved = S @ k_norm
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    sum += S[i * N_STATE + tid] * (-d_delta[i]);
                }
                d_k_norm[tid] += sum;
            }
            __syncthreads();

            // d_k_raw from normalization
            {
                __shared__ float k_dot_dk;
                if (tid == 0) {
                    k_dot_dk = 0.0f;
                    for (int i = 0; i < N_STATE; i++) {
                        k_dot_dk += k_raw[i] * d_k_norm[i];
                    }
                }
                __syncthreads();

                if (tid < N_STATE) {
                    float norm = k_norm_val_t;
                    float norm3 = norm * norm * norm;
                    d_k_raw[tid] = d_k_norm[tid] / norm - k_raw[tid] * k_dot_dk / norm3;
                }
                __syncthreads();
            }

            // Write gradients
            // Note: d_q_all is [T, B, n_blocks, N_STATE] - we write per-block, reduce later
            int d_q_offset = ((t * B + b_idx) * n_blocks + blk) * N_STATE;
            if (tid < N_STATE) {
                d_k_all[kv_offset + tid] = __float2bfloat16(d_k_raw[tid]);
                d_v_all[kv_offset + tid] = __float2bfloat16(d_v_raw[tid]);
                d_q_all[d_q_offset + tid] = __float2bfloat16(d_q_raw[tid]);
            }
            __syncthreads();

            // Update dS for next iteration (earlier timestep)
            // dS_prev contribution from blended update
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                float update = beta[row] * S[i] + delta[row] * k_norm[col];
                float S_updated = tanhf(update);
                float d_pre = dS[i] * update_w * (1.0f - S_updated * S_updated);
                // dS from (1-w)*S term plus dS from beta*S in delta rule
                dS[i] = dS[i] * (1.0f - update_w) + d_pre * beta[row] + (-d_delta[row]) * k_norm[col];
            }
            __syncthreads();
        }
    }
}

// Backward through output aggregation (distribute gradients to blocks)
__global__ void E87_BackwardAggregation_BF16(
    const __nv_bfloat16* __restrict__ d_output,        // [T, B, n_state]
    const __nv_bfloat16* __restrict__ read_weights,   // [T, B, n_blocks]
    const __nv_bfloat16* __restrict__ block_outputs,  // [T, B, n_blocks, n_state]
    __nv_bfloat16* __restrict__ d_block_outputs,      // [T, B, n_blocks, n_state]
    __nv_bfloat16* __restrict__ d_read_weights,       // [T, B, n_blocks]
    int T,
    int B,
    int n_blocks,
    int n_state
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * B * n_blocks * n_state;
    if (idx >= total) return;

    int n = idx % n_state;
    int rest = idx / n_state;
    int blk = rest % n_blocks;
    int tb = rest / n_blocks;

    float d_out = __bfloat162float(d_output[tb * n_state + n]);
    float w = __bfloat162float(read_weights[tb * n_blocks + blk]);

    // d_block_output = read_weight * d_output
    d_block_outputs[idx] = __float2bfloat16(w * d_out);

    // d_read_weight contribution (only for n=0 to avoid duplicate accumulation)
    if (n == 0) {
        float d_w = 0.0f;
        for (int nn = 0; nn < n_state; nn++) {
            float out_blk = __bfloat162float(block_outputs[tb * n_blocks * n_state + blk * n_state + nn]);
            d_w += __bfloat162float(d_output[tb * n_state + nn]) * out_blk;
        }
        d_read_weights[tb * n_blocks + blk] = __float2bfloat16(d_w);
    }
}

// Backward through routing weights (softmax)
__global__ void E87_BackwardRouting_BF16(
    const __nv_bfloat16* __restrict__ d_update_weights,
    const __nv_bfloat16* __restrict__ d_read_weights,
    const __nv_bfloat16* __restrict__ update_weights,
    const __nv_bfloat16* __restrict__ read_weights,
    __nv_bfloat16* __restrict__ d_router_logits,
    int n_blocks,
    int top_k,
    float router_temp,
    int T_B
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T_B) return;

    // Load weights and gradients
    float update_w[16], read_w[16];
    float d_update_w[16], d_read_w[16];
    for (int b = 0; b < n_blocks; b++) {
        update_w[b] = __bfloat162float(update_weights[idx * n_blocks + b]);
        read_w[b] = __bfloat162float(read_weights[idx * n_blocks + b]);
        d_update_w[b] = __bfloat162float(d_update_weights[idx * n_blocks + b]);
        d_read_w[b] = __bfloat162float(d_read_weights[idx * n_blocks + b]);
    }

    // Backward through read softmax
    float sum_d_read = 0.0f;
    for (int b = 0; b < n_blocks; b++) {
        sum_d_read += d_read_w[b] * read_w[b];
    }

    float d_logits[16];
    for (int b = 0; b < n_blocks; b++) {
        d_logits[b] = read_w[b] * (d_read_w[b] - sum_d_read);
    }

    // Backward through update softmax (sparse)
    // This is trickier - we have masked softmax with renormalization
    // For simplicity, treat it similarly but scaled by the mask
    float sum_d_update = 0.0f;
    for (int b = 0; b < n_blocks; b++) {
        if (update_w[b] > 1e-8f) {
            sum_d_update += d_update_w[b] * update_w[b];
        }
    }

    for (int b = 0; b < n_blocks; b++) {
        if (update_w[b] > 1e-8f) {
            d_logits[b] += update_w[b] * (d_update_w[b] - sum_d_update);
        }
    }

    // Apply temperature
    for (int b = 0; b < n_blocks; b++) {
        d_router_logits[idx * n_blocks + b] = __float2bfloat16(d_logits[b] / router_temp);
    }
}

// ============================================================================
// E87SparseBlockForward Implementation
// ============================================================================

template<typename DataT>
E87SparseBlockForward<DataT>::E87SparseBlockForward(
    bool training,
    int batch_size,
    int n_state,
    int n_blocks,
    int top_k,
    int dim,
    float router_temp,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : training_(training),
      batch_size_(batch_size),
      n_state_(n_state),
      n_blocks_(n_blocks),
      top_k_(top_k),
      dim_(dim),
      router_temp_(router_temp),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename DataT>
void E87SparseBlockForward<DataT>::Run(
    int steps,
    const DataT* W_router,   // [n_blocks, dim]
    const DataT* W_k,        // [n_blocks * n_state, dim]
    const DataT* W_v,        // [n_blocks * n_state, dim]
    const DataT* W_q,        // [n_state, dim] (shared)
    const DataT* W_beta,     // [n_blocks * n_state, dim]
    const DataT* b_beta,     // [n_blocks, n_state]
    const DataT* x,          // [T, B, dim]
    DataT* S,                // [B, n_blocks, n_state, n_state]
    DataT* output,           // [T, B, n_state]
    DataT* router_cache,     // [T, B, n_blocks]
    DataT* k_cache,          // [T, B, n_blocks, n_state]
    DataT* v_cache,
    DataT* q_cache,          // [T, B, n_state] (shared)
    DataT* beta_cache,
    DataT* update_weights,   // [T, B, n_blocks]
    DataT* read_weights,     // [T, B, n_blocks]
    DataT* S_cache           // checkpoints + Sq_cache + block_outputs
) {
    int T = steps;
    int B = batch_size_;
    int n = n_state_;
    int nb = n_blocks_;
    int d = dim_;

    const float alpha = 1.0f, beta_zero = 0.0f;

    // 1. Compute router logits: W_router @ x -> [T*B, n_blocks]
    cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        nb, T * B, d, &alpha,
        W_router, CUDA_R_16BF, d, x, CUDA_R_16BF, d,
        &beta_zero, router_cache, CUDA_R_16BF, nb,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // 2. Compute routing weights (top-k sparse update, dense read)
    int routing_threads = 256;
    int routing_blocks = (T * B + routing_threads - 1) / routing_threads;
    E87_ComputeRoutingWeights_BF16<<<routing_blocks, routing_threads, 0, stream_>>>(
        (const __nv_bfloat16*)router_cache,
        (__nv_bfloat16*)update_weights,
        (__nv_bfloat16*)read_weights,
        nb, top_k_, router_temp_, T * B);

    // 3. Project k, v for all blocks: [T*B, n_blocks * n_state]
    cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        nb * n, T * B, d, &alpha,
        W_k, CUDA_R_16BF, d, x, CUDA_R_16BF, d,
        &beta_zero, k_cache, CUDA_R_16BF, nb * n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        nb * n, T * B, d, &alpha,
        W_v, CUDA_R_16BF, d, x, CUDA_R_16BF, d,
        &beta_zero, v_cache, CUDA_R_16BF, nb * n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // 4. Project shared q: [T*B, n_state]
    cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        n, T * B, d, &alpha,
        W_q, CUDA_R_16BF, d, x, CUDA_R_16BF, d,
        &beta_zero, q_cache, CUDA_R_16BF, n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // 5. Project beta for all blocks
    cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        nb * n, T * B, d, &alpha,
        W_beta, CUDA_R_16BF, d, x, CUDA_R_16BF, d,
        &beta_zero, beta_cache, CUDA_R_16BF, nb * n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // Apply bias and sigmoid to beta
    int total_beta = T * B * nb * n;
    int threads = 256;
    int blocks = (total_beta + threads - 1) / threads;
    E87_AddBiasSigmoid_BF16<<<blocks, threads, 0, stream_>>>(
        (__nv_bfloat16*)beta_cache, (const __nv_bfloat16*)b_beta, nb, n, total_beta);

    // Workspace layout
    int num_checkpoints = (T + CHECKPOINT_INTERVAL - 1) / CHECKPOINT_INTERVAL + 1;
    DataT* s_checkpoints = S_cache;
    DataT* sq_cache = S_cache + num_checkpoints * B * nb * n * n;
    DataT* block_outputs = sq_cache + T * B * nb * n;

    // 6. Run forward kernel - one CUDA block per (batch, memory_block)
    int shared_size = (n * n + 6 * n) * sizeof(float);
    int kernel_threads = min(256, n * n);
    int num_cuda_blocks = B * nb;

    #define DISPATCH_E87_FORWARD(N) \
        E87ForwardKernel_BF16<N><<<num_cuda_blocks, kernel_threads, shared_size, stream_>>>( \
            T, B, nb, \
            (const __nv_bfloat16*)k_cache, \
            (const __nv_bfloat16*)v_cache, \
            (const __nv_bfloat16*)q_cache, \
            (const __nv_bfloat16*)beta_cache, \
            (const __nv_bfloat16*)update_weights, \
            (const __nv_bfloat16*)read_weights, \
            (__nv_bfloat16*)S, \
            (__nv_bfloat16*)block_outputs, \
            (__nv_bfloat16*)s_checkpoints, \
            (__nv_bfloat16*)sq_cache, \
            CHECKPOINT_INTERVAL)

    if (n == 8) { DISPATCH_E87_FORWARD(8); }
    else if (n == 16) { DISPATCH_E87_FORWARD(16); }
    else if (n == 24) { DISPATCH_E87_FORWARD(24); }
    else if (n == 32) { DISPATCH_E87_FORWARD(32); }
    else if (n == 48) { DISPATCH_E87_FORWARD(48); }
    else if (n == 64) { DISPATCH_E87_FORWARD(64); }
    else {
        fprintf(stderr, "E87 Forward: unsupported n_state=%d\n", n);
    }

    #undef DISPATCH_E87_FORWARD

    // 7. Aggregate block outputs with read weights
    int agg_threads = 256;
    int agg_blocks = (T * B * n + agg_threads - 1) / agg_threads;
    E87_AggregateOutputs_BF16<<<agg_blocks, agg_threads, 0, stream_>>>(
        (const __nv_bfloat16*)block_outputs,
        (const __nv_bfloat16*)read_weights,
        (__nv_bfloat16*)output,
        T, B, nb, n);
}

template struct E87SparseBlockForward<__nv_bfloat16>;

// ============================================================================
// E87SparseBlockBackward Implementation
// ============================================================================

template<typename DataT>
E87SparseBlockBackward<DataT>::E87SparseBlockBackward(
    int batch_size,
    int n_state,
    int n_blocks,
    int top_k,
    int dim,
    float router_temp,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      n_state_(n_state),
      n_blocks_(n_blocks),
      top_k_(top_k),
      dim_(dim),
      router_temp_(router_temp),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename DataT>
void E87SparseBlockBackward<DataT>::Run(
    int steps,
    const DataT* W_router,
    const DataT* W_k,
    const DataT* W_v,
    const DataT* W_q,
    const DataT* W_beta,
    const DataT* x,
    const DataT* S_checkpoints,
    const DataT* Sq_cache,
    const DataT* block_outputs_cache,
    const DataT* router_cache,
    const DataT* k_cache,
    const DataT* v_cache,
    const DataT* q_cache,
    const DataT* beta_cache,
    const DataT* update_weights,
    const DataT* read_weights,
    const DataT* d_output,
    DataT* dx,
    DataT* dW_router,
    DataT* dW_k,
    DataT* dW_v,
    DataT* dW_q,
    DataT* dW_beta,
    DataT* db_beta,
    DataT* workspace
) {
    int T = steps;
    int B = batch_size_;
    int n = n_state_;
    int nb = n_blocks_;
    int d = dim_;

    // Workspace layout:
    // [d_k: T*B*nb*n] [d_v: T*B*nb*n] [d_q_perblock: T*B*nb*n] [d_q_reduced: T*B*n]
    // [d_beta: T*B*nb*n] [d_router: T*B*nb] [d_block_outputs: T*B*nb*n]
    // [d_update_weights: T*B*nb] [d_read_weights: T*B*nb]
    DataT* d_k_all = workspace;
    DataT* d_v_all = d_k_all + T * B * nb * n;
    DataT* d_q_perblock = d_v_all + T * B * nb * n;      // [T, B, nb, n] - per-block d_q
    DataT* d_q_reduced = d_q_perblock + T * B * nb * n;  // [T, B, n] - reduced d_q
    DataT* d_beta_all = d_q_reduced + T * B * n;
    DataT* d_router_all = d_beta_all + T * B * nb * n;
    DataT* d_block_outputs = d_router_all + T * B * nb;
    DataT* d_update_weights = d_block_outputs + T * B * nb * n;
    DataT* d_read_weights = d_update_weights + T * B * nb;

    // 1. Backward through output aggregation
    int agg_threads = 256;
    int agg_blocks = (T * B * nb * n + agg_threads - 1) / agg_threads;
    E87_BackwardAggregation_BF16<<<agg_blocks, agg_threads, 0, stream_>>>(
        (const __nv_bfloat16*)d_output,
        (const __nv_bfloat16*)read_weights,
        (const __nv_bfloat16*)block_outputs_cache,
        (__nv_bfloat16*)d_block_outputs,
        (__nv_bfloat16*)d_read_weights,
        T, B, nb, n);

    // 2. Run backward kernel for each block
    int shared_size = (2 * n * n + 13 * n) * sizeof(float);
    int threads = min(256, n * n);
    int num_cuda_blocks = B * nb;

    #define DISPATCH_E87_BACKWARD(N) \
        E87BackwardKernel_BF16<N><<<num_cuda_blocks, threads, shared_size, stream_>>>( \
            T, B, nb, \
            (const __nv_bfloat16*)k_cache, \
            (const __nv_bfloat16*)v_cache, \
            (const __nv_bfloat16*)q_cache, \
            (const __nv_bfloat16*)beta_cache, \
            (const __nv_bfloat16*)update_weights, \
            (const __nv_bfloat16*)read_weights, \
            (const __nv_bfloat16*)S_checkpoints, \
            (const __nv_bfloat16*)Sq_cache, \
            (const __nv_bfloat16*)d_block_outputs, \
            (__nv_bfloat16*)d_k_all, \
            (__nv_bfloat16*)d_v_all, \
            (__nv_bfloat16*)d_q_perblock, \
            (__nv_bfloat16*)d_beta_all, \
            (__nv_bfloat16*)d_update_weights, \
            CHECKPOINT_INTERVAL)

    if (n == 8) { DISPATCH_E87_BACKWARD(8); }
    else if (n == 16) { DISPATCH_E87_BACKWARD(16); }
    else if (n == 24) { DISPATCH_E87_BACKWARD(24); }
    else if (n == 32) { DISPATCH_E87_BACKWARD(32); }
    else if (n == 48) { DISPATCH_E87_BACKWARD(48); }
    else if (n == 64) { DISPATCH_E87_BACKWARD(64); }
    else {
        fprintf(stderr, "E87 Backward: unsupported n_state=%d\n", n);
    }

    #undef DISPATCH_E87_BACKWARD

    // 2b. Reduce d_q across blocks: d_q_perblock[T,B,nb,n] -> d_q_reduced[T,B,n]
    int reduce_threads = 256;
    int reduce_blocks = (T * B * n + reduce_threads - 1) / reduce_threads;
    E87_ReduceDq_BF16<<<reduce_blocks, reduce_threads, 0, stream_>>>(
        (const __nv_bfloat16*)d_q_perblock,
        (__nv_bfloat16*)d_q_reduced,
        nb, n, T * B);

    // 3. Backward through routing weights
    int routing_threads = 256;
    int routing_blocks_num = (T * B + routing_threads - 1) / routing_threads;
    E87_BackwardRouting_BF16<<<routing_blocks_num, routing_threads, 0, stream_>>>(
        (const __nv_bfloat16*)d_update_weights,
        (const __nv_bfloat16*)d_read_weights,
        (const __nv_bfloat16*)update_weights,
        (const __nv_bfloat16*)read_weights,
        (__nv_bfloat16*)d_router_all,
        nb, top_k_, router_temp_, T * B);

    // 4. Apply sigmoid derivative to beta gradients
    int total_beta = T * B * nb * n;
    int threads_deriv = 256;
    int blocks_deriv = (total_beta + threads_deriv - 1) / threads_deriv;
    E87_ApplySigmoidDeriv_BF16<<<blocks_deriv, threads_deriv, 0, stream_>>>(
        (__nv_bfloat16*)d_beta_all, (const __nv_bfloat16*)beta_cache, total_beta);

    // 5. Compute weight gradients
    const float alpha = 1.0f, beta_zero = 0.0f;

    // dW_router: [d, nb] = x.T @ d_router
    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        d, nb, T * B, &alpha,
        x, CUDA_R_16BF, d, d_router_all, CUDA_R_16BF, nb,
        &beta_zero, dW_router, CUDA_R_16BF, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // dW_k: [d, nb*n] = x.T @ d_k_all
    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        d, nb * n, T * B, &alpha,
        x, CUDA_R_16BF, d, d_k_all, CUDA_R_16BF, nb * n,
        &beta_zero, dW_k, CUDA_R_16BF, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // dW_v
    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        d, nb * n, T * B, &alpha,
        x, CUDA_R_16BF, d, d_v_all, CUDA_R_16BF, nb * n,
        &beta_zero, dW_v, CUDA_R_16BF, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // dW_q (shared): [d, n] = x.T @ d_q_reduced
    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        d, n, T * B, &alpha,
        x, CUDA_R_16BF, d, d_q_reduced, CUDA_R_16BF, n,
        &beta_zero, dW_q, CUDA_R_16BF, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // dW_beta
    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        d, nb * n, T * B, &alpha,
        x, CUDA_R_16BF, d, d_beta_all, CUDA_R_16BF, nb * n,
        &beta_zero, dW_beta, CUDA_R_16BF, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // db_beta = sum over T*B
    int threads_db = 256;
    int blocks_db = (nb * n + threads_db - 1) / threads_db;
    E87_ReduceBiasGrad_BF16<<<blocks_db, threads_db, 0, stream_>>>(
        (const __nv_bfloat16*)d_beta_all, (__nv_bfloat16*)db_beta, nb, n, T * B);

    // 6. Compute dx
    // dx = W_router @ d_router + W_k @ d_k + W_v @ d_v + W_q @ d_q + W_beta @ d_beta
    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
        d, T * B, nb, &alpha,
        W_router, CUDA_R_16BF, d, d_router_all, CUDA_R_16BF, nb,
        &beta_zero, dx, CUDA_R_16BF, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    const float alpha_add = 1.0f, beta_add = 1.0f;
    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
        d, T * B, nb * n, &alpha_add,
        W_k, CUDA_R_16BF, d, d_k_all, CUDA_R_16BF, nb * n,
        &beta_add, dx, CUDA_R_16BF, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
        d, T * B, nb * n, &alpha_add,
        W_v, CUDA_R_16BF, d, d_v_all, CUDA_R_16BF, nb * n,
        &beta_add, dx, CUDA_R_16BF, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
        d, T * B, n, &alpha_add,
        W_q, CUDA_R_16BF, d, d_q_reduced, CUDA_R_16BF, n,
        &beta_add, dx, CUDA_R_16BF, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
        d, T * B, nb * n, &alpha_add,
        W_beta, CUDA_R_16BF, d, d_beta_all, CUDA_R_16BF, nb * n,
        &beta_add, dx, CUDA_R_16BF, d,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

template struct E87SparseBlockBackward<__nv_bfloat16>;

}  // namespace elman
