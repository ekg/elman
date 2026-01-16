/**
 * E74 Full Matrix State Fused CUDA Kernel with Gradient Checkpointing
 *
 * Processes ALL timesteps in SINGLE kernel launch for full matrix state.
 * State S is [B, n_state, n_state] - O(n²) memory per batch item.
 *
 * E74 DELTA RULE (NOT E70's decay!):
 * Update: S = tanh(S + outer(v - S@k, k))  -- erase before write
 * Output: (S @ q) * silu(S @ q)
 *
 * Gradient Checkpointing:
 * - Forward: Save checkpoints every CHECKPOINT_INTERVAL steps
 * - Backward: Recompute intermediate states from checkpoints
 * - Memory: O(T/K × n²) instead of O(T × n²)
 *
 * Optimization: For small n_state (32-96), keep state in shared memory.
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
// Forward Kernel - Full Matrix State with Checkpointing (all projection types)
// proj_type: 0=tied_kvq (k=v=q), 1=tied_kq (k=q, v separate), 2=no_z (k,v,q separate)
// USE_TANH: 0=linear (identity), 1=tanh nonlinearity
// ============================================================================

template<int N_STATE, int PROJ_TYPE, int USE_TANH>
__global__ void E74FullMatrixForwardKernel_BF16_General(
    int T,
    int B,
    const __nv_bfloat16* __restrict__ k_all,  // [T, B, N_STATE]
    const __nv_bfloat16* __restrict__ v_all,  // [T, B, N_STATE] (nullptr for tied_kvq)
    const __nv_bfloat16* __restrict__ q_all,  // [T, B, N_STATE] (nullptr for tied_kvq/tied_kq)
    __nv_bfloat16* __restrict__ S,            // [B, N_STATE, N_STATE] state matrix (in/out)
    __nv_bfloat16* __restrict__ output,       // [T, B, N_STATE] output
    __nv_bfloat16* __restrict__ S_checkpoints,// [num_checkpoints, B, N_STATE, N_STATE]
    __nv_bfloat16* __restrict__ Sq_cache,     // [T, B, N_STATE] cache Sq for backward
    int checkpoint_interval
) {
    int b = blockIdx.x;
    if (b >= B) return;

    // Shared memory for state matrix and vectors
    extern __shared__ float shared_mem[];
    float* S_shared = shared_mem;                    // [N_STATE * N_STATE]
    float* k_shared = S_shared + N_STATE * N_STATE;  // [N_STATE]
    float* v_shared = k_shared + N_STATE;            // [N_STATE]
    float* q_shared = v_shared + N_STATE;            // [N_STATE]
    float* retrieved = q_shared + N_STATE;           // [N_STATE]

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;

    // Load initial state into shared memory
    for (int i = tid; i < n2; i += blockDim.x) {
        S_shared[i] = __bfloat162float(S[b * n2 + i]);
    }
    __syncthreads();

    // Save initial state as checkpoint 0
    for (int i = tid; i < n2; i += blockDim.x) {
        S_checkpoints[0 * B * n2 + b * n2 + i] = __float2bfloat16(S_shared[i]);
    }

    // Process each timestep
    for (int t = 0; t < T; t++) {
        // Load k, v, q based on projection type
        if (tid < N_STATE) {
            k_shared[tid] = __bfloat162float(k_all[t * B * N_STATE + b * N_STATE + tid]);

            if (PROJ_TYPE == 0) {
                // tied_kvq: k = v = q
                v_shared[tid] = k_shared[tid];
                q_shared[tid] = k_shared[tid];
            } else if (PROJ_TYPE == 1) {
                // tied_kq: k = q, v separate
                v_shared[tid] = __bfloat162float(v_all[t * B * N_STATE + b * N_STATE + tid]);
                q_shared[tid] = k_shared[tid];
            } else {
                // no_z: k, v, q all separate
                v_shared[tid] = __bfloat162float(v_all[t * B * N_STATE + b * N_STATE + tid]);
                q_shared[tid] = __bfloat162float(q_all[t * B * N_STATE + b * N_STATE + tid]);
            }
        }
        __syncthreads();

        // CRITICAL: Normalize k (Python does this, CUDA was missing it!)
        // k_norm = k / (||k|| + eps)
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

        // Compute retrieved = S @ k (matrix-vector multiply)
        if (tid < N_STATE) {
            float sum = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < N_STATE; j++) {
                sum += S_shared[tid * N_STATE + j] * k_shared[j];
            }
            retrieved[tid] = sum;
        }
        __syncthreads();

        // Update state: S = f(S + outer(v - retrieved, k)) where f = tanh or identity
        for (int i = tid; i < n2; i += blockDim.x) {
            int row = i / N_STATE;
            int col = i % N_STATE;
            float delta_i = v_shared[row] - retrieved[row];
            float update = S_shared[i] + delta_i * k_shared[col];
            if (USE_TANH) {
                S_shared[i] = tanhf(update);
            } else {
                S_shared[i] = update;  // Linear: identity function
            }
        }
        __syncthreads();

        // Save checkpoint if at checkpoint boundary
        if ((t + 1) % checkpoint_interval == 0) {
            int cp_idx = (t + 1) / checkpoint_interval;
            for (int i = tid; i < n2; i += blockDim.x) {
                S_checkpoints[cp_idx * B * n2 + b * n2 + i] = __float2bfloat16(S_shared[i]);
            }
        }

        // Compute output: Sq = S @ q, then Sq * silu(Sq)
        if (tid < N_STATE) {
            float Sq = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < N_STATE; j++) {
                Sq += S_shared[tid * N_STATE + j] * q_shared[j];
            }
            // Cache Sq for backward
            Sq_cache[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(Sq);
            // Self-gating: Sq * silu(Sq) = Sq * Sq * sigmoid(Sq)
            float sig = 1.0f / (1.0f + expf(-Sq));
            output[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(Sq * Sq * sig);
        }
        __syncthreads();
    }

    // Write final state back
    for (int i = tid; i < n2; i += blockDim.x) {
        S[b * n2 + i] = __float2bfloat16(S_shared[i]);
    }
}

// ============================================================================
// Legacy Forward Kernel - tied_kvq only (kept for compatibility)
// ============================================================================

template<int N_STATE>
__global__ void E74FullMatrixForwardKernel_BF16(
    int T,
    int B,
    const __nv_bfloat16* __restrict__ k_all,  // [T, B, N_STATE] pre-computed k=v=q
    __nv_bfloat16* __restrict__ S,            // [B, N_STATE, N_STATE] state matrix (in/out)
    __nv_bfloat16* __restrict__ output,       // [T, B, N_STATE] output
    __nv_bfloat16* __restrict__ S_checkpoints,// [num_checkpoints, B, N_STATE, N_STATE]
    __nv_bfloat16* __restrict__ Sq_cache,     // [T, B, N_STATE] cache Sq for backward
    int checkpoint_interval
) {
    int b = blockIdx.x;
    if (b >= B) return;

    // Shared memory for state matrix and vectors
    extern __shared__ float shared_mem[];
    float* S_shared = shared_mem;                    // [N_STATE * N_STATE]
    float* k_shared = S_shared + N_STATE * N_STATE;  // [N_STATE]
    float* retrieved = k_shared + N_STATE;           // [N_STATE]

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;
    int num_checkpoints = (T + checkpoint_interval - 1) / checkpoint_interval + 1;

    // Load initial state into shared memory
    for (int i = tid; i < n2; i += blockDim.x) {
        S_shared[i] = __bfloat162float(S[b * n2 + i]);
    }
    __syncthreads();

    // Save initial state as checkpoint 0
    for (int i = tid; i < n2; i += blockDim.x) {
        S_checkpoints[0 * B * n2 + b * n2 + i] = __float2bfloat16(S_shared[i]);
    }

    // Process each timestep
    for (int t = 0; t < T; t++) {
        // Load k (= v = q for tied_kvq)
        if (tid < N_STATE) {
            k_shared[tid] = __bfloat162float(k_all[t * B * N_STATE + b * N_STATE + tid]);
        }
        __syncthreads();

        // CRITICAL: Normalize k (Python does this, CUDA was missing it!)
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

        // Compute retrieved = S @ k (matrix-vector multiply)
        if (tid < N_STATE) {
            float sum = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < N_STATE; j++) {
                sum += S_shared[tid * N_STATE + j] * k_shared[j];
            }
            retrieved[tid] = sum;
        }
        __syncthreads();

        // Update state: S = tanh(S + outer(v - retrieved, k))
        // For tied_kvq: v = k (note: using normalized k)
        for (int i = tid; i < n2; i += blockDim.x) {
            int row = i / N_STATE;
            int col = i % N_STATE;
            float v_i = k_shared[row];  // v = k for tied_kvq
            float delta_i = v_i - retrieved[row];
            float update = S_shared[i] + delta_i * k_shared[col];
            S_shared[i] = tanhf(update);
        }
        __syncthreads();

        // Save checkpoint if at checkpoint boundary
        if ((t + 1) % checkpoint_interval == 0) {
            int cp_idx = (t + 1) / checkpoint_interval;
            for (int i = tid; i < n2; i += blockDim.x) {
                S_checkpoints[cp_idx * B * n2 + b * n2 + i] = __float2bfloat16(S_shared[i]);
            }
        }

        // Compute output: Sq = S @ q, then Sq * silu(Sq)
        // For tied_kvq: q = k
        if (tid < N_STATE) {
            float Sq = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < N_STATE; j++) {
                Sq += S_shared[tid * N_STATE + j] * k_shared[j];
            }
            // Cache Sq for backward
            Sq_cache[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(Sq);
            // Self-gating: Sq * silu(Sq) = Sq * Sq * sigmoid(Sq)
            float sig = 1.0f / (1.0f + expf(-Sq));
            output[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(Sq * Sq * sig);
        }
        __syncthreads();
    }

    // Write final state back
    for (int i = tid; i < n2; i += blockDim.x) {
        S[b * n2 + i] = __float2bfloat16(S_shared[i]);
    }
}


// ============================================================================
// Backward Kernel - Full Matrix State with Checkpointing (tied_kvq)
// CORRECT implementation following the gradient derivation:
//
// Forward: k_norm = k_raw / ||k_raw||
//          retrieved = S_prev @ k_norm
//          delta = k_raw - retrieved  (v = k_raw for tied_kvq)
//          S_curr = tanh(S_prev + outer(delta, k_norm))
//          Sq = S_curr @ k_raw  (q = k_raw for tied_kvq)
//          out = Sq * Sq * sigmoid(Sq)
//
// Backward: For each t from T-1 to 0:
//   1. d_Sq = d_out * (2*Sq*sig + Sq²*sig*(1-sig))
//   2. dS += outer(d_Sq, k_raw)  // from Sq = S @ q
//   3. d_q = S^T @ d_Sq
//   4. d_pre_tanh = dS * (1 - S_curr²)  // tanh gradient
//   5. dS_prev = d_pre_tanh  // from S_prev term
//   6. d_delta = d_pre_tanh @ k_norm  // sum over columns
//   7. d_k_norm = d_pre_tanh^T @ delta  // sum over rows
//   8. d_v = d_delta  // from delta = v - retrieved
//   9. d_retrieved = -d_delta
//   10. dS_prev += outer(d_retrieved, k_norm)  // from retrieved = S_prev @ k_norm
//   11. d_k_norm += S_prev^T @ d_retrieved
//   12. d_k_from_norm = d_k_norm / ||k|| - k * (k · d_k_norm) / ||k||³
//   13. d_k_raw = d_k_from_norm + d_v + d_q
// ============================================================================

template<int N_STATE>
__global__ void E74FullMatrixBackwardKernel_BF16(
    int T,
    int B,
    const __nv_bfloat16* __restrict__ k_all,        // [T, B, N_STATE] - projected k (before normalization!)
    const __nv_bfloat16* __restrict__ S_checkpoints,// [num_checkpoints, B, N_STATE, N_STATE]
    const __nv_bfloat16* __restrict__ Sq_cache,     // [T, B, N_STATE]
    const __nv_bfloat16* __restrict__ d_output,     // [T, B, N_STATE]
    __nv_bfloat16* __restrict__ d_k_all,            // [T, B, N_STATE]
    int checkpoint_interval
) {
    int b = blockIdx.x;
    if (b >= B) return;

    extern __shared__ float shared_mem[];
    float* S_prev = shared_mem;                       // [N_STATE * N_STATE] - state BEFORE update
    float* S_curr = S_prev + N_STATE * N_STATE;       // [N_STATE * N_STATE] - state AFTER update
    float* dS = S_curr + N_STATE * N_STATE;           // [N_STATE * N_STATE] - gradient wrt S
    float* k_raw = dS + N_STATE * N_STATE;            // [N_STATE] - original k (before normalization)
    float* k_norm = k_raw + N_STATE;                  // [N_STATE] - normalized k
    float* delta = k_norm + N_STATE;                  // [N_STATE] - v - retrieved
    float* retrieved = delta + N_STATE;               // [N_STATE] - S @ k_norm
    float* d_k_raw = retrieved + N_STATE;             // [N_STATE] - gradient wrt k
    float* d_Sq_shared = d_k_raw + N_STATE;           // [N_STATE] - gradient wrt Sq
    float* d_delta = d_Sq_shared + N_STATE;           // [N_STATE] - gradient wrt delta
    float* d_k_norm = d_delta + N_STATE;              // [N_STATE] - gradient wrt k_norm

    int tid = threadIdx.x;
    int n2 = N_STATE * N_STATE;

    // Initialize dS to zero
    for (int i = tid; i < n2; i += blockDim.x) {
        dS[i] = 0.0f;
    }
    __syncthreads();

    // Process timesteps in reverse, segment by segment
    int num_segments = (T + checkpoint_interval - 1) / checkpoint_interval;

    for (int seg = num_segments - 1; seg >= 0; seg--) {
        int t_start = seg * checkpoint_interval;
        int t_end = min(t_start + checkpoint_interval, T);

        // ====== FORWARD RECOMPUTE through segment ======
        // Load checkpoint (this is S at start of segment)
        for (int i = tid; i < n2; i += blockDim.x) {
            S_prev[i] = __bfloat162float(S_checkpoints[seg * B * n2 + b * n2 + i]);
        }
        __syncthreads();

        // We need to store all intermediate S states for this segment
        // For simplicity, we'll recompute in both forward and backward directions
        // (less memory, more compute)

        // First, recompute forward to get S_curr at end of segment
        for (int t = t_start; t < t_end; t++) {
            // Load original k
            if (tid < N_STATE) {
                k_raw[tid] = __bfloat162float(k_all[t * B * N_STATE + b * N_STATE + tid]);
            }
            __syncthreads();

            // Normalize k
            __shared__ float k_norm_val;
            if (tid == 0) {
                float sum_sq = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    sum_sq += k_raw[i] * k_raw[i];
                }
                k_norm_val = sqrtf(sum_sq) + 1e-6f;
            }
            __syncthreads();
            if (tid < N_STATE) {
                k_norm[tid] = k_raw[tid] / k_norm_val;
            }
            __syncthreads();

            // retrieved = S_prev @ k_norm
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    sum += S_prev[tid * N_STATE + j] * k_norm[j];
                }
                retrieved[tid] = sum;
            }
            __syncthreads();

            // delta = k_raw - retrieved (v = k_raw for tied_kvq)
            if (tid < N_STATE) {
                delta[tid] = k_raw[tid] - retrieved[tid];
            }
            __syncthreads();

            // S_curr = tanh(S_prev + outer(delta, k_norm))
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                float pre_tanh = S_prev[i] + delta[row] * k_norm[col];
                S_curr[i] = tanhf(pre_tanh);
            }
            __syncthreads();

            // Copy S_curr to S_prev for next timestep (if not last)
            if (t < t_end - 1) {
                for (int i = tid; i < n2; i += blockDim.x) {
                    S_prev[i] = S_curr[i];
                }
                __syncthreads();
            }
        }

        // ====== BACKWARD through segment ======
        // Now S_curr contains the state at end of segment
        // We need to process backward, recomputing S_prev for each timestep

        for (int t = t_end - 1; t >= t_start; t--) {
            // Recompute S_prev for this timestep by forward from checkpoint
            // Load checkpoint
            for (int i = tid; i < n2; i += blockDim.x) {
                S_prev[i] = __bfloat162float(S_checkpoints[seg * B * n2 + b * n2 + i]);
            }
            __syncthreads();

            // Forward to timestep t-1 (to get S_prev at timestep t)
            for (int tt = t_start; tt < t; tt++) {
                if (tid < N_STATE) {
                    k_raw[tid] = __bfloat162float(k_all[tt * B * N_STATE + b * N_STATE + tid]);
                }
                __syncthreads();

                __shared__ float k_norm_val_fwd;
                if (tid == 0) {
                    float sum_sq = 0.0f;
                    for (int i = 0; i < N_STATE; i++) {
                        sum_sq += k_raw[i] * k_raw[i];
                    }
                    k_norm_val_fwd = sqrtf(sum_sq) + 1e-6f;
                }
                __syncthreads();
                if (tid < N_STATE) {
                    k_norm[tid] = k_raw[tid] / k_norm_val_fwd;
                }
                __syncthreads();

                if (tid < N_STATE) {
                    float sum = 0.0f;
                    for (int j = 0; j < N_STATE; j++) {
                        sum += S_prev[tid * N_STATE + j] * k_norm[j];
                    }
                    retrieved[tid] = sum;
                }
                __syncthreads();

                if (tid < N_STATE) {
                    delta[tid] = k_raw[tid] - retrieved[tid];
                }
                __syncthreads();

                for (int i = tid; i < n2; i += blockDim.x) {
                    int row = i / N_STATE;
                    int col = i % N_STATE;
                    float pre_tanh = S_prev[i] + delta[row] * k_norm[col];
                    S_prev[i] = tanhf(pre_tanh);
                }
                __syncthreads();
            }
            // Now S_prev contains the state BEFORE timestep t

            // Load k_raw for timestep t
            if (tid < N_STATE) {
                k_raw[tid] = __bfloat162float(k_all[t * B * N_STATE + b * N_STATE + tid]);
            }
            __syncthreads();

            // Compute k_norm
            __shared__ float k_norm_val_t;
            if (tid == 0) {
                float sum_sq = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    sum_sq += k_raw[i] * k_raw[i];
                }
                k_norm_val_t = sqrtf(sum_sq) + 1e-6f;
            }
            __syncthreads();
            if (tid < N_STATE) {
                k_norm[tid] = k_raw[tid] / k_norm_val_t;
            }
            __syncthreads();

            // Compute retrieved and delta for timestep t
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    sum += S_prev[tid * N_STATE + j] * k_norm[j];
                }
                retrieved[tid] = sum;
                delta[tid] = k_raw[tid] - retrieved[tid];
            }
            __syncthreads();

            // Compute S_curr = tanh(S_prev + outer(delta, k_norm))
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                float pre_tanh = S_prev[i] + delta[row] * k_norm[col];
                S_curr[i] = tanhf(pre_tanh);
            }
            __syncthreads();

            // ====== BACKWARD PASS for timestep t ======

            // Initialize d_k_raw for this timestep
            if (tid < N_STATE) {
                d_k_raw[tid] = 0.0f;
                d_k_norm[tid] = 0.0f;
                d_delta[tid] = 0.0f;
            }
            __syncthreads();

            // Step 1: d_Sq from output gradient
            // out = Sq * Sq * sigmoid(Sq)
            // d_Sq = d_out * (2*Sq*sig + Sq²*sig*(1-sig))
            if (tid < N_STATE) {
                float Sq_val = __bfloat162float(Sq_cache[t * B * N_STATE + b * N_STATE + tid]);
                float d_out = __bfloat162float(d_output[t * B * N_STATE + b * N_STATE + tid]);
                float sig = 1.0f / (1.0f + expf(-Sq_val));
                d_Sq_shared[tid] = d_out * (2.0f * Sq_val * sig + Sq_val * Sq_val * sig * (1.0f - sig));
            }
            __syncthreads();

            // Step 2: dS += outer(d_Sq, k_raw)  (q = k_raw for tied_kvq)
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                dS[i] += d_Sq_shared[row] * k_raw[col];
            }
            __syncthreads();

            // Step 3: d_q = S_curr^T @ d_Sq  (contributes to d_k_raw)
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    sum += S_curr[i * N_STATE + tid] * d_Sq_shared[i];
                }
                d_k_raw[tid] += sum;  // d_q contribution
            }
            __syncthreads();

            // Step 4: d_pre_tanh = dS * (1 - S_curr²)  (tanh gradient)
            // This modifies dS in place for efficiency (becomes d_pre_tanh)
            // We'll use a temp copy approach to preserve dS for next iteration
            // Actually, we need to accumulate into dS for the PREVIOUS timestep
            // Let's compute d_S_prev separately

            // Step 5-11: Backprop through state update
            // d_pre_tanh = dS * (1 - S_curr²)
            // d_delta[i] = sum_j(d_pre_tanh[i,j] * k_norm[j])
            // d_k_norm[j] += sum_i(d_pre_tanh[i,j] * delta[i])
            // d_S_prev = d_pre_tanh (from S_prev term in pre_tanh)

            // Compute d_delta and d_k_norm from d_pre_tanh
            if (tid < N_STATE) {
                float d_delta_local = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    float d_pre_tanh_ij = dS[tid * N_STATE + j] * (1.0f - S_curr[tid * N_STATE + j] * S_curr[tid * N_STATE + j]);
                    d_delta_local += d_pre_tanh_ij * k_norm[j];
                }
                d_delta[tid] = d_delta_local;
            }
            __syncthreads();

            if (tid < N_STATE) {
                float d_k_norm_local = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    float d_pre_tanh_ij = dS[i * N_STATE + tid] * (1.0f - S_curr[i * N_STATE + tid] * S_curr[i * N_STATE + tid]);
                    d_k_norm_local += d_pre_tanh_ij * delta[i];
                }
                d_k_norm[tid] = d_k_norm_local;
            }
            __syncthreads();

            // Step 8-9: d_v = d_delta, d_retrieved = -d_delta
            // d_v contributes to d_k_raw (since v = k_raw)
            if (tid < N_STATE) {
                d_k_raw[tid] += d_delta[tid];  // d_v contribution
            }
            __syncthreads();

            // Step 10: dS_prev (for next iteration) += outer(d_retrieved, k_norm)
            //        = outer(-d_delta, k_norm)
            // This will be the dS for the next backward iteration

            // Step 11: d_k_norm += S_prev^T @ d_retrieved = S_prev^T @ (-d_delta)
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    sum += S_prev[i * N_STATE + tid] * (-d_delta[i]);
                }
                d_k_norm[tid] += sum;
            }
            __syncthreads();

            // Step 12: d_k_from_norm = d_k_norm / ||k|| - k * (k · d_k_norm) / ||k||³
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
                    float d_k_from_norm = d_k_norm[tid] / norm - k_raw[tid] * k_dot_dk / norm3;
                    d_k_raw[tid] += d_k_from_norm;
                }
                __syncthreads();
            }

            // Write d_k_raw to output
            if (tid < N_STATE) {
                d_k_all[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(d_k_raw[tid]);
            }
            __syncthreads();

            // Update dS for next iteration (previous timestep)
            // dS_prev = d_pre_tanh + outer(-d_delta, k_norm)
            // But we need to be careful: dS already has contributions from future timesteps
            // We need to compute the gradient that flows to S_prev
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                float d_pre_tanh_ij = dS[i] * (1.0f - S_curr[i] * S_curr[i]);
                // dS for previous timestep = d_pre_tanh (from S_prev in pre_tanh) + outer(-d_delta, k_norm)
                dS[i] = d_pre_tanh_ij + (-d_delta[row]) * k_norm[col];
            }
            __syncthreads();
        }
    }
}


// ============================================================================
// Backward Kernel - BF16 Storage for State Matrices (for n >= 64)
// Uses bf16 for S_prev and S_curr to reduce shared memory usage
// Memory: 2*n*n*2 (bf16) + n*n*4 (fp32 dS) + 8*n*4 (vectors) = ~35KB for n=64
// ============================================================================

template<int N_STATE>
__global__ void E74FullMatrixBackwardKernel_BF16Storage(
    int T,
    int B,
    const __nv_bfloat16* __restrict__ k_all,
    const __nv_bfloat16* __restrict__ S_checkpoints,
    const __nv_bfloat16* __restrict__ Sq_cache,
    const __nv_bfloat16* __restrict__ d_output,
    __nv_bfloat16* __restrict__ d_k_all,
    int checkpoint_interval
) {
    int b = blockIdx.x;
    if (b >= B) return;

    extern __shared__ char shared_mem_raw[];

    // Memory layout: S_prev (bf16), S_curr (bf16), dS (fp32), vectors (fp32)
    __nv_bfloat16* S_prev = reinterpret_cast<__nv_bfloat16*>(shared_mem_raw);
    __nv_bfloat16* S_curr = S_prev + N_STATE * N_STATE;
    float* dS = reinterpret_cast<float*>(S_curr + N_STATE * N_STATE);
    float* k_raw = dS + N_STATE * N_STATE;
    float* k_norm = k_raw + N_STATE;
    float* delta = k_norm + N_STATE;
    float* retrieved = delta + N_STATE;
    float* d_k_raw = retrieved + N_STATE;
    float* d_Sq_shared = d_k_raw + N_STATE;
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

        // Load checkpoint
        for (int i = tid; i < n2; i += blockDim.x) {
            S_prev[i] = S_checkpoints[seg * B * n2 + b * n2 + i];
        }
        __syncthreads();

        // Forward recompute through segment
        for (int t = t_start; t < t_end; t++) {
            if (tid < N_STATE) {
                k_raw[tid] = __bfloat162float(k_all[t * B * N_STATE + b * N_STATE + tid]);
            }
            __syncthreads();

            __shared__ float k_norm_val;
            if (tid == 0) {
                float sum_sq = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    sum_sq += k_raw[i] * k_raw[i];
                }
                k_norm_val = sqrtf(sum_sq) + 1e-6f;
            }
            __syncthreads();
            if (tid < N_STATE) {
                k_norm[tid] = k_raw[tid] / k_norm_val;
            }
            __syncthreads();

            // retrieved = S_prev @ k_norm (load bf16, compute fp32)
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    sum += __bfloat162float(S_prev[tid * N_STATE + j]) * k_norm[j];
                }
                retrieved[tid] = sum;
            }
            __syncthreads();

            if (tid < N_STATE) {
                delta[tid] = k_raw[tid] - retrieved[tid];
            }
            __syncthreads();

            // S_curr = tanh(S_prev + outer(delta, k_norm))
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                float pre_tanh = __bfloat162float(S_prev[i]) + delta[row] * k_norm[col];
                S_curr[i] = __float2bfloat16(tanhf(pre_tanh));
            }
            __syncthreads();

            if (t < t_end - 1) {
                for (int i = tid; i < n2; i += blockDim.x) {
                    S_prev[i] = S_curr[i];
                }
                __syncthreads();
            }
        }

        // Backward through segment
        for (int t = t_end - 1; t >= t_start; t--) {
            // Reload checkpoint and recompute to timestep t
            for (int i = tid; i < n2; i += blockDim.x) {
                S_prev[i] = S_checkpoints[seg * B * n2 + b * n2 + i];
            }
            __syncthreads();

            for (int tt = t_start; tt < t; tt++) {
                if (tid < N_STATE) {
                    k_raw[tid] = __bfloat162float(k_all[tt * B * N_STATE + b * N_STATE + tid]);
                }
                __syncthreads();

                __shared__ float k_norm_val_fwd;
                if (tid == 0) {
                    float sum_sq = 0.0f;
                    for (int i = 0; i < N_STATE; i++) {
                        sum_sq += k_raw[i] * k_raw[i];
                    }
                    k_norm_val_fwd = sqrtf(sum_sq) + 1e-6f;
                }
                __syncthreads();
                if (tid < N_STATE) {
                    k_norm[tid] = k_raw[tid] / k_norm_val_fwd;
                }
                __syncthreads();

                if (tid < N_STATE) {
                    float sum = 0.0f;
                    for (int j = 0; j < N_STATE; j++) {
                        sum += __bfloat162float(S_prev[tid * N_STATE + j]) * k_norm[j];
                    }
                    retrieved[tid] = sum;
                }
                __syncthreads();

                if (tid < N_STATE) {
                    delta[tid] = k_raw[tid] - retrieved[tid];
                }
                __syncthreads();

                for (int i = tid; i < n2; i += blockDim.x) {
                    int row = i / N_STATE;
                    int col = i % N_STATE;
                    float pre_tanh = __bfloat162float(S_prev[i]) + delta[row] * k_norm[col];
                    S_prev[i] = __float2bfloat16(tanhf(pre_tanh));
                }
                __syncthreads();
            }

            // Load k for timestep t
            if (tid < N_STATE) {
                k_raw[tid] = __bfloat162float(k_all[t * B * N_STATE + b * N_STATE + tid]);
            }
            __syncthreads();

            __shared__ float k_norm_val_t;
            if (tid == 0) {
                float sum_sq = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    sum_sq += k_raw[i] * k_raw[i];
                }
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
                    sum += __bfloat162float(S_prev[tid * N_STATE + j]) * k_norm[j];
                }
                retrieved[tid] = sum;
                delta[tid] = k_raw[tid] - retrieved[tid];
            }
            __syncthreads();

            // Compute S_curr
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                float pre_tanh = __bfloat162float(S_prev[i]) + delta[row] * k_norm[col];
                S_curr[i] = __float2bfloat16(tanhf(pre_tanh));
            }
            __syncthreads();

            // ====== BACKWARD PASS for timestep t ======
            if (tid < N_STATE) {
                d_k_raw[tid] = 0.0f;
                d_k_norm[tid] = 0.0f;
                d_delta[tid] = 0.0f;
            }
            __syncthreads();

            // Step 1: d_Sq
            if (tid < N_STATE) {
                float Sq_val = __bfloat162float(Sq_cache[t * B * N_STATE + b * N_STATE + tid]);
                float d_out = __bfloat162float(d_output[t * B * N_STATE + b * N_STATE + tid]);
                float sig = 1.0f / (1.0f + expf(-Sq_val));
                d_Sq_shared[tid] = d_out * (2.0f * Sq_val * sig + Sq_val * Sq_val * sig * (1.0f - sig));
            }
            __syncthreads();

            // Step 2: dS += outer(d_Sq, k_raw)
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                dS[i] += d_Sq_shared[row] * k_raw[col];
            }
            __syncthreads();

            // Step 3: d_q = S_curr^T @ d_Sq
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    sum += __bfloat162float(S_curr[i * N_STATE + tid]) * d_Sq_shared[i];
                }
                d_k_raw[tid] += sum;
            }
            __syncthreads();

            // Steps 5-6: d_delta and d_k_norm from d_pre_tanh
            if (tid < N_STATE) {
                float d_delta_local = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    float S_curr_val = __bfloat162float(S_curr[tid * N_STATE + j]);
                    float d_pre_tanh_ij = dS[tid * N_STATE + j] * (1.0f - S_curr_val * S_curr_val);
                    d_delta_local += d_pre_tanh_ij * k_norm[j];
                }
                d_delta[tid] = d_delta_local;
            }
            __syncthreads();

            if (tid < N_STATE) {
                float d_k_norm_local = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    float S_curr_val = __bfloat162float(S_curr[i * N_STATE + tid]);
                    float d_pre_tanh_ij = dS[i * N_STATE + tid] * (1.0f - S_curr_val * S_curr_val);
                    d_k_norm_local += d_pre_tanh_ij * delta[i];
                }
                d_k_norm[tid] = d_k_norm_local;
            }
            __syncthreads();

            // Step 8: d_v contribution
            if (tid < N_STATE) {
                d_k_raw[tid] += d_delta[tid];
            }
            __syncthreads();

            // Step 11: d_k_norm += S_prev^T @ (-d_delta)
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    sum += __bfloat162float(S_prev[i * N_STATE + tid]) * (-d_delta[i]);
                }
                d_k_norm[tid] += sum;
            }
            __syncthreads();

            // Step 12: d_k_from_norm
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
                    float d_k_from_norm = d_k_norm[tid] / norm - k_raw[tid] * k_dot_dk / norm3;
                    d_k_raw[tid] += d_k_from_norm;
                }
                __syncthreads();
            }

            // Write d_k_raw
            if (tid < N_STATE) {
                d_k_all[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(d_k_raw[tid]);
            }
            __syncthreads();

            // Update dS for next iteration
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                float S_curr_val = __bfloat162float(S_curr[i]);
                float d_pre_tanh_ij = dS[i] * (1.0f - S_curr_val * S_curr_val);
                dS[i] = d_pre_tanh_ij + (-d_delta[row]) * k_norm[col];
            }
            __syncthreads();
        }
    }
}


// ============================================================================
// Backward Kernel - Global Memory for State Matrices (for n >= 96)
// Uses global memory for S_prev, S_curr, dS to support arbitrary n_state
// Slower than shared memory version but no size limit
// ============================================================================

template<int N_STATE>
__global__ void E74FullMatrixBackwardKernel_GlobalMem(
    int T,
    int B,
    const __nv_bfloat16* __restrict__ k_all,
    const __nv_bfloat16* __restrict__ S_checkpoints,
    const __nv_bfloat16* __restrict__ Sq_cache,
    const __nv_bfloat16* __restrict__ d_output,
    __nv_bfloat16* __restrict__ d_k_all,
    float* __restrict__ state_workspace,  // [B, 3, N_STATE, N_STATE] for S_prev, S_curr, dS
    int checkpoint_interval
) {
    int b = blockIdx.x;
    if (b >= B) return;

    // Global memory for state matrices (per batch element)
    int n2 = N_STATE * N_STATE;
    float* S_prev = state_workspace + b * 3 * n2;
    float* S_curr = S_prev + n2;
    float* dS = S_curr + n2;

    // Shared memory only for vectors
    extern __shared__ float shared_mem[];
    float* k_raw = shared_mem;
    float* k_norm = k_raw + N_STATE;
    float* delta = k_norm + N_STATE;
    float* retrieved = delta + N_STATE;
    float* d_k_raw = retrieved + N_STATE;
    float* d_Sq_shared = d_k_raw + N_STATE;
    float* d_delta = d_Sq_shared + N_STATE;
    float* d_k_norm = d_delta + N_STATE;

    int tid = threadIdx.x;

    // Initialize dS to zero
    for (int i = tid; i < n2; i += blockDim.x) {
        dS[i] = 0.0f;
    }
    __syncthreads();

    int num_segments = (T + checkpoint_interval - 1) / checkpoint_interval;

    for (int seg = num_segments - 1; seg >= 0; seg--) {
        int t_start = seg * checkpoint_interval;
        int t_end = min(t_start + checkpoint_interval, T);

        // Load checkpoint
        for (int i = tid; i < n2; i += blockDim.x) {
            S_prev[i] = __bfloat162float(S_checkpoints[seg * B * n2 + b * n2 + i]);
        }
        __syncthreads();

        // Forward recompute through segment
        for (int t = t_start; t < t_end; t++) {
            if (tid < N_STATE) {
                k_raw[tid] = __bfloat162float(k_all[t * B * N_STATE + b * N_STATE + tid]);
            }
            __syncthreads();

            __shared__ float k_norm_val;
            if (tid == 0) {
                float sum_sq = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    sum_sq += k_raw[i] * k_raw[i];
                }
                k_norm_val = sqrtf(sum_sq) + 1e-6f;
            }
            __syncthreads();
            if (tid < N_STATE) {
                k_norm[tid] = k_raw[tid] / k_norm_val;
            }
            __syncthreads();

            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    sum += S_prev[tid * N_STATE + j] * k_norm[j];
                }
                retrieved[tid] = sum;
            }
            __syncthreads();

            if (tid < N_STATE) {
                delta[tid] = k_raw[tid] - retrieved[tid];
            }
            __syncthreads();

            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                float pre_tanh = S_prev[i] + delta[row] * k_norm[col];
                S_curr[i] = tanhf(pre_tanh);
            }
            __syncthreads();

            if (t < t_end - 1) {
                for (int i = tid; i < n2; i += blockDim.x) {
                    S_prev[i] = S_curr[i];
                }
                __syncthreads();
            }
        }

        // Backward through segment
        for (int t = t_end - 1; t >= t_start; t--) {
            // Reload checkpoint and recompute to timestep t
            for (int i = tid; i < n2; i += blockDim.x) {
                S_prev[i] = __bfloat162float(S_checkpoints[seg * B * n2 + b * n2 + i]);
            }
            __syncthreads();

            for (int tt = t_start; tt < t; tt++) {
                if (tid < N_STATE) {
                    k_raw[tid] = __bfloat162float(k_all[tt * B * N_STATE + b * N_STATE + tid]);
                }
                __syncthreads();

                __shared__ float k_norm_val_fwd;
                if (tid == 0) {
                    float sum_sq = 0.0f;
                    for (int i = 0; i < N_STATE; i++) {
                        sum_sq += k_raw[i] * k_raw[i];
                    }
                    k_norm_val_fwd = sqrtf(sum_sq) + 1e-6f;
                }
                __syncthreads();
                if (tid < N_STATE) {
                    k_norm[tid] = k_raw[tid] / k_norm_val_fwd;
                }
                __syncthreads();

                if (tid < N_STATE) {
                    float sum = 0.0f;
                    for (int j = 0; j < N_STATE; j++) {
                        sum += S_prev[tid * N_STATE + j] * k_norm[j];
                    }
                    retrieved[tid] = sum;
                }
                __syncthreads();

                if (tid < N_STATE) {
                    delta[tid] = k_raw[tid] - retrieved[tid];
                }
                __syncthreads();

                for (int i = tid; i < n2; i += blockDim.x) {
                    int row = i / N_STATE;
                    int col = i % N_STATE;
                    float pre_tanh = S_prev[i] + delta[row] * k_norm[col];
                    S_prev[i] = tanhf(pre_tanh);
                }
                __syncthreads();
            }

            // Load k for timestep t
            if (tid < N_STATE) {
                k_raw[tid] = __bfloat162float(k_all[t * B * N_STATE + b * N_STATE + tid]);
            }
            __syncthreads();

            __shared__ float k_norm_val_t;
            if (tid == 0) {
                float sum_sq = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    sum_sq += k_raw[i] * k_raw[i];
                }
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
                    sum += S_prev[tid * N_STATE + j] * k_norm[j];
                }
                retrieved[tid] = sum;
                delta[tid] = k_raw[tid] - retrieved[tid];
            }
            __syncthreads();

            // Compute S_curr
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                float pre_tanh = S_prev[i] + delta[row] * k_norm[col];
                S_curr[i] = tanhf(pre_tanh);
            }
            __syncthreads();

            // ====== BACKWARD PASS for timestep t ======
            if (tid < N_STATE) {
                d_k_raw[tid] = 0.0f;
                d_k_norm[tid] = 0.0f;
                d_delta[tid] = 0.0f;
            }
            __syncthreads();

            // Step 1: d_Sq
            if (tid < N_STATE) {
                float Sq_val = __bfloat162float(Sq_cache[t * B * N_STATE + b * N_STATE + tid]);
                float d_out = __bfloat162float(d_output[t * B * N_STATE + b * N_STATE + tid]);
                float sig = 1.0f / (1.0f + expf(-Sq_val));
                d_Sq_shared[tid] = d_out * (2.0f * Sq_val * sig + Sq_val * Sq_val * sig * (1.0f - sig));
            }
            __syncthreads();

            // Step 2: dS += outer(d_Sq, k_raw)
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                dS[i] += d_Sq_shared[row] * k_raw[col];
            }
            __syncthreads();

            // Step 3: d_q = S_curr^T @ d_Sq
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    sum += S_curr[i * N_STATE + tid] * d_Sq_shared[i];
                }
                d_k_raw[tid] += sum;
            }
            __syncthreads();

            // Steps 5-6: d_delta and d_k_norm
            if (tid < N_STATE) {
                float d_delta_local = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    float d_pre_tanh_ij = dS[tid * N_STATE + j] * (1.0f - S_curr[tid * N_STATE + j] * S_curr[tid * N_STATE + j]);
                    d_delta_local += d_pre_tanh_ij * k_norm[j];
                }
                d_delta[tid] = d_delta_local;
            }
            __syncthreads();

            if (tid < N_STATE) {
                float d_k_norm_local = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    float d_pre_tanh_ij = dS[i * N_STATE + tid] * (1.0f - S_curr[i * N_STATE + tid] * S_curr[i * N_STATE + tid]);
                    d_k_norm_local += d_pre_tanh_ij * delta[i];
                }
                d_k_norm[tid] = d_k_norm_local;
            }
            __syncthreads();

            // Step 8: d_v contribution
            if (tid < N_STATE) {
                d_k_raw[tid] += d_delta[tid];
            }
            __syncthreads();

            // Step 11: d_k_norm += S_prev^T @ (-d_delta)
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    sum += S_prev[i * N_STATE + tid] * (-d_delta[i]);
                }
                d_k_norm[tid] += sum;
            }
            __syncthreads();

            // Step 12: d_k_from_norm
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
                    float d_k_from_norm = d_k_norm[tid] / norm - k_raw[tid] * k_dot_dk / norm3;
                    d_k_raw[tid] += d_k_from_norm;
                }
                __syncthreads();
            }

            // Write d_k_raw
            if (tid < N_STATE) {
                d_k_all[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(d_k_raw[tid]);
            }
            __syncthreads();

            // Update dS for next iteration
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                float d_pre_tanh_ij = dS[i] * (1.0f - S_curr[i] * S_curr[i]);
                dS[i] = d_pre_tanh_ij + (-d_delta[row]) * k_norm[col];
            }
            __syncthreads();
        }
    }
}


// ============================================================================
// Backward Kernel - no_z projection (separate k, v, q)
// Template: N_STATE, USE_TANH (0=linear, 1=tanh)
//
// Forward (no_z): k_norm = k / ||k||, retrieved = S @ k_norm
//                 delta = v - retrieved (v separate from k!)
//                 S = f(S + outer(delta, k_norm)) where f = tanh or identity
//                 Sq = S @ q (q separate from k!)
//                 out = Sq * silu(Sq)
// ============================================================================

template<int N_STATE, int USE_TANH>
__global__ void E74FullMatrixBackwardKernel_NoZ(
    int T,
    int B,
    const __nv_bfloat16* __restrict__ k_all,        // [T, B, N_STATE]
    const __nv_bfloat16* __restrict__ v_all,        // [T, B, N_STATE]
    const __nv_bfloat16* __restrict__ q_all,        // [T, B, N_STATE]
    const __nv_bfloat16* __restrict__ S_checkpoints,// [num_checkpoints, B, N_STATE, N_STATE]
    const __nv_bfloat16* __restrict__ Sq_cache,     // [T, B, N_STATE]
    const __nv_bfloat16* __restrict__ d_output,     // [T, B, N_STATE]
    __nv_bfloat16* __restrict__ d_k_all,            // [T, B, N_STATE]
    __nv_bfloat16* __restrict__ d_v_all,            // [T, B, N_STATE]
    __nv_bfloat16* __restrict__ d_q_all,            // [T, B, N_STATE]
    int checkpoint_interval
) {
    int b = blockIdx.x;
    if (b >= B) return;

    extern __shared__ float shared_mem[];
    float* S_prev = shared_mem;                       // [N_STATE * N_STATE]
    float* S_curr = S_prev + N_STATE * N_STATE;       // [N_STATE * N_STATE]
    float* dS = S_curr + N_STATE * N_STATE;           // [N_STATE * N_STATE]
    float* k_raw = dS + N_STATE * N_STATE;            // [N_STATE]
    float* v_raw = k_raw + N_STATE;                   // [N_STATE]
    float* q_raw = v_raw + N_STATE;                   // [N_STATE]
    float* k_norm = q_raw + N_STATE;                  // [N_STATE]
    float* delta = k_norm + N_STATE;                  // [N_STATE]
    float* retrieved = delta + N_STATE;               // [N_STATE]
    float* d_k_raw = retrieved + N_STATE;             // [N_STATE]
    float* d_v_raw = d_k_raw + N_STATE;               // [N_STATE]
    float* d_q_raw = d_v_raw + N_STATE;               // [N_STATE]
    float* d_Sq_shared = d_q_raw + N_STATE;           // [N_STATE]
    float* d_delta = d_Sq_shared + N_STATE;           // [N_STATE]
    float* d_k_norm = d_delta + N_STATE;              // [N_STATE]

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

        // ====== BACKWARD through segment ======
        for (int t = t_end - 1; t >= t_start; t--) {
            // Reload checkpoint and recompute to timestep t
            for (int i = tid; i < n2; i += blockDim.x) {
                S_prev[i] = __bfloat162float(S_checkpoints[seg * B * n2 + b * n2 + i]);
            }
            __syncthreads();

            // Recompute forward to step t
            for (int tt = t_start; tt <= t; tt++) {
                if (tid < N_STATE) {
                    k_raw[tid] = __bfloat162float(k_all[tt * B * N_STATE + b * N_STATE + tid]);
                    v_raw[tid] = __bfloat162float(v_all[tt * B * N_STATE + b * N_STATE + tid]);
                    q_raw[tid] = __bfloat162float(q_all[tt * B * N_STATE + b * N_STATE + tid]);
                }
                __syncthreads();

                // Normalize k
                __shared__ float k_norm_val_fwd;
                if (tid == 0) {
                    float sum_sq = 0.0f;
                    for (int i = 0; i < N_STATE; i++) sum_sq += k_raw[i] * k_raw[i];
                    k_norm_val_fwd = sqrtf(sum_sq) + 1e-6f;
                }
                __syncthreads();
                if (tid < N_STATE) {
                    k_norm[tid] = k_raw[tid] / k_norm_val_fwd;
                }
                __syncthreads();

                // retrieved = S_prev @ k_norm
                if (tid < N_STATE) {
                    float sum = 0.0f;
                    for (int j = 0; j < N_STATE; j++) {
                        sum += S_prev[tid * N_STATE + j] * k_norm[j];
                    }
                    retrieved[tid] = sum;
                }
                __syncthreads();

                // delta = v - retrieved (v, not k!)
                if (tid < N_STATE) {
                    delta[tid] = v_raw[tid] - retrieved[tid];
                }
                __syncthreads();

                // S_curr = f(S_prev + outer(delta, k_norm))
                for (int i = tid; i < n2; i += blockDim.x) {
                    int row = i / N_STATE;
                    int col = i % N_STATE;
                    float pre_nonlin = S_prev[i] + delta[row] * k_norm[col];
                    if (USE_TANH) {
                        S_curr[i] = tanhf(pre_nonlin);
                    } else {
                        S_curr[i] = pre_nonlin;  // Linear: just pass through
                    }
                }
                __syncthreads();

                if (tt < t) {
                    // Copy S_curr to S_prev for next iteration
                    for (int i = tid; i < n2; i += blockDim.x) {
                        S_prev[i] = S_curr[i];
                    }
                    __syncthreads();
                }
            }

            // ====== BACKWARD at timestep t ======
            __shared__ float k_norm_val_t;
            if (tid == 0) {
                float sum_sq = 0.0f;
                for (int i = 0; i < N_STATE; i++) sum_sq += k_raw[i] * k_raw[i];
                k_norm_val_t = sqrtf(sum_sq) + 1e-6f;
            }
            __syncthreads();

            // Initialize gradients
            if (tid < N_STATE) {
                d_k_raw[tid] = 0.0f;
                d_v_raw[tid] = 0.0f;
                d_q_raw[tid] = 0.0f;
            }
            __syncthreads();

            // Step 1: d_Sq from self-gating output
            if (tid < N_STATE) {
                float Sq = __bfloat162float(Sq_cache[t * B * N_STATE + b * N_STATE + tid]);
                float d_out = __bfloat162float(d_output[t * B * N_STATE + b * N_STATE + tid]);
                float sig = 1.0f / (1.0f + expf(-Sq));
                // out = Sq * Sq * sig, d_out/d_Sq = 2*Sq*sig + Sq²*sig*(1-sig)
                float d_Sq = d_out * (2.0f * Sq * sig + Sq * Sq * sig * (1.0f - sig));
                d_Sq_shared[tid] = d_Sq;
            }
            __syncthreads();

            // Step 2: dS += outer(d_Sq, q) from Sq = S @ q
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                dS[i] += d_Sq_shared[row] * q_raw[col];
            }
            __syncthreads();

            // Step 3: d_q = S_curr^T @ d_Sq
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    sum += S_curr[i * N_STATE + tid] * d_Sq_shared[i];
                }
                d_q_raw[tid] = sum;
            }
            __syncthreads();

            // Step 4-6: Nonlinearity gradient and d_delta
            if (tid < N_STATE) {
                float d_delta_local = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    float d_pre_nonlin;
                    if (USE_TANH) {
                        d_pre_nonlin = dS[tid * N_STATE + j] * (1.0f - S_curr[tid * N_STATE + j] * S_curr[tid * N_STATE + j]);
                    } else {
                        d_pre_nonlin = dS[tid * N_STATE + j];  // Linear: gradient passes through
                    }
                    d_delta_local += d_pre_nonlin * k_norm[j];
                }
                d_delta[tid] = d_delta_local;
            }
            __syncthreads();

            // Step 7: d_k_norm = sum over rows of (d_pre_nonlin * delta)
            if (tid < N_STATE) {
                float d_k_norm_local = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    float d_pre_nonlin;
                    if (USE_TANH) {
                        d_pre_nonlin = dS[i * N_STATE + tid] * (1.0f - S_curr[i * N_STATE + tid] * S_curr[i * N_STATE + tid]);
                    } else {
                        d_pre_nonlin = dS[i * N_STATE + tid];
                    }
                    d_k_norm_local += d_pre_nonlin * delta[i];
                }
                d_k_norm[tid] = d_k_norm_local;
            }
            __syncthreads();

            // Step 8: d_v = d_delta (from delta = v - retrieved)
            if (tid < N_STATE) {
                d_v_raw[tid] = d_delta[tid];
            }
            __syncthreads();

            // Step 9-11: d_retrieved = -d_delta, d_k_norm += S_prev^T @ (-d_delta)
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    sum += S_prev[i * N_STATE + tid] * (-d_delta[i]);
                }
                d_k_norm[tid] += sum;
            }
            __syncthreads();

            // Step 12: d_k_from_norm = d_k_norm/||k|| - k*(k·d_k_norm)/||k||³
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
            if (tid < N_STATE) {
                d_k_all[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(d_k_raw[tid]);
                d_v_all[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(d_v_raw[tid]);
                d_q_all[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(d_q_raw[tid]);
            }
            __syncthreads();

            // Update dS for next iteration (going backward)
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                float d_pre_nonlin;
                if (USE_TANH) {
                    d_pre_nonlin = dS[i] * (1.0f - S_curr[i] * S_curr[i]);
                } else {
                    d_pre_nonlin = dS[i];
                }
                dS[i] = d_pre_nonlin + (-d_delta[row]) * k_norm[col];
            }
            __syncthreads();
        }
    }
}


// ============================================================================
// Backward Kernel - no_z projection with Global Memory (for n >= 48)
// Uses global memory for S_prev, S_curr, dS to support larger n_state values
// Shared memory only for vectors: 14*N_STATE floats
// ============================================================================

template<int N_STATE, int USE_TANH>
__global__ void E74FullMatrixBackwardKernel_NoZ_GlobalMem(
    int T,
    int B,
    const __nv_bfloat16* __restrict__ k_all,        // [T, B, N_STATE]
    const __nv_bfloat16* __restrict__ v_all,        // [T, B, N_STATE]
    const __nv_bfloat16* __restrict__ q_all,        // [T, B, N_STATE]
    const __nv_bfloat16* __restrict__ S_checkpoints,// [num_checkpoints, B, N_STATE, N_STATE]
    const __nv_bfloat16* __restrict__ Sq_cache,     // [T, B, N_STATE]
    const __nv_bfloat16* __restrict__ d_output,     // [T, B, N_STATE]
    __nv_bfloat16* __restrict__ d_k_all,            // [T, B, N_STATE]
    __nv_bfloat16* __restrict__ d_v_all,            // [T, B, N_STATE]
    __nv_bfloat16* __restrict__ d_q_all,            // [T, B, N_STATE]
    float* __restrict__ state_workspace,            // [B, 3, N_STATE, N_STATE] for S_prev, S_curr, dS
    int checkpoint_interval
) {
    int b = blockIdx.x;
    if (b >= B) return;

    // Global memory for state matrices (per batch element)
    int n2 = N_STATE * N_STATE;
    float* S_prev = state_workspace + b * 3 * n2;
    float* S_curr = S_prev + n2;
    float* dS = S_curr + n2;

    // Shared memory only for vectors (14 * N_STATE floats)
    extern __shared__ float shared_mem[];
    float* k_raw = shared_mem;                        // [N_STATE]
    float* v_raw = k_raw + N_STATE;                   // [N_STATE]
    float* q_raw = v_raw + N_STATE;                   // [N_STATE]
    float* k_norm = q_raw + N_STATE;                  // [N_STATE]
    float* delta = k_norm + N_STATE;                  // [N_STATE]
    float* retrieved = delta + N_STATE;               // [N_STATE]
    float* d_k_raw = retrieved + N_STATE;             // [N_STATE]
    float* d_v_raw = d_k_raw + N_STATE;               // [N_STATE]
    float* d_q_raw = d_v_raw + N_STATE;               // [N_STATE]
    float* d_Sq_shared = d_q_raw + N_STATE;           // [N_STATE]
    float* d_delta = d_Sq_shared + N_STATE;           // [N_STATE]
    float* d_k_norm = d_delta + N_STATE;              // [N_STATE]

    int tid = threadIdx.x;

    // Initialize dS to zero
    for (int i = tid; i < n2; i += blockDim.x) {
        dS[i] = 0.0f;
    }
    __syncthreads();

    int num_segments = (T + checkpoint_interval - 1) / checkpoint_interval;

    for (int seg = num_segments - 1; seg >= 0; seg--) {
        int t_start = seg * checkpoint_interval;
        int t_end = min(t_start + checkpoint_interval, T);

        // ====== BACKWARD through segment ======
        for (int t = t_end - 1; t >= t_start; t--) {
            // Reload checkpoint and recompute to timestep t
            for (int i = tid; i < n2; i += blockDim.x) {
                S_prev[i] = __bfloat162float(S_checkpoints[seg * B * n2 + b * n2 + i]);
            }
            __syncthreads();

            // Recompute forward to step t
            for (int tt = t_start; tt <= t; tt++) {
                if (tid < N_STATE) {
                    k_raw[tid] = __bfloat162float(k_all[tt * B * N_STATE + b * N_STATE + tid]);
                    v_raw[tid] = __bfloat162float(v_all[tt * B * N_STATE + b * N_STATE + tid]);
                    q_raw[tid] = __bfloat162float(q_all[tt * B * N_STATE + b * N_STATE + tid]);
                }
                __syncthreads();

                // Normalize k
                __shared__ float k_norm_val_fwd;
                if (tid == 0) {
                    float sum_sq = 0.0f;
                    for (int i = 0; i < N_STATE; i++) sum_sq += k_raw[i] * k_raw[i];
                    k_norm_val_fwd = sqrtf(sum_sq) + 1e-6f;
                }
                __syncthreads();
                if (tid < N_STATE) {
                    k_norm[tid] = k_raw[tid] / k_norm_val_fwd;
                }
                __syncthreads();

                // retrieved = S_prev @ k_norm
                if (tid < N_STATE) {
                    float sum = 0.0f;
                    for (int j = 0; j < N_STATE; j++) {
                        sum += S_prev[tid * N_STATE + j] * k_norm[j];
                    }
                    retrieved[tid] = sum;
                }
                __syncthreads();

                // delta = v - retrieved (v, not k!)
                if (tid < N_STATE) {
                    delta[tid] = v_raw[tid] - retrieved[tid];
                }
                __syncthreads();

                // S_curr = f(S_prev + outer(delta, k_norm))
                for (int i = tid; i < n2; i += blockDim.x) {
                    int row = i / N_STATE;
                    int col = i % N_STATE;
                    float pre_nonlin = S_prev[i] + delta[row] * k_norm[col];
                    if (USE_TANH) {
                        S_curr[i] = tanhf(pre_nonlin);
                    } else {
                        S_curr[i] = pre_nonlin;  // Linear: just pass through
                    }
                }
                __syncthreads();

                if (tt < t) {
                    // Copy S_curr to S_prev for next iteration
                    for (int i = tid; i < n2; i += blockDim.x) {
                        S_prev[i] = S_curr[i];
                    }
                    __syncthreads();
                }
            }

            // ====== BACKWARD at timestep t ======
            __shared__ float k_norm_val_t;
            if (tid == 0) {
                float sum_sq = 0.0f;
                for (int i = 0; i < N_STATE; i++) sum_sq += k_raw[i] * k_raw[i];
                k_norm_val_t = sqrtf(sum_sq) + 1e-6f;
            }
            __syncthreads();

            // Initialize gradients
            if (tid < N_STATE) {
                d_k_raw[tid] = 0.0f;
                d_v_raw[tid] = 0.0f;
                d_q_raw[tid] = 0.0f;
            }
            __syncthreads();

            // Step 1: d_Sq from self-gating output
            if (tid < N_STATE) {
                float Sq = __bfloat162float(Sq_cache[t * B * N_STATE + b * N_STATE + tid]);
                float d_out = __bfloat162float(d_output[t * B * N_STATE + b * N_STATE + tid]);
                float sig = 1.0f / (1.0f + expf(-Sq));
                // out = Sq * Sq * sig, d_out/d_Sq = 2*Sq*sig + Sq²*sig*(1-sig)
                float d_Sq = d_out * (2.0f * Sq * sig + Sq * Sq * sig * (1.0f - sig));
                d_Sq_shared[tid] = d_Sq;
            }
            __syncthreads();

            // Step 2: dS += outer(d_Sq, q) from Sq = S @ q
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                dS[i] += d_Sq_shared[row] * q_raw[col];
            }
            __syncthreads();

            // Step 3: d_q = S_curr^T @ d_Sq
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    sum += S_curr[i * N_STATE + tid] * d_Sq_shared[i];
                }
                d_q_raw[tid] = sum;
            }
            __syncthreads();

            // Step 4-6: Nonlinearity gradient and d_delta
            if (tid < N_STATE) {
                float d_delta_local = 0.0f;
                for (int j = 0; j < N_STATE; j++) {
                    float d_pre_nonlin;
                    if (USE_TANH) {
                        d_pre_nonlin = dS[tid * N_STATE + j] * (1.0f - S_curr[tid * N_STATE + j] * S_curr[tid * N_STATE + j]);
                    } else {
                        d_pre_nonlin = dS[tid * N_STATE + j];  // Linear: gradient passes through
                    }
                    d_delta_local += d_pre_nonlin * k_norm[j];
                }
                d_delta[tid] = d_delta_local;
            }
            __syncthreads();

            // Step 7: d_k_norm = sum over rows of (d_pre_nonlin * delta)
            if (tid < N_STATE) {
                float d_k_norm_local = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    float d_pre_nonlin;
                    if (USE_TANH) {
                        d_pre_nonlin = dS[i * N_STATE + tid] * (1.0f - S_curr[i * N_STATE + tid] * S_curr[i * N_STATE + tid]);
                    } else {
                        d_pre_nonlin = dS[i * N_STATE + tid];
                    }
                    d_k_norm_local += d_pre_nonlin * delta[i];
                }
                d_k_norm[tid] = d_k_norm_local;
            }
            __syncthreads();

            // Step 8: d_v = d_delta (from delta = v - retrieved)
            if (tid < N_STATE) {
                d_v_raw[tid] = d_delta[tid];
            }
            __syncthreads();

            // Step 9-11: d_retrieved = -d_delta, d_k_norm += S_prev^T @ (-d_delta)
            if (tid < N_STATE) {
                float sum = 0.0f;
                for (int i = 0; i < N_STATE; i++) {
                    sum += S_prev[i * N_STATE + tid] * (-d_delta[i]);
                }
                d_k_norm[tid] += sum;
            }
            __syncthreads();

            // Step 12: d_k_from_norm = d_k_norm/||k|| - k*(k·d_k_norm)/||k||³
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
            if (tid < N_STATE) {
                d_k_all[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(d_k_raw[tid]);
                d_v_all[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(d_v_raw[tid]);
                d_q_all[t * B * N_STATE + b * N_STATE + tid] = __float2bfloat16(d_q_raw[tid]);
            }
            __syncthreads();

            // Update dS for next iteration (going backward)
            for (int i = tid; i < n2; i += blockDim.x) {
                int row = i / N_STATE;
                int col = i % N_STATE;
                float d_pre_nonlin;
                if (USE_TANH) {
                    d_pre_nonlin = dS[i] * (1.0f - S_curr[i] * S_curr[i]);
                } else {
                    d_pre_nonlin = dS[i];
                }
                dS[i] = d_pre_nonlin + (-d_delta[row]) * k_norm[col];
            }
            __syncthreads();
        }
    }
}


// ============================================================================
// Host Functions
// ============================================================================

template<typename T>
E74FullMatrixForward<T>::E74FullMatrixForward(
    bool training,
    int batch_size,
    int n_state,
    int dim,
    int proj_type,
    bool use_tanh,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream
) : training_(training),
    batch_size_(batch_size),
    n_state_(n_state),
    dim_(dim),
    proj_type_(proj_type),
    use_tanh_(use_tanh),
    blas_handle_(blas_handle),
    stream_(stream) {}

template<typename DataT>
void E74FullMatrixForward<DataT>::Run(
    int steps,
    const DataT* W_kvq,
    const DataT* W_k,
    const DataT* W_v,
    const DataT* W_q,
    const DataT* x,
    DataT* S,
    DataT* output,
    DataT* k_cache,
    DataT* v_cache,
    DataT* q_cache,
    DataT* S_cache,
    DataT* workspace
) {
    int T_steps = steps;
    int B = batch_size_;
    int n = n_state_;
    int d = dim_;

    // Step 1: Batch projection with cuBLAS
    const float alpha = 1.0f, beta = 0.0f;

    if (proj_type_ == 0) {
        // tied_kvq: single projection k = v = q = W_kvq @ x
        cublasGemmEx(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            n, T_steps * B, d,
            &alpha,
            W_kvq, CUDA_R_16BF, d,
            x, CUDA_R_16BF, d,
            &beta,
            k_cache, CUDA_R_16BF, n,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        );
        // v_cache and q_cache not used for tied_kvq
    } else if (proj_type_ == 1) {
        // tied_kq: k = q = W_k @ x, v = W_v @ x
        cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            n, T_steps * B, d, &alpha,
            W_k, CUDA_R_16BF, d, x, CUDA_R_16BF, d,
            &beta, k_cache, CUDA_R_16BF, n,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            n, T_steps * B, d, &alpha,
            W_v, CUDA_R_16BF, d, x, CUDA_R_16BF, d,
            &beta, v_cache, CUDA_R_16BF, n,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        // q_cache not used for tied_kq (q = k)
    } else {
        // no_z: separate k, v, q projections
        cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            n, T_steps * B, d, &alpha,
            W_k, CUDA_R_16BF, d, x, CUDA_R_16BF, d,
            &beta, k_cache, CUDA_R_16BF, n,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            n, T_steps * B, d, &alpha,
            W_v, CUDA_R_16BF, d, x, CUDA_R_16BF, d,
            &beta, v_cache, CUDA_R_16BF, n,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            n, T_steps * B, d, &alpha,
            W_q, CUDA_R_16BF, d, x, CUDA_R_16BF, d,
            &beta, q_cache, CUDA_R_16BF, n,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }

    // Step 2: Run fused forward kernel
    // Shared memory: state [n*n] + vectors [5*n] for k, v, q, retrieved, temp
    int shared_size = (n * n + 5 * n) * sizeof(float);
    int threads = min(256, n * n);

    // S_cache layout: [num_checkpoints, B, n, n]
    // Plus sq_cache: [T, B, n]
    int num_checkpoints = (T_steps + CHECKPOINT_INTERVAL - 1) / CHECKPOINT_INTERVAL + 1;
    DataT* s_checkpoints = S_cache;
    DataT* sq_cache = S_cache + num_checkpoints * B * n * n;

    // Dispatch to appropriate kernel based on proj_type, n_state, and use_tanh
    #define DISPATCH_KERNEL(N_STATE, PROJ_TYPE, USE_TANH) \
        E74FullMatrixForwardKernel_BF16_General<N_STATE, PROJ_TYPE, USE_TANH><<<B, threads, shared_size, stream_>>>( \
            T_steps, B, \
            (const __nv_bfloat16*)k_cache, \
            (const __nv_bfloat16*)v_cache, \
            (const __nv_bfloat16*)q_cache, \
            (__nv_bfloat16*)S, (__nv_bfloat16*)output, \
            (__nv_bfloat16*)s_checkpoints, (__nv_bfloat16*)sq_cache, \
            CHECKPOINT_INTERVAL)

    #define DISPATCH_BY_SIZE(PROJ_TYPE) \
        if (use_tanh_) { \
            if (n == 1) { DISPATCH_KERNEL(1, PROJ_TYPE, 1); } \
            else if (n == 2) { DISPATCH_KERNEL(2, PROJ_TYPE, 1); } \
            else if (n == 4) { DISPATCH_KERNEL(4, PROJ_TYPE, 1); } \
            else if (n == 8) { DISPATCH_KERNEL(8, PROJ_TYPE, 1); } \
            else if (n == 16) { DISPATCH_KERNEL(16, PROJ_TYPE, 1); } \
            else if (n == 24) { DISPATCH_KERNEL(24, PROJ_TYPE, 1); } \
            else if (n == 32) { DISPATCH_KERNEL(32, PROJ_TYPE, 1); } \
            else if (n == 48) { DISPATCH_KERNEL(48, PROJ_TYPE, 1); } \
            else if (n == 64) { DISPATCH_KERNEL(64, PROJ_TYPE, 1); } \
            else if (n == 96) { DISPATCH_KERNEL(96, PROJ_TYPE, 1); } \
            else if (n == 128) { DISPATCH_KERNEL(128, PROJ_TYPE, 1); } \
            else if (n == 192) { DISPATCH_KERNEL(192, PROJ_TYPE, 1); } \
            else if (n == 256) { DISPATCH_KERNEL(256, PROJ_TYPE, 1); } \
        } else { \
            if (n == 1) { DISPATCH_KERNEL(1, PROJ_TYPE, 0); } \
            else if (n == 2) { DISPATCH_KERNEL(2, PROJ_TYPE, 0); } \
            else if (n == 4) { DISPATCH_KERNEL(4, PROJ_TYPE, 0); } \
            else if (n == 8) { DISPATCH_KERNEL(8, PROJ_TYPE, 0); } \
            else if (n == 16) { DISPATCH_KERNEL(16, PROJ_TYPE, 0); } \
            else if (n == 24) { DISPATCH_KERNEL(24, PROJ_TYPE, 0); } \
            else if (n == 32) { DISPATCH_KERNEL(32, PROJ_TYPE, 0); } \
            else if (n == 48) { DISPATCH_KERNEL(48, PROJ_TYPE, 0); } \
            else if (n == 64) { DISPATCH_KERNEL(64, PROJ_TYPE, 0); } \
            else if (n == 96) { DISPATCH_KERNEL(96, PROJ_TYPE, 0); } \
            else if (n == 128) { DISPATCH_KERNEL(128, PROJ_TYPE, 0); } \
            else if (n == 192) { DISPATCH_KERNEL(192, PROJ_TYPE, 0); } \
            else if (n == 256) { DISPATCH_KERNEL(256, PROJ_TYPE, 0); } \
        }

    if (proj_type_ == 0) {
        // tied_kvq
        DISPATCH_BY_SIZE(0);
    } else if (proj_type_ == 1) {
        // tied_kq
        DISPATCH_BY_SIZE(1);
    } else {
        // no_z
        DISPATCH_BY_SIZE(2);
    }

    #undef DISPATCH_KERNEL
    #undef DISPATCH_BY_SIZE
}

// E74FullMatrixBackward implementation
template<typename T>
E74FullMatrixBackward<T>::E74FullMatrixBackward(
    int batch_size,
    int n_state,
    int dim,
    int proj_type,
    bool use_tanh,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream
) : batch_size_(batch_size),
    n_state_(n_state),
    dim_(dim),
    proj_type_(proj_type),
    use_tanh_(use_tanh),
    blas_handle_(blas_handle),
    stream_(stream) {}

template<typename T>
void E74FullMatrixBackward<T>::Run(
    int steps,
    const T* W_kvq,
    const T* W_k,
    const T* W_v,
    const T* W_q,
    const T* x,
    const T* S_checkpoints,
    const T* Sq_cache,
    const T* k_cache,
    const T* v_cache,
    const T* q_cache,
    const T* d_output,
    T* dx,
    T* dW_kvq,
    T* dW_k,
    T* dW_v,
    T* dW_q,
    T* workspace
) {
    int T_steps = steps;
    int B = batch_size_;
    int n = n_state_;
    int d = dim_;

    // Allocate d_k in workspace
    T* d_k = workspace;

    // Zero out d_k
    cudaMemsetAsync(d_k, 0, T_steps * B * n * sizeof(T), stream_);

    // Shared memory calculation:
    // n <= 48: all fp32: 3*n*n*4 + 8*n*4 bytes
    // n == 64: bf16 for S_prev/S_curr, fp32 for dS: 2*n*n*2 + n*n*4 + 8*n*4 = 34816 bytes
    // n == 96: global memory kernel, only vectors in shared: 8*n*4 = 3072 bytes
    int shared_size;
    float* state_workspace = nullptr;
    if (n <= 48) {
        shared_size = (3 * n * n + 8 * n) * sizeof(float);
    } else if (n == 64) {
        // S_prev + S_curr in bf16, dS + vectors in fp32
        shared_size = 2 * n * n * sizeof(__nv_bfloat16) + (n * n + 8 * n) * sizeof(float);
    } else {
        // n >= 96: global memory kernel, shared only for vectors
        shared_size = 8 * n * sizeof(float);
        // State workspace comes after d_k in workspace: [B, 3, n, n] fp32
        state_workspace = reinterpret_cast<float*>(d_k + T_steps * B * n);
        // Zero out state workspace
        cudaMemsetAsync(state_workspace, 0, B * 3 * n * n * sizeof(float), stream_);
    }
    int threads = min(256, n * n);

    if (proj_type_ == 0) {
        // tied_kvq
        if (n == 32) {
            E74FullMatrixBackwardKernel_BF16<32><<<B, threads, shared_size, stream_>>>(
                T_steps, B, (const __nv_bfloat16*)k_cache,
                (const __nv_bfloat16*)S_checkpoints, (const __nv_bfloat16*)Sq_cache,
                (const __nv_bfloat16*)d_output, (__nv_bfloat16*)d_k,
                CHECKPOINT_INTERVAL);
        } else if (n == 48) {
            E74FullMatrixBackwardKernel_BF16<48><<<B, threads, shared_size, stream_>>>(
                T_steps, B, (const __nv_bfloat16*)k_cache,
                (const __nv_bfloat16*)S_checkpoints, (const __nv_bfloat16*)Sq_cache,
                (const __nv_bfloat16*)d_output, (__nv_bfloat16*)d_k,
                CHECKPOINT_INTERVAL);
        } else if (n == 64) {
            // Use bf16 storage kernel for n=64
            E74FullMatrixBackwardKernel_BF16Storage<64><<<B, threads, shared_size, stream_>>>(
                T_steps, B, (const __nv_bfloat16*)k_cache,
                (const __nv_bfloat16*)S_checkpoints, (const __nv_bfloat16*)Sq_cache,
                (const __nv_bfloat16*)d_output, (__nv_bfloat16*)d_k,
                CHECKPOINT_INTERVAL);
        } else if (n == 96) {
            // Global memory kernel for n=96 (state matrices in global memory)
            E74FullMatrixBackwardKernel_GlobalMem<96><<<B, threads, shared_size, stream_>>>(
                T_steps, B, (const __nv_bfloat16*)k_cache,
                (const __nv_bfloat16*)S_checkpoints, (const __nv_bfloat16*)Sq_cache,
                (const __nv_bfloat16*)d_output, (__nv_bfloat16*)d_k,
                state_workspace, CHECKPOINT_INTERVAL);
        }

        // Backprop through projection: dx = W_kvq^T @ d_k, dW_kvq = d_k @ x^T
        const float alpha = 1.0f, beta = 0.0f;

        // dx = W_kvq^T @ d_k: [d, T*B] = [d, n] @ [n, T*B]
        cublasGemmEx(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            d, T_steps * B, n,
            &alpha,
            W_kvq, CUDA_R_16BF, d,
            d_k, CUDA_R_16BF, n,
            &beta,
            dx, CUDA_R_16BF, d,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        );

        // dW_kvq: compute [d, n] col-major so PyTorch sees [n, d] row-major
        // C = x @ d_k^T: [d, n] = [d, T*B] @ [T*B, n]
        const float beta_one = 1.0f;
        cublasGemmEx(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_T,
            d, n, T_steps * B,  // M=d, N=n, K=T*B
            &alpha,
            x, CUDA_R_16BF, d,      // A: [d, T*B], lda=d
            d_k, CUDA_R_16BF, n,    // B: [n, T*B], use B^T, ldb=n
            &beta_one,  // Accumulate
            dW_kvq, CUDA_R_16BF, d, // C: [d, n], ldc=d
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        );

    } else if (proj_type_ == 2) {
        // no_z: separate k, v, q projections
        // Allocate d_v, d_q after d_k in workspace
        T* d_v = d_k + T_steps * B * n;
        T* d_q = d_v + T_steps * B * n;

        // Zero out d_v, d_q
        cudaMemsetAsync(d_v, 0, T_steps * B * n * sizeof(T), stream_);
        cudaMemsetAsync(d_q, 0, T_steps * B * n * sizeof(T), stream_);

        int threads_noz = min(256, n * n);

        // For n >= 48, use global memory kernel to avoid shared memory overflow
        // Shared memory with 3*n*n: n=48 would need 3*48*48*4 + 14*48*4 = 30,336 bytes (OK)
        // n=64 needs 3*64*64*4 + 14*64*4 = 52,736 bytes (exceeds 48KB limit)
        // So use global memory kernel for n >= 48 to be safe
        if (n >= 48) {
            // Global memory kernel: shared memory only for 14*n vectors
            int shared_size_noz_global = 14 * n * sizeof(float);
            // State workspace comes after d_q: [B, 3, n, n] fp32
            float* state_workspace_noz = reinterpret_cast<float*>(d_q + T_steps * B * n);
            // Zero out state workspace
            cudaMemsetAsync(state_workspace_noz, 0, B * 3 * n * n * sizeof(float), stream_);

            #define DISPATCH_NOZ_BACKWARD_GLOBAL(N_STATE) \
                if (use_tanh_) { \
                    E74FullMatrixBackwardKernel_NoZ_GlobalMem<N_STATE, 1><<<B, threads_noz, shared_size_noz_global, stream_>>>( \
                        T_steps, B, \
                        (const __nv_bfloat16*)k_cache, \
                        (const __nv_bfloat16*)v_cache, \
                        (const __nv_bfloat16*)q_cache, \
                        (const __nv_bfloat16*)S_checkpoints, (const __nv_bfloat16*)Sq_cache, \
                        (const __nv_bfloat16*)d_output, \
                        (__nv_bfloat16*)d_k, (__nv_bfloat16*)d_v, (__nv_bfloat16*)d_q, \
                        state_workspace_noz, CHECKPOINT_INTERVAL); \
                } else { \
                    E74FullMatrixBackwardKernel_NoZ_GlobalMem<N_STATE, 0><<<B, threads_noz, shared_size_noz_global, stream_>>>( \
                        T_steps, B, \
                        (const __nv_bfloat16*)k_cache, \
                        (const __nv_bfloat16*)v_cache, \
                        (const __nv_bfloat16*)q_cache, \
                        (const __nv_bfloat16*)S_checkpoints, (const __nv_bfloat16*)Sq_cache, \
                        (const __nv_bfloat16*)d_output, \
                        (__nv_bfloat16*)d_k, (__nv_bfloat16*)d_v, (__nv_bfloat16*)d_q, \
                        state_workspace_noz, CHECKPOINT_INTERVAL); \
                }

            if (n == 48) { DISPATCH_NOZ_BACKWARD_GLOBAL(48); }
            else if (n == 64) { DISPATCH_NOZ_BACKWARD_GLOBAL(64); }
            else if (n == 96) { DISPATCH_NOZ_BACKWARD_GLOBAL(96); }
            else if (n == 128) { DISPATCH_NOZ_BACKWARD_GLOBAL(128); }
            else if (n == 192) { DISPATCH_NOZ_BACKWARD_GLOBAL(192); }
            else if (n == 256) { DISPATCH_NOZ_BACKWARD_GLOBAL(256); }

            #undef DISPATCH_NOZ_BACKWARD_GLOBAL
        } else {
            // n < 48: use shared memory kernel (fits in 48KB)
            int shared_size_noz = (3 * n * n + 14 * n) * sizeof(float);

            #define DISPATCH_NOZ_BACKWARD(N_STATE) \
                if (use_tanh_) { \
                    E74FullMatrixBackwardKernel_NoZ<N_STATE, 1><<<B, threads_noz, shared_size_noz, stream_>>>( \
                        T_steps, B, \
                        (const __nv_bfloat16*)k_cache, \
                        (const __nv_bfloat16*)v_cache, \
                        (const __nv_bfloat16*)q_cache, \
                        (const __nv_bfloat16*)S_checkpoints, (const __nv_bfloat16*)Sq_cache, \
                        (const __nv_bfloat16*)d_output, \
                        (__nv_bfloat16*)d_k, (__nv_bfloat16*)d_v, (__nv_bfloat16*)d_q, \
                        CHECKPOINT_INTERVAL); \
                } else { \
                    E74FullMatrixBackwardKernel_NoZ<N_STATE, 0><<<B, threads_noz, shared_size_noz, stream_>>>( \
                        T_steps, B, \
                        (const __nv_bfloat16*)k_cache, \
                        (const __nv_bfloat16*)v_cache, \
                        (const __nv_bfloat16*)q_cache, \
                        (const __nv_bfloat16*)S_checkpoints, (const __nv_bfloat16*)Sq_cache, \
                        (const __nv_bfloat16*)d_output, \
                        (__nv_bfloat16*)d_k, (__nv_bfloat16*)d_v, (__nv_bfloat16*)d_q, \
                        CHECKPOINT_INTERVAL); \
                }

            if (n == 1) { DISPATCH_NOZ_BACKWARD(1); }
            else if (n == 2) { DISPATCH_NOZ_BACKWARD(2); }
            else if (n == 4) { DISPATCH_NOZ_BACKWARD(4); }
            else if (n == 8) { DISPATCH_NOZ_BACKWARD(8); }
            else if (n == 16) { DISPATCH_NOZ_BACKWARD(16); }
            else if (n == 24) { DISPATCH_NOZ_BACKWARD(24); }
            else if (n == 32) { DISPATCH_NOZ_BACKWARD(32); }

            #undef DISPATCH_NOZ_BACKWARD
        }

        // Backprop through projections
        const float alpha = 1.0f, beta = 0.0f, beta_one = 1.0f;

        // dx = W_k^T @ d_k + W_v^T @ d_v + W_q^T @ d_q
        // First: dx = W_k^T @ d_k
        cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            d, T_steps * B, n, &alpha,
            W_k, CUDA_R_16BF, d, d_k, CUDA_R_16BF, n, &beta,
            dx, CUDA_R_16BF, d, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // dx += W_v^T @ d_v
        cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            d, T_steps * B, n, &alpha,
            W_v, CUDA_R_16BF, d, d_v, CUDA_R_16BF, n, &beta_one,
            dx, CUDA_R_16BF, d, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // dx += W_q^T @ d_q
        cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            d, T_steps * B, n, &alpha,
            W_q, CUDA_R_16BF, d, d_q, CUDA_R_16BF, n, &beta_one,
            dx, CUDA_R_16BF, d, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // dW_k = d_k @ x^T (accumulated)
        cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            d, n, T_steps * B, &alpha,
            x, CUDA_R_16BF, d, d_k, CUDA_R_16BF, n, &beta_one,
            dW_k, CUDA_R_16BF, d, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // dW_v = d_v @ x^T (accumulated)
        cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            d, n, T_steps * B, &alpha,
            x, CUDA_R_16BF, d, d_v, CUDA_R_16BF, n, &beta_one,
            dW_v, CUDA_R_16BF, d, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // dW_q = d_q @ x^T (accumulated)
        cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            d, n, T_steps * B, &alpha,
            x, CUDA_R_16BF, d, d_q, CUDA_R_16BF, n, &beta_one,
            dW_q, CUDA_R_16BF, d, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }
}

// Explicit instantiations
template struct E74FullMatrixForward<__nv_bfloat16>;
template struct E74FullMatrixBackward<__nv_bfloat16>;

}  // namespace elman
