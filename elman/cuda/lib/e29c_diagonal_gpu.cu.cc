/**
 * E29c: SSM-Style Diagonal Gating Dual-Memory Elman
 *
 * HYBRID TEMPLATE PATTERN (per CUDA_KERNEL_PATTERNS.md):
 *   - N_SLOTS: Templated (small, benefits from unrolling)
 *   - DIM: Runtime parameter (large, memory-bound anyway)
 *
 * Gate mechanism: gate = silu(z * g_z + read * g_r + h_work * g_h + b_gate)
 *
 * This is the SSM approach: element-wise scaling instead of matrix projection.
 * Backward is much cheaper than E29b (no dW_gate GEMM).
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <algorithm>

#include "hasty/elman_ladder.h"

namespace hasty {
namespace v0 {
namespace elman_ladder {

namespace {

constexpr int BLOCK_SIZE = 256;

// =============================================================================
// Device helpers
// =============================================================================

template<int N_SLOTS>
__device__ void softmax_device(float* attn_sh, const float scale) {
    float max_val = attn_sh[0] * scale;
    #pragma unroll
    for (int i = 1; i < N_SLOTS; i++) {
        float v = attn_sh[i] * scale;
        if (v > max_val) max_val = v;
    }

    float sum_exp = 0.0f;
    #pragma unroll
    for (int i = 0; i < N_SLOTS; i++) {
        attn_sh[i] = expf(attn_sh[i] * scale - max_val);
        sum_exp += attn_sh[i];
    }

    sum_exp = fmaxf(sum_exp, 1e-9f);
    #pragma unroll
    for (int i = 0; i < N_SLOTS; i++) {
        attn_sh[i] /= sum_exp;
    }
}

// Clamp value to prevent numerical overflow in gate computation
constexpr float GATE_CLAMP = 20.0f;

__device__ __forceinline__ float clamp_val(float x) {
    return fmaxf(-GATE_CLAMP, fminf(GATE_CLAMP, x));
}

__device__ __forceinline__ float silu_fwd(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float silu_bwd(float x) {
    float sig = 1.0f / (1.0f + expf(-x));
    return sig * (1.0f + x * (1.0f - sig));
}

// =============================================================================
// FORWARD KERNELS - Hybrid template (N_SLOTS templated, DIM runtime)
// =============================================================================

/**
 * Phase 1: Read attention + h_work update
 *
 * Shared memory layout: [h_work_sh(DIM), attn_sh(N_SLOTS), read_val_sh(DIM)]
 */
template<int N_SLOTS>
__global__ void E29cPhase1Kernel_BF16(
    const int DIM,
    const int batch_size,
    const __nv_bfloat16* __restrict__ Rh,
    const __nv_bfloat16* __restrict__ x_proj_t,
    const __nv_bfloat16* __restrict__ b_h,
    const __nv_bfloat16* __restrict__ h_tape,
    const __nv_bfloat16* __restrict__ h_work,
    __nv_bfloat16* __restrict__ h_work_out,
    __nv_bfloat16* __restrict__ read_attn_out,
    __nv_bfloat16* __restrict__ read_val_out,
    const float scale
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    extern __shared__ char shared_mem[];
    float* h_work_sh = (float*)shared_mem;
    float* attn_sh = h_work_sh + DIM;
    float* read_val_sh = attn_sh + N_SLOTS;

    // Load h_work into shared memory
    for (int d = tid; d < DIM; d += BLOCK_SIZE) {
        h_work_sh[d] = __bfloat162float(h_work[b * DIM + d]);
    }
    __syncthreads();

    // Compute attention scores
    if (tid < N_SLOTS) {
        float score = 0.0f;
        const __nv_bfloat16* tape_slot = h_tape + b * N_SLOTS * DIM + tid * DIM;
        for (int d = 0; d < DIM; d++) {
            score += __bfloat162float(tape_slot[d]) * h_work_sh[d];
        }
        attn_sh[tid] = score;
    }
    __syncthreads();

    // Softmax
    if (tid == 0) {
        softmax_device<N_SLOTS>(attn_sh, scale);
    }
    __syncthreads();

    // Compute read value
    for (int d = tid; d < DIM; d += BLOCK_SIZE) {
        float val = 0.0f;
        #pragma unroll
        for (int n = 0; n < N_SLOTS; n++) {
            val += attn_sh[n] * __bfloat162float(h_tape[b * N_SLOTS * DIM + n * DIM + d]);
        }
        read_val_sh[d] = val;
    }
    __syncthreads();

    // Update h_work: tanh(x_proj + Rh + read + b)
    for (int d = tid; d < DIM; d += BLOCK_SIZE) {
        float x = __bfloat162float(x_proj_t[b * DIM + d]);
        float r = __bfloat162float(Rh[b * DIM + d]);
        float bias = __bfloat162float(b_h[d]);
        float h_new = tanhf(x + r + read_val_sh[d] + bias);

        h_work_out[b * DIM + d] = __float2bfloat16(h_new);
        read_val_out[b * DIM + d] = __float2bfloat16(read_val_sh[d]);
    }

    // Store attention weights
    if (tid < N_SLOTS) {
        read_attn_out[b * N_SLOTS + tid] = __float2bfloat16(attn_sh[tid]);
    }
}

/**
 * Phase 2: Write attention + diagonal gate output
 *
 * gate = silu(z * g_z + read * g_r + h_work * g_h + b_gate)
 * output = h_work * gate
 */
template<int N_SLOTS>
__global__ void E29cPhase2Kernel_BF16(
    const int DIM,
    const int batch_size,
    const __nv_bfloat16* __restrict__ h_tape,
    const __nv_bfloat16* __restrict__ h_work_new,
    const __nv_bfloat16* __restrict__ z_t,
    const __nv_bfloat16* __restrict__ read_val,
    const __nv_bfloat16* __restrict__ write_val,
    const __nv_bfloat16* __restrict__ g_z,
    const __nv_bfloat16* __restrict__ g_r,
    const __nv_bfloat16* __restrict__ g_h,
    const __nv_bfloat16* __restrict__ b_gate,
    __nv_bfloat16* __restrict__ h_tape_out,
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ write_attn_out,
    const float scale
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    extern __shared__ char shared_mem[];
    float* write_val_sh = (float*)shared_mem;
    float* attn_sh = write_val_sh + DIM;

    // Load write_val
    for (int d = tid; d < DIM; d += BLOCK_SIZE) {
        write_val_sh[d] = __bfloat162float(write_val[b * DIM + d]);
    }
    __syncthreads();

    // Compute write attention scores
    if (tid < N_SLOTS) {
        float score = 0.0f;
        const __nv_bfloat16* tape_slot = h_tape + b * N_SLOTS * DIM + tid * DIM;
        for (int d = 0; d < DIM; d++) {
            score += __bfloat162float(tape_slot[d]) * write_val_sh[d];
        }
        attn_sh[tid] = score;
    }
    __syncthreads();

    // Softmax
    if (tid == 0) {
        softmax_device<N_SLOTS>(attn_sh, scale);
    }
    __syncthreads();

    // Update tape and compute output with DIAGONAL gate
    for (int d = tid; d < DIM; d += BLOCK_SIZE) {
        float wv = write_val_sh[d];

        // Update each tape slot
        #pragma unroll
        for (int n = 0; n < N_SLOTS; n++) {
            float old_val = __bfloat162float(h_tape[b * N_SLOTS * DIM + n * DIM + d]);
            float new_val = old_val * (1.0f - attn_sh[n]) + wv * attn_sh[n];
            h_tape_out[b * N_SLOTS * DIM + n * DIM + d] = __float2bfloat16(new_val);
        }

        // E29c DIAGONAL GATE: gate = silu(z * g_z + read * g_r + h * g_h + b)
        float z = __bfloat162float(z_t[b * DIM + d]);
        float rv = __bfloat162float(read_val[b * DIM + d]);
        float hw = __bfloat162float(h_work_new[b * DIM + d]);
        float gz = __bfloat162float(g_z[d]);
        float gr = __bfloat162float(g_r[d]);
        float gh = __bfloat162float(g_h[d]);
        float bg = __bfloat162float(b_gate[d]);

        float gate_input = clamp_val(z * gz + rv * gr + hw * gh + bg);
        float gate = silu_fwd(gate_input);
        output[b * DIM + d] = __float2bfloat16(hw * gate);
    }

    // Store write attention
    if (tid < N_SLOTS) {
        write_attn_out[b * N_SLOTS + tid] = __float2bfloat16(attn_sh[tid]);
    }
}

// =============================================================================
// BACKWARD KERNELS - Hybrid template
// =============================================================================

/**
 * Initialize backward gradients
 */
__global__ void E29cBackwardInit_BF16(
    const int DIM,
    const int batch_size,
    const int n_slots,
    const __nv_bfloat16* __restrict__ d_h_tape_final,
    __nv_bfloat16* __restrict__ d_h_tape,
    __nv_bfloat16* __restrict__ d_h_work
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize d_h_tape
    if (idx < batch_size * n_slots * DIM) {
        d_h_tape[idx] = d_h_tape_final[idx];
    }

    // Initialize d_h_work to zero
    if (idx < batch_size * DIM) {
        d_h_work[idx] = __float2bfloat16(0.0f);
    }
}

/**
 * Recompute read_val from read_attn and h_tape for backward pass
 * read_val[b, d] = sum_n(read_attn[b, n] * h_tape[b, n, d])
 */
template<int N_SLOTS>
__global__ void E29cRecomputeReadVal_BF16(
    const int DIM,
    const int batch_size,
    const __nv_bfloat16* __restrict__ read_attn,  // [B, N]
    const __nv_bfloat16* __restrict__ h_tape,     // [B, N, D]
    __nv_bfloat16* __restrict__ read_val_out      // [B, D]
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    extern __shared__ char shared_mem[];
    float* attn_sh = (float*)shared_mem;

    // Load read_attn into shared memory
    if (tid < N_SLOTS) {
        attn_sh[tid] = __bfloat162float(read_attn[b * N_SLOTS + tid]);
    }
    __syncthreads();

    // Compute read_val = attn @ tape
    for (int d = tid; d < DIM; d += BLOCK_SIZE) {
        float val = 0.0f;
        #pragma unroll
        for (int n = 0; n < N_SLOTS; n++) {
            val += attn_sh[n] * __bfloat162float(h_tape[b * N_SLOTS * DIM + n * DIM + d]);
        }
        read_val_out[b * DIM + d] = __float2bfloat16(val);
    }
}

/**
 * Backward Phase 1: Gate backward with diagonal weights
 *
 * Computes d_h_work_t, d_z_t, d_read_val_from_gate, and accumulates dg_z, dg_r, dg_h, db_gate
 */
template<int N_SLOTS>
__global__ void E29cBackwardGateKernel_BF16(
    const int DIM,
    const int batch_size,
    const __nv_bfloat16* __restrict__ h_work_t,
    const __nv_bfloat16* __restrict__ z_t,
    const __nv_bfloat16* __restrict__ read_val,
    const __nv_bfloat16* __restrict__ g_z,
    const __nv_bfloat16* __restrict__ g_r,
    const __nv_bfloat16* __restrict__ g_h,
    const __nv_bfloat16* __restrict__ b_gate,
    const __nv_bfloat16* __restrict__ d_output_t,
    const __nv_bfloat16* __restrict__ d_h_work_carry,
    __nv_bfloat16* __restrict__ d_h_work_t_out,
    __nv_bfloat16* __restrict__ d_z_t_out,
    __nv_bfloat16* __restrict__ d_read_val_from_gate_out,
    float* __restrict__ dg_z_accum,
    float* __restrict__ dg_r_accum,
    float* __restrict__ dg_h_accum,
    float* __restrict__ db_gate_accum
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    for (int d = tid; d < DIM; d += BLOCK_SIZE) {
        float hw = __bfloat162float(h_work_t[b * DIM + d]);
        float z = __bfloat162float(z_t[b * DIM + d]);
        float rv = __bfloat162float(read_val[b * DIM + d]);
        float gz = __bfloat162float(g_z[d]);
        float gr = __bfloat162float(g_r[d]);
        float gh = __bfloat162float(g_h[d]);
        float bg = __bfloat162float(b_gate[d]);

        // Recompute gate (with same clamping as forward)
        float gate_input = clamp_val(z * gz + rv * gr + hw * gh + bg);
        float gate = silu_fwd(gate_input);

        // d_output for this timestep
        float d_out = __bfloat162float(d_output_t[b * DIM + d]) +
                      __bfloat162float(d_h_work_carry[b * DIM + d]);

        // output = h_work * gate
        float d_gate = d_out * hw;
        float d_h_work_from_output = d_out * gate;

        // silu backward
        float d_silu = silu_bwd(gate_input);
        float d_gate_input = d_gate * d_silu;

        // Diagonal gate gradients (element-wise!)
        float d_z = d_gate_input * gz;
        float d_rv_from_gate = d_gate_input * gr;
        float d_hw_from_gate = d_gate_input * gh;

        // Accumulate diagonal weight gradients
        atomicAdd(&dg_z_accum[d], d_gate_input * z);
        atomicAdd(&dg_r_accum[d], d_gate_input * rv);
        atomicAdd(&dg_h_accum[d], d_gate_input * hw);
        atomicAdd(&db_gate_accum[d], d_gate_input);

        // Output gradients
        d_h_work_t_out[b * DIM + d] = __float2bfloat16(d_h_work_from_output + d_hw_from_gate);
        d_z_t_out[b * DIM + d] = __float2bfloat16(d_z);
        d_read_val_from_gate_out[b * DIM + d] = __float2bfloat16(d_rv_from_gate);
    }
}

/**
 * Backward Phase 2: Write attention backward
 */
template<int N_SLOTS>
__global__ void E29cBackwardWriteKernel_BF16(
    const int DIM,
    const int batch_size,
    const __nv_bfloat16* __restrict__ h_work_t,
    const __nv_bfloat16* __restrict__ h_tape_t,
    const __nv_bfloat16* __restrict__ write_attn,
    const __nv_bfloat16* __restrict__ W_write,
    const __nv_bfloat16* __restrict__ d_h_tape,
    const __nv_bfloat16* __restrict__ d_h_work_t_in,
    __nv_bfloat16* __restrict__ d_h_work_t_out,
    __nv_bfloat16* __restrict__ d_h_tape_pre_write,
    __nv_bfloat16* __restrict__ d_h_tape_from_write_attn,
    __nv_bfloat16* __restrict__ d_write_val_out,
    const float scale
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    extern __shared__ char shared_mem[];
    float* write_val_sh = (float*)shared_mem;
    float* attn_sh = write_val_sh + DIM;
    float* d_write_attn_sh = attn_sh + N_SLOTS;

    // Load write_attn
    if (tid < N_SLOTS) {
        attn_sh[tid] = __bfloat162float(write_attn[b * N_SLOTS + tid]);
    }
    __syncthreads();

    // Recompute write_val = h_work_t @ W_write.T
    for (int d = tid; d < DIM; d += BLOCK_SIZE) {
        float val = 0.0f;
        for (int k = 0; k < DIM; k++) {
            val += __bfloat162float(h_work_t[b * DIM + k]) *
                   __bfloat162float(W_write[d * DIM + k]);
        }
        write_val_sh[d] = val;
    }
    __syncthreads();

    // Compute d_write_val and d_write_attn
    float d_write_val_local[4];  // Local accumulator
    for (int i = 0; i < 4; i++) d_write_val_local[i] = 0.0f;

    if (tid < N_SLOTS) {
        float d_attn = 0.0f;
        for (int d = 0; d < DIM; d++) {
            float d_tape = __bfloat162float(d_h_tape[b * N_SLOTS * DIM + tid * DIM + d]);
            float tape_val = __bfloat162float(h_tape_t[b * N_SLOTS * DIM + tid * DIM + d]);
            d_attn += d_tape * (write_val_sh[d] - tape_val);
        }
        d_write_attn_sh[tid] = d_attn;
    }
    __syncthreads();

    // Softmax backward for write attention
    if (tid == 0) {
        float p_dp_sum = 0.0f;
        #pragma unroll
        for (int n = 0; n < N_SLOTS; n++) {
            p_dp_sum += attn_sh[n] * d_write_attn_sh[n];
        }
        #pragma unroll
        for (int n = 0; n < N_SLOTS; n++) {
            d_write_attn_sh[n] = attn_sh[n] * (d_write_attn_sh[n] - p_dp_sum) * scale;
        }
    }
    __syncthreads();

    // Compute d_write_val, d_h_tape_pre_write, and d_h_tape_from_write_attn
    for (int d = tid; d < DIM; d += BLOCK_SIZE) {
        float d_wv = 0.0f;
        float wv = write_val_sh[d];
        #pragma unroll
        for (int n = 0; n < N_SLOTS; n++) {
            float d_tape = __bfloat162float(d_h_tape[b * N_SLOTS * DIM + n * DIM + d]);
            d_wv += d_tape * attn_sh[n];

            // d_h_tape_pre_write
            float d_tape_pre = d_tape * (1.0f - attn_sh[n]);
            d_h_tape_pre_write[b * N_SLOTS * DIM + n * DIM + d] = __float2bfloat16(d_tape_pre);

            // d_h_tape_from_write_attn = d_write_scores * write_val
            float d_tape_from_attn = d_write_attn_sh[n] * wv;
            d_h_tape_from_write_attn[b * N_SLOTS * DIM + n * DIM + d] = __float2bfloat16(d_tape_from_attn);
        }
        d_write_val_out[b * DIM + d] = __float2bfloat16(d_wv);
    }
    __syncthreads();

    // Compute d_h_work from write_attn only (d_write_val @ W_write is done via GEMM after this kernel)
    for (int d = tid; d < DIM; d += BLOCK_SIZE) {
        float d_hw = __bfloat162float(d_h_work_t_in[b * DIM + d]);

        // d_h_work from write attention
        #pragma unroll
        for (int n = 0; n < N_SLOTS; n++) {
            d_hw += d_write_attn_sh[n] * __bfloat162float(h_tape_t[b * N_SLOTS * DIM + n * DIM + d]);
        }

        d_h_work_t_out[b * DIM + d] = __float2bfloat16(d_hw);
    }
}

/**
 * Backward Phase 3: tanh backward
 */
__global__ void E29cBackwardTanhKernel_BF16(
    const int DIM,
    const int batch_size,
    const __nv_bfloat16* __restrict__ h_work_t,
    const __nv_bfloat16* __restrict__ d_h_work_t_total,
    const __nv_bfloat16* __restrict__ d_read_val_from_gate,
    __nv_bfloat16* __restrict__ d_pre_act_out,
    __nv_bfloat16* __restrict__ d_read_val_out
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * DIM) return;

    float hw = __bfloat162float(h_work_t[idx]);
    float d_hw = __bfloat162float(d_h_work_t_total[idx]);
    float d_rv_gate = __bfloat162float(d_read_val_from_gate[idx]);

    // tanh backward: d_pre_act = d_h_work * (1 - h_work^2)
    float d_pre = d_hw * (1.0f - hw * hw);

    d_pre_act_out[idx] = __float2bfloat16(d_pre);
    d_read_val_out[idx] = __float2bfloat16(d_pre + d_rv_gate);
}

/**
 * Accumulate db_h from d_pre_act over batch dimension
 * db_h[d] += sum_b(d_pre_act[b, d])
 */
__global__ void E29cAccumulateDbH_BF16(
    const int DIM,
    const int batch_size,
    const __nv_bfloat16* __restrict__ d_pre_act,
    float* __restrict__ db_h_accum
) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= DIM) return;

    float sum = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        sum += __bfloat162float(d_pre_act[b * DIM + d]);
    }
    atomicAdd(&db_h_accum[d], sum);
}

/**
 * Backward Phase 4: Read attention backward
 */
template<int N_SLOTS>
__global__ void E29cBackwardReadKernel_BF16(
    const int DIM,
    const int batch_size,
    const __nv_bfloat16* __restrict__ h_tape_t,
    const __nv_bfloat16* __restrict__ h_work_prev,
    const __nv_bfloat16* __restrict__ read_attn,
    const __nv_bfloat16* __restrict__ d_read_val,
    const __nv_bfloat16* __restrict__ d_h_tape_pre_write,
    const __nv_bfloat16* __restrict__ d_h_tape_from_write_attn,
    __nv_bfloat16* __restrict__ d_h_work_carry,
    __nv_bfloat16* __restrict__ d_h_tape_out,
    const float scale
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    extern __shared__ char shared_mem[];
    float* attn_sh = (float*)shared_mem;
    float* d_read_attn_sh = attn_sh + N_SLOTS;

    // Load read_attn
    if (tid < N_SLOTS) {
        attn_sh[tid] = __bfloat162float(read_attn[b * N_SLOTS + tid]);
    }
    __syncthreads();

    // Compute d_read_attn
    if (tid < N_SLOTS) {
        float d_attn = 0.0f;
        for (int d = 0; d < DIM; d++) {
            float d_rv = __bfloat162float(d_read_val[b * DIM + d]);
            float tape_val = __bfloat162float(h_tape_t[b * N_SLOTS * DIM + tid * DIM + d]);
            d_attn += d_rv * tape_val;
        }
        d_read_attn_sh[tid] = d_attn;
    }
    __syncthreads();

    // Softmax backward
    if (tid == 0) {
        float p_dp_sum = 0.0f;
        #pragma unroll
        for (int n = 0; n < N_SLOTS; n++) {
            p_dp_sum += attn_sh[n] * d_read_attn_sh[n];
        }
        #pragma unroll
        for (int n = 0; n < N_SLOTS; n++) {
            d_read_attn_sh[n] = attn_sh[n] * (d_read_attn_sh[n] - p_dp_sum) * scale;
        }
    }
    __syncthreads();

    // Compute d_h_work_carry (from read attention) - ACCUMULATE to existing gradient
    for (int d = tid; d < DIM; d += BLOCK_SIZE) {
        // Start from existing gradient (includes d_pre_act @ W_h from earlier)
        float d_hw = __bfloat162float(d_h_work_carry[b * DIM + d]);
        #pragma unroll
        for (int n = 0; n < N_SLOTS; n++) {
            d_hw += d_read_attn_sh[n] * __bfloat162float(h_tape_t[b * N_SLOTS * DIM + n * DIM + d]);
        }
        d_h_work_carry[b * DIM + d] = __float2bfloat16(d_hw);
    }

    // Compute d_h_tape
    for (int d = tid; d < DIM; d += BLOCK_SIZE) {
        float d_rv = __bfloat162float(d_read_val[b * DIM + d]);

        #pragma unroll
        for (int n = 0; n < N_SLOTS; n++) {
            float d_tape = __bfloat162float(d_h_tape_pre_write[b * N_SLOTS * DIM + n * DIM + d]);
            d_tape += __bfloat162float(d_h_tape_from_write_attn[b * N_SLOTS * DIM + n * DIM + d]);
            d_tape += d_rv * attn_sh[n];
            d_tape += d_read_attn_sh[n] * __bfloat162float(h_work_prev[b * DIM + d]);
            d_h_tape_out[b * N_SLOTS * DIM + n * DIM + d] = __float2bfloat16(d_tape);
        }
    }
}

// =============================================================================
// Launch wrappers with N_SLOTS dispatch
// =============================================================================

template<int N_SLOTS>
void launch_e29c_phase1(
    int DIM, int batch_size,
    const __nv_bfloat16* Rh, const __nv_bfloat16* x_proj_t,
    const __nv_bfloat16* b_h, const __nv_bfloat16* h_tape,
    const __nv_bfloat16* h_work, __nv_bfloat16* h_work_out,
    __nv_bfloat16* read_attn_out, __nv_bfloat16* read_val_out,
    float scale, cudaStream_t stream
) {
    size_t shared_size = (2 * DIM + N_SLOTS) * sizeof(float);
    E29cPhase1Kernel_BF16<N_SLOTS><<<batch_size, BLOCK_SIZE, shared_size, stream>>>(
        DIM, batch_size, Rh, x_proj_t, b_h, h_tape, h_work,
        h_work_out, read_attn_out, read_val_out, scale
    );
}

template<int N_SLOTS>
void launch_e29c_phase2(
    int DIM, int batch_size,
    const __nv_bfloat16* h_tape, const __nv_bfloat16* h_work_new,
    const __nv_bfloat16* z_t, const __nv_bfloat16* read_val,
    const __nv_bfloat16* write_val,
    const __nv_bfloat16* g_z, const __nv_bfloat16* g_r,
    const __nv_bfloat16* g_h, const __nv_bfloat16* b_gate,
    __nv_bfloat16* h_tape_out, __nv_bfloat16* output,
    __nv_bfloat16* write_attn_out, float scale, cudaStream_t stream
) {
    size_t shared_size = (DIM + N_SLOTS) * sizeof(float);
    E29cPhase2Kernel_BF16<N_SLOTS><<<batch_size, BLOCK_SIZE, shared_size, stream>>>(
        DIM, batch_size, h_tape, h_work_new, z_t, read_val, write_val,
        g_z, g_r, g_h, b_gate, h_tape_out, output, write_attn_out, scale
    );
}

}  // namespace

// =============================================================================
// E29c Forward Implementation
// =============================================================================

template<>
E29cDiagonalForward<__nv_bfloat16>::E29cDiagonalForward(
    int batch_size, int n_slots, int dim,
    const cublasHandle_t& blas_handle, const cudaStream_t& stream)
    : batch_size_(batch_size), n_slots_(n_slots), dim_(dim),
      blas_handle_(blas_handle), stream_(stream) {

    // Allocate workspace for intermediate results
    size_t Rh_size = batch_size * dim * sizeof(__nv_bfloat16);
    size_t h_work_size = batch_size * dim * sizeof(__nv_bfloat16);
    size_t read_val_size = batch_size * dim * sizeof(__nv_bfloat16);
    size_t write_val_size = batch_size * dim * sizeof(__nv_bfloat16);

    cudaMalloc(&Rh_workspace_, Rh_size);
    cudaMalloc(&h_work_workspace_, h_work_size);
    cudaMalloc(&read_val_workspace_, read_val_size);
    cudaMalloc(&write_val_workspace_, write_val_size);
}

template<>
E29cDiagonalForward<__nv_bfloat16>::~E29cDiagonalForward() {
    cudaFree(Rh_workspace_);
    cudaFree(h_work_workspace_);
    cudaFree(read_val_workspace_);
    cudaFree(write_val_workspace_);
}

template<>
void E29cDiagonalForward<__nv_bfloat16>::Run(
    int seq_len,
    const __nv_bfloat16* x_proj,
    const __nv_bfloat16* z,
    const __nv_bfloat16* h_tape_init,
    const __nv_bfloat16* h_work_init,
    const __nv_bfloat16* W_h,
    const __nv_bfloat16* b_h,
    const __nv_bfloat16* W_write,
    const __nv_bfloat16* g_z,
    const __nv_bfloat16* g_r,
    const __nv_bfloat16* g_h,
    const __nv_bfloat16* b_gate,
    __nv_bfloat16* output_all,
    __nv_bfloat16* h_work_all,
    __nv_bfloat16* h_tape_all,
    __nv_bfloat16* read_attn_all,
    __nv_bfloat16* write_attn_all,
    __nv_bfloat16* read_val_all,
    cublasHandle_t blas_handle
) {
    const float scale = 1.0f / sqrtf((float)dim_);
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Copy initial states
    cudaMemcpyAsync(h_tape_all, h_tape_init,
                    batch_size_ * n_slots_ * dim_ * sizeof(__nv_bfloat16),
                    cudaMemcpyDeviceToDevice, stream_);
    cudaMemcpyAsync(h_work_workspace_, h_work_init,
                    batch_size_ * dim_ * sizeof(__nv_bfloat16),
                    cudaMemcpyDeviceToDevice, stream_);

    __nv_bfloat16* h_tape_curr = h_tape_all;
    __nv_bfloat16* h_tape_next = h_tape_all + batch_size_ * n_slots_ * dim_;

    for (int t = 0; t < seq_len; ++t) {
        const __nv_bfloat16* x_proj_t = x_proj + t * batch_size_ * dim_;
        const __nv_bfloat16* z_t = z + t * batch_size_ * dim_;
        __nv_bfloat16* output_t = output_all + t * batch_size_ * dim_;
        __nv_bfloat16* h_work_t = h_work_all + t * batch_size_ * dim_;
        __nv_bfloat16* read_attn_t = read_attn_all + t * batch_size_ * n_slots_;
        __nv_bfloat16* write_attn_t = write_attn_all + t * batch_size_ * n_slots_;

        // Where to store read_val: output array if training, workspace otherwise
        __nv_bfloat16* read_val_t = read_val_all ?
            read_val_all + t * batch_size_ * dim_ : read_val_workspace_;

        // Rh = h_work @ W_h.T
        cublasGemmEx(blas_handle,
                     CUBLAS_OP_T, CUBLAS_OP_N,
                     dim_, batch_size_, dim_,
                     &alpha,
                     W_h, CUDA_R_16BF, dim_,
                     h_work_workspace_, CUDA_R_16BF, dim_,
                     &beta,
                     Rh_workspace_, CUDA_R_16BF, dim_,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // Phase 1: Read + h_work update
        switch (n_slots_) {
            case 8:
                launch_e29c_phase1<8>(dim_, batch_size_, Rh_workspace_, x_proj_t, b_h,
                                       h_tape_curr, h_work_workspace_, h_work_t,
                                       read_attn_t, read_val_t, scale, stream_);
                break;
            case 16:
                launch_e29c_phase1<16>(dim_, batch_size_, Rh_workspace_, x_proj_t, b_h,
                                        h_tape_curr, h_work_workspace_, h_work_t,
                                        read_attn_t, read_val_t, scale, stream_);
                break;
            case 32:
                launch_e29c_phase1<32>(dim_, batch_size_, Rh_workspace_, x_proj_t, b_h,
                                        h_tape_curr, h_work_workspace_, h_work_t,
                                        read_attn_t, read_val_t, scale, stream_);
                break;
            case 64:
                launch_e29c_phase1<64>(dim_, batch_size_, Rh_workspace_, x_proj_t, b_h,
                                        h_tape_curr, h_work_workspace_, h_work_t,
                                        read_attn_t, read_val_t, scale, stream_);
                break;
            default:
                fprintf(stderr, "E29c CUDA: unsupported n_slots=%d\n", n_slots_);
                return;
        }

        // write_val = h_work_new @ W_write.T
        cublasGemmEx(blas_handle,
                     CUBLAS_OP_T, CUBLAS_OP_N,
                     dim_, batch_size_, dim_,
                     &alpha,
                     W_write, CUDA_R_16BF, dim_,
                     h_work_t, CUDA_R_16BF, dim_,
                     &beta,
                     write_val_workspace_, CUDA_R_16BF, dim_,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // Phase 2: Write + output with diagonal gate (reads from same read_val location)
        switch (n_slots_) {
            case 8:
                launch_e29c_phase2<8>(dim_, batch_size_, h_tape_curr, h_work_t, z_t,
                                       read_val_t, write_val_workspace_,
                                       g_z, g_r, g_h, b_gate,
                                       h_tape_next, output_t, write_attn_t, scale, stream_);
                break;
            case 16:
                launch_e29c_phase2<16>(dim_, batch_size_, h_tape_curr, h_work_t, z_t,
                                        read_val_t, write_val_workspace_,
                                        g_z, g_r, g_h, b_gate,
                                        h_tape_next, output_t, write_attn_t, scale, stream_);
                break;
            case 32:
                launch_e29c_phase2<32>(dim_, batch_size_, h_tape_curr, h_work_t, z_t,
                                        read_val_t, write_val_workspace_,
                                        g_z, g_r, g_h, b_gate,
                                        h_tape_next, output_t, write_attn_t, scale, stream_);
                break;
            case 64:
                launch_e29c_phase2<64>(dim_, batch_size_, h_tape_curr, h_work_t, z_t,
                                        read_val_t, write_val_workspace_,
                                        g_z, g_r, g_h, b_gate,
                                        h_tape_next, output_t, write_attn_t, scale, stream_);
                break;
        }

        // Update h_work for next timestep
        cudaMemcpyAsync(h_work_workspace_, h_work_t,
                        batch_size_ * dim_ * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToDevice, stream_);

        // Swap tape pointers
        h_tape_curr = h_tape_next;
        h_tape_next = h_tape_all + ((t + 2) % (seq_len + 1)) * batch_size_ * n_slots_ * dim_;
    }
}

// =============================================================================
// E29c Backward Implementation
// =============================================================================

template<>
E29cDiagonalBackward<__nv_bfloat16>::E29cDiagonalBackward(
    int batch_size, int n_slots, int dim,
    const cublasHandle_t& blas_handle, const cudaStream_t& stream)
    : batch_size_(batch_size), n_slots_(n_slots), dim_(dim),
      blas_handle_(blas_handle), stream_(stream) {

    // Allocate workspace
    cudaMalloc(&d_h_tape_workspace_, batch_size * n_slots * dim * sizeof(__nv_bfloat16));
    cudaMalloc(&d_h_work_workspace_, batch_size * dim * sizeof(__nv_bfloat16));
    cudaMalloc(&d_h_work_t_workspace_, batch_size * dim * sizeof(__nv_bfloat16));
    cudaMalloc(&d_z_t_workspace_, batch_size * dim * sizeof(__nv_bfloat16));
    cudaMalloc(&d_read_val_workspace_, batch_size * dim * sizeof(__nv_bfloat16));
    cudaMalloc(&d_pre_act_workspace_, batch_size * dim * sizeof(__nv_bfloat16));
    cudaMalloc(&d_write_val_workspace_, batch_size * dim * sizeof(__nv_bfloat16));
    cudaMalloc(&d_h_tape_pre_write_workspace_, batch_size * n_slots * dim * sizeof(__nv_bfloat16));
    cudaMalloc(&d_h_tape_from_write_workspace_, batch_size * n_slots * dim * sizeof(__nv_bfloat16));
    // Note: read_val_workspace_ no longer needed in backward - we use read_val_all from forward

    // Float accumulators for diagonal weight gradients
    cudaMalloc(&dg_z_accum_, dim * sizeof(float));
    cudaMalloc(&dg_r_accum_, dim * sizeof(float));
    cudaMalloc(&dg_h_accum_, dim * sizeof(float));
    cudaMalloc(&db_gate_accum_, dim * sizeof(float));
    cudaMalloc(&db_h_accum_, dim * sizeof(float));
}

template<>
E29cDiagonalBackward<__nv_bfloat16>::~E29cDiagonalBackward() {
    cudaFree(d_h_tape_workspace_);
    cudaFree(d_h_work_workspace_);
    cudaFree(d_h_work_t_workspace_);
    cudaFree(d_z_t_workspace_);
    cudaFree(d_read_val_workspace_);
    cudaFree(d_pre_act_workspace_);
    cudaFree(d_write_val_workspace_);
    cudaFree(d_h_tape_pre_write_workspace_);
    cudaFree(d_h_tape_from_write_workspace_);
    // Note: read_val_workspace_ no longer allocated in backward
    cudaFree(dg_z_accum_);
    cudaFree(dg_r_accum_);
    cudaFree(dg_h_accum_);
    cudaFree(db_gate_accum_);
    cudaFree(db_h_accum_);
}

template<>
void E29cDiagonalBackward<__nv_bfloat16>::Run(
    int seq_len,
    const __nv_bfloat16* h_work_all,
    const __nv_bfloat16* h_work_init,
    const __nv_bfloat16* h_tape_all,
    const __nv_bfloat16* read_attn_all,
    const __nv_bfloat16* write_attn_all,
    const __nv_bfloat16* read_val_all,
    const __nv_bfloat16* z_all,
    const __nv_bfloat16* W_h,
    const __nv_bfloat16* W_write,
    const __nv_bfloat16* g_z,
    const __nv_bfloat16* g_r,
    const __nv_bfloat16* g_h,
    const __nv_bfloat16* b_gate,
    const __nv_bfloat16* d_output_all,
    const __nv_bfloat16* d_h_tape_final,
    __nv_bfloat16* dx_proj,
    __nv_bfloat16* dz,
    __nv_bfloat16* d_pre_act_all,
    __nv_bfloat16* d_write_val_all,
    float* db_h,
    __nv_bfloat16* d_h_tape,
    float* dW_h,
    float* dW_write,
    float* dg_z_out,
    float* dg_r_out,
    float* dg_h_out,
    float* db_gate_out
) {
    const float scale = 1.0f / sqrtf((float)dim_);
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Initialize
    int init_threads = (batch_size_ * n_slots_ * dim_ + 255) / 256;
    E29cBackwardInit_BF16<<<init_threads, 256, 0, stream_>>>(
        dim_, batch_size_, n_slots_,
        d_h_tape_final, d_h_tape_workspace_, d_h_work_workspace_
    );

    // Zero diagonal weight accumulators
    cudaMemsetAsync(dg_z_accum_, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dg_r_accum_, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dg_h_accum_, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_gate_accum_, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_h_accum_, 0, dim_ * sizeof(float), stream_);

    // Backward loop
    for (int t = seq_len - 1; t >= 0; --t) {
        const __nv_bfloat16* h_work_t = h_work_all + t * batch_size_ * dim_;
        const __nv_bfloat16* h_tape_t = h_tape_all + t * batch_size_ * n_slots_ * dim_;
        const __nv_bfloat16* read_attn_t = read_attn_all + t * batch_size_ * n_slots_;
        const __nv_bfloat16* write_attn_t = write_attn_all + t * batch_size_ * n_slots_;
        const __nv_bfloat16* z_t = z_all + t * batch_size_ * dim_;
        const __nv_bfloat16* d_output_t = d_output_all + t * batch_size_ * dim_;
        const __nv_bfloat16* h_work_prev = (t > 0) ? h_work_all + (t - 1) * batch_size_ * dim_ : h_work_init;

        __nv_bfloat16* dx_proj_t = dx_proj + t * batch_size_ * dim_;
        __nv_bfloat16* dz_t = dz + t * batch_size_ * dim_;
        __nv_bfloat16* d_pre_act_t = d_pre_act_all + t * batch_size_ * dim_;
        __nv_bfloat16* d_write_val_t = d_write_val_all + t * batch_size_ * dim_;

        // Use read_val from forward (saved in read_val_all) - avoids recompute bug
        const __nv_bfloat16* read_val_t = read_val_all + t * batch_size_ * dim_;

        // Phase 1: Gate backward (diagonal - no GEMM needed!)
        switch (n_slots_) {
            case 8:
                E29cBackwardGateKernel_BF16<8><<<batch_size_, BLOCK_SIZE, 0, stream_>>>(
                    dim_, batch_size_, h_work_t, z_t, read_val_t,
                    g_z, g_r, g_h, b_gate, d_output_t, d_h_work_workspace_,
                    d_h_work_t_workspace_, d_z_t_workspace_, d_read_val_workspace_,
                    dg_z_accum_, dg_r_accum_, dg_h_accum_, db_gate_accum_
                );
                break;
            case 16:
                E29cBackwardGateKernel_BF16<16><<<batch_size_, BLOCK_SIZE, 0, stream_>>>(
                    dim_, batch_size_, h_work_t, z_t, read_val_t,
                    g_z, g_r, g_h, b_gate, d_output_t, d_h_work_workspace_,
                    d_h_work_t_workspace_, d_z_t_workspace_, d_read_val_workspace_,
                    dg_z_accum_, dg_r_accum_, dg_h_accum_, db_gate_accum_
                );
                break;
            case 32:
                E29cBackwardGateKernel_BF16<32><<<batch_size_, BLOCK_SIZE, 0, stream_>>>(
                    dim_, batch_size_, h_work_t, z_t, read_val_t,
                    g_z, g_r, g_h, b_gate, d_output_t, d_h_work_workspace_,
                    d_h_work_t_workspace_, d_z_t_workspace_, d_read_val_workspace_,
                    dg_z_accum_, dg_r_accum_, dg_h_accum_, db_gate_accum_
                );
                break;
            case 64:
                E29cBackwardGateKernel_BF16<64><<<batch_size_, BLOCK_SIZE, 0, stream_>>>(
                    dim_, batch_size_, h_work_t, z_t, read_val_t,
                    g_z, g_r, g_h, b_gate, d_output_t, d_h_work_workspace_,
                    d_h_work_t_workspace_, d_z_t_workspace_, d_read_val_workspace_,
                    dg_z_accum_, dg_r_accum_, dg_h_accum_, db_gate_accum_
                );
                break;
        }

        // Store dz_t
        cudaMemcpyAsync(dz_t, d_z_t_workspace_, batch_size_ * dim_ * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToDevice, stream_);

        // Phase 2: Write backward
        size_t write_shared = (dim_ + 2 * n_slots_) * sizeof(float);
        switch (n_slots_) {
            case 8:
                E29cBackwardWriteKernel_BF16<8><<<batch_size_, BLOCK_SIZE, write_shared, stream_>>>(
                    dim_, batch_size_, h_work_t, h_tape_t, write_attn_t, W_write,
                    d_h_tape_workspace_, d_h_work_t_workspace_, d_h_work_t_workspace_,
                    d_h_tape_pre_write_workspace_, d_h_tape_from_write_workspace_, d_write_val_t, scale
                );
                break;
            case 16:
                E29cBackwardWriteKernel_BF16<16><<<batch_size_, BLOCK_SIZE, write_shared, stream_>>>(
                    dim_, batch_size_, h_work_t, h_tape_t, write_attn_t, W_write,
                    d_h_tape_workspace_, d_h_work_t_workspace_, d_h_work_t_workspace_,
                    d_h_tape_pre_write_workspace_, d_h_tape_from_write_workspace_, d_write_val_t, scale
                );
                break;
            case 32:
                E29cBackwardWriteKernel_BF16<32><<<batch_size_, BLOCK_SIZE, write_shared, stream_>>>(
                    dim_, batch_size_, h_work_t, h_tape_t, write_attn_t, W_write,
                    d_h_tape_workspace_, d_h_work_t_workspace_, d_h_work_t_workspace_,
                    d_h_tape_pre_write_workspace_, d_h_tape_from_write_workspace_, d_write_val_t, scale
                );
                break;
            case 64:
                E29cBackwardWriteKernel_BF16<64><<<batch_size_, BLOCK_SIZE, write_shared, stream_>>>(
                    dim_, batch_size_, h_work_t, h_tape_t, write_attn_t, W_write,
                    d_h_tape_workspace_, d_h_work_t_workspace_, d_h_work_t_workspace_,
                    d_h_tape_pre_write_workspace_, d_h_tape_from_write_workspace_, d_write_val_t, scale
                );
                break;
        }

        // d_h_work_t += d_write_val @ W_write (GEMM for matrix multiplication)
        cublasGemmEx(blas_handle_,
                     CUBLAS_OP_N, CUBLAS_OP_N,
                     dim_, batch_size_, dim_,
                     &alpha,
                     W_write, CUDA_R_16BF, dim_,
                     d_write_val_t, CUDA_R_16BF, dim_,
                     &alpha,  // accumulate into d_h_work_t_workspace_
                     d_h_work_t_workspace_, CUDA_R_16BF, dim_,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // Phase 3: tanh backward
        int tanh_blocks = (batch_size_ * dim_ + 255) / 256;
        E29cBackwardTanhKernel_BF16<<<tanh_blocks, 256, 0, stream_>>>(
            dim_, batch_size_, h_work_t, d_h_work_t_workspace_,
            d_read_val_workspace_, d_pre_act_t, d_read_val_workspace_
        );

        // Accumulate db_h from d_pre_act
        int dbh_blocks = (dim_ + 255) / 256;
        E29cAccumulateDbH_BF16<<<dbh_blocks, 256, 0, stream_>>>(
            dim_, batch_size_, d_pre_act_t, db_h_accum_
        );

        // d_x_proj = d_pre_act
        cudaMemcpyAsync(dx_proj_t, d_pre_act_t, batch_size_ * dim_ * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToDevice, stream_);

        // dW_h += d_pre_act.T @ h_work_prev (accumulate)
        const float alpha_f = 1.0f;
        const float beta_f = 1.0f;
        cublasGemmEx(blas_handle_,
                     CUBLAS_OP_N, CUBLAS_OP_T,
                     dim_, dim_, batch_size_,
                     &alpha_f,
                     d_pre_act_t, CUDA_R_16BF, dim_,
                     h_work_prev, CUDA_R_16BF, dim_,
                     &beta_f,
                     dW_h, CUDA_R_32F, dim_,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // d_h_work = d_pre_act @ W_h (REPLACE - carry was consumed by gate backward)
        cublasGemmEx(blas_handle_,
                     CUBLAS_OP_N, CUBLAS_OP_N,
                     dim_, batch_size_, dim_,
                     &alpha,
                     W_h, CUDA_R_16BF, dim_,
                     d_pre_act_t, CUDA_R_16BF, dim_,
                     &beta,  // REPLACE, not accumulate!
                     d_h_work_workspace_, CUDA_R_16BF, dim_,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // dW_write += d_write_val.T @ h_work_t
        cublasGemmEx(blas_handle_,
                     CUBLAS_OP_N, CUBLAS_OP_T,
                     dim_, dim_, batch_size_,
                     &alpha_f,
                     d_write_val_t, CUDA_R_16BF, dim_,
                     h_work_t, CUDA_R_16BF, dim_,
                     &beta_f,
                     dW_write, CUDA_R_32F, dim_,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // Phase 4: Read backward
        size_t read_shared = 2 * n_slots_ * sizeof(float);
        switch (n_slots_) {
            case 8:
                E29cBackwardReadKernel_BF16<8><<<batch_size_, BLOCK_SIZE, read_shared, stream_>>>(
                    dim_, batch_size_, h_tape_t, h_work_prev, read_attn_t,
                    d_read_val_workspace_, d_h_tape_pre_write_workspace_,
                    d_h_tape_from_write_workspace_, d_h_work_workspace_,
                    d_h_tape_workspace_, scale
                );
                break;
            case 16:
                E29cBackwardReadKernel_BF16<16><<<batch_size_, BLOCK_SIZE, read_shared, stream_>>>(
                    dim_, batch_size_, h_tape_t, h_work_prev, read_attn_t,
                    d_read_val_workspace_, d_h_tape_pre_write_workspace_,
                    d_h_tape_from_write_workspace_, d_h_work_workspace_,
                    d_h_tape_workspace_, scale
                );
                break;
            case 32:
                E29cBackwardReadKernel_BF16<32><<<batch_size_, BLOCK_SIZE, read_shared, stream_>>>(
                    dim_, batch_size_, h_tape_t, h_work_prev, read_attn_t,
                    d_read_val_workspace_, d_h_tape_pre_write_workspace_,
                    d_h_tape_from_write_workspace_, d_h_work_workspace_,
                    d_h_tape_workspace_, scale
                );
                break;
            case 64:
                E29cBackwardReadKernel_BF16<64><<<batch_size_, BLOCK_SIZE, read_shared, stream_>>>(
                    dim_, batch_size_, h_tape_t, h_work_prev, read_attn_t,
                    d_read_val_workspace_, d_h_tape_pre_write_workspace_,
                    d_h_tape_from_write_workspace_, d_h_work_workspace_,
                    d_h_tape_workspace_, scale
                );
                break;
        }
    }

    // Copy diagonal weight gradients to output
    cudaMemcpyAsync(dg_z_out, dg_z_accum_, dim_ * sizeof(float), cudaMemcpyDeviceToDevice, stream_);
    cudaMemcpyAsync(dg_r_out, dg_r_accum_, dim_ * sizeof(float), cudaMemcpyDeviceToDevice, stream_);
    cudaMemcpyAsync(dg_h_out, dg_h_accum_, dim_ * sizeof(float), cudaMemcpyDeviceToDevice, stream_);
    cudaMemcpyAsync(db_gate_out, db_gate_accum_, dim_ * sizeof(float), cudaMemcpyDeviceToDevice, stream_);
    cudaMemcpyAsync(db_h, db_h_accum_, dim_ * sizeof(float), cudaMemcpyDeviceToDevice, stream_);
}

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
