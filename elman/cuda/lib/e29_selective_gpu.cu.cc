/**
 * E29: Selective Gating Dual-Memory Elman CUDA Kernels
 *
 * Extends E26 (softmax dual-memory) with selective output gating.
 *
 * E29a: gate = silu(z + read + h_work_new)  -- additive, no extra params
 * E29b: gate = silu(W_gate @ [z; read; h_work])  -- learned, +3D² params
 *
 * Key insight: Output gate depends on z (input), read (tape), AND h_work (hidden).
 * This is like Mamba's selective mechanism but for dual-memory RNNs.
 *
 * Uses hybrid template pattern: N_SLOTS as template param, DIM as runtime param
 * with dynamic shared memory allocation.
 */

#include "hasty/elman_ladder.h"
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

namespace {

constexpr int E29_BLOCK_SIZE = 256;

/**
 * Softmax device function for small N
 */
template<int N_SLOTS>
__device__ void e29_softmax_device(float* attn_sh, const int tid, const float scale) {
    __shared__ float max_val;
    if (tid == 0) {
        max_val = attn_sh[0] * scale;
        #pragma unroll
        for (int i = 1; i < N_SLOTS; i++) {
            float v = attn_sh[i] * scale;
            if (v > max_val) max_val = v;
        }
    }
    __syncthreads();

    __shared__ float sum_exp;
    if (tid == 0) {
        sum_exp = 0.0f;
        #pragma unroll
        for (int i = 0; i < N_SLOTS; i++) {
            float e = expf(attn_sh[i] * scale - max_val);
            attn_sh[i] = e;
            sum_exp += e;
        }
        sum_exp = fmaxf(sum_exp, 1e-9f);
    }
    __syncthreads();

    if (tid < N_SLOTS) {
        attn_sh[tid] /= sum_exp;
    }
}

/**
 * SiLU device function: silu(x) = x * sigmoid(x)
 */
__device__ __forceinline__ float silu_device(float x) {
    return x / (1.0f + expf(-x));
}

// =============================================================================
// E29a Kernels: Additive selective gate
// =============================================================================

/**
 * E29a Phase 1: Read attention + Update h_work + Compute read_val
 * Stores read_val for use in output phase
 *
 * Shared memory layout: [attn_sh: N_SLOTS][h_work_sh: DIM][read_val_sh: DIM]
 */
template<int N_SLOTS>
__global__ void E29aPhase1Kernel_BF16(
    const int batch_size,
    const int DIM,
    const __nv_bfloat16* __restrict__ Rh,          // [B, D] W_h @ h_work_prev
    const __nv_bfloat16* __restrict__ x_proj_t,    // [B, D]
    const __nv_bfloat16* __restrict__ b_h,         // [D]
    const __nv_bfloat16* __restrict__ h_tape,      // [B, N, D]
    const __nv_bfloat16* __restrict__ h_work,      // [B, D]
    __nv_bfloat16* __restrict__ h_work_out,        // [B, D]
    __nv_bfloat16* __restrict__ read_attn_out,     // [B, N]
    __nv_bfloat16* __restrict__ read_val_out,      // [B, D] - store for output phase
    const float scale
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    extern __shared__ char shared_mem[];
    float* attn_sh = (float*)shared_mem;
    float* h_work_sh = attn_sh + N_SLOTS;
    float* read_val_sh = h_work_sh + DIM;

    // Load h_work into shared memory
    for (int d = tid; d < DIM; d += E29_BLOCK_SIZE) {
        h_work_sh[d] = __bfloat162float(h_work[b * DIM + d]);
    }
    __syncthreads();

    // Compute attention scores: score[n] = h_tape[n] @ h_work
    if (tid < N_SLOTS) {
        float score = 0.0f;
        const __nv_bfloat16* tape_n = h_tape + b * N_SLOTS * DIM + tid * DIM;
        for (int d = 0; d < DIM; d++) {
            score += __bfloat162float(tape_n[d]) * h_work_sh[d];
        }
        attn_sh[tid] = score;
    }
    __syncthreads();

    // Softmax
    e29_softmax_device<N_SLOTS>(attn_sh, tid, scale);
    __syncthreads();

    // Store read attention
    if (tid < N_SLOTS) {
        read_attn_out[b * N_SLOTS + tid] = __float2bfloat16(attn_sh[tid]);
    }

    // Compute read_val and h_work_new
    for (int d = tid; d < DIM; d += E29_BLOCK_SIZE) {
        float read_d = 0.0f;
        #pragma unroll
        for (int n = 0; n < N_SLOTS; n++) {
            read_d += attn_sh[n] * __bfloat162float(h_tape[b * N_SLOTS * DIM + n * DIM + d]);
        }
        read_val_sh[d] = read_d;

        float val = __bfloat162float(Rh[b * DIM + d])
                  + __bfloat162float(x_proj_t[b * DIM + d])
                  + read_d
                  + __bfloat162float(b_h[d]);

        h_work_out[b * DIM + d] = __float2bfloat16(tanhf(val));
        read_val_out[b * DIM + d] = __float2bfloat16(read_d);
    }
}

/**
 * E29a Phase 2: Write attention + Update tape
 *
 * Shared memory layout: [attn_sh: N_SLOTS][h_work_sh: DIM][write_val_sh: DIM]
 */
template<int N_SLOTS>
__global__ void E29aPhase2Kernel_BF16(
    const int batch_size,
    const int DIM,
    const __nv_bfloat16* __restrict__ write_val,   // [B, D] W_write @ h_work_new
    const __nv_bfloat16* __restrict__ h_work_new,  // [B, D]
    __nv_bfloat16* __restrict__ h_tape,            // [B, N, D]
    __nv_bfloat16* __restrict__ write_attn_out,    // [B, N]
    const float scale
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    extern __shared__ char shared_mem[];
    float* attn_sh = (float*)shared_mem;
    float* h_work_sh = attn_sh + N_SLOTS;
    float* write_val_sh = h_work_sh + DIM;

    for (int d = tid; d < DIM; d += E29_BLOCK_SIZE) {
        h_work_sh[d] = __bfloat162float(h_work_new[b * DIM + d]);
        write_val_sh[d] = __bfloat162float(write_val[b * DIM + d]);
    }
    __syncthreads();

    __nv_bfloat16* tape_b = h_tape + b * N_SLOTS * DIM;
    if (tid < N_SLOTS) {
        float score = 0.0f;
        for (int d = 0; d < DIM; d++) {
            score += __bfloat162float(tape_b[tid * DIM + d]) * h_work_sh[d];
        }
        attn_sh[tid] = score;
    }
    __syncthreads();

    e29_softmax_device<N_SLOTS>(attn_sh, tid, scale);
    __syncthreads();

    if (tid < N_SLOTS) {
        write_attn_out[b * N_SLOTS + tid] = __float2bfloat16(attn_sh[tid]);
    }

    for (int i = tid; i < N_SLOTS * DIM; i += E29_BLOCK_SIZE) {
        int n = i / DIM;
        int d = i % DIM;
        float attn_n = attn_sh[n];
        float old_val = __bfloat162float(tape_b[i]);
        float new_val = (1.0f - attn_n) * old_val + attn_n * write_val_sh[d];
        tape_b[i] = __float2bfloat16(new_val);
    }
}

/**
 * E29a Phase 3: Selective output gate
 * gate = silu(z + read_val + h_work_new)
 * output = h_work_new * gate
 *
 * No shared memory needed - simple element-wise kernel
 */
__global__ void E29aPhase3Kernel_BF16(
    const int batch_size,
    const int DIM,
    const __nv_bfloat16* __restrict__ z_t,         // [B, D]
    const __nv_bfloat16* __restrict__ read_val,    // [B, D]
    const __nv_bfloat16* __restrict__ h_work_new,  // [B, D]
    __nv_bfloat16* __restrict__ output             // [B, D]
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    for (int d = tid; d < DIM; d += E29_BLOCK_SIZE) {
        float z = __bfloat162float(z_t[b * DIM + d]);
        float r = __bfloat162float(read_val[b * DIM + d]);
        float h = __bfloat162float(h_work_new[b * DIM + d]);

        // E29a selective gate: silu(z + read + h_work)
        float gate_input = z + r + h;
        float gate = silu_device(gate_input);
        float out = h * gate;

        output[b * DIM + d] = __float2bfloat16(out);
    }
}

// =============================================================================
// E29b Kernels: Learned selective gate
// =============================================================================

/**
 * E29b Phase 3: Learned selective output gate
 * gate_input = [z; read_val; h_work_new]  (concatenated)
 * gate = silu(gate_input @ W_gate.T)
 * output = h_work_new * gate
 *
 * Shared memory layout: [gate_input_sh: 3*DIM]
 *
 * Note: W_gate is [D, 3*D], so we compute output[d] = silu(sum_i(gate_input[i] * W_gate[d, i]))
 */
__global__ void E29bPhase3Kernel_BF16(
    const int batch_size,
    const int DIM,
    const __nv_bfloat16* __restrict__ z_t,         // [B, D]
    const __nv_bfloat16* __restrict__ read_val,    // [B, D]
    const __nv_bfloat16* __restrict__ h_work_new,  // [B, D]
    const __nv_bfloat16* __restrict__ W_gate,      // [D, 3*D]
    __nv_bfloat16* __restrict__ output             // [B, D]
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    // Load gate_input components into shared memory
    extern __shared__ char shared_mem[];
    float* gate_input_sh = (float*)shared_mem;

    for (int d = tid; d < DIM; d += E29_BLOCK_SIZE) {
        gate_input_sh[d] = __bfloat162float(z_t[b * DIM + d]);
        gate_input_sh[DIM + d] = __bfloat162float(read_val[b * DIM + d]);
        gate_input_sh[2 * DIM + d] = __bfloat162float(h_work_new[b * DIM + d]);
    }
    __syncthreads();

    // Compute gate = silu(gate_input @ W_gate.T)
    for (int d = tid; d < DIM; d += E29_BLOCK_SIZE) {
        float gate_pre = 0.0f;
        const __nv_bfloat16* W_row = W_gate + d * 3 * DIM;
        for (int i = 0; i < 3 * DIM; i++) {
            gate_pre += gate_input_sh[i] * __bfloat162float(W_row[i]);
        }

        float gate = silu_device(gate_pre);
        float h = gate_input_sh[2 * DIM + d];  // h_work_new[d]
        float out = h * gate;

        output[b * DIM + d] = __float2bfloat16(out);
    }
}

// =============================================================================
// E29a Backward Kernels
// =============================================================================

/**
 * E29a Backward Phase 1: Initialize gradients
 */
template<int N_SLOTS>
__global__ void E29aBackwardInit_BF16(
    const int batch_size,
    const int DIM,
    const __nv_bfloat16* __restrict__ d_h_tape_final,
    __nv_bfloat16* __restrict__ d_h_tape,
    __nv_bfloat16* __restrict__ d_h_work
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = idx / (N_SLOTS * DIM);
    if (b >= batch_size) return;

    const int local_idx = idx % (N_SLOTS * DIM);
    d_h_tape[idx] = d_h_tape_final[idx];

    if (local_idx < DIM) {
        d_h_work[b * DIM + local_idx] = __float2bfloat16(0.0f);
    }
}

/**
 * E29a Backward Phase 2: Backward through selective gate
 * Forward: gate = silu(z + read + h_work); output = h_work * gate
 * Backward: d_gate = d_output * h_work
 *           d_h_work += d_output * gate
 *           d_gate_input = d_gate * silu'(gate_input)
 *           d_z = d_read = d_h_work_gate = d_gate_input
 *
 * No shared memory needed - simple element-wise kernel
 */
__global__ void E29aBackwardGateKernel_BF16(
    const int batch_size,
    const int DIM,
    const __nv_bfloat16* __restrict__ z_t,
    const __nv_bfloat16* __restrict__ read_val,
    const __nv_bfloat16* __restrict__ h_work,
    const __nv_bfloat16* __restrict__ d_output,
    __nv_bfloat16* __restrict__ d_z,
    __nv_bfloat16* __restrict__ d_read_val,
    __nv_bfloat16* __restrict__ d_h_work
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    for (int d = tid; d < DIM; d += E29_BLOCK_SIZE) {
        float z = __bfloat162float(z_t[b * DIM + d]);
        float r = __bfloat162float(read_val[b * DIM + d]);
        float h = __bfloat162float(h_work[b * DIM + d]);
        float d_out = __bfloat162float(d_output[b * DIM + d]);

        // Forward: gate_input = z + r + h; gate = silu(gate_input); output = h * gate
        float gate_input = z + r + h;
        float sigmoid_gi = 1.0f / (1.0f + expf(-gate_input));
        float gate = gate_input * sigmoid_gi;

        // d_gate = d_output * h
        float d_gate = d_out * h;

        // d_h_work from output multiplication
        float d_h_from_out = d_out * gate;

        // silu backward: d_silu/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        float d_silu = sigmoid_gi * (1.0f + gate_input * (1.0f - sigmoid_gi));
        float d_gate_input = d_gate * d_silu;

        // gate_input = z + read + h_work, so all get the same gradient
        d_z[b * DIM + d] = __float2bfloat16(d_gate_input);
        d_read_val[b * DIM + d] = __float2bfloat16(d_gate_input);

        // d_h_work gets contribution from both output and gate
        float d_h_total = d_h_from_out + d_gate_input;
        d_h_work[b * DIM + d] = __float2bfloat16(
            __bfloat162float(d_h_work[b * DIM + d]) + d_h_total
        );
    }
}

/**
 * E29a Backward Phase 3: Backward through write
 *
 * Shared memory layout: [attn_sh: N_SLOTS][d_h_work_sh: DIM]
 */
template<int N_SLOTS>
__global__ void E29aBackwardWriteKernel_BF16(
    const int batch_size,
    const int DIM,
    const __nv_bfloat16* __restrict__ write_attn,
    const __nv_bfloat16* __restrict__ h_tape,
    const __nv_bfloat16* __restrict__ h_work,
    __nv_bfloat16* __restrict__ d_h_tape,
    __nv_bfloat16* __restrict__ d_write_val,
    __nv_bfloat16* __restrict__ d_h_work
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    extern __shared__ char shared_mem[];
    float* attn_sh = (float*)shared_mem;
    float* d_h_work_sh = attn_sh + N_SLOTS;

    if (tid < N_SLOTS) {
        attn_sh[tid] = __bfloat162float(write_attn[b * N_SLOTS + tid]);
    }
    for (int d = tid; d < DIM; d += E29_BLOCK_SIZE) {
        d_h_work_sh[d] = __bfloat162float(d_h_work[b * DIM + d]);
    }
    __syncthreads();

    // d_write_val = sum_n(d_tape_n * attn_n)
    // d_tape_pre = d_tape * (1 - attn)
    __nv_bfloat16* d_tape_b = d_h_tape + b * N_SLOTS * DIM;
    for (int d = tid; d < DIM; d += E29_BLOCK_SIZE) {
        float d_wv = 0.0f;
        #pragma unroll
        for (int n = 0; n < N_SLOTS; n++) {
            float d_t = __bfloat162float(d_tape_b[n * DIM + d]);
            float a = attn_sh[n];
            d_wv += d_t * a;
            d_tape_b[n * DIM + d] = __float2bfloat16(d_t * (1.0f - a));
        }
        d_write_val[b * DIM + d] = __float2bfloat16(d_wv);
    }
}

/**
 * E29a Backward Phase 4: Backward through tanh
 *
 * No shared memory needed - simple element-wise kernel
 */
__global__ void E29aBackwardTanhKernel_BF16(
    const int batch_size,
    const int DIM,
    const __nv_bfloat16* __restrict__ h_work,
    __nv_bfloat16* __restrict__ d_h_work,
    __nv_bfloat16* __restrict__ d_pre_act,
    __nv_bfloat16* __restrict__ d_x_proj,
    float* __restrict__ db_h
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    for (int d = tid; d < DIM; d += E29_BLOCK_SIZE) {
        float h = __bfloat162float(h_work[b * DIM + d]);
        float dh = __bfloat162float(d_h_work[b * DIM + d]);
        float dpa = dh * (1.0f - h * h);

        d_pre_act[b * DIM + d] = __float2bfloat16(dpa);
        d_x_proj[b * DIM + d] = __float2bfloat16(dpa);
        atomicAdd(&db_h[d], dpa);
    }
}

/**
 * E29a Backward Phase 5: Backward through read attention
 *
 * Shared memory layout: [attn_sh: N_SLOTS][d_read_sh: DIM]
 */
template<int N_SLOTS>
__global__ void E29aBackwardReadKernel_BF16(
    const int batch_size,
    const int DIM,
    const __nv_bfloat16* __restrict__ read_attn,
    const __nv_bfloat16* __restrict__ d_pre_act,
    const __nv_bfloat16* __restrict__ d_read_val_gate,  // gradient from gate path
    const __nv_bfloat16* __restrict__ h_tape,
    const __nv_bfloat16* __restrict__ h_work_prev,
    const float scale,
    __nv_bfloat16* __restrict__ d_h_tape,
    __nv_bfloat16* __restrict__ d_h_work_prev
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    extern __shared__ char shared_mem[];
    float* attn_sh = (float*)shared_mem;
    float* d_read_sh = attn_sh + N_SLOTS;

    if (tid < N_SLOTS) {
        attn_sh[tid] = __bfloat162float(read_attn[b * N_SLOTS + tid]);
    }

    // d_read = d_pre_act (from h_work update) + d_read_val_gate (from gate)
    for (int d = tid; d < DIM; d += E29_BLOCK_SIZE) {
        d_read_sh[d] = __bfloat162float(d_pre_act[b * DIM + d])
                     + __bfloat162float(d_read_val_gate[b * DIM + d]);
    }
    __syncthreads();

    // d_tape from read: d_tape_n += d_read * attn_n
    __nv_bfloat16* d_tape_b = d_h_tape + b * N_SLOTS * DIM;
    for (int i = tid; i < N_SLOTS * DIM; i += E29_BLOCK_SIZE) {
        int n = i / DIM;
        int d = i % DIM;
        float d_t = __bfloat162float(d_tape_b[i]);
        d_t += d_read_sh[d] * attn_sh[n];
        d_tape_b[i] = __float2bfloat16(d_t);
    }
}

// =============================================================================
// E29b Backward Kernels
// =============================================================================

/**
 * E29b Backward through learned gate
 * Forward: gate_input = [z; read; h_work]; gate = silu(gate_input @ W_gate.T); output = h_work * gate
 *
 * Shared memory layout: [gate_input_sh: 3*DIM][d_gate_input_sh: 3*DIM]
 */
__global__ void E29bBackwardGateKernel_BF16(
    const int batch_size,
    const int DIM,
    const __nv_bfloat16* __restrict__ z_t,
    const __nv_bfloat16* __restrict__ read_val,
    const __nv_bfloat16* __restrict__ h_work,
    const __nv_bfloat16* __restrict__ W_gate,      // [D, 3*D]
    const __nv_bfloat16* __restrict__ d_output,
    __nv_bfloat16* __restrict__ d_z,
    __nv_bfloat16* __restrict__ d_read_val,
    __nv_bfloat16* __restrict__ d_h_work,
    float* __restrict__ dW_gate                     // [D, 3*D] accumulated
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    extern __shared__ char shared_mem[];
    float* gate_input_sh = (float*)shared_mem;
    float* d_gate_input_sh = gate_input_sh + 3 * DIM;

    // Load gate_input
    for (int d = tid; d < DIM; d += E29_BLOCK_SIZE) {
        gate_input_sh[d] = __bfloat162float(z_t[b * DIM + d]);
        gate_input_sh[DIM + d] = __bfloat162float(read_val[b * DIM + d]);
        gate_input_sh[2 * DIM + d] = __bfloat162float(h_work[b * DIM + d]);
    }
    __syncthreads();

    // Initialize d_gate_input to zero
    for (int i = tid; i < 3 * DIM; i += E29_BLOCK_SIZE) {
        d_gate_input_sh[i] = 0.0f;
    }
    __syncthreads();

    // For each output dimension d:
    // gate_pre[d] = sum_i(gate_input[i] * W_gate[d, i])
    // gate[d] = silu(gate_pre[d])
    // output[d] = h_work[d] * gate[d]
    for (int d = tid; d < DIM; d += E29_BLOCK_SIZE) {
        // Recompute gate_pre for this output dim
        float gate_pre = 0.0f;
        const __nv_bfloat16* W_row = W_gate + d * 3 * DIM;
        for (int i = 0; i < 3 * DIM; i++) {
            gate_pre += gate_input_sh[i] * __bfloat162float(W_row[i]);
        }

        float sigmoid_gp = 1.0f / (1.0f + expf(-gate_pre));
        float gate = gate_pre * sigmoid_gp;
        float h = gate_input_sh[2 * DIM + d];
        float d_out = __bfloat162float(d_output[b * DIM + d]);

        // d_gate = d_output * h_work
        float d_gate = d_out * h;

        // d_h_work from output = d_output * gate
        float d_h_from_out = d_out * gate;

        // silu backward
        float d_silu = sigmoid_gp * (1.0f + gate_pre * (1.0f - sigmoid_gp));
        float d_gate_pre = d_gate * d_silu;

        // dW_gate[d, i] += d_gate_pre * gate_input[i]
        for (int i = 0; i < 3 * DIM; i++) {
            atomicAdd(&dW_gate[d * 3 * DIM + i], d_gate_pre * gate_input_sh[i]);
        }

        // d_gate_input[i] += d_gate_pre * W_gate[d, i]
        for (int i = 0; i < 3 * DIM; i++) {
            atomicAdd(&d_gate_input_sh[i], d_gate_pre * __bfloat162float(W_row[i]));
        }

        // d_h_work from gate
        atomicAdd(&d_gate_input_sh[2 * DIM + d], d_h_from_out);
    }
    __syncthreads();

    // Write out d_z, d_read_val, d_h_work
    for (int d = tid; d < DIM; d += E29_BLOCK_SIZE) {
        d_z[b * DIM + d] = __float2bfloat16(d_gate_input_sh[d]);
        d_read_val[b * DIM + d] = __float2bfloat16(d_gate_input_sh[DIM + d]);
        d_h_work[b * DIM + d] = __float2bfloat16(
            __bfloat162float(d_h_work[b * DIM + d]) + d_gate_input_sh[2 * DIM + d]
        );
    }
}

}  // namespace


namespace hasty { namespace v0 { namespace elman_ladder {

// =============================================================================
// E29a Forward Implementation
// =============================================================================

template<typename T>
E29aSelectiveForward<T>::E29aSelectiveForward(
    bool training, int batch_size, int n_slots, int dim,
    const cublasHandle_t& blas_handle, const cudaStream_t& stream)
    : training_(training), batch_size_(batch_size), n_slots_(n_slots), dim_(dim),
      stream_(stream), blas_handle_(blas_handle) {}

template<>
void E29aSelectiveForward<__nv_bfloat16>::Run(
    int seq_len,
    const __nv_bfloat16* x_proj,      // [T, B, D] pre-computed x projections
    const __nv_bfloat16* z_all,       // [T, B, D] pre-computed z values
    const __nv_bfloat16* W_h,
    const __nv_bfloat16* b_h,
    const __nv_bfloat16* W_write,
    const __nv_bfloat16* h_tape_init,
    const __nv_bfloat16* h_work_init,
    __nv_bfloat16* output_all,        // [T, B, D]
    __nv_bfloat16* h_work_all,        // [T, B, D]
    __nv_bfloat16* h_tape_final,
    __nv_bfloat16* h_tape_all,
    __nv_bfloat16* read_attn_all,
    __nv_bfloat16* write_attn_all,
    __nv_bfloat16* workspace
) {
    seq_len_ = seq_len;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    const int BD = batch_size_ * dim_;
    const int BND = batch_size_ * n_slots_ * dim_;

    // Workspace layout:
    // tmp_Rh: [B, D]
    // tmp_write_val: [B, D]
    // tmp_read_val: [B, D]
    __nv_bfloat16* tmp_Rh = workspace;
    __nv_bfloat16* tmp_write_val = tmp_Rh + BD;
    __nv_bfloat16* tmp_read_val = tmp_write_val + BD;

    cudaMemcpyAsync(h_tape_final, h_tape_init, BND * sizeof(__nv_bfloat16),
                    cudaMemcpyDeviceToDevice, stream_);

    if (training_ && h_tape_all) {
        cudaMemcpyAsync(h_tape_all, h_tape_init, BND * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToDevice, stream_);
    }

    const float scale = 1.0f / sqrtf(static_cast<float>(dim_));
    const int num_blocks = batch_size_;

    // Shared memory sizes for Phase 1 and 2: [attn: N_SLOTS][h_work: DIM][extra: DIM]
    const size_t smem_phase1 = (n_slots_ + 2 * dim_) * sizeof(float);
    const size_t smem_phase2 = (n_slots_ + 2 * dim_) * sizeof(float);

    #define LAUNCH_E29A_PHASE1(N) \
        E29aPhase1Kernel_BF16<N><<<num_blocks, E29_BLOCK_SIZE, smem_phase1, stream_>>>( \
            batch_size_, dim_, tmp_Rh, x_proj_t, b_h, h_tape_final, h_work_prev, \
            h_work_cur, read_attn_t, tmp_read_val, scale)

    #define LAUNCH_E29A_PHASE2(N) \
        E29aPhase2Kernel_BF16<N><<<num_blocks, E29_BLOCK_SIZE, smem_phase2, stream_>>>( \
            batch_size_, dim_, tmp_write_val, h_work_cur, h_tape_final, write_attn_t, scale)

    for (int t = 0; t < seq_len; ++t) {
        const __nv_bfloat16* x_proj_t = x_proj + t * BD;
        const __nv_bfloat16* z_t = z_all + t * BD;
        const __nv_bfloat16* h_work_prev = (t == 0) ? h_work_init : (h_work_all + (t - 1) * BD);
        __nv_bfloat16* h_work_cur = h_work_all + t * BD;
        __nv_bfloat16* output_t = output_all + t * BD;
        __nv_bfloat16* read_attn_t = read_attn_all + t * batch_size_ * n_slots_;
        __nv_bfloat16* write_attn_t = write_attn_all + t * batch_size_ * n_slots_;

        // W_h @ h_work_prev
        cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha,
            W_h, CUDA_R_16BF, dim_, h_work_prev, CUDA_R_16BF, dim_,
            &beta, tmp_Rh, CUDA_R_16BF, dim_,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // Phase 1: read attention + h_work update + store read_val
        if (n_slots_ == 8) { LAUNCH_E29A_PHASE1(8); }
        else if (n_slots_ == 16) { LAUNCH_E29A_PHASE1(16); }
        else if (n_slots_ == 32) { LAUNCH_E29A_PHASE1(32); }
        else if (n_slots_ == 64) { LAUNCH_E29A_PHASE1(64); }
        else { fprintf(stderr, "E29a CUDA: unsupported n_slots=%d\n", n_slots_); }

        // W_write @ h_work_new
        cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha,
            W_write, CUDA_R_16BF, dim_, h_work_cur, CUDA_R_16BF, dim_,
            &beta, tmp_write_val, CUDA_R_16BF, dim_,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // Phase 2: write attention + tape update
        if (n_slots_ == 8) { LAUNCH_E29A_PHASE2(8); }
        else if (n_slots_ == 16) { LAUNCH_E29A_PHASE2(16); }
        else if (n_slots_ == 32) { LAUNCH_E29A_PHASE2(32); }
        else if (n_slots_ == 64) { LAUNCH_E29A_PHASE2(64); }

        // Phase 3: selective output gate (no template needed)
        E29aPhase3Kernel_BF16<<<num_blocks, E29_BLOCK_SIZE, 0, stream_>>>(
            batch_size_, dim_, z_t, tmp_read_val, h_work_cur, output_t);

        if (training_ && h_tape_all) {
            cudaMemcpyAsync(h_tape_all + (t + 1) * BND, h_tape_final,
                BND * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream_);
        }
    }

    #undef LAUNCH_E29A_PHASE1
    #undef LAUNCH_E29A_PHASE2
}

// =============================================================================
// E29b Forward Implementation
// =============================================================================

template<typename T>
E29bSelectiveForward<T>::E29bSelectiveForward(
    bool training, int batch_size, int n_slots, int dim,
    const cublasHandle_t& blas_handle, const cudaStream_t& stream)
    : training_(training), batch_size_(batch_size), n_slots_(n_slots), dim_(dim),
      stream_(stream), blas_handle_(blas_handle) {}

template<>
void E29bSelectiveForward<__nv_bfloat16>::Run(
    int seq_len,
    const __nv_bfloat16* x_proj,
    const __nv_bfloat16* z_all,
    const __nv_bfloat16* W_h,
    const __nv_bfloat16* b_h,
    const __nv_bfloat16* W_write,
    const __nv_bfloat16* W_gate,      // [D, 3*D]
    const __nv_bfloat16* h_tape_init,
    const __nv_bfloat16* h_work_init,
    __nv_bfloat16* output_all,
    __nv_bfloat16* h_work_all,
    __nv_bfloat16* h_tape_final,
    __nv_bfloat16* h_tape_all,
    __nv_bfloat16* read_attn_all,
    __nv_bfloat16* write_attn_all,
    __nv_bfloat16* workspace
) {
    seq_len_ = seq_len;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    const int BD = batch_size_ * dim_;
    const int BND = batch_size_ * n_slots_ * dim_;

    __nv_bfloat16* tmp_Rh = workspace;
    __nv_bfloat16* tmp_write_val = tmp_Rh + BD;
    __nv_bfloat16* tmp_read_val = tmp_write_val + BD;

    cudaMemcpyAsync(h_tape_final, h_tape_init, BND * sizeof(__nv_bfloat16),
                    cudaMemcpyDeviceToDevice, stream_);

    if (training_ && h_tape_all) {
        cudaMemcpyAsync(h_tape_all, h_tape_init, BND * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToDevice, stream_);
    }

    const float scale = 1.0f / sqrtf(static_cast<float>(dim_));
    const int num_blocks = batch_size_;

    // Shared memory sizes
    const size_t smem_phase1 = (n_slots_ + 2 * dim_) * sizeof(float);
    const size_t smem_phase2 = (n_slots_ + 2 * dim_) * sizeof(float);
    const size_t smem_phase3b = 3 * dim_ * sizeof(float);  // gate_input_sh

    // E29b uses same Phase 1 and Phase 2 as E29a, only Phase 3 differs
    #define LAUNCH_E29B_PHASE1(N) \
        E29aPhase1Kernel_BF16<N><<<num_blocks, E29_BLOCK_SIZE, smem_phase1, stream_>>>( \
            batch_size_, dim_, tmp_Rh, x_proj_t, b_h, h_tape_final, h_work_prev, \
            h_work_cur, read_attn_t, tmp_read_val, scale)

    #define LAUNCH_E29B_PHASE2(N) \
        E29aPhase2Kernel_BF16<N><<<num_blocks, E29_BLOCK_SIZE, smem_phase2, stream_>>>( \
            batch_size_, dim_, tmp_write_val, h_work_cur, h_tape_final, write_attn_t, scale)

    for (int t = 0; t < seq_len; ++t) {
        const __nv_bfloat16* x_proj_t = x_proj + t * BD;
        const __nv_bfloat16* z_t = z_all + t * BD;
        const __nv_bfloat16* h_work_prev = (t == 0) ? h_work_init : (h_work_all + (t - 1) * BD);
        __nv_bfloat16* h_work_cur = h_work_all + t * BD;
        __nv_bfloat16* output_t = output_all + t * BD;
        __nv_bfloat16* read_attn_t = read_attn_all + t * batch_size_ * n_slots_;
        __nv_bfloat16* write_attn_t = write_attn_all + t * batch_size_ * n_slots_;

        cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha,
            W_h, CUDA_R_16BF, dim_, h_work_prev, CUDA_R_16BF, dim_,
            &beta, tmp_Rh, CUDA_R_16BF, dim_,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        if (n_slots_ == 8) { LAUNCH_E29B_PHASE1(8); }
        else if (n_slots_ == 16) { LAUNCH_E29B_PHASE1(16); }
        else if (n_slots_ == 32) { LAUNCH_E29B_PHASE1(32); }
        else if (n_slots_ == 64) { LAUNCH_E29B_PHASE1(64); }
        else { fprintf(stderr, "E29b CUDA: unsupported n_slots=%d\n", n_slots_); }

        cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha,
            W_write, CUDA_R_16BF, dim_, h_work_cur, CUDA_R_16BF, dim_,
            &beta, tmp_write_val, CUDA_R_16BF, dim_,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        if (n_slots_ == 8) { LAUNCH_E29B_PHASE2(8); }
        else if (n_slots_ == 16) { LAUNCH_E29B_PHASE2(16); }
        else if (n_slots_ == 32) { LAUNCH_E29B_PHASE2(32); }
        else if (n_slots_ == 64) { LAUNCH_E29B_PHASE2(64); }

        // Phase 3: E29b learned gate (no N_SLOTS template needed)
        E29bPhase3Kernel_BF16<<<num_blocks, E29_BLOCK_SIZE, smem_phase3b, stream_>>>(
            batch_size_, dim_, z_t, tmp_read_val, h_work_cur, W_gate, output_t);

        if (training_ && h_tape_all) {
            cudaMemcpyAsync(h_tape_all + (t + 1) * BND, h_tape_final,
                BND * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream_);
        }
    }

    #undef LAUNCH_E29B_PHASE1
    #undef LAUNCH_E29B_PHASE2
}

// =============================================================================
// E29a Backward Implementation
// =============================================================================

template<typename T>
E29aSelectiveBackward<T>::E29aSelectiveBackward(
    int batch_size, int n_slots, int dim,
    const cublasHandle_t& blas_handle, const cudaStream_t& stream)
    : batch_size_(batch_size), n_slots_(n_slots), dim_(dim),
      stream_(stream), blas_handle_(blas_handle) {}

template<>
void E29aSelectiveBackward<__nv_bfloat16>::Run(
    int seq_len,
    const __nv_bfloat16* h_work_all,
    const __nv_bfloat16* h_work_init,
    const __nv_bfloat16* h_tape_all,
    const __nv_bfloat16* read_attn,
    const __nv_bfloat16* write_attn,
    const __nv_bfloat16* z_all,
    const __nv_bfloat16* W_h,
    const __nv_bfloat16* W_write,
    const __nv_bfloat16* d_output_all,
    const __nv_bfloat16* d_h_tape_final,
    __nv_bfloat16* dx_proj,
    __nv_bfloat16* dz,
    __nv_bfloat16* d_pre_act_all,
    __nv_bfloat16* d_write_val_all,
    float* db_h,
    __nv_bfloat16* d_h_tape,
    float* dW_h,
    float* dW_write
) {
    const int num_blocks = batch_size_;
    const int BD = batch_size_ * dim_;
    const int BN = batch_size_ * n_slots_;
    const int BND = batch_size_ * n_slots_ * dim_;
    const float alpha_one = 1.0f;
    const float beta_zero = 0.0f;
    const float beta_one = 1.0f;

    // Allocate workspace for intermediate gradients
    __nv_bfloat16* d_h_work;
    __nv_bfloat16* d_read_val;  // gradient from gate path
    __nv_bfloat16* tmp_read_val;  // recomputed read_val for each timestep
    cudaMalloc(&d_h_work, BD * sizeof(__nv_bfloat16));
    cudaMalloc(&d_read_val, BD * sizeof(__nv_bfloat16));
    cudaMalloc(&tmp_read_val, BD * sizeof(__nv_bfloat16));

    // Shared memory sizes
    const size_t smem_write = (n_slots_ + dim_) * sizeof(float);
    const size_t smem_read = (n_slots_ + dim_) * sizeof(float);

    // Initialize
    #define LAUNCH_E29A_BWD_INIT(N) \
        E29aBackwardInit_BF16<N><<<(batch_size_ * N * dim_ + 255) / 256, 256, 0, stream_>>>( \
            batch_size_, dim_, d_h_tape_final, d_h_tape, d_h_work)

    if (n_slots_ == 8) { LAUNCH_E29A_BWD_INIT(8); }
    else if (n_slots_ == 16) { LAUNCH_E29A_BWD_INIT(16); }
    else if (n_slots_ == 32) { LAUNCH_E29A_BWD_INIT(32); }
    else if (n_slots_ == 64) { LAUNCH_E29A_BWD_INIT(64); }

    const float scale = 1.0f / sqrtf(static_cast<float>(dim_));

    // Kernel for recomputing read_val from read_attn and h_tape
    auto recompute_read_val = [&](int t) {
        const __nv_bfloat16* read_attn_t = read_attn + t * BN;
        const __nv_bfloat16* h_tape_t = h_tape_all + t * BND;

        // read_val = sum_n(read_attn[n] * h_tape[n])
        // Simple kernel for this - could optimize later
        cudaMemsetAsync(tmp_read_val, 0, BD * sizeof(__nv_bfloat16), stream_);

        // Use a small kernel to compute read_val
        // For now, use batch GEMM: read_val[b,d] = sum_n attn[b,n] * tape[b,n,d]
        // This is equivalent to: read_val = einsum('bn,bnd->bd', attn, tape)
        // Can be done with batched GEMV
        for (int b = 0; b < batch_size_; ++b) {
            cublasGemmEx(blas_handle_,
                CUBLAS_OP_T, CUBLAS_OP_N,
                dim_, 1, n_slots_,
                &alpha_one,
                h_tape_t + b * n_slots_ * dim_, CUDA_R_16BF, dim_,  // [N, D].T
                read_attn_t + b * n_slots_, CUDA_R_16BF, n_slots_,  // [N, 1]
                &beta_zero,
                tmp_read_val + b * dim_, CUDA_R_16BF, dim_,  // [D, 1]
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        }
    };

    #define LAUNCH_E29A_BWD_WRITE(N) \
        E29aBackwardWriteKernel_BF16<N><<<num_blocks, E29_BLOCK_SIZE, smem_write, stream_>>>( \
            batch_size_, dim_, write_attn_t, h_tape_t, h_work_t, d_h_tape, d_write_val_t, d_h_work)

    #define LAUNCH_E29A_BWD_READ(N) \
        E29aBackwardReadKernel_BF16<N><<<num_blocks, E29_BLOCK_SIZE, smem_read, stream_>>>( \
            batch_size_, dim_, read_attn_t, d_pre_act_t, d_read_val, h_tape_t, h_work_prev, scale, d_h_tape, d_h_work)

    // Backward through time
    for (int t = seq_len - 1; t >= 0; --t) {
        const __nv_bfloat16* h_work_t = h_work_all + t * BD;
        const __nv_bfloat16* h_work_prev = (t == 0) ? h_work_init : (h_work_all + (t - 1) * BD);
        const __nv_bfloat16* h_tape_t = h_tape_all + t * BND;
        const __nv_bfloat16* read_attn_t = read_attn + t * BN;
        const __nv_bfloat16* write_attn_t = write_attn + t * BN;
        const __nv_bfloat16* z_t = z_all + t * BD;
        const __nv_bfloat16* d_output_t = d_output_all + t * BD;
        __nv_bfloat16* dx_proj_t = dx_proj + t * BD;
        __nv_bfloat16* dz_t = dz + t * BD;
        __nv_bfloat16* d_pre_act_t = d_pre_act_all + t * BD;
        __nv_bfloat16* d_write_val_t = d_write_val_all + t * BD;

        // Recompute read_val for this timestep
        recompute_read_val(t);

        // Phase 1: Backward through selective gate (no template needed)
        E29aBackwardGateKernel_BF16<<<num_blocks, E29_BLOCK_SIZE, 0, stream_>>>(
            batch_size_, dim_, z_t, tmp_read_val, h_work_t, d_output_t, dz_t, d_read_val, d_h_work);

        // Phase 2: Backward through write
        if (n_slots_ == 8) { LAUNCH_E29A_BWD_WRITE(8); }
        else if (n_slots_ == 16) { LAUNCH_E29A_BWD_WRITE(16); }
        else if (n_slots_ == 32) { LAUNCH_E29A_BWD_WRITE(32); }
        else if (n_slots_ == 64) { LAUNCH_E29A_BWD_WRITE(64); }

        // dW_write += d_write_val^T @ h_work
        cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_, &alpha_one,
            d_write_val_t, CUDA_R_16BF, dim_,
            h_work_t, CUDA_R_16BF, dim_,
            &beta_one, dW_write, CUDA_R_32F, dim_,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // d_h_work += d_write_val @ W_write
        cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha_one,
            W_write, CUDA_R_16BF, dim_,
            d_write_val_t, CUDA_R_16BF, dim_,
            &beta_one, d_h_work, CUDA_R_16BF, dim_,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // Phase 3: Backward through tanh (no template needed)
        E29aBackwardTanhKernel_BF16<<<num_blocks, E29_BLOCK_SIZE, 0, stream_>>>(
            batch_size_, dim_, h_work_t, d_h_work, d_pre_act_t, dx_proj_t, db_h);

        // dW_h += d_pre_act^T @ h_work_prev
        cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_, &alpha_one,
            d_pre_act_t, CUDA_R_16BF, dim_,
            h_work_prev, CUDA_R_16BF, dim_,
            &beta_one, dW_h, CUDA_R_32F, dim_,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // d_h_work_prev = d_pre_act @ W_h (propagate to previous timestep)
        cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha_one,
            W_h, CUDA_R_16BF, dim_,
            d_pre_act_t, CUDA_R_16BF, dim_,
            &beta_zero, d_h_work, CUDA_R_16BF, dim_,  // Reset d_h_work for next iteration
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // Phase 4: Backward through read (adds to d_h_tape and d_h_work)
        if (n_slots_ == 8) { LAUNCH_E29A_BWD_READ(8); }
        else if (n_slots_ == 16) { LAUNCH_E29A_BWD_READ(16); }
        else if (n_slots_ == 32) { LAUNCH_E29A_BWD_READ(32); }
        else if (n_slots_ == 64) { LAUNCH_E29A_BWD_READ(64); }
    }

    #undef LAUNCH_E29A_BWD_INIT
    #undef LAUNCH_E29A_BWD_WRITE
    #undef LAUNCH_E29A_BWD_READ

    cudaFree(d_h_work);
    cudaFree(d_read_val);
    cudaFree(tmp_read_val);
}

// =============================================================================
// E29b Backward Implementation
// =============================================================================

template<typename T>
E29bSelectiveBackward<T>::E29bSelectiveBackward(
    int batch_size, int n_slots, int dim,
    const cublasHandle_t& blas_handle, const cudaStream_t& stream)
    : batch_size_(batch_size), n_slots_(n_slots), dim_(dim),
      stream_(stream), blas_handle_(blas_handle) {}

template<>
void E29bSelectiveBackward<__nv_bfloat16>::Run(
    int seq_len,
    const __nv_bfloat16* h_work_all,
    const __nv_bfloat16* h_work_init,
    const __nv_bfloat16* h_tape_all,
    const __nv_bfloat16* read_attn,
    const __nv_bfloat16* write_attn,
    const __nv_bfloat16* z_all,
    const __nv_bfloat16* W_h,
    const __nv_bfloat16* W_write,
    const __nv_bfloat16* W_gate,
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
    float* dW_gate
) {
    const int num_blocks = batch_size_;
    const int BD = batch_size_ * dim_;
    const int BN = batch_size_ * n_slots_;
    const int BND = batch_size_ * n_slots_ * dim_;
    const float alpha_one = 1.0f;
    const float beta_zero = 0.0f;
    const float beta_one = 1.0f;

    __nv_bfloat16* d_h_work;
    __nv_bfloat16* d_read_val;
    __nv_bfloat16* tmp_read_val;
    cudaMalloc(&d_h_work, BD * sizeof(__nv_bfloat16));
    cudaMalloc(&d_read_val, BD * sizeof(__nv_bfloat16));
    cudaMalloc(&tmp_read_val, BD * sizeof(__nv_bfloat16));

    // Shared memory sizes
    const size_t smem_write = (n_slots_ + dim_) * sizeof(float);
    const size_t smem_read = (n_slots_ + dim_) * sizeof(float);
    const size_t smem_gate_bwd = 6 * dim_ * sizeof(float);  // gate_input + d_gate_input

    // Initialize (same as E29a)
    #define LAUNCH_E29B_BWD_INIT(N) \
        E29aBackwardInit_BF16<N><<<(batch_size_ * N * dim_ + 255) / 256, 256, 0, stream_>>>( \
            batch_size_, dim_, d_h_tape_final, d_h_tape, d_h_work)

    if (n_slots_ == 8) { LAUNCH_E29B_BWD_INIT(8); }
    else if (n_slots_ == 16) { LAUNCH_E29B_BWD_INIT(16); }
    else if (n_slots_ == 32) { LAUNCH_E29B_BWD_INIT(32); }
    else if (n_slots_ == 64) { LAUNCH_E29B_BWD_INIT(64); }

    const float scale = 1.0f / sqrtf(static_cast<float>(dim_));

    auto recompute_read_val = [&](int t) {
        const __nv_bfloat16* read_attn_t = read_attn + t * BN;
        const __nv_bfloat16* h_tape_t = h_tape_all + t * BND;
        cudaMemsetAsync(tmp_read_val, 0, BD * sizeof(__nv_bfloat16), stream_);
        for (int b = 0; b < batch_size_; ++b) {
            cublasGemmEx(blas_handle_,
                CUBLAS_OP_T, CUBLAS_OP_N,
                dim_, 1, n_slots_,
                &alpha_one,
                h_tape_t + b * n_slots_ * dim_, CUDA_R_16BF, dim_,
                read_attn_t + b * n_slots_, CUDA_R_16BF, n_slots_,
                &beta_zero,
                tmp_read_val + b * dim_, CUDA_R_16BF, dim_,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        }
    };

    #define LAUNCH_E29B_BWD_WRITE(N) \
        E29aBackwardWriteKernel_BF16<N><<<num_blocks, E29_BLOCK_SIZE, smem_write, stream_>>>( \
            batch_size_, dim_, write_attn_t, h_tape_t, h_work_t, d_h_tape, d_write_val_t, d_h_work)

    #define LAUNCH_E29B_BWD_READ(N) \
        E29aBackwardReadKernel_BF16<N><<<num_blocks, E29_BLOCK_SIZE, smem_read, stream_>>>( \
            batch_size_, dim_, read_attn_t, d_pre_act_t, d_read_val, h_tape_t, h_work_prev, scale, d_h_tape, d_h_work)

    for (int t = seq_len - 1; t >= 0; --t) {
        const __nv_bfloat16* h_work_t = h_work_all + t * BD;
        const __nv_bfloat16* h_work_prev = (t == 0) ? h_work_init : (h_work_all + (t - 1) * BD);
        const __nv_bfloat16* h_tape_t = h_tape_all + t * BND;
        const __nv_bfloat16* read_attn_t = read_attn + t * BN;
        const __nv_bfloat16* write_attn_t = write_attn + t * BN;
        const __nv_bfloat16* z_t = z_all + t * BD;
        const __nv_bfloat16* d_output_t = d_output_all + t * BD;
        __nv_bfloat16* dx_proj_t = dx_proj + t * BD;
        __nv_bfloat16* dz_t = dz + t * BD;
        __nv_bfloat16* d_pre_act_t = d_pre_act_all + t * BD;
        __nv_bfloat16* d_write_val_t = d_write_val_all + t * BD;

        recompute_read_val(t);

        // E29b gate backward (different from E29a - no template needed)
        E29bBackwardGateKernel_BF16<<<num_blocks, E29_BLOCK_SIZE, smem_gate_bwd, stream_>>>(
            batch_size_, dim_, z_t, tmp_read_val, h_work_t, W_gate, d_output_t, dz_t, d_read_val, d_h_work, dW_gate);

        // Rest is same as E29a
        if (n_slots_ == 8) { LAUNCH_E29B_BWD_WRITE(8); }
        else if (n_slots_ == 16) { LAUNCH_E29B_BWD_WRITE(16); }
        else if (n_slots_ == 32) { LAUNCH_E29B_BWD_WRITE(32); }
        else if (n_slots_ == 64) { LAUNCH_E29B_BWD_WRITE(64); }

        cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_, &alpha_one,
            d_write_val_t, CUDA_R_16BF, dim_,
            h_work_t, CUDA_R_16BF, dim_,
            &beta_one, dW_write, CUDA_R_32F, dim_,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha_one,
            W_write, CUDA_R_16BF, dim_,
            d_write_val_t, CUDA_R_16BF, dim_,
            &beta_one, d_h_work, CUDA_R_16BF, dim_,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        // Phase 3: Backward through tanh (no template needed)
        E29aBackwardTanhKernel_BF16<<<num_blocks, E29_BLOCK_SIZE, 0, stream_>>>(
            batch_size_, dim_, h_work_t, d_h_work, d_pre_act_t, dx_proj_t, db_h);

        cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_, &alpha_one,
            d_pre_act_t, CUDA_R_16BF, dim_,
            h_work_prev, CUDA_R_16BF, dim_,
            &beta_one, dW_h, CUDA_R_32F, dim_,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        cublasGemmEx(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha_one,
            W_h, CUDA_R_16BF, dim_,
            d_pre_act_t, CUDA_R_16BF, dim_,
            &beta_zero, d_h_work, CUDA_R_16BF, dim_,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        if (n_slots_ == 8) { LAUNCH_E29B_BWD_READ(8); }
        else if (n_slots_ == 16) { LAUNCH_E29B_BWD_READ(16); }
        else if (n_slots_ == 32) { LAUNCH_E29B_BWD_READ(32); }
        else if (n_slots_ == 64) { LAUNCH_E29B_BWD_READ(64); }
    }

    #undef LAUNCH_E29B_BWD_INIT
    #undef LAUNCH_E29B_BWD_WRITE
    #undef LAUNCH_E29B_BWD_READ

    cudaFree(d_h_work);
    cudaFree(d_read_val);
    cudaFree(tmp_read_val);
}

// Explicit instantiations
template class E29aSelectiveForward<__nv_bfloat16>;
template class E29bSelectiveForward<__nv_bfloat16>;
template class E29aSelectiveBackward<__nv_bfloat16>;
template class E29bSelectiveBackward<__nv_bfloat16>;

}}}  // namespace hasty::v0::elman_ladder
