// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E21: Structured Elman - MIMO with Nonlinear State Mixing
//
// Key operations:
// 1. MIMO update: update[b,h,n,p] = sum_r B[b,h,n,r] * X[b,h,p,r]
// 2. Nonlinear state: H = silu(alpha * H_prev + update)
// 3. Output reduction: y = sum_n H[b,h,n,p]
// 4. E18-A gating: output = y * silu(z + y)

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <algorithm>

#include "hasty/elman_ladder.h"
#include "blas.h"
#include "inline_ops.h"

namespace {

// =============================================================================
// Forward Kernels
// =============================================================================

// Fused MIMO update + nonlinear state + output reduction + gating
// This is the main forward kernel that does everything per timestep
//
// H_prev: [B, nheads, d_state, headdim]
// B_proj: [B, nheads, d_state, mimo_rank]
// X_proj: [B, nheads, headdim, mimo_rank]
// alpha: [B, nheads] (scalar decay per head)
// z: [B, d_inner]
// H_out: [B, nheads, d_state, headdim]
// output: [B, d_inner]
//
// Each thread handles one (b, h, p) position and loops over n, r
template<typename T, int MIMO_RANK>
__global__ void StructuredElmanForwardKernel(
    const int batch_size,
    const int nheads,
    const int d_state,
    const int headdim,
    const T* __restrict__ H_prev,
    const T* __restrict__ B_proj,
    const T* __restrict__ X_proj,
    const T* __restrict__ alpha_raw,
    const T* __restrict__ alpha_bias,
    const T* __restrict__ z,
    T* __restrict__ H_out,
    T* __restrict__ output,
    T* __restrict__ y_cache) {  // [B, d_inner] for backward

    const int d_inner = nheads * headdim;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * d_inner;

    if (idx < total) {
        const int b = idx / d_inner;
        const int hp = idx % d_inner;
        const int h = hp / headdim;
        const int p = hp % headdim;

        // Compute alpha = sigmoid(-softplus(raw + bias))
        float raw = static_cast<float>(alpha_raw[b * nheads + h]);
        float bias = static_cast<float>(alpha_bias[h]);
        float softplus_val = logf(1.0f + expf(raw + bias));
        float alpha = 1.0f / (1.0f + expf(softplus_val));  // sigmoid(-softplus)

        // Accumulate y = sum_n H_new[n, p]
        float y_acc = 0.0f;

        // Loop over state dimension
        for (int n = 0; n < d_state; ++n) {
            // MIMO update: update = sum_r B[n,r] * X[p,r]
            float update = 0.0f;

            const int B_base = ((b * nheads + h) * d_state + n) * MIMO_RANK;
            const int X_base = ((b * nheads + h) * headdim + p) * MIMO_RANK;

            #pragma unroll
            for (int r = 0; r < MIMO_RANK; ++r) {
                update += static_cast<float>(B_proj[B_base + r]) *
                          static_cast<float>(X_proj[X_base + r]);
            }

            // Get H_prev for this position
            const int h_idx = ((b * nheads + h) * d_state + n) * headdim + p;
            float h_prev_val = static_cast<float>(H_prev[h_idx]);

            // Nonlinear state update: H = silu(alpha * H_prev + update)
            float pre_act = alpha * h_prev_val + update;
            float sigmoid_val = 1.0f / (1.0f + expf(-pre_act));
            float h_new_val = pre_act * sigmoid_val;

            // Store H_new
            H_out[h_idx] = static_cast<T>(h_new_val);

            // Accumulate y
            y_acc += h_new_val;
        }

        // Cache y for backward
        if (y_cache) y_cache[idx] = static_cast<T>(y_acc);

        // E18-A gating: output = y * silu(z + y)
        float z_val = static_cast<float>(z[idx]);
        float gate_input = z_val + y_acc;
        float gate_sigmoid = 1.0f / (1.0f + expf(-gate_input));
        float gate = gate_input * gate_sigmoid;

        output[idx] = static_cast<T>(y_acc * gate);
    }
}

// Nonlinearity modes: 0=silu, 1=tanh, 2=linear
// BF16 specialization with nonlinearity mode
template<int MIMO_RANK>
__global__ void StructuredElmanForwardKernel_BF16(
    const int batch_size,
    const int nheads,
    const int d_state,
    const int headdim,
    const int nonlinearity_mode,    // 0=silu, 1=tanh, 2=linear
    const __nv_bfloat16* __restrict__ H_prev,
    const __nv_bfloat16* __restrict__ B_proj,
    const __nv_bfloat16* __restrict__ X_proj,
    const __nv_bfloat16* __restrict__ alpha_raw,
    const __nv_bfloat16* __restrict__ alpha_bias,
    const __nv_bfloat16* __restrict__ z,
    __nv_bfloat16* __restrict__ H_out,
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ y_cache) {

    const int d_inner = nheads * headdim;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * d_inner;

    if (idx < total) {
        const int b = idx / d_inner;
        const int hp = idx % d_inner;
        const int h = hp / headdim;
        const int p = hp % headdim;

        float raw = __bfloat162float(alpha_raw[b * nheads + h]);
        float bias = __bfloat162float(alpha_bias[h]);
        float softplus_val = logf(1.0f + __expf(raw + bias));
        float alpha = 1.0f / (1.0f + __expf(softplus_val));

        float y_acc = 0.0f;

        for (int n = 0; n < d_state; ++n) {
            float update = 0.0f;

            const int B_base = ((b * nheads + h) * d_state + n) * MIMO_RANK;
            const int X_base = ((b * nheads + h) * headdim + p) * MIMO_RANK;

            #pragma unroll
            for (int r = 0; r < MIMO_RANK; ++r) {
                update += __bfloat162float(B_proj[B_base + r]) *
                          __bfloat162float(X_proj[X_base + r]);
            }

            const int h_idx = ((b * nheads + h) * d_state + n) * headdim + p;
            float h_prev_val = __bfloat162float(H_prev[h_idx]);

            float pre_act = alpha * h_prev_val + update;
            float h_new_val;

            // Apply nonlinearity based on mode
            if (nonlinearity_mode == 0) {  // silu
                float sigmoid_val = 1.0f / (1.0f + __expf(-pre_act));
                h_new_val = pre_act * sigmoid_val;
            } else if (nonlinearity_mode == 1) {  // tanh
                h_new_val = tanhf(pre_act);
            } else {  // linear
                h_new_val = pre_act;
            }

            H_out[h_idx] = __float2bfloat16(h_new_val);
            y_acc += h_new_val;
        }

        if (y_cache) y_cache[idx] = __float2bfloat16(y_acc);

        float z_val = __bfloat162float(z[idx]);
        float gate_input = z_val + y_acc;
        float gate_sigmoid = 1.0f / (1.0f + __expf(-gate_input));
        float gate = gate_input * gate_sigmoid;

        output[idx] = __float2bfloat16(y_acc * gate);
    }
}

// =============================================================================
// Backward Kernels
// =============================================================================

// Backward through E18-A gate: output = y * silu(z + y)
// Returns dy and dz
template<typename T>
__global__ void GateBackwardKernel(
    const int batch_size,
    const int d_inner,
    const T* __restrict__ y,
    const T* __restrict__ z,
    const T* __restrict__ d_output,
    T* __restrict__ dy,
    T* __restrict__ dz) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * d_inner;

    if (idx < total) {
        float y_val = static_cast<float>(y[idx]);
        float z_val = static_cast<float>(z[idx]);
        float dout = static_cast<float>(d_output[idx]);

        float gate_input = z_val + y_val;
        float sigmoid_g = 1.0f / (1.0f + expf(-gate_input));
        float silu_g = gate_input * sigmoid_g;
        float dsilu = sigmoid_g * (1.0f + gate_input * (1.0f - sigmoid_g));

        // dy = dout * silu + dout * y * dsilu (y appears in gate_input)
        float dy_val = dout * silu_g + dout * y_val * dsilu;
        float dz_val = dout * y_val * dsilu;

        dy[idx] = static_cast<T>(dy_val);
        dz[idx] = static_cast<T>(dz_val);
    }
}

__global__ void GateBackwardKernel_BF16(
    const int batch_size,
    const int d_inner,
    const __nv_bfloat16* __restrict__ y,
    const __nv_bfloat16* __restrict__ z,
    const __nv_bfloat16* __restrict__ d_output,
    __nv_bfloat16* __restrict__ dy,
    __nv_bfloat16* __restrict__ dz) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * d_inner;

    if (idx < total) {
        float y_val = __bfloat162float(y[idx]);
        float z_val = __bfloat162float(z[idx]);
        float dout = __bfloat162float(d_output[idx]);

        float gate_input = z_val + y_val;
        float sigmoid_g = 1.0f / (1.0f + __expf(-gate_input));
        float silu_g = gate_input * sigmoid_g;
        float dsilu = sigmoid_g * (1.0f + gate_input * (1.0f - sigmoid_g));

        float dy_val = dout * silu_g + dout * y_val * dsilu;
        float dz_val = dout * y_val * dsilu;

        dy[idx] = __float2bfloat16(dy_val);
        dz[idx] = __float2bfloat16(dz_val);
    }
}

// Backward through state update + MIMO - OPTIMIZED VERSION
// Block structure: one block per (b, h), headdim threads per block
// This eliminates atomics for dX (unique per thread) and uses shared mem reduction for dB
// Now with nonlinearity_mode support: 0=silu, 1=tanh, 2=linear
template<typename T, int MIMO_RANK, int D_STATE, int HEADDIM>
__global__ void StructuredElmanBackwardKernel_Opt(
    const int batch_size,
    const int nheads,
    const int nonlinearity_mode,    // 0=silu, 1=tanh, 2=linear
    const T* __restrict__ H_prev,
    const T* __restrict__ H_new,
    const T* __restrict__ B_proj,
    const T* __restrict__ X_proj,
    const T* __restrict__ alpha_raw,
    const T* __restrict__ alpha_bias,
    const T* __restrict__ dy,
    const T* __restrict__ dH_out,      // from next timestep
    T* __restrict__ dH_prev,
    float* __restrict__ dB_f,
    float* __restrict__ dX_f,
    float* __restrict__ dalpha_f) {

    // Shared memory for dB reduction: [D_STATE][MIMO_RANK]
    __shared__ float dB_shared[D_STATE][MIMO_RANK];

    const int bh = blockIdx.x;  // one block per (b, h)
    const int p = threadIdx.x;  // thread per p (headdim threads per block)
    const int b = bh / nheads;
    const int h = bh % nheads;

    if (b >= batch_size) return;

    // Initialize shared memory
    if (p < D_STATE) {
        #pragma unroll
        for (int r = 0; r < MIMO_RANK; ++r) {
            dB_shared[p][r] = 0.0f;
        }
    }
    __syncthreads();

    // Compute alpha for this (b, h)
    float raw = static_cast<float>(alpha_raw[b * nheads + h]);
    float bias = static_cast<float>(alpha_bias[h]);
    float sp_input = raw + bias;
    float softplus_val = logf(1.0f + expf(sp_input));
    float alpha = 1.0f / (1.0f + expf(softplus_val));

    const int d_inner = nheads * HEADDIM;
    const int idx = b * d_inner + h * HEADDIM + p;
    float dy_val = static_cast<float>(dy[idx]);
    float dalpha_acc = 0.0f;

    // Local accumulators for dX (no atomics needed - unique per thread)
    float dX_local[MIMO_RANK];
    #pragma unroll
    for (int r = 0; r < MIMO_RANK; ++r) {
        dX_local[r] = 0.0f;
    }

    // Main loop over state dimension
    for (int n = 0; n < D_STATE; ++n) {
        const int h_idx = ((b * nheads + h) * D_STATE + n) * HEADDIM + p;
        const int B_base = ((b * nheads + h) * D_STATE + n) * MIMO_RANK;
        const int X_base = ((b * nheads + h) * HEADDIM + p) * MIMO_RANK;

        // Recompute pre_act
        float h_prev_val = static_cast<float>(H_prev[h_idx]);
        float update = 0.0f;
        float B_vals[MIMO_RANK], X_vals[MIMO_RANK];

        #pragma unroll
        for (int r = 0; r < MIMO_RANK; ++r) {
            B_vals[r] = static_cast<float>(B_proj[B_base + r]);
            X_vals[r] = static_cast<float>(X_proj[X_base + r]);
            update += B_vals[r] * X_vals[r];
        }
        float pre_act = alpha * h_prev_val + update;

        // Compute nonlinearity derivative based on mode
        float d_nonlin;
        if (nonlinearity_mode == 0) {  // silu
            float sigmoid_val = 1.0f / (1.0f + expf(-pre_act));
            d_nonlin = sigmoid_val * (1.0f + pre_act * (1.0f - sigmoid_val));
        } else if (nonlinearity_mode == 1) {  // tanh
            float h_new_val = tanhf(pre_act);
            d_nonlin = 1.0f - h_new_val * h_new_val;
        } else {  // linear
            d_nonlin = 1.0f;
        }

        // dH = dH_out + dy (y = sum_n H)
        float dH_from_out = static_cast<float>(dH_out[h_idx]);
        float dH = dH_from_out + dy_val;
        float d_pre_act = dH * d_nonlin;

        // dH_prev = d_pre_act * alpha (direct write - unique per thread)
        dH_prev[h_idx] = static_cast<T>(d_pre_act * alpha);

        float d_update = d_pre_act;

        // Accumulate dX locally (will write once at end - no atomic!)
        #pragma unroll
        for (int r = 0; r < MIMO_RANK; ++r) {
            dX_local[r] += d_update * B_vals[r];
        }

        // Accumulate dB to shared memory (atomic within block only)
        #pragma unroll
        for (int r = 0; r < MIMO_RANK; ++r) {
            atomicAdd(&dB_shared[n][r], d_update * X_vals[r]);
        }

        // dalpha accumulator
        dalpha_acc += d_pre_act * h_prev_val;
    }

    __syncthreads();

    // Write dX to global memory (NO ATOMIC - each thread has unique X_base!)
    const int X_base = ((b * nheads + h) * HEADDIM + p) * MIMO_RANK;
    #pragma unroll
    for (int r = 0; r < MIMO_RANK; ++r) {
        dX_f[X_base + r] = dX_local[r];
    }

    // Write dB from shared memory (one thread per n writes)
    if (p < D_STATE) {
        const int B_base = ((b * nheads + h) * D_STATE + p) * MIMO_RANK;
        #pragma unroll
        for (int r = 0; r < MIMO_RANK; ++r) {
            dB_f[B_base + r] = dB_shared[p][r];
        }
    }

    // Reduce dalpha within warp, then atomic to global
    // Use warp shuffle reduction
    float dalpha_sum = dalpha_acc;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        dalpha_sum += __shfl_down_sync(0xffffffff, dalpha_sum, offset);
    }

    // First thread of each warp does atomicAdd
    if ((p % 32) == 0) {
        float sig_sp = 1.0f / (1.0f + expf(softplus_val));
        float sig_raw = 1.0f / (1.0f + expf(-sp_input));
        float dalpha_d_raw = -sig_sp * (1.0f - sig_sp) * sig_raw;
        atomicAdd(&dalpha_f[b * nheads + h], dalpha_sum * dalpha_d_raw);
    }
}

// Legacy kernel for non-standard dimensions (fallback)
template<typename T, int MIMO_RANK>
__global__ void StructuredElmanBackwardKernel(
    const int batch_size,
    const int nheads,
    const int d_state,
    const int headdim,
    const int nonlinearity_mode,    // 0=silu, 1=tanh, 2=linear
    const T* __restrict__ H_prev,
    const T* __restrict__ H_new,
    const T* __restrict__ B_proj,
    const T* __restrict__ X_proj,
    const T* __restrict__ alpha_raw,
    const T* __restrict__ alpha_bias,
    const T* __restrict__ dy,
    const T* __restrict__ dH_out,      // from next timestep
    T* __restrict__ dH_prev,
    float* __restrict__ dB_f,
    float* __restrict__ dX_f,
    float* __restrict__ dalpha_f) {

    const int d_inner = nheads * headdim;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * d_inner;

    if (idx < total) {
        const int b = idx / d_inner;
        const int hp = idx % d_inner;
        const int h = hp / headdim;
        const int p = hp % headdim;

        // Recompute alpha
        float raw = static_cast<float>(alpha_raw[b * nheads + h]);
        float bias = static_cast<float>(alpha_bias[h]);
        float sp_input = raw + bias;
        float softplus_val = logf(1.0f + expf(sp_input));
        float alpha = 1.0f / (1.0f + expf(softplus_val));

        float dy_val = static_cast<float>(dy[idx]);
        float dalpha_acc = 0.0f;

        // Local dX accumulator
        float dX_local[16] = {0};  // Max MIMO_RANK

        for (int n = 0; n < d_state; ++n) {
            const int h_idx = ((b * nheads + h) * d_state + n) * headdim + p;
            const int B_base = ((b * nheads + h) * d_state + n) * MIMO_RANK;
            const int X_base = ((b * nheads + h) * headdim + p) * MIMO_RANK;

            // Recompute pre_act
            float h_prev_val = static_cast<float>(H_prev[h_idx]);
            float update = 0.0f;
            float B_vals[16], X_vals[16];
            #pragma unroll
            for (int r = 0; r < MIMO_RANK; ++r) {
                B_vals[r] = static_cast<float>(B_proj[B_base + r]);
                X_vals[r] = static_cast<float>(X_proj[X_base + r]);
                update += B_vals[r] * X_vals[r];
            }
            float pre_act = alpha * h_prev_val + update;

            // Compute nonlinearity derivative based on mode
            float d_nonlin;
            if (nonlinearity_mode == 0) {  // silu
                float sigmoid_val = 1.0f / (1.0f + expf(-pre_act));
                d_nonlin = sigmoid_val * (1.0f + pre_act * (1.0f - sigmoid_val));
            } else if (nonlinearity_mode == 1) {  // tanh
                float h_new_val = tanhf(pre_act);
                d_nonlin = 1.0f - h_new_val * h_new_val;  // dtanh = 1 - tanh^2
            } else {  // linear
                d_nonlin = 1.0f;
            }

            // dH = dH_out + dy (y = sum_n H)
            float dH_from_out = static_cast<float>(dH_out[h_idx]);
            float dH = dH_from_out + dy_val;

            // d_pre_act
            float d_pre_act = dH * d_nonlin;

            // dH_prev += d_pre_act * alpha
            dH_prev[h_idx] = static_cast<T>(d_pre_act * alpha);

            // d_update = d_pre_act
            float d_update = d_pre_act;

            // dX local accumulation (no atomic!)
            #pragma unroll
            for (int r = 0; r < MIMO_RANK; ++r) {
                dX_local[r] += d_update * B_vals[r];
            }

            // dB still needs atomic (different threads have same B_base)
            #pragma unroll
            for (int r = 0; r < MIMO_RANK; ++r) {
                atomicAdd(&dB_f[B_base + r], d_update * X_vals[r]);
            }

            // dalpha += d_pre_act * H_prev
            dalpha_acc += d_pre_act * h_prev_val;
        }

        // Write dX (no atomic - unique per thread!)
        const int X_base = ((b * nheads + h) * headdim + p) * MIMO_RANK;
        #pragma unroll
        for (int r = 0; r < MIMO_RANK; ++r) {
            dX_f[X_base + r] = dX_local[r];
        }

        // dalpha needs to go through sigmoid(-softplus) derivative
        float sig_sp = 1.0f / (1.0f + expf(softplus_val));
        float sig_raw = 1.0f / (1.0f + expf(-sp_input));
        float dalpha_d_raw = -sig_sp * (1.0f - sig_sp) * sig_raw;
        atomicAdd(&dalpha_f[b * nheads + h], dalpha_acc * dalpha_d_raw);
    }
}

// Optimized BF16 backward kernel V2 - fully parallel reduction, NO atomics
// Block per (b,h), HEADDIM threads per block
// Uses warp shuffles for all reductions
template<int MIMO_RANK, int D_STATE, int HEADDIM>
__global__ void StructuredElmanBackwardKernel_BF16_Opt(
    const int batch_size,
    const int nheads,
    const int nonlinearity_mode,    // 0=silu, 1=tanh, 2=linear
    const __nv_bfloat16* __restrict__ H_prev,
    const __nv_bfloat16* __restrict__ H_new,
    const __nv_bfloat16* __restrict__ B_proj,
    const __nv_bfloat16* __restrict__ X_proj,
    const __nv_bfloat16* __restrict__ alpha_raw,
    const __nv_bfloat16* __restrict__ alpha_bias,
    const __nv_bfloat16* __restrict__ dy,
    const __nv_bfloat16* __restrict__ dH_out,
    __nv_bfloat16* __restrict__ dH_prev,
    float* __restrict__ dB_f,
    float* __restrict__ dX_f,
    float* __restrict__ dalpha_f) {

    // Shared memory for cross-warp reduction
    // Layout: [2][D_STATE * MIMO_RANK] for dB reduction from 2 warps
    // Plus [2] for dalpha from 2 warps
    __shared__ float reduce_shared[2 * D_STATE * MIMO_RANK + 2];
    float* dB_warp = reduce_shared;  // [2][D_STATE * MIMO_RANK]
    float* dalpha_warp = reduce_shared + 2 * D_STATE * MIMO_RANK;  // [2]

    const int bh = blockIdx.x;
    const int p = threadIdx.x;  // 0..HEADDIM-1 (64 threads)
    const int b = bh / nheads;
    const int h = bh % nheads;
    const int warp_id = p / 32;
    const int lane_id = p % 32;

    if (b >= batch_size) return;

    // Compute alpha for this (b, h)
    float raw = __bfloat162float(alpha_raw[b * nheads + h]);
    float bias = __bfloat162float(alpha_bias[h]);
    float sp_input = raw + bias;
    float softplus_val = logf(1.0f + __expf(sp_input));
    float alpha = 1.0f / (1.0f + __expf(softplus_val));

    const int d_inner = nheads * HEADDIM;
    const int idx = b * d_inner + h * HEADDIM + p;
    float dy_val = __bfloat162float(dy[idx]);

    // Thread-local accumulators
    float dX_local[MIMO_RANK] = {0};
    float dalpha_local = 0.0f;

    // Process in chunks of n to reduce register pressure for dB
    // Each thread computes dB contributions for its p value
    // We'll reduce across p using warp shuffles + shared memory

    for (int n = 0; n < D_STATE; ++n) {
        const int h_idx = ((b * nheads + h) * D_STATE + n) * HEADDIM + p;
        const int B_base = ((b * nheads + h) * D_STATE + n) * MIMO_RANK;
        const int X_base = ((b * nheads + h) * HEADDIM + p) * MIMO_RANK;

        float h_prev_val = __bfloat162float(H_prev[h_idx]);
        float update = 0.0f;
        float B_vals[MIMO_RANK], X_vals[MIMO_RANK];

        #pragma unroll
        for (int r = 0; r < MIMO_RANK; ++r) {
            B_vals[r] = __bfloat162float(B_proj[B_base + r]);
            X_vals[r] = __bfloat162float(X_proj[X_base + r]);
            update += B_vals[r] * X_vals[r];
        }
        float pre_act = alpha * h_prev_val + update;

        float d_nonlin;
        if (nonlinearity_mode == 0) {
            float sigmoid_val = 1.0f / (1.0f + __expf(-pre_act));
            d_nonlin = sigmoid_val * (1.0f + pre_act * (1.0f - sigmoid_val));
        } else if (nonlinearity_mode == 1) {
            float h_new_val = tanhf(pre_act);
            d_nonlin = 1.0f - h_new_val * h_new_val;
        } else {
            d_nonlin = 1.0f;
        }

        float dH_from_out = __bfloat162float(dH_out[h_idx]);
        float dH = dH_from_out + dy_val;
        float d_pre_act = dH * d_nonlin;

        // Direct write for dH_prev (unique per thread)
        dH_prev[h_idx] = __float2bfloat16(d_pre_act * alpha);

        float d_update = d_pre_act;

        // Accumulate dX locally (unique per thread)
        #pragma unroll
        for (int r = 0; r < MIMO_RANK; ++r) {
            dX_local[r] += d_update * B_vals[r];
        }

        // dB reduction: each thread has d_update * X_vals[r] for this n
        // Need to sum across all 64 threads (p values)
        // Do warp-level reduction first, then cross-warp via shared mem
        #pragma unroll
        for (int r = 0; r < MIMO_RANK; ++r) {
            float dB_contrib = d_update * X_vals[r];

            // Warp reduction
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                dB_contrib += __shfl_down_sync(0xffffffff, dB_contrib, offset);
            }

            // Lane 0 of each warp writes to shared memory
            if (lane_id == 0) {
                dB_warp[warp_id * D_STATE * MIMO_RANK + n * MIMO_RANK + r] = dB_contrib;
            }
        }

        // dalpha: accumulate locally
        dalpha_local += d_pre_act * h_prev_val;
    }

    __syncthreads();

    // Write dX (no reduction needed - unique per thread)
    const int X_base = ((b * nheads + h) * HEADDIM + p) * MIMO_RANK;
    #pragma unroll
    for (int r = 0; r < MIMO_RANK; ++r) {
        dX_f[X_base + r] = dX_local[r];
    }

    // Final dB reduction: combine results from 2 warps
    // Thread p handles (n, r) where p < D_STATE * MIMO_RANK
    if (p < D_STATE * MIMO_RANK) {
        float dB_val = dB_warp[p] + dB_warp[D_STATE * MIMO_RANK + p];
        int n = p / MIMO_RANK;
        int r = p % MIMO_RANK;
        const int B_base = ((b * nheads + h) * D_STATE + n) * MIMO_RANK + r;
        dB_f[B_base] = dB_val;
    }

    // dalpha reduction: warp shuffle then cross-warp
    float dalpha_sum = dalpha_local;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        dalpha_sum += __shfl_down_sync(0xffffffff, dalpha_sum, offset);
    }
    if (lane_id == 0) {
        dalpha_warp[warp_id] = dalpha_sum;
    }
    __syncthreads();

    // Thread 0 combines both warps and writes
    if (p == 0) {
        float total_dalpha = dalpha_warp[0] + dalpha_warp[1];
        float sig_sp = 1.0f / (1.0f + __expf(softplus_val));
        float sig_raw = 1.0f / (1.0f + __expf(-sp_input));
        float dalpha_d_raw = -sig_sp * (1.0f - sig_sp) * sig_raw;
        dalpha_f[b * nheads + h] = total_dalpha * dalpha_d_raw;  // Direct write, no atomic!
    }
}

// Legacy fallback kernel for non-standard dimensions
template<int MIMO_RANK>
__global__ void StructuredElmanBackwardKernel_BF16(
    const int batch_size,
    const int nheads,
    const int d_state,
    const int headdim,
    const int nonlinearity_mode,    // 0=silu, 1=tanh, 2=linear
    const __nv_bfloat16* __restrict__ H_prev,
    const __nv_bfloat16* __restrict__ H_new,
    const __nv_bfloat16* __restrict__ B_proj,
    const __nv_bfloat16* __restrict__ X_proj,
    const __nv_bfloat16* __restrict__ alpha_raw,
    const __nv_bfloat16* __restrict__ alpha_bias,
    const __nv_bfloat16* __restrict__ dy,
    const __nv_bfloat16* __restrict__ dH_out,
    __nv_bfloat16* __restrict__ dH_prev,
    float* __restrict__ dB_f,
    float* __restrict__ dX_f,
    float* __restrict__ dalpha_f) {

    const int d_inner = nheads * headdim;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * d_inner;

    if (idx < total) {
        const int b = idx / d_inner;
        const int hp = idx % d_inner;
        const int h = hp / headdim;
        const int p = hp % headdim;

        float raw = __bfloat162float(alpha_raw[b * nheads + h]);
        float bias = __bfloat162float(alpha_bias[h]);
        float sp_input = raw + bias;
        float softplus_val = logf(1.0f + __expf(sp_input));
        float alpha = 1.0f / (1.0f + __expf(softplus_val));

        float dy_val = __bfloat162float(dy[idx]);
        float dalpha_acc = 0.0f;

        // Local dX accumulator
        float dX_local[16] = {0};

        for (int n = 0; n < d_state; ++n) {
            const int h_idx = ((b * nheads + h) * d_state + n) * headdim + p;
            const int B_base = ((b * nheads + h) * d_state + n) * MIMO_RANK;
            const int X_base = ((b * nheads + h) * headdim + p) * MIMO_RANK;

            float h_prev_val = __bfloat162float(H_prev[h_idx]);
            float update = 0.0f;
            float B_vals[16], X_vals[16];
            #pragma unroll
            for (int r = 0; r < MIMO_RANK; ++r) {
                B_vals[r] = __bfloat162float(B_proj[B_base + r]);
                X_vals[r] = __bfloat162float(X_proj[X_base + r]);
                update += B_vals[r] * X_vals[r];
            }
            float pre_act = alpha * h_prev_val + update;

            // Compute nonlinearity derivative based on mode
            float d_nonlin;
            if (nonlinearity_mode == 0) {  // silu
                float sigmoid_val = 1.0f / (1.0f + __expf(-pre_act));
                d_nonlin = sigmoid_val * (1.0f + pre_act * (1.0f - sigmoid_val));
            } else if (nonlinearity_mode == 1) {  // tanh
                float h_new_val = tanhf(pre_act);
                d_nonlin = 1.0f - h_new_val * h_new_val;
            } else {  // linear
                d_nonlin = 1.0f;
            }

            float dH_from_out = __bfloat162float(dH_out[h_idx]);
            float dH = dH_from_out + dy_val;
            float d_pre_act = dH * d_nonlin;

            dH_prev[h_idx] = __float2bfloat16(d_pre_act * alpha);

            float d_update = d_pre_act;

            // dX local accumulation (no atomic!)
            #pragma unroll
            for (int r = 0; r < MIMO_RANK; ++r) {
                dX_local[r] += d_update * B_vals[r];
            }

            // dB still needs atomic
            #pragma unroll
            for (int r = 0; r < MIMO_RANK; ++r) {
                atomicAdd(&dB_f[B_base + r], d_update * X_vals[r]);
            }

            dalpha_acc += d_pre_act * h_prev_val;
        }

        // Write dX (no atomic - unique per thread!)
        const int X_base = ((b * nheads + h) * headdim + p) * MIMO_RANK;
        #pragma unroll
        for (int r = 0; r < MIMO_RANK; ++r) {
            dX_f[X_base + r] = dX_local[r];
        }

        float sig_sp = 1.0f / (1.0f + __expf(softplus_val));
        float sig_raw = 1.0f / (1.0f + __expf(-sp_input));
        float dalpha_d_raw = -sig_sp * (1.0f - sig_sp) * sig_raw;
        atomicAdd(&dalpha_f[b * nheads + h], dalpha_acc * dalpha_d_raw);
    }
}

// Copy float to T
template<typename T>
__global__ void CopyFloatToT(const int n, const float* __restrict__ src, T* __restrict__ dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = static_cast<T>(src[idx]);
    }
}

// Sum dalpha_f over batches to get dalpha_bias_f
// dalpha_f: [B, nheads] -> dalpha_bias_f: [nheads]
__global__ void SumAlphaBiasKernel(
    const int batch_size,
    const int nheads,
    const float* __restrict__ dalpha_f,
    float* __restrict__ dalpha_bias_f) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h < nheads) {
        float sum = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            sum += dalpha_f[b * nheads + h];
        }
        atomicAdd(&dalpha_bias_f[h], sum);
    }
}

}  // anonymous namespace


namespace hasty {
namespace v0 {
namespace elman_ladder {

// =============================================================================
// StructuredElmanForward - BF16 Specialization
// =============================================================================

template<>
StructuredElmanForward<__nv_bfloat16>::StructuredElmanForward(
    bool training,
    int batch_size,
    int nheads,
    int d_state,
    int headdim,
    int mimo_rank,
    int nonlinearity_mode,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : training_(training),
      batch_size_(batch_size),
      nheads_(nheads),
      d_state_(d_state),
      headdim_(headdim),
      mimo_rank_(mimo_rank),
      nonlinearity_mode_(nonlinearity_mode),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void StructuredElmanForward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* B_proj,      // [T, B, nheads, d_state, mimo_rank]
    const __nv_bfloat16* X_proj,      // [T, B, nheads, headdim, mimo_rank]
    const __nv_bfloat16* alpha_raw,   // [T, B, nheads]
    const __nv_bfloat16* alpha_bias,  // [nheads]
    const __nv_bfloat16* z,           // [T, B, d_inner]
    __nv_bfloat16* H,                 // [(T+1), B, nheads, d_state, headdim]
    __nv_bfloat16* output,            // [T, B, d_inner]
    __nv_bfloat16* y_cache,           // [T, B, d_inner]
    __nv_bfloat16* workspace) {

    const int d_inner = nheads_ * headdim_;
    const int BD = batch_size_ * d_inner;
    const int state_size = nheads_ * d_state_ * headdim_;
    const int B_proj_step = batch_size_ * nheads_ * d_state_ * mimo_rank_;
    const int X_proj_step = batch_size_ * nheads_ * headdim_ * mimo_rank_;
    const int alpha_step = batch_size_ * nheads_;
    const int B_state = batch_size_ * state_size;
    const int block_size = 256;

    for (int t = 0; t < steps; ++t) {
        const __nv_bfloat16* B_t = B_proj + t * B_proj_step;
        const __nv_bfloat16* X_t = X_proj + t * X_proj_step;
        const __nv_bfloat16* alpha_t = alpha_raw + t * alpha_step;
        const __nv_bfloat16* z_t = z + t * BD;
        const __nv_bfloat16* H_prev = H + t * B_state;
        __nv_bfloat16* H_t = H + (t + 1) * B_state;
        __nv_bfloat16* out_t = output + t * BD;
        __nv_bfloat16* y_t = training_ ? (y_cache + t * BD) : nullptr;

        // Dispatch based on MIMO rank
        if (mimo_rank_ == 4) {
            StructuredElmanForwardKernel_BF16<4><<<(BD + block_size - 1) / block_size, block_size, 0, stream_>>>(
                batch_size_, nheads_, d_state_, headdim_, nonlinearity_mode_,
                H_prev, B_t, X_t, alpha_t, alpha_bias, z_t, H_t, out_t, y_t);
        } else if (mimo_rank_ == 8) {
            StructuredElmanForwardKernel_BF16<8><<<(BD + block_size - 1) / block_size, block_size, 0, stream_>>>(
                batch_size_, nheads_, d_state_, headdim_, nonlinearity_mode_,
                H_prev, B_t, X_t, alpha_t, alpha_bias, z_t, H_t, out_t, y_t);
        } else if (mimo_rank_ == 16) {
            StructuredElmanForwardKernel_BF16<16><<<(BD + block_size - 1) / block_size, block_size, 0, stream_>>>(
                batch_size_, nheads_, d_state_, headdim_, nonlinearity_mode_,
                H_prev, B_t, X_t, alpha_t, alpha_bias, z_t, H_t, out_t, y_t);
        } else {
            // Fallback - not template optimized
            StructuredElmanForwardKernel_BF16<8><<<(BD + block_size - 1) / block_size, block_size, 0, stream_>>>(
                batch_size_, nheads_, d_state_, headdim_, nonlinearity_mode_,
                H_prev, B_t, X_t, alpha_t, alpha_bias, z_t, H_t, out_t, y_t);
        }
    }
}

// =============================================================================
// StructuredElmanBackward - BF16 Specialization
// =============================================================================

template<>
StructuredElmanBackward<__nv_bfloat16>::StructuredElmanBackward(
    int batch_size,
    int nheads,
    int d_state,
    int headdim,
    int mimo_rank,
    int nonlinearity_mode,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      nheads_(nheads),
      d_state_(d_state),
      headdim_(headdim),
      mimo_rank_(mimo_rank),
      nonlinearity_mode_(nonlinearity_mode),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void StructuredElmanBackward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* B_proj,
    const __nv_bfloat16* X_proj,
    const __nv_bfloat16* alpha_raw,
    const __nv_bfloat16* alpha_bias,
    const __nv_bfloat16* z,
    const __nv_bfloat16* H,
    const __nv_bfloat16* y_cache,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* dz,
    __nv_bfloat16* dB_proj,
    __nv_bfloat16* dX_proj,
    __nv_bfloat16* dalpha_raw,
    __nv_bfloat16* dalpha_bias,
    __nv_bfloat16* workspace) {

    const int d_inner = nheads_ * headdim_;
    const int BD = batch_size_ * d_inner;
    const int state_size = nheads_ * d_state_ * headdim_;
    const int B_proj_step = batch_size_ * nheads_ * d_state_ * mimo_rank_;
    const int X_proj_step = batch_size_ * nheads_ * headdim_ * mimo_rank_;
    const int alpha_step = batch_size_ * nheads_;
    const int B_state = batch_size_ * state_size;
    const int block_size = 256;

    // Workspace layout:
    // dy: [B, d_inner]
    // dH: [B, state_size]
    // dB_f: [B, nheads, d_state, mimo_rank] float
    // dX_f: [B, nheads, headdim, mimo_rank] float
    // dalpha_f: [B, nheads] float
    // dalpha_bias_f: [nheads] float
    __nv_bfloat16* dy = workspace;
    __nv_bfloat16* dH = dy + BD;
    float* dB_f = reinterpret_cast<float*>(dH + B_state);
    float* dX_f = dB_f + B_proj_step;
    float* dalpha_f = dX_f + X_proj_step;
    float* dalpha_bias_f = dalpha_f + alpha_step;

    // Initialize
    cudaMemsetAsync(dH, 0, B_state * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dalpha_bias_f, 0, nheads_ * sizeof(float), stream_);

    // BPTT loop
    for (int t = steps - 1; t >= 0; --t) {
        const __nv_bfloat16* B_t = B_proj + t * B_proj_step;
        const __nv_bfloat16* X_t = X_proj + t * X_proj_step;
        const __nv_bfloat16* alpha_t = alpha_raw + t * alpha_step;
        const __nv_bfloat16* z_t = z + t * BD;
        const __nv_bfloat16* y_t = y_cache + t * BD;
        const __nv_bfloat16* H_prev = H + t * B_state;
        const __nv_bfloat16* H_t = H + (t + 1) * B_state;
        const __nv_bfloat16* dout_t = d_output + t * BD;
        __nv_bfloat16* dz_t = dz + t * BD;
        __nv_bfloat16* dB_t = dB_proj + t * B_proj_step;
        __nv_bfloat16* dX_t = dX_proj + t * X_proj_step;
        __nv_bfloat16* dalpha_t = dalpha_raw + t * alpha_step;

        // Zero per-timestep accumulators
        cudaMemsetAsync(dB_f, 0, B_proj_step * sizeof(float), stream_);
        cudaMemsetAsync(dX_f, 0, X_proj_step * sizeof(float), stream_);
        cudaMemsetAsync(dalpha_f, 0, alpha_step * sizeof(float), stream_);

        // 1. Backward through gate
        GateBackwardKernel_BF16<<<(BD + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, d_inner, y_t, z_t, dout_t, dy, dz_t);

        // 2. Backward through state update + MIMO
        __nv_bfloat16* dH_prev_out = dH;  // Will be used for next iteration

        // Use optimized kernel for d_state=32, headdim=64 (common case)
        // Optimized kernel: one block per (b,h), headdim threads per block
        const bool use_opt = (d_state_ == 32 && headdim_ == 64);
        const int num_bh = batch_size_ * nheads_;

        if (use_opt) {
            // Optimized path: block per (b,h), 64 threads per block
            if (mimo_rank_ == 4) {
                StructuredElmanBackwardKernel_BF16_Opt<4, 32, 64><<<num_bh, 64, 0, stream_>>>(
                    batch_size_, nheads_, nonlinearity_mode_,
                    H_prev, H_t, B_t, X_t, alpha_t, alpha_bias, dy, dH, dH_prev_out,
                    dB_f, dX_f, dalpha_f);
            } else if (mimo_rank_ == 8) {
                StructuredElmanBackwardKernel_BF16_Opt<8, 32, 64><<<num_bh, 64, 0, stream_>>>(
                    batch_size_, nheads_, nonlinearity_mode_,
                    H_prev, H_t, B_t, X_t, alpha_t, alpha_bias, dy, dH, dH_prev_out,
                    dB_f, dX_f, dalpha_f);
            } else if (mimo_rank_ == 16) {
                StructuredElmanBackwardKernel_BF16_Opt<16, 32, 64><<<num_bh, 64, 0, stream_>>>(
                    batch_size_, nheads_, nonlinearity_mode_,
                    H_prev, H_t, B_t, X_t, alpha_t, alpha_bias, dy, dH, dH_prev_out,
                    dB_f, dX_f, dalpha_f);
            } else {
                // Fallback for unusual mimo_rank
                StructuredElmanBackwardKernel_BF16<8><<<(BD + block_size - 1) / block_size, block_size, 0, stream_>>>(
                    batch_size_, nheads_, d_state_, headdim_, nonlinearity_mode_,
                    H_prev, H_t, B_t, X_t, alpha_t, alpha_bias, dy, dH, dH_prev_out,
                    dB_f, dX_f, dalpha_f);
            }
        } else {
            // Legacy path for non-standard dimensions
            if (mimo_rank_ == 4) {
                StructuredElmanBackwardKernel_BF16<4><<<(BD + block_size - 1) / block_size, block_size, 0, stream_>>>(
                    batch_size_, nheads_, d_state_, headdim_, nonlinearity_mode_,
                    H_prev, H_t, B_t, X_t, alpha_t, alpha_bias, dy, dH, dH_prev_out,
                    dB_f, dX_f, dalpha_f);
            } else if (mimo_rank_ == 8) {
                StructuredElmanBackwardKernel_BF16<8><<<(BD + block_size - 1) / block_size, block_size, 0, stream_>>>(
                    batch_size_, nheads_, d_state_, headdim_, nonlinearity_mode_,
                    H_prev, H_t, B_t, X_t, alpha_t, alpha_bias, dy, dH, dH_prev_out,
                    dB_f, dX_f, dalpha_f);
            } else {
                StructuredElmanBackwardKernel_BF16<8><<<(BD + block_size - 1) / block_size, block_size, 0, stream_>>>(
                    batch_size_, nheads_, d_state_, headdim_, nonlinearity_mode_,
                    H_prev, H_t, B_t, X_t, alpha_t, alpha_bias, dy, dH, dH_prev_out,
                    dB_f, dX_f, dalpha_f);
            }
        }

        // 3. Copy float grads to output
        CopyFloatToT<__nv_bfloat16><<<(B_proj_step + 255) / 256, 256, 0, stream_>>>(
            B_proj_step, dB_f, dB_t);
        CopyFloatToT<__nv_bfloat16><<<(X_proj_step + 255) / 256, 256, 0, stream_>>>(
            X_proj_step, dX_f, dX_t);
        CopyFloatToT<__nv_bfloat16><<<(alpha_step + 255) / 256, 256, 0, stream_>>>(
            alpha_step, dalpha_f, dalpha_t);

        // Accumulate alpha_bias gradient from this timestep
        SumAlphaBiasKernel<<<(nheads_ + 255) / 256, 256, 0, stream_>>>(
            batch_size_, nheads_, dalpha_f, dalpha_bias_f);
    }

    // Copy alpha_bias gradient
    CopyFloatToT<__nv_bfloat16><<<(nheads_ + 255) / 256, 256, 0, stream_>>>(
        nheads_, dalpha_bias_f, dalpha_bias);
}

// =============================================================================
// Generic Template Implementations
// =============================================================================

template<typename T>
StructuredElmanForward<T>::StructuredElmanForward(
    bool training,
    int batch_size,
    int nheads,
    int d_state,
    int headdim,
    int mimo_rank,
    int nonlinearity_mode,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : training_(training),
      batch_size_(batch_size),
      nheads_(nheads),
      d_state_(d_state),
      headdim_(headdim),
      mimo_rank_(mimo_rank),
      nonlinearity_mode_(nonlinearity_mode),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void StructuredElmanForward<T>::Run(
    int steps,
    const T* B_proj,
    const T* X_proj,
    const T* alpha_raw,
    const T* alpha_bias,
    const T* z,
    T* H,
    T* output,
    T* y_cache,
    T* workspace) {
    // Generic implementation placeholder
}

template<typename T>
StructuredElmanBackward<T>::StructuredElmanBackward(
    int batch_size,
    int nheads,
    int d_state,
    int headdim,
    int mimo_rank,
    int nonlinearity_mode,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      nheads_(nheads),
      d_state_(d_state),
      headdim_(headdim),
      mimo_rank_(mimo_rank),
      nonlinearity_mode_(nonlinearity_mode),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void StructuredElmanBackward<T>::Run(
    int steps,
    const T* B_proj,
    const T* X_proj,
    const T* alpha_raw,
    const T* alpha_bias,
    const T* z,
    const T* H,
    const T* y_cache,
    const T* d_output,
    T* dz,
    T* dB_proj,
    T* dX_proj,
    T* dalpha_raw,
    T* dalpha_bias,
    T* workspace) {
    // Generic implementation placeholder
}

// Explicit template instantiations
template struct StructuredElmanForward<__half>;
template struct StructuredElmanForward<float>;
template struct StructuredElmanForward<double>;

template struct StructuredElmanBackward<__half>;
template struct StructuredElmanBackward<float>;
template struct StructuredElmanBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
