// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// Level 2: Slot-Based Elman - Mamba2-style multi-slot memory (OPTIMIZED)
//
// h_t[:, i] = decay[:, i] * h_{t-1}[:, i] + B[:, i] * x_t  (for each slot i)
// output = sum_i(C[i] * h_t[:, i]) * silu(z)
//
// Optimization: Parallelize across slots instead of looping.
// - SlotUpdateKernel: B*D*S threads, each does ONE slot update
// - SlotCombineKernel: B*D threads, each reduces 64 slots

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>

#include "hasty/elman_ladder.h"
#include "inline_ops.h"

namespace {

// =============================================================================
// Forward: Slot update kernel - parallelized across all (batch, dim, slot)
// =============================================================================
template<typename T>
__global__ void SlotUpdateKernel(
    const int batch_size,
    const int dim,
    const int n_slots,
    const T* __restrict__ x,           // [B, dim]
    const T* __restrict__ h_prev,      // [B, dim, n_slots]
    const T* __restrict__ decay,       // [dim, n_slots]
    const T* __restrict__ B,           // [dim, n_slots]
    T* __restrict__ h_out) {           // [B, dim, n_slots]

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim * n_slots;

    if (idx < total) {
        // Decompose idx into (b, d, s)
        const int s = idx % n_slots;
        const int d = (idx / n_slots) % dim;
        const int b = idx / (n_slots * dim);

        const int bd_idx = b * dim + d;
        const int param_idx = d * n_slots + s;

        float x_val = static_cast<float>(x[bd_idx]);
        float h_prev_val = static_cast<float>(h_prev[idx]);
        float decay_val = static_cast<float>(decay[param_idx]);
        float B_val = static_cast<float>(B[param_idx]);

        // h_new = decay * h_prev + B * x
        float h_new = decay_val * h_prev_val + B_val * x_val;
        h_out[idx] = static_cast<T>(h_new);
    }
}

// =============================================================================
// Forward: Combine slots kernel - sum across slots and apply gating
// =============================================================================
template<typename T>
__global__ void SlotCombineKernel(
    const int batch_size,
    const int dim,
    const int n_slots,
    const T* __restrict__ h,           // [B, dim, n_slots]
    const T* __restrict__ z,           // [B, dim]
    const T* __restrict__ C,           // [n_slots]
    T* __restrict__ output) {          // [B, dim]

    const int bd_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_bd = batch_size * dim;

    if (bd_idx < total_bd) {
        float z_val = static_cast<float>(z[bd_idx]);

        // silu(z)
        float sigmoid_z = 1.0f / (1.0f + expf(-z_val));
        float silu_z = z_val * sigmoid_z;

        // Sum across slots: h_combined = sum_s(C[s] * h[b,d,s])
        float h_combined = 0.0f;
        const int h_base = bd_idx * n_slots;

        #pragma unroll 8
        for (int s = 0; s < n_slots; ++s) {
            float h_val = static_cast<float>(h[h_base + s]);
            float C_val = static_cast<float>(C[s]);
            h_combined += C_val * h_val;
        }

        output[bd_idx] = static_cast<T>(h_combined * silu_z);
    }
}

// =============================================================================
// Backward: Slot gradient kernel - parallelized across (batch, dim, slot)
// Computes dh_prev for each slot and accumulates parameter gradients
// =============================================================================
template<typename T>
__global__ void SlotBackwardKernel(
    const int batch_size,
    const int dim,
    const int n_slots,
    const T* __restrict__ x,           // [B, dim]
    const T* __restrict__ h_prev,      // [B, dim, n_slots]
    const T* __restrict__ h_curr,      // [B, dim, n_slots]
    const T* __restrict__ decay,       // [dim, n_slots]
    const T* __restrict__ C,           // [n_slots]
    const T* __restrict__ dh_comb,     // [B, dim] gradient w.r.t h_combined
    const T* __restrict__ dh_next,     // [B, dim, n_slots] gradient from next timestep (or null)
    T* __restrict__ dh_prev,           // [B, dim, n_slots]
    float* __restrict__ d_decay,       // [dim, n_slots] accumulated
    float* __restrict__ dB,            // [dim, n_slots] accumulated
    float* __restrict__ dC) {          // [n_slots] accumulated

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim * n_slots;

    if (idx < total) {
        const int s = idx % n_slots;
        const int d = (idx / n_slots) % dim;
        const int b = idx / (n_slots * dim);

        const int bd_idx = b * dim + d;
        const int param_idx = d * n_slots + s;

        float x_val = static_cast<float>(x[bd_idx]);
        float h_prev_val = static_cast<float>(h_prev[idx]);
        float h_curr_val = static_cast<float>(h_curr[idx]);
        float decay_val = static_cast<float>(decay[param_idx]);
        float C_val = static_cast<float>(C[s]);
        float dh_comb_val = static_cast<float>(dh_comb[bd_idx]);

        // dh_slot = dh_comb * C[s] + dh_next[slot] (if exists)
        float dh_slot = dh_comb_val * C_val;
        if (dh_next != nullptr) {
            dh_slot += static_cast<float>(dh_next[idx]);
        }

        // dh_prev = dh_slot * decay (for BPTT)
        dh_prev[idx] = static_cast<T>(dh_slot * decay_val);

        // Parameter gradients (atomic for thread safety)
        atomicAdd(&d_decay[param_idx], dh_slot * h_prev_val);
        atomicAdd(&dB[param_idx], dh_slot * x_val);
        atomicAdd(&dC[s], dh_comb_val * h_curr_val);
    }
}

// =============================================================================
// Backward: Combine gradients for dx - reduce dh_slot * B across slots
// =============================================================================
template<typename T>
__global__ void SlotGradDxKernel(
    const int batch_size,
    const int dim,
    const int n_slots,
    const T* __restrict__ dh_comb,     // [B, dim]
    const T* __restrict__ dh_next,     // [B, dim, n_slots] or null
    const T* __restrict__ decay,       // [dim, n_slots]
    const T* __restrict__ B,           // [dim, n_slots]
    const T* __restrict__ C,           // [n_slots]
    T* __restrict__ dx) {              // [B, dim]

    const int bd_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_bd = batch_size * dim;

    if (bd_idx < total_bd) {
        const int d = bd_idx % dim;
        const int h_base = bd_idx * n_slots;
        const int param_base = d * n_slots;

        float dh_comb_val = static_cast<float>(dh_comb[bd_idx]);
        float dx_acc = 0.0f;

        #pragma unroll 8
        for (int s = 0; s < n_slots; ++s) {
            float C_val = static_cast<float>(C[s]);
            float B_val = static_cast<float>(B[param_base + s]);

            // dh_slot = dh_comb * C[s] + dh_next[slot]
            float dh_slot = dh_comb_val * C_val;
            if (dh_next != nullptr) {
                dh_slot += static_cast<float>(dh_next[h_base + s]);
            }

            // dx += dh_slot * B[slot]
            dx_acc += dh_slot * B_val;
        }

        dx[bd_idx] = static_cast<T>(dx_acc);
    }
}

// =============================================================================
// Backward: Compute dz and dh_comb from d_output
// output = h_combined * silu(z) => need h_combined to compute dz
// =============================================================================
template<typename T>
__global__ void SlotGradGateKernel(
    const int batch_size,
    const int dim,
    const int n_slots,
    const T* __restrict__ z,           // [B, dim]
    const T* __restrict__ h_curr,      // [B, dim, n_slots]
    const T* __restrict__ C,           // [n_slots]
    const T* __restrict__ d_output,    // [B, dim]
    T* __restrict__ dz,                // [B, dim]
    T* __restrict__ dh_comb) {         // [B, dim] intermediate

    const int bd_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_bd = batch_size * dim;

    if (bd_idx < total_bd) {
        float z_val = static_cast<float>(z[bd_idx]);
        float dout = static_cast<float>(d_output[bd_idx]);

        // Recompute silu(z) and derivative
        float sigmoid_z = 1.0f / (1.0f + expf(-z_val));
        float silu_z = z_val * sigmoid_z;
        float dsilu_z = sigmoid_z * (1.0f + z_val * (1.0f - sigmoid_z));

        // Recompute h_combined
        float h_combined = 0.0f;
        const int h_base = bd_idx * n_slots;

        #pragma unroll 8
        for (int s = 0; s < n_slots; ++s) {
            float h_val = static_cast<float>(h_curr[h_base + s]);
            float C_val = static_cast<float>(C[s]);
            h_combined += C_val * h_val;
        }

        // dz = d_output * h_combined * dsilu_z
        dz[bd_idx] = static_cast<T>(dout * h_combined * dsilu_z);

        // dh_comb = d_output * silu_z (for downstream gradient computation)
        dh_comb[bd_idx] = static_cast<T>(dout * silu_z);
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

}  // anonymous namespace


namespace hasty {
namespace v0 {
namespace elman_ladder {

// =============================================================================
// Slot Elman Forward (OPTIMIZED)
// =============================================================================

template<typename T>
SlotElmanForward<T>::SlotElmanForward(
    bool training,
    int batch_size,
    int dim,
    int n_slots,
    const cudaStream_t& stream)
    : training_(training),
      batch_size_(batch_size),
      dim_(dim),
      n_slots_(n_slots),
      stream_(stream) {}

template<typename T>
void SlotElmanForward<T>::Run(
    int steps,
    const T* x,           // [T, B, dim] pre-activated input
    const T* z,           // [T, B, dim] gate input
    const T* decay,       // [dim, n_slots] sigmoid-normalized
    const T* B,           // [dim, n_slots]
    const T* C,           // [n_slots]
    T* h,                 // [T+1, B, dim, n_slots] hidden states
    T* output) {          // [T, B, dim] output

    const int BD = batch_size_ * dim_;
    const int BDS = batch_size_ * dim_ * n_slots_;
    const int block_size = 256;

    // Grid sizes for different kernels
    const int slot_blocks = (BDS + block_size - 1) / block_size;
    const int combine_blocks = (BD + block_size - 1) / block_size;

    for (int t = 0; t < steps; ++t) {
        const T* x_t = x + t * BD;
        const T* z_t = z + t * BD;
        const T* h_prev = h + t * BDS;
        T* h_out = h + (t + 1) * BDS;
        T* out_t = output + t * BD;

        // Step 1: Update all slots in parallel (BDS threads)
        SlotUpdateKernel<T><<<slot_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, n_slots_,
            x_t, h_prev, decay, B, h_out);

        // Step 2: Combine slots and apply gating (BD threads)
        SlotCombineKernel<T><<<combine_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, n_slots_,
            h_out, z_t, C, out_t);
    }
}

// =============================================================================
// Slot Elman Backward (OPTIMIZED)
// =============================================================================

template<typename T>
SlotElmanBackward<T>::SlotElmanBackward(
    int batch_size,
    int dim,
    int n_slots,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      n_slots_(n_slots),
      stream_(stream) {}

template<typename T>
void SlotElmanBackward<T>::Run(
    int steps,
    const T* x,           // [T, B, dim]
    const T* z,           // [T, B, dim]
    const T* h,           // [T+1, B, dim, n_slots]
    const T* decay,       // [dim, n_slots]
    const T* B,           // [dim, n_slots]
    const T* C,           // [n_slots]
    const T* d_output,    // [T, B, dim]
    T* dx,                // [T, B, dim]
    T* dz,                // [T, B, dim]
    T* d_decay,           // [dim, n_slots]
    T* dB,                // [dim, n_slots]
    T* dC,                // [n_slots]
    T* workspace) {       // Workspace

    const int BD = batch_size_ * dim_;
    const int BDS = batch_size_ * dim_ * n_slots_;
    const int D_S = dim_ * n_slots_;
    const int block_size = 256;

    const int slot_blocks = (BDS + block_size - 1) / block_size;
    const int combine_blocks = (BD + block_size - 1) / block_size;

    // =========================================================================
    // Workspace layout:
    // [dh_prev: BDS] [dh_next: BDS] [dh_comb: BD] [d_decay_float: D_S] [dB_float: D_S] [dC_float: n_slots]
    // =========================================================================
    T* dh_prev = workspace;
    T* dh_next = workspace + BDS;
    T* dh_comb = workspace + 2 * BDS;
    int float_offset = 2 * BDS + BD;
    float* d_decay_float = reinterpret_cast<float*>(workspace + float_offset);
    float* dB_float = d_decay_float + D_S;
    float* dC_float = dB_float + D_S;

    // Initialize workspace
    cudaMemsetAsync(dh_next, 0, BDS * sizeof(T), stream_);
    cudaMemsetAsync(d_decay_float, 0, D_S * sizeof(float), stream_);
    cudaMemsetAsync(dB_float, 0, D_S * sizeof(float), stream_);
    cudaMemsetAsync(dC_float, 0, n_slots_ * sizeof(float), stream_);

    // BPTT loop
    for (int t = steps - 1; t >= 0; --t) {
        const T* x_t = x + t * BD;
        const T* z_t = z + t * BD;
        const T* h_prev_t = h + t * BDS;
        const T* h_curr_t = h + (t + 1) * BDS;
        const T* d_out_t = d_output + t * BD;
        T* dx_t = dx + t * BD;
        T* dz_t = dz + t * BD;

        // Step 1: Compute dz and dh_comb from d_output (BD threads)
        SlotGradGateKernel<T><<<combine_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, n_slots_,
            z_t, h_curr_t, C, d_out_t, dz_t, dh_comb);

        // Step 2: Compute slot gradients in parallel (BDS threads)
        SlotBackwardKernel<T><<<slot_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, n_slots_,
            x_t, h_prev_t, h_curr_t, decay, C,
            dh_comb, (t < steps - 1) ? dh_next : nullptr,
            dh_prev,
            d_decay_float, dB_float, dC_float);

        // Step 3: Compute dx by reducing across slots (BD threads)
        SlotGradDxKernel<T><<<combine_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, n_slots_,
            dh_comb, (t < steps - 1) ? dh_next : nullptr,
            decay, B, C, dx_t);

        // Copy dh_prev to dh_next for next iteration
        if (t > 0) {
            cudaMemcpyAsync(dh_next, dh_prev, BDS * sizeof(T), cudaMemcpyDeviceToDevice, stream_);
        }
    }

    // Copy float gradients to output tensors
    int param_blocks = (D_S + 255) / 256;
    int slot_blocks_small = (n_slots_ + 255) / 256;
    CopyFloatToT<T><<<param_blocks, 256, 0, stream_>>>(D_S, d_decay_float, d_decay);
    CopyFloatToT<T><<<param_blocks, 256, 0, stream_>>>(D_S, dB_float, dB);
    CopyFloatToT<T><<<slot_blocks_small, 256, 0, stream_>>>(n_slots_, dC_float, dC);
}

// Explicit template instantiations
template struct SlotElmanForward<__half>;
template struct SlotElmanForward<__nv_bfloat16>;
template struct SlotElmanForward<float>;
template struct SlotElmanForward<double>;

template struct SlotElmanBackward<__half>;
template struct SlotElmanBackward<__nv_bfloat16>;
template struct SlotElmanBackward<float>;
template struct SlotElmanBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
