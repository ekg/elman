// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// Level 2: Slot-Based Elman - Multi-slot memory with cuBLAS GEMMs
//
// Architecture (same as e0, but with n_slots independent hidden states):
// h_t[s] = tanh(W_x @ x + W_h @ h_prev[s] + b)    for each slot s
// output = sum(C[s] * h_t[s]) * silu(z)
//
// Key optimization: Batch slots into GEMM by treating [B, n_slots, d] as [B*n_slots, d]
// This gives same speed as e0 but with n_slots more memory capacity.

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
// Forward: Fused Wx (broadcast) + Rh + bias + tanh
// Wx is [B, dim], Rh is [B*n_slots, dim] = [B, n_slots, dim]
// We broadcast Wx across the n_slots dimension
// =============================================================================
template<typename T>
__global__ void SlotFusedTanhKernel(
    const int batch_size,
    const int n_slots,
    const int dim,
    const T* __restrict__ Wx,        // [B, dim] pre-computed W_x @ x (for this timestep)
    const T* __restrict__ Rh,        // [B*n_slots, dim] = [B, n_slots, dim]
    const T* __restrict__ b,         // [dim] bias
    T* __restrict__ h_out,           // [B*n_slots, dim] = [B, n_slots, dim]
    T* __restrict__ v_cache) {       // [B*n_slots, dim] pre-activation cache (optional)

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * n_slots * dim;

    if (idx < total) {
        // Decompose idx: [B, n_slots, dim] in row-major
        const int d = idx % dim;
        const int s = (idx / dim) % n_slots;
        const int b_idx = idx / (n_slots * dim);

        // Wx index: [B, dim] - broadcast across slots
        const int wx_idx = b_idx * dim + d;

        float val = static_cast<float>(Wx[wx_idx]) + static_cast<float>(Rh[idx]) + static_cast<float>(b[d]);
        if (v_cache) v_cache[idx] = static_cast<T>(val);
        h_out[idx] = static_cast<T>(tanhf(val));
    }
}

// =============================================================================
// Forward: Combine slots and apply gating
// h is [B, n_slots, dim], output is [B, dim]
// output = sum_s(C[s] * h[:, s, :]) * silu(z)
// =============================================================================
template<typename T>
__global__ void SlotCombineGateKernel(
    const int batch_size,
    const int n_slots,
    const int dim,
    const T* __restrict__ h,         // [B, n_slots, dim]
    const T* __restrict__ z,         // [B, dim] gate input
    const T* __restrict__ C,         // [n_slots] slot weights
    T* __restrict__ output) {        // [B, dim]

    const int bd_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_bd = batch_size * dim;

    if (bd_idx < total_bd) {
        const int d = bd_idx % dim;
        const int b_idx = bd_idx / dim;

        // Sum across slots
        float h_combined = 0.0f;
        for (int s = 0; s < n_slots; ++s) {
            int h_idx = (b_idx * n_slots + s) * dim + d;
            float h_val = static_cast<float>(h[h_idx]);
            float c_val = static_cast<float>(C[s]);
            h_combined += c_val * h_val;
        }

        // silu(z) gating
        float z_val = static_cast<float>(z[bd_idx]);
        float sigmoid_z = 1.0f / (1.0f + expf(-z_val));
        float silu_z = z_val * sigmoid_z;

        output[bd_idx] = static_cast<T>(h_combined * silu_z);
    }
}

// =============================================================================
// Backward: Through tanh for all slots
// =============================================================================
template<typename T>
__global__ void SlotTanhBackwardKernel(
    const int batch_size,
    const int n_slots,
    const int dim,
    const T* __restrict__ v,           // [B, n_slots, dim] pre-activation
    const T* __restrict__ dh,          // [B, n_slots, dim] gradient from gate backward
    const T* __restrict__ dh_recurrent,// [B, n_slots, dim] gradient from next timestep (or null)
    T* __restrict__ dv,                // [B, n_slots, dim]
    float* __restrict__ db) {          // [dim] bias gradient (atomic)

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * n_slots * dim;

    if (idx < total) {
        const int d = idx % dim;

        float grad = static_cast<float>(dh[idx]);
        if (dh_recurrent) grad += static_cast<float>(dh_recurrent[idx]);

        float h = tanhf(static_cast<float>(v[idx]));
        float dtanh = 1.0f - h * h;
        float dv_val = grad * dtanh;
        dv[idx] = static_cast<T>(dv_val);

        atomicAdd(&db[d], dv_val);
    }
}

// =============================================================================
// Backward: Through slot combine + gate
// output = sum_s(C[s] * h[:, s, :]) * silu(z)
// Computes dh and dz (dC computed separately to avoid atomic contention)
// =============================================================================
template<typename T>
__global__ void SlotCombineGateBackwardKernel(
    const int batch_size,
    const int n_slots,
    const int dim,
    const T* __restrict__ h,           // [B, n_slots, dim]
    const T* __restrict__ z,           // [B, dim]
    const T* __restrict__ C,           // [n_slots]
    const T* __restrict__ d_output,    // [B, dim]
    T* __restrict__ dh,                // [B, n_slots, dim]
    T* __restrict__ dz) {              // [B, dim]

    const int bd_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_bd = batch_size * dim;

    if (bd_idx < total_bd) {
        const int d = bd_idx % dim;
        const int b_idx = bd_idx / dim;

        float dout = static_cast<float>(d_output[bd_idx]);
        float z_val = static_cast<float>(z[bd_idx]);

        // Recompute silu(z) and derivatives
        float sigmoid_z = 1.0f / (1.0f + expf(-z_val));
        float silu_z = z_val * sigmoid_z;
        float dsilu_z = sigmoid_z * (1.0f + z_val * (1.0f - sigmoid_z));

        // Recompute h_combined for dz
        float h_combined = 0.0f;
        for (int s = 0; s < n_slots; ++s) {
            int h_idx = (b_idx * n_slots + s) * dim + d;
            h_combined += static_cast<float>(C[s]) * static_cast<float>(h[h_idx]);
        }

        // dz = d_output * h_combined * dsilu_z
        dz[bd_idx] = static_cast<T>(dout * h_combined * dsilu_z);

        // dh[s] = d_output * silu_z * C[s] for each slot
        float d_hcomb = dout * silu_z;
        for (int s = 0; s < n_slots; ++s) {
            int h_idx = (b_idx * n_slots + s) * dim + d;
            dh[h_idx] = static_cast<T>(d_hcomb * static_cast<float>(C[s]));
        }
    }
}

// =============================================================================
// Backward: Compute dC with parallel reduction (no atomic contention)
// dC[s] = sum_{b,d} d_output[b,d] * silu(z[b,d]) * h[b,s,d]
// =============================================================================
template<typename T>
__global__ void SlotComputeDCKernel(
    const int batch_size,
    const int n_slots,
    const int dim,
    const T* __restrict__ h,           // [B, n_slots, dim]
    const T* __restrict__ z,           // [B, dim]
    const T* __restrict__ d_output,    // [B, dim]
    float* __restrict__ dC) {          // [n_slots]

    // One block per slot, threads reduce over B*D
    const int slot = blockIdx.x;
    const int tid = threadIdx.x;
    const int total_bd = batch_size * dim;

    // Shared memory for block reduction
    extern __shared__ float sdata[];

    float local_sum = 0.0f;

    // Each thread sums its portion
    for (int bd_idx = tid; bd_idx < total_bd; bd_idx += blockDim.x) {
        const int d = bd_idx % dim;
        const int b_idx = bd_idx / dim;

        float dout = static_cast<float>(d_output[bd_idx]);
        float z_val = static_cast<float>(z[bd_idx]);
        float sigmoid_z = 1.0f / (1.0f + expf(-z_val));
        float silu_z = z_val * sigmoid_z;

        int h_idx = (b_idx * n_slots + slot) * dim + d;
        float h_val = static_cast<float>(h[h_idx]);

        local_sum += dout * silu_z * h_val;
    }

    sdata[tid] = local_sum;
    __syncthreads();

    // Block reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Write result (one atomic per slot, not per thread)
    if (tid == 0) {
        atomicAdd(&dC[slot], sdata[0]);
    }
}

// =============================================================================
// Backward: Reduce dv across slots to get dx contribution
// dv is [B, n_slots, dim], we need dx which is [B, dim]
// dx = sum_s(W_x @ dv[s]) = W_x @ sum_s(dv[s])
// So we first sum dv across slots
// =============================================================================
template<typename T>
__global__ void SlotReduceDvKernel(
    const int batch_size,
    const int n_slots,
    const int dim,
    const T* __restrict__ dv,          // [B, n_slots, dim]
    T* __restrict__ dv_sum) {          // [B, dim]

    const int bd_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_bd = batch_size * dim;

    if (bd_idx < total_bd) {
        const int d = bd_idx % dim;
        const int b_idx = bd_idx / dim;

        float sum = 0.0f;
        for (int s = 0; s < n_slots; ++s) {
            int dv_idx = (b_idx * n_slots + s) * dim + d;
            sum += static_cast<float>(dv[dv_idx]);
        }
        dv_sum[bd_idx] = static_cast<T>(sum);
    }
}

// Vector add inplace
template<typename T>
__global__ void VectorAddInplace(const int n, T* __restrict__ a, const T* __restrict__ b) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = static_cast<T>(static_cast<float>(a[idx]) + static_cast<float>(b[idx]));
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
// Slot Elman Forward (with cuBLAS GEMMs)
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
    const T* W_x,         // [dim, dim]
    const T* W_h,         // [dim, dim]
    const T* b,           // [dim]
    const T* C,           // [n_slots]
    const T* x,           // [T, B, dim]
    const T* z,           // [T, B, dim]
    T* h,                 // [T+1, B, n_slots, dim]
    T* output,            // [T, B, dim]
    T* v,                 // [T, B, n_slots, dim] pre-activation cache
    T* workspace,         // [T*B*dim + B*n_slots*dim]
    cublasHandle_t blas_handle) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int BD = batch_size_ * dim_;
    const int BSD = batch_size_ * n_slots_ * dim_;
    const int block_size = 256;
    const int bd_blocks = (BD + block_size - 1) / block_size;
    const int bsd_blocks = (BSD + block_size - 1) / block_size;

    // Workspace layout: [tmp_Wx: T*B*dim] [tmp_Rh: B*n_slots*dim]
    T* tmp_Wx = workspace;
    T* tmp_Rh = workspace + steps * BD;

    // Pre-compute W_x @ x for ALL timesteps (HASTE pattern)
    blas<T>::gemm(
        blas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha,
        W_x, dim_,
        x, dim_,
        &beta_zero,
        tmp_Wx, dim_);

    // Process each timestep
    for (int t = 0; t < steps; ++t) {
        const T* Wx_t = tmp_Wx + t * BD;      // [B, dim]
        const T* z_t = z + t * BD;            // [B, dim]
        const T* h_prev = h + t * BSD;        // [B, n_slots, dim] = [B*n_slots, dim]
        T* h_t = h + (t + 1) * BSD;           // [B, n_slots, dim]
        T* out_t = output + t * BD;           // [B, dim]
        T* v_t = training_ ? (v + t * BSD) : nullptr;

        // tmp_Rh = h_prev @ W_h.T for all slots at once
        // h_prev is [B*n_slots, dim], W_h is [dim, dim]
        blas<T>::gemm(
            blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_ * n_slots_, dim_,
            &alpha,
            W_h, dim_,
            h_prev, dim_,
            &beta_zero,
            tmp_Rh, dim_);

        // h_t = tanh(Wx_t (broadcast) + tmp_Rh + b)
        SlotFusedTanhKernel<T><<<bsd_blocks, block_size, 0, stream_>>>(
            batch_size_, n_slots_, dim_, Wx_t, tmp_Rh, b, h_t, v_t);

        // output = combine_slots(h_t) * silu(z_t)
        SlotCombineGateKernel<T><<<bd_blocks, block_size, 0, stream_>>>(
            batch_size_, n_slots_, dim_, h_t, z_t, C, out_t);
    }
}

// =============================================================================
// Slot Elman Backward (with cuBLAS GEMMs)
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
    const T* W_x,         // [dim, dim]
    const T* W_h,         // [dim, dim]
    const T* C,           // [n_slots]
    const T* x,           // [T, B, dim]
    const T* z,           // [T, B, dim]
    const T* h,           // [T+1, B, n_slots, dim]
    const T* v,           // [T, B, n_slots, dim]
    const T* d_output,    // [T, B, dim]
    T* dx,                // [T, B, dim]
    T* dz,                // [T, B, dim]
    T* dW_x,              // [dim, dim]
    T* dW_h,              // [dim, dim]
    T* db,                // [dim]
    T* dC,                // [n_slots]
    T* workspace,
    cublasHandle_t blas_handle) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int BD = batch_size_ * dim_;
    const int BSD = batch_size_ * n_slots_ * dim_;
    const int block_size = 256;
    const int bd_blocks = (BD + block_size - 1) / block_size;
    const int bsd_blocks = (BSD + block_size - 1) / block_size;

    // Workspace layout:
    // [dv_all: T*BSD] [dh: BSD] [dh_recurrent: BSD] [dv_sum: T*BD]
    // [db_float: dim] [dC_float: n_slots]
    T* dv_all = workspace;
    T* dh = workspace + steps * BSD;
    T* dh_recurrent = workspace + (steps + 1) * BSD;
    T* dv_sum_all = workspace + (steps + 2) * BSD;
    int float_offset = (steps + 2) * BSD + steps * BD;
    float* db_float = reinterpret_cast<float*>(workspace + float_offset);
    float* dC_float = db_float + dim_;

    // Initialize
    cudaMemsetAsync(dh_recurrent, 0, BSD * sizeof(T), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dC_float, 0, n_slots_ * sizeof(float), stream_);
    cudaMemsetAsync(dW_x, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dW_h, 0, dim_ * dim_ * sizeof(T), stream_);

    // BPTT loop
    const int dc_block_size = 256;
    const size_t dc_shared_mem = dc_block_size * sizeof(float);

    for (int t = steps - 1; t >= 0; --t) {
        const T* h_t = h + (t + 1) * BSD;
        const T* z_t = z + t * BD;
        const T* v_t = v + t * BSD;
        const T* d_out_t = d_output + t * BD;
        T* dz_t = dz + t * BD;
        T* dv_t = dv_all + t * BSD;
        T* dv_sum_t = dv_sum_all + t * BD;

        // Backward through combine + gate (dh and dz only, dC computed separately)
        SlotCombineGateBackwardKernel<T><<<bd_blocks, block_size, 0, stream_>>>(
            batch_size_, n_slots_, dim_, h_t, z_t, C, d_out_t,
            dh, dz_t);

        // Compute dC with proper parallel reduction (one block per slot)
        SlotComputeDCKernel<T><<<n_slots_, dc_block_size, dc_shared_mem, stream_>>>(
            batch_size_, n_slots_, dim_, h_t, z_t, d_out_t, dC_float);

        // Add recurrent gradient
        VectorAddInplace<T><<<bsd_blocks, block_size, 0, stream_>>>(BSD, dh, dh_recurrent);

        // Backward through tanh
        SlotTanhBackwardKernel<T><<<bsd_blocks, block_size, 0, stream_>>>(
            batch_size_, n_slots_, dim_, v_t, dh, nullptr, dv_t, db_float);

        // Reduce dv across slots for dx computation later
        SlotReduceDvKernel<T><<<bd_blocks, block_size, 0, stream_>>>(
            batch_size_, n_slots_, dim_, dv_t, dv_sum_t);

        // dh_recurrent = W_h @ dv for each slot
        if (t > 0) {
            blas<T>::gemm(
                blas_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                dim_, batch_size_ * n_slots_, dim_,
                &alpha,
                W_h, dim_,
                dv_t, dim_,
                &beta_zero,
                dh_recurrent, dim_);
        }
    }

    // Batch GEMMs
    // dx = W_x @ sum_s(dv) for each timestep
    blas<T>::gemm(
        blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha,
        W_x, dim_,
        dv_sum_all, dim_,
        &beta_zero,
        dx, dim_);

    // dW_x = x^T @ dv_sum_all
    blas<T>::gemm(
        blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha,
        x, dim_,
        dv_sum_all, dim_,
        &beta_one,
        dW_x, dim_);

    // dW_h = h^T @ dv_all (all slots)
    blas<T>::gemm(
        blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_ * n_slots_,
        &alpha,
        h, dim_,
        dv_all, dim_,
        &beta_one,
        dW_h, dim_);

    // Copy float gradients
    CopyFloatToT<T><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_float, db);
    CopyFloatToT<T><<<(n_slots_ + 255) / 256, 256, 0, stream_>>>(n_slots_, dC_float, dC);
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
