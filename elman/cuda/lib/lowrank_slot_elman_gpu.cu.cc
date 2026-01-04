// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// Level 3: Low-Rank Slot Elman - Independent low-rank W_h per slot
//
// Architecture:
// h_t[s] = tanh(W_x @ x + U_s @ (V_s @ h_prev[s]) + b)    for each slot s
// output = sum(C[s] * h_t[s]) * silu(z)
//
// OPTIMIZED: Custom fused kernels that avoid cuBLAS overhead for small matrices.
// Each (slot, batch) pair is processed by one thread block using shared memory.

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
// Forward: Fused low-rank recurrence kernel
// Computes: Vh = V @ h_prev, Uh = U @ Vh, h_new = tanh(Wx + Uh + b)
// One block per (slot, batch) pair for maximum parallelism
// Stores Vh in v_cache for efficient backward GEMM (not pre-activation)
// =============================================================================
template<typename T>
__global__ void FusedLowRankRecurrenceKernel(
    const int batch_size,
    const int n_slots,
    const int dim,
    const int rank,
    const T* __restrict__ Wx,        // [B, dim] pre-computed W_x @ x
    const T* __restrict__ h_prev,    // [n_slots, B, dim]
    const T* __restrict__ V,         // [n_slots, rank, dim]
    const T* __restrict__ U,         // [n_slots, dim, rank]
    const T* __restrict__ b,         // [dim]
    T* __restrict__ h_new,           // [n_slots, B, dim]
    T* __restrict__ Vh_cache) {      // [n_slots, B, rank] - store Vh for backward

    // Each block handles one (slot, batch) pair
    const int slot = blockIdx.x / batch_size;
    const int batch = blockIdx.x % batch_size;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // Shared memory layout: [h_shared: dim] [Vh_shared: rank]
    extern __shared__ float smem[];
    float* h_shared = smem;
    float* Vh_shared = smem + dim;

    // Pointers for this (slot, batch)
    const int h_idx = (slot * batch_size + batch) * dim;
    const int Vh_idx = (slot * batch_size + batch) * rank;
    const T* V_slot = V + slot * rank * dim;
    const T* U_slot = U + slot * dim * rank;
    const int wx_idx = batch * dim;

    // Step 1: Load h_prev into shared memory
    for (int d = tid; d < dim; d += block_size) {
        h_shared[d] = static_cast<float>(h_prev[h_idx + d]);
    }
    __syncthreads();

    // Step 2: Compute Vh = V @ h_prev (each thread handles some rank elements)
    for (int r = tid; r < rank; r += block_size) {
        float sum = 0.0f;
        const T* V_row = V_slot + r * dim;
        for (int d = 0; d < dim; d++) {
            sum += static_cast<float>(V_row[d]) * h_shared[d];
        }
        Vh_shared[r] = sum;
        // Store Vh for backward pass GEMM
        if (Vh_cache) Vh_cache[Vh_idx + r] = static_cast<T>(sum);
    }
    __syncthreads();

    // Step 3: Compute Uh = U @ Vh, then h_new = tanh(Wx + Uh + b)
    for (int d = tid; d < dim; d += block_size) {
        float sum = 0.0f;
        const T* U_row = U_slot + d * rank;
        for (int r = 0; r < rank; r++) {
            sum += static_cast<float>(U_row[r]) * Vh_shared[r];
        }
        float val = static_cast<float>(Wx[wx_idx + d]) + sum + static_cast<float>(b[d]);
        h_new[h_idx + d] = static_cast<T>(tanhf(val));
    }
}

// =============================================================================
// Forward: Combine slots and apply gating
// =============================================================================
template<typename T>
__global__ void LowRankSlotCombineGateKernel(
    const int batch_size,
    const int n_slots,
    const int dim,
    const T* __restrict__ h,         // [n_slots, B, dim]
    const T* __restrict__ z,         // [B, dim]
    const T* __restrict__ C,         // [n_slots]
    T* __restrict__ output) {        // [B, dim]

    const int bd_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_bd = batch_size * dim;
    const int BD = batch_size * dim;

    if (bd_idx < total_bd) {
        float h_combined = 0.0f;
        for (int s = 0; s < n_slots; ++s) {
            int h_idx = s * BD + bd_idx;
            h_combined += static_cast<float>(C[s]) * static_cast<float>(h[h_idx]);
        }

        float z_val = static_cast<float>(z[bd_idx]);
        float sigmoid_z = 1.0f / (1.0f + expf(-z_val));
        float silu_z = z_val * sigmoid_z;

        output[bd_idx] = static_cast<T>(h_combined * silu_z);
    }
}

// =============================================================================
// Backward: Fused gradient through low-rank recurrence
// Computes: dv from dh (using h for dtanh), dVh, dh_recurrent for BPTT
// Also outputs dVh for efficient GEMM-based dV computation after time loop
// =============================================================================
template<typename T>
__global__ void FusedLowRankBackwardKernel(
    const int batch_size,
    const int n_slots,
    const int dim,
    const int rank,
    const T* __restrict__ h,         // [n_slots, B, dim] hidden state (= tanh(v))
    const T* __restrict__ dh,        // [n_slots, B, dim] gradient from gate
    const T* __restrict__ dh_rec_in, // [n_slots, B, dim] gradient from next timestep (or null)
    const T* __restrict__ V,         // [n_slots, rank, dim]
    const T* __restrict__ U,         // [n_slots, dim, rank]
    T* __restrict__ dv,              // [n_slots, B, dim] gradient through tanh
    T* __restrict__ dVh_out,         // [n_slots, B, rank] for GEMM-based dV
    T* __restrict__ dh_recurrent,    // [n_slots, B, dim] gradient to prev timestep
    float* __restrict__ db) {        // [dim] bias gradient (atomic)

    const int slot = blockIdx.x / batch_size;
    const int batch = blockIdx.x % batch_size;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    extern __shared__ float smem[];
    float* dv_shared = smem;              // [dim]
    float* dVh_shared = smem + dim;       // [rank]

    const int h_idx = (slot * batch_size + batch) * dim;
    const int Vh_idx = (slot * batch_size + batch) * rank;
    const T* V_slot = V + slot * rank * dim;
    const T* U_slot = U + slot * dim * rank;

    // Step 1: Compute dv = (dh + dh_rec_in) * (1 - h²) where h = tanh(v)
    for (int d = tid; d < dim; d += block_size) {
        float grad = static_cast<float>(dh[h_idx + d]);
        if (dh_rec_in) grad += static_cast<float>(dh_rec_in[h_idx + d]);

        float h_val = static_cast<float>(h[h_idx + d]);  // h is already tanh(v)
        float dtanh = 1.0f - h_val * h_val;
        float dv_val = grad * dtanh;
        dv_shared[d] = dv_val;
        dv[h_idx + d] = static_cast<T>(dv_val);

        atomicAdd(&db[d], dv_val);
    }
    __syncthreads();

    // Step 2: dVh = U.T @ dv (each thread handles some rank elements)
    // Also store dVh for GEMM-based dV computation
    for (int r = tid; r < rank; r += block_size) {
        float sum = 0.0f;
        for (int d = 0; d < dim; d++) {
            sum += static_cast<float>(U_slot[d * rank + r]) * dv_shared[d];
        }
        dVh_shared[r] = sum;
        if (dVh_out) dVh_out[Vh_idx + r] = static_cast<T>(sum);
    }
    __syncthreads();

    // Step 3: dh_recurrent = V.T @ dVh
    for (int d = tid; d < dim; d += block_size) {
        float sum = 0.0f;
        for (int r = 0; r < rank; r++) {
            sum += static_cast<float>(V_slot[r * dim + d]) * dVh_shared[r];
        }
        dh_recurrent[h_idx + d] = static_cast<T>(sum);
    }
}

// =============================================================================
// Backward: Accumulate dU and dV gradients (NO ATOMICS VERSION)
// Each thread handles one (slot, d, r) combination, summing over all T*B
// Single kernel call after time loop - no synchronization issues
// =============================================================================
template<typename T>
__global__ void AccumulateDUDVAllTimesteps(
    const int steps,
    const int batch_size,
    const int n_slots,
    const int dim,
    const int rank,
    const T* __restrict__ dv_all,    // [T, n_slots, B, dim]
    const T* __restrict__ Vh_all,    // [T, n_slots, B, rank]
    const T* __restrict__ dVh_all,   // [T, n_slots, B, rank]
    const T* __restrict__ h,         // [T+1, n_slots, B, dim] (use h[0:T])
    float* __restrict__ dU,          // [n_slots, dim, rank]
    float* __restrict__ dV) {        // [n_slots, rank, dim]

    // Grid: (n_slots, (dim*rank + 255)/256, 1)
    const int slot = blockIdx.x;
    const int dr_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (dr_idx >= dim * rank) return;

    const int d = dr_idx / rank;
    const int r = dr_idx % rank;

    const int SBD = n_slots * batch_size * dim;
    const int SBR = n_slots * batch_size * rank;
    const int BD = batch_size * dim;
    const int BR = batch_size * rank;

    float dU_acc = 0.0f;
    float dV_acc = 0.0f;

    // Sum over all timesteps and batches
    for (int t = 0; t < steps; t++) {
        for (int b = 0; b < batch_size; b++) {
            // dv_all[t, slot, b, d]
            const int dv_idx = t * SBD + slot * BD + b * dim + d;
            // Vh_all[t, slot, b, r]
            const int Vh_idx = t * SBR + slot * BR + b * rank + r;
            // dVh_all[t, slot, b, r]
            const int dVh_idx = t * SBR + slot * BR + b * rank + r;
            // h_prev[t, slot, b, d] = h[t, slot, b, d]
            const int h_idx = t * SBD + slot * BD + b * dim + d;

            dU_acc += static_cast<float>(dv_all[dv_idx]) * static_cast<float>(Vh_all[Vh_idx]);
            dV_acc += static_cast<float>(dVh_all[dVh_idx]) * static_cast<float>(h[h_idx]);
        }
    }

    // Write result (no atomics needed - each thread writes to unique location)
    dU[slot * dim * rank + d * rank + r] = dU_acc;
    dV[slot * rank * dim + r * dim + d] = dV_acc;
}

// =============================================================================
// Backward: Through combine + gate
// =============================================================================
template<typename T>
__global__ void LowRankSlotCombineGateBackwardKernel(
    const int batch_size,
    const int n_slots,
    const int dim,
    const T* __restrict__ h,
    const T* __restrict__ z,
    const T* __restrict__ C,
    const T* __restrict__ d_output,
    T* __restrict__ dh,
    T* __restrict__ dz) {

    const int bd_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_bd = batch_size * dim;
    const int BD = batch_size * dim;

    if (bd_idx < total_bd) {
        float dout = static_cast<float>(d_output[bd_idx]);
        float z_val = static_cast<float>(z[bd_idx]);

        float sigmoid_z = 1.0f / (1.0f + expf(-z_val));
        float silu_z = z_val * sigmoid_z;
        float dsilu_z = sigmoid_z * (1.0f + z_val * (1.0f - sigmoid_z));

        float h_combined = 0.0f;
        for (int s = 0; s < n_slots; ++s) {
            int h_idx = s * BD + bd_idx;
            h_combined += static_cast<float>(C[s]) * static_cast<float>(h[h_idx]);
        }

        dz[bd_idx] = static_cast<T>(dout * h_combined * dsilu_z);

        float d_hcomb = dout * silu_z;
        for (int s = 0; s < n_slots; ++s) {
            int h_idx = s * BD + bd_idx;
            dh[h_idx] = static_cast<T>(d_hcomb * static_cast<float>(C[s]));
        }
    }
}

// =============================================================================
// Backward: Compute dC with parallel reduction
// =============================================================================
template<typename T>
__global__ void LowRankSlotComputeDCKernel(
    const int batch_size,
    const int n_slots,
    const int dim,
    const T* __restrict__ h,
    const T* __restrict__ z,
    const T* __restrict__ d_output,
    float* __restrict__ dC) {

    const int slot = blockIdx.x;
    const int tid = threadIdx.x;
    const int total_bd = batch_size * dim;
    const int BD = batch_size * dim;

    extern __shared__ float sdata[];

    float local_sum = 0.0f;
    for (int bd_idx = tid; bd_idx < total_bd; bd_idx += blockDim.x) {
        float dout = static_cast<float>(d_output[bd_idx]);
        float z_val = static_cast<float>(z[bd_idx]);
        float sigmoid_z = 1.0f / (1.0f + expf(-z_val));
        float silu_z = z_val * sigmoid_z;

        int h_idx = slot * BD + bd_idx;
        float h_val = static_cast<float>(h[h_idx]);

        local_sum += dout * silu_z * h_val;
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&dC[slot], sdata[0]);
    }
}

// =============================================================================
// Backward: Reduce dv across slots for dx
// =============================================================================
template<typename T>
__global__ void LowRankSlotReduceDvKernel(
    const int batch_size,
    const int n_slots,
    const int dim,
    const T* __restrict__ dv,
    T* __restrict__ dv_sum) {

    const int bd_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_bd = batch_size * dim;
    const int BD = batch_size * dim;

    if (bd_idx < total_bd) {
        float sum = 0.0f;
        for (int s = 0; s < n_slots; ++s) {
            int dv_idx = s * BD + bd_idx;
            sum += static_cast<float>(dv[dv_idx]);
        }
        dv_sum[bd_idx] = static_cast<T>(sum);
    }
}

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
// Low-Rank Slot Elman Forward (FUSED - no cuBLAS for small matrices)
// =============================================================================

template<typename T>
LowRankSlotElmanForward<T>::LowRankSlotElmanForward(
    bool training,
    int batch_size,
    int dim,
    int n_slots,
    int rank,
    const cudaStream_t& stream)
    : training_(training),
      batch_size_(batch_size),
      dim_(dim),
      n_slots_(n_slots),
      rank_(rank),
      stream_(stream) {}

template<typename T>
void LowRankSlotElmanForward<T>::Run(
    int steps,
    const T* W_x,         // [dim, dim]
    const T* U,           // [n_slots, dim, rank]
    const T* V,           // [n_slots, rank, dim]
    const T* b,           // [dim]
    const T* C,           // [n_slots]
    const T* x,           // [T, B, dim]
    const T* z,           // [T, B, dim]
    T* h,                 // [T+1, n_slots, B, dim]
    T* output,            // [T, B, dim]
    T* Vh_cache,          // [T, n_slots, B, rank] - store Vh for backward GEMM
    T* workspace,         // [T*B*dim]
    cublasHandle_t blas_handle) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int BD = batch_size_ * dim_;
    const int SBD = n_slots_ * batch_size_ * dim_;
    const int SBR = n_slots_ * batch_size_ * rank_;

    // Workspace for pre-computed Wx
    T* tmp_Wx = workspace;

    // Pre-compute W_x @ x for ALL timesteps using cuBLAS (this is efficient)
    blas<T>::gemm(
        blas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha,
        W_x, dim_,
        x, dim_,
        &beta_zero,
        tmp_Wx, dim_);

    // Kernel configurations
    const int fused_blocks = n_slots_ * batch_size_;
    const int fused_threads = 256;
    const size_t fused_smem = (dim_ + rank_) * sizeof(float);

    const int gate_threads = 256;
    const int gate_blocks = (BD + gate_threads - 1) / gate_threads;

    // Process each timestep with fused kernels
    for (int t = 0; t < steps; ++t) {
        const T* Wx_t = tmp_Wx + t * BD;
        const T* z_t = z + t * BD;
        const T* h_prev = h + t * SBD;
        T* h_t = h + (t + 1) * SBD;
        T* out_t = output + t * BD;
        T* Vh_t = training_ ? (Vh_cache + t * SBR) : nullptr;

        // Fused low-rank recurrence: V @ h, U @ Vh, tanh
        // Also stores Vh for backward pass
        FusedLowRankRecurrenceKernel<T><<<fused_blocks, fused_threads, fused_smem, stream_>>>(
            batch_size_, n_slots_, dim_, rank_,
            Wx_t, h_prev, V, U, b, h_t, Vh_t);

        // Combine slots and apply gating
        LowRankSlotCombineGateKernel<T><<<gate_blocks, gate_threads, 0, stream_>>>(
            batch_size_, n_slots_, dim_, h_t, z_t, C, out_t);
    }
}

// =============================================================================
// Low-Rank Slot Elman Backward (FUSED)
// =============================================================================

template<typename T>
LowRankSlotElmanBackward<T>::LowRankSlotElmanBackward(
    int batch_size,
    int dim,
    int n_slots,
    int rank,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      n_slots_(n_slots),
      rank_(rank),
      stream_(stream) {}

template<typename T>
void LowRankSlotElmanBackward<T>::Run(
    int steps,
    const T* W_x,
    const T* U,
    const T* V,
    const T* C,
    const T* x,
    const T* z,
    const T* h,
    const T* Vh_all,      // [T, n_slots, B, rank] - cached from forward
    const T* d_output,
    T* dx,
    T* dz,
    T* dW_x,
    T* dU,
    T* dV,
    T* db,
    T* dC,
    T* workspace,
    cublasHandle_t blas_handle) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int BD = batch_size_ * dim_;
    const int SBD = n_slots_ * batch_size_ * dim_;
    const int SBR = n_slots_ * batch_size_ * rank_;

    // Workspace layout:
    // [dv_all: T*SBD] [dVh_all: T*SBR] [dh: SBD] [dh_recurrent: SBD] [dv_sum: T*BD]
    // [dU_float: n_slots*dim*rank] [dV_float: n_slots*rank*dim]
    // [db_float: dim] [dC_float: n_slots]
    T* dv_all = workspace;
    T* dVh_all = dv_all + steps * SBD;
    T* dh = dVh_all + steps * SBR;
    T* dh_recurrent = dh + SBD;
    T* dv_sum_all = dh_recurrent + SBD;

    const int64_t dUV_size = n_slots_ * dim_ * rank_;
    int64_t float_offset = steps * SBD + steps * SBR + 2 * SBD + steps * BD;
    float* dU_float = reinterpret_cast<float*>(workspace + float_offset);
    float* dV_float = dU_float + dUV_size;
    float* db_float = dV_float + dUV_size;
    float* dC_float = db_float + dim_;

    // Initialize gradients
    cudaMemsetAsync(dh_recurrent, 0, SBD * sizeof(T), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dC_float, 0, n_slots_ * sizeof(float), stream_);
    cudaMemsetAsync(dW_x, 0, dim_ * dim_ * sizeof(T), stream_);

    // Kernel configurations
    const int fused_blocks = n_slots_ * batch_size_;
    const int fused_threads = 256;
    const size_t fused_smem = (dim_ + rank_) * sizeof(float);  // dv, dVh

    const int gate_threads = 256;
    const int gate_blocks = (BD + gate_threads - 1) / gate_threads;

    const int dc_threads = 256;
    const size_t dc_smem = dc_threads * sizeof(float);

    // BPTT loop - compute dv, dVh, dh_recurrent for each timestep
    for (int t = steps - 1; t >= 0; --t) {
        const T* h_t = h + (t + 1) * SBD;  // h at timestep t+1 (output of forward at t)
        const T* z_t = z + t * BD;
        const T* d_out_t = d_output + t * BD;
        T* dz_t = dz + t * BD;
        T* dv_t = dv_all + t * SBD;
        T* dVh_t = dVh_all + t * SBR;
        T* dv_sum_t = dv_sum_all + t * BD;

        // Backward through combine + gate
        LowRankSlotCombineGateBackwardKernel<T><<<gate_blocks, gate_threads, 0, stream_>>>(
            batch_size_, n_slots_, dim_, h_t, z_t, C, d_out_t, dh, dz_t);

        // Compute dC
        LowRankSlotComputeDCKernel<T><<<n_slots_, dc_threads, dc_smem, stream_>>>(
            batch_size_, n_slots_, dim_, h_t, z_t, d_out_t, dC_float);

        // Fused backward through low-rank recurrence
        // Uses h_t (= tanh(v)) for dtanh computation, outputs dv and dVh
        FusedLowRankBackwardKernel<T><<<fused_blocks, fused_threads, fused_smem, stream_>>>(
            batch_size_, n_slots_, dim_, rank_,
            h_t, dh, (t < steps - 1) ? dh_recurrent : nullptr,
            V, U, dv_t, dVh_t, dh_recurrent, db_float);

        // Reduce dv across slots for dx/dW_x
        LowRankSlotReduceDvKernel<T><<<gate_blocks, gate_threads, 0, stream_>>>(
            batch_size_, n_slots_, dim_, dv_t, dv_sum_t);
    }

    // =========================================================================
    // Compute dU and dV using parallel kernel (NO ATOMICS)
    // dU[s,d,r] = sum_t,b dv[t,s,b,d] * Vh[t,s,b,r]
    // dV[s,r,d] = sum_t,b dVh[t,s,b,r] * h[t,s,b,d]
    // =========================================================================
    const int dr_total = dim_ * rank_;
    const int dudv_blocks_y = (dr_total + 255) / 256;
    dim3 dudv_grid(n_slots_, dudv_blocks_y, 1);
    dim3 dudv_block(256, 1, 1);

    AccumulateDUDVAllTimesteps<T><<<dudv_grid, dudv_block, 0, stream_>>>(
        steps, batch_size_, n_slots_, dim_, rank_,
        dv_all, Vh_all, dVh_all, h, dU_float, dV_float);

    // dx = W_x @ sum_s(dv)
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

    // Copy float gradients to output type
    const int copy_threads = 256;
    CopyFloatToT<T><<<(dim_ + copy_threads - 1) / copy_threads, copy_threads, 0, stream_>>>(
        dim_, db_float, db);
    CopyFloatToT<T><<<(n_slots_ + copy_threads - 1) / copy_threads, copy_threads, 0, stream_>>>(
        n_slots_, dC_float, dC);

    // Copy dU and dV from float to T
    const int dUV_total = n_slots_ * dim_ * rank_;
    CopyFloatToT<T><<<(dUV_total + copy_threads - 1) / copy_threads, copy_threads, 0, stream_>>>(
        dUV_total, dU_float, dU);
    CopyFloatToT<T><<<(dUV_total + copy_threads - 1) / copy_threads, copy_threads, 0, stream_>>>(
        dUV_total, dV_float, dV);
}

// Explicit instantiations
template struct LowRankSlotElmanForward<__half>;
template struct LowRankSlotElmanForward<__nv_bfloat16>;
template struct LowRankSlotElmanForward<float>;
template struct LowRankSlotElmanForward<double>;

template struct LowRankSlotElmanBackward<__half>;
template struct LowRankSlotElmanBackward<__nv_bfloat16>;
template struct LowRankSlotElmanBackward<float>;
template struct LowRankSlotElmanBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
