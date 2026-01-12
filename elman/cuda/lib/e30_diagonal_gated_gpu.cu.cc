// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E30: E1 + SSM-style diagonal gating
//
// Like E1 but with learned element-wise scales on the gate:
//   gate = silu(z * g_z + h * g_h + b_gate)
//   output = h * gate
//
// Extra params: 3*dim (g_z, g_h, b_gate) - negligible

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

// Native bf16 operations
__device__ __forceinline__ __nv_bfloat16 bf16_add(__nv_bfloat16 a, __nv_bfloat16 b) {
#if __CUDA_ARCH__ >= 800
    return __hadd(a, b);
#else
    return __float2bfloat16(__bfloat162float(a) + __bfloat162float(b));
#endif
}

__device__ __forceinline__ __nv_bfloat16 bf16_mul(__nv_bfloat16 a, __nv_bfloat16 b) {
#if __CUDA_ARCH__ >= 800
    return __hmul(a, b);
#else
    return __float2bfloat16(__bfloat162float(a) * __bfloat162float(b));
#endif
}

// =============================================================================
// Forward Kernels
// =============================================================================

// Fused tanh kernel (same as E1)
__global__ void FusedTanhKernel_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ Wx,
    const __nv_bfloat16* __restrict__ Rh,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ h_out,
    __nv_bfloat16* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;
        __nv_bfloat16 sum = bf16_add(bf16_add(Wx[idx], Rh[idx]), b[d]);
        if (v_cache) v_cache[idx] = sum;
        float val = __bfloat162float(sum);
        h_out[idx] = __float2bfloat16(tanhf(val));
    }
}

// E30 diagonal gate forward: gate = silu(z * g_z + h * g_h + b_gate), output = h * gate
__global__ void DiagonalGateForward_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ h,
    const __nv_bfloat16* __restrict__ z,
    const __nv_bfloat16* __restrict__ g_z,
    const __nv_bfloat16* __restrict__ g_h,
    const __nv_bfloat16* __restrict__ b_gate,
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ gate_input_cache) {  // Cache for backward

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float h_val = __bfloat162float(h[idx]);
        float z_val = __bfloat162float(z[idx]);
        float gz = __bfloat162float(g_z[d]);
        float gh = __bfloat162float(g_h[d]);
        float bg = __bfloat162float(b_gate[d]);

        // gate_input = z * g_z + h * g_h + b_gate
        float gate_input = z_val * gz + h_val * gh + bg;

        // Cache for backward
        if (gate_input_cache) gate_input_cache[idx] = __float2bfloat16(gate_input);

        // gate = silu(gate_input)
        float sigmoid_gi = 1.0f / (1.0f + __expf(-gate_input));
        float gate = gate_input * sigmoid_gi;

        output[idx] = __float2bfloat16(h_val * gate);
    }
}

// Fused tanh + diagonal gate (reduces memory traffic)
__global__ void FusedTanhDiagonalGateKernel_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ Wx,
    const __nv_bfloat16* __restrict__ Rh,
    const __nv_bfloat16* __restrict__ b,
    const __nv_bfloat16* __restrict__ z,
    const __nv_bfloat16* __restrict__ g_z,
    const __nv_bfloat16* __restrict__ g_h,
    const __nv_bfloat16* __restrict__ b_gate,
    __nv_bfloat16* __restrict__ h_out,
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ v_cache,
    __nv_bfloat16* __restrict__ gate_input_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // Compute h = tanh(Wx + Rh + b)
        __nv_bfloat16 sum = bf16_add(bf16_add(Wx[idx], Rh[idx]), b[d]);
        if (v_cache) v_cache[idx] = sum;
        float val = __bfloat162float(sum);
        float h_val = tanhf(val);
        h_out[idx] = __float2bfloat16(h_val);

        // Compute gate = silu(z * g_z + h * g_h + b_gate)
        float z_val = __bfloat162float(z[idx]);
        float gz = __bfloat162float(g_z[d]);
        float gh = __bfloat162float(g_h[d]);
        float bg = __bfloat162float(b_gate[d]);

        float gate_input = z_val * gz + h_val * gh + bg;
        if (gate_input_cache) gate_input_cache[idx] = __float2bfloat16(gate_input);

        float sigmoid_gi = 1.0f / (1.0f + __expf(-gate_input));
        float gate = gate_input * sigmoid_gi;

        output[idx] = __float2bfloat16(h_val * gate);
    }
}

// =============================================================================
// Backward Kernels
// =============================================================================

// E30 diagonal gate backward
// d_output -> dh, dz, dg_z, dg_h, db_gate
// dh_recurrent: gradient from next timestep (can be null for last timestep)
__global__ void DiagonalGateBackward_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ h,
    const __nv_bfloat16* __restrict__ z,
    const __nv_bfloat16* __restrict__ g_z,
    const __nv_bfloat16* __restrict__ g_h,
    const __nv_bfloat16* __restrict__ gate_input_cache,
    const __nv_bfloat16* __restrict__ d_output,
    const __nv_bfloat16* __restrict__ dh_recurrent,  // Gradient from next timestep
    __nv_bfloat16* __restrict__ dh,
    __nv_bfloat16* __restrict__ dz,
    float* __restrict__ dg_z,  // Accumulated in float
    float* __restrict__ dg_h,
    float* __restrict__ db_gate) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float h_val = __bfloat162float(h[idx]);
        float z_val = __bfloat162float(z[idx]);
        float gz = __bfloat162float(g_z[d]);
        float gh = __bfloat162float(g_h[d]);
        float gate_input = __bfloat162float(gate_input_cache[idx]);
        float dout = __bfloat162float(d_output[idx]);

        // gate = silu(gate_input) = gate_input * sigmoid(gate_input)
        float sigmoid_gi = 1.0f / (1.0f + __expf(-gate_input));
        float gate = gate_input * sigmoid_gi;

        // dsilu/d(gate_input) = sigmoid + gate_input * sigmoid * (1 - sigmoid)
        //                     = sigmoid * (1 + gate_input * (1 - sigmoid))
        float dsilu = sigmoid_gi * (1.0f + gate_input * (1.0f - sigmoid_gi));

        // output = h * gate
        // d_gate = dout * h
        float d_gate = dout * h_val;

        // d_gate_input = d_gate * dsilu
        float d_gate_input = d_gate * dsilu;

        // gate_input = z * g_z + h * g_h + b_gate
        // dh from gate: d_gate_input * g_h
        // dh from output: dout * gate
        float dh_val = dout * gate + d_gate_input * gh;

        // Add recurrent gradient from next timestep
        if (dh_recurrent) {
            dh_val += __bfloat162float(dh_recurrent[idx]);
        }

        // dz = d_gate_input * g_z
        float dz_val = d_gate_input * gz;

        dh[idx] = __float2bfloat16(dh_val);
        dz[idx] = __float2bfloat16(dz_val);

        // Accumulate gradients for diagonal params
        atomicAdd(&dg_z[d], d_gate_input * z_val);
        atomicAdd(&dg_h[d], d_gate_input * h_val);
        atomicAdd(&db_gate[d], d_gate_input);
    }
}

// Tanh backward (same as E1)
__global__ void TanhBackwardKernel_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ dh,
    const __nv_bfloat16* __restrict__ dh_recurrent,
    __nv_bfloat16* __restrict__ dv,
    float* __restrict__ db) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float grad;
        if (dh_recurrent) {
            __nv_bfloat16 combined = bf16_add(dh[idx], dh_recurrent[idx]);
            grad = __bfloat162float(combined);
        } else {
            grad = __bfloat162float(dh[idx]);
        }

        float h = tanhf(__bfloat162float(v[idx]));
        float dtanh = 1.0f - h * h;
        float dv_val = grad * dtanh;

        dv[idx] = __float2bfloat16(dv_val);
        atomicAdd(&db[d], dv_val);
    }
}

// Vector add inplace
__global__ void VectorAddInplace_BF16(
    const int n,
    __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = bf16_add(a[idx], b[idx]);
    }
}

// Copy float to bf16
__global__ void CopyFloatToBF16(const int n, const float* __restrict__ src, __nv_bfloat16* __restrict__ dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2bfloat16(src[idx]);
    }
}

}  // anonymous namespace


namespace hasty {
namespace v0 {
namespace elman_ladder {

// =============================================================================
// E30 Diagonal Gated Forward - BF16 Specialization
// =============================================================================

template<>
E30DiagonalGatedForward<__nv_bfloat16>::E30DiagonalGatedForward(
    bool training,
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : training_(training),
      batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void E30DiagonalGatedForward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_x,
    const __nv_bfloat16* W_h,
    const __nv_bfloat16* b,
    const __nv_bfloat16* g_z,
    const __nv_bfloat16* g_h,
    const __nv_bfloat16* b_gate,
    const __nv_bfloat16* x,
    const __nv_bfloat16* z,
    __nv_bfloat16* h,
    __nv_bfloat16* output,
    __nv_bfloat16* v,
    __nv_bfloat16* gate_input_cache,
    __nv_bfloat16* workspace) {

    static const __nv_bfloat16 alpha = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    __nv_bfloat16* tmp_Wx = workspace;
    __nv_bfloat16* tmp_Rh = workspace + steps * BD;

    // Pre-compute W_x @ x for all timesteps
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha,
        W_x, dim_,
        x, dim_,
        &beta_zero,
        tmp_Wx, dim_);

    // Sequential recurrence
    for (int t = 0; t < steps; ++t) {
        const __nv_bfloat16* h_prev = h + t * BD;
        __nv_bfloat16* h_curr = h + (t + 1) * BD;
        __nv_bfloat16* Wx_t = tmp_Wx + t * BD;
        const __nv_bfloat16* z_t = z + t * BD;
        __nv_bfloat16* out_t = output + t * BD;
        __nv_bfloat16* v_t = training_ ? (v + t * BD) : nullptr;
        __nv_bfloat16* gi_t = training_ ? (gate_input_cache + t * BD) : nullptr;

        // W_h @ h_prev
        blas<__nv_bfloat16>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha,
            W_h, dim_,
            h_prev, dim_,
            &beta_zero,
            tmp_Rh, dim_);

        // Fused tanh + diagonal gate
        FusedTanhDiagonalGateKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_,
            Wx_t, tmp_Rh, b, z_t,
            g_z, g_h, b_gate,
            h_curr, out_t, v_t, gi_t);
    }
}

// =============================================================================
// E30 Diagonal Gated Backward - BF16 Specialization
// =============================================================================

template<>
E30DiagonalGatedBackward<__nv_bfloat16>::E30DiagonalGatedBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream) {
    // Allocate float accumulators for diagonal gradients
    cudaMalloc(&dg_z_accum_, dim * sizeof(float));
    cudaMalloc(&dg_h_accum_, dim * sizeof(float));
    cudaMalloc(&db_gate_accum_, dim * sizeof(float));
    cudaMalloc(&db_accum_, dim * sizeof(float));
}

template<>
E30DiagonalGatedBackward<__nv_bfloat16>::~E30DiagonalGatedBackward() {
    cudaFree(dg_z_accum_);
    cudaFree(dg_h_accum_);
    cudaFree(db_gate_accum_);
    cudaFree(db_accum_);
}

template<>
void E30DiagonalGatedBackward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_x,
    const __nv_bfloat16* W_h,
    const __nv_bfloat16* g_z,
    const __nv_bfloat16* g_h,
    const __nv_bfloat16* x,
    const __nv_bfloat16* z,
    const __nv_bfloat16* h,
    const __nv_bfloat16* v,
    const __nv_bfloat16* gate_input_cache,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* dx,
    __nv_bfloat16* dz,
    __nv_bfloat16* dW_x,
    __nv_bfloat16* dW_h,
    __nv_bfloat16* db,
    __nv_bfloat16* dg_z,
    __nv_bfloat16* dg_h,
    __nv_bfloat16* db_gate,
    __nv_bfloat16* workspace) {

    static const __nv_bfloat16 alpha = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);
    static const __nv_bfloat16 beta_one = __float2bfloat16(1.0f);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;
    const int num_blocks_dim = (dim_ + block_size - 1) / block_size;

    // Workspace layout: dh_t, dv_t
    __nv_bfloat16* dh_t = workspace;
    __nv_bfloat16* dv_t = workspace + BD;

    // Zero accumulators
    cudaMemsetAsync(dg_z_accum_, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dg_h_accum_, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_gate_accum_, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_accum_, 0, dim_ * sizeof(float), stream_);

    // Zero weight gradients
    cudaMemsetAsync(dW_x, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_h, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);

    // Initialize dh_t with zeros (will be accumulated from recurrence)
    cudaMemsetAsync(dh_t, 0, BD * sizeof(__nv_bfloat16), stream_);

    // Backward through time
    for (int t = steps - 1; t >= 0; --t) {
        const __nv_bfloat16* h_t = h + (t + 1) * BD;  // h after tanh
        const __nv_bfloat16* h_prev = h + t * BD;
        const __nv_bfloat16* z_t = z + t * BD;
        const __nv_bfloat16* v_t = v + t * BD;
        const __nv_bfloat16* gi_t = gate_input_cache + t * BD;
        const __nv_bfloat16* d_out_t = d_output + t * BD;
        __nv_bfloat16* dx_t = dx + t * BD;
        __nv_bfloat16* dz_t = dz + t * BD;

        // Diagonal gate backward: d_output -> dh, dz, dg_z, dg_h, db_gate
        // Pass dh_t as recurrent gradient (contains dh_prev from previous iteration)
        // At t=steps-1, dh_t is zero from memset, so no recurrent contribution
        const __nv_bfloat16* dh_recurrent = (t < steps - 1) ? dh_t : nullptr;
        DiagonalGateBackward_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_,
            h_t, z_t, g_z, g_h, gi_t,
            d_out_t,
            dh_recurrent,  // Input: recurrent gradient from next timestep
            dh_t,          // Output: total dh
            dz_t,
            dg_z_accum_, dg_h_accum_, db_gate_accum_);

        // Tanh backward: dh -> dv, db
        TanhBackwardKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_,
            v_t, dh_t, nullptr,
            dv_t, db_accum_);

        // dW_x += dv.T @ x
        // Row-major: dv [B,D].T @ x [B,D] = [D,D]
        // cuBLAS: gemm(OP_N, OP_T, D, D, B, x, D, dv, D, C, D) gives C_row = dv.T @ x
        blas<__nv_bfloat16>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_,
            &alpha,
            x + t * BD, dim_,
            dv_t, dim_,
            &beta_one,
            dW_x, dim_);

        // dW_h += dv.T @ h_prev
        blas<__nv_bfloat16>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_,
            &alpha,
            h_prev, dim_,
            dv_t, dim_,
            &beta_one,
            dW_h, dim_);

        // dx = dv @ W_x
        blas<__nv_bfloat16>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha,
            W_x, dim_,
            dv_t, dim_,
            &beta_zero,
            dx_t, dim_);

        // dh_prev = dv @ W_h (for next iteration)
        blas<__nv_bfloat16>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha,
            W_h, dim_,
            dv_t, dim_,
            &beta_zero,
            dh_t, dim_);
    }

    // Copy float accumulators to bf16 outputs
    CopyFloatToBF16<<<num_blocks_dim, block_size, 0, stream_>>>(dim_, dg_z_accum_, dg_z);
    CopyFloatToBF16<<<num_blocks_dim, block_size, 0, stream_>>>(dim_, dg_h_accum_, dg_h);
    CopyFloatToBF16<<<num_blocks_dim, block_size, 0, stream_>>>(dim_, db_gate_accum_, db_gate);
    CopyFloatToBF16<<<num_blocks_dim, block_size, 0, stream_>>>(dim_, db_accum_, db);
}

// Explicit instantiations
template struct E30DiagonalGatedForward<__nv_bfloat16>;
template struct E30DiagonalGatedBackward<__nv_bfloat16>;

}}}  // namespace hasty::v0::elman_ladder
