// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E18: h-Aware Gate Elman - Three variants for output gating
//
// E18-A: output = h * silu(z + h)     -- add h to gate (FREE)
// E18-B: output = h * silu(z + Rh)    -- add Rh to gate (FREE, cache Rh)
// E18-E: output = h                   -- no gate (faster, fewer params)
//
// Base architecture same as E1:
// h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b)

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
// Native BF16 operations
// =============================================================================

__device__ __forceinline__ __nv_bfloat16 bf16_add(__nv_bfloat16 a, __nv_bfloat16 b) {
#if __CUDA_ARCH__ >= 800
    return __hadd(a, b);
#else
    return __float2bfloat16(__bfloat162float(a) + __bfloat162float(b));
#endif
}

// =============================================================================
// E18-A Forward: output = h * silu(z + h)
// =============================================================================

__global__ void FusedTanhGateWithH_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ Wx,
    const __nv_bfloat16* __restrict__ Rh,
    const __nv_bfloat16* __restrict__ b,
    const __nv_bfloat16* __restrict__ z,
    __nv_bfloat16* __restrict__ h_out,
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        __nv_bfloat16 sum = bf16_add(bf16_add(Wx[idx], Rh[idx]), b[d]);
        if (v_cache) v_cache[idx] = sum;

        float val = __bfloat162float(sum);
        float h_val = tanhf(val);
        h_out[idx] = __float2bfloat16(h_val);

        // E18-A: gate = silu(z + h)
        float z_val = __bfloat162float(z[idx]);
        float gate_input = z_val + h_val;  // Add h to gate
        float sigmoid_g = 1.0f / (1.0f + __expf(-gate_input));
        float silu_g = gate_input * sigmoid_g;

        output[idx] = __float2bfloat16(h_val * silu_g);
    }
}

template<typename T>
__global__ void FusedTanhGateWithH(
    const int batch_size,
    const int dim,
    const T* __restrict__ Wx,
    const T* __restrict__ Rh,
    const T* __restrict__ b,
    const T* __restrict__ z,
    T* __restrict__ h_out,
    T* __restrict__ output,
    T* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;
        float val = static_cast<float>(Wx[idx]) + static_cast<float>(Rh[idx]) + static_cast<float>(b[d]);
        if (v_cache) v_cache[idx] = static_cast<T>(val);

        float h_val = tanhf(val);
        h_out[idx] = static_cast<T>(h_val);

        float z_val = static_cast<float>(z[idx]);
        float gate_input = z_val + h_val;
        float sigmoid_g = 1.0f / (1.0f + expf(-gate_input));
        float silu_g = gate_input * sigmoid_g;

        output[idx] = static_cast<T>(h_val * silu_g);
    }
}

// =============================================================================
// E18-B Forward: output = h * silu(z + Rh)
// =============================================================================

__global__ void FusedTanhGateWithRh_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ Wx,
    const __nv_bfloat16* __restrict__ Rh,
    const __nv_bfloat16* __restrict__ b,
    const __nv_bfloat16* __restrict__ z,
    __nv_bfloat16* __restrict__ h_out,
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float rh_val = __bfloat162float(Rh[idx]);
        __nv_bfloat16 sum = bf16_add(bf16_add(Wx[idx], Rh[idx]), b[d]);
        if (v_cache) v_cache[idx] = sum;

        float val = __bfloat162float(sum);
        float h_val = tanhf(val);
        h_out[idx] = __float2bfloat16(h_val);

        // E18-B: gate = silu(z + Rh)
        float z_val = __bfloat162float(z[idx]);
        float gate_input = z_val + rh_val;  // Add Rh to gate
        float sigmoid_g = 1.0f / (1.0f + __expf(-gate_input));
        float silu_g = gate_input * sigmoid_g;

        output[idx] = __float2bfloat16(h_val * silu_g);
    }
}

template<typename T>
__global__ void FusedTanhGateWithRh(
    const int batch_size,
    const int dim,
    const T* __restrict__ Wx,
    const T* __restrict__ Rh,
    const T* __restrict__ b,
    const T* __restrict__ z,
    T* __restrict__ h_out,
    T* __restrict__ output,
    T* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;
        float rh_val = static_cast<float>(Rh[idx]);
        float val = static_cast<float>(Wx[idx]) + rh_val + static_cast<float>(b[d]);
        if (v_cache) v_cache[idx] = static_cast<T>(val);

        float h_val = tanhf(val);
        h_out[idx] = static_cast<T>(h_val);

        float z_val = static_cast<float>(z[idx]);
        float gate_input = z_val + rh_val;
        float sigmoid_g = 1.0f / (1.0f + expf(-gate_input));
        float silu_g = gate_input * sigmoid_g;

        output[idx] = static_cast<T>(h_val * silu_g);
    }
}

// =============================================================================
// E18-E Forward: output = h (no gate)
// =============================================================================

__global__ void FusedTanhNoGate_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ Wx,
    const __nv_bfloat16* __restrict__ Rh,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ h_out,
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        __nv_bfloat16 sum = bf16_add(bf16_add(Wx[idx], Rh[idx]), b[d]);
        if (v_cache) v_cache[idx] = sum;

        float val = __bfloat162float(sum);
        float h_val = tanhf(val);
        __nv_bfloat16 h_bf16 = __float2bfloat16(h_val);

        h_out[idx] = h_bf16;
        output[idx] = h_bf16;  // E18-E: no gate, output = h
    }
}

template<typename T>
__global__ void FusedTanhNoGate(
    const int batch_size,
    const int dim,
    const T* __restrict__ Wx,
    const T* __restrict__ Rh,
    const T* __restrict__ b,
    T* __restrict__ h_out,
    T* __restrict__ output,
    T* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;
        float val = static_cast<float>(Wx[idx]) + static_cast<float>(Rh[idx]) + static_cast<float>(b[d]);
        if (v_cache) v_cache[idx] = static_cast<T>(val);

        float h_val = tanhf(val);
        T h_t = static_cast<T>(h_val);
        h_out[idx] = h_t;
        output[idx] = h_t;  // No gate
    }
}

// =============================================================================
// E18-A Backward: gate = silu(z + h)
// d_output/dh has TWO paths: through multiply AND through gate
// =============================================================================

__global__ void GateBackwardWithH_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ h,
    const __nv_bfloat16* __restrict__ z,
    const __nv_bfloat16* __restrict__ d_output,
    __nv_bfloat16* __restrict__ dh,
    __nv_bfloat16* __restrict__ dz) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        float h_val = __bfloat162float(h[idx]);
        float z_val = __bfloat162float(z[idx]);
        float dout = __bfloat162float(d_output[idx]);

        // Forward: gate_input = z + h, gate = silu(gate_input), output = h * gate
        float gate_input = z_val + h_val;
        float sigmoid_g = 1.0f / (1.0f + __expf(-gate_input));
        float silu_g = gate_input * sigmoid_g;

        // d_silu/d_gate_input
        float dsilu = sigmoid_g * (1.0f + gate_input * (1.0f - sigmoid_g));

        // d_output/d_h = gate (from h * gate) + h * dsilu * 1 (h appears in gate_input)
        float dh_val = dout * silu_g + dout * h_val * dsilu;

        // d_output/d_z = h * dsilu * 1 (z appears in gate_input)
        float dz_val = dout * h_val * dsilu;

        dh[idx] = __float2bfloat16(dh_val);
        dz[idx] = __float2bfloat16(dz_val);
    }
}

template<typename T>
__global__ void GateBackwardWithH(
    const int batch_size,
    const int dim,
    const T* __restrict__ h,
    const T* __restrict__ z,
    const T* __restrict__ d_output,
    T* __restrict__ dh,
    T* __restrict__ dz) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        float h_val = static_cast<float>(h[idx]);
        float z_val = static_cast<float>(z[idx]);
        float dout = static_cast<float>(d_output[idx]);

        float gate_input = z_val + h_val;
        float sigmoid_g = 1.0f / (1.0f + expf(-gate_input));
        float silu_g = gate_input * sigmoid_g;
        float dsilu = sigmoid_g * (1.0f + gate_input * (1.0f - sigmoid_g));

        float dh_val = dout * silu_g + dout * h_val * dsilu;
        float dz_val = dout * h_val * dsilu;

        dh[idx] = static_cast<T>(dh_val);
        dz[idx] = static_cast<T>(dz_val);
    }
}

// =============================================================================
// E18-B Backward: gate = silu(z + Rh)
// Need to also compute dRh for W_h gradient
// =============================================================================

__global__ void GateBackwardWithRh_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ h,
    const __nv_bfloat16* __restrict__ z,
    const __nv_bfloat16* __restrict__ Rh_cache,  // Cached Rh from forward
    const __nv_bfloat16* __restrict__ d_output,
    __nv_bfloat16* __restrict__ dh,
    __nv_bfloat16* __restrict__ dz,
    __nv_bfloat16* __restrict__ dRh_gate) {  // Extra gradient to Rh from gate

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        float h_val = __bfloat162float(h[idx]);
        float z_val = __bfloat162float(z[idx]);
        float rh_val = __bfloat162float(Rh_cache[idx]);
        float dout = __bfloat162float(d_output[idx]);

        // Forward: gate_input = z + Rh, gate = silu(gate_input), output = h * gate
        float gate_input = z_val + rh_val;
        float sigmoid_g = 1.0f / (1.0f + __expf(-gate_input));
        float silu_g = gate_input * sigmoid_g;
        float dsilu = sigmoid_g * (1.0f + gate_input * (1.0f - sigmoid_g));

        // d_output/d_h = gate (h only appears in multiply, not in gate)
        float dh_val = dout * silu_g;

        // d_output/d_z = h * dsilu
        float dz_val = dout * h_val * dsilu;

        // d_output/d_Rh = h * dsilu (Rh appears in gate_input)
        float dRh_val = dout * h_val * dsilu;

        dh[idx] = __float2bfloat16(dh_val);
        dz[idx] = __float2bfloat16(dz_val);
        dRh_gate[idx] = __float2bfloat16(dRh_val);
    }
}

template<typename T>
__global__ void GateBackwardWithRh(
    const int batch_size,
    const int dim,
    const T* __restrict__ h,
    const T* __restrict__ z,
    const T* __restrict__ Rh_cache,
    const T* __restrict__ d_output,
    T* __restrict__ dh,
    T* __restrict__ dz,
    T* __restrict__ dRh_gate) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        float h_val = static_cast<float>(h[idx]);
        float z_val = static_cast<float>(z[idx]);
        float rh_val = static_cast<float>(Rh_cache[idx]);
        float dout = static_cast<float>(d_output[idx]);

        float gate_input = z_val + rh_val;
        float sigmoid_g = 1.0f / (1.0f + expf(-gate_input));
        float silu_g = gate_input * sigmoid_g;
        float dsilu = sigmoid_g * (1.0f + gate_input * (1.0f - sigmoid_g));

        dh[idx] = static_cast<T>(dout * silu_g);
        dz[idx] = static_cast<T>(dout * h_val * dsilu);
        dRh_gate[idx] = static_cast<T>(dout * h_val * dsilu);
    }
}

// =============================================================================
// E18-E Backward: no gate, trivial
// =============================================================================

// For E18-E, dh = d_output directly (no gate backward needed)
// Just need tanh backward

// =============================================================================
// Shared backward kernels
// =============================================================================

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
            grad = __bfloat162float(dh[idx]) + __bfloat162float(dh_recurrent[idx]);
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

template<typename T>
__global__ void TanhBackwardKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ v,
    const T* __restrict__ dh,
    const T* __restrict__ dh_recurrent,
    T* __restrict__ dv,
    float* __restrict__ db) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

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

__global__ void VectorAddInplace_BF16(const int n, __nv_bfloat16* a, const __nv_bfloat16* b) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = bf16_add(a[idx], b[idx]);
    }
}

template<typename T>
__global__ void VectorAddInplace(const int n, T* a, const T* b) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = static_cast<T>(static_cast<float>(a[idx]) + static_cast<float>(b[idx]));
    }
}

template<typename T>
__global__ void CopyFloatToT(const int n, const float* src, T* dst) {
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
// E18-A: h-Aware Gate Forward - BF16 Specialization
// =============================================================================

template<>
HAwareGateElmanForward<__nv_bfloat16>::HAwareGateElmanForward(
    bool training,
    int batch_size,
    int dim,
    int gate_mode,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : training_(training),
      batch_size_(batch_size),
      dim_(dim),
      gate_mode_(gate_mode),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void HAwareGateElmanForward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_x,
    const __nv_bfloat16* W_h,
    const __nv_bfloat16* b,
    const __nv_bfloat16* x,
    const __nv_bfloat16* z,
    __nv_bfloat16* h,
    __nv_bfloat16* output,
    __nv_bfloat16* v,
    __nv_bfloat16* Rh_cache,
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

    // Process each timestep
    for (int t = 0; t < steps; ++t) {
        const __nv_bfloat16* Wx_t = tmp_Wx + t * BD;
        const __nv_bfloat16* h_prev = h + t * BD;
        const __nv_bfloat16* z_t = z + t * BD;
        __nv_bfloat16* h_t = h + (t + 1) * BD;
        __nv_bfloat16* out_t = output + t * BD;
        __nv_bfloat16* v_t = training_ ? (v + t * BD) : nullptr;

        // tmp_Rh = h_prev @ W_h.T
        blas<__nv_bfloat16>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha,
            W_h, dim_,
            h_prev, dim_,
            &beta_zero,
            tmp_Rh, dim_);

        // Cache Rh for E18-B backward (if needed)
        if (gate_mode_ == 1 && Rh_cache && training_) {
            cudaMemcpyAsync(Rh_cache + t * BD, tmp_Rh, BD * sizeof(__nv_bfloat16),
                           cudaMemcpyDeviceToDevice, stream_);
        }

        // Select kernel based on gate_mode
        if (gate_mode_ == 0) {
            // E18-A: gate = silu(z + h)
            FusedTanhGateWithH_BF16<<<num_blocks, block_size, 0, stream_>>>(
                batch_size_, dim_, Wx_t, tmp_Rh, b, z_t, h_t, out_t, v_t);
        } else if (gate_mode_ == 1) {
            // E18-B: gate = silu(z + Rh)
            FusedTanhGateWithRh_BF16<<<num_blocks, block_size, 0, stream_>>>(
                batch_size_, dim_, Wx_t, tmp_Rh, b, z_t, h_t, out_t, v_t);
        } else {
            // E18-E: no gate
            FusedTanhNoGate_BF16<<<num_blocks, block_size, 0, stream_>>>(
                batch_size_, dim_, Wx_t, tmp_Rh, b, h_t, out_t, v_t);
        }
    }
}

// =============================================================================
// E18 Backward - BF16 Specialization
// =============================================================================

template<>
HAwareGateElmanBackward<__nv_bfloat16>::HAwareGateElmanBackward(
    int batch_size,
    int dim,
    int gate_mode,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      gate_mode_(gate_mode),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void HAwareGateElmanBackward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_x,
    const __nv_bfloat16* W_h,
    const __nv_bfloat16* x,
    const __nv_bfloat16* z,
    const __nv_bfloat16* h,
    const __nv_bfloat16* v,
    const __nv_bfloat16* Rh_cache,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* dx,
    __nv_bfloat16* dz,
    __nv_bfloat16* dW_x,
    __nv_bfloat16* dW_h,
    __nv_bfloat16* db,
    __nv_bfloat16* workspace) {

    static const __nv_bfloat16 alpha_one = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);
    static const __nv_bfloat16 beta_one = __float2bfloat16(1.0f);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    __nv_bfloat16* dv_all = workspace;
    __nv_bfloat16* dh = workspace + steps * BD;
    __nv_bfloat16* dh_recurrent = workspace + (steps + 1) * BD;
    __nv_bfloat16* dRh_gate = workspace + (steps + 2) * BD;  // For E18-B
    float* db_float = reinterpret_cast<float*>(workspace + (steps + 3) * BD);

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW_x, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_h, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);

    // For E18-E, we also need to zero dz
    if (gate_mode_ == 2) {
        cudaMemsetAsync(dz, 0, steps * BD * sizeof(__nv_bfloat16), stream_);
    }

    for (int t = steps - 1; t >= 0; --t) {
        const __nv_bfloat16* v_t = v + t * BD;
        const __nv_bfloat16* h_t = h + (t + 1) * BD;
        const __nv_bfloat16* z_t = z + t * BD;
        const __nv_bfloat16* d_out_t = d_output + t * BD;
        __nv_bfloat16* dv_t = dv_all + t * BD;
        __nv_bfloat16* dz_t = dz + t * BD;

        // Backward through gate based on mode
        if (gate_mode_ == 0) {
            // E18-A: gate = silu(z + h)
            GateBackwardWithH_BF16<<<num_blocks, block_size, 0, stream_>>>(
                batch_size_, dim_, h_t, z_t, d_out_t, dh, dz_t);
        } else if (gate_mode_ == 1) {
            // E18-B: gate = silu(z + Rh)
            const __nv_bfloat16* Rh_t = Rh_cache + t * BD;
            GateBackwardWithRh_BF16<<<num_blocks, block_size, 0, stream_>>>(
                batch_size_, dim_, h_t, z_t, Rh_t, d_out_t, dh, dz_t, dRh_gate);
        } else {
            // E18-E: no gate, dh = d_output directly
            cudaMemcpyAsync(dh, d_out_t, BD * sizeof(__nv_bfloat16),
                           cudaMemcpyDeviceToDevice, stream_);
        }

        // Add recurrent gradient
        VectorAddInplace_BF16<<<num_blocks, block_size, 0, stream_>>>(BD, dh, dh_recurrent);

        // Backward through tanh
        TanhBackwardKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, v_t, dh, nullptr, dv_t, db_float);

        // For E18-B, add dRh_gate to dv (since Rh = W_h @ h_prev, gradient flows through)
        if (gate_mode_ == 1) {
            VectorAddInplace_BF16<<<num_blocks, block_size, 0, stream_>>>(BD, dv_t, dRh_gate);
        }

        // dh_recurrent = W_h @ dv
        if (t > 0) {
            blas<__nv_bfloat16>::gemm(
                blas_handle_,
                CUBLAS_OP_N, CUBLAS_OP_N,
                dim_, batch_size_, dim_,
                &alpha_one,
                W_h, dim_,
                dv_t, dim_,
                &beta_zero,
                dh_recurrent, dim_);
        }
    }

    // Batch GEMMs for weight gradients
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha_one,
        W_x, dim_,
        dv_all, dim_,
        &beta_zero,
        dx, dim_);

    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        dv_all, dim_,
        &beta_one,
        dW_x, dim_);

    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        h, dim_,
        dv_all, dim_,
        &beta_one,
        dW_h, dim_);

    CopyFloatToT<__nv_bfloat16><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_float, db);
}

// =============================================================================
// Generic Template Implementations
// =============================================================================

template<typename T>
HAwareGateElmanForward<T>::HAwareGateElmanForward(
    bool training,
    int batch_size,
    int dim,
    int gate_mode,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : training_(training),
      batch_size_(batch_size),
      dim_(dim),
      gate_mode_(gate_mode),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void HAwareGateElmanForward<T>::Run(
    int steps,
    const T* W_x,
    const T* W_h,
    const T* b,
    const T* x,
    const T* z,
    T* h,
    T* output,
    T* v,
    T* Rh_cache,
    T* workspace) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    T* tmp_Wx = workspace;
    T* tmp_Rh = workspace + steps * BD;

    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha,
        W_x, dim_,
        x, dim_,
        &beta_zero,
        tmp_Wx, dim_);

    for (int t = 0; t < steps; ++t) {
        const T* Wx_t = tmp_Wx + t * BD;
        const T* h_prev = h + t * BD;
        const T* z_t = z + t * BD;
        T* h_t = h + (t + 1) * BD;
        T* out_t = output + t * BD;
        T* v_t = training_ ? (v + t * BD) : nullptr;

        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_,
            &alpha,
            W_h, dim_,
            h_prev, dim_,
            &beta_zero,
            tmp_Rh, dim_);

        if (gate_mode_ == 1 && Rh_cache && training_) {
            cudaMemcpyAsync(Rh_cache + t * BD, tmp_Rh, BD * sizeof(T),
                           cudaMemcpyDeviceToDevice, stream_);
        }

        if (gate_mode_ == 0) {
            FusedTanhGateWithH<T><<<num_blocks, block_size, 0, stream_>>>(
                batch_size_, dim_, Wx_t, tmp_Rh, b, z_t, h_t, out_t, v_t);
        } else if (gate_mode_ == 1) {
            FusedTanhGateWithRh<T><<<num_blocks, block_size, 0, stream_>>>(
                batch_size_, dim_, Wx_t, tmp_Rh, b, z_t, h_t, out_t, v_t);
        } else {
            FusedTanhNoGate<T><<<num_blocks, block_size, 0, stream_>>>(
                batch_size_, dim_, Wx_t, tmp_Rh, b, h_t, out_t, v_t);
        }
    }
}

template<typename T>
HAwareGateElmanBackward<T>::HAwareGateElmanBackward(
    int batch_size,
    int dim,
    int gate_mode,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      gate_mode_(gate_mode),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void HAwareGateElmanBackward<T>::Run(
    int steps,
    const T* W_x,
    const T* W_h,
    const T* x,
    const T* z,
    const T* h,
    const T* v,
    const T* Rh_cache,
    const T* d_output,
    T* dx,
    T* dz,
    T* dW_x,
    T* dW_h,
    T* db,
    T* workspace) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    T* dv_all = workspace;
    T* dh = workspace + steps * BD;
    T* dh_recurrent = workspace + (steps + 1) * BD;
    T* dRh_gate = workspace + (steps + 2) * BD;
    float* db_float = reinterpret_cast<float*>(workspace + (steps + 3) * BD);

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(T), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW_x, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dW_h, 0, dim_ * dim_ * sizeof(T), stream_);

    if (gate_mode_ == 2) {
        cudaMemsetAsync(dz, 0, steps * BD * sizeof(T), stream_);
    }

    for (int t = steps - 1; t >= 0; --t) {
        const T* v_t = v + t * BD;
        const T* h_t = h + (t + 1) * BD;
        const T* z_t = z + t * BD;
        const T* d_out_t = d_output + t * BD;
        T* dv_t = dv_all + t * BD;
        T* dz_t = dz + t * BD;

        if (gate_mode_ == 0) {
            GateBackwardWithH<T><<<num_blocks, block_size, 0, stream_>>>(
                batch_size_, dim_, h_t, z_t, d_out_t, dh, dz_t);
        } else if (gate_mode_ == 1) {
            const T* Rh_t = Rh_cache + t * BD;
            GateBackwardWithRh<T><<<num_blocks, block_size, 0, stream_>>>(
                batch_size_, dim_, h_t, z_t, Rh_t, d_out_t, dh, dz_t, dRh_gate);
        } else {
            cudaMemcpyAsync(dh, d_out_t, BD * sizeof(T), cudaMemcpyDeviceToDevice, stream_);
        }

        VectorAddInplace<T><<<num_blocks, block_size, 0, stream_>>>(BD, dh, dh_recurrent);

        TanhBackwardKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, v_t, dh, nullptr, dv_t, db_float);

        if (gate_mode_ == 1) {
            VectorAddInplace<T><<<num_blocks, block_size, 0, stream_>>>(BD, dv_t, dRh_gate);
        }

        if (t > 0) {
            blas<T>::gemm(
                blas_handle_,
                CUBLAS_OP_N, CUBLAS_OP_N,
                dim_, batch_size_, dim_,
                &alpha,
                W_h, dim_,
                dv_t, dim_,
                &beta_zero,
                dh_recurrent, dim_);
        }
    }

    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha,
        W_x, dim_,
        dv_all, dim_,
        &beta_zero,
        dx, dim_);

    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha,
        x, dim_,
        dv_all, dim_,
        &beta_one,
        dW_x, dim_);

    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha,
        h, dim_,
        dv_all, dim_,
        &beta_one,
        dW_h, dim_);

    CopyFloatToT<T><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, db_float, db);
}

// Explicit template instantiations
template struct HAwareGateElmanForward<__half>;
template struct HAwareGateElmanForward<float>;
template struct HAwareGateElmanForward<double>;

template struct HAwareGateElmanBackward<__half>;
template struct HAwareGateElmanBackward<float>;
template struct HAwareGateElmanBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
