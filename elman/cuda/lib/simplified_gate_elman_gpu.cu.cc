// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E19: Simplified Gate Elman - Various gate simplifications
//
// E19-A: gate = silu(Wx + h + b_gate)    -- reuse Wx, DROP W_gate
// E19-B: gate = silu(h + b_gate)         -- h-only gate
// E19-D: h = tanh(Wx + Rh + h_prev + b)  -- residual connection
// E19-E: Combined A + D                  -- residual + reuse Wx
//
// Key insight: E18-A showed h in gate helps. E19 tests if W_gate is redundant.

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
// E19-A Forward: gate = silu(Wx + h + b_gate), no residual
// =============================================================================

__global__ void FusedTanhGateWxPlusH_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ Wx,
    const __nv_bfloat16* __restrict__ Rh,
    const __nv_bfloat16* __restrict__ b,
    const __nv_bfloat16* __restrict__ b_gate,
    __nv_bfloat16* __restrict__ h_out,
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float wx_val = __bfloat162float(Wx[idx]);
        float rh_val = __bfloat162float(Rh[idx]);
        float b_val = __bfloat162float(b[d]);
        float bg_val = __bfloat162float(b_gate[d]);

        // h = tanh(Wx + Rh + b)
        float val = wx_val + rh_val + b_val;
        if (v_cache) v_cache[idx] = __float2bfloat16(val);
        float h_val = tanhf(val);
        h_out[idx] = __float2bfloat16(h_val);

        // E19-A: gate = silu(Wx + h + b_gate)
        float gate_input = wx_val + h_val + bg_val;
        float sigmoid_g = 1.0f / (1.0f + __expf(-gate_input));
        float silu_g = gate_input * sigmoid_g;

        output[idx] = __float2bfloat16(h_val * silu_g);
    }
}

template<typename T>
__global__ void FusedTanhGateWxPlusH(
    const int batch_size,
    const int dim,
    const T* __restrict__ Wx,
    const T* __restrict__ Rh,
    const T* __restrict__ b,
    const T* __restrict__ b_gate,
    T* __restrict__ h_out,
    T* __restrict__ output,
    T* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float wx_val = static_cast<float>(Wx[idx]);
        float rh_val = static_cast<float>(Rh[idx]);
        float b_val = static_cast<float>(b[d]);
        float bg_val = static_cast<float>(b_gate[d]);

        float val = wx_val + rh_val + b_val;
        if (v_cache) v_cache[idx] = static_cast<T>(val);
        float h_val = tanhf(val);
        h_out[idx] = static_cast<T>(h_val);

        float gate_input = wx_val + h_val + bg_val;
        float sigmoid_g = 1.0f / (1.0f + expf(-gate_input));
        float silu_g = gate_input * sigmoid_g;

        output[idx] = static_cast<T>(h_val * silu_g);
    }
}

// =============================================================================
// E19-B Forward: gate = silu(h + b_gate), h-only gate
// =============================================================================

__global__ void FusedTanhGateHOnly_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ Wx,
    const __nv_bfloat16* __restrict__ Rh,
    const __nv_bfloat16* __restrict__ b,
    const __nv_bfloat16* __restrict__ b_gate,
    __nv_bfloat16* __restrict__ h_out,
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float wx_val = __bfloat162float(Wx[idx]);
        float rh_val = __bfloat162float(Rh[idx]);
        float b_val = __bfloat162float(b[d]);
        float bg_val = __bfloat162float(b_gate[d]);

        float val = wx_val + rh_val + b_val;
        if (v_cache) v_cache[idx] = __float2bfloat16(val);
        float h_val = tanhf(val);
        h_out[idx] = __float2bfloat16(h_val);

        // E19-B: gate = silu(h + b_gate) - h only!
        float gate_input = h_val + bg_val;
        float sigmoid_g = 1.0f / (1.0f + __expf(-gate_input));
        float silu_g = gate_input * sigmoid_g;

        output[idx] = __float2bfloat16(h_val * silu_g);
    }
}

template<typename T>
__global__ void FusedTanhGateHOnly(
    const int batch_size,
    const int dim,
    const T* __restrict__ Wx,
    const T* __restrict__ Rh,
    const T* __restrict__ b,
    const T* __restrict__ b_gate,
    T* __restrict__ h_out,
    T* __restrict__ output,
    T* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float wx_val = static_cast<float>(Wx[idx]);
        float rh_val = static_cast<float>(Rh[idx]);
        float val = wx_val + rh_val + static_cast<float>(b[d]);
        if (v_cache) v_cache[idx] = static_cast<T>(val);
        float h_val = tanhf(val);
        h_out[idx] = static_cast<T>(h_val);

        float gate_input = h_val + static_cast<float>(b_gate[d]);
        float sigmoid_g = 1.0f / (1.0f + expf(-gate_input));
        output[idx] = static_cast<T>(h_val * gate_input * sigmoid_g);
    }
}

// =============================================================================
// E19-D Forward: h = tanh(Wx + Rh + h_prev + b), gate = silu(z + h)
// =============================================================================

__global__ void FusedTanhResidualGate_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ Wx,
    const __nv_bfloat16* __restrict__ Rh,
    const __nv_bfloat16* __restrict__ h_prev,
    const __nv_bfloat16* __restrict__ b,
    const __nv_bfloat16* __restrict__ z,
    __nv_bfloat16* __restrict__ h_out,
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float wx_val = __bfloat162float(Wx[idx]);
        float rh_val = __bfloat162float(Rh[idx]);
        float hp_val = __bfloat162float(h_prev[idx]);
        float b_val = __bfloat162float(b[d]);

        // E19-D: h = tanh(Wx + Rh + h_prev + b) - RESIDUAL!
        float val = wx_val + rh_val + hp_val + b_val;
        if (v_cache) v_cache[idx] = __float2bfloat16(val);
        float h_val = tanhf(val);
        h_out[idx] = __float2bfloat16(h_val);

        // E18-A style gate: silu(z + h)
        float z_val = __bfloat162float(z[idx]);
        float gate_input = z_val + h_val;
        float sigmoid_g = 1.0f / (1.0f + __expf(-gate_input));
        float silu_g = gate_input * sigmoid_g;

        output[idx] = __float2bfloat16(h_val * silu_g);
    }
}

template<typename T>
__global__ void FusedTanhResidualGate(
    const int batch_size,
    const int dim,
    const T* __restrict__ Wx,
    const T* __restrict__ Rh,
    const T* __restrict__ h_prev,
    const T* __restrict__ b,
    const T* __restrict__ z,
    T* __restrict__ h_out,
    T* __restrict__ output,
    T* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float wx_val = static_cast<float>(Wx[idx]);
        float rh_val = static_cast<float>(Rh[idx]);
        float hp_val = static_cast<float>(h_prev[idx]);

        float val = wx_val + rh_val + hp_val + static_cast<float>(b[d]);
        if (v_cache) v_cache[idx] = static_cast<T>(val);
        float h_val = tanhf(val);
        h_out[idx] = static_cast<T>(h_val);

        float z_val = static_cast<float>(z[idx]);
        float gate_input = z_val + h_val;
        float sigmoid_g = 1.0f / (1.0f + expf(-gate_input));
        output[idx] = static_cast<T>(h_val * gate_input * sigmoid_g);
    }
}

// =============================================================================
// E19-E Forward: h = tanh(Wx + Rh + h_prev + b), gate = silu(Wx + h + b_gate)
// =============================================================================

__global__ void FusedTanhResidualGateWxPlusH_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ Wx,
    const __nv_bfloat16* __restrict__ Rh,
    const __nv_bfloat16* __restrict__ h_prev,
    const __nv_bfloat16* __restrict__ b,
    const __nv_bfloat16* __restrict__ b_gate,
    __nv_bfloat16* __restrict__ h_out,
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float wx_val = __bfloat162float(Wx[idx]);
        float rh_val = __bfloat162float(Rh[idx]);
        float hp_val = __bfloat162float(h_prev[idx]);
        float b_val = __bfloat162float(b[d]);
        float bg_val = __bfloat162float(b_gate[d]);

        // E19-E: h = tanh(Wx + Rh + h_prev + b) - RESIDUAL!
        float val = wx_val + rh_val + hp_val + b_val;
        if (v_cache) v_cache[idx] = __float2bfloat16(val);
        float h_val = tanhf(val);
        h_out[idx] = __float2bfloat16(h_val);

        // E19-E: gate = silu(Wx + h + b_gate) - reuse Wx!
        float gate_input = wx_val + h_val + bg_val;
        float sigmoid_g = 1.0f / (1.0f + __expf(-gate_input));
        float silu_g = gate_input * sigmoid_g;

        output[idx] = __float2bfloat16(h_val * silu_g);
    }
}

template<typename T>
__global__ void FusedTanhResidualGateWxPlusH(
    const int batch_size,
    const int dim,
    const T* __restrict__ Wx,
    const T* __restrict__ Rh,
    const T* __restrict__ h_prev,
    const T* __restrict__ b,
    const T* __restrict__ b_gate,
    T* __restrict__ h_out,
    T* __restrict__ output,
    T* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float wx_val = static_cast<float>(Wx[idx]);
        float rh_val = static_cast<float>(Rh[idx]);
        float hp_val = static_cast<float>(h_prev[idx]);

        float val = wx_val + rh_val + hp_val + static_cast<float>(b[d]);
        if (v_cache) v_cache[idx] = static_cast<T>(val);
        float h_val = tanhf(val);
        h_out[idx] = static_cast<T>(h_val);

        float gate_input = wx_val + h_val + static_cast<float>(b_gate[d]);
        float sigmoid_g = 1.0f / (1.0f + expf(-gate_input));
        output[idx] = static_cast<T>(h_val * gate_input * sigmoid_g);
    }
}

// =============================================================================
// E19-A Backward: gate = silu(Wx + h + b_gate)
// Gradient paths: dh from output multiply AND from gate
// dWx from dv AND from d_gate (since Wx appears in both h and gate)
// =============================================================================

__global__ void GateBackwardWxPlusH_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ h,
    const __nv_bfloat16* __restrict__ Wx,
    const __nv_bfloat16* __restrict__ b_gate,
    const __nv_bfloat16* __restrict__ d_output,
    __nv_bfloat16* __restrict__ dh,
    __nv_bfloat16* __restrict__ dWx_gate,  // gradient to Wx from gate
    float* __restrict__ db_gate) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float h_val = __bfloat162float(h[idx]);
        float wx_val = __bfloat162float(Wx[idx]);
        float bg_val = __bfloat162float(b_gate[d]);
        float dout = __bfloat162float(d_output[idx]);

        // Forward: gate_input = Wx + h + b_gate, gate = silu(gate_input), output = h * gate
        float gate_input = wx_val + h_val + bg_val;
        float sigmoid_g = 1.0f / (1.0f + __expf(-gate_input));
        float silu_g = gate_input * sigmoid_g;
        float dsilu = sigmoid_g * (1.0f + gate_input * (1.0f - sigmoid_g));

        // d_output/d_h = gate (from h * gate) + h * dsilu * 1 (h appears in gate_input)
        float dh_val = dout * silu_g + dout * h_val * dsilu;

        // d_output/d_Wx = h * dsilu (Wx appears in gate_input)
        float dWx_val = dout * h_val * dsilu;

        // d_output/d_b_gate = h * dsilu
        float dbg_val = dout * h_val * dsilu;

        dh[idx] = __float2bfloat16(dh_val);
        dWx_gate[idx] = __float2bfloat16(dWx_val);
        atomicAdd(&db_gate[d], dbg_val);
    }
}

template<typename T>
__global__ void GateBackwardWxPlusH(
    const int batch_size,
    const int dim,
    const T* __restrict__ h,
    const T* __restrict__ Wx,
    const T* __restrict__ b_gate,
    const T* __restrict__ d_output,
    T* __restrict__ dh,
    T* __restrict__ dWx_gate,
    float* __restrict__ db_gate) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float h_val = static_cast<float>(h[idx]);
        float wx_val = static_cast<float>(Wx[idx]);
        float bg_val = static_cast<float>(b_gate[d]);
        float dout = static_cast<float>(d_output[idx]);

        float gate_input = wx_val + h_val + bg_val;
        float sigmoid_g = 1.0f / (1.0f + expf(-gate_input));
        float silu_g = gate_input * sigmoid_g;
        float dsilu = sigmoid_g * (1.0f + gate_input * (1.0f - sigmoid_g));

        dh[idx] = static_cast<T>(dout * silu_g + dout * h_val * dsilu);
        dWx_gate[idx] = static_cast<T>(dout * h_val * dsilu);
        atomicAdd(&db_gate[d], dout * h_val * dsilu);
    }
}

// =============================================================================
// E19-B Backward: gate = silu(h + b_gate)
// =============================================================================

__global__ void GateBackwardHOnly_BF16(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ h,
    const __nv_bfloat16* __restrict__ b_gate,
    const __nv_bfloat16* __restrict__ d_output,
    __nv_bfloat16* __restrict__ dh,
    float* __restrict__ db_gate) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float h_val = __bfloat162float(h[idx]);
        float bg_val = __bfloat162float(b_gate[d]);
        float dout = __bfloat162float(d_output[idx]);

        // Forward: gate_input = h + b_gate, gate = silu(gate_input), output = h * gate
        float gate_input = h_val + bg_val;
        float sigmoid_g = 1.0f / (1.0f + __expf(-gate_input));
        float silu_g = gate_input * sigmoid_g;
        float dsilu = sigmoid_g * (1.0f + gate_input * (1.0f - sigmoid_g));

        // d_output/d_h = gate (from h * gate) + h * dsilu * 1 (h appears in gate_input)
        float dh_val = dout * silu_g + dout * h_val * dsilu;

        dh[idx] = __float2bfloat16(dh_val);
        atomicAdd(&db_gate[d], dout * h_val * dsilu);
    }
}

template<typename T>
__global__ void GateBackwardHOnly(
    const int batch_size,
    const int dim,
    const T* __restrict__ h,
    const T* __restrict__ b_gate,
    const T* __restrict__ d_output,
    T* __restrict__ dh,
    float* __restrict__ db_gate) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float h_val = static_cast<float>(h[idx]);
        float bg_val = static_cast<float>(b_gate[d]);
        float dout = static_cast<float>(d_output[idx]);

        float gate_input = h_val + bg_val;
        float sigmoid_g = 1.0f / (1.0f + expf(-gate_input));
        float silu_g = gate_input * sigmoid_g;
        float dsilu = sigmoid_g * (1.0f + gate_input * (1.0f - sigmoid_g));

        dh[idx] = static_cast<T>(dout * silu_g + dout * h_val * dsilu);
        atomicAdd(&db_gate[d], dout * h_val * dsilu);
    }
}

// =============================================================================
// E19-D/E Backward through gate with h-awareness (same as E18-A)
// =============================================================================

__global__ void GateBackwardZPlusH_BF16(
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

        float gate_input = z_val + h_val;
        float sigmoid_g = 1.0f / (1.0f + __expf(-gate_input));
        float silu_g = gate_input * sigmoid_g;
        float dsilu = sigmoid_g * (1.0f + gate_input * (1.0f - sigmoid_g));

        dh[idx] = __float2bfloat16(dout * silu_g + dout * h_val * dsilu);
        dz[idx] = __float2bfloat16(dout * h_val * dsilu);
    }
}

template<typename T>
__global__ void GateBackwardZPlusH(
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

        dh[idx] = static_cast<T>(dout * silu_g + dout * h_val * dsilu);
        dz[idx] = static_cast<T>(dout * h_val * dsilu);
    }
}

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
// E19 Forward - BF16 Specialization
// gate_mode: 0=A (Wx+h), 1=B (h only), 2=D (residual+z+h), 3=E (residual+Wx+h)
// =============================================================================

template<>
SimplifiedGateElmanForward<__nv_bfloat16>::SimplifiedGateElmanForward(
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
void SimplifiedGateElmanForward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_x,
    const __nv_bfloat16* W_h,
    const __nv_bfloat16* b,
    const __nv_bfloat16* b_gate,
    const __nv_bfloat16* x,
    const __nv_bfloat16* z,
    __nv_bfloat16* h,
    __nv_bfloat16* output,
    __nv_bfloat16* v,
    __nv_bfloat16* Wx_cache,
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

    // Cache Wx for backward (needed for E19-A/E)
    if (Wx_cache && training_ && (gate_mode_ == 0 || gate_mode_ == 3)) {
        cudaMemcpyAsync(Wx_cache, tmp_Wx, steps * BD * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToDevice, stream_);
    }

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

        // Select kernel based on gate_mode
        if (gate_mode_ == 0) {
            // E19-A: gate = silu(Wx + h + b_gate)
            FusedTanhGateWxPlusH_BF16<<<num_blocks, block_size, 0, stream_>>>(
                batch_size_, dim_, Wx_t, tmp_Rh, b, b_gate, h_t, out_t, v_t);
        } else if (gate_mode_ == 1) {
            // E19-B: gate = silu(h + b_gate)
            FusedTanhGateHOnly_BF16<<<num_blocks, block_size, 0, stream_>>>(
                batch_size_, dim_, Wx_t, tmp_Rh, b, b_gate, h_t, out_t, v_t);
        } else if (gate_mode_ == 2) {
            // E19-D: residual h, gate = silu(z + h)
            FusedTanhResidualGate_BF16<<<num_blocks, block_size, 0, stream_>>>(
                batch_size_, dim_, Wx_t, tmp_Rh, h_prev, b, z_t, h_t, out_t, v_t);
        } else {
            // E19-E: residual h, gate = silu(Wx + h + b_gate)
            FusedTanhResidualGateWxPlusH_BF16<<<num_blocks, block_size, 0, stream_>>>(
                batch_size_, dim_, Wx_t, tmp_Rh, h_prev, b, b_gate, h_t, out_t, v_t);
        }
    }
}

// =============================================================================
// E19 Backward - BF16 Specialization
// =============================================================================

template<>
SimplifiedGateElmanBackward<__nv_bfloat16>::SimplifiedGateElmanBackward(
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
void SimplifiedGateElmanBackward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_x,
    const __nv_bfloat16* W_h,
    const __nv_bfloat16* b_gate,
    const __nv_bfloat16* x,
    const __nv_bfloat16* z,
    const __nv_bfloat16* h,
    const __nv_bfloat16* v,
    const __nv_bfloat16* Wx_cache,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* dx,
    __nv_bfloat16* dz,
    __nv_bfloat16* dW_x,
    __nv_bfloat16* dW_h,
    __nv_bfloat16* db,
    __nv_bfloat16* db_gate,
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
    __nv_bfloat16* dWx_gate_all = workspace + (steps + 2) * BD;  // For E19-A/E
    float* db_float = reinterpret_cast<float*>(workspace + (2 * steps + 2) * BD);
    float* dbg_float = db_float + dim_;

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dbg_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW_x, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_h, 0, dim_ * dim_ * sizeof(__nv_bfloat16), stream_);

    // dz is only used for mode 2 (E19-D), nullptr otherwise
    // Do NOT memset nullptr!

    for (int t = steps - 1; t >= 0; --t) {
        const __nv_bfloat16* v_t = v + t * BD;
        const __nv_bfloat16* h_t = h + (t + 1) * BD;
        const __nv_bfloat16* z_t = (gate_mode_ == 2) ? (z + t * BD) : nullptr;
        const __nv_bfloat16* d_out_t = d_output + t * BD;
        __nv_bfloat16* dv_t = dv_all + t * BD;
        __nv_bfloat16* dz_t = (gate_mode_ == 2 && dz) ? (dz + t * BD) : nullptr;
        __nv_bfloat16* dWx_gate_t = dWx_gate_all + t * BD;

        // Backward through gate based on mode
        if (gate_mode_ == 0) {
            // E19-A: gate = silu(Wx + h + b_gate)
            const __nv_bfloat16* Wx_t = Wx_cache + t * BD;
            GateBackwardWxPlusH_BF16<<<num_blocks, block_size, 0, stream_>>>(
                batch_size_, dim_, h_t, Wx_t, b_gate, d_out_t, dh, dWx_gate_t, dbg_float);
        } else if (gate_mode_ == 1) {
            // E19-B: gate = silu(h + b_gate)
            GateBackwardHOnly_BF16<<<num_blocks, block_size, 0, stream_>>>(
                batch_size_, dim_, h_t, b_gate, d_out_t, dh, dbg_float);
        } else if (gate_mode_ == 2) {
            // E19-D: gate = silu(z + h)
            GateBackwardZPlusH_BF16<<<num_blocks, block_size, 0, stream_>>>(
                batch_size_, dim_, h_t, z_t, d_out_t, dh, dz_t);
        } else {
            // E19-E: gate = silu(Wx + h + b_gate)
            const __nv_bfloat16* Wx_t = Wx_cache + t * BD;
            GateBackwardWxPlusH_BF16<<<num_blocks, block_size, 0, stream_>>>(
                batch_size_, dim_, h_t, Wx_t, b_gate, d_out_t, dh, dWx_gate_t, dbg_float);
        }

        // Add recurrent gradient
        VectorAddInplace_BF16<<<num_blocks, block_size, 0, stream_>>>(BD, dh, dh_recurrent);

        // Backward through tanh
        TanhBackwardKernel_BF16<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, v_t, dh, nullptr, dv_t, db_float);

        // For E19-D/E with residual: dh_recurrent also gets dv directly
        bool has_residual = (gate_mode_ == 2 || gate_mode_ == 3);

        // dh_recurrent = W_h @ dv (+ dv for residual)
        if (t > 0) {
            if (has_residual) {
                // Initialize dh_recurrent = dv_t, then add W_h @ dv_t
                cudaMemcpyAsync(dh_recurrent, dv_t, BD * sizeof(__nv_bfloat16),
                               cudaMemcpyDeviceToDevice, stream_);
                blas<__nv_bfloat16>::gemm(
                    blas_handle_,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    dim_, batch_size_, dim_,
                    &alpha_one,
                    W_h, dim_,
                    dv_t, dim_,
                    &beta_one,  // Add to existing (residual)
                    dh_recurrent, dim_);
            } else {
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

    // dW_x from recurrence
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        dv_all, dim_,
        &beta_one,
        dW_x, dim_);

    // For E19-A/E, also add gate gradient to dW_x
    if (gate_mode_ == 0 || gate_mode_ == 3) {
        blas<__nv_bfloat16>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, steps * batch_size_,
            &alpha_one,
            x, dim_,
            dWx_gate_all, dim_,
            &beta_one,
            dW_x, dim_);
    }

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
    // Mode 2 doesn't use db_gate (it's nullptr)
    if (gate_mode_ != 2 && db_gate) {
        CopyFloatToT<__nv_bfloat16><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, dbg_float, db_gate);
    }
}

// =============================================================================
// Generic Template Implementations
// =============================================================================

template<typename T>
SimplifiedGateElmanForward<T>::SimplifiedGateElmanForward(
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
void SimplifiedGateElmanForward<T>::Run(
    int steps,
    const T* W_x,
    const T* W_h,
    const T* b,
    const T* b_gate,
    const T* x,
    const T* z,
    T* h,
    T* output,
    T* v,
    T* Wx_cache,
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

    if (Wx_cache && training_ && (gate_mode_ == 0 || gate_mode_ == 3)) {
        cudaMemcpyAsync(Wx_cache, tmp_Wx, steps * BD * sizeof(T),
                       cudaMemcpyDeviceToDevice, stream_);
    }

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

        if (gate_mode_ == 0) {
            FusedTanhGateWxPlusH<T><<<num_blocks, block_size, 0, stream_>>>(
                batch_size_, dim_, Wx_t, tmp_Rh, b, b_gate, h_t, out_t, v_t);
        } else if (gate_mode_ == 1) {
            FusedTanhGateHOnly<T><<<num_blocks, block_size, 0, stream_>>>(
                batch_size_, dim_, Wx_t, tmp_Rh, b, b_gate, h_t, out_t, v_t);
        } else if (gate_mode_ == 2) {
            FusedTanhResidualGate<T><<<num_blocks, block_size, 0, stream_>>>(
                batch_size_, dim_, Wx_t, tmp_Rh, h_prev, b, z_t, h_t, out_t, v_t);
        } else {
            FusedTanhResidualGateWxPlusH<T><<<num_blocks, block_size, 0, stream_>>>(
                batch_size_, dim_, Wx_t, tmp_Rh, h_prev, b, b_gate, h_t, out_t, v_t);
        }
    }
}

template<typename T>
SimplifiedGateElmanBackward<T>::SimplifiedGateElmanBackward(
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
void SimplifiedGateElmanBackward<T>::Run(
    int steps,
    const T* W_x,
    const T* W_h,
    const T* b_gate,
    const T* x,
    const T* z,
    const T* h,
    const T* v,
    const T* Wx_cache,
    const T* d_output,
    T* dx,
    T* dz,
    T* dW_x,
    T* dW_h,
    T* db,
    T* db_gate,
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
    T* dWx_gate_all = workspace + (steps + 2) * BD;
    float* db_float = reinterpret_cast<float*>(workspace + (2 * steps + 2) * BD);
    float* dbg_float = db_float + dim_;

    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(T), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dbg_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW_x, 0, dim_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dW_h, 0, dim_ * dim_ * sizeof(T), stream_);

    // dz is only used for mode 2 (E19-D), nullptr otherwise
    // Do NOT memset nullptr!

    for (int t = steps - 1; t >= 0; --t) {
        const T* v_t = v + t * BD;
        const T* h_t = h + (t + 1) * BD;
        const T* z_t = (gate_mode_ == 2) ? (z + t * BD) : nullptr;
        const T* d_out_t = d_output + t * BD;
        T* dv_t = dv_all + t * BD;
        T* dz_t = (gate_mode_ == 2 && dz) ? (dz + t * BD) : nullptr;
        T* dWx_gate_t = dWx_gate_all + t * BD;

        if (gate_mode_ == 0) {
            const T* Wx_t = Wx_cache + t * BD;
            GateBackwardWxPlusH<T><<<num_blocks, block_size, 0, stream_>>>(
                batch_size_, dim_, h_t, Wx_t, b_gate, d_out_t, dh, dWx_gate_t, dbg_float);
        } else if (gate_mode_ == 1) {
            GateBackwardHOnly<T><<<num_blocks, block_size, 0, stream_>>>(
                batch_size_, dim_, h_t, b_gate, d_out_t, dh, dbg_float);
        } else if (gate_mode_ == 2) {
            GateBackwardZPlusH<T><<<num_blocks, block_size, 0, stream_>>>(
                batch_size_, dim_, h_t, z_t, d_out_t, dh, dz_t);
        } else {
            const T* Wx_t = Wx_cache + t * BD;
            GateBackwardWxPlusH<T><<<num_blocks, block_size, 0, stream_>>>(
                batch_size_, dim_, h_t, Wx_t, b_gate, d_out_t, dh, dWx_gate_t, dbg_float);
        }

        VectorAddInplace<T><<<num_blocks, block_size, 0, stream_>>>(BD, dh, dh_recurrent);

        TanhBackwardKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, v_t, dh, nullptr, dv_t, db_float);

        bool has_residual = (gate_mode_ == 2 || gate_mode_ == 3);

        if (t > 0) {
            if (has_residual) {
                cudaMemcpyAsync(dh_recurrent, dv_t, BD * sizeof(T), cudaMemcpyDeviceToDevice, stream_);
                blas<T>::gemm(
                    blas_handle_,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    dim_, batch_size_, dim_,
                    &alpha,
                    W_h, dim_,
                    dv_t, dim_,
                    &beta_one,
                    dh_recurrent, dim_);
            } else {
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

    if (gate_mode_ == 0 || gate_mode_ == 3) {
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, steps * batch_size_,
            &alpha,
            x, dim_,
            dWx_gate_all, dim_,
            &beta_one,
            dW_x, dim_);
    }

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
    // Mode 2 doesn't use db_gate (it's nullptr)
    if (gate_mode_ != 2 && db_gate) {
        CopyFloatToT<T><<<(dim_ + 255) / 256, 256, 0, stream_>>>(dim_, dbg_float, db_gate);
    }
}

// Explicit template instantiations
template struct SimplifiedGateElmanForward<__half>;
template struct SimplifiedGateElmanForward<float>;
template struct SimplifiedGateElmanForward<double>;

template struct SimplifiedGateElmanBackward<__half>;
template struct SimplifiedGateElmanBackward<float>;
template struct SimplifiedGateElmanBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
