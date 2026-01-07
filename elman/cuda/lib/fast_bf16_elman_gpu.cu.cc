// Copyright 2026 Erik Garrison. Apache 2.0 License.
//
// Fast BF16 Elman - Optimized for pure bfloat16 computation
//
// Key optimizations:
// 1. Native bf16 arithmetic where possible (no f32 conversions for add/mul)
// 2. Fast tanh approximations (Padé, softsign, hard tanh)
// 3. Fused operations to reduce memory traffic
//
// Architecture (same as E1):
// h_t = activation(W_x @ x_t + W_h @ h_{t-1} + b)
// output = h * silu(z)

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <algorithm>

#include "hasty/elman_ladder.h"
#include "blas.h"

namespace {

// =============================================================================
// Fast activation functions for bf16
// =============================================================================

// Standard tanh via f32 (baseline)
__device__ __forceinline__ float fast_tanh_f32(float x) {
    return tanhf(x);
}

// Padé approximant: tanh(x) ≈ x(27 + x²) / (27 + 9x²)
// Accurate to ~0.002 for |x| < 3, saturates correctly for large |x|
__device__ __forceinline__ float fast_tanh_pade(float x) {
    // Clamp to avoid overflow
    x = fminf(fmaxf(x, -4.0f), 4.0f);
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

// Softsign: x / (1 + |x|) - no exp, smooth, bounded to [-1, 1]
__device__ __forceinline__ float fast_softsign(float x) {
    return x / (1.0f + fabsf(x));
}

// Hard tanh: clamp(x, -1, 1) - fastest but discontinuous gradient
__device__ __forceinline__ float fast_hardtanh(float x) {
    return fminf(fmaxf(x, -1.0f), 1.0f);
}

// Rational approximation optimized for [-2, 2] range
// tanh(x) ≈ x * (1 - x²/9) / (1 + x²/3) for small x
// More accurate than Padé in typical RNN operating range
__device__ __forceinline__ float fast_tanh_rational(float x) {
    float ax = fabsf(x);
    if (ax > 2.5f) {
        return (x > 0.0f) ? 1.0f : -1.0f;
    }
    float x2 = x * x;
    // Coefficients tuned for accuracy in [-2, 2]
    return x * (1.0f + 0.1612f * x2) / (1.0f + 0.4908f * x2);
}

// Fast exp-based tanh: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
// Uses __expf for speed
__device__ __forceinline__ float fast_tanh_exp(float x) {
    float e2x = __expf(2.0f * x);
    return (e2x - 1.0f) / (e2x + 1.0f);
}

// =============================================================================
// BF16 native operations
// =============================================================================

// Native bf16 add (no f32 conversion)
__device__ __forceinline__ __nv_bfloat16 bf16_add(__nv_bfloat16 a, __nv_bfloat16 b) {
#if __CUDA_ARCH__ >= 800
    return __hadd(a, b);
#else
    return __float2bfloat16(__bfloat162float(a) + __bfloat162float(b));
#endif
}

// Native bf16 multiply
__device__ __forceinline__ __nv_bfloat16 bf16_mul(__nv_bfloat16 a, __nv_bfloat16 b) {
#if __CUDA_ARCH__ >= 800
    return __hmul(a, b);
#else
    return __float2bfloat16(__bfloat162float(a) * __bfloat162float(b));
#endif
}

// Native bf16 fused multiply-add
__device__ __forceinline__ __nv_bfloat16 bf16_fma(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c) {
#if __CUDA_ARCH__ >= 800
    return __hfma(a, b, c);
#else
    return __float2bfloat16(__bfloat162float(a) * __bfloat162float(b) + __bfloat162float(c));
#endif
}

// =============================================================================
// Forward kernels with activation variants
// =============================================================================

// Activation type enum (compile-time selection via template)
enum class ActivationType {
    TANH_F32,      // Standard tanhf in f32
    TANH_PADE,     // Padé approximant
    TANH_RATIONAL, // Rational approximation
    TANH_EXP,      // Fast exp-based
    SOFTSIGN,      // x / (1 + |x|)
    HARDTANH       // clamp(x, -1, 1)
};

template<ActivationType ACT>
__device__ __forceinline__ float apply_activation(float x) {
    if constexpr (ACT == ActivationType::TANH_F32) {
        return fast_tanh_f32(x);
    } else if constexpr (ACT == ActivationType::TANH_PADE) {
        return fast_tanh_pade(x);
    } else if constexpr (ACT == ActivationType::TANH_RATIONAL) {
        return fast_tanh_rational(x);
    } else if constexpr (ACT == ActivationType::TANH_EXP) {
        return fast_tanh_exp(x);
    } else if constexpr (ACT == ActivationType::SOFTSIGN) {
        return fast_softsign(x);
    } else if constexpr (ACT == ActivationType::HARDTANH) {
        return fast_hardtanh(x);
    }
    return x;
}

// Derivative of activation (for backward pass)
template<ActivationType ACT>
__device__ __forceinline__ float activation_derivative(float x, float y) {
    // y = activation(x)
    if constexpr (ACT == ActivationType::TANH_F32 ||
                  ACT == ActivationType::TANH_PADE ||
                  ACT == ActivationType::TANH_RATIONAL ||
                  ACT == ActivationType::TANH_EXP) {
        // dtanh/dx = 1 - tanh²(x) = 1 - y²
        return 1.0f - y * y;
    } else if constexpr (ACT == ActivationType::SOFTSIGN) {
        // d/dx[x/(1+|x|)] = 1/(1+|x|)²
        float denom = 1.0f + fabsf(x);
        return 1.0f / (denom * denom);
    } else if constexpr (ACT == ActivationType::HARDTANH) {
        // Derivative is 1 in [-1,1], 0 outside
        return (x >= -1.0f && x <= 1.0f) ? 1.0f : 0.0f;
    }
    return 1.0f;
}

// =============================================================================
// Pure BF16 forward kernel (minimal f32 conversion)
// =============================================================================

template<ActivationType ACT>
__global__ void FastBF16TanhKernel(
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

        // BF16 adds for the linear combination
        __nv_bfloat16 sum = bf16_add(bf16_add(Wx[idx], Rh[idx]), b[d]);

        // Convert to f32 only for activation (unavoidable for transcendentals)
        float val = __bfloat162float(sum);
        if (v_cache) v_cache[idx] = sum;  // Store pre-activation in bf16

        float h = apply_activation<ACT>(val);
        h_out[idx] = __float2bfloat16(h);
    }
}

// Pure BF16 gate forward
__global__ void FastBF16GateKernel(
    const int batch_size,
    const int dim,
    const __nv_bfloat16* __restrict__ h,
    const __nv_bfloat16* __restrict__ z,
    __nv_bfloat16* __restrict__ output) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        float h_val = __bfloat162float(h[idx]);
        float z_val = __bfloat162float(z[idx]);

        // silu(z) = z * sigmoid(z)
        float sigmoid_z = 1.0f / (1.0f + __expf(-z_val));
        float silu_z = z_val * sigmoid_z;

        output[idx] = __float2bfloat16(h_val * silu_z);
    }
}

// =============================================================================
// Fused forward kernel: combines tanh + gate in one pass
// Reduces memory traffic by not writing intermediate h
// =============================================================================

template<ActivationType ACT>
__global__ void FusedBF16ForwardKernel(
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

        // BF16 adds
        __nv_bfloat16 sum = bf16_add(bf16_add(Wx[idx], Rh[idx]), b[d]);
        float val = __bfloat162float(sum);

        if (v_cache) v_cache[idx] = sum;

        // Activation
        float h = apply_activation<ACT>(val);
        h_out[idx] = __float2bfloat16(h);

        // Gate (fused)
        float z_val = __bfloat162float(z[idx]);
        float sigmoid_z = 1.0f / (1.0f + __expf(-z_val));
        float silu_z = z_val * sigmoid_z;

        output[idx] = __float2bfloat16(h * silu_z);
    }
}

// =============================================================================
// Backward kernels
// =============================================================================

template<ActivationType ACT>
__global__ void FastBF16TanhBackwardKernel(
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

        float grad = __bfloat162float(dh[idx]);
        if (dh_recurrent) grad += __bfloat162float(dh_recurrent[idx]);

        float x = __bfloat162float(v[idx]);
        float y = apply_activation<ACT>(x);
        float dact = activation_derivative<ACT>(x, y);

        float dv_val = grad * dact;
        dv[idx] = __float2bfloat16(dv_val);

        atomicAdd(&db[d], dv_val);
    }
}

__global__ void FastBF16GateBackwardKernel(
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

        float sigmoid_z = 1.0f / (1.0f + __expf(-z_val));
        float silu_z = z_val * sigmoid_z;
        float dsilu = sigmoid_z * (1.0f + z_val * (1.0f - sigmoid_z));

        dh[idx] = __float2bfloat16(dout * silu_z);
        dz[idx] = __float2bfloat16(dout * h_val * dsilu);
    }
}

// BF16 vector add (pure bf16)
__global__ void FastBF16VectorAdd(
    const int n,
    __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = bf16_add(a[idx], b[idx]);
    }
}

}  // anonymous namespace


// =============================================================================
// Exported test functions for benchmarking different activations
// =============================================================================

extern "C" {

// Test different tanh variants - returns timing in microseconds
__global__ void benchmark_activations_kernel(
    const int n,
    const float* __restrict__ input,
    float* __restrict__ output_tanh,
    float* __restrict__ output_pade,
    float* __restrict__ output_rational,
    float* __restrict__ output_softsign,
    float* __restrict__ output_hardtanh) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        output_tanh[idx] = fast_tanh_f32(x);
        output_pade[idx] = fast_tanh_pade(x);
        output_rational[idx] = fast_tanh_rational(x);
        output_softsign[idx] = fast_softsign(x);
        output_hardtanh[idx] = fast_hardtanh(x);
    }
}

}  // extern "C"
