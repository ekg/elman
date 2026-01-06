// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E6: Circulant FFT Elman - O(n log n) hidden state updates via FFT
//
// h_t = tanh(circ(c_h) @ h_{t-1} + circ(c_x) @ x_t + b)
// output_t = h_t * silu(W_gate @ x_t + b_gate)
//
// Circulant matrix-vector multiply via FFT:
// circ(c) @ v = IFFT(FFT(c) * FFT(v))
//
// This gives an effective n×n matrix using only n parameters per circulant.
// Complexity: O(n log n) vs O(n²) for dense matmul

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <algorithm>

#include "hasty/elman_ladder.h"
#include "blas.h"
#include "inline_ops.h"

namespace {

// =============================================================================
// Complex arithmetic helpers
// =============================================================================

__device__ __forceinline__ cufftComplex complex_mul(cufftComplex a, cufftComplex b) {
    return make_cuFloatComplex(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

__device__ __forceinline__ cufftComplex complex_add(cufftComplex a, cufftComplex b) {
    return make_cuFloatComplex(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ cufftComplex complex_conj(cufftComplex a) {
    return make_cuFloatComplex(a.x, -a.y);
}

__device__ __forceinline__ cufftComplex complex_scale(cufftComplex a, float s) {
    return make_cuFloatComplex(a.x * s, a.y * s);
}

// =============================================================================
// Kernels for circulant operations
// =============================================================================

// Convert real vector to complex (with zero imaginary part)
template<typename T>
__global__ void RealToComplex(
    const int n,
    const T* __restrict__ real_in,
    cufftComplex* __restrict__ complex_out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        complex_out[idx] = make_cuFloatComplex(static_cast<float>(real_in[idx]), 0.0f);
    }
}

// Batched real to complex: [B, dim] -> [B, dim] complex
template<typename T>
__global__ void BatchedRealToComplex(
    const int batch_size,
    const int dim,
    const T* __restrict__ real_in,
    cufftComplex* __restrict__ complex_out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;
    if (idx < total) {
        complex_out[idx] = make_cuFloatComplex(static_cast<float>(real_in[idx]), 0.0f);
    }
}

// Pointwise complex multiply: out = a * b (element-wise)
__global__ void PointwiseComplexMul(
    const int n,
    const cufftComplex* __restrict__ a,
    const cufftComplex* __restrict__ b,
    cufftComplex* __restrict__ out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = complex_mul(a[idx], b[idx]);
    }
}

// Batched pointwise complex multiply with broadcast:
// FFT(c) is [dim], FFT(v) is [B, dim], output is [B, dim]
__global__ void BatchedPointwiseComplexMulBroadcast(
    const int batch_size,
    const int dim,
    const cufftComplex* __restrict__ fft_c,   // [dim] - broadcast
    const cufftComplex* __restrict__ fft_v,   // [B, dim]
    cufftComplex* __restrict__ out) {         // [B, dim]
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;
    if (idx < total) {
        const int d = idx % dim;
        out[idx] = complex_mul(fft_c[d], fft_v[idx]);
    }
}

// Batched complex add: a += b
__global__ void BatchedComplexAdd(
    const int n,
    cufftComplex* __restrict__ a,
    const cufftComplex* __restrict__ b) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = complex_add(a[idx], b[idx]);
    }
}

// Extract real part, scale by 1/dim (FFT normalization), add bias, apply tanh
template<typename T>
__global__ void ComplexToRealTanhBias(
    const int batch_size,
    const int dim,
    const cufftComplex* __restrict__ complex_in,  // [B, dim]
    const T* __restrict__ b,                      // [dim]
    T* __restrict__ h_out,                        // [B, dim]
    T* __restrict__ v_cache) {                    // [B, dim] pre-activation cache
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;
    if (idx < total) {
        const int d = idx % dim;
        // FFT normalization: divide by dim
        float val = complex_in[idx].x / static_cast<float>(dim) + static_cast<float>(b[d]);
        if (v_cache) v_cache[idx] = static_cast<T>(val);
        h_out[idx] = static_cast<T>(tanhf(val));
    }
}

// Selective output: output = h * silu(gate_proj + b_gate)
// gate_proj is pre-computed W_gate @ x for all timesteps
template<typename T>
__global__ void SelectiveOutputForward(
    const int batch_size,
    const int dim,
    const T* __restrict__ h,            // [B, dim]
    const T* __restrict__ gate_proj,    // [B, dim] pre-computed W_gate @ x
    const T* __restrict__ b_gate,       // [dim]
    T* __restrict__ output,             // [B, dim]
    T* __restrict__ gate_cache) {       // [B, dim] gate_raw for backward
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;
    if (idx < total) {
        const int d = idx % dim;
        float h_val = static_cast<float>(h[idx]);
        float gp_val = static_cast<float>(gate_proj[idx]);
        float b_val = static_cast<float>(b_gate[d]);
        float gate_raw = gp_val + b_val;
        float sigmoid_val = 1.0f / (1.0f + expf(-gate_raw));
        float silu_val = gate_raw * sigmoid_val;
        output[idx] = static_cast<T>(h_val * silu_val);
        if (gate_cache) gate_cache[idx] = static_cast<T>(gate_raw);
    }
}

// =============================================================================
// Backward kernels
// =============================================================================

// Backward through selective output: output = h * silu(gate_proj + b_gate)
template<typename T>
__global__ void SelectiveOutputBackward(
    const int batch_size,
    const int dim,
    const T* __restrict__ h,
    const T* __restrict__ gate_cache,
    const T* __restrict__ d_output,
    T* __restrict__ dh,
    T* __restrict__ d_gate_proj,
    float* __restrict__ d_b_gate) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;
    if (idx < total) {
        const int d = idx % dim;
        float h_val = static_cast<float>(h[idx]);
        float gate_raw = static_cast<float>(gate_cache[idx]);
        float dout = static_cast<float>(d_output[idx]);
        float sigmoid_val = 1.0f / (1.0f + expf(-gate_raw));
        float silu_val = gate_raw * sigmoid_val;
        float dsilu = sigmoid_val * (1.0f + gate_raw * (1.0f - sigmoid_val));
        float dh_val = dout * silu_val;
        float dg_val = dout * h_val * dsilu;
        dh[idx] = static_cast<T>(dh_val);
        d_gate_proj[idx] = static_cast<T>(dg_val);
        atomicAdd(&d_b_gate[d], dg_val);
    }
}

// Backward through tanh: dv = dh * (1 - tanh(v)^2)
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

// Prepare dv for circulant backward: convert to complex and FFT
// For circulant matrix: d_c = sum_batch conj(FFT(h_prev)) * FFT(dv)
// d_h_prev needs IFFT(conj(FFT(c)) * FFT(dv))
template<typename T>
__global__ void PrepareCirculantBackward(
    const int batch_size,
    const int dim,
    const T* __restrict__ dv,
    cufftComplex* __restrict__ dv_complex) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;
    if (idx < total) {
        dv_complex[idx] = make_cuFloatComplex(static_cast<float>(dv[idx]), 0.0f);
    }
}

// Accumulate gradient for circulant vector: d_c += conj(FFT(h_prev)) * FFT(dv)
// Sum over batch dimension
__global__ void AccumulateCirculantGrad(
    const int batch_size,
    const int dim,
    const cufftComplex* __restrict__ fft_h_prev,  // [B, dim]
    const cufftComplex* __restrict__ fft_dv,      // [B, dim]
    cufftComplex* __restrict__ d_fft_c) {         // [dim] accumulated
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d < dim) {
        cufftComplex sum = make_cuFloatComplex(0.0f, 0.0f);
        for (int b = 0; b < batch_size; ++b) {
            int idx = b * dim + d;
            cufftComplex conj_h = complex_conj(fft_h_prev[idx]);
            sum = complex_add(sum, complex_mul(conj_h, fft_dv[idx]));
        }
        d_fft_c[d] = complex_add(d_fft_c[d], sum);
    }
}

// Compute dh_prev in frequency domain: conj(FFT(c)) * FFT(dv)
__global__ void CirculantBackwardH(
    const int batch_size,
    const int dim,
    const cufftComplex* __restrict__ fft_c,       // [dim]
    const cufftComplex* __restrict__ fft_dv,      // [B, dim]
    cufftComplex* __restrict__ dh_prev_complex) { // [B, dim]
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;
    if (idx < total) {
        const int d = idx % dim;
        cufftComplex conj_c = complex_conj(fft_c[d]);
        dh_prev_complex[idx] = complex_mul(conj_c, fft_dv[idx]);
    }
}

// Extract real part of complex after IFFT (with 1/dim normalization)
template<typename T>
__global__ void ComplexToRealNormalize(
    const int n,
    const int dim,
    const cufftComplex* __restrict__ complex_in,
    T* __restrict__ real_out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        real_out[idx] = static_cast<T>(complex_in[idx].x / static_cast<float>(dim));
    }
}

// Convert FFT gradient (in frequency domain) to time domain circulant gradient
// d_c (time domain) = IFFT(d_fft_c) / dim
__global__ void FreqToTimeGrad(
    const int dim,
    const cufftComplex* __restrict__ d_fft_c,
    cufftComplex* __restrict__ d_c_complex) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim) {
        d_c_complex[idx] = d_fft_c[idx];
    }
}

template<typename T>
__global__ void ExtractCirculantGrad(
    const int dim,
    const cufftComplex* __restrict__ d_c_complex,
    T* __restrict__ d_c) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim) {
        // IFFT normalization
        d_c[idx] = static_cast<T>(d_c_complex[idx].x / static_cast<float>(dim));
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
// Circulant FFT Elman Forward
// =============================================================================

template<typename T>
CirculantElmanForward<T>::CirculantElmanForward(
    bool training,
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : training_(training),
      batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream),
      fft_plan_c_(0),
      fft_plan_batch_(0) {

    // Create FFT plan for single dim-length FFT (for c_h, c_x)
    cufftPlan1d(&fft_plan_c_, dim_, CUFFT_C2C, 1);
    cufftSetStream(fft_plan_c_, stream_);

    // Create FFT plan for batched FFT [B, dim]
    cufftPlan1d(&fft_plan_batch_, dim_, CUFFT_C2C, batch_size_);
    cufftSetStream(fft_plan_batch_, stream_);
}

template<typename T>
CirculantElmanForward<T>::~CirculantElmanForward() {
    if (fft_plan_c_) cufftDestroy(fft_plan_c_);
    if (fft_plan_batch_) cufftDestroy(fft_plan_batch_);
}

template<typename T>
void CirculantElmanForward<T>::Run(
    int steps,
    const T* c_h,           // [dim] circulant vector for hidden
    const T* c_x,           // [dim] circulant vector for input
    const T* W_gate,        // [dim, dim] gate projection
    const T* b,             // [dim]
    const T* b_gate,        // [dim]
    const T* x,             // [T, B, dim]
    T* h,                   // [T+1, B, dim] hidden states
    T* output,              // [T, B, dim]
    T* v,                   // [T, B, dim] pre-activation cache
    T* gate_cache,          // [T, B, dim] gate cache for backward
    float* fft_workspace,   // Complex workspace for FFT operations (float32)
    T* gate_proj) {         // [T, B, dim] pre-computed gate projections

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;
    const int num_blocks_dim = (dim_ + block_size - 1) / block_size;

    // FFT workspace layout (all float32/cufftComplex):
    // [fft_c_h: dim complex] [fft_c_x: dim complex]
    // [c_h_complex: dim complex] [c_x_complex: dim complex]
    // [h_complex: BD complex] [x_complex: BD complex] [result_complex: BD complex]
    cufftComplex* fft_c_h = reinterpret_cast<cufftComplex*>(fft_workspace);
    cufftComplex* fft_c_x = fft_c_h + dim_;
    cufftComplex* c_h_complex = fft_c_x + dim_;
    cufftComplex* c_x_complex = c_h_complex + dim_;
    cufftComplex* h_complex = c_x_complex + dim_;
    cufftComplex* x_complex = h_complex + BD;
    cufftComplex* result_complex = x_complex + BD;

    // Pre-compute FFT(c_h) and FFT(c_x)
    RealToComplex<T><<<num_blocks_dim, block_size, 0, stream_>>>(dim_, c_h, c_h_complex);
    RealToComplex<T><<<num_blocks_dim, block_size, 0, stream_>>>(dim_, c_x, c_x_complex);
    cufftExecC2C(fft_plan_c_, c_h_complex, fft_c_h, CUFFT_FORWARD);
    cufftExecC2C(fft_plan_c_, c_x_complex, fft_c_x, CUFFT_FORWARD);

    // Pre-compute gate_proj = x @ W_gate.T for all timesteps
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha,
        W_gate, dim_,
        x, dim_,
        &beta_zero,
        gate_proj, dim_);

    // Process each timestep
    for (int t = 0; t < steps; ++t) {
        const T* x_t = x + t * BD;
        const T* gate_proj_t = gate_proj + t * BD;
        const T* h_prev = h + t * BD;
        T* h_t = h + (t + 1) * BD;
        T* out_t = output + t * BD;
        T* v_t = training_ ? (v + t * BD) : nullptr;
        T* gate_t = training_ ? (gate_cache + t * BD) : nullptr;

        // Convert h_prev and x_t to complex
        BatchedRealToComplex<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_prev, h_complex);
        BatchedRealToComplex<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, x_t, x_complex);

        // FFT(h_prev) and FFT(x_t)
        cufftExecC2C(fft_plan_batch_, h_complex, h_complex, CUFFT_FORWARD);
        cufftExecC2C(fft_plan_batch_, x_complex, x_complex, CUFFT_FORWARD);

        // result = FFT(c_h) * FFT(h_prev)
        BatchedPointwiseComplexMulBroadcast<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, fft_c_h, h_complex, result_complex);

        // tmp = FFT(c_x) * FFT(x_t)
        BatchedPointwiseComplexMulBroadcast<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, fft_c_x, x_complex, x_complex);  // reuse x_complex as temp

        // result += tmp
        BatchedComplexAdd<<<num_blocks, block_size, 0, stream_>>>(BD, result_complex, x_complex);

        // IFFT(result) -> h_t (with bias and tanh)
        cufftExecC2C(fft_plan_batch_, result_complex, result_complex, CUFFT_INVERSE);

        // Extract real part, add bias, apply tanh
        ComplexToRealTanhBias<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, result_complex, b, h_t, v_t);

        // output = h * silu(gate_proj + b_gate)
        SelectiveOutputForward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, gate_proj_t, b_gate, out_t, gate_t);
    }
}

// =============================================================================
// Circulant FFT Elman Backward
// =============================================================================

template<typename T>
CirculantElmanBackward<T>::CirculantElmanBackward(
    int batch_size,
    int dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      blas_handle_(blas_handle),
      stream_(stream),
      fft_plan_c_(0),
      fft_plan_batch_(0) {

    cufftPlan1d(&fft_plan_c_, dim_, CUFFT_C2C, 1);
    cufftSetStream(fft_plan_c_, stream_);

    cufftPlan1d(&fft_plan_batch_, dim_, CUFFT_C2C, batch_size_);
    cufftSetStream(fft_plan_batch_, stream_);
}

template<typename T>
CirculantElmanBackward<T>::~CirculantElmanBackward() {
    if (fft_plan_c_) cufftDestroy(fft_plan_c_);
    if (fft_plan_batch_) cufftDestroy(fft_plan_batch_);
}

template<typename T>
void CirculantElmanBackward<T>::Run(
    int steps,
    const T* c_h,
    const T* c_x,
    const T* W_gate,
    const T* x,
    const T* h,
    const T* v,
    const T* gate_cache,
    const T* d_output,
    T* dx,
    T* d_c_h,
    T* d_c_x,
    T* dW_gate,
    T* db,
    T* d_b_gate,
    float* fft_workspace,
    T* work_T) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int BD = batch_size_ * dim_;
    const int TBD = steps * BD;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;
    const int num_blocks_dim = (dim_ + block_size - 1) / block_size;
    const int num_blocks_TBD = (TBD + block_size - 1) / block_size;

    // FFT workspace layout (all float32/cufftComplex):
    // [fft_c_h: dim complex] [fft_c_x: dim complex]
    // [d_fft_c_h: dim complex] [d_fft_c_x: dim complex]
    // [c_h_complex: dim complex] [c_x_complex: dim complex]
    // [dv_complex: BD complex] [h_prev_complex: BD complex] [x_complex: BD complex]
    // [dh_prev_complex: BD complex]
    // [d_c_h_complex: dim complex] [d_c_x_complex: dim complex]
    // [db_float: dim] [d_b_gate_float: dim]
    cufftComplex* fft_c_h = reinterpret_cast<cufftComplex*>(fft_workspace);
    cufftComplex* fft_c_x = fft_c_h + dim_;
    cufftComplex* d_fft_c_h = fft_c_x + dim_;
    cufftComplex* d_fft_c_x = d_fft_c_h + dim_;
    cufftComplex* c_h_complex = d_fft_c_x + dim_;
    cufftComplex* c_x_complex = c_h_complex + dim_;
    cufftComplex* dv_complex = c_x_complex + dim_;
    cufftComplex* h_prev_complex = dv_complex + BD;
    cufftComplex* x_complex = h_prev_complex + BD;
    cufftComplex* dh_prev_complex = x_complex + BD;
    cufftComplex* d_c_h_complex = dh_prev_complex + BD;
    cufftComplex* d_c_x_complex = d_c_h_complex + dim_;

    // Float accumulators after complex arrays
    float* db_float = reinterpret_cast<float*>(d_c_x_complex + dim_);
    float* d_b_gate_float = db_float + dim_;

    // Model dtype workspace: [dh: BD] [dh_recurrent: BD] [dv_all: T*BD] [d_gate_proj_all: T*BD]
    T* dh = work_T;
    T* dh_recurrent = dh + BD;
    T* dv_all = dh_recurrent + BD;
    T* d_gate_proj_all = dv_all + TBD;

    // Initialize gradients
    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(T), stream_);
    cudaMemsetAsync(d_fft_c_h, 0, dim_ * sizeof(cufftComplex), stream_);
    cudaMemsetAsync(d_fft_c_x, 0, dim_ * sizeof(cufftComplex), stream_);
    cudaMemsetAsync(db_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(d_b_gate_float, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dW_gate, 0, dim_ * dim_ * sizeof(T), stream_);

    // Pre-compute FFT(c_h) and FFT(c_x) for backward
    RealToComplex<T><<<num_blocks_dim, block_size, 0, stream_>>>(dim_, c_h, c_h_complex);
    RealToComplex<T><<<num_blocks_dim, block_size, 0, stream_>>>(dim_, c_x, c_x_complex);
    cufftExecC2C(fft_plan_c_, c_h_complex, fft_c_h, CUFFT_FORWARD);
    cufftExecC2C(fft_plan_c_, c_x_complex, fft_c_x, CUFFT_FORWARD);

    // BPTT loop
    for (int t = steps - 1; t >= 0; --t) {
        const T* v_t = v + t * BD;
        const T* h_t = h + (t + 1) * BD;
        const T* h_prev = h + t * BD;
        const T* x_t = x + t * BD;
        const T* gate_cache_t = gate_cache + t * BD;
        const T* d_out_t = d_output + t * BD;
        T* dv_t = dv_all + t * BD;
        T* d_gate_proj_t = d_gate_proj_all + t * BD;

        // Backward through selective output
        SelectiveOutputBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_t, gate_cache_t, d_out_t,
            dh, d_gate_proj_t, d_b_gate_float);

        // Add recurrent gradient
        VectorAddInplace<T><<<num_blocks, block_size, 0, stream_>>>(BD, dh, dh_recurrent);

        // Backward through tanh
        TanhBackwardKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, v_t, dh, nullptr, dv_t, db_float);

        // Backward through circulant operations
        // dv_complex = FFT(dv)
        PrepareCirculantBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, dv_t, dv_complex);
        cufftExecC2C(fft_plan_batch_, dv_complex, dv_complex, CUFFT_FORWARD);

        // FFT(h_prev) and FFT(x_t) for gradient accumulation
        BatchedRealToComplex<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_prev, h_prev_complex);
        BatchedRealToComplex<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, x_t, x_complex);
        cufftExecC2C(fft_plan_batch_, h_prev_complex, h_prev_complex, CUFFT_FORWARD);
        cufftExecC2C(fft_plan_batch_, x_complex, x_complex, CUFFT_FORWARD);

        // Accumulate d_c_h: d_fft_c_h += sum_batch(conj(FFT(h_prev)) * FFT(dv))
        AccumulateCirculantGrad<<<num_blocks_dim, block_size, 0, stream_>>>(
            batch_size_, dim_, h_prev_complex, dv_complex, d_fft_c_h);

        // Accumulate d_c_x: d_fft_c_x += sum_batch(conj(FFT(x)) * FFT(dv))
        AccumulateCirculantGrad<<<num_blocks_dim, block_size, 0, stream_>>>(
            batch_size_, dim_, x_complex, dv_complex, d_fft_c_x);

        // dh_recurrent = IFFT(conj(FFT(c_h)) * FFT(dv))
        if (t > 0) {
            CirculantBackwardH<<<num_blocks, block_size, 0, stream_>>>(
                batch_size_, dim_, fft_c_h, dv_complex, dh_prev_complex);
            cufftExecC2C(fft_plan_batch_, dh_prev_complex, dh_prev_complex, CUFFT_INVERSE);
            ComplexToRealNormalize<T><<<num_blocks, block_size, 0, stream_>>>(
                BD, dim_, dh_prev_complex, dh_recurrent);
        }
    }

    // Initialize dx to zero
    cudaMemsetAsync(dx, 0, steps * BD * sizeof(T), stream_);

    // dx += circ(c_x)^T @ dv = circ(flip(c_x)) @ dv
    // For backward: we need IFFT(conj(FFT(c_x)) * FFT(dv)) for each timestep
    // This is expensive - batch it
    for (int t = 0; t < steps; ++t) {
        T* dv_t = dv_all + t * BD;
        T* dx_t = dx + t * BD;

        PrepareCirculantBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, dv_t, dv_complex);
        cufftExecC2C(fft_plan_batch_, dv_complex, dv_complex, CUFFT_FORWARD);

        CirculantBackwardH<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, fft_c_x, dv_complex, dh_prev_complex);
        cufftExecC2C(fft_plan_batch_, dh_prev_complex, dh_prev_complex, CUFFT_INVERSE);
        ComplexToRealNormalize<T><<<num_blocks, block_size, 0, stream_>>>(
            BD, dim_, dh_prev_complex, dx_t);
    }

    // dx += W_gate @ d_gate_proj_all
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_,
        &alpha,
        W_gate, dim_,
        d_gate_proj_all, dim_,
        &beta_one,
        dx, dim_);

    // dW_gate = x^T @ d_gate_proj_all
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, dim_, steps * batch_size_,
        &alpha,
        x, dim_,
        d_gate_proj_all, dim_,
        &beta_one,
        dW_gate, dim_);

    // Convert frequency domain gradients to time domain
    // d_c_h = IFFT(d_fft_c_h)
    cufftExecC2C(fft_plan_c_, d_fft_c_h, d_c_h_complex, CUFFT_INVERSE);
    ExtractCirculantGrad<T><<<num_blocks_dim, block_size, 0, stream_>>>(dim_, d_c_h_complex, d_c_h);

    // d_c_x = IFFT(d_fft_c_x)
    cufftExecC2C(fft_plan_c_, d_fft_c_x, d_c_x_complex, CUFFT_INVERSE);
    ExtractCirculantGrad<T><<<num_blocks_dim, block_size, 0, stream_>>>(dim_, d_c_x_complex, d_c_x);

    // Copy float gradients
    CopyFloatToT<T><<<num_blocks_dim, block_size, 0, stream_>>>(dim_, db_float, db);
    CopyFloatToT<T><<<num_blocks_dim, block_size, 0, stream_>>>(dim_, d_b_gate_float, d_b_gate);
}

// Explicit template instantiations
template struct CirculantElmanForward<__half>;
template struct CirculantElmanForward<__nv_bfloat16>;
template struct CirculantElmanForward<float>;
template struct CirculantElmanForward<double>;

template struct CirculantElmanBackward<__half>;
template struct CirculantElmanBackward<__nv_bfloat16>;
template struct CirculantElmanBackward<float>;
template struct CirculantElmanBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
