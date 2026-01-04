// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// Level 7: Diagonal Triple R (Linear Space) with Learned Gate Projection
//
// Like Level 5 (Triple R) but with diagonal r_h and r_delta instead of
// full matrices. More efficient O(d) recurrence.
//
// Architecture:
//   v = r_h * h_prev + W_x @ x + b              -- diagonal r_h (element-wise)
//   delta_raw = W_delta @ x + r_delta * h_prev + b_delta  -- diagonal r_delta
//   delta = sigmoid(delta_raw)
//   h_new = (1 - delta) * h_prev + delta * tanh(v)
//
//   // Learned gate projection output
//   output = h_new * silu(W_gate @ x + b_gate)  -- learned gate projection!

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cfloat>

#include "hasty/elman_ladder.h"
#include "blas.h"
#include "inline_ops.h"

namespace {

// =============================================================================
// Forward: Diagonal Triple R Update Kernel
// =============================================================================

template<typename T>
__global__ void DiagTripleRUpdateKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ h_prev,       // [B, dim]
    const T* __restrict__ wx_x,         // [B, dim] W_x @ x (precomputed)
    const T* __restrict__ r_h,          // [dim] diagonal recurrence
    const T* __restrict__ wdelta_x,     // [B, dim] W_delta @ x (precomputed)
    const T* __restrict__ r_delta,      // [dim] diagonal delta modulation
    const T* __restrict__ b,            // [dim]
    const T* __restrict__ b_delta,      // [dim]
    T* __restrict__ h_out,              // [B, dim]
    T* __restrict__ v_cache,            // [B, dim] for backward
    T* __restrict__ delta_cache) {      // [B, dim] for backward

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float hp = static_cast<float>(h_prev[idx]);
        float rh = static_cast<float>(r_h[d]);
        float rd = static_cast<float>(r_delta[d]);
        float wx = static_cast<float>(wx_x[idx]);
        float wd = static_cast<float>(wdelta_x[idx]);
        float bias = static_cast<float>(b[d]);
        float bias_delta = static_cast<float>(b_delta[d]);

        // v = r_h * h_prev + W_x @ x + b
        float v = rh * hp + wx + bias;
        float tanh_v = tanhf(v);

        // delta = sigmoid(W_delta @ x + r_delta * h_prev + b_delta)
        float delta_raw = wd + rd * hp + bias_delta;
        float delta = 1.0f / (1.0f + expf(-delta_raw));

        // h_new = (1 - delta) * h_prev + delta * tanh(v)
        float h_new = (1.0f - delta) * hp + delta * tanh_v;

        h_out[idx] = static_cast<T>(h_new);
        if (v_cache) v_cache[idx] = static_cast<T>(v);
        if (delta_cache) delta_cache[idx] = static_cast<T>(delta);
    }
}

// =============================================================================
// Forward: Learned Gate Projection Output
// output = h * silu(W_gate @ x + b_gate)
// =============================================================================

template<typename T>
__global__ void DiagTripleRSelectiveOutput(
    const int batch_size,
    const int dim,
    const T* __restrict__ h,            // [B, dim]
    const T* __restrict__ gate_proj,    // [B, dim] pre-computed W_gate @ x
    const T* __restrict__ b_gate,       // [dim] learned gate bias
    T* __restrict__ output,             // [B, dim]
    T* __restrict__ gate_cache) {       // [B, dim] cache gate_raw for backward

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float h_val = static_cast<float>(h[idx]);
        float gp_val = static_cast<float>(gate_proj[idx]);
        float b_val = static_cast<float>(b_gate[d]);

        // gate_raw = W_gate @ x + b_gate (W_gate @ x pre-computed)
        float gate_raw = gp_val + b_val;

        // silu(gate_raw) = gate_raw * sigmoid(gate_raw)
        float sigmoid_val = 1.0f / (1.0f + expf(-gate_raw));
        float silu_val = gate_raw * sigmoid_val;

        // output = h * silu(W_gate @ x + b_gate)
        output[idx] = static_cast<T>(h_val * silu_val);

        // Cache gate_raw for backward (not silu_val)
        if (gate_cache) gate_cache[idx] = static_cast<T>(gate_raw);
    }
}

// =============================================================================
// Backward: Learned Gate Projection Output
// output = h * silu(W_gate @ x + b_gate)
// =============================================================================

template<typename T>
__global__ void DiagTripleRSelectiveOutputBackward(
    const int batch_size,
    const int dim,
    const T* __restrict__ h,
    const T* __restrict__ gate_cache,    // [B, dim] cached gate_raw values
    const T* __restrict__ d_output,
    T* __restrict__ dh,              // gradient to h
    T* __restrict__ d_gate_proj,     // gradient to W_gate @ x (for W_gate gradient)
    float* __restrict__ d_b_gate) {  // accumulated gradient for b_gate

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float h_val = static_cast<float>(h[idx]);
        float gate_raw = static_cast<float>(gate_cache[idx]);
        float dout = static_cast<float>(d_output[idx]);

        // Forward: gate_raw = W_gate @ x + b_gate
        //          silu = gate_raw * sigmoid(gate_raw)
        //          output = h * silu
        float sigmoid_val = 1.0f / (1.0f + expf(-gate_raw));
        float silu_val = gate_raw * sigmoid_val;

        // d_silu/d_gate_raw = sigmoid + gate_raw * sigmoid * (1 - sigmoid)
        //                   = sigmoid * (1 + gate_raw * (1 - sigmoid))
        float dsilu = sigmoid_val * (1.0f + gate_raw * (1.0f - sigmoid_val));

        // d_output/d_h = silu (h only affects output through multiplication)
        float dh_val = dout * silu_val;

        // d_output/d_gate_proj = h * dsilu
        float d_gp_val = dout * h_val * dsilu;

        // d_output/d_b_gate = h * dsilu (same as d_gate_proj)
        float db_val = dout * h_val * dsilu;

        dh[idx] = static_cast<T>(dh_val);
        d_gate_proj[idx] = static_cast<T>(d_gp_val);
        atomicAdd(&d_b_gate[d], db_val);
    }
}

// =============================================================================
// Backward: Diagonal Triple R Update
// =============================================================================

template<typename T>
__global__ void DiagTripleRUpdateBackward(
    const int batch_size,
    const int dim,
    const T* __restrict__ h_prev,
    const T* __restrict__ v_cache,
    const T* __restrict__ delta_cache,
    const T* __restrict__ r_h,
    const T* __restrict__ r_delta,
    const T* __restrict__ dh,           // gradient from output + recurrent
    T* __restrict__ dh_prev,            // gradient to previous h
    T* __restrict__ d_wx_x,             // gradient for W_x @ x
    T* __restrict__ d_wdelta_x,         // gradient for W_delta @ x
    float* __restrict__ d_r_h,          // accumulated gradient for r_h
    float* __restrict__ d_r_delta,      // accumulated gradient for r_delta
    float* __restrict__ d_b,            // accumulated gradient for b
    float* __restrict__ d_b_delta) {    // accumulated gradient for b_delta

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float hp = static_cast<float>(h_prev[idx]);
        float v = static_cast<float>(v_cache[idx]);
        float delta = static_cast<float>(delta_cache[idx]);
        float rh = static_cast<float>(r_h[d]);
        float rd = static_cast<float>(r_delta[d]);
        float grad_h = static_cast<float>(dh[idx]);

        float tanh_v = tanhf(v);
        float one_minus_delta = 1.0f - delta;

        // Backward through gated update: h_new = (1-delta)*h_prev + delta*tanh(v)
        float d_tanh_v = grad_h * delta;
        float d_hp_decay = grad_h * one_minus_delta;
        float d_delta_val = grad_h * (tanh_v - hp);

        // Backward through tanh
        float tanh_grad = 1.0f - tanh_v * tanh_v;
        float d_v = d_tanh_v * tanh_grad;

        // Backward through delta sigmoid
        float d_delta_raw = d_delta_val * delta * one_minus_delta;

        // Gradients for r_h path: v = r_h * h_prev + wx + b
        float d_hp_from_v = d_v * rh;
        float d_rh_val = d_v * hp;

        // Gradients for r_delta path: delta_raw = wd + r_delta * h_prev + b_delta
        float d_hp_from_delta = d_delta_raw * rd;
        float d_rd_val = d_delta_raw * hp;

        // Total h_prev gradient
        float d_hp_total = d_hp_decay + d_hp_from_v + d_hp_from_delta;
        dh_prev[idx] = static_cast<T>(d_hp_total);

        // Gradients for GEMM inputs
        d_wx_x[idx] = static_cast<T>(d_v);
        d_wdelta_x[idx] = static_cast<T>(d_delta_raw);

        // Accumulate diagonal parameter gradients
        atomicAdd(&d_r_h[d], d_rh_val);
        atomicAdd(&d_r_delta[d], d_rd_val);
        atomicAdd(&d_b[d], d_v);
        atomicAdd(&d_b_delta[d], d_delta_raw);
    }
}

// Kernel: Copy float array to type T
template<typename T>
__global__ void CopyFloatToT(
    const int n,
    const float* __restrict__ src,
    T* __restrict__ dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = static_cast<T>(src[idx]);
    }
}

// Kernel: Vector add (a += b)
template<typename T>
__global__ void VectorAddInplace(
    const int n,
    T* __restrict__ a,
    const T* __restrict__ b) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = static_cast<T>(static_cast<float>(a[idx]) + static_cast<float>(b[idx]));
    }
}

}  // anonymous namespace

namespace hasty {
namespace v0 {
namespace elman_ladder {

// =============================================================================
// DiagTripleRForward Implementation
// =============================================================================

template<typename T>
DiagTripleRForward<T>::DiagTripleRForward(
    bool training,
    int batch_size,
    int dim,
    int n_groups,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : training_(training),
      batch_size_(batch_size),
      dim_(dim),
      n_groups_(n_groups),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void DiagTripleRForward<T>::Run(
    int steps,
    const T* W_x,
    const T* r_h,
    const T* r_delta,
    const T* W_delta,
    const T* W_gate,     // [dim, dim] learned gate projection
    const T* b,
    const T* b_delta,
    const T* b_gate,     // [dim] gate bias
    const T* x,
    T* h,
    T* output,
    T* v_cache,
    T* delta_cache,
    T* gate_cache) {

    const int BD = batch_size_ * dim_;
    const int block_size = 256;

    // BLAS constants
    const T alpha_one = static_cast<T>(1.0f);
    const T beta_zero = static_cast<T>(0.0f);

    // Allocate workspace for pre-computed projections
    T* all_wx_x;
    T* all_wdelta_x;
    T* all_gate_proj;
    cudaMalloc(&all_wx_x, steps * BD * sizeof(T));
    cudaMalloc(&all_wdelta_x, steps * BD * sizeof(T));
    cudaMalloc(&all_gate_proj, steps * BD * sizeof(T));

    // Pre-compute W_x @ x for all timesteps
    blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_, &alpha_one, W_x, dim_, x, dim_, &beta_zero, all_wx_x, dim_);

    // Pre-compute W_delta @ x for all timesteps
    blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_, &alpha_one, W_delta, dim_, x, dim_, &beta_zero, all_wdelta_x, dim_);

    // Pre-compute W_gate @ x for all timesteps (learned gate projection)
    blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, dim_, &alpha_one, W_gate, dim_, x, dim_, &beta_zero, all_gate_proj, dim_);

    for (int t = 0; t < steps; ++t) {
        const T* h_prev = h + t * BD;
        T* h_out = h + (t + 1) * BD;
        T* output_t = output + t * BD;

        const T* wx_x_t = all_wx_x + t * BD;
        const T* wdelta_x_t = all_wdelta_x + t * BD;
        const T* gate_proj_t = all_gate_proj + t * BD;

        T* v_t = training_ ? v_cache + t * BD : nullptr;
        T* delta_t = training_ ? delta_cache + t * BD : nullptr;
        T* gate_t = training_ ? gate_cache + t * BD : nullptr;

        // Update kernel
        const int num_blocks = (BD + block_size - 1) / block_size;
        DiagTripleRUpdateKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_,
            h_prev,
            wx_x_t,
            r_h,
            wdelta_x_t,
            r_delta,
            b,
            b_delta,
            h_out,
            v_t,
            delta_t);

        // Learned gate projection output: output = h * silu(W_gate @ x + b_gate)
        DiagTripleRSelectiveOutput<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_,
            h_out,
            gate_proj_t,
            b_gate,
            output_t,
            gate_t);
    }

    cudaFree(all_wx_x);
    cudaFree(all_wdelta_x);
    cudaFree(all_gate_proj);
}

// =============================================================================
// DiagTripleRBackward Implementation
// =============================================================================

template<typename T>
DiagTripleRBackward<T>::DiagTripleRBackward(
    int batch_size,
    int dim,
    int n_groups,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      n_groups_(n_groups),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void DiagTripleRBackward<T>::Run(
    int steps,
    const T* W_x,
    const T* r_h,
    const T* r_delta,
    const T* W_delta,
    const T* W_gate,     // [dim, dim] learned gate projection
    const T* x,
    const T* h,
    const T* v_cache,
    const T* delta_cache,
    const T* gate_cache,
    const T* d_output,
    T* dx,
    T* dW_x,
    T* d_r_h,
    T* d_r_delta,
    T* dW_delta,
    T* dW_gate,          // [dim, dim] gradient for gate projection
    T* db,
    T* db_delta,
    T* db_gate,          // [dim] gradient for gate bias
    T* workspace) {

    const int BD = batch_size_ * dim_;
    const int block_size = 256;

    // BLAS constants
    const T alpha_one = static_cast<T>(1.0f);
    const T beta_zero = static_cast<T>(0.0f);

    // Workspace layout
    T* dh = workspace;                    // [BD]
    T* d_gate_proj = workspace + BD;      // [BD] gradient for W_gate @ x
    T* dh_recurrent = workspace + 2 * BD; // [BD]
    T* dh_prev = workspace + 3 * BD;      // [BD]
    T* d_wx_x = workspace + 4 * BD;       // [BD]
    T* d_wdelta_x = workspace + 5 * BD;   // [BD]
    float* d_r_h_f = reinterpret_cast<float*>(workspace + 6 * BD);
    float* d_r_delta_f = d_r_h_f + dim_;
    float* db_f = d_r_delta_f + dim_;
    float* db_delta_f = db_f + dim_;
    float* d_b_gate_f = db_delta_f + dim_;

    // Zero accumulators
    cudaMemsetAsync(d_r_h_f, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(d_r_delta_f, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_f, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(db_delta_f, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(d_b_gate_f, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dh_recurrent, 0, BD * sizeof(T), stream_);

    for (int t = steps - 1; t >= 0; --t) {
        const T* x_t = x + t * BD;
        const T* h_prev = h + t * BD;
        const T* h_out = h + (t + 1) * BD;
        const T* d_output_t = d_output + t * BD;

        const T* v_t = v_cache + t * BD;
        const T* delta_t = delta_cache + t * BD;
        const T* gate_t = gate_cache + t * BD;

        T* dx_t = dx + t * BD;
        const int num_blocks = (BD + block_size - 1) / block_size;

        // Backward through learned gate projection: output = h * silu(W_gate @ x + b_gate)
        DiagTripleRSelectiveOutputBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_,
            h_out, gate_t,  // gate_cache contains cached gate_raw values
            d_output_t,
            dh, d_gate_proj, d_b_gate_f);

        // Add recurrent gradient from next timestep: dh += dh_recurrent
        VectorAddInplace<T><<<num_blocks, block_size, 0, stream_>>>(BD, dh, dh_recurrent);

        // Backward through update
        DiagTripleRUpdateBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_,
            h_prev,
            v_t,
            delta_t,
            r_h,
            r_delta,
            dh,
            dh_prev,
            d_wx_x,
            d_wdelta_x,
            d_r_h_f,
            d_r_delta_f,
            db_f,
            db_delta_f);

        // Save dh_prev for next iteration
        cudaMemcpyAsync(dh_recurrent, dh_prev, BD * sizeof(T), cudaMemcpyDeviceToDevice, stream_);

        // Accumulate W_x gradient: dW_x += d_wx_x @ x^T
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_, &alpha_one, x_t, dim_, d_wx_x, dim_, &alpha_one, dW_x, dim_);

        // Accumulate W_delta gradient
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_, &alpha_one, x_t, dim_, d_wdelta_x, dim_, &alpha_one, dW_delta, dim_);

        // Accumulate W_gate gradient: dW_gate += d_gate_proj @ x^T
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_, &alpha_one, x_t, dim_, d_gate_proj, dim_, &alpha_one, dW_gate, dim_);

        // dx = W_x^T @ d_wx_x + W_delta^T @ d_wdelta_x + W_gate^T @ d_gate_proj
        cudaMemsetAsync(dx_t, 0, BD * sizeof(T), stream_);
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha_one, W_x, dim_, d_wx_x, dim_, &alpha_one, dx_t, dim_);
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha_one, W_delta, dim_, d_wdelta_x, dim_, &alpha_one, dx_t, dim_);
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha_one, W_gate, dim_, d_gate_proj, dim_, &alpha_one, dx_t, dim_);
    }

    // Copy float accumulators to output
    const int copy_blocks = (dim_ + block_size - 1) / block_size;
    CopyFloatToT<T><<<copy_blocks, block_size, 0, stream_>>>(dim_, d_r_h_f, d_r_h);
    CopyFloatToT<T><<<copy_blocks, block_size, 0, stream_>>>(dim_, d_r_delta_f, d_r_delta);
    CopyFloatToT<T><<<copy_blocks, block_size, 0, stream_>>>(dim_, db_f, db);
    CopyFloatToT<T><<<copy_blocks, block_size, 0, stream_>>>(dim_, db_delta_f, db_delta);
    CopyFloatToT<T><<<copy_blocks, block_size, 0, stream_>>>(dim_, d_b_gate_f, db_gate);
}

// Explicit instantiations
template class DiagTripleRForward<float>;
template class DiagTripleRForward<__half>;
template class DiagTripleRForward<__nv_bfloat16>;
template class DiagTripleRForward<double>;
template class DiagTripleRBackward<float>;
template class DiagTripleRBackward<__half>;
template class DiagTripleRBackward<__nv_bfloat16>;
template class DiagTripleRBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
