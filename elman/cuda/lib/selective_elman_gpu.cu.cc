// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// Level 2: Selective Elman - Gated recurrence + compete×silu output
// delta = sigmoid(W_delta @ x_t + b_delta)
// h_t = (1 - delta) * h_{t-1} + delta * tanh(W_x @ x_t + W_h @ h_{t-1} + b)
// compete = softmax(h_t.reshape(groups), dim=-1)
// output = compete * silu(W_out @ h_t)

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cfloat>

#include "haste/elman_ladder.h"
#include "blas.h"
#include "inline_ops.h"

namespace {

// Kernel: Compute gated update (same as GatedElman)
template<typename T>
__global__ void SelectiveElmanGatedUpdate(
    const int batch_size,
    const int dim,
    const T* __restrict__ h_prev,
    const T* __restrict__ v_in,
    const T* __restrict__ delta_raw,
    const T* __restrict__ b,
    const T* __restrict__ b_delta,
    T* __restrict__ h_out,
    T* __restrict__ v_cache,
    T* __restrict__ delta_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // Delta gate: sigmoid(delta_raw + b_delta)
        float delta_in = static_cast<float>(delta_raw[idx]) + static_cast<float>(b_delta[d]);
        float delta = 1.0f / (1.0f + expf(-delta_in));
        if (delta_cache) delta_cache[idx] = static_cast<T>(delta);

        // Candidate: tanh(v + b)
        float v = static_cast<float>(v_in[idx]) + static_cast<float>(b[d]);
        if (v_cache) v_cache[idx] = static_cast<T>(v);
        float candidate = tanhf(v);

        // Gated update: h = (1 - delta) * h_prev + delta * candidate
        float h_p = static_cast<float>(h_prev[idx]);
        float h_new = (1.0f - delta) * h_p + delta * candidate;
        h_out[idx] = static_cast<T>(h_new);
    }
}

// Kernel: Compute compete×silu output
// Each block handles one batch element
// Groups are processed within each block
template<typename T>
__global__ void SelectiveOutput(
    const int batch_size,
    const int dim,
    const int n_groups,
    const int group_size,
    const T* __restrict__ h,          // [B, dim] hidden state
    const T* __restrict__ w_out_h,    // [B, dim] W_out @ h
    T* __restrict__ output,           // [B, dim] output
    T* __restrict__ compete_cache) {  // [B, dim] cached compete

    extern __shared__ float smem[];

    const int b = blockIdx.x;
    const int g = blockIdx.y;

    if (b >= batch_size || g >= n_groups) return;

    const int base = b * dim + g * group_size;

    // Step 1: Find max in group (for softmax stability)
    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        max_val = fmaxf(max_val, static_cast<float>(h[base + i]));
    }
    // Reduce max within block
    smem[threadIdx.x] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        }
        __syncthreads();
    }
    max_val = smem[0];
    __syncthreads();

    // Step 2: Compute exp(h - max) and sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float exp_val = expf(static_cast<float>(h[base + i]) - max_val);
        smem[threadIdx.x + i - (threadIdx.x % blockDim.x)] = exp_val;  // Store for later
        sum += exp_val;
    }
    // Reduce sum
    __syncthreads();
    float* sum_smem = smem + group_size;
    sum_smem[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sum_smem[threadIdx.x] += sum_smem[threadIdx.x + s];
        }
        __syncthreads();
    }
    sum = sum_smem[0];

    // Step 3: Compute compete = exp(h - max) / sum, then output = compete * silu(w_out_h)
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float exp_val = expf(static_cast<float>(h[base + i]) - max_val);
        float compete = exp_val / sum;
        if (compete_cache) compete_cache[base + i] = static_cast<T>(compete);

        // silu(x) = x * sigmoid(x)
        float w = static_cast<float>(w_out_h[base + i]);
        float silu_val = w / (1.0f + expf(-w));

        output[base + i] = static_cast<T>(compete * silu_val);
    }
}

// Backward through selective output
template<typename T>
__global__ void SelectiveOutputBackward(
    const int batch_size,
    const int dim,
    const int n_groups,
    const int group_size,
    const T* __restrict__ h,
    const T* __restrict__ w_out_h,
    const T* __restrict__ compete,
    const T* __restrict__ d_output,
    T* __restrict__ dh_compete,       // [B, dim] gradient through compete
    T* __restrict__ d_w_out_h) {      // [B, dim] gradient through silu path

    extern __shared__ float smem[];

    const int b = blockIdx.x;
    const int g = blockIdx.y;

    if (b >= batch_size || g >= n_groups) return;

    const int base = b * dim + g * group_size;

    // For each output element: output = compete * silu(w_out_h)
    // d_silu = d_output * compete
    // d_compete = d_output * silu(w_out_h)

    // First compute d_silu and d_compete
    float sum_compete_dcompete = 0.0f;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float dout = static_cast<float>(d_output[base + i]);
        float w = static_cast<float>(w_out_h[base + i]);
        float sig = 1.0f / (1.0f + expf(-w));
        float silu_val = w * sig;

        // d_silu through silu: dsilu/dw = sig + w * sig * (1 - sig) = sig * (1 + w * (1 - sig))
        float dsilu_dw = sig * (1.0f + w * (1.0f - sig));
        float comp = static_cast<float>(compete[base + i]);
        d_w_out_h[base + i] = static_cast<T>(dout * comp * dsilu_dw);

        // d_compete for softmax backward
        float d_comp = dout * silu_val;
        // Softmax backward: dh = compete * (d_compete - sum(compete * d_compete))
        sum_compete_dcompete += comp * d_comp;
    }

    // Reduce sum
    smem[threadIdx.x] = sum_compete_dcompete;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] += smem[threadIdx.x + s];
        }
        __syncthreads();
    }
    sum_compete_dcompete = smem[0];

    // Compute final dh through compete
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float dout = static_cast<float>(d_output[base + i]);
        float w = static_cast<float>(w_out_h[base + i]);
        float sig = 1.0f / (1.0f + expf(-w));
        float silu_val = w * sig;
        float comp = static_cast<float>(compete[base + i]);
        float d_comp = dout * silu_val;

        // Softmax backward
        float dh = comp * (d_comp - sum_compete_dcompete);
        dh_compete[base + i] = static_cast<T>(dh);
    }
}

// Backward through gated update
template<typename T>
__global__ void SelectiveElmanGatedBackward(
    const int batch_size,
    const int dim,
    const T* __restrict__ h_prev,
    const T* __restrict__ v,
    const T* __restrict__ delta,
    const T* __restrict__ dh,
    const T* __restrict__ dh_recurrent,
    T* __restrict__ dv,
    T* __restrict__ d_delta_raw,
    T* __restrict__ dh_prev_out,
    float* __restrict__ db,
    float* __restrict__ db_delta) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float grad_h = static_cast<float>(dh[idx]);
        if (dh_recurrent) grad_h += static_cast<float>(dh_recurrent[idx]);

        float cand = tanhf(static_cast<float>(v[idx]));
        float del = static_cast<float>(delta[idx]);
        float one_minus_del = 1.0f - del;

        float d_cand = grad_h * del;
        float dtanh = 1.0f - cand * cand;
        float dv_val = d_cand * dtanh;
        dv[idx] = static_cast<T>(dv_val);

        float h_p = static_cast<float>(h_prev[idx]);
        float d_delta = grad_h * (cand - h_p);
        float dsigmoid = del * one_minus_del;
        float d_delta_raw_val = d_delta * dsigmoid;
        d_delta_raw[idx] = static_cast<T>(d_delta_raw_val);

        dh_prev_out[idx] = static_cast<T>(grad_h * one_minus_del);

        atomicAdd(&db[d], dv_val);
        atomicAdd(&db_delta[d], d_delta_raw_val);
    }
}

}  // anonymous namespace


namespace haste {
namespace v0 {
namespace elman_ladder {

// =============================================================================
// Selective Elman Forward
// =============================================================================

template<typename T>
SelectiveElmanForward<T>::SelectiveElmanForward(
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
void SelectiveElmanForward<T>::Run(
    int steps,
    const T* W_x,
    const T* W_h,
    const T* W_delta,
    const T* W_out,
    const T* b,
    const T* b_delta,
    const T* x,
    T* h,
    T* output,
    T* v,
    T* delta_cache,
    T* compete_cache) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;
    const int group_size = dim_ / n_groups_;

    // Workspace
    T *v_tmp, *delta_tmp, *w_out_h;
    cudaMalloc(&v_tmp, BD * sizeof(T));
    cudaMalloc(&delta_tmp, BD * sizeof(T));
    cudaMalloc(&w_out_h, BD * sizeof(T));

    for (int t = 0; t < steps; ++t) {
        const T* x_t = x + t * BD;
        const T* h_prev = h + t * BD;
        T* h_t = h + (t + 1) * BD;
        T* out_t = output + t * BD;
        T* v_t = training_ ? (v + t * BD) : nullptr;
        T* delta_t = training_ ? (delta_cache + t * BD) : nullptr;
        T* compete_t = training_ ? (compete_cache + t * BD) : nullptr;

        // v_tmp = W_x @ x_t + W_h @ h_prev
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, W_x, dim_, x_t, dim_, &beta_zero, v_tmp, dim_);
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, W_h, dim_, h_prev, dim_, &alpha, v_tmp, dim_);

        // delta_tmp = W_delta @ x_t
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, W_delta, dim_, x_t, dim_, &beta_zero, delta_tmp, dim_);

        // Gated update
        SelectiveElmanGatedUpdate<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_prev, v_tmp, delta_tmp, b, b_delta, h_t, v_t, delta_t);

        // w_out_h = W_out @ h_t
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, W_out, dim_, h_t, dim_, &beta_zero, w_out_h, dim_);

        // Selective output: compete × silu
        dim3 grid(batch_size_, n_groups_);
        int smem_size = (group_size + block_size) * sizeof(float);
        SelectiveOutput<T><<<grid, block_size, smem_size, stream_>>>(
            batch_size_, dim_, n_groups_, group_size, h_t, w_out_h, out_t, compete_t);
    }

    cudaFree(v_tmp);
    cudaFree(delta_tmp);
    cudaFree(w_out_h);
}

// =============================================================================
// Selective Elman Backward
// =============================================================================

template<typename T>
SelectiveElmanBackward<T>::SelectiveElmanBackward(
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
void SelectiveElmanBackward<T>::Run(
    int steps,
    const T* W_x,
    const T* W_h,
    const T* W_delta,
    const T* W_out,
    const T* x,
    const T* h,
    const T* v,
    const T* delta_cache,
    const T* compete_cache,
    const T* d_output,
    T* dx,
    T* dW_x,
    T* dW_h,
    T* dW_delta,
    T* dW_out,
    T* db,
    T* db_delta) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;
    const int group_size = dim_ / n_groups_;

    // Workspace
    T *dv, *d_delta_raw, *dh_recurrent, *dh_prev;
    T *dh_compete, *d_w_out_h, *w_out_h;
    cudaMalloc(&dv, BD * sizeof(T));
    cudaMalloc(&d_delta_raw, BD * sizeof(T));
    cudaMalloc(&dh_recurrent, BD * sizeof(T));
    cudaMalloc(&dh_prev, BD * sizeof(T));
    cudaMalloc(&dh_compete, BD * sizeof(T));
    cudaMalloc(&d_w_out_h, BD * sizeof(T));
    cudaMalloc(&w_out_h, BD * sizeof(T));
    cudaMemset(dh_recurrent, 0, BD * sizeof(T));

    // Float buffers for bias gradients
    float *db_float, *db_delta_float;
    cudaMalloc(&db_float, dim_ * sizeof(float));
    cudaMalloc(&db_delta_float, dim_ * sizeof(float));
    cudaMemset(db_float, 0, dim_ * sizeof(float));
    cudaMemset(db_delta_float, 0, dim_ * sizeof(float));

    // Zero weight gradients
    cudaMemset(dW_x, 0, dim_ * dim_ * sizeof(T));
    cudaMemset(dW_h, 0, dim_ * dim_ * sizeof(T));
    cudaMemset(dW_delta, 0, dim_ * dim_ * sizeof(T));
    cudaMemset(dW_out, 0, dim_ * dim_ * sizeof(T));

    for (int t = steps - 1; t >= 0; --t) {
        const T* x_t = x + t * BD;
        const T* h_prev = h + t * BD;
        const T* h_t = h + (t + 1) * BD;
        const T* v_t = v + t * BD;
        const T* delta_t = delta_cache + t * BD;
        const T* compete_t = compete_cache + t * BD;
        const T* d_out_t = d_output + t * BD;
        T* dx_t = dx + t * BD;

        // Recompute w_out_h = W_out @ h_t
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, W_out, dim_, h_t, dim_, &beta_zero, w_out_h, dim_);

        // Backward through selective output
        dim3 grid(batch_size_, n_groups_);
        int smem_size = block_size * sizeof(float);
        SelectiveOutputBackward<T><<<grid, block_size, smem_size, stream_>>>(
            batch_size_, dim_, n_groups_, group_size, h_t, w_out_h, compete_t,
            d_out_t, dh_compete, d_w_out_h);

        // dW_out += d_w_out_h @ h_t^T
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_, &alpha, d_w_out_h, dim_, h_t, dim_, &alpha, dW_out, dim_);

        // dh from W_out path: W_out^T @ d_w_out_h
        // This gets added to dh_compete for total dh_t
        blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, W_out, dim_, d_w_out_h, dim_, &alpha, dh_compete, dim_);

        // Now backward through gated update
        SelectiveElmanGatedBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_prev, v_t, delta_t, dh_compete,
            (t < steps - 1) ? dh_recurrent : nullptr,
            dv, d_delta_raw, dh_prev, db_float, db_delta_float);

        // dx = W_x^T @ dv + W_delta^T @ d_delta_raw
        blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, W_x, dim_, dv, dim_, &beta_zero, dx_t, dim_);
        blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, W_delta, dim_, d_delta_raw, dim_, &alpha, dx_t, dim_);

        // dh_recurrent = W_h^T @ dv + dh_prev
        blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, W_h, dim_, dv, dim_, &beta_zero, dh_recurrent, dim_);
        cudaMemcpy(dh_recurrent, dh_prev, BD * sizeof(T), cudaMemcpyDeviceToDevice);

        // Weight gradients
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_, &alpha, dv, dim_, x_t, dim_, &alpha, dW_x, dim_);
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_, &alpha, dv, dim_, h_prev, dim_, &alpha, dW_h, dim_);
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_, &alpha, d_delta_raw, dim_, x_t, dim_, &alpha, dW_delta, dim_);
    }

    // Copy float bias gradients to T type
    cudaMemset(db, 0, dim_ * sizeof(T));
    cudaMemset(db_delta, 0, dim_ * sizeof(T));
    if constexpr (std::is_same<T, float>::value) {
        cudaMemcpy(db, db_float, dim_ * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(db_delta, db_delta_float, dim_ * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    cudaFree(dv);
    cudaFree(d_delta_raw);
    cudaFree(dh_recurrent);
    cudaFree(dh_prev);
    cudaFree(dh_compete);
    cudaFree(d_w_out_h);
    cudaFree(w_out_h);
    cudaFree(db_float);
    cudaFree(db_delta_float);
}

// Explicit template instantiations
template struct SelectiveElmanForward<__half>;
template struct SelectiveElmanForward<__nv_bfloat16>;
template struct SelectiveElmanForward<float>;
template struct SelectiveElmanForward<double>;

template struct SelectiveElmanBackward<__half>;
template struct SelectiveElmanBackward<__nv_bfloat16>;
template struct SelectiveElmanBackward<float>;
template struct SelectiveElmanBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace haste
