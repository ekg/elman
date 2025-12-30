// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// Level 5: Linear Triple R Elman
// v = R_x @ x + R_h @ h_prev + b
// delta = sigmoid(W_delta @ x + R_delta @ h_prev + b_delta)
// h_new = (1-delta) * h_prev + delta * tanh(v)
// compete = softmax(h.reshape(groups), dim=-1)
// output = compete * silu(W_out @ h)

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

// Simple kernel for y += x (element-wise vector addition)
template<typename T>
__global__ void VectorAdd(const int n, const T* __restrict__ x, T* __restrict__ y) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = static_cast<T>(static_cast<float>(y[idx]) + static_cast<float>(x[idx]));
    }
}

// Kernel: Compute gated update with Triple R matrices
template<typename T>
__global__ void LinearTripleRGatedUpdate(
    const int batch_size,
    const int dim,
    const T* __restrict__ h_prev,
    const T* __restrict__ rx_x,        // [B, dim] R_x @ x
    const T* __restrict__ rh_h,        // [B, dim] R_h @ h_prev
    const T* __restrict__ wdelta_x,    // [B, dim] W_delta @ x
    const T* __restrict__ rdelta_h,    // [B, dim] R_delta @ h_prev
    const T* __restrict__ b,           // [dim]
    const T* __restrict__ b_delta,     // [dim]
    T* __restrict__ h_out,
    T* __restrict__ v_cache,
    T* __restrict__ delta_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // Delta gate: sigmoid(W_delta @ x + R_delta @ h_prev + b_delta)
        float delta_in = static_cast<float>(wdelta_x[idx]) +
                         static_cast<float>(rdelta_h[idx]) +
                         static_cast<float>(b_delta[d]);
        float delta = 1.0f / (1.0f + expf(-delta_in));
        if (delta_cache) delta_cache[idx] = static_cast<T>(delta);

        // Candidate: v = R_x @ x + R_h @ h_prev + b
        float v = static_cast<float>(rx_x[idx]) +
                  static_cast<float>(rh_h[idx]) +
                  static_cast<float>(b[d]);
        if (v_cache) v_cache[idx] = static_cast<T>(v);
        float candidate = tanhf(v);

        // Gated update: h = (1 - delta) * h_prev + delta * candidate
        float h_p = static_cast<float>(h_prev[idx]);
        float h_new = (1.0f - delta) * h_p + delta * candidate;
        h_out[idx] = static_cast<T>(h_new);
    }
}

// Kernel: Compute compete×silu output
template<typename T>
__global__ void LinearTripleROutput(
    const int batch_size,
    const int dim,
    const int n_groups,
    const int group_size,
    const T* __restrict__ h,
    const T* __restrict__ w_out_h,
    T* __restrict__ output,
    T* __restrict__ compete_cache) {

    extern __shared__ float smem[];

    const int b = blockIdx.x;
    const int g = blockIdx.y;

    if (b >= batch_size || g >= n_groups) return;

    const int base = b * dim + g * group_size;

    // Find max for softmax stability
    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        max_val = fmaxf(max_val, static_cast<float>(h[base + i]));
    }
    smem[threadIdx.x] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        __syncthreads();
    }
    max_val = smem[0];
    __syncthreads();

    // Compute exp sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        sum += expf(static_cast<float>(h[base + i]) - max_val);
    }
    float* sum_smem = smem + blockDim.x;
    sum_smem[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sum_smem[threadIdx.x] += sum_smem[threadIdx.x + s];
        __syncthreads();
    }
    sum = sum_smem[0];

    // Compute output = compete * silu(w_out_h)
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float compete = expf(static_cast<float>(h[base + i]) - max_val) / sum;
        if (compete_cache) compete_cache[base + i] = static_cast<T>(compete);

        float w = static_cast<float>(w_out_h[base + i]);
        float silu_val = w / (1.0f + expf(-w));
        output[base + i] = static_cast<T>(compete * silu_val);
    }
}

// Backward through selective output
template<typename T>
__global__ void LinearTripleROutputBackward(
    const int batch_size,
    const int dim,
    const int n_groups,
    const int group_size,
    const T* __restrict__ h,
    const T* __restrict__ w_out_h,
    const T* __restrict__ compete,
    const T* __restrict__ d_output,
    T* __restrict__ dh_compete,
    T* __restrict__ d_w_out_h) {

    extern __shared__ float smem[];

    const int b = blockIdx.x;
    const int g = blockIdx.y;

    if (b >= batch_size || g >= n_groups) return;

    const int base = b * dim + g * group_size;

    float sum_compete_dcompete = 0.0f;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float dout = static_cast<float>(d_output[base + i]);
        float w = static_cast<float>(w_out_h[base + i]);
        float sig = 1.0f / (1.0f + expf(-w));
        float silu_val = w * sig;
        float dsilu_dw = sig * (1.0f + w * (1.0f - sig));
        float comp = static_cast<float>(compete[base + i]);

        d_w_out_h[base + i] = static_cast<T>(dout * comp * dsilu_dw);
        sum_compete_dcompete += comp * dout * silu_val;
    }

    smem[threadIdx.x] = sum_compete_dcompete;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    sum_compete_dcompete = smem[0];

    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float dout = static_cast<float>(d_output[base + i]);
        float w = static_cast<float>(w_out_h[base + i]);
        float sig = 1.0f / (1.0f + expf(-w));
        float silu_val = w * sig;
        float comp = static_cast<float>(compete[base + i]);
        float d_comp = dout * silu_val;
        dh_compete[base + i] = static_cast<T>(comp * (d_comp - sum_compete_dcompete));
    }
}

// Backward through Triple R gated update
template<typename T>
__global__ void LinearTripleRGatedBackward(
    const int batch_size,
    const int dim,
    const T* __restrict__ h_prev,
    const T* __restrict__ v,
    const T* __restrict__ delta,
    const T* __restrict__ dh,
    const T* __restrict__ dh_recurrent,
    T* __restrict__ dv,
    T* __restrict__ d_delta_raw,
    T* __restrict__ dh_prev_gated,
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

        float dh_prev_val = one_minus_del * grad_h;
        dh_prev_gated[idx] = static_cast<T>(dh_prev_val);

        atomicAdd(&db[d], dv_val);
        atomicAdd(&db_delta[d], d_delta_raw_val);
    }
}

}  // anonymous namespace


namespace hasty {
namespace v0 {
namespace elman_ladder {

// =============================================================================
// Linear Triple R Forward
// =============================================================================

template<typename T>
LinearTripleRForward<T>::LinearTripleRForward(
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
void LinearTripleRForward<T>::Run(
    int steps,
    const T* R_h,
    const T* R_x,
    const T* R_delta,
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
    T *rx_x, *rh_h, *wdelta_x, *rdelta_h, *w_out_h;
    cudaMalloc(&rx_x, BD * sizeof(T));
    cudaMalloc(&rh_h, BD * sizeof(T));
    cudaMalloc(&wdelta_x, BD * sizeof(T));
    cudaMalloc(&rdelta_h, BD * sizeof(T));
    cudaMalloc(&w_out_h, BD * sizeof(T));

    for (int t = 0; t < steps; ++t) {
        const T* x_t = x + t * BD;
        const T* h_prev = h + t * BD;
        T* h_t = h + (t + 1) * BD;
        T* out_t = output + t * BD;
        T* v_t = training_ ? (v + t * BD) : nullptr;
        T* delta_t = training_ ? (delta_cache + t * BD) : nullptr;
        T* compete_t = training_ ? (compete_cache + t * BD) : nullptr;

        // rx_x = x_t @ R_x.T
        blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, R_x, dim_, x_t, dim_, &beta_zero, rx_x, dim_);

        // rh_h = h_prev @ R_h.T
        blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, R_h, dim_, h_prev, dim_, &beta_zero, rh_h, dim_);

        // wdelta_x = x_t @ W_delta.T
        blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, W_delta, dim_, x_t, dim_, &beta_zero, wdelta_x, dim_);

        // rdelta_h = h_prev @ R_delta.T
        blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, R_delta, dim_, h_prev, dim_, &beta_zero, rdelta_h, dim_);

        // Triple R gated update
        LinearTripleRGatedUpdate<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_prev, rx_x, rh_h, wdelta_x, rdelta_h, b, b_delta, h_t, v_t, delta_t);

        // w_out_h = h_t @ W_out.T
        blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, W_out, dim_, h_t, dim_, &beta_zero, w_out_h, dim_);

        // Selective output
        dim3 grid(batch_size_, n_groups_);
        int smem_size = 2 * block_size * sizeof(float);
        LinearTripleROutput<T><<<grid, block_size, smem_size, stream_>>>(
            batch_size_, dim_, n_groups_, group_size, h_t, w_out_h, out_t, compete_t);
    }

    cudaFree(rx_x);
    cudaFree(rh_h);
    cudaFree(wdelta_x);
    cudaFree(rdelta_h);
    cudaFree(w_out_h);
}

// =============================================================================
// Linear Triple R Backward
// =============================================================================

template<typename T>
LinearTripleRBackward<T>::LinearTripleRBackward(
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
void LinearTripleRBackward<T>::Run(
    int steps,
    const T* R_h,
    const T* R_x,
    const T* R_delta,
    const T* W_delta,
    const T* W_out,
    const T* x,
    const T* h,
    const T* v,
    const T* delta_cache,
    const T* compete_cache,
    const T* d_output,
    T* dx,
    T* dR_h,
    T* dR_x,
    T* dR_delta,
    T* dW_delta,
    T* dW_out,
    T* db,
    T* db_delta) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;
    const int group_size = dim_ / n_groups_;

    // Zero out weight gradients
    cudaMemset(dR_h, 0, dim_ * dim_ * sizeof(T));
    cudaMemset(dR_x, 0, dim_ * dim_ * sizeof(T));
    cudaMemset(dR_delta, 0, dim_ * dim_ * sizeof(T));
    cudaMemset(dW_delta, 0, dim_ * dim_ * sizeof(T));
    cudaMemset(dW_out, 0, dim_ * dim_ * sizeof(T));

    // Float buffers for bias gradients
    float *db_f, *db_delta_f;
    cudaMalloc(&db_f, dim_ * sizeof(float));
    cudaMalloc(&db_delta_f, dim_ * sizeof(float));
    cudaMemset(db_f, 0, dim_ * sizeof(float));
    cudaMemset(db_delta_f, 0, dim_ * sizeof(float));

    // Workspace
    T *w_out_h, *dh_compete, *d_w_out_h, *dv, *d_delta_raw, *dh_prev_gated, *dh_recurrent;
    T *dh_rh, *dh_rdelta;
    cudaMalloc(&w_out_h, BD * sizeof(T));
    cudaMalloc(&dh_compete, BD * sizeof(T));
    cudaMalloc(&d_w_out_h, BD * sizeof(T));
    cudaMalloc(&dv, BD * sizeof(T));
    cudaMalloc(&d_delta_raw, BD * sizeof(T));
    cudaMalloc(&dh_prev_gated, BD * sizeof(T));
    cudaMalloc(&dh_recurrent, BD * sizeof(T));
    cudaMalloc(&dh_rh, BD * sizeof(T));
    cudaMalloc(&dh_rdelta, BD * sizeof(T));
    cudaMemset(dh_recurrent, 0, BD * sizeof(T));

    for (int t = steps - 1; t >= 0; --t) {
        const T* x_t = x + t * BD;
        const T* h_prev = h + t * BD;
        const T* h_t = h + (t + 1) * BD;
        const T* v_t = v + t * BD;
        const T* delta_t = delta_cache + t * BD;
        const T* compete_t = compete_cache + t * BD;
        const T* d_out_t = d_output + t * BD;
        T* dx_t = dx + t * BD;

        // Recompute w_out_h = h_t @ W_out.T
        blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, W_out, dim_, h_t, dim_, &beta_zero, w_out_h, dim_);

        // Backward through output
        dim3 grid(batch_size_, n_groups_);
        int smem_size = block_size * sizeof(float);
        LinearTripleROutputBackward<T><<<grid, block_size, smem_size, stream_>>>(
            batch_size_, dim_, n_groups_, group_size, h_t, w_out_h, compete_t, d_out_t, dh_compete, d_w_out_h);

        // dW_out += d_w_out_h.T @ h_t
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_, &alpha, d_w_out_h, dim_, h_t, dim_, &beta_one, dW_out, dim_);

        // dh from W_out path
        T* dh_wout;
        cudaMalloc(&dh_wout, BD * sizeof(T));
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, W_out, dim_, d_w_out_h, dim_, &beta_zero, dh_wout, dim_);

        // Combine dh
        T* dh;
        cudaMalloc(&dh, BD * sizeof(T));
        cudaMemcpy(dh, dh_compete, BD * sizeof(T), cudaMemcpyDeviceToDevice);
        VectorAdd<T><<<(BD + 255) / 256, 256, 0, stream_>>>(BD, dh_wout, dh);
        cudaFree(dh_wout);

        // Backward through gated update
        LinearTripleRGatedBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, h_prev, v_t, delta_t, dh, dh_recurrent, dv, d_delta_raw, dh_prev_gated, db_f, db_delta_f);

        // dR_x += dv.T @ x_t
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_, &alpha, dv, dim_, x_t, dim_, &beta_one, dR_x, dim_);

        // dR_h += dv.T @ h_prev
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_, &alpha, dv, dim_, h_prev, dim_, &beta_one, dR_h, dim_);

        // dW_delta += d_delta_raw.T @ x_t
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_, &alpha, d_delta_raw, dim_, x_t, dim_, &beta_one, dW_delta, dim_);

        // dR_delta += d_delta_raw.T @ h_prev
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_, &alpha, d_delta_raw, dim_, h_prev, dim_, &beta_one, dR_delta, dim_);

        // dx = dv @ R_x + d_delta_raw @ W_delta
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, R_x, dim_, dv, dim_, &beta_zero, dx_t, dim_);
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, W_delta, dim_, d_delta_raw, dim_, &beta_one, dx_t, dim_);

        // dh_prev from R_h and R_delta paths
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, R_h, dim_, dv, dim_, &beta_zero, dh_rh, dim_);
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, R_delta, dim_, d_delta_raw, dim_, &beta_zero, dh_rdelta, dim_);

        // dh_recurrent = dh_prev_gated + dh_rh + dh_rdelta
        cudaMemcpy(dh_recurrent, dh_prev_gated, BD * sizeof(T), cudaMemcpyDeviceToDevice);
        VectorAdd<T><<<(BD + 255) / 256, 256, 0, stream_>>>(BD, dh_rh, dh_recurrent);
        VectorAdd<T><<<(BD + 255) / 256, 256, 0, stream_>>>(BD, dh_rdelta, dh_recurrent);

        cudaFree(dh);
    }

    // Convert float gradients to T
    cudaMemcpy(db, db_f, dim_ * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(db_delta, db_delta_f, dim_ * sizeof(float), cudaMemcpyDeviceToDevice);

    cudaFree(db_f);
    cudaFree(db_delta_f);
    cudaFree(w_out_h);
    cudaFree(dh_compete);
    cudaFree(d_w_out_h);
    cudaFree(dv);
    cudaFree(d_delta_raw);
    cudaFree(dh_prev_gated);
    cudaFree(dh_recurrent);
    cudaFree(dh_rh);
    cudaFree(dh_rdelta);
}

// Explicit instantiations
template class LinearTripleRForward<float>;
template class LinearTripleRForward<__half>;
template class LinearTripleRForward<__nv_bfloat16>;
template class LinearTripleRForward<double>;
template class LinearTripleRBackward<float>;
template class LinearTripleRBackward<__half>;
template class LinearTripleRBackward<__nv_bfloat16>;
template class LinearTripleRBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
