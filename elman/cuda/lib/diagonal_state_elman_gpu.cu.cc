// Diagonal State-Expanded Elman - E16
//
// h' = tanh(A ⊙ h + B @ x)
// y = C @ h * silu(z)
//
// Uses cuBLAS for GEMMs, fused kernels for element-wise ops only.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#include "hasty/elman_ladder.h"
#include "blas.h"
#include "inline_ops.h"

namespace {

using namespace hasty::v0::elman_ladder;

// =============================================================================
// Fused element-wise kernels
// =============================================================================

// Fused: h_new = tanh(A * h_prev + Bx), also cache pre-activation
__global__ void DiagRecurrenceTanh_BF16(
    const int n,
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ Bx,
    const __nv_bfloat16* __restrict__ h_prev,
    __nv_bfloat16* __restrict__ h_new,
    __nv_bfloat16* __restrict__ v_cache,
    const int d_state) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const int s = idx % d_state;
        float a = __bfloat162float(A[s]);
        float h = __bfloat162float(h_prev[idx]);
        float bx = __bfloat162float(Bx[idx]);
        float pre = a * h + bx;
        if (v_cache) v_cache[idx] = __float2bfloat16(pre);
        h_new[idx] = __float2bfloat16(tanhf(pre));
    }
}

// Fused: output = y * silu(z)
__global__ void GateSilu_BF16(
    const int n,
    const __nv_bfloat16* __restrict__ y,
    const __nv_bfloat16* __restrict__ z,
    __nv_bfloat16* __restrict__ output) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float y_val = __bfloat162float(y[idx]);
        float z_val = __bfloat162float(z[idx]);
        float sig = 1.0f / (1.0f + __expf(-z_val));
        output[idx] = __float2bfloat16(y_val * z_val * sig);
    }
}

// Backward: dv = dh * dtanh, update dh_rec
__global__ void TanhBackward_BF16(
    const int n,
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ dh,
    const __nv_bfloat16* __restrict__ dh_rec_in,
    const __nv_bfloat16* __restrict__ h_prev,
    __nv_bfloat16* __restrict__ dv,
    __nv_bfloat16* __restrict__ dh_rec_out,
    float* __restrict__ dA_accum,
    const int d_state) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const int s = idx % d_state;
        float grad = __bfloat162float(dh[idx]);
        if (dh_rec_in) grad += __bfloat162float(dh_rec_in[idx]);

        float v_val = __bfloat162float(v[idx]);
        float h_val = tanhf(v_val);
        float dtanh = 1.0f - h_val * h_val;
        float dv_val = grad * dtanh;

        dv[idx] = __float2bfloat16(dv_val);

        // dA accumulation
        float hp = __bfloat162float(h_prev[idx]);
        atomicAdd(&dA_accum[s], dv_val * hp);

        // dh_rec = dv * A
        float a = __bfloat162float(A[s]);
        dh_rec_out[idx] = __float2bfloat16(dv_val * a);
    }
}

// Backward through gate: dy, dz
__global__ void GateBackward_BF16(
    const int n,
    const __nv_bfloat16* __restrict__ y,
    const __nv_bfloat16* __restrict__ z,
    const __nv_bfloat16* __restrict__ d_output,
    __nv_bfloat16* __restrict__ dy,
    __nv_bfloat16* __restrict__ dz) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float y_val = __bfloat162float(y[idx]);
        float z_val = __bfloat162float(z[idx]);
        float dout = __bfloat162float(d_output[idx]);

        float sig = 1.0f / (1.0f + __expf(-z_val));
        float silu = z_val * sig;

        dy[idx] = __float2bfloat16(dout * silu);

        float dsilu = sig * (1.0f + z_val * (1.0f - sig));
        dz[idx] = __float2bfloat16(dout * y_val * dsilu);
    }
}

// Float to BF16 conversion
__global__ void FloatToBF16(const int n, const float* src, __nv_bfloat16* dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2bfloat16(src[idx]);
}

}  // namespace

namespace hasty {
namespace v0 {
namespace elman_ladder {

// =============================================================================
// Forward - BF16
// =============================================================================

template<>
DiagonalStateElmanForward<__nv_bfloat16>::DiagonalStateElmanForward(
    bool training, int batch_size, int d_model, int d_state,
    const cublasHandle_t& blas_handle, const cudaStream_t& stream)
    : training_(training), batch_size_(batch_size), d_model_(d_model),
      d_state_(d_state), blas_handle_(blas_handle), stream_(stream) {}

template<>
void DiagonalStateElmanForward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* B,
    const __nv_bfloat16* C,
    const __nv_bfloat16* A,
    const __nv_bfloat16* x,
    const __nv_bfloat16* z,
    __nv_bfloat16* h,
    __nv_bfloat16* output,
    __nv_bfloat16* v,
    __nv_bfloat16* workspace) {

    const int BS = batch_size_ * d_state_;
    const int BD = batch_size_ * d_model_;

    __nv_bfloat16* Bx_all = workspace;
    __nv_bfloat16* Cy = workspace + steps * BS;

    __nv_bfloat16 alpha = __float2bfloat16(1.0f);
    __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);

    cublasSetStream(blas_handle_, stream_);

    // Pre-compute Bx for all timesteps: [T*B, d_model] @ [d_model, d_state]
    blas<__nv_bfloat16>::gemm(blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        d_state_, steps * batch_size_, d_model_,
        &alpha, B, d_state_, x, d_model_,
        &beta_zero, Bx_all, d_state_);

    const int block = 256;
    const int grid_state = (BS + block - 1) / block;
    const int grid_model = (BD + block - 1) / block;

    for (int t = 0; t < steps; t++) {
        __nv_bfloat16* h_prev = h + t * BS;
        __nv_bfloat16* h_t = h + (t + 1) * BS;
        __nv_bfloat16* Bx_t = Bx_all + t * BS;
        __nv_bfloat16* v_t = training_ ? v + t * BS : nullptr;
        const __nv_bfloat16* z_t = z + t * BD;
        __nv_bfloat16* out_t = output + t * BD;

        // h_t = tanh(A * h_prev + Bx_t)
        DiagRecurrenceTanh_BF16<<<grid_state, block, 0, stream_>>>(
            BS, A, Bx_t, h_prev, h_t, v_t, d_state_);

        // Cy = h_t @ C
        blas<__nv_bfloat16>::gemm(blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            d_model_, batch_size_, d_state_,
            &alpha, C, d_model_, h_t, d_state_,
            &beta_zero, Cy, d_model_);

        // output = Cy * silu(z)
        GateSilu_BF16<<<grid_model, block, 0, stream_>>>(BD, Cy, z_t, out_t);
    }
}

// =============================================================================
// Backward - BF16
// =============================================================================

template<>
DiagonalStateElmanBackward<__nv_bfloat16>::DiagonalStateElmanBackward(
    int batch_size, int d_model, int d_state,
    const cublasHandle_t& blas_handle, const cudaStream_t& stream)
    : batch_size_(batch_size), d_model_(d_model), d_state_(d_state),
      blas_handle_(blas_handle), stream_(stream) {}

template<>
void DiagonalStateElmanBackward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* B,
    const __nv_bfloat16* C,
    const __nv_bfloat16* A,
    const __nv_bfloat16* x,
    const __nv_bfloat16* z,
    const __nv_bfloat16* h,
    const __nv_bfloat16* v,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* dx,
    __nv_bfloat16* dz,
    __nv_bfloat16* dB,
    __nv_bfloat16* dC,
    __nv_bfloat16* dA,
    __nv_bfloat16* workspace) {

    const int BS = batch_size_ * d_state_;
    const int BD = batch_size_ * d_model_;

    // Workspace: [dBx: T*BS] [dh_rec: BS] [dy: BD] [Cy: BD] [dA_f: d_state floats]
    __nv_bfloat16* dBx_all = workspace;
    __nv_bfloat16* dh_rec = workspace + steps * BS;
    __nv_bfloat16* dy = dh_rec + BS;
    __nv_bfloat16* Cy = dy + BD;
    float* dA_float = reinterpret_cast<float*>(Cy + BD);

    cudaMemsetAsync(dh_rec, 0, BS * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dA_float, 0, d_state_ * sizeof(float), stream_);
    cudaMemsetAsync(dC, 0, d_state_ * d_model_ * sizeof(__nv_bfloat16), stream_);

    __nv_bfloat16 alpha = __float2bfloat16(1.0f);
    __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);
    __nv_bfloat16 beta_one = __float2bfloat16(1.0f);

    cublasSetStream(blas_handle_, stream_);

    const int block = 256;
    const int grid_state = (BS + block - 1) / block;
    const int grid_model = (BD + block - 1) / block;

    for (int t = steps - 1; t >= 0; t--) {
        const __nv_bfloat16* h_t = h + (t + 1) * BS;
        const __nv_bfloat16* h_prev = h + t * BS;
        const __nv_bfloat16* v_t = v + t * BS;
        const __nv_bfloat16* z_t = z + t * BD;
        const __nv_bfloat16* dout_t = d_output + t * BD;
        __nv_bfloat16* dz_t = dz + t * BD;
        __nv_bfloat16* dBx_t = dBx_all + t * BS;

        // Compute Cy = h_t @ C for gate backward
        blas<__nv_bfloat16>::gemm(blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            d_model_, batch_size_, d_state_,
            &alpha, C, d_model_, h_t, d_state_,
            &beta_zero, Cy, d_model_);

        // Gate backward: dy, dz
        GateBackward_BF16<<<grid_model, block, 0, stream_>>>(BD, Cy, z_t, dout_t, dy, dz_t);

        // dC += h_t.T @ dy (accumulate)
        blas<__nv_bfloat16>::gemm(blas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_T,
            d_model_, d_state_, batch_size_,
            &alpha, dy, d_model_, h_t, d_state_,
            &beta_one, dC, d_model_);

        // dh = C.T @ dy
        __nv_bfloat16* dh_temp = dBx_t;  // Reuse dBx_t temporarily
        blas<__nv_bfloat16>::gemm(blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            d_state_, batch_size_, d_model_,
            &alpha, C, d_model_, dy, d_model_,
            &beta_zero, dh_temp, d_state_);

        // Backward through tanh: dv, dh_rec, dA accumulation
        TanhBackward_BF16<<<grid_state, block, 0, stream_>>>(
            BS, A, v_t, dh_temp, (t < steps - 1) ? dh_rec : nullptr,
            h_prev, dBx_t, dh_rec, dA_float, d_state_);
    }

    // Convert dA_float to bf16
    const int grid_A = (d_state_ + block - 1) / block;
    FloatToBF16<<<grid_A, block, 0, stream_>>>(d_state_, dA_float, dA);

    // dB = x.T @ dBx_all
    blas<__nv_bfloat16>::gemm(blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        d_state_, d_model_, steps * batch_size_,
        &alpha, dBx_all, d_state_, x, d_model_,
        &beta_zero, dB, d_state_);

    // dx = dBx_all @ B.T
    blas<__nv_bfloat16>::gemm(blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        d_model_, steps * batch_size_, d_state_,
        &alpha, B, d_state_, dBx_all, d_state_,
        &beta_zero, dx, d_model_);
}

// =============================================================================
// Generic templates (not implemented)
// =============================================================================

template<typename T>
DiagonalStateElmanForward<T>::DiagonalStateElmanForward(
    bool training, int batch_size, int d_model, int d_state,
    const cublasHandle_t& blas_handle, const cudaStream_t& stream)
    : training_(training), batch_size_(batch_size), d_model_(d_model),
      d_state_(d_state), blas_handle_(blas_handle), stream_(stream) {}

template<typename T>
void DiagonalStateElmanForward<T>::Run(int, const T*, const T*, const T*,
    const T*, const T*, T*, T*, T*, T*) {}

template<typename T>
DiagonalStateElmanBackward<T>::DiagonalStateElmanBackward(
    int batch_size, int d_model, int d_state,
    const cublasHandle_t& blas_handle, const cudaStream_t& stream)
    : batch_size_(batch_size), d_model_(d_model), d_state_(d_state),
      blas_handle_(blas_handle), stream_(stream) {}

template<typename T>
void DiagonalStateElmanBackward<T>::Run(int, const T*, const T*, const T*,
    const T*, const T*, const T*, const T*, const T*, T*, T*, T*, T*, T*, T*) {}

template struct DiagonalStateElmanForward<__nv_bfloat16>;
template struct DiagonalStateElmanForward<__half>;
template struct DiagonalStateElmanForward<float>;
template struct DiagonalStateElmanForward<double>;

template struct DiagonalStateElmanBackward<__nv_bfloat16>;
template struct DiagonalStateElmanBackward<__half>;
template struct DiagonalStateElmanBackward<float>;
template struct DiagonalStateElmanBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
