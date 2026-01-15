// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E70: Matrix Linear Elman - Square Matrix State with E42-style Linear Update
//
// Architecture (E42-style - LINEAR recurrence):
// State S is [B, n_state, n_state] square matrix
//
// Forward per timestep:
//     k_t = x @ W_k.T                      # [B, n_state]
//     v_t = x @ W_v.T                      # [B, n_state]
//     q_t = x @ W_q.T                      # [B, n_state]
//
//     S = decay * S + outer(v, k)          # [B, n_state, n_state] - LINEAR, no tanh!
//
//     out = S @ q                          # [B, n_state]
//     out = out * silu(out)                # Self-gating output (ONLY nonlinearity)
//
// Key Properties (E42-style):
// - Square matrix state S ∈ ℝ^(n×n) enables rich associative storage
// - LINEAR recurrence: NO tanh! Better gradient flow (E42's key insight)
// - Self-gating output h * silu(h) is the ONLY nonlinearity
// - Stability via spectral norm on W_k, W_v (done in Python)
// - Decay < 1 controls memory retention
//
// Key Optimization: Batch k, v, q projections upfront (3 batched GEMMs)

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
// Forward Kernels
// =============================================================================

// State update: S_new = decay * S_prev + outer(v, k)
// E42-style LINEAR update - NO TANH!
// S: [B, N, N], v: [B, N], k: [B, N] -> S_new: [B, N, N]
// decay is a scalar
template<typename T>
__global__ void StateUpdateLinearKernel(
    const int batch_size,
    const int N,   // n_state
    const float decay,
    const T* __restrict__ S_prev,    // [B, N, N]
    const T* __restrict__ v,         // [B, N]
    const T* __restrict__ k,         // [B, N]
    T* __restrict__ S_new) {         // [B, N, N]

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N * N;

    if (idx < total) {
        const int j = idx % N;                    // column
        const int i = (idx / N) % N;              // row
        const int b = idx / (N * N);              // batch

        const int bi_idx = b * N + i;
        const int bj_idx = b * N + j;

        float s_prev_val = static_cast<float>(S_prev[idx]);
        float v_val = static_cast<float>(v[bi_idx]);
        float k_val = static_cast<float>(k[bj_idx]);

        // S_new = decay * S_prev + v * k  (LINEAR - no tanh!)
        float s_new_val = decay * s_prev_val + v_val * k_val;
        S_new[idx] = static_cast<T>(s_new_val);  // NO tanh!
    }
}

__global__ void StateUpdateLinearKernel_BF16(
    const int batch_size,
    const int N,
    const float decay,
    const __nv_bfloat16* __restrict__ S_prev,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ S_new) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N * N;

    if (idx < total) {
        const int j = idx % N;
        const int i = (idx / N) % N;
        const int b = idx / (N * N);

        const int bi_idx = b * N + i;
        const int bj_idx = b * N + j;

        float s_prev_val = __bfloat162float(S_prev[idx]);
        float v_val = __bfloat162float(v[bi_idx]);
        float k_val = __bfloat162float(k[bj_idx]);

        // LINEAR update - no tanh!
        float s_new_val = decay * s_prev_val + v_val * k_val;
        S_new[idx] = __float2bfloat16(s_new_val);  // NO tanh!
    }
}

// Compute out = S @ q using cuBLAS batched GEMM
// S: [B, N, N], q: [B, N] -> out: [B, N]
// This is a matrix-vector multiply, but we use batched GEMM with k=1 for efficiency
// cuBLAS: C = alpha * A * B + beta * C
// For S @ q: out = S @ q (treating q as [N, 1] column vector)
//
// Note: The naive kernel below is kept as fallback but the main path uses cuBLAS
template<typename T>
__global__ void MatVecKernel(
    const int batch_size,
    const int N,
    const T* __restrict__ S,      // [B, N, N]
    const T* __restrict__ q,      // [B, N]
    T* __restrict__ out) {        // [B, N]

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int b = idx / N;
        const int i = idx % N;

        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            // S[b, i, j] = S[b * N * N + i * N + j]
            float s_val = static_cast<float>(S[b * N * N + i * N + j]);
            float q_val = static_cast<float>(q[b * N + j]);
            sum += s_val * q_val;
        }
        out[idx] = static_cast<T>(sum);
    }
}

__global__ void MatVecKernel_BF16(
    const int batch_size,
    const int N,
    const __nv_bfloat16* __restrict__ S,
    const __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ out) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int b = idx / N;
        const int i = idx % N;

        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            float s_val = __bfloat162float(S[b * N * N + i * N + j]);
            float q_val = __bfloat162float(q[b * N + j]);
            sum += s_val * q_val;
        }
        out[idx] = __float2bfloat16(sum);
    }
}

// cuBLAS-based mat-vec: out = S @ q using strided batched GEMM
// S: [B, N, N] row-major, q: [B, N], out: [B, N]
// cuBLAS expects column-major, so we compute: out^T = q^T @ S^T
// Which is: out[b] = S[b] @ q[b]
template<typename T>
inline void MatVecCuBLAS(
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream,
    int batch_size,
    int N,
    const T* S,    // [B, N, N]
    const T* q,    // [B, N]
    T* out) {      // [B, N]

    // cuBLAS is column-major, our tensors are row-major
    // S[b] is [N, N] row-major = [N, N]^T column-major
    // To compute out = S @ q in row-major:
    // out = S @ q  (row-major)
    // = (q^T @ S^T)^T  (column-major view)
    //
    // Using GEMM: C = alpha * op(A) * op(B) + beta * C
    // We want: out = S @ q
    //
    // In cuBLAS column-major with our row-major data:
    // - S is stored as S^T in column-major (N x N)
    // - q is stored as q^T in column-major (1 x N)
    // - out will be stored as out^T in column-major (1 x N)
    //
    // So: out^T = q^T @ S^T  =>  out = S @ q
    // GEMM: C(1,N) = A(1,N) @ B(N,N) = q^T @ S^T
    // m=1, n=N, k=N, A=q, B=S, C=out (all transposed view)
    //
    // Actually simpler: use CUBLAS_OP_T on S to get S^T^T = S
    // Then: out = S @ q as GEMM(N, 1, N): C(N,1) = A(N,N) @ B(N,1)

    static const T alpha_one = static_cast<T>(1.0f);
    static const T beta_zero = static_cast<T>(0.0f);

    // GEMM: out = S @ q
    // m=N, n=1, k=N
    // A=S (N x N), B=q (N x 1), C=out (N x 1)
    // But S is row-major, cuBLAS expects column-major
    // Row-major S[i,j] = Column-major S^T[j,i]
    // So S in memory is S^T to cuBLAS
    // We want out = S @ q, i.e., out = S^T^T @ q
    // cuBLAS: C = op(A) @ op(B)
    // With A=S (stored as S^T), op(A)=CUBLAS_OP_T gives S^T^T = S
    // Result: out = S @ q  (correct!)

    blas<T>::gemmStridedBatched(
        blas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,  // transpose S (row->col), no transpose q
        N, 1, N,                   // m, n, k
        &alpha_one,
        S, N, N * N,               // A=S, lda=N, strideA=N*N
        q, N, N,                   // B=q, ldb=N (as column vector), strideB=N
        &beta_zero,
        out, N, N,                 // C=out, ldc=N, strideC=N
        batch_size);
}

// Self-gate output: output = Sq * silu(Sq)
template<typename T>
__global__ void SelfGateKernel(
    const int n,
    const T* __restrict__ Sq,      // [B, N] (S @ q result)
    T* __restrict__ output) {      // [B, N]

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sq_val = static_cast<float>(Sq[idx]);
        float sigmoid_sq = 1.0f / (1.0f + expf(-sq_val));
        float silu_sq = sq_val * sigmoid_sq;
        output[idx] = static_cast<T>(sq_val * silu_sq);
    }
}

__global__ void SelfGateKernel_BF16(
    const int n,
    const __nv_bfloat16* __restrict__ Sq,
    __nv_bfloat16* __restrict__ output) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sq_val = __bfloat162float(Sq[idx]);
        float sigmoid_sq = 1.0f / (1.0f + __expf(-sq_val));
        float silu_sq = sq_val * sigmoid_sq;
        output[idx] = __float2bfloat16(sq_val * silu_sq);
    }
}

// =============================================================================
// Backward Kernels
// =============================================================================

// Backward through self-gate: output = Sq * silu(Sq) = Sq^2 * sigmoid(Sq)
// d(output)/dSq = silu(Sq) * (2 + Sq * (1 - sigmoid(Sq)))
template<typename T>
__global__ void SelfGateBackwardKernel(
    const int n,
    const T* __restrict__ Sq,          // [B, N]
    const T* __restrict__ d_output,    // [B, N]
    T* __restrict__ d_Sq) {            // [B, N]

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sq_val = static_cast<float>(Sq[idx]);
        float dout = static_cast<float>(d_output[idx]);

        float sigmoid_sq = 1.0f / (1.0f + expf(-sq_val));
        float silu_sq = sq_val * sigmoid_sq;
        float grad_factor = silu_sq * (2.0f + sq_val * (1.0f - sigmoid_sq));

        d_Sq[idx] = static_cast<T>(dout * grad_factor);
    }
}

__global__ void SelfGateBackwardKernel_BF16(
    const int n,
    const __nv_bfloat16* __restrict__ Sq,
    const __nv_bfloat16* __restrict__ d_output,
    __nv_bfloat16* __restrict__ d_Sq) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sq_val = __bfloat162float(Sq[idx]);
        float dout = __bfloat162float(d_output[idx]);

        float sigmoid_sq = 1.0f / (1.0f + __expf(-sq_val));
        float silu_sq = sq_val * sigmoid_sq;
        float grad_factor = silu_sq * (2.0f + sq_val * (1.0f - sigmoid_sq));

        d_Sq[idx] = __float2bfloat16(dout * grad_factor);
    }
}

// Backward through S @ q:
// d_S[b,i,j] += d_Sq[b,i] * q[b,j]
// d_q[b,j] += sum_i(d_Sq[b,i] * S[b,i,j])
template<typename T>
__global__ void MatVecBackwardKernel(
    const int batch_size,
    const int N,
    const T* __restrict__ S,          // [B, N, N]
    const T* __restrict__ q,          // [B, N]
    const T* __restrict__ d_Sq,       // [B, N]
    T* __restrict__ d_S,              // [B, N, N] - add to existing
    float* __restrict__ d_q_f) {      // [B, N] float accumulator

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int b = idx / N;
        const int i = idx % N;

        float d_sq_val = static_cast<float>(d_Sq[idx]);

        // d_S[b,i,j] += d_Sq[b,i] * q[b,j]
        for (int j = 0; j < N; ++j) {
            float q_val = static_cast<float>(q[b * N + j]);
            int s_idx = b * N * N + i * N + j;
            d_S[s_idx] = static_cast<T>(static_cast<float>(d_S[s_idx]) + d_sq_val * q_val);
        }

        // d_q[b,j] += d_Sq[b,i] * S[b,i,j]
        for (int j = 0; j < N; ++j) {
            float s_val = static_cast<float>(S[b * N * N + i * N + j]);
            atomicAdd(&d_q_f[b * N + j], d_sq_val * s_val);
        }
    }
}

__global__ void MatVecBackwardKernel_BF16(
    const int batch_size,
    const int N,
    const __nv_bfloat16* __restrict__ S,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ d_Sq,
    __nv_bfloat16* __restrict__ d_S,
    float* __restrict__ d_q_f) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N;

    if (idx < total) {
        const int b = idx / N;
        const int i = idx % N;

        float d_sq_val = __bfloat162float(d_Sq[idx]);

        for (int j = 0; j < N; ++j) {
            float q_val = __bfloat162float(q[b * N + j]);
            int s_idx = b * N * N + i * N + j;
            d_S[s_idx] = __float2bfloat16(__bfloat162float(d_S[s_idx]) + d_sq_val * q_val);
        }

        for (int j = 0; j < N; ++j) {
            float s_val = __bfloat162float(S[b * N * N + i * N + j]);
            atomicAdd(&d_q_f[b * N + j], d_sq_val * s_val);
        }
    }
}

// Optimized backward kernel for outer product only: d_S[b,i,j] += d_Sq[b,i] * q[b,j]
// This is fully parallel over all B*N*N elements
__global__ void OuterProductAddKernel_BF16(
    const int batch_size,
    const int N,
    const __nv_bfloat16* __restrict__ d_Sq,   // [B, N]
    const __nv_bfloat16* __restrict__ q,       // [B, N]
    __nv_bfloat16* __restrict__ d_S) {         // [B, N, N] - add to existing

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N * N;

    if (idx < total) {
        const int j = idx % N;                    // column
        const int i = (idx / N) % N;              // row
        const int b = idx / (N * N);              // batch

        float d_sq_val = __bfloat162float(d_Sq[b * N + i]);
        float q_val = __bfloat162float(q[b * N + j]);
        float d_s_val = __bfloat162float(d_S[idx]);

        d_S[idx] = __float2bfloat16(d_s_val + d_sq_val * q_val);
    }
}

// cuBLAS-based backward for d_q: d_q = S^T @ d_Sq
// This is faster than the naive atomicAdd-based kernel
template<typename T>
inline void MatVecTransposeCuBLAS(
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream,
    int batch_size,
    int N,
    const T* S,      // [B, N, N]
    const T* d_Sq,   // [B, N]
    T* d_q) {        // [B, N]

    static const T alpha_one = static_cast<T>(1.0f);
    static const T beta_zero = static_cast<T>(0.0f);

    // We want: d_q = S^T @ d_Sq
    // S is row-major, so in cuBLAS column-major view it's S^T
    // Using CUBLAS_OP_N on our row-major S gives S^T in math
    // GEMM: d_q = S^T @ d_Sq where S^T is [N, N] and d_Sq is [N, 1]

    blas<T>::gemmStridedBatched(
        blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,  // no transpose (S stored as S^T), no transpose d_Sq
        N, 1, N,                   // m, n, k
        &alpha_one,
        S, N, N * N,               // A=S (stored as S^T), lda=N, strideA=N*N
        d_Sq, N, N,                // B=d_Sq (column vector), ldb=N, strideB=N
        &beta_zero,
        d_q, N, N,                 // C=d_q, ldc=N, strideC=N
        batch_size);
}

// Backward through state update: S_new = decay * S_prev + outer(v, k)
// E42-style LINEAR update - NO tanh derivative!
// d_S_prev[b,i,j] = d_S_new[b,i,j] * decay
// d_v[b,i] += sum_j(d_S_new[b,i,j] * k[b,j])
// d_k[b,j] += sum_i(d_S_new[b,i,j] * v[b,i])
// d_decay += sum_b,i,j(d_S_new[b,i,j] * S_prev[b,i,j])
template<typename T>
__global__ void StateBackwardLinearKernel(
    const int batch_size,
    const int N,
    const float decay,
    const T* __restrict__ S_prev,     // [B, N, N]
    const T* __restrict__ S_new,      // [B, N, N] (unused now - linear, no tanh)
    const T* __restrict__ v,          // [B, N]
    const T* __restrict__ k,          // [B, N]
    const T* __restrict__ d_S_new,    // [B, N, N]
    T* __restrict__ d_S_prev,         // [B, N, N]
    float* __restrict__ d_v_f,        // [B, N] float accumulator
    float* __restrict__ d_k_f,        // [B, N] float accumulator
    float* __restrict__ d_decay_f) {  // scalar float accumulator

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N * N;

    if (idx < total) {
        const int j = idx % N;
        const int i = (idx / N) % N;
        const int b = idx / (N * N);

        const int bi_idx = b * N + i;
        const int bj_idx = b * N + j;

        float ds_new = static_cast<float>(d_S_new[idx]);

        // LINEAR backward: no tanh derivative! Gradient flows directly.
        // d_S_prev = d_S_new * decay
        d_S_prev[idx] = static_cast<T>(ds_new * decay);

        // d_v[b,i] += d_S_new * k[b,j]
        float k_val = static_cast<float>(k[bj_idx]);
        atomicAdd(&d_v_f[bi_idx], ds_new * k_val);

        // d_k[b,j] += d_S_new * v[b,i]
        float v_val = static_cast<float>(v[bi_idx]);
        atomicAdd(&d_k_f[bj_idx], ds_new * v_val);

        // d_decay += d_S_new * S_prev
        float s_prev_val = static_cast<float>(S_prev[idx]);
        atomicAdd(d_decay_f, ds_new * s_prev_val);
    }
}

__global__ void StateBackwardLinearKernel_BF16(
    const int batch_size,
    const int N,
    const float decay,
    const __nv_bfloat16* __restrict__ S_prev,
    const __nv_bfloat16* __restrict__ S_new,  // unused - linear, no tanh
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ d_S_new,
    __nv_bfloat16* __restrict__ d_S_prev,
    float* __restrict__ d_v_f,
    float* __restrict__ d_k_f,
    float* __restrict__ d_decay_f) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * N * N;

    if (idx < total) {
        const int j = idx % N;
        const int i = (idx / N) % N;
        const int b = idx / (N * N);

        const int bi_idx = b * N + i;
        const int bj_idx = b * N + j;

        float ds_new = __bfloat162float(d_S_new[idx]);

        // LINEAR backward: no tanh derivative!
        d_S_prev[idx] = __float2bfloat16(ds_new * decay);

        float k_val = __bfloat162float(k[bj_idx]);
        atomicAdd(&d_v_f[bi_idx], ds_new * k_val);

        float v_val = __bfloat162float(v[bi_idx]);
        atomicAdd(&d_k_f[bj_idx], ds_new * v_val);

        float s_prev_val = __bfloat162float(S_prev[idx]);
        atomicAdd(d_decay_f, ds_new * s_prev_val);
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
// E70 Matrix Linear Forward - BF16 Specialization
// =============================================================================

template<>
E70MatrixLinearForward<__nv_bfloat16>::E70MatrixLinearForward(
    bool training,
    int batch_size,
    int dim,
    int n_state,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : training_(training),
      batch_size_(batch_size),
      dim_(dim),
      n_state_(n_state),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void E70MatrixLinearForward<__nv_bfloat16>::Run(
    int steps,
    float decay,
    const __nv_bfloat16* W_k,       // [n_state, dim]
    const __nv_bfloat16* W_v,       // [n_state, dim]
    const __nv_bfloat16* W_q,       // [n_state, dim]
    const __nv_bfloat16* x,         // [T, B, dim] input
    __nv_bfloat16* S,               // [T+1, B, n_state, n_state] state matrices
    __nv_bfloat16* output,          // [T, B, n_state] output
    __nv_bfloat16* k_cache,         // [T, B, n_state] for backward
    __nv_bfloat16* v_cache,         // [T, B, n_state] for backward
    __nv_bfloat16* q_cache,         // [T, B, n_state] for backward
    __nv_bfloat16* Sq_cache,        // [T, B, n_state] for backward (S @ q result)
    __nv_bfloat16* workspace) {

    static const __nv_bfloat16 alpha_one = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);

    const int BD = batch_size_ * dim_;
    const int BN = batch_size_ * n_state_;
    const int BNN = batch_size_ * n_state_ * n_state_;
    const int block_size = 256;

    // Workspace layout:
    // k_all: [T, B, n_state]
    // v_all: [T, B, n_state]
    // q_all: [T, B, n_state]
    // Sq_tmp: [B, n_state]
    __nv_bfloat16* k_all = workspace;
    __nv_bfloat16* v_all = k_all + steps * BN;
    __nv_bfloat16* q_all = v_all + steps * BN;
    __nv_bfloat16* Sq_tmp = q_all + steps * BN;

    // Pre-compute all x projections in batched GEMMs
    // k_all = x @ W_k.T  [T*B, n_state]
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        n_state_, steps * batch_size_, dim_,
        &alpha_one,
        W_k, dim_,
        x, dim_,
        &beta_zero,
        k_all, n_state_);

    // v_all = x @ W_v.T  [T*B, n_state]
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        n_state_, steps * batch_size_, dim_,
        &alpha_one,
        W_v, dim_,
        x, dim_,
        &beta_zero,
        v_all, n_state_);

    // q_all = x @ W_q.T  [T*B, n_state]
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        n_state_, steps * batch_size_, dim_,
        &alpha_one,
        W_q, dim_,
        x, dim_,
        &beta_zero,
        q_all, n_state_);

    // Clamp decay to valid range
    float clamped_decay = fminf(fmaxf(decay, 0.0f), 0.999f);

    // Process each timestep sequentially (cannot parallelize - S-dependent)
    for (int t = 0; t < steps; ++t) {
        const __nv_bfloat16* k_t = k_all + t * BN;
        const __nv_bfloat16* v_t = v_all + t * BN;
        const __nv_bfloat16* q_t = q_all + t * BN;
        const __nv_bfloat16* S_prev = S + t * BNN;
        __nv_bfloat16* S_t = S + (t + 1) * BNN;
        __nv_bfloat16* out_t = output + t * BN;

        // Cache pointers for backward
        __nv_bfloat16* Sq_c = training_ ? (Sq_cache + t * BN) : Sq_tmp;

        // Copy k, v, q to cache if training
        if (training_) {
            cudaMemcpyAsync(k_cache + t * BN, k_t, BN * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream_);
            cudaMemcpyAsync(v_cache + t * BN, v_t, BN * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream_);
            cudaMemcpyAsync(q_cache + t * BN, q_t, BN * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream_);
        }

        // 1. State update: S_new = decay * S_prev + outer(v, k)  [LINEAR - no tanh!]
        StateUpdateLinearKernel_BF16<<<(BNN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_state_, clamped_decay, S_prev, v_t, k_t, S_t);

        // 2. Compute Sq = S_new @ q using cuBLAS batched GEMM
        // Much faster than naive MatVecKernel for n_state >= 32
        MatVecCuBLAS<__nv_bfloat16>(
            blas_handle_, stream_, batch_size_, n_state_, S_t, q_t, Sq_c);

        // 3. Self-gate output: output = Sq * silu(Sq)
        SelfGateKernel_BF16<<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            BN, Sq_c, out_t);
    }
}

// =============================================================================
// E70 Matrix Linear Backward - BF16 Specialization
// =============================================================================

template<>
E70MatrixLinearBackward<__nv_bfloat16>::E70MatrixLinearBackward(
    int batch_size,
    int dim,
    int n_state,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      n_state_(n_state),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void E70MatrixLinearBackward<__nv_bfloat16>::Run(
    int steps,
    float decay,
    const __nv_bfloat16* W_k,
    const __nv_bfloat16* W_v,
    const __nv_bfloat16* W_q,
    const __nv_bfloat16* x,
    const __nv_bfloat16* S,
    const __nv_bfloat16* k_cache,
    const __nv_bfloat16* v_cache,
    const __nv_bfloat16* q_cache,
    const __nv_bfloat16* Sq_cache,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* dx,
    __nv_bfloat16* dW_k,
    __nv_bfloat16* dW_v,
    __nv_bfloat16* dW_q,
    __nv_bfloat16* d_decay_out,
    __nv_bfloat16* workspace) {

    static const __nv_bfloat16 alpha_one = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);
    static const __nv_bfloat16 beta_one = __float2bfloat16(1.0f);

    const int BD = batch_size_ * dim_;
    const int BN = batch_size_ * n_state_;
    const int BNN = batch_size_ * n_state_ * n_state_;
    const int block_size = 256;

    // Clamp decay
    float clamped_decay = fminf(fmaxf(decay, 0.0f), 0.999f);

    // Workspace layout:
    // d_S: [B, n_state, n_state] - current state gradient
    // d_S_prev: [B, n_state, n_state] - state gradient for previous timestep
    // d_Sq: [B, n_state]
    // d_k_all: [T, B, n_state]
    // d_v_all: [T, B, n_state]
    // d_q_all: [T, B, n_state]
    // d_k_f: [B, n_state] float
    // d_v_f: [B, n_state] float
    // d_q_f: [B, n_state] float
    // d_decay_f: [1] float
    __nv_bfloat16* d_S = workspace;
    __nv_bfloat16* d_S_prev = d_S + BNN;
    __nv_bfloat16* d_Sq = d_S_prev + BNN;
    __nv_bfloat16* d_k_all = d_Sq + BN;
    __nv_bfloat16* d_v_all = d_k_all + steps * BN;
    __nv_bfloat16* d_q_all = d_v_all + steps * BN;

    float* float_ws = reinterpret_cast<float*>(d_q_all + steps * BN);
    float* d_k_f = float_ws;
    float* d_v_f = d_k_f + BN;
    float* d_q_f = d_v_f + BN;
    float* d_decay_f = d_q_f + BN;

    // Initialize gradients to zero
    cudaMemsetAsync(d_S, 0, BNN * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_k, 0, n_state_ * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_v, 0, n_state_ * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(dW_q, 0, n_state_ * dim_ * sizeof(__nv_bfloat16), stream_);
    cudaMemsetAsync(d_decay_f, 0, sizeof(float), stream_);

    // BPTT loop
    for (int t = steps - 1; t >= 0; --t) {
        const __nv_bfloat16* S_t = S + (t + 1) * BNN;
        const __nv_bfloat16* S_prev = S + t * BNN;
        const __nv_bfloat16* k_t = k_cache + t * BN;
        const __nv_bfloat16* v_t = v_cache + t * BN;
        const __nv_bfloat16* q_t = q_cache + t * BN;
        const __nv_bfloat16* Sq_t = Sq_cache + t * BN;
        const __nv_bfloat16* d_out_t = d_output + t * BN;

        __nv_bfloat16* d_k_t = d_k_all + t * BN;
        __nv_bfloat16* d_v_t = d_v_all + t * BN;
        __nv_bfloat16* d_q_t = d_q_all + t * BN;

        // Zero per-timestep float accumulators (only for v and k now - q uses cuBLAS)
        cudaMemsetAsync(d_k_f, 0, BN * sizeof(float), stream_);
        cudaMemsetAsync(d_v_f, 0, BN * sizeof(float), stream_);

        // 1. Backward through self-gate: output = Sq * silu(Sq)
        SelfGateBackwardKernel_BF16<<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            BN, Sq_t, d_out_t, d_Sq);

        // 2. Backward through S @ q using optimized kernels
        // 2a. d_S[b,i,j] += d_Sq[b,i] * q[b,j] (outer product - fully parallel)
        OuterProductAddKernel_BF16<<<(BNN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_state_, d_Sq, q_t, d_S);

        // 2b. d_q = S^T @ d_Sq using cuBLAS (much faster than atomicAdd)
        MatVecTransposeCuBLAS<__nv_bfloat16>(
            blas_handle_, stream_, batch_size_, n_state_, S_t, d_Sq, d_q_t);

        // 3. Backward through state update: S_new = decay * S_prev + outer(v, k) [LINEAR - no tanh!]
        StateBackwardLinearKernel_BF16<<<(BNN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_state_, clamped_decay, S_prev, S_t, v_t, k_t, d_S,
            d_S_prev, d_v_f, d_k_f, d_decay_f);

        // Copy float accumulators to output (v and k only - q done by cuBLAS)
        CopyFloatToT<__nv_bfloat16><<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            BN, d_k_f, d_k_t);
        CopyFloatToT<__nv_bfloat16><<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            BN, d_v_f, d_v_t);

        // Swap d_S and d_S_prev for next iteration
        std::swap(d_S, d_S_prev);
    }

    // Batch compute weight gradients and dx from accumulated projection gradients
    // dx = d_k_all @ W_k + d_v_all @ W_v + d_q_all @ W_q

    // dx from k projection: dx = d_k_all @ W_k
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, n_state_,
        &alpha_one,
        W_k, dim_,
        d_k_all, n_state_,
        &beta_zero,
        dx, dim_);

    // dx += d_v_all @ W_v
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, n_state_,
        &alpha_one,
        W_v, dim_,
        d_v_all, n_state_,
        &beta_one,
        dx, dim_);

    // dx += d_q_all @ W_q
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim_, steps * batch_size_, n_state_,
        &alpha_one,
        W_q, dim_,
        d_q_all, n_state_,
        &beta_one,
        dx, dim_);

    // dW_k = x.T @ d_k_all
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, n_state_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        d_k_all, n_state_,
        &beta_one,
        dW_k, dim_);

    // dW_v = x.T @ d_v_all
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, n_state_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        d_v_all, n_state_,
        &beta_one,
        dW_v, dim_);

    // dW_q = x.T @ d_q_all
    blas<__nv_bfloat16>::gemm(
        blas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        dim_, n_state_, steps * batch_size_,
        &alpha_one,
        x, dim_,
        d_q_all, n_state_,
        &beta_one,
        dW_q, dim_);

    // Copy decay gradient
    CopyFloatToT<__nv_bfloat16><<<1, 1, 0, stream_>>>(1, d_decay_f, d_decay_out);
}

// =============================================================================
// Generic Template Implementations
// =============================================================================

template<typename T>
E70MatrixLinearForward<T>::E70MatrixLinearForward(
    bool training,
    int batch_size,
    int dim,
    int n_state,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : training_(training),
      batch_size_(batch_size),
      dim_(dim),
      n_state_(n_state),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void E70MatrixLinearForward<T>::Run(
    int steps,
    float decay,
    const T* W_k,
    const T* W_v,
    const T* W_q,
    const T* x,
    T* S,
    T* output,
    T* k_cache,
    T* v_cache,
    T* q_cache,
    T* Sq_cache,
    T* workspace) {

    static const T alpha_one = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int BN = batch_size_ * n_state_;
    const int BNN = batch_size_ * n_state_ * n_state_;
    const int block_size = 256;

    T* k_all = workspace;
    T* v_all = k_all + steps * BN;
    T* q_all = v_all + steps * BN;
    T* Sq_tmp = q_all + steps * BN;

    // Pre-compute projections
    blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        n_state_, steps * batch_size_, dim_, &alpha_one, W_k, dim_, x, dim_, &beta_zero, k_all, n_state_);
    blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        n_state_, steps * batch_size_, dim_, &alpha_one, W_v, dim_, x, dim_, &beta_zero, v_all, n_state_);
    blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        n_state_, steps * batch_size_, dim_, &alpha_one, W_q, dim_, x, dim_, &beta_zero, q_all, n_state_);

    float clamped_decay = fminf(fmaxf(decay, 0.0f), 0.999f);

    for (int t = 0; t < steps; ++t) {
        const T* k_t = k_all + t * BN;
        const T* v_t = v_all + t * BN;
        const T* q_t = q_all + t * BN;
        const T* S_prev = S + t * BNN;
        T* S_t = S + (t + 1) * BNN;
        T* out_t = output + t * BN;
        T* Sq_c = training_ ? (Sq_cache + t * BN) : Sq_tmp;

        if (training_) {
            cudaMemcpyAsync(k_cache + t * BN, k_t, BN * sizeof(T), cudaMemcpyDeviceToDevice, stream_);
            cudaMemcpyAsync(v_cache + t * BN, v_t, BN * sizeof(T), cudaMemcpyDeviceToDevice, stream_);
            cudaMemcpyAsync(q_cache + t * BN, q_t, BN * sizeof(T), cudaMemcpyDeviceToDevice, stream_);
        }

        StateUpdateLinearKernel<T><<<(BNN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_state_, clamped_decay, S_prev, v_t, k_t, S_t);

        MatVecKernel<T><<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            batch_size_, n_state_, S_t, q_t, Sq_c);

        SelfGateKernel<T><<<(BN + block_size - 1) / block_size, block_size, 0, stream_>>>(
            BN, Sq_c, out_t);
    }
}

template<typename T>
E70MatrixLinearBackward<T>::E70MatrixLinearBackward(
    int batch_size,
    int dim,
    int n_state,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      n_state_(n_state),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void E70MatrixLinearBackward<T>::Run(
    int steps,
    float decay,
    const T* W_k,
    const T* W_v,
    const T* W_q,
    const T* x,
    const T* S,
    const T* k_cache,
    const T* v_cache,
    const T* q_cache,
    const T* Sq_cache,
    const T* d_output,
    T* dx,
    T* dW_k,
    T* dW_v,
    T* dW_q,
    T* d_decay_out,
    T* workspace) {

    // Placeholder - follows BF16 pattern
    cudaMemsetAsync(dx, 0, steps * batch_size_ * dim_ * sizeof(T), stream_);
}

// Explicit template instantiations
template struct E70MatrixLinearForward<__half>;
template struct E70MatrixLinearForward<float>;
template struct E70MatrixLinearForward<double>;

template struct E70MatrixLinearBackward<__half>;
template struct E70MatrixLinearBackward<float>;
template struct E70MatrixLinearBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
