// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E74 Fused: Diagonal State Minimal RNN - Optimized with persistent kernel
//
// Key optimization: Fuse all timesteps into a SINGLE kernel launch to eliminate
// the ~100K kernel launch overhead that was destroying performance.
//
// Architecture (same as E74):
//     Diagonal state: s ∈ [B, n_state]
//     Delta update: s_new = f(s*(1-k²) + v*k)
//     Simple update: s_new = f(α*s + v*k)
//     Output: out = s * silu(s)
//
// The key insight: E74's recurrence is O(n) element-wise ops, not O(n²) like
// matrix models. We can process all timesteps in a single kernel where each
// thread handles one element of state across all timesteps.

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
// FUSED Forward Kernel - Process ALL timesteps in ONE kernel launch
// =============================================================================

// Each thread processes one element of [B, n_state] across ALL timesteps
// Thread idx maps to (batch_idx, state_idx)
template<bool USE_TANH, bool IS_DELTA>
__global__ void E74FusedForwardKernel_BF16(
    const int steps,           // Number of timesteps
    const int batch_size,
    const int n_state,
    const float decay,         // For simple update
    const __nv_bfloat16* __restrict__ k_all,  // [T, B, n] pre-computed projections
    const __nv_bfloat16* __restrict__ v_all,  // [T, B, n] (may alias k_all for tied)
    __nv_bfloat16* __restrict__ s_all,        // [T+1, B, n] state history
    __nv_bfloat16* __restrict__ output,       // [T, B, n]
    __nv_bfloat16* __restrict__ pre_nonlin_cache,  // [T, B, n] for backward
    __nv_bfloat16* __restrict__ s_cache)      // [T, B, n] pre-self-gate cache
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int BN = batch_size * n_state;

    if (idx >= BN) return;

    // Load initial state
    float s = __bfloat162float(s_all[idx]);  // s_all[0, b, n]

    // Process all timesteps sequentially within this thread
    for (int t = 0; t < steps; ++t) {
        const int offset = t * BN + idx;

        float k_val = __bfloat162float(k_all[offset]);
        float v_val = __bfloat162float(v_all[offset]);

        // Compute update
        float pre_nonlin;
        if constexpr (IS_DELTA) {
            // Delta rule: s_new = s*(1-k²) + v*k
            float k_sq = k_val * k_val;
            pre_nonlin = s * (1.0f - k_sq) + v_val * k_val;
        } else {
            // Simple rule: s_new = α*s + v*k
            pre_nonlin = decay * s + v_val * k_val;
        }

        // Cache for backward
        if (pre_nonlin_cache) {
            pre_nonlin_cache[offset] = __float2bfloat16(pre_nonlin);
        }

        // Apply nonlinearity
        if constexpr (USE_TANH) {
            s = tanhf(pre_nonlin);
        } else {
            s = pre_nonlin;
        }

        // Write state to history
        s_all[(t + 1) * BN + idx] = __float2bfloat16(s);

        // Cache s for self-gate backward
        if (s_cache) {
            s_cache[offset] = __float2bfloat16(s);
        }

        // Self-gate output: out = s * silu(s)
        float sigmoid_s = 1.0f / (1.0f + __expf(-s));
        float silu_s = s * sigmoid_s;
        float out = s * silu_s;

        output[offset] = __float2bfloat16(out);
    }
}

// =============================================================================
// FUSED Backward Kernel - Process ALL timesteps in ONE kernel launch (BPTT)
// =============================================================================

template<bool USE_TANH, bool IS_DELTA>
__global__ void E74FusedBackwardKernel_BF16(
    const int steps,
    const int batch_size,
    const int n_state,
    const float decay,
    const __nv_bfloat16* __restrict__ k_all,
    const __nv_bfloat16* __restrict__ v_all,
    const __nv_bfloat16* __restrict__ s_all,
    const __nv_bfloat16* __restrict__ s_cache,
    const __nv_bfloat16* __restrict__ pre_nonlin_cache,
    const __nv_bfloat16* __restrict__ d_output,
    __nv_bfloat16* __restrict__ d_k_all,
    __nv_bfloat16* __restrict__ d_v_all)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int BN = batch_size * n_state;

    if (idx >= BN) return;

    // Recurrent gradient accumulator
    float d_s = 0.0f;

    // BPTT: process timesteps in reverse
    for (int t = steps - 1; t >= 0; --t) {
        const int offset = t * BN + idx;

        // Load cached values
        float s_val = __bfloat162float(s_cache[offset]);
        float dout = __bfloat162float(d_output[offset]);

        // Backward through self-gate: out = s * silu(s)
        // d(out)/d(s) = silu(s) * (2 + s*(1-sigmoid(s)))
        float sigmoid_s = 1.0f / (1.0f + __expf(-s_val));
        float silu_s = s_val * sigmoid_s;
        float grad_factor = silu_s * (2.0f + s_val * (1.0f - sigmoid_s));
        float d_s_new = dout * grad_factor + d_s;

        // Backward through nonlinearity
        float d_pre;
        if constexpr (USE_TANH) {
            float pre = __bfloat162float(pre_nonlin_cache[offset]);
            float tanh_pre = tanhf(pre);
            d_pre = d_s_new * (1.0f - tanh_pre * tanh_pre);
        } else {
            d_pre = d_s_new;
        }

        // Load k, v for this timestep
        float k_val = __bfloat162float(k_all[offset]);
        float v_val = __bfloat162float(v_all[offset]);

        // Backward through update
        float dk, dv;
        if constexpr (IS_DELTA) {
            // Delta: pre = s*(1-k²) + v*k
            // Need s_prev for gradient
            float s_prev = (t > 0) ?
                __bfloat162float(s_all[t * BN + idx]) :
                __bfloat162float(s_all[idx]);

            float k_sq = k_val * k_val;

            // d/ds_prev = (1 - k²)
            d_s = d_pre * (1.0f - k_sq);

            // d/dk = s_prev*(-2k) + v = v - 2*s_prev*k
            dk = d_pre * (v_val - 2.0f * s_prev * k_val);

            // d/dv = k
            dv = d_pre * k_val;
        } else {
            // Simple: pre = α*s + v*k
            // d/ds_prev = α
            d_s = d_pre * decay;

            // d/dk = v
            dk = d_pre * v_val;

            // d/dv = k
            dv = d_pre * k_val;
        }

        // Store gradients
        d_k_all[offset] = __float2bfloat16(dk);
        d_v_all[offset] = __float2bfloat16(dv);
    }
}

}  // anonymous namespace


namespace hasty {
namespace v0 {
namespace elman_ladder {

// =============================================================================
// E74 Fused Forward Implementation
// =============================================================================

template<>
E74FusedForward<__nv_bfloat16>::E74FusedForward(
    bool training,
    int batch_size,
    int n_state,
    int dim,
    int proj_type,
    bool use_tanh,
    bool is_delta,
    float decay,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : training_(training),
      batch_size_(batch_size),
      n_state_(n_state),
      dim_(dim),
      proj_type_(proj_type),
      use_tanh_(use_tanh),
      is_delta_(is_delta),
      decay_(decay),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void E74FusedForward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_kvq,
    const __nv_bfloat16* W_kq,
    const __nv_bfloat16* W_v,
    const __nv_bfloat16* W_k,
    const __nv_bfloat16* W_q,
    const __nv_bfloat16* W_z,
    const __nv_bfloat16* x,
    __nv_bfloat16* s,
    __nv_bfloat16* output,
    __nv_bfloat16* k_cache,
    __nv_bfloat16* v_cache,
    __nv_bfloat16* q_cache,
    __nv_bfloat16* z_cache,
    __nv_bfloat16* pre_nonlin_cache,
    __nv_bfloat16* s_cache,
    __nv_bfloat16* workspace)
{
    static const __nv_bfloat16 alpha_one = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);

    const int BN = batch_size_ * n_state_;
    const int block_size = 256;
    const int num_blocks = (BN + block_size - 1) / block_size;

    // Workspace for projections
    __nv_bfloat16* k_all = training_ ? k_cache : workspace;
    __nv_bfloat16* v_all = training_ ? v_cache : (workspace + steps * BN);

    // =========================================================================
    // Batch all projections upfront
    // =========================================================================

    if (proj_type_ == 0) {  // TIED_KVQ: k = v = q
        blas<__nv_bfloat16>::gemm(
            blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            n_state_, steps * batch_size_, dim_,
            &alpha_one, W_kvq, dim_, x, dim_,
            &beta_zero, k_all, n_state_);
        // v_all = k_all (no copy needed, just use same pointer)
        v_all = k_all;
    }
    else if (proj_type_ == 1) {  // TIED_KQ: k = q, separate v
        blas<__nv_bfloat16>::gemm(
            blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            n_state_, steps * batch_size_, dim_,
            &alpha_one, W_kq, dim_, x, dim_,
            &beta_zero, k_all, n_state_);
        blas<__nv_bfloat16>::gemm(
            blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            n_state_, steps * batch_size_, dim_,
            &alpha_one, W_v, dim_, x, dim_,
            &beta_zero, v_all, n_state_);
    }
    else if (proj_type_ == 2) {  // NO_Z: k, v, q separate
        blas<__nv_bfloat16>::gemm(
            blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            n_state_, steps * batch_size_, dim_,
            &alpha_one, W_k, dim_, x, dim_,
            &beta_zero, k_all, n_state_);
        blas<__nv_bfloat16>::gemm(
            blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            n_state_, steps * batch_size_, dim_,
            &alpha_one, W_v, dim_, x, dim_,
            &beta_zero, v_all, n_state_);
        // q_cache handled separately if needed
        if (training_ && q_cache) {
            blas<__nv_bfloat16>::gemm(
                blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                n_state_, steps * batch_size_, dim_,
                &alpha_one, W_q, dim_, x, dim_,
                &beta_zero, q_cache, n_state_);
        }
    }
    else {  // FULL: k, v, q, z all separate
        blas<__nv_bfloat16>::gemm(
            blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            n_state_, steps * batch_size_, dim_,
            &alpha_one, W_k, dim_, x, dim_,
            &beta_zero, k_all, n_state_);
        blas<__nv_bfloat16>::gemm(
            blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            n_state_, steps * batch_size_, dim_,
            &alpha_one, W_v, dim_, x, dim_,
            &beta_zero, v_all, n_state_);
        if (training_) {
            if (q_cache) {
                blas<__nv_bfloat16>::gemm(
                    blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                    n_state_, steps * batch_size_, dim_,
                    &alpha_one, W_q, dim_, x, dim_,
                    &beta_zero, q_cache, n_state_);
            }
            if (z_cache) {
                blas<__nv_bfloat16>::gemm(
                    blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                    n_state_, steps * batch_size_, dim_,
                    &alpha_one, W_z, dim_, x, dim_,
                    &beta_zero, z_cache, n_state_);
            }
        }
    }

    // =========================================================================
    // SINGLE kernel launch for ALL timesteps!
    // =========================================================================

    __nv_bfloat16* pre_cache = training_ ? pre_nonlin_cache : nullptr;
    __nv_bfloat16* sc = training_ ? s_cache : nullptr;

    // Dispatch based on use_tanh and is_delta
    if (is_delta_) {
        if (use_tanh_) {
            E74FusedForwardKernel_BF16<true, true><<<num_blocks, block_size, 0, stream_>>>(
                steps, batch_size_, n_state_, decay_,
                k_all, v_all, s, output, pre_cache, sc);
        } else {
            E74FusedForwardKernel_BF16<false, true><<<num_blocks, block_size, 0, stream_>>>(
                steps, batch_size_, n_state_, decay_,
                k_all, v_all, s, output, pre_cache, sc);
        }
    } else {
        if (use_tanh_) {
            E74FusedForwardKernel_BF16<true, false><<<num_blocks, block_size, 0, stream_>>>(
                steps, batch_size_, n_state_, decay_,
                k_all, v_all, s, output, pre_cache, sc);
        } else {
            E74FusedForwardKernel_BF16<false, false><<<num_blocks, block_size, 0, stream_>>>(
                steps, batch_size_, n_state_, decay_,
                k_all, v_all, s, output, pre_cache, sc);
        }
    }
}

// =============================================================================
// E74 Fused Backward Implementation
// =============================================================================

template<>
E74FusedBackward<__nv_bfloat16>::E74FusedBackward(
    int batch_size,
    int n_state,
    int dim,
    int proj_type,
    bool use_tanh,
    bool is_delta,
    float decay,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      n_state_(n_state),
      dim_(dim),
      proj_type_(proj_type),
      use_tanh_(use_tanh),
      is_delta_(is_delta),
      decay_(decay),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<>
void E74FusedBackward<__nv_bfloat16>::Run(
    int steps,
    const __nv_bfloat16* W_kvq,
    const __nv_bfloat16* W_kq,
    const __nv_bfloat16* W_v,
    const __nv_bfloat16* W_k,
    const __nv_bfloat16* W_q,
    const __nv_bfloat16* W_z,
    const __nv_bfloat16* x,
    const __nv_bfloat16* s,
    const __nv_bfloat16* k_cache,
    const __nv_bfloat16* v_cache,
    const __nv_bfloat16* q_cache,
    const __nv_bfloat16* z_cache,
    const __nv_bfloat16* pre_nonlin_cache,
    const __nv_bfloat16* s_cache,
    const __nv_bfloat16* d_output,
    __nv_bfloat16* dx,
    __nv_bfloat16* dW_kvq,
    __nv_bfloat16* dW_kq,
    __nv_bfloat16* dW_v,
    __nv_bfloat16* dW_k,
    __nv_bfloat16* dW_q,
    __nv_bfloat16* dW_z,
    __nv_bfloat16* workspace)
{
    static const __nv_bfloat16 alpha_one = __float2bfloat16(1.0f);
    static const __nv_bfloat16 beta_zero = __float2bfloat16(0.0f);
    static const __nv_bfloat16 beta_one = __float2bfloat16(1.0f);

    const int n = n_state_;
    const int BN = batch_size_ * n;
    const int block_size = 256;
    const int num_blocks = (BN + block_size - 1) / block_size;

    // Workspace for gradients
    __nv_bfloat16* d_k_all = workspace;
    __nv_bfloat16* d_v_all = workspace + steps * BN;

    // Initialize weight gradients to zero
    if (proj_type_ == 0) {
        cudaMemsetAsync(dW_kvq, 0, n * dim_ * sizeof(__nv_bfloat16), stream_);
    } else if (proj_type_ == 1) {
        cudaMemsetAsync(dW_kq, 0, n * dim_ * sizeof(__nv_bfloat16), stream_);
        cudaMemsetAsync(dW_v, 0, n * dim_ * sizeof(__nv_bfloat16), stream_);
    } else if (proj_type_ == 2) {
        cudaMemsetAsync(dW_k, 0, n * dim_ * sizeof(__nv_bfloat16), stream_);
        cudaMemsetAsync(dW_v, 0, n * dim_ * sizeof(__nv_bfloat16), stream_);
        cudaMemsetAsync(dW_q, 0, n * dim_ * sizeof(__nv_bfloat16), stream_);
    } else {
        cudaMemsetAsync(dW_k, 0, n * dim_ * sizeof(__nv_bfloat16), stream_);
        cudaMemsetAsync(dW_v, 0, n * dim_ * sizeof(__nv_bfloat16), stream_);
        cudaMemsetAsync(dW_q, 0, n * dim_ * sizeof(__nv_bfloat16), stream_);
        cudaMemsetAsync(dW_z, 0, n * dim_ * sizeof(__nv_bfloat16), stream_);
    }

    // =========================================================================
    // SINGLE kernel launch for backward through ALL timesteps!
    // =========================================================================

    // BUG FIX: For TIED_KVQ (proj_type_==0), v=k so v_cache was never filled.
    // Use k_cache as v_cache in this case.
    const __nv_bfloat16* v_cache_actual = (proj_type_ == 0) ? k_cache : v_cache;

    if (is_delta_) {
        if (use_tanh_) {
            E74FusedBackwardKernel_BF16<true, true><<<num_blocks, block_size, 0, stream_>>>(
                steps, batch_size_, n, decay_,
                k_cache, v_cache_actual, s, s_cache, pre_nonlin_cache, d_output,
                d_k_all, d_v_all);
        } else {
            E74FusedBackwardKernel_BF16<false, true><<<num_blocks, block_size, 0, stream_>>>(
                steps, batch_size_, n, decay_,
                k_cache, v_cache_actual, s, s_cache, pre_nonlin_cache, d_output,
                d_k_all, d_v_all);
        }
    } else {
        if (use_tanh_) {
            E74FusedBackwardKernel_BF16<true, false><<<num_blocks, block_size, 0, stream_>>>(
                steps, batch_size_, n, decay_,
                k_cache, v_cache_actual, s, s_cache, pre_nonlin_cache, d_output,
                d_k_all, d_v_all);
        } else {
            E74FusedBackwardKernel_BF16<false, false><<<num_blocks, block_size, 0, stream_>>>(
                steps, batch_size_, n, decay_,
                k_cache, v_cache_actual, s, s_cache, pre_nonlin_cache, d_output,
                d_k_all, d_v_all);
        }
    }

    // =========================================================================
    // Batched GEMMs for weight gradients and dx
    // =========================================================================

    if (proj_type_ == 0) {  // TIED_KVQ: k=v share the same projection
        // BUG FIX: Since k=v from W_kvq @ x, gradient must include BOTH d_k and d_v!
        // dx = (d_k + d_v) @ W_kvq
        blas<__nv_bfloat16>::gemm(
            blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, steps * batch_size_, n,
            &alpha_one, W_kvq, dim_, d_k_all, n,
            &beta_zero, dx, dim_);
        blas<__nv_bfloat16>::gemm(
            blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, steps * batch_size_, n,
            &alpha_one, W_kvq, dim_, d_v_all, n,
            &beta_one, dx, dim_);  // ACCUMULATE d_v contribution

        // dW_kvq = x.T @ (d_k + d_v)
        blas<__nv_bfloat16>::gemm(
            blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, n, steps * batch_size_,
            &alpha_one, x, dim_, d_k_all, n,
            &beta_one, dW_kvq, dim_);
        blas<__nv_bfloat16>::gemm(
            blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, n, steps * batch_size_,
            &alpha_one, x, dim_, d_v_all, n,
            &beta_one, dW_kvq, dim_);  // ACCUMULATE d_v contribution
    }
    else if (proj_type_ == 1) {  // TIED_KQ
        // dx = d_k @ W_kq + d_v @ W_v
        blas<__nv_bfloat16>::gemm(
            blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, steps * batch_size_, n,
            &alpha_one, W_kq, dim_, d_k_all, n,
            &beta_zero, dx, dim_);
        blas<__nv_bfloat16>::gemm(
            blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, steps * batch_size_, n,
            &alpha_one, W_v, dim_, d_v_all, n,
            &beta_one, dx, dim_);

        // Weight gradients
        blas<__nv_bfloat16>::gemm(
            blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, n, steps * batch_size_,
            &alpha_one, x, dim_, d_k_all, n,
            &beta_one, dW_kq, dim_);
        blas<__nv_bfloat16>::gemm(
            blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, n, steps * batch_size_,
            &alpha_one, x, dim_, d_v_all, n,
            &beta_one, dW_v, dim_);
    }
    else if (proj_type_ == 2) {  // NO_Z
        // dx = d_k @ W_k + d_v @ W_v (q gradient flows through k for backward)
        blas<__nv_bfloat16>::gemm(
            blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, steps * batch_size_, n,
            &alpha_one, W_k, dim_, d_k_all, n,
            &beta_zero, dx, dim_);
        blas<__nv_bfloat16>::gemm(
            blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, steps * batch_size_, n,
            &alpha_one, W_v, dim_, d_v_all, n,
            &beta_one, dx, dim_);

        // Weight gradients
        blas<__nv_bfloat16>::gemm(
            blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, n, steps * batch_size_,
            &alpha_one, x, dim_, d_k_all, n,
            &beta_one, dW_k, dim_);
        blas<__nv_bfloat16>::gemm(
            blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, n, steps * batch_size_,
            &alpha_one, x, dim_, d_v_all, n,
            &beta_one, dW_v, dim_);
    }
    else {  // FULL
        blas<__nv_bfloat16>::gemm(
            blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, steps * batch_size_, n,
            &alpha_one, W_k, dim_, d_k_all, n,
            &beta_zero, dx, dim_);
        blas<__nv_bfloat16>::gemm(
            blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, steps * batch_size_, n,
            &alpha_one, W_v, dim_, d_v_all, n,
            &beta_one, dx, dim_);

        blas<__nv_bfloat16>::gemm(
            blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, n, steps * batch_size_,
            &alpha_one, x, dim_, d_k_all, n,
            &beta_one, dW_k, dim_);
        blas<__nv_bfloat16>::gemm(
            blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, n, steps * batch_size_,
            &alpha_one, x, dim_, d_v_all, n,
            &beta_one, dW_v, dim_);
    }
}

// =============================================================================
// Explicit Template Instantiations
// =============================================================================

template struct E74FusedForward<float>;
template struct E74FusedForward<__half>;
template struct E74FusedForward<double>;

template struct E74FusedBackward<float>;
template struct E74FusedBackward<__half>;
template struct E74FusedBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
