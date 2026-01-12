// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E-Series Elman CUDA Kernels
//
// This header defines CUDA kernels for the E-series Elman models.
// Archived kernels are in elman_ladder_full.h.archive
//
// E0: Stock Elman - tanh recurrence + h*silu(W_gate@x) gating
//     - Spectral normalization on W_h (radius < 0.99)

#ifndef HASTY_ELMAN_LADDER_H
#define HASTY_ELMAN_LADDER_H

#include <cuda.h>
#include <cublas_v2.h>
#include <cufft.h>

namespace hasty {
namespace v0 {
namespace elman_ladder {

// =============================================================================
// E0: Stock Elman (learned gate projection)
// h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b)
// output = h * silu(W_gate @ x + b_gate)
// =============================================================================

template<typename T>
struct StockElmanForward {
    StockElmanForward(
        bool training,
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,       // [dim, dim]
        const T* W_h,       // [dim, dim]
        const T* W_gate,    // [dim, dim] gate projection
        const T* b,         // [dim]
        const T* b_gate,    // [dim] gate bias
        const T* x,         // [T, B, dim]
        T* h,               // [T+1, B, dim] hidden states
        T* output,          // [T, B, dim] selective output
        T* v,               // [T, B, dim] pre-activation for backward
        T* gate_cache,      // [T, B, dim] gate cache for backward (stores gate_raw)
        T* workspace);      // [2*T*B*dim + B*dim] for Wx, gate_proj, Rh

private:
    bool training_;
    int batch_size_;
    int dim_;
    int seq_len_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct StockElmanBackward {
    StockElmanBackward(
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    ~StockElmanBackward();

    void Run(
        int steps,
        const T* W_x,
        const T* W_h,
        const T* W_gate,    // [dim, dim] gate projection
        const T* x,
        const T* h,
        const T* v,
        const T* gate_cache,
        const T* d_output,  // [T, B, dim] gradient from output
        T* dx,              // [T, B, dim]
        T* dW_x,            // [dim, dim]
        T* dW_h,            // [dim, dim]
        T* dW_gate,         // [dim, dim] gradient for gate projection
        T* db,              // [dim]
        T* d_b_gate,        // [dim]
        T* workspace);      // [(2*T+2)*B*dim + ceil(2*dim*4/sizeof(T))]

private:
    int batch_size_;
    int dim_;
    int seq_len_;
    cublasHandle_t blas_handle_;
    cudaStream_t sync_stream_;
    cudaStream_t stream_[2];
    cudaEvent_t event_;
};

// =============================================================================
// E1: Mamba-Gated Elman (Mamba2-style split projection gating)
// x, z = split(in_proj(x))           # Pre-computed before kernel
// x = silu(x)                        # Pre-computed before kernel
// h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b)  # Elman recurrence
// output = h * silu(z)               # Gate with z branch
// =============================================================================

template<typename T>
struct MambaGatedElmanForward {
    MambaGatedElmanForward(
        bool training,
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,       // [dim, dim]
        const T* W_h,       // [dim, dim]
        const T* b,         // [dim]
        const T* x,         // [T, B, dim] pre-activated input
        const T* z,         // [T, B, dim] gate input (pre silu)
        T* h,               // [T+1, B, dim] hidden states
        T* output,          // [T, B, dim] output
        T* v,               // [T, B, dim] pre-activation cache
        T* workspace);      // [T*B*dim + B*dim] for Wx, Rh

private:
    bool training_;
    int batch_size_;
    int dim_;
    int seq_len_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct MambaGatedElmanBackward {
    MambaGatedElmanBackward(
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,
        const T* W_h,
        const T* x,
        const T* z,
        const T* h,
        const T* v,
        const T* d_output,
        T* dx,
        T* dz,
        T* dW_x,
        T* dW_h,
        T* db,
        T* workspace);      // [(T+2)*B*dim + ceil(dim*4/sizeof(T))]

private:
    int batch_size_;
    int dim_;
    int seq_len_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E33: Self-Gated Elman (simplification test: h gates itself)
// h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b)  # Same recurrence as E1
// output = h * silu(h)                       # KEY: self-gating, no z needed
// =============================================================================

template<typename T>
struct E33SelfGateForward {
    E33SelfGateForward(
        bool training,
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,       // [dim, dim]
        const T* W_h,       // [dim, dim]
        const T* b,         // [dim]
        const T* x,         // [T, B, dim] pre-activated input
        T* h,               // [T+1, B, dim] hidden states
        T* output,          // [T, B, dim] output
        T* v,               // [T, B, dim] pre-activation cache
        T* workspace);      // [T*B*dim + B*dim] for Wx, Rh

private:
    bool training_;
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E33SelfGateBackward {
    E33SelfGateBackward(
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,
        const T* W_h,
        const T* x,
        const T* h,
        const T* v,
        const T* d_output,
        T* dx,
        T* dW_x,
        T* dW_h,
        T* db,
        T* workspace);      // [(T+2)*B*dim + ceil(dim*4/sizeof(T))]

private:
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E34: Diagonal W_h Elman - W_h is diagonal (vector instead of matrix)
// h_t = tanh(W_x @ x_t + d * h_{t-1} + b)  # d is [dim] vector, element-wise
// output = h * silu(h)                      # Self-gating from E33
// =============================================================================

template<typename T>
struct E34DiagonalWhForward {
    E34DiagonalWhForward(
        bool training,
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,       // [dim, dim]
        const T* d,         // [dim] diagonal vector (replaces W_h matrix)
        const T* b,         // [dim]
        const T* x,         // [T, B, dim] pre-activated input
        T* h,               // [T+1, B, dim] hidden states
        T* output,          // [T, B, dim] output
        T* v,               // [T, B, dim] pre-activation cache (training only)
        T* workspace);      // [T*B*dim] for Wx precompute

private:
    bool training_;
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E34DiagonalWhBackward {
    E34DiagonalWhBackward(
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,
        const T* d,         // [dim] diagonal vector
        const T* x,
        const T* h,
        const T* v,
        const T* d_output,
        T* dx,
        T* dW_x,
        T* dd,              // [dim] gradient for diagonal
        T* db,
        T* workspace);      // [(T+2)*B*dim + 2*dim*sizeof(float)/sizeof(T)]

private:
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E41: Diagonal W_x Elman - W_x is diagonal (vector instead of matrix)
// h_t = tanh(d_x * x_t + W_h @ h_{t-1} + b)  # d_x is [dim] vector, element-wise
// output = h * silu(h)                        # Self-gating from E33
// Key: Removes batch W_x @ x GEMM, keeps per-step W_h @ h GEMM
// =============================================================================

template<typename T>
struct E41DiagonalWxForward {
    E41DiagonalWxForward(
        bool training,
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* d_x,       // [dim] diagonal vector (replaces W_x matrix)
        const T* W_h,       // [dim, dim]
        const T* b,         // [dim]
        const T* x,         // [T, B, dim] pre-activated input
        T* h,               // [T+1, B, dim] hidden states
        T* output,          // [T, B, dim] output
        T* v,               // [T, B, dim] pre-activation cache (training only)
        T* workspace);      // [B*dim] for Rh computation

private:
    bool training_;
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E41DiagonalWxBackward {
    E41DiagonalWxBackward(
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* d_x,       // [dim] diagonal vector
        const T* W_h,       // [dim, dim]
        const T* x,
        const T* h,
        const T* v,
        const T* d_output,
        T* dx,
        T* dd_x,            // [dim] gradient for diagonal
        T* dW_h,
        T* db,
        T* workspace);      // [(T+2)*B*dim + 2*dim*sizeof(float)/sizeof(T)]

private:
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E37: Tied Weights Elman (single W for both input and hidden)
// h_t = tanh(W @ x_t + W @ h_{t-1} + b) = tanh(W @ (x_t + h_{t-1}) + b)
// output = h * silu(h)                       # Self-gating from E33
// Key: Single GEMM per timestep instead of two
// =============================================================================

template<typename T>
struct E37TiedWeightsForward {
    E37TiedWeightsForward(
        bool training,
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W,         // [dim, dim] - SINGLE weight matrix
        const T* b,         // [dim]
        const T* x,         // [T, B, dim] pre-activated input
        T* h,               // [T+1, B, dim] hidden states
        T* output,          // [T, B, dim] output
        T* v,               // [T, B, dim] pre-activation cache
        T* workspace);      // [2*B*dim] for tmp_sum, tmp_Wsum

private:
    bool training_;
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E37TiedWeightsBackward {
    E37TiedWeightsBackward(
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W,
        const T* x,
        const T* h,
        const T* v,
        const T* d_output,
        T* dx,
        T* dW,              // [dim, dim] - single gradient
        T* db,
        T* workspace);      // [(T+3)*B*dim + ceil(dim*4/sizeof(T))]

private:
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E35: Cubic-Gated Elman (simplification test: output = h^3)
// h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b)  # Same recurrence as E1
// output = h^3                               # KEY: cubic gating, no z needed
// =============================================================================

template<typename T>
struct E35CubicGateForward {
    E35CubicGateForward(
        bool training,
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,       // [dim, dim]
        const T* W_h,       // [dim, dim]
        const T* b,         // [dim]
        const T* x,         // [T, B, dim] pre-activated input
        T* h,               // [T+1, B, dim] hidden states
        T* output,          // [T, B, dim] output
        T* v,               // [T, B, dim] pre-activation cache
        T* workspace);      // [T*B*dim + B*dim] for Wx, Rh

private:
    bool training_;
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E35CubicGateBackward {
    E35CubicGateBackward(
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,
        const T* W_h,
        const T* x,
        const T* h,
        const T* v,
        const T* d_output,
        T* dx,
        T* dW_x,
        T* dW_h,
        T* db,
        T* workspace);      // [(T+2)*B*dim + ceil(dim*4/sizeof(T))]

private:
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E36: Linear Recurrence + Self-Gate (no tanh in recurrence)
// h_t = W_x @ x_t + W_h @ h_{t-1} + b       # LINEAR recurrence (no tanh!)
// output = h * silu(h)                       # Self-gating provides nonlinearity
// IMPORTANT: Spectral normalization of W_h (radius < 1) critical for stability
// =============================================================================

template<typename T>
struct E36LinearRecurrenceForward {
    E36LinearRecurrenceForward(
        bool training,
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,       // [dim, dim]
        const T* W_h,       // [dim, dim]
        const T* b,         // [dim]
        const T* x,         // [T, B, dim] pre-activated input
        T* h,               // [T+1, B, dim] hidden states
        T* output,          // [T, B, dim] output
        T* v,               // [T, B, dim] pre-activation cache
        T* workspace);      // [T*B*dim + B*dim] for Wx, Rh

private:
    bool training_;
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E36LinearRecurrenceBackward {
    E36LinearRecurrenceBackward(
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,
        const T* W_h,
        const T* x,
        const T* h,
        const T* v,
        const T* d_output,
        T* dx,
        T* dW_x,
        T* dW_h,
        T* db,
        T* workspace);      // [(T+3)*B*dim + ceil(dim*4/sizeof(T))]

private:
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E17: Selective W_h Elman (input-dependent gating on recurrence)
// h_t = tanh(W_x @ x_t + (W_h @ h_{t-1}) * sigmoid(W_gate @ x_t) + b)
// output = h * silu(z)
// Key: Diagonal selectivity on W_h @ h (like Mamba2's selective A)
// =============================================================================

template<typename T>
struct SelectiveWhElmanForward {
    SelectiveWhElmanForward(
        bool training,
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,       // [dim, dim]
        const T* W_h,       // [dim, dim]
        const T* W_gate,    // [dim, dim] gate projection (NEW)
        const T* b,         // [dim]
        const T* b_gate,    // [dim] gate bias (optional, can be nullptr)
        const T* x,         // [T, B, dim] pre-activated input
        const T* z,         // [T, B, dim] gate input (pre silu)
        T* h,               // [T+1, B, dim] hidden states
        T* output,          // [T, B, dim] output
        T* v,               // [T, B, dim] pre-activation cache
        T* gate_cache,      // [T, B, dim] gate cache (sigmoid(G))
        T* Rh_cache,        // [T, B, dim] cache W_h @ h for backward
        T* workspace);      // [2*T*B*dim + B*dim] for Wx, G, Rh

private:
    bool training_;
    int batch_size_;
    int dim_;
    int seq_len_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct SelectiveWhElmanBackward {
    SelectiveWhElmanBackward(
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,
        const T* W_h,
        const T* W_gate,
        const T* x,
        const T* z,
        const T* h,
        const T* v,
        const T* gate_cache,
        const T* Rh_cache,
        const T* d_output,
        T* dx,
        T* dz,
        T* dW_x,
        T* dW_h,
        T* dW_gate,         // [dim, dim] gradient for gate projection
        T* db,
        T* workspace);      // [(3*T+2)*B*dim + ceil(dim*4/sizeof(T))]

private:
    int batch_size_;
    int dim_;
    int seq_len_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E18: h-Aware Gate Elman (h-aware output gating)
// Variants controlled by gate_mode parameter:
// 0 = E18-A: output = h * silu(z + h)     -- add h to gate (FREE)
// 1 = E18-B: output = h * silu(z + Rh)    -- add Rh to gate (FREE, cache Rh)
// 2 = E18-E: output = h                   -- no gate (faster, fewer params)
// =============================================================================

template<typename T>
struct HAwareGateElmanForward {
    HAwareGateElmanForward(
        bool training,
        int batch_size,
        int dim,
        int gate_mode,  // 0=A, 1=B, 2=E
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,       // [dim, dim]
        const T* W_h,       // [dim, dim]
        const T* b,         // [dim]
        const T* x,         // [T, B, dim] pre-activated input
        const T* z,         // [T, B, dim] gate input (ignored for mode=2)
        T* h,               // [T+1, B, dim] hidden states
        T* output,          // [T, B, dim] output
        T* v,               // [T, B, dim] pre-activation cache
        T* Rh_cache,        // [T, B, dim] Rh cache for mode=1 backward (can be nullptr)
        T* workspace);      // [T*B*dim + B*dim] for Wx, Rh

private:
    bool training_;
    int batch_size_;
    int dim_;
    int seq_len_;
    int gate_mode_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct HAwareGateElmanBackward {
    HAwareGateElmanBackward(
        int batch_size,
        int dim,
        int gate_mode,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,
        const T* W_h,
        const T* x,
        const T* z,
        const T* h,
        const T* v,
        const T* Rh_cache,  // [T, B, dim] for mode=1
        const T* d_output,
        T* dx,
        T* dz,
        T* dW_x,
        T* dW_h,
        T* db,
        T* workspace);      // [(T+3)*B*dim + dim]

private:
    int batch_size_;
    int dim_;
    int seq_len_;
    int gate_mode_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E19: Simplified Gate Elman (remove W_gate, reuse Wx in gate)
// Variants controlled by gate_mode parameter:
// 0 = E19-A: gate = silu(Wx + h + b_gate)    -- reuse Wx in gate, no W_gate
// 1 = E19-B: gate = silu(h + b_gate)         -- h-only gate, no Wx
// 2 = E19-D: h = tanh(Wx + Rh + h_prev + b)  -- residual h + E18-A gate
// 3 = E19-E: A + D combined (residual + Wx in gate)
// =============================================================================

template<typename T>
struct SimplifiedGateElmanForward {
    SimplifiedGateElmanForward(
        bool training,
        int batch_size,
        int dim,
        int gate_mode,  // 0=A, 1=B, 2=D, 3=E
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,       // [dim, dim]
        const T* W_h,       // [dim, dim]
        const T* b,         // [dim]
        const T* b_gate,    // [dim] gate bias (for modes 0,1,3)
        const T* x,         // [T, B, dim] pre-activated input
        const T* z,         // [T, B, dim] gate input (for mode 2 only - E18-A style)
        T* h,               // [T+1, B, dim] hidden states
        T* output,          // [T, B, dim] output
        T* v,               // [T, B, dim] pre-activation cache
        T* Wx_cache,        // [T, B, dim] cache Wx for backward (modes 0,3)
        T* workspace);      // [T*B*dim + B*dim] for Wx, Rh

private:
    bool training_;
    int batch_size_;
    int dim_;
    int seq_len_;
    int gate_mode_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct SimplifiedGateElmanBackward {
    SimplifiedGateElmanBackward(
        int batch_size,
        int dim,
        int gate_mode,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,
        const T* W_h,
        const T* b_gate,    // [dim] for gradient computation
        const T* x,
        const T* z,         // [T, B, dim] for mode 2 only
        const T* h,
        const T* v,
        const T* Wx_cache,  // [T, B, dim] from forward (modes 0,3)
        const T* d_output,
        T* dx,
        T* dz,              // Output for mode 2 only
        T* dW_x,
        T* dW_h,
        T* db,
        T* db_gate,         // [dim] for modes 0,1,3
        T* workspace);      // [(2*T+2)*B*dim + 2*dim]

private:
    int batch_size_;
    int dim_;
    int seq_len_;
    int gate_mode_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// Softsign Elman - E1 variant using softsign instead of tanh
// softsign(x) = x / (1 + |x|)
// - Cheaper than tanh (no exp)
// - Bounded (-1, 1) like tanh
// - Smoother gradients: derivative = 1/(1+|x|)^2
// =============================================================================

template<typename T>
struct SoftsignElmanForward {
    SoftsignElmanForward(
        bool training,
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,       // [dim, dim]
        const T* W_h,       // [dim, dim]
        const T* b,         // [dim]
        const T* x,         // [T, B, dim] pre-activated input
        const T* z,         // [T, B, dim] gate input (pre silu)
        T* h,               // [T+1, B, dim] hidden states
        T* output,          // [T, B, dim] output
        T* v,               // [T, B, dim] pre-activation cache
        T* workspace);      // [T*B*dim + B*dim] for Wx, Rh

private:
    bool training_;
    int batch_size_;
    int dim_;
    int seq_len_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct SoftsignElmanBackward {
    SoftsignElmanBackward(
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,
        const T* W_h,
        const T* x,
        const T* z,
        const T* h,
        const T* v,
        const T* d_output,
        T* dx,
        T* dz,
        T* dW_x,
        T* dW_h,
        T* db,
        T* workspace);      // [(T+2)*B*dim + ceil(dim*4/sizeof(T))]

private:
    int batch_size_;
    int dim_;
    int seq_len_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E16: Diagonal State-Expanded Elman
// h' = tanh(A ⊙ h + B @ x)  where A is diagonal
// y = C @ h * silu(z)
// - State expansion (d_state > d_model)
// - Diagonal recurrence O(n) instead of O(n²)
// - tanh nonlinearity for composition depth
// =============================================================================

template<typename T>
struct DiagonalStateElmanForward {
    DiagonalStateElmanForward(
        bool training,
        int batch_size,
        int d_model,
        int d_state,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* B,         // [d_model, d_state]
        const T* C,         // [d_state, d_model]
        const T* A,         // [d_state] diagonal
        const T* x,         // [T, B, d_model]
        const T* z,         // [T, B, d_model] gate input
        T* h,               // [T+1, B, d_state] hidden states
        T* output,          // [T, B, d_model]
        T* v,               // [T, B, d_state] pre-activation cache
        T* workspace);      // [T*B*d_state + B*d_model]

private:
    bool training_;
    int batch_size_;
    int d_model_;
    int d_state_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct DiagonalStateElmanBackward {
    DiagonalStateElmanBackward(
        int batch_size,
        int d_model,
        int d_state,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* B,
        const T* C,
        const T* A,
        const T* x,
        const T* z,
        const T* h,
        const T* v,
        const T* d_output,
        T* dx,
        T* dz,
        T* dB,
        T* dC,
        T* dA,
        T* workspace);      // [T*B*d_state + 2*B*d_state + B*d_model + d_state*4]

private:
    int batch_size_;
    int d_model_;
    int d_state_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E2: Slot-Based Elman (with cuBLAS GEMMs - same speed as e0, more memory)
// h_t[s] = tanh(W_x @ x + W_h @ h_prev[s] + b)    for each slot s
// output = sum(C[s] * h_t[s]) * silu(z)
// Key: Batch slots into GEMM by treating [B, n_slots, d] as [B*n_slots, d]
// =============================================================================

template<typename T>
struct SlotElmanForward {
    SlotElmanForward(
        bool training,
        int batch_size,
        int dim,
        int n_slots,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,         // [dim, dim]
        const T* W_h,         // [dim, dim]
        const T* b,           // [dim]
        const T* C,           // [n_slots]
        const T* x,           // [T, B, dim]
        const T* z,           // [T, B, dim]
        T* h,                 // [T+1, B, n_slots, dim]
        T* output,            // [T, B, dim]
        T* v,                 // [T, B, n_slots, dim] pre-activation cache
        T* workspace,         // [T*B*dim + B*n_slots*dim]
        cublasHandle_t blas_handle);

private:
    bool training_;
    int batch_size_;
    int dim_;
    int seq_len_;
    int n_slots_;
    cudaStream_t stream_;
};

template<typename T>
struct SlotElmanBackward {
    SlotElmanBackward(
        int batch_size,
        int dim,
        int n_slots,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,         // [dim, dim]
        const T* W_h,         // [dim, dim]
        const T* C,           // [n_slots]
        const T* x,           // [T, B, dim]
        const T* z,           // [T, B, dim]
        const T* h,           // [T+1, B, n_slots, dim]
        const T* v,           // [T, B, n_slots, dim]
        const T* d_output,    // [T, B, dim]
        T* dx,                // [T, B, dim]
        T* dz,                // [T, B, dim]
        T* dW_x,              // [dim, dim]
        T* dW_h,              // [dim, dim]
        T* db,                // [dim]
        T* dC,                // [n_slots]
        T* workspace,         // [(T+2)*B*n_slots*dim + T*B*dim + dim + n_slots]
        cublasHandle_t blas_handle);

private:
    int batch_size_;
    int dim_;
    int seq_len_;
    int n_slots_;
    cudaStream_t stream_;
};

// =============================================================================
// E3: Low-Rank Slot Elman (independent low-rank W_h per slot)
// h_t[s] = tanh(W_x @ x + U_s @ (V_s @ h_prev[s]) + b)    for each slot s
// output = sum(C[s] * h_t[s]) * silu(z)
// Key: Low-rank W_h_s = U_s @ V_s gives unique dynamics per slot with O(2dr) compute
// =============================================================================

template<typename T>
struct LowRankSlotElmanForward {
    LowRankSlotElmanForward(
        bool training,
        int batch_size,
        int dim,
        int n_slots,
        int rank,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,         // [dim, dim]
        const T* U,           // [n_slots, dim, rank] per-slot output projection
        const T* V,           // [n_slots, rank, dim] per-slot input projection
        const T* b,           // [dim]
        const T* C,           // [n_slots]
        const T* x,           // [T, B, dim]
        const T* z,           // [T, B, dim]
        T* h,                 // [T+1, n_slots, B, dim]
        T* output,            // [T, B, dim]
        T* Vh_cache,          // [T, n_slots, B, rank] - cache Vh for backward GEMM
        T* workspace,         // [T*B*dim]
        cublasHandle_t blas_handle);

private:
    bool training_;
    int batch_size_;
    int dim_;
    int seq_len_;
    int n_slots_;
    int rank_;
    cudaStream_t stream_;
};

template<typename T>
struct LowRankSlotElmanBackward {
    LowRankSlotElmanBackward(
        int batch_size,
        int dim,
        int n_slots,
        int rank,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,         // [dim, dim]
        const T* U,           // [n_slots, dim, rank]
        const T* V,           // [n_slots, rank, dim]
        const T* C,           // [n_slots]
        const T* x,           // [T, B, dim]
        const T* z,           // [T, B, dim]
        const T* h,           // [T+1, n_slots, B, dim]
        const T* Vh_all,      // [T, n_slots, B, rank] - from forward cache
        const T* d_output,    // [T, B, dim]
        T* dx,                // [T, B, dim]
        T* dz,                // [T, B, dim]
        T* dW_x,              // [dim, dim]
        T* dU,                // [n_slots, dim, rank]
        T* dV,                // [n_slots, rank, dim]
        T* db,                // [dim]
        T* dC,                // [n_slots]
        T* workspace,         // [T*SBD + T*SBR + 2*SBD + T*BD + float_grads]
        cublasHandle_t blas_handle);

private:
    int batch_size_;
    int dim_;
    int seq_len_;
    int n_slots_;
    int rank_;
    cudaStream_t stream_;
};

// =============================================================================
// E4: Low-Rank Elman (SVD-style for fat hidden state)
// h_t = tanh(W_x @ x_t + U @ V @ h_{t-1} + b)
// output = h_t * silu(z_t)
// Key: U is [dim, rank], V is [rank, dim]. Same params as E1, 2x hidden dim.
// =============================================================================

template<typename T>
struct LowRankElmanForward {
    LowRankElmanForward(
        bool training,
        int batch_size,
        int dim,
        int rank,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,       // [dim, dim]
        const T* U,         // [dim, rank]
        const T* V,         // [rank, dim]
        const T* b,         // [dim]
        const T* x,         // [T, B, dim]
        const T* z,         // [T, B, dim]
        T* h,               // [T+1, B, dim]
        T* output,          // [T, B, dim]
        T* v,               // [T, B, dim] pre-activation cache
        T* workspace);      // [T*B*dim + B*rank + B*dim]

private:
    bool training_;
    int batch_size_;
    int dim_;
    int seq_len_;
    int rank_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct LowRankElmanBackward {
    LowRankElmanBackward(
        int batch_size,
        int dim,
        int rank,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,
        const T* U,
        const T* V,
        const T* x,
        const T* z,
        const T* h,
        const T* v,
        const T* d_output,
        T* dx,
        T* dz,
        T* dW_x,
        T* dU,
        T* dV,
        T* db,
        T* workspace);      // [(steps+2)*B*dim + B*rank + dim]

private:
    int batch_size_;
    int dim_;
    int seq_len_;
    int rank_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E5: Pure Low-Rank Elman (no projections, all low-rank on full dim)
// h_t = tanh(U_h @ V_h @ h_{t-1} + U_x @ V_x @ x_t + b)
// y_t = h_t * silu(U_z @ V_z @ x_t)
// Key: No in_proj/out_proj. Hidden state IS dim. All ops are low-rank.
// =============================================================================

template<typename T>
struct PureLowRankElmanForward {
    PureLowRankElmanForward(
        bool training,
        int batch_size,
        int dim,
        int rank,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* U_h,       // [dim, rank]
        const T* V_h,       // [rank, dim]
        const T* U_x,       // [dim, rank]
        const T* V_x,       // [rank, dim]
        const T* U_z,       // [dim, rank]
        const T* V_z,       // [rank, dim]
        const T* b,         // [dim]
        const T* x,         // [T, B, dim]
        T* h,               // [T+1, B, dim]
        T* output,          // [T, B, dim]
        T* v,               // [T, B, dim] pre-activation cache
        T* workspace);      // [2*T*BR + 2*T*BD + BR + BD]

private:
    bool training_;
    int batch_size_;
    int dim_;
    int seq_len_;
    int rank_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct PureLowRankElmanBackward {
    PureLowRankElmanBackward(
        int batch_size,
        int dim,
        int rank,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* U_h,
        const T* V_h,
        const T* U_x,
        const T* V_x,
        const T* U_z,
        const T* V_z,
        const T* x,
        const T* h,
        const T* v,
        const T* d_output,
        T* dx,
        T* dU_h,
        T* dV_h,
        T* dU_x,
        T* dV_x,
        T* dU_z,
        T* dV_z,
        T* db,
        T* workspace);      // [4*BD + 6*BR + dim]

private:
    int batch_size_;
    int dim_;
    int seq_len_;
    int rank_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E5 Fused: Pure Low-Rank Elman with Optimized Fused Kernels
// Same math as E5, but fuses tanh+gate into single kernel (3 ops/step vs 4)
// =============================================================================

template<typename T>
struct PureLowRankElmanForwardFused {
    PureLowRankElmanForwardFused(
        bool training,
        int batch_size,
        int dim,
        int rank,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* U_h,       // [dim, rank]
        const T* V_h,       // [rank, dim]
        const T* U_x,       // [dim, rank]
        const T* V_x,       // [rank, dim]
        const T* U_z,       // [dim, rank]
        const T* V_z,       // [rank, dim]
        const T* b,         // [dim]
        const T* x,         // [T, B, dim]
        T* h,               // [T+1, B, dim]
        T* output,          // [T, B, dim]
        T* v,               // [T, B, dim] pre-activation cache
        T* workspace);      // [2*T*BR + 2*T*BD + BR + BD]

private:
    bool training_;
    int batch_size_;
    int dim_;
    int seq_len_;
    int rank_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
    bool graph_captured_;
    int captured_steps_;
};

template<typename T>
struct PureLowRankElmanBackwardFused {
    PureLowRankElmanBackwardFused(
        int batch_size,
        int dim,
        int rank,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* U_h,
        const T* V_h,
        const T* U_x,
        const T* V_x,
        const T* U_z,
        const T* V_z,
        const T* x,
        const T* h,
        const T* v,
        const T* d_output,
        T* dx,
        T* dU_h,
        T* dV_h,
        T* dU_x,
        T* dV_x,
        T* dU_z,
        T* dV_z,
        T* db,
        T* workspace);      // [5*BD + 7*BR + dim]

private:
    int batch_size_;
    int dim_;
    int seq_len_;
    int rank_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E5 B2B: Pure Low-Rank Elman with CUTLASS B2B GEMM Fusion
// Same math as E5, but fuses the two sequential GEMMs (V_h @ h, U_h @ result)
// into a single kernel using CUTLASS two-tensor-op fusion.
// Keeps intermediate result in shared memory, eliminating global memory roundtrip.
//
// Constraints: rank must be 64, 128, or 256 (ThreadblockShape requirement)
// =============================================================================

template<typename T>
struct B2bLowRankElmanForward {
    B2bLowRankElmanForward(
        bool training,
        int batch_size,
        int dim,
        int rank,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* U_h,       // [dim, rank]
        const T* V_h,       // [rank, dim]
        const T* U_x,       // [dim, rank]
        const T* V_x,       // [rank, dim]
        const T* U_z,       // [dim, rank]
        const T* V_z,       // [rank, dim]
        const T* b,         // [dim]
        const T* x,         // [T, B, dim]
        T* h,               // [T+1, B, dim]
        T* output,          // [T, B, dim]
        T* v,               // [T, B, dim] pre-activation cache
        T* workspace);      // [2*T*BR + 2*T*BD + 2*BR + 2*BD]

    bool isB2bSupported() const { return b2b_supported_; }
    bool usesTiledB2b() const { return use_tiled_b2b_; }

private:
    bool training_;
    int batch_size_;
    int dim_;
    int seq_len_;
    int rank_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
    bool b2b_supported_;      // CUTLASS B2B GEMM (small dim only)
    bool use_tiled_b2b_;      // Custom tiled B2B GEMM (any dim)
};

template<typename T>
struct B2bLowRankElmanBackward {
    B2bLowRankElmanBackward(
        int batch_size,
        int dim,
        int rank,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* U_h,
        const T* V_h,
        const T* U_x,
        const T* V_x,
        const T* U_z,
        const T* V_z,
        const T* x,
        const T* h,
        const T* v,
        const T* d_output,
        T* dx,
        T* dU_h,
        T* dV_h,
        T* dU_x,
        T* dV_x,
        T* dU_z,
        T* dV_z,
        T* db,
        T* workspace);      // [5*BD + 7*BR + dim]

private:
    int batch_size_;
    int dim_;
    int seq_len_;
    int rank_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E6: Circulant FFT Elman (O(n log n) hidden state updates via FFT)
// h_t = tanh(circ(c_h) @ h_{t-1} + circ(c_x) @ x_t + b)
// output_t = h_t * silu(W_gate @ x_t + b_gate)
//
// Circulant matrix-vector multiply via FFT:
// circ(c) @ v = IFFT(FFT(c) * FFT(v))
//
// This gives an effective n×n matrix using only n parameters per circulant.
// Complexity: O(n log n) vs O(n²) for dense matmul
// =============================================================================

template<typename T>
struct CirculantElmanForward {
    CirculantElmanForward(
        bool training,
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    ~CirculantElmanForward();

    void Run(
        int steps,
        const T* c_h,         // [dim] circulant vector for hidden
        const T* c_x,         // [dim] circulant vector for input
        const T* W_gate,      // [dim, dim] gate projection
        const T* b,           // [dim]
        const T* b_gate,      // [dim]
        const T* x,           // [T, B, dim]
        T* h,                 // [T+1, B, dim] hidden states
        T* output,            // [T, B, dim]
        T* v,                 // [T, B, dim] pre-activation cache
        T* gate_cache,        // [T, B, dim] gate cache for backward
        float* fft_workspace, // Complex workspace for FFT operations (float32)
        T* gate_proj);        // [T, B, dim] pre-computed gate projections

private:
    bool training_;
    int batch_size_;
    int dim_;
    int seq_len_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
    cufftHandle fft_plan_c_;      // cufftHandle for single FFT
    cufftHandle fft_plan_batch_;  // cufftHandle for batched FFT
};

template<typename T>
struct CirculantElmanBackward {
    CirculantElmanBackward(
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    ~CirculantElmanBackward();

    void Run(
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
        T* d_c_h,           // [dim] gradient for c_h
        T* d_c_x,           // [dim] gradient for c_x
        T* dW_gate,         // [dim, dim] gradient for W_gate
        T* db,              // [dim]
        T* d_b_gate,        // [dim]
        float* fft_workspace,  // Complex workspace for FFT operations (float32)
        T* work_T);         // Workspace in model dtype [2*BD]

private:
    int batch_size_;
    int dim_;
    int seq_len_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
    cufftHandle fft_plan_c_;
    cufftHandle fft_plan_batch_;
};

// =============================================================================
// E6 Diagonal: Diagonal Elman (per-channel scalar recurrence + low-rank mixing)
// h_t = gate * h_{t-1} + (1 - gate) * x_t   (per-channel EMA)
// y_t = U @ V @ h_t * silu(x_t)             (low-rank cross-channel mix)
// Key: Diagonal recurrence is O(dim). Allows MASSIVE depth (~755 layers at 50M).
// =============================================================================

template<typename T>
struct DiagonalElmanForward {
    DiagonalElmanForward(
        bool training,
        int batch_size,
        int dim,
        int rank,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* gate_logit,  // [dim] raw gate logits
        const T* U,           // [dim, rank]
        const T* V,           // [rank, dim]
        const T* x,           // [T, B, dim]
        T* h,                 // [T+1, B, dim]
        T* output,            // [T, B, dim]
        T* workspace);        // [BR + BD]

private:
    bool training_;
    int batch_size_;
    int dim_;
    int seq_len_;
    int rank_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct DiagonalElmanBackward {
    DiagonalElmanBackward(
        int batch_size,
        int dim,
        int rank,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* gate_logit,
        const T* U,
        const T* V,
        const T* x,
        const T* h,
        const T* d_output,
        T* dx,
        T* d_gate_logit,
        T* dU,
        T* dV,
        T* workspace);      // [5*BD + 2*BR + dim]

private:
    int batch_size_;
    int dim_;
    int seq_len_;
    int rank_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E7: Monarch Elman (O(n*sqrt(n)) hidden state updates via Monarch matrices)
// h_t = tanh(monarch(B1_h, B2_h) @ h_{t-1} + monarch(B1_x, B2_x) @ x_t + b)
// output = h * silu(W_gate @ x + b_gate)
// Key: Block-diagonal Monarch matrices give O(n*sqrt(n)) instead of O(n^2)
// =============================================================================

template<typename T>
struct MonarchElmanForward {
    MonarchElmanForward(
        bool training,
        int batch_size,
        int dim,
        int m,  // sqrt(dim), block size
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* B1_h,      // [m, m, m] monarch blocks for hidden
        const T* B2_h,      // [m, m, m]
        const T* B1_x,      // [m, m, m] monarch blocks for input
        const T* B2_x,      // [m, m, m]
        const T* W_gate,    // [dim, dim] gate projection
        const T* b,         // [dim]
        const T* b_gate,    // [dim]
        const T* x,         // [T, B, dim]
        T* h,               // [T+1, B, dim]
        T* output,          // [T, B, dim]
        T* v,               // [T, B, dim] pre-activation cache
        T* gate_cache,      // [T, B, dim] gate cache
        T* workspace);      // [2*T*BD + 3*BD] for Mx, gate_proj, tmp1, tmp2, Mh

private:
    bool training_;
    int batch_size_;
    int dim_;
    int seq_len_;
    int m_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct MonarchElmanBackward {
    MonarchElmanBackward(
        int batch_size,
        int dim,
        int m,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* B1_h,
        const T* B2_h,
        const T* B1_x,
        const T* B2_x,
        const T* W_gate,
        const T* x,
        const T* h,
        const T* v,
        const T* gate_cache,
        const T* d_output,
        T* dx,
        T* dB1_h,           // [m, m, m]
        T* dB2_h,           // [m, m, m]
        T* dB1_x,           // [m, m, m]
        T* dB2_x,           // [m, m, m]
        T* dW_gate,         // [dim, dim]
        T* db,              // [dim]
        T* d_b_gate,        // [dim]
        T* workspace);      // [2*T*BD + 5*BD + 2*dim*sizeof(float)/sizeof(T)]

private:
    int batch_size_;
    int dim_;
    int seq_len_;
    int m_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E8: Scaled Low-Rank Elman (learn to sparsify via importance scaling)
// h_t = tanh(U_h @ diag(s_h) @ V_h @ h_{t-1} + U_x @ diag(s_x) @ V_x @ x_t + b)
// output = h * silu(z)
// Key: Scale vectors s_h, s_x learn which rank components matter (sparsification)
// =============================================================================

template<typename T>
struct ScaledLowRankElmanForward {
    ScaledLowRankElmanForward(
        bool training,
        int batch_size,
        int dim,
        int rank,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* U_h,       // [dim, rank]
        const T* V_h,       // [rank, dim]
        const T* s_h,       // [rank] scale for hidden
        const T* U_x,       // [dim, rank]
        const T* V_x,       // [rank, dim]
        const T* s_x,       // [rank] scale for input
        const T* b,         // [dim]
        const T* x,         // [T, B, dim] pre-activated input
        const T* z,         // [T, B, dim] gate input
        T* h,               // [T+1, B, dim]
        T* output,          // [T, B, dim]
        T* v,               // [T, B, dim] pre-activation cache
        T* workspace);      // [T*BR + 4*BR + 2*BD]

private:
    bool training_;
    int batch_size_;
    int dim_;
    int seq_len_;
    int rank_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct ScaledLowRankElmanBackward {
    ScaledLowRankElmanBackward(
        int batch_size,
        int dim,
        int rank,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* U_h,
        const T* V_h,
        const T* s_h,
        const T* U_x,
        const T* V_x,
        const T* s_x,
        const T* x,
        const T* z,
        const T* h,
        const T* v,
        const T* d_output,
        T* dx,
        T* dz,
        T* dU_h,
        T* dV_h,
        T* ds_h,
        T* dU_x,
        T* dV_x,
        T* ds_x,
        T* db,
        T* workspace);      // [T*BD + 2*T*BR + 4*BD + 6*BR + dim + 2*rank]

private:
    int batch_size_;
    int dim_;
    int seq_len_;
    int rank_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E9: Hybrid Elman (small dense core + large diagonal memory)
// Core: h_core_t = tanh(W_x @ x_core + W_h @ h_core_prev + b)  [nonlinear mixing]
// Memory: h_mem_t = sigmoid(a) * h_mem_prev + x_mem  [linear long-range storage]
// Output: [h_core * silu(z_core), h_mem * silu(z_mem)]  (concatenated)
// Key: Large hidden state (core_dim + mem_dim) with O(core_dim²) compute
// =============================================================================

template<typename T>
struct HybridElmanForward {
    HybridElmanForward(
        bool training,
        int batch_size,
        int core_dim,
        int mem_dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x_core,   // [core_dim, core_dim]
        const T* W_h,        // [core_dim, core_dim]
        const T* b_core,     // [core_dim]
        const T* a_mem,      // [mem_dim] decay logits
        const T* x_core,     // [T, B, core_dim] pre-activated
        const T* z_core,     // [T, B, core_dim] gate
        const T* x_mem,      // [T, B, mem_dim]
        const T* z_mem,      // [T, B, mem_dim] gate
        T* h_core,           // [T+1, B, core_dim]
        T* h_mem,            // [T+1, B, mem_dim]
        T* out_core,         // [T, B, core_dim]
        T* out_mem,          // [T, B, mem_dim]
        T* v_core,           // [T, B, core_dim] pre-activation cache
        T* workspace);       // [T*B*core_dim + B*core_dim]

private:
    bool training_;
    int batch_size_;
    int core_dim_;
    int mem_dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct HybridElmanBackward {
    HybridElmanBackward(
        int batch_size,
        int core_dim,
        int mem_dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x_core,
        const T* W_h,
        const T* a_mem,
        const T* x_core,
        const T* z_core,
        const T* x_mem,
        const T* z_mem,
        const T* h_core,
        const T* h_mem,
        const T* v_core,
        const T* d_out_core,
        const T* d_out_mem,
        T* dx_core,
        T* dz_core,
        T* dx_mem,
        T* dz_mem,
        T* dW_x_core,
        T* dW_h,
        T* db_core,
        T* da_mem,
        T* workspace);      // [(T+2)*B*core + B*mem + core + mem]

private:
    int batch_size_;
    int core_dim_;
    int mem_dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E10: Multi-Scale EMA Elman (multiple EMA memory banks with learned decay)
// h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b)  -- same as E1
// m_i_t = alpha_i * m_i_prev + (1 - alpha_i) * h_t  -- k EMA banks
// out = h * silu(z) + sum_i(m_i * silu(z_i))
// Key: Multi-timescale memory with learned per-dimension decay, zero additional GEMMs
// =============================================================================

template<typename T>
struct MultiScaleElmanForward {
    MultiScaleElmanForward(
        bool training,
        int batch_size,
        int dim,
        int n_banks,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,        // [dim, dim]
        const T* W_h,        // [dim, dim]
        const T* b,          // [dim]
        const T* a,          // [n_banks, dim] EMA decay logits
        const T* x,          // [T, B, dim] pre-activated input
        const T* z,          // [T, B, (1+n_banks)*dim] gates for h and each m_i
        T* h,                // [T+1, B, dim] hidden states
        T* m,                // [T+1, n_banks, B, dim] memory banks
        T* output,           // [T, B, dim]
        T* v,                // [T, B, dim] pre-activation cache
        T* workspace);       // [T*BD + BD]

private:
    bool training_;
    int batch_size_;
    int dim_;
    int seq_len_;
    int n_banks_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct MultiScaleElmanBackward {
    MultiScaleElmanBackward(
        int batch_size,
        int dim,
        int n_banks,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,
        const T* W_h,
        const T* a,
        const T* x,
        const T* z,
        const T* h,
        const T* m,
        const T* v,
        const T* d_output,
        T* dx,
        T* dz,
        T* dW_x,
        T* dW_h,
        T* db,
        T* da,
        T* workspace);      // [(T+2)*BD + 2*n_banks*BD + dim + n_banks*dim (floats)]

private:
    int batch_size_;
    int dim_;
    int seq_len_;
    int n_banks_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E11: Selective Memory Elman (Mamba-inspired input-dependent memory)
// h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b)  -- same as E1
// a_scale = x_t @ W_a  -- decay modulation
// alpha_i = sigmoid(a_i + a_scale[:, i])  -- input-dependent decay
// m_i_t = alpha_i * m_i_prev + (1 - alpha_i) * h_t
// w = softmax(x_t @ W_w)  -- read attention
// memory_out = sum(w_i * m_i)
// out = h * silu(z_h) + memory_out * silu(z_m)  -- just 2 gates
// Key: Mamba-style selectivity with cheap projections (W_a, W_w are [dim, k])
// =============================================================================

template<typename T>
struct SelectiveElmanForward {
    SelectiveElmanForward(
        bool training,
        int batch_size,
        int dim,
        int n_banks,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,        // [dim, dim]
        const T* W_h,        // [dim, dim]
        const T* b,          // [dim]
        const T* a,          // [n_banks, dim] base decay logits
        const T* W_a,        // [dim, n_banks] decay modulation projection
        const T* W_w,        // [dim, n_banks] read weights projection
        const T* x,          // [T, B, dim] pre-activated input
        const T* z,          // [T, B, 2*dim] gates (h and memory)
        T* h,                // [T+1, B, dim] hidden states
        T* m,                // [T+1, n_banks, B, dim] memory banks
        T* output,           // [T, B, dim]
        T* v,                // [T, B, dim] pre-activation cache
        T* a_scale_cache,    // [T, B, n_banks] decay modulation cache
        T* read_weights_cache,// [T, B, n_banks] read weights cache
        T* workspace);       // [T*BD + BD + 2*BK]

private:
    bool training_;
    int batch_size_;
    int dim_;
    int seq_len_;
    int n_banks_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct SelectiveElmanBackward {
    SelectiveElmanBackward(
        int batch_size,
        int dim,
        int n_banks,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,
        const T* W_h,
        const T* a,
        const T* W_a,
        const T* W_w,
        const T* x,
        const T* z,
        const T* h,
        const T* m,
        const T* v,
        const T* a_scale_cache,
        const T* read_weights_cache,
        const T* d_output,
        T* dx,
        T* dz,
        T* dW_x,
        T* dW_h,
        T* db,
        T* da,
        T* dW_a,
        T* dW_w,
        T* workspace);      // [(T+2)*BD + 2*n_banks*BD + BK + floats]

private:
    int batch_size_;
    int dim_;
    int seq_len_;
    int n_banks_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E12: Selective Gated Elman (hidden-state-dependent gating)
// h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b)  -- same as E1
// g_t = W_g @ h_t                            -- project h for gating
// output = h_t * sigmoid(z_t + g_t)          -- selective gate
// Key: Gate depends on hidden state, not just input (like Mamba2 selectivity)
// =============================================================================

template<typename T>
struct SelectiveGatedElmanForward {
    SelectiveGatedElmanForward(
        bool training,
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,        // [dim, dim]
        const T* W_h,        // [dim, dim]
        const T* W_g,        // [dim, dim] gate projection (NEW)
        const T* b,          // [dim]
        const T* x,          // [T, B, dim] pre-activated input
        const T* z,          // [T, B, dim] gate input
        T* h,                // [T+1, B, dim] hidden states
        T* output,           // [T, B, dim]
        T* v,                // [T, B, dim] pre-activation cache
        T* gate_cache,       // [T, B, dim] gate pre-activation cache (NEW)
        T* workspace);       // [T*BD + 2*BD]

private:
    bool training_;
    int batch_size_;
    int dim_;
    int seq_len_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct SelectiveGatedElmanBackward {
    SelectiveGatedElmanBackward(
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,
        const T* W_h,
        const T* W_g,        // [dim, dim] gate projection (NEW)
        const T* x,
        const T* z,
        const T* h,
        const T* v,
        const T* gate_cache, // [T, B, dim] gate pre-activation cache (NEW)
        const T* d_output,
        T* dx,
        T* dz,
        T* dW_x,
        T* dW_h,
        T* dW_g,             // [dim, dim] gradient for gate projection (NEW)
        T* db,
        T* workspace);       // [(T+3)*BD + dim]

private:
    int batch_size_;
    int dim_;
    int seq_len_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E14: Matrix State Elman (trading weight capacity for state capacity)
// State: H ∈ ℝ^(d×k) matrix instead of vector
// key = tanh(W_key @ x)           # key ∈ ℝ^d
// value = W_val @ x               # value ∈ ℝ^k
// decay = sigmoid(W_decay @ x)    # decay ∈ ℝ^d
// H_new = decay[:,None] * H + key[:,None] * value[None,:]  # outer product
// query = W_query @ x             # query ∈ ℝ^k
// output = (H_new @ query) * silu(z)  # gated output
// Key: d*k dynamic state parameters for O(dk) element-wise cost
// =============================================================================

template<typename T>
struct MatrixStateElmanForward {
    MatrixStateElmanForward(
        bool training,
        int batch_size,
        int d,          // model dimension
        int k,          // state dimension
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_key,       // [d, d]
        const T* b_key,       // [d]
        const T* W_val,       // [d, k]
        const T* b_val,       // [k]
        const T* W_query,     // [d, k]
        const T* b_query,     // [k]
        const T* W_decay,     // [d, d]
        const T* b_decay,     // [d]
        const T* x,           // [T, B, d] pre-activated input
        const T* z,           // [T, B, d] gate input
        T* H,                 // [T+1, B, d, k] matrix hidden states
        T* output,            // [T, B, d]
        T* key_cache,         // [T, B, d] cache for backward
        T* value_cache,       // [T, B, k] cache for backward
        T* decay_cache,       // [T, B, d] cache for backward
        T* query_cache,       // [T, B, k] cache for backward
        T* workspace);        // [2*T*BD + 2*T*BK + BD + BK + BD + BK]

private:
    bool training_;
    int batch_size_;
    int d_;
    int k_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct MatrixStateElmanBackward {
    MatrixStateElmanBackward(
        int batch_size,
        int d,
        int k,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_key,
        const T* b_key,
        const T* W_val,
        const T* b_val,
        const T* W_query,
        const T* b_query,
        const T* W_decay,
        const T* b_decay,
        const T* x,
        const T* z,
        const T* H,
        const T* key_cache,
        const T* value_cache,
        const T* decay_cache,
        const T* query_cache,
        const T* d_output,
        T* dx,
        T* dz,
        T* dW_key,
        T* db_key,
        T* dW_val,
        T* db_val,
        T* dW_query,
        T* db_query,
        T* dW_decay,
        T* db_decay,
        T* workspace);        // [2*T*BD + 2*T*BK + BDK + BD + 4*BD + 4*k + 2*d + 2*k (floats)]

private:
    int batch_size_;
    int d_;
    int k_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E20: Mamba2-Informed Elman (matrix state with Mamba2 lessons)
// Key differences from E14:
// - Combined in_proj (1 GEMM) instead of 4 separate projections
// - Scalar decay per HEAD (nheads params) instead of per-element (d params)
// - No tanh in state update (only silu pre-activation on x)
// - State shape: [B, nheads, headdim, d_state]
// - E18-A style h-aware gating: output = y * silu(z + y)
//
// State update:
//   decay = sigmoid(dt + dt_bias)  # [B, nheads] scalar per head
//   H = decay * H + outer(x, B)    # [B, nheads, headdim, d_state]
//   y = einsum("bhpn,bn->bhp", H, C)  # [B, nheads, headdim]
//   output = y * silu(z + y)
// =============================================================================

template<typename T>
struct Mamba2InformedElmanForward {
    Mamba2InformedElmanForward(
        bool training,
        int batch_size,
        int nheads,
        int headdim,
        int d_state,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* x,              // [T, B, d_model]
        const T* in_proj_weight, // [d_proj, d_model] combined projection
        const T* dt_bias,        // [nheads] decay bias
        T* H,                    // [(T+1), B, nheads, headdim, d_state]
        T* output,               // [T, B, d_inner]
        T* x_proj_cache,         // [T, B, d_inner] pre-silu x for backward
        T* B_cache,              // [T, B, d_state]
        T* C_cache,              // [T, B, d_state]
        T* decay_cache,          // [T, B, nheads]
        T* workspace);           // [T*B*d_proj + T*B*d_inner + B*nheads]

private:
    bool training_;
    int batch_size_;
    int nheads_;
    int headdim_;
    int d_state_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E21: Structured Elman (MIMO with Nonlinear State Mixing)
// Key operations:
// 1. MIMO update: update[b,h,n,p] = sum_r B[b,h,n,r] * X[b,h,p,r]
// 2. Nonlinear state: H = silu(alpha * H_prev + update)
// 3. Output reduction: y = sum_n H[b,h,n,p]
// 4. E18-A gating: output = y * silu(z + y)
// =============================================================================

template<typename T>
struct StructuredElmanForward {
    StructuredElmanForward(
        bool training,
        int batch_size,
        int nheads,
        int d_state,
        int headdim,
        int mimo_rank,
        int nonlinearity_mode,  // 0=silu, 1=tanh, 2=linear
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* B_proj,      // [T, B, nheads, d_state, mimo_rank]
        const T* X_proj,      // [T, B, nheads, headdim, mimo_rank]
        const T* alpha_raw,   // [T, B, nheads]
        const T* alpha_bias,  // [nheads]
        const T* z,           // [T, B, d_inner]
        T* H,                 // [(T+1), B, nheads, d_state, headdim]
        T* output,            // [T, B, d_inner]
        T* y_cache,           // [T, B, d_inner] for backward
        T* workspace);        // minimal workspace

private:
    bool training_;
    int batch_size_;
    int nheads_;
    int d_state_;
    int headdim_;
    int mimo_rank_;
    int nonlinearity_mode_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct StructuredElmanBackward {
    StructuredElmanBackward(
        int batch_size,
        int nheads,
        int d_state,
        int headdim,
        int mimo_rank,
        int nonlinearity_mode,  // 0=silu, 1=tanh, 2=linear
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* B_proj,
        const T* X_proj,
        const T* alpha_raw,
        const T* alpha_bias,
        const T* z,
        const T* H,
        const T* y_cache,
        const T* d_output,
        T* dz,
        T* dB_proj,
        T* dX_proj,
        T* dalpha_raw,
        T* dalpha_bias,
        T* workspace);        // [BD + B_state + float_grads]

private:
    int batch_size_;
    int nheads_;
    int d_state_;
    int headdim_;
    int mimo_rank_;
    int nonlinearity_mode_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct Mamba2InformedElmanBackward {
    Mamba2InformedElmanBackward(
        int batch_size,
        int nheads,
        int headdim,
        int d_state,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* x,
        const T* in_proj_weight,
        const T* dt_bias,
        const T* H,
        const T* x_proj_cache,
        const T* B_cache,
        const T* C_cache,
        const T* decay_cache,
        const T* d_output,
        T* dx,
        T* d_in_proj_weight,
        T* d_dt_bias,
        T* workspace);           // [T*B*d_proj + state_size + d_inner + floats]

private:
    int batch_size_;
    int nheads_;
    int headdim_;
    int d_state_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E22: Structured Elman with State Attention (UTM class)
// Extends E21 with periodic state attention for state-dependent routing.
// H = silu(α * H_prev + B @ X.T)          # MIMO rank-R update
// H = H + StateAttention(H)               # Every K steps: routing via attention
// output = y * silu(z + y)                # E18-A style gating
// =============================================================================

template<typename T>
struct StructuredElmanAttentionForward {
    StructuredElmanAttentionForward(
        bool training,
        int batch_size,
        int nheads,
        int d_state,
        int headdim,
        int mimo_rank,
        int attn_period,      // K: attend every K steps
        int attn_dim,         // d_k: attention key dimension
        int nonlinearity_mode,  // 0=silu, 1=tanh, 2=linear
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* B_proj,      // [T, B, nheads, d_state, mimo_rank]
        const T* X_proj,      // [T, B, nheads, headdim, mimo_rank]
        const T* alpha_raw,   // [T, B, nheads]
        const T* alpha_bias,  // [nheads]
        const T* z,           // [T, B, d_inner]
        const T* W_q,         // [nheads, headdim, attn_dim]
        const T* W_k,         // [nheads, headdim, attn_dim]
        const T* W_v,         // [nheads, headdim, attn_dim]
        const T* W_o,         // [nheads, attn_dim, headdim]
        const T* H_init,      // [B, nheads, d_state, headdim]
        T* output,            // [T, B, d_inner]
        T* H_final,           // [B, nheads, d_state, headdim]
        T* H_all,             // [(T+1), B, nheads, d_state, headdim] for backward
        T* y_cache);          // [T, B, d_inner] for backward

private:
    bool training_;
    int batch_size_;
    int nheads_;
    int d_state_;
    int headdim_;
    int mimo_rank_;
    int attn_period_;
    int attn_dim_;
    int nonlinearity_mode_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct StructuredElmanAttentionBackward {
    StructuredElmanAttentionBackward(
        int batch_size,
        int nheads,
        int d_state,
        int headdim,
        int mimo_rank,
        int attn_period,
        int attn_dim,
        int nonlinearity_mode,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* B_proj,
        const T* X_proj,
        const T* alpha_raw,
        const T* alpha_bias,
        const T* z,
        const T* W_q,
        const T* W_k,
        const T* W_v,
        const T* W_o,
        const T* H_all,
        const T* y_cache,
        const T* d_output,
        T* dz,
        T* dB_proj,
        T* dX_proj,
        T* dalpha_raw,
        float* dW_q,              // [nheads, headdim, attn_dim] - fp32 accumulator
        float* dW_k,
        float* dW_v,
        float* dW_o);

private:
    int batch_size_;
    int nheads_;
    int d_state_;
    int headdim_;
    int mimo_rank_;
    int attn_period_;
    int attn_dim_;
    int nonlinearity_mode_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E23c: Chunked Dual-Memory Elman (Batched Attention Ops)
// Key architectural change from E23:
//   h_work_t = tanh(W_h @ h_work_{t-1} + W_x @ x_t + b)  -- NO read dependency!
//   output_t = h_work_t + read_t                         -- Additive read
// This allows batching reads and writes within chunks:
//   1. Pre-compute h_work for K steps (sequential RNN)
//   2. Batch ALL read attentions: [B, K, D] @ [B, D, N] - ONE BIG GEMM
//   3. Batch ALL write attentions from frozen tape
//   4. Parallel tape update via cumulative products
// =============================================================================

template<typename T>
struct E23cChunkedForward {
    E23cChunkedForward(
        bool training,
        int batch_size,
        int n_slots,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int seq_len,
        int chunk_size,
        const T* x_proj,       // [T, B, D] - pre-projected input
        const T* W_h,          // [D, D]
        const T* b_h,          // [D]
        const T* W_write,      // [D, D]
        const T* h_tape_init,  // [B, N, D]
        const T* h_work_init,  // [B, D]
        T* output,             // [T, B, D]
        T* h_tape_final,       // [B, N, D]
        T* h_work_all,         // [T, B, D] - all h_work states
        T* workspace);         // workspace for intermediate results

private:
    bool training_;
    int batch_size_;
    int n_slots_;
    int dim_;
    int seq_len_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E23c_v2: Chunked Dual-Memory Elman with Read Feedback
// Key difference from E23c: read_{t-1} feeds back into h_work_t
//   h_work_t = tanh(W_h @ h_work_{t-1} + W_x @ x_t + W_r @ read_{t-1} + b)
//   read_t = attention(h_work_t, tape)
//   output_t = h_work_t
// Sequential per-timestep but still batches tape updates at chunk boundaries.
// =============================================================================

template<typename T>
struct E23cv2ChunkedForward {
    E23cv2ChunkedForward(
        bool training,
        int batch_size,
        int n_slots,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int seq_len,
        int chunk_size,
        const T* x_proj,       // [T, B, D] - pre-projected input
        const T* W_h,          // [D, D]
        const T* W_r,          // [D, D] - NEW: read projection
        const T* b_h,          // [D]
        const T* W_write,      // [D, D]
        const T* h_tape_init,  // [B, N, D]
        const T* h_work_init,  // [B, D]
        T* output,             // [T, B, D]
        T* h_tape_final,       // [B, N, D]
        T* h_work_all,         // [T, B, D]
        T* workspace);

private:
    bool training_;
    int batch_size_;
    int n_slots_;
    int dim_;
    int seq_len_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E23: Dual-Memory Elman (Tape + Working Memory)
// Architecture:
//   - Tape: [B, N, D] - Large linear storage
//   - Working Memory: [B, D] - Small nonlinear compute
// Per timestep:
//   1. Read: h_work queries tape via attention
//   2. Update: h_work_new = tanh(W_h @ h_work + W_x @ x + read + b)
//   3. Write: h_tape = (1-attn)*h_tape + attn*write_value
// =============================================================================

template<typename T>
struct DualMemoryElmanForward {
    DualMemoryElmanForward(
        bool training,
        int batch_size,
        int n_slots,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int seq_len,
        const T* x_proj,          // [T, B, D] - pre-computed x @ W_x^T
        const T* W_h,             // [D, D]
        const T* b_h,             // [D]
        const T* W_write,         // [D, D]
        const T* h_tape_init,     // [B, N, D]
        const T* h_work_init,     // [B, D]
        T* h_work_out,            // [T, B, D]
        T* h_tape_final,          // [B, N, D]
        T* h_tape_all,            // [T+1, B, N, D] - tape history for backward (null if inference)
        T* read_attn,             // [T, B, N]
        T* write_attn,            // [T, B, N]
        T* workspace);           // Workspace: tmp_Rh [B, D] + tmp_write_val [B, D]

private:
    bool training_;
    int batch_size_;
    int n_slots_;
    int dim_;
    int seq_len_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct DualMemoryElmanBackward {
    DualMemoryElmanBackward(
        int batch_size,
        int n_slots,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int seq_len,
        const T* h_work_all,      // [T, B, D]
        const T* h_work_init,     // [B, D] - initial working memory (for t=0)
        const T* h_tape_all,      // [T+1, B, N, D]
        const T* read_attn,       // [T, B, N]
        const T* write_attn,      // [T, B, N]
        const T* W_h,
        const T* W_write,
        const T* d_h_work_out,    // [T, B, D]
        const T* d_h_tape_final,  // [B, N, D]
        T* dx_proj,               // [T, B, D]
        T* d_pre_act_all,         // [T, B, D] - workspace
        T* d_write_val_all,       // [T, B, D] - workspace
        float* db_h,              // [D] - fp32 accumulator
        T* d_h_tape,              // [B, N, D] - scratch
        float* dW_h,              // [D, D] - computed via GEMM
        float* dW_write);         // [D, D] - computed via GEMM

private:
    int batch_size_;
    int n_slots_;
    int dim_;
    int seq_len_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E23 Optimized Forward: cuBLAS strided batched GEMM for attention
// Uses tensor cores for attention score computation and weighted read/write.
// ~2-5x faster than naive sequential dot product approach.
// =============================================================================

template<typename T>
struct DualMemoryElmanForwardOpt {
    DualMemoryElmanForwardOpt(
        bool training,
        int batch_size,
        int n_slots,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int seq_len,
        const T* x_proj,           // [T, B, D] - pre-projected input
        const T* W_h,              // [D, D]
        const T* b_h,              // [D]
        const T* W_write,          // [D, D]
        const T* h_tape_init,      // [B, N, D]
        const T* h_work_init,      // [B, D]
        T* h_work_out,             // [T, B, D]
        T* h_tape_final,           // [B, N, D]
        T* h_tape_all,             // [T+1, B, N, D] - tape history (null if inference)
        T* read_attn,              // [T, B, N]
        T* write_attn,             // [T, B, N]
        T* workspace);             // tmp_Rh[BD], tmp_write_val[BD], tmp_scores[BN], tmp_read[BD]

private:
    bool training_;
    int batch_size_;
    int n_slots_;
    int dim_;
    int seq_len_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E24: Single-GEMM Dual Memory (1 GEMM per timestep)
// Architecture:
//   - Tape: [B, N, D] - Large linear storage
//   - Working Memory: [B, D] - Small nonlinear compute
// Key optimization: Concatenate [h_work; x] and use single [2D, 2D] GEMM
// to produce both h_update and write_val in one operation.
// Per timestep (1 GEMM!):
//   0. SINGLE GEMM: [h_work; x] @ W_all.T -> [h_update; write_val]
//   1. Read: h_work queries tape via attention
//   2. Update: h_work_new = tanh(h_update + read + b)
//   3. Write: h_tape = (1-attn)*h_tape + attn*write_val
// =============================================================================

template<typename T>
struct E24SingleGemmForward {
    E24SingleGemmForward(
        bool training,
        int batch_size,
        int n_slots,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int seq_len,
        const T* x_seq,           // [T, B, D] - input sequence
        const T* W_all,           // [2D, 2D] - fused weight matrix
        const T* b_h,             // [D]
        const T* h_tape_init,     // [B, N, D]
        const T* h_work_init,     // [B, D]
        T* h_work_out,            // [T, B, D]
        T* h_tape_final,          // [B, N, D]
        T* h_tape_all,            // [T+1, B, N, D] - tape history (null if inference)
        T* read_attn,             // [T, B, N]
        T* write_attn,            // [T, B, N]
        T* write_val_all,         // [T, B, D] - write values for backward
        T* workspace);            // input_concat [B, 2D] + gemm_output [B, 2D]

private:
    bool training_;
    int batch_size_;
    int n_slots_;
    int dim_;
    int seq_len_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E24SingleGemmBackward {
    E24SingleGemmBackward(
        int batch_size,
        int n_slots,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int seq_len,
        const T* x_seq,           // [T, B, D]
        const T* h_work_all,      // [T, B, D]
        const T* h_work_init,     // [B, D]
        const T* h_tape_all,      // [T+1, B, N, D]
        const T* read_attn,       // [T, B, N]
        const T* write_attn,      // [T, B, N]
        const T* write_val_all,   // [T, B, D]
        const T* W_all,           // [2D, 2D]
        const T* d_h_work_out,    // [T, B, D]
        const T* d_h_tape_final,  // [B, N, D]
        T* dx,                    // [T, B, D]
        float* db_h,              // [D] - accumulated
        float* dW_all,            // [2D, 2D] - accumulated
        T* workspace);            // See implementation for layout

private:
    int batch_size_;
    int n_slots_;
    int dim_;
    int seq_len_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E25: Dual-Memory Elman with 1.5-Entmax Attention
// Same as E23 but with sparse attention via 1.5-entmax instead of softmax.
// Produces exact zeros for low-scoring slots.
// =============================================================================

template<typename T>
struct E25EntmaxForward {
    E25EntmaxForward(
        bool training,
        int batch_size,
        int n_slots,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int seq_len,
        const T* x_proj,          // [T, B, D]
        const T* W_h,             // [D, D]
        const T* b_h,             // [D]
        const T* W_write,         // [D, D]
        const T* h_tape_init,     // [B, N, D]
        const T* h_work_init,     // [B, D]
        T* h_work_out,            // [T, B, D]
        T* h_tape_final,          // [B, N, D]
        T* h_tape_all,            // [T+1, B, N, D]
        T* read_attn,             // [T, B, N] - sparse
        T* write_attn,            // [T, B, N] - sparse
        T* workspace);            // tmp_Rh [B, D] + tmp_write_val [B, D]

private:
    bool training_;
    int batch_size_;
    int n_slots_;
    int dim_;
    int seq_len_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E25EntmaxBackward {
    E25EntmaxBackward(
        int batch_size,
        int n_slots,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int seq_len,
        const T* h_work_all,      // [T, B, D]
        const T* h_work_init,     // [B, D]
        const T* h_tape_all,      // [T+1, B, N, D]
        const T* read_attn,       // [T, B, N]
        const T* write_attn,      // [T, B, N]
        const T* W_h,
        const T* W_write,
        const T* d_h_work_out,    // [T, B, D]
        const T* d_h_tape_final,  // [B, N, D]
        T* dx_proj,               // [T, B, D]
        T* d_pre_act_all,         // [T, B, D]
        T* d_write_val_all,       // [T, B, D]
        float* db_h,              // [D]
        T* d_h_tape,              // [B, N, D]
        float* dW_h,              // [D, D]
        float* dW_write);         // [D, D]

private:
    int batch_size_;
    int n_slots_;
    int dim_;
    int seq_len_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E26: Parallel Dual-Memory Elman (softmax attention)
// Separates "what" (parallel GEMMs) from "where" (sequential routing).
// Same interface as E25 but with softmax instead of entmax.
// =============================================================================

template<typename T>
struct E26ParallelForward {
    E26ParallelForward(
        bool training,
        int batch_size,
        int n_slots,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int seq_len,
        const T* x_proj,          // [T, B, D]
        const T* W_h,             // [D, D]
        const T* b_h,             // [D]
        const T* W_write,         // [D, D]
        const T* h_tape_init,     // [B, N, D]
        const T* h_work_init,     // [B, D]
        T* h_work_out,            // [T, B, D]
        T* h_tape_final,          // [B, N, D]
        T* h_tape_all,            // [T+1, B, N, D]
        T* read_attn,             // [T, B, N]
        T* write_attn,            // [T, B, N]
        T* workspace);            // tmp_Rh [B, D] + tmp_write_val [B, D]

private:
    bool training_;
    int batch_size_;
    int n_slots_;
    int dim_;
    int seq_len_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E26ParallelBackward {
    E26ParallelBackward(
        int batch_size,
        int n_slots,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int seq_len,
        const T* h_work_all,      // [T, B, D]
        const T* h_work_init,     // [B, D]
        const T* h_tape_all,      // [T+1, B, N, D]
        const T* read_attn,       // [T, B, N]
        const T* write_attn,      // [T, B, N]
        const T* W_h,
        const T* W_write,
        const T* d_h_work_out,    // [T, B, D]
        const T* d_h_tape_final,  // [B, N, D]
        T* dx_proj,               // [T, B, D]
        T* d_pre_act_all,         // [T, B, D]
        T* d_write_val_all,       // [T, B, D]
        float* db_h,              // [D]
        T* d_h_tape,              // [B, N, D]
        float* dW_h,              // [D, D]
        float* dW_write);         // [D, D]

private:
    int batch_size_;
    int n_slots_;
    int dim_;
    int seq_len_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E28: E1 + Mamba2 Conv System
// x_conv = causal_conv1d(x, conv_weight)  # Depthwise K=4
// x_act = silu(x_conv)
// h_t = tanh(W_x @ x_act_t + W_h @ h_{t-1} + b)
// output = h * silu(z)
// =============================================================================

template<typename T>
class E28ConvForward {
public:
    E28ConvForward(
        int batch_size,
        int seq_len,
        int dim,
        int d_conv,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);
    ~E28ConvForward();

    void Run(
        bool training,
        const T* x,               // [B, T, D]
        const T* z,               // [B, T, D]
        const T* h_init,          // [B, D]
        const T* W_x,             // [D, D]
        const T* W_h,             // [D, D]
        const T* b,               // [D]
        const T* conv_weight,     // [D, K]
        const T* conv_bias,       // [D]
        T* h_all,                 // [B, T, D] output
        T* output,                // [B, T, D] output
        T* v_cache);              // [B, T, D] optional

private:
    int batch_size_;
    int seq_len_;
    int dim_;
    int d_conv_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
    T* tmp_x_conv_;
    T* tmp_Wx_;
    T* tmp_Rh_;
};

template<typename T>
class E28ConvBackward {
public:
    E28ConvBackward(
        int batch_size,
        int seq_len,
        int dim,
        int d_conv,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        const T* x,               // [T, B, D] time-major
        const T* z,               // [T, B, D]
        const T* h_init,          // [B, D]
        const T* h_all,           // [T, B, D]
        const T* W_x,             // [D, D]
        const T* W_h,             // [D, D]
        const T* conv_weight,     // [D, K]
        const T* conv_bias,       // [D]
        const T* d_output,        // [T, B, D]
        T* d_x,                   // [T, B, D]
        T* d_z,                   // [T, B, D]
        T* d_W_x,                 // [D, D]
        T* d_W_h,                 // [D, D]
        T* d_b,                   // [D]
        T* d_conv_weight,         // [D, K]
        T* d_conv_bias,           // [D]
        T* workspace);            // workspace buffer

    // Workspace size in number of T elements
    static int64_t WorkspaceSize(int batch_size, int seq_len, int dim, int d_conv) {
        const int64_t BD = batch_size * dim;
        const int64_t BTD = batch_size * seq_len * dim;
        // bf16: tmp_d_h(BD) + tmp_d_pre_act(BD) + d_h_accum(BD) + x_conv(BTD) + d_x_conv(BTD)
        // float (2x bf16): d_x_accum(2*BTD) + d_conv_weight_accum(2*D*K) + d_conv_bias_accum(2*D)
        return 3 * BD + 2 * BTD + 2 * BTD + 2 * dim * d_conv + 2 * dim;
    }

private:
    int batch_size_;
    int seq_len_;
    int dim_;
    int d_conv_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E29a: Selective Gating Dual-Memory Elman (additive gate)
// Extends E26 with selective output: gate = silu(z + read + h_work_new)
// =============================================================================

template<typename T>
struct E29aSelectiveForward {
    E29aSelectiveForward(
        bool training,
        int batch_size,
        int n_slots,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int seq_len,
        const T* x_proj,          // [T, B, D] pre-computed x projections
        const T* z_all,           // [T, B, D] pre-computed z values
        const T* W_h,             // [D, D]
        const T* b_h,             // [D]
        const T* W_write,         // [D, D]
        const T* h_tape_init,     // [B, N, D]
        const T* h_work_init,     // [B, D]
        T* output_all,            // [T, B, D] - selective gated output
        T* h_work_all,            // [T, B, D]
        T* h_tape_final,          // [B, N, D]
        T* h_tape_all,            // [T+1, B, N, D]
        T* read_attn_all,         // [T, B, N]
        T* write_attn_all,        // [T, B, N]
        T* workspace);            // tmp_Rh [B, D] + tmp_write_val [B, D] + tmp_read_val [B, D]

private:
    bool training_;
    int batch_size_;
    int n_slots_;
    int dim_;
    int seq_len_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E29aSelectiveBackward {
    E29aSelectiveBackward(
        int batch_size,
        int n_slots,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int seq_len,
        const T* h_work_all,       // [T, B, D]
        const T* h_work_init,      // [B, D]
        const T* h_tape_all,       // [T+1, B, N, D]
        const T* read_attn,        // [T, B, N]
        const T* write_attn,       // [T, B, N]
        const T* z_all,            // [T, B, D] - gate input
        const T* W_h,              // [D, D]
        const T* W_write,          // [D, D]
        const T* d_output_all,     // [T, B, D]
        const T* d_h_tape_final,   // [B, N, D]
        T* dx_proj,                // [T, B, D]
        T* dz,                     // [T, B, D] - gradient for z
        T* d_pre_act_all,          // [T, B, D]
        T* d_write_val_all,        // [T, B, D]
        float* db_h,               // [D]
        T* d_h_tape,               // [B, N, D]
        float* dW_h,               // [D, D]
        float* dW_write);          // [D, D]

private:
    int batch_size_;
    int n_slots_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E29b: Selective Gating Dual-Memory Elman (learned gate)
// Extends E26 with learned selective: gate = silu(W_gate @ [z; read; h_work_new])
// =============================================================================

template<typename T>
struct E29bSelectiveForward {
    E29bSelectiveForward(
        bool training,
        int batch_size,
        int n_slots,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int seq_len,
        const T* x_proj,          // [T, B, D]
        const T* z_all,           // [T, B, D]
        const T* W_h,             // [D, D]
        const T* b_h,             // [D]
        const T* W_write,         // [D, D]
        const T* W_gate,          // [D, 3*D] - learned gate projection
        const T* h_tape_init,     // [B, N, D]
        const T* h_work_init,     // [B, D]
        T* output_all,            // [T, B, D]
        T* h_work_all,            // [T, B, D]
        T* h_tape_final,          // [B, N, D]
        T* h_tape_all,            // [T+1, B, N, D]
        T* read_attn_all,         // [T, B, N]
        T* write_attn_all,        // [T, B, N]
        T* workspace);

private:
    bool training_;
    int batch_size_;
    int n_slots_;
    int dim_;
    int seq_len_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E29bSelectiveBackward {
    E29bSelectiveBackward(
        int batch_size,
        int n_slots,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int seq_len,
        const T* h_work_all,       // [T, B, D]
        const T* h_work_init,      // [B, D]
        const T* h_tape_all,       // [T+1, B, N, D]
        const T* read_attn,        // [T, B, N]
        const T* write_attn,       // [T, B, N]
        const T* z_all,            // [T, B, D]
        const T* W_h,              // [D, D]
        const T* W_write,          // [D, D]
        const T* W_gate,           // [D, 3*D]
        const T* d_output_all,     // [T, B, D]
        const T* d_h_tape_final,   // [B, N, D]
        T* dx_proj,                // [T, B, D]
        T* dz,                     // [T, B, D]
        T* d_pre_act_all,          // [T, B, D]
        T* d_write_val_all,        // [T, B, D]
        float* db_h,               // [D]
        T* d_h_tape,               // [B, N, D]
        float* dW_h,               // [D, D]
        float* dW_write,           // [D, D]
        float* dW_gate);           // [D, 3*D]

private:
    int batch_size_;
    int n_slots_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E29c: SSM-Style Diagonal Gating Dual-Memory Elman
//
// HYBRID TEMPLATE PATTERN:
//   - N_SLOTS: Templated (8, 16, 32, 64)
//   - DIM: Runtime parameter (dynamic shared memory)
//
// Gate mechanism: gate = silu(z * g_z + read * g_r + h_work * g_h + b_gate)
// Extra params: 4*D (vs 3*D² for E29b)
// =============================================================================

template<typename T>
struct E29cDiagonalForward {
    E29cDiagonalForward(
        int batch_size,
        int n_slots,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    ~E29cDiagonalForward();

    void Run(
        int seq_len,
        const T* x_proj,          // [T, B, D]
        const T* z_all,           // [T, B, D]
        const T* h_tape_init,     // [B, N, D]
        const T* h_work_init,     // [B, D]
        const T* W_h,             // [D, D]
        const T* b_h,             // [D]
        const T* W_write,         // [D, D]
        const T* g_z,             // [D] - z gate scale
        const T* g_r,             // [D] - read gate scale
        const T* g_h,             // [D] - h_work gate scale
        const T* b_gate,          // [D] - gate bias
        T* output_all,            // [T, B, D]
        T* h_work_all,            // [T, B, D]
        T* h_tape_all,            // [T+1, B, N, D]
        T* read_attn_all,         // [T, B, N]
        T* write_attn_all,        // [T, B, N]
        T* read_val_all,          // [T, B, D] - saved for backward (avoids recompute bug)
        cublasHandle_t blas_handle);

private:
    int batch_size_;
    int n_slots_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
    // Workspace pointers
    T* Rh_workspace_;
    T* h_work_workspace_;
    T* read_val_workspace_;
    T* write_val_workspace_;
};

template<typename T>
struct E29cDiagonalBackward {
    E29cDiagonalBackward(
        int batch_size,
        int n_slots,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    ~E29cDiagonalBackward();

    void Run(
        int seq_len,
        const T* h_work_all,       // [T, B, D]
        const T* h_work_init,      // [B, D]
        const T* h_tape_all,       // [T+1, B, N, D]
        const T* read_attn_all,    // [T, B, N]
        const T* write_attn_all,   // [T, B, N]
        const T* read_val_all,     // [T, B, D] - from forward (avoids recompute bug)
        const T* z_all,            // [T, B, D]
        const T* W_h,              // [D, D]
        const T* W_write,          // [D, D]
        const T* g_z,              // [D]
        const T* g_r,              // [D]
        const T* g_h,              // [D]
        const T* b_gate,           // [D]
        const T* d_output_all,     // [T, B, D]
        const T* d_h_tape_final,   // [B, N, D]
        T* dx_proj,                // [T, B, D]
        T* dz,                     // [T, B, D]
        T* d_pre_act_all,          // [T, B, D]
        T* d_write_val_all,        // [T, B, D]
        float* db_h,               // [D]
        T* d_h_tape,               // [B, N, D]
        float* dW_h,               // [D, D]
        float* dW_write,           // [D, D]
        float* dg_z,               // [D] - diagonal weight gradient
        float* dg_r,               // [D]
        float* dg_h,               // [D]
        float* db_gate);           // [D]

private:
    int batch_size_;
    int n_slots_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
    // Workspace pointers
    T* d_h_tape_workspace_;
    T* d_h_work_workspace_;
    T* d_h_work_t_workspace_;
    T* d_z_t_workspace_;
    T* d_read_val_workspace_;
    T* d_pre_act_workspace_;
    T* d_write_val_workspace_;
    T* d_h_tape_pre_write_workspace_;
    T* d_h_tape_from_write_workspace_;
    // Note: read_val_workspace_ no longer needed - we use read_val_all from forward
    float* dg_z_accum_;
    float* dg_r_accum_;
    float* dg_h_accum_;
    float* db_gate_accum_;
    float* db_h_accum_;
};

// =============================================================================
// E30: E1 + SSM-style diagonal gating
// gate = silu(z * g_z + h * g_h + b_gate)
// output = h * gate
// =============================================================================

template<typename T>
struct E30DiagonalGatedForward {
    E30DiagonalGatedForward(
        bool training,
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,             // [dim, dim]
        const T* W_h,             // [dim, dim]
        const T* b,               // [dim]
        const T* g_z,             // [dim] gate scale for z
        const T* g_h,             // [dim] gate scale for h
        const T* b_gate,          // [dim] gate bias
        const T* x,               // [T, B, dim] pre-activated input
        const T* z,               // [T, B, dim] gate input
        T* h,                     // [T+1, B, dim] hidden states
        T* output,                // [T, B, dim] output
        T* v,                     // [T, B, dim] pre-activation cache
        T* gate_input_cache,      // [T, B, dim] gate input cache for backward
        T* workspace);            // [T*B*dim + B*dim] for Wx, Rh

private:
    bool training_;
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E30DiagonalGatedBackward {
    E30DiagonalGatedBackward(
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    ~E30DiagonalGatedBackward();

    void Run(
        int steps,
        const T* W_x,
        const T* W_h,
        const T* g_z,
        const T* g_h,
        const T* x,
        const T* z,
        const T* h,
        const T* v,
        const T* gate_input_cache,
        const T* d_output,
        T* dx,
        T* dz,
        T* dW_x,
        T* dW_h,
        T* db,
        T* dg_z,
        T* dg_h,
        T* db_gate,
        T* workspace);

private:
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
    float* dg_z_accum_;
    float* dg_h_accum_;
    float* db_gate_accum_;
    float* db_accum_;
};

// =============================================================================
// E38: No W_x Elman - removes W_x matrix entirely from E33
// h_t = tanh(x_t + W_h @ h_{t-1} + b)         # NO W_x! Direct add
// output = h * silu(h)                         # Self-gating from E33
// Key: Input projection already done by in_proj, no need for W_x @ x
// =============================================================================

template<typename T>
struct E38NoWxForward {
    E38NoWxForward(
        bool training,
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_h,       // [dim, dim] - only hidden-to-hidden weight
        const T* b,         // [dim] bias
        const T* x,         // [T, B, dim] pre-activated input (goes directly into recurrence)
        T* h,               // [T+1, B, dim] hidden states
        T* output,          // [T, B, dim] output
        T* v,               // [T, B, dim] pre-activation cache
        T* workspace);      // [B*dim] for Rh only (no Wx precompute needed)

private:
    bool training_;
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E38NoWxBackward {
    E38NoWxBackward(
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_h,
        const T* x,
        const T* h,
        const T* v,
        const T* d_output,
        T* dx,              // [T, B, dim] - gradient flows directly (no W_x)
        T* dW_h,            // [dim, dim] - only hidden gradient
        T* db,              // [dim] bias gradient
        T* workspace);      // [(T+2)*B*dim + ceil(dim*4/sizeof(T))]

private:
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E39: No-Bias Elman (E38 without bias term)
// h_t = tanh(x_t + W_h @ h_{t-1})            # No W_x, no bias!
// output = h * silu(h)                        # Self-gating from E33
// Key: Simplest possible recurrence - just input + hidden transition
// =============================================================================

template<typename T>
struct E39NoBiasForward {
    E39NoBiasForward(
        bool training,
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_h,       // [dim, dim] - only hidden-to-hidden weight
        const T* x,         // [T, B, dim] pre-activated input (goes directly into recurrence)
        T* h,               // [T+1, B, dim] hidden states
        T* output,          // [T, B, dim] output
        T* v,               // [T, B, dim] pre-activation cache
        T* workspace);      // [B*dim] for Rh only (no Wx precompute needed)

private:
    bool training_;
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E39NoBiasBackward {
    E39NoBiasBackward(
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_h,
        const T* x,
        const T* h,
        const T* v,
        const T* d_output,
        T* dx,              // [T, B, dim] - gradient flows directly (no W_x)
        T* dW_h,            // [dim, dim] - only hidden gradient
        T* workspace);      // [(T+2)*B*dim]

private:
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E40: No Pre-SiLU Elman (E38 without pre-activation silu)
// x_proj = in_proj(x)                        # No silu! Direct projection
// h_t = tanh(x_t + W_h @ h_{t-1} + b)        # No W_x, raw x goes in
// output = h * silu(h)                        # Self-gating from E33
// Key: Testing if pre-silu is needed when W_x is already removed
// =============================================================================

template<typename T>
struct E40NoPresiluForward {
    E40NoPresiluForward(
        bool training,
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_h,       // [dim, dim] - only hidden-to-hidden weight
        const T* b,         // [dim]
        const T* x,         // [T, B, dim] input (NOT pre-activated, no silu)
        T* h,               // [T+1, B, dim] hidden states
        T* output,          // [T, B, dim] output
        T* v,               // [T, B, dim] pre-activation cache
        T* workspace);      // [B*dim] for Rh only (no Wx precompute needed)

private:
    bool training_;
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E40NoPresiluBackward {
    E40NoPresiluBackward(
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_h,
        const T* x,
        const T* h,
        const T* v,
        const T* d_output,
        T* dx,              // [T, B, dim] - gradient flows directly (no W_x)
        T* dW_h,            // [dim, dim] - only hidden gradient
        T* db,              // [dim]
        T* workspace);      // [(T+2)*B*dim + ceil(dim*4/sizeof(T))]

private:
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty

#endif  // HASTY_ELMAN_LADDER_H
