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
// E37v2: Optimized Tied Weights Elman - Uses W @ x + W @ h instead of W @ (x + h)
// Same math as E37 but allows batching W @ x across all timesteps (like E33)
// This recovers the performance lost in E37 due to sequential W @ (x + h)
// =============================================================================

template<typename T>
struct E37TiedWeightsV2Forward {
    E37TiedWeightsV2Forward(
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
        T* workspace);      // [(T+1)*B*dim] for tmp_Wx (T*BD), tmp_Rh (BD)

private:
    bool training_;
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E37TiedWeightsV2Backward {
    E37TiedWeightsV2Backward(
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
        T* workspace);      // [(T+2)*B*dim + ceil(dim*4/sizeof(T))]

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
// E42: Linear Tied Self-Gated Elman
// Combines E36 (linear recurrence) + E37 (tied weights) + E37v2 batched GEMM
// h_t = W @ x_t + W @ h_{t-1} + b        # LINEAR recurrence, tied (NO tanh!)
// output = h * silu(h)                    # Self-gating (only nonlinearity)
// Uses E37v2 pattern: batch W @ x for all timesteps, W @ h_prev per step
// =============================================================================

template<typename T>
struct E42LinearTiedForward {
    E42LinearTiedForward(
        bool training,
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W,         // [dim, dim] - SINGLE weight matrix (tied)
        const T* b,         // [dim]
        const T* x,         // [T, B, dim] pre-activated input
        T* h,               // [T+1, B, dim] hidden states
        T* output,          // [T, B, dim] output
        T* v,               // [T, B, dim] pre-activation cache
        T* workspace);      // [(T+1)*B*dim] for tmp_Wx (T*BD), tmp_Wh (BD)

private:
    bool training_;
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E42LinearTiedBackward {
    E42LinearTiedBackward(
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
        T* dW,              // [dim, dim] - single gradient (tied)
        T* db,
        T* workspace);      // [(T+2)*B*dim + ceil(dim*4/sizeof(T))]

private:
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
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

// =============================================================================
// E54: Diagonal Decay + No Projections Elman
// h_t = d * (x_t + h_{t-1}) + b   # Per-dimension decay (d is [dim] vector)
// output = h * silu(h)           # Self-gating
// Key: NO GEMM - pure element-wise ops. Mamba2-style diagonal without complexity.
// =============================================================================

template<typename T>
struct E54DiagonalNoProjForward {
    E54DiagonalNoProjForward(
        bool training,
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* d,         // [dim] - per-dimension decay (already sigmoid)
        const T* b,         // [dim] - bias
        const T* x,         // [T, B, dim] - input (already silu'd)
        T* h,               // [T+1, B, dim] - hidden states
        T* output,          // [T, B, dim] - output
        T* v,               // [T, B, dim] - pre-activation cache
        T* workspace);      // [0] - no workspace needed!

private:
    bool training_;
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E54DiagonalNoProjBackward {
    E54DiagonalNoProjBackward(
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* d,         // [dim] - decay
        const T* x,         // [T, B, dim]
        const T* h,         // [T+1, B, dim]
        const T* v,         // [T, B, dim] - unused but kept for API
        const T* d_output,  // [T, B, dim]
        T* dx,              // [T, B, dim]
        T* dd,              // [dim] - gradient for decay
        T* db,              // [dim] - gradient for bias
        T* workspace);      // [2*B*dim + 2*dim*sizeof(float)/sizeof(T)]

private:
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E55: Scalar Decay + No Projections Elman
// h_t = lambda * (x_t + h_{t-1}) + b   # SINGLE scalar lambda!
// output = h * silu(h)                  # Self-gating
// Key: Ultimate minimal - 1 scalar lambda, no GEMM, pure element-wise.
// Total recurrence params: 1 (lambda) + dim (b) = dim + 1
// =============================================================================

template<typename T>
struct E55ScalarNoProjForward {
    E55ScalarNoProjForward(
        bool training,
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const float lambda, // Single scalar (already sigmoid)
        const T* b,         // [dim] - bias
        const T* x,         // [T, B, dim] - input (already silu'd)
        T* h,               // [T+1, B, dim] - hidden states
        T* output,          // [T, B, dim] - output
        T* v,               // [T, B, dim] - pre-activation cache
        T* workspace);      // [0] - no workspace needed!

private:
    bool training_;
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E55ScalarNoProjBackward {
    E55ScalarNoProjBackward(
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const float lambda, // Single scalar
        const T* x,         // [T, B, dim]
        const T* h,         // [T+1, B, dim]
        const T* v,         // [T, B, dim] - unused but kept for API
        const T* d_output,  // [T, B, dim]
        T* dx,              // [T, B, dim]
        float* dlambda,     // [1] - gradient for scalar lambda (float output!)
        T* db,              // [dim] - gradient for bias
        T* workspace);      // [2*B*dim + dim*sizeof(float)/sizeof(T)]

private:
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E48: No Projections Elman
// Removes BOTH in_proj and out_proj - operates directly on embedding space.
// h_t = W @ (x_t + h_{t-1}) + b   # W is dim×dim, operates on embeddings
// output = h * silu(h)            # Self-gating (only nonlinearity)
// y = output                       # Direct to residual (no out_proj!)
// =============================================================================

template<typename T>
struct E48NoProjectionsForward {
    E48NoProjectionsForward(
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
        T* workspace);      // [(T+1)*B*dim] for tmp_Wx (T*BD), tmp_Wh (BD)

private:
    bool training_;
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E48NoProjectionsBackward {
    E48NoProjectionsBackward(
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
        T* workspace);      // [(2*T+2)*B*dim + ceil(dim*4/sizeof(T))]

private:
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E51: No Self-Gate Elman
// Tests if self-gating is necessary: removes h * silu(h) and uses linear output.
// h_t = W @ (x_t + h_{t-1}) + b    # Same as E42
// output = h_t                      # LINEAR! No gating!
// =============================================================================

template<typename T>
struct E51NoSelfGateForward {
    E51NoSelfGateForward(
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
        T* output,          // [T, B, dim] output (= h, no gating!)
        T* v,               // [T, B, dim] pre-activation cache
        T* workspace);      // [(T+1)*B*dim] for tmp_Wx (T*BD), tmp_Wh (BD)

private:
    bool training_;
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E51NoSelfGateBackward {
    E51NoSelfGateBackward(
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
        T* workspace);      // [(2*T+2)*B*dim + ceil(dim*4/sizeof(T))]

private:
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E45: Pure Accumulation Elman (NO GEMM - simplest possible)
// h_t = x_t + h_{t-1}                   # Just add! No parameters in recurrence!
// output = h_t * silu(h_t)               # Self-gating (only nonlinearity)
// Key: NO GEMM at all! Pure element-wise operations.
// =============================================================================

template<typename T>
struct E45PureAccumulationForward {
    E45PureAccumulationForward(
        bool training,
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* x,         // [T, B, dim] pre-activated input
        T* h,               // [T+1, B, dim] hidden states
        T* output,          // [T, B, dim] output
        T* workspace);      // None needed, but kept for API consistency

private:
    bool training_;
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E45PureAccumulationBackward {
    E45PureAccumulationBackward(
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* h,
        const T* d_output,
        T* dx,
        T* workspace);      // [2*B*dim] for dh, dh_recurrent

private:
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E45b: Pure Accumulation with Decay (learned scalar alpha)
// h_t = x_t + alpha * h_{t-1}           # Decay prevents unbounded growth
// output = h_t * silu(h_t)               # Same self-gating
// =============================================================================

template<typename T>
struct E45bWithDecayForward {
    E45bWithDecayForward(
        bool training,
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const float alpha,  // Scalar decay rate (from sigmoid(log_alpha))
        const T* x,         // [T, B, dim] pre-activated input
        T* h,               // [T+1, B, dim] hidden states
        T* output,          // [T, B, dim] output
        T* workspace);      // None needed

private:
    bool training_;
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E45bWithDecayBackward {
    E45bWithDecayBackward(
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const float alpha,
        const T* h,
        const T* d_output,
        T* dx,
        float* d_alpha,     // Gradient for alpha (scalar)
        T* workspace);      // [2*B*dim] for dh, dh_recurrent

private:
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E46: No In-Projection Elman (operates directly on embeddings)
// h_t = W @ (x_t + h_{t-1}) + b         # W is dim*dim, no expansion
// output = h_t * silu(h_t)               # Self-gating
// Key: Removes in_proj - W can incorporate the projection's role
// =============================================================================

template<typename T>
struct E46NoInProjForward {
    E46NoInProjForward(
        bool training,
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W,         // [dim, dim] - weight matrix
        const T* b,         // [dim] - bias
        const T* x,         // [T, B, dim] - input (pre-activated with silu)
        T* h,               // [T+1, B, dim] - hidden states
        T* output,          // [T, B, dim] - output
        T* v,               // [T, B, dim] - pre-activation cache (training only)
        T* workspace);      // [2*B*dim] for tmp_sum, tmp_Wsum

private:
    bool training_;
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E46NoInProjBackward {
    E46NoInProjBackward(
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
        T* dW,              // [dim, dim] - weight gradient
        T* db,              // [dim] - bias gradient
        T* workspace);      // [(2*T+2)*B*dim + ceil(dim*4/sizeof(T))]

private:
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E52: Quadratic Gate Elman
// Tests if sigmoid in self-gate matters: uses pure h^2 instead of h * silu(h).
// h_t = W @ x_t + W @ h_{t-1} + b        # LINEAR recurrence, tied (NO tanh!)
// output = h^2                            # Pure quadratic (E52)
// output = h * |h|                        # Signed quadratic (E52b)
// =============================================================================

template<typename T>
struct E52QuadraticGateForward {
    E52QuadraticGateForward(
        bool training,
        int batch_size,
        int dim,
        bool signed_quadratic,  // false = h^2, true = h*|h|
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W,         // [dim, dim] - SINGLE weight matrix (tied)
        const T* b,         // [dim]
        const T* x,         // [T, B, dim] pre-activated input
        T* h,               // [T+1, B, dim] hidden states
        T* output,          // [T, B, dim] output
        T* v,               // [T, B, dim] pre-activation cache
        T* workspace);      // [(T+1)*B*dim] for tmp_Wx (T*BD), tmp_Wh (BD)

private:
    bool training_;
    int batch_size_;
    int dim_;
    bool signed_quadratic_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E52QuadraticGateBackward {
    E52QuadraticGateBackward(
        int batch_size,
        int dim,
        bool signed_quadratic,  // false = h^2, true = h*|h|
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
        T* dW,              // [dim, dim] - single gradient (tied)
        T* db,
        T* workspace);      // [(2*T+2)*B*dim + ceil(dim*4/sizeof(T))]

private:
    int batch_size_;
    int dim_;
    bool signed_quadratic_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E53: Sigmoid Gate Only Elman
// Tests if the quadratic component matters: uses silu(h) instead of h * silu(h).
// h_t = W @ x_t + W @ h_{t-1} + b        # LINEAR recurrence, tied (NO tanh!)
// output = silu(h)                        # Just silu, NOT h * silu(h)!
// =============================================================================

template<typename T>
struct E53SigmoidGateForward {
    E53SigmoidGateForward(
        bool training,
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W,         // [dim, dim] - SINGLE weight matrix (tied)
        const T* b,         // [dim]
        const T* x,         // [T, B, dim] pre-activated input
        T* h,               // [T+1, B, dim] hidden states
        T* output,          // [T, B, dim] output
        T* v,               // [T, B, dim] pre-activation cache
        T* workspace);      // [(T+1)*B*dim] for tmp_Wx (T*BD), tmp_Wh (BD)

private:
    bool training_;
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E53SigmoidGateBackward {
    E53SigmoidGateBackward(
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
        T* dW,              // [dim, dim] - single gradient (tied)
        T* db,
        T* workspace);      // [(2*T+2)*B*dim + ceil(dim*4/sizeof(T))]

private:
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};


// =============================================================================
// E43: Scalar Decay Elman
// The most radical W simplification: a single scalar λ replaces the d×d matrix.
// h_t = λ * (x_t + h_{t-1}) + b         # Scalar decay (NO matrix!)
// output = h_t * silu(h_t)               # Self-gating (only nonlinearity)
// λ = sigmoid(log_lambda) ∈ (0, 1) for stability
// =============================================================================

template<typename T>
struct E43ScalarDecayForward {
    E43ScalarDecayForward(
        bool training,
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* log_lambda,    // [1] - log(sigmoid^-1(λ))
        const T* b,             // [dim]
        const T* x,             // [T, B, dim] pre-activated input
        T* h,                   // [T+1, B, dim] hidden states
        T* output,              // [T, B, dim] output
        T* v,                   // [T, B, dim] cache (stores x + h_prev)
        T* workspace);          // unused

private:
    bool training_;
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E43ScalarDecayBackward {
    E43ScalarDecayBackward(
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* log_lambda,
        const T* x,
        const T* h,
        const T* v,
        const T* d_output,
        T* dx,
        T* d_log_lambda,        // [1] gradient for log_lambda
        T* db,
        T* workspace);          // [2*B*dim + dim + 1]

private:
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E44: Diagonal W Elman (Mamba2-style)
// Per-dimension decay rates, no cross-dimension mixing.
// h_t = d * (x_t + h_{t-1}) + b         # d is [dim] vector, element-wise
// output = h_t * silu(h_t)               # Self-gating (only nonlinearity)
// d = sigmoid(log_d) ∈ (0, 1)^dim for stability
// =============================================================================

template<typename T>
struct E44DiagonalWForward {
    E44DiagonalWForward(
        bool training,
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* log_d,         // [dim] - per-dimension log decay
        const T* b,             // [dim]
        const T* x,             // [T, B, dim] pre-activated input
        T* h,                   // [T+1, B, dim] hidden states
        T* output,              // [T, B, dim] output
        T* v,                   // [T, B, dim] cache (stores x + h_prev)
        T* workspace);          // unused

private:
    bool training_;
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E44DiagonalWBackward {
    E44DiagonalWBackward(
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* log_d,
        const T* x,
        const T* h,
        const T* v,
        const T* d_output,
        T* dx,
        T* d_log_d,             // [dim] gradient for log_d
        T* db,
        T* workspace);          // [2*B*dim + 2*dim]

private:
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E56: Concat Elman - Single GEMM on concatenated [x, h] input
// h_t = tanh(W @ [x_t; h_{t-1}] + b)  # Single GEMM with W [dim, 2*dim]
// output = h * silu(z)                 # Gate with z branch
// Key: Same params as E1, but one GEMM instead of two per timestep.
// =============================================================================

template<typename T>
struct E56ConcatElmanForward {
    E56ConcatElmanForward(
        bool training,
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W,         // [dim, 2*dim] - single weight matrix
        const T* b,         // [dim] - bias
        const T* x,         // [T, B, dim] pre-activated input
        const T* z,         // [T, B, dim] gate input
        T* h,               // [T+1, B, dim] hidden states
        T* output,          // [T, B, dim] output
        T* v,               // [T, B, dim] pre-activation cache
        T* workspace);      // [B*2*dim + B*dim] for xh_concat, Wxh

private:
    bool training_;
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E56ConcatElmanBackward {
    E56ConcatElmanBackward(
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W,         // [dim, 2*dim]
        const T* x,         // [T, B, dim]
        const T* z,         // [T, B, dim]
        const T* h,         // [T+1, B, dim]
        const T* v,         // [T, B, dim]
        const T* d_output,  // [T, B, dim]
        T* dx,              // [T, B, dim]
        T* dz,              // [T, B, dim]
        T* dW,              // [dim, 2*dim]
        T* db,              // [dim]
        T* workspace);      // [(T+6)*B*dim + ceil(dim*4/sizeof(T))]

private:
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E58: Per-Dimension Learned Radii Elman
// h_t = tanh(W_x @ x_t + (W_h * radii.unsqueeze(1)) @ h_{t-1} + b)
// output = h * silu(z)
// Key: Each hidden dimension has its own learned spectral radius
// =============================================================================

template<typename T>
struct E58LearnedRadiiForward {
    E58LearnedRadiiForward(
        bool training,
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,       // [dim, dim]
        const T* W_h,       // [dim, dim] (unscaled)
        const T* radii,     // [dim] per-dimension scaling factors
        const T* b,         // [dim]
        const T* x,         // [T, B, dim] pre-activated input
        const T* z,         // [T, B, dim] gate input (pre silu)
        T* h,               // [T+1, B, dim] hidden states
        T* output,          // [T, B, dim] output
        T* v,               // [T, B, dim] pre-activation cache
        T* Rh_cache,        // [T, B, dim] cache W_h @ h for backward
        T* workspace);      // [T*B*dim + B*dim] for Wx, Rh

private:
    bool training_;
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E58LearnedRadiiBackward {
    E58LearnedRadiiBackward(
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,
        const T* W_h,
        const T* radii,     // [dim] per-dimension scaling
        const T* x,
        const T* z,
        const T* h,
        const T* v,
        const T* Rh_cache,  // [T, B, dim] from forward
        const T* d_output,
        T* dx,
        T* dz,
        T* dW_x,
        T* dW_h,
        T* d_radii,         // [dim] gradient for radii
        T* db,
        T* workspace);      // [(2*T+2)*B*dim + 2*dim*sizeof(float)/sizeof(T)]

private:
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E59: Highway Elman - Residual Recurrence with Perfect Gradient Flow
// h_t = h_{t-1} + alpha * (W @ x_t + b)   # Residual accumulation (gradient = I)
// output_t = h_t * silu(h_t)              # Nonlinearity at output only
// Where alpha = exp(log_alpha) is a learned positive scalar.
// Key: The Jacobian dh_t/dh_{t-1} = I (identity), providing perfect gradient flow.
// =============================================================================

template<typename T>
struct E59HighwayForward {
    E59HighwayForward(
        bool training,
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const float alpha,  // exp(log_alpha) - positive scalar
        const T* W,         // [dim, dim]
        const T* b,         // [dim]
        const T* x,         // [T, B, dim] pre-activated input
        T* h,               // [T+1, B, dim] hidden states
        T* output,          // [T, B, dim] output
        T* Wx_cache,        // [T, B, dim] cache of W@x+b for backward (training only)
        T* workspace);      // [T*B*dim] for Wx_all

private:
    bool training_;
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E59HighwayBackward {
    E59HighwayBackward(
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const float alpha,
        const T* W,
        const T* x,
        const T* h,
        const T* Wx_cache,      // [T, B, dim] from forward
        const T* d_output,
        T* dx,
        T* dW,                  // [dim, dim]
        T* db,                  // [dim]
        float* d_log_alpha,     // [1] gradient for log_alpha
        T* workspace);          // [(T+2)*B*dim + dim*sizeof(float)/sizeof(T)]

private:
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E60: Residual Nonlinear Elman
// h_t = h_{t-1} + alpha * tanh(W_h @ h_{t-1} + W_x @ x_t + b)
// output = h_t * silu(h_t)
// alpha = exp(log_alpha) is a learned positive scalar
// =============================================================================

template<typename T>
struct E60ResidualNonlinearForward {
    E60ResidualNonlinearForward(
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
        const float* log_alpha,  // [1] scalar (in log space for positivity)
        const T* x,         // [T, B, dim] pre-activated input
        T* h,               // [T+1, B, dim] hidden states
        T* output,          // [T, B, dim] output
        T* tanh_cache,      // [T, B, dim] stores tanh values for backward
        T* workspace);      // [T*B*dim + B*dim] for Wx, Rh

private:
    bool training_;
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E60ResidualNonlinearBackward {
    E60ResidualNonlinearBackward(
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,
        const T* W_h,
        const float* log_alpha,  // [1] scalar
        const T* x,
        const T* h,
        const T* tanh_cache,
        const T* d_output,
        T* dx,
        T* dW_x,
        T* dW_h,
        T* db,
        float* d_log_alpha,  // [1] gradient for log_alpha
        T* workspace);       // [(T+2)*B*dim + dim*sizeof(float)/sizeof(T) + 1]

private:
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E61: Decay-Gated Elman - Mamba2-style Input-Dependent Decay
// alpha_t = sigmoid(x @ W_alpha.T + b_alpha)    # Decay gate (input-dependent)
// v_t = x @ W_v.T + b_v                         # New value (linear, no tanh)
// h_t = alpha_t * h_{t-1} + (1 - alpha_t) * v_t # Gated update
// output = h * silu(h)                          # Self-gating
//
// Linear in h: Jacobian dh_t/dh_{t-1} = diag(alpha_t)
// Parallelizable via associative scan (future optimization)
// =============================================================================

template<typename T>
struct E61DecayGatedForward {
    E61DecayGatedForward(
        bool training,
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_alpha,   // [dim, dim] decay gate weight
        const T* b_alpha,   // [dim] decay gate bias
        const T* W_v,       // [dim, dim] value weight
        const T* b_v,       // [dim] value bias
        const T* x,         // [T, B, dim] pre-activated input
        T* h,               // [T+1, B, dim] hidden states
        T* output,          // [T, B, dim] output
        T* alpha_cache,     // [T, B, dim] stores alpha values for backward
        T* workspace);      // [2*T*B*dim] for alpha_logits, v

private:
    bool training_;
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E61DecayGatedBackward {
    E61DecayGatedBackward(
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_alpha,
        const T* W_v,
        const T* b_v,
        const T* x,
        const T* h,
        const T* alpha_cache,
        const T* d_output,
        T* dx,
        T* dW_alpha,
        T* db_alpha,
        T* dW_v,
        T* db_v,
        T* workspace);      // [2*BD + 3*T*BD + 2*dim*sizeof(float)/sizeof(T)]

private:
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E62: Selective Write Elman
// Vector analog of DeltaNet's selective memory updates.
//
// k_t = sigmoid(W_k @ x_t + b_k)     # Selection mask (0-1 per dimension)
// v_t = tanh(W_v @ x_t + b_v)        # New values
// h_t = (1 - k_t) * h_{t-1} + k_t * v_t   # Selective replacement
// output_t = h_t * silu(h_t)         # Self-gating
//
// This is LINEAR in h - potentially parallelizable with associative scan.
// =============================================================================

template<typename T>
struct E62SelectiveWriteForward {
    E62SelectiveWriteForward(
        bool training,
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_k,       // [dim, dim] selection weight
        const T* b_k,       // [dim] selection bias
        const T* W_v,       // [dim, dim] value weight
        const T* b_v,       // [dim] value bias
        const T* x,         // [T, B, dim] pre-activated input
        T* h,               // [T+1, B, dim] hidden states
        T* output,          // [T, B, dim] output
        T* k_cache,         // [T, B, dim] stores k values for backward
        T* v_cache,         // [T, B, dim] stores v values for backward
        T* workspace);      // [2*T*B*dim] for Wk_x, Wv_x

private:
    bool training_;
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E62SelectiveWriteBackward {
    E62SelectiveWriteBackward(
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_k,
        const T* W_v,
        const T* x,
        const T* h,
        const T* k_cache,
        const T* v_cache,
        const T* d_output,
        T* dx,
        T* dW_k,
        T* db_k,
        T* dW_v,
        T* db_v,
        T* workspace);      // [(2*T+2)*B*dim + 2*dim*sizeof(float)/sizeof(T)]

private:
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E63: Nonlinear Delta Elman (UTM-Class Expressivity)
// Key insight: E63 adds nonlinear h-dependence while preserving gated gradient control.
// This makes E63 Turing complete while E61/E62 are not.
//
// E63a (Complementary) Architecture:
// alpha_t = sigmoid(W_alpha @ x_t + b_alpha)           # Retain gate (x-only)
// v_t = tanh(W_h @ h_{t-1} + W_x @ x_t + b)           # NONLINEAR value (h-dependent!)
// h_t = alpha_t * h_{t-1} + (1 - alpha_t) * v_t       # Gated mixing
// output = h * silu(h)                                 # Self-gating
// =============================================================================

template<typename T>
struct E63NonlinearDeltaForward {
    E63NonlinearDeltaForward(
        bool training,
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_alpha,     // [dim, dim] retain gate weight
        const T* b_alpha,     // [dim] retain gate bias
        const T* W_h,         // [dim, dim] hidden-to-value weight (nonlinear path!)
        const T* W_x,         // [dim, dim] input-to-value weight
        const T* b,           // [dim] value bias
        const T* x,           // [T, B, dim] pre-activated input
        T* h,                 // [T+1, B, dim] hidden states
        T* output,            // [T, B, dim] output
        T* v_pre_cache,       // [T, B, dim] pre-tanh value cache (training only)
        T* alpha_cache,       // [T, B, dim] alpha cache (training only)
        T* workspace);        // [2*T*B*dim + B*dim] for alpha_x_all, Wx_all, tmp_Wh

    // Workspace size calculation
    static int64_t WorkspaceSize(int steps, int batch_size, int dim) {
        return 2 * steps * batch_size * dim +  // alpha_x_all + Wx_all
               batch_size * dim;                // tmp_Wh
    }

private:
    bool training_;
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E63NonlinearDeltaBackward {
    E63NonlinearDeltaBackward(
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_alpha,
        const T* W_h,
        const T* W_x,
        const T* x,
        const T* h,
        const T* v_pre_cache,
        const T* alpha_cache,
        const T* d_output,
        T* dx,
        T* dW_alpha,
        T* db_alpha,
        T* dW_h,
        T* dW_x,
        T* db,
        T* workspace);       // See WorkspaceSize

    // Workspace size calculation
    static int64_t WorkspaceSize(int steps, int batch_size, int dim) {
        // dh, dh_recurrent, dh_prev: 3*B*dim
        // dv_pre_all, dalpha_x_all: 2*T*B*dim
        // db_float, db_alpha_float: 2*dim floats
        // alpha_x_all: T*B*dim (for recompute)
        int64_t float_bytes = 2 * dim * sizeof(float);
        int64_t float_in_T = (float_bytes + sizeof(T) - 1) / sizeof(T);
        return 3 * batch_size * dim +
               2 * steps * batch_size * dim +
               float_in_T +
               steps * batch_size * dim;
    }

private:
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};


// =============================================================================
// E64: Additive H-Dependence - Cheapest UTM-Class Recurrence
// The simplest way to get h into the nonlinearity:
//    v_t = tanh(h_{t-1} + W_x @ x_t + b)
//
// Cost: O(d) per step vs O(d^2) for E63's W_h @ h
// Still UTM-class: h is inside the tanh!
//
// Architecture:
// alpha_t = sigmoid(W_alpha @ x_t + b_alpha)    # Retain gate (x-only)
// v_t = tanh(h_{t-1} + W_x @ x_t + b)          # ADDITIVE h-dependence (no W_h!)
// h_t = alpha_t * h_{t-1} + (1 - alpha_t) * v_t  # Gated mixing
// output = h * silu(h)                          # Self-gating
// =============================================================================

template<typename T>
struct E64AdditiveHForward {
    E64AdditiveHForward(
        bool training,
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_alpha,     // [dim, dim] retain gate weight
        const T* b_alpha,     // [dim] retain gate bias
        const T* W_x,         // [dim, dim] input-to-value weight
        const T* b,           // [dim] value bias
        const T* x,           // [T, B, dim] pre-activated input
        T* h,                 // [T+1, B, dim] hidden states
        T* output,            // [T, B, dim] output
        T* v_pre_cache,       // [T, B, dim] pre-tanh value cache (training only)
        T* alpha_cache,       // [T, B, dim] alpha cache (training only)
        T* workspace);        // [2*T*B*dim] for alpha_x_all, Wx_all

    // Workspace size calculation
    static int64_t WorkspaceSize(int steps, int batch_size, int dim) {
        return 2 * steps * batch_size * dim;  // alpha_x_all + Wx_all
    }

private:
    bool training_;
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E64AdditiveHBackward {
    E64AdditiveHBackward(
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_alpha,
        const T* W_x,
        const T* x,
        const T* h,
        const T* v_pre_cache,
        const T* alpha_cache,
        const T* d_output,
        T* dx,
        T* dW_alpha,
        T* db_alpha,
        T* dW_x,
        T* db,
        T* workspace);       // See WorkspaceSize

    // Workspace size calculation
    static int64_t WorkspaceSize(int steps, int batch_size, int dim) {
        // dh, dh_recurrent, dh_prev: 3*B*dim
        // dWx_pre_all, dalpha_x_all: 2*T*B*dim
        // db_float, db_alpha_float: 2*dim floats
        int64_t float_bytes = 2 * dim * sizeof(float);
        int64_t float_in_T = (float_bytes + sizeof(T) - 1) / sizeof(T);
        return 3 * batch_size * dim +
               2 * steps * batch_size * dim +
               float_in_T;
    }

private:
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};
// =============================================================================
// E67: H-Dependent Gate Only - Nonlinearity Through Gate Selection
//
// Put h-dependence in the GATE, not the value:
//     alpha_t = sigmoid(W_alpha @ x_t + d_alpha * h_{t-1} + b_alpha)   # h affects gate!
//     v_t = tanh(W_x @ x_t + b_v)                                      # v is h-independent
//     h_t = alpha_t * h_{t-1} + (1 - alpha_t) * v_t
//     output = h * silu(h)                                             # Self-gating
//
// Key insight: Gate h-dependence through diagonal d_alpha (O(d) per step).
// UTM-class because gate depends nonlinearly on h through sigmoid.
// =============================================================================

template<typename T>
struct E67HGatedForward {
    E67HGatedForward(
        bool training,
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_alpha,     // [dim, dim] gate weight
        const T* d_alpha,     // [dim] diagonal for h in gate
        const T* b_alpha,     // [dim] gate bias
        const T* W_x,         // [dim, dim] value weight
        const T* b_v,         // [dim] value bias
        const T* x,           // [T, B, dim] pre-activated input
        T* h,                 // [T+1, B, dim] hidden states
        T* output,            // [T, B, dim] output
        T* v_cache,           // [T, B, dim] value cache (training only)
        T* alpha_cache,       // [T, B, dim] alpha cache (training only)
        T* workspace);        // [2*T*B*dim] for alpha_x_all, v_all

    // Workspace size calculation
    static int64_t WorkspaceSize(int steps, int batch_size, int dim) {
        return 2 * steps * batch_size * dim;  // alpha_x_all + v_all
    }

private:
    bool training_;
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E67HGatedBackward {
    E67HGatedBackward(
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_alpha,
        const T* d_alpha,
        const T* W_x,
        const T* x,
        const T* h,
        const T* v_cache,
        const T* alpha_cache,
        const T* d_output,
        T* dx,
        T* dW_alpha,
        T* dd_alpha,
        T* db_alpha,
        T* dW_x,
        T* db_v,
        T* workspace);       // See WorkspaceSize

    // Workspace size calculation
    static int64_t WorkspaceSize(int steps, int batch_size, int dim) {
        // dh, dh_recurrent, dh_prev: 3*B*dim
        // dalpha_x_all, dWx_pre_all: 2*T*B*dim
        // dd_alpha_float, db_alpha_float, db_v_float: 3*dim floats
        int64_t float_bytes = 3 * dim * sizeof(float);
        int64_t float_in_T = (float_bytes + sizeof(T) - 1) / sizeof(T);
        return 3 * batch_size * dim +
               2 * steps * batch_size * dim +
               float_in_T;
    }

private:
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// GRU: Gated Recurrent Unit - BF16 Optimized
// Custom implementation to avoid cuDNN's bfloat16 performance regression.
//
// GRU Equations:
// z_t = sigmoid(W_z @ x_t + U_z @ h_{t-1} + b_z)  # update gate
// r_t = sigmoid(W_r @ x_t + U_r @ h_{t-1} + b_r)  # reset gate
// h_tilde = tanh(W_h @ x_t + U_h @ (r_t * h_{t-1}) + b_h)  # candidate
// h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde  # new hidden state
// =============================================================================

template<typename T>
struct GRUForward {
    GRUForward(
        bool training,
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_zr,         // [2*dim, dim] - W_z and W_r stacked
        const T* W_h,          // [dim, dim]
        const T* U_zr,         // [2*dim, dim] - U_z and U_r stacked
        const T* U_h,          // [dim, dim]
        const T* b_zr,         // [2*dim] - b_z and b_r
        const T* b_h,          // [dim]
        const T* x,            // [T, B, dim]
        T* h,                  // [T+1, B, dim] hidden states (h[0] is h_init)
        T* z_cache,            // [T, B, dim] z values for backward (training only)
        T* h_tilde_cache,      // [T, B, dim] h_tilde for backward (training only)
        T* r_h_cache,          // [T, B, dim] r*h_prev for backward (training only)
        T* workspace);         // [T*B*3*dim + 4*B*dim]

    // Workspace size calculation
    static int64_t WorkspaceSize(int steps, int batch_size, int dim) {
        // Wx_zr_all: T*B*2*dim, Wx_h_all: T*B*dim, Uh_zr: B*2*dim, Uh_h: B*dim
        // r_h_prev: B*dim, z_tmp: B*dim
        return steps * batch_size * 2 * dim +  // Wx_zr_all
               steps * batch_size * dim +       // Wx_h_all
               batch_size * 2 * dim +           // Uh_zr
               batch_size * dim +               // Uh_h
               batch_size * dim +               // r_h_prev
               batch_size * dim;                // z_tmp
    }

private:
    bool training_;
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct GRUBackward {
    GRUBackward(
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_zr,
        const T* W_h,
        const T* U_zr,
        const T* U_h,
        const T* b_zr,
        const T* b_h,
        const T* x,
        const T* h,
        const T* z_cache,
        const T* h_tilde_cache,
        const T* r_h_cache,
        const T* d_h_all,      // [T, B, dim] gradient on all hidden states
        T* dx,
        T* dW_zr,
        T* dW_h,
        T* dU_zr,
        T* dU_h,
        T* db_zr,
        T* db_h,
        T* workspace);

    // Workspace size calculation
    static int64_t WorkspaceSize(int steps, int batch_size, int dim) {
        // d_z_pre, d_h_tilde_pre, d_h_prev, d_r_pre, d_r_h_prev, dh_next: 6*B*dim
        // db_zr_float: 2*dim, db_h_float: dim (in floats)
        // Wx_zr_all: T*B*2*dim, Wx_h_all: T*B*dim, Uh_zr: B*2*dim
        int64_t float_bytes = 3 * dim * sizeof(float);
        int64_t float_in_T = (float_bytes + sizeof(T) - 1) / sizeof(T);
        return 6 * batch_size * dim +          // Temp buffers
               float_in_T +                     // Float bias accumulators
               steps * batch_size * 2 * dim +  // Wx_zr_all
               steps * batch_size * dim +       // Wx_h_all
               batch_size * 2 * dim;            // Uh_zr
    }

private:
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// LSTM: Long Short-Term Memory - BF16 Optimized
// Custom implementation to avoid cuDNN's bfloat16 performance regression.
//
// LSTM Equations:
// f_t = sigmoid(W_f @ x_t + U_f @ h_{t-1} + b_f)  # forget gate
// i_t = sigmoid(W_i @ x_t + U_i @ h_{t-1} + b_i)  # input gate
// o_t = sigmoid(W_o @ x_t + U_o @ h_{t-1} + b_o)  # output gate
// c_tilde = tanh(W_c @ x_t + U_c @ h_{t-1} + b_c) # candidate cell
// c_t = f_t * c_{t-1} + i_t * c_tilde             # cell state
// h_t = o_t * tanh(c_t)                           # hidden state
// =============================================================================

template<typename T>
struct LSTMForward {
    LSTMForward(
        bool training,
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_fio,        // [3*dim, dim] - W_f, W_i, W_o stacked
        const T* W_c,          // [dim, dim]
        const T* U_fio,        // [3*dim, dim] - U_f, U_i, U_o stacked
        const T* U_c,          // [dim, dim]
        const T* b_fio,        // [3*dim] - b_f, b_i, b_o
        const T* b_c,          // [dim]
        const T* x,            // [T, B, dim]
        T* h,                  // [T+1, B, dim] hidden states (h[0] is h_init)
        T* c,                  // [T+1, B, dim] cell states (c[0] is c_init)
        T* f_cache,            // [T, B, dim] forget gate cache (training only)
        T* i_cache,            // [T, B, dim] input gate cache (training only)
        T* o_cache,            // [T, B, dim] output gate cache (training only)
        T* c_tilde_cache,      // [T, B, dim] candidate cache (training only)
        T* tanh_c_cache,       // [T, B, dim] tanh(c) cache (training only)
        T* workspace);         // See WorkspaceSize

    // Workspace size calculation
    static int64_t WorkspaceSize(int steps, int batch_size, int dim) {
        // Wx_fio_all: T*B*3*dim, Wx_c_all: T*B*dim, Uh_fio: B*3*dim, Uh_c: B*dim
        return steps * batch_size * 3 * dim +  // Wx_fio_all
               steps * batch_size * dim +       // Wx_c_all
               batch_size * 3 * dim +           // Uh_fio
               batch_size * dim;                // Uh_c
    }

private:
    bool training_;
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct LSTMBackward {
    LSTMBackward(
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_fio,
        const T* W_c,
        const T* U_fio,
        const T* U_c,
        const T* x,
        const T* h,
        const T* c,
        const T* f_cache,
        const T* i_cache,
        const T* o_cache,
        const T* c_tilde_cache,
        const T* tanh_c_cache,
        const T* dh_all,       // [T, B, dim] gradients on ALL hidden states (can be null)
        const T* d_c_final,    // [B, dim] gradient on final cell state (can be null)
        T* dx,
        T* dW_fio,
        T* dW_c,
        T* dU_fio,
        T* dU_c,
        T* db_fio,
        T* db_c,
        T* workspace);

    // Workspace size calculation
    static int64_t WorkspaceSize(int steps, int batch_size, int dim) {
        // d_f_pre, d_i_pre, d_o_pre, d_c_tilde_pre, dc_prev, dh_next, dc_next: 7*B*dim
        // db_fio_float: 3*dim, db_c_float: dim (in floats)
        int64_t float_bytes = 4 * dim * sizeof(float);
        int64_t float_in_T = (float_bytes + sizeof(T) - 1) / sizeof(T);
        return 7 * batch_size * dim + float_in_T;
    }

private:
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E63m: Matrix State Nonlinear Delta - Maximum Expressivity
// Matrix state S ∈ ℝ^(N×D) with NONLINEAR retrieval and update.
// This is UTM-class (nonlinear h-dependence prevents parallel scan).
//
// Forward:
//     k_t = x @ W_k.T                                # [B, D] key
//     q_t = x @ W_q.T                                # [B, D] query
//     Sk = bmm(S, k_t.unsqueeze(-1)).squeeze(-1)     # [B, N]
//     retrieved = tanh(Sk)                           # [B, N] nonlinear!
//     v_t = tanh(retrieved @ W_r.T + x @ W_x.T + b)  # [B, N]
//     alpha_t = sigmoid(x @ W_alpha.T + b_alpha)     # [B, N]
//     v_outer_k = bmm(v_t.unsqueeze(-1), k_t.unsqueeze(1))  # [B, N, D]
//     S_new = alpha_t * S + (1 - alpha_t) * v_outer_k
//     output = tanh(S_new @ q_t)                     # [B, N]
// =============================================================================

template<typename T>
struct E63mMatrixNonlinearForward {
    E63mMatrixNonlinearForward(
        bool training,
        int batch_size,
        int n_slots,     // N - number of slots in matrix state
        int dim,         // D - dimension
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_k,           // [D, D] key projection
        const T* W_q,           // [D, D] query projection
        const T* W_x,           // [N, D] input-to-value projection
        const T* W_r,           // [N, N] retrieval transformation
        const T* b,             // [N] value bias
        const T* W_alpha,       // [N, D] gate projection
        const T* b_alpha,       // [N] gate bias
        const T* x,             // [T, B, D] input
        T* S,                   // [T+1, B, N, D] matrix states
        T* output,              // [T, B, N] output
        T* k_cache,             // [T, B, D] for backward
        T* q_cache,             // [T, B, D] for backward
        T* Wx_cache,            // [T, B, N] for backward
        T* alpha_x_cache,       // [T, B, N] for backward
        T* Sk_cache,            // [T, B, N] for backward (pre-tanh)
        T* retrieved_cache,     // [T, B, N] for backward
        T* Wr_ret_cache,        // [T, B, N] for backward
        T* v_cache,             // [T, B, N] for backward
        T* alpha_cache,         // [T, B, N] for backward
        T* workspace);          // See WorkspaceSize

    static int64_t WorkspaceSize(int steps, int batch_size, int n_slots, int dim) {
        int64_t BD = batch_size * dim;
        int64_t BN = batch_size * n_slots;
        // k_all, q_all: 2*T*BD
        // Wx_all, alpha_x_all: 2*T*BN
        // Sk_tmp, retrieved_tmp, Wr_ret_tmp, v_tmp, alpha_tmp: 5*BN
        return 2 * steps * BD + 2 * steps * BN + 5 * BN;
    }

private:
    bool training_;
    int batch_size_;
    int n_slots_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E63mMatrixNonlinearBackward {
    E63mMatrixNonlinearBackward(
        int batch_size,
        int n_slots,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_k,
        const T* W_q,
        const T* W_x,
        const T* W_r,
        const T* b,
        const T* W_alpha,
        const T* b_alpha,
        const T* x,
        const T* S,
        const T* output,
        const T* k_cache,
        const T* q_cache,
        const T* Wx_cache,
        const T* alpha_x_cache,
        const T* Sk_cache,
        const T* retrieved_cache,
        const T* Wr_ret_cache,
        const T* v_cache,
        const T* alpha_cache,
        const T* d_output,
        T* dx,
        T* dW_k,
        T* dW_q,
        T* dW_x,
        T* dW_r,
        T* db,
        T* dW_alpha,
        T* db_alpha,
        T* workspace);

    static int64_t WorkspaceSize(int steps, int batch_size, int n_slots, int dim) {
        int64_t BD = batch_size * dim;
        int64_t BN = batch_size * n_slots;
        int64_t BND = batch_size * n_slots * dim;
        // d_S, d_S_tmp: 2*BND (need two buffers for race-free state gradient propagation)
        // d_k_all, d_q_all: 2*T*BD
        // d_Wx_all, d_alpha_x_all: 2*T*BN
        // d_retrieved, d_Wr_ret: 2*BN
        // Float workspace: d_v_f, d_alpha_f (2*BN), d_k_f, d_q_f (2*BD), db_f, db_alpha_f (2*N)
        int64_t float_elems = 2 * BN + 2 * BD + 2 * n_slots;
        int64_t float_bytes = float_elems * sizeof(float);
        int64_t float_in_T = (float_bytes + sizeof(T) - 1) / sizeof(T);
        return 2 * BND + 2 * steps * BD + 2 * steps * BN + 2 * BN + float_in_T;
    }

private:
    int batch_size_;
    int n_slots_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E65: Diagonal H-Dependence - Learnable Per-Dimension Scaling
// O(d) h-dependence cost instead of O(d^2)
//
// alpha_t = sigmoid(W_alpha @ x_t + b_alpha)        # Decay gate (x-only)
// v_t = tanh(d_h * h_{t-1} + W_x @ x_t + b)         # d_h is [dim] diagonal vector
// h_t = alpha_t * h_{t-1} + (1 - alpha_t) * v_t    # Gated update
// output = h * silu(h)                              # Self-gating
//
// UTM-class: h is inside the tanh nonlinearity
// =============================================================================

template<typename T>
struct E65DiagonalHForward {
    E65DiagonalHForward(
        bool training,
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_alpha,     // [dim, dim] decay gate weight
        const T* b_alpha,     // [dim] decay gate bias
        const T* d_h,         // [dim] diagonal h-scaling vector
        const T* W_x,         // [dim, dim] input-to-value weight
        const T* b,           // [dim] value bias
        const T* x,           // [T, B, dim] pre-activated input
        T* h,                 // [T+1, B, dim] hidden states
        T* output,            // [T, B, dim] output
        T* v_pre_cache,       // [T, B, dim] pre-tanh cache for backward
        T* alpha_cache,       // [T, B, dim] alpha cache for backward
        T* workspace);        // [2*T*B*dim] for alpha_logits, Wx

    // Workspace size calculation
    static int64_t WorkspaceSize(int steps, int batch_size, int dim) {
        return 2 * steps * batch_size * dim;  // alpha_logits_all + Wx_all
    }

private:
    bool training_;
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E65DiagonalHBackward {
    E65DiagonalHBackward(
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_alpha,
        const T* d_h,
        const T* W_x,
        const T* x,
        const T* h,
        const T* v_pre_cache,
        const T* alpha_cache,
        const T* d_output,
        T* dx,
        T* dW_alpha,
        T* db_alpha,
        T* d_d_h,             // [dim] gradient for diagonal vector
        T* dW_x,
        T* db,
        T* workspace);        // See WorkspaceSize

    // Workspace size calculation
    static int64_t WorkspaceSize(int steps, int batch_size, int dim) {
        // dh, dh_recurrent: 2*B*dim
        // d_alpha_logit_all, dWx_all: 2*T*B*dim
        // db_alpha_float, db_float, d_d_h_float: 3*dim floats
        int64_t float_bytes = 3 * dim * sizeof(float);
        int64_t float_in_T = (float_bytes + sizeof(T) - 1) / sizeof(T);
        return 2 * batch_size * dim +
               2 * steps * batch_size * dim +
               float_in_T;
    }

private:
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E66: Low-Rank H-Dependence (UTM-Class with Cross-Dimension Mixing)
// alpha_t = sigmoid(W_alpha @ x_t + b_alpha)           # Retain gate (x-only)
// h_compressed = V @ h_{t-1}                           # Compress h to rank
// h_transformed = U @ h_compressed                     # Expand back to dim
// v_t = tanh(h_transformed + W_x @ x_t + b)           # NONLINEAR value
// h_t = alpha_t * h_{t-1} + (1 - alpha_t) * v_t       # Gated mixing
// output = h * silu(h)                                 # Self-gating
//
// Key: O(d*rank) cost per timestep instead of O(d^2) for full W_h @ h
// UTM-class: h is inside tanh, providing Turing completeness
// =============================================================================

template<typename T>
struct E66LowRankHForward {
    E66LowRankHForward(
        bool training,
        int batch_size,
        int dim,
        int rank,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_alpha,     // [dim, dim] retain gate weight
        const T* b_alpha,     // [dim] retain gate bias
        const T* U,           // [dim, rank] expand matrix
        const T* V,           // [rank, dim] compress matrix
        const T* W_x,         // [dim, dim] input weight
        const T* b,           // [dim] bias
        const T* x,           // [T, B, dim] input
        T* h,                 // [T+1, B, dim] hidden states (h[0] = h0)
        T* output,            // [T, B, dim] output
        T* v_pre_cache,       // [T, B, dim] pre-tanh cache (training)
        T* alpha_cache,       // [T, B, dim] alpha cache (training)
        T* Vh_cache,          // [T, B, rank] V @ h cache (training)
        T* workspace);        // See WorkspaceSize

    // Workspace size calculation
    static int64_t WorkspaceSize(int steps, int batch_size, int dim, int rank) {
        // alpha_x_all: T*B*dim
        // Wx_all: T*B*dim
        // tmp_Vh: B*rank
        // tmp_Uh: B*dim
        return 2 * steps * batch_size * dim +
               batch_size * rank +
               batch_size * dim;
    }

private:
    bool training_;
    int batch_size_;
    int dim_;
    int rank_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E66LowRankHBackward {
    E66LowRankHBackward(
        int batch_size,
        int dim,
        int rank,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_alpha,
        const T* U,
        const T* V,
        const T* W_x,
        const T* x,
        const T* h,
        const T* v_pre_cache,
        const T* alpha_cache,
        const T* Vh_cache,      // [T, B, rank] from forward
        const T* d_output,
        T* dx,
        T* dW_alpha,
        T* db_alpha,
        T* dU,                  // [dim, rank]
        T* dV,                  // [rank, dim]
        T* dW_x,
        T* db,
        T* workspace);          // See WorkspaceSize

    // Workspace size calculation
    static int64_t WorkspaceSize(int steps, int batch_size, int dim, int rank) {
        // dh, dh_recurrent, dh_prev: 3*B*dim
        // dv_pre_all: T*B*dim
        // dalpha_x_all: T*B*dim
        // dUh_all: T*B*dim
        // db_float, db_alpha_float: 2*dim floats
        // alpha_x_all: T*B*dim
        // tmp_dVh: B*rank
        int64_t float_bytes = 2 * dim * sizeof(float);
        int64_t float_in_T = (float_bytes + sizeof(T) - 1) / sizeof(T);
        return 3 * batch_size * dim +
               4 * steps * batch_size * dim +
               float_in_T +
               batch_size * rank;
    }

private:
    int batch_size_;
    int dim_;
    int rank_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E68: Self-Gating Elman - Multiplicative H-Dependence
// alpha_t = sigmoid(x @ W_alpha.T + b_alpha)        # Retain gate (input-dependent)
// g_t = sigmoid(d_g * h_{t-1} + b_g)               # SELF-GATING: h gates the value!
// v_raw_t = tanh(x @ W_x.T + b_v)                  # Raw new value
// v_t = v_raw_t * g_t                              # Gated value
// h_t = alpha_t * h_{t-1} + (1 - alpha_t) * v_t    # Gated update
// output = h * silu(h)                             # Self-gating output
//
// Key: O(d) cost per timestep, UTM-class (h inside sigmoid)
// =============================================================================

template<typename T>
struct E68SelfGatingForward {
    E68SelfGatingForward(
        bool training,
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_alpha,       // [dim, dim] retain gate weight
        const T* b_alpha,       // [dim] retain gate bias
        const T* W_x,           // [dim, dim] value weight
        const T* b_v,           // [dim] value bias
        const T* d_g,           // [dim] diagonal gating weights (self-gating)
        const T* b_g,           // [dim] gating bias
        const T* x,             // [T, B, dim] pre-activated input
        T* h,                   // [T+1, B, dim] hidden states
        T* output,              // [T, B, dim] output
        T* alpha_cache,         // [T, B, dim] stores alpha values for backward
        T* g_cache,             // [T, B, dim] stores g values for backward
        T* v_raw_tanh_cache,    // [T, B, dim] stores tanh(v_raw+b_v) for backward
        T* workspace);          // [2*T*B*dim] for alpha_logits, v_raw

    static int64_t WorkspaceSize(int steps, int batch_size, int dim) {
        return 2 * steps * batch_size * dim;
    }

private:
    bool training_;
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E68SelfGatingBackward {
    E68SelfGatingBackward(
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_alpha,
        const T* W_x,
        const T* d_g,
        const T* x,
        const T* h,
        const T* alpha_cache,
        const T* g_cache,
        const T* v_raw_tanh_cache,
        const T* d_output,
        T* dx,
        T* dW_alpha,
        T* db_alpha,
        T* dW_x,
        T* db_v,
        T* dd_g,                // [dim] gradient for diagonal gating weights
        T* db_g,                // [dim] gradient for gating bias
        T* workspace);          // [2*BD + 2*T*BD + 4*dim*sizeof(float)/sizeof(T)]

    static int64_t WorkspaceSize(int steps, int batch_size, int dim) {
        int64_t BD = batch_size * dim;
        // dh, dh_recurrent: 2*BD
        // d_alpha_logit_all, dv_raw_all: 2*T*BD
        // db_alpha_float, dd_g_float, db_g_float, db_v_float: 4*dim floats
        int64_t float_bytes = 4 * dim * sizeof(float);
        int64_t float_in_T = (float_bytes + sizeof(T) - 1) / sizeof(T);
        return 2 * BD + 2 * steps * BD + float_in_T;
    }

private:
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E70: Matrix Linear Elman - Square Matrix State with Linear Update
// Forward per timestep:
//     k_t = x @ W_k.T                      # [B, n_state]
//     v_t = x @ W_v.T                      # [B, n_state]
//     q_t = x @ W_q.T                      # [B, n_state]
//
//     S = decay * S + outer(v, k)          # [B, n_state, n_state]
//     S = tanh(S)                          # Nonlinear state update
//
//     out = S @ q                          # [B, n_state]
//     out = out * silu(out)                # Self-gating output
//
// Key: Square matrix state S enables rich associative storage
// =============================================================================

template<typename T>
struct E70MatrixLinearForward {
    E70MatrixLinearForward(
        bool training,
        int batch_size,
        int dim,
        int n_state,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        float decay,                // Clamped to [0, 0.999]
        const T* W_k,               // [n_state, dim] key projection
        const T* W_v,               // [n_state, dim] value projection
        const T* W_q,               // [n_state, dim] query projection
        const T* x,                 // [T, B, dim] input
        T* S,                       // [T+1, B, n_state, n_state] state matrices
        T* output,                  // [T, B, n_state] output
        T* k_cache,                 // [T, B, n_state] for backward
        T* v_cache,                 // [T, B, n_state] for backward
        T* q_cache,                 // [T, B, n_state] for backward
        T* Sq_cache,                // [T, B, n_state] for backward (S @ q result)
        T* workspace);              // See WorkspaceSize

    static int64_t WorkspaceSize(int steps, int batch_size, int n_state) {
        // k_all, v_all, q_all: 3*T*B*n_state
        // Sq_tmp: B*n_state
        return 3 * steps * batch_size * n_state + batch_size * n_state;
    }

private:
    bool training_;
    int batch_size_;
    int dim_;
    int n_state_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E70MatrixLinearBackward {
    E70MatrixLinearBackward(
        int batch_size,
        int dim,
        int n_state,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
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
        T* d_decay,                 // [1] gradient for decay parameter
        T* workspace);              // See WorkspaceSize

    static int64_t WorkspaceSize(int steps, int batch_size, int n_state) {
        int64_t BN = batch_size * n_state;
        int64_t BNN = batch_size * n_state * n_state;
        // d_S, d_S_prev: 2*B*N*N
        // d_Sq: B*N
        // d_k_all, d_v_all, d_q_all: 3*T*B*N
        // d_k_f, d_v_f, d_q_f: 3*B*N floats
        // d_decay_f: 1 float
        int64_t float_bytes = (3 * BN + 1) * sizeof(float);
        int64_t float_in_T = (float_bytes + sizeof(T) - 1) / sizeof(T);
        return 2 * BNN + BN + 3 * steps * BN + float_in_T;
    }

private:
    int batch_size_;
    int dim_;
    int n_state_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E71: Matrix Gated Elman - E67-style state-dependent gating with matrix state
//
// Architecture:
//     k_t = W_k @ x_t                              # [B, n_state] key
//     v_t = W_v @ x_t                              # [B, n_state] value
//     q_t = W_q @ x_t                              # [B, n_state] query
//     alpha_x_t = W_alpha @ x_t                    # [B, n_state]
//
//     # S-dependent gate (E67 insight: state affects gating decision)
//     retrieved = S @ k_t                          # [B, n_state] (matrix-vector product)
//     alpha = sigmoid(alpha_x + d_alpha * retrieved + b_alpha)
//
//     # Gated update
//     S_new = alpha.unsqueeze(-1) * S + (1 - alpha.unsqueeze(-1)) * outer(v, k)
//
//     # Self-gating output
//     out = S_new @ q
//     output = out * silu(out)
//
// State S is [B, n_state, n_state] - square matrix
// Cannot parallelize due to S-dependence in gating (UTM-class)
// =============================================================================

template<typename T>
struct E71MatrixGatedForward {
    E71MatrixGatedForward(
        bool training,
        int batch_size,
        int n_state,    // N - square matrix dimension
        int dim,        // D - input dimension
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_k,           // [n_state, dim] key projection
        const T* W_v,           // [n_state, dim] value projection
        const T* W_q,           // [n_state, dim] query projection
        const T* W_alpha,       // [n_state, dim] gate projection
        const T* d_alpha,       // [n_state] gate h-scaling
        const T* b_alpha,       // [n_state] gate bias
        const T* x,             // [T, B, dim] input
        T* S,                   // [T+1, B, n_state, n_state] state matrices
        T* output,              // [T, B, n_state] output
        T* k_cache,             // [T, B, n_state] for backward
        T* v_cache,             // [T, B, n_state] for backward
        T* q_cache,             // [T, B, n_state] for backward
        T* alpha_x_cache,       // [T, B, n_state] for backward
        T* retrieved_cache,     // [T, B, n_state] for backward
        T* alpha_cache,         // [T, B, n_state] for backward
        T* workspace);          // See WorkspaceSize

    static int64_t WorkspaceSize(int steps, int batch_size, int n_state) {
        int64_t BN = batch_size * n_state;
        // k_all, v_all, q_all, alpha_x_all: 4*T*BN
        // retrieved_tmp, alpha_tmp: 2*BN
        return 4 * steps * BN + 2 * BN;
    }

private:
    bool training_;
    int batch_size_;
    int n_state_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E71MatrixGatedBackward {
    E71MatrixGatedBackward(
        int batch_size,
        int n_state,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_k,
        const T* W_v,
        const T* W_q,
        const T* W_alpha,
        const T* d_alpha,
        const T* b_alpha,
        const T* x,
        const T* S,
        const T* k_cache,
        const T* v_cache,
        const T* q_cache,
        const T* alpha_x_cache,
        const T* retrieved_cache,
        const T* alpha_cache,
        const T* d_output,
        T* dx,
        T* dW_k,
        T* dW_v,
        T* dW_q,
        T* dW_alpha,
        T* dd_alpha,            // [n_state] gradient for gate h-scaling
        T* db_alpha,            // [n_state] gradient for gate bias
        T* workspace);

    static int64_t WorkspaceSize(int steps, int batch_size, int n_state) {
        int64_t BN = batch_size * n_state;
        int64_t BNN = batch_size * n_state * n_state;
        // d_S, d_S_tmp: 2*BNN
        // d_k_all, d_v_all, d_q_all, d_alpha_x_all: 4*T*BN
        // d_retrieved: BN
        // Float workspace: d_alpha_f, d_k_f, d_v_f, d_q_f (4*BN), dd_alpha_f, db_alpha_f (2*n_state)
        int64_t float_elems = 4 * BN + 2 * n_state;
        int64_t float_bytes = float_elems * sizeof(float);
        int64_t float_in_T = (float_bytes + sizeof(T) - 1) / sizeof(T);
        return 2 * BNN + 4 * steps * BN + BN + float_in_T;
    }

private:
    int batch_size_;
    int n_state_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E73: Matrix Nonlinear Elman - E1-style with S inside tanh
//
// Architecture (column variant):
//     k_t = W_k @ x_t                              # [B, n] key
//     v_t = W_v @ x_t                              # [B, n] value
//     q_t = W_q @ x_t                              # [B, n] query
//     z_t = sigmoid(W_z @ x_t + b_z)               # [B, n] modulation gate
//
//     S = tanh(S * z.unsqueeze(1) + outer(v, k))   # [B, n, n] (column modulation)
//
//     out = S @ q                                   # [B, n]
//     out = out * silu(out)                        # Self-gating output
//
// State S is [B, n, n] square matrix.
//
// Variants:
// - 0 (column): S[i,j] *= z[j] (scale each column by z[j])
// - 1 (row):    S[i,j] *= z[i] (scale each row by z[i])
// - 2 (full):   S[i,j] *= z[i] * z[j] (outer product scaling)
// =============================================================================

template<typename T>
struct E73MatrixNonlinearForward {
    E73MatrixNonlinearForward(
        bool training,
        int batch_size,
        int n_state,     // n - state matrix dimension (n x n)
        int dim,         // input dimension
        int variant,     // 0=column, 1=row, 2=full
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_k,           // [n, dim] key projection
        const T* W_v,           // [n, dim] value projection
        const T* W_q,           // [n, dim] query projection
        const T* W_z,           // [n, dim] modulation gate projection
        const T* b_z,           // [n] modulation gate bias
        const T* x,             // [T, B, dim] input
        T* S,                   // [T+1, B, n, n] state matrices
        T* output,              // [T, B, n] output
        T* k_cache,             // [T, B, n] for backward
        T* v_cache,             // [T, B, n] for backward
        T* q_cache,             // [T, B, n] for backward
        T* z_cache,             // [T, B, n] for backward (post-sigmoid)
        T* pre_tanh_cache,      // [T, B, n, n] for backward
        T* Sq_cache,            // [T, B, n] for backward (pre-self-gate)
        T* workspace);          // See WorkspaceSize

    static int64_t WorkspaceSize(int steps, int batch_size, int n_state) {
        int64_t BN = batch_size * n_state;
        // k_all, v_all, q_all, z_logit_all, z_all: 5*T*BN
        return 5 * steps * BN;
    }

private:
    bool training_;
    int batch_size_;
    int n_state_;
    int dim_;
    int variant_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E73MatrixNonlinearBackward {
    E73MatrixNonlinearBackward(
        int batch_size,
        int n_state,
        int dim,
        int variant,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_k,
        const T* W_v,
        const T* W_q,
        const T* W_z,
        const T* x,
        const T* S,
        const T* k_cache,
        const T* v_cache,
        const T* q_cache,
        const T* z_cache,
        const T* pre_tanh_cache,
        const T* Sq_cache,
        const T* d_output,
        T* dx,
        T* dW_k,
        T* dW_v,
        T* dW_q,
        T* dW_z,
        T* db_z,
        T* workspace);          // See WorkspaceSize

    static int64_t WorkspaceSize(int steps, int batch_size, int n_state) {
        int64_t BN = batch_size * n_state;
        int64_t BNN = batch_size * n_state * n_state;
        // d_S, d_S_tmp: 2*BNN
        // d_Sq: BN
        // d_k_all, d_v_all, d_q_all, d_z_logit_all: 4*T*BN
        // Float workspace: d_k_f, d_v_f, d_q_f, d_z_f (4*BN), db_z_f (n_state)
        int64_t float_bytes = (4 * BN + n_state) * sizeof(float);
        int64_t float_in_T = (float_bytes + sizeof(T) - 1) / sizeof(T);
        return 2 * BNN + BN + 4 * steps * BN + float_in_T;
    }

private:
    int batch_size_;
    int n_state_;
    int dim_;
    int variant_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// E72: Matrix SelfGate Elman - Memory content controls writing
// k = W_k @ x, v = W_v @ x, q = W_q @ x
// alpha = sigmoid(W_alpha @ x + b_alpha)     # Retain gate
// retrieved = S @ k                          # Query memory
// g = sigmoid(d_g * retrieved + b_g)         # Gate from memory content
// v_gated = v * g                            # Memory controls writing
// S = alpha * S + (1 - alpha) * outer(v_gated, k)
// out = S @ q
// out = out * silu(out)                      # Self-gating output
//
// Variants:
// - standard (inverse_gate=false): g = sigmoid(d_g * retrieved + b_g) - content enables writing
// - inverse (inverse_gate=true):   g = sigmoid(-d_g * |retrieved| + b_g) - content resists writing
// =============================================================================

template<typename T>
struct E72MatrixSelfGateForward {
    E72MatrixSelfGateForward(
        bool training,
        int batch_size,
        int dim,
        int n_state,
        bool inverse_gate,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_k,               // [n_state, dim] key projection
        const T* W_v,               // [n_state, dim] value projection
        const T* W_q,               // [n_state, dim] query projection
        const T* W_alpha,           // [n_state, dim] retain gate weight
        const T* b_alpha,           // [n_state] retain gate bias
        const T* d_g,               // [n_state] gating weight (self-gating)
        const T* b_g,               // [n_state] gating bias
        const T* x,                 // [T, B, dim] input
        T* S,                       // [T+1, B, n_state, n_state] state matrices
        T* output,                  // [T, B, n_state] output
        T* k_cache,                 // [T, B, n_state] for backward
        T* v_cache,                 // [T, B, n_state] for backward
        T* q_cache,                 // [T, B, n_state] for backward
        T* alpha_cache,             // [T, B, n_state] for backward
        T* retrieved_cache,         // [T, B, n_state] for backward
        T* g_cache,                 // [T, B, n_state] for backward
        T* workspace);              // See WorkspaceSize

    static int64_t WorkspaceSize(int steps, int batch_size, int n_state) {
        // k_all, v_all, q_all, alpha_x_all: 4*T*B*n_state
        // v_gated_tmp, alpha_tmp: 2*B*n_state
        return 4 * steps * batch_size * n_state + 2 * batch_size * n_state;
    }

private:
    bool training_;
    int batch_size_;
    int dim_;
    int n_state_;
    bool inverse_gate_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct E72MatrixSelfGateBackward {
    E72MatrixSelfGateBackward(
        int batch_size,
        int dim,
        int n_state,
        bool inverse_gate,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_k,
        const T* W_v,
        const T* W_q,
        const T* W_alpha,
        const T* d_g,
        const T* x,
        const T* S,
        const T* k_cache,
        const T* v_cache,
        const T* q_cache,
        const T* alpha_cache,
        const T* retrieved_cache,
        const T* g_cache,
        const T* d_output,
        T* dx,
        T* dW_k,
        T* dW_v,
        T* dW_q,
        T* dW_alpha,
        T* db_alpha,
        T* dd_g,
        T* db_g,
        T* workspace);              // See WorkspaceSize

    static int64_t WorkspaceSize(int steps, int batch_size, int n_state) {
        int64_t BN = batch_size * n_state;
        int64_t BNN = batch_size * n_state * n_state;
        // d_S, d_S_tmp: 2*B*N*N
        // d_k_all, d_v_all, d_q_all, d_alpha_x_all: 4*T*B*N
        // v_gated_tmp: B*N (recomputed in backward)
        // Float accumulators: 7*B*N + 3*N floats
        int64_t float_bytes = (7 * BN + 3 * n_state) * sizeof(float);
        int64_t float_in_T = (float_bytes + sizeof(T) - 1) / sizeof(T);
        return 2 * BNN + 4 * steps * BN + BN + float_in_T;
    }

private:
    int batch_size_;
    int dim_;
    int n_state_;
    bool inverse_gate_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty

#endif  // HASTY_ELMAN_LADDER_H
