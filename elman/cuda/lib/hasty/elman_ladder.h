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
        T* workspace);        // Complex workspace for FFT operations

private:
    bool training_;
    int batch_size_;
    int dim_;
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
        T* workspace);      // Complex workspace for FFT operations

private:
    int batch_size_;
    int dim_;
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
    int m_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty

#endif  // HASTY_ELMAN_LADDER_H
