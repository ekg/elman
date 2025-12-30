// Copyright 2024 Erik Garrison. Apache 2.0 License.
//
// Elman Ablation Ladder - Log-Space Experiment Series
//
// This header defines CUDA kernels for the systematic ablation from
// stock Elman to full log-space Triple R, following:
//
// Level 0: Stock Elman - Basic tanh recurrence
// Level 1: Gated Elman - + Input-dependent delta gate
// Level 2: Selective Elman - + compete×silu output
// Level 3: Diagonal Selective - Diagonal W_h (like Mamba2's diagonal A)
// Level 4: Log-Storage Diagonal - + signed log storage for hidden state
// Level 5: Log-Compute Full - Full R via logsumexp decomposition
// Level 6: Triple R - + R_delta modulation
//
// Key insight: Test each modification incrementally to find what matches Mamba2.

#ifndef HASTY_ELMAN_LADDER_H
#define HASTY_ELMAN_LADDER_H

#include <cuda.h>
#include <cublas_v2.h>

namespace hasty {
namespace v0 {
namespace elman_ladder {

// =============================================================================
// Level 0: Stock Elman
// h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b)
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
        const T* b,         // [dim]
        const T* x,         // [T, B, dim]
        T* h,               // [T+1, B, dim] output
        T* v);              // [T, B, dim] pre-activation for backward

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

    void Run(
        int steps,
        const T* W_x,
        const T* W_h,
        const T* x,
        const T* h,
        const T* v,
        const T* dh_out,    // [T, B, dim] gradient from above
        T* dx,              // [T, B, dim]
        T* dW_x,            // [dim, dim]
        T* dW_h,            // [dim, dim]
        T* db);             // [dim]

private:
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// Level 1: Gated Elman
// delta = sigmoid(W_delta @ x_t + b_delta)
// h_t = (1 - delta) * h_{t-1} + delta * tanh(W_x @ x_t + W_h @ h_{t-1} + b)
// =============================================================================

template<typename T>
struct GatedElmanForward {
    GatedElmanForward(
        bool training,
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,       // [dim, dim]
        const T* W_h,       // [dim, dim]
        const T* W_delta,   // [dim, dim]
        const T* b,         // [dim]
        const T* b_delta,   // [dim]
        const T* x,         // [T, B, dim]
        T* h,               // [T+1, B, dim] output
        T* v,               // [T, B, dim] pre-activation
        T* delta_cache);    // [T, B, dim] cached delta for backward

private:
    bool training_;
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct GatedElmanBackward {
    GatedElmanBackward(
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,
        const T* W_h,
        const T* W_delta,
        const T* x,
        const T* h,
        const T* v,
        const T* delta_cache,
        const T* dh_out,
        T* dx,
        T* dW_x,
        T* dW_h,
        T* dW_delta,
        T* db,
        T* db_delta);

private:
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// Level 2: Selective Elman
// Same recurrence as Gated Elman, but with compete×silu output:
// compete = softmax(h_t, groups=n_groups)
// output = compete * silu(W_out @ h_t)
// =============================================================================

template<typename T>
struct SelectiveElmanForward {
    SelectiveElmanForward(
        bool training,
        int batch_size,
        int dim,
        int n_groups,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,       // [dim, dim]
        const T* W_h,       // [dim, dim]
        const T* W_delta,   // [dim, dim]
        const T* W_out,     // [dim, dim]
        const T* b,         // [dim]
        const T* b_delta,   // [dim]
        const T* x,         // [T, B, dim]
        T* h,               // [T+1, B, dim] internal hidden
        T* output,          // [T, B, dim] selective output
        T* v,               // [T, B, dim] pre-activation
        T* delta_cache,     // [T, B, dim] cached delta
        T* compete_cache);  // [T, B, dim] cached compete for backward

private:
    bool training_;
    int batch_size_;
    int dim_;
    int n_groups_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct SelectiveElmanBackward {
    SelectiveElmanBackward(
        int batch_size,
        int dim,
        int n_groups,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
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
        const T* d_output,   // [T, B, dim] gradient from above (on output)
        T* dx,
        T* dW_x,
        T* dW_h,
        T* dW_delta,
        T* dW_out,
        T* db,
        T* db_delta);

private:
    int batch_size_;
    int dim_;
    int n_groups_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// Level 3: Diagonal Selective Elman
// delta = sigmoid(W_delta @ x_t + b_delta)
// h_t = (1 - delta) * h_{t-1} + delta * tanh(W_x @ x_t + r_h * h_{t-1} + b)
// where r_h is a VECTOR (diagonal), not full matrix
// + compete×silu output
// =============================================================================

template<typename T>
struct DiagonalSelectiveElmanForward {
    DiagonalSelectiveElmanForward(
        bool training,
        int batch_size,
        int dim,
        int n_groups,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,       // [dim, dim]
        const T* r_h,       // [dim] DIAGONAL decay
        const T* W_delta,   // [dim, dim]
        const T* W_out,     // [dim, dim]
        const T* b,         // [dim]
        const T* b_delta,   // [dim]
        const T* x,         // [T, B, dim]
        T* h,               // [T+1, B, dim] internal hidden
        T* output,          // [T, B, dim] selective output
        T* v,               // [T, B, dim] pre-activation
        T* delta_cache,     // [T, B, dim]
        T* compete_cache);  // [T, B, dim]

private:
    bool training_;
    int batch_size_;
    int dim_;
    int n_groups_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct DiagonalSelectiveElmanBackward {
    DiagonalSelectiveElmanBackward(
        int batch_size,
        int dim,
        int n_groups,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,
        const T* r_h,
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
        T* dr_h,            // [dim] gradient for diagonal decay
        T* dW_delta,
        T* dW_out,
        T* db,
        T* db_delta);

private:
    int batch_size_;
    int dim_;
    int n_groups_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// Level 4: Log-Storage Diagonal Elman (TRUE LOG-SPACE BACKWARD)
// Same as Level 3, but hidden state stored as (log|h|, sign(h))
// h_t = (1 - delta) * h_{t-1} + delta * tanh(W_x @ x_t + r_h * h_{t-1} + b)
// Store: log_h = log(|h_t|), sign_h = sign(h_t)
//
// KEY INNOVATION: Backward pass computes gradients w.r.t. log|h| using
// softmax weights from logaddexp. This prevents gradient vanishing at depth!
// =============================================================================

template<typename T>
struct LogStorageDiagonalElmanForward {
    LogStorageDiagonalElmanForward(
        bool training,
        int batch_size,
        int dim,
        int n_groups,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,           // [dim, dim]
        const T* r_h,           // [dim] DIAGONAL decay
        const T* W_delta,       // [dim, dim]
        const T* W_out,         // [dim, dim]
        const T* b,             // [dim]
        const T* b_delta,       // [dim]
        const T* x,             // [T, B, dim]
        T* log_h,               // [T+1, B, dim] log(|h|)
        T* sign_h,              // [T+1, B, dim] sign(h) in {-1, +1}
        T* output,              // [T, B, dim] selective output
        T* v,                   // [T, B, dim] pre-activation
        T* delta_cache,         // [T, B, dim]
        T* compete_cache,       // [T, B, dim]
        T* weight1_cache,       // [T, B, dim] softmax weight for log-space backward
        T* log_term1_cache,     // [T, B, dim] log|(1-δ)*h_prev|
        T* log_term2_cache);    // [T, B, dim] log|δ*candidate|

private:
    bool training_;
    int batch_size_;
    int dim_;
    int n_groups_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct LogStorageDiagonalElmanBackward {
    LogStorageDiagonalElmanBackward(
        int batch_size,
        int dim,
        int n_groups,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,
        const T* r_h,
        const T* W_delta,
        const T* W_out,
        const T* x,
        const T* log_h,
        const T* sign_h,
        const T* v,
        const T* delta_cache,
        const T* compete_cache,
        const T* weight1_cache,     // softmax weight for log-space backward
        const T* log_term1_cache,   // log|(1-δ)*h_prev|
        const T* log_term2_cache,   // log|δ*candidate|
        const T* d_output,
        T* dx,
        T* dW_x,
        T* dr_h,
        T* dW_delta,
        T* dW_out,
        T* db,
        T* db_delta);

private:
    int batch_size_;
    int dim_;
    int n_groups_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// Level 5: Log-Compute Full Elman
// Full R matrix with logsumexp decomposition for numerical stability
// R is decomposed into R+ = max(R, 0) and R- = max(-R, 0)
// h is stored as (log|h|, sign(h))
// R @ h computed via logsumexp over positive/negative contributions
// =============================================================================

template<typename T>
struct LogComputeFullElmanForward {
    LogComputeFullElmanForward(
        bool training,
        int batch_size,
        int dim,
        int n_groups,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,        // [dim, dim]
        const T* R_h,        // [dim, dim] FULL recurrence matrix
        const T* W_delta,    // [dim, dim]
        const T* W_out,      // [dim, dim]
        const T* b,          // [dim]
        const T* b_delta,    // [dim]
        const T* x,          // [T, B, dim]
        T* log_h,            // [T+1, B, dim] log(|h|)
        T* sign_h,           // [T+1, B, dim] sign(h)
        T* output,           // [T, B, dim]
        T* v,                // [T, B, dim]
        T* delta_cache,      // [T, B, dim]
        T* compete_cache,    // [T, B, dim]
        // Workspace for decomposed R
        T* log_R_pos,        // [dim, dim] log(max(R, 0))
        T* log_R_neg);       // [dim, dim] log(max(-R, 0))

private:
    bool training_;
    int batch_size_;
    int dim_;
    int n_groups_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct LogComputeFullElmanBackward {
    LogComputeFullElmanBackward(
        int batch_size,
        int dim,
        int n_groups,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,
        const T* R_h,
        const T* W_delta,
        const T* W_out,
        const T* x,
        const T* log_h,
        const T* sign_h,
        const T* v,
        const T* delta_cache,
        const T* compete_cache,
        const T* log_R_pos,
        const T* log_R_neg,
        const T* d_output,
        T* dx,
        T* dW_x,
        T* dR_h,
        T* dW_delta,
        T* dW_out,
        T* db,
        T* db_delta);

private:
    int batch_size_;
    int dim_;
    int n_groups_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// Level 6: Triple R with Log-Space
// Full Triple R architecture with log-space numerics:
// - R_h @ h via logsumexp
// - R_x @ x (fresh input, no log-space needed)
// - R_delta @ h for gate modulation via logsumexp
// - compete×silu output
// =============================================================================

template<typename T>
struct LogSpaceTripleRForward {
    LogSpaceTripleRForward(
        bool training,
        int batch_size,
        int dim,
        int n_groups,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* R_h,        // [dim, dim] recurrence
        const T* R_x,        // [dim, dim] input mixing
        const T* R_delta,    // [dim, dim] gate modulation
        const T* W_delta,    // [dim, dim] gate input path
        const T* W_out,      // [dim, dim]
        const T* b,          // [dim]
        const T* b_delta,    // [dim]
        const T* x,          // [T, B, dim]
        T* log_h,            // [T+1, B, dim]
        T* sign_h,           // [T+1, B, dim]
        T* output,           // [T, B, dim]
        T* v,                // [T, B, dim]
        T* delta_cache,      // [T, B, dim]
        T* compete_cache);   // [T, B, dim]

private:
    bool training_;
    int batch_size_;
    int dim_;
    int n_groups_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct LogSpaceTripleRBackward {
    LogSpaceTripleRBackward(
        int batch_size,
        int dim,
        int n_groups,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* R_h,
        const T* R_x,
        const T* R_delta,
        const T* W_delta,
        const T* W_out,
        const T* x,
        const T* log_h,
        const T* sign_h,
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
        T* db_delta);

private:
    int batch_size_;
    int dim_;
    int n_groups_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// Log-Space Polynomial Levels (log_0, log_1, log_2)
// True log-space RNN with polynomial activation
// =============================================================================

// Log-Space Level 0 (log_0): Log-Space Polynomial
// α_t = 1 + softplus(W_α @ x_t + b_α)
// v = r_h * h_prev + W_x @ x + b
// log|h_cand| = α_t * log|v|
// log|h_bounded| = -softplus(-log|h_cand|)
// h_new = (1-δ) * h_prev + δ * h_bounded

template<typename T>
struct LogPolyElmanForward {
    LogPolyElmanForward(
        bool training,
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,           // [dim, dim]
        const T* log_r_h,       // [dim]
        const T* sign_r_h,      // [dim]
        const T* W_alpha,       // [dim, dim]
        const T* b_alpha,       // [dim]
        const T* W_delta,       // [dim, dim]
        const T* b,             // [dim]
        const T* b_delta,       // [dim]
        const T* log_gamma,     // [dim] RMSNorm scale in log-space
        const T* x,             // [T, B, dim]
        T* log_h,               // [T+1, B, dim]
        T* sign_h,              // [T+1, B, dim]
        T* h_linear,            // [T, B, dim] NORMALIZED output
        T* log_v_cache,
        T* sign_v_cache,
        T* alpha_cache,
        T* log_h_unbounded_cache,
        T* delta_cache,
        T* weight_rh_cache,
        T* alpha_raw_cache,
        T* log_rms_cache);      // [T, B] cache for backward

private:
    bool training_;
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct LogPolyElmanBackward {
    LogPolyElmanBackward(
        int batch_size,
        int dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,
        const T* log_r_h,
        const T* sign_r_h,
        const T* W_alpha,
        const T* W_delta,
        const T* log_gamma,         // [dim] RMSNorm scale
        const T* x,
        const T* log_h,
        const T* sign_h,
        const T* log_v_cache,
        const T* sign_v_cache,
        const T* alpha_cache,
        const T* alpha_raw_cache,
        const T* log_h_unbounded_cache,
        const T* delta_cache,
        const T* weight_rh_cache,
        const T* log_rms_cache,     // [T, B] cached from forward
        const T* d_h_linear,
        T* dx,
        T* dW_x,
        T* d_log_r_h,
        T* dW_alpha,
        T* db_alpha,
        T* dW_delta,
        T* db,
        T* db_delta,
        T* d_log_gamma);            // [dim] gradient for RMSNorm scale

private:
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// Log-Space Level 1 (log_1): Selective Log-Space Polynomial
// Same as log_0 but with compete×silu output

template<typename T>
struct LogSelectiveElmanForward {
    LogSelectiveElmanForward(
        bool training,
        int batch_size,
        int dim,
        int n_groups,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,
        const T* log_r_h,
        const T* sign_r_h,
        const T* W_alpha,
        const T* b_alpha,
        const T* W_delta,
        const T* W_out,
        const T* b,
        const T* b_delta,
        const T* log_gamma,         // [dim] RMSNorm scale in log-space
        const T* x,
        T* log_h,
        T* sign_h,
        T* output,
        T* log_v_cache,
        T* sign_v_cache,
        T* alpha_cache,
        T* log_h_unbounded_cache,
        T* delta_cache,
        T* weight_rh_cache,
        T* alpha_raw_cache,
        T* h_linear_cache,
        T* compete_cache,
        T* log_rms_cache);          // [T, B] cache for backward

private:
    bool training_;
    int batch_size_;
    int dim_;
    int n_groups_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct LogSelectiveElmanBackward {
    LogSelectiveElmanBackward(
        int batch_size,
        int dim,
        int n_groups,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,
        const T* log_r_h,
        const T* sign_r_h,
        const T* W_alpha,
        const T* W_delta,
        const T* W_out,
        const T* log_gamma,         // [dim] RMSNorm scale
        const T* x,
        const T* log_h,
        const T* sign_h,
        const T* log_v_cache,
        const T* sign_v_cache,
        const T* alpha_cache,
        const T* alpha_raw_cache,
        const T* log_h_unbounded_cache,
        const T* delta_cache,
        const T* weight_rh_cache,
        const T* h_linear_cache,
        const T* compete_cache,
        const T* log_rms_cache,     // [T, B] cached from forward
        const T* d_output,
        T* dx,
        T* dW_x,
        T* d_log_r_h,
        T* dW_alpha,
        T* db_alpha,
        T* dW_delta,
        T* dW_out,
        T* db,
        T* db_delta,
        T* d_log_gamma);            // [dim] gradient for RMSNorm scale

private:
    int batch_size_;
    int dim_;
    int n_groups_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// Log-Space Level 2 (log_2): Diagonal Selective Log-Space Polynomial
// Like log_1 but with diagonal r_h stored in log space

template<typename T>
struct LogDiagSelectiveElmanForward {
    LogDiagSelectiveElmanForward(
        bool training,
        int batch_size,
        int dim,
        int n_groups,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,
        const T* log_r_h,       // [dim] diagonal in log space
        const T* sign_r_h,      // [dim]
        const T* W_alpha,
        const T* b_alpha,
        const T* W_delta,
        const T* W_out,
        const T* b,
        const T* b_delta,
        const T* log_gamma,     // [dim] RMSNorm scale in log-space
        const T* x,
        T* log_h,
        T* sign_h,
        T* output,
        T* log_v_cache,
        T* sign_v_cache,
        T* alpha_cache,
        T* log_h_unbounded_cache,
        T* delta_cache,
        T* weight_rh_cache,
        T* alpha_raw_cache,
        T* h_linear_cache,
        T* compete_cache,
        T* log_rms_cache);      // [T, B] cache for backward

private:
    bool training_;
    int batch_size_;
    int dim_;
    int n_groups_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct LogDiagSelectiveElmanBackward {
    LogDiagSelectiveElmanBackward(
        int batch_size,
        int dim,
        int n_groups,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,
        const T* log_r_h,
        const T* sign_r_h,
        const T* W_alpha,
        const T* W_delta,
        const T* W_out,
        const T* log_gamma,     // [dim] RMSNorm scale
        const T* x,
        const T* log_h,
        const T* sign_h,
        const T* log_v_cache,
        const T* sign_v_cache,
        const T* alpha_cache,
        const T* alpha_raw_cache,
        const T* log_h_unbounded_cache,
        const T* delta_cache,
        const T* weight_rh_cache,
        const T* h_linear_cache,
        const T* compete_cache,
        const T* log_rms_cache, // [T, B] cached from forward
        const T* d_output,
        T* dx,
        T* dW_x,
        T* d_log_r_h,           // [dim] gradient for diagonal
        T* dW_alpha,
        T* db_alpha,
        T* dW_delta,
        T* dW_out,
        T* db,
        T* db_delta,
        T* d_log_gamma);        // [dim] gradient for RMSNorm scale

private:
    int batch_size_;
    int dim_;
    int n_groups_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty

#endif  // HASTY_ELMAN_LADDER_H
