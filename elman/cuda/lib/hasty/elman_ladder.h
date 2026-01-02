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
        const T* b_gate,    // [dim] h+x gate bias
        const T* x,         // [T, B, dim]
        T* h,               // [T+1, B, dim] hidden states
        T* output,          // [T, B, dim] selective output
        T* v,               // [T, B, dim] pre-activation for backward
        T* gate_cache);     // [T, B, dim] gate cache for backward

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
        const T* b_gate,    // [dim]
        const T* x,
        const T* h,
        const T* v,
        const T* gate_cache,
        const T* d_output,  // [T, B, dim] gradient from output
        T* dx,              // [T, B, dim]
        T* dW_x,            // [dim, dim]
        T* dW_h,            // [dim, dim]
        T* db,              // [dim]
        T* d_b_gate,        // [dim]
        T* workspace);      // [(T+3)*B*dim + ceil(2*dim*4/sizeof(T))]

private:
    int batch_size_;
    int dim_;
    cublasHandle_t blas_handle_;
    cudaStream_t sync_stream_;
    cudaStream_t stream_[2];
    cudaEvent_t event_;
};

// =============================================================================
// Level 1: Gated Elman with h+x selective output
// delta = sigmoid(W_delta @ x_t + b_delta)
// h_t = (1 - delta) * h_{t-1} + delta * tanh(W_x @ x_t + W_h @ h_{t-1} + b)
// output = h_t * silu(h_t + x_t + b_gate)
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
        const T* b_gate,    // [dim] h+x gate bias
        const T* x,         // [T, B, dim]
        T* h,               // [T+1, B, dim] hidden states
        T* output,          // [T, B, dim] selective output
        T* v,               // [T, B, dim] pre-activation
        T* delta_cache,     // [T, B, dim] cached delta for backward
        T* gate_cache);     // [T, B, dim] cached gate for backward

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
        const T* b_gate,
        const T* x,
        const T* h,
        const T* v,
        const T* delta_cache,
        const T* gate_cache,
        const T* d_output,
        T* dx,
        T* dW_x,
        T* dW_h,
        T* dW_delta,
        T* db,
        T* db_delta,
        T* d_b_gate,
        T* workspace);  // [6*B*dim + ceil(3*dim*sizeof(float)/sizeof(T))]

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
        T* db_delta,
        T* workspace);  // [7*B*dim + ceil(2*dim*sizeof(float)/sizeof(T))]

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
        T* db_delta,
        T* workspace);  // [7*B*dim + ceil(3*dim*sizeof(float)/sizeof(T))]

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
        T* db_delta,
        T* workspace);  // [(4*T+4)*B*dim + ceil(3*dim*sizeof(float)/sizeof(T))]

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
        T* log_R_neg,        // [dim, dim] log(max(-R, 0))
        T* workspace);       // [2*T*B*dim + 4*B*dim] for pre-computed projections

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
        T* db_delta,
        T* workspace);  // workspace for backward

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
        T* compete_cache,    // [T, B, dim]
        T* workspace);       // [2*T*B*dim + 5*B*dim] - uses cuBLAS GEMM

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
        T* db_delta,
        T* workspace);  // workspace for backward

private:
    int batch_size_;
    int dim_;
    int n_groups_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// Level 7: Log-Space Diagonal Triple R
// Like log_2 but with diagonal R_delta modulation for the delta gate
// v = r_h * h_prev + W_x @ x + b
// delta_raw = W_delta @ x + r_delta * h_prev + b_delta
// h_t = (1 - delta) * h_prev + delta * tanh(v)
// =============================================================================

template<typename T>
struct LogDiagTripleRForward {
    LogDiagTripleRForward(
        bool training,
        int batch_size,
        int dim,
        int n_groups,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,           // [dim, dim]
        const T* log_r_h,       // [dim] diagonal in log space
        const T* sign_r_h,      // [dim]
        const T* log_r_delta,   // [dim] diagonal R_delta in log space
        const T* sign_r_delta,  // [dim]
        const T* W_delta,       // [dim, dim]
        const T* W_out,         // [dim, dim]
        const T* b,             // [dim]
        const T* b_delta,       // [dim]
        const T* log_gamma,     // [dim] RMSNorm scale
        const T* x,             // [T, B, dim]
        T* log_h,               // [T+1, B, dim]
        T* sign_h,              // [T+1, B, dim]
        T* output,              // [T, B, dim]
        T* h_linear_cache,      // [T, B, dim]
        T* log_v_cache,         // [T, B, dim]
        T* sign_v_cache,        // [T, B, dim]
        T* log_h_unbounded_cache, // [T, B, dim]
        T* delta_cache,         // [T, B, dim]
        T* weight_rh_cache,     // [T, B, dim]
        T* rdelta_h_cache,      // [T, B, dim] cached r_delta * h
        T* compete_cache,       // [T, B, dim]
        T* log_rms_cache);      // [T, B]

private:
    bool training_;
    int batch_size_;
    int dim_;
    int n_groups_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct LogDiagTripleRBackward {
    LogDiagTripleRBackward(
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
        const T* log_r_delta,
        const T* sign_r_delta,
        const T* W_delta,
        const T* W_out,
        const T* log_gamma,
        const T* x,
        const T* log_h,
        const T* sign_h,
        const T* log_v_cache,
        const T* sign_v_cache,
        const T* log_h_unbounded_cache,
        const T* delta_cache,
        const T* weight_rh_cache,
        const T* rdelta_h_cache,
        const T* h_linear_cache,
        const T* compete_cache,
        const T* log_rms_cache,
        const T* d_output,
        T* dx,
        T* dW_x,
        T* d_log_r_h,           // [dim] gradient for diagonal r_h
        T* d_log_r_delta,       // [dim] gradient for diagonal r_delta
        T* dW_delta,
        T* dW_out,
        T* db,
        T* db_delta,
        T* d_log_gamma,
        T* workspace);

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

// Log-Space Level 0 (log_0): Log-Space Polynomial with Selective Output
// α_t = 1 + softplus(W_α @ x_t + b_α)
// v = r_h * h_prev + W_x @ x + b
// log|h_cand| = α_t * log|v|
// log|h_bounded| = -softplus(-log|h_cand|)
// h_new = (1-δ) * h_prev + δ * h_bounded
// output = compete(h_linear) * silu(W_out @ h_linear)  // ADDED: selective output

template<typename T>
struct LogPolyElmanForward {
    LogPolyElmanForward(
        bool training,
        int batch_size,
        int dim,
        int n_groups,           // NEW: for selective output
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
        const T* W_out,         // [dim, dim] NEW: for selective output
        const T* b,             // [dim]
        const T* b_delta,       // [dim]
        const T* log_gamma,     // [dim] RMSNorm scale in log-space
        const T* x,             // [T, B, dim]
        T* log_h,               // [T+1, B, dim]
        T* sign_h,              // [T+1, B, dim]
        T* output,              // [T, B, dim] final output after selective
        T* h_linear_cache,      // [T, B, dim] intermediate h_linear for backward
        T* log_v_cache,
        T* sign_v_cache,
        T* alpha_cache,
        T* log_h_unbounded_cache,
        T* delta_cache,
        T* weight_rh_cache,
        T* alpha_raw_cache,
        T* log_rms_cache,       // [T, B] cache for backward
        T* compete_cache);      // [T, B, dim] NEW: cached compete weights

private:
    bool training_;
    int batch_size_;
    int dim_;
    int n_groups_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct LogPolyElmanBackward {
    LogPolyElmanBackward(
        int batch_size,
        int dim,
        int n_groups,           // NEW: for selective output
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,
        const T* log_r_h,
        const T* sign_r_h,
        const T* W_alpha,
        const T* W_delta,
        const T* W_out,             // NEW: for selective output
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
        const T* h_linear_cache,    // [T, B, dim] cached h_linear
        const T* compete_cache,     // NEW: cached compete weights
        const T* log_rms_cache,     // [T, B] cached from forward
        const T* d_output,          // gradient from final output
        T* dx,
        T* dW_x,
        T* d_log_r_h,
        T* dW_alpha,
        T* db_alpha,
        T* dW_delta,
        T* dW_out,                  // NEW: gradient for W_out
        T* db,
        T* db_delta,
        T* d_log_gamma,             // [dim] gradient for RMSNorm scale
        T* workspace);              // workspace for backward

private:
    int batch_size_;
    int dim_;
    int n_groups_;
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
        T* d_log_gamma,             // [dim] gradient for RMSNorm scale
        T* workspace);              // workspace for backward

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
        T* d_log_gamma,         // [dim] gradient for RMSNorm scale
        T* workspace);          // workspace for backward

private:
    int batch_size_;
    int dim_;
    int n_groups_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// Level 4: Full Recurrence Elman (Linear Space)
// Like Diagonal Selective but with FULL R_h matrix
// h_t = (1 - delta) * h_{t-1} + delta * tanh(W_x @ x_t + R_h @ h_{t-1} + b)
// =============================================================================

template<typename T>
struct FullRecurrenceElmanForward {
    FullRecurrenceElmanForward(
        bool training,
        int batch_size,
        int dim,
        int n_groups,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,       // [dim, dim]
        const T* R_h,       // [dim, dim] FULL recurrence matrix
        const T* W_delta,   // [dim, dim]
        const T* W_out,     // [dim, dim]
        const T* b,         // [dim]
        const T* b_delta,   // [dim]
        const T* x,         // [T, B, dim]
        T* h,               // [T+1, B, dim]
        T* output,          // [T, B, dim]
        T* v,               // [T, B, dim]
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
struct FullRecurrenceElmanBackward {
    FullRecurrenceElmanBackward(
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
        const T* h,
        const T* v,
        const T* delta_cache,
        const T* compete_cache,
        const T* d_output,
        T* dx,
        T* dW_x,
        T* dR_h,
        T* dW_delta,
        T* dW_out,
        T* db,
        T* db_delta,
        T* workspace);  // [8*B*dim + ceil(2*dim*sizeof(float)/sizeof(T))]

private:
    int batch_size_;
    int dim_;
    int n_groups_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// Level 5: Linear Triple R Elman
// Full Triple R architecture in linear space:
// v = R_x @ x + R_h @ h_prev + b
// delta = sigmoid(W_delta @ x + R_delta @ h_prev + b_delta)
// h_new = (1-delta) * h_prev + delta * tanh(v)
// =============================================================================

template<typename T>
struct LinearTripleRForward {
    LinearTripleRForward(
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
        T* h,                // [T+1, B, dim]
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
struct LinearTripleRBackward {
    LinearTripleRBackward(
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
        T* db_delta,
        T* workspace);  // [10*B*dim + ceil(2*dim*sizeof(float)/sizeof(T))]

private:
    int batch_size_;
    int dim_;
    int n_groups_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// Level 6: Linear Polynomial Elman
// Polynomial activation in linear space:
// alpha = 1 + softplus(W_alpha @ x + b_alpha)
// v = W_x @ x + r_h * h_prev + b
// candidate = sign(v) * |v|^alpha
// h_new = (1-delta) * h_prev + delta * candidate
// =============================================================================

template<typename T>
struct LinearPolynomialForward {
    LinearPolynomialForward(
        bool training,
        int batch_size,
        int dim,
        int n_groups,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,        // [dim, dim]
        const T* r_h,        // [dim] diagonal
        const T* W_alpha,    // [dim, dim]
        const T* b_alpha,    // [dim]
        const T* W_delta,    // [dim, dim]
        const T* W_out,      // [dim, dim]
        const T* b,          // [dim]
        const T* b_delta,    // [dim]
        const T* x,          // [T, B, dim]
        T* h,                // [T+1, B, dim]
        T* output,           // [T, B, dim]
        T* v,                // [T, B, dim]
        T* alpha_cache,      // [T, B, dim]
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
struct LinearPolynomialBackward {
    LinearPolynomialBackward(
        int batch_size,
        int dim,
        int n_groups,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,
        const T* r_h,
        const T* W_alpha,
        const T* W_delta,
        const T* W_out,
        const T* x,
        const T* h,
        const T* v,
        const T* alpha_cache,
        const T* delta_cache,
        const T* compete_cache,
        const T* d_output,
        T* dx,
        T* dW_x,
        T* dr_h,
        T* dW_alpha,
        T* db_alpha,
        T* dW_delta,
        T* dW_out,
        T* db,
        T* db_delta,
        T* workspace);  // [(4*T+5)*B*dim + ceil(4*dim*sizeof(float)/sizeof(T))]

private:
    int batch_size_;
    int dim_;
    int n_groups_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// =============================================================================
// Level 7: Diagonal Triple R (Linear Space)
// Like Level 5 (Triple R) but with DIAGONAL r_h and r_delta instead of
// full matrices. More efficient O(d) recurrence.
//
// Architecture:
//   v = r_h * h_prev + W_x @ x + b              -- diagonal r_h (element-wise)
//   delta_raw = W_delta @ x + r_delta * h_prev + b_delta  -- diagonal r_delta
//   delta = sigmoid(delta_raw)
//   h_new = (1 - delta) * h_prev + delta * tanh(v)
//
//   // Selective output
//   compete = softmax(h_new.reshape(groups), dim=-1)
//   output = compete * silu(W_out @ h_new)
// =============================================================================

template<typename T>
struct DiagTripleRForward {
    DiagTripleRForward(
        bool training,
        int batch_size,
        int dim,
        int n_groups,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,           // [dim, dim]
        const T* r_h,           // [dim] diagonal recurrence
        const T* r_delta,       // [dim] diagonal delta modulation
        const T* W_delta,       // [dim, dim]
        const T* W_out,         // [dim] b_gate for h+x selective output
        const T* b,             // [dim]
        const T* b_delta,       // [dim]
        const T* x,             // [T, B, dim]
        T* h,                   // [T+1, B, dim]
        T* output,              // [T, B, dim]
        T* v_cache,             // [T, B, dim] for backward
        T* delta_cache,         // [T, B, dim] for backward
        T* compete_cache);      // [T, B, dim] gate_cache for backward

private:
    bool training_;
    int batch_size_;
    int dim_;
    int n_groups_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

template<typename T>
struct DiagTripleRBackward {
    DiagTripleRBackward(
        int batch_size,
        int dim,
        int n_groups,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream);

    void Run(
        int steps,
        const T* W_x,           // [dim, dim]
        const T* r_h,           // [dim]
        const T* r_delta,       // [dim]
        const T* W_delta,       // [dim, dim]
        const T* W_out,         // [dim] b_gate
        const T* x,             // [T, B, dim]
        const T* h,             // [T+1, B, dim]
        const T* v_cache,       // [T, B, dim]
        const T* delta_cache,   // [T, B, dim]
        const T* compete_cache, // [T, B, dim] gate_cache
        const T* d_output,      // [T, B, dim]
        T* dx,                  // [T, B, dim]
        T* dW_x,                // [dim, dim]
        T* d_r_h,               // [dim]
        T* d_r_delta,           // [dim]
        T* dW_delta,            // [dim, dim]
        T* dW_out,              // [dim] d_b_gate
        T* db,                  // [dim]
        T* db_delta,            // [dim]
        T* workspace);          // [6*B*dim + ceil(5*dim*sizeof(float)/sizeof(T))]

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
