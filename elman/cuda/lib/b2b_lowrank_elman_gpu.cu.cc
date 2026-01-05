// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// E5 B2B GEMM Fusion: Uses CUTLASS two-tensor-op fusion for low-rank recurrence.
//
// The recurrence: result = U_h @ (V_h @ h_prev)
// Using transpose trick: result^T = h_prev^T @ V_h^T @ U_h^T
//
// GEMM0: (batch × dim) @ (dim × rank) = (batch × rank)
// GEMM1: (batch × rank) @ (rank × dim) = (batch × dim)
//
// B2B Constraints satisfied:
// - M0 = M1 = batch
// - N0 = K1 = rank
// - ThreadblockShape0::kN = rank (requires rank = 64, 128, or 256)
//
// Key insight: Column-major memory layout = row-major transposed (same bytes)
// V_h stored as (rank, dim) row-major = V_h^T (dim, rank) column-major
// U_h stored as (dim, rank) row-major = U_h^T (rank, dim) column-major

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <type_traits>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/numeric_types.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"

// B2B GEMM device header (from CUTLASS example 13)
#include "cutlass_b2b/device/b2b_gemm.h"

#include "hasty/elman_ladder.h"
#include "blas.h"
#include "inline_ops.h"

// Type mapping from CUDA to CUTLASS (only for supported types)
template<typename T> struct CutlassType {
    using type = void;  // Default: not supported
    static constexpr bool supported = false;
};
template<> struct CutlassType<__nv_bfloat16> {
    using type = cutlass::bfloat16_t;
    static constexpr bool supported = true;
};
template<> struct CutlassType<__half> {
    using type = cutlass::half_t;
    static constexpr bool supported = true;
};
// Note: float and double are NOT supported by TensorOp B2B GEMM

namespace {

// Fused kernel for post-B2B operations: add bias + tanh + gate
template<typename T>
__global__ void PostB2BKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ UVh,       // [batch, dim] - from B2B GEMM
    const T* __restrict__ UVx,       // [batch, dim] - precomputed
    const T* __restrict__ UVz,       // [batch, dim] - precomputed
    const T* __restrict__ b,         // [dim]
    T* __restrict__ h_out,           // [batch, dim]
    T* __restrict__ v_cache,         // [batch, dim] or null
    T* __restrict__ output) {        // [batch, dim]

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // Compute pre-activation: UVh + UVx + b
        float val = static_cast<float>(UVh[idx]) +
                    static_cast<float>(UVx[idx]) +
                    static_cast<float>(b[d]);

        // Save for backward if training
        if (v_cache) v_cache[idx] = static_cast<T>(val);

        // Compute h = tanh(val)
        float h_val = tanhf(val);
        h_out[idx] = static_cast<T>(h_val);

        // Compute output = h * silu(z)
        float z_val = static_cast<float>(UVz[idx]);
        float sigmoid_z = 1.0f / (1.0f + expf(-z_val));
        float silu_z = z_val * sigmoid_z;
        output[idx] = static_cast<T>(h_val * silu_z);
    }
}

// ============================================================================
// B2B GEMM Type Definitions for rank=64
// ============================================================================

// For bf16/fp16 tensor cores on SM80+
template<typename ElementT>
struct B2bGemmTypes_Rank64 {
    using ElementOutput = ElementT;
    using ElementAccumulator = float;  // Use fp32 accumulator for stability
    using ElementCompute = float;

    // Threadblock tile sizes - N0 must equal rank (64)
    using ThreadblockShape0 = cutlass::gemm::GemmShape<128, 64, 32>;  // M, N, K
    using ThreadblockShape1 = cutlass::gemm::GemmShape<128, 128, 32>; // M, N, K (N=dim tiles)
    using WarpShape0 = cutlass::gemm::GemmShape<64, 32, 32>;
    using WarpShape1 = cutlass::gemm::GemmShape<64, 64, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

    // GEMM0 epilogue - LinearCombinationRelu required by CUTLASS B2B
    // NOTE: This applies max(0, x) to intermediate values which is incorrect
    // for our use case. TODO: Modify CUTLASS to support identity epilogue.
    using EpilogueOutputOp0 = cutlass::epilogue::thread::LinearCombinationRelu<
        ElementOutput,
        InstructionShape::kM * InstructionShape::kN / 32,
        ElementAccumulator,
        ElementCompute,
        cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
    >;

    using EpilogueOutputOp1 = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator,
        ElementCompute
    >;

    // Use shared memory accumulator
    static constexpr bool SmemAccumulator = true;

    // B2B GEMM device type
    // A0: h_prev^T (batch × dim) - RowMajor
    // B0: V_h^T (dim × rank) - ColumnMajor (V_h stored as rank×dim row-major)
    // B1: U_h^T (rank × dim) - ColumnMajor (U_h stored as dim×rank row-major)
    // D1: result^T (batch × dim) - RowMajor
    using B2bGemm = cutlass::gemm::device::B2bGemm<
        ElementT,                              // ElementA
        cutlass::layout::RowMajor,             // LayoutA
        ElementT,                              // ElementB
        cutlass::layout::ColumnMajor,          // LayoutB
        ElementOutput,                         // ElementC
        cutlass::layout::RowMajor,             // LayoutC
        ElementAccumulator,                    // ElementAccumulator
        cutlass::arch::OpClassTensorOp,        // OperatorClass
        cutlass::arch::Sm80,                   // Architecture
        ThreadblockShape0,
        ThreadblockShape1,
        WarpShape0,
        WarpShape1,
        InstructionShape,
        EpilogueOutputOp0,
        EpilogueOutputOp1,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
        3,  // Stages
        SmemAccumulator
    >;
};

// ============================================================================
// B2B GEMM Type Definitions for rank=128
// ============================================================================

template<typename ElementT>
struct B2bGemmTypes_Rank128 {
    using ElementOutput = ElementT;
    using ElementAccumulator = float;
    using ElementCompute = float;

    // N0 must equal rank (128)
    using ThreadblockShape0 = cutlass::gemm::GemmShape<128, 128, 32>;
    using ThreadblockShape1 = cutlass::gemm::GemmShape<128, 128, 32>;
    using WarpShape0 = cutlass::gemm::GemmShape<64, 64, 32>;
    using WarpShape1 = cutlass::gemm::GemmShape<64, 64, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

    // GEMM0 epilogue - LinearCombinationRelu required by CUTLASS B2B
    // NOTE: This applies max(0, x) which is incorrect for our use case
    using EpilogueOutputOp0 = cutlass::epilogue::thread::LinearCombinationRelu<
        ElementOutput,
        InstructionShape::kM * InstructionShape::kN / 32,
        ElementAccumulator,
        ElementCompute,
        cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
    >;

    using EpilogueOutputOp1 = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator,
        ElementCompute
    >;

    static constexpr bool SmemAccumulator = true;

    using B2bGemm = cutlass::gemm::device::B2bGemm<
        ElementT,
        cutlass::layout::RowMajor,
        ElementT,
        cutlass::layout::ColumnMajor,
        ElementOutput,
        cutlass::layout::RowMajor,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        ThreadblockShape0,
        ThreadblockShape1,
        WarpShape0,
        WarpShape1,
        InstructionShape,
        EpilogueOutputOp0,
        EpilogueOutputOp1,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
        3,
        SmemAccumulator
    >;
};

// ============================================================================
// B2B GEMM Type Definitions for rank=256
// ============================================================================

template<typename ElementT>
struct B2bGemmTypes_Rank256 {
    using ElementOutput = ElementT;
    using ElementAccumulator = float;
    using ElementCompute = float;

    // N0 must equal rank (256)
    using ThreadblockShape0 = cutlass::gemm::GemmShape<64, 256, 32>;
    using ThreadblockShape1 = cutlass::gemm::GemmShape<64, 256, 32>;
    using WarpShape0 = cutlass::gemm::GemmShape<32, 64, 32>;
    using WarpShape1 = cutlass::gemm::GemmShape<32, 64, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

    // GEMM0 epilogue - LinearCombinationRelu required by CUTLASS B2B
    // NOTE: This applies max(0, x) which is incorrect for our use case
    using EpilogueOutputOp0 = cutlass::epilogue::thread::LinearCombinationRelu<
        ElementOutput,
        InstructionShape::kM * InstructionShape::kN / 32,
        ElementAccumulator,
        ElementCompute,
        cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
    >;

    using EpilogueOutputOp1 = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator,
        ElementCompute
    >;

    static constexpr bool SmemAccumulator = true;

    using B2bGemm = cutlass::gemm::device::B2bGemm<
        ElementT,
        cutlass::layout::RowMajor,
        ElementT,
        cutlass::layout::ColumnMajor,
        ElementOutput,
        cutlass::layout::RowMajor,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        ThreadblockShape0,
        ThreadblockShape1,
        WarpShape0,
        WarpShape1,
        InstructionShape,
        EpilogueOutputOp0,
        EpilogueOutputOp1,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
        3,
        SmemAccumulator
    >;
};

// =============================================================================
// Helper to run CUTLASS B2B GEMM for h path: result = U_h @ V_h @ h_prev
// Using transpose: result^T = h_prev^T @ V_h^T @ U_h^T
// =============================================================================

template<typename B2bGemmOp, typename T>
void RunB2bGemm(
    int batch_size,
    int dim,
    int rank,
    const T* h_prev,   // [batch, dim] row-major
    const T* V_h,      // [rank, dim] row-major (interpreted as V_h^T column-major)
    const T* U_h,      // [dim, rank] row-major (interpreted as U_h^T column-major)
    T* output,         // [batch, dim] row-major
    cudaStream_t stream) {

    using CutlassT = typename CutlassType<T>::type;
    using LayoutRowMajor = cutlass::layout::RowMajor;
    using LayoutColMajor = cutlass::layout::ColumnMajor;

    // Problem sizes:
    // GEMM0: h_prev^T (batch × dim) @ V_h^T (dim × rank) = intermediate (batch × rank)
    // GEMM1: intermediate (batch × rank) @ U_h^T (rank × dim) = result (batch × dim)
    cutlass::gemm::GemmCoord problem_size_0(batch_size, rank, dim);
    cutlass::gemm::GemmCoord problem_size_1(batch_size, dim, rank);

    // Create TensorRefs (non-const element types to match iterator expectations)
    // A0: h_prev, RowMajor (batch × dim), stride = dim
    cutlass::TensorRef<CutlassT, LayoutRowMajor> ref_A0(
        reinterpret_cast<CutlassT*>(const_cast<T*>(h_prev)),
        LayoutRowMajor(dim));

    // B0: V_h^T, ColumnMajor (dim × rank)
    // V_h stored as (rank, dim) row-major, reinterpreted as column-major with stride = dim
    cutlass::TensorRef<CutlassT, LayoutColMajor> ref_B0(
        reinterpret_cast<CutlassT*>(const_cast<T*>(V_h)),
        LayoutColMajor(dim));

    // C0: unused (no accumulation into intermediate), use dummy with proper stride
    cutlass::TensorRef<CutlassT, LayoutRowMajor> ref_C0(
        nullptr,
        LayoutRowMajor(rank));

    // Scale0, Bias0: unused (identity epilogue for GEMM0)
    // ElementScaleBias = float based on EpilogueOutputOp0::ElementCompute
    cutlass::TensorRef<typename B2bGemmOp::ElementScaleBias, typename B2bGemmOp::LayoutScaleBias> ref_Scale0(
        nullptr,
        typename B2bGemmOp::LayoutScaleBias(0));
    cutlass::TensorRef<typename B2bGemmOp::ElementScaleBias, typename B2bGemmOp::LayoutScaleBias> ref_Bias0(
        nullptr,
        typename B2bGemmOp::LayoutScaleBias(0));

    // B1: U_h^T, ColumnMajor (rank × dim)
    // U_h stored as (dim, rank) row-major, reinterpreted as column-major with stride = rank
    cutlass::TensorRef<CutlassT, LayoutColMajor> ref_B1(
        reinterpret_cast<CutlassT*>(const_cast<T*>(U_h)),
        LayoutColMajor(rank));

    // C1: unused - for bias addition in final epilogue, pass null with stride 0
    cutlass::TensorRef<CutlassT, LayoutRowMajor> ref_C1(
        nullptr,
        LayoutRowMajor::Stride(0));

    // D1: output, RowMajor (batch × dim), stride = dim
    cutlass::TensorRef<CutlassT, LayoutRowMajor> ref_D1(
        reinterpret_cast<CutlassT*>(output),
        LayoutRowMajor(dim));

    // Epilogue params: alpha=1, beta=0 for both GEMMs (pure matrix multiply)
    typename B2bGemmOp::EpilogueOutputOp0::Params epilogue0(1.0f, 0.0f);
    typename B2bGemmOp::EpilogueOutputOp1::Params epilogue1(1.0f, 0.0f);

    // Create Arguments with proper int64_t batch strides
    typename B2bGemmOp::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size_0,
        problem_size_1,
        ref_A0,
        ref_B0,
        ref_C0,
        ref_Scale0,
        ref_Bias0,
        ref_B1,
        ref_C1,
        ref_D1,
        int64_t(0),  // batch_stride_A0
        int64_t(0),  // batch_stride_B0
        int64_t(0),  // batch_stride_B1
        int64_t(0),  // batch_stride_C1
        int64_t(0),  // batch_stride_D1
        int64_t(0),  // batch_stride_Bias0
        int64_t(0),  // batch_stride_Scale0
        epilogue0,
        epilogue1,
        1   // batch_count
    };

    // Create and run B2B GEMM operator
    B2bGemmOp b2b_gemm_op;
    cutlass::Status status = b2b_gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        // Fall back should not happen if we validated rank
        return;
    }

    status = b2b_gemm_op.initialize(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        return;
    }

    status = b2b_gemm_op(stream);
    // Ignore status for now - caller should handle errors
}

// Dispatch to correct rank template - only for bf16/half
template<typename T>
typename std::enable_if<CutlassType<T>::supported>::type
RunB2bGemmDispatch(
    int batch_size,
    int dim,
    int rank,
    const T* h_prev,
    const T* V_h,
    const T* U_h,
    T* output,
    cudaStream_t stream) {

    using CutlassT = typename CutlassType<T>::type;

    if (rank == 64) {
        using B2bTypes = B2bGemmTypes_Rank64<CutlassT>;
        RunB2bGemm<typename B2bTypes::B2bGemm, T>(
            batch_size, dim, rank, h_prev, V_h, U_h, output, stream);
    } else if (rank == 128) {
        using B2bTypes = B2bGemmTypes_Rank128<CutlassT>;
        RunB2bGemm<typename B2bTypes::B2bGemm, T>(
            batch_size, dim, rank, h_prev, V_h, U_h, output, stream);
    } else if (rank == 256) {
        using B2bTypes = B2bGemmTypes_Rank256<CutlassT>;
        RunB2bGemm<typename B2bTypes::B2bGemm, T>(
            batch_size, dim, rank, h_prev, V_h, U_h, output, stream);
    }
    // else: unsupported rank, caller should have validated
}

// Fallback for unsupported types (float, double) - does nothing
template<typename T>
typename std::enable_if<!CutlassType<T>::supported>::type
RunB2bGemmDispatch(
    int /*batch_size*/,
    int /*dim*/,
    int /*rank*/,
    const T* /*h_prev*/,
    const T* /*V_h*/,
    const T* /*U_h*/,
    T* /*output*/,
    cudaStream_t /*stream*/) {
    // B2B GEMM not supported for this type - caller should use cuBLAS fallback
}

}  // anonymous namespace


namespace hasty {
namespace v0 {
namespace elman_ladder {

// =============================================================================
// B2B Low-Rank Elman Forward
// =============================================================================

template<typename T>
B2bLowRankElmanForward<T>::B2bLowRankElmanForward(
    bool training,
    int batch_size,
    int dim,
    int rank,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : training_(training),
      batch_size_(batch_size),
      dim_(dim),
      rank_(rank),
      blas_handle_(blas_handle),
      stream_(stream),
      // B2B GEMM disabled: CUTLASS B2B requires ReLU epilogue which breaks correctness
      // TODO: Modify CUTLASS B2B to support identity epilogue (no activation)
      // Infrastructure is in place - just need to enable when identity epilogue works
      b2b_supported_(false) {}
      // When ready: CutlassType<T>::supported && (rank == 64 || rank == 128 || rank == 256)

template<typename T>
void B2bLowRankElmanForward<T>::Run(
    int steps,
    const T* U_h,
    const T* V_h,
    const T* U_x,
    const T* V_x,
    const T* U_z,
    const T* V_z,
    const T* b,
    const T* x,
    T* h,
    T* output,
    T* v,
    T* workspace) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int BD = batch_size_ * dim_;
    const int BR = batch_size_ * rank_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;

    // Workspace layout:
    // [tmp_Vx_all: T*BR] [tmp_UVx_all: T*BD] [tmp_Vz_all: T*BR] [tmp_UVz_all: T*BD]
    // [tmp_UVh: BD] (for B2B output or sequential intermediate)
    T* tmp_Vx_all = workspace;
    T* tmp_UVx_all = tmp_Vx_all + steps * BR;
    T* tmp_Vz_all = tmp_UVx_all + steps * BD;
    T* tmp_UVz_all = tmp_Vz_all + steps * BR;
    T* tmp_UVh = tmp_UVz_all + steps * BD;

    // Pre-compute V_x @ x for all timesteps (time-parallel)
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        rank_, steps * batch_size_, dim_,
        &alpha, V_x, dim_,
        x, dim_,
        &beta_zero, tmp_Vx_all, rank_);

    // Pre-compute U_x @ tmp_Vx for all timesteps
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, rank_,
        &alpha, U_x, rank_,
        tmp_Vx_all, rank_,
        &beta_zero, tmp_UVx_all, dim_);

    // Pre-compute V_z @ x for all timesteps
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        rank_, steps * batch_size_, dim_,
        &alpha, V_z, dim_,
        x, dim_,
        &beta_zero, tmp_Vz_all, rank_);

    // Pre-compute U_z @ tmp_Vz for all timesteps
    blas<T>::gemm(
        blas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim_, steps * batch_size_, rank_,
        &alpha, U_z, rank_,
        tmp_Vz_all, rank_,
        &beta_zero, tmp_UVz_all, dim_);

    // Sequential recurrence with CUTLASS B2B GEMM when supported
    T* tmp_Vh = tmp_UVh + BD;  // Additional workspace for fallback

    for (int t = 0; t < steps; ++t) {
        const T* h_prev = h + t * BD;
        T* h_curr = h + (t + 1) * BD;
        const T* UVx_t = tmp_UVx_all + t * BD;
        const T* UVz_t = tmp_UVz_all + t * BD;
        T* v_t = training_ ? v + t * BD : nullptr;
        T* out_t = output + t * BD;

        if (b2b_supported_) {
            // Use CUTLASS B2B GEMM fusion for h path
            // Computes: tmp_UVh = U_h @ (V_h @ h_prev)
            RunB2bGemmDispatch<T>(
                batch_size_, dim_, rank_,
                h_prev, V_h, U_h, tmp_UVh, stream_);
        } else {
            // Fallback: sequential cuBLAS GEMMs
            // V_h @ h_prev -> tmp_Vh
            blas<T>::gemm(
                blas_handle_,
                CUBLAS_OP_T, CUBLAS_OP_N,
                rank_, batch_size_, dim_,
                &alpha, V_h, dim_,
                h_prev, dim_,
                &beta_zero, tmp_Vh, rank_);

            // U_h @ tmp_Vh -> tmp_UVh
            blas<T>::gemm(
                blas_handle_,
                CUBLAS_OP_T, CUBLAS_OP_N,
                dim_, batch_size_, rank_,
                &alpha, U_h, rank_,
                tmp_Vh, rank_,
                &beta_zero, tmp_UVh, dim_);
        }

        // Fused post-B2B kernel: add, tanh, gate
        PostB2BKernel<<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, tmp_UVh, UVx_t, UVz_t, b, h_curr, v_t, out_t);
    }
}

template struct B2bLowRankElmanForward<double>;
template struct B2bLowRankElmanForward<float>;
template struct B2bLowRankElmanForward<__nv_bfloat16>;
template struct B2bLowRankElmanForward<__half>;

// =============================================================================
// B2B Low-Rank Elman Backward (same as fused version)
// =============================================================================

template<typename T>
B2bLowRankElmanBackward<T>::B2bLowRankElmanBackward(
    int batch_size,
    int dim,
    int rank,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      rank_(rank),
      blas_handle_(blas_handle),
      stream_(stream) {}

// Backward through post-B2B kernel
template<typename T>
__global__ void PostB2BBackwardKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ v,         // [B, dim] pre-activation
    const T* __restrict__ z,         // [B, dim] (U_z @ V_z @ x)
    const T* __restrict__ d_output,  // [B, dim]
    const T* __restrict__ dh_rec,    // [B, dim] recurrent gradient (or null)
    T* __restrict__ dv,              // [B, dim] gradient to pre-activation
    T* __restrict__ dz,              // [B, dim] gradient to z
    float* __restrict__ db) {        // [dim] accumulates

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float v_val = static_cast<float>(v[idx]);
        float z_val = static_cast<float>(z[idx]);
        float h = tanhf(v_val);

        float sigmoid_z = 1.0f / (1.0f + expf(-z_val));
        float silu_z = z_val * sigmoid_z;
        float dsilu = sigmoid_z * (1.0f + z_val * (1.0f - sigmoid_z));

        float dout = static_cast<float>(d_output[idx]);
        float dh_from_output = dout * silu_z;
        dz[idx] = static_cast<T>(dout * h * dsilu);

        float grad_h = dh_from_output;
        if (dh_rec) grad_h += static_cast<float>(dh_rec[idx]);

        float dtanh = 1.0f - h * h;
        float dv_val = grad_h * dtanh;
        dv[idx] = static_cast<T>(dv_val);

        atomicAdd(&db[d], dv_val);
    }
}

template<typename T>
__global__ void CopyF32ToT(const int n, const float* __restrict__ src, T* __restrict__ dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = static_cast<T>(src[idx]);
    }
}

template<typename T>
void B2bLowRankElmanBackward<T>::Run(
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
    T* workspace) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int BD = batch_size_ * dim_;
    const int BR = batch_size_ * rank_;
    const int block_size = 256;
    const int num_blocks_d = (BD + block_size - 1) / block_size;
    const int num_blocks_dim = (dim_ + block_size - 1) / block_size;

    // Workspace
    T* dh_rec = workspace;
    T* dv_t = dh_rec + BD;
    T* dz = dv_t + BD;
    T* tmp_Vh = dz + BD;
    T* dVh = tmp_Vh + BR;
    T* tmp_Vx = dVh + BR;
    T* dVx = tmp_Vx + BR;
    T* tmp_Vz = dVx + BR;
    T* dVz = tmp_Vz + BR;
    T* tmp_UVz = dVz + BR;
    float* db_f32 = reinterpret_cast<float*>(tmp_UVz + BD);

    // Initialize gradients
    cudaMemsetAsync(dU_h, 0, dim_ * rank_ * sizeof(T), stream_);
    cudaMemsetAsync(dV_h, 0, rank_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dU_x, 0, dim_ * rank_ * sizeof(T), stream_);
    cudaMemsetAsync(dV_x, 0, rank_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(dU_z, 0, dim_ * rank_ * sizeof(T), stream_);
    cudaMemsetAsync(dV_z, 0, rank_ * dim_ * sizeof(T), stream_);
    cudaMemsetAsync(db_f32, 0, dim_ * sizeof(float), stream_);
    cudaMemsetAsync(dh_rec, 0, BD * sizeof(T), stream_);
    cudaMemsetAsync(dx, 0, steps * BD * sizeof(T), stream_);

    for (int t = steps - 1; t >= 0; --t) {
        const T* h_t = h + (t + 1) * BD;
        const T* h_prev = h + t * BD;
        const T* v_t_in = v + t * BD;
        const T* x_t = x + t * BD;
        const T* d_out_t = d_output + t * BD;
        T* dx_t = dx + t * BD;

        // Recompute UVz
        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            rank_, batch_size_, dim_,
            &alpha, V_z, dim_,
            x_t, dim_,
            &beta_zero, tmp_Vz, rank_);

        blas<T>::gemm(
            blas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, rank_,
            &alpha, U_z, rank_,
            tmp_Vz, rank_,
            &beta_zero, tmp_UVz, dim_);

        // Backward through fused kernel
        PostB2BBackwardKernel<<<num_blocks_d, block_size, 0, stream_>>>(
            batch_size_, dim_, v_t_in, tmp_UVz, d_out_t,
            (t < steps - 1) ? dh_rec : nullptr,
            dv_t, dz, db_f32);

        // Backward through z path
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            rank_, batch_size_, dim_,
            &alpha, U_z, rank_, dz, dim_, &beta_zero, dVz, rank_);

        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            rank_, dim_, batch_size_,
            &alpha, tmp_Vz, rank_, dz, dim_, &beta_one, dU_z, rank_);

        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, rank_, batch_size_,
            &alpha, x_t, dim_, dVz, rank_, &beta_one, dV_z, dim_);

        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, rank_,
            &alpha, V_z, dim_, dVz, rank_, &beta_one, dx_t, dim_);

        // Backward through x path
        blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            rank_, batch_size_, dim_,
            &alpha, V_x, dim_, x_t, dim_, &beta_zero, tmp_Vx, rank_);

        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            rank_, batch_size_, dim_,
            &alpha, U_x, rank_, dv_t, dim_, &beta_zero, dVx, rank_);

        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            rank_, dim_, batch_size_,
            &alpha, tmp_Vx, rank_, dv_t, dim_, &beta_one, dU_x, rank_);

        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, rank_, batch_size_,
            &alpha, x_t, dim_, dVx, rank_, &beta_one, dV_x, dim_);

        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, rank_,
            &alpha, V_x, dim_, dVx, rank_, &beta_one, dx_t, dim_);

        // Backward through h path
        blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            rank_, batch_size_, dim_,
            &alpha, V_h, dim_, h_prev, dim_, &beta_zero, tmp_Vh, rank_);

        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            rank_, batch_size_, dim_,
            &alpha, U_h, rank_, dv_t, dim_, &beta_zero, dVh, rank_);

        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            rank_, dim_, batch_size_,
            &alpha, tmp_Vh, rank_, dv_t, dim_, &beta_one, dU_h, rank_);

        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, rank_, batch_size_,
            &alpha, h_prev, dim_, dVh, rank_, &beta_one, dV_h, dim_);

        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, rank_,
            &alpha, V_h, dim_, dVh, rank_, &beta_zero, dh_rec, dim_);
    }

    CopyF32ToT<<<num_blocks_dim, block_size, 0, stream_>>>(dim_, db_f32, db);
}

template struct B2bLowRankElmanBackward<double>;
template struct B2bLowRankElmanBackward<float>;
template struct B2bLowRankElmanBackward<__nv_bfloat16>;
template struct B2bLowRankElmanBackward<__half>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace hasty
