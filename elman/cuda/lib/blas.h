// Copyright 2020 LMNT, Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#pragma once

#include <cublas_v2.h>

template<typename T>
struct blas {
  struct set_pointer_mode {
    set_pointer_mode(cublasHandle_t handle) : handle_(handle) {
      cublasGetPointerMode(handle_, &old_mode_);
      cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST);
    }
    ~set_pointer_mode() {
      cublasSetPointerMode(handle_, old_mode_);
    }
    private:
      cublasHandle_t handle_;
      cublasPointerMode_t old_mode_;
  };
  struct enable_tensor_cores {
    enable_tensor_cores(cublasHandle_t handle) : handle_(handle) {
      cublasGetMathMode(handle_, &old_mode_);
      cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH);
    }
    ~enable_tensor_cores() {
      cublasSetMathMode(handle_, old_mode_);
    }
    private:
      cublasHandle_t handle_;
      cublasMath_t old_mode_;
  };
};

// BF16 wrapper using cublasGemmEx (no direct cublasBgemm exists)
#include <cuda_bf16.h>

inline cublasStatus_t cublasBf16Gemm(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const __nv_bfloat16* alpha,
    const __nv_bfloat16* A, int lda,
    const __nv_bfloat16* B, int ldb,
    const __nv_bfloat16* beta,
    __nv_bfloat16* C, int ldc) {
  // Convert alpha/beta to float for compute
  float alpha_f = __bfloat162float(*alpha);
  float beta_f = __bfloat162float(*beta);
  return cublasGemmEx(
      handle, transa, transb, m, n, k,
      &alpha_f,
      A, CUDA_R_16BF, lda,
      B, CUDA_R_16BF, ldb,
      &beta_f,
      C, CUDA_R_16BF, ldc,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT);
}

template<>
struct blas<__nv_bfloat16> {
  static constexpr decltype(cublasBf16Gemm)* gemm = &cublasBf16Gemm;

  static cublasStatus_t gemmStridedBatched(
      cublasHandle_t handle,
      cublasOperation_t transa, cublasOperation_t transb,
      int m, int n, int k,
      const __nv_bfloat16* alpha,
      const __nv_bfloat16* A, int lda, long long strideA,
      const __nv_bfloat16* B, int ldb, long long strideB,
      const __nv_bfloat16* beta,
      __nv_bfloat16* C, int ldc, long long strideC,
      int batchCount) {
    float alpha_f = __bfloat162float(*alpha);
    float beta_f = __bfloat162float(*beta);
    return cublasGemmStridedBatchedEx(
        handle, transa, transb, m, n, k,
        &alpha_f,
        A, CUDA_R_16BF, lda, strideA,
        B, CUDA_R_16BF, ldb, strideB,
        &beta_f,
        C, CUDA_R_16BF, ldc, strideC,
        batchCount,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);
  }
};

// Strided batched GEMM for float
template<>
struct blas<float> {
  static constexpr decltype(cublasSgemm)* gemm = &cublasSgemm;

  static cublasStatus_t gemmStridedBatched(
      cublasHandle_t handle,
      cublasOperation_t transa, cublasOperation_t transb,
      int m, int n, int k,
      const float* alpha,
      const float* A, int lda, long long strideA,
      const float* B, int ldb, long long strideB,
      const float* beta,
      float* C, int ldc, long long strideC,
      int batchCount) {
    return cublasSgemmStridedBatched(
        handle, transa, transb, m, n, k,
        alpha, A, lda, strideA,
        B, ldb, strideB,
        beta, C, ldc, strideC,
        batchCount);
  }
};

// Strided batched GEMM for double
template<>
struct blas<double> {
  static constexpr decltype(cublasDgemm)* gemm = &cublasDgemm;

  static cublasStatus_t gemmStridedBatched(
      cublasHandle_t handle,
      cublasOperation_t transa, cublasOperation_t transb,
      int m, int n, int k,
      const double* alpha,
      const double* A, int lda, long long strideA,
      const double* B, int ldb, long long strideB,
      const double* beta,
      double* C, int ldc, long long strideC,
      int batchCount) {
    return cublasDgemmStridedBatched(
        handle, transa, transb, m, n, k,
        alpha, A, lda, strideA,
        B, ldb, strideB,
        beta, C, ldc, strideC,
        batchCount);
  }
};

// Strided batched GEMM for half
template<>
struct blas<__half> {
  static constexpr decltype(cublasHgemm)* gemm = &cublasHgemm;

  static cublasStatus_t gemmStridedBatched(
      cublasHandle_t handle,
      cublasOperation_t transa, cublasOperation_t transb,
      int m, int n, int k,
      const __half* alpha,
      const __half* A, int lda, long long strideA,
      const __half* B, int ldb, long long strideB,
      const __half* beta,
      __half* C, int ldc, long long strideC,
      int batchCount) {
    return cublasHgemmStridedBatched(
        handle, transa, transb, m, n, k,
        alpha, A, lda, strideA,
        B, ldb, strideB,
        beta, C, ldc, strideC,
        batchCount);
  }
};
