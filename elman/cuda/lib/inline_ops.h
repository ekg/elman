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

#include <cuda_fp16.h>
#include <cuda_bf16.h>

template<typename T>
__device__ __forceinline__
T sigmoid(const T x) {
  return static_cast<T>(1.0) / (static_cast<T>(1.0) + exp(-x));
}

template<typename T>
__device__ __forceinline__
T tanh(const T x) {
  return std::tanh(x);
}

template<typename T>
__device__ __forceinline__
T d_sigmoid(const T sigmoid_output) {
  return sigmoid_output * (static_cast<T>(1.0) - sigmoid_output);
}

template<typename T>
__device__ __forceinline__
T d_tanh(const T tanh_output) {
  return (static_cast<T>(1.0) - tanh_output * tanh_output);
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)

__device__ __forceinline__
double atomicAdd(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600)

template<>
__device__ __forceinline__
half sigmoid(const half x) {
  return static_cast<half>(1.0) / (static_cast<half>(1.0) + hexp(-x));
}

template<>
__device__ __forceinline__
half tanh(const half x) {
  return std::tanh(float(x));
}

#endif

// bfloat16 specializations (requires SM 80+, but compute via float conversion)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

template<>
__device__ __forceinline__
__nv_bfloat16 sigmoid(const __nv_bfloat16 x) {
  float xf = __bfloat162float(x);
  return __float2bfloat16(1.0f / (1.0f + expf(-xf)));
}

template<>
__device__ __forceinline__
__nv_bfloat16 tanh(const __nv_bfloat16 x) {
  return __float2bfloat16(tanhf(__bfloat162float(x)));
}

template<>
__device__ __forceinline__
__nv_bfloat16 d_sigmoid(const __nv_bfloat16 sigmoid_output) {
  float s = __bfloat162float(sigmoid_output);
  return __float2bfloat16(s * (1.0f - s));
}

template<>
__device__ __forceinline__
__nv_bfloat16 d_tanh(const __nv_bfloat16 tanh_output) {
  float t = __bfloat162float(tanh_output);
  return __float2bfloat16(1.0f - t * t);
}

#endif
