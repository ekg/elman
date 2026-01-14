// Stub for B2B LowRank Elman (requires CUTLASS)
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

namespace hasty { namespace v0 { namespace elman_ladder {

template<typename T>
struct B2bLowRankElmanForward {
    B2bLowRankElmanForward(bool, int, int, int, const cublasHandle_t&, const cudaStream_t&) {}
    void Run(int, const T*, const T*, const T*, const T*, const T*, const T*, 
             const T*, const T*, T*, T*, T*, T*) {}
};

template<typename T>
struct B2bLowRankElmanBackward {
    B2bLowRankElmanBackward(int, int, int, const cublasHandle_t&, const cudaStream_t&) {}
    // Run(steps, U_h, V_h, U_x, V_x, U_z, V_z, x, h, v, d_output, dx, dU_h, dV_h, dU_x, dV_x, dU_z, dV_z, db, workspace)
    void Run(int, const T*, const T*, const T*, const T*, const T*, const T*,
             const T*, const T*, const T*, const T*, T*, T*, T*, T*, T*, T*, T*, T*, T*) {}
};

// Explicit instantiations - all types
template struct B2bLowRankElmanForward<double>;
template struct B2bLowRankElmanForward<float>;
template struct B2bLowRankElmanForward<__half>;
template struct B2bLowRankElmanForward<__nv_bfloat16>;
template struct B2bLowRankElmanBackward<double>;
template struct B2bLowRankElmanBackward<float>;
template struct B2bLowRankElmanBackward<__half>;
template struct B2bLowRankElmanBackward<__nv_bfloat16>;

}}}
