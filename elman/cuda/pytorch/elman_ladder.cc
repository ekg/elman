// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// PyTorch bindings for E-Series Elman kernels.
// Archived kernels are in elman_ladder_full.cc.archive

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>

#include "hasty.h"
#include "support.h"

namespace {

using torch::Tensor;

// =============================================================================
// E0: Stock Elman (tanh recurrence + h*silu(W_gate@x) gating)
// =============================================================================

std::vector<Tensor> stock_elman_forward(
    bool training,
    Tensor x,           // [T, B, dim]
    Tensor h0,          // [B, dim]
    Tensor W_x,         // [dim, dim]
    Tensor W_h,         // [dim, dim]
    Tensor W_gate,      // [dim, dim] gate projection
    Tensor b,           // [dim]
    Tensor b_gate) {    // [dim] gate bias

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(W_x);
    CHECK_INPUT(W_h);
    CHECK_INPUT(W_gate);
    CHECK_INPUT(b);
    CHECK_INPUT(b_gate);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, dim}, options)
                        : torch::empty({0}, options);
    Tensor gate_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                 : torch::empty({0}, options);

    // Forward workspace: [tmp_Wx: T*BD] [gate_proj: T*BD] [tmp_Rh: BD]
    const int64_t BD = batch_size * dim;
    Tensor workspace = torch::empty({2 * time_steps * BD + BD}, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "stock_elman_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        StockElmanForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(W_h),
            ptr<scalar_t>(W_gate),
            ptr<scalar_t>(b),
            ptr<scalar_t>(b_gate),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v) : nullptr,
            training ? ptr<scalar_t>(gate_cache) : nullptr,
            ptr<scalar_t>(workspace));
    }));

    return {h, output, v, gate_cache};
}

std::vector<Tensor> stock_elman_backward(
    Tensor W_x,
    Tensor W_h,
    Tensor W_gate,
    Tensor x,
    Tensor h,
    Tensor v,
    Tensor gate_cache,
    Tensor d_output) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(W_x);
    CHECK_INPUT(W_h);
    CHECK_INPUT(W_gate);
    CHECK_INPUT(x);
    CHECK_INPUT(h);
    CHECK_INPUT(v);
    CHECK_INPUT(gate_cache);
    CHECK_INPUT(d_output);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::empty_like(x);
    Tensor dW_x = torch::zeros({dim, dim}, options);
    Tensor dW_h = torch::zeros({dim, dim}, options);
    Tensor dW_gate = torch::zeros({dim, dim}, options);
    Tensor db = torch::zeros({dim}, options);
    Tensor db_gate = torch::zeros({dim}, options);

    // Workspace layout: [dv_all: T*BD] [d_gate_proj_all: T*BD] [dh: BD] [dh_recurrent: BD]
    //                   [db_float: dim] [db_gate_float: dim]
    const int64_t BD = batch_size * dim;
    // Float arrays need 2*dim floats = ceil(2*dim * sizeof(float) / sizeof(T)) T elements
    const int64_t float_in_T = (2 * dim * sizeof(float) + sizeof(float) - 1) / sizeof(float);
    const int64_t workspace_size = (2 * time_steps + 2) * BD + float_in_T * 2;  // *2 for bfloat16
    Tensor workspace = torch::empty({workspace_size}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "stock_elman_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        StockElmanBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(W_h),
            ptr<scalar_t>(W_gate),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(gate_cache),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dW_x),
            ptr<scalar_t>(dW_h),
            ptr<scalar_t>(dW_gate),
            ptr<scalar_t>(db),
            ptr<scalar_t>(db_gate),
            ptr<scalar_t>(workspace));
    }));

    return {dx, dW_x, dW_h, dW_gate, db, db_gate};
}

// =============================================================================
// E1: Mamba-Gated Elman (Mamba2-style split projection gating)
// =============================================================================

std::vector<Tensor> mamba_gated_elman_forward(
    bool training,
    Tensor x,           // [T, B, dim] pre-activated input
    Tensor z,           // [T, B, dim] gate input (pre silu)
    Tensor h0,          // [B, dim]
    Tensor W_x,         // [dim, dim]
    Tensor W_h,         // [dim, dim] (already spectrally normalized)
    Tensor b) {         // [dim]

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(z);
    CHECK_INPUT(h0);
    CHECK_INPUT(W_x);
    CHECK_INPUT(W_h);
    CHECK_INPUT(b);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, dim}, options)
                        : torch::empty({0}, options);

    // Forward workspace: [tmp_Wx: T*BD] [tmp_Rh: BD]
    const int64_t BD = batch_size * dim;
    Tensor workspace = torch::empty({time_steps * BD + BD}, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "mamba_gated_elman_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        MambaGatedElmanForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(W_h),
            ptr<scalar_t>(b),
            ptr<scalar_t>(x),
            ptr<scalar_t>(z),
            ptr<scalar_t>(h),
            ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v) : nullptr,
            ptr<scalar_t>(workspace));
    }));

    return {h, output, v};
}

std::vector<Tensor> mamba_gated_elman_backward(
    Tensor W_x,
    Tensor W_h,
    Tensor x,
    Tensor z,
    Tensor h,
    Tensor v,
    Tensor d_output) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(W_x);
    CHECK_INPUT(W_h);
    CHECK_INPUT(x);
    CHECK_INPUT(z);
    CHECK_INPUT(h);
    CHECK_INPUT(v);
    CHECK_INPUT(d_output);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::empty_like(x);
    Tensor dz = torch::empty_like(z);
    Tensor dW_x = torch::zeros({dim, dim}, options);
    Tensor dW_h = torch::zeros({dim, dim}, options);
    Tensor db = torch::zeros({dim}, options);

    // Workspace layout: [dv_all: T*BD] [dh: BD] [dh_recurrent: BD] [db_float: dim]
    const int64_t BD = batch_size * dim;
    const int64_t float_in_T = (dim * sizeof(float) + sizeof(float) - 1) / sizeof(float);
    const int64_t workspace_size = (time_steps + 2) * BD + float_in_T * 2;
    Tensor workspace = torch::empty({workspace_size}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "mamba_gated_elman_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        MambaGatedElmanBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(W_h),
            ptr<scalar_t>(x),
            ptr<scalar_t>(z),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dz),
            ptr<scalar_t>(dW_x),
            ptr<scalar_t>(dW_h),
            ptr<scalar_t>(db),
            ptr<scalar_t>(workspace));
    }));

    return {dx, dz, dW_x, dW_h, db};
}

// =============================================================================
// E2: Slot-Based Elman (with cuBLAS GEMMs - same as e0, more memory via slots)
// h_t[s] = tanh(W_x @ x + W_h @ h_prev[s] + b)    for each slot s
// output = sum(C[s] * h_t[s]) * silu(z)
// =============================================================================

std::vector<Tensor> slot_elman_forward(
    bool training,
    Tensor x,           // [T, B, dim] pre-activated input
    Tensor z,           // [T, B, dim] gate input
    Tensor h0,          // [B, n_slots, dim]
    Tensor W_x,         // [dim, dim]
    Tensor W_h,         // [dim, dim]
    Tensor b,           // [dim]
    Tensor C) {         // [n_slots]

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    const auto n_slots = C.size(0);

    CHECK_INPUT(x);
    CHECK_INPUT(z);
    CHECK_INPUT(h0);
    CHECK_INPUT(W_x);
    CHECK_INPUT(W_h);
    CHECK_INPUT(b);
    CHECK_INPUT(C);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    // h layout: [T+1, B, n_slots, dim] to enable batched GEMM
    Tensor h = torch::empty({time_steps + 1, batch_size, n_slots, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, n_slots, dim}, options)
                        : torch::empty({0}, options);

    // Forward workspace: [T*B*dim + B*n_slots*dim]
    const int64_t BD = batch_size * dim;
    const int64_t BSD = batch_size * n_slots * dim;
    Tensor workspace = torch::empty({time_steps * BD + BSD}, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "slot_elman_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        SlotElmanForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, n_slots,
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(W_h),
            ptr<scalar_t>(b),
            ptr<scalar_t>(C),
            ptr<scalar_t>(x),
            ptr<scalar_t>(z),
            ptr<scalar_t>(h),
            ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v) : nullptr,
            ptr<scalar_t>(workspace),
            at::cuda::getCurrentCUDABlasHandle());
    }));

    return {h, output, v};
}

std::vector<Tensor> slot_elman_backward(
    Tensor W_x,         // [dim, dim]
    Tensor W_h,         // [dim, dim]
    Tensor C,           // [n_slots]
    Tensor x,           // [T, B, dim]
    Tensor z,           // [T, B, dim]
    Tensor h,           // [T+1, B, n_slots, dim]
    Tensor v,           // [T, B, n_slots, dim]
    Tensor d_output) {  // [T, B, dim]

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    const auto n_slots = C.size(0);

    CHECK_INPUT(W_x);
    CHECK_INPUT(W_h);
    CHECK_INPUT(C);
    CHECK_INPUT(x);
    CHECK_INPUT(z);
    CHECK_INPUT(h);
    CHECK_INPUT(v);
    CHECK_INPUT(d_output);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::empty_like(x);
    Tensor dz = torch::empty_like(z);
    Tensor dW_x = torch::zeros({dim, dim}, options);
    Tensor dW_h = torch::zeros({dim, dim}, options);
    Tensor db = torch::zeros({dim}, options);
    Tensor dC = torch::zeros({n_slots}, options);

    // Workspace: [dv_all: T*BSD] [dh: BSD] [dh_recurrent: BSD] [dv_sum: T*BD] [db_float: dim] [dC_float: n_slots]
    const int64_t BD = batch_size * dim;
    const int64_t BSD = batch_size * n_slots * dim;
    // Float elements: dim + n_slots for db_float and dC_float
    const int64_t float_elements = dim + n_slots;
    const int64_t float_in_T = (float_elements * sizeof(float) + sizeof(float) - 1) / sizeof(float);
    const int64_t workspace_size = time_steps * BSD + 2 * BSD + time_steps * BD + float_in_T * 2;

    Tensor workspace = torch::empty({workspace_size}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "slot_elman_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        SlotElmanBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, n_slots,
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(W_h),
            ptr<scalar_t>(C),
            ptr<scalar_t>(x),
            ptr<scalar_t>(z),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dz),
            ptr<scalar_t>(dW_x),
            ptr<scalar_t>(dW_h),
            ptr<scalar_t>(db),
            ptr<scalar_t>(dC),
            ptr<scalar_t>(workspace),
            at::cuda::getCurrentCUDABlasHandle());
    }));

    return {dx, dz, dW_x, dW_h, db, dC};
}

// =============================================================================
// E3: Low-Rank Slot Elman (independent low-rank W_h per slot)
// =============================================================================

std::vector<Tensor> lowrank_slot_elman_forward(
    bool training,
    Tensor x,           // [T, B, dim]
    Tensor z,           // [T, B, dim]
    Tensor h0,          // [n_slots, B, dim] - CUDA layout for batched GEMM
    Tensor W_x,         // [dim, dim]
    Tensor U,           // [n_slots, dim, rank]
    Tensor V,           // [n_slots, rank, dim]
    Tensor b,           // [dim]
    Tensor C) {         // [n_slots]

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    const auto n_slots = C.size(0);
    const auto rank = U.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(z);
    CHECK_INPUT(h0);
    CHECK_INPUT(W_x);
    CHECK_INPUT(U);
    CHECK_INPUT(V);
    CHECK_INPUT(b);
    CHECK_INPUT(C);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    // h layout: [T+1, n_slots, B, dim] for efficient batched GEMM
    Tensor h = torch::empty({time_steps + 1, n_slots, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    // Vh_cache: [T, n_slots, B, rank] - stores Vh for backward GEMM (smaller than old v!)
    Tensor Vh_cache = training ? torch::empty({time_steps, n_slots, batch_size, rank}, options)
                               : torch::empty({0}, options);

    // Workspace: [T*B*dim] for pre-computed Wx only
    const int64_t BD = batch_size * dim;
    Tensor workspace = torch::empty({time_steps * BD}, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "lowrank_slot_elman_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        LowRankSlotElmanForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, n_slots, rank,
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(U),
            ptr<scalar_t>(V),
            ptr<scalar_t>(b),
            ptr<scalar_t>(C),
            ptr<scalar_t>(x),
            ptr<scalar_t>(z),
            ptr<scalar_t>(h),
            ptr<scalar_t>(output),
            training ? ptr<scalar_t>(Vh_cache) : nullptr,
            ptr<scalar_t>(workspace),
            at::cuda::getCurrentCUDABlasHandle());
    }));

    return {h, output, Vh_cache};
}

std::vector<Tensor> lowrank_slot_elman_backward(
    Tensor W_x,         // [dim, dim]
    Tensor U,           // [n_slots, dim, rank]
    Tensor V,           // [n_slots, rank, dim]
    Tensor C,           // [n_slots]
    Tensor x,           // [T, B, dim]
    Tensor z,           // [T, B, dim]
    Tensor h,           // [T+1, n_slots, B, dim]
    Tensor Vh_all,      // [T, n_slots, B, rank] - from forward cache
    Tensor d_output) {  // [T, B, dim]

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    const auto n_slots = C.size(0);
    const auto rank = U.size(2);

    CHECK_INPUT(W_x);
    CHECK_INPUT(U);
    CHECK_INPUT(V);
    CHECK_INPUT(C);
    CHECK_INPUT(x);
    CHECK_INPUT(z);
    CHECK_INPUT(h);
    CHECK_INPUT(Vh_all);
    CHECK_INPUT(d_output);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::empty_like(x);
    Tensor dz = torch::empty_like(z);
    Tensor dW_x = torch::zeros({dim, dim}, options);
    Tensor dU = torch::zeros({n_slots, dim, rank}, options);
    Tensor dV = torch::zeros({n_slots, rank, dim}, options);
    Tensor db = torch::zeros({dim}, options);
    Tensor dC = torch::zeros({n_slots}, options);

    // Workspace layout:
    // [dv_all: T*SBD] [dVh_all: T*SBR] [dh: SBD] [dh_recurrent: SBD] [dv_sum: T*BD]
    // [dU_float: dUV_size] [dV_float: dUV_size] [db_float: dim] [dC_float: n_slots]
    const int64_t BD = batch_size * dim;
    const int64_t SBD = n_slots * batch_size * dim;
    const int64_t SBR = n_slots * batch_size * rank;
    const int64_t dUV_size = n_slots * dim * rank;
    const int64_t float_elements = 2 * dUV_size + dim + n_slots;
    const int64_t float_in_T = (float_elements * sizeof(float) + sizeof(float) - 1) / sizeof(float);
    const int64_t workspace_size = time_steps * SBD + time_steps * SBR + 2 * SBD + time_steps * BD + float_in_T * 2;
    Tensor workspace = torch::empty({workspace_size}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "lowrank_slot_elman_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        LowRankSlotElmanBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, n_slots, rank,
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(U),
            ptr<scalar_t>(V),
            ptr<scalar_t>(C),
            ptr<scalar_t>(x),
            ptr<scalar_t>(z),
            ptr<scalar_t>(h),
            ptr<scalar_t>(Vh_all),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dz),
            ptr<scalar_t>(dW_x),
            ptr<scalar_t>(dU),
            ptr<scalar_t>(dV),
            ptr<scalar_t>(db),
            ptr<scalar_t>(dC),
            ptr<scalar_t>(workspace),
            at::cuda::getCurrentCUDABlasHandle());
    }));

    return {dx, dz, dW_x, dU, dV, db, dC};
}

}  // anonymous namespace


void elman_ladder_init(py::module& m) {
    m.def("stock_elman_forward", &stock_elman_forward,
          "E0: Stock Elman forward");
    m.def("stock_elman_backward", &stock_elman_backward,
          "E0: Stock Elman backward");
    m.def("mamba_gated_elman_forward", &mamba_gated_elman_forward,
          "E1: Mamba-Gated Elman forward");
    m.def("mamba_gated_elman_backward", &mamba_gated_elman_backward,
          "E1: Mamba-Gated Elman backward");
    m.def("slot_elman_forward", &slot_elman_forward,
          "E2: Slot Elman forward");
    m.def("slot_elman_backward", &slot_elman_backward,
          "E2: Slot Elman backward");
    m.def("lowrank_slot_elman_forward", &lowrank_slot_elman_forward,
          "E3: Low-Rank Slot Elman forward");
    m.def("lowrank_slot_elman_backward", &lowrank_slot_elman_backward,
          "E3: Low-Rank Slot Elman backward");
}
