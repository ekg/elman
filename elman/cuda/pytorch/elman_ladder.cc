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

// =============================================================================
// E4: Low-Rank Elman (SVD-style for fat hidden state)
// =============================================================================

std::vector<Tensor> lowrank_elman_forward(
    bool training,
    Tensor x,           // [T, B, dim] pre-activated input
    Tensor z,           // [T, B, dim] gate input
    Tensor h0,          // [B, dim] initial hidden
    Tensor W_x,         // [dim, dim]
    Tensor U,           // [dim, rank]
    Tensor V,           // [rank, dim]
    Tensor b) {         // [dim]

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    const auto rank = U.size(1);

    CHECK_INPUT(x);
    CHECK_INPUT(z);
    CHECK_INPUT(h0);
    CHECK_INPUT(W_x);
    CHECK_INPUT(U);
    CHECK_INPUT(V);
    CHECK_INPUT(b);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, dim}, options)
                        : torch::empty({0}, options);

    // Workspace: [tmp_Wx: T*BD] [tmp_Vh: BR] [tmp_UVh: BD]
    const int64_t BD = batch_size * dim;
    const int64_t BR = batch_size * rank;
    Tensor workspace = torch::empty({time_steps * BD + BR + BD}, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "lowrank_elman_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        LowRankElmanForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, rank,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(U),
            ptr<scalar_t>(V),
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

std::vector<Tensor> lowrank_elman_backward(
    Tensor W_x,         // [dim, dim]
    Tensor U,           // [dim, rank]
    Tensor V,           // [rank, dim]
    Tensor x,           // [T, B, dim]
    Tensor z,           // [T, B, dim]
    Tensor h,           // [T+1, B, dim]
    Tensor v,           // [T, B, dim] pre-activation cache
    Tensor d_output) {  // [T, B, dim]

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    const auto rank = U.size(1);

    CHECK_INPUT(W_x);
    CHECK_INPUT(U);
    CHECK_INPUT(V);
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
    Tensor dU = torch::zeros({dim, rank}, options);
    Tensor dV = torch::zeros({rank, dim}, options);
    Tensor db = torch::zeros({dim}, options);

    // Workspace: [dh_curr: BD] [dh_next: BD] [dv_t: BD] [dVh: BR] [db_f32: dim]
    const int64_t BD = batch_size * dim;
    const int64_t BR = batch_size * rank;
    const int64_t float_dim = (dim * sizeof(float) + sizeof(float) - 1) / sizeof(float);
    Tensor workspace = torch::empty({3 * BD + BR + float_dim * 2}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "lowrank_elman_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        LowRankElmanBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, rank,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(U),
            ptr<scalar_t>(V),
            ptr<scalar_t>(x),
            ptr<scalar_t>(z),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dz),
            ptr<scalar_t>(dW_x),
            ptr<scalar_t>(dU),
            ptr<scalar_t>(dV),
            ptr<scalar_t>(db),
            ptr<scalar_t>(workspace));
    }));

    return {dx, dz, dW_x, dU, dV, db};
}

// =============================================================================
// E5: Pure Low-Rank Elman (no projections, all low-rank on full dim)
// =============================================================================

std::vector<Tensor> pure_lowrank_elman_forward(
    bool training,
    Tensor x,           // [T, B, dim]
    Tensor h0,          // [B, dim]
    Tensor U_h,         // [dim, rank]
    Tensor V_h,         // [rank, dim]
    Tensor U_x,         // [dim, rank]
    Tensor V_x,         // [rank, dim]
    Tensor U_z,         // [dim, rank]
    Tensor V_z,         // [rank, dim]
    Tensor b) {         // [dim]

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    const auto rank = U_h.size(1);

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(U_h);
    CHECK_INPUT(V_h);
    CHECK_INPUT(U_x);
    CHECK_INPUT(V_x);
    CHECK_INPUT(U_z);
    CHECK_INPUT(V_z);
    CHECK_INPUT(b);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, dim}, options)
                        : torch::empty({0}, options);

    // Workspace: [2*T*BR + 2*T*BD + BR + BD]
    const int64_t BD = batch_size * dim;
    const int64_t BR = batch_size * rank;
    Tensor workspace = torch::empty({2 * time_steps * BR + 2 * time_steps * BD + BR + BD}, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "pure_lowrank_elman_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        PureLowRankElmanForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, rank,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(U_h),
            ptr<scalar_t>(V_h),
            ptr<scalar_t>(U_x),
            ptr<scalar_t>(V_x),
            ptr<scalar_t>(U_z),
            ptr<scalar_t>(V_z),
            ptr<scalar_t>(b),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v) : nullptr,
            ptr<scalar_t>(workspace));
    }));

    return {h, output, v};
}

std::vector<Tensor> pure_lowrank_elman_backward(
    Tensor U_h,         // [dim, rank]
    Tensor V_h,         // [rank, dim]
    Tensor U_x,         // [dim, rank]
    Tensor V_x,         // [rank, dim]
    Tensor U_z,         // [dim, rank]
    Tensor V_z,         // [rank, dim]
    Tensor x,           // [T, B, dim]
    Tensor h,           // [T+1, B, dim]
    Tensor v,           // [T, B, dim] pre-activation cache
    Tensor d_output) {  // [T, B, dim]

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    const auto rank = U_h.size(1);

    CHECK_INPUT(U_h);
    CHECK_INPUT(V_h);
    CHECK_INPUT(U_x);
    CHECK_INPUT(V_x);
    CHECK_INPUT(U_z);
    CHECK_INPUT(V_z);
    CHECK_INPUT(x);
    CHECK_INPUT(h);
    CHECK_INPUT(v);
    CHECK_INPUT(d_output);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::empty_like(x);
    Tensor dU_h = torch::zeros({dim, rank}, options);
    Tensor dV_h = torch::zeros({rank, dim}, options);
    Tensor dU_x = torch::zeros({dim, rank}, options);
    Tensor dV_x = torch::zeros({rank, dim}, options);
    Tensor dU_z = torch::zeros({dim, rank}, options);
    Tensor dV_z = torch::zeros({rank, dim}, options);
    Tensor db = torch::zeros({dim}, options);

    // Workspace: [4*BD + 6*BR + dim]
    const int64_t BD = batch_size * dim;
    const int64_t BR = batch_size * rank;
    const int64_t float_dim = (dim * sizeof(float) + sizeof(float) - 1) / sizeof(float);
    Tensor workspace = torch::empty({4 * BD + 6 * BR + float_dim * 2}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "pure_lowrank_elman_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        PureLowRankElmanBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, rank,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(U_h),
            ptr<scalar_t>(V_h),
            ptr<scalar_t>(U_x),
            ptr<scalar_t>(V_x),
            ptr<scalar_t>(U_z),
            ptr<scalar_t>(V_z),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dU_h),
            ptr<scalar_t>(dV_h),
            ptr<scalar_t>(dU_x),
            ptr<scalar_t>(dV_x),
            ptr<scalar_t>(dU_z),
            ptr<scalar_t>(dV_z),
            ptr<scalar_t>(db),
            ptr<scalar_t>(workspace));
    }));

    return {dx, dU_h, dV_h, dU_x, dV_x, dU_z, dV_z, db};
}

// =============================================================================
// E5 Fused: Pure Low-Rank Elman with optimized kernel fusion
// Same API as E5, but uses fused tanh+gate kernel (25% fewer kernel launches)
// =============================================================================

std::vector<Tensor> pure_lowrank_elman_forward_fused(
    bool training,
    Tensor x,           // [T, B, dim]
    Tensor h0,          // [B, dim]
    Tensor U_h,         // [dim, rank]
    Tensor V_h,         // [rank, dim]
    Tensor U_x,         // [dim, rank]
    Tensor V_x,         // [rank, dim]
    Tensor U_z,         // [dim, rank]
    Tensor V_z,         // [rank, dim]
    Tensor b) {         // [dim]

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    const auto rank = U_h.size(1);

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(U_h);
    CHECK_INPUT(V_h);
    CHECK_INPUT(U_x);
    CHECK_INPUT(V_x);
    CHECK_INPUT(U_z);
    CHECK_INPUT(V_z);
    CHECK_INPUT(b);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, dim}, options)
                        : torch::empty({0}, options);

    // Workspace: [2*T*BR + 2*T*BD + BR + BD]
    const int64_t BD = batch_size * dim;
    const int64_t BR = batch_size * rank;
    Tensor workspace = torch::empty({2 * time_steps * BR + 2 * time_steps * BD + BR + BD}, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "pure_lowrank_elman_forward_fused", ([&] {
        using namespace hasty::v0::elman_ladder;
        PureLowRankElmanForwardFused<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, rank,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(U_h),
            ptr<scalar_t>(V_h),
            ptr<scalar_t>(U_x),
            ptr<scalar_t>(V_x),
            ptr<scalar_t>(U_z),
            ptr<scalar_t>(V_z),
            ptr<scalar_t>(b),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v) : nullptr,
            ptr<scalar_t>(workspace));
    }));

    return {h, output, v};
}

std::vector<Tensor> pure_lowrank_elman_backward_fused(
    Tensor U_h,         // [dim, rank]
    Tensor V_h,         // [rank, dim]
    Tensor U_x,         // [dim, rank]
    Tensor V_x,         // [rank, dim]
    Tensor U_z,         // [dim, rank]
    Tensor V_z,         // [rank, dim]
    Tensor x,           // [T, B, dim]
    Tensor h,           // [T+1, B, dim]
    Tensor v,           // [T, B, dim] pre-activation cache
    Tensor d_output) {  // [T, B, dim]

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    const auto rank = U_h.size(1);

    CHECK_INPUT(U_h);
    CHECK_INPUT(V_h);
    CHECK_INPUT(U_x);
    CHECK_INPUT(V_x);
    CHECK_INPUT(U_z);
    CHECK_INPUT(V_z);
    CHECK_INPUT(x);
    CHECK_INPUT(h);
    CHECK_INPUT(v);
    CHECK_INPUT(d_output);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::empty_like(x);
    Tensor dU_h = torch::zeros({dim, rank}, options);
    Tensor dV_h = torch::zeros({rank, dim}, options);
    Tensor dU_x = torch::zeros({dim, rank}, options);
    Tensor dV_x = torch::zeros({rank, dim}, options);
    Tensor dU_z = torch::zeros({dim, rank}, options);
    Tensor dV_z = torch::zeros({rank, dim}, options);
    Tensor db = torch::zeros({dim}, options);

    // Workspace: [5*BD + 7*BR + dim]
    const int64_t BD = batch_size * dim;
    const int64_t BR = batch_size * rank;
    const int64_t float_dim = (dim * sizeof(float) + sizeof(float) - 1) / sizeof(float);
    Tensor workspace = torch::empty({5 * BD + 7 * BR + float_dim * 2}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "pure_lowrank_elman_backward_fused", ([&] {
        using namespace hasty::v0::elman_ladder;
        PureLowRankElmanBackwardFused<typename native_type<scalar_t>::T> backward(
            batch_size, dim, rank,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(U_h),
            ptr<scalar_t>(V_h),
            ptr<scalar_t>(U_x),
            ptr<scalar_t>(V_x),
            ptr<scalar_t>(U_z),
            ptr<scalar_t>(V_z),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dU_h),
            ptr<scalar_t>(dV_h),
            ptr<scalar_t>(dU_x),
            ptr<scalar_t>(dV_x),
            ptr<scalar_t>(dU_z),
            ptr<scalar_t>(dV_z),
            ptr<scalar_t>(db),
            ptr<scalar_t>(workspace));
    }));

    return {dx, dU_h, dV_h, dU_x, dV_x, dU_z, dV_z, db};
}

// =============================================================================
// E5 B2B: Pure Low-Rank Elman with CUTLASS B2B GEMM fusion
// Fuses V_h @ h and U_h @ result into single kernel (keeps intermediate in smem)
// Requires rank = 64, 128, or 256 (ThreadblockShape constraint)
// =============================================================================

std::vector<Tensor> b2b_lowrank_elman_forward(
    bool training,
    Tensor x,           // [T, B, dim]
    Tensor h0,          // [B, dim]
    Tensor U_h,         // [dim, rank]
    Tensor V_h,         // [rank, dim]
    Tensor U_x,         // [dim, rank]
    Tensor V_x,         // [rank, dim]
    Tensor U_z,         // [dim, rank]
    Tensor V_z,         // [rank, dim]
    Tensor b) {         // [dim]

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    const auto rank = U_h.size(1);

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(U_h);
    CHECK_INPUT(V_h);
    CHECK_INPUT(U_x);
    CHECK_INPUT(V_x);
    CHECK_INPUT(U_z);
    CHECK_INPUT(V_z);
    CHECK_INPUT(b);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, dim}, options)
                        : torch::empty({0}, options);

    // Workspace: [2*T*BR + 2*T*BD + 2*BR + 2*BD]
    const int64_t BD = batch_size * dim;
    const int64_t BR = batch_size * rank;
    Tensor workspace = torch::empty({2 * time_steps * BR + 2 * time_steps * BD + 2 * BR + 2 * BD}, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "b2b_lowrank_elman_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        B2bLowRankElmanForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, rank,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(U_h),
            ptr<scalar_t>(V_h),
            ptr<scalar_t>(U_x),
            ptr<scalar_t>(V_x),
            ptr<scalar_t>(U_z),
            ptr<scalar_t>(V_z),
            ptr<scalar_t>(b),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v) : nullptr,
            ptr<scalar_t>(workspace));
    }));

    return {h, output, v};
}

std::vector<Tensor> b2b_lowrank_elman_backward(
    Tensor U_h,         // [dim, rank]
    Tensor V_h,         // [rank, dim]
    Tensor U_x,         // [dim, rank]
    Tensor V_x,         // [rank, dim]
    Tensor U_z,         // [dim, rank]
    Tensor V_z,         // [rank, dim]
    Tensor x,           // [T, B, dim]
    Tensor h,           // [T+1, B, dim]
    Tensor v,           // [T, B, dim] pre-activation cache
    Tensor d_output) {  // [T, B, dim]

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    const auto rank = U_h.size(1);

    CHECK_INPUT(U_h);
    CHECK_INPUT(V_h);
    CHECK_INPUT(U_x);
    CHECK_INPUT(V_x);
    CHECK_INPUT(U_z);
    CHECK_INPUT(V_z);
    CHECK_INPUT(x);
    CHECK_INPUT(h);
    CHECK_INPUT(v);
    CHECK_INPUT(d_output);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::empty_like(x);
    Tensor dU_h = torch::zeros({dim, rank}, options);
    Tensor dV_h = torch::zeros({rank, dim}, options);
    Tensor dU_x = torch::zeros({dim, rank}, options);
    Tensor dV_x = torch::zeros({rank, dim}, options);
    Tensor dU_z = torch::zeros({dim, rank}, options);
    Tensor dV_z = torch::zeros({rank, dim}, options);
    Tensor db = torch::zeros({dim}, options);

    // Workspace: [5*BD + 7*BR + dim]
    const int64_t BD = batch_size * dim;
    const int64_t BR = batch_size * rank;
    const int64_t float_dim = (dim * sizeof(float) + sizeof(float) - 1) / sizeof(float);
    Tensor workspace = torch::empty({5 * BD + 7 * BR + float_dim * 2}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "b2b_lowrank_elman_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        B2bLowRankElmanBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, rank,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(U_h),
            ptr<scalar_t>(V_h),
            ptr<scalar_t>(U_x),
            ptr<scalar_t>(V_x),
            ptr<scalar_t>(U_z),
            ptr<scalar_t>(V_z),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dU_h),
            ptr<scalar_t>(dV_h),
            ptr<scalar_t>(dU_x),
            ptr<scalar_t>(dV_x),
            ptr<scalar_t>(dU_z),
            ptr<scalar_t>(dV_z),
            ptr<scalar_t>(db),
            ptr<scalar_t>(workspace));
    }));

    return {dx, dU_h, dV_h, dU_x, dV_x, dU_z, dV_z, db};
}

// =============================================================================
// E6: Diagonal Elman (per-channel scalar recurrence + low-rank mixing)
// =============================================================================

std::vector<Tensor> diagonal_elman_forward(
    bool training,
    Tensor x,           // [T, B, dim]
    Tensor h0,          // [B, dim]
    Tensor gate_logit,  // [dim] raw gate logits
    Tensor U,           // [dim, rank]
    Tensor V) {         // [rank, dim]

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    const auto rank = U.size(1);

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(gate_logit);
    CHECK_INPUT(U);
    CHECK_INPUT(V);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);

    // Workspace: [BR + BD]
    const int64_t BD = batch_size * dim;
    const int64_t BR = batch_size * rank;
    Tensor workspace = torch::empty({BR + BD}, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "diagonal_elman_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        DiagonalElmanForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, rank,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(gate_logit),
            ptr<scalar_t>(U),
            ptr<scalar_t>(V),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(output),
            ptr<scalar_t>(workspace));
    }));

    return {h, output};
}

std::vector<Tensor> diagonal_elman_backward(
    Tensor gate_logit,  // [dim]
    Tensor U,           // [dim, rank]
    Tensor V,           // [rank, dim]
    Tensor x,           // [T, B, dim]
    Tensor h,           // [T+1, B, dim]
    Tensor d_output) {  // [T, B, dim]

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    const auto rank = U.size(1);

    CHECK_INPUT(gate_logit);
    CHECK_INPUT(U);
    CHECK_INPUT(V);
    CHECK_INPUT(x);
    CHECK_INPUT(h);
    CHECK_INPUT(d_output);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::empty_like(x);
    Tensor d_gate_logit = torch::zeros({dim}, options);
    Tensor dU = torch::zeros({dim, rank}, options);
    Tensor dV = torch::zeros({rank, dim}, options);

    // Workspace: [5*BD + 2*BR + dim]
    const int64_t BD = batch_size * dim;
    const int64_t BR = batch_size * rank;
    const int64_t float_dim = (dim * sizeof(float) + sizeof(float) - 1) / sizeof(float);
    Tensor workspace = torch::empty({5 * BD + 2 * BR + float_dim * 2}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "diagonal_elman_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        DiagonalElmanBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, rank,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(gate_logit),
            ptr<scalar_t>(U),
            ptr<scalar_t>(V),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(d_gate_logit),
            ptr<scalar_t>(dU),
            ptr<scalar_t>(dV),
            ptr<scalar_t>(workspace));
    }));

    return {dx, d_gate_logit, dU, dV};
}

// =============================================================================
// E6 Circulant: Circulant FFT Elman (O(n log n) via FFT)
// =============================================================================

std::vector<Tensor> circulant_elman_forward(
    bool training,
    Tensor x,           // [T, B, dim]
    Tensor h0,          // [B, dim]
    Tensor c_h,         // [dim] circulant vector for hidden
    Tensor c_x,         // [dim] circulant vector for input
    Tensor W_gate,      // [dim, dim] gate projection
    Tensor b,           // [dim]
    Tensor b_gate) {    // [dim] gate bias

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(c_h);
    CHECK_INPUT(c_x);
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

    // Workspace for FFT (all float32 for cuFFT compatibility):
    // [fft_c_h: dim complex] [fft_c_x: dim complex] [c_h_complex: dim complex] [c_x_complex: dim complex]
    // [h_complex: BD complex] [x_complex: BD complex] [result_complex: BD complex]
    // Total: 4*dim + 3*BD complex = (4*dim + 3*BD) * 2 floats
    const int64_t BD = batch_size * dim;
    const int64_t fft_workspace_floats = (4 * dim + 3 * BD) * 2;
    Tensor fft_workspace = torch::empty({fft_workspace_floats}, torch::dtype(torch::kFloat32).device(options.device()));

    // Separate workspace for gate_proj (in model dtype)
    Tensor gate_proj = torch::empty({time_steps, batch_size, dim}, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "circulant_elman_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        CirculantElmanForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(c_h),
            ptr<scalar_t>(c_x),
            ptr<scalar_t>(W_gate),
            ptr<scalar_t>(b),
            ptr<scalar_t>(b_gate),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v) : nullptr,
            training ? ptr<scalar_t>(gate_cache) : nullptr,
            fft_workspace.data_ptr<float>(),
            ptr<scalar_t>(gate_proj));
    }));

    return {h, output, v, gate_cache};
}

std::vector<Tensor> circulant_elman_backward(
    Tensor c_h,
    Tensor c_x,
    Tensor W_gate,
    Tensor x,
    Tensor h,
    Tensor v,
    Tensor gate_cache,
    Tensor d_output) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(c_h);
    CHECK_INPUT(c_x);
    CHECK_INPUT(W_gate);
    CHECK_INPUT(x);
    CHECK_INPUT(h);
    CHECK_INPUT(v);
    CHECK_INPUT(gate_cache);
    CHECK_INPUT(d_output);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::empty_like(x);
    Tensor d_c_h = torch::zeros({dim}, options);
    Tensor d_c_x = torch::zeros({dim}, options);
    Tensor dW_gate = torch::zeros({dim, dim}, options);
    Tensor db = torch::zeros({dim}, options);
    Tensor db_gate = torch::zeros({dim}, options);

    const int64_t BD = batch_size * dim;
    // FFT workspace (float32): 8 complex arrays of dim + 4 complex arrays of BD + 2 float arrays of dim
    // Each complex = 2 floats, so: (8*dim + 4*BD)*2 + 2*dim = 18*dim + 8*BD
    const int64_t fft_workspace_floats = 18 * dim + 8 * BD;
    Tensor fft_workspace = torch::empty({fft_workspace_floats}, torch::dtype(torch::kFloat32).device(options.device()));
    // Model dtype workspace: dh, dh_recurrent, dv_all, d_gate_proj_all
    // Size: (2 + 2*T) * BD
    Tensor work_T = torch::empty({(2 + 2 * time_steps) * BD}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "circulant_elman_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        CirculantElmanBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(c_h),
            ptr<scalar_t>(c_x),
            ptr<scalar_t>(W_gate),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(gate_cache),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(d_c_h),
            ptr<scalar_t>(d_c_x),
            ptr<scalar_t>(dW_gate),
            ptr<scalar_t>(db),
            ptr<scalar_t>(db_gate),
            fft_workspace.data_ptr<float>(),
            ptr<scalar_t>(work_T));
    }));

    return {dx, d_c_h, d_c_x, dW_gate, db, db_gate};
}

// =============================================================================
// E7 Monarch: Monarch Elman (O(n*sqrt(n)) via block-diagonal matrices)
// =============================================================================

std::vector<Tensor> monarch_elman_forward(
    bool training,
    Tensor x,           // [T, B, dim]
    Tensor h0,          // [B, dim]
    Tensor B1_h,        // [m, m, m] monarch blocks for hidden
    Tensor B2_h,        // [m, m, m]
    Tensor B1_x,        // [m, m, m] monarch blocks for input
    Tensor B2_x,        // [m, m, m]
    Tensor W_gate,      // [dim, dim] gate projection
    Tensor b,           // [dim]
    Tensor b_gate) {    // [dim]

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    const auto m = B1_h.size(0);  // sqrt(dim)

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(B1_h);
    CHECK_INPUT(B2_h);
    CHECK_INPUT(B1_x);
    CHECK_INPUT(B2_x);
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

    // Workspace: [2*T*BD + 3*BD] for Mx, gate_proj, tmp1, tmp2, Mh
    const int64_t BD = batch_size * dim;
    Tensor workspace = torch::empty({2 * time_steps * BD + 3 * BD}, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "monarch_elman_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        MonarchElmanForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, m,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(B1_h),
            ptr<scalar_t>(B2_h),
            ptr<scalar_t>(B1_x),
            ptr<scalar_t>(B2_x),
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

std::vector<Tensor> monarch_elman_backward(
    Tensor B1_h,
    Tensor B2_h,
    Tensor B1_x,
    Tensor B2_x,
    Tensor W_gate,
    Tensor x,
    Tensor h,
    Tensor v,
    Tensor gate_cache,
    Tensor d_output) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    const auto m = B1_h.size(0);

    CHECK_INPUT(B1_h);
    CHECK_INPUT(B2_h);
    CHECK_INPUT(B1_x);
    CHECK_INPUT(B2_x);
    CHECK_INPUT(W_gate);
    CHECK_INPUT(x);
    CHECK_INPUT(h);
    CHECK_INPUT(v);
    CHECK_INPUT(gate_cache);
    CHECK_INPUT(d_output);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::empty_like(x);
    Tensor dB1_h = torch::zeros({m, m, m}, options);
    Tensor dB2_h = torch::zeros({m, m, m}, options);
    Tensor dB1_x = torch::zeros({m, m, m}, options);
    Tensor dB2_x = torch::zeros({m, m, m}, options);
    Tensor dW_gate = torch::zeros({dim, dim}, options);
    Tensor db = torch::zeros({dim}, options);
    Tensor db_gate = torch::zeros({dim}, options);

    const int64_t BD = batch_size * dim;
    const int64_t float_in_T = (2 * dim * sizeof(float) + sizeof(float) - 1) / sizeof(float);
    Tensor workspace = torch::empty({2 * time_steps * BD + 5 * BD + float_in_T * 2}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "monarch_elman_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        MonarchElmanBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, m,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(B1_h),
            ptr<scalar_t>(B2_h),
            ptr<scalar_t>(B1_x),
            ptr<scalar_t>(B2_x),
            ptr<scalar_t>(W_gate),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(gate_cache),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dB1_h),
            ptr<scalar_t>(dB2_h),
            ptr<scalar_t>(dB1_x),
            ptr<scalar_t>(dB2_x),
            ptr<scalar_t>(dW_gate),
            ptr<scalar_t>(db),
            ptr<scalar_t>(db_gate),
            ptr<scalar_t>(workspace));
    }));

    return {dx, dB1_h, dB2_h, dB1_x, dB2_x, dW_gate, db, db_gate};
}

// =============================================================================
// E8 Scaled Low-Rank: Learn to sparsify via importance scaling
// =============================================================================

std::vector<Tensor> scaled_lowrank_elman_forward(
    bool training,
    Tensor x,           // [T, B, dim]
    Tensor z,           // [T, B, dim] gate input
    Tensor h0,          // [B, dim]
    Tensor U_h,         // [dim, rank]
    Tensor V_h,         // [rank, dim]
    Tensor s_h,         // [rank] scale for hidden
    Tensor U_x,         // [dim, rank]
    Tensor V_x,         // [rank, dim]
    Tensor s_x,         // [rank] scale for input
    Tensor b) {         // [dim]

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    const auto rank = U_h.size(1);

    CHECK_INPUT(x);
    CHECK_INPUT(z);
    CHECK_INPUT(h0);
    CHECK_INPUT(U_h);
    CHECK_INPUT(V_h);
    CHECK_INPUT(s_h);
    CHECK_INPUT(U_x);
    CHECK_INPUT(V_x);
    CHECK_INPUT(s_x);
    CHECK_INPUT(b);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, dim}, options)
                        : torch::empty({0}, options);

    // Workspace: [T*BR + 4*BR + 2*BD]
    const int64_t BD = batch_size * dim;
    const int64_t BR = batch_size * rank;
    Tensor workspace = torch::empty({time_steps * BR + 4 * BR + 2 * BD}, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "scaled_lowrank_elman_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        ScaledLowRankElmanForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, rank,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(U_h),
            ptr<scalar_t>(V_h),
            ptr<scalar_t>(s_h),
            ptr<scalar_t>(U_x),
            ptr<scalar_t>(V_x),
            ptr<scalar_t>(s_x),
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

std::vector<Tensor> scaled_lowrank_elman_backward(
    Tensor U_h,
    Tensor V_h,
    Tensor s_h,
    Tensor U_x,
    Tensor V_x,
    Tensor s_x,
    Tensor x,
    Tensor z,
    Tensor h,
    Tensor v,
    Tensor d_output) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    const auto rank = U_h.size(1);

    CHECK_INPUT(U_h);
    CHECK_INPUT(V_h);
    CHECK_INPUT(s_h);
    CHECK_INPUT(U_x);
    CHECK_INPUT(V_x);
    CHECK_INPUT(s_x);
    CHECK_INPUT(x);
    CHECK_INPUT(z);
    CHECK_INPUT(h);
    CHECK_INPUT(v);
    CHECK_INPUT(d_output);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::empty_like(x);
    Tensor dz = torch::empty_like(z);
    Tensor dU_h = torch::zeros({dim, rank}, options);
    Tensor dV_h = torch::zeros({rank, dim}, options);
    Tensor ds_h = torch::zeros({rank}, options);
    Tensor dU_x = torch::zeros({dim, rank}, options);
    Tensor dV_x = torch::zeros({rank, dim}, options);
    Tensor ds_x = torch::zeros({rank}, options);
    Tensor db = torch::zeros({dim}, options);

    // Workspace: [T*BD + 2*T*BR + 4*BD + 6*BR + dim + 2*rank (in floats)]
    const int64_t BD = batch_size * dim;
    const int64_t BR = batch_size * rank;
    const int64_t float_space = (dim + 2 * rank) * sizeof(float);
    const int64_t T_space = (float_space + sizeof(float) - 1) / sizeof(float);
    Tensor workspace = torch::empty({time_steps * BD + 2 * time_steps * BR + 4 * BD + 6 * BR + T_space}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "scaled_lowrank_elman_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        ScaledLowRankElmanBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, rank,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(U_h),
            ptr<scalar_t>(V_h),
            ptr<scalar_t>(s_h),
            ptr<scalar_t>(U_x),
            ptr<scalar_t>(V_x),
            ptr<scalar_t>(s_x),
            ptr<scalar_t>(x),
            ptr<scalar_t>(z),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dz),
            ptr<scalar_t>(dU_h),
            ptr<scalar_t>(dV_h),
            ptr<scalar_t>(ds_h),
            ptr<scalar_t>(dU_x),
            ptr<scalar_t>(dV_x),
            ptr<scalar_t>(ds_x),
            ptr<scalar_t>(db),
            ptr<scalar_t>(workspace));
    }));

    return {dx, dz, dU_h, dV_h, ds_h, dU_x, dV_x, ds_x, db};
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
    m.def("lowrank_elman_forward", &lowrank_elman_forward,
          "E4: Low-Rank Elman forward");
    m.def("lowrank_elman_backward", &lowrank_elman_backward,
          "E4: Low-Rank Elman backward");
    m.def("pure_lowrank_elman_forward", &pure_lowrank_elman_forward,
          "E5: Pure Low-Rank Elman forward");
    m.def("pure_lowrank_elman_backward", &pure_lowrank_elman_backward,
          "E5: Pure Low-Rank Elman backward");
    m.def("pure_lowrank_elman_forward_fused", &pure_lowrank_elman_forward_fused,
          "E5 Fused: Pure Low-Rank Elman forward (optimized)");
    m.def("pure_lowrank_elman_backward_fused", &pure_lowrank_elman_backward_fused,
          "E5 Fused: Pure Low-Rank Elman backward (optimized)");
    m.def("b2b_lowrank_elman_forward", &b2b_lowrank_elman_forward,
          "E5 B2B: Pure Low-Rank Elman forward (CUTLASS B2B fusion, requires rank=64/128/256)");
    m.def("b2b_lowrank_elman_backward", &b2b_lowrank_elman_backward,
          "E5 B2B: Pure Low-Rank Elman backward");
    m.def("diagonal_elman_forward", &diagonal_elman_forward,
          "E6 Diag: Diagonal Elman forward");
    m.def("diagonal_elman_backward", &diagonal_elman_backward,
          "E6 Diag: Diagonal Elman backward");
    m.def("circulant_elman_forward", &circulant_elman_forward,
          "E6: Circulant FFT Elman forward (O(n log n))");
    m.def("circulant_elman_backward", &circulant_elman_backward,
          "E6: Circulant FFT Elman backward");
    m.def("monarch_elman_forward", &monarch_elman_forward,
          "E7: Monarch Elman forward (O(n*sqrt(n)))");
    m.def("monarch_elman_backward", &monarch_elman_backward,
          "E7: Monarch Elman backward");
    m.def("scaled_lowrank_elman_forward", &scaled_lowrank_elman_forward,
          "E8: Scaled Low-Rank Elman forward (learn to sparsify)");
    m.def("scaled_lowrank_elman_backward", &scaled_lowrank_elman_backward,
          "E8: Scaled Low-Rank Elman backward");
}
