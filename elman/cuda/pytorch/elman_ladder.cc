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
// Softsign Elman - E1 variant using softsign instead of tanh
// softsign(x) = x / (1 + |x|)
// - Cheaper than tanh (no exp)
// - Bounded (-1, 1) like tanh
// - Smoother gradients: derivative = 1/(1+|x|)^2
// =============================================================================

std::vector<Tensor> softsign_elman_forward(
    bool training,
    Tensor x,           // [T, B, dim] pre-activated input
    Tensor z,           // [T, B, dim] gate input (pre silu)
    Tensor h0,          // [B, dim]
    Tensor W_x,         // [dim, dim]
    Tensor W_h,         // [dim, dim]
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
        x.scalar_type(), "softsign_elman_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        SoftsignElmanForward<typename native_type<scalar_t>::T> forward(
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

std::vector<Tensor> softsign_elman_backward(
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
        x.scalar_type(), "softsign_elman_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        SoftsignElmanBackward<typename native_type<scalar_t>::T> backward(
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
// E16: Diagonal State-Expanded Elman
// h' = tanh(A ⊙ h + B @ x)  where A is diagonal
// y = C @ h * silu(z)
// =============================================================================

std::vector<Tensor> diagonal_state_elman_forward(
    bool training,
    Tensor x,           // [T, B, d_model]
    Tensor z,           // [T, B, d_model] gate input
    Tensor h0,          // [B, d_state]
    Tensor B_proj,      // [d_model, d_state]
    Tensor C_proj,      // [d_state, d_model]
    Tensor A) {         // [d_state] diagonal

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto d_model = x.size(2);
    const auto d_state = A.size(0);

    CHECK_INPUT(x);
    CHECK_INPUT(z);
    CHECK_INPUT(h0);
    CHECK_INPUT(B_proj);
    CHECK_INPUT(C_proj);
    CHECK_INPUT(A);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor h = torch::empty({time_steps + 1, batch_size, d_state}, options);
    Tensor output = torch::empty({time_steps, batch_size, d_model}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, d_state}, options)
                        : torch::empty({0}, options);

    // Workspace: [T*B*d_state + B*d_model]
    const int64_t BS = batch_size * d_state;
    const int64_t BD = batch_size * d_model;
    Tensor workspace = torch::empty({time_steps * BS + BD}, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "diagonal_state_elman_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        DiagonalStateElmanForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, d_model, d_state,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(B_proj),
            ptr<scalar_t>(C_proj),
            ptr<scalar_t>(A),
            ptr<scalar_t>(x),
            ptr<scalar_t>(z),
            ptr<scalar_t>(h),
            ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v) : nullptr,
            ptr<scalar_t>(workspace));
    }));

    return {h, output, v};
}

std::vector<Tensor> diagonal_state_elman_backward(
    Tensor B_proj,
    Tensor C_proj,
    Tensor A,
    Tensor x,
    Tensor z,
    Tensor h,
    Tensor v,
    Tensor d_output) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto d_model = x.size(2);
    const auto d_state = A.size(0);

    CHECK_INPUT(B_proj);
    CHECK_INPUT(C_proj);
    CHECK_INPUT(A);
    CHECK_INPUT(x);
    CHECK_INPUT(z);
    CHECK_INPUT(h);
    CHECK_INPUT(v);
    CHECK_INPUT(d_output);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::empty_like(x);
    Tensor dz = torch::empty_like(z);
    Tensor dB = torch::zeros({d_model, d_state}, options);
    Tensor dC = torch::zeros({d_state, d_model}, options);
    Tensor dA = torch::zeros({d_state}, options);

    // Workspace: [T*B*d_state + 2*B*d_state + B*d_model + d_state*4]
    const int64_t BS = batch_size * d_state;
    const int64_t BD = batch_size * d_model;
    const int64_t workspace_size = time_steps * BS + 2 * BS + BD + d_state * 4;
    Tensor workspace = torch::empty({workspace_size}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "diagonal_state_elman_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        DiagonalStateElmanBackward<typename native_type<scalar_t>::T> backward(
            batch_size, d_model, d_state,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(B_proj),
            ptr<scalar_t>(C_proj),
            ptr<scalar_t>(A),
            ptr<scalar_t>(x),
            ptr<scalar_t>(z),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dz),
            ptr<scalar_t>(dB),
            ptr<scalar_t>(dC),
            ptr<scalar_t>(dA),
            ptr<scalar_t>(workspace));
    }));

    return {dx, dz, dB, dC, dA};
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

// =============================================================================
// E9: Hybrid Elman (small dense core + large diagonal memory)
// =============================================================================

std::vector<Tensor> hybrid_elman_forward(
    bool training,
    Tensor x_core,      // [T, B, core_dim] pre-activated
    Tensor z_core,      // [T, B, core_dim] gate
    Tensor x_mem,       // [T, B, mem_dim]
    Tensor z_mem,       // [T, B, mem_dim] gate
    Tensor h0_core,     // [B, core_dim]
    Tensor h0_mem,      // [B, mem_dim]
    Tensor W_x_core,    // [core_dim, core_dim]
    Tensor W_h,         // [core_dim, core_dim]
    Tensor b_core,      // [core_dim]
    Tensor a_mem) {     // [mem_dim] decay logits

    const auto time_steps = x_core.size(0);
    const auto batch_size = x_core.size(1);
    const auto core_dim = x_core.size(2);
    const auto mem_dim = x_mem.size(2);

    CHECK_INPUT(x_core);
    CHECK_INPUT(z_core);
    CHECK_INPUT(x_mem);
    CHECK_INPUT(z_mem);
    CHECK_INPUT(h0_core);
    CHECK_INPUT(h0_mem);
    CHECK_INPUT(W_x_core);
    CHECK_INPUT(W_h);
    CHECK_INPUT(b_core);
    CHECK_INPUT(a_mem);

    const auto options = x_core.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor h_core = torch::empty({time_steps + 1, batch_size, core_dim}, options);
    Tensor h_mem = torch::empty({time_steps + 1, batch_size, mem_dim}, options);
    Tensor out_core = torch::empty({time_steps, batch_size, core_dim}, options);
    Tensor out_mem = torch::empty({time_steps, batch_size, mem_dim}, options);
    Tensor v_core = training ? torch::empty({time_steps, batch_size, core_dim}, options)
                              : torch::empty({0}, options);

    // Workspace: [T*B*core_dim + B*core_dim]
    const int64_t B_core = batch_size * core_dim;
    Tensor workspace = torch::empty({time_steps * B_core + B_core}, options);

    h_core[0] = h0_core;
    h_mem[0] = h0_mem;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x_core.scalar_type(), "hybrid_elman_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        HybridElmanForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, core_dim, mem_dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W_x_core),
            ptr<scalar_t>(W_h),
            ptr<scalar_t>(b_core),
            ptr<scalar_t>(a_mem),
            ptr<scalar_t>(x_core),
            ptr<scalar_t>(z_core),
            ptr<scalar_t>(x_mem),
            ptr<scalar_t>(z_mem),
            ptr<scalar_t>(h_core),
            ptr<scalar_t>(h_mem),
            ptr<scalar_t>(out_core),
            ptr<scalar_t>(out_mem),
            training ? ptr<scalar_t>(v_core) : nullptr,
            ptr<scalar_t>(workspace));
    }));

    return {h_core, h_mem, out_core, out_mem, v_core};
}

std::vector<Tensor> hybrid_elman_backward(
    Tensor W_x_core,
    Tensor W_h,
    Tensor a_mem,
    Tensor x_core,
    Tensor z_core,
    Tensor x_mem,
    Tensor z_mem,
    Tensor h_core,
    Tensor h_mem,
    Tensor v_core,
    Tensor d_out_core,
    Tensor d_out_mem) {

    const auto time_steps = x_core.size(0);
    const auto batch_size = x_core.size(1);
    const auto core_dim = x_core.size(2);
    const auto mem_dim = x_mem.size(2);

    CHECK_INPUT(W_x_core);
    CHECK_INPUT(W_h);
    CHECK_INPUT(a_mem);
    CHECK_INPUT(x_core);
    CHECK_INPUT(z_core);
    CHECK_INPUT(x_mem);
    CHECK_INPUT(z_mem);
    CHECK_INPUT(h_core);
    CHECK_INPUT(h_mem);
    CHECK_INPUT(v_core);
    CHECK_INPUT(d_out_core);
    CHECK_INPUT(d_out_mem);

    const auto options = x_core.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx_core = torch::empty_like(x_core);
    Tensor dz_core = torch::empty_like(z_core);
    Tensor dx_mem = torch::empty_like(x_mem);
    Tensor dz_mem = torch::empty_like(z_mem);
    Tensor dW_x_core = torch::zeros({core_dim, core_dim}, options);
    Tensor dW_h = torch::zeros({core_dim, core_dim}, options);
    Tensor db_core = torch::zeros({core_dim}, options);
    Tensor da_mem = torch::zeros({mem_dim}, options);

    // Workspace: [(T+2)*B*core + 2*B*mem + core + mem (floats)]
    const int64_t B_core = batch_size * core_dim;
    const int64_t B_mem = batch_size * mem_dim;
    const int64_t float_space = (core_dim + mem_dim) * sizeof(float);
    const int64_t T_space = (float_space + sizeof(float) - 1) / sizeof(float);
    Tensor workspace = torch::empty({(time_steps + 2) * B_core + 2 * B_mem + T_space}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x_core.scalar_type(), "hybrid_elman_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        HybridElmanBackward<typename native_type<scalar_t>::T> backward(
            batch_size, core_dim, mem_dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W_x_core),
            ptr<scalar_t>(W_h),
            ptr<scalar_t>(a_mem),
            ptr<scalar_t>(x_core),
            ptr<scalar_t>(z_core),
            ptr<scalar_t>(x_mem),
            ptr<scalar_t>(z_mem),
            ptr<scalar_t>(h_core),
            ptr<scalar_t>(h_mem),
            ptr<scalar_t>(v_core),
            ptr<scalar_t>(d_out_core),
            ptr<scalar_t>(d_out_mem),
            ptr<scalar_t>(dx_core),
            ptr<scalar_t>(dz_core),
            ptr<scalar_t>(dx_mem),
            ptr<scalar_t>(dz_mem),
            ptr<scalar_t>(dW_x_core),
            ptr<scalar_t>(dW_h),
            ptr<scalar_t>(db_core),
            ptr<scalar_t>(da_mem),
            ptr<scalar_t>(workspace));
    }));

    return {dx_core, dz_core, dx_mem, dz_mem, dW_x_core, dW_h, db_core, da_mem};
}

// =============================================================================
// E10: Multi-Scale EMA Elman (multiple EMA memory banks with learned decay)
// =============================================================================

std::vector<Tensor> multiscale_elman_forward(
    bool training,
    Tensor x,           // [T, B, dim] pre-activated input
    Tensor z,           // [T, B, (1+n_banks)*dim] gates
    Tensor h0,          // [B, dim]
    Tensor m0,          // [n_banks, B, dim]
    Tensor W_x,         // [dim, dim]
    Tensor W_h,         // [dim, dim]
    Tensor b,           // [dim]
    Tensor a) {         // [n_banks, dim] EMA decay logits

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    const auto n_banks = a.size(0);

    CHECK_INPUT(x);
    CHECK_INPUT(z);
    CHECK_INPUT(h0);
    CHECK_INPUT(m0);
    CHECK_INPUT(W_x);
    CHECK_INPUT(W_h);
    CHECK_INPUT(b);
    CHECK_INPUT(a);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor m = torch::empty({time_steps + 1, n_banks, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, dim}, options)
                        : torch::empty({0}, options);

    // Workspace: [tmp_Wx: T*BD] [tmp_Rh: BD]
    const int64_t BD = batch_size * dim;
    Tensor workspace = torch::empty({time_steps * BD + BD}, options);

    h[0] = h0;
    m[0] = m0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "multiscale_elman_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        MultiScaleElmanForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, n_banks,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(W_h),
            ptr<scalar_t>(b),
            ptr<scalar_t>(a),
            ptr<scalar_t>(x),
            ptr<scalar_t>(z),
            ptr<scalar_t>(h),
            ptr<scalar_t>(m),
            ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v) : nullptr,
            ptr<scalar_t>(workspace));
    }));

    return {h, m, output, v};
}

std::vector<Tensor> multiscale_elman_backward(
    Tensor W_x,
    Tensor W_h,
    Tensor a,
    Tensor x,
    Tensor z,
    Tensor h,
    Tensor m,
    Tensor v,
    Tensor d_output) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    const auto n_banks = a.size(0);

    CHECK_INPUT(W_x);
    CHECK_INPUT(W_h);
    CHECK_INPUT(a);
    CHECK_INPUT(x);
    CHECK_INPUT(z);
    CHECK_INPUT(h);
    CHECK_INPUT(m);
    CHECK_INPUT(v);
    CHECK_INPUT(d_output);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::empty_like(x);
    Tensor dz = torch::empty_like(z);
    Tensor dW_x = torch::zeros({dim, dim}, options);
    Tensor dW_h = torch::zeros({dim, dim}, options);
    Tensor db = torch::zeros({dim}, options);
    Tensor da = torch::zeros({n_banks, dim}, options);

    // Workspace: [dv_all: T*BD] [dh: BD] [dh_recurrent: BD]
    //            [dm: n_banks*BD] [dm_recurrent: n_banks*BD]
    //            [dh_ema_float: BD floats] [db_float: dim floats] [da_float: n_banks*dim floats]
    const int64_t BD = batch_size * dim;
    // Float buffers: BD + dim + n_banks*dim floats
    // Compute how many T elements are needed to hold all the float buffers
    // (assuming bfloat16/float16 is 2 bytes, float is 4 bytes)
    const int64_t float_elems = BD + dim + n_banks * dim;
    const int64_t float_bytes = float_elems * sizeof(float);
    // Round up to nearest T element
    const int64_t float_space_in_T = (float_bytes + sizeof(float) - 1) / sizeof(float);
    const int64_t workspace_size = (time_steps + 2) * BD + 2 * n_banks * BD + float_space_in_T * 2;
    Tensor workspace = torch::empty({workspace_size}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "multiscale_elman_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        MultiScaleElmanBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, n_banks,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(W_h),
            ptr<scalar_t>(a),
            ptr<scalar_t>(x),
            ptr<scalar_t>(z),
            ptr<scalar_t>(h),
            ptr<scalar_t>(m),
            ptr<scalar_t>(v),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dz),
            ptr<scalar_t>(dW_x),
            ptr<scalar_t>(dW_h),
            ptr<scalar_t>(db),
            ptr<scalar_t>(da),
            ptr<scalar_t>(workspace));
    }));

    return {dx, dz, dW_x, dW_h, db, da};
}

// =============================================================================
// E11: Selective Memory Elman (Mamba-inspired input-dependent memory)
// =============================================================================

std::vector<Tensor> selective_elman_forward(
    bool training,
    Tensor x,           // [T, B, dim] pre-activated input
    Tensor z,           // [T, B, 2*dim] gates
    Tensor h0,          // [B, dim]
    Tensor m0,          // [n_banks, B, dim]
    Tensor W_x,         // [dim, dim]
    Tensor W_h,         // [dim, dim]
    Tensor b,           // [dim]
    Tensor a,           // [n_banks, dim]
    Tensor W_a,         // [dim, n_banks]
    Tensor W_w) {       // [dim, n_banks]

    const int64_t time_steps = x.size(0);
    const int64_t batch_size = x.size(1);
    const int64_t dim = x.size(2);
    const int64_t n_banks = a.size(0);

    auto options = x.options();

    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor m = torch::empty({time_steps + 1, n_banks, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, dim}, options)
                        : torch::empty({0}, options);
    Tensor a_scale_cache = training ? torch::empty({time_steps, batch_size, n_banks}, options)
                                    : torch::empty({0}, options);
    Tensor read_weights_cache = training ? torch::empty({time_steps, batch_size, n_banks}, options)
                                          : torch::empty({0}, options);

    const int64_t BD = batch_size * dim;
    const int64_t BK = batch_size * n_banks;
    const int64_t workspace_size = time_steps * BD + BD + 2 * BK;
    Tensor workspace = torch::empty({workspace_size}, options);

    h[0] = h0;
    m[0] = m0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "selective_elman_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        SelectiveElmanForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, n_banks,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(W_h),
            ptr<scalar_t>(b),
            ptr<scalar_t>(a),
            ptr<scalar_t>(W_a),
            ptr<scalar_t>(W_w),
            ptr<scalar_t>(x),
            ptr<scalar_t>(z),
            ptr<scalar_t>(h),
            ptr<scalar_t>(m),
            ptr<scalar_t>(output),
            ptr<scalar_t>(v),
            ptr<scalar_t>(a_scale_cache),
            ptr<scalar_t>(read_weights_cache),
            ptr<scalar_t>(workspace));
    }));

    return {h, m, output, v, a_scale_cache, read_weights_cache};
}

std::vector<Tensor> selective_elman_backward(
    Tensor W_x,
    Tensor W_h,
    Tensor a,
    Tensor W_a,
    Tensor W_w,
    Tensor x,
    Tensor z,
    Tensor h,
    Tensor m,
    Tensor v,
    Tensor a_scale_cache,
    Tensor read_weights_cache,
    Tensor d_output) {

    const int64_t time_steps = x.size(0);
    const int64_t batch_size = x.size(1);
    const int64_t dim = x.size(2);
    const int64_t n_banks = a.size(0);

    auto options = x.options();

    Tensor dx = torch::empty_like(x);
    Tensor dz = torch::empty_like(z);
    Tensor dW_x = torch::zeros({dim, dim}, options);
    Tensor dW_h = torch::zeros({dim, dim}, options);
    Tensor db = torch::zeros({dim}, options);
    Tensor da = torch::zeros({n_banks, dim}, options);
    Tensor dW_a = torch::zeros({dim, n_banks}, options);
    Tensor dW_w = torch::zeros({dim, n_banks}, options);

    const int64_t BD = batch_size * dim;
    const int64_t BK = batch_size * n_banks;
    // Workspace: dv_all + dh + dh_rec + dm + dm_rec + d_read_logits + float buffers
    const int64_t float_bytes = BD * sizeof(float) + BK * sizeof(float) +
                                dim * sizeof(float) + n_banks * dim * sizeof(float) +
                                BK * sizeof(float);
    const int64_t float_space_in_T = (float_bytes + sizeof(float) - 1) / sizeof(float);
    const int64_t workspace_size = (time_steps + 2) * BD + 2 * n_banks * BD + BK + float_space_in_T * 2;
    Tensor workspace = torch::empty({workspace_size}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "selective_elman_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        SelectiveElmanBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, n_banks,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(W_h),
            ptr<scalar_t>(a),
            ptr<scalar_t>(W_a),
            ptr<scalar_t>(W_w),
            ptr<scalar_t>(x),
            ptr<scalar_t>(z),
            ptr<scalar_t>(h),
            ptr<scalar_t>(m),
            ptr<scalar_t>(v),
            ptr<scalar_t>(a_scale_cache),
            ptr<scalar_t>(read_weights_cache),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dz),
            ptr<scalar_t>(dW_x),
            ptr<scalar_t>(dW_h),
            ptr<scalar_t>(db),
            ptr<scalar_t>(da),
            ptr<scalar_t>(dW_a),
            ptr<scalar_t>(dW_w),
            ptr<scalar_t>(workspace));
    }));

    return {dx, dz, dW_x, dW_h, db, da, dW_a, dW_w};
}

// =============================================================================
// E12: Selective Gated Elman (hidden-state-dependent gating)
// h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b)
// g_t = W_g @ h_t
// output = h_t * sigmoid(z_t + g_t)
// =============================================================================

std::vector<Tensor> selective_gated_elman_forward(
    bool training,
    Tensor x,           // [T, B, dim] pre-activated input
    Tensor z,           // [T, B, dim] gate input
    Tensor h0,          // [B, dim]
    Tensor W_x,         // [dim, dim]
    Tensor W_h,         // [dim, dim]
    Tensor W_g,         // [dim, dim] gate projection (NEW)
    Tensor b) {         // [dim]

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(z);
    CHECK_INPUT(h0);
    CHECK_INPUT(W_x);
    CHECK_INPUT(W_h);
    CHECK_INPUT(W_g);
    CHECK_INPUT(b);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, dim}, options)
                        : torch::empty({0}, options);
    Tensor gate_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                 : torch::empty({0}, options);

    // Forward workspace: [tmp_Wx: T*BD] [tmp_Rh: BD] [tmp_Gh: BD]
    const int64_t BD = batch_size * dim;
    Tensor workspace = torch::empty({time_steps * BD + 2 * BD}, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "selective_gated_elman_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        SelectiveGatedElmanForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(W_h),
            ptr<scalar_t>(W_g),
            ptr<scalar_t>(b),
            ptr<scalar_t>(x),
            ptr<scalar_t>(z),
            ptr<scalar_t>(h),
            ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v) : nullptr,
            training ? ptr<scalar_t>(gate_cache) : nullptr,
            ptr<scalar_t>(workspace));
    }));

    return {h, output, v, gate_cache};
}

std::vector<Tensor> selective_gated_elman_backward(
    Tensor W_x,
    Tensor W_h,
    Tensor W_g,
    Tensor x,
    Tensor z,
    Tensor h,
    Tensor v,
    Tensor gate_cache,
    Tensor d_output) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(W_x);
    CHECK_INPUT(W_h);
    CHECK_INPUT(W_g);
    CHECK_INPUT(x);
    CHECK_INPUT(z);
    CHECK_INPUT(h);
    CHECK_INPUT(v);
    CHECK_INPUT(gate_cache);
    CHECK_INPUT(d_output);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::empty_like(x);
    Tensor dz = torch::empty_like(z);
    Tensor dW_x = torch::zeros({dim, dim}, options);
    Tensor dW_h = torch::zeros({dim, dim}, options);
    Tensor dW_g = torch::zeros({dim, dim}, options);
    Tensor db = torch::zeros({dim}, options);

    // Workspace layout: [dv_all: T*BD] [dh_gate: BD] [dh_recurrent: BD] [d_Gh: BD] [db_float: dim]
    const int64_t BD = batch_size * dim;
    const int64_t float_in_T = (dim * sizeof(float) + sizeof(float) - 1) / sizeof(float);
    const int64_t workspace_size = (time_steps + 3) * BD + float_in_T * 2;
    Tensor workspace = torch::empty({workspace_size}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "selective_gated_elman_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        SelectiveGatedElmanBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(W_h),
            ptr<scalar_t>(W_g),
            ptr<scalar_t>(x),
            ptr<scalar_t>(z),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(gate_cache),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dz),
            ptr<scalar_t>(dW_x),
            ptr<scalar_t>(dW_h),
            ptr<scalar_t>(dW_g),
            ptr<scalar_t>(db),
            ptr<scalar_t>(workspace));
    }));

    return {dx, dz, dW_x, dW_h, dW_g, db};
}

// =============================================================================
// E14: Matrix State Elman (trading weight capacity for state capacity)
// H ∈ ℝ^(d×k) matrix state instead of vector
// key = tanh(W_key @ x), value = W_val @ x
// decay = sigmoid(W_decay @ x), query = W_query @ x
// H_new = decay * H + key ⊗ value (outer product)
// output = (H_new @ query) * silu(z)
// =============================================================================

std::vector<Tensor> matrix_state_elman_forward(
    bool training,
    Tensor x,           // [T, B, d] pre-activated input
    Tensor z,           // [T, B, d] gate input
    Tensor H0,          // [B, d, k] initial state
    Tensor W_key,       // [d, d]
    Tensor b_key,       // [d]
    Tensor W_val,       // [d, k]
    Tensor b_val,       // [k]
    Tensor W_query,     // [d, k]
    Tensor b_query,     // [k]
    Tensor W_decay,     // [d, d]
    Tensor b_decay) {   // [d]

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto d = x.size(2);
    const auto k = W_val.size(1);

    CHECK_INPUT(x);
    CHECK_INPUT(z);
    CHECK_INPUT(H0);
    CHECK_INPUT(W_key);
    CHECK_INPUT(b_key);
    CHECK_INPUT(W_val);
    CHECK_INPUT(b_val);
    CHECK_INPUT(W_query);
    CHECK_INPUT(b_query);
    CHECK_INPUT(W_decay);
    CHECK_INPUT(b_decay);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    // Outputs
    Tensor H = torch::empty({time_steps + 1, batch_size, d, k}, options);
    Tensor output = torch::empty({time_steps, batch_size, d}, options);

    // Caches for backward
    Tensor key_cache = training ? torch::empty({time_steps, batch_size, d}, options)
                                : torch::empty({0}, options);
    Tensor value_cache = training ? torch::empty({time_steps, batch_size, k}, options)
                                  : torch::empty({0}, options);
    Tensor decay_cache = training ? torch::empty({time_steps, batch_size, d}, options)
                                  : torch::empty({0}, options);
    Tensor query_cache = training ? torch::empty({time_steps, batch_size, k}, options)
                                  : torch::empty({0}, options);

    // Workspace: [proj_key_all: T*BD] [proj_val_all: T*BK] [proj_query_all: T*BK] [proj_decay_all: T*BD]
    //            [key_tmp: BD] [value_tmp: BK] [query_tmp: BK] [decay_tmp: BD]
    const int64_t BD = batch_size * d;
    const int64_t BK = batch_size * k;
    const int64_t workspace_size = 2 * time_steps * BD + 2 * time_steps * BK + 2 * BD + 2 * BK;
    Tensor workspace = torch::empty({workspace_size}, options);

    H[0] = H0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "matrix_state_elman_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        MatrixStateElmanForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, d, k,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W_key),
            ptr<scalar_t>(b_key),
            ptr<scalar_t>(W_val),
            ptr<scalar_t>(b_val),
            ptr<scalar_t>(W_query),
            ptr<scalar_t>(b_query),
            ptr<scalar_t>(W_decay),
            ptr<scalar_t>(b_decay),
            ptr<scalar_t>(x),
            ptr<scalar_t>(z),
            ptr<scalar_t>(H),
            ptr<scalar_t>(output),
            training ? ptr<scalar_t>(key_cache) : nullptr,
            training ? ptr<scalar_t>(value_cache) : nullptr,
            training ? ptr<scalar_t>(decay_cache) : nullptr,
            training ? ptr<scalar_t>(query_cache) : nullptr,
            ptr<scalar_t>(workspace));
    }));

    return {H, output, key_cache, value_cache, decay_cache, query_cache};
}

std::vector<Tensor> matrix_state_elman_backward(
    Tensor W_key,
    Tensor b_key,
    Tensor W_val,
    Tensor b_val,
    Tensor W_query,
    Tensor b_query,
    Tensor W_decay,
    Tensor b_decay,
    Tensor x,
    Tensor z,
    Tensor H,
    Tensor key_cache,
    Tensor value_cache,
    Tensor decay_cache,
    Tensor query_cache,
    Tensor d_output) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto d = x.size(2);
    const auto k = W_val.size(1);

    CHECK_INPUT(W_key);
    CHECK_INPUT(b_key);
    CHECK_INPUT(W_val);
    CHECK_INPUT(b_val);
    CHECK_INPUT(W_query);
    CHECK_INPUT(b_query);
    CHECK_INPUT(W_decay);
    CHECK_INPUT(b_decay);
    CHECK_INPUT(x);
    CHECK_INPUT(z);
    CHECK_INPUT(H);
    CHECK_INPUT(key_cache);
    CHECK_INPUT(value_cache);
    CHECK_INPUT(decay_cache);
    CHECK_INPUT(query_cache);
    CHECK_INPUT(d_output);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    // Output gradients
    Tensor dx = torch::empty_like(x);
    Tensor dz = torch::empty_like(z);
    Tensor dW_key = torch::zeros({d, d}, options);
    Tensor db_key = torch::zeros({d}, options);
    Tensor dW_val = torch::zeros({d, k}, options);
    Tensor db_val = torch::zeros({k}, options);
    Tensor dW_query = torch::zeros({d, k}, options);
    Tensor db_query = torch::zeros({k}, options);
    Tensor dW_decay = torch::zeros({d, d}, options);
    Tensor db_decay = torch::zeros({d}, options);

    // Workspace size calculation
    const int64_t BD = batch_size * d;
    const int64_t BK = batch_size * k;
    const int64_t BDK = batch_size * d * k;
    // [d_proj_key_all: T*BD] [d_proj_val_all: T*BK] [d_proj_query_all: T*BK] [d_proj_decay_all: T*BD]
    // [d_H: BDK] [d_pre_out: BD]
    // Float workspace: [d_key_f: BD] [d_value_f: BK] [d_decay_f: BD] [d_query_f: BK]
    //                  [db_key_f: d] [db_val_f: k] [db_query_f: k] [db_decay_f: d]
    const int64_t float_ws_elems = 2 * BD + 2 * BK + 2 * d + 2 * k;
    const int64_t float_in_T = (float_ws_elems * sizeof(float) + sizeof(float) - 1) / sizeof(float);
    const int64_t workspace_size = 2 * time_steps * BD + 2 * time_steps * BK + BDK + BD + float_in_T * 2;
    Tensor workspace = torch::empty({workspace_size}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "matrix_state_elman_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        MatrixStateElmanBackward<typename native_type<scalar_t>::T> backward(
            batch_size, d, k,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W_key),
            ptr<scalar_t>(b_key),
            ptr<scalar_t>(W_val),
            ptr<scalar_t>(b_val),
            ptr<scalar_t>(W_query),
            ptr<scalar_t>(b_query),
            ptr<scalar_t>(W_decay),
            ptr<scalar_t>(b_decay),
            ptr<scalar_t>(x),
            ptr<scalar_t>(z),
            ptr<scalar_t>(H),
            ptr<scalar_t>(key_cache),
            ptr<scalar_t>(value_cache),
            ptr<scalar_t>(decay_cache),
            ptr<scalar_t>(query_cache),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dz),
            ptr<scalar_t>(dW_key),
            ptr<scalar_t>(db_key),
            ptr<scalar_t>(dW_val),
            ptr<scalar_t>(db_val),
            ptr<scalar_t>(dW_query),
            ptr<scalar_t>(db_query),
            ptr<scalar_t>(dW_decay),
            ptr<scalar_t>(db_decay),
            ptr<scalar_t>(workspace));
    }));

    return {dx, dz, dW_key, db_key, dW_val, db_val, dW_query, db_query, dW_decay, db_decay};
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
    m.def("softsign_elman_forward", &softsign_elman_forward,
          "Softsign Elman forward (E1 variant with softsign instead of tanh)");
    m.def("softsign_elman_backward", &softsign_elman_backward,
          "Softsign Elman backward");
    m.def("diagonal_state_elman_forward", &diagonal_state_elman_forward,
          "E16: Diagonal State Elman forward (Mamba2 efficiency + E1 nonlinearity)");
    m.def("diagonal_state_elman_backward", &diagonal_state_elman_backward,
          "E16: Diagonal State Elman backward");
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
    m.def("hybrid_elman_forward", &hybrid_elman_forward,
          "E9: Hybrid Elman forward (dense core + diagonal memory)");
    m.def("hybrid_elman_backward", &hybrid_elman_backward,
          "E9: Hybrid Elman backward");
    m.def("multiscale_elman_forward", &multiscale_elman_forward,
          "E10: Multi-Scale EMA Elman forward (learned decay memory banks)");
    m.def("multiscale_elman_backward", &multiscale_elman_backward,
          "E10: Multi-Scale EMA Elman backward");
    m.def("selective_elman_forward", &selective_elman_forward,
          "E11: Selective Memory Elman forward (input-dependent decay and read)");
    m.def("selective_elman_backward", &selective_elman_backward,
          "E11: Selective Memory Elman backward");
    m.def("selective_gated_elman_forward", &selective_gated_elman_forward,
          "E12: Selective Gated Elman forward (hidden-state-dependent gating)");
    m.def("selective_gated_elman_backward", &selective_gated_elman_backward,
          "E12: Selective Gated Elman backward");
    m.def("matrix_state_elman_forward", &matrix_state_elman_forward,
          "E14: Matrix State Elman forward (d*k matrix state, outer product update)");
    m.def("matrix_state_elman_backward", &matrix_state_elman_backward,
          "E14: Matrix State Elman backward");
}
