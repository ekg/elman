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
// E30: E1 + SSM-style diagonal gating
// gate = silu(z * g_z + h * g_h + b_gate)
// output = h * gate
// =============================================================================

std::vector<Tensor> e30_diagonal_gated_forward(
    bool training,
    Tensor x,           // [T, B, dim] pre-activated input
    Tensor z,           // [T, B, dim] gate input
    Tensor h0,          // [B, dim]
    Tensor W_x,         // [dim, dim]
    Tensor W_h,         // [dim, dim]
    Tensor b,           // [dim]
    Tensor g_z,         // [dim] gate scale for z
    Tensor g_h,         // [dim] gate scale for h
    Tensor b_gate) {    // [dim] gate bias

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(z);
    CHECK_INPUT(h0);
    CHECK_INPUT(W_x);
    CHECK_INPUT(W_h);
    CHECK_INPUT(b);
    CHECK_INPUT(g_z);
    CHECK_INPUT(g_h);
    CHECK_INPUT(b_gate);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, dim}, options)
                        : torch::empty({0}, options);
    Tensor gate_input_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                       : torch::empty({0}, options);

    // Forward workspace: [tmp_Wx: T*BD] [tmp_Rh: BD]
    const int64_t BD = batch_size * dim;
    Tensor workspace = torch::empty({time_steps * BD + BD}, options);

    h[0] = h0;

    TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16,
                "E30 forward only supports BFloat16 tensors, got ", x.scalar_type());

    using namespace hasty::v0::elman_ladder;
    using scalar_t = at::BFloat16;
    E30DiagonalGatedForward<__nv_bfloat16> forward(
        training, batch_size, dim,
        at::cuda::getCurrentCUDABlasHandle(),
        at::cuda::getCurrentCUDAStream());

    forward.Run(
        time_steps,
        ptr<scalar_t>(W_x),
        ptr<scalar_t>(W_h),
        ptr<scalar_t>(b),
        ptr<scalar_t>(g_z),
        ptr<scalar_t>(g_h),
        ptr<scalar_t>(b_gate),
        ptr<scalar_t>(x),
        ptr<scalar_t>(z),
        ptr<scalar_t>(h),
        ptr<scalar_t>(output),
        training ? ptr<scalar_t>(v) : nullptr,
        training ? ptr<scalar_t>(gate_input_cache) : nullptr,
        ptr<scalar_t>(workspace));

    return {h, output, v, gate_input_cache};
}

std::vector<Tensor> e30_diagonal_gated_backward(
    Tensor W_x,
    Tensor W_h,
    Tensor g_z,
    Tensor g_h,
    Tensor x,
    Tensor z,
    Tensor h,
    Tensor v,
    Tensor gate_input_cache,
    Tensor d_output) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(W_x);
    CHECK_INPUT(W_h);
    CHECK_INPUT(g_z);
    CHECK_INPUT(g_h);
    CHECK_INPUT(x);
    CHECK_INPUT(z);
    CHECK_INPUT(h);
    CHECK_INPUT(v);
    CHECK_INPUT(gate_input_cache);
    CHECK_INPUT(d_output);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::empty_like(x);
    Tensor dz = torch::empty_like(z);
    Tensor dW_x = torch::zeros({dim, dim}, options);
    Tensor dW_h = torch::zeros({dim, dim}, options);
    Tensor db = torch::zeros({dim}, options);
    Tensor dg_z = torch::zeros({dim}, options);
    Tensor dg_h = torch::zeros({dim}, options);
    Tensor db_gate = torch::zeros({dim}, options);

    // Workspace: [dh_t: BD] [dv_t: BD]
    const int64_t BD = batch_size * dim;
    Tensor workspace = torch::empty({2 * BD}, options);

    TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16,
                "E30 backward only supports BFloat16 tensors, got ", x.scalar_type());

    using namespace hasty::v0::elman_ladder;
    using scalar_t = at::BFloat16;
    E30DiagonalGatedBackward<__nv_bfloat16> backward(
        batch_size, dim,
        at::cuda::getCurrentCUDABlasHandle(),
        at::cuda::getCurrentCUDAStream());

    backward.Run(
        time_steps,
        ptr<scalar_t>(W_x),
        ptr<scalar_t>(W_h),
        ptr<scalar_t>(g_z),
        ptr<scalar_t>(g_h),
        ptr<scalar_t>(x),
        ptr<scalar_t>(z),
        ptr<scalar_t>(h),
        ptr<scalar_t>(v),
        ptr<scalar_t>(gate_input_cache),
        ptr<scalar_t>(d_output),
        ptr<scalar_t>(dx),
        ptr<scalar_t>(dz),
        ptr<scalar_t>(dW_x),
        ptr<scalar_t>(dW_h),
        ptr<scalar_t>(db),
        ptr<scalar_t>(dg_z),
        ptr<scalar_t>(dg_h),
        ptr<scalar_t>(db_gate),
        ptr<scalar_t>(workspace));

    return {dx, dz, dW_x, dW_h, db, dg_z, dg_h, db_gate};
}

// =============================================================================
// E33: Self-Gated Elman - Simplification test
// output = h * silu(h) instead of h * silu(z)
// This removes the need for the z branch entirely.
// =============================================================================

std::vector<Tensor> e33_self_gate_forward(
    bool training,
    Tensor x,           // [T, B, dim] pre-activated input
    Tensor h0,          // [B, dim]
    Tensor W_x,         // [dim, dim]
    Tensor W_h,         // [dim, dim] (already spectrally normalized)
    Tensor b) {         // [dim]

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(x);
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
        x.scalar_type(), "e33_self_gate_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E33SelfGateForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(W_h),
            ptr<scalar_t>(b),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v) : nullptr,
            ptr<scalar_t>(workspace));
    }));

    return {h, output, v};
}

std::vector<Tensor> e33_self_gate_backward(
    Tensor W_x,
    Tensor W_h,
    Tensor x,
    Tensor h,
    Tensor v,
    Tensor d_output) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(W_x);
    CHECK_INPUT(W_h);
    CHECK_INPUT(x);
    CHECK_INPUT(h);
    CHECK_INPUT(v);
    CHECK_INPUT(d_output);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::empty_like(x);
    Tensor dW_x = torch::zeros({dim, dim}, options);
    Tensor dW_h = torch::zeros({dim, dim}, options);
    Tensor db = torch::zeros({dim}, options);

    // Workspace layout: [dv_all: T*BD] [dh: BD] [dh_recurrent: BD] [dz_dummy: BD] [db_float: dim]
    // Note: dz_dummy needed because E33 reuses MambaGateBackward kernel which expects z gradient buffer
    const int64_t BD = batch_size * dim;
    const int64_t float_in_T = (dim * sizeof(float) + sizeof(float) - 1) / sizeof(float);
    const int64_t workspace_size = (time_steps + 3) * BD + float_in_T * 2;
    Tensor workspace = torch::empty({workspace_size}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e33_self_gate_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E33SelfGateBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(W_h),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dW_x),
            ptr<scalar_t>(dW_h),
            ptr<scalar_t>(db),
            ptr<scalar_t>(workspace));
    }));

    return {dx, dW_x, dW_h, db};
}

// =============================================================================
// E40: No Pre-SiLU Elman (E38 without pre-activation silu)
// h_t = tanh(x_t + W_h @ h_{t-1} + b)     # No W_x, no pre-silu
// output = h * silu(h)                     # Self-gating from E33
// Key: Testing if pre-silu is needed when W_x is already removed
// =============================================================================

std::vector<Tensor> e40_no_presilu_forward(
    bool training,
    Tensor x,           // [T, B, dim] input (NOT pre-activated, no silu!)
    Tensor h0,          // [B, dim]
    Tensor W_h,         // [dim, dim] - only hidden-to-hidden weight
    Tensor b) {         // [dim]

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(W_h);
    CHECK_INPUT(b);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, dim}, options)
                        : torch::empty({0}, options);

    // Forward workspace: [tmp_Rh: BD] - no Wx precompute needed!
    const int64_t BD = batch_size * dim;
    Tensor workspace = torch::empty({BD}, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e40_no_presilu_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E40NoPresiluForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W_h),
            ptr<scalar_t>(b),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v) : nullptr,
            ptr<scalar_t>(workspace));
    }));

    return {h, output, v};
}

std::vector<Tensor> e40_no_presilu_backward(
    Tensor W_h,
    Tensor x,
    Tensor h,
    Tensor v,
    Tensor d_output) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(W_h);
    CHECK_INPUT(x);
    CHECK_INPUT(h);
    CHECK_INPUT(v);
    CHECK_INPUT(d_output);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::empty({time_steps, batch_size, dim}, options);
    Tensor dW_h = torch::zeros({dim, dim}, options);
    Tensor db = torch::zeros({dim}, options);

    // Workspace layout: [dv_all: T*BD] [dh: BD] [dh_recurrent: BD] [db_float: dim]
    const int64_t BD = batch_size * dim;
    const int64_t float_in_T = (dim * sizeof(float) + sizeof(float) - 1) / sizeof(float);
    const int64_t workspace_size = (time_steps + 2) * BD + float_in_T * 2;
    Tensor workspace = torch::empty({workspace_size}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e40_no_presilu_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E40NoPresiluBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W_h),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dW_h),
            ptr<scalar_t>(db),
            ptr<scalar_t>(workspace));
    }));

    return {dx, dW_h, db};
}

// =============================================================================
// E34: Diagonal W_h Elman - W_h is diagonal vector instead of matrix
// =============================================================================

std::vector<Tensor> e34_diagonal_wh_forward(
    bool training,
    Tensor x, Tensor h0, Tensor W_x, Tensor d, Tensor b) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    CHECK_INPUT(x); CHECK_INPUT(h0); CHECK_INPUT(W_x); CHECK_INPUT(d); CHECK_INPUT(b);
    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, dim}, options) : torch::empty({0}, options);
    const int64_t BD = batch_size * dim;
    Tensor workspace = torch::empty({time_steps * BD + BD}, options);
    h[0] = h0;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e34_diagonal_wh_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E34DiagonalWhForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream());
        forward.Run(time_steps, ptr<scalar_t>(W_x), ptr<scalar_t>(d), ptr<scalar_t>(b),
            ptr<scalar_t>(x), ptr<scalar_t>(h), ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v) : nullptr, ptr<scalar_t>(workspace));
    }));
    return {h, output, v};
}

std::vector<Tensor> e34_diagonal_wh_backward(
    Tensor W_x, Tensor d, Tensor x, Tensor h, Tensor v, Tensor d_output) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    CHECK_INPUT(W_x); CHECK_INPUT(d); CHECK_INPUT(x); CHECK_INPUT(h); CHECK_INPUT(v); CHECK_INPUT(d_output);
    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    Tensor dx = torch::empty_like(x);
    Tensor dW_x = torch::zeros({dim, dim}, options);
    Tensor dd = torch::zeros({dim}, options);
    Tensor db = torch::zeros({dim}, options);
    const int64_t BD = batch_size * dim;
    const int64_t float_in_T = (dim * sizeof(float) + sizeof(float) - 1) / sizeof(float);
    Tensor workspace = torch::empty({(time_steps + 2) * BD + float_in_T * 2}, options);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e34_diagonal_wh_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E34DiagonalWhBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream());
        backward.Run(time_steps, ptr<scalar_t>(W_x), ptr<scalar_t>(d), ptr<scalar_t>(x),
            ptr<scalar_t>(h), ptr<scalar_t>(v), ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx), ptr<scalar_t>(dW_x), ptr<scalar_t>(dd), ptr<scalar_t>(db), ptr<scalar_t>(workspace));
    }));
    return {dx, dW_x, dd, db};
}

// =============================================================================
// E35: Cubic Gate Elman - output = h^3 instead of h * silu(h)
// =============================================================================

std::vector<Tensor> e35_cubic_gate_forward(
    bool training,
    Tensor x, Tensor h0, Tensor W_x, Tensor W_h, Tensor b) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    CHECK_INPUT(x); CHECK_INPUT(h0); CHECK_INPUT(W_x); CHECK_INPUT(W_h); CHECK_INPUT(b);
    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, dim}, options) : torch::empty({0}, options);
    const int64_t BD = batch_size * dim;
    Tensor workspace = torch::empty({time_steps * BD + BD}, options);
    h[0] = h0;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e35_cubic_gate_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E35CubicGateForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream());
        forward.Run(time_steps, ptr<scalar_t>(W_x), ptr<scalar_t>(W_h), ptr<scalar_t>(b),
            ptr<scalar_t>(x), ptr<scalar_t>(h), ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v) : nullptr, ptr<scalar_t>(workspace));
    }));
    return {h, output, v};
}

std::vector<Tensor> e35_cubic_gate_backward(
    Tensor W_x, Tensor W_h, Tensor x, Tensor h, Tensor v, Tensor d_output) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    CHECK_INPUT(W_x); CHECK_INPUT(W_h); CHECK_INPUT(x); CHECK_INPUT(h); CHECK_INPUT(v); CHECK_INPUT(d_output);
    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    Tensor dx = torch::empty_like(x);
    Tensor dW_x = torch::zeros({dim, dim}, options);
    Tensor dW_h = torch::zeros({dim, dim}, options);
    Tensor db = torch::zeros({dim}, options);
    const int64_t BD = batch_size * dim;
    const int64_t float_in_T = (dim * sizeof(float) + sizeof(float) - 1) / sizeof(float);
    Tensor workspace = torch::empty({(time_steps + 2) * BD + float_in_T * 2}, options);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e35_cubic_gate_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E35CubicGateBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream());
        backward.Run(time_steps, ptr<scalar_t>(W_x), ptr<scalar_t>(W_h), ptr<scalar_t>(x),
            ptr<scalar_t>(h), ptr<scalar_t>(v), ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx), ptr<scalar_t>(dW_x), ptr<scalar_t>(dW_h), ptr<scalar_t>(db), ptr<scalar_t>(workspace));
    }));
    return {dx, dW_x, dW_h, db};
}

// =============================================================================
// E36: Linear Recurrence - h_t = W_x @ x + W_h @ h (no tanh!)
// =============================================================================

std::vector<Tensor> e36_linear_recurrence_forward(
    bool training,
    Tensor x, Tensor h0, Tensor W_x, Tensor W_h, Tensor b) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    CHECK_INPUT(x); CHECK_INPUT(h0); CHECK_INPUT(W_x); CHECK_INPUT(W_h); CHECK_INPUT(b);
    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, dim}, options) : torch::empty({0}, options);
    const int64_t BD = batch_size * dim;
    Tensor workspace = torch::empty({time_steps * BD + BD}, options);
    h[0] = h0;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e36_linear_recurrence_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E36LinearRecurrenceForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream());
        forward.Run(time_steps, ptr<scalar_t>(W_x), ptr<scalar_t>(W_h), ptr<scalar_t>(b),
            ptr<scalar_t>(x), ptr<scalar_t>(h), ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v) : nullptr, ptr<scalar_t>(workspace));
    }));
    return {h, output, v};
}

std::vector<Tensor> e36_linear_recurrence_backward(
    Tensor W_x, Tensor W_h, Tensor x, Tensor h, Tensor v, Tensor d_output) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    CHECK_INPUT(W_x); CHECK_INPUT(W_h); CHECK_INPUT(x); CHECK_INPUT(h); CHECK_INPUT(v); CHECK_INPUT(d_output);
    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    Tensor dx = torch::empty_like(x);
    Tensor dW_x = torch::zeros({dim, dim}, options);
    Tensor dW_h = torch::zeros({dim, dim}, options);
    Tensor db = torch::zeros({dim}, options);
    const int64_t BD = batch_size * dim;
    const int64_t float_in_T = (dim * sizeof(float) + sizeof(float) - 1) / sizeof(float);
    Tensor workspace = torch::empty({(time_steps + 2) * BD + float_in_T * 2}, options);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e36_linear_recurrence_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E36LinearRecurrenceBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream());
        backward.Run(time_steps, ptr<scalar_t>(W_x), ptr<scalar_t>(W_h), ptr<scalar_t>(x),
            ptr<scalar_t>(h), ptr<scalar_t>(v), ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx), ptr<scalar_t>(dW_x), ptr<scalar_t>(dW_h), ptr<scalar_t>(db), ptr<scalar_t>(workspace));
    }));
    return {dx, dW_x, dW_h, db};
}

// =============================================================================
// E37: Tied Weights - W_x = W_h = W (single GEMM per timestep)
// =============================================================================

std::vector<Tensor> e37_tied_weights_forward(
    bool training,
    Tensor x, Tensor h0, Tensor W, Tensor b) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    CHECK_INPUT(x); CHECK_INPUT(h0); CHECK_INPUT(W); CHECK_INPUT(b);
    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, dim}, options) : torch::empty({0}, options);
    const int64_t BD = batch_size * dim;
    Tensor workspace = torch::empty({BD * 2}, options);  // tmp_sum + tmp_Wsum
    h[0] = h0;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e37_tied_weights_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E37TiedWeightsForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream());
        forward.Run(time_steps, ptr<scalar_t>(W), ptr<scalar_t>(b),
            ptr<scalar_t>(x), ptr<scalar_t>(h), ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v) : nullptr, ptr<scalar_t>(workspace));
    }));
    return {h, output, v};
}

std::vector<Tensor> e37_tied_weights_backward(
    Tensor W, Tensor x, Tensor h, Tensor v, Tensor d_output) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    CHECK_INPUT(W); CHECK_INPUT(x); CHECK_INPUT(h); CHECK_INPUT(v); CHECK_INPUT(d_output);
    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    Tensor dx = torch::empty_like(x);
    Tensor dW = torch::zeros({dim, dim}, options);
    Tensor db = torch::zeros({dim}, options);
    const int64_t BD = batch_size * dim;
    const int64_t float_in_T = (dim * sizeof(float) + sizeof(float) - 1) / sizeof(float);
    Tensor workspace = torch::empty({(time_steps + 3) * BD + float_in_T}, options);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e37_tied_weights_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E37TiedWeightsBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream());
        backward.Run(time_steps, ptr<scalar_t>(W), ptr<scalar_t>(x),
            ptr<scalar_t>(h), ptr<scalar_t>(v), ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx), ptr<scalar_t>(dW), ptr<scalar_t>(db), ptr<scalar_t>(workspace));
    }));
    return {dx, dW, db};
}

// =============================================================================
// E37v2: Optimized Tied Weights - Uses W @ x + W @ h (batched GEMM for W @ x)
// Same math as E37 but faster due to batched GEMM pattern
// =============================================================================

std::vector<Tensor> e37_tied_weights_v2_forward(
    bool training,
    Tensor x, Tensor h0, Tensor W, Tensor b) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    CHECK_INPUT(x); CHECK_INPUT(h0); CHECK_INPUT(W); CHECK_INPUT(b);
    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, dim}, options) : torch::empty({0}, options);
    const int64_t BD = batch_size * dim;
    // Workspace: tmp_Wx (T*BD) + tmp_Rh (BD)
    Tensor workspace = torch::empty({(time_steps + 1) * BD}, options);
    h[0] = h0;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e37_tied_weights_v2_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E37TiedWeightsV2Forward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream());
        forward.Run(time_steps, ptr<scalar_t>(W), ptr<scalar_t>(b),
            ptr<scalar_t>(x), ptr<scalar_t>(h), ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v) : nullptr, ptr<scalar_t>(workspace));
    }));
    return {h, output, v};
}

std::vector<Tensor> e37_tied_weights_v2_backward(
    Tensor W, Tensor x, Tensor h, Tensor v, Tensor d_output) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    CHECK_INPUT(W); CHECK_INPUT(x); CHECK_INPUT(h); CHECK_INPUT(v); CHECK_INPUT(d_output);
    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    Tensor dx = torch::empty_like(x);
    Tensor dW = torch::zeros({dim, dim}, options);
    Tensor db = torch::zeros({dim}, options);
    const int64_t BD = batch_size * dim;
    const int64_t float_in_T = (dim * sizeof(float) + sizeof(float) - 1) / sizeof(float);
    // Workspace: dv_all (T*BD) + dh (BD) + dh_recurrent (BD) + db_float (dim floats)
    Tensor workspace = torch::empty({(time_steps + 2) * BD + float_in_T}, options);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e37_tied_weights_v2_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E37TiedWeightsV2Backward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream());
        backward.Run(time_steps, ptr<scalar_t>(W), ptr<scalar_t>(x),
            ptr<scalar_t>(h), ptr<scalar_t>(v), ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx), ptr<scalar_t>(dW), ptr<scalar_t>(db), ptr<scalar_t>(workspace));
    }));
    return {dx, dW, db};
}

// =============================================================================
// E38: No W_x - h_t = tanh(x + W_h @ h + b), removes W_x entirely
// =============================================================================

std::vector<Tensor> e38_no_wx_forward(
    bool training,
    Tensor x, Tensor h0, Tensor W_h, Tensor b) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    CHECK_INPUT(x); CHECK_INPUT(h0); CHECK_INPUT(W_h); CHECK_INPUT(b);
    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, dim}, options) : torch::empty({0}, options);
    const int64_t BD = batch_size * dim;
    Tensor workspace = torch::empty({BD}, options);
    h[0] = h0;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e38_no_wx_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E38NoWxForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream());
        forward.Run(time_steps, ptr<scalar_t>(W_h), ptr<scalar_t>(b),
            ptr<scalar_t>(x), ptr<scalar_t>(h), ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v) : nullptr, ptr<scalar_t>(workspace));
    }));
    return {h, output, v};
}

std::vector<Tensor> e38_no_wx_backward(
    Tensor W_h, Tensor x, Tensor h, Tensor v, Tensor d_output) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    CHECK_INPUT(W_h); CHECK_INPUT(x); CHECK_INPUT(h); CHECK_INPUT(v); CHECK_INPUT(d_output);
    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    Tensor dx = torch::empty_like(x);
    Tensor dW_h = torch::zeros({dim, dim}, options);
    Tensor db = torch::zeros({dim}, options);
    const int64_t BD = batch_size * dim;
    const int64_t float_in_T = (dim * sizeof(float) + sizeof(float) - 1) / sizeof(float);
    Tensor workspace = torch::empty({(time_steps + 2) * BD + float_in_T}, options);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e38_no_wx_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E38NoWxBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream());
        backward.Run(time_steps, ptr<scalar_t>(W_h), ptr<scalar_t>(x),
            ptr<scalar_t>(h), ptr<scalar_t>(v), ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx), ptr<scalar_t>(dW_h), ptr<scalar_t>(db), ptr<scalar_t>(workspace));
    }));
    return {dx, dW_h, db};
}

// =============================================================================
// E39: No Bias - h_t = tanh(x + W_h @ h), removes bias entirely
// =============================================================================

std::vector<Tensor> e39_no_bias_forward(
    bool training,
    Tensor x, Tensor h0, Tensor W_h) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    CHECK_INPUT(x); CHECK_INPUT(h0); CHECK_INPUT(W_h);
    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, dim}, options) : torch::empty({0}, options);
    const int64_t BD = batch_size * dim;
    Tensor workspace = torch::empty({BD}, options);
    h[0] = h0;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e39_no_bias_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E39NoBiasForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream());
        forward.Run(time_steps, ptr<scalar_t>(W_h),
            ptr<scalar_t>(x), ptr<scalar_t>(h), ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v) : nullptr, ptr<scalar_t>(workspace));
    }));
    return {h, output, v};
}

std::vector<Tensor> e39_no_bias_backward(
    Tensor W_h, Tensor x, Tensor h, Tensor v, Tensor d_output) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    CHECK_INPUT(W_h); CHECK_INPUT(x); CHECK_INPUT(h); CHECK_INPUT(v); CHECK_INPUT(d_output);
    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    Tensor dx = torch::empty_like(x);
    Tensor dW_h = torch::zeros({dim, dim}, options);
    const int64_t BD = batch_size * dim;
    Tensor workspace = torch::empty({(time_steps + 2) * BD}, options);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e39_no_bias_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E39NoBiasBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream());
        backward.Run(time_steps, ptr<scalar_t>(W_h), ptr<scalar_t>(x),
            ptr<scalar_t>(h), ptr<scalar_t>(v), ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx), ptr<scalar_t>(dW_h), ptr<scalar_t>(workspace));
    }));
    return {dx, dW_h};
}

// =============================================================================
// E41: Diagonal W_x - d_x * x instead of W_x @ x
// =============================================================================

std::vector<Tensor> e41_diagonal_wx_forward(
    bool training,
    Tensor x, Tensor h0, Tensor d_x, Tensor W_h, Tensor b) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    CHECK_INPUT(x); CHECK_INPUT(h0); CHECK_INPUT(d_x); CHECK_INPUT(W_h); CHECK_INPUT(b);
    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, dim}, options) : torch::empty({0}, options);
    const int64_t BD = batch_size * dim;
    Tensor workspace = torch::empty({BD}, options);
    h[0] = h0;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e41_diagonal_wx_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E41DiagonalWxForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream());
        forward.Run(time_steps, ptr<scalar_t>(d_x), ptr<scalar_t>(W_h), ptr<scalar_t>(b),
            ptr<scalar_t>(x), ptr<scalar_t>(h), ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v) : nullptr, ptr<scalar_t>(workspace));
    }));
    return {h, output, v};
}

std::vector<Tensor> e41_diagonal_wx_backward(
    Tensor d_x, Tensor W_h, Tensor x, Tensor h, Tensor v, Tensor d_output) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    CHECK_INPUT(d_x); CHECK_INPUT(W_h); CHECK_INPUT(x); CHECK_INPUT(h); CHECK_INPUT(v); CHECK_INPUT(d_output);
    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    Tensor dx = torch::empty_like(x);
    Tensor dd_x = torch::zeros({dim}, options);
    Tensor dW_h = torch::zeros({dim, dim}, options);
    Tensor db = torch::zeros({dim}, options);
    const int64_t BD = batch_size * dim;
    const int64_t float_in_T = (dim * sizeof(float) + sizeof(float) - 1) / sizeof(float);
    Tensor workspace = torch::empty({(time_steps + 2) * BD + float_in_T * 2}, options);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e41_diagonal_wx_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E41DiagonalWxBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream());
        backward.Run(time_steps, ptr<scalar_t>(d_x), ptr<scalar_t>(W_h), ptr<scalar_t>(x),
            ptr<scalar_t>(h), ptr<scalar_t>(v), ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx), ptr<scalar_t>(dd_x), ptr<scalar_t>(dW_h), ptr<scalar_t>(db), ptr<scalar_t>(workspace));
    }));
    return {dx, dd_x, dW_h, db};
}

// =============================================================================
// E42: Linear Tied Self-Gated Elman - Linear recurrence + tied weights
// h_t = W @ x_t + W @ h_{t-1} + b    # LINEAR (no tanh!), tied
// output = h * silu(h)               # Self-gating
// =============================================================================

std::vector<Tensor> e42_linear_tied_forward(
    bool training,
    Tensor x, Tensor h0, Tensor W, Tensor b) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    CHECK_INPUT(x); CHECK_INPUT(h0); CHECK_INPUT(W); CHECK_INPUT(b);
    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, dim}, options) : torch::empty({0}, options);
    const int64_t BD = batch_size * dim;
    // Workspace: tmp_Wx (T*BD) + tmp_Wh (BD)
    Tensor workspace = torch::empty({(time_steps + 1) * BD}, options);
    h[0] = h0;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e42_linear_tied_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E42LinearTiedForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream());
        forward.Run(time_steps, ptr<scalar_t>(W), ptr<scalar_t>(b),
            ptr<scalar_t>(x), ptr<scalar_t>(h), ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v) : nullptr, ptr<scalar_t>(workspace));
    }));
    return {h, output, v};
}

std::vector<Tensor> e42_linear_tied_backward(
    Tensor W, Tensor x, Tensor h, Tensor v, Tensor d_output) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    CHECK_INPUT(W); CHECK_INPUT(x); CHECK_INPUT(h); CHECK_INPUT(v); CHECK_INPUT(d_output);
    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    Tensor dx = torch::empty_like(x);
    Tensor dW = torch::zeros({dim, dim}, options);
    Tensor db = torch::zeros({dim}, options);
    const int64_t BD = batch_size * dim;
    const int64_t float_in_T = (dim * sizeof(float) + sizeof(float) - 1) / sizeof(float);
    // Workspace: dv_all (T*BD) + dh (BD) + dh_recurrent (BD) + x_plus_h (T*BD) + db_float (dim floats)
    // x_plus_h is for fused dW GEMM optimization: dW = (x + h) @ dv_all.T
    Tensor workspace = torch::empty({(2 * time_steps + 2) * BD + float_in_T}, options);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e42_linear_tied_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E42LinearTiedBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream());
        backward.Run(time_steps, ptr<scalar_t>(W), ptr<scalar_t>(x),
            ptr<scalar_t>(h), ptr<scalar_t>(v), ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx), ptr<scalar_t>(dW), ptr<scalar_t>(db), ptr<scalar_t>(workspace));
    }));
    return {dx, dW, db};
}

// =============================================================================
// E52: Quadratic Gate Elman
// Tests if sigmoid in self-gate matters: uses pure h^2 instead of h * silu(h).
// h_t = W @ x_t + W @ h_{t-1} + b        # LINEAR (no tanh!), tied
// output = h^2                            # Pure quadratic (E52)
// output = h * |h|                        # Signed quadratic (E52b)
// =============================================================================

std::vector<Tensor> e52_quadratic_gate_forward(
    bool training,
    bool signed_quadratic,  // false = h^2, true = h*|h|
    Tensor x, Tensor h0, Tensor W, Tensor b) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    CHECK_INPUT(x); CHECK_INPUT(h0); CHECK_INPUT(W); CHECK_INPUT(b);
    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, dim}, options) : torch::empty({0}, options);
    const int64_t BD = batch_size * dim;
    // Workspace: tmp_Wx (T*BD) + tmp_Wh (BD)
    Tensor workspace = torch::empty({(time_steps + 1) * BD}, options);
    h[0] = h0;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e52_quadratic_gate_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E52QuadraticGateForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, signed_quadratic, at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream());
        forward.Run(time_steps, ptr<scalar_t>(W), ptr<scalar_t>(b),
            ptr<scalar_t>(x), ptr<scalar_t>(h), ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v) : nullptr, ptr<scalar_t>(workspace));
    }));
    return {h, output, v};
}

std::vector<Tensor> e52_quadratic_gate_backward(
    bool signed_quadratic,  // false = h^2, true = h*|h|
    Tensor W, Tensor x, Tensor h, Tensor v, Tensor d_output) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    CHECK_INPUT(W); CHECK_INPUT(x); CHECK_INPUT(h); CHECK_INPUT(v); CHECK_INPUT(d_output);
    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    Tensor dx = torch::empty_like(x);
    Tensor dW = torch::zeros({dim, dim}, options);
    Tensor db = torch::zeros({dim}, options);
    const int64_t BD = batch_size * dim;
    const int64_t float_in_T = (dim * sizeof(float) + sizeof(float) - 1) / sizeof(float);
    // Workspace: dv_all (T*BD) + dh (BD) + dh_recurrent (BD) + x_plus_h (T*BD) + db_float (dim floats)
    Tensor workspace = torch::empty({(2 * time_steps + 2) * BD + float_in_T}, options);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e52_quadratic_gate_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E52QuadraticGateBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, signed_quadratic, at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream());
        backward.Run(time_steps, ptr<scalar_t>(W), ptr<scalar_t>(x),
            ptr<scalar_t>(h), ptr<scalar_t>(v), ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx), ptr<scalar_t>(dW), ptr<scalar_t>(db), ptr<scalar_t>(workspace));
    }));
    return {dx, dW, db};
}

// =============================================================================
// E53: Sigmoid Gate Only Elman
// Tests if the quadratic component matters: uses silu(h) instead of h * silu(h).
// h_t = W @ x_t + W @ h_{t-1} + b        # LINEAR (no tanh!), tied
// output = silu(h)                        # Just silu, NOT h * silu(h)!
// =============================================================================

std::vector<Tensor> e53_sigmoid_gate_forward(
    bool training,
    Tensor x, Tensor h0, Tensor W, Tensor b) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    CHECK_INPUT(x); CHECK_INPUT(h0); CHECK_INPUT(W); CHECK_INPUT(b);
    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, dim}, options) : torch::empty({0}, options);
    const int64_t BD = batch_size * dim;
    // Workspace: tmp_Wx (T*BD) + tmp_Wh (BD)
    Tensor workspace = torch::empty({(time_steps + 1) * BD}, options);
    h[0] = h0;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e53_sigmoid_gate_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E53SigmoidGateForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream());
        forward.Run(time_steps, ptr<scalar_t>(W), ptr<scalar_t>(b),
            ptr<scalar_t>(x), ptr<scalar_t>(h), ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v) : nullptr, ptr<scalar_t>(workspace));
    }));
    return {h, output, v};
}

std::vector<Tensor> e53_sigmoid_gate_backward(
    Tensor W, Tensor x, Tensor h, Tensor v, Tensor d_output) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    CHECK_INPUT(W); CHECK_INPUT(x); CHECK_INPUT(h); CHECK_INPUT(v); CHECK_INPUT(d_output);
    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    Tensor dx = torch::empty_like(x);
    Tensor dW = torch::zeros({dim, dim}, options);
    Tensor db = torch::zeros({dim}, options);
    const int64_t BD = batch_size * dim;
    const int64_t float_in_T = (dim * sizeof(float) + sizeof(float) - 1) / sizeof(float);
    // Workspace: dv_all (T*BD) + dh (BD) + dh_recurrent (BD) + x_plus_h (T*BD) + db_float (dim floats)
    Tensor workspace = torch::empty({(2 * time_steps + 2) * BD + float_in_T}, options);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e53_sigmoid_gate_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E53SigmoidGateBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream());
        backward.Run(time_steps, ptr<scalar_t>(W), ptr<scalar_t>(x),
            ptr<scalar_t>(h), ptr<scalar_t>(v), ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx), ptr<scalar_t>(dW), ptr<scalar_t>(db), ptr<scalar_t>(workspace));
    }));
    return {dx, dW, db};
}

// =============================================================================
// E48: No Projections Elman
// Removes BOTH in_proj and out_proj - operates directly on embedding space.
// h_t = W @ (x_t + h_{t-1}) + b   # W is dim×dim, operates on embeddings
// output = h * silu(h)            # Self-gating (only nonlinearity)
// y = output                       # Direct to residual (no out_proj!)
// =============================================================================

std::vector<Tensor> e48_no_projections_forward(
    bool training,
    Tensor x, Tensor h0, Tensor W, Tensor b) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    CHECK_INPUT(x); CHECK_INPUT(h0); CHECK_INPUT(W); CHECK_INPUT(b);
    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, dim}, options) : torch::empty({0}, options);
    const int64_t BD = batch_size * dim;
    // Workspace: tmp_Wx (T*BD) + tmp_Wh (BD)
    Tensor workspace = torch::empty({(time_steps + 1) * BD}, options);
    h[0] = h0;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e48_no_projections_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E48NoProjectionsForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream());
        forward.Run(time_steps, ptr<scalar_t>(W), ptr<scalar_t>(b),
            ptr<scalar_t>(x), ptr<scalar_t>(h), ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v) : nullptr, ptr<scalar_t>(workspace));
    }));
    return {h, output, v};
}

std::vector<Tensor> e48_no_projections_backward(
    Tensor W, Tensor x, Tensor h, Tensor v, Tensor d_output) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    CHECK_INPUT(W); CHECK_INPUT(x); CHECK_INPUT(h); CHECK_INPUT(v); CHECK_INPUT(d_output);
    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    Tensor dx = torch::empty_like(x);
    Tensor dW = torch::zeros({dim, dim}, options);
    Tensor db = torch::zeros({dim}, options);
    const int64_t BD = batch_size * dim;
    const int64_t float_in_T = (dim * sizeof(float) + sizeof(float) - 1) / sizeof(float);
    // Workspace: dv_all (T*BD) + dh (BD) + dh_recurrent (BD) + x_plus_h (T*BD) + db_float (dim floats)
    Tensor workspace = torch::empty({(2 * time_steps + 2) * BD + float_in_T}, options);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e48_no_projections_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E48NoProjectionsBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream());
        backward.Run(time_steps, ptr<scalar_t>(W), ptr<scalar_t>(x),
            ptr<scalar_t>(h), ptr<scalar_t>(v), ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx), ptr<scalar_t>(dW), ptr<scalar_t>(db), ptr<scalar_t>(workspace));
    }));
    return {dx, dW, db};
}

// =============================================================================
// E51: No Self-Gate Elman
// Tests if self-gating is necessary: removes h * silu(h) and uses linear output.
// h_t = W @ (x_t + h_{t-1}) + b    # Same as E42
// output = h_t                      # LINEAR! No gating!
// =============================================================================

std::vector<Tensor> e51_no_self_gate_forward(
    bool training,
    Tensor x, Tensor h0, Tensor W, Tensor b) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    CHECK_INPUT(x); CHECK_INPUT(h0); CHECK_INPUT(W); CHECK_INPUT(b);
    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, dim}, options) : torch::empty({0}, options);
    const int64_t BD = batch_size * dim;
    // Workspace: tmp_Wx (T*BD) + tmp_Wh (BD)
    Tensor workspace = torch::empty({(time_steps + 1) * BD}, options);
    h[0] = h0;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e51_no_self_gate_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E51NoSelfGateForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream());
        forward.Run(time_steps, ptr<scalar_t>(W), ptr<scalar_t>(b),
            ptr<scalar_t>(x), ptr<scalar_t>(h), ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v) : nullptr, ptr<scalar_t>(workspace));
    }));
    return {h, output, v};
}

std::vector<Tensor> e51_no_self_gate_backward(
    Tensor W, Tensor x, Tensor h, Tensor v, Tensor d_output) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    CHECK_INPUT(W); CHECK_INPUT(x); CHECK_INPUT(h); CHECK_INPUT(v); CHECK_INPUT(d_output);
    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    Tensor dx = torch::empty_like(x);
    Tensor dW = torch::zeros({dim, dim}, options);
    Tensor db = torch::zeros({dim}, options);
    const int64_t BD = batch_size * dim;
    const int64_t float_in_T = (dim * sizeof(float) + sizeof(float) - 1) / sizeof(float);
    // Workspace: dv_all (T*BD) + dh (BD) + dh_recurrent (BD) + x_plus_h (T*BD) + db_float (dim floats)
    Tensor workspace = torch::empty({(2 * time_steps + 2) * BD + float_in_T}, options);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e51_no_self_gate_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E51NoSelfGateBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream());
        backward.Run(time_steps, ptr<scalar_t>(W), ptr<scalar_t>(x),
            ptr<scalar_t>(h), ptr<scalar_t>(v), ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx), ptr<scalar_t>(dW), ptr<scalar_t>(db), ptr<scalar_t>(workspace));
    }));
    return {dx, dW, db};
}

// =============================================================================
// E45: Pure Accumulation Elman (NO GEMM - simplest possible)
// h_t = x_t + h_{t-1}               # Just add! No parameters in recurrence!
// output = h_t * silu(h_t)          # Self-gating
// =============================================================================

std::vector<Tensor> e45_pure_accumulation_forward(
    bool training,
    Tensor x, Tensor h0) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    CHECK_INPUT(x); CHECK_INPUT(h0);
    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor workspace = torch::empty({0}, options);
    h[0] = h0;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e45_pure_accumulation_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E45PureAccumulationForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream());
        forward.Run(time_steps, ptr<scalar_t>(x),
            ptr<scalar_t>(h), ptr<scalar_t>(output), ptr<scalar_t>(workspace));
    }));
    return {h, output};
}

std::vector<Tensor> e45_pure_accumulation_backward(
    Tensor h, Tensor d_output) {
    const auto time_steps = d_output.size(0);
    const auto batch_size = d_output.size(1);
    const auto dim = d_output.size(2);
    CHECK_INPUT(h); CHECK_INPUT(d_output);
    const auto options = d_output.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    Tensor dx = torch::empty_like(d_output);
    const int64_t BD = batch_size * dim;
    Tensor workspace = torch::empty({2 * BD}, options);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        d_output.scalar_type(), "e45_pure_accumulation_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E45PureAccumulationBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream());
        backward.Run(time_steps, ptr<scalar_t>(h), ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx), ptr<scalar_t>(workspace));
    }));
    return {dx};
}

// =============================================================================
// E45b: Pure Accumulation with Decay (learned scalar alpha)
// h_t = x_t + alpha * h_{t-1}       # Decay prevents unbounded growth
// output = h_t * silu(h_t)          # Self-gating
// =============================================================================

std::vector<Tensor> e45b_with_decay_forward(
    bool training,
    Tensor x, Tensor h0, float alpha) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    CHECK_INPUT(x); CHECK_INPUT(h0);
    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor workspace = torch::empty({0}, options);
    h[0] = h0;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e45b_with_decay_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E45bWithDecayForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream());
        forward.Run(time_steps, alpha, ptr<scalar_t>(x),
            ptr<scalar_t>(h), ptr<scalar_t>(output), ptr<scalar_t>(workspace));
    }));
    return {h, output};
}

std::vector<Tensor> e45b_with_decay_backward(
    float alpha, Tensor h, Tensor d_output) {
    const auto time_steps = d_output.size(0);
    const auto batch_size = d_output.size(1);
    const auto dim = d_output.size(2);
    CHECK_INPUT(h); CHECK_INPUT(d_output);
    const auto options = d_output.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    Tensor dx = torch::empty_like(d_output);
    Tensor d_alpha = torch::zeros({1}, options.dtype(torch::kFloat32));
    const int64_t BD = batch_size * dim;
    Tensor workspace = torch::empty({2 * BD}, options);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        d_output.scalar_type(), "e45b_with_decay_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E45bWithDecayBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream());
        backward.Run(time_steps, alpha, ptr<scalar_t>(h), ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx), d_alpha.data_ptr<float>(), ptr<scalar_t>(workspace));
    }));
    return {dx, d_alpha};
}

// =============================================================================
// E46: No In-Projection Elman (operates directly on embeddings)
// h_t = W @ (x_t + h_{t-1}) + b     # W is dim*dim, no expansion
// output = h_t * silu(h_t)          # Self-gating
// =============================================================================

std::vector<Tensor> e46_no_in_proj_forward(
    bool training,
    Tensor x, Tensor h0, Tensor W, Tensor b) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    CHECK_INPUT(x); CHECK_INPUT(h0); CHECK_INPUT(W); CHECK_INPUT(b);
    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, dim}, options) : torch::empty({0}, options);
    const int64_t BD = batch_size * dim;
    Tensor workspace = torch::empty({2 * BD}, options);
    h[0] = h0;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e46_no_in_proj_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E46NoInProjForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream());
        forward.Run(time_steps, ptr<scalar_t>(W), ptr<scalar_t>(b),
            ptr<scalar_t>(x), ptr<scalar_t>(h), ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v) : nullptr, ptr<scalar_t>(workspace));
    }));
    return {h, output, v};
}

std::vector<Tensor> e46_no_in_proj_backward(
    Tensor W, Tensor x, Tensor h, Tensor v, Tensor d_output) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    CHECK_INPUT(W); CHECK_INPUT(x); CHECK_INPUT(h); CHECK_INPUT(v); CHECK_INPUT(d_output);
    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    Tensor dx = torch::empty_like(x);
    Tensor dW = torch::zeros({dim, dim}, options);
    Tensor db = torch::zeros({dim}, options);
    const int64_t BD = batch_size * dim;
    const int64_t float_in_T = (dim * sizeof(float) + sizeof(float) - 1) / sizeof(float);
    Tensor workspace = torch::empty({(2 * time_steps + 2) * BD + float_in_T}, options);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e46_no_in_proj_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E46NoInProjBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream());
        backward.Run(time_steps, ptr<scalar_t>(W), ptr<scalar_t>(x),
            ptr<scalar_t>(h), ptr<scalar_t>(v), ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx), ptr<scalar_t>(dW), ptr<scalar_t>(db), ptr<scalar_t>(workspace));
    }));
    return {dx, dW, db};
}

// =============================================================================
// E54: Diagonal Decay + No Projections Elman
// h_t = d * (x_t + h_{t-1}) + b    # Per-dimension decay, NO GEMM!
// output = h * silu(h)             # Self-gating
// =============================================================================

std::vector<Tensor> e54_diagonal_no_proj_forward(
    bool training,
    Tensor x, Tensor h0, Tensor d, Tensor b) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    CHECK_INPUT(x); CHECK_INPUT(h0); CHECK_INPUT(d); CHECK_INPUT(b);
    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, dim}, options) : torch::empty({0}, options);
    // E54 needs NO workspace - pure element-wise ops!
    Tensor workspace = torch::empty({0}, options);
    h[0] = h0;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e54_diagonal_no_proj_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E54DiagonalNoProjForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream());
        forward.Run(time_steps, ptr<scalar_t>(d), ptr<scalar_t>(b),
            ptr<scalar_t>(x), ptr<scalar_t>(h), ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v) : nullptr, ptr<scalar_t>(workspace));
    }));
    return {h, output, v};
}

std::vector<Tensor> e54_diagonal_no_proj_backward(
    Tensor d, Tensor x, Tensor h, Tensor v, Tensor d_output) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    CHECK_INPUT(d); CHECK_INPUT(x); CHECK_INPUT(h); CHECK_INPUT(v); CHECK_INPUT(d_output);
    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    Tensor dx = torch::empty_like(x);
    Tensor dd = torch::zeros({dim}, options);
    Tensor db = torch::zeros({dim}, options);
    const int64_t BD = batch_size * dim;
    // Workspace: dh (BD) + dh_recurrent (BD) + dd_float (dim floats) + db_float (dim floats)
    const int64_t float_size = (2 * dim * sizeof(float) + sizeof(float) - 1) / sizeof(float);
    Tensor workspace = torch::empty({2 * BD + float_size}, options);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e54_diagonal_no_proj_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E54DiagonalNoProjBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream());
        backward.Run(time_steps, ptr<scalar_t>(d), ptr<scalar_t>(x),
            ptr<scalar_t>(h), ptr<scalar_t>(v), ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx), ptr<scalar_t>(dd), ptr<scalar_t>(db), ptr<scalar_t>(workspace));
    }));
    return {dx, dd, db};
}

// =============================================================================
// E55: Scalar Decay + No Projections Elman
// h_t = lambda * (x_t + h_{t-1}) + b    # SINGLE scalar lambda, NO GEMM!
// output = h * silu(h)                   # Self-gating
// =============================================================================

std::vector<Tensor> e55_scalar_no_proj_forward(
    bool training,
    Tensor x, Tensor h0, float lambda, Tensor b) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    CHECK_INPUT(x); CHECK_INPUT(h0); CHECK_INPUT(b);
    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, dim}, options) : torch::empty({0}, options);
    // E55 needs NO workspace - pure element-wise ops!
    Tensor workspace = torch::empty({0}, options);
    h[0] = h0;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e55_scalar_no_proj_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E55ScalarNoProjForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream());
        forward.Run(time_steps, lambda, ptr<scalar_t>(b),
            ptr<scalar_t>(x), ptr<scalar_t>(h), ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v) : nullptr, ptr<scalar_t>(workspace));
    }));
    return {h, output, v};
}

std::vector<Tensor> e55_scalar_no_proj_backward(
    float lambda, Tensor x, Tensor h, Tensor v, Tensor d_output) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    CHECK_INPUT(x); CHECK_INPUT(h); CHECK_INPUT(v); CHECK_INPUT(d_output);
    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    Tensor dx = torch::empty_like(x);
    Tensor dlambda = torch::zeros({1}, options.dtype(torch::kFloat32));  // float output!
    Tensor db = torch::zeros({dim}, options);
    const int64_t BD = batch_size * dim;
    // Workspace: dh (BD) + dh_recurrent (BD) + db_float (dim floats)
    const int64_t float_size = (dim * sizeof(float) + sizeof(float) - 1) / sizeof(float);
    Tensor workspace = torch::empty({2 * BD + float_size}, options);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e55_scalar_no_proj_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E55ScalarNoProjBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream());
        backward.Run(time_steps, lambda, ptr<scalar_t>(x),
            ptr<scalar_t>(h), ptr<scalar_t>(v), ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx), dlambda.data_ptr<float>(), ptr<scalar_t>(db), ptr<scalar_t>(workspace));
    }));
    return {dx, dlambda, db};
}

// =============================================================================
// E43: Scalar Decay Elman
// h_t = λ * (x_t + h_{t-1}) + b         # Scalar decay (NO matrix!)
// output = h_t * silu(h_t)               # Self-gating (only nonlinearity)
// λ = sigmoid(log_lambda) ∈ (0, 1) for stability
// =============================================================================

std::vector<Tensor> e43_scalar_decay_forward(
    bool training,
    Tensor x, Tensor h0, Tensor log_lambda, Tensor b) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    CHECK_INPUT(x); CHECK_INPUT(h0); CHECK_INPUT(log_lambda); CHECK_INPUT(b);
    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, dim}, options) : torch::empty({0}, options);
    // E43 needs NO workspace - pure element-wise ops!
    Tensor workspace = torch::empty({0}, options);
    h[0] = h0;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e43_scalar_decay_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E43ScalarDecayForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream());
        forward.Run(time_steps, ptr<scalar_t>(log_lambda), ptr<scalar_t>(b),
            ptr<scalar_t>(x), ptr<scalar_t>(h), ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v) : nullptr, ptr<scalar_t>(workspace));
    }));
    return {h, output, v};
}

std::vector<Tensor> e43_scalar_decay_backward(
    Tensor log_lambda, Tensor x, Tensor h, Tensor v, Tensor d_output) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    CHECK_INPUT(log_lambda); CHECK_INPUT(x); CHECK_INPUT(h); CHECK_INPUT(v); CHECK_INPUT(d_output);
    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    Tensor dx = torch::empty_like(x);
    Tensor d_log_lambda = torch::zeros({1}, options);
    Tensor db = torch::zeros({dim}, options);
    const int64_t BD = batch_size * dim;
    // Workspace: dh (BD) + dh_recurrent (BD) + db_float (dim floats) + d_lambda_float (1 float)
    const int64_t float_size = ((dim + 1) * sizeof(float) + sizeof(float) - 1) / sizeof(float);
    Tensor workspace = torch::empty({2 * BD + float_size}, options);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e43_scalar_decay_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E43ScalarDecayBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream());
        backward.Run(time_steps, ptr<scalar_t>(log_lambda), ptr<scalar_t>(x),
            ptr<scalar_t>(h), ptr<scalar_t>(v), ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx), ptr<scalar_t>(d_log_lambda), ptr<scalar_t>(db), ptr<scalar_t>(workspace));
    }));
    return {dx, d_log_lambda, db};
}

// =============================================================================
// E44: Diagonal W Elman (Mamba2-style)
// h_t = d * (x_t + h_{t-1}) + b         # d is [dim] vector, element-wise
// output = h_t * silu(h_t)               # Self-gating (only nonlinearity)
// d = sigmoid(log_d) ∈ (0, 1)^dim for stability
// =============================================================================

std::vector<Tensor> e44_diagonal_w_forward(
    bool training,
    Tensor x, Tensor h0, Tensor log_d, Tensor b) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    CHECK_INPUT(x); CHECK_INPUT(h0); CHECK_INPUT(log_d); CHECK_INPUT(b);
    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, dim}, options) : torch::empty({0}, options);
    // E44 needs NO workspace - pure element-wise ops!
    Tensor workspace = torch::empty({0}, options);
    h[0] = h0;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e44_diagonal_w_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E44DiagonalWForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream());
        forward.Run(time_steps, ptr<scalar_t>(log_d), ptr<scalar_t>(b),
            ptr<scalar_t>(x), ptr<scalar_t>(h), ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v) : nullptr, ptr<scalar_t>(workspace));
    }));
    return {h, output, v};
}

std::vector<Tensor> e44_diagonal_w_backward(
    Tensor log_d, Tensor x, Tensor h, Tensor v, Tensor d_output) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    CHECK_INPUT(log_d); CHECK_INPUT(x); CHECK_INPUT(h); CHECK_INPUT(v); CHECK_INPUT(d_output);
    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    Tensor dx = torch::empty_like(x);
    Tensor d_log_d = torch::zeros({dim}, options);
    Tensor db = torch::zeros({dim}, options);
    const int64_t BD = batch_size * dim;
    // Workspace: dh (BD) + dh_recurrent (BD) + db_float (dim floats) + d_log_d_float (dim floats)
    const int64_t float_size = (2 * dim * sizeof(float) + sizeof(float) - 1) / sizeof(float);
    Tensor workspace = torch::empty({2 * BD + float_size}, options);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e44_diagonal_w_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E44DiagonalWBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream());
        backward.Run(time_steps, ptr<scalar_t>(log_d), ptr<scalar_t>(x),
            ptr<scalar_t>(h), ptr<scalar_t>(v), ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx), ptr<scalar_t>(d_log_d), ptr<scalar_t>(db), ptr<scalar_t>(workspace));
    }));
    return {dx, d_log_d, db};
}

// =============================================================================
// E56: Concat Elman - Single GEMM on concatenated [x, h] input
// h_t = tanh(W @ [x_t; h_{t-1}] + b)  # Single GEMM with W [dim, 2*dim]
// output = h * silu(z)                 # Gate with z branch
// =============================================================================

std::vector<Tensor> e56_concat_elman_forward(
    bool training,
    Tensor x,           // [T, B, dim] pre-activated input
    Tensor z,           // [T, B, dim] gate input
    Tensor h0,          // [B, dim]
    Tensor W,           // [dim, 2*dim] - single weight matrix
    Tensor b) {         // [dim]

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(z);
    CHECK_INPUT(h0);
    CHECK_INPUT(W);
    CHECK_INPUT(b);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, dim}, options) : torch::empty({0}, options);

    // Workspace: xh_concat [B, 2*dim] + Wxh [B, dim]
    Tensor workspace = torch::empty({batch_size * 3 * dim}, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e56_concat_elman_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E56ConcatElmanForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream());
        forward.Run(time_steps, ptr<scalar_t>(W), ptr<scalar_t>(b),
            ptr<scalar_t>(x), ptr<scalar_t>(z), ptr<scalar_t>(h),
            ptr<scalar_t>(output), training ? ptr<scalar_t>(v) : nullptr,
            ptr<scalar_t>(workspace));
    }));

    return {h, output, v};
}

std::vector<Tensor> e56_concat_elman_backward(
    Tensor W,           // [dim, 2*dim]
    Tensor x,           // [T, B, dim]
    Tensor z,           // [T, B, dim]
    Tensor h,           // [T+1, B, dim]
    Tensor v,           // [T, B, dim]
    Tensor d_output) {  // [T, B, dim]

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(W);
    CHECK_INPUT(x);
    CHECK_INPUT(z);
    CHECK_INPUT(h);
    CHECK_INPUT(v);
    CHECK_INPUT(d_output);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::empty_like(x);
    Tensor dz = torch::empty_like(z);
    Tensor dW = torch::zeros({dim, 2 * dim}, options);
    Tensor db = torch::zeros({dim}, options);

    const int64_t BD = batch_size * dim;
    // Workspace: dv_all (T*BD) + dh (BD) + dh_recurrent (BD) + dxh (2*BD) + xh_concat (2*BD) + db_float (dim floats)
    const int64_t float_size = (dim * sizeof(float) + sizeof(float) - 1) / sizeof(float);
    Tensor workspace = torch::empty({(time_steps + 6) * BD + float_size}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e56_concat_elman_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E56ConcatElmanBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream());
        backward.Run(time_steps, ptr<scalar_t>(W), ptr<scalar_t>(x), ptr<scalar_t>(z),
            ptr<scalar_t>(h), ptr<scalar_t>(v), ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx), ptr<scalar_t>(dz), ptr<scalar_t>(dW), ptr<scalar_t>(db),
            ptr<scalar_t>(workspace));
    }));

    return {dx, dz, dW, db};
}

// =============================================================================
// E17: Selective W_h Elman (input-dependent gating on recurrence)
// h_t = tanh(W_x @ x_t + (W_h @ h_{t-1}) * sigmoid(W_gate @ x_t) + b)
// output = h * silu(z)
// Key: Diagonal selectivity on W_h @ h (like Mamba2's selective A)
// =============================================================================

std::vector<Tensor> selective_wh_elman_forward(
    bool training,
    Tensor x,           // [T, B, dim] pre-activated input
    Tensor z,           // [T, B, dim] gate input (pre silu)
    Tensor h0,          // [B, dim]
    Tensor W_x,         // [dim, dim]
    Tensor W_h,         // [dim, dim]
    Tensor W_gate,      // [dim, dim] gate projection
    Tensor b) {         // [dim]

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(z);
    CHECK_INPUT(h0);
    CHECK_INPUT(W_x);
    CHECK_INPUT(W_h);
    CHECK_INPUT(W_gate);
    CHECK_INPUT(b);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, dim}, options)
                        : torch::empty({0}, options);
    Tensor gate_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                 : torch::empty({0}, options);
    Tensor Rh_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                               : torch::empty({0}, options);

    // Forward workspace: [tmp_Wx: T*BD] [tmp_G: T*BD] [tmp_Rh: BD]
    const int64_t BD = batch_size * dim;
    Tensor workspace = torch::empty({2 * time_steps * BD + BD}, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "selective_wh_elman_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        SelectiveWhElmanForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(W_h),
            ptr<scalar_t>(W_gate),
            ptr<scalar_t>(b),
            nullptr,  // b_gate (optional)
            ptr<scalar_t>(x),
            ptr<scalar_t>(z),
            ptr<scalar_t>(h),
            ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v) : nullptr,
            training ? ptr<scalar_t>(gate_cache) : nullptr,
            training ? ptr<scalar_t>(Rh_cache) : nullptr,
            ptr<scalar_t>(workspace));
    }));

    return {h, output, v, gate_cache, Rh_cache};
}

std::vector<Tensor> selective_wh_elman_backward(
    Tensor W_x,
    Tensor W_h,
    Tensor W_gate,
    Tensor x,
    Tensor z,
    Tensor h,
    Tensor v,
    Tensor gate_cache,
    Tensor Rh_cache,
    Tensor d_output) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(W_x);
    CHECK_INPUT(W_h);
    CHECK_INPUT(W_gate);
    CHECK_INPUT(x);
    CHECK_INPUT(z);
    CHECK_INPUT(h);
    CHECK_INPUT(v);
    CHECK_INPUT(gate_cache);
    CHECK_INPUT(Rh_cache);
    CHECK_INPUT(d_output);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::empty_like(x);
    Tensor dz = torch::empty_like(z);
    Tensor dW_x = torch::zeros({dim, dim}, options);
    Tensor dW_h = torch::zeros({dim, dim}, options);
    Tensor dW_gate = torch::zeros({dim, dim}, options);
    Tensor db = torch::zeros({dim}, options);

    // Workspace layout: [dv_all: T*BD] [dG_all: T*BD] [dRh_all: T*BD] [dh: BD] [dh_rec: BD] [db_float: dim]
    const int64_t BD = batch_size * dim;
    const int64_t float_in_T = (dim * sizeof(float) + sizeof(float) - 1) / sizeof(float);
    const int64_t workspace_size = (3 * time_steps + 2) * BD + float_in_T * 2;
    Tensor workspace = torch::empty({workspace_size}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "selective_wh_elman_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        SelectiveWhElmanBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(W_h),
            ptr<scalar_t>(W_gate),
            ptr<scalar_t>(x),
            ptr<scalar_t>(z),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(gate_cache),
            ptr<scalar_t>(Rh_cache),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dz),
            ptr<scalar_t>(dW_x),
            ptr<scalar_t>(dW_h),
            ptr<scalar_t>(dW_gate),
            ptr<scalar_t>(db),
            ptr<scalar_t>(workspace));
    }));

    return {dx, dz, dW_x, dW_h, dW_gate, db};
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

    // Workspace: [dBx: T*BS] [dh_rec: BS] [dy: BD] [Cy: BD] [dA_float: d_state*2]
    const int64_t BS = batch_size * d_state;
    const int64_t BD = batch_size * d_model;
    const int64_t workspace_size = time_steps * BS + BS + 2 * BD + d_state * 2;
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

// =============================================================================
// E18: h-Aware Gate Elman (h-aware output gating)
// =============================================================================

std::vector<Tensor> haware_gate_elman_forward(
    bool training,
    int gate_mode,      // 0=A (z+h), 1=B (z+Rh), 2=E (no gate)
    Tensor x,           // [T, B, dim] pre-activated input
    Tensor z,           // [T, B, dim] gate input (ignored for mode=2)
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
    // Rh_cache only needed for mode=1 (E18-B)
    Tensor Rh_cache = (training && gate_mode == 1)
                        ? torch::empty({time_steps, batch_size, dim}, options)
                        : torch::empty({0}, options);

    // Forward workspace: [tmp_Wx: T*BD] [tmp_Rh: BD]
    const int64_t BD = batch_size * dim;
    Tensor workspace = torch::empty({time_steps * BD + BD}, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "haware_gate_elman_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        HAwareGateElmanForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, gate_mode,
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
            (training && gate_mode == 1) ? ptr<scalar_t>(Rh_cache) : nullptr,
            ptr<scalar_t>(workspace));
    }));

    return {h, output, v, Rh_cache};
}

std::vector<Tensor> haware_gate_elman_backward(
    int gate_mode,
    Tensor W_x,
    Tensor W_h,
    Tensor x,
    Tensor z,
    Tensor h,
    Tensor v,
    Tensor Rh_cache,    // Only used for mode=1
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

    // Workspace: [dv_all: T*BD] [dh: BD] [dh_recurrent: BD] [dRh_gate: BD] [db_float: dim]
    const int64_t BD = batch_size * dim;
    const int64_t float_in_T = (dim * sizeof(float) + sizeof(float) - 1) / sizeof(float);
    const int64_t workspace_size = (time_steps + 3) * BD + float_in_T * 2;
    Tensor workspace = torch::empty({workspace_size}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "haware_gate_elman_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        HAwareGateElmanBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, gate_mode,
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
            (gate_mode == 1 && Rh_cache.numel() > 0) ? ptr<scalar_t>(Rh_cache) : nullptr,
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
// E19: Simplified Gate Elman (remove W_gate, reuse Wx in gate)
// =============================================================================

std::vector<Tensor> simplified_gate_elman_forward(
    bool training,
    int gate_mode,      // 0=A (Wx+h), 1=B (h-only), 2=D (residual+z), 3=E (residual+Wx+h)
    Tensor x,           // [T, B, dim] pre-activated input
    Tensor z,           // [T, B, dim] gate input (only for mode=2)
    Tensor h0,          // [B, dim]
    Tensor W_x,         // [dim, dim]
    Tensor W_h,         // [dim, dim]
    Tensor b,           // [dim]
    Tensor b_gate) {    // [dim] gate bias

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(z);
    CHECK_INPUT(h0);
    CHECK_INPUT(W_x);
    CHECK_INPUT(W_h);
    CHECK_INPUT(b);
    CHECK_INPUT(b_gate);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, dim}, options)
                        : torch::empty({0}, options);
    // Wx_cache needed for modes 0 and 3 (where Wx appears in gate)
    Tensor Wx_cache = (training && (gate_mode == 0 || gate_mode == 3))
                        ? torch::empty({time_steps, batch_size, dim}, options)
                        : torch::empty({0}, options);

    // Forward workspace: [tmp_Wx: T*BD] [tmp_Rh: BD]
    const int64_t BD = batch_size * dim;
    Tensor workspace = torch::empty({time_steps * BD + BD}, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "simplified_gate_elman_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        SimplifiedGateElmanForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, gate_mode,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(W_h),
            ptr<scalar_t>(b),
            ptr<scalar_t>(b_gate),
            ptr<scalar_t>(x),
            (gate_mode == 2) ? ptr<scalar_t>(z) : nullptr,
            ptr<scalar_t>(h),
            ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v) : nullptr,
            (training && (gate_mode == 0 || gate_mode == 3)) ? ptr<scalar_t>(Wx_cache) : nullptr,
            ptr<scalar_t>(workspace));
    }));

    return {h, output, v, Wx_cache};
}

std::vector<Tensor> simplified_gate_elman_backward(
    int gate_mode,
    Tensor W_x,
    Tensor W_h,
    Tensor b_gate,      // [dim] for gradient computation
    Tensor x,
    Tensor z,           // Only used for mode=2
    Tensor h,
    Tensor v,
    Tensor Wx_cache,    // From forward, modes 0,3
    Tensor d_output) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(W_x);
    CHECK_INPUT(W_h);
    CHECK_INPUT(b_gate);
    CHECK_INPUT(x);
    CHECK_INPUT(z);
    CHECK_INPUT(h);
    CHECK_INPUT(v);
    CHECK_INPUT(d_output);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::empty_like(x);
    Tensor dz = (gate_mode == 2) ? torch::empty_like(z) : torch::empty({0}, options);
    Tensor dW_x = torch::zeros({dim, dim}, options);
    Tensor dW_h = torch::zeros({dim, dim}, options);
    Tensor db = torch::zeros({dim}, options);
    Tensor db_gate = (gate_mode != 2) ? torch::zeros({dim}, options) : torch::empty({0}, options);

    // Workspace: [dv_all: T*BD] [dWx_gate: T*BD] [dh: BD] [dh_recurrent: BD] [db_float: dim] [dbg_float: dim]
    const int64_t BD = batch_size * dim;
    // Need 2 * dim floats = 2 * dim * 4 bytes, convert to T-elements (e.g., bfloat16 = 2 bytes)
    // For bfloat16: 2 * dim * 4 / 2 = 4 * dim elements
    // For float32: 2 * dim * 4 / 4 = 2 * dim elements
    // General: ceil(2 * dim * sizeof(float) / sizeof(T))
    const int64_t float_bytes = 2 * dim * sizeof(float);
    const int64_t workspace_size = (2 * time_steps + 2) * BD + float_bytes;  // float_bytes works as element count ceiling for T<=4 bytes
    Tensor workspace = torch::empty({workspace_size}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "simplified_gate_elman_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        SimplifiedGateElmanBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, gate_mode,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(W_h),
            ptr<scalar_t>(b_gate),
            ptr<scalar_t>(x),
            (gate_mode == 2) ? ptr<scalar_t>(z) : nullptr,
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            (gate_mode == 0 || gate_mode == 3) ? ptr<scalar_t>(Wx_cache) : nullptr,
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            (gate_mode == 2) ? ptr<scalar_t>(dz) : nullptr,
            ptr<scalar_t>(dW_x),
            ptr<scalar_t>(dW_h),
            ptr<scalar_t>(db),
            (gate_mode != 2) ? ptr<scalar_t>(db_gate) : nullptr,
            ptr<scalar_t>(workspace));
    }));

    return {dx, dz, dW_x, dW_h, db, db_gate};
}

// =============================================================================
// E21: Structured Elman (MIMO with Nonlinear State Mixing)
// =============================================================================

std::vector<Tensor> structured_elman_forward(
    bool training,
    int nheads,
    int d_state,
    int mimo_rank,
    int nonlinearity_mode,  // 0=silu, 1=tanh, 2=linear
    Tensor B_proj,        // [T, B, nheads, d_state, mimo_rank]
    Tensor X_proj,        // [T, B, nheads, headdim, mimo_rank]
    Tensor alpha_raw,     // [T, B, nheads]
    Tensor alpha_bias,    // [nheads]
    Tensor z,             // [T, B, d_inner]
    Tensor H0) {          // [B, nheads, d_state, headdim]

    const auto time_steps = B_proj.size(0);
    const auto batch_size = B_proj.size(1);
    const auto headdim = X_proj.size(3);
    const auto d_inner = nheads * headdim;

    CHECK_INPUT(B_proj);
    CHECK_INPUT(X_proj);
    CHECK_INPUT(alpha_raw);
    CHECK_INPUT(alpha_bias);
    CHECK_INPUT(z);
    CHECK_INPUT(H0);

    const auto options = B_proj.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    // H: [(T+1), B, nheads, d_state, headdim]
    Tensor H = torch::empty({time_steps + 1, batch_size, nheads, d_state, headdim}, options);
    Tensor output = torch::empty({time_steps, batch_size, d_inner}, options);
    Tensor y_cache = training ? torch::empty({time_steps, batch_size, d_inner}, options)
                              : torch::empty({0}, options);

    // Initialize H0
    H[0] = H0;

    // Minimal workspace
    const int64_t BD = batch_size * d_inner;
    Tensor workspace = torch::empty({BD}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        B_proj.scalar_type(), "structured_elman_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        StructuredElmanForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, nheads, d_state, headdim, mimo_rank, nonlinearity_mode,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(B_proj),
            ptr<scalar_t>(X_proj),
            ptr<scalar_t>(alpha_raw),
            ptr<scalar_t>(alpha_bias),
            ptr<scalar_t>(z),
            ptr<scalar_t>(H),
            ptr<scalar_t>(output),
            training ? ptr<scalar_t>(y_cache) : nullptr,
            ptr<scalar_t>(workspace));
    }));

    return {H, output, y_cache};
}

std::vector<Tensor> structured_elman_backward(
    int nheads,
    int d_state,
    int mimo_rank,
    int nonlinearity_mode,  // 0=silu, 1=tanh, 2=linear
    Tensor B_proj,        // [T, B, nheads, d_state, mimo_rank]
    Tensor X_proj,        // [T, B, nheads, headdim, mimo_rank]
    Tensor alpha_raw,     // [T, B, nheads]
    Tensor alpha_bias,    // [nheads]
    Tensor z,             // [T, B, d_inner]
    Tensor H,             // [(T+1), B, nheads, d_state, headdim]
    Tensor y_cache,       // [T, B, d_inner]
    Tensor d_output) {    // [T, B, d_inner]

    const auto time_steps = B_proj.size(0);
    const auto batch_size = B_proj.size(1);
    const auto headdim = X_proj.size(3);
    const auto d_inner = nheads * headdim;

    CHECK_INPUT(B_proj);
    CHECK_INPUT(X_proj);
    CHECK_INPUT(alpha_raw);
    CHECK_INPUT(alpha_bias);
    CHECK_INPUT(z);
    CHECK_INPUT(H);
    CHECK_INPUT(y_cache);
    CHECK_INPUT(d_output);

    const auto options = B_proj.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dz = torch::empty_like(z);
    Tensor dB_proj = torch::zeros_like(B_proj);
    Tensor dX_proj = torch::zeros_like(X_proj);
    Tensor dalpha_raw = torch::zeros_like(alpha_raw);
    Tensor dalpha_bias = torch::zeros({nheads}, options);

    // Workspace: [BD + B_state + float_grads]
    const int64_t BD = batch_size * d_inner;
    const int64_t B_state = batch_size * nheads * d_state * headdim;
    const int64_t B_proj_size = batch_size * nheads * d_state * mimo_rank;
    const int64_t X_proj_size = batch_size * nheads * headdim * mimo_rank;
    const int64_t alpha_size = batch_size * nheads;
    const int64_t float_size = (B_proj_size + X_proj_size + alpha_size + nheads) * sizeof(float);
    const int64_t float_in_T = (float_size + sizeof(float) - 1) / sizeof(float);
    Tensor workspace = torch::empty({BD + B_state + float_in_T * 2}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        B_proj.scalar_type(), "structured_elman_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        StructuredElmanBackward<typename native_type<scalar_t>::T> backward(
            batch_size, nheads, d_state, headdim, mimo_rank, nonlinearity_mode,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(B_proj),
            ptr<scalar_t>(X_proj),
            ptr<scalar_t>(alpha_raw),
            ptr<scalar_t>(alpha_bias),
            ptr<scalar_t>(z),
            ptr<scalar_t>(H),
            ptr<scalar_t>(y_cache),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dz),
            ptr<scalar_t>(dB_proj),
            ptr<scalar_t>(dX_proj),
            ptr<scalar_t>(dalpha_raw),
            ptr<scalar_t>(dalpha_bias),
            ptr<scalar_t>(workspace));
    }));

    return {dz, dB_proj, dX_proj, dalpha_raw, dalpha_bias};
}


// =============================================================================
// E22: Structured Elman with State Attention (UTM class)
// =============================================================================

std::vector<Tensor> structured_elman_attention_forward(
    bool training,
    int nheads,
    int d_state,
    int mimo_rank,
    int attn_period,
    int attn_dim,
    int nonlinearity_mode,  // 0=silu, 1=tanh, 2=linear
    Tensor B_proj,        // [T, B, nheads, d_state, mimo_rank]
    Tensor X_proj,        // [T, B, nheads, headdim, mimo_rank]
    Tensor alpha_raw,     // [T, B, nheads]
    Tensor alpha_bias,    // [nheads]
    Tensor z,             // [T, B, d_inner]
    Tensor W_q,           // [nheads, headdim, attn_dim]
    Tensor W_k,           // [nheads, headdim, attn_dim]
    Tensor W_v,           // [nheads, headdim, attn_dim]
    Tensor W_o,           // [nheads, attn_dim, headdim]
    Tensor H0) {          // [B, nheads, d_state, headdim]

    const auto time_steps = B_proj.size(0);
    const auto batch_size = B_proj.size(1);
    const auto headdim = X_proj.size(3);
    const auto d_inner = nheads * headdim;

    CHECK_INPUT(B_proj);
    CHECK_INPUT(X_proj);
    CHECK_INPUT(alpha_raw);
    CHECK_INPUT(alpha_bias);
    CHECK_INPUT(z);
    CHECK_INPUT(W_q);
    CHECK_INPUT(W_k);
    CHECK_INPUT(W_v);
    CHECK_INPUT(W_o);
    CHECK_INPUT(H0);

    const auto options = B_proj.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    // Outputs
    Tensor output = torch::empty({time_steps, batch_size, d_inner}, options);
    Tensor H_final = torch::empty({batch_size, nheads, d_state, headdim}, options);

    // For backward: H_all[(T+1), B, H, N, P], y_cache[T, B, d_inner]
    Tensor H_all = training ?
        torch::empty({time_steps + 1, batch_size, nheads, d_state, headdim}, options) :
        torch::empty({0}, options);
    Tensor y_cache = training ?
        torch::empty({time_steps, batch_size, d_inner}, options) :
        torch::empty({0}, options);

    // E22 only supports bfloat16 - check type and use directly
    TORCH_CHECK(B_proj.scalar_type() == at::ScalarType::BFloat16,
                "E22 CUDA kernel only supports bfloat16, got ", B_proj.scalar_type());

    using namespace hasty::v0::elman_ladder;
    StructuredElmanAttentionForward<__nv_bfloat16> forward(
        training, batch_size, nheads, d_state, headdim, mimo_rank,
        attn_period, attn_dim, nonlinearity_mode,
        at::cuda::getCurrentCUDABlasHandle(),
        at::cuda::getCurrentCUDAStream());

    forward.Run(
        time_steps,
        reinterpret_cast<const __nv_bfloat16*>(B_proj.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(X_proj.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(alpha_raw.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(alpha_bias.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(z.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_k.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_v.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_o.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(H0.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(H_final.data_ptr()),
        training ? reinterpret_cast<__nv_bfloat16*>(H_all.data_ptr()) : nullptr,
        training ? reinterpret_cast<__nv_bfloat16*>(y_cache.data_ptr()) : nullptr);

    return {output, H_final, H_all, y_cache};
}


std::vector<Tensor> structured_elman_attention_backward(
    int nheads,
    int d_state,
    int mimo_rank,
    int attn_period,
    int attn_dim,
    int nonlinearity_mode,
    Tensor B_proj,        // [T, B, nheads, d_state, mimo_rank]
    Tensor X_proj,        // [T, B, nheads, headdim, mimo_rank]
    Tensor alpha_raw,     // [T, B, nheads]
    Tensor alpha_bias,    // [nheads]
    Tensor z,             // [T, B, d_inner]
    Tensor W_q,           // [nheads, headdim, attn_dim]
    Tensor W_k,           // [nheads, headdim, attn_dim]
    Tensor W_v,           // [nheads, headdim, attn_dim]
    Tensor W_o,           // [nheads, attn_dim, headdim]
    Tensor H_all,         // [(T+1), B, nheads, d_state, headdim]
    Tensor y_cache,       // [T, B, d_inner]
    Tensor d_output) {    // [T, B, d_inner]

    const auto time_steps = B_proj.size(0);
    const auto batch_size = B_proj.size(1);
    const auto headdim = X_proj.size(3);
    const auto d_inner = nheads * headdim;

    CHECK_INPUT(B_proj);
    CHECK_INPUT(X_proj);
    CHECK_INPUT(alpha_raw);
    CHECK_INPUT(alpha_bias);
    CHECK_INPUT(z);
    CHECK_INPUT(W_q);
    CHECK_INPUT(W_k);
    CHECK_INPUT(W_v);
    CHECK_INPUT(W_o);
    CHECK_INPUT(H_all);
    CHECK_INPUT(y_cache);
    CHECK_INPUT(d_output);

    const auto options = B_proj.options();
    const auto float_options = options.dtype(torch::kFloat32);
    const at::cuda::CUDAGuard guard(options.device_index());

    // Gradient outputs
    Tensor dz = torch::empty_like(z);
    Tensor dB_proj = torch::zeros_like(B_proj);
    Tensor dX_proj = torch::zeros_like(X_proj);
    Tensor dalpha_raw = torch::zeros_like(alpha_raw);

    // Attention weight gradients (fp32 for accumulation)
    Tensor dW_q = torch::zeros({nheads, headdim, attn_dim}, float_options.device(options.device()));
    Tensor dW_k = torch::zeros({nheads, headdim, attn_dim}, float_options.device(options.device()));
    Tensor dW_v = torch::zeros({nheads, headdim, attn_dim}, float_options.device(options.device()));
    Tensor dW_o = torch::zeros({nheads, attn_dim, headdim}, float_options.device(options.device()));

    // E22 only supports bfloat16
    TORCH_CHECK(B_proj.scalar_type() == at::ScalarType::BFloat16,
                "E22 CUDA kernel only supports bfloat16, got ", B_proj.scalar_type());

    using namespace hasty::v0::elman_ladder;
    StructuredElmanAttentionBackward<__nv_bfloat16> backward(
        batch_size, nheads, d_state, headdim, mimo_rank,
        attn_period, attn_dim, nonlinearity_mode,
        at::cuda::getCurrentCUDABlasHandle(),
        at::cuda::getCurrentCUDAStream());

    backward.Run(
        time_steps,
        reinterpret_cast<const __nv_bfloat16*>(B_proj.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(X_proj.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(alpha_raw.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(alpha_bias.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(z.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_k.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_v.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_o.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(H_all.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(y_cache.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(d_output.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dB_proj.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dX_proj.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dalpha_raw.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dz.data_ptr()),
        dW_q.data_ptr<float>(),
        dW_k.data_ptr<float>(),
        dW_v.data_ptr<float>(),
        dW_o.data_ptr<float>());

    // Convert fp32 gradients back to input dtype
    dW_q = dW_q.to(options.dtype());
    dW_k = dW_k.to(options.dtype());
    dW_v = dW_v.to(options.dtype());
    dW_o = dW_o.to(options.dtype());

    return {dz, dB_proj, dX_proj, dalpha_raw, dW_q, dW_k, dW_v, dW_o};
}


// =============================================================================
// E23: Dual-Memory Elman (Tape + Working Memory)
// =============================================================================

std::vector<Tensor> dual_memory_elman_forward(
    bool training,
    Tensor x,              // [B, T, D] - raw input
    Tensor h_tape_init,    // [B, N, D] - initial tape state
    Tensor h_work_init,    // [B, D] - initial working memory
    Tensor W_h,            // [D, D]
    Tensor W_x,            // [D, D]
    Tensor b_h,            // [D]
    Tensor W_write) {      // [D, D]

    const auto batch_size = x.size(0);
    const auto seq_len = x.size(1);
    const auto dim = x.size(2);
    const auto n_slots = h_tape_init.size(1);

    CHECK_INPUT(x);
    CHECK_INPUT(h_tape_init);
    CHECK_INPUT(h_work_init);
    CHECK_INPUT(W_h);
    CHECK_INPUT(W_x);
    CHECK_INPUT(b_h);
    CHECK_INPUT(W_write);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Transpose x from [B, T, D] to [T, B, D] for efficient batch GEMM
    auto x_t = x.permute({1, 0, 2}).contiguous();  // [T, B, D]

    // Pre-compute x_proj = x @ W_x^T for ALL timesteps in ONE GEMM
    // x_t is [T, B, D] = [T*B, D] in memory
    // Result x_proj is [T, B, D] = [T*B, D]
    auto x_proj = torch::empty({seq_len, batch_size, dim}, options);

    const float alpha = 1.0f;
    const float beta_zero = 0.0f;
    cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim, seq_len * batch_size, dim,
        &alpha,
        W_x.data_ptr(), CUDA_R_16BF, dim,
        x_t.data_ptr(), CUDA_R_16BF, dim,
        &beta_zero,
        x_proj.data_ptr(), CUDA_R_16BF, dim,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
    );

    // Allocate outputs - kernel outputs [T, B, ...], we transpose to [B, T, ...] before returning
    auto h_work_out = torch::empty({seq_len, batch_size, dim}, options);
    auto h_tape_final = torch::empty({batch_size, n_slots, dim}, options);
    auto read_attn = torch::empty({seq_len, batch_size, n_slots}, options);
    auto write_attn = torch::empty({seq_len, batch_size, n_slots}, options);

    // Workspace: tmp_Rh [B, D] + tmp_write_val [B, D]
    auto workspace = torch::empty({batch_size * dim * 2}, options);

    // For training, save h_tape_all for backward
    auto h_tape_all = training ?
        torch::empty({seq_len + 1, batch_size, n_slots, dim}, options) :
        torch::empty({0}, options);

    hasty::v0::elman_ladder::DualMemoryElmanForward<__nv_bfloat16> forward_op(
        training, batch_size, n_slots, dim, handle, stream);

    forward_op.Run(
        seq_len,
        reinterpret_cast<const __nv_bfloat16*>(x_proj.data_ptr()),  // Pre-computed x @ W_x^T
        reinterpret_cast<const __nv_bfloat16*>(W_h.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(b_h.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_write.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h_tape_init.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h_work_init.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(h_work_out.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(h_tape_final.data_ptr()),
        training ? reinterpret_cast<__nv_bfloat16*>(h_tape_all.data_ptr()) : nullptr,
        reinterpret_cast<__nv_bfloat16*>(read_attn.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(write_attn.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(workspace.data_ptr()));

    // Transpose from [T, B, ...] to [B, T, ...] for Python/backward
    return {h_work_out.permute({1, 0, 2}).contiguous(),
            h_tape_final, h_tape_all,
            read_attn.permute({1, 0, 2}).contiguous(),
            write_attn.permute({1, 0, 2}).contiguous()};
}

std::vector<Tensor> dual_memory_elman_backward(
    Tensor x_proj,           // [B, T, D] - not used, kept for API compat
    Tensor h_work_all,       // [B, T, D]
    Tensor h_work_init,      // [B, D] - initial working memory (for t=0)
    Tensor h_tape_all,       // [T+1, B, N, D]
    Tensor read_attn,        // [B, T, N]
    Tensor write_attn,       // [B, T, N]
    Tensor W_h,              // [D, D]
    Tensor W_write,          // [D, D]
    Tensor d_h_work_out,     // [B, T, D]
    Tensor d_h_tape_final) { // [B, N, D]

    const auto batch_size = h_work_all.size(0);
    const auto seq_len = h_work_all.size(1);
    const auto dim = h_work_all.size(2);
    const auto n_slots = h_tape_all.size(2);

    CHECK_INPUT(h_work_all);
    CHECK_INPUT(h_work_init);
    CHECK_INPUT(h_tape_all);
    CHECK_INPUT(read_attn);
    CHECK_INPUT(write_attn);
    CHECK_INPUT(W_h);
    CHECK_INPUT(W_write);
    CHECK_INPUT(d_h_work_out);
    CHECK_INPUT(d_h_tape_final);

    const auto options = h_work_all.options();
    const auto float_options = options.dtype(torch::kFloat32);
    const at::cuda::CUDAGuard guard(options.device_index());
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Transpose inputs from [B, T, ...] to [T, B, ...] for kernel
    auto h_work_t = h_work_all.permute({1, 0, 2}).contiguous();  // [T, B, D]
    auto read_attn_t = read_attn.permute({1, 0, 2}).contiguous();  // [T, B, N]
    auto write_attn_t = write_attn.permute({1, 0, 2}).contiguous();  // [T, B, N]
    auto d_h_work_out_t = d_h_work_out.permute({1, 0, 2}).contiguous();  // [T, B, D]

    // Allocate gradient outputs in [T, B, D] format
    auto dx_proj_t = torch::empty({seq_len, batch_size, dim}, options);
    auto d_pre_act_all = torch::empty({seq_len, batch_size, dim}, options);
    auto d_write_val_all = torch::empty({seq_len, batch_size, dim}, options);
    auto dW_h = torch::zeros({dim, dim}, float_options);
    auto db_h = torch::zeros({dim}, float_options);
    auto dW_write = torch::zeros({dim, dim}, float_options);

    // Scratch buffer for tape gradients
    auto d_h_tape = torch::empty({batch_size, n_slots, dim}, options);

    hasty::v0::elman_ladder::DualMemoryElmanBackward<__nv_bfloat16> backward_op(
        batch_size, n_slots, dim, handle, stream);

    backward_op.Run(
        seq_len,
        reinterpret_cast<const __nv_bfloat16*>(h_work_t.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h_work_init.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h_tape_all.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(read_attn_t.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(write_attn_t.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_h.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_write.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(d_h_work_out_t.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(d_h_tape_final.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dx_proj_t.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(d_pre_act_all.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(d_write_val_all.data_ptr()),
        db_h.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(d_h_tape.data_ptr()),
        dW_h.data_ptr<float>(),
        dW_write.data_ptr<float>());

    // Transpose dx_proj back to [B, T, D]
    auto dx_proj = dx_proj_t.permute({1, 0, 2}).contiguous();

    // Convert weight gradients back to original dtype
    dW_h = dW_h.to(options.dtype());
    db_h = db_h.to(options.dtype());
    dW_write = dW_write.to(options.dtype());

    return {dx_proj, dW_h, db_h, dW_write};
}

// E23 Optimized forward using cuBLAS strided batched GEMM for attention
std::vector<Tensor> dual_memory_elman_forward_opt(
    bool training,
    Tensor x,              // [B, T, D] - raw input
    Tensor h_tape_init,    // [B, N, D] - initial tape state
    Tensor h_work_init,    // [B, D] - initial working memory
    Tensor W_h,            // [D, D]
    Tensor W_x,            // [D, D]
    Tensor b_h,            // [D]
    Tensor W_write) {      // [D, D]

    const auto batch_size = x.size(0);
    const auto seq_len = x.size(1);
    const auto dim = x.size(2);
    const auto n_slots = h_tape_init.size(1);

    CHECK_INPUT(x);
    CHECK_INPUT(h_tape_init);
    CHECK_INPUT(h_work_init);
    CHECK_INPUT(W_h);
    CHECK_INPUT(W_x);
    CHECK_INPUT(b_h);
    CHECK_INPUT(W_write);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Transpose x from [B, T, D] to [T, B, D] for efficient batch GEMM
    auto x_t = x.permute({1, 0, 2}).contiguous();  // [T, B, D]

    // Pre-compute x_proj = x @ W_x^T for ALL timesteps in ONE GEMM
    auto x_proj = torch::empty({seq_len, batch_size, dim}, options);

    const float alpha = 1.0f;
    const float beta_zero = 0.0f;
    cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim, seq_len * batch_size, dim,
        &alpha,
        W_x.data_ptr(), CUDA_R_16BF, dim,
        x_t.data_ptr(), CUDA_R_16BF, dim,
        &beta_zero,
        x_proj.data_ptr(), CUDA_R_16BF, dim,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
    );

    // Allocate outputs
    auto h_work_out = torch::empty({seq_len, batch_size, dim}, options);
    auto h_tape_final = torch::empty({batch_size, n_slots, dim}, options);
    auto read_attn = torch::empty({seq_len, batch_size, n_slots}, options);
    auto write_attn = torch::empty({seq_len, batch_size, n_slots}, options);

    // Workspace: tmp_Rh [B, D] + tmp_write_val [B, D] + tmp_scores [B, N] + tmp_read [B, D]
    auto workspace = torch::empty({batch_size * (dim * 3 + n_slots)}, options);

    // For training, save h_tape_all for backward
    auto h_tape_all = training ?
        torch::empty({seq_len + 1, batch_size, n_slots, dim}, options) :
        torch::empty({0}, options);

    hasty::v0::elman_ladder::DualMemoryElmanForwardOpt<__nv_bfloat16> forward_op(
        training, batch_size, n_slots, dim, handle, stream);

    forward_op.Run(
        seq_len,
        reinterpret_cast<const __nv_bfloat16*>(x_proj.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_h.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(b_h.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_write.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h_tape_init.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h_work_init.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(h_work_out.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(h_tape_final.data_ptr()),
        training ? reinterpret_cast<__nv_bfloat16*>(h_tape_all.data_ptr()) : nullptr,
        reinterpret_cast<__nv_bfloat16*>(read_attn.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(write_attn.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(workspace.data_ptr()));

    // Transpose from [T, B, ...] to [B, T, ...] for Python/backward
    return {h_work_out.permute({1, 0, 2}).contiguous(),
            h_tape_final, h_tape_all,
            read_attn.permute({1, 0, 2}).contiguous(),
            write_attn.permute({1, 0, 2}).contiguous()};
}

// =============================================================================
// E23c: Chunked Dual-Memory Elman (Batched Attention Ops)
// Key architectural change from E23:
//   h_work_t = tanh(W_h @ h_work_{t-1} + W_x @ x_t + b)  -- NO read dependency!
//   output_t = h_work_t + read_t                         -- Additive read
// =============================================================================

std::vector<Tensor> e23c_chunked_forward(
    bool training,
    Tensor x,              // [B, T, D] - raw input
    Tensor h_tape_init,    // [B, N, D] - initial tape state
    Tensor h_work_init,    // [B, D] - initial working memory
    Tensor W_h,            // [D, D]
    Tensor W_x,            // [D, D]
    Tensor b_h,            // [D]
    Tensor W_write,        // [D, D]
    int chunk_size) {      // K - chunk size for batching

    const auto batch_size = x.size(0);
    const auto seq_len = x.size(1);
    const auto dim = x.size(2);
    const auto n_slots = h_tape_init.size(1);

    CHECK_INPUT(x);
    CHECK_INPUT(h_tape_init);
    CHECK_INPUT(h_work_init);
    CHECK_INPUT(W_h);
    CHECK_INPUT(W_x);
    CHECK_INPUT(b_h);
    CHECK_INPUT(W_write);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Transpose x from [B, T, D] to [T, B, D] for efficient batch GEMM
    auto x_t = x.permute({1, 0, 2}).contiguous();  // [T, B, D]

    // Pre-compute x_proj = x @ W_x^T for ALL timesteps in ONE GEMM
    auto x_proj = torch::empty({seq_len, batch_size, dim}, options);

    const float alpha = 1.0f;
    const float beta_zero = 0.0f;
    cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim, seq_len * batch_size, dim,
        &alpha,
        W_x.data_ptr(), CUDA_R_16BF, dim,
        x_t.data_ptr(), CUDA_R_16BF, dim,
        &beta_zero,
        x_proj.data_ptr(), CUDA_R_16BF, dim,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
    );

    // Allocate outputs
    auto output = torch::empty({seq_len, batch_size, dim}, options);
    auto h_tape_final = torch::empty({batch_size, n_slots, dim}, options);
    auto h_work_all = torch::empty({seq_len, batch_size, dim}, options);

    // Workspace layout (per chunk K):
    // tmp_Rh: [B, D]
    // tmp_h_chunk: [K, B, D]
    // tmp_read_scores: [B, K, N]
    // tmp_read_attn: [B, K, N]
    // tmp_read_vals: [K, B, D]
    // tmp_write_vals: [K, B, D]
    // tmp_write_attn: [K, B, N]
    // tmp_cumprod: [K, B, N]
    int64_t K = chunk_size;
    int64_t BD = batch_size * dim;
    int64_t BN = batch_size * n_slots;
    int64_t workspace_size = BD + K * BD + 2 * batch_size * K * n_slots +
                             K * BD + K * BD + K * BN + K * BN;
    auto workspace = torch::empty({workspace_size}, options);

    hasty::v0::elman_ladder::E23cChunkedForward<__nv_bfloat16> forward_op(
        training, batch_size, n_slots, dim, handle, stream);

    forward_op.Run(
        seq_len,
        chunk_size,
        reinterpret_cast<const __nv_bfloat16*>(x_proj.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_h.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(b_h.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_write.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h_tape_init.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h_work_init.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(h_tape_final.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(h_work_all.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(workspace.data_ptr()));

    // Transpose outputs to [B, T, ...] format
    return {output.permute({1, 0, 2}).contiguous(),      // [B, T, D] output
            h_tape_final,                                 // [B, N, D]
            h_work_all.permute({1, 0, 2}).contiguous()};  // [B, T, D] h_work states
}

// =============================================================================
// E23c_v2: Chunked Dual-Memory Elman with Read Feedback
// =============================================================================

std::vector<Tensor> e23cv2_chunked_forward(
    bool training,
    Tensor x,              // [B, T, D] - raw input
    Tensor h_tape_init,    // [B, N, D] - initial tape state
    Tensor h_work_init,    // [B, D] - initial working memory
    Tensor W_h,            // [D, D]
    Tensor W_x,            // [D, D]
    Tensor W_r,            // [D, D] - NEW: read projection
    Tensor b_h,            // [D]
    Tensor W_write,        // [D, D]
    int chunk_size) {      // K - chunk size for batching

    const auto batch_size = x.size(0);
    const auto seq_len = x.size(1);
    const auto dim = x.size(2);
    const auto n_slots = h_tape_init.size(1);

    CHECK_INPUT(x);
    CHECK_INPUT(h_tape_init);
    CHECK_INPUT(h_work_init);
    CHECK_INPUT(W_h);
    CHECK_INPUT(W_x);
    CHECK_INPUT(W_r);
    CHECK_INPUT(b_h);
    CHECK_INPUT(W_write);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Transpose x from [B, T, D] to [T, B, D]
    auto x_t = x.permute({1, 0, 2}).contiguous();  // [T, B, D]

    // Pre-compute x_proj = x @ W_x^T for ALL timesteps in ONE GEMM
    auto x_proj = torch::empty({seq_len, batch_size, dim}, options);

    const float alpha = 1.0f;
    const float beta_zero = 0.0f;
    cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim, seq_len * batch_size, dim,
        &alpha,
        W_x.data_ptr(), CUDA_R_16BF, dim,
        x_t.data_ptr(), CUDA_R_16BF, dim,
        &beta_zero,
        x_proj.data_ptr(), CUDA_R_16BF, dim,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
    );

    // Allocate outputs
    auto output = torch::empty({seq_len, batch_size, dim}, options);
    auto h_tape_final = torch::empty({batch_size, n_slots, dim}, options);
    auto h_work_all = torch::empty({seq_len, batch_size, dim}, options);

    // Workspace layout (per chunk K):
    // tmp_Rh: [B, D]
    // tmp_Rr: [B, D]
    // tmp_scores: [B, N]
    // tmp_attn: [B, N]
    // tmp_read: [B, D]
    // tmp_h_chunk: [B, K, D]
    // tmp_attn_chunk: [B, K, N]
    // tmp_write_vals: [B, K, D]
    // tmp_cumprod: [B, K, N]
    int64_t K = chunk_size;
    int64_t BD = batch_size * dim;
    int64_t BN = batch_size * n_slots;
    int64_t workspace_size = BD + BD + BN + BN + BD +
                             K * BD + K * BN + K * BD + K * BN;
    auto workspace = torch::empty({workspace_size}, options);

    hasty::v0::elman_ladder::E23cv2ChunkedForward<__nv_bfloat16> forward_op(
        training, batch_size, n_slots, dim, handle, stream);

    forward_op.Run(
        seq_len,
        chunk_size,
        reinterpret_cast<const __nv_bfloat16*>(x_proj.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_h.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_r.data_ptr()),  // NEW
        reinterpret_cast<const __nv_bfloat16*>(b_h.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_write.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h_tape_init.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h_work_init.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(h_tape_final.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(h_work_all.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(workspace.data_ptr()));

    // Transpose outputs to [B, T, ...] format
    return {output.permute({1, 0, 2}).contiguous(),      // [B, T, D] output
            h_tape_final,                                 // [B, N, D]
            h_work_all.permute({1, 0, 2}).contiguous()};  // [B, T, D] h_work states
}

// =============================================================================
// E24: Single-GEMM Dual Memory (1 GEMM per timestep)
// =============================================================================

std::vector<Tensor> e24_single_gemm_forward(
    bool training,
    Tensor x,              // [B, T, D] - raw input
    Tensor h_tape_init,    // [B, N, D] - initial tape state
    Tensor h_work_init,    // [B, D] - initial working memory
    Tensor W_all,          // [2D, 2D] - fused weight matrix
    Tensor b_h) {          // [D]

    const auto batch_size = x.size(0);
    const auto seq_len = x.size(1);
    const auto dim = x.size(2);
    const auto n_slots = h_tape_init.size(1);

    CHECK_INPUT(x);
    CHECK_INPUT(h_tape_init);
    CHECK_INPUT(h_work_init);
    CHECK_INPUT(W_all);
    CHECK_INPUT(b_h);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Transpose x from [B, T, D] to [T, B, D] for efficient processing
    auto x_t = x.permute({1, 0, 2}).contiguous();  // [T, B, D]

    // Allocate outputs - kernel outputs [T, B, ...], we transpose to [B, T, ...] before returning
    auto h_work_out = torch::empty({seq_len, batch_size, dim}, options);
    auto h_tape_final = torch::empty({batch_size, n_slots, dim}, options);
    auto read_attn = torch::empty({seq_len, batch_size, n_slots}, options);
    auto write_attn = torch::empty({seq_len, batch_size, n_slots}, options);
    auto write_val_all = torch::empty({seq_len, batch_size, dim}, options);

    // Workspace: input_concat [B, 2D] + gemm_output [B, 2D]
    auto workspace = torch::empty({batch_size * dim * 4}, options);

    // For training, save h_tape_all for backward
    auto h_tape_all = training ?
        torch::empty({seq_len + 1, batch_size, n_slots, dim}, options) :
        torch::empty({0}, options);

    hasty::v0::elman_ladder::E24SingleGemmForward<__nv_bfloat16> forward_op(
        training, batch_size, n_slots, dim, handle, stream);

    forward_op.Run(
        seq_len,
        reinterpret_cast<const __nv_bfloat16*>(x_t.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_all.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(b_h.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h_tape_init.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h_work_init.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(h_work_out.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(h_tape_final.data_ptr()),
        training ? reinterpret_cast<__nv_bfloat16*>(h_tape_all.data_ptr()) : nullptr,
        reinterpret_cast<__nv_bfloat16*>(read_attn.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(write_attn.data_ptr()),
        training ? reinterpret_cast<__nv_bfloat16*>(write_val_all.data_ptr()) : nullptr,
        reinterpret_cast<__nv_bfloat16*>(workspace.data_ptr()));

    // Transpose outputs to [B, T, ...] format
    return {h_work_out.permute({1, 0, 2}).contiguous(),  // [B, T, D]
            h_tape_final,                                 // [B, N, D]
            h_tape_all,                                   // [T+1, B, N, D]
            read_attn.permute({1, 0, 2}).contiguous(),    // [B, T, N]
            write_attn.permute({1, 0, 2}).contiguous(),   // [B, T, N]
            write_val_all.permute({1, 0, 2}).contiguous()};  // [B, T, D]
}

std::vector<Tensor> e24_single_gemm_backward(
    Tensor x,                // [B, T, D]
    Tensor h_work_all,       // [B, T, D]
    Tensor h_work_init,      // [B, D]
    Tensor h_tape_all,       // [T+1, B, N, D]
    Tensor read_attn,        // [B, T, N]
    Tensor write_attn,       // [B, T, N]
    Tensor write_val_all,    // [B, T, D]
    Tensor W_all,            // [2D, 2D]
    Tensor d_h_work_out,     // [B, T, D]
    Tensor d_h_tape_final) { // [B, N, D]

    const auto batch_size = h_work_all.size(0);
    const auto seq_len = h_work_all.size(1);
    const auto dim = h_work_all.size(2);
    const auto n_slots = h_tape_all.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(h_work_all);
    CHECK_INPUT(h_work_init);
    CHECK_INPUT(h_tape_all);
    CHECK_INPUT(read_attn);
    CHECK_INPUT(write_attn);
    CHECK_INPUT(write_val_all);
    CHECK_INPUT(W_all);
    CHECK_INPUT(d_h_work_out);
    CHECK_INPUT(d_h_tape_final);

    const auto options = h_work_all.options();
    const auto float_options = options.dtype(torch::kFloat32);
    const at::cuda::CUDAGuard guard(options.device_index());
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Transpose inputs from [B, T, ...] to [T, B, ...] for kernel
    auto x_t = x.permute({1, 0, 2}).contiguous();  // [T, B, D]
    auto h_work_t = h_work_all.permute({1, 0, 2}).contiguous();  // [T, B, D]
    auto read_attn_t = read_attn.permute({1, 0, 2}).contiguous();  // [T, B, N]
    auto write_attn_t = write_attn.permute({1, 0, 2}).contiguous();  // [T, B, N]
    auto write_val_t = write_val_all.permute({1, 0, 2}).contiguous();  // [T, B, D]
    auto d_h_work_out_t = d_h_work_out.permute({1, 0, 2}).contiguous();  // [T, B, D]

    // Allocate gradient outputs
    auto dx_t = torch::empty({seq_len, batch_size, dim}, options);
    auto dW_all = torch::zeros({2 * dim, 2 * dim}, float_options);
    auto db_h = torch::zeros({dim}, float_options);

    // OPTIMIZED Workspace layout:
    // d_h_tape: [B, N, D]
    // d_h_work: [B, D] - accumulated gradient to h_work_prev
    // d_h_work_from_read: [B, D] - gradient from read attention
    // d_gemm_output_all: [T, B, 2D] - stored for final dW_all GEMM
    // input_concat_all: [T, B, 2D] - stored for final dW_all GEMM
    // d_input_concat: [B, 2D] - temporary for per-timestep GEMM
    size_t workspace_size = batch_size * n_slots * dim +     // d_h_tape
                           batch_size * dim +                // d_h_work
                           batch_size * dim +                // d_h_work_from_read
                           seq_len * batch_size * 2 * dim +  // d_gemm_output_all
                           seq_len * batch_size * 2 * dim +  // input_concat_all
                           batch_size * 2 * dim;             // d_input_concat
    auto workspace = torch::empty({(int64_t)workspace_size}, options);

    hasty::v0::elman_ladder::E24SingleGemmBackward<__nv_bfloat16> backward_op(
        batch_size, n_slots, dim, handle, stream);

    backward_op.Run(
        seq_len,
        reinterpret_cast<const __nv_bfloat16*>(x_t.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h_work_t.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h_work_init.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h_tape_all.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(read_attn_t.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(write_attn_t.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(write_val_t.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_all.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(d_h_work_out_t.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(d_h_tape_final.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dx_t.data_ptr()),
        db_h.data_ptr<float>(),
        dW_all.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(workspace.data_ptr()));

    // Transpose dx back to [B, T, D]
    auto dx = dx_t.permute({1, 0, 2}).contiguous();

    // Convert weight gradients back to original dtype
    dW_all = dW_all.to(options.dtype());
    db_h = db_h.to(options.dtype());

    return {dx, dW_all, db_h};
}

// =============================================================================
// E25: Dual-Memory Elman with 1.5-Entmax Attention (Sparse Attention)
// Same interface as E23 but uses 1.5-entmax instead of softmax.
// =============================================================================

std::vector<Tensor> e25_entmax_forward(
    bool training,
    Tensor x,              // [B, T, D] - raw input
    Tensor h_tape_init,    // [B, N, D] - initial tape state
    Tensor h_work_init,    // [B, D] - initial working memory
    Tensor W_h,            // [D, D]
    Tensor W_x,            // [D, D]
    Tensor b_h,            // [D]
    Tensor W_write) {      // [D, D]

    const auto batch_size = x.size(0);
    const auto seq_len = x.size(1);
    const auto dim = x.size(2);
    const auto n_slots = h_tape_init.size(1);

    CHECK_INPUT(x);
    CHECK_INPUT(h_tape_init);
    CHECK_INPUT(h_work_init);
    CHECK_INPUT(W_h);
    CHECK_INPUT(W_x);
    CHECK_INPUT(b_h);
    CHECK_INPUT(W_write);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Transpose x from [B, T, D] to [T, B, D] for efficient batch GEMM
    auto x_t = x.permute({1, 0, 2}).contiguous();  // [T, B, D]

    // Pre-compute x_proj = x @ W_x^T for ALL timesteps in ONE GEMM
    auto x_proj = torch::empty({seq_len, batch_size, dim}, options);

    const float alpha = 1.0f;
    const float beta_zero = 0.0f;
    cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim, seq_len * batch_size, dim,
        &alpha,
        W_x.data_ptr(), CUDA_R_16BF, dim,
        x_t.data_ptr(), CUDA_R_16BF, dim,
        &beta_zero,
        x_proj.data_ptr(), CUDA_R_16BF, dim,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
    );

    // Allocate outputs
    auto h_work_out = torch::empty({seq_len, batch_size, dim}, options);
    auto h_tape_final = torch::empty({batch_size, n_slots, dim}, options);
    auto read_attn = torch::empty({seq_len, batch_size, n_slots}, options);
    auto write_attn = torch::empty({seq_len, batch_size, n_slots}, options);

    // For training, save h_tape_all for backward
    auto h_tape_all = training ?
        torch::empty({seq_len + 1, batch_size, n_slots, dim}, options) :
        torch::empty({0}, options);

    // Workspace: tmp_Rh [B, D] + tmp_write_val [B, D]
    auto workspace = torch::empty({batch_size * dim * 2}, options);

    hasty::v0::elman_ladder::E25EntmaxForward<__nv_bfloat16> forward_op(
        training, batch_size, n_slots, dim, handle, stream);

    forward_op.Run(
        seq_len,
        reinterpret_cast<const __nv_bfloat16*>(x_proj.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_h.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(b_h.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_write.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h_tape_init.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h_work_init.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(h_work_out.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(h_tape_final.data_ptr()),
        training ? reinterpret_cast<__nv_bfloat16*>(h_tape_all.data_ptr()) : nullptr,
        reinterpret_cast<__nv_bfloat16*>(read_attn.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(write_attn.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(workspace.data_ptr()));

    // Transpose outputs to [B, T, ...] format
    return {h_work_out.permute({1, 0, 2}).contiguous(),  // [B, T, D] h_work states (=output)
            h_tape_final,                                  // [B, N, D]
            h_tape_all,                                    // [T+1, B, N, D] or empty
            read_attn.permute({1, 0, 2}).contiguous(),   // [B, T, N]
            write_attn.permute({1, 0, 2}).contiguous()}; // [B, T, N]
}

std::vector<Tensor> e25_entmax_backward(
    Tensor x,              // [B, T, D]
    Tensor h_work_all,     // [B, T, D]
    Tensor h_work_init,    // [B, D]
    Tensor h_tape_all,     // [T+1, B, N, D]
    Tensor read_attn,      // [B, T, N]
    Tensor write_attn,     // [B, T, N]
    Tensor W_h,            // [D, D]
    Tensor W_x,            // [D, D]
    Tensor W_write,        // [D, D]
    Tensor d_h_work_out,   // [B, T, D]
    Tensor d_h_tape_final) { // [B, N, D]

    const auto batch_size = h_work_all.size(0);
    const auto seq_len = h_work_all.size(1);
    const auto dim = h_work_all.size(2);
    const auto n_slots = h_tape_all.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(h_work_all);
    CHECK_INPUT(h_work_init);
    CHECK_INPUT(h_tape_all);
    CHECK_INPUT(read_attn);
    CHECK_INPUT(write_attn);
    CHECK_INPUT(W_h);
    CHECK_INPUT(W_x);
    CHECK_INPUT(W_write);
    CHECK_INPUT(d_h_work_out);
    CHECK_INPUT(d_h_tape_final);

    const auto options = h_work_all.options();
    const auto float_options = options.dtype(torch::kFloat32);
    const at::cuda::CUDAGuard guard(options.device_index());
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Transpose inputs from [B, T, ...] to [T, B, ...] for kernel
    auto h_work_t = h_work_all.permute({1, 0, 2}).contiguous();  // [T, B, D]
    auto read_attn_t = read_attn.permute({1, 0, 2}).contiguous();  // [T, B, N]
    auto write_attn_t = write_attn.permute({1, 0, 2}).contiguous();  // [T, B, N]
    auto d_h_work_out_t = d_h_work_out.permute({1, 0, 2}).contiguous();  // [T, B, D]

    // Allocate gradient outputs
    auto dx_proj = torch::empty({seq_len, batch_size, dim}, options);
    auto d_pre_act_all = torch::empty({seq_len, batch_size, dim}, options);
    auto d_write_val_all = torch::empty({seq_len, batch_size, dim}, options);
    auto db_h = torch::zeros({dim}, float_options);
    auto d_h_tape = torch::empty({batch_size, n_slots, dim}, options);
    auto dW_h = torch::zeros({dim, dim}, float_options);
    auto dW_write = torch::zeros({dim, dim}, float_options);

    hasty::v0::elman_ladder::E25EntmaxBackward<__nv_bfloat16> backward_op(
        batch_size, n_slots, dim, handle, stream);

    backward_op.Run(
        seq_len,
        reinterpret_cast<const __nv_bfloat16*>(h_work_t.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h_work_init.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h_tape_all.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(read_attn_t.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(write_attn_t.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_h.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_write.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(d_h_work_out_t.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(d_h_tape_final.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dx_proj.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(d_pre_act_all.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(d_write_val_all.data_ptr()),
        db_h.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(d_h_tape.data_ptr()),
        dW_h.data_ptr<float>(),
        dW_write.data_ptr<float>());

    // Compute dW_x = x^T @ d_pre_act (accumulated over all timesteps)
    auto x_t = x.permute({1, 0, 2}).contiguous();  // [T, B, D]
    auto dW_x = torch::zeros({dim, dim}, float_options);

    // Batch GEMM for dW_x
    const float alpha = 1.0f;
    const float beta_one = 1.0f;
    const __nv_bfloat16* d_pre_act_ptr = reinterpret_cast<const __nv_bfloat16*>(d_pre_act_all.data_ptr());
    const __nv_bfloat16* x_t_ptr = reinterpret_cast<const __nv_bfloat16*>(x_t.data_ptr());
    for (int t = 0; t < seq_len; ++t) {
        cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            dim, dim, batch_size,
            &alpha,
            d_pre_act_ptr + t * batch_size * dim, CUDA_R_16BF, dim,
            x_t_ptr + t * batch_size * dim, CUDA_R_16BF, dim,
            &beta_one,
            dW_x.data_ptr<float>(), CUDA_R_32F, dim,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
        );
    }

    // Compute dx = d_pre_act @ W_x (row-major interpretation)
    // In cuBLAS (column-major): we need W_x @ d_pre_act (NO transpose)
    // d_pre_act viewed as [D, T*B] col-major, W_x is [D, D]
    // Result dx is [D, T*B] col-major = [T*B, D] row-major
    auto dx_t = torch::empty({seq_len, batch_size, dim}, options);
    const float beta_zero = 0.0f;
    cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,  // No transpose needed
        dim, seq_len * batch_size, dim,
        &alpha,
        W_x.data_ptr(), CUDA_R_16BF, dim,
        d_pre_act_all.data_ptr(), CUDA_R_16BF, dim,
        &beta_zero,
        dx_t.data_ptr(), CUDA_R_16BF, dim,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
    );

    // Transpose dx back to [B, T, D]
    auto dx = dx_t.permute({1, 0, 2}).contiguous();

    // Convert weight gradients back to original dtype
    dW_h = dW_h.to(options.dtype());
    dW_x = dW_x.to(options.dtype());
    dW_write = dW_write.to(options.dtype());
    db_h = db_h.to(options.dtype());

    return {dx, dW_h, dW_x, db_h, dW_write};
}

// =============================================================================
// E26: Parallel Dual-Memory Elman (softmax attention)
// Same interface as E25 but uses softmax instead of entmax
// =============================================================================

std::vector<Tensor> e26_parallel_forward(
    bool training,
    Tensor x,              // [B, T, D] - raw input
    Tensor h_tape_init,    // [B, N, D] - initial tape state
    Tensor h_work_init,    // [B, D] - initial working memory
    Tensor W_h,            // [D, D]
    Tensor W_x,            // [D, D]
    Tensor b_h,            // [D]
    Tensor W_write) {      // [D, D]

    const auto batch_size = x.size(0);
    const auto seq_len = x.size(1);
    const auto dim = x.size(2);
    const auto n_slots = h_tape_init.size(1);

    CHECK_INPUT(x);
    CHECK_INPUT(h_tape_init);
    CHECK_INPUT(h_work_init);
    CHECK_INPUT(W_h);
    CHECK_INPUT(W_x);
    CHECK_INPUT(b_h);
    CHECK_INPUT(W_write);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Transpose x from [B, T, D] to [T, B, D] for efficient batch GEMM
    auto x_t = x.permute({1, 0, 2}).contiguous();  // [T, B, D]

    // PARALLEL PHASE: Pre-compute x_proj = x @ W_x^T for ALL timesteps in ONE GEMM
    auto x_proj = torch::empty({seq_len, batch_size, dim}, options);

    const float alpha = 1.0f;
    const float beta_zero = 0.0f;
    cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim, seq_len * batch_size, dim,
        &alpha,
        W_x.data_ptr(), CUDA_R_16BF, dim,
        x_t.data_ptr(), CUDA_R_16BF, dim,
        &beta_zero,
        x_proj.data_ptr(), CUDA_R_16BF, dim,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
    );

    // Allocate outputs
    auto h_work_out = torch::empty({seq_len, batch_size, dim}, options);
    auto h_tape_final = torch::empty({batch_size, n_slots, dim}, options);
    auto read_attn = torch::empty({seq_len, batch_size, n_slots}, options);
    auto write_attn = torch::empty({seq_len, batch_size, n_slots}, options);

    // For training, save h_tape_all for backward
    auto h_tape_all = training ?
        torch::empty({seq_len + 1, batch_size, n_slots, dim}, options) :
        torch::empty({0}, options);

    // Workspace: tmp_Rh [B, D] + tmp_write_val [B, D]
    auto workspace = torch::empty({batch_size * dim * 2}, options);

    hasty::v0::elman_ladder::E26ParallelForward<__nv_bfloat16> forward_op(
        training, batch_size, n_slots, dim, handle, stream);

    forward_op.Run(
        seq_len,
        reinterpret_cast<const __nv_bfloat16*>(x_proj.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_h.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(b_h.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_write.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h_tape_init.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h_work_init.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(h_work_out.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(h_tape_final.data_ptr()),
        training ? reinterpret_cast<__nv_bfloat16*>(h_tape_all.data_ptr()) : nullptr,
        reinterpret_cast<__nv_bfloat16*>(read_attn.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(write_attn.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(workspace.data_ptr()));

    // Transpose outputs to [B, T, ...] format
    return {h_work_out.permute({1, 0, 2}).contiguous(),  // [B, T, D] h_work states (=output)
            h_tape_final,                                  // [B, N, D]
            h_tape_all,                                    // [T+1, B, N, D] or empty
            read_attn.permute({1, 0, 2}).contiguous(),   // [B, T, N]
            write_attn.permute({1, 0, 2}).contiguous()}; // [B, T, N]
}

std::vector<Tensor> e26_parallel_backward(
    Tensor x,              // [B, T, D]
    Tensor h_work_all,     // [B, T, D]
    Tensor h_work_init,    // [B, D]
    Tensor h_tape_all,     // [T+1, B, N, D]
    Tensor read_attn,      // [B, T, N]
    Tensor write_attn,     // [B, T, N]
    Tensor W_h,            // [D, D]
    Tensor W_x,            // [D, D]
    Tensor W_write,        // [D, D]
    Tensor d_h_work_out,   // [B, T, D]
    Tensor d_h_tape_final) { // [B, N, D]

    const auto batch_size = h_work_all.size(0);
    const auto seq_len = h_work_all.size(1);
    const auto dim = h_work_all.size(2);
    const auto n_slots = h_tape_all.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(h_work_all);
    CHECK_INPUT(h_work_init);
    CHECK_INPUT(h_tape_all);
    CHECK_INPUT(read_attn);
    CHECK_INPUT(write_attn);
    CHECK_INPUT(W_h);
    CHECK_INPUT(W_x);
    CHECK_INPUT(W_write);
    CHECK_INPUT(d_h_work_out);
    CHECK_INPUT(d_h_tape_final);

    const auto options = h_work_all.options();
    const auto float_options = options.dtype(torch::kFloat32);
    const at::cuda::CUDAGuard guard(options.device_index());
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Transpose inputs from [B, T, ...] to [T, B, ...] for kernel
    auto h_work_t = h_work_all.permute({1, 0, 2}).contiguous();  // [T, B, D]
    auto read_attn_t = read_attn.permute({1, 0, 2}).contiguous();  // [T, B, N]
    auto write_attn_t = write_attn.permute({1, 0, 2}).contiguous();  // [T, B, N]
    auto d_h_work_out_t = d_h_work_out.permute({1, 0, 2}).contiguous();  // [T, B, D]

    // Allocate gradient outputs
    auto dx_proj = torch::empty({seq_len, batch_size, dim}, options);
    auto d_pre_act_all = torch::empty({seq_len, batch_size, dim}, options);
    auto d_write_val_all = torch::empty({seq_len, batch_size, dim}, options);
    auto db_h = torch::zeros({dim}, float_options);
    auto d_h_tape = torch::empty({batch_size, n_slots, dim}, options);
    auto dW_h = torch::zeros({dim, dim}, float_options);
    auto dW_write = torch::zeros({dim, dim}, float_options);

    hasty::v0::elman_ladder::E26ParallelBackward<__nv_bfloat16> backward_op(
        batch_size, n_slots, dim, handle, stream);

    backward_op.Run(
        seq_len,
        reinterpret_cast<const __nv_bfloat16*>(h_work_t.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h_work_init.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h_tape_all.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(read_attn_t.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(write_attn_t.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_h.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_write.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(d_h_work_out_t.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(d_h_tape_final.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dx_proj.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(d_pre_act_all.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(d_write_val_all.data_ptr()),
        db_h.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(d_h_tape.data_ptr()),
        dW_h.data_ptr<float>(),
        dW_write.data_ptr<float>());

    // Compute dW_x = x^T @ d_pre_act (accumulated over all timesteps)
    auto x_t = x.permute({1, 0, 2}).contiguous();  // [T, B, D]
    auto dW_x = torch::zeros({dim, dim}, float_options);

    // Batch GEMM for dW_x
    const float alpha = 1.0f;
    const float beta_one = 1.0f;
    const __nv_bfloat16* d_pre_act_ptr = reinterpret_cast<const __nv_bfloat16*>(d_pre_act_all.data_ptr());
    const __nv_bfloat16* x_t_ptr = reinterpret_cast<const __nv_bfloat16*>(x_t.data_ptr());
    for (int t = 0; t < seq_len; ++t) {
        cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            dim, dim, batch_size,
            &alpha,
            d_pre_act_ptr + t * batch_size * dim, CUDA_R_16BF, dim,
            x_t_ptr + t * batch_size * dim, CUDA_R_16BF, dim,
            &beta_one,
            dW_x.data_ptr<float>(), CUDA_R_32F, dim,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
        );
    }

    // Compute dx = d_pre_act @ W_x
    auto dx_t = torch::empty({seq_len, batch_size, dim}, options);
    const float beta_zero = 0.0f;
    cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim, seq_len * batch_size, dim,
        &alpha,
        W_x.data_ptr(), CUDA_R_16BF, dim,
        d_pre_act_all.data_ptr(), CUDA_R_16BF, dim,
        &beta_zero,
        dx_t.data_ptr(), CUDA_R_16BF, dim,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
    );

    // Transpose dx back to [B, T, D]
    auto dx = dx_t.permute({1, 0, 2}).contiguous();

    // Convert weight gradients back to original dtype
    dW_h = dW_h.to(options.dtype());
    dW_x = dW_x.to(options.dtype());
    dW_write = dW_write.to(options.dtype());
    db_h = db_h.to(options.dtype());

    return {dx, dW_h, dW_x, db_h, dW_write};
}

// =============================================================================
// E28: E1 + Mamba2 Conv System
// =============================================================================

std::vector<Tensor> e28_conv_forward(
    bool training,
    Tensor x,              // [B, T, D] - input (after in_proj split)
    Tensor z,              // [B, T, D] - gate branch
    Tensor h_init,         // [B, D] - initial hidden state
    Tensor W_x,            // [D, D]
    Tensor W_h,            // [D, D]
    Tensor b,              // [D]
    Tensor conv_weight,    // [D, 1, K] or [D, K]
    Tensor conv_bias) {    // [D]

    const auto batch_size = x.size(0);
    const auto seq_len = x.size(1);
    const auto dim = x.size(2);
    const auto d_conv = conv_weight.size(-1);

    CHECK_INPUT(x);
    CHECK_INPUT(z);
    CHECK_INPUT(h_init);
    CHECK_INPUT(W_x);
    CHECK_INPUT(W_h);
    CHECK_INPUT(b);
    CHECK_INPUT(conv_weight);
    CHECK_INPUT(conv_bias);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Flatten conv_weight if needed: [D, 1, K] -> [D, K]
    auto conv_w = conv_weight.dim() == 3 ? conv_weight.squeeze(1) : conv_weight;
    conv_w = conv_w.contiguous();

    // Transpose inputs from [B, T, D] to [T, B, D] for kernel
    auto x_t = x.transpose(0, 1).contiguous();  // [T, B, D]
    auto z_t = z.transpose(0, 1).contiguous();  // [T, B, D]

    // Allocate outputs in [T, B, D] layout for kernel
    auto h_all_t = torch::empty({seq_len, batch_size, dim}, options);
    auto output_t = torch::empty({seq_len, batch_size, dim}, options);
    auto v_cache = training ?
        torch::empty({seq_len, batch_size, dim}, options) :
        torch::empty({0}, options);

    // Create kernel instance
    using namespace hasty::v0::elman_ladder;
    E28ConvForward<__nv_bfloat16> kernel(
        batch_size, seq_len, dim, d_conv, handle, stream);

    // Run forward
    kernel.Run(
        training,
        reinterpret_cast<const __nv_bfloat16*>(x_t.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(z_t.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h_init.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_x.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_h.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(b.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(conv_w.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(conv_bias.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(h_all_t.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output_t.data_ptr()),
        training ? reinterpret_cast<__nv_bfloat16*>(v_cache.data_ptr()) : nullptr
    );

    // Transpose outputs back to [B, T, D]
    auto h_all = h_all_t.transpose(0, 1).contiguous();
    auto output = output_t.transpose(0, 1).contiguous();

    return {h_all, output};
}

std::vector<Tensor> e28_conv_backward(
    Tensor x,              // [B, T, D]
    Tensor z,              // [B, T, D]
    Tensor h_init,         // [B, D]
    Tensor h_all,          // [B, T, D]
    Tensor W_x,            // [D, D]
    Tensor W_h,            // [D, D]
    Tensor conv_weight,    // [D, K]
    Tensor conv_bias,      // [D]
    Tensor d_output) {     // [B, T, D]

    const auto batch_size = x.size(0);
    const auto seq_len = x.size(1);
    const auto dim = x.size(2);
    const auto d_conv = conv_weight.size(-1);

    CHECK_INPUT(x);
    CHECK_INPUT(z);
    CHECK_INPUT(h_init);
    CHECK_INPUT(h_all);
    CHECK_INPUT(W_x);
    CHECK_INPUT(W_h);
    CHECK_INPUT(conv_weight);
    CHECK_INPUT(conv_bias);
    CHECK_INPUT(d_output);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Flatten conv_weight if needed
    auto conv_w = conv_weight.dim() == 3 ? conv_weight.squeeze(1) : conv_weight;
    conv_w = conv_w.contiguous();

    // Transpose inputs from [B, T, D] to [T, B, D] for kernel
    auto x_t = x.transpose(0, 1).contiguous();  // [T, B, D]
    auto z_t = z.transpose(0, 1).contiguous();  // [T, B, D]
    auto h_all_t = h_all.transpose(0, 1).contiguous();  // [T, B, D]
    auto d_output_t = d_output.transpose(0, 1).contiguous();  // [T, B, D]

    // Allocate gradients in [T, B, D] layout
    auto d_x_t = torch::zeros({seq_len, batch_size, dim}, x.options());
    auto d_z_t = torch::zeros({seq_len, batch_size, dim}, z.options());
    auto d_W_x = torch::zeros_like(W_x);
    auto d_W_h = torch::zeros_like(W_h);
    auto d_b = torch::zeros_like(conv_bias);
    auto d_conv_weight = torch::zeros_like(conv_w);
    auto d_conv_bias = torch::zeros_like(conv_bias);

    // Allocate workspace
    using namespace hasty::v0::elman_ladder;
    const int64_t workspace_size = E28ConvBackward<__nv_bfloat16>::WorkspaceSize(
        batch_size, seq_len, dim, d_conv);
    auto workspace = torch::empty({workspace_size}, options);

    // Create kernel instance
    E28ConvBackward<__nv_bfloat16> kernel(
        batch_size, seq_len, dim, d_conv, handle, stream);

    // Run backward
    kernel.Run(
        reinterpret_cast<const __nv_bfloat16*>(x_t.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(z_t.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h_init.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h_all_t.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_x.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_h.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(conv_w.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(conv_bias.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(d_output_t.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(d_x_t.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(d_z_t.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(d_W_x.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(d_W_h.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(d_b.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(d_conv_weight.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(d_conv_bias.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(workspace.data_ptr())
    );

    // Transpose gradients back to [B, T, D]
    auto d_x = d_x_t.transpose(0, 1).contiguous();
    auto d_z = d_z_t.transpose(0, 1).contiguous();

    // Reshape d_conv_weight back to [D, 1, K] if needed
    if (conv_weight.dim() == 3) {
        d_conv_weight = d_conv_weight.unsqueeze(1);
    }

    return {d_x, d_z, d_W_x, d_W_h, d_b, d_conv_weight, d_conv_bias};
}

// =============================================================================
// E29a: Selective Dual-Memory Elman (additive gate: silu(z + read + h_work))
// =============================================================================

std::vector<Tensor> e29a_selective_forward(
    bool training,
    Tensor x,              // [B, T, D] - raw input
    Tensor h_tape_init,    // [B, N, D] - initial tape state
    Tensor h_work_init,    // [B, D] - initial working memory
    Tensor W_h,            // [D, D]
    Tensor W_xz,           // [2*D, D] - projects to x_proj and z
    Tensor b_h,            // [D]
    Tensor W_write) {      // [D, D]

    const auto batch_size = x.size(0);
    const auto seq_len = x.size(1);
    const auto dim = x.size(2);
    const auto n_slots = h_tape_init.size(1);

    CHECK_INPUT(x);
    CHECK_INPUT(h_tape_init);
    CHECK_INPUT(h_work_init);
    CHECK_INPUT(W_h);
    CHECK_INPUT(W_xz);
    CHECK_INPUT(b_h);
    CHECK_INPUT(W_write);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Transpose x from [B, T, D] to [T, B, D] for efficient batch GEMM
    auto x_t = x.permute({1, 0, 2}).contiguous();  // [T, B, D]

    // PARALLEL PHASE: Pre-compute xz_proj = x @ W_xz^T for ALL timesteps
    // W_xz is [2*D, D] row-major = [D, 2*D] col-major, output is [T, B, 2*D]
    auto xz_proj = torch::empty({seq_len, batch_size, 2 * dim}, options);

    const float alpha = 1.0f;
    const float beta_zero = 0.0f;
    cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        2 * dim, seq_len * batch_size, dim,
        &alpha,
        W_xz.data_ptr(), CUDA_R_16BF, dim,  // lda = dim (row stride of row-major [2D, D])
        x_t.data_ptr(), CUDA_R_16BF, dim,
        &beta_zero,
        xz_proj.data_ptr(), CUDA_R_16BF, 2 * dim,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
    );

    // Split xz_proj into x_proj and z
    auto x_proj = xz_proj.narrow(2, 0, dim).contiguous();  // [T, B, D]
    auto z_all = xz_proj.narrow(2, dim, dim).contiguous(); // [T, B, D]

    // Allocate outputs
    auto output_all = torch::empty({seq_len, batch_size, dim}, options);
    auto h_work_all = torch::empty({seq_len, batch_size, dim}, options);
    auto h_tape_final = torch::empty({batch_size, n_slots, dim}, options);
    auto read_attn = torch::empty({seq_len, batch_size, n_slots}, options);
    auto write_attn = torch::empty({seq_len, batch_size, n_slots}, options);

    // For training, save h_tape_all for backward
    auto h_tape_all = training ?
        torch::empty({seq_len + 1, batch_size, n_slots, dim}, options) :
        torch::empty({0}, options);

    // Workspace: tmp_Rh [B, D] + tmp_write_val [B, D] + tmp_read_val [B, D]
    auto workspace = torch::empty({batch_size * dim * 3}, options);

    hasty::v0::elman_ladder::E29aSelectiveForward<__nv_bfloat16> forward_op(
        training, batch_size, n_slots, dim, handle, stream);

    forward_op.Run(
        seq_len,
        reinterpret_cast<const __nv_bfloat16*>(x_proj.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(z_all.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_h.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(b_h.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_write.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h_tape_init.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h_work_init.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output_all.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(h_work_all.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(h_tape_final.data_ptr()),
        training ? reinterpret_cast<__nv_bfloat16*>(h_tape_all.data_ptr()) : nullptr,
        reinterpret_cast<__nv_bfloat16*>(read_attn.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(write_attn.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(workspace.data_ptr()));

    // Transpose outputs to [B, T, ...] format
    // h_tape_all needs to be [B, T+1, N, D] for Python backward
    auto h_tape_all_transposed = training ?
        h_tape_all.permute({1, 0, 2, 3}).contiguous() :  // [T+1, B, N, D] -> [B, T+1, N, D]
        h_tape_all;
    return {output_all.permute({1, 0, 2}).contiguous(),   // [B, T, D] selective gated output
            h_work_all.permute({1, 0, 2}).contiguous(),   // [B, T, D] h_work states
            h_tape_final,                                  // [B, N, D]
            h_tape_all_transposed,                         // [B, T+1, N, D] or empty
            read_attn.permute({1, 0, 2}).contiguous(),    // [B, T, N]
            write_attn.permute({1, 0, 2}).contiguous()};  // [B, T, N]
}

// =============================================================================
// E29b: Selective Dual-Memory Elman (learned gate: W_gate @ [z; read; h_work])
// =============================================================================

std::vector<Tensor> e29b_selective_forward(
    bool training,
    Tensor x,              // [B, T, D] - raw input
    Tensor h_tape_init,    // [B, N, D] - initial tape state
    Tensor h_work_init,    // [B, D] - initial working memory
    Tensor W_h,            // [D, D]
    Tensor W_xz,           // [2*D, D] - projects to x_proj and z
    Tensor b_h,            // [D]
    Tensor W_write,        // [D, D]
    Tensor W_gate) {       // [D, 3*D] - learned gate projection

    const auto batch_size = x.size(0);
    const auto seq_len = x.size(1);
    const auto dim = x.size(2);
    const auto n_slots = h_tape_init.size(1);

    CHECK_INPUT(x);
    CHECK_INPUT(h_tape_init);
    CHECK_INPUT(h_work_init);
    CHECK_INPUT(W_h);
    CHECK_INPUT(W_xz);
    CHECK_INPUT(b_h);
    CHECK_INPUT(W_write);
    CHECK_INPUT(W_gate);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Transpose x from [B, T, D] to [T, B, D]
    auto x_t = x.permute({1, 0, 2}).contiguous();

    // PARALLEL PHASE: Pre-compute xz_proj = x @ W_xz^T
    // W_xz is [2*D, D] row-major = [D, 2*D] col-major
    auto xz_proj = torch::empty({seq_len, batch_size, 2 * dim}, options);

    const float alpha = 1.0f;
    const float beta_zero = 0.0f;
    cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        2 * dim, seq_len * batch_size, dim,
        &alpha,
        W_xz.data_ptr(), CUDA_R_16BF, dim,  // lda = dim (row stride of row-major [2D, D])
        x_t.data_ptr(), CUDA_R_16BF, dim,
        &beta_zero,
        xz_proj.data_ptr(), CUDA_R_16BF, 2 * dim,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
    );

    auto x_proj = xz_proj.narrow(2, 0, dim).contiguous();
    auto z_all = xz_proj.narrow(2, dim, dim).contiguous();

    // Allocate outputs
    auto output_all = torch::empty({seq_len, batch_size, dim}, options);
    auto h_work_all = torch::empty({seq_len, batch_size, dim}, options);
    auto h_tape_final = torch::empty({batch_size, n_slots, dim}, options);
    auto read_attn = torch::empty({seq_len, batch_size, n_slots}, options);
    auto write_attn = torch::empty({seq_len, batch_size, n_slots}, options);

    auto h_tape_all = training ?
        torch::empty({seq_len + 1, batch_size, n_slots, dim}, options) :
        torch::empty({0}, options);

    // Workspace: tmp_Rh [B, D] + tmp_write_val [B, D] + tmp_read_val [B, D]
    auto workspace = torch::empty({batch_size * dim * 3}, options);

    hasty::v0::elman_ladder::E29bSelectiveForward<__nv_bfloat16> forward_op(
        training, batch_size, n_slots, dim, handle, stream);

    forward_op.Run(
        seq_len,
        reinterpret_cast<const __nv_bfloat16*>(x_proj.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(z_all.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_h.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(b_h.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_write.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_gate.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h_tape_init.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h_work_init.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output_all.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(h_work_all.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(h_tape_final.data_ptr()),
        training ? reinterpret_cast<__nv_bfloat16*>(h_tape_all.data_ptr()) : nullptr,
        reinterpret_cast<__nv_bfloat16*>(read_attn.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(write_attn.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(workspace.data_ptr()));

    // Transpose outputs to [B, T, ...] format
    // h_tape_all needs to be [B, T+1, N, D] for Python backward
    auto h_tape_all_transposed = training ?
        h_tape_all.permute({1, 0, 2, 3}).contiguous() :  // [T+1, B, N, D] -> [B, T+1, N, D]
        h_tape_all;
    return {output_all.permute({1, 0, 2}).contiguous(),
            h_work_all.permute({1, 0, 2}).contiguous(),
            h_tape_final,
            h_tape_all_transposed,
            read_attn.permute({1, 0, 2}).contiguous(),
            write_attn.permute({1, 0, 2}).contiguous()};
}

// =============================================================================
// E29a Backward
// =============================================================================

std::vector<Tensor> e29a_selective_backward(
    Tensor x,                // [B, T, D]
    Tensor h_work_all,       // [B, T, D]
    Tensor h_work_init,      // [B, D]
    Tensor h_tape_all,       // [B, T+1, N, D]
    Tensor read_attn,        // [B, T, N]
    Tensor write_attn,       // [B, T, N]
    Tensor W_h,              // [D, D]
    Tensor W_xz,             // [2*D, D]
    Tensor W_write,          // [D, D]
    Tensor d_output_all,     // [B, T, D]
    Tensor d_h_tape_final) { // [B, N, D]

    const auto batch_size = h_work_all.size(0);
    const auto seq_len = h_work_all.size(1);
    const auto dim = h_work_all.size(2);
    const auto n_slots = h_tape_all.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(h_work_all);
    CHECK_INPUT(h_work_init);
    CHECK_INPUT(h_tape_all);
    CHECK_INPUT(read_attn);
    CHECK_INPUT(write_attn);
    CHECK_INPUT(W_h);
    CHECK_INPUT(W_xz);
    CHECK_INPUT(W_write);
    CHECK_INPUT(d_output_all);
    CHECK_INPUT(d_h_tape_final);

    const auto options = h_work_all.options();
    const auto float_options = options.dtype(torch::kFloat32);
    const at::cuda::CUDAGuard guard(options.device_index());
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Transpose inputs to [T, B, ...] layout
    auto h_work_t = h_work_all.permute({1, 0, 2}).contiguous();  // [T, B, D]
    auto h_tape_t = h_tape_all.permute({1, 0, 2, 3}).contiguous();  // [T+1, B, N, D]
    auto read_attn_t = read_attn.permute({1, 0, 2}).contiguous();   // [T, B, N]
    auto write_attn_t = write_attn.permute({1, 0, 2}).contiguous(); // [T, B, N]
    auto d_output_t = d_output_all.permute({1, 0, 2}).contiguous(); // [T, B, D]

    // Recompute z from x @ W_xz
    auto x_t = x.permute({1, 0, 2}).contiguous();  // [T, B, D]
    auto xz_proj = torch::empty({seq_len, batch_size, 2 * dim}, options);
    const float alpha = 1.0f;
    const float beta_zero = 0.0f;
    cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        2 * dim, seq_len * batch_size, dim,
        &alpha,
        W_xz.data_ptr(), CUDA_R_16BF, dim,
        x_t.data_ptr(), CUDA_R_16BF, dim,
        &beta_zero,
        xz_proj.data_ptr(), CUDA_R_16BF, 2 * dim,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
    );
    auto z_all = xz_proj.narrow(2, dim, dim).contiguous();  // [T, B, D]

    // Allocate gradient outputs
    auto dx_proj = torch::empty({seq_len, batch_size, dim}, options);
    auto dz = torch::empty({seq_len, batch_size, dim}, options);
    auto d_pre_act_all = torch::empty({seq_len, batch_size, dim}, options);
    auto d_write_val_all = torch::empty({seq_len, batch_size, dim}, options);
    auto db_h = torch::zeros({dim}, float_options);
    auto d_h_tape = torch::empty({batch_size, n_slots, dim}, options);
    auto dW_h = torch::zeros({dim, dim}, float_options);
    auto dW_write = torch::zeros({dim, dim}, float_options);

    hasty::v0::elman_ladder::E29aSelectiveBackward<__nv_bfloat16> backward_op(
        batch_size, n_slots, dim, handle, stream);

    backward_op.Run(
        seq_len,
        reinterpret_cast<const __nv_bfloat16*>(h_work_t.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h_work_init.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h_tape_t.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(read_attn_t.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(write_attn_t.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(z_all.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_h.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_write.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(d_output_t.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(d_h_tape_final.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dx_proj.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dz.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(d_pre_act_all.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(d_write_val_all.data_ptr()),
        db_h.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(d_h_tape.data_ptr()),
        dW_h.data_ptr<float>(),
        dW_write.data_ptr<float>());

    // Compute dx from dx_proj and dz: dxz = [dx_proj; dz], dx = dxz @ W_xz
    auto dxz = torch::cat({dx_proj, dz}, 2);  // [T, B, 2*D]
    auto dx_t = torch::empty({seq_len, batch_size, dim}, options);
    cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim, seq_len * batch_size, 2 * dim,
        &alpha,
        W_xz.data_ptr(), CUDA_R_16BF, dim,
        dxz.data_ptr(), CUDA_R_16BF, 2 * dim,
        &beta_zero,
        dx_t.data_ptr(), CUDA_R_16BF, dim,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
    );

    // Compute dW_xz = dxz^T @ x
    auto dW_xz = torch::zeros({2 * dim, dim}, float_options);
    const float beta_one = 1.0f;
    for (int t = 0; t < seq_len; ++t) {
        cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            2 * dim, dim, batch_size,
            &alpha,
            reinterpret_cast<const __nv_bfloat16*>(dxz.data_ptr()) + t * batch_size * 2 * dim, CUDA_R_16BF, 2 * dim,
            reinterpret_cast<const __nv_bfloat16*>(x_t.data_ptr()) + t * batch_size * dim, CUDA_R_16BF, dim,
            &beta_one,
            dW_xz.data_ptr<float>(), CUDA_R_32F, 2 * dim,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
        );
    }

    // Transpose dx back
    auto dx = dx_t.permute({1, 0, 2}).contiguous();

    // Convert weight gradients to original dtype
    dW_h = dW_h.to(options.dtype());
    dW_xz = dW_xz.to(options.dtype());
    db_h = db_h.to(options.dtype());
    dW_write = dW_write.to(options.dtype());

    return {dx, dW_h, dW_xz, db_h, dW_write};
}

// =============================================================================
// E29b Backward
// =============================================================================

std::vector<Tensor> e29b_selective_backward(
    Tensor x,                // [B, T, D]
    Tensor h_work_all,       // [B, T, D]
    Tensor h_work_init,      // [B, D]
    Tensor h_tape_all,       // [B, T+1, N, D]
    Tensor read_attn,        // [B, T, N]
    Tensor write_attn,       // [B, T, N]
    Tensor W_h,              // [D, D]
    Tensor W_xz,             // [2*D, D]
    Tensor W_write,          // [D, D]
    Tensor W_gate,           // [D, 3*D]
    Tensor d_output_all,     // [B, T, D]
    Tensor d_h_tape_final) { // [B, N, D]

    const auto batch_size = h_work_all.size(0);
    const auto seq_len = h_work_all.size(1);
    const auto dim = h_work_all.size(2);
    const auto n_slots = h_tape_all.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(h_work_all);
    CHECK_INPUT(h_work_init);
    CHECK_INPUT(h_tape_all);
    CHECK_INPUT(read_attn);
    CHECK_INPUT(write_attn);
    CHECK_INPUT(W_h);
    CHECK_INPUT(W_xz);
    CHECK_INPUT(W_write);
    CHECK_INPUT(W_gate);
    CHECK_INPUT(d_output_all);
    CHECK_INPUT(d_h_tape_final);

    const auto options = h_work_all.options();
    const auto float_options = options.dtype(torch::kFloat32);
    const at::cuda::CUDAGuard guard(options.device_index());
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Transpose inputs
    auto h_work_t = h_work_all.permute({1, 0, 2}).contiguous();
    auto h_tape_t = h_tape_all.permute({1, 0, 2, 3}).contiguous();
    auto read_attn_t = read_attn.permute({1, 0, 2}).contiguous();
    auto write_attn_t = write_attn.permute({1, 0, 2}).contiguous();
    auto d_output_t = d_output_all.permute({1, 0, 2}).contiguous();

    // Recompute z
    auto x_t = x.permute({1, 0, 2}).contiguous();
    auto xz_proj = torch::empty({seq_len, batch_size, 2 * dim}, options);
    const float alpha = 1.0f;
    const float beta_zero = 0.0f;
    cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        2 * dim, seq_len * batch_size, dim,
        &alpha,
        W_xz.data_ptr(), CUDA_R_16BF, dim,
        x_t.data_ptr(), CUDA_R_16BF, dim,
        &beta_zero,
        xz_proj.data_ptr(), CUDA_R_16BF, 2 * dim,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
    );
    auto z_all = xz_proj.narrow(2, dim, dim).contiguous();

    // Allocate outputs
    auto dx_proj = torch::empty({seq_len, batch_size, dim}, options);
    auto dz = torch::empty({seq_len, batch_size, dim}, options);
    auto d_pre_act_all = torch::empty({seq_len, batch_size, dim}, options);
    auto d_write_val_all = torch::empty({seq_len, batch_size, dim}, options);
    auto db_h = torch::zeros({dim}, float_options);
    auto d_h_tape = torch::empty({batch_size, n_slots, dim}, options);
    auto dW_h = torch::zeros({dim, dim}, float_options);
    auto dW_write = torch::zeros({dim, dim}, float_options);
    auto dW_gate = torch::zeros({dim, 3 * dim}, float_options);

    hasty::v0::elman_ladder::E29bSelectiveBackward<__nv_bfloat16> backward_op(
        batch_size, n_slots, dim, handle, stream);

    backward_op.Run(
        seq_len,
        reinterpret_cast<const __nv_bfloat16*>(h_work_t.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h_work_init.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h_tape_t.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(read_attn_t.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(write_attn_t.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(z_all.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_h.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_write.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_gate.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(d_output_t.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(d_h_tape_final.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dx_proj.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dz.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(d_pre_act_all.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(d_write_val_all.data_ptr()),
        db_h.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(d_h_tape.data_ptr()),
        dW_h.data_ptr<float>(),
        dW_write.data_ptr<float>(),
        dW_gate.data_ptr<float>());

    // Compute dx
    auto dxz = torch::cat({dx_proj, dz}, 2);
    auto dx_t = torch::empty({seq_len, batch_size, dim}, options);
    cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim, seq_len * batch_size, 2 * dim,
        &alpha,
        W_xz.data_ptr(), CUDA_R_16BF, dim,
        dxz.data_ptr(), CUDA_R_16BF, 2 * dim,
        &beta_zero,
        dx_t.data_ptr(), CUDA_R_16BF, dim,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
    );

    // Compute dW_xz
    auto dW_xz = torch::zeros({2 * dim, dim}, float_options);
    const float beta_one = 1.0f;
    for (int t = 0; t < seq_len; ++t) {
        cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            2 * dim, dim, batch_size,
            &alpha,
            reinterpret_cast<const __nv_bfloat16*>(dxz.data_ptr()) + t * batch_size * 2 * dim, CUDA_R_16BF, 2 * dim,
            reinterpret_cast<const __nv_bfloat16*>(x_t.data_ptr()) + t * batch_size * dim, CUDA_R_16BF, dim,
            &beta_one,
            dW_xz.data_ptr<float>(), CUDA_R_32F, 2 * dim,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
        );
    }

    auto dx = dx_t.permute({1, 0, 2}).contiguous();

    dW_h = dW_h.to(options.dtype());
    dW_xz = dW_xz.to(options.dtype());
    db_h = db_h.to(options.dtype());
    dW_write = dW_write.to(options.dtype());
    dW_gate = dW_gate.to(options.dtype());

    return {dx, dW_h, dW_xz, db_h, dW_write, dW_gate};
}

// =============================================================================
// E29c: SSM-Style Diagonal Gating Dual-Memory Elman
// gate = silu(z * g_z + read * g_r + h_work * g_h + b_gate)
// =============================================================================

std::vector<Tensor> e29c_diagonal_forward(
    bool training,
    Tensor x,              // [B, T, D] - raw input
    Tensor h_tape_init,    // [B, N, D] - initial tape state
    Tensor h_work_init,    // [B, D] - initial working memory
    Tensor W_h,            // [D, D]
    Tensor W_xz,           // [2*D, D] - projects to x_proj and z
    Tensor b_h,            // [D]
    Tensor W_write,        // [D, D]
    Tensor g_z,            // [D] - z gate scale
    Tensor g_r,            // [D] - read gate scale
    Tensor g_h,            // [D] - h_work gate scale
    Tensor b_gate) {       // [D] - gate bias

    const auto batch_size = x.size(0);
    const auto seq_len = x.size(1);
    const auto dim = x.size(2);
    const auto n_slots = h_tape_init.size(1);

    CHECK_INPUT(x);
    CHECK_INPUT(h_tape_init);
    CHECK_INPUT(h_work_init);
    CHECK_INPUT(W_h);
    CHECK_INPUT(W_xz);
    CHECK_INPUT(b_h);
    CHECK_INPUT(W_write);
    CHECK_INPUT(g_z);
    CHECK_INPUT(g_r);
    CHECK_INPUT(g_h);
    CHECK_INPUT(b_gate);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Transpose x from [B, T, D] to [T, B, D]
    auto x_t = x.permute({1, 0, 2}).contiguous();

    // PARALLEL PHASE: Pre-compute xz_proj = x @ W_xz^T for ALL timesteps
    auto xz_proj = torch::empty({seq_len, batch_size, 2 * dim}, options);

    const float alpha = 1.0f;
    const float beta_zero = 0.0f;
    cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        2 * dim, seq_len * batch_size, dim,
        &alpha,
        W_xz.data_ptr(), CUDA_R_16BF, dim,
        x_t.data_ptr(), CUDA_R_16BF, dim,
        &beta_zero,
        xz_proj.data_ptr(), CUDA_R_16BF, 2 * dim,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
    );

    // Split xz_proj into x_proj and z
    auto x_proj = xz_proj.narrow(2, 0, dim).contiguous();
    auto z_all = xz_proj.narrow(2, dim, dim).contiguous();

    // Allocate outputs
    auto output_all = torch::empty({seq_len, batch_size, dim}, options);
    auto h_work_all = torch::empty({seq_len, batch_size, dim}, options);
    auto h_tape_final = torch::empty({batch_size, n_slots, dim}, options);
    auto read_attn = torch::empty({seq_len, batch_size, n_slots}, options);
    auto write_attn = torch::empty({seq_len, batch_size, n_slots}, options);

    // For training, save h_tape_all and read_val_all for backward
    auto h_tape_all = training ?
        torch::empty({seq_len + 1, batch_size, n_slots, dim}, options) :
        torch::empty({0}, options);
    auto read_val_all = training ?
        torch::empty({seq_len, batch_size, dim}, options) :
        torch::empty({0}, options);

    hasty::v0::elman_ladder::E29cDiagonalForward<__nv_bfloat16> forward_op(
        batch_size, n_slots, dim, handle, stream);

    forward_op.Run(
        seq_len,
        reinterpret_cast<const __nv_bfloat16*>(x_proj.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(z_all.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h_tape_init.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h_work_init.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_h.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(b_h.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_write.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(g_z.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(g_r.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(g_h.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(b_gate.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output_all.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(h_work_all.data_ptr()),
        training ? reinterpret_cast<__nv_bfloat16*>(h_tape_all.data_ptr()) : nullptr,
        reinterpret_cast<__nv_bfloat16*>(read_attn.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(write_attn.data_ptr()),
        training ? reinterpret_cast<__nv_bfloat16*>(read_val_all.data_ptr()) : nullptr,
        handle);

    // Copy final tape state
    if (training) {
        cudaMemcpyAsync(
            h_tape_final.data_ptr(),
            reinterpret_cast<__nv_bfloat16*>(h_tape_all.data_ptr()) + seq_len * batch_size * n_slots * dim,
            batch_size * n_slots * dim * sizeof(__nv_bfloat16),
            cudaMemcpyDeviceToDevice, stream);
    }

    // Transpose outputs to [B, T, ...] format
    auto h_tape_all_transposed = training ?
        h_tape_all.permute({1, 0, 2, 3}).contiguous() :
        h_tape_all;
    auto read_val_all_transposed = training ?
        read_val_all.permute({1, 0, 2}).contiguous() :
        read_val_all;
    return {output_all.permute({1, 0, 2}).contiguous(),
            h_work_all.permute({1, 0, 2}).contiguous(),
            h_tape_final,
            h_tape_all_transposed,
            read_attn.permute({1, 0, 2}).contiguous(),
            write_attn.permute({1, 0, 2}).contiguous(),
            read_val_all_transposed};
}

std::vector<Tensor> e29c_diagonal_backward(
    Tensor x,              // [B, T, D]
    Tensor h_work_all,     // [B, T, D]
    Tensor h_work_init,    // [B, D]
    Tensor h_tape_all,     // [B, T+1, N, D]
    Tensor read_attn_all,  // [B, T, N]
    Tensor write_attn_all, // [B, T, N]
    Tensor read_val_all,   // [B, T, D] - from forward (avoids recompute bug)
    Tensor W_h,            // [D, D]
    Tensor W_xz,           // [2*D, D]
    Tensor W_write,        // [D, D]
    Tensor g_z,            // [D]
    Tensor g_r,            // [D]
    Tensor g_h,            // [D]
    Tensor b_gate,         // [D]
    Tensor d_output_all,   // [B, T, D]
    Tensor d_h_tape_final) { // [B, N, D]

    const auto batch_size = x.size(0);
    const auto seq_len = x.size(1);
    const auto dim = x.size(2);
    const auto n_slots = h_tape_all.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(h_work_all);
    CHECK_INPUT(h_work_init);
    CHECK_INPUT(h_tape_all);
    CHECK_INPUT(read_attn_all);
    CHECK_INPUT(write_attn_all);
    CHECK_INPUT(read_val_all);
    CHECK_INPUT(W_h);
    CHECK_INPUT(W_xz);
    CHECK_INPUT(W_write);
    CHECK_INPUT(g_z);
    CHECK_INPUT(g_r);
    CHECK_INPUT(g_h);
    CHECK_INPUT(b_gate);
    CHECK_INPUT(d_output_all);
    CHECK_INPUT(d_h_tape_final);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Transpose inputs to [T, B, ...] format
    auto x_t = x.permute({1, 0, 2}).contiguous();
    auto h_work_all_t = h_work_all.permute({1, 0, 2}).contiguous();
    auto h_tape_all_t = h_tape_all.permute({1, 0, 2, 3}).contiguous();
    auto read_attn_t = read_attn_all.permute({1, 0, 2}).contiguous();
    auto write_attn_t = write_attn_all.permute({1, 0, 2}).contiguous();
    auto read_val_t = read_val_all.permute({1, 0, 2}).contiguous();
    auto d_output_t = d_output_all.permute({1, 0, 2}).contiguous();

    // Recompute z_all
    auto xz_proj = torch::empty({seq_len, batch_size, 2 * dim}, options);
    const float alpha = 1.0f;
    const float beta_zero = 0.0f;
    cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        2 * dim, seq_len * batch_size, dim,
        &alpha,
        W_xz.data_ptr(), CUDA_R_16BF, dim,
        x_t.data_ptr(), CUDA_R_16BF, dim,
        &beta_zero,
        xz_proj.data_ptr(), CUDA_R_16BF, 2 * dim,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
    );
    auto z_all = xz_proj.narrow(2, dim, dim).contiguous();

    // Allocate outputs
    auto dx_proj_t = torch::empty({seq_len, batch_size, dim}, options);
    auto dz_t = torch::empty({seq_len, batch_size, dim}, options);
    auto d_pre_act_all = torch::empty({seq_len, batch_size, dim}, options);
    auto d_write_val_all = torch::empty({seq_len, batch_size, dim}, options);
    auto d_h_tape = torch::empty({batch_size, n_slots, dim}, options);

    // Float accumulators for weight gradients
    auto dW_h = torch::zeros({dim, dim}, options.dtype(torch::kFloat32));
    auto dW_write = torch::zeros({dim, dim}, options.dtype(torch::kFloat32));
    auto db_h = torch::zeros({dim}, options.dtype(torch::kFloat32));
    auto dg_z = torch::zeros({dim}, options.dtype(torch::kFloat32));
    auto dg_r = torch::zeros({dim}, options.dtype(torch::kFloat32));
    auto dg_h_out = torch::zeros({dim}, options.dtype(torch::kFloat32));
    auto db_gate_out = torch::zeros({dim}, options.dtype(torch::kFloat32));

    hasty::v0::elman_ladder::E29cDiagonalBackward<__nv_bfloat16> backward_op(
        batch_size, n_slots, dim, handle, stream);

    backward_op.Run(
        seq_len,
        reinterpret_cast<const __nv_bfloat16*>(h_work_all_t.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h_work_init.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h_tape_all_t.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(read_attn_t.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(write_attn_t.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(read_val_t.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(z_all.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_h.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_write.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(g_z.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(g_r.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(g_h.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(b_gate.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(d_output_t.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(d_h_tape_final.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dx_proj_t.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dz_t.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(d_pre_act_all.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(d_write_val_all.data_ptr()),
        db_h.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(d_h_tape.data_ptr()),
        dW_h.data_ptr<float>(),
        dW_write.data_ptr<float>(),
        dg_z.data_ptr<float>(),
        dg_r.data_ptr<float>(),
        dg_h_out.data_ptr<float>(),
        db_gate_out.data_ptr<float>());

    // Compute dW_xz from dx_proj and dz
    auto dxz = torch::cat({dx_proj_t, dz_t}, 2);
    auto dW_xz = torch::zeros({2 * dim, dim}, options.dtype(torch::kFloat32));
    const float beta_one = 1.0f;

    for (int t = 0; t < seq_len; ++t) {
        cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            2 * dim, dim, batch_size,
            &alpha,
            reinterpret_cast<const __nv_bfloat16*>(dxz.data_ptr()) + t * batch_size * 2 * dim, CUDA_R_16BF, 2 * dim,
            reinterpret_cast<const __nv_bfloat16*>(x_t.data_ptr()) + t * batch_size * dim, CUDA_R_16BF, dim,
            &beta_one,
            dW_xz.data_ptr<float>(), CUDA_R_32F, 2 * dim,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
        );
    }

    // Compute dx from dxz @ W_xz
    auto dx_t = torch::empty({seq_len, batch_size, dim}, options);
    cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim, seq_len * batch_size, 2 * dim,
        &alpha,
        W_xz.data_ptr(), CUDA_R_16BF, dim,
        dxz.data_ptr(), CUDA_R_16BF, 2 * dim,
        &beta_zero,
        dx_t.data_ptr(), CUDA_R_16BF, dim,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
    );

    auto dx = dx_t.permute({1, 0, 2}).contiguous();

    // CUBLAS writes column-major [2*D, D]. When PyTorch reads it as row-major,
    // the bytes correspond to row-major [D, 2*D]. Reshape and transpose to get [2*D, D].
    dW_xz = dW_xz.view({dim, 2 * dim}).t().contiguous();

    // Same column-major issue for dW_h and dW_write [D, D]
    dW_h = dW_h.t().contiguous();
    dW_write = dW_write.t().contiguous();

    // Convert to output dtype
    dW_h = dW_h.to(options.dtype());
    dW_xz = dW_xz.to(options.dtype());
    db_h = db_h.to(options.dtype());
    dW_write = dW_write.to(options.dtype());
    dg_z = dg_z.to(options.dtype());
    dg_r = dg_r.to(options.dtype());
    dg_h_out = dg_h_out.to(options.dtype());
    db_gate_out = db_gate_out.to(options.dtype());

    return {dx, dW_h, dW_xz, db_h, dW_write, dg_z, dg_r, dg_h_out, db_gate_out};
}

// =============================================================================
// E58: Per-Dimension Learned Radii Elman
// =============================================================================

std::vector<Tensor> e58_learned_radii_forward(
    bool training,
    Tensor x,           // [T, B, dim] pre-activated input
    Tensor z,           // [T, B, dim] gate input (pre silu)
    Tensor h0,          // [B, dim]
    Tensor W_x,         // [dim, dim]
    Tensor W_h,         // [dim, dim] (unscaled)
    Tensor radii,       // [dim] per-dimension scaling
    Tensor b) {         // [dim]

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(z);
    CHECK_INPUT(h0);
    CHECK_INPUT(W_x);
    CHECK_INPUT(W_h);
    CHECK_INPUT(radii);
    CHECK_INPUT(b);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, dim}, options)
                        : torch::empty({0}, options);
    Tensor Rh_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                               : torch::empty({0}, options);

    // Forward workspace: [tmp_Wx: T*BD] [tmp_Rh: BD]
    const int64_t BD = batch_size * dim;
    Tensor workspace = torch::empty({time_steps * BD + BD}, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e58_learned_radii_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E58LearnedRadiiForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(W_h),
            ptr<scalar_t>(radii),
            ptr<scalar_t>(b),
            ptr<scalar_t>(x),
            ptr<scalar_t>(z),
            ptr<scalar_t>(h),
            ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v) : nullptr,
            training ? ptr<scalar_t>(Rh_cache) : nullptr,
            ptr<scalar_t>(workspace));
    }));

    return {h, output, v, Rh_cache};
}

std::vector<Tensor> e58_learned_radii_backward(
    Tensor W_x,
    Tensor W_h,
    Tensor radii,
    Tensor x,
    Tensor z,
    Tensor h,
    Tensor v,
    Tensor Rh_cache,
    Tensor d_output) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(W_x);
    CHECK_INPUT(W_h);
    CHECK_INPUT(radii);
    CHECK_INPUT(x);
    CHECK_INPUT(z);
    CHECK_INPUT(h);
    CHECK_INPUT(v);
    CHECK_INPUT(Rh_cache);
    CHECK_INPUT(d_output);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::empty_like(x);
    Tensor dz = torch::empty_like(z);
    Tensor dW_x = torch::zeros({dim, dim}, options);
    Tensor dW_h = torch::zeros({dim, dim}, options);
    Tensor d_radii = torch::zeros({dim}, options);
    Tensor db = torch::zeros({dim}, options);

    // Workspace layout: [dv_all: T*BD] [dRh_all: T*BD] [dh: BD] [dh_recurrent: BD]
    //                   [db_float: dim] [d_radii_float: dim]
    const int64_t BD = batch_size * dim;
    const int64_t float_in_T = (2 * dim * sizeof(float) + sizeof(float) - 1) / sizeof(float);
    const int64_t workspace_size = (2 * time_steps + 2) * BD + float_in_T * 2;
    Tensor workspace = torch::empty({workspace_size}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e58_learned_radii_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E58LearnedRadiiBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(W_h),
            ptr<scalar_t>(radii),
            ptr<scalar_t>(x),
            ptr<scalar_t>(z),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(Rh_cache),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dz),
            ptr<scalar_t>(dW_x),
            ptr<scalar_t>(dW_h),
            ptr<scalar_t>(d_radii),
            ptr<scalar_t>(db),
            ptr<scalar_t>(workspace));
    }));

    return {dx, dz, dW_x, dW_h, d_radii, db};
}

// =============================================================================
// E59: Highway Elman - Residual Recurrence with Perfect Gradient Flow
// h_t = h_{t-1} + alpha * (W @ x_t + b)   # Residual accumulation (gradient = I)
// output_t = h_t * silu(h_t)              # Nonlinearity at output only
// =============================================================================

std::vector<Tensor> e59_highway_forward(
    bool training,
    Tensor x,           // [T, B, dim]
    Tensor h0,          // [B, dim]
    Tensor W,           // [dim, dim]
    Tensor b,           // [dim]
    float alpha) {      // exp(log_alpha)

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(W);
    CHECK_INPUT(b);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor Wx_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                               : torch::empty({0}, options);

    // Workspace: [T*BD] for Wx_all
    const int64_t BD = batch_size * dim;
    Tensor workspace = torch::empty({time_steps * BD}, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e59_highway_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E59HighwayForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            alpha,
            ptr<scalar_t>(W),
            ptr<scalar_t>(b),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(output),
            training ? ptr<scalar_t>(Wx_cache) : nullptr,
            ptr<scalar_t>(workspace));
    }));

    return {h, output, Wx_cache};
}

std::vector<Tensor> e59_highway_backward(
    float alpha,
    Tensor W,
    Tensor x,
    Tensor h,
    Tensor Wx_cache,
    Tensor d_output) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(W);
    CHECK_INPUT(x);
    CHECK_INPUT(h);
    CHECK_INPUT(Wx_cache);
    CHECK_INPUT(d_output);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::empty_like(x);
    Tensor dW = torch::zeros({dim, dim}, options);
    Tensor db = torch::zeros({dim}, options);
    Tensor d_log_alpha = torch::zeros({1}, options.dtype(torch::kFloat32));

    // Workspace: [dWx_all: T*BD] [dh: BD] [dh_raw: BD] [dh_recurrent: BD] [h_raw: BD] [db_float: dim]
    const int64_t BD = batch_size * dim;
    const int64_t float_in_T = (dim * sizeof(float) + sizeof(float) - 1) / sizeof(float);
    Tensor workspace = torch::empty({(time_steps + 4) * BD + float_in_T}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e59_highway_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E59HighwayBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            alpha,
            ptr<scalar_t>(W),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(Wx_cache),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dW),
            ptr<scalar_t>(db),
            d_log_alpha.data_ptr<float>(),
            ptr<scalar_t>(workspace));
    }));

    return {dx, dW, db, d_log_alpha};
}

// =============================================================================
// E60: Residual Nonlinear Elman
// h_t = h_{t-1} + alpha * tanh(W_h @ h_{t-1} + W_x @ x_t + b)
// output = h_t * silu(h_t)
// alpha = exp(log_alpha) is a learned positive scalar
// =============================================================================

std::vector<Tensor> e60_residual_nonlinear_forward(
    bool training,
    Tensor x,           // [T, B, dim] pre-activated input
    Tensor h0,          // [B, dim]
    Tensor W_x,         // [dim, dim]
    Tensor W_h,         // [dim, dim]
    Tensor b,           // [dim]
    Tensor log_alpha) { // [1] scalar (exp(log_alpha) gives alpha)

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(W_x);
    CHECK_INPUT(W_h);
    CHECK_INPUT(b);
    CHECK_INPUT(log_alpha);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor tanh_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                 : torch::empty({0}, options);

    // Forward workspace: [tmp_Wx: T*BD] [tmp_Rh: BD]
    const int64_t BD = batch_size * dim;
    Tensor workspace = torch::empty({time_steps * BD + BD}, options);

    h[0] = h0;

    // log_alpha needs to be float for the kernel
    Tensor log_alpha_float = log_alpha.to(torch::kFloat32).contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e60_residual_nonlinear_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E60ResidualNonlinearForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(W_h),
            ptr<scalar_t>(b),
            log_alpha_float.data_ptr<float>(),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(output),
            training ? ptr<scalar_t>(tanh_cache) : nullptr,
            ptr<scalar_t>(workspace));
    }));

    return {h, output, tanh_cache};
}

std::vector<Tensor> e60_residual_nonlinear_backward(
    Tensor W_x,
    Tensor W_h,
    Tensor log_alpha,
    Tensor x,
    Tensor h,
    Tensor tanh_cache,
    Tensor d_output) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(W_x);
    CHECK_INPUT(W_h);
    CHECK_INPUT(log_alpha);
    CHECK_INPUT(x);
    CHECK_INPUT(h);
    CHECK_INPUT(tanh_cache);
    CHECK_INPUT(d_output);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::empty_like(x);
    Tensor dW_x = torch::zeros({dim, dim}, options);
    Tensor dW_h = torch::zeros({dim, dim}, options);
    Tensor db = torch::zeros({dim}, options);
    Tensor d_log_alpha = torch::zeros({1}, options.dtype(torch::kFloat32));

    // Workspace layout: [dv_all: T*BD] [dh: BD] [dh_raw: BD] [dh_recurrent: BD] [h_raw: BD]
    //                   [db_float: dim] [d_log_alpha_float: 1]
    const int64_t BD = batch_size * dim;
    const int64_t float_in_T = (dim * sizeof(float) + sizeof(float) + sizeof(float) - 1) / sizeof(float);
    const int64_t workspace_size = (time_steps + 4) * BD + float_in_T * 2;
    Tensor workspace = torch::empty({workspace_size}, options);

    // log_alpha needs to be float for the kernel
    Tensor log_alpha_float = log_alpha.to(torch::kFloat32).contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e60_residual_nonlinear_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E60ResidualNonlinearBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(W_h),
            log_alpha_float.data_ptr<float>(),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(tanh_cache),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dW_x),
            ptr<scalar_t>(dW_h),
            ptr<scalar_t>(db),
            d_log_alpha.data_ptr<float>(),
            ptr<scalar_t>(workspace));
    }));

    return {dx, dW_x, dW_h, db, d_log_alpha};
}

// =============================================================================
// E61: Decay-Gated Elman - Mamba2-style Input-Dependent Decay
// alpha_t = sigmoid(x @ W_alpha.T + b_alpha)    # Decay gate
// v_t = x @ W_v.T + b_v                         # New value (linear)
// h_t = alpha_t * h_{t-1} + (1 - alpha_t) * v_t # Gated update
// output = h * silu(h)                          # Self-gating
// =============================================================================

std::vector<Tensor> e61_decay_gated_forward(
    bool training,
    Tensor x,           // [T, B, dim] pre-activated input
    Tensor h0,          // [B, dim]
    Tensor W_alpha,     // [dim, dim] decay gate weight
    Tensor b_alpha,     // [dim] decay gate bias
    Tensor W_v,         // [dim, dim] value weight
    Tensor b_v) {       // [dim] value bias

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(W_alpha);
    CHECK_INPUT(b_alpha);
    CHECK_INPUT(W_v);
    CHECK_INPUT(b_v);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor alpha_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                  : torch::empty({0}, options);

    // Workspace: [alpha_logits: T*BD] [v: T*BD]
    const int64_t BD = batch_size * dim;
    Tensor workspace = torch::empty({2 * time_steps * BD}, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e61_decay_gated_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E61DecayGatedForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W_alpha),
            ptr<scalar_t>(b_alpha),
            ptr<scalar_t>(W_v),
            ptr<scalar_t>(b_v),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(output),
            training ? ptr<scalar_t>(alpha_cache) : nullptr,
            ptr<scalar_t>(workspace));
    }));

    return {h, output, alpha_cache};
}

std::vector<Tensor> e61_decay_gated_backward(
    Tensor W_alpha,
    Tensor W_v,
    Tensor b_v,
    Tensor x,
    Tensor h,
    Tensor alpha_cache,
    Tensor d_output) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(W_alpha);
    CHECK_INPUT(W_v);
    CHECK_INPUT(b_v);
    CHECK_INPUT(x);
    CHECK_INPUT(h);
    CHECK_INPUT(alpha_cache);
    CHECK_INPUT(d_output);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::empty_like(x);
    Tensor dW_alpha = torch::zeros({dim, dim}, options);
    Tensor db_alpha = torch::zeros({dim}, options);
    Tensor dW_v = torch::zeros({dim, dim}, options);
    Tensor db_v = torch::zeros({dim}, options);

    // Workspace layout: [dh: BD] [dh_recurrent: BD] [d_alpha_logit_all: T*BD]
    //                   [dv_all: T*BD] [v_all: T*BD] [db_alpha_float: dim] [db_v_float: dim]
    const int64_t BD = batch_size * dim;
    const int64_t float_in_T = (2 * dim * sizeof(float) + sizeof(float) - 1) / sizeof(float);
    const int64_t workspace_size = 2 * BD + 3 * time_steps * BD + float_in_T;
    Tensor workspace = torch::empty({workspace_size}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e61_decay_gated_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E61DecayGatedBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W_alpha),
            ptr<scalar_t>(W_v),
            ptr<scalar_t>(b_v),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(alpha_cache),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dW_alpha),
            ptr<scalar_t>(db_alpha),
            ptr<scalar_t>(dW_v),
            ptr<scalar_t>(db_v),
            ptr<scalar_t>(workspace));
    }));

    return {dx, dW_alpha, db_alpha, dW_v, db_v};
}

// =============================================================================
// E62: Selective Write Elman
// Vector analog of DeltaNet's selective memory updates.
// k_t = sigmoid(W_k @ x_t + b_k)     # Selection mask
// v_t = tanh(W_v @ x_t + b_v)        # New values
// h_t = (1 - k_t) * h_{t-1} + k_t * v_t   # Selective replacement
// output_t = h_t * silu(h_t)         # Self-gating
// =============================================================================

std::vector<Tensor> e62_selective_write_forward(
    bool training,
    Tensor x,           // [T, B, dim] pre-activated input
    Tensor h0,          // [B, dim]
    Tensor W_k,         // [dim, dim] selection weight
    Tensor b_k,         // [dim] selection bias
    Tensor W_v,         // [dim, dim] value weight
    Tensor b_v) {       // [dim] value bias

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(W_k);
    CHECK_INPUT(b_k);
    CHECK_INPUT(W_v);
    CHECK_INPUT(b_v);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor k_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                              : torch::empty({0}, options);
    Tensor v_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                              : torch::empty({0}, options);

    // Workspace: [tmp_Wk_x: T*BD] [tmp_Wv_x: T*BD]
    const int64_t BD = batch_size * dim;
    Tensor workspace = torch::empty({2 * time_steps * BD}, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e62_selective_write_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E62SelectiveWriteForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W_k),
            ptr<scalar_t>(b_k),
            ptr<scalar_t>(W_v),
            ptr<scalar_t>(b_v),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(output),
            training ? ptr<scalar_t>(k_cache) : nullptr,
            training ? ptr<scalar_t>(v_cache) : nullptr,
            ptr<scalar_t>(workspace));
    }));

    return {h, output, k_cache, v_cache};
}

std::vector<Tensor> e62_selective_write_backward(
    Tensor W_k,
    Tensor W_v,
    Tensor x,
    Tensor h,
    Tensor k_cache,
    Tensor v_cache,
    Tensor d_output) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(W_k);
    CHECK_INPUT(W_v);
    CHECK_INPUT(x);
    CHECK_INPUT(h);
    CHECK_INPUT(k_cache);
    CHECK_INPUT(v_cache);
    CHECK_INPUT(d_output);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::zeros({time_steps, batch_size, dim}, options);
    Tensor dW_k = torch::zeros({dim, dim}, options);
    Tensor db_k = torch::zeros({dim}, options);
    Tensor dW_v = torch::zeros({dim, dim}, options);
    Tensor db_v = torch::zeros({dim}, options);

    // Workspace: [dk_pre_all: T*BD] [dv_pre_all: T*BD] [dh: BD] [dh_recurrent: BD]
    //            [db_k_float: dim] [db_v_float: dim]
    const int64_t BD = batch_size * dim;
    const int64_t float_in_T = (2 * dim * sizeof(float) + sizeof(float) - 1) / sizeof(float);
    const int64_t workspace_size = (2 * time_steps + 2) * BD + float_in_T;
    Tensor workspace = torch::empty({workspace_size}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e62_selective_write_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E62SelectiveWriteBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W_k),
            ptr<scalar_t>(W_v),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(k_cache),
            ptr<scalar_t>(v_cache),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dW_k),
            ptr<scalar_t>(db_k),
            ptr<scalar_t>(dW_v),
            ptr<scalar_t>(db_v),
            ptr<scalar_t>(workspace));
    }));

    return {dx, dW_k, db_k, dW_v, db_v};
}

// =============================================================================
// E63: Nonlinear Delta Elman (UTM-Class Expressivity)
// alpha_t = sigmoid(W_alpha @ x_t + b_alpha)           # Retain gate (x-only)
// v_t = tanh(W_h @ h_{t-1} + W_x @ x_t + b)           # NONLINEAR value (h-dependent!)
// h_t = alpha_t * h_{t-1} + (1 - alpha_t) * v_t       # Gated mixing
// output = h * silu(h)                                 # Self-gating
// =============================================================================

std::vector<Tensor> e63_nonlinear_delta_forward(
    bool training,
    Tensor x,           // [T, B, dim] pre-activated input
    Tensor h0,          // [B, dim]
    Tensor W_alpha,     // [dim, dim] retain gate weight
    Tensor b_alpha,     // [dim] retain gate bias
    Tensor W_h,         // [dim, dim] hidden-to-value weight (nonlinear path!)
    Tensor W_x,         // [dim, dim] input-to-value weight
    Tensor b) {         // [dim] value bias

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(W_alpha);
    CHECK_INPUT(b_alpha);
    CHECK_INPUT(W_h);
    CHECK_INPUT(W_x);
    CHECK_INPUT(b);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v_pre_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                  : torch::empty({0}, options);
    Tensor alpha_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                  : torch::empty({0}, options);

    // Workspace: [alpha_x_all: T*BD] [Wx_all: T*BD] [tmp_Wh: BD]
    const int64_t BD = batch_size * dim;
    Tensor workspace = torch::empty({2 * time_steps * BD + BD}, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e63_nonlinear_delta_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E63NonlinearDeltaForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W_alpha),
            ptr<scalar_t>(b_alpha),
            ptr<scalar_t>(W_h),
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(b),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v_pre_cache) : nullptr,
            training ? ptr<scalar_t>(alpha_cache) : nullptr,
            ptr<scalar_t>(workspace));
    }));

    return {h, output, v_pre_cache, alpha_cache};
}

std::vector<Tensor> e63_nonlinear_delta_backward(
    Tensor W_alpha,
    Tensor W_h,
    Tensor W_x,
    Tensor x,
    Tensor h,
    Tensor v_pre_cache,
    Tensor alpha_cache,
    Tensor d_output) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(W_alpha);
    CHECK_INPUT(W_h);
    CHECK_INPUT(W_x);
    CHECK_INPUT(x);
    CHECK_INPUT(h);
    CHECK_INPUT(v_pre_cache);
    CHECK_INPUT(alpha_cache);
    CHECK_INPUT(d_output);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::zeros({time_steps, batch_size, dim}, options);
    Tensor dW_alpha = torch::zeros({dim, dim}, options);
    Tensor db_alpha = torch::zeros({dim}, options);
    Tensor dW_h = torch::zeros({dim, dim}, options);
    Tensor dW_x = torch::zeros({dim, dim}, options);
    Tensor db = torch::zeros({dim}, options);

    // Workspace: [dh: BD] [dh_recurrent: BD] [dh_prev: BD]
    //            [dv_pre_all: T*BD] [dalpha_x_all: T*BD]
    //            [db_float: dim] [db_alpha_float: dim]
    //            [alpha_x_all: T*BD] (recompute)
    const int64_t BD = batch_size * dim;
    const int64_t float_in_T = (2 * dim * sizeof(float) + sizeof(float) - 1) / sizeof(float);
    const int64_t workspace_size = 3 * BD + 2 * time_steps * BD + float_in_T + time_steps * BD;
    Tensor workspace = torch::empty({workspace_size}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e63_nonlinear_delta_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E63NonlinearDeltaBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W_alpha),
            ptr<scalar_t>(W_h),
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v_pre_cache),
            ptr<scalar_t>(alpha_cache),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dW_alpha),
            ptr<scalar_t>(db_alpha),
            ptr<scalar_t>(dW_h),
            ptr<scalar_t>(dW_x),
            ptr<scalar_t>(db),
            ptr<scalar_t>(workspace));
    }));

    return {dx, dW_alpha, db_alpha, dW_h, dW_x, db};
}

// =============================================================================
// =============================================================================
// E64: Additive H-Dependence - Cheapest UTM-Class Recurrence
// alpha_t = sigmoid(W_alpha @ x_t + b_alpha)    # Retain gate (x-only)
// v_t = tanh(h_{t-1} + W_x @ x_t + b)          # ADDITIVE h-dependence (no W_h!)
// h_t = alpha_t * h_{t-1} + (1 - alpha_t) * v_t  # Gated mixing
// output = h * silu(h)                          # Self-gating
// =============================================================================

std::vector<Tensor> e64_additive_h_forward(
    bool training,
    Tensor x,           // [T, B, dim] pre-activated input
    Tensor h0,          // [B, dim]
    Tensor W_alpha,     // [dim, dim] retain gate weight
    Tensor b_alpha,     // [dim] retain gate bias
    Tensor W_x,         // [dim, dim] input-to-value weight
    Tensor b) {         // [dim] value bias

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(W_alpha);
    CHECK_INPUT(b_alpha);
    CHECK_INPUT(W_x);
    CHECK_INPUT(b);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v_pre_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                  : torch::empty({0}, options);
    Tensor alpha_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                  : torch::empty({0}, options);

    // Workspace: [alpha_x_all: T*BD] [Wx_all: T*BD]
    const int64_t BD = batch_size * dim;
    Tensor workspace = torch::empty({2 * time_steps * BD}, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e64_additive_h_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E64AdditiveHForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W_alpha),
            ptr<scalar_t>(b_alpha),
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(b),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v_pre_cache) : nullptr,
            training ? ptr<scalar_t>(alpha_cache) : nullptr,
            ptr<scalar_t>(workspace));
    }));

    return {h, output, v_pre_cache, alpha_cache};
}

std::vector<Tensor> e64_additive_h_backward(
    Tensor W_alpha,
    Tensor W_x,
    Tensor x,
    Tensor h,
    Tensor v_pre_cache,
    Tensor alpha_cache,
    Tensor d_output) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(W_alpha);
    CHECK_INPUT(W_x);
    CHECK_INPUT(x);
    CHECK_INPUT(h);
    CHECK_INPUT(v_pre_cache);
    CHECK_INPUT(alpha_cache);
    CHECK_INPUT(d_output);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::zeros({time_steps, batch_size, dim}, options);
    Tensor dW_alpha = torch::zeros({dim, dim}, options);
    Tensor db_alpha = torch::zeros({dim}, options);
    Tensor dW_x = torch::zeros({dim, dim}, options);
    Tensor db = torch::zeros({dim}, options);

    // Workspace: [dh: BD] [dh_recurrent: BD] [dh_prev: BD]
    //            [dWx_pre_all: T*BD] [dalpha_x_all: T*BD]
    //            [db_float: dim] [db_alpha_float: dim]
    const int64_t BD = batch_size * dim;
    const int64_t float_in_T = (2 * dim * sizeof(float) + sizeof(float) - 1) / sizeof(float);
    const int64_t workspace_size = 3 * BD + 2 * time_steps * BD + float_in_T;
    Tensor workspace = torch::empty({workspace_size}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e64_additive_h_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E64AdditiveHBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W_alpha),
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v_pre_cache),
            ptr<scalar_t>(alpha_cache),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dW_alpha),
            ptr<scalar_t>(db_alpha),
            ptr<scalar_t>(dW_x),
            ptr<scalar_t>(db),
            ptr<scalar_t>(workspace));
    }));

    return {dx, dW_alpha, db_alpha, dW_x, db};
}

// E65: Diagonal H-Dependence - Learnable Per-Dimension Scaling
// alpha_t = sigmoid(W_alpha @ x_t + b_alpha)        # Decay gate (x-only)
// v_t = tanh(d_h * h_{t-1} + W_x @ x_t + b)         # d_h is [dim] diagonal vector
// h_t = alpha_t * h_{t-1} + (1 - alpha_t) * v_t    # Gated update
// output = h * silu(h)                              # Self-gating
// O(d) h-dependence instead of O(d^2), UTM-class
// =============================================================================

std::vector<Tensor> e65_diagonal_h_forward(
    bool training,
    Tensor x,           // [T, B, dim] pre-activated input
    Tensor h0,          // [B, dim]
    Tensor W_alpha,     // [dim, dim] decay gate weight
    Tensor b_alpha,     // [dim] decay gate bias
    Tensor d_h,         // [dim] diagonal h-scaling vector
    Tensor W_x,         // [dim, dim] input-to-value weight
    Tensor b) {         // [dim] value bias

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(W_alpha);
    CHECK_INPUT(b_alpha);
    CHECK_INPUT(d_h);
    CHECK_INPUT(W_x);
    CHECK_INPUT(b);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v_pre_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                  : torch::empty({0}, options);
    Tensor alpha_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                  : torch::empty({0}, options);

    // Workspace: [alpha_logits_all: T*BD] [Wx_all: T*BD]
    const int64_t BD = batch_size * dim;
    Tensor workspace = torch::empty({2 * time_steps * BD}, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e65_diagonal_h_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E65DiagonalHForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W_alpha),
            ptr<scalar_t>(b_alpha),
            ptr<scalar_t>(d_h),
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(b),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v_pre_cache) : nullptr,
            training ? ptr<scalar_t>(alpha_cache) : nullptr,
            ptr<scalar_t>(workspace));
    }));

    return {h, output, v_pre_cache, alpha_cache};
}

std::vector<Tensor> e65_diagonal_h_backward(
    Tensor W_alpha,
    Tensor d_h,
    Tensor W_x,
    Tensor x,
    Tensor h,
    Tensor v_pre_cache,
    Tensor alpha_cache,
    Tensor d_output) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(W_alpha);
    CHECK_INPUT(d_h);
    CHECK_INPUT(W_x);
    CHECK_INPUT(x);
    CHECK_INPUT(h);
    CHECK_INPUT(v_pre_cache);
    CHECK_INPUT(alpha_cache);
    CHECK_INPUT(d_output);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::zeros({time_steps, batch_size, dim}, options);
    Tensor dW_alpha = torch::zeros({dim, dim}, options);
    Tensor db_alpha = torch::zeros({dim}, options);
    Tensor dd_h = torch::zeros({dim}, options);  // gradient for diagonal
    Tensor dW_x = torch::zeros({dim, dim}, options);
    Tensor db = torch::zeros({dim}, options);

    // Workspace: [dh: BD] [dh_recurrent: BD]
    //            [d_alpha_logit_all: T*BD] [dWx_all: T*BD]
    //            [db_alpha_float: dim] [db_float: dim] [d_d_h_float: dim]
    const int64_t BD = batch_size * dim;
    const int64_t float_in_T = (3 * dim * sizeof(float) + sizeof(float) - 1) / sizeof(float);
    const int64_t workspace_size = 2 * BD + 2 * time_steps * BD + float_in_T;
    Tensor workspace = torch::empty({workspace_size}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e65_diagonal_h_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E65DiagonalHBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W_alpha),
            ptr<scalar_t>(d_h),
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v_pre_cache),
            ptr<scalar_t>(alpha_cache),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dW_alpha),
            ptr<scalar_t>(db_alpha),
            ptr<scalar_t>(dd_h),
            ptr<scalar_t>(dW_x),
            ptr<scalar_t>(db),
            ptr<scalar_t>(workspace));
    }));

    return {dx, dW_alpha, db_alpha, dd_h, dW_x, db};
}

// =============================================================================
// E66: Low-Rank H-Dependence (UTM-Class with Cross-Dimension Mixing)
// alpha_t = sigmoid(W_alpha @ x_t + b_alpha)           # Retain gate (x-only)
// h_compressed = V @ h_{t-1}                           # Compress h to rank
// h_transformed = U @ h_compressed                     # Expand back to dim
// v_t = tanh(h_transformed + W_x @ x_t + b)           # NONLINEAR value
// h_t = alpha_t * h_{t-1} + (1 - alpha_t) * v_t       # Gated mixing
// output = h * silu(h)                                 # Self-gating
// Key: O(d*rank) cost per timestep instead of O(d^2)
// =============================================================================

std::vector<Tensor> e66_lowrank_h_forward(
    bool training,
    Tensor x,           // [T, B, dim] pre-activated input
    Tensor h0,          // [B, dim]
    Tensor W_alpha,     // [dim, dim] retain gate weight
    Tensor b_alpha,     // [dim] retain gate bias
    Tensor U,           // [dim, rank] expand matrix
    Tensor V,           // [rank, dim] compress matrix
    Tensor W_x,         // [dim, dim] input-to-value weight
    Tensor b) {         // [dim] value bias

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    const auto rank = V.size(0);  // V is [rank, dim]

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(W_alpha);
    CHECK_INPUT(b_alpha);
    CHECK_INPUT(U);
    CHECK_INPUT(V);
    CHECK_INPUT(W_x);
    CHECK_INPUT(b);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v_pre_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                  : torch::empty({0}, options);
    Tensor alpha_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                  : torch::empty({0}, options);
    Tensor Vh_cache = training ? torch::empty({time_steps, batch_size, rank}, options)
                               : torch::empty({0}, options);

    // Workspace: [alpha_x_all: T*BD] [Wx_all: T*BD] [tmp_Vh: BR] [tmp_Uh: BD]
    const int64_t BD = batch_size * dim;
    const int64_t BR = batch_size * rank;
    Tensor workspace = torch::empty({2 * time_steps * BD + BR + BD}, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e66_lowrank_h_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E66LowRankHForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, rank,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W_alpha),
            ptr<scalar_t>(b_alpha),
            ptr<scalar_t>(U),
            ptr<scalar_t>(V),
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(b),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v_pre_cache) : nullptr,
            training ? ptr<scalar_t>(alpha_cache) : nullptr,
            training ? ptr<scalar_t>(Vh_cache) : nullptr,
            ptr<scalar_t>(workspace));
    }));

    return {h, output, v_pre_cache, alpha_cache, Vh_cache};
}

std::vector<Tensor> e66_lowrank_h_backward(
    Tensor W_alpha,
    Tensor U,
    Tensor V,
    Tensor W_x,
    Tensor x,
    Tensor h,
    Tensor v_pre_cache,
    Tensor alpha_cache,
    Tensor Vh_cache,
    Tensor d_output) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    const auto rank = V.size(0);

    CHECK_INPUT(W_alpha);
    CHECK_INPUT(U);
    CHECK_INPUT(V);
    CHECK_INPUT(W_x);
    CHECK_INPUT(x);
    CHECK_INPUT(h);
    CHECK_INPUT(v_pre_cache);
    CHECK_INPUT(alpha_cache);
    CHECK_INPUT(Vh_cache);
    CHECK_INPUT(d_output);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::zeros({time_steps, batch_size, dim}, options);
    Tensor dW_alpha = torch::zeros({dim, dim}, options);
    Tensor db_alpha = torch::zeros({dim}, options);
    Tensor dU = torch::zeros({dim, rank}, options);
    Tensor dV = torch::zeros({rank, dim}, options);
    Tensor dW_x = torch::zeros({dim, dim}, options);
    Tensor db = torch::zeros({dim}, options);

    // Workspace: [dh: BD] [dh_recurrent: BD] [dh_prev: BD]
    //            [dv_pre_all: T*BD] [dalpha_x_all: T*BD] [dUh_all: T*BD]
    //            [db_float: dim] [db_alpha_float: dim]
    //            [alpha_x_all: T*BD] [tmp_dVh: BR]
    const int64_t BD = batch_size * dim;
    const int64_t BR = batch_size * rank;
    const int64_t float_in_T = (2 * dim * sizeof(float) + sizeof(float) - 1) / sizeof(float);
    const int64_t workspace_size = 3 * BD + 4 * time_steps * BD + float_in_T + BR;
    Tensor workspace = torch::empty({workspace_size}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e66_lowrank_h_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E66LowRankHBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, rank,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W_alpha),
            ptr<scalar_t>(U),
            ptr<scalar_t>(V),
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v_pre_cache),
            ptr<scalar_t>(alpha_cache),
            ptr<scalar_t>(Vh_cache),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dW_alpha),
            ptr<scalar_t>(db_alpha),
            ptr<scalar_t>(dU),
            ptr<scalar_t>(dV),
            ptr<scalar_t>(dW_x),
            ptr<scalar_t>(db),
            ptr<scalar_t>(workspace));
    }));

    return {dx, dW_alpha, db_alpha, dU, dV, dW_x, db};
}

// =============================================================================
// E67: H-Dependent Gate Only - Nonlinearity Through Gate Selection
// alpha_t = sigmoid(W_alpha @ x_t + d_alpha * h_{t-1} + b_alpha)   # h affects gate!
// v_t = tanh(W_x @ x_t + b_v)                                      # v is h-independent
// h_t = alpha_t * h_{t-1} + (1 - alpha_t) * v_t
// output = h * silu(h)                                             # Self-gating
// =============================================================================

std::vector<Tensor> e67_h_gated_forward(
    bool training,
    Tensor x,           // [T, B, dim] pre-activated input
    Tensor h0,          // [B, dim]
    Tensor W_alpha,     // [dim, dim] gate weight
    Tensor d_alpha,     // [dim] diagonal for h in gate
    Tensor b_alpha,     // [dim] gate bias
    Tensor W_x,         // [dim, dim] value weight
    Tensor b_v) {       // [dim] value bias

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(W_alpha);
    CHECK_INPUT(d_alpha);
    CHECK_INPUT(b_alpha);
    CHECK_INPUT(W_x);
    CHECK_INPUT(b_v);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                              : torch::empty({0}, options);
    Tensor alpha_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                  : torch::empty({0}, options);

    // Workspace: [alpha_x_all: T*BD] [v_all: T*BD]
    const int64_t BD = batch_size * dim;
    Tensor workspace = torch::empty({2 * time_steps * BD}, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e67_h_gated_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E67HGatedForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W_alpha),
            ptr<scalar_t>(d_alpha),
            ptr<scalar_t>(b_alpha),
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(b_v),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v_cache) : nullptr,
            training ? ptr<scalar_t>(alpha_cache) : nullptr,
            ptr<scalar_t>(workspace));
    }));

    return {h, output, v_cache, alpha_cache};
}

std::vector<Tensor> e67_h_gated_backward(
    Tensor W_alpha,
    Tensor d_alpha,
    Tensor W_x,
    Tensor x,
    Tensor h,
    Tensor v_cache,
    Tensor alpha_cache,
    Tensor d_output) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(W_alpha);
    CHECK_INPUT(d_alpha);
    CHECK_INPUT(W_x);
    CHECK_INPUT(x);
    CHECK_INPUT(h);
    CHECK_INPUT(v_cache);
    CHECK_INPUT(alpha_cache);
    CHECK_INPUT(d_output);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::zeros({time_steps, batch_size, dim}, options);
    Tensor dW_alpha = torch::zeros({dim, dim}, options);
    Tensor dd_alpha = torch::zeros({dim}, options);
    Tensor db_alpha = torch::zeros({dim}, options);
    Tensor dW_x = torch::zeros({dim, dim}, options);
    Tensor db_v = torch::zeros({dim}, options);

    // Workspace: [dh: BD] [dh_recurrent: BD] [dh_prev: BD]
    //            [dalpha_x_all: T*BD] [dWx_pre_all: T*BD]
    //            [dd_alpha_float: dim] [db_alpha_float: dim] [db_v_float: dim]
    const int64_t BD = batch_size * dim;
    const int64_t float_in_T = (3 * dim * sizeof(float) + sizeof(float) - 1) / sizeof(float);
    const int64_t workspace_size = 3 * BD + 2 * time_steps * BD + float_in_T;
    Tensor workspace = torch::empty({workspace_size}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e67_h_gated_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E67HGatedBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W_alpha),
            ptr<scalar_t>(d_alpha),
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v_cache),
            ptr<scalar_t>(alpha_cache),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dW_alpha),
            ptr<scalar_t>(dd_alpha),
            ptr<scalar_t>(db_alpha),
            ptr<scalar_t>(dW_x),
            ptr<scalar_t>(db_v),
            ptr<scalar_t>(workspace));
    }));

    return {dx, dW_alpha, dd_alpha, db_alpha, dW_x, db_v};
}

// =============================================================================
// E68: Self-Gating Elman - Multiplicative H-Dependence
// alpha_t = sigmoid(x @ W_alpha.T + b_alpha)        # Retain gate (input-dependent)
// g_t = sigmoid(d_g * h_{t-1} + b_g)               # SELF-GATING: h gates the value!
// v_raw_t = tanh(x @ W_x.T + b_v)                  # Raw new value
// v_t = v_raw_t * g_t                              # Gated value
// h_t = alpha_t * h_{t-1} + (1 - alpha_t) * v_t    # Gated update
// output = h * silu(h)                             # Self-gating output
// =============================================================================

std::vector<Tensor> e68_self_gating_forward(
    bool training,
    Tensor x,           // [T, B, dim] pre-activated input
    Tensor h0,          // [B, dim]
    Tensor W_alpha,     // [dim, dim] retain gate weight
    Tensor b_alpha,     // [dim] retain gate bias
    Tensor W_x,         // [dim, dim] value weight
    Tensor b_v,         // [dim] value bias
    Tensor d_g,         // [dim] diagonal gating weights (self-gating)
    Tensor b_g) {       // [dim] gating bias

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(W_alpha);
    CHECK_INPUT(b_alpha);
    CHECK_INPUT(W_x);
    CHECK_INPUT(b_v);
    CHECK_INPUT(d_g);
    CHECK_INPUT(b_g);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor alpha_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                  : torch::empty({0}, options);
    Tensor g_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                              : torch::empty({0}, options);
    Tensor v_raw_tanh_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                       : torch::empty({0}, options);

    // Workspace: [alpha_logits: T*BD] [v_raw: T*BD]
    const int64_t BD = batch_size * dim;
    Tensor workspace = torch::empty({2 * time_steps * BD}, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e68_self_gating_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E68SelfGatingForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W_alpha),
            ptr<scalar_t>(b_alpha),
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(b_v),
            ptr<scalar_t>(d_g),
            ptr<scalar_t>(b_g),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(output),
            training ? ptr<scalar_t>(alpha_cache) : nullptr,
            training ? ptr<scalar_t>(g_cache) : nullptr,
            training ? ptr<scalar_t>(v_raw_tanh_cache) : nullptr,
            ptr<scalar_t>(workspace));
    }));

    return {h, output, alpha_cache, g_cache, v_raw_tanh_cache};
}

std::vector<Tensor> e68_self_gating_backward(
    Tensor W_alpha,
    Tensor W_x,
    Tensor d_g,
    Tensor x,
    Tensor h,
    Tensor alpha_cache,
    Tensor g_cache,
    Tensor v_raw_tanh_cache,
    Tensor d_output) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(W_alpha);
    CHECK_INPUT(W_x);
    CHECK_INPUT(d_g);
    CHECK_INPUT(x);
    CHECK_INPUT(h);
    CHECK_INPUT(alpha_cache);
    CHECK_INPUT(g_cache);
    CHECK_INPUT(v_raw_tanh_cache);
    CHECK_INPUT(d_output);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::zeros({time_steps, batch_size, dim}, options);
    Tensor dW_alpha = torch::zeros({dim, dim}, options);
    Tensor db_alpha = torch::zeros({dim}, options);
    Tensor dW_x = torch::zeros({dim, dim}, options);
    Tensor db_v = torch::zeros({dim}, options);
    Tensor dd_g = torch::zeros({dim}, options);  // gradient for d_g
    Tensor db_g = torch::zeros({dim}, options);  // gradient for b_g

    // Workspace: [dh: BD] [dh_recurrent: BD]
    //            [d_alpha_logit_all: T*BD] [dv_raw_all: T*BD]
    //            [db_alpha_float: dim] [dd_g_float: dim] [db_g_float: dim] [db_v_float: dim]
    const int64_t BD = batch_size * dim;
    const int64_t float_in_T = (4 * dim * sizeof(float) + sizeof(float) - 1) / sizeof(float);
    const int64_t workspace_size = 2 * BD + 2 * time_steps * BD + float_in_T;
    Tensor workspace = torch::empty({workspace_size}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e68_self_gating_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        E68SelfGatingBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W_alpha),
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(d_g),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(alpha_cache),
            ptr<scalar_t>(g_cache),
            ptr<scalar_t>(v_raw_tanh_cache),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dW_alpha),
            ptr<scalar_t>(db_alpha),
            ptr<scalar_t>(dW_x),
            ptr<scalar_t>(db_v),
            ptr<scalar_t>(dd_g),
            ptr<scalar_t>(db_g),
            ptr<scalar_t>(workspace));
    }));

    return {dx, dW_alpha, db_alpha, dW_x, db_v, dd_g, db_g};
}

// =============================================================================
// GRU: Gated Recurrent Unit - BF16 Optimized
// =============================================================================

std::vector<Tensor> gru_forward(
    bool training,
    Tensor x,           // [T, B, dim]
    Tensor h0,          // [B, dim]
    Tensor W_zr,        // [2*dim, dim] - W_z and W_r stacked
    Tensor W_h,         // [dim, dim]
    Tensor U_zr,        // [2*dim, dim] - U_z and U_r stacked
    Tensor U_h,         // [dim, dim]
    Tensor b_zr,        // [2*dim] - b_z and b_r
    Tensor b_h) {       // [dim]

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(W_zr);
    CHECK_INPUT(W_h);
    CHECK_INPUT(U_zr);
    CHECK_INPUT(U_h);
    CHECK_INPUT(b_zr);
    CHECK_INPUT(b_h);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor z_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                              : torch::empty({0}, options);
    Tensor h_tilde_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                    : torch::empty({0}, options);
    Tensor r_h_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                : torch::empty({0}, options);

    // Workspace
    using namespace hasty::v0::elman_ladder;
    const int64_t workspace_size = GRUForward<__nv_bfloat16>::WorkspaceSize(time_steps, batch_size, dim);
    Tensor workspace = torch::empty({workspace_size}, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "gru_forward", ([&] {
        GRUForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W_zr),
            ptr<scalar_t>(W_h),
            ptr<scalar_t>(U_zr),
            ptr<scalar_t>(U_h),
            ptr<scalar_t>(b_zr),
            ptr<scalar_t>(b_h),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            training ? ptr<scalar_t>(z_cache) : nullptr,
            training ? ptr<scalar_t>(h_tilde_cache) : nullptr,
            training ? ptr<scalar_t>(r_h_cache) : nullptr,
            ptr<scalar_t>(workspace));
    }));

    return {h, z_cache, h_tilde_cache, r_h_cache};
}

std::vector<Tensor> gru_backward(
    Tensor W_zr,
    Tensor W_h,
    Tensor U_zr,
    Tensor U_h,
    Tensor b_zr,
    Tensor b_h,
    Tensor x,
    Tensor h,
    Tensor z_cache,
    Tensor h_tilde_cache,
    Tensor r_h_cache,
    Tensor d_h_final) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(W_zr);
    CHECK_INPUT(W_h);
    CHECK_INPUT(U_zr);
    CHECK_INPUT(U_h);
    CHECK_INPUT(b_zr);
    CHECK_INPUT(b_h);
    CHECK_INPUT(x);
    CHECK_INPUT(h);
    CHECK_INPUT(z_cache);
    CHECK_INPUT(h_tilde_cache);
    CHECK_INPUT(r_h_cache);
    CHECK_INPUT(d_h_final);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::empty_like(x);
    Tensor dW_zr = torch::zeros({2 * dim, dim}, options);
    Tensor dW_h = torch::zeros({dim, dim}, options);
    Tensor dU_zr = torch::zeros({2 * dim, dim}, options);
    Tensor dU_h = torch::zeros({dim, dim}, options);
    Tensor db_zr = torch::zeros({2 * dim}, options);
    Tensor db_h = torch::zeros({dim}, options);

    using namespace hasty::v0::elman_ladder;
    const int64_t workspace_size = GRUBackward<__nv_bfloat16>::WorkspaceSize(time_steps, batch_size, dim);
    Tensor workspace = torch::empty({workspace_size}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "gru_backward", ([&] {
        GRUBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W_zr),
            ptr<scalar_t>(W_h),
            ptr<scalar_t>(U_zr),
            ptr<scalar_t>(U_h),
            ptr<scalar_t>(b_zr),
            ptr<scalar_t>(b_h),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(z_cache),
            ptr<scalar_t>(h_tilde_cache),
            ptr<scalar_t>(r_h_cache),
            ptr<scalar_t>(d_h_final),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dW_zr),
            ptr<scalar_t>(dW_h),
            ptr<scalar_t>(dU_zr),
            ptr<scalar_t>(dU_h),
            ptr<scalar_t>(db_zr),
            ptr<scalar_t>(db_h),
            ptr<scalar_t>(workspace));
    }));

    return {dx, dW_zr, dW_h, dU_zr, dU_h, db_zr, db_h};
}

// =============================================================================
// LSTM: Long Short-Term Memory - BF16 Optimized
// =============================================================================

std::vector<Tensor> lstm_forward(
    bool training,
    Tensor x,           // [T, B, dim]
    Tensor h0,          // [B, dim]
    Tensor c0,          // [B, dim]
    Tensor W_fio,       // [3*dim, dim] - W_f, W_i, W_o stacked
    Tensor W_c,         // [dim, dim]
    Tensor U_fio,       // [3*dim, dim] - U_f, U_i, U_o stacked
    Tensor U_c,         // [dim, dim]
    Tensor b_fio,       // [3*dim] - b_f, b_i, b_o
    Tensor b_c) {       // [dim]

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(c0);
    CHECK_INPUT(W_fio);
    CHECK_INPUT(W_c);
    CHECK_INPUT(U_fio);
    CHECK_INPUT(U_c);
    CHECK_INPUT(b_fio);
    CHECK_INPUT(b_c);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor c = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor f_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                              : torch::empty({0}, options);
    Tensor i_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                              : torch::empty({0}, options);
    Tensor o_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                              : torch::empty({0}, options);
    Tensor c_tilde_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                    : torch::empty({0}, options);
    Tensor tanh_c_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                   : torch::empty({0}, options);

    // Workspace
    using namespace hasty::v0::elman_ladder;
    const int64_t workspace_size = LSTMForward<__nv_bfloat16>::WorkspaceSize(time_steps, batch_size, dim);
    Tensor workspace = torch::empty({workspace_size}, options);

    h[0] = h0;
    c[0] = c0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "lstm_forward", ([&] {
        LSTMForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W_fio),
            ptr<scalar_t>(W_c),
            ptr<scalar_t>(U_fio),
            ptr<scalar_t>(U_c),
            ptr<scalar_t>(b_fio),
            ptr<scalar_t>(b_c),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(c),
            training ? ptr<scalar_t>(f_cache) : nullptr,
            training ? ptr<scalar_t>(i_cache) : nullptr,
            training ? ptr<scalar_t>(o_cache) : nullptr,
            training ? ptr<scalar_t>(c_tilde_cache) : nullptr,
            training ? ptr<scalar_t>(tanh_c_cache) : nullptr,
            ptr<scalar_t>(workspace));
    }));

    return {h, c, f_cache, i_cache, o_cache, c_tilde_cache, tanh_c_cache};
}

std::vector<Tensor> lstm_backward(
    Tensor W_fio,
    Tensor W_c,
    Tensor U_fio,
    Tensor U_c,
    Tensor x,
    Tensor h,
    Tensor c,
    Tensor f_cache,
    Tensor i_cache,
    Tensor o_cache,
    Tensor c_tilde_cache,
    Tensor tanh_c_cache,
    Tensor dh_all,      // [T, B, dim] gradients on ALL hidden states (can be empty)
    Tensor d_c_final) { // [B, dim] gradient on final cell state (can be empty)

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(W_fio);
    CHECK_INPUT(W_c);
    CHECK_INPUT(U_fio);
    CHECK_INPUT(U_c);
    CHECK_INPUT(x);
    CHECK_INPUT(h);
    CHECK_INPUT(c);
    CHECK_INPUT(f_cache);
    CHECK_INPUT(i_cache);
    CHECK_INPUT(o_cache);
    CHECK_INPUT(c_tilde_cache);
    CHECK_INPUT(tanh_c_cache);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::empty_like(x);
    Tensor dW_fio = torch::zeros({3 * dim, dim}, options);
    Tensor dW_c = torch::zeros({dim, dim}, options);
    Tensor dU_fio = torch::zeros({3 * dim, dim}, options);
    Tensor dU_c = torch::zeros({dim, dim}, options);
    Tensor db_fio = torch::zeros({3 * dim}, options);
    Tensor db_c = torch::zeros({dim}, options);

    using namespace hasty::v0::elman_ladder;
    const int64_t workspace_size = LSTMBackward<__nv_bfloat16>::WorkspaceSize(time_steps, batch_size, dim);
    Tensor workspace = torch::empty({workspace_size}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "lstm_backward", ([&] {
        LSTMBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W_fio),
            ptr<scalar_t>(W_c),
            ptr<scalar_t>(U_fio),
            ptr<scalar_t>(U_c),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(c),
            ptr<scalar_t>(f_cache),
            ptr<scalar_t>(i_cache),
            ptr<scalar_t>(o_cache),
            ptr<scalar_t>(c_tilde_cache),
            ptr<scalar_t>(tanh_c_cache),
            dh_all.numel() > 0 ? ptr<scalar_t>(dh_all) : nullptr,
            d_c_final.numel() > 0 ? ptr<scalar_t>(d_c_final) : nullptr,
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dW_fio),
            ptr<scalar_t>(dW_c),
            ptr<scalar_t>(dU_fio),
            ptr<scalar_t>(dU_c),
            ptr<scalar_t>(db_fio),
            ptr<scalar_t>(db_c),
            ptr<scalar_t>(workspace));
    }));

    return {dx, dW_fio, dW_c, dU_fio, dU_c, db_fio, db_c};
}

// =============================================================================
// E63m: Matrix State Nonlinear Delta - Maximum Expressivity
// Matrix state S ∈ ℝ^(N×D) with NONLINEAR retrieval and update.
// This is UTM-class (nonlinear h-dependence prevents parallel scan).
// =============================================================================

std::vector<Tensor> e63m_matrix_nonlinear_forward(
    bool training,
    Tensor x,           // [T, B, D] input
    Tensor S0,          // [B, N, D] initial state matrix
    Tensor W_k,         // [D, D] key projection
    Tensor W_q,         // [D, D] query projection
    Tensor W_x,         // [N, D] input-to-value projection
    Tensor W_r,         // [N, N] retrieval transformation
    Tensor b,           // [N] value bias
    Tensor W_alpha,     // [N, D] gate projection
    Tensor b_alpha) {   // [N] gate bias

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    const auto n_slots = W_x.size(0);

    CHECK_INPUT(x);
    CHECK_INPUT(S0);
    CHECK_INPUT(W_k);
    CHECK_INPUT(W_q);
    CHECK_INPUT(W_x);
    CHECK_INPUT(W_r);
    CHECK_INPUT(b);
    CHECK_INPUT(W_alpha);
    CHECK_INPUT(b_alpha);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    // Outputs
    Tensor S = torch::empty({time_steps + 1, batch_size, n_slots, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, n_slots}, options);

    // Caches for backward
    Tensor k_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                              : torch::empty({0}, options);
    Tensor q_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                              : torch::empty({0}, options);
    Tensor Wx_cache = training ? torch::empty({time_steps, batch_size, n_slots}, options)
                               : torch::empty({0}, options);
    Tensor alpha_x_cache = training ? torch::empty({time_steps, batch_size, n_slots}, options)
                                    : torch::empty({0}, options);
    Tensor Sk_cache = training ? torch::empty({time_steps, batch_size, n_slots}, options)
                               : torch::empty({0}, options);
    Tensor retrieved_cache = training ? torch::empty({time_steps, batch_size, n_slots}, options)
                                      : torch::empty({0}, options);
    Tensor Wr_ret_cache = training ? torch::empty({time_steps, batch_size, n_slots}, options)
                                   : torch::empty({0}, options);
    Tensor v_cache = training ? torch::empty({time_steps, batch_size, n_slots}, options)
                              : torch::empty({0}, options);
    Tensor alpha_cache = training ? torch::empty({time_steps, batch_size, n_slots}, options)
                                  : torch::empty({0}, options);

    // Workspace
    using namespace hasty::v0::elman_ladder;
    const int64_t workspace_size = E63mMatrixNonlinearForward<__nv_bfloat16>::WorkspaceSize(
        time_steps, batch_size, n_slots, dim);
    Tensor workspace = torch::empty({workspace_size}, options);

    S[0] = S0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e63m_matrix_nonlinear_forward", ([&] {
        E63mMatrixNonlinearForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, n_slots, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W_k),
            ptr<scalar_t>(W_q),
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(W_r),
            ptr<scalar_t>(b),
            ptr<scalar_t>(W_alpha),
            ptr<scalar_t>(b_alpha),
            ptr<scalar_t>(x),
            ptr<scalar_t>(S),
            ptr<scalar_t>(output),
            training ? ptr<scalar_t>(k_cache) : nullptr,
            training ? ptr<scalar_t>(q_cache) : nullptr,
            training ? ptr<scalar_t>(Wx_cache) : nullptr,
            training ? ptr<scalar_t>(alpha_x_cache) : nullptr,
            training ? ptr<scalar_t>(Sk_cache) : nullptr,
            training ? ptr<scalar_t>(retrieved_cache) : nullptr,
            training ? ptr<scalar_t>(Wr_ret_cache) : nullptr,
            training ? ptr<scalar_t>(v_cache) : nullptr,
            training ? ptr<scalar_t>(alpha_cache) : nullptr,
            ptr<scalar_t>(workspace));
    }));

    return {S, output, k_cache, q_cache, Wx_cache, alpha_x_cache,
            Sk_cache, retrieved_cache, Wr_ret_cache, v_cache, alpha_cache};
}

std::vector<Tensor> e63m_matrix_nonlinear_backward(
    Tensor W_k,
    Tensor W_q,
    Tensor W_x,
    Tensor W_r,
    Tensor b,
    Tensor W_alpha,
    Tensor b_alpha,
    Tensor x,
    Tensor S,
    Tensor output,
    Tensor k_cache,
    Tensor q_cache,
    Tensor Wx_cache,
    Tensor alpha_x_cache,
    Tensor Sk_cache,
    Tensor retrieved_cache,
    Tensor Wr_ret_cache,
    Tensor v_cache,
    Tensor alpha_cache,
    Tensor d_output) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    const auto n_slots = W_x.size(0);

    CHECK_INPUT(W_k);
    CHECK_INPUT(W_q);
    CHECK_INPUT(W_x);
    CHECK_INPUT(W_r);
    CHECK_INPUT(b);
    CHECK_INPUT(W_alpha);
    CHECK_INPUT(b_alpha);
    CHECK_INPUT(x);
    CHECK_INPUT(S);
    CHECK_INPUT(output);
    CHECK_INPUT(k_cache);
    CHECK_INPUT(q_cache);
    CHECK_INPUT(Wx_cache);
    CHECK_INPUT(alpha_x_cache);
    CHECK_INPUT(Sk_cache);
    CHECK_INPUT(retrieved_cache);
    CHECK_INPUT(Wr_ret_cache);
    CHECK_INPUT(v_cache);
    CHECK_INPUT(alpha_cache);
    CHECK_INPUT(d_output);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    // Output gradients
    Tensor dx = torch::empty_like(x);
    Tensor dW_k = torch::zeros({dim, dim}, options);
    Tensor dW_q = torch::zeros({dim, dim}, options);
    Tensor dW_x = torch::zeros({n_slots, dim}, options);
    Tensor dW_r = torch::zeros({n_slots, n_slots}, options);
    Tensor db = torch::zeros({n_slots}, options);
    Tensor dW_alpha = torch::zeros({n_slots, dim}, options);
    Tensor db_alpha = torch::zeros({n_slots}, options);

    // Workspace
    using namespace hasty::v0::elman_ladder;
    const int64_t workspace_size = E63mMatrixNonlinearBackward<__nv_bfloat16>::WorkspaceSize(
        time_steps, batch_size, n_slots, dim);
    Tensor workspace = torch::empty({workspace_size}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e63m_matrix_nonlinear_backward", ([&] {
        E63mMatrixNonlinearBackward<typename native_type<scalar_t>::T> backward(
            batch_size, n_slots, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W_k),
            ptr<scalar_t>(W_q),
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(W_r),
            ptr<scalar_t>(b),
            ptr<scalar_t>(W_alpha),
            ptr<scalar_t>(b_alpha),
            ptr<scalar_t>(x),
            ptr<scalar_t>(S),
            ptr<scalar_t>(output),
            ptr<scalar_t>(k_cache),
            ptr<scalar_t>(q_cache),
            ptr<scalar_t>(Wx_cache),
            ptr<scalar_t>(alpha_x_cache),
            ptr<scalar_t>(Sk_cache),
            ptr<scalar_t>(retrieved_cache),
            ptr<scalar_t>(Wr_ret_cache),
            ptr<scalar_t>(v_cache),
            ptr<scalar_t>(alpha_cache),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dW_k),
            ptr<scalar_t>(dW_q),
            ptr<scalar_t>(dW_x),
            ptr<scalar_t>(dW_r),
            ptr<scalar_t>(db),
            ptr<scalar_t>(dW_alpha),
            ptr<scalar_t>(db_alpha),
            ptr<scalar_t>(workspace));
    }));

    return {dx, dW_k, dW_q, dW_x, dW_r, db, dW_alpha, db_alpha};
}

// =============================================================================
// E70: Matrix Linear Elman - Square Matrix State with Linear Update
// S = tanh(decay * S + outer(v, k))
// out = (S @ q) * silu(S @ q)
// =============================================================================

std::vector<Tensor> e70_matrix_linear_forward(
    bool training,
    Tensor x,           // [T, B, dim] input
    Tensor S0,          // [B, n_state, n_state] initial state matrix
    float decay,        // Decay factor for state retention
    Tensor W_k,         // [n_state, dim] key projection
    Tensor W_v,         // [n_state, dim] value projection
    Tensor W_q) {       // [n_state, dim] query projection

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    const auto n_state = W_k.size(0);

    CHECK_INPUT(x);
    CHECK_INPUT(S0);
    CHECK_INPUT(W_k);
    CHECK_INPUT(W_v);
    CHECK_INPUT(W_q);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    // Outputs
    Tensor S = torch::empty({time_steps + 1, batch_size, n_state, n_state}, options);
    Tensor output = torch::empty({time_steps, batch_size, n_state}, options);

    // Caches for backward
    Tensor k_cache = training ? torch::empty({time_steps, batch_size, n_state}, options)
                              : torch::empty({0}, options);
    Tensor v_cache = training ? torch::empty({time_steps, batch_size, n_state}, options)
                              : torch::empty({0}, options);
    Tensor q_cache = training ? torch::empty({time_steps, batch_size, n_state}, options)
                              : torch::empty({0}, options);
    Tensor Sq_cache = training ? torch::empty({time_steps, batch_size, n_state}, options)
                               : torch::empty({0}, options);

    // Workspace
    using namespace hasty::v0::elman_ladder;
    const int64_t workspace_size = E70MatrixLinearForward<__nv_bfloat16>::WorkspaceSize(
        time_steps, batch_size, n_state);
    Tensor workspace = torch::empty({workspace_size}, options);

    S[0] = S0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e70_matrix_linear_forward", ([&] {
        E70MatrixLinearForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, n_state,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            decay,
            ptr<scalar_t>(W_k),
            ptr<scalar_t>(W_v),
            ptr<scalar_t>(W_q),
            ptr<scalar_t>(x),
            ptr<scalar_t>(S),
            ptr<scalar_t>(output),
            training ? ptr<scalar_t>(k_cache) : nullptr,
            training ? ptr<scalar_t>(v_cache) : nullptr,
            training ? ptr<scalar_t>(q_cache) : nullptr,
            training ? ptr<scalar_t>(Sq_cache) : nullptr,
            ptr<scalar_t>(workspace));
    }));

    return {S, output, k_cache, v_cache, q_cache, Sq_cache};
}

std::vector<Tensor> e70_matrix_linear_backward(
    float decay,
    Tensor W_k,
    Tensor W_v,
    Tensor W_q,
    Tensor x,
    Tensor S,
    Tensor k_cache,
    Tensor v_cache,
    Tensor q_cache,
    Tensor Sq_cache,
    Tensor d_output) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    const auto n_state = W_k.size(0);

    CHECK_INPUT(W_k);
    CHECK_INPUT(W_v);
    CHECK_INPUT(W_q);
    CHECK_INPUT(x);
    CHECK_INPUT(S);
    CHECK_INPUT(k_cache);
    CHECK_INPUT(v_cache);
    CHECK_INPUT(q_cache);
    CHECK_INPUT(Sq_cache);
    CHECK_INPUT(d_output);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::zeros({time_steps, batch_size, dim}, options);
    Tensor dW_k = torch::zeros({n_state, dim}, options);
    Tensor dW_v = torch::zeros({n_state, dim}, options);
    Tensor dW_q = torch::zeros({n_state, dim}, options);
    Tensor d_decay_out = torch::zeros({1}, options);

    // Workspace
    using namespace hasty::v0::elman_ladder;
    const int64_t workspace_size = E70MatrixLinearBackward<__nv_bfloat16>::WorkspaceSize(
        time_steps, batch_size, n_state);
    Tensor workspace = torch::empty({workspace_size}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e70_matrix_linear_backward", ([&] {
        E70MatrixLinearBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, n_state,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            decay,
            ptr<scalar_t>(W_k),
            ptr<scalar_t>(W_v),
            ptr<scalar_t>(W_q),
            ptr<scalar_t>(x),
            ptr<scalar_t>(S),
            ptr<scalar_t>(k_cache),
            ptr<scalar_t>(v_cache),
            ptr<scalar_t>(q_cache),
            ptr<scalar_t>(Sq_cache),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dW_k),
            ptr<scalar_t>(dW_v),
            ptr<scalar_t>(dW_q),
            ptr<scalar_t>(d_decay_out),
            ptr<scalar_t>(workspace));
    }));

    return {dx, dW_k, dW_v, dW_q, d_decay_out};
}

// =============================================================================
// E71: Matrix Gated Elman - E67-style state-dependent gating with matrix state
// retrieved = S @ k, alpha = sigmoid(alpha_x + d_alpha * retrieved + b_alpha)
// S_new = alpha * S + (1 - alpha) * outer(v, k)
// out = (S @ q) * silu(S @ q)
// =============================================================================

std::vector<Tensor> e71_matrix_gated_forward(
    bool training,
    Tensor x,           // [T, B, dim] input
    Tensor S0,          // [B, n_state, n_state] initial state matrix
    Tensor W_k,         // [n_state, dim] key projection
    Tensor W_v,         // [n_state, dim] value projection
    Tensor W_q,         // [n_state, dim] query projection
    Tensor W_alpha,     // [n_state, dim] gate projection
    Tensor d_alpha,     // [n_state] gate h-scaling
    Tensor b_alpha) {   // [n_state] gate bias

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    const auto n_state = W_k.size(0);

    CHECK_INPUT(x);
    CHECK_INPUT(S0);
    CHECK_INPUT(W_k);
    CHECK_INPUT(W_v);
    CHECK_INPUT(W_q);
    CHECK_INPUT(W_alpha);
    CHECK_INPUT(d_alpha);
    CHECK_INPUT(b_alpha);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    // S: [T+1, B, n_state, n_state]
    Tensor S = torch::empty({time_steps + 1, batch_size, n_state, n_state}, options);
    Tensor output = torch::empty({time_steps, batch_size, n_state}, options);

    // Caches for backward
    Tensor k_cache = training ? torch::empty({time_steps, batch_size, n_state}, options)
                              : torch::empty({0}, options);
    Tensor v_cache = training ? torch::empty({time_steps, batch_size, n_state}, options)
                              : torch::empty({0}, options);
    Tensor q_cache = training ? torch::empty({time_steps, batch_size, n_state}, options)
                              : torch::empty({0}, options);
    Tensor alpha_x_cache = training ? torch::empty({time_steps, batch_size, n_state}, options)
                                    : torch::empty({0}, options);
    Tensor retrieved_cache = training ? torch::empty({time_steps, batch_size, n_state}, options)
                                      : torch::empty({0}, options);
    Tensor alpha_cache = training ? torch::empty({time_steps, batch_size, n_state}, options)
                                  : torch::empty({0}, options);

    // Workspace
    using namespace hasty::v0::elman_ladder;
    const int64_t workspace_size = E71MatrixGatedForward<__nv_bfloat16>::WorkspaceSize(
        time_steps, batch_size, n_state);
    Tensor workspace = torch::empty({workspace_size}, options);

    S[0] = S0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e71_matrix_gated_forward", ([&] {
        E71MatrixGatedForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, n_state, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W_k),
            ptr<scalar_t>(W_v),
            ptr<scalar_t>(W_q),
            ptr<scalar_t>(W_alpha),
            ptr<scalar_t>(d_alpha),
            ptr<scalar_t>(b_alpha),
            ptr<scalar_t>(x),
            ptr<scalar_t>(S),
            ptr<scalar_t>(output),
            training ? ptr<scalar_t>(k_cache) : nullptr,
            training ? ptr<scalar_t>(v_cache) : nullptr,
            training ? ptr<scalar_t>(q_cache) : nullptr,
            training ? ptr<scalar_t>(alpha_x_cache) : nullptr,
            training ? ptr<scalar_t>(retrieved_cache) : nullptr,
            training ? ptr<scalar_t>(alpha_cache) : nullptr,
            ptr<scalar_t>(workspace));
    }));

    return {S, output, k_cache, v_cache, q_cache, alpha_x_cache, retrieved_cache, alpha_cache};
}

std::vector<Tensor> e71_matrix_gated_backward(
    Tensor W_k,
    Tensor W_v,
    Tensor W_q,
    Tensor W_alpha,
    Tensor d_alpha,
    Tensor b_alpha,
    Tensor x,
    Tensor S,
    Tensor k_cache,
    Tensor v_cache,
    Tensor q_cache,
    Tensor alpha_x_cache,
    Tensor retrieved_cache,
    Tensor alpha_cache,
    Tensor d_output) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    const auto n_state = W_k.size(0);

    CHECK_INPUT(W_k);
    CHECK_INPUT(W_v);
    CHECK_INPUT(W_q);
    CHECK_INPUT(W_alpha);
    CHECK_INPUT(d_alpha);
    CHECK_INPUT(b_alpha);
    CHECK_INPUT(x);
    CHECK_INPUT(S);
    CHECK_INPUT(k_cache);
    CHECK_INPUT(v_cache);
    CHECK_INPUT(q_cache);
    CHECK_INPUT(alpha_x_cache);
    CHECK_INPUT(retrieved_cache);
    CHECK_INPUT(alpha_cache);
    CHECK_INPUT(d_output);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    // Output gradients
    Tensor dx = torch::empty_like(x);
    Tensor dW_k = torch::zeros({n_state, dim}, options);
    Tensor dW_v = torch::zeros({n_state, dim}, options);
    Tensor dW_q = torch::zeros({n_state, dim}, options);
    Tensor dW_alpha = torch::zeros({n_state, dim}, options);
    Tensor dd_alpha = torch::zeros({n_state}, options);
    Tensor db_alpha = torch::zeros({n_state}, options);

    // Workspace
    using namespace hasty::v0::elman_ladder;
    const int64_t workspace_size = E71MatrixGatedBackward<__nv_bfloat16>::WorkspaceSize(
        time_steps, batch_size, n_state);
    Tensor workspace = torch::empty({workspace_size}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e71_matrix_gated_backward", ([&] {
        E71MatrixGatedBackward<typename native_type<scalar_t>::T> backward(
            batch_size, n_state, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W_k),
            ptr<scalar_t>(W_v),
            ptr<scalar_t>(W_q),
            ptr<scalar_t>(W_alpha),
            ptr<scalar_t>(d_alpha),
            ptr<scalar_t>(b_alpha),
            ptr<scalar_t>(x),
            ptr<scalar_t>(S),
            ptr<scalar_t>(k_cache),
            ptr<scalar_t>(v_cache),
            ptr<scalar_t>(q_cache),
            ptr<scalar_t>(alpha_x_cache),
            ptr<scalar_t>(retrieved_cache),
            ptr<scalar_t>(alpha_cache),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dW_k),
            ptr<scalar_t>(dW_v),
            ptr<scalar_t>(dW_q),
            ptr<scalar_t>(dW_alpha),
            ptr<scalar_t>(dd_alpha),
            ptr<scalar_t>(db_alpha),
            ptr<scalar_t>(workspace));
    }));

    return {dx, dW_k, dW_v, dW_q, dW_alpha, dd_alpha, db_alpha};
}

// =============================================================================
// E71 Delta: Matrix Gated with Delta Rule - State-dependent learning rate
// k_norm = k / (||k|| + eps)
// retrieved = S @ k_norm, beta = sigmoid(beta_x + d_beta * retrieved + b_beta)
// S_new = S + beta * outer(v - retrieved, k_norm)
// out = (S @ q) * silu(S @ q)
// =============================================================================

std::vector<Tensor> e71_delta_forward(
    bool training,
    Tensor x,           // [T, B, dim] input
    Tensor S0,          // [B, n_state, n_state] initial state matrix
    Tensor W_k,         // [n_state, dim] key projection
    Tensor W_v,         // [n_state, dim] value projection
    Tensor W_q,         // [n_state, dim] query projection
    Tensor W_beta,      // [n_state, dim] beta gate projection
    Tensor d_beta,      // [n_state] beta h-scaling
    Tensor b_beta) {    // [n_state] beta bias

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    const auto n_state = W_k.size(0);

    CHECK_INPUT(x);
    CHECK_INPUT(S0);
    CHECK_INPUT(W_k);
    CHECK_INPUT(W_v);
    CHECK_INPUT(W_q);
    CHECK_INPUT(W_beta);
    CHECK_INPUT(d_beta);
    CHECK_INPUT(b_beta);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    // S: [T+1, B, n_state, n_state]
    Tensor S = torch::empty({time_steps + 1, batch_size, n_state, n_state}, options);
    Tensor output = torch::empty({time_steps, batch_size, n_state}, options);

    // Caches for backward
    Tensor k_cache = training ? torch::empty({time_steps, batch_size, n_state}, options)
                              : torch::empty({0}, options);
    Tensor v_cache = training ? torch::empty({time_steps, batch_size, n_state}, options)
                              : torch::empty({0}, options);
    Tensor q_cache = training ? torch::empty({time_steps, batch_size, n_state}, options)
                              : torch::empty({0}, options);
    Tensor beta_x_cache = training ? torch::empty({time_steps, batch_size, n_state}, options)
                                   : torch::empty({0}, options);
    Tensor retrieved_cache = training ? torch::empty({time_steps, batch_size, n_state}, options)
                                      : torch::empty({0}, options);
    Tensor beta_cache = training ? torch::empty({time_steps, batch_size, n_state}, options)
                                 : torch::empty({0}, options);
    Tensor k_norm_cache = training ? torch::empty({time_steps, batch_size, n_state}, options)
                                   : torch::empty({0}, options);

    // Workspace
    using namespace hasty::v0::elman_ladder;
    const int64_t workspace_size = E71DeltaForward<__nv_bfloat16>::WorkspaceSize(
        time_steps, batch_size, n_state);
    Tensor workspace = torch::empty({workspace_size}, options);

    S[0] = S0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e71_delta_forward", ([&] {
        E71DeltaForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, n_state, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W_k),
            ptr<scalar_t>(W_v),
            ptr<scalar_t>(W_q),
            ptr<scalar_t>(W_beta),
            ptr<scalar_t>(d_beta),
            ptr<scalar_t>(b_beta),
            ptr<scalar_t>(x),
            ptr<scalar_t>(S),
            ptr<scalar_t>(output),
            training ? ptr<scalar_t>(k_cache) : nullptr,
            training ? ptr<scalar_t>(v_cache) : nullptr,
            training ? ptr<scalar_t>(q_cache) : nullptr,
            training ? ptr<scalar_t>(beta_x_cache) : nullptr,
            training ? ptr<scalar_t>(retrieved_cache) : nullptr,
            training ? ptr<scalar_t>(beta_cache) : nullptr,
            training ? ptr<scalar_t>(k_norm_cache) : nullptr,
            ptr<scalar_t>(workspace));
    }));

    return {S, output, k_cache, v_cache, q_cache, beta_x_cache, retrieved_cache, beta_cache, k_norm_cache};
}

std::vector<Tensor> e71_delta_backward(
    Tensor W_k,
    Tensor W_v,
    Tensor W_q,
    Tensor W_beta,
    Tensor d_beta,
    Tensor b_beta,
    Tensor x,
    Tensor S,
    Tensor k_cache,
    Tensor v_cache,
    Tensor q_cache,
    Tensor beta_x_cache,
    Tensor retrieved_cache,
    Tensor beta_cache,
    Tensor k_norm_cache,
    Tensor d_output) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    const auto n_state = W_k.size(0);

    CHECK_INPUT(W_k);
    CHECK_INPUT(W_v);
    CHECK_INPUT(W_q);
    CHECK_INPUT(W_beta);
    CHECK_INPUT(d_beta);
    CHECK_INPUT(b_beta);
    CHECK_INPUT(x);
    CHECK_INPUT(S);
    CHECK_INPUT(k_cache);
    CHECK_INPUT(v_cache);
    CHECK_INPUT(q_cache);
    CHECK_INPUT(beta_x_cache);
    CHECK_INPUT(retrieved_cache);
    CHECK_INPUT(beta_cache);
    CHECK_INPUT(k_norm_cache);
    CHECK_INPUT(d_output);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    // Output gradients
    Tensor dx = torch::empty_like(x);
    Tensor dW_k = torch::zeros({n_state, dim}, options);
    Tensor dW_v = torch::zeros({n_state, dim}, options);
    Tensor dW_q = torch::zeros({n_state, dim}, options);
    Tensor dW_beta = torch::zeros({n_state, dim}, options);
    Tensor dd_beta = torch::zeros({n_state}, options);
    Tensor db_beta = torch::zeros({n_state}, options);

    // Workspace
    using namespace hasty::v0::elman_ladder;
    const int64_t workspace_size = E71DeltaBackward<__nv_bfloat16>::WorkspaceSize(
        time_steps, batch_size, n_state);
    Tensor workspace = torch::empty({workspace_size}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e71_delta_backward", ([&] {
        E71DeltaBackward<typename native_type<scalar_t>::T> backward(
            batch_size, n_state, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W_k),
            ptr<scalar_t>(W_v),
            ptr<scalar_t>(W_q),
            ptr<scalar_t>(W_beta),
            ptr<scalar_t>(d_beta),
            ptr<scalar_t>(b_beta),
            ptr<scalar_t>(x),
            ptr<scalar_t>(S),
            ptr<scalar_t>(k_cache),
            ptr<scalar_t>(v_cache),
            ptr<scalar_t>(q_cache),
            ptr<scalar_t>(beta_x_cache),
            ptr<scalar_t>(retrieved_cache),
            ptr<scalar_t>(beta_cache),
            ptr<scalar_t>(k_norm_cache),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dW_k),
            ptr<scalar_t>(dW_v),
            ptr<scalar_t>(dW_q),
            ptr<scalar_t>(dW_beta),
            ptr<scalar_t>(dd_beta),
            ptr<scalar_t>(db_beta),
            ptr<scalar_t>(workspace));
    }));

    return {dx, dW_k, dW_v, dW_q, dW_beta, dd_beta, db_beta};
}

// =============================================================================
// E72: Matrix SelfGate Elman - Memory content controls writing
// =============================================================================

std::vector<Tensor> e72_matrix_selfgate_forward(
    bool training,
    Tensor x,           // [T, B, dim] input
    Tensor S0,          // [B, n_state, n_state] initial state matrix
    bool inverse_gate,  // false=standard, true=inverse
    Tensor W_k,         // [n_state, dim] key projection
    Tensor W_v,         // [n_state, dim] value projection
    Tensor W_q,         // [n_state, dim] query projection
    Tensor W_alpha,     // [n_state, dim] retain gate weight
    Tensor b_alpha,     // [n_state] retain gate bias
    Tensor d_g,         // [n_state] gating weight
    Tensor b_g) {       // [n_state] gating bias

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    const auto n_state = W_k.size(0);

    CHECK_INPUT(x);
    CHECK_INPUT(S0);
    CHECK_INPUT(W_k);
    CHECK_INPUT(W_v);
    CHECK_INPUT(W_q);
    CHECK_INPUT(W_alpha);
    CHECK_INPUT(b_alpha);
    CHECK_INPUT(d_g);
    CHECK_INPUT(b_g);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    // Outputs
    Tensor S = torch::empty({time_steps + 1, batch_size, n_state, n_state}, options);
    Tensor output = torch::empty({time_steps, batch_size, n_state}, options);

    // Caches for backward
    Tensor k_cache = training ? torch::empty({time_steps, batch_size, n_state}, options)
                              : torch::empty({0}, options);
    Tensor v_cache = training ? torch::empty({time_steps, batch_size, n_state}, options)
                              : torch::empty({0}, options);
    Tensor q_cache = training ? torch::empty({time_steps, batch_size, n_state}, options)
                              : torch::empty({0}, options);
    Tensor alpha_cache = training ? torch::empty({time_steps, batch_size, n_state}, options)
                                  : torch::empty({0}, options);
    Tensor retrieved_cache = training ? torch::empty({time_steps, batch_size, n_state}, options)
                                      : torch::empty({0}, options);
    Tensor g_cache = training ? torch::empty({time_steps, batch_size, n_state}, options)
                              : torch::empty({0}, options);

    // Workspace
    using namespace hasty::v0::elman_ladder;
    const int64_t workspace_size = E72MatrixSelfGateForward<__nv_bfloat16>::WorkspaceSize(
        time_steps, batch_size, n_state);
    Tensor workspace = torch::empty({workspace_size}, options);

    S[0] = S0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e72_matrix_selfgate_forward", ([&] {
        E72MatrixSelfGateForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, n_state, inverse_gate,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W_k),
            ptr<scalar_t>(W_v),
            ptr<scalar_t>(W_q),
            ptr<scalar_t>(W_alpha),
            ptr<scalar_t>(b_alpha),
            ptr<scalar_t>(d_g),
            ptr<scalar_t>(b_g),
            ptr<scalar_t>(x),
            ptr<scalar_t>(S),
            ptr<scalar_t>(output),
            training ? ptr<scalar_t>(k_cache) : nullptr,
            training ? ptr<scalar_t>(v_cache) : nullptr,
            training ? ptr<scalar_t>(q_cache) : nullptr,
            training ? ptr<scalar_t>(alpha_cache) : nullptr,
            training ? ptr<scalar_t>(retrieved_cache) : nullptr,
            training ? ptr<scalar_t>(g_cache) : nullptr,
            ptr<scalar_t>(workspace));
    }));

    return {S, output, k_cache, v_cache, q_cache, alpha_cache, retrieved_cache, g_cache};
}

std::vector<Tensor> e72_matrix_selfgate_backward(
    bool inverse_gate,
    Tensor W_k,
    Tensor W_v,
    Tensor W_q,
    Tensor W_alpha,
    Tensor d_g,
    Tensor x,
    Tensor S,
    Tensor k_cache,
    Tensor v_cache,
    Tensor q_cache,
    Tensor alpha_cache,
    Tensor retrieved_cache,
    Tensor g_cache,
    Tensor d_output) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    const auto n_state = W_k.size(0);

    CHECK_INPUT(W_k);
    CHECK_INPUT(W_v);
    CHECK_INPUT(W_q);
    CHECK_INPUT(W_alpha);
    CHECK_INPUT(d_g);
    CHECK_INPUT(x);
    CHECK_INPUT(S);
    CHECK_INPUT(k_cache);
    CHECK_INPUT(v_cache);
    CHECK_INPUT(q_cache);
    CHECK_INPUT(alpha_cache);
    CHECK_INPUT(retrieved_cache);
    CHECK_INPUT(g_cache);
    CHECK_INPUT(d_output);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::zeros({time_steps, batch_size, dim}, options);
    Tensor dW_k = torch::zeros({n_state, dim}, options);
    Tensor dW_v = torch::zeros({n_state, dim}, options);
    Tensor dW_q = torch::zeros({n_state, dim}, options);
    Tensor dW_alpha = torch::zeros({n_state, dim}, options);
    Tensor db_alpha = torch::zeros({n_state}, options);
    Tensor dd_g = torch::zeros({n_state}, options);
    Tensor db_g = torch::zeros({n_state}, options);

    // Workspace
    using namespace hasty::v0::elman_ladder;
    const int64_t workspace_size = E72MatrixSelfGateBackward<__nv_bfloat16>::WorkspaceSize(
        time_steps, batch_size, n_state);
    Tensor workspace = torch::empty({workspace_size}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e72_matrix_selfgate_backward", ([&] {
        E72MatrixSelfGateBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, n_state, inverse_gate,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W_k),
            ptr<scalar_t>(W_v),
            ptr<scalar_t>(W_q),
            ptr<scalar_t>(W_alpha),
            ptr<scalar_t>(d_g),
            ptr<scalar_t>(x),
            ptr<scalar_t>(S),
            ptr<scalar_t>(k_cache),
            ptr<scalar_t>(v_cache),
            ptr<scalar_t>(q_cache),
            ptr<scalar_t>(alpha_cache),
            ptr<scalar_t>(retrieved_cache),
            ptr<scalar_t>(g_cache),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dW_k),
            ptr<scalar_t>(dW_v),
            ptr<scalar_t>(dW_q),
            ptr<scalar_t>(dW_alpha),
            ptr<scalar_t>(db_alpha),
            ptr<scalar_t>(dd_g),
            ptr<scalar_t>(db_g),
            ptr<scalar_t>(workspace));
    }));

    return {dx, dW_k, dW_v, dW_q, dW_alpha, db_alpha, dd_g, db_g};
}

// =============================================================================
// E73: Matrix Nonlinear Elman - E1-style with S inside tanh
// S = tanh(S * z_mod + outer(v, k))
// out = (S @ q) * silu(S @ q)
// Variants: 0=column (z[j]), 1=row (z[i]), 2=full (z[i]*z[j])
// =============================================================================

std::vector<Tensor> e73_matrix_nonlinear_forward(
    bool training,
    Tensor x,           // [T, B, dim] input
    Tensor S0,          // [B, n_state, n_state] initial state matrix
    int variant,        // 0=column, 1=row, 2=full
    Tensor W_k,         // [n_state, dim] key projection
    Tensor W_v,         // [n_state, dim] value projection
    Tensor W_q,         // [n_state, dim] query projection
    Tensor W_z,         // [n_state, dim] modulation gate projection
    Tensor b_z) {       // [n_state] modulation gate bias

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    const auto n_state = W_k.size(0);

    CHECK_INPUT(x);
    CHECK_INPUT(S0);
    CHECK_INPUT(W_k);
    CHECK_INPUT(W_v);
    CHECK_INPUT(W_q);
    CHECK_INPUT(W_z);
    CHECK_INPUT(b_z);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    // Outputs
    Tensor S = torch::empty({time_steps + 1, batch_size, n_state, n_state}, options);
    Tensor output = torch::empty({time_steps, batch_size, n_state}, options);

    // Caches for backward
    Tensor k_cache = training ? torch::empty({time_steps, batch_size, n_state}, options)
                              : torch::empty({0}, options);
    Tensor v_cache = training ? torch::empty({time_steps, batch_size, n_state}, options)
                              : torch::empty({0}, options);
    Tensor q_cache = training ? torch::empty({time_steps, batch_size, n_state}, options)
                              : torch::empty({0}, options);
    Tensor z_cache = training ? torch::empty({time_steps, batch_size, n_state}, options)
                              : torch::empty({0}, options);
    Tensor pre_tanh_cache = training ? torch::empty({time_steps, batch_size, n_state, n_state}, options)
                                     : torch::empty({0}, options);
    Tensor Sq_cache = training ? torch::empty({time_steps, batch_size, n_state}, options)
                               : torch::empty({0}, options);

    // Workspace
    using namespace hasty::v0::elman_ladder;
    const int64_t workspace_size = E73MatrixNonlinearForward<__nv_bfloat16>::WorkspaceSize(
        time_steps, batch_size, n_state);
    Tensor workspace = torch::empty({workspace_size}, options);

    S[0] = S0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e73_matrix_nonlinear_forward", ([&] {
        E73MatrixNonlinearForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, n_state, dim, variant,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W_k),
            ptr<scalar_t>(W_v),
            ptr<scalar_t>(W_q),
            ptr<scalar_t>(W_z),
            ptr<scalar_t>(b_z),
            ptr<scalar_t>(x),
            ptr<scalar_t>(S),
            ptr<scalar_t>(output),
            training ? ptr<scalar_t>(k_cache) : nullptr,
            training ? ptr<scalar_t>(v_cache) : nullptr,
            training ? ptr<scalar_t>(q_cache) : nullptr,
            training ? ptr<scalar_t>(z_cache) : nullptr,
            training ? ptr<scalar_t>(pre_tanh_cache) : nullptr,
            training ? ptr<scalar_t>(Sq_cache) : nullptr,
            ptr<scalar_t>(workspace));
    }));

    return {S, output, k_cache, v_cache, q_cache, z_cache, pre_tanh_cache, Sq_cache};
}

std::vector<Tensor> e73_matrix_nonlinear_backward(
    int variant,
    Tensor W_k,
    Tensor W_v,
    Tensor W_q,
    Tensor W_z,
    Tensor x,
    Tensor S,
    Tensor k_cache,
    Tensor v_cache,
    Tensor q_cache,
    Tensor z_cache,
    Tensor pre_tanh_cache,
    Tensor Sq_cache,
    Tensor d_output) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    const auto n_state = W_k.size(0);

    CHECK_INPUT(W_k);
    CHECK_INPUT(W_v);
    CHECK_INPUT(W_q);
    CHECK_INPUT(W_z);
    CHECK_INPUT(x);
    CHECK_INPUT(S);
    CHECK_INPUT(k_cache);
    CHECK_INPUT(v_cache);
    CHECK_INPUT(q_cache);
    CHECK_INPUT(z_cache);
    CHECK_INPUT(pre_tanh_cache);
    CHECK_INPUT(Sq_cache);
    CHECK_INPUT(d_output);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::zeros({time_steps, batch_size, dim}, options);
    Tensor dW_k = torch::zeros({n_state, dim}, options);
    Tensor dW_v = torch::zeros({n_state, dim}, options);
    Tensor dW_q = torch::zeros({n_state, dim}, options);
    Tensor dW_z = torch::zeros({n_state, dim}, options);
    Tensor db_z = torch::zeros({n_state}, options);

    // Workspace
    using namespace hasty::v0::elman_ladder;
    const int64_t workspace_size = E73MatrixNonlinearBackward<__nv_bfloat16>::WorkspaceSize(
        time_steps, batch_size, n_state);
    Tensor workspace = torch::empty({workspace_size}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "e73_matrix_nonlinear_backward", ([&] {
        E73MatrixNonlinearBackward<typename native_type<scalar_t>::T> backward(
            batch_size, n_state, dim, variant,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W_k),
            ptr<scalar_t>(W_v),
            ptr<scalar_t>(W_q),
            ptr<scalar_t>(W_z),
            ptr<scalar_t>(x),
            ptr<scalar_t>(S),
            ptr<scalar_t>(k_cache),
            ptr<scalar_t>(v_cache),
            ptr<scalar_t>(q_cache),
            ptr<scalar_t>(z_cache),
            ptr<scalar_t>(pre_tanh_cache),
            ptr<scalar_t>(Sq_cache),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dW_k),
            ptr<scalar_t>(dW_v),
            ptr<scalar_t>(dW_q),
            ptr<scalar_t>(dW_z),
            ptr<scalar_t>(db_z),
            ptr<scalar_t>(workspace));
    }));

    return {dx, dW_k, dW_v, dW_q, dW_z, db_z};
}

// =============================================================================
// E73 Fused: Optimized Matrix Nonlinear with persistent kernel
// Single kernel launch for all timesteps, gradient checkpointing
// =============================================================================

std::vector<Tensor> e73_fused_forward(
    bool training,
    Tensor x,           // [T, B, dim] input
    Tensor S0,          // [B, n_state, n_state] initial state matrix
    int variant,        // 0=column, 1=row, 2=full
    Tensor W_k,         // [n_state, dim]
    Tensor W_v,         // [n_state, dim]
    Tensor W_q,         // [n_state, dim]
    Tensor W_z,         // [n_state, dim]
    Tensor b_z) {       // [n_state]

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    const auto n_state = W_k.size(0);

    CHECK_INPUT(x);
    CHECK_INPUT(S0);
    CHECK_INPUT(W_k);
    CHECK_INPUT(W_v);
    CHECK_INPUT(W_q);
    CHECK_INPUT(W_z);
    CHECK_INPUT(b_z);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    // State - in-place update (only keep current state, not all history)
    Tensor S = S0.clone();  // [B, n_state, n_state]
    Tensor output = torch::empty({time_steps, batch_size, n_state}, options);

    // Workspace
    const int64_t workspace_size = elman::E73FusedForward<__nv_bfloat16>::WorkspaceSize(
        time_steps, batch_size, n_state);
    Tensor workspace = torch::empty({workspace_size}, options);

    // Reshape x from [T, B, dim] to [T*B, dim] contiguous
    Tensor x_flat = x.reshape({time_steps * batch_size, dim}).contiguous();

    // Only BFloat16 is implemented for E73 Fused
    TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16,
                "E73 Fused only supports bfloat16, got ", x.scalar_type());

    elman::E73FusedForward<__nv_bfloat16> forward(
            batch_size, n_state, dim, variant,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

    forward.Run(
        time_steps,
        reinterpret_cast<const __nv_bfloat16*>(W_k.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_v.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_z.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(b_z.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(x_flat.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(S.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(workspace.data_ptr()),
        training);

    return {S, output};  // Just return final state and outputs
}

// =============================================================================
// E73 Checkpointed: In-place S with gradient checkpointing
// 50x+ memory reduction for S storage
// =============================================================================

std::vector<Tensor> e73_checkpointed_forward(
    bool training,
    Tensor x,           // [T, B, dim] input
    Tensor S0,          // [B, n_state, n_state] initial state matrix
    int variant,        // 0=column, 1=row, 2=full
    int checkpoint_interval,  // Checkpoint every K steps (default 32)
    Tensor W_k,         // [n_state, dim]
    Tensor W_v,         // [n_state, dim]
    Tensor W_q,         // [n_state, dim]
    Tensor W_z,         // [n_state, dim]
    Tensor b_z) {       // [n_state]

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    const auto n_state = W_k.size(0);

    CHECK_INPUT(x);
    CHECK_INPUT(S0);
    CHECK_INPUT(W_k);
    CHECK_INPUT(W_v);
    CHECK_INPUT(W_q);
    CHECK_INPUT(W_z);
    CHECK_INPUT(b_z);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    // S is updated in-place - just need [B, n, n]
    Tensor S = S0.clone();
    Tensor output = torch::empty({time_steps, batch_size, n_state}, options);

    // Checkpoints: num_checkpoints * [B, n, n]
    int K = checkpoint_interval > 0 ? checkpoint_interval : 32;
    int num_checkpoints = elman::E73CheckpointedForward<__nv_bfloat16>::NumCheckpoints(time_steps, K);
    Tensor S_checkpoints = training ? torch::empty({num_checkpoints, batch_size, n_state, n_state}, options)
                                    : torch::empty({0}, options);

    // Caches for backward (smaller than original - no pre_tanh_cache!)
    Tensor k_norm_cache = training ? torch::empty({time_steps, batch_size, n_state}, options)
                                   : torch::empty({0}, options);
    Tensor v_cache = training ? torch::empty({time_steps, batch_size, n_state}, options)
                              : torch::empty({0}, options);
    Tensor q_cache = training ? torch::empty({time_steps, batch_size, n_state}, options)
                              : torch::empty({0}, options);
    Tensor z_cache = training ? torch::empty({time_steps, batch_size, n_state}, options)
                              : torch::empty({0}, options);
    Tensor Sq_cache = training ? torch::empty({time_steps, batch_size, n_state}, options)
                               : torch::empty({0}, options);

    // Workspace
    const int64_t workspace_size = elman::E73CheckpointedForward<__nv_bfloat16>::WorkspaceSize(
        time_steps, batch_size, n_state, K, training);
    Tensor workspace = torch::empty({workspace_size}, options);

    // Reshape x from [T, B, dim] to [T*B, dim] contiguous
    Tensor x_flat = x.reshape({time_steps * batch_size, dim}).contiguous();

    // Only BFloat16 is implemented
    TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16,
                "E73 Checkpointed only supports bfloat16, got ", x.scalar_type());

    elman::E73CheckpointedForward<__nv_bfloat16> forward(
        batch_size, n_state, dim, variant, K,
        at::cuda::getCurrentCUDABlasHandle(),
        at::cuda::getCurrentCUDAStream());

    forward.Run(
        time_steps,
        reinterpret_cast<const __nv_bfloat16*>(W_k.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_v.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_z.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(b_z.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(x_flat.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(S.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        training ? reinterpret_cast<__nv_bfloat16*>(S_checkpoints.data_ptr()) : nullptr,
        training ? reinterpret_cast<__nv_bfloat16*>(k_norm_cache.data_ptr()) : nullptr,
        training ? reinterpret_cast<__nv_bfloat16*>(v_cache.data_ptr()) : nullptr,
        training ? reinterpret_cast<__nv_bfloat16*>(q_cache.data_ptr()) : nullptr,
        training ? reinterpret_cast<__nv_bfloat16*>(z_cache.data_ptr()) : nullptr,
        training ? reinterpret_cast<__nv_bfloat16*>(Sq_cache.data_ptr()) : nullptr,
        reinterpret_cast<__nv_bfloat16*>(workspace.data_ptr()),
        training);

    // Return: S (final), output, S_checkpoints, k_norm_cache, v_cache, q_cache, z_cache, Sq_cache
    return {S, output, S_checkpoints, k_norm_cache, v_cache, q_cache, z_cache, Sq_cache};
}

// =============================================================================
// E73 Checkpointed Backward
// =============================================================================

std::vector<Tensor> e73_checkpointed_backward(
    Tensor x,               // [T, B, dim] input
    Tensor d_output,        // [T, B, n_state] gradient from output
    Tensor S_checkpoints,   // [num_checkpoints, B, n_state, n_state]
    Tensor k_norm_cache,    // [T, B, n_state]
    Tensor v_cache,         // [T, B, n_state]
    Tensor q_cache,         // [T, B, n_state]
    Tensor z_cache,         // [T, B, n_state]
    Tensor Sq_cache,        // [T, B, n_state]
    int variant,            // 0=column, 1=row, 2=full
    int checkpoint_interval,
    Tensor W_k,             // [n_state, dim]
    Tensor W_v,             // [n_state, dim]
    Tensor W_q,             // [n_state, dim]
    Tensor W_z) {           // [n_state, dim]

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    const auto n_state = W_k.size(0);

    CHECK_INPUT(x);
    CHECK_INPUT(d_output);
    CHECK_INPUT(S_checkpoints);
    CHECK_INPUT(k_norm_cache);
    CHECK_INPUT(v_cache);
    CHECK_INPUT(q_cache);
    CHECK_INPUT(z_cache);
    CHECK_INPUT(Sq_cache);
    CHECK_INPUT(W_k);
    CHECK_INPUT(W_v);
    CHECK_INPUT(W_q);
    CHECK_INPUT(W_z);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    // Output gradients
    Tensor dx = torch::empty_like(x);
    Tensor dW_k = torch::zeros({n_state, dim}, options);
    Tensor dW_v = torch::zeros({n_state, dim}, options);
    Tensor dW_q = torch::zeros({n_state, dim}, options);
    Tensor dW_z = torch::zeros({n_state, dim}, options);
    Tensor db_z = torch::zeros({n_state}, options);

    // Workspace
    int K = checkpoint_interval > 0 ? checkpoint_interval : 32;
    const int64_t workspace_size = elman::E73CheckpointedBackward<__nv_bfloat16>::WorkspaceSize(
        time_steps, batch_size, n_state, K, dim);
    Tensor workspace = torch::empty({workspace_size}, options);

    // Reshape inputs for CUDA kernel
    Tensor x_flat = x.reshape({time_steps * batch_size, dim}).contiguous();
    Tensor d_output_flat = d_output.reshape({time_steps * batch_size, n_state}).contiguous();

    // Only BFloat16 is implemented
    TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16,
                "E73 Checkpointed Backward only supports bfloat16, got ", x.scalar_type());

    elman::E73CheckpointedBackward<__nv_bfloat16> backward(
        batch_size, n_state, dim, variant, K,
        at::cuda::getCurrentCUDABlasHandle(),
        at::cuda::getCurrentCUDAStream());

    backward.Run(
        time_steps,
        reinterpret_cast<const __nv_bfloat16*>(W_k.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_v.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_z.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(x_flat.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(S_checkpoints.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k_norm_cache.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v_cache.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(q_cache.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(z_cache.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(Sq_cache.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(d_output_flat.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dx.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dW_k.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dW_v.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dW_q.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dW_z.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(db_z.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(workspace.data_ptr()));

    return {dx, dW_k, dW_v, dW_q, dW_z, db_z};
}

// =============================================================================
// E74: Diagonal State Minimal RNN - Delta Update Rule
// =============================================================================

std::vector<Tensor> e74_delta_forward(
    bool training,
    Tensor x,           // [T, B, dim] input
    Tensor s0,          // [B, n_state] initial state
    int proj_type,      // 0=tied_kvq, 1=tied_kq, 2=no_z, 3=full
    bool use_tanh,      // Apply tanh nonlinearity
    Tensor W_kvq,       // [n_state, dim] for TIED_KVQ (optional)
    Tensor W_kq,        // [n_state, dim] for TIED_KQ (optional)
    Tensor W_v,         // [n_state, dim] for TIED_KQ, NO_Z, FULL (optional)
    Tensor W_k,         // [n_state, dim] for NO_Z, FULL (optional)
    Tensor W_q,         // [n_state, dim] for NO_Z, FULL (optional)
    Tensor W_z) {       // [n_state, dim] for FULL (optional)

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    const auto n_state = s0.size(1);

    CHECK_INPUT(x);
    CHECK_INPUT(s0);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    // Outputs
    Tensor s = torch::empty({time_steps + 1, batch_size, n_state}, options);
    Tensor output = torch::empty({time_steps, batch_size, n_state}, options);

    // Copy initial state
    s.slice(0, 0, 1).copy_(s0.unsqueeze(0));

    // Caches for backward (only if training)
    Tensor k_cache = training ? torch::empty({time_steps, batch_size, n_state}, options) : torch::empty({0}, options);
    Tensor v_cache = training ? torch::empty({time_steps, batch_size, n_state}, options) : torch::empty({0}, options);
    Tensor q_cache = training ? torch::empty({time_steps, batch_size, n_state}, options) : torch::empty({0}, options);
    Tensor z_cache = (training && proj_type == 3) ? torch::empty({time_steps, batch_size, n_state}, options) : torch::empty({0}, options);
    Tensor pre_nonlin_cache = training ? torch::empty({time_steps, batch_size, n_state}, options) : torch::empty({0}, options);
    Tensor s_cache = training ? torch::empty({time_steps, batch_size, n_state}, options) : torch::empty({0}, options);

    // Only BFloat16 is implemented
    TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16,
                "E74 Delta Forward only supports bfloat16, got ", x.scalar_type());

    using namespace hasty::v0::elman_ladder;

    // Workspace
    const int64_t workspace_size = E74DeltaForward<__nv_bfloat16>::WorkspaceSize(time_steps, batch_size, n_state);
    Tensor workspace = torch::empty({workspace_size}, options);
    E74DeltaForward<__nv_bfloat16> forward(
        training, batch_size, n_state, dim, proj_type, use_tanh,
        at::cuda::getCurrentCUDABlasHandle(),
        at::cuda::getCurrentCUDAStream());

    forward.Run(
        time_steps,
        W_kvq.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(W_kvq.data_ptr()) : nullptr,
        W_kq.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(W_kq.data_ptr()) : nullptr,
        W_v.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(W_v.data_ptr()) : nullptr,
        W_k.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(W_k.data_ptr()) : nullptr,
        W_q.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(W_q.data_ptr()) : nullptr,
        W_z.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(W_z.data_ptr()) : nullptr,
        reinterpret_cast<__nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(s.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        training ? reinterpret_cast<__nv_bfloat16*>(k_cache.data_ptr()) : nullptr,
        training ? reinterpret_cast<__nv_bfloat16*>(v_cache.data_ptr()) : nullptr,
        training ? reinterpret_cast<__nv_bfloat16*>(q_cache.data_ptr()) : nullptr,
        (training && proj_type == 3) ? reinterpret_cast<__nv_bfloat16*>(z_cache.data_ptr()) : nullptr,
        training ? reinterpret_cast<__nv_bfloat16*>(pre_nonlin_cache.data_ptr()) : nullptr,
        training ? reinterpret_cast<__nv_bfloat16*>(s_cache.data_ptr()) : nullptr,
        reinterpret_cast<__nv_bfloat16*>(workspace.data_ptr()));

    return {s, output, k_cache, v_cache, q_cache, z_cache, pre_nonlin_cache, s_cache};
}

std::vector<Tensor> e74_delta_backward(
    int proj_type,
    bool use_tanh,
    Tensor W_kvq,
    Tensor W_kq,
    Tensor W_v,
    Tensor W_k,
    Tensor W_q,
    Tensor W_z,
    Tensor x,
    Tensor s,
    Tensor k_cache,
    Tensor v_cache,
    Tensor q_cache,
    Tensor z_cache,
    Tensor pre_nonlin_cache,
    Tensor s_cache,
    Tensor d_output) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    const auto n_state = s.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(s);
    CHECK_INPUT(k_cache);
    CHECK_INPUT(v_cache);
    CHECK_INPUT(d_output);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    // Output gradients
    Tensor dx = torch::empty_like(x);
    Tensor dW_kvq = (proj_type == 0) ? torch::zeros({n_state, dim}, options) : torch::empty({0}, options);
    Tensor dW_kq = (proj_type == 1) ? torch::zeros({n_state, dim}, options) : torch::empty({0}, options);
    Tensor dW_v = (proj_type >= 1) ? torch::zeros({n_state, dim}, options) : torch::empty({0}, options);
    Tensor dW_k = (proj_type >= 2) ? torch::zeros({n_state, dim}, options) : torch::empty({0}, options);
    Tensor dW_q = (proj_type >= 2) ? torch::zeros({n_state, dim}, options) : torch::empty({0}, options);
    Tensor dW_z = (proj_type == 3) ? torch::zeros({n_state, dim}, options) : torch::empty({0}, options);

    TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16,
                "E74 Delta Backward only supports bfloat16, got ", x.scalar_type());

    using namespace hasty::v0::elman_ladder;

    // Workspace
    const int64_t workspace_size = E74DeltaBackward<__nv_bfloat16>::WorkspaceSize(time_steps, batch_size, n_state);
    Tensor workspace = torch::empty({workspace_size}, options);

    E74DeltaBackward<__nv_bfloat16> backward(
        batch_size, n_state, dim, proj_type, use_tanh,
        at::cuda::getCurrentCUDABlasHandle(),
        at::cuda::getCurrentCUDAStream());

    backward.Run(
        time_steps,
        W_kvq.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(W_kvq.data_ptr()) : nullptr,
        W_kq.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(W_kq.data_ptr()) : nullptr,
        W_v.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(W_v.data_ptr()) : nullptr,
        W_k.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(W_k.data_ptr()) : nullptr,
        W_q.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(W_q.data_ptr()) : nullptr,
        W_z.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(W_z.data_ptr()) : nullptr,
        reinterpret_cast<__nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(s.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(k_cache.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(v_cache.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(q_cache.data_ptr()),
        z_cache.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(z_cache.data_ptr()) : nullptr,
        reinterpret_cast<__nv_bfloat16*>(pre_nonlin_cache.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(s_cache.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(d_output.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dx.data_ptr()),
        dW_kvq.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(dW_kvq.data_ptr()) : nullptr,
        dW_kq.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(dW_kq.data_ptr()) : nullptr,
        dW_v.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(dW_v.data_ptr()) : nullptr,
        dW_k.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(dW_k.data_ptr()) : nullptr,
        dW_q.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(dW_q.data_ptr()) : nullptr,
        dW_z.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(dW_z.data_ptr()) : nullptr,
        reinterpret_cast<__nv_bfloat16*>(workspace.data_ptr()));

    return {dx, dW_kvq, dW_kq, dW_v, dW_k, dW_q, dW_z};
}

// =============================================================================
// E74: Diagonal State Minimal RNN - Simple Update Rule
// =============================================================================

std::vector<Tensor> e74_simple_forward(
    bool training,
    Tensor x,
    Tensor s0,
    int proj_type,
    bool use_tanh,
    float decay,
    Tensor W_kvq,
    Tensor W_kq,
    Tensor W_v,
    Tensor W_k,
    Tensor W_q,
    Tensor W_z) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    const auto n_state = s0.size(1);

    CHECK_INPUT(x);
    CHECK_INPUT(s0);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor s = torch::empty({time_steps + 1, batch_size, n_state}, options);
    Tensor output = torch::empty({time_steps, batch_size, n_state}, options);

    s.slice(0, 0, 1).copy_(s0.unsqueeze(0));

    Tensor k_cache = training ? torch::empty({time_steps, batch_size, n_state}, options) : torch::empty({0}, options);
    Tensor v_cache = training ? torch::empty({time_steps, batch_size, n_state}, options) : torch::empty({0}, options);
    Tensor q_cache = training ? torch::empty({time_steps, batch_size, n_state}, options) : torch::empty({0}, options);
    Tensor z_cache = (training && proj_type == 3) ? torch::empty({time_steps, batch_size, n_state}, options) : torch::empty({0}, options);
    Tensor pre_nonlin_cache = training ? torch::empty({time_steps, batch_size, n_state}, options) : torch::empty({0}, options);
    Tensor s_cache = training ? torch::empty({time_steps, batch_size, n_state}, options) : torch::empty({0}, options);

    TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16,
                "E74 Simple Forward only supports bfloat16, got ", x.scalar_type());

    using namespace hasty::v0::elman_ladder;

    const int64_t workspace_size = E74SimpleForward<__nv_bfloat16>::WorkspaceSize(time_steps, batch_size, n_state);
    Tensor workspace = torch::empty({workspace_size}, options);

    E74SimpleForward<__nv_bfloat16> forward(
        training, batch_size, n_state, dim, proj_type, use_tanh, decay,
        at::cuda::getCurrentCUDABlasHandle(),
        at::cuda::getCurrentCUDAStream());

    forward.Run(
        time_steps,
        W_kvq.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(W_kvq.data_ptr()) : nullptr,
        W_kq.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(W_kq.data_ptr()) : nullptr,
        W_v.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(W_v.data_ptr()) : nullptr,
        W_k.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(W_k.data_ptr()) : nullptr,
        W_q.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(W_q.data_ptr()) : nullptr,
        W_z.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(W_z.data_ptr()) : nullptr,
        reinterpret_cast<__nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(s.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        training ? reinterpret_cast<__nv_bfloat16*>(k_cache.data_ptr()) : nullptr,
        training ? reinterpret_cast<__nv_bfloat16*>(v_cache.data_ptr()) : nullptr,
        training ? reinterpret_cast<__nv_bfloat16*>(q_cache.data_ptr()) : nullptr,
        (training && proj_type == 3) ? reinterpret_cast<__nv_bfloat16*>(z_cache.data_ptr()) : nullptr,
        training ? reinterpret_cast<__nv_bfloat16*>(pre_nonlin_cache.data_ptr()) : nullptr,
        training ? reinterpret_cast<__nv_bfloat16*>(s_cache.data_ptr()) : nullptr,
        reinterpret_cast<__nv_bfloat16*>(workspace.data_ptr()));

    return {s, output, k_cache, v_cache, q_cache, z_cache, pre_nonlin_cache, s_cache};
}

std::vector<Tensor> e74_simple_backward(
    int proj_type,
    bool use_tanh,
    float decay,
    Tensor W_kvq,
    Tensor W_kq,
    Tensor W_v,
    Tensor W_k,
    Tensor W_q,
    Tensor W_z,
    Tensor x,
    Tensor s,
    Tensor k_cache,
    Tensor v_cache,
    Tensor q_cache,
    Tensor z_cache,
    Tensor pre_nonlin_cache,
    Tensor s_cache,
    Tensor d_output) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    const auto n_state = s.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(s);
    CHECK_INPUT(k_cache);
    CHECK_INPUT(v_cache);
    CHECK_INPUT(d_output);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::empty_like(x);
    Tensor dW_kvq = (proj_type == 0) ? torch::zeros({n_state, dim}, options) : torch::empty({0}, options);
    Tensor dW_kq = (proj_type == 1) ? torch::zeros({n_state, dim}, options) : torch::empty({0}, options);
    Tensor dW_v = (proj_type >= 1) ? torch::zeros({n_state, dim}, options) : torch::empty({0}, options);
    Tensor dW_k = (proj_type >= 2) ? torch::zeros({n_state, dim}, options) : torch::empty({0}, options);
    Tensor dW_q = (proj_type >= 2) ? torch::zeros({n_state, dim}, options) : torch::empty({0}, options);
    Tensor dW_z = (proj_type == 3) ? torch::zeros({n_state, dim}, options) : torch::empty({0}, options);

    TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16,
                "E74 Simple Backward only supports bfloat16, got ", x.scalar_type());

    using namespace hasty::v0::elman_ladder;

    const int64_t workspace_size = E74SimpleBackward<__nv_bfloat16>::WorkspaceSize(time_steps, batch_size, n_state);
    Tensor workspace = torch::empty({workspace_size}, options);

    E74SimpleBackward<__nv_bfloat16> backward(
        batch_size, n_state, dim, proj_type, use_tanh, decay,
        at::cuda::getCurrentCUDABlasHandle(),
        at::cuda::getCurrentCUDAStream());

    backward.Run(
        time_steps,
        W_kvq.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(W_kvq.data_ptr()) : nullptr,
        W_kq.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(W_kq.data_ptr()) : nullptr,
        W_v.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(W_v.data_ptr()) : nullptr,
        W_k.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(W_k.data_ptr()) : nullptr,
        W_q.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(W_q.data_ptr()) : nullptr,
        W_z.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(W_z.data_ptr()) : nullptr,
        reinterpret_cast<__nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(s.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(k_cache.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(v_cache.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(q_cache.data_ptr()),
        z_cache.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(z_cache.data_ptr()) : nullptr,
        reinterpret_cast<__nv_bfloat16*>(pre_nonlin_cache.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(s_cache.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(d_output.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dx.data_ptr()),
        dW_kvq.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(dW_kvq.data_ptr()) : nullptr,
        dW_kq.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(dW_kq.data_ptr()) : nullptr,
        dW_v.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(dW_v.data_ptr()) : nullptr,
        dW_k.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(dW_k.data_ptr()) : nullptr,
        dW_q.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(dW_q.data_ptr()) : nullptr,
        dW_z.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(dW_z.data_ptr()) : nullptr,
        reinterpret_cast<__nv_bfloat16*>(workspace.data_ptr()));

    return {dx, dW_kvq, dW_kq, dW_v, dW_k, dW_q, dW_z};
}

// =============================================================================
// E74 Fused: Optimized kernel that processes ALL timesteps in ONE launch
// =============================================================================

std::vector<Tensor> e74_fused_forward(
    bool training,
    Tensor x,           // [T, B, dim] input
    Tensor s0,          // [B, n_state] initial state
    int proj_type,      // 0=tied_kvq, 1=tied_kq, 2=no_z, 3=full
    bool use_tanh,
    bool is_delta,      // true=delta update, false=simple update
    float decay,        // for simple update
    Tensor W_kvq,
    Tensor W_kq,
    Tensor W_v,
    Tensor W_k,
    Tensor W_q,
    Tensor W_z) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    const auto n_state = s0.size(1);

    CHECK_INPUT(x);
    CHECK_INPUT(s0);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    // Outputs
    Tensor s = torch::empty({time_steps + 1, batch_size, n_state}, options);
    Tensor output = torch::empty({time_steps, batch_size, n_state}, options);

    // Copy initial state
    s.slice(0, 0, 1).copy_(s0.unsqueeze(0));

    // Caches for backward (only if training)
    Tensor k_cache = training ? torch::empty({time_steps, batch_size, n_state}, options) : torch::empty({0}, options);
    Tensor v_cache = training ? torch::empty({time_steps, batch_size, n_state}, options) : torch::empty({0}, options);
    Tensor q_cache = training ? torch::empty({time_steps, batch_size, n_state}, options) : torch::empty({0}, options);
    Tensor z_cache = (training && proj_type == 3) ? torch::empty({time_steps, batch_size, n_state}, options) : torch::empty({0}, options);
    Tensor pre_nonlin_cache = training ? torch::empty({time_steps, batch_size, n_state}, options) : torch::empty({0}, options);
    Tensor s_cache = training ? torch::empty({time_steps, batch_size, n_state}, options) : torch::empty({0}, options);

    TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16,
                "E74 Fused Forward only supports bfloat16, got ", x.scalar_type());

    using namespace hasty::v0::elman_ladder;

    const int64_t workspace_size = E74FusedForward<__nv_bfloat16>::WorkspaceSize(time_steps, batch_size, n_state);
    Tensor workspace = torch::empty({workspace_size}, options);

    E74FusedForward<__nv_bfloat16> forward(
        training, batch_size, n_state, dim, proj_type, use_tanh, is_delta, decay,
        at::cuda::getCurrentCUDABlasHandle(),
        at::cuda::getCurrentCUDAStream());

    forward.Run(
        time_steps,
        W_kvq.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(W_kvq.data_ptr()) : nullptr,
        W_kq.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(W_kq.data_ptr()) : nullptr,
        W_v.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(W_v.data_ptr()) : nullptr,
        W_k.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(W_k.data_ptr()) : nullptr,
        W_q.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(W_q.data_ptr()) : nullptr,
        W_z.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(W_z.data_ptr()) : nullptr,
        reinterpret_cast<__nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(s.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        training ? reinterpret_cast<__nv_bfloat16*>(k_cache.data_ptr()) : nullptr,
        training ? reinterpret_cast<__nv_bfloat16*>(v_cache.data_ptr()) : nullptr,
        training ? reinterpret_cast<__nv_bfloat16*>(q_cache.data_ptr()) : nullptr,
        (training && proj_type == 3) ? reinterpret_cast<__nv_bfloat16*>(z_cache.data_ptr()) : nullptr,
        training ? reinterpret_cast<__nv_bfloat16*>(pre_nonlin_cache.data_ptr()) : nullptr,
        training ? reinterpret_cast<__nv_bfloat16*>(s_cache.data_ptr()) : nullptr,
        reinterpret_cast<__nv_bfloat16*>(workspace.data_ptr()));

    return {s, output, k_cache, v_cache, q_cache, z_cache, pre_nonlin_cache, s_cache};
}

std::vector<Tensor> e74_fused_backward(
    int proj_type,
    bool use_tanh,
    bool is_delta,
    float decay,
    Tensor W_kvq,
    Tensor W_kq,
    Tensor W_v,
    Tensor W_k,
    Tensor W_q,
    Tensor W_z,
    Tensor x,
    Tensor s,
    Tensor k_cache,
    Tensor v_cache,
    Tensor q_cache,
    Tensor z_cache,
    Tensor pre_nonlin_cache,
    Tensor s_cache,
    Tensor d_output) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);
    const auto n_state = s.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(s);
    CHECK_INPUT(k_cache);
    CHECK_INPUT(v_cache);
    CHECK_INPUT(d_output);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::empty_like(x);
    Tensor dW_kvq = (proj_type == 0) ? torch::zeros({n_state, dim}, options) : torch::empty({0}, options);
    Tensor dW_kq = (proj_type == 1) ? torch::zeros({n_state, dim}, options) : torch::empty({0}, options);
    Tensor dW_v = (proj_type >= 1) ? torch::zeros({n_state, dim}, options) : torch::empty({0}, options);
    Tensor dW_k = (proj_type >= 2) ? torch::zeros({n_state, dim}, options) : torch::empty({0}, options);
    Tensor dW_q = (proj_type >= 2) ? torch::zeros({n_state, dim}, options) : torch::empty({0}, options);
    Tensor dW_z = (proj_type == 3) ? torch::zeros({n_state, dim}, options) : torch::empty({0}, options);

    TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16,
                "E74 Fused Backward only supports bfloat16, got ", x.scalar_type());

    using namespace hasty::v0::elman_ladder;

    const int64_t workspace_size = E74FusedBackward<__nv_bfloat16>::WorkspaceSize(time_steps, batch_size, n_state);
    Tensor workspace = torch::empty({workspace_size}, options);

    E74FusedBackward<__nv_bfloat16> backward(
        batch_size, n_state, dim, proj_type, use_tanh, is_delta, decay,
        at::cuda::getCurrentCUDABlasHandle(),
        at::cuda::getCurrentCUDAStream());

    backward.Run(
        time_steps,
        W_kvq.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(W_kvq.data_ptr()) : nullptr,
        W_kq.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(W_kq.data_ptr()) : nullptr,
        W_v.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(W_v.data_ptr()) : nullptr,
        W_k.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(W_k.data_ptr()) : nullptr,
        W_q.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(W_q.data_ptr()) : nullptr,
        W_z.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(W_z.data_ptr()) : nullptr,
        reinterpret_cast<__nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(s.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(k_cache.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(v_cache.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(q_cache.data_ptr()),
        z_cache.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(z_cache.data_ptr()) : nullptr,
        reinterpret_cast<__nv_bfloat16*>(pre_nonlin_cache.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(s_cache.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(d_output.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dx.data_ptr()),
        dW_kvq.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(dW_kvq.data_ptr()) : nullptr,
        dW_kq.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(dW_kq.data_ptr()) : nullptr,
        dW_v.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(dW_v.data_ptr()) : nullptr,
        dW_k.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(dW_k.data_ptr()) : nullptr,
        dW_q.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(dW_q.data_ptr()) : nullptr,
        dW_z.numel() > 0 ? reinterpret_cast<__nv_bfloat16*>(dW_z.data_ptr()) : nullptr,
        reinterpret_cast<__nv_bfloat16*>(workspace.data_ptr()));

    return {dx, dW_kvq, dW_kq, dW_v, dW_k, dW_q, dW_z};
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
    m.def("selective_wh_elman_forward", &selective_wh_elman_forward,
          "E17: Selective W_h Elman forward (input-dependent gating on recurrence)");
    m.def("selective_wh_elman_backward", &selective_wh_elman_backward,
          "E17: Selective W_h Elman backward");
    m.def("haware_gate_elman_forward", &haware_gate_elman_forward,
          "E18: h-Aware Gate Elman forward (gate_mode: 0=A z+h, 1=B z+Rh, 2=E no gate)");
    m.def("haware_gate_elman_backward", &haware_gate_elman_backward,
          "E18: h-Aware Gate Elman backward");
    m.def("simplified_gate_elman_forward", &simplified_gate_elman_forward,
          "E19: Simplified Gate Elman forward (gate_mode: 0=A Wx+h, 1=B h-only, 2=D residual+z, 3=E residual+Wx+h)");
    m.def("simplified_gate_elman_backward", &simplified_gate_elman_backward,
          "E19: Simplified Gate Elman backward");
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
    m.def("structured_elman_forward", &structured_elman_forward,
          "E21: Structured Elman forward (MIMO with nonlinear state mixing)");
    m.def("structured_elman_backward", &structured_elman_backward,
          "E21: Structured Elman backward");
    m.def("structured_elman_attention_forward", &structured_elman_attention_forward,
          "E22: Structured Elman with State Attention forward (UTM class)");
    m.def("structured_elman_attention_backward", &structured_elman_attention_backward,
          "E22: Structured Elman with State Attention backward");
    m.def("dual_memory_elman_forward", &dual_memory_elman_forward,
          "E23: Dual-Memory Elman forward (tape + working memory)");
    m.def("dual_memory_elman_backward", &dual_memory_elman_backward,
          "E23: Dual-Memory Elman backward");
    m.def("dual_memory_elman_forward_opt", &dual_memory_elman_forward_opt,
          "E23: Dual-Memory Elman optimized forward (cuBLAS batched GEMM attention)");
    m.def("e23c_chunked_forward", &e23c_chunked_forward,
          "E23c: Chunked Dual-Memory Elman forward (batched attention, no read in h_work)");
    m.def("e23cv2_chunked_forward", &e23cv2_chunked_forward,
          "E23c_v2: Chunked Dual-Memory Elman forward with read feedback");
    m.def("e24_single_gemm_forward", &e24_single_gemm_forward,
          "E24: Single-GEMM Dual Memory forward (1 GEMM per timestep)");
    m.def("e24_single_gemm_backward", &e24_single_gemm_backward,
          "E24: Single-GEMM Dual Memory backward");
    m.def("e25_entmax_forward", &e25_entmax_forward,
          "E25: Dual-Memory Elman with 1.5-Entmax forward (sparse attention)");
    m.def("e25_entmax_backward", &e25_entmax_backward,
          "E25: Dual-Memory Elman with 1.5-Entmax backward");
    m.def("e26_parallel_forward", &e26_parallel_forward,
          "E26: Parallel Dual-Memory Elman forward (softmax attention)");
    m.def("e26_parallel_backward", &e26_parallel_backward,
          "E26: Parallel Dual-Memory Elman backward");
    m.def("e28_conv_forward", &e28_conv_forward,
          "E28: E1 + Mamba2 Conv forward (depthwise causal conv1d + Elman)");
    m.def("e28_conv_backward", &e28_conv_backward,
          "E28: E1 + Mamba2 Conv backward");
    m.def("e29a_selective_forward", &e29a_selective_forward,
          "E29a: Selective Dual-Memory Elman forward (additive gate: silu(z + read + h_work))");
    m.def("e29b_selective_forward", &e29b_selective_forward,
          "E29b: Selective Dual-Memory Elman forward (learned gate: W_gate @ [z; read; h_work])");
    m.def("e29a_selective_backward", &e29a_selective_backward,
          "E29a: Selective Dual-Memory Elman backward");
    m.def("e29b_selective_backward", &e29b_selective_backward,
          "E29b: Selective Dual-Memory Elman backward");
    m.def("e29c_diagonal_forward", &e29c_diagonal_forward,
          "E29c: SSM-Style Diagonal Gating Dual-Memory Elman forward (gate = silu(z*g_z + read*g_r + h*g_h + b))");
    m.def("e29c_diagonal_backward", &e29c_diagonal_backward,
          "E29c: SSM-Style Diagonal Gating Dual-Memory Elman backward");
    m.def("e30_diagonal_gated_forward", &e30_diagonal_gated_forward,
          "E30: E1 + Diagonal Gating forward (gate = silu(z*g_z + h*g_h + b))");
    m.def("e30_diagonal_gated_backward", &e30_diagonal_gated_backward,
          "E30: E1 + Diagonal Gating backward");
    m.def("e33_self_gate_forward", &e33_self_gate_forward,
          "E33: Self-Gated Elman forward (output = h * silu(h))");
    m.def("e33_self_gate_backward", &e33_self_gate_backward,
          "E33: Self-Gated Elman backward");
    m.def("e40_no_presilu_forward", &e40_no_presilu_forward,
          "E40: No Pre-SiLU Elman forward (h = tanh(x + Rh + b), output = h * silu(h), no W_x, no pre-silu)");
    m.def("e40_no_presilu_backward", &e40_no_presilu_backward,
          "E40: No Pre-SiLU Elman backward");
    m.def("e34_diagonal_wh_forward", &e34_diagonal_wh_forward,
          "E34: Diagonal W_h Elman forward (W_h is diagonal, element-wise multiply)");
    m.def("e34_diagonal_wh_backward", &e34_diagonal_wh_backward,
          "E34: Diagonal W_h Elman backward");
    m.def("e35_cubic_gate_forward", &e35_cubic_gate_forward,
          "E35: Cubic Gate Elman forward (output = h * h * h instead of h * silu(h))");
    m.def("e35_cubic_gate_backward", &e35_cubic_gate_backward,
          "E35: Cubic Gate Elman backward");
    m.def("e36_linear_recurrence_forward", &e36_linear_recurrence_forward,
          "E36: Linear Recurrence Elman forward (h = x + Wh*h_prev, no tanh)");
    m.def("e36_linear_recurrence_backward", &e36_linear_recurrence_backward,
          "E36: Linear Recurrence Elman backward");
    m.def("e37_tied_weights_forward", &e37_tied_weights_forward,
          "E37: Tied Weights Elman forward (W_x = W_h)");
    m.def("e37_tied_weights_backward", &e37_tied_weights_backward,
          "E37: Tied Weights Elman backward");
    m.def("e37_tied_weights_v2_forward", &e37_tied_weights_v2_forward,
          "E37v2: Optimized Tied Weights forward (batched W @ x)");
    m.def("e37_tied_weights_v2_backward", &e37_tied_weights_v2_backward,
          "E37v2: Optimized Tied Weights backward");
    m.def("e38_no_wx_forward", &e38_no_wx_forward,
          "E38: No W_x Elman forward (x goes directly into recurrence)");
    m.def("e38_no_wx_backward", &e38_no_wx_backward,
          "E38: No W_x Elman backward");
    m.def("e39_no_bias_forward", &e39_no_bias_forward,
          "E39: No Bias Elman forward (E38 without bias term)");
    m.def("e39_no_bias_backward", &e39_no_bias_backward,
          "E39: No Bias Elman backward");
    m.def("e41_diagonal_wx_forward", &e41_diagonal_wx_forward,
          "E41: Diagonal W_x Elman forward (W_x is diagonal, element-wise scaling)");
    m.def("e41_diagonal_wx_backward", &e41_diagonal_wx_backward,
          "E41: Diagonal W_x Elman backward");
    m.def("e42_linear_tied_forward", &e42_linear_tied_forward,
          "E42: Linear Tied Self-Gated Elman forward (h = W@x + W@h + b, no tanh)");
    m.def("e42_linear_tied_backward", &e42_linear_tied_backward,
          "E42: Linear Tied Self-Gated Elman backward");
    m.def("e52_quadratic_gate_forward", &e52_quadratic_gate_forward,
          "E52: Quadratic Gate Elman forward (output = h^2 or h*|h|)");
    m.def("e52_quadratic_gate_backward", &e52_quadratic_gate_backward,
          "E52: Quadratic Gate Elman backward");
    m.def("e53_sigmoid_gate_forward", &e53_sigmoid_gate_forward,
          "E53: Sigmoid Gate Only Elman forward (output = silu(h), NOT h*silu(h))");
    m.def("e53_sigmoid_gate_backward", &e53_sigmoid_gate_backward,
          "E53: Sigmoid Gate Only Elman backward");
    m.def("e48_no_projections_forward", &e48_no_projections_forward,
          "E48: No Projections Elman forward (h = W@(x+h) + b, output = h*silu(h), NO in_proj/out_proj)");
    m.def("e48_no_projections_backward", &e48_no_projections_backward,
          "E48: No Projections Elman backward");
    m.def("e51_no_self_gate_forward", &e51_no_self_gate_forward,
          "E51: No Self-Gate Elman forward (h = W@(x+h) + b, output = h, NO self-gating)");
    m.def("e51_no_self_gate_backward", &e51_no_self_gate_backward,
          "E51: No Self-Gate Elman backward");
    m.def("e54_diagonal_no_proj_forward", &e54_diagonal_no_proj_forward,
          "E54: Diagonal No-Proj forward (h = d*(x+h) + b, output = h*silu(h), NO GEMM)");
    m.def("e54_diagonal_no_proj_backward", &e54_diagonal_no_proj_backward,
          "E54: Diagonal No-Proj backward");
    m.def("e55_scalar_no_proj_forward", &e55_scalar_no_proj_forward,
          "E55: Scalar No-Proj forward (h = lambda*(x+h) + b, output = h*silu(h), NO GEMM)");
    m.def("e55_scalar_no_proj_backward", &e55_scalar_no_proj_backward,
          "E55: Scalar No-Proj backward");
    m.def("e45_pure_accumulation_forward", &e45_pure_accumulation_forward,
          "E45: Pure Accumulation forward (h = x + h_prev, NO GEMM, simplest possible)");
    m.def("e45_pure_accumulation_backward", &e45_pure_accumulation_backward,
          "E45: Pure Accumulation backward");
    m.def("e45b_with_decay_forward", &e45b_with_decay_forward,
          "E45b: Pure Accumulation with Decay forward (h = x + alpha*h_prev)");
    m.def("e45b_with_decay_backward", &e45b_with_decay_backward,
          "E45b: Pure Accumulation with Decay backward");
    m.def("e46_no_in_proj_forward", &e46_no_in_proj_forward,
          "E46: No In-Proj forward (h = W@(x+h) + b, operates on embedding dim)");
    m.def("e46_no_in_proj_backward", &e46_no_in_proj_backward,
          "E46: No In-Proj backward");
    m.def("e43_scalar_decay_forward", &e43_scalar_decay_forward,
          "E43: Scalar Decay forward (h = sigmoid(log_lambda)*(x+h) + b, NO GEMM)");
    m.def("e43_scalar_decay_backward", &e43_scalar_decay_backward,
          "E43: Scalar Decay backward");
    m.def("e44_diagonal_w_forward", &e44_diagonal_w_forward,
          "E44: Diagonal W forward (h = sigmoid(log_d)*(x+h) + b, per-dim decay, NO GEMM)");
    m.def("e44_diagonal_w_backward", &e44_diagonal_w_backward,
          "E44: Diagonal W backward");
    m.def("e56_concat_elman_forward", &e56_concat_elman_forward,
          "E56: Concat Elman forward (h = tanh(W@[x;h] + b), output = h*silu(z), single GEMM)");
    m.def("e56_concat_elman_backward", &e56_concat_elman_backward,
          "E56: Concat Elman backward");
    m.def("e58_learned_radii_forward", &e58_learned_radii_forward,
          "E58: Per-Dimension Learned Radii Elman forward (W_h scaled by per-dim radii)");
    m.def("e58_learned_radii_backward", &e58_learned_radii_backward,
          "E58: Per-Dimension Learned Radii Elman backward");
    m.def("e59_highway_forward", &e59_highway_forward,
          "E59: Highway Elman forward (h = h_prev + alpha*(W@x + b), perfect gradient flow)");
    m.def("e59_highway_backward", &e59_highway_backward,
          "E59: Highway Elman backward");
    m.def("e60_residual_nonlinear_forward", &e60_residual_nonlinear_forward,
          "E60: Residual Nonlinear Elman forward (h = h + alpha*tanh(Wh + Wx + b))");
    m.def("e60_residual_nonlinear_backward", &e60_residual_nonlinear_backward,
          "E60: Residual Nonlinear Elman backward");
    m.def("e61_decay_gated_forward", &e61_decay_gated_forward,
          "E61: Decay-Gated Elman forward (h = alpha*h + (1-alpha)*v, Mamba2-style)");
    m.def("e61_decay_gated_backward", &e61_decay_gated_backward,
          "E61: Decay-Gated Elman backward");
    m.def("e62_selective_write_forward", &e62_selective_write_forward,
          "E62: Selective Write Elman forward (h = (1-k)*h + k*v, vector DeltaNet)");
    m.def("e62_selective_write_backward", &e62_selective_write_backward,
          "E62: Selective Write Elman backward");
    m.def("e63_nonlinear_delta_forward", &e63_nonlinear_delta_forward,
          "E63: Nonlinear Delta Elman forward (h = alpha*h + (1-alpha)*tanh(Wh+Wx+b), UTM-class)");
    m.def("e63_nonlinear_delta_backward", &e63_nonlinear_delta_backward,
          "E63: Nonlinear Delta Elman backward");
    m.def("e64_additive_h_forward", &e64_additive_h_forward,
          "E64: Additive H-Dependence forward (v = tanh(h + Wx + b), UTM-class O(d) h-dep)");
    m.def("e64_additive_h_backward", &e64_additive_h_backward,
          "E64: Additive H-Dependence backward");
    m.def("e65_diagonal_h_forward", &e65_diagonal_h_forward,
          "E65: Diagonal H-Dependence forward (v = tanh(d_h*h + Wx + b), UTM-class O(d) h-dep)");
    m.def("e65_diagonal_h_backward", &e65_diagonal_h_backward,
          "E65: Diagonal H-Dependence backward");
    m.def("e66_lowrank_h_forward", &e66_lowrank_h_forward,
          "E66: Low-Rank H-Dependence forward (v = tanh(U@V@h + Wx + b), UTM-class O(d*rank) h-dep)");
    m.def("e66_lowrank_h_backward", &e66_lowrank_h_backward,
          "E66: Low-Rank H-Dependence backward");
    m.def("e67_h_gated_forward", &e67_h_gated_forward,
          "E67: H-Gated forward (alpha = sigmoid(Wx + d_alpha*h + b), v = tanh(Wx + b), UTM-class O(d) gate h-dep)");
    m.def("e67_h_gated_backward", &e67_h_gated_backward,
          "E67: H-Gated backward");
    m.def("e68_self_gating_forward", &e68_self_gating_forward,
          "E68: Self-Gating forward (g = sigmoid(d_g*h + b_g), v = tanh(Wx)*g, UTM-class O(d) value gating)");
    m.def("e68_self_gating_backward", &e68_self_gating_backward,
          "E68: Self-Gating backward");

    // GRU and LSTM - BF16 optimized (avoid cuDNN bf16 regression)
    m.def("gru_forward", &gru_forward,
          "GRU: Gated Recurrent Unit forward (BF16 optimized, avoids cuDNN regression)");
    m.def("gru_backward", &gru_backward,
          "GRU: Gated Recurrent Unit backward");
    m.def("lstm_forward", &lstm_forward,
          "LSTM: Long Short-Term Memory forward (BF16 optimized, avoids cuDNN regression)");
    m.def("lstm_backward", &lstm_backward,
          "LSTM: Long Short-Term Memory backward");
    m.def("e63m_matrix_nonlinear_forward", &e63m_matrix_nonlinear_forward,
          "E63m: Matrix State Nonlinear Delta forward (UTM-class with N×D matrix state)");
    m.def("e63m_matrix_nonlinear_backward", &e63m_matrix_nonlinear_backward,
          "E63m: Matrix State Nonlinear Delta backward");
    m.def("e70_matrix_linear_forward", &e70_matrix_linear_forward,
          "E70: Matrix Linear forward (S = tanh(decay*S + outer(v,k)), out = (S@q)*silu(S@q))");
    m.def("e70_matrix_linear_backward", &e70_matrix_linear_backward,
          "E70: Matrix Linear backward");
    m.def("e71_matrix_gated_forward", &e71_matrix_gated_forward,
          "E71: Matrix Gated forward (alpha = sigmoid(W_alpha@x + d_alpha*retrieved + b_alpha), S-dependent gating)");
    m.def("e71_matrix_gated_backward", &e71_matrix_gated_backward,
          "E71: Matrix Gated backward");
    m.def("e71_delta_forward", &e71_delta_forward,
          "E71 Delta: Matrix Gated with Delta Rule forward (beta = sigmoid(W_beta@x + d_beta*retrieved), S + beta*outer(v-ret, k_norm))");
    m.def("e71_delta_backward", &e71_delta_backward,
          "E71 Delta: Matrix Gated with Delta Rule backward");
    m.def("e72_matrix_selfgate_forward", &e72_matrix_selfgate_forward,
          "E72: Matrix SelfGate forward (g = sigmoid(d_g*retrieved + b_g), memory content controls writing)");
    m.def("e72_matrix_selfgate_backward", &e72_matrix_selfgate_backward,
          "E72: Matrix SelfGate backward");
    m.def("e73_matrix_nonlinear_forward", &e73_matrix_nonlinear_forward,
          "E73: Matrix Nonlinear forward (S = tanh(S*z + outer(v,k)), variants: 0=column, 1=row, 2=full)");
    m.def("e73_matrix_nonlinear_backward", &e73_matrix_nonlinear_backward,
          "E73: Matrix Nonlinear backward");
    m.def("e73_fused_forward", &e73_fused_forward,
          "E73 Fused: Optimized Matrix Nonlinear with single persistent kernel");
    m.def("e73_checkpointed_forward", &e73_checkpointed_forward,
          "E73 Checkpointed: In-place S with gradient checkpointing (50x memory reduction)");
    m.def("e73_checkpointed_backward", &e73_checkpointed_backward,
          "E73 Checkpointed: Backward with checkpoint-based recomputation");

    // E74: Diagonal State Minimal RNN
    m.def("e74_delta_forward", &e74_delta_forward,
          "E74 Delta: Diagonal state with delta update rule (proj_type: 0=tied_kvq, 1=tied_kq, 2=no_z, 3=full)");
    m.def("e74_delta_backward", &e74_delta_backward,
          "E74 Delta: Backward pass for delta update rule");
    m.def("e74_simple_forward", &e74_simple_forward,
          "E74 Simple: Diagonal state with simple update rule (s_new = decay*s + v*k)");
    m.def("e74_simple_backward", &e74_simple_backward,
          "E74 Simple: Backward pass for simple update rule");
    m.def("e74_fused_forward", &e74_fused_forward,
          "E74 Fused: Optimized kernel that processes ALL timesteps in ONE launch");
    m.def("e74_fused_backward", &e74_fused_backward,
          "E74 Fused: Backward pass with single kernel launch");
}
