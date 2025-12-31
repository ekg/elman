// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// PyTorch bindings for Elman Ablation Ladder kernels.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>

#include "hasty.h"
#include "support.h"

namespace {

using torch::Tensor;

// =============================================================================
// Level 0: Stock Elman
// =============================================================================

std::vector<Tensor> stock_elman_forward(
    bool training,
    Tensor x,           // [T, B, dim]
    Tensor h0,          // [B, dim]
    Tensor W_x,         // [dim, dim]
    Tensor W_h,         // [dim, dim]
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
    Tensor v = training ? torch::empty({time_steps, batch_size, dim}, options)
                        : torch::empty({0}, options);

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
            ptr<scalar_t>(b),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            training ? ptr<scalar_t>(v) : nullptr);
    }));

    return {h, v};
}

std::vector<Tensor> stock_elman_backward(
    Tensor W_x,
    Tensor W_h,
    Tensor x,
    Tensor h,
    Tensor v,
    Tensor dh_out) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(W_x);
    CHECK_INPUT(W_h);
    CHECK_INPUT(x);
    CHECK_INPUT(h);
    CHECK_INPUT(v);
    CHECK_INPUT(dh_out);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::empty_like(x);
    Tensor dW_x = torch::zeros({dim, dim}, options);
    Tensor dW_h = torch::zeros({dim, dim}, options);
    Tensor db = torch::zeros({dim}, options);

    // Workspace: (T+1) * B * dim for dv_all + dh_recurrent, plus dim floats for db
    // Use bytes to properly handle float buffer at end
    const int64_t elem_size = x.element_size();
    const int64_t workspace_elems = (time_steps + 1) * batch_size * dim;
    const int64_t float_bytes = dim * sizeof(float);
    const int64_t float_elems = (float_bytes + elem_size - 1) / elem_size;  // Round up
    Tensor workspace = torch::empty({workspace_elems + float_elems}, options);

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
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(dh_out),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dW_x),
            ptr<scalar_t>(dW_h),
            ptr<scalar_t>(db),
            ptr<scalar_t>(workspace));
    }));

    return {dx, dW_x, dW_h, db};
}

// =============================================================================
// Level 1: Gated Elman
// =============================================================================

std::vector<Tensor> gated_elman_forward(
    bool training,
    Tensor x,
    Tensor h0,
    Tensor W_x,
    Tensor W_h,
    Tensor W_delta,
    Tensor b,
    Tensor b_delta) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(W_x);
    CHECK_INPUT(W_h);
    CHECK_INPUT(W_delta);
    CHECK_INPUT(b);
    CHECK_INPUT(b_delta);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, dim}, options)
                        : torch::empty({0}, options);
    Tensor delta_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                  : torch::empty({0}, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "gated_elman_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        GatedElmanForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(W_h),
            ptr<scalar_t>(W_delta),
            ptr<scalar_t>(b),
            ptr<scalar_t>(b_delta),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            training ? ptr<scalar_t>(v) : nullptr,
            training ? ptr<scalar_t>(delta_cache) : nullptr);
    }));

    return {h, v, delta_cache};
}

std::vector<Tensor> gated_elman_backward(
    Tensor W_x,
    Tensor W_h,
    Tensor W_delta,
    Tensor x,
    Tensor h,
    Tensor v,
    Tensor delta_cache,
    Tensor dh_out) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::empty_like(x);
    Tensor dW_x = torch::zeros({dim, dim}, options);
    Tensor dW_h = torch::zeros({dim, dim}, options);
    Tensor dW_delta = torch::zeros({dim, dim}, options);
    Tensor db = torch::zeros({dim}, options);
    Tensor db_delta = torch::zeros({dim}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "gated_elman_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        GatedElmanBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(W_h),
            ptr<scalar_t>(W_delta),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(delta_cache),
            ptr<scalar_t>(dh_out),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dW_x),
            ptr<scalar_t>(dW_h),
            ptr<scalar_t>(dW_delta),
            ptr<scalar_t>(db),
            ptr<scalar_t>(db_delta));
    }));

    return {dx, dW_x, dW_h, dW_delta, db, db_delta};
}

// =============================================================================
// Level 2: Selective Elman
// =============================================================================

std::vector<Tensor> selective_elman_forward(
    bool training,
    Tensor x,
    Tensor h0,
    Tensor W_x,
    Tensor W_h,
    Tensor W_delta,
    Tensor W_out,
    Tensor b,
    Tensor b_delta,
    int n_groups) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(W_x);
    CHECK_INPUT(W_h);
    CHECK_INPUT(W_delta);
    CHECK_INPUT(W_out);
    CHECK_INPUT(b);
    CHECK_INPUT(b_delta);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, dim}, options)
                        : torch::empty({0}, options);
    Tensor delta_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                  : torch::empty({0}, options);
    Tensor compete_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                    : torch::empty({0}, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "selective_elman_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        SelectiveElmanForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, n_groups,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(W_h),
            ptr<scalar_t>(W_delta),
            ptr<scalar_t>(W_out),
            ptr<scalar_t>(b),
            ptr<scalar_t>(b_delta),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v) : nullptr,
            training ? ptr<scalar_t>(delta_cache) : nullptr,
            training ? ptr<scalar_t>(compete_cache) : nullptr);
    }));

    return {h, output, v, delta_cache, compete_cache};
}

std::vector<Tensor> selective_elman_backward(
    Tensor W_x,
    Tensor W_h,
    Tensor W_delta,
    Tensor W_out,
    Tensor x,
    Tensor h,
    Tensor v,
    Tensor delta_cache,
    Tensor compete_cache,
    Tensor d_output,
    int n_groups) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::empty_like(x);
    Tensor dW_x = torch::zeros({dim, dim}, options);
    Tensor dW_h = torch::zeros({dim, dim}, options);
    Tensor dW_delta = torch::zeros({dim, dim}, options);
    Tensor dW_out = torch::zeros({dim, dim}, options);
    Tensor db = torch::zeros({dim}, options);
    Tensor db_delta = torch::zeros({dim}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "selective_elman_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        SelectiveElmanBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, n_groups,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(W_h),
            ptr<scalar_t>(W_delta),
            ptr<scalar_t>(W_out),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(delta_cache),
            ptr<scalar_t>(compete_cache),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dW_x),
            ptr<scalar_t>(dW_h),
            ptr<scalar_t>(dW_delta),
            ptr<scalar_t>(dW_out),
            ptr<scalar_t>(db),
            ptr<scalar_t>(db_delta));
    }));

    return {dx, dW_x, dW_h, dW_delta, dW_out, db, db_delta};
}

// =============================================================================
// Level 3: Diagonal Selective Elman
// =============================================================================

std::vector<Tensor> diagonal_selective_forward(
    bool training,
    Tensor x,
    Tensor h0,
    Tensor W_x,
    Tensor r_h,         // [dim] diagonal, not matrix
    Tensor W_delta,
    Tensor W_out,
    Tensor b,
    Tensor b_delta,
    int n_groups) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(W_x);
    CHECK_INPUT(r_h);
    CHECK_INPUT(W_delta);
    CHECK_INPUT(W_out);
    CHECK_INPUT(b);
    CHECK_INPUT(b_delta);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, dim}, options)
                        : torch::empty({0}, options);
    Tensor delta_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                  : torch::empty({0}, options);
    Tensor compete_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                    : torch::empty({0}, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "diagonal_selective_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        DiagonalSelectiveElmanForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, n_groups,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(r_h),
            ptr<scalar_t>(W_delta),
            ptr<scalar_t>(W_out),
            ptr<scalar_t>(b),
            ptr<scalar_t>(b_delta),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v) : nullptr,
            training ? ptr<scalar_t>(delta_cache) : nullptr,
            training ? ptr<scalar_t>(compete_cache) : nullptr);
    }));

    return {h, output, v, delta_cache, compete_cache};
}

std::vector<Tensor> diagonal_selective_backward(
    Tensor W_x,
    Tensor r_h,
    Tensor W_delta,
    Tensor W_out,
    Tensor x,
    Tensor h,
    Tensor v,
    Tensor delta_cache,
    Tensor compete_cache,
    Tensor d_output,
    int n_groups) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::empty_like(x);
    Tensor dW_x = torch::zeros({dim, dim}, options);
    Tensor dr_h = torch::zeros({dim}, options);  // Diagonal gradient
    Tensor dW_delta = torch::zeros({dim, dim}, options);
    Tensor dW_out = torch::zeros({dim, dim}, options);
    Tensor db = torch::zeros({dim}, options);
    Tensor db_delta = torch::zeros({dim}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "diagonal_selective_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        DiagonalSelectiveElmanBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, n_groups,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(r_h),
            ptr<scalar_t>(W_delta),
            ptr<scalar_t>(W_out),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(delta_cache),
            ptr<scalar_t>(compete_cache),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dW_x),
            ptr<scalar_t>(dr_h),
            ptr<scalar_t>(dW_delta),
            ptr<scalar_t>(dW_out),
            ptr<scalar_t>(db),
            ptr<scalar_t>(db_delta));
    }));

    return {dx, dW_x, dr_h, dW_delta, dW_out, db, db_delta};
}

// =============================================================================
// Level 4: Full Recurrence Elman (Linear Space)
// Like Diagonal Selective but with FULL R_h matrix
// =============================================================================

std::vector<Tensor> full_recurrence_forward(
    bool training,
    Tensor x,
    Tensor h0,
    Tensor W_x,
    Tensor R_h,         // [dim, dim] FULL matrix
    Tensor W_delta,
    Tensor W_out,
    Tensor b,
    Tensor b_delta,
    int n_groups) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(W_x);
    CHECK_INPUT(R_h);
    CHECK_INPUT(W_delta);
    CHECK_INPUT(W_out);
    CHECK_INPUT(b);
    CHECK_INPUT(b_delta);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, dim}, options)
                        : torch::empty({0}, options);
    Tensor delta_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                  : torch::empty({0}, options);
    Tensor compete_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                    : torch::empty({0}, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "full_recurrence_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        FullRecurrenceElmanForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, n_groups,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(R_h),
            ptr<scalar_t>(W_delta),
            ptr<scalar_t>(W_out),
            ptr<scalar_t>(b),
            ptr<scalar_t>(b_delta),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v) : nullptr,
            training ? ptr<scalar_t>(delta_cache) : nullptr,
            training ? ptr<scalar_t>(compete_cache) : nullptr);
    }));

    return {h, output, v, delta_cache, compete_cache};
}

std::vector<Tensor> full_recurrence_backward(
    Tensor W_x,
    Tensor R_h,
    Tensor W_delta,
    Tensor W_out,
    Tensor x,
    Tensor h,
    Tensor v,
    Tensor delta_cache,
    Tensor compete_cache,
    Tensor d_output,
    int n_groups) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::empty_like(x);
    Tensor dW_x = torch::zeros({dim, dim}, options);
    Tensor dR_h = torch::zeros({dim, dim}, options);
    Tensor dW_delta = torch::zeros({dim, dim}, options);
    Tensor dW_out = torch::zeros({dim, dim}, options);
    Tensor db = torch::zeros({dim}, options);
    Tensor db_delta = torch::zeros({dim}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "full_recurrence_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        FullRecurrenceElmanBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, n_groups,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(R_h),
            ptr<scalar_t>(W_delta),
            ptr<scalar_t>(W_out),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(delta_cache),
            ptr<scalar_t>(compete_cache),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dW_x),
            ptr<scalar_t>(dR_h),
            ptr<scalar_t>(dW_delta),
            ptr<scalar_t>(dW_out),
            ptr<scalar_t>(db),
            ptr<scalar_t>(db_delta));
    }));

    return {dx, dW_x, dR_h, dW_delta, dW_out, db, db_delta};
}

// =============================================================================
// Level 5: Linear Triple R Elman
// v = R_x @ x + R_h @ h_prev + b
// delta = sigmoid(W_delta @ x + R_delta @ h_prev + b_delta)
// =============================================================================

std::vector<Tensor> linear_triple_r_forward(
    bool training,
    Tensor x,
    Tensor h0,
    Tensor R_h,
    Tensor R_x,
    Tensor R_delta,
    Tensor W_delta,
    Tensor W_out,
    Tensor b,
    Tensor b_delta,
    int n_groups) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(R_h);
    CHECK_INPUT(R_x);
    CHECK_INPUT(R_delta);
    CHECK_INPUT(W_delta);
    CHECK_INPUT(W_out);
    CHECK_INPUT(b);
    CHECK_INPUT(b_delta);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, dim}, options)
                        : torch::empty({0}, options);
    Tensor delta_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                  : torch::empty({0}, options);
    Tensor compete_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                    : torch::empty({0}, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "linear_triple_r_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        LinearTripleRForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, n_groups,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(R_h),
            ptr<scalar_t>(R_x),
            ptr<scalar_t>(R_delta),
            ptr<scalar_t>(W_delta),
            ptr<scalar_t>(W_out),
            ptr<scalar_t>(b),
            ptr<scalar_t>(b_delta),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v) : nullptr,
            training ? ptr<scalar_t>(delta_cache) : nullptr,
            training ? ptr<scalar_t>(compete_cache) : nullptr);
    }));

    return {h, output, v, delta_cache, compete_cache};
}

std::vector<Tensor> linear_triple_r_backward(
    Tensor R_h,
    Tensor R_x,
    Tensor R_delta,
    Tensor W_delta,
    Tensor W_out,
    Tensor x,
    Tensor h,
    Tensor v,
    Tensor delta_cache,
    Tensor compete_cache,
    Tensor d_output,
    int n_groups) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::empty_like(x);
    Tensor dR_h = torch::zeros({dim, dim}, options);
    Tensor dR_x = torch::zeros({dim, dim}, options);
    Tensor dR_delta = torch::zeros({dim, dim}, options);
    Tensor dW_delta = torch::zeros({dim, dim}, options);
    Tensor dW_out = torch::zeros({dim, dim}, options);
    Tensor db = torch::zeros({dim}, options);
    Tensor db_delta = torch::zeros({dim}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "linear_triple_r_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        LinearTripleRBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, n_groups,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(R_h),
            ptr<scalar_t>(R_x),
            ptr<scalar_t>(R_delta),
            ptr<scalar_t>(W_delta),
            ptr<scalar_t>(W_out),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(delta_cache),
            ptr<scalar_t>(compete_cache),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dR_h),
            ptr<scalar_t>(dR_x),
            ptr<scalar_t>(dR_delta),
            ptr<scalar_t>(dW_delta),
            ptr<scalar_t>(dW_out),
            ptr<scalar_t>(db),
            ptr<scalar_t>(db_delta));
    }));

    return {dx, dR_h, dR_x, dR_delta, dW_delta, dW_out, db, db_delta};
}

// =============================================================================
// Level 6: Linear Polynomial Elman
// alpha = 1 + softplus(W_alpha @ x + b_alpha)
// candidate = sign(v) * |v|^alpha
// =============================================================================

std::vector<Tensor> linear_polynomial_forward(
    bool training,
    Tensor x,
    Tensor h0,
    Tensor W_x,
    Tensor r_h,
    Tensor W_alpha,
    Tensor b_alpha,
    Tensor W_delta,
    Tensor W_out,
    Tensor b,
    Tensor b_delta,
    int n_groups) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(W_x);
    CHECK_INPUT(r_h);
    CHECK_INPUT(W_alpha);
    CHECK_INPUT(b_alpha);
    CHECK_INPUT(W_delta);
    CHECK_INPUT(W_out);
    CHECK_INPUT(b);
    CHECK_INPUT(b_delta);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, dim}, options)
                        : torch::empty({0}, options);
    Tensor alpha_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                  : torch::empty({0}, options);
    Tensor delta_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                  : torch::empty({0}, options);
    Tensor compete_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                    : torch::empty({0}, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "linear_polynomial_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        LinearPolynomialForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, n_groups,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(r_h),
            ptr<scalar_t>(W_alpha),
            ptr<scalar_t>(b_alpha),
            ptr<scalar_t>(W_delta),
            ptr<scalar_t>(W_out),
            ptr<scalar_t>(b),
            ptr<scalar_t>(b_delta),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v) : nullptr,
            training ? ptr<scalar_t>(alpha_cache) : nullptr,
            training ? ptr<scalar_t>(delta_cache) : nullptr,
            training ? ptr<scalar_t>(compete_cache) : nullptr);
    }));

    return {h, output, v, alpha_cache, delta_cache, compete_cache};
}

std::vector<Tensor> linear_polynomial_backward(
    Tensor W_x,
    Tensor r_h,
    Tensor W_alpha,
    Tensor W_delta,
    Tensor W_out,
    Tensor x,
    Tensor h,
    Tensor v,
    Tensor alpha_cache,
    Tensor delta_cache,
    Tensor compete_cache,
    Tensor d_output,
    int n_groups) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::empty_like(x);
    Tensor dW_x = torch::zeros({dim, dim}, options);
    Tensor dr_h = torch::zeros({dim}, options);
    Tensor dW_alpha = torch::zeros({dim, dim}, options);
    Tensor db_alpha = torch::zeros({dim}, options);
    Tensor dW_delta = torch::zeros({dim, dim}, options);
    Tensor dW_out = torch::zeros({dim, dim}, options);
    Tensor db = torch::zeros({dim}, options);
    Tensor db_delta = torch::zeros({dim}, options);

    // Workspace: (4*T+5)*B*dim T elements + 4*dim floats
    // Layout: [dv_all: TBD][d_alpha_all: TBD][d_delta_all: TBD][d_w_out_h_all: TBD]
    //         [w_out_h: BD][dh_compete: BD][dh: BD][dh_prev_out: BD][dh_recurrent: BD]
    //         [dr_h_f: dim floats][db_f: dim][db_delta_f: dim][db_alpha_f: dim]
    const int64_t elem_size = x.element_size();
    const int64_t BD = batch_size * dim;
    const int64_t TBD = time_steps * BD;
    const int64_t t_elems = 4 * TBD + 5 * BD;  // T-type elements
    const int64_t float_bytes = 4 * dim * sizeof(float);
    const int64_t float_elems = (float_bytes + elem_size - 1) / elem_size;  // Round up
    Tensor workspace = torch::empty({t_elems + float_elems}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "linear_polynomial_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        LinearPolynomialBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, n_groups,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(r_h),
            ptr<scalar_t>(W_alpha),
            ptr<scalar_t>(W_delta),
            ptr<scalar_t>(W_out),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(alpha_cache),
            ptr<scalar_t>(delta_cache),
            ptr<scalar_t>(compete_cache),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dW_x),
            ptr<scalar_t>(dr_h),
            ptr<scalar_t>(dW_alpha),
            ptr<scalar_t>(db_alpha),
            ptr<scalar_t>(dW_delta),
            ptr<scalar_t>(dW_out),
            ptr<scalar_t>(db),
            ptr<scalar_t>(db_delta),
            ptr<scalar_t>(workspace));
    }));

    return {dx, dW_x, dr_h, dW_alpha, db_alpha, dW_delta, dW_out, db, db_delta};
}

// =============================================================================
// Level log_3: Log-Storage Diagonal Elman (TRUE LOG-SPACE BACKWARD)
// Stores hidden state as (log|h|, sign(h)) pairs
// Uses softmax weights from logaddexp for bounded gradients!
// =============================================================================

std::vector<Tensor> log_storage_diagonal_forward(
    bool training,
    Tensor x,
    Tensor log_h0,      // [B, dim] log|h0|
    Tensor sign_h0,     // [B, dim] sign(h0)
    Tensor W_x,
    Tensor r_h,         // [dim] diagonal
    Tensor W_delta,
    Tensor W_out,
    Tensor b,
    Tensor b_delta,
    int n_groups) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(log_h0);
    CHECK_INPUT(sign_h0);
    CHECK_INPUT(W_x);
    CHECK_INPUT(r_h);
    CHECK_INPUT(W_delta);
    CHECK_INPUT(W_out);
    CHECK_INPUT(b);
    CHECK_INPUT(b_delta);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor log_h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor sign_h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, dim}, options)
                        : torch::empty({0}, options);
    Tensor delta_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                  : torch::empty({0}, options);
    Tensor compete_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                    : torch::empty({0}, options);
    // NEW: Caches for true log-space backward
    Tensor weight1_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                    : torch::empty({0}, options);
    Tensor log_term1_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                      : torch::empty({0}, options);
    Tensor log_term2_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                      : torch::empty({0}, options);

    log_h[0] = log_h0;
    sign_h[0] = sign_h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "log_storage_diagonal_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        LogStorageDiagonalElmanForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, n_groups,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(r_h),
            ptr<scalar_t>(W_delta),
            ptr<scalar_t>(W_out),
            ptr<scalar_t>(b),
            ptr<scalar_t>(b_delta),
            ptr<scalar_t>(x),
            ptr<scalar_t>(log_h),
            ptr<scalar_t>(sign_h),
            ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v) : nullptr,
            training ? ptr<scalar_t>(delta_cache) : nullptr,
            training ? ptr<scalar_t>(compete_cache) : nullptr,
            training ? ptr<scalar_t>(weight1_cache) : nullptr,
            training ? ptr<scalar_t>(log_term1_cache) : nullptr,
            training ? ptr<scalar_t>(log_term2_cache) : nullptr);
    }));

    return {log_h, sign_h, output, v, delta_cache, compete_cache,
            weight1_cache, log_term1_cache, log_term2_cache};
}

std::vector<Tensor> log_storage_diagonal_backward(
    Tensor W_x,
    Tensor r_h,
    Tensor W_delta,
    Tensor W_out,
    Tensor x,
    Tensor log_h,
    Tensor sign_h,
    Tensor v,
    Tensor delta_cache,
    Tensor compete_cache,
    Tensor weight1_cache,       // NEW: softmax weights for log-space backward
    Tensor log_term1_cache,     // NEW: log|(1-δ)*h_prev|
    Tensor log_term2_cache,     // NEW: log|δ*candidate|
    Tensor d_output,
    int n_groups) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::empty_like(x);
    Tensor dW_x = torch::zeros({dim, dim}, options);
    Tensor dr_h = torch::zeros({dim}, options);
    Tensor dW_delta = torch::zeros({dim, dim}, options);
    Tensor dW_out = torch::zeros({dim, dim}, options);
    Tensor db = torch::zeros({dim}, options);
    Tensor db_delta = torch::zeros({dim}, options);

    // Workspace: (4*T+4)*B*dim T elements + 3*dim floats
    // Layout: [dv_all: TBD][d_delta_all: TBD][d_w_out_h_all: TBD][h_linear_all: TBD]
    //         [dh_linear: BD][dh_recurrent: BD][w_out_h: BD][h_prev_linear: BD]
    //         [dr_h_float: dim floats][db_float: dim floats][db_delta_float: dim floats]
    const int64_t elem_size = x.element_size();
    const int64_t BD = batch_size * dim;
    const int64_t TBD = time_steps * BD;
    const int64_t t_elems = 4 * TBD + 4 * BD;  // T-type elements
    const int64_t float_bytes = 3 * dim * sizeof(float);
    const int64_t float_elems = (float_bytes + elem_size - 1) / elem_size;  // Round up
    Tensor workspace = torch::empty({t_elems + float_elems}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "log_storage_diagonal_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        LogStorageDiagonalElmanBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, n_groups,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(r_h),
            ptr<scalar_t>(W_delta),
            ptr<scalar_t>(W_out),
            ptr<scalar_t>(x),
            ptr<scalar_t>(log_h),
            ptr<scalar_t>(sign_h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(delta_cache),
            ptr<scalar_t>(compete_cache),
            ptr<scalar_t>(weight1_cache),
            ptr<scalar_t>(log_term1_cache),
            ptr<scalar_t>(log_term2_cache),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dW_x),
            ptr<scalar_t>(dr_h),
            ptr<scalar_t>(dW_delta),
            ptr<scalar_t>(dW_out),
            ptr<scalar_t>(db),
            ptr<scalar_t>(db_delta),
            ptr<scalar_t>(workspace));
    }));

    return {dx, dW_x, dr_h, dW_delta, dW_out, db, db_delta};
}

// =============================================================================
// Level 5: Log-Compute Full Elman
// Full R matrix with log-space computation
// =============================================================================

std::vector<Tensor> log_compute_full_forward(
    bool training,
    Tensor x,
    Tensor log_h0,
    Tensor sign_h0,
    Tensor W_x,
    Tensor R_h,         // [dim, dim] full matrix
    Tensor W_delta,
    Tensor W_out,
    Tensor b,
    Tensor b_delta,
    int n_groups) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(log_h0);
    CHECK_INPUT(sign_h0);
    CHECK_INPUT(W_x);
    CHECK_INPUT(R_h);
    CHECK_INPUT(W_delta);
    CHECK_INPUT(W_out);
    CHECK_INPUT(b);
    CHECK_INPUT(b_delta);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor log_h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor sign_h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, dim}, options)
                        : torch::empty({0}, options);
    Tensor delta_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                  : torch::empty({0}, options);
    Tensor compete_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                    : torch::empty({0}, options);
    // Workspace for R decomposition
    Tensor log_R_pos = torch::empty({dim, dim}, options);
    Tensor log_R_neg = torch::empty({dim, dim}, options);

    log_h[0] = log_h0;
    sign_h[0] = sign_h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "log_compute_full_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        LogComputeFullElmanForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, n_groups,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(R_h),
            ptr<scalar_t>(W_delta),
            ptr<scalar_t>(W_out),
            ptr<scalar_t>(b),
            ptr<scalar_t>(b_delta),
            ptr<scalar_t>(x),
            ptr<scalar_t>(log_h),
            ptr<scalar_t>(sign_h),
            ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v) : nullptr,
            training ? ptr<scalar_t>(delta_cache) : nullptr,
            training ? ptr<scalar_t>(compete_cache) : nullptr,
            ptr<scalar_t>(log_R_pos),
            ptr<scalar_t>(log_R_neg));
    }));

    return {log_h, sign_h, output, v, delta_cache, compete_cache};
}

std::vector<Tensor> log_compute_full_backward(
    Tensor W_x,
    Tensor R_h,
    Tensor W_delta,
    Tensor W_out,
    Tensor x,
    Tensor log_h,
    Tensor sign_h,
    Tensor v,
    Tensor delta_cache,
    Tensor compete_cache,
    Tensor d_output,
    int n_groups) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::empty_like(x);
    Tensor dW_x = torch::zeros({dim, dim}, options);
    Tensor dR_h = torch::zeros({dim, dim}, options);
    Tensor dW_delta = torch::zeros({dim, dim}, options);
    Tensor dW_out = torch::zeros({dim, dim}, options);
    Tensor db = torch::zeros({dim}, options);
    Tensor db_delta = torch::zeros({dim}, options);

    // Workspace for R decomposition
    Tensor log_R_pos = torch::empty({dim, dim}, options);
    Tensor log_R_neg = torch::empty({dim, dim}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "log_compute_full_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        LogComputeFullElmanBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, n_groups,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(R_h),
            ptr<scalar_t>(W_delta),
            ptr<scalar_t>(W_out),
            ptr<scalar_t>(x),
            ptr<scalar_t>(log_h),
            ptr<scalar_t>(sign_h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(delta_cache),
            ptr<scalar_t>(compete_cache),
            ptr<scalar_t>(log_R_pos),
            ptr<scalar_t>(log_R_neg),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dW_x),
            ptr<scalar_t>(dR_h),
            ptr<scalar_t>(dW_delta),
            ptr<scalar_t>(dW_out),
            ptr<scalar_t>(db),
            ptr<scalar_t>(db_delta));
    }));

    return {dx, dW_x, dR_h, dW_delta, dW_out, db, db_delta};
}

// =============================================================================
// Level 6: Log-Space Triple R
// Three R matrices with full log-space computation
// =============================================================================

std::vector<Tensor> logspace_triple_r_forward(
    bool training,
    Tensor x,
    Tensor log_h0,
    Tensor sign_h0,
    Tensor R_h,         // [dim, dim] hidden recurrence
    Tensor R_x,         // [dim, dim] input transformation
    Tensor R_delta,     // [dim, dim] delta modulation
    Tensor W_delta,
    Tensor W_out,
    Tensor b,
    Tensor b_delta,
    int n_groups) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(log_h0);
    CHECK_INPUT(sign_h0);
    CHECK_INPUT(R_h);
    CHECK_INPUT(R_x);
    CHECK_INPUT(R_delta);
    CHECK_INPUT(W_delta);
    CHECK_INPUT(W_out);
    CHECK_INPUT(b);
    CHECK_INPUT(b_delta);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor log_h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor sign_h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);
    Tensor v = training ? torch::empty({time_steps, batch_size, dim}, options)
                        : torch::empty({0}, options);
    Tensor delta_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                  : torch::empty({0}, options);
    Tensor compete_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                    : torch::empty({0}, options);

    log_h[0] = log_h0;
    sign_h[0] = sign_h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "logspace_triple_r_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        LogSpaceTripleRForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, n_groups,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(R_h),
            ptr<scalar_t>(R_x),
            ptr<scalar_t>(R_delta),
            ptr<scalar_t>(W_delta),
            ptr<scalar_t>(W_out),
            ptr<scalar_t>(b),
            ptr<scalar_t>(b_delta),
            ptr<scalar_t>(x),
            ptr<scalar_t>(log_h),
            ptr<scalar_t>(sign_h),
            ptr<scalar_t>(output),
            training ? ptr<scalar_t>(v) : nullptr,
            training ? ptr<scalar_t>(delta_cache) : nullptr,
            training ? ptr<scalar_t>(compete_cache) : nullptr);
    }));

    return {log_h, sign_h, output, v, delta_cache, compete_cache};
}

std::vector<Tensor> logspace_triple_r_backward(
    Tensor R_h,
    Tensor R_x,
    Tensor R_delta,
    Tensor W_delta,
    Tensor W_out,
    Tensor x,
    Tensor log_h,
    Tensor sign_h,
    Tensor v,
    Tensor delta_cache,
    Tensor compete_cache,
    Tensor d_output,
    int n_groups) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::empty_like(x);
    Tensor dR_h = torch::zeros({dim, dim}, options);
    Tensor dR_x = torch::zeros({dim, dim}, options);
    Tensor dR_delta = torch::zeros({dim, dim}, options);
    Tensor dW_delta = torch::zeros({dim, dim}, options);
    Tensor dW_out = torch::zeros({dim, dim}, options);
    Tensor db = torch::zeros({dim}, options);
    Tensor db_delta = torch::zeros({dim}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "logspace_triple_r_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        LogSpaceTripleRBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, n_groups,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(R_h),
            ptr<scalar_t>(R_x),
            ptr<scalar_t>(R_delta),
            ptr<scalar_t>(W_delta),
            ptr<scalar_t>(W_out),
            ptr<scalar_t>(x),
            ptr<scalar_t>(log_h),
            ptr<scalar_t>(sign_h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(delta_cache),
            ptr<scalar_t>(compete_cache),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dR_h),
            ptr<scalar_t>(dR_x),
            ptr<scalar_t>(dR_delta),
            ptr<scalar_t>(dW_delta),
            ptr<scalar_t>(dW_out),
            ptr<scalar_t>(db),
            ptr<scalar_t>(db_delta));
    }));

    return {dx, dR_h, dR_x, dR_delta, dW_delta, dW_out, db, db_delta};
}

// =============================================================================
// Log-Space Polynomial (log_0)
// =============================================================================

std::vector<Tensor> logspace_polynomial_forward(
    bool training,
    Tensor x,
    Tensor log_h0,
    Tensor sign_h0,
    Tensor W_x,
    Tensor log_r_h,
    Tensor sign_r_h,
    Tensor W_alpha,
    Tensor b_alpha,
    Tensor W_delta,
    Tensor b,
    Tensor b_delta,
    Tensor log_gamma) {  // [dim] RMSNorm scale in log-space

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(log_h0);
    CHECK_INPUT(sign_h0);
    CHECK_INPUT(W_x);
    CHECK_INPUT(log_r_h);
    CHECK_INPUT(sign_r_h);
    CHECK_INPUT(W_alpha);
    CHECK_INPUT(b_alpha);
    CHECK_INPUT(W_delta);
    CHECK_INPUT(b);
    CHECK_INPUT(b_delta);
    CHECK_INPUT(log_gamma);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor log_h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor sign_h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor h_linear = torch::empty({time_steps, batch_size, dim}, options);

    // Caches for backward
    Tensor log_v_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                  : torch::empty({0}, options);
    Tensor sign_v_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                   : torch::empty({0}, options);
    Tensor alpha_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                  : torch::empty({0}, options);
    Tensor log_h_unbounded_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                            : torch::empty({0}, options);
    Tensor delta_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                  : torch::empty({0}, options);
    Tensor weight_rh_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                      : torch::empty({0}, options);
    Tensor alpha_raw_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                      : torch::empty({0}, options);
    Tensor log_rms_cache = training ? torch::empty({time_steps, batch_size}, options)
                                    : torch::empty({0}, options);

    log_h[0] = log_h0;
    sign_h[0] = sign_h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "logspace_polynomial_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        LogPolyElmanForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(log_r_h),
            ptr<scalar_t>(sign_r_h),
            ptr<scalar_t>(W_alpha),
            ptr<scalar_t>(b_alpha),
            ptr<scalar_t>(W_delta),
            ptr<scalar_t>(b),
            ptr<scalar_t>(b_delta),
            ptr<scalar_t>(log_gamma),
            ptr<scalar_t>(x),
            ptr<scalar_t>(log_h),
            ptr<scalar_t>(sign_h),
            ptr<scalar_t>(h_linear),
            training ? ptr<scalar_t>(log_v_cache) : nullptr,
            training ? ptr<scalar_t>(sign_v_cache) : nullptr,
            training ? ptr<scalar_t>(alpha_cache) : nullptr,
            training ? ptr<scalar_t>(log_h_unbounded_cache) : nullptr,
            training ? ptr<scalar_t>(delta_cache) : nullptr,
            training ? ptr<scalar_t>(weight_rh_cache) : nullptr,
            training ? ptr<scalar_t>(alpha_raw_cache) : nullptr,
            training ? ptr<scalar_t>(log_rms_cache) : nullptr);
    }));

    return {log_h, sign_h, h_linear, log_v_cache, sign_v_cache, alpha_cache,
            log_h_unbounded_cache, delta_cache, weight_rh_cache, alpha_raw_cache, log_rms_cache};
}

std::vector<Tensor> logspace_polynomial_backward(
    Tensor W_x,
    Tensor log_r_h,
    Tensor sign_r_h,
    Tensor W_alpha,
    Tensor W_delta,
    Tensor log_gamma,       // [dim] RMSNorm scale
    Tensor x,
    Tensor log_h,
    Tensor sign_h,
    Tensor log_v_cache,
    Tensor sign_v_cache,
    Tensor alpha_cache,
    Tensor alpha_raw_cache,
    Tensor log_h_unbounded_cache,
    Tensor delta_cache,
    Tensor weight_rh_cache,
    Tensor log_rms_cache,   // [T, B] cached from forward
    Tensor d_h_linear) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::empty_like(x);
    Tensor dW_x = torch::zeros({dim, dim}, options);
    Tensor d_log_r_h = torch::zeros({dim}, options);
    Tensor dW_alpha = torch::zeros({dim, dim}, options);
    Tensor db_alpha = torch::zeros({dim}, options);
    Tensor dW_delta = torch::zeros({dim, dim}, options);
    Tensor db = torch::zeros({dim}, options);
    Tensor db_delta = torch::zeros({dim}, options);
    Tensor d_log_gamma = torch::zeros({dim}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "logspace_polynomial_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        LogPolyElmanBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(log_r_h),
            ptr<scalar_t>(sign_r_h),
            ptr<scalar_t>(W_alpha),
            ptr<scalar_t>(W_delta),
            ptr<scalar_t>(log_gamma),
            ptr<scalar_t>(x),
            ptr<scalar_t>(log_h),
            ptr<scalar_t>(sign_h),
            ptr<scalar_t>(log_v_cache),
            ptr<scalar_t>(sign_v_cache),
            ptr<scalar_t>(alpha_cache),
            ptr<scalar_t>(alpha_raw_cache),
            ptr<scalar_t>(log_h_unbounded_cache),
            ptr<scalar_t>(delta_cache),
            ptr<scalar_t>(weight_rh_cache),
            ptr<scalar_t>(log_rms_cache),
            ptr<scalar_t>(d_h_linear),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dW_x),
            ptr<scalar_t>(d_log_r_h),
            ptr<scalar_t>(dW_alpha),
            ptr<scalar_t>(db_alpha),
            ptr<scalar_t>(dW_delta),
            ptr<scalar_t>(db),
            ptr<scalar_t>(db_delta),
            ptr<scalar_t>(d_log_gamma));
    }));

    return {dx, dW_x, d_log_r_h, dW_alpha, db_alpha, dW_delta, db, db_delta, d_log_gamma};
}

// =============================================================================
// Log-Space Selective (log_1)
// =============================================================================

std::vector<Tensor> logspace_selective_forward(
    bool training,
    Tensor x,
    Tensor log_h0,
    Tensor sign_h0,
    Tensor W_x,
    Tensor log_r_h,
    Tensor sign_r_h,
    Tensor W_alpha,
    Tensor b_alpha,
    Tensor W_delta,
    Tensor W_out,
    Tensor b,
    Tensor b_delta,
    Tensor log_gamma,    // NEW: RMSNorm scale in log-space
    int n_groups) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(log_h0);
    CHECK_INPUT(sign_h0);
    CHECK_INPUT(W_x);
    CHECK_INPUT(log_r_h);
    CHECK_INPUT(sign_r_h);
    CHECK_INPUT(W_alpha);
    CHECK_INPUT(b_alpha);
    CHECK_INPUT(W_delta);
    CHECK_INPUT(W_out);
    CHECK_INPUT(b);
    CHECK_INPUT(b_delta);
    CHECK_INPUT(log_gamma);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor log_h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor sign_h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);

    // Caches
    Tensor log_v_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                  : torch::empty({0}, options);
    Tensor sign_v_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                   : torch::empty({0}, options);
    Tensor alpha_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                  : torch::empty({0}, options);
    Tensor log_h_unbounded_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                            : torch::empty({0}, options);
    Tensor delta_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                  : torch::empty({0}, options);
    Tensor weight_rh_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                      : torch::empty({0}, options);
    Tensor alpha_raw_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                      : torch::empty({0}, options);
    Tensor h_linear_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                     : torch::empty({0}, options);
    Tensor compete_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                    : torch::empty({0}, options);
    Tensor log_rms_cache = training ? torch::empty({time_steps, batch_size}, options)
                                    : torch::empty({0}, options);  // NEW: for RMSNorm backward

    log_h[0] = log_h0;
    sign_h[0] = sign_h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "logspace_selective_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        LogSelectiveElmanForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, n_groups,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(log_r_h),
            ptr<scalar_t>(sign_r_h),
            ptr<scalar_t>(W_alpha),
            ptr<scalar_t>(b_alpha),
            ptr<scalar_t>(W_delta),
            ptr<scalar_t>(W_out),
            ptr<scalar_t>(b),
            ptr<scalar_t>(b_delta),
            ptr<scalar_t>(log_gamma),
            ptr<scalar_t>(x),
            ptr<scalar_t>(log_h),
            ptr<scalar_t>(sign_h),
            ptr<scalar_t>(output),
            training ? ptr<scalar_t>(log_v_cache) : nullptr,
            training ? ptr<scalar_t>(sign_v_cache) : nullptr,
            training ? ptr<scalar_t>(alpha_cache) : nullptr,
            training ? ptr<scalar_t>(log_h_unbounded_cache) : nullptr,
            training ? ptr<scalar_t>(delta_cache) : nullptr,
            training ? ptr<scalar_t>(weight_rh_cache) : nullptr,
            training ? ptr<scalar_t>(alpha_raw_cache) : nullptr,
            training ? ptr<scalar_t>(h_linear_cache) : nullptr,
            training ? ptr<scalar_t>(compete_cache) : nullptr,
            training ? ptr<scalar_t>(log_rms_cache) : nullptr);
    }));

    return {log_h, sign_h, output, log_v_cache, sign_v_cache, alpha_cache,
            log_h_unbounded_cache, delta_cache, weight_rh_cache, alpha_raw_cache,
            h_linear_cache, compete_cache, log_rms_cache};
}

std::vector<Tensor> logspace_selective_backward(
    Tensor W_x,
    Tensor log_r_h,
    Tensor sign_r_h,
    Tensor W_alpha,
    Tensor W_delta,
    Tensor W_out,
    Tensor log_gamma,        // NEW: RMSNorm scale
    Tensor x,
    Tensor log_h,
    Tensor sign_h,
    Tensor log_v_cache,
    Tensor sign_v_cache,
    Tensor alpha_cache,
    Tensor alpha_raw_cache,
    Tensor log_h_unbounded_cache,
    Tensor delta_cache,
    Tensor weight_rh_cache,
    Tensor h_linear_cache,
    Tensor compete_cache,
    Tensor log_rms_cache,    // NEW: cached log_rms from forward
    Tensor d_output,
    int n_groups) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::empty_like(x);
    Tensor dW_x = torch::zeros({dim, dim}, options);
    Tensor d_log_r_h = torch::zeros({dim}, options);
    Tensor dW_alpha = torch::zeros({dim, dim}, options);
    Tensor db_alpha = torch::zeros({dim}, options);
    Tensor dW_delta = torch::zeros({dim, dim}, options);
    Tensor dW_out = torch::zeros({dim, dim}, options);
    Tensor db = torch::zeros({dim}, options);
    Tensor db_delta = torch::zeros({dim}, options);
    Tensor d_log_gamma = torch::zeros({dim}, options);  // NEW

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "logspace_selective_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        LogSelectiveElmanBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, n_groups,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(log_r_h),
            ptr<scalar_t>(sign_r_h),
            ptr<scalar_t>(W_alpha),
            ptr<scalar_t>(W_delta),
            ptr<scalar_t>(W_out),
            ptr<scalar_t>(log_gamma),
            ptr<scalar_t>(x),
            ptr<scalar_t>(log_h),
            ptr<scalar_t>(sign_h),
            ptr<scalar_t>(log_v_cache),
            ptr<scalar_t>(sign_v_cache),
            ptr<scalar_t>(alpha_cache),
            ptr<scalar_t>(alpha_raw_cache),
            ptr<scalar_t>(log_h_unbounded_cache),
            ptr<scalar_t>(delta_cache),
            ptr<scalar_t>(weight_rh_cache),
            ptr<scalar_t>(h_linear_cache),
            ptr<scalar_t>(compete_cache),
            ptr<scalar_t>(log_rms_cache),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dW_x),
            ptr<scalar_t>(d_log_r_h),
            ptr<scalar_t>(dW_alpha),
            ptr<scalar_t>(db_alpha),
            ptr<scalar_t>(dW_delta),
            ptr<scalar_t>(dW_out),
            ptr<scalar_t>(db),
            ptr<scalar_t>(db_delta),
            ptr<scalar_t>(d_log_gamma));
    }));

    return {dx, dW_x, d_log_r_h, dW_alpha, db_alpha, dW_delta, dW_out, db, db_delta, d_log_gamma};
}

// =============================================================================
// Log-Space Diagonal Selective (log_2)
// =============================================================================

std::vector<Tensor> logspace_diag_selective_forward(
    bool training,
    Tensor x,
    Tensor log_h0,
    Tensor sign_h0,
    Tensor W_x,
    Tensor log_r_h,
    Tensor sign_r_h,
    Tensor W_alpha,
    Tensor b_alpha,
    Tensor W_delta,
    Tensor W_out,
    Tensor b,
    Tensor b_delta,
    Tensor log_gamma,    // NEW: RMSNorm scale in log-space
    int n_groups) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(log_h0);
    CHECK_INPUT(sign_h0);
    CHECK_INPUT(W_x);
    CHECK_INPUT(log_r_h);
    CHECK_INPUT(sign_r_h);
    CHECK_INPUT(W_alpha);
    CHECK_INPUT(b_alpha);
    CHECK_INPUT(W_delta);
    CHECK_INPUT(W_out);
    CHECK_INPUT(b);
    CHECK_INPUT(b_delta);
    CHECK_INPUT(log_gamma);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor log_h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor sign_h = torch::empty({time_steps + 1, batch_size, dim}, options);
    Tensor output = torch::empty({time_steps, batch_size, dim}, options);

    // Caches
    Tensor log_v_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                  : torch::empty({0}, options);
    Tensor sign_v_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                   : torch::empty({0}, options);
    Tensor alpha_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                  : torch::empty({0}, options);
    Tensor log_h_unbounded_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                            : torch::empty({0}, options);
    Tensor delta_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                  : torch::empty({0}, options);
    Tensor weight_rh_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                      : torch::empty({0}, options);
    Tensor alpha_raw_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                      : torch::empty({0}, options);
    Tensor h_linear_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                     : torch::empty({0}, options);
    Tensor compete_cache = training ? torch::empty({time_steps, batch_size, dim}, options)
                                    : torch::empty({0}, options);
    Tensor log_rms_cache = training ? torch::empty({time_steps, batch_size}, options)
                                    : torch::empty({0}, options);  // NEW: for RMSNorm backward

    log_h[0] = log_h0;
    sign_h[0] = sign_h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "logspace_diag_selective_forward", ([&] {
        using namespace hasty::v0::elman_ladder;
        LogDiagSelectiveElmanForward<typename native_type<scalar_t>::T> forward(
            training, batch_size, dim, n_groups,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(log_r_h),
            ptr<scalar_t>(sign_r_h),
            ptr<scalar_t>(W_alpha),
            ptr<scalar_t>(b_alpha),
            ptr<scalar_t>(W_delta),
            ptr<scalar_t>(W_out),
            ptr<scalar_t>(b),
            ptr<scalar_t>(b_delta),
            ptr<scalar_t>(log_gamma),
            ptr<scalar_t>(x),
            ptr<scalar_t>(log_h),
            ptr<scalar_t>(sign_h),
            ptr<scalar_t>(output),
            training ? ptr<scalar_t>(log_v_cache) : nullptr,
            training ? ptr<scalar_t>(sign_v_cache) : nullptr,
            training ? ptr<scalar_t>(alpha_cache) : nullptr,
            training ? ptr<scalar_t>(log_h_unbounded_cache) : nullptr,
            training ? ptr<scalar_t>(delta_cache) : nullptr,
            training ? ptr<scalar_t>(weight_rh_cache) : nullptr,
            training ? ptr<scalar_t>(alpha_raw_cache) : nullptr,
            training ? ptr<scalar_t>(h_linear_cache) : nullptr,
            training ? ptr<scalar_t>(compete_cache) : nullptr,
            training ? ptr<scalar_t>(log_rms_cache) : nullptr);
    }));

    return {log_h, sign_h, output, log_v_cache, sign_v_cache, alpha_cache,
            log_h_unbounded_cache, delta_cache, weight_rh_cache, alpha_raw_cache,
            h_linear_cache, compete_cache, log_rms_cache};
}

std::vector<Tensor> logspace_diag_selective_backward(
    Tensor W_x,
    Tensor log_r_h,
    Tensor sign_r_h,
    Tensor W_alpha,
    Tensor W_delta,
    Tensor W_out,
    Tensor log_gamma,        // NEW: RMSNorm scale
    Tensor x,
    Tensor log_h,
    Tensor sign_h,
    Tensor log_v_cache,
    Tensor sign_v_cache,
    Tensor alpha_cache,
    Tensor alpha_raw_cache,
    Tensor log_h_unbounded_cache,
    Tensor delta_cache,
    Tensor weight_rh_cache,
    Tensor h_linear_cache,
    Tensor compete_cache,
    Tensor log_rms_cache,    // NEW: cached log_rms from forward
    Tensor d_output,
    int n_groups) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto dim = x.size(2);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::empty_like(x);
    Tensor dW_x = torch::zeros({dim, dim}, options);
    Tensor d_log_r_h = torch::zeros({dim}, options);
    Tensor dW_alpha = torch::zeros({dim, dim}, options);
    Tensor db_alpha = torch::zeros({dim}, options);
    Tensor dW_delta = torch::zeros({dim, dim}, options);
    Tensor dW_out = torch::zeros({dim, dim}, options);
    Tensor db = torch::zeros({dim}, options);
    Tensor db_delta = torch::zeros({dim}, options);
    Tensor d_log_gamma = torch::zeros({dim}, options);  // NEW

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "logspace_diag_selective_backward", ([&] {
        using namespace hasty::v0::elman_ladder;
        LogDiagSelectiveElmanBackward<typename native_type<scalar_t>::T> backward(
            batch_size, dim, n_groups,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(log_r_h),
            ptr<scalar_t>(sign_r_h),
            ptr<scalar_t>(W_alpha),
            ptr<scalar_t>(W_delta),
            ptr<scalar_t>(W_out),
            ptr<scalar_t>(log_gamma),
            ptr<scalar_t>(x),
            ptr<scalar_t>(log_h),
            ptr<scalar_t>(sign_h),
            ptr<scalar_t>(log_v_cache),
            ptr<scalar_t>(sign_v_cache),
            ptr<scalar_t>(alpha_cache),
            ptr<scalar_t>(alpha_raw_cache),
            ptr<scalar_t>(log_h_unbounded_cache),
            ptr<scalar_t>(delta_cache),
            ptr<scalar_t>(weight_rh_cache),
            ptr<scalar_t>(h_linear_cache),
            ptr<scalar_t>(compete_cache),
            ptr<scalar_t>(log_rms_cache),
            ptr<scalar_t>(d_output),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dW_x),
            ptr<scalar_t>(d_log_r_h),
            ptr<scalar_t>(dW_alpha),
            ptr<scalar_t>(db_alpha),
            ptr<scalar_t>(dW_delta),
            ptr<scalar_t>(dW_out),
            ptr<scalar_t>(db),
            ptr<scalar_t>(db_delta),
            ptr<scalar_t>(d_log_gamma));
    }));

    return {dx, dW_x, d_log_r_h, dW_alpha, db_alpha, dW_delta, dW_out, db, db_delta, d_log_gamma};
}

}  // anonymous namespace


void elman_ladder_init(py::module& m) {
    m.def("stock_elman_forward", &stock_elman_forward,
          "Level 0: Stock Elman forward");
    m.def("stock_elman_backward", &stock_elman_backward,
          "Level 0: Stock Elman backward");

    m.def("gated_elman_forward", &gated_elman_forward,
          "Level 1: Gated Elman forward");
    m.def("gated_elman_backward", &gated_elman_backward,
          "Level 1: Gated Elman backward");

    m.def("selective_elman_forward", &selective_elman_forward,
          "Level 2: Selective Elman forward");
    m.def("selective_elman_backward", &selective_elman_backward,
          "Level 2: Selective Elman backward");

    m.def("diagonal_selective_forward", &diagonal_selective_forward,
          "Level 3: Diagonal Selective forward");
    m.def("diagonal_selective_backward", &diagonal_selective_backward,
          "Level 3: Diagonal Selective backward");

    m.def("full_recurrence_forward", &full_recurrence_forward,
          "Level 4: Full Recurrence forward");
    m.def("full_recurrence_backward", &full_recurrence_backward,
          "Level 4: Full Recurrence backward");

    m.def("linear_triple_r_forward", &linear_triple_r_forward,
          "Level 5: Linear Triple R forward");
    m.def("linear_triple_r_backward", &linear_triple_r_backward,
          "Level 5: Linear Triple R backward");

    m.def("linear_polynomial_forward", &linear_polynomial_forward,
          "Level 6: Linear Polynomial forward");
    m.def("linear_polynomial_backward", &linear_polynomial_backward,
          "Level 6: Linear Polynomial backward");

    m.def("log_storage_diagonal_forward", &log_storage_diagonal_forward,
          "Log-Space Level 3: Log-Storage Diagonal forward");
    m.def("log_storage_diagonal_backward", &log_storage_diagonal_backward,
          "Level 4: Log-Storage Diagonal backward");

    m.def("log_compute_full_forward", &log_compute_full_forward,
          "Level 5: Log-Compute Full forward");
    m.def("log_compute_full_backward", &log_compute_full_backward,
          "Level 5: Log-Compute Full backward");

    m.def("logspace_triple_r_forward", &logspace_triple_r_forward,
          "Level 6: Log-Space Triple R forward");
    m.def("logspace_triple_r_backward", &logspace_triple_r_backward,
          "Level 6: Log-Space Triple R backward");

    // Log-Space Polynomial Levels
    m.def("logspace_polynomial_forward", &logspace_polynomial_forward,
          "Log-Space Level 0: Polynomial forward");
    m.def("logspace_polynomial_backward", &logspace_polynomial_backward,
          "Log-Space Level 0: Polynomial backward");

    m.def("logspace_selective_forward", &logspace_selective_forward,
          "Log-Space Level 1: Selective forward");
    m.def("logspace_selective_backward", &logspace_selective_backward,
          "Log-Space Level 1: Selective backward");

    m.def("logspace_diag_selective_forward", &logspace_diag_selective_forward,
          "Log-Space Level 2: Diagonal Selective forward");
    m.def("logspace_diag_selective_backward", &logspace_diag_selective_backward,
          "Log-Space Level 2: Diagonal Selective backward");
}
