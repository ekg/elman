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
            training ? ptr<scalar_t>(gate_cache) : nullptr);
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
    Tensor workspace = torch::empty({batch_size, dim}, options);

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

}  // anonymous namespace


void elman_ladder_init(py::module& m) {
    m.def("stock_elman_forward", &stock_elman_forward,
          "E0: Stock Elman forward");
    m.def("stock_elman_backward", &stock_elman_backward,
          "E0: Stock Elman backward");
}
