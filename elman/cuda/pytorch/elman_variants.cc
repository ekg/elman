// Copyright 2024 Erik Garrison. Apache 2.0 License.
//
// PyTorch bindings for Elman RNN variants.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>

#include "hasty.h"
#include "support.h"

namespace {

using torch::Tensor;

// ============================================================================
// ElmanTanh: tanh + tanh gate
// ============================================================================

template<typename T>
using ElmanTanhForward = hasty::v0::elman_tanh::ForwardPass<T>;
template<typename T>
using ElmanTanhBackward = hasty::v0::elman_tanh::BackwardPass<T>;

std::vector<Tensor> elman_tanh_forward(
    bool training,
    Tensor x,
    Tensor h0,
    Tensor Wx,
    Tensor R,
    Tensor bias) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto D = x.size(2);
    const auto gate_dim = 2 * D;

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(Wx);
    CHECK_INPUT(R);
    CHECK_INPUT(bias);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor x_flat = x.reshape({time_steps * batch_size, D});
    Tensor Wx_all = torch::matmul(x_flat, Wx.t());
    Wx_all = Wx_all.reshape({time_steps, batch_size, gate_dim});

    Tensor h = torch::empty({ time_steps + 1, batch_size, D }, options);
    Tensor v = training ? torch::empty({ time_steps, batch_size, gate_dim }, options)
                        : torch::empty({ 0 }, options);
    Tensor tmp_Rh = torch::empty({ batch_size, gate_dim }, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_tanh_forward", ([&] {
        ElmanTanhForward<typename native_type<scalar_t>::T> forward(
            training,
            batch_size,
            D,
            D,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(R),
            ptr<scalar_t>(bias),
            ptr<scalar_t>(Wx_all),
            ptr<scalar_t>(h),
            training ? ptr<scalar_t>(v) : nullptr,
            ptr<scalar_t>(tmp_Rh));
    }));

    return { h, v };
}

std::vector<Tensor> elman_tanh_backward(
    Tensor x,
    Tensor Wx,
    Tensor R,
    Tensor h,
    Tensor v,
    Tensor dh_new) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto D = x.size(2);
    const auto gate_dim = 2 * D;

    CHECK_INPUT(x);
    CHECK_INPUT(Wx);
    CHECK_INPUT(R);
    CHECK_INPUT(h);
    CHECK_INPUT(v);
    CHECK_INPUT(dh_new);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dWx_all = torch::empty({ time_steps, batch_size, gate_dim }, options);
    Tensor dR = torch::zeros({ gate_dim, D }, options);
    Tensor dbias = torch::zeros({ gate_dim }, options);
    Tensor dh0 = torch::zeros({ batch_size, D }, options);
    Tensor tmp_dRh = torch::empty({ batch_size, gate_dim }, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_tanh_backward", ([&] {
        ElmanTanhBackward<typename native_type<scalar_t>::T> backward(
            batch_size,
            D,
            D,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(R),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(dh_new),
            ptr<scalar_t>(dWx_all),
            ptr<scalar_t>(dR),
            ptr<scalar_t>(dbias),
            ptr<scalar_t>(dh0),
            ptr<scalar_t>(tmp_dRh));
    }));

    Tensor dWx_flat = dWx_all.reshape({time_steps * batch_size, gate_dim});
    Tensor x_flat = x.reshape({time_steps * batch_size, D});
    Tensor dWx = torch::matmul(dWx_flat.t(), x_flat);
    Tensor dx_flat = torch::matmul(dWx_flat, Wx);
    Tensor dx = dx_flat.reshape({time_steps, batch_size, D});

    return { dx, dh0, dWx, dR, dbias };
}

// ============================================================================
// ElmanSigmoid: tanh + sigmoid gate
// ============================================================================

template<typename T>
using ElmanSigmoidForward = hasty::v0::elman_sigmoid::ForwardPass<T>;
template<typename T>
using ElmanSigmoidBackward = hasty::v0::elman_sigmoid::BackwardPass<T>;

std::vector<Tensor> elman_sigmoid_forward(
    bool training,
    Tensor x,
    Tensor h0,
    Tensor Wx,
    Tensor R,
    Tensor bias) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto D = x.size(2);
    const auto gate_dim = 2 * D;

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(Wx);
    CHECK_INPUT(R);
    CHECK_INPUT(bias);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor x_flat = x.reshape({time_steps * batch_size, D});
    Tensor Wx_all = torch::matmul(x_flat, Wx.t());
    Wx_all = Wx_all.reshape({time_steps, batch_size, gate_dim});

    Tensor h = torch::empty({ time_steps + 1, batch_size, D }, options);
    Tensor v = training ? torch::empty({ time_steps, batch_size, gate_dim }, options)
                        : torch::empty({ 0 }, options);
    Tensor tmp_Rh = torch::empty({ batch_size, gate_dim }, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_sigmoid_forward", ([&] {
        ElmanSigmoidForward<typename native_type<scalar_t>::T> forward(
            training,
            batch_size,
            D,
            D,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(R),
            ptr<scalar_t>(bias),
            ptr<scalar_t>(Wx_all),
            ptr<scalar_t>(h),
            training ? ptr<scalar_t>(v) : nullptr,
            ptr<scalar_t>(tmp_Rh));
    }));

    return { h, v };
}

std::vector<Tensor> elman_sigmoid_backward(
    Tensor x,
    Tensor Wx,
    Tensor R,
    Tensor h,
    Tensor v,
    Tensor dh_new) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto D = x.size(2);
    const auto gate_dim = 2 * D;

    CHECK_INPUT(x);
    CHECK_INPUT(Wx);
    CHECK_INPUT(R);
    CHECK_INPUT(h);
    CHECK_INPUT(v);
    CHECK_INPUT(dh_new);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dWx_all = torch::empty({ time_steps, batch_size, gate_dim }, options);
    Tensor dR = torch::zeros({ gate_dim, D }, options);
    Tensor dbias = torch::zeros({ gate_dim }, options);
    Tensor dh0 = torch::zeros({ batch_size, D }, options);
    Tensor tmp_dRh = torch::empty({ batch_size, gate_dim }, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_sigmoid_backward", ([&] {
        ElmanSigmoidBackward<typename native_type<scalar_t>::T> backward(
            batch_size,
            D,
            D,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(R),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(dh_new),
            ptr<scalar_t>(dWx_all),
            ptr<scalar_t>(dR),
            ptr<scalar_t>(dbias),
            ptr<scalar_t>(dh0),
            ptr<scalar_t>(tmp_dRh));
    }));

    Tensor dWx_flat = dWx_all.reshape({time_steps * batch_size, gate_dim});
    Tensor x_flat = x.reshape({time_steps * batch_size, D});
    Tensor dWx = torch::matmul(dWx_flat.t(), x_flat);
    Tensor dx_flat = torch::matmul(dWx_flat, Wx);
    Tensor dx = dx_flat.reshape({time_steps, batch_size, D});

    return { dx, dh0, dWx, dR, dbias };
}

// ============================================================================
// ElmanSwish: silu + silu gate
// ============================================================================

template<typename T>
using ElmanSwishForward = hasty::v0::elman_swish::ForwardPass<T>;
template<typename T>
using ElmanSwishBackward = hasty::v0::elman_swish::BackwardPass<T>;

std::vector<Tensor> elman_swish_forward(
    bool training,
    Tensor x,
    Tensor h0,
    Tensor Wx,
    Tensor R,
    Tensor bias) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto D = x.size(2);
    const auto gate_dim = 2 * D;

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(Wx);
    CHECK_INPUT(R);
    CHECK_INPUT(bias);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor x_flat = x.reshape({time_steps * batch_size, D});
    Tensor Wx_all = torch::matmul(x_flat, Wx.t());
    Wx_all = Wx_all.reshape({time_steps, batch_size, gate_dim});

    Tensor h = torch::empty({ time_steps + 1, batch_size, D }, options);
    Tensor v = training ? torch::empty({ time_steps, batch_size, gate_dim }, options)
                        : torch::empty({ 0 }, options);
    Tensor tmp_Rh = torch::empty({ batch_size, gate_dim }, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_swish_forward", ([&] {
        ElmanSwishForward<typename native_type<scalar_t>::T> forward(
            training,
            batch_size,
            D,
            D,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(R),
            ptr<scalar_t>(bias),
            ptr<scalar_t>(Wx_all),
            ptr<scalar_t>(h),
            training ? ptr<scalar_t>(v) : nullptr,
            ptr<scalar_t>(tmp_Rh));
    }));

    return { h, v };
}

std::vector<Tensor> elman_swish_backward(
    Tensor x,
    Tensor Wx,
    Tensor R,
    Tensor h,
    Tensor v,
    Tensor dh_new) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto D = x.size(2);
    const auto gate_dim = 2 * D;

    CHECK_INPUT(x);
    CHECK_INPUT(Wx);
    CHECK_INPUT(R);
    CHECK_INPUT(h);
    CHECK_INPUT(v);
    CHECK_INPUT(dh_new);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dWx_all = torch::empty({ time_steps, batch_size, gate_dim }, options);
    Tensor dR = torch::zeros({ gate_dim, D }, options);
    Tensor dbias = torch::zeros({ gate_dim }, options);
    Tensor dh0 = torch::zeros({ batch_size, D }, options);
    Tensor tmp_dRh = torch::empty({ batch_size, gate_dim }, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_swish_backward", ([&] {
        ElmanSwishBackward<typename native_type<scalar_t>::T> backward(
            batch_size,
            D,
            D,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(R),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(dh_new),
            ptr<scalar_t>(dWx_all),
            ptr<scalar_t>(dR),
            ptr<scalar_t>(dbias),
            ptr<scalar_t>(dh0),
            ptr<scalar_t>(tmp_dRh));
    }));

    Tensor dWx_flat = dWx_all.reshape({time_steps * batch_size, gate_dim});
    Tensor x_flat = x.reshape({time_steps * batch_size, D});
    Tensor dWx = torch::matmul(dWx_flat.t(), x_flat);
    Tensor dx_flat = torch::matmul(dWx_flat, Wx);
    Tensor dx = dx_flat.reshape({time_steps, batch_size, D});

    return { dx, dh0, dWx, dR, dbias };
}

// ============================================================================
// ElmanGelu: tanh + gelu gate
// ============================================================================

template<typename T>
using ElmanGeluForward = hasty::v0::elman_gelu::ForwardPass<T>;
template<typename T>
using ElmanGeluBackward = hasty::v0::elman_gelu::BackwardPass<T>;

std::vector<Tensor> elman_gelu_forward(
    bool training,
    Tensor x,
    Tensor h0,
    Tensor Wx,
    Tensor R,
    Tensor bias) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto D = x.size(2);
    const auto gate_dim = 2 * D;

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(Wx);
    CHECK_INPUT(R);
    CHECK_INPUT(bias);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor x_flat = x.reshape({time_steps * batch_size, D});
    Tensor Wx_all = torch::matmul(x_flat, Wx.t());
    Wx_all = Wx_all.reshape({time_steps, batch_size, gate_dim});

    Tensor h = torch::empty({ time_steps + 1, batch_size, D }, options);
    Tensor v = training ? torch::empty({ time_steps, batch_size, gate_dim }, options)
                        : torch::empty({ 0 }, options);
    Tensor tmp_Rh = torch::empty({ batch_size, gate_dim }, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_gelu_forward", ([&] {
        ElmanGeluForward<typename native_type<scalar_t>::T> forward(
            training,
            batch_size,
            D,
            D,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(R),
            ptr<scalar_t>(bias),
            ptr<scalar_t>(Wx_all),
            ptr<scalar_t>(h),
            training ? ptr<scalar_t>(v) : nullptr,
            ptr<scalar_t>(tmp_Rh));
    }));

    return { h, v };
}

std::vector<Tensor> elman_gelu_backward(
    Tensor x,
    Tensor Wx,
    Tensor R,
    Tensor h,
    Tensor v,
    Tensor dh_new) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto D = x.size(2);
    const auto gate_dim = 2 * D;

    CHECK_INPUT(x);
    CHECK_INPUT(Wx);
    CHECK_INPUT(R);
    CHECK_INPUT(h);
    CHECK_INPUT(v);
    CHECK_INPUT(dh_new);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dWx_all = torch::empty({ time_steps, batch_size, gate_dim }, options);
    Tensor dR = torch::zeros({ gate_dim, D }, options);
    Tensor dbias = torch::zeros({ gate_dim }, options);
    Tensor dh0 = torch::zeros({ batch_size, D }, options);
    Tensor tmp_dRh = torch::empty({ batch_size, gate_dim }, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_gelu_backward", ([&] {
        ElmanGeluBackward<typename native_type<scalar_t>::T> backward(
            batch_size,
            D,
            D,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(R),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(dh_new),
            ptr<scalar_t>(dWx_all),
            ptr<scalar_t>(dR),
            ptr<scalar_t>(dbias),
            ptr<scalar_t>(dh0),
            ptr<scalar_t>(tmp_dRh));
    }));

    Tensor dWx_flat = dWx_all.reshape({time_steps * batch_size, gate_dim});
    Tensor x_flat = x.reshape({time_steps * batch_size, D});
    Tensor dWx = torch::matmul(dWx_flat.t(), x_flat);
    Tensor dx_flat = torch::matmul(dWx_flat, Wx);
    Tensor dx = dx_flat.reshape({time_steps, batch_size, D});

    return { dx, dh0, dWx, dR, dbias };
}

// ============================================================================
// ElmanNoGate: tanh only, no gating (ablation baseline)
// ============================================================================

template<typename T>
using ElmanNoGateForward = hasty::v0::elman_nogate::ForwardPass<T>;
template<typename T>
using ElmanNoGateBackward = hasty::v0::elman_nogate::BackwardPass<T>;

std::vector<Tensor> elman_nogate_forward(
    bool training,
    Tensor x,
    Tensor h0,
    Tensor Wx,         // [D, D] - no gate dimension
    Tensor R,          // [D, D]
    Tensor bias) {     // [D]
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto D = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(Wx);
    CHECK_INPUT(R);
    CHECK_INPUT(bias);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor x_flat = x.reshape({time_steps * batch_size, D});
    Tensor Wx_all = torch::matmul(x_flat, Wx.t());
    Wx_all = Wx_all.reshape({time_steps, batch_size, D});

    Tensor h = torch::empty({ time_steps + 1, batch_size, D }, options);
    Tensor v = training ? torch::empty({ time_steps, batch_size, D }, options)
                        : torch::empty({ 0 }, options);
    Tensor tmp_Rh = torch::empty({ batch_size, D }, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_nogate_forward", ([&] {
        ElmanNoGateForward<typename native_type<scalar_t>::T> forward(
            training,
            batch_size,
            D,
            D,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(R),
            ptr<scalar_t>(bias),
            ptr<scalar_t>(Wx_all),
            ptr<scalar_t>(h),
            training ? ptr<scalar_t>(v) : nullptr,
            ptr<scalar_t>(tmp_Rh));
    }));

    return { h, v };
}

std::vector<Tensor> elman_nogate_backward(
    Tensor x,
    Tensor Wx,
    Tensor R,
    Tensor h,
    Tensor v,
    Tensor dh_new) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto D = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(Wx);
    CHECK_INPUT(R);
    CHECK_INPUT(h);
    CHECK_INPUT(v);
    CHECK_INPUT(dh_new);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dWx_all = torch::empty({ time_steps, batch_size, D }, options);
    Tensor dR = torch::zeros({ D, D }, options);
    Tensor dbias = torch::zeros({ D }, options);
    Tensor dh0 = torch::zeros({ batch_size, D }, options);
    Tensor tmp_dRh = torch::empty({ batch_size, D }, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_nogate_backward", ([&] {
        ElmanNoGateBackward<typename native_type<scalar_t>::T> backward(
            batch_size,
            D,
            D,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(R),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(dh_new),
            ptr<scalar_t>(dWx_all),
            ptr<scalar_t>(dR),
            ptr<scalar_t>(dbias),
            ptr<scalar_t>(dh0),
            ptr<scalar_t>(tmp_dRh));
    }));

    Tensor dWx_flat = dWx_all.reshape({time_steps * batch_size, D});
    Tensor x_flat = x.reshape({time_steps * batch_size, D});
    Tensor dWx = torch::matmul(dWx_flat.t(), x_flat);
    Tensor dx_flat = torch::matmul(dWx_flat, Wx);
    Tensor dx = dx_flat.reshape({time_steps, batch_size, D});

    return { dx, dh0, dWx, dR, dbias };
}

// ============================================================================
// ElmanLeaky: tanh + input-dependent leaky integration (Mamba2-style Δ)
// ============================================================================

template<typename T>
using ElmanLeakyForward = hasty::v0::elman_leaky::ForwardPass<T>;
template<typename T>
using ElmanLeakyBackward = hasty::v0::elman_leaky::BackwardPass<T>;

std::vector<Tensor> elman_leaky_forward(
    bool training,
    Tensor x,
    Tensor h0,
    Tensor Wx,         // [D, D] - no gate dimension
    Tensor R,          // [D, D]
    Tensor bias,       // [D]
    Tensor delta) {    // [T, B, D] - precomputed delta values
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto D = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(Wx);
    CHECK_INPUT(R);
    CHECK_INPUT(bias);
    CHECK_INPUT(delta);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor x_flat = x.reshape({time_steps * batch_size, D});
    Tensor Wx_all = torch::matmul(x_flat, Wx.t());
    Wx_all = Wx_all.reshape({time_steps, batch_size, D});

    // Reshape delta to match kernel expectations
    Tensor delta_flat = delta.reshape({time_steps * batch_size, D});

    Tensor h = torch::empty({ time_steps + 1, batch_size, D }, options);
    Tensor v = training ? torch::empty({ time_steps, batch_size, D }, options)
                        : torch::empty({ 0 }, options);
    Tensor tmp_Rh = torch::empty({ batch_size, D }, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_leaky_forward", ([&] {
        ElmanLeakyForward<typename native_type<scalar_t>::T> forward(
            training,
            batch_size,
            D,
            D,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(R),
            ptr<scalar_t>(bias),
            ptr<scalar_t>(Wx_all),
            ptr<scalar_t>(delta_flat),
            ptr<scalar_t>(h),
            training ? ptr<scalar_t>(v) : nullptr,
            ptr<scalar_t>(tmp_Rh));
    }));

    return { h, v };
}

std::vector<Tensor> elman_leaky_backward(
    Tensor x,
    Tensor Wx,
    Tensor R,
    Tensor h,
    Tensor v,
    Tensor delta,      // [T, B, D] - same delta used in forward
    Tensor dh_new) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto D = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(Wx);
    CHECK_INPUT(R);
    CHECK_INPUT(h);
    CHECK_INPUT(v);
    CHECK_INPUT(delta);
    CHECK_INPUT(dh_new);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor delta_flat = delta.reshape({time_steps * batch_size, D});

    Tensor dWx_all = torch::empty({ time_steps, batch_size, D }, options);
    Tensor dR = torch::zeros({ D, D }, options);
    Tensor dbias = torch::zeros({ D }, options);
    Tensor d_delta = torch::empty({ time_steps, batch_size, D }, options);
    Tensor dh0 = torch::zeros({ batch_size, D }, options);
    Tensor tmp_dRh = torch::empty({ batch_size, D }, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_leaky_backward", ([&] {
        ElmanLeakyBackward<typename native_type<scalar_t>::T> backward(
            batch_size,
            D,
            D,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(R),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(delta_flat),
            ptr<scalar_t>(dh_new),
            ptr<scalar_t>(dWx_all),
            ptr<scalar_t>(dR),
            ptr<scalar_t>(dbias),
            ptr<scalar_t>(d_delta),
            ptr<scalar_t>(dh0),
            ptr<scalar_t>(tmp_dRh));
    }));

    Tensor dWx_flat = dWx_all.reshape({time_steps * batch_size, D});
    Tensor x_flat = x.reshape({time_steps * batch_size, D});
    Tensor dWx = torch::matmul(dWx_flat.t(), x_flat);
    Tensor dx_flat = torch::matmul(dWx_flat, Wx);
    Tensor dx = dx_flat.reshape({time_steps, batch_size, D});

    // Return d_delta reshaped to match input shape
    d_delta = d_delta.reshape({time_steps, batch_size, D});

    return { dx, dh0, dWx, dR, dbias, d_delta };
}

// ============================================================================
// ElmanLeakyMamba2Delta: Mamba2-style softplus/exp delta parameterization
// ============================================================================

template<typename T>
using ElmanLeakyMamba2DeltaForward = hasty::v0::elman_leaky_mamba2_delta::ForwardPass<T>;
template<typename T>
using ElmanLeakyMamba2DeltaBackward = hasty::v0::elman_leaky_mamba2_delta::BackwardPass<T>;

std::vector<Tensor> elman_leaky_mamba2_delta_forward(
    bool training,
    Tensor x,
    Tensor h0,
    Tensor Wx,            // [D, D]
    Tensor R,             // [D, D]
    Tensor bias,          // [D]
    Tensor delta_raw) {   // [T, B, D] - raw delta (before softplus!)
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto D = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(Wx);
    CHECK_INPUT(R);
    CHECK_INPUT(bias);
    CHECK_INPUT(delta_raw);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor x_flat = x.reshape({time_steps * batch_size, D});
    Tensor Wx_all = torch::matmul(x_flat, Wx.t());
    Wx_all = Wx_all.reshape({time_steps, batch_size, D});

    Tensor delta_raw_flat = delta_raw.reshape({time_steps * batch_size, D});

    Tensor h = torch::empty({ time_steps + 1, batch_size, D }, options);
    Tensor v = training ? torch::empty({ time_steps, batch_size, D }, options)
                        : torch::empty({ 0 }, options);
    Tensor decay_cache = training ? torch::empty({ time_steps, batch_size, D }, options)
                                  : torch::empty({ 0 }, options);
    Tensor tmp_Rh = torch::empty({ batch_size, D }, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_leaky_mamba2_delta_forward", ([&] {
        ElmanLeakyMamba2DeltaForward<typename native_type<scalar_t>::T> forward(
            training,
            batch_size,
            D,
            D,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(R),
            ptr<scalar_t>(bias),
            ptr<scalar_t>(Wx_all),
            ptr<scalar_t>(delta_raw_flat),
            ptr<scalar_t>(h),
            training ? ptr<scalar_t>(v) : nullptr,
            training ? ptr<scalar_t>(decay_cache) : nullptr,
            ptr<scalar_t>(tmp_Rh));
    }));

    return { h, v, decay_cache };
}

std::vector<Tensor> elman_leaky_mamba2_delta_backward(
    Tensor x,
    Tensor Wx,
    Tensor R,
    Tensor h,
    Tensor v,
    Tensor delta_raw,
    Tensor decay_cache,
    Tensor dh_new) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto D = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(Wx);
    CHECK_INPUT(R);
    CHECK_INPUT(h);
    CHECK_INPUT(v);
    CHECK_INPUT(delta_raw);
    CHECK_INPUT(decay_cache);
    CHECK_INPUT(dh_new);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor delta_raw_flat = delta_raw.reshape({time_steps * batch_size, D});
    Tensor decay_cache_flat = decay_cache.reshape({time_steps * batch_size, D});

    Tensor dWx_all = torch::empty({ time_steps, batch_size, D }, options);
    Tensor dR = torch::zeros({ D, D }, options);
    Tensor dbias = torch::zeros({ D }, options);
    Tensor d_delta_raw = torch::empty({ time_steps, batch_size, D }, options);
    Tensor dh0 = torch::zeros({ batch_size, D }, options);
    Tensor tmp_dRh = torch::empty({ batch_size, D }, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_leaky_mamba2_delta_backward", ([&] {
        ElmanLeakyMamba2DeltaBackward<typename native_type<scalar_t>::T> backward(
            batch_size,
            D,
            D,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(R),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(delta_raw_flat),
            ptr<scalar_t>(decay_cache_flat),
            ptr<scalar_t>(dh_new),
            ptr<scalar_t>(dWx_all),
            ptr<scalar_t>(dh0),       // dh (gradient to pass back)
            ptr<scalar_t>(dR),        // dR (gradient for R matrix)
            ptr<scalar_t>(dbias),
            ptr<scalar_t>(d_delta_raw),
            ptr<scalar_t>(tmp_dRh));
    }));

    Tensor dWx_flat = dWx_all.reshape({time_steps * batch_size, D});
    Tensor x_flat = x.reshape({time_steps * batch_size, D});
    Tensor dWx = torch::matmul(dWx_flat.t(), x_flat);
    Tensor dx_flat = torch::matmul(dWx_flat, Wx);
    Tensor dx = dx_flat.reshape({time_steps, batch_size, D});

    d_delta_raw = d_delta_raw.reshape({time_steps, batch_size, D});

    return { dx, dh0, dWx, dR, dbias, d_delta_raw };
}

// ============================================================================
// ElmanNoDelta: FIXED decay, no input-dependent delta (ablation)
// ============================================================================

template<typename T>
using ElmanNoDeltaForward = hasty::v0::elman_no_delta::ForwardPass<T>;
template<typename T>
using ElmanNoDeltaBackward = hasty::v0::elman_no_delta::BackwardPass<T>;

std::vector<Tensor> elman_no_delta_forward(
    bool training,
    Tensor x,
    Tensor h0,
    Tensor Wx,         // [D, D] - no gate dimension
    Tensor R,          // [D, D]
    Tensor bias,       // [D]
    float alpha) {     // SCALAR fixed decay factor
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto D = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(Wx);
    CHECK_INPUT(R);
    CHECK_INPUT(bias);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor x_flat = x.reshape({time_steps * batch_size, D});
    Tensor Wx_all = torch::matmul(x_flat, Wx.t());
    Wx_all = Wx_all.reshape({time_steps, batch_size, D});

    Tensor h = torch::empty({ time_steps + 1, batch_size, D }, options);
    Tensor v = training ? torch::empty({ time_steps, batch_size, D }, options)
                        : torch::empty({ 0 }, options);
    Tensor tmp_Rh = torch::empty({ batch_size, D }, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_no_delta_forward", ([&] {
        ElmanNoDeltaForward<typename native_type<scalar_t>::T> forward(
            training,
            batch_size,
            D,
            D,
            alpha,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(R),
            ptr<scalar_t>(bias),
            ptr<scalar_t>(Wx_all),
            ptr<scalar_t>(h),
            training ? ptr<scalar_t>(v) : nullptr,
            ptr<scalar_t>(tmp_Rh));
    }));

    return { h, v };
}

std::vector<Tensor> elman_no_delta_backward(
    Tensor x,
    Tensor Wx,
    Tensor R,
    Tensor h,
    Tensor v,
    Tensor dh_new,
    float alpha) {     // SCALAR fixed decay factor
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto D = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(Wx);
    CHECK_INPUT(R);
    CHECK_INPUT(h);
    CHECK_INPUT(v);
    CHECK_INPUT(dh_new);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dWx_all = torch::empty({ time_steps, batch_size, D }, options);
    Tensor dR = torch::zeros({ D, D }, options);
    Tensor dbias = torch::zeros({ D }, options);
    Tensor dh0 = torch::zeros({ batch_size, D }, options);
    Tensor tmp_dRh = torch::empty({ batch_size, D }, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_no_delta_backward", ([&] {
        ElmanNoDeltaBackward<typename native_type<scalar_t>::T> backward(
            batch_size,
            D,
            D,
            alpha,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(R),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(dh_new),
            ptr<scalar_t>(dWx_all),
            ptr<scalar_t>(dR),
            ptr<scalar_t>(dbias),
            ptr<scalar_t>(dh0),
            ptr<scalar_t>(tmp_dRh));
    }));

    Tensor dWx_flat = dWx_all.reshape({time_steps * batch_size, D});
    Tensor x_flat = x.reshape({time_steps * batch_size, D});
    Tensor dWx = torch::matmul(dWx_flat.t(), x_flat);
    Tensor dx_flat = torch::matmul(dWx_flat, Wx);
    Tensor dx = dx_flat.reshape({time_steps, batch_size, D});

    // NO d_delta return since alpha is fixed!
    return { dx, dh0, dWx, dR, dbias };
}

// ============================================================================
// ElmanMamba2: Mamba2-style linear recurrence (NO R matrix, NO tanh!)
// ============================================================================

template<typename T>
using ElmanMamba2Forward = hasty::v0::elman_mamba2::ForwardPass<T>;
template<typename T>
using ElmanMamba2Backward = hasty::v0::elman_mamba2::BackwardPass<T>;

std::vector<Tensor> elman_mamba2_forward(
    bool training,
    Tensor x,           // [T, B, D] - input sequence
    Tensor h0,          // [B, D] - initial hidden state
    Tensor W_delta,     // [D, D] - delta projection weights
    Tensor b_delta,     // [D] - delta projection bias
    Tensor W_B,         // [D, D] - B projection weights
    Tensor b_B) {       // [D] - B projection bias
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto D = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(W_delta);
    CHECK_INPUT(b_delta);
    CHECK_INPUT(W_B);
    CHECK_INPUT(b_B);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    // Output tensors
    Tensor h = torch::empty({ time_steps + 1, batch_size, D }, options);
    Tensor Wx_delta_cache = training ? torch::empty({ time_steps, batch_size, D }, options)
                                     : torch::empty({ 0 }, options);
    Tensor Bx_cache = training ? torch::empty({ time_steps, batch_size, D }, options)
                               : torch::empty({ 0 }, options);

    // Workspace
    Tensor tmp_Wx_delta = torch::empty({ batch_size, D }, options);
    Tensor tmp_Bx = torch::empty({ batch_size, D }, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_mamba2_forward", ([&] {
        ElmanMamba2Forward<typename native_type<scalar_t>::T> forward(
            training,
            batch_size,
            D,
            D,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W_delta),
            ptr<scalar_t>(b_delta),
            ptr<scalar_t>(W_B),
            ptr<scalar_t>(b_B),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            training ? ptr<scalar_t>(Wx_delta_cache) : nullptr,
            training ? ptr<scalar_t>(Bx_cache) : nullptr,
            ptr<scalar_t>(tmp_Wx_delta),
            ptr<scalar_t>(tmp_Bx));
    }));

    return { h, Wx_delta_cache, Bx_cache };
}

std::vector<Tensor> elman_mamba2_backward(
    Tensor x,            // [T, B, D]
    Tensor W_delta,      // [D, D]
    Tensor W_B,          // [D, D]
    Tensor h,            // [T+1, B, D]
    Tensor Wx_delta_cache,  // [T, B, D]
    Tensor Bx_cache,     // [T, B, D]
    Tensor dh_new) {     // [T+1, B, D]
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto D = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(W_delta);
    CHECK_INPUT(W_B);
    CHECK_INPUT(h);
    CHECK_INPUT(Wx_delta_cache);
    CHECK_INPUT(Bx_cache);
    CHECK_INPUT(dh_new);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    // Output tensors
    Tensor dx = torch::empty({ time_steps, batch_size, D }, options);
    Tensor dW_delta = torch::zeros({ D, D }, options);
    Tensor db_delta = torch::zeros({ D }, options);
    Tensor dW_B = torch::zeros({ D, D }, options);
    Tensor db_B = torch::zeros({ D }, options);
    Tensor dh0 = torch::zeros({ batch_size, D }, options);

    // Workspace
    Tensor tmp_d_Wx_delta = torch::empty({ batch_size, D }, options);
    Tensor tmp_d_Bx = torch::empty({ batch_size, D }, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_mamba2_backward", ([&] {
        ElmanMamba2Backward<typename native_type<scalar_t>::T> backward(
            batch_size,
            D,
            D,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W_delta),
            ptr<scalar_t>(W_B),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(Wx_delta_cache),
            ptr<scalar_t>(Bx_cache),
            ptr<scalar_t>(dh_new),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dW_delta),
            ptr<scalar_t>(db_delta),
            ptr<scalar_t>(dW_B),
            ptr<scalar_t>(db_B),
            ptr<scalar_t>(dh0),
            ptr<scalar_t>(tmp_d_Wx_delta),
            ptr<scalar_t>(tmp_d_Bx));
    }));

    return { dx, dh0, dW_delta, db_delta, dW_B, db_B };
}

// ============================================================================
// ElmanMamba2Silu: Mamba2 structure + silu nonlinearity (no R matrix)
// ============================================================================

template<typename T>
using ElmanMamba2SiluForward = hasty::v0::elman_mamba2_silu::ForwardPass<T>;
template<typename T>
using ElmanMamba2SiluBackward = hasty::v0::elman_mamba2_silu::BackwardPass<T>;

std::vector<Tensor> elman_mamba2_silu_forward(
    bool training,
    Tensor x,           // [T, B, D] - input sequence
    Tensor h0,          // [B, D] - initial hidden state
    Tensor W_delta,     // [D, D] - delta projection weights
    Tensor b_delta,     // [D] - delta projection bias
    Tensor W_B,         // [D, D] - B projection weights
    Tensor b_B) {       // [D] - B projection bias
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto D = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(W_delta);
    CHECK_INPUT(b_delta);
    CHECK_INPUT(W_B);
    CHECK_INPUT(b_B);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    // Output tensors
    Tensor h = torch::empty({ time_steps + 1, batch_size, D }, options);
    Tensor Wx_delta_cache = training ? torch::empty({ time_steps, batch_size, D }, options)
                                     : torch::empty({ 0 }, options);
    Tensor Bx_cache = training ? torch::empty({ time_steps, batch_size, D }, options)
                               : torch::empty({ 0 }, options);

    // Workspace
    Tensor tmp_Wx_delta = torch::empty({ batch_size, D }, options);
    Tensor tmp_Bx = torch::empty({ batch_size, D }, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_mamba2_silu_forward", ([&] {
        ElmanMamba2SiluForward<typename native_type<scalar_t>::T> forward(
            training,
            batch_size,
            D,
            D,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W_delta),
            ptr<scalar_t>(b_delta),
            ptr<scalar_t>(W_B),
            ptr<scalar_t>(b_B),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            training ? ptr<scalar_t>(Wx_delta_cache) : nullptr,
            training ? ptr<scalar_t>(Bx_cache) : nullptr,
            ptr<scalar_t>(tmp_Wx_delta),
            ptr<scalar_t>(tmp_Bx));
    }));

    return { h, Wx_delta_cache, Bx_cache };
}

std::vector<Tensor> elman_mamba2_silu_backward(
    Tensor x,            // [T, B, D]
    Tensor W_delta,      // [D, D]
    Tensor W_B,          // [D, D]
    Tensor h,            // [T+1, B, D]
    Tensor Wx_delta_cache,  // [T, B, D]
    Tensor Bx_cache,     // [T, B, D]
    Tensor dh_new) {     // [T+1, B, D]
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto D = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(W_delta);
    CHECK_INPUT(W_B);
    CHECK_INPUT(h);
    CHECK_INPUT(Wx_delta_cache);
    CHECK_INPUT(Bx_cache);
    CHECK_INPUT(dh_new);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    // Output tensors
    Tensor dx = torch::empty({ time_steps, batch_size, D }, options);
    Tensor dW_delta = torch::zeros({ D, D }, options);
    Tensor db_delta = torch::zeros({ D }, options);
    Tensor dW_B = torch::zeros({ D, D }, options);
    Tensor db_B = torch::zeros({ D }, options);
    Tensor dh0 = torch::zeros({ batch_size, D }, options);

    // Workspace
    Tensor tmp_d_Wx_delta = torch::empty({ batch_size, D }, options);
    Tensor tmp_d_Bx = torch::empty({ batch_size, D }, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_mamba2_silu_backward", ([&] {
        ElmanMamba2SiluBackward<typename native_type<scalar_t>::T> backward(
            batch_size,
            D,
            D,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W_delta),
            ptr<scalar_t>(W_B),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(Wx_delta_cache),
            ptr<scalar_t>(Bx_cache),
            ptr<scalar_t>(dh_new),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dW_delta),
            ptr<scalar_t>(db_delta),
            ptr<scalar_t>(dW_B),
            ptr<scalar_t>(db_B),
            ptr<scalar_t>(dh0),
            ptr<scalar_t>(tmp_d_Wx_delta),
            ptr<scalar_t>(tmp_d_Bx));
    }));

    return { dx, dh0, dW_delta, db_delta, dW_B, db_B };
}

// ============================================================================
// ElmanLeakySilu: silu + leaky integration (NO output gate)
// ============================================================================

template<typename T>
using ElmanLeakySiluForward = hasty::v0::elman_leaky_silu::ForwardPass<T>;
template<typename T>
using ElmanLeakySiluBackward = hasty::v0::elman_leaky_silu::BackwardPass<T>;

std::vector<Tensor> elman_leaky_silu_forward(
    bool training,
    Tensor x,
    Tensor h0,
    Tensor Wx,         // [D, D]
    Tensor R,          // [D, D]
    Tensor bias,       // [D]
    Tensor delta) {    // [T, B, D] - precomputed delta values (sigmoid output)
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto D = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(Wx);
    CHECK_INPUT(R);
    CHECK_INPUT(bias);
    CHECK_INPUT(delta);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor x_flat = x.reshape({time_steps * batch_size, D});
    Tensor Wx_all = torch::matmul(x_flat, Wx.t());
    Wx_all = Wx_all.reshape({time_steps, batch_size, D});

    Tensor delta_flat = delta.reshape({time_steps * batch_size, D});

    Tensor h = torch::empty({ time_steps + 1, batch_size, D }, options);
    Tensor v = training ? torch::empty({ time_steps, batch_size, D }, options)
                        : torch::empty({ 0 }, options);
    Tensor tmp_Rh = torch::empty({ batch_size, D }, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_leaky_silu_forward", ([&] {
        ElmanLeakySiluForward<typename native_type<scalar_t>::T> forward(
            training,
            batch_size,
            D,
            D,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(R),
            ptr<scalar_t>(bias),
            ptr<scalar_t>(Wx_all),
            ptr<scalar_t>(delta_flat),
            ptr<scalar_t>(h),
            training ? ptr<scalar_t>(v) : nullptr,
            ptr<scalar_t>(tmp_Rh));
    }));

    return { h, v };
}

std::vector<Tensor> elman_leaky_silu_backward(
    Tensor x,
    Tensor Wx,
    Tensor R,
    Tensor h,
    Tensor v,
    Tensor delta,
    Tensor dh_new) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto D = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(Wx);
    CHECK_INPUT(R);
    CHECK_INPUT(h);
    CHECK_INPUT(v);
    CHECK_INPUT(delta);
    CHECK_INPUT(dh_new);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor delta_flat = delta.reshape({time_steps * batch_size, D});

    Tensor dWx_all = torch::empty({ time_steps, batch_size, D }, options);
    Tensor dR = torch::zeros({ D, D }, options);
    Tensor dbias = torch::zeros({ D }, options);
    Tensor d_delta = torch::empty({ time_steps, batch_size, D }, options);
    Tensor dh0 = torch::zeros({ batch_size, D }, options);
    Tensor tmp_dRh = torch::empty({ batch_size, D }, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_leaky_silu_backward", ([&] {
        ElmanLeakySiluBackward<typename native_type<scalar_t>::T> backward(
            batch_size,
            D,
            D,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(R),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(delta_flat),
            ptr<scalar_t>(dh_new),
            ptr<scalar_t>(dWx_all),
            ptr<scalar_t>(dR),
            ptr<scalar_t>(dbias),
            ptr<scalar_t>(d_delta),
            ptr<scalar_t>(dh0),
            ptr<scalar_t>(tmp_dRh));
    }));

    Tensor dWx_flat = dWx_all.reshape({time_steps * batch_size, D});
    Tensor x_flat = x.reshape({time_steps * batch_size, D});
    Tensor dWx = torch::matmul(dWx_flat.t(), x_flat);
    Tensor dx_flat = torch::matmul(dWx_flat, Wx);
    Tensor dx = dx_flat.reshape({time_steps, batch_size, D});

    d_delta = d_delta.reshape({time_steps, batch_size, D});

    return { dx, dh0, dWx, dR, dbias, d_delta };
}

// ============================================================================
// ElmanLeakyDiag: tanh + DIAGONAL R + leaky integration (like Mamba's diagonal A!)
// ============================================================================

template<typename T>
using ElmanLeakyDiagForward = hasty::v0::elman_leaky_diag::ForwardPass<T>;
template<typename T>
using ElmanLeakyDiagBackward = hasty::v0::elman_leaky_diag::BackwardPass<T>;

std::vector<Tensor> elman_leaky_diag_forward(
    bool training,
    Tensor x,
    Tensor h0,
    Tensor Wx,         // [D, input_size]
    Tensor r,          // [D] - DIAGONAL recurrence weights!
    Tensor bias,       // [D]
    Tensor delta) {    // [T, B, D] - precomputed delta values
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto D = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(Wx);
    CHECK_INPUT(r);
    CHECK_INPUT(bias);
    CHECK_INPUT(delta);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor x_flat = x.reshape({time_steps * batch_size, D});
    Tensor Wx_all = torch::matmul(x_flat, Wx.t());
    Wx_all = Wx_all.reshape({time_steps, batch_size, D});

    // Reshape delta to match kernel expectations
    Tensor delta_flat = delta.reshape({time_steps * batch_size, D});

    Tensor h = torch::empty({ time_steps + 1, batch_size, D }, options);
    Tensor v = training ? torch::empty({ time_steps, batch_size, D }, options)
                        : torch::empty({ 0 }, options);
    // Note: no tmp_Rh needed for diagonal version!

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_leaky_diag_forward", ([&] {
        ElmanLeakyDiagForward<typename native_type<scalar_t>::T> forward(
            training,
            batch_size,
            D,
            D,
            at::cuda::getCurrentCUDABlasHandle(),  // unused but kept for API compat
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(r),
            ptr<scalar_t>(bias),
            ptr<scalar_t>(Wx_all),
            ptr<scalar_t>(delta_flat),
            ptr<scalar_t>(h),
            training ? ptr<scalar_t>(v) : nullptr);
    }));

    return { h, v };
}

std::vector<Tensor> elman_leaky_diag_backward(
    Tensor x,
    Tensor Wx,
    Tensor r,          // [D] - DIAGONAL recurrence weights
    Tensor h,
    Tensor v,
    Tensor delta,      // [T, B, D] - same delta used in forward
    Tensor dh_new) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto D = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(Wx);
    CHECK_INPUT(r);
    CHECK_INPUT(h);
    CHECK_INPUT(v);
    CHECK_INPUT(delta);
    CHECK_INPUT(dh_new);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor delta_flat = delta.reshape({time_steps * batch_size, D});

    Tensor dWx_all = torch::empty({ time_steps, batch_size, D }, options);
    Tensor dr = torch::zeros({ D }, options);  // DIAGONAL! [D] not [D,D]
    Tensor dbias = torch::zeros({ D }, options);
    Tensor d_delta = torch::empty({ time_steps, batch_size, D }, options);
    Tensor dh0 = torch::zeros({ batch_size, D }, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_leaky_diag_backward", ([&] {
        ElmanLeakyDiagBackward<typename native_type<scalar_t>::T> backward(
            batch_size,
            D,
            D,
            at::cuda::getCurrentCUDABlasHandle(),  // unused but kept for API compat
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(r),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(delta_flat),
            ptr<scalar_t>(dh_new),
            ptr<scalar_t>(dWx_all),
            ptr<scalar_t>(dr),
            ptr<scalar_t>(dbias),
            ptr<scalar_t>(d_delta),
            ptr<scalar_t>(dh0));
    }));

    Tensor dWx_flat = dWx_all.reshape({time_steps * batch_size, D});
    Tensor x_flat = x.reshape({time_steps * batch_size, D});
    Tensor dWx = torch::matmul(dWx_flat.t(), x_flat);
    Tensor dx_flat = torch::matmul(dWx_flat, Wx);
    Tensor dx = dx_flat.reshape({time_steps, batch_size, D});

    d_delta = d_delta.reshape({time_steps, batch_size, D});

    return { dx, dh0, dWx, dr, dbias, d_delta };
}

// ============================================================================
// ElmanLeakySelective: tanh + Mamba2-style discretization + per-channel decay
// ============================================================================

template<typename T>
using ElmanLeakySelectiveForward = hasty::v0::elman_leaky_selective::ForwardPass<T>;
template<typename T>
using ElmanLeakySelectiveBackward = hasty::v0::elman_leaky_selective::BackwardPass<T>;

std::vector<Tensor> elman_leaky_selective_forward(
    bool training,
    Tensor x,
    Tensor h0,
    Tensor Wx,         // [D, D] - candidate input projection
    Tensor R,          // [D, D] - recurrence matrix
    Tensor bias,       // [D] - candidate bias
    Tensor delta_raw,  // [T, B, D] - precomputed W_delta @ x + b_delta (before softplus)
    Tensor A) {        // [D] - per-channel decay rates (log-space)
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto D = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(Wx);
    CHECK_INPUT(R);
    CHECK_INPUT(bias);
    CHECK_INPUT(delta_raw);
    CHECK_INPUT(A);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor x_flat = x.reshape({time_steps * batch_size, D});
    Tensor Wx_all = torch::matmul(x_flat, Wx.t());
    Wx_all = Wx_all.reshape({time_steps, batch_size, D});

    // Reshape delta_raw to match kernel expectations
    Tensor delta_raw_flat = delta_raw.reshape({time_steps * batch_size, D});

    Tensor h = torch::empty({ time_steps + 1, batch_size, D }, options);
    // v stores [raw, delta_raw] for backward, so 2*D per timestep
    Tensor v = training ? torch::empty({ time_steps, batch_size, 2 * D }, options)
                        : torch::empty({ 0 }, options);
    Tensor tmp_Rh = torch::empty({ batch_size, D }, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_leaky_selective_forward", ([&] {
        ElmanLeakySelectiveForward<typename native_type<scalar_t>::T> forward(
            training,
            batch_size,
            D,
            D,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(R),
            ptr<scalar_t>(bias),
            ptr<scalar_t>(Wx_all),
            ptr<scalar_t>(delta_raw_flat),
            ptr<scalar_t>(A),
            ptr<scalar_t>(h),
            training ? ptr<scalar_t>(v) : nullptr,
            ptr<scalar_t>(tmp_Rh));
    }));

    return { h, v };
}

std::vector<Tensor> elman_leaky_selective_backward(
    Tensor x,
    Tensor Wx,
    Tensor R,
    Tensor h,
    Tensor v,
    Tensor A,          // [D] - per-channel decay rates
    Tensor dh_new) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto D = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(Wx);
    CHECK_INPUT(R);
    CHECK_INPUT(h);
    CHECK_INPUT(v);
    CHECK_INPUT(A);
    CHECK_INPUT(dh_new);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dWx_all = torch::empty({ time_steps, batch_size, D }, options);
    Tensor dR = torch::zeros({ D, D }, options);
    Tensor dbias = torch::zeros({ D }, options);
    Tensor d_delta_raw = torch::empty({ time_steps, batch_size, D }, options);
    Tensor dA = torch::zeros({ D }, options);
    Tensor dh0 = torch::zeros({ batch_size, D }, options);
    Tensor tmp_dRh = torch::empty({ batch_size, D }, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_leaky_selective_backward", ([&] {
        ElmanLeakySelectiveBackward<typename native_type<scalar_t>::T> backward(
            batch_size,
            D,
            D,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(R),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(A),
            ptr<scalar_t>(dh_new),
            ptr<scalar_t>(dWx_all),
            ptr<scalar_t>(dR),
            ptr<scalar_t>(dbias),
            ptr<scalar_t>(d_delta_raw),
            ptr<scalar_t>(dA),
            ptr<scalar_t>(dh0),
            ptr<scalar_t>(tmp_dRh));
    }));

    Tensor dWx_flat = dWx_all.reshape({time_steps * batch_size, D});
    Tensor x_flat = x.reshape({time_steps * batch_size, D});
    Tensor dWx = torch::matmul(dWx_flat.t(), x_flat);
    Tensor dx_flat = torch::matmul(dWx_flat, Wx);
    Tensor dx = dx_flat.reshape({time_steps, batch_size, D});

    // Reshape d_delta_raw to match input shape
    d_delta_raw = d_delta_raw.reshape({time_steps, batch_size, D});

    return { dx, dh0, dWx, dR, dbias, d_delta_raw, dA };
}

}  // anonymous namespace

void elman_variants_init(py::module& m) {
    // ElmanTanh
    m.def("elman_tanh_forward", &elman_tanh_forward, "ElmanTanh forward",
          py::call_guard<py::gil_scoped_release>());
    m.def("elman_tanh_backward", &elman_tanh_backward, "ElmanTanh backward",
          py::call_guard<py::gil_scoped_release>());

    // ElmanSigmoid
    m.def("elman_sigmoid_forward", &elman_sigmoid_forward, "ElmanSigmoid forward",
          py::call_guard<py::gil_scoped_release>());
    m.def("elman_sigmoid_backward", &elman_sigmoid_backward, "ElmanSigmoid backward",
          py::call_guard<py::gil_scoped_release>());

    // ElmanSwish
    m.def("elman_swish_forward", &elman_swish_forward, "ElmanSwish forward",
          py::call_guard<py::gil_scoped_release>());
    m.def("elman_swish_backward", &elman_swish_backward, "ElmanSwish backward",
          py::call_guard<py::gil_scoped_release>());

    // ElmanGelu
    m.def("elman_gelu_forward", &elman_gelu_forward, "ElmanGelu forward",
          py::call_guard<py::gil_scoped_release>());
    m.def("elman_gelu_backward", &elman_gelu_backward, "ElmanGelu backward",
          py::call_guard<py::gil_scoped_release>());

    // ElmanNoGate
    m.def("elman_nogate_forward", &elman_nogate_forward, "ElmanNoGate forward",
          py::call_guard<py::gil_scoped_release>());
    m.def("elman_nogate_backward", &elman_nogate_backward, "ElmanNoGate backward",
          py::call_guard<py::gil_scoped_release>());

    // ElmanLeaky - input-dependent delta (Mamba2-style discretization)
    m.def("elman_leaky_forward", &elman_leaky_forward, "ElmanLeaky forward",
          py::call_guard<py::gil_scoped_release>());
    m.def("elman_leaky_backward", &elman_leaky_backward, "ElmanLeaky backward",
          py::call_guard<py::gil_scoped_release>());

    // ElmanLeakyMamba2Delta - Mamba2-style softplus/exp delta parameterization
    m.def("elman_leaky_mamba2_delta_forward", &elman_leaky_mamba2_delta_forward, "ElmanLeakyMamba2Delta forward",
          py::call_guard<py::gil_scoped_release>());
    m.def("elman_leaky_mamba2_delta_backward", &elman_leaky_mamba2_delta_backward, "ElmanLeakyMamba2Delta backward",
          py::call_guard<py::gil_scoped_release>());

    // ElmanNoDelta - FIXED decay, no input-dependent delta (ablation)
    m.def("elman_no_delta_forward", &elman_no_delta_forward, "ElmanNoDelta forward",
          py::call_guard<py::gil_scoped_release>());
    m.def("elman_no_delta_backward", &elman_no_delta_backward, "ElmanNoDelta backward",
          py::call_guard<py::gil_scoped_release>());

    // ElmanMamba2 - Mamba2-style linear recurrence (NO R matrix, NO tanh!)
    m.def("elman_mamba2_forward", &elman_mamba2_forward, "ElmanMamba2 forward",
          py::call_guard<py::gil_scoped_release>());
    m.def("elman_mamba2_backward", &elman_mamba2_backward, "ElmanMamba2 backward",
          py::call_guard<py::gil_scoped_release>());

    // ElmanMamba2Silu - Mamba2 structure + silu nonlinearity (no R matrix)
    m.def("elman_mamba2_silu_forward", &elman_mamba2_silu_forward, "ElmanMamba2Silu forward",
          py::call_guard<py::gil_scoped_release>());
    m.def("elman_mamba2_silu_backward", &elman_mamba2_silu_backward, "ElmanMamba2Silu backward",
          py::call_guard<py::gil_scoped_release>());

    // ElmanLeakySilu - silu + leaky integration (NO output gate)
    m.def("elman_leaky_silu_forward", &elman_leaky_silu_forward, "ElmanLeakySilu forward",
          py::call_guard<py::gil_scoped_release>());
    m.def("elman_leaky_silu_backward", &elman_leaky_silu_backward, "ElmanLeakySilu backward",
          py::call_guard<py::gil_scoped_release>());

    // ElmanLeakyDiag - diagonal R + leaky integration (like Mamba's diagonal A!)
    m.def("elman_leaky_diag_forward", &elman_leaky_diag_forward, "ElmanLeakyDiag forward",
          py::call_guard<py::gil_scoped_release>());
    m.def("elman_leaky_diag_backward", &elman_leaky_diag_backward, "ElmanLeakyDiag backward",
          py::call_guard<py::gil_scoped_release>());

    // ElmanLeakySelective - Mamba2-style discretization + per-channel decay + nonlinearity
    m.def("elman_leaky_selective_forward", &elman_leaky_selective_forward, "ElmanLeakySelective forward",
          py::call_guard<py::gil_scoped_release>());
    m.def("elman_leaky_selective_backward", &elman_leaky_selective_backward, "ElmanLeakySelective backward",
          py::call_guard<py::gil_scoped_release>());
}
