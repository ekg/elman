"""Hybrid E88 with fused silu gate — drop-in for E88FusedGateCUDAFunction.

E88's best CUDA kernel fuses `Sq * silu(g)` into the backward.  For the
Pararnn hybrid to match or beat it in training, we need the same gate
fusion.

This wraps PararnnHybridE88V2 with Python autograd for the gate:
  forward:  Sq = hybrid(...); output = Sq * silu(g)
  backward: d_Sq = d_output * silu(g)
            d_g = d_output * Sq * silu'(g)    (autograd computes this)

The gate ops are ~elementwise, so Triton fusion would save a tiny amount
of HBM bandwidth.  Measurable gains come from avoiding the extra kernel
launch for the gate.  We use a single custom autograd.Function so the
gate ops share the same save_for_backward with the recurrence.
"""

import torch
import torch.nn.functional as F
import sys
import os

THIS = os.path.dirname(os.path.abspath(__file__))
if THIS not in sys.path:
    sys.path.insert(0, THIS)

from phase6_hybrid import PararnnHybridE88V2


class PararnnHybridE88FusedGate(torch.autograd.Function):
    """Same signature as E88FusedGateCUDAFunction:
       forward(training, k, v, q, decay, g, S0, n_heads).
    """

    @staticmethod
    def forward(ctx, training, k, v, q, decay, g, S0, n_heads):
        # Run the hybrid recurrence to get the un-gated Sq (output).
        S_final, Sq = PararnnHybridE88V2.apply(training, k, v, q, decay, S0, n_heads)

        # Apply silu gate: output = Sq * silu(g) = Sq * g * sigmoid(g)
        # Save Sq, g, S_final-grad-path tensors for backward.
        sig_g = torch.sigmoid(g)
        silu_g = g * sig_g
        output = Sq * silu_g

        ctx.save_for_backward(Sq, g, sig_g)
        # Pass the inputs through so we can call PararnnHybridE88V2.backward
        # via ctx when d_output arrives.  Trick: save k, v, q, decay, S0
        # shapes by running PararnnHybridE88V2 directly — but we need
        # autograd to track it.  Easiest: re-run the hybrid with ctx
        # delegation.
        #
        # Instead, we let PararnnHybridE88V2 track its OWN autograd, and we
        # just do the gate.  So we use Tensor autograd for everything by
        # NOT making this a custom Function but a Python wrapper.  See
        # hybrid_with_fused_gate() below for the implementation actually
        # used.  This class is kept for reference / structural clarity.
        return S_final, output

    @staticmethod
    def backward(ctx, dS_final, d_output):
        # Not used — see hybrid_with_fused_gate.
        raise NotImplementedError


def hybrid_with_fused_gate(training, k, v, q, decay, g, S0, n_heads):
    """Functional equivalent of E88FusedGateCUDAFunction using Pararnn hybrid.

    Autograd tracks the gate operation; PararnnHybridE88V2 tracks the
    recurrence.  No custom backward needed — PyTorch composes them.
    """
    S_final, Sq = PararnnHybridE88V2.apply(training, k, v, q, decay, S0, n_heads)
    output = Sq * F.silu(g)
    return S_final, output


if __name__ == '__main__':
    import time

    # Correctness vs E88FusedGateCUDAFunction
    from elman.models.e88_fla_hybrid import E88FusedGateCUDAFunction

    print("Correctness vs E88FusedGateCUDAFunction:")
    for B, H, T, N in [(16, 141, 512, 16), (16, 83, 512, 32)]:
        dt = torch.bfloat16
        torch.manual_seed(0)
        k = (0.3 * torch.randn(T, B, H, N, dtype=dt, device='cuda')).requires_grad_(True)
        v = (0.3 * torch.randn(T, B, H, N, dtype=dt, device='cuda')).requires_grad_(True)
        q = (0.3 * torch.randn(T, B, H, N, dtype=dt, device='cuda')).requires_grad_(True)
        decay = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, dtype=dt, device='cuda')).detach().requires_grad_(True)
        g = torch.randn(T, B, H, N, dtype=dt, device='cuda').requires_grad_(True)
        S0 = (0.1 * torch.randn(B, H, N, N, dtype=dt, device='cuda')).requires_grad_(True)

        # CUDA fused
        k1, v1, q1, decay1, g1, S01 = [x.detach().clone().requires_grad_(True) for x in [k, v, q, decay, g, S0]]
        S_f_c, out_c = E88FusedGateCUDAFunction.apply(True, k1, v1, q1, decay1, g1, S01, H)
        torch.manual_seed(1)
        dL = 0.01 * torch.randn_like(out_c)
        (out_c * dL).sum().backward()

        # Hybrid functional
        k2, v2, q2, decay2, g2, S02 = [x.detach().clone().requires_grad_(True) for x in [k, v, q, decay, g, S0]]
        S_f_h, out_h = hybrid_with_fused_gate(True, k2, v2, q2, decay2, g2, S02, H)
        (out_h * dL).sum().backward()

        def rel(a, b):
            return (a.float() - b.float()).abs().max().item() / max(b.float().abs().max().item(), 1e-10)
        print(f"  B={B} H={H} T={T} N={N}:")
        print(f"    output rel={rel(out_h, out_c):.2e}")
        print(f"    dK rel={rel(k2.grad, k1.grad):.2e}")
        print(f"    dV rel={rel(v2.grad, v1.grad):.2e}")
        print(f"    dQ rel={rel(q2.grad, q1.grad):.2e}")
        print(f"    dg rel={rel(g2.grad, g1.grad):.2e}")
        print(f"    ddec rel={rel(decay2.grad, decay1.grad):.2e}")

    # Benchmark
    print("\nBenchmark (includes fwd + bwd, per-layer):")
    for B, H, T, N in [(16, 141, 512, 16), (16, 83, 512, 32)]:
        dt = torch.bfloat16
        g_rng = torch.Generator(device='cuda').manual_seed(0)
        k = (0.3 * torch.randn(T, B, H, N, generator=g_rng, dtype=dt, device='cuda')).requires_grad_(True)
        v = (0.3 * torch.randn(T, B, H, N, generator=g_rng, dtype=dt, device='cuda')).requires_grad_(True)
        q = (0.3 * torch.randn(T, B, H, N, generator=g_rng, dtype=dt, device='cuda')).requires_grad_(True)
        decay = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, generator=g_rng, dtype=dt, device='cuda')).detach().requires_grad_(True)
        gt = torch.randn(T, B, H, N, generator=g_rng, dtype=dt, device='cuda').requires_grad_(True)
        S0 = (0.1 * torch.randn(B, H, N, N, generator=g_rng, dtype=dt, device='cuda')).requires_grad_(True)

        def run_cuda():
            _, out = E88FusedGateCUDAFunction.apply(True, k, v, q, decay, gt, S0, H)
            loss = out.sum()
            loss.backward()
            for t in [k, v, q, decay, gt, S0]:
                if t.grad is not None: t.grad = None

        def run_hyb():
            _, out = hybrid_with_fused_gate(True, k, v, q, decay, gt, S0, H)
            loss = out.sum()
            loss.backward()
            for t in [k, v, q, decay, gt, S0]:
                if t.grad is not None: t.grad = None

        for _ in range(5): run_cuda()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(20): run_cuda()
        torch.cuda.synchronize()
        cuda_ms = (time.time() - t0) / 20 * 1000

        for _ in range(5): run_hyb()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(20): run_hyb()
        torch.cuda.synchronize()
        hyb_ms = (time.time() - t0) / 20 * 1000

        print(f"  B={B} H={H} T={T} N={N}:  CUDA-fused={cuda_ms:>6.3f}ms  Hybrid-fused={hyb_ms:>6.3f}ms  "
              f"spd={cuda_ms/hyb_ms:.2f}×")
