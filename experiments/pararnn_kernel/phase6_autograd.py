"""Phase 6 — wrap Phase 4 (Newton forward) + Phase 5 (reverse-scan backward)
into a torch.autograd.Function and validate end-to-end.

Interface: given inputs (S0, K, V, decay), return the full trajectory S
including the initial state. Backward computes gradients w.r.t. all four.
This is the primitive that plugs into E88's layer (its recurrence block);
the projections and gating around it are normal autograd-friendly code.
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase4_newton_driver import newton_e88_triton, sequential_e88_forward
from phase5_backward import backward_e88_triton, _autograd_forward


# -----------------------------------------------------------------------------
# torch.autograd.Function wrapping Newton forward + reverse-scan backward.
# -----------------------------------------------------------------------------

class PararnnE88Function(torch.autograd.Function):
    """Forward: Newton + Triton scan to get full trajectory.
       Backward: reverse Triton scan for gradients.
    """

    @staticmethod
    def forward(ctx, S0, K, V, decay, max_iters=20, tol=1e-5):
        """
        Args:
          S0:    [B, H, n, n]
          K, V:  [B, H, T, n]
          decay: [B, H, T]
        Returns:
          S:     [B, H, T+1, n, n]
        """
        S, iters_used, final_res = newton_e88_triton(
            S0, K, V, decay, max_iters=max_iters, tol=tol
        )
        ctx.save_for_backward(S, K, V, decay)
        return S

    @staticmethod
    def backward(ctx, grad_S):
        """grad_S is dL/dS[:, :, t] for t=0..T."""
        S, K, V, decay = ctx.saved_tensors
        dS0, dK, dV, ddec = backward_e88_triton(S, K, V, decay, grad_S)
        return dS0, dK, dV, ddec, None, None   # grads for max_iters/tol are None


def pararnn_e88(S0, K, V, decay, max_iters=20, tol=1e-5):
    """Convenience wrapper."""
    return PararnnE88Function.apply(S0, K, V, decay, max_iters, tol)


# -----------------------------------------------------------------------------
# End-to-end parity test: both forward and backward against a sequential
# autograd reference.
# -----------------------------------------------------------------------------

def test_e2e(B, H, T, n, seed=0, dtype=torch.float32, device='cuda'):
    torch.manual_seed(seed)
    S0 = 0.1 * torch.randn(B, H, n, n, dtype=dtype, device=device, requires_grad=True)
    K = 0.3 * torch.randn(B, H, T, n, dtype=dtype, device=device, requires_grad=True)
    V = 0.3 * torch.randn(B, H, T, n, dtype=dtype, device=device, requires_grad=True)
    decay = (0.9 + 0.1 * torch.rand(B, H, T, dtype=dtype, device=device)
             ).requires_grad_(True)

    # Common grad target
    target = torch.randn(B, H, T + 1, n, n, dtype=dtype, device=device)

    # --- Reference: sequential forward via autograd-friendly stack ---
    S_ref = _autograd_forward(S0, K, V, decay)
    loss_ref = (target * S_ref).sum()
    dS0_ref, dK_ref, dV_ref, ddec_ref = torch.autograd.grad(
        loss_ref, [S0, K, V, decay], retain_graph=False
    )

    # --- Test: PararnnE88Function forward + backward ---
    S0_t = S0.detach().clone().requires_grad_(True)
    K_t = K.detach().clone().requires_grad_(True)
    V_t = V.detach().clone().requires_grad_(True)
    decay_t = decay.detach().clone().requires_grad_(True)
    S_par = pararnn_e88(S0_t, K_t, V_t, decay_t, tol=1e-4)
    loss_par = (target * S_par).sum()
    dS0_par, dK_par, dV_par, ddec_par = torch.autograd.grad(
        loss_par, [S0_t, K_t, V_t, decay_t], retain_graph=False
    )

    def rel_err(a, b):
        return (a - b).abs().max().item() / max(b.abs().max().item(), 1e-30)

    S_err   = rel_err(S_par, S_ref)
    dS0_err = rel_err(dS0_par, dS0_ref)
    dK_err  = rel_err(dK_par, dK_ref)
    dV_err  = rel_err(dV_par, dV_ref)
    dd_err  = rel_err(ddec_par, ddec_ref)
    worst = max(S_err, dS0_err, dK_err, dV_err, dd_err)
    status = "PASS" if worst < 1e-3 else "FAIL"
    print(f"  B={B} H={H:3d} T={T:4d} n={n:3d}  "
          f"S={S_err:.2e} dS0={dS0_err:.2e} dK={dK_err:.2e} "
          f"dV={dV_err:.2e} dd={dd_err:.2e}  [{status}]")
    return worst


# -----------------------------------------------------------------------------
# Benchmark: full E88 scale, forward + backward
# -----------------------------------------------------------------------------

def benchmark_e2e():
    import time
    device = 'cuda'
    dtype = torch.float32

    def fresh_inputs(B, H, T, n, seed=0):
        g = torch.Generator(device=device).manual_seed(seed)
        S0 = (0.1 * torch.randn(B, H, n, n, generator=g, dtype=dtype, device=device)
              ).requires_grad_(True)
        K = (0.3 * torch.randn(B, H, T, n, generator=g, dtype=dtype, device=device)
             ).requires_grad_(True)
        V = (0.3 * torch.randn(B, H, T, n, generator=g, dtype=dtype, device=device)
             ).requires_grad_(True)
        decay = (0.9 + 0.1 * torch.rand(B, H, T, generator=g, dtype=dtype, device=device)
                 ).requires_grad_(True)
        target = torch.randn(B, H, T + 1, n, n, generator=g, dtype=dtype, device=device)
        return S0, K, V, decay, target

    def run_once(fn):
        S0, K, V, decay, target = fresh_inputs(B, H, T, n)
        S = fn(S0, K, V, decay)
        loss = (target * S).sum()
        loss.backward()

    for (B, H, T, n) in [(4, 112, 512, 32),
                         (2, 32, 1024, 32),
                         (1, 32, 2048, 32)]:
        print(f"\nForward+backward at B={B} H={H} T={T} n={n}:")
        # Warmup
        for _ in range(2):
            run_once(lambda S0, K, V, decay: pararnn_e88(S0, K, V, decay, tol=1e-4))
            run_once(lambda S0, K, V, decay: _autograd_forward(S0, K, V, decay))
        torch.cuda.synchronize()
        # Pararnn
        N = 3
        t0 = time.time()
        for _ in range(N):
            run_once(lambda S0, K, V, decay: pararnn_e88(S0, K, V, decay, tol=1e-4))
        torch.cuda.synchronize()
        par_ms = (time.time() - t0) / N * 1000
        # Sequential
        t0 = time.time()
        for _ in range(N):
            run_once(lambda S0, K, V, decay: _autograd_forward(S0, K, V, decay))
        torch.cuda.synchronize()
        seq_ms = (time.time() - t0) / N * 1000
        print(f"  Pararnn-Triton:  {par_ms:.1f} ms/step")
        print(f"  Sequential autograd: {seq_ms:.1f} ms/step")
        print(f"  Ratio (par/seq): {par_ms / seq_ms:.2f}×")


if __name__ == '__main__':
    print("Phase 6: autograd.Function end-to-end parity.\n")
    for B, H, T, n in [(1, 1, 16, 4),
                       (1, 2, 32, 8),
                       (1, 4, 64, 16),
                       (2, 8, 128, 32),
                       (2, 32, 256, 32),
                       (4, 112, 512, 32)]:   # full E88 scale
        test_e2e(B, H, T, n)

    benchmark_e2e()
