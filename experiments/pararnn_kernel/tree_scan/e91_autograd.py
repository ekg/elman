"""E91 autograd.Function — drop-in replacement for the Python loop in E91MatMat.

Uses Triton fwd/bwd kernels. Saves S_traj for backward (~T·N² memory per (b,h)).

Usage:
    from experiments.pararnn_kernel.tree_scan.e91_autograd import E91Function
    Sq, S_final = E91Function.apply(S0, K, V, Q, decay)
"""
import os, sys
THIS = os.path.dirname(os.path.abspath(__file__))
if THIS not in sys.path:
    sys.path.insert(0, THIS)

import torch
from e91_seq_fwd import e91_seq_fwd
from e91_seq_bwd import e91_seq_bwd


class E91Function(torch.autograd.Function):
    """Forward and backward for E91 rank-r delta rule recurrence.

    Forward args:
        S0:     [B, H, N, N]
        K, V:   [B, T, H, N, R]
        Q:      [B, T, H, N]
        decay:  [B, T, H]
    Returns:
        Sq:      [B, T, H, N]
        S_final: [B, H, N, N]   (last state, equals S_traj[:, :, -1])
    """

    @staticmethod
    def forward(ctx, S0, K, V, Q, decay):
        S_traj, Sq = e91_seq_fwd(S0, K, V, Q, decay)
        ctx.save_for_backward(S0, S_traj, K, V, Q, decay)
        S_final = S_traj[:, :, -1].contiguous()
        return Sq, S_final

    @staticmethod
    def backward(ctx, d_Sq, d_S_final):
        S0, S_traj, K, V, Q, decay = ctx.saved_tensors
        # d_S_final flows into the LAST state; equivalent to dS_T in the kernel.
        dS0, dK, dV, dQ, ddec = e91_seq_bwd(
            S0, S_traj, K, V, Q, decay, d_Sq.contiguous(), d_S_final.contiguous()
        )
        return dS0, dK, dV, dQ, ddec


# ============================================================================
# Self-test: gradient check via torch.autograd.gradcheck (small)
# ============================================================================
if __name__ == '__main__':
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '7')
    import torch
    torch.manual_seed(0)

    # Round-trip test: torch ref vs autograd Function output
    from e91_seq_fwd import e91_seq_fwd_torch_ref
    B, T, H, N, R = 2, 16, 4, 16, 16
    S0 = (0.1 * torch.randn(B, H, N, N, dtype=torch.float32, device='cuda')).requires_grad_(True)
    K = (0.3 * torch.randn(B, T, H, N, R, dtype=torch.float32, device='cuda')).requires_grad_(True)
    V = (0.3 * torch.randn(B, T, H, N, R, dtype=torch.float32, device='cuda')).requires_grad_(True)
    Q = (0.3 * torch.randn(B, T, H, N, dtype=torch.float32, device='cuda')).requires_grad_(True)
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(B, T, H, dtype=torch.float32, device='cuda')).requires_grad_(True)

    # Reference
    S_traj_ref, Sq_ref = e91_seq_fwd_torch_ref(S0, K, V, Q, decay)
    S_final_ref = S_traj_ref[:, :, -1]
    loss_ref = Sq_ref.sum() + S_final_ref.sum()
    grads_ref = torch.autograd.grad(loss_ref, [S0, K, V, Q, decay], retain_graph=False)

    # Triton via autograd.Function
    S0_2 = S0.detach().clone().requires_grad_(True)
    K_2 = K.detach().clone().requires_grad_(True)
    V_2 = V.detach().clone().requires_grad_(True)
    Q_2 = Q.detach().clone().requires_grad_(True)
    decay_2 = decay.detach().clone().requires_grad_(True)
    Sq, S_final = E91Function.apply(S0_2, K_2, V_2, Q_2, decay_2)
    loss = Sq.sum() + S_final.sum()
    grads_tri = torch.autograd.grad(loss, [S0_2, K_2, V_2, Q_2, decay_2])

    def rel(a, b): return (a - b).abs().max().item() / max(b.abs().max().item(), 1e-10)
    print("Forward + backward through autograd.Function vs Python ref:")
    print(f"  Sq:      {rel(Sq, Sq_ref):.2e}")
    print(f"  S_final: {rel(S_final, S_final_ref):.2e}")
    for name, g_t, g_r in zip(['dS0', 'dK', 'dV', 'dQ', 'ddec'], grads_tri, grads_ref):
        print(f"  {name}: {rel(g_t, g_r):.2e}")
