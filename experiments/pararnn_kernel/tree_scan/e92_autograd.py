"""E92 autograd.Function — drop-in replacement for the Python loop in E92MatMat."""
import os, sys
THIS = os.path.dirname(os.path.abspath(__file__))
if THIS not in sys.path:
    sys.path.insert(0, THIS)

import torch
from e92_seq_fwd import e92_seq_fwd
from e92_seq_bwd import e92_seq_bwd


class E92Function(torch.autograd.Function):
    """Forward and backward for E92 matrix-matrix nonlinear RNN with W_h.

    Args:
        S0:    [B, H, N, N]
        W_h:   [H, N, N]
        K, V:  [B, T, H, N]
        Q:     [B, T, H, N]
        decay: [B, T, H]
    Returns:
        Sq:      [B, T, H, N]
        S_final: [B, H, N, N]
    """

    @staticmethod
    def forward(ctx, S0, W_h, K, V, Q, decay):
        S_traj, Sq = e92_seq_fwd(S0, W_h, K, V, Q, decay)
        ctx.save_for_backward(S0, S_traj, W_h, K, V, Q, decay)
        S_final = S_traj[:, :, -1].contiguous()
        return Sq, S_final

    @staticmethod
    def backward(ctx, d_Sq, d_S_final):
        S0, S_traj, W_h, K, V, Q, decay = ctx.saved_tensors
        dS0, dWh, dK, dV, dQ, ddec = e92_seq_bwd(
            S0, S_traj, W_h, K, V, Q, decay,
            d_Sq.contiguous(), d_S_final.contiguous(),
        )
        return dS0, dWh, dK, dV, dQ, ddec
