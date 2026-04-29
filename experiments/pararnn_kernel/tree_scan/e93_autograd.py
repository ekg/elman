"""E93 autograd.Function — drop-in for the Python loop in E93Minimal."""
import os, sys
THIS = os.path.dirname(os.path.abspath(__file__))
if THIS not in sys.path:
    sys.path.insert(0, THIS)

import torch
from e93_seq_fwd import e93_seq_fwd
from e93_seq_bwd import e93_seq_bwd


class E93Function(torch.autograd.Function):
    """Forward + backward for E93 single-state matrix-matrix RNN.

    Args:
        S0:    [B, N, M]
        W_h:   [N, N]
        K:     [B, T, N]   (assumed L2-normalized along N by caller)
        V:     [B, T, M]
        decay: [B, T]
        M_TILE: tile size along M
        use_w_h, use_decay, use_delta, nl_kind: ablation flags
    """
    @staticmethod
    def forward(ctx, S0, W_h, K, V, decay, M_TILE: int = 64,
                use_w_h: bool = True, use_decay: bool = True,
                use_delta: bool = True, nl_kind: int = 0):
        S_traj, Sflat = e93_seq_fwd(
            S0, W_h, K, V, decay, M_TILE=M_TILE,
            use_w_h=use_w_h, use_decay=use_decay,
            use_delta=use_delta, nl_kind=nl_kind,
        )
        ctx.save_for_backward(S0, S_traj, W_h, K, V, decay)
        ctx.M_TILE = M_TILE
        ctx.use_w_h = use_w_h
        ctx.use_decay = use_decay
        ctx.use_delta = use_delta
        ctx.nl_kind = nl_kind
        S_final = S_traj[:, -1].contiguous()
        return Sflat, S_final

    @staticmethod
    def backward(ctx, d_Sflat, d_S_final):
        S0, S_traj, W_h, K, V, decay = ctx.saved_tensors
        dS0, dWh, dK, dV, ddec = e93_seq_bwd(
            S0, S_traj, W_h, K, V, decay,
            d_Sflat.contiguous(), d_S_final.contiguous(),
            M_TILE=ctx.M_TILE,
            use_w_h=ctx.use_w_h, use_decay=ctx.use_decay,
            use_delta=ctx.use_delta, nl_kind=ctx.nl_kind,
        )
        # Pass through None for ablated grads (not needed but keeps shape)
        if not ctx.use_w_h:
            dWh = None
        if not ctx.use_decay:
            ddec = None
        return dS0, dWh, dK, dV, ddec, None, None, None, None, None
