"""E94 autograd.Function — wraps Triton fwd+bwd for one layer's time recurrence."""
import os, sys
THIS = os.path.dirname(os.path.abspath(__file__))
if THIS not in sys.path:
    sys.path.insert(0, THIS)

import torch
from e94_time_fwd import e94_time_fwd
from e94_time_bwd import e94_time_bwd


class E94TimeFunction(torch.autograd.Function):
    """One-layer time recurrence (delta-rule mode, l=0).

    Forward args:
        S0:        [B, H, N, HD]   — initial state
        W_h_time:  [H, N, N]       — per-head time matrix
        K:         [B, T, H, N]    — keys (assumed L2-normalized along N by caller)
        V:         [B, T, H, HD]   — values

    Returns:
        S_traj: [B, T, H, N, HD]
    """
    @staticmethod
    def forward(ctx, S0, W_h_time, K, V):
        S_traj = e94_time_fwd(S0, W_h_time, K=K, V=V, use_delta=True)
        ctx.save_for_backward(S0, W_h_time, K, V, S_traj)
        return S_traj

    @staticmethod
    def backward(ctx, d_S_traj):
        S0, W_h_time, K, V, S_traj = ctx.saved_tensors
        d_S_T = torch.zeros_like(S0)  # no separate final-state grad — all goes through d_S_traj
        dS0, dWh, dK, dV, _ = e94_time_bwd(
            S0, S_traj, W_h_time,
            d_S_traj.contiguous(), d_S_T,
            K=K, V=V, use_delta=True,
        )
        return dS0, dWh, dK, dV


class E94TimeWriteFunction(torch.autograd.Function):
    """One-layer time recurrence (pre-computed write mode, l>0).

    Forward args:
        S0:        [B, H, N, HD]
        W_h_time:  [H, N, N]
        Write:     [B, T, H, N, HD]   — pre-computed inputs (e.g. W_h_layer · prev_layer_state)

    Returns:
        S_traj: [B, T, H, N, HD]
    """
    @staticmethod
    def forward(ctx, S0, W_h_time, Write):
        S_traj = e94_time_fwd(S0, W_h_time, Write=Write, use_delta=False)
        ctx.save_for_backward(S0, W_h_time, Write, S_traj)
        return S_traj

    @staticmethod
    def backward(ctx, d_S_traj):
        S0, W_h_time, Write, S_traj = ctx.saved_tensors
        d_S_T = torch.zeros_like(S0)
        dS0, dWh, _, _, dWrite = e94_time_bwd(
            S0, S_traj, W_h_time,
            d_S_traj.contiguous(), d_S_T,
            Write=Write, use_delta=False,
        )
        return dS0, dWh, dWrite
