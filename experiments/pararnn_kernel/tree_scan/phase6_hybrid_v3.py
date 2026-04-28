"""Phase 6 V3 — [B, T, H, N]-native hybrid E88 (no internal permutes).

Drop-in equivalent to PararnnHybridE88V2 but accepts inputs directly in the
production [B, T, H, N] layout.  Eliminates all 10+ permute/contiguous calls
per invocation.

Signature difference: accepts k,v,q in [B, T, H, N] and decay in [B, T, H].
This is what E88OptimizedCUDAFunction sees.  Returns S_final in [B, H, Ns, Hv]
(CUDA layout) and output in [B, T, H, Hv].

Only implements square state (Ns == Hv); rectangular paths fall back to V2.
"""

import os
import sys
import torch

THIS = os.path.dirname(os.path.abspath(__file__))
if THIS not in sys.path:
    sys.path.insert(0, THIS)
PARENT = os.path.dirname(THIS)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
ROOT = os.path.dirname(PARENT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pararnn_seq_fwd_v3_bthn import pararnn_seq_fwd_v3_bthn
from pararnn_seq_bwd_v3_bthn import backward_v3_bthn


class PararnnHybridE88V3(torch.autograd.Function):
    """[B, T, H, N]-native hybrid.

    Expects:
      k:     [B, T, H, N]
      v:     [B, T, H, N]
      q:     [B, T, H, N]
      decay: [B, T, H]
      S0:    [B, H, Ns, Hv]   (CUDA convention)

    Returns:
      S_final: [B, H, Ns, Hv]  (CUDA convention)
      output:  [B, T, H, Hv]   (production layout — no permute back)
    """

    @staticmethod
    def forward(ctx, training, k, v, q, decay, S0, n_heads):
        B, T, H, N = k.shape
        assert v.shape == q.shape == (B, T, H, N), "v3 only supports square state (Ns == Hv)"
        assert decay.shape == (B, T, H)
        assert S0.shape == (B, H, N, N)

        # S0 conversion: CUDA is [B, H, Ns, Hv]; Pararnn is [B, H, M, N] = [B, H, Hv, Ns].
        # For square N = Ns = Hv, this is just a transpose of the last two dims.
        S0_p = S0.transpose(-1, -2).contiguous()

        # Forward: returns S_traj [B, H, T, N, N] and Sq [B, T, H, N]
        fwd_nw = 8 if N >= 32 else 2
        S_traj, Sq = pararnn_seq_fwd_v3_bthn(
            S0_p, k, v, q, decay, num_warps=fwd_nw,
        )
        # S_final in Pararnn is S_traj[-1] [B, H, N, N]; convert to CUDA [B, H, N, N] (transpose)
        S_final_p = S_traj[:, :, -1]
        S_final = S_final_p.transpose(-1, -2).contiguous()
        output = Sq  # already [B, T, H, N]

        # Save for backward
        ctx.save_for_backward(k, v, q, decay, S_traj, S0_p)
        ctx.dims = (B, T, H, N, fwd_nw)
        return S_final, output

    @staticmethod
    def backward(ctx, dS_final, d_output):
        k, v, q, decay, S_traj, S0_p = ctx.saved_tensors
        B, T, H, N, fwd_nw = ctx.dims
        bwd_nw = 2 if N >= 32 else 1

        # g_T: dL/dS_T in Pararnn convention. CUDA layout is [B, H, Ns, Hv],
        # Pararnn is [B, H, Hv, Ns] = transpose.
        g_T_p = dS_final.transpose(-1, -2).contiguous()

        dS0_p, dK, dV, dQ, ddec = backward_v3_bthn(
            S0_p, S_traj, k, v, q, decay,
            g_T_p, d_output,
            num_warps=bwd_nw, num_stages=1,
        )

        # dS0 output: convert back to CUDA layout
        dS0 = dS0_p.transpose(-1, -2).contiguous()

        # Forward signature:
        # (ctx, training, k, v, q, decay, S0, n_heads)
        return None, dK, dV, dQ, ddec, dS0, None


# ===========================================================================
# Fused gate variant (matches E88FusedGateCUDAFunction signature)
# ===========================================================================
import torch.nn.functional as F


def hybrid_v3_with_fused_gate(training, k, v, q, decay, g, S0, n_heads):
    """[B, T, H, N]-native version of hybrid_with_fused_gate.

    Signature matches E88FusedGateCUDAFunction but expects [B, T, H, N] inputs.
    """
    S_final, Sq = PararnnHybridE88V3.apply(training, k, v, q, decay, S0, n_heads)
    output = Sq * F.silu(g)
    return S_final, output


# ===========================================================================
# Correctness vs V2 at production shapes
# ===========================================================================
if __name__ == '__main__':
    import time
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '3')

    from phase6_hybrid import PararnnHybridE88V2

    def rel(a, b):
        return (a.float() - b.float()).abs().max().item() / max(b.float().abs().max().item(), 1e-10)

    print("Correctness V3 vs V2 at production shapes (fp32 for exact match, then bf16 for noise check):\n")
    for dt_label, dt, thresh in [("fp32", torch.float32, 1e-5), ("bf16", torch.bfloat16, 1e-2)]:
      print(f"--- {dt_label} ---")
      for B, T, H, N in [(1, 4096, 4, 16), (1, 4096, 4, 32), (2, 2048, 8, 32), (4, 512, 16, 16)]:
        torch.manual_seed(0)
        # Inputs in [B, T, H, N] layout
        k = (0.3 * torch.randn(B, T, H, N, dtype=dt, device='cuda')).requires_grad_(True)
        v = (0.3 * torch.randn(B, T, H, N, dtype=dt, device='cuda')).requires_grad_(True)
        q = (0.3 * torch.randn(B, T, H, N, dtype=dt, device='cuda')).requires_grad_(True)
        decay = torch.sigmoid(0.5 + 0.1 * torch.randn(B, T, H, dtype=dt, device='cuda')).detach().requires_grad_(True)
        S0 = (0.1 * torch.randn(B, H, N, N, dtype=dt, device='cuda')).requires_grad_(True)

        # V3 direct
        k3, v3t, q3, decay3, S03 = [x.detach().clone().requires_grad_(True) for x in [k, v, q, decay, S0]]
        S_f3, out3 = PararnnHybridE88V3.apply(True, k3, v3t, q3, decay3, S03, H)

        # V2 reference: transpose to [T, B, H, N], run V2, transpose back
        k_tb = k.detach().transpose(0, 1).contiguous().requires_grad_(True)
        v_tb = v.detach().transpose(0, 1).contiguous().requires_grad_(True)
        q_tb = q.detach().transpose(0, 1).contiguous().requires_grad_(True)
        decay_tb = decay.detach().transpose(0, 1).contiguous().requires_grad_(True)
        S0_v2 = S0.detach().clone().requires_grad_(True)
        S_f2, out2_tb = PararnnHybridE88V2.apply(True, k_tb, v_tb, q_tb, decay_tb, S0_v2, H)
        out2_bt = out2_tb.transpose(0, 1).contiguous()

        torch.manual_seed(1)
        dL_dout = 0.01 * torch.randn_like(out3)
        dL_dout_tb = dL_dout.transpose(0, 1).contiguous()

        (out3 * dL_dout).sum().backward()
        (out2_tb * dL_dout_tb).sum().backward()

        # Compare
        r = {
            'output': rel(out3, out2_bt),
            'S_final': rel(S_f3, S_f2),
            'dK': rel(k3.grad, k_tb.grad.transpose(0, 1).contiguous()),
            'dV': rel(v3t.grad, v_tb.grad.transpose(0, 1).contiguous()),
            'dQ': rel(q3.grad, q_tb.grad.transpose(0, 1).contiguous()),
            'ddec': rel(decay3.grad, decay_tb.grad.transpose(0, 1).contiguous()),
            'dS0': rel(S03.grad, S0_v2.grad),
        }
        worst = max(r.values())
        ok = "PASS" if worst < thresh else "FAIL"
        details = "  ".join(f"{k}={v:.1e}" for k, v in r.items())
        print(f"  B={B} T={T} H={H} N={N}:")
        print(f"    {details}  [{ok}]")
