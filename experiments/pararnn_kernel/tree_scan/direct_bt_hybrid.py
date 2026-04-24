"""Skip the BT->TB intermediate layout: go directly [B,T,H,N] -> [B,H,T,N].

Current install_hybrid.patched_optimized does:
  [B,T,H,N] -> [T,B,H,N] (via .transpose(0,1).contiguous())     one copy
  [T,B,H,N] -> [B,H,T,N] (via .permute(1,2,0,3).contiguous())   another copy
  ... then kernel...
  [B,H,T,M] -> [T,B,H,M] (output permute in phase6)             copy
  [T,B,H,M] -> [B,T,H,M] (final transpose in patched_optimized) copy

That's 4 copies just for layout changes. With direct [B,T,H,N] -> [B,H,T,N]
(and output reversed), we'd have just 2 copies.
"""

import os, sys
import torch
import torch.nn.functional as F

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, THIS)
sys.path.insert(0, os.path.dirname(THIS))

from pararnn_seq_fwd_v2 import pararnn_seq_fwd_v2
from pararnn_seq_fwd_rect import pararnn_seq_fwd_output_triton
from pararnn_bwd_fused_dq import backward_with_dq


class PararnnHybridE88_DirectBT(torch.autograd.Function):
    """Hybrid taking [B,T,H,N] directly, eliminating the intermediate [T,B,H,N] pass."""

    @staticmethod
    def forward(ctx, training, k, v, q, decay, S0, n_heads):
        """
        k, v, q: [B, T, H, N]
        decay:   [B, T, H]
        S0:      [B, H, Ns, Hv]
        Returns (S_final [B, H, Ns, Hv], output [B, T, H, Hv]).
        """
        B, T, H, Ns = k.shape
        _, _, _, Hv = v.shape

        # Direct [B,T,H,N] -> [B,H,T,N] (single permute)
        K_p = k.permute(0, 2, 1, 3).contiguous()
        V_p = v.permute(0, 2, 1, 3).contiguous()
        Q_p = q.permute(0, 2, 1, 3).contiguous()
        decay_p = decay.permute(0, 2, 1).contiguous()
        S0_p = S0.transpose(-1, -2).contiguous()

        use_v2 = (Ns == Hv) and (Ns >= 32)
        if use_v2:
            fwd_nw = 4
            S_traj = pararnn_seq_fwd_v2(S0_p, K_p, V_p, decay_p, num_warps=fwd_nw)
            Sq = torch.einsum('bhtpq,bhtq->bhtp', S_traj, Q_p)
            S_final_p = S_traj[:, :, -1]
        else:
            fwd_nw = 4 if max(Ns, Hv) >= 24 else 1
            S_traj, Sq = pararnn_seq_fwd_output_triton(S0_p, K_p, V_p, Q_p, decay_p, num_warps=fwd_nw)
            S_final_p = S_traj[:, :, -1]

        S_final = S_final_p.transpose(-1, -2).contiguous()
        # Direct [B,H,T,M] -> [B,T,H,M]
        output = Sq.permute(0, 2, 1, 3).contiguous()

        if use_v2:
            ctx.save_for_backward(K_p, V_p, Q_p, decay_p, S_traj, S0_p)
        else:
            ctx.save_for_backward(K_p, V_p, Q_p, decay_p, S_traj)
        ctx.dims = (T, B, H, Ns, Hv, fwd_nw, use_v2)
        return S_final, output

    @staticmethod
    def backward(ctx, dS_final, d_output):
        use_v2 = ctx.dims[6]
        if use_v2:
            K_p, V_p, Q_p, decay_p, S_traj, S0_p = ctx.saved_tensors
        else:
            K_p, V_p, Q_p, decay_p, S_traj = ctx.saved_tensors
        T, B, H, Ns, Hv = ctx.dims[:5]

        # d_output is [B,T,H,Hv] -> [B,H,T,Hv]
        dL_dout_p = d_output.permute(0, 2, 1, 3).contiguous()
        g_T_p = dS_final.transpose(-1, -2).contiguous()
        bwd_nw = 2 if Ns >= 32 else 1

        if use_v2:
            from pararnn_seq_fwd_v2 import backward_v2
            dS0_p, dK_p, dV_p, dQ_p, ddec_p = backward_v2(
                S0_p, S_traj, K_p, V_p, decay_p, g_T_p, dL_dout_p, Q_p,
                num_warps=bwd_nw, num_stages=1,
            )
        elif Ns == Hv:
            dS0_p, dK_p, dV_p, dQ_p, ddec_p = backward_with_dq(
                S_traj, K_p, V_p, decay_p, g_T_p, dL_dout_p, Q_p,
                num_warps=bwd_nw, num_stages=1,
            )
        else:
            from pararnn_bwd_rect import pararnn_bwd_rect
            dS0_p, dK_p, dV_p, ddec_p = pararnn_bwd_rect(
                S_traj, K_p, V_p, decay_p, g_T_p, dL_dout_p, Q_p,
                num_warps=bwd_nw, num_stages=1,
            )
            dQ_p = torch.einsum('bhti,bhtij->bhtj', dL_dout_p, S_traj[:, :, 1:])

        # [B,H,T,*] -> [B,T,H,*]
        dK_out = dK_p.permute(0, 2, 1, 3).contiguous()
        dV_out = dV_p.permute(0, 2, 1, 3).contiguous()
        dQ_out = dQ_p.permute(0, 2, 1, 3).contiguous()
        ddec_out = ddec_p.permute(0, 2, 1).contiguous()
        dS0_out = dS0_p.transpose(-1, -2).contiguous()

        return None, dK_out, dV_out, dQ_out, ddec_out, dS0_out, None


def hybrid_bt_fused_gate(training, k, v, q, decay, g, S0, n_heads):
    """Like hybrid_with_fused_gate but in [B,T,H,N] layout end-to-end."""
    S_final, Sq = PararnnHybridE88_DirectBT.apply(training, k, v, q, decay, S0, n_heads)
    output = Sq * F.silu(g)
    return S_final, output


# =====================================
# Correctness + timing
# =====================================
if __name__ == '__main__':
    import time
    from phase7_fused_gate_hybrid import hybrid_with_fused_gate
    from install_hybrid import install
    install()
    import elman.models.e88_fla_hybrid as e88m

    # Correctness vs CUDA optimized
    print("=== Correctness: DirectBT vs CUDA E88OptimizedCUDAFunction ===")
    torch.manual_seed(0)
    for B, T, H, N in [(1, 512, 4, 16), (1, 1024, 8, 32), (2, 256, 4, 16)]:
        dt = torch.bfloat16
        k = (0.3 * torch.randn(B, T, H, N, dtype=dt, device='cuda')).requires_grad_(True)
        v = (0.3 * torch.randn(B, T, H, N, dtype=dt, device='cuda')).requires_grad_(True)
        q = (0.3 * torch.randn(B, T, H, N, dtype=dt, device='cuda')).requires_grad_(True)
        decay = torch.sigmoid(0.5 + 0.1 * torch.randn(B, T, H, dtype=dt, device='cuda')).detach().requires_grad_(True)
        g = torch.randn(B, T, H, N, dtype=dt, device='cuda').requires_grad_(True)
        S0 = (0.1 * torch.randn(B, H, N, N, dtype=dt, device='cuda')).requires_grad_(True)

        # DirectBT
        k1, v1, q1, d1, g1, S01 = [x.detach().clone().requires_grad_(True) for x in [k, v, q, decay, g, S0]]
        _, out_d = hybrid_bt_fused_gate(True, k1, v1, q1, d1, g1, S01, H)
        torch.manual_seed(1)
        dL = 0.01 * torch.randn_like(out_d)
        (out_d * dL).sum().backward()

        # Reference: via original hybrid_with_fused_gate (needs TB layout)
        k2, v2, q2, d2, g2, S02 = [x.detach().clone().requires_grad_(True) for x in [k, v, q, decay, g, S0]]
        k2_tb = k2.transpose(0, 1).contiguous()
        v2_tb = v2.transpose(0, 1).contiguous()
        q2_tb = q2.transpose(0, 1).contiguous()
        d2_tb = d2.transpose(0, 1).contiguous()
        g2_tb = g2.transpose(0, 1).contiguous()
        # Actually just use E88OptimizedCUDAFunction (patched to hybrid)
        _, out_r = e88m.E88OptimizedCUDAFunction.apply(True, k2, v2, q2, d2, g2, S02, H, True, False, 16)
        (out_r * dL).sum().backward()

        def rel(a, b):
            return (a.float() - b.float()).abs().max().item() / max(b.float().abs().max().item(), 1e-10)
        print(f"  B={B} T={T} H={H} N={N}:")
        print(f"    output rel={rel(out_d, out_r):.2e}   dK={rel(k1.grad, k2.grad):.2e}   "
              f"dV={rel(v1.grad, v2.grad):.2e}   dQ={rel(q1.grad, q2.grad):.2e}   "
              f"dg={rel(g1.grad, g2.grad):.2e}   ddec={rel(d1.grad, d2.grad):.2e}")

    # Benchmark
    print("\n=== Benchmark: DirectBT vs install_hybrid (via patched_optimized) ===")
    for B, T, H, N in [(1, 32768, 141, 16), (1, 16384, 141, 16), (1, 8192, 141, 16)]:
        dt = torch.bfloat16
        k = torch.randn(B, T, H, N, dtype=dt, device='cuda').requires_grad_(True)
        v = torch.randn(B, T, H, N, dtype=dt, device='cuda').requires_grad_(True)
        q = torch.randn(B, T, H, N, dtype=dt, device='cuda').requires_grad_(True)
        d = torch.sigmoid(0.5 + 0.1 * torch.randn(B, T, H, dtype=dt, device='cuda')).detach().requires_grad_(True)
        g = torch.randn(B, T, H, N, dtype=dt, device='cuda').requires_grad_(True)
        S0 = (0.1 * torch.randn(B, H, N, N, dtype=dt, device='cuda')).requires_grad_(True)

        def run_direct():
            for t in [k, v, q, d, g, S0]: t.grad = None
            _, out = hybrid_bt_fused_gate(True, k, v, q, d, g, S0, H)
            out.sum().backward()

        def run_orig():
            for t in [k, v, q, d, g, S0]: t.grad = None
            _, out = e88m.E88OptimizedCUDAFunction.apply(True, k, v, q, d, g, S0, H, True, False, 16)
            out.sum().backward()

        for _ in range(3): run_direct()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10): run_direct()
        torch.cuda.synchronize()
        t_direct = (time.time() - t0) / 10 * 1000

        for _ in range(3): run_orig()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10): run_orig()
        torch.cuda.synchronize()
        t_orig = (time.time() - t0) / 10 * 1000

        print(f"  B={B} T={T} H={H} N={N}:  orig={t_orig:>6.2f}ms  direct={t_direct:>6.2f}ms  "
              f"speedup={t_orig/t_direct:.2f}x")
