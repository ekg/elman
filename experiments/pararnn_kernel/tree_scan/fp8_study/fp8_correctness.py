"""Step 4: correctness through the hybrid.

Compares PararnnHybridE88V2 bf16 path vs a fp8-storage variant end-to-end:
- output rel error
- gradient rel error (dK, dV, dQ, ddec, dS0)
"""
import os, sys
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from phase6_hybrid import PararnnHybridE88V2
from pararnn_seq_fwd_v2 import pararnn_seq_fwd_v2, backward_v2
from pararnn_seq_fwd_v2_fp8 import pararnn_seq_fwd_v2_fp8, backward_v2_fp8


class PararnnHybridE88V2FP8(torch.autograd.Function):
    """FP8-storage variant of PararnnHybridE88V2. Square state only (Ns==Hv)."""

    @staticmethod
    def forward(ctx, training, k, v, q, decay, S0, n_heads):
        T, B, H, Ns = k.shape
        _, _, _, Hv = v.shape
        assert Ns == Hv, "fp8 hybrid supports square state only"
        K_p = k.permute(1, 2, 0, 3).contiguous()
        V_p = v.permute(1, 2, 0, 3).contiguous()
        Q_p = q.permute(1, 2, 0, 3).contiguous()
        decay_p = decay.permute(1, 2, 0).contiguous()
        S0_p = S0.transpose(-1, -2).contiguous()
        fwd_nw = 4 if Ns >= 32 else 1
        S_traj_fp8 = pararnn_seq_fwd_v2_fp8(S0_p, K_p, V_p, decay_p, num_warps=fwd_nw)
        # Sq still needs a non-fp8 type for einsum to work efficiently.
        # Rehydrate to bf16. This is extra traffic — the optimal fp8 path
        # would fuse Sq computation into the fwd kernel.
        S_traj_bf16 = S_traj_fp8.to(torch.bfloat16)
        Sq = torch.einsum('bhtpq,bhtq->bhtp', S_traj_bf16, Q_p)
        S_final_p = S_traj_bf16[:, :, -1]
        S_final = S_final_p.transpose(-1, -2).contiguous()
        output = Sq.permute(2, 0, 1, 3).contiguous()
        ctx.save_for_backward(K_p, V_p, Q_p, decay_p, S_traj_fp8, S0_p)
        ctx.dims = (T, B, H, Ns, Hv)
        return S_final, output

    @staticmethod
    def backward(ctx, dS_final, d_output):
        K_p, V_p, Q_p, decay_p, S_traj_fp8, S0_p = ctx.saved_tensors
        T, B, H, Ns, Hv = ctx.dims
        dL_dout_p = d_output.permute(1, 2, 0, 3).contiguous()
        g_T_p = dS_final.transpose(-1, -2).contiguous()
        # fp8 backward: tune num_warps=1 on N=16, num_warps=4 on N=32 per bench
        bwd_nw = 1 if Ns <= 16 else 4
        dS0_p, dK_p, dV_p, dQ_p, ddec_p = backward_v2_fp8(
            S0_p, S_traj_fp8, K_p, V_p, decay_p, g_T_p, dL_dout_p, Q_p,
            num_warps=bwd_nw, num_stages=1,
        )
        dK_out = dK_p.permute(2, 0, 1, 3).contiguous()
        dV_out = dV_p.permute(2, 0, 1, 3).contiguous()
        dQ_out = dQ_p.permute(2, 0, 1, 3).contiguous()
        ddec_out = ddec_p.permute(2, 0, 1).contiguous()
        dS0_out = dS0_p.transpose(-1, -2).contiguous()
        return None, dK_out, dV_out, dQ_out, ddec_out, dS0_out, None


def rel(a, b):
    num = (a.float() - b.float()).abs().max().item()
    denom = max(b.float().abs().max().item(), 1e-10)
    return num / denom


def compare_hybrid(B, H, T, N, seed=0):
    dt = torch.bfloat16
    torch.manual_seed(seed)
    k = (0.3 * torch.randn(T, B, H, N, dtype=dt, device='cuda')).requires_grad_(True)
    v = (0.3 * torch.randn(T, B, H, N, dtype=dt, device='cuda')).requires_grad_(True)
    q = (0.3 * torch.randn(T, B, H, N, dtype=dt, device='cuda')).requires_grad_(True)
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, dtype=dt, device='cuda')).detach().requires_grad_(True)
    S0 = (0.1 * torch.randn(B, H, N, N, dtype=dt, device='cuda')).requires_grad_(True)
    torch.manual_seed(seed + 1)
    dL = 0.01 * torch.randn(T, B, H, N, dtype=dt, device='cuda')

    # BF16 path
    k1, v1, q1, d1, S01 = [x.detach().clone().requires_grad_(True) for x in [k, v, q, decay, S0]]
    S_f_b, out_b = PararnnHybridE88V2.apply(True, k1, v1, q1, d1, S01, H)
    (out_b * dL).sum().backward()

    # FP8 path
    k2, v2, q2, d2, S02 = [x.detach().clone().requires_grad_(True) for x in [k, v, q, decay, S0]]
    S_f_f, out_f = PararnnHybridE88V2FP8.apply(True, k2, v2, q2, d2, S02, H)
    (out_f * dL).sum().backward()

    return {
        'output': rel(out_f, out_b),
        'S_final': rel(S_f_f, S_f_b),
        'dK': rel(k2.grad, k1.grad),
        'dV': rel(v2.grad, v1.grad),
        'dQ': rel(q2.grad, q1.grad),
        'ddec': rel(d2.grad, d1.grad),
        'dS0': rel(S02.grad, S01.grad),
    }


if __name__ == '__main__':
    print("Hybrid FP8 vs BF16 correctness:\n")
    shapes = [(1, 4, 1024, 16), (1, 4, 4096, 16), (1, 4, 16384, 16),
              (1, 4, 1024, 32), (1, 4, 4096, 32)]
    for B, H, T, N in shapes:
        r = compare_hybrid(B, H, T, N)
        worst = max(r.values())
        ok = "PASS" if worst < 0.05 else "WARN" if worst < 0.10 else "FAIL"
        print(f"  B={B} H={H} T={T:5d} N={N}: " +
              "  ".join(f"{k}={v:.2e}" for k, v in r.items()) + f"  [{ok}]")
