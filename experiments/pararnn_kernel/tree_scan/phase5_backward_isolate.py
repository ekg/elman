"""Isolate the Pararnn backward kernel from the hybrid forward.

Test: run Python sequential Pararnn forward in fp32 (autograd), use its
S_traj as ground-truth input to backward_e88_fused_rank1, then compare
grads to autograd grads.

This answers: does backward_e88_fused_rank1 produce correct gradients
for Pararnn-convention forward? Independent of ADMM/warmup-scan errors.
"""

import sys, os
import torch

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase4_newton_driver import e88_step_batched, sequential_e88_forward
from phase7_fused_backward import backward_e88_fused_rank1


def autograd_forward(S0, K, V, decay, Q):
    T = K.shape[0]
    S = S0
    outputs = []
    for t in range(T):
        S = e88_step_batched(S, K[t], V[t], decay[t])
        out_t = torch.einsum('bhij,bhj->bhi', S, Q[t])
        outputs.append(out_t)
    output = torch.stack(outputs, dim=0)
    return S, output


def test_backward(B, H, T, N, seed=0):
    """Pararnn backward vs autograd in fp32."""
    g = torch.Generator(device='cuda').manual_seed(seed)
    dt32 = torch.float32
    k_f = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt32, device='cuda'))
    v_f = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt32, device='cuda'))
    q_f = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt32, device='cuda'))
    decay_f = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, generator=g, dtype=dt32, device='cuda'))
    S0_f = (0.1 * torch.randn(B, H, N, N, generator=g, dtype=dt32, device='cuda'))

    # === Autograd fp32 forward + backward ===
    k_ag = k_f.clone().requires_grad_(True)
    v_ag = v_f.clone().requires_grad_(True)
    q_ag = q_f.clone().requires_grad_(True)
    decay_ag = decay_f.clone().requires_grad_(True)
    S0_ag = S0_f.clone().requires_grad_(True)
    S_final, output = autograd_forward(S0_ag, k_ag, v_ag, decay_ag, q_ag)

    torch.manual_seed(seed + 1)
    dL_dout = 0.01 * torch.randn_like(output)
    dL_dSfinal = 0.1 * torch.randn_like(S_final)
    loss = (output * dL_dout).sum() + (S_final * dL_dSfinal).sum()
    loss.backward()

    dK_ag = k_ag.grad.clone()
    dV_ag = v_ag.grad.clone()
    dQ_ag = q_ag.grad.clone()
    ddec_ag = decay_ag.grad.clone()
    dS0_ag = S0_ag.grad.clone()

    # === Build S_traj from autograd forward (fp32, ground truth) ===
    # S_traj[:, :, 0] = S0, S_traj[:, :, t+1] = state after step t
    # Shape for Pararnn kernel: [B, H, T+1, N, N]
    with torch.no_grad():
        K_bhhtn = k_f.permute(1, 2, 0, 3).contiguous()
        V_bhhtn = v_f.permute(1, 2, 0, 3).contiguous()
        decay_bhht = decay_f.permute(1, 2, 0).contiguous()
        S_traj = sequential_e88_forward(S0_f, K_bhhtn, V_bhhtn, decay_bhht)  # [B,H,T+1,N,N]

    # === Call Pararnn backward kernel in fp32 (no precision loss) ===
    dt = torch.float32
    S_traj_bf = S_traj.to(dt)
    K_bf = k_f.permute(1, 2, 0, 3).contiguous().to(dt)
    V_bf = v_f.permute(1, 2, 0, 3).contiguous().to(dt)
    decay_bf = decay_f.permute(1, 2, 0).contiguous().to(dt)
    q_bf = q_f.permute(1, 2, 0, 3).contiguous().to(dt)
    dL_dout_bf = dL_dout.permute(1, 2, 0, 3).contiguous().to(dt)
    g_T = dL_dSfinal.to(dt).contiguous()

    dS0_k, dK_k, dV_k, ddec_k = backward_e88_fused_rank1(
        S_traj_bf, K_bf, V_bf, decay_bf, g_T, dL_dout_bf, q_bf,
        num_warps=4 if N == 32 else 1, num_stages=1)

    # Pararnn output layout: [B, H, T, N] → reshape to [T, B, H, N]
    dK_p = dK_k.permute(2, 0, 1, 3).contiguous()
    dV_p = dV_k.permute(2, 0, 1, 3).contiguous()
    ddec_p = ddec_k.permute(2, 0, 1).contiguous()
    dS0_p = dS0_k

    # dQ via einsum, independent of backward kernel
    dQ_einsum = torch.einsum('bhti,bhtij->bhtj', dL_dout_bf, S_traj_bf[:, :, 1:])
    dQ_p = dQ_einsum.permute(2, 0, 1, 3).contiguous()

    def rel(a, b, tag):
        num = (a.float() - b.float()).abs().max().item()
        denom = max(b.float().abs().max().item(), 1e-10)
        r = num / denom
        status = "PASS" if r < 1e-3 else "FAIL"  # fp32 — expect tight match
        print(f"    {tag:>8s}: max_rel={r:.2e}  (|a|={a.float().abs().max().item():.2e}, |b|={b.float().abs().max().item():.2e})  [{status}]")
        return r

    print(f"\n  B={B} H={H} T={T} N={N}")
    rel(dK_p, dK_ag, "dK")
    rel(dV_p, dV_ag, "dV")
    rel(dQ_p, dQ_ag, "dQ")
    rel(ddec_p, ddec_ag, "ddecay")
    rel(dS0_p, dS0_ag, "dS0")


if __name__ == '__main__':
    print("Backward isolate: kernel vs fp32 autograd (both Pararnn convention)\n")
    for shape in [(1, 2, 256, 16), (1, 4, 1024, 16), (1, 4, 1024, 32), (1, 2, 4096, 16)]:
        try:
            test_backward(*shape)
        except Exception as e:
            import traceback
            traceback.print_exc()
