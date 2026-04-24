"""Direct fp32 test: CUDA convention autograd vs Pararnn-backward-with-transposes.

Python forward in CUDA convention:
  S[i,j] = tanh(dec*S[i,j] + (v[j] - sum_r S[r,j]*k[r]) * k[i])
  output[j] = sum_i S[i,j] * q[i]

Then compare autograd gradients against those produced by the Pararnn
backward kernel after applying transpose conventions.
"""

import sys, os
import torch

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase4_newton_driver import sequential_e88_forward
from phase7_fused_backward import backward_e88_fused_rank1


def cuda_step(S_prev, k, v, decay):
    """One CUDA-convention E88 step.
    S_prev: [B, H, Ns, Hv]
    k: [B, H, Ns]
    v: [B, H, Hv]
    decay: [B, H]
    """
    # retrieved[j] = sum_i S[i,j] * k[i]
    retrieved = torch.einsum('bhij,bhi->bhj', S_prev, k)  # [B,H,Hv]
    delta = v - retrieved                                   # [B,H,Hv]
    outer = torch.einsum('bhj,bhi->bhij', delta, k)        # [B,H,Ns,Hv]
    pre = decay[..., None, None] * S_prev + outer
    return torch.tanh(pre)


def cuda_forward(S0, K, V, decay, Q):
    """K: [T, B, H, Ns], V: [T, B, H, Hv], Q: [T, B, H, Ns], decay: [T, B, H]
    S0: [B, H, Ns, Hv]
    Returns (S_final, output [T, B, H, Hv])."""
    T = K.shape[0]
    S = S0
    outputs = []
    for t in range(T):
        S = cuda_step(S, K[t], V[t], decay[t])
        # output[j] = sum_i S[i,j] * q[i]
        out_t = torch.einsum('bhij,bhi->bhj', S, Q[t])
        outputs.append(out_t)
    return S, torch.stack(outputs, dim=0)


def test(B, H, T, N, seed=0):
    """N is both Ns and Hv (square state)."""
    dt = torch.float32
    g = torch.Generator(device='cuda').manual_seed(seed)
    k = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda'))
    v = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda'))
    q = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda'))
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, generator=g, dtype=dt, device='cuda'))
    S0 = (0.1 * torch.randn(B, H, N, N, generator=g, dtype=dt, device='cuda'))

    # === Autograd through CUDA-convention Python forward ===
    k1 = k.clone().requires_grad_(True)
    v1 = v.clone().requires_grad_(True)
    q1 = q.clone().requires_grad_(True)
    decay1 = decay.clone().requires_grad_(True)
    S01 = S0.clone().requires_grad_(True)
    S_final, output = cuda_forward(S01, k1, v1, decay1, q1)

    torch.manual_seed(seed + 1)
    dL_dout = 0.01 * torch.randn_like(output)
    dL_dSfinal = 0.1 * torch.randn_like(S_final)
    loss = (output * dL_dout).sum() + (S_final * dL_dSfinal).sum()
    loss.backward()

    dK_ref = k1.grad.clone()
    dV_ref = v1.grad.clone()
    dQ_ref = q1.grad.clone()
    ddec_ref = decay1.grad.clone()
    dS0_ref = S01.grad.clone()

    # === Hybrid: Pararnn backward with transposes ===
    # Build Pararnn S_traj = CUDA_S_traj transposed on last two dims.
    # Easy way: run Pararnn forward on S0^T with same (k, v, decay).
    K_bhtn = k.permute(1, 2, 0, 3).contiguous()  # [B, H, T, N]
    V_bhtn = v.permute(1, 2, 0, 3).contiguous()
    decay_bht = decay.permute(1, 2, 0).contiguous()
    q_bhtn = q.permute(1, 2, 0, 3).contiguous()
    dL_dout_bhtn = dL_dout.permute(1, 2, 0, 3).contiguous()  # CUDA's output index = Pararnn's p

    S0_pararnn = S0.transpose(-1, -2).contiguous()
    S_traj = sequential_e88_forward(S0_pararnn, K_bhtn, V_bhtn, decay_bht)

    g_T_pararnn = dL_dSfinal.transpose(-1, -2).contiguous()

    dS0_p, dK_p, dV_p, ddec_p = backward_e88_fused_rank1(
        S_traj, K_bhtn, V_bhtn, decay_bht, g_T_pararnn, dL_dout_bhtn, q_bhtn,
        num_warps=4 if N == 32 else 1, num_stages=1)

    dQ_p = torch.einsum('bhti,bhtij->bhtj', dL_dout_bhtn, S_traj[:, :, 1:])

    # Permute to match autograd outputs
    dK_hyb = dK_p.permute(2, 0, 1, 3).contiguous()
    dV_hyb = dV_p.permute(2, 0, 1, 3).contiguous()
    dQ_hyb = dQ_p.permute(2, 0, 1, 3).contiguous()
    ddec_hyb = ddec_p.permute(2, 0, 1).contiguous()
    dS0_hyb = dS0_p.transpose(-1, -2).contiguous()

    def rel(a, b, tag):
        num = (a.float() - b.float()).abs().max().item()
        denom = max(b.float().abs().max().item(), 1e-10)
        r = num / denom
        status = "PASS" if r < 1e-3 else "FAIL"
        print(f"    {tag:>8s}: max_rel={r:.2e}  |a|={a.float().abs().max().item():.3e} |b|={b.float().abs().max().item():.3e} [{status}]")

    print(f"\n  B={B} H={H} T={T} N={N}  (fp32)")
    rel(dK_hyb, dK_ref, "dK")
    rel(dV_hyb, dV_ref, "dV")
    rel(dQ_hyb, dQ_ref, "dQ")
    rel(ddec_hyb, ddec_ref, "ddecay")
    rel(dS0_hyb, dS0_ref, "dS0")


if __name__ == '__main__':
    print("CUDA-convention autograd vs Pararnn-backward-with-transposes\n")
    for shape in [(1, 2, 64, 16), (1, 4, 256, 16), (1, 4, 256, 32)]:
        try:
            test(*shape)
        except Exception as e:
            import traceback
            traceback.print_exc()
