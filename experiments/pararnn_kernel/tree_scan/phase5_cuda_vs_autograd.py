"""Check whether E88FLAHybridCUDAFunction backward matches fp32 autograd
through the same forward formula.

If CUDA matches autograd, then my hybrid (which matches autograd) should
also match CUDA.  If CUDA disagrees with autograd, then CUDA has its own
convention and we'd need to match that.
"""

import sys, os
import torch

sys.path.insert(0, '/home/erikg/elman')

from elman.models.e88_fla_hybrid import E88FLAHybridCUDAFunction


def cuda_step(S_prev, k, v, decay):
    """CUDA-convention step: S_new[i,j] = tanh(dec*S[i,j] + delta[j]*k[i])"""
    retrieved = torch.einsum('bhij,bhi->bhj', S_prev, k)
    delta = v - retrieved
    outer = torch.einsum('bhj,bhi->bhij', delta, k)
    return torch.tanh(decay[..., None, None] * S_prev + outer)


def cuda_forward_py(S0, K, V, decay, Q):
    """Python reference for CUDA forward."""
    T = K.shape[0]
    S = S0
    outputs = []
    for t in range(T):
        S = cuda_step(S, K[t], V[t], decay[t])
        outputs.append(torch.einsum('bhij,bhi->bhj', S, Q[t]))
    return S, torch.stack(outputs, 0)


def test(B, H, T, N, seed=0):
    # bf16 inputs
    dt = torch.bfloat16
    g = torch.Generator(device='cuda').manual_seed(seed)
    k = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda'))
    v = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda'))
    q = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda'))
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, generator=g, dtype=dt, device='cuda'))
    S0 = (0.1 * torch.randn(B, H, N, N, generator=g, dtype=dt, device='cuda'))

    # === fp32 autograd reference (through Python CUDA-convention forward) ===
    k_f = k.float().clone().requires_grad_(True)
    v_f = v.float().clone().requires_grad_(True)
    q_f = q.float().clone().requires_grad_(True)
    decay_f = decay.float().clone().requires_grad_(True)
    S0_f = S0.float().clone().requires_grad_(True)
    S_final_ref, output_ref = cuda_forward_py(S0_f, k_f, v_f, decay_f, q_f)

    torch.manual_seed(seed + 1)
    # CUDA backward ignores dS_final (only uses d_output), so set to zero
    # for fair comparison.
    dL_dout_f = 0.01 * torch.randn_like(output_ref)
    dL_dSfinal_f = torch.zeros_like(S_final_ref)
    loss_ref = (output_ref * dL_dout_f).sum() + (S_final_ref * dL_dSfinal_f).sum()
    loss_ref.backward()

    dK_ref, dV_ref, dQ_ref = k_f.grad.clone(), v_f.grad.clone(), q_f.grad.clone()
    ddec_ref = decay_f.grad.clone()
    dS0_ref = S0_f.grad.clone()

    # === CUDA (bf16) — cast same-seed grads down ===
    k1 = k.clone().requires_grad_(True)
    v1 = v.clone().requires_grad_(True)
    q1 = q.clone().requires_grad_(True)
    decay1 = decay.clone().requires_grad_(True)
    S01 = S0.clone().requires_grad_(True)
    S_final_c, output_c = E88FLAHybridCUDAFunction.apply(True, k1, v1, q1, decay1, S01, H)
    loss_c = (output_c * dL_dout_f.to(dt)).sum() + (S_final_c * dL_dSfinal_f.to(dt)).sum()
    loss_c.backward()

    dK_c = k1.grad.clone()
    dV_c = v1.grad.clone()
    dQ_c = q1.grad.clone()
    ddec_c = decay1.grad.clone()
    dS0_c = S01.grad.clone() if S01.grad is not None else None

    def rel(a, b, tag):
        if a is None or b is None:
            print(f"    {tag:>8s}: (one is None)")
            return
        num = (a.float() - b.float()).abs().max().item()
        denom = max(b.float().abs().max().item(), 1e-10)
        r = num / denom
        status = "PASS" if r < 0.10 else "FAIL"
        print(f"    {tag:>8s}: max_rel={r:.2e}  |cuda|={a.float().abs().max().item():.3e} |ref|={b.float().abs().max().item():.3e} [{status}]")

    print(f"\n  B={B} H={H} T={T} N={N}  (bf16 CUDA vs fp32 autograd through Python CUDA-conv)")
    # Also check outputs
    rel(output_c.permute(1, 2, 0, 3) if output_c.dim() == 4 else output_c,
        output_ref.permute(1, 2, 0, 3) if output_ref.dim() == 4 else output_ref, "output")
    rel(S_final_c, S_final_ref, "S_final")
    rel(dK_c, dK_ref, "dK")
    rel(dV_c, dV_ref, "dV")
    rel(dQ_c, dQ_ref, "dQ")
    rel(ddec_c, ddec_ref, "ddecay")
    rel(dS0_c, dS0_ref, "dS0")


if __name__ == '__main__':
    print("CUDA (bf16) vs fp32 autograd through Python CUDA-convention forward\n")
    for shape in [(1, 2, 64, 16), (1, 4, 256, 16), (1, 4, 256, 32)]:
        try:
            test(*shape)
        except Exception as e:
            import traceback
            traceback.print_exc()
