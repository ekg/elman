"""Verify PararnnHybridE88Function gradients against autograd-through-
Pararnn-sequential (same convention on both sides).

If this passes, our backward kernel is correct and the divergence in
phase5_validation.py is a CUDA-convention vs Pararnn-convention issue.
"""

import sys, os
import torch

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase1_reference import _random_inputs
from phase4_newton_driver import sequential_e88_forward, e88_step_batched
from phase7_fused_backward import backward_e88_fused_rank1
from phase2_warmup_scan import warmup_scan_boundaries
from phase1_warmstart_bench import admm_forward_fixed_iters
from phase5_validation import PararnnHybridE88Function


def autograd_forward(S0, K, V, decay, Q):
    """Pararnn sequential forward, autograd-compatible (Python loop).
    Returns S_final [B, H, N, N], output [T, B, H, N].
    K, V, Q: [T, B, H, N]
    decay: [T, B, H]
    S0: [B, H, N, N]
    """
    T = K.shape[0]
    B, H, N, _ = S0.shape
    S = S0
    outputs = []
    for t in range(T):
        # e88_step expects [B, H, N, N], [B, H, N], [B, H, N], [B, H]
        S = e88_step_batched(S, K[t], V[t], decay[t])
        # output[t] = S @ Q[t]  (matches Pararnn kernel: einsum('bhij,bhj->bhi'))
        out_t = torch.einsum('bhij,bhj->bhi', S, Q[t])
        outputs.append(out_t)
    output = torch.stack(outputs, dim=0)  # [T, B, H, N]
    return S, output


def test_hybrid_vs_autograd(B, H, T, N, P=16, W=128, seed=0):
    """Compare hybrid F/B vs autograd through Python sequential (same conv).

    Both sides use bf16 (forced by CUDA path in hybrid), so tolerance is
    relaxed to bf16 accumulation slack (~5%).  A true convention mismatch
    would appear as 50-500% error on gradients while forward still agrees,
    so even this loose tolerance separates signal from noise.
    """
    dt = torch.bfloat16
    g = torch.Generator(device='cuda').manual_seed(seed)
    k = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')).requires_grad_(True)
    v = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')).requires_grad_(True)
    q = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')).requires_grad_(True)
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, generator=g, dtype=dt, device='cuda')
                          ).detach().requires_grad_(True)
    S0 = (0.1 * torch.randn(B, H, N, N, generator=g, dtype=dt, device='cuda')).requires_grad_(True)

    # === Autograd through Python sequential (Pararnn convention) ===
    # Run in fp32 for reference stability, then compare at bf16 tolerance.
    k1 = k.clone().detach().float().requires_grad_(True)
    v1 = v.clone().detach().float().requires_grad_(True)
    q1 = q.clone().detach().float().requires_grad_(True)
    decay1 = decay.clone().detach().float().requires_grad_(True)
    S01 = S0.clone().detach().float().requires_grad_(True)
    S_final_ag, output_ag = autograd_forward(S01, k1, v1, decay1, q1)

    torch.manual_seed(seed + 1)
    fake_dL_dout = 0.01 * torch.randn_like(output_ag)
    fake_dL_dSfinal = 0.1 * torch.randn_like(S_final_ag)
    loss_ag = (output_ag * fake_dL_dout).sum() + (S_final_ag * fake_dL_dSfinal).sum()
    loss_ag.backward()

    dK_ag = k1.grad.clone()
    dV_ag = v1.grad.clone()
    dQ_ag = q1.grad.clone()
    ddec_ag = decay1.grad.clone()
    dS0_ag = S01.grad.clone()

    # === Our hybrid path (bf16, required by CUDA kernel) ===
    k2 = k.clone().detach().requires_grad_(True)
    v2 = v.clone().detach().requires_grad_(True)
    q2 = q.clone().detach().requires_grad_(True)
    decay2 = decay.clone().detach().requires_grad_(True)
    S02 = S0.clone().detach().requires_grad_(True)

    S_final_h, output_h = PararnnHybridE88Function.apply(True, k2, v2, q2, decay2, S02, H, P, W)

    loss_h = (output_h * fake_dL_dout.to(dt)).sum() + (S_final_h * fake_dL_dSfinal.to(dt)).sum()
    loss_h.backward()

    dK_h = k2.grad.clone()
    dV_h = v2.grad.clone()
    dQ_h = q2.grad.clone()
    ddec_h = decay2.grad.clone()
    dS0_h = S02.grad.clone()

    def rel(a, b, tag):
        num = (a.float() - b.float()).abs().max().item()
        denom = max(b.float().abs().max().item(), 1e-10)
        r = num / denom
        status = "PASS" if r < 0.10 else "FAIL"  # bf16 tolerance
        print(f"    {tag:>8s}: max_rel={r:.2e}  (|a|={a.abs().max().item():.2e}, |b|={b.abs().max().item():.2e})  [{status}]")
        return r

    print(f"\n  B={B} H={H} T={T} N={N}  (bf16, Pararnn convention on both sides)")
    print(f"    Forward:")
    rel(output_h, output_ag, "output")
    rel(S_final_h, S_final_ag, "S_final")
    print(f"    Gradients:")
    rel(dK_h, dK_ag, "dK")
    rel(dV_h, dV_ag, "dV")
    rel(dQ_h, dQ_ag, "dQ")
    rel(ddec_h, ddec_ag, "ddecay")
    rel(dS0_h, dS0_ag, "dS0")


if __name__ == '__main__':
    print("Phase 5 self-check — Hybrid vs autograd-through-Pararnn (same conv)\n")
    # Shape constraint: T >= P*W (each chunk needs >= W warmup positions)
    # P=16, W=128 → T >= 2048
    for shape in [(1, 2, 2048, 16), (1, 4, 4096, 16), (1, 4, 4096, 32)]:
        try:
            test_hybrid_vs_autograd(*shape)
        except Exception as e:
            import traceback
            print(f"  FAIL {shape}: {str(e)[:300]}")
            traceback.print_exc()
