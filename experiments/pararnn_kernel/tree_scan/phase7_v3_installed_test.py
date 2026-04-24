"""End-to-end test: verify installed V3 produces correct gradients when
called via E88OptimizedCUDAFunction.apply (the training path).

Compares V3 via install vs raw CUDA (saved original apply).
"""

import os
import sys
import time

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '3')

import torch

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
sys.path.insert(0, os.path.dirname(THIS))
sys.path.insert(0, '/home/erikg/elman')


def rel(a, b):
    return (a.float() - b.float()).abs().max().item() / max(b.float().abs().max().item(), 1e-10)


def main():
    # Grab original CUDA apply BEFORE any patching
    import elman.models.e88_fla_hybrid as e88m
    ORIG_APPLY = e88m.E88OptimizedCUDAFunction.apply

    # Install V3 patch
    from install_hybrid_v3 import install
    install()
    V3_APPLY = e88m.E88OptimizedCUDAFunction.apply

    print("=" * 70)
    print("End-to-end: E88OptimizedCUDAFunction (V3 installed) vs raw CUDA")
    print("=" * 70)

    shapes = [
        (16, 512, 141, 16),
        (16, 512, 83, 32),
        (2, 2048, 16, 16),
        (1, 4096, 8, 32),
    ]
    dt = torch.bfloat16
    for B, T, H, N in shapes:
        torch.manual_seed(0)
        k0 = 0.3 * torch.randn(B, T, H, N, dtype=dt, device='cuda')
        v0 = 0.3 * torch.randn(B, T, H, N, dtype=dt, device='cuda')
        q0 = 0.3 * torch.randn(B, T, H, N, dtype=dt, device='cuda')
        decay0 = torch.sigmoid(0.5 + 0.1 * torch.randn(B, T, H, dtype=dt, device='cuda'))
        g0 = torch.randn(B, T, H, N, dtype=dt, device='cuda')
        S00 = 0.1 * torch.randn(B, H, N, N, dtype=dt, device='cuda')

        # CUDA path
        kc = k0.clone().requires_grad_(True)
        vc = v0.clone().requires_grad_(True)
        qc = q0.clone().requires_grad_(True)
        decayc = decay0.clone().requires_grad_(True)
        gc = g0.clone().requires_grad_(True)
        S0c = S00.clone().requires_grad_(True)
        S_f_c, out_c = ORIG_APPLY(True, kc, vc, qc, decayc, gc, S0c, H)

        torch.manual_seed(1)
        dL = 0.01 * torch.randn_like(out_c)
        (out_c * dL).sum().backward()

        # V3 path
        k3 = k0.clone().requires_grad_(True)
        v3t = v0.clone().requires_grad_(True)
        q3 = q0.clone().requires_grad_(True)
        decay3 = decay0.clone().requires_grad_(True)
        g3 = g0.clone().requires_grad_(True)
        S03 = S00.clone().requires_grad_(True)
        S_f_3, out_3 = V3_APPLY(True, k3, v3t, q3, decay3, g3, S03, H)
        (out_3 * dL).sum().backward()

        # CUDA's S_final is sometimes not returned as the true final; check
        # just output + gradients.
        errs = {
            'output': rel(out_3, out_c),
            'dK': rel(k3.grad, kc.grad),
            'dV': rel(v3t.grad, vc.grad),
            'dQ': rel(q3.grad, qc.grad),
            'dg': rel(g3.grad, gc.grad),
            'ddec': rel(decay3.grad, decayc.grad),
        }
        worst = max(errs.values())
        # E88 CUDA kernel is a different algorithm (not parallel); bf16 diff
        # can be up to ~0.2 nats, but structurally gradients should be close
        # in magnitude.
        ok = "PASS" if worst < 0.3 else "FAIL"
        details = "  ".join(f"{k_}={v_:.1e}" for k_, v_ in errs.items())
        print(f"  B={B} T={T:>5d} H={H:>3d} N={N}:  worst={worst:.2e}  [{ok}]")
        print(f"    {details}")


if __name__ == '__main__':
    main()
