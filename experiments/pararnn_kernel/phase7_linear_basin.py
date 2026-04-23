"""Is the linearized warm-start inside Newton's basin of attraction?

Linear approx: S[t] = decay*S[t-1] + outer(V[t], K[t])  (drop tanh, drop -(k·s)k)
True:          S[t] = tanh(decay*S[t-1] + outer(V[t] - K·S[t-1], K[t]))

Measure: max|S_linear - S_true| across T. If within perturb ~ 0.1, Newton
from linear warm-start will converge in ~4 iters. If within 0.01, 2 iters.
"""

import sys, os, time
import torch
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase4_newton_driver import sequential_e88_forward
from phase7_fused_iter import fused_newton_iter_inplace


def linear_warmstart_pytorch(S0, K, V, decay):
    """Compute S_linear via sequential evaluation (not parallel, just for quality check).

    In production this would be a parallel-scan Triton kernel."""
    B, H, T, n = K.shape
    S_traj = torch.empty(B, H, T + 1, n, n, dtype=S0.dtype, device=S0.device)
    S_traj[:, :, 0] = S0
    for t in range(T):
        dec = decay[:, :, t, None, None]   # [B, H, 1, 1]
        V_t = V[:, :, t]                   # [B, H, n]
        K_t = K[:, :, t]                   # [B, H, n]
        outer_VK = torch.einsum('bhi,bhj->bhij', V_t, K_t)
        S_traj[:, :, t + 1] = dec * S_traj[:, :, t] + outer_VK
    return S_traj[:, :, 1:]


def measure(B, H, T, n):
    g = torch.Generator(device='cuda').manual_seed(0)
    dt = torch.float32
    S0 = 0.1 * torch.randn(B, H, n, n, generator=g, dtype=dt, device='cuda')
    K = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    V = 0.3 * torch.randn(B, H, T, n, generator=g, dtype=dt, device='cuda')
    decay = 0.9 + 0.1 * torch.rand(B, H, T, generator=g, dtype=dt, device='cuda')

    print(f"\nT={T} H={H} n={n}:")

    S_true = sequential_e88_forward(S0, K, V, decay)[:, :, 1:]
    S_linear = linear_warmstart_pytorch(S0, K, V, decay)

    # Quality of warm-start
    diff = (S_linear - S_true).abs()
    print(f"  |S_linear - S_true|:  max={diff.max().item():.3e}  "
          f"mean={diff.mean().item():.3e}  "
          f"max|S_true|={S_true.abs().max().item():.3e}")

    # Does Newton from S_linear converge?
    S_var = S_linear.clone()
    line = "  Newton from S_linear:       "
    for it in range(6):
        d = fused_newton_iter_inplace(S0, S_var, K, V, decay)
        line += f" it{it+1}|δ|={d:.2e}"
        if d < 1e-5:
            line += f" (converged)"
            break
    print(line)

    # Try tanh(S_linear) — clips to within (-1, 1)
    S_var = torch.tanh(S_linear).clone()
    line = "  Newton from tanh(S_linear): "
    for it in range(6):
        d = fused_newton_iter_inplace(S0, S_var, K, V, decay)
        line += f" it{it+1}|δ|={d:.2e}"
        if d < 1e-5:
            line += f" (converged)"
            break
    err = (S_var - S_true).abs().max().item()
    print(line + f"  (final err={err:.1e})")

    # Compare to perturbed S_true (realistic warm-start) for reference
    for pert_scale in [0.001, 0.01, 0.1]:
        S_test = S_true.clone() + pert_scale * torch.randn_like(S_true)
        d_max = None
        iters = 0
        for _ in range(6):
            d_max = fused_newton_iter_inplace(S0, S_test, K, V, decay)
            iters += 1
            if d_max < 1e-5:
                break
        err = (S_test - S_true).abs().max().item()
        print(f"    (ref pert={pert_scale}: {iters} iters → δ={d_max:.1e}, err={err:.1e})")


if __name__ == '__main__':
    for T in [1024, 8192]:
        for H in [16, 83]:
            try:
                measure(1, H, T, 32)
            except Exception as e:
                print(f"FAIL H={H} T={T}: {e}")
