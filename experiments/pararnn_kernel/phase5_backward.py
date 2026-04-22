"""Phase 5 — backward pass.

Given E88's forward trajectory {S_0, S_1, ..., S_T} and gradients
dL/dS_t (possibly only dL/dS_T), compute gradients w.r.t. inputs:
  dL/dK   [B, H, T, n]
  dL/dV   [B, H, T, n]
  dL/ddecay [B, H, T]
  dL/dS_0 [B, H, n, n]

Mathematical content:
  g_t = dL/dS_t  (shape [B, H, n, n], but gradient flows per row)
  backward: for t = T, T-1, ..., 1:
    g_{t-1} ← J_t^T · g_t  (plus external grad at t-1 if any)
  J_t^T for row i is  diag(D_t[i,:]) - v_t[i,:] ⊗ u_t[i,:] = diag(D_t[i]) - k_t ⊗ (tanh'⊙k)
    where u, v were defined in Phase 3 forward (u = tanh'⊙k, v = k).

Parameter gradients (summed over rows within a head):
  dV[t, i, :]       = sum_{row} g_t[i, :] · u_t[i, :]    (actually: dV[t, i] scalar per row i... let me recompute)

Actually V has shape [B, H, T, n] where V[t, i] is the scalar value for
row i at time t. Then
  dV[t, i] = sum_j  g_t[i, j] · tanh'(pre[i,j]) · k[t, j]
          = g_t[i, :] · u_t[i, :]

dK[t, l]  (shared across rows) depends on all rows:
  dK[t, l] = sum_i [ delta[i] · g_t[i, l] · tanh'(pre[i, l])
                    − S_{t-1}[i, l] · sum_j g_t[i, j] · tanh'(pre[i, j]) · k[j] ]
          = sum_i [ delta[i] · (g_t ⊙ tanh')[i, l]  −  S_{t-1}[i, l] · (g_t · u_t)[i] ]

ddecay[t]  (scalar per (b,h,t)):
  ddecay[t] = sum_i sum_j g_t[i, j] · tanh'(pre[i, j]) · S_{t-1}[i, j]
           = sum over (i,j) of  g_t ⊙ tanh' ⊙ S_{t-1}

Phase 5 step 1 validates these formulas against PyTorch autograd through
the sequential forward. Step 2 ports the reverse scan to Triton.
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase4_newton_driver import (
    e88_step_batched, sequential_e88_forward, newton_e88_triton
)


# -----------------------------------------------------------------------------
# Backward through the E88 recurrence, given converged forward trajectory.
# -----------------------------------------------------------------------------

def backward_e88(S_traj, K, V, decay, dL_dS_traj):
    """Compute parameter gradients via one reverse scan.

    Args:
      S_traj:   [B, H, T+1, n, n]  forward trajectory (S[..., 0] = S_0)
      K, V:     [B, H, T, n]       forward inputs
      decay:    [B, H, T]
      dL_dS_traj: [B, H, T+1, n, n]  gradients at each time (inc t=0)
                  The loss function contributes a gradient at every t
                  it depends on. Typically only dL/dS_T is nonzero if
                  loss is on the final state, but we allow full-traj.

    Returns:
      dL_dS0:    [B, H, n, n]
      dL_dK:     [B, H, T, n]
      dL_dV:     [B, H, T, n]
      dL_ddecay: [B, H, T]
    """
    B, H, T1, n, _ = S_traj.shape
    T = T1 - 1
    device = K.device
    dtype = K.dtype

    # Running gradient g_t = dL/dS_t. Start at t=T with the external grad there.
    g = dL_dS_traj[:, :, T].clone()   # [B, H, n, n]

    dL_dK = torch.zeros_like(K)
    dL_dV = torch.zeros_like(V)
    dL_ddecay = torch.zeros_like(decay)

    for t in range(T, 0, -1):
        # Recompute forward intermediates at step t (using S_{t-1})
        S_prev = S_traj[:, :, t - 1]                                 # [B, H, n, n]
        K_t = K[:, :, t - 1]                                         # [B, H, n]
        V_t = V[:, :, t - 1]                                         # [B, H, n]
        dec_t = decay[:, :, t - 1]                                   # [B, H]

        retrieved = torch.einsum('bhij,bhj->bhi', S_prev, K_t)       # [B, H, n]
        delta = V_t - retrieved                                       # [B, H, n]
        outer = torch.einsum('bhi,bhj->bhij', delta, K_t)            # [B, H, n, n]
        pre = dec_t[..., None, None] * S_prev + outer                # [B, H, n, n]
        tanh_deriv = 1.0 - torch.tanh(pre) ** 2                       # [B, H, n, n]

        # u = tanh' ⊙ k  (per row i: u_t[i, :] = tanh_deriv[i, :] * k_t)
        u = tanh_deriv * K_t[..., None, :]                           # [B, H, n, n]

        # Parameter gradients (sums over rows for shared params):
        # g_u_row = g_t · u_t  summed over col axis → per-row scalar per (b,h,i)
        gt_times_u_rowsum = (g * u).sum(dim=-1)                      # [B, H, n]

        # dV[b,h,t,i] = g_t[b,h,i,:] · u_t[b,h,i,:]
        dL_dV[:, :, t - 1] = gt_times_u_rowsum

        # ddecay[b,h,t] = sum over (i,j) of  g_t ⊙ tanh' ⊙ S_{t-1}
        dL_ddecay[:, :, t - 1] = (g * tanh_deriv * S_prev).sum(dim=(-1, -2))

        # dK[b,h,t,l] =
        #   sum_i delta[b,h,i] * (g_t ⊙ tanh')[b,h,i,l]
        #   - sum_i S_prev[b,h,i,l] · (g_t · u)[b,h,i]
        g_times_tanhd = g * tanh_deriv                                # [B, H, n, n]
        term_1 = torch.einsum('bhi,bhij->bhj', delta, g_times_tanhd)
        term_2 = torch.einsum('bhil,bhi->bhl', S_prev, gt_times_u_rowsum)
        dL_dK[:, :, t - 1] = term_1 - term_2

        # g_{t-1} = J_t^T · g_t + dL_dS_traj[:, :, t-1]
        # Applied per row: g_new[i, :] = diag(D[i,:]) · g[i, :] - v_t · (u[i,:] · g[i,:])
        #                              = D_row ⊙ g_row  -  k_t · (u_row · g_row)
        D = dec_t[..., None, None] * tanh_deriv                      # [B, H, n, n]
        u_dot_g_per_row = (u * g).sum(dim=-1)                        # [B, H, n]
        g_new = D * g - K_t[..., None, :] * u_dot_g_per_row[..., :, None]
        g = g_new + dL_dS_traj[:, :, t - 1]

    dL_dS0 = g                                                       # [B, H, n, n]

    return dL_dS0, dL_dK, dL_dV, dL_ddecay


# -----------------------------------------------------------------------------
# Validation against PyTorch autograd on the sequential forward.
# -----------------------------------------------------------------------------

def _autograd_forward(S0, K, V, decay):
    """Sequential forward built via list+stack so autograd can walk it."""
    B, H, T, n = K.shape
    S_list = [S0]
    for t in range(T):
        S_next = e88_step_batched(
            S_list[-1], K[:, :, t], V[:, :, t], decay[:, :, t]
        )
        S_list.append(S_next)
    return torch.stack(S_list, dim=2)   # [B, H, T+1, n, n]


def autograd_reference(S0, K, V, decay, dL_dS_traj):
    """Compute gradients via PyTorch autograd through an autograd-friendly forward."""
    S0 = S0.detach().clone().requires_grad_(True)
    K = K.detach().clone().requires_grad_(True)
    V = V.detach().clone().requires_grad_(True)
    decay = decay.detach().clone().requires_grad_(True)

    S_traj = _autograd_forward(S0, K, V, decay)
    loss = (dL_dS_traj * S_traj).sum()
    loss.backward()
    return S0.grad, K.grad, V.grad, decay.grad


def _random_case(B, H, T, n, seed=0, dtype=torch.float64, device='cuda'):
    torch.manual_seed(seed)
    S0 = 0.1 * torch.randn(B, H, n, n, dtype=dtype, device=device)
    K  = 0.3 * torch.randn(B, H, T, n, dtype=dtype, device=device)
    V  = 0.3 * torch.randn(B, H, T, n, dtype=dtype, device=device)
    decay = 0.9 + 0.1 * torch.rand(B, H, T, dtype=dtype, device=device)
    # random gradients at every t (including t=0 entry; in practice that's often 0)
    dL_dS_traj = 0.1 * torch.randn(B, H, T + 1, n, n, dtype=dtype, device=device)
    return S0, K, V, decay, dL_dS_traj


def test_backward(B, H, T, n, seed=0, dtype=torch.float64, device='cuda'):
    S0, K, V, decay, dL_dS_traj = _random_case(B, H, T, n, seed=seed, dtype=dtype, device=device)

    # Ground truth via autograd
    gS0_ag, gK_ag, gV_ag, gdec_ag = autograd_reference(S0, K, V, decay, dL_dS_traj)

    # Our manual backward, using the sequential forward's trajectory
    S_traj = sequential_e88_forward(S0, K, V, decay)
    gS0, gK, gV, gdec = backward_e88(S_traj, K, V, decay, dL_dS_traj)

    def rel_err(a, b):
        return (a - b).abs().max().item() / max(b.abs().max().item(), 1e-30)

    eS0 = rel_err(gS0, gS0_ag)
    eK  = rel_err(gK,  gK_ag)
    eV  = rel_err(gV,  gV_ag)
    ed  = rel_err(gdec, gdec_ag)
    worst = max(eS0, eK, eV, ed)
    status = "PASS" if worst < 1e-8 else "FAIL"
    print(f"  B={B} H={H:3d} T={T:3d} n={n:3d}  "
          f"relErr  S0={eS0:.2e}  K={eK:.2e}  V={eV:.2e}  decay={ed:.2e}  [{status}]")
    return worst


if __name__ == '__main__':
    print("Phase 5 step 1: manual backward matches PyTorch autograd (fp64).\n")
    for B, H, T, n in [(1, 1, 8, 4),
                       (1, 2, 16, 4),
                       (2, 4, 32, 8),
                       (1, 4, 64, 16),
                       (2, 8, 128, 32)]:
        test_backward(B, H, T, n)

    print("\nIf all PASS, the reverse-scan backward formulas are correct.")
    print("Next step: port the reverse scan to Triton (Phase 5 step 2).")
