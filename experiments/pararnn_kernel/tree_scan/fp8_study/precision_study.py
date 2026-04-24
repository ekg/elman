"""Step 2: precision study — does fp8 round-tripping of S_traj compound
across T-steps, or does tanh's contraction damp the noise?

Strategy: a fp32 Python reference forward+backward, then repeat with the
same math but casting S to fp8-E4M3 and back at every step in fwd AND at
every read in bwd.  Compare gradients.
"""
import os, sys
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def fp32_forward(S0, K, V, Q, decay):
    """Pure fp32 sequential forward, stores S_1..S_T and returns outputs.
    Layout matches Pararnn: S is [B, H, M, N] (but M==N here).
    K, V, Q: [B, H, T, N]
    decay: [B, H, T]
    """
    B, H, T, N = K.shape
    S = S0.clone()
    S_traj = torch.empty(B, H, T, N, N, dtype=torch.float32, device=S0.device)
    outputs = torch.empty(B, H, T, N, dtype=torch.float32, device=S0.device)
    for t in range(T):
        K_t = K[:, :, t]  # [B, H, N]
        V_t = V[:, :, t]  # [B, H, N]
        dec = decay[:, :, t]  # [B, H]
        # retrieved[b,h,i] = sum_j S[b,h,i,j] * K_t[b,h,j]
        retrieved = torch.einsum('bhij,bhj->bhi', S, K_t)
        delta = V_t - retrieved
        # pre = dec * S + outer(delta, K)
        pre = dec[..., None, None] * S + delta[..., :, None] * K_t[..., None, :]
        S = torch.tanh(pre)
        S_traj[:, :, t] = S
        # output_t = S @ Q_t
        Q_t = Q[:, :, t]
        outputs[:, :, t] = torch.einsum('bhij,bhj->bhi', S, Q_t)
    return S_traj, outputs


def fp32_forward_with_fp8_storage(S0, K, V, Q, decay):
    """Same as fp32_forward but S_traj is stored as fp8_e4m3 and rehydrated
    as fp32 on read; forward computation uses fp32 from the rehydrated
    state for continuity AT THE NEXT STEP — modelling what the fp8-store
    kernel would do if it reuses the stored value.
    """
    B, H, T, N = K.shape
    S = S0.clone()
    # Cast S0 through fp8 to match: no — S0 is not stored in S_traj, it's
    # separate. Only subsequent S_t go through fp8.
    S_traj_fp8 = torch.empty(B, H, T, N, N, dtype=torch.float8_e4m3fn, device=S0.device)
    outputs = torch.empty(B, H, T, N, dtype=torch.float32, device=S0.device)
    for t in range(T):
        K_t = K[:, :, t]
        V_t = V[:, :, t]
        dec = decay[:, :, t]
        retrieved = torch.einsum('bhij,bhj->bhi', S, K_t)
        delta = V_t - retrieved
        pre = dec[..., None, None] * S + delta[..., :, None] * K_t[..., None, :]
        S_new = torch.tanh(pre)
        # Cast to fp8, then rehydrate (model what the next fwd step reads)
        S_fp8 = S_new.to(torch.float8_e4m3fn)
        S_traj_fp8[:, :, t] = S_fp8
        # In the Triton fwd kernel, S stays in registers across the T-loop
        # — there is NO re-read from HBM.  The fp8 rounding only affects
        # S_traj used by backward. So we DO NOT reload S here — fwd uses
        # the in-register fp32 value, bwd uses the fp8 stored copy.
        S = S_new
        # But output uses the current S directly
        Q_t = Q[:, :, t]
        outputs[:, :, t] = torch.einsum('bhij,bhj->bhi', S, Q_t)
    return S_traj_fp8, outputs


def fp32_backward_from_traj(S0, S_traj_fp32, K, V, Q, decay, g_T, dL_dout):
    """Reverse-scan backward, takes S_traj in fp32 (or fp8 rehydrated).
    Returns dK, dV, dQ, ddec, dS0.
    S_traj_fp32[:, :, t] = S_{t+1} (post-step state).
    """
    B, H, T, N = K.shape
    dK = torch.zeros_like(K, dtype=torch.float32)
    dV = torch.zeros_like(V, dtype=torch.float32)
    dQ = torch.zeros_like(Q, dtype=torch.float32)
    ddec = torch.zeros_like(decay, dtype=torch.float32)
    g = g_T.clone().to(torch.float32)

    for t in range(T - 1, -1, -1):
        dL_t = dL_dout[:, :, t]  # [B, H, N]
        Q_t = Q[:, :, t]
        g = g + dL_t[..., :, None] * Q_t[..., None, :]

        K_t = K[:, :, t]
        V_t = V[:, :, t]
        dec = decay[:, :, t]
        if t == 0:
            S_prev = S0
        else:
            S_prev = S_traj_fp32[:, :, t - 1]

        retrieved = torch.einsum('bhij,bhj->bhi', S_prev, K_t)
        delta = V_t - retrieved
        pre = dec[..., None, None] * S_prev + delta[..., :, None] * K_t[..., None, :]
        tanh_val = torch.tanh(pre)
        tanh_deriv = 1.0 - tanh_val * tanh_val

        # dQ_t[j] = sum_i dL_t[i] * tanh_val[i,j]
        dQ[:, :, t] = torch.einsum('bhi,bhij->bhj', dL_t, tanh_val)

        # u[i,j] = tanh_deriv[i,j] * K_t[j]
        u = tanh_deriv * K_t[..., None, :]
        # gu[i] = sum_j g[i,j] * u[i,j]
        gu = (g * u).sum(dim=-1)  # [B, H, N]

        dV[:, :, t] = gu

        g_times_td = g * tanh_deriv
        # dK_contrib[i,j] = delta[i] * g_times_td[i,j] - S_prev[i,j] * gu[i]
        dK_contrib = delta[..., :, None] * g_times_td - S_prev * gu[..., :, None]
        dK[:, :, t] = dK_contrib.sum(dim=-2)  # sum over i

        ddec[:, :, t] = (g_times_td * S_prev).sum(dim=(-1, -2))

        D_mat = dec[..., None, None] * tanh_deriv
        g = D_mat * g - K_t[..., None, :] * gu[..., :, None]

    dS0 = g
    return dK, dV, dQ, ddec, dS0


def rel_err(a, b):
    num = (a.float() - b.float()).abs().max().item()
    denom = max(b.float().abs().max().item(), 1e-10)
    return num / denom


def study_shape(B, H, T, N, seed=0):
    dev = 'cuda'
    torch.manual_seed(seed)
    K = 0.3 * torch.randn(B, H, T, N, dtype=torch.float32, device=dev)
    V = 0.3 * torch.randn(B, H, T, N, dtype=torch.float32, device=dev)
    Q = 0.3 * torch.randn(B, H, T, N, dtype=torch.float32, device=dev)
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(B, H, T, dtype=torch.float32, device=dev))
    S0 = 0.1 * torch.randn(B, H, N, N, dtype=torch.float32, device=dev)
    dL_dout = 0.01 * torch.randn(B, H, T, N, dtype=torch.float32, device=dev)
    g_T = torch.zeros(B, H, N, N, dtype=torch.float32, device=dev)

    # Ground truth: fp32 forward, fp32 traj, fp32 backward
    S_traj_gt, out_gt = fp32_forward(S0, K, V, Q, decay)
    dK_gt, dV_gt, dQ_gt, ddec_gt, dS0_gt = fp32_backward_from_traj(
        S0, S_traj_gt, K, V, Q, decay, g_T, dL_dout)

    # fp8 storage: forward uses full fp32 in-register; S_traj stored as fp8.
    # For the backward, read S_traj_fp8 and rehydrate to fp32.
    S_traj_fp8, out_fp8 = fp32_forward_with_fp8_storage(S0, K, V, Q, decay)
    S_traj_rehydrated = S_traj_fp8.to(torch.float32)

    # Measure how much fp8 perturbs S_traj itself
    s_traj_err = rel_err(S_traj_rehydrated, S_traj_gt)
    s_traj_max_abs = (S_traj_rehydrated - S_traj_gt).abs().max().item()

    # Backward with the fp8-rehydrated S_traj
    dK_fp8, dV_fp8, dQ_fp8, ddec_fp8, dS0_fp8 = fp32_backward_from_traj(
        S0, S_traj_rehydrated, K, V, Q, decay, g_T, dL_dout)

    errs = {
        'S_traj': s_traj_err,
        'S_traj_abs_max': s_traj_max_abs,
        'dK': rel_err(dK_fp8, dK_gt),
        'dV': rel_err(dV_fp8, dV_gt),
        'dQ': rel_err(dQ_fp8, dQ_gt),
        'ddec': rel_err(ddec_fp8, ddec_gt),
        'dS0': rel_err(dS0_fp8, dS0_gt),
    }
    return errs


if __name__ == '__main__':
    print("Precision study: fp8-E4M3 storage of S_traj, bwd gradient impact\n")
    # Keep B=1 H=4 for speed, vary T to see compounding.
    shapes = [(1, 4, 256, 16), (1, 4, 1024, 16), (1, 4, 4096, 16),
              (1, 4, 16384, 16), (1, 4, 32768, 16),
              (1, 4, 1024, 32), (1, 4, 4096, 32), (1, 4, 16384, 32)]
    print(f"{'B':>2} {'H':>3} {'T':>6} {'N':>3}  " + "  ".join(f"{k:>12}" for k in
          ['S_traj_rel', 'S_traj_abs', 'dK_rel', 'dV_rel', 'dQ_rel', 'ddec_rel', 'dS0_rel']))
    for B, H, T, N in shapes:
        errs = study_shape(B, H, T, N)
        row = f"{B:2d} {H:3d} {T:6d} {N:3d}  "
        row += f"{errs['S_traj']:12.3e}  {errs['S_traj_abs_max']:12.3e}  "
        row += f"{errs['dK']:12.3e}  {errs['dV']:12.3e}  {errs['dQ']:12.3e}  "
        row += f"{errs['ddec']:12.3e}  {errs['dS0']:12.3e}"
        print(row)
