"""Trace E79 backward pass to find gradient divergence at t=1."""
import sys
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, '/home/erikg/elman/elman/cuda')

import torch
import torch.nn.functional as F
import hasty_pytorch_lib

def test_backward_trace():
    print("E79 Backward Pass Trace")
    print("="*60)

    torch.manual_seed(42)
    T, B, dim, n_state = 2, 1, 32, 16

    x = torch.randn(T, B, dim, device='cuda', dtype=torch.float32)
    S0 = torch.zeros(B, n_state, n_state, device='cuda', dtype=torch.float32)
    M0 = torch.zeros(B, n_state, n_state, device='cuda', dtype=torch.float32)

    torch.manual_seed(42)
    from elman.models.e79_coupled_matrix import E79CoupledMatrixCell
    cell = E79CoupledMatrixCell(
        dim=dim,
        n_state=n_state,
        use_cuda=True,
        input_bias=True,
    ).cuda().float()

    W_kvqm = cell.W_kvqm.clone()
    W_bs = cell.W_bs.clone()
    W_bm = cell.W_bm.clone()

    n = n_state
    x_flat = x.reshape(T * B, dim)
    kvqm_all = (x_flat @ W_kvqm.T).reshape(T, B, 4 * n)
    bs_all = (x_flat @ W_bs.T).reshape(T, B, n)
    bm_all = (x_flat @ W_bm.T).reshape(T, B, n)

    # Run CUDA forward and backward
    S_cuda, M_cuda, output_cuda, kvqm_cache, bs_cache, bm_cache, S_checkpoints, M_checkpoints, \
        Sq_cache, s_row_cache, s_col_cache, m_row_cache, m_col_cache = \
        hasty_pytorch_lib.e79_coupled_input_bias_forward(
            True, x, S0, M0, W_kvqm, W_bs, W_bm
        )

    d_output = torch.ones_like(output_cuda)

    dx_cuda, dW_kvqm_cuda, dW_bs_cuda, dW_bm_cuda = hasty_pytorch_lib.e79_coupled_input_bias_backward(
        x, S_checkpoints, M_checkpoints, Sq_cache, kvqm_cache, bs_cache, bm_cache,
        s_row_cache, s_col_cache, m_row_cache, m_col_cache,
        d_output, W_kvqm, W_bs, W_bm
    )

    # Now compute Python backward manually
    # First run Python forward to get states
    S_states = [S0[0].clone()]
    M_states = [M0[0].clone()]
    S = S0[0].clone()
    M = M0[0].clone()
    outputs = []

    for t in range(T):
        k = kvqm_all[t, 0, :n]
        v = kvqm_all[t, 0, n:2*n]
        q = kvqm_all[t, 0, 2*n:3*n]
        m_vec = kvqm_all[t, 0, 3*n:]
        b_s = bs_all[t, 0]
        b_m = bm_all[t, 0]

        k_norm = k / (k.norm() + 1e-6)
        m_norm = m_vec / (m_vec.norm() + 1e-6)

        s_row_decay = torch.sigmoid((M @ k_norm) + b_s)
        s_col_decay = torch.sigmoid((M.T @ k_norm) + b_s)
        s_retrieved = S @ k_norm
        s_delta = v - s_retrieved
        S = (s_row_decay.unsqueeze(-1) * S * s_col_decay.unsqueeze(0)) + \
            torch.outer(s_delta, k_norm)

        m_row_decay = torch.sigmoid((S @ m_norm) + b_m)
        m_col_decay = torch.sigmoid((S.T @ m_norm) + b_m)
        m_retrieved = M @ m_norm
        m_delta = s_delta - m_retrieved
        M = (m_row_decay.unsqueeze(-1) * M * m_col_decay.unsqueeze(0)) + \
            torch.outer(m_delta, m_norm)

        Sq = S @ q
        out = Sq * F.silu(Sq)  # Sq * sigmoid(Sq) * Sq = Sq^2 * sigmoid(Sq)
        outputs.append(out)

        S_states.append(S.clone())
        M_states.append(M.clone())

    # Now backward pass manually for t=1
    print("\n=== Manual Backward for t=1 ===")
    t = 1
    dS = torch.zeros(n, n, device='cuda', dtype=torch.float32)
    dM = torch.zeros(n, n, device='cuda', dtype=torch.float32)

    # Get forward values at t=1
    S_before = S_states[t].clone()  # S after t=0, before t=1
    M_before = M_states[t].clone()

    k = kvqm_all[t, 0, :n]
    v = kvqm_all[t, 0, n:2*n]
    q = kvqm_all[t, 0, 2*n:3*n]
    m_vec = kvqm_all[t, 0, 3*n:]
    b_s = bs_all[t, 0]
    b_m = bm_all[t, 0]

    k_norm = k / (k.norm() + 1e-6)
    m_norm = m_vec / (m_vec.norm() + 1e-6)

    # Recompute forward values at t=1
    s_row_decay = torch.sigmoid((M_before @ k_norm) + b_s)
    s_col_decay = torch.sigmoid((M_before.T @ k_norm) + b_s)
    s_retrieved = S_before @ k_norm
    s_delta = v - s_retrieved

    S_t = (s_row_decay.unsqueeze(-1) * S_before * s_col_decay.unsqueeze(0)) + \
          torch.outer(s_delta, k_norm)

    m_row_decay = torch.sigmoid((S_t @ m_norm) + b_m)
    m_col_decay = torch.sigmoid((S_t.T @ m_norm) + b_m)
    m_retrieved = M_before @ m_norm
    m_delta = s_delta - m_retrieved

    Sq = S_t @ q
    sig_Sq = torch.sigmoid(Sq)

    # Gradient of output w.r.t Sq: d(Sq^2 * sig)/dSq
    d_out_t = torch.ones(n, device='cuda', dtype=torch.float32)
    d_Sq = d_out_t * (2 * Sq * sig_Sq + Sq * Sq * sig_Sq * (1 - sig_Sq))

    print(f"d_Sq[:4]: {d_Sq[:4].tolist()}")

    # dS from Sq = S_t @ q: dS_ij += d_Sq_i * q_j
    dS += torch.outer(d_Sq, q)

    # d_q from Sq = S_t @ q: d_q_j = sum_i S_t_ij * d_Sq_i
    d_q = S_t.T @ d_Sq

    print(f"d_q[:4] (from Sq): {d_q[:4].tolist()}")

    # Now we need to backprop through M update and S update
    # M_t = m_row_decay * M_before * m_col_decay + outer(m_delta, m_norm)

    # Since dM=0 (from future), no gradient flows through M_t directly
    # But dS might have contributions from M gates

    # Let's compute d_s_delta (from dS going through the S update)
    # S_t[i,j] = s_row_decay[i] * S_before[i,j] * s_col_decay[j] + s_delta[i] * k_norm[j]
    # dS_t -> d_s_delta[i] = sum_j dS_t[i,j] * k_norm[j]
    d_s_delta = dS @ k_norm

    print(f"d_s_delta[:4]: {d_s_delta[:4].tolist()}")

    # d_s_delta = d(v - s_retrieved) = -d_s_retrieved
    # s_retrieved[i] = sum_j S_before[i,j] * k_norm[j]
    # d_k_norm from s_retrieved: d_k_norm[j] = sum_i S_before[i,j] * (-d_s_delta[i])
    d_k_norm_from_sret = S_before.T @ (-d_s_delta)

    # d_v from s_delta
    d_v = d_s_delta

    print(f"d_v[:4]: {d_v[:4].tolist()}")

    # d_k_norm also from dS: dS_t[i,j] = ... + s_delta[i] * k_norm[j]
    # d_k_norm[j] += sum_i dS[i,j] * s_delta[i]
    d_k_norm_from_ds = dS.T @ s_delta

    d_k_norm = d_k_norm_from_sret + d_k_norm_from_ds

    print(f"d_k_norm[:4]: {d_k_norm[:4].tolist()}")

    # d_k from d_k_norm (normalize gradient)
    k_val = k.norm() + 1e-6
    d_k = d_k_norm / k_val - k * (k @ d_k_norm) / (k_val ** 3)

    print(f"d_k[:4]: {d_k[:4].tolist()}")

    # Compute d_kvqm = [d_k, d_v, d_q, d_m]
    # For m, we need to trace through m_delta
    # m_delta = s_delta - m_retrieved = s_delta - M_before @ m_norm
    # Since dM = 0, d_m_delta comes only from contribution to M update
    # But since there's no future dM, d_m_delta = 0

    # Actually, dM=0 means no gradient flows from future M
    # The contribution to d_m_norm comes from:
    # 1. M_t depends on m_norm (in the outer product term)
    # 2. m_row_decay/m_col_decay depend on m_norm (via S_t @ m_norm)

    # For t=T-1, dM=0, so:
    # d_m_delta = 0 (no contribution from dM)
    # d_m_norm = 0 (from dM)

    # Wait, but dS might contribute through the gate gradients
    # Let me reconsider...

    # The decay gates s_row_decay/s_col_decay depend on M_before @ k_norm
    # dS_t[i,j] = s_row_decay[i] * S_before[i,j] * s_col_decay[j] + s_delta[i] * k_norm[j]
    # d_s_row_decay[i] = sum_j dS_t[i,j] * S_before[i,j] * s_col_decay[j]
    d_s_row_decay = (dS * S_before * s_col_decay.unsqueeze(0)).sum(dim=1)
    d_s_col_decay = (dS * S_before * s_row_decay.unsqueeze(-1)).sum(dim=0)

    print(f"d_s_row_decay[:4]: {d_s_row_decay[:4].tolist()}")

    # s_row_decay = sigmoid(M_before @ k_norm + b_s)
    # d(M @ k) = sig * (1 - sig) * d_row_decay
    sig_s_row = s_row_decay * (1 - s_row_decay)
    d_Mk_row = sig_s_row * d_s_row_decay

    print(f"d_Mk (from row)[:4]: {d_Mk_row[:4].tolist()}")

    # For m_vec, need to trace through m_row_decay and m_delta
    # m_row_decay = sigmoid(S_t @ m_norm + b_m)
    # Since dM=0, d_m_row_decay = 0

    d_m = torch.zeros_like(m_vec)  # Simplified - need to properly compute

    # The full d_kvqm for t=1
    d_kvqm_t1_python = torch.cat([d_k, d_v, d_q, d_m])

    print(f"\n=== Compare d_kvqm at t=1 ===")
    # We don't have direct access to CUDA's intermediate d_kvqm
    # But we can compute d_x from d_kvqm: d_x = d_kvqm @ W_kvqm
    d_x_t1_from_d_kvqm = d_kvqm_t1_python @ W_kvqm

    print(f"d_x[t=1] from manual d_kvqm[:8]: {d_x_t1_from_d_kvqm[:8].tolist()}")
    print(f"d_x CUDA[t=1,0,:8]: {dx_cuda[1,0,:8].tolist()}")

    # Full Python backward with autograd
    x_py = x.clone().requires_grad_(True)
    cell.use_cuda_input_bias = False
    out_py, _, _ = cell(x_py, S0.clone(), M0.clone())
    loss_py = out_py.sum()
    loss_py.backward()
    dx_py = x_py.grad.clone()

    print(f"d_x Python[t=1,0,:8]: {dx_py[1,0,:8].tolist()}")

    print(f"\nManual vs Python diff: {(d_x_t1_from_d_kvqm - dx_py[1,0]).abs().max().item():.2e}")
    print(f"CUDA vs Python diff: {(dx_cuda[1,0] - dx_py[1,0]).abs().max().item():.2e}")

if __name__ == "__main__":
    test_backward_trace()
