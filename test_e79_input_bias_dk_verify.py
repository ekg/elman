"""Verify d_k computation at t=1."""
import sys
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, '/home/erikg/elman/elman/cuda')

import torch
import torch.nn.functional as F

def test_dk():
    print("E79 Verify d_k at t=1")
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

    # Get Python gradients with autograd
    kvqm_py = kvqm_all.clone().detach().requires_grad_(True)
    kvqm_py.retain_grad()

    S = S0[0].clone()
    M = M0[0].clone()
    outputs = []

    for t in range(T):
        k = kvqm_py[t, 0, :n]
        v = kvqm_py[t, 0, n:2*n]
        q = kvqm_py[t, 0, 2*n:3*n]
        m_vec = kvqm_py[t, 0, 3*n:]
        b_s = bs_all[t, 0].detach()  # Don't track grad for simplicity
        b_m = bm_all[t, 0].detach()

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
        out = Sq * F.silu(Sq)
        outputs.append(out)

    output_py = torch.stack(outputs, dim=0)
    loss = output_py.sum()
    loss.backward()

    d_kvqm_py = kvqm_py.grad.clone()

    print("Python d_k at t=1 (first 8):", d_kvqm_py[1, 0, :8].tolist())

    # Now manually compute d_k for t=1
    # Reset state
    S = S0[0].clone()
    M = M0[0].clone()

    # Run t=0 to get state before t=1
    t = 0
    k = kvqm_all[t, 0, :n].detach()
    v = kvqm_all[t, 0, n:2*n].detach()
    q = kvqm_all[t, 0, 2*n:3*n].detach()
    m_vec = kvqm_all[t, 0, 3*n:].detach()
    b_s = bs_all[t, 0].detach()
    b_m = bm_all[t, 0].detach()

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

    # Now state is S_before_t1, M_before_t1
    S_before = S.clone()
    M_before = M.clone()

    # Compute forward at t=1
    t = 1
    k = kvqm_all[t, 0, :n].detach()
    v = kvqm_all[t, 0, n:2*n].detach()
    q = kvqm_all[t, 0, 2*n:3*n].detach()
    m_vec = kvqm_all[t, 0, 3*n:].detach()
    b_s = bs_all[t, 0].detach()
    b_m = bm_all[t, 0].detach()

    k_norm_val = k.norm() + 1e-6
    k_norm = k / k_norm_val
    m_norm = m_vec / (m_vec.norm() + 1e-6)

    s_row_decay = torch.sigmoid((M_before @ k_norm) + b_s)
    s_col_decay = torch.sigmoid((M_before.T @ k_norm) + b_s)
    s_retrieved = S_before @ k_norm
    s_delta = v - s_retrieved
    S_t = (s_row_decay.unsqueeze(-1) * S_before * s_col_decay.unsqueeze(0)) + \
          torch.outer(s_delta, k_norm)

    Sq = S_t @ q
    sig_Sq = torch.sigmoid(Sq)

    # Backward pass at t=1
    d_output = torch.ones(n, device='cuda', dtype=torch.float32)
    d_Sq = d_output * (2 * Sq * sig_Sq + Sq * Sq * sig_Sq * (1 - sig_Sq))

    # dS from Sq
    dS = torch.outer(d_Sq, q)

    # dL/ds_delta[i] = sum_j dS[i,j] * k_norm[j]
    d_s_delta = dS @ k_norm

    # dL/dk_norm from dS: dL/dk_norm[j] = sum_i dS[i,j] * s_delta[i]
    d_k_norm_from_ds = dS.T @ s_delta

    # dL/dk_norm from s_retrieved: s_retrieved[i] = sum_j S_before[i,j] * k_norm[j]
    # d_k_norm[j] += sum_i (-d_s_delta[i]) * S_before[i,j]
    d_k_norm_from_sret = S_before.T @ (-d_s_delta)

    # dL/ds_row_decay and dL/ds_col_decay from dS (through S_t update)
    # S_t[i,j] = s_row_decay[i] * S_before[i,j] * s_col_decay[j] + s_delta[i] * k_norm[j]
    # dL/ds_row_decay[i] = sum_j dS[i,j] * S_before[i,j] * s_col_decay[j]
    # dL/ds_col_decay[j] = sum_i dS[i,j] * S_before[i,j] * s_row_decay[i]
    d_s_row_decay = (dS * S_before * s_col_decay.unsqueeze(0)).sum(dim=1)
    d_s_col_decay = (dS * S_before * s_row_decay.unsqueeze(-1)).sum(dim=0)

    print(f"\nd_s_row_decay[:4]: {d_s_row_decay[:4].tolist()}")
    print(f"d_s_col_decay[:4]: {d_s_col_decay[:4].tolist()}")

    # s_row_decay[i] = sigmoid(sum_j M_before[i,j] * k_norm[j] + b_s[i])
    # dL/dk_norm[j] from s_row_decay = sum_i d_s_row_decay[i] * s_row_decay[i] * (1 - s_row_decay[i]) * M_before[i,j]
    d_pre_act_s_row = d_s_row_decay * s_row_decay * (1 - s_row_decay)
    d_k_norm_from_s_row = M_before.T @ d_pre_act_s_row  # sum over i

    # s_col_decay[j] = sigmoid(sum_i M_before[i,j] * k_norm[i] + b_s[j])
    # Note: M.T @ k_norm means (M.T)[j,i] * k_norm[i] = M[i,j] * k_norm[i]
    # So dL/dk_norm[i] from s_col_decay = sum_j d_pre_act_s_col[j] * M_before[i,j]
    d_pre_act_s_col = d_s_col_decay * s_col_decay * (1 - s_col_decay)
    d_k_norm_from_s_col = M_before @ d_pre_act_s_col  # M @ vec gives sum over j

    print(f"\nd_k_norm from s_row[:4]: {d_k_norm_from_s_row[:4].tolist()}")
    print(f"d_k_norm from s_col[:4]: {d_k_norm_from_s_col[:4].tolist()}")

    d_k_norm = d_k_norm_from_ds + d_k_norm_from_sret + d_k_norm_from_s_row + d_k_norm_from_s_col

    print(f"\nd_k_norm (before unnormalize)[:8]: {d_k_norm[:8].tolist()}")

    # Unnormalize: d_k = d_k_norm / norm - k * (k @ d_k_norm) / norm^3
    k_dot_dk = (k * d_k_norm).sum()
    d_k = d_k_norm / k_norm_val - k * k_dot_dk / (k_norm_val ** 3)

    print(f"d_k manual[:8]: {d_k[:8].tolist()}")

    # Check if this matches the CUDA formula
    # CUDA computes: d_k_raw[tid] = d_k_norm[tid] / norm - k_raw[tid] * k_dot_dk / norm3
    # where norm = k_norm_val, norm3 = norm^3, k_dot_dk = sum(k_raw * d_k_norm)

    print(f"\nComparison:")
    print(f"  Python autograd d_k[:8]: {d_kvqm_py[1, 0, :8].tolist()}")
    print(f"  Manual d_k[:8]: {d_k[:8].tolist()}")
    print(f"  Diff: {(d_k - d_kvqm_py[1, 0, :n]).abs().max().item():.2e}")

if __name__ == "__main__":
    test_dk()
