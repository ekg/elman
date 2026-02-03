"""Debug E79 M matrix difference."""
import sys
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, '/home/erikg/elman/elman/cuda')

import torch
import hasty_pytorch_lib

def test_m_diff():
    print("E79 Input-Bias M Matrix Debug")
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

    # Projections
    n = n_state
    x_flat = x.reshape(T * B, dim)
    kvqm_all = (x_flat @ W_kvqm.T).reshape(T, B, 4 * n)
    bs_all = (x_flat @ W_bs.T).reshape(T, B, n)
    bm_all = (x_flat @ W_bm.T).reshape(T, B, n)

    k_all = kvqm_all[:, :, :n]
    v_all = kvqm_all[:, :, n:2*n]
    q_all = kvqm_all[:, :, 2*n:3*n]
    m_all = kvqm_all[:, :, 3*n:]

    # ========== Python Forward ==========
    print("\n=== Python Forward ===")
    S = S0.clone()
    M = M0.clone()

    for t in range(T):
        k = k_all[t]
        v = v_all[t]
        q = q_all[t]
        m_vec = m_all[t]
        b_s = bs_all[t]
        b_m = bm_all[t]

        k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
        m_norm = m_vec / (m_vec.norm(dim=-1, keepdim=True) + 1e-6)

        # S gates (using M)
        s_row_decay = torch.sigmoid(torch.einsum('bij,bj->bi', M, k_norm) + b_s)
        s_col_decay = torch.sigmoid(torch.einsum('bji,bj->bi', M, k_norm) + b_s)
        s_retrieved = torch.einsum('bij,bj->bi', S, k_norm)
        s_delta = v - s_retrieved

        # S update
        S = (s_row_decay.unsqueeze(-1) * S * s_col_decay.unsqueeze(1)) + \
            torch.einsum('bi,bj->bij', s_delta, k_norm)

        # M gates (using S)
        m_row_decay = torch.sigmoid(torch.einsum('bij,bj->bi', S, q) +
                                    torch.einsum('bij,bj->bi', M, m_norm) + b_m)
        m_col_decay = torch.sigmoid(torch.einsum('bji,bj->bi', S, q) +
                                    torch.einsum('bji,bj->bi', M, m_norm) + b_m)
        m_retrieved = torch.einsum('bij,bj->bi', M, m_norm)
        m_delta = s_delta - m_retrieved

        # M update
        M = (m_row_decay.unsqueeze(-1) * M * m_col_decay.unsqueeze(1)) + \
            torch.einsum('bi,bj->bij', m_delta, m_norm)

        print(f"t={t}:")
        print(f"  k_norm[0,:4]: {k_norm[0,:4].tolist()}")
        print(f"  m_norm[0,:4]: {m_norm[0,:4].tolist()}")
        print(f"  s_delta[0,:4]: {s_delta[0,:4].tolist()}")
        print(f"  m_delta[0,:4]: {m_delta[0,:4].tolist()}")
        print(f"  m_row_decay[0,:4]: {m_row_decay[0,:4].tolist()}")
        print(f"  m_col_decay[0,:4]: {m_col_decay[0,:4].tolist()}")
        print(f"  M[0,0,:4]: {M[0,0,:4].tolist()}")
        print(f"  S[0,0,:4]: {S[0,0,:4].tolist()}")

    S_py = S.clone()
    M_py = M.clone()

    # ========== CUDA Forward ==========
    print("\n=== CUDA Forward ===")
    S_cuda, M_cuda, output_cuda, kvqm_cache, bs_cache, bm_cache, S_checkpoints, M_checkpoints, \
        Sq_cache, s_row_cache, s_col_cache, m_row_cache, m_col_cache = \
        hasty_pytorch_lib.e79_coupled_input_bias_forward(
            True, x, S0, M0, W_kvqm, W_bs, W_bm
        )

    print(f"\nFinal S diff: {(S_cuda - S_py).abs().max().item():.2e}")
    print(f"Final M diff: {(M_cuda - M_py).abs().max().item():.2e}")

    # Check decay caches
    print(f"\nm_row_cache[0,0,:4]: {m_row_cache[0,0,:4].tolist()}")
    print(f"m_row_cache[1,0,:4]: {m_row_cache[1,0,:4].tolist()}")

    # Check S and M checkpoints
    print(f"\nS_checkpoints shape: {S_checkpoints.shape}")
    print(f"S_checkpoints[0,0,0,:4]: {S_checkpoints[0,0,0,:4].tolist()}")

    # Compare element by element
    print(f"\nM_py[0]:")
    print(M_py[0,:4,:4])
    print(f"\nM_cuda[0]:")
    print(M_cuda[0,:4,:4])

    print(f"\nM diff per element:")
    diff = (M_cuda - M_py).abs()
    print(diff[0,:4,:4])

if __name__ == "__main__":
    test_m_diff()
