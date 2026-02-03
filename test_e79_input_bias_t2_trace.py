"""Trace E79 T=2 to find divergence at t=1."""
import sys
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, '/home/erikg/elman/elman/cuda')

import torch
import hasty_pytorch_lib

def test_t2_trace():
    print("E79 T=2 Trace Debug")
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

    # First, run CUDA to get its caches
    S_cuda, M_cuda, output_cuda, kvqm_cache, bs_cache, bm_cache, S_checkpoints, M_checkpoints, \
        Sq_cache, s_row_cache, s_col_cache, m_row_cache, m_col_cache = \
        hasty_pytorch_lib.e79_coupled_input_bias_forward(
            True, x, S0, M0, W_kvqm, W_bs, W_bm
        )

    # Now Python forward step by step
    S = S0[0].clone()
    M = M0[0].clone()

    for t in range(T):
        print(f"\n{'='*60}")
        print(f"Timestep t={t}")
        print(f"{'='*60}")

        k = kvqm_all[t, 0, :n]
        v = kvqm_all[t, 0, n:2*n]
        q = kvqm_all[t, 0, 2*n:3*n]
        m_vec = kvqm_all[t, 0, 3*n:]
        b_s = bs_all[t, 0]
        b_m = bm_all[t, 0]

        k_norm = k / (k.norm() + 1e-6)
        m_norm = m_vec / (m_vec.norm() + 1e-6)

        print(f"k_norm[:4]: {k_norm[:4].tolist()}")
        print(f"m_norm[:4]: {m_norm[:4].tolist()}")

        # S gates (using current M before update)
        s_row_decay = torch.sigmoid((M @ k_norm) + b_s)
        s_col_decay = torch.sigmoid((M.T @ k_norm) + b_s)

        print(f"\nPython s_row_decay[:4]: {s_row_decay[:4].tolist()}")
        print(f"CUDA   s_row_cache[{t}][:4]: {s_row_cache[t,0,:4].tolist()}")
        print(f"Diff: {(s_row_decay - s_row_cache[t,0]).abs().max().item():.2e}")

        # S update
        s_retrieved = S @ k_norm
        s_delta = v - s_retrieved
        S = (s_row_decay.unsqueeze(-1) * S * s_col_decay.unsqueeze(0)) + \
            torch.outer(s_delta, k_norm)

        print(f"\nS_updated[0,:4]: {S[0,:4].tolist()}")

        # M gates (using updated S)
        m_row_decay = torch.sigmoid((S @ m_norm) + b_m)
        m_col_decay = torch.sigmoid((S.T @ m_norm) + b_m)

        print(f"\nPython m_row_decay[:4]: {m_row_decay[:4].tolist()}")
        print(f"CUDA   m_row_cache[{t}][:4]: {m_row_cache[t,0,:4].tolist()}")
        print(f"Diff: {(m_row_decay - m_row_cache[t,0]).abs().max().item():.2e}")

        # M update
        m_retrieved = M @ m_norm
        m_delta = s_delta - m_retrieved
        M = (m_row_decay.unsqueeze(-1) * M * m_col_decay.unsqueeze(0)) + \
            torch.outer(m_delta, m_norm)

        print(f"\nM_updated[0,:4]: {M[0,:4].tolist()}")

    print(f"\n{'='*60}")
    print("Final comparison")
    print(f"{'='*60}")
    print(f"S diff: {(S_cuda[0] - S).abs().max().item():.2e}")
    print(f"M diff: {(M_cuda[0] - M).abs().max().item():.2e}")

    print(f"\nPython M[0,:4,:4]:")
    print(M[:4,:4])
    print(f"\nCUDA M[0,:4,:4]:")
    print(M_cuda[0,:4,:4])

if __name__ == "__main__":
    test_t2_trace()
