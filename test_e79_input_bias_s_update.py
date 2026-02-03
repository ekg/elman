"""Debug E79 S update and m_norm to find divergence."""
import sys
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, '/home/erikg/elman/elman/cuda')

import torch
import hasty_pytorch_lib

def test_s_update():
    print("E79 S Update Debug")
    print("="*60)

    torch.manual_seed(42)
    T, B, dim, n_state = 1, 1, 32, 16  # Just T=1 to simplify

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

    k = kvqm_all[0, 0, :n]
    v = kvqm_all[0, 0, n:2*n]
    q = kvqm_all[0, 0, 2*n:3*n]
    m_vec = kvqm_all[0, 0, 3*n:]
    b_s = bs_all[0, 0]
    b_m = bm_all[0, 0]

    print(f"k[:4]: {k[:4].tolist()}")
    print(f"v[:4]: {v[:4].tolist()}")
    print(f"m_vec[:4]: {m_vec[:4].tolist()}")
    print(f"b_s[:4]: {b_s[:4].tolist()}")
    print(f"b_m[:4]: {b_m[:4].tolist()}")

    # Normalize
    k_norm = k / (k.norm() + 1e-6)
    m_norm = m_vec / (m_vec.norm() + 1e-6)

    print(f"\nk_norm[:4]: {k_norm[:4].tolist()}")
    print(f"m_norm[:4]: {m_norm[:4].tolist()}")

    # Python forward t=0
    S = S0[0].clone()  # [n, n]
    M = M0[0].clone()

    # S gates
    s_row_decay = torch.sigmoid((M @ k_norm) + b_s)
    s_col_decay = torch.sigmoid((M.T @ k_norm) + b_s)

    print(f"\nM @ k_norm (should be 0): {(M @ k_norm)[:4].tolist()}")
    print(f"s_row_decay (from Python)[:4]: {s_row_decay[:4].tolist()}")

    # S update
    s_retrieved = S @ k_norm  # [n]
    s_delta = v - s_retrieved  # [n]

    print(f"\ns_retrieved (should be 0)[:4]: {s_retrieved[:4].tolist()}")
    print(f"s_delta[:4]: {s_delta[:4].tolist()}")

    # S = s_row_decay[:, None] * S * s_col_decay[None, :] + s_delta[:, None] @ k_norm[None, :]
    S_updated = (s_row_decay.unsqueeze(-1) * S * s_col_decay.unsqueeze(0)) + \
                torch.outer(s_delta, k_norm)

    print(f"\nS_updated[0,:4]: {S_updated[0,:4].tolist()}")

    # M gates (using updated S)
    m_row_decay = torch.sigmoid((S_updated @ m_norm) + b_m)
    m_col_decay = torch.sigmoid((S_updated.T @ m_norm) + b_m)

    print(f"\nS_updated @ m_norm[:4]: {(S_updated @ m_norm)[:4].tolist()}")
    print(f"m_row_decay (Python)[:4]: {m_row_decay[:4].tolist()}")

    # Now CUDA
    print("\n" + "="*60)
    print("CUDA Forward")
    S_cuda, M_cuda, output_cuda, kvqm_cache, bs_cache, bm_cache, S_checkpoints, M_checkpoints, \
        Sq_cache, s_row_cache, s_col_cache, m_row_cache, m_col_cache = \
        hasty_pytorch_lib.e79_coupled_input_bias_forward(
            True, x, S0, M0, W_kvqm, W_bs, W_bm
        )

    print(f"s_row_cache[0,0,:4]: {s_row_cache[0,0,:4].tolist()}")
    print(f"m_row_cache[0,0,:4]: {m_row_cache[0,0,:4].tolist()}")
    print(f"\nS_cuda[0,0,:4]: {S_cuda[0,0,:4].tolist()}")
    print(f"M_cuda[0,0,:4]: {M_cuda[0,0,:4].tolist()}")

    print(f"\ns_row_decay diff: {(s_row_cache[0,0] - s_row_decay).abs().max().item():.2e}")
    print(f"m_row_decay diff: {(m_row_cache[0,0] - m_row_decay).abs().max().item():.2e}")
    print(f"S diff: {(S_cuda[0] - S_updated).abs().max().item():.2e}")

    # Check kvqm_cache to ensure projections match
    print(f"\nkvqm_cache shape: {kvqm_cache.shape}")
    print(f"kvqm_cache[0,0,:4] (k part): {kvqm_cache[0,0,:4].tolist()}")
    print(f"k from Python[:4]: {k[:4].tolist()}")

if __name__ == "__main__":
    test_s_update()
