"""Verify d_q computation at t=1."""
import sys
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, '/home/erikg/elman/elman/cuda')

import torch
import torch.nn.functional as F
import hasty_pytorch_lib

def test_dq():
    print("E79 Verify d_q at t=1")
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

    # Run CUDA forward to get Sq_cache and decays
    S_cuda, M_cuda, output_cuda, kvqm_cache, bs_cache, bm_cache, S_checkpoints, M_checkpoints, \
        Sq_cache, s_row_cache, s_col_cache, m_row_cache, m_col_cache = \
        hasty_pytorch_lib.e79_coupled_input_bias_forward(
            True, x, S0, M0, W_kvqm, W_bs, W_bm
        )

    # Manually compute Python forward to get S_before_t1
    S = S0[0].clone()
    M = M0[0].clone()

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

        if t == 1:
            S_before_t1 = S.clone()
            s_row_t1 = s_row_decay.clone()
            s_col_t1 = s_col_decay.clone()
            s_delta_t1 = s_delta.clone()
            k_norm_t1 = k_norm.clone()
            q_t1 = q.clone()

        S = (s_row_decay.unsqueeze(-1) * S * s_col_decay.unsqueeze(0)) + \
            torch.outer(s_delta, k_norm)

        if t == 1:
            S_t1 = S.clone()

        m_row_decay = torch.sigmoid((S @ m_norm) + b_m)
        m_col_decay = torch.sigmoid((S.T @ m_norm) + b_m)
        m_retrieved = M @ m_norm
        m_delta = s_delta - m_retrieved
        M = (m_row_decay.unsqueeze(-1) * M * m_col_decay.unsqueeze(0)) + \
            torch.outer(m_delta, m_norm)

    # Compute d_Sq at t=1
    Sq_t1 = S_t1 @ q_t1
    sig_Sq = torch.sigmoid(Sq_t1)
    d_output_t1 = torch.ones(n, device='cuda', dtype=torch.float32)
    d_Sq = d_output_t1 * (2 * Sq_t1 * sig_Sq + Sq_t1 * Sq_t1 * sig_Sq * (1 - sig_Sq))

    print(f"Sq_t1[:4]: {Sq_t1[:4].tolist()}")
    print(f"Sq_cache[t=1][:4]: {Sq_cache[1,0,:4].tolist()}")
    print(f"Sq diff: {(Sq_t1 - Sq_cache[1,0]).abs().max().item():.2e}")

    print(f"\nd_Sq[:4]: {d_Sq[:4].tolist()}")

    # d_q = S_t1.T @ d_Sq
    d_q_manual = S_t1.T @ d_Sq
    print(f"\nd_q manual[:4]: {d_q_manual[:4].tolist()}")

    # Also compute using the CUDA formula: S_t_ij computed inline
    # S_t_ij = s_row_decay[i] * S_before[i,j] * s_col_decay[j] + s_delta[i] * k_norm[j]
    d_q_cuda_formula = torch.zeros(n, device='cuda', dtype=torch.float32)
    for j in range(n):
        total = 0.0
        for i in range(n):
            S_t_ij = s_row_t1[i] * S_before_t1[i, j] * s_col_t1[j] + s_delta_t1[i] * k_norm_t1[j]
            total += S_t_ij * d_Sq[i]
        d_q_cuda_formula[j] = total

    print(f"d_q CUDA formula[:4]: {d_q_cuda_formula[:4].tolist()}")
    print(f"d_q diff (manual vs CUDA formula): {(d_q_manual - d_q_cuda_formula).abs().max().item():.2e}")

    # Verify S_t1 = s_row * S_before * s_col + s_delta ⊗ k_norm
    S_t1_verify = (s_row_t1.unsqueeze(-1) * S_before_t1 * s_col_t1.unsqueeze(0)) + \
                  torch.outer(s_delta_t1, k_norm_t1)
    print(f"\nS_t1 verify diff: {(S_t1 - S_t1_verify).abs().max().item():.2e}")

    # Check CUDA's checkpoints
    print(f"\n=== CUDA Checkpoints ===")
    print(f"S_checkpoints shape: {S_checkpoints.shape}")
    print(f"S_checkpoints[0,0,0,:4]: {S_checkpoints[0,0,0,:4].tolist()}")  # segment 0 (S0)
    print(f"S_checkpoints[1,0,0,:4]: {S_checkpoints[1,0,0,:4].tolist()}")  # segment 1 (after checkpt interval)

    # The checkpoint at segment 0 should be S0 = 0
    # If checkpoint_interval=16 and T=2, there's no checkpoint at t=16
    # So segment 0 checkpoint is S0, and we run forward from t=0 to reconstruct

if __name__ == "__main__":
    test_dq()
