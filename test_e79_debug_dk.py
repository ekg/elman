"""Debug d_k computation for E79 input-bias CUDA."""
import sys
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, '/home/erikg/elman/elman/cuda')

import torch
import torch.nn.functional as F
import hasty_pytorch_lib

def test_debug_dk():
    print("E79 Debug d_k Computation")
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

    # Run CUDA forward
    S_cuda, M_cuda, output_cuda, kvqm_cache, bs_cache, bm_cache, S_checkpoints, M_checkpoints, \
        Sq_cache, s_row_cache, s_col_cache, m_row_cache, m_col_cache = \
        hasty_pytorch_lib.e79_coupled_input_bias_forward(
            True, x, S0, M0, W_kvqm, W_bs, W_bm
        )

    # Python forward to get expected values
    x_flat = x.reshape(T * B, dim)
    kvqm_all = (x_flat @ W_kvqm.T).reshape(T, B, 4 * n)
    bs_all = (x_flat @ W_bs.T).reshape(T, B, n)
    bm_all = (x_flat @ W_bm.T).reshape(T, B, n)

    S = S0[0].clone()
    M = M0[0].clone()

    # Store intermediate states
    S_before_list = []
    M_before_list = []
    s_row_list = []
    s_col_list = []

    for t in range(T):
        k = kvqm_all[t, 0, :n]
        v = kvqm_all[t, 0, n:2*n]
        q = kvqm_all[t, 0, 2*n:3*n]
        m_vec = kvqm_all[t, 0, 3*n:]
        b_s = bs_all[t, 0]
        b_m = bm_all[t, 0]

        k_norm = k / (k.norm() + 1e-6)
        m_norm = m_vec / (m_vec.norm() + 1e-6)

        # Save states BEFORE update
        S_before_list.append(S.clone())
        M_before_list.append(M.clone())

        s_row_decay = torch.sigmoid((M @ k_norm) + b_s)
        s_col_decay = torch.sigmoid((M.T @ k_norm) + b_s)
        s_row_list.append(s_row_decay.clone())
        s_col_list.append(s_col_decay.clone())

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

    print("=== Verifying cached values for t=1 ===")
    t = 1
    print(f"Python s_row_decay[t=1][:4]: {s_row_list[t][:4].tolist()}")
    print(f"CUDA s_row_cache[t=1][:4]: {s_row_cache[t,0,:4].tolist()}")
    print(f"s_row_decay diff: {(s_row_list[t] - s_row_cache[t,0]).abs().max().item():.2e}")

    print(f"\nPython s_col_decay[t=1][:4]: {s_col_list[t][:4].tolist()}")
    print(f"CUDA s_col_cache[t=1][:4]: {s_col_cache[t,0,:4].tolist()}")
    print(f"s_col_decay diff: {(s_col_list[t] - s_col_cache[t,0]).abs().max().item():.2e}")

    print(f"\nPython M_before[t=1][0,:4]: {M_before_list[t][0,:4].tolist()}")
    print(f"CUDA M_checkpoints[0][0,0,:4]: {M_checkpoints[0,0,0,:4].tolist()}")

    # M_before[t=1] should be M after timestep 0
    # The checkpoint at seg=0 is M0 = zeros
    # After forward re-run of timestep 0, M should be M_before[t=1]
    print(f"\nNote: CUDA checkpoint stores M0 (zeros), then re-runs timestep 0 to get M_before[t=1]")

    # Let's compute what CUDA's M should be at t=1
    # Checkpoint seg=0 stores M0 = zeros
    # Forward re-run tt=0: M updates from zeros to M_1
    # Forward re-run tt=1: M doesn't update, stays at M_1
    # So CUDA's M at t=1 gradient computation should be M_1 = M_before_list[1]
    print(f"\nExpected M at t=1 gradient computation (M_1):")
    print(f"  M_before_list[1][0,:4] = {M_before_list[1][0,:4].tolist()}")

    # Now let's verify the gate gradients
    # For simplicity, run Python backward to get expected d_s_row_decay, d_s_col_decay
    kvqm_py = kvqm_all.clone().requires_grad_(True)
    kvqm_py.retain_grad()
    bs_py = bs_all.clone().requires_grad_(True)
    bs_py.retain_grad()
    bm_py = bm_all.clone().requires_grad_(True)
    bm_py.retain_grad()

    S = S0[0].clone()
    M = M0[0].clone()
    outputs = []

    # Track tensors for gradient inspection
    s_row_decays = []
    s_col_decays = []

    for t in range(T):
        k = kvqm_py[t, 0, :n]
        v = kvqm_py[t, 0, n:2*n]
        q = kvqm_py[t, 0, 2*n:3*n]
        m_vec = kvqm_py[t, 0, 3*n:]
        b_s = bs_py[t, 0]
        b_m = bm_py[t, 0]

        k_norm = k / (k.norm() + 1e-6)
        m_norm = m_vec / (m_vec.norm() + 1e-6)

        s_row_decay = torch.sigmoid((M @ k_norm) + b_s)
        s_col_decay = torch.sigmoid((M.T @ k_norm) + b_s)
        s_row_decay.retain_grad()
        s_col_decay.retain_grad()
        s_row_decays.append(s_row_decay)
        s_col_decays.append(s_col_decay)

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

    print(f"\n=== Gate gradients at t=1 ===")
    print(f"Python d_s_row_decay[t=1][:4]: {s_row_decays[1].grad[:4].tolist()}")
    print(f"Python d_s_col_decay[t=1][:4]: {s_col_decays[1].grad[:4].tolist()}")

    # Compute what the d_k_norm contribution from gates should be
    d_s_row = s_row_decays[1].grad
    d_s_col = s_col_decays[1].grad
    s_row = s_row_list[1]
    s_col = s_col_list[1]
    M_before = M_before_list[1]

    d_pre_s_row = d_s_row * s_row * (1 - s_row)
    d_pre_s_col = d_s_col * s_col * (1 - s_col)

    # d_k_norm += M_before.T @ d_pre_s_row + M_before @ d_pre_s_col
    d_k_norm_from_s_row = M_before.T @ d_pre_s_row
    d_k_norm_from_s_col = M_before @ d_pre_s_col

    print(f"\nd_k_norm contribution from gates (expected):")
    print(f"  from s_row[:4]: {d_k_norm_from_s_row[:4].tolist()}")
    print(f"  from s_col[:4]: {d_k_norm_from_s_col[:4].tolist()}")
    print(f"  total[:4]: {(d_k_norm_from_s_row + d_k_norm_from_s_col)[:4].tolist()}")

    # Check M_before_list[1] is non-zero (it should be zeros because M0=zeros and t=0 doesn't update M since there's no dM)
    print(f"\nM_before[t=1] norm: {M_before.norm().item():.6f}")
    print(f"M_before[t=1] is zero: {(M_before == 0).all().item()}")

    # Wait, M_before[t=1] = M after timestep 0
    # At timestep 0, M starts at zeros and gets updated by:
    # M = m_row_decay * M * m_col_decay + outer(m_delta, m_norm)
    # Since M=0, the decay term is 0, so M = outer(m_delta, m_norm)
    # m_delta = s_delta - m_retrieved = s_delta - M@m_norm = s_delta - 0 = s_delta
    # So M_1 = outer(s_delta_0, m_norm_0)

    print(f"\nActually, M_1 should be outer(s_delta_0, m_norm_0):")
    k0 = kvqm_all[0, 0, :n]
    v0 = kvqm_all[0, 0, n:2*n]
    m_vec0 = kvqm_all[0, 0, 3*n:]
    k_norm0 = k0 / (k0.norm() + 1e-6)
    m_norm0 = m_vec0 / (m_vec0.norm() + 1e-6)
    S0_mat = S0[0]
    s_delta_0 = v0 - S0_mat @ k_norm0
    M_1_expected = torch.outer(s_delta_0, m_norm0)
    print(f"  M_1_expected[0,:4]: {M_1_expected[0,:4].tolist()}")
    print(f"  M_before[1][0,:4]: {M_before[0,:4].tolist()}")
    print(f"  Diff: {(M_1_expected - M_before).abs().max().item():.2e}")

if __name__ == "__main__":
    test_debug_dk()
