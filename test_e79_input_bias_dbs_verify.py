"""Verify d_bs computation by comparing the effect on d_x."""
import sys
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, '/home/erikg/elman/elman/cuda')

import torch
import torch.nn.functional as F
import hasty_pytorch_lib

def test_dbs_effect():
    print("E79 d_bs Effect on d_x")
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

    # Python forward with autograd
    kvqm_py = kvqm_all.clone().detach().requires_grad_(True)
    kvqm_py.retain_grad()
    bs_py = bs_all.clone().detach().requires_grad_(True)
    bs_py.retain_grad()
    bm_py = bm_all.clone().detach().requires_grad_(True)
    bm_py.retain_grad()

    S = S0[0].clone()
    M = M0[0].clone()
    outputs = []

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
    d_bs_py = bs_py.grad.clone()
    d_bm_py = bm_py.grad.clone()

    print("Python d_bs per timestep:")
    for t in range(T):
        print(f"  t={t}: {d_bs_py[t,0,:4].tolist()}")

    print("\nPython d_bm per timestep:")
    for t in range(T):
        print(f"  t={t}: {d_bm_py[t,0,:4].tolist()}")

    # Compute d_x contributions
    d_x_from_kvqm = d_kvqm_py.reshape(T*B, 4*n) @ W_kvqm
    d_x_from_bs = d_bs_py.reshape(T*B, n) @ W_bs
    d_x_from_bm = d_bm_py.reshape(T*B, n) @ W_bm

    print("\n=== d_x breakdown per timestep ===")
    for t in range(T):
        print(f"\nt={t}:")
        print(f"  from kvqm: {d_x_from_kvqm[t,:8].tolist()}")
        print(f"  from bs: {d_x_from_bs[t,:8].tolist()}")
        print(f"  from bm: {d_x_from_bm[t,:8].tolist()}")
        total = d_x_from_kvqm[t] + d_x_from_bs[t] + d_x_from_bm[t]
        print(f"  total: {total[:8].tolist()}")

    # Now CUDA
    print("\n" + "="*60)
    print("CUDA")

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

    print(f"\nCUDA d_x per timestep:")
    for t in range(T):
        print(f"  t={t}: {dx_cuda[t,0,:8].tolist()}")

    print(f"\n=== d_x diff ===")
    for t in range(T):
        d_x_py_total = d_x_from_kvqm[t] + d_x_from_bs[t] + d_x_from_bm[t]
        diff = (dx_cuda[t,0] - d_x_py_total).abs().max().item()
        print(f"  t={t}: {diff:.2e}")

    # Check if the diff is consistent with a d_bs error
    # If d_bs_cuda = 0 (missing), then d_x diff = -d_bs @ W_bs
    print(f"\n=== If d_bs were zero, d_x diff would be: ===")
    for t in range(T):
        d_x_if_no_bs = d_x_from_kvqm[t] + d_x_from_bm[t]
        diff_actual = (dx_cuda[t,0] - d_x_from_kvqm[t] - d_x_from_bs[t] - d_x_from_bm[t]).abs().max().item()
        diff_if_no_bs = (dx_cuda[t,0] - d_x_if_no_bs).abs().max().item()
        print(f"  t={t}: actual diff={diff_actual:.2e}, if no d_bs={diff_if_no_bs:.2e}")

if __name__ == "__main__":
    test_dbs_effect()
