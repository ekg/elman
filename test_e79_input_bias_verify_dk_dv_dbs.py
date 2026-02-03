"""Verify d_k, d_v, and d_bs at t=1."""
import sys
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, '/home/erikg/elman/elman/cuda')

import torch
import torch.nn.functional as F
import hasty_pytorch_lib

def test_dk_dv_dbs():
    print("E79 Verify d_k, d_v, d_bs at t=1")
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
    kvqm_py = kvqm_all.clone().requires_grad_(True)
    kvqm_py.retain_grad()
    bs_py = bs_all.clone().requires_grad_(True)
    bs_py.retain_grad()
    bm_py = bm_all.clone().requires_grad_(True)
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

    print("Python gradients at t=1:")
    print(f"  d_k[:4]: {d_kvqm_py[1,0,:4].tolist()}")
    print(f"  d_v[:4]: {d_kvqm_py[1,0,n:n+4].tolist()}")
    print(f"  d_q[:4]: {d_kvqm_py[1,0,2*n:2*n+4].tolist()}")
    print(f"  d_m[:4]: {d_kvqm_py[1,0,3*n:3*n+4].tolist()}")
    print(f"  d_bs[:4]: {d_bs_py[1,0,:4].tolist()}")
    print(f"  d_bm[:4]: {d_bm_py[1,0,:4].tolist()}")

    # Now CUDA
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

    # Try to back out CUDA d_kvqm from dx_cuda
    # dx = d_kvqm @ W_kvqm + d_bs @ W_bs + d_bm @ W_bm
    # This is underdetermined, but let's check the total d_x

    print(f"\n=== d_x comparison at t=1 ===")
    d_x_py_from_kvqm = d_kvqm_py[1,0] @ W_kvqm
    d_x_py_from_bs = d_bs_py[1,0] @ W_bs
    d_x_py_from_bm = d_bm_py[1,0] @ W_bm
    d_x_py_total = d_x_py_from_kvqm + d_x_py_from_bs + d_x_py_from_bm

    print(f"Python d_x[t=1] from kvqm[:8]: {d_x_py_from_kvqm[:8].tolist()}")
    print(f"Python d_x[t=1] from bs[:8]: {d_x_py_from_bs[:8].tolist()}")
    print(f"Python d_x[t=1] from bm[:8]: {d_x_py_from_bm[:8].tolist()}")
    print(f"Python d_x[t=1] total[:8]: {d_x_py_total[:8].tolist()}")
    print(f"CUDA d_x[t=1][:8]: {dx_cuda[1,0,:8].tolist()}")

    diff = dx_cuda[1,0] - d_x_py_total
    print(f"\nDiff[:8]: {diff[:8].tolist()}")
    print(f"Max diff: {diff.abs().max().item():.2e}")

    # Check if the diff could come from d_bs/d_bm
    # If CUDA d_bs differs from Python, then d_x diff = (d_bs_cuda - d_bs_py) @ W_bs
    print(f"\n=== Weight gradient comparison ===")
    print(f"dW_kvqm CUDA shape: {dW_kvqm_cuda.shape}")
    print(f"dW_kvqm diff: {(dW_kvqm_cuda - cell.W_kvqm.grad).abs().max().item():.2e}")

    # Compute expected dW_kvqm from Python
    # dW_kvqm = sum_t (x[t] @ d_kvqm[t].T)
    dW_kvqm_expected = torch.zeros_like(W_kvqm)
    for t in range(T):
        dW_kvqm_expected += torch.outer(d_kvqm_py[t,0], x[t,0]).T  # [4n, dim].T = [dim, 4n] but W is [4n, dim]
        # Actually: dW_kvqm[i,j] = sum_t d_kvqm[t,i] * x[t,j]
        # So dW_kvqm += d_kvqm[t].unsqueeze(1) @ x[t].unsqueeze(0) -- no that's wrong too
        # W_kvqm is [4n, dim], kvqm = x @ W.T = [B, 4n]
        # d_kvqm @ W + kvqm @ dW.T = 0 (if we're computing dL/dW)
        # Actually: kvqm = x @ W.T, so dL/dW.T = dL/dkvqm^T @ x, so dL/dW = x^T @ dL/dkvqm = outer(x, d_kvqm)

    # dW_kvqm[i,j] = sum_t d_kvqm[t,i] * x[t,j]
    dW_kvqm_expected = torch.zeros(4*n, dim, device='cuda', dtype=torch.float32)
    for t in range(T):
        dW_kvqm_expected += torch.outer(d_kvqm_py[t,0], x[t,0])

    print(f"dW_kvqm expected[0,:8]: {dW_kvqm_expected[0,:8].tolist()}")
    print(f"dW_kvqm CUDA[0,:8]: {dW_kvqm_cuda[0,:8].tolist()}")
    print(f"dW_kvqm expected diff: {(dW_kvqm_expected - dW_kvqm_cuda).abs().max().item():.2e}")

if __name__ == "__main__":
    test_dk_dv_dbs()
