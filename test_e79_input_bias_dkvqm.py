"""Compare d_kvqm between Python and CUDA."""
import sys
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, '/home/erikg/elman/elman/cuda')

import torch
import torch.nn.functional as F
import hasty_pytorch_lib

def test_dkvqm():
    print("E79 d_kvqm Comparison")
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

    # Python forward with hooks to capture gradients
    x_py = x.clone().requires_grad_(True)
    x_flat = x_py.reshape(T * B, dim)

    # Create kvqm_all with requires_grad to capture gradient
    kvqm_all = (x_flat @ W_kvqm.T).reshape(T, B, 4 * n)
    kvqm_all.retain_grad()

    bs_all = (x_flat @ W_bs.T).reshape(T, B, n)
    bs_all.retain_grad()

    bm_all = (x_flat @ W_bm.T).reshape(T, B, n)
    bm_all.retain_grad()

    # Run Python forward manually with autograd
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
        out = Sq * F.silu(Sq)
        outputs.append(out)

    output_py = torch.stack(outputs, dim=0).unsqueeze(1)  # [T, 1, n]
    loss = output_py.sum()
    loss.backward()

    d_kvqm_py = kvqm_all.grad.clone()
    d_bs_py = bs_all.grad.clone()
    d_bm_py = bm_all.grad.clone()

    print("Python d_kvqm per timestep:")
    for t in range(T):
        print(f"  t={t}:")
        print(f"    d_k[:4]: {d_kvqm_py[t,0,:4].tolist()}")
        print(f"    d_v[:4]: {d_kvqm_py[t,0,n:n+4].tolist()}")
        print(f"    d_q[:4]: {d_kvqm_py[t,0,2*n:2*n+4].tolist()}")
        print(f"    d_m[:4]: {d_kvqm_py[t,0,3*n:3*n+4].tolist()}")

    # Now CUDA
    print("\n" + "="*60)
    print("CUDA forward and backward")

    # Reset for CUDA
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

    # CUDA's d_kvqm can be inferred from dW_kvqm
    # dW_kvqm = sum_t,b (x[t,b] @ d_kvqm[t,b].T)
    # This is tricky to back out...

    # Instead, let's compute d_x from d_kvqm: d_x = d_kvqm @ W_kvqm
    # So: d_kvqm ≈ (d_x @ W_kvqm^-1) but W_kvqm is not square

    # Actually, we can estimate d_kvqm from the relationship:
    # d_x = d_kvqm @ W_kvqm + d_bs @ W_bs + d_bm @ W_bm

    # Let's just compare the final d_x per timestep
    print("\nComparing d_x per timestep:")
    print(f"d_x[t=0] diff: {(dx_cuda[0,0] - x_py.grad[0,0]).abs().max().item():.2e}")
    print(f"d_x[t=1] diff: {(dx_cuda[1,0] - x_py.grad[1,0]).abs().max().item():.2e}")

    print("\nCUDA d_x[t=1][:8]:", dx_cuda[1,0,:8].tolist())
    print("Python d_x[t=1][:8]:", x_py.grad[1,0,:8].tolist())

    # Check if the d_kvqm contribution explains the difference
    # d_x_from_kvqm = d_kvqm @ W_kvqm
    d_x_from_kvqm_py = (d_kvqm_py.reshape(T*B, 4*n) @ W_kvqm).reshape(T, B, dim)
    d_x_from_bs_py = (d_bs_py.reshape(T*B, n) @ W_bs).reshape(T, B, dim)
    d_x_from_bm_py = (d_bm_py.reshape(T*B, n) @ W_bm).reshape(T, B, dim)

    d_x_total_py = d_x_from_kvqm_py + d_x_from_bs_py + d_x_from_bm_py

    print(f"\nd_x from (d_kvqm + d_bs + d_bm) @ W:")
    print(f"  t=0 diff from autograd: {(d_x_total_py[0,0] - x_py.grad[0,0]).abs().max().item():.2e}")
    print(f"  t=1 diff from autograd: {(d_x_total_py[1,0] - x_py.grad[1,0]).abs().max().item():.2e}")

    print("\n=== d_bs and d_bm comparison ===")
    print(f"Python d_bs[t=1][:4]: {d_bs_py[1,0,:4].tolist()}")
    print(f"Python d_bm[t=1][:4]: {d_bm_py[1,0,:4].tolist()}")

    # Note: CUDA returns dW_bs, not d_bs
    # dW_bs = sum_t,b (x[t,b] @ d_bs[t,b].T)
    # We can't easily extract per-timestep d_bs from dW_bs

if __name__ == "__main__":
    test_dkvqm()
