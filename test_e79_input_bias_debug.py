"""Debug E79 input-bias gradient difference at t=1."""
import sys
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, '/home/erikg/elman/elman/cuda')

import torch
import hasty_pytorch_lib
from elman.models.e79_coupled_matrix import E79CoupledMatrixCell

def test_debug():
    print("E79 Input-Bias Debug Test T=2, B=1")
    print("="*60)

    torch.manual_seed(42)
    T, B, dim, n_state = 2, 1, 32, 16

    x = torch.randn(T, B, dim, device='cuda', dtype=torch.float32)
    S0 = torch.zeros(B, n_state, n_state, device='cuda', dtype=torch.float32)
    M0 = torch.zeros(B, n_state, n_state, device='cuda', dtype=torch.float32)

    torch.manual_seed(42)
    cell = E79CoupledMatrixCell(
        dim=dim,
        n_state=n_state,
        use_cuda=True,
        input_bias=True,
    ).cuda().float()

    W_kvqm = cell.W_kvqm.clone()
    W_bs = cell.W_bs.clone()
    W_bm = cell.W_bm.clone()

    # Manually compute projections
    T_size, B_size, D = x.shape
    n = n_state
    x_flat = x.reshape(T * B, D)
    kvqm_all = (x_flat @ W_kvqm.T).reshape(T, B, 4 * n)
    bs_all = (x_flat @ W_bs.T).reshape(T, B, n)
    bm_all = (x_flat @ W_bm.T).reshape(T, B, n)

    k_all = kvqm_all[:, :, :n]
    v_all = kvqm_all[:, :, n:2*n]
    q_all = kvqm_all[:, :, 2*n:3*n]
    m_all = kvqm_all[:, :, 3*n:]

    print(f"\n=== Forward Pass Manual (Python) ===")

    # Forward pass manually
    S = S0.clone()
    M = M0.clone()
    outputs_py = []
    S_states = [S.clone()]  # S_states[t] = S before timestep t
    M_states = [M.clone()]

    for t in range(T):
        k = k_all[t]
        v = v_all[t]
        q = q_all[t]
        m_vec = m_all[t]
        b_s = bs_all[t]
        b_m = bm_all[t]

        k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
        m_norm = m_vec / (m_vec.norm(dim=-1, keepdim=True) + 1e-6)

        # S gates
        s_row_decay = torch.sigmoid(torch.einsum('bij,bj->bi', M, k_norm) + b_s)
        s_col_decay = torch.sigmoid(torch.einsum('bji,bj->bi', M, k_norm) + b_s)

        # Delta
        s_retrieved = torch.einsum('bij,bj->bi', S, k_norm)
        s_delta = v - s_retrieved

        # S update
        S = (s_row_decay.unsqueeze(-1) * S * s_col_decay.unsqueeze(1)) + \
            torch.einsum('bi,bj->bij', s_delta, k_norm)

        # M gates
        m_row_decay = torch.sigmoid(torch.einsum('bij,bj->bi', S, q) +
                                    torch.einsum('bij,bj->bi', M, m_norm) + b_m)
        m_col_decay = torch.sigmoid(torch.einsum('bji,bj->bi', S, q) +
                                    torch.einsum('bji,bj->bi', M, m_norm) + b_m)

        m_retrieved = torch.einsum('bij,bj->bi', M, m_norm)
        m_delta = s_delta - m_retrieved

        M = (m_row_decay.unsqueeze(-1) * M * m_col_decay.unsqueeze(1)) + \
            torch.einsum('bi,bj->bij', m_delta, m_norm)

        # Output
        Sq = torch.einsum('bij,bj->bi', S, q)
        output = Sq * torch.sigmoid(Sq) * Sq  # Sq * silu(Sq) = Sq^2 * sigmoid(Sq)
        outputs_py.append(output)

        S_states.append(S.clone())
        M_states.append(M.clone())

        print(f"t={t}: S[0,0,0]={S[0,0,0].item():.6f}, M[0,0,0]={M[0,0,0].item():.6f}, "
              f"output[0,0]={output[0,0].item():.6f}")
        print(f"      s_row_decay[0,:4]={s_row_decay[0,:4].tolist()}")
        print(f"      Sq[0,:4]={Sq[0,:4].tolist()}")

    output_py = torch.stack(outputs_py, dim=0)

    # Now CUDA forward
    print(f"\n=== Forward Pass CUDA ===")
    S_cuda, M_cuda, output_cuda, kvqm_cache, bs_cache, bm_cache, S_checkpoints, M_checkpoints, \
        Sq_cache, s_row_cache, s_col_cache, m_row_cache, m_col_cache = \
        hasty_pytorch_lib.e79_coupled_input_bias_forward(
            True, x, S0, M0, W_kvqm, W_bs, W_bm
        )

    print(f"Output diff: {(output_cuda - output_py).abs().max().item():.2e}")
    print(f"S diff: {(S_cuda - S).abs().max().item():.2e}")
    print(f"M diff: {(M_cuda - M).abs().max().item():.2e}")

    print(f"\nSq_cache CUDA t=1: {Sq_cache[1,0,:4].tolist()}")
    print(f"s_row_cache CUDA t=1: {s_row_cache[1,0,:4].tolist()}")

    # Backward pass
    print(f"\n=== Backward Pass ===")
    d_output = torch.ones_like(output_cuda)

    # CUDA backward
    dx_cuda, dW_kvqm_cuda, dW_bs_cuda, dW_bm_cuda = hasty_pytorch_lib.e79_coupled_input_bias_backward(
        x, S_checkpoints, M_checkpoints, Sq_cache, kvqm_cache, bs_cache, bm_cache,
        s_row_cache, s_col_cache, m_row_cache, m_col_cache,
        d_output, W_kvqm, W_bs, W_bm
    )

    # Python backward using autograd
    x_py = x.clone().requires_grad_(True)
    cell.use_cuda_input_bias = False
    out_py, S_final_py, M_final_py = cell(x_py, S0.clone(), M0.clone())
    loss_py = out_py.sum()
    loss_py.backward()
    dx_py = x_py.grad.clone()
    dW_kvqm_py = cell.W_kvqm.grad.clone()
    dW_bs_py = cell.W_bs.grad.clone()
    dW_bm_py = cell.W_bm.grad.clone()

    print(f"\nd_x diff: {(dx_cuda - dx_py).abs().max().item():.2e}")
    print(f"d_W_kvqm diff: {(dW_kvqm_cuda - dW_kvqm_py).abs().max().item():.2e}")
    print(f"d_W_bs diff: {(dW_bs_cuda - dW_bs_py).abs().max().item():.2e}")
    print(f"d_W_bm diff: {(dW_bm_cuda - dW_bm_py).abs().max().item():.2e}")

    print(f"\nd_x[t=0] diff: {(dx_cuda[0] - dx_py[0]).abs().max().item():.2e}")
    print(f"d_x[t=1] diff: {(dx_cuda[1] - dx_py[1]).abs().max().item():.2e}")

    # Compute d_kvqm per-timestep from dx
    # d_kvqm = d_x @ W_kvqm (chain rule)
    # Actually need intermediate d_kvqm_all to debug

    # Let's compute expected d_kvqm for t=1 manually
    print(f"\n=== Manual Gradient Computation for t=1 ===")

    # At t=1: S is S_states[1] (S after t=0, before t=1)
    S_before_t1 = S_states[1]
    M_before_t1 = M_states[1]

    print(f"S_before_t1[0,0,:4]: {S_before_t1[0,0,:4].tolist()}")
    print(f"M_before_t1[0,0,:4]: {M_before_t1[0,0,:4].tolist()}")

    # Recompute values at t=1
    t = 1
    k = k_all[t]
    v = v_all[t]
    q = q_all[t]
    m_vec = m_all[t]
    b_s = bs_all[t]
    b_m = bm_all[t]

    k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
    m_norm = m_vec / (m_vec.norm(dim=-1, keepdim=True) + 1e-6)

    # Forward values at t=1
    s_row_decay = torch.sigmoid(torch.einsum('bij,bj->bi', M_before_t1, k_norm) + b_s)
    s_col_decay = torch.sigmoid(torch.einsum('bji,bj->bi', M_before_t1, k_norm) + b_s)
    s_retrieved = torch.einsum('bij,bj->bi', S_before_t1, k_norm)
    s_delta = v - s_retrieved

    S_t1 = (s_row_decay.unsqueeze(-1) * S_before_t1 * s_col_decay.unsqueeze(1)) + \
           torch.einsum('bi,bj->bij', s_delta, k_norm)

    m_row_decay = torch.sigmoid(torch.einsum('bij,bj->bi', S_t1, q) +
                                torch.einsum('bij,bj->bi', M_before_t1, m_norm) + b_m)
    m_col_decay = torch.sigmoid(torch.einsum('bji,bj->bi', S_t1, q) +
                                torch.einsum('bji,bj->bi', M_before_t1, m_norm) + b_m)
    m_retrieved = torch.einsum('bij,bj->bi', M_before_t1, m_norm)
    m_delta = s_delta - m_retrieved

    Sq = torch.einsum('bij,bj->bi', S_t1, q)
    sig_Sq = torch.sigmoid(Sq)
    output_t1 = Sq * sig_Sq * Sq  # Sq^2 * sigmoid(Sq)

    # Gradient of output w.r.t Sq
    # d(Sq^2 * sig)/dSq = 2*Sq*sig + Sq^2*sig*(1-sig)
    d_out_t1 = torch.ones_like(Sq)
    d_Sq = d_out_t1 * (2 * Sq * sig_Sq + Sq * Sq * sig_Sq * (1 - sig_Sq))

    print(f"\nSq[0,:4]: {Sq[0,:4].tolist()}")
    print(f"d_Sq[0,:4]: {d_Sq[0,:4].tolist()}")

    # d_q from d_Sq: d_Sq_i = sum_j S_t1_ij * q_j  ->  d_q_j = sum_i S_t1_ij * d_Sq_i
    d_q = torch.einsum('bij,bi->bj', S_t1, d_Sq)
    print(f"\nd_q[0,:4] (from Sq): {d_q[0,:4].tolist()}")

    # dS from d_Sq: dS_ij += d_Sq_i * q_j
    dS = torch.einsum('bi,bj->bij', d_Sq, q)
    print(f"dS[0,0,:4] (from Sq): {dS[0,0,:4].tolist()}")

    # At t=1, we also have gradients from dS propagated from later timesteps
    # But since T=2, there's no t=2, so dS and dM from future are 0

if __name__ == "__main__":
    test_debug()
