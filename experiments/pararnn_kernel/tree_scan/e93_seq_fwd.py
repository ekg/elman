"""E93 forward Triton kernel — single rectangular state, no heads.

Per (b, m_tile), per t:
  retrieved_tile = S_tile^T @ k_t             [M_TILE]
  delta_tile = v_tile - retrieved_tile        [M_TILE]
  update_tile = k_t outer delta_tile          [N, M_TILE]
  Wh_S_tile = W_h @ S_tile                    [N, N] @ [N, M_TILE]   ← TC friendly
  S_tile = tanh(alpha · Wh_S_tile + update_tile)
  Sout_tile = S_tile flattened back

Each program handles ONE m-tile (column slice) for one batch element. Programs
are independent because columns of S are independent under W_h (which only
mixes rows) and under the rank-1 outer-product update.

Grid: (B * M_TILES,)  where M_TILES = M / M_TILE_SIZE.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _e93_fwd_kernel(
    S0_ptr, K_ptr, V_ptr, decay_ptr, W_h_ptr,
    S_traj_ptr,  # [B, T, N, M] flattened along N*M
    Sflat_ptr,   # [B, T, N*M]
    # K strides: [B, T, N]
    k_b, k_t, k_n,
    # V strides: [B, T, M]
    v_b, v_t, v_m,
    # decay strides: [B, T]
    dec_b, dec_t,
    # W_h strides: [N, N]
    wh_n1, wh_n2,
    # S0 strides: [B, N, M]
    s0_b, s0_n, s0_m,
    # S_traj strides: [B, T, N, M]
    st_b, st_t, st_n, st_m,
    # Sflat strides: [B, T, NM]
    sf_b, sf_t, sf_nm,
    B: tl.constexpr, T: tl.constexpr, N: tl.constexpr,
    M: tl.constexpr, M_TILE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    M_TILES = M // M_TILE
    b = pid // M_TILES
    m_tile = pid % M_TILES
    m_offset = m_tile * M_TILE

    n_idx = tl.arange(0, N).to(tl.int64)
    m_idx = tl.arange(0, M_TILE).to(tl.int64)
    m_global_idx = m_offset + m_idx
    nm_tile = n_idx[:, None] * M + m_global_idx[None, :]
    wh_tile = n_idx[:, None] * wh_n1 + n_idx[None, :] * wh_n2

    # Load W_h once
    W_h = tl.load(W_h_ptr + wh_tile).to(tl.float32)

    # Load initial S tile (per batch, this m_tile)
    s0_off = b * s0_b + n_idx[:, None] * s0_n + m_global_idx[None, :] * s0_m
    S = tl.load(S0_ptr + s0_off).to(tl.float32)

    for t in range(T):
        # Load k_t [N], v_tile [M_TILE], decay scalar
        k_off = b * k_b + t * k_t + n_idx * k_n
        k_t_v = tl.load(K_ptr + k_off).to(tl.float32)

        v_off = b * v_b + t * v_t + m_global_idx * v_m
        v_tile = tl.load(V_ptr + v_off).to(tl.float32)

        dec = tl.load(decay_ptr + b * dec_b + t * dec_t).to(tl.float32)

        # retrieved_tile = S^T @ k = sum over N axis of (S * k[:, None])
        retrieved_tile = tl.sum(S * k_t_v[:, None], axis=0)  # [M_TILE]
        delta_tile = v_tile - retrieved_tile                  # [M_TILE]
        # update_tile = k outer delta = [N, M_TILE]
        update_tile = k_t_v[:, None] * delta_tile[None, :]
        # W_h @ S_tile  [N, N] @ [N, M_TILE]
        Wh_S = tl.dot(W_h, S, allow_tf32=False)
        pre = dec * Wh_S + update_tile
        e2x = tl.exp(2.0 * pre)
        S = (e2x - 1.0) / (e2x + 1.0)

        # Store S_traj at this t for this tile
        st_off = b * st_b + t * st_t + n_idx[:, None] * st_n + m_global_idx[None, :] * st_m
        tl.store(S_traj_ptr + st_off, S.to(S_traj_ptr.dtype.element_ty))

        # Store flat S to Sflat for output use
        flat_idx = n_idx[:, None] * M + m_global_idx[None, :]  # flat [N*M] index
        sf_off = b * sf_b + t * sf_t + flat_idx * sf_nm
        tl.store(Sflat_ptr + sf_off, S.to(Sflat_ptr.dtype.element_ty))


def e93_seq_fwd(S0, W_h, K, V, decay, M_TILE=64, num_warps=2):
    """E93 forward Triton.

    Args:
      S0: [B, N, M]
      W_h: [N, N]
      K: [B, T, N]
      V: [B, T, M]
      decay: [B, T]
    Returns:
      S_traj: [B, T, N, M]
      Sflat: [B, T, N*M]   (per-step state flattened — used as output before out_proj)
    """
    B, T, N = K.shape
    M = V.shape[-1]
    assert V.shape == (B, T, M)
    assert decay.shape == (B, T)
    assert S0.shape == (B, N, M)
    assert W_h.shape == (N, N)
    assert M % M_TILE == 0, f"M={M} must be divisible by M_TILE={M_TILE}"

    dtype = S0.dtype
    device = S0.device
    S_traj = torch.empty(B, T, N, M, dtype=dtype, device=device)
    Sflat = torch.empty(B, T, N * M, dtype=dtype, device=device)

    k_b, k_t, k_n = K.stride()
    v_b, v_t, v_m = V.stride()
    dec_b, dec_t = decay.stride()
    wh_n1, wh_n2 = W_h.stride()
    s0_b, s0_n, s0_m = S0.stride()
    st_b, st_t, st_n, st_m = S_traj.stride()
    sf_b, sf_t, sf_nm = Sflat.stride()

    M_TILES = M // M_TILE
    grid = (B * M_TILES,)
    _e93_fwd_kernel[grid](
        S0.contiguous(), K, V, decay, W_h.contiguous(),
        S_traj, Sflat,
        k_b, k_t, k_n,
        v_b, v_t, v_m,
        dec_b, dec_t,
        wh_n1, wh_n2,
        s0_b, s0_n, s0_m,
        st_b, st_t, st_n, st_m,
        sf_b, sf_t, sf_nm,
        B=B, T=T, N=N, M=M, M_TILE=M_TILE,
        num_warps=num_warps,
    )
    return S_traj, Sflat


def e93_seq_fwd_torch_ref(S0, W_h, K, V, decay):
    """Pure PyTorch reference."""
    B, T, N = K.shape
    M = V.shape[-1]
    S = S0.float().clone()
    K = K.float(); V = V.float()
    decay = decay.float(); W_h = W_h.float()
    S_traj = torch.empty(B, T, N, M, dtype=S0.dtype, device=S0.device)
    Sflat = torch.empty(B, T, N * M, dtype=S0.dtype, device=S0.device)

    for t in range(T):
        k_t = K[:, t]    # [B, N]
        v_t = V[:, t]    # [B, M]
        dec = decay[:, t].view(B, 1, 1)
        retrieved = torch.einsum('bnm,bn->bm', S, k_t)
        delta = v_t - retrieved
        update = torch.einsum('bn,bm->bnm', k_t, delta)
        Wh_S = torch.einsum('np,bpm->bnm', W_h, S)
        S = torch.tanh(dec * Wh_S + update)
        S_traj[:, t] = S.to(S0.dtype)
        Sflat[:, t] = S.reshape(B, N * M).to(S0.dtype)
    return S_traj, Sflat


if __name__ == '__main__':
    import os
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '5')
    print("E93 forward correctness: Triton vs PyTorch ref (fp32)\n")
    for B, T, N, M in [(2, 32, 16, 64), (1, 64, 16, 128), (4, 16, 16, 256)]:
        torch.manual_seed(0)
        dt = torch.float32
        S0 = (0.1 * torch.randn(B, N, M, dtype=dt, device='cuda')).contiguous()
        W_h = (torch.eye(N) + 0.05 * torch.randn(N, N)).to('cuda').contiguous()
        K = 0.3 * torch.randn(B, T, N, dtype=dt, device='cuda')
        V = 0.3 * torch.randn(B, T, M, dtype=dt, device='cuda')
        decay = torch.sigmoid(0.5 + 0.1 * torch.randn(B, T, dtype=dt, device='cuda'))

        # L2 norm K (matches model)
        K = torch.nn.functional.normalize(K, dim=-1)

        S_tri, Sf_tri = e93_seq_fwd(S0, W_h, K, V, decay, M_TILE=min(M, 64))
        S_ref, Sf_ref = e93_seq_fwd_torch_ref(S0, W_h, K, V, decay)

        rt = (S_tri - S_ref).abs().max().item() / max(S_ref.abs().max().item(), 1e-10)
        rf = (Sf_tri - Sf_ref).abs().max().item() / max(Sf_ref.abs().max().item(), 1e-10)
        ok = "PASS" if max(rt, rf) < 1e-4 else "FAIL"
        print(f"  B={B} T={T} N={N} M={M}:  S={rt:.2e}  Sf={rf:.2e}  [{ok}]")
