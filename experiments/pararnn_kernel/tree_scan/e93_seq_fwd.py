"""E93 forward Triton kernel — single rectangular state, no heads.

Now parametrized with ablation flags (USE_W_H, USE_DECAY, USE_DELTA, NL_KIND)
so all ablation cells share the same Triton fast path. Compile-time flags
mean the compiler specializes each variant — no runtime branching cost.

Per (b, m_tile), per t:
  retrieved_tile = S_tile^T @ k_t             [M_TILE]   (only if USE_DELTA)
  delta_tile = v_tile - retrieved_tile        [M_TILE]   (else delta = v_tile)
  update_tile = k_t outer delta_tile          [N, M_TILE]
  Wh_S_tile = W_h @ S_tile  (if USE_W_H else S_tile)
  pre = (dec if USE_DECAY else 1) * Wh_S_tile + update_tile
  S_tile = NL(pre)   where NL is tanh (NL_KIND=0) or linear (NL_KIND=1)

NL_KIND: 0=tanh, 1=linear
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _e93_fwd_kernel(
    S0_ptr, K_ptr, V_ptr, decay_ptr, W_h_ptr,
    S_traj_ptr,  # [B, T, N, M]
    Sflat_ptr,   # [B, T, N*M]
    k_b, k_t, k_n,
    v_b, v_t, v_m,
    dec_b, dec_t,
    wh_n1, wh_n2,
    s0_b, s0_n, s0_m,
    st_b, st_t, st_n, st_m,
    sf_b, sf_t, sf_nm,
    B: tl.constexpr, T: tl.constexpr, N: tl.constexpr,
    M: tl.constexpr, M_TILE: tl.constexpr,
    USE_W_H: tl.constexpr, USE_DECAY: tl.constexpr,
    USE_DELTA: tl.constexpr, NL_KIND: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    M_TILES = M // M_TILE
    b = pid // M_TILES
    m_tile = pid % M_TILES
    m_offset = m_tile * M_TILE

    n_idx = tl.arange(0, N).to(tl.int64)
    m_idx = tl.arange(0, M_TILE).to(tl.int64)
    m_global_idx = m_offset + m_idx

    if USE_W_H:
        wh_tile = n_idx[:, None] * wh_n1 + n_idx[None, :] * wh_n2
        W_h = tl.load(W_h_ptr + wh_tile).to(tl.float32)

    s0_off = b * s0_b + n_idx[:, None] * s0_n + m_global_idx[None, :] * s0_m
    S = tl.load(S0_ptr + s0_off).to(tl.float32)

    for t in range(T):
        k_off = b * k_b + t * k_t + n_idx * k_n
        k_t_v = tl.load(K_ptr + k_off).to(tl.float32)

        v_off = b * v_b + t * v_t + m_global_idx * v_m
        v_tile = tl.load(V_ptr + v_off).to(tl.float32)

        if USE_DECAY:
            dec = tl.load(decay_ptr + b * dec_b + t * dec_t).to(tl.float32)
        else:
            dec = 1.0

        if USE_DELTA:
            retrieved_tile = tl.sum(S * k_t_v[:, None], axis=0)
            delta_tile = v_tile - retrieved_tile
        else:
            delta_tile = v_tile

        update_tile = k_t_v[:, None] * delta_tile[None, :]

        if USE_W_H:
            Wh_S = tl.dot(W_h, S, allow_tf32=False)
        else:
            Wh_S = S

        if USE_DECAY:
            pre = dec * Wh_S + update_tile
        else:
            pre = Wh_S + update_tile

        if NL_KIND == 0:  # tanh
            e2x = tl.exp(2.0 * pre)
            S = (e2x - 1.0) / (e2x + 1.0)
        else:  # linear
            S = pre

        st_off = b * st_b + t * st_t + n_idx[:, None] * st_n + m_global_idx[None, :] * st_m
        tl.store(S_traj_ptr + st_off, S.to(S_traj_ptr.dtype.element_ty))

        flat_idx = n_idx[:, None] * M + m_global_idx[None, :]
        sf_off = b * sf_b + t * sf_t + flat_idx * sf_nm
        tl.store(Sflat_ptr + sf_off, S.to(Sflat_ptr.dtype.element_ty))


def e93_seq_fwd(S0, W_h, K, V, decay, M_TILE=64, num_warps=2,
                use_w_h=True, use_decay=True, use_delta=True, nl_kind=0):
    """E93 forward Triton.

    Args:
      S0: [B, N, M]
      W_h: [N, N]   (used only if use_w_h=True; pass eye(N) otherwise)
      K: [B, T, N]
      V: [B, T, M]
      decay: [B, T] (used only if use_decay=True; pass ones otherwise)
      use_w_h, use_decay, use_delta: bool flags
      nl_kind: 0=tanh, 1=linear
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
        USE_W_H=use_w_h, USE_DECAY=use_decay,
        USE_DELTA=use_delta, NL_KIND=nl_kind,
        num_warps=num_warps,
    )
    return S_traj, Sflat


def e93_seq_fwd_torch_ref(S0, W_h, K, V, decay,
                          use_w_h=True, use_decay=True, use_delta=True, nl_kind=0):
    """Pure PyTorch reference."""
    B, T, N = K.shape
    M = V.shape[-1]
    S = S0.float().clone()
    K = K.float(); V = V.float()
    decay = decay.float(); W_h = W_h.float()
    S_traj = torch.empty(B, T, N, M, dtype=S0.dtype, device=S0.device)
    Sflat = torch.empty(B, T, N * M, dtype=S0.dtype, device=S0.device)

    for t in range(T):
        k_t = K[:, t]
        v_t = V[:, t]
        if use_decay:
            dec = decay[:, t].view(B, 1, 1)
        else:
            dec = 1.0
        if use_delta:
            retrieved = torch.einsum('bnm,bn->bm', S, k_t)
            delta = v_t - retrieved
        else:
            delta = v_t
        update = torch.einsum('bn,bm->bnm', k_t, delta)
        if use_w_h:
            Wh_S = torch.einsum('np,bpm->bnm', W_h, S)
        else:
            Wh_S = S
        pre = dec * Wh_S + update
        if nl_kind == 0:
            S = torch.tanh(pre)
        else:
            S = pre
        S_traj[:, t] = S.to(S0.dtype)
        Sflat[:, t] = S.reshape(B, N * M).to(S0.dtype)
    return S_traj, Sflat


if __name__ == '__main__':
    import os
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '5')
    print("E93 forward correctness across ablation flags:\n")
    configs = [
        ('vanilla',  dict(use_w_h=True,  use_decay=True,  use_delta=True,  nl_kind=0)),
        ('no_wh',    dict(use_w_h=False, use_decay=True,  use_delta=True,  nl_kind=0)),
        ('no_delta', dict(use_w_h=True,  use_decay=True,  use_delta=False, nl_kind=0)),
        ('no_decay', dict(use_w_h=True,  use_decay=False, use_delta=True,  nl_kind=0)),
        ('linear',   dict(use_w_h=True,  use_decay=True,  use_delta=True,  nl_kind=1)),
        ('min_tanh', dict(use_w_h=False, use_decay=False, use_delta=False, nl_kind=0)),
        ('min_lin',  dict(use_w_h=False, use_decay=False, use_delta=False, nl_kind=1)),
    ]
    for name, flags in configs:
        for B, T, N, M in [(2, 32, 16, 64), (1, 64, 16, 128)]:
            torch.manual_seed(0)
            dt = torch.float32
            S0 = (0.1 * torch.randn(B, N, M, dtype=dt, device='cuda')).contiguous()
            W_h = (torch.eye(N) + 0.05 * torch.randn(N, N)).to('cuda').contiguous()
            K = 0.3 * torch.randn(B, T, N, dtype=dt, device='cuda')
            V = 0.3 * torch.randn(B, T, M, dtype=dt, device='cuda')
            decay = torch.sigmoid(0.5 + 0.1 * torch.randn(B, T, dtype=dt, device='cuda'))
            K = torch.nn.functional.normalize(K, dim=-1)

            S_tri, Sf_tri = e93_seq_fwd(S0, W_h, K, V, decay, M_TILE=min(M, 64), **flags)
            S_ref, Sf_ref = e93_seq_fwd_torch_ref(S0, W_h, K, V, decay, **flags)

            rt = (S_tri - S_ref).abs().max().item() / max(S_ref.abs().max().item(), 1e-10)
            rf = (Sf_tri - Sf_ref).abs().max().item() / max(Sf_ref.abs().max().item(), 1e-10)
            ok = "PASS" if max(rt, rf) < 1e-4 else "FAIL"
            print(f"  {name:10s} B={B} T={T} N={N} M={M}:  S={rt:.2e}  Sf={rf:.2e}  [{ok}]")
