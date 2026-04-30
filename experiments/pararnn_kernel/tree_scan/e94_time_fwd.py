"""E94 time-recurrence forward Triton kernel.

E94 has a state S ∈ ℝ^(B, T, N, M) per layer where M = H · head_dim.
Each "head" h owns the contiguous M-slice [h·head_dim : (h+1)·head_dim],
i.e., a 16×16 sub-state.

Per-head W_h_time[h] ∈ ℝ^(N, N) mixes the N rows of head h's state.

Time recurrence (sequential over t, parallel over (B, H)):
    For t = 0..T-1:
        write_h = (delta-rule write at l=0) OR (prev-layer mix at l>0) — passed as input
        S_h = tanh( W_h_time[h] · S_h^{t-1} + write_h )

Grid: (B · H,)  — each program handles one (b, h) pair, does sequential time loop.

For l=0 (delta-rule input):
    write_h = k_h ⊗ (v_h - retrieved_h)         where retrieved_h = S_h^T · k_h

For l>0 (prev-layer state input):
    write_h is pre-computed by Python: write_h = einsum('np, bhpc -> bhnc', W_h_layer, prev_state)
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _e94_time_fwd_kernel(
    S0_ptr,            # [B, H, N, hd] - initial state (zeros for first layer fwd)
    Wh_time_ptr,       # [H, N, N] - per-head time matrices
    K_ptr,             # [B, T, H, N] - keys (only used if USE_DELTA)
    V_ptr,             # [B, T, H, hd] - values (used as write at l=0 if USE_DELTA, else as direct write)
    Write_ptr,         # [B, T, H, N, hd] - precomputed writes (used if not USE_DELTA, e.g. for l>0)
    S_traj_ptr,        # [B, T, H, N, hd] - output trajectory
    # strides (in elements)
    s0_b, s0_h, s0_n, s0_c,
    wh_h, wh_n1, wh_n2,
    k_b, k_t, k_h, k_n,
    v_b, v_t, v_h, v_c,
    w_b, w_t, w_h_, w_n, w_c,
    st_b, st_t, st_h, st_n, st_c,
    B: tl.constexpr, T: tl.constexpr, H: tl.constexpr,
    N: tl.constexpr, HD: tl.constexpr,
    USE_DELTA: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    b = pid // H
    h = pid % H

    n_idx = tl.arange(0, N).to(tl.int64)
    c_idx = tl.arange(0, HD).to(tl.int64)

    # Load W_h_time for this head
    wh_tile = h * wh_h + n_idx[:, None] * wh_n1 + n_idx[None, :] * wh_n2
    W_h = tl.load(Wh_time_ptr + wh_tile).to(tl.float32)

    # Initial state for this (b, h)
    s0_off = b * s0_b + h * s0_h + n_idx[:, None] * s0_n + c_idx[None, :] * s0_c
    S = tl.load(S0_ptr + s0_off).to(tl.float32)

    for t in range(T):
        if USE_DELTA:
            # Delta-rule write: k ⊗ (v - retrieved)
            k_off = b * k_b + t * k_t + h * k_h + n_idx * k_n
            k_vec = tl.load(K_ptr + k_off).to(tl.float32)

            v_off = b * v_b + t * v_t + h * v_h + c_idx * v_c
            v_vec = tl.load(V_ptr + v_off).to(tl.float32)

            retrieved = tl.sum(S * k_vec[:, None], axis=0)   # [HD]
            delta = v_vec - retrieved
            write = k_vec[:, None] * delta[None, :]           # [N, HD]
        else:
            # Pre-computed write
            w_off = b * w_b + t * w_t + h * w_h_ + n_idx[:, None] * w_n + c_idx[None, :] * w_c
            write = tl.load(Write_ptr + w_off).to(tl.float32)

        # Time mix: W_h · S
        wh_S = tl.dot(W_h, S, allow_tf32=False)              # [N, HD]

        # Combine + tanh
        pre = wh_S + write
        e2x = tl.exp(2.0 * pre)
        S = (e2x - 1.0) / (e2x + 1.0)

        # Store at this (b, t, h)
        st_off = b * st_b + t * st_t + h * st_h + n_idx[:, None] * st_n + c_idx[None, :] * st_c
        tl.store(S_traj_ptr + st_off, S.to(S_traj_ptr.dtype.element_ty))


def e94_time_fwd(S0, W_h_time, K=None, V=None, Write=None, use_delta=True, num_warps=2):
    """E94 per-layer time recurrence forward.

    Args:
        S0: [B, H, N, HD]    — initial state (typically zeros)
        W_h_time: [H, N, N]   — per-head time mixing matrices
        K: [B, T, H, N]      — keys (required if use_delta=True; L2-normalized by caller)
        V: [B, T, H, HD]     — values (required if use_delta=True)
        Write: [B, T, H, N, HD] — precomputed writes (required if use_delta=False)

    Returns:
        S_traj: [B, T, H, N, HD] — full state trajectory at this layer
    """
    if use_delta:
        assert K is not None and V is not None
        B, T, H, N = K.shape
        HD = V.shape[-1]
    else:
        assert Write is not None
        B, T, H, N, HD = Write.shape

    assert S0.shape == (B, H, N, HD)
    assert W_h_time.shape == (H, N, N)

    dtype = S0.dtype
    device = S0.device
    S_traj = torch.empty(B, T, H, N, HD, dtype=dtype, device=device)

    # Strides (in elements)
    s0_b, s0_h, s0_n, s0_c = S0.stride()
    wh_h_, wh_n1, wh_n2 = W_h_time.stride()
    if use_delta:
        k_b, k_t, k_h, k_n = K.stride()
        v_b, v_t, v_h, v_c = V.stride()
        w_b = w_t = w_h__ = w_n = w_c = 0
        K_arg = K
        V_arg = V
        Write_arg = K  # placeholder, won't be loaded
    else:
        k_b = k_t = k_h = k_n = 0
        v_b = v_t = v_h = v_c = 0
        w_b, w_t, w_h__, w_n, w_c = Write.stride()
        K_arg = Write  # placeholder
        V_arg = Write
        Write_arg = Write
    st_b, st_t, st_h, st_n, st_c = S_traj.stride()

    grid = (B * H,)
    _e94_time_fwd_kernel[grid](
        S0.contiguous(), W_h_time.contiguous(),
        K_arg, V_arg, Write_arg,
        S_traj,
        s0_b, s0_h, s0_n, s0_c,
        wh_h_, wh_n1, wh_n2,
        k_b, k_t, k_h, k_n,
        v_b, v_t, v_h, v_c,
        w_b, w_t, w_h__, w_n, w_c,
        st_b, st_t, st_h, st_n, st_c,
        B=B, T=T, H=H, N=N, HD=HD,
        USE_DELTA=use_delta,
        num_warps=num_warps,
    )
    return S_traj


def e94_time_fwd_torch_ref(S0, W_h_time, K=None, V=None, Write=None, use_delta=True):
    """Pure PyTorch reference for parity testing."""
    if use_delta:
        B, T, H, N = K.shape
        HD = V.shape[-1]
    else:
        B, T, H, N, HD = Write.shape
    S = S0.float().clone()
    S_traj = torch.empty(B, T, H, N, HD, dtype=S0.dtype, device=S0.device)
    for t in range(T):
        if use_delta:
            k_t = K[:, t].float()                                   # [B, H, N]
            v_t = V[:, t].float()                                   # [B, H, HD]
            retrieved = torch.einsum('bhnc, bhn -> bhc', S, k_t)    # [B, H, HD]
            delta = v_t - retrieved
            write = torch.einsum('bhn, bhc -> bhnc', k_t, delta)    # [B, H, N, HD]
        else:
            write = Write[:, t].float()
        wh_S = torch.einsum('hnp, bhpc -> bhnc', W_h_time.float(), S)
        S = torch.tanh(wh_S + write)
        S_traj[:, t] = S.to(S0.dtype)
    return S_traj


if __name__ == '__main__':
    import os
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '6')

    print("E94 time-recurrence forward parity tests\n")

    for B, T, H, N, HD in [(2, 32, 16, 16, 16), (1, 64, 32, 16, 16), (2, 16, 8, 16, 16)]:
        torch.manual_seed(0)
        dt = torch.float32
        S0 = (0.05 * torch.randn(B, H, N, HD, dtype=dt, device='cuda')).contiguous()
        W_h = (torch.eye(N, device='cuda').unsqueeze(0).expand(H, -1, -1).contiguous() +
               0.05 * torch.randn(H, N, N, dtype=dt, device='cuda')).contiguous()

        # Test with delta-rule (l=0)
        K_raw = 0.3 * torch.randn(B, T, H, N, dtype=dt, device='cuda')
        K = torch.nn.functional.normalize(K_raw, dim=-1)
        V = 0.3 * torch.randn(B, T, H, HD, dtype=dt, device='cuda')

        S_tri = e94_time_fwd(S0, W_h, K=K, V=V, use_delta=True)
        S_ref = e94_time_fwd_torch_ref(S0, W_h, K=K, V=V, use_delta=True)

        rel = (S_tri - S_ref).abs().max().item() / max(S_ref.abs().max().item(), 1e-10)
        ok = "PASS" if rel < 1e-4 else "FAIL"
        print(f"  delta B={B} T={T} H={H}: max_rel={rel:.2e} [{ok}]")

        # Test without delta-rule (l>0)
        Write = 0.05 * torch.randn(B, T, H, N, HD, dtype=dt, device='cuda')
        S_tri = e94_time_fwd(S0, W_h, Write=Write, use_delta=False)
        S_ref = e94_time_fwd_torch_ref(S0, W_h, Write=Write, use_delta=False)
        rel = (S_tri - S_ref).abs().max().item() / max(S_ref.abs().max().item(), 1e-10)
        ok = "PASS" if rel < 1e-4 else "FAIL"
        print(f"  write B={B} T={T} H={H}: max_rel={rel:.2e} [{ok}]")
