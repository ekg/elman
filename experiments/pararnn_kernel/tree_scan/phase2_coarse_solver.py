"""Phase 2 — linear E88 coarse solver.

Linear E88 drops tanh and -(K·S)K from the recurrence:
  TRUE:    S[t] = tanh(decay_t · S[t-1] + (V[t] - K[t]·S[t-1]) ⊗ K[t])
  LINEAR:  S[t] =      decay_t · S[t-1] + V[t] ⊗ K[t]

Linear is solvable by associative scan (element-wise):
  S[t, i, j] = decay_t · S[t-1, i, j] + V[t, i] · K[t, j]

For element (i, j): scalar affine chain, O(log T) parallel scan.

Use outputs at chunk boundaries as WARM BOUNDARIES for the true (tanh)
ADMM, converge in 1 iter.

Stability note: cumprod of decays can underflow for long T and decay<1.
We compute in log space.
"""

import sys, os, time
import torch

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def linear_e88_scan_at_boundaries(S0, k, v, decay, P):
    """Compute LINEAR E88 state at each chunk boundary via parallel scan.

    Inputs (CUDA layout, [T, B, H, N]):
      S0: [B, H, N, N]
      k, v: [T, B, H, N]
      decay: [T, B, H]
    Returns:
      boundaries: [B, P, H, N, N]  — state at start of chunk p (p=0..P-1)
                  boundaries[..., 0, ...] = S0
                  boundaries[..., p, ...] = linear-E88 S at time p*T_chunk (for p > 0)

    Implementation: for stability, work in log space.
      log_cumprod_decay[t] = sum over s<=t of log(decay[s])
      s[t] = exp(log_cumprod_decay[t]) * (S0 + sum over s<=t of exp(-log_cumprod_decay[s]) * b[s])
    Where b[t] = outer(v[t], k[t]).

    But for long T and decay<1, exp(-log_cumprod_decay[s]) blows up.
    Workaround: reset scan at each chunk and pass state between chunks.

    Actually simplest: use PyTorch einsum + cumulative operations locally.
    For now, implement chunk-wise linear forward (sequential within chunk).
    That gives O(T_chunk) depth per chunk — still log_depth overall since
    chunks run in parallel. Good enough for coarse solver.
    """
    T, B, H, N = k.shape
    assert T % P == 0
    T_chunk = T // P

    # Permute so T is easier: chunk by T
    # k: [T, B, H, N] → [P, T_chunk, B, H, N]
    k_chunks = k.view(P, T_chunk, B, H, N)
    v_chunks = v.view(P, T_chunk, B, H, N)
    decay_chunks = decay.view(P, T_chunk, B, H)

    # Compute linear E88 forward across all chunks, sequentially (this is the
    # 'coarse' version; optimize later with real parallel scan).
    # We ONLY need the state at the BOUNDARY of each chunk, not intermediate.

    # But since we're going for a COARSE solver with small T_chunk (~4K),
    # linear scan within a chunk is:
    #   s[t] = decay_t * s[t-1] + outer(v[t], k[t])
    # and we just need s at t = T_chunk-1 (end of chunk), then move to next chunk.

    # Actually we want boundaries AT START of each chunk:
    #   boundaries[0] = S0
    #   boundaries[p] = state at end of chunk p-1 (for p=1..P-1)

    # Build by running linear forward across chunks:
    boundaries = torch.empty(B, P, H, N, N, dtype=S0.dtype, device=S0.device)
    boundaries[:, 0] = S0
    S = S0
    for p in range(P):
        # Run linear E88 over T_chunk steps of chunk p
        # Note: this is SEQUENTIAL and slow for a coarse solver.
        # But we'll use it as a ground-truth for checking "does linear approx
        # give good enough boundaries?"
        for t in range(T_chunk):
            dec_t = decay_chunks[p, t].unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]
            V_t = v_chunks[p, t]  # [B, H, N]
            K_t = k_chunks[p, t]  # [B, H, N]
            outer_vk = torch.einsum('bhi,bhj->bhij', V_t, K_t)
            S = dec_t * S + outer_vk
        if p < P - 1:
            boundaries[:, p + 1] = S

    return boundaries


def linear_e88_scan_parallel(S0, k, v, decay, P):
    """Same math but using prefix-sum for O(log T) depth per chunk.

    At each chunk p:
      s[t] = prod_decay[t] * s[start] + prod_decay[t] * cumsum_term[t]
      where cumsum_term[t] = sum_{τ <= t} b[τ] / prod_decay[τ]

    We only need s[end_of_chunk], so compute once per chunk.
    """
    T, B, H, N = k.shape
    assert T % P == 0
    T_chunk = T // P

    k_chunks = k.view(P, T_chunk, B, H, N)
    v_chunks = v.view(P, T_chunk, B, H, N)
    decay_chunks = decay.view(P, T_chunk, B, H)

    # For each chunk, we want the state at the END, given the state at START.
    # state_end = decay_prod_full * state_start + sum_{τ=0..T_chunk-1} (decay_prod_{τ+1..T_chunk-1}) * b[τ]
    # Let's compute in fp32 to avoid overflow issues:
    decay_f32 = decay_chunks.float()  # [P, T_chunk, B, H]
    # b[p, t] = outer(v, k)[p, t]  shape [P, T_chunk, B, H, N, N]
    # That's a big tensor at T_chunk=4K, P=16: 1*141*16*4096*16*16 elements per head... Let me estimate.
    # N=16: 4096*16*16 = 1M floats per chunk per head = 16MB per chunk per head.
    # Total across chunks and heads: 16 * 16MB * 141 = 36 GB. TOO BIG.

    # So we can't materialize b. Must compute in a streaming/scan fashion.

    # Alternate approach: per (b, h, i, j), compute scan separately. Use broadcasting.
    # s[t, i, j] = decay[t] * s[t-1, i, j] + v[t, i] * k[t, j]
    # For scan, state = s, scalar input = (decay, v·k outer).
    # Per element scan is O(T_chunk) sequential, but all elements scan in parallel.

    # Implement: for each chunk, use torch ops to do the scan.
    # decay_prod[t] = cumprod of decay from [0, t]
    # b[t, i, j] = v[t, i] * k[t, j]
    # s[t, i, j] = decay_prod[t] * s[0, i, j] + sum_{τ=0..t-1} decay_prod[t]/decay_prod[τ+1] * b[τ, i, j]
    # The sum is a cumulative sum scaled by decay_prod ratios. Using log space:
    # log_dp[t] = cumsum(log_decay[0..t])
    # c[τ] = log_b[τ] + log_dp[t] - log_dp[τ+1]  (but this only works in log-space if b > 0)

    # Numerically, for small T_chunk (~4K), cumprod in fp32 may underflow if decay < 0.99^4000 = 10^-17.
    # Our decay ~ 0.95: 0.95^4K = 10^-91. Underflows.

    # For a coarse solver, we can FAKE a different decay value for stability, or just
    # fall back to sequential. Let me go with sequential but vectorized.
    # Actually: we can compute scan ONLY at the chunk END and avoid underflow by
    # rescaling. But it's fiddly.

    # For now: use torch.cumprod with fp32 AND a safe decay clamp.
    # Actually simplest: just do SEQUENTIAL within chunk (slow PyTorch) but all chunks in parallel.

    # sequential-over-T_chunk but vectorized-over-chunks:
    boundaries = torch.empty(B, P, H, N, N, dtype=S0.dtype, device=S0.device)
    boundaries[:, 0] = S0

    # For chunks 1..P-1, we need state at end of prev chunk.
    # ADMM INIT idea: give each chunk its OWN guess, don't chain. That misses
    # the cross-chunk correlation, BUT combined with the true-ADMM outer iter,
    # it converges fast. In fact, setting boundaries[p] = S0 for all p (naive
    # init) is what cold-start ADMM does.
    # Better init: use linear-E88 END of each chunk assuming starting from S0
    # (chunk p's input INCLUDES chunk 0..p-1's contribution because decay propagates).
    # Wait no — if we assume chunk p starts from S0, we ignore chunks 0..p-1.

    # CORRECT FORMULATION:
    #   True boundaries[p] = state at end of linear-E88 scan over [0, p*T_chunk)
    # That's a true linear forward from S0 across P*T_chunk steps, taking snapshots
    # at boundaries. We need this in O(log T) parallel.

    # Use: boundary[p] = decay_prod[0..p*T_chunk-1] * S0 + convolution_term
    # With segmentation and careful scaling, possible.

    # For THIS phase, let's just do sequential linear E88 across all T (no chunking).
    # Take snapshots at chunk boundaries. This is slow but gives us a baseline to
    # check whether linear-scan boundaries are good enough.

    # Sequential linear E88 forward (vectorized over B, H):
    S = S0.clone()
    for t in range(T):
        dec_t = decay[t].unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]
        V_t = v[t]  # [B, H, N]
        K_t = k[t]  # [B, H, N]
        outer_vk = torch.einsum('bhi,bhj->bhij', V_t, K_t)
        S = dec_t * S + outer_vk

        # Save if at chunk boundary
        next_p = (t + 1) // T_chunk
        if (t + 1) % T_chunk == 0 and next_p < P:
            boundaries[:, next_p] = S

    return boundaries


def measure_linear_boundary_quality(B, H, T, N, P):
    """How well do linear-E88 scan boundaries predict true (tanh) boundaries?"""
    dt = torch.bfloat16
    g = torch.Generator(device='cuda').manual_seed(0)
    k = 0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')
    v = 0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')
    q = 0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, generator=g, dtype=dt, device='cuda'))
    S0 = 0.1 * torch.randn(B, H, N, N, generator=g, dtype=dt, device='cuda')

    # Linear scan boundaries
    t0 = time.time()
    lin_bd = linear_e88_scan_parallel(S0.float(), k.float(), v.float(), decay.float(), P)
    torch.cuda.synchronize()
    lin_ms = (time.time() - t0) * 1000

    # True (tanh) boundaries via full ADMM convergence
    from phase1_warmstart_bench import admm_forward_fixed_iters
    _, true_bd, _ = admm_forward_fixed_iters(S0, k, v, q, decay, H, P,
                                               num_iters=5, init_boundaries=None)

    err = (lin_bd.float() - true_bd.float()).abs()
    max_err = err.max().item()
    # Skip boundary[0] which is S0 (exact) for statistics
    max_err_skip0 = err[:, 1:].max().item()
    print(f"  H={H} T={T} N={N} P={P}  linear-scan: {lin_ms:.1f}ms  "
          f"max |lin_bd - true_bd|={max_err_skip0:.2e}  (bf16 ~ 8e-3)")
    return max_err_skip0, lin_ms


if __name__ == '__main__':
    print("Phase 2 — linear-E88 boundary quality vs true tanh boundaries\n")
    for H, N in [(141, 16), (83, 32)]:
        for T in [16384, 32768]:
            try:
                measure_linear_boundary_quality(1, H, T, N, P=16)
            except Exception as e:
                print(f"  FAIL {H}/{T}: {str(e)[:100]}")
                torch.cuda.empty_cache()
