"""ADMM / Parareal prototype for parallel-in-time E88 forward.

Structure:
  1. Split T into P chunks of size T_chunk = T/P
  2. Boundaries: S_boundary[p] for p=0..P (S_boundary[0] = S0, rest initialized)
  3. Outer iter k:
     a. For each chunk p in parallel: run sequential E88 from S_boundary[p]
        for T_chunk steps. Output: S_chunk_end[p].
     b. Update boundaries: S_boundary[p+1] = S_chunk_end[p]
     c. Check convergence of boundaries
  4. After convergence: concat per-chunk trajectories = global trajectory.

Why this might win:
  - All P chunks run sequential E88 IN PARALLEL (on separate GPU resources)
  - With more concurrent chains, GPU HBM utilization improves (we measured
    sequential at only 12-35% of HBM peak — ADMM adds P× more parallelism)
  - E88 has decay<1 → boundary errors contract exponentially → fast convergence

Questions this prototype answers:
  - How many outer iterations to converge for realistic E88 inputs?
  - Speedup vs pure sequential?
  - Does the PyTorch/sequential-loop version even reflect the real potential?
"""

import sys, os, time
import torch

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase0_pytorch_ref import _random_case
from phase4_newton_driver import sequential_e88_forward


def e88_step_batch(S_prev, K_t, V_t, decay_t):
    """One step of E88 recurrence.
    S_prev: [..., N, N]
    K_t, V_t: [..., N]
    decay_t: [...]
    """
    retrieved = torch.einsum('...ij,...j->...i', S_prev, K_t)
    delta = V_t - retrieved
    outer = torch.einsum('...i,...j->...ij', delta, K_t)
    pre = decay_t[..., None, None] * S_prev + outer
    return torch.tanh(pre)


def sequential_chunk(S_start, K_chunk, V_chunk, decay_chunk, time_axis=2):
    """Sequential E88 over one chunk. Returns final state only.

    S_start: [..., N, N]
    K_chunk, V_chunk: [..., T_chunk, N]  (T_chunk along `time_axis`)
    decay_chunk: [..., T_chunk]
    """
    S = S_start
    T_chunk = K_chunk.shape[time_axis]
    for t in range(T_chunk):
        # Select time t along time_axis — use torch.select for generality
        K_t = K_chunk.select(time_axis, t)
        V_t = V_chunk.select(time_axis, t)
        d_t = decay_chunk.select(time_axis, t)
        S = e88_step_batch(S, K_t, V_t, d_t)
    return S


def sequential_chunk_with_traj(S_start, K_chunk, V_chunk, decay_chunk):
    """Same but returns full trajectory [T_chunk+1, ...] including S_start.
    Expects K_chunk: [B, H, T_chunk, N] (standard layout)."""
    B, H, T_chunk, N = K_chunk.shape
    traj = torch.empty(B, H, T_chunk + 1, N, N, dtype=S_start.dtype, device=S_start.device)
    traj[:, :, 0] = S_start
    for t in range(T_chunk):
        traj[:, :, t + 1] = e88_step_batch(traj[:, :, t], K_chunk[:, :, t],
                                             V_chunk[:, :, t], decay_chunk[:, :, t])
    return traj


def admm_forward(S0, K, V, decay, P, max_iters=20, tol=1e-4, verbose=False):
    """Parallel-in-time E88 forward using ADMM/Parareal.

    S0: [B, H, N, N]
    K, V: [B, H, T, N]
    decay: [B, H, T]
    P: number of chunks

    Returns:
      S_traj: [B, H, T+1, N, N]
      iters: number of outer iterations used
    """
    B, H, T, N = K.shape
    assert T % P == 0, f"T={T} must be divisible by P={P}"
    T_chunk = T // P

    # Chunk the inputs. Reshape P into the head dim so sequential_chunk sees
    # P*H independent chains.
    # K: [B, H, T, N] → [B, H, P, T_chunk, N] → [B, H*P, T_chunk, N]
    K_chunked = K.view(B, H, P, T_chunk, N).permute(0, 2, 1, 3, 4).reshape(B, P * H, T_chunk, N)
    V_chunked = V.view(B, H, P, T_chunk, N).permute(0, 2, 1, 3, 4).reshape(B, P * H, T_chunk, N)
    decay_chunked = decay.view(B, H, P, T_chunk).permute(0, 2, 1, 3).reshape(B, P * H, T_chunk)

    # Wait — we actually want per-chunk groups. Let me use a cleaner layout:
    # Flatten (H, P) → HP with P outermost: chunk p of head h lives at HP index p*H + h.
    # But for boundary propagation we need to know which chunk is which.
    # Easier: just keep [B, H, P, ...] and use explicit dim indexing inside sequential.

    # Revert: use [B, H, P, T_chunk, N] and index by axis 3 for time.
    K_chunked = K.view(B, H, P, T_chunk, N)
    V_chunked = V.view(B, H, P, T_chunk, N)
    decay_chunked = decay.view(B, H, P, T_chunk)

    # Initialize boundaries. S_boundary[p] is the state AT the start of chunk p.
    # S_boundary[0] = S0 (exact). Others: initialize from S0 (bad guess).
    S_boundary = S0.unsqueeze(2).expand(B, H, P, N, N).clone()
    # S_boundary shape: [B, H, P, N, N]

    d_max = float('inf')
    for it in range(max_iters):
        # Parallel: for each chunk p, run sequential from S_boundary[p]
        # Batch all P chunks together. Permute so P becomes a "chain" dim
        # that runs alongside B, H. Treat as [B*H*P, T_chunk, N] batched.
        # sequential_chunk can handle arbitrary leading batch dims, so we
        # flatten P into the batch.
        # Actually simpler: reshape so P joins head dim.
        S_start_batched = S_boundary  # [B, H, P, N, N]
        K_batched = K_chunked          # [B, H, P, T_chunk, N]
        V_batched = V_chunked
        decay_batched = decay_chunked

        # Run sequential chunk on the batched chunks — one sequential loop,
        # all P chunks process in parallel via vectorization.
        # Shapes: S_start [B,H,P,N,N], K/V [B,H,P,T_chunk,N], decay [B,H,P,T_chunk]
        # time_axis = 3 (after B,H,P: T_chunk is axis 3 for K/V, but for decay it's also 3)
        S_end_per_chunk = sequential_chunk(
            S_start_batched, K_batched, V_batched, decay_batched,
            time_axis=3
        )  # [B, H, P, N, N]

        # Update: S_boundary[p+1] = S_end[p] for p=0..P-2
        S_boundary_new = S_boundary.clone()
        S_boundary_new[:, :, 1:P] = S_end_per_chunk[:, :, 0:P - 1]

        # Convergence check: boundary changes
        d_max = (S_boundary_new - S_boundary).abs().max().item()
        S_boundary = S_boundary_new

        if verbose:
            print(f"    iter {it+1}: max Δboundary = {d_max:.3e}")
        if d_max < tol:
            break

    # Build full trajectory from converged boundaries
    # For each chunk p, run sequential from S_boundary[p] to get trajectory
    S_traj = torch.empty(B, H, T + 1, N, N, dtype=S0.dtype, device=S0.device)
    for p in range(P):
        chunk_traj = sequential_chunk_with_traj(
            S_boundary[:, :, p],
            K_chunked[:, :, p], V_chunked[:, :, p], decay_chunked[:, :, p]
        )
        if p == 0:
            S_traj[:, :, 0:T_chunk + 1] = chunk_traj
        else:
            # chunk_traj includes S_start. Skip that; next chunk starts where prev ended.
            S_traj[:, :, p * T_chunk + 1:(p + 1) * T_chunk + 1] = chunk_traj[:, :, 1:]

    return S_traj, it + 1


def test_correctness(B, H, T, N, P, seed=0, tol_factor=1e-3):
    S0, K, V, decay = _random_case(B, H, T, N, seed=seed, dtype=torch.float64)

    # Ground truth
    S_seq = sequential_e88_forward(S0, K, V, decay)

    # ADMM
    S_admm, iters = admm_forward(S0, K, V, decay, P, max_iters=30, tol=1e-10, verbose=False)

    diff = (S_admm - S_seq).abs().max().item()
    status = "PASS" if diff < tol_factor else "FAIL"
    print(f"  B={B} H={H:2d} T={T:5d} N={N} P={P:3d}  iters={iters:2d}  "
          f"max|admm-seq|={diff:.2e}  [{status}]")
    return iters, diff


def measure_convergence(B, H, T, N, P_values, seed=0):
    """How many outer iters does each P need?"""
    S0, K, V, decay = _random_case(B, H, T, N, seed=seed, dtype=torch.float64)

    print(f"  B={B} H={H} T={T} N={N} — iters needed to converge (tol=1e-6):")
    for P in P_values:
        if T % P != 0:
            continue
        _, iters = admm_forward(S0, K, V, decay, P, max_iters=50, tol=1e-6, verbose=False)
        print(f"    P={P:3d} (chunk size {T//P:5d}): {iters:2d} iters")


if __name__ == '__main__':
    print("Phase ADMM-1 — correctness\n")
    for B, H, T, N, P in [(1, 2, 64, 16, 4),
                           (1, 4, 128, 16, 8),
                           (1, 2, 256, 16, 16),
                           (1, 2, 1024, 16, 32)]:
        test_correctness(B, H, T, N, P)

    print("\nPhase ADMM-2 — iterations to converge vs P\n")
    for B, H, T, N in [(1, 2, 1024, 16),
                        (1, 2, 4096, 16),
                        (1, 2, 16384, 16)]:
        measure_convergence(B, H, T, N, P_values=[2, 4, 8, 16, 32, 64])
        print()
