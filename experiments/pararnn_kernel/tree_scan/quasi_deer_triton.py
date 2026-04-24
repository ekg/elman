"""Triton kernel for quasi-DEER Newton step (diagonal Jacobian).

Problem:
  For each (B, H, row p, col c), we have a length-T scalar affine prefix scan:
      x[t] = D[t] * x[t-1] + b[t]
  where D, b are of shape [B, H, T, N_row, N_col].

Because combine is SCALAR, tl.associative_scan applies directly. This is
the one Triton primitive that's known to work for scan fusion (per
docs/TREESCAN_5X_FINAL.md).

Kernel design:
  Grid: one program per (B*H) head.
  Each program processes a block of BLOCK_T timesteps × (N_row*N_col) scalar
  chains. For T > BLOCK_T we chain blocks by carrying the running (D_cum, b_cum)
  state across blocks.

  We pack D and b into a 2D block [BLOCK_T, N_row*N_col] and run
  tl.associative_scan along axis=0.

Combine (scalar):
  (D1, b1) . (D2, b2) = (D1*D2, b1*D2 + b2)    (element-wise)
  Semantics: earlier . later (earlier state goes first).
  Triton associative_scan: if combine_fn(left, right) represents
    "apply right AFTER left", then for inclusive prefix from left:
      result = combine(combine(left, ...), right)
    so combine_fn(l, r) should return r_apply_then_after_l.

  Applying op_l first, then op_r:  x -> D_r * (D_l * x + b_l) + b_r
                                     = (D_r * D_l) * x + (D_r * b_l + b_r)
  So combine(left=(D_l, b_l), right=(D_r, b_r)) = (D_l*D_r, D_r*b_l + b_r).
"""

import sys
import os
import time

import torch
import triton
import triton.language as tl

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Combine function for scalar associative scan
# ---------------------------------------------------------------------------

@triton.jit
def _qd_combine(D_l, b_l, D_r, b_r):
    """Compose: apply left first, then right.
    (D_l, b_l) then (D_r, b_r):
        x -> D_r * (D_l * x + b_l) + b_r
    """
    return D_l * D_r, D_r * b_l + b_r


# ---------------------------------------------------------------------------
# Kernel: one program per (B, H) pair, scan over T with chained blocks.
# ---------------------------------------------------------------------------

@triton.jit
def _qd_scan_kernel(
    D_ptr,          # [B, H, T, N, N]  diagonal Jacobian
    b_ptr,          # [B, H, T, N, N]  -r  (residual * -1)
    out_ptr,        # [B, H, T, N, N]  delta output
    T: tl.constexpr, N: tl.constexpr,
    BLOCK_T: tl.constexpr,
    NN: tl.constexpr,     # N*N
):
    """One program scans one (B, H) head's full T axis.

    We process in BLOCK_T chunks, running tl.associative_scan on each chunk
    then carrying the running prefix-scan state (D_acc, b_acc) into the next
    chunk.
    """
    bh = tl.program_id(0).to(tl.int64)

    head_stride_T = T * NN    # bytes: B*H*T*N*N -> per (b,h): T*N*N
    # Pointers into this head's slice
    D_head = D_ptr + bh * head_stride_T
    b_head = b_ptr + bh * head_stride_T
    out_head = out_ptr + bh * head_stride_T

    # NN-vector of column offsets within the (N, N) slab
    nn_off = tl.arange(0, NN).to(tl.int64)

    # Running prefix state (applies to x just before current chunk)
    D_acc = tl.zeros([NN], dtype=tl.float32) + 1.0
    b_acc = tl.zeros([NN], dtype=tl.float32)

    t0 = 0
    while t0 < T:
        # Load a [BLOCK_T, NN] block of D and b
        t_off = t0 + tl.arange(0, BLOCK_T).to(tl.int64)
        mask_t = t_off < T

        # 2D offsets: [BLOCK_T, NN]
        ptr_off = t_off[:, None] * NN + nn_off[None, :]

        D_block = tl.load(D_head + ptr_off, mask=mask_t[:, None], other=1.0).to(tl.float32)
        b_block = tl.load(b_head + ptr_off, mask=mask_t[:, None], other=0.0).to(tl.float32)

        # Associative scan along axis=0
        D_cum, b_cum = tl.associative_scan(
            (D_block, b_block), axis=0, combine_fn=_qd_combine,
        )

        # Apply the carry from prior blocks:
        #   after scan, D_cum[t] = prod(D[t0..t]), b_cum[t] = b_cum starting x_{-1}=0
        # Full result for x_{t0-1} = carry_b:
        #   x[t0+k] = D_cum[k] * carry_b_at_start + b_cum[k]     (???)
        # Let's re-derive. Let carry = (D_acc, b_acc) meaning if the initial
        # state BEFORE the block were x_prev = P, then x_AT_end_of_prefix =
        # D_acc * P + b_acc. But actually x_prev is ALWAYS 0 at the very start
        # — we're computing the absolute x sequence. So x_{t0-1} =
        # b_acc (with x_{-1} = 0) = b_acc. Composed into the block:
        #   For position k in block, x[t0+k] = D_cum[k] * x_{t0-1} + b_cum[k]
        #                                    = D_cum[k] * b_acc + b_cum[k]
        # Great. So:
        out_block = D_cum * b_acc[None, :] + b_cum

        # Store
        tl.store(out_head + ptr_off, out_block.to(out_ptr.dtype.element_ty),
                 mask=mask_t[:, None])

        # Update carry for next block: new b_acc = last b_cum after carry,
        # new D_acc = D_acc * product_of_D_in_block = D_cum[last] ... but
        # actually we don't need D_acc for anything since we always start
        # with x_{-1} = 0 at position 0. The carry is simply the last
        # value of x. So b_acc_new = out_block[last_valid].
        # We need the last VALID entry — if last block is partially filled.
        # Easiest: compute last index = min(BLOCK_T, T - t0) - 1
        last_valid_k = tl.minimum(BLOCK_T, T - t0) - 1
        # Gather: sum over mask (k == last_valid_k)
        idx_range = tl.arange(0, BLOCK_T).to(tl.int64)
        pick = (idx_range == last_valid_k)[:, None].to(tl.float32)
        b_acc = tl.sum(out_block * pick, axis=0)
        # D_acc isn't needed (x_{-1}=0 at block 0; from then on we use b_acc directly).
        # But keep D_acc = 1.0 for symmetry.
        t0 += BLOCK_T


def qd_diagonal_scan_triton(D, b, block_T=256):
    """Run quasi-DEER diagonal scan in Triton.

    Args:
      D, b: [B, H, T, N, N]  — fp32 or bf16
      block_T: tile size along T for associative_scan. Must be a power of 2.

    Returns:
      delta: [B, H, T, N, N]
    """
    B, H, T, N, N2 = D.shape
    assert N == N2
    dtype = D.dtype
    device = D.device

    NN = N * N
    # Shared memory budget: Triton allocates BLOCK_T * NN elements × 2 tensors
    # × 4 bytes + overhead. Cap BLOCK_T × NN ≤ ~8K elements for safety on 100KB
    # shared memory limit.
    max_block_elems = 8192
    max_block_T = max(16, max_block_elems // NN)
    effective_block_T = min(block_T, max_block_T)

    BLOCK_T = 1
    while BLOCK_T * 2 <= min(effective_block_T, max(T, 2)):
        BLOCK_T *= 2
    if T < BLOCK_T:
        BLOCK_T = 1
        while BLOCK_T * 2 <= T:
            BLOCK_T *= 2
        BLOCK_T = max(BLOCK_T, 16)

    out = torch.empty_like(D)

    grid = (B * H,)
    _qd_scan_kernel[grid](
        D.contiguous(), b.contiguous(), out,
        T=T, N=N,
        BLOCK_T=BLOCK_T, NN=NN,
        num_warps=4, num_stages=2,
    )
    return out


# ---------------------------------------------------------------------------
# Full quasi-DEER Newton driver using Triton scan
# ---------------------------------------------------------------------------

from quasi_deer_ref import build_diag_ingredients, diag_scan_sequential


def quasi_deer_newton_triton(
    S0, K, V, decay,
    *,
    max_iters=30,
    tol=1e-4,
    block_T=256,
    verbose=False,
):
    B, H, T, N = K.shape
    dtype = S0.dtype
    device = S0.device
    S_var = torch.zeros(B, H, T, N, N, dtype=dtype, device=device)

    history = []
    for it in range(max_iters):
        D, b_vec, r_norm = build_diag_ingredients(S0, S_var, K, V, decay)
        history.append(r_norm)
        if verbose:
            print(f"  iter {it}: max|r| = {r_norm:.3e}")
        if r_norm < tol:
            break
        delta = qd_diagonal_scan_triton(D, b_vec, block_T=block_T)
        S_var = S_var + delta

    S = torch.empty(B, H, T + 1, N, N, dtype=dtype, device=device)
    S[:, :, 0] = S0
    S[:, :, 1:] = S_var
    return S, it + 1, r_norm, history


# ---------------------------------------------------------------------------
# Correctness tests: Triton scan vs PyTorch reference
# ---------------------------------------------------------------------------

def _test_scan_correctness():
    """Check Triton scan matches PyTorch diag scan on random inputs."""
    print("Test: Triton diagonal scan vs PyTorch sequential (random D, b)")
    for B, H, T, N, block_T in [
        (1, 2, 64, 16, 64),
        (1, 4, 128, 16, 128),
        (1, 4, 512, 16, 256),
        (1, 16, 1024, 16, 256),
        (1, 16, 1024, 32, 256),
        (1, 8, 4096, 16, 512),
        (1, 8, 4096, 32, 512),
        (1, 4, 16384, 16, 512),
    ]:
        dtype = torch.float32
        g = torch.Generator(device='cuda').manual_seed(0)
        # D typical range: small positive values (tanh_deriv * decay)
        D = 0.3 + 0.2 * torch.randn(B, H, T, N, N, generator=g, dtype=dtype, device='cuda')
        bv = 0.1 * torch.randn(B, H, T, N, N, generator=g, dtype=dtype, device='cuda')

        s_ref = diag_scan_sequential(D, bv)
        s_tri = qd_diagonal_scan_triton(D, bv, block_T=block_T)
        diff = (s_ref - s_tri).abs().max().item()
        rel = diff / max(s_ref.abs().max().item(), 1e-10)
        status = "PASS" if rel < 1e-4 else "FAIL"
        print(f"  B={B} H={H:2d} T={T:5d} N={N:2d}  blockT={block_T:4d}  "
              f"max|ref-tri|={diff:.2e}  rel={rel:.2e}  [{status}]")


def _test_full_newton_correctness():
    """Compare Triton quasi-DEER Newton to Python ref (both use same math)."""
    from quasi_deer_ref import _random_case, quasi_deer_newton

    print("\nTest: Triton quasi-DEER Newton convergence (vs sequential forward)")
    from phase4_newton_driver import sequential_e88_forward
    for B, H, T, N in [(1, 4, 128, 16), (1, 16, 512, 16), (1, 16, 1024, 16),
                       (1, 16, 4096, 16), (1, 4, 1024, 32)]:
        S0, K, V, decay = _random_case(B, H, T, N, seed=0, dtype=torch.float32,
                                        l2_normalize_k=True, v_scale=0.3)
        S_seq = sequential_e88_forward(S0, K, V, decay)

        S_tri, it_tri, res_tri, _ = quasi_deer_newton_triton(
            S0, K, V, decay, max_iters=40, tol=1e-4,
        )
        diff = (S_tri - S_seq).abs().max().item()
        rel = diff / max(S_seq.abs().max().item(), 1e-10)
        status = "PASS" if rel < 1e-3 else "FAIL"
        print(f"  B={B} H={H:3d} T={T:5d} N={N:2d}  iters={it_tri:3d}  "
              f"res={res_tri:.2e}  max|tri-seq|={diff:.2e}  rel={rel:.2e}  [{status}]")


if __name__ == '__main__':
    _test_scan_correctness()
    _test_full_newton_correctness()
