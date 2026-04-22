"""Phase 3b — real Triton kernel for the r=1 structured prefix scan.

Starts minimal: one Triton program per row, [T, N] block tensors,
manual Hillis-Steele with log₂(T) passes, ping-pong via gmem scratch
between passes.

Correctness criterion: matches Phase 3a's pure-PyTorch scan output to
float32 tolerance for the full shape range we've tested before.

Incremental path:
  Step 1 — (this file) non-Newton, single-row, single-pass test:
           kernel takes precomputed (D, u, v, b) for one row, returns
           prefix (b-component) values. Compared directly to the
           Phase 3a PyTorch scan on the same inputs. No Newton loop yet.
  Step 2 — wrap the kernel into a Newton iteration (same loop as
           Phase 3a.newton_scan_r1 but with Triton scan replacing
           the Python list scan).
"""

import sys
import os
import math

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Reuse Phase 3a's already-validated PyTorch combine + scan + Newton,
# so we have a ground-truth oracle to diff against.
from phase3_triton import (
    _make_step, _combine_r1, scan_r1_pytorch,
    _rank2_to_rank1_triton, _combine_r1_triton,
)
from phase1_reference import _random_inputs


# -----------------------------------------------------------------------------
# Minimal Triton scan kernel — step 1: single row, no Newton loop.
# -----------------------------------------------------------------------------

@triton.jit
def _scan_r1_kernel_step1(
    D_ptr, u_ptr, v_ptr, b_ptr,              # inputs, [T, N]
    scratch_D, scratch_u, scratch_v, scratch_b,  # scratch, [T, N]
    out_ptr,                                  # output delta [T, N]
    T: tl.constexpr,
    N: tl.constexpr,
    LOG_T: tl.constexpr,
):
    """Single-program Hillis-Steele prefix scan over T timesteps.

    Invariant: at start of pass s, `state` holds the prefix products
    combined over stride 2^s (i.e., state[t] = combine of positions
    max(0, t - 2^s + 1) .. t). After LOG_T passes, state[t] = full prefix.

    Pass s:
      for each t with t ≥ 2^s: state[t] ← combine(state[t - 2^s], state[t])
    """
    t_idx = tl.arange(0, T)[:, None]         # [T, 1]
    n_idx = tl.arange(0, N)[None, :]         # [1, N]
    offs = t_idx * N + n_idx

    D = tl.load(D_ptr + offs)
    u = tl.load(u_ptr + offs)
    v = tl.load(v_ptr + offs)
    b = tl.load(b_ptr + offs)

    for s in tl.static_range(LOG_T):
        stride = 1 << s

        # Publish current state to scratch
        tl.store(scratch_D + offs, D)
        tl.store(scratch_u + offs, u)
        tl.store(scratch_v + offs, v)
        tl.store(scratch_b + offs, b)
        # Barrier so all writes are visible before the shifted reads.
        tl.debug_barrier()

        # Read shifted state (zero if t < stride, we'll mask it out)
        mask_combine = t_idx >= stride                    # [T, 1]
        shifted_t = tl.where(mask_combine, t_idx - stride, 0)
        shifted_offs = shifted_t * N + n_idx

        D_sh = tl.load(scratch_D + shifted_offs)
        u_sh = tl.load(scratch_u + shifted_offs)
        v_sh = tl.load(scratch_v + shifted_offs)
        b_sh = tl.load(scratch_b + shifted_offs)

        # Combine (shifted = "a", current = "b"):  current ← combine(shifted, current)
        D_new, u_new, v_new, b_new = _combine_r1_triton(
            D_sh, u_sh, v_sh, b_sh, D, u, v, b
        )

        # Apply only where mask_combine
        D = tl.where(mask_combine, D_new, D)
        u = tl.where(mask_combine, u_new, u)
        v = tl.where(mask_combine, v_new, v)
        b = tl.where(mask_combine, b_new, b)

    tl.store(out_ptr + offs, b)


def scan_r1_triton_step1(D, u, v, b):
    """Minimal Triton scan: one row, no Newton. Returns δ = [T, N]."""
    T, N = D.shape
    assert (T & (T - 1)) == 0 and T >= 2, f"T must be power of 2 ≥ 2, got {T}"
    log_T = int(math.log2(T))

    Dc = D.contiguous()
    uc = u.contiguous()
    vc = v.contiguous()
    bc = b.contiguous()

    sD = torch.empty_like(Dc)
    su = torch.empty_like(uc)
    sv = torch.empty_like(vc)
    sb = torch.empty_like(bc)
    out = torch.empty_like(bc)

    _scan_r1_kernel_step1[(1,)](
        Dc, uc, vc, bc,
        sD, su, sv, sb,
        out,
        T=T, N=N, LOG_T=log_T,
    )
    return out


# -----------------------------------------------------------------------------
# Step-1 test: kernel output matches Phase 3a PyTorch scan on same inputs.
# -----------------------------------------------------------------------------

def build_steps(S0_row, K, V_i, decay, S_var, r):
    """Build the per-step (D, u, v, b) tuples used by the scan."""
    T = K.shape[0]
    steps = []
    for t in range(T):
        s_prev = S0_row if t == 0 else S_var[t - 1]
        steps.append(_make_step(s_prev, K[t], V_i[t], decay[t], r[t]))
    return steps


def stack_steps(steps):
    """Pack T list of per-row tuples into [T, N] tensors."""
    D = torch.stack([s[0] for s in steps])
    u = torch.stack([s[1] for s in steps])
    v = torch.stack([s[2] for s in steps])
    b = torch.stack([s[3] for s in steps])
    return D, u, v, b


def test_step1(T, n, seed=0, dtype=torch.float32, device='cuda'):
    """Build random per-step inputs (no Newton loop), compare scans."""
    S0_row, K, V_i, decay = _random_inputs(T, n, seed=seed, dtype=dtype, device=device)
    # Set S_var = 0 (like first Newton iteration) so all (D, u, v, b) are built
    # from (S_0, K, V, decay) without circular dependence.
    S_var = torch.zeros(T, n, dtype=dtype, device=device)
    # Residual for zero S_var: r[t] = S_var[t] − f(prev, x_t) = -f(prev, x_t)
    from phase1_reference import e88_row_step
    r = torch.empty(T, n, dtype=dtype, device=device)
    r[0] = -e88_row_step(S0_row, K[0], V_i[0], decay[0])
    for t in range(1, T):
        r[t] = -e88_row_step(S_var[t - 1], K[t], V_i[t], decay[t])

    steps = build_steps(S0_row, K, V_i, decay, S_var, r)

    # Phase 3a (PyTorch) scan
    prefix = scan_r1_pytorch(steps)
    delta_pt = torch.stack([p[3] for p in prefix])   # [T, N]

    # Triton scan on same inputs
    D, u, v, b = stack_steps(steps)
    delta_tr = scan_r1_triton_step1(D, u, v, b)

    diff = (delta_pt - delta_tr).abs().max().item()
    max_val = delta_pt.abs().max().item()
    status = "PASS" if diff < max(1e-5, 1e-5 * max_val) else "FAIL"
    print(f"  T={T:4d} n={n:3d} seed={seed:2d}  "
          f"max|δ_pt|={max_val:.3e}  "
          f"max|δ_pt − δ_tr|={diff:.3e}  [{status}]")
    return diff


if __name__ == '__main__':
    print("Phase 3b step 1: single-row Triton prefix scan (float32).")
    for T, n in [(2, 4), (4, 4), (8, 4), (16, 8), (32, 8), (64, 16), (128, 32)]:
        test_step1(T, n)
    print()
    print("If all PASS, the Triton kernel correctly implements the r=1")
    print("structured combine + Hillis-Steele prefix scan.")
